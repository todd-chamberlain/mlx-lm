# Copyright © 2026 Apple Inc.

import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import deepseek_v32, glm_moe_dsa
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import BatchKVCache, CacheList, KVCache
from mlx_lm.models.mla import latent_length, max_absorbed_queries

TOPK = 8

GLM_ARGS = dict(
    model_type="glm_moe_dsa",
    vocab_size=128,
    hidden_size=64,
    index_head_dim=16,
    index_n_heads=4,
    index_topk=TOPK,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    routed_scaling_factor=2.5,
    kv_lora_rank=16,
    q_lora_rank=32,
    qk_rope_head_dim=8,
    v_head_dim=12,
    qk_nope_head_dim=8,
    topk_method="noaux_tc",
    scoring_func="sigmoid",
    norm_topk_prob=True,
    n_group=1,
    topk_group=1,
    num_experts_per_tok=2,
    moe_layer_freq=1,
    first_k_dense_replace=1,
    max_position_embeddings=1024,
    rms_norm_eps=1e-5,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    attention_bias=False,
    indexer_types=["full", "shared", "shared", "full", "shared", "shared"],
)

DSV32_ARGS = dict(
    model_type="deepseek_v32",
    vocab_size=128,
    hidden_size=64,
    index_head_dim=16,
    index_n_heads=4,
    index_topk=TOPK,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=4,
    routed_scaling_factor=2.5,
    kv_lora_rank=16,
    q_lora_rank=32,
    qk_rope_head_dim=8,
    v_head_dim=12,
    qk_nope_head_dim=8,
    num_experts_per_tok=2,
    first_k_dense_replace=1,
    max_position_embeddings=1024,
)

MODELS = [(glm_moe_dsa, GLM_ARGS), (deepseek_v32, DSV32_ARGS)]


def dense_masked_attention(self, q_nope, q_pe, kv_latent, k_pe, topk_indices, mask):
    """The L > 1 top-k path before sparse prefill: dense SDPA, top-k as a mask."""
    shape = list(topk_indices.shape)
    shape[-1] = kv_latent.shape[2]
    sparse_mask = mx.zeros(shape, dtype=mx.bool_)
    sparse_mask = mx.put_along_axis(sparse_mask, topk_indices, mx.array(True), axis=-1)
    if mask is not None:
        sparse_mask = sparse_mask & mask
    pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
    pe_scores = mx.where(
        sparse_mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
    )
    L = q_nope.shape[2]
    absorbed = L <= max_absorbed_queries(*self._absorbed_dims, latent_length(kv_latent))
    if absorbed:
        q_nope = self.embed_q(q_nope)
        k = v = kv_latent
    else:
        k = self.embed_q(kv_latent, transpose=False)
        v = self.unembed_out(kv_latent)
    out = mx.fast.scaled_dot_product_attention(
        q_nope, k, v, scale=self.scale, mask=pe_scores
    )
    return self.unembed_out(out) if absorbed else out


def unblocked_indexer(self, x, qr, mask, cache=None):
    """The indexer before query blocks."""
    b, s, _ = x.shape
    q = self.wq_b(qr)
    q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
    k = self.wk(x)
    k = self.k_norm(k)
    k = mx.reshape(k, (b, 1, s, self.head_dim))
    offset = cache.offset if cache is not None else 0
    q = self.rope(q, offset=offset)
    k = self.rope(k, offset=offset)
    if cache is not None:
        k, _ = cache.update_and_fetch(k, k)
        cache.values = mx.zeros_like(cache.keys)
    if k.shape[2] <= self.index_topk:
        return None
    scores = q @ k.swapaxes(-1, -2)
    scores = mx.maximum(scores, 0)
    weights = self.weights_proj(x) * (self.n_heads**-0.5 * self.softmax_scale)
    weights = weights.swapaxes(-1, -2)[..., None]
    scores = scores * weights
    if s > 1:
        scores = sum(scores[:, h : h + 1] for h in range(scores.shape[1]))
    else:
        scores = scores.sum(axis=1, keepdims=True)
    if mask is not None:
        scores = mx.where(mask, scores, -float("inf"))
    return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
        ..., -self.index_topk :
    ]


@contextmanager
def dense_reference():
    """Run the model as it was before sparse prefill and query blocks."""
    with (
        mock.patch.object(
            deepseek_v32.DeepseekV32Attention,
            "_sparse_attention",
            dense_masked_attention,
        ),
        mock.patch.object(deepseek_v32.Indexer, "__call__", unblocked_indexer),
    ):
        yield


def block_bytes(n):
    return mock.patch.object(deepseek_v32, "DSA_BLOCK_BYTES", n)


def make_model(module, args, seed=0):
    mx.random.seed(seed)
    model = module.Model(module.ModelArgs.from_dict(args))
    mx.eval(model.parameters())
    return model


def tokens(shape, seed):
    mx.random.seed(seed)
    return mx.random.randint(0, 128, shape)


def prefill(model, inputs, cache, chunks):
    assert sum(chunks) == inputs.shape[1]
    logits, start = [], 0
    for n in chunks:
        logits.append(model(inputs[:, start : start + n], cache=cache))
        mx.eval(logits[-1], [c.state for c in cache])
        start += n
    return mx.concatenate(logits, axis=1)


def greedy(model, cache, y, steps):
    out, logits = [], []
    for _ in range(steps):
        logits.append(model(y, cache=cache)[:, -1])
        y = mx.argmax(logits[-1], axis=-1, keepdims=True)
        mx.eval(y)
        out.append(y)
    return mx.concatenate(out, axis=1).tolist(), mx.stack(logits, axis=1)


def batch_cache(model, left_padding):
    return [
        CacheList(*(BatchKVCache(left_padding) for _ in c.caches))
        for c in model.make_cache()
    ]


class SparseCalls:
    """Count the layers that take the sparse prefill path."""

    def __init__(self):
        self.count = 0
        self._sparse = deepseek_v32.DeepseekV32Attention._sparse_attention

    def __enter__(self):
        def counted(attn, *args):
            self.count += 1
            return self._sparse(attn, *args)

        self._patch = mock.patch.object(
            deepseek_v32.DeepseekV32Attention, "_sparse_attention", counted
        )
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


class DSATestCase(unittest.TestCase):
    # Run on the CPU: the checks compare top-k indices and tokens exactly.
    @classmethod
    def setUpClass(cls):
        cls._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls._device)

    def assertClose(self, a, b, msg=None):
        self.assertEqual(a.shape, b.shape, msg)
        diff = mx.abs(a - b).max().item()
        self.assertTrue(
            mx.allclose(a, b, rtol=1e-4, atol=1e-5).item(), f"{msg}: {diff}"
        )

    def assertCachesMatch(self, cache_a, cache_b, left_padding=None):
        """Same layout; same contents at every key a real query can attend to.

        Layer 0 is computed from the embeddings only, so it must be bitwise
        equal. Later layers see attention outputs, which differ by fp error.
        """
        for i, (ca, cb) in enumerate(zip(cache_a, cache_b)):
            self.assertEqual(len(ca.caches), len(cb.caches))
            for j, (a, b) in enumerate(zip(ca.caches, cb.caches)):
                where = f"layer {i} cache {j}"
                self.assertIs(type(a), type(b), where)
                if isinstance(a, BatchKVCache):
                    self.assertEqual(a._idx, b._idx, where)
                    self.assertEqual(
                        a.left_padding.tolist(), b.left_padding.tolist(), where
                    )
                    self.assertEqual(a.offset.tolist(), b.offset.tolist(), where)
                else:
                    self.assertEqual(a.offset, b.offset, where)
                for x, y in ((a.keys, b.keys), (a.values, b.values)):
                    self.assertEqual(x.shape, y.shape, where)
                    self.assertEqual(x.dtype, y.dtype, where)
                    if left_padding is not None:
                        cols = mx.arange(x.shape[2])[None] >= left_padding[:, None]
                        cols = cols[:, None, :, None]
                        x, y = x * cols, y * cols
                    if i == 0:
                        self.assertTrue(mx.array_equal(x, y).item(), where)
                    else:
                        self.assertClose(x, y, where)
                if j == 1:
                    self.assertFalse(mx.any(a.values).item(), where)

    def assertCachesEqual(self, cache_a, cache_b):
        for a, b in zip(cache_a.caches, cache_b.caches):
            self.assertEqual(a.offset, b.offset)
            self.assertTrue(mx.array_equal(a.keys, b.keys).item())
            self.assertTrue(mx.array_equal(a.values, b.values).item())


class TestSparsePrefill(DSATestCase):
    def test_contexts_below_at_and_above_topk(self):
        for module, args in MODELS:
            model = make_model(module, args)
            n_layers = args["num_hidden_layers"]
            for T in (TOPK - 2, TOPK, TOPK + 1, 3 * TOPK + 5):
                with self.subTest(model=args["model_type"], T=T):
                    x = tokens((1, T), seed=T)
                    with SparseCalls() as calls:
                        out = model(x, cache=model.make_cache())
                        mx.eval(out)
                    with dense_reference():
                        ref = model(x, cache=model.make_cache())
                    self.assertClose(out, ref)
                    # Full and shared layers all take the sparse path above K.
                    self.assertEqual(calls.count, n_layers if T > TOPK else 0)

    def test_no_cache(self):
        for module, args in MODELS:
            with self.subTest(model=args["model_type"]):
                model = make_model(module, args)
                x = tokens((2, 3 * TOPK), seed=1)
                out = model(x)
                with dense_reference():
                    ref = model(x)
                self.assertClose(out, ref)

    def test_chunked_prefill_and_cache_state(self):
        # The first chunk is dense; the others start at offsets 5, 12, 28, 44.
        chunks = [5, 7, 16, 16, 3]
        for module, args in MODELS:
            with self.subTest(model=args["model_type"]):
                model = make_model(module, args)
                x = tokens((1, sum(chunks)), seed=2)
                cache = model.make_cache()
                with SparseCalls() as calls:
                    out = prefill(model, x, cache, chunks)
                ref_cache = model.make_cache()
                with dense_reference():
                    ref = prefill(model, x, ref_cache, chunks)
                self.assertEqual(calls.count, 4 * args["num_hidden_layers"])
                self.assertClose(out, ref)
                self.assertCachesMatch(cache, ref_cache)

    def test_query_blocks(self):
        chunks = [12, 16, 13]
        for module, args in MODELS:
            model = make_model(module, args)
            x = tokens((2, sum(chunks)), seed=3)
            with dense_reference():
                ref = prefill(model, x, model.make_cache(), chunks)
            # One query per block, then three queries per sparse attention
            # block, which do not divide the chunks.
            dims = args["kv_lora_rank"] + args["qk_rope_head_dim"]
            dims += 4 * args["num_attention_heads"]
            for n in (1, 3 * 2 * TOPK * dims * mx.float32.size):
                with self.subTest(model=args["model_type"], block_bytes=n):
                    with block_bytes(n):
                        out = prefill(model, x, model.make_cache(), chunks)
                    self.assertClose(out, ref)

    def test_left_padded_batch(self):
        pad = 10
        chunks = [12, 12, 6]
        left_padding = mx.array([0, pad])
        for module, args in MODELS:
            with self.subTest(model=args["model_type"]):
                model = make_model(module, args)
                x = tokens((2, sum(chunks)), seed=4)
                x[1, :pad] = 0
                cache = batch_cache(model, [0, pad])
                with SparseCalls() as calls:
                    out = prefill(model, x, cache, chunks)
                ref_cache = batch_cache(model, [0, pad])
                with dense_reference():
                    ref = prefill(model, x, ref_cache, chunks)
                self.assertGreater(calls.count, 0)
                # Rows for pad tokens attend to nothing, so only real rows count.
                self.assertClose(out[0], ref[0])
                self.assertClose(out[1, pad:], ref[1, pad:])
                self.assertCachesMatch(cache, ref_cache, left_padding)

                y = mx.argmax(out[:, -1:], axis=-1)
                toks, logits = greedy(model, cache, y, 6)
                ref_toks, ref_logits = greedy(model, ref_cache, y, 6)
                self.assertEqual(toks, ref_toks)
                self.assertClose(logits, ref_logits)

    def test_merged_batch_with_history(self):
        # A new sequence merged next to one with history is left padded, and
        # its logical offset differs from the physical key positions.
        for module, args in MODELS:
            with self.subTest(model=args["model_type"]):
                model = make_model(module, args)
                history = tokens((1, 10), seed=5)
                x = tokens((2, 16), seed=6)

                def run():
                    old = model.make_cache()
                    prefill(model, history, old, [10])
                    new = model.make_cache()
                    cache = [CacheList.merge([a, b]) for a, b in zip(old, new)]
                    out = prefill(model, x, cache, [16])
                    y = mx.argmax(out[:, -1:], axis=-1)
                    return out, cache, greedy(model, cache, y, 6)

                with SparseCalls() as calls:
                    out, cache, (toks, logits) = run()
                with dense_reference():
                    ref, ref_cache, (ref_toks, ref_logits) = run()
                self.assertGreater(calls.count, 0)
                self.assertEqual(cache[0][0].left_padding.tolist(), [0, 10])
                self.assertClose(out, ref)
                self.assertCachesMatch(cache, ref_cache, cache[0][0].left_padding)
                self.assertEqual(toks, ref_toks)
                self.assertClose(logits, ref_logits)

    def test_greedy_decode_after_prefill(self):
        chunks = [16, 16, 8]
        for module, args in MODELS:
            with self.subTest(model=args["model_type"]):
                model = make_model(module, args)
                x = tokens((1, sum(chunks)), seed=7)
                cache = model.make_cache()
                out = prefill(model, x, cache, chunks)
                ref_cache = model.make_cache()
                with dense_reference():
                    ref = prefill(model, x, ref_cache, chunks)
                y = mx.argmax(out[:, -1:], axis=-1)
                self.assertEqual(y.tolist(), mx.argmax(ref[:, -1:], axis=-1).tolist())
                toks, logits = greedy(model, cache, y, 8)
                ref_toks, ref_logits = greedy(model, ref_cache, y, 8)
                self.assertEqual(toks, ref_toks)
                self.assertClose(logits, ref_logits)

    def test_prefill_matches_decode_steps(self):
        # The sparse prefill must agree with the L == 1 top-k gather.
        for module, args in MODELS:
            with self.subTest(model=args["model_type"]):
                model = make_model(module, args)
                x = tokens((1, 3 * TOPK), seed=8)
                out = prefill(model, x, model.make_cache(), [TOPK + 4, 2 * TOPK - 4])
                steps = prefill(model, x, model.make_cache(), [1] * x.shape[1])
                self.assertClose(out, steps)

    def test_gradients(self):
        for module, args in MODELS:
            model = make_model(module, args)
            x = tokens((2, 3 * TOPK), seed=12)

            def loss(model):
                return model(x).square().mean()

            with dense_reference():
                ref = dict(tree_flatten(nn.value_and_grad(model, loss)(model)[1]))
            for n in (deepseek_v32.DSA_BLOCK_BYTES, 1):
                with self.subTest(model=args["model_type"], block_bytes=n):
                    with block_bytes(n):
                        grads = nn.value_and_grad(model, loss)(model)[1]
                    for k, g in tree_flatten(grads):
                        self.assertClose(g, ref[k], k)

    def test_attention_cache_writes_are_unchanged(self):
        # With equal inputs, both paths write bitwise equal caches.
        model = make_model(glm_moe_dsa, GLM_ARGS)
        layers = model.model.layers
        mx.random.seed(9)
        x = mx.random.normal((1, 3 * TOPK, GLM_ARGS["hidden_size"]))
        history = mx.random.normal((1, TOPK + 2, GLM_ARGS["hidden_size"]))

        def run(i, prev_topk):
            attn = layers[i].self_attn
            cache = model.make_cache()[i]
            attn(history, create_causal_mask(history.shape[1]), cache)
            mask = create_causal_mask(x.shape[1], offset=history.shape[1])
            out, next_topk = attn(x, mask, cache, prev_topk_indices=prev_topk)
            mx.eval(out, cache.state)
            return out, next_topk, cache

        out, topk, cache = run(0, None)
        with dense_reference():
            ref, ref_topk, ref_cache = run(0, None)
        self.assertClose(out, ref)
        self.assertTrue(mx.array_equal(topk, ref_topk).item())
        self.assertCachesEqual(cache, ref_cache)

        # A shared layer has no indexer cache and reuses the top-k it is given,
        # which here also selects keys after the query.
        self.assertEqual(len(model.make_cache()[1].caches), 1)
        prev = mx.argsort(mx.random.normal((1, 1, x.shape[1], 2 * TOPK + 2)))
        prev = prev[..., -TOPK:]
        out, _, cache = run(1, prev)
        with dense_reference():
            ref, _, ref_cache = run(1, prev)
        self.assertClose(out, ref)
        self.assertCachesEqual(cache, ref_cache)


class TestBlockedIndexer(DSATestCase):
    def run_indexer(self, indexer, xs, qrs, masks, cache):
        out = None
        for x, qr, mask in zip(xs, qrs, masks):
            out = indexer(x, qr, mask, cache)
        mx.eval(out, cache.state)
        return out

    def test_topk_is_identical(self):
        args = deepseek_v32.ModelArgs.from_dict(DSV32_ARGS)
        for dtype in (mx.float32, mx.bfloat16):
            mx.random.seed(10)
            indexer = deepseek_v32.Indexer(args)
            indexer.set_dtype(dtype)
            B, H, S0, L = 2, args.index_n_heads, 120, 11
            xs = [
                mx.random.normal((B, n, args.hidden_size)).astype(dtype)
                for n in (S0, L)
            ]
            qrs = [
                mx.random.normal((B, n, args.q_lora_rank)).astype(dtype)
                for n in (S0, L)
            ]
            for left_padding in (None, [0, 3]):
                if left_padding is None:
                    masks = [create_causal_mask(S0), create_causal_mask(L, offset=S0)]
                    make_cache = KVCache
                else:
                    lp = mx.array(left_padding)
                    masks = [
                        create_causal_mask(S0, left_padding=lp),
                        create_causal_mask(L, offset=S0, left_padding=lp),
                    ]
                    make_cache = partial(BatchKVCache, left_padding)
                ref_cache = make_cache()
                with dense_reference():
                    ref = self.run_indexer(indexer, xs, qrs, masks, ref_cache)
                row_bytes = B * H * (S0 + L) * dtype.size
                for rows in (1, 4, L):
                    with self.subTest(dtype=dtype, padding=left_padding, rows=rows):
                        calls = []
                        query_blocks = deepseek_v32._query_blocks

                        def record(length, row_bytes):
                            calls.append(list(query_blocks(length, row_bytes)))
                            return iter(calls[-1])

                        cache = make_cache()
                        with (
                            block_bytes(rows * row_bytes),
                            mock.patch.object(deepseek_v32, "_query_blocks", record),
                        ):
                            out = self.run_indexer(indexer, xs, qrs, masks, cache)
                        self.assertEqual(len(calls[-1]), -(-L // rows))
                        self.assertEqual(calls[-1][-1], ((L - 1) // rows * rows, L))
                        self.assertTrue(mx.array_equal(out, ref).item())
                        self.assertEqual(out.dtype, ref.dtype)
                        for a, b in zip(cache.state, ref_cache.state):
                            if isinstance(a, mx.array):
                                self.assertTrue(mx.array_equal(a, b).item())
                        self.assertFalse(mx.any(cache.values).item())

    def test_decode_topk_is_unchanged(self):
        args = deepseek_v32.ModelArgs.from_dict(DSV32_ARGS)
        mx.random.seed(11)
        indexer = deepseek_v32.Indexer(args)
        n = 2 * TOPK
        xs = [mx.random.normal((1, m, args.hidden_size)) for m in (n, 1)]
        qrs = [mx.random.normal((1, m, args.q_lora_rank)) for m in (n, 1)]
        masks = [create_causal_mask(n), None]
        with dense_reference():
            ref = self.run_indexer(indexer, xs, qrs, masks, KVCache())
        with block_bytes(1):
            out = self.run_indexer(indexer, xs, qrs, masks, KVCache())
        self.assertEqual(out.shape, (1, 1, 1, TOPK))
        self.assertTrue(mx.array_equal(out, ref).item())


class TestPeakMemory(DSATestCase):
    """With a fixed block budget, the peak memory of a call is flat in S."""

    L = 1024
    CONTEXTS = (16384, 65536)
    BUDGET = 2 * 2**20

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.args = deepseek_v32.ModelArgs.from_dict({**DSV32_ARGS, "index_topk": 256})
        mx.random.seed(13)
        cls.attn = deepseek_v32.DeepseekV32Attention(cls.args)
        mx.eval(cls.attn.parameters())

    def peak(self, fn, inputs):
        mx.eval(inputs)
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        with block_bytes(self.BUDGET):
            mx.eval(fn())
        return mx.get_peak_memory() - base

    def assertBounded(self, peaks, factor):
        (s1, p1), (s2, p2) = peaks
        # One byte for each new (query, key) pair would add L * (s2 - s1).
        self.assertLess(p2 - p1, self.L * (s2 - s1) // 4, peaks)
        # Blocks in flight overlap, so the peak is a few times the budget.
        # Without the depends chains, ~20 blocks interleave and it is ~12x.
        self.assertLess(max(p1, p2), factor * self.BUDGET, peaks)

    def test_indexer(self):
        args, L = self.args, self.L
        peaks = []
        for S in self.CONTEXTS:
            # Room for all S keys, so the cache does not grow in the call.
            cache = KVCache()
            keys = mx.zeros((1, 1, S, args.index_head_dim))
            cache.state = (keys, mx.zeros_like(keys), S - L)
            del keys
            x = mx.random.normal((1, L, args.hidden_size))
            qr = mx.random.normal((1, L, args.q_lora_rank))
            mask = create_causal_mask(L, offset=S - L)
            fn = partial(self.attn.indexer, x, qr, mask, cache)
            peaks.append((S, self.peak(fn, (x, qr, mask, cache.state))))
        # Each block holds about 3 score tensors of the budget size.
        self.assertBounded(peaks, factor=8)

    def test_sparse_attention(self):
        args, L, H = self.args, self.L, self.args.num_attention_heads
        peaks = []
        for S in self.CONTEXTS:
            q_nope = mx.random.normal((1, H, L, args.qk_nope_head_dim))
            q_pe = mx.random.normal((1, H, L, args.qk_rope_head_dim))
            kv_latent = mx.random.normal((1, 1, S, args.kv_lora_rank))
            k_pe = mx.random.normal((1, 1, S, args.qk_rope_head_dim))
            topk = mx.random.randint(0, S, (1, 1, L, args.index_topk), mx.uint32)
            mask = create_causal_mask(L, offset=S - L)
            inputs = (q_nope, q_pe, kv_latent, k_pe, topk, mask)
            fn = partial(self.attn._sparse_attention, *inputs)
            peaks.append((S, self.peak(fn, inputs)))
        self.assertBounded(peaks, factor=4)


if __name__ == "__main__":
    unittest.main()
