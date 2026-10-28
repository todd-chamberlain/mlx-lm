# Copyright © 2026 Apple Inc.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import deepseek_v32 as D
from mlx_lm.models import dsa
from mlx_lm.models import glm_moe_dsa as G
from mlx_lm.models.cache import make_prompt_cache

GLM_CONFIG = {
    "model_type": "glm_moe_dsa",
    "vocab_size": 256,
    "hidden_size": 64,
    "index_head_dim": 16,
    "index_n_heads": 8,
    "index_topk": 16,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "routed_scaling_factor": 1.0,
    "kv_lora_rank": 32,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "qk_nope_head_dim": 16,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 1,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "attention_bias": False,
    "index_topk_pattern": "FSFSFS",
}


# DeepSeek-V3.2 with the same sizes; every layer has its own indexer.
DSV32_CONFIG = {
    **{
        k: v
        for k, v in GLM_CONFIG.items()
        if k not in ("rope_parameters", "index_topk_pattern")
    },
    "model_type": "deepseek_v32",
    "rope_theta": 10000.0,
}


class _SparsePrefillTests:
    module = config = None  # set by the model test classes below

    def _args(self):
        return self.module.ModelArgs.from_dict({**self.config, "sparse_prefill": False})

    def setUp(self):
        # Tiny random models: GPU fp32 matmul rounding is larger than the dense/sparse difference we check.
        self._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self._device)

    def _prefill(self, model, tokens, chunk):
        cache = make_prompt_cache(model)
        outs = []
        for s in range(0, tokens.shape[1], chunk):
            outs.append(model(tokens[:, s : s + chunk], cache=cache))
        return mx.concatenate(outs, axis=1)

    def test_default_is_on(self):
        layers = self.module.Model(
            self.module.ModelArgs.from_dict(self.config)
        ).model.layers
        self.assertTrue(all(layer.self_attn.sparse_prefill for layer in layers))

    def test_sparse_prefill_matches_dense(self):
        mx.random.seed(0)
        model = self.module.Model(self._args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(
            0, 256, (1, 128)
        )  # longer than index_topk, so sparse path is used
        dense = self._prefill(model, tokens, 32)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        sparse = self._prefill(model, tokens, 32)
        mx.eval(dense, sparse)
        self.assertLess(float(mx.abs(dense - sparse).max()), 1e-4)
        self.assertTrue(bool(mx.all(mx.argmax(dense, -1) == mx.argmax(sparse, -1))))

    def test_blocked_topk_matches_indexer(self):
        from mlx_lm.models.base import create_causal_mask

        model = self.module.Model(self._args())
        mx.eval(model.parameters())
        indexer = model.model.layers[0].self_attn.indexer
        x = mx.random.normal((1, 64, 64))
        qr = mx.random.normal((1, 64, 32))
        mask = create_causal_mask(64)
        stock = indexer(x, qr, mask)
        blocked = dsa.blocked_topk(indexer, x, qr, mask, None, block=16)
        # Rows with fewer than index_topk causal keys also select masked keys.
        rows = slice(indexer.index_topk - 1, None)
        self.assertTrue(
            bool(
                mx.array_equal(
                    mx.sort(stock[..., rows, :], -1), mx.sort(blocked[..., rows, :], -1)
                )
            )
        )

    def test_short_context_is_unchanged(self):
        # With at most index_topk keys every key is selected, so the dense path runs.
        mx.random.seed(0)
        model = self.module.Model(self._args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(0, 256, (1, 16))
        dense = self._prefill(model, tokens, 16)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        sparse = self._prefill(model, tokens, 16)
        self.assertTrue(bool(mx.array_equal(dense, sparse)))

    def test_batch_uses_dense_path(self):
        mx.random.seed(0)
        model = self.module.Model(self._args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(0, 256, (2, 96))
        dense = self._prefill(model, tokens, 32)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        sparse = self._prefill(model, tokens, 32)
        self.assertTrue(bool(mx.array_equal(dense, sparse)))

    def test_min_context_uses_dense_path(self):
        mx.random.seed(0)
        model = self.module.Model(self._args())
        mx.eval(model.parameters())
        tokens = mx.random.randint(0, 256, (1, 128))
        dense = self._prefill(model, tokens, 32)
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 129
        gated = self._prefill(model, tokens, 32)
        self.assertTrue(bool(mx.array_equal(dense, gated)))

    def test_query_blocks_match(self):
        mx.random.seed(0)
        model = self.module.Model(self._args())
        mx.eval(model.parameters())
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        tokens = mx.random.randint(0, 256, (1, 128))
        whole = self._prefill(model, tokens, 64)
        with patch.object(dsa, "SPARSE_QUERY_BLOCK", 24):
            blocked = self._prefill(model, tokens, 64)
        self.assertTrue(bool(mx.array_equal(whole, blocked)))

    def test_decode_after_sparse_prefill(self):
        mx.random.seed(0)
        model = self.module.Model(self._args())
        mx.eval(model.parameters())
        for layer in model.model.layers:
            layer.self_attn.sparse_prefill = True
            layer.self_attn.sparse_prefill_min_context = 0
        cache = make_prompt_cache(model)
        tokens = mx.random.randint(0, 256, (1, 150))
        for s in range(0, 150, 64):
            mx.eval(model(tokens[:, s : s + 64], cache=cache))
        out = model(mx.array([[5]]), cache=cache)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 1, 256))


class TestGlmSparsePrefill(_SparsePrefillTests, unittest.TestCase):
    module, config = G, GLM_CONFIG


class TestDeepseekV32SparsePrefill(_SparsePrefillTests, unittest.TestCase):
    module, config = D, DSV32_CONFIG


class Pipeline:
    """Run the stages of a pipelined model in order, in one process.

    A mailbox replaces the distributed send and recv, so each stage gets
    exactly what the stage before it sent.
    """

    def __init__(self, args, ranges):
        # ranges[r] is the [start, end) of layers of rank r.
        self.size = len(ranges)
        self.stages = []
        for rank in range(self.size):
            mx.random.seed(0)
            model = G.Model(args)
            mx.eval(model.parameters())
            group = SimpleNamespace(rank=lambda r=rank: r, size=lambda: self.size)
            model.model.pipeline(group, split=[end - start for start, end in ranges])
            self.stages.append(model)
        self.sent = []

    def make_cache(self):
        return [make_prompt_cache(stage) for stage in self.stages]

    def __call__(self, inputs, cache):
        box = {}

        def send(x, dst, **kwargs):
            box[(rank, dst)] = x
            self.sent.append((rank, x.shape))
            return x

        def recv(shape, dtype, src, **kwargs):
            x = box.pop((src, rank))
            assert x.shape == tuple(shape) and x.dtype == dtype, (x.shape, shape)
            return x

        def recv_like(x, src, **kwargs):
            return recv(x.shape, x.dtype, src)

        with patch.multiple(
            mx.distributed,
            send=send,
            recv=recv,
            recv_like=recv_like,
            all_gather=lambda x, **kwargs: x,
        ):
            for rank in reversed(range(self.size)):
                out = self.stages[rank](inputs, cache=cache[rank])
                mx.eval(out, [c.state for c in cache[rank]])
        assert not box, "every message is received"
        return out


class TestPipelineTopk(unittest.TestCase):
    """A stage that starts on a shared layer attends with the topk it reuses."""

    # Layer types: full, shared, full, shared, full, shared.
    SHARED_BOUNDARIES = [(3, 6), (1, 3), (0, 1)]
    FULL_BOUNDARIES = [(4, 6), (2, 4), (0, 2)]

    def setUp(self):
        self._device = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self._device)

    def run_both(self, ranges, chunks, batch=1, sparse_prefill=False, steps=4):
        args = G.ModelArgs.from_dict({**GLM_CONFIG, "sparse_prefill": sparse_prefill})
        args.sparse_prefill_min_context = 0
        mx.random.seed(0)
        ref = G.Model(args)
        mx.eval(ref.parameters())
        pipe = Pipeline(args, ranges)
        inputs = mx.random.randint(0, 256, (batch, sum(chunks)))
        ref_cache, pipe_cache = make_prompt_cache(ref), pipe.make_cache()
        start = 0
        for n in chunks:
            chunk = inputs[:, start : start + n]
            start += n
            expected = ref(chunk, cache=ref_cache)
            self.assertTrue(mx.array_equal(pipe(chunk, pipe_cache), expected).item())
        y = mx.argmax(expected[:, -1:], axis=-1)
        for _ in range(steps):
            expected = ref(y, cache=ref_cache)
            self.assertTrue(mx.array_equal(pipe(y, pipe_cache), expected).item())
            y = mx.argmax(expected[:, -1:], axis=-1)
        return pipe

    def test_stage_starts_on_shared_layer(self):
        # Chunks cross index_topk, then decode with the topk handed over.
        pipe = self.run_both(self.SHARED_BOUNDARIES, [5, 7, 16, 9])
        D = GLM_CONFIG["hidden_size"]
        widths = {shape[-1] for rank, shape in pipe.sent if rank > 0}
        self.assertEqual(widths, {D, D + GLM_CONFIG["index_topk"]})

    def test_sparse_prefill_hands_over_topk(self):
        self.run_both(self.SHARED_BOUNDARIES, [5, 7, 16, 9], sparse_prefill=True)

    def test_batch(self):
        self.run_both(self.SHARED_BOUNDARIES, [6, 14, 4], batch=2)

    def test_stage_starts_on_full_layer(self):
        pipe = self.run_both(self.FULL_BOUNDARIES, [5, 7, 16, 9])
        D = GLM_CONFIG["hidden_size"]
        self.assertEqual({shape[-1] for _, shape in pipe.sent}, {D})

    def test_message_dtypes(self):
        # The topk travels as the bits of h's dtype.
        model = G.Model(G.ModelArgs.from_dict(GLM_CONFIG)).model
        K = GLM_CONFIG["index_topk"]
        topk = mx.random.randint(0, 2**30, (2, 1, 3, K)).astype(mx.uint32)
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            h = mx.random.normal((2, 3, GLM_CONFIG["hidden_size"])).astype(dtype)
            message = model._with_topk(h, topk)
            self.assertEqual(message.shape[-1], h.shape[-1] + K * 4 // dtype.size)
            with patch.object(mx.distributed, "recv", lambda *a, **k: message):
                got_h, got_topk = model._recv_with_topk(h, 1)
            self.assertTrue(mx.array_equal(got_h, h).item(), dtype)
            self.assertEqual(got_topk.dtype, mx.uint32)
            self.assertTrue(mx.array_equal(got_topk, topk).item(), dtype)


if __name__ == "__main__":
    unittest.main()
