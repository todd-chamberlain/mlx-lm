# Copyright © 2026 Apple Inc.

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, CacheList, KVCache
from .deepseek_v41 import (
    DeepseekV41SwiGLU,
    HyperConnection,
    Router,
    _collapse,
    _expand,
    _gather_rows,
)
from .kimi_linear import KimiDeltaAttention
from .mla import MultiLinear, latent_length, max_absorbed_queries
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU

# Size limit for the largest tensor of one query block in the indexer and in the
# sparse attention, so that their memory does not grow with queries x keys.
QUERY_BLOCK_BYTES = 1 << 30


def _query_blocks(length: int, row_bytes: int):
    step = max(1, QUERY_BLOCK_BYTES // row_bytes)
    for start in range(0, length, step):
        yield start, min(start + step, length)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm5_next"
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    scoring_func: str = "sigmoid"
    gate_temp: float = 1.0
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    attention_bias: bool = False
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True
    layer_types: Optional[List[str]] = None
    mlp_layer_types: Optional[List[str]] = None
    indexer_types: Optional[List[str]] = None
    linear_attn_config: Optional[Dict[str, Any]] = None
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        if "text_config" in params:
            params = {**params["text_config"], "model_type": params["model_type"]}
        return super().from_dict(params)

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if i % 4 != 3 else "indexed_attention"
                for i in range(n)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, n) + ["sparse"] * (n - 3)
        cfg = dict(self.linear_attn_config or {})
        # The safe gate is on unless the config turns it off.
        if cfg.get("gate_lower_bound") is None and cfg.get("safe_gate", True):
            cfg["gate_lower_bound"] = -5.0
        self.linear_attn_config = cfg
        if self.qk_rope_head_dim:
            raise ValueError("glm5_next supports only NoPE attention.")
        if any(t != "full" for t in self.indexer_types or ()):
            raise ValueError("glm5_next does not support shared indexer layers.")
        if not self.index_kpool_always_select_tail:
            raise ValueError("glm5_next needs index_kpool_always_select_tail.")
        if self.index_topk % self.index_kpool:
            raise ValueError("index_topk must be a multiple of index_kpool.")
        if self.n_group != self.topk_group:
            raise ValueError("glm5_next does not support grouped routing.")


class Glm5NextMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.activation = DeepseekV41SwiGLU(args.swiglu_limit)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.activation(self.up_proj(x), self.gate_proj(x))
        return self.down_proj(h.astype(x.dtype))


class Glm5NextMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=DeepseekV41SwiGLU(args.swiglu_limit),
        )
        self.shared_experts = Glm5NextMLP(
            args, args.moe_intermediate_size * args.n_shared_experts
        )

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights = self.gate(x)
        routed = self.switch_mlp(x, indices)
        routed = mx.sum(routed.astype(mx.float32) * weights[..., None], axis=-2)
        return (routed + self.shared_experts(x).astype(mx.float32)).astype(x.dtype)


class Indexer(nn.Module):
    """Pick the keys of each query: the top scoring pools of index_kpool keys,
    and the keys after the last full pool."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.pool = args.index_kpool
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.index_kpool_compress_ape = mx.zeros((self.pool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, args.hidden_size))

    def _pool_keys(self, keys: mx.array, gates: mx.array) -> mx.array:
        """One key for each full pool: the gated mean of its keys."""
        batch, length, dim = keys.shape
        pools = length // self.pool
        keys = keys[:, : pools * self.pool].reshape(batch, pools, self.pool, dim)
        gates = gates[:, : pools * self.pool].reshape(batch, pools, self.pool, dim)
        logits = gates + self.index_kpool_compress_ape.astype(mx.float32)
        probs = mx.softmax(logits, axis=2, precise=True).astype(keys.dtype)
        return mx.sum(probs * keys, axis=2)

    def __call__(
        self, x: mx.array, qr: mx.array, cache: Optional[KVCache]
    ) -> Optional[mx.array]:
        """The key indices of each query as [B, L, index_topk + index_kpool - 1],
        -1 for none, or None if each query can attend to all its keys."""
        batch, length, _ = x.shape
        k = self.k_norm(self.wk(x))[:, None]
        gate = x.astype(mx.float32) @ self.index_kpool_compress_gate.astype(
            mx.float32
        ).swapaxes(0, 1)
        gate = gate[:, None]
        if cache is not None:
            k, gate = cache.update_and_fetch(k, gate)
        n_keys = k.shape[2]
        # A query with no more keys than index_topk selects all of them.
        if n_keys <= self.index_topk:
            return None

        pool_keys = self._pool_keys(k[:, 0], gate[:, 0]).astype(mx.float32)
        pools = pool_keys.shape[1]
        select = self.index_topk // self.pool
        q = self.wq_b(qr).reshape(batch, length, self.n_heads, self.head_dim)
        scale = self.n_heads**-0.5 * self.head_dim**-0.5
        weights = self.weights_proj(x).astype(mx.float32) * scale
        positions = n_keys - length + mx.arange(length)
        pool_ends = (mx.arange(pools) + 1) * self.pool - 1
        offsets = mx.arange(self.pool)

        row_bytes = batch * self.n_heads * pools * 4
        selected = []
        for start, end in _query_blocks(length, row_bytes):
            q_block = q[:, start:end].astype(mx.float32)
            if selected:
                # Run the blocks in sequence, so only one block is in memory.
                q_block = mx.depends(q_block, selected[-1])
            scores = mx.maximum(q_block @ pool_keys[:, None].swapaxes(-1, -2), 0)
            scores = mx.sum(scores * weights[:, start:end, :, None], axis=2)
            # A pool is visible when the query sees its last key.
            visible = pool_ends[None] <= positions[start:end, None]
            scores = mx.where(visible, scores, -mx.inf)
            chosen = mx.argpartition(-scores, select - 1, axis=-1)[..., :select]
            valid = mx.take_along_axis(
                mx.broadcast_to(visible, scores.shape), chosen, axis=-1
            )
            keys = chosen[..., None] * self.pool + offsets
            keys = mx.where(valid[..., None], keys, -1)
            selected.append(keys.reshape(batch, end - start, -1))
        topk = selected[0] if len(selected) == 1 else mx.concatenate(selected, 1)

        # The keys after the last full pool that the query sees.
        tail_count = (positions + 1) % self.pool
        tail = (positions + 1 - tail_count)[:, None] + offsets[:-1]
        tail = mx.where(offsets[:-1] < tail_count[:, None], tail, -1)
        tail = mx.broadcast_to(tail, (batch, *tail.shape))
        return mx.concatenate([topk, tail], axis=-1).astype(mx.int32)


class Glm5NextAttention(nn.Module):
    """MLA without RoPE, over the keys that the indexer picks."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = args.qk_nope_head_dim**-0.5
        self._absorbed_dims = (
            args.kv_lora_rank,
            args.qk_nope_head_dim,
            args.v_head_dim,
        )
        self.q_a_proj = nn.Linear(
            args.hidden_size, args.q_lora_rank, bias=args.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * args.qk_nope_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size, args.kv_lora_rank, bias=args.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            args.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, args.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * args.v_head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.indexer = Indexer(args)

    def _sparse_attention(
        self, q: mx.array, latent: mx.array, topk: mx.array
    ) -> mx.array:
        """Attend each query to the latent rows that its indices pick."""
        batch, heads, length, _ = q.shape
        k = topk.shape[-1]
        q = self.embed_q(q)
        rows = latent[:, 0]
        valid = topk >= 0
        safe = mx.maximum(topk, 0)
        row_bytes = batch * k * self.kv_lora_rank * rows.dtype.size
        outputs = []
        for start, end in _query_blocks(length, row_bytes):
            n = end - start
            idx = safe[:, start:end]
            if outputs:
                # Run the blocks in sequence, so only one block is in memory.
                idx = mx.depends(idx, outputs[-1])
            kv = _gather_rows(rows, idx).reshape(batch * n, 1, k, -1)
            # Each query has its own keys, so each query is one batch entry.
            q_block = q[:, :, start:end].transpose(0, 2, 1, 3)
            out = mx.fast.scaled_dot_product_attention(
                q_block.reshape(batch * n, heads, 1, -1),
                kv,
                kv,
                scale=self.scale,
                mask=valid[:, start:end].reshape(batch * n, 1, 1, k),
            )
            outputs.append(out.reshape(batch, n, heads, -1).transpose(0, 2, 1, 3))
        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)
        return self.unembed_out(out)

    def __call__(
        self, x: mx.array, mask: Optional[Any], cache: Optional[CacheList]
    ) -> mx.array:
        batch, length, _ = x.shape
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(batch, length, self.num_heads, -1)
        q = q.transpose(0, 2, 1, 3)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))[:, None]
        if cache is not None:
            # NoPE: the latent is the whole key, so the values are empty.
            latent, _ = cache[0].update_and_fetch(latent, latent[..., :0])
        topk = self.indexer(x, qr, None if cache is None else cache[1])
        if cache is not None:
            # Keep the indexer cache evaluated when the topk is not used.
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if topk is not None:
            out = self._sparse_attention(q, latent, topk)
        else:
            absorbed = length == 1 or length <= max_absorbed_queries(
                *self._absorbed_dims, latent_length(latent)
            )
            if absorbed:
                q = self.embed_q(q)
                k = v = latent
            else:
                k = self.embed_q(latent, transpose=False)
                v = self.unembed_out(latent)
            out = scaled_dot_product_attention(
                q,
                k,
                v,
                cache=None if cache is None else cache[0],
                scale=self.scale,
                mask=mask,
            )
            if absorbed:
                out = self.unembed_out(out)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = Glm5NextAttention(args)
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm5NextMoE(args)
        else:
            self.mlp = Glm5NextMLP(args, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(
        self, streams: mx.array, mask: Optional[Any], cache: Optional[Any]
    ) -> mx.array:
        # Unlike DeepSeek-V4.1, each block mixes its input with its own pre.
        pre, post, comb = self.attn_hc(streams)
        h = self.self_attn(self.input_layernorm(_collapse(streams, pre)), mask, cache)
        streams = _expand(h, streams, post, comb)
        pre, post, comb = self.ffn_hc(streams)
        h = self.mlp(self.post_attention_layernorm(_collapse(streams, pre)))
        return _expand(h, streams, post, comb)


def _depend_on(cache: Any, sent: mx.array) -> None:
    """Make the state of the cache depend on the send."""
    if isinstance(cache, CacheList):
        cache[0].keys = mx.depends(cache[0].keys, sent)
    else:
        cache[3] = mx.depends(cache[3], sent)


class Glm5NextModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None):
        h = self.embed_tokens(inputs)
        layers = self.pipeline_layers
        cache = cache or [None] * len(layers)
        ssm_cache = next((c for l, c in zip(layers, cache) if l.is_linear), None)
        attn_cache = next((c for l, c in zip(layers, cache) if not l.is_linear), None)
        ssm_mask = create_ssm_mask(h, ssm_cache)
        attn_mask = create_attention_mask(h, attn_cache[0] if attn_cache else None)

        batch, length, dim = h.shape
        streams = mx.broadcast_to(
            h[:, :, None], (batch, length, self.args.hc_mult, dim)
        )
        rank, size = self.pipeline_rank, self.pipeline_size
        if rank < size - 1:
            streams = mx.distributed.recv_like(streams, rank + 1)
        for layer, layer_cache in zip(layers, cache):
            mask = ssm_mask if layer.is_linear else attn_mask
            streams = layer(streams, mask, layer_cache)
        if rank > 0:
            sent = mx.distributed.send(streams, rank - 1)
            streams = mx.depends(streams, sent)
            # Prefill only evaluates the cache, which must still send.
            if cache[-1] is not None:
                _depend_on(cache[-1], sent)

        out = mx.mean(streams.astype(mx.float32), axis=2).astype(h.dtype)
        if size > 1:
            # Rank 0 runs the last layers, and every rank needs its output.
            out = mx.distributed.all_gather(out)[:batch]
        return self.norm(out)


_HC = re.compile(r"\.hc_(attn|ffn)_(fn|base|scale)$")
_CONV = re.compile(r"\.([qkv])_conv1d\.weight$")
_EXPERT = re.compile(
    r"(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm5NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else CacheList(KVCache(), KVCache())
            for layer in self.layers
        ]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Drop the vision and MTP weights, and map the names of the text model."""
        args = self.args
        clean = {}
        experts = {}
        for key, value in weights.items():
            key = key.replace("model.language_model.", "model.", 1)
            if not key.startswith(
                ("model.layers.", "model.embed_tokens.")
            ) and key not in (
                "model.norm.weight",
                "lm_head.weight",
            ):
                continue
            layer = re.match(r"model\.layers\.(\d+)\.", key)
            if layer is not None and int(layer.group(1)) >= args.num_hidden_layers:
                continue
            key = _HC.sub(r".\1_hc.\2", key)
            if conv := _CONV.search(key):
                key = key[: conv.start()] + f".{conv.group(1)}_conv.conv.weight"
                # [channels, 1, kernel] -> [channels, kernel, 1]
                value = value.moveaxis(2, 1)
            if key.endswith(".self_attn.A_log"):
                value = value.reshape(1, 1, -1, 1)
            if expert := _EXPERT.match(key):
                prefix, index, proj, kind = expert.groups()
                experts.setdefault((prefix, proj, kind), {})[int(index)] = value
                continue
            clean[key] = value

        for (prefix, proj, kind), parts in experts.items():
            if len(parts) != args.n_routed_experts:
                raise ValueError(
                    f"{prefix} has {len(parts)} {proj} experts, but the config "
                    f"declares {args.n_routed_experts}."
                )
            clean[f"{prefix}.switch_mlp.{proj}.{kind}"] = mx.stack(
                [parts[i] for i in range(args.n_routed_experts)]
            )

        heads, nope = args.num_attention_heads, args.qk_nope_head_dim
        for key in [k for k in clean if k.endswith(".kv_b_proj.weight")]:
            prefix = key.removesuffix(".kv_b_proj.weight")
            if f"{prefix}.kv_b_proj.scales" in clean:
                raise ValueError(f"{key} is quantized; convert it in bf16.")
            weight = clean.pop(key).reshape(heads, nope + args.v_head_dim, -1)
            clean[f"{prefix}.embed_q.weight"] = mx.contiguous(
                weight[:, :nope].swapaxes(-1, -2)
            )
            clean[f"{prefix}.unembed_out.weight"] = mx.contiguous(weight[:, nope:])
        return clean

    @property
    def cast_predicate(self):
        keep = (
            "attn_hc.",
            "ffn_hc.",
            "e_score_correction_bias",
            "A_log",
            "dt_bias",
            "index_kpool_compress",
        )

        def predicate(k: str) -> bool:
            return not any(s in k for s in keep)

        return predicate
