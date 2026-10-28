# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models import glm5_next

SMALL = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=8,
    num_experts_per_tok=2,
    kv_lora_rank=32,
    q_lora_rank=48,
    qk_nope_head_dim=16,
    v_head_dim=16,
    index_topk=8,
    index_kpool=4,
    index_head_dim=16,
    index_n_heads=2,
    layer_types=[
        "linear_attention",
        "indexed_attention",
        "linear_attention",
        "indexed_attention",
    ],
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    linear_attn_config=dict(num_heads=2, head_dim=32, short_conv_kernel_size=4),
)


class TestGlm5Next(unittest.TestCase):
    def test_safe_gate_is_default(self):
        args = glm5_next.ModelArgs(**SMALL)
        self.assertEqual(args.linear_attn_config["gate_lower_bound"], -5.0)

    def test_cached_matches_forward(self):
        mx.random.seed(0)
        model = glm5_next.Model(glm5_next.ModelArgs(**SMALL))
        tokens = mx.random.randint(0, 128, (2, 23))
        full = model(tokens)

        # The steps cross index_topk, where the indexer starts to select.
        cache = model.make_cache()
        steps = []
        for start, end in ((0, 5), (5, 9), (9, 21), (21, 22), (22, 23)):
            steps.append(model(tokens[:, start:end], cache=cache))
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4)
        )

    def test_indexer_selection(self):
        mx.random.seed(0)
        indexer = glm5_next.Indexer(glm5_next.ModelArgs(**SMALL))
        x = mx.random.normal((1, 23, 64))
        self.assertIsNone(indexer(x[:, :8], mx.random.normal((1, 8, 48)), None))
        topk = indexer(x, mx.random.normal((1, 23, 48)), None)
        self.assertEqual(topk.shape, (1, 23, 8 + 3))
        for position, row in enumerate(topk[0].tolist()):
            picked = [i for i in row if i >= 0]
            self.assertEqual(len(picked), len(set(picked)))
            self.assertTrue(all(i <= position for i in picked))
            # Two pools of the full pools it sees, and all keys after them.
            full_pools = (position + 1) // 4
            tail = list(range(full_pools * 4, position + 1))
            self.assertEqual(picked[-len(tail) :] if tail else [], tail)
            self.assertEqual(len(picked), 4 * min(2, full_pools) + len(tail))

    def test_sanitize(self):
        args = glm5_next.ModelArgs(**SMALL)
        model = glm5_next.Model(args)
        prefix = "model.language_model.layers"
        weights = {
            "model.visual.blocks.0.attn.qkv.weight": mx.zeros((3, 3)),
            f"{prefix}.4.self_attn.q_proj.weight": mx.zeros((1,)),
            f"{prefix}.0.hc_attn_fn": mx.zeros((24, 256)),
            f"{prefix}.0.self_attn.q_conv1d.weight": mx.zeros((64, 1, 4)),
            f"{prefix}.0.self_attn.A_log": mx.zeros((2,)),
            f"{prefix}.1.self_attn.kv_b_proj.weight": mx.zeros((128, 32)),
            "model.language_model.norm.weight": mx.zeros((64,)),
            "lm_head.weight": mx.zeros((128, 64)),
        }
        for e in range(8):
            weights[f"{prefix}.1.mlp.experts.{e}.gate_proj.weight"] = mx.zeros((32, 64))
        clean = model.sanitize(weights)
        layers = "model.layers"
        self.assertEqual(
            sorted(clean),
            sorted(
                [
                    f"{layers}.0.attn_hc.fn",
                    f"{layers}.0.self_attn.q_conv.conv.weight",
                    f"{layers}.0.self_attn.A_log",
                    f"{layers}.1.self_attn.embed_q.weight",
                    f"{layers}.1.self_attn.unembed_out.weight",
                    f"{layers}.1.mlp.switch_mlp.gate_proj.weight",
                    "model.norm.weight",
                    "lm_head.weight",
                ]
            ),
        )
        self.assertEqual(
            clean[f"{layers}.0.self_attn.q_conv.conv.weight"].shape, (64, 4, 1)
        )
        self.assertEqual(clean[f"{layers}.0.self_attn.A_log"].shape, (1, 1, 2, 1))
        self.assertEqual(
            clean[f"{layers}.1.self_attn.embed_q.weight"].shape, (4, 32, 16)
        )
        self.assertEqual(
            clean[f"{layers}.1.mlp.switch_mlp.gate_proj.weight"].shape, (8, 32, 64)
        )
        # A converted checkpoint keeps its names.
        self.assertEqual(sorted(model.sanitize(dict(clean))), sorted(clean))

    def test_unsupported_variants(self):
        for change in (
            {"indexer_types": ["full", "shared", "full", "full"]},
            {"qk_rope_head_dim": 64},
            {"index_kpool_always_select_tail": False},
        ):
            with self.assertRaises(ValueError):
                glm5_next.ModelArgs(**{**SMALL, **change})


if __name__ == "__main__":
    unittest.main()
