# Copyright © 2026 Apple Inc.

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import mlx.core as mx

from mlx_lm.models import deepseek_v41

SMALL = dict(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=6,
    num_attention_heads=4,
    head_dim=32,
    q_lora_rank=32,
    qk_rope_head_dim=8,
    o_groups=2,
    o_lora_rank=16,
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=2,
    sliding_window=4,
    compress_ratios=(0, 2, 2, 2, 1, 1),
    kv_source_layer_ids=(1, 4),
    index_source_layer_ids=(1, 2, 4, 5),
    index_n_heads=4,
    index_head_dim=32,
    index_topk=3,
    candidate_source_layer_id=4,
    candidate_topk_blocks=2,
    candidate_block_size=2,
)

ENGRAM = dict(
    engram_layer_ids=(1, 3),
    engram_num_embeddings=(240, 306),
    engram_max_ngram_size=3,
    engram_vocab_size=50,
    engram_n_heads=2,
    engram_head_dim=32,
    engram_compressed_vocab_size=40,
)


def engram_model(**overrides):
    args = deepseek_v41.ModelArgs(**{**SMALL, **ENGRAM, **overrides})
    model = deepseek_v41.Model(args)
    model.model.engram_token_map = mx.random.randint(
        0, args.engram_compressed_vocab_size, (args.vocab_size,)
    ).astype(mx.int32)
    for layer in model.model.layers:
        if layer.has_engram:
            table = layer.engram.embed
            rows = table.weight.shape[0]
            table.weight, table.scales = mx.quantize(
                mx.random.normal((rows, args.engram_head_dim)), mode="mxfp8"
            )
    return model


from mlx_lm.models.rope_utils import YarnRoPE


class TestDeepseekV41(unittest.TestCase):
    def test_rope_inverse(self):
        rope = YarnRoPE(
            dims=4,
            traditional=True,
            base=160000.0,
            scaling_factor=2.0,
            original_max_position_embeddings=16,
            mscale=0.0,
            mscale_all_dim=0.0,
        )
        for shape in ((1, 5, 8), (1, 5, 2, 8)):
            x = mx.random.normal(shape)
            y = deepseek_v41._apply_rope(x, rope, rope_dim=4, offset=4, scale=2.0)
            y = deepseek_v41._apply_rope(
                y, rope, rope_dim=4, offset=4, scale=2.0, inverse=True
            )
            self.assertTrue(mx.allclose(y, x, atol=1e-5))

    def test_cached_matches_forward(self):
        args = deepseek_v41.ModelArgs(**SMALL)
        mx.random.seed(0)
        model = deepseek_v41.Model(args)
        tokens = mx.random.randint(0, args.vocab_size, (2, 16))
        full = model(tokens)

        cache = model.make_cache()
        steps = []
        for start, end in ((0, 7), (7, 15), (15, 16)):
            steps.append(model(tokens[:, start:end], cache=cache))
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4)
        )

    def test_index_blocks_match(self):
        # Scoring the queries in blocks selects the same entries.
        args = deepseek_v41.ModelArgs(**SMALL)
        mx.random.seed(0)
        model = deepseek_v41.Model(args)
        tokens = mx.random.randint(0, args.vocab_size, (2, 24))

        def run():
            cache = model.make_cache()
            out = [model(tokens[:, s:e], cache=cache) for s, e in ((0, 11), (11, 24))]
            return mx.concatenate(out, axis=1)

        whole = run()
        # One query per block: 2 sequences x 4 heads x at most 24 keys x 4 bytes.
        with patch.object(deepseek_v41, "INDEX_BLOCK_BYTES", 2 * 4 * 24 * 4):
            blocked = run()
        self.assertTrue(mx.array_equal(whole, blocked).item())

    def test_engram_cached_matches_forward(self):
        mx.random.seed(0)
        model = engram_model()
        tokens = mx.random.randint(0, 128, (2, 16))
        full = model(tokens)

        cache = model.make_cache()
        steps = []
        for start, end in ((0, 1), (1, 7), (7, 15), (15, 16)):
            steps.append(model(tokens[:, start:end], cache=cache))
        self.assertTrue(
            mx.allclose(mx.concatenate(steps, axis=1), full, rtol=1e-4, atol=1e-4)
        )

    def test_engram_rows_match_reference(self):
        # The rows that DeepSeek's reference engram.py gives for this config.
        model = engram_model()
        model.model.engram_token_map = mx.arange(128, dtype=mx.int32)
        ngrams = model.model._ngrams(
            mx.array([[5, 7, 5, 7, 39]]), model.make_cache()[0]
        )
        layers = model.model.layers
        self.assertEqual(
            layers[1].engram._rows(ngrams)[0].tolist(),
            [
                [1, 53, 165, 198],
                [49, 54, 128, 190],
                [23, 86, 168, 201],
                [49, 54, 123, 221],
                [0, 106, 156, 190],
            ],
        )
        self.assertEqual(
            layers[3].engram._rows(ngrams)[0].tolist(),
            [
                [13, 122, 195, 288],
                [11, 128, 166, 304],
                [17, 117, 180, 244],
                [11, 128, 219, 244],
                [18, 76, 165, 275],
            ],
        )

    def test_engram_table_size_checked(self):
        with self.assertRaises(ValueError):
            deepseek_v41.Model(
                deepseek_v41.ModelArgs(
                    **{**SMALL, **ENGRAM, "engram_num_embeddings": (240, 300)}
                )
            )

    def test_compressed_token_map(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        vocab = ["The", "the", " the", "ÉTÉ", "été", " ", "\t", "<pad>"]
        tokenizer = Tokenizer(WordLevel({t: i for i, t in enumerate(vocab)}, "<pad>"))
        args = deepseek_v41.ModelArgs(
            vocab_size=len(vocab), engram_compressed_vocab_size=4
        )
        with tempfile.TemporaryDirectory() as path:
            tokenizer.save(str(Path(path) / "tokenizer.json"))
            token_map = deepseek_v41._compressed_token_map(
                Path(path) / "tokenizer.json", args
            )
            self.assertEqual(token_map.tolist(), [0, 0, 0, 1, 1, 2, 2, 3])
            with self.assertRaises(ValueError):
                deepseek_v41._compressed_token_map(
                    Path(path) / "tokenizer.json",
                    deepseek_v41.ModelArgs(
                        vocab_size=len(vocab), engram_compressed_vocab_size=5
                    ),
                )

    def test_sanitize_keeps_engram_packed(self):
        model = engram_model()
        table = mx.random.normal((240, 32))
        weights = {
            "layers.1.engram.embed.weight": mx.to_fp8(table),
            "layers.1.engram.embed.scale": mx.full((240, 1), 127, dtype=mx.uint8),
            "layers.1.engram.wkv.weight": mx.to_fp8(mx.random.normal((320, 128))),
            "layers.1.engram.wkv.scale": mx.full((10, 4), 127, dtype=mx.uint8),
            "layers.1.engram.q_weight": mx.ones((4, 64)),
            "layers.1.engram.k_weight": mx.ones((4, 64)),
        }
        clean = model.sanitize(weights)
        prefix = "model.layers.1.engram."
        self.assertEqual(
            sorted(clean),
            sorted(
                prefix + k
                for k in (
                    "embed.weight",
                    "embed.scales",
                    "kv_proj.weight",
                    "kv_proj.scales",
                    "q_weight",
                    "k_weight",
                )
            ),
        )
        self.assertEqual(clean[prefix + "kv_proj.scales"].shape, (320, 4))
        embed = model.model.layers[1].engram.embed
        embed.load_weights(
            [(k, clean[prefix + "embed." + k]) for k in ("weight", "scales")]
        )
        rows = mx.array([0, 17, 239])
        expected = mx.from_fp8(mx.to_fp8(table), mx.float32)[rows]
        self.assertTrue(mx.allclose(embed(rows).astype(mx.float32), expected))

    def test_engram_table_past_quantize_limit(self):
        # Real tables have more than 2**31 groups of 32, which mx.quantize rejects.
        table = deepseek_v41.EngramTable(384_016_682, 256)
        self.assertEqual(table.weight.shape, (384_016_682, 64))
        self.assertEqual(table.scales.shape, (384_016_682, 8))

    def test_tokenizer_weights(self):
        model = engram_model()
        key = "model.engram_token_map"
        self.assertEqual(model.tokenizer_weights(Path("/missing"), {key: 0}), {})
        plain = deepseek_v41.Model(deepseek_v41.ModelArgs(**SMALL))
        self.assertEqual(plain.tokenizer_weights(Path("/missing"), {}), {})


class TestDeepseekV41ChatTemplate(unittest.TestCase):
    """Prompts as DeepSeek's encoding/encoding.py renders them."""

    def setUp(self):
        from mlx_lm.chat_templates import deepseek_v41

        self.t = deepseek_v41
        self.messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]

    def test_thinking(self):
        out = self.t.apply_chat_template(self.messages, add_generation_prompt=True)
        self.assertEqual(
            out,
            "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, "
            "the higher the value, the more thorough the reasoning)\n\nBe brief."
            "<｜User｜>Hi<｜Assistant｜><think>",
        )

    def test_chat(self):
        out = self.t.apply_chat_template(
            self.messages, add_generation_prompt=True, enable_thinking=False
        )
        self.assertEqual(
            out,
            "<｜begin▁of▁sentence｜><｜System｜>Be brief.<｜User｜>Hi"
            "<｜Assistant｜></think>",
        )

    def test_no_generation_prompt(self):
        out = self.t.apply_chat_template(self.messages, enable_thinking=False)
        self.assertTrue(out.endswith("<｜User｜>Hi"))

    def test_history_drops_old_thinking(self):
        messages = self.messages + [
            {"role": "assistant", "content": "Hello.", "reasoning_content": "greet"},
            {"role": "user", "content": "Bye"},
        ]
        out = self.t.apply_chat_template(messages, add_generation_prompt=True)
        self.assertIn("<｜Assistant｜></think>Hello.<｜end▁of▁sentence｜>", out)
        self.assertNotIn("greet", out)

    def test_tools_and_results(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "now", "parameters": {"type": "object"}},
            }
        ]
        messages = self.messages + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "function": {"name": "now", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "12:00"},
        ]
        out = self.t.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True
        )
        self.assertIn('"name": "now"', out)
        self.assertIn('<｜DSML｜ invoke name="now">', out)
        self.assertTrue(
            out.endswith("<tool_result>12:00</tool_result><｜Assistant｜><think>")
        )

    def test_model_declares_template(self):
        from mlx_lm.models import deepseek_v41

        self.assertEqual(
            deepseek_v41.TOKENIZER_DEFAULTS["chat_template_type"], "deepseek_v41"
        )


if __name__ == "__main__":
    unittest.main()
