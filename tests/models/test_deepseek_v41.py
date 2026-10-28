# Copyright © 2026 Apple Inc.

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import deepseek_v41
from mlx_lm.utils import infer_quant_config

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
        params = infer_quant_config(prefix + "embed", embed, clean)
        self.assertEqual(params["mode"], "mxfp8")
        quantized = nn.QuantizedEmbedding(240, 32, **params)
        quantized.weight = clean[prefix + "embed.weight"]
        quantized.scales = clean[prefix + "embed.scales"]
        rows = mx.array([0, 17, 239])
        expected = mx.from_fp8(mx.to_fp8(table), mx.float32)[rows]
        self.assertTrue(mx.allclose(quantized(rows).astype(mx.float32), expected))

    def test_tokenizer_weights(self):
        model = engram_model()
        key = "model.engram_token_map"
        self.assertEqual(model.tokenizer_weights(Path("/missing"), {key: 0}), {})
        plain = deepseek_v41.Model(deepseek_v41.ModelArgs(**SMALL))
        self.assertEqual(plain.tokenizer_weights(Path("/missing"), {}), {})


if __name__ == "__main__":
    unittest.main()
