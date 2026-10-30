# Copyright © 2026 Apple Inc.

"""Requests cancelled on a distributed server without batching.

Three local ranks run a pipelined model on the CPU and the server's single
request path. Rank 0 cancels one request while the prompt is processed and one
while tokens are generated; a request after them must still complete. Before
the ranks stopped together, rank 0 alone left the request and the ranks hung.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import mlx.core as mx

mx.set_default_device(mx.cpu)

TIMEOUT = 180
TOKENIZER_REPO = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


def cached_tokenizer():
    """The cached snapshot that holds the test tokenizer, or None."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(TOKENIZER_REPO, "tokenizer.json")
    return Path(path).parent if isinstance(path, str) else None


def worker(out_dir):
    from mlx_lm.models import deepseek_v3
    from mlx_lm.server import (
        CompletionRequest,
        GenerationArguments,
        LogitsProcessorArguments,
        LRUPromptCache,
        ModelDescription,
        ResponseGenerator,
        SamplingArguments,
    )
    from mlx_lm.utils import load_tokenizer

    world = mx.distributed.init()
    rank = world.rank()
    tokenizer = load_tokenizer(cached_tokenizer())
    args = deepseek_v3.ModelArgs(
        vocab_size=151936,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=8,
        qk_nope_head_dim=8,
    )
    mx.random.seed(0)
    model = deepseek_v3.Model(args)
    mx.eval(model.parameters())
    model.model.pipeline(world)

    class Provider:
        def __init__(self):
            self.model, self.tokenizer = model, tokenizer
            self.draft_model = None
            self.processor = None
            self.model_key = ("tiny", None)
            self.is_batchable = False
            self.cli_args = type(
                "cli_args",
                (),
                {
                    "adapter_path": None,
                    "chat_template": None,
                    "use_default_chat_template": False,
                    "trust_remote_code": False,
                    "draft_model": None,
                    "num_draft_tokens": 3,
                    "max_tokens": 64,
                    "chat_template_args": {},
                    "model": None,
                    "decode_concurrency": 1,
                    "prompt_concurrency": 1,
                    "prefill_step_size": 16,
                    "prompt_cache_size": 4,
                    "prompt_cache_bytes": 1 << 63,
                    "prompt_cache_total_bytes": None,
                    "allowed_origins": ["*"],
                    "kv_bits": None,
                    "kv_group_size": 64,
                    "quantized_kv_start": 0,
                },
            )

        def load(self, model, adapter=None, draft_model=None):
            return self.model, self.tokenizer

        def load_default(self):
            return self.load("default_model")

        def reset(self):
            pass

    generator = ResponseGenerator(Provider(), LRUPromptCache())
    done = Path(out_dir) / "done"
    if rank != 0:
        while not done.exists():
            time.sleep(0.1)
        os._exit(0)

    def arguments(max_tokens):
        return GenerationArguments(
            model=ModelDescription("default_model", None, None),
            sampling=SamplingArguments(0.0, 1.0, 0, 0.0, 0.0, 0.0),
            logits=LogitsProcessorArguments(None, 1.0, 20, 0.0, 20, 0.0, 20),
            stop_words=[],
            max_tokens=max_tokens,
            num_draft_tokens=3,
            logprobs=False,
            top_logprobs=0,
            seed=None,
            chat_template_kwargs=None,
        )

    results = {}
    long_prompt = "one two three four five six seven eight " * 40

    # Cancelled while the prompt is processed.
    held = {}
    ctx, responses = generator.generate(
        CompletionRequest("text", long_prompt, [], None, None),
        arguments(64),
        progress_callback=lambda done_, total: done_ > 0 and held["ctx"].stop(),
    )
    held["ctx"] = ctx
    results["prefill_tokens"] = len(list(responses))

    # Cancelled while tokens are generated.
    ctx, responses = generator.generate(
        CompletionRequest("text", "count: one two", [], None, None), arguments(64)
    )
    tokens = 0
    for _ in responses:
        tokens += 1
        if tokens == 3:
            ctx.stop()
    results["decode_tokens"] = tokens

    # A request after both completes.
    _, responses = generator.generate(
        CompletionRequest("text", "hello", [], None, None), arguments(8)
    )
    results["after_tokens"] = len(list(responses))
    (Path(out_dir) / "result.json").write_text(json.dumps(results))
    done.touch()
    os._exit(0)


class TestDistributedCancellation(unittest.TestCase):
    def test_cancelled_requests_do_not_hang_the_ranks(self):
        launcher = shutil.which("mlx.launch") or str(
            Path(sys.executable).with_name("mlx.launch")
        )
        if not os.path.exists(launcher):
            self.skipTest("mlx.launch is not installed")
        if cached_tokenizer() is None:
            self.skipTest("the test tokenizer is not in the local Hugging Face cache")
        with tempfile.TemporaryDirectory() as out_dir:
            env = {**os.environ, "MLX_TEST_CANCEL_WORKER": out_dir}
            cmd = [launcher, "-n", "3", "--backend", "ring"]
            cmd += ["--python", sys.executable, __file__]
            try:
                subprocess.run(
                    cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT
                )
            except subprocess.TimeoutExpired:
                self.fail("the ranks hung after a cancelled request")
            result = Path(out_dir) / "result.json"
            self.assertTrue(result.exists(), "rank 0 did not finish")
            results = json.loads(result.read_text())
        self.assertLess(results["decode_tokens"], 64)
        self.assertGreater(results["after_tokens"], 0)


if __name__ == "__main__":
    if "MLX_TEST_CANCEL_WORKER" in os.environ:
        worker(os.environ["MLX_TEST_CANCEL_WORKER"])
    else:
        unittest.main()
