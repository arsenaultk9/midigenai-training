import unittest
from types import SimpleNamespace

import numpy as np

from generate_with_onnx import (
    continuation_prompt,
    generate_with_onnx,
    repeated_ngram_next_tokens,
)


class CacheSession:
    def __init__(self, output_prefix, cache_axis):
        self.output_prefix = output_prefix
        self.cache_axis = cache_axis
        self.cache_lengths = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in ("ids", "past_k0", "past_v0")]

    def get_outputs(self):
        names = ("logits", f"{self.output_prefix}_k0", f"{self.output_prefix}_v0")
        return [SimpleNamespace(name=name) for name in names]

    def run(self, output_names, inputs):
        past = inputs["past_k0"]
        self.cache_lengths.append(past.shape[self.cache_axis])
        np.testing.assert_array_equal(past, inputs["past_v0"])
        shape = list(past.shape)
        shape[self.cache_axis] = inputs["ids"].shape[1]
        rows = np.ones(shape, dtype=np.float32)
        cache = (
            rows if self.output_prefix == "new"
            else np.concatenate((past, rows), axis=self.cache_axis)
        )
        return [np.array([[0.0, 100.0]]), cache, cache]


class GenerateWithOnnxTests(unittest.TestCase):
    def test_continuation_prompt_adds_bos_once(self):
        tok = SimpleNamespace(vocab={"BOS_None": 1})
        self.assertEqual(continuation_prompt(tok, [70, 71]), [1, 70, 71])
        self.assertEqual(continuation_prompt(tok, [1, 70]), [1, 70])
        self.assertEqual(continuation_prompt(SimpleNamespace(vocab={}), [70]), [70])

    def test_end_token_stops_generation(self):
        session = CacheSession("new", 1)
        generated = generate_with_onnx(
            session, [3], {"n_layers": 1, "n_heads": 2, "head_dim": 3},
            4, eos_id=1,
        )
        self.assertEqual(generated, [])
        self.assertEqual(session.cache_lengths, [0])

    def test_invalid_continuation_tokens_are_not_sampled(self):
        session = CacheSession("new", 1)
        generated = generate_with_onnx(
            session, [3], {"n_layers": 1, "n_heads": 2, "head_dim": 3},
            2, ban_ids=[1],
        )
        self.assertEqual(generated, [0, 0])

    def test_ngram_ban_blocks_previous_sequence_not_unrelated_tokens(self):
        self.assertEqual(repeated_ngram_next_tokens([1, 2, 3, 1, 2, 4, 1, 2], 3), {3, 4})
        self.assertEqual(repeated_ngram_next_tokens([1], 3), set())
        session = CacheSession("new", 1)
        generated = generate_with_onnx(
            session, [1], {"n_layers": 1, "n_heads": 2, "head_dim": 3},
            3, no_repeat_ngram_size=2,
        )
        self.assertEqual(generated, [1, 0, 1])

    def test_cache_grows_with_prompt_and_each_generated_token(self):
        config = {"n_layers": 1, "n_heads": 2, "head_dim": 3}
        for prefix, axis in (("new", 1), ("present", 2)):
            with self.subTest(prefix=prefix):
                session = CacheSession(prefix, axis)
                generated = generate_with_onnx(session, [3, 4, 5], config, 4)
                self.assertEqual(generated, [1, 1, 1, 1])
                self.assertEqual(session.cache_lengths, [0, 3, 4, 5])

    def test_unrecognized_cache_outputs_fail_explicitly(self):
        session = CacheSession("unknown", 1)
        with self.assertRaisesRegex(ValueError, "unrecognized KV cache"):
            generate_with_onnx(
                session, [3], {"n_layers": 1, "n_heads": 2, "head_dim": 3}, 1
            )


if __name__ == "__main__":
    unittest.main()
