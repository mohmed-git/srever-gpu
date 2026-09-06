"""Unit test verifying that QwenCt2Engine prompt splitting exactly matches
the official Qwen2.5 chat template reference, without injecting duplicate system prompts.
Runs purely with AutoTokenizer (no GPU/model required).
"""

import unittest
from transformers import AutoTokenizer

from app.mt import _TURN_FMT, make_translation_messages


class TestQwenCt2PromptSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    def _split_prompt(self, text: str, src: str, dst: str) -> tuple[str, list[str], list[str]]:
        msgs = make_translation_messages(text, src, dst)
        self.assertEqual(msgs[-1]["role"], "user", "prompt builder must end with the user turn")
        key = f"{src}->{dst}"
        prefix_text = self.tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=False
        )
        static = self.tokenizer.tokenize(prefix_text)
        per_request = self.tokenizer.tokenize(_TURN_FMT.format(content=msgs[-1]["content"]))
        return key, static, per_request

    def test_prompt_split_matches_reference(self):
        test_pairs = [
            ("en", "ar", "Hello, where is the nearest hospital?"),
            ("ar", "en", "مرحبا، أين أقرب مستشفى من فضلك؟"),
            ("en", "tr", "Good morning, how are you today?"),
            ("zh", "ar", "早上好，你今天好吗？"),
        ]

        for src, dst, text in test_pairs:
            with self.subTest(pair=f"{src}->{dst}"):
                _, static, per_req = self._split_prompt(text, src, dst)

                ref_text = self.tokenizer.apply_chat_template(
                    make_translation_messages(text, src, dst),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                ref_tokens = self.tokenizer.tokenize(ref_text)

                # 1. Assembled tokens must match reference exactly
                self.assertEqual(
                    static + per_req,
                    ref_tokens,
                    f"static + per_request diverges from reference for {src}->{dst}",
                )

                # 2. System prompt must appear EXACTLY once
                system_turn_count = ref_text.count("<|im_start|>system")
                self.assertEqual(
                    system_turn_count,
                    1,
                    f"Reference prompt contains {system_turn_count} system turns, expected exactly 1",
                )

                # 3. Per-request turn must start with user turn, not a rogue system turn
                self.assertEqual(per_req[0], "<|im_start|>")
                self.assertEqual(per_req[1], "user")


class TestQwenCt2WarmupGate(unittest.TestCase):
    def test_warmup_failure_without_env_var_sets_error_and_not_ready(self):
        import os
        from app.config import Settings
        from app.mt import QwenCt2Engine

        old_env = os.environ.pop("MT_ALLOW_DEGRADED_FALLBACK", None)
        try:
            settings = Settings()
            engine = QwenCt2Engine(settings)
            engine._generator = object()
            engine._tokenizer = object()
            self.assertTrue(engine.ready)

            engine._handle_warmup_failure("target script ratio 10.0% < 60% in en->ar: 'test'")

            self.assertFalse(engine.ready)
            self.assertIsNotNone(engine.error)
            self.assertIn("degenerate", engine.error)
            self.assertIsNone(engine._fallback_engine)
        finally:
            if old_env is not None:
                os.environ["MT_ALLOW_DEGRADED_FALLBACK"] = old_env


if __name__ == "__main__":
    unittest.main()
