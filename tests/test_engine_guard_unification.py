import unittest
import sys
from unittest.mock import MagicMock
if "vllm" not in sys.modules:
    sys.modules["vllm"] = MagicMock()
from app.config import Settings
from app.mt import (
    _apply_output_guards,
    check_needs_retry,
    QwenCt2Engine,
    QwenVllmEngine,
    QwenHfEngine,
    M2M100Ct2Engine,
)

class TestEngineGuardUnification(unittest.TestCase):
    def test_check_needs_retry(self):
        # 1. Normal translation: no retry needed
        needs_retry, reason = check_needs_retry("The sky is blue.", "السماء زرقاء.", "en", "ar")
        self.assertFalse(needs_retry)
        self.assertIsNone(reason)

        # 2. Degenerate short: retry needed
        needs_retry, reason = check_needs_retry("Stop.", "أ.", "en", "ar")
        self.assertTrue(needs_retry)
        self.assertEqual(reason, "degenerate_short")

        # 3. Chatter pattern: retry needed
        needs_retry, reason = check_needs_retry("Hello", "Comment puis-je vous aider?", "en", "fr")
        self.assertTrue(needs_retry)
        self.assertEqual(reason, "conversational_chatter")

        # 4. Person mismatch (en->ar): retry needed
        needs_retry, reason = check_needs_retry("Stupid.", "بكيت.", "en", "ar")
        self.assertTrue(needs_retry)
        self.assertEqual(reason, "person_mismatch")

        # 5. Annotation leak: retry needed
        needs_retry, reason = check_needs_retry("But if", "But if (French: Mais si)", "en", "fr")
        self.assertTrue(needs_retry)
        self.assertEqual(reason, "annotation_leak")

    def test_apply_output_guards_unification(self):
        # 1. Clean translation passes
        text, hollow, reason, soft_flags = _apply_output_guards(
            "The sky is blue.", "السماء زرقاء.", "en", "ar", retried=False
        )
        self.assertEqual(text, "السماء زرقاء.")
        self.assertFalse(hollow)
        self.assertIsNone(reason)
        self.assertEqual(soft_flags, {})

        # 2. Post-retry person mismatch alone is soft-downgraded (emitted, NOT hollow)
        text, hollow, reason, soft_flags = _apply_output_guards(
            "Stupid.", "بكيت.", "en", "ar", retried=True, primary_decoded="بكيش."
        )
        self.assertEqual(text, "بكيت.")
        self.assertFalse(hollow)
        self.assertTrue(soft_flags.get("person_mismatch_soft"))

        # 3. Post-retry degenerate short IS hollowed and carries primary & retry in reason
        text, hollow, reason, soft_flags = _apply_output_guards(
            "Stop.", "أ.", "en", "ar", retried=True, primary_decoded="توقف."
        )
        self.assertEqual(text, "")
        self.assertTrue(hollow)
        self.assertIn("degenerate_short", reason)
        self.assertIn("primary='توقف.'", reason)
        self.assertIn("retry='أ.'", reason)

        # 4. Post-retry script ratio failure IS hollowed and carries primary & retry in reason
        text, hollow, reason, soft_flags = _apply_output_guards(
            "Stop.", "STOP.", "en", "ar", retried=True, primary_decoded="توقف."
        )
        self.assertEqual(text, "")
        self.assertTrue(hollow)
        self.assertIn("target script ratio", reason)
        self.assertIn("primary='توقف.'", reason)
        self.assertIn("retry='STOP.'", reason)

        # 5. Post-retry annotation leak IS hollowed and carries primary & retry in reason
        text, hollow, reason, soft_flags = _apply_output_guards(
            "But if", "But if (French: Mais si)", "en", "fr", retried=True, primary_decoded="Mais si"
        )
        self.assertEqual(text, "")
        self.assertTrue(hollow)
        self.assertIn("annotation_leak", reason)
        self.assertIn("primary='Mais si'", reason)
        self.assertIn("retry='But if (French: Mais si)'", reason)

    def test_fake_generator_identical_guards_across_all_engines(self):
        settings = Settings()
        test_cases = [
            ("The sky is blue.", "السماء زرقاء.", "en", "ar"),
            ("Stop.", "STOP.", "en", "ar"),
            ("Stop.", "أ.", "en", "ar"),
            ("Stupid.", "بألف خير.", "en", "ar"),
            ("But if", "But if (French: Mais si)", "en", "fr"),
        ]

        for src_text, fixed_gen_output, src, dst in test_cases:
            # 1. QwenCt2Engine
            ct2 = QwenCt2Engine(settings)
            ct2._tokenizer = MagicMock()
            ct2._tokenizer.decode.return_value = fixed_gen_output
            ct2._tokenizer.convert_tokens_to_string.return_value = fixed_gen_output
            ct2._static_tokens = {'k': []}
            ct2._split_prompt = MagicMock(return_value=("k", [], []))
            mock_out = MagicMock()
            mock_out.sequences_ids = [[1, 2]]
            mock_out.sequences = [["tok"]]
            ct2._generator = MagicMock()
            ct2._generator.generate_batch.return_value = [mock_out]
            ct2._translate_single_retry = MagicMock(return_value=fixed_gen_output)
            res_ct2 = ct2.translate_batch([(src_text, src, dst)])[0]

            # 2. QwenVllmEngine
            vllm = QwenVllmEngine(settings)
            vllm._fallback = None
            vllm._tokenizer = MagicMock()
            vllm._tokenizer.encode.return_value = [1, 2]
            vllm._prompt = MagicMock(return_value="prompt")
            mock_vllm_out = MagicMock()
            mock_comp = MagicMock()
            mock_comp.text = fixed_gen_output
            mock_comp.token_ids = [1, 2]
            mock_vllm_out.outputs = [mock_comp]
            mock_vllm_out.prompt_token_ids = [1, 2, 3]
            vllm._llm = MagicMock()
            vllm._llm.generate.return_value = [mock_vllm_out]
            vllm._translate_single_retry = MagicMock(return_value=fixed_gen_output)
            res_vllm = vllm.translate_batch([(src_text, src, dst)])[0]

            # 3. QwenHfEngine
            hf = QwenHfEngine(settings)
            hf._tokenizer = MagicMock()
            hf._tokenizer.decode.return_value = fixed_gen_output
            hf._tokenizer.pad_token_id = 0
            hf._tokenizer.eos_token_id = 1
            import torch
            mock_encoded = MagicMock()
            mock_tensor = torch.zeros((1, 5), dtype=torch.long)
            mock_encoded.__getitem__.side_effect = lambda k: mock_tensor
            mock_encoded.to.return_value = mock_encoded
            hf._tokenizer.return_value = mock_encoded
            import torch
            hf._model = MagicMock()
            hf._model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
            res_hf = hf.translate_batch([(src_text, src, dst)])[0]

            # Assert identical (text, hollow, reason) between CT2 and VLLM
            self.assertEqual(
                (res_ct2.text, res_ct2.hollow, res_ct2.hollow_reason),
                (res_vllm.text, res_vllm.hollow, res_vllm.hollow_reason),
                f"Mismatch between CT2 and VLLM on input '{src_text}' -> '{fixed_gen_output}'"
            )

            # 4. M2M100Ct2Engine
            m2m = M2M100Ct2Engine(settings, "dummy_model")
            m2m._sp = MagicMock()
            m2m._sp.EncodeAsPieces.return_value = ["tok"]
            mock_trans = MagicMock()
            mock_res = MagicMock()
            mock_res.hypotheses = [["__ar__", fixed_gen_output]]
            mock_trans.translate_batch.return_value = [mock_res]
            m2m._translator = mock_trans
            res_m2m = m2m.translate_batch([(src_text, src, dst)])[0]

            self.assertEqual(
                (res_ct2.text, res_ct2.hollow),
                (res_m2m.text, res_m2m.hollow),
                f"Mismatch between CT2 and M2M100 on input '{src_text}' -> '{fixed_gen_output}'"
            )

            # Assert identical (text, hollow) on HF
            self.assertEqual(
                (res_ct2.text, res_ct2.hollow),
                (res_hf.text, res_hf.hollow),
                f"Mismatch between CT2 and HF on input '{src_text}' -> '{fixed_gen_output}'"
            )

if __name__ == "__main__":
    unittest.main()
