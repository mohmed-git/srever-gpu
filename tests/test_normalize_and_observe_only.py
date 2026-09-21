import unittest
from app.mt import (
    _normalize_ar,
    has_1p_ar,
    has_1p_en,
    detect_person_mismatch,
    check_needs_retry,
    _apply_output_guards,
    _is_conversational_chatter,
)


class TestNormalizeAndObserveOnly(unittest.TestCase):
    def test_normalize_ar_cases(self):
        # 1. Hamza normalization
        self.assertEqual(_normalize_ar("أنا إنسان آكل أرز"), "انا انسان اكل ارز")
        # 2. Ta-marbuta and alif-maqsura
        self.assertEqual(_normalize_ar("مدرسة ومستشفى"), "مدرسه ومستشفي")
        # 3. Tashkeel and tatweel
        self.assertEqual(_normalize_ar("الـحَمْدُ لِلَّـهِ"), "الحمد لله")

    def test_has_1p_ar_unhamzaed_spoken_arabic(self):
        # Whisper-normalized spoken Arabic without hamza
        self.assertTrue(has_1p_ar("وانا لا احب الطعام الصحي"))
        self.assertTrue(has_1p_ar("انا رايح الحين"))
        self.assertTrue(has_1p_ar("اعتقد انه تمام"))
        self.assertTrue(has_1p_ar("ودنا نساعدك"))
        self.assertTrue(has_1p_ar("قلت له كل شيء"))

        # Formal Arabic with hamza
        self.assertTrue(has_1p_ar("وأنا لا أحب هذا"))
        self.assertTrue(has_1p_ar("أنا بخير"))
        self.assertTrue(has_1p_ar("أعتقد ذلك"))

        # Neutral 3rd-person statements (Must NOT match 1p)
        self.assertFalse(has_1p_ar("ذهب إلى المدرسة"))
        self.assertFalse(has_1p_ar("السيارة سريعة"))
        self.assertFalse(has_1p_ar("توقف عن الكلام"))
        self.assertFalse(has_1p_ar("أحمق"))
        self.assertFalse(has_1p_ar("شكراً لك"))

    def test_exact_production_log_sentence_zero_retry(self):
        """Negative test mandated by Chief Architect ruling §4 item 1:
        The exact log sentence from 14:45:10 must produce ZERO retries and emit cleanly on the first pass.
        """
        src = "مرحبا اسمي محمد وانا لا احب الطعام الصحي ولا احب الاكل بشكل عام"
        primary_tgt = "Hello, my name is Mohammad and I don't like healthy food, and I don't like eating in general."

        # Verify 1p detection matches on both sides
        self.assertTrue(has_1p_ar(src), "Expected Whisper transcript to match 1p via _normalize_ar")
        self.assertTrue(has_1p_en(primary_tgt), "Expected English output to match 1p")

        # Must NOT be detected as mismatch
        self.assertFalse(
            detect_person_mismatch(src, primary_tgt, "ar", "en"),
            "Expected 1p match (no mismatch)",
        )

        # check_needs_retry MUST return False (zero retry)
        needs_retry, retry_reason = check_needs_retry(src, primary_tgt, "ar", "en")
        self.assertFalse(needs_retry, f"Expected NO retry, got {retry_reason}")

        # _apply_output_guards MUST NOT hollow or retry
        out_text, hollow, reason, soft_flags = _apply_output_guards(src, primary_tgt, "ar", "en", retried=False)
        self.assertFalse(hollow, f"Expected output to NOT be hollowed, got {reason}")
        self.assertEqual(out_text, primary_tgt)

    def test_ar_en_omission_is_observe_only_not_retry(self):
        """Verify ar->en person mismatch remains observe-only and never triggers a pipeline retry."""
        src = "أنا بخير"
        tgt_omitted = "Fine."

        # Detector identifies mismatch for telemetry
        self.assertTrue(detect_person_mismatch(src, tgt_omitted, "ar", "en"))

        # Pipeline retry check MUST NOT trigger for ar->en
        needs_retry, _ = check_needs_retry(src, tgt_omitted, "ar", "en")
        self.assertFalse(needs_retry, "Expected ar->en mismatch to be observe-only (NO retry)")

        # Output guard MUST emit the translation without hollow/retry
        out_text, hollow, reason, _ = _apply_output_guards(src, tgt_omitted, "ar", "en", retried=False)
        self.assertFalse(hollow, "Expected ar->en mismatch to NOT hollow the translation")
        self.assertEqual(out_text, tgt_omitted)

    def test_en_ar_injection_still_triggers_retry(self):
        """Ensure en->ar chat leaks (where English has no 1p, but Arabic injects 'أنا') still retry."""
        src_en = "The sky is blue."
        tgt_ar_leak = "أنا أعتقد أن السماء زرقاء."

        self.assertTrue(detect_person_mismatch(src_en, tgt_ar_leak, "en", "ar"))
        needs_retry, reason = check_needs_retry(src_en, tgt_ar_leak, "en", "ar")
        self.assertTrue(needs_retry)
        self.assertEqual(reason, "person_mismatch")


if __name__ == "__main__":
    unittest.main()
