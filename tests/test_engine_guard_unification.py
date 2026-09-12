import unittest
from app.mt import _apply_output_guards, check_needs_retry

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

if __name__ == "__main__":
    unittest.main()
