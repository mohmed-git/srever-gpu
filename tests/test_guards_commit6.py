import unittest

from app.mt import (
    has_1p_ar,
    detect_person_mismatch,
    is_degenerate_short,
    has_annotation_or_passthrough,
)

class TestCommit6Guards(unittest.TestCase):
    def test_minimum_output_guard_single_letter(self):
        # Stop. -> أ. must be flagged as degenerate short
        self.assertTrue(is_degenerate_short("أ.", "ar"))
        self.assertTrue(is_degenerate_short("و.", "ar"))
        # Real short words (2+ letters) must NOT be flagged
        self.assertFalse(is_degenerate_short("لا.", "ar"))
        self.assertFalse(is_degenerate_short("قف.", "ar"))
        self.assertFalse(is_degenerate_short("نعم", "ar"))

    def test_annotation_and_passthrough_guard(self):
        # But if (French: Mais si) must be flagged
        self.assertTrue(has_annotation_or_passthrough("But if", "But if (French: Mais si)"))
        self.assertTrue(has_annotation_or_passthrough("hello", "(Arabic: مرحبا)"))
        self.assertTrue(has_annotation_or_passthrough("Stop translation", "Stop translation (en: stop)"))
        # Normal translations must NOT be flagged
        self.assertFalse(has_annotation_or_passthrough("But if", "Mais si"))
        self.assertFalse(has_annotation_or_passthrough("hello", "مرحبا"))

    def test_arabic_suffix_past_first_person(self):
        # Stupid. -> بكيت. must be flagged as mismatch (1p suffix on short verb)
        self.assertTrue(has_1p_ar("بكيت."))
        self.assertTrue(has_1p_ar("ذهبت"))
        self.assertTrue(has_1p_ar("فعلت"))
        self.assertTrue(detect_person_mismatch("Stupid.", "بكيت.", "en", "ar"))

        # Negatives: non-1p nouns/pronouns ending in taa must NOT be flagged as 1p
        self.assertFalse(has_1p_ar("بيت"))
        self.assertFalse(has_1p_ar("وقت"))
        self.assertFalse(has_1p_ar("أنتِ"))
        self.assertFalse(has_1p_ar("أنت"))
        self.assertFalse(has_1p_ar("صوت"))
        self.assertFalse(has_1p_ar("بنت"))
        self.assertFalse(detect_person_mismatch("the house", "بيت", "en", "ar"))
        self.assertFalse(detect_person_mismatch("the time", "وقت", "en", "ar"))

if __name__ == "__main__":
    unittest.main()
