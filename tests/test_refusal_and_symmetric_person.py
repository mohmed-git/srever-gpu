import unittest
from app.mt import _is_conversational_chatter, detect_person_mismatch

class TestRefusalAndSymmetricPerson(unittest.TestCase):
    def test_refusal_patterns_positive(self):
        positives = [
            "I will not translate that.",
            "I will not translate this sentence.",
            "I won't translate that",
            "I cannot translate this input.",
            "I refuse to translate that.",
            "I am unable to assist with this.",
            "I refuse to do that.",
            "je ne peux pas traduire",
            "je ne vais pas traduire",
            "لن أترجم هذا النص",
            "لا يمكنني ترجمة ذلك",
        ]
        for text in positives:
            self.assertTrue(_is_conversational_chatter(text), f"Expected refusal to match: {text!r}")

    def test_refusal_patterns_negative(self):
        negatives = [
            "I will not go.",
            "I won't stop now.",
            "I cannot leave yet.",
            "I refuse to accept the offer.",
            "Stop talking.",
            "Stop translation.",
            "The sky is blue.",
            "Je vais partir demain.",
            "لن أذهب إلى السوق.",
            "لا أريد هذا الشيء.",
        ]
        for text in negatives:
            self.assertFalse(_is_conversational_chatter(text), f"Did not expect refusal to match: {text!r}")

    def test_ar_en_person_mismatch_symmetric(self):
        # 1. Injection (source has NO 1p, target INJECTS 1p)
        self.assertTrue(
            detect_person_mismatch("توقف عن الكلام", "I will not translate that.", "ar", "en"),
            "Expected ar->en 1p injection to be flagged"
        )
        self.assertTrue(
            detect_person_mismatch("ذهب إلى المدرسة", "I went to school.", "ar", "en"),
            "Expected ar->en 1p injection to be flagged"
        )

        # 2. Omission (source HAS 1p, target DROPS 1p)
        self.assertTrue(
            detect_person_mismatch("أنا بخير", "Fine.", "ar", "en"),
            "Expected ar->en 1p omission to be flagged"
        )

        # 3. Negatives (both have 1p, or neither has 1p)
        self.assertFalse(
            detect_person_mismatch("أنا ذاهب إلى العمل", "I am going to work.", "ar", "en"),
            "Expected ar->en matching 1p to pass"
        )
        self.assertFalse(
            detect_person_mismatch("نحن نعمل معاً", "We work together.", "ar", "en"),
            "Expected ar->en matching 1p plural to pass"
        )
        self.assertFalse(
            detect_person_mismatch("السماء زرقاء.", "The sky is blue.", "ar", "en"),
            "Expected ar->en neutral sentence to pass"
        )

if __name__ == "__main__":
    unittest.main()
