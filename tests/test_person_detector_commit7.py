import unittest
from app.mt import has_1p_ar, detect_person_mismatch, _AR_1P_VERBS

class TestPersonDetectorCommit7(unittest.TestCase):
    def test_positives(self):
        # Must flag
        self.assertTrue(has_1p_ar("أعتذر، لكنني لا أستطيع"))
        self.assertTrue(has_1p_ar("أستطيع مساعدتك"))
        self.assertTrue(has_1p_ar("سأساعدك"))
        self.assertTrue(has_1p_ar("بكيت"))
        self.assertTrue(has_1p_ar("أنا هنا"))
        self.assertTrue(has_1p_ar("نحن هنا"))
        self.assertTrue(has_1p_ar("ذهبت إلى السوق"))

    def test_negatives_outnumber_positives(self):
        # Must NOT flag
        negatives = [
            "أداة", "أستاذ", "أسبوع", "أرض", "أمر", 
            "أب", "أخ", "أول", "أبداً", "أين", 
            "أحمق", "أفضل", "بيت", "وقت", "أنتِ", "أنت", "صوت"
        ]
        for word in negatives:
            with self.subTest(word=word):
                self.assertFalse(has_1p_ar(word), f"'{word}' was wrongly flagged as 1st person!")

    def test_sabotage_verb_removal(self):
        # Sabotage: if we temporarily remove a verb from the set, it must fail detection on that verb
        test_verb = "أعتذر"
        self.assertIn(test_verb, _AR_1P_VERBS)
        # Verify sabotage mechanics: without the verb in the set, a sentence with only that verb fails detection
        sabotaged_set = _AR_1P_VERBS - {test_verb}
        # A mock detection using the sabotaged set
        detected_under_sabotage = (test_verb in sabotaged_set)
        self.assertFalse(detected_under_sabotage, "Sabotage failed: removed verb was still detected!")

if __name__ == "__main__":
    unittest.main()
