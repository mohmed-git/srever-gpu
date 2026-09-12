import unittest
from app.mt import has_1p_ar, detect_person_mismatch

class TestPersonMismatchFixed(unittest.TestCase):
    def test_negative_cases(self):
        # The words should NOT trigger 1st person
        self.assertFalse(has_1p_ar("أحمق"))
        self.assertFalse(has_1p_ar("انتظروني"))
        self.assertFalse(has_1p_ar("شكراً لك"))
        self.assertFalse(has_1p_ar("كتابي"))
        
        # Test detection logic
        self.assertFalse(detect_person_mismatch("Idiot", "أحمق", "en", "ar"))
        self.assertFalse(detect_person_mismatch("Wait for me", "انتظروني", "en", "ar"))

if __name__ == '__main__':
    unittest.main()
