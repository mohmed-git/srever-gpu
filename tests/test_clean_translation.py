import unittest
from app.mt import clean_translation

class TestCleanTranslationParentheses(unittest.TestCase):
    def test_strip_enclosing_parentheses(self):
        # Must strip enclosing parens
        self.assertEqual(clean_translation("(stop talking)"), "stop talking")
        self.assertEqual(clean_translation("(Bonjour)"), "Bonjour")
        self.assertEqual(clean_translation(" (مرحبا) "), "مرحبا")
        # Must NOT strip internal parens
        self.assertEqual(clean_translation("hello (world) test"), "hello (world) test")
        self.assertEqual(clean_translation("(hello) (world)"), "(hello) (world)")

if __name__ == "__main__":
    unittest.main()
