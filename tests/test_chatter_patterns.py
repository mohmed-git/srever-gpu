import unittest
from app.mt import _is_conversational_chatter

class TestChatterPatterns(unittest.TestCase):
    def test_french_positive(self):
        self.assertTrue(_is_conversational_chatter("Voici la traduction : Bonjour"))
        self.assertTrue(_is_conversational_chatter("Voilà la traduction"))
        self.assertTrue(_is_conversational_chatter("Je ne peux pas traduire cela."))
        self.assertTrue(_is_conversational_chatter("Comment puis-je vous aider ?"))
        self.assertTrue(_is_conversational_chatter("Je suis un traducteur AI."))
        
    def test_french_negative(self):
        self.assertFalse(_is_conversational_chatter("Il est là."))
        self.assertFalse(_is_conversational_chatter("Je ne sais pas."))
        self.assertFalse(_is_conversational_chatter("Comment ça va?"))
        
    def test_arabic_positive(self):
        self.assertTrue(_is_conversational_chatter("إليك الترجمة: مرحبا"))
        self.assertTrue(_is_conversational_chatter("هذه الترجمة للنص"))
        self.assertTrue(_is_conversational_chatter("عذراً، لا يمكنني ترجمة ذلك"))
        self.assertTrue(_is_conversational_chatter("كيف أستطيع مساعدتك اليوم؟"))
        self.assertTrue(_is_conversational_chatter("أنا مجرد مترجم آلي"))
        
    def test_arabic_negative(self):
        self.assertFalse(_is_conversational_chatter("إليك الكتاب الذي طلبته."))
        self.assertFalse(_is_conversational_chatter("هذه السيارة سريعة."))
        self.assertFalse(_is_conversational_chatter("لا يمكنني الذهاب معك."))
        self.assertFalse(_is_conversational_chatter("أنا طالب."))

if __name__ == '__main__':
    unittest.main()
