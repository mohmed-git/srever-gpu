import unittest
from app.mt import has_1p_fr, detect_person_mismatch

class TestPersonDetectorFr(unittest.TestCase):
    def test_positives(self):
        # Must detect first person pronouns/adjectives in French
        self.assertTrue(has_1p_fr("Je suis ici"))
        self.assertTrue(has_1p_fr("j'ai faim"))
        self.assertTrue(has_1p_fr("Aide-moi"))
        self.assertTrue(has_1p_fr("Laisse-moi tranquille."))
        self.assertTrue(has_1p_fr("C'est mon ami"))
        self.assertTrue(has_1p_fr("C'est ma maison"))
        self.assertTrue(has_1p_fr("Ce sont mes affaires"))
        self.assertTrue(has_1p_fr("Nous venons"))
        self.assertTrue(has_1p_fr("Notre projet"))
        self.assertTrue(has_1p_fr("Nos enfants"))

        # en -> fr person injection
        self.assertTrue(detect_person_mismatch("shut up", "Laisse-moi tranquille.", "en", "fr"))

    def test_negatives(self):
        # Must NOT flag sentences without first person
        self.assertFalse(has_1p_fr("Bonjour"))
        self.assertFalse(has_1p_fr("Arrêtez la traduction."))
        self.assertFalse(has_1p_fr("Qu'est-ce que vous faites?"))
        self.assertFalse(has_1p_fr("Mais si"))

    def test_symmetric_source_checking(self):
        # When source already has 1st person, target with 1st person is NOT a mismatch!
        # Arabic: كيف أساعدك (أساعد is 1p) -> French: Comment puis-je t'aider? (puis-je is 1p)
        self.assertFalse(detect_person_mismatch("كيف أساعدك", "Comment puis-je t'aider?", "ar", "fr"))
        # English: I need help -> French: J'ai besoin d'aide
        self.assertFalse(detect_person_mismatch("I need help", "J'ai besoin d'aide", "en", "fr"))

if __name__ == "__main__":
    unittest.main()
