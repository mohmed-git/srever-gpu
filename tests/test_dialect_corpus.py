"""Tests for tests/dialect_corpus.jsonl and prompt construction over the corpus."""
import json
import unittest
from pathlib import Path

from app.mt import make_translation_messages, _DIALECT_NAMES


class TestDialectCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_path = Path(__file__).parent / "dialect_corpus.jsonl"
        cls.assertTrue(cls.corpus_path.exists(), "dialect_corpus.jsonl must exist")
        cls.items = []
        with open(cls.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cls.items.append(json.loads(line))

    def test_corpus_size_and_tiers(self):
        """Corpus must have 21 items with JO, SA, EG, IQ across easy, medium, and hard."""
        self.assertEqual(len(self.items), 21, "Benchmark must contain exactly 21 items")
        dialects = {it["dialect"] for it in self.items}
        self.assertEqual(dialects, {"JO", "SA", "EG", "IQ"})
        tiers = {it["tier"] for it in self.items}
        self.assertEqual(tiers, {"easy", "medium", "hard"})

    def test_corpus_schema(self):
        """Every item must have required fields and non-empty strings."""
        for it in self.items:
            self.assertIn("id", it)
            self.assertIn("dialect", it)
            self.assertIn("tier", it)
            self.assertIn("text", it)
            self.assertIn("expected", it)
            self.assertIn("category", it)
            self.assertTrue(len(it["text"].strip()) > 0)
            self.assertTrue(len(it["expected"].strip()) > 0)

    def test_prompt_generation_for_all_items(self):
        """Prompt generator must handle all items without crash, emitting dialect hints."""
        for it in self.items:
            msgs = make_translation_messages(it["text"], "ar", "en", source_variant=it["dialect"])
            self.assertGreater(len(msgs), 2)
            sys_msg = msgs[0]["content"]
            dialect_name = _DIALECT_NAMES[it["dialect"]]
            self.assertIn(f"The speaker uses {dialect_name} colloquial Arabic", sys_msg)


if __name__ == "__main__":
    unittest.main()
