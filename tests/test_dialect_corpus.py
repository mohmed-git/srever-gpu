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

    def test_extended_evaluation_corpora_and_grand_total(self):
        """All 4 committed evaluation corpora must exist, parse cleanly, and total 116 items."""
        test_dir = Path(__file__).parent
        file_counts = {
            "dialect_corpus.jsonl": 21,
            "dialect_heldout_v1.jsonl": 46,
            "podcast_sa_001.jsonl": 19,
            "a5000_eval_30.jsonl": 30,
        }
        total_items = 0
        for fname, expected_count in file_counts.items():
            fpath = test_dir / fname
            self.assertTrue(fpath.exists(), f"{fname} must exist")
            count = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        self.assertIn("id", item)
                        self.assertIn("text", item)
                        self.assertIn("expected", item)
                        self.assertTrue(len(item["text"].strip()) > 0)
                        count += 1
            self.assertEqual(count, expected_count, f"{fname} count mismatch")
            total_items += count

        self.assertEqual(total_items, 116, "Grand total across all committed corpora must be exactly 116 items")


if __name__ == "__main__":
    unittest.main()
