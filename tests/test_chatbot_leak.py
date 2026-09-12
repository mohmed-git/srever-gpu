import json
import os
import unittest

from app.mt import QwenCt2Engine, _script_ratio, _is_conversational_chatter
from app.config import Settings

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "leak_corpus.jsonl")

class TestChatbotLeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings()
        cls.engine = QwenCt2Engine(cls.settings)
        cls.engine.load()
        if not cls.engine.ready:
            raise unittest.SkipTest("MT Engine not loaded")

        cls.corpus = []
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cls.corpus.append(json.loads(line))

    def test_chatbot_leak_properties(self):
        items = [(c["text"], c["src"], c["dst"]) for c in self.corpus]
        
        results = self.engine.translate_batch(items)
        
        failures = []
        for c, res in zip(self.corpus, results):
            src, dst, text = c["src"], c["dst"], c["text"]
            out = res.text
            
            sr = _script_ratio(out, dst)
            if sr < 0.5:
                failures.append(f"[{src}->{dst}] '{text}': Target script ratio {sr:.0%} < 50% for '{out}'")
                
            if _is_conversational_chatter(out):
                failures.append(f"[{src}->{dst}] '{text}': Chatter pattern detected in '{out}'")
                
            from app.mt import detect_person_mismatch
            if detect_person_mismatch(text, out, src, dst):
                failures.append(f"[{src}->{dst}] '{text}': Person mismatch (1st person injected) in '{out}'")
                
            in_words = len(text.split())
            out_words = len(out.split())
            if in_words > 0 and (out_words / in_words) > 3.0:
                if in_words == 1 and out_words <= 4:
                    pass
                else:
                    failures.append(f"[{src}->{dst}] '{text}': Length explosion ({in_words}w -> {out_words}w) in '{out}'")

        if failures:
            self.fail(f"Chatbot Leak Failed on {len(failures)}/{len(self.corpus)} cases:\n" + "\n".join(failures))

if __name__ == "__main__":
    unittest.main()
