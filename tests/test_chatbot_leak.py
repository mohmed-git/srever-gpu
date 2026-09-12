import json
import os
import unittest
import requests

from app.mt import _script_ratio, _is_conversational_chatter

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "leak_corpus.jsonl")
MT_URL = os.environ.get("LINGUA_MT_URL")
AUTH_TOKEN = os.environ.get("LINGUA_AUTH_TOKEN")

class TestChatbotLeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = []
        with open(CORPUS_PATH, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    cls.corpus.append(json.loads(line))
        
        cls.use_http = bool(MT_URL)
        if not cls.use_http:
            # Fall back to in-process
            from app.mt import QwenCt2Engine
            from app.config import Settings
            cls.settings = Settings()
            cls.engine = QwenCt2Engine(cls.settings)
            cls.engine.load()
            if not cls.engine.ready:
                raise unittest.SkipTest("MT Engine not loaded")

    def _translate(self, items):
        if self.use_http:
            results = []
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
            url = MT_URL.rstrip('/') + '/translate'
            for text, src, dst in items:
                resp = requests.post(url, json={"text": text, "source": src, "target": dst}, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                results.append(data.get("text", data.get("translation", "")))
            return results
        else:
            res = self.engine.translate_batch(items)
            return [r.text for r in res]

    def test_chatbot_leak_properties(self):
        items = [(c["text"], c["src"], c["dst"]) for c in self.corpus]
        
        outputs = self._translate(items)
        
        failures = []
        for c, out in zip(self.corpus, outputs):
            src, dst, text = c["src"], c["dst"], c["text"]
            known_bad = c.get("known_bad")
            
            if out:
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
            
            # General output/input ratio <= 3 (if input > 0)
            if in_words > 0 and (out_words / in_words) > 3.0:
                failures.append(f"[{src}->{dst}] '{text}': Length explosion ({in_words}w -> {out_words}w) in '{out}'")
                    
            # 4th property: single-token source -> out_words <= 3 and output != known_bad
            if in_words == 1:
                if out_words > 3:
                    failures.append(f"[{src}->{dst}] '{text}': Single-token source output exceeded 3 words ({out_words}w) in '{out}'")
                if known_bad and out == known_bad:
                    failures.append(f"[{src}->{dst}] '{text}': Generated exact known_bad hallucination: '{out}'")

            # Emit per-case results to stdout as requested
            # Fix JSON logging to check if this specific item failed
            item_failed = any(text in f for f in failures)
            print(json.dumps({"text": text, "out": out, "checks_failed": item_failed}, ensure_ascii=False))

        if failures:
            self.fail(f"Chatbot Leak Failed on {len(failures)}/{len(self.corpus)} cases:\n" + "\n".join(failures))

if __name__ == "__main__":
    unittest.main()
