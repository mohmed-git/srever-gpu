import json
import os
import unittest
import requests

from app.mt import _script_ratio, _is_conversational_chatter, is_degenerate_short, has_annotation_or_passthrough, detect_person_mismatch

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
        cls.backend_name = "Unknown"
        cls.backend_class = "Unknown"
        if cls.use_http:
            try:
                headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
                resp = requests.get(MT_URL.rstrip('/') + '/health', headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    cls.backend_name = data.get("mt_backend", "Unknown")
                    cls.backend_class = data.get("mt_backend_class", "Unknown")
            except Exception:
                pass
        else:
            # Fall back to in-process
            from app.mt import QwenCt2Engine
            from app.config import Settings
            cls.settings = Settings()
            cls.engine = QwenCt2Engine(cls.settings)
            cls.engine.load()
            if not cls.engine.ready:
                raise unittest.SkipTest("MT Engine not loaded")
            cls.backend_name = cls.engine.name
            cls.backend_class = cls.engine.__class__.__name__

    def _translate(self, items):
        if self.use_http:
            results = []
            headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
            url = MT_URL.rstrip('/') + '/translate'
            for text, src, dst in items:
                resp = requests.post(url, json={"text": text, "source": src, "target": dst}, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                results.append(data)
            return results
        else:
            res = self.engine.translate_batch(items)
            results = []
            for r in res:
                results.append({
                    "text": r.text, 
                    "hollow_reason": r.hollow_reason,
                    "mt_ms": getattr(r, "mt_ms", 0),
                    "input_tokens": getattr(r, "input_tokens", 0),
                    "output_tokens": getattr(r, "output_tokens", 0)
                })
            return results

    def test_chatbot_leak_properties(self):
        print(f"\nACTIVE MT BACKEND: {self.backend_class} ({self.backend_name})")
        items = [(c["text"], c["src"], c["dst"]) for c in self.corpus]
        
        outputs = self._translate(items)
        
        failures = []
        hollow_count = 0
        total_mt_ms = 0
        total_in_toks = 0
        total_out_toks = 0
        
        for c, output_data in zip(self.corpus, outputs):
            src, dst, text = c["src"], c["dst"], c["text"]
            known_bad = c.get("known_bad")
            must_translate = c.get("must_translate", False)
            out = output_data.get("text", output_data.get("translation", ""))
            hollow_reason = output_data.get("hollow_reason", "")
            
            mt_ms = output_data.get("mt_ms", 0)
            in_toks = output_data.get("input_tokens", 0)
            out_toks = output_data.get("output_tokens", 0)
            
            total_mt_ms += mt_ms
            total_in_toks += in_toks
            total_out_toks += out_toks
            
            item_failures = []
            
            if out == "":
                hollow_count += 1
                if must_translate:
                    item_failures.append(f"[{src}->{dst}] '{text}': hollow_from_guard (must_translate is true). Reason: {hollow_reason}")
            else:
                sr = _script_ratio(out, dst)
                if sr < 0.5:
                    item_failures.append(f"[{src}->{dst}] '{text}': Target script ratio {sr:.0%} < 50% for '{out}'")
                
            if _is_conversational_chatter(out):
                item_failures.append(f"[{src}->{dst}] '{text}': Chatter pattern detected in '{out}'")
                
            if in_toks == 0 and len(text.strip()) > 0:
                item_failures.append(f"[{src}->{dst}] '{text}': input_tokens is 0 for non-empty source")

            if is_degenerate_short(out, dst):
                item_failures.append(f"[{src}->{dst}] '{text}': Degenerate short output in '{out}'")

            if has_annotation_or_passthrough(text, out):
                item_failures.append(f"[{src}->{dst}] '{text}': Annotation or passthrough leak in '{out}'")

            if detect_person_mismatch(text, out, src, dst):
                if dst == "fr":
                    # Observe-only in this phase as instructed
                    pass
                else:
                    item_failures.append(f"[{src}->{dst}] '{text}': Person mismatch (1st person injected) in '{out}'")
                
            in_words = len(text.split())
            out_words = len(out.split())
            
            # General output/input ratio <= 3 (if input > 0)
            if in_words > 0 and (out_words / in_words) > 3.0:
                item_failures.append(f"[{src}->{dst}] '{text}': Length explosion ({in_words}w -> {out_words}w) in '{out}'")
                    
            # 4th property: single-token source -> out_words <= 3 and output != known_bad
            if in_words == 1:
                if out_words > 3:
                    item_failures.append(f"[{src}->{dst}] '{text}': Single-token source output exceeded 3 words ({out_words}w) in '{out}'")
                if known_bad and out == known_bad:
                    item_failures.append(f"[{src}->{dst}] '{text}': Generated exact known_bad hallucination: '{out}'")

            failures.extend(item_failures)
            
            # Emit per-case results to stdout as requested
            print(json.dumps({
                "text": text,
                "out": out,
                "hollow_reason": hollow_reason,
                "mt_ms": mt_ms,
                "in_tok": in_toks,
                "out_tok": out_toks,
                "checks_failed": len(item_failures) > 0,
                "failures": item_failures
            }, ensure_ascii=False))

        # Hollow budget: hollow_count / len(corpus) <= 0.10
        hollow_ratio = hollow_count / len(self.corpus)
        if hollow_ratio > 0.10:
            failures.append(f"Hollow budget exceeded: {hollow_count}/{len(self.corpus)} ({hollow_ratio:.1%}) > 10%")

        print(f"\\n--- Token & Latency Table ---")
        print(f"Total MT MS: {total_mt_ms} ms")
        print(f"Total Input Tokens: {total_in_toks}")
        print(f"Total Output Tokens: {total_out_toks}")
        
        if failures:
            self.fail(f"Chatbot Leak Failed on {len(failures)} cases:\\n" + "\\n".join(failures))

if __name__ == "__main__":
    unittest.main()
