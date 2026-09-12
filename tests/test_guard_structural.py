import os
import re
import unittest

MT_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "mt.py")

class TestGuardStructural(unittest.TestCase):
    def setUp(self):
        with open(MT_PATH, "r", encoding="utf-8") as f:
            self.code = f.read()

    def test_detected_after_retry_appears_exactly_once(self):
        # Assert the string "detected after retry" appears exactly once in the entire file
        count = self.code.count("detected after retry")
        self.assertEqual(
            count, 1,
            f"'detected after retry' should appear exactly once in app/mt.py, found {count}"
        )

    def test_all_translate_batch_bodies_reference_apply_output_guards(self):
        # Find all classes that inherit from MtEngine and check their translate_batch
        engine_classes = ["QwenCt2Engine", "QwenVllmEngine", "QwenHfEngine", "M2M100Ct2Engine"]
        for cls_name in engine_classes:
            pattern = rf"class {cls_name}\(.*?\):.*?(?=class |\Z)"
            match = re.search(pattern, self.code, re.DOTALL)
            self.assertIsNotNone(match, f"Class {cls_name} not found in app/mt.py")
            class_body = match.group(0)
            
            # Find translate_batch inside this class
            tb_match = re.search(r"def translate_batch\(self,.*?\n\s+def |\Z", class_body, re.DOTALL)
            self.assertIsNotNone(tb_match, f"translate_batch not found in {cls_name}")
            tb_body = tb_match.group(0)
            
            self.assertIn(
                "_apply_output_guards", tb_body,
                f"{cls_name}.translate_batch does NOT call _apply_output_guards!"
            )

    def test_sabotage_structural_test(self):
        # Sabotage: if an engine carries duplicate guard string, the test would catch it
        sabotaged_code = self.code + '\n# Sabotage: f"chatter detected after retry: ..."'
        self.assertNotEqual(sabotaged_code.count("detected after retry"), 1)

if __name__ == "__main__":
    unittest.main()
