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

    def test_zero_tuple_unpacking_over_items_outside_unpack_item(self):
        # Remove def _unpack_item block from code before checking
        unpack_func_pattern = r"def _unpack_item\(.*?\n(?=\S)"
        code_without_unpack = re.sub(unpack_func_pattern, "", self.code, flags=re.DOTALL)
        
        tuple_unpack_patterns = [
            r"for\s+\([^\)]+\)\s+in\s+items",
            r"for\s+\([^\)]+\)\s*,\s*\w+\s+in\s+zip\(\s*items",
            r"for\s+\w+\s*,\s*\([^\)]+\)\s+in\s+zip\(\s*items",
            r"for\s+\w+\s*,\s*\w+\s*,\s*\w+\s+in\s+items",
        ]
        found_matches = []
        for pat in tuple_unpack_patterns:
            matches = re.findall(pat, code_without_unpack)
            if matches:
                found_matches.extend(matches)
        
        self.assertEqual(
            len(found_matches), 0,
            f"Found forbidden tuple unpacking over items outside _unpack_item in app/mt.py: {found_matches}"
        )

    def test_sabotage_tuple_unpacking_structural_test(self):
        # Sabotage: re-adding line 1922 from 1deaf8b MUST be caught by the pattern
        sabotaged_code = self.code + "\nfor (text, _s, _d), row in zip(items, generated):\n    pass\n"
        pattern = r"for\s+\([^\)]+\)\s*,\s*\w+\s+in\s+zip\(\s*items"
        matches = re.findall(pattern, sabotaged_code)
        self.assertGreater(len(matches), 0, "Sabotage pattern was not detected!")

    def test_all_translate_batch_bodies_use_unpack_item(self):
        engine_classes = ["QwenCt2Engine", "QwenVllmEngine", "QwenHfEngine", "M2M100Ct2Engine"]
        for cls_name in engine_classes:
            pattern = rf"class {cls_name}\(.*?\):.*?(?=class |\Z)"
            match = re.search(pattern, self.code, re.DOTALL)
            self.assertIsNotNone(match, f"Class {cls_name} not found in app/mt.py")
            class_body = match.group(0)
            
            tb_match = re.search(r"def translate_batch\(self,.*?\n\s+def |\Z", class_body, re.DOTALL)
            self.assertIsNotNone(tb_match, f"translate_batch not found in {cls_name}")
            tb_body = tb_match.group(0)
            
            self.assertIn(
                "_unpack_item", tb_body,
                f"{cls_name}.translate_batch does NOT call _unpack_item!"
            )

    def test_sabotage_structural_test(self):
        # Sabotage: if an engine carries duplicate guard string, the test would catch it
        sabotaged_code = self.code + '\n# Sabotage: f"chatter detected after retry: ..."'
        self.assertNotEqual(sabotaged_code.count("detected after retry"), 1)

if __name__ == "__main__":
    unittest.main()
