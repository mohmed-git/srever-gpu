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
        import ast

        tree = ast.parse(self.code, filename=MT_PATH)
        violations = []

        for node in ast.walk(tree):
            # Skip AST nodes inside _unpack_item function definition
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_unpack_item":
                continue

            # Case 1: For loop unpacking: `for (text, ...) in items:` or `for (text, ...), row in zip(items, ...):`
            if isinstance(node, ast.For):
                target = node.target
                has_tuple_target = isinstance(target, ast.Tuple)
                has_nested_tuple = (
                    isinstance(target, ast.Tuple)
                    and any(isinstance(elt, ast.Tuple) for elt in target.elts)
                )
                
                # Accurately verify if the variable name 'items' is being iterated over (excluding dict .items())
                iterates_items_var = any(
                    isinstance(sub, ast.Name) and sub.id == "items" for sub in ast.walk(node.iter)
                )
                if iterates_items_var and (has_tuple_target or has_nested_tuple):
                    violations.append(f"Line {node.lineno}: for-loop tuple unpacking over {ast.unparse(node.iter)}: '{ast.unparse(node).splitlines()[0]}'")

            # Case 2: Assignment unpacking on its own line: `text, src, dst = item`
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Tuple):
                        val_str = ast.unparse(node.value)
                        if val_str in {"item", "items[0]", "j.payload"}:
                            violations.append(f"Line {node.lineno}: direct assignment tuple unpacking from '{val_str}': '{ast.unparse(node)}'")

        self.assertEqual(
            violations, [],
            f"Found forbidden tuple unpacking over items outside _unpack_item via AST walk:\n" + "\n".join(violations)
        )

    def test_sabotage_tuple_unpacking_structural_test(self):
        import ast

        def check_code_violations(source: str):
            tree = ast.parse(source)
            violations = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_unpack_item":
                    continue
                if isinstance(node, ast.For):
                    target = node.target
                    has_tuple = isinstance(target, ast.Tuple) or (
                        isinstance(target, ast.Tuple) and any(isinstance(elt, ast.Tuple) for elt in target.elts)
                    )
                    iterates_items_var = any(
                        isinstance(sub, ast.Name) and sub.id == "items" for sub in ast.walk(node.iter)
                    )
                    if iterates_items_var and has_tuple:
                        violations.append(f"Line {node.lineno}: {ast.unparse(node).splitlines()[0]}")
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Tuple) and ast.unparse(node.value) in {"item", "items[0]", "j.payload"}:
                            violations.append(f"Line {node.lineno}: {ast.unparse(node)}")
            return violations

        # Sabotage 1: re-adding line 1922 from 1deaf8b (for (text, _s, _d), row in zip(items, generated):)
        sabotage_for = self.code + "\ndef bad_func(items, generated):\n    for (text, _s, _d), row in zip(items, generated):\n        pass\n"
        v1 = check_code_violations(sabotage_for)
        self.assertGreater(len(v1), 0, "AST check failed to detect sabotaged for-loop tuple unpacking!")

        # Sabotage 2: standalone assignment on its own line (text, src, dst = item)
        sabotage_assign = self.code + "\ndef bad_func2(item):\n    text, src, dst = item\n    return text\n"
        v2 = check_code_violations(sabotage_assign)
        self.assertGreater(len(v2), 0, "AST check failed to detect sabotaged standalone assignment tuple unpacking!")

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
