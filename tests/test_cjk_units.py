import os
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.mt import _count_units, is_cjk_lang, check_needs_retry, _apply_output_guards

class TestCJKUnits(unittest.TestCase):
    def test_is_cjk_lang(self):
        self.assertTrue(is_cjk_lang('zh'))
        self.assertTrue(is_cjk_lang('zh-CN'))
        self.assertTrue(is_cjk_lang('ja'))
        self.assertTrue(is_cjk_lang('ko'))
        self.assertFalse(is_cjk_lang('en'))
        self.assertFalse(is_cjk_lang('ar'))
        self.assertFalse(is_cjk_lang('fr'))

    def test_count_units_spaced(self):
        self.assertEqual(_count_units('Hello world from python', 'en'), 4)
        self.assertEqual(_count_units('صباح الخير يا صديقي', 'ar'), 4)
        self.assertEqual(_count_units('Bonjour tout le monde', 'fr'), 4)

    def test_count_units_cjk(self):
        self.assertEqual(_count_units('你是做什么工作的？', 'zh'), 9)
        self.assertEqual(_count_units('こんにちは 世界', 'ja'), 7)
        self.assertEqual(_count_units('안녕하세요', 'ko'), 5)

    def test_cjk_no_false_length_explosion(self):
        zh_text = '你是做什么工作的？'
        en_decoded = 'What is your job?'
        needs_retry, reason = check_needs_retry(zh_text, en_decoded, src='zh', dst='en')
        self.assertFalse(needs_retry, f'Expected no retry, got {reason}')
        final, hollow, reason, _ = _apply_output_guards(zh_text, en_decoded, src='zh', dst='en')
        self.assertFalse(hollow, f'Expected not hollow, got {reason}')
        self.assertEqual(final, en_decoded)

        zh_text_2 = '火车站建在哪里？'
        en_decoded_2 = 'Where is the train station located?'
        needs_retry, reason = check_needs_retry(zh_text_2, en_decoded_2, src='zh', dst='en')
        self.assertFalse(needs_retry, f'Expected no retry, got {reason}')
        final, hollow, reason, _ = _apply_output_guards(zh_text_2, en_decoded_2, src='zh', dst='en')
        self.assertFalse(hollow, f'Expected not hollow, got {reason}')

    def test_matrix_corpus_file_exists(self):
        matrix_file = os.path.join(os.path.dirname(__file__), 'matrix_corpus.jsonl')
        self.assertTrue(os.path.isfile(matrix_file), 'matrix_corpus.jsonl should exist in tests/')
        with open(matrix_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), 37, f'Expected 37 items in matrix corpus, got {len(lines)}')

if __name__ == '__main__':
    unittest.main()
