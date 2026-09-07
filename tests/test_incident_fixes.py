"""Tests T1 through T7 for Lingua Buds server directives S1 through S7.

Covers:
  T1: _hollow_check rejects orphan punctuation without word characters ('” ،', 'make the days count', 'ar')
  T2: _low_target_script identifies untranslated or low-script text under the 0.5 threshold
  T3: Single retry on low script / orphan punctuation succeeds via retry prompt and increments mt_retry
  T4: Exhausted retry with hollow result blanks output text and reports retry in hollow reason
  T5: Slot language snapshot preserves utterance languages across mid-session control frame changes,
      and config ack reports applies_from_utt
  T6: Pinned language verification runs observe-only check after language control change,
      emitting detected_lang/detected_prob and incrementing asr_pinned_lang_mismatch on divergence
  T7: ASR hallucination guard drops English hallucination segments regardless of language
"""

import asyncio
import os
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from transformers import AutoTokenizer

from app.asr import AsrEngine, AsrResult, _EN_HALLUCINATION_BLOCKLIST
from app.config import Settings
from app.metrics import Metrics
from app.mt import (
    _LOW_SCRIPT_RATIO,
    _english_leak,
    _hollow_check,
    _low_target_script,
    make_translation_messages,
    QwenCt2Engine,
)
import app.server
from app.server import (
    _Slot,
    _StreamState,
    _commit_slot,
    _on_audio_frame,
    _on_control_frame,
    _stream_utterance,
)


def pack_v2(flags: int, utt: int, seq: int, payload: bytes = b"") -> bytes:
    return struct.pack("<BBHH", 2, flags, utt, seq) + payload


class DummyWebSocket:
    def __init__(self):
        self.sent_json = []
        self.closed_code = None

    async def send_json(self, data):
        self.sent_json.append(data)

    async def close(self, code=1000):
        self.closed_code = code


class TestIncidentFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
        except Exception:
            cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # ---- T1: _hollow_check on orphan punctuation ----------------------------
    def test_t1_hollow_check_orphan_punctuation(self):
        hollow, reason = _hollow_check("” ،", "make the days count", "ar")
        self.assertTrue(hollow, "Expected hollow=True for orphan punctuation '” ،'")
        self.assertIsNotNone(reason)
        self.assertIn("no word characters", reason.lower())

    # ---- T2: _low_target_script threshold ----------------------------------
    def test_t2_low_target_script(self):
        self.assertEqual(_LOW_SCRIPT_RATIO, 0.5, "Threshold must be 0.5")
        self.assertTrue(_low_target_script("Hello.", "ar"), "Latin text must be detected as low Arabic script")
        self.assertFalse(_low_target_script("رقم الرحلة BA123", "ar"), "Dominant Arabic with flight code must pass")
        # Ratio for 'Welcome مرحبا' is 5/12 = ~41.7% (< 50% threshold, but >= 20%)
        self.assertTrue(_low_target_script("Welcome مرحبا", "ar"), "Mixed sentence with 42% Arabic must trigger low script at 0.5")

    # ---- T3: Mocked generator retry recovery --------------------------------
    def test_t3_mocked_generator_retry_recovery(self):
        settings = Settings()
        engine = QwenCt2Engine(settings)
        engine._tokenizer = self.tokenizer
        engine.metrics = Metrics()

        # Call 1 returns '”', Call 2 (retry) returns 'لا تعد الأيام'
        tok_quote = engine._tokenizer.encode("”")
        tok_arabic = engine._tokenizer.encode("لا تعد الأيام")
        mock_out1 = SimpleNamespace(sequences_ids=[tok_quote], sequences=[engine._tokenizer.tokenize("”")])
        mock_out2 = SimpleNamespace(sequences_ids=[tok_arabic], sequences=[engine._tokenizer.tokenize("لا تعد الأيام")])

        call_count = 0

        def fake_generate_batch(batch, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_out1]
            return [mock_out2]

        mock_generator = MagicMock()
        mock_generator.generate_batch.side_effect = fake_generate_batch
        engine._generator = mock_generator

        results = engine.translate_batch([("Don't count the days", "en", "ar")])
        self.assertEqual(len(results), 1)
        res = results[0]

        self.assertFalse(res.hollow, "Recovered translation must not be hollow")
        self.assertTrue(res.retried, "Must indicate retried=True")
        self.assertEqual(engine.metrics.counter("mt_retry"), 1, "mt_retry counter must be incremented")
        self.assertIn("الأيام", res.text, "Output must contain Arabic words")

    # ---- T4: Mocked generator both outputs hollow --------------------------
    def test_t4_mocked_generator_both_outputs_hollow(self):
        settings = Settings()
        engine = QwenCt2Engine(settings)
        engine._tokenizer = self.tokenizer
        engine.metrics = Metrics()

        tok_dot = engine._tokenizer.encode(".")
        mock_out_dot = SimpleNamespace(sequences_ids=[tok_dot], sequences=[engine._tokenizer.tokenize(".")])

        mock_generator = MagicMock()
        mock_generator.generate_batch.return_value = [mock_out_dot]
        engine._generator = mock_generator

        results = engine.translate_batch([("Don't count the days", "en", "ar")])
        self.assertEqual(len(results), 1)
        res = results[0]

        self.assertTrue(res.hollow, "Double failed retry must remain hollow")
        self.assertEqual(res.text, "", "Hollow translation must be blanked to avoid handing punctuation to TTS")
        self.assertIsNotNone(res.hollow_reason)
        self.assertIn("retry", res.hollow_reason.lower(), "Hollow reason must mention retry")

    # ---- T8: _english_leak detector ----------------------------------------
    def test_t8_english_leak_detector(self):
        self.assertTrue(_english_leak("How are you?", "cs"), "'How are you?' must be detected as English leak into Czech")
        self.assertFalse(_english_leak("Jak se máš?", "cs"), "'Jak se máš?' is native Czech and must not be flagged")
        self.assertFalse(_english_leak("Wie geht es dir?", "de"), "'Wie geht es dir?' is native German and must not be flagged")
        self.assertFalse(_english_leak("How are you?", "en"), "English target must never be flagged as English leak")
        self.assertTrue(_english_leak("Hello, how are you?", "de"), "English phrase into German must be flagged as English leak")

    # ---- T9: Mocked generator english_leak retry recovery ------------------
    def test_t9_mocked_generator_english_leak_retry_recovery(self):
        settings = Settings()
        engine = QwenCt2Engine(settings)
        engine._tokenizer = self.tokenizer
        engine.metrics = Metrics()

        # Call 1 emits 'How are you?' (leak), Call 2 (retry) emits 'Jak se máš?'
        tok_leak = engine._tokenizer.encode("How are you?")
        tok_czech = engine._tokenizer.encode("Jak se máš?")
        mock_out1 = SimpleNamespace(sequences_ids=[tok_leak], sequences=[engine._tokenizer.tokenize("How are you?")])
        mock_out2 = SimpleNamespace(sequences_ids=[tok_czech], sequences=[engine._tokenizer.tokenize("Jak se máš?")])

        call_count = 0

        def fake_generate_batch(batch, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_out1]
            return [mock_out2]

        engine._generator = MagicMock()
        engine._generator.generate_batch.side_effect = fake_generate_batch

        results = engine.translate_batch([("كيف حالك", "ar", "cs")])
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(call_count, 2, "Generator must be called twice (initial + retry)")
        self.assertTrue(res.retried, "Result must be marked as retried")
        self.assertEqual(res.text, "Jak se máš?", "Decoded text must be the recovered Czech output")
        self.assertFalse(res.hollow, "Recovered translation must not be hollow")
        self.assertEqual(engine.metrics.counter("mt_english_leak"), 1, "mt_english_leak counter must be incremented")
        self.assertEqual(engine.metrics.counter("mt_retry"), 1, "mt_retry counter must be incremented")

    # ---- T10: Mocked generator english_leak exhausted hollow ---------------
    def test_t10_mocked_generator_english_leak_exhausted_hollow(self):
        settings = Settings()
        engine = QwenCt2Engine(settings)
        engine._tokenizer = self.tokenizer
        engine.metrics = Metrics()

        # Both initial call and retry emit 'How are you?'
        tok_leak = engine._tokenizer.encode("How are you?")
        mock_out = SimpleNamespace(sequences_ids=[tok_leak], sequences=[engine._tokenizer.tokenize("How are you?")])

        engine._generator = MagicMock()
        engine._generator.generate_batch.return_value = [mock_out]

        results = engine.translate_batch([("كيف حالك", "ar", "cs")])
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertTrue(res.retried, "Result must be marked as retried")
        self.assertTrue(res.hollow, "Exhausted leak must be marked hollow")
        self.assertEqual(res.text, "", "Hollow text must be blanked out for TTS suppression")
        self.assertIsNotNone(res.hollow_reason)
        self.assertIn("english_leak", res.hollow_reason, "Hollow reason must explicitly cite english_leak")

    # ---- T11: Multilingual prompt isolation & bare user turn ---------------
    def test_t11_multilingual_prompt_isolation(self):
        # ar -> cs must be zero-shot with bare user turn
        msgs_cs = make_translation_messages("كيف حالك", "ar", "cs")
        self.assertEqual(len(msgs_cs), 2, "ar -> cs must have exactly 2 turns (zero-shot)")
        self.assertEqual(msgs_cs[1]["content"], "كيف حالك", "User turn must be bare text")
        self.assertNotIn("Where is the train station?", msgs_cs[0]["content"])

        # ar -> de must be zero-shot with bare user turn
        msgs_de = make_translation_messages("مرحبا", "ar", "de")
        self.assertEqual(len(msgs_de), 2, "ar -> de must have exactly 2 turns (zero-shot)")
        self.assertEqual(msgs_de[1]["content"], "مرحبا")

        # fr -> ar must be zero-shot with bare user turn
        msgs_fr_ar = make_translation_messages("Bonjour", "fr", "ar")
        self.assertEqual(len(msgs_fr_ar), 2, "fr -> ar must have exactly 2 turns (zero-shot)")
        self.assertEqual(msgs_fr_ar[1]["content"], "Bonjour")

        # en -> ar and ar -> en must retain hardened few-shot prompts
        msgs_en_ar = make_translation_messages("Where is the train station?", "en", "ar")
        self.assertGreater(len(msgs_en_ar), 2, "en -> ar must retain few-shot examples")
        msgs_ar_en = make_translation_messages("أين محطة القطار؟", "ar", "en")
        self.assertGreater(len(msgs_ar_en), 2, "ar -> en must retain few-shot examples")


class TestIncidentFixesAsync(unittest.IsolatedAsyncioTestCase):
    # ---- T5: Per-slot language snapshot and config applies_from_utt ---------
    async def test_t5_slot_language_snapshot_and_applies_from_utt(self):
        settings = Settings(sample_rate=16000, max_utterance_seconds=2.0)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.source = "en"
        state.target = "ar"

        metrics = Metrics()

        captured_calls = []

        class MockPipeline:
            def __init__(self):
                self.metrics = metrics
                self.ready = True
                self.settings = settings
                self.asr = MagicMock()

            async def _decode_checked(self, raw, **kwargs):
                return SimpleNamespace(samples=np.zeros(1600, dtype=np.float32), duration_s=0.1, source_format="pcm_s16le"), 1.0

            async def translate_audio_streaming(self, raw, source, target, **kwargs):
                captured_calls.append((source, target))
                yield {
                    "type": "sentence",
                    "index": 0,
                    "sentence_count": 1,
                    "is_last": True,
                    "original_text": "hello",
                    "translated_text": "مرحبا",
                    "source_lang": source,
                    "target_lang": target,
                    "elapsed_ms": 10.0,
                    "mt_ms": 5.0,
                    "hollow": False,
                    "retried": False,
                }
                yield {
                    "type": "final",
                    "original_text": "hello",
                    "translated_text": "مرحبا",
                    "source_lang": source,
                    "target_lang": target,
                    "asr_ms": 10.0,
                    "mt_ms": 5.0,
                    "total_server_ms": 15.0,
                    "hollow": False,
                    "retried": False,
                }

        pipeline = MockPipeline()
        app.server.PIPELINE = pipeline

        try:
            # 1. Audio frame opens slot 1 under (en -> ar)
            audio_frame = pack_v2(flags=0, utt=1, seq=0, payload=b"\x00\x00" * 160)
            await _on_audio_frame(state, audio_frame)
            self.assertIn(1, state.slots)
            self.assertEqual(state.slots[1].source, "en")
            self.assertEqual(state.slots[1].target, "ar")

            # 2. Control frame arrives mid-utterance swapping direction to (ar -> en)
            await _on_control_frame(state, {"source": "ar", "target": "en"})
            self.assertEqual(state.source, "ar")
            self.assertEqual(state.target, "en")

            # Verify config ack has applies_from_utt == next utt (2)
            config_acks = [m for m in ws.sent_json if m.get("event") == "config"]
            self.assertEqual(len(config_acks), 1)
            self.assertEqual(config_acks[0]["applies_from_utt"], 2)

            # 3. Commit slot 1
            slot = state.slots[1]
            await _commit_slot(state, slot, seq=0)

            # Process queued utterance via worker
            item = await state.utterance_queue.get()
            self.assertEqual(item[0], "fresh")
            _, raw, utt_id, commit_time, slot_src, slot_dst = item
            self.assertEqual(slot_src, "en")
            self.assertEqual(slot_dst, "ar")

            await _stream_utterance(state, raw, utt_id, source=slot_src, target=slot_dst)

            self.assertEqual(len(captured_calls), 1)
            src_called, dst_called = captured_calls[0]
            self.assertEqual(src_called, "en", "Committed slot 1 must use snapshotted source 'en'")
            self.assertEqual(dst_called, "ar", "Committed slot 1 must use snapshotted target 'ar'")

            # Check frames received by client
            frames = [m for m in ws.sent_json if m.get("type") in ("sentence", "final")]
            for f in frames:
                self.assertEqual(f["source_lang"], "en")
                self.assertEqual(f["target_lang"], "ar")
        finally:
            app.server.PIPELINE = None

    # ---- T6: Observe-only language verification -----------------------------
    async def test_t6_observe_only_language_verification(self):
        settings = Settings(sample_rate=16000, max_utterance_seconds=2.0)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.source = "en"
        state.target = "ar"

        metrics = Metrics()

        class MockPipeline:
            def __init__(self):
                self.metrics = metrics
                self.ready = True
                self.settings = settings
                self.asr = MagicMock()
                # Mock detect_language returning ("ar", 0.9)
                self.asr.detect_language.return_value = ("ar", 0.9)

            async def _decode_checked(self, raw, **kwargs):
                return SimpleNamespace(samples=np.zeros(1600, dtype=np.float32), duration_s=0.1, source_format="pcm_s16le"), 1.0

            async def translate_audio_streaming(self, raw, source, target, **kwargs):
                yield {
                    "type": "final",
                    "original_text": "hello",
                    "translated_text": "مرحبا",
                    "source_lang": source,
                    "target_lang": target,
                    "hollow": False,
                }

        pipeline = MockPipeline()
        app.server.PIPELINE = pipeline

        try:
            # Control frame sets verify_lang_next
            await _on_control_frame(state, {"source": "en", "target": "ar"})
            self.assertTrue(state.verify_lang_next)

            # Stream utterance on pinned "en"
            await _stream_utterance(state, b"\x00\x00" * 160, utt_tag=1, source="en", target="ar")

            self.assertEqual(
                metrics.counter("asr_pinned_lang_mismatch"),
                1,
                "asr_pinned_lang_mismatch must increment when detected ('ar') != pinned ('en') with prob >= 0.6",
            )
            final_frames = [m for m in ws.sent_json if m.get("type") == "final"]
            self.assertEqual(len(final_frames), 1)
            self.assertEqual(final_frames[0].get("detected_lang"), "ar")
            self.assertEqual(final_frames[0].get("detected_prob"), 0.9)
            self.assertFalse(state.verify_lang_next, "verify_lang_next must be cleared after check")
        finally:
            app.server.PIPELINE = None

    # ---- T7: English hallucination blocklist in ASR -------------------------
    def test_t7_asr_hallucination_guard_drops_en_blocklist(self):
        settings = Settings()
        engine = AsrEngine(settings)
        engine._model = MagicMock()

        # Mock segment matching English blocklist item with trailing punctuation
        seg_hallucinated = SimpleNamespace(
            text="I hope you guys enjoyed it.",
            no_speech_prob=0.0,
            avg_logprob=0.0,
            compression_ratio=1.0,
        )
        engine._model.transcribe.return_value = ([seg_hallucinated], SimpleNamespace(language="en", language_probability=0.99))

        res = engine.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        self.assertEqual(res.dropped_segments, 1, "English blocklist segment must be dropped")
        self.assertEqual(res.text, "", "Result text must be empty when all segments are dropped")
        self.assertTrue(res.hollow, "Result must be marked hollow")
        self.assertIn("hallucination guard", res.hollow_reason)


if __name__ == "__main__":
    unittest.main()
