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

from collections import deque

from app.asr import AsrEngine, AsrResult, _EN_HALLUCINATION_BLOCKLIST
from app.config import Settings
from app.languages import AR_VARIANTS, catalogue, normalise_variant
from app.metrics import Metrics
from app.mt import (
    _LOW_SCRIPT_RATIO,
    _english_leak,
    _hollow_check,
    _low_target_script,
    _script_ratio,
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
    is_echo_match,
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

    # ---- T12: Script ratio families and cross-script verification ----------
    def test_t12_script_ratio_families(self):
        # cs is Latin script (unlisted in _LANG_TO_SCRIPT_FAMILY, falls back to LATIN)
        self.assertEqual(_script_ratio("Hello", "cs"), 1.0, "Latin text into Czech must have ratio 1.0")
        self.assertEqual(_script_ratio("مرحبا", "cs"), 0.0, "Arabic text into Czech must have ratio 0.0")
        # ru is Cyrillic script
        self.assertEqual(_script_ratio("Hello", "ru"), 0.0, "Latin text into Russian must have ratio 0.0")
        self.assertEqual(_script_ratio("Привет", "ru"), 1.0, "Cyrillic text into Russian must have ratio 1.0")
        # ja has Japanese ideographs / kana
        self.assertEqual(_script_ratio("こんにちは", "ja"), 1.0, "Japanese text into Japanese must have ratio 1.0")
        # fa is Arabic/Persian script
        self.assertEqual(_script_ratio("Hello", "fa"), 0.0, "Latin text into Persian must have ratio 0.0")
        self.assertEqual(_script_ratio("سلام", "fa"), 1.0, "Persian text into Persian must have ratio 1.0")
        # _low_target_script check
        self.assertTrue(_low_target_script("Hello", "ru"), "Latin text into Russian must trigger low target script")
        self.assertFalse(_low_target_script("Привет", "ru"), "Native Russian must not trigger low target script")

    # ---- Dutch / Afrikaans / Frisian stopword exclusion --------------------
    def test_dutch_afrikaans_stopword_exclusion(self):
        # Shared Germanic stopwords like 'is', 'in' in Dutch, Afrikaans, Frisian must not trigger false leak
        self.assertFalse(_english_leak("Dit is in de winkel", "nl"), "'Dit is in de winkel' in Dutch must not flag as English leak")
        self.assertFalse(_english_leak("Dit is in de winkel", "af"), "'Dit is in de winkel' in Afrikaans must not flag as English leak")
        self.assertFalse(_english_leak("Dit is in de winkel", "fy"), "'Dit is in de winkel' in Frisian must not flag as English leak")
        # Genuine English sentence into Dutch must still be flagged
        self.assertTrue(_english_leak("This is in the shop", "nl"), "English sentence into Dutch must be flagged as leak")

    # ---- Dialect variants in catalogue and prompt messages -----------------
    def test_dialect_messages_and_catalogue(self):
        cat = catalogue()
        ar_entry = next((e for e in cat if e["code"] == "ar"), None)
        self.assertIsNotNone(ar_entry, "'ar' must be present in catalogue")
        self.assertIn("variants", ar_entry, "'ar' entry must have variants attribute")
        self.assertEqual(ar_entry["variants"], AR_VARIANTS)

        # Variant normalisation
        self.assertEqual(normalise_variant("eg"), "EG")
        self.assertEqual(normalise_variant("SA"), "SA")
        self.assertIsNone(normalise_variant("invalid_code"), "Unknown variant must normalise to None")

        # Source variant hint injection
        msgs_eg = make_translation_messages("إزيك عامل إيه", "ar", "en", source_variant="EG")
        self.assertIn("The speaker uses Egyptian colloquial Arabic; interpret idioms accordingly.", msgs_eg[0]["content"])

        # Context prefix for S9 split-repair
        msgs_context = make_translation_messages("عن الذهاب للبيت", "ar", "en", context_prefix="كنت أفكر")
        self.assertIn("[Context from preceding speech: كنت أفكر]\n\nعن الذهاب للبيت", msgs_context[-1]["content"])

    # ---- S8: is_echo_match unit tests --------------------------------------
    def test_s8_is_echo_match_unit(self):
        # Exact match
        self.assertTrue(is_echo_match("Hello world", deque(["Hello world"])))
        # Case & punctuation normalisation
        self.assertTrue(is_echo_match("hello, world!", deque(["Hello World"])))
        # High token Jaccard >= 0.6
        self.assertTrue(is_echo_match("I would really love that", deque(["I really love that"])))
        # Difflib ratio >= 0.75
        self.assertTrue(is_echo_match("the weather is sunny today", deque(["the weather is very sunny today"])))
        # Non-matching user turn (barge-in candidate)
        self.assertFalse(is_echo_match("Can I get a coffee please?", deque(["The train arrives at nine"])))
        # Empty inputs
        self.assertFalse(is_echo_match("", deque(["Test"])))
        self.assertFalse(is_echo_match("Test", deque()))


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
            _, raw, utt_id, commit_time, slot_src, slot_dst = item[:6]
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

    # ---- T13: S8 Echo filter drops playback reflection ---------------------
    async def test_t13_s8_echo_dropped_during_playback(self):
        settings = Settings(sample_rate=16000, max_utterance_seconds=2.0)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.source = "en"
        state.target = "fr"
        # Seed recent TTS output
        state.recent_tts_outputs.append("Bonjour tout le monde")

        metrics = Metrics()
        mt_called = []

        class MockPipeline:
            def __init__(self):
                self.metrics = metrics
                self.ready = True
                self.settings = settings
                self.asr = MagicMock()

            async def _decode_checked(self, raw, **kwargs):
                return SimpleNamespace(samples=np.zeros(1600, dtype=np.float32), duration_s=0.1, source_format="pcm_s16le"), 1.0

            async def translate_audio_streaming(self, raw, source, target, is_echo=None, **kwargs):
                asr_text = "Bonjour tout le monde"
                if is_echo is not None and is_echo(asr_text):
                    self.metrics.incr("echo_dropped")
                    yield {
                        "type": "dropped",
                        "reason": "echo",
                        "original_text": asr_text,
                        "asr_ms": 15.0,
                    }
                    return
                mt_called.append(asr_text)
                yield {
                    "type": "final",
                    "original_text": asr_text,
                    "translated_text": "Hello world",
                    "source_lang": source,
                    "target_lang": target,
                    "asr_ms": 15.0,
                    "mt_ms": 10.0,
                    "total_server_ms": 25.0,
                }

        pipeline = MockPipeline()
        app.server.PIPELINE = pipeline

        try:
            # Send framed audio with bit 2 set (DURING_PLAYBACK = 0x04)
            await _on_audio_frame(state, pack_v2(flags=0x04, utt=1, seq=0, payload=b"\x00\x00" * 160))
            await _on_audio_frame(state, pack_v2(flags=0x06, utt=1, seq=1, payload=b"\x00\x00" * 160))

            item = await state.utterance_queue.get()
            self.assertEqual(item[0], "fresh")
            _, raw, utt_id, commit_time, slot_src, slot_dst, slot_src_var, slot_tgt_var, during_ratio, context_prefix = item
            self.assertGreaterEqual(during_ratio, 0.5, "during_ratio must be >= 0.5 when all frames have bit 2 set")

            from app.server import _handle_utterance_payload
            await _handle_utterance_payload(
                state,
                raw,
                utt_id,
                source=slot_src,
                target=slot_dst,
                source_variant=slot_src_var,
                target_variant=slot_tgt_var,
                during_ratio=during_ratio,
                context_prefix=context_prefix,
            )

            self.assertEqual(metrics.counter("echo_dropped"), 1, "echo_dropped metric must be incremented")
            self.assertEqual(len(mt_called), 0, "MT must not be called on dropped echo utterance")
            dropped_frames = [m for m in ws.sent_json if m.get("event") == "dropped"]
            self.assertEqual(len(dropped_frames), 1)
            self.assertEqual(dropped_frames[0]["reason"], "echo")
            self.assertEqual(dropped_frames[0]["utt"], 1)
        finally:
            app.server.PIPELINE = None

    # ---- T14: S8 Human barge-in passes through during playback -------------
    async def test_t14_s8_barge_in_passes_during_playback(self):
        settings = Settings(sample_rate=16000, max_utterance_seconds=2.0)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.source = "en"
        state.target = "fr"
        # Seed recent TTS output
        state.recent_tts_outputs.append("Bonjour tout le monde")

        metrics = Metrics()
        mt_called = []

        class MockPipeline:
            def __init__(self):
                self.metrics = metrics
                self.ready = True
                self.settings = settings
                self.asr = MagicMock()

            async def _decode_checked(self, raw, **kwargs):
                return SimpleNamespace(samples=np.zeros(1600, dtype=np.float32), duration_s=0.1, source_format="pcm_s16le"), 1.0

            async def translate_audio_streaming(self, raw, source, target, is_echo=None, **kwargs):
                asr_text = "Wait, stop talking please"
                if is_echo is not None and is_echo(asr_text):
                    self.metrics.incr("echo_dropped")
                    yield {
                        "type": "dropped",
                        "reason": "echo",
                        "original_text": asr_text,
                        "asr_ms": 15.0,
                    }
                    return
                mt_called.append(asr_text)
                yield {
                    "type": "final",
                    "original_text": asr_text,
                    "translated_text": "Attendez, arrêtez de parler s'il vous plaît",
                    "source_lang": source,
                    "target_lang": target,
                    "asr_ms": 15.0,
                    "mt_ms": 10.0,
                    "total_server_ms": 25.0,
                }

        pipeline = MockPipeline()
        app.server.PIPELINE = pipeline

        try:
            # Case 1: Utterance with during_ratio < 0.5 (e.g. 1 out of 4 frames) matching TTS text
            # Must NOT be dropped as echo because during_ratio is below the 50% threshold (Directive S8 Gate 1)
            await _on_audio_frame(state, pack_v2(flags=0x04, utt=1, seq=0, payload=b"\x00\x00" * 160))
            await _on_audio_frame(state, pack_v2(flags=0x00, utt=1, seq=1, payload=b"\x00\x00" * 160))
            await _on_audio_frame(state, pack_v2(flags=0x00, utt=1, seq=2, payload=b"\x00\x00" * 160))
            await _on_audio_frame(state, pack_v2(flags=0x02, utt=1, seq=3, payload=b"\x00\x00" * 160))

            item1 = await state.utterance_queue.get()
            _, raw1, utt_id1, _, slot_src, slot_dst, slot_src_var, slot_tgt_var, during_ratio1, context_prefix = item1

            # Mock pipeline returning text that matches recent_tts_outputs
            class MockPipelineEchoText:
                def __init__(self):
                    self.metrics = metrics
                    self.ready = True
                    self.settings = settings
                    self.asr = MagicMock()

                async def _decode_checked(self, raw, **kwargs):
                    return SimpleNamespace(samples=np.zeros(1600, dtype=np.float32), duration_s=0.1, source_format="pcm_s16le"), 1.0

                async def translate_audio_streaming(self, raw, source, target, is_echo=None, **kwargs):
                    asr_text = "Bonjour tout le monde"
                    if is_echo is not None and is_echo(asr_text):
                        self.metrics.incr("echo_dropped")
                        yield {"type": "dropped", "reason": "echo", "original_text": asr_text, "asr_ms": 15.0}
                        return
                    mt_called.append(asr_text)
                    yield {
                        "type": "final",
                        "original_text": asr_text,
                        "translated_text": "Hello everyone",
                        "source_lang": source,
                        "target_lang": target,
                        "asr_ms": 15.0,
                        "mt_ms": 10.0,
                        "total_server_ms": 25.0,
                    }

            app.server.PIPELINE = MockPipelineEchoText()
            from app.server import _handle_utterance_payload
            await _handle_utterance_payload(
                state,
                raw1,
                utt_id1,
                source=slot_src,
                target=slot_dst,
                source_variant=slot_src_var,
                target_variant=slot_tgt_var,
                during_ratio=during_ratio1,
                context_prefix=context_prefix,
            )

            self.assertEqual(metrics.counter("echo_dropped"), 0, "Utterance with during_ratio < 0.5 must not be dropped as echo")
            self.assertEqual(len(mt_called), 1, "MT must be called when during_ratio < 0.5")

            # Case 2: User speaks over TTS playback (during_ratio = 1.0 >= 0.5) with different words (barge-in)
            app.server.PIPELINE = pipeline
            await _on_audio_frame(state, pack_v2(flags=0x04, utt=2, seq=0, payload=b"\x00\x00" * 160))
            await _on_audio_frame(state, pack_v2(flags=0x06, utt=2, seq=1, payload=b"\x00\x00" * 160))

            item2 = await state.utterance_queue.get()
            _, raw2, utt_id2, _, slot_src, slot_dst, slot_src_var, slot_tgt_var, during_ratio2, context_prefix = item2

            await _handle_utterance_payload(
                state,
                raw2,
                utt_id2,
                source=slot_src,
                target=slot_dst,
                source_variant=slot_src_var,
                target_variant=slot_tgt_var,
                during_ratio=during_ratio2,
                context_prefix=context_prefix,
            )

            self.assertEqual(metrics.counter("echo_dropped"), 0, "Barge-in must not increment echo_dropped")
            self.assertEqual(len(mt_called), 2, "MT must be executed for human barge-in turn")
            final_frames = [m for m in ws.sent_json if m.get("type") == "final"]
            self.assertEqual(len(final_frames), 2)
            self.assertEqual(final_frames[-1]["translated_text"], "Attendez, arrêtez de parler s'il vous plaît")
        finally:
            app.server.PIPELINE = None

    # ---- S9: Split-repair context across rapid mid-sentence pauses ----------
    async def test_s9_split_repair_context_prefix(self):
        settings = Settings(sample_rate=16000, max_utterance_seconds=2.0, split_repair_enabled=True)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.source = "en"
        state.target = "ar"

        metrics = Metrics()
        context_prefixes = []

        class MockPipeline:
            def __init__(self):
                self.metrics = metrics
                self.ready = True
                self.settings = settings
                self.asr = MagicMock()

            async def _decode_checked(self, raw, **kwargs):
                return SimpleNamespace(samples=np.zeros(1600, dtype=np.float32), duration_s=0.1, source_format="pcm_s16le"), 1.0

            async def translate_audio_streaming(self, raw, source, target, context_prefix="", **kwargs):
                context_prefixes.append(context_prefix)
                yield {
                    "type": "final",
                    "original_text": "I was thinking",
                    "translated_text": "كنت أفكر",
                    "source_lang": source,
                    "target_lang": target,
                    "asr_ms": 15.0,
                    "mt_ms": 10.0,
                    "total_server_ms": 25.0,
                }

        pipeline = MockPipeline()
        app.server.PIPELINE = pipeline

        try:
            # Slot 1: ends without terminal punctuation
            await _on_audio_frame(state, pack_v2(flags=0x02, utt=1, seq=0, payload=b"\x00\x00" * 160))
            item1 = await state.utterance_queue.get()
            from app.server import _handle_utterance_payload
            await _handle_utterance_payload(state, item1[1], item1[2], source=item1[4], target=item1[5], source_variant=item1[6], target_variant=item1[7], during_ratio=item1[8], context_prefix=item1[9])

            # State now has slot 1 commit record
            self.assertEqual(state.last_committed_src_text, "I was thinking")
            self.assertFalse(state.last_has_terminal_punct, "Must lack terminal punctuation")

            # Slot 2: arrives shortly after (< 800ms)
            await _on_audio_frame(state, pack_v2(flags=0x02, utt=2, seq=0, payload=b"\x00\x00" * 160))
            item2 = await state.utterance_queue.get()
            # Slot 2 queued context_prefix should be slot 1's source text
            self.assertEqual(item2[9], "I was thinking")
            self.assertEqual(metrics.counter("split_repair_count"), 1)
        finally:
            app.server.PIPELINE = None

    # ---- Chatter refusal detection & hollow suppression -------------------
    def test_conversational_chatter_detection(self):
        from app.mt import _is_conversational_chatter, _hollow_check
        chatty = "I'm sorry, but your input is incomplete. Could you please provide more context or finish the"
        self.assertTrue(_is_conversational_chatter(chatty))
        self.assertTrue(_is_conversational_chatter("As an AI, I cannot translate this."))
        self.assertTrue(_is_conversational_chatter("Sorry, please provide more context"))
        self.assertFalse(_is_conversational_chatter("Hello, how are you?"))
        self.assertFalse(_is_conversational_chatter("Where is the train station?"))

        hollow, reason = _hollow_check(chatty, "مرحمن", target_lang="en")
        self.assertTrue(hollow, "Conversational chatter must be marked hollow")
        self.assertIn("conversational chatter", reason)

    # ---- Sentence cache language & variant isolation ----------------------
    def test_norm_hash_language_isolation(self):
        from app.server import norm_hash, _StreamState
        h_es = norm_hash("مرحبا كيف حالك؟", source="ar", target="es")
        h_en = norm_hash("مرحبا كيف حالك؟", source="ar", target="en")
        h_ca = norm_hash("مرحبا كيف حالك؟", source="ar", target="ca")
        self.assertNotEqual(h_es, h_en, "Hashes for different target languages must differ")
        self.assertNotEqual(h_es, h_ca, "Hashes for different target languages must differ")

        # Test cache clearance on apply()
        ws = DummyWebSocket()
        settings = Settings()
        state = _StreamState(settings, ws)
        state.mt_cache["key1"] = "val1"
        state.apply({"target": "en"})
        self.assertEqual(len(state.mt_cache), 0, "mt_cache must be cleared when target language changes")

    # ---- T13b: HIT path echo dropped ---------------------------------------
    async def test_t13b_s8_hit_path_echo_dropped(self):
        """T13b: HIT path must drop tentative result if text matches recent TTS and during_ratio >= 0.5."""
        settings = Settings(sample_rate=16000)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.target = "en"
        state.append_tts_output("Bonjour tout le monde")

        metrics = Metrics()
        pipeline = SimpleNamespace(metrics=metrics, ready=True)
        app.server.PIPELINE = pipeline

        try:
            slot = _Slot(utt_id=1, source="fr", target="en")
            slot.during_playback_frames = 4
            slot.total_frames = 4
            slot.tentative_seq = 1
            slot.tentative_result = (
                [],
                {"original_text": "Bonjour tout le monde", "translated_text": "Hello world", "asr_ms": 12.0, "mt_ms": 8.0},
                20.0,
            )
            state.slots[1] = slot

            from app.server import _commit_slot
            await _commit_slot(state, slot, seq=1)

            self.assertEqual(metrics.counter("echo_dropped"), 1, "echo_dropped must increment on HIT path echo match")
            self.assertTrue(state.utterance_queue.empty(), "utterance_queue must be empty (dropped before queuing)")
            dropped = [m for m in ws.sent_json if m.get("event") == "dropped"]
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0]["mode"], "HIT")
            self.assertTrue(dropped[0]["echo_dropped"])
        finally:
            app.server.PIPELINE = None

    # ---- T14b: HIT path barge-in passes and ratio gate verification --------
    async def test_t14b_s8_hit_path_barge_in_passes_and_ratio_gate(self):
        """T14b: HIT path human barge-in must pass when text differs, and during_ratio < 0.5 must NOT drop."""
        settings = Settings(sample_rate=16000)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.target = "en"
        state.append_tts_output("Bonjour tout le monde")

        metrics = Metrics()
        pipeline = SimpleNamespace(metrics=metrics, ready=True)
        app.server.PIPELINE = pipeline

        try:
            # Case 1: text differs (human barge-in), during_ratio = 1.0 (>= 0.5)
            slot = _Slot(utt_id=1, source="fr", target="en")
            slot.during_playback_frames = 4
            slot.total_frames = 4
            slot.tentative_seq = 1
            slot.tentative_result = (
                [],
                {"original_text": "Quelle heure est-il?", "translated_text": "What time is it?", "asr_ms": 12.0, "mt_ms": 8.0},
                20.0,
            )
            state.slots[1] = slot

            from app.server import _commit_slot
            await _commit_slot(state, slot, seq=1)

            self.assertEqual(metrics.counter("echo_dropped"), 0, "Human barge-in must NOT be dropped")
            self.assertEqual(metrics.counter("tentative_hit"), 1)
            item = await state.utterance_queue.get()
            self.assertEqual(item[0], "serve_cached")

            # Case 2: text matches echo, BUT during_ratio < 0.5 (1 frame out of 4 = 0.25 < 0.5)
            # Gate assertion: if during_ratio gate is sabotaged to 0.0, this case would erroneously drop!
            slot2 = _Slot(utt_id=2, source="fr", target="en")
            slot2.during_playback_frames = 1
            slot2.total_frames = 4
            slot2.tentative_seq = 1
            slot2.tentative_result = (
                [],
                {"original_text": "Bonjour tout le monde", "translated_text": "Hello world", "asr_ms": 12.0, "mt_ms": 8.0},
                20.0,
            )
            state.slots[2] = slot2
            await _commit_slot(state, slot2, seq=1)

            self.assertEqual(metrics.counter("echo_dropped"), 0, "during_ratio < 0.5 must NOT be dropped by echo filter")
            self.assertEqual(metrics.counter("tentative_hit"), 2)
            item2 = await state.utterance_queue.get()
            self.assertEqual(item2[0], "serve_cached")
        finally:
            app.server.PIPELINE = None

    # ---- T13c: AWAIT path echo dropped -------------------------------------
    async def test_t13c_s8_await_path_echo_dropped(self):
        """T13c: AWAIT path must drop completed tentative result if text matches echo and during_ratio >= 0.5."""
        settings = Settings(sample_rate=16000)
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)
        state.protocol_version = 2
        state.target = "en"
        state.append_tts_output("Bonjour tout le monde")

        metrics = Metrics()
        pipeline = SimpleNamespace(metrics=metrics, ready=True)
        app.server.PIPELINE = pipeline

        try:
            slot = _Slot(utt_id=1, source="fr", target="en")
            slot.during_playback_frames = 4
            slot.total_frames = 4
            slot.tentative_seq = 1

            async def mock_tentative():
                await asyncio.sleep(0.01)
                slot.tentative_result = (
                    [],
                    {"original_text": "Bonjour tout le monde", "translated_text": "Hello world", "asr_ms": 12.0, "mt_ms": 8.0},
                    20.0,
                )

            slot.tentative_task = asyncio.create_task(mock_tentative())
            state.slots[1] = slot

            from app.server import _commit_slot
            await _commit_slot(state, slot, seq=1)

            self.assertEqual(metrics.counter("tentative_await"), 1)
            item = await state.utterance_queue.get()
            self.assertEqual(item[0], "await_then_serve")

            # Run worker on await_then_serve
            state.utterance_queue.put_nowait(item)
            state.worker_task = asyncio.create_task(app.server._utterance_worker(state))
            await asyncio.sleep(0.05)
            state.worker_task.cancel()
            try:
                await state.worker_task
            except (asyncio.CancelledError, Exception):
                pass

            self.assertEqual(metrics.counter("echo_dropped"), 1, "AWAIT path must drop echo when completed")
            dropped = [m for m in ws.sent_json if m.get("event") == "dropped"]
            self.assertEqual(len(dropped), 1)
            self.assertEqual(dropped[0]["mode"], "AWAIT")
            self.assertTrue(dropped[0]["echo_dropped"])
        finally:
            app.server.PIPELINE = None

    # ---- T15: Target Arabic strictly Modern Standard Arabic (MSA) ----------
    def test_t15_target_arabic_locked_to_msa(self):
        """T15: Target Arabic must strictly be Modern Standard Arabic (MSA).
        make_translation_messages must never emit listener colloquial hints for Arabic targets."""
        for src in ["en", "fr", "es"]:
            for variant in ["EG", "SA", "SY", "DZ", "MA", "AE", "KW", "QA", "OM", "BH", "IQ", "YE", "SD"]:
                msgs = make_translation_messages("Hello", src, "ar", target_variant=variant)
                sys_content = msgs[0]["content"]
                self.assertNotIn("colloquial", sys_content.lower())
                self.assertNotIn("prefers", sys_content.lower())
                self.assertNotIn("listener", sys_content.lower())
                self.assertIn("Modern Standard Arabic", sys_content)

    # ---- T16: Wire config target_variant ignored & ack null ----------------
    async def test_t16_wire_target_variant_ignored(self):
        """T16: Wire config with target_variant must ignore it, store '', and ack target_variant as None (null in JSON)."""
        ws = DummyWebSocket()
        settings = Settings()
        state = _StreamState(settings, ws)
        await _on_control_frame(state, {"action": "config", "source": "en", "target": "ar", "target_variant": "EG"})
        self.assertEqual(state.target_variant, "")
        config_acks = [m for m in ws.sent_json if m.get("event") == "config"]
        self.assertGreater(len(config_acks), 0)
        self.assertIsNone(config_acks[-1]["target_variant"], "config ack must echo target_variant: None (null in JSON)")

    # ---- T-chatter-integration: Chatter refusal triggers retry in translate_batch
    def test_chatter_integration_retry(self):
        """Integration test: conversational chatter in generate_batch triggers retry in translate_batch."""
        from app.mt import QwenCt2Engine
        import app.mt as mt_mod
        settings = Settings()
        engine = QwenCt2Engine(settings)
        engine._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        engine._generator = MagicMock()
        engine._generator.generate_batch = MagicMock()

        # First call: chatter refusal. Second call: valid translation.
        chatty_seq = [SimpleNamespace(sequences_ids=[engine._tokenizer.encode("I'm sorry, but I cannot translate that without more context.")])]
        valid_seq = [SimpleNamespace(sequences_ids=[engine._tokenizer.encode("Where is the train station?")])]

        engine._generator.generate_batch.side_effect = [chatty_seq, valid_seq]

        initial_count = mt_mod.CHAT_LEAK_SUSPECTED_COUNT
        # ar -> en: target is English, so english_leak and low_script are FALSE; ONLY chatter triggers retry!
        results = engine.translate_batch([("أين محطة القطار؟", "ar", "en")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "Where is the train station?")
        self.assertTrue(results[0].retried, "Must mark retried=True on chatter recovery")
        self.assertGreater(mt_mod.CHAT_LEAK_SUSPECTED_COUNT, initial_count, "Chatter count must increment")

    # ---- Deduplicate recent_tts_outputs ------------------------------------
    def test_deduplicate_recent_tts_outputs(self):
        """_StreamState.append_tts_output must deduplicate repeated identical consecutive phrases."""
        ws = DummyWebSocket()
        settings = Settings()
        state = _StreamState(settings, ws)
        state.append_tts_output("Hello world")
        state.append_tts_output("Hello world")
        state.append_tts_output("  Hello world  ")
        state.append_tts_output("How are you?")
        state.append_tts_output("How are you?")
        self.assertEqual(list(state.recent_tts_outputs), ["Hello world", "How are you?"])

    # ---- Bearer Token Authentication ---------------------------------------
    async def test_bearer_token_auth(self):
        """When auth_token is set, WebSocket and REST /translate require valid bearer or query token."""
        from fastapi.testclient import TestClient
        import app.server as server_mod

        orig_token = server_mod.SETTINGS.auth_token
        orig_pipe = server_mod.PIPELINE
        try:
            object.__setattr__(server_mod.SETTINGS, "auth_token", "secret-token-123")
            server_mod.PIPELINE = SimpleNamespace(ready=True, metrics=Metrics())
            client = TestClient(server_mod.app)

            # REST /translate without auth -> 401
            res = client.post("/translate", json={"text": "hi", "source": "en", "target": "ar"})
            self.assertEqual(res.status_code, 401)
            self.assertIn("unauthorized", res.json()["error"])

            # REST /translate with invalid auth -> 401
            res = client.post(
                "/translate",
                json={"text": "hi", "source": "en", "target": "ar"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            self.assertEqual(res.status_code, 401)

            # WebSocket without auth -> rejected
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws/v1/translate-stream") as ws:
                    pass

            # WebSocket with valid query token -> connects
            with client.websocket_connect("/ws/v1/translate-stream?token=secret-token-123") as ws:
                msg = ws.receive_json()
                self.assertEqual(msg.get("event"), "ready")

            # WebSocket with valid header -> connects
            with client.websocket_connect("/ws/v1/translate-stream", headers={"Authorization": "Bearer secret-token-123"}) as ws:
                msg = ws.receive_json()
                self.assertEqual(msg.get("event"), "ready")
        finally:
            object.__setattr__(server_mod.SETTINGS, "auth_token", orig_token)
            server_mod.PIPELINE = orig_pipe

    # ---- Multi-dialect: all 13 dialects prompt and invariants test --------
    def test_all_13_dialects_prompt_and_invariants(self):
        """Verify prompt construction, few-shot conditioning, and MSA invariants
        across all 13 regional Arabic dialects."""
        from app.languages import AR_VARIANTS
        from app.mt import _DIALECT_NAMES, _DIALECT_FAMILIES, _DIALECT_FEW_SHOTS

        self.assertEqual(len(AR_VARIANTS), 13)
        for code in AR_VARIANTS:
            self.assertIn(code, _DIALECT_NAMES)
            self.assertIn(code, _DIALECT_FAMILIES)
            name, family = _DIALECT_FAMILIES[code]
            self.assertIn(family, _DIALECT_FEW_SHOTS)

            # 1. ar -> en dialect prompt & few-shots
            msgs_en = make_translation_messages("جملة عامية", "ar", "en", source_variant=code)
            sys_en = msgs_en[0]["content"]
            self.assertIn(f"The speaker uses {name} colloquial Arabic; interpret idioms accordingly.", sys_en)
            self.assertIn("specialized in regional colloquial dialects", sys_en)
            self.assertIn("بدري", sys_en)
            self.assertNotIn("strict, literal", sys_en)

            expected_shots = _DIALECT_FEW_SHOTS[family]
            user_shots = [m["content"] for m in msgs_en if m["role"] == "user"]
            self.assertIn(expected_shots[0][0], user_shots)

            # 2. ar -> fr dialect prompt
            msgs_fr = make_translation_messages("جملة عامية", "ar", "fr", source_variant=code)
            sys_fr = msgs_fr[0]["content"]
            self.assertIn(f"The speaker uses {name} colloquial Arabic; interpret idioms accordingly.", sys_fr)
            self.assertIn("specialized in regional colloquial dialects", sys_fr)

            # 3. target Arabic strictly MSA
            msgs_tgt = make_translation_messages("Hello", "en", "ar", target_variant=code)
            sys_tgt = msgs_tgt[0]["content"]
            self.assertIn("Modern Standard Arabic", sys_tgt)
            self.assertNotIn("colloquial", sys_tgt.lower())

        # Empty variant -> MSA default
        msgs_msa = make_translation_messages("كيف حالك", "ar", "en", source_variant="")
        self.assertIn("strict, literal", msgs_msa[0]["content"])

    # ---- Short-audio phantom hallucination guard & Idempotent cache tests ----
    def test_short_audio_phantom_tokens_dropped(self):
        """Whisper trailing breath / noise hallucinations ('you', 'Thank you.', 'شكرا', 'أنت')
        must be dropped on short utterances to prevent trailing phantom bubbles."""
        settings = Settings()
        engine = AsrEngine(settings)
        engine._model = MagicMock()

        # 1. "you" on short audio (0.8s) -> must be dropped
        seg_you = SimpleNamespace(
            text="you",
            start=0.0,
            end=0.8,
            no_speech_prob=0.1,
            avg_logprob=-0.3,
            compression_ratio=1.0,
        )
        engine._model.transcribe.return_value = ([seg_you], SimpleNamespace(language="en", language_probability=0.95))
        res = engine.transcribe(np.zeros(12800, dtype=np.float32), language="en")
        self.assertEqual(res.text, "", "Isolated 'you' on short audio must be dropped")
        self.assertTrue(res.hollow)

        # 2. "Thank you." on short audio (1.0s) with no_speech_prob=0.3 -> must be dropped
        seg_ty = SimpleNamespace(
            text="Thank you.",
            start=0.0,
            end=1.0,
            no_speech_prob=0.35,
            avg_logprob=-0.4,
            compression_ratio=1.0,
        )
        engine._model.transcribe.return_value = ([seg_ty], SimpleNamespace(language="en", language_probability=0.95))
        res = engine.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        self.assertEqual(res.text, "", "Trailing noise 'Thank you.' must be dropped")
        self.assertTrue(res.hollow)

        # 3. "شكرا" on short audio (0.9s) with no_speech_prob=0.3 -> must be dropped
        seg_shukran = SimpleNamespace(
            text="شكرا",
            start=0.0,
            end=0.9,
            no_speech_prob=0.30,
            avg_logprob=-0.4,
            compression_ratio=1.0,
        )
        engine._model.transcribe.return_value = ([seg_shukran], SimpleNamespace(language="ar", language_probability=0.98))
        res = engine.transcribe(np.zeros(14400, dtype=np.float32), language="ar")
        self.assertEqual(res.text, "", "Trailing noise 'شكرا' must be dropped")
        self.assertTrue(res.hollow)

        # 4. "أنت" on short audio (0.7s) -> must be dropped
        seg_anta = SimpleNamespace(
            text="أنت",
            start=0.0,
            end=0.7,
            no_speech_prob=0.1,
            avg_logprob=-0.3,
            compression_ratio=1.0,
        )
        engine._model.transcribe.return_value = ([seg_anta], SimpleNamespace(language="ar", language_probability=0.98))
        res = engine.transcribe(np.zeros(11200, dtype=np.float32), language="ar")
        self.assertEqual(res.text, "", "Isolated 'أنت' on short audio must be dropped")
        self.assertTrue(res.hollow)

        # 5. Legitimate sentence -> must be kept
        seg_legit = SimpleNamespace(
            text="Hello, how are you doing today?",
            start=0.0,
            end=2.2,
            no_speech_prob=0.01,
            avg_logprob=-0.2,
            compression_ratio=1.0,
        )
        engine._model.transcribe.return_value = ([seg_legit], SimpleNamespace(language="en", language_probability=0.99))
        res = engine.transcribe(np.zeros(35200, dtype=np.float32), language="en")
        self.assertEqual(res.text, "Hello, how are you doing today?")
        self.assertFalse(res.hollow)

    def test_cache_clearing_idempotent_on_repeated_config(self):
        """Repeated config frames with same dialect/languages must NOT flush mt_cache."""
        settings = Settings()
        ws = DummyWebSocket()
        state = _StreamState(settings, ws)

        # First config sets source_variant
        state.apply({"source_variant": "SY"})
        self.assertEqual(state.source_variant, "SY")

        # Seed cache
        state.mt_cache["key1"] = "cached_val"
        self.assertIn("key1", state.mt_cache)

        # Identical config frame re-sent by client -> cache must be preserved!
        state.apply({"source_variant": "SY"})
        self.assertIn("key1", state.mt_cache, "Identical config must not clear mt_cache")

        # Config with DIFFERENT dialect -> cache must be cleared
        state.apply({"source_variant": "EG"})
        self.assertEqual(state.source_variant, "EG")
        self.assertNotIn("key1", state.mt_cache, "Changed dialect must clear mt_cache")


if __name__ == "__main__":
    unittest.main()

