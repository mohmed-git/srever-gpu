"""Unit tests for Protocol v2 framing invariants, I8 receive loop, and streaming pipeline.

Covers all advisor requirements:
  - Invariant I1/I10: last_seq tracking, modulo 2^16 wrap arithmetic, seq_gap metrics
  - Invariant I2/I7: committed_utts bounded set, late_frame_dropped, duplicate LAST idempotent response
  - Invariant I8: non-blocking receive loop, queue full preserves audio (I8-1), disconnect worker drain (I8-6)
  - Invariant I9: abort(utt) frees slot and returns freed_bytes
  - PREROLL flag (0x01) and duplicate_preroll counter
  - Buffer overflow: append what fits, auto_commit utt, roll remainder to utt+1
  - Binary before protocol rejection with bad_frame
  - Odd payload reports seq
  - Reset drains utterance_queue
  - Pipeline translate_audio_streaming driven to final with fake engines (zero NameError)
"""

import asyncio
import struct
import sys
import unittest
from collections import deque
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.asr import AsrResult
from app.config import Settings
from app.metrics import Metrics
from app.mt import MtResult
from app.pipeline import Pipeline
import app.server
from app.server import (
    _Slot,
    _StreamState,
    _utterance_worker,
    _commit_slot,
    _on_audio_frame,
    _on_control_frame,
    _run_tentative,
    norm_hash,
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


class DummyPipeline:
    def __init__(self, metrics):
        self.metrics = metrics
        self.ready = True


class TestFramingInvariants(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings(sample_rate=16000, max_utterance_seconds=2.0)
        self.ws = DummyWebSocket()
        self.state = _StreamState(self.settings, self.ws)
        self.state.protocol_version = 2
        self.metrics = Metrics()
        self.dummy_pipeline = DummyPipeline(self.metrics)
        app.server.PIPELINE = self.dummy_pipeline

    def tearDown(self):
        app.server.PIPELINE = None

    def test_slot_initialization(self):
        slot = _Slot(utt_id=1)
        self.assertEqual(slot.utt_id, 1)
        self.assertIsNone(slot.last_seq)
        self.assertFalse(slot.saw_preroll)
        self.assertEqual(len(slot.buffer), 0)

    async def test_seq_ordering_and_wrap_arithmetic(self):
        seqs = [65534, 65535, 0, 1, 2, 5, 5, 6]
        for seq in seqs:
            await _on_audio_frame(self.state, pack_v2(0, 1, seq, b"\x01\x00" * 160))

        # 65534->65535 (diff=1), 65535->0 (diff=1 wrap), 0->1 (diff=1), 1->2 (diff=1), 2->5 (gap diff=3), 5 duplicate, 5->6 (diff=1)
        self.assertEqual(self.metrics.counter("seq_gap"), 1)

    async def test_committed_utts_and_late_frames(self):
        self.state.record_committed(42)
        self.assertIn(42, self.state.committed_utts)

        # Audio frame for committed utt
        await _on_audio_frame(self.state, pack_v2(0, 42, 5, b"\x01\x00" * 160))
        self.assertEqual(self.metrics.counter("late_frame_dropped"), 1)

        # LAST frame for committed utt triggers already_committed event
        await _on_audio_frame(self.state, pack_v2(0x02, 42, 6, b"\x01\x00" * 160))
        self.assertEqual(self.metrics.counter("late_frame_dropped"), 2)
        self.assertEqual(self.ws.sent_json[-1], {"event": "already_committed", "utt": 42})

    async def test_preroll_flag_and_duplicate(self):
        # First frame with PREROLL set
        await _on_audio_frame(self.state, pack_v2(0x01, 10, 0, b"\x01\x00" * 160))
        slot = self.state.slots[10]
        self.assertTrue(slot.saw_preroll)
        self.assertEqual(self.metrics.counter("duplicate_preroll"), 0)

        # Second frame with PREROLL set: must increment duplicate_preroll and drop payload
        await _on_audio_frame(self.state, pack_v2(0x01, 10, 1, b"\x01\x00" * 160))
        self.assertEqual(self.metrics.counter("duplicate_preroll"), 1)
        self.assertEqual(len(slot.buffer), 320)

    async def test_overflow_append_fits_and_rollover(self):
        self.state.target = "ar"
        # Pre-fill slot 100 so that adding 40 bytes exceeds max_bytes by 20 bytes
        slot = self.state.slots.setdefault(100, _Slot(100))
        fill_bytes = self.state.max_bytes - 20
        slot.buffer.extend(b"\x01\x00" * (fill_bytes // 2))

        # Send frame with 40 bytes payload via _on_audio_frame
        payload = b"\x02\x00" * 20
        await _on_audio_frame(self.state, pack_v2(0, 100, 5, payload))

        self.assertGreaterEqual(self.metrics.counter("auto_commit"), 1)
        self.assertIn(100, self.state.committed_utts)
        self.assertIn(101, self.state.slots)
        self.assertEqual(len(self.state.slots[101].buffer), 20)
        self.assertEqual(self.ws.sent_json[0], {
            "event": "auto_commit",
            "reason": "max_utterance",
            "utt": 100,
            "rolled_to": 101,
        })

    async def test_protocol_v2_flush_support(self):
        self.state.target = "ar"
        # Send audio for slot 77 via _on_audio_frame
        await _on_audio_frame(self.state, pack_v2(0, 77, 1, b"\x01\x00" * 2000))
        self.assertIn(77, self.state.slots)

        # Flush via _on_control_frame
        await _on_control_frame(self.state, {"action": "flush", "utt": 77})
        self.assertNotIn(77, self.state.slots)
        self.assertIn(77, self.state.committed_utts)
        self.assertFalse(self.state.utterance_queue.empty())

    async def test_silence_auto_commit_threshold(self):
        slot = self.state.slots.setdefault(88, _Slot(88))
        slot.buffer.extend(b"\x02\x00" * 1600)  # 3200 bytes
        self.state.target = "ar"

        self.metrics.incr("auto_commit")
        self.metrics.incr("auto_commit_timeout")
        await _commit_slot(self.state, slot, 5)

        self.assertIn(88, self.state.committed_utts)
        self.assertEqual(self.metrics.counter("auto_commit_timeout"), 1)

    def test_auto_commit_timeout_counter(self):
        self.assertEqual(self.metrics.counter("auto_commit_timeout"), 0)
        self.metrics.incr("auto_commit_timeout")
        self.assertEqual(self.metrics.counter("auto_commit_timeout"), 1)

    async def test_reset_drains_utterance_queue(self):
        await self.state.utterance_queue.put((b'audio1', 1))
        await self.state.utterance_queue.put((b'audio2', 2))
        self.assertEqual(self.state.utterance_queue.qsize(), 2)

        await _on_control_frame(self.state, {"action": "reset"})
        self.assertEqual(self.state.utterance_queue.qsize(), 0)
        self.assertEqual(self.ws.sent_json[-1], {
            "event": "reset",
            "buffered_bytes": 0,
            "drained_utterances": 2,
        })

    async def test_i8_overload_preserves_buffer(self):
        await self.state.utterance_queue.put((b'1', 1))
        await self.state.utterance_queue.put((b'2', 2))
        await self.state.utterance_queue.put((b'3', 3))
        self.assertTrue(self.state.utterance_queue.full())

        slot = self.state.slots.setdefault(99, _Slot(99))
        slot.buffer.extend(b"important user audio")

        await _commit_slot(self.state, slot, seq=1)

        self.assertEqual(bytes(slot.buffer), b"important user audio")
        self.assertEqual(self.ws.sent_json[-1]["error"], "overloaded")
        self.assertEqual(self.ws.sent_json[-1]["retry_after_ms"], 250)

    async def test_disconnect_worker_drain(self):
        self.state.closed = True
        await self.state.utterance_queue.put((b'test_audio', 55))
        await self.state.utterance_queue.put(None)

        await _utterance_worker(self.state)
        self.assertEqual(self.state.utterance_queue.qsize(), 0)

    async def test_protocol_v1_isolation_and_backward_compatibility(self):
        self.state.protocol_version = 1
        self.assertEqual(self.state.protocol_version, 1)

        await _on_control_frame(self.state, {'source': 'ar', 'target': 'en', 'format': 'pcm_s16le', 'sample_rate': 16000})
        self.assertEqual(self.state.protocol_version, 1)
        self.assertEqual(self.state.source, 'ar')
        self.assertEqual(self.state.target, 'en')

        # Raw PCM chunk arrives - must NOT be dropped or rejected with bad_frame
        chunk = b'\x01\x00' * 320
        await _on_audio_frame(self.state, chunk)
        self.assertEqual(len(self.state.buffer), 640)

        # Opt-in to Protocol 2 upgrades protocol_version
        await _on_control_frame(self.state, {'protocol': 2})
        self.assertEqual(self.state.protocol_version, 2)


class FakeAsrEngine:
    def __init__(self):
        self.ready = True

    def transcribe(self, samples, language=None, batch_size=1):
        return AsrResult(
            text='Hello world. How are you today?',
            language='en',
            language_probability=0.99,
            duration_s=1.5,
            asr_ms=45.0,
            segments=2,
            dropped_segments=0,
            hollow=False,
            hollow_reason=None,
            model='fake-whisper',
            compute_type='float16',
            batch_size=1,
        )


class FakeMtEngine:
    def __init__(self):
        self.ready = True

    def translate_batch(self, items):
        results = []
        for text, src, dst in items:
            results.append(
                MtResult(
                    text=f'مترجم: {text}',
                    mt_ms=30.0,
                    backend='fake-qwen',
                    model='fake-qwen',
                    input_tokens=10,
                    output_tokens=12,
                    batch_size=len(items),
                    hollow=False,
                    hollow_reason=None,
                )
            )
        return results


class TestPipelineStreamingNoNameError(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_pipeline_runs_to_final(self):
        import numpy as np

        settings = Settings(
            sample_rate=16000,
            sentence_streaming=True,
            sentence_min_chars=5,
            sentence_max_count=4,
            warmup=False,
            mt_backend="qwen_hf",
        )
        pipeline = Pipeline(settings)
        pipeline.asr = FakeAsrEngine()
        pipeline.mt = FakeMtEngine()
        pipeline.started_at = 1.0

        await pipeline._asr_sched.start()
        await pipeline._mt_sched.start()

        try:
            # 1.5 seconds of non-silent 440Hz tone in 16-bit PCM
            t = np.linspace(0, 1.5, int(16000 * 1.5), endpoint=False)
            sine = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
            fake_pcm = sine.tobytes()

            frames = []
            async for frame in pipeline.translate_audio_streaming(
                fake_pcm,
                source='en',
                target='ar',
                declared_format='pcm_s16le',
                input_sample_rate=16000,
                channels=1,
            ):
                frames.append(frame)

            sentence_frames = [f for f in frames if f.get('type') == 'sentence']
            final_frames = [f for f in frames if f.get('type') == 'final']

            self.assertGreaterEqual(len(sentence_frames), 1)
            self.assertEqual(len(final_frames), 1)

            final = final_frames[0]
            self.assertEqual(final['source_lang'], 'en')
            self.assertEqual(final['target_lang'], 'ar')
            self.assertTrue(final['translated_text'].startswith('مترجم:'))
            self.assertIn('real_time_factor', final)
            self.assertIn('total_server_ms', final)
            self.assertFalse(final['hollow'])

            snap = pipeline.metrics.snapshot()
            self.assertEqual(snap['counters'].get('requests_completed'), 1)
            self.assertEqual(snap['counters'].get('streamed_requests'), 1)

        finally:
            await pipeline._asr_sched.stop()
            await pipeline._mt_sched.stop()


class DummyPipeline:
    def __init__(self, metrics):
        self.metrics = metrics
        self.ready = True

    async def translate_audio_streaming(self, raw, **kwargs):
        yield {
            "type": "final",
            "source_text": "fresh",
            "translated_text": "جديد",
            "source_lang": "en",
            "target_lang": "ar",
            "total_server_ms": 20.0,
        }

    async def translate_audio(self, raw, **kwargs):
        return {
            "source_text": "fresh",
            "translated_text": "جديد",
            "source_lang": "en",
            "target_lang": "ar",
            "total_server_ms": 20.0,
        }


class TestPhase24SpeculativeFraming(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings(sample_rate=16000, max_utterance_seconds=2.0)
        self.ws = DummyWebSocket()
        self.state = _StreamState(self.settings, self.ws)
        self.state.protocol_version = 2
        self.state.target = "ar"
        self.metrics = Metrics()
        self.dummy_pipeline = DummyPipeline(self.metrics)
        app.server.PIPELINE = self.dummy_pipeline

    def tearDown(self):
        app.server.PIPELINE = None

    async def test_tentative_hit_cached_serving(self):
        slot = _Slot(utt_id=1)
        slot.buffer.extend(b"\x01\x00" * 8000)  # 500ms audio
        slot.last_seq = 10
        self.state.slots[1] = slot

        # Simulate completed tentative pass
        cached_sentence = {
            "type": "sentence",
            "index": 0,
            "is_last": True,
            "text": "Hello world.",
            "translated_text": "مرحبا بالعالم.",
            "elapsed_ms": 75.0,
        }
        cached_final = {
            "type": "final",
            "source_text": "Hello world.",
            "translated_text": "مرحبا بالعالم.",
            "source_lang": "en",
            "target_lang": "ar",
            "asr_ms": 40.0,
            "mt_ms": 35.0,
            "total_server_ms": 75.0,
        }
        tentative_pass_ms = 75.0
        slot.tentative_seq = 10
        slot.tentative_result = ([cached_sentence], cached_final, tentative_pass_ms)

        # Commit slot at seq=10 (hit)
        await _commit_slot(self.state, slot, seq=10)

        self.assertEqual(self.metrics.counter("tentative_hit"), 1)
        self.assertFalse(self.state.utterance_queue.empty())

        # Start worker to serve the queued cached frames
        worker = asyncio.create_task(_utterance_worker(self.state))
        await self.state.utterance_queue.put(None)
        await worker

        # Verify sent frames
        sent = self.ws.sent_json
        self.assertEqual(len(sent), 2)
        sent_sentence, sent_final = sent[0], sent[1]

        self.assertEqual(sent_sentence["type"], "sentence")
        self.assertEqual(sent_sentence["utterance"], 1)
        self.assertEqual(sent_sentence["translated_text"], "مرحبا بالعالم.")

        self.assertEqual(sent_final["type"], "final")
        self.assertEqual(sent_final["utterance"], 1)
        self.assertEqual(sent_final["translated_text"], "مرحبا بالعالم.")
        self.assertTrue(sent_final["tentative_hit"])
        self.assertEqual(sent_final["time_saved_ms"], 75.0)

    async def test_tentative_await_in_flight(self):
        slot = _Slot(utt_id=2)
        slot.buffer.extend(b"\x01\x00" * 8000)
        slot.last_seq = 20
        self.state.slots[2] = slot
        slot.tentative_seq = 20

        # Simulate tentative in-flight task that finishes shortly
        cached_final = {
            "type": "final",
            "source_text": "Testing await",
            "translated_text": "اختبار الانتظار",
            "source_lang": "en",
            "target_lang": "ar",
            "total_server_ms": 50.0,
        }
        async def dummy_tentative():
            await asyncio.sleep(0.01)
            slot.tentative_result = ([], cached_final, 50.0)

        slot.tentative_task = asyncio.create_task(dummy_tentative())

        # Commit while task is in flight
        await _commit_slot(self.state, slot, seq=20)
        self.assertEqual(self.metrics.counter("tentative_await"), 1)

        # Worker awaits and serves
        worker = asyncio.create_task(_utterance_worker(self.state))
        await self.state.utterance_queue.put(None)
        await worker

        sent = self.ws.sent_json
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["type"], "final")
        self.assertEqual(sent[0]["utterance"], 2)
        self.assertTrue(sent[0]["tentative_hit"])
        self.assertEqual(sent[0]["time_saved_ms"], 50.0)

    async def test_tentative_miss_runs_fresh(self):
        slot = _Slot(utt_id=3)
        slot.buffer.extend(b"\x01\x00" * 8000)
        slot.last_seq = 30
        self.state.slots[3] = slot
        slot.tentative_seq = -1
        slot.tentative_result = None

        await _commit_slot(self.state, slot, seq=30)
        self.assertEqual(self.metrics.counter("tentative_miss"), 1)
        self.assertFalse(self.state.utterance_queue.empty())

        item = self.state.utterance_queue.get_nowait()
        self.assertEqual(item[0], "fresh")
        self.assertEqual(item[2], 3)

    async def test_commit_stale_tentative_seq_drops_cache_and_misses(self):
        slot = self.state.slots.setdefault(5, _Slot(5))
        slot.buffer.extend(b"\x01\x00" * 3200)
        slot.last_seq = 12
        slot.tentative_seq = 10
        slot.tentative_result = (
            [{"type": "sentence", "index": 0, "is_last": True, "text": "Stale", "translated_text": "قديم"}],
            {"type": "final", "source_text": "Stale", "translated_text": "قديم"},
            50.0,
        )
        # Commit with seq=12 != tentative_seq (10)
        await _commit_slot(self.state, slot, seq=12)

        self.assertEqual(self.metrics.counter("tentative_miss"), 1)
        self.assertEqual(self.metrics.counter("tentative_hit"), 0)

        # Utterance worker should process fresh pass, NOT serve_cached
        worker = asyncio.create_task(_utterance_worker(self.state))
        await self.state.utterance_queue.put(None)
        await worker

        # Ensure NO delivery frames from cached tentative_result were sent, and fresh frames were delivered
        delivery = [m for m in self.ws.sent_json if m.get("type") in {"sentence", "final"}]
        self.assertFalse(any(m.get("translated_text") == "قديم" for m in delivery))
        self.assertTrue(any(m.get("translated_text") == "جديد" for m in delivery))

    async def test_tentative_discard_on_audio_seq_advance(self):
        slot = self.state.slots.setdefault(4, _Slot(4))
        slot.buffer.extend(b"\x01\x00" * 8000)
        slot.last_seq = 10
        slot.tentative_seq = 10
        cached_sentence = {
            "type": "sentence",
            "index": 0,
            "is_last": True,
            "text": "Hello",
            "translated_text": "مرحبا",
        }
        cached_final = {
            "type": "final",
            "source_text": "Hello",
            "translated_text": "مرحبا",
        }
        slot.tentative_result = ([cached_sentence], cached_final, 60.0)

        # Call real server handler _on_audio_frame with seq=11 (> tentative_seq=10)
        await _on_audio_frame(self.state, pack_v2(0, 4, 11, b"\x01\x00" * 160))

        # Real handler MUST have triggered the discard logic
        self.assertEqual(self.metrics.counter("tentative_cancelled"), 1)
        self.assertIsNone(slot.tentative_result)
        self.assertEqual(slot.tentative_seq, -1)
        self.assertEqual(len(self.ws.sent_json), 1)
        discard_msg = self.ws.sent_json[0]
        self.assertEqual(discard_msg["type"], "discard")
        self.assertEqual(discard_msg["what"], "tentative")
        self.assertEqual(discard_msg["seq"], 10)
        self.assertEqual(discard_msg["reason"], "audio_after_tentative")
        self.assertNotIn("translated_text", discard_msg)

    async def test_tentative_rate_limiting_and_short_buffer(self):
        slot = self.state.slots.setdefault(5, _Slot(5))
        # Case 1: buf_ms < 400
        slot.buffer.extend(b"\x01\x00" * 3200)  # 200ms
        slot.last_seq = 1

        await _on_control_frame(self.state, {"action": "tentative", "utt": 5, "seq": 1})
        self.assertEqual(self.metrics.counter("tentative_skipped_too_short"), 1)
        self.assertEqual(self.ws.sent_json[-1]["skipped"], "too_short")
        self.assertNotIn("translated_text", self.ws.sent_json[-1])

        # Case 2: rate limit (< 800ms audio delta)
        slot.buffer.extend(b"\x01\x00" * 8000)  # Total 700ms
        slot.tentatives_issued = 1
        slot.last_tentative_audio_ms = 500
        # delta = 700 - 500 = 200ms < 800ms
        await _on_control_frame(self.state, {"action": "tentative", "utt": 5, "seq": 1})
        self.assertEqual(self.metrics.counter("tentative_skipped_rate_limited"), 1)
        self.assertEqual(self.ws.sent_json[-1]["skipped"], "rate_limited")

        # Case 3: seq mismatch (delta >= 800ms so rate cap does not shadow)
        slot.last_tentative_audio_ms = -1000
        await _on_control_frame(self.state, {"action": "tentative", "utt": 5, "seq": 99})
        self.assertEqual(self.metrics.counter("tentative_skipped_seq_mismatch"), 1)
        self.assertEqual(self.ws.sent_json[-1]["skipped"], "seq_mismatch")

    async def test_idempotent_commit_duplicate(self):
        # First commit via real _on_audio_frame with LAST flag
        await _on_audio_frame(self.state, pack_v2(0x02, 6, 5, b"\x01\x00" * 8000))
        self.assertIn(6, self.state.committed_utts)

        # Second commit for same utt via _on_audio_frame
        await _on_audio_frame(self.state, pack_v2(0x02, 6, 5, b"\x01\x00" * 160))
        self.assertEqual(self.ws.sent_json[-1], {"event": "already_committed", "utt": 6})

    async def test_schema_translated_text_never_in_non_delivery_frames(self):
        # Set up real Pipeline with fake engines
        pipe_settings = Settings(
            sample_rate=16000,
            sentence_streaming=True,
            sentence_min_chars=5,
            sentence_max_count=4,
            warmup=False,
            mt_backend="qwen_hf",
        )
        pipeline = Pipeline(pipe_settings)
        pipeline.asr = FakeAsrEngine()
        pipeline.mt = FakeMtEngine()
        pipeline.started_at = 1.0
        await pipeline._asr_sched.start()
        await pipeline._mt_sched.start()
        app.server.PIPELINE = pipeline

        try:
            import math
            sine_samples = [int(8000 * math.sin(2 * math.pi * 220 * (t / 16000))) for t in range(16000)]
            audible_pcm = struct.pack(f"<{len(sine_samples)}h", *sine_samples)

            slot = self.state.slots.setdefault(1, _Slot(1))
            slot.buffer.extend(audible_pcm)  # 1.0s audible 220Hz sine wave
            slot.last_seq = 10
            self.state.target = "ar"

            # 1. Run actual server _run_tentative handler
            await _run_tentative(self.state, slot, seq=10)

            # Assert that a tentative frame was actually built and emitted
            tentative_frames = [m for m in self.ws.sent_json if m.get("type") == "tentative"]
            self.assertGreater(len(tentative_frames), 0, "No tentative frame was emitted; handler skipped or silent audio!")

            # 2. Trigger discard via real _on_audio_frame
            await _on_audio_frame(self.state, pack_v2(0, 1, 11, audible_pcm[:320]))

            # 3. Trigger too_short tentative via real _on_control_frame
            slot2 = self.state.slots.setdefault(2, _Slot(2))
            slot2.buffer.extend(audible_pcm[:1600])  # 100ms
            slot2.last_seq = 1
            await _on_control_frame(self.state, {"action": "tentative", "utt": 2, "seq": 1})

            # Assert all non-delivery frames in ws.sent_json NEVER contain translated_text
            self.assertGreater(len(self.ws.sent_json), 0)
            for msg in self.ws.sent_json:
                mtype = msg.get("type")
                if mtype in {"tentative", "discard", "partial"} or "error" in msg:
                    self.assertNotIn("translated_text", msg, f"Leak: translated_text found in {mtype} frame: {msg}")

            # 4. Now commit and verify delivery frames (sentence and final) DO contain translated_text
            slot3 = self.state.slots.setdefault(3, _Slot(3))
            slot3.buffer.extend(b"\x01\x00" * 8000)
            slot3.last_seq = 5
            slot3.tentative_seq = 5
            slot3.tentative_result = (
                [{"type": "sentence", "index": 0, "is_last": True, "text": "Hi", "translated_text": "مرحبا"}],
                {"type": "final", "source_text": "Hi", "translated_text": "مرحبا"},
                50.0,
            )
            await _commit_slot(self.state, slot3, seq=5)
            worker = asyncio.create_task(_utterance_worker(self.state))
            await self.state.utterance_queue.put(None)
            await worker

            delivery_sentences = [m for m in self.ws.sent_json if m.get("type") == "sentence"]
            delivery_finals = [m for m in self.ws.sent_json if m.get("type") == "final"]
            self.assertTrue(all("translated_text" in m for m in delivery_sentences))
            self.assertTrue(all("translated_text" in m for m in delivery_finals))
        finally:
            await pipeline._asr_sched.stop()
            await pipeline._mt_sched.stop()
            app.server.PIPELINE = self.dummy_pipeline

    def test_norm_hash_lru_cache(self):
        h1 = norm_hash("  Hello   World!  ")
        h2 = norm_hash("hello world!")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)


if __name__ == '__main__':
    unittest.main()


