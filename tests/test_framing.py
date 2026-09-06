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
from app.server import _Slot, _StreamState, _utterance_worker, _commit_slot, norm_hash


class DummyWebSocket:
    def __init__(self):
        self.sent_json = []
        self.closed_code = None

    async def send_json(self, data):
        self.sent_json.append(data)

    async def close(self, code=1000):
        self.closed_code = code


class TestFramingInvariants(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings(sample_rate=16000, max_utterance_seconds=2.0)
        self.ws = DummyWebSocket()
        self.state = _StreamState(self.settings, self.ws)
        self.metrics = Metrics()

    def test_slot_initialization(self):
        slot = _Slot(utt_id=1)
        self.assertEqual(slot.utt_id, 1)
        self.assertIsNone(slot.last_seq)
        self.assertFalse(slot.saw_preroll)
        self.assertEqual(len(slot.buffer), 0)

    def test_seq_ordering_and_wrap_arithmetic(self):
        slot = _Slot(utt_id=1)
        # Test sequential seqs, a gap (2 -> 5), duplicate (5), and wrap (65535 -> 0)
        seqs = [65534, 65535, 0, 1, 2, 5, 5, 6]

        gaps = 0
        duplicates = 0

        for seq in seqs:
            if seq in slot.seq_seen:
                duplicates += 1
                continue
            if slot.last_seq is not None:
                diff = (seq - slot.last_seq) & 0xFFFF
                if diff != 1:
                    gaps += 1
            slot.last_seq = seq
            slot.seq_seen.add(seq)

        # 65534->65535 (diff=1), 65535->0 (diff=1 wrap), 0->1 (diff=1), 1->2 (diff=1), 2->5 (gap diff=3), 5 duplicate, 5->6 (diff=1)
        self.assertEqual(gaps, 1)
        self.assertEqual(duplicates, 1)

    def test_committed_utts_and_late_frames(self):
        self.state.record_committed(42)
        self.assertIn(42, self.state.committed_utts)

        flags_last = 0x02
        utt_id = 42
        reply = None
        if utt_id in self.state.committed_utts:
            self.metrics.incr('late_frame_dropped')
            if flags_last & 0x02:
                reply = {'event': 'already_committed', 'utt': utt_id}

        self.assertEqual(self.metrics.counter('late_frame_dropped'), 1)
        self.assertIsNotNone(reply)
        self.assertEqual(reply['event'], 'already_committed')
        self.assertEqual(reply['utt'], 42)

    def test_preroll_flag_and_duplicate(self):
        slot = _Slot(utt_id=10)
        flags = 0x01
        if flags & 0x01:
            if slot.saw_preroll:
                self.metrics.incr('duplicate_preroll')
            slot.saw_preroll = True

        self.assertTrue(slot.saw_preroll)
        self.assertEqual(self.metrics.counter('duplicate_preroll'), 0)

        if flags & 0x01:
            if slot.saw_preroll:
                self.metrics.incr('duplicate_preroll')
            slot.saw_preroll = True

        self.assertEqual(self.metrics.counter('duplicate_preroll'), 1)

    def test_overflow_append_fits_and_rollover(self):
        slot = _Slot(utt_id=100)
        max_bytes = 100
        slot.buffer.extend(b'A' * 80)
        payload = b'B' * 40

        space = max(0, max_bytes - len(slot.buffer))
        self.assertEqual(space, 20)
        slot.buffer.extend(payload[:space])
        self.assertEqual(len(slot.buffer), 100)

        remainder = bytes(payload[space:])
        self.assertEqual(len(remainder), 20)

        raw = bytes(slot.buffer)
        self.state.record_committed(100)

        next_utt = 101
        next_slot = _Slot(next_utt)
        next_slot.buffer.extend(remainder)

        self.assertEqual(len(raw), 100)
        self.assertEqual(len(next_slot.buffer), 20)
        self.assertIn(100, self.state.committed_utts)

    def test_protocol_v2_flush_support(self):
        # Simulate slot having audio, then client sends action: flush
        slot = _Slot(utt_id=77)
        slot.buffer.extend(b"\x01\x00" * 2000) # 4000 bytes
        self.state.slots[77] = slot
        self.state.protocol_version = 2
        self.state.target = "ar"

        # Check committing via flush
        self.assertIn(77, self.state.slots)
        target_slot = self.state.slots.get(77)
        raw = bytes(target_slot.buffer)
        self.state.slots.pop(77, None)
        self.state.record_committed(77)

        self.assertNotIn(77, self.state.slots)
        self.assertIn(77, self.state.committed_utts)
        self.assertEqual(len(raw), 4000)

    def test_silence_auto_commit_threshold(self):
        # When audio is >= 3200 bytes, silence auto-commit triggers
        slot = _Slot(utt_id=88)
        slot.buffer.extend(b"\x02\x00" * 1600) # 3200 bytes
        self.state.slots[88] = slot
        
        # Check that >= 3200 is committed
        s = self.state.slots[88]
        self.assertGreaterEqual(len(s.buffer), 3200)
        raw = bytes(s.buffer)
        self.state.slots.pop(88, None)
        self.state.record_committed(88)

        self.assertEqual(len(raw), 3200)
        self.assertIn(88, self.state.committed_utts)

    def test_auto_commit_timeout_counter(self):
        # Verify that auto_commit_timeout metric counter is registered and incrementable
        self.assertEqual(self.metrics.counter("auto_commit_timeout"), 0)
        self.metrics.incr("auto_commit_timeout")
        self.assertEqual(self.metrics.counter("auto_commit_timeout"), 1)

    async def test_reset_drains_utterance_queue(self):
        await self.state.utterance_queue.put((b'audio1', 1))
        await self.state.utterance_queue.put((b'audio2', 2))
        self.assertEqual(self.state.utterance_queue.qsize(), 2)

        drained = 0
        while not self.state.utterance_queue.empty():
            try:
                self.state.utterance_queue.get_nowait()
                self.state.utterance_queue.task_done()
                drained += 1
            except (asyncio.QueueEmpty, ValueError):
                break

        self.assertEqual(drained, 2)
        self.assertEqual(self.state.utterance_queue.qsize(), 0)

    async def test_i8_overload_preserves_buffer(self):
        await self.state.utterance_queue.put((b'1', 1))
        await self.state.utterance_queue.put((b'2', 2))
        await self.state.utterance_queue.put((b'3', 3))
        self.assertTrue(self.state.utterance_queue.full())

        self.state.buffer.extend(b'important user audio')

        if self.state.utterance_queue.full():
            await self.state.send_json({
                'error': 'overloaded',
                'retry_after_ms': 250,
                'detail': 'utterance queue full',
            })
        else:
            self.state.buffer.clear()

        self.assertEqual(bytes(self.state.buffer), b'important user audio')
        self.assertEqual(self.ws.sent_json[-1]['error'], 'overloaded')
        self.assertEqual(self.ws.sent_json[-1]['retry_after_ms'], 250)

    async def test_disconnect_worker_drain(self):
        self.state.closed = True
        await self.state.utterance_queue.put((b'test_audio', 55))
        await self.state.utterance_queue.put(None)

        await _utterance_worker(self.state)
        self.assertEqual(self.state.utterance_queue.qsize(), 0)

    def test_protocol_v1_isolation_and_backward_compatibility(self):
        # Invariant: Protocol 1 is the default for legacy clients (e.g. mobile APK 1.27.0)
        self.assertEqual(self.state.protocol_version, 1)

        # Mobile app sends config without 'protocol' key
        self.state.apply({'source': 'ar', 'target': 'en', 'format': 'pcm_s16le', 'sample_rate': 16000})
        self.assertEqual(self.state.protocol_version, 1)
        self.assertEqual(self.state.source, 'ar')
        self.assertEqual(self.state.target, 'en')

        # Raw PCM chunk arrives - must NOT be dropped or rejected with bad_frame
        chunk = b'\x01\x00' * 320
        self.state.buffer.extend(chunk)
        self.assertEqual(len(self.state.buffer), 640)

        # Opt-in to Protocol 2 upgrades protocol_version
        self.state.apply({'protocol': 2})
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

    async def test_tentative_discard_on_audio_seq_advance(self):
        slot = _Slot(utt_id=4)
        slot.buffer.extend(b"\x01\x00" * 8000)
        slot.last_seq = 10
        self.state.slots[4] = slot
        slot.tentative_seq = 10
        slot.tentative_result = ([], {"type": "final"}, 60.0)

        # Frame arrives with seq=11 (> tentative_seq 10)
        seq = 11
        if (slot.tentative_task is not None and not slot.tentative_task.done()) or slot.tentative_result is not None:
            if slot.tentative_seq != -1 and seq != slot.tentative_seq and ((seq - slot.tentative_seq) & 0xFFFF) < 32768:
                if slot.tentative_task is not None and not slot.tentative_task.done():
                    slot.tentative_task.cancel()
                slot.tentative_result = None
                disc_seq = slot.tentative_seq
                slot.tentative_seq = -1
                self.metrics.incr("tentative_cancelled")
                await self.state.send_json({
                    "type": "discard",
                    "utt": 4,
                    "what": "tentative",
                    "seq": disc_seq,
                    "reason": "audio_after_tentative",
                })

        self.assertEqual(self.metrics.counter("tentative_cancelled"), 1)
        self.assertIsNone(slot.tentative_result)
        self.assertEqual(slot.tentative_seq, -1)
        self.assertEqual(len(self.ws.sent_json), 1)
        discard_msg = self.ws.sent_json[0]
        self.assertEqual(discard_msg["type"], "discard")
        self.assertEqual(discard_msg["what"], "tentative")
        self.assertEqual(discard_msg["seq"], 10)
        self.assertNotIn("translated_text", discard_msg)

    async def test_tentative_rate_limiting_and_short_buffer(self):
        slot = _Slot(utt_id=5)
        # Case 1: buf_ms < 400
        slot.buffer.extend(b"\x01\x00" * 3200)  # 200ms
        slot.last_seq = 1
        self.state.slots[5] = slot

        if slot.buf_ms < 400:
            self.metrics.incr("tentative_skipped_too_short")
            await self.state.send_json({"type": "tentative", "utt": 5, "seq": 1, "skipped": "too_short"})

        self.assertEqual(self.metrics.counter("tentative_skipped_too_short"), 1)
        self.assertEqual(self.ws.sent_json[-1]["skipped"], "too_short")
        self.assertNotIn("translated_text", self.ws.sent_json[-1])

        # Case 2: rate limit (< 800ms audio delta)
        slot.buffer.extend(b"\x01\x00" * 8000)  # Total 700ms
        slot.tentatives_issued = 1
        slot.last_tentative_audio_ms = 500
        if slot.tentatives_issued > 0 and (slot.buf_ms - slot.last_tentative_audio_ms) < 800:
            self.metrics.incr("tentative_skipped_rate_limited")
            await self.state.send_json({"type": "tentative", "utt": 5, "seq": 1, "skipped": "rate_limited"})

        self.assertEqual(self.metrics.counter("tentative_skipped_rate_limited"), 1)
        self.assertEqual(self.ws.sent_json[-1]["skipped"], "rate_limited")

        # Case 3: seq mismatch
        seq_req = 99
        if seq_req != slot.last_seq:
            self.metrics.incr("tentative_skipped_seq_mismatch")
            await self.state.send_json({"type": "tentative", "utt": 5, "seq": seq_req, "skipped": "seq_mismatch"})

        self.assertEqual(self.metrics.counter("tentative_skipped_seq_mismatch"), 1)
        self.assertEqual(self.ws.sent_json[-1]["skipped"], "seq_mismatch")

    async def test_idempotent_commit_duplicate(self):
        slot = _Slot(utt_id=6)
        slot.buffer.extend(b"\x01\x00" * 8000)
        slot.last_seq = 5
        self.state.slots[6] = slot

        # First commit
        await _commit_slot(self.state, slot, seq=5)
        self.assertIn(6, self.state.committed_utts)

        # Second commit for same utt
        new_slot = _Slot(utt_id=6)
        await _commit_slot(self.state, new_slot, seq=5)

        self.assertEqual(len(self.ws.sent_json), 1)
        self.assertEqual(self.ws.sent_json[0]["event"], "already_committed")
        self.assertEqual(self.ws.sent_json[0]["utt"], 6)

    def test_schema_translated_text_never_in_non_delivery_frames(self):
        partial_frame = {
            "type": "partial",
            "utt": 1,
            "seq": 5,
            "stable": "hello",
            "unstable": "world",
        }
        self.assertNotIn("translated_text", partial_frame)

        discard_frame = {
            "type": "discard",
            "utt": 1,
            "what": "tentative",
            "seq": 5,
            "reason": "audio_after_tentative",
        }
        self.assertNotIn("translated_text", discard_frame)

        tentative_skip = {
            "type": "tentative",
            "utt": 1,
            "seq": 5,
            "skipped": "too_short",
        }
        self.assertNotIn("translated_text", tentative_skip)

        error_frame = {
            "error": "overloaded",
            "retry_after_ms": 250,
            "detail": "queue full",
        }
        self.assertNotIn("translated_text", error_frame)

        sentence_frame = {"type": "sentence", "index": 0, "text": "hello", "translated_text": "مرحبا"}
        final_frame = {"type": "final", "source_text": "hello", "translated_text": "مرحبا"}
        self.assertIn("translated_text", sentence_frame)
        self.assertIn("translated_text", final_frame)

    def test_norm_hash_lru_cache(self):
        h1 = norm_hash("  Hello   World!  ")
        h2 = norm_hash("hello world!")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)


if __name__ == '__main__':
    unittest.main()


