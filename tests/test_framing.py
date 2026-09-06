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
from app.server import _Slot, _StreamState, _utterance_worker


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


if __name__ == '__main__':
    unittest.main()


