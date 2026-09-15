import os
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.pipeline import Pipeline
from app.asr import AsrResult
from app.mt import MtResult

class FakeAsrForLID:
    def __init__(self, detected_lang='ar', detected_prob=0.95):
        self.ready = True
        self._detected_lang = detected_lang
        self._detected_prob = detected_prob

    def detect_language(self, samples):
        return (self._detected_lang, self._detected_prob)

    def transcribe(self, samples, **kwargs):
        return AsrResult(
            text='hello this is test speech',
            language=self._detected_lang,
            language_probability=self._detected_prob,
            duration_s=1.0,
            asr_ms=20.0,
            segments=1,
            dropped_segments=0,
            hollow=False,
            hollow_reason=None,
            model='fake-whisper',
            compute_type='float16',
            batch_size=1,
            rms_dbfs=-40.0,
        )

class FakeMtEngine:
    def __init__(self):
        self.ready = True

    def translate_batch(self, items):
        results = []
        for item in items:
            text, src, dst = item[0], item[1], item[2]
            results.append(
                MtResult(
                    text=f'translated: {text}',
                    mt_ms=25.0,
                    backend='fake-qwen',
                    model='fake-qwen',
                    input_tokens=10,
                    output_tokens=10,
                    batch_size=len(items),
                    hollow=False,
                    hollow_reason=None,
                )
            )
        return results

class TestLIDGate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        settings = Settings(sample_rate=16000, sentence_streaming=True, warmup=False, mt_backend='qwen_hf')
        self.pipeline = Pipeline(settings)
        self.fake_asr = FakeAsrForLID('ar', 0.95)
        self.pipeline.asr = self.fake_asr
        self.pipeline.mt = FakeMtEngine()
        self.pipeline.started_at = 1.0
        await self.pipeline._asr_sched.start()
        await self.pipeline._mt_sched.start()

        # Audio PCM 1.0s
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        self.fake_pcm = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()

    async def asyncTearDown(self):
        await self.pipeline._asr_sched.stop()
        await self.pipeline._mt_sched.stop()

    async def test_lid_mismatch_blocks_mt_submit_nonstreaming(self):
        self.pipeline._mt_sched.submit = AsyncMock(wraps=self.pipeline._mt_sched.submit)
        outcome = await self.pipeline.translate_audio(
            self.fake_pcm,
            source='en',
            target='fr',
            declared_format='pcm_s16le',
            input_sample_rate=16000,
            channels=1,
        )
        self.assertTrue(outcome.detail.get('hollow', False))
        self.assertEqual(outcome.detail.get('hollow_reason'), 'lid_mismatch')
        self.assertEqual(self.pipeline._mt_sched.submit.call_count, 0, 'MT submit must NOT be called on lid_mismatch')

    async def test_lid_mismatch_blocks_mt_submit_streaming(self):
        self.pipeline._mt_sched.submit = AsyncMock(wraps=self.pipeline._mt_sched.submit)
        frames = []
        async for frame in self.pipeline.translate_audio_streaming(
            self.fake_pcm,
            source='en',
            target='fr',
            declared_format='pcm_s16le',
            input_sample_rate=16000,
            channels=1,
        ):
            frames.append(frame)

        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0].get('hollow', False))
        self.assertEqual(frames[0].get('hollow_reason'), 'lid_mismatch')
        self.assertEqual(self.pipeline._mt_sched.submit.call_count, 0, 'MT submit must NOT be called on lid_mismatch in streaming')

    async def test_lid_match_allows_mt_submit(self):
        self.fake_asr._detected_lang = 'en'
        self.pipeline._mt_sched.submit = AsyncMock(wraps=self.pipeline._mt_sched.submit)
        outcome = await self.pipeline.translate_audio(
            self.fake_pcm,
            source='en',
            target='fr',
            declared_format='pcm_s16le',
            input_sample_rate=16000,
            channels=1,
        )
        self.assertFalse(outcome.detail.get('hollow', False))
        self.assertGreater(self.pipeline._mt_sched.submit.call_count, 0, 'MT submit must be called when LID matches')

    async def test_lid_low_confidence_does_not_block(self):
        self.fake_asr._detected_lang = 'ar'
        self.fake_asr._detected_prob = 0.40 # Below 0.6 threshold
        self.pipeline._mt_sched.submit = AsyncMock(wraps=self.pipeline._mt_sched.submit)
        outcome = await self.pipeline.translate_audio(
            self.fake_pcm,
            source='en',
            target='fr',
            declared_format='pcm_s16le',
            input_sample_rate=16000,
            channels=1,
        )
        self.assertFalse(outcome.detail.get('hollow', False))
        self.assertGreater(self.pipeline._mt_sched.submit.call_count, 0, 'MT submit must be called when LID is low confidence')

if __name__ == '__main__':
    unittest.main()
