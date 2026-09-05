"""ASR engine: faster-whisper (CTranslate2) with optional batched inference.

Two things this wrapper refuses to do:

  1. Report a latency for a transcription that produced no text. A synthetic
     tone can make Whisper return `segments=[]` in ~150 ms with `vad_filter`
     both on and off; that timing measures *rejection*, not transcription, and
     publishing it as an ASR figure is a hollow measurement. Every result
     therefore carries `hollow: bool` plus the reason, and the caller decides.
  2. Silently fall back to a different model size. If the configured model
     cannot be loaded, load() raises and the server reports the real
     exception, instead of serving a smaller model while claiming the
     configured one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings

log = logging.getLogger("lingua.asr")

_AR_HALLUCINATION_BLOCKLIST: set[str] = {
    "اشتركوا في القناة",
    "اشترك في القناة",
    "اشتركوا بالقناة",
    "اشترك بالقناة",
    "ترجمة نانسي قنقر",
    "شكرا للمشاهدة",
    "شكراً للمشاهدة",
    "شكرا على المشاهدة",
    "شكراً على المشاهدة",
    "Thank you for watching",
    "Thank you for watching.",
    "Please subscribe",
    "Subscribe to the channel",
}


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str
    language_probability: float | None
    duration_s: float
    asr_ms: float
    segments: int
    hollow: bool
    hollow_reason: str | None
    model: str
    compute_type: str
    batch_size: int
    dropped_segments: int = 0


class AsrEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.resolved_asr_model()
        self.compute_type = settings.resolved_asr_compute_type()
        self._model: Any = None
        self._batched: Any = None
        self.load_seconds: float | None = None
        self.error: str | None = None

    # ---- lifecycle -----------------------------------------------------
    def load(self) -> None:
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "device": self.settings.device,
            "compute_type": self.compute_type,
        }
        if not self.settings.on_cuda:
            kwargs["cpu_threads"] = self.settings.resolved_asr_cpu_threads()
        try:
            self._model = WhisperModel(self.model_name, **kwargs)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        # The batched pipeline is what makes concurrency worth anything on a
        # GPU: it amortises the encoder pass across simultaneous utterances.
        if self.settings.asr_batch_size > 1:
            try:
                from faster_whisper import BatchedInferencePipeline

                self._batched = BatchedInferencePipeline(model=self._model)
            except Exception as exc:  # not fatal: sequential path still works
                log.warning("batched ASR pipeline unavailable: %s", exc)
                self._batched = None
        self.load_seconds = time.perf_counter() - started
        log.info(
            "ASR ready: model=%s device=%s compute=%s batched=%s load=%.2fs",
            self.model_name,
            self.settings.device,
            self.compute_type,
            self._batched is not None,
            self.load_seconds,
        )

    @property
    def ready(self) -> bool:
        return self._model is not None

    def warmup(self) -> dict[str, Any]:
        """Run one real transcription so the first user request is not the cold one."""
        if not self.ready:
            return {"available": False, "reason": "model not loaded"}
        rng = np.random.default_rng(7)
        audio = (rng.standard_normal(self.settings.sample_rate) * 0.01).astype(np.float32)
        started = time.perf_counter()
        try:
            # Warmup deliberately ignores the hollow flag: noise is expected to
            # transcribe to nothing. The point is to page in the weights.
            self.transcribe(audio, language="en")
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return {"available": True, "warmup_ms": round((time.perf_counter() - started) * 1000.0, 2)}

    # ---- inference -----------------------------------------------------
    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        batch_size: int | None = None,
    ) -> AsrResult:
        if not self.ready:
            raise RuntimeError("ASR model not loaded")

        duration_s = samples.size / float(self.settings.sample_rate)
        effective_batch = batch_size if batch_size and batch_size > 1 else 1
        lang_arg = None if (language in (None, "auto")) else language

        options: dict[str, Any] = {
            "language": lang_arg,
            "beam_size": self.settings.asr_beam_size,
            "vad_filter": self.settings.asr_vad_filter,
            # Conversation turns are independent; carrying previous text across
            # turns is a known source of hallucinated repeats in short audio.
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }

        started = time.perf_counter()
        if self._batched is not None and effective_batch > 1 and duration_s >= 30.0:
            segments, info = self._batched.transcribe(
                samples, batch_size=effective_batch, **options
            )
        else:
            segments, info = self._model.transcribe(samples, **options)
        seg_list = list(segments)  # faster-whisper is lazy; this forces the work
        asr_ms = (time.perf_counter() - started) * 1000.0

        detected = getattr(info, "language", None) or lang_arg or "unknown"
        prob = getattr(info, "language_probability", None)

        hollow_reason: str | None = None
        dropped_count = 0
        if not seg_list:
            hollow_reason = (
                "whisper returned 0 segments: the audio was rejected (silence/VAD/noise), "
                "so this timing measures rejection, not transcription"
            )
            text = ""
        else:
            # Hallucination guard: filter per-segment using AND + compression_ratio + blocklist
            kept = [
                s for s in seg_list
                if not (getattr(s, "no_speech_prob", 0.0) > 0.6 and getattr(s, "avg_logprob", 0.0) < -1.0)
                and getattr(s, "compression_ratio", 0.0) <= 2.4
                and s.text.strip().rstrip(".!؟،, ") not in _AR_HALLUCINATION_BLOCKLIST
            ]
            dropped_count = len(seg_list) - len(kept)
            text = " ".join(s.text.strip() for s in kept if s.text).strip()
            if not text and dropped_count > 0:
                hollow_reason = f"hallucination guard dropped {dropped_count}/{len(seg_list)} segments"

        return AsrResult(
            text=text,
            language=detected,
            language_probability=round(float(prob), 4) if isinstance(prob, float) else None,
            duration_s=round(duration_s, 3),
            asr_ms=round(asr_ms, 2),
            segments=len(seg_list),
            dropped_segments=dropped_count,
            hollow=hollow_reason is not None,
            hollow_reason=hollow_reason,
            model=self.model_name,
            compute_type=self.compute_type,
            batch_size=effective_batch,
        )

    def info(self) -> dict[str, Any]:
        return {
            "engine": "faster-whisper",
            "model": self.model_name,
            "device": self.settings.device,
            "compute_type": self.compute_type,
            "beam_size": self.settings.asr_beam_size,
            "vad_filter": self.settings.asr_vad_filter,
            "batched_pipeline": self._batched is not None,
            "max_batch_size": self.settings.asr_batch_size,
            "cpu_threads": (
                None if self.settings.on_cuda else self.settings.resolved_asr_cpu_threads()
            ),
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds else None,
            "ready": self.ready,
            "error": self.error,
        }
