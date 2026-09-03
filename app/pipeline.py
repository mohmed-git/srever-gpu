"""The translation pipeline: audio -> ASR -> MT -> one unified result.

Latency accounting rule used throughout: `total_server_ms` is measured with a
single wall clock spanning the whole server-side handling, and the stage
figures (`decode_ms`, `queue_ms`, `asr_ms`, `mt_ms`) are reported alongside it.
The stages will not sum exactly to the total, and the response says so via
`unaccounted_ms` rather than quietly reconciling them -- a total that always
equals the sum of its parts is a total that has been massaged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import audio as audio_mod
from . import languages as lang_mod
from .asr import AsrEngine, AsrResult
from .config import Settings
from .metrics import Metrics
from .mt import MtEngine, build_mt_engine, is_translatable
from .scheduler import BatchScheduler, Overloaded
from .sentences import split_sentences

log = logging.getLogger("lingua.pipeline")


class RequestError(ValueError):
    """A client-side problem: bad language code, unusable audio, empty text."""

    def __init__(self, message: str, *, code: str = "bad_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class TranslationOutcome:
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    asr_ms: float | None
    mt_ms: float
    total_server_ms: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "asr_ms": self.asr_ms,
            "mt_ms": self.mt_ms,
            "total_server_ms": self.total_server_ms,
        }
        payload.update(self.detail)
        return payload


class Pipeline:
    """Owns the engines and their schedulers. One instance per process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metrics = Metrics(window=settings.metrics_window, budget_ms=settings.latency_budget_ms)
        self.asr = AsrEngine(settings)
        self.mt: MtEngine = build_mt_engine(settings)
        self.started_at: float | None = None
        self.load_error: str | None = None
        self.warmup_report: dict[str, Any] = {}

        self._asr_sched: BatchScheduler[dict[str, Any], AsrResult] = BatchScheduler(
            "asr",
            self._run_asr_batch,
            max_batch=settings.asr_batch_size,
            wait_ms=settings.asr_batch_wait_ms,
            max_queue_depth=settings.max_queue_depth,
            budget_ms=settings.latency_budget_ms,
            overload_factor=settings.overload_factor,
            admission_enabled=settings.admission_enabled,
            reject_over_budget=settings.reject_over_budget,
        )
        self._mt_sched: BatchScheduler[tuple[str, str, str], Any] = BatchScheduler(
            "mt",
            self._run_mt_batch,
            max_batch=settings.mt_batch_size,
            wait_ms=settings.mt_batch_wait_ms,
            max_queue_depth=settings.max_queue_depth,
            budget_ms=settings.latency_budget_ms,
            overload_factor=settings.overload_factor,
            admission_enabled=settings.admission_enabled,
            reject_over_budget=settings.reject_over_budget,
        )

    # ---- lifecycle -----------------------------------------------------
    async def start(self) -> None:
        loop_start = time.perf_counter()
        try:
            await asyncio.to_thread(self.asr.load)
            await asyncio.to_thread(self.mt.load)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.error("engine load failed: %s", self.load_error)
            raise
        await self._asr_sched.start()
        await self._mt_sched.start()
        if self.settings.warmup:
            self.warmup_report = {
                "asr": await asyncio.to_thread(self.asr.warmup),
                "mt": await asyncio.to_thread(self.mt.warmup),
            }
        self.started_at = time.time()
        log.info("pipeline ready in %.2fs", time.perf_counter() - loop_start)

    async def stop(self) -> None:
        await self._asr_sched.stop()
        await self._mt_sched.stop()

    @property
    def ready(self) -> bool:
        return self.asr.ready and self.mt.ready and self.started_at is not None

    # ---- batch runners (blocking, called in worker threads) ------------
    def _run_asr_batch(self, items: list[dict[str, Any]]) -> list[AsrResult]:
        # faster-whisper's batched pipeline batches *within* one utterance, not
        # across utterances, so a cross-request batch is executed item by item.
        # Reporting a shared span here would understate per-request latency, so
        # each item is timed on its own.
        results: list[AsrResult] = []
        for item in items:
            results.append(
                self.asr.transcribe(
                    item["samples"],
                    language=item["language"],
                    batch_size=self.settings.asr_batch_size,
                )
            )
        return results

    def _run_mt_batch(self, items: list[tuple[str, str, str]]) -> list[Any]:
        return self.mt.translate_batch(items)

    # ---- validation ----------------------------------------------------
    def resolve_languages(self, source: str | None, target: str | None) -> tuple[str, str]:
        src = lang_mod.normalise(source) if source else "auto"
        dst = lang_mod.normalise(target)
        if src is None:
            raise RequestError(
                f"unsupported source language: {source!r}; see GET /languages",
                code="unsupported_source",
            )
        if dst is None:
            raise RequestError(
                f"unsupported target language: {target!r}; see GET /languages",
                code="unsupported_target",
            )
        if dst == "auto":
            raise RequestError("target language cannot be 'auto'", code="unsupported_target")
        supports = getattr(self.mt, "supports", None)
        if callable(supports):
            for code, role in ((src, "source"), (dst, "target")):
                if code != "auto" and not supports(code):
                    raise RequestError(
                        (
                            f"the loaded MT model ({self.mt.info().get('model')}) has no "
                            f"token for {role} language {code!r}: it would emit plausible "
                            f"garbage rather than fail, so the request is refused"
                        ),
                        code=f"unsupported_{role}",
                    )
        return src, dst

    # ---- text translation ---------------------------------------------
    async def translate_text(
        self, text: str, source: str | None, target: str
    ) -> TranslationOutcome:
        wall_start = time.perf_counter()
        self.metrics.enter()
        try:
            src, dst = self.resolve_languages(source, target)
            if src == "auto":
                raise RequestError(
                    "source language must be explicit for text translation "
                    "(there is no audio to detect it from)",
                    code="unsupported_source",
                )
            if not is_translatable(text):
                raise RequestError(
                    "text contains no word characters: nothing to translate "
                    "(an engine given blank input returns a hallucination, not a blank)",
                    code="empty_text",
                )
            mt_result, mt_timing = await self._mt_sched.submit((text, src, dst))
            total_ms = (time.perf_counter() - wall_start) * 1000.0

            self.metrics.observe_many(
                {
                    "mt_ms": mt_result.mt_ms,
                    "mt_queue_ms": mt_timing["queue_wait_ms"],
                    "total_server_ms": total_ms,
                }
            )
            self.metrics.incr("requests_completed")
            self.metrics.incr("rest_requests")
            if mt_result.hollow:
                self.metrics.incr("hollow_results")

            accounted = mt_result.mt_ms + mt_timing["queue_wait_ms"]
            return TranslationOutcome(
                original_text=text,
                translated_text=mt_result.text,
                source_lang=src,
                target_lang=dst,
                asr_ms=None,
                mt_ms=mt_result.mt_ms,
                total_server_ms=round(total_ms, 2),
                detail={
                    "asr_ms_note": "no audio in this request: ASR was not run",
                    "queue_ms": mt_timing["queue_wait_ms"],
                    "mt_batch_size": mt_timing["batch_size"],
                    "mt_backend": mt_result.backend,
                    "mt_model": mt_result.model,
                    "output_tokens": mt_result.output_tokens,
                    "hollow": mt_result.hollow,
                    "hollow_reason": mt_result.hollow_reason,
                    "unaccounted_ms": round(max(0.0, total_ms - accounted), 2),
                    "within_budget": total_ms <= self.settings.latency_budget_ms,
                    "latency_budget_ms": self.settings.latency_budget_ms,
                    "rtl": lang_mod.is_rtl(dst),
                },
            )
        except Overloaded:
            self.metrics.incr("rejected_overload")
            raise
        except RequestError:
            self.metrics.incr("rejected_bad_request")
            raise
        except Exception:
            self.metrics.incr("errors")
            raise
        finally:
            self.metrics.leave()

    # ---- audio translation --------------------------------------------
    async def translate_audio(
        self,
        raw: bytes,
        *,
        source: str | None,
        target: str,
        declared_format: str | None = None,
        input_sample_rate: int | None = None,
        channels: int = 1,
    ) -> TranslationOutcome:
        wall_start = time.perf_counter()
        self.metrics.enter()
        try:
            src, dst = self.resolve_languages(source, target)

            decode_start = time.perf_counter()
            try:
                decoded = await asyncio.to_thread(
                    audio_mod.decode,
                    raw,
                    declared_format=declared_format,
                    sample_rate=input_sample_rate or self.settings.sample_rate,
                    channels=channels,
                )
            except audio_mod.AudioError as exc:
                raise RequestError(str(exc), code="bad_audio") from exc
            decode_ms = (time.perf_counter() - decode_start) * 1000.0

            if decoded.duration_s < self.settings.min_utterance_seconds:
                raise RequestError(
                    f"audio is {decoded.duration_s:.3f}s, shorter than the "
                    f"{self.settings.min_utterance_seconds}s minimum",
                    code="audio_too_short",
                )
            if decoded.duration_s > self.settings.max_utterance_seconds:
                raise RequestError(
                    f"audio is {decoded.duration_s:.1f}s, longer than the "
                    f"{self.settings.max_utterance_seconds}s maximum for one turn",
                    code="audio_too_long",
                )
            if decoded.looks_silent:
                raise RequestError(
                    (
                        f"audio is effectively silent (peak {decoded.peak:.2e}); "
                        "transcribing it would produce a hollow result whose latency "
                        "measures rejection, not translation"
                    ),
                    code="silent_audio",
                )

            asr_result, asr_timing = await self._asr_sched.submit(
                {"samples": decoded.samples, "language": None if src == "auto" else src}
            )

            detected = asr_result.language if src == "auto" else src
            if src == "auto":
                normalised = lang_mod.normalise(detected)
                if normalised is None or normalised == "auto":
                    raise RequestError(
                        f"language detection returned {detected!r}, which this server "
                        "cannot route",
                        code="detection_failed",
                    )
                detected = normalised

            if asr_result.hollow:
                total_ms = (time.perf_counter() - wall_start) * 1000.0
                self.metrics.incr("hollow_results")
                self.metrics.observe("asr_ms", asr_result.asr_ms)
                self.metrics.observe("total_server_ms", total_ms)
                self.metrics.incr("requests_completed")
                return TranslationOutcome(
                    original_text="",
                    translated_text="",
                    source_lang=detected,
                    target_lang=dst,
                    asr_ms=asr_result.asr_ms,
                    mt_ms=0.0,
                    total_server_ms=round(total_ms, 2),
                    detail={
                        "hollow": True,
                        "hollow_reason": asr_result.hollow_reason,
                        "mt_ms_note": "MT was not run: there was no transcript to translate",
                        "decode_ms": round(decode_ms, 2),
                        "asr_segments": asr_result.segments,
                        "audio_seconds": decoded.duration_s,
                        "audio_peak": round(decoded.peak, 5),
                        "queue_ms": asr_timing["queue_wait_ms"],
                        "within_budget": total_ms <= self.settings.latency_budget_ms,
                        "latency_budget_ms": self.settings.latency_budget_ms,
                    },
                )

            if detected == dst:
                # Same language in and out: translating would waste the budget
                # and risk paraphrasing the user's own words back at them.
                total_ms = (time.perf_counter() - wall_start) * 1000.0
                self.metrics.observe_many(
                    {"asr_ms": asr_result.asr_ms, "total_server_ms": total_ms}
                )
                self.metrics.incr("requests_completed")
                self.metrics.incr("passthrough")
                return TranslationOutcome(
                    original_text=asr_result.text,
                    translated_text=asr_result.text,
                    source_lang=detected,
                    target_lang=dst,
                    asr_ms=asr_result.asr_ms,
                    mt_ms=0.0,
                    total_server_ms=round(total_ms, 2),
                    detail={
                        "passthrough": True,
                        "passthrough_reason": "source and target are the same language",
                        "mt_ms_note": "MT skipped",
                        "decode_ms": round(decode_ms, 2),
                        "audio_seconds": decoded.duration_s,
                        "queue_ms": asr_timing["queue_wait_ms"],
                        "within_budget": total_ms <= self.settings.latency_budget_ms,
                        "latency_budget_ms": self.settings.latency_budget_ms,
                        "rtl": lang_mod.is_rtl(dst),
                    },
                )

            mt_result, mt_timing = await self._mt_sched.submit((asr_result.text, detected, dst))
            total_ms = (time.perf_counter() - wall_start) * 1000.0

            queue_ms = asr_timing["queue_wait_ms"] + mt_timing["queue_wait_ms"]
            accounted = decode_ms + asr_result.asr_ms + mt_result.mt_ms + queue_ms

            self.metrics.observe_many(
                {
                    "decode_ms": decode_ms,
                    "asr_ms": asr_result.asr_ms,
                    "mt_ms": mt_result.mt_ms,
                    "asr_queue_ms": asr_timing["queue_wait_ms"],
                    "mt_queue_ms": mt_timing["queue_wait_ms"],
                    "total_server_ms": total_ms,
                }
            )
            self.metrics.incr("requests_completed")
            self.metrics.incr("audio_requests")
            if mt_result.hollow:
                self.metrics.incr("hollow_results")

            return TranslationOutcome(
                original_text=asr_result.text,
                translated_text=mt_result.text,
                source_lang=detected,
                target_lang=dst,
                asr_ms=asr_result.asr_ms,
                mt_ms=mt_result.mt_ms,
                total_server_ms=round(total_ms, 2),
                detail={
                    "decode_ms": round(decode_ms, 2),
                    "queue_ms": round(queue_ms, 2),
                    "unaccounted_ms": round(max(0.0, total_ms - accounted), 2),
                    "unaccounted_note": (
                        "wall clock minus the sum of measured stages: event-loop "
                        "scheduling and serialisation, not attributed to any stage"
                    ),
                    "audio_seconds": decoded.duration_s,
                    "audio_format": decoded.source_format,
                    "real_time_factor": (
                        round(total_ms / (decoded.duration_s * 1000.0), 3)
                        if decoded.duration_s > 0
                        else None
                    ),
                    "asr_model": asr_result.model,
                    "asr_segments": asr_result.segments,
                    "language_detected": src == "auto",
                    "language_probability": asr_result.language_probability,
                    "mt_backend": mt_result.backend,
                    "mt_model": mt_result.model,
                    "mt_batch_size": mt_timing["batch_size"],
                    "output_tokens": mt_result.output_tokens,
                    "hollow": mt_result.hollow,
                    "hollow_reason": mt_result.hollow_reason,
                    "within_budget": total_ms <= self.settings.latency_budget_ms,
                    "latency_budget_ms": self.settings.latency_budget_ms,
                    "rtl": lang_mod.is_rtl(dst),
                },
            )
        except Overloaded:
            self.metrics.incr("rejected_overload")
            raise
        except RequestError:
            self.metrics.incr("rejected_bad_request")
            raise
        except Exception:
            self.metrics.incr("errors")
            raise
        finally:
            self.metrics.leave()

    # ---- shared front half of the audio path ---------------------------
    # Extracted so the streaming path and the unified path cannot drift: if
    # only one of them enforced the silence or duration checks, the other
    # would happily transcribe noise and report a plausible latency for it.
    async def _decode_checked(
        self,
        raw: bytes,
        *,
        declared_format: str | None,
        input_sample_rate: int | None,
        channels: int,
    ) -> tuple[Any, float]:
        started = time.perf_counter()
        try:
            decoded = await asyncio.to_thread(
                audio_mod.decode,
                raw,
                declared_format=declared_format,
                sample_rate=input_sample_rate or self.settings.sample_rate,
                channels=channels,
            )
        except audio_mod.AudioError as exc:
            raise RequestError(str(exc), code="bad_audio") from exc
        decode_ms = (time.perf_counter() - started) * 1000.0

        if decoded.duration_s < self.settings.min_utterance_seconds:
            raise RequestError(
                f"audio is {decoded.duration_s:.3f}s, shorter than the "
                f"{self.settings.min_utterance_seconds}s minimum",
                code="audio_too_short",
            )
        if decoded.duration_s > self.settings.max_utterance_seconds:
            raise RequestError(
                f"audio is {decoded.duration_s:.1f}s, longer than the "
                f"{self.settings.max_utterance_seconds}s maximum for one turn",
                code="audio_too_long",
            )
        if decoded.looks_silent:
            raise RequestError(
                f"audio is effectively silent (peak {decoded.peak:.2e}); "
                "transcribing it would produce a hollow result whose latency "
                "measures rejection, not translation",
                code="silent_audio",
            )
        return decoded, decode_ms

    def _resolve_detected(self, src: str, asr_result: AsrResult) -> str:
        """Normalise the ASR-detected language, or refuse to route it."""
        if src != "auto":
            return src
        normalised = lang_mod.normalise(asr_result.language)
        if normalised is None or normalised == "auto":
            raise RequestError(
                f"language detection returned {asr_result.language!r}, which "
                "this server cannot route",
                code="detection_failed",
            )
        return normalised

    # ---- streaming (sentence-by-sentence) ------------------------------
    async def translate_audio_streaming(
        self,
        raw: bytes,
        *,
        source: str | None,
        target: str,
        declared_format: str | None = None,
        input_sample_rate: int | None = None,
        channels: int = 1,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield one frame per sentence, then a final summary frame.

        WHY: MEASURED in the existing app, waiting for the whole utterance
        before any audio plays costs ~595 ms before the first audible word.
        Emitting sentence 1 as soon as it is translated lets the earbuds start
        speaking while sentences 2..n are still in flight.

        WHAT THIS DOES NOT DO: it does not reduce `total_server_ms`. More,
        smaller MT calls carry slightly MORE total overhead than one large
        call. The gain is entirely in time-to-first-word, so that is reported
        separately as `first_sentence_ms` and never folded into the total --
        one figure covering both would hide which of them actually improved.

        Single-sentence utterances take the unsplit path: there is nothing to
        overlap, and splitting would add overhead for no benefit.
        """
        wall_start = time.perf_counter()
        self.metrics.enter()
        try:
            src, dst = self.resolve_languages(source, target)
            decoded, decode_ms = await self._decode_checked(
                raw,
                declared_format=declared_format,
                input_sample_rate=input_sample_rate,
                channels=channels,
            )

            asr_result, asr_timing = await self._asr_sched.submit(
                {"samples": decoded.samples, "language": None if src == "auto" else src}
            )
            detected = self._resolve_detected(src, asr_result)

            # Hollow ASR: no transcript, so there is nothing to stream. Emit a
            # final frame saying so rather than an empty stream, which a client
            # could not distinguish from a dropped connection.
            if asr_result.hollow:
                total_ms = (time.perf_counter() - wall_start) * 1000.0
                self.metrics.incr("hollow_results")
                self.metrics.observe_many(
                    {"asr_ms": asr_result.asr_ms, "total_server_ms": total_ms}
                )
                self.metrics.incr("requests_completed")
                yield {
                    "type": "final",
                    "original_text": "",
                    "translated_text": "",
                    "source_lang": detected,
                    "target_lang": dst,
                    "asr_ms": asr_result.asr_ms,
                    "mt_ms": 0.0,
                    "total_server_ms": round(total_ms, 2),
                    "sentence_count": 0,
                    "streamed": False,
                    "first_sentence_ms": None,
                    "first_sentence_ms_note": (
                        "no sentence was produced: ASR returned nothing to translate"
                    ),
                    "hollow": True,
                    "hollow_reason": asr_result.hollow_reason,
                }
                return

            split = split_sentences(
                asr_result.text,
                min_chars=self.settings.sentence_min_chars,
                max_sentences=self.settings.sentence_max_count,
                enabled=self.settings.sentence_streaming,
            )
            sentences = list(split.sentences) or [asr_result.text]
            translated: list[str] = []
            mt_total_ms = 0.0
            first_sentence_ms: float | None = None
            hollow_reasons: list[str] = []

            mt_tasks: list[asyncio.Task] = []
            try:
                for index, sentence in enumerate(sentences):
                    if passthrough:
                        # Same language in and out: no MT call, but still streamed
                        # per sentence so the client's playback path is identical.
                        piece, piece_ms = sentence, 0.0
                        piece_hollow, piece_reason = False, None
                        mt_backend, out_tokens = None, None
                    elif index == 0:
                        # Sentence 0 is submitted and awaited immediately so Early Dispatch
                        # delivers the first speakable word without waiting for sentences 1..n!
                        mt_result, mt_timing = await self._mt_sched.submit(
                            (sentence, detected, dst)
                        )
                        piece, piece_ms = mt_result.text, mt_result.mt_ms
                        piece_hollow, piece_reason = mt_result.hollow, mt_result.hollow_reason
                        mt_backend, out_tokens = mt_result.backend, mt_result.output_tokens
                        mt_total_ms += mt_result.mt_ms
                        self.metrics.observe("mt_ms", mt_result.mt_ms)
                        self.metrics.observe("mt_queue_ms", mt_timing["queue_wait_ms"])

                        # Launch remaining sentences (1..n) concurrently while sentence 0 is yielded
                        if len(sentences) > 1:
                            for s in sentences[1:]:
                                mt_tasks.append(
                                    asyncio.create_task(self._mt_sched.submit((s, detected, dst)))
                                )
                    else:
                        task_idx = index - 1
                        mt_result, mt_timing = await mt_tasks[task_idx]
                        piece, piece_ms = mt_result.text, mt_result.mt_ms
                        piece_hollow, piece_reason = mt_result.hollow, mt_result.hollow_reason
                        mt_backend, out_tokens = mt_result.backend, mt_result.output_tokens
                        mt_total_ms += mt_result.mt_ms
                        self.metrics.observe("mt_ms", mt_result.mt_ms)
                        self.metrics.observe("mt_queue_ms", mt_timing["queue_wait_ms"])

                    translated.append(piece)
                    if piece_hollow and piece_reason:
                        hollow_reasons.append(f"sentence {index + 1}: {piece_reason}")

                    elapsed = (time.perf_counter() - wall_start) * 1000.0
                    if index == 0:
                        first_sentence_ms = elapsed
                        self.metrics.observe("first_sentence_ms", elapsed)

                    yield {
                        "type": "sentence",
                        "index": index,
                        "sentence_count": len(sentences),
                        "is_last": index == len(sentences) - 1,
                        "original_text": sentence,
                        "translated_text": piece,
                        "source_lang": detected,
                        "target_lang": dst,
                        # Cumulative wall clock at the moment this sentence became
                        # speakable. This -- not mt_ms -- is what the user perceives.
                        "elapsed_ms": round(elapsed, 2),
                        "mt_ms": round(piece_ms, 2),
                        "mt_backend": mt_backend,
                        "output_tokens": out_tokens,
                        "hollow": piece_hollow,
                        "hollow_reason": piece_reason,
                        "rtl": lang_mod.is_rtl(dst),
                    }
            finally:
                # Task leak guard: cancel any remaining MT tasks if an exception or disconnect occurs
                for t in mt_tasks:
                    if not t.done():
                        t.cancel()

            total_ms = (time.perf_counter() - wall_start) * 1000.0
            self.metrics.observe_many(
                {
                    "decode_ms": decode_ms,
                    "asr_ms": asr_result.asr_ms,
                    "asr_queue_ms": asr_timing["queue_wait_ms"],
                    "total_server_ms": total_ms,
                }
            )
            self.metrics.incr("requests_completed")
            self.metrics.incr("audio_requests")
            self.metrics.incr("streamed_requests")
            if hollow_reasons:
                self.metrics.incr("hollow_results")

            multi = len(sentences) > 1
            yield {
                "type": "final",
                "original_text": asr_result.text,
                "translated_text": " ".join(t for t in translated if t).strip(),
                "source_lang": detected,
                "target_lang": dst,
                "asr_ms": asr_result.asr_ms,
                "mt_ms": round(mt_total_ms, 2),
                "total_server_ms": round(total_ms, 2),
                "sentence_count": len(sentences),
                "streamed": multi,
                "split_reason": split.reason,
                "first_sentence_ms": (
                    round(first_sentence_ms, 2) if first_sentence_ms is not None else None
                ),
                "time_saved_to_first_word_ms": (
                    round(total_ms - first_sentence_ms, 2)
                    if multi and first_sentence_ms is not None
                    else 0.0
                ),
                "time_saved_note": (
                    "total_server_ms minus first_sentence_ms: how much sooner "
                    "playback can start. It does NOT mean the server got faster "
                    "-- total_server_ms is unchanged or slightly worse, because "
                    "n small MT calls cost more overhead than one large one."
                    if multi
                    else "single sentence: nothing to overlap, so no saving exists"
                ),
                "decode_ms": round(decode_ms, 2),
                "audio_seconds": decoded.duration_s,
                "audio_format": decoded.source_format,
                # Present in the unified payload, so present here too. A test
                # caught these two missing: a client that switched to streaming
                # would have silently lost both, which is worse than a mode
                # that refuses to work, because nothing would look broken.
                "real_time_factor": (
                    round(total_ms / (decoded.duration_s * 1000.0), 3)
                    if decoded.duration_s > 0
                    else None
                ),
                "language_detected": src == "auto",
                "passthrough": passthrough,
                "hollow": bool(hollow_reasons),
                "hollow_reason": "; ".join(hollow_reasons) or None,
                "within_budget": total_ms <= self.settings.latency_budget_ms,
                "first_sentence_within_budget": (
                    first_sentence_ms is not None
                    and first_sentence_ms <= self.settings.latency_budget_ms
                ),
                "latency_budget_ms": self.settings.latency_budget_ms,
                "rtl": lang_mod.is_rtl(dst),
            }
        except Overloaded:
            self.metrics.incr("rejected_overload")
            raise
        except RequestError:
            self.metrics.incr("rejected_bad_request")
            raise
        except Exception:
            self.metrics.incr("errors")
            raise
        finally:
            self.metrics.leave()

    # ---- reporting -----------------------------------------------------
    def scheduler_stats(self) -> dict[str, Any]:
        return {"asr": self._asr_sched.stats(), "mt": self._mt_sched.stats()}

    def engine_info(self) -> dict[str, Any]:
        return {"asr": self.asr.info(), "mt": self.mt.info()}
