"""FastAPI application: WebSocket streaming + REST translation.

Endpoints
---------
WS   /ws/v1/translate-stream   audio chunks in, one unified JSON per utterance
POST /translate                {"text","source","target"} -> {"translation","latency_ms"}
GET  /health                   readiness, engines, measured latency summary
GET  /metrics                  full latency percentiles, queues, GPU/VRAM
GET  /languages                the catalogue, with ASR vs MT coverage separated
GET  /capabilities             what this build can and cannot do, with numbers
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import re
import struct
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import languages as lang_mod
from . import mt as mt_mod
from .config import SETTINGS, Settings
from .metrics import resource_report
from .pipeline import Pipeline, RequestError
from .scheduler import Overloaded
from .sentences import _TERMINATORS, split_sentences

VERSION = "1.0.0"

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lingua.server")

PIPELINE: Pipeline | None = None
STARTUP_ERROR: str | None = None


async def _background_load() -> None:
    global PIPELINE, STARTUP_ERROR
    settings = SETTINGS
    log.info(
        "starting background model load: device=%s asr=%s/%s mt_backend=%s budget=%.0fms",
        settings.device,
        settings.resolved_asr_model(),
        settings.resolved_asr_compute_type(),
        settings.mt_backend,
        settings.latency_budget_ms,
    )
    try:
        pipeline = Pipeline(settings)
        await pipeline.start()
        PIPELINE = pipeline
        log.info("background model load complete: pipeline ready to serve!")
    except Exception as exc:
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        log.error("startup failed: %s", STARTUP_ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_task = asyncio.create_task(_background_load())
    yield
    if not load_task.done():
        load_task.cancel()
    if PIPELINE is not None:
        await PIPELINE.stop()


app = FastAPI(
    title="Lingua Buds Translation Server",
    version=VERSION,
    description=(
        "Ultra-low-latency speech translation. Reports measured latency, and "
        "refuses work it cannot finish inside the configured budget rather "
        "than letting latency diverge silently."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pipeline_or_503() -> Pipeline:
    if PIPELINE is None or not PIPELINE.ready:
        raise RequestError("server is not ready", code="not_ready")
    return PIPELINE


# =====================================================================
# REST
# =====================================================================
class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source: str = Field(..., description="Source language code, e.g. 'ar'")
    target: str = Field(..., description="Target language code, e.g. 'en'")
    source_variant: str | None = Field(default=None, description="Source dialect variant, e.g. 'EG'")
    target_variant: str | None = Field(default=None, description="Target dialect variant, e.g. 'SA'")

    @field_validator("source", "target")
    @classmethod
    def _norm(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("source_variant", "target_variant")
    @classmethod
    def _norm_variant(cls, value: str | None) -> str | None:
        return lang_mod.normalise_variant(value)


@app.post("/translate")
async def translate(req: TranslateRequest, request: Request) -> JSONResponse:
    if SETTINGS.auth_token:
        auth_header = request.headers.get("authorization", "")
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if token != SETTINGS.auth_token:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": "invalid or missing bearer token"},
            )
    if PIPELINE is None or not PIPELINE.ready:
        return JSONResponse(
            status_code=503,
            content={
                "error": "not_ready",
                "detail": STARTUP_ERROR or "engines still loading",
            },
        )
    started = time.perf_counter()
    try:
        outcome = await PIPELINE.translate_text(
            req.text,
            req.source,
            req.target,
            source_variant=req.source_variant or "",
            target_variant=req.target_variant or "",
        )
    except Overloaded as exc:
        return JSONResponse(
            status_code=503,
            content=exc.payload(),
            headers={"Retry-After": str(max(1, int(exc.retry_after_s + 0.5)))},
        )
    except RequestError as exc:
        return JSONResponse(status_code=400, content={"error": exc.code, "detail": str(exc)})
    except Exception as exc:
        log.exception("translate failed")
        return JSONResponse(
            status_code=500,
            content={"error": "internal", "detail": f"{type(exc).__name__}: {exc}"},
        )

    body = outcome.to_json()
    return JSONResponse(
        content={
            # The two fields the brief asked for, first.
            "translation": outcome.translated_text,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            # Everything else is diagnostic, and deliberately visible.
            **body,
        }
    )


# =====================================================================
# WebSocket
# =====================================================================
_CONTROL_KEYS = {
    "source",
    "target",
    "source_variant",
    "target_variant",
    "format",
    "sample_rate",
    "channels",
    "action",
    # Per-connection opt-out from sentence streaming. A client that wants
    # exactly one JSON per utterance sends {"stream": false} once.
    "stream",
    "protocol",
    "utt",
    "seq",
    "text",
    "mode",
    "languages",
    "channel_map",
    "floor_dbfs",
}


def norm_hash(
    text: str,
    source: str = "",
    target: str = "",
    source_variant: str = "",
    target_variant: str = "",
) -> str:
    """Normalized hash for sentence caching in mt_cache, scoped to translation pair."""
    cleaned = " ".join(text.strip().lower().split())
    key = f"{source}:{target}:{source_variant}:{target_variant}:{cleaned}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def is_echo_match(asr_text: str, recent_tts_outputs: deque[str] | list[str]) -> bool:
    """Check if ASR transcript matches recent TTS playback output (Directive S8).
    Match condition: token Jaccard >= 0.6 or difflib.SequenceMatcher.ratio() >= 0.75.
    """
    if not asr_text or not recent_tts_outputs:
        return False

    def _normalize_words(t: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", " ", (t or "").lower(), flags=re.UNICODE)
        return [w for w in cleaned.split() if w]

    asr_words = set(_normalize_words(asr_text))
    if not asr_words:
        return False
    asr_norm = " ".join(_normalize_words(asr_text))

    for tts_text in recent_tts_outputs:
        if not tts_text:
            continue
        tts_words = set(_normalize_words(tts_text))
        if not tts_words:
            continue
        intersection = len(asr_words & tts_words)
        union = len(asr_words | tts_words)
        jaccard = (intersection / union) if union > 0 else 0.0
        if jaccard >= 0.6:
            return True
        tts_norm = " ".join(_normalize_words(tts_text))
        ratio = difflib.SequenceMatcher(None, asr_norm, tts_norm).ratio()
        if ratio >= 0.75:
            return True
    return False


class _Slot:
    """Per-utterance buffer slot for Protocol v2 framed streaming."""

    def __init__(
        self,
        utt_id: int,
        source: str | None = None,
        target: str | None = None,
        source_variant: str | None = None,
        target_variant: str | None = None,
    ) -> None:
        self.utt_id = utt_id
        self.source = source
        self.target = target
        self.source_variant = source_variant
        self.target_variant = target_variant
        self.buffer = bytearray()
        self.seq_seen: set[int] = set()
        self.last_seq: int | None = None
        self.last_audio_seq: int | None = None
        self.saw_preroll: bool = False
        self.closed: bool = False
        self.committed: bool = False
        self.created_at: float = time.monotonic()
        self.during_playback_frames: int = 0
        self.total_frames: int = 0

        # Tentative pass (Phase 2.4)
        self.tentative_task: asyncio.Task[None] | None = None
        self.tentative_seq: int = -1
        self.tentative_result: tuple[list[dict[str, Any]], dict[str, Any], float] | None = None
        self.tentative_started: float = 0.0
        self.tentatives_issued: int = 0
        self.last_tentative_audio_ms: int = 0

        # UI Partials (Phase 2.4.4)
        self.partial_task: asyncio.Task[None] | None = None
        self.partial_rev: int = 0
        self.partial_prev_words: list[str] = []
        self.last_partial_started: float = 0.0

        # Bidirectional pair_auto (Phase 3)
        self.speaker_id: int | None = None
        self.direction: str | None = None
        self.channel: str | None = None
        self.lid_low_confidence: bool = False

    @property
    def buf_ms(self) -> int:
        return len(self.buffer) // 32


class _StreamState:
    """Per-connection state: languages plus audio buffers, mt_cache, and async commit queue."""

    def __init__(self, settings: Settings, websocket: WebSocket) -> None:
        self.settings = settings
        self.websocket = websocket
        self.mode: str = "single"
        self.languages: list[str] = ["ar", "en"]
        self.channel_map: dict[str, str] = {"0": "L", "1": "R"}
        self.floor_dbfs: float | None = None
        self.source: str | None = None
        self.target: str | None = None
        self.source_variant: str | None = None
        self.target_variant: str | None = None
        self.verify_lang_next: bool = True
        self.audio_format: str | None = None
        self.sample_rate: int = settings.sample_rate
        self.channels: int = 1
        self.buffer = bytearray()
        self.utterances = 0
        # Streaming stays opt-outable per connection
        self.stream: bool = settings.sentence_streaming
        self.protocol_version: int = 1
        self.slots: dict[int, _Slot] = {}
        self.mt_cache: OrderedDict[str, Any] = OrderedDict()  # LRU 32, key = norm_hash(sentence)
        self.committed_utts: deque[int] = deque(maxlen=64)
        self.recent_tts_outputs: deque[str] = deque(maxlen=5)
        self.last_commit_time: float = 0.0
        self.last_committed_src_text: str = ""
        self.last_has_terminal_punct: bool = True
        self.send_lock = asyncio.Lock()
        self.utterance_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=3)
        self.worker_task: asyncio.Task[None] | None = None
        self.closed: bool = False

    def cache_get(self, key: str) -> Any:
        if key in self.mt_cache:
            self.mt_cache.move_to_end(key)
            return self.mt_cache[key]
        return None

    def cache_put(self, key: str, val: Any) -> None:
        if key in self.mt_cache:
            self.mt_cache.move_to_end(key)
        self.mt_cache[key] = val
        if len(self.mt_cache) > 32:
            self.mt_cache.popitem(last=False)

    def record_committed(self, utt_id: int) -> None:
        self.committed_utts.append(utt_id)

    def append_tts_output(self, text: str | None) -> None:
        if not text:
            return
        t = str(text).strip()
        if not t:
            return
        if self.recent_tts_outputs and self.recent_tts_outputs[-1] == t:
            return
        self.recent_tts_outputs.append(t)

    async def send_json(self, data: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self.send_lock:
            try:
                await self.websocket.send_json(data)
            except Exception:
                self.closed = True

    @property
    def max_bytes(self) -> int:
        # int16 mono at the declared rate.
        return int(self.settings.max_utterance_seconds * self.sample_rate * 2 * self.channels)

    def apply(self, message: dict[str, Any]) -> None:
        changed = False
        if "source" in message:
            new_source = str(message["source"]) if message["source"] is not None else None
            if new_source != self.source:
                self.source = new_source
                self.verify_lang_next = True
                changed = True
        if "target" in message:
            new_target = str(message["target"]) if message["target"] is not None else None
            if new_target != self.target:
                self.target = new_target
                self.verify_lang_next = True
                changed = True
        if "source_variant" in message:
            raw_v = message["source_variant"]
            norm_v = lang_mod.normalise_variant(raw_v)
            if norm_v != self.source_variant:
                if norm_v:
                    dialect_name = mt_mod._DIALECT_NAMES.get(norm_v, norm_v)
                    log.info("Dialect configured: code=%s, name=%s (raw=%r)", norm_v, dialect_name, raw_v)
                elif raw_v:
                    log.info("variant_unknown: %s", raw_v)
                self.source_variant = norm_v
                changed = True
        if "target_variant" in message:
            raw_v = (message.get("target_variant") or "").strip()
            if raw_v:
                log.warning("target_variant_ignored: %s (target Arabic is strictly Modern Standard Arabic)", raw_v)
            if self.target_variant != "":
                self.target_variant = ""
                changed = True

        if changed:
            self.mt_cache.clear()
        if "format" in message and message["format"]:
            self.audio_format = str(message["format"]).strip().lower()
        if "sample_rate" in message and message["sample_rate"]:
            try:
                rate = int(message["sample_rate"])
                if 8000 <= rate <= 192000:
                    self.sample_rate = rate
            except (TypeError, ValueError):
                pass
        if "channels" in message and message["channels"]:
            try:
                channels = int(message["channels"])
                if 1 <= channels <= 2:
                    self.channels = channels
            except (TypeError, ValueError):
                pass
        if "protocol" in message and message["protocol"]:
            try:
                proto = int(message["protocol"])
                if proto in {1, 2}:
                    self.protocol_version = proto
            except (TypeError, ValueError):
                pass
        if "stream" in message:
            value = message["stream"]
            if isinstance(value, bool):
                self.stream = value
            elif isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    self.stream = True
                elif lowered in {"0", "false", "no", "off"}:
                    self.stream = False
        if "mode" in message and message["mode"]:
            m_val = str(message["mode"]).strip().lower()
            if m_val != self.mode:
                self.mode = m_val
                changed = True
        if "languages" in message and isinstance(message["languages"], list):
            langs = [str(l).strip().lower() for l in message["languages"] if str(l).strip()]
            if langs and langs != self.languages:
                self.languages = langs
                changed = True
        if "channel_map" in message and isinstance(message["channel_map"], dict):
            self.channel_map = {str(k): str(v) for k, v in message["channel_map"].items()}
        if "floor_dbfs" in message and message["floor_dbfs"] is not None:
            try:
                self.floor_dbfs = float(message["floor_dbfs"])
            except (TypeError, ValueError):
                pass


async def _verify_pinned_language(
    pipeline: Any,
    state: _StreamState,
    raw: bytes,
    slot_source: str | None,
) -> dict[str, Any]:
    if not state.verify_lang_next:
        return {}
    if slot_source in (None, "auto"):
        return {}
    state.verify_lang_next = False
    try:
        decoded_tmp, _ = await pipeline._decode_checked(
            raw,
            declared_format=state.audio_format,
            input_sample_rate=state.sample_rate,
            channels=state.channels,
        )
        det = await asyncio.to_thread(pipeline.asr.detect_language, decoded_tmp.samples)
        if det is not None and isinstance(det, tuple) and len(det) >= 2:
            det_lang, det_prob = det[0], det[1]
            out = {
                "detected_lang": det_lang,
                "detected_prob": round(float(det_prob), 4),
            }
            if det_prob >= 0.6 and det_lang != slot_source:
                pipeline.metrics.incr("asr_pinned_lang_mismatch")
            return out
    except Exception as exc:
        log.warning("Verification detect_language failed: %s", exc)
    return {}


async def _run_tentative(state: _StreamState, slot: _Slot, seq: int) -> None:
    slot_id = slot.utt_id
    started = time.perf_counter()
    raw = bytes(slot.buffer)
    pipeline = PIPELINE
    if pipeline is None or not pipeline.ready:
        return

    try:
        decoded, decode_ms = await pipeline._decode_checked(
            raw,
            declared_format=state.audio_format,
            input_sample_rate=state.sample_rate,
            channels=state.channels,
        )

        is_pair_auto = (state.mode == "pair_auto" and len(state.languages) >= 2)
        if is_pair_auto:
            if slot.speaker_id is None:
                winner, conf, low_conf = await asyncio.to_thread(
                    pipeline.asr.detect_language_constrained,
                    decoded.samples,
                    state.languages,
                )
                cand0, cand1 = state.languages[0], state.languages[1]
                slot.source = winner
                slot.target = cand1 if winner == cand0 else cand0
                slot.speaker_id = 0 if winner == cand0 else 1
                slot.direction = f"{slot.source}->{slot.target}"
                slot.channel = state.channel_map.get(
                    str(slot.speaker_id), "L" if slot.speaker_id == 0 else "R"
                )
                slot.lid_low_confidence = low_conf
                if low_conf and pipeline is not None:
                    pipeline.metrics.incr("lid_low_confidence")

                # Invariant: Modern Standard Arabic (MSA) locked for Arabic target
                if slot.target == "ar":
                    slot.target_variant = None
                if slot.source == "ar":
                    slot.source_variant = state.source_variant
                else:
                    slot.source_variant = None

            src = slot.source
            dst = slot.target
            pinned_src = src
        else:
            src = slot.source or "auto"
            dst = slot.target or ""
            pinned_src = None if src == "auto" else src

        asr_result, asr_timing = await pipeline._asr_sched.submit(
            {"samples": decoded.samples, "language": pinned_src},
            priority=1,
        )

        # Adaptive SNR Floor check (Directives §1 & §4):
        if state.floor_dbfs is not None and asr_result.rms_dbfs is not None:
            if asr_result.rms_dbfs < (state.floor_dbfs + 6.0):
                asr_result = AsrResult(
                    text="",
                    language=asr_result.language,
                    language_probability=asr_result.language_probability,
                    duration_s=asr_result.duration_s,
                    asr_ms=asr_result.asr_ms,
                    segments=0,
                    dropped_segments=0,
                    hollow=True,
                    hollow_reason="snr_floor",
                    model=asr_result.model,
                    compute_type=asr_result.compute_type,
                    batch_size=asr_result.batch_size,
                    rms_dbfs=asr_result.rms_dbfs,
                )

        detected = pipeline._resolve_detected(src, asr_result)

        # Record RMS in metrics by outcome
        if pipeline is not None and asr_result.rms_dbfs is not None:
            if asr_result.hollow:
                if asr_result.hollow_reason == "silence_energy":
                    pipeline.metrics.observe_rms("silence_energy", asr_result.rms_dbfs)
                elif asr_result.hollow_reason == "vad_no_speech":
                    pipeline.metrics.observe_rms("vad_no_speech", asr_result.rms_dbfs)
                else:
                    pipeline.metrics.observe_rms("hollow_hallucination", asr_result.rms_dbfs)

        extra_pair_fields: dict[str, Any] = {}
        if is_pair_auto:
            extra_pair_fields = {
                "speaker_id": slot.speaker_id,
                "direction": slot.direction,
                "channel": slot.channel,
                "lid_low_confidence": slot.lid_low_confidence,
            }

        if asr_result.hollow or not asr_result.text.strip():
            await state.send_json({
                "type": "tentative",
                "utt": slot_id,
                "seq": seq,
                "terminal": False,
                "hollow": True,
                "hollow_reason": asr_result.hollow_reason,
                "rms_dbfs": asr_result.rms_dbfs,
                **extra_pair_fields,
            })
            return

        terminal = bool(asr_result.text.rstrip() and asr_result.text.rstrip()[-1] in _TERMINATORS)
        await state.send_json({
            "type": "tentative",
            "utt": slot_id,
            "seq": seq,
            "terminal": terminal,
            "original_text": asr_result.text,
            "asr_ms": asr_result.asr_ms,
            "rms_dbfs": asr_result.rms_dbfs,
            **extra_pair_fields,
        })

        if not dst:
            return

        split = split_sentences(
            asr_result.text,
            min_chars=state.settings.sentence_min_chars,
            max_sentences=state.settings.sentence_max_count,
            enabled=state.settings.sentence_streaming,
        )
        sentences = list(split.sentences) or [asr_result.text]
        passthrough = detected == dst

        uncached: list[str] = []
        presplit_hits = 0
        s_var = slot.source_variant or state.source_variant or ""
        t_var = slot.target_variant or state.target_variant or ""
        for s in sentences:
            h = norm_hash(s, detected, dst, s_var, t_var)
            if state.cache_get(h) is not None:
                presplit_hits += 1
            else:
                if s not in uncached and not passthrough:
                    uncached.append(s)

        if presplit_hits > 0 and pipeline is not None:
            pipeline.metrics.incr("presplit_hit", presplit_hits)
        misses = len(sentences) - presplit_hits
        if misses > 0 and pipeline is not None:
            pipeline.metrics.incr("presplit_miss", misses)

        if uncached and not passthrough:
            mt_jobs = [
                pipeline._mt_sched.submit(
                    (s, detected, dst, s_var, t_var, ""),
                    priority=1,
                )
                for s in uncached
            ]
            mt_outcomes = await asyncio.gather(*mt_jobs)
            for s, (res, _) in zip(uncached, mt_outcomes):
                state.cache_put(norm_hash(s, detected, dst, s_var, t_var), res)

        sentence_frames: list[dict[str, Any]] = []
        translated_pieces: list[str] = []
        mt_total_ms = 0.0
        hollow_reasons: list[str] = []
        first_sent_elapsed: float | None = None

        for idx, s in enumerate(sentences):
            if passthrough:
                piece, piece_ms = s, 0.0
                piece_hollow, piece_reason = False, None
                mt_backend, out_tokens = None, None
            else:
                mt_res = state.cache_get(norm_hash(s, detected, dst, s_var, t_var))
                if mt_res:
                    piece = mt_res.text
                    piece_ms = mt_res.mt_ms
                    piece_hollow = mt_res.hollow
                    piece_reason = mt_res.hollow_reason
                    mt_backend = mt_res.backend
                    out_tokens = mt_res.output_tokens
                    mt_total_ms += piece_ms
                else:
                    piece = s
                    piece_ms = 0.0
                    piece_hollow, piece_reason = False, None
                    mt_backend, out_tokens = None, None

            translated_pieces.append(piece)
            if piece_hollow and piece_reason:
                hollow_reasons.append(f"sentence {idx + 1}: {piece_reason}")

            elapsed = (time.perf_counter() - started) * 1000.0
            if idx == 0:
                first_sent_elapsed = elapsed

            sentence_frames.append({
                "type": "sentence",
                "index": idx,
                "sentence_count": len(sentences),
                "is_last": idx == len(sentences) - 1,
                "original_text": s,
                "translated_text": piece,
                "source_lang": detected,
                "target_lang": dst,
                "source_variant": slot.source_variant or None,
                "target_variant": slot.target_variant or None,
                "elapsed_ms": round(elapsed, 2),
                "mt_ms": round(piece_ms, 2),
                "mt_backend": mt_backend,
                "output_tokens": out_tokens,
                "hollow": piece_hollow,
                "hollow_reason": piece_reason,
                "rtl": lang_mod.is_rtl(dst),
                "from_tentative": True,
                "tentative_seq": seq,
                **extra_pair_fields,
            })

        tentative_pass_ms = (time.perf_counter() - started) * 1000.0
        multi = len(sentences) > 1

        final_frame = {
            "type": "final",
            "original_text": asr_result.text,
            "translated_text": " ".join(t for t in translated_pieces if t).strip(),
            "source_lang": detected,
            "target_lang": dst,
            "source_variant": slot.source_variant or None,
            "target_variant": slot.target_variant or None,
            "asr_ms": asr_result.asr_ms,
            "mt_ms": round(mt_total_ms, 2),
            "total_server_ms": round(tentative_pass_ms, 2),
            "sentence_count": len(sentences),
            "streamed": multi,
            "split_reason": split.reason,
            "first_sentence_ms": round(first_sent_elapsed, 2) if first_sent_elapsed is not None else None,
            "time_saved_to_first_word_ms": round(tentative_pass_ms - first_sent_elapsed, 2) if (multi and first_sent_elapsed is not None) else 0.0,
            "decode_ms": round(decode_ms, 2),
            "audio_seconds": decoded.duration_s,
            "audio_format": decoded.source_format,
            "real_time_factor": round(tentative_pass_ms / (decoded.duration_s * 1000.0), 3) if decoded.duration_s > 0 else None,
            "language_detected": src == "auto",
            "passthrough": passthrough,
            "hollow": bool(hollow_reasons),
            "hollow_reason": "; ".join(hollow_reasons) or None,
            "within_budget": tentative_pass_ms <= state.settings.latency_budget_ms,
            "first_sentence_within_budget": (first_sent_elapsed is not None and first_sent_elapsed <= state.settings.latency_budget_ms),
            "latency_budget_ms": state.settings.latency_budget_ms,
            "rtl": lang_mod.is_rtl(dst),
            "rms_dbfs": asr_result.rms_dbfs,
            "tentative_hit": None,
            "time_saved_ms": None,
            "presplit_hits": presplit_hits,
            **extra_pair_fields,
        }

        slot.tentative_result = (sentence_frames, final_frame, tentative_pass_ms)

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("Tentative run error for slot %d: %s", slot_id, exc)


async def _run_partial(state: _StreamState, slot: _Slot, seq: int) -> None:
    slot_id = slot.utt_id
    pipeline = PIPELINE
    if pipeline is None or not pipeline.ready:
        return
    started = time.perf_counter()
    slot.last_partial_started = started
    raw = bytes(slot.buffer)

    try:
        pipeline.metrics.incr("partials_run")
        decoded, _ = await pipeline._decode_checked(
            raw,
            declared_format=state.audio_format,
            input_sample_rate=state.sample_rate,
            channels=state.channels,
        )
        src = slot.source or "auto"
        pinned_src = None if src == "auto" else src

        asr_result, _ = await pipeline._asr_sched.submit(
            {"samples": decoded.samples, "language": pinned_src},
            priority=3,
        )
        if asr_result.hollow or not asr_result.text.strip():
            return

        words = asr_result.text.split()
        prev = slot.partial_prev_words
        match_len = 0
        for w1, w2 in zip(words, prev):
            if w1 == w2:
                match_len += 1
            else:
                break

        stable_words = words[:match_len]
        unstable_words = words[match_len:]
        slot.partial_prev_words = words

        stable_str = " ".join(stable_words)
        unstable_str = " ".join(unstable_words)
        slot.partial_rev += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detected = pipeline._resolve_detected(src, asr_result)
        await state.send_json({
            "type": "partial",
            "utt": slot_id,
            "seq": seq,
            "rev": slot.partial_rev,
            "stable": stable_str,
            "unstable": unstable_str,
            "lang": detected,
            "elapsed_ms": round(elapsed_ms, 2),
        })

        # Closed-sentence pre-translation (priority 2)
        tgt = slot.target or state.target
        if tgt and detected != tgt and len(stable_words) >= 2:
            split = split_sentences(stable_str, min_chars=10, max_sentences=4, enabled=True)
            candidate_sents = list(split.sentences)
            if len(candidate_sents) > 1:
                closed = candidate_sents[:-1]
                cs_s_var = slot.source_variant or state.source_variant or ""
                cs_t_var = slot.target_variant or state.target_variant or ""
                for cs in closed:
                    h = norm_hash(cs, detected, tgt, cs_s_var, cs_t_var)
                    if state.cache_get(h) is None:
                        try:
                            mt_res, _ = await pipeline._mt_sched.submit(
                                (
                                    cs,
                                    detected,
                                    tgt,
                                    cs_s_var,
                                    cs_t_var,
                                    "",
                                ),
                                priority=2,
                            )
                            state.cache_put(h, mt_res)
                        except Exception:
                            pass

    except asyncio.CancelledError:
        raise
    except Overloaded:
        pipeline.metrics.incr("partials_skipped")
    except Exception as exc:
        log.debug("Partial error on slot %d: %s", slot_id, exc)


async def _commit_slot(state: _StreamState, slot: _Slot, seq: int) -> None:
    utt_id = slot.utt_id
    pipeline = PIPELINE

    if utt_id in state.committed_utts:
        await state.send_json({"event": "already_committed", "utt": utt_id})
        return

    slot.closed = True
    slot.committed = True
    state.record_committed(utt_id)
    state.slots.pop(utt_id, None)
    commit_time = time.perf_counter()

    if slot.partial_task and not slot.partial_task.done():
        slot.partial_task.cancel()

    during_ratio = (slot.during_playback_frames / slot.total_frames) if slot.total_frames > 0 else 0.0

    context_prefix = ""
    if (
        state.settings.split_repair_enabled
        and state.last_commit_time > 0
        and (commit_time - state.last_commit_time) <= 0.800
        and not state.last_has_terminal_punct
        and state.last_committed_src_text
    ):
        context_prefix = state.last_committed_src_text
        if pipeline is not None:
            pipeline.metrics.incr("split_repair_count")

    # HIT: tentative_result present and tentative_seq == seq
    if slot.tentative_result is not None and slot.tentative_seq == seq:
        frames, final_frame, tentative_pass_ms = slot.tentative_result
        orig_text = final_frame.get("original_text", "")
        if during_ratio >= 0.5 and is_echo_match(orig_text, state.recent_tts_outputs):
            if pipeline is not None:
                pipeline.metrics.incr("echo_dropped")
            await state.send_json({
                "event": "dropped",
                "type": "dropped",
                "utt": utt_id,
                "utterance": utt_id,
                "reason": "echo",
                "mode": "HIT",
                "echo_dropped": True,
            })
            return

        if pipeline is not None:
            pipeline.metrics.incr("tentative_hit")
        if not state.utterance_queue.full():
            await state.utterance_queue.put(
                ("serve_cached", slot, seq, slot.tentative_result, commit_time, during_ratio, context_prefix)
            )
        else:
            await state.send_json({"error": "overloaded", "retry_after_ms": 250, "detail": "utterance queue full", "utterance": utt_id})
        return

    # AWAIT: tentative_task running and tentative_seq == seq
    if slot.tentative_task is not None and not slot.tentative_task.done() and slot.tentative_seq == seq:
        if pipeline is not None:
            pipeline.metrics.incr("tentative_await")
        if not state.utterance_queue.full():
            await state.utterance_queue.put(
                ("await_then_serve", slot, seq, slot.tentative_task, commit_time, during_ratio, context_prefix)
            )
        else:
            await state.send_json({"error": "overloaded", "retry_after_ms": 250, "detail": "utterance queue full", "utterance": utt_id})
        return

    # MISS: cancel task, enqueue fresh
    if pipeline is not None:
        pipeline.metrics.incr("tentative_miss")
    if slot.tentative_task and not slot.tentative_task.done():
        slot.tentative_task.cancel()
    raw = bytes(slot.buffer)
    if not state.utterance_queue.full():
        if state.mode == "pair_auto":
            queue_item = (
                "fresh",
                raw,
                utt_id,
                commit_time,
                slot.source,
                slot.target,
                slot.source_variant,
                slot.target_variant,
                during_ratio,
                context_prefix,
                slot.speaker_id,
                slot.direction,
                slot.channel,
                slot.lid_low_confidence,
            )
        else:
            queue_item = (
                "fresh",
                raw,
                utt_id,
                commit_time,
                slot.source,
                slot.target,
                slot.source_variant,
                slot.target_variant,
                during_ratio,
                context_prefix,
            )
        await state.utterance_queue.put(queue_item)
    else:
        await state.send_json({"error": "overloaded", "retry_after_ms": 250, "detail": "utterance queue full", "utterance": utt_id})


async def _on_control_frame(state: _StreamState, control: dict[str, Any]) -> None:
    pipeline = PIPELINE
    unknown = set(control) - _CONTROL_KEYS
    state.apply(control)
    action = str(control.get("action", "")).strip().lower()

    if action in {"commit", "flush", "end", "eou"}:
        if state.protocol_version == 2:
            utt_val = control.get("utt")
            seq_val = control.get("seq")
            target_slot = None
            if utt_val is not None:
                try:
                    target_slot = state.slots.get(int(utt_val))
                except (TypeError, ValueError):
                    pass
            if target_slot is None and state.slots:
                for s in reversed(list(state.slots.values())):
                    if s.buffer:
                        target_slot = s
                        break

            if target_slot is not None:
                slot_id = target_slot.utt_id
                if not state.target and not (state.mode == "pair_auto" and len(state.languages) >= 2):
                    await state.send_json({
                        "error": "missing_target",
                        "detail": "target language not set",
                        "utterance": slot_id,
                    })
                    return
                c_seq = int(seq_val) if seq_val is not None else (target_slot.last_audio_seq if target_slot.last_audio_seq is not None else target_slot.last_seq)
                await _commit_slot(state, target_slot, c_seq)
                return

            if utt_val is not None:
                try:
                    u_id = int(utt_val)
                    if u_id in state.committed_utts:
                        await state.send_json({"event": "already_committed", "utt": u_id})
                        return
                except (TypeError, ValueError):
                    pass

        # Legacy v1 buffer commit
        if not state.buffer:
            await state.send_json(
                {"error": "empty_utterance", "detail": "flush received but no audio was buffered"}
            )
        elif not state.target:
            await state.send_json(
                {
                    "error": "missing_target",
                    "detail": "send {'target': '<lang>'} before flushing audio",
                }
            )
            state.buffer.clear()
        else:
            if state.utterance_queue.full():
                await state.send_json({
                    "error": "overloaded",
                    "retry_after_ms": 250,
                    "detail": "utterance queue full (backpressure cap: 3)",
                })
            else:
                raw = bytes(state.buffer)
                state.buffer.clear()
                await state.utterance_queue.put(("fresh", raw, None, time.perf_counter()))
        return

    if action == "reset":
        state.buffer.clear()
        for s in state.slots.values():
            if s.tentative_task and not s.tentative_task.done():
                s.tentative_task.cancel()
            if s.partial_task and not s.partial_task.done():
                s.partial_task.cancel()
        state.slots.clear()
        state.mt_cache.clear()
        state.committed_utts.clear()
        drained = 0
        while not state.utterance_queue.empty():
            try:
                state.utterance_queue.get_nowait()
                state.utterance_queue.task_done()
                drained += 1
            except (asyncio.QueueEmpty, ValueError):
                break
        await state.send_json({"event": "reset", "buffered_bytes": 0, "drained_utterances": drained})
        return

    if action == "ping":
        await state.send_json({"event": "pong", "time": time.time()})
        return

    if action == "played":
        text_val = control.get("text")
        if text_val:
            state.append_tts_output(str(text_val))
        await state.send_json({"event": "played_ack", "utt": control.get("utt")})
        return

    if action == "tentative":
        if "floor_dbfs" in control and control["floor_dbfs"] is not None:
            try:
                state.floor_dbfs = float(control["floor_dbfs"])
            except (TypeError, ValueError):
                pass
        utt_val = control.get("utt")
        seq_val = control.get("seq")
        try:
            utt_int = int(utt_val) if utt_val is not None else None
            seq_int = int(seq_val) if seq_val is not None else None
        except (TypeError, ValueError):
            await state.send_json({"error": "bad_request", "detail": "tentative utt and seq must be integers"})
            return

        if utt_int is None or seq_int is None:
            await state.send_json({"error": "bad_request", "detail": "tentative requires utt and seq"})
            return

        # 1. utt unknown or committed -> {"error":"unknown_utt"}
        if utt_int in state.committed_utts or utt_int not in state.slots:
            await state.send_json({"error": "unknown_utt", "utt": utt_int})
            return

        slot = state.slots[utt_int]

        # 2. slot.buf_ms < 400 -> {"type":"tentative","utt","seq","skipped":"too_short"}; no work.
        if slot.buf_ms < 400:
            if pipeline is not None:
                pipeline.metrics.incr("tentative_skipped_too_short")
            await state.send_json({"type": "tentative", "utt": utt_int, "seq": seq_int, "skipped": "too_short"})
            return

        # 3. Rate cap: if slot.buf_ms - last_tentative_audio_ms < 800 and a tentative already ran -> skipped:"rate_limited"
        if slot.tentatives_issued > 0 and (slot.buf_ms - slot.last_tentative_audio_ms) < 800:
            if pipeline is not None:
                pipeline.metrics.incr("tentative_skipped_rate_limited")
            await state.send_json({"type": "tentative", "utt": utt_int, "seq": seq_int, "skipped": "rate_limited"})
            return

        # 4. seq != slot.last_seq -> skipped:"seq_mismatch" (client's view is stale; a frame is still in flight).
        if seq_int != slot.last_seq:
            if pipeline is not None:
                pipeline.metrics.incr("tentative_skipped_seq_mismatch")
            await state.send_json({"type": "tentative", "utt": utt_int, "seq": seq_int, "skipped": "seq_mismatch"})
            return

        # 5. Cancel running tentative, update tracking, launch new run_tentative
        if slot.tentative_task is not None and not slot.tentative_task.done():
            slot.tentative_task.cancel()

        slot.tentative_seq = seq_int
        slot.last_tentative_audio_ms = slot.buf_ms
        slot.tentatives_issued += 1
        slot.tentative_task = asyncio.create_task(_run_tentative(state, slot, seq_int))
        return

    if action == "abort":
        utt_val = control.get("utt")
        try:
            utt_int = int(utt_val) if utt_val is not None else None
        except (TypeError, ValueError):
            utt_int = None
        if utt_int is not None:
            slot = state.slots.pop(utt_int, None)
            if slot is not None:
                if slot.tentative_task and not slot.tentative_task.done():
                    slot.tentative_task.cancel()
                if slot.partial_task and not slot.partial_task.done():
                    slot.partial_task.cancel()
                slot.tentative_result = None
                freed_bytes = len(slot.buffer)
            else:
                freed_bytes = 0
            if pipeline is not None:
                pipeline.metrics.incr("aborted")
            await state.send_json({"event": "aborted", "utt": utt_int, "freed_bytes": freed_bytes})
        return

    applies_from_utt: int | None = None
    if state.slots:
        applies_from_utt = max(state.slots.keys()) + 1
    elif state.committed_utts:
        applies_from_utt = max(state.committed_utts) + 1
    elif state.utterances > 0:
        applies_from_utt = state.utterances + 1

    await state.send_json(
        {
            "event": "config",
            "mode": state.mode,
            "languages": state.languages,
            "channel_map": state.channel_map,
            "source": state.source,
            "target": state.target,
            "source_variant": state.source_variant,
            "target_variant": None,
            "format": state.audio_format,
            "sample_rate": state.sample_rate,
            "channels": state.channels,
            "protocol": state.protocol_version,
            "floor_dbfs": state.floor_dbfs,
            "applies_from_utt": applies_from_utt,
            "ignored_keys": sorted(unknown) or None,
        }
    )


async def _on_audio_frame(state: _StreamState, chunk: bytes) -> None:
    pipeline = PIPELINE
    settings = state.settings

    # Protocol v2 framed audio: struct "<BBHH" (version, flags, utt_id, seq)
    if state.protocol_version == 2 or state.audio_format == "pcm_s16le_framed":
        if len(chunk) < 6:
            if pipeline is not None:
                pipeline.metrics.incr("bad_frame")
            await state.send_json({
                "error": "bad_frame",
                "detail": f"chunk length {len(chunk)} < 6 bytes header",
            })
            return

        try:
            version, flags, utt_id, seq = struct.unpack("<BBHH", chunk[:6])
            if version != 2:
                if pipeline is not None:
                    pipeline.metrics.incr("bad_frame")
                await state.send_json({"error": "bad_frame", "detail": f"unsupported protocol version {version}"})
                try:
                    await state.websocket.close(code=1003)
                except Exception:
                    pass
                return

            payload = memoryview(chunk)[6:]
            if len(payload) % 2 != 0:
                if pipeline is not None:
                    pipeline.metrics.incr("bad_frame")
                await state.send_json({
                    "error": "bad_frame",
                    "detail": f"odd payload length {len(payload)}: 16-bit PCM requires even bytes",
                    "utterance": utt_id,
                    "seq": seq,
                })
                return

            # I2 & I7: Check if utterance was already committed
            if utt_id in state.committed_utts:
                if pipeline is not None:
                    pipeline.metrics.incr("late_frame_dropped")
                if flags & 0x02:
                    await state.send_json({"event": "already_committed", "utt": utt_id})
                return

            slot = state.slots.setdefault(
                utt_id,
                _Slot(
                    utt_id,
                    state.source,
                    state.target,
                    source_variant=state.source_variant,
                    target_variant=state.target_variant,
                ),
            )
            slot.total_frames += 1
            if flags & 0x04:
                slot.during_playback_frames += 1

            # PREROLL flag check (bit 0: 0x01)
            if flags & 0x01:
                if slot.saw_preroll:
                    if pipeline is not None:
                        pipeline.metrics.incr("duplicate_preroll")
                    return
                slot.saw_preroll = True

            # Sequence check and modulo 2^16 wrap arithmetic (I1 / I10)
            if seq in slot.seq_seen:
                if flags & 0x02 and seq == slot.last_seq:
                    pass  # fall through to commit (idempotent) instead of dropping
                else:
                    # Duplicate frame: drop
                    return

            if slot.last_seq is not None:
                diff = (seq - slot.last_seq) & 0xFFFF
                if diff != 1 and not (flags & 0x02 and seq == slot.last_seq):
                    if pipeline is not None:
                        pipeline.metrics.incr("seq_gap")
                    log.warning("slot %d seq gap: expected %d, got %d (diff=%d)", utt_id, (slot.last_seq + 1) & 0xFFFF, seq, diff)

            # Note: slot.last_seq = seq is updated before the discard check uses slot.tentative_seq.
            # Order is fine; seq_mismatch in the tentative handler depends on slot.last_seq
            # recording the most recently processed audio frame before comparing with tentative_seq.
            if len(payload) > 0:
                slot.last_audio_seq = seq
            slot.last_seq = seq
            slot.seq_seen.add(seq)

            # B.2 Discard: If tentative running or cached result present and seq > tentative_seq
            if len(payload) > 0 and ((slot.tentative_task is not None and not slot.tentative_task.done()) or slot.tentative_result is not None):
                if slot.tentative_seq != -1 and seq != slot.tentative_seq and ((seq - slot.tentative_seq) & 0xFFFF) < 32768:
                    if slot.tentative_task is not None and not slot.tentative_task.done():
                        slot.tentative_task.cancel()
                    slot.tentative_result = None
                    disc_seq = slot.tentative_seq
                    slot.tentative_seq = -1
                    if pipeline is not None:
                        pipeline.metrics.incr("tentative_cancelled")
                    await state.send_json({
                        "type": "discard",
                        "utt": utt_id,
                        "what": "tentative",
                        "seq": disc_seq,
                        "reason": "audio_after_tentative",
                    })

            # Overflow handling: append what fits, auto-commit, roll remainder to utt+1
            if len(slot.buffer) + len(payload) > state.max_bytes:
                if pipeline is not None:
                    pipeline.metrics.incr("auto_commit")
                space = max(0, state.max_bytes - len(slot.buffer))
                if space > 0:
                    slot.buffer.extend(payload[:space])
                remainder = bytes(payload[space:])

                next_utt = (utt_id + 1) & 0xFFFF
                await state.send_json({
                    "event": "auto_commit",
                    "reason": "max_utterance",
                    "utt": utt_id,
                    "rolled_to": next_utt,
                })
                await _commit_slot(state, slot, seq)

                next_slot = state.slots.setdefault(
                    next_utt,
                    _Slot(
                        next_utt,
                        state.source,
                        state.target,
                        source_variant=state.source_variant,
                        target_variant=state.target_variant,
                    ),
                )
                if remainder:
                    next_slot.buffer.extend(remainder)
                next_slot.last_seq = seq
                next_slot.seq_seen.add(seq)
                return

            slot.buffer.extend(payload)

            # Partials: while frames are arriving and slot.buf_ms >= 800, every partial_ms
            if state.settings.partial_ms > 0 and slot.buf_ms >= 800:
                interval_s = state.settings.partial_ms / 1000.0
                now = time.monotonic()
                if now - slot.last_partial_started >= interval_s:
                    if slot.partial_task is not None and not slot.partial_task.done():
                        if pipeline is not None:
                            pipeline.metrics.incr("partials_skipped")
                    elif slot.tentative_task is None or slot.tentative_task.done() or slot.tentative_seq != seq:
                        slot.last_partial_started = now
                        slot.partial_task = asyncio.create_task(_run_partial(state, slot, seq))

            # Check flags: bit 1 is LAST (commit)
            if flags & 0x02:
                commit_seq = slot.last_audio_seq if not payload and slot.last_audio_seq is not None else seq
                await _commit_slot(state, slot, commit_seq)
                return

        except Exception as exc:
            if pipeline is not None:
                pipeline.metrics.incr("bad_frame")
            log.warning("error parsing framed binary audio: %s", exc)
            return

        return

    # Protocol v1 legacy un-framed audio
    if len(state.buffer) + len(chunk) > state.max_bytes:
        state.buffer.clear()
        await state.send_json(
            {
                "error": "utterance_too_long",
                "detail": (
                    f"buffered audio exceeded {settings.max_utterance_seconds}s; "
                    "buffer cleared. Send {'action':'flush'} at end of speech."
                ),
            }
        )
        return
    state.buffer.extend(chunk)


@app.websocket("/ws/v1/translate-stream")
async def translate_stream(websocket: WebSocket) -> None:
    settings = SETTINGS
    if settings.auth_token:
        auth_header = websocket.headers.get("authorization", "")
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        if not token:
            # TODO(prod): header-only (query params appear in access/proxy logs)
            token = websocket.query_params.get("token", "").strip()
        if token != settings.auth_token:
            await websocket.close(code=4001, reason="Unauthorized")
            return

    await websocket.accept()

    if PIPELINE is None or not PIPELINE.ready:
        await websocket.send_json(
            {
                "error": "not_ready",
                "detail": STARTUP_ERROR or "engines still loading",
            }
        )
        await websocket.close(code=1013)
        return

    state = _StreamState(settings, websocket)
    state.worker_task = asyncio.create_task(_utterance_worker(state))

    await state.send_json(
        {
            "event": "ready",
            "version": VERSION,
            "protocols": [1, 2],
            "negotiated_protocol": state.protocol_version,
            "tentative": "ack_only",
            "protocol": {
                "version": 2,
                "send_json": {
                    "source": "language code or 'auto'",
                    "target": "language code (required)",
                    "format": "pcm_s16le | pcm_s16le_framed | opus | ogg | webm | wav (optional; magic bytes win)",
                    "sample_rate": settings.sample_rate,
                    "channels": 1,
                    "protocol": "1 or 2 (default 1; 2 uses 6-byte binary header <BBHH)",
                    "action": "'flush' to end the utterance and translate now, 'ping' for keep-alive, 'abort' with 'utt'",
                    "stream": (
                        "true (default) for one frame per sentence then a "
                        "'final' frame; false for a single unified JSON"
                    ),
                },
                "send_binary": "audio chunks (Protocol v1: raw PCM16; Protocol v2: 6-byte header <BBHH + PCM16)",
                "receive": [
                    "original_text",
                    "translated_text",
                    "source_lang",
                    "target_lang",
                    "asr_ms",
                    "mt_ms",
                    "total_server_ms",
                ],
            },
            "sentence_streaming": state.stream,
            "streaming_contract": (
                "frames carry \"type\": \"sentence\" (index, is_last, "
                "elapsed_ms) followed by exactly one \"type\": \"final\" "
                "carrying the unified fields. Play each sentence as it "
                "arrives; the utterance is done at is_last/final."
                if state.stream
                else "one JSON per utterance; no \"type\" field is sent"
            ),
            "latency_budget_ms": settings.latency_budget_ms,
            "max_utterance_seconds": settings.max_utterance_seconds,
        }
    )

    try:
        while True:
            has_pending_audio = bool(state.buffer) or any(len(s.buffer) > 0 for s in state.slots.values())
            recv_timeout = 3.0 if has_pending_audio else 120.0
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=recv_timeout)
            except asyncio.TimeoutError:
                if has_pending_audio:
                    # Leak guard: 3.0s silence safety net.
                    # In clean operation the client owns utterance closure via LAST flag.
                    # If 3.0s elapses with pending audio, auto-commit as a client_timeout fallback.
                    if state.protocol_version == 2:
                        for s in list(state.slots.values()):
                            slot_id = s.utt_id
                            if len(s.buffer) >= 3200:
                                if PIPELINE is not None:
                                    PIPELINE.metrics.incr("auto_commit")
                                    PIPELINE.metrics.incr("auto_commit_timeout")
                                log.warning("Slot %d committed by 3.0s leak-guard (client_timeout, %d bytes)", slot_id, len(s.buffer))
                                await state.send_json({
                                    "event": "auto_commit",
                                    "reason": "client_timeout",
                                    "utt": slot_id,
                                })
                                await _commit_slot(state, s, s.last_seq)
                            else:
                                if s.tentative_task and not s.tentative_task.done():
                                    s.tentative_task.cancel()
                                if s.partial_task and not s.partial_task.done():
                                    s.partial_task.cancel()
                                state.slots.pop(slot_id, None)
                    elif state.buffer:
                        if len(state.buffer) >= 3200:
                            raw = bytes(state.buffer)
                            state.buffer.clear()
                            if PIPELINE is not None:
                                PIPELINE.metrics.incr("auto_commit")
                                PIPELINE.metrics.incr("auto_commit_timeout")
                            log.warning("V1 buffer committed by 3.0s leak-guard (client_timeout, %d bytes)", len(raw))
                            await state.send_json({
                                "event": "auto_commit",
                                "reason": "client_timeout",
                            })
                            if state.target and not state.utterance_queue.full():
                                await state.utterance_queue.put(("fresh", raw, None, time.perf_counter()))
                        else:
                            state.buffer.clear()
                    continue
                else:
                    log.info("closing idle websocket after 120s timeout")
                    try:
                        await websocket.close(code=1000)
                    except Exception:
                        pass
                    break
            if message.get("type") == "websocket.disconnect":
                break

            # ---- control frames -------------------------------------
            text_payload = message.get("text")
            if text_payload is not None:
                try:
                    control = json.loads(text_payload)
                except json.JSONDecodeError as exc:
                    await state.send_json(
                        {"error": "bad_json", "detail": f"control frame is not JSON: {exc}"}
                    )
                    continue
                if not isinstance(control, dict):
                    await state.send_json(
                        {"error": "bad_json", "detail": "control frame must be a JSON object"}
                    )
                    continue

                if control.get("action") == "close":
                    break

                await _on_control_frame(state, control)
                continue

            # ---- audio frames ---------------------------------------
            chunk = message.get("bytes")
            if chunk is not None:
                await _on_audio_frame(state, chunk)
                if state.closed:
                    break
                continue

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket handler failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        state.closed = True
        for s in list(state.slots.values()):
            if s.tentative_task and not s.tentative_task.done():
                s.tentative_task.cancel()
            if s.partial_task and not s.partial_task.done():
                s.partial_task.cancel()
        state.slots.clear()
        state.mt_cache.clear()
        if state.worker_task is not None and not state.worker_task.done():
            try:
                state.utterance_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            state.worker_task.cancel()
            try:
                await state.worker_task
            except (asyncio.CancelledError, Exception):
                pass


async def _utterance_worker(state: _StreamState) -> None:
    while True:
        try:
            item = await state.utterance_queue.get()
        except asyncio.CancelledError:
            break
        if item is None:
            break
        try:
            if state.closed:
                log.debug("connection closed; draining queued item")
                continue

            # Check item kind
            if isinstance(item, tuple) and item[0] == "serve_cached":
                _, slot, seq, (frames, final_frame, tentative_pass_ms), commit_time, *rest = item
                during_ratio = rest[0] if len(rest) > 0 else 0.0
                context_prefix = rest[1] if len(rest) > 1 else ""
                utt_id = slot.utt_id

                if during_ratio >= 0.5 and is_echo_match(final_frame.get("original_text", ""), state.recent_tts_outputs):
                    if PIPELINE is not None:
                        PIPELINE.metrics.incr("echo_dropped")
                    await state.send_json({
                        "event": "dropped",
                        "type": "dropped",
                        "utt": utt_id,
                        "utterance": utt_id,
                        "reason": "echo",
                        "mode": "HIT",
                        "echo_dropped": True,
                    })
                    continue

                c2f_ms = (time.perf_counter() - commit_time) * 1000.0
                if PIPELINE is not None:
                    PIPELINE.metrics.observe("commit_to_first_frame_ms", c2f_ms)
                    PIPELINE.metrics.observe("tentative_pass_ms", tentative_pass_ms)

                for sf in frames:
                    await state.send_json({**sf, "utterance": utt_id, "utt": utt_id})

                final_copy = dict(final_frame)
                final_copy["mode"] = "HIT"
                final_copy["echo_dropped"] = False
                final_copy["retried"] = final_copy.get("retried", False)
                final_copy["hollow_reason"] = final_copy.get("hollow_reason", None)
                final_copy["rms_dbfs"] = final_copy.get("rms_dbfs", None)
                final_copy["total_server_ms"] = round(c2f_ms, 2)
                final_copy["tentative_hit"] = True
                final_copy["time_saved_ms"] = round(tentative_pass_ms, 2)
                if PIPELINE is not None:
                    ver_fields = await _verify_pinned_language(PIPELINE, state, bytes(slot.buffer), slot.source)
                    final_copy.update(ver_fields)
                await state.send_json({**final_copy, "utterance": utt_id, "utt": utt_id})

                if final_copy.get("translated_text"):
                    state.append_tts_output(final_copy["translated_text"])
                state.last_commit_time = commit_time
                src_text = final_copy.get("original_text", "")
                state.last_committed_src_text = src_text
                state.last_has_terminal_punct = bool(src_text.rstrip() and src_text.rstrip()[-1] in _TERMINATORS)

            elif isinstance(item, tuple) and item[0] == "await_then_serve":
                _, slot, seq, task, commit_time, *rest = item
                during_ratio = rest[0] if len(rest) > 0 else 0.0
                context_prefix = rest[1] if len(rest) > 1 else ""
                utt_id = slot.utt_id
                try:
                    await task
                except Exception:
                    pass

                if slot.tentative_result is not None:
                    frames, final_frame, tentative_pass_ms = slot.tentative_result
                    if during_ratio >= 0.5 and is_echo_match(final_frame.get("original_text", ""), state.recent_tts_outputs):
                        if PIPELINE is not None:
                            PIPELINE.metrics.incr("echo_dropped")
                        await state.send_json({
                            "event": "dropped",
                            "type": "dropped",
                            "utt": utt_id,
                            "utterance": utt_id,
                            "reason": "echo",
                            "mode": "AWAIT",
                            "echo_dropped": True,
                        })
                        continue

                    c2f_ms = (time.perf_counter() - commit_time) * 1000.0
                    if PIPELINE is not None:
                        PIPELINE.metrics.observe("commit_to_first_frame_ms", c2f_ms)
                        PIPELINE.metrics.observe("tentative_pass_ms", tentative_pass_ms)

                    for sf in frames:
                        await state.send_json({**sf, "utterance": utt_id, "utt": utt_id})

                    final_copy = dict(final_frame)
                    final_copy["mode"] = "AWAIT"
                    final_copy["echo_dropped"] = False
                    final_copy["retried"] = final_copy.get("retried", False)
                    final_copy["hollow_reason"] = final_copy.get("hollow_reason", None)
                    final_copy["rms_dbfs"] = final_copy.get("rms_dbfs", None)
                    final_copy["total_server_ms"] = round(c2f_ms, 2)
                    final_copy["tentative_hit"] = True
                    final_copy["time_saved_ms"] = round(tentative_pass_ms, 2)
                    if PIPELINE is not None:
                        ver_fields = await _verify_pinned_language(PIPELINE, state, bytes(slot.buffer), slot.source)
                        final_copy.update(ver_fields)
                    await state.send_json({**final_copy, "utterance": utt_id, "utt": utt_id})

                    if final_copy.get("translated_text"):
                        state.append_tts_output(final_copy["translated_text"])
                    state.last_commit_time = commit_time
                    src_text = final_copy.get("original_text", "")
                    state.last_committed_src_text = src_text
                    state.last_has_terminal_punct = bool(src_text.rstrip() and src_text.rstrip()[-1] in _TERMINATORS)
                else:
                    raw = bytes(slot.buffer)
                    await _handle_utterance_payload(
                        state,
                        raw,
                        utt_id,
                        source=slot.source,
                        target=slot.target,
                        source_variant=slot.source_variant,
                        target_variant=slot.target_variant,
                        during_ratio=during_ratio,
                        context_prefix=context_prefix,
                    )

            elif isinstance(item, tuple) and item[0] == "fresh":
                _, raw, utt_id, commit_time, *extra = item
                slot_src = extra[0] if len(extra) > 0 else None
                slot_dst = extra[1] if len(extra) > 1 else None
                slot_src_var = extra[2] if len(extra) > 2 else None
                slot_tgt_var = extra[3] if len(extra) > 3 else None
                during_ratio = extra[4] if len(extra) > 4 else 0.0
                context_prefix = extra[5] if len(extra) > 5 else ""
                slot_spk = extra[6] if len(extra) > 6 else None
                slot_dir = extra[7] if len(extra) > 7 else None
                slot_chan = extra[8] if len(extra) > 8 else None
                slot_low_conf = extra[9] if len(extra) > 9 else False
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
                    speaker_id=slot_spk,
                    direction=slot_dir,
                    channel=slot_chan,
                    lid_low_confidence=slot_low_conf,
                )

            elif isinstance(item, tuple) and len(item) == 2:
                raw, utt_id = item
                await _handle_utterance_payload(state, raw, utt_id)

        except Exception:
            log.exception("utterance worker error")
        finally:
            state.utterance_queue.task_done()


async def _handle_utterance_payload(
    state: _StreamState,
    raw: bytes,
    utt_id: int | None = None,
    source: str | None = None,
    target: str | None = None,
    source_variant: str | None = None,
    target_variant: str | None = None,
    during_ratio: float = 0.0,
    context_prefix: str = "",
    speaker_id: int | None = None,
    direction: str | None = None,
    channel: str | None = None,
    lid_low_confidence: bool = False,
) -> None:
    state.utterances += 1
    utt_tag = utt_id if utt_id is not None else state.utterances

    assert PIPELINE is not None

    if state.mode == "pair_auto" and len(state.languages) >= 2:
        if speaker_id is None or source is None:
            try:
                decoded, _ = await PIPELINE._decode_checked(
                    raw,
                    declared_format=state.audio_format,
                    input_sample_rate=state.sample_rate,
                    channels=state.channels,
                )
                winner, conf, low_conf = await asyncio.to_thread(
                    PIPELINE.asr.detect_language_constrained,
                    decoded.samples,
                    state.languages,
                )
                cand0, cand1 = state.languages[0], state.languages[1]
                source = winner
                target = cand1 if winner == cand0 else cand0
                speaker_id = 0 if winner == cand0 else 1
                direction = f"{source}->{target}"
                channel = state.channel_map.get(str(speaker_id), "L" if speaker_id == 0 else "R")
                lid_low_confidence = low_conf
                if low_conf and PIPELINE is not None:
                    PIPELINE.metrics.incr("lid_low_confidence")
                if target == "ar":
                    target_variant = None
                if source == "ar":
                    source_variant = state.source_variant
                else:
                    source_variant = None
            except Exception as exc:
                log.warning("pair_auto LID failed in _handle_utterance_payload: %s", exc)

    if state.stream:
        await _stream_utterance(
            state,
            raw,
            utt_tag,
            source=source,
            target=target,
            source_variant=source_variant,
            target_variant=target_variant,
            during_ratio=during_ratio,
            context_prefix=context_prefix,
            speaker_id=speaker_id,
            direction=direction,
            channel=channel,
            lid_low_confidence=lid_low_confidence,
        )
        return

    effective_src = source if source is not None else (state.source or "auto")
    effective_dst = target if target is not None else state.target
    effective_src_var = source_variant if source_variant is not None else (state.source_variant or "")
    effective_tgt_var = target_variant if target_variant is not None else (state.target_variant or "")

    echo_fn = (lambda asr_text: is_echo_match(asr_text, state.recent_tts_outputs)) if during_ratio >= 0.5 else None

    try:
        outcome = await PIPELINE.translate_audio(
            raw,
            source=effective_src,
            target=effective_dst,
            source_variant=effective_src_var,
            target_variant=effective_tgt_var,
            context_prefix=context_prefix,
            is_echo=echo_fn,
            declared_format=state.audio_format,
            input_sample_rate=state.sample_rate,
            channels=state.channels,
        )
    except Overloaded as exc:
        await state.send_json({**exc.payload(), "utterance": utt_tag, "utt": utt_tag})
        return
    except RequestError as exc:
        await state.send_json(
            {"error": exc.code, "detail": str(exc), "utterance": utt_tag, "utt": utt_tag}
        )
        return
    except Exception as exc:
        log.exception("utterance failed")
        await state.send_json(
            {
                "error": "internal",
                "detail": f"{type(exc).__name__}: {exc}",
                "utterance": utt_tag,
                "utt": utt_tag,
            }
        )
        return

    if outcome.detail and outcome.detail.get("dropped"):
        await state.send_json({
            "event": "dropped",
            "type": "dropped",
            "utt": utt_tag,
            "utterance": utt_tag,
            "reason": outcome.detail.get("drop_reason", "echo"),
            "original_text": outcome.original_text,
            "asr_ms": outcome.asr_ms,
            "rms_dbfs": outcome.detail.get("rms_dbfs", None) if outcome.detail else None,
            "mode": "MISS",
            "echo_dropped": True,
        })
        return

    out_payload = outcome.to_json()
    out_payload["mode"] = "MISS"
    out_payload["echo_dropped"] = False
    out_payload["retried"] = outcome.detail.get("retried", False) if outcome.detail else False
    out_payload["hollow_reason"] = outcome.detail.get("hollow_reason", None) if outcome.detail else None
    out_payload["rms_dbfs"] = outcome.detail.get("rms_dbfs", None) if outcome.detail else None
    if state.mode == "pair_auto":
        out_payload["speaker_id"] = speaker_id
        out_payload["direction"] = direction
        out_payload["channel"] = channel
        out_payload["lid_low_confidence"] = lid_low_confidence
    if effective_src_var:
        out_payload["source_variant"] = effective_src_var
    if effective_tgt_var:
        out_payload["target_variant"] = effective_tgt_var
    ver_fields = await _verify_pinned_language(PIPELINE, state, raw, effective_src)
    out_payload.update(ver_fields)
    await state.send_json({**out_payload, "utterance": utt_tag, "utt": utt_tag})
    src_text = out_payload.get("original_text", "")
    mt_text = out_payload.get("translated_text", "")
    if PIPELINE is not None and out_payload.get("rms_dbfs") is not None:
        PIPELINE.metrics.observe_rms("rendered", out_payload.get("rms_dbfs"))
    log.info(
        "Utterance committed [utt=%s]: %s[%s] -> %s | src=%r -> mt=%r | rms_dbfs=%s | server_ms=%.1f",
        utt_tag,
        out_payload.get("source_lang"),
        out_payload.get("source_variant") or "none",
        out_payload.get("target_lang"),
        src_text,
        mt_text,
        out_payload.get("rms_dbfs"),
        out_payload.get("total_server_ms", 0.0),
    )

    if out_payload.get("translated_text"):
        state.append_tts_output(out_payload["translated_text"])
    state.last_commit_time = time.perf_counter()
    state.last_committed_src_text = src_text
    state.last_has_terminal_punct = bool(src_text.rstrip() and src_text.rstrip()[-1] in _TERMINATORS)


async def _stream_utterance(
    state: _StreamState,
    raw: bytes,
    utt_tag: int,
    source: str | None = None,
    target: str | None = None,
    source_variant: str | None = None,
    target_variant: str | None = None,
    during_ratio: float = 0.0,
    context_prefix: str = "",
    speaker_id: int | None = None,
    direction: str | None = None,
    channel: str | None = None,
    lid_low_confidence: bool = False,
) -> None:
    """Relay the pipeline's per-sentence frames to the client."""
    assert PIPELINE is not None
    effective_src = source if source is not None else (state.source or "auto")
    effective_dst = target if target is not None else (state.target or "")
    effective_src_var = source_variant if source_variant is not None else (state.source_variant or "")
    effective_tgt_var = target_variant if target_variant is not None else (state.target_variant or "")

    echo_fn = (lambda asr_text: is_echo_match(asr_text, state.recent_tts_outputs)) if during_ratio >= 0.5 else None

    frames = 0
    ver_fields = await _verify_pinned_language(PIPELINE, state, raw, effective_src)
    try:
        async for frame in PIPELINE.translate_audio_streaming(
            raw,
            source=effective_src,
            target=effective_dst,
            source_variant=effective_src_var,
            target_variant=effective_tgt_var,
            context_prefix=context_prefix,
            is_echo=echo_fn,
            declared_format=state.audio_format,
            input_sample_rate=state.sample_rate,
            channels=state.channels,
        ):
            if frame.get("type") == "dropped":
                await state.send_json({
                    "event": "dropped",
                    "type": "dropped",
                    "utt": utt_tag,
                    "utterance": utt_tag,
                    "reason": frame.get("reason", "echo"),
                    "original_text": frame.get("original_text", ""),
                    "asr_ms": frame.get("asr_ms", 0.0),
                    "mode": "MISS",
                    "echo_dropped": True,
                })
                return

            if frame.get("type") == "sentence" and frame.get("translated_text"):
                state.append_tts_output(frame["translated_text"])

            if frame.get("type") == "final":
                if ver_fields:
                    frame = {**frame, **ver_fields}
                frame["mode"] = "MISS"
                frame["echo_dropped"] = False
                if "retried" not in frame:
                    frame["retried"] = False
                if "hollow_reason" not in frame:
                    frame["hollow_reason"] = None
                if "rms_dbfs" not in frame:
                    frame["rms_dbfs"] = None
                if frame.get("translated_text"):
                    state.append_tts_output(frame["translated_text"])
                state.last_commit_time = time.perf_counter()
                src_text = frame.get("original_text", "")
                state.last_committed_src_text = src_text
                state.last_has_terminal_punct = bool(src_text.rstrip() and src_text.rstrip()[-1] in _TERMINATORS)

            frame_payload = {**frame, "utterance": utt_tag, "utt": utt_tag}
            if state.mode == "pair_auto":
                frame_payload["speaker_id"] = speaker_id
                frame_payload["direction"] = direction
                frame_payload["channel"] = channel
                frame_payload["lid_low_confidence"] = lid_low_confidence
            await state.send_json(frame_payload)
            if frame.get("type") == "final":
                if PIPELINE is not None and frame.get("rms_dbfs") is not None:
                    PIPELINE.metrics.observe_rms("rendered", frame.get("rms_dbfs"))
                log.info(
                    "Utterance streamed final [utt=%s]: %s[%s] -> %s | src=%r -> mt=%r | rms_dbfs=%s | server_ms=%.1f",
                    utt_tag,
                    frame.get("source_lang"),
                    frame.get("source_variant") or "none",
                    frame.get("target_lang"),
                    src_text,
                    frame.get("translated_text", ""),
                    frame.get("rms_dbfs"),
                    frame.get("total_server_ms", 0.0),
                )
            frames += 1
        return
    except Overloaded as exc:
        payload = exc.payload()
    except RequestError as exc:
        payload = {"error": exc.code, "detail": str(exc)}
    except Exception as exc:
        log.exception("streamed utterance failed")
        payload = {"error": "internal", "detail": f"{type(exc).__name__}: {exc}"}

    await state.send_json(
        {
            "type": "error",
            **payload,
            "utterance": utt_tag,
            "utt": utt_tag,
            "frames_sent": frames,
            "partial": frames > 0,
            "terminal": True,
            "detail_note": (
                "this utterance failed AFTER {n} frame(s) were already sent; "
                "stop playback and do not wait for a 'final' frame".format(n=frames)
                if frames
                else "this utterance failed before any sentence was produced"
            ),
        }
    )


# =====================================================================
# Observability & RunPod Health Probe
# =====================================================================
@app.get("/ping")
async def ping() -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "ok", "ping": "pong"})


@app.get("/health")
async def health() -> JSONResponse:
    ready = PIPELINE is not None and PIPELINE.ready
    body: dict[str, Any] = {
        "status": "ok" if ready else ("error" if STARTUP_ERROR else "loading"),
        "version": VERSION,
        "ready": ready,
        "device": SETTINGS.device,
        "startup_error": STARTUP_ERROR,
        "asr_language_count": lang_mod.ASR_LANGUAGE_COUNT,
        "mt_language_count": lang_mod.MT_LANGUAGE_COUNT,
        "language_count": lang_mod.ASR_LANGUAGE_COUNT,
        "pair_count": lang_mod.MT_LANGUAGE_COUNT * (lang_mod.MT_LANGUAGE_COUNT - 1),
        "latency_budget_ms": SETTINGS.latency_budget_ms,
    }
    if PIPELINE is not None:
        body["uptime_seconds"] = (
            round(time.time() - PIPELINE.started_at, 1) if PIPELINE.started_at else 0.0
        )
        body["engines"] = PIPELINE.engine_info()
        body["engine_list"] = [PIPELINE.asr.info(), PIPELINE.mt.info()]
        body["warmup"] = PIPELINE.warmup_report
        # The honest headline: how often we actually met the budget.
        body["latency"] = PIPELINE.metrics.budget_report()
        body["queues"] = PIPELINE.scheduler_stats()
    body["resources"] = resource_report()
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.get("/metrics")
async def metrics() -> JSONResponse:
    if PIPELINE is None:
        return JSONResponse(
            status_code=503,
            content={"error": "not_ready", "detail": STARTUP_ERROR or "loading"},
        )
    from .mt import CHAT_LEAK_SUSPECTED_COUNT, PERSON_MISMATCH_RETRY_COUNT

    return JSONResponse(
        content={
            "version": VERSION,
            "device": SETTINGS.device,
            **PIPELINE.metrics.snapshot(),
            "chat_leak_suspected": CHAT_LEAK_SUSPECTED_COUNT,
            "person_mismatch_retry": PERSON_MISMATCH_RETRY_COUNT,
            "queues": PIPELINE.scheduler_stats(),
            "engines": PIPELINE.engine_info(),
            "resources": resource_report(),
            "notes": [
                "All latency figures are MEASURED on this process, this device.",
                "p95 is withheld below 20 samples and p99 below 100: a percentile "
                "from too few points is not a percentile.",
                "'unavailable' is not zero. A null GPU figure means no CUDA device "
                "was visible, not that the GPU used 0 MB.",
            ],
        }
    )


@app.get("/languages")
async def languages() -> dict[str, Any]:
    catalogue = lang_mod.catalogue()
    return {
        "languages": catalogue,
        "asr_language_count": lang_mod.ASR_LANGUAGE_COUNT,
        "mt_language_count": lang_mod.MT_LANGUAGE_COUNT,
        "language_count": lang_mod.ASR_LANGUAGE_COUNT,
        "source": "faster_whisper.tokenizer._LANGUAGE_CODES (read at import, never hardcoded)",
        "caveat": (
            "'present in the tokenizer' is not 'usable in a commercial earbud'. "
            "The Whisper paper reports WER above 50% on roughly 20 of the "
            "lowest-resource languages. Per-language quality is NOT verified by "
            "this server; it needs per-language test corpora."
        ),
        "mt_coverage_note": (
            "The loaded MT model may not cover every ASR language. Requests for a "
            "pair the MT model has no token for are refused, not answered with "
            "plausible garbage."
        ),
    }


@app.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    """What this build does, with the arithmetic behind the claims.

    Kept as an endpoint rather than a README paragraph so the numbers travel
    with the running server and cannot drift from it.
    """
    queues = PIPELINE.scheduler_stats() if PIPELINE else {}
    asr_stats = queues.get("asr", {})
    mt_stats = queues.get("mt", {})
    measured_rps = [
        s.get("capacity_rps") for s in (asr_stats, mt_stats) if s.get("capacity_rps")
    ]
    bottleneck_rps = min(measured_rps) if measured_rps else None
    return {
        "version": VERSION,
        "device": SETTINGS.device,
        "latency_budget_ms": SETTINGS.latency_budget_ms,
        "measured": {
            "capacity_rps_this_device": bottleneck_rps,
            "asr_per_item_ms": asr_stats.get("measured_per_item_ms"),
            "mt_per_item_ms": mt_stats.get("measured_per_item_ms"),
            "note": (
                "null until at least one batch has run. These are the only "
                "throughput numbers this server will state as fact."
            ),
        },
        "concurrency_reality": {
            "claim_in_brief": "100+ concurrent users under 150 ms on one RTX 3060",
            "verdict": "not reachable as specified",
            "arithmetic": {
                "service_time_ms_from_brief": 127.5,
                "implied_capacity_rps": 7.8,
                "demand_rps_100_users_one_turn_every_3s": 33.3,
                "demand_rps_100_users_one_turn_every_6s": 16.7,
                "demand_rps_100_users_one_turn_every_10s": 10.0,
                "utilisation": "1.28x to 4.25x",
                "consequence": (
                    "utilisation >= 1 means the queue grows without bound: "
                    "latency diverges, it does not settle at 150 ms"
                ),
            },
            "batching_tradeoff_modelled": {
                "B=8": {"throughput_rps": 30.6, "p50_ms": 392.1},
                "B=32": {"throughput_rps": 44.4, "p50_ms": 1080.6},
                "note": (
                    "batching reaches the required throughput but not at 150 ms. "
                    "MODELLED, not measured: no GPU was available to this build."
                ),
            },
            "what_this_server_does": (
                "refuses work it cannot finish inside the budget (503 + Retry-After) "
                "so the client can degrade visibly instead of silently"
            ),
        },
        "weight_sizes_measured": {
            "method": (
                "the safetensors header of each real checkpoint was read over HTTP "
                "range requests and the per-tensor byte offsets summed. Nothing was "
                "downloaded and nothing was estimated from parameter counts."
            ),
            "qwen2.5-1.5b_fp16_gb": 3.087467144,
            "qwen2.5-1.5b_awq_int4_gb": 1.61455384,
            "awq_breakdown_gb": {
                "qweight_int4_packed": 0.655,
                "lm_head_fp16": 0.467,
                "embed_tokens_fp16": 0.467,
                "scales": 0.020,
                "qzeros": 0.005,
            },
            "retraction": (
                "an earlier version of this endpoint reported int4 as 0.72 GB. That "
                "was WRONG. It quantised the whole parameter count and ignored that "
                "AWQ leaves the embedding and lm_head matrices in fp16 -- together "
                "0.934 GB, MORE than the 0.655 GB of quantised linear layers. The "
                "measured file is 1.614 GB, 2.2x the retracted figure."
            ),
            "why_lm_head_ships_twice": (
                "config.json says tie_word_embeddings: true, yet the checkpoint "
                "contains lm_head.weight as a separate fp16 tensor. Trusting the "
                "flag instead of the header is exactly how the 0.72 GB error "
                "survived three rounds of arithmetic."
            ),
            "whisper-large-v3-turbo_modelled": {"fp16_gb": 1.51, "int8_gb": 0.75},
            "finding": (
                "the brief's 1.5-2.0 GB MT budget cannot hold Qwen2.5-1.5B at fp16 "
                "(3.09 GB of weights before KV cache, activations and the ~0.3-0.6 GB "
                "CUDA context). AWQ int4 at 1.61 GB fits, but only just."
            ),
        },
        "decode_roofline": {
            "rtx_3060_12gb_bandwidth_gb_s": 360,
            "status": (
                "MODELLED arithmetic over MEASURED weight sizes: the GB column is "
                "measured, the ms columns are a bandwidth ceiling, not a benchmark. "
                "No CUDA device was available to this build."
            ),
            "method": (
                "an autoregressive decode step reads every weight once, so "
                "max tok/s = bandwidth / weight_bytes. The 'ideal' column assumes "
                "100% bandwidth efficiency and zero overhead; the 'real' column "
                "applies the 70% efficiency that production systems actually reach."
            ),
            "median_output_tokens_measured": 8,
            "max_output_tokens_measured": 16,
            "qwen1.5b_fp16": {
                "weights_gb": 3.087,
                "ceiling_tok_s": 116.6,
                "ms_at_8_ideal": 68.6,
                "ms_at_8_real_70pct": 98.0,
            },
            "qwen1.5b_awq_int4": {
                "weights_gb": 1.614,
                "ceiling_tok_s": 223.0,
                "ms_at_8_ideal": 35.9,
                "ms_at_8_real_70pct": 51.2,
                "note": "as shipped: both embedding copies resident",
            },
            "qwen1.5b_awq_int4_tied": {
                "weights_gb": 1.147,
                "ceiling_tok_s": 313.9,
                "ms_at_8_ideal": 25.5,
                "ms_at_8_real_70pct": 36.4,
                "note": (
                    "hypothetical best case if the duplicate lm_head copy were "
                    "dropped at load time. Still misses 20-27 ms."
                ),
            },
            "finding": (
                "the requested 20-27 ms MT figure is NOT reachable on an RTX 3060 "
                "with this model at any quantisation: the best case is 36.4 ms and "
                "the shipped configuration is ~51 ms. AWQ is still the right default "
                "-- it is ~1.9x faster than fp16 and 1.47 GB lighter, and ~51 ms "
                "leaves room inside the 150 ms whole-request budget -- but it must "
                "not be sold on a number it cannot hit."
            ),
        },
        "measured_vs_modelled": {
            "measured_here": [
                "ASR latency on real speech, CPU",
                "MT latency on real text, CPU",
                "language count (100, read from the tokenizer)",
                "scheduler batching, admission and shedding behaviour",
                "Qwen output-token counts (real tokenizer)",
                "checkpoint weight sizes (safetensors headers, both fp16 and AWQ)",
                "sentence splitting on 6 scripts, including the CJK case that "
                "the first implementation silently failed",
            ],
            "modelled_only": [
                "every GPU latency figure",
                "the decode roofline (its GB inputs are measured; its ms are not)",
                "batching throughput at B=8 and B=32",
                "the ~595 ms time-to-first-word saving on real earbud hardware",
            ],
            "not_verified_at_all": [
                "the Dockerfile: docker is not installed in this environment, so "
                "the image was never built. Structure and the healthcheck pattern "
                "were checked; the build was not."
            ],
            "reason": "no CUDA device was available to the build environment",
        },
        "sentence_streaming": {
            "enabled_by_default": SETTINGS.sentence_streaming,
            "what_it_saves": (
                "time-to-first-audible-word, by translating sentence 1 and sending "
                "it while sentences 2..n are still in flight"
            ),
            "what_it_does_NOT_save": (
                "total_server_ms. n small MT calls carry slightly MORE total "
                "overhead than one large call, so the total is unchanged or very "
                "slightly worse. This is why first_sentence_ms is reported as its "
                "own field and never folded into the total."
            ),
            "per_connection_opt_out": "send {'stream': false} on the WebSocket",
        },
    }


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "Lingua Buds Translation Server",
        "version": VERSION,
        "endpoints": {
            "ws": "/ws/v1/translate-stream",
            "rest": "POST /translate",
            "health": "/health",
            "metrics": "/metrics",
            "languages": "/languages",
            "capabilities": "/capabilities",
            "docs": "/docs",
        },
    }


@app.exception_handler(RequestError)
async def _request_error_handler(_: Request, exc: RequestError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": exc.code, "detail": str(exc)})
