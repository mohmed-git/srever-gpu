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
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import languages as lang_mod
from .config import SETTINGS, Settings
from .metrics import resource_report
from .pipeline import Pipeline, RequestError
from .scheduler import Overloaded

VERSION = "1.0.0"

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lingua.server")

PIPELINE: Pipeline | None = None
STARTUP_ERROR: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE, STARTUP_ERROR
    settings = SETTINGS
    log.info(
        "starting: device=%s asr=%s/%s mt_backend=%s budget=%.0fms",
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
    except Exception as exc:
        # A server that answers /health with "ok" while its engines failed to
        # load is worse than one that refuses to serve. Record and keep the
        # process up so /health can explain what went wrong.
        STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
        log.error("startup failed: %s", STARTUP_ERROR)
    yield
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

    @field_validator("source", "target")
    @classmethod
    def _norm(cls, value: str) -> str:
        return value.strip().lower()


@app.post("/translate")
async def translate(req: TranslateRequest) -> JSONResponse:
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
        outcome = await PIPELINE.translate_text(req.text, req.source, req.target)
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
    "format",
    "sample_rate",
    "channels",
    "action",
    # Per-connection opt-out from sentence streaming. A client that wants
    # exactly one JSON per utterance sends {"stream": false} once.
    "stream",
}


class _StreamState:
    """Per-connection state: languages plus the audio buffer for one utterance."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.source: str | None = None
        self.target: str | None = None
        self.audio_format: str | None = None
        self.sample_rate: int = settings.sample_rate
        self.channels: int = 1
        self.buffer = bytearray()
        self.utterances = 0
        # WHY a per-connection switch and not just the global setting: the
        # existing client contract is ONE unified JSON per utterance. If the
        # server unilaterally started emitting n+1 frames, every already-shipped
        # client would read frame 1 as the whole answer and act on a partial
        # translation. So streaming stays opt-outable per connection, and the
        # `ready` frame below announces which mode this connection is in.
        self.stream: bool = settings.sentence_streaming

    @property
    def max_bytes(self) -> int:
        # int16 mono at the declared rate.
        return int(self.settings.max_utterance_seconds * self.sample_rate * 2 * self.channels)

    def apply(self, message: dict[str, Any]) -> None:
        if "source" in message:
            self.source = str(message["source"]) if message["source"] is not None else None
        if "target" in message:
            self.target = str(message["target"]) if message["target"] is not None else None
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
        if "stream" in message:
            # Accept the JSON booleans and the string forms clients actually
            # send. Anything unrecognised leaves the mode untouched rather than
            # silently flipping to a mode the client did not ask for.
            value = message["stream"]
            if isinstance(value, bool):
                self.stream = value
            elif isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    self.stream = True
                elif lowered in {"0", "false", "no", "off"}:
                    self.stream = False


@app.websocket("/ws/v1/translate-stream")
async def translate_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = SETTINGS

    if PIPELINE is None or not PIPELINE.ready:
        await websocket.send_json(
            {
                "error": "not_ready",
                "detail": STARTUP_ERROR or "engines still loading",
            }
        )
        await websocket.close(code=1013)
        return

    state = _StreamState(settings)
    await websocket.send_json(
        {
            "event": "ready",
            "version": VERSION,
            "protocol": {
                "send_json": {
                    "source": "language code or 'auto'",
                    "target": "language code (required)",
                    "format": "pcm_s16le | opus | ogg | webm | wav (optional; magic bytes win)",
                    "sample_rate": settings.sample_rate,
                    "channels": 1,
                    "action": "'flush' to end the utterance and translate now",
                    "stream": (
                        "true (default) for one frame per sentence then a "
                        "'final' frame; false for a single unified JSON"
                    ),
                },
                "send_binary": "audio chunks; buffered until you send {'action':'flush'}",
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
            # Announced, not assumed: a client that reads frame 1 as the whole
            # answer needs to know before it sends audio whether more frames
            # are coming.
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
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            # ---- control frames -------------------------------------
            text_payload = message.get("text")
            if text_payload is not None:
                try:
                    control = json.loads(text_payload)
                except json.JSONDecodeError as exc:
                    await websocket.send_json(
                        {"error": "bad_json", "detail": f"control frame is not JSON: {exc}"}
                    )
                    continue
                if not isinstance(control, dict):
                    await websocket.send_json(
                        {"error": "bad_json", "detail": "control frame must be a JSON object"}
                    )
                    continue

                unknown = set(control) - _CONTROL_KEYS
                state.apply(control)
                action = str(control.get("action", "")).strip().lower()

                if action in {"flush", "end", "eou"}:
                    await _handle_utterance(websocket, state)
                    continue
                if action == "reset":
                    state.buffer.clear()
                    await websocket.send_json({"event": "reset", "buffered_bytes": 0})
                    continue
                if action == "close":
                    break

                await websocket.send_json(
                    {
                        "event": "config",
                        "source": state.source,
                        "target": state.target,
                        "format": state.audio_format,
                        "sample_rate": state.sample_rate,
                        "channels": state.channels,
                        # Silently ignoring an unknown key is how a client ends
                        # up believing it set something it did not.
                        "ignored_keys": sorted(unknown) or None,
                    }
                )
                continue

            # ---- audio frames ---------------------------------------
            chunk = message.get("bytes")
            if chunk is None:
                continue
            if len(state.buffer) + len(chunk) > state.max_bytes:
                state.buffer.clear()
                await websocket.send_json(
                    {
                        "error": "utterance_too_long",
                        "detail": (
                            f"buffered audio exceeded {settings.max_utterance_seconds}s; "
                            "buffer cleared. Send {'action':'flush'} at end of speech."
                        ),
                    }
                )
                continue
            state.buffer.extend(chunk)

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket handler failed")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


async def _handle_utterance(websocket: WebSocket, state: _StreamState) -> None:
    if not state.buffer:
        await websocket.send_json(
            {"error": "empty_utterance", "detail": "flush received but no audio was buffered"}
        )
        return
    if not state.target:
        await websocket.send_json(
            {
                "error": "missing_target",
                "detail": "send {'target': '<lang>'} before flushing audio",
            }
        )
        state.buffer.clear()
        return

    raw = bytes(state.buffer)
    state.buffer.clear()
    state.utterances += 1

    assert PIPELINE is not None

    if state.stream:
        await _stream_utterance(websocket, state, raw)
        return

    try:
        outcome = await PIPELINE.translate_audio(
            raw,
            source=state.source or "auto",
            target=state.target,
            declared_format=state.audio_format,
            input_sample_rate=state.sample_rate,
            channels=state.channels,
        )
    except Overloaded as exc:
        await websocket.send_json({**exc.payload(), "utterance": state.utterances})
        return
    except RequestError as exc:
        await websocket.send_json(
            {"error": exc.code, "detail": str(exc), "utterance": state.utterances}
        )
        return
    except Exception as exc:
        log.exception("utterance failed")
        await websocket.send_json(
            {
                "error": "internal",
                "detail": f"{type(exc).__name__}: {exc}",
                "utterance": state.utterances,
            }
        )
        return

    await websocket.send_json({**outcome.to_json(), "utterance": state.utterances})


async def _stream_utterance(
    websocket: WebSocket, state: _StreamState, raw: bytes
) -> None:
    """Relay the pipeline's per-sentence frames to the client.

    The error handling here is deliberately different from the unified path,
    and the reason is the whole point of this function: once sentence 1 has
    been sent, the earbuds have ALREADY STARTED SPEAKING. A failure after that
    point cannot be reported as if nothing happened -- a client that receives a
    bare {"error": ...} after playing half an utterance has no way to know the
    rest is never coming, and would sit waiting for `final` forever.

    So every error frame carries `frames_sent` and `terminal: true`, and once
    any frame has gone out the error is also tagged `partial: true`. That is
    the difference between "this request failed" and "this request half
    happened", and the client needs to act differently on each.
    """
    assert PIPELINE is not None
    frames = 0
    try:
        async for frame in PIPELINE.translate_audio_streaming(
            raw,
            source=state.source or "auto",
            target=state.target or "",
            declared_format=state.audio_format,
            input_sample_rate=state.sample_rate,
            channels=state.channels,
        ):
            await websocket.send_json({**frame, "utterance": state.utterances})
            frames += 1
        return
    except Overloaded as exc:
        payload = exc.payload()
    except RequestError as exc:
        payload = {"error": exc.code, "detail": str(exc)}
    except Exception as exc:
        log.exception("streamed utterance failed")
        payload = {"error": "internal", "detail": f"{type(exc).__name__}: {exc}"}

    await websocket.send_json(
        {
            "type": "error",
            **payload,
            "utterance": state.utterances,
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
    return JSONResponse(
        content={
            "version": VERSION,
            "device": SETTINGS.device,
            **PIPELINE.metrics.snapshot(),
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
