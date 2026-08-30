"""Configuration for the Lingua Buds translation server.

Every value is overridable by environment variable so the same image runs on
the RTX 3060 and on a CPU box (this sandbox) without code changes.

Design note on defaults
-----------------------
The defaults here are deliberately *not* the values from the project brief.
The brief asked for whisper-large-v3-turbo + Qwen2.5-1.5B fp16 on a 12 GB
RTX 3060. Facts that shaped these defaults, measured by reading the real
safetensors headers over HTTP range requests rather than by multiplying a
parameter count:

  * Qwen2.5-1.5B fp16       model.safetensors = 3.087 GB  (MEASURED)
    Qwen2.5-1.5B-Instruct-AWQ                = 1.614 GB  (MEASURED)

    The fp16 figure alone exceeds the brief's 1.5-2.0 GB budget for the whole
    MT stage, before KV cache, activations and the CUDA context (0.3-0.6 GB).

  * AWQ is 1.614 GB and NOT the 0.72 GB an earlier estimate in this project
    claimed. The AWQ file keeps embed_tokens (0.467 GB) and lm_head
    (0.467 GB) in fp16; only the 196 linear tensors are int4 (0.655 GB).
    Together the unquantised parts outweigh the quantised ones.

  * RTX 3060 12GB has 360 GB/s (192-bit GDDR6) and a decode step must read
    every resident weight, so at the MEASURED median of 8 output tokens:
        fp16      3.087 GB -> 117 tok/s -> 68.6 ms ideal / 98.0 ms at 70% BW
        AWQ int4  1.614 GB -> 223 tok/s -> 35.9 ms ideal / 51.2 ms at 70% BW
    AWQ therefore does NOT hit the brief's 20-35 ms; it lands near 36-51 ms.

`MT_QUANT` defaults to **awq** because it is ~2x faster than fp16 and 1.5 GB
lighter, which is what makes the 150 ms whole-request budget plausible once
ASR is added -- not because it reaches a number it cannot reach. Requesting
awq/int4 also redirects MT_MODEL to the official `-AWQ` checkpoint, since
vLLM expects already-quantised weights and cannot quantise fp16 at load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Final

_TRUE: Final = {"1", "true", "yes", "on"}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in _TRUE


def _auto_device() -> str:
    """Detect CUDA without importing torch at module import time.

    ctranslate2 is always installed here (faster-whisper depends on it) and it
    reports its own CUDA device count, which is exactly the thing that decides
    whether faster-whisper can use the GPU.
    """
    override = os.environ.get("DEVICE", "").strip().lower()
    if override in {"cuda", "cpu"}:
        return override
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass(frozen=True)
class Settings:
    # ---- runtime -------------------------------------------------------
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))
    device: str = field(default_factory=_auto_device)

    # ---- ASR -----------------------------------------------------------
    # On CUDA the brief's model; on CPU a small model, because the sandbox has
    # 2 cores / 985 MB RAM / 2.4 GB free disk and cannot host large-v3-turbo.
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", ""))
    asr_compute_type: str = field(default_factory=lambda: _env("ASR_COMPUTE_TYPE", ""))
    asr_beam_size: int = field(default_factory=lambda: _env_int("ASR_BEAM_SIZE", 1))
    asr_vad_filter: bool = field(default_factory=lambda: _env_bool("ASR_VAD_FILTER", True))
    asr_cpu_threads: int = field(default_factory=lambda: _env_int("ASR_CPU_THREADS", 0))
    asr_batch_size: int = field(default_factory=lambda: _env_int("ASR_BATCH_SIZE", 8))
    asr_batch_wait_ms: float = field(default_factory=lambda: _env_float("ASR_BATCH_WAIT_MS", 8.0))

    # ---- MT ------------------------------------------------------------
    # backend: auto | qwen_vllm | qwen_ct2 | qwen_hf | m2m100_ct2
    mt_backend: str = field(default_factory=lambda: _env("MT_BACKEND", "auto"))
    mt_model: str = field(default_factory=lambda: _env("MT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    mt_quant: str = field(default_factory=lambda: _env("MT_QUANT", "awq"))
    mt_max_new_tokens: int = field(default_factory=lambda: _env_int("MT_MAX_NEW_TOKENS", 96))
    mt_batch_size: int = field(default_factory=lambda: _env_int("MT_BATCH_SIZE", 16))
    mt_batch_wait_ms: float = field(default_factory=lambda: _env_float("MT_BATCH_WAIT_MS", 5.0))
    mt_gpu_mem_fraction: float = field(
        default_factory=lambda: _env_float("MT_GPU_MEM_FRACTION", 0.45)
    )
    # CPU verification path: a real CT2 translation model that fits this box.
    mt_cpu_model_path: str = field(default_factory=lambda: _env("MT_CPU_MODEL_PATH", ""))

    # ---- admission control --------------------------------------------
    # The brief asks for a fixed <150 ms under 100 concurrent users. Measured
    # queueing arithmetic says a single 3060 cannot hold that (see README), so
    # the server refuses work it cannot finish in time instead of letting
    # latency diverge silently.
    latency_budget_ms: float = field(default_factory=lambda: _env_float("LATENCY_BUDGET_MS", 150.0))
    admission_enabled: bool = field(default_factory=lambda: _env_bool("ADMISSION_ENABLED", True))
    max_queue_depth: int = field(default_factory=lambda: _env_int("MAX_QUEUE_DEPTH", 256))
    max_inflight: int = field(default_factory=lambda: _env_int("MAX_INFLIGHT", 512))
    reject_over_budget: bool = field(default_factory=lambda: _env_bool("REJECT_OVER_BUDGET", True))
    # How far past the budget we tolerate before shedding load. 1.0 == shed as
    # soon as the projected wait exceeds the budget itself.
    overload_factor: float = field(default_factory=lambda: _env_float("OVERLOAD_FACTOR", 1.0))

    # ---- sentence streaming --------------------------------------------
    # MEASURED in the existing app: waiting for the whole utterance before any
    # audio plays costs ~595 ms before the first audible word. Translating and
    # emitting sentence 1 immediately lets TTS start while 2..n are still in
    # flight. This does NOT reduce total server time -- n small MT calls cost
    # slightly more overhead than one large call -- so the win is reported
    # separately as `first_sentence_ms`, never folded into total_server_ms.
    sentence_streaming: bool = field(
        default_factory=lambda: _env_bool("SENTENCE_STREAMING", True)
    )
    # Fragments shorter than this (in script-weighted units, so CJK is not
    # penalised) are merged: a 2-syllable fragment spends a whole MT round
    # trip on nothing and gives the model no context to translate correctly.
    sentence_min_chars: int = field(default_factory=lambda: _env_int("SENTENCE_MIN_CHARS", 12))
    # Cap on pieces per utterance. Past this the per-call overhead outweighs
    # the overlap gain, so the tail is merged.
    sentence_max_count: int = field(default_factory=lambda: _env_int("SENTENCE_MAX_COUNT", 8))

    # ---- audio ---------------------------------------------------------
    sample_rate: int = field(default_factory=lambda: _env_int("SAMPLE_RATE", 16000))
    max_utterance_seconds: float = field(
        default_factory=lambda: _env_float("MAX_UTTERANCE_SECONDS", 30.0)
    )
    min_utterance_seconds: float = field(
        default_factory=lambda: _env_float("MIN_UTTERANCE_SECONDS", 0.20)
    )

    # ---- misc ----------------------------------------------------------
    metrics_window: int = field(default_factory=lambda: _env_int("METRICS_WINDOW", 2048))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    warmup: bool = field(default_factory=lambda: _env_bool("WARMUP", True))

    # ------------------------------------------------------------------
    @property
    def on_cuda(self) -> bool:
        return self.device == "cuda"

    def resolved_asr_model(self) -> str:
        if self.asr_model:
            return self.asr_model
        return "large-v3-turbo" if self.on_cuda else "tiny"

    def resolved_asr_compute_type(self) -> str:
        if self.asr_compute_type:
            return self.asr_compute_type
        return "float16" if self.on_cuda else "int8"

    def resolved_asr_cpu_threads(self) -> int:
        if self.asr_cpu_threads > 0:
            return self.asr_cpu_threads
        return max(1, (os.cpu_count() or 2))

    def describe(self) -> dict[str, Any]:
        data = asdict(self)
        data["resolved_asr_model"] = self.resolved_asr_model()
        data["resolved_asr_compute_type"] = self.resolved_asr_compute_type()
        return data


SETTINGS: Final[Settings] = Settings()
