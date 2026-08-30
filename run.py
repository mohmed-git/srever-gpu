"""One-command entry point.

    python run.py

Reads everything from the environment (see .env.example). Uses a single
uvicorn worker on purpose: the GPU is the shared resource, and two workers
would each load their own copy of Whisper and Qwen, doubling VRAM for no
throughput gain. Concurrency comes from the async batching scheduler inside
the process, not from forking.
"""

from __future__ import annotations

import sys

import uvicorn

from app.config import SETTINGS


def main() -> int:
    print(
        f"Lingua Buds translation server\n"
        f"  device          : {SETTINGS.device}\n"
        f"  ASR             : {SETTINGS.resolved_asr_model()} "
        f"({SETTINGS.resolved_asr_compute_type()}, vad={SETTINGS.asr_vad_filter})\n"
        f"  MT backend      : {SETTINGS.mt_backend} (quant={SETTINGS.mt_quant})\n"
        f"  latency budget  : {SETTINGS.latency_budget_ms:.0f} ms\n"
        f"  admission ctrl  : {'on' if SETTINGS.admission_enabled else 'off'}\n"
        f"  listening on    : http://{SETTINGS.host}:{SETTINGS.port}\n",
        flush=True,
    )
    if SETTINGS.device == "cpu":
        print(
            "NOTE: no CUDA device detected. This process will run, and every "
            "number it reports will be real, but a CPU cannot meet the "
            "150 ms budget: expect /health to show within_budget_fraction "
            "near 0. That is the measurement, not a bug.\n",
            flush=True,
        )
    uvicorn.run(
        "app.server:app",
        host=SETTINGS.host,
        port=SETTINGS.port,
        log_level=SETTINGS.log_level.lower(),
        workers=1,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
        timeout_keep_alive=30,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
