"""Latency and load metrics.

Rules this module exists to enforce:

  1. A percentile computed from too few samples is not a percentile. Every
     reported figure carries the sample count that produced it, and p95/p99
     are `null` until enough samples exist to place them (>= 20 / >= 100).
  2. A missing measurement and a zero are different facts and must never
     print the same. Absent values are `null` with an explicit
     `available: false`, never 0.0.
  3. Counting only successes flatters the numbers. Rejections, errors and
     empty-transcript ("hollow") results are counted separately and are
     visible in the same payload as the latencies.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Final

_P95_MIN_SAMPLES: Final[int] = 20
_P99_MIN_SAMPLES: Final[int] = 100


def percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile. `sorted_values` must be sorted, non-empty."""
    if not sorted_values:
        raise ValueError("percentile of empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


@dataclass
class Series:
    """A bounded window of latency samples."""

    name: str
    window: int = 2048
    _values: deque[float] = field(default_factory=deque, repr=False)
    _count_total: int = 0
    _sum_total: float = 0.0
    _max_total: float = 0.0

    def observe(self, value_ms: float) -> None:
        if value_ms < 0 or value_ms != value_ms:  # negative or NaN
            return
        if len(self._values) >= self.window:
            self._values.popleft()
        self._values.append(value_ms)
        self._count_total += 1
        self._sum_total += value_ms
        self._max_total = max(self._max_total, value_ms)

    def snapshot(self) -> dict[str, Any]:
        vals = sorted(self._values)
        n = len(vals)
        if n == 0:
            return {
                "available": False,
                "samples": 0,
                "samples_total": self._count_total,
                "reason": "no samples recorded yet",
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "mean_ms": None,
                "min_ms": None,
                "max_ms": None,
                "max_ever_ms": None,
            }
        return {
            "available": True,
            "samples": n,
            "samples_total": self._count_total,
            "window": self.window,
            "p50_ms": round(percentile(vals, 0.50), 2),
            # Withheld rather than fabricated from too few points.
            "p95_ms": round(percentile(vals, 0.95), 2) if n >= _P95_MIN_SAMPLES else None,
            "p95_note": None if n >= _P95_MIN_SAMPLES
            else f"withheld: {n} samples < {_P95_MIN_SAMPLES} needed",
            "p99_ms": round(percentile(vals, 0.99), 2) if n >= _P99_MIN_SAMPLES else None,
            "p99_note": None if n >= _P99_MIN_SAMPLES
            else f"withheld: {n} samples < {_P99_MIN_SAMPLES} needed",
            "mean_ms": round(sum(vals) / n, 2),
            "min_ms": round(vals[0], 2),
            "max_ms": round(vals[-1], 2),
            "max_ever_ms": round(self._max_total, 2),
        }


class Metrics:
    """Thread-safe metric registry. Cheap enough to call on every request."""

    def __init__(self, window: int = 2048, budget_ms: float = 150.0) -> None:
        self._lock = threading.Lock()
        self._window = window
        self._budget_ms = budget_ms
        self._series: dict[str, Series] = {}
        self._counters: dict[str, int] = {
            "seq_gap": 0,
            "late_frame_dropped": 0,
            "duplicate_preroll": 0,
            "bad_frame": 0,
            "auto_commit": 0,
            "auto_commit_timeout": 0,
            "guard_dropped_segments": 0,
            "person_mismatch_observed": 0,
            "tentative_hit": 0,
            "tentative_await": 0,
            "tentative_miss": 0,
            "tentative_cancelled": 0,
        }
        self._started_at = time.time()
        self._inflight = 0
        self._peak_inflight = 0

    # ---- series --------------------------------------------------------
    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            s = self._series.get(name)
            if s is None:
                s = Series(name=name, window=self._window)
                self._series[name] = s
            s.observe(value_ms)

    def observe_many(self, values: dict[str, float]) -> None:
        for k, v in values.items():
            self.observe(k, v)

    # ---- counters ------------------------------------------------------
    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    # ---- inflight ------------------------------------------------------
    def enter(self) -> None:
        with self._lock:
            self._inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._inflight)

    def leave(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    # ---- reporting -----------------------------------------------------
    def budget_report(self) -> dict[str, Any]:
        """How often we actually met the latency budget. The honest headline."""
        with self._lock:
            series = self._series.get("total_server_ms")
            vals = sorted(series._values) if series else []
            n = len(vals)
        if n == 0:
            return {
                "available": False,
                "budget_ms": self._budget_ms,
                "reason": "no completed requests measured yet",
                "within_budget_fraction": None,
                "samples": 0,
            }
        within = sum(1 for v in vals if v <= self._budget_ms)
        return {
            "available": True,
            "budget_ms": self._budget_ms,
            "samples": n,
            "within_budget": within,
            "within_budget_fraction": round(within / n, 4),
            "p50_ms": round(percentile(vals, 0.50), 2),
            "p95_ms": round(percentile(vals, 0.95), 2) if n >= _P95_MIN_SAMPLES else None,
            "worst_ms": round(vals[-1], 2),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            names = list(self._series)
            counters = dict(self._counters)
            inflight = self._inflight
            peak = self._peak_inflight
            uptime = time.time() - self._started_at
        latencies = {name: self._series[name].snapshot() for name in names}
        completed = counters.get("requests_completed", 0)
        return {
            "uptime_seconds": round(uptime, 1),
            "inflight": inflight,
            "peak_inflight": peak,
            "counters": counters,
            "latency": latencies,
            "budget": self.budget_report(),
            "throughput_rps_since_start": (
                round(completed / uptime, 3) if uptime > 0 and completed else 0.0
            ),
        }


def resource_report() -> dict[str, Any]:
    """Process RSS + GPU memory, with 'unavailable' distinct from zero."""
    report: dict[str, Any] = {"cpu": _cpu_memory(), "gpu": _gpu_memory()}
    return report


def _cpu_memory() -> dict[str, Any]:
    try:
        with open("/proc/self/status", "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return {
                        "available": True,
                        "rss_mb": round(kb / 1024.0, 1),
                        "source": "/proc/self/status",
                    }
    except Exception:
        pass
    try:  # Windows fallback, so this never reports a false zero there either.
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
            handle, ctypes.byref(counters), counters.cb
        )
        if ok:
            return {
                "available": True,
                "rss_mb": round(counters.WorkingSetSize / (1024.0 * 1024.0), 1),
                "source": "GetProcessMemoryInfo",
            }
    except Exception:
        pass
    return {"available": False, "rss_mb": None, "source": "unavailable"}


def _gpu_memory() -> dict[str, Any]:
    """VRAM via torch.cuda, else pynvml, else an explicit 'unavailable'.

    This must not report 0 MB on a CPU box: that would read as "the GPU is
    idle" when the truth is "there is no GPU".
    """
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            free_b, total_b = torch.cuda.mem_get_info(idx)
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(idx),
                "total_mb": round(total_b / (1024.0**2), 1),
                "free_mb": round(free_b / (1024.0**2), 1),
                "used_mb": round((total_b - free_b) / (1024.0**2), 1),
                "torch_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024.0**2), 1),
                "torch_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024.0**2), 1),
                "source": "torch.cuda.mem_get_info",
            }
    except Exception:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return {
            "available": True,
            "device_name": name.decode() if isinstance(name, bytes) else name,
            "total_mb": round(info.total / (1024.0**2), 1),
            "free_mb": round(info.free / (1024.0**2), 1),
            "used_mb": round(info.used / (1024.0**2), 1),
            "gpu_util_percent": util.gpu,
            "source": "pynvml",
        }
    except Exception:
        pass
    return {
        "available": False,
        "total_mb": None,
        "used_mb": None,
        "source": "unavailable",
        "reason": "no CUDA device visible to this process (not the same as 0 MB used)",
    }
