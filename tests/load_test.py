"""Concurrent load test: measure what the server actually does under pressure.

    python tests/load_test.py [base_url] [concurrency] [requests]

This exists to answer one question with numbers rather than opinion: when
demand exceeds capacity, does the server (a) shed load with an explicit 503,
or (b) accept everything and let latency quietly diverge? The brief's target
(100 concurrent users under 150 ms on one RTX 3060) is arithmetically out of
reach, so (a) is the designed behaviour and this test proves it happens.

It reports the served p50/p95/p99, the shed count, and -- the figure that
matters most -- how many requests met the budget as a fraction of ALL offered
load, not just of those served. A server that serves 5% inside 150 ms and
rejects the other 95% has not met the target; it has only been honest about
missing it.
"""

from __future__ import annotations

import asyncio
import statistics as stats
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
CONCURRENCY = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 64

PHRASES = [
    ("Where is the nearest pharmacy?", "en", "ar"),
    ("I need a doctor now, please help me.", "en", "ar"),
    ("How much does this cost?", "en", "fr"),
    ("أين أقرب صيدلية؟", "ar", "en"),
    ("Thank you very much.", "en", "ja"),
]


# Minimum samples before a percentile means anything. Same rule the server
# applies in /metrics: this test printed "p95 = 2662.0 ms" from 2 samples on an
# earlier run, which is not a p95, it is the maximum wearing a label.
_MIN_SAMPLES = {0.50: 3, 0.95: 20, 0.99: 100}


def pct(values: list[float], q: float) -> float | None:
    if not values or len(values) < _MIN_SAMPLES.get(q, 1):
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def fmt_pct(values: list[float], q: float) -> str:
    value = pct(values, q)
    if value is None:
        need = _MIN_SAMPLES.get(q, 1)
        return f"withheld ({len(values)} samples < {need} needed)"
    return f"{value:.1f} ms"


async def one(client: httpx.AsyncClient, index: int, sem: asyncio.Semaphore) -> dict:
    text, src, dst = PHRASES[index % len(PHRASES)]
    async with sem:
        started = time.perf_counter()
        try:
            response = await client.post(
                f"{BASE}/translate",
                json={"text": text, "source": src, "target": dst},
                timeout=180.0,
            )
        except Exception as exc:
            return {"outcome": "error", "detail": f"{type(exc).__name__}: {exc}"}
        wall_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code == 503:
            body = response.json()
            return {
                "outcome": "shed",
                "wall_ms": wall_ms,
                "projected_wait_ms": body.get("projected_wait_ms"),
            }
        if response.status_code != 200:
            return {"outcome": "error", "detail": f"HTTP {response.status_code}"}
        body = response.json()
        return {
            "outcome": "served",
            "wall_ms": wall_ms,
            "server_ms": body.get("total_server_ms"),
            "mt_ms": body.get("mt_ms"),
            "queue_ms": body.get("queue_ms"),
            "within_budget": bool(body.get("within_budget")),
            "empty": not (body.get("translation") or "").strip(),
        }


async def main() -> int:
    print(f"target      : {BASE}")
    print(f"concurrency : {CONCURRENCY}")
    print(f"requests    : {TOTAL}\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        results = await asyncio.gather(*(one(client, i, sem) for i in range(TOTAL)))
        wall_s = time.perf_counter() - wall_start
        # Capacity figures come from the server's own measurements, not from
        # this client's guesses.
        after = (await client.get(f"{BASE}/metrics", timeout=30)).json()

    served = [r for r in results if r["outcome"] == "served"]
    shed = [r for r in results if r["outcome"] == "shed"]
    errors = [r for r in results if r["outcome"] == "error"]
    empties = [r for r in served if r.get("empty")]

    server_ms = [r["server_ms"] for r in served if isinstance(r.get("server_ms"), (int, float))]
    within = [r for r in served if r["within_budget"]]

    print("=" * 68)
    print(f"offered load        : {TOTAL} requests at concurrency {CONCURRENCY}")
    print(f"wall clock          : {wall_s:.2f} s")
    print(f"offered rate        : {TOTAL / wall_s:.2f} req/s")
    print(f"served              : {len(served)}")
    print(f"shed (503)          : {len(shed)}")
    print(f"errors              : {len(errors)}")
    print(f"empty translations  : {len(empties)}  <- must be 0")
    print(f"goodput             : {len(served) / wall_s:.2f} req/s")
    print("-" * 68)
    if server_ms:
        print(f"served p50          : {fmt_pct(server_ms, 0.50)}")
        print(f"served p95          : {fmt_pct(server_ms, 0.95)}")
        print(f"served p99          : {fmt_pct(server_ms, 0.99)}")
        print(f"served min/max      : {min(server_ms):.1f} / {max(server_ms):.1f} ms")
        print(f"served mean         : {stats.mean(server_ms):.1f} ms  (n={len(server_ms)})")
    else:
        print("served p50          : n/a (nothing was served)")
    budget = after.get("budget", {}).get("budget_ms")
    print("-" * 68)
    print(f"latency budget      : {budget} ms")
    print(
        f"served within budget: {len(within)}/{len(served)}"
        + (f" ({100.0 * len(within) / len(served):.1f}%)" if served else "")
    )
    print(
        f"of ALL offered      : {len(within)}/{TOTAL}"
        f" ({100.0 * len(within) / TOTAL:.1f}%)   <- the honest headline"
    )
    print("-" * 68)
    for name in ("asr", "mt"):
        q = after.get("queues", {}).get(name, {})
        print(
            f"{name:>3} queue          : per_item={q.get('measured_per_item_ms')} ms  "
            f"capacity={q.get('capacity_rps')} rps  rejected={q.get('rejected')}  "
            f"peak_depth={q.get('peak_queue_depth')}"
        )
        if q.get("budget_unreachable_on_this_device"):
            print(f"                    ! {q.get('budget_unreachable_note')}")
    print("=" * 68)

    verdict: list[str] = []
    if empties:
        verdict.append(f"BROKEN: {len(empties)} served requests returned an empty translation")
    if errors:
        verdict.append(f"ERRORS: {len(errors)} requests failed: {errors[0].get('detail')}")
    if shed:
        verdict.append(
            f"Load shedding worked: {len(shed)} requests were refused with an explicit "
            "503 + Retry-After instead of being absorbed into a growing queue."
        )
    if served and not within:
        verdict.append(
            f"No served request met the {budget} ms budget on this device. That is the "
            "measurement, not a defect: see /capabilities for the arithmetic."
        )
    for line in verdict:
        print(f"  * {line}")

    # Fails only on incorrectness, not on missing a target the arithmetic
    # already says is unreachable on this hardware.
    return 1 if (empties or errors) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
