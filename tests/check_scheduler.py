"""Verify the admission-controlled scheduler by measurement, not by reading it.

Checks:
  1. Batching really coalesces: 16 concurrent submits must run in fewer than
     16 batches.
  2. Results are not crossed: every caller gets the answer to its own item.
  3. Admission actually sheds: with a tiny queue and a slow runner, some
     submits must raise Overloaded rather than queueing forever.
  4. Admission never refuses on an unmeasured estimate: the very first request
     must be admitted, because no batch has completed yet.
  5. A failing batch fails its callers and does not kill the worker.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.scheduler import BatchScheduler, Overloaded  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"[{'ok' if condition else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        failures.append(label)


async def test_batching_and_identity() -> None:
    calls: list[int] = []

    def runner(items: list[int]) -> list[int]:
        calls.append(len(items))
        time.sleep(0.02)  # simulate GPU work; releases the GIL like CT2/vLLM do
        return [i * 10 for i in items]

    sched: BatchScheduler[int, int] = BatchScheduler(
        "batch", runner, max_batch=8, wait_ms=5.0, budget_ms=10_000.0
    )
    await sched.start()
    started = time.perf_counter()
    results = await asyncio.gather(*(sched.submit(i) for i in range(16)))
    elapsed = (time.perf_counter() - started) * 1000.0

    values = [r[0] for r in results]
    check(values == [i * 10 for i in range(16)], "results match their own inputs", str(values[:5]))
    check(len(calls) < 16, "batching coalesced", f"{len(calls)} batches for 16 items: {calls}")
    check(max(calls) > 1, "at least one real multi-item batch", f"max batch = {max(calls)}")
    stats = sched.stats()
    check(
        stats["measured_per_item_ms"] is not None,
        "per-item service time measured",
        f"{stats['measured_per_item_ms']} ms, capacity {stats['capacity_rps']} rps",
    )
    print(f"       16 items in {elapsed:.1f} ms, mean batch {stats['mean_batch_size']}")
    await sched.stop()


async def test_first_request_admitted() -> None:
    def runner(items: list[int]) -> list[int]:
        time.sleep(0.05)
        return list(items)

    sched: BatchScheduler[int, int] = BatchScheduler(
        "first", runner, max_batch=1, wait_ms=0.0, budget_ms=1.0  # absurdly small budget
    )
    await sched.start()
    try:
        await sched.submit(1)
        admitted = True
        detail = "admitted with no prior measurement, as intended"
    except Overloaded as exc:
        admitted = False
        detail = f"refused on a guess: {exc}"
    check(admitted, "first request admitted before any measurement exists", detail)
    stats = sched.stats()
    check(
        stats["projection_available"] is True,
        "projection becomes available after one batch",
        f"projected {stats['projected_wait_ms']} ms",
    )
    await sched.stop()


async def test_admission_sheds() -> None:
    def runner(items: list[int]) -> list[int]:
        time.sleep(0.10)  # 100 ms per batch: well over the budget below
        return list(items)

    sched: BatchScheduler[int, int] = BatchScheduler(
        "shed",
        runner,
        max_batch=1,
        wait_ms=0.0,
        max_queue_depth=4,
        budget_ms=20.0,
        admission_enabled=True,
        reject_over_budget=True,
    )
    await sched.start()
    await sched.submit(0)  # establishes the measurement

    async def attempt(i: int) -> str:
        try:
            await sched.submit(i)
            return "ok"
        except Overloaded:
            return "shed"

    outcomes = await asyncio.gather(*(attempt(i) for i in range(1, 21)))
    shed = outcomes.count("shed")
    served = outcomes.count("ok")
    check(shed > 0, "overload is signalled, not absorbed", f"{shed} shed / {served} served of 20")
    stats = sched.stats()
    check(stats["rejected"] == shed, "rejections are counted", f"counter = {stats['rejected']}")
    await sched.stop()


async def test_failure_isolation() -> None:
    state = {"calls": 0}

    def runner(items: list[int]) -> list[int]:
        state["calls"] += 1
        if state["calls"] == 1:
            raise ValueError("simulated backend failure")
        return [i + 1 for i in items]

    sched: BatchScheduler[int, int] = BatchScheduler(
        "fail", runner, max_batch=1, wait_ms=0.0, budget_ms=10_000.0
    )
    await sched.start()
    try:
        await sched.submit(1)
        raised = False
    except ValueError:
        raised = True
    check(raised, "a failing batch raises to its caller")

    # The worker must still be alive.
    result, _ = await sched.submit(41)
    check(result == 42, "worker survived the failure", f"got {result}")
    await sched.stop()


async def test_shutdown_fails_pending() -> None:
    def runner(items: list[int]) -> list[int]:
        time.sleep(0.30)
        return list(items)

    sched: BatchScheduler[int, int] = BatchScheduler(
        "shutdown", runner, max_batch=1, wait_ms=0.0, budget_ms=10_000.0, admission_enabled=False
    )
    await sched.start()
    tasks = [asyncio.create_task(sched.submit(i)) for i in range(4)]
    await asyncio.sleep(0.05)
    await sched.stop()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    hung = [o for o in outcomes if isinstance(o, asyncio.CancelledError)]
    check(len(hung) == 0, "no caller is left hanging on shutdown", f"outcomes={len(outcomes)}")


async def main() -> int:
    for test in (
        test_batching_and_identity,
        test_first_request_admitted,
        test_admission_sheds,
        test_failure_isolation,
        test_shutdown_fails_pending,
    ):
        print(f"\n--- {test.__name__} ---")
        await test()
    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
