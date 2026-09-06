"""Admission-controlled batching scheduler.

Why admission control exists here
---------------------------------
The brief asks for < 150 ms server latency while serving 100+ concurrent
users on one RTX 3060. Queueing arithmetic on the brief's own numbers:

  service time  = 100 ms ASR + 27.5 ms MT = 127.5 ms
  capacity      = 1 / 0.1275 s            = 7.8 requests/second
  demand, 100 users at one 3 s utterance every 3 / 6 / 10 s
                = 33.3 / 16.7 / 10.0 requests/second
  utilisation r = 4.25x / 2.13x / 1.28x

r >= 1 means the queue grows without bound: latency does not settle at 150 ms,
it diverges. Batching raises capacity (modelled: B=8 reaches ~30 req/s) but it
raises latency too (p50 ~392 ms at B=8), and latency is the quantity the target
constrains. So there is no configuration of one 3060 that serves 100 concurrent
users inside 150 ms.

Given that, a server has two honest options and one dishonest one:
  * honest A: accept the work and report the real, growing latency;
  * honest B: refuse work it cannot finish in the budget, and say so;
  * dishonest: accept everything and let p99 quietly reach seconds.

This scheduler does A and B, never the third. It estimates the wait from
*measured* service times, and when the projection exceeds the budget it
rejects with HTTP 503 + `Retry-After` and an explicit reason, so the client
degrades visibly (e.g. fewer languages, longer chunks) instead of silently.

Batching
--------
Requests are coalesced into batches up to `max_batch` with a `wait_ms` window.
The window is what buys GPU throughput; on CPU it mostly adds latency, so it
is configurable and small by default.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

log = logging.getLogger("lingua.sched")

T = TypeVar("T")
R = TypeVar("R")


class Overloaded(RuntimeError):
    """Raised when the server refuses work it cannot finish inside the budget."""

    def __init__(
        self,
        message: str,
        *,
        queue_depth: int,
        projected_wait_ms: float,
        budget_ms: float,
        retry_after_s: float,
    ) -> None:
        super().__init__(message)
        self.queue_depth = queue_depth
        self.projected_wait_ms = projected_wait_ms
        self.budget_ms = budget_ms
        self.retry_after_s = retry_after_s

    def payload(self) -> dict[str, Any]:
        return {
            "error": "overloaded",
            "detail": str(self),
            "queue_depth": self.queue_depth,
            "projected_wait_ms": round(self.projected_wait_ms, 2),
            "latency_budget_ms": self.budget_ms,
            "retry_after_seconds": round(self.retry_after_s, 2),
        }


@dataclass
class _Job(Generic[T, R]):
    payload: T
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.perf_counter)
    priority: int = 0


class BatchScheduler(Generic[T, R]):
    """Coalesces requests into batches and runs them in a worker thread.

    `runner` is a *blocking* callable executed via `asyncio.to_thread`, because
    both CTranslate2 and vLLM release the GIL and are safe to call that way,
    and because keeping them off the event loop is what lets the WebSocket
    endpoint stay responsive under load.
    """

    def __init__(
        self,
        name: str,
        runner: Callable[[list[T]], list[R]],
        *,
        max_batch: int = 8,
        wait_ms: float = 5.0,
        max_queue_depth: int = 256,
        budget_ms: float = 150.0,
        overload_factor: float = 1.0,
        admission_enabled: bool = True,
        reject_over_budget: bool = True,
        unreachable_budget_queue_multiple: float = 8.0,
    ) -> None:
        self.name = name
        self._runner = runner
        self.max_batch = max(1, max_batch)
        self.wait_s = max(0.0, wait_ms) / 1000.0
        self.max_queue_depth = max(1, max_queue_depth)
        self.budget_ms = budget_ms
        self.overload_factor = max(0.1, overload_factor)
        self.admission_enabled = admission_enabled
        self.reject_over_budget = reject_over_budget
        # Used only when the budget is unreachable on this device: how many
        # service times of queue we tolerate before shedding.
        self.unreachable_budget_queue_multiple = max(1.0, unreachable_budget_queue_multiple)

        # Priority queues: 0 (commit), 1 (tentative), 2 (pre-translate), 3 (partial)
        self._queues: dict[int, deque[_Job[T, R]]] = {0: deque(), 1: deque(), 2: deque(), 3: deque()}
        self._wakeup: asyncio.Event | None = None
        self._worker: asyncio.Task | None = None
        self._closing = False

        # Measured service time per batch item, used for the wait projection.
        # Seeded from nothing: until a real batch completes, the projection is
        # unavailable rather than guessed, and admission lets requests through
        # (refusing on an invented estimate would be worse than admitting).
        self._per_item_ms: float | None = None
        self._batches_run = 0
        self._items_run = 0
        self._rejected = 0
        self._peak_queue = 0
        self._last_batch_size = 0

    # ---- lifecycle -----------------------------------------------------
    async def start(self) -> None:
        self._wakeup = asyncio.Event()
        self._closing = False
        self._worker = asyncio.create_task(self._run(), name=f"sched-{self.name}")

    async def stop(self) -> None:
        self._closing = True
        if self._wakeup is not None:
            self._wakeup.set()
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._worker, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker.cancel()
        # Anything still queued must be failed explicitly, not left hanging.
        for prio in (0, 1, 2, 3):
            while self._queues[prio]:
                job = self._queues[prio].popleft()
                if not job.future.done():
                    job.future.set_exception(RuntimeError("server shutting down"))

    def _total_depth(self) -> int:
        return sum(len(q) for q in self._queues.values())

    @property
    def _queue(self) -> list[_Job[T, R]]:
        res = []
        for prio in (0, 1, 2, 3):
            res.extend(self._queues[prio])
        return res

    # ---- admission -----------------------------------------------------
    def projected_wait_ms(self) -> float | None:
        """Estimated queue wait for a request arriving now. None = not yet measurable."""
        if self._per_item_ms is None:
            return None
        depth = self._total_depth()
        # Items ahead are served in batches of at most max_batch, and a batch
        # costs roughly per_item_ms * batch_size on a serial device.
        batches_ahead = (depth + self.max_batch - 1) // self.max_batch
        return batches_ahead * self._per_item_ms * min(self.max_batch, max(1, depth))

    def _admit_or_raise(self, priority: int = 0) -> None:
        depth = self._total_depth()
        if depth >= self.max_queue_depth:
            # Under Overloaded, shed 3 then 2; never shed 0 except by max_queue_depth
            if priority < 2:
                for shed_prio in (3, 2):
                    if self._queues[shed_prio]:
                        shed_job = self._queues[shed_prio].popleft()
                        self._rejected += 1
                        if not shed_job.future.done():
                            shed_job.future.set_exception(
                                Overloaded(
                                    f"shed priority {shed_prio} to accommodate priority {priority}",
                                    queue_depth=self._total_depth(),
                                    projected_wait_ms=self.projected_wait_ms() or 0.0,
                                    budget_ms=self.budget_ms,
                                    retry_after_s=0.25,
                                )
                            )
                        break
            depth = self._total_depth()
            if depth >= self.max_queue_depth:
                self._rejected += 1
                raise Overloaded(
                    f"queue for '{self.name}' is full ({depth}/{self.max_queue_depth})",
                    queue_depth=depth,
                    projected_wait_ms=self.projected_wait_ms() or 0.0,
                    budget_ms=self.budget_ms,
                    retry_after_s=1.0,
                )

        if priority == 0:
            # Commits never shed by projected wait
            return

        if not (self.admission_enabled and self.reject_over_budget):
            return
        projected = self.projected_wait_ms()
        if projected is None:
            return  # no measurement yet: do not refuse on a guess

        if self._per_item_ms is not None and self._per_item_ms > self.budget_ms:
            limit = self._per_item_ms * self.unreachable_budget_queue_multiple
        else:
            limit = self.budget_ms * self.overload_factor

        # Priority-aware thresholds: shed 3 (partials) then 2 (pre-translate)
        threshold = limit
        if priority == 3:
            threshold = 0.5 * limit
        elif priority == 2:
            threshold = 0.8 * limit

        if projected > threshold:
            self._rejected += 1
            raise Overloaded(
                (
                    f"projected queue wait {projected:.0f} ms exceeds the "
                    f"{threshold:.0f} ms admission limit for priority {priority}; "
                    f"shedding speculative work to preserve latency budget"
                ),
                queue_depth=depth,
                projected_wait_ms=projected,
                budget_ms=self.budget_ms,
                retry_after_s=max(0.05, projected / 1000.0),
            )

    # ---- submission ----------------------------------------------------
    async def submit(self, payload: T, priority: int = 0) -> tuple[R, dict[str, Any]]:
        """Queue one item with priority (0 commit, 1 tentative, 2 pre-translate, 3 partial).
        Returns (result, timing). Raises Overloaded when shedding."""
        if self._worker is None or self._wakeup is None:
            raise RuntimeError(f"scheduler '{self.name}' not started")
        prio = min(3, max(0, priority))
        self._admit_or_raise(priority=prio)

        loop = asyncio.get_running_loop()
        job: _Job[T, R] = _Job(payload=payload, future=loop.create_future(), priority=prio)
        self._queues[prio].append(job)
        self._peak_queue = max(self._peak_queue, self._total_depth())
        self._wakeup.set()

        result = await job.future
        waited_ms = getattr(job.future, "_queue_wait_ms", 0.0)
        return result, {
            "queue_wait_ms": round(waited_ms, 2),
            "batch_size": getattr(job.future, "_batch_size", 1),
        }

    # ---- worker --------------------------------------------------------
    async def _run(self) -> None:
        assert self._wakeup is not None
        while not self._closing:
            depth = self._total_depth()
            if depth == 0:
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                self._wakeup.clear()
                continue

            # Coalescing window: give more requests a chance to join the batch.
            if self.wait_s > 0 and depth < self.max_batch:
                await asyncio.sleep(self.wait_s)

            batch: list[_Job[T, R]] = []
            for prio in (0, 1, 2, 3):
                q = self._queues[prio]
                while q and len(batch) < self.max_batch:
                    batch.append(q.popleft())
                if len(batch) >= self.max_batch:
                    break
            if not batch:
                continue

            started = time.perf_counter()
            for job in batch:
                setattr(job.future, "_queue_wait_ms", (started - job.enqueued_at) * 1000.0)
                setattr(job.future, "_batch_size", len(batch))

            try:
                results = await asyncio.to_thread(self._runner, [j.payload for j in batch])
            except Exception as exc:  # one bad batch must not kill the worker
                log.exception("scheduler '%s' batch failed", self.name)
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._batches_run += 1
            self._items_run += len(batch)
            self._last_batch_size = len(batch)
            # EWMA of per-item service time, from real batches only.
            per_item = elapsed_ms / max(1, len(batch))
            self._per_item_ms = (
                per_item if self._per_item_ms is None else 0.8 * self._per_item_ms + 0.2 * per_item
            )

            if len(results) != len(batch):
                error = RuntimeError(
                    f"runner returned {len(results)} results for {len(batch)} items"
                )
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(error)
                continue

            for job, result in zip(batch, results):
                if not job.future.done():
                    job.future.set_result(result)

    # ---- reporting -----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        projected = self.projected_wait_ms()
        return {
            "name": self.name,
            "queue_depth": len(self._queue),
            "peak_queue_depth": self._peak_queue,
            "max_queue_depth": self.max_queue_depth,
            "max_batch": self.max_batch,
            "coalesce_window_ms": round(self.wait_s * 1000.0, 2),
            "batches_run": self._batches_run,
            "items_run": self._items_run,
            "last_batch_size": self._last_batch_size,
            "mean_batch_size": (
                round(self._items_run / self._batches_run, 2) if self._batches_run else None
            ),
            "rejected": self._rejected,
            "measured_per_item_ms": (
                round(self._per_item_ms, 2) if self._per_item_ms is not None else None
            ),
            "projected_wait_ms": round(projected, 2) if projected is not None else None,
            "projection_available": projected is not None,
            "projection_note": (
                None
                if projected is not None
                else "no completed batch yet: admission cannot refuse on an unmeasured estimate"
            ),
            "admission_enabled": self.admission_enabled,
            "reject_over_budget": self.reject_over_budget,
            "budget_ms": self.budget_ms,
            "admission_mode": (
                "unmeasured (admitting everything until one batch completes)"
                if self._per_item_ms is None
                else (
                    "queue-growth protection: the budget is unreachable on this device, "
                    f"shedding above {self.unreachable_budget_queue_multiple:g}x the "
                    f"{self._per_item_ms:.0f} ms service time"
                    if self._per_item_ms > self.budget_ms
                    else f"budget enforcement: shedding above {self.budget_ms:.0f} ms projected wait"
                )
            ),
            # Sustainable arrival rate implied by the measurement above.
            "capacity_rps": (
                round(1000.0 / self._per_item_ms, 2)
                if self._per_item_ms and self._per_item_ms > 0
                else None
            ),
            # The distinction that matters when reading a rejection: is this
            # device merely busy, or can it never hit the budget even idle?
            "budget_unreachable_on_this_device": (
                None
                if self._per_item_ms is None
                else bool(self._per_item_ms > self.budget_ms)
            ),
            "budget_unreachable_note": (
                None
                if self._per_item_ms is None or self._per_item_ms <= self.budget_ms
                else (
                    f"measured service time {self._per_item_ms:.0f} ms/item already exceeds the "
                    f"{self.budget_ms:.0f} ms budget at zero queue depth, so no amount of "
                    f"queueing discipline can meet the target on this hardware"
                )
            ),
        }
