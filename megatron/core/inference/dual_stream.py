# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dual-stream inference coordination.

Overlaps the critical inference path (forward + sampling) with bookkeeping
(logprobs, stop detection, finishing/eviction) via two CUDA streams.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import heapq
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch

#: Default capacity of the reservation pool (number of pre-allocated KV
#: block IDs). The pool must hold enough blocks for the worst-case
#: single-iteration demand, which is one block per active request (when
#: every request crosses a block boundary simultaneously). 256 matches
#: the typical ``max_requests`` ceiling in production configs.
#: Deployments with larger batch sizes should raise this via the
#: ``reservation_pool_capacity`` parameter on ``DualStreamCoordinator``.
DEFAULT_RESERVATION_POOL_CAPACITY = 256


@dataclass
class SideDecision:
    """Emitted by side stream at each bookkeeping pass.

    ``finished_request_ids`` are the requests whose sample triggered EOS
    or exhausted their output length; main stream removes them from the
    active set at the next ``_ds_enqueue_one_iter``.

    ``resume_request_ids`` are previously-paused requests that the side
    stream determined can be unpaused (because the allocator has room
    for their pre-allocated block). Main stream moves them back into the
    active region at the next ``_ds_enqueue_one_iter``.

    ``evict_request_ids`` are paused requests that the side stream
    selected for overflow eviction — their KV blocks are quarantined,
    they are removed from the paused region, and they get pushed back
    through ``post_process_requests`` as evicted so the engine can
    re-admit them via the waiting queue. This is the escape valve for
    workloads that accumulate more pauses than resumes can keep up
    with.
    """

    finished_request_ids: List[int] = field(default_factory=list)
    resume_request_ids: List[int] = field(default_factory=list)
    evict_request_ids: List[int] = field(default_factory=list)


@dataclass
class QuarantineEntry:
    """KV blocks awaiting safe reclaim once main stream advances past their generation."""

    block_ids: List[int]
    quarantine_gen: int
    _seq: int = field(default=0, repr=False)

    def __lt__(self, other: QuarantineEntry) -> bool:
        if self.quarantine_gen != other.quarantine_gen:
            return self.quarantine_gen < other.quarantine_gen
        return self._seq < other._seq


class ReservationPool:
    """CPU-side free-list of pre-allocated KV block IDs.

    Main stream pops, side stream pushes (via reclaim), and
    ``_rewind_kv_cache`` pushes from the main stream. Thread-safe
    via a simple deque + lock.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._pool: Deque[int] = deque()
        self._lock = threading.Lock()

    def pop(self) -> Optional[int]:
        """Pop a block ID for main-stream pre-allocation. Returns None if empty."""
        with self._lock:
            if self._pool:
                return self._pool.popleft()
            return None

    def push(self, block_id: int) -> None:
        """Push a reclaimed block ID back into the pool."""
        with self._lock:
            self._pool.append(block_id)

    def push_many(self, block_ids: List[int]) -> None:
        """Bulk-push reclaimed blocks."""
        with self._lock:
            self._pool.extend(block_ids)

    @property
    def depth(self) -> int:
        """Return the number of blocks currently in the pool."""
        with self._lock:
            return len(self._pool)

    def __len__(self) -> int:
        """Return the number of blocks currently in the pool."""
        return self.depth


class BlockQuarantine:
    """Epoch-based quarantine for KV blocks awaiting safe reclaim.

    Blocks are tagged with the main-stream generation at which they stopped
    being referenced. They are released in bulk once main_gen advances past
    the tag.
    """

    def __init__(self):
        self._entries: List[QuarantineEntry] = []  # min-heap by quarantine_gen
        self._seq_counter: int = 0

    def add(self, block_ids: List[int], quarantine_gen: int) -> None:
        """Tag ``block_ids`` with ``quarantine_gen`` for deferred reclaim."""
        if block_ids:
            self._seq_counter += 1
            entry = QuarantineEntry(block_ids, quarantine_gen, _seq=self._seq_counter)
            heapq.heappush(self._entries, entry)

    def drain(self, main_gen_current: int) -> List[int]:
        """Drain all entries whose quarantine_gen < main_gen_current.

        Returns a flat list of block IDs safe to reclaim.

        Safety: ``main_gen_current`` tracks CPU-side enqueue order on the main
        stream, not GPU execution order. Reuse of a drained block at iteration
        Y (> quarantine_gen X) is still ordered correctly on the device because
        both the freeing write at iter X and the reuse write at iter Y are
        issued on the same CUDA stream, so stream ordering guarantees X's
        kernels complete before Y's kernels execute. The side stream never
        reads raw KV cache bytes (only the cloned logits/sample tensors in the
        snapshot), so draining by CPU-side main_gen is safe. If a future change
        makes the side stream touch KV buffers directly, or moves reuse to a
        different CUDA stream, this safety argument no longer holds and drain
        must be gated on an actual GPU event.
        """
        released = []
        while self._entries and self._entries[0].quarantine_gen < main_gen_current:
            entry = heapq.heappop(self._entries)
            released.extend(entry.block_ids)
        return released

    def __len__(self) -> int:
        return len(self._entries)


#: Number of iterations the critical loop primes before the first
#: wake-event handoff. Priming seeds the snapshot ring so the side
#: stream has work to process during the first bookkeeping cycle.
#: The value 2 is the minimum that keeps the side stream occupied
#: for the full duration of the first main-stream iteration: one
#: snapshot is being processed while the second buffers the next.
#: Higher values would cost extra snapshot memory without buying
#: real depth, because the CPU sync inside ``chain_update`` (the
#: ``.tolist()`` of the pre-allocation crossing indices, even
#: hoisted via the helper thread) bottlenecks main-stream dispatch
#: to GPU speed regardless.
PIPELINE_PRIME_ITERATIONS = 2


class DualStreamCoordinator:
    """Coordinates the critical and bookkeeping asyncio tasks.

    Manages:
    - Two CUDA streams (main for forward/sample, side for bookkeeping/logprobs).
    - STOP + PROCEED two-signal asyncio protocol.
    - CUDA-event-to-asyncio bridge via helper thread.
    - Reservation pool, block quarantine, pause queue.
    - Generation counters for epoch-based reclaim.

    This class is created and owned by DynamicInferenceEngine. It holds
    coordination state but does NOT hold a reference to the model, context,
    or controller — those are passed in by the engine at call sites.

    Args:
        reservation_pool_capacity: Size of the pre-allocated block pool.
    """

    def __init__(self, reservation_pool_capacity: int = DEFAULT_RESERVATION_POOL_CAPACITY):
        # CUDA streams.
        device = torch.cuda.current_device()
        self.main_stream = torch.cuda.Stream(device=device)
        self.side_stream = torch.cuda.Stream(device=device)

        # Two-signal asyncio coordination.
        # STOP: plain bool, set by helper thread when the wake CUDA event fires.
        # Gates ONLY picking up a new snapshot — never preempts a pass already
        # in progress, to keep snapshot processing strictly in-order.
        self._stop_bookkeeping: bool = False
        # PROCEED: asyncio gate — cleared while critical is active.
        self._bookkeeping_proceed: Optional[asyncio.Event] = None  # created in start()

        # CUDA wake event + helper thread.
        self._wake_event = torch.cuda.Event()
        self.wake_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cuda-wake"
        )

        # Cross-stream ordering events.
        #
        # ``main_sample_done_event`` is recorded on the main stream right
        # after sampling and is waited on by the side stream before it
        # reads ``self._all_logits_cuda`` / ``self._sampled_tokens_cuda``.
        # This guarantees the side stream never reads into uninitialized
        # logits / sample buffers when main stream's forward or sample
        # kernels are still executing on the device.
        #
        # ``side_reads_done_event`` is recorded on the side stream after
        # side stream has finished reading all main-stream-owned context
        # state for a given iteration (active_* buffers, sampled tokens,
        # logits). Main stream waits on it at the top of the next
        # iteration's ``_ds_enqueue_one_iter`` before running
        # ``_dynamic_step_context_init``, which overwrites the active_*
        # mirrors. Without this wait, the next init could race side
        # stream's in-flight reads on the GPU.
        #
        # Both events are unrecorded at startup. ``wait_event`` on an
        # unrecorded ``torch.cuda.Event`` is a no-op (cudaStreamWaitEvent
        # completes immediately), so the first iteration's waits degenerate
        # cleanly without any special-casing.
        self.main_sample_done_event = torch.cuda.Event()
        self.side_reads_done_event = torch.cuda.Event()

        # Generation counters.
        self.main_gen: int = 0
        self.side_gen: int = 0

        # Reservation pool.
        self.reservation_pool = ReservationPool(reservation_pool_capacity)

        # Block quarantine for epoch-based reclaim.
        self.quarantine = BlockQuarantine()

        # Pause queue: main → side. Request IDs paused because the
        # reservation pool ran dry while they needed a new KV block.
        # Paused requests retain their KV cache and can be resumed
        # later (via a ``resume_request_ids`` decision) without
        # re-prefilling.
        self._pauses_queue: Deque[int] = deque()
        self._pauses_lock = threading.Lock()

        # Decisions queue: side → main.
        self._decisions_queue: Deque[SideDecision] = deque()
        self._decisions_lock = threading.Lock()

        # Sample snapshot ring buffer (main publishes, side consumes).
        # Each entry is a dict of cloned tensors for the side stream to process.
        # Capped at PIPELINE_PRIME_ITERATIONS + 2: priming seeds up to
        # PIPELINE_PRIME_ITERATIONS entries, and in steady state at most one
        # additional snapshot is published before bookkeeping drains one.
        # publish_snapshot asserts if the cap is exceeded, catching bugs
        # where main out-runs bookkeeping (each snapshot clones GPU tensors,
        # so unbounded growth would also be a memory leak).
        self._snapshot_ring_max = PIPELINE_PRIME_ITERATIONS + 2
        self._snapshot_ring: Deque[Dict] = deque()
        self._snapshot_lock = threading.Lock()

        # Running flag.
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Initialize asyncio-dependent state. Must be called from the event loop."""
        self._bookkeeping_proceed = asyncio.Event()
        self._bookkeeping_proceed.set()
        self._running = True

    def stop(self):
        """Signal both loops to exit and tear down the helper thread pool."""
        self._running = False
        # Unblock bookkeeping if it's waiting.
        if self._bookkeeping_proceed is not None:
            self._bookkeeping_proceed.set()
        # Shut down the wake helper thread without waiting; if it's mid-
        # synchronize() the call will return as soon as the GPU drains.
        self.wake_pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # CUDA event → asyncio bridge
    # ------------------------------------------------------------------

    def _thread_wait_and_signal(self):
        """Runs in the helper thread. Blocks on CUDA event, then raises STOP.

        The helper thread exists so the asyncio event loop can ``await`` the
        wake notification without blocking the whole loop on a CUDA sync.
        A plain ``cudaStreamWaitEvent`` would only serialize the side stream
        behind the event; it wouldn't notify Python.
        """
        self._wake_event.synchronize()
        self._stop_bookkeeping = True

    async def _await_wake(self):
        """Await the CUDA wake event, bridging to asyncio via thread pool."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.wake_pool, self._thread_wait_and_signal)

    # ------------------------------------------------------------------
    # Pause queue (main → side)
    # ------------------------------------------------------------------

    def emit_pause(self, request_id: int) -> None:
        """Called by main stream's chain_update when block-pressure pause fires."""
        with self._pauses_lock:
            self._pauses_queue.append(request_id)

    def drain_pauses(self) -> List[int]:
        """Called by side stream to consume pending pauses. Returns request IDs."""
        with self._pauses_lock:
            pauses = list(self._pauses_queue)
            self._pauses_queue.clear()
        return pauses

    # ------------------------------------------------------------------
    # Decisions queue (side → main)
    # ------------------------------------------------------------------

    def publish_decisions(self, decisions: SideDecision) -> None:
        """Called by side stream to publish bookkeeping decisions for main."""
        with self._decisions_lock:
            self._decisions_queue.append(decisions)

    def drain_decisions(self) -> List[SideDecision]:
        """Called by main stream at handoff to consume side's decisions."""
        with self._decisions_lock:
            decs = list(self._decisions_queue)
            self._decisions_queue.clear()
        return decs

    # ------------------------------------------------------------------
    # Snapshot ring (main → side)
    # ------------------------------------------------------------------

    def publish_snapshot(self, snapshot: Dict) -> None:
        """Called by main stream after sampling to publish data for side to process."""
        with self._snapshot_lock:
            assert len(self._snapshot_ring) < self._snapshot_ring_max, (
                f"Snapshot ring overflow: {len(self._snapshot_ring)} entries "
                f"(max {self._snapshot_ring_max}). Bookkeeping is not draining fast enough."
            )
            self._snapshot_ring.append(snapshot)

    def consume_snapshot(self) -> Optional[Dict]:
        """Called by side stream to get the next snapshot to process."""
        with self._snapshot_lock:
            if self._snapshot_ring:
                return self._snapshot_ring.popleft()
            return None

    # ------------------------------------------------------------------
    # Batched block reclaim
    # ------------------------------------------------------------------

    def reclaim_blocks(self) -> int:
        """Drain quarantine and push freed blocks to reservation pool.

        Returns the number of blocks reclaimed.
        """
        released = self.quarantine.drain(self.main_gen)
        if released:
            self.reservation_pool.push_many(released)
        return len(released)
