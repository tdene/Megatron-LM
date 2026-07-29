# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Caller-owned orchestration of grouped rollout generation over an agent."""

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, AsyncIterator, NamedTuple

import numpy as np

from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

from ..inference import ReturnsRaw
from ..inflight_tracker import add_inflight, remove_inflight
from ..rollout_granularity import (
    GRANULARITY_RANK,
    ConsumptionGranularity,
    SubmissionGranularity,
)
from .api import EpisodeResult, GroupedRolloutRequest, GroupRolloutParams, RolloutGroup

if TYPE_CHECKING:
    from .api import GroupedRolloutGenerator


class _GranularityConfig(NamedTuple):
    submission: SubmissionGranularity
    consumption: ConsumptionGranularity
    num_groups_per_batch: int
    num_groups_per_env: tuple[int, ...]

    @classmethod
    def from_request(
        cls, request: GroupedRolloutRequest, num_groups_per_env: list[int]
    ) -> "_GranularityConfig":
        """Build the per-request granularity policy.

        Args:
            request: Grouped rollout request carrying the granularity choices.
            num_groups_per_env: Groups each env contributes to one batch, in env order.

        Returns:
            A validated _GranularityConfig.
        """
        cls._validate(request, num_groups_per_env)
        return cls(
            submission=request.submission_granularity,
            consumption=request.consumption_granularity,
            num_groups_per_batch=request.num_groups,
            num_groups_per_env=tuple(num_groups_per_env),
        )

    def env_of_index(self, index_in_batch: int) -> int:
        """Map a batch slot to the env owning it (slots are env-blocked, in env order).

        Args:
            index_in_batch: Slot index within one trainer batch.

        Returns:
            The env_index owning the slot.

        Raises:
            IndexError: If the slot lies outside the batch.
        """
        boundary = 0
        for env_index, groups in enumerate(self.num_groups_per_env):
            boundary += groups
            if index_in_batch < boundary:
                return env_index
        raise IndexError(
            f"index_in_batch {index_in_batch} outside batch of {self.num_groups_per_batch}"
        )

    def units_per_batch(self, rollouts_per_group: int) -> int:
        """Submission units in one batch; gate capacity = depth-in-batches x this.

        Args:
            rollouts_per_group: Rollouts per group, needed only for R granularity.

        Returns:
            The number of submission units one trainer batch contains.
        """
        return {
            "R": self.num_groups_per_batch * rollouts_per_group,
            "G": self.num_groups_per_batch,
            "E": len(self.num_groups_per_env),
            "B": 1,
        }[self.submission]

    @staticmethod
    def _validate(request: GroupedRolloutRequest, num_groups_per_env: list[int]) -> None:
        """Reject invalid granularity, layout, and filter combinations.

        Args:
            request: Grouped rollout request to check.
            num_groups_per_env: Constant per-env group layout.

        Raises:
            AssertionError: If consumption is finer than submission, the layout
                starves an env or does not sum to num_groups, or reward
                filtering is requested.
        """
        assert (
            GRANULARITY_RANK[request.consumption_granularity]
            >= GRANULARITY_RANK[request.submission_granularity]
        ), (
            f"Consumption granularity ({request.consumption_granularity}) must be no finer "
            f"than submission granularity ({request.submission_granularity})."
        )
        assert all(
            groups > 0 for groups in num_groups_per_env
        ), "Each environment must request at least one group per batch."
        assert (
            sum(num_groups_per_env) == request.num_groups
        ), "The sum of groups per environment must equal the total number of groups requested."


class _SubmissionGate:
    """Gate capacity is measured in units of the configured submission granularity.

    Each granularity has a single release point: R slots free when inference
    completes, so the gate bounds engine concurrency in rollouts. G, E, and B
    slots free when the trainer consumes the group/env-unit/batch, so the gate
    enforces the --rl-generation-lag run-ahead cap in groups, env-units, and
    batches respectively. A group dropped by the reward filter never reaches
    consumption; its slot (and its env-unit's/batch's under E/B submission)
    transfers to the regenerated replacement (see stage_filter) rather than
    being released.
    """

    def __init__(
        self,
        *,
        capacity: int,
        submission: SubmissionGranularity,
    ) -> None:
        """Create a gate with `capacity` slots counted at `submission` granularity.

        Args:
            capacity: Maximum submission units in flight.
            submission: Configured submission granularity; only matching
                acquire_for/release_for calls touch the semaphore.
        """
        self._sem = asyncio.Semaphore(capacity)
        self._submission = submission
        self.capacity = capacity
        # Observability counters, updated only on the configured submission
        # granularity (the only path that touches the semaphore). `held`
        # counts slots currently held; `prepare_blocked_seconds` accumulates
        # time stage_prepare spent waiting on the semaphore.
        self.held = 0
        self.prepare_blocked_seconds = 0.0
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire_for(self, granularity: SubmissionGranularity) -> None:
        """Take one slot when crossing a boundary of the configured granularity.

        Args:
            granularity: The dispatch boundary being crossed; no-op unless it
                matches the gate's configured submission granularity.
        """
        if self._submission == granularity:
            start = time.monotonic()
            await self._sem.acquire()
            self.prepare_blocked_seconds += time.monotonic() - start
            self.held += 1
            self.acquire_calls += 1

    def release_for(self, granularity: SubmissionGranularity) -> None:
        """Release one slot when work at the given granularity reaches its release point.

        Args:
            granularity: The granularity whose release point was just reached;
                no-op unless it matches the gate's configured submission
                granularity.
        """
        if self._submission == granularity:
            self._sem.release()
            self.held -= 1
            self.release_calls += 1


class _InferWorkItem(NamedTuple):
    """One rollout's worth of work flowing from prepare to infer.

    Timestamps are wall-clock monotonic seconds: `prepared_at` is stamped at
    construction and `infer_dequeued_at` is filled in via `_replace` when an
    infer worker dequeues the item. Zero means "not yet reached".
    """

    group_id: int
    rollout_idx: int
    batch_id: int
    index_in_batch: int
    params: GroupRolloutParams
    env_index: int = 0
    prepared_at: float = 0.0
    infer_dequeued_at: float = 0.0


class _InferredItem(NamedTuple):
    """One rollout post-inference, flowing from infer to assemble."""

    item: _InferWorkItem
    episode: EpisodeResult
    inferred_at: float = 0.0


class _AssembledGroup(NamedTuple):
    """One complete group flowing from assemble to filter."""

    group: RolloutGroup
    assembled_at: float = 0.0


class RolloutPipeline:
    """Orchestrates grouped rollout generation over an agent, one instance per request.

    Constructed and driven by the caller (e.g. the trainer via run()); the agent
    only supplies the env layout, per-group preparation, and inference calls.
    """

    def __init__(
        self,
        agent: "GroupedRolloutGenerator",
        request: GroupedRolloutRequest,
        parallel_generation_tasks: float,
    ) -> None:
        """Validate the request and size the gate, queues, and worker pool.

        Args:
            agent: Agent supplying the env layout, preparation, and inference.
            request: Grouped rollout request to serve; one pipeline per request.
            parallel_generation_tasks: Submission gate depth in trainer
                batches; units_per_batch scales it to submission units. May be
                fractional (e.g. an autotuned lag): the slot count is rounded
                and clamped to at least one unit.
        """
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."
        self.agent = agent
        self.request = request
        self.gran_policy = _GranularityConfig.from_request(
            request, agent.rollout_group_layout(request.num_groups)
        )
        self.gate = _SubmissionGate(
            capacity=max(
                1,
                round(
                    parallel_generation_tasks
                    * self.gran_policy.units_per_batch(request.rollouts_per_group)
                ),
            ),
            submission=self.gran_policy.submission,
        )
        self.num_infer_workers = max(
            1,
            round(
                parallel_generation_tasks
                * self.gran_policy.num_groups_per_batch
                * request.rollouts_per_group
            ),
        )

        # Core queues.
        self.infer_queue = asyncio_Queue()
        self.assemble_queue = asyncio_Queue()
        self.filter_queue = asyncio_Queue()
        self.output_queue = asyncio_Queue()
        self.bank = asyncio_Queue()
        self.banked_batches = 0
        self.consumed_batches = 0
        # Regenerated groups draw ids from a negative namespace to avoid collisions.
        self._next_regen_group_id = -1
        # Groups submitted but not yet delivered by stage_filter.
        self._groups_in_flight = 0
        self._prepare_done = False

        # Buffers of partial results.
        self._assemble_pending: dict[int, list[_InferredItem]] = {}
        self._consume_pending: dict[int, list[RolloutGroup]] = {}
        self._output_enqueued_at: dict[tuple[int, int], float] = {}

        # Observability accumulators.
        self.infer_queue_dwell: list[float] = []
        self.engine_dwell: list[float] = []
        self.assemble_queue_dwell: list[float] = []
        self.filter_queue_dwell: list[float] = []
        self.output_queue_dwell: list[float] = []
        self.prepared_count = 0
        self.inferred_count = 0
        self.assembled_count = 0
        self.filtered_count = 0
        self.yielded_count = 0
        self.prepared_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)
        self.assembled_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)
        self.yielded_groups_per_env = [0] * len(self.gran_policy.num_groups_per_env)

    async def run(self) -> AsyncIterator[RolloutGroup]:
        """Run the pipeline stages; cancels them when the iterator is closed.

        Yields:
            RolloutGroup: Groups in consumption-granularity order.
        """
        tasks = (
            asyncio.create_task(self.stage_prepare()),
            asyncio.create_task(self.stage_infer()),
            asyncio.create_task(self.stage_assemble()),
            asyncio.create_task(self.stage_filter()),
            asyncio.create_task(self.stage_bank()),
        )
        try:
            async for group in self.stage_consume():
                yield group
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _submit_group(
        self, *, group_id: int, batch_id: int, index_in_batch: int, env_index: int
    ) -> None:
        """Enqueue one group's inference items, acquiring per-rollout gate slots.

        The group's coarse (G/E/B) submission slots are the caller's concern:
        stage_prepare acquires fresh ones per boundary, while stage_filter
        regeneration reuses the dropped group's still-held slots.
        """
        params: GroupRolloutParams = await self.agent.prepare_group_rollout(
            self.request, env_index=env_index
        )
        self.prepared_groups_per_env[env_index] += 1
        # Mark this group's rollouts in flight for the duration of generation
        # (regenerated replacements re-enter flight exactly like fresh groups).
        add_inflight(self.request.rollouts_per_group)

        for rollout_idx in range(self.request.rollouts_per_group):
            await self.gate.acquire_for("R")
            item = _InferWorkItem(
                group_id=group_id,
                rollout_idx=rollout_idx,
                batch_id=batch_id,
                index_in_batch=index_in_batch,
                params=params,
                env_index=env_index,
                prepared_at=time.monotonic(),
            )
            await self.infer_queue.put(item)
            self.prepared_count += 1

    def _maybe_close_intake(self) -> None:
        """Shut down infer_queue once no work can ever be submitted again."""
        if self._prepare_done and self._groups_in_flight <= 0:
            self.infer_queue.shutdown()

    async def stage_prepare(self) -> None:
        """Generate gated inference work items."""
        group_id = 0
        batch_id = 0
        try:
            while True:
                await self.gate.acquire_for("B")

                index_in_batch = 0
                for env_index, env_groups in enumerate(self.gran_policy.num_groups_per_env):
                    # Env-unit boundary: under E submission, hold one gate
                    # slot per env-unit until the trainer consumes it.
                    await self.gate.acquire_for("E")
                    for _ in range(env_groups):
                        await self.gate.acquire_for("G")
                        self._groups_in_flight += 1
                        await self._submit_group(
                            group_id=group_id,
                            batch_id=batch_id,
                            index_in_batch=index_in_batch,
                            env_index=env_index,
                        )
                        group_id += 1
                        index_in_batch += 1
                batch_id += 1
        except BaseException:
            self.infer_queue.shutdown()
            raise
        finally:
            self._prepare_done = True
            self._maybe_close_intake()

    async def stage_infer(self) -> None:
        """Run a persistent pool of inference workers, spawned once per pipeline."""
        workers = [
            asyncio.create_task(self._infer_worker()) for _ in range(self.num_infer_workers)
        ]
        try:
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            for worker in workers:
                worker.cancel()
            self.assemble_queue.shutdown()

    async def _infer_worker(self) -> None:
        while True:
            try:
                item = await self.infer_queue.get()
            except asyncio_QueueShutDown:
                return
            item = item._replace(infer_dequeued_at=time.monotonic())
            if item.prepared_at:
                self.infer_queue_dwell.append(item.infer_dequeued_at - item.prepared_at)
            await self._infer_one(item)

    @trace_async_exceptions(verbose=True)
    async def _infer_one(self, item: _InferWorkItem) -> None:
        """Run one episode for one work item and hand the result to assemble.

        Args:
            item: The dequeued work item; its params carry the episode closure.
        """
        episode = await item.params.run_episode()
        inferred_at = time.monotonic()
        self.gate.release_for("R")
        if item.infer_dequeued_at:
            self.engine_dwell.append(inferred_at - item.infer_dequeued_at)
        self.inferred_count += 1
        await self.assemble_queue.put(
            _InferredItem(item=item, episode=episode, inferred_at=inferred_at)
        )

    async def stage_assemble(self) -> None:
        """Build complete rollout groups from inferred items."""
        pending = self._assemble_pending
        try:
            while True:
                try:
                    inferred = await self.assemble_queue.get()
                except asyncio_QueueShutDown:
                    break
                dequeued_at = time.monotonic()
                if inferred.inferred_at:
                    self.assemble_queue_dwell.append(dequeued_at - inferred.inferred_at)
                bucket = pending.setdefault(inferred.item.group_id, [])
                bucket.append(inferred)
                if len(bucket) < self.request.rollouts_per_group:
                    continue
                completed = pending.pop(inferred.item.group_id)
                completed.sort(key=lambda item: item.item.rollout_idx)
                rollouts = await asyncio.gather(
                    *[item.item.params.build_rollout(item.episode) for item in completed]
                )
                first = completed[0]
                self.assembled_count += 1
                self.assembled_groups_per_env[first.item.env_index] += 1
                await self.filter_queue.put(
                    _AssembledGroup(
                        group=RolloutGroup(
                            rollouts=rollouts,
                            batch_id=first.item.batch_id,
                            index_in_batch=first.item.index_in_batch,
                        ),
                        assembled_at=time.monotonic(),
                    )
                )
        finally:
            self.filter_queue.shutdown()

    @trace_async_exceptions(verbose=True)
    async def stage_filter(self) -> None:
        """Deliver assembled groups, regenerating any dropped by the reward filter."""
        try:
            while True:
                try:
                    assembled = await self.filter_queue.get()
                except asyncio_QueueShutDown:
                    break
                dequeued_at = time.monotonic()
                if assembled.assembled_at:
                    self.filter_queue_dwell.append(dequeued_at - assembled.assembled_at)
                group = assembled.group
                if self._should_drop(group):
                    self.filtered_count += 1
                    # Balance add_inflight: a dropped group is never consumed.
                    remove_inflight(self.request.rollouts_per_group)
                    # G/E/B gate slots free on consumption, which a dropped group
                    # never reaches: like its in-flight count, its slot carries
                    # over to the replacement (no release here, no fresh coarse
                    # acquire in _submit_group) and frees when the replacement
                    # is ultimately consumed. Releasing here and re-acquiring
                    # instead could deadlock: with the gate fully held, the
                    # freed slot can be won by stage_prepare's FIFO-earlier
                    # waiter, parking regeneration forever while batch-order
                    # consumption waits on the very replacement it must yield.
                    try:
                        await self._regenerate_group(group)
                    except asyncio_QueueShutDown:
                        # Intake closed mid-regeneration (teardown or prepare
                        # failure): no replacement can be submitted, so return
                        # the inherited slot and retire the group. (Under E/B
                        # submission the coarse slot is shared with the rest of
                        # its unit/batch, which can no longer complete either;
                        # the gate dies with the pipeline at teardown.)
                        self.gate.release_for("G")
                        self._groups_in_flight -= 1
                        self._maybe_close_intake()
                    continue
                self._output_enqueued_at[(group.batch_id, group.index_in_batch)] = (
                    time.monotonic()
                )
                await self.output_queue.put(group)
                self._groups_in_flight -= 1
                self._maybe_close_intake()
        finally:
            self.output_queue.shutdown()

    def _should_drop(self, group: RolloutGroup) -> bool:
        """A group with zero reward variance carries no learning signal."""
        if not self.request.filter_groups_with_same_reward:
            return False
        return np.std([rollout.reward for rollout in group.rollouts]) <= 1e-6

    async def _regenerate_group(self, dropped: RolloutGroup) -> None:
        """Resubmit a replacement group for a dropped group's batch slot.

        The replacement inherits the dropped group's submission-gate slot and
        in-flight count, so no coarse slot is acquired here. _submit_group still
        acquires "R" slots: the replacement's rollouts are new engine work, and
        R slots free on inference completion, never on consumption, so waiting
        for one cannot deadlock against the consumer.
        """
        group_id = self._next_regen_group_id
        self._next_regen_group_id -= 1
        await self._submit_group(
            group_id=group_id,
            batch_id=dropped.batch_id,
            index_in_batch=dropped.index_in_batch,
            env_index=self.gran_policy.env_of_index(dropped.index_in_batch),
        )

    def _record_output_dwell(self, group: RolloutGroup) -> None:
        """Record how long a group waited between assembly and being yielded.

        Args:
            group: The group being yielded to the consumer.
        """
        key = (group.batch_id, group.index_in_batch)
        enqueued_at = self._output_enqueued_at.pop(key, 0.0)
        if enqueued_at:
            self.output_queue_dwell.append(time.monotonic() - enqueued_at)
        self.yielded_count += 1
        self.yielded_groups_per_env[self.gran_policy.env_of_index(group.index_in_batch)] += 1

    async def _next_group(self) -> RolloutGroup | None:
        """Pop the next group off output_queue.

        Returns:
            The next RolloutGroup, or None once the queue shuts down.
        """
        try:
            return await self.output_queue.get()
        except asyncio_QueueShutDown:
            return None

    @property
    def ready_batches(self) -> int:
        """Full batches banked and not yet dequeued for consumption."""
        return self.banked_batches - self.consumed_batches

    @trace_async_exceptions(verbose=True)
    async def stage_bank(self) -> None:
        """Bank complete batches cut from the consumption-ordered group stream."""
        order = {
            "G": self._consume_completion_order,
            "E": self._consume_env_units,
            "B": self._consume_batch_order,
        }[self.gran_policy.consumption]
        batch: list[RolloutGroup] = []
        try:
            async for group in order():
                batch.append(group)
                if len(batch) == self.gran_policy.num_groups_per_batch:
                    self.bank.put_nowait(batch)
                    self.banked_batches += 1
                    batch = []
            # Per-batch env-weight rounding keeps every batch identically sized, so a
            # naturally-ended stream can never leave a partial batch behind.
            assert not (batch or self._consume_pending), (
                "Stream ended with groups not forming a full batch."
            )
        finally:
            self.bank.shutdown()

    async def stage_consume(self) -> AsyncIterator[RolloutGroup]:
        """Unwrap banked batches for the consumer, freeing gate slots as it goes.

        Yields:
            RolloutGroup: Groups ordered by the configured consumption mode.
        """
        while True:
            try:
                batch = await self.bank.get()
            except asyncio_QueueShutDown:
                return
            self.consumed_batches += 1
            consumed_per_env = [0] * len(self.gran_policy.num_groups_per_env)
            for group in batch:
                self._record_output_dwell(group)
                yield group
                # G/E/B slots free on trainer consumption: the release for a
                # group fires when the consumer comes back for the next one.
                # Releasing in the ordering generators instead would fire at
                # banking time, uncapping generation run-ahead.
                self.gate.release_for("G")
                env_index = self.gran_policy.env_of_index(group.index_in_batch)
                consumed_per_env[env_index] += 1
                # E and B consumption bank each env's unit contiguously, so the
                # unit's slot frees with its last group. (Under G consumption
                # envs may interleave, but then submission is R or G and the E
                # release is a no-op.)
                if consumed_per_env[env_index] == self.gran_policy.num_groups_per_env[env_index]:
                    self.gate.release_for("E")
            self.gate.release_for("B")

    async def _consume_completion_order(self) -> AsyncIterator[RolloutGroup]:
        """G consumption: deliver each group as soon as it assembles.

        Yields:
            RolloutGroup: Groups in global completion order.
        """
        while (group := await self._next_group()) is not None:
            yield group

    async def _consume_env_units(self) -> AsyncIterator[RolloutGroup]:
        """Balanced-E consumption.

        Within each env, deliver groups in completion order, cut into env-units
        of num_groups_per_env[e] — each unit is the env's earliest unclaimed groups,
        which may span dispatch batches. One unit per env per delivered batch;
        a fast env's extra units wait until every env has served the current
        batch, so no env runs more than one delivered batch ahead.

        Yields:
            RolloutGroup: Groups in balanced env-unit order.
        """
        num_envs = len(self.gran_policy.num_groups_per_env)
        pending: list[list[RolloutGroup]] = [[] for _ in range(num_envs)]
        ready_units: list[deque[list[RolloutGroup]]] = [deque() for _ in range(num_envs)]
        delivered_units = [0] * num_envs
        current_batch = 0
        while (group := await self._next_group()) is not None:
            env_index = self.gran_policy.env_of_index(group.index_in_batch)
            pending[env_index].append(group)
            unit_size = self.gran_policy.num_groups_per_env[env_index]
            if len(pending[env_index]) >= unit_size:
                ready_units[env_index].append(pending[env_index][:unit_size])
                pending[env_index] = pending[env_index][unit_size:]
            progressed = True
            while progressed:
                progressed = False
                for env in range(num_envs):
                    if delivered_units[env] == current_batch and ready_units[env]:
                        for unit_group in ready_units[env].popleft():
                            yield unit_group
                        delivered_units[env] += 1
                        progressed = True
                if all(count > current_batch for count in delivered_units):
                    current_batch += 1
                    progressed = True

    async def _consume_batch_order(self) -> AsyncIterator[RolloutGroup]:
        """B consumption: deliver whole batches in dataset order.

        Yields:
            RolloutGroup: Batch b's groups sorted by index_in_batch, batches in order.
        """
        next_batch_id = 0
        pending = self._consume_pending
        while (group := await self._next_group()) is not None:
            pending.setdefault(group.batch_id, []).append(group)
            while (
                len(pending.get(next_batch_id, []))
                >= self.gran_policy.num_groups_per_batch
            ):
                batch = pending.pop(next_batch_id)
                # Env blocks are contiguous in index_in_batch order, so the
                # sorted batch is env 0's unit, then env 1's, and so on.
                batch.sort(key=lambda group: group.index_in_batch)
                next_batch_id += 1
                for group in batch:
                    yield group
