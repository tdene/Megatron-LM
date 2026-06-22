# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import numpy as np
from pydantic import BaseModel

from megatron.core.inference.utils import asyncio_Queue
from megatron.core.utils import trace_async_exceptions

from ..__init__ import Request, TypeLookupable
from ..inference import (
    InferenceInterface,
    LLMChatMessage,
    ReturnsRaw,
)


class AgentBaseModel(BaseModel, extra='allow'):
    pass


class RolloutRequest(Request):
    """Request to agent to generate Rollouts."""

    num_rollouts: int
    inference_interface: InferenceInterface
    validation: bool = False


class GroupedRolloutRequest(Request):
    """Request to agent to generate grouped Rollouts."""

    groups_per_batch: int
    rollouts_per_group: int
    inference_interface: InferenceInterface
    validation: bool = False
    filter_groups_with_same_reward: bool = False
    streaming: bool = False
    oversubscription_factor: int = 1
    submission_granularity: Literal["R", "G", "B"] = "B"
    consumption_granularity: Literal["R", "G", "B"] = "B"


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: list[str]
    prompt_length: list[int] | None = None
    reward: float = None
    env_id: str = ''
    problem_id: str | None = None
    policy_epoch: list[list[tuple[int, int]]]
    kv_cache_epoch: list[list[tuple[int, int]]]
    num_evictions: list[int]


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[list[int]]
    reward: list[float] | float
    generation_mask: list[list[bool]] | None = None
    logprobs: list[list[float]] | None = None
    env_id: str = ''
    problem_id: str | None = None
    policy_epoch: list[list[tuple[int, int]]]
    kv_cache_epoch: list[list[tuple[int, int]]]
    num_evictions: list[int]


Rollouts = list[TokenRollout | Rollout]


class RolloutGroup(AgentBaseModel):
    """A group of rollouts (e.g. multiple completions for one prompt) with batch metadata."""

    rollouts: Rollouts
    batch_id: int = 0
    index_in_batch: int = 0

    def __iter__(self):
        return iter(self.rollouts)

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]


GroupedRollouts = list[RolloutGroup]


class ContrastiveRollout(AgentBaseModel):
    """Contrastive/Preference data for language-based Rollout."""

    chosen_trajectory: list[str]
    rejected_trajectory: list[str]


class Head2HeadRolloutRequest(Request):
    num_rollouts: int
    inference_interface: list[InferenceInterface]
    validation: bool = False


class EvaluationRequest(Request):
    """Request to evaluate N prompts, optionally distributed across ranks."""

    inference_interface: InferenceInterface
    num_prompts: int
    rank_info: tuple[int, int] | None = (
        None  # (rank, total_ranks) if distributed, None for full evaluation
    )
    validation: bool = True


class EvaluationResult(AgentBaseModel):
    prompt: str | list[LLMChatMessage]
    response: str | LLMChatMessage


class RewardEvaluationResult(EvaluationResult):
    reward: float
    problem_id: str | None = None


T = TypeVar('T', bound=EvaluationResult)


class EvaluationResponse(AgentBaseModel, TypeLookupable, Generic[T]):
    env_id: str
    results: list[T]

    def metrics(self):
        raise NotImplementedError(f"{type(self)} did not provide metric aggregation.")


class Agent(ABC, AgentBaseModel):
    pass


class RolloutGenerator(Agent, ABC):
    """An agent that produces Rollout objects containing rollout string and associated reward."""

    @abstractmethod
    async def rollout(self, request: RolloutRequest) -> Rollout: ...

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        return await asyncio.gather(
            *[self.rollout(request=request) for _ in range(request.num_rollouts)]
        )


class ContrastiveRolloutGenerator(Agent, ABC):
    """An agent that produces ContrastiveRollout objects containing two rollout strings, one chosen and one rejected."""

    @abstractmethod
    async def get_contrastive_rollouts(
        self, request: RolloutRequest
    ) -> list[ContrastiveRollout]: ...


class TokenizedRolloutGenerator(Agent, ABC):
    """An agent that produces TokenRollout objects containing rollout token ids and associated rewards.

    Optionally can also provide generation masks to indicate which tokens were generated and token masks to indicate which
    tokens were possible at any given step.
    """

    @abstractmethod
    async def rollout(self, request: RolloutRequest) -> TokenRollout: ...

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        return await asyncio.gather(
            *[self.rollout(request=request) for _ in range(request.num_rollouts)]
        )


class GroupedRolloutGenerator(Agent, ABC):
    """An interface to return grouped Rollout objects to support algorithms like GRPO."""

    buffer_size: int = 10

    async def prepare_group(self, request: GroupedRolloutRequest):
        raise NotImplementedError

    async def generate_rollout(self, request: GroupedRolloutRequest, prepared) -> Rollout:
        raise NotImplementedError

    async def get_grouped_rollouts(self, request: GroupedRolloutRequest):
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        groups_per_batch = request.groups_per_batch
        rollouts_per_group = request.rollouts_per_group

        num_workers = request.oversubscription_factor * groups_per_batch * rollouts_per_group
        gate = asyncio.Semaphore(num_workers)
        work_queue: asyncio_Queue = asyncio_Queue()  # dispatch -> production
        produced_rollouts: asyncio_Queue = asyncio_Queue()  # production -> assemble
        regen_groups: asyncio_Queue = asyncio_Queue()  # assemble -> regen worker
        filtered_groups: asyncio_Queue = asyncio_Queue()  # assemble -> consumption

        # Because GRPO's training granularity is inherently group-level, anything above that level
        # must release the gate on consumption to correctly implement the desired lag shape.
        release_on_consume = request.consumption_granularity == "B"

        def release_group_permits():
            for _ in range(rollouts_per_group):
                gate.release()

        async def dispatch_group(batch_id, index_in_batch, acquire):
            ctx = await self.prepare_group(request)
            for _ in range(rollouts_per_group):
                if acquire:
                    await gate.acquire()
                await work_queue.put((batch_id, index_in_batch, ctx))

        @trace_async_exceptions(verbose=True)
        async def regen_worker():
            regen_acquire = not release_on_consume
            while True:
                await dispatch_group(*await regen_groups.get(), acquire=regen_acquire)

        @trace_async_exceptions(verbose=True)
        async def dispatch_worker():
            batch_id = 0
            while request.streaming or batch_id < 1:
                # Acquire the batch's rollout units at the submission granularity, then dispatch.
                if request.submission_granularity == "B":
                    for _ in range(groups_per_batch * rollouts_per_group):
                        await gate.acquire()
                for index_in_batch in range(groups_per_batch):
                    if request.submission_granularity == "G":
                        for _ in range(rollouts_per_group):
                            await gate.acquire()
                    await dispatch_group(
                        batch_id, index_in_batch, acquire=request.submission_granularity == "R"
                    )
                batch_id += 1

        @trace_async_exceptions(verbose=True)
        async def production_worker():
            while True:
                batch_id, index_in_batch, ctx = await work_queue.get()
                rollout = await self.generate_rollout(request, ctx)
                if not release_on_consume:
                    gate.release()  # on finish -> bounds (dispatched - finished)
                await produced_rollouts.put((batch_id, index_in_batch, rollout))

        @trace_async_exceptions(verbose=True)
        async def assemble_worker():
            rollout_buffers: dict[tuple[int, int], Rollouts] = {}
            batch_buffers: dict[int, dict[int, RolloutGroup]] = {}
            next_batch_id = 0
            while True:
                # Collate rollouts into a group.
                batch_id, index_in_batch, rollout = await produced_rollouts.get()
                buffer = rollout_buffers.setdefault((batch_id, index_in_batch), [])
                buffer.append(rollout)
                if len(buffer) < rollouts_per_group:
                    continue
                del rollout_buffers[(batch_id, index_in_batch)]
                group = RolloutGroup(
                    rollouts=buffer, batch_id=batch_id, index_in_batch=index_in_batch
                )

                if request.filter_groups_with_same_reward:
                    if np.std([r.reward for r in group]) <= 1e-6:
                        # Drop and regenerate the slot.
                        regen_groups.put_nowait((batch_id, index_in_batch))
                        continue

                # Assemble according to consumption granularity.
                if request.consumption_granularity == "B":
                    batch_buffers.setdefault(batch_id, {})[index_in_batch] = group
                    while len(batch_buffers.get(next_batch_id, {})) == groups_per_batch:
                        ready = batch_buffers.pop(next_batch_id)
                        next_batch_id += 1
                        await filtered_groups.put([ready[i] for i in range(groups_per_batch)])
                else:
                    await filtered_groups.put([group])

        async def consumption_worker():
            delivered_groups = 0
            while request.streaming or delivered_groups < groups_per_batch:
                for group in await filtered_groups.get():
                    yield group
                    delivered_groups += 1
                    if release_on_consume:
                        release_group_permits()  # on consume -> bounds (dispatched - consumed)

        tasks = [asyncio.create_task(production_worker()) for _ in range(num_workers)]
        tasks.append(asyncio.create_task(assemble_worker()))
        tasks.append(asyncio.create_task(dispatch_worker()))
        tasks.append(asyncio.create_task(regen_worker()))

        try:
            async for group in consumption_worker():
                yield group
        finally:
            for task in tasks:
                task.cancel()


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
