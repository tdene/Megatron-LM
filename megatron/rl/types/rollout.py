# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import deque
from typing import TypeAlias

from pydantic import BaseModel

#: An environment identifier, as reported by an agent's ``env_id`` attribute
#: (see ``WeightedMultiTask._rollout_env_ids``).
EnvId: TypeAlias = str

#: Maps each ``EnvId`` to a number of rollout groups for that env (e.g. the
#: per-batch generation target used to weight-balance restored bank groups).
GroupsPerEnv: TypeAlias = dict[EnvId, int]


class AgentBaseModel(BaseModel, extra='allow'):
    """Base model for agent data types."""


class Rollout(AgentBaseModel):
    """Data for language-based Rollout."""

    trajectory: list[str]
    prompt_length: list[int] | None = None
    reward: float | None = None
    env_id: str = ''
    # Metrics-only label. env_id is the ROUTING identity (WeightedMultiTask
    # registration, bank-restore bucketing, per-env quotas) — a blend
    # dispatcher stamps its own env_id on every child so restores stay within
    # known envs, which collapses per-env dashboards to one stream. When set,
    # this label keys the wandb/per-env metric prefixes instead (e.g. the
    # leaf agent ref), restoring per-env panels without touching routing.
    metrics_env_id: str | None = None
    problem_id: str | None = None


class TokenRollout(AgentBaseModel):
    """Tokenized representation of a language-based Rollout."""

    trajectory: list[list[int]]
    reward: list[float] | float
    generation_mask: list[list[bool]] | None = None
    logprobs: list[list[float]] | None = None
    env_id: str = ''
    # See Rollout.metrics_env_id: metrics-only label; env_id stays the
    # routing/restore identity.
    metrics_env_id: str | None = None
    problem_id: str | None = None
    # Per-turn output-token cap the agent stamped on generation (e.g.
    # NemoGym's max_output_tokens_per_step riding every /run body); None =
    # uncapped/unknown. Lets data-integrity checks verify arithmetically that
    # a non-eod turn under seq_length stopped exactly at its cap rather than
    # being silently truncated in transit (see single_turn_termination_ok).
    generation_cap: int | None = None


Rollouts = list[TokenRollout | Rollout]


class RolloutGroup(AgentBaseModel):
    """A group of rollouts (e.g. multiple completions for one prompt) with batch metadata."""

    rollouts: Rollouts
    batch_id: int = 0
    index_in_batch: int = 0
    # Stable identity in the durable rollout bank; None until (unless) banked.
    uid: str | None = None

    def __iter__(self):
        return iter(self.rollouts)

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        return self.rollouts[idx]


GroupedRollouts = list[RolloutGroup]

#: Maps each ``EnvId`` to a FIFO queue of completed ``RolloutGroup``s for that env.
#: Used by the durable rollout bank to bucket restored groups (and buffer streaming
#: overflow) per env so injection can respect per-env weight targets.
GroupQueuesPerEnv: TypeAlias = dict[EnvId, deque[RolloutGroup]]
