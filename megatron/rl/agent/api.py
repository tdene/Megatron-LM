# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Generic, NamedTuple, TypeVar

from ..__init__ import Request, TypeLookupable
from ..inference import (
    InferenceInterface,
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
)
from ..rollout_granularity import ConsumptionGranularity, SubmissionGranularity
from ..types import AgentBaseModel, GroupedRollouts, Rollout, RolloutGroup, Rollouts, TokenRollout


# TODO: Move these models to ``megatron.rl.types`` after moving ``Request``,
# ``InferenceInterface``, and their dependencies there to avoid circular imports.
class RolloutRequest(Request):
    """Request to agent to generate Rollouts."""

    num_rollouts: int
    inference_interface: InferenceInterface
    validation: bool = False


class GroupedRolloutRequest(Request):
    """Request to agent to generate grouped Rollouts."""

    num_groups: int
    rollouts_per_group: int
    inference_interface: InferenceInterface
    validation: bool = False
    filter_groups_with_same_reward: bool = False
    submission_granularity: SubmissionGranularity = "B"
    consumption_granularity: ConsumptionGranularity = "B"


class EpisodeResult(NamedTuple):
    """All per-turn responses of one (possibly multi-turn) episode plus the final conversation."""

    responses: list[InferenceResponse]
    conversation: list[LLMChatMessage]


class GroupRolloutParams(NamedTuple):
    """Returned by agent.prepare_group_rollout.

    One instance is created per group call and reused for all rollouts in that group.
    Every rollout is an episode: run_episode generates it (one or more turns), while
    build_rollout turns the completed episode into a Rollout.
    """

    run_episode: Callable[[], Awaitable[EpisodeResult]]
    build_rollout: Callable[[EpisodeResult], Awaitable[Rollout]]


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

    @abstractmethod
    async def get_rollout_response(
        self,
        request: "RolloutRequest | GroupedRolloutRequest | EvaluationRequest",
        inference_request: InferenceRequest,
    ) -> InferenceResponse:
        """Obtain the model response for a single rollout. Subclasses implement how."""
        ...


class RolloutGenerator(Agent, ABC):
    """An agent that produces Rollout objects containing rollout string and associated reward."""

    @abstractmethod
    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout]: ...


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
    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]: ...


class GroupedRolloutGenerator(Agent, ABC):
    """Agent contract consumed by RolloutPipeline to generate grouped rollouts (e.g. GRPO)."""

    @abstractmethod
    async def prepare_group_rollout(
        self,
        request: GroupedRolloutRequest,
        env_index: int = 0,
    ) -> GroupRolloutParams:
        """Return the params for one group's rollouts.

        Called once per group by `RolloutPipeline.stage_prepare`; the returned `run_episode` is
        awaited once per rollout in `RolloutPipeline.stage_infer` and `build_rollout` is invoked
        once per completed episode in `RolloutPipeline.stage_assemble`.

        Args:
            request: The grouped rollout request being served.
            env_index: An index of the environment this group belongs to.

        Returns:
            GroupRolloutParams carrying the run_episode and build_rollout closures.
        """
        ...

    def rollout_group_layout(self, num_groups: int) -> list[int]:
        """Returns the groups each env contributes to every trainer batch, in env order.

        Args:
            num_groups: Total groups in one trainer batch.

        Returns:
            Positive per-env group counts summing to num_groups.
        """
        return [num_groups]


class EvaluationAgent(Agent, ABC):
    """An agent that can take an inference interface and return a benchmark score."""

    @abstractmethod
    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse: ...
