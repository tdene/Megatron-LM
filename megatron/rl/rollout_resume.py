# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resume partially generated rollout groups and mid-decode requests across a restart.

The durable rollout bank keeps completed groups. This module covers what a kill
interrupts: episodes with turns still to run, and requests still decoding.

Identity. The pipeline mints a uid per prepared group, and every request an episode
makes is tagged ``<group uid>/<member>/<turn>`` in its ``offload_params``
(``EPISODE_FIELD``), so the engine's snapshot of a live request names the turn it
belongs to. The ``EpisodeJournal`` on rank 0's pipeline records each episode's turns as
the interface sees them: the prompt messages sent and the reply received. At every
collection the journal is written to the bank beside the engines' live-request files.

Restart. ``plan_partial_groups`` joins the manifest, the finished-request records and the
live requests into ``PartialGroup``s. The pipeline re-prepares each from its ``blueprint``
(``GroupedRolloutGenerator.restore_group_rollout``) and installs the plan in the journal.
The interface then answers a finished turn from the journal without inference, and
resumes a mid-decode turn by naming its live request (``RESUME_FIELD``), which
``ResumingPromptPreparer`` turns into the engine's ``RESUMED_GENERATION_FIELD`` on the
model-parallel coordinator; the engine continues from the snapshotted tokens under the
same budget, stamps the new epoch at the cut and counts one eviction. ``RewardOnlyAgent``
re-enters its loop at the resumed turn; any other agent replays its episode and gets
the finished turns served.

Everything here is best effort: a turn whose prompt no longer matches its plan, whose
record did not reach disk or whose live request is gone is regenerated from there.
"""

import contextvars
import dataclasses
import logging
from typing import Any, Awaitable, Callable, Iterable, NamedTuple, Optional, Sequence

import torch

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    RESUMED_GENERATION_FIELD,
    FinishedRequestRecord,
    RequestPromptPreparationResult,
    RequestPromptPreparer,
    ResumedGeneration,
)

from .gym_capture import NG_CAPTURE_FIELD
from .inference.api import InferenceRequest, InferenceResponse, LLMChatMessage
from .rollout_bank import LiveRequestIndex, LiveRequestSummary, _joinable

logger = logging.getLogger(__name__)

#: `offload_params` key naming the episode turn a request serves: "<group uid>/<member>/<turn>".
EPISODE_FIELD = "mrl_episode"
#: `offload_params` key naming the live request a re-issued turn resumes (its uid).
RESUME_FIELD = "mrl_resume"


# ------------------------------------------------------------------ journal types
@dataclasses.dataclass
class TurnRecord:
    """One turn of an episode: the prompt messages sent and, once answered, the reply."""

    prompt: list[dict]
    response: Optional[dict] = None
    resume_uid: Optional[str] = None  # restore only: the live request that resumes this turn


class EpisodeKey(NamedTuple):
    """Identifies one member episode of a prepared group."""

    group_uid: str
    member: int


class TurnTicket(NamedTuple):
    """What the journal says about the turn an episode is about to issue."""

    index: int
    tag: str
    served: Optional[dict]  # the reply recorded before the restart: no inference needed
    resume_uid: Optional[str]  # the live request to continue, if the turn was mid-decode


@dataclasses.dataclass
class EpisodeResume:
    """Where a resumed episode re-enters its loop."""

    responses: list[InferenceResponse]  # the turns already answered, in order
    conversation: list[LLMChatMessage]  # the prompt of the last answered turn
    pending_prompt: Optional[list[LLMChatMessage]]  # the next turn's prompt, if it was issued


@dataclasses.dataclass
class _Episode:
    turns: list[TurnRecord] = dataclasses.field(default_factory=list)
    cursor: int = 0  # turns before it were consumed by the episode running now


@dataclasses.dataclass
class _Group:
    uid: str
    env_id: str
    blueprint: Optional[dict]
    batch_id: int
    index_in_batch: int
    members: list[_Episode]


def _messages(messages: Sequence[LLMChatMessage]) -> list[dict]:
    return [message.model_dump() for message in messages]


def _as_messages(messages: Iterable[dict]) -> list[LLMChatMessage]:
    return [LLMChatMessage(**message) for message in messages]


def turn_response(response: InferenceResponse) -> dict:
    """The journal's record of a reply: what an agent reads off an InferenceResponse."""
    return {
        **response.response.model_dump(),
        "finish_reason": response.finish_reason,
        "completion_id": response.completion_id,
    }


def served_response(turn: dict) -> InferenceResponse:
    """A finished turn as the interface returns it: the reply, its reason and its request id."""
    return InferenceResponse(
        response=LLMChatMessage(role=turn["role"], content=turn["content"]),
        finish_reason=turn["finish_reason"],
        completion_id=turn["completion_id"],
    )


class EpisodeJournal:
    """Rank 0's record of the turns of every in-flight episode, by group and member."""

    def __init__(self) -> None:
        self._groups: dict[str, _Group] = {}
        # Turns answered from the journal instead of the engine.
        self.served_turns = 0
        # Requests sent naming a live request to continue.
        self.resumed_requests = 0
        # Episodes whose replayed prompt diverged from their plan and regenerated from there.
        self.abandoned_plans = 0

    def open_group(
        self,
        uid: str,
        *,
        env_id: str,
        blueprint: Optional[dict],
        batch_id: int,
        index_in_batch: int,
        members: int,
        plans: Optional[list[list[TurnRecord]]] = None,
    ) -> None:
        """Start recording a group's episodes, each continuing its plan when one is given."""
        if plans is None:
            plans = [[] for _ in range(members)]
        assert len(plans) == members, f"{len(plans)} plans for {members} members"
        episodes = [_Episode([dataclasses.replace(turn) for turn in plan]) for plan in plans]
        self._groups[uid] = _Group(uid, env_id, blueprint, batch_id, index_in_batch, episodes)

    def close_group(self, uid: str) -> None:
        """Forget a group once all of its episodes are in."""
        self._groups.pop(uid, None)

    def _episode(self, key: EpisodeKey) -> _Episode:
        return self._groups[key.group_uid].members[key.member]

    def resume_point(self, key: EpisodeKey) -> Optional[EpisodeResume]:
        """Consume the answered turns of `key`'s plan, for an agent that re-enters its loop.

        Returns None when there is nothing to skip: the episode runs from its first turn, and
        a planned first turn is matched by `begin_turn`.
        """
        episode = self._episode(key)
        if episode.cursor:
            return None
        answered = []
        for turn in episode.turns:
            if turn.response is None:
                break
            answered.append(turn)
        if not answered:
            return None
        pending = episode.turns[len(answered)] if len(answered) < len(episode.turns) else None
        episode.cursor = len(answered)
        self.served_turns += len(answered)
        return EpisodeResume(
            responses=[served_response(turn.response) for turn in answered],
            conversation=_as_messages(answered[-1].prompt),
            pending_prompt=None if pending is None else _as_messages(pending.prompt),
        )

    def begin_turn(self, key: EpisodeKey, prompt: list[dict]) -> TurnTicket:
        """Record that `key` issues `prompt`; say whether the plan already answers it."""
        episode = self._episode(key)
        index = episode.cursor
        tag = f"{key.group_uid}/{key.member}/{index}"
        if index < len(episode.turns):
            planned = episode.turns[index]
            if planned.prompt == prompt:
                episode.cursor += 1
                if planned.response is not None:
                    self.served_turns += 1
                    return TurnTicket(index, tag, planned.response, None)
                if planned.resume_uid is not None:
                    self.resumed_requests += 1
                return TurnTicket(index, tag, None, planned.resume_uid)
            # The replayed episode built a different prompt: its plan ends here.
            self.abandoned_plans += 1
            logger.warning(
                "Episode %s/%d replayed turn %d with a different prompt; regenerating from there",
                key.group_uid,
                key.member,
                index,
            )
            del episode.turns[index:]
        episode.turns.append(TurnRecord(prompt))
        episode.cursor = len(episode.turns)
        return TurnTicket(index, tag, None, None)

    def finish_turn(self, key: EpisodeKey, index: int, response: dict) -> None:
        """Record the reply of turn `index`."""
        self._episode(key).turns[index].response = response

    def unsettled_resumes(self) -> set[str]:
        """Tags of the planned turns that resume a live request and are not answered yet."""
        return {
            f"{group.uid}/{member}/{index}"
            for group in self._groups.values()
            for member, episode in enumerate(group.members)
            for index, turn in enumerate(episode.turns)
            if turn.resume_uid is not None and turn.response is None
        }

    def manifest(self) -> list[dict]:
        """The in-flight groups a restart can resume, with every turn recorded so far."""
        return [
            {
                "uid": group.uid,
                "env_id": group.env_id,
                "blueprint": group.blueprint,
                "batch_id": group.batch_id,
                "index_in_batch": group.index_in_batch,
                "members": [
                    [{"prompt": turn.prompt, "response": turn.response} for turn in episode.turns]
                    for episode in group.members
                ],
            }
            for group in self._groups.values()
            if group.blueprint is not None
        ]


# ------------------------------------------------------------------ episode context
class EpisodeContext(NamedTuple):
    """The journal and key of the episode running on the current task."""

    journal: EpisodeJournal
    key: EpisodeKey


_CURRENT_EPISODE: contextvars.ContextVar[Optional[EpisodeContext]] = contextvars.ContextVar(
    "mrl_current_episode", default=None
)


def current_episode() -> Optional[EpisodeContext]:
    """The episode the calling task runs, if the pipeline is running one."""
    return _CURRENT_EPISODE.get()


async def run_in_episode(journal: EpisodeJournal, key: EpisodeKey, run_episode):
    """Run an episode coroutine with its journal and key visible to the inference interface."""
    token = _CURRENT_EPISODE.set(EpisodeContext(journal, key))
    try:
        return await run_episode()
    finally:
        _CURRENT_EPISODE.reset(token)


def resume_point() -> Optional[EpisodeResume]:
    """Where the current episode re-enters its loop, for agents that can skip answered turns."""
    episode = current_episode()
    return None if episode is None else episode.journal.resume_point(episode.key)


async def generate_in_episode(
    request: InferenceRequest,
    complete: Callable[[InferenceRequest, Optional[dict]], Awaitable[InferenceResponse]],
) -> InferenceResponse:
    """Serve one turn of the current episode."""
    episode = current_episode()
    if episode is None:
        return await complete(request, None)
    journal, key = episode
    ticket = journal.begin_turn(key, _messages(request.prompt))
    if ticket.served is not None:
        return served_response(ticket.served)
    offload_params: dict[str, Any] = {EPISODE_FIELD: ticket.tag}
    if ticket.resume_uid is not None:
        offload_params[RESUME_FIELD] = ticket.resume_uid
    response = await complete(request, offload_params)
    journal.finish_turn(key, ticket.index, turn_response(response))
    return response


# ------------------------------------------------------------------ restore plan
@dataclasses.dataclass
class PartialGroup:
    """A group the last run left in flight, with each member's plan of recorded turns."""

    uid: str
    env_id: str
    blueprint: dict
    members: list[list[TurnRecord]]

    @property
    def record_ids(self) -> list[str]:
        """The finished turns' request ids, which the join reads from the restored records."""
        return [
            turn.response["completion_id"]
            for member in self.members
            for turn in member
            if turn.response is not None
        ]


def plan_partial_groups(
    groups: list[dict],
    records: dict[str, FinishedRequestRecord],
    live: dict[str, LiveRequestSummary],
) -> list[PartialGroup]:
    """Decide what of each in-flight group survives: finished turns whose record is on disk,
    then at most one turn to re-issue, resumed from the live request tagged with it."""
    by_tag: dict[str, LiveRequestSummary] = {}
    for summary in live.values():
        params = summary.offload_params
        tag = params.get(EPISODE_FIELD) if isinstance(params, dict) else None
        if isinstance(tag, str):
            best = by_tag.get(tag)
            # A turn resumed and snapshotted again appears twice; the longer one is newer.
            if best is None or summary.generated > best.generated:
                by_tag[tag] = summary
    plans = []
    for group in groups:
        try:
            uid, env_id, blueprint = group["uid"], group["env_id"], group["blueprint"]
            member_turns = group["members"]
            if not (
                isinstance(uid, str)
                and isinstance(env_id, str)
                and isinstance(blueprint, dict)
                and isinstance(member_turns, list)
            ):
                raise TypeError("malformed in-flight group")
            members = []
            for member, turns in enumerate(member_turns):
                planned: list[TurnRecord] = []
                for index, turn in enumerate(turns):
                    prompt, response = turn["prompt"], turn.get("response")
                    if response is not None:
                        if not _joinable(records.get(response["completion_id"])):
                            break  # the turn's record did not reach disk: regenerate from here
                        planned.append(TurnRecord(prompt, response))
                        continue
                    # The turn in flight at the kill: continued from its snapshot, if one holds it.
                    summary = by_tag.get(f"{uid}/{member}/{index}")
                    planned.append(
                        TurnRecord(prompt, None, None if summary is None else summary.uid)
                    )
                    break
                members.append(planned)
        except (KeyError, TypeError) as error:
            logger.warning("Skipping a malformed in-flight group record: %s", error)
            continue
        plans.append(PartialGroup(uid, env_id, blueprint, members))
    return plans


# ------------------------------------------------------------------ preparer
class ResumingPromptPreparer:
    """RequestPromptPreparer that continues a re-issued turn from the live request it names."""

    def __init__(
        self, inner: Optional[RequestPromptPreparer], live_requests: LiveRequestIndex
    ) -> None:
        self.inner = inner
        self.live_requests = live_requests

    def prepare_prompt(
        self, prompt, *, offload_params: Optional[dict[str, Any]] = None
    ) -> RequestPromptPreparationResult:
        if self.inner is not None:
            result = self.inner.prepare_prompt(prompt, offload_params=offload_params)
            prompt, offload_params = result.prompt, result.offload_params
        if not isinstance(offload_params, dict) or EPISODE_FIELD not in offload_params:
            return RequestPromptPreparationResult(prompt=prompt, offload_params=offload_params)
        params = dict(offload_params)
        if NG_CAPTURE_FIELD not in params:
            # The endpoint ships prefix-splice metadata with any client offload params;
            # an episode turn is not a Gym continuation, so nothing is spliced.
            for field in (
                PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
                PREFIX_EOS_TOKEN_ID_FIELD,
                PREFIX_MEDIA_COUNT_FIELD,
            ):
                params.pop(field, None)
        uid = params.pop(RESUME_FIELD, None)
        if uid is not None:
            wire = self._resume_wire(uid, prompt, params[EPISODE_FIELD])
            if wire is not None:
                params[RESUMED_GENERATION_FIELD] = wire
        return RequestPromptPreparationResult(prompt=prompt, offload_params=params)

    def _resume_wire(self, uid: str, prompt, tag: str) -> Optional[dict]:
        live = self.live_requests.get(uid)
        if live is None:
            logger.warning("Turn %s cannot resume live request %r: not on disk", tag, uid)
            return None
        payload = live.record.payload
        if isinstance(prompt, torch.Tensor):
            tokens = prompt.tolist()
        elif isinstance(prompt, (list, tuple)):
            tokens = list(prompt)
        else:
            tokens = None
        if payload is None or tokens is None or tokens != list(payload.prompt_token_ids or []):
            logger.warning(
                "Turn %s cannot resume live request %r: its prompt differs; generating anew",
                tag,
                uid,
            )
            return None
        try:
            resumed = ResumedGeneration(
                token_ids=list(payload.generated_token_ids),
                num_tokens_to_generate=live.record.num_tokens_to_generate,
                log_probs=payload.generated_log_probs,
                prompt_log_probs=payload.prompt_log_probs,
                policy_epoch=live.record.policy_epoch,
                kv_cache_epoch=live.record.kv_cache_epoch,
                num_evictions=live.record.num_evictions,
                finish_reason=live.record.finish_reason,
                routing_indices=payload.routing_indices,
            )
            # The engine refuses a resume without what the request returns;
            # RL returns log probs and stamps epochs on every request.
            if resumed.log_probs is None or resumed.policy_epoch is None:
                raise ValueError("the live request carries no log probs or no epoch stamps")
        except ValueError as invalid:
            logger.warning("Turn %s cannot resume live request %r: %s", tag, uid, invalid)
            return None
        return resumed.to_wire()
