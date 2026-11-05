# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resuming partially generated groups and mid-decode requests across a restart."""

import asyncio
import copy
import dataclasses
from contextlib import aclosing
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from pydantic import Field

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    RESUMED_GENERATION_FIELD,
    LiveRequestSnapshot,
    RequestPromptPreparationResult,
    ResumedGeneration,
)
from megatron.rl.agent import reward_only_agent
from megatron.rl.agent.api import GroupedRolloutRequest, GroupRolloutParams, TokenRollout
from megatron.rl.agent.rollout_pipeline import RolloutPipeline
from megatron.rl.inference import InferenceRequest, InferenceResponse, LLMChatMessage, ReturnsTokens
from megatron.rl.inference.megatron import MegatronLocal
from megatron.rl.rollout_bank import (
    LiveRequestIndex,
    LiveRequestSummary,
    live_request_path,
    live_request_paths,
    pack_live_request,
    write_live_requests,
)
from megatron.rl.rollout_resume import (
    EPISODE_FIELD,
    RESUME_FIELD,
    EpisodeJournal,
    EpisodeKey,
    ResumingPromptPreparer,
    TurnRecord,
    current_episode,
    generate_in_episode,
    plan_partial_groups,
    run_in_episode,
)
from tests.unit_tests.rl.test_rollout_bank import _record as bank_record
from tests.unit_tests.rl.test_rollout_generation import (
    EpisodeAgent,
    MockGenerator,
    MockInferenceInterface,
)

BLUEPRINT = {"prompt": "q"}


def _msg(role, content):
    return {"role": role, "content": content}


def _reply(cid, content="a"):
    return {**_msg("assistant", content), "finish_reason": "stop", "completion_id": cid}


def _turn(prompt, cid=None, content="a"):
    """A manifest turn, answered when it has a completion id."""
    return {"prompt": prompt, "response": _reply(cid, content) if cid else None}


def _group(uid, members, *, blueprint=BLUEPRINT, env_id="e", batch_id=0, index_in_batch=0):
    """A manifest group; ``_open`` opens journal groups with the same defaults."""
    slot = {"batch_id": batch_id, "index_in_batch": index_in_batch}
    return {"uid": uid, "env_id": env_id, "blueprint": blueprint, **slot, "members": members}


def _open(journal, uid, n=1, *, plans=None, blueprint=BLUEPRINT):
    journal.open_group(
        uid, env_id="e", blueprint=blueprint, batch_id=0, index_in_batch=0, members=n, plans=plans
    )


def _record(generated=(2,), prompt=(1,), **fields):
    return bank_record(generated, prompt=prompt, num_tokens_to_generate=8, **fields)


def _live(uid, tag, generated=1):
    """A live request as the restore scans it, tagged with an episode turn (or not)."""
    return LiveRequestSummary(uid, None if tag is None else {EPISODE_FIELD: tag}, generated, "p")


def _dump(messages):
    return [message.model_dump() for message in messages]


P0 = [_msg("user", "hello")]
P1 = [*P0, _msg("assistant", "a0"), _msg("user", "obs0")]
P2 = [*P1, _msg("assistant", "a1"), _msg("user", "obs1")]
OTHER = [*P1[:2], _msg("user", "another observation")]
T4 = [_msg("user", "t4")]
R0, R1 = _reply("cc-0"), _reply("cc-1")
PLAN3 = [TurnRecord(P0, R0), TurnRecord(P1, R1), TurnRecord(P2, None, "live-2")]

JOURNAL_CASES = {
    # plan, re-enter, resume point (ids, conversation, pending), issued turns
    # (prompt, index, served id, resume uid, answer id), counters, unsettled tags before/after
    "planned": (
        [TurnRecord(P0, R0), TurnRecord(P1, None, "live-1")],
        False,
        None,
        [(P0, 0, "cc-0", None, None), (P1, 1, None, "live-1", "cc-new"), (P2, 2, None, None, None)],
        (1, 1, 0),
        ({"g/0/1"}, set()),
    ),
    "diverged": (
        PLAN3,
        False,
        None,
        [(P0, 0, "cc-0", None, None), (OTHER, 1, None, None, None)],
        (1, 0, 1),
        ({"g/0/2"}, set()),
    ),
    "reentered": (
        PLAN3,
        True,
        (["cc-0", "cc-1"], P1, P2),
        [(P2, 2, None, "live-2", None)],
        (2, 1, 0),
        ({"g/0/2"}, {"g/0/2"}),
    ),
}


@pytest.mark.parametrize(
    "plan, reenter, point, issued, counters, unsettled", JOURNAL_CASES.values(), ids=JOURNAL_CASES
)
def test_journal(plan, reenter, point, issued, counters, unsettled):
    """A planned episode gets its answered turns served and its unanswered one resumed; a replay
    whose prompt differs abandons the rest of its plan; a re-entering agent consumes the answered
    turns once. Every turn is tagged, a resume stays unsettled until answered, and blueprint
    groups stay in the manifest until they close."""
    journal = EpisodeJournal()
    _open(journal, "g", plans=[plan])
    _open(journal, "no-blueprint", 2, blueprint=None)
    key = EpisodeKey("g", 0)
    assert journal.unsettled_resumes() == unsettled[0]
    if reenter:
        resumed = journal.resume_point(key)
        pending = None if resumed.pending_prompt is None else _dump(resumed.pending_prompt)
        ids = [r.completion_id for r in resumed.responses]
        assert (ids, _dump(resumed.conversation), pending) == point
        assert journal.resume_point(key) is None
    for prompt, index, served, resume_uid, answer in issued:
        ticket = journal.begin_turn(key, prompt)
        assert (ticket.index, ticket.tag, ticket.resume_uid) == (index, f"g/0/{index}", resume_uid)
        assert (None if ticket.served is None else ticket.served["completion_id"]) == served
        if answer is not None:
            journal.finish_turn(key, index, _reply(answer))
    assert (journal.served_turns, journal.resumed_requests, journal.abandoned_plans) == counters
    assert journal.unsettled_resumes() == unsettled[1]
    consumed = plan[: len(point[0])] if reenter else []
    turns = [{"prompt": t.prompt, "response": t.response} for t in consumed]
    turns += [_turn(prompt, served or answer) for prompt, _, served, _, answer in issued]
    assert journal.manifest() == [_group("g", [turns])]
    journal.close_group("g")
    assert journal.manifest() == []


def test_plan_partial_groups_keeps_what_the_disk_can_rebuild():
    """Answered turns survive while their records are on disk and joinable, the first missing
    record cuts the plan there, the unanswered turn resumes from the longest live request tagged
    with it; blueprint-less and malformed groups are skipped."""
    records = {"cc-a": _record(), "cc-b": _record(log_probs=None), "cc-c": _record()}
    live = [_live("live-old", "g/1/1", 3), _live("live-new", "g/1/1", 5), _live("live-bare", None)]
    live.append(LiveRequestSummary("live-gym", {"ng_capture": {}}, 1, "p"))
    members = [
        [_turn(P0, "cc-a"), _turn(P1, "cc-b"), _turn(P2)],
        [_turn(P0, "cc-c"), _turn(P1)],
        [_turn(P0, "cc-missing")],
        [],
    ]
    groups = [
        _group("g", members),
        _group("no-blueprint", [[]], blueprint=None),
        {"uid": "malformed", "env_id": "e", "blueprint": {}, "members": "nope"},
        {"uid": "truncated"},
    ]
    (plan,) = plan_partial_groups(groups, records, {s.uid: s for s in live})
    assert (plan.uid, plan.env_id, plan.blueprint) == ("g", "e", BLUEPRINT)
    assert plan.record_ids == ["cc-a", "cc-c"]
    assert plan.members == [
        [TurnRecord(P0, _reply("cc-a"))],
        [TurnRecord(P0, _reply("cc-c")), TurnRecord(P1, None, "live-new")],
        [],
        [],
    ]


def test_group_blueprint_warns_once_when_the_golden_is_not_json(monkeypatch, caplog):
    """A golden JSON cannot carry leaves the group without a blueprint, so it is regenerated after
    a restart rather than resumed; the agent says so once per process, not once per group."""
    monkeypatch.setattr(reward_only_agent, "_NON_JSON_GOLDEN_LOGGED", False)
    blueprint = reward_only_agent.RewardOnlyAgent._group_blueprint
    with caplog.at_level("WARNING", logger="megatron.rl.agent.reward_only_agent"):
        blueprints = [blueprint("q", golden) for golden in (object(), {"a": object()}, {"a": 1})]
    assert blueprints == [None, None, {"prompt": "q", "golden": {"a": 1}}]
    assert sum("not JSON-serializable" in r.message for r in caplog.records) == 1


def _preparer(tmp_path, inner=None):
    """A preparer over one run's live file: a resumable request and one without log probs."""
    frames = [
        pack_live_request("live-1", _record([10, 11], [1, 2, 3]), {EPISODE_FIELD: "g/0/0"}),
        pack_live_request("live-nolp", _record([5], log_probs=None), {EPISODE_FIELD: "g/1/0"}),
    ]
    write_live_requests(live_request_path(str(tmp_path), "run1", 0), frames)
    return ResumingPromptPreparer(inner, LiveRequestIndex(str(tmp_path)))


def _splice(prompt, *, offload_params):
    """An inner preparer that resolves every prompt to the snapshot's."""
    return RequestPromptPreparationResult([1, 2, 3], offload_params)


# "live-1" as the engine resumes it: tokens, budget, log probs, prompt log probs, epochs.
WIRE = ResumedGeneration([10, 11], 8, [-0.5] * 2, None, [(0, 3)], [(0, 3)]).to_wire()
INNER = SimpleNamespace(prepare_prompt=_splice)
SAME = object()  # the prompt object the preparer was handed
FRESH = {EPISODE_FIELD: "g/0/0"}
TAGGED = {**FRESH, RESUME_FIELD: "live-1"}
# What the endpoint adds to a tagged multi-turn call, plus a key of the caller's own.
FULL = {**TAGGED, PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [1], PREFIX_EOS_TOKEN_ID_FIELD: [2]}
FULL |= {PREFIX_MEDIA_COUNT_FIELD: 0, "keep": 1}
SPLICED = {**FRESH, RESUMED_GENERATION_FIELD: WIRE}
RESUMED = {**SPLICED, "keep": 1}

PREPARER_CASES = {
    # prompt, offload params, inner preparer, (prompt out, params out), "cannot resume" lines
    "list-prompt": ([1, 2, 3], FULL, None, (SAME, RESUMED), 0),
    "tensor-prompt": (torch.tensor([1, 2, 3]), FULL, None, (SAME, RESUMED), 0),
    "untagged": ([7], None, None, (SAME, None), 0),
    "gym-admission": ([7], {"ng_capture": 1}, None, (SAME, {"ng_capture": 1}), 0),
    "inner-splice": ([9, 9], TAGGED, INNER, ([1, 2, 3], SPLICED), 0),
    "prompt-differs": ([1, 2, 4], TAGGED, None, (SAME, FRESH), 1),
    "unknown-request": ([1, 2, 3], {**FRESH, RESUME_FIELD: "live-missing"}, None, (SAME, FRESH), 1),
    "string-prompt": ("one two", TAGGED, None, (SAME, FRESH), 1),
    "no-log-probs": ([1], {**FRESH, RESUME_FIELD: "live-nolp"}, None, (SAME, FRESH), 1),
}


@pytest.mark.parametrize(
    "prompt, params, inner, expected, refused", PREPARER_CASES.values(), ids=PREPARER_CASES
)
def test_preparer(tmp_path, caplog, prompt, params, inner, expected, refused):
    """A re-issued turn whose prompt matches its snapshot gets the engine's resumed generation,
    once the inner preparer resolved the prompt, minus the resume key and the endpoint's splice
    metadata; anything that does not fit is admitted fresh with a log line, since a refused
    resume would fail the request; untagged and Gym calls pass through untouched."""
    given = copy.deepcopy(params)
    with caplog.at_level("WARNING", logger="megatron.rl.rollout_resume"):
        result = _preparer(tmp_path, inner).prepare_prompt(prompt, offload_params=params)
    out_prompt, out_params = expected
    assert result.prompt is prompt if out_prompt is SAME else result.prompt == out_prompt
    assert (result.offload_params, params) == (out_params, given)  # caller's dict untouched
    assert sum("cannot resume" in r.message for r in caplog.records) == refused


class JournalingInterface(ReturnsTokens):
    """What MegatronLocal does per turn, over a scripted engine: the reply is a function of the
    conversation length, ids are minted per call, each call's offload params are kept, and a
    call for a (member, turn) in ``stall`` never answers (a kill lands there)."""

    by_prompt_length: dict = Field(default_factory=dict)
    stall: set = Field(default_factory=set)
    calls: list = Field(default_factory=list)

    async def base_generate(self, request):
        return await generate_in_episode(request, self._complete)

    async def _complete(self, request, offload_params):
        episode = current_episode()
        self.calls.append((_dump(request.prompt), offload_params))
        if episode is not None and (episode.key.member, len(request.prompt) // 2) in self.stall:
            await asyncio.Event().wait()
        reply = LLMChatMessage(role="assistant", content=self.by_prompt_length[len(request.prompt)])
        return InferenceResponse(
            response=reply, finish_reason="stop", completion_id=f"cc-{len(self.calls)}"
        )


def _scripted(script):
    return partial(JournalingInterface, by_prompt_length=script)


class StallingEpisodeAgent(EpisodeAgent):
    """EpisodeAgent whose observation for a (member, turn) in ``stall_observation`` hangs."""

    stall_observation: set = Field(default_factory=set)

    async def get_observation(self, turn_idx, response, conversation, golden):
        episode = current_episode()
        if episode is not None and (episode.key.member, turn_idx) in self.stall_observation:
            await asyncio.Event().wait()
        return await super().get_observation(turn_idx, response, conversation, golden)


class BlueprintMockGenerator(MockGenerator):
    """A custom episode loop that returns a blueprint: its groups resume by replay, served turns
    coming back from the journal through the interface."""

    async def prepare_group_rollout(self, request):
        params = await super().prepare_group_rollout(request)
        idx = self._call_count - 1

        async def build_rollout(episode):
            reward = float(episode.responses[-1].response.content.removeprefix("t"))
            ids = [r.completion_id for r in episode.responses]
            return TokenRollout(
                trajectory=None, reward=reward, env_id=self.env_id, completion_ids=ids
            )

        return GroupRolloutParams(params.run_episode, build_rollout, blueprint={"idx": idx})

    async def restore_group_rollout(self, request, blueprint):
        self._call_count = blueprint["idx"]
        return await self.prepare_group_rollout(request)


SCRIPT = {1: "a0", 3: "a1", 5: "a2"}


def _pipeline(agent, interface, members, partial=None):
    request = GroupedRolloutRequest(
        num_groups=1, rollouts_per_group=members, inference_interface=interface
    )
    return RolloutPipeline(agent, request, parallel_generation_tasks=1, partial_groups=partial)


async def _collect_one(agent, interface, members, partial=None):
    """Run a fresh pipeline until it yields one group."""
    pipeline = _pipeline(agent, interface, members, partial)
    async with aclosing(pipeline.run()) as stream:
        return pipeline, await asyncio.wait_for(anext(stream), timeout=5)


async def _interrupt(pipeline, settled, rounds=500):
    """Run a pipeline until ``settled()`` holds, then cancel it where it stands (the kill)."""
    stream = pipeline.run()
    consumer = asyncio.create_task(anext(stream))
    try:
        for _ in range(rounds):
            if settled():
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("the pipeline did not reach the expected state")
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await stream.aclose()


@pytest.mark.asyncio
async def test_pipeline_resumes_a_partial_group_where_each_episode_stopped():
    """A three-member group is killed with its episodes finished, mid-decode on the last turn and
    between turns. At restart the finished episode is served without inference or environment
    steps, the mid-decode turn is re-issued naming its live request, the between-turns episode
    re-runs only the observation it had not taken, and the group names served and new ids."""
    interface = JournalingInterface(by_prompt_length=SCRIPT, stall={(1, 2)})
    agent = StallingEpisodeAgent(max_turns=3, stall_observation={(2, 0)})
    pipeline = _pipeline(agent, interface, 3)
    # Members 0 and 1 take two observations each; member 2's first one never returns.
    settled = lambda: len(interface.calls) == 7 and sorted(agent.observation_turns) == [0, 0, 1, 1]
    await _interrupt(pipeline, settled)
    (group,) = manifest = pipeline.inflight_manifest()
    uid, members = group["uid"], group["members"]
    assert group["blueprint"] == {"prompt": "hello", "golden": {"problem_id": "p0"}}
    assert [[t["prompt"] for t in m] for m in members] == [[P0, P1, P2], [P0, P1, P2], [P0]]
    answered = [[t["response"]["completion_id"] for t in m if t["response"]] for m in members]
    assert [len(ids) for ids in answered] == [3, 2, 1]
    finished = {cid: _record() for ids in answered for cid in ids}
    (plan,) = plan_partial_groups(manifest, finished, {"live-1": _live("live-1", f"{uid}/1/2", 4)})
    assert (plan.record_ids, plan.members[1][2]) == (list(finished), TurnRecord(P2, None, "live-1"))

    interface = JournalingInterface(by_prompt_length=SCRIPT)
    agent = StallingEpisodeAgent(max_turns=3)
    resumed, restored = await _collect_one(agent, interface, 3, [plan])
    calls = {p[EPISODE_FIELD].removeprefix(f"{uid}/"): (prompt, p) for prompt, p in interface.calls}
    assert calls == {
        "1/2": (P2, {EPISODE_FIELD: f"{uid}/1/2", RESUME_FIELD: "live-1"}),
        "2/1": (P1, {EPISODE_FIELD: f"{uid}/2/1"}),
        "2/2": (P2, {EPISODE_FIELD: f"{uid}/2/2"}),
    }
    assert agent.observation_turns == [0, 1]  # the between-turns episode only
    assert all(isinstance(r, TokenRollout) and r.reward == 1.5 for r in restored.rollouts)
    ids = [r.completion_ids for r in restored.rollouts]
    assert ids[0] == answered[0] and ids[1] == [*answered[1], "cc-1"]
    assert ids[2][:1] == answered[2] and len(ids[2]) == 3
    # Rewards saw the full dialogues, the served turns included.
    dialogue = [(m["role"], m["content"]) for m in [*P2, _msg("assistant", "a2")]]
    assert [(m.role, m.content) for m in agent.reward_conversation[:6]] == dialogue
    journal = resumed.journal
    assert (resumed.resumed_count, journal.served_turns, journal.resumed_requests) == (1, 6, 1)
    assert (resumed.unsettled_resumes(), resumed.inflight_manifest()) == (set(), [])


def _replay_plan():
    """One group of two single-turn episodes, the first answered, the second mid-decode."""
    members = [[_turn(T4, "cc-old", "t4")], [_turn(T4)]]
    manifest = [_group("g", members, blueprint={"idx": 4}, env_id="test")]
    return plan_partial_groups(
        manifest, {"cc-old": _record()}, {"live-7": _live("live-7", "g/1/0", 2)}
    )


REPLAY_CASES = {
    # agent, interface, members, groups resumed, turns served
    "replayed": (BlueprintMockGenerator, _scripted({1: "t4"}), 2, 1, 1),
    "no-blueprint": (MockGenerator, MockInferenceInterface, 2, 0, 0),
    "member-count": (BlueprintMockGenerator, _scripted({1: "t0"}), 3, 0, 0),
}


@pytest.mark.parametrize(
    "agent, interface, members, resumed, served", REPLAY_CASES.values(), ids=REPLAY_CASES
)
@pytest.mark.asyncio
async def test_pipeline_replays_a_partial_group_or_regenerates_it(
    agent, interface, members, resumed, served
):
    """An agent that cannot re-enter its loop replays each episode: the journal serves the
    answered turn without a call and the unanswered one is re-issued naming its live request.
    A group with no blueprint, or whose member count changed, is prepared afresh."""
    agent, interface = agent(), interface()
    pipeline, group = await _collect_one(agent, interface, members, _replay_plan())
    counts = (len(group), pipeline.resumed_count, pipeline.journal.served_turns)
    assert counts == (members, resumed, served)
    assert agent.prepare_group_rollout_calls == 1
    if resumed:
        assert [r.completion_ids for r in group.rollouts] == [["cc-old"], ["cc-1"]]
        assert interface.calls == [(T4, {EPISODE_FIELD: "g/1/0", RESUME_FIELD: "live-7"})]


@pytest.mark.asyncio
async def test_megatron_local_snapshots_live_requests_and_tags_episode_turns(
    tmp_path, monkeypatch, caplog
):
    """At a suspend the interface writes its engine's live requests on coordinator ranks only and
    never fails the run; earlier runs' files go once every unsettled resume decodes under this
    run. In a pipeline episode a completion is tagged and journaled; otherwise it is plain."""
    monkeypatch.setattr("megatron.rl.inference.megatron.get_run_config", lambda: MagicMock())
    monkeypatch.setattr("megatron.rl.inference.megatron.get_tokenizer", lambda: MagicMock(bos=None))
    server = MegatronLocal(host="localhost", port=0)
    server._request_record_backup_dir, server._live_run_tag = str(tmp_path), "run2"
    server._live_request_path = mine = live_request_path(str(tmp_path), "run2", 0)
    record, tag = _record([3], [1, 2]), {EPISODE_FIELD: "g/0/0"}
    snapshot = LiveRequestSnapshot(
        "live-1", 1, tag, record.payload, dataclasses.replace(record, payload=None)
    )
    snapshots = AsyncMock(return_value=[snapshot])
    server._inference_engine = engine = SimpleNamespace(
        is_mp_coordinator=False, snapshot_live_requests=snapshots
    )
    await server._save_live_requests()
    assert live_request_paths(str(tmp_path)) == []  # a worker rank writes nothing
    engine.is_mp_coordinator = True
    await server._save_live_requests()
    assert LiveRequestIndex(str(tmp_path)).get("live-1") == ("live-1", record, tag)
    engine.snapshot_live_requests = AsyncMock(side_effect=RuntimeError("no snapshot"))
    with caplog.at_level("ERROR", logger="megatron.rl.inference.megatron"):
        await server._save_live_requests()  # logged, never raised
    assert any("Could not save the live requests" in r.message for r in caplog.records)
    stale = live_request_path(str(tmp_path), "run1", 0)
    # An unadmitted tag blocks pruning, an admitted one frees the older run, rank 1 never prunes.
    for rank, unsettled, left in (
        (0, {"g/0/0", "g/0/1"}, [stale, mine]),
        (0, {"g/0/0"}, [mine]),
        (1, set(), [stale, mine]),
    ):
        write_live_requests(stale, [])
        with patch("torch.distributed.get_rank", return_value=rank):
            server.prune_stale_live_requests(unsettled)
        assert live_request_paths(str(tmp_path)) == left

    choice = MagicMock(finish_reason="stop")
    choice.message.model_dump.return_value = _msg("assistant", "response")
    completion = MagicMock(id="completion-id", choices=[choice])
    server._openai_client = client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    request = InferenceRequest(prompt=[LLMChatMessage(role="user", content="prompt")])
    journal = EpisodeJournal()
    for uid, in_episode, enabled in (
        ("plain", False, True),
        ("on", True, True),
        ("off", True, False),
    ):
        server._live_request_path = mine if enabled else None
        _open(journal, uid, blueprint={})
        generate = lambda: server.base_generate(request)
        response = await (
            run_in_episode(journal, EpisodeKey(uid, 0), generate) if in_episode else generate()
        )
        sent = client.chat.completions.create.await_args.kwargs["extra_body"].get("offload_params")
        assert (response.completion_id, sent) == (
            "completion-id",
            {EPISODE_FIELD: "on/0/0"} if uid == "on" else None,
        )
    turn = _turn([_msg("user", "prompt")], "completion-id", "response")
    assert {g["uid"]: g["members"] for g in journal.manifest()} == {
        "plain": [[]],
        "on": [[turn]],
        "off": [[]],
    }
