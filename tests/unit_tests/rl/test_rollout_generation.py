# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from contextlib import aclosing
from unittest.mock import MagicMock

import pytest
from pydantic import Field, ValidationError

from megatron.rl.agent.api import (
    EpisodeResult,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.agent.rollout_pipeline import RolloutPipeline, _SubmissionGate
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import InferenceResponse, LLMChatMessage, ReturnsRaw, ReturnsTokens


class MockInferenceInterface(ReturnsRaw):
    """Mock raw-text inference interface with configurable per-prompt delays.

    Prompts at index >= stall_after_calls park forever, modeling a suspended
    engine whose set of completable rollouts is exact and scheduling-independent.
    """

    num_slow_calls: int = 0
    stall_after_calls: int | None = None
    active_requests: int = 0
    max_active_requests: int = 0

    async def base_generate(self, request):
        prompt = request.prompt[0].content
        idx = int(prompt.removeprefix("t"))
        if self.stall_after_calls is not None and idx >= self.stall_after_calls:
            await asyncio.Event().wait()
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            if idx < self.num_slow_calls:
                await asyncio.sleep(0.03)
            else:
                await asyncio.sleep(0)
            return InferenceResponse(
                response=LLMChatMessage(role="assistant", content=prompt),
                raw_text=prompt,
                finish_reason="stop",
            )
        finally:
            self.active_requests -= 1


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self._call_count = 0
        self.prepare_group_rollout_calls = 0
        self.get_rollout_response_calls = 0

    async def get_reward_rollouts(self, request):
        raise NotImplementedError

    async def get_rollout_response(self, request, inference_request):
        self.get_rollout_response_calls += 1
        return await request.inference_interface.agenerate(inference_request)

    async def prepare_group_rollout(self, request, env_index: int = 0):
        idx = self._call_count
        self._call_count += 1
        self.prepare_group_rollout_calls += 1

        async def run_episode():
            # Single-turn agent: the episode is one inference on the group's prompt.
            turn_request = request.inference_interface.prepare_request(
                f"t{idx}", request.generation_args
            )
            response = await self.get_rollout_response(request, turn_request)
            return EpisodeResult(
                responses=[response], conversation=[*turn_request.prompt, response.response]
            )

        async def build_rollout(episode):
            responses = episode.responses
            reward = float(responses[-1].response.content.removeprefix("t"))
            return Rollout(
                trajectory=[r.raw_text for r in responses],
                reward=reward,
                env_id=self.env_id,
            )

        return GroupRolloutParams(run_episode=run_episode, build_rollout=build_rollout)


class CountingRewardAgent(RewardOnlyAgent):
    """Minimal RewardOnlyAgent: prompts t0, t1, ... and reward = echoed index."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_id = "reward-test"
        self._prompt_count = 0

    async def get_prompt(self, validation):
        idx = self._prompt_count
        self._prompt_count += 1
        return f"t{idx}", {"idx": idx}

    async def get_reward(self, response, golden, finish_reason):
        return float(int(response.removeprefix("t")) == golden["idx"])


async def _flush(rounds: int = 50):
    """Let pipeline stage tasks settle (mock inference is zero-delay)."""
    for _ in range(rounds):
        await asyncio.sleep(0)


class TestSubmissionGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("submission", ["R", "G", "E", "B"])
    async def test_release_requires_matching_granularity(self, submission):
        gate = _SubmissionGate(capacity=1, submission=submission)
        await gate.acquire_for(submission)
        assert gate.held == 1
        for granularity in ("R", "G", "E", "B"):
            if granularity == submission:
                continue
            gate.release_for(granularity)
        assert gate.held == 1
        assert gate.release_calls == 0
        gate.release_for(submission)
        assert gate.held == 0
        assert gate.release_calls == 1


class TestConsumptionRelease:
    """G-submission gate slots must recycle on trainer consumption, not assembly."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "consumption_granularity, num_groups",
        [
            pytest.param("G", 1, id="group_consumption"),
            pytest.param("E", 1, id="env_consumption"),
            pytest.param("B", 2, id="batch_consumption"),
        ],
    )
    async def test_group_submission_stalls_until_consumption(
        self, consumption_granularity, num_groups
    ):
        # Gate capacity in G-submission slots is parallel_generation_tasks
        # (a depth in batches) x num_groups (groups per batch).
        capacity = 4
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity="G",
            consumption_granularity=consumption_granularity,
        )
        pipeline = RolloutPipeline(
            gen, request, parallel_generation_tasks=capacity // num_groups
        )
        it = pipeline.run()
        try:
            for pulled in range(1, capacity + 3):
                # wait_for turns the deadlock failure mode (a slot never freed)
                # into a test failure instead of a hang.
                await asyncio.wait_for(anext(it), timeout=10)
                await _flush()
                # Each yield frees exactly one group slot on the consumer's next
                # resume, so submission tracks consumption with a one-slot skew
                # (the release for the latest pull hasn't fired yet). On
                # assembly-release semantics this runs away unbounded; if no
                # consume-site release existed, the loop would deadlock at
                # `pulled == capacity + 1`.
                assert gen.prepare_group_rollout_calls == capacity + pulled - 1
        finally:
            await it.aclose()

    @pytest.mark.asyncio
    async def test_batch_submission_releases_once_per_batch(self):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=1)
        it = pipeline.run()
        try:
            await asyncio.wait_for(anext(it), timeout=10)
            await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            gate = pipeline.gate
            # Batch 0 fully yielded but the consumer hasn't come back yet: its
            # single batch slot is still held (a per-group release here would
            # show release_calls == 2 and prepared == 4).
            assert gate.release_calls == 0
            assert gen.prepare_group_rollout_calls == 2
            await asyncio.wait_for(anext(it), timeout=10)
            await _flush()
            assert gate.release_calls == 1
            assert gen.prepare_group_rollout_calls == 4
        finally:
            await it.aclose()


class TestRewardRollouts:
    @pytest.mark.asyncio
    async def test_get_reward_rollouts_matches_per_rollout_composition(self):
        agent = CountingRewardAgent()
        request = RolloutRequest(num_rollouts=4, inference_interface=MockInferenceInterface())
        rollouts = await agent.get_reward_rollouts(request)
        assert len(rollouts) == 4
        assert sorted(r.trajectory[0] for r in rollouts) == ["t0", "t1", "t2", "t3"]
        assert all(r.reward == 1.0 for r in rollouts)
        assert all(r.env_id == "reward-test" for r in rollouts)


class TestGroupedRollouts:
    @pytest.mark.parametrize("field", ["submission_granularity", "consumption_granularity"])
    def test_grouped_rollout_request_rejects_unknown_granularity(self, field):
        request_kwargs = {
            "num_groups": 1,
            "rollouts_per_group": 1,
            "inference_interface": MagicMock(spec=ReturnsRaw),
            field: "X",
        }
        with pytest.raises(ValidationError) as exc_info:
            GroupedRolloutRequest(**request_kwargs)
        assert any(error["loc"] == (field,) for error in exc_info.value.errors())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_groups, submission_granularity, consumption_granularity",
        [
            pytest.param(1, "B", "B", id="num_groups_1_batch"),
            pytest.param(4, "G", "G", id="num_groups_gt_1_group"),
            pytest.param(4, "R", "B", id="num_groups_gt_1_rollout"),
        ],
    )
    async def test_filter_groups_with_same_reward_rejected(
        self, num_groups, submission_granularity, consumption_granularity
    ):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            filter_groups_with_same_reward=True,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        with pytest.raises(AssertionError, match="filter_groups_with_same_reward"):
            RolloutPipeline(gen, request, parallel_generation_tasks=8)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "num_slow_calls, stall_after_calls, num_groups, "
            "submission_granularity, consumption_granularity, expected_count, "
            "expected_batch_ids, expected_trajectories, expects_ready_batch"
        ),
        [
            pytest.param(0, None, 8, "B", "B", 8, None, None, False, id="single_batch"),
            pytest.param(
                0, None, 4, "B", "B", 4, None, None, False, id="fewer_groups_than_parallel"
            ),
            pytest.param(
                4,
                None,
                2,
                "B",
                "B",
                8,
                [0, 0, 1, 1, 2, 2, 3, 3],
                None,
                None,
                id="batched_submission_order",
            ),
            pytest.param(0, None, 1, "G", "B", 10, None, None, None, id="streaming"),
            pytest.param(
                4,
                None,
                1,
                "G",
                "G",
                8,
                None,
                [f"t{i}" for i in range(4, 8)],
                None,
                id="group_consume_completion_order",
            ),
            pytest.param(
                4,
                None,
                1,
                "G",
                "B",
                8,
                list(range(8)),
                [f"t{i}" for i in range(8)],
                None,
                id="batch_consume_submission_order",
            ),
            # 6 completable rollouts, then the engine stalls (as when
            # suspended): whole batches bank without further generation and
            # the can_skip_inference read finds one ready.
            pytest.param(
                0,
                6,
                2,
                "B",
                "B",
                2,
                [0, 0],
                ["t0", "t1"],
                True,
                id="stalled_engine_banks_ready_batches",
            ),
        ],
    )
    async def test_grouped_rollout_generation(
        self,
        num_slow_calls,
        stall_after_calls,
        num_groups,
        submission_granularity,
        consumption_granularity,
        expected_count,
        expected_batch_ids,
        expected_trajectories,
        expects_ready_batch,
    ):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(
                num_slow_calls=num_slow_calls, stall_after_calls=stall_after_calls
            ),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )

        groups = []
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=8)
        # Hold the iterator open through the assertions: abandoning it lets the
        # event loop finalize run(), cancelling the stages that bank batches.
        async with aclosing(pipeline.run()) as iterator:
            async for group in iterator:
                groups.append(group)
                if len(groups) >= expected_count:
                    break

            assert len(groups) == expected_count
            if expected_batch_ids is not None:
                assert [g.batch_id for g in groups] == expected_batch_ids
            if expected_trajectories is not None:
                trajectories = [group[0].trajectory[0] for group in groups]
                assert trajectories[: len(expected_trajectories)] == expected_trajectories
            if expects_ready_batch is not None:
                if expects_ready_batch:
                    for _ in range(2 * request.num_groups + 16):
                        await asyncio.sleep(0)
                    assert pipeline.ready_batches >= 1
                else:
                    assert pipeline.ready_batches == 0
            assert pipeline.yielded_count == len(groups)
            assert len(pipeline.output_queue_dwell) == len(groups)

    @pytest.mark.asyncio
    async def test_rollout_submission_granularity_limits_inference_concurrency(self):
        # parallel_generation_tasks is a depth in batches; the R gate admits at
        # most depth x (num_groups x rollouts_per_group) rollouts at once.
        parallel_generation_tasks = 1
        gen = MockGenerator()
        inference_interface = MockInferenceInterface(num_slow_calls=100)
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=2,
            inference_interface=inference_interface,
            submission_granularity="R",
            consumption_granularity="B",
        )

        groups = []
        pipeline = RolloutPipeline(
            gen, request, parallel_generation_tasks=parallel_generation_tasks
        )
        async for group in pipeline.run():
            groups.append(group)
            if len(groups) >= 4:
                break

        assert all(len(group) == 2 for group in groups)
        assert inference_interface.max_active_requests <= (
            parallel_generation_tasks * request.num_groups * request.rollouts_per_group
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity",
        [
            pytest.param("B", "B", id="batch_batch"),
            pytest.param("G", "G", id="group_group"),
            pytest.param("E", "E", id="env_env"),
            pytest.param("G", "E", id="group_env"),
        ],
    )
    async def test_weighted_multi_task(self, submission_granularity, consumption_granularity):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=1)
        gen = pipeline.run()
        groups = [await anext(gen) for _ in range(8)]

        # Weights 3:1 → env "a" owns 3 of every 4 batch slots; the single
        # pipeline routes preparation and generation to the owning sub-agent.
        env_ids = [g[0].env_id for g in groups]
        assert sorted(env_ids) == ["a"] * 6 + ["b"] * 2
        assert mt.latest_distribution["agent_groups"] == [3, 1]
        if consumption_granularity in ("B", "E"):
            # Batch-order and balanced-env consumption keep every 4-window at the exact env mix.
            for start in (0, 4):
                assert sorted(env_ids[start : start + 4]) == ["a", "a", "a", "b"]
        if consumption_granularity == "B":
            # With depth-1 gating and consumed-release, nothing is buffered or in flight.
            assert pipeline.prepared_count == (pipeline.yielded_count * request.rollouts_per_group)
            assert pipeline.infer_queue.qsize() == 0
            assert pipeline.assemble_queue.qsize() == 0
            assert pipeline.output_queue.qsize() == 0
            assert not pipeline._assemble_pending
            assert not pipeline._consume_pending

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_slow_calls, stall_after_calls, collect_count, expects_ready_batch",
        [
            pytest.param(2, None, 12, None, id="balanced_windows"),
            # Both envs complete prompts t0-t5: env "a" (3 groups/batch) has
            # two complete units, env "b" six, so balanced rounds bank and
            # the can_skip_inference read finds one ready.
            pytest.param(0, 6, 4, True, id="stalled_engine_banks_complete_rounds"),
        ],
    )
    async def test_env_consumption_balances_each_batch(
        self, num_slow_calls, stall_after_calls, collect_count, expects_ready_batch
    ):
        """Balanced-E: every trainer-batch window holds each env's exact share,
        and a banked batch is one complete round."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(
                num_slow_calls=num_slow_calls, stall_after_calls=stall_after_calls
            ),
            submission_granularity="E",
            consumption_granularity="E",
        )
        groups = []
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=2)
        async with aclosing(pipeline.run()) as iterator:
            async for group in iterator:
                groups.append(group)
                if len(groups) >= collect_count:
                    break

            for start in range(0, collect_count, 4):
                env_ids = [g[0].env_id for g in groups[start : start + 4]]
                assert sorted(env_ids) == ["a", "a", "a", "b"]
            if expects_ready_batch:
                for _ in range(2 * request.num_groups + 16):
                    await asyncio.sleep(0)
                assert pipeline.ready_batches >= 1

    @pytest.mark.asyncio
    async def test_lag0_streaming_matches_non_streaming_boundaries(self):
        """lag=0 (B/B, depth-1 gate): each iteration of the persistent stream is exactly
        one batch, generated entirely after the previous boundary — the old
        non-streaming per-iteration contract, enforced by assert_no_inflight_rollouts."""
        from megatron.rl.rl_utils import assert_no_inflight_rollouts

        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=1.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)
        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=2,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(mt, request, parallel_generation_tasks=1)
        gen = pipeline.run()
        for iteration in range(3):
            groups = [await anext(gen) for _ in range(4)]
            # Exactly this iteration's batch, whole and in order.
            assert [g.batch_id for g in groups] == [iteration] * 4
            # Nothing of the next batch has even been prepared: everything the
            # next iteration consumes is generated after this boundary.
            assert sum(a.prepare_group_rollout_calls for a in mt.agents) == (iteration + 1) * 4
            assert_no_inflight_rollouts(pipeline)

    @pytest.mark.asyncio
    async def test_assert_no_inflight_rollouts_detects_run_ahead(self):
        """With lag>0 the gate legitimately runs ahead; the boundary checker must fire."""
        from megatron.rl.rl_utils import assert_no_inflight_rollouts

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity="B",
            consumption_granularity="B",
        )
        pipeline = RolloutPipeline(MockGenerator(), request, parallel_generation_tasks=2)
        gen = pipeline.run()
        [await anext(gen) for _ in range(4)]
        with pytest.raises(AssertionError, match="Non-streaming RL"):
            assert_no_inflight_rollouts(pipeline)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity",
        [
            pytest.param("B", "G", id="batch_group"),
            pytest.param("B", "E", id="batch_env"),
            pytest.param("E", "G", id="env_group"),
        ],
    )
    async def test_consumption_finer_than_submission_rejected(
        self, submission_granularity, consumption_granularity
    ):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        with pytest.raises(AssertionError, match="no finer"):
            RolloutPipeline(gen, request, parallel_generation_tasks=1)

    @pytest.mark.parametrize(
        "weights, num_groups, expected_layout, warns",
        [
            # 8 groups cannot realize 1:2 exactly; quantized with a warning.
            pytest.param([1.0, 2.0], 8, [3, 5], True, id="quantized"),
            # A weight below 1/num_groups keeps one group per batch.
            pytest.param([0.01, 0.99], 8, [1, 7], True, id="zero_share_rounded_up"),
            pytest.param([3.0, 1.0], 8, [6, 2], False, id="exact"),
            pytest.param([1.0, 1.0, 1.0], 3, [1, 1, 1], False, id="one_group_each"),
            # Only an env count exceeding the batch size is infeasible.
            pytest.param([1.0, 1.0, 1.0], 2, None, False, id="too_many_envs"),
        ],
    )
    def test_multi_env_layout(self, caplog, weights, num_groups, expected_layout, warns):
        """Weights quantize to a constant split (warned); eval-only envs take no slot."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": f"e{i}"}, weight=w)
            for i, w in enumerate(weights)
        ] + [
            AgentConfig(
                agent_type=MockGenerator,
                agent_args={"env_id": "eval"},
                weight=1.0,
                evaluation_only=True,
            )
        ]
        mt = WeightedMultiTask(configs)
        if expected_layout is None:
            with pytest.raises(ValueError, match="cannot fit"):
                mt.rollout_group_layout(num_groups)
            return
        # The split is identical on every call.
        assert [mt.rollout_group_layout(num_groups) for _ in range(3)] == [expected_layout] * 3
        assert warns == any("weights changed" in message for message in caplog.messages)


def make_response(prompt_length, total_len, content="resp", finish_reason="stop"):
    return InferenceResponse(
        response=LLMChatMessage(role="assistant", content=content),
        raw_text=content,
        token_ids=list(range(total_len)),
        prompt_length=prompt_length,
        logprobs=[0.0] * (total_len - prompt_length),
        finish_reason=finish_reason,
    )


# Conversation length -> response spec: length 1 is the first turn (the bare prompt), length 3
# the second (assistant reply + observation appended).
TWO_TURN_SCRIPT = {
    1: dict(prompt_length=3, total_len=7, content="a0"),
    3: dict(prompt_length=6, total_len=11, content="a1"),
}

# Both two-turn termination modes (env-signaled done, max_turns exhausted) must produce this
# identical episode; only the env-consultation trace (observation_turns) differs per case.
TWO_TURN_EXPECTED = dict(
    seen_roles=[["user"], ["user", "assistant", "user"]],
    reward_conv=[("user", "hello"), ("assistant", "a0"), ("user", "obs0"), ("assistant", "a1")],
    rewarded=[("a1", "stop")],
    genmask_sums=[4, 5],
)


class ScriptedInterface(ReturnsTokens, ReturnsRaw):
    """Inference stub whose reply is a pure function of the request: the conversation length
    maps to a response spec, so it stays deterministic under pipeline concurrency."""

    by_prompt_length: dict = Field(default_factory=dict)
    seen_conversations: list = Field(default_factory=list)

    async def agenerate(self, request):
        self.seen_conversations.append(list(request.prompt))
        return make_response(**self.by_prompt_length[len(request.prompt)])


class EpisodeAgent(RewardOnlyAgent):
    """Configurable multi-turn agent.

    `done_at_turn` controls when get_observation signals done: at every turn >= done_at_turn
    it returns (None, True); None means it never signals done, so the episode ends only by
    exhausting max_turns. Records get_reward calls and the conversation get_trajectory_reward saw.
    """

    env_id: str = "test"
    max_turns: int = 1
    done_at_turn: int | None = None
    rewarded: list = Field(default_factory=list)
    reward_conversation: list = Field(default_factory=list)
    observation_turns: list = Field(default_factory=list)

    async def get_prompt(self, validation):
        return "hello", {"problem_id": "p0"}

    async def get_observation(self, turn_idx, response, conversation, golden):
        self.observation_turns.append(turn_idx)
        if self.done_at_turn is not None and turn_idx >= self.done_at_turn:
            return None, True
        return f"obs{turn_idx}", False

    async def get_reward(self, response, golden, finish_reason):
        self.rewarded.append((response, finish_reason))
        return 1.5

    async def get_trajectory_reward(self, responses, conversation, golden):
        self.reward_conversation.extend(conversation)
        return await super().get_trajectory_reward(responses, conversation, golden)


class TestMultiTurnEpisode:

    @pytest.mark.parametrize("driver", ["reward_rollouts", "pipeline"])
    @pytest.mark.parametrize(
        "max_turns, done_at_turn, scripted, expected",
        [
            # Single turn: get_observation is never consulted (no continuation is possible).
            pytest.param(
                1,
                None,
                {1: dict(prompt_length=2, total_len=6, content="only")},
                dict(
                    seen_roles=[["user"]],
                    reward_conv=[("user", "hello"), ("assistant", "only")],
                    rewarded=[("only", "stop")],
                    genmask_sums=[4],
                    observation_turns=[],
                ),
                id="single_turn",
            ),
            # Multi-turn ended by the environment: turn 0 yields an observation, turn 1 is done.
            pytest.param(
                3,
                1,
                TWO_TURN_SCRIPT,
                dict(TWO_TURN_EXPECTED, observation_turns=[0, 1]),
                id="multi_turn_env_done",
            ),
            # Ended by exhausting max_turns instead (env never signals done): the same episode,
            # except get_observation must not run for the final allowed turn.
            pytest.param(
                2,
                None,
                TWO_TURN_SCRIPT,
                dict(TWO_TURN_EXPECTED, observation_turns=[0]),
                id="multi_turn_max_turns_exhausted",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_episode(self, driver, max_turns, done_at_turn, scripted, expected):
        """Episodes grow the conversation each turn and collapse into one per-turn rollout,
        identically through get_reward_rollouts and through the real _RolloutPipeline
        (get_grouped_rollouts) -- the latter proving run_episode runs in the infer stage."""
        iface = ScriptedInterface(by_prompt_length=scripted)
        agent = EpisodeAgent(max_turns=max_turns, done_at_turn=done_at_turn)

        if driver == "reward_rollouts":
            rollouts = await agent.get_reward_rollouts(
                RolloutRequest(num_rollouts=1, inference_interface=iface)
            )
        else:
            groups = []

            async def _drain():
                async for group in agent.get_grouped_rollouts(
                    GroupedRolloutRequest(
                        num_groups=1, rollouts_per_group=1, inference_interface=iface
                    )
                ):
                    groups.append(group)

            # Bounded so a wedged pipeline fails fast instead of hanging.
            await asyncio.wait_for(_drain(), timeout=5.0)
            (group,) = groups
            rollouts = group.rollouts
        (rollout,) = rollouts

        assert isinstance(rollout, TokenRollout)
        assert rollout.reward == 1.5
        assert rollout.problem_id == "p0"
        # One trajectory entry per generated turn.
        assert len(rollout.trajectory) == len(expected["genmask_sums"])
        # Each turn's inference request = prior conversation (reply + observation appended).
        assert [[m.role for m in conv] for conv in iface.seen_conversations] == expected[
            "seen_roles"
        ]
        # Default trajectory reward scores only the final response.
        assert agent.rewarded == expected["rewarded"]
        # Per-turn generation masks cover exactly each turn's generated tokens.
        assert [sum(mask) for mask in rollout.generation_mask] == expected["genmask_sums"]
        # get_observation is consulted only when another generation is still possible -- never on
        # the final allowed turn.
        assert agent.observation_turns == expected["observation_turns"]
        # get_trajectory_reward sees the full dialogue, ending on the final reply exactly once.
        assert [(m.role, m.content) for m in agent.reward_conversation] == expected["reward_conv"]
