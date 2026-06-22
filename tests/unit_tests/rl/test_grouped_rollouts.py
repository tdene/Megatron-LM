# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    Rollout,
    RolloutGenerator,
)
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import ReturnsRaw


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", num_slow_calls=0, uniform_reward_calls=0, **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self.num_slow_calls = num_slow_calls
        self.uniform_reward_calls = uniform_reward_calls
        self._call_count = 0
        self.prepare_group_calls = 0
        self.generate_rollout_calls = 0

    async def rollout(self, request):
        raise NotImplementedError

    def _make_rollout(self, idx, reward=None):
        return Rollout(
            trajectory=[f"t{idx}"],
            reward=float(idx if reward is None else reward),
            env_id=self.env_id,
            policy_epoch=[[(0, 0)]],
            kv_cache_epoch=[[(0, 0)]],
            num_evictions=[0],
        )

    async def prepare_group(self, request):
        self.prepare_group_calls += 1
        return None

    async def generate_rollout(self, request, prepared):
        idx = self._call_count
        self._call_count += 1
        self.generate_rollout_calls += 1
        if idx < self.num_slow_calls:
            await asyncio.sleep(0.03)
        # The first `uniform_reward_calls` rollouts share a reward, so the opening group is dropped
        # under filter_groups_with_same_reward; later rollouts vary by index and pass the filter.
        reward = 1.0 if idx < self.uniform_reward_calls else None
        return self._make_rollout(idx, reward=reward)


class TestGroupedRollouts:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "consumption_granularity, runs_ahead",
        [
            pytest.param("B", False, id="batch_consume_bounds_runahead"),
            pytest.param("G", True, id="group_consume_runs_ahead"),
        ],
    )
    async def test_consumption_granularity_shapes_lag(self, consumption_granularity, runs_ahead):
        """Batch consumption releases the gate on consume, capping run-ahead at the lag window;
        group consumption releases on finish, so generation races ahead of a paused trainer."""
        hold = asyncio.Event()  # hard cap so the finish gate cannot generate forever

        class CappedGenerator(MockGenerator):
            async def generate_rollout(self, request, prepared):
                if self.generate_rollout_calls >= 16:
                    await hold.wait()
                return await super().generate_rollout(request, prepared)

        gen = CappedGenerator()
        request = GroupedRolloutRequest(
            groups_per_batch=2,
            rollouts_per_group=2,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=True,
            oversubscription_factor=1,  # lag=0 -> one batch (4 rollouts) in flight
            submission_granularity="G",
            consumption_granularity=consumption_granularity,
        )
        agen = gen.get_grouped_rollouts(request)
        try:
            consumed = [await agen.__anext__() for _ in range(2)]  # consume one batch's worth
            assert len(consumed) == 2
            for _ in range(200):  # let the pipeline run while the trainer is paused
                await asyncio.sleep(0)
            if runs_ahead:
                assert gen.generate_rollout_calls >= 12  # raced past the lag window to the cap
            else:
                assert gen.generate_rollout_calls <= 8  # bounded to ~lag+1 batches
        finally:
            hold.set()
            await agen.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "num_slow_calls, streaming, groups_per_batch, rollouts_per_group, "
            "oversubscription_factor, submission_granularity, consumption_granularity, "
            "filter_same_reward, expected_count, expected_batch_ids, expected_trajectories"
        ),
        [
            pytest.param(0, False, 8, 1, 1, "B", "B", False, 8, None, None, id="non_batched"),
            pytest.param(
                0, False, 4, 1, 1, "B", "B", False, 4, None, None,
                id="non_streaming_fewer_than_parallel",
            ),
            pytest.param(
                4, True, 2, 1, 4, "B", "B", False, 8,
                [0, 0, 1, 1, 2, 2, 3, 3], None,
                id="batched_submission_order",
            ),
            pytest.param(0, True, 1, 1, 8, "G", "B", False, 10, None, None, id="streaming"),
            pytest.param(
                4, True, 1, 1, 8, "G", "G", False, 8,
                None, [f"t{i}" for i in range(4, 8)],
                id="group_consume_completion_order",
            ),
            pytest.param(
                4, True, 1, 1, 8, "G", "B", False, 8,
                list(range(8)), [f"t{i}" for i in range(8)],
                id="batch_consume_submission_order",
            ),
            pytest.param(
                0, True, 1, 2, 1, "R", "B", False, 1, None, None,
                id="rollout_submission_assembles_group",
            ),
            pytest.param(
                0, True, 1, 2, 1, "R", "B", True, 1, None, None,
                id="rollout_filter_regenerates_slot",
            ),
            pytest.param(
                0, True, 1, 2, 1, "G", "B", True, 1, None, None,
                id="group_filter_regenerates_slot",
            ),
        ],
    )
    async def test_get_grouped_rollouts(
        self,
        num_slow_calls,
        streaming,
        groups_per_batch,
        rollouts_per_group,
        oversubscription_factor,
        submission_granularity,
        consumption_granularity,
        filter_same_reward,
        expected_count,
        expected_batch_ids,
        expected_trajectories,
    ):
        # When filtering, seed the opening group with a shared reward so it is dropped and the slot
        # regenerated, exercising the regen path end-to-end (R/G submission, consume-release).
        gen = MockGenerator(
            num_slow_calls=num_slow_calls,
            uniform_reward_calls=rollouts_per_group if filter_same_reward else 0,
        )
        request = GroupedRolloutRequest(
            groups_per_batch=groups_per_batch,
            rollouts_per_group=rollouts_per_group,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            oversubscription_factor=oversubscription_factor,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
            filter_groups_with_same_reward=filter_same_reward,
        )

        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        assert all(len(group) == rollouts_per_group for group in groups)
        assert gen.prepare_group_calls >= 1
        assert gen.generate_rollout_calls >= expected_count * rollouts_per_group
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids
        if expected_trajectories is not None:
            trajectories = [group[0].trajectory[0] for group in groups]
            assert trajectories[: len(expected_trajectories)] == expected_trajectories
        if filter_same_reward:
            # Delivered groups are varied; the dropped degenerate group forced a regeneration.
            assert all(len({r.reward for r in group}) > 1 for group in groups)
            assert gen.generate_rollout_calls >= (expected_count + 1) * rollouts_per_group

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission_granularity, consumption_granularity",
        [
            pytest.param("B", "B", id="batch_submission"),
            pytest.param("G", "G", id="group_submission"),
        ],
    )
    async def test_weighted_multi_task(self, submission_granularity, consumption_granularity):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=3.0),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=1.0),
        ]
        mt = WeightedMultiTask(configs)

        captured = []
        for agent in mt.agents:
            original = agent.get_grouped_rollouts

            async def spy(req, orig=original):
                captured.append(req)
                async for group in orig(req):
                    yield group

            agent.get_grouped_rollouts = spy

        request = GroupedRolloutRequest(
            groups_per_batch=4,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=False,
            submission_granularity=submission_granularity,
            consumption_granularity=consumption_granularity,
        )
        groups = []
        async for group in mt.get_grouped_rollouts(request):
            groups.append(group)

        assert len(groups) == 4
        # Weights 3:1 → agent "a" produces 3 groups, agent "b" produces 1.
        env_ids = [g[0].env_id for g in groups]
        assert sorted(env_ids) == ["a", "a", "a", "b"]
        for sub_req in captured:
            assert sub_req.groups_per_batch in (1, 3)  # distributed proportionally by weight
            assert sub_req.streaming == request.streaming
            assert sub_req.submission_granularity == request.submission_granularity
            assert sub_req.consumption_granularity == request.consumption_granularity
