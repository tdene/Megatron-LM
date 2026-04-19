# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock

import pytest

from megatron.rl.agent.api import (
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    Rollout,
    RolloutGenerator,
    RolloutGroup,
)
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference import ReturnsRaw


class MockGenerator(RolloutGenerator, GroupedRolloutGenerator):
    """Mock generator with configurable per-call delays."""

    def __init__(self, env_id="test", num_slow_calls=0, **kwargs):
        super().__init__(**kwargs)
        self.env_id = env_id
        self.num_slow_calls = num_slow_calls
        self._call_count = 0

    async def rollout(self, request):
        raise NotImplementedError

    async def group_rollout(self, request):
        idx = self._call_count
        self._call_count += 1
        if idx < self.num_slow_calls:
            await asyncio.sleep(0.03)
        return [
            Rollout(
                trajectory=[f"t{idx}"],
                reward=float(idx),
                env_id=self.env_id,
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
            for _ in range(request.rollouts_per_group)
        ]


class TestGroupedRollouts:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_slow_calls, streaming, num_groups, expected_count, expected_batch_ids",
        [
            pytest.param(0, False, 8, 8, None, id="non_batched"),
            pytest.param(0, False, 4, 4, None, id="non_streaming_fewer_than_parallel"),
            pytest.param(4, True, 2, 8, [0, 0, 1, 1, 2, 2, 3, 3], id="batched_submission_order"),
            pytest.param(0, True, 1, 10, None, id="streaming"),
        ],
    )
    async def test_get_grouped_rollouts(
        self, num_slow_calls, streaming, num_groups, expected_count, expected_batch_ids
    ):
        gen = MockGenerator(parallel_generation_tasks=8, num_slow_calls=num_slow_calls)
        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            enforce_order=num_groups > 1,
        )
        groups = []
        async for group in gen.get_grouped_rollouts(request):
            groups.append(group)
            if request.streaming and len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        if expected_batch_ids is not None:
            assert [g.batch_id for g in groups] == expected_batch_ids

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agents, num_groups, pgt, streaming, enforce_order, expected_count, expected_env_ids",
        [
            pytest.param(
                [("a", 0.1), ("b", 0.2), ("c", 0.7)],
                10,
                100,
                True,
                True,
                10,
                ["a", "b", "b", "c", "c", "c", "c", "c", "c", "c"],
                id="inexact_float_weights",
            ),
            pytest.param(
                [("a", 3), ("b", 1)],
                4,
                100,
                False,
                True,
                4,
                ["a", "a", "a", "b"],
                id="fixed_unequal_w",
            ),
            pytest.param(
                [("a", 1.0), ("b", 1.0)],
                2,
                100,
                True,
                True,
                2,
                ["a", "b"],
                id="divide_by_zero_regression",
            ),
        ],
    )
    async def test_weighted_multi_task(
        self, agents, num_groups, pgt, streaming, enforce_order, expected_count, expected_env_ids
    ):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": env_id}, weight=weight)
            for env_id, weight in agents
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = pgt

        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=streaming,
            enforce_order=enforce_order,
        )
        groups = []
        async for group in mt.get_grouped_rollouts(request):
            groups.append(group)
            if len(groups) >= expected_count:
                break

        assert len(groups) == expected_count
        env_ids = sorted(g[0].env_id for g in groups)
        assert env_ids == sorted(expected_env_ids)

    @pytest.mark.asyncio
    async def test_weighted_multi_task_multi_step(self):
        """Verify steps from a persistent generator have correct aggregate composition."""
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "a"}, weight=1),
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": "b"}, weight=2),
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = 100

        request = GroupedRolloutRequest(
            num_groups=4,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            streaming=True,
            enforce_order=True,
        )

        gen = mt.get_grouped_rollouts(request)
        step_size = 4
        prev_max_per_agent = {}
        all_env_ids = []

        for step in range(3):
            groups = [await anext(gen) for _ in range(step_size)]

            # Per-agent rewards must advance beyond the previous step.
            # MockGenerator increments _call_count per call, so reward is monotonic
            # per agent. A stale group would have a reward <= prev step's max.
            current_per_agent = {}
            for g in groups:
                current_per_agent.setdefault(g[0].env_id, []).append(g[0].reward)
                all_env_ids.append(g[0].env_id)

            for env_id, rewards in current_per_agent.items():
                if env_id in prev_max_per_agent:
                    assert (
                        min(rewards) > prev_max_per_agent[env_id]
                    ), f"Step {step}, agent {env_id}: possible cross-step leakage"
                prev_max_per_agent[env_id] = max(rewards)

        # 12 groups = 3 batches of [1a, 3b] (weights 1:2 rounded into batch of 4).
        assert sorted(all_env_ids) == ["a"] * 3 + ["b"] * 9

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agents, num_groups, pgt, match",
        [
            pytest.param(
                [("a", 0.01), ("b", 0.01), ("c", 0.98)],
                3,
                100,
                "would receive 0",
                id="extreme_weights",
            ),
        ],
    )
    async def test_weighted_multi_task_error(self, agents, num_groups, pgt, match):
        configs = [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": env_id}, weight=weight)
            for env_id, weight in agents
        ]
        mt = WeightedMultiTask(configs)
        mt.parallel_generation_tasks = pgt

        request = GroupedRolloutRequest(
            num_groups=num_groups,
            rollouts_per_group=1,
            inference_interface=MagicMock(spec=ReturnsRaw),
            enforce_order=True,
        )
        with pytest.raises(ValueError, match=match):
            async for _ in mt.get_grouped_rollouts(request):
                pass

