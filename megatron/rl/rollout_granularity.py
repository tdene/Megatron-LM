# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""RL rollout submission and consumption granularity values."""

from typing import Literal

SubmissionGranularity = Literal["R", "G", "E", "B"]
ConsumptionGranularity = Literal["G", "E", "B"]

# Coarseness order of the granularity ladder (rollout < group < env < batch).
# Consumption must be no finer than submission.
GRANULARITY_RANK: dict[str, int] = {"R": 0, "G": 1, "E": 2, "B": 3}


def resolve_rl_generation_lag(args, dp_size: int, max_requests: int) -> None:
    """Resolve args.rl_generation_lag against the inference engine's request capacity.

    Autotunes the lag to fill the engine when unset; otherwise reports how the requested
    lag compares to the maximum effective lag the engine can serve. The lag may be
    fractional or negative (>= -1); the RolloutPipeline gate rounds the implied slot
    count and clamps it to at least one submission unit.
    """
    # Import here to avoid circular imports.
    from megatron.training.utils import print_rank_0

    G = args.grpo_group_size
    P = args.grpo_prompts_per_step
    max_effective_groups = max(1, dp_size * max_requests // G)
    max_effective_lag = max_effective_groups / P - 1
    if args.rl_generation_lag is None:
        args.rl_generation_lag = max_effective_lag
        print_rank_0(
            f"Autotuned rl-generation-lag={max_effective_lag:.2f} "
            f"({max_effective_groups} groups in flight at "
            f"submission granularity {args.rl_submission_granularity}; "
            f"DP={dp_size}, max_requests={max_requests}, G={G}, P={P}).")
    else:
        groups_in_flight = (args.rl_generation_lag + 1) * P
        print_rank_0(
            f"Using rl-generation-lag={args.rl_generation_lag} "
            f"(~{groups_in_flight:.1f} groups in flight at "
            f"submission granularity {args.rl_submission_granularity}; "
            f"max effective lag={max_effective_lag:.2f}; "
            f"DP={dp_size}, max_requests={max_requests}, G={G}, P={P}).")
        if groups_in_flight > max_effective_groups:
            print_rank_0(
                f"WARNING: --rl-generation-lag {args.rl_generation_lag} oversubscribes the "
                f"inference engine (max effective lag is {max_effective_lag:.2f}). "
                f"Additional lag beyond that point has no benefit.")
    if max_effective_lag < 0:
        print_rank_0(
            f"WARNING: max effective lag is {max_effective_lag:.2f} (negative) — the "
            f"inference engine cannot hold even one training step's worth of rollouts "
            f"({max_effective_groups} groups < P={P}). Even fully-synchronous GRPO would "
            f"oversubscribe. Consider scaling up inference resources.")
