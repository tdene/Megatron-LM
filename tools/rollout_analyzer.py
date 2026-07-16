#!/usr/bin/env python3
"""Offline rollout analyzer for the Megatron RL pipeline.

Loads rollout JSON files collected via --skip-train --perform-rl-step (with
--langrl-log-dir set), estimates per-level tail factors from the actual rollout
length distribution, and predicts utilization and first-token staleness for all
granularity modes (B/B, G/G, G/B, R/B, R/G; plus G/E, R/E, E/E, E/B for
multi-environment runs).

Also recommends matched capacity weights for the E rung (C_i ∝ μ_i · P_i) and
shows the utilization cost of the default length-blind allocation.

This is a thin CLI: the tail-factor math and mode predictions live in
``megatron.rl.rollout_capacity`` (shared with the live OnlineCapacityController);
this file only loads JSON, calls those functions, and prints.

Theory background
-----------------
Each level of the rollout hierarchy has a "tail factor" -- the ratio of the
expected maximum of its children's completions to the expected child completion:

    λ_RG = E[max of K rollout lengths] / E[rollout length]    (within a group)
    λ_GE = E[max of P_i group maxima]  / E[group max]         (within an env)
    λ_EB = E[max_i env_max_i]          / wmean_i E[env_max_i] (across envs)
    λ_GB = λ_GE · λ_EB                                        (single-env split)

Cumulative tails (product along the hierarchy):
    τ_K = λ_RG          (group rung)
    τ_E = τ_K · λ_GE   (environment rung; single-env: τ_E = τ_N)
    τ_N = τ_E · λ_EB   (batch rung)

Utilization and first-token staleness (suspend regime, engine sized to gate):
    U    = 1 / τ_submit
    D    = (τ_consume / τ_submit) · (1 + L)     [B/B: D = L exactly]

Usage
-----
# Step 1 -- collect data on the training cluster (no optimizer needed):
#   torchrun ... pretrain_gpt.py \\
#       --skip-train --perform-rl-step --no-load-optim \\
#       --train-iters 40 --langrl-log-dir /path/to/logs [other RL args ...]

# Step 2 -- analyze offline (no GPU; imports the shared core from
#           megatron.rl.rollout_capacity, so run from the repo root):
#   python3 tools/rollout_analyzer.py --rollout-dir /path/to/logs --lag 2
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from megatron.rl.rollout_capacity import (
    MULTI_ENV_MODES,
    SINGLE_ENV_MODES,
    capacity_recommendation,
    compute_levels,
    predict_mode,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _rollout_length(rollout: dict) -> int:
    """Total token (or character) length of a rollout across all turns."""
    traj = rollout.get("trajectory", [])
    if not traj:
        return 0
    first = traj[0]
    if isinstance(first, list):
        # TokenRollout: trajectory is list[list[int]]
        return sum(len(t) for t in traj)
    else:
        # Rollout: trajectory is list[str]
        return sum(len(t) for t in traj)


def load_batches(rollout_dir: str, rank: int = 0) -> list[list[dict]]:
    """Load rollout JSON files and return a list of batches.

    Each batch is a list of group dicts:
        {"env_id": str, "lengths": [int, ...]}   # one entry per rollout in the group

    Files are matched on rank and sorted by iteration number so batches are in
    the order they were collected.
    """
    pattern = os.path.join(rollout_dir, f"rollouts_rank{rank}_iteration*_*.json")
    files = sorted(glob.glob(pattern), key=lambda p: _iteration_of(p))
    if not files:
        # Fallback: any JSON file in the directory
        files = sorted(glob.glob(os.path.join(rollout_dir, "*.json")))
    if not files:
        raise FileNotFoundError(
            f"No rollout JSON files found in {rollout_dir!r}.\n"
            "Make sure you ran with --langrl-log-dir pointing to that directory."
        )

    batches = []
    for path in files:
        try:
            with open(path) as f:
                raw = json.load(f)  # [[rollout_dict, ...], ...]  (groups x rollouts)
        except Exception as e:
            print(f"[warn] skipping {path}: {e}", file=sys.stderr)
            continue
        batch = []
        for group in raw:
            if not group:
                continue
            env_id = group[0].get("env_id", "")
            lengths = [_rollout_length(r) for r in group]
            batch.append({"env_id": env_id, "lengths": lengths})
        if batch:
            batches.append(batch)
    return batches


def _iteration_of(path: str) -> int:
    stem = Path(path).stem  # rollouts_rank0_iteration7_math
    for part in stem.split("_"):
        if part.startswith("iteration") and part[9:].isdigit():
            return int(part[9:])
    return 0


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def print_report(lv: dict, lag: int, total_slots: int | None = None) -> None:
    P, K, V, N = lv["P"], lv["K"], lv["V"], lv["N"]
    mu = lv["mu"]
    n_batches = lv["n_batches"]

    print()
    print("=" * 72)
    print(f"  Rollout length profile   "
          f"({n_batches} batches · P={P} groups · K={K} rollouts/group · V={V} env)")
    print("=" * 72)
    print(f"  overall mean length : {mu:.1f}")
    print(f"  lag L               : {lag}   (gate holds {lag+1} batches in flight)")
    print()

    if V > 1:
        print("  Per-environment breakdown:")
        hdr = f"  {'env':<24}  {'P_i':>4}  {'μ_i':>8}  {'std':>7}  {'cv':>5}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for eid in lv["env_ids"]:
            e = lv["envs"][eid]
            print(f"  {eid:<24}  {e['P_i']:>4}  {e['mu_i']:>8.1f}  "
                  f"{e['std_i']:>7.1f}  {e['cv_i']:>5.2f}")
        print()

    print("  Per-level tail factors  (empirical; each estimated at its own level):")
    print(f"    λ_RG = τ_K  = {lv['lambda_RG']:.3f}   rollout lengths within a group")
    if V > 1:
        print(f"    λ_GE        = {lv['lambda_GE']:.3f}   group completions within an environment")
        print(f"    λ_EB        = {lv['lambda_EB']:.3f}   environment completions within a batch")
    print(f"    λ_GB        = {lv['lambda_GB']:.3f}   group completions within a batch  "
          f"{'(= λ_GE · λ_EB)' if V > 1 else ''}")
    print(f"    τ_K         = {lv['tau_K']:.3f}")
    if V > 1:
        print(f"    τ_E         = {lv['tau_E']:.3f}   (τ_K · λ_GE)")
    print(f"    τ_N         = {lv['tau_N']:.3f}   (τ_K · λ_GB)")
    print()

    # ---- mode table --------------------------------------------------------
    modes = SINGLE_ENV_MODES + (MULTI_ENV_MODES if V > 1 else [])
    # Remove E-modes when V == 1 (they reduce to B)
    if V == 1:
        modes = [m for m in modes if "E" not in m]

    col_w = [8, 10, 18, 18, 12]
    header = (
        f"  {'mode':<{col_w[0]}}  "
        f"{'util':>{col_w[1]}}  "
        f"{'first-staleness':>{col_w[2]}}  "
        f"{'staleness-infl':>{col_w[3]}}  "
        f"{'within-span':>{col_w[4]}}"
    )
    print("  Granularity mode analysis   (suspend regime, engine sized S = (1+L)·N):")
    print(header)
    print("  " + "-" * (sum(col_w) + 4 * 2))

    prev_sub = None
    for sub, con in modes:
        if prev_sub is not None and sub != prev_sub:
            print()
        prev_sub = sub

        # skip invalid combinations
        if sub == "B" and con != "B":
            continue
        if sub == "E" and con == "G":
            continue
        if sub == "G" and con == "R":
            continue

        p = predict_mode(sub, con, lv, lag)
        tag = ""
        if sub == con:
            tag = "← diagonal (matched)"
        if (sub, con) == ("R", "R"):
            tag = "← GRPO-unrealizable reference"
        if (sub, con) == ("B", "B"):
            tag = f"← phase-locked; staleness exactly {lag}"

        util_str = f"{p['util']*100:.0f}%"
        first_str = f"{p['first']:.2f}"
        infl_str = f"{p['inflation']:.3f}"
        span_str = f"{p['span']:.2f}"
        mode_str = p['mode']
        if (sub, con) == ("R", "R"):
            mode_str = "R/R†"

        print(
            f"  {mode_str:<{col_w[0]}}  "
            f"{util_str:>{col_w[1]}}  "
            f"{first_str:>{col_w[2]}}  "
            f"{infl_str:>{col_w[3]}}  "
            f"{span_str:>{col_w[4]}}"
            + (f"  {tag}" if tag else "")
        )

    print()
    print("  Columns: util = engine utilization; first-staleness = expected versions "
          "since first token was generated;\n"
          "  staleness-inflation = factor over (1+L) baseline; within-span = version "
          "spread within one rollout.")

    # ---- capacity recommendation -------------------------------------------
    if V > 1:
        cap = capacity_recommendation(lv, total_slots)
        print()
        print("  Matched capacity recommendation for the E rung  "
              "(C_i ∝ μ_i · P_i, THEORY.tex §caps):")
        hdr2 = (f"  {'env':<24}  {'P_i':>4}  {'μ_i':>8}  "
                f"{'matched %':>10}  {'blind %':>8}  {'equal %':>8}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for eid in lv["env_ids"]:
            c = cap[eid]
            slots_str = ""
            if total_slots is not None:
                slots_str = f"  ({c['matched_slots']} / {c['blind_slots']} slots)"
            print(f"  {eid:<24}  {c['P_i']:>4}  "
                  f"{lv['envs'][eid]['mu_i']:>8.1f}  "
                  f"{c['matched_frac']*100:>9.1f}%  "
                  f"{c['blind_frac']*100:>7.1f}%  "
                  f"{c['equal_frac']*100:>7.1f}%"
                  + slots_str)
        print()
        print("  blind = current default (C_i ∝ P_i, ignores rollout length).")
        print("  matched = length-aware (equalized delivery rates; see THEORY.tex §caps).")
        print("  The matched weights are the recommended initial values for")
        print("  WeightedMultiTask.set_capacity_weights().")
        _print_matched_weights(lv)

    print()
    print("=" * 72)


def _print_matched_weights(lv: dict) -> None:
    """Print matched capacity weights as a ready-to-use Python snippet."""
    envs = lv["envs"]
    total = sum(e["matched_weight"] for e in envs.values())
    weights = [envs[eid]["matched_weight"] / total for eid in lv["env_ids"]]
    print()
    print("  # Snippet for set_capacity_weights() (normalized matched weights):")
    print(f"  # agent.set_capacity_weights([")
    for eid, w in zip(lv["env_ids"], weights):
        print(f"  #     {w:.4f},  # {eid}")
    print(f"  # ])")


# ---------------------------------------------------------------------------
# Speed-up / trade-off summary
# ---------------------------------------------------------------------------

def print_speedup_summary(lv: dict, lag: int) -> None:
    """Print a concise speed-up table comparing key mode pairs."""
    modes_to_compare = [
        (("B", "B"), ("G", "G"), "G/G vs B/B: throughput gain (at diagonal staleness)"),
        (("G", "G"), ("R", "G"), "R/G vs G/G: util gain (at cost of λ_RG staleness)"),
        (("G", "G"), ("G", "B"), "G/B vs G/G: staleness cost for zero util change"),
    ]
    if lv["V"] > 1:
        modes_to_compare += [
            (("B", "B"), ("E", "E"), "E/E vs B/B: throughput gain (env rung)"),
            (("G", "G"), ("E", "E"), "E/E vs G/G: util change from env rung"),
            (("E", "E"), ("E", "B"), "E/B vs E/E: staleness cost of top edge λ_EB"),
        ]

    print()
    print("  Key trade-offs:")
    for (s1, c1), (s2, c2), label in modes_to_compare:
        p1 = predict_mode(s1, c1, lv, lag)
        p2 = predict_mode(s2, c2, lv, lag)
        util_ratio = p2["util"] / p1["util"] if p1["util"] > 0 else float("inf")
        stale_ratio = p2["first"] / p1["first"] if p1["first"] > 0 else float("inf")
        print(f"  {label}")
        print(f"    util  {p1['util']*100:.0f}% → {p2['util']*100:.0f}%  "
              f"(×{util_ratio:.2f})")
        print(f"    first {p1['first']:.2f} → {p2['first']:.2f} versions  "
              f"(×{stale_ratio:.2f})")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rollout-dir", required=True,
        help="Directory containing rollout JSON files written by --langrl-log-dir.",
    )
    p.add_argument(
        "--rank", type=int, default=0,
        help="Rank whose files to load (default: 0).",
    )
    p.add_argument(
        "--lag", type=int, default=2,
        help="rl_generation_lag L (default: 2).",
    )
    p.add_argument(
        "--total-slots", type=int, default=None,
        help="Total engine KV slots S. When given, capacity recommendation "
             "also shows absolute slot counts. Default: (1+L)·N (right-sized).",
    )
    p.add_argument(
        "--max-batches", type=int, default=None,
        help="Cap the number of batches analyzed (default: all).",
    )
    p.add_argument(
        "--trade-offs", action="store_true",
        help="Print a concise speed-up / trade-off summary in addition to the main table.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading rollout data from {args.rollout_dir!r} (rank {args.rank}) ...",
          file=sys.stderr)
    batches = load_batches(args.rollout_dir, rank=args.rank)
    if args.max_batches:
        batches = batches[: args.max_batches]
    print(f"  loaded {len(batches)} batches.", file=sys.stderr)

    lv = compute_levels(batches)

    total_slots = args.total_slots
    if total_slots is None:
        total_slots = (1 + args.lag) * lv["N"]

    print_report(lv, lag=args.lag, total_slots=total_slots)

    if args.trade_offs:
        print_speedup_summary(lv, lag=args.lag)


if __name__ == "__main__":
    main()
