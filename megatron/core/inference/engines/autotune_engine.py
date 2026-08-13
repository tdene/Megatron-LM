# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Startup memory autotune for the dynamic inference engine.

`AutotuneProfile` is the pure solver: activation tables, the cost model, and
the max_requests solve with buffer-size inversion.
`AutotuneDynamicInferenceEngine` drives it: a workspace-warmup forward pass on
the tiny bootstrap context, a geometric decode/prefill activation sweep with
empirical metadata measurement, then the solve and a context rebuild with the
tuned parameters (before CUDA-graph capture).
"""

import bisect
import gc
import json
import logging
import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import torch

from megatron.core.inference.config import CudaGraphSizingDistribution, KVCacheManagementMode
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.utils import get_pg_size


def _emit_data(enabled: bool, record: str, **fields) -> None:
    """Emit one machine-readable ``AUTOTUNE_DATA`` record.

    GPU-validation graphs are built by grepping ``AUTOTUNE_DATA `` out of the
    run log and json-parsing the remainder, so keep records flat, one line
    each, and unit-suffixed (``_bytes``, ``_count``). ``enabled`` carries the
    caller's rank-0 / verbose gate.
    """
    if enabled:
        logging.info("AUTOTUNE_DATA %s", json.dumps({"record": record, **fields}))


def _emit_context_record(enabled: bool, phase: str, context) -> None:
    """Emit the per-context memory census as an ``AUTOTUNE_DATA`` record."""
    if not enabled:
        return
    cats = context.owned_cuda_tensor_bytes()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    _emit_data(
        True,
        "context",
        phase=phase,
        max_requests=context.max_requests,
        max_tokens=context.max_tokens,
        max_sequence_length=context.max_sequence_length,
        pool_size_blocks=context.kv_block_allocator.pool_size,
        paused_limit_blocks=context.kv_block_allocator.paused_limit,
        kv_cache_bytes=cats["kv_cache"],
        mamba_bytes=cats["mamba"],
        per_request_meta_bytes=cats["per_request"],
        per_token_meta_bytes=cats["per_token"],
        other_bytes=cats["other"],
        nvls_buffer_bytes=context._nvls_buffer_bytes,
        device_total_bytes=total_bytes,
        device_free_bytes=free_bytes,
        torch_allocated_bytes=torch.cuda.memory_allocated(),
        torch_reserved_bytes=torch.cuda.memory_reserved(),
    )


@dataclass
class AutotuneProfile:
    """Profiling data collected during the autotune activation sweep.

    The engine runs forward passes at geometric token counts, recording memory stats for each.
    Prefill and decode activations are tracked separately because a prefill step and a decode step
    have different activation footprints.

    After profiling, `compute_optimal_params` obtains `(max_requests, max_tokens, buffer_size_gb)`.
    """

    # Per-sample measurements (parallel lists, separated by step type).
    prefill_token_counts: List[int] = field(default_factory=list)
    prefill_peak_activation_bytes: List[int] = field(default_factory=list)
    decode_token_counts: List[int] = field(default_factory=list)
    decode_peak_activation_bytes: List[int] = field(default_factory=list)

    # Memory accounting constants from the context.
    block_size_bytes: int = 0
    mamba_memory_per_request: int = 0
    max_kv_block_count: int = 0

    # Empirically measured per-request and per-token metadata bytes.
    # Computed by inspecting all CUDA tensors on the profiling context.
    per_request_metadata_bytes: int = 0
    per_token_metadata_bytes: int = 0

    # Empirically measured runtime overhead per request (sampling + logprobs
    # temporaries outside the CUDA-graph pool). Computed as the median of
    # (sampling-only peak / batch size) across the decode samples.
    runtime_overhead_per_request: int = 0

    # GPU memory state captured before context allocation.
    gpu_total_bytes: int = 0
    memory_after_model_load_bytes: int = 0

    # NVLS EP dispatcher accounting (active only when EP > 1 with the 'nvls'
    # MoE dispatcher; ep_size == 0 means inactive). Its symmetric staging
    # buffers are class-level singletons sized from max_tokens, released by the
    # engine before the final memory measurement, so the solver charges
    # NVLSAllGatherVDispatcher.required_buffer_bytes() at each candidate's
    # derived max_tokens. Plain ints keep the profile persistable.
    nvls_ep_size: int = 0
    nvls_topk: int = 0
    nvls_hidden_size: int = 0

    # Per-term breakdown of the most recent compute_optimal_params solve
    # (derived output, not profiling input): every cost component in bytes for
    # the chosen max_requests, plus the budget and derived block counts. The
    # engine logs it against post-capture measurements as the
    # predicted-vs-actual memory validation report.
    last_solve_breakdown: Optional[Dict[str, int]] = None

    def add_prefill_sample(self, token_count: int, peak_bytes: int):
        """Record one prefill profiling sample."""
        self.prefill_token_counts.append(token_count)
        self.prefill_peak_activation_bytes.append(peak_bytes)

    def add_decode_sample(self, token_count: int, peak_bytes: int):
        """Record one decode profiling sample."""
        self.decode_token_counts.append(token_count)
        self.decode_peak_activation_bytes.append(peak_bytes)

    @staticmethod
    def _build_activation_table(
        token_counts: List[int], peak_bytes: List[int],
    ) -> Dict[int, int]:
        """Deduplicate samples: when multiple share a token count, keep the max."""
        table: Dict[int, int] = {}
        for tc, pb in zip(token_counts, peak_bytes):
            if tc not in table or pb > table[tc]:
                table[tc] = pb
        return dict(sorted(table.items()))

    @staticmethod
    def _interpolate(table: Dict[int, int], x: int) -> int:
        """Linearly interpolate or extrapolate from a sorted table."""
        keys = list(table.keys())
        vals = list(table.values())

        if x <= keys[0]:
            if len(keys) >= 2:
                slope = (vals[1] - vals[0]) / max(keys[1] - keys[0], 1)
                return max(0, int(vals[0] + slope * (x - keys[0])))
            return vals[0]
        if x >= keys[-1]:
            if len(keys) >= 2:
                slope = (vals[-1] - vals[-2]) / max(keys[-1] - keys[-2], 1)
                return max(0, int(vals[-1] + slope * (x - keys[-1])))
            return vals[-1]

        i = bisect.bisect_right(keys, x) - 1
        t = (x - keys[i]) / max(keys[i + 1] - keys[i], 1)
        return int(vals[i] + t * (vals[i + 1] - vals[i]))

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    def _cg_pool_cost(
        self,
        prefill_table: Dict[int, int],
        decode_table: Dict[int, int],
        decode_tokens: int,
        max_tokens: int,
    ) -> int:
        """Bytes retained by the CUDA graph pool.

        The pool keeps the allocation watermark of the largest captured graph:
        the max of a pure-decode step (``max_requests * (1 + spec)`` tokens)
        and a pure-prefill step chunked to ``max_tokens``.
        """
        decode_cost = (
            max(0, self._interpolate(decode_table, decode_tokens)) if decode_table else 0
        )
        prefill_cost = max(0, self._interpolate(prefill_table, max_tokens))
        return max(decode_cost, prefill_cost)

    def _cost_breakdown(
        self,
        prefill_table: Dict[int, int],
        decode_table: Dict[int, int],
        max_requests: int,
        blocks_per_request: int,
        paused_block_estimate: int,
        spec_factor: int,
        token_rounder: int,
        tp_size: int,
    ) -> Dict[str, int]:
        """Per-term physical bytes committed by a candidate ``max_requests``.

        Returns a dict of every cost component plus the derived
        ``max_tokens``; ``_config_cost_bytes`` sums it for the solver's search
        and `compute_optimal_params` stores it as `last_solve_breakdown` for
        the chosen value.
        """
        max_tokens = self._derive_max_tokens(max_requests, spec_factor, token_rounder)
        decode_tokens = max_requests * spec_factor
        cg_pool = self._cg_pool_cost(prefill_table, decode_table, decode_tokens, max_tokens)
        # +1 block: the allocator's pool includes a permanently-reserved dummy
        # block (pool_avail = pool_size - 1), and the paused retention limit
        # holds blocks inside the same pool — so full concurrency needs
        # R x blocks_per_request usable blocks on top of both.
        min_blocks = max_requests * blocks_per_request + 1 + paused_block_estimate

        # EP>1 nvls MoE: symmetric staging buffers, sized by the context as
        # round_up_tokens(max_tokens) // tp — identical to max_tokens // tp
        # here because the derived max_tokens is already token_rounder-aligned
        # (and the rounder is a multiple of tp).
        dispatcher_cost = 0
        if self.nvls_ep_size > 0:
            dispatcher_cost = NVLSAllGatherVDispatcher.required_buffer_bytes(
                per_rank_worst_case_token_count=max_tokens // tp_size,
                topk=self.nvls_topk,
                hidden_size=self.nvls_hidden_size,
                ep_size=self.nvls_ep_size,
            )
        return {
            "max_tokens": max_tokens,
            "cg_pool": cg_pool,
            "token_metadata": max_tokens * self.per_token_metadata_bytes,
            "request_metadata": max_requests * self.per_request_metadata_bytes,
            "mamba_states": max_requests * self.mamba_memory_per_request,
            "runtime_overhead": max_requests * self.runtime_overhead_per_request,
            "kv_blocks": min_blocks * self.block_size_bytes,
            "nvls_dispatcher": dispatcher_cost,
        }

    _COST_TERMS = (
        "cg_pool", "token_metadata", "request_metadata", "mamba_states",
        "runtime_overhead", "kv_blocks", "nvls_dispatcher",
    )

    def _config_cost_bytes(
        self,
        prefill_table: Dict[int, int],
        decode_table: Dict[int, int],
        max_requests: int,
        blocks_per_request: int,
        paused_block_estimate: int,
        spec_factor: int,
        token_rounder: int,
        tp_size: int,
    ) -> Tuple[int, int]:
        """Total physical bytes committed by a candidate ``max_requests``.

        Returns:
            (cost_bytes, derived_max_tokens).
        """
        breakdown = self._cost_breakdown(
            prefill_table, decode_table, max_requests, blocks_per_request,
            paused_block_estimate, spec_factor, token_rounder, tp_size,
        )
        return sum(breakdown[term] for term in self._COST_TERMS), breakdown["max_tokens"]

    @staticmethod
    def _derive_max_tokens(max_requests: int, spec_factor: int, token_rounder: int) -> int:
        """``max_tokens`` is tied to the decode envelope, not solved.

        Prefill runs in chunks of up to ``max_tokens`` via dedicated steps, so
        at solved-``max_requests`` scale, larger chunks are throughput-neutral
        while their prefill CUDA-graph pool permanently eats KV blocks. The
        decode envelope is ``max_requests * spec_factor`` tokens (speculative
        steps carry ``1 + num_speculative_tokens`` tokens per request).
        """
        return math.ceil(max_requests * spec_factor / token_rounder) * token_rounder

    # ------------------------------------------------------------------
    # Buffer-size inversion
    # ------------------------------------------------------------------

    def _simulate_context_block_derive(
        self, buffer_bytes: int, paused_buffer_bytes: int, max_requests: int,
    ) -> Tuple[int, int]:
        """Mirror ``DynamicInferenceContext``'s block-count derivation.

        The tuned config always sets ``max_requests`` and clears
        ``mamba_memory_ratio``, so the context takes either the
        hybrid+max_requests branch (mamba carved proportionally from the
        active and paused buffer inputs) or the non-hybrid branch; both floor
        the result at 2 blocks (>= 1 usable + the dummy block). At
        ``unified_memory_level == 0``, ``block_count`` becomes the allocator's
        ``pool_size`` — the dummy block and the paused retention limit both
        live inside it.

        Returns:
            (block_count, paused_block_count) as the context would derive.
        """
        if self.mamba_memory_per_request > 0:
            total_memory = buffer_bytes + paused_buffer_bytes
            mamba_memory_needed = max_requests * self.mamba_memory_per_request
            if mamba_memory_needed >= total_memory:
                return 0, 0
            mamba_ratio = mamba_memory_needed / total_memory
            scaled_buffer = int(buffer_bytes * (1.0 - mamba_ratio))
            scaled_paused = int(paused_buffer_bytes * (1.0 - mamba_ratio))
            block_count = max(2, scaled_buffer // self.block_size_bytes)
            paused_block_count = scaled_paused // self.block_size_bytes
        else:
            block_count = max(2, buffer_bytes // self.block_size_bytes)
            paused_block_count = paused_buffer_bytes // self.block_size_bytes
        return block_count, paused_block_count

    def _invert_buffer_size(
        self, target_block_count: int, paused_buffer_bytes: int, max_requests: int,
    ) -> int:
        """Smallest ``buffer_bytes`` whose context-derived block count reaches the target.

        The context floors and (for hybrids) rescales, so invert numerically
        against the mirrored derivation instead of trusting closed-form
        algebra to reproduce the same rounding.
        """
        lo = target_block_count * self.block_size_bytes
        if self.mamba_memory_per_request == 0:
            return lo

        hi = lo + max_requests * self.mamba_memory_per_request + self.block_size_bytes
        while self._simulate_context_block_derive(hi, paused_buffer_bytes, max_requests)[0] < (
            target_block_count
        ):
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            derived, _ = self._simulate_context_block_derive(
                mid, paused_buffer_bytes, max_requests
            )
            if derived >= target_block_count:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _validate(self, prefill_table: Dict[int, int]) -> None:
        """Check preconditions before solving."""
        if not prefill_table:
            raise ValueError("No profiling data collected during autotune")
        if self.block_size_bytes <= 0:
            raise ValueError(
                f"Autotune: block_size_bytes must be positive, got {self.block_size_bytes}"
            )

    def _log_result(
        self,
        gpu_free: int,
        cost_bytes: int,
        max_requests: int,
        max_tokens: int,
        buffer_size_gb: float,
        block_count: int,
        blocks_per_request: int,
        extra_blocks: int,
        paused_block_count: int,
        pinned: bool,
    ) -> None:
        """Emit structured log lines summarising the solver result."""
        active_blocks = block_count - paused_block_count
        logging.info(
            "Autotune result (%s): max_requests=%d, max_tokens=%d, buffer_size_gb=%.2f "
            "(%d blocks: %d active + %d paused quota, %d per request + %d extra)",
            "pinned max_requests" if pinned else "solved max_requests",
            max_requests, max_tokens, buffer_size_gb,
            block_count, active_blocks, paused_block_count,
            blocks_per_request, extra_blocks,
        )
        logging.info(
            "Autotune budget: committed %.1f MB of %.1f MB free "
            "(runtime overhead %.1f MB, request metadata %.1f MB, "
            "token metadata %.1f MB, mamba %.1f MB)",
            cost_bytes / (1024 ** 2),
            gpu_free / (1024 ** 2),
            max_requests * self.runtime_overhead_per_request / (1024 ** 2),
            max_requests * self.per_request_metadata_bytes / (1024 ** 2),
            max_tokens * self.per_token_metadata_bytes / (1024 ** 2),
            max_requests * self.mamba_memory_per_request / (1024 ** 2),
        )

    def compute_optimal_params(
        self,
        blocks_per_request: int,
        tp_size: int = 1,
        request_rounder: int = 4,
        paused_buffer_size_bytes: int = 0,
        prefix_caching_mamba_bytes: int = 0,
        token_rounder: int = 64,
        num_speculative_tokens: int = 0,
        pinned_max_requests: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[int, int, float]:
        """Solve for (max_requests, max_tokens, buffer_size_gb).

        ``max_requests`` is the solved quantity: the largest aligned request
        count whose full memory commitment fits the budget. Each unit of
        concurrency is charged its own sequence capacity
        (``blocks_per_request = ceil(average_seq_len / block_size_tokens)``),
        metadata, mamba state, runtime overhead, and its share of the CUDA
        graph pool — so concurrency and sequence capacity are balanced by the
        average-sequence-length workload input rather than by a cap.
        Throughput is monotone in ``max_requests`` under saturated demand, so
        the budget-maximal value is the throughput-optimal one.

        ``max_tokens`` is derived, not solved: it is tied to the decode
        envelope (``max_requests * (1 + num_speculative_tokens)``, aligned to
        the token rounder). See ``_derive_max_tokens``.

        ``buffer_size_gb`` pours all leftover budget into extra KV blocks and
        is inverted through a mirror of the context's block-count derivation
        so the runtime context reproduces the solved block counts exactly.

        Args:
            blocks_per_request: KV blocks per concurrent request, from the
                average runtime sequence length.
            tp_size: Tensor parallel size (for alignment).
            request_rounder: Request count alignment (typically 4).
            paused_buffer_size_bytes: The user's paused-quota request
                (``paused_buffer_size_gb``), passed through to the tuned
                config unchanged; the solver reserves matching blocks so the
                quota never eats the provisioned active capacity.
            prefix_caching_mamba_bytes: The Mamba prefix-cache allocation
                (``prefix_caching_mamba_gb``), a flat reservation off the
                budget; passed through to the tuned config unchanged.
            token_rounder: Token count alignment boundary (typically 64).
            num_speculative_tokens: Speculative tokens per request per step;
                a decode step carries ``max_requests * (1 + this)`` tokens.
            pinned_max_requests: When set (the user passed an explicit
                ``--inference-dynamic-batching-max-requests``), skip the
                search and validate/use this value instead.
            verbose: Emit the solver's info logs. Callers gate this on rank 0
                (and turn it off entirely for probe solves, e.g. the
                retune-on-resume fit check).

        Returns:
            (max_requests, max_tokens, buffer_size_gb).

        Raises:
            ValueError: If profiling data is empty, ``block_size_bytes`` is
                non-positive, or the budget cannot fit the minimum (or the
                pinned) request count.
        """
        prefill_table = self._build_activation_table(
            self.prefill_token_counts, self.prefill_peak_activation_bytes,
        )
        decode_table = self._build_activation_table(
            self.decode_token_counts, self.decode_peak_activation_bytes,
        )
        self._validate(prefill_table)

        # Available GPU memory = total - model weights - mamba prefix cache (a
        # flat allocation the tuned context makes outside the block pool).
        gpu_free = (
            self.gpu_total_bytes
            - self.memory_after_model_load_bytes
            - prefix_caching_mamba_bytes
        )

        alignment = max(tp_size, request_rounder)
        tr = math.ceil(token_rounder / tp_size) * tp_size
        spec_factor = 1 + num_speculative_tokens

        # Conservative quota estimate for the feasibility model: the context
        # scales the paused buffer down when carving mamba, so the unscaled
        # count over-reserves slightly — never under.
        paused_block_estimate = math.ceil(paused_buffer_size_bytes / self.block_size_bytes)

        def cost(r: int) -> Tuple[int, int]:
            return self._config_cost_bytes(
                prefill_table, decode_table, r, blocks_per_request,
                paused_block_estimate, spec_factor, tr, tp_size,
            )

        if verbose:
            logging.info(
                "Autotune: GPU total %d MB, after model load %d MB, free %d MB "
                "(mamba prefix cache reserved %d MB), "
                "blocks_per_request=%d, spec_factor=%d, paused_quota_est=%d blocks",
                self.gpu_total_bytes // (1024 ** 2),
                self.memory_after_model_load_bytes // (1024 ** 2),
                gpu_free // (1024 ** 2),
                prefix_caching_mamba_bytes // (1024 ** 2),
                blocks_per_request,
                spec_factor,
                paused_block_estimate,
            )
            logging.info(
                "Autotune: prefill activation table (tokens -> MB): %s",
                {tc: round(b / (1024 ** 2), 1) for tc, b in prefill_table.items()},
            )
            logging.info(
                "Autotune: decode activation table (tokens -> MB): %s",
                {tc: round(b / (1024 ** 2), 1) for tc, b in decode_table.items()},
            )

        if pinned_max_requests is not None:
            max_requests = max(alignment, (pinned_max_requests // alignment) * alignment)
            cost_bytes, max_tokens = cost(max_requests)
            if cost_bytes > gpu_free:
                raise ValueError(
                    f"Autotune: not enough GPU memory for the pinned "
                    f"max_requests={max_requests} with {blocks_per_request} blocks "
                    f"each. Need {cost_bytes / (1024 ** 2):.0f} MB but only "
                    f"{gpu_free / (1024 ** 2):.0f} MB available. Lower "
                    f"--inference-dynamic-batching-max-requests (or unset it to "
                    f"let autotune solve it), or reduce "
                    f"--inference-dynamic-batching-autotune-average-seq-len."
                )
        else:
            # Largest feasible aligned max_requests: cost() is monotone in r
            # (every term is non-decreasing), so double then binary-search.
            max_requests = alignment
            cost_bytes, max_tokens = cost(max_requests)
            if cost_bytes > gpu_free:
                raise ValueError(
                    f"Autotune: not enough GPU memory for even {alignment} "
                    f"requests with {blocks_per_request} blocks each. Need "
                    f"{cost_bytes / (1024 ** 2):.0f} MB but only "
                    f"{gpu_free / (1024 ** 2):.0f} MB available. Reduce "
                    f"--inference-dynamic-batching-autotune-average-seq-len."
                )
            hi = max_requests
            while True:
                candidate = hi * 2
                if candidate * spec_factor > 1 << 24:
                    break
                c, _ = cost(candidate)
                if c > gpu_free:
                    break
                hi = candidate
            # Invariant: hi is feasible; (hi, 2*hi) may hold a larger value.
            lo = hi
            hi2 = hi * 2
            while lo + alignment < hi2:
                mid = ((lo + hi2) // 2 // alignment) * alignment
                mid = max(mid, lo + alignment)
                c, _ = cost(mid)
                if c <= gpu_free:
                    lo = mid
                else:
                    hi2 = mid
            max_requests = lo
            cost_bytes, max_tokens = cost(max_requests)

        # Pour the leftover budget into extra KV blocks.
        leftover = gpu_free - cost_bytes
        extra_blocks = max(0, int(leftover // self.block_size_bytes))
        target_block_count = (
            max_requests * blocks_per_request + 1 + paused_block_estimate + extra_blocks
        )

        # Invert the context's derivation so the runtime block counts land on
        # target. The derived quota can undercut the unscaled estimate by a
        # few blocks (mamba rescale + floors); active capacity only gains.
        buffer_bytes = self._invert_buffer_size(
            target_block_count, paused_buffer_size_bytes, max_requests
        )
        block_count, derived_paused = self._simulate_context_block_derive(
            buffer_bytes, paused_buffer_size_bytes, max_requests
        )
        assert block_count - derived_paused - 1 >= max_requests * blocks_per_request, (
            f"Autotune internal error: derived active blocks "
            f"{block_count - derived_paused - 1} < provisioned "
            f"{max_requests * blocks_per_request}"
        )
        buffer_size_gb = buffer_bytes / (1024 ** 3)

        # Per-term prediction for the chosen max_requests; the engine logs it
        # against post-capture measurements (predicted-vs-actual validation).
        self.last_solve_breakdown = {
            **self._cost_breakdown(
                prefill_table, decode_table, max_requests, blocks_per_request,
                paused_block_estimate, spec_factor, tr, tp_size,
            ),
            "max_requests": max_requests,
            "gpu_free": gpu_free,
            "prefix_caching_mamba_bytes": prefix_caching_mamba_bytes,
            "committed_min": cost_bytes,
            "extra_kv_bytes": extra_blocks * self.block_size_bytes,
            "committed_total": cost_bytes + extra_blocks * self.block_size_bytes,
            "buffer_bytes": buffer_bytes,
            "block_count": block_count,
            "paused_block_count": derived_paused,
        }

        if verbose:
            self._log_result(
                gpu_free, cost_bytes + extra_blocks * self.block_size_bytes,
                max_requests, max_tokens, buffer_size_gb, block_count,
                blocks_per_request, extra_blocks, derived_paused,
                pinned=pinned_max_requests is not None,
            )
        _emit_data(
            verbose,
            "solve",
            blocks_per_request=blocks_per_request,
            pinned=pinned_max_requests is not None,
            runtime_overhead_per_request_bytes=self.runtime_overhead_per_request,
            memory_after_model_load_bytes=self.memory_after_model_load_bytes,
            **self.last_solve_breakdown,
        )
        if verbose:
            # Budget landscape for the validation graphs: every cost term at
            # candidate max_requests values, from the alignment floor up to
            # ~1.5x the chosen value so the infeasible region is visible.
            curve_points = sorted({
                max(alignment, (max_requests * i // 20 // alignment) * alignment)
                for i in range(1, 31)
            })
            for r in curve_points:
                bd = self._cost_breakdown(
                    prefill_table, decode_table, r, blocks_per_request,
                    paused_block_estimate, spec_factor, tr, tp_size,
                )
                total = sum(bd[term] for term in self._COST_TERMS)
                _emit_data(
                    True,
                    "cost_curve",
                    candidate_max_requests=r,
                    total_bytes=total,
                    gpu_free_bytes=gpu_free,
                    feasible=total <= gpu_free,
                    **bd,
                )

        return max_requests, max_tokens, buffer_size_gb


class AutotuneDynamicInferenceEngine(DynamicInferenceEngine):
    """Dynamic engine that solves its memory parameters at startup.

    Construct with a context built from an ``autotune=True`` `InferenceConfig`
    (the context constructor substitutes a tiny bootstrap context); this engine
    profiles on it and rebuilds with the tuned config before CUDA graphs are
    captured. See `InferenceConfig.autotune` for the supported engine options.
    """

    def _before_cuda_graph_capture(self) -> None:
        self._autotune_and_rebuild()

    def _autotune_and_rebuild(self):
        """Profile activation memory on the existing context, then rebuild with tuned parameters."""
        model_config = self.controller.inference_wrapped_model.model.config
        # The context constructor substituted the tiny bootstrap context and stashed
        # the user's config; the solve targets that config, not the bootstrap's.
        old_config = self.context.autotune_target_config
        assert old_config is not None, (
            "Autotune requires a bootstrap context: construct DynamicInferenceContext "
            "with autotune=True in its InferenceConfig."
        )
        self._validate_autotune_requirements(model_config, old_config)
        controller = self.controller
        gpu_total = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        tp_size = max(model_config.tensor_model_parallel_size, 1)

        # --- Step 1: Workspace warmup ---
        self._warmup_workspace(controller, rank)

        # --- Step 2: Profiling ---
        profile = self._build_profiling_context_and_profile(
            model_config, old_config, controller, gpu_total, rank, tp_size,
        )

        # --- Step 3: Solve and rebuild ---
        self._solve_and_rebuild(
            profile, model_config, old_config, controller,
            gpu_total, rank, tp_size,
        )

    def create_cuda_graphs(self, reset_context: bool = True):
        """Capture CUDA graphs, then log the predicted-vs-actual memory report.

        The base implementation captures on the tuned context; afterwards
        every solver prediction is compared against a measurement of the same
        consumer — the core artifact for validating the autotune budget on
        real hardware.
        """
        super().create_cuda_graphs(reset_context=reset_context)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        prediction = getattr(self, "_autotune_prediction", None)
        if rank != 0 or prediction is None:
            return

        context = self.context
        stats = self.capture_stats or {}
        pool_reserved = stats.get("pool_reserved_bytes", 0)
        per_req_meta, per_tok_meta = context.measure_metadata_bytes()
        cats = context.owned_cuda_tensor_bytes()
        sampling_bytes = sum(
            t.untyped_storage().nbytes()
            for t in (
                getattr(self.controller, "_all_logits_cuda", None),
                getattr(self.controller, "_sampled_tokens_cuda", None),
                getattr(self.controller, "_async_sched_sample_values_cuda", None),
            )
            if isinstance(t, torch.Tensor) and t.is_cuda
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info()

        def mb(x):
            return x / (1024 ** 2)

        lines = [
            "========== Autotune memory validation (predicted vs actual) ==========",
            f"  tuned max_requests={context.max_requests}, max_tokens={context.max_tokens} "
            f"(rank-0 solve: {prediction['max_requests']} / {prediction['max_tokens']})",
            f"  CUDA-graph pool:  predicted {mb(prediction['cg_pool']):11.1f} MB | "
            f"actual reserved {mb(pool_reserved):11.1f} MB"
            + ("" if stats else " (graphs were not captured)"),
            f"  KV cache:         predicted "
            f"{mb(prediction['kv_blocks'] + prediction['extra_kv_bytes']):11.1f} MB "
            f"({prediction['block_count']} blocks, {prediction['paused_block_count']} paused) | "
            f"actual {mb(cats['kv_cache']):11.1f} MB "
            f"({context.kv_block_allocator.pool_size} blocks, "
            f"paused limit {context.kv_block_allocator.paused_limit})",
            f"  Request metadata: predicted {mb(prediction['request_metadata']):11.1f} MB | "
            f"actual {mb(per_req_meta * context.max_requests):11.1f} MB",
            f"  Token metadata:   predicted {mb(prediction['token_metadata']):11.1f} MB | "
            f"actual {mb(per_tok_meta * context.max_tokens):11.1f} MB",
            f"  Mamba states:     predicted {mb(prediction['mamba_states']):11.1f} MB | "
            f"actual {mb(cats['mamba']):11.1f} MB (incl. spec buffers + prefix cache)",
        ]
        if prediction["prefix_caching_mamba_bytes"] > 0:
            lines.append(
                f"  Mamba prefix cache: reserved "
                f"{mb(prediction['prefix_caching_mamba_bytes']):9.1f} MB off the budget"
            )
        if prediction["nvls_dispatcher"] > 0 or context._nvls_buffer_bytes > 0:
            lines.append(
                f"  NVLS buffers:     predicted {mb(prediction['nvls_dispatcher']):11.1f} MB | "
                f"actual {mb(context._nvls_buffer_bytes):11.1f} MB"
            )
        lines += [
            f"  Unbudgeted: other context tensors {mb(cats['other']):.1f} MB "
            f"(gpu view, attn metadata, ...) "
            f"+ controller sampling tensors {mb(sampling_bytes):.1f} MB",
            f"  Runtime overhead: predicted {mb(prediction['runtime_overhead']):11.1f} MB "
            f"(not preallocated — device free must stay above it)",
            f"  Budget: committed {mb(prediction['committed_total']):.1f} MB "
            f"of {mb(prediction['gpu_free']):.1f} MB usable",
            f"  Device: total {total_bytes / (1024 ** 3):.1f} GB, "
            f"used {(total_bytes - free_bytes) / (1024 ** 3):.1f} GB, "
            f"free {free_bytes / (1024 ** 3):.1f} GB",
            "=======================================================================",
        ]
        logging.info("\n".join(lines))
        _emit_data(
            True,
            "validation",
            predicted=prediction,
            actual_pool_reserved_bytes=int(pool_reserved),
            actual_pool_allocated_bytes=int(stats.get("pool_allocated_bytes", 0)),
            capture_time_s=stats.get("time", 0),
            actual_kv_cache_bytes=cats["kv_cache"],
            actual_mamba_bytes=cats["mamba"],
            actual_request_metadata_bytes=per_req_meta * context.max_requests,
            actual_token_metadata_bytes=per_tok_meta * context.max_tokens,
            actual_nvls_buffer_bytes=context._nvls_buffer_bytes,
            actual_sampling_bytes=int(sampling_bytes),
            actual_other_bytes=cats["other"],
            device_total_bytes=total_bytes,
            device_free_bytes=free_bytes,
            tuned_max_requests=context.max_requests,
            tuned_max_tokens=context.max_tokens,
        )

    # ---- autotune sub-steps ------------------------------------------------

    def _validate_autotune_requirements(self, model_config, config) -> None:
        """Enforce every engine option the autotune flow assumes.

        This is the single enforcement point for the requirement list on
        `InferenceConfig.autotune`; ``megatron/training/arguments.py`` merely
        auto-remediates the CLI surface ahead of it (with warnings), so API
        users get the same guarantees as flag users. All violations are
        collected and raised together.

        The solver's premises: every step type must run from CUDA graphs whose
        pool the solver budgets — an eager or under-covered step would draw
        its activations from headroom the solver poured into KV blocks — and
        free memory must be measurable with ``mem_get_info``, which UVM
        overcommit defeats.

        Args:
            model_config: The model's `TransformerConfig`.
            config: The autotune target `InferenceConfig` (the user's config,
                not the bootstrap substitution).

        Raises:
            ValueError: Listing every unmet requirement.
        """
        problems = []
        if not config.autotune_average_seq_len or config.autotune_average_seq_len <= 0:
            problems.append(
                "autotune_average_seq_len must be a positive integer (the expected "
                "average prompt+generation sequence length); got "
                f"{config.autotune_average_seq_len!r}."
            )
        elif config.autotune_average_seq_len > config.max_sequence_length:
            problems.append(
                f"autotune_average_seq_len ({config.autotune_average_seq_len}) exceeds "
                f"max_sequence_length ({config.max_sequence_length}): no request can "
                "average more than the sequence cap, and the solver would charge every "
                "request KV blocks it can never use."
            )
        if model_config.inference_moe_token_dispatcher_type != 'nvls':
            problems.append(
                "inference_moe_token_dispatcher_type must be 'nvls' regardless of expert "
                "parallelism (the nccl and training a2a dispatchers force-disable "
                "non-decode CUDA graphs the moment EP > 1); got "
                f"{model_config.inference_moe_token_dispatcher_type!r}."
            )
        if not config.use_cuda_graphs_for_non_decode_steps:
            problems.append(
                "use_cuda_graphs_for_non_decode_steps must be enabled so prefill "
                "activations are retained by the budgeted CUDA-graph pool."
            )
        elif not self.context.use_cuda_graphs_for_non_decode_steps:
            problems.append(
                "non-decode CUDA graphs were force-disabled by the MoE dispatcher; with "
                "expert parallelism use --transformer-impl=inference_optimized and the "
                "'nvls' inference token dispatcher."
            )
        if not config.cuda_graph_all_prefills:
            problems.append(
                "cuda_graph_all_prefills must be enabled so prefill graphs extend to "
                "the tuned max_tokens instead of stopping at cuda_graph_max_tokens."
            )
        if model_config.cuda_graph_impl != 'local':
            problems.append(
                f"cuda_graph_impl must be 'local'; got {model_config.cuda_graph_impl!r}."
            )
        if model_config.moe_pad_experts_for_cuda_graph_inference:
            problems.append(
                "moe_pad_experts_for_cuda_graph_inference must be disabled: the "
                "inference-optimized stack (which autotune requires) masks CUDA-graph "
                "padding at the router instead, and the controller rejects the "
                "combination at the first dynamic step."
            )
        if config.num_cuda_graphs is None:
            problems.append(
                "num_cuda_graphs must be set (None disables CUDA-graph capture "
                "entirely, leaving every step eager)."
            )
        if not config.cuda_graph_mixed_prefill_count or config.cuda_graph_mixed_prefill_count <= 0:
            problems.append(
                "cuda_graph_mixed_prefill_count must be > 0 so mixed prefill+decode "
                f"steps are graphed; got {config.cuda_graph_mixed_prefill_count!r}."
            )
        if config.cuda_graph_sizing_distribution != CudaGraphSizingDistribution.EXPONENTIAL:
            problems.append(
                "cuda_graph_sizing_distribution must be EXPONENTIAL; got "
                f"{config.cuda_graph_sizing_distribution!r}."
            )
        if not config.enable_chunked_prefill:
            problems.append(
                "enable_chunked_prefill must be enabled so any prompt's prefill step "
                "is bounded by the tuned max_tokens."
            )
        if config.unified_memory_level != 0:
            problems.append(
                "unified_memory_level must be 0 (autotune budgets physical GPU memory "
                f"via mem_get_info, which UVM overcommit defeats); got "
                f"{config.unified_memory_level!r}."
            )
        if problems:
            raise ValueError(
                "Autotune configuration errors (see InferenceConfig.autotune for the "
                "supported engine options):\n  - " + "\n  - ".join(problems)
            )

    def _warmup_workspace(self, controller, rank) -> None:
        """Run one forward pass on the bootstrap context to trigger cuBLAS /
        MoE workspace allocation.

        The context constructor already substituted the tiny bootstrap context
        (no user-sized context ever exists under autotune). Autotune is opt-in:
        a failure here (or anywhere in the autotune flow) crashes startup.
        There is no graceful fallback — a silently degraded configuration
        would be worse than an honest crash.

        The prompt is one full token-rounder chunk so padded == real token
        count: every autotune forward stays in the exact-padding regime (a
        1-token prompt pads its query buffer to the token rounder while
        attention metadata pads requests, and the mismatch crashes varlen
        attention's single-token fast path).
        """
        warmup_tokens = self.context.round_up_tokens(1)
        dummy = [DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.ones(
                warmup_tokens, dtype=torch.long, device=torch.cuda.current_device()
            ),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
        )]
        self.context.add_dummy_requests_parallel(dummy)
        input_ids, position_ids, _ = controller._dynamic_step_context_init()
        with torch.inference_mode():
            controller._dynamic_step_forward_logits(input_ids, position_ids)

        # Second (workspace-warm) pass, measured: a floor sample at the
        # bootstrap context's max_tokens. Together with the sweep's smallest
        # sample on the profiling context — which is pinned to a different
        # max_tokens — this gives two per-forward-floor measurements at two
        # max_tokens values in every run.
        self.context.reset()
        self.context.add_dummy_requests_parallel(dummy)
        input_ids, position_ids, _ = controller._dynamic_step_context_init()
        baseline_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            controller._dynamic_step_forward_logits(input_ids, position_ids)
        torch.cuda.synchronize()
        warmup_activation = torch.cuda.max_memory_allocated() - baseline_bytes
        self.context.reset()

        if rank == 0:
            free, _ = torch.cuda.mem_get_info()
            logging.info(
                "Autotune: workspace warmup done. Free memory: %.1f GB "
                "(floor sample: %.1f MB at max_tokens=%d)",
                free / (1024 ** 3),
                warmup_activation / (1024 ** 2),
                self.context.max_tokens,
            )
            logging.info("Autotune: bootstrap context:\n%s", self.context.memory_report())
        _emit_data(
            rank == 0,
            "sample",
            phase="bootstrap",
            step_type="prefill",
            token_count=warmup_tokens,
            num_requests=1,
            activation_bytes=int(warmup_activation),
            context_max_tokens=self.context.max_tokens,
        )
        _emit_context_record(rank == 0, "bootstrap", self.context)

    def _build_profiling_context_and_profile(
        self, model_config, old_config, controller,
        gpu_total, rank, tp_size,
    ):
        """Free the bootstrap context, build a profiling-sized context, run
        the activation sweep, tear the profiling context down, and return the
        profile."""
        alignment = max(tp_size, DynamicInferenceContext.REQUEST_ROUNDER)

        # Read per-request cost from the bootstrap context, then release it
        # for real (all references dropped — see _release_context) so the
        # profiling-context sizing below sees the true free memory.
        per_req_bytes = (
            self.context.block_size_bytes + self.context.mamba_states_memory_per_request
        )
        self._release_context()

        # Size the profiling context to fit as many requests as possible
        # (up to half the free memory), so the activation sweep covers a
        # wide range.  The solver will later determine the actual max_requests.
        free_for_profiling, _ = torch.cuda.mem_get_info()
        profiling_max_requests = max(
            alignment,
            int(free_for_profiling * 0.5) // max(per_req_bytes, 1),
        )
        profiling_max_requests = (profiling_max_requests // alignment) * alignment
        profiling_max_requests = max(alignment, profiling_max_requests)
        profiling_max_requests = min(
            profiling_max_requests, DynamicInferenceContext.DEFAULT_MAX_TOKENS,
        )

        # All ranks must agree to avoid NCCL deadlocks.
        if torch.distributed.is_initialized():
            sync_tensor = torch.tensor(
                [profiling_max_requests], dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.MIN)
            profiling_max_requests = int(sync_tensor.item())
            profiling_max_requests = (profiling_max_requests // alignment) * alignment
            profiling_max_requests = max(alignment, profiling_max_requests)

        profiling_buffer_gb = max(
            0.1, (profiling_max_requests * per_req_bytes * 1.05) / (1024 ** 3),
        )

        # Pin the profiling max_tokens to the sweep's needs instead of the
        # 16384 default: sized tight for the sweep, and — because the
        # bootstrap runs at the default — every run measures the per-forward
        # floor at two different max_tokens values (the floor hypothesis:
        # fixed per-step buffers scale with the context's max_tokens).
        profiling_max_tokens = DynamicInferenceContext.round_up_tokens(profiling_max_requests)
        profiling_config = replace(
            old_config,
            buffer_size_gb=profiling_buffer_gb,
            max_requests=profiling_max_requests,
            max_tokens=profiling_max_tokens,
            autotune=False,
            static_kv_memory_pointers=False,
            mamba_memory_ratio=None,
            paused_buffer_size_gb=None,
            enable_prefix_caching=False,
            prefix_caching_mamba_gb=None,
            num_cuda_graphs=None,
            kv_cache_management_mode=KVCacheManagementMode.PERSIST,
        )
        self.context = DynamicInferenceContext(model_config, profiling_config)
        controller.inference_wrapped_model.inference_context = self.context
        controller._init_dynamic_sampling_tensors()
        self.reset()
        context = self.context

        free_after_ctx, _ = torch.cuda.mem_get_info()
        if rank == 0:
            logging.info(
                "Autotune: profiling context created (max_requests=%d, buffer=%.2f GB). "
                "Free memory: %.1f GB",
                context.max_requests, profiling_buffer_gb, free_after_ctx / (1024 ** 3),
            )
            logging.info("Autotune: profiling context:\n%s", context.memory_report())
        _emit_context_record(rank == 0, "profiling", context)

        # Measure per-request / per-token metadata empirically.
        per_request_metadata, per_token_metadata = context.measure_metadata_bytes()
        if rank == 0:
            logging.info(
                "Autotune: measured metadata: %d bytes/request, %d bytes/token",
                per_request_metadata, per_token_metadata,
            )

        # NVLS EP dispatcher constants for the solver's staging-buffer cost
        # term; the activation condition mirrors the context's
        # _nvls_dispatcher (EP > 1 with the 'nvls' dispatcher).
        nvls_ep_size = (
            get_pg_size(context.expert_model_parallel_group)
            if context._nvls_dispatcher
            else 0
        )
        _emit_data(
            rank == 0,
            "constants",
            block_size_bytes=context.block_size_bytes,
            block_size_tokens=old_config.block_size_tokens,
            mamba_per_request_bytes=context.mamba_states_memory_per_request,
            per_request_metadata_bytes=per_request_metadata,
            per_token_metadata_bytes=per_token_metadata,
            max_sequence_length=old_config.max_sequence_length,
            tp_size=tp_size,
            ep_size=get_pg_size(context.expert_model_parallel_group),
            nvls_active=bool(nvls_ep_size),
            gpu_total_bytes=gpu_total,
        )
        profile = AutotuneProfile(
            gpu_total_bytes=gpu_total,
            block_size_bytes=context.block_size_bytes,
            mamba_memory_per_request=context.mamba_states_memory_per_request,
            max_kv_block_count=context.max_kv_block_count,
            per_request_metadata_bytes=per_request_metadata,
            per_token_metadata_bytes=per_token_metadata,
            nvls_ep_size=nvls_ep_size,
            nvls_topk=model_config.moe_router_topk if nvls_ep_size else 0,
            nvls_hidden_size=(
                (model_config.moe_latent_size or model_config.hidden_size)
                if nvls_ep_size
                else 0
            ),
        )

        # --- Activation sweep ---
        self._run_profiling_sweep(profile, context, controller, rank)

        # Re-measure memory AFTER profiling (workspace is now allocated), with
        # the profiling context genuinely torn down first: it does not use
        # TMS, so it frees only once every reference is dropped. Measuring
        # with it resident would fold its ~half-of-free-memory buffer into
        # memory_after_model_load_bytes and roughly halve the tuned budget.
        del context
        self._release_context()
        free_after_profiling, _ = torch.cuda.mem_get_info()
        profile.memory_after_model_load_bytes = gpu_total - free_after_profiling
        if rank == 0:
            logging.info(
                "Autotune: after profiling: model+runtime+workspace = %.1f GB "
                "(free %.1f GB)",
                profile.memory_after_model_load_bytes / (1024 ** 3),
                free_after_profiling / (1024 ** 3),
            )

        return profile

    def _run_profiling_sweep(self, profile, context, controller, rank):
        """Run decode and prefill forward passes at geometric token counts,
        measuring activation memory and sampling overhead.

        Only token counts that are exact token-rounder multiples are swept, so
        padded == real for every profiling forward: prefill pads its query
        buffer to the token rounder and decode pads to the request rounder,
        and rounder multiples satisfy both. The sub-rounder regime is never
        hit at solved scale, and its padding-request attribution is fragile
        (a padded single-token prefill crashes varlen attention's
        one-token fast path).

        Prefill is profiled at two compositions per token count and the
        activation table keeps the max: one long request (worst case for
        attention) and fragmented single-token requests (worst case for the
        Mamba chunk scan, whose state tensors scale with the number of
        sequences — the prefill-only CUDA graph dims run up to
        ``min(max_requests, token_count)`` prefill sequences, which a
        one-request profile under-bounds by orders of magnitude on hybrid
        models).
        """
        tr = context.round_up_tokens(1)

        def _geometric_sweep(upper: int) -> List[int]:
            counts = []
            t = tr
            while t <= upper:
                counts.append(t)
                t += max(tr, (t // 4) // tr * tr)
            top = (upper // tr) * tr
            if top >= tr and top not in counts:
                counts.append(top)
            return sorted(counts)

        sweep_upper = context.max_requests
        assert sweep_upper >= tr, (
            f"Autotune: profiling context max_requests ({sweep_upper}) is below "
            f"the token rounder ({tr}); not enough free memory to profile at "
            f"rounder granularity."
        )
        prefill_tcs = _geometric_sweep(sweep_upper)

        # Decode runs the full ladder too: the decode table (and the
        # activation-vs-requests data it yields) is per-request-linear with a
        # knee, and a dense curve both tightens the solve's interpolation and
        # feeds the validation graphs. Decode forwards are cheap.
        decode_tcs = prefill_tcs

        if rank == 0:
            logging.info(
                "Autotune: profiling %d decode + 2x%d prefill samples (≤%d)",
                len(decode_tcs), len(prefill_tcs), sweep_upper,
            )

        # kinds: "decode" = tc requests x 1 token (decode step);
        # "prefill" = 1 request x tc tokens (longest-sequence attention worst
        # case); "prefill_fragmented" = tc requests x 1 token counted as
        # prefill (max-sequence-count Mamba chunk-scan worst case, matching
        # the largest prefill-only CUDA graph dims). Every composition keeps
        # padded == real: token and request counts are rounder multiples.
        sweep = (
            [(tc, "decode") for tc in decode_tcs]
            + [(tc, "prefill") for tc in prefill_tcs]
            + [(tc, "prefill_fragmented") for tc in prefill_tcs]
        )
        overhead_samples: List[Tuple[int, int]] = []
        device = torch.cuda.current_device()

        for tc, kind in sweep:
            # Autotune is opt-in: a failure anywhere in the sweep crashes
            # startup rather than skipping the sample — a silently thinner
            # profile would mis-budget the solve.
            if kind == "prefill":
                dummy = [DynamicInferenceRequest(
                    request_id=0,
                    prompt_tokens=torch.ones(tc, dtype=torch.long, device=device),
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=1, termination_id=-1,
                    ),
                )]
            else:
                dummy = [
                    DynamicInferenceRequest(
                        request_id=i,
                        prompt_tokens=torch.ones(1, dtype=torch.long, device=device),
                        sampling_params=SamplingParams(
                            num_tokens_to_generate=1, termination_id=-1,
                        ),
                    )
                    for i in range(tc)
                ]
            context.add_dummy_requests_parallel(dummy, count_as_prefill=(kind != "decode"))

            # Forward pass.  Inside NCCL collectives — NOT caught.
            baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                input_ids, position_ids, _ = controller._dynamic_step_context_init()
                controller._dynamic_step_forward_logits(input_ids, position_ids)
            torch.cuda.synchronize()

            fwd_activation = torch.cuda.max_memory_allocated() - baseline_bytes

            if kind == "decode":
                profile.add_decode_sample(tc, fwd_activation)
            else:
                # Both prefill compositions land in one table; construction
                # keeps the max per token count, so the solver's CG-pool term
                # bounds the worst captured graph shape.
                profile.add_prefill_sample(tc, fwd_activation)
            if rank == 0:
                logging.info(
                    "Autotune: %s sample tc=%d: forward activation %.1f MB",
                    kind, tc, fwd_activation / (1024 ** 2),
                )
            _emit_data(
                rank == 0,
                "sample",
                phase="profiling",
                step_type=kind,
                token_count=tc,
                num_requests=1 if kind == "prefill" else tc,
                activation_bytes=int(fwd_activation),
                context_max_tokens=context.max_tokens,
            )

            # Sampling overhead (decode only). Unguarded: if sampling cannot
            # run during profiling, the overhead measurement is broken, and a
            # silent zero would under-budget the solve.
            if kind == "decode":
                torch.cuda.synchronize()
                sampling_baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    controller._dynamic_step_sample_logits()
                torch.cuda.synchronize()
                sampling_peak = max(0, torch.cuda.max_memory_allocated() - sampling_baseline)
                overhead_samples.append((tc, sampling_peak))
                if rank == 0:
                    logging.info(
                        "Autotune: decode sample tc=%d: sampling overhead %.1f MB "
                        "(%.2f MB/request)",
                        tc,
                        sampling_peak / (1024 ** 2),
                        sampling_peak / tc / (1024 ** 2),
                    )
                _emit_data(
                    rank == 0,
                    "sampling_overhead",
                    num_requests=tc,
                    overhead_bytes=int(sampling_peak),
                )

            context.reset()

        # Per-request runtime overhead: true median. decode_tcs is never
        # empty, so an empty sample set is a bug and median() will raise.
        per_req_values = [oh / tc for tc, oh in overhead_samples]
        profile.runtime_overhead_per_request = int(statistics.median(per_req_values))
        if rank == 0:
            logging.info(
                "Autotune: runtime overhead measured at %d batch sizes, "
                "median %.2f MB/req",
                len(overhead_samples),
                profile.runtime_overhead_per_request / (1024 ** 2),
            )

        if rank == 0:
            logging.info(
                "Autotune: collected %d prefill + %d decode profiling samples",
                len(profile.prefill_token_counts),
                len(profile.decode_token_counts),
            )

    def _solve_and_rebuild(
        self, profile, model_config, old_config, controller,
        gpu_total, rank, tp_size,
    ):
        """Run the solver, sync results across ranks, and rebuild the
        context with tuned parameters."""
        paused_gb = old_config.paused_buffer_size_gb or 0.0
        paused_bytes = int(paused_gb * (1024 ** 3))
        # Mirror the context's mamba prefix-cache allocation condition: the tuned
        # context allocates this flat cache outside the block pool, so the solver
        # reserves it off the budget. The profiling context is already released
        # (self.context is None here); mamba_memory_per_request > 0 is the same
        # hybrid test the solver's block derivation uses.
        mamba_prefix_cache_bytes = 0
        if (
            profile.mamba_memory_per_request > 0
            and old_config.enable_prefix_caching
            and old_config.prefix_caching_mamba_gb is not None
            and old_config.prefix_caching_mamba_gb > 0
        ):
            mamba_prefix_cache_bytes = int(old_config.prefix_caching_mamba_gb * (1024 ** 3))
        block_size_tokens = old_config.block_size_tokens
        avg_seq_len = old_config.autotune_average_seq_len
        blocks_per_request = math.ceil(avg_seq_len / block_size_tokens)

        new_max_requests, new_max_tokens, new_buffer_size_gb = profile.compute_optimal_params(
            blocks_per_request=blocks_per_request,
            tp_size=tp_size,
            request_rounder=DynamicInferenceContext.REQUEST_ROUNDER,
            paused_buffer_size_bytes=paused_bytes,
            prefix_caching_mamba_bytes=mamba_prefix_cache_bytes,
            token_rounder=DynamicInferenceContext.TOKEN_ROUNDER,
            num_speculative_tokens=old_config.num_speculative_tokens,
            pinned_max_requests=old_config.max_requests,
            verbose=(rank == 0),
        )

        # Kept for the post-capture predicted-vs-actual validation report;
        # rank 0's own solve (the sync below can only lower the values).
        self._autotune_prediction = dict(profile.last_solve_breakdown)

        new_max_requests, new_max_tokens, new_buffer_size_gb = (
            self._sync_tuned_params_across_ranks(
                new_max_requests, new_max_tokens, new_buffer_size_gb,
            )
        )

        if rank == 0:
            logging.info(
                "Autotune: rebuilding context with max_requests=%d, max_tokens=%d, "
                "buffer_size_gb=%.2f",
                new_max_requests, new_max_tokens, new_buffer_size_gb,
            )

        # The profiling context was fully released before the final memory
        # measurement, so the tuned context is the only large allocation from
        # here on. paused_buffer_size_gb, prefix_caching_mamba_gb, and
        # static_kv_memory_pointers pass through unchanged: the solver reserved
        # the quota blocks and the cache bytes, and the RL suspend/resume path
        # needs the user's static-pointer setting (TMS tags are per-instance,
        # so autotune rebuilds cannot collide).
        tuned_config = replace(
            old_config,
            max_requests=new_max_requests,
            max_tokens=new_max_tokens,
            buffer_size_gb=new_buffer_size_gb,
            mamba_memory_ratio=None,
            autotune=False,
        )
        self.context = DynamicInferenceContext(model_config, tuned_config)
        controller.inference_wrapped_model.inference_context = self.context
        controller._init_dynamic_sampling_tensors()
        self.reset()

        # CUDA graphs are built by the caller (create_cuda_graphs) after return.

        # Cache scaling vs sequence length (arithmetic from measured
        # constants): per-request KV grows in block quanta with the average
        # sequence length, mamba state is flat — the "cache vs tokens" graph.
        if rank == 0:
            seq_step = max(block_size_tokens, old_config.max_sequence_length // 16)
            for s in range(seq_step, old_config.max_sequence_length + 1, seq_step):
                bpr_s = math.ceil(s / block_size_tokens)
                _emit_data(
                    True,
                    "kv_curve",
                    avg_seq_len=s,
                    blocks_per_request=bpr_s,
                    kv_bytes_per_request=bpr_s * profile.block_size_bytes,
                    mamba_bytes_per_request=profile.mamba_memory_per_request,
                    total_cache_bytes_per_request=(
                        bpr_s * profile.block_size_bytes + profile.mamba_memory_per_request
                    ),
                )
        _emit_context_record(rank == 0, "tuned", self.context)

        if rank == 0:
            logging.info("Autotune: tuned context:\n%s", self.context.memory_report())
            free_pre_cg, total_pre_cg = torch.cuda.mem_get_info()
            logging.info(
                "Autotune: before CG build: GPU total %.1f GB, "
                "used %.1f GB, free %.1f GB",
                total_pre_cg / (1024 ** 3),
                (total_pre_cg - free_pre_cg) / (1024 ** 3),
                free_pre_cg / (1024 ** 3),
            )
            logging.info(
                "\n"
                "========== Autotune Complete ==========\n"
                "  max_requests:    %d\n"
                "  max_tokens:      %d\n"
                "  buffer_size_gb:  %.2f\n"
                "\n"
                "To reproduce without autotune, replace\n"
                "  --inference-dynamic-batching-autotune --inference-dynamic-batching-autotune-average-seq-len ...\n"
                "with:\n"
                "  --inference-dynamic-batching-max-requests %d \\\n"
                "  --inference-dynamic-batching-max-tokens %d \\\n"
                "  --inference-dynamic-batching-buffer-size-gb %.2f\n"
                "=======================================",
                new_max_requests,
                new_max_tokens,
                new_buffer_size_gb,
                new_max_requests,
                new_max_tokens,
                new_buffer_size_gb,
            )

    # ---- helpers -------------------------------------------------------------

    def _release_context(self) -> None:
        """Actually free the current context's GPU memory.

        The autotune bootstrap and profiling contexts never use TMS
        (``static_kv_memory_pointers=False``, PERSIST management mode), so
        their buffers are plain caching-allocator tensors:
        ``_force_free_context_tensors`` cannot release them — they free only
        once every live reference is dropped. This clears the engine's and the
        model wrapper's context references plus the controller's context-sized
        sampling state (``_all_logits_cuda`` alone is ``max_logits × vocab`` —
        gigabytes at profiling size), collects, and returns the cached
        segments to the driver.

        The caller must install a new context and re-run
        ``controller._init_dynamic_sampling_tensors()`` before the next
        forward pass.
        """
        controller = self.controller
        context = self.context
        # TMS-backed contexts (not the autotune-internal ones) release their
        # physical pages here; also drops any pinned-host backups.
        self._force_free_context_tensors(context)
        self.context = None
        controller.inference_wrapped_model.inference_context = None
        controller._all_logits_cuda = None
        controller._sampled_tokens_cuda = None
        controller._async_sched_sample_values_cuda = None
        controller._sampling = None
        # EP>1 nvls MoE: the dispatcher's symmetric staging buffers are
        # class-level singletons sized from this context's max_tokens. Release
        # them so the re-measured free memory excludes them; the next context
        # re-creates them at its own size, which the solver charges via
        # required_buffer_bytes(). Safe collectively: every rank releases and
        # re-allocates in lockstep because all context sizes are all_reduced.
        if context._nvls_dispatcher:
            NVLSAllGatherVDispatcher.release_buffers()
        del context
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _force_free_context_tensors(context):
        """Free GPU memory held by a context's TMS region.

        Pauses the context's TMS tag (freeing physical pages). The MemPool
        entries are left intact to avoid corrupting TMS's internal state:
        freed virtual address reservations are harmless (CUDA VA space is
        256 TB).

        Non-TMS tensors are left alone: they'll be garbage collected
        when all references to the context are dropped.
        """
        if context._uses_torch_memory_saver:
            from torch_memory_saver import torch_memory_saver as tms

            tms.pause(context.TMS_TAG)
            # The context is being discarded, never resumed: free any
            # pinned-host backups its paused allocations hold (OFFLOAD
            # mode would otherwise pin the buffer size in host memory
            # forever, since the segments stay cached in the allocator).
            # Feature-detected: older forks lack the API.
            if hasattr(tms, "drop_cpu_backups"):
                freed = tms.drop_cpu_backups(context.TMS_TAG)
                if freed:
                    logging.info(
                        "Autotune: dropped %.1f MB of pinned-host backups "
                        "for retired context tag %r",
                        freed / (1024 ** 2),
                        context.TMS_TAG,
                    )

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _sync_tuned_params_across_ranks(
        max_requests: int, max_tokens: int, buffer_size_gb: float,
    ) -> Tuple[int, int, float]:
        """Sync tuned parameters across ranks via all_reduce(MIN).

        Returns the original values unchanged if distributed is not
        initialised.
        """
        if not torch.distributed.is_initialized():
            return max_requests, max_tokens, buffer_size_gb
        buffer_size_bytes_int = int(buffer_size_gb * (1024 ** 3))
        sync_tensor = torch.tensor(
            [max_requests, max_tokens, buffer_size_bytes_int],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.MIN)
        return (
            int(sync_tensor[0].item()),
            int(sync_tensor[1].item()),
            int(sync_tensor[2].item()) / (1024 ** 3),
        )
