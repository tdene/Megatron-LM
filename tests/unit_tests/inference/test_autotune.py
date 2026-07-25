# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from megatron.core.inference.config import CudaGraphSizingDistribution, InferenceConfig
from megatron.core.inference.engines.autotune_engine import (
    AutotuneDynamicInferenceEngine,
    AutotuneProfile,
)
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

GB = 1024 ** 3
MB = 1024 ** 2


def _make_profile(
    gpu_total_gb=80,
    model_gb=30,
    block_size_bytes=512 * 1024,
    max_kv_block_count=80,
    mamba_memory_per_request=0,
    per_request_metadata_bytes=400,
    per_token_metadata_bytes=48,
    runtime_overhead_per_request=1 * MB,
    prefill_samples=None,
    decode_samples=None,
    nvls_ep_size=0,
    nvls_topk=0,
    nvls_hidden_size=0,
):
    """Build an AutotuneProfile with synthetic data."""
    p = AutotuneProfile(
        gpu_total_bytes=int(gpu_total_gb * GB),
        memory_after_model_load_bytes=int(model_gb * GB),
        block_size_bytes=block_size_bytes,
        max_kv_block_count=max_kv_block_count,
        mamba_memory_per_request=mamba_memory_per_request,
        per_request_metadata_bytes=per_request_metadata_bytes,
        per_token_metadata_bytes=per_token_metadata_bytes,
        runtime_overhead_per_request=runtime_overhead_per_request,
        nvls_ep_size=nvls_ep_size,
        nvls_topk=nvls_topk,
        nvls_hidden_size=nvls_hidden_size,
    )
    for tc, peak in (prefill_samples or [(32, 32 * MB), (256, 256 * MB)]):
        p.add_prefill_sample(tc, peak)
    if decode_samples:
        for tc, peak in decode_samples:
            p.add_decode_sample(tc, peak)
    return p


def _expected_max_tokens(max_requests, spec_factor=1, token_rounder=64):
    """max_tokens is derived: the decode envelope, token-rounder aligned."""
    return math.ceil(max_requests * spec_factor / token_rounder) * token_rounder


def _context_derive(profile, buffer_size_gb, paused_buffer_size_gb, max_requests):
    """Independent mirror of DynamicInferenceContext's block-count derivation
    for a tuned config (max_requests set, mamba_memory_ratio=None).

    Returns (block_count, paused_block_count) — the runtime allocator's
    total_count and paused quota at unified_memory_level == 0.
    """
    bs = profile.block_size_bytes
    buffer_bytes = int(buffer_size_gb * GB)
    paused_bytes = int(paused_buffer_size_gb * GB)
    mamba_needed = max_requests * profile.mamba_memory_per_request
    if mamba_needed > 0:
        total = buffer_bytes + paused_bytes
        assert mamba_needed < total, "tuned config must leave room for mamba states"
        ratio = mamba_needed / total
        buffer_bytes = int(buffer_bytes * (1.0 - ratio))
        paused_bytes = int(paused_bytes * (1.0 - ratio))
    return buffer_bytes // bs, paused_bytes // bs


def _actual_physical_bytes(
    profile, max_requests, max_tokens, buffer_size_gb, spec_factor=1, mamba_prefix_cache_bytes=0,
    tp_size=1,
):
    """Sum of GPU bytes the solver commits to at runtime.

    Note: mamba_memory_per_request is folded into buffer_size_gb (the
    context carves mamba states out of the buffer), so it must NOT be
    counted separately here — doing so would double-count.
    """
    prefill_table = AutotuneProfile._build_activation_table(
        profile.prefill_token_counts, profile.prefill_peak_activation_bytes,
    )
    decode_table = AutotuneProfile._build_activation_table(
        profile.decode_token_counts, profile.decode_peak_activation_bytes,
    )
    prefill_cg = (
        max(0, AutotuneProfile._interpolate(prefill_table, max_tokens)) if prefill_table else 0
    )
    decode_cg = (
        max(0, AutotuneProfile._interpolate(decode_table, max_requests * spec_factor))
        if decode_table
        else 0
    )
    cg_pool = max(prefill_cg, decode_cg)
    runtime_overhead = max_requests * profile.runtime_overhead_per_request
    per_request_cost = max_requests * profile.per_request_metadata_bytes
    per_token_cost = max_tokens * profile.per_token_metadata_bytes
    buffer_bytes = buffer_size_gb * GB
    dispatcher_bytes = 0
    if profile.nvls_ep_size > 0:
        dispatcher_bytes = NVLSAllGatherVDispatcher.required_buffer_bytes(
            per_rank_worst_case_token_count=max_tokens // tp_size,
            topk=profile.nvls_topk,
            hidden_size=profile.nvls_hidden_size,
            ep_size=profile.nvls_ep_size,
        )
    return (
        cg_pool
        + runtime_overhead
        + per_request_cost
        + per_token_cost
        + buffer_bytes
        + mamba_prefix_cache_bytes
        + dispatcher_bytes
    )


class TestAutotuneSolver:
    """Tests for the autotune solver (interpolation, solve, pin, error paths)."""

    @pytest.mark.parametrize(
        "table, x, expected",
        [
            ({10: 100, 20: 200}, 10, 100),
            ({10: 100, 20: 200}, 15, 150),
            ({10: 100, 20: 200}, 5, 50),
            ({10: 100, 20: 200}, 30, 300),
            ({10: 100}, 5, 100),
            ({10: 100}, 20, 100),
            ({10: 100, 20: 200, 40: 600}, 30, 400),
            ({10: 100, 20: 200}, -100, 0),
        ],
    )
    def test_interpolation(self, table, x, expected):
        assert AutotuneProfile._interpolate(table, x) == expected

    @pytest.mark.parametrize(
        "desc, profile_kwargs, solver_kwargs",
        [
            ("baseline_80gb", {}, {}),
            ("small_gpu_40gb", dict(gpu_total_gb=40), {}),
            ("large_gpu_160gb", dict(gpu_total_gb=160), {}),
            ("with_mamba", dict(mamba_memory_per_request=4 * MB), {}),
            ("heavy_mamba", dict(mamba_memory_per_request=16 * MB), {}),
            ("with_decode", dict(decode_samples=[(32, 16 * MB), (128, 64 * MB), (256, 128 * MB)]),
             {}),
            ("tp8_alignment", {}, dict(tp_size=8)),
            ("short_sequences", {}, dict(blocks_per_request=4)),
            ("long_sequences", {}, dict(blocks_per_request=64)),
            ("speculative", {}, dict(num_speculative_tokens=2)),
            ("paused_quota", {}, dict(paused_buffer_size_bytes=2 * GB)),
            ("mamba_prefix_cache", dict(mamba_memory_per_request=4 * MB),
             dict(prefix_caching_mamba_bytes=2 * GB)),
            ("mamba_plus_paused", dict(mamba_memory_per_request=4 * MB),
             dict(paused_buffer_size_bytes=2 * GB)),
            ("nvls_dispatcher", dict(nvls_ep_size=8, nvls_topk=8, nvls_hidden_size=2048), {}),
            ("nvls_dispatcher_tp8", dict(nvls_ep_size=4, nvls_topk=8, nvls_hidden_size=2048),
             dict(tp_size=8)),
        ],
    )
    def test_solver_constraints(self, desc, profile_kwargs, solver_kwargs):
        """The solved output satisfies every structural constraint."""
        profile = _make_profile(**profile_kwargs)
        kwargs = dict(blocks_per_request=16)
        kwargs.update(solver_kwargs)
        r, t, b = profile.compute_optimal_params(**kwargs)

        tp_size = kwargs.get("tp_size", 1)
        spec_factor = 1 + kwargs.get("num_speculative_tokens", 0)
        blocks_per_request = kwargs["blocks_per_request"]
        alignment = max(tp_size, 4)
        tr = math.ceil(64 / tp_size) * tp_size

        # Alignment and derivation.
        assert r > 0 and r % alignment == 0, f"{desc}: max_requests misaligned"
        assert t == _expected_max_tokens(r, spec_factor, tr), f"{desc}: max_tokens not derived"

        # The runtime context must re-derive enough blocks: quota + sentinel +
        # the provisioned per-request blocks. A requested paused quota must
        # survive the derivation as a positive block count.
        paused_gb = kwargs.get("paused_buffer_size_bytes", 0) / GB
        block_count, paused_blocks = _context_derive(profile, b, paused_gb, r)
        assert block_count - paused_blocks - 1 >= r * blocks_per_request, (
            f"{desc}: active blocks {block_count - paused_blocks - 1} < provisioned "
            f"{r * blocks_per_request}"
        )
        if paused_gb > 0:
            assert paused_blocks > 0, f"{desc}: paused quota lost in the derivation"

        # Physical commitment fits the budget.
        gpu_free = profile.gpu_total_bytes - profile.memory_after_model_load_bytes
        total_used = _actual_physical_bytes(
            profile, r, t, b, spec_factor,
            mamba_prefix_cache_bytes=kwargs.get("prefix_caching_mamba_bytes", 0),
            tp_size=tp_size,
        )
        assert total_used <= gpu_free * 1.01, (
            f"{desc}: total {total_used / GB:.2f} GB exceeds budget {gpu_free / GB:.2f} GB"
        )

    @pytest.mark.parametrize(
        "desc, profile_small, kwargs_small, profile_large, kwargs_large",
        [
            ("longer_sequences_reduce_r",
             {}, dict(blocks_per_request=64), {}, dict(blocks_per_request=4)),
            ("speculation_reduces_r",
             {}, dict(blocks_per_request=16, num_speculative_tokens=3),
             {}, dict(blocks_per_request=16)),
            ("paused_quota_reduces_r",
             {}, dict(blocks_per_request=16, paused_buffer_size_bytes=8 * GB),
             {}, dict(blocks_per_request=16)),
            ("mamba_prefix_cache_reduces_r",
             {}, dict(blocks_per_request=16, prefix_caching_mamba_bytes=8 * GB),
             {}, dict(blocks_per_request=16)),
            ("decode_floor_reduces_r",
             dict(decode_samples=[(64, 2 * GB), (256, 4 * GB)]), dict(blocks_per_request=16),
             {}, dict(blocks_per_request=16)),
            ("nvls_buffers_reduce_r",
             dict(nvls_ep_size=8, nvls_topk=8, nvls_hidden_size=2048),
             dict(blocks_per_request=16), {}, dict(blocks_per_request=16)),
        ],
    )
    def test_monotonicity(self, desc, profile_small, kwargs_small, profile_large, kwargs_large):
        """Every additional cost (solver input or profile shape) must shrink
        (or keep) the solved max_requests."""
        r_small, _, _ = _make_profile(**profile_small).compute_optimal_params(**kwargs_small)
        r_large, _, _ = _make_profile(**profile_large).compute_optimal_params(**kwargs_large)
        assert r_small <= r_large, f"{desc}: {r_small} > {r_large}"

    def test_duplicate_samples_keep_max(self):
        """Duplicate token counts collapse to the max peak. (Note: this is a
        table-construction property; at the solve level a higher interior
        sample can flatten the extrapolation slope and legally *increase*
        the solved max_requests.)"""
        table = AutotuneProfile._build_activation_table(
            [128, 128, 256], [200 * MB, 100 * MB, 300 * MB]
        )
        assert table == {128: 200 * MB, 256: 300 * MB}

    def test_pin(self):
        """An explicit max_requests pins the solve: aligned down to the request
        rounder and honored even when the solver could go higher."""
        profile = _make_profile()
        r_solved, _, _ = profile.compute_optimal_params(blocks_per_request=16)
        r, t, _ = profile.compute_optimal_params(blocks_per_request=16, pinned_max_requests=130)
        assert (r, t) == (128, _expected_max_tokens(128))
        assert r < r_solved

    def test_no_decode_samples_gives_zero_floor(self):
        """Without decode samples, the solver behaves as if decode cost is 0."""
        profile_a = _make_profile()  # no decode samples
        profile_b = _make_profile(decode_samples=[(128, 0)])  # decode samples with 0 cost
        r_a, t_a, b_a = profile_a.compute_optimal_params(blocks_per_request=16)
        r_b, t_b, b_b = profile_b.compute_optimal_params(blocks_per_request=16)
        assert (r_a, t_a) == (r_b, t_b)
        assert abs(b_a - b_b) < 0.01

    @pytest.mark.parametrize(
        "desc, make, solver_kwargs, match",
        [
            ("pinned_infeasible",
             lambda: _make_profile(gpu_total_gb=32, model_gb=30),
             dict(blocks_per_request=16, pinned_max_requests=100_000), "pinned"),
            ("minimum_infeasible",
             lambda: _make_profile(gpu_total_gb=30.5, model_gb=30),
             dict(blocks_per_request=100_000), "not enough GPU memory for even"),
            ("zero_block_size",
             lambda: _make_profile(block_size_bytes=0),
             dict(blocks_per_request=16), "block_size_bytes must be positive"),
            ("empty_profile",
             lambda: AutotuneProfile(
                 gpu_total_bytes=80 * GB, memory_after_model_load_bytes=30 * GB
             ),
             dict(blocks_per_request=16), "No profiling data"),
        ],
    )
    def test_error_paths(self, desc, make, solver_kwargs, match):
        """Infeasible or malformed inputs raise instead of degrading."""
        with pytest.raises(ValueError, match=match):
            make().compute_optimal_params(**solver_kwargs)


class TestAutotuneRequirements:
    """Tests for the engine-side option validation — the single enforcement
    point for the requirement list on `InferenceConfig.autotune`."""

    @staticmethod
    def _validate(config_overrides=None, model_overrides=None, non_decode_graphs_derived=True):
        """Run the validator against a compliant baseline plus overrides."""
        config = InferenceConfig(
            autotune=True,
            autotune_average_seq_len=2048,
            num_cuda_graphs=16,
            cuda_graph_all_prefills=True,
            enable_chunked_prefill=True,
        )
        if config_overrides:
            config = replace(config, **config_overrides)
        model_fields = dict(inference_moe_token_dispatcher_type='nvls', cuda_graph_impl='local')
        model_fields.update(model_overrides or {})
        model_config = SimpleNamespace(**model_fields)
        # The validator reads the derived non-decode-graphs flag off the
        # engine's (bootstrap) context; stub just that.
        engine = SimpleNamespace(
            context=SimpleNamespace(
                use_cuda_graphs_for_non_decode_steps=non_decode_graphs_derived
            )
        )
        AutotuneDynamicInferenceEngine._validate_autotune_requirements(
            engine, model_config, config
        )

    def test_compliant_config_passes(self):
        self._validate()

    @pytest.mark.parametrize(
        "desc, config_overrides, model_overrides, non_decode_derived, match",
        [
            ("missing_avg_seq_len", dict(autotune_average_seq_len=None), {}, True,
             "autotune_average_seq_len"),
            ("zero_avg_seq_len", dict(autotune_average_seq_len=0), {}, True,
             "autotune_average_seq_len"),
            ("nccl_dispatcher", {}, dict(inference_moe_token_dispatcher_type='nccl'), True,
             "nvls"),
            ("non_decode_graphs_disabled", dict(use_cuda_graphs_for_non_decode_steps=False),
             {}, False, "use_cuda_graphs_for_non_decode_steps"),
            ("non_decode_graphs_force_disabled", {}, {}, False, "force-disabled"),
            ("no_all_prefills", dict(cuda_graph_all_prefills=False), {}, True,
             "cuda_graph_all_prefills"),
            ("wrong_graph_impl", {}, dict(cuda_graph_impl='none'), True, "cuda_graph_impl"),
            ("graphs_disabled", dict(num_cuda_graphs=None), {}, True, "num_cuda_graphs"),
            ("zero_mixed_prefill_count", dict(cuda_graph_mixed_prefill_count=0), {}, True,
             "cuda_graph_mixed_prefill_count"),
            ("linear_sizing",
             dict(cuda_graph_sizing_distribution=CudaGraphSizingDistribution.LINEAR), {}, True,
             "EXPONENTIAL"),
            ("no_chunked_prefill", dict(enable_chunked_prefill=False), {}, True,
             "enable_chunked_prefill"),
            ("unified_memory", dict(unified_memory_level=1), {}, True, "unified_memory_level"),
        ],
    )
    def test_each_requirement_enforced(
        self, desc, config_overrides, model_overrides, non_decode_derived, match
    ):
        with pytest.raises(ValueError, match=match):
            self._validate(config_overrides, model_overrides, non_decode_derived)

    def test_violations_reported_together(self):
        """All unmet requirements surface in one error, not one at a time."""
        with pytest.raises(ValueError) as exc_info:
            self._validate(dict(
                autotune_average_seq_len=None,
                cuda_graph_all_prefills=False,
                enable_chunked_prefill=False,
                unified_memory_level=2,
            ))
        message = str(exc_info.value)
        for needle in (
            "autotune_average_seq_len",
            "cuda_graph_all_prefills",
            "enable_chunked_prefill",
            "unified_memory_level",
        ):
            assert needle in message


class TestNVLSBufferSizing:
    """Exact byte math for the dispatcher's symmetric-buffer sizing.

    The solver charges required_buffer_bytes() at every candidate max_tokens,
    so it must match what allocate_buffers reserves exactly (both read
    _buffer_specs)."""

    @pytest.mark.parametrize(
        "desc, per_rank, topk, hidden, ep, expected_mb",
        [
            # G = per_rank * ep; buffers: agv_h bf16 [G,H], agv_r int64 [G,K],
            # agv_p fp32 [G,K], rsv fp32 [G,H], meta int32 [ep]; each rounded
            # up to whole MiB with a 1 MiB floor.
            ("mib_exact", 1024, 8, 2048, 4, 16 + 1 + 1 + 32 + 1),
            ("mib_rounded", 1000, 3, 1000, 2, 4 + 1 + 1 + 8 + 1),
            ("per_buffer_floor", 1, 1, 1, 1, 5),
        ],
    )
    def test_required_buffer_bytes(self, desc, per_rank, topk, hidden, ep, expected_mb):
        assert (
            NVLSAllGatherVDispatcher.required_buffer_bytes(
                per_rank_worst_case_token_count=per_rank,
                topk=topk,
                hidden_size=hidden,
                ep_size=ep,
            )
            == expected_mb * MB
        ), desc

    def test_monotone_in_token_count(self):
        """The solver's search requires the cost term to be non-decreasing."""
        sizes = [
            NVLSAllGatherVDispatcher.required_buffer_bytes(
                per_rank_worst_case_token_count=p, topk=8, hidden_size=2048, ep_size=8
            )
            for p in (256, 1024, 4096, 16384)
        ]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
