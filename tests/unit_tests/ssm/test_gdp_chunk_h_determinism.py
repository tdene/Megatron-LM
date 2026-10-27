# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Repeated-run race test for the in-tree GDP `fwd_h` kernel (`ops/gdp/chunk_h.py`).

`chunk_gated_delta_product_fwd_kernel_h_blockdim64` carries the recurrent state across
chunks; GDP prefill runs it through `chunk_gated_delta_product_varlen`.
On Blackwell its 4-warp autotune configs race (fla#945, triton-lang/triton#10590; seen
here with Triton 3.6): identical calls return different chunk states, off float64 by up
to ~1. The autotuner can pick those configs because they are fast, so `chunk_h.py` only
offers 2 warps on Blackwell.

The kernel has no atomics and no split reductions, so a correct config returns bitwise
identical outputs for identical inputs. These tests run it many times on fixed inputs at
Qwen3.5-9B-like shapes (32 value heads, 128x128 state, packed documents) and require
bitwise-identical reruns and a match to a float64 recurrence. The first test pins the kernel
to every config the autotuner may pick. The second leaves the choice to the autotuner, as in
serving.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE, HAVE_TRITON, IS_NVIDIA_BLACKWELL, RCP_LN2

if HAVE_TRITON:
    from megatron.core.ssm.ops.gdp.chunk_h import (
        chunk_gated_delta_product_fwd_h,
        chunk_gated_delta_product_fwd_kernel_h_blockdim64,
    )
    from megatron.core.ssm.ops.gdp.cumsum import chunk_local_cumsum
    from megatron.core.ssm.ops.gdp.scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from megatron.core.ssm.ops.gdp.solve_tril import solve_tril
    from megatron.core.ssm.ops.gdp.wy_fast import recompute_w_u_fwd

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not HAVE_TRITON, reason="needs Triton"),
]

# Qwen3.5-9B's linear-attention layers: 16 key heads repeated to 32 value heads, head dim 128.
NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 16, 32, 128
REPEATS = 30


def _autotuner(kernel):
    while not hasattr(kernel, "configs"):
        kernel = kernel.fn
    return kernel


def _doc_lengths(num_docs: int, total: int, seed: int) -> list[int]:
    """Packed-document lengths summing to `total`, most not multiples of the chunk size."""
    gen = torch.Generator().manual_seed(seed)
    cuts = torch.randperm(total - 1, generator=gen)[: num_docs - 1].add(1).sort().values.tolist()
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _prefill_inputs(doc_lengths: list[int], seed: int = 0):
    """fwd_h inputs for one Householder step: log2 chunk-cumsum gates and the WY factors w, u."""
    torch.manual_seed(seed)
    T, dev = sum(doc_lengths), "cuda"
    cu = torch.tensor([0] + doc_lengths, device=dev).cumsum(0).to(torch.int32)
    k = F.normalize(torch.randn(1, T, NUM_K_HEADS, HEAD_DIM, device=dev), p=2, dim=-1)
    k = k.to(torch.bfloat16).repeat_interleave(NUM_V_HEADS // NUM_K_HEADS, dim=2).contiguous()
    v = torch.randn(1, T, NUM_V_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(1, T, NUM_V_HEADS, device=dev).sigmoid().to(torch.bfloat16)
    g = F.logsigmoid(torch.randn(1, T, NUM_V_HEADS, device=dev))
    h0 = 0.1 * torch.randn(len(doc_lengths), NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=dev)
    g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, scale=RCP_LN2, cu_seqlens=cu)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, cu_seqlens=cu, chunk_size=CHUNK_SIZE)
    A = solve_tril(A=A, cu_seqlens=cu, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, cu_seqlens=cu)
    return dict(k=k, w=w, u=u, g=g, initial_state=h0, cu_seqlens=cu)


def _run_fwd_h(inputs):
    # As prefill with prefix caching: fp32 initial and per-chunk states, final state out.
    return chunk_gated_delta_product_fwd_h(
        **inputs,
        output_final_state=True,
        chunk_size=CHUNK_SIZE,
        num_householder=1,
        states_dtype=torch.float32,
    )


def _fwd_h_reference(k, w, u, g, initial_state, cu_seqlens):
    """Float64 fwd_h. Per chunk: h = S, v_new = u - wS, S = 2^g_last S + k^T 2^(g_last-g) v_new."""
    k, w, u, g = (x.double()[0].transpose(0, 1) for x in (k, w, u, g))
    h, v_new, ht = [], torch.zeros_like(u), []
    bounds = cu_seqlens.tolist()
    for i_n, (bos, eos) in enumerate(zip(bounds[:-1], bounds[1:])):
        S = initial_state[i_n].double()
        for s in range(bos, eos, CHUNK_SIZE):
            e = min(s + CHUNK_SIZE, eos)
            h.append(S)
            v_new[:, s:e] = u[:, s:e] - w[:, s:e] @ S
            g_last = g[:, e - 1, None]
            decayed = v_new[:, s:e] * torch.exp2(g_last - g[:, s:e])[..., None]
            S = S * torch.exp2(g_last)[..., None] + k[:, s:e].transpose(-1, -2) @ decayed
        ht.append(S)
    return torch.stack(h)[None], v_new.transpose(0, 1)[None], torch.stack(ht)


def _check_config(inputs, refs, repeats: int = REPEATS) -> str | None:
    """Run fwd_h `1 + repeats` times; describe any float64 mismatch or rerun difference."""
    first = _run_fwd_h(inputs)
    wrong, errs = [], []
    for name, out, ref in zip(("h", "v_new", "ht"), first, refs):
        err, scale = (out.double() - ref).abs().max().item(), ref.abs().max().item()
        errs.append(f"{name} {err:.3g}/{scale:.3g}")
        # Correct configs sit at the bf16 floor, ~2e-3 of the max; racing ones at 2e-2 to 0.7.
        if err > 0.02 * scale:
            wrong.append(f"{name} max err {err:.3g} (max |ref| {scale:.3g})")
    differ = 0
    for _ in range(repeats):
        differ += any(not torch.equal(x, y) for x, y in zip(_run_fwd_h(inputs), first))
    print(f"max err/max |ref|: {', '.join(errs)}; {differ}/{repeats} reruns differ")
    if wrong or differ:
        return f"{', '.join(wrong) or 'matches float64'}; {differ}/{repeats} reruns differ"
    return None


def test_fwd_h_offers_only_two_warps_on_blackwell():
    if not IS_NVIDIA_BLACKWELL:
        pytest.skip("the 2-warp pin only applies on Blackwell")
    configs = _autotuner(chunk_gated_delta_product_fwd_kernel_h_blockdim64).configs
    assert {config.num_warps for config in configs} == {2}


@pytest.mark.parametrize(
    "num_docs,total_tokens",
    [pytest.param(8, 4096, id="8docs-4k"), pytest.param(16, 16384, id="16docs-16k")],
)
def test_fwd_h_every_autotune_config_is_deterministic_and_exact(num_docs, total_tokens):
    inputs = _prefill_inputs(_doc_lengths(num_docs, total_tokens, seed=num_docs))
    refs = _fwd_h_reference(**inputs)
    tuner = _autotuner(chunk_gated_delta_product_fwd_kernel_h_blockdim64)
    configs, failures = list(tuner.configs), []
    try:
        for config in configs:
            # One config skips benchmarking, so this pins the kernel to it.
            tuner.configs, tuner.cache = [config], {}
            print(f"{config}: ", end="")
            problem = _check_config(inputs, refs)
            if problem:
                failures.append(f"{config}: {problem}")
    finally:
        tuner.configs, tuner.cache = configs, {}
    assert not failures, "\n".join(failures)


def test_fwd_h_autotuned_reruns_are_deterministic_and_exact():
    # The first call at 32 packed documents / 32k tokens autotuned to a racy 4-warp config
    # in serving before the pin (the choice is timing-based, so not on every run).
    inputs = _prefill_inputs(_doc_lengths(32, 32768, seed=32))
    refs = _fwd_h_reference(**inputs)
    tuner = _autotuner(chunk_gated_delta_product_fwd_kernel_h_blockdim64)
    tuner.cache = {}
    problem = _check_config(inputs, refs)
    print(f"autotuner picked: {tuner.best_config}")
    assert problem is None, f"autotuned {tuner.best_config}: {problem}"
