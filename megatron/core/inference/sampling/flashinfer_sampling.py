# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""FlashInfer-based sampling backend with CUDA graph support."""

from typing import Optional

import torch
from torch import Tensor

try:
    import flashinfer
except ImportError:
    flashinfer = None

from megatron.core.inference.sampling.base import Sampling
from megatron.core.inference.utils import CUDAGraphCache
from megatron.core.transformer.cuda_graphs import CudaGraphManager


class FlashInferSampling(Sampling):
    """Fused FlashInfer sampling with optional CUDA graph capture/replay.

    Unlike :class:`TorchSampling`, FlashInfer kernels accept per-row parameter
    tensors (temperature, top_k, top_p) directly, so no bucketing is required.

    Two kernel variants are used depending on a batch-level flag:

    - **Unfiltered**: :func:`flashinfer.sampling.sampling_from_probs` when no
      request in the batch uses top-k or top-p filtering.
    - **Filtered**: :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`
      when at least one request uses filtering.

    The flag is computed in :meth:`pre_forward_bookkeeping` and stored in
    pinned memory so it can be read on the CPU without a device sync.
    """

    def __init__(
        self, vocab_size: int, rng: torch.Generator, enable_cuda_graph: bool = False
    ) -> None:
        self._vocab_size = vocab_size
        self._rng = rng
        self._enable_cuda_graph = enable_cuda_graph
        self._sampling_cuda_graphs = CUDAGraphCache()
        self._fi_any_filtered_pinned = torch.empty(1, dtype=torch.bool, pin_memory=True)

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    def pre_forward_bookkeeping(self, context) -> None:
        active_request_count = context.total_request_count - context.paused_request_count
        n = active_request_count
        md = context.active_request_metadata

        # Batch-level decision: if any request uses top-k/top-p the whole
        # batch runs the filtered kernel.
        top_k = md["top_k"][:n]
        top_p = md["top_p"][:n]
        flag = ((top_k != 0) & (top_k < self._vocab_size)).any() | (
            (top_p != 0.0) & (top_p < 1.0)
        ).any()
        self._fi_any_filtered_pinned.copy_(flag, non_blocking=True)

    def sample(
        self,
        logits: Tensor,
        n: int,
        output: Tensor,
        context,
        *,
        eager: bool = False,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> None:
        md = context.active_request_metadata
        if token_to_request_index is None:
            temperature = md["temperature"][:n]
            top_k = md["top_k"][:n]
            top_p = md["top_p"][:n]
        else:
            temperature = md["temperature"][token_to_request_index]
            top_k = md["top_k"][token_to_request_index]
            top_p = md["top_p"][token_to_request_index]

        filtered = bool(self._fi_any_filtered_pinned.item())
        pool = CudaGraphManager.global_mempool if not eager else None
        for _ in self._sampling_cuda_graphs(n, filtered, pool=pool, eager=eager):
            if gather_indices is None:
                scaled = logits[:n].to(torch.float32, copy=True)
            else:
                scaled = logits[gather_indices[:n], :].to(torch.float32, copy=True)
            scaled.div_(temperature.unsqueeze(1))
            probs = torch.softmax(scaled, dim=-1)
            if filtered:
                top_k_kernel = top_k.masked_fill(top_k == 0, self._vocab_size)
                top_p_kernel = top_p.masked_fill(top_p == 0.0, 1.0)
                output[:n].copy_(
                    flashinfer.sampling.top_k_top_p_sampling_from_probs(
                        probs, top_k_kernel, top_p_kernel, generator=self._rng
                    )
                )
            else:
                output[:n].copy_(
                    flashinfer.sampling.sampling_from_probs(probs, generator=self._rng)
                )
