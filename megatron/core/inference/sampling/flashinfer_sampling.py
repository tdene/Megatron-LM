# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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

    Unlike `TorchSampling`, FlashInfer kernels accept per-row parameter
    tensors (temperature, top_k, top_p) directly, so no bucketing is required.
    """

    def __init__(
        self, vocab_size: int, rng: torch.Generator, enable_cuda_graph: bool = False
    ) -> None:
        self._vocab_size = vocab_size
        self._rng = rng
        self._enable_cuda_graph = enable_cuda_graph
        self._sampling_cuda_graphs = CUDAGraphCache()
        self._verification_cuda_graphs = CUDAGraphCache()

    def pre_forward_bookkeeping(self, context) -> None:
        """No-op; FlashInfer needs no per-step bookkeeping."""

    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        output: Tensor,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> None:
        """FlashInfer fused top-k / top-p sampling kernel.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            output: Destination buffer for sampled token ids.
            context: The active DynamicInferenceContext.
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping; when set,
                sampling parameters are gathered per-token instead of per-request.
        """
        md = context.active_request_metadata
        if token_to_request_index is None:
            temperature = md["temperature"][:n]
            top_k = md["top_k"][:n]
            top_p = md["top_p"][:n]
        else:
            temperature = md["temperature"][token_to_request_index]
            top_k = md["top_k"][token_to_request_index]
            top_p = md["top_p"][token_to_request_index]

        if gather_indices is None:
            scaled = logits[:n].to(torch.float32, copy=True)
        else:
            scaled = logits[gather_indices[:n], :].to(torch.float32, copy=True)
        scaled.div_(temperature.unsqueeze(1))
        probs = torch.softmax(scaled, dim=-1)

        # Sentinel values disable filtering: top_k=vocab_size keeps all
        # tokens, top_p=1.0 keeps the full probability mass.
        top_k_safe = top_k.masked_fill(top_k == 0, self._vocab_size)
        top_p_safe = top_p.masked_fill(top_p == 0.0, 1.0)
        output[:n].copy_(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k_safe, top_p_safe, generator=self._rng
            )
        )

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
        """Sample tokens, with CUDA graph capture/replay.

        Wraps `sample_kernel` inside a `CUDAGraphCache` keyed on `n`.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            output: Destination buffer for sampled token ids.
            context: The active DynamicInferenceContext.
            eager: If True, skip CUDA graph capture/replay.
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping;
                when set, sampling parameters are gathered per-token instead of per-request.
        """
        pool = CudaGraphManager.global_mempool if not eager else None
        for _ in self._sampling_cuda_graphs(n, pool=pool, eager=eager):
            self.sample_kernel(
                logits,
                n,
                output,
                context,
                gather_indices=gather_indices,
                token_to_request_index=token_to_request_index,
            )

    def sample_and_verify(
        self,
        logits: Tensor,
        input_ids: Tensor,
        num_speculative_tokens: int,
        num_decode: int,
        num_prefill: int,
        context,
        *,
        eager: bool = False,
        sampled_tokens: Tensor,
        last_accepted_indices: Tensor,
        accepted_tokens: Tensor,
        accepted_counts: Tensor,
    ) -> None:
        """Sample and verify speculative tokens, with CUDA graph capture/replay.

        Wraps :meth:`sample_and_verify_kernel` inside a `CUDAGraphCache`
        keyed on `(num_decode, num_prefill)`.

        Args:
            logits: Raw model logits, shape `[1, seq_len, vocab_size]`.
            input_ids: Input token ids fed to the model,
                shape `[1, seq_len]`.
            num_speculative_tokens: Number of speculative tokens per decode
                request (S).
            num_decode: Number of active decode requests.
            num_prefill: Number of active prefill requests.
            context: The active DynamicInferenceContext.
            eager: If True, skip CUDA graph capture/replay.
            sampled_tokens: Output buffer for the final sampled token per
                request.
            last_accepted_indices: Output buffer for the logit index of each
                request's last accepted position.
            accepted_tokens: Output buffer for accepted speculative tokens.
            accepted_counts: Output buffer for per-request accepted counts.
        """
        pool = CudaGraphManager.global_mempool if not eager else None
        for _ in self._verification_cuda_graphs(
            num_decode, num_prefill, pool=pool, eager=eager
        ):
            self.sample_and_verify_kernel(
                logits,
                input_ids,
                num_speculative_tokens,
                num_decode,
                num_prefill,
                context,
                sampled_tokens=sampled_tokens,
                last_accepted_indices=last_accepted_indices,
                accepted_tokens=accepted_tokens,
                accepted_counts=accepted_counts,
            )
