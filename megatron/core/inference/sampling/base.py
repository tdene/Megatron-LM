# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Abstract base for inference sampling.

    Subclasses implement the raw sampling kernel (`sample_kernel`) and `pre_forward_bookkeeping`.
    CUDA graph management, if any, is handled by the subclass in the public entry points
    (`sample`, `sample_and_verify`), which wrap the kernel methods.
    """

    @abstractmethod
    def pre_forward_bookkeeping(self, context) -> None:
        """Prepare sampling state before the forward pass."""
        ...

    @abstractmethod
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
        """Pure sampling kernel; no CUDA-graph awareness.

        Subclasses implement backend-specific sampling here.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            output: Destination buffer for sampled token ids.
            context: The active DynamicInferenceContext.
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping; when set, sampling parameters are
                gathered per-token instead of per-request.
        """
        ...

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
        """Sample `n` tokens from `logits` into `output[:n]`.

        The base implementation delegates directly to `sample_kernel`.
        Subclasses that support CUDA graphs override this to wrap the kernel in a `CUDAGraphCache`.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            output: Destination buffer for sampled token ids.
            context: The active DynamicInferenceContext.
            eager: If True, skip CUDA graph capture/replay.
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping used by the speculative path.
        """
        self.sample_kernel(
            logits,
            n,
            output,
            context,
            gather_indices=gather_indices,
            token_to_request_index=token_to_request_index,
        )

    def sample_speculative(
        self,
        required_logits: Tensor,
        num_decode_requests: int,
        num_prefill_requests: int,
        num_speculative_tokens: int,
        context,
    ) -> Tensor:
        """Sample tokens from speculative logits.

        Builds a token-to-request mapping from the known decode/prefill, then calls `sample_kernel`.

        Args:
            required_logits: Logits for the required token positions,
                shape `[num_decode * (1+S) + num_prefill, vocab_size]`.
            num_decode_requests: Number of active decode requests.
            num_prefill_requests: Number of active prefill requests.
            num_speculative_tokens: Number of speculative tokens per decode request.
            context: The active DynamicInferenceContext.

        Returns:
            Sampled token ids with shape `[num_decode * (1+num_speculative_tokens) + num_prefill]`.
        """
        device = required_logits.device
        n_spec = num_speculative_tokens

        num_decode_tokens = num_decode_requests * (1 + n_spec)
        num_tokens = num_decode_tokens + num_prefill_requests
        token_to_request_index = torch.cat(
            [
                torch.arange(num_decode_requests, device=device).repeat_interleave(
                    1 + n_spec, output_size=num_decode_tokens
                ),
                torch.arange(
                    num_decode_requests, num_decode_requests + num_prefill_requests, device=device
                ),
            ]
        )
        output_tokens = torch.empty(num_tokens, device=device, dtype=torch.int64)
        self.sample_kernel(
            required_logits,
            num_tokens,
            output_tokens,
            context,
            token_to_request_index=token_to_request_index,
        )
        return output_tokens

    @staticmethod
    def verify_speculative_tokens(
        output_tokens: Tensor,
        input_tokens_required: Tensor,
        num_decode_requests: int,
        num_prefill_requests: int,
        active_request_count: int,
        num_speculative_tokens: int,
        accepted_tokens_buffer: Tensor,
        accepted_counts_buffer: Tensor,
    ) -> Tensor:
        """Verify speculative tokens against input tokens and compute acceptance.

        This is a pure kernel; no CUDA-graph awareness.

        Args:
            output_tokens: Sampled token ids from the base model,
                shape `[num_decode * (S+1) + num_prefill]`.
            input_tokens_required: Speculative input token ids at the same positions,
                shape `[num_decode * (S+1) + num_prefill]`.
            num_decode_requests: Number of active decode requests.
            num_prefill_requests: Number of active prefill requests.
            active_request_count: Total active requests (`num_decode + num_prefill`).
            num_speculative_tokens: Number of speculative tokens per decode request.
            accepted_tokens_buffer: Output buffer for accepted speculative tokens,
                shape `[max_requests, num_speculative_tokens]`.
                Filled with accepted token ids or -1.
            accepted_counts_buffer: Output buffer for per-request accepted counts,
                shape `[max_requests]`.

        Returns:
            Flat indices into the token sequence of the last accepted token per request,
                shape `[active_request_count]`.
        """
        if input_tokens_required.ndim == 2:
            input_tokens_required = input_tokens_required.squeeze(0)

        decode_len = num_decode_requests * (num_speculative_tokens + 1)

        # Decode verification (zero-row no-op when num_decode_requests == 0).
        decode_inputs = input_tokens_required[:decode_len].reshape(
            num_decode_requests, num_speculative_tokens + 1
        )
        decode_outputs = output_tokens[:decode_len].reshape(
            num_decode_requests, num_speculative_tokens + 1
        )
        decode_outputs_shifted = decode_outputs.roll(1, dims=1)
        decode_mask_2d = decode_inputs == decode_outputs_shifted
        decode_mask_2d[:, 0] = True
        decode_mask_2d = decode_mask_2d.cummin(dim=1).values

        last_one_indices = torch.zeros(
            active_request_count, device=input_tokens_required.device, dtype=torch.long
        )

        local_last_indices = decode_mask_2d.sum(dim=1) - 1
        row_offsets = torch.arange(num_decode_requests, device=last_one_indices.device) * (
            num_speculative_tokens + 1
        )
        last_one_indices[:num_decode_requests] = row_offsets + local_last_indices

        last_one_indices[num_decode_requests : num_decode_requests + num_prefill_requests] = (
            torch.arange(num_prefill_requests, device=last_one_indices.device) + decode_len
        )

        # Extract accepted tokens into static buffers.
        decode_accepted = decode_inputs.masked_fill(~decode_mask_2d, -1)
        accepted_tokens_buffer[:num_decode_requests, :] = decode_accepted[:, 1:]
        accepted_counts_buffer.copy_((accepted_tokens_buffer != -1).sum(dim=1))

        return last_one_indices

    def sample_and_verify_kernel(
        self,
        logits: Tensor,
        input_ids: Tensor,
        num_speculative_tokens: int,
        num_decode: int,
        num_prefill: int,
        context,
        *,
        sampled_tokens: Tensor,
        last_accepted_indices: Tensor,
        accepted_tokens: Tensor,
        accepted_counts: Tensor,
    ) -> None:
        """Pure sample-and-verify kernel; no CUDA-graph awareness.

        Selects required logit positions, calls `sample_speculative`, `verify_speculative_tokens`,
        and writes results into the provided output buffers.

        Args:
            logits: Raw model logits, shape `[1, seq_len, vocab_size]`.
            input_ids: Input token ids fed to the model,
                shape `[1, seq_len]`.
            num_speculative_tokens: Number of speculative tokens per decode request.
            num_decode: Number of active decode requests.
            num_prefill: Number of active prefill requests.
            context: The active DynamicInferenceContext.
            sampled_tokens: Output buffer for the final sampled token per
                request, shape `[max_requests]`.
            last_accepted_indices: Output buffer for the logit index of each
                request's last accepted position, shape `[max_requests]`.
            accepted_tokens: Output buffer for accepted speculative tokens,
                shape `[max_requests, num_speculative_tokens]`.
            accepted_counts: Output buffer for per-request accepted counts,
                shape `[max_requests]`.
        """
        active_request_count = num_decode + num_prefill
        device = logits.device

        decode_token_count = num_decode * (num_speculative_tokens + 1)
        decode_indices = torch.arange(decode_token_count, device=device)
        query_lengths = context.active_request_query_lengths[:active_request_count]
        cumsum = torch.cumsum(query_lengths, dim=0)
        prefill_last_indices = cumsum[num_decode:] - 1
        required_logit_indices = torch.cat([decode_indices, prefill_last_indices])

        if context.config.materialize_only_last_token_logits:
            required_logits = logits.squeeze(0)
        else:
            required_logits = logits.squeeze(0)[required_logit_indices, :]

        output_tokens = self.sample_speculative(
            required_logits, num_decode, num_prefill, num_speculative_tokens, context
        )

        input_tokens_required = input_ids[0, required_logit_indices]
        last_one_indices = self.verify_speculative_tokens(
            output_tokens,
            input_tokens_required,
            num_decode,
            num_prefill,
            active_request_count,
            num_speculative_tokens,
            accepted_tokens,
            accepted_counts,
        )

        sampled_tokens[:active_request_count] = output_tokens[last_one_indices]
        last_accepted_indices[:active_request_count] = required_logit_indices[last_one_indices]

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
        """Sample from speculative logits, verify, and store results.

        The base implementation delegates directly to `sample_and_verify_kernel`.
        Subclasses that support CUDA graphs override this to wrap the kernel in a `CUDAGraphCache`.

        Args:
            logits: Raw model logits, shape `[1, seq_len, vocab_size]`.
            input_ids: Input token ids fed to the model,
                shape `[1, seq_len]`.
            num_speculative_tokens: Number of speculative tokens per decode request.
            num_decode: Number of active decode requests.
            num_prefill: Number of active prefill requests.
            context: The active DynamicInferenceContext.
            eager: If True, skip CUDA graph capture/replay.
            sampled_tokens: Output buffer for the final sampled token per request,
                shape `[max_requests]`.
            last_accepted_indices: Output buffer for the index of each last accepted position,
                shape `[max_requests]`.
            accepted_tokens: Output buffer for accepted speculative tokens,
                shape `[max_requests, num_speculative_tokens]`.
            accepted_counts: Output buffer for per-request accepted counts,
                shape `[max_requests]`.
        """
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
