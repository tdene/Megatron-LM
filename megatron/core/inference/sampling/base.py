# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Base class for inference sampling backends."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Abstract base for inference sampling.

    Subclasses implement :meth:`sample` (the raw sampling kernel) and
    :meth:`pre_forward_bookkeeping`.  CUDA graph management, if any, is
    internal to the subclass.
    """

    @abstractmethod
    def pre_forward_bookkeeping(self, context) -> None:
        """Prepare sampling state before the forward pass."""
        ...

    @abstractmethod
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
        """Sample ``n`` tokens from ``logits`` into ``output[:n]``.

        Args:
            logits: Logits tensor of shape ``[>=n, vocab_size]``.
            n: Number of rows to sample.
            output: Destination buffer for sampled token ids.
            context: The active :class:`DynamicInferenceContext`.
            eager: If True, skip CUDA graph capture/replay.
            gather_indices: If provided, gather ``logits[gather_indices[:n], :]``
                before sampling.
            token_to_request_index: Per-token request mapping used by the
                speculative path to expand per-request params to per-token.
        """
        ...

    # ------------------------------------------------------------------
    # Speculative decoding
    # ------------------------------------------------------------------

    def sample_speculative(
        self,
        required_logits: Tensor,
        num_decode_requests: int,
        num_prefill_requests: int,
        num_speculative_tokens: int,
        context,
    ) -> Tensor:
        """Sample tokens from speculative logits.

        Builds a token-to-request mapping from the known decode/prefill counts
        then delegates to :meth:`sample`.

        Returns:
            output_tokens with shape ``[num_decode * (1+S) + num_prefill]``.
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
        self.sample(
            required_logits,
            num_tokens,
            output_tokens,
            context,
            eager=True,
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

        Returns:
            last_one_indices: Index of the last accepted token per request.
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
