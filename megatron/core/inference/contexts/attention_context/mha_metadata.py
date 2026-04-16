# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions

from .metadata_base import MetadataBase


class MHAMetadata(MetadataBase):
    """
    Metadata for MHA layer using flash-attention.
    """

    def __init__(
        self,
        max_requests: int,
        max_seqlen: int,
        *,
        query_lengths_buf: torch.Tensor,
        cu_query_seq_lengths_buf: torch.Tensor,
        kv_seq_lengths_buf: torch.Tensor,
        cu_kv_seq_lengths_buf: torch.Tensor,
        block_table_buf: torch.Tensor,
    ):
        super().__init__()
        self.max_bs = max_requests
        self.max_seqlen = max_seqlen

        self._query_lengths_buf = query_lengths_buf
        self._cu_query_seq_lengths_buf = cu_query_seq_lengths_buf
        self._kv_seq_lengths_buf = kv_seq_lengths_buf
        self._cu_kv_seq_lengths_buf = cu_kv_seq_lengths_buf
        self._block_table_buf = block_table_buf

        self._max_seqlen_q = 0
        self._max_seqlen_k = 0

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        real_request_count_gpu: torch.Tensor,
        arange_buf: torch.Tensor,
        padded_batch_dimensions: InferenceBatchDimensions,
        num_speculative_tokens: int = 0,
    ):
        """Copy source data into metadata buffers, compute derived values, and pad.

        All ops use fixed-shape slices with `real_request_count_gpu` and `arange_buf` as
        the real/padding boundary, so shapes and addresses are static.
        """
        pbs = padded_batch_dimensions.req_count

        if pbs > 0:
            mask = arange_buf[:pbs] < real_request_count_gpu

            self._query_lengths_buf[:pbs] = torch.where(mask, request_query_lengths[:pbs], 0)
            self._kv_seq_lengths_buf[:pbs] = torch.where(
                mask, request_kv_length_offsets[:pbs] + request_query_lengths[:pbs], 0
            )
            self._block_table_buf[:pbs] = torch.where(
                mask.unsqueeze(1), request_to_kv_block_ids[:pbs], -1
            )

            self._cu_query_seq_lengths_buf[0] = 0
            torch.cumsum(
                self._query_lengths_buf[:pbs],
                dim=0,
                out=self._cu_query_seq_lengths_buf[1 : pbs + 1],
            )
            self._cu_kv_seq_lengths_buf[0] = 0
            torch.cumsum(
                self._kv_seq_lengths_buf[:pbs], dim=0, out=self._cu_kv_seq_lengths_buf[1 : pbs + 1]
            )
        else:
            self._cu_query_seq_lengths_buf[0] = 0
            self._cu_kv_seq_lengths_buf[0] = 0

        if padded_batch_dimensions.prefill_req_count == 0:
            self._max_seqlen_q = num_speculative_tokens + 1
        else:
            # Make sure we will launch the prefill kernel for prefill graphs
            self._max_seqlen_q = max(2, padded_batch_dimensions.token_count)

        self._max_seqlen_k = self.max_seqlen

        self.state_data = {
            "query_lengths": self._query_lengths_buf[:pbs],
            "cu_query_seq_lengths": self._cu_query_seq_lengths_buf[: pbs + 1],
            "cu_kv_seq_lengths": self._cu_kv_seq_lengths_buf[: pbs + 1],
            "kv_seq_lengths": self._kv_seq_lengths_buf[:pbs],
            "block_table": self._block_table_buf[:pbs, :],
            "max_seqlen_q": self._max_seqlen_q,
            "max_seqlen_k": self._max_seqlen_k,
        }


class NonGraphedMHAMetadata(MHAMetadata):
    """
    Metadata for MHA layer using flash-attention without CUDA graphs.
    """

    def update(
        self,
        request_query_lengths: torch.Tensor,
        request_kv_length_offsets: torch.Tensor,
        request_to_kv_block_ids: torch.Tensor,
        real_request_count_gpu: torch.Tensor,
        arange_buf: torch.Tensor,
        padded_batch_dimensions: InferenceBatchDimensions,
        num_speculative_tokens: int = 0,
    ):
        """
        Args:
            request_query_lengths (Tensor): per-request query lengths.
            request_kv_length_offsets (Tensor): per-request KV length offsets.
            request_to_kv_block_ids (Tensor): per-request block-table rows.
            real_request_count_gpu (Tensor): GPU scalar with the unpadded request count.
            arange_buf (Tensor): Pre-allocated arange buffer.
            padded_batch_dimensions (InferenceBatchDimensions): Padded decode/prefill/token counts.
            num_speculative_tokens (int): Speculative-decode depth.
        """
        super().update(
            request_query_lengths,
            request_kv_length_offsets,
            request_to_kv_block_ids,
            real_request_count_gpu,
            arange_buf,
            padded_batch_dimensions,
            num_speculative_tokens,
        )
        query_lengths = self.state_data["query_lengths"]
        if len(query_lengths) > 0:
            self.state_data["max_seqlen_q"] = torch.max(query_lengths)
            self.state_data["max_seqlen_k"] = torch.max(self.state_data["kv_seq_lengths"])
        else:
            # Empty-batch fallback: stay tensor-valued so the caller's
            # post-forward resolution can unconditionally .item().
            self.state_data["max_seqlen_q"] = query_lengths.new_full((), num_speculative_tokens + 1)
            self.state_data["max_seqlen_k"] = query_lengths.new_full((), 1)
