# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for context-parallel helpers in megatron/rl/rl_utils.py.

These tests exercise _scatter_for_context_parallel and
_gather_logprobs_context_parallel without requiring a real distributed
environment by patching megatron.core.mpu.

The helpers implement the canonical Megatron-LM "zigzag" CP layout:
each rank receives chunks ``(r, 2*CP-r-1)`` of a 2*CP partition of the
sequence dim, matching what TE ring attention / RoPE-CP / Mamba-CP all
assume. A contiguous slice would silently corrupt those downstream paths.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers to build a fake MagicMock process group that satisfies isinstance checks.
# ---------------------------------------------------------------------------

def _make_fake_pg():
    pg = MagicMock()
    pg.__class__ = torch.distributed.ProcessGroup
    return pg


def _zigzag_chunks(t: torch.Tensor, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Reference implementation of the zigzag slice for use in expectations."""
    seq_len = t.shape[1]
    chunk_size = seq_len // (2 * cp_size)
    chunks = t.view(t.shape[0], 2 * cp_size, chunk_size, *t.shape[2:])
    a = chunks[:, cp_rank]
    b = chunks[:, 2 * cp_size - cp_rank - 1]
    return torch.cat([a, b], dim=1)


# ---------------------------------------------------------------------------
# Tests for _scatter_for_context_parallel
# ---------------------------------------------------------------------------

class TestScatterForContextParallel:
    """Test _scatter_for_context_parallel in isolation."""

    def _run(self, cp_size, cp_rank, batch=1, seq_len=8, vocab=4):
        """Run scatter for one (cp_size, cp_rank) pair and return outputs."""
        from megatron.rl.rl_utils import _scatter_for_context_parallel
        from megatron.core.packed_seq_params import PackedSeqParams

        tokens      = torch.arange(batch * seq_len).reshape(batch, seq_len)
        position_ids = torch.arange(batch * seq_len).reshape(batch, seq_len)
        cu = torch.tensor([0, seq_len], dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
            total_tokens=seq_len,
        )
        fake_pg = _make_fake_pg()

        with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
            mock_mpu.get_context_parallel_rank.return_value  = cp_rank
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            result = _scatter_for_context_parallel(tokens, position_ids, packed_seq_params, cp_size)

        return result, tokens, packed_seq_params

    # --- shape tests ----------------------------------------------------------

    def test_local_tokens_shape(self):
        (lt, lp, _, _), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert lt.shape == (1, 4), f"expected (1,4) got {lt.shape}"

    def test_local_position_ids_shape(self):
        (_, lp, _, _), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert lp.shape == (1, 4)

    def test_local_labels_shape(self):
        (_, _, _, ll), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert ll.shape == (1, 4)

    def test_cp4_shape(self):
        (lt, _, _, ll), _, _ = self._run(cp_size=4, cp_rank=2, seq_len=16)
        assert lt.shape == (1, 4)
        assert ll.shape == (1, 4)

    # --- value tests: zigzag layout ------------------------------------------

    def test_rank0_tokens_are_zigzag_chunks_0_and_3(self):
        """CP=2 rank 0 gets chunks 0 and 3 (first and last quarter)."""
        (lt, _, _, _), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        torch.testing.assert_close(lt, _zigzag_chunks(tokens, 2, 0))

    def test_rank1_tokens_are_zigzag_chunks_1_and_2(self):
        """CP=2 rank 1 gets chunks 1 and 2 (middle two quarters)."""
        (lt, _, _, _), tokens, _ = self._run(cp_size=2, cp_rank=1, seq_len=8)
        torch.testing.assert_close(lt, _zigzag_chunks(tokens, 2, 1))

    def test_cp4_rank2_tokens_are_zigzag_chunks_2_and_5(self):
        """CP=4 rank 2 gets chunks 2 and 5 of an 8-chunk partition."""
        (lt, _, _, _), tokens, _ = self._run(cp_size=4, cp_rank=2, seq_len=16)
        torch.testing.assert_close(lt, _zigzag_chunks(tokens, 4, 2))

    def test_labels_aligned_with_tokens_zigzag(self):
        """Local labels = zigzag(tokens_shifted) where tokens_shifted[i]=tokens[i+1]."""
        (_, _, _, ll), tokens, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        tokens_shifted = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
        torch.testing.assert_close(ll, _zigzag_chunks(tokens_shifted, 2, 0))

    def test_labels_aligned_with_tokens_zigzag_other_rank(self):
        (_, _, _, ll), tokens, _ = self._run(cp_size=2, cp_rank=1, seq_len=8)
        tokens_shifted = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
        torch.testing.assert_close(ll, _zigzag_chunks(tokens_shifted, 2, 1))

    def test_labels_are_contiguous(self):
        (_, _, _, ll), _, _ = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert ll.is_contiguous()

    # --- PackedSeqParams mutation test ----------------------------------------

    def test_cp_fields_set_on_copy(self):
        """cp_group, local_cp_size and cu_seqlens_*_padded must be set;
        original must be unchanged."""
        from megatron.core.packed_seq_params import PackedSeqParams
        (_, _, cp_params, _), _, orig = self._run(cp_size=2, cp_rank=0, seq_len=8)
        assert cp_params.local_cp_size == 2
        assert cp_params.cp_group is not None
        # THD CP path requires cu_seqlens_*_padded — set from cu_seqlens_* fallback.
        assert cp_params.cu_seqlens_q_padded is not None
        assert cp_params.cu_seqlens_kv_padded is not None
        # Original must not have been mutated.
        assert orig.local_cp_size is None
        assert orig.cp_group is None
        assert orig.cu_seqlens_q_padded is None
        assert orig.cu_seqlens_kv_padded is None

    def test_assertion_on_indivisible_seq_len(self):
        """Zigzag layout requires seq_len % (2*cp_size) == 0."""
        from megatron.rl.rl_utils import _scatter_for_context_parallel
        from megatron.core.packed_seq_params import PackedSeqParams
        # 9 is not divisible by 2*2=4
        tokens = torch.zeros(1, 9, dtype=torch.long)
        pos    = torch.zeros(1, 9, dtype=torch.long)
        cu     = torch.tensor([0, 9], dtype=torch.int32)
        psp    = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                 max_seqlen_q=9, max_seqlen_kv=9, total_tokens=9)
        fake_pg = _make_fake_pg()
        with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
            mock_mpu.get_context_parallel_rank.return_value  = 0
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            with pytest.raises(AssertionError, match="divisible"):
                _scatter_for_context_parallel(tokens, pos, psp, cp_size=2)

    def test_assertion_on_seq_len_divisible_by_cp_but_not_2cp(self):
        """seq_len=6, cp_size=2: 6 % 2 == 0 but 6 % 4 != 0 — must fail."""
        from megatron.rl.rl_utils import _scatter_for_context_parallel
        from megatron.core.packed_seq_params import PackedSeqParams
        tokens = torch.zeros(1, 6, dtype=torch.long)
        pos    = torch.zeros(1, 6, dtype=torch.long)
        cu     = torch.tensor([0, 6], dtype=torch.int32)
        psp    = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                 max_seqlen_q=6, max_seqlen_kv=6, total_tokens=6)
        fake_pg = _make_fake_pg()
        with patch('megatron.rl.rl_utils.mpu') as mock_mpu:
            mock_mpu.get_context_parallel_rank.return_value  = 0
            mock_mpu.get_context_parallel_group.return_value = fake_pg
            with pytest.raises(AssertionError, match="divisible"):
                _scatter_for_context_parallel(tokens, pos, psp, cp_size=2)


# ---------------------------------------------------------------------------
# Tests for _gather_logprobs_context_parallel
# ---------------------------------------------------------------------------

class TestGatherLogprobsContextParallel:
    """Test _gather_logprobs_context_parallel without a real dist backend.

    The gather inverts the zigzag scatter: each rank's local logprobs
    contain ``[chunk_r | chunk_{2*CP-r-1}]``; the gather splits each rank's
    slice in half and places them in their global chunk slots.
    """

    def _gather_no_grad(self, cp_size, local_logprobs_per_rank, calling_rank=0):
        """Simulate the no_grad gather by patching all_gather."""
        from megatron.rl.rl_utils import _gather_logprobs_context_parallel
        fake_pg = _make_fake_pg()

        def fake_all_gather(out_list, tensor, group):
            for i, t in enumerate(local_logprobs_per_rank):
                out_list[i].copy_(t)

        with (
            patch('megatron.rl.rl_utils.mpu') as mock_mpu,
            patch('torch.distributed.all_gather', side_effect=fake_all_gather),
        ):
            mock_mpu.get_context_parallel_group.return_value   = fake_pg
            mock_mpu.get_context_parallel_world_size.return_value = cp_size
            result = _gather_logprobs_context_parallel(
                local_logprobs_per_rank[calling_rank], no_grad=True
            )
        return result

    def test_shape_after_gather(self):
        """Output shape must be [batch, cp_size*local_size - 1]."""
        cp_size    = 2
        local_size = 4  # must be even so it splits into two chunks
        ranks = [torch.arange(local_size, dtype=torch.float32).unsqueeze(0) for _ in range(cp_size)]
        out = self._gather_no_grad(cp_size, ranks)
        assert out.shape == (1, cp_size * local_size - 1)

    def test_values_invert_zigzag(self):
        """CP=2: rank 0 has [chunk_0 | chunk_3], rank 1 has [chunk_1 | chunk_2].
        Output (before drop-last) must be chunk_0, chunk_1, chunk_2, chunk_3."""
        # 4 chunks of size 1 each → global seq_len = 4.
        # rank 0 holds chunks 0 and 3, rank 1 holds chunks 1 and 2.
        rank0 = torch.tensor([[10.0, 13.0]])  # [c0, c3]
        rank1 = torch.tensor([[11.0, 12.0]])  # [c1, c2]
        out = self._gather_no_grad(cp_size=2, local_logprobs_per_rank=[rank0, rank1])
        # global = [c0, c1, c2, c3] = [10, 11, 12, 13], drop last → [10, 11, 12]
        expected = torch.tensor([[10.0, 11.0, 12.0]])
        torch.testing.assert_close(out, expected)

    def test_cp4_shape(self):
        cp_size    = 4
        local_size = 4  # even → splits into two chunks of size 2
        ranks = [torch.zeros(1, local_size) for _ in range(cp_size)]
        out = self._gather_no_grad(cp_size, ranks)
        assert out.shape == (1, cp_size * local_size - 1)

    def test_cp4_invert_zigzag(self):
        """CP=4: rank r holds [chunk_r | chunk_{7-r}] of an 8-chunk seq."""
        # 8 chunks of size 1: chunk i contains the scalar i+100.
        # rank 0: [c0, c7]  rank 1: [c1, c6]  rank 2: [c2, c5]  rank 3: [c3, c4]
        rank0 = torch.tensor([[100.0, 107.0]])
        rank1 = torch.tensor([[101.0, 106.0]])
        rank2 = torch.tensor([[102.0, 105.0]])
        rank3 = torch.tensor([[103.0, 104.0]])
        out = self._gather_no_grad(cp_size=4, local_logprobs_per_rank=[rank0, rank1, rank2, rank3])
        # global = [c0..c7] = [100..107], drop last → [100..106]
        expected = torch.tensor([[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]])
        torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# Tests that verify _scatter + manual computation == reference logprobs
# ---------------------------------------------------------------------------

class TestScatterGatherEquivalence:
    """Verify that scattering tokens (zigzag), computing logprobs per rank,
    then gathering (zigzag-inverted), gives the same result as the
    single-rank reference computation."""

    @staticmethod
    def _reference_logprobs(logits, tokens):
        """selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])."""
        from megatron.rl.rl_utils import selective_log_softmax
        return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])

    @staticmethod
    def _cp_logprobs(logits, tokens, cp_size):
        """Simulate the CP path: zigzag-scatter, compute per rank,
        zigzag-invert-gather (no actual distributed runtime)."""
        from megatron.rl.rl_utils import selective_log_softmax, _zigzag_slice
        seq_len = tokens.shape[1]
        # Pre-shifted labels (matches _scatter_for_context_parallel).
        tokens_shifted = torch.cat([tokens[:, 1:], tokens[:, -1:]], dim=1)
        per_rank_lp = []
        for cp_rank in range(cp_size):
            # Zigzag-slice logits along seq dim — note logits is [B, S, V],
            # but _zigzag_slice operates on the seq dim (dim=1).
            local_logits = _zigzag_slice(logits, cp_size, cp_rank)
            local_labels = _zigzag_slice(tokens_shifted, cp_size, cp_rank)
            per_rank_lp.append(selective_log_softmax(local_logits, local_labels))
        # Invert zigzag: each rank's lp is [chunk_r | chunk_{2*CP-r-1}].
        chunk_size = per_rank_lp[0].shape[1] // 2
        chunks = [None] * (2 * cp_size)
        for r in range(cp_size):
            chunks[r]                  = per_rank_lp[r][:, :chunk_size]
            chunks[2 * cp_size - r - 1] = per_rank_lp[r][:, chunk_size:]
        return torch.cat(chunks, dim=1)[:, :-1]

    def test_cp2_matches_reference(self):
        torch.manual_seed(0)
        batch, seq_len, vocab = 1, 8, 16
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp  = self._cp_logprobs(logits, tokens, cp_size=2)
        torch.testing.assert_close(ref, cp)

    def test_cp4_matches_reference(self):
        torch.manual_seed(42)
        batch, seq_len, vocab = 2, 16, 32
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp  = self._cp_logprobs(logits, tokens, cp_size=4)
        torch.testing.assert_close(ref, cp)

    def test_cp2_with_boundary_spanning_sequence(self):
        """Sequence tokens that cross zigzag boundaries must still match reference."""
        torch.manual_seed(7)
        batch, seq_len, vocab = 1, 12, 8
        logits = torch.randn(batch, seq_len, vocab)
        tokens = torch.randint(0, vocab, (batch, seq_len))
        ref = self._reference_logprobs(logits, tokens)
        cp  = self._cp_logprobs(logits, tokens, cp_size=2)
        torch.testing.assert_close(ref, cp)
