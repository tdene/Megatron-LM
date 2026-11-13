# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams


@dataclass(frozen=True)
class SharedPrefixPlan:
    """Geometry of the gathered attention call, unsharded, padded token order."""

    q_index: Tensor
    kv_index: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_kv: Tensor
    max_seqlen_q: int
    max_seqlen_kv: int
    start_positions: Tensor

    @property
    def num_real_tokens(self) -> int:
        """Number of rows attention computes an output for."""
        return int(self.q_index.numel())


def get_shared_prefix_plan(
    packed_seq_params: PackedSeqParams, device: torch.device | str | None = None
) -> SharedPrefixPlan:
    """Derive and validate the plan on the host."""
    if packed_seq_params.seq_prefix is None:
        raise ValueError("packed_seq_params has no seq_prefix")
    device = torch.device(device if device is not None else packed_seq_params.cu_seqlens_q.device)
    cache = packed_seq_params.seq_prefix_cache
    if isinstance(cache, SharedPrefixPlan) and cache.q_index.device == device:
        return cache

    real = packed_seq_params.cu_seqlens_q.tolist()
    padded = packed_seq_params.cu_seqlens_q_padded
    padded = real if padded is None else padded.tolist()
    prefix = packed_seq_params.seq_prefix.tolist()
    prefix_len = packed_seq_params.seq_prefix_len.tolist()
    halo = packed_seq_params.seq_prefix_halo
    n = len(real) - 1
    if len(padded) != n + 1 or len(prefix) != n or len(prefix_len) != n:
        raise ValueError("seq_prefix and seq_prefix_len need one entry per cu_seqlens_q segment")
    total = packed_seq_params.total_tokens
    if total is not None and padded[-1] > total:
        raise ValueError(f"padded layout ends at {padded[-1]} but the batch has {total} tokens")
    lengths = [real[s + 1] - real[s] for s in range(n)]
    for s in range(n):
        p, slot = prefix[s], padded[s + 1] - padded[s]
        if not 0 <= lengths[s] <= slot:
            raise ValueError(f"segment {s} has {lengths[s]} tokens in a slot of {slot}")
        if p == -1:
            if prefix_len[s] != 0:
                raise ValueError(f"independent segment {s} has seq_prefix_len {prefix_len[s]}")
        elif not (0 <= p < n and prefix[p] == -1):
            raise ValueError(f"segment {s} must continue an independent segment, not {p}")
        elif not 1 <= prefix_len[s] <= lengths[p] - halo:
            raise ValueError(
                f"segment {s} continues {prefix_len[s]} of segment {p}'s {lengths[p]} tokens: the "
                f"head must be non-empty and leave {halo} halo tokens after it"
            )
        elif lengths[s] < halo:
            raise ValueError(f"segment {s} has {lengths[s]} tokens, fewer than the halo of {halo}")

    q_rows, kv_rows, cu_q, cu_kv = [], [], [0], [0]
    for s in range(n):
        if lengths[s] == 0:
            continue
        p = prefix[s]
        head = range(padded[p], padded[p] + prefix_len[s]) if p >= 0 else ()
        own = range(padded[s], padded[s] + lengths[s])
        q_rows.extend(own)
        kv_rows.extend(head)
        kv_rows.extend(own)
        cu_q.append(cu_q[-1] + len(own))
        cu_kv.append(cu_kv[-1] + len(head) + len(own))
    plan = SharedPrefixPlan(
        q_index=torch.tensor(q_rows, dtype=torch.int64, device=device),
        kv_index=torch.tensor(kv_rows, dtype=torch.int64, device=device),
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_kv=torch.tensor(cu_kv, dtype=torch.int32, device=device),
        max_seqlen_q=max(lengths, default=0),
        max_seqlen_kv=max((prefix_len[s] + lengths[s] for s in range(n)), default=0),
        start_positions=torch.tensor(prefix_len, dtype=torch.int32, device=device),
    )
    packed_seq_params.seq_prefix_cache = plan
    return plan
