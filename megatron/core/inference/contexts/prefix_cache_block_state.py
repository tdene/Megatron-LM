# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional

import torch
from torch import Tensor

from megatron.core.inference.config import PrefixCachingEvictionPolicy

from .prefix_cache_registry import PrefixCacheRegistry


class PrefixCacheBlockState:
    """Per-block CPU shadow state for prefix caching."""

    def __init__(
        self,
        total_count: int,
        eviction_policy: PrefixCachingEvictionPolicy,
        registry: PrefixCacheRegistry,
    ):
        self.total_count = total_count
        self.eviction_policy = eviction_policy
        self.registry = registry

        # `-1` = uncomputed; positive = registered hash.
        self.block_hashes = torch.full((total_count,), -1, dtype=torch.int64, device='cpu')

        # `0` = cached/evictable; `>0` = actively held.
        self.block_ref_counts = torch.zeros((total_count,), dtype=torch.int32, device='cpu')

        # LRU only; REF_ZERO evicts immediately on `ref_count == 0`.
        if eviction_policy == PrefixCachingEvictionPolicy.LRU:
            self.block_timestamps = torch.zeros((total_count,), dtype=torch.int64, device='cpu')
        else:
            self.block_timestamps = None

    @property
    def is_lru(self) -> bool:
        """True when the eviction policy is LRU (`block_timestamps` is allocated)."""
        return self.eviction_policy == PrefixCachingEvictionPolicy.LRU

    def reset(self) -> None:
        """Reset all per-block state and clear the host KV registry."""
        self.block_hashes.fill_(-1)
        self.block_ref_counts.fill_(0)
        if self.block_timestamps is not None:
            self.block_timestamps.zero_()
        self.registry.clear_kv()

    def register_kv_block_hashes(
        self, block_ids: List[int], block_hashes: List[int]
    ) -> None:
        """Register blocks in the hash-to-block mapping for discovery (batch)."""
        if not block_ids:
            return
        id_tensor = torch.tensor(block_ids, dtype=torch.int64, device=self.block_hashes.device)
        hash_tensor = torch.tensor(
            block_hashes, dtype=torch.int64, device=self.block_hashes.device
        )
        self.block_hashes[id_tensor] = hash_tensor
        self.registry.register_kv(block_ids, block_hashes)

    def update_timestamps(self, block_ids: Tensor, lru_clock: int) -> None:
        """Stamp `block_timestamps[block_ids] = lru_clock`. No-op in REF_ZERO mode."""
        if not self.is_lru or block_ids.numel() == 0:
            return
        self.block_timestamps[block_ids] = lru_clock

    def on_allocate(self, block_ids: Tensor, lru_clock: int) -> None:
        """Initialize ref counts (and timestamps under LRU) for newly-allocated blocks.

        Called by the allocator immediately after popping `block_ids` from the free pool.
        """
        self.block_ref_counts[block_ids] = 1
        self.update_timestamps(block_ids, lru_clock)

    def on_prefix_match(self, block_ids: Tensor, lru_clock: int) -> None:
        """Bump ref counts (and timestamps under LRU) for blocks shared via a prefix match.

        Called by `add_request` when an incoming request reuses cached blocks.
        Each matched block gains one more owner; LRU timestamps are refreshed.
        """
        if block_ids.numel() == 0:
            return
        self.block_ref_counts[block_ids] += 1
        self.update_timestamps(block_ids, lru_clock)

    def get_evictable_block_count(self) -> Tensor:
        """Count of blocks that are cached (`ref_count == 0`) and have a registered hash."""
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        return cached_mask.sum()

    def extra_blocks_available(self) -> int:
        """How many cached blocks could be made available via eviction.

        Zero under REF_ZERO: in that policy a block whose ref count hits zero is immediately
        deregistered and pushed to the pool; the cached reservoir is always empty by construction.
        Equals the evictable count under LRU.
        """
        if not self.is_lru:
            return 0
        return int(self.get_evictable_block_count().item())

    def try_lru_evict_for_pool(self, num_blocks_needed: int) -> Optional[Tensor]:
        """Pick + deregister the LRU-oldest victims and return them to the caller.

        Returns:
            - `None` under REF_ZERO (no cached reservoir to evict from).
            - `None` under LRU when fewer than `num_blocks_needed` evictable blocks exist.
            - A tensor of exactly `num_blocks_needed` victim block IDs on success.
              The caller (the allocator) is responsible for pushing them onto the free pool;
              the victims have already been deregistered here.
        """
        if not self.is_lru:
            return None
        victims = self.find_lru_evictable(num_blocks_needed)
        if victims is None:
            return None
        self.deregister_blocks(victims)
        return victims

    def find_lru_evictable(self, num_blocks_needed: int) -> Optional[Tensor]:
        """Return up to `num_blocks_needed` LRU-oldest evictable block IDs.

        `None` if not enough evictable blocks exist.
        The returned tensor is always exactly `num_blocks_needed` entries on success.
        Callers are responsible for pushing the evicted blocks back onto the free pool
        via the allocator after `deregister_blocks` clears their state.
        """
        assert self.is_lru, "find_lru_evictable requires LRU eviction policy"
        cached_mask = (self.block_ref_counts == 0) & (self.block_hashes != -1)
        cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]
        if cached_block_ids.numel() < num_blocks_needed:
            return None
        cached_timestamps = self.block_timestamps[cached_block_ids]
        sorted_indices = torch.argsort(cached_timestamps)
        return cached_block_ids[sorted_indices[:num_blocks_needed]]

    def deregister_blocks(self, block_ids: Tensor) -> None:
        """Drop hashes from the registry and reset per-block state.

        Does NOT push the blocks back to the free pool: that is the
        allocator's responsibility, called immediately after this returns.

        Used by both the LRU eviction path and the REF_ZERO release path.
        """
        if block_ids.numel() == 0:
            return
        block_ids_i64 = block_ids.to(torch.int64)
        hashes = self.block_hashes[block_ids_i64].tolist()

        # Drop hashes from the registry. The cascade fires the Mamba
        # evict callback, which clears the corresponding Mamba GPU slots.
        self.registry.evict_kv(hashes)

        # Reset per-block state (batched tensor ops).
        self.block_hashes[block_ids] = -1
        self.block_ref_counts[block_ids] = 0
        if self.block_timestamps is not None:
            self.block_timestamps[block_ids] = 0

    def on_release_compute_pool_returns(self, blocks: Tensor) -> Tensor:
        """Decrement ref counts and return the subset that should go back to the pool.

        Policy-specific:

        - REF_ZERO: every block with zero ref count is deregistered and returned to the pool.
        - LRU: only unregistered (`hash == -1`) zero-ref blocks are returned.
          Blocks with a registered hash stay cached for later reuse and are evicted on demand.

        The allocator pushes the returned tensor onto the block bag.
        """
        if blocks.numel() == 0:
            return blocks

        self.block_ref_counts[blocks] -= 1

        if self.eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO:
            zero_mask = self.block_ref_counts[blocks] == 0
            if not zero_mask.any():
                return blocks[:0]  # empty slice, preserves dtype/device
            zero_blocks = blocks[zero_mask]
            self.deregister_blocks(zero_blocks)
            return zero_blocks

        # LRU: return only unregistered (no-hash) zero-ref blocks; cached blocks remain
        # in the pool's eviction-eligible reservoir until `find_lru_evictable` claims them.
        unreg_mask = (self.block_ref_counts[blocks] == 0) & (self.block_hashes[blocks] == -1)
        if not unreg_mask.any():
            return blocks[:0]
        return blocks[unreg_mask]
