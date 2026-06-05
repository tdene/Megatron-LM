# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from .moe_routing_replay import MoERoutingReplayCache
from .prefix_cache_block_state import PrefixCacheBlockState


class KVBlockAllocator:
    """Allocator that manages blocks of memory for the KV cache.

    This allocator owns:

    - The free-pool stack (`block_bag`, `total_avail`).
    - Allocation, release, and reset orchestration.
    - The MoE routing-replay per-block storage.

    Prefix caching is handled by a separate `PrefixCacheBlockState` instance.

    Args:
        context (DynamicInferenceContext): Dynamic inference context.
        total_count (int): Total number of blocks in the buffer.
        paused_count (int): Number of paused blocks in the buffer. Must be less than `total_count`.
        pc_state (Optional[PrefixCacheBlockState]): Per-block prefix-caching state.
            `None` disables prefix caching entirely on this allocator.
    """

    def __init__(
        self,
        context: "DynamicInferenceContext",
        total_count: int,
        paused_count: int,
        pc_state: Optional[PrefixCacheBlockState] = None,
    ):

        self.context = context
        self.pc_state = pc_state

        self.total_count = total_count
        self.total_avail = total_count - 1  # -1 for dummy_block_idx (see below)
        self.paused_count = paused_count
        self.active_count = total_count - paused_count - 1  # -1 for dummy_block_idx
        assert self.active_count >= 1  # ensures paused_count < total_count - 1
        self.dummy_block_idx = self.total_count - 1

        # Initialize block pool as a "stack" data structure (CPU for bookkeeping).
        self.block_bag = torch.arange(self.total_count, dtype=torch.int32, device='cpu')

        # MoE routing-replay storage.
        self.routing_replay = MoERoutingReplayCache(
            block_size_tokens=context.block_size_tokens
        )

    def __str__(self):
        return (
            f"using: total {self.get_total_used()}/{self.total_count - 1}"
            f"; active {self.get_active_used()}/{self.active_count}"
            f"; paused {self.get_paused_used()}/{self.paused_count}"
        )

    def get_total_used(self):
        """Compute number of total blocks used."""
        return self.total_count - self.total_avail - 1

    def get_active_used(self):
        """Compute number of active blocks used."""
        if self.pc_state is None:
            return (
                self.context.request_kv_block_counts[
                    self.context.paused_request_count : self.context.total_request_count
                ]
                .sum()
                .item()
            )

        active_start = self.context.paused_request_count
        active_end = self.context.total_request_count
        if active_end > active_start:
            active_rows = self.context.request_to_kv_block_ids[active_start:active_end]
            valid_ids = active_rows[active_rows >= 0]
            if valid_ids.numel() > 0:
                return int(torch.unique(valid_ids).numel())
        return 0

    def get_paused_used(self):
        """Compute number of paused blocks used."""
        if self.pc_state is None:
            return (
                self.context.request_kv_block_counts[: self.context.paused_request_count]
                .sum()
                .item()
            )

        if self.context.paused_request_count > 0:
            paused_rows = self.context.request_to_kv_block_ids[: self.context.paused_request_count]
            valid_ids = paused_rows[paused_rows >= 0]
            if valid_ids.numel() > 0:
                return int(torch.unique(valid_ids).numel())
        return 0

    def get_active_avail(self):
        """Compute number of active blocks available."""
        return self.active_count - self.get_active_used()

    def get_paused_avail(self):
        """Compute number of paused blocks available."""
        return self.paused_count - self.get_paused_used()

    def is_memory_available(self, num_blocks: int) -> bool:
        """Check if memory blocks are available.

        Includes both free pool blocks and (under LRU prefix caching) evictable cached blocks.

        Args:
            num_blocks (int): Number of blocks to check.

        Return:
            (bool) Is memory available?
        """
        # Fast path: avoid querying pc_state when the free pool already suffices.
        if self.total_avail >= num_blocks:
            return True
        if self.pc_state is None:
            return False
        return (self.total_avail + self.pc_state.extra_blocks_available()) >= num_blocks

    def allocate_memory_blocks(self, num_blocks: int) -> Optional[Tensor]:
        """Allocate memory blocks if available, else return None.

        Under LRU prefix caching, falls back to evicting cached blocks when the free pool is short.
        Returns `None` when even eviction cannot satisfy the request.

        Args:
            num_blocks (int): Number of blocks to allocate.

        Return:
            (Optional[Tensor]) Allocated block IDs.
        """
        # Try to evict cached blocks if free pool is insufficient.
        if self.total_avail < num_blocks:
            if self.pc_state is None:
                return None
            result = self.pc_state.try_lru_evict_for_pool(num_blocks - self.total_avail)
            if result is None:
                return None
            victims, hashes_to_drop = result
            self.context.prefix_cache_registry.evict_kv(hashes_to_drop)
            self._push_to_pool(victims)

        # Now allocate from the free pool
        self.total_avail -= num_blocks
        block_ids = self.block_bag[self.total_avail : (self.total_avail + num_blocks)]
        assert num_blocks == block_ids.numel()

        if self.pc_state is not None:
            self.pc_state.on_allocate(block_ids, self.context.prefix_cache_lru_clock)

        # Clear stale routing data for re-allocated blocks
        self.routing_replay.clear_for_reallocated(block_ids.tolist())

        return block_ids

    def release_memory_blocks(self, blocks: Tensor) -> None:
        """Release memory blocks.

        Without prefix caching: blocks return directly to the free pool.
        With prefix caching: ref counts are decremented and blocks are released via the policy.

        Args:
            blocks (Tensor): Block IDs to release.
        """
        if blocks.numel() == 0:
            return

        if self.pc_state is None:
            self._push_to_pool(blocks)
            return

        pool_returns, hashes_to_drop = self.pc_state.on_release_compute_pool_returns(blocks)
        if hashes_to_drop:
            self.context.prefix_cache_registry.evict_kv(hashes_to_drop)
        if pool_returns.numel() > 0:
            self._push_to_pool(pool_returns)

    def reset(self) -> None:
        """Reset the allocator to initial state.

        This resets the available block count to the entire memory pool
        (except for the dummy block).
        """

        # Reset block bag to so we start consuming from the beginning of the pool
        # for UVM performance.
        # *Note*: Resetting the block bag is essential because if engine has been
        # suspended, then the block bag contains non-unique IDs since the
        # right-most IDs have been 'popped' off and are owned by the context.
        # Without resetting the block bag, context request memory will clash and
        # requests will point to each other's memory blocks, resulting in faulty
        # generations.
        self.block_bag = torch.arange(self.total_count, dtype=torch.int32, device='cpu')

        self.total_avail = self.total_count - 1

        if self.pc_state is not None:
            self.pc_state.reset()
            self.context.prefix_cache_registry.clear_kv()

        # Reset routing-replay storage.
        self.routing_replay.reset()

    def _push_to_pool(self, blocks: Tensor) -> None:
        """Push blocks back onto the free-pool stack."""
        num_blocks = blocks.numel()
        if num_blocks == 0:
            return
        self.block_bag[self.total_avail : self.total_avail + num_blocks] = blocks
        self.total_avail += num_blocks
