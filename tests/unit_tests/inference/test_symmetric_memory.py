# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Registry-level tests for SymmetricMemoryManager's grow-on-demand semantics.

A real symmetric-memory rendezvous needs NVLink-connected GPUs and an
initialized process group, so these tests stub SymmetricMemoryBuffer and pin
down only the bookkeeping the autotune context rebuilds rely on: same-or-
smaller requests reuse the cached buffer, larger requests destroy and
re-create it (reusing the stored process group), and failed buffers are
returned as-is rather than "grown".
"""

from megatron.core.inference import symmetric_memory as sm


class _FakeBuffer:
    """Stands in for SymmetricMemoryBuffer: records ctor args, always healthy."""

    def __init__(self, size_in_mb, process_group):
        self.size_in_mb = size_in_mb
        self.process_group = process_group
        self.symm_buffer = object()
        self.init_failure_reason = None


class _FakeFailedBuffer(_FakeBuffer):
    """A buffer whose symmetric-memory initialization failed."""

    def __init__(self, size_in_mb, process_group):
        super().__init__(size_in_mb, process_group)
        self.symm_buffer = None
        self.init_failure_reason = "no symmetric memory"


def _with_fake_buffer(fake_cls, fn):
    """Run fn() with SymmetricMemoryBuffer stubbed and a clean registry."""
    saved_cls = sm.SymmetricMemoryBuffer
    saved_buffers = sm.SymmetricMemoryManager._buffers
    sm.SymmetricMemoryBuffer = fake_cls
    sm.SymmetricMemoryManager._buffers = {}
    try:
        return fn()
    finally:
        sm.SymmetricMemoryBuffer = saved_cls
        sm.SymmetricMemoryManager._buffers = saved_buffers


class TestSymmetricMemoryManagerGrowth:
    """get_buffer registry semantics (creation, reuse, growth, failure)."""

    def test_same_or_smaller_request_reuses_cached_buffer(self):
        def check():
            pg = object()
            first = sm.SymmetricMemoryManager.get_buffer("k", process_group=pg, size_mb=8)
            assert first.size_in_mb == 8 and first.process_group is pg
            assert sm.SymmetricMemoryManager.get_buffer("k", size_mb=8) is first
            assert sm.SymmetricMemoryManager.get_buffer("k", size_mb=4) is first
            assert sm.SymmetricMemoryManager.get_buffer("k") is first

        _with_fake_buffer(_FakeBuffer, check)

    def test_larger_request_grows_and_reuses_stored_group(self):
        def check():
            pg = object()
            first = sm.SymmetricMemoryManager.get_buffer("k", process_group=pg, size_mb=8)
            grown = sm.SymmetricMemoryManager.get_buffer("k", size_mb=16)
            assert grown is not first
            assert grown.size_in_mb == 16
            assert grown.process_group is pg  # reused from the destroyed buffer
            assert sm.SymmetricMemoryManager.get_buffer("k") is grown

        _with_fake_buffer(_FakeBuffer, check)

    def test_larger_request_honors_explicit_group(self):
        def check():
            pg1, pg2 = object(), object()
            sm.SymmetricMemoryManager.get_buffer("k", process_group=pg1, size_mb=8)
            grown = sm.SymmetricMemoryManager.get_buffer("k", process_group=pg2, size_mb=16)
            assert grown.size_in_mb == 16 and grown.process_group is pg2

        _with_fake_buffer(_FakeBuffer, check)

    def test_failed_buffer_is_not_grown(self):
        def check():
            pg = object()
            failed = sm.SymmetricMemoryManager.get_buffer("k", process_group=pg, size_mb=8)
            assert failed.init_failure_reason is not None
            # Growing cannot fix a missing backend: the failed buffer is
            # returned unchanged so callers see init_failure_reason.
            assert sm.SymmetricMemoryManager.get_buffer("k", size_mb=16) is failed

        _with_fake_buffer(_FakeFailedBuffer, check)

    def test_destroy_forgets_key(self):
        def check():
            pg = object()
            first = sm.SymmetricMemoryManager.get_buffer("k", process_group=pg, size_mb=8)
            sm.SymmetricMemoryManager.destroy("k")
            assert not sm.SymmetricMemoryManager.is_initialized("k")
            recreated = sm.SymmetricMemoryManager.get_buffer("k", process_group=pg, size_mb=4)
            assert recreated is not first and recreated.size_in_mb == 4

        _with_fake_buffer(_FakeBuffer, check)
