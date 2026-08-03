# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit + pipeline tests for the durable rollout bank (queued-group path).

Covers the append -> attach-records -> restore round trip, checksum/torn-write
handling for both the JSONL index and the binary sidecars, the consumption-marker
filter, manifest + compaction, the restored-records rejoin into the runtime
request ledger (through the real ``compute_group_stats``), and an end-to-end
write-through/restore through the real ``RolloutPipeline`` (reusing the mocks
from ``test_rollout_generation``).
"""

import asyncio
import json
import os
from collections import deque
from contextlib import aclosing

import numpy as np
import pytest

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.inference.inference_request import FinishedRequestRecord
from megatron.rl import rl_utils, rollout_bank
from megatron.rl.agent.api import GroupedRolloutRequest, Rollout, RolloutGroup, TokenRollout
from megatron.rl.agent.rollout_pipeline import RolloutPipeline
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.rollout_bank import (
    _CONSUMED,
    _FORMAT_VERSION,
    _LEDGER,
    _MANIFEST,
    _TOKENS_BIN,
    RolloutBank,
    _segment_name,
    rollout_request_keys,
)
from megatron.rl.types import Rollout as SharedRollout
from megatron.rl.types import RolloutGroup as SharedRolloutGroup
from megatron.rl.types import TokenRollout as SharedTokenRollout
from megatron.training.checkpointing import _register_rollout_bank_compaction

# Reuse the tokenizer mock so the rejoin test drives the real compute_group_stats,
# and the pipeline mocks so the integration tests drive the real pipeline.
from tests.unit_tests.rl.test_rl_utils import MockTokenizer
from tests.unit_tests.rl.test_rollout_generation import (
    FilteringMockGenerator,
    MockGenerator,
    MockInferenceInterface,
)


def test_agent_api_reexports_shared_rollout_types():
    assert SharedRollout is Rollout
    assert SharedRolloutGroup is RolloutGroup
    assert SharedTokenRollout is TokenRollout


def test_rollout_reward_accepts_none():
    rollout = Rollout(trajectory=["prompt"], reward=None)

    assert rollout.reward is None


def make_token_group(members, *, batch_id=0, index_in_batch=0):
    """Build a RolloutGroup of TokenRollout members.

    ``members`` is a list of (tokens, logprobs, mask) triples, each a per-turn
    jagged list, so the sidecar packing is exercised with multi-turn, ragged data.
    """
    rollouts = []
    for tokens, logprobs, mask in members:
        rollouts.append(
            TokenRollout(
                trajectory=tokens,
                reward=1.0,
                logprobs=logprobs,
                generation_mask=mask,
                env_id="test",
                problem_id="p",
            )
        )
    return RolloutGroup(rollouts=rollouts, batch_id=batch_id, index_in_batch=index_in_batch)


def sample_group(batch_id=0):
    return make_token_group(
        [
            ([[1, 2, 3], [4, 5]], [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
             [[False, True, True], [True, True]]),
            ([[7, 8]], [[-1.5, -2.5]], [[True, True]]),
        ],
        batch_id=batch_id,
    )


def text_group():
    return RolloutGroup(
        rollouts=[Rollout(trajectory=["hello world"], reward=0.5, env_id="t")],
    )


def build_ledger(groups, epoch=0):
    """A runtime request ledger holding one record per turn of every token rollout.

    Keys come from the shared ``rollout_request_keys`` helper — exactly the keys
    the engine drain and the consumption-time pop use — with one record appended
    per occurrence so same-content turns get their full multiplicity.
    """
    ledger = {}
    for group in groups:
        for rollout in group.rollouts:
            if not isinstance(rollout, TokenRollout):
                continue
            for key in rollout_request_keys(rollout):
                ledger.setdefault(key, []).append(
                    FinishedRequestRecord(
                        policy_epoch=[(0, epoch)],
                        kv_cache_epoch=[(0, epoch)],
                        num_evictions=0,
                    )
                )
    return ledger


def attach_records(bank, groups, epoch=0):
    """Persist the pending groups' records, as the post-drain merge point would."""
    return bank.attach_pending_request_records(build_ledger(groups, epoch))


class TestRoundTrip:
    def test_manifest_and_ledger_record_current_format_version(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        group = sample_group()
        bank.append(group)
        attach_records(bank, [group])
        bank.close()

        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        ledger_path = tmp_path / _segment_name(3) / _LEDGER
        group_record, records_amendment = map(json.loads, ledger_path.read_text().splitlines())

        assert manifest["format_version"] == _FORMAT_VERSION
        assert group_record["format_version"] == _FORMAT_VERSION
        assert records_amendment["format_version"] == _FORMAT_VERSION
        assert records_amendment["kind"] == "records"
        assert records_amendment["uid"] == group_record["uid"]

    @pytest.mark.parametrize(
        "invalid_version",
        [
            pytest.param(None, id="missing"),
            pytest.param(_FORMAT_VERSION + 1, id="unsupported"),
        ],
    )
    def test_restore_rejects_incompatible_manifest_version(self, tmp_path, invalid_version):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        manifest_path = tmp_path / _MANIFEST
        manifest = json.loads(manifest_path.read_text())
        if invalid_version is None:
            manifest.pop("format_version")
        else:
            manifest["format_version"] = invalid_version
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Unsupported RolloutBank format_version"):
            RolloutBank(str(tmp_path)).restore(0)

    @pytest.mark.parametrize(
        "invalid_version",
        [
            pytest.param(None, id="missing"),
            pytest.param(_FORMAT_VERSION + 1, id="unsupported"),
        ],
    )
    def test_restore_rejects_incompatible_ledger_version(self, tmp_path, invalid_version):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        ledger_path = tmp_path / _segment_name(3) / _LEDGER
        record = json.loads(ledger_path.read_text())
        if invalid_version is None:
            record.pop("format_version")
        else:
            record["format_version"] = invalid_version
        ledger_path.write_text(json.dumps(record) + "\n")

        with pytest.raises(ValueError, match="Unsupported RolloutBank format_version"):
            RolloutBank(str(tmp_path)).restore(0)

    def test_encode_returns_named_payload(self, tmp_path):
        assert hasattr(rollout_bank, "EncodedGroup")

        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        encoded = bank._encode(sample_group(), "gen-000003/0")

        assert isinstance(encoded, rollout_bank.EncodedGroup)
        assert encoded.record["uid"] == "gen-000003/0"
        assert encoded.tok_bytes
        assert encoded.lp_bytes
        assert encoded.mask_bytes

    def test_token_group_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        original = sample_group(batch_id=2)
        uid = bank.append(original)
        attach_records(bank, [original], epoch=5)
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 1
        g = restored[0]
        assert g.uid == uid
        assert g.batch_id == 2
        # token ids are exact (int32)
        assert g.rollouts[0].trajectory == [[1, 2, 3], [4, 5]]
        assert g.rollouts[1].trajectory == [[7, 8]]
        # generation_mask preserved exactly
        assert g.rollouts[0].generation_mask == [[False, True, True], [True, True]]
        # logprobs recovered within fp16 tolerance
        assert np.allclose(g.rollouts[0].logprobs[0], [-0.1, -0.2, -0.3], atol=1e-3)
        assert np.allclose(g.rollouts[1].logprobs[0], [-1.5, -2.5], atol=1e-3)
        # the persisted finished-request records ride along: one per turn per member
        assert [len(member) for member in g.request_records] == [2, 1]
        assert g.request_records[0][0]["policy_epoch"] == [[0, 5]]
        assert g.request_records[1][0]["kv_cache_epoch"] == [[0, 5]]

    def test_text_group_round_trip(self, tmp_path):
        # Inline (text) groups never join the request ledger, so they are fully
        # restorable from the write-through append alone — no records amendment.
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        bank.append(text_group())
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 1
        assert isinstance(restored[0].rollouts[0], Rollout)
        assert restored[0].rollouts[0].trajectory == ["hello world"]
        assert getattr(restored[0], "request_records", None) is None

    def test_token_group_without_records_amendment_is_dropped_on_restore(self, tmp_path):
        # A kill before the engine drain leaves a banked token group without its
        # finished-request records; the consumption-time ledger join would
        # hard-assert on it, so restore drops it (it is regenerated instead).
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.close()

        assert RolloutBank(str(tmp_path)).restore(trained_through=0) == []

    def test_fp16_logprobs_lossy_but_close_tokens_exact(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        toks = list(range(50))
        lps = [round(-0.01 * i, 4) for i in range(50)]
        group = make_token_group([([toks], [lps], [[True] * 50])])
        bank.append(group)
        attach_records(bank, [group])
        bank.close()

        g = RolloutBank(str(tmp_path)).restore(0)[0]
        assert g.rollouts[0].trajectory[0] == toks  # int32 exact
        assert np.allclose(g.rollouts[0].logprobs[0], lps, atol=1e-3)

    @pytest.mark.parametrize("field", ["logprobs", "generation_mask"])
    def test_mixed_optional_field_presence_is_rejected(self, tmp_path, field):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        group = sample_group()
        setattr(group.rollouts[1], field, None)

        with pytest.raises(ValueError, match=f"{field} must be present for all or no rollouts"):
            bank.append(group)


class TestDurability:
    def test_manifest_replace_is_followed_by_bank_directory_fsync(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        events = []
        real_replace = os.replace

        def replace(src, dst):
            real_replace(src, dst)
            events.append("replace")

        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )

        bank._write_manifest_atomic(
            {"trained_through": 1, "segments": [], "compacted_at": 0}
        )

        assert events == ["replace", f"dir:{tmp_path}"]

    def test_first_append_fsyncs_new_entries_after_file_contents(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        segment = tmp_path / _segment_name(0)
        events = []
        monkeypatch.setattr(os, "fsync", lambda fd: events.append("file"))
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )

        bank.append(sample_group())

        assert events == ["file", f"dir:{segment}"] * 4

        events.clear()
        bank.append(sample_group())
        assert events == ["file"] * 4

    def test_new_segment_is_durable_before_manifest_publication(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        events = []
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )
        monkeypatch.setattr(
            bank,
            "_write_manifest_atomic",
            lambda manifest: events.append(f"manifest:{manifest['segments'][-1]}"),
        )

        bank.set_collection(7)

        assert events == [f"dir:{tmp_path}", f"manifest:{_segment_name(7)}"]

    def test_compacted_segment_is_durable_before_manifest_publication(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        events = []
        real_replace = os.replace

        def replace(src, dst):
            real_replace(src, dst)
            if str(src).endswith(".compact"):
                events.append("segment_replace")

        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )
        monkeypatch.setattr(bank, "restore", lambda iteration: [])
        monkeypatch.setattr(bank, "_rewrite_segment", lambda *args: None)
        monkeypatch.setattr(
            bank,
            "_write_manifest_atomic",
            lambda manifest: events.append(f"manifest:{manifest['trained_through']}"),
        )

        bank.checkpoint(2)

        replace_index = events.index("segment_replace")
        assert events[replace_index : replace_index + 3] == [
            "segment_replace",
            f"dir:{tmp_path}",
            "manifest:2",
        ]

    def test_first_consumed_marker_fsyncs_bank_directory(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        events = []
        monkeypatch.setattr(os, "fsync", lambda fd: events.append("file"))
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )

        bank.mark_consumed("gen-000000/0", 1)
        bank.mark_consumed("gen-000000/1", 1)

        assert events == ["file", f"dir:{tmp_path}", "file"]

    def test_torn_final_ledger_line_dropped_and_append_recovers_after_restart(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        first, second = sample_group(), sample_group()
        bank.append(first)
        bank.append(second)
        attach_records(bank, [first, second])
        bank.close()

        # Simulate a kill mid-append: a truncated JSON line at the end of the ledger.
        ledger = os.path.join(str(tmp_path), _segment_name(0), _LEDGER)
        with open(ledger, "a") as f:
            f.write('{"uid": "gen-000000/2", "kind": "toke')  # torn, no newline

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 2  # the two intact records survive

        restarted = RolloutBank(str(tmp_path))
        restarted.set_collection(0)
        third = sample_group()
        new_uid = restarted.append(third)
        attach_records(restarted, [third])
        restarted.close()

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 3
        assert new_uid == f"{_segment_name(0)}/2"
        assert new_uid in {group.uid for group in restored}

    def test_truncated_sidecar_slice_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        first, second = sample_group(), sample_group()
        bank.append(first)
        bank.append(second)
        attach_records(bank, [first, second])
        bank.close()

        # Chop the tail of tokens.bin so the second record's slice is short.
        tokens_bin = os.path.join(str(tmp_path), _segment_name(0), _TOKENS_BIN)
        size = os.path.getsize(tokens_bin)
        with open(tokens_bin, "r+b") as f:
            f.truncate(size - 4)

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 1  # only the first record's slice is intact

    def test_checksum_mismatch_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        group = sample_group()
        bank.append(group)
        attach_records(bank, [group])
        bank.close()

        ledger = os.path.join(str(tmp_path), _segment_name(0), _LEDGER)
        with open(ledger) as f:
            lines = [json.loads(line) for line in f.read().splitlines()]
        lines[0]["checksum"] = "0" * 32  # tamper the group record
        with open(ledger, "w") as f:
            f.write("\n".join(json.dumps(line) for line in lines) + "\n")

        assert RolloutBank(str(tmp_path)).restore(0) == []

    def test_tampered_records_amendment_drops_group(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        group = sample_group()
        bank.append(group)
        attach_records(bank, [group], epoch=3)
        bank.close()

        ledger = os.path.join(str(tmp_path), _segment_name(0), _LEDGER)
        with open(ledger) as f:
            lines = [json.loads(line) for line in f.read().splitlines()]
        assert lines[1]["kind"] == "records"
        lines[1]["request_records"][0][0]["policy_epoch"] = [[0, 999]]  # tamper
        with open(ledger, "w") as f:
            f.write("\n".join(json.dumps(line) for line in lines) + "\n")

        # The amendment fails its checksum and is discarded; without records the
        # token group cannot rejoin the ledger, so it is dropped too.
        assert RolloutBank(str(tmp_path)).restore(0) == []


class TestMarkerFilter:
    def test_marker_filter_rules(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(5)
        groups = [sample_group() for _ in range(3)]
        trained = bank.append(groups[0])   # consumed at 5 <= T=10 -> discard
        rolled_back = bank.append(groups[1])  # consumed at 12 > T=10 -> restore
        _never = bank.append(groups[2])     # no marker -> restore
        attach_records(bank, groups)
        bank.mark_consumed(trained, 5)
        bank.mark_consumed(rolled_back, 12)
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=10)
        uids = {g.uid for g in restored}
        assert trained not in uids
        assert rolled_back in uids
        assert _never in uids


class TestCompaction:
    def test_async_compaction_finalize_runs_with_captured_iteration(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        monkeypatch.setattr(rl_utils, "_ROLLOUT_BANK", bank)
        bank.set_collection(1)
        first_group = sample_group()
        first = bank.append(first_group)
        attach_records(bank, [first_group])
        bank.mark_consumed(first, 1)

        first_save = AsyncRequest(None, (), [])
        second_save = AsyncRequest(None, (), [])
        _register_rollout_bank_compaction(first_save, 1)
        _register_rollout_bank_compaction(second_save, 2)

        bank.set_collection(2)
        second_group = sample_group()
        second = bank.append(second_group)
        attach_records(bank, [second_group])
        bank.mark_consumed(second, 2)

        first_save.finalize_fns[0]()
        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 1
        assert {group.uid for group in bank.restore(1)} == {second}

        second_save.finalize_fns[0]()
        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 2
        assert manifest["segments"] == [_segment_name(2)]
        assert bank.restore(2) == []

    def test_marker_after_compaction_is_not_orphaned(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        group = sample_group()
        old_uid = bank.append(group)
        attach_records(bank, [group])

        bank.checkpoint(2)
        bank.mark_consumed(old_uid, 4)

        assert bank.restore(2)[0].uid == old_uid
        assert bank.restore(4) == []

    def test_fresh_append_after_compaction_has_unique_uid(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(2)
        survivor_group = sample_group()
        survivor_uid = bank.append(survivor_group)
        attach_records(bank, [survivor_group])

        bank.checkpoint(2)
        fresh_group = sample_group()
        fresh_uid = bank.append(fresh_group)
        attach_records(bank, [fresh_group])

        assert fresh_uid != survivor_uid
        assert {group.uid for group in bank.restore(2)} == {survivor_uid, fresh_uid}

    def test_restore_reads_legacy_segment_marker(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        group = sample_group()
        uid = bank.append(group)
        attach_records(bank, [group])
        bank.mark_consumed(uid, 1)
        os.replace(tmp_path / _CONSUMED, tmp_path / _segment_name(1) / _CONSUMED)

        assert RolloutBank(str(tmp_path)).restore(1) == []

    def test_compaction_prunes_and_flips_manifest(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        consumed_group = sample_group()
        consumed = bank.append(consumed_group)
        attach_records(bank, [consumed_group])
        bank.mark_consumed(consumed, 1)
        bank.set_collection(2)
        survivor_group = sample_group()
        survivor = bank.append(survivor_group)
        attach_records(bank, [survivor_group], epoch=7)

        bank.checkpoint(2)  # trained_through=2: prune consumed(<=2), keep survivor

        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 2
        assert manifest["segments"] == [_segment_name(2)]
        assert manifest["compacted_at"] == 2
        # stale segment dir removed
        assert not (tmp_path / _segment_name(1)).exists()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=2)
        assert len(restored) == 1
        # the survivor's payload is intact after being rewritten by compaction
        assert restored[0].rollouts[0].trajectory == [[1, 2, 3], [4, 5]]
        assert restored[0].rollouts[1].trajectory == [[7, 8]]
        # ... and its finished-request records carried forward with it
        assert restored[0].request_records[0][0]["policy_epoch"] == [[0, 7]]
        assert survivor  # uid was assigned at append time

    def test_compaction_survivor_survives_next_kill(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(2)
        group = sample_group()
        bank.append(group)
        attach_records(bank, [group])
        bank.checkpoint(2)
        # A fresh process restores the compacted survivor.
        assert len(RolloutBank(str(tmp_path)).restore(2)) == 1


class TestRestoreRejoinsLedger:
    """A restored group's persisted records must rejoin the runtime request ledger.

    On this branch per-rollout staleness metadata lives in the engine-drained
    request ledger and compute_group_stats hard-asserts when a TokenRollout turn
    finds no record. These tests pin the full restart path: bank -> disk ->
    restore -> _merge_restored_request_records -> compute_group_stats.
    """

    def _group(self, eod):
        # Two members with identical content: their turns share one ledger key,
        # so record multiplicity (not just presence) must survive the round trip.
        # Plus a multi-turn member with a prompt prefix.
        single = lambda: TokenRollout(
            trajectory=[[1, 2, eod]],
            generation_mask=[[False, True, True]],
            logprobs=[[0.0, -0.1, -0.2]],
            reward=1.0,
            env_id="test",
            problem_id="dup",
        )
        multi = TokenRollout(
            trajectory=[[1, 2, 3, eod], [1, 2, 3, eod, 9, 8, eod]],
            generation_mask=[
                [False, False, True, True],
                [False, False, False, False, False, True, True],
            ],
            logprobs=[[0.0] * 4, [0.0] * 7],
            reward=0.0,
            env_id="test",
            problem_id="m",
        )
        return RolloutGroup(rollouts=[single(), single(), multi], batch_id=0, index_in_batch=0)

    def test_restored_group_and_records_rejoin_compute_group_stats(self, tmp_path):
        tokenizer = MockTokenizer()
        group = self._group(tokenizer.eod)
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(7)
        uid = bank.append(group)

        # The post-drain merge point: records are COPIED into the bank...
        live_ledger = build_ledger([group], epoch=5)
        assert bank.attach_pending_request_records(live_ledger) == 1
        # ... never popped: the live consumption pops every record exactly once.
        live_stats = rl_utils.compute_group_stats(
            [group], tokenizer, seq_len=16, request_ledger=live_ledger
        )
        assert live_stats.policy_first_epoch == [[5, 5, 5]]
        assert all(not bucket for bucket in live_ledger.values())
        bank.close()

        # Process restart: fresh bank, fresh (empty) runtime ledger.
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert [g.uid for g in restored] == [uid]
        fresh_ledger = {}
        rl_utils._merge_restored_request_records(restored, fresh_ledger)
        # The carrier attribute is stripped once merged.
        assert getattr(restored[0], "request_records", None) is None
        # Duplicate-content turns kept their multiplicity.
        assert sorted(len(bucket) for bucket in fresh_ledger.values()) == [1, 1, 2]

        stats = rl_utils.compute_group_stats(
            restored, tokenizer, seq_len=16, request_ledger=fresh_ledger
        )
        assert stats.policy_first_epoch == [[5, 5, 5]]
        assert stats.kv_first_epoch == [[5, 5, 5]]
        # Every restored record was popped exactly once — the ledger is drained.
        assert all(not bucket for bucket in fresh_ledger.values())

    def test_merge_skips_groups_without_restored_records(self):
        # Live groups (no request_records attribute) and text groups pass through
        # the merge untouched.
        ledger = {}
        rl_utils._merge_restored_request_records([text_group(), sample_group()], ledger)
        assert ledger == {}


class TestPipelineIntegration:
    """Write-through + restore through the real RolloutPipeline."""

    def _collect(self, tmp_path, num_groups=4, stop_after=None, generator=None,
                 filter_groups_with_same_reward=False):
        async def run():
            gen = generator or MockGenerator()
            bank = RolloutBank(str(tmp_path))
            bank.set_collection(0)
            request = GroupedRolloutRequest(
                num_groups=num_groups,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
                filter_groups_with_same_reward=filter_groups_with_same_reward,
                submission_granularity="B",
                consumption_granularity="B",
            )
            pipeline = RolloutPipeline(
                gen, request, parallel_generation_tasks=1, durable_bank=bank
            )
            request_groups = []
            async with aclosing(pipeline.run()) as iterator:
                async for group in iterator:
                    request_groups.append(group)
                    if stop_after is not None and len(request_groups) >= stop_after:
                        break
                    if stop_after is None and len(request_groups) >= num_groups:
                        break
            bank.close()
            return pipeline, request_groups

        return asyncio.run(run())

    def test_write_through_then_restore(self, tmp_path):
        _pipeline, groups = self._collect(tmp_path, num_groups=4)
        assert len(groups) == 4
        assert all(getattr(g, "uid", None) for g in groups)

        # Fresh process (restart): no markers, T=0 -> all completed groups restored,
        # never regenerated. (Text groups need no records amendment.)
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 4
        assert {g.uid for g in restored} == {g.uid for g in groups}

    def test_early_exit_keeps_assembled_groups(self, tmp_path):
        # Break after the first group; write-through means at least that group is
        # already durable (assembly precedes consumption).
        _pipeline, groups = self._collect(tmp_path, num_groups=4, stop_after=1)
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) >= len(groups) >= 1

    def test_filtered_groups_are_never_banked(self, tmp_path):
        # Zero-variance groups are dropped by stage_filter before the bank hook:
        # only delivered groups (including the regenerated replacements) persist.
        pipeline, groups = self._collect(
            tmp_path,
            num_groups=4,
            stop_after=8,
            generator=FilteringMockGenerator(num_degenerate=3),
            filter_groups_with_same_reward=True,
        )
        assert pipeline.filtered_count == 3
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert {g.uid for g in restored} == {g.uid for g in groups}
        assert len(restored) == len(groups) == 8

    @pytest.mark.asyncio
    async def test_pipeline_without_bank_assigns_no_uids(self):
        gen = MockGenerator()
        request = GroupedRolloutRequest(
            num_groups=2,
            rollouts_per_group=1,
            inference_interface=MockInferenceInterface(),
        )
        pipeline = RolloutPipeline(gen, request, parallel_generation_tasks=1)
        assert pipeline.durable_bank is None

        groups = []
        async with aclosing(pipeline.run()) as iterator:
            async for group in iterator:
                groups.append(group)
                if len(groups) >= 2:
                    break
        assert all(group.uid is None for group in groups)


def _env_group(env_id, problem_id="p"):
    """A minimal inline (text) RolloutGroup tagged with ``env_id``."""
    return RolloutGroup(
        rollouts=[Rollout(trajectory=["x"], reward=1.0, env_id=env_id, problem_id=problem_id)]
    )


def _weighted_agent(env_weights):
    return WeightedMultiTask(
        [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": e}, weight=w)
            for e, w in env_weights
        ]
    )


class TestRestoreBalancing:
    """Cap-and-defer injection + per-env residual balancing for restored groups."""

    def test_env_group_targets_matches_layout(self):
        agent = _weighted_agent([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        for n in (3, 6, 7, 10):
            expected: dict = {}
            for eid, c in zip(agent._rollout_env_ids, agent.rollout_group_layout(n)):
                expected[eid] = expected.get(eid, 0) + c
            assert agent.env_group_targets(n) == expected
            assert sum(agent.env_group_targets(n).values()) == n

    def test_env_group_targets_requires_env_id(self):
        agent = _weighted_agent([("a", 1.0)])
        agent._rollout_agents[0].env_id = None
        with pytest.raises(ValueError, match="has no env_id"):
            agent.env_group_targets(4)

    def test_env_group_targets_excludes_zero_weight_envs(self):
        agent = _weighted_agent([("a", 1.0), ("boot_only", 0.0)])
        assert agent.env_group_targets(8) == {"a": 8}

    def test_plan_restore_injection_caps_and_defers(self):
        target = {"a": 2, "b": 2, "c": 2}
        restored = {"a": deque(range(6))}  # 6 restored groups, all env "a"
        inject, residual = rl_utils._plan_restore_injection(target, restored)
        assert inject == {"a": 2, "b": 0, "c": 0}  # capped at target["a"], 4 deferred
        assert residual == {"a": 0, "b": 2, "c": 2}
        for env in target:
            assert inject[env] + residual[env] == target[env]

    def test_restore_injection_drain_window_stays_balanced(self):
        target = {"a": 2, "b": 2, "c": 2}
        restored = {"a": deque(f"a{i}" for i in range(6))}
        injected_total = []
        steps = 0
        while any(restored.values()):
            inject, residual = rl_utils._plan_restore_injection(target, restored)
            for env in target:
                # No env ever injects more than its weighted target for the batch.
                assert inject[env] <= target[env]
                assert inject[env] + residual[env] == target[env]
            for env, count in inject.items():
                for _ in range(count):
                    injected_total.append(restored[env].popleft())
            steps += 1
            assert steps < 100, "drain did not terminate"
        # Every restored group is eventually injected; none dropped.
        assert len(injected_total) == 6

    def test_bucket_restored_groups_buckets_by_env(self):
        groups = [_env_group("a"), _env_group("b"), _env_group("a")]
        buckets = rl_utils._bucket_restored_groups(groups, {"a", "b", "c"})
        assert set(buckets) == {"a", "b"}
        assert len(buckets["a"]) == 2 and len(buckets["b"]) == 1

    def test_bucket_restored_groups_asserts_env_config_drift(self):
        with pytest.raises(AssertionError, match="not in the current"):
            rl_utils._bucket_restored_groups([_env_group("z")], {"a", "b"})

    def test_pull_fresh_balanced_routes_quotas_and_buffers_overflow(self):
        loop = asyncio.new_event_loop()
        try:

            async def gen():
                for env in ["b", "a", "b", "a"]:
                    yield _env_group(env)

            g = gen()
            overflow = {}
            fresh = rl_utils._pull_fresh_balanced(loop, g, {"a": 2, "b": 0}, overflow)
            assert [x[0].env_id for x in fresh] == ["a", "a"]
            assert [x[0].env_id for x in overflow["b"]] == ["b", "b"]
            # A later step's quota is satisfied from the buffered overflow first.
            fresh2 = rl_utils._pull_fresh_balanced(loop, g, {"b": 2}, overflow)
            assert [x[0].env_id for x in fresh2] == ["b", "b"]
            assert not overflow["b"]
        finally:
            loop.close()

    def test_bucket_and_plan_from_real_bank(self, tmp_path):
        # End-to-end: bank 6 groups all env "a", restore, bucket, and plan step 1.
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        for i in range(6):
            bank.append(_env_group("a", problem_id=f"p{i}"))
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 6
        buckets = rl_utils._bucket_restored_groups(restored, {"a", "b"})
        assert len(buckets["a"]) == 6
        inject, residual = rl_utils._plan_restore_injection({"a": 2, "b": 2}, buckets)
        assert inject == {"a": 2, "b": 0}
        assert residual == {"a": 0, "b": 2}
