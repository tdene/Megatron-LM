# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import importlib.util
import logging
import os
import signal
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPayloadStageResult,
)
from megatron.rl.gym_capture import (
    NG_COMMIT_COORDS_FIELD,
    GymPromptPreparer,
    PrefixStoreClient,
    PrefixStoreError,
    PrefixStoreServer,
    replace_prefix_tokens,
)
from megatron.rl.rollout_bank import LedgerPayloadStager

CHAIN_HASH = "1" * 64
# Nothing listens on port 1: a store that never answers (the connection is refused, ZMQ keeps
# reconnecting in the background and every attempt times out).
DEAD_STORE = "tcp://127.0.0.1:1"


def _admission(**fields):
    admission = {
        "capture_key": "r",
        "model_call_id": "c2",
        "mode": "token_in",
        "parent_call_id": "c1",
        "prev_len": 5,
        "staging_chain": ["u1"],
        "parent_chain_hash": CHAIN_HASH,
    }
    admission.update(fields)
    return admission


def _record():
    # Spans are (token position, epoch): admitted under epoch 3, continued under epoch 4.
    return FinishedRequestRecord(
        policy_epoch=[(0, 3), (2, 4)], kv_cache_epoch=None, num_evictions=0
    )


def _payload():
    return OffloadedRequestPayload(
        prompt_token_ids=[1, 2, 3, 6],
        generated_token_ids=[7],
        generated_log_probs=[-0.5],
        prompt_log_probs=None,
        routing_indices=None,
    )


@pytest.fixture
def store():
    server = PrefixStoreServer(host="127.0.0.1")
    yield server, PrefixStoreClient(server.address, timeout_ms=2000)
    server.close()


@pytest.fixture
def fake_nemo_gym(monkeypatch):
    """A stand-in for Gym's capture helper, so the stager's control flow runs without nemo_gym.

    The fake mirrors what stage_admitted_call uses: CaptureAdmission.model_validate, StageResult,
    RolloutTokenCapture.begin_call / complete_call_from_response with model_dump(mode="json"). It
    fails closed on a token_in admission without a resolved prefix, as Gym does.
    """

    class StageResult:
        def __init__(self, ok, staging_key):
            self.ok, self.staging_key = ok, staging_key

    class CaptureAdmission:
        def __init__(self, fields):
            self.fields = fields
            self.required_prefix_token_ids = list(fields.get("required_prefix_token_ids") or [])

        @classmethod
        def model_validate(cls, fields):
            if fields.get("mode") not in ("text", "token_in"):
                raise ValueError(f"unknown admission mode {fields.get('mode')!r}")
            return cls(fields)

    class Coordinates:
        def __init__(self, data):
            self.data = data

        def model_dump(self, mode="json"):
            return dict(self.data)

    class RolloutTokenCapture:
        def __init__(self, sink, weight_version_fn, adapter):
            self.sink = sink

        def begin_call(self, parsed, prefix_token_ids, weight_version):
            if parsed.fields.get("mode") == "token_in" and not prefix_token_ids:
                raise ValueError("token_in admission without a resolved prefix")
            return (parsed, list(prefix_token_ids or []), weight_version)

        def complete_call_from_response(self, call, payload):
            parsed, prefix, weight_version = call
            cum_len = len(payload.prompt_token_ids) + len(payload.generated_token_ids)
            return Coordinates(
                {
                    "capture_key": parsed.fields.get("capture_key"),
                    "model_call_id": parsed.fields.get("model_call_id"),
                    "parent_call_id": parsed.fields.get("parent_call_id"),
                    "prev_len": len(prefix),
                    "delta_len": cum_len - len(prefix),
                    "cum_len": cum_len,
                    "weight_version": weight_version,
                    "disposition": "staged",
                    "staging_key": self.sink.stage(None).staging_key,
                }
            )

    class MegatronCaptureAdapter:
        pass

    modules = {
        "nemo_gym": {},
        "nemo_gym.token_id_capture": {},
        "nemo_gym.token_id_capture.adapters": {},
        "nemo_gym.token_id_capture.adapters.megatron": {
            "MegatronCaptureAdapter": MegatronCaptureAdapter
        },
        "nemo_gym.token_id_capture.staging": {},
        "nemo_gym.token_id_capture.staging.capture": {"RolloutTokenCapture": RolloutTokenCapture},
        "nemo_gym.token_id_capture.staging.records": {
            "CaptureAdmission": CaptureAdmission,
            "StageResult": StageResult,
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "prefix, expected",
    [
        pytest.param([10, 11, 7, 2], [10, 11, 7, 2, 12], id="exact-turn-ends-with-eos"),
        pytest.param([10, 11, 7], [10, 11, 7, 2, 12], id="cut-turn-takes-the-rendered-eos"),
    ],
)
def test_replace_prefix_tokens_keeps_the_exact_turn_and_the_new_suffix(prefix, expected):
    assert replace_prefix_tokens(prefix, [10, 11, 2], [10, 11, 2, 12], [2]) == expected
    with pytest.raises(ValueError, match="EOS"):
        replace_prefix_tokens(prefix, [10, 11], [10, 11, 12], [2])


def test_store_survives_bad_frames_and_clients_time_out(store):
    server, client = store
    # The launch wiring tests the server for None, never for truthiness, and pings it once.
    assert server is not None and bool(server) and server.size() == 0
    client.ping()
    with pytest.raises(RuntimeError, match="failed"):
        PrefixStoreClient(DEAD_STORE, timeout_ms=200).ping()
    client.put("u1", [1, 2], [3])
    assert client.get("u1") == ([1, 2], [3]) and client.get("absent") is None
    with pytest.raises(RuntimeError, match="malformed"):
        client._request("bogus-op")
    assert client.get("u1") == ([1, 2], [3])  # the server kept serving after the bad frame
    client.forget(["u1", "never"])
    assert server.size() == 0
    with pytest.raises(RuntimeError, match="failed"):
        PrefixStoreClient(DEAD_STORE, timeout_ms=200).get("u1")


def test_preparer_resolves_the_parent_from_the_store_and_inlines_the_prefix(store):
    server, client = store
    preparer = GymPromptPreparer(client)
    client.put("u1", [1, 2], [3, 2])
    params = {
        "ng_capture": _admission(prev_len=4),
        PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [1, 2, 9, 2],
        PREFIX_EOS_TOKEN_ID_FIELD: [2],
    }
    result = preparer.prepare_prompt([1, 2, 9, 2, 5], offload_params=params)
    assert result.prompt == [1, 2, 3, 2, 5]
    assert result.offload_params["ng_capture"]["required_prefix_token_ids"] == [1, 2, 3, 2]
    assert params["ng_capture"].get("required_prefix_token_ids") is None  # input left alone
    root = _admission(
        mode="text", parent_call_id=None, prev_len=0, staging_chain=[], parent_chain_hash=None
    )
    for passthrough in ({}, {"ng_capture": root}):
        result = GymPromptPreparer(client).prepare_prompt([7, 8], offload_params=passthrough)
        assert (result.prompt, result.offload_params) == ([7, 8], passthrough)
    failures = [
        (
            {**params, "ng_capture": _admission(staging_chain=["absent"])},
            LookupError,
            "does not hold",
        ),
        ({**params, "ng_capture": _admission(prev_len=3)}, ValueError, "expects a prefix of 3"),
        ({"ng_capture": params["ng_capture"]}, ValueError, "splice metadata"),
        ({**params, PREFIX_TEMPLATE_TOKEN_IDS_FIELD: [1, 9]}, ValueError, "EOS"),
    ]
    for offload_params, error, match in failures:
        with pytest.raises(error, match=match):
            GymPromptPreparer(client).prepare_prompt([1, 2, 9, 2, 5], offload_params=offload_params)
    client.forget(["u1"])
    # Cached from the first resolution: the forgotten parent still serves this preparer.
    assert preparer.prepare_prompt([1, 2, 9, 2, 5], offload_params=params).prompt == [1, 2, 3, 2, 5]


@pytest.mark.skipif(importlib.util.find_spec("nemo_gym") is None, reason="needs nemo_gym")
@pytest.mark.parametrize(
    "admission, expected",
    [
        pytest.param(
            {"capture_key": "r", "model_call_id": "c1", "mode": "text"},
            ("staged", 0, 5, "u9"),
            id="root",
        ),
        pytest.param(
            _admission(prev_len=3, required_prefix_token_ids=[1, 2, 3]),
            ("staged", 3, 5, "u9"),
            id="exact-prefix",
        ),
        pytest.param(_admission(prev_len=3), ("capture_failed", 3, 3, None), id="unresolved-chain"),
        pytest.param(
            {"capture_key": "r", "model_call_id": "c1", "mode": "no-such-mode"},
            ("capture_failed", 0, 0, None),
            id="unknown-schema",
        ),
    ],
)
def test_stager_answers_gym_admissions_with_commit_coordinates(store, admission, expected):
    server, client = store
    stager = LedgerPayloadStager(prefix_store=client)
    result = stager.stage(
        "u9", _payload(), finished_metadata=_record(), offload_params={"ng_capture": admission}
    )
    coords = result.response_metadata[NG_COMMIT_COORDS_FIELD]
    assert (
        coords["disposition"],
        coords["prev_len"],
        coords["cum_len"],
        coords.get("staging_key"),
    ) == expected
    assert (coords["capture_key"], coords["weight_version"]) == ("r", 3)
    assert client.get("u9") == ([1, 2, 3, 6], [7])  # published for the next turn's prefix
    # Without an admission the stager only takes custody and publishes nothing.
    assert (
        stager.stage("u10", _payload(), finished_metadata=_record()) == RequestPayloadStageResult()
    )
    assert server.size() == 1
    assert stager.take_capture_failures() == {}
    # A store that stopped answering is no longer acknowledged as capture_failed: stage() raises
    # once the client's retry budget is spent and records the loss (the dead-store test below).


def test_live_store_put_is_one_quiet_attempt_and_the_capture_is_staged(
    store, fake_nemo_gym, caplog
):
    """The normal path: one round trip, no retry line, the admission acknowledged as staged."""
    server, client = store
    stager = LedgerPayloadStager(prefix_store=client)
    with caplog.at_level(logging.WARNING, logger="megatron.rl.prefix_store"):
        result = stager.stage(
            "u9",
            _payload(),
            finished_metadata=_record(),
            offload_params={
                "ng_capture": {"capture_key": "r", "model_call_id": "c1", "mode": "text"}
            },
        )
    coords = result.response_metadata[NG_COMMIT_COORDS_FIELD]
    assert (coords["disposition"], coords["staging_key"], coords["cum_len"]) == ("staged", "u9", 5)
    assert caplog.records == []  # no retry, nothing slow
    assert client.get("u9") == ([1, 2, 3, 6], [7]) and server.size() == 1
    assert stager.take_capture_failures() == {} and set(stager.drain()) == {"u9"}


def test_put_retries_through_a_frozen_store_and_the_episode_is_captured(
    store, fake_nemo_gym, caplog
):
    """A store that stalls longer than one attempt (the profiler held rank 0's interpreter 5-6 s)
    costs retries, not the episode: the put lands after the thaw and the capture is staged."""
    server, _ = store
    client = PrefixStoreClient(server.address, timeout_ms=150, retry_budget_s=10.0)
    stager = LedgerPayloadStager(prefix_store=client)
    os.kill(server.pid, signal.SIGSTOP)  # the store process cannot answer, like a GIL-held server
    thaw = threading.Timer(0.6, os.kill, args=(server.pid, signal.SIGCONT))
    thaw.start()
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="megatron.rl.prefix_store"):
            result = stager.stage(
                "u9",
                _payload(),
                finished_metadata=_record(),
                offload_params={
                    "ng_capture": {"capture_key": "r", "model_call_id": "c1", "mode": "text"}
                },
            )
        elapsed = time.monotonic() - started
    finally:
        thaw.join()
        os.kill(server.pid, signal.SIGCONT)  # harmless when already running
    coords = result.response_metadata[NG_COMMIT_COORDS_FIELD]
    assert (coords["disposition"], coords["staging_key"]) == ("staged", "u9")
    assert elapsed >= 0.5, elapsed  # the put waited for the thaw instead of giving up
    assert client.get("u9") == ([1, 2, 3, 6], [7])
    retries = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
    assert retries and "put for 'u9'" in retries[0] and "attempt 1 failed" in retries[0]
    assert any("took" in r.getMessage() and "attempt(s)" in r.getMessage() for r in caplog.records)
    assert stager.take_capture_failures() == {}


def test_dead_store_raises_naming_the_episode_and_keeps_the_record(caplog):
    """When the budget is spent the stager raises with the request and episode ids, records the
    loss for the trainer and keeps the ledger record; it never acknowledges capture_failed."""
    client = PrefixStoreClient(DEAD_STORE, timeout_ms=100, retry_budget_s=0.35)
    stager = LedgerPayloadStager(prefix_store=client)
    admission = _admission(prev_len=3, required_prefix_token_ids=[1, 2, 3])
    started = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="megatron.rl.rollout_bank"):
        with pytest.raises(PrefixStoreError) as info:
            stager.stage(
                "u11",
                _payload(),
                finished_metadata=_record(),
                offload_params={"ng_capture": admission},
            )
    elapsed = time.monotonic() - started
    message = str(info.value)
    assert "put for 'u11'" in message and "attempt(s)" in message and "budget" in message
    assert 0.35 <= elapsed < 3.0, elapsed  # bounded by the budget, not by an attempt count
    failures = stager.take_capture_failures()
    assert set(failures) == {"u11"}
    assert "capture_key='r'" in failures["u11"] and "model_call_id='c2'" in failures["u11"]
    assert stager.take_capture_failures() == {}  # taken once
    assert set(stager.drain()) == {"u11"}  # the ledger record itself is intact
    assert any("lost" in r.getMessage() and "u11" in r.getMessage() for r in caplog.records)


def test_ledger_drain_refuses_a_batch_with_a_lost_capture(monkeypatch):
    """Rank 0's ledger drain gathers every engine's lost captures and raises with their ids."""
    from megatron.rl.inference import megatron as megatron_interface

    stager = LedgerPayloadStager(
        prefix_store=PrefixStoreClient(DEAD_STORE, timeout_ms=50, retry_budget_s=0.05)
    )
    with pytest.raises(PrefixStoreError):
        stager.stage(
            "u11",
            _payload(),
            finished_metadata=_record(),
            offload_params={"ng_capture": _admission()},
        )
    stager.stage("u12", _payload(), finished_metadata=_record())  # plain custody, no admission
    interface = megatron_interface.MegatronLocal.model_construct(host="h", port=1)
    interface._inference_engine = SimpleNamespace(payload_stager=stager)
    other_rank = ({"u20": object()}, {})  # one record, no lost capture on that engine

    def gather_object(obj, shards, dst=0):
        shards[0], shards[1] = obj, other_rank

    monkeypatch.setattr(megatron_interface.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(megatron_interface.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(megatron_interface.dist, "gather_object", gather_object)
    with pytest.raises(RuntimeError, match="Gym capture lost for 1 request.*u11.*capture_key='r'"):
        interface.merge_global_request_ledgers()
    # The loss was taken with the drain: the next window merges every shard and is clean.
    stager.stage("u13", _payload(), finished_metadata=_record())
    assert set(interface.merge_global_request_ledgers()) == {"u13", "u20"}
