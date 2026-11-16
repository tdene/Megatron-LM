# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import gc
import json
import logging
import socket
import warnings
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from pydantic import ValidationError

from megatron.rl import GenericGenerationArgs
from megatron.rl.agent import responses_env_agent
from megatron.rl.agent.api import TokenRollout
from megatron.rl.agent.registry import get_agent_class
from megatron.rl.agent.responses_env_agent import (
    CAPTURE_CONTROL_PREFIX,
    ROLLOUT_ID_KEY,
    ROLLOUT_ID_PATTERN,
    EnvConnectExhausted,
    EnvRunRequest,
    EpisodeKeys,
    ResponsesEnvAgent,
    RunResult,
    curriculum_collections,
    episode_keys_from_manifest,
    generation_args_to_run_payload,
    new_rollout_id,
    next_curriculum_index,
    resolve_curriculum_training_state,
    run_result_to_evaluation_response,
)
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask

TOKEN_ENV = "NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN"


def _write_dataset(tmp_path, text=None):
    rows = (
        {"responses_create_params": {"input": f"q {i}"}, "problem_id": f"p{i}"} for i in range(4)
    )
    path = tmp_path / "dataset.jsonl"
    path.write_text((text if text is not None else "\n".join(map(json.dumps, rows))) + "\n")
    return str(path)


def _agent(tmp_path, **kwargs):
    kwargs.setdefault("agent_name", "test_env")
    if "dataset_file" not in kwargs:
        kwargs["dataset_file"] = _write_dataset(tmp_path)
    kwargs.setdefault("env_server_host_port", "envhost:11000")
    kwargs.setdefault("capture_control_host_port", "policyhost:12000")
    return ResponsesEnvAgent(**kwargs)


def _row(call_id, parent=None, prev_len=0, cum_len=4, response_id=None, **extra):
    """One manifest row; the staging key is the engine uid the ledger join pops."""
    return {
        "model_call_id": call_id,
        "parent_call_id": parent,
        "prev_len": prev_len,
        "cum_len": cum_len,
        "staging_key": f"uid-{call_id}",
        "response_id": response_id or f"resp-{call_id}",
        **extra,
    }


def _manifest(*rows, failures=(), **extra):
    return {"capture_key": "r", "records": list(rows), "failures": list(failures), **extra}


# A three-turn conversation (c1 -> c2 -> c3) plus a retried second turn (c2b, a dead branch).
_CHAIN = _manifest(
    _row("c1", cum_len=4), _row("c2", "c1", 4, 9), _row("c2b", "c1", 4, 8), _row("c3", "c2", 9, 15)
)


def _run_result(reward=1.0, response_id="resp-c3", **extra):
    return RunResult.model_validate(
        {"reward": reward, "response": {"id": response_id, "output": []}, **extra}
    )


class _LocalServer:
    """A real HTTP/1.1 server on 127.0.0.1 (aiohttp.web) standing in for the environment
    server and the policy model server, so the agent's whole client path runs against it.

    ``serve`` points an agent at it and installs the ``async (web.Request) -> response``
    handler; every request is logged as (method, path, client port) so tests can count
    connections; the agents served are closed on exit.
    """

    def __init__(self):
        self.handler = None
        self.requests = []
        self.host_port = None
        self._agents = []

    async def __aenter__(self):
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._dispatch)
        # A client that gives up (rollout_timeout_s) cancels its handler; teardown is short.
        self._runner = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=0.5)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()  # an OS-assigned free port
        host, port = self._runner.addresses[0][:2]
        self.host_port = f"{host}:{port}"
        return self

    async def __aexit__(self, *exc):
        for agent in self._agents:
            await _close(agent)
        await self._runner.cleanup()

    def serve(self, agent, handler):
        self.handler = handler
        agent.env_server_host_port = agent.capture_control_host_port = self.host_port
        self._agents.append(agent)
        return agent

    def forget(self, agent):
        """The test closes (or drops) this agent itself."""
        self._agents.remove(agent)

    def ports(self, suffix):
        """Client ports of the requests whose path ends with ``suffix``: one per connection."""
        return [port for _, path, port in self.requests if path.endswith(suffix)]

    async def _dispatch(self, request):
        peer = request.transport.get_extra_info("peername")
        self.requests.append((request.method, request.path, peer[1]))
        return await self.handler(request)


@pytest_asyncio.fixture(loop_scope="function")
async def http():
    async with _LocalServer() as server:
        yield server


async def _close(agent):
    close = getattr(agent, "aclose", None)  # the parent's agent had no sessions to close
    if close is not None:
        await close()


def _respond(status, body):
    if isinstance(body, dict):
        return web.json_response(body, status=status)
    return web.Response(status=status, text=body)


def _refused_host_port():
    """A loopback address nothing listens on: connecting to it is refused at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


def _request(**generation_args):
    return SimpleNamespace(
        validation=True, generation_args=GenericGenerationArgs(**generation_args)
    )


_GOOD_BODY = {"reward": 0.5, "response": {"id": "resp-c1", "output": []}}
_CAP100 = {"max_output_tokens_per_step": 100}


def _mo(value):
    return {"max_output_tokens": value}


_EVAL_OUTPUT = [
    {"type": "message", "role": "assistant", "content": "part one"},
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "part two"}],
    },
    {"type": "message", "role": "user", "content": "ignored"},
]


class _ControlServer:
    """Scripted env server + policy control routes behind one local HTTP server."""

    def __init__(self, run_responses, manifest=None, manifest_status=200, retire_status=200):
        self.run_responses = list(run_responses)
        self.manifest = manifest
        self.manifest_status = manifest_status
        self.retire_status = retire_status
        self.runs, self.manifests, self.retired, self.auth = [], [], [], []
        self.content_types = []

    async def __call__(self, request):
        path = request.path
        if path == "/run":
            self.content_types.append(request.content_type)
            self.runs.append(json.loads(await request.read()))
            status, body = self.run_responses.pop(0)
            return _respond(status, body)
        self.auth.append(request.headers.get("authorization"))
        assert path.startswith(CAPTURE_CONTROL_PREFIX + "/rollouts/")
        if path.endswith("/manifest"):
            self.manifests.append(path.split("/")[-2])
            if self.manifest_status != 200:
                return web.Response(status=self.manifest_status, text="gone")
            return web.json_response(self.manifest)
        assert path.endswith("/retire")
        self.content_types.append(request.content_type)
        self.retired.extend(json.loads(await request.read())["rollout_ids"])
        return web.json_response({"removed": [], "absent": []}, status=self.retire_status)


_RUN_C3 = {"reward": 1.0, "response": {"id": "resp-c3", "output": []}}


class _ScriptedControl:
    """Env server + control routes whose control answers follow one script per route.

    ``manifest`` and ``retire`` list the behaviour of each request in turn ("ok" once a script
    is exhausted): "ok" answers at once (the manifest; ``{"removed": [id], "absent": []}``),
    "stall" holds the answer past the agent's control timeout, "drop" closes the connection
    without answering (the client sees ServerDisconnectedError), "absent" is Gym's answer to the
    retire of an already retired rollout (200, the id under ``absent``), "gone" a 410.
    """

    def __init__(self, manifest=(), retire=(), stall_s=0.5):
        self.manifest_script, self.retire_script = list(manifest), list(retire)
        self.stall_s = stall_s
        self.manifests, self.retired = [], []

    async def __call__(self, request):
        if request.path == "/run":
            return web.json_response(_RUN_C3)
        if request.path.endswith("/manifest"):
            self.manifests.append(request.path.split("/")[-2])
            step = self.manifest_script.pop(0) if self.manifest_script else "ok"
            ids, answer = None, web.json_response(_CHAIN)
        else:
            ids = json.loads(await request.read())["rollout_ids"]
            self.retired.extend(ids)
            step = self.retire_script.pop(0) if self.retire_script else "ok"
            answer = web.json_response({"removed": ids, "absent": []})
        if step == "stall":
            await asyncio.sleep(self.stall_s)
        elif step == "drop":
            # The server closes the keep-alive connection before answering; the response built
            # above is never written (the transport is closing).
            request.transport.close()
        elif step == "absent":
            answer = web.json_response({"removed": [], "absent": ids})
        elif step == "gone":
            answer = web.Response(status=410, text="gone")
        return answer


def _agent_records(caplog):
    """(level, message) of the agent's log records, in order."""
    return [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == "megatron.rl.agent.responses_env_agent"
    ]


_AGENT_LOGGER = "megatron.rl.agent.responses_env_agent"


class TestResponsesEnvAgent:

    # (manifest, scored response id, declared response id, expected EpisodeKeys attributes)
    @pytest.mark.parametrize(
        "manifest, scored, declared, expected",
        [
            (
                _CHAIN,
                "resp-c3",
                None,
                dict(
                    turns=["uid-c1", "uid-c2", "uid-c3"],
                    others=["uid-c2b"],
                    failure_reason=None,
                    terminal_selection="response_id",
                ),
            ),
            (
                _CHAIN,
                "resp-c2b",
                None,
                dict(turns=["uid-c1", "uid-c2b"], others=["uid-c2", "uid-c3"]),
            ),
            (
                _CHAIN,
                "resp-c3",
                "resp-c2",
                dict(turns=["uid-c1", "uid-c2"], terminal_selection="declared"),
            ),
            (
                _CHAIN,
                "resp-c3",
                "resp-nope",
                dict(
                    turns=[],
                    others=["uid-c1", "uid-c2", "uid-c2b", "uid-c3"],
                    failure_reason="declared_terminal_not_captured",
                    terminal_selection="declared",
                ),
            ),
            # No witness: the parent-link heuristic walks the extended branch to its leaf.
            (
                _manifest(
                    _row("c1"),
                    _row("c2", "c1", 4, 9),
                    _row("c2b", "c1", 4, 8),
                    _row("c3", "c2", 9, 15),
                ),
                None,
                None,
                dict(turns=["uid-c1", "uid-c2", "uid-c3"], terminal_selection="heuristic"),
            ),
            (
                _manifest(_row("c1"), _row("c2", "c1", 4, 9), _row("c2b", "c1", 4, 8)),
                "resp-unknown",
                None,
                dict(turns=[], failure_reason="ambiguous_terminal", terminal_selection="heuristic"),
            ),
            # Token-identical retries of the scored response collapse onto the smallest call id.
            (
                _manifest(
                    _row("c1"),
                    _row("c2", "c1", 4, 9, response_id="resp-x", cumulative_hash="h"),
                    _row("c2a", "c1", 4, 9, response_id="resp-x", cumulative_hash="h"),
                ),
                "resp-x",
                None,
                dict(turns=["uid-c1", "uid-c2"], others=["uid-c2a"]),
            ),
            (
                _manifest(
                    _row("c1"),
                    failures=[{"model_call_id": "c2", "reason": "worker_capture_failed"}],
                ),
                "resp-c1",
                None,
                dict(turns=[], others=["uid-c1"], failure_reason="worker_capture_failed"),
            ),
            # A call that never returned a completion is off-chain and does not poison.
            (
                _manifest(
                    _row("c1"),
                    failures=[
                        {
                            "model_call_id": "c2",
                            "reason": "request_finished_without_staged_coordinates",
                        }
                    ],
                ),
                "resp-c1",
                None,
                dict(turns=["uid-c1"], failure_reason=None),
            ),
            (
                _manifest(_row("c1"), {"model_call_id": "bad"}),
                "resp-c1",
                None,
                dict(turns=[], others=["uid-c1"], failure_reason="invalid_manifest_row"),
            ),
            (
                _manifest(),
                None,
                None,
                dict(turns=[], others=[], failure_reason="no_captured_calls"),
            ),
            (
                _manifest(_row("c1"), _row("c2", "c1", 5, 9)),
                "resp-c2",
                None,
                dict(
                    turns=[], others=["uid-c1", "uid-c2"], failure_reason="parent_length_mismatch"
                ),
            ),
            (
                _manifest(_row("c2", "c0", 4, 9)),
                "resp-c2",
                None,
                dict(turns=[], others=["uid-c2"], failure_reason="missing_parent"),
            ),
        ],
        ids=[
            "scored-id-selects-the-chain",
            "scored-id-on-the-retry-makes-the-other-branch-dead",
            "declared-id-is-authoritative",
            "declared-id-unknown-never-falls-back",
            "heuristic-prefers-the-extended-branch",
            "heuristic-ambiguity-fails-closed",
            "identical-retries-collapse",
            "failure-row-poisons",
            "uncommitted-call-is-off-chain",
            "invalid-row-poisons-keeps-keys",
            "empty-manifest",
            "non-contiguous-chain",
            "missing-parent",
        ],
    )
    def test_episode_keys_from_manifest(self, manifest, scored, declared, expected):
        keys = episode_keys_from_manifest(
            manifest, scored_response_id=scored, declared_response_id=declared
        )
        for attr, value in expected.items():
            assert getattr(keys, attr) == value, attr
        assert sorted(keys.all_keys) == sorted(
            row["staging_key"] for row in manifest["records"] if "staging_key" in row
        )

    def test_new_rollout_id(self):
        rollout_id = new_rollout_id("my env/2")
        assert rollout_id.startswith("my-env-2-") and ROLLOUT_ID_PATTERN.match(rollout_id)
        assert new_rollout_id("").startswith("episode-")
        assert len({new_rollout_id("e") for _ in range(50)}) == 50

    # (result extras, episode keys, agent kwargs, expected rollout attributes)
    @pytest.mark.parametrize(
        "result_extra, keys, agent_kwargs, expected",
        [
            (
                {},
                EpisodeKeys(turns=["u1", "u2"], others=["u3"]),
                {},
                dict(
                    trajectory=None,
                    generation_mask=None,
                    logprobs=None,
                    reward=1.0,
                    env_id="test_env",
                    problem_id="p0",
                    rollout_status="ok",
                    completion_ids=["u1", "u2"],
                    retired_completion_ids=["u3"],
                ),
            ),
            (
                {},
                EpisodeKeys(others=["u1", "u2"], failure_reason="worker_capture_failed"),
                {},
                dict(
                    trajectory=[],
                    reward=0.0,
                    rollout_status="placeholder",
                    failure_reason="worker_capture_failed",
                    completion_ids=[],
                    retired_completion_ids=["u1", "u2"],
                ),
            ),
            (
                {},
                None,
                {},
                dict(
                    rollout_status="placeholder",
                    failure_reason="manifest_unavailable",
                    retired_completion_ids=[],
                ),
            ),
            (
                {"mask_sample": True},
                EpisodeKeys(turns=["u1"], others=["u2"]),
                {},
                dict(
                    trajectory=[],
                    rollout_status="masked",
                    failure_reason="env_masked",
                    completion_ids=[],
                    retired_completion_ids=["u1", "u2"],
                ),
            ),
            (
                {"mask_sample": True, "failure_reason": "agent_timeout"},
                EpisodeKeys(turns=["u1"]),
                {},
                dict(rollout_status="masked", failure_reason="agent_timeout"),
            ),
            (
                {"instance_config": {"mask_sample": True}},
                EpisodeKeys(turns=["u1"]),
                {},
                dict(rollout_status="masked", failure_reason="env_masked"),
            ),
            (
                {"final_eval_time": 12.5},
                EpisodeKeys(turns=["u1"]),
                {"max_output_tokens_per_step": 128},
                dict(rollout_status="graded", generation_cap=128),
            ),
        ],
        ids=[
            "chain-becomes-completion-ids",
            "poisoned-capture-pads-placeholder-draining-all",
            "no-manifest-pads-placeholder",
            "masked-direct-field-drains-all",
            "masked-keeps-server-reason",
            "masked-via-instance-config",
            "graded-with-generation-cap",
        ],
    )
    def test_build_token_rollout(self, tmp_path, result_extra, keys, agent_kwargs, expected):
        agent = _agent(tmp_path, **agent_kwargs)
        rollout = agent.build_token_rollout(
            _run_result(**result_extra), "p0", keys, generation_cap=agent.max_output_tokens_per_step
        )
        for attr, value in expected.items():
            assert getattr(rollout, attr) == value, attr
        assert rollout.is_placeholder == (not rollout.completion_ids)

    def test_token_rollout_placeholder(self):
        rollout = TokenRollout.placeholder("e", failure_reason="http_500", generation_cap=7)
        assert (rollout.trajectory, rollout.generation_mask, rollout.reward) == ([], [], 0.0)
        assert (rollout.problem_id, rollout.rollout_status) == ("placeholder", "placeholder")
        assert (rollout.failure_reason, rollout.generation_cap) == ("http_500", 7)

    # (generation args, agent kwargs, pre-existing params, expected params subset, warns);
    # _CAP100 caps the agent at 100 output tokens per step, _mo(v) = {"max_output_tokens": v}.
    @pytest.mark.parametrize(
        "ga, agent_kwargs, existing, expected, warns",
        [
            ({"temperature": 0.7, "top_p": 0.9}, {}, {}, {"temperature": 0.7, "top_p": 0.9}, False),
            ({"top_k": 5}, {}, {}, {}, True),
            ({}, _CAP100, {}, _mo(100), False),
            ({}, _CAP100, _mo(50), _mo(50), False),
            ({}, _CAP100, _mo(300), _mo(100), False),
            ({}, {"defer_output_cap_to_engine": True}, _mo(300), _mo(None), False),
            ({}, {}, _mo(300), _mo(300), False),
        ],
        ids=[
            "generation-args-merged",
            "top-k-warns",
            "cap-fills-empty",
            "cap-keeps-smaller",
            "cap-wins-over-larger",
            "defer-nulls-the-cap",
            "no-cap-passthrough",
        ],
    )
    def test_run_payload_and_output_caps(
        self, tmp_path, ga, agent_kwargs, existing, expected, warns
    ):
        agent = _agent(tmp_path, **agent_kwargs)
        row = {"responses_create_params": dict(existing), "problem_id": "p0"}
        with pytest.warns(UserWarning, match="top_k") if warns else nullcontext():
            payload = agent._stamp_output_caps(
                generation_args_to_run_payload(GenericGenerationArgs(**ga), row)
            )
        params = payload["responses_create_params"]
        for key, value in expected.items():
            assert key in params and params[key] == value, key
        # The dataset row itself is never mutated.
        assert row["responses_create_params"] == existing

    @pytest.mark.parametrize(
        "cursor, share, expected",
        [(None, 1.0, (48, 49)), (50, 1.0, (50, 51)), (None, 0.25, (12, 13)), (17, 0.25, (17, 18))],
        ids=[
            "restart-seed",
            "monotonic-never-rewinds",
            "seed-scales-with-share",
            "monotonic-with-share",
        ],
    )
    def test_next_curriculum_index(self, cursor, share, expected):
        assert (
            next_curriculum_index(
                cursor, collections=4, prompts_per_collection=12, prompt_share=share
            )
            == expected
        )

    # (iteration, rl config attrs, global batch size, expected (collections, prompts) or raises)
    @pytest.mark.parametrize(
        "iteration, rl, global_batch_size, expected",
        [
            (3, {"grpo_prompts_per_step": 8}, None, (3, 8)),
            (5, {}, None, AttributeError),
            (5, {"grpo_prompts_per_step": None}, None, AttributeError),
            (8, {"grpo_prompts_per_step": 4, "grpo_group_size": 16}, 32, (4, 4)),
            (
                8,
                {"grpo_prompts_per_step": 4, "grpo_group_size": 16, "grpo_iterations": 2},
                32,
                (2, 4),
            ),
            (1, {"grpo_prompts_per_step": 0}, None, ValueError),
        ],
        ids=[
            "cadence-one-passthrough",
            "missing-grpo-state-raises",
            "explicit-none-grpo-state-raises",
            "batches-per-collection-cadence",
            "grpo-iterations-cadence",
            "invalid-prompts-raises",
        ],
    )
    def test_resolve_curriculum_training_state(self, iteration, rl, global_batch_size, expected):
        train_state = SimpleNamespace(iteration=iteration)
        args = SimpleNamespace(global_batch_size=global_batch_size)
        if expected is AttributeError:
            # Missing/None grpo state routes to get_prompt's logged uniform fallback.
            with pytest.raises(AttributeError, match="grpo_prompts_per_step"):
                resolve_curriculum_training_state(train_state, SimpleNamespace(**rl), args)
        elif expected is ValueError:
            with pytest.raises(ValueError, match="must be positive"):
                resolve_curriculum_training_state(train_state, SimpleNamespace(**rl), args)
        else:
            assert (
                resolve_curriculum_training_state(train_state, SimpleNamespace(**rl), args)
                == expected
            )
        with pytest.raises(AttributeError, match="not initialized"):
            resolve_curriculum_training_state(None, SimpleNamespace(**rl), args)
        assert (
            curriculum_collections(12, 4, group_size=2, grpo_iterations=3, global_batch_size=4) == 2
        )

    @pytest.mark.asyncio
    async def test_curriculum_get_prompt(self, tmp_path, monkeypatch):
        agent = _agent(tmp_path, curriculum_sampling=True)
        train_state = SimpleNamespace(iteration=0)
        run_config = SimpleNamespace(rl=SimpleNamespace(grpo_prompts_per_step=2))
        monkeypatch.setattr("megatron.training.global_vars.get_train_state", lambda: train_state)
        monkeypatch.setattr("megatron.training.global_vars.get_run_config", lambda: run_config)
        monkeypatch.setattr("megatron.training.global_vars.get_args", lambda: SimpleNamespace())
        rows = [await agent.get_prompt(validation=False) for _ in range(3)]
        assert [row["problem_id"] for row in rows] == ["p0", "p1", "p2"]

        # A restart at a later iteration re-seeds the cursor from training progress.
        agent2 = _agent(tmp_path, curriculum_sampling=True)
        train_state.iteration = 1
        assert (await agent2.get_prompt(validation=False))["problem_id"] == "p2"

        # Validation and missing training state fall back to uniform, cursor untouched.
        assert await agent2.get_prompt(validation=True) in agent2.dataset

        def _no_run_config():
            # What get_run_config really raises when Megatron global state is uninitialized.
            raise AssertionError("run config is not initialized.")

        monkeypatch.setattr("megatron.training.global_vars.get_run_config", _no_run_config)
        assert await agent2.get_prompt(validation=False) in agent2.dataset
        assert agent2._curriculum_cursor == 3

    def test_weighted_multi_task_stamps_prompt_share(self, tmp_path):
        agent_cls = get_agent_class("ResponsesEnvAgent")
        assert agent_cls is ResponsesEnvAgent

        def config(name, weight, **kwargs):
            args = {
                "agent_name": name,
                "dataset_file": _write_dataset(tmp_path),
                "env_server_host_port": "h:1",
            }
            return AgentConfig(agent_type=agent_cls, agent_args=args, weight=weight, **kwargs)

        multi = WeightedMultiTask(
            [config("a", 3.0), config("b", 1.0), config("c", 1.0, evaluation_only=True)]
        )
        shares = {a.agent_name: getattr(a, "_prompt_share", None) for a in multi.agents}
        assert shares == {"a": pytest.approx(0.75), "b": pytest.approx(0.25), "c": None}

    @pytest.mark.parametrize(
        "kind, error",
        [
            ("env-var-expanded", None),
            ("env-var-unset", "unexpanded environment variable"),
            ("bad-json", "Error decoding dataset JSON"),
        ],
        ids=["env-var-expanded", "env-var-unset", "bad-json"],
    )
    def test_dataset_loading(self, tmp_path, monkeypatch, kind, error):
        _write_dataset(tmp_path)
        if kind == "env-var-expanded":
            monkeypatch.setenv("TEST_DATA_ROOT", str(tmp_path))
            dataset_file = "${TEST_DATA_ROOT}/dataset.jsonl"
        elif kind == "env-var-unset":
            monkeypatch.delenv("SURELY_UNSET_VAR", raising=False)
            dataset_file = "${SURELY_UNSET_VAR}/dataset.jsonl"
        else:
            valid = '{"responses_create_params": {"input": "q"}, "problem_id": "p"}'
            dataset_file = _write_dataset(tmp_path, text=valid + "\nnot json")
        with pytest.raises(ValueError, match=error) if error else nullcontext():
            assert len(_agent(tmp_path, dataset_file=dataset_file).dataset) == 4

    # (canned responses, agent kwargs, expected failure_reason/detail, expected backoff sleeps)
    @pytest.mark.parametrize(
        "responses, agent_kwargs, failure, detail, sleeps",
        [
            ([(200, _GOOD_BODY)], {}, None, None, []),
            ([(503, "busy"), (503, "busy"), (200, _GOOD_BODY)], {}, None, None, [5, 10]),
            ([(500, "boom"), (500, "boom")], {"http_retries": 2}, "http_500", "boom", [5]),
            ([(404, "gone")], {}, "http_404", "gone", []),
            ([(200, "not json")], {}, "ValidationError", None, []),
            (
                [("hang", None)],
                {"rollout_timeout_s": 0.01, "http_retries": 1},
                "rollout_timeout",
                None,
                [],
            ),
        ],
        ids=[
            "ok-first-try",
            "5xx-retried-then-ok",
            "5xx-exhaustion-fails-as-value",
            "4xx-fails-without-retry",
            "malformed-body-fails-as-value",
            "rollout-timeout-fails-as-value",
        ],
    )
    @pytest.mark.asyncio
    async def test_http_handling(
        self, tmp_path, monkeypatch, http, responses, agent_kwargs, failure, detail, sleeps
    ):
        agent = _agent(tmp_path, **agent_kwargs)
        remaining, seen_sleeps = list(responses), []

        async def handler(request):
            status, body = remaining.pop(0)
            if status == "hang":  # outlast the wall-clock budget (immune to the sleep patch)
                await asyncio.Event().wait()
            return _respond(status, body)

        http.serve(agent, handler)

        async def fake_sleep(seconds):
            seen_sleeps.append(seconds)

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "sleep", fake_sleep)
            response = await agent.get_rollout_response(None, EnvRunRequest(payload={}))

        assert seen_sleeps == sleeps
        if failure is None:
            assert not remaining and response.result.reward == 0.5
        else:
            assert (response.result, response.failure_reason) == (None, failure)
            assert detail is None or response.failure_detail == detail

    @pytest.mark.asyncio
    async def test_connect_failures_refresh_routes_then_hit_the_cap(self, tmp_path):
        refreshes = []

        class RefreshCounting(ResponsesEnvAgent):
            async def refresh_routes(self):
                refreshes.append(1)

        agent = RefreshCounting(
            agent_name="t",
            dataset_file=_write_dataset(tmp_path),
            env_server_host_port=_refused_host_port(),
            connect_failure_cap=5,
            connect_refresh_every=2,
            connect_retry_wait_s=0,
        )
        with pytest.raises(EnvConnectExhausted):
            await agent._post_run({})
        assert len(refreshes) == 2  # at failures 2 and 4; the cap hits at 5

        # Through the pipeline path the exhaustion travels as a failure value.
        response = await agent.get_rollout_response(None, EnvRunRequest(payload={}))
        assert (response.result, response.failure_reason) == (None, "EnvConnectExhausted")
        await _close(agent)

    @pytest.mark.parametrize(
        "kind", ["ok", "row-cap-binds", "http-failure", "manifest-gone", "retire-fails"]
    )
    @pytest.mark.asyncio
    async def test_episode_end_to_end(self, tmp_path, monkeypatch, http, kind):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        if kind == "row-cap-binds":
            # A row-level max_output_tokens below the agent cap must win the stamp AND be
            # what the rollout self-reports as generation_cap (termination-check input).
            row = {"responses_create_params": {"input": "q 0", "max_output_tokens": 32}}
            capped = tmp_path / "capped.jsonl"
            capped.write_text(json.dumps({**row, "problem_id": "p0"}) + "\n")
            agent = _agent(tmp_path, max_output_tokens_per_step=64, dataset_file=str(capped))
        else:
            agent = _agent(tmp_path, max_output_tokens_per_step=64)
        expected_cap = 32 if kind == "row-cap-binds" else 64
        run = (
            (404, "gone")
            if kind == "http-failure"
            else (200, {**_GOOD_BODY, "reward": 1.0, "response": {"id": "resp-c3", "output": []}})
        )
        server = _ControlServer(
            [run] * 5,
            manifest=_CHAIN,
            manifest_status=410 if kind == "manifest-gone" else 200,
            retire_status=500 if kind == "retire-fails" else 200,
        )
        http.serve(agent, server)

        params = await agent.prepare_group_rollout(_request(temperature=0.7, top_p=0.9))
        rollout = await params.build_rollout(await params.run_episode())

        body = server.runs[0]
        assert body["responses_create_params"]["temperature"] == 0.7
        assert body["responses_create_params"]["max_output_tokens"] == expected_cap
        rollout_id = body[ROLLOUT_ID_KEY]
        assert rollout_id.startswith("test_env-") and ROLLOUT_ID_PATTERN.match(rollout_id)
        # The manifest is read (and the ledger retired) for every episode, failed ones included.
        assert server.manifests == [rollout_id]
        assert set(server.auth) == {"Bearer s3cret"}
        assert server.retired == ([] if kind == "manifest-gone" else [rollout_id])
        if kind == "http-failure":
            assert (rollout.rollout_status, rollout.failure_reason) == ("placeholder", "http_404")
            # Whatever Gym recorded for the failed episode is still drained from the ledger.
            assert sorted(rollout.retired_completion_ids) == [
                "uid-c1",
                "uid-c2",
                "uid-c2b",
                "uid-c3",
            ]
            assert rollout.env_id == "test_env"
            # Failure placeholders keep the real problem_id of the drawn row.
            assert rollout.problem_id in {"p0", "p1", "p2", "p3"}
        elif kind == "manifest-gone":
            assert (rollout.failure_reason, rollout.retired_completion_ids) == (
                "manifest_unavailable",
                [],
            )
        else:
            assert rollout.trajectory is None
            assert rollout.completion_ids == ["uid-c1", "uid-c2", "uid-c3"]
            assert rollout.retired_completion_ids == ["uid-c2b"]
            assert rollout.generation_cap == expected_cap
            assert not rollout.is_placeholder
        if kind == "ok":
            # The pipeline runs every rollout of a group from the same params: each episode gets
            # its own capture key and reads its own manifest.
            episodes = [await params.run_episode() for _ in range(2)]
            built = [await params.build_rollout(episode) for episode in episodes]
            keys = [run[ROLLOUT_ID_KEY] for run in server.runs]
            assert len(set(keys)) == 3 and server.manifests == keys and server.retired == keys
            assert [r.completion_ids for r in built] == [["uid-c1", "uid-c2", "uid-c3"]] * 2
            # The non-grouped path composes the same episode primitives.
            request = _request()
            request.num_rollouts = 2
            rollouts = await agent.get_reward_rollouts(request)
            assert [r.completion_ids for r in rollouts] == [["uid-c1", "uid-c2", "uid-c3"]] * 2
            assert len({run[ROLLOUT_ID_KEY] for run in server.runs}) == 5

    @pytest.mark.asyncio
    async def test_missing_control_token_fails_the_episode_as_a_value(
        self, tmp_path, monkeypatch, http
    ):
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        agent = _agent(tmp_path)
        http.serve(agent, _ControlServer([(200, _GOOD_BODY)], manifest=_CHAIN))
        params = await agent.prepare_group_rollout(_request())
        rollout = await params.build_rollout(await params.run_episode())
        assert (rollout.rollout_status, rollout.failure_reason) == (
            "placeholder",
            "manifest_unavailable",
        )

    @pytest.mark.asyncio
    async def test_evaluation_reports_every_recorded_key(self, tmp_path, monkeypatch, http):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        agent = _agent(tmp_path)
        server = _ControlServer([(200, _GOOD_BODY)], manifest=_CHAIN)
        http.serve(agent, server)
        response = await agent._evaluation(agent.dataset[0], agent.dataset[0], _request())
        [entry] = response.results
        assert (entry.reward, entry.problem_id) == (0.5, "p0")
        assert sorted(entry.completion_ids) == ["uid-c1", "uid-c2", "uid-c2b", "uid-c3"]
        assert server.retired == [server.runs[0][ROLLOUT_ID_KEY]]

    @pytest.mark.parametrize(
        "input_value, expected_prompt",
        [
            ("2+2?", "2+2?"),
            (
                [
                    {"type": "message", "role": "system", "content": "be brief"},
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "2+2?"}],
                    },
                    {"type": "function_call", "name": "ignored"},
                ],
                [("system", "be brief"), ("user", "2+2?")],
            ),
        ],
        ids=["string-input-passes-through", "message-items-converted"],
    )
    def test_evaluation_response_building(self, input_value, expected_prompt):
        result = RunResult.model_validate(
            {
                "reward": 1.0,
                "response": {"id": "resp-1", "output": _EVAL_OUTPUT},
                "responses_create_params": {"input": input_value},
            }
        )
        response = run_result_to_evaluation_response(result, "p0", "test_env", ["u1"])

        assert response.env_id == "test_env"
        [entry] = response.results
        assert (entry.reward, entry.problem_id, entry.completion_ids) == (1.0, "p0", ["u1"])
        assert entry.response.content == "part one\npart two"
        prompt = (
            entry.prompt
            if isinstance(expected_prompt, str)
            else [(m.role, m.content) for m in entry.prompt]
        )
        assert prompt == expected_prompt

    @pytest.mark.asyncio
    async def test_two_http_sessions_configured_for_held_episodes(self, tmp_path):
        agent = _agent(tmp_path, http_connect_timeout_s=7.0, capture_control_timeout_s=3.0)
        episodes, control = agent._episode_http_session(), agent._control_http_session()
        try:
            assert episodes is not control
            # Created once: every call returns the same session.
            assert agent._episode_http_session() is episodes
            assert agent._control_http_session() is control
            # Episodes hold their connection for the whole run: no connection limit, no read or
            # total timeout; only connection establishment is bounded.
            assert episodes.connector.limit == 0
            assert (episodes.timeout.total, episodes.timeout.sock_read) == (None, None)
            assert episodes.timeout.connect == 7.0
            # Control calls are short, reuse keep-alive connections, and are bounded in total.
            assert control.timeout.total == 3.0
            assert control.connector.limit > 0 and not control.connector.force_close
        finally:
            await agent.aclose()
        assert episodes.closed and control.closed

    @pytest.mark.asyncio
    async def test_wire_bodies_are_json(self, tmp_path, monkeypatch, http):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        server = _ControlServer([(200, _GOOD_BODY)], manifest=_CHAIN)
        agent = http.serve(_agent(tmp_path), server)
        params = await agent.prepare_group_rollout(_request(temperature=0.7))
        await params.build_rollout(await params.run_episode())
        # The /run body and the retire body are JSON objects declared as such, parsed back to
        # the objects the agent built; every control call carries the bearer token.
        assert server.content_types == ["application/json"] * 2
        assert server.runs[0]["responses_create_params"]["temperature"] == 0.7
        assert server.auth == ["Bearer s3cret"] * 2

    @pytest.mark.asyncio
    async def test_held_run_outlasts_the_control_timeout_unless_rollout_timeout_is_set(
        self, tmp_path, http
    ):
        async def slow_run(request):
            await asyncio.sleep(0.3)
            return web.json_response(_GOOD_BODY)

        # No read or total timeout on /run: a response slower than a control call's whole
        # budget still arrives.
        agent = http.serve(_agent(tmp_path, capture_control_timeout_s=0.05), slow_run)
        response = await agent.get_rollout_response(None, EnvRunRequest(payload={}))
        assert response.result.reward == 0.5
        # rollout_timeout_s is the opt-in wall-clock bound, with the parent's failure reason.
        bounded = http.serve(_agent(tmp_path, rollout_timeout_s=0.05, http_retries=1), slow_run)
        response = await bounded.get_rollout_response(None, EnvRunRequest(payload={}))
        assert (response.result, response.failure_reason) == (None, "rollout_timeout")

    @pytest.mark.asyncio
    async def test_control_call_is_bounded_by_its_total_timeout(self, tmp_path, monkeypatch, http):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")

        async def handler(request):
            if request.path == "/run":
                return web.json_response(_GOOD_BODY)
            await asyncio.sleep(0.3)
            return web.json_response(_CHAIN)

        agent = http.serve(_agent(tmp_path, capture_control_timeout_s=0.05), handler)
        params = await agent.prepare_group_rollout(_request())
        rollout = await params.build_rollout(await params.run_episode())
        assert (rollout.rollout_status, rollout.failure_reason) == (
            "placeholder",
            "manifest_unavailable",
        )

    @pytest.mark.asyncio
    async def test_control_calls_reuse_one_connection_beside_held_episodes(
        self, tmp_path, monkeypatch, http
    ):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        held, release = 32, asyncio.Event()

        async def handler(request):
            if request.path == "/run":
                await release.wait()
                return web.json_response(_GOOD_BODY)
            return web.json_response(_CHAIN)

        agent = http.serve(_agent(tmp_path), handler)
        episodes = [asyncio.ensure_future(agent._post_run({})) for _ in range(held)]
        while len(http.ports("/run")) < held:
            await asyncio.sleep(0.01)
        for i in range(5):
            assert await agent._control("GET", f"/rollouts/r{i}/manifest") == _CHAIN
        release.set()
        assert [result.reward for result in await asyncio.gather(*episodes)] == [0.5] * held
        run_ports, control_ports = http.ports("/run"), http.ports("/manifest")
        # The 32 held episodes sit on 32 connections; the 5 control calls made while they are
        # held go over ONE other connection, not through the episode session (whose keep-alive
        # the held connections defeated on the parent's client: a new connection per call).
        assert len(set(run_ports)) == held
        assert len(control_ports) == 5 and len(set(control_ports)) == 1
        assert not set(control_ports) & set(run_ports)
        assert agent._control_http_session() is not agent._episode_http_session()

    @pytest.mark.asyncio
    async def test_held_episodes_run_at_once_on_their_own_connections(self, tmp_path, http):
        held, arrived, release = 200, [], asyncio.Event()

        async def handler(request):
            arrived.append(request.path)
            if len(arrived) == held:
                release.set()  # every episode is in flight before any one of them completes
            await release.wait()
            return web.json_response(_GOOD_BODY)

        agent = http.serve(_agent(tmp_path), handler)
        responses = await asyncio.wait_for(
            asyncio.gather(
                *[agent.get_rollout_response(None, EnvRunRequest(payload={})) for _ in range(held)]
            ),
            timeout=30,
        )
        assert [response.result.reward for response in responses] == [0.5] * held
        assert len(set(http.ports("/run"))) == held

    # The three failure kinds of the parent's transport-error and status paths, through real
    # sockets: the same retries (the recorded sleeps), the same final outcome.
    @pytest.mark.parametrize("kind", ["connection-refused", "reset-mid-body", "http-500"])
    @pytest.mark.asyncio
    async def test_failure_mapping(self, tmp_path, monkeypatch, http, kind):
        seen_sleeps, runs = [], []

        async def handler(request):
            runs.append(request.path)
            if kind == "reset-mid-body" and len(runs) == 1:
                response = web.StreamResponse()
                response.content_length = 4000
                await response.prepare(request)
                await response.write(b'{"reward": 0.5, ')
                request.transport.abort()  # RST with the body unfinished
                return response
            if kind == "http-500":
                return web.Response(status=500, text="boom")
            return web.json_response(_GOOD_BODY)

        agent = http.serve(
            _agent(tmp_path, http_retries=2, connect_failure_cap=3, connect_retry_wait_s=0), handler
        )
        if kind == "connection-refused":
            agent.env_server_host_port = _refused_host_port()

        async def fake_sleep(seconds):
            seen_sleeps.append(seconds)

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "sleep", fake_sleep)
            response = await agent.get_rollout_response(None, EnvRunRequest(payload={}))
        if kind == "connection-refused":
            # Re-POSTed after each refused connect (the connect wait), a value at the cap.
            assert (response.result, response.failure_reason) == (None, "EnvConnectExhausted")
            assert seen_sleeps == [0, 0] and runs == []
        elif kind == "reset-mid-body":
            # A body cut short is a transport error: re-POSTed once, then the result.
            assert response.result.reward == 0.5
            assert seen_sleeps == [0] and runs == ["/run", "/run"]
        else:
            # A 5xx is retried with the backoff, then fails as a value with its detail.
            assert (response.failure_reason, response.failure_detail) == ("http_500", "boom")
            assert seen_sleeps == [5] and runs == ["/run", "/run"]

    @pytest.mark.asyncio
    async def test_aclose_closes_both_sessions_and_leaves_no_unclosed_warning(
        self, tmp_path, monkeypatch, http
    ):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        agent = http.serve(_agent(tmp_path), _ControlServer([(200, _GOOD_BODY)], manifest=_CHAIN))
        params = await agent.prepare_group_rollout(_request())
        assert not (await params.build_rollout(await params.run_episode())).is_placeholder
        episodes, control = agent._episode_http_session(), agent._control_http_session()
        await agent.aclose()
        assert episodes.closed and control.closed
        await agent.aclose()  # idempotent
        # A closed agent is dropped silently; the same check does catch sessions left open.
        reported = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: reported.append(context["message"]))
        try:
            http.forget(agent)
            await asyncio.sleep(0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                del agent, params, episodes, control
                gc.collect()
            assert not [w for w in caught if "Unclosed" in str(w.message)] and not reported
            leaky = _agent(tmp_path)
            leaky._episode_http_session()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                del leaky
                gc.collect()
            assert any("Unclosed client session" in str(w.message) for w in caught)
        finally:
            loop.set_exception_handler(None)

    @pytest.mark.asyncio
    async def test_weighted_multi_task_aclose_closes_every_sub_agent(self, tmp_path):
        agent_cls = get_agent_class("ResponsesEnvAgent")

        def config(name, weight, **kwargs):
            args = {
                "agent_name": name,
                "dataset_file": _write_dataset(tmp_path),
                "env_server_host_port": "h:1",
            }
            return AgentConfig(agent_type=agent_cls, agent_args=args, weight=weight, **kwargs)

        multi = WeightedMultiTask([config("a", 1.0), config("b", 1.0, evaluation_only=True)])
        sessions = [agent._episode_http_session() for agent in multi.agents]
        sessions.append(multi.agents[0]._control_http_session())
        multi.agents.append(SimpleNamespace())  # an agent type without aclose holds no sessions
        await multi.aclose()
        assert all(session.closed for session in sessions)

    # su11 (2026-11-16, saturated GRPO): the trainer's loop was blocked about two minutes across
    # the iteration boundary (suspend, optimizer step, resume) while up to 64 group builds had
    # control calls in flight; aiohttp's total timeout is loop time, so 283 manifest GETs and 42
    # retire POSTs expired on the first turn after the resume although Gym had answered. Here the
    # server stalls the first answer past the control timeout, or drops the connection, and the
    # agent's retry on the live loop gets the answer: the keys, no warning, one counted retry.
    @pytest.mark.parametrize(
        "kind",
        [
            "manifest-stall",
            "retire-stall-already-retired",
            "retire-stall-ok",
            "retire-drop",
            "manifest-drop-twice",
        ],
    )
    @pytest.mark.asyncio
    async def test_control_call_is_retried_after_a_stall_or_a_dropped_connection(
        self, tmp_path, monkeypatch, http, caplog, kind
    ):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        script = {
            "manifest-stall": dict(manifest=["stall"]),
            # The timed-out first POST did retire the rollout on Gym: the second is answered with
            # 200 and the id under "absent" (CaptureLedger.retire: retiring again is a no-op).
            "retire-stall-already-retired": dict(retire=["stall", "absent"]),
            "retire-stall-ok": dict(retire=["stall"]),
            "retire-drop": dict(retire=["drop"]),
            # aiohttp itself re-sends an idempotent GET once after a disconnect; the second drop
            # reaches the agent, whose retry gets the manifest (three GETs on the wire).
            "manifest-drop-twice": dict(manifest=["drop", "drop"]),
        }[kind]
        server = _ScriptedControl(**script)
        agent = http.serve(_agent(tmp_path, capture_control_timeout_s=0.1), server)
        params = await agent.prepare_group_rollout(_request())
        with caplog.at_level(logging.INFO, logger=_AGENT_LOGGER):
            rollout = await params.build_rollout(await params.run_episode())
        # The keys arrived: a trainable rollout with the chain's uids, the dead branch retired.
        assert not rollout.is_placeholder
        assert rollout.completion_ids == ["uid-c1", "uid-c2", "uid-c3"]
        assert rollout.retired_completion_ids == ["uid-c2b"]
        [rollout_id] = set(server.manifests)
        gets = {"manifest-stall": 2, "manifest-drop-twice": 3}.get(kind, 1)
        posts = 1 if kind.startswith("manifest") else 2
        assert (len(http.ports("/manifest")), len(http.ports("/retire"))) == (gets, posts)
        assert server.retired == [rollout_id] * posts
        # A retry that succeeds is counted, not logged: no WARNING (su11 printed 325 per
        # boundary), and the INFO summary is rate limited, so none yet.
        assert _agent_records(caplog) == []
        assert (agent.control_retries, agent.control_retry_failures) == (1, 0)

    @pytest.mark.parametrize(
        "kind",
        ["every-attempt-stalls", "no-retries-configured", "4xx-not-retried", "retry-answered-4xx"],
    )
    @pytest.mark.asyncio
    async def test_exhausted_control_retries_keep_the_parent_warning(
        self, tmp_path, monkeypatch, http, caplog, kind
    ):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        retries, script = {
            "every-attempt-stalls": (1, ["stall", "stall"]),
            "no-retries-configured": (0, ["stall"]),
            "4xx-not-retried": (1, ["gone"]),
            # The retry is answered with a status: Gym's answer, the retried call's failure.
            "retry-answered-4xx": (1, ["stall", "gone"]),
        }[kind]
        server = _ScriptedControl(manifest=script)
        agent = http.serve(
            _agent(tmp_path, capture_control_timeout_s=0.1, capture_control_retries=retries), server
        )
        params = await agent.prepare_group_rollout(_request())
        with caplog.at_level(logging.INFO, logger=_AGENT_LOGGER):
            rollout = await params.build_rollout(await params.run_episode())
        # What the parent left behind: the placeholder without retired ids (the episode's ledger
        # records are never claimed), no retire of a manifest that never came.
        assert (rollout.rollout_status, rollout.failure_reason) == (
            "placeholder",
            "manifest_unavailable",
        )
        assert rollout.retired_completion_ids == []
        [rollout_id] = set(server.manifests)
        # Every attempt stalled: the configured retry was made; a 4xx is Gym's answer and no
        # retries configured is one attempt: today's single GET either way.
        two_attempts = kind in ("every-attempt-stalls", "retry-answered-4xx")
        assert len(http.ports("/manifest")) == (2 if two_attempts else 1)
        assert http.ports("/retire") == []
        # Today's WARNING, same text, once.
        answered = kind in ("4xx-not-retried", "retry-answered-4xx")
        cause = "EnvHTTPError: HTTP 410: gone" if answered else "TimeoutError: "
        assert _agent_records(caplog) == [
            (
                "WARNING",
                f"ResponsesEnvAgent[test_env]: manifest of rollout {rollout_id} unavailable "
                f"({cause}); its requests' ledger records cannot be claimed: "
                f"problem_id={rollout.problem_id}",
            )
        ]
        counted = (1, 1) if two_attempts else (0, 0)
        assert (agent.control_retries, agent.control_retry_failures) == counted

    @pytest.mark.asyncio
    async def test_control_retry_summary_is_one_line_per_interval(
        self, tmp_path, monkeypatch, http, caplog
    ):
        monkeypatch.setenv(TOKEN_ENV, "s3cret")
        monkeypatch.setattr(responses_env_agent, "CONTROL_RETRY_SUMMARY_INTERVAL_S", 0.4)
        # Four episodes at once, as the concurrent assembly has them at a boundary: every first
        # manifest GET stalls, and the first retry to arrive stalls again (retries exhausted).
        server = _ScriptedControl(manifest=["stall"] * 5)
        agent = http.serve(
            _agent(tmp_path, capture_control_timeout_s=0.1, capture_control_retries=1), server
        )
        params = await agent.prepare_group_rollout(_request())

        async def episode():
            return await params.build_rollout(await params.run_episode())

        def summary(retried, failed):
            return (
                "INFO",
                f"ResponsesEnvAgent[test_env]: {retried} control calls retried after a timeout "
                f"or a dropped connection since the last line, {failed} failed",
            )

        with caplog.at_level(logging.INFO, logger=_AGENT_LOGGER):
            rollouts = await asyncio.gather(*[episode() for _ in range(4)])
            assert sorted(rollout.is_placeholder for rollout in rollouts) == [False] * 3 + [True]
            assert (agent.control_retries, agent.control_retry_failures) == (4, 1)
            # Right after the burst: the exhausted call's WARNING, no summary yet.
            assert [level for level, _ in _agent_records(caplog)] == ["WARNING"]
            await asyncio.sleep(0.6)
            # One line for the burst, not one per retry.
            assert _agent_records(caplog)[1:] == [summary(4, 1)]
            # A later retry opens a new window; its line carries only the movement since.
            server.manifest_script[:] = ["stall"]
            assert not (await episode()).is_placeholder
            await asyncio.sleep(0.6)
            assert _agent_records(caplog)[2:] == [summary(1, 0)]
            # Nothing moved: no line.
            await asyncio.sleep(0.6)
            assert len(_agent_records(caplog)) == 3
            # A summary pending at aclose is dropped; the counters keep their totals.
            server.manifest_script[:] = ["stall"]
            assert not (await episode()).is_placeholder
            await agent.aclose()
            await asyncio.sleep(0.6)
            assert len(_agent_records(caplog)) == 3
            assert (agent.control_retries, agent.control_retry_failures) == (6, 1)

    def test_capture_control_retries_field(self, tmp_path):
        agent = _agent(tmp_path)
        # The per-attempt bound keeps its default; two retries by default; none is allowed.
        assert (agent.capture_control_timeout_s, agent.capture_control_retries) == (60.0, 2)
        assert _agent(tmp_path, capture_control_retries=0).capture_control_retries == 0
        with pytest.raises(ValidationError):
            _agent(tmp_path, capture_control_retries=-1)
        assert (agent.control_retries, agent.control_retry_failures) == (0, 0)
