# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The Megatron RL prefix store: rank 0's table of finished calls, uid -> packed token ids.

Gym ``token_in`` admissions need the parent call's exact tokens on whichever engine continues
the conversation. Every engine's stager publishes each finished call here (``put``), every
engine's prompt preparer fetches the parent (``get``), and the trainer drops consumed calls
(``forget``). The service is a ZMQ REQ/REP key-value store over msgpack frames.

The server runs in its OWN process, started by rank 0. It used to be a thread of rank 0's
trainer process and shared that process's GIL: any C call that holds the GIL for longer than
a client's timeout (``torch.cuda.cudart().cudaProfilerStop()`` while nsys writes a report took
5-6 s in the 2026-11-12 traced runs) froze the server, and a client that had just sent its put
timed out. Out of process the store answers whatever rank 0's interpreter is doing.

This module deliberately imports nothing from ``megatron`` or ``torch``: the child process runs
it as a script, so it must stay standalone (standard library, ``zmq``, ``msgpack``).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import msgpack
import zmq

logger = logging.getLogger(__name__)

# One store round trip may take this long before the attempt is given up (the REQ socket is then
# replaced: a REQ socket cannot be reused after a failed exchange).
STORE_TIMEOUT_MS = 5000
# A ``put`` keeps retrying failed attempts until this much wall time has passed, then raises
# ``PrefixStoreError``. The default covers every stall seen on the node (the profiler's 5-6 s GIL
# hold on rank 0 was the longest) with a wide margin; a dead store is reported within a minute.
STORE_RETRY_BUDGET_S = 60.0
# A successful round trip slower than this is logged, so stalls short of a timeout stay visible.
STORE_SLOW_MS = 1000
# How long rank 0 waits for the store process to report its address.
STORE_START_TIMEOUT_S = 120.0

HeldCall = tuple[list[int], list[int]]


class PrefixStoreError(RuntimeError):
    """A store round trip failed for good; the message names the op, the uid and the attempts."""


def _serve(sock: zmq.Socket, parent_pid: int | None, stop: threading.Event | None = None) -> None:
    """Answer requests until ``stop`` is set, SIGTERM arrives or the parent process is gone."""
    held: dict[str, bytes] = {}

    def handle(frame: bytes) -> bytes:
        op, uid, payload = msgpack.unpackb(frame, raw=False)
        if op == "put":
            held[uid] = payload
            return msgpack.packb(None)
        if op == "get":
            return held.get(uid, msgpack.packb(None))
        if op == "forget":
            for key in payload:
                held.pop(key, None)
            return msgpack.packb(None)
        if op == "size":
            return msgpack.packb(len(held))
        raise ValueError(f"unknown prefix store op {op!r}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while stop is None or not stop.is_set():
        if parent_pid is not None and os.getppid() != parent_pid:
            logger.info("prefix store: parent %d is gone, exiting", parent_pid)
            break
        if not poller.poll(timeout=100):
            continue
        frame = sock.recv()
        try:
            reply = handle(frame)
        except Exception:  # a bad frame must not take the server down or hang its client
            logger.exception("prefix store rejected a request")
            reply = msgpack.packb({"error": "malformed prefix store request"})
        sock.send(reply)


def _serve_main(argv: list[str] | None = None) -> None:
    """Entry point of the store process: bind, print the address once, serve until told to stop."""
    parser = argparse.ArgumentParser(description="Megatron RL prefix store process")
    parser.add_argument("--host", default=None, help="host name advertised in the address")
    parser.add_argument("--parent-pid", type=int, default=None, help="exit when this process dies")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[prefix-store %(process)d] %(message)s")
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://0.0.0.0")
    # The one line the parent reads; everything else this process says goes to stderr.
    sys.stdout.write(f"tcp://{args.host or socket.gethostname()}:{port}\n")
    sys.stdout.flush()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
        _serve(sock, args.parent_pid, stop)
    finally:
        sock.close(linger=0)
        context.term()


def _server_command(host: str | None) -> tuple[list[str], dict[str, str]]:
    """The interpreter command that runs this module as the store process."""
    env = dict(os.environ)
    script = os.path.abspath(__file__)
    # -P keeps the script's own directory off sys.path: megatron/rl holds a logging.py and a
    # types/ package, which would shadow the standard library for a script started from there.
    if script.endswith(".py") and os.path.exists(script):
        command = [sys.executable, "-P", script]
    else:  # imported from somewhere a script path cannot name: run it as a module on our sys.path
        command = [sys.executable, "-P", "-m", __spec__.name]
        env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    command += ["--parent-pid", str(os.getpid())]
    if host is not None:
        command += ["--host", host]
    return command, env


class PrefixStoreServer:
    """Rank 0's prefix store, served by a child process this object owns.

    The child is a fresh interpreter running this module (no torch, no CUDA, nothing of the
    trainer), so a GIL-holding call or a long garbage collection in rank 0 cannot stall a put.
    It exits on ``close()``, on SIGTERM, or as soon as the parent process is gone.
    """

    def __init__(
        self, host: str | None = None, start_timeout_s: float = STORE_START_TIMEOUT_S
    ) -> None:
        command, env = _server_command(host)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,  # the child's log lines land in the trainer's log
            env=env,
            close_fds=True,
            text=True,
        )
        self.address = self._read_address(start_timeout_s)
        self._client: PrefixStoreClient | None = None

    def _read_address(self, start_timeout_s: float) -> str:
        """The child's first stdout line, read on a helper thread so a stuck child cannot hang us."""
        line: list[str] = []

        def read() -> None:
            line.append(self._process.stdout.readline())

        reader = threading.Thread(target=read, name="prefix-store-address", daemon=True)
        reader.start()
        reader.join(start_timeout_s)
        address = line[0].strip() if line else ""
        if not address.startswith("tcp://"):
            self._process.kill()
            returncode = self._process.wait(timeout=10)
            raise RuntimeError(
                f"prefix store process did not report an address within {start_timeout_s:.0f} s "
                f"(exit code {returncode}, command {self._process.args!r})"
            )
        return address

    @property
    def pid(self) -> int:
        """The store process id (tests freeze it with SIGSTOP to emulate a stalled server)."""
        return self._process.pid

    def size(self) -> int:
        """Held calls, asked of the store process. A method so an empty store is still truthy."""
        if self._client is None:
            self._client = PrefixStoreClient(self.address, retry_budget_s=0.0)
        return int(self._client._request("size"))

    def close(self) -> None:
        """Stop the store process; its table dies with it, as the old thread's did."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        if self._process.stdout is not None:
            self._process.stdout.close()


class PrefixStoreClient:
    """Thread-safe client of ``PrefixStoreServer``; one REQ socket per calling thread.

    ``put`` retries failed attempts for ``retry_budget_s`` of wall time and then raises
    ``PrefixStoreError``; ``get`` (the engine's admission path) and ``forget`` make one attempt.
    """

    def __init__(
        self,
        address: str,
        timeout_ms: int = STORE_TIMEOUT_MS,
        retry_budget_s: float = STORE_RETRY_BUDGET_S,
    ) -> None:
        self.address = address
        self.timeout_ms = int(timeout_ms)
        self.retry_budget_s = float(retry_budget_s)
        self._context = zmq.Context.instance()
        self._local = threading.local()

    def _socket(self, timeout_ms: int | None = None) -> zmq.Socket:
        sock = getattr(self._local, "socket", None)
        if sock is None:
            sock = self._local.socket = self._context.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self.address)
            self._local.timeout_ms = None
        timeout_ms = self.timeout_ms if timeout_ms is None else timeout_ms
        if getattr(self._local, "timeout_ms", None) != timeout_ms:
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self._local.timeout_ms = timeout_ms
        return sock

    def _drop_socket(self) -> None:
        """A REQ socket cannot be reused after a failed exchange: close it, the next call reconnects."""
        sock = getattr(self._local, "socket", None)
        if sock is not None:
            sock.close(linger=0)
        self._local.socket = None

    def close(self) -> None:
        """Close this thread's socket (other threads' sockets close with their thread-locals)."""
        self._drop_socket()

    def _request_once(self, op: str, uid: str | None, payload: Any, timeout_ms: int) -> Any:
        sock = self._socket(timeout_ms)
        sock.send(msgpack.packb([op, uid, payload], use_bin_type=True))
        reply = msgpack.unpackb(sock.recv(), raw=False)
        if isinstance(reply, dict) and "error" in reply:
            raise PrefixStoreError(f"prefix store {op} at {self.address} failed: {reply['error']}")
        return reply

    def _request(
        self, op: str, uid: str | None = None, payload: Any = None, *, budget_s: float = 0.0
    ) -> Any:
        """One round trip, retried on transport errors until ``budget_s`` of wall time is spent."""
        started = time.monotonic()
        deadline = started + budget_s
        attempt = 0
        while True:
            attempt += 1
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            timeout_ms = (
                self.timeout_ms if attempt == 1 else max(1, min(self.timeout_ms, remaining_ms))
            )
            try:
                reply = self._request_once(op, uid, payload, timeout_ms)
            except zmq.ZMQError as error:
                self._drop_socket()
                elapsed = time.monotonic() - started
                if time.monotonic() >= deadline:
                    raise PrefixStoreError(
                        f"prefix store {op} for {uid!r} at {self.address} failed after {attempt} "
                        f"attempt(s) in {elapsed:.1f} s (budget {budget_s:.1f} s): {error}"
                    ) from error
                logger.warning(
                    "prefix store %s for %r at %s: attempt %d failed after %.1f s (%s); "
                    "retrying for up to %.1f s more",
                    op,
                    uid,
                    self.address,
                    attempt,
                    elapsed,
                    error,
                    deadline - time.monotonic(),
                )
                continue
            elapsed_ms = (time.monotonic() - started) * 1000
            if elapsed_ms >= STORE_SLOW_MS or attempt > 1:
                logger.warning(
                    "prefix store %s for %r at %s took %.2f s (%d attempt(s))",
                    op,
                    uid,
                    self.address,
                    elapsed_ms / 1000,
                    attempt,
                )
            return reply

    def ping(self) -> None:
        """Round-trip once; raises when the store is unreachable, so the launch fails early."""
        self._request("get", "")

    def put(self, uid: str, prompt_token_ids: list[int], generated_token_ids: list[int]) -> None:
        """Publish a finished call; retried within the budget, idempotent across attempts."""
        packed = msgpack.packb(
            [list(prompt_token_ids), list(generated_token_ids)], use_bin_type=True
        )
        self._request("put", uid, packed, budget_s=self.retry_budget_s)

    def get(self, uid: str) -> HeldCall | None:
        """One attempt: the caller is the engine's admission path, which must not stall."""
        held = self._request("get", uid)
        return None if held is None else (held[0], held[1])

    def forget(self, uids) -> None:
        uids = list(uids)
        if uids:
            self._request("forget", None, uids)


if __name__ == "__main__":
    _serve_main()
