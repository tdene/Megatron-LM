# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The prefix store client keeps its sockets off the process-wide zmq context.

The inference engine's shutdown terminates ``zmq.Context.instance()``, and ``term()`` returns
only once every socket of the context is closed. A REQ socket the client left open there (one
per calling thread; the launch's ``ping()`` leaves one on the main thread) kept the shutdown
blocked until the launcher killed the trainer, and took every in-flight traceback with it.
"""

import threading

import pytest
import zmq

from megatron.rl.prefix_store import PrefixStoreClient, _serve


@pytest.fixture
def store_address():
    """An in-process prefix store on its own context, served as the store process serves."""
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()
    thread = threading.Thread(target=_serve, args=(sock, None, stop), daemon=True)
    thread.start()
    try:
        yield f"tcp://127.0.0.1:{port}"
    finally:
        stop.set()
        thread.join(timeout=5.0)
        sock.close(linger=0)
        context.term()


def test_the_client_keeps_its_sockets_off_the_shared_context(store_address):
    """After ``ping()`` the client's REQ socket stays open on the calling thread for reuse, as it
    does for the trainer's life; the process-wide context must still terminate at once, as the
    engine's shutdown asks it to."""
    client = PrefixStoreClient(store_address, timeout_ms=5000)
    client.ping()
    shared = zmq.Context.instance()
    assert client._context is not shared
    terminated = threading.Event()

    def terminate():
        shared.term()
        terminated.set()

    threading.Thread(target=terminate, daemon=True).start()
    assert terminated.wait(5.0), "zmq.Context.instance().term() blocked on the client's socket"
    client.close()
