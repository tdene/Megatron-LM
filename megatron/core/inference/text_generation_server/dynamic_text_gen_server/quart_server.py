# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
import logging
import socket
from contextlib import contextmanager

try:
    from quart import Quart
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    HAS_QUART = True
except ImportError as e:
    HAS_QUART = False

import megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints as endpoints
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.utils import trace_async_exceptions

logger = logging.getLogger(__name__)


@contextmanager
def temp_log_level(level, logger=None):
    """Enables temporarily overriding the logging level."""
    logger = logger or logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


@trace_async_exceptions
async def run_server_on_client(
    client: InferenceClient,
    tokenizer,
    port: int,
    parsers: list[str] = None,
    verbose: bool = False,
):
    """Initializes and runs the async Quart server using the provided InferenceClient."""
    if not HAS_QUART:
        raise RuntimeError("Quart not available")

    try:
        hostname = socket.gethostname()
    except Exception as e:
        logger.warning(f"Could not get hostname: {e}")
        hostname = "0.0.0.0"

    app = Quart(__name__)

    # Store client and tokenizer in app config for Blueprints to use
    app.config['client'] = client
    app.config['tokenizer'] = tokenizer
    app.config['parsers'] = parsers
    app.config['verbose'] = verbose

    # Register all blueprints from the 'endpoints' package
    for endpoint in endpoints.__all__:
        app.register_blueprint(endpoint)

    @app.route('/')
    async def health_check():
        return "Megatron Dynamic Inference Server is running."

    config = Config()
    config.keep_alive_timeout = 30.0  # RL inference requests are long; avoid premature disconnects.
    config.backlog = 2048  # Generous margin for burst starts of RL training steps.
    config.h2_max_concurrent_streams = 1024  # Support high HTTP/2 concurrency from RL clients.
    config.bind = [f"0.0.0.0:{port}"]

    # Force logging level to INFO to ensure that hostname is printed
    with temp_log_level(logging.INFO, logger):
        logger.info(f"Starting inference server on http://{hostname}:{port}")
        logger.info(f"Using tokenizer: {type(tokenizer)}")
        logger.info(f"Using parsers: {parsers}")

    await serve(app, config)


@trace_async_exceptions
async def run_server(
    coordinator_addr: str,
    tokenizer,
    rank: int,
    port: int,
    parsers: list[str] = None,
    verbose: bool = False,
):
    """Initializes and runs the async Quart server
    starting an InferenceClient with the provided coordinator address."""
    inference_client = InferenceClient(coordinator_addr)
    await inference_client.start()
    logger.info(f"Rank {rank}: InferenceClient connected.")
    try:
        await run_server_on_client(inference_client, tokenizer, port, parsers, verbose)
    finally:
        await inference_client.stop()
        logger.info(f"Rank {rank}: Inference server and client shut down.")


# Backward-compatible aliases
run_flask_server_on_client = run_server_on_client
run_flask_server = run_server
