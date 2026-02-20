# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging

import torch.distributed as dist
from pydantic import PrivateAttr

from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.headers import EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import log_single_rank
from megatron.training.global_vars import get_args, get_tokenizer

from ..inference.inference_interface import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from ..server.api import InferenceServer

logger = logging.getLogger(__name__)


class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    host: str
    port: int

    _server_task: asyncio.Task = PrivateAttr(None)
    _shutdown_event: asyncio.Event = PrivateAttr(None)
    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)
    _rl_kv_cache_management_mode: KVCacheManagementMode = PrivateAttr(None)

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:

        assert self._server_task is not None, "Inference server is not initialized"
        tokenizer = get_tokenizer()
        args = get_args()

        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=f"http://{self.host}:{self.port}", api_key="NONE")

        # Things that may be problematic when doign this switch
        # - Add BOS token
        # - Skip prompt logprobs
        response = await client.chat.completions.create(
            model="",
            messages=[message.model_dump() for message in request.prompt],
            temperature=request.generation_args.temperature or 1.0,
            top_p=request.generation_args.top_p or 0.0,
            n=1,
            logprobs=True,
            extra_body={
                "skip_prompt_log_probs": True,
                "add_BOS": (not args.rl_skip_bos_token and tokenizer.bos is not None),
            },
        )

        choice = response.choices[0]

        return InferenceResponse(
            # TODO: Handle tool calls and reasoning in LLMChatMessage
            response=LLMChatMessage(**choice.message.model_dump(include={'role', 'content'})),
            raw_text=choice.raw_text,
            token_ids=choice.prompt_token_ids + choice.generation_token_ids,
            logprobs=choice.generation_log_probs,
            prompt_length=len(choice.prompt_token_ids),
        )

    @classmethod
    async def launch(cls, model: GPTModel, **kwargs):
        # Import here to avoid circular imports
        from megatron.inference.utils import get_dynamic_inference_engine

        args = get_args()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        inference_engine: DynamicInferenceEngine = get_dynamic_inference_engine(model=model)
        dp_addr = await inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=41521, launch_inference_coordinator=True,
        )

        if dist.get_rank() == 0:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server.flask_server import run_flask_server_on_client
            loop = asyncio.get_event_loop()
            client = InferenceClient(inference_coordinator_address=dp_addr)
            await client.start()
            shutdown_event = asyncio.Event()
            server_task = loop.create_task(run_flask_server_on_client(
                client=client,
                tokenizer=inference_engine.controller.tokenizer,
                flask_port=kwargs.get('port', 8294),
                parsers=[],
                verbose=kwargs.get('verbose', False),
                shutdown_event=shutdown_event,
            ))
        else:
            client = None
            server_task = None
            shutdown_event = None

        launched_server = cls(**kwargs)
        launched_server._client = client
        launched_server._server_task = server_task
        launched_server._shutdown_event = shutdown_event
        launched_server._inference_engine = inference_engine
        launched_server._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )

        return launched_server

    async def kill(self):
        """Graceful shutdown from any engine state.

        Navigates the state machine to PAUSED (the prerequisite for STOP),
        shuts down the Flask server, then stops the engines.

        State navigation:
            RUNNING    → send PAUSE → wait for PAUSED
            PAUSING    → wait for PAUSED (EP consensus in progress)
            PAUSED     → ready for STOP
            SUSPENDING → wait for SUSPENDED → send RESUME → wait for PAUSED
            SUSPENDED  → send RESUME → wait for PAUSED
            RESUMING   → wait for PAUSED (DP all-reduce in progress)
            STOPPING   → wait for STOPPED (already shutting down)
            STOPPED    → nothing to do
        """
        engine = self._inference_engine

        if engine.state == EngineState.STOPPED:
            return

        if dist.get_rank() == 0:
            # Navigate to PAUSED based on current state.
            state = engine.state
            if state == EngineState.RUNNING:
                await self._client.pause_engines()
            elif state == EngineState.SUSPENDED:
                self._client.resume_engines()
            elif state == EngineState.SUSPENDING:
                await engine.suspended.wait()
                self._client.resume_engines()
            # PAUSING → will reach PAUSED after EP consensus
            # RESUMING → will reach PAUSED after DP all-reduce
            # PAUSED → already there
            # STOPPING → will reach STOPPED

            # Wait for a stable state we can act on.
            while engine.state not in (EngineState.PAUSED, EngineState.STOPPED):
                await asyncio.sleep(0.02)

            if engine.state != EngineState.STOPPED:
                # Shut down Flask server (drain HTTP connections, clean up executor).
                self._shutdown_event.set()
                await self._server_task
                # Stop engines (DP all-reduce among all ranks).
                self._client.stop_engines()
                self._client.stop()

        # All ranks: wait for engine loop to exit.
        await engine.stopped.wait()

    async def force_kill(self):
        """Emergency cleanup when graceful shutdown fails.

        Cancels tasks, terminates processes, and closes sockets directly
        without coordinating through the state machine.
        """
        engine = self._inference_engine

        if dist.get_rank() == 0:
            self._shutdown_event.set()
            await self._server_task
        # Step 3: Stop engines (DP all-reduce among all ranks)
        if dist.get_rank() == 0:
            self._client.stop_engines()
            self._client.stop()
        await self._inference_engine.stopped.wait()
        # Step 3: Cleanup — engine loop's finally already closed sockets.
        # Join the coordinator process and close the client.
        self._inference_engine.stop()
        if dist.get_rank() == 0:
            self._client.stop()

    async def suspend(self):
        # Step 1: Global pause (ACK/re-ACK through coordinator)
        if dist.get_rank() == 0:
            await self._client.pause_engines()
        await self._inference_engine.paused.wait()
        # Step 2: Suspend (DP all-reduce among all ranks)
        if dist.get_rank() == 0:
            self._client.suspend_engines()
        await self._inference_engine.suspended.wait()

    async def resume(self):
        # Step 1: Resume GPU (DP all-reduce among all ranks)
        if dist.get_rank() == 0:
            self._client.resume_engines()
        await self._inference_engine.paused.wait()
        # Step 2: Unpause (simple broadcast)
        if dist.get_rank() == 0:
            self._client.unpause_engines()
        await self._inference_engine.running.wait()
