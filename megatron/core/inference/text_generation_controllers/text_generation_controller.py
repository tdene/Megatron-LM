# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import concurrent
import copy
import functools
from collections import defaultdict
from typing import Any, Callable, Dict, Generator, List, Optional, OrderedDict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
    is_pipeline_last_stage,
)
from megatron.core.inference.contexts.dynamic_context import MaxSequenceLengthOverflowError
from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import get_attention_mask, set_decode_expert_padding
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.utils import set_model_to_sequence_parallel
from megatron.core.utils import (
    accepts_parameter,
    get_asyncio_loop,
    get_model_config,
    get_pg_size,
    unwrap_model,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import

    HAVE_TE = True

except ImportError:
    HAVE_TE = False

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.dual_stream import SideDecision


# pylint: disable=line-too-long
class TextGenerationController:
    """The text generation controller (the main sampling loop)

    This class tokenizes the input, runs inference, samples from logits, and detokenizes the output.

    Args:
        inference_wrapped_model (AbstractModelInferenceWrapper): A model that
            is wrapped using the specs given in the abstract_model_inference_wrapper.py
        tokenizer (_type_): Tokenizer used for tokenizing and detokenizing the prompts
    """

    def __init__(self, inference_wrapped_model: AbstractModelInferenceWrapper, tokenizer):
        self.inference_wrapped_model = inference_wrapped_model
        self.model_config = self.inference_wrapped_model.model.config
        inference_config = self.inference_wrapped_model.inference_context.config
        self.tokenizer = tokenizer
        self.num_speculative_tokens = inference_config.num_speculative_tokens

        pg_collection = inference_config.pg_collection
        if pg_collection is not None:
            self.pp_group = pg_collection.pp
        else:
            self.pp_group = parallel_state.get_pipeline_model_parallel_group()

        self.model_is_pipeline_parallel = self.model_config.pipeline_model_parallel_size > 1

        # Use padded vocab size because tokenizer vocab size might pad to nearest power of 2.
        # TODO(ksanthanam): Consider deprecating this check if LLaVAModel is no longer used
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        if isinstance(unwrapped_model, LLaVAModel):
            self.vocab_size = unwrapped_model.language_model.vocab_size
        else:
            self.vocab_size = unwrapped_model.vocab_size

        self.sampling_rng = torch.Generator(device=torch.cuda.current_device())
        self.num_mtp_heads = self._get_mtp_num_heads()
        self.sampling_rng.manual_seed(self.model_config.inference_sampling_seed)

        if (
            self.model_config.cuda_graph_impl == "local"
            and self.model_config.expert_model_parallel_size > 1
            and self.model_config.transformer_impl != "inference_optimized"
        ):
            assert self.model_config.moe_pad_experts_for_cuda_graph_inference, (
                "--moe-pad-experts-for-cuda-graph-inference must be set when using "
                "CUDA graphs with expert parallelism"
            )

        if self.inference_wrapped_model.inference_context.is_dynamic_batching():
            self._init_dynamic_sampling_tensors()

    def _get_mtp_num_heads(self) -> int:
        """Get the number of MTP layers from the model config."""
        model = self.inference_wrapped_model.model
        if hasattr(model, 'config') and hasattr(model.config, 'mtp_num_layers'):
            return model.config.mtp_num_layers or 0
        return 0

    def set_stop_word_finished_ids_callback(self, callback):
        """Set a callback to get request IDs that should be marked as finished due to stop words.

        The callback should have signature: callback(active_request_ids: List[int]) -> Set[int]
        Returns a set of request IDs from active_request_ids that should be marked as finished.

        Args:
            callback: Function that returns request IDs to mark as finished.
        """
        self._get_stop_word_finished_ids_callback = callback

    def _init_dynamic_sampling_tensors(self):
        """Initialize tensors needed for dynamic sampling."""
        context = self.inference_wrapped_model.inference_context
        max_requests = context.max_requests
        if context.config.materialize_only_last_token_logits:
            max_logits = max_requests
        else:
            max_logits = context.max_tokens

        # Callback to get request IDs that should be marked as finished due to stop words
        self._get_stop_word_finished_ids_callback = None

        device = torch.cuda.current_device()
        logits_dtype = self.inference_wrapped_model.config.params_dtype

        self._sampling_backend = "torch"
        self._enable_cuda_graph = self.model_config.cuda_graph_impl == "local"

        # Initialize bookkeeping tensors.
        if self._enable_cuda_graph:
            self._all_logits_cuda = torch.zeros(
                (1, max_logits, self.vocab_size), dtype=logits_dtype, device=device
            )
        else:
            self._all_logits_cuda = None
        self._sampled_tokens_cuda = torch.empty(max_requests, dtype=torch.int64, device=device)

        # Side stream for pre-forward bookkeeping. Work issued here runs concurrently
        # with the forward pass; post-forward consumers synchronize on the event.
        self._pre_forward_bookkeeping_stream = torch.cuda.Stream(device=device)
        self._pre_forward_bookkeeping_event = torch.cuda.Event()

        # Speculative tokens tensor will be allocated later when num_speculative_tokens is set by the engine
        self._accepted_tokens_per_request = None
        # MTP tensor will be allocated later when num_speculative_tokens is set by the engine
        self._sampled_mtp_tokens_cuda = None
        # Last accepted sequence indices for serial MTP computation
        self._last_accepted_seq_indices = None

        # Used for inefficient torch sampling.
        if self._sampling_backend == "torch":
            self._torch_sampling_buckets: List[Tuple] = []

        self._init_mtp_sampling_tensor()

    def _init_mtp_sampling_tensor(self):
        """Initialize the MTP sampling tensor after num_speculative_tokens is set."""
        if self.num_speculative_tokens is not None and self.num_speculative_tokens > 0:
            context = self.inference_wrapped_model.inference_context
            max_requests = context.max_requests
            device = torch.cuda.current_device()
            self._sampled_mtp_tokens_cuda = torch.empty(
                [self.num_speculative_tokens, max_requests], dtype=torch.int64, device=device
            )
            self._accepted_tokens_per_request = (
                torch.ones(
                    [max_requests, self.num_speculative_tokens], dtype=torch.int64, device=device
                )
                * -1
            )
            self._accepted_token_counts_per_request = torch.zeros(
                [max_requests], dtype=torch.int64, device=device
            )

    @staticmethod
    def tokenize_prompt(tokenizer, prompt: str, add_BOS: bool = False) -> List[int]:
        """Utility to tokenize the input prompts.

        Args:
            tokenizer: The tokenizer to use.
            prompt (str): The input prompt.
            add_BOS (bool): Whether to add a BOS token.

        Returns:
            List[int]: Returns the tokenized prompt.
        """

        prompt_tokens = tokenizer.tokenize(prompt)

        if add_BOS:
            assert tokenizer.bos is not None

        while prompt_tokens and prompt_tokens[0] == tokenizer.bos:
            prompt_tokens.pop(0)

        if add_BOS:
            prompt_tokens = [tokenizer.bos] + prompt_tokens

        return prompt_tokens

    @staticmethod
    def detokenize(
        tokenizer, tokens: List[int], remove_EOD: bool = True, skip_special_tokens: bool = True
    ) -> str:
        """
        Detokenize a sequence of token IDs, optionally removing trailing EOD
        tokens and handling skip_special_tokens for different tokenizer APIs.

        Args:
            tokenizer: The tokenizer to use for detokenization.
            tokens (List[int]): The token IDs to convert back to text.
            remove_EOD (bool): Whether to remove trailing EOD tokens before
                detokenization. Defaults to True.
            skip_special_tokens (bool): Whether to remove special tokens (e.g. BOS/EOS)
                during detokenization. Only passed through if the tokenizer supports it.

        Returns:
            str: The detokenized string.
        """
        if remove_EOD and getattr(tokenizer, "eod", None) is not None:
            while tokens and tokens[-1] == tokenizer.eod:
                tokens = tokens[:-1]

        if accepts_parameter(tokenizer.detokenize, "skip_special_tokens"):
            return tokenizer.detokenize(tokens, skip_special_tokens=skip_special_tokens)
        else:
            return tokenizer.detokenize(tokens)

    def detokenize_generations(
        self,
        tokens_gpu_tensor: torch.Tensor,
        lengths_gpu_tensor: torch.Tensor,
        detokenize_segments: bool,
        skip_special_tokens: bool = True,
    ) -> tuple[str, Optional[List[List[str]]]]:
        """Detokenize the generated tokens.

        Args:
            tokens_gpu_tensor (torch.Tensor): Tensor containing the tokens
            lengths_gpu_tensor (torch.Tensor): Tensor containing the lengths of each sequence
            detokenize_segments (bool): If True, returns individually detokenized tokens. If False,
            returns None as second element. Helpful for understanding per-token boundaries in
            generated text.
            skip_special_tokens (bool): If True removes special tokens like bos
            during detokenization.

        Returns:
            tuple[str, List[str] | None]: A tuple containing:
            - str: The complete detokenized text
            - List[str] | None: List of segmented tokens if detokenize_segments is True, else None
        """
        # TODO(helenn): Unify with `detokenize_generations` from legacy textgen path

        if not detokenize_segments:
            tokens = tokens_gpu_tensor.tolist()
            return (
                self.detokenize(self.tokenizer, tokens, skip_special_tokens=skip_special_tokens),
                None,
            )

        prompts_plus_generations: List[str] = []
        prompts_plus_generations_segments: List[List[str]] = []
        tokens_gpu_tensor = torch.unsqueeze(tokens_gpu_tensor, 0)
        tokens = tokens_gpu_tensor.tolist()
        lengths = lengths_gpu_tensor.tolist()

        for sequence_tokens, length in zip(tokens, lengths):
            sequence_tokens = sequence_tokens[:length]
            detok_str = self.detokenize(self.tokenizer, sequence_tokens)
            prompts_plus_generations.append(detok_str)
            offsets = self.tokenizer.offsets(sequence_tokens, detok_str)
            words = [
                detok_str[start:end] for start, end in zip(offsets, offsets[1:] + [len(detok_str)])
            ]

            prompts_plus_generations_segments.append(words)

        text = self.detokenize(self.tokenizer, tokens[0], skip_special_tokens=skip_special_tokens)

        return text, prompts_plus_generations_segments

    def _torch_sampling_func(
        self,
        last_token_logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        vocab_size: Optional[int] = None,
    ):
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it
        according to the parameters defined in sampling_params
        and returns the samples. If sampling parameters top_n_logprobs > 0
        at each step it also updates the top_n_logprobs dict.

        Args:
            last_token_logits (torch.Tensor): The last token logits. A tensor of
                size [batch_size, vocab_size].
            temperature (float): The temperature to use for sampling.
            top_k (int): The top-k value to use for sampling.
            top_p (float): The top-p value to use for sampling.
            vocab_size (int): Obtained from the tokenizer. Defaults to None.

        Returns:
            sampled_logits (torch.Tensor): 1D tensor with [batch_size] elements
        """
        assert isinstance(top_p, float)
        assert isinstance(top_k, int)
        assert not (top_k > 0 and top_p > 0.0), "Cannot have top-p and top-k both greater than zero"
        assert top_p <= 1.0, "top-p should be in (0,1]"

        def modify_logits_for_top_k_filtering(logits, top_k):
            """Set the logits for none top-k values to -inf."""
            filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits.masked_fill_(filter_, float("-Inf"))

        def modify_logits_for_top_p_filtering(logits, top_p):
            """Set the logits for none top-p values to -inf."""
            # First sort and calculate cumulative sum of probabilities.
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Filteration based on the cumulative sum.
            filter_ = cumulative_probs > top_p
            # This shift by 1 is weird and I cannot justify it. This existed
            # in the original implementation:
            #   https://github.com/ari-holtzman/degen/blob/master/gen.py
            # and I guess it is needed so keeping it for now.
            # Clone needed: filter_[:, 1:] and filter_[:, :-1] are overlapping views;
            # without clone, each write would corrupt the next read during the shift.
            filter_[:, 1:] = filter_[:, :-1].clone()
            # Make sure we at least have one token to select from.
            filter_[..., 0] = 0

            # Fill in the filtered part
            filter_ = filter_.scatter(1, sorted_indices, filter_)
            logits.masked_fill_(filter_, float("-Inf"))

        # Greedy sampling
        if top_k == 1:
            sampled_logits = torch.argmax(last_token_logits, dim=-1)
        else:
            # Clone needed: .div_() and masked_fill_() below modify in-place,
            # which would mutate the caller's tensor without this clone.
            last_token_logits = last_token_logits.clone()
            if temperature != 1.0:
                last_token_logits.div_(temperature)
            if top_k > 1:
                assert top_k <= last_token_logits.size(1), "top-k is larger than logit size."
                if vocab_size:
                    assert top_k < vocab_size, "top-k is larger than vocab size."
                modify_logits_for_top_k_filtering(last_token_logits, top_k)

            elif top_p > 0.0:
                modify_logits_for_top_p_filtering(last_token_logits, top_p)

            # After filtering, we need to recalculate the distribution.
            probabilities = last_token_logits.softmax(dim=-1)

            sampled_logits = torch.multinomial(
                probabilities, num_samples=1, generator=self.sampling_rng
            ).view(-1)

            # If vocab size is provided, make sure the samples are in in the range [0, vocab-size).
            if vocab_size:
                sampled_logits = torch.clamp(sampled_logits, min=0, max=(vocab_size - 1))

        return sampled_logits

    def sample_from_logits(
        self,
        last_token_logits: torch.Tensor,
        sampling_params: Optional[SamplingParams] = None,
        vocab_size: Optional[int] = None,
        generation_started: Optional[torch.Tensor] = None,
        top_n_logprobs_dict: Dict[int, List[Dict[str, float]]] = None,
        logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Samples the logits to generate outputs

        Given the logits of the last token, this function samples it
        according to the parameters defined in sampling_params
        and returns the samples. If sampling parameters top_n_logprobs > 0
        at each step it also updates the top_n_logprobs dict.

        Args:
            last_token_logits (torch.Tensor): The last token logits. A tensor of
                size [batch_size, vocab_size]
            sampling_params (SamplingParams): The parameters to use for inference.
            vocab_size (int): Obtained from the tokenizer. Defaults to None
            generation_started (torch.Tensor): A boolean tensor of shape [batch_size]. True
                            indicates the prompt at that index has started generating tokens.
            top_n_logprobs_dict (top_n_logprobs_dict): The dict to be updated

        Returns:
            sampled_logits (torch.Tensor): 1D tensor with [batch_size] elements
            top_n_logprobs_this_step (torch.return_types.topk): a topk tensor with values as logits
                and indices as the top k elements. None if sampling params top_n_logprobs is 0.
        """

        if kwargs.get("common_inference_params"):
            sampling_params = kwargs["common_inference_params"]

        if sampling_params.top_n_logprobs > 0:
            # NOTE : This thing can also be clubbed with where we compute log probs
            # when --return-log-probs is enabled. This is just more efficient
            assert generation_started is not None
            if logits is None:
                batch_size = last_token_logits.shape[0]
                last_token_log_probs = F.log_softmax(last_token_logits, dim=1).to(torch.float32)
                top_n_logits_this_step = torch.topk(
                    last_token_log_probs, k=sampling_params.top_n_logprobs
                )
                top_n_logprobs_this_step = top_n_logits_this_step.values.cpu()
                top_n_logprobs_indices = top_n_logits_this_step.indices.cpu()

                # If we skip prompt log_probs then we only append for generated tokens.
                # Otherwise we always append to the logprobs dict.
                if sampling_params.skip_prompt_log_probs:
                    mask = generation_started.cpu()
                else:
                    mask = torch.ones(batch_size, dtype=torch.bool)

                self._update_top_n_logprobs_dict(
                    top_n_logprobs_this_step, top_n_logprobs_indices, mask, top_n_logprobs_dict
                )
            else:
                assert not sampling_params.skip_prompt_log_probs

                # Compute the prompt logprobs
                batch_size, seq_length, _ = logits.shape
                log_probs = F.log_softmax(logits, dim=2).to(torch.float32)
                top_n_logits_this_step = torch.topk(log_probs, k=sampling_params.top_n_logprobs)

                # Move the token dimension to the front and then add each token logprobs
                # individually for every request in the batch
                top_n_logprobs_this_step = top_n_logits_this_step.values.permute(1, 0, 2).cpu()
                top_n_logprobs_indices = top_n_logits_this_step.indices.permute(1, 0, 2).cpu()

                # We append to the logprobs dict for every prompt token
                mask = torch.ones(batch_size, dtype=torch.bool)

                for i in range(seq_length):
                    self._update_top_n_logprobs_dict(
                        top_n_logprobs_this_step[i],
                        top_n_logprobs_indices[i],
                        mask,
                        top_n_logprobs_dict,
                    )

        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        temperature = sampling_params.temperature

        return self._torch_sampling_func(last_token_logits, temperature, top_k, top_p, vocab_size)

    def update_generation_status(
        self,
        updated_prompts_tokens: torch.Tensor,
        generation_started: torch.Tensor,
        current_context_end_position: int,
        is_generation_done_tensor: torch.Tensor,
        generated_sequence_lengths: torch.Tensor,
        termination_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Checks which prompts have reached an end condition

        We check which prompts have reached an end condition and set the corresponding
        flags of the is_generation_done_tensor to True. The generated sequence lengths
        increase as we keep generating, until that prompts hits an end condition. The
        generation_started tensor determines which prompts have started generating.

        Args:
            updated_prompts_tokens (torch.Tensor): The prompts tokens updated with the latest
                generated tokens. A tensor of shape [batch_size, max_seq_len]
                (i.e max_seq_len = max_prompt_len + tokens_to_generate)
            generation_started (torch.Tensor): A boolean tensor of shape [batch_size]. True
                indicates the prompt at that index has started generating tokens.
            current_context_end_position (int): An integer indicating which position to
                extract from the prompts tokens to get the latest generated tokens.
            is_generation_done_tensor (torch.Tensor): A boolean tensor of shape [batch_size].
                True indicates the prompt at that index has reached end condition.
            generated_sequence_lengths (torch.Tensor): A int tensor of shape [batch_size].
                Each value represents the generated sequence lengths for that prompt.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the boolean
                is_generation_done_tensor and the generated_sequence_lengths after updating it
        """
        if termination_id is None:
            termination_id = self.tokenizer.eod
        latest_samples = updated_prompts_tokens[:, current_context_end_position]
        # Make sure we are checking eod criterion only for prompts that have started generating
        # (i.e) We only look at the generated tokenns and not the input tokens.
        reached_eod = (latest_samples == termination_id) & generation_started
        is_generation_done_tensor = is_generation_done_tensor | reached_eod
        # We increment generated sequence lengths when that prompt has not hit the
        # EOD and generation has started
        generated_sequence_lengths += ~is_generation_done_tensor & generation_started

        return is_generation_done_tensor, generated_sequence_lengths.int()

    def pad_input_prompt_tokens(
        self,
        batch_prompt_tokens_list: List[List[int]],
        padded_batch_size: int,
        padded_sequence_length: int,
    ) -> torch.Tensor:
        """Method to pad input prompts

        Given a list of prompts, pad them all to uniform length

        Args:
            batch_prompt_tokens_list (List[List[int]]): A list containing the prompt tokens
            padded_batch_size (int): The maximum number of requests for this batch
            padded_sequence_length (int): The maximum number of input + output tokens for this batch

        Returns:
            torch.Tensor: A torch tensor of shape [padded_batch_size, padded_sequence_length]
        """
        batch_size = len(batch_prompt_tokens_list)

        # Pad existing tokens to maximum sequence length
        for prompt_tokens in batch_prompt_tokens_list:
            padding_size = padded_sequence_length - len(prompt_tokens)
            prompt_tokens.extend([self.tokenizer.eod] * padding_size)

        # Pad to maximum batch size
        padded_prompt_tokens_list = batch_prompt_tokens_list
        num_padded_requests = padded_batch_size - len(batch_prompt_tokens_list)
        padded_prompt_tokens_list += [
            [self.tokenizer.eod] * padded_sequence_length for _ in range(num_padded_requests)
        ]

        tokens = torch.tensor(padded_prompt_tokens_list, device=torch.cuda.current_device())

        return tokens

    def unpad_input_prompt_tokens(
        self, padded_batch_prompt_tokens: torch.Tensor, original_batch_size: int
    ):
        """Truncates the given input tensor back to the original prompt size before padding.

        Args:
            padded_batch_prompt_tokens (torch.Tensor): The padded tokens tensor
            original_batch_size (int): The original batch size before padding
        """
        return padded_batch_prompt_tokens[:original_batch_size]

    def _dynamic_step_context_init(
        self,
        construct_graph_dimensions: Optional[InferenceBatchDimensions] = None,
        is_dummy_forward: bool = False,
    ):
        """Initializes the inference context for dynamic batching.

        Args:
            construct_graph_dimensions (Optional[InferenceBatchDimensions]): The graph config to use
                for constructing the cuda graphs.
            is_dummy_forward (bool): Whether we are running an expert parallel dummy forward pass

        Return:
            input_ids (Tensor): The active input IDs.
            position_ids (Tensor): The active position IDs.
        """
        context = self.inference_wrapped_model.inference_context

        # Remove Float16Module wrapper if it exists
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        model_config = get_model_config(unwrapped_model)

        # Initialize attention state.
        context.initialize_attention_state(
            construct_graph_dimensions=construct_graph_dimensions,
            is_expert_parallel_dummy_cuda_graph_step=is_dummy_forward,
        )

        # If using symmetric kernels and we are using using nccl
        # for prefill turn off symmetric kernels
        symmetric_ar_type = self.model_config.symmetric_ar_type
        nccl_all_reduce_for_prefill = self.model_config.nccl_all_reduce_for_prefill
        # Turning on/off MoE padding for cuda-graphs
        moe_pad_experts_for_cuda_graph_inference = (
            self.model_config.moe_pad_experts_for_cuda_graph_inference
        )
        is_inference_optimized = self.model_config.transformer_impl == "inference_optimized"
        if is_inference_optimized:
            assert not moe_pad_experts_for_cuda_graph_inference, (
                "moe_pad_experts_for_cuda_graph_inference cannot be True when "
                "transformer_impl is 'inference_optimized'"
            )
        if moe_pad_experts_for_cuda_graph_inference:
            if context.using_cuda_graph_this_step():
                capacity_factor = model_config.num_moe_experts / model_config.moe_router_topk
                set_decode_expert_padding(unwrapped_model, True, capacity_factor=capacity_factor)
            else:
                set_decode_expert_padding(unwrapped_model, False)

        if nccl_all_reduce_for_prefill and symmetric_ar_type is not None:
            if context.is_decode_only():
                # Turn on symmetric all reduce when in decode mode
                unwrapped_model.set_symmetric_ar(symmetric_ar_type)
            else:
                # Turn off symmetric all reduces for prefill
                unwrapped_model.set_symmetric_ar(None)

        # Get flat tokens, position ids.
        # If we are running a dummy forward step we want to use the token count agreed upon
        # by all EP ranks rather than the minimum number of tokens.
        if construct_graph_dimensions is not None and not is_dummy_forward:
            return context.current_input_and_position_ids(
                num_warmup_tokens=construct_graph_dimensions.token_count
            )
        else:
            return context.current_input_and_position_ids()

    def _dynamic_step_forward_logits(self, input_ids: Tensor, position_ids: Tensor):
        """Forward step the model to get logits for dynamic batching.

        This also handles logits-broadcasting for pipeline parallelism.

        Args:
            input_ids (Tensor): The input token IDs.
            position_ids (Tensor): The position IDs.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        logits_seq_len = (
            active_request_count
            if context.config.materialize_only_last_token_logits
            else context.padded_active_token_count
        )

        with torch.inference_mode():
            logits = self.inference_wrapped_model.run_one_forward_step(
                {"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}
            )
            # logits shape: [1, seq_len, vocab_size]

        assert logits_seq_len == (
            active_request_count
            if context.config.materialize_only_last_token_logits
            else input_ids.shape[1]
        )

        # Note: When speculative decoding is active (num_speculative_tokens > 0),
        # the model skips MTP computation during the forward pass. MTP logits
        # will be computed serially after verification to ensure they are
        # conditioned on verified tokens only.

        if self.model_is_pipeline_parallel:
            if context.config.materialize_only_last_token_logits:
                if self.num_speculative_tokens > 0:
                    num_prefill_requests = context.active_num_prefill_requests
                    num_decode_requests = active_request_count - num_prefill_requests
                    logits_seq_len = (
                        num_decode_requests * (self.num_speculative_tokens + 1)
                        + num_prefill_requests
                    )
                else:
                    logits_seq_len = active_request_count
            else:
                logits_seq_len = input_ids.shape[1]
            logits_shape = [1, logits_seq_len, self.vocab_size]

            if is_pipeline_last_stage(self.pp_group):
                assert logits is not None and torch.Size(logits_shape) == logits.shape

            logits = broadcast_from_last_pipeline_stage(
                logits_shape,
                dtype=self.model_config.params_dtype,
                tensor=logits,
                pp_group=self.pp_group,
            )

        # Copy logits to contiguous buffer.
        if self._enable_cuda_graph:
            self._all_logits_cuda[:, :logits_seq_len, :].copy_(logits)
        else:
            self._all_logits_cuda = logits

    def _dynamic_step_sample_bookkeeping(self):
        """Perform bookkeeping necessary to sample logits for dynamic batching."""
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        if self._sampling_backend == "torch":
            # Bucketize the core sampling parameters.
            # Doing so via list comprehension is orders of magnitude faster than via torch.
            bucket_map = defaultdict(list)

            # Shorthands for the dictionary comprehension.
            temp = context.active_request_metadata["temperature"][:active_request_count].tolist()
            top_k = context.active_request_metadata["top_k"][:active_request_count].tolist()
            top_p = context.active_request_metadata["top_p"][:active_request_count].tolist()

            for request_index, (t, k, p) in enumerate(zip(temp, top_k, top_p)):
                sampling_params = (t, k, p)
                bucket_map[sampling_params].append(request_index)

            # Just unpack the key directly!
            self._torch_sampling_buckets = [
                (indices, *sampling_params) for sampling_params, indices in bucket_map.items()
            ]

    def _rewind_kv_cache(self, reservation_pool=None):
        """Update the KV cache bookkeeping for speculative decoding.

        After forward pass with speculative tokens, some tokens may be rejected.
        This function "rewinds" the KV cache bookkeeping to reflect only the accepted tokens.

        When speculative tokens are rejected, we need to:
        1. Update request_kv_length_offsets (total sequence length)
        2. Update request_last_kv_block_offset (position within last block)
        3. If rewinding crosses a block boundary:
           - Reduce request_kv_block_counts
           - Update request_last_kv_block_id to point to the previous block
           - Clear the entry in request_to_kv_block_ids for the released block
           - Release the block (to reservation_pool if provided, else to allocator)

        The block-release path runs unconditionally: when no requests cross a
        block boundary, ``torch.nonzero`` returns an empty tensor and all
        downstream indexing becomes a no-op. This removes the explicit
        ``if remove_allocated_blocks_mask.any():`` branch from the caller.
        ``torch.nonzero`` still synchronizes on its data-dependent output
        shape, and the dual-stream path below calls ``.tolist()`` to push
        freed block IDs back into the reservation pool, so this function is
        not sync-free — it only skips one extra host round-trip.

        Args:
            reservation_pool: Optional ``ReservationPool`` from dual-stream
                coordinator. When provided, rewind-released blocks go directly
                to the pool (immediately safe — main stream just freed them).
                When ``None``, blocks go back to the KV block allocator.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        active_request_slice = slice(context.paused_request_count, context.total_request_count)

        # Get the accepted token counts for each request
        # Note: _accepted_token_counts is indexed from 0 to active_request_count-1
        accepted_tokens_per_request = self._accepted_token_counts_per_request[:active_request_count]

        # Number of tokens to rewind (rejected speculative tokens)
        num_tokens_to_rewind = self.num_speculative_tokens - accepted_tokens_per_request

        # For prefill requests, no speculative tokens were forwarded through the model,
        # so there is nothing to rewind. Read from the forward-time mirror
        # (populated in ``build_active_slices``) so dual-stream's side
        # stream cannot race ``chain_update``'s prefill→decode transitions.
        request_in_prefill_status = context.active_request_in_prefill_status_tensor[
            :active_request_count
        ]
        num_tokens_to_rewind[request_in_prefill_status == 1] = 0

        # Save the original offset BEFORE modifying to correctly detect block boundary crossing
        original_offset = context.request_last_kv_block_offset[active_request_slice].clone()

        # Check which requests need to rewind to a previous block BEFORE modifying
        # A request crosses back to a previous block if: original_offset - num_tokens_to_rewind < 0
        remove_allocated_blocks_mask = (original_offset - num_tokens_to_rewind) < 0

        # Update the offsets
        context.request_last_kv_block_offset[active_request_slice] = (
            original_offset - num_tokens_to_rewind
        ) % context.block_size_tokens

        context.request_kv_length_offsets[active_request_slice] = (
            context.request_kv_length_offsets[active_request_slice] - num_tokens_to_rewind
        )

        # No need to update request_query_lengths (It will be set correctly in the next iteration)

        # Block release for requests that crossed back to a previous block.
        if remove_allocated_blocks_mask.any():
            requests_needing_release = torch.nonzero(remove_allocated_blocks_mask, as_tuple=True)[0]
            absolute_indices = requests_needing_release + context.paused_request_count

            blocks_to_release = context.request_last_kv_block_id[absolute_indices]

            context.request_kv_block_counts[absolute_indices] -= 1
            new_block_counts = context.request_kv_block_counts[absolute_indices]

            context.request_last_kv_block_id[absolute_indices] = context.request_to_kv_block_ids[
                absolute_indices, new_block_counts - 1
            ]

            context.request_to_kv_block_ids[absolute_indices, new_block_counts] = -1

            if reservation_pool is not None:
                reservation_pool.push_many(blocks_to_release.tolist())
            else:
                context.kv_block_allocator.release_memory_blocks(blocks_to_release)

        # Mamba speculative rewind state update
        if context.is_hybrid_model:
            active_mamba_indices = context.mamba_metadata.request_to_mamba_state_idx[
                active_request_slice
            ]
            is_decode_mask = (
                context.active_request_in_prefill_status_tensor[:active_request_count] == 0
            )
            decode_mamba_indices = active_mamba_indices[is_decode_mask]
            accepted_tokens_per_decode_request = accepted_tokens_per_request[is_decode_mask]

            if decode_mamba_indices.numel() > 0:
                context.mamba_conv_states[:, decode_mamba_indices] = (
                    context.mamba_intermediate_conv_states[
                        :, decode_mamba_indices, accepted_tokens_per_decode_request
                    ]
                )
                context.mamba_ssm_states[:, decode_mamba_indices] = (
                    context.mamba_intermediate_ssm_states[
                        :, decode_mamba_indices, accepted_tokens_per_decode_request
                    ]
                )

        # Refresh active slices of tensors that rewind just mutated.
        context.build_active_kv_slices(active_request_count)

    def _sample_from_logits_2d(self, logits_2d: Tensor) -> Tensor:
        """Sample tokens from 2D logits using existing sampling parameters.

        Args:
            logits_2d (Tensor): Logits of shape [num_requests, vocab_size].

        Returns:
            Tensor: Sampled tokens of shape [num_requests].
        """
        spec_token_list = []
        indices_list = []
        for request_indices, temp, top_k, top_p in self._torch_sampling_buckets:
            request_indices_tensor = torch.tensor(
                request_indices, device=logits_2d.device, dtype=torch.long
            )
            spec_token_list.append(
                self._torch_sampling_func(logits_2d[request_indices_tensor, :], temp, top_k, top_p)
            )
            indices_list.append(request_indices_tensor)

        spec_tokens = torch.empty(logits_2d.shape[0], device=logits_2d.device, dtype=torch.int64)
        for tokens, indices in zip(spec_token_list, indices_list):
            spec_tokens[indices] = tokens
        return spec_tokens

    def _compute_serial_mtp_and_sample(self):
        """Compute MTP logits serially after verification and sample speculative tokens.

        This ensures that MTP predictions are always conditioned on verified tokens.
        Each MTP depth receives the correctly sampled token from the previous depth
        (or the base token for depth 0) rather than stale speculative tokens from
        the previous step.

        When sequence parallelism is active, hidden states are kept in SP format
        (scattered along the first dimension) between MTP depths to avoid a
        redundant gather + scatter round-trip per depth.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count
        active_slice = slice(context.paused_request_count, context.total_request_count)

        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)

        # On non-last pipeline stages, the model won't have decoder hidden states.
        has_mtp = is_pipeline_last_stage(self.pp_group) and hasattr(
            unwrapped_model, '_decoder_hidden_states_cache'
        )

        if has_mtp:
            # Get decoder hidden states at last accepted positions.
            hidden_states = unwrapped_model._decoder_hidden_states_cache

            # When SP is active the decoder output is in scattered format
            # [S/TP, B, H], but _last_accepted_seq_indices are indices into
            # the full (gathered) sequence.
            if self.model_config.sequence_parallel:
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, group=self.inference_wrapped_model.tp_group
                )
            last_accepted_hidden = hidden_states[self._last_accepted_seq_indices, :, :]
            # Shape: [active_request_count, 1, hidden_size]
        else:
            last_accepted_hidden = None

        # Compute position IDs for the next tokens.
        # After rewind, request_kv_length_offsets has been adjusted. The actual
        # KV cache length is: adjusted_offset + processed_tokens.
        # The next position to predict starts at that cache length.
        adjusted_offsets = context.request_kv_length_offsets[active_slice]
        processed_tokens = context.request_query_lengths[active_slice]
        base_position = adjusted_offsets + processed_tokens

        # Start with the freshly sampled base token.
        next_token_ids = self._sampled_tokens_cuda[:active_request_count].clone()
        current_hidden = last_accepted_hidden if has_mtp else None

        # Compute padding needed to make batch a multiple of tp_size for SP compatibility.
        tp_size = get_pg_size(self.inference_wrapped_model.tp_group)
        sp_enabled = self.model_config.sequence_parallel and tp_size > 1
        if sp_enabled:
            pad_count = (tp_size - active_request_count % tp_size) % tp_size
            padded_count = active_request_count + pad_count
        else:
            pad_count = 0

        # Pad hidden states to align with the tensor parallel size.
        if has_mtp and sp_enabled:
            if pad_count > 0:
                current_hidden = F.pad(current_hidden, (0, 0, 0, 0, 0, pad_count))

            current_hidden = scatter_to_sequence_parallel_region(
                current_hidden, group=self.inference_wrapped_model.tp_group
            )

        num_depths = min(self.num_speculative_tokens, self.num_mtp_heads)
        for depth in range(num_depths):
            position_ids = (base_position + depth).unsqueeze(0)  # [1, active_request_count]
            token_ids = next_token_ids.unsqueeze(0)  # [1, active_request_count]

            mtp_logits_2d = None
            if has_mtp:
                # Pad token_ids and position_ids each iteration (they change per depth).
                if pad_count > 0:
                    token_ids = F.pad(token_ids, (0, pad_count))
                    position_ids = F.pad(position_ids, (0, pad_count))

                current_hidden, mtp_logits = unwrapped_model.compute_mtp_single_step(
                    hidden_states=current_hidden,
                    next_token_ids=token_ids,
                    position_ids=position_ids,
                    depth=depth,
                )

                # Strip padding from logits only.  Hidden states stay padded+SP
                # between depths to avoid redundant gather/scatter round-trips.
                if pad_count > 0:
                    mtp_logits = mtp_logits[:active_request_count]

                # mtp_logits: [active_request_count, 1, vocab_size]
                mtp_logits_2d = mtp_logits.squeeze(1)  # [active_request_count, vocab_size]

            # Broadcast MTP logits across pipeline stages.
            if self.model_is_pipeline_parallel:
                mtp_logits_2d = broadcast_from_last_pipeline_stage(
                    [active_request_count, self.vocab_size],
                    dtype=self.model_config.params_dtype,
                    tensor=mtp_logits_2d,
                    pp_group=self.pp_group,
                )

            # Sample speculative token using the same sampling parameters.
            spec_tokens = self._sample_from_logits_2d(mtp_logits_2d)
            self._sampled_mtp_tokens_cuda[depth, :active_request_count] = spec_tokens

            # Use sampled token as input for the next depth.
            next_token_ids = spec_tokens

        # Clean up cached hidden states.
        if has_mtp:
            del unwrapped_model._decoder_hidden_states_cache

    def _sample_speculative_logits(
        self, required_logits: Tensor, request_in_prefill_status_tensor: Tensor
    ) -> tuple:
        """Sample tokens from logits using sampling buckets.

        For torch sampling buckets: [request_indices, temp, top_k, top_p]

        Example with 5 requests:
            token_to_request_idx :              [ 0    0     0  |  1     1     1     |  2     2     2     |   3    |   4  ]
            required_logits :                   [ a5l  a6l  a7l |  b3l    b4l  b5l   |  c6l   c7l   c8l   |  d2l   | e4l  ]  # Shape [11, vocab_size]

            Sampling buckets: [[[0,2], temp1, top_k1, top_p1], [[1], temp3, top_k3, top_p3], [[3, 4], temp2, top_k2, top_p2]]

            Final output tokens : [a5s  a6s  a7s  c6s  c7s  c8s  b3s  b4s  b5s  d2s  e4s]  # Shape [11]
            (Rearranged from sampling bucket order back to input order using token_order)

        Returns:
            tuple: (output_tokens, repeats) where output_tokens has shape [total_required_tokens]
        """
        repeats = torch.where(
            request_in_prefill_status_tensor == 0, 1 + self.num_speculative_tokens, 1
        )
        token_to_request_index = torch.repeat_interleave(
            torch.arange(
                len(request_in_prefill_status_tensor),
                device=request_in_prefill_status_tensor.device,
            ),
            repeats,
        )

        output_tokens_jumbled_list = []
        token_order_list = []

        for request_indices, temp, top_k, top_p in self._torch_sampling_buckets:
            request_indices_tensor = torch.tensor(
                request_indices, device=token_to_request_index.device
            )
            required_indices = torch.where(
                torch.isin(token_to_request_index, request_indices_tensor)
            )[0]
            output_tokens_jumbled_list.append(
                self._torch_sampling_func(required_logits[required_indices, :], temp, top_k, top_p)
            )
            token_order_list.append(required_indices)

        output_tokens_jumbled = torch.cat(output_tokens_jumbled_list, dim=0)
        output_tokens = torch.empty(
            len(output_tokens_jumbled),
            device=output_tokens_jumbled.device,
            dtype=output_tokens_jumbled.dtype,
        )
        token_order = torch.cat(token_order_list, dim=0)
        # Rearrange output tokens from sampling_bucket request order back to input ids order
        output_tokens[token_order] = output_tokens_jumbled

        return output_tokens, repeats

    def _verify_speculative_tokens(
        self,
        output_tokens: Tensor,
        input_tokens_required: Tensor,
        request_in_prefill_status_tensor: Tensor,
        repeats: Tensor,
        num_decode_requests: int,
        num_prefill_requests: int,
        active_request_count: int,
    ) -> tuple:
        """Verify speculative tokens against input tokens and compute acceptance.

        Creates an accepted tokens mask where:
        - For prefill requests, the token is always accepted.
        - For decode requests, the first token (base token) is always accepted, then we compare
          sampled tokens with input tokens and accept consecutive matches.
        Then finds the index of the last accepted token per request.

        Example (assume 1, 2, and 0 spec tokens are accepted in the first 3 decode requests):
            input_tokens_required:              [ a5  a6s  a7s |  b3    b4s  b5s   |  c6   c7s   c8s   |     d2      |         e4         ]  # Size 11
            Output tokens                       [ a6o a7o  a8o |  b40   b5o  b6o   |  c7o  c8o   c9o   |     d3o     |         e5o        ]
            Output tokens right shift           [ d3o a6o  a7o |  a8o   b40  b5o   |  b6o  c7o   c8o   |     c9o     |         d3o        ]
            Accepted tokens  mask               [  1   1    0  |  1      1    1    |   1    0     0    |      1      |         1          ]
            Last one indices                    [      1       |         5         |        6          |      9      |         10         ]

        Returns:
            tuple: (last_one_indices, accepted_tokens_mask, input_tokens_required) where
                last_one_indices contains the index of the last accepted token per request.
        """
        if input_tokens_required.ndim == 2:
            assert (
                input_tokens_required.shape[0] == 1
            ), f"Expected input_tokens_required to have 1 row, but got {input_tokens_required.shape}"
            input_tokens_required = input_tokens_required.squeeze(0)

        # Initialize mask with False to prevent boundary bleed
        accepted_tokens_mask = torch.zeros_like(input_tokens_required, dtype=torch.bool)

        # Make all prefill tokens accepted
        token_to_prefill_idx = torch.repeat_interleave(request_in_prefill_status_tensor, repeats)
        accepted_tokens_mask[token_to_prefill_idx == 1] = True

        # Safe decode token verification without cross-batch boundary contamination
        decode_mask_2d = None
        if num_decode_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)

            decode_inputs = input_tokens_required[:decode_len].reshape(
                num_decode_requests, self.num_speculative_tokens + 1
            )
            decode_outputs = output_tokens[:decode_len].reshape(
                num_decode_requests, self.num_speculative_tokens + 1
            )

            # Shift outputs right by 1 *within* each request to align sampled tokens with input targets
            decode_outputs_shifted = decode_outputs.roll(1, dims=1)
            decode_mask_2d = decode_inputs == decode_outputs_shifted
            # The first token (base token) is always accepted
            decode_mask_2d[:, 0] = True
            # Enforce consecutive acceptance: cummin propagates False to the right
            decode_mask_2d = decode_mask_2d.cummin(dim=1).values
            accepted_tokens_mask[:decode_len] = decode_mask_2d.flatten()

        last_one_indices = torch.full(
            (active_request_count,), -1, device=input_tokens_required.device
        )

        if num_decode_requests > 0:
            # Summing the consecutive mask gives the count; subtract 1 for the local index
            local_last_indices = decode_mask_2d.sum(dim=1) - 1
            row_offsets = torch.arange(num_decode_requests, device=last_one_indices.device) * (
                self.num_speculative_tokens + 1
            )
            last_one_indices[:num_decode_requests] = row_offsets + local_last_indices

        if num_prefill_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            prefill_valid = (
                torch.nonzero(accepted_tokens_mask[decode_len:]).squeeze(-1) + decode_len
            )
            last_one_indices[num_decode_requests:] = prefill_valid

        return last_one_indices, accepted_tokens_mask, input_tokens_required

    def _dynamic_step_sample_logits_and_verify_tokens(self, input_ids: Tensor):
        """
        Sample tokens from logits for dynamic batching with speculative tokens and verify the tokens.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        # Read forward-time state from the stable mirrors populated by
        # ``build_active_slices``. The backing ``request_in_prefill_status_tensor``
        # and ``num_prefill_requests`` are mutated by ``chain_update`` at the
        # end of each iteration (chunked-prefill transitions), so any code
        # that runs after the forward — especially on the dual-stream side
        # stream — must consume the mirrors to see the forward-time values.
        request_in_prefill_status_tensor = context.active_request_in_prefill_status_tensor[
            :active_request_count
        ]
        request_query_lengths = context.active_request_query_lengths[:active_request_count]

        num_prefill_requests = context.active_num_prefill_requests
        num_decode_requests = active_request_count - num_prefill_requests

        # Get the logit indices for tokens that need sampling.
        # These indices are always needed for input_ids slicing and tracking
        # accepted sequence positions, even when logits are pre-sliced.
        logits = self._all_logits_cuda
        required_logit_indices = context.speculative_required_logit_indices(logits.device)

        if context.config.materialize_only_last_token_logits:
            # last_token_logits already selected exactly the required positions.
            required_logits = logits.squeeze(0)
        else:
            required_logits = logits.squeeze(0)[
                required_logit_indices, :
            ]  # Shape [num_required, vocab_size]

        # Sample tokens from logits
        output_tokens, repeats = self._sample_speculative_logits(
            required_logits, request_in_prefill_status_tensor
        )

        # Verify speculative tokens against input tokens.
        input_tokens_required = input_ids[0, required_logit_indices]
        last_one_indices, accepted_tokens_mask, input_tokens_required = (
            self._verify_speculative_tokens(
                output_tokens,
                input_tokens_required,
                request_in_prefill_status_tensor,
                repeats,
                num_decode_requests,
                num_prefill_requests,
                active_request_count,
            )
        )

        # Store the final sampled tokens for the next forward pass.
        final_sampled_tokens = output_tokens[last_one_indices]
        self._sampled_tokens_cuda[: len(final_sampled_tokens)] = final_sampled_tokens

        # Store the last accepted positions in the packed sequence for serial
        # MTP computation after verification.
        self._last_accepted_seq_indices = required_logit_indices[last_one_indices]

        # Extract accepted tokens and counts for decode requests.
        # For prefill it is always set to 1. For decode, the first token is always accepted,
        # then we compare with input tokens and accept the next tokens if its a match.
        #
        # Example (continuing from above):
        #   input_tokens_required:              [ a5  a6s  a7s |  b3    b4s  b5s   |  c6   c7s   c8s   |     d2      |         e4         ]
        #   Accepted tokens  mask               [  1   1    0  |  1      1    1    |   1    0     0    |      1      |         1          ]
        #   Accepted tokens                     [   [a6s  -1]  |     [b4s  b5s]    |     [-1  -1]      ]  # Only decode requests (prefill defaults to -1)
        #   Accepted token counts               [      1       |         2         |         0         ]  # Prefill defaults to 0
        input_tokens_required[accepted_tokens_mask == 0] = -1  # Mask out non-accepted tokens
        input_tokens_decode_mode = input_tokens_required[
            : num_decode_requests * (self.num_speculative_tokens + 1)
        ]
        input_tokens_reshaped = input_tokens_decode_mode.reshape(
            -1, self.num_speculative_tokens + 1
        )  # shape: [num_decode_requests, num_speculative_tokens + 1]

        # Skip the first token of every decode request (i.e a5, b3, c6)
        accepted_tokens = input_tokens_reshaped[:, 1:]
        self._accepted_tokens_per_request[: accepted_tokens.shape[0], :] = accepted_tokens
        self._accepted_token_counts_per_request.copy_(
            (self._accepted_tokens_per_request != -1).sum(dim=1)
        )

    def _dynamic_step_sample_logits(self):
        """Sample tokens from logits for dynamic batching."""
        # TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank
        # and then broadcast the sampled tokens rather than broadcasting the raw logits.

        # Last token logits.
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        if context.config.materialize_only_last_token_logits:
            # When materialize_only_last_token_logits is true, last_token_logits is
            # already called in the forward pass of GPT.
            required_token_logits = self._all_logits_cuda.squeeze(0)[:active_request_count, :]
        else:
            required_token_logits = context.last_token_logits(self._all_logits_cuda)

        if self._sampling_backend == "torch":
            # Concatenate the outputs once to prevent repeated small writes.
            token_list = []
            indices_list = []

            # e.g torch sample buckets will be
            # i.e (for all unique comibnation of t, topk, topk what are the associated
            # requests indices (based on the active slices)
            # [ [req at index 0, req at index 2], t1, topk1, topp1 ]]
            # [ [req at index 1, req at index 3, req at index 4] , t2, topk2, topp2]
            for indices, temp, top_k, top_p in self._torch_sampling_buckets:
                token_list.append(
                    self._torch_sampling_func(required_token_logits[indices, :], temp, top_k, top_p)
                )
                indices_list.append(torch.tensor(indices))

            # Single write to the output tensor.
            sampled_tokens = torch.cat(token_list, dim=0)
            sampled_indices = torch.cat(indices_list, dim=0)

            self._sampled_tokens_cuda[sampled_indices] = sampled_tokens

    def _dynamic_step_log_probs_bookkeeping(
        self, active_request_count: Optional[int] = None
    ) -> Tuple[bool, bool]:
        """Perform bookkeeping necessary to compute log probs for dynamic batching.

        Args:
            active_request_count: Optional override for the active request
                count. Single-stream callers leave this as ``None`` and the
                method reads the live context; the dual-stream bookkeeping
                pass passes the forward-time value from the snapshot so the
                decision doesn't race ``chain_update``'s pause-induced
                mutations of ``paused_request_count``.

        Returns:
            (return_log_probs, return_top_n_logprobs).
        """
        context = self.inference_wrapped_model.inference_context
        if active_request_count is None:
            active_request_count = context.total_request_count - context.paused_request_count

        return (
            (context.active_request_metadata["return_log_probs"][:active_request_count]).any(),
            (context.active_request_metadata["top_n_logprobs"][:active_request_count] > 0).any(),
        )

    def _router_record_bookkeeping(self) -> Optional[Dict[int, Tensor]]:
        """Collect and map routing indices per request for MoE router recording.

        This method retrieves recorded routing decisions and maps them to individual
        requests using the context's request_ids and query_lengths. Uses the context's
        routing_metadata when available (which handles CUDA graph static buffers automatically).
        Must be called while context attributes are still valid (before request transitions).

        Returns:
            Optional[Dict[int, Tensor]]: A dictionary mapping request_id to a tensor of
                shape [num_tokens, num_layers, topk]. Returns None if routing replay is
                disabled or no routing data was recorded.
        """
        config = self.inference_wrapped_model.model.config
        if not config.moe_enable_routing_replay:
            return None

        # Get routing indices - use routing_metadata if available (handles CUDA graph static buffers)
        context = self.inference_wrapped_model.inference_context
        if context.moe_routing_metadata is None:
            return None

        stacked_routing = context.moe_routing_metadata.get_routing_indices()

        if stacked_routing is None:
            return None

        # Get active request info from context
        active_request_count = context.total_request_count - context.paused_request_count
        active_request_ids = context.active_request_ids[:active_request_count].tolist()
        active_query_lengths = context.active_request_query_lengths[:active_request_count].tolist()
        active_token_count = context.active_token_count

        # Get TP group for all-gather if using sequence parallelism
        # With sequence parallelism, each TP rank only sees a portion of the tokens,
        # so we need to gather routing indices across all TP ranks.
        tp_group = self.inference_wrapped_model.tp_group
        tp_size = get_pg_size(tp_group)

        # All-gather across TP group if using sequence parallelism (tp_size > 1)
        if tp_size > 1 and get_model_config(self.inference_wrapped_model.model).sequence_parallel:
            # gather_from_sequence_parallel_region gathers along dim 0
            # [local_token_count, num_layers, topk] -> [global_token_count, num_layers, topk]
            stacked_routing = gather_from_sequence_parallel_region(stacked_routing, group=tp_group)

        # Slice to real tokens (remove CUDA padding)
        stacked_routing = stacked_routing[:active_token_count]

        # Split by request along token dimension
        # stacked_routing has shape [active_token_count, num_layers, topk]
        routing_splits = stacked_routing.split(active_query_lengths, dim=0)

        # Map to request IDs
        routing_indices_per_request = {}
        for req_id, routing_split in zip(active_request_ids, routing_splits):
            # routing_split has shape [num_tokens_for_request, num_layers, topk]
            routing_indices_per_request[req_id] = routing_split

        return routing_indices_per_request

    def _compute_finished_mask(
        self,
        sampled_tokens: Tensor,
        active_request_count: int,
        accepted_token_counts: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the finished-request mask from stop conditions.

        Shared by both single-stream and dual-stream paths. Checks the
        termination-ID condition, the max-sequence-length condition, and
        (when configured) the stop-word callback.

        Args:
            sampled_tokens: The sampled base tokens for active requests.
                Shape ``[active_request_count]``.
            active_request_count: Number of active requests.
            accepted_token_counts: Per-request accepted speculative token
                counts (speculative decoding only). When ``None``,
                ``seq_len_increment`` defaults to 1.

        Returns:
            A boolean mask of shape ``[active_request_count]`` where
            ``True`` marks finished requests.
        """
        context = self.inference_wrapped_model.inference_context

        termination_ids = context.active_request_metadata["termination_id"][:active_request_count]
        seq_lengths = context.active_sequence_lengths[:active_request_count]
        output_lengths = context.active_request_output_lengths[:active_request_count]

        if accepted_token_counts is not None:
            seq_len_increment = accepted_token_counts + 1
        else:
            seq_len_increment = 1

        active_mask = (
            sampled_tokens[:active_request_count] != termination_ids
        ).byte() & torch.less(seq_lengths + seq_len_increment, output_lengths).byte()

        if self._get_stop_word_finished_ids_callback is not None:
            active_request_ids = context.active_request_ids[:active_request_count]
            request_ids_list = active_request_ids.tolist()
            stop_word_finished_ids = self._get_stop_word_finished_ids_callback(request_ids_list)
            if stop_word_finished_ids:
                for idx, request_id in enumerate(request_ids_list):
                    if request_id in stop_word_finished_ids:
                        active_mask[idx] = 0

        return active_mask == 0

    def _dynamic_step_calculate_log_probs(
        self, active_request_count: Optional[int] = None, active_token_count: Optional[int] = None
    ) -> Tuple[List[List[float]], Tensor]:
        """Calculate log probs from logits.

        Args:
            active_request_count: Optional override for the active request
                count. Defaults to the live context value; the dual-stream
                bookkeeping pass passes the forward-time value so the
                computation doesn't race ``chain_update``'s pause-induced
                mutations.
            active_token_count: Optional override for
                ``context.active_token_count``. Same rationale — passed by
                the dual-stream pass.
        """
        context = self.inference_wrapped_model.inference_context
        if active_request_count is None:
            active_request_count = context.total_request_count - context.paused_request_count
        logits_seq_len = (
            active_request_count
            if context.config.materialize_only_last_token_logits
            else context.padded_active_token_count
        )

        return context.calculate_log_probs(
            self._all_logits_cuda[:, :logits_seq_len, :],
            self._sampled_tokens_cuda[:active_request_count],
            only_last_token_logits=context.config.materialize_only_last_token_logits,
            active_token_count=active_token_count,
        )

    def _dynamic_step_calculate_log_probs_speculative(
        self,
        active_request_count: Optional[int] = None,
        num_prefill_requests: Optional[int] = None,
        active_token_count: Optional[int] = None,
        accepted_token_counts: Optional[Tensor] = None,
    ) -> Tuple[List[List[float]], Tensor]:
        """Calculate log probs from logits for speculative decoding.

        For decode requests, computes log probs for each accepted speculative token
        and the newly sampled token using the main model logits. For prefill requests,
        handles prompt log probs the same way as non-speculative decoding.

        The main model logits at position j predict the token at position j+1. So:
        - log_prob(accepted_token[j]) comes from logits at position j
        - log_prob(newly_sampled_token) comes from logits at position accepted_count

        Args:
            active_request_count: Optional override for the active request count.
                Defaults to the live context value; dual-stream's side-stream
                pass passes the forward-time value from the snapshot.
            num_prefill_requests: Optional override for the forward-time prefill
                count. Defaults to ``context.active_num_prefill_requests``
                (the forward-time mirror populated in ``build_active_slices``).
                Dual-stream can still pass the snapshot's value explicitly;
                both are equivalent because the mirror is captured at forward
                init and ``chain_update`` does not touch it.
            active_token_count: Optional override for
                ``context.active_token_count``. Same rationale.

        Returns:
            Tuple of (log_probs_list, log_probs_tensor):
                log_probs_list: List of lists, one per active request, containing
                    log probs for the tokens emitted in this step.
                log_probs_tensor: Full log_softmax tensor for top-n computation.
        """
        context = self.inference_wrapped_model.inference_context
        if active_request_count is None:
            active_request_count = context.total_request_count - context.paused_request_count
        if num_prefill_requests is None:
            num_prefill_requests = context.active_num_prefill_requests
        if active_token_count is None:
            active_token_count = context.active_token_count
        if accepted_token_counts is None:
            accepted_token_counts = self._accepted_token_counts_per_request

        request_query_lengths = context.active_request_query_lengths[:active_request_count]

        num_decode_requests = active_request_count - num_prefill_requests

        only_last = context.config.materialize_only_last_token_logits
        logits = self._all_logits_cuda
        logits_squeezed = logits.squeeze(0).float()
        if only_last:
            log_probs_tensor = F.log_softmax(logits_squeezed, dim=-1)
        else:
            log_probs_tensor = F.log_softmax(logits_squeezed[:active_token_count], dim=-1)

        log_probs_list_decode = []

        if num_decode_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            decode_log_probs = log_probs_tensor[:decode_len].reshape(
                num_decode_requests, self.num_speculative_tokens + 1, -1
            )
            accepted_counts = accepted_token_counts[:num_decode_requests]

            # Build a [num_decode, num_spec+1] token ID matrix for gathering.
            # Columns 0..num_spec-1 hold accepted speculative tokens (clamped to 0
            # where rejected, since those positions will be masked out).
            # At column accepted_count[i], place the newly sampled token.
            gather_tokens = torch.zeros(
                num_decode_requests,
                self.num_speculative_tokens + 1,
                device=logits.device,
                dtype=torch.long,
            )
            gather_tokens[:, : self.num_speculative_tokens] = self._accepted_tokens_per_request[
                :num_decode_requests
            ].clamp(min=0)
            gather_tokens[
                torch.arange(num_decode_requests, device=logits.device), accepted_counts
            ] = self._sampled_tokens_cuda[:num_decode_requests]

            # Gather: [num_decode, num_spec+1]
            gathered_log_probs = decode_log_probs.gather(2, gather_tokens.unsqueeze(-1)).squeeze(-1)

            log_probs_list_decode = [
                gathered_log_probs[i, : accepted_counts[i].item() + 1].tolist()
                for i in range(num_decode_requests)
            ]

        log_probs_list_prefill = []
        if num_prefill_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            prefill_log_probs = log_probs_tensor[decode_len:]

            if only_last:
                # Only last-token logits were materialized per prefill request.
                prefill_new_tokens = self._sampled_tokens_cuda[
                    num_decode_requests:active_request_count
                ]
                selected_log_probs = prefill_log_probs[
                    torch.arange(num_prefill_requests, device=logits.device), prefill_new_tokens
                ]
                log_probs_list_prefill = [[lp.item()] for lp in selected_log_probs]
            else:
                # Read from the stable active_token_to_input_ids mirror so
                # dual-stream's side stream sees the forward-time token
                # layout (chain_update mutates token_to_input_ids at the
                # end of the iteration).
                prefill_token_ids = context.active_token_to_input_ids[
                    decode_len:active_token_count
                ].roll(-1, 0)
                # Prefill requests are guaranteed to be laid out at
                # positions [num_decode_requests, active_request_count)
                # in the active ordering (the forward pass places decode
                # requests first, then prefill), so we can slice
                # active_request_query_lengths directly rather than
                # masking on request_in_prefill_status_tensor (which
                # chain_update mutates).
                prefill_query_lengths = request_query_lengths[num_decode_requests:]
                new_token_idx = prefill_query_lengths.cumsum(0) - 1
                prefill_new_tokens = self._sampled_tokens_cuda[
                    num_decode_requests:active_request_count
                ]
                prefill_token_ids[new_token_idx] = prefill_new_tokens

                prefill_token_count = active_token_count - decode_len
                seq_idx = torch.arange(prefill_token_count, device=logits.device)
                selected_log_probs = prefill_log_probs[seq_idx, prefill_token_ids]

                prefill_log_probs_split = selected_log_probs.cpu().split(
                    prefill_query_lengths.tolist(), dim=0
                )
                log_probs_list_prefill = [lp.tolist() for lp in prefill_log_probs_split]

        log_probs_list = log_probs_list_decode + log_probs_list_prefill

        return log_probs_list, log_probs_tensor

    def _dynamic_step_calculate_top_n_logprobs_speculative(
        self,
        log_probs_tensor: Tensor,
        active_request_count: Optional[int] = None,
        num_prefill_requests: Optional[int] = None,
        accepted_token_counts: Optional[Tensor] = None,
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """Calculate top-n log probs for speculative decoding.

        For decode requests, computes top-n at each position that produced an
        emitted token (accepted speculative positions + the newly sampled position).
        For prefill requests, behaves identically to the non-speculative path.

        Args:
            log_probs_tensor (Tensor): Pre-computed log_softmax tensor from
                _dynamic_step_calculate_log_probs_speculative.
            active_request_count: Optional override; dual-stream passes the
                snapshot's forward-time value.
            num_prefill_requests: Optional override for the forward-time prefill
                count. Defaults to ``context.active_num_prefill_requests``
                (the forward-time mirror populated in ``build_active_slices``).

        Returns:
            A dictionary mapping request_idx to list of (top_n_values, top_n_indices)
            tuples, one per emitted token position.
        """
        context = self.inference_wrapped_model.inference_context
        if active_request_count is None:
            active_request_count = context.total_request_count - context.paused_request_count
        if num_prefill_requests is None:
            num_prefill_requests = context.active_num_prefill_requests
        if accepted_token_counts is None:
            accepted_token_counts = self._accepted_token_counts_per_request

        request_query_lengths = context.active_request_query_lengths[:active_request_count]

        num_decode_requests = active_request_count - num_prefill_requests

        top_n_results = {}

        if num_decode_requests > 0:
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            decode_log_probs = log_probs_tensor[:decode_len].reshape(
                num_decode_requests, self.num_speculative_tokens + 1, -1
            )
            accepted_counts = accepted_token_counts[:num_decode_requests]
            top_n_per_request = context.active_request_metadata["top_n_logprobs"][
                :num_decode_requests
            ]
            max_top_n = int(top_n_per_request.max().item())

            if max_top_n > 0:

                # Single batched topk on GPU: [num_decode, num_spec+1, max_top_n]
                topk_results = torch.topk(decode_log_probs, k=max_top_n, dim=-1)

                # Single CPU transfer instead of O(num_decode * num_spec) transfers
                topk_values_cpu = topk_results.values.cpu()
                topk_indices_cpu = topk_results.indices.cpu()

                for i in range(num_decode_requests):
                    top_n = int(top_n_per_request[i].item())
                    if top_n > 0:
                        num_valid = accepted_counts[i].item() + 1
                        top_n_results[i] = [
                            (topk_values_cpu[i, j, :top_n], topk_indices_cpu[i, j, :top_n])
                            for j in range(num_valid)
                        ]

        if num_prefill_requests > 0:
            only_last = context.config.materialize_only_last_token_logits
            decode_len = num_decode_requests * (self.num_speculative_tokens + 1)
            prefill_log_probs = log_probs_tensor[decode_len:]

            # Batch metadata reads: single CPU transfer for all prefill requests.
            prefill_top_n = context.active_request_metadata["top_n_logprobs"][
                num_decode_requests:active_request_count
            ].tolist()
            max_top_n_prefill = int(max(prefill_top_n)) if prefill_top_n else 0

            if max_top_n_prefill > 0:
                if only_last:
                    # One logit row per prefill request — single batched topk.
                    topk_results_prefill = torch.topk(
                        prefill_log_probs, k=max_top_n_prefill, dim=-1
                    )
                    topk_vals_cpu = topk_results_prefill.values.cpu()
                    topk_idxs_cpu = topk_results_prefill.indices.cpu()

                    for i in range(num_prefill_requests):
                        top_n = int(prefill_top_n[i])
                        if top_n > 0:
                            req_idx = num_decode_requests + i
                            top_n_results[req_idx] = [
                                (topk_vals_cpu[i, :top_n], topk_idxs_cpu[i, :top_n])
                            ]
                else:
                    # Prefill requests are laid out at positions
                    # [num_decode_requests, active_request_count) in the
                    # forward-time ordering, so we slice
                    # active_request_query_lengths directly instead of
                    # masking on request_in_prefill_status_tensor (which
                    # chain_update mutates).
                    prefill_query_lengths = request_query_lengths[num_decode_requests:]
                    prefill_log_probs_per_request = prefill_log_probs.split(
                        prefill_query_lengths.tolist(), dim=0
                    )
                    prefill_skip_prompt = context.active_request_metadata["skip_prompt_log_probs"][
                        num_decode_requests:active_request_count
                    ].tolist()

                    for i in range(num_prefill_requests):
                        top_n = int(prefill_top_n[i])
                        if top_n > 0:
                            req_idx = num_decode_requests + i
                            request_lp = prefill_log_probs_per_request[i]
                            skip_prompt = bool(prefill_skip_prompt[i])

                            if skip_prompt and request_lp.size(0) > 1:
                                top_n_logits = torch.topk(request_lp[-1], k=top_n)
                                top_n_results[req_idx] = [
                                    (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                                ]
                            else:
                                top_n_logits = torch.topk(request_lp, k=top_n, dim=-1)
                                top_n_values_cpu = top_n_logits.values.cpu()
                                top_n_indices_cpu = top_n_logits.indices.cpu()
                                top_n_results[req_idx] = [
                                    (top_n_values_cpu[t], top_n_indices_cpu[t])
                                    for t in range(request_lp.size(0))
                                ]

        return top_n_results if top_n_results else None

    def _dynamic_step_calculate_top_n_logprobs(
        self,
        log_probs_tensor: Optional[Tensor] = None,
        active_request_count: Optional[int] = None,
        active_token_count: Optional[int] = None,
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """Calculate top-n log probs from logits for dynamic batching.

        Args:
            log_probs_tensor (Optional[Tensor]): Pre-computed log probabilities tensor.
                If provided, avoids recomputing log_softmax. Should be the tensor
                returned by calculate_log_probs.
            active_request_count: Optional override; dual-stream passes the
                snapshot's forward-time value so the decision doesn't race
                chain_update's pause-induced mutations.
            active_token_count: Optional override for
                ``context.active_token_count``. Same rationale.

        Returns:
            A dictionary mapping request_idx to list of (top_n_logprobs, top_n_indices) tuples.
            Each tuple in the list represents one token position.
        """
        assert log_probs_tensor is not None, (
            "log_probs_tensor must be provided. This should be guaranteed by the calling code "
            "computing log_probs when return_top_n_logprobs is True."
        )

        context = self.inference_wrapped_model.inference_context
        if active_request_count is None:
            active_request_count = context.total_request_count - context.paused_request_count
        if active_token_count is None:
            active_token_count = context.active_token_count

        # Handle decode-only mode (only last token)
        if context.config.materialize_only_last_token_logits or context.is_decode_only():
            # In decode mode or when only last token logits are materialized,
            # logits already represent only the last tokens
            log_probs = log_probs_tensor[:active_request_count]

            top_n_results = {}
            for req_idx in range(active_request_count):
                top_n = int(context.active_request_metadata["top_n_logprobs"][req_idx].item())
                if top_n > 0:
                    # Get top-n logprobs and indices for this request (single token)
                    top_n_logits = torch.topk(log_probs[req_idx], k=top_n)
                    top_n_results[req_idx] = [
                        (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                    ]
            return top_n_results if top_n_results else None

        # Handle prefill mode - need to extract top-n for tokens per request
        # This follows the same pattern as calculate_log_probs in dynamic_context.py
        # Note: logits may be padded, so we only take the first active_token_count tokens
        log_probs = log_probs_tensor[:active_token_count]

        active_query_lengths = context.active_request_query_lengths[:active_request_count]

        # Split log_probs across request boundaries
        # log_probs has shape [active_token_count, vocab_size]
        log_probs_per_request = log_probs.split(active_query_lengths.tolist(), dim=0)

        top_n_results = {}
        for req_idx in range(active_request_count):
            top_n = int(context.active_request_metadata["top_n_logprobs"][req_idx].item())
            if top_n > 0:
                request_log_probs = log_probs_per_request[
                    req_idx
                ]  # [num_tokens_for_request, vocab_size]
                skip_prompt = bool(
                    context.active_request_metadata["skip_prompt_log_probs"][req_idx].item()
                )

                # If skip_prompt_log_probs is True, only compute for last token
                if skip_prompt and request_log_probs.size(0) > 1:
                    # Only compute top-n for the last token (first generated token)
                    top_n_logits = torch.topk(request_log_probs[-1], k=top_n)
                    top_n_results[req_idx] = [
                        (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                    ]
                else:
                    # Compute top-n for all tokens in the request
                    top_n_per_token = []
                    for token_idx in range(request_log_probs.size(0)):
                        top_n_logits = torch.topk(request_log_probs[token_idx], k=top_n)
                        top_n_per_token.append(
                            (top_n_logits.values.cpu(), top_n_logits.indices.cpu())
                        )
                    top_n_results[req_idx] = top_n_per_token

        return top_n_results if top_n_results else None

    def graph_capture_variants(self) -> Generator[Callable, None, None]:
        """Yield context-setup callables for each graph-capture variant.

        During graph warmup, the engine runs the full step pipeline once per yielded callable.
        Each callable exercises a different kernel path.
        """
        if not self._enable_cuda_graph:
            yield lambda context: None
            return

        yield lambda context: None

    def dummy_forward(self):
        """Perform a dummy forward pass. This is used in expert model parallelism
        on ranks that do not have any real requests. It may run in eager mode."""

        context = self.inference_wrapped_model.inference_context
        # if no cuda graphs, directly use dummy forward
        if not context.cuda_graph_batch_dimensions_list:
            self.inference_wrapped_model.dummy_forward()

            # Disable MoE padding for MTP computation
            if self.model_config.moe_pad_experts_for_cuda_graph_inference:
                unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
                set_decode_expert_padding(unwrapped_model, False)

            self._dummy_serial_mtp_forward()

            return

        # attempt to use cuda-graph if possible
        input_ids, position_ids = self._dynamic_step_context_init(is_dummy_forward=True)

        # _dynamic_step_context_init tries to find a cuda-graph that is compatible
        # with all EP ranks. It can also return no match, in which case
        # we run in eager mode.

        if context.using_cuda_graph_this_step():
            # we found a cuda-graph to run
            self._dynamic_step_forward_logits(input_ids, position_ids)
        else:
            # fallback to eager dummy forward
            self.inference_wrapped_model.dummy_forward()

        # Disable MoE padding for MTP computation
        if self.model_config.moe_pad_experts_for_cuda_graph_inference:
            unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
            set_decode_expert_padding(unwrapped_model, False)

        # When speculative decoding is active, the real EP ranks perform serial
        # MTP forward passes after the main forward pass. MTP layers may contain
        # MoE sublayers (inherited from the decoder spec), which require EP
        # all-to-all collectives. The dummy rank must participate in these
        # collectives to avoid a hang.
        self._dummy_serial_mtp_forward()

        # clear the context of any temporary state from the dummy forward
        context.reset()

    @torch.inference_mode()
    def _dummy_serial_mtp_forward(self):
        """Run dummy MTP forward passes to participate in EP collectives.

        When speculative decoding is active and MTP layers contain MoE sublayers
        (inherited from the decoder layer spec), each serial MTP step triggers
        EP all-to-all collectives. The dummy EP rank must issue matching
        collective calls so the real ranks do not hang.

        This mirrors the structure of ``_compute_serial_mtp_and_sample``:
        - On the last PP stage (where MTP resides): run ``compute_mtp_single_step``
          with dummy tensors so the MoE all-to-all is executed.
        - When PP > 1: participate in the ``broadcast_from_last_pipeline_stage``
          that the real ranks also perform.
        """
        if self.num_speculative_tokens == 0 or self.num_mtp_heads == 0:
            return
        if self.model_config.expert_model_parallel_size <= 1:
            return

        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)

        is_last_stage = is_pipeline_last_stage(self.pp_group)
        has_mtp = is_last_stage and hasattr(unwrapped_model, '_decoder_hidden_states_cache')
        if not has_mtp and not self.model_is_pipeline_parallel:
            # No MTP on this rank and no PP broadcast to participate in.
            return

        device = torch.cuda.current_device()
        dtype = self.model_config.params_dtype
        hidden_size = self.model_config.hidden_size
        num_depths = min(self.num_speculative_tokens, self.num_mtp_heads)

        # Pad token_ids/position_ids to nearest multiple of tp_size so that the
        # embedding can reduce-scatter evenly across TP ranks.
        tp_size = get_pg_size(self.inference_wrapped_model.tp_group)
        sp_enabled = self.model_config.sequence_parallel and tp_size > 1
        padded_count = tp_size if sp_enabled else 1

        dummy_hidden = None
        if has_mtp:
            # Minimal dummy tensors — just enough to drive the MTP layer forward
            # so that the MoE all-to-all collectives are issued.
            # Depth 0 uses full-format hidden; subsequent depths use SP format.
            dummy_hidden = torch.zeros((1, 1, hidden_size), device=device, dtype=dtype)
            dummy_token_ids = torch.zeros((1, padded_count), device=device, dtype=torch.long)
            dummy_position_ids = torch.zeros((1, padded_count), device=device, dtype=torch.long)

        for depth in range(num_depths):
            mtp_logits_2d = None
            if has_mtp:
                dummy_hidden, mtp_logits = unwrapped_model.compute_mtp_single_step(
                    hidden_states=dummy_hidden,
                    next_token_ids=dummy_token_ids,
                    position_ids=dummy_position_ids,
                    depth=depth,
                )
                mtp_logits_2d = mtp_logits.squeeze(1)  # [padded_count, vocab_size]

            # Match the PP broadcast that real ranks do in _compute_serial_mtp_and_sample.
            if self.model_is_pipeline_parallel:
                broadcast_from_last_pipeline_stage(
                    [padded_count, self.vocab_size],
                    dtype=dtype,
                    tensor=mtp_logits_2d,
                    pp_group=self.pp_group,
                )

    async def async_generate_output_tokens_dynamic_batch(
        self, skip_bookkeeping: Optional[bool] = False
    ) -> Optional[Dict]:
        """Single-stream iteration: forward → sample → allocate → finalize.

        Uses the shared ``_advance_iteration_state`` /
        ``_finalize_iteration_state`` helpers directly. No reservation pool,
        no quarantine, no thread pool — blocks are allocated from the KV
        block allocator inline. Requests that can't get a block under
        pressure are evicted immediately (their KV blocks are released
        back to the allocator and they're re-admitted through the engine's
        waiting queue).

        Args:
            skip_bookkeeping: If true, skip the stop-detection and
                finished-drop phase.

        Return:
            (Optional[Dict]): Step result dictionary.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        if context.active_token_count == 0 and active_request_count == 0:
            return None

        sampled_mtp_tokens: Optional[Tensor] = None
        num_gen = 1 + context.num_speculative_tokens

        with torch.inference_mode():
            input_ids, position_ids = self._dynamic_step_context_init()

            cuda_graph_request_count = (
                context.padded_active_request_count
                if context.using_cuda_graph_this_step()
                else None
            )

            config = self.inference_wrapped_model.model.config
            if config.moe_enable_routing_replay:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

            # Launch bookkeeping on a side stream so it overlaps with forward.
            self._pre_forward_bookkeeping_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._pre_forward_bookkeeping_stream):
                self._pre_forward_bookkeeping_event.record()

            self._dynamic_step_forward_logits(input_ids, position_ids)

            if context.is_hybrid_model and context.mamba_slot_allocator is not None:
                context.mamba_slot_allocator.commit_intermediate_states()

            routing_indices_per_request = self._router_record_bookkeeping()

        await asyncio.sleep(0)

        self._pre_forward_bookkeeping_event.synchronize()
        with torch.inference_mode():
            return_log_probs, return_top_n_logprobs = self._dynamic_step_log_probs_bookkeeping()
            self._dynamic_step_sample_bookkeeping()

            if self.num_speculative_tokens > 0:
                self._dynamic_step_sample_logits_and_verify_tokens(input_ids)
                self._rewind_kv_cache()

                if self.model_config.moe_pad_experts_for_cuda_graph_inference:
                    unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
                    set_decode_expert_padding(unwrapped_model, False)

                self._compute_serial_mtp_and_sample()
                sampled_mtp_tokens = self._sampled_mtp_tokens_cuda[:, :active_request_count]
            else:
                self._dynamic_step_sample_logits()

            log_probs = None
            top_n_logprobs = None
            if return_log_probs or return_top_n_logprobs:
                if self.num_speculative_tokens > 0:
                    log_probs, log_probs_tensor = (
                        self._dynamic_step_calculate_log_probs_speculative()
                    )
                    if return_top_n_logprobs:
                        top_n_logprobs = self._dynamic_step_calculate_top_n_logprobs_speculative(
                            log_probs_tensor
                        )
                else:
                    log_probs, log_probs_tensor = self._dynamic_step_calculate_log_probs()
                    if return_top_n_logprobs:
                        top_n_logprobs = self._dynamic_step_calculate_top_n_logprobs(
                            log_probs_tensor
                        )

            # Stop detection.
            spec_accepted = (
                self._accepted_token_counts_per_request[:active_request_count]
                if self.num_speculative_tokens > 0
                else None
            )
            finished_mask = self._compute_finished_mask(
                self._sampled_tokens_cuda, active_request_count, accepted_token_counts=spec_accepted
            )
            if skip_bookkeeping:
                finished_mask = torch.zeros_like(finished_mask, dtype=torch.bool)

            active_request_ids = context.active_request_ids[:active_request_count]
            finished_request_ids = active_request_ids[finished_mask]
            finished_ids_list = (
                finished_request_ids.tolist() if finished_request_ids.numel() > 0 else []
            )

            sample_clone = self._sampled_tokens_cuda[:active_request_count].clone()
            accepted_tokens_clone = (
                self._accepted_tokens_per_request.clone()
                if self.num_speculative_tokens > 0
                else None
            )

            # Drop finished requests, releasing blocks directly to allocator.
            # compact_tokens=False: _finalize_iteration_state rebuilds
            # token_to_input_ids from scratch.
            num_dropped, surviving = context.apply_finished_drops(
                finished_ids_list, num_gen, compact_tokens=False
            )
            active_request_count -= num_dropped

            # Compact _sampled_tokens_cuda to match the post-drop active
            # layout. apply_finished_drops compacts the bookkeeping tensors
            # but doesn't touch sample buffers; _finalize_iteration_state
            # reads sampled_tokens[0:active_request_count] and expects it
            # to correspond to the compacted bookkeeping positions.
            if num_dropped > 0 and surviving is not None:
                self._sampled_tokens_cuda[:active_request_count] = self._sampled_tokens_cuda[
                    surviving
                ]
                if sampled_mtp_tokens is not None:
                    self._sampled_mtp_tokens_cuda[:, :active_request_count] = (
                        self._sampled_mtp_tokens_cuda[:, surviving]
                    )
                    sampled_mtp_tokens = self._sampled_mtp_tokens_cuda[:, :active_request_count]

            if self.num_speculative_tokens > 0:
                self._accepted_tokens_per_request.fill_(-1)
                self._accepted_token_counts_per_request.fill_(0)

            # ── Advance + allocate + finalize ────────────────────────
            if active_request_count > 0 and not skip_bookkeeping:
                (
                    crossing_indices_padded,
                    chunked_prefill_idx,
                    forward_paused_count,
                    active_request_count,
                ) = context._advance_iteration_state(num_gen)

                # Inline .tolist() — no thread pool needed.
                crossing_indices_cpu = (
                    crossing_indices_padded.tolist()
                    if crossing_indices_padded is not None
                    else None
                )

                # Allocate blocks directly from the allocator. Evict
                # requests that can't get a block.
                evict_abs_list = []
                if crossing_indices_cpu is not None:
                    crossings = [
                        forward_paused_count + idx
                        for idx in crossing_indices_cpu
                        if idx >= 0 and (forward_paused_count + idx) != chunked_prefill_idx
                    ]
                    if crossings:
                        allocator = context.kv_block_allocator
                        num_allocable = min(len(crossings), allocator.get_active_avail())
                        if num_allocable > 0:
                            block_ids = allocator.allocate_memory_blocks(num_allocable)
                            for i, abs_idx in enumerate(crossings[:num_allocable]):
                                blk_count = context.request_kv_block_counts[abs_idx].item()
                                context.request_to_kv_block_ids[abs_idx, blk_count] = block_ids[i]
                                context.request_kv_block_counts[abs_idx] += 1
                                context.request_last_kv_block_id[abs_idx] = block_ids[i]
                        evict_abs_list = crossings[num_allocable:]

                # Evict block-pressure failures: release their KV blocks
                # and compact the active set.
                evict_ids_from_pressure = []
                if evict_abs_list:
                    device = context.request_ids.device
                    evict_rids = context.request_ids[
                        torch.tensor(evict_abs_list, dtype=torch.long, device=device)
                    ].tolist()
                    evict_ids_from_pressure = [int(rid) for rid in evict_rids]
                    num_evicted, evict_surviving = context.apply_finished_drops(
                        evict_ids_from_pressure, num_gen, compact_tokens=False
                    )
                    active_request_count -= num_evicted

                    # Compact sample buffers again after eviction.
                    if num_evicted > 0 and evict_surviving is not None:
                        self._sampled_tokens_cuda[:active_request_count] = (
                            self._sampled_tokens_cuda[evict_surviving]
                        )
                        if sampled_mtp_tokens is not None:
                            self._sampled_mtp_tokens_cuda[:, :active_request_count] = (
                                self._sampled_mtp_tokens_cuda[:, evict_surviving]
                            )
                            sampled_mtp_tokens = self._sampled_mtp_tokens_cuda[
                                :, :active_request_count
                            ]

                context._finalize_iteration_state(
                    self._sampled_tokens_cuda,
                    sampled_mtp_tokens,
                    active_request_count,
                    0,
                    forward_paused_count,
                    num_gen,
                )
            else:
                evict_ids_from_pressure = []

        device = sample_clone.device
        evict_request_ids_ret = (
            torch.tensor(evict_ids_from_pressure, dtype=torch.long, device=device)
            if evict_ids_from_pressure
            else None
        )

        return {
            "active_request_ids": active_request_ids,
            "finished_request_ids": finished_request_ids,
            "newly_paused_request_ids": None,
            "evict_request_ids": evict_request_ids_ret,
            "sample": sample_clone,
            "accepted_tokens": accepted_tokens_clone,
            "log_probs": log_probs,
            "top_n_logprobs": top_n_logprobs,
            "routing_indices_per_request": routing_indices_per_request,
            "cuda_graph_request_count": cuda_graph_request_count,
        }

    @torch.inference_mode()
    def generate_output_tokens_dynamic_batch(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Optional[Dict]:
        """Synchronous wrapper for `self.async_generate_output_tokens_dynamic_batch."""
        loop = get_asyncio_loop(loop)
        return loop.run_until_complete(self.async_generate_output_tokens_dynamic_batch())

    # ------------------------------------------------------------------
    # Dual-stream methods
    # ------------------------------------------------------------------

    async def _ds_enqueue_one_iter(self, coordinator) -> Optional[Dict]:
        """Main-stream per-iteration work: init → forward → sample → chain_update.

        Enqueues all GPU work for one iteration on ``coordinator.main_stream``.
        Returns a snapshot dict for the side stream to process, or None if
        there are no active requests.

        This method is ``async`` because ``chain_update`` awaits its CPU-GPU
        sync via the coordinator's helper thread pool — letting the side
        stream's bookkeeping task run during the wait. The GPU work leading
        up to ``chain_update`` is wrapped in a ``torch.cuda.stream(...)``
        block that ends *before* the await, so the main-stream context
        manager state never straddles an await boundary (which would leak
        into the side-stream task). ``chain_update`` enters its own
        main-stream context internally.

        Must be called while the critical task owns the event loop
        (i.e., between PROCEED.clear() and the next yield at the top of
        the critical loop).
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = context.total_request_count - context.paused_request_count

        if context.active_token_count == 0 and active_request_count == 0:
            return None

        # CUDA graph replay on coordinator.main_stream is safe:
        # cudaGraphLaunch targets whichever stream it's called on, and
        # all internal ops within the replay run on the launch stream.
        # However, CUDA graph *capture* during dual-stream would silently
        # produce wrong results (capture records the default stream while
        # kernels are issued on main_stream). Guard against this.
        assert not context.is_creating_cuda_graphs, (
            "CUDA graph capture must not run during dual-stream inference. "
            "Graphs should be created at engine startup before generate_dual_stream."
        )

        sampled_mtp_tokens = None
        snapshot: Optional[Dict] = None

        # ── Part 1: all forward-path GPU work on main stream ────────
        # Everything from decision application through snapshot build
        # runs inside this stream-scoped block. We exit before awaiting
        # chain_update so the stream context doesn't cross the await.
        # Part 2 (chain_update) enters its own inference_mode block.
        with torch.inference_mode(), torch.cuda.stream(coordinator.main_stream):
            # Cross-stream ordering: block main stream's next kernels
            # (notably _dynamic_step_context_init, which overwrites the
            # context.active_* mirrors) until side stream has finished
            # reading them in the previous iteration's bookkeeping pass.
            # On the first iteration, side_reads_done_event is unrecorded
            # and the wait is a no-op.
            coordinator.main_stream.wait_event(coordinator.side_reads_done_event)

            # ── Apply handoff decisions from side stream ─────────────
            # Main stream is the sole mutator of context tensors. Side
            # stream publishes SideDecision with:
            #   - finished_request_ids (EOS / max-len),
            #   - resume_request_ids (paused requests to bring back),
            #   - evict_request_ids (paused-region overflow to evict).
            # Main stream applies all three here. Resume is done first
            # so the active set includes any resumed requests before
            # finished-removal runs; eviction is done after finished so
            # the order of operations on total_request_count stays
            # consistent with single-stream's update_requests.
            decisions = coordinator.drain_decisions()
            num_gen = 1 + context.num_speculative_tokens

            # Flatten all resumes / finishes / evictions across decisions
            # so we can handle them with a single shift of
            # token_to_input_ids on the resume path.
            all_resume_ids: List[int] = []
            all_finished_ids: List[int] = []
            all_evict_ids: List[int] = []
            for dec in decisions:
                all_resume_ids.extend(dec.resume_request_ids)
                all_finished_ids.extend(dec.finished_request_ids)
                all_evict_ids.extend(dec.evict_request_ids)

            # Apply resumes (LIFO: the most recently paused requests).
            # For each resumed request we optionally pop a block from
            # the pool (only if its current KV block is actually full)
            # and decrement paused_request_count. The decremented
            # position becomes the new first active slot.
            #
            # chain_update may have pushed new pauses between the side
            # stream's decision and this apply, so we verify the request
            # ID at the tail matches the side stream's intended target
            # before committing each resume.
            resume_id_set = set(int(rid) for rid in all_resume_ids)
            num_resumed = 0
            remaining_resumes = len(all_resume_ids)
            while remaining_resumes > 0:
                if context.paused_request_count == 0:
                    break
                resumed_abs_idx = context.paused_request_count - 1
                candidate_rid = int(context.request_ids[resumed_abs_idx].item())
                if candidate_rid not in resume_id_set:
                    # Tail of paused region is a request chain_update
                    # added after the side stream's decision — skip it.
                    break
                resume_id_set.discard(candidate_rid)
                remaining_resumes -= 1
                offset = context.request_last_kv_block_offset[resumed_abs_idx].item()
                block_boundary = context.block_size_tokens - 1 - context.num_speculative_tokens
                needs_block = offset >= block_boundary
                block_id = None
                if needs_block:
                    block_id = coordinator.reservation_pool.pop()
                    if block_id is None:
                        # Pool ran dry between side's resume decision
                        # and main's apply; leave remaining resumes
                        # for the next iteration.
                        break
                # Commit the resume.
                context.paused_request_count -= 1
                if block_id is not None:
                    blk_count = context.request_kv_block_counts[resumed_abs_idx].item()
                    context.request_to_kv_block_ids[resumed_abs_idx, blk_count] = block_id
                    context.request_kv_block_counts[resumed_abs_idx] += 1
                    context.request_last_kv_block_id[resumed_abs_idx] = block_id
                active_request_count += 1
                num_resumed += 1

            if num_resumed > 0:
                # Shift the existing token_to_input_ids entries right to
                # make room for the resumed requests, and write the
                # resumed samples (from context.paused_tokens /
                # paused_speculative_tokens) at the front. This keeps
                # the flat token layout in sync with the post-resume
                # active set: flat index i → absolute position
                # paused_request_count + i.
                resume_token_count = num_resumed * num_gen
                prev_active_token_count = context.active_token_count
                if prev_active_token_count > 0:
                    existing = context.token_to_input_ids[:prev_active_token_count].clone()
                    context.token_to_input_ids[
                        resume_token_count : resume_token_count + prev_active_token_count
                    ] = existing
                # Read the resumed samples.
                new_paused = context.paused_request_count
                old_paused = new_paused + num_resumed
                if (
                    context.num_speculative_tokens > 0
                    and context.paused_speculative_tokens is not None
                ):
                    base = context.paused_tokens[new_paused:old_paused]
                    mtp = context.paused_speculative_tokens[:, new_paused:old_paused]
                    resumed_tokens = torch.vstack([base.unsqueeze(0), mtp]).T.reshape(-1)
                else:
                    resumed_tokens = context.paused_tokens[new_paused:old_paused]
                context.token_to_input_ids[:resume_token_count] = resumed_tokens
                # paused_request_count was already decremented in the
                # resume loop above, so the valid range of
                # paused_tokens is implicitly truncated.
                # Update active_token_count to reflect the additional
                # resumed tokens.
                context.active_token_count = prev_active_token_count + resume_token_count

            # Apply finished decisions (vectorized compaction).
            num_dropped, _ = context.apply_finished_drops(
                all_finished_ids,
                num_gen,
                quarantine=coordinator.quarantine,
                quarantine_gen=coordinator.main_gen,
            )
            active_request_count -= num_dropped

            # Apply eviction decisions (vectorized compaction of paused region).
            context.apply_paused_evictions(
                all_evict_ids, coordinator.quarantine, coordinator.main_gen
            )

            if active_request_count == 0:
                return None

            # ── Init ─────────────────────────────────────────────────
            input_ids, position_ids = self._dynamic_step_context_init()

            cuda_graph_request_count = (
                context.padded_active_request_count if context.is_decode_only() else None
            )

            # Enable routing recording.
            config = self.inference_wrapped_model.model.config
            if config.moe_enable_routing_replay:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

            # ── Forward ──────────────────────────────────────────────
            # Launch pre-forward bookkeeping on a side stream so it
            # overlaps with the forward pass (mirrors single-stream).
            self._pre_forward_bookkeeping_stream.wait_stream(coordinator.main_stream)
            with torch.cuda.stream(self._pre_forward_bookkeeping_stream):
                self._pre_forward_bookkeeping_event.record()

            self._dynamic_step_forward_logits(input_ids, position_ids)

            # Commit Mamba intermediate states.
            if context.is_hybrid_model and context.mamba_slot_allocator is not None:
                context.mamba_slot_allocator.commit_intermediate_states()

            # Collect routing indices.
            routing_indices_per_request = self._router_record_bookkeeping()

            # ── Sample bookkeeping + sample ──────────────────────────
            self._dynamic_step_sample_bookkeeping()

            if self.num_speculative_tokens > 0:
                # Phase 1: Verify speculative tokens using base logits.
                # (Reads logits from self._all_logits_cuda internally.)
                self._dynamic_step_sample_logits_and_verify_tokens(input_ids)

                # Phase 2: Rewind KV cache for rejected tokens.
                # Released blocks go directly to reservation pool
                # (immediately safe).
                self._rewind_kv_cache(reservation_pool=coordinator.reservation_pool)

                # Disable MoE padding for MTP computation.
                if self.model_config.moe_pad_experts_for_cuda_graph_inference:
                    unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
                    set_decode_expert_padding(unwrapped_model, False)

                # Phase 3: Compute MTP serially with correct (verified) inputs.
                self._compute_serial_mtp_and_sample()

                # NB: this is a view, not a copy. chain_update mutates it
                # in-place via pause-swaps. Do not read it after chain_update.
                sampled_mtp_tokens = self._sampled_mtp_tokens_cuda[:, :active_request_count]
            else:
                # Reads logits from self._all_logits_cuda internally.
                self._dynamic_step_sample_logits()

            # Cross-stream ordering: publish "main stream has finished
            # sampling, self._all_logits_cuda and self._sampled_tokens_cuda
            # are populated for this iteration." The side stream's
            # bookkeeping pass waits on this event before reading either.
            coordinator.main_sample_done_event.record(coordinator.main_stream)

            # ── Build minimal snapshot ───────────────────────────────
            # Under Option 2 the side stream reads log-probs / stop
            # detection inputs directly from stable context buffers
            # (context.active_* and self._all_logits_cuda), so this
            # snapshot only needs to carry:
            #   - data that would otherwise be corrupted by subsequent
            #     main-stream work (sampled tokens, accepted_tokens,
            #     routing indices);
            #   - CPU-side scalar counts that chain_update mutates
            #     (active_request_count, active_token_count) so the
            #     log-probs methods can override the live context values
            #     with the forward-time values they were produced under.
            # ``num_prefill_requests`` comes from
            # ``active_num_prefill_requests`` — the forward-time mirror
            # captured in ``build_active_slices`` — so snapshot and
            # log-probs fallback share a single canonical source.
            snapshot = {
                "main_gen": coordinator.main_gen,
                "sample": self._sampled_tokens_cuda[:active_request_count].clone(),
                "active_request_count": active_request_count,
                "active_token_count": context.active_token_count,
                "num_prefill_requests": context.active_num_prefill_requests,
                "paused_request_count": context.paused_request_count,
                "cuda_graph_request_count": cuda_graph_request_count,
                "routing_indices_per_request": routing_indices_per_request,
                "accepted_tokens": (
                    self._accepted_tokens_per_request.clone()
                    if self.num_speculative_tokens > 0
                    else None
                ),
                "accepted_token_counts": (
                    self._accepted_token_counts_per_request[:active_request_count].clone()
                    if self.num_speculative_tokens > 0
                    else None
                ),
            }

            if self.num_speculative_tokens > 0:
                self._accepted_tokens_per_request.fill_(-1)
                self._accepted_token_counts_per_request.fill_(0)

        # ── Part 2: chain_update (async, enters its own stream context) ──
        # Stream context above is exited. Synchronize the pre-forward
        # bookkeeping event before chain_update touches context state.
        self._pre_forward_bookkeeping_event.synchronize()

        # chain_update awaits the CPU-GPU sync via the coordinator's
        # helper thread pool; during that await, the bookkeeping task
        # on the side stream gets to run.
        with torch.inference_mode():
            await context.chain_update(
                self._sampled_tokens_cuda, coordinator, sampled_mtp_tokens=sampled_mtp_tokens
            )
        coordinator.main_gen += 1
        return snapshot

    async def _ds_bookkeeping_pass(self, coordinator, snapshot: Dict) -> Dict:
        """Side-stream per-iteration bookkeeping pass.

        Calls the refactored ``_dynamic_step_calculate_log_probs[_speculative]``
        and ``_dynamic_step_calculate_top_n_logprobs[_speculative]`` methods
        directly, with the side CUDA stream as current. Those methods read
        from ``self._all_logits_cuda``, ``self._sampled_tokens_cuda``, and
        ``context.active_request_metadata`` / ``context.active_request_query_lengths``
        — buffers populated once per iteration at forward init and not
        mutated by ``chain_update``, so side stream can read them post-chain
        without snapshot cloning.

        Stop detection also reads from the stable ``active_*`` mirrors
        (``active_request_metadata["termination_id"]``, ``active_sequence_lengths``,
        ``active_request_output_lengths``) and the live
        ``self._sampled_tokens_cuda`` (which is only overwritten by the next
        iteration's sample, not by chain_update).

        Returns a result dict compatible with the engine's ``async_bookkeep``.
        """
        context = self.inference_wrapped_model.inference_context
        active_request_count = snapshot["active_request_count"]
        active_token_count = snapshot["active_token_count"]
        num_prefill_requests = snapshot["num_prefill_requests"]
        sample = snapshot["sample"]
        accepted_token_counts = snapshot.get("accepted_token_counts")

        with torch.inference_mode(), torch.cuda.stream(coordinator.side_stream):
            # Cross-stream ordering: block side stream's reads until main
            # stream has finished sampling this iteration.
            # self._all_logits_cuda and self._sampled_tokens_cuda are only
            # valid past this point.
            coordinator.side_stream.wait_event(coordinator.main_sample_done_event)

            # ── Log probs (refactored methods, reading stable buffers) ──
            # Pass the snapshot's forward-time counts so the log-probs
            # computations see the state as it was at the moment the
            # forward ran, not the post-chain_update state main stream
            # has since moved to.
            log_probs = None
            top_n_logprobs = None
            log_probs_tensor = None
            (return_log_probs, return_top_n_logprobs) = self._dynamic_step_log_probs_bookkeeping(
                active_request_count=active_request_count
            )

            if return_log_probs or return_top_n_logprobs:
                if self.num_speculative_tokens > 0:
                    log_probs, log_probs_tensor = (
                        self._dynamic_step_calculate_log_probs_speculative(
                            active_request_count=active_request_count,
                            num_prefill_requests=num_prefill_requests,
                            active_token_count=active_token_count,
                            accepted_token_counts=accepted_token_counts,
                        )
                    )
                    if return_top_n_logprobs:
                        top_n_logprobs = self._dynamic_step_calculate_top_n_logprobs_speculative(
                            log_probs_tensor,
                            active_request_count=active_request_count,
                            num_prefill_requests=num_prefill_requests,
                            accepted_token_counts=accepted_token_counts,
                        )
                else:
                    log_probs, log_probs_tensor = self._dynamic_step_calculate_log_probs(
                        active_request_count=active_request_count,
                        active_token_count=active_token_count,
                    )
                    if return_top_n_logprobs:
                        top_n_logprobs = self._dynamic_step_calculate_top_n_logprobs(
                            log_probs_tensor,
                            active_request_count=active_request_count,
                            active_token_count=active_token_count,
                        )

            # ── Stop detection (shared method, includes stop-word callback) ──
            finished_mask = self._compute_finished_mask(
                sample, active_request_count, accepted_token_counts=accepted_token_counts
            )

            # active_request_ids is the stable mirror populated at
            # forward init. Keep the ``.tolist()`` sync inside the
            # side-stream context so the drain and subsequent event
            # record happen on the side stream, not on the default
            # stream the caller may be holding.
            active_request_ids = context.active_request_ids[:active_request_count]
            finished_request_ids_tensor = active_request_ids[finished_mask]
            finished_ids_list = (
                finished_request_ids_tensor.tolist()
                if finished_request_ids_tensor.numel() > 0
                else []
            )

            # All reads of main-stream-owned state are done for this
            # iteration. Publish the side-reads-done event so main
            # stream's next _dynamic_step_context_init can proceed.
            # (The ``.tolist()`` above already drained the side stream,
            # so this record is effectively instant; the event still
            # matters because it unblocks main stream's wait_event on
            # the GPU without requiring another Python-level sync.)
            coordinator.side_reads_done_event.record(coordinator.side_stream)

        # ── Drain main→side pause events ─────────────────────────────
        # chain_update emits a pause event for each request it pauses
        # under block pressure. We forward those IDs as
        # ``newly_paused_request_ids`` to async_bookkeep so the engine
        # can call ``add_event_pause`` on each.
        pause_request_ids = coordinator.drain_pauses()

        # ── Reclaim quarantined blocks + top-up from allocator ───────
        # Reclaim adds finished/evicted blocks back to the pool; the
        # allocator top-up adds any fresh blocks. Both have to happen
        # before the resume / overflow-eviction decisions below so the
        # pool depth and paused-block budget reflect the maximum space
        # we can give paused requests.
        coordinator.reclaim_blocks()
        allocator = context.kv_block_allocator
        need = coordinator.reservation_pool.capacity - len(coordinator.reservation_pool)
        target = min(need, allocator.get_active_avail())
        if target > 0:
            new_block_ids = allocator.allocate_memory_blocks(target)
            if new_block_ids is not None:
                coordinator.reservation_pool.push_many(new_block_ids.tolist())

        # ── Resume decision ──────────────────────────────────────────
        # If the pool has blocks and there are paused requests, pick the
        # most recently paused ones (LIFO) to bring back to the active
        # set. Main stream applies the decision at the top of its next
        # _ds_enqueue_one_iter by decrementing paused_request_count and
        # popping one block per resumed request. The paused region is
        # stable between this read and main's apply because chain_update
        # only touches it via pause-swap (adding to the end) and no
        # chain_update runs between side iter K's read and main iter
        # K+1's apply.
        paused_count = context.paused_request_count
        pool_depth = len(coordinator.reservation_pool)
        resume_count = min(pool_depth, paused_count)
        if resume_count > 0:
            resume_start = paused_count - resume_count
            # This ``.tolist()`` is the only resume-related CPU sync.
            resume_ids_list = context.request_ids[resume_start:paused_count].tolist()
        else:
            resume_ids_list = []

        # ── Overflow eviction ────────────────────────────────────────
        # Check whether paused requests are using more KV blocks than
        # the allocator's paused budget. If so, evict the oldest
        # (head-most) paused requests until we're back under budget.
        # FIFO eviction is chosen so that resumes — which pick from
        # the tail — don't race the eviction. The disjoint-region
        # guarantee: resumes select [paused_count - resume_count,
        # paused_count), evictions select [0, num_to_evict); these
        # don't overlap as long as num_to_evict + resume_count <=
        # paused_count, which we enforce below.
        evict_ids_list: List[int] = []
        overflow = allocator.get_paused_used() - allocator.paused_count
        if overflow > 0 and paused_count > resume_count:
            paused_block_counts = context.request_kv_block_counts[:paused_count].tolist()
            freed = 0
            num_to_evict = 0
            for i in range(paused_count):
                freed += paused_block_counts[i]
                num_to_evict += 1
                if freed >= overflow:
                    break
            max_evict = paused_count - resume_count
            num_to_evict = min(num_to_evict, max_evict)
            if num_to_evict > 0:
                evict_ids_list = context.request_ids[:num_to_evict].tolist()

        # ── Publish combined decisions for main stream ──────────────
        coordinator.publish_decisions(
            SideDecision(
                finished_request_ids=finished_ids_list,
                resume_request_ids=resume_ids_list,
                evict_request_ids=evict_ids_list,
            )
        )

        # ── Build result ─────────────────────────────────────────────
        # evict_request_ids forwarded to async_bookkeep (via
        # post_process_requests) covers the overflow-evicted paused
        # requests, so the engine can checkpoint them and re-admit
        # them via the waiting queue.
        ret = self._ds_build_result(
            snapshot,
            active_request_ids,
            log_probs,
            top_n_logprobs,
            finished_ids_list,
            evict_ids_list,
            pause_request_ids,
        )

        coordinator.side_gen += 1
        return ret

    def _ds_build_result(
        self,
        snapshot: Dict,
        active_request_ids: Tensor,
        log_probs,
        top_n_logprobs,
        finished_ids: List[int],
        evict_ids: List[int],
        pause_ids: List[int],
    ) -> Dict:
        """Build the step result dict for engine bookkeeping.

        ``active_request_ids`` comes from the bookkeeping pass (reading
        from the stable ``context.active_request_ids`` mirror).
        ``evict_ids`` are the paused-region overflow evictions picked by
        the side stream's resume/evict decision logic — they're
        forwarded as ``evict_request_ids`` so
        ``post_process_requests`` can checkpoint them and push them
        back onto the waiting queue for re-admission. ``pause_ids``
        come from ``coordinator.drain_pauses()`` and are forwarded as
        ``newly_paused_request_ids`` so ``async_bookkeep`` can call
        ``add_event_pause`` on each.
        """
        sample = snapshot["sample"]
        device = sample.device

        finished_request_ids = (
            torch.tensor(finished_ids, dtype=torch.long, device=device)
            if finished_ids
            else torch.tensor([], dtype=torch.long, device=device)
        )
        evict_request_ids = (
            torch.tensor(evict_ids, dtype=torch.long, device=device) if evict_ids else None
        )
        newly_paused_request_ids = (
            torch.tensor(pause_ids, dtype=torch.long, device=device) if pause_ids else None
        )

        return {
            "sample": sample,
            "accepted_tokens": snapshot.get("accepted_tokens"),
            "log_probs": log_probs,
            "top_n_logprobs": top_n_logprobs,
            "routing_indices_per_request": snapshot.get("routing_indices_per_request"),
            "cuda_graph_request_count": snapshot.get("cuda_graph_request_count"),
            "active_request_ids": active_request_ids,
            "finished_request_ids": finished_request_ids,
            "evict_request_ids": evict_request_ids,
            "newly_paused_request_ids": newly_paused_request_ids,
        }

    def _update_top_n_logprobs_dict(
        self,
        top_n_logprobs_this_step: torch.Tensor,
        top_n_logprobs_indices: torch.Tensor,
        mask: torch.Tensor,
        top_n_logprobs_dict: Dict[int, List[Dict[str, float]]],
    ):
        """Function to update the top_n_logprobs at each step

        This function goes through the topn logprobs generated for each, and for whichever
        batch has started generating tokens, it updates the top_n_logprobs_dict with the
        decoded token (string) as the key and the logit as the value.
        top_n_logprobs_dict has as keys the batch idx, the values is a list, where each element
        represents a dictionary of decoded token as key and logit as value generated at each step

        Args:
            top_n_logprobs_this_step (torch.Tensor): The top n logprob values
            top_n_logprobs_indices (torch.Tensor): The indices corresponding to the top n logprobs
            mask (torch.Tensor): A mask to indicate which requests should append to the dict
            top_n_logprobs_dict (top_n_logprobs_dict): The dict to be updated
        """
        for batch_idx, (logprob_values, logprob_indices) in enumerate(
            zip(top_n_logprobs_this_step, top_n_logprobs_indices)
        ):
            if mask[batch_idx]:
                logit_dict = {}
                for logprob, logprob_index in zip(logprob_values, logprob_indices):
                    key = self.tokenizer.detokenize([logprob_index.item()])
                    logit_dict[key] = logprob.item()
                top_n_logprobs_dict[batch_idx].append(logit_dict)

    @torch.inference_mode()
    def generate_all_output_tokens_static_batch(
        self,
        active_requests: OrderedDict[int, InferenceRequest],
        active_streams: Optional[OrderedDict[str, AsyncStream]] = None,
    ) -> OrderedDict[int, InferenceRequest]:
        """Utility to generate all the output tokens and probabilities for the prompts.

        This utility generates the output tokens for a static batch. It runs the forward steps till
        all prompts complete generation, updates the status of these requests to completed, adds
        the generated result and returns these requests

        Args:
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests.

        Returns:
            OrderedDict[int, InferenceRequest]: The result for each of the incoming requests
        """
        assert all(request.prompt_tokens is not None for request in active_requests.values())

        # Perform a deep copy so that the request prompt tokens do not get modified.
        batch_prompt_tokens_list: List[List[int]] = list(
            map(
                lambda request: copy.deepcopy(request.prompt_tokens),  # type: ignore[arg-type]
                active_requests.values(),
            )
        )
        prompt_lengths_in_batch = torch.tensor(
            [len(prompt_tokens) for prompt_tokens in batch_prompt_tokens_list],
            device=torch.cuda.current_device(),
        )
        max_prompt_length_in_batch = max(prompt_lengths_in_batch)
        min_prompt_length_in_batch = min(prompt_lengths_in_batch)

        # For batch inference the sampling params are the same for all request
        sampling_params: SamplingParams = list(active_requests.values())[0].sampling_params

        # Remove Float16Module wrapper if it exists
        unwrapped_model = unwrap_model(self.inference_wrapped_model.model)
        model_config = get_model_config(unwrapped_model)

        # We only need an attention mask if we are exclusively doing prefill over
        # prompts of variable length
        use_attention_mask = (
            sampling_params.num_tokens_to_generate == 0
            and min_prompt_length_in_batch != max_prompt_length_in_batch
        )

        # Check whether CUDA graphs are enabled
        enable_cuda_graph = (
            model_config.cuda_graph_impl == "local"
            and CudaGraphScope.full_iteration not in model_config.cuda_graph_scope
        )

        # Pad batch tokens if necessary
        batch_size = len(active_requests)
        max_sequence_length = max_prompt_length_in_batch + sampling_params.num_tokens_to_generate
        context = self.inference_wrapped_model.inference_context
        assert isinstance(context, StaticInferenceContext)
        inference_max_batch_size = context.max_batch_size
        inference_max_sequence_length = context.max_sequence_length
        padded_batch_size = inference_max_batch_size if enable_cuda_graph else batch_size
        if padded_batch_size > inference_max_batch_size:
            raise ValueError(
                f"Padded batch size {padded_batch_size} > max batch size {inference_max_batch_size}"
            )
        padded_batch_prompt_tokens = self.pad_input_prompt_tokens(
            batch_prompt_tokens_list,
            padded_batch_size=padded_batch_size,
            padded_sequence_length=max_sequence_length,
        )

        # Verify that output sequence length is within configured limit
        if max_sequence_length > inference_max_sequence_length:
            raise MaxSequenceLengthOverflowError(
                f"Maximum allowed sequence length was set to {inference_max_sequence_length} "
                f"tokens but requested generation of {max_sequence_length} tokens"
            )

        top_n_logprobs_dict = defaultdict(list)

        # Pre allocate log probs tensor
        output_log_probs = None
        if sampling_params.return_log_probs:
            output_log_probs = torch.empty(
                (batch_size, max_sequence_length - 1),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )

        # An array to check which of the prompts have reached end of generation condition
        is_generation_done_tensor = torch.zeros(
            batch_size, dtype=torch.bool, device=torch.cuda.current_device()
        )

        # An array to act as a counter to keep track of generated sequence lengths
        generated_sequence_lengths = torch.zeros(
            batch_size, device=torch.cuda.current_device()
        ).cuda()

        # Check whether early termination is enabled
        no_early_termination = getattr(sampling_params, "no_early_termination", False)
        termination_id = -1 if no_early_termination else self.tokenizer.eod

        streaming_enabled = active_streams is not None and len(active_streams) > 0
        if streaming_enabled:
            # Start a separate thread for streaming tokens to avoid blocking the
            # main computation
            streaming_idx: List[int] = [
                i
                for (i, request_id) in enumerate(active_requests.keys())
                if request_id in active_streams
            ]
            streaming_request_ids: List[int] = list(active_streams.keys())
            streams: List[AsyncStream] = list(active_streams.values())
            streaming_requests: List[InferenceRequest] = [
                active_requests[request_id] for request_id in streaming_request_ids
            ]
            streaming_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            stream_tokens = functools.partial(self.stream_tokens, sampling_params)

        for request in active_requests.values():
            # Initialize to a list to store a latency measurement for each generated token.
            request.tpot = []
        timing_events = []

        with torch.inference_mode():
            self.inference_wrapped_model.prep_model_for_inference()

            inference_input: Dict[str, Any] = self.prep_inference_input(
                prompts_tokens=padded_batch_prompt_tokens,
                active_requests=active_requests,
                use_attention_mask=use_attention_mask,
            )

            assert (
                not self.inference_wrapped_model.inference_context.is_decode_only()
            ), f"Generation must start in prefill mode"

            # Sequence parallelism is required for MoE layers when using expert parallelism (EP)
            # becausethe expert routing mechanism relies on sequence parallelism's communication
            # infrastructure to distribute tokens across expert ranks. However, sequence parallelism
            # is not currently supported for non-MoE layers during inference, so we selectively
            # disable it for all other layer types. This is safe because MoE layers perform an
            # all-gather operation on sequences before passing data to subsequent layers, ensuring
            # that each rank has the complete sequence data needed for the next non-MoE layer.
            tp_size = model_config.tensor_model_parallel_size
            ep_size = model_config.expert_model_parallel_size
            model_is_tp_ep = tp_size > 1 and ep_size > 1
            if model_is_tp_ep:
                set_model_to_sequence_parallel(
                    unwrapped_model, False, exclude_modules=[BaseMoELayer]
                )
            elif model_config.sequence_parallel and (ep_size == 1 or tp_size == 1):
                raise NotImplementedError(
                    f"Sequence parallellism is only supported for static batching with MoE models"
                )

            # If using symmetric kernels and we are using using nccl
            # for prefill turn off symmetric kernels
            symmetric_ar_type = self.model_config.symmetric_ar_type
            nccl_all_reduce_for_prefill = self.model_config.nccl_all_reduce_for_prefill
            if symmetric_ar_type is not None and nccl_all_reduce_for_prefill:
                unwrapped_model.set_symmetric_ar(None)

            # Turning off MoE padding for prefill
            moe_pad_experts_for_cuda_graph_inference = (
                self.model_config.moe_pad_experts_for_cuda_graph_inference
            )
            if moe_pad_experts_for_cuda_graph_inference:
                set_decode_expert_padding(unwrapped_model, False)

            context_start_position = 0

            # If we are exclusively doing prefill then we can process all prompt tokens
            # together even if the prompt lengths are different
            if sampling_params.num_tokens_to_generate == 0:
                context_end_position = max_prompt_length_in_batch
            else:
                context_end_position = min_prompt_length_in_batch

            # The initial iteration of this loop runs the prefill phase up to the shortest
            # prompt length in the batch. Then every subsequent iterations runs a decode step.
            # At least one new token will be generated in each iteration. The generated token
            # will be ignored for requests which have prompt length > the current generated
            # sequence length. Similarly, the generated token is ignored for requests which
            # have maximum total sequence length < the current generated sequence length.
            while True:
                # Add a timing event at the start of each iteration. The token generation
                # time will be the elapsed time between consective timing events.
                timing_events.append(torch.cuda.Event(enable_timing=True))
                timing_events[-1].record()

                # Pick the context window that we need to pass through the network.
                inference_input_for_context_window: Dict[str, Any] = (
                    self.inference_wrapped_model.get_batch_for_context_window(
                        inference_input, context_start_position, context_end_position
                    )
                )

                # Disable attention mask when using CUDA graphs for decode
                if (
                    enable_cuda_graph
                    and self.inference_wrapped_model.inference_context.is_decode_only()
                    and "attention_mask" in inference_input_for_context_window
                ):
                    inference_input_for_context_window["attention_mask"] = None
                elif use_attention_mask:
                    assert (
                        attention_mask := inference_input_for_context_window.get(
                            "attention_mask", None
                        )
                        is not None
                    )

                # Only materialize prompt log probs if the user requests log probs
                materialize_only_last_token_logits = (
                    self.inference_wrapped_model.inference_context.is_decode_only()
                    or not (sampling_params.return_log_probs or sampling_params.top_n_logprobs > 0)
                )
                inference_context = self.inference_wrapped_model.inference_context
                inference_context.config.materialize_only_last_token_logits = (
                    materialize_only_last_token_logits
                )

                # Returns the final logits of shape [batch_size, context_length, vocab_size]
                # Note: This is returned in all TP ranks or last PP stage in PP models
                logits = self.inference_wrapped_model.run_one_forward_step(
                    inference_input_for_context_window
                )

                # Undo padding if necessary
                batch_prompt_tokens = self.unpad_input_prompt_tokens(
                    padded_batch_prompt_tokens, batch_size
                )
                assert batch_prompt_tokens.shape[0] == batch_size, batch_prompt_tokens.shape[0]
                if is_pipeline_last_stage(self.pp_group):
                    logits = logits[:batch_size]

                if self.model_is_pipeline_parallel:
                    context_length = context_end_position - context_start_position
                    logits_seq_len = 1 if materialize_only_last_token_logits else context_length
                    logits_shape = [batch_size, logits_seq_len, self.vocab_size]
                    if is_pipeline_last_stage(self.pp_group):
                        assert logits is not None and torch.Size(logits_shape) == logits.shape
                    # TODO(ksanthanam): Evaluate whether it makes more sense to sample on 1 rank
                    # and then broadcast the sampled tokens rather than broadcasting the raw logits.
                    logits = broadcast_from_last_pipeline_stage(
                        [batch_size, logits_seq_len, self.vocab_size],
                        dtype=self.model_config.params_dtype,
                        tensor=logits,
                        pp_group=self.pp_group,
                    )

                # Turn on symmetric all reduce kernels for decode stage
                # if we turned it off for prefill
                if (
                    context_end_position == min_prompt_length_in_batch
                    and symmetric_ar_type is not None
                    and nccl_all_reduce_for_prefill
                ):
                    if symmetric_ar_type is not None and nccl_all_reduce_for_prefill:
                        unwrapped_model.set_symmetric_ar(symmetric_ar_type)

                # Indicates which of the input prompts have started generating tokens.
                # A 1D boolean tensor with [batch_size] elements (i.e) The shortest
                # prompts will start generating first and so on
                generation_started = prompt_lengths_in_batch <= context_end_position
                last_token_logits = logits[:, -1, :]

                logits_for_top_n_prompt_logprobs = (
                    logits
                    if context_start_position == 0 and not sampling_params.skip_prompt_log_probs
                    else None
                )
                sampled_logits = self.sample_from_logits(
                    last_token_logits,
                    sampling_params,
                    self.vocab_size,
                    generation_started=generation_started,
                    top_n_logprobs_dict=top_n_logprobs_dict,
                    logits=logits_for_top_n_prompt_logprobs,
                )

                if sampling_params.num_tokens_to_generate > 0:
                    # Substitute the sampled logits only for the prompts that
                    # have started generating tokens
                    batch_prompt_tokens[generation_started, context_end_position] = sampled_logits[
                        generation_started
                    ]

                # Compute log probs
                if sampling_params.return_log_probs:
                    log_probs = F.log_softmax(logits, dim=2).to(torch.float32)

                    indices = torch.unsqueeze(
                        batch_prompt_tokens[
                            :, (context_start_position + 1) : (context_end_position + 1)
                        ],
                        2,
                    )
                    # Get the log probabilities for only the prompt tokens
                    assert output_log_probs is not None
                    output_log_probs[:, context_start_position:context_end_position] = torch.gather(
                        log_probs, 2, indices
                    ).squeeze(2)

                context_start_position = context_end_position

                if sampling_params.num_tokens_to_generate > 0:
                    # Check end of generation status for each tensor
                    # and update generated sequence lengths
                    (is_generation_done_tensor, generated_sequence_lengths) = (
                        self.update_generation_status(
                            updated_prompts_tokens=batch_prompt_tokens,
                            generation_started=generation_started,
                            current_context_end_position=context_end_position,
                            is_generation_done_tensor=is_generation_done_tensor,
                            generated_sequence_lengths=generated_sequence_lengths,
                            termination_id=termination_id,
                        )
                    )

                    # Stream intermediate outputs
                    if streaming_enabled:
                        streaming_executor.submit(
                            stream_tokens,
                            streaming_request_ids,
                            streaming_requests,
                            streams,
                            generation_started[streaming_idx].cpu(),
                            is_generation_done_tensor[streaming_idx].cpu(),
                            batch_prompt_tokens[streaming_idx].cpu(),
                            prompt_lengths_in_batch[streaming_idx].cpu(),
                            generated_sequence_lengths[streaming_idx].cpu(),
                            (
                                output_log_probs[streaming_idx].cpu()
                                if output_log_probs is not None
                                else [None] * len(streaming_idx)
                            ),
                        )

                # Boolean flag indicating if all prompts are finished
                all_prompts_done = torch.all(is_generation_done_tensor)
                if all_prompts_done:
                    break

                # Change to decode mode if all prefill is complete
                if torch.all(generation_started):
                    self.inference_wrapped_model.inference_context.enable_decode_mode()
                    # Turn on padding for decode if flag set
                    if moe_pad_experts_for_cuda_graph_inference:
                        capacity_factor = (
                            model_config.num_moe_experts / model_config.moe_router_topk
                        )
                        set_decode_expert_padding(
                            unwrapped_model, True, capacity_factor=capacity_factor
                        )

                context_end_position = context_start_position + 1
                if context_end_position >= max_sequence_length:
                    break

        # Add a final timing event to compute the latency of every loop iteration
        timing_events.append(torch.cuda.Event(enable_timing=True))
        timing_events[-1].record()

        # Close all streams
        if streaming_enabled:
            streaming_executor.shutdown()
            for stream in streams:
                stream.finish()

        # Include all the generated tokens
        batch_prompt_tokens_with_generations = padded_batch_prompt_tokens[
            :batch_size, : (context_end_position + 1)
        ]
        if sampling_params.return_log_probs:
            assert output_log_probs is not None
            output_log_probs = output_log_probs[:, :context_end_position]

        generated_sequence_lengths[
            generated_sequence_lengths > sampling_params.num_tokens_to_generate
        ] = sampling_params.num_tokens_to_generate

        timing_events[-1].synchronize()
        tpot = torch.tensor(
            [
                timing_events[i].elapsed_time(timing_events[i + 1]) / 1e3
                for i in range(len(timing_events) - 1)
            ],
            dtype=torch.float32,
        )

        for idx, request in enumerate(active_requests.values()):
            input_prompt_length = int(prompt_lengths_in_batch[idx])
            # Shorter prompts might have generated more than required tokens. So we trim them down
            required_sequence_length = int(
                min(generated_sequence_lengths[idx], sampling_params.num_tokens_to_generate)
            )
            # Extract only the generated tokens
            required_result_tokens = batch_prompt_tokens_with_generations[
                idx, input_prompt_length : (input_prompt_length + required_sequence_length)
            ]
            generated_sequence_lengths = generated_sequence_lengths.to(dtype=torch.int32)
            request.generated_sequence_lengths = generated_sequence_lengths.to(dtype=torch.int32)
            request.generated_length = required_sequence_length
            request.generated_tokens = required_result_tokens

            # Record the decode latencies for only the generated tokens
            request_tpot = tpot.clone()
            # Sum up the latencies of the first prompt tokens if the
            # request prompt length > minimum prompt length
            spill_length = input_prompt_length - min_prompt_length_in_batch
            if spill_length > 0:
                spill_latency = request_tpot[:spill_length].sum()
                request_tpot = torch.cat((spill_latency.unsqueeze(0), request_tpot[spill_length:]))

            # Remove the extraneous latencies if the
            # request sequence length < maximum sequence length
            request_tpot = request_tpot[:required_sequence_length]
            request.tpot = request_tpot.tolist()

            if output_log_probs is not None:
                request.prompt_log_probs = output_log_probs[idx, : input_prompt_length - 1].tolist()
                request.generated_log_probs = output_log_probs[
                    idx,
                    input_prompt_length - 1 : (input_prompt_length + required_sequence_length - 1),
                ].tolist()
            if sampling_params.top_n_logprobs > 0:
                if not sampling_params.skip_prompt_log_probs:
                    assert (
                        len(top_n_logprobs_dict[idx])
                        >= input_prompt_length + required_sequence_length - 1
                    ), (
                        "Did not collect required number of top-N logprobs: "
                        f"{len(top_n_logprobs_dict[idx])}"
                    )
                    request.prompt_top_n_logprobs = top_n_logprobs_dict[idx][
                        : input_prompt_length - 1
                    ]
                    request.generated_top_n_logprobs = top_n_logprobs_dict[idx][
                        input_prompt_length
                        - 1 : (input_prompt_length + required_sequence_length - 1)
                    ]
                else:
                    assert len(top_n_logprobs_dict[idx]) >= required_sequence_length, (
                        "Did not collect required number of top-N logprobs: "
                        f"{len(top_n_logprobs_dict[idx])}"
                    )
                    request.generated_top_n_logprobs = top_n_logprobs_dict[idx][
                        :required_sequence_length
                    ]

            request.status = Status.COMPLETED

            text, segments = self.detokenize_generations(
                batch_prompt_tokens_with_generations[
                    idx, : (input_prompt_length + required_sequence_length)
                ],
                input_prompt_length + generated_sequence_lengths,
                sampling_params.return_segments,
            )
            request.text = text  # Inference server returns prompts & generations together
            if sampling_params.return_segments:
                request.segments = segments[0]
            request.generated_text = text[len(request.prompt) :]
        return active_requests

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        active_requests: OrderedDict[int, InferenceRequest],
        use_attention_mask: bool = False,
    ) -> Dict[str, Any]:
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[int, InferenceRequest]): The input active requests
            use_attention_mask (bool): Whether to use an attention mask. Should be set to True only
                when exclusively doing prefill (no decode) with variable prompt lengths.

        Returns:
            A dict of the inference input for the current batch.
        """
        inference_input = self.inference_wrapped_model.prep_inference_input(prompts_tokens)

        if use_attention_mask and (
            attention_mask := inference_input.get("attention_mask", None) is None
        ):
            inference_input["attention_mask"] = get_attention_mask(prompts_tokens.size(1))

        return inference_input

    def stream_tokens(
        self,
        sampling_params: SamplingParams,
        request_ids: List[int],
        requests: List[InferenceRequest],
        streams: List[AsyncStream],
        generation_started: List[bool],
        is_generation_done: List[bool],
        tokens: torch.Tensor,
        prompt_lengths: List[int],
        generated_lengths: List[int],
        output_log_probs: Union[torch.Tensor, None],
    ):
        """Asynchronously streams tokens for the given requests.

        Args:
            sampling_params (SamplingParams): The sampling parameters.
            request_ids (List[int]): The request IDs.
            request (List[InferenceRequest]): The requests.
            stream (List[AsyncStream]): The streams over which to send tokens.
            generation_started (List[bool]): Whether the decode step has started.
            is_generation_done (List[bool]): Whether generation has completed.
            tokens (torch.Tensor): The tokens for this request.
            prompt_lengths (List[int]): The number of prompt tokens for each request.
            generated_lengths (List[int]): The number of output tokens for each request.
            output_log_probs (torch.Tensor, optional): The log probs for each request.
        """

        def stream_token(
            request_id: int,
            request: InferenceRequest,
            stream: AsyncStream,
            generation_started: bool,
            is_generation_done: bool,
            tokens: torch.Tensor,
            prompt_length: int,
            generated_length: int,
            output_log_probs: Union[torch.Tensor, None],
        ):
            """Asynchronously streams a token for the given request."""

            if (
                not generation_started
                or stream.finished
                or sampling_params.num_tokens_to_generate == 0
            ):
                return

            return_segments = sampling_params.return_segments
            detokenize_streaming_text = not getattr(
                sampling_params, "no_detokenize_streaming_text", False
            )

            generated_tokens = tokens[prompt_length : prompt_length + generated_length]

            if detokenize_streaming_text:
                generated_text, generated_segments = self.detokenize_generations(
                    generated_tokens, prompt_length + generated_length, return_segments
                )
            else:
                generated_text = ""
                generated_segments = []

            if output_log_probs is not None:
                generated_log_probs = output_log_probs[
                    prompt_length - 1 : prompt_length + generated_length - 1
                ].tolist()
            else:
                generated_log_probs = None

            stream.put(
                InferenceRequest(
                    request_id=request_id,
                    prompt=request.prompt,
                    sampling_params=request.sampling_params,
                    prompt_tokens=request.prompt_tokens,
                    arrival_time=request.arrival_time,
                    status=request.status,
                    encoder_prompt=request.encoder_prompt,
                    generated_text=generated_text,
                    generated_segments=generated_segments,
                    generated_tokens=generated_tokens,
                    generated_log_probs=generated_log_probs,
                    generated_length=generated_length,
                )
            )

            if is_generation_done or generated_length == sampling_params.num_tokens_to_generate:
                stream.finish()

        ret = map(
            stream_token,
            request_ids,
            requests,
            streams,
            generation_started,
            is_generation_done,
            tokens,
            prompt_lengths,
            generated_lengths,
            output_log_probs,
        )
        list(ret)
