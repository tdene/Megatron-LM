# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import time
import traceback
import uuid
import warnings

from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.text.parsers import PARSER_MAPPING

logger = logging.getLogger(__name__)


def _serialize_result(record):
    """Serialize an inference result into a plain dict suitable for cross-process sharing."""
    result = record if isinstance(record, dict) else record.serialize()
    # Unwrap ("tensor", [...]) tuples from serialize() into plain lists.
    return {
        k: v[1] if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] == "tensor" else v
        for k, v in result.items()
    }


def _on_future_done(future, response_id, shared_requests):
    """Callback fired when the inference future completes. Writes the result to shared state."""
    try:
        record = future.result()
        result = _serialize_result(record)
        # Manager dict requires full reassignment for nested updates.
        entry = dict(shared_requests[response_id])
        entry['status'] = 'completed'
        entry['result'] = result
        shared_requests[response_id] = entry
    except Exception as e:
        logger.error(f"Response {response_id} failed: {e}")
        try:
            entry = dict(shared_requests[response_id])
            entry['status'] = 'failed'
            entry['error'] = str(e)
            shared_requests[response_id] = entry
        except Exception:
            logger.error(f"Failed to update shared state for {response_id}: {traceback.format_exc()}")


try:
    from quart import Blueprint, current_app, jsonify, request

    bp = Blueprint('responses_api', __name__)

    @bp.route('/responses', methods=['POST'])
    @bp.route('/v1/responses', methods=['POST'])
    async def create_response():
        """Submit an inference request. Returns immediately with a response ID."""
        client = current_app.config['client']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']
        verbose = current_app.config['verbose']
        shared_requests = current_app.config['shared_requests']

        req = await request.get_json()

        # --- 1. Parse Messages (from 'input' field, Responses API convention) ---
        messages = req.get("input")
        if not messages:
            return jsonify({"error": "Missing 'input' field"}), 400
        if not isinstance(messages, list):
            return jsonify({"error": "'input' must be a list"}), 400

        try:
            prompt_tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, tools=req.get("tools", None)
            )
        except (AttributeError, AssertionError):
            warnings.warn(
                "Tokenizer does not support 'apply_chat_template'. Using tokenize instead."
            )
            prompt_tokens = tokenizer.tokenize(
                "\n".join([message["content"] for message in messages])
            )
        except Exception as e:
            logger.error(f"{traceback.format_exc()}")
            return jsonify({"error": f"Error processing 'input': {e}"}), 500

        # --- 2. Parse Sampling Params ---
        try:
            temperature = float(req.get("temperature", 1.0))
            top_p = float(req.get("top_p", 1.0))
            top_k = int(req.get("top_k", 0))

            if temperature == 0.0:
                top_k = 1
                top_p = 0.0

            return_log_probs = bool(req.get("logprobs", False))
            top_n_logprobs = int(req.get("top_logprobs", 0)) if return_log_probs else 0
            skip_prompt_log_probs = bool(req.get("skip_prompt_log_probs", True))
            add_BOS = bool(req.get("add_BOS", False))

            if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
                while prompt_tokens and prompt_tokens[0] == tokenizer.bos:
                    prompt_tokens.pop(0)
                if add_BOS:
                    prompt_tokens = [tokenizer.bos] + prompt_tokens

            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                top_n_logprobs=top_n_logprobs,
                num_tokens_to_generate=(
                    int(max_tokens)
                    if ((max_tokens := req.get("max_tokens", None)) is not None)
                    else None
                ),
                skip_prompt_log_probs=skip_prompt_log_probs,
                add_BOS=add_BOS,
            )
        except ValueError as e:
            return jsonify({"error": f"Invalid sampling parameter: {e}"}), 400

        # --- 3. Submit Request to Engine (non-blocking) ---
        future = client.add_request(prompt_tokens, sampling_params)

        response_id = str(uuid.uuid4())
        created_at = int(time.time())

        # Store request metadata in shared cross-process dict.
        # The result will be written by _on_future_done when the future completes.
        shared_requests[response_id] = {
            "status": "queued",
            "sampling_params": sampling_params,
            "request_body": req,
            "created_at": created_at,
        }

        # When the future resolves, write the result to shared state so any
        # process can serve the GET poll.
        future.add_done_callback(
            lambda f: _on_future_done(f, response_id, shared_requests)
        )

        return jsonify({
            "id": response_id,
            "object": "response",
            "status": "queued",
            "created_at": created_at,
            "model": "EMPTY",
            "output": [],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }), 202

    @bp.route('/responses/<response_id>', methods=['GET'])
    @bp.route('/v1/responses/<response_id>', methods=['GET'])
    async def get_response(response_id):
        """Poll for the result of a previously submitted inference request."""
        shared_requests = current_app.config['shared_requests']
        tokenizer = current_app.config['tokenizer']
        parsers = current_app.config['parsers']
        verbose = current_app.config['verbose']

        entry = shared_requests.get(response_id)

        if entry is None:
            return jsonify({"error": "Response not found", "id": response_id}), 404

        # Convert Manager proxy to a regular dict for local access.
        entry = dict(entry)
        status = entry["status"]

        # --- Failed ---
        if status == "failed":
            return jsonify({
                "id": response_id,
                "object": "response",
                "status": "failed",
                "error": entry.get("error", "unknown error"),
            }), 200

        # --- Still in progress ---
        if status == "queued":
            return jsonify({
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "created_at": entry["created_at"],
                "model": "EMPTY",
                "output": [],
                "tools": [],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }), 200

        # --- Completed: format response ---
        sampling_params = entry["sampling_params"]
        req = entry["request_body"]
        result = entry["result"]

        try:
            prompt_tokens_out = result["prompt_tokens"]
            text_output = result["generated_text"]
            prompt_tokens_count = len(prompt_tokens_out) if prompt_tokens_out is not None else 0

            logprobs_content = None
            if sampling_params.return_log_probs:
                token_logprobs = result.get('log_probs', [])
                tokens = [tokenizer.detokenize([tok]) for tok in result["generated_tokens"]]

                generated_top_n_logprobs = result.get('generated_top_n_logprobs')

                logprobs_content = []
                for i, (tok, lp) in enumerate(zip(tokens, token_logprobs)):
                    top_logprobs_list = []
                    if generated_top_n_logprobs and i < len(generated_top_n_logprobs):
                        top_n_dict = generated_top_n_logprobs[i]
                        for token_str, logprob in top_n_dict.items():
                            top_logprobs_list.append(
                                {
                                    "token": token_str,
                                    "logprob": logprob,
                                    "bytes": list(token_str.encode("utf-8")),
                                }
                            )

                    entry_data = {
                        "token": tok,
                        "logprob": lp,
                        "bytes": list(tok.encode("utf-8")),
                        "top_logprobs": top_logprobs_list,
                    }
                    logprobs_content.append(entry_data)

            metadata = {}
            message_text = text_output
            if parsers:
                for parser in parsers:
                    if parser not in PARSER_MAPPING:
                        raise ValueError(f"Parser {parser} not found in PARSER_MAPPING")
                    message_text, new_info = PARSER_MAPPING[parser].parse(
                        message_text, tools=req.get("tools", None)
                    )
                    assert not (
                        metadata.keys() & new_info.keys()
                    ), "Multiple parsers found the same information."
                    metadata.update(new_info)
            message = {"role": "assistant", "content": message_text}
            if "tool_calls" in metadata:
                message["tool_calls"] = metadata["tool_calls"]
            if "reasoning" in metadata:
                message["reasoning"] = metadata["reasoning"]

            message["prompt_token_ids"] = result["prompt_tokens"]
            message["generation_token_ids"] = result["generated_tokens"]
            message["generation_log_probs"] = result.get("generated_log_probs", None)

            response_data = {
                "id": response_id,
                "object": "response",
                "status": "completed",
                "created_at": entry["created_at"],
                "model": "EMPTY",
                "output": [],
                "tools": [],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "message": message,
                "prompt_token_ids": result["prompt_tokens"],
                "generation_token_ids": result["generated_tokens"],
                "generation_log_probs": result["generated_log_probs"],
                "raw_text": result["prompt"] + result["generated_text"],
                "logprobs": {"content": logprobs_content} if sampling_params.return_log_probs else None,
                "finish_reason": (
                    "tool_calls" if metadata.get("tool_calls", []) else "stop"
                ),
                "usage": {
                    "prompt_tokens": prompt_tokens_count,
                    "completion_tokens": len(result["generated_tokens"]),
                    "total_tokens": prompt_tokens_count + len(result["generated_tokens"]),
                },
            }
            if result.get("policy_staleness") is not None:
                response_data["policy_staleness"] = result["policy_staleness"]
            if result.get("kv_cache_staleness") is not None:
                response_data["kv_cache_staleness"] = result["kv_cache_staleness"]
            events = result.get("events")
            if events is not None:
                num_evictions = sum(1 for e in events if e.get("type") == "EVICT")
                if num_evictions > 0:
                    response_data["num_evictions"] = num_evictions
            if verbose:
                logging.info(result)
            if result["routing_indices"] is not None:
                response_data["moe_topk_indices"] = result["routing_indices"]
                if prompt_tokens_count:
                    response_data["prompt_moe_topk_indices"] = result["routing_indices"][
                        :prompt_tokens_count
                    ]
        except Exception as e:
            logger.error(f"Response {response_id} failed during formatting: {traceback.format_exc()}")
            return jsonify({
                "id": response_id,
                "object": "response",
                "status": "failed",
                "error": str(e),
            }), 200

        # Clean up shared state on successful retrieval
        try:
            del shared_requests[response_id]
        except KeyError:
            pass

        return jsonify(response_data)

except ImportError as e:
    logger.warning(f"Could not import quart: {e}")
