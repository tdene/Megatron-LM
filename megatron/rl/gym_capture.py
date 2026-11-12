# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Honor NeMo Gym worker-custody admissions with a Megatron RL-owned prefix store.

NeMo RL answers a Gym ``token_in`` admission by fetching the parent call's exact tokens from
TransferQueue in its prompt preparer and staging the finished call through Gym's capture
helper. Megatron RL has no data plane, so ``megatron.rl.prefix_store`` provides the store: a
ZMQ key-value service rank 0 runs for the job in a process of its own, fed by every engine's
stager and read by every engine's preparer through the two consumer hooks the engine already
exposes. This module is the Gym-facing side: the prompt preparer and the stager's capture step.

Admission modes other than ``token_in`` and ``text`` (Gym's ``candidate`` source-history mode,
generation-cut resumes) are not prepared here and fail closed in the stager.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from megatron.core.inference.inference_request import (
    PREFIX_EOS_TOKEN_ID_FIELD,
    PREFIX_TEMPLATE_TOKEN_IDS_FIELD,
    FinishedRequestRecord,
    OffloadedRequestPayload,
    RequestPromptPreparationResult,
)

from .prefix_store import (
    STORE_RETRY_BUDGET_S,
    STORE_TIMEOUT_MS,
    HeldCall,
    PrefixStoreClient,
    PrefixStoreError,
    PrefixStoreServer,
)

__all__ = [
    "NG_CAPTURE_FIELD",
    "NG_COMMIT_COORDS_FIELD",
    "STORE_RETRY_BUDGET_S",
    "STORE_TIMEOUT_MS",
    "GymPromptPreparer",
    "PrefixStoreClient",
    "PrefixStoreError",
    "PrefixStoreServer",
    "replace_prefix_tokens",
    "stage_admitted_call",
]

logger = logging.getLogger(__name__)

# Gym's admission rides in offload_params under this key (nemo_gym's NG_CAPTURE_FIELD).
NG_CAPTURE_FIELD = "ng_capture"
NG_COMMIT_COORDS_FIELD = "ng_commit_coords"


def replace_prefix_tokens(
    prefix_token_ids, template_prefix_token_ids, template_token_ids, eos_token_ids
):
    """Splice the exact earlier turns in place of their re-rendered form (EOS-count boundary)."""
    eos_token_ids = set(eos_token_ids)
    eos_count = sum(token_id in eos_token_ids for token_id in template_prefix_token_ids)
    if eos_count <= 0:
        raise ValueError("the chat template's turn terminator is not among the EOS token ids")
    seen = 0
    for position, token_id in enumerate(template_token_ids):
        if token_id in eos_token_ids:
            seen += 1
            if seen == eos_count:
                suffix = template_token_ids[position:]
                break
    else:
        raise ValueError(f"expected {eos_count} EOS token(s) before the new turn, found {seen}")
    if prefix_token_ids and prefix_token_ids[-1] in eos_token_ids:
        suffix = suffix[1:]  # the exact turn already ends with its own EOS
    return list(prefix_token_ids) + list(suffix)


class GymPromptPreparer:
    """RequestPromptPreparer that resolves a Gym ``token_in`` admission from the prefix store.

    Anything it cannot honor raises: the engine fails the request at admission, which is
    cheaper and clearer than generating a turn whose capture the stager must then poison.
    """

    def __init__(self, store: PrefixStoreClient, cache_size: int = 256) -> None:
        self.store = store
        # Recently fetched parents, so a conversation's turns cost one round trip, not one per turn.
        self._cache: OrderedDict[str, HeldCall] = OrderedDict()
        self._cache_size = cache_size

    def _parent(self, uid: str) -> HeldCall:
        held = self._cache.get(uid)
        if held is None:
            held = self.store.get(uid)
            if held is None:
                raise LookupError(
                    f"Gym admission names parent call {uid!r}, which the prefix store does not hold"
                )
            self._cache[uid] = held
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return held

    def prepare_prompt(self, prompt, *, offload_params=None) -> RequestPromptPreparationResult:
        admission = (offload_params or {}).get(NG_CAPTURE_FIELD)
        if not isinstance(admission, dict) or admission.get("mode") != "token_in":
            return RequestPromptPreparationResult(prompt=prompt, offload_params=offload_params)
        prefix = list(admission.get("required_prefix_token_ids") or [])
        if not prefix:
            # The parent's prompt already began with its own exact prefix: its whole call is ours.
            chain = admission.get("staging_chain") or []
            if not chain:
                raise ValueError(
                    "Gym token_in admission carries neither a prefix nor a staging chain"
                )
            held = self._parent(chain[-1])
            prefix = held[0] + held[1]
        if len(prefix) != admission.get("prev_len"):
            raise ValueError(
                f"Gym admission expects a prefix of {admission.get('prev_len')} tokens, "
                f"resolved {len(prefix)}"
            )
        template_prefix = offload_params.get(PREFIX_TEMPLATE_TOKEN_IDS_FIELD)
        eos_token_ids = offload_params.get(PREFIX_EOS_TOKEN_ID_FIELD)
        if template_prefix is None or eos_token_ids is None:
            raise ValueError(
                "Gym token_in admission arrived without the endpoint's prefix splice metadata"
            )
        tokens = list(prompt) if isinstance(prompt, (list, tuple)) else prompt.tolist()
        spliced = replace_prefix_tokens(prefix, template_prefix, tokens, eos_token_ids)
        # Inline the resolved prefix so the stager can prove the prompt began with it.
        offload_params = {
            **offload_params,
            NG_CAPTURE_FIELD: {**admission, "required_prefix_token_ids": prefix},
        }
        return RequestPromptPreparationResult(prompt=spliced, offload_params=offload_params)


def _weight_version(record: FinishedRequestRecord) -> int:
    """The policy epoch the call was admitted under; spans are ``(token position, epoch)``."""
    return record.policy_epoch[0][1] if record.policy_epoch else 0


def _failed_coordinates(admission: dict[str, Any], weight_version: int) -> dict[str, Any]:
    """Gym's ``capture_failed`` acknowledgement, built without Gym types so it cannot raise."""
    prev_len = admission.get("prev_len") or 0
    return {
        "capture_key": admission.get("capture_key"),
        "model_call_id": admission.get("model_call_id"),
        "parent_call_id": admission.get("parent_call_id"),
        "prev_len": prev_len,
        "delta_len": 0,
        "cum_len": prev_len,
        "weight_version": weight_version,
        "disposition": "capture_failed",
    }


def stage_admitted_call(
    uid: str,
    payload: OffloadedRequestPayload,
    finished_metadata: FinishedRequestRecord,
    admission: dict[str, Any],
    prefix_store: PrefixStoreClient | None = None,
) -> dict[str, Any]:
    """Run Gym's worker capture over the finished call; return the coordinates it owes Gym.

    The call is published to the prefix store first, so a continuation can only name a held
    parent. The client retries that ``put`` within its budget; a store that still does not
    answer raises ``PrefixStoreError`` out of here, which the stager records and the trainer
    turns into a hard stop at the next ledger drain (a lost capture is a placeholder episode
    in the batch, never acknowledged quietly). Gym's own capture step never fails the
    completion: a schema the installed ``nemo_gym`` no longer accepts, or a staging-chain
    admission the preparer did not resolve, becomes ``capture_failed`` coordinates.
    """
    weight_version = _weight_version(finished_metadata)
    if prefix_store is not None:
        prefix_store.put(uid, payload.prompt_token_ids, payload.generated_token_ids)
    try:
        from nemo_gym.token_id_capture.adapters.megatron import MegatronCaptureAdapter
        from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission, StageResult

        class _LedgerSink:
            """The ledger already holds the full record; its uid is the staging key."""

            def stage(self, record, *, attachments=None):
                return StageResult(ok=True, staging_key=uid)

        parsed = CaptureAdmission.model_validate(admission)
        capture = RolloutTokenCapture(
            sink=_LedgerSink(),
            weight_version_fn=lambda: weight_version,
            adapter=MegatronCaptureAdapter(),
        )
        # A staging-chain admission the preparer did not resolve raises here and fails closed.
        call = capture.begin_call(
            parsed,
            prefix_token_ids=list(parsed.required_prefix_token_ids) or None,
            weight_version=weight_version,
        )
        coords = capture.complete_call_from_response(call, payload).model_dump(mode="json")
    except Exception:
        logger.exception("Gym capture failed for request %s; acknowledging capture_failed", uid)
        coords = _failed_coordinates(admission, weight_version)
    return {NG_COMMIT_COORDS_FIELD: coords}
