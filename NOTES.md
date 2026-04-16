# Building feature test branches

This document is a handoff doc for anyone who needs to build or rebuild the `tde/feature_test/*` family of branches. It covers the branch inventory, the step-by-step build recipe, the merge conflicts you'll hit and how to resolve them, and the post-merge verification checklist.

## What are these branches for?

Each feature test branch is a merge of one or more in-flight feature branches onto `tde/feature_branch` (the integration branch) plus `tde/active_mask_refactor` (a prerequisite refactor that everything else depends on). They exist so the user can run end-to-end inference benchmarks on specific combinations of features to compare performance and isolate bugs. They're always force-pushed to `gh` (the personal fork `tdene/Megatron-LM`); never pushed to `NVIDIA/Megatron-LM`.

## Branch inventory

All branches are force-pushed to the `gh` remote. The source branches are updated frequently; always rebuild from scratch rather than trying to rebase.

| Feature test branch | Source branches merged (on top of the base) |
|---------------------|----------------------------------------------|
| `tde/feature_test/active_mask_refactor` | *(base only -- `tde/feature_branch` + `tde/active_mask_refactor`)* |
| `tde/feature_test/flashinfer_sampling` | `tde/flashinfer_sampling` |
| `tde/feature_test/per_request_logprobs` | `tde/per_request_logprobs` |
| `tde/feature_test/dedup_attn_tensor_ops` | `tde/dedup_attn_tensor_ops` |
| `tde/feature_test/init` | `tde/dedup_attn_tensor_ops` *(alias for the init-related commits now in dedup_attn)* |
| `tde/feature_test/async_zmq` | `tde/zmq_async` |
| `tde/feature_test/sampling_logprobs_yield` | `tde/flashinfer_sampling` + `tde/per_request_logprobs` + `tde/exclusive_asyncio_gpu_work` |
| `tde/feature_test/sampling_logprobs_yield_zmq` | `tde/flashinfer_sampling` + `tde/per_request_logprobs` + `tde/exclusive_asyncio_gpu_work` + `tde/zmq_async` |

"Source branches" are merged in that order. Most singles have minimal conflicts; the compound branches have many.

## Source branches: what each one does

- **`tde/active_mask_refactor`**: Refactors how the dynamic context slices request-level tensors for the active step. Introduces `build_active_slices` / `pad_active_slices` and removes per-tensor args from `mha_metadata.update`. Most feature branches depend on this.
- **`tde/flashinfer_sampling`**: Adds FlashInfer sampling as an alternative to torch sampling (selectable via `--inference-dynamic-batching-sampling-backend=flashinfer`). Also rewrites speculative sampling/verification to be CUDA-graphable.
- **`tde/per_request_logprobs`**: Rewrites log-prob computation to be selective (only compute for requests that asked) and graphable. Splits `calculate_log_probs` into indexing / softmax / extract phases. Introduces a post-forward bookkeeping stream.
- **`tde/dedup_attn_tensor_ops`**: Deduplicates attention tensor operations by sharing context static tensors. Recent commits on this branch also add sync-less init and graphed init -- this is where the "init" branch name comes from.
- **`tde/exclusive_asyncio_gpu_work`**: Replaces `asyncio.sleep(0)` with a `GPUFuture` that resolves via `cudaLaunchHostFunc`. Passes `loop` through the method signature.
- **`tde/zmq_async`**: Rewrites ZMQ collectives to be async. Removes the `--inference-use-synchronous-zmq-collectives` argument entirely.

## Build recipe

Use these commands from the repo root. Substitute `gh` if your fork remote is named something else. The base must be rebuilt whenever `tde/feature_branch` or `tde/active_mask_refactor` changes; everything downstream must be rebuilt whenever its source branches change (this happens frequently).

### 1. Rebuild the base

```bash
git checkout -B tde/feature_test/active_mask_refactor tde/feature_branch
git merge tde/active_mask_refactor --no-edit
```

Resolve the phantom-token conflict (see the "phantom resolution" section below). Commit.

### 2. Singles (pick whichever you need)

Each of these starts from the base.

```bash
# flashinfer_sampling
git checkout -B tde/feature_test/flashinfer_sampling tde/feature_test/active_mask_refactor
git merge tde/flashinfer_sampling --no-edit

# per_request_logprobs
git checkout -B tde/feature_test/per_request_logprobs tde/feature_test/active_mask_refactor
git merge tde/per_request_logprobs --no-edit

# dedup_attn_tensor_ops (a.k.a. the "init" branch)
git checkout -B tde/feature_test/dedup_attn_tensor_ops tde/feature_test/active_mask_refactor
git merge tde/dedup_attn_tensor_ops --no-edit
# also mirror to tde/feature_test/init for the alias name
git branch -f tde/feature_test/init tde/feature_test/dedup_attn_tensor_ops

# async_zmq
git checkout -B tde/feature_test/async_zmq tde/feature_test/active_mask_refactor
git merge tde/zmq_async --no-edit
```

### 3. Compound branches

`sampling_logprobs_yield` and its `_zmq` variant require sequential merges with conflict resolution at each step.

```bash
# sampling_logprobs_yield = base + flashinfer + logprobs + exclusive_asyncio
git checkout -B tde/feature_test/sampling_logprobs_yield tde/feature_test/active_mask_refactor
git merge tde/flashinfer_sampling --no-edit
git merge tde/per_request_logprobs --no-edit          # <-- 5 conflicts, resolve per the flashinfer+logprobs recipe
git merge tde/exclusive_asyncio_gpu_work --no-edit    # <-- 1 import conflict, then apply the GPUFuture placement fix by hand

# sampling_logprobs_yield_zmq = sampling_logprobs_yield + zmq_async
git checkout -B tde/feature_test/sampling_logprobs_yield_zmq tde/feature_test/sampling_logprobs_yield
git merge tde/zmq_async --no-edit                     # <-- 2 conflicts (engine imports + inference/utils.py)
```

After each merge, verify there are no leftover conflict markers (`grep -rn '<<<<<<<' megatron/ tests/`) before committing.

### 4. Push everything

```bash
git push gh --force \
  tde/feature_test/active_mask_refactor \
  tde/feature_test/flashinfer_sampling \
  tde/feature_test/per_request_logprobs \
  tde/feature_test/dedup_attn_tensor_ops \
  tde/feature_test/init \
  tde/feature_test/async_zmq \
  tde/feature_test/sampling_logprobs_yield \
  tde/feature_test/sampling_logprobs_yield_zmq
```

## Shared conflict zones

Essentially every conflict lives in one of these five files:

- `megatron/core/inference/inference_request.py` -- metadata device placement
- `megatron/core/inference/contexts/dynamic_context.py` -- padding, phantom handling, attn metadata construction
- `megatron/core/inference/text_generation_controllers/text_generation_controller.py` -- init tensors, step pipeline ordering, sampling/logprob method bodies
- `megatron/core/inference/engines/dynamic_engine.py` -- imports and step-level sync
- `megatron/inference/utils.py` -- `InferenceConfig` construction

## Resolution recipe per source branch

### `tde/active_mask_refactor` onto `tde/feature_branch` (base build)

One conflict, in `dynamic_context.py`, inside `initialize_attention_state`. Feature branch added phantom prefill requests for Mamba + MoE; active_mask_refactor added `build_active_slices` / `pad_active_slices` and removed the old per-tensor args from `mha_metadata.update`.

Steps:

1. Keep the phantom code from feature_branch verbatim -- it writes fake query lengths, KV offsets, block IDs, and expands `attn_dimensions`.
2. Add `build_active_slices(self.padded_active_request_count)` and `pad_active_slices()` calls *after* the phantom block. `padded_active_request_count` covers phantom positions, so the active tensors pick them up.
3. Define `attn_req_count = attn_dimensions.req_count` (post-phantom) and slice active tensors with `[:attn_req_count]` when calling `mha_metadata.update`. Do **not** use `batch_size` (= `total_request_count - paused_request_count`) here -- that count excludes phantoms and the MHA assertion `request_query_lengths.shape[0] == real_batch_size` will fail.
4. In the `if self.is_hybrid_model:` block, define `batch_size` locally and pass `mamba_dimensions` (pre-phantom) to the Mamba update. Mamba must ignore phantoms because phantom requests have no allocated Mamba state slot.

### `tde/flashinfer_sampling`

Usually a clean merge; conflicts only appear when dedup_attn or per_request_logprobs are already merged.

- **`inference_request.py`**: flashinfer adds a `sampling_backend` parameter to `get_metadata_types()` that conditionally promotes `temperature`/`top_k`/`top_p` to GPU. Keep it unchanged.
- **`dynamic_context.py` `pad_active_slices`**: flashinfer adds sentinel padding (`temperature=1.0`, `top_k=0`, `top_p=0.0`) for padded slots when `sampling_backend == 'flashinfer'`. Coexists with other padding blocks (logprobs, dedup).
- **TGC init**: flashinfer adds `_fi_any_filtered_pinned`, `_sampling_cuda_graphs`, `_verification_cuda_graphs`, `_last_accepted_seq_indices`. All stay.
- **TGC speculative**: flashinfer's rewritten `_dynamic_step_sample_logits_and_verify_tokens` (with graphed sampling) replaces the old monolithic version. Take flashinfer's version.

### `tde/per_request_logprobs`

The biggest source of conflicts. Five or six at a time.

- **`inference_request.py`**: set `return_log_probs`, `skip_prompt_log_probs`, `top_n_logprobs` to `on_device=True`. The graphable indexing kernels read these inside CUDA graphs, so they must be on GPU. If flashinfer's `sampling_backend` conditional is also present, it still runs afterward and promotes the sampling params -- the two policies are orthogonal.
- **`dynamic_context.py` `pad_active_slices`**: adds `return_log_probs` padding to `False` for padded request slots. Coexists with flashinfer padding and (if present) dedup's `mha_metadata.update` call. Reuse the `active_request_count` / `padding_request_slice` variables across the blocks.
- **`dynamic_context.py` log-prob methods**: per_request_logprobs introduces selective kernel methods (`log_probs_decode_indexing_kernel`, `log_probs_speculative_softmax_kernel`, etc.) and replaces the old monolithic `calculate_log_probs`. Take the new methods; discard the old dead code.
- **TGC init**: adds `_log_prob_count_pinned`, `_top_n_max_pinned`, `_log_prob_graphs`, `_log_prob_graph_outputs`, `_post_forward_bookkeeping_stream`, `_post_forward_bookkeeping_event`, `_log_prob_reduction_event`, `_accepted_token_counts_per_request`. Keep all of these alongside flashinfer's init tensors.
- **TGC methods**: flashinfer's `_sample_logits` and per_request_logprobs's `accepted_tokens`/`_accepted_token_counts_per_request` code are adjacent but not overlapping -- keep both. Take per_request_logprobs's dispatch structure (`_calculate_log_probs_speculative`, `_calculate_log_probs_decode`, `_calculate_log_probs_prefill`) over the old monolithic version.
- **TGC pre-forward stream**: the three calls `sample_bookkeeping`, `log_probs_bookkeeping`, `log_probs_indexing` all run on the pre-forward bookkeeping stream. Combine them in sequence inside the `with torch.cuda.stream(...)` block.
- **TGC post-yield**: after the async yield, use GPU-side `torch.cuda.current_stream().wait_event(self._pre_forward_bookkeeping_event)` -- **never** `event.synchronize()`. The pre-forward indexing kernels write GPU tensors that log-prob softmax kernels on the default stream read -- this is a cross-stream GPU ordering constraint, not a CPU concern. Since both `sample_bookkeeping` and `log_probs_bookkeeping` ran on the pre-forward stream, neither needs to run post-yield.

Two subtleties to keep in mind:

- `_log_prob_reduction_event.synchronize()` inside `_dynamic_step_log_probs_indexing()` IS a deliberate CPU-blocking sync. It's tiny (a `.sum()` + D2H copy) and necessary -- the CPU needs the pinned `.item()` value to decide whether to launch indexing kernels. Do not confuse this with the cross-stream `wait_event` above.
- The `_post_forward_bookkeeping_event` also gets a GPU-side `wait_event` before log-prob calculation, not a `synchronize`. Using `synchronize` here caused a measurable throughput regression (see memory `feedback_cuda_events.md`) because the post-forward stream `wait_stream`'d the default stream, so synchronizing effectively blocked the CPU for the entire forward pass.

### `tde/dedup_attn_tensor_ops`

Three conflicts in `dynamic_context.py`.

- **`__init__` conflict**: dedup adds `num_prefill_requests = 0` and a deferred-init comment. Keep this alongside feature_branch's `model_sync_needs_reservation` / `max_schedulable_requests` block (which dedup doesn't know about). Both blocks live; put them in sequence.
- **Phantom-code-area conflict**: dedup moves the `mha_metadata.update` call from `initialize_attention_state` into `pad_active_slices`, and removes the per-tensor args because `build_active_slices` now writes into shared tensors that MHA reads directly. When merging with feature_branch's phantom code, keep the phantom code first, set `self.attn_dimensions = attn_dimensions` after it (so `pad_active_slices` sees the phantom-expanded dimensions), and drop the explicit `mha_metadata.update` call.
- **Mamba metadata update conflict**: dedup introduces `self.cu_active_request_query_lengths` -- a cumsum tensor shared across MHA and Mamba. In the Mamba update call, use this instead of reading from `mha_metadata.state_data["cu_query_seq_lengths"]`. Keep `batch_dimensions=mamba_dimensions` (pre-phantom).

Recent dedup commits ("Switch active and paused request indices") flipped the request layout to `[active | paused | dead]`. See the phantom layout fix below.

### `tde/exclusive_asyncio_gpu_work`

Typically one import conflict plus a nontrivial hand-edit in the step pipeline.

- **Imports**: combine the multi-line `CUDAGraphCache` import with the single-line `GPUFuture` import. Output in alphabetical order inside the parens: `CUDAGraphCache, GPUFuture, get_attention_mask, set_decode_expert_padding`.
- **Method signature**: add `loop: Optional[asyncio.AbstractEventLoop] = None`. Inside the method, `loop = get_asyncio_loop(loop)` then `gpu_done = GPUFuture(loop)`.
- **Sync wrapper**: the synchronous `generate_output_tokens_dynamic_batch` must pass `loop=loop` through to the async version, otherwise the GPUFuture can't wire its callback to the right event loop.
- **GPUFuture placement**: record `gpu_done` *immediately after* `self._dynamic_step_forward_logits()`, not outside the `with torch.inference_mode()` block. Git auto-merge consistently gets this wrong -- it tends to keep the old `gpu_done.record()` placement (after mamba/routing bookkeeping), and you have to move it up by hand.
- **Yield placement** (per path):
  - Non-speculative: enqueue sampling and log-prob kernels on the default stream *before* `await gpu_done`. They queue behind the forward pass via stream ordering. The CPU wakes up after forward completes, by which time sampling and logprobs are done or nearly done.
  - Speculative: `await gpu_done` immediately after forward, before verification. Verification is CPU+GPU interactive and needs the forward logits materialized.
- **Cross-stream ordering**: after the await, use `torch.cuda.current_stream().wait_event(...)` for the pre-forward and post-forward events. Never `.synchronize()`.

### `tde/zmq_async`

Two conflicts.

- **Engine imports**: zmq_async adds `EngineCoordinatorClient`; base has `Headers, UnknownHeaderError` and `InferenceFLOPsCalculator`. Keep all three imports in alphabetical order.
- **`megatron/inference/utils.py` engine construction**: zmq_async **removed** `use_synchronous_zmq_collectives=args.inference_use_synchronous_zmq_collectives` because the argument no longer exists. Do not keep this kwarg. If flashinfer's `sampling_backend=args.inference_dynamic_batching_sampling_backend` is also present, keep only that one. Falling into the trap of keeping the `use_synchronous_zmq_collectives` line produces a runtime `AttributeError: 'Namespace' object has no attribute 'inference_use_synchronous_zmq_collectives'` on startup.

## The phantom request system

Phantom prefill requests are injected when a model uses both Mamba and expert parallelism (EP). EP requires all ranks to agree on the total token count; Mamba adds constraints on the decode/prefill split. When the padded token count exceeds the decode token budget but there are no real prefill requests, the code injects "phantom prefills" to absorb the extra tokens so the attention kernel has a request to attribute them to.

### How phantoms interact with the layout

The new request layout (from the "Switch active and paused request indices" commit in dedup_attn) is:

```
[0, active_count)                              -> active requests
[active_count, total_request_count)            -> paused requests
[total_request_count, ...)                     -> dead space / reserved
```

Phantoms are logically prefill requests, so they must sit at the *end* of the active region (after real prefills) so MHA sees them as part of the active range. The naive phantom code from feature_branch writes at `[total_request_count, total_request_count + phantom_count)` (the dead zone), which only worked under the old `[paused | active]` layout where the active region extended up to `total_request_count`.

### What the merged code does

Phantom handling is now split across `prepare_attn_init` (before the graphable init body) and `finalize_attn_init` (after):

In `prepare_attn_init`:
1. Compute `phantom_count` and `phantom_tokens` from the gap between `padded_batch_dimensions` and `attn_dimensions`.
2. If any paused requests occupy `[active_count, active_count + phantom_count)`, save their `request_query_lengths`, `request_kv_length_offsets`, and `request_to_kv_block_ids` entries.
3. Write phantom data at `[active_count, active_count + phantom_count)` (contiguous with active).
4. Set `_real_request_count_gpu = active + phantom_count` so MHA's mask lets phantom query lengths contribute to `cu_query_seq_lengths` all the way out to `padded_active_token_count`.
5. Leave `_real_token_count_gpu` pre-phantom so phantom tokens still get dummy block indices at the token level (safe "null" attention via dummy blocks).
6. Leave `_real_decode_count_gpu` / `_real_prefill_count_gpu` pre-phantom so Mamba continues to ignore phantoms (they have no Mamba state slot).

In `finalize_attn_init`:
7. Restore any saved paused entries to their original positions.
8. Clear any phantom positions beyond the overlap (when `phantom_count > paused_count`).

### Why save/restore instead of a reserved phantom slot

The save/restore adds roughly 6 tiny kernel launches per step (only when both `phantom_count > 0` and `paused_request_count > 0`). The overhead is in the tens of microseconds range per step -- not load-bearing given that a forward pass is millisecond-scale.

The more invasive alternatives are all complicated by the MHA layout invariant that requires the active region to be contiguous with `[0, decode_count)` for decode requests and `[decode_count, active_count + phantom_count)` for prefill requests (including phantoms). Specifically:

- Reserving slot 0 as "always phantom" breaks the decode-first ordering invariant (phantom is prefill; can't be at the decode position).
- Reserving a slot at the end of `max_requests` forces MHA to gather from two non-contiguous regions instead of a single `request_query_lengths[:pbs]` read.
- Enlarging `max_requests` by `max_phantom_count` and reserving the tail is a cleaner version of the above but still requires two-region gather in MHA.
- Permanently living with `[active | phantom | paused | dead]` instead of `[active | paused | dead]` pushes the cost from phantom save/restore onto every paused-request shift -- same cycles, different place.

The save/restore is architecturally the least invasive option. If profiling ever shows those kernel launches as meaningful, the best optimization is probably enlarging the request tensors by `max_phantom_count` and adjusting `mha_metadata.update` to gather from two regions (graphable with pre-built indices).

## Recurring mistakes

Catch these with the post-merge verification checklist below.

- **CPU-blocking `synchronize()` survives a merge**. Grep for `\.synchronize()` in the text_generation_controller after every merge. The only legitimate CPU-blocking ones in the step pipeline are `_log_prob_reduction_event.synchronize()` (tiny deliberate sync inside `_dynamic_step_log_probs_indexing`) and the engine's `step_end_event.synchronize()` (also deliberate, after the controller returns). Anything on a bookkeeping event should be `wait_event`.
- **`gpu_done.record()` stays in the wrong place**. After merging exclusive_asyncio, verify that `gpu_done.record()` is directly after `_dynamic_step_forward_logits(...)`, not after mamba commit or routing bookkeeping, and not outside the `with torch.inference_mode()` block.
- **`attn_req_count` vs `batch_size`**. When the phantom code is active, `mha_metadata.update` needs `attn_req_count`-sized slices, not `batch_size`-sized. Mamba wants `batch_size`-sized slices with `mamba_dimensions`. Mixing these up produces `AssertionError: request_query_lengths.shape[0] == real_batch_size`.
- **Sync wrapper forgetting to pass `loop`**. Every time exclusive_asyncio is merged, check that `generate_output_tokens_dynamic_batch` (the sync wrapper) passes `loop=loop` to the async version.
- **Dead references to removed arguments**. zmq_async and other branches sometimes remove argparse arguments. If a conflict kept the `args.foo` reference from HEAD but the branch being merged deleted `args.foo`, you get a runtime `AttributeError` on startup. Always check both sides of a "HEAD vs empty" conflict.

## Post-merge verification checklist

Do this after every merge before committing:

1. **No leftover conflict markers**:
   ```bash
   grep -rn '<<<<<<<' megatron/ tests/
   ```
2. **No illegitimate `synchronize()` calls**:
   ```bash
   grep -n '\.synchronize()' megatron/core/inference/text_generation_controllers/text_generation_controller.py
   grep -n '\.synchronize()' megatron/core/inference/engines/dynamic_engine.py
   ```
   The only expected matches are `_log_prob_reduction_event.synchronize()` (inside `_dynamic_step_log_probs_indexing`) and `step_end_event.synchronize()` (in the engine).
3. **For exclusive_asyncio merges**, eyeball `gpu_done.record()` placement:
   ```bash
   grep -n 'gpu_done\.\|await gpu_done' megatron/core/inference/text_generation_controllers/text_generation_controller.py
   ```
   `gpu_done.record()` should be on the line immediately after `self._dynamic_step_forward_logits(...)`.
4. **For zmq_async merges**, confirm the removed arg is gone:
   ```bash
   grep -n 'inference_use_synchronous_zmq_collectives' megatron/inference/utils.py megatron/core/inference/config.py
   ```
   Both should be empty.
5. **Commit with a descriptive merge message**. Do not skip hooks.
6. **Force-push** to `gh`. Never `NVIDIA/Megatron-LM`.
