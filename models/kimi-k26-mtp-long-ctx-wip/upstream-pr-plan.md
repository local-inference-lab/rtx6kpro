# vLLM upstream issue / PR plan

Status: plan only. No GitHub issue or PR was changed from this file.

## Goal

Make the existing vLLM tracking issue sufficient for another engineer or agent
to reconstruct the exact code state used by the shipping Docker image:

```text
voipmonitor/vllm:cu130-mtp-tuned-v3-20260423
```

The reconstruction must include the previously validated local Kimi patches and
the newer Kimi-K2.6 MTP long-context changes documented in this directory.

## Reference state

Authoritative runtime container:

```text
mystifying_herschel
```

Base vLLM snapshot:

```text
voipmonitor/vllm tag:    kimi-k25-eagle3mla-current-20260423
voipmonitor/vllm branch: snapshot/kimi-k25-eagle3mla-current-20260423
base commit:             7dab799d36af5c3eb4dd980f248ab4ba7a56e170
```

Observed final dirty delta inside the running container:

```text
vllm/v1/attention/backends/mla/triton_mla.py
vllm/v1/attention/backends/mla/triton_mla_tuning.py
vllm/v1/worker/gpu_model_runner.py
```

Checksum verification already done:

```text
patches/triton_mla_final.py
  == /opt/vllm/vllm/v1/attention/backends/mla/triton_mla.py

patches/triton_mla_tuning.py
  == /opt/vllm/vllm/v1/attention/backends/mla/triton_mla_tuning.py
```

Important correction to the README:

The README says the material code changes are only in `triton_mla.py` and
`triton_mla_tuning.py`. That is not fully accurate. The running container also
has a real final delta in `vllm/v1/worker/gpu_model_runner.py`, represented by
`patches/final_diff.patch`.

This `gpu_model_runner.py` delta is needed only when reconstructing from the
older snapshot branch. Upstream `main` already has the final desired behavior:

```python
if self.use_async_spec_decode:
    # GPU tensors are authoritative in async mode.
    seq_lens_cpu = None
    num_computed_tokens_cpu = None
```

Do not open a new upstream PR for this; document it as a local snapshot drift
that must be dropped/reverted when moving from the snapshot to upstream `main`.

## Issue update plan

Existing tracking issue:

```text
vllm-project/vllm#40608
https://github.com/vllm-project/vllm/issues/40608
```

Current status: updated to the Kimi-K2.6-first reconstruction plan and linked
to draft consolidation PR `vllm-project/vllm#40750`.

The current issue is still centered on the earlier Kimi-K2.5/K2.6 DCP/FP8/Eagle
work. It should be updated to the current canonical stack:

```text
target model: moonshotai/Kimi-K2.6
draft model:  lightseekorg/kimi-k2.5-eagle3-mla
hardware:     8x RTX 6000 Pro Blackwell, sm120
attention:    TRITON_MLA target + draft
KV cache:     fp8
spec decode:  eagle3, num_speculative_tokens=3
image:        voipmonitor/vllm:cu130-mtp-tuned-v3-20260423
```

The revised issue should include:

- Exact base tag / branch / commit.
- Exact Docker image tag.
- Exact list of source files required to reproduce the final image.
- Link to this documentation directory in `voipmonitor/rtx6kpro`.
- Link to `patches/triton_mla_final.py`.
- Link to `patches/triton_mla_tuning.py`.
- Link to `patches/final_diff.patch`.
- Final benchmark matrix from the README.
- KV-cache capacity table from the README.
- Short-context speed recipe using `--max-model-len 150000`.
- Long-context flexibility recipe using `--max-model-len 262144`.
- Note that `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN` is no longer required for
  the final MTP path.
- Note that XML/NCCL is a runtime dependency for best-known performance today,
  not a vLLM code change.

The issue should make this reconstruction path explicit:

1. Start from `voipmonitor/vllm:kimi-k25-eagle3mla-current-20260423`.
2. Apply or cherry-pick the upstream PR set that corresponds to the base local
   Kimi stack.
3. Apply the new MTP long-context TRITON_MLA patches.
4. If starting from the snapshot branch, apply `patches/final_diff.patch` or
   otherwise ensure `gpu_model_runner.py` matches upstream `main` at the
   `use_async_spec_decode` block. If starting from upstream `main`, no patch is
   needed for this file.
5. Verify resulting files match the Docker-image checksums.

## Existing vLLM PR status

Existing PRs from the previous work:

```text
#40609 [Core] Enable FP8 KV cache with DCP for MLA
#40610 [SpecDecode] Fix async proposer synchronization
#40611 [SpecDecode] Allow draft-specific attention backend and KV dtype
#40612 [SpecDecode] Add local argmax helper for Llama Eagle3 drafts
#40613 [SpecDecode] Add seq-length gate for speculative decode
#40614 [Attention] Tune TRITON_MLA for SM120 + FP8 decode
```

Recommended handling:

- Keep / rebase `#40609`. It is still a prerequisite for MLA + DCP + fp8 KV.
- Keep / rebase `#40610`. It is a separate async proposer lifetime/race fix.
- Keep / rework `#40611`. It is still needed for target/draft backend and KV
  dtype split, but Gemini correctly asked for cleaner centralization of the CP
  override logic.
- Remove `#40612` from the Kimi canonical path. It is about Llama Eagle3 local
  argmax and is not needed for the Kimi MLA draft stack.
- Close `#40613`. The old seq-length speculation gate was a local workaround.
  The final MTP image no longer requires `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN`.
- Close/supersede `#40614`. The old narrow SM120 fp8 split/BLOCK_H patch is
  replaced by the newer FULL-CG + batch-aware split + tuning-table implementation.

## New PRs to prepare

### Draft consolidation PR: `#40750`

```text
https://github.com/vllm-project/vllm/pull/40750
```

This is the runnable upstream-main reconstruction branch used for end-to-end
container validation. It includes the prerequisite local Kimi PRs plus the new
TRITON_MLA MTP work. It can be split after performance is validated.

### PR A: TRITON_MLA FULL CUDA graph support + DCP correctness

Scope:

```text
vllm/v1/attention/backends/mla/triton_mla.py
```

Content:

- Add `TritonMLAMetadataBuilder`.
- Set `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH`.
- Set `query_len_support = QueryLenSupport.UNIFORM`.
- Route eagle3/MTP spec-verify batches through decode instead of prefill.
- Add persistent CUDA-graph-safe shared buffers for `o`, `lse`, `attn_logits`.
- Add persistent metadata-builder buffers for expanded `block_table` and
  per-query `seq_lens`.
- Set `supports_dcp_with_varlen=True` for the TRITON_MLA metadata builder.
- Fix per-query sequence length expansion under DCP.
- Use `dcp_tot_seq_lens_device` to build global per-query lengths.
- Convert global per-query lengths back to local lengths with
  `get_dcp_local_seq_lens`.

Reason:

This is the primary long-context MTP performance and correctness fix. It removes
the PIECEWISE attention gap, enables full CUDA graph capture for TRITON_MLA
spec-verify shapes, and keeps DCP=4/DCP=8 correct. Splitting the DCP part out
would make the FULL-CG PR unsafe for DCP users, so keep this as one PR.

### PR C: Batch-aware num_kv_splits and CG block-table zeroing cleanup

Scope:

```text
vllm/v1/attention/backends/mla/triton_mla.py
```

Content:

- Replace fixed single-request `CG_NUM_KV_SPLITS=64` behavior with
  `_pick_num_kv_splits(B, q_num_heads)`.
- Keep shared `attn_logits` allocated at `MAX_NUM_KV_SPLITS=64`.
- Pass a sliced view for the selected per-bucket split count.
- Initialize `_cg_buf_block_table` with `torch.zeros` once.
- Remove the per-step tail `zero_()`.
- Remove the dead `SM120_FP8_SINGLE_REQ_MAX_KV_SPLITS` constant if it is still
  present after rebasing.

Reason:

The fixed 64-split version wins for single-request long context but regresses
high-concurrency decode by oversubscribing stage-1 and stage-2 merge work.
The per-step `zero_()` was also a measurable MTP overhead at large
`max_model_len`.

### PR B: sm120/fp8 TRITON_MLA tuning table and tuning tools

Scope:

```text
vllm/v1/attention/backends/mla/triton_mla.py
vllm/v1/attention/backends/mla/triton_mla_tuning.py
benchmarks or scripts for regeneration
```

Content:

- Add generated `triton_mla_tuning.py`.
- Use `lookup_config(q_num_heads, max_model_len, B)` when available.
- Fall back to the analytical split heuristic when the table has no match.
- Guard the tuned path to CUDA + sm120 + fp8 KV + TRITON_MLA only.
- Keep the exact measured table in vLLM, not just the generator script. The
  point of this PR is to ship the measured Blackwell behavior, not require users
  to regenerate it before they get the speedup.
- Include strict fallback behavior for all non-matching hardware/dtype/model
  shapes.
- Inline `_fwd_grouped_kernel_stage1` + `_decode_softmax_reducev_fwd` in
  `forward_mqa`. This belongs in this PR because the existing wrapper fixes
  `BLOCK_N`, `BLOCK_H`, `num_stages`, and `num_warps`; the tuning table cannot
  be used without passing these constants directly.
- Include or reference the tuning scripts:
  - `bench/tune_triton_mla.py`
  - `bench/aggregate_tune.py`

Reason:

This is hardware-specific performance tuning. It should not affect other GPUs or
non-fp8 MLA paths. The table should be treated like other device-specific
kernel heuristics: guarded tightly, generated reproducibly, and ignored outside
its measured scope.

## Not an upstream PR: final async-spec metadata cleanup

`patches/final_diff.patch` changes `gpu_model_runner.py` back to upstream
behavior:

```python
if self.use_async_spec_decode:
    # GPU tensors are authoritative in async mode.
    seq_lens_cpu = None
    num_computed_tokens_cpu = None
```

This is needed only when reproducing the Docker image from the older snapshot
branch. Upstream `main` already contains this code, so filing a separate PR
would be a no-op.

## What not to include

Do not include these in upstream PRs unless explicitly promoted later:

- `.bak_*` files from the Docker container.
- `mla_eagle_instrumentation.patch`.
- `mtp_sync_fix_and_instrumentation.patch`.
- `flashinfer_mla_spec_fix_PROPOSED.patch`.
- Any local debug timing module unless converted into a proper benchmark/test.
- The NCCL XML itself.
- NCCL no-XML topology patch; that belongs to the NCCL PR, not vLLM.

## Verification checklist

Before changing the GitHub issue or opening new PRs:

1. Create a clean branch from tag `kimi-k25-eagle3mla-current-20260423`.
2. Apply only the three final file deltas listed above.
3. Compare checksums against the running container:

```text
vllm/v1/attention/backends/mla/triton_mla.py
vllm/v1/attention/backends/mla/triton_mla_tuning.py
vllm/v1/worker/gpu_model_runner.py
```

4. Confirm no instrumentation/debug files are included.
5. Confirm the documented launch recipe uses `--served-model-name Kimi-K2.6`.
   The currently running container showed `--served-model-name Kimi-K2.5` while
   serving `moonshotai/Kimi-K2.6`; that should not be copied into the canonical
   issue recipe.
6. Confirm whether the issue wants to present DCP=1, DCP=4, and DCP=8 all as
   supported recipes or just recommend:
   - DCP=1 for best single-stream latency.
   - DCP=4 for mixed concurrency / long context.
   - DCP=8 for max KV capacity and high concurrency.

## Upstream test checklist

Expected tests for the PRs:

- Correctness smoke for TRITON_MLA spec-verify decode under DCP=1 and DCP>1.
- Shape coverage for the FULL-CG path across representative batch buckets.
- Test that the tuning lookup is disabled outside CUDA + sm120 + fp8 KV.
- Test that the tuning lookup falls back cleanly when `(heads, max_model_len, B)`
  has no exact entry.
- Lightweight round-trip for the tuning table generator and aggregator format.

## Open decisions for approval

Resolved decisions:

- Close/supersede `#40614` (done).
- Close `#40613` (done).
- Merge FULL-CG and DCP correctness into one TRITON_MLA PR.
- Submit the tuning table as part of vLLM, guarded to CUDA + sm120 + fp8 KV.
- Make the issue Kimi-K2.6-first. Mention Kimi-K2.5 only as historical baseline
  and as the draft-model name where relevant.
