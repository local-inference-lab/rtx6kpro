# Review of `upstream-pr-plan.md`

**Verdict: plan is directionally correct and largely approvable**, with
**one critical fix (PR E)**, **two moderate issues**, and a handful of minor
tightenings. Detail below. Every claim I check is verified against the
running container (`mystifying_herschel`) or against
`origin/main` in `/opt/vllm`.

---

## Verified correct

- **Base tag / branch / commit** `7dab799d…` on
  `snapshot/kimi-k25-eagle3mla-current-20260423`, tag
  `kimi-k25-eagle3mla-current-20260423`. Matches
  `git rev-parse HEAD` in the container exactly.
- **Three modified files** are confirmed by `git diff --stat`:
  ```
  vllm/v1/attention/backends/mla/triton_mla.py   | 460 ++++++++++++++++---------
  vllm/v1/worker/gpu_model_runner.py             |  12 +-
  ```
  plus the new untracked file `triton_mla_tuning.py`. The plan's correction
  over my writeup's README (which claimed only `triton_mla.py` was material)
  is correct. **The README in this directory needs fixing accordingly.**
- **Checksums match**:
  ```
  patches/triton_mla_final.py        md5=5560dfe1…  ==  container
  patches/triton_mla_tuning.py       md5=f0604423…  ==  container
  ```
- **Served-model-name observation** (`Kimi-K2.5` tag in a container serving
  `moonshotai/Kimi-K2.6`) is a real inconsistency — my launch scripts in the
  session used the K2.5 tag for client compatibility during testing, but
  the canonical issue recipe should use `--served-model-name Kimi-K2.6`.
- **"What not to include"** list is accurate — the four patches named
  (`mla_eagle_instrumentation`, `mtp_sync_fix_and_instrumentation`,
  `flashinfer_mla_spec_fix_PROPOSED`, NCCL XML) are all local/debug state
  that should not go upstream.
- **Existing-PR disposition** (#40609 keep, #40610 keep, #40611 rework,
  #40612 drop, #40613 close, #40614 supersede) matches the code reality.

---

## Critical: PR E is almost certainly unnecessary upstream

The plan proposes PR E:
> *In `use_async_spec_decode`, keep GPU tensors authoritative. Do not retain
> optimistic `seq_lens_cpu`. Set `seq_lens_cpu = None`.
> Set `num_computed_tokens_cpu = None`.*

Problem: **upstream `origin/main` already has this exact code.** Verified
directly:

```
# origin/main vllm/v1/worker/gpu_model_runner.py, around the same block:
if self.use_async_spec_decode:
    # GPU tensors are authoritative in async mode.
    seq_lens_cpu = None
    num_computed_tokens_cpu = None
```

That is bit-for-bit identical to the shipping image's final state. The
`final_diff.patch` is not a forward-going improvement against upstream — it
is **a revert** of an earlier local-fork change that lived on the snapshot
branch `kimi-k25-eagle3mla-current-20260423`. That earlier change kept
`seq_lens_cpu` populated with `optimistic_seq_lens_cpu` values for MLA
chunked-prefill workspace sizing and was (correctly) abandoned as a dead end
in v2 of the v1 Docker stack. The snapshot branch simply still had the v1
code when my session started, so the container now shows a dirty delta that
reverts it back to upstream behavior.

**Action required in the plan:** drop or rewrite PR E. Options:

- **Drop PR E entirely.** Upstream is already correct. Document in the
  issue: "When rebasing from the snapshot branch onto upstream master, drop
  the old `optimistic_seq_lens_cpu` retention so that the `async_spec_decode`
  branch in GPUModelRunner matches upstream. No upstream PR needed."
- **Keep PR E as a no-op explanatory commit**, only if
  reproduction-from-snapshot-branch is part of the issue's audience. Either
  label it clearly as a revert, or just skip it.

This matters because filing a PR that turns out to be a no-op against master
wastes reviewer time and looks like the author didn't verify.

---

## Moderate issue 1: tuning-table hardware guard missing

The plan's PR D says:

> *Guard the tuned path to sm120 + fp8 MLA only.*

but the current code at `patches/triton_mla_final.py` does not do that. The
`lookup_config(q_num_heads, max_model_len, B)` call inside `forward_mqa`
fires on **every** hardware/dtype combination, and if a match is found it
will apply an sm120-fp8-tuned `(num_kv_splits, BLOCK_N, BLOCK_H, num_stages,
num_warps)` tuple to an H100 / MI300 / bf16-KV workload. That's wrong.

**Fix direction** (needs to be added to PR D):

```python
# In TritonMLAImpl.__init__:
self._use_tuned_config = (
    current_platform.is_cuda()
    and current_platform.has_device_capability(120)
    and is_quantized_kv_cache(self.kv_cache_dtype)
    and _lookup_tuned_config is not None
)

# In forward_mqa:
kernel_cfg = None
if self._use_tuned_config and not envs.VLLM_BATCH_INVARIANT:
    kernel_cfg = _lookup_tuned_config(q_num_heads, self._tuning_max_model_len, B)
if kernel_cfg is None:
    kernel_cfg = {...analytical fallback...}
```

Without this, PR D is a regression risk for non-sm120 users and is unlikely
to be accepted upstream.

---

## Moderate issue 2: PR C / PR D boundary is ambiguous on the kernel-call inlining

The session change includes **inlining** of `_fwd_grouped_kernel_stage1` +
`_decode_softmax_reducev_fwd` in `forward_mqa`, replacing the
`decode_attention_fwd` wrapper call. That inlining is required because the
wrapper hard-codes `BLOCK_N`, `BLOCK_H`, `num_stages`, `num_warps`, and we
need to override them from the tuning table.

The plan puts:

- **PR C**: batch-aware `num_kv_splits` + `zero_()` removal
- **PR D**: tuning-table lookup + sm120/fp8 guard + tuning scripts

But the inlining of the two kernel calls, which is *co-resident in the
same file* as both changes, isn't explicitly assigned to either PR. Because
`num_kv_splits` alone can be varied via the existing wrapper
(`decode_attention_fwd(…, num_kv_splits=…)` already takes it as an arg),
PR C does **not** strictly require inlining. PR D does.

**Recommendation**: explicitly state the inlining lives in PR D. Keep PR C
clean as a minimal `num_kv_splits = _pick(…) + buffer zeros-init + remove
per-step zero_()` diff — that makes PR C independently useful even without
the tuning-table infrastructure.

If reviewers instead prefer to collapse PR C + PR D into one (fair — it's
the same file, one subsystem), label it clearly.

---

## Minor items

### 1. My README's base commit hash was wrong

The `README.md` I wrote says:
> *vLLM base commit: `47fcb8ca6` on branch
> `snapshot/kimi-k25-eagle3mla-current-20260423`.*

That's **upstream `origin/main`**, not the snapshot branch. Actual snapshot
HEAD is `7dab799d…` (tag `kimi-k25-eagle3mla-current-20260423`). The plan
has this right; the README doesn't. Fix the README provenance block in a
follow-up commit, or include the correction when opening the issue.

### 2. Dead constant `SM120_FP8_SINGLE_REQ_MAX_KV_SPLITS`

`triton_mla_final.py` still defines:
```python
SM120_FP8_SINGLE_REQ_MAX_KV_SPLITS = 128
```
but the constant is unused after `MAX_NUM_KV_SPLITS=64` + the tuning-table
replaced the old single-request adaptive path. Remove it in PR C or PR D
for cleanliness. (Reviewers will grep for "unused" and complain otherwise.)

### 3. Ship the tuning table generator, not the table, upstream

The plan's Open Decisions #3 asks:
> *Should the tuning table be submitted upstream as generated code guarded
> to sm120/fp8, or should upstream only get the analytical fallback plus
> tuning script?*

Strong recommendation: **ship the script and a stub `TUNED_KV_CONFIGS = {}`
upstream**, plus a README explaining how to regenerate per-hardware. Reasons:

- An sm120-specific table adds no value on H100 / MI300 / consumer Blackwell
  SKUs with different SM count / shmem size.
- A generated data file ties the PR to one vendor's HW revision and dates
  immediately.
- The analytical `_pick_num_kv_splits` is a good default; the tuning-table
  path can be opt-in (e.g. `if _lookup_tuned_config is not None`).

If upstream reviewers disagree and want a pre-generated table, treat it as a
maintained artefact like the AMD/cuBLAS GEMM heuristics: guarded strictly by
`has_device_capability(120) + fp8_kv + matching head count`.

### 4. Tests

The plan doesn't mention upstream-expected tests. Upstream will at minimum
want:

- A **correctness smoke test** of the spec-verify decode path under DCP=1
  (asserts outputs match between the old `decode_attention_fwd` wrapper path
  and the new inline path on identical inputs).
- A **shape-coverage test** for the CG capture path (exercise the full range
  of `B` buckets so `_pick_num_kv_splits` / `lookup_config` is hit for each).
- If a tuning table is shipped, a test that round-trips
  `tune_triton_mla.py → aggregate_tune.py → TUNED_KV_CONFIGS` for one outer
  point, to catch accidental format changes.

These are roughly a day's work and should be added as part of PR A (for the
correctness smoke) and PR C / PR D (for shape coverage).

### 5. PR A + PR B can reasonably be a single PR

Open Decisions #2: "*Should PR A and PR B be split, or merged into one
TRITON_MLA correctness/CG PR because the code is tightly coupled?*"

Recommendation: **merge them.** The DCP-aware `_build_decode` modifies the
same function that Phase-1 introduces; they touch the same builder subclass
and the same `_build_decode` method. Splitting invites review ping-pong over
whether the Phase-1 PR can be merged without the Phase-2 correctness fix
(short answer: no, DCP>1 without `supports_dcp_with_varlen=True` asserts).

The disposition of the existing `#40609–#40614` PRs stays as in the plan.

### 6. Plan should mention `BLOCK_H` / `num_stages` findings

The tuning discovered that **num_stages=1 wins 108/108** and **BLOCK_H=32
wins majority** on sm120 fp8 — i.e. the `decode_attention_fwd` wrapper's
existing `BLOCK_H=8` override for sm120 fp8 is a mis-tuned default.

This finding is arguably worth **an additional small PR** against
`vllm/v1/attention/ops/triton_decode_attention.py` to revise or remove that
override. Such a PR is trivial (two-line heuristic revision), helps
non-TRITON_MLA sm120 fp8 callers too, and doesn't need the full tuning
infrastructure. Consider adding to the plan as a "PR F (optional)".

---

## Verification checklist additions

The plan's own verification checklist is good. Add two items:

- **PR E sanity**: confirm `origin/main`'s `gpu_model_runner.py` at the same
  block (around `use_async_spec_decode`) already has `seq_lens_cpu = None`
  and `num_computed_tokens_cpu = None`. (I did this; it does.) This
  retrospectively confirms PR E is a no-op.
- **Tuning script reproducibility**: run `bench/tune_triton_mla.py` on a
  clean machine of the same HW class, confirm the regenerated
  `triton_mla_tuning.py` matches the committed one modulo noise (exact ms
  values will differ, but winners should agree for most outer points).

---

## Plan's "Open decisions" — my recommended answers

1. *Close and replace `#40614`?* — **close**. The new PR set (A/B/C/D) is a
   strict superset and the old one was narrow to one `SM120_FP8_SINGLE_REQ_MAX_KV_SPLITS`
   constant that is about to become dead code.
2. *PR A+B split or merge?* — **merge** (see §5 above).
3. *Tuning table upstream as generated code or script-only?* —
   **script-only** (see §3 above).
4. *Keep Kimi-K2.5 in the issue?* — **no**, make the canonical issue
   Kimi-K2.6-first. Mention K2.5 only in a "historical baseline" paragraph
   that links to whatever earlier issue tracked it.

---

## Summary — what to change in `upstream-pr-plan.md`

1. **Rewrite or drop PR E.** Upstream main is already in the target state.
2. **Add sm120+fp8 gate** to PR D content.
3. **Assign the kernel-call inlining** explicitly to PR D in the PR C / PR D
   scope sections.
4. **Add a testing section** with the three test items above.
5. **Optionally add "PR F"** for the `_decode_grouped_att_m_fwd` BLOCK_H /
   num_stages default revision.
6. **Fix README base-commit hash** (unrelated to the plan file, but blocks
   the "Verify checksums" step in the plan's checklist).
7. **Note the dead `SM120_FP8_SINGLE_REQ_MAX_KV_SPLITS` constant** for
   removal in PR C/D.

With those changes applied, I approve the plan.

---

*Reviewer: agent, 2026-04-24. All file-level claims verified against
`mystifying_herschel` container + `origin/main` via `git show`. No upstream
GitHub state (issues, PRs) was queried; the plan's claims about #40608,
#40609–#40614 are accepted as stated.*
