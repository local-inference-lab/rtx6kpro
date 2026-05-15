# B12X W4A16 Regression Checkpoint - 2026-05-15

This page records the temporary B12X W4A16 merge/fix work that was tested with
GLM-5.1 and then removed from the production canonical path because it regressed
decode throughput.

## Summary

We merged Luke's newer B12X W4A16 work into our B12X stack and added local fixes
so vLLM could load and run it. The resulting GLM-5.1 DCP1, MTP-off, A16-on run
started successfully, used CUDA graphs, and registered custom allreduce graph
addresses, but decode throughput was lower than the production GLM-5.1 v2 image.

The experiment is preserved for later work, but it should not be used as the
current production baseline.

## Production Baseline

Use the GLM-5.1 v2 production recipe unless intentionally retesting this W4A16
experiment:

- Report: [../glm5.1_v2.md](../glm5.1_v2.md)
- Production image: `voipmonitor/vllm:glm-kimi-canonical-rebase-layered-vllm68b3569f-b12xc929144-flashinfergit1a60071-cutedsl45-20260514`
- Production vLLM commit used by that image: `68b3569f2bd2e8f6568f25f2144fb648b200d832`
- Production B12X commit used by that image: `c929144c7689668b07ca65af10ceadf1c745165d`

The local B12X checkout was restored to:

- Branch: `codex/glm51-kimi-b12x-a16-cpuhangfix-cutedsl45-20260512`
- Commit: `c929144c7689668b07ca65af10ceadf1c745165d`

The local main vLLM checkout was restored to:

- Branch: `codex/glm51-kimi-canonical-a16-upstream-20260511`
- Commit: `d7f89c83eceb35c6bb187bc1415cbc5e287df6ff`

## Archived vLLM Changes

The vLLM-side experimental changes were committed and pushed to an archive
branch:

- Branch: `archive/b12x-w4a16-regression-vllm-20260515`
- Commit: `1d782d0a28ba036de10c99b6e0173b4f78ffa246`
- Link: <https://github.com/voipmonitor/vllm/tree/archive/b12x-w4a16-regression-vllm-20260515>

That archive commit preserves:

- `docker/glm51-canonical-editable-20260512.md`
- `scripts/build-glm51-canonical-editable-image.sh`
- `vllm/model_executor/layers/fused_moe/b12x_moe.py`

The main vLLM changes were the experimental FlashInfer/B12X MoE dispatch path,
scale-factor layout conversion/cache, and editable image env cleanup so model
settings are supplied by explicit launch recipes rather than baked defaults.

Related experimental worktrees that existed at checkpoint time:

- `/root/vllm/worktrees/vllm-b12x-direct-w4a16-20260515`
- Branch: `codex/b12x-direct-w4a16-vllm-20260515`
- HEAD: `51497e10c4259ac6fb2bc0a59add37cdc7cf8b6a`

## Archived B12X Changes

The B12X-side experiment was pushed to a fork because the local GitHub identity
did not have write permission to `local-inference-lab/b12x`.

- Fork branch: `voipmonitor/b12x:codex/b12x-merge-luke-master-20260515`
- Link: <https://github.com/voipmonitor/b12x/tree/codex/b12x-merge-luke-master-20260515>
- Commit: `d77c6ebc9beced6504c7daa075384696bc616993`
- Base production commit: `c929144c7689668b07ca65af10ceadf1c745165d`

The B12X archive branch contains:

- `f76012b` - merge Luke B12X master for W4A16 testing
- `3ba650c` - allow W4A16 workspace planning
- `d77c6eb` - avoid W4A16 scale-factor scratch allocation

It also includes Luke's upstream W4A16 commits that were merged:

- `3267992` - add experimental W4A16 FC2 MMA path
- `5aa74cb` - rewrite W4A16 kernels
- `3dcd574` - W4A16 cleanups

## Local Backup

A local bundle and patch series were also written as a recovery fallback:

- Directory: `/root/vllm/archives/b12x-w4a16-regression-20260515/`
- Bundle: `/root/vllm/archives/b12x-w4a16-regression-20260515/b12x-merge-luke-master-20260515.bundle`
- Patches: `/root/vllm/archives/b12x-w4a16-regression-20260515/patches/`
- Checksum file: `/root/vllm/archives/b12x-w4a16-regression-20260515/SHA256SUMS`

Restore from bundle:

```bash
git clone /root/vllm/archives/b12x-w4a16-regression-20260515/b12x-merge-luke-master-20260515.bundle b12x-archive
cd b12x-archive
git switch codex/b12x-merge-luke-master-20260515
```

Replay patches on top of production B12X:

```bash
git switch -c replay-b12x-w4a16 c929144
git am /root/vllm/archives/b12x-w4a16-regression-20260515/patches/*.patch
```

## Tested Runtime

The tested experimental Docker image was:

```text
voipmonitor/vllm:glm51-prod-b12xmerged77c6eb-directw4a16-test-20260515
```

The GLM test was launched with the production GLM DCP1, MTP-off config plus:

```text
B12X_MOE_FORCE_A16=1
```

Observed startup signals:

```text
speculative_config=None
B12X_MOE_FORCE_A16=1 changes B12X MoE activation numerics
Using AttentionBackendEnum.B12X_MLA_SPARSE backend
Using 'B12X' NvFp4 MoE backend
GPU KV cache size: 249,599 tokens
Registering 1256 cuda graph addresses
Graph capturing finished in 33 secs
Application startup complete
```

The previous production GLM-5.1 v2 DCP1, MTP-off, A16-on reference was around
`59.4 tok/s` on the user's decode test. The experimental B12X W4A16 merge was
around `55 tok/s`, so it was archived instead of promoted.

## Why It Was Archived

The experimental branch is still useful because it gets the latest W4A16 path to
load and run in vLLM, but it is not production-ready:

- Decode throughput regressed versus the GLM-5.1 v2 production image.
- It consumes more model memory with A16 forced, reducing available KV cache.
- It needs a focused kernel-level comparison before promotion.

If revisiting this, start from the archived B12X branch and compare B12X kernels
against the production `c929144` path using the exact GLM-5.1 v2 DCP1 MTP-off
recipe first. Do not merge it into canonical until the DCP1 decode regression is
explained or recovered.
