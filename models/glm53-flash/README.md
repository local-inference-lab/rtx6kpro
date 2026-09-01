# GLM-5.3-Flash optional Jovian Judgement cuMem sidecar

This directory records the **opt-in** Jovian Judgement r10 cuMem CUDA-IPC
sidecar profile. It is additive: it does not replace the in-tree community
runbook at [`../glm-5.3-flash.md`](../glm-5.3-flash.md), and it does not ship
the default Jovian production Compose, Dockerfiles, or launchers.

The base production-stack proposal (TP4/DCP4 Jovian NVFP4 host and public
Compose, image pins, and `STACK=jovian` launcher) lives in
[PR #85](https://github.com/local-inference-lab/rtx6kpro/pull/85). Those files
are not assumed to exist on this branch. This profile only adds:

- [`compose/glm53-flash-nvfp4-jovian-cumem.yml`](../../compose/glm53-flash-nvfp4-jovian-cumem.yml)
- [`scripts/run-glm53-flash-jovian-cumem-compose.sh`](../../scripts/run-glm53-flash-jovian-cumem-compose.sh)
- [`scripts/test-glm53-cumem-contract.sh`](../../scripts/test-glm53-cumem-contract.sh)

The topology is one GPU-enabled LMCache sidecar process mapped across GPUs 0-3
and one TP4/DCP4 vLLM service. Both run as UID 0 and mount the same root-owned
named broker volume at `/run/lmcache-cumem`.

The runtime contract is:

- served name `glm-5.3-flash`;
- TP4, DCP4, exact model length 1,048,576;
- MTP3, 8,192 maximum batched tokens, 16 maximum sequences;
- explicit FP8 KV with exactly 33,554,432,000 bytes (31.25 GiB) per GPU;
- GPU-memory utilization 0.945;
- CUDA graphs enabled with maximum capture size 128;
- automatic prefix caching and prompt-token details enabled;
- LMCache `lmcache_driven`, chunk size 8,192, 48 GiB L1, and separate object
  groups;
- manual cuMem allocation scoped by the r10 source change to KV only; model
  weights remain on the normal allocator;
- 60-second vLLM and worker shutdown budgets; and
- `restart: unless-stopped` for both services.

## Image and source recipe

The final qualified image ID is
`sha256:1d43855573a38e90215b785fb158498bb3654d75c45cef258c512e08c0036ffb`.
It is a fleet-qualified local artifact, not a published public-registry
digest. The launcher therefore requires `IMAGE` and rejects any local tag
that does not resolve to that exact ID. Do not invent or substitute a registry
location.

The qualified r10 source recipe consists of:

- LMCache
  [`c43866fa9aecd18a7c9f49fa791fdad0655506da`](https://github.com/local-inference-lab/LMCache/commit/c43866fa9aecd18a7c9f49fa791fdad0655506da);
- vLLM transfer-mode validator
  [`b39d501b26`](https://github.com/local-inference-lab/vllm/commit/b39d501b26cbc2acd449b00188ee6321aecc407e)
  ([PR #553](https://github.com/local-inference-lab/vllm/pull/553));
- the r10 KV-only cuMem allocator change, whose public PR is still pending;
- graceful worker shutdown
  [`88cff3d5a9`](https://github.com/local-inference-lab/vllm/commit/88cff3d5a98b95df9ceeff77aa676ef2f18a03b6)
  ([PR #554](https://github.com/local-inference-lab/vllm/pull/554)); and
- the
  [fleet qualification receipt](https://github.com/Apple-Federal-Credit-Union/fleet-infra/pull/310).

Because the KV-only allocator PR is pending and the final image is not
published, these pins document provenance but are not presented as a
standalone public image build. Use the exact qualified local image and the
public Compose/launcher surface:

```bash
export IMAGE=<local-tag-resolving-to-the-qualified-image-id>
export MODEL_DIR=/path/to/glm-5.3-flash
export CACHE_DIR=/path/to/writable-glm53-cache
./scripts/run-glm53-flash-jovian-cumem-compose.sh up
```

Render without starting services:

```bash
IMAGE=sha256:1d43855573a38e90215b785fb158498bb3654d75c45cef258c512e08c0036ffb \
MODEL_DIR=/path/to/glm-5.3-flash \
CACHE_DIR=/path/to/writable-glm53-cache \
./scripts/run-glm53-flash-jovian-cumem-compose.sh config
```

## Qualification evidence

| Gate | Qualified result |
| --- | --- |
| Native cuMem lifecycle | One sidecar PID on GPUs 0-3; allocation/alias deduplication; bidirectional writes; padding canary; partial rollback; unregister/re-register |
| Cold 32k | Coherent; 64 objects / 818,151,424 bytes |
| Automatic unregister | 4 allocations / 184 aliases to 0 / 0; L1 preserved |
| Warm external reload | Coherent; `cached_tokens=24576` |
| Isolation | Unrelated sentinel remained coherent |
| Long context | 524,288 tokens coherent in 84.67 seconds; zero restarts/OOM |
| 32k prefill | 12,336 tok/s; TTFT 2.702 seconds |
| C1 decode | 224.135 tok/s |
| C8 decode | 806.988 tok/s; 305.480 steps/s; acceptance 2.642; queue 0; errors 0 |
| Source hardening | 160 tests passed |
| Selected runtime suites | 129/129, then 116/116 in the subsequent pass |

The 32 GiB KV setting is explicitly rejected. A real 524,288-token request
OOMed during a transient 512 MiB DCP all-gather with approximately 149 MiB
free. The qualified 31.25 GiB setting completed that request coherently.
Other rejected configurations are ordinary CUDA IPC (Xid 31), merged
recurrent object groups (handle corruption), manual cuMem allocation applied
to model weights (load-time OOM), and a zero shutdown timeout (incomplete
unregister).

## Stop and rollback

Always stop vLLM before LMCache so automatic unregister can finish:

```bash
IMAGE=<qualified-local-image> \
MODEL_DIR=/path/to/glm-5.3-flash \
CACHE_DIR=/path/to/writable-glm53-cache \
./scripts/run-glm53-flash-jovian-cumem-compose.sh down
```

A vLLM-only restart preserves the sidecar's L1; stopping the sidecar discards
process-local L1. To roll back publicly, stop this cuMem profile and return to
the production stack proposed in
[PR #85](https://github.com/local-inference-lab/rtx6kpro/pull/85), or to the
community Jovian Judgement runbook in [`../glm-5.3-flash.md`](../glm-5.3-flash.md).
Do not silently approximate an unpublished r10 no-LMCache image.
