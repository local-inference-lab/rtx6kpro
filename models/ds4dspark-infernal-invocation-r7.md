# DeepSeek-V4-Flash-0731 Infernal Invocation r7

**Status: qualified.** This page specifies the reproducible serving profile for
`deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell GPUs. Fixed
probabilistic DSpark K5 is the general-purpose profile. Target-only serving,
fixed probabilistic K7, and confidence-controlled probabilistic K7 use the same
image with explicit runtime settings.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm7ed814e-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r7` |
| Registry digest | `sha256:58568d18ac87bf79095c758f5bc985f3b7a00d133819bf5bd47b935038f3f759` |
| Image ID | `sha256:d870cbe0d5126972cf2986662525a77c856bdb968031c58ec21abfda4afa6f78` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@ce5f50f6d01b02336c4207f11277fd7bedacb4d6` |
| vLLM integration tree | `7ed814e29c18c0b9580a8b09e707d377d10af847` |
| B12X base | `master@954fd0174c49502b62547a01f09c404029e6035a` |
| B12X integration tree | `5d648d944a047d4fac5c2035309c207b3faebd9c` |
| LMCache integration tree | `5fdf59cfa184bc15dc5414df0bd633da9e49aaae` |
| FlashInfer revision | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| Docker build commit | [`9f0ddc8`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/9f0ddc8a3899bef11e57fadf2f6cdd727c034e2d) |
| Qualification receipt | [`validation/infernal-invocation-r7-local-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r7-local-gpu.json) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The image uses immutable integration locks for vLLM, B12X, and LMCache. Each
lock records the base commit, ordered pull-request heads, resulting Git tree,
and integration-patch digest. Every lock has an empty `source_patches` list.

Infernal Invocation revisions identify images built from
`dev/infernal-invocation`. Gilded Gnosis `v20-r*` pages specify a different
source branch and remain separate deployment records.

## Start The Server

Download the immutable Compose profile and start TP2/DCP1 fixed K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/9f0ddc8a3899bef11e57fadf2f6cdd727c034e2d/examples/docker-compose-ds4-infernal-invocation-cu133-r7.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r7.yml up -d
```

The profile defaults to B12X W4A8, FP8 compressed MLA KV, InstantTensor
`BUFFERED`, persistent JIT storage, and FULL target and DSpark CUDA graphs.

TP4/DCP1 uses the same profile with four visible GPUs:

```bash
GPUS=0,1,2,3 \
TP_SIZE=4 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=16 GRAPH=auto \
MAX_MODEL_LEN=131072 MAX_NUM_BATCHED_TOKENS=8192 \
GPU_MEMORY_UTILIZATION=0.975 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r7.yml up -d
```

`GRAPH=auto` derives the verifier-row envelope from
`MAX_NUM_SEQS * (1 + DSPARK_TOKENS)`. A smaller graph cap cannot represent all
scheduler-reachable verifier rows.

## DSpark Profiles

| Purpose | Environment | Status |
|---|---|---|
| Fixed probabilistic K5 | `MODE=dspark DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5` | Qualified default |
| Fixed probabilistic K7 | `MODE=dspark DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=7` | Qualified |
| Confidence-controlled K7 | `MODE=dspark DSPARK_DEPTH_MODE=dynamic DSPARK_TOKENS=7 DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=0` | Qualified |
| Target-only 0731 checkpoint | `MODE=dspark-mtp0` | Qualified target and cache baseline |

Confidence-controlled K7 uses compact variable-length verification. It reduces
useful draft depth when the confidence policy predicts that deeper proposals
are unlikely to survive. Throughput depends on workload entropy and acceptance,
so fixed K5 remains the general-purpose default.

## CUDA Graph Contract

| Stage | Execution contract |
|---|---|
| Target/verifier decode | FULL CUDA graph for captured all-decode scheduler rows |
| DSpark proposal | FULL CUDA graph for captured rows and draft depths |
| DSpark context-KV compression/update | Dedicated FULL CUDA graph |
| Prefill | PIECEWISE or non-FULL model path |
| Rejection sampling | Post-verification orchestration outside the model FULL graph |
| Request metadata and output bookkeeping | Host path |

The model runner dispatches a FULL uniform-decode graph only when every
scheduled request is in decode state. A prefill chunk can have the same row
count as a speculative decode batch, so token shape alone is not a valid graph
selector. The explicit state predicate is implemented by
[vLLM PR #298](https://github.com/local-inference-lab/vllm/pull/298).

## CC1 Performance Qualification

The table reports one active request with zero initial context. `Decode` is the
fixed synthetic stream; `Coding` is the median generation rate from five coding
workload runs. Every row completed without request errors, response-integrity
violations, or JIT compilation after the measurement boundary.

| Parallelism | Backend | Mode | Decode tok/s | Coding median tok/s |
|---|---|---|---:|---:|
| TP2/DCP1 | B12X W4A16 | target-only | 152.57 | 154.60 |
| TP2/DCP1 | B12X W4A16 | fixed K5 | 236.46 | 344.28 |
| TP2/DCP1 | B12X W4A16 | fixed K7 | 208.01 | 321.76 |
| TP2/DCP1 | B12X W4A16 | confidence-controlled K7 | 206.24 | 315.70 |
| TP2/DCP1 | B12X W4A8 | target-only | 150.49 | 152.49 |
| TP2/DCP1 | B12X W4A8 | fixed K5 | 216.57 | 308.10 |
| TP2/DCP1 | B12X W4A8 | fixed K7 | 200.85 | 312.54 |
| TP2/DCP1 | B12X W4A8 | confidence-controlled K7 | 187.25 | 265.74 |
| TP2/DCP1 | B12X W4A8 + DeepGEMM | target-only | 148.47 | 150.38 |
| TP2/DCP1 | B12X W4A8 + DeepGEMM | fixed K5 | 206.51 | 322.70 |
| TP2/DCP1 | B12X W4A8 + DeepGEMM | fixed K7 | 198.12 | 318.31 |
| TP2/DCP1 | B12X W4A8 + DeepGEMM | confidence-controlled K7 | 186.20 | 260.18 |
| TP4/DCP1 | B12X W4A16 | target-only | 178.73 | 181.68 |
| TP4/DCP1 | B12X W4A16 | fixed K5 | 301.24 | 461.20 |
| TP4/DCP1 | B12X W4A16 | fixed K7 | 250.58 | 392.48 |
| TP4/DCP1 | B12X W4A16 | confidence-controlled K7 | 231.75 | 316.69 |
| TP4/DCP1 | B12X W4A8 | target-only | 180.77 | 183.72 |
| TP4/DCP1 | B12X W4A8 | fixed K5 | 260.23 | 423.40 |
| TP4/DCP1 | B12X W4A8 | fixed K7 | 231.46 | 376.55 |
| TP4/DCP1 | B12X W4A8 | confidence-controlled K7 | 248.02 | 319.81 |
| TP4/DCP1 | B12X W4A8 + DeepGEMM | target-only | 183.54 | 186.57 |
| TP4/DCP1 | B12X W4A8 + DeepGEMM | fixed K5 | 272.17 | 417.54 |
| TP4/DCP1 | B12X W4A8 + DeepGEMM | fixed K7 | 271.44 | 428.72 |
| TP4/DCP1 | B12X W4A8 + DeepGEMM | confidence-controlled K7 | 233.40 | 331.81 |

Single-stream DSpark throughput is phase-sensitive because draft acceptance
changes with generated content. The coding workload and fixed stream are
reported separately instead of treating one number as a general hardware
limit.

## Prefill Qualification

A paired TP4/DCP1 B12X W4A8 K5 test used identical 8,194-token prompts,
alternated submission order, and excluded one full-shape warmup per server.

| Image composition | Median prefill tok/s | Range tok/s |
|---|---:|---:|
| r6, B12X tree `1bea00c4` | 12,733.64 | 12,543.26-12,805.38 |
| r7, B12X tree `5d648d94` | 12,822.87 | 12,787.65-12,893.52 |

The r7/r6 median ratio was `1.0070`. The paired measurement qualifies
short-prefill parity; it does not establish performance at higher concurrency.

## Native KV Offload

Native vLLM offload provides a CPU cache with an optional bounded filesystem
tier:

```bash
KV_OFFLOADING_SIZE=40 \
NATIVE_L2_GB=200 \
NATIVE_L2_PATH=/cache/native-kv/8000 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r7.yml up -d
```

TP4/DCP1 fixed K5 with native offload returned valid strict tool calls for
150,003-token and 300,128-token prompts and remained healthy afterward. Native
offload and LMCache are independent KV ownership models and must not be enabled
for one engine at the same time.

## LMCache KV Offload

LMCache uses one cache worker per visible GPU and exposes health, control, and
Prometheus interfaces through one HTTP port:

```bash
MODE=dspark-mtp0 \
LMCACHE_MODE=disk \
LMCACHE_L1_GB=1 \
LMCACHE_L2_GB=16 \
LMCACHE_L2_PATH=/cache/lmcache/8000 \
LMCACHE_HTTP_PORT=8099 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r7.yml up -d
```

The TP4/DCP1 qualification issued three concurrent requests with a shared
65,519-token prefix. All requests returned HTTP 200; all 202 submitted object
chunks completed; no store work or temporary writer files remained. After L1
and prefix-cache reset, an 8,184-token request loaded 7,680 aligned tokens from
filesystem L2 in 0.357 seconds without storing another object.

The filesystem test used ext4. It qualifies the userspace publication and
replay contracts but does not directly execute a btrfs-specific kernel failure
signature.

Health and metrics are available at:

```text
http://127.0.0.1:8099/healthcheck
http://127.0.0.1:8099/metrics
```

## Source Merge Contract

The image composes these source responsibilities:

| Repository | Pull requests | Purpose |
|---|---|---|
| vLLM | #285-#296, #298, #300 | Model identity, DS4 launch and tensor contracts, sparse metadata bounds, memory profiling, structured output, decode-state graph dispatch, and projection-mixed EXL3 loading |
| B12X | #145-#146, #148-#150 | CUTLASS DSL 4.6.2, W4A16 and projection-mixed EXL3 kernels, mixed-rate validation, and capture-safe W4A16 routing |
| LMCache | #7-#17, #22-#23 | Bounded worker errors, retrieval recovery, durable bounded tiers, hybrid object-group replay, and writer-owned filesystem publication |

The exact pull-request heads and merge status are maintained in
[`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).

## Validation And Limits

- Docker source-composition and CUDA 13.3 runtime-contract tests passed.
- Twenty-four TP2/TP4 CC1 profiles passed their request and timestamped runtime
  JIT gates.
- Target, fixed K5, fixed K7, and confidence-controlled K7 decode rows used
  FULL CUDA graph dispatch.
- K7 profiles are qualified but are not promoted over fixed probabilistic K5
  for mixed workloads.
- Performance values describe a PCIe-switch host. Direct-root-port and
  dual-socket systems require topology-local measurements.
- Higher-concurrency throughput is outside this receipt. Set
  `DECODE_CONCURRENCY` explicitly when a workload needs its own concurrency
  matrix.

Machine-readable evidence is stored in the
[Infernal Invocation r7 qualification receipt](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r7-local-gpu.json).
