# DeepSeek-V4-Flash-0731 Infernal Invocation r9

**Status: qualified.** This page specifies a reproducible fixed probabilistic
DSpark K5 profile for `deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000
Blackwell GPUs. The qualification covers startup from the official checkpoint,
reasoning-aware strict tool calls, FULL CUDA graph capture, and one warmed
single-request decode sanity run.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm88aafbf-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r9` |
| Registry digest | `sha256:17a1f7fe09b55e2b0ae05631d6e2248b22d5f91f0fe3d1695bdf9de782b4b5b5` |
| Image ID | `sha256:a6db729681ee898a0cf8d3e4f381c2c7aaec846ac7f3d75dda151b4493c57992` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@ce5f50f6d01b02336c4207f11277fd7bedacb4d6` |
| vLLM integration tree | `88aafbfa10cdb73adc50265a129edc0306541288` |
| B12X integration tree | `5d648d944a047d4fac5c2035309c207b3faebd9c` |
| LMCache integration tree | `5fdf59cfa184bc15dc5414df0bd633da9e49aaae` |
| Image build source | [`33bd3c5`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/33bd3c5b273afa51211d6f6b331d65d31154db00) |
| Release metadata | [`a3eed4e`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/a3eed4e49ac3ab67da19dd283719a0fc410362a6) |
| Qualification receipt | [`validation/infernal-invocation-r9-local-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r9-local-gpu.json) |
| Merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The vLLM, B12X, and LMCache source locks contain ordered pull-request heads,
resulting Git trees, and patch digests. Their `source_patches` arrays are empty.

## Start The Server

Download the immutable Compose profile and start TP2/DCP1 fixed K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/934cf1260ab0a0008a16e7ff9810415318d5e0f5/examples/docker-compose-ds4-infernal-invocation-cu133-r9.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r9.yml up -d
```

The profile defaults to B12X W4A8, FP8 compressed MLA KV, InstantTensor
`BUFFERED`, persistent release-scoped JIT storage, and FULL target and DSpark
CUDA graphs.

The qualified TP2 profile with 40 GiB native CPU KV offload is:

```bash
GPUS=0,1 \
TP_SIZE=2 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=4 GRAPH=auto \
MAX_MODEL_LEN=524288 MAX_NUM_BATCHED_TOKENS=4096 \
GPU_MEMORY_UTILIZATION=0.975 OMP_NUM_THREADS=2 \
KV_OFFLOADING_SIZE=40 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r9.yml up -d
```

`OMP_NUM_THREADS=2` is supported. The helper leaves Torch at two threads and
logs that spin-waiting CPU operators can reduce serving throughput on hosts
with CPU quotas. It does not affect structured-output semantics.

## Serving Contract

| Component | Qualified behavior |
|---|---|
| Checkpoint | Official `deepseek-ai/DeepSeek-V4-Flash-0731` revision |
| Target and draft quantization | `deepseek_v4_fp8` |
| Attention | B12X compressed MLA |
| MoE and linear layers | B12X W4A8 |
| KV cache | FP8 compressed MLA |
| Speculative decoding | Fixed probabilistic DSpark K5 |
| Target/verifier decode | FULL CUDA graph for captured all-decode rows |
| DSpark proposal | FULL CUDA graph for captured verifier rows |
| DSpark context-KV update | Dedicated FULL CUDA graph |
| Prefill | PIECEWISE or non-FULL model path |
| Model loading | InstantTensor `BUFFERED` |

The graph cap is derived from scheduler-reachable verifier rows when
`GRAPH=auto`. A manually configured cap must represent
`MAX_NUM_SEQS * (1 + DSPARK_TOKENS)` rows.

Target-only, fixed K7, and confidence-controlled K7 are implemented by the
entrypoint but were not requalified for this source composition. Their runtime
selectors are `MODE=dspark-mtp0`, `DSPARK_TOKENS=7`, and
`DSPARK_DEPTH_MODE=dynamic`, respectively.

## Strict Tool Calls

Reasoning-aware structural grammars must be active before the first sampled
token. Otherwise, an unconstrained reasoning prefix can consume a complete
tool block and the structural suffix can require the same block a second time.
[vLLM PR #302](https://github.com/local-inference-lab/vllm/pull/302) enforces
the token-zero activation contract for grammars supplied by a reasoning
parser.

The r9 image passed 24 requests across these conditions:

| Dimension | Values |
|---|---|
| Tool selection | `auto`, `required`, named `get_weather` function |
| Transport | buffered, streaming |
| Repetitions | four per tool-selection and transport pair |
| `parallel_tool_calls` | omitted |
| Schema | strict object with one required `city` string |

Every response contained exactly one `get_weather` call with
`{"city":"Prague"}`. There were no duplicate calls, wrong arguments,
protocol-token leaks, or request errors. A normal chat control returned exactly
`ready`.

## DSpark Checkpoint Loading

The in-checkpoint draft model temporarily uses generic FP8 configuration while
vLLM resolves its architecture. The resolved DeepSeek V4 target identity must
restore `deepseek_v4_fp8` before workers construct B12X methods; otherwise the
embedded MXFP4 expert tensors are interpreted through an incompatible generic
FP8 path. [vLLM PR #303](https://github.com/local-inference-lab/vllm/pull/303)
ports the configuration repair from
[vLLM PR #51835](https://github.com/vllm-project/vllm/pull/51835).

The qualified startup loaded the target weights and the DSpark draft from the
official checkpoint; the loader reported 97 draft parameter entries. Runtime
logs reported
`quantization=deepseek_v4_fp8`.

## CC1 Sanity Result

The runtime sanity measurement used TP2/DCP1, B12X W4A8, fixed probabilistic
K5, zero initial context, 5 seconds of warmup, and a 20-second measured stream.
Native CPU KV offload was configured at 40 GiB.

| Metric | Result |
|---|---:|
| Aggregate decode | 194.02 tok/s |
| Active-user decode | 196.13 tok/s |
| TTFT | 259.78 ms |
| ITL | 5.10 ms |
| Strict draft acceptance | 28.35% |
| Request errors | 0 |

This is one sanity measurement, not a performance sweep. Probabilistic DSpark
throughput changes with generated content and draft acceptance. The broader r7
TP2/TP4 sweep remains available in the
[Infernal Invocation r7 specification](ds4dspark-infernal-invocation-r7.md),
but its values are not presented as r9 measurements.

## KV Offload

Native vLLM offload and LMCache are separate KV ownership models and must not
be enabled for one engine at the same time. The r9 qualification initialized a
40 GiB native CPU cache. It did not exercise eviction, filesystem replay, or
LMCache replay. Use the r7 page for the qualified LMCache filesystem recipe
until those workloads are measured on the r9 source composition.

## Source Merge Contract

| Repository | Pull requests | Responsibility |
|---|---|---|
| vLLM | #285-#296, #298, #300-#303 | Model identity, DS4 runtime contracts, structured output, decode-state graph dispatch, GLM sparse MLA compatibility, token-zero tool grammar, and DSpark quantization restoration |
| B12X | #145-#146, #148-#150 | CUTLASS DSL 4.6.2, W4A16 and projection-mixed EXL3 kernels, mixed-rate validation, and capture-safe routing |
| LMCache | #7-#17, #22-#23 | Bounded worker errors, retrieval recovery, durable tiers, hybrid replay, and writer-owned filesystem publication |

The exact pull-request heads and merge status are maintained in
[`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).

## Qualification Limits

- Qualified: official-checkpoint startup, fixed probabilistic K5, normal chat,
  24 strict-tool requests, FULL target and DSpark graph capture, and one warmed
  CC1 sanity measurement.
- Implemented but not requalified: target-only serving, K7 modes, LMCache,
  native filesystem L2, alternate B12X modes, prefill, and higher concurrency.
- Performance describes one TP2 subset on a PCIe-switch host. Direct-root-port
  and dual-socket topologies require topology-local measurements.
