# DeepSeek-V4-Flash-0731 Infernal Invocation r10

**Status: qualified.** This page specifies fixed probabilistic DSpark K5
serving for `deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell
GPUs. Qualification covers official-checkpoint startup, FULL CUDA graph
capture, native CPU and filesystem KV tiers under concurrent shared-prefix
load, and one warmed single-request decode sanity measurement.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllma7f04eb-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r10` |
| Registry digest | `sha256:2cc10f7ffad3bf390329929e8580fdb96fd4b071816d0ac71f2aba7177cc3756` |
| Image ID | `sha256:9a175e3500b2a0afae875df8773b5a8eaaa66dcaa70a0169a70b1c4400537e70` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@ce5f50f6d01b02336c4207f11277fd7bedacb4d6` |
| vLLM integration tree | `a7f04eb1215330d18421c0179e86077be01d9086` |
| B12X integration tree | `5d648d944a047d4fac5c2035309c207b3faebd9c` |
| LMCache integration tree | `5fdf59cfa184bc15dc5414df0bd633da9e49aaae` |
| Image build source | [`694a55c`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/694a55c77e89fb3ff8ea4e7654df3d90b2d75a23) |
| Qualification receipt | [`validation/infernal-invocation-r10-local-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r10-local-gpu.json) |
| vLLM filesystem publication | [vLLM PR #304](https://github.com/local-inference-lab/vllm/pull/304) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The vLLM, B12X, and LMCache lock directories record each base commit, ordered
pull-request heads, resulting Git tree, and patch digest. Their
`source_patches` arrays are empty.

## Start The Server

Download the immutable Compose profile and start TP2/DCP1 fixed K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/694a55c77e89fb3ff8ea4e7654df3d90b2d75a23/examples/docker-compose-ds4-infernal-invocation-cu133-r10.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r10.yml up -d
```

The profile defaults to B12X W4A8, FP8 compressed MLA KV, InstantTensor
`BUFFERED`, persistent release-scoped JIT storage, and FULL target and DSpark
CUDA graphs.

Native filesystem L2 requires the native host-memory L1 tier. The following
profile uses 5.5 GiB of host-memory L1 and an 8 GiB bounded filesystem tier:

```bash
GPUS=0,1 \
TP_SIZE=2 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=16 GRAPH=auto \
MAX_MODEL_LEN=131072 MAX_NUM_BATCHED_TOKENS=8192 \
GPU_MEMORY_UTILIZATION=0.975 \
KV_OFFLOADING_SIZE=5.5 NATIVE_L2_GB=8 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r10.yml up -d
```

The Compose default stores filesystem blocks under
`/cache/native-kv/8000`, inside the persistent `JIT_CACHE` volume. Set
`NATIVE_L2_PATH` to another absolute container path only when that path is
backed by a persistent mounted volume.

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

`GRAPH=auto` derives the graph cap from scheduler-reachable verifier rows. A
manual cap must represent `MAX_NUM_SEQS * (1 + DSPARK_TOKENS)` rows.

Target-only, fixed K7, and confidence-controlled K7 are implemented by the
entrypoint but are not qualified by this receipt. Their selectors are
`MODE=dspark-mtp0`, `DSPARK_TOKENS=7`, and `DSPARK_DEPTH_MODE=dynamic`.

## Native Filesystem KV Publication

Filesystem KV keys are content-addressed and immutable. Each writer stores a
complete block in a private temporary file, then atomically creates the final
directory entry only if it is absent. If another writer has already published
that key, the completed destination remains unchanged and the losing writer
removes its temporary file.

Replacing the destination with `rename()` is not valid for this contract. Two
writers cannot interleave bytes because they use separate temporary inodes,
but repeated renames can replace the visible inode while readers or the
filesystem track it. [vLLM PR #304](https://github.com/local-inference-lab/vllm/pull/304)
uses create-if-absent publication in both the Python fallback and compiled C
I/O path.

Validation evidence:

| Condition | Result |
|---|---|
| Compiled filesystem unit suite | 49 passed |
| 16 writers, one 8 MiB key, five repetitions | One visible inode in every repetition |
| Two concurrent 72,020-token requests with one shared prefix | Correct responses `17` and `29` |
| Shared-prefix replay | 0.50 s after concurrent stores of 5.75 s each |
| Filesystem state | 1,390 `.bin` blocks, zero `.tmp` files |
| Runtime errors | 0 |

The qualification host uses ext4. The publication invariant is independent of
filesystem type, but a reported btrfs host lockup was not reproduced on ext4.
A btrfs deployment must validate the image under its own concurrent-prefill
workload before treating the kernel-level symptom as closed.

## CC1 Sanity Result

The runtime sanity measurement used TP2/DCP1, B12X W4A8, fixed probabilistic
K5, zero initial context, 5 seconds of warmup, and a 20-second measured stream.
Native CPU KV offload was 5.5 GiB and filesystem L2 was 8 GiB.

| Metric | Result |
|---|---:|
| Aggregate decode | 194.81 tok/s |
| Active-user decode | 206.52 tok/s |
| TTFT | 260.79 ms |
| ITL | 4.87 ms |
| Strict draft acceptance | 31.0% |
| Request errors | 0 |

This is one sanity measurement, not a performance sweep. Probabilistic DSpark
throughput changes with generated content and draft acceptance. The
[Infernal Invocation r7 specification](ds4dspark-infernal-invocation-r7.md)
contains a separate TP2/TP4 performance matrix.

## Structured Tool Calls

The source composition contains reasoning-aware token-zero grammar activation
from [vLLM PR #302](https://github.com/local-inference-lab/vllm/pull/302).
Its 24-request matrix across `auto`, `required`, and named tool selection is
recorded in the
[Infernal Invocation r9 specification](ds4dspark-infernal-invocation-r9.md).
The r10 composition changes filesystem KV publication only; the structured
tool matrix was not re-executed for this receipt.

## Source Merge Contract

| Repository | Pull requests | Responsibility |
|---|---|---|
| vLLM | #285-#296, #298, #300-#304 | Model identity, DS4 runtime contracts, structured output, decode-state graph dispatch, GLM sparse MLA compatibility, DSpark quantization restoration, and immutable filesystem KV publication |
| B12X | #145-#146, #148-#150 | CUTLASS DSL 4.6.2, W4A16 and projection-mixed EXL3 kernels, mixed-rate validation, and capture-safe routing |
| LMCache | #7-#17, #22-#23 | Bounded worker errors, retrieval recovery, durable tiers, hybrid replay, and writer-owned filesystem publication |

The exact pull-request heads and merge state are maintained in
[`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).

## Qualification Limits

- **Qualified:** official-checkpoint startup, fixed probabilistic K5, FULL
  target and DSpark graph capture, native CPU plus filesystem KV tiers under
  two concurrent long-prefix requests, and one warmed CC1 measurement.
- **Implemented:** target-only serving, K7 modes, LMCache, and alternate B12X
  modes.
- **Unsupported claim:** btrfs kernel-level stability is not established by
  the ext4 qualification.
- Performance describes one TP2 subset on a PCIe-switch host. Direct-root-port
  and dual-socket topologies require topology-local measurements.
