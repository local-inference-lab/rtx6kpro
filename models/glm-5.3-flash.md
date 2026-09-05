# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke for
Local Inference Lab.</em></p>

This page specifies GLM-5.3-Flash deployments for NVIDIA RTX PRO 6000
Blackwell Workstation Edition GPUs. The main published R25 image is qualified
at TP4. A separate published, pull-ready R21 source-overlay child is
hardware-qualified at TP3 for the
`local-inference-lab/GLM-5.3-Flash-NVFP4` target checkpoint without
speculation, with three-token Multi-Token Prediction (MTP), and with the
`local-inference-lab/GLM-5.3-Flash-DFlash2` draft checkpoint.

The published-artifact commands use Hugging Face repository names and named
Docker volumes. The TP3 evidence remains locked to its three R21 review heads
and the R21 child; it does not qualify the main published R25 image at TP3.

## Status

| Capability | Status |
|---|---|
| Tensor parallelism of three with expert parallelism of three | **qualified and published** as the pull-ready R21 child for no speculation, MTP depth 3, and DFlash2 depth 7; exact heads and evidence are linked below |
| Tensor parallelism of four with one decode-context rank | **qualified** for no speculation, MTP depth 3, and DFlash2 depth 7 |
| Tensor parallelism of four with four decode-context ranks | **qualified** for the same three serving modes, including complete-KV prefill |
| Two decode-context ranks | **implemented**; not independently performance-qualified for this artifact |
| Tensor parallelism of eight | **implemented**; not independently hardware-qualified for this artifact |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4`; Hugging Face `main` unless `MODEL_REVISION` is set |
| QAD research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step1750`](../kld/glm-5.3-flash-qad-step1750.md); distribution fidelity, verifier-backed behavior, and AA-LCR are measured, but the checkpoint is not a qualified serving target |
| AA-LCR capability evaluation | **qualified** for the exact BF16, published-NVFP4, and QAD checkpoint-and-runtime configurations in the [three-configuration report](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) |
| Verifier-backed behavioral fidelity | **qualified execution; inconclusive one-point decision** for the topology-matched BF16, published-NVFP4, and QAD checkpoints in the [VBF report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2`; Hugging Face `main` unless `DFLASH_MODEL_REVISION` is set |
| Target routed experts | ModelOpt NVFP4 using B12X 4-bit weights and 4-bit activations |
| DFlash2 weights | Offline-serialized ModelOpt MXFP8; no online weight quantization |
| Target KV cache | FP8 by default; packed NVFP4 is selectable |
| GPU prefix cache | **qualified** with independently sized target and recurrent allocations |
| Native DRAM offload | **qualified** and opt-in with `CACHE_MODE=native` |
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache`; asynchronous engine-driven pinned shared memory is the default transfer path |
| CUDA graphs | **qualified** with launcher default `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` for target and speculative decode |
| Scheduler | 4,096 target tokens per step; execution-time compute-share fairness assigns 40% of contended model execution to prefill; scheduling interval 1 |
| Root filesystem | Three layers, within standard Docker overlay2 limits |
| FlashKDA numerical stability | **qualified** with the stable FP32 forward-substitution inverse |
| Qualification date | 2026-09-04 |

The [BF16-to-NVFP4 distribution-fidelity report](../kld/glm-5.3-flash-bf16-nvfp4.md)
and [QAD step 1,750 comparison](../kld/glm-5.3-flash-qad-step1750.md)
are research-only. They measure a reproducible FlashInfer CUTLASS path rather
than the B12X serving path specified here.

The [AA-LCR result](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) qualifies the BF16,
published NVFP4, and QAD configurations on 100 long-context questions with
three independent generations each. The published checkpoint scores 74.00%,
QAD scores 73.00%, and BF16 scores 71.67%; paired evidence does not distinguish
the three complete configurations. The accompanying
[reproduction specification](glm-5.3-flash/aa-lcr-reproduction.md) fixes the
dataset, prompt, sampling, runtime, equality checker, and receipt validation.

The [Verifier-Backed Behavioral Fidelity report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md)
uses 224 deterministic tasks with executable answer keys and no language-model
judge. BF16 scores 93.11%, published NVFP4 scores 91.63%, and QAD step 1,750
scores 92.35% on the primary fractional metric. All paired one-point decisions
are inconclusive; the report separates qualified execution provenance from the
statistical power required to claim behavioral equivalence or improvement.

## Docker artifact

```text
voipmonitor/vllm:jovian-judgement-community-20260904-r25
voipmonitor/vllm@sha256:89376e9aa49442a90754662ca1bb281bffbeca29bb7393e6e8281506e5ac4804
```

The embedded source-lock SHA-256 is
`8d99c847855c32bb2348cd58f8a3a72f97df9c5fbeb38223f94a3bdfb590d9d0`.
The Docker digest fixes the runtime. Model repository names follow Hugging Face
`main` unless an optional revision variable is supplied.

The vLLM and B12X packages are byte-identical to the qualified R24 package
trees. The R25 package replaces the LMCache Python control plane with the
retrieve fast path specified below; native extensions are unchanged.

### R21 TP3 qualification overlay

The published R21 parent is
`voipmonitor/vllm@sha256:f096012c508f9bc12e8c4e617b8ed19da3a2cecb525e9479904e848730f0c8ac`.
It has embedded source-lock SHA-256
`e7adbbb9833b5cb1716bfc673a343b538b96672bad94d78753d1fd3c87940026`.
That parent does not contain the TP3 launcher or ported sources. The published
child was built from `Dockerfile.glm53-r21-tp3-overlay` at runtime commit
`01c67936a364009ff6b42e8bd10a01628d1e7078`. The child hard-pins the parent
digest, vLLM commit `e96b18dbb8c19230591e79e0ed056b12947b2ea1`, B12X commit
`6d47b10eddf408799796650baf3e802bd56bf844`, and exact target and DFlash2
revisions. Its canonical source-lock SHA-256 is
`21fd2d6ffa3e842ee656f780a8530cce0ffb6601dfa47a5138409247ec4df0d4`.

The pull-ready child is published with the immutable digest first and the
convenience tag second:

```text
infernix/vllm@sha256:e81f9399aa9fe800593cc8f646d8a2c7958e1938da50c5ae65effbe47d8604eb
infernix/vllm:glm53-r21-tp3-qualified-vllme96b18dbb8c1-b12x6d47b10eddf4-recipe01c67936a364
```

The convenience tag is not immutable; use the digest for reproducible pulls.
The qualified image config ID is
`sha256:dbdc64cb31b0c2bc1a0bbd5eaa4e5a91a1539333123547cc6de9b08c426bf6c1`.
It is the config ID, not the manifest digest or a pull reference.

That exact child passed startup and inference in ordinary, MTP3, and DFlash2
K7 modes. DFlash2 additionally passed an eight-request concurrent text smoke,
a red-image vision smoke, and exact 1,000,031-prompt-token natural-context retrieval
of `ORCHID-7319`. The same child passed a four-accelerator TP4 MTP3 regression
against the pinned target revision.
The packaged B12X profile suites passed 122 tests with one skip, and the exact
SM120 source gates passed all 11 regressions.

The fail-closed TP3 launcher policy in
[PR local-inference-lab/blackwell-llm-docker#30](https://github.com/local-inference-lab/blackwell-llm-docker/pull/30)
accepts only dense `CACHE_MODE=vram` under TP=3 and rejects `CACHE_MODE`
values `native` and `lmcache` with exit status 2 before service startup. TP3
accepts no caller CLI arguments: mode selection is environment-based, and
every qualified geometry, scheduler, backend, cache-layout, speculation,
collective, and B12X policy selector is either locked or required unset. The
qualified envelope includes TP3/EP3/DCP1, an 8,192-token scheduler budget,
eight sequences, full CUDA graphs at capture sizes 1/2/4/8/16, FP8 KV cache,
B12X PCIe tensor-parallel collectives, PyNCCL expert-parallel collectives,
and weights-sharded multimodal encoding. See the
[qualification receipt](../benchmarks/data/glm53-r21-tp3-20260904/qualification-receipt.json),
its three TP3 runtime proofs, and the adjacent TP4 regression proof. The
`GLM53_R17_TP3_RUNTIME_PROOF` label in the TP3 logs is a legacy marker API
retained by the R21 source overlay; its payload describes the R21 qualification.

## Runtime backends

| Operation | Selected implementation |
|---|---|
| Target sparse attention and C4 index selection | B12X |
| Target gated-delta-network prefill | FlashKDA by default; B12X KDA is explicitly selectable; the R21 TP3 overlay locks FlashKDA |
| Target gated-delta-network decode | B12X when eligible, with Triton fallback; the R21 TP3 overlay requires B12X |
| Target routed experts | B12X NVFP4 W4A4 for the published R25 TP4 image; the R21 TP3 child's automatic selection chose FlashInfer CUTLASS with EP3 |
| Target linear layers | B12X |
| Tensor-parallel all-reduce | B12X PCIe one-shot/two-shot first; PyNCCL outside the qualified B12X ranges; the R21 TP3 proof selected B12X PCIe one-shot |
| Expert-parallel collectives | PyNCCL for TP3 |
| MTP attention | B12X |
| MTP experts | Marlin for the published R25 TP4 image; Humming for the R21 TP3 child |
| DFlash2 MXFP8 linear and fused key/value projections | B12X |
| DFlash2 local attention | Graph-safe split-KV FlashAttention |
| Sampling | FlashInfer |
| External cache | LMCache DRAM L1 and native-filesystem L2 through asynchronous engine-driven pinned shared memory when selected; unsupported by the R21 TP3 overlay |

DeepGEMM and TileLang are installed dependencies but are not selected for the
target, MTP, or DFlash2 hot paths.

## Qualified performance

The measurements used physical GPUs 4 through 7 on one RTX PRO 6000 Blackwell
Workstation Edition host at stock clocks, PCIe Gen5 x16 links, tensor
parallelism of four, FP8 target KV cache, a 4,096-token scheduler budget, full
and piecewise decode graphs, exactly 16 NCCL channels, and a 2 MiB NCCL buffer.
The measured vLLM tree, B12X tree, and model launcher are byte-identical to the
R25 artifact. Each prefill value is a 30-second 32K-token measurement after a
30-second warmup. C1 and C8 are 30-second sustained context-zero decode cells.
Other serving workloads were active on the host, so the table gives
conservative rather than isolated peak throughput.

| DCP | Serving mode | 32K prefill tok/s | C1 output tok/s | C1 steps/s | C1 accepted/step | C8 output tok/s | C8 steps/s | C8 accepted/step |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | No speculation | 14,911 | 169.9 | — | — | 737.8 | — | — |
| 1 | MTP, depth 3 | 14,485 | 247.8 | 99.0 | 2.50 | 903.2 | 364.7 | 2.48 |
| 1 | DFlash2, depth 7 | 14,648 | 221.1 | 89.8 | 2.46 | 689.6 | 289.4 | 2.38 |
| 4 | No speculation | 13,368 | 151.8 | — | — | 667.5 | — | — |
| 4 | MTP, depth 3 | 12,912 | 227.0 | 88.8 | 2.56 | 830.9 | 331.2 | 2.51 |
| 4 | DFlash2, depth 7 | 13,181 | 206.7 | 81.8 | 2.53 | 630.8 | 267.7 | 2.36 |

Speculative output throughput varies with accepted length. Target steps per
second isolates target-model execution speed.

The published three-layer image additionally completed a package-level
full-CKV smoke on the same four stock-clock GPUs. DCP4 DFlash2 with
packed-NVFP4 cache and 1,024-token LMCache pages measured 12,394 prompt tok/s
and 81.1 verifier steps/s. The byte-identical R24 vLLM/B12X control measured
12,397 prompt tok/s and 81.45 verifier steps/s under matched conditions.

### Complete-KV prefill

With four decode-context ranks, every rank must select sparse-attention
candidates from the complete target KV sequence rather than its rank-local
quarter. The qualified implementation gathers the complete target KV view and
keeps recurrent-state cache ownership independent. A matched feature-isolation
test measured 12,999 prompt tok/s with complete-KV selection versus 9,858
prompt tok/s with rank-local selection, a 31.86% increase.

### Memory-weighted cache allocation

GLM-5.3 target attention and recurrent state have different bytes-per-request
costs. The runtime partitions compatible layers by their actual cost instead
of forcing equal layer counts into each shared pool. The deterministic search
is capped at eight cache groups, preserving bounded scheduler and connector
overhead.

For TP4/DCP4 without speculation, the weighted allocation admits 21,929,984
tokens instead of 20,873,216 under equal-count grouping: 1,056,768 additional
tokens, or 5.06%. A matched DCP1 DFlash2 A/B found no attributable execution
regression: 32K prefill was 15,151 versus 15,138 tok/s, C1 verifier throughput
was 101.6 versus 103.3 steps/s, and C8 verifier throughput was 332.1 versus
327.3 steps/s.

### Compute-share fairness

The scheduler measures model-execution time separately for prefill and decode
and targets 40% prefill share only while both classes are contending. A mixed
C8 DFlash2 decode and cold 32K-prefill run assigned 42.97% of measured
contended execution time to prefill. Forward passes are indivisible, so short
runs oscillate around the configured target.

Set `FAIRNESS_ENGINE=none` to disable fairness. Compute-share fairness requires
`PREFILL_SCHEDULE_INTERVAL=1`; the launcher rejects incompatible values rather
than silently changing scheduling behavior.

### FlashKDA numerical stability

The deterministic 16,384-token near-collinear BF16-key reproducer produces
zero non-finite output or recurrent-state elements with the packaged FlashKDA
extension. The extension replaced by this artifact produced 55,264 non-finite
output elements and 16,000 non-finite recurrent-state elements for identical
inputs. The qualified server also completed 24 long concurrent TP4/DCP4 MTP3
requests without a wrong or runaway result.

### Gated-delta-network prefill backend

B12X KDA remains qualified as an explicit backend. FlashKDA is the default.
The stable FlashKDA extension measured 102.98 microseconds at the packed TP4/C4
kernel shape versus 105.03 microseconds for the numerically unstable extension,
a 1.95% kernel-time reduction.

## Start the server

Select a serving mode and run the common command. The image already contains
the qualified B12X, full-and-piecewise CUDA graph, FlashInfer sampler,
16-channel NCCL, 2 MiB NCCL buffer, and one-thread OpenMP defaults. No
`CUDAGRAPH_MODE` override is required.

```bash
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260904-r25
GPU_DEVICES=0,1,2,3
PORT=8000
docker pull "$IMAGE"
```

Ordinary serving without speculative tokens:

```bash
NAME=jovian-judgement-nospec
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=0)
```

Three-token MTP:

```bash
NAME=jovian-judgement-mtp3
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=3)
```

DFlash2 with its trained seven-draft-token configuration:

```bash
NAME=jovian-judgement-dflash2
MODE_ARGS=(
  -e SPECULATOR=dflash2
  -e DFLASH_DEPTH=7
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2
)
```

Common GPU-cache command:

```bash
docker run -d \
  --name "$NAME" \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v jovian-judgement-runtime-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4 \
  -e CACHE_MODE=vram \
  -e KV_CACHE_QUANT=fp8_ds_mla \
  -e CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  -e PORT="$PORT" \
  -e TP=4 \
  -e DCP=1 \
  -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_SEQS=32 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=1 \
  -e FAIRNESS_ENGINE=compute_share \
  -e PREFILL_COMPUTE_SHARE=0.4 \
  -e GPU_MEMORY_UTILIZATION=0.93 \
  "${MODE_ARGS[@]}" \
  "$IMAGE"
```

For two or four decode-context ranks, replace `DCP=1` with `DCP=2` or `DCP=4`.
The launcher enables complete-KV gathering automatically when DCP is greater
than one. This applies to no speculation, MTP, and DFlash2.

To test the qualified B12X gated-delta-network prefill backend, add:

```bash
-e GLM53_KDA_PREFILL_BACKEND=b12x
```

The default `GLM53_KDA_PREFILL_BACKEND=flashkda` is faster or equal in the
qualified configurations.

### R21 TP3 no-build deployment

The pull-ready R21 child needs exactly three visible GPUs. Its strict launcher
locks the qualified runtime policy and accepts no caller CLI arguments. The
target and DFlash2 revisions below match the
[qualification receipt](../benchmarks/data/glm53-r21-tp3-20260904/qualification-receipt.json).
This copy-paste path starts ordinary serving by default:

```bash
IMAGE=infernix/vllm@sha256:e81f9399aa9fe800593cc8f646d8a2c7958e1938da50c5ae65effbe47d8604eb
GPU_DEVICES=0,1,2
PORT=8000
NAME=jovian-judgement-r21-tp3
MODE_ARGS=()

docker pull "$IMAGE"
docker run -d \
  --name "$NAME" \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v jovian-judgement-runtime-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4 \
  -e MODEL_REVISION=378ca54585c46542bad1f3cb3ed0d73ae51cdb62 \
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2 \
  -e DFLASH_MODEL_REVISION=aea0ac8a05624512ca9e106c09c16087da998426 \
  -e TP=3 \
  -e PORT="$PORT" \
  "${MODE_ARGS[@]}" \
  "$IMAGE"
```

For MTP3, set `MODE_ARGS=(-e MTP_DEPTH=3)`. For DFlash2 K7, set
`MODE_ARGS=(-e SPECULATOR=dflash2 -e DFLASH_DEPTH=7)`. The immutable digest
above is the release locator; the convenience tag in the R21 overlay section
can move.

### Cache page geometry

The launcher owns cache page geometry; normal deployments should not pass a
page-size argument. GPU-only and native-offload modes use 2,048-token target
and recurrent pages. LMCache derives per-rank pages from its 4,096-token object
size and the selected DCP value. A DFlash2 sliding window smaller than one
engine page is transferred as one complete engine page so no live cache bytes
are omitted.

The 2,048-token GPU page increases usable KV capacity without a measurable
prefill or decode regression. It does not change the public vLLM attention
block size, which remains 256 tokens.

## Native DRAM offload

Use the common server command with these cache settings:

```bash
-e CACHE_MODE=native
-e NATIVE_KV_OFFLOADING_SIZE_GB=64
```

The launcher enables the shareable cuMem allocator required by the native
offload backend.

## LMCache DRAM and filesystem storage

LMCache uses a CPU-only sidecar process in the same container. GPU gather and
scatter run in the existing vLLM workers; the sidecar receives an empty
`CUDA_VISIBLE_DEVICES` and creates no additional CUDA context. Asynchronous
engine-driven transfer through pinned shared memory is selected automatically.
The cache stores complete 4,096-token objects in DRAM and optionally in a
mounted filesystem. Use a private shared-memory allocation of at least 96 GiB;
128 GiB is the qualified setting.

Replace the cache settings, shared-memory size, and GPU-memory-utilization line
in the common command with:

```bash
--shm-size 128g
-v jovian-judgement-lmcache-l2:/lmcache-l2
-e CACHE_MODE=lmcache
-e KV_CACHE_QUANT=fp8_ds_mla
-e LMCACHE_CHUNK_SIZE=4096
-e LMCACHE_TARGET_TOKEN_BUDGET=4096
-e LMCACHE_L1_SIZE_GB=64
-e LMCACHE_L2_ENABLED=1
-e LMCACHE_L2_ROOT=/lmcache-l2
-e GPU_MEMORY_UTILIZATION=0.95
```

`KV_CACHE_QUANT=nvfp4_ds_mla` selects the qualified packed-NVFP4 target cache
instead. The filesystem namespace includes the target and draft revisions,
cache format, parallelism, DCP gathering policy, speculation mode, and object
size, preventing incompatible cache objects from being reused.

Qualification covered cold compute, vLLM automatic prefix reuse, LMCache DRAM
restore, and filesystem restore for FP8 and NVFP4. External bytes were compared
on every tensor-parallel rank across target attention, recurrent state, and
DFlash sliding attention. Five matched one-million-token filesystem replays
restored 999,424 tokens, recomputed the 576-token suffix, and completed in
1.028 to 1.048 seconds; the median was 1.041 seconds. The R24 control median was
1.274 seconds, so compact lookup-session references reduced replay latency by
18.3%. The exact published image also passed a full-process restart restore in
1.215 seconds with DCP4 full-CKV enabled and zero local-prefix-cache tokens.

The retrieve path reads each reserved shared-memory tensor view once. It also
references chunk hashes retained by the active lookup session instead of
serializing and decoding the complete token sequence for prepare and commit on
every worker. Capability negotiation retains complete retrieve keys when the
server does not advertise session-reference support. Failure cleanup is
idempotent: a failed prepare releases one lookup reader, while unregister owns
pending-read cleanup before a rejected late commit.

The matched TP4/DCP4 performance test used packed-NVFP4 KV cache, 1,024-token
per-rank LMCache pages, stock clocks, and the same vLLM and B12X package trees
on both arms. Decode comparisons use target steps per second for speculative
modes so stochastic acceptance does not masquerade as an execution change.

| Serving mode | GPU-only 32K prefill tok/s | LMCache 32K prefill tok/s | Prefill change | GPU-only C1 | LMCache C1 | C1 change |
|---|---:|---:|---:|---:|---:|---:|
| No speculation | 12,647 | 12,569 | -0.62% | 150.94 tok/s | 150.79 tok/s | -0.10% |
| MTP, depth 3 | 12,303 | 12,185 | -0.96% | 93.3 steps/s | 93.2 steps/s | -0.11% |
| DFlash2, depth 7 | 12,476 | 12,397 | -0.63% | 80.88 steps/s | 81.45 steps/s | +0.70% |

The measured cold-prefill overhead remains below 1%. DFlash2 C1 output was
202.03 tok/s without LMCache and 204.11 tok/s with LMCache; the +1.03%
difference is consistent with its small accepted-length variation rather than
a cache execution cost.

## Source and review contract

| Component | Qualified source |
|---|---|
| vLLM | [R24 integration source](https://github.com/local-inference-lab/vllm/tree/integration/glm53-r23-lmcache-parser-20260904); commit `d49385468458cf97dff0fc8d9c8863f8082abf4f`; tree `e2c687bb823dbe1b37c3d9f9742a0ae54419fdb0`; package tree `17acb470467c1a6d4b318a3c4a0960794fb4da6a` |
| FlashKDA | commit `3b225bf26bb8e218928a1fe14751cb48cf31d11b`; extension SHA-256 `16aece5ffb83c2dfb0355758bbbc9d6e0ea50a2cfc36ecee4936607d445aba0a` |
| B12X | [R24 integration source](https://github.com/local-inference-lab/b12x/tree/integration/glm53-r23-lmcache-parser-20260904); commit `e3d0ae067f607538e3709ac3c30c7042276c6f88`; tree `d93cd222b027ed1df7f7df221007196994c80354`; package tree `fc977aa2b732935cd0f70c365d7f767b449d21da` |
| LMCache | [PR 43](https://github.com/local-inference-lab/LMCache/pull/43) preserves complete DFlash pages; [PR 45](https://github.com/local-inference-lab/LMCache/pull/45) adds stride-correct asynchronous engine-driven hybrid stores; [PR 47](https://github.com/local-inference-lab/LMCache/pull/47) reuses retrieve tensor views; [PR 48](https://github.com/local-inference-lab/LMCache/pull/48) adds validated lookup-session references; [source mirror](https://github.com/local-inference-lab/LMCache/tree/artifact/jovian-judgement-community-20260904-r25-lmcache-source); commit `cf52fc51418c6b0146e1fcea0690c25ef4e947a0`; tree `f1f38f35c3e4975810d1e1c03d4fb8f845bf5cb3`; package tree `9cf07ca20e1dc7d11bb14e460662faa563f1c10d` |

### R21 TP3 qualification source lock

| Component | Published R21 source | Hardware-qualified TP3 review head |
|---|---|---|
| vLLM | branch `artifact/jovian-judgement-community-20260904-r21-source`; commit `f2d77086163e899f87f54a59af216d18ffa3a2b7`; tree `95bff0df5f40df443884c282de6fbda1b1fdb8d6`; package tree `4fbb1c257ac59e5e68450655ad4061d2c8a05e5c` | commit `e96b18dbb8c19230591e79e0ed056b12947b2ea1`; tree `31e73a43eb8a03e932f03c51341df2c73c60f3d4` |
| B12X | commit `1e59a1fd09f782d302b1068b15c8a0bd66103894`; tree `f322c804eec1c58a63bd4fe6e7901a95a678a575`; package tree `aaa5f189acae0206d886553421f6e9044f4c458a` | commit `6d47b10eddf408799796650baf3e802bd56bf844`; tree `afdd4b4cc589fddb079f1661d91e932f9d99b8c5` |
| Runtime policy | image recipe label `10901bcc31e7596d9b75c976b460778ac4bbe62f` | commit `01c67936a364009ff6b42e8bd10a01628d1e7078`; tree `36fbabcbf2f16603341787e7e7c9a58cbf24783b` |
| LMCache | commit `aefe3ab701ab7a835532e701be89f5055b13ec0f`; tree `683ab2c165a9aa0e2d1a1ab757af4a8b193688c5`; package tree `976a97f22c0497f34db089dc5f02a713dd0b5888` | unchanged |

The [vLLM merge checklist](https://github.com/local-inference-lab/vllm/issues/590)
lists each open pull request, dependency, resulting behavior, attribution, and
qualification result. The published R24 image embeds its source contract at
`/opt/glm53-flash/source.lock`.
