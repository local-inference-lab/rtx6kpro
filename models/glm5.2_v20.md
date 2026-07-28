# GLM-5.2 v20: Gilded Gnosis DCP Release

v20 is the tested successor to [v19](glm5.2_v19.md). It keeps the same GLM-5.2,
NF3, MXFP4, DCP, MTP, and InstantTensor launch contract while updating the
canonical GG/SparkInfer stack, fixing two release blockers, and adding the
measured DCP prefill topology:

- DCP outputs now preserve a cuBLAS-safe physical head-major layout without a
  hot-path clone or tail-padding reservation;
- virtual TP6 accepts partial pitched DCP workspaces and correctly plans the
  N128-padded W4A8-MX scratch extent;
- exact owner top-k merge, partial indexer replication, and a bounded
  one-layer CKV prefetch improve DCP prefill without lossy transport;
- a pre-model lossless PCIe probe now measures DMA, query-split, and CKV
  overlap crossovers for the selected GPU/NUMA topology instead of assuming
  that the development host's overlap policy is portable;
- the launcher preserves an intentional `CUDA_VISIBLE_DEVICES` order when
  Compose leaves `GPUS` empty, allows 600 seconds for a cold probe, and
  terminates the complete probe process group if calibration times out;
- the optional `glm52-exl3` profile builds the EXL3 extension from its pinned
  public source and exposes the Trellis MoE path without a binary ABI shim.

Historical comparison data remains on [v18](glm5.2_v18.md), while the DCP
optimization background remains on [v19](glm5.2_v19.md). This page is
self-contained for building, starting, operating, and validating v20; older
pages are provenance, not required setup instructions.

Canonical source merging and the required post-merge rebuild are tracked in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).
The image below is the exact measured release candidate. Open source deltas
remain independently reviewable; already-merged fixes are pinned through the
stated GG and SparkInfer base commits.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r5
Docker manifest: sha256:473e4c5a6a3795ee48a8f50493f9628f7c53f449d5a512f5cc04add77693044c
Local image ID: sha256:178fe7439b3a83f6dc78259dc36bfbedfa9f85f1d79ef1b5551eb4940cf32e42
```

This supersedes all earlier v20 candidates. The registry manifest is the exact
24,930,661,390-byte local image used for the final helper and runtime-contract
gates below; there was no rebuild between those gates and push. Both source trees were
composed from clean public bases plus exact public PR heads. The generated
integration patches and lockfiles are public, immutable release artifacts;
there is no private overlay or source bind mount.

`si` identifies SparkInfer, the renamed B12X project. Legacy B12X environment
variable names remain accepted for compatibility.

Pinned source stack:

| Component | Ref / commit |
|---|---|
| Canonical GG base | `local-inference-lab/vllm dev/gilded-gnosis` @ `4247d6765398fd42de3c108a8d991b2634fe88d1` |
| Composed vLLM tree | `936ed4829ed6b6a34b9052a7a2614333ee3b2623` |
| SparkInfer base | `local-inference-lab/sparkinfer master` @ `f9be2724953a5b412d19c20482aeb0a64fbd5d2a` |
| Composed SparkInfer tree | `f532ec965a70b710ba45e6f751fe5d7135001108` |
| EXL3 extension | `brandonmmusic-max/exllamav3 a1-retile-sm120` @ `704aefd743b390af4bd0fb429d1906f9b964c7d8` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| CUTLASS C++ / DSL | `e6233cbac5d7c7a865c19c91cd684ceece19513c` / `4.6.0` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| NCCL | local-inference `2.30.4` |
| PyTorch / CUDA / loaded cuDNN | `2.12.0+cu132` / `13.2.1` / `9.20.0.48` |
| CUDA system-base cuDNN packages | `9.22.0.52` |
| Launcher source | `local-inference-lab/blackwell-llm-docker` @ `a2129e983b07fbfaa5b872a1a0b25a07c3f01876` |
| Validated build execution | `local-inference-lab/blackwell-llm-docker` @ `f1abd5c3ab38832b52625be9fe112801906e51ca` |
| Immutable reproduction recipe | `local-inference-lab/blackwell-llm-docker` @ `1e4d0c7e7981046f65926235f02824d795691e57` |

The image uses no `VLLM_PATCH_URL`, private source overlay, or source bind
mount. It does contain generated `VLLM_PATCH_FILE` and SparkInfer patch
artifacts derived solely from the public manifests. Image labels expose both
base commits, every PR head, both result trees, patch and lock hashes, and a
cache fingerprint derived from the pinned sources.

## Build It Exactly

The canonical build entry point is
[`build-gilded-gnosis-v20-final-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/1e4d0c7e7981046f65926235f02824d795691e57/build-gilded-gnosis-v20-final-cu132.sh).
The explicit reproduction mode uses archived, hash-verified locks and patches,
then verifies that applying them to the pinned bases produces the exact trees
above. It validates runtime symbols, helper contracts, and image labels before
allowing an optional push.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 1e4d0c7e7981046f65926235f02824d795691e57
VLLM_RELEASE_COMPOSITION=reproduce-r5 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

For a new release candidate, omit `VLLM_RELEASE_COMPOSITION`. The default
always resolves the current clean GG and SparkInfer bases, composes the exact
versioned PR manifests, and fails if either base or any PR head moves during
the build. The review for this composer, archived r5 source artifacts, and
their tests is [blackwell-llm-docker #7](https://github.com/local-inference-lab/blackwell-llm-docker/pull/7).

The build deliberately excludes the separate weight-lifetime experiments in
vLLM PR #154, vLLM PR #157, and SparkInfer PR #62. It also excludes the
experimental sparse-CKV decode stack in vLLM PRs #159-#161 and SparkInfer PRs
#64-#65. It also excludes the later `bounded_compat` commits on build PR #5;
that selector policy was not part of this candidate or its validation.

## Source Changes

The cuBLAS/Xid correction is already in the pinned GG base through
[vLLM PR #147](https://github.com/local-inference-lab/vllm/pull/147) and
[SparkInfer PR #54](https://github.com/local-inference-lab/sparkinfer/pull/54).
The pinned bases also already contain vLLM #177, #178, and #180 and
SparkInfer #79 and #85. The clean r5 manifests apply only these exact PR heads
on top of those bases:

| Project | Review | Purpose |
|---|---|---|
| vLLM | [#145](https://github.com/local-inference-lab/vllm/pull/145) | Calibrated NVFP4 MLA KV outer-scale wiring. |
| vLLM | [#172](https://github.com/local-inference-lab/vllm/pull/172) | Profile persistent kernel resources before allocating KV cache. |
| vLLM | [#175](https://github.com/local-inference-lab/vllm/pull/175) | Split sparse prefill queries and reduce gathered result traffic. |
| vLLM | [#179](https://github.com/local-inference-lab/vllm/pull/179) | Add partial replicated-indexer topology and mixed target/draft grouping. |
| vLLM | [#184](https://github.com/local-inference-lab/vllm/pull/184) | Dispatch lossless BF16 PCIe DMA above a measured byte crossover. |
| vLLM | [#185](https://github.com/local-inference-lab/vllm/pull/185) | Gate DCP query split by a measured context crossover. |
| vLLM | [#190](https://github.com/local-inference-lab/vllm/pull/190) | Add the EXL3 rank-sliced MoE integration and prefill planner. |
| SparkInfer | [#81](https://github.com/local-inference-lab/sparkinfer/pull/81) | Measure lossless collective/overlap crossovers and derive a cached serving policy. |
| SparkInfer | [#49](https://github.com/local-inference-lab/sparkinfer/pull/49) | Add the planned Trellis execution path used by EXL3. |

The release build itself does not merge canonical branches and does not consume
a precomposed integration branch. It generates both build-time patches from
the clean bases and manifests, verifies their result trees, and archives the
exact r5 artifacts. SparkInfer #76 is closed and is not an additional release
delta: the resulting PCIe output-lifetime implementation matches the pinned
master. There is no runtime source patching or source bind mount.

### Canonical Merge Status

[Issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33) is the
authoritative ordered merge checklist. The image pins the exact base commits
and additional PR heads above rather than following moving source refs.

PR #145 is intentionally present in the image but is not requested for merge
yet. The exact SparkInfer candidate-owner transport in
[SparkInfer #79](https://github.com/local-inference-lab/sparkinfer/pull/79)
and the runtime stride correction in
[SparkInfer #85](https://github.com/local-inference-lab/sparkinfer/pull/85)
are both included. The helper keeps exact v20 top-k selection; the later
`bounded_compat` experiment from the older build PR #5 is deliberately absent.
The broader DCP design and rejected experiments are recorded in
[research issue #35](https://github.com/local-inference-lab/rtx6kpro/issues/35).
The subsequent remote selected-record and query-sharding POC is archived with
its exact source branches, tests, and measurements in
[research issue #36](https://github.com/local-inference-lab/rtx6kpro/issues/36).
Those paths were correct but slower than the retained local-CKV design; they
do not change this image, its defaults, or the canonical merge checklist.

## Start The Server

The helper is inside the image, so users do not need to download a launch
script. Docker with NVIDIA Container Toolkit, host IPC, and at least four
Blackwell GPUs is required. Pull the immutable image first:

```bash
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r5
```

Save the following as `compose.yml`. Bare environment entries pass a host
variable only when it is set; otherwise the helper chooses the correct default
for `MODEL_FAMILY`. The two explicit memory entries raise the recommended
standard TP8 service to a 262k context and a validated 0.96 memory budget. The
TP6 recipe below overrides the memory budget with its separately validated
limit.

```yaml
services:
  glm52:
    image: voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r5
    entrypoint: ["/usr/local/bin/serve-gilded-gnosis.sh"]
    network_mode: host
    ipc: host
    privileged: true
    init: true
    shm_size: 32gb
    gpus: all
    ulimits:
      memlock: -1
      stack: 67108864
      nofile:
        soft: 1048576
        hard: 1048576
    environment:
      - MODEL_FAMILY=${MODEL_FAMILY:-glm52}
      - MODEL
      - MODEL_REVISION
      - SERVED_MODEL_NAME
      - GPUS
      - CUDA_VISIBLE_DEVICES
      - PORT
      - TP
      - DCP
      - DCP_BACKEND
      - DCP_A2A_MAX_TOKENS
      - DCP_A2A_LARGE_BACKEND
      - DCP_QUERY_SPLIT
      - DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS
      - DCP_CKV_GATHER
      - DCP_TOPK_OWNER_MERGE
      - DCP_INDEXER_SHARDS
      - DCP_CKV_PREFETCH_DEPTH
      - DCP_CKV_PREFETCH_WORKSPACE_MIB
      - DCP_CKV_PREFETCH_TOPOLOGY
      - DCP_PREFILL_WORKSPACE
      - PCIE_CALIBRATION
      - PCIE_CALIBRATION_ONLY
      - PCIE_CALIBRATION_TIMEOUT
      - PCIE_CALIBRATION_CACHE_DIR
      - PCIE_DMA_MIN_BYTES
      - MTP
      - MAX_NUM_SEQS
      - GRAPH
      - MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
      - MAX_BATCHED_TOKENS
      - GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.96}
      - MOE_MODE
      - MOE_BACKEND
      - LINEAR_BACKEND
      - QUANTIZATION
      - ONLINE_QUANT
      - QUANTIZATION_CONFIG_JSON
      - KV_CACHE_DTYPE
      - F8_DMA
      - B12X_PCIE_DMA
      - NF3_GRID188
      - LOAD_FORMAT
      - INSTANTTENSOR_BACKEND
      - PYTORCH_CUDA_ALLOC_CONF
      - DRY_RUN
    volumes:
      - ${HF_CACHE:-/root/.cache/huggingface}:/root/.cache/huggingface
      - ${MODEL_ROOT:-/root/models}:/root/models:ro
      - ${JIT_CACHE:-./cache/glm52-v20}:/cache
      - ${CONTAINER_TMP:-./cache/glm52-v20/tmp}:/container-tmp
```

The image helper and Compose contract both use `MAX_MODEL_LEN=262144` and
`GPU_MEMORY_UTILIZATION=0.96` for standard TP8. Virtual TP6 remains separately
validated at `128000` and `0.95`.

### Start, Inspect, And Stop

The standard model preset is Luke NVFP4, TP8/DCP1, native A4, MTP off. The
highest-accuracy standard launch changes only `MOE_MODE` to A16:

```bash
MOE_MODE=a16 docker compose up -d
docker compose logs -f glm52
```

Wait for the health endpoint before sending traffic:

```bash
curl -fsS http://127.0.0.1:${PORT:-8000}/health
curl -fsS http://127.0.0.1:${PORT:-8000}/v1/models | jq .
```

The first start compiles kernels. Reuse the same `JIT_CACHE` for the same image
and configuration family; do not benchmark while this or another model is
still loading. Stop the service without deleting either model or JIT cache:

```bash
docker compose down
```

Inspect the fully expanded environment and `vllm serve` command without loading
weights:

```bash
DRY_RUN=1 MOE_MODE=a16 docker compose run --rm --no-deps glm52
```

### Common Launch Recipes

These commands use the same Compose file. Variables not shown remain owned by
the image helper.

```bash
# Luke NVFP4, highest-accuracy routed-expert mode, no speculation.
MOE_MODE=a16 MTP=0 TP=8 DCP=1 docker compose up -d

# Luke NVFP4, native A4 expert activations with three-token MTP.
MOE_MODE=a4 MTP=3 TP=8 DCP=1 docker compose up -d

# Luke NVFP4 with eligible BF16 dense linears converted online to MXFP8.
MOE_MODE=a16 ONLINE_QUANT=mxfp8 MTP=0 TP=8 DCP=1 \
  QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}' \
  docker compose up -d

# AMD MXFP4 experts, forced A8 path, native BF16 dense linears.
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
  SERVED_MODEL_NAME=GLM-5.2-BF16-AMDMXFP4experts \
  QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental \
  ONLINE_QUANT=none MTP=0 TP=8 DCP=1 docker compose up -d

# The same AMD checkpoint with online MXFP8 dense linears.
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
  SERVED_MODEL_NAME=GLM-5.2-BF16-AMDMXFP4experts \
  QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental \
  ONLINE_QUANT=mxfp8 MTP=0 TP=8 DCP=1 \
  QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}' \
  docker compose up -d

# Virtual TP6/DCP3 validation profile for the AMD checkpoint.
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
  SERVED_MODEL_NAME=GLM-5.2-BF16-AMDMXFP4experts \
  QUANTIZATION=mxfp4 MOE_MODE=force-a8-experimental \
  TP=6 DCP=3 MTP=3 MAX_NUM_SEQS=16 GRAPH=64 \
  MAX_MODEL_LEN=128000 MAX_BATCHED_TOKENS=4096 \
  GPU_MEMORY_UTILIZATION=0.95 docker compose up -d

# TP8/DCP4 full-CKV prefill profile for Luke A16, MTP off.
MOE_MODE=a16 TP=8 DCP=4 MTP=0 MAX_NUM_SEQS=32 GRAPH=128 \
  MAX_BATCHED_TOKENS=8192 docker compose up -d

# NF3 hybrid. MODEL_FAMILY selects its TP4/A16/NVFP4-KV defaults.
MODEL_FAMILY=glm52-hybrid DCP=4 MTP=3 docker compose up -d

# Community EXL3 profile. The helper pins its tested checkpoint revision and
# TP4/DCP4 defaults; full model performance validation remains community-run.
MODEL_FAMILY=glm52-exl3 docker compose up -d
```

For a local checkpoint, `MODEL` must use its in-container path below
`/root/models`. For another Hugging Face repository, set both `MODEL` and its
immutable `MODEL_REVISION`; the standard preset otherwise pins Luke's tested
revision `8a1f4a13204acf2b7ac840375efaed64c231c522`.

### Stable Controls

| Variable | Default and meaning |
|---|---|
| `MODEL_FAMILY` | `glm52`; use `glm52-hybrid` for TP4 NF3 or `glm52-exl3` for the source-built EXL3/Trellis profile. The unified image also accepts `ds4`. |
| `MODEL` | Luke NVFP4 for `glm52`; the madeby561 NF3 checkpoint for `glm52-hybrid`; the pinned Brandon EXL3 checkpoint for `glm52-exl3`; local paths are supported. |
| `MODEL_REVISION` | Immutable tested Hugging Face revision. Set the correct revision when changing a remote `MODEL`. |
| `SERVED_MODEL_NAME` | API model name; defaults to the selected checkpoint preset. |
| `GPUS` | Ordered physical GPU list. Resolution is explicit `GPUS`, then existing `CUDA_VISIBLE_DEVICES`, then the preset default. Standard default is `0,1,2,3,4,5,6,7`; NF3 and EXL3 default to `0,1,2,3`. |
| `PORT` | `8000`. Host networking exposes it directly. |
| `TP` | Standard `8`, virtual-sharded `6`, or NF3/EXL3 `4`. |
| `DCP` | Decode context parallel size. `1` disables DCP communication; validated values are topology-dependent. |
| `MTP` | `0` disables speculation. `3` is the principal validated speculative mode; the helper accepts an integer token count. |
| `MAX_NUM_SEQS` | Standard `64`; scheduler concurrency and the input to automatic graph sizing. |
| `GRAPH` | When unset, standard GLM uses `4 * MAX_NUM_SEQS`; the NF3 preset uses `64`. |
| `MAX_MODEL_LEN` | Recommended standard and NF3 default: `262144`. TP6 remains `128000`. Raise only within the KV capacity reported at startup. |
| `MAX_BATCHED_TOKENS` | Standard `8192`; NF3 `2048`. The validated virtual-TP6 profile uses `4096`. |
| `GPU_MEMORY_UTILIZATION` | Recommended TP8 and NF3 default: `0.96`; TP6 at most `0.95`. TP8 `0.98` boots but is unsafe for long-prefill runtime allocations. |
| `MOE_MODE` | `a4`, `a16`, or `force-a8-experimental`. |
| `ONLINE_QUANT` | `none`, `mxfp8`, `fp8`, `nf3-mxfp8`, or `custom`. |
| `QUANTIZATION_CONFIG_JSON` | Explicit online quantization policy; overrides the helper preset. |
| `KV_CACHE_DTYPE` | Standard `fp8`; NF3 uses `nvfp4_ds_mla`. |
| `F8_DMA` | Default `0` (lossless BF16 wire). `ag`, `ring`, `a2a`, `i8*`, and `mx*` are explicit compressed-wire experiments and are never auto-selected. |
| `PCIE_CALIBRATION` | `auto` uses a matching cached result or measures before model loading; `force` remeasures; `off` uses the conservative static/topology policy. |
| `PCIE_CALIBRATION_ONLY` | `1` prints the effective policy and exits without loading the model. |
| `PCIE_CALIBRATION_TIMEOUT` | Cold-probe limit in seconds; default `600`. A timeout terminates `torchrun` and every probe worker before serving can start. |
| `PCIE_CALIBRATION_CACHE_DIR` | Defaults below the active fingerprinted XDG cache, normally `/cache/jit/<fingerprint>/pcie-calibration`. |
| `PCIE_DMA_MIN_BYTES` | `auto`, `off`, or an explicit byte/KiB/MiB threshold for lossless BF16 PCIe DMA dispatch. |
| `DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS` | `auto` uses the measured crossover; an integer is an explicit minimum context. |
| `DCP_CKV_GATHER_MAX_TOKENS` | `140000`; maximum pure-prefill size eligible for transient full-CKV gather. Raise explicitly for longer prefills, accepting the documented workspace cost. |

Advanced A/B controls are `DCP_QUERY_SPLIT`, `DCP_CKV_GATHER`,
`DCP_TOPK_OWNER_MERGE`, `DCP_INDEXER_SHARDS`, `DCP_CKV_PREFETCH_DEPTH`,
`DCP_CKV_PREFETCH_WORKSPACE_MIB`, `DCP_CKV_PREFETCH_TOPOLOGY`, and
`DCP_PREFILL_WORKSPACE`. Keep them on `auto` or their defaults for published
results. An explicit low-level `VLLM_*` value also wins over calibration.
`B12X_PCIE_DMA=1`,
`DCP_A2A_MAX_TOKENS=64` (`16` for NF3), and
`DCP_A2A_LARGE_BACKEND=ag_rs` remain transport defaults. Backend overrides
such as `MOE_BACKEND` and `LINEAR_BACKEND` are diagnostic controls, not
separate release modes.

The 262k/0.96 standard memory pair was validated on the exact v20 image with
Luke A16, MTP3, seq=64, graph=256, batch=8,192, FP8 KV, and no online quant.
Each topology processed a 240,041-token prompt followed by 512 decode tokens:

| Topology | GMU | KV tokens | Max concurrency at 262,144 | 240k + decode |
|---|---:|---:|---:|---|
| TP8 / DCP1 | `0.96` | 603,456 | 2.30x | pass; server remained healthy |
| TP8 / DCP4 | `0.96` | 2,285,824 | 8.72x | pass; query-split/full-CKV active |

Do not raise the generic default to `0.98`. That value booted TP8/DCP1 and
reported 641,088 KV tokens, but the same 240k request OOMed when an unprofiled
Inductor buffer requested another 64 MiB with only 66.38 MiB physically free.
This is why successful startup and reported KV capacity alone are insufficient
for selecting the serving memory budget.

### Checkpoint And Quantization Modes

| Checkpoint | `QUANTIZATION` | `MOE_MODE` | Supported tested online mode |
|---|---|---|---|
| `lukealonso/GLM-5.2-NVFP4` | `modelopt_fp4` | `a4` or `a16` | `none` or `mxfp8` |
| `festr2/GLM-5.2-BF16-AMDMXFP4experts` | `mxfp4` | `force-a8-experimental` | `none`, `mxfp8`, or `fp8` |
| `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` | `nvfp4_nf3_hybrid` | `a16` | `nf3-mxfp8` |
| `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` | `exl3` | `a16` / Trellis | none |

For Luke NVFP4, A4 and A16 select the routed-expert activation path; they do
not rewrite the NVFP4 checkpoint weights. A16 uses BF16 expert activations and
is the highest-accuracy tested mode. Force-A8 selects MXFP4 expert W4A8 and
applies to the AMD checkpoint, not Luke NVFP4. Generic online MXFP8 converts
eligible BF16 dense linears and does not rewrite existing NVFP4/MXFP4 routed
expert tensors.

With `MTP>0`, the helper creates a same-checkpoint MTP draft using the same MoE
backend and probabilistic draft sampling. The target and draft share the
virtual 66-head layout at TP6. Acceptance must be read from the server log for
the exact measurement window; the client acceptance field is not the release
source of truth.

### DCP Dispatch

`auto` is a launcher decision, not a value passed into vLLM. The helper now
combines two layers:

1. a conservative topology/profile policy chooses eligible DCP mechanisms;
2. a pre-model lossless probe measures whether DMA, query split, and CKV
   overlap actually repay their overhead on the selected machine.

The helper controls map to these runtime variables:

```text
DCP_QUERY_SPLIT  -> VLLM_DCP_QUERY_SPLIT
DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS
                 -> VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS
DCP_CKV_GATHER   -> VLLM_B12X_MLA_CKV_GATHER
DCP_CKV_GATHER_MAX_TOKENS
                 -> VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
DCP_TOPK_OWNER_MERGE -> VLLM_DCP_TOPK_OWNER_MERGE
DCP_INDEXER_SHARDS   -> VLLM_DCP_INDEXER_SHARDS
DCP_CKV_PREFETCH_DEPTH -> VLLM_B12X_MLA_CKV_PREFETCH_DEPTH
PCIE_DMA_MIN_BYTES -> VLLM_PCIE_DMA_MIN_BYTES
```

An explicit helper value bypasses the decision independently for that feature.
`DCP_INDEXER_SHARDS` and `DCP_CKV_PREFETCH_DEPTH` also accept non-negative
integers. The static eligibility mapping is:

| TP / DCP | Query split | Full CKV | Owner merge | Indexer shards | Prefetch depth |
|---|---:|---:|---:|---:|---:|
| TP8 / DCP1 | eligible | off | off | `0` | `0` |
| TP8 / DCP2 | eligible | on | on | `0` | measured `0/1` |
| TP8 / DCP4 | eligible | on | on | `2` | measured `0/1` |
| TP8 / DCP8 | eligible | on | on | `4` | measured `0/1` |
| TP4 / DCP1 | eligible | off | off | `0` | `0` |
| TP4 / DCP2, DCP4 | eligible | on | on | `0` | measured `0/1` |
| virtual TP6 / DCP1 | off | off | off | `0` | `0` |
| virtual TP6 / DCP2, DCP3, DCP6 | off | off | on | `0` | `0` |

`DCP_INDEXER_SHARDS=0` means the ordinary fully sharded indexer. At TP8/DCP4,
`2` creates a measured partial `2x2` topology; at TP8/DCP8, `4` creates `2x4`.
The CKV cache remains sharded by the full DCP size. The query-split flag at
DCP1 does not create inter-rank DCP traffic.

#### Full-CKV gather capacity

Full-CKV gather is a pure-prefill optimization. It gathers each layer's
compressed CKV once so every DCP rank can run its local attention heads over
the complete context. It replaces the more communication-heavy fallback chain
for that prefill. It does not affect DCP1, decode, mixed prefill/decode batches,
CUDA graph capture, or virtual TP6. The validated automatic geometries are
TP4/DCP2,4 and TP8/DCP2,4,8.

The source default is 524,288 tokens. An older external EXL3 recipe overrode
it to 16,384, which silently forced 64k and 128k prefills onto the fallback.
The release helper now sets and prints an explicit balanced default of 140,000:
large enough for the published 8k/64k/128k matrix while reserving much less
workspace than the source default. This preserves the historical correctly
configured v20 throughput; it is not a new 40-90% gain over that baseline.

Matched TP8/MTP0/A4/FP8-KV A/B runs on the r5 source stack demonstrate the
regression prevented by the explicit policy. Values are client prefill tok/s:

| DCP | Capacity | 8k | 64k | 128k | Global KV budget |
|---:|---:|---:|---:|---:|---:|
| 2 | 16,384 | 5,874 | 5,275 | 5,182 | 1,250,176 |
| 2 | 140,000 | 5,878 | 5,828 | 5,561 | 1,245,696 |
| 4 | 16,384 | 5,884 | 4,045 | 4,170 | 2,377,984 |
| 4 | 140,000 | 5,875 | 5,891 | 5,654 | 2,370,816 |
| 8 | 16,384 | 5,607 | 3,066 | 2,877 | 4,699,136 |
| 8 | 140,000 | 5,614 | 5,644 | 5,463 | 4,686,336 |

The 16,384 rows are deliberately forced negative controls, not the historical
v20 baseline. Correctly configured TP8/DCP4 had already measured about
5.6-5.85k tok/s before r5; the explicit 140k helper default prevents an old
external override from silently losing that established fast path.

At prefetch depth 0 and one execution lane, the 140k FP8-KV workspace is
approximately 131.4 MiB/GPU for DCP2, 109.5 MiB for DCP4, and 98.7 MiB for
DCP8. MTP and CKV prefetch may require additional lanes, so the startup log is
the authoritative reservation. To retain full gather for a 400k campaign,
set `DCP_CKV_GATHER_MAX_TOKENS` above that exact prompt size and re-check the
reported KV budget. Setting only the low-level `VLLM_*` alias is also accepted,
but the helper-facing name is preferred.

#### Lossless calibration

For supported TP4/TP8 configurations, calibration runs before the first model
load when at least one relevant control is still `auto`. It uses the same
SparkInfer/NCCL environment and actual selected GPU order as the eventual
server. The result is cached under the active fingerprinted XDG cache, normally
`/cache/jit/<fingerprint>/pcie-calibration`. The calibration fingerprint itself
includes GPU order and topology, TP/DCP/indexer geometry, CPU affinity and NUMA
placement, image/software/probe revisions, NCCL configuration, and relevant
environment overrides.

The probe independently determines:

- the smallest byte size at which **lossless BF16** SparkInfer DMA beats NCCL,
  or `off` if it never wins the measured ladder;
- whether query split wins and the first measured context crossover;
- whether one-layer CKV prefetch overlap wins over synchronous full gather.

On this host's adjacent TP8/DCP4 layout, the final release calibration selected:

```text
VLLM_PCIE_DMA_MIN_BYTES=25165824
VLLM_DCP_QUERY_SPLIT=1
VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=8192
VLLM_B12X_MLA_CKV_GATHER=1
VLLM_DCP_TOPK_OWNER_MERGE=1
VLLM_DCP_INDEXER_SHARDS=2
VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1
```

The isolated query-split gain was 23.5% at 8k, 57.9% at 64k, and 59.9% at
131k. The lossless DMA crossover was 24 MiB. On the tested interleaved layout,
the probe retained beneficial query split but selected prefetch depth 0 because
the overlap contended with TP traffic. This is the dual-socket/PCIe topology
failure mode that a hard-coded depth 1 could not handle.

Calibration never changes numerical wire format. `F8_DMA=ag`, `ring`, `i8*`,
or `mx*` is an explicit compressed-wire choice and causes the BF16 calibrator
to be skipped. DMA compression remains user opt-in even when a lossless probe
would favor DMA.

If calibration is unsupported, disabled, times out, or fails, the helper keeps
the conservative static/topology policy and the prior 6 MiB lossless DMA
crossover. Explicit values always take precedence, including an explicit
`PCIE_DMA_MIN_BYTES=off`.

For normal serving, this is sufficient; no low-level flags are required:

```bash
TP=8 DCP=4 docker compose up -d
```

To run the real probe, print its effective policy, and exit before loading
weights:

```bash
PCIE_CALIBRATION_ONLY=1 TP=8 DCP=4 docker compose run --rm --no-deps glm52
```

`DRY_RUN=1` is faster but deliberately skips measurement and shows only the
static fallback expansion:

```bash
DRY_RUN=1 TP=8 DCP=4 docker compose run --rm --no-deps glm52
# VLLM_DCP_QUERY_SPLIT=1
# VLLM_B12X_MLA_CKV_GATHER=1
# VLLM_DCP_TOPK_OWNER_MERGE=1
# VLLM_DCP_INDEXER_SHARDS=2
```

Use `PCIE_CALIBRATION=force` after changing hardware placement or when auditing
a cached result. `PCIE_CALIBRATION=off` disables the probe but not the eligible
DCP features. Individual explicit overrides remain available for A/B tests.

#### GPU order and timeout recovery

Calibration and serving must use the same ordered physical GPU list. The
helper resolves that list in this order:

1. non-empty `GPUS`;
2. existing `CUDA_VISIBLE_DEVICES`;
3. the model-family preset.

This matters on dual-socket systems. For TP8/DCP4, an interleaved order such as
`0,2,4,6,1,3,5,7` maps the query-split rank pairs `{0,4}`, `{1,5}`, `{2,6}`,
and `{3,7}` to adjacent physical pairs `(0,1)`, `(2,3)`, `(4,5)`, and `(6,7)`.
Replacing it with natural rank order makes those same logical pairs cross root
complexes and invalidates the calibration result.

A cold probe may compile kernels. v20 r4 raised the default timeout from 180
to 600 seconds, and r5 retains that behavior. If it still expires, the helper
terminates the complete `torchrun` process group before falling back; probe
workers cannot remain on the GPUs and contend with the vLLM NCCL initialization
that follows.

Audit an intentionally ordered placement without loading model weights:

```bash
GPUS= CUDA_VISIBLE_DEVICES=0,2,4,6,1,3,5,7 \
  PCIE_CALIBRATION=force PCIE_CALIBRATION_ONLY=1 \
  TP=8 DCP=4 docker compose run --rm --no-deps glm52
```

The first run should report `PCIE_CALIBRATION_STATUS=measured`. Preserve the
same ordered placement for the normal start that consumes the cached result:

```bash
GPUS= CUDA_VISIBLE_DEVICES=0,2,4,6,1,3,5,7 \
  PCIE_CALIBRATION=auto docker compose up -d
```

That start should report `cache-hit`. If an older image already timed out,
remove and recreate that complete container before retrying. Killing only the
frontend is insufficient because the older calibrator may have left worker
processes in the container. Until the fixed image can be pulled, this explicit
old-image fallback avoids running the probe while preserving the intended rank
order:

```bash
docker compose down --remove-orphans
GPUS=0,2,4,6,1,3,5,7 PCIE_CALIBRATION=off docker compose up -d
```

The `r4` image was validated with the analogous ordered list
`8,10,12,14,9,11,13,15` while leaving `GPUS` empty. The cold run completed as
`measured`, and its cache fingerprint preserved all eight physical indices and
PCI bus IDs in that exact order. The second run was a `cache-hit` in 2.16
seconds. A separate forced 20-second timeout returned the conservative policy;
inspection of the still-running container showed only its supervisor process,
no calibration workers, and 0 MiB allocated on all eight test GPUs.

A final release-image E2E run used that cache hit with TP8/DCP4/MTP3, A16,
online MXFP8, InstantTensor BUFFERED, and the same ordered GPUs. All eight NCCL
ranks initialized, InstantTensor loaded the complete 87-shard checkpoint,
CUDA-graph setup finished, and the API became ready on port 5668. A chat
request returned exactly `calibration-ok`. The first cold JIT/graph setup took
439.30 seconds and included 117 seconds of graph capture; quiet periods during
that work are expected and are distinct from a stall at `ncclCommInitRank`.
The run reported 2,262,784 KV-cache tokens at `GPU_MEMORY_UTILIZATION=0.96`.
The test container was then removed and all eight test GPUs returned to 0 MiB.

At runtime, full-CKV use is confirmed by
`Using transient full-CKV gather for B12X sparse MLA prefill`. Query split
creates a `query_split` process group. Owner merge keeps candidate scores FP32
and final indices exact; the release policy does not use lossy peer-DMA score
transport.

Virtual TP6 pads 64 attention heads to 66, leaving 11 local heads per rank.
The aligned full-CKV kernel is not its default: the measured experimental
11-to-16 padding made TP6/DCP3 64k prefill slower. v20 instead validates and
compacts the exact pitched partial workspace returned by that topology.

For short DCP messages, the helper uses the SparkInfer PCIe A2A pool. Messages
above `DCP_A2A_MAX_TOKENS=64` use `ag_rs`. The measured
`VLLM_PCIE_DMA_MIN_BYTES` applies to lossless byte dispatch; `F8_DMA` changes
the payload representation only when explicitly requested. It is irrelevant
to DCP1 and does not change decode arithmetic.

### Helper-Owned Serving Contract

The embedded helper, not the Compose file, owns these release defaults:

- InstantTensor `BUFFERED`, page-cache-aware model loading;
- local-inference NCCL 2.30.4 through both `LD_PRELOAD` and
  `VLLM_NCCL_SO_PATH`;
- B12X sparse MLA, B12X MoE, B12X PCIe all-reduce, and hybrid DCP transport;
- AOT/mega-AOT, FlashInfer sampler and autotune, async scheduling, chunked
  prefill, and prefix caching;
- attention-inclusive memory profiling, CUDA-graph memory estimation, and the
  v2 model runner;
- FP8 KV for standard GLM and NVFP4 MLA KV for the NF3 preset;
- `--enable-prompt-tokens-details`, `--enable-force-include-usage`, and
  `--enable-request-id-headers`;
- GLM tool/reasoning parsers and `reasoning_effort=high`;
- the exact 78-character sparse-indexer pattern:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

Do not manually duplicate these flags in Compose. The startup gate must contain
both loader/runtime lines before benchmarking:

```text
Loading safetensors using InstantTensor loader
vLLM is using nccl==2.30.4
```

## Accuracy Reference

Lower KLD is better. The unchanged checkpoint modes retain the corrected
five-run reference campaign used by v18/v19; v20 did not rerun every unchanged
cell. The A16 online-MXFP8 release smoke and the larger ignore-pattern study
below independently remain in the same range.

| Case | Corrected-reference KLD mean +/- sample SD | Role |
|---|---:|---|
| Luke NVFP4 A4 original | `0.10228 +/- 0.00634` | Native A4 activation path. |
| Luke NVFP4 A4 online MXFP8 | `0.10800 +/- 0.00697` | Faster BF16 dense linears, with an accuracy cost. |
| Luke NVFP4 A16 original | **`0.05994 +/- 0.00129`** | Highest-accuracy tested standard mode. |
| Luke NVFP4 A16 online MXFP8 | `0.06587 +/- 0.00253` | A16 accuracy/speed balance. |
| AMD MXFP4 experts A8 original | `0.08160 +/- 0.00432` | Native BF16 dense linears. |
| AMD MXFP4 experts A8 online MXFP8 | `0.08030 +/- 0.00309` | Faster dense linears; same measured distribution. |

These values compare each served checkpoint against the same corrected BF16
reference logits. They are not directly comparable to old June logits or a
different prompt/window policy.

## Online MXFP8 Attention Precision

A 2026-07-22 factorial KLD test measured which BF16 GLM-5.2 attention
projections should be excluded from online MXFP8 conversion. Each run used the
same Luke NVFP4 snapshot, corrected BF16 reference logits, TP8/DCP1, A16,
MTP off, FP8 KV, and 2,047 teacher-forced positions. Lower KLD is better.

| MXFP8 ignore set | Runs | Mean KLD | SD between runs | VRAM delta / GPU |
|---|---:|---:|---:|---:|
| none | 10 | `0.066006794` | `0.002060655` | baseline |
| `kv_b_proj` only | 20 | `0.065398317` | `0.002308562` | about `+0.13 GiB` |
| `q_a_proj` + `kv_a_proj_with_mqa` | 20 | **`0.064174724`** | `0.001603532` | about `+1.09 GiB` |
| all three | 10 | `0.065975578` | `0.001666660` | about `+1.22 GiB` |

The old `kv_b_proj`-only exclusion has no detectable benefit versus quantizing
all eligible linears (`p=0.83`). Keeping all three projections in BF16 is also
indistinguishable from ignoring none: the mean changes by only `-0.0000312`
(`-0.05%`, `p` approximately `0.97`) while consuming about `1.22 GiB/GPU`.
Therefore the current helper source no longer excludes `kv_b_proj` by default.
The corrected launcher is
[`serve-glm52-v19.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/a2129e983b07fbfaa5b872a1a0b25a07c3f01876/launchers/serve-glm52-v19.sh).

Keeping only the fused q/kv-a pair in BF16 is an optional quality experiment.
Its aggregate mean was 1.87% lower than the old `kv_b_proj`-only preset;
bootstrap P(improvement) was 97.69%, while the Welch test remained borderline
at `p=0.0599`. It costs about `1.09 GiB/GPU`, so it is not the memory-efficient
default.

The default online MXFP8 config in the updated helper source is:

```json
{"linear":{"weight":"mxfp8"}}
```

To retain the fused q/kv-a projection in BF16, set an explicit override:

```bash
ONLINE_QUANT=mxfp8 \
QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"},"ignore":["re:.*[.]q_a_proj$","re:.*[.]kv_a_proj_with_mqa$"]}' \
docker compose up
```

Both q/kv-a patterns must be supplied together because GLM-5.2 maps their
checkpoint shards into the runtime `fused_qkv_a_proj` module. Ignoring only one
creates an invalid mixed-precision fused module. Additional ignore patterns can
be appended to the same JSON array. For example, the historical `kv_b_proj`
override is `"re:.*kv_b_proj"`, although the KLD result above does not justify
using it.

The release image embeds this no-ignore default. An explicit value remains
useful when auditing a deployment or comparing alternate ignore sets:

```bash
ONLINE_QUANT=mxfp8 \
QUANTIZATION_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}' \
docker compose up
```

Reference-logit provenance, Hugging Face artifacts, metric definitions, and
the exact corrected-reference workflow are documented on the standalone
[GLM-5.2 KLD evaluation page](../benchmarks/glm52-kld-evaluation.md). Do not
mix these results with the superseded June GLM logits.

## Validation Method

The reproducible v20 wrapper is
[`scripts/bench-glm52-v20-validation.sh`](../scripts/bench-glm52-v20-validation.sh).
It pins both the immutable tag and Docker image ID, sets the corrected no-ignore
online-MXFP8 policy, and delegates execution to the maintained v18/v19 runner.
The complete `all` campaign contains 40 configurations:

- seven TP8/DCP1/MTP0 checkpoint and online-quant cases;
- six TP8/DCP1/MTP3 cases;
- seven cases each at TP8/DCP2, DCP4, and DCP8 with MTP off;
- native and online-MXFP8 AMD cases at TP6/DCP3 and DCP6 with MTP3;
- NF3 TP4/DCP4 with MTP off and MTP3.

The runner uses two topology- and CPU-isolated slots on the 16-GPU host. TP8
uses all 16 GPUs, TP6 uses 12, and TP4 uses 8. It starts both containers and
waits for both health checks and required loader logs, then waits another 30
seconds before starting either client. The two clients run serially. No result
is accepted while another model is loading.

The following are fixed historical-comparison benchmark profiles, not the
recommended serving memory defaults. Keeping their model length and GMU fixed
avoids attributing a changed memory shape to a runtime performance change.

| Validation profile | TP | DCP | MTP | Max seqs | Graph | Batched tokens | Max model len | GMU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard DCP1 | 8 | 1 | 0 or 3 | 32 | 128 | 8,192 | 131,072 | 0.90 |
| Standard fast DCP | 8 | 2, 4, 8 | 0 | 32 | 128 | 8,192 | 131,072 | 0.90 |
| Virtual TP6 | 6 | 3, 6 | 3 | 16 | 64 | 4,096 | 128,000 | 0.950 |
| NF3 hybrid | 4 | 4 | 0 or 3 | 8 | 64 | 3,072 | 131,072 | 0.960 |

Every configuration performs:

1. image-ID and source-mount rejection checks;
2. a short greedy coding response check that rejects empty or obviously
   corrupted output;
3. a 30-second context-zero CC1 decode run with up to 2,048 output tokens;
4. one discarded standalone 64k prefill warmup;
5. three standalone 64k prefill measurements, reported as the median;
6. mode-specific log assertions for A16/A8, online MXFP8/FP8, full-CKV DCP,
   borrowed TP6 workspace, and NF3 Grid188 execution;
7. server logs, container inspection, thermal snapshots, client JSON, and a
   per-case `summary.json` plus completion marker.

Published serving comparisons use `F8_DMA=0`. DMA `ag` and `ring` are separate
transport experiments and do not belong in the main decode table. Acceptance
statistics come from the exact post-decode server-log window. Prefill token
targeting must be recorded as either `estimate` for historical comparison or
`exact` for an exact 65,536-token prompt; never combine the two silently.

## Release Gate

Every benchmark started only after all required model instances were healthy;
no benchmark overlapped another model load. The 2026-07-27 gate adds MTP3 and
batched correctness coverage for the final runtime-stride image. The retained
2026-07-26 comparison immediately below it is MTP0.

### Final 2026-07-27 candidate

| Gate | Configuration | Result |
|---|---|---:|
| Standard decode | Luke NVFP4 online MXFP8, TP8/DCP1/MTP3, A16, seq=32, graph=128 | CC1 `158.299 tok/s`; CC32 aggregate `1,188.087 tok/s`; `96/96` clean |
| DCP4 | Luke NVFP4 original dense weights, TP8/DCP4/MTP3, A16, seq=32, graph=128 | CC1 `124.915 tok/s`; CC32 aggregate `939.761 tok/s`; 64k prefill `5,627 tok/s`; `64/64` clean |
| DCP8 | Luke NVFP4 original dense weights, TP8/DCP8/MTP3, A16, seq=32, graph=128 | CC1 `113.688 tok/s`; CC32 aggregate `758.172 tok/s`; 64k prefill `5,329 tok/s`; `64/64` clean |

The DCP8 query-split/owner-merge matrix was rerun on the complete source stack.
Unlike the older transport, merged SparkInfer #79 makes exact owner exchange
beneficial: `query_split=1` plus `owner_merge=1` was best at every tested
length.

| DCP8 mode | Prefill 8k | Prefill 64k | Prefill 128k |
|---|---:|---:|---:|
| query split off, owner merge off | `4,881` | `4,949.5` | `4,763` |
| query split off, owner merge on | `4,793` | `4,870.5` | `4,688` |
| query split on, owner merge off | `4,926.5` | `5,251` | `5,081.5` |
| query split on, owner merge on | **`5,283.5`** | **`5,404`** | **`5,230`** |

The standalone probe selected query split and retained owner merge. CKV
prefetch remains topology-calibrated rather than hard-coded: its isolated gain
was near the threshold on this DCP8 placement, and forcing overlap is known to
hurt some dual-socket layouts.

Raw final artifacts are under:

```text
/root/bench-results/glm52-v20-dcp8-query-owner-matrix-20260727/
/root/bench-results/glm52-v20-release-auto-gate-20260727/
```

### Clean r5 release gates

The broad concurrency gates below were run through the embedded helper on the
immediately preceding clean r5 candidate. Model loading and graph capture
completed before any client started. The final image adds the isolated EXL3
integration and explicit 140k full-CKV policy; its standard GLM path is gated
separately below rather than silently attributing the older measurements to a
different image ID.

The exact pushed digest was then started on GPUs 8-15, the slightly slower GPU
group on this host. Both DCP1 runs completed without request errors, and the
DCP4 log confirmed query split, partial `2x2` indexer replication, owner merge,
native CKV prefetch depth 1, and transient full-CKV gather at the printed 140k
capacity.

| Final-image gate | Result |
|---|---:|
| TP8/DCP1/MTP0 A16 decode, run 1 | `86.656` aggregate / `87.352` active tok/s |
| TP8/DCP1/MTP0 A16 decode, run 2 | `86.553` aggregate / `87.268` active tok/s |
| TP8/DCP4/MTP0 A16 exact 64k prefill | `5,685 tok/s`; `11.528 s` TTFT |
| Reported global KV budget | DCP1 `682,816`; DCP4 `2,379,520` tokens |

These values validate the final image on that GPU group; use the retained
GPU0-7 tables below for like-for-like historical performance comparisons.

| Gate | Configuration | Result |
|---|---|---:|
| Decode run 1 | Luke NVFP4, TP8/DCP1/MTP0, A16, seq=1, graph=6 | `87.909` aggregate / `88.580` active tok/s |
| Decode run 2 | Luke NVFP4, TP8/DCP1/MTP0, A16, seq=1, graph=6 | `87.845` aggregate / `88.548` active tok/s |
| MTP decode | Luke NVFP4, TP8/DCP1/MTP3, A16, online MXFP8, seq=32, graph=128 | `162.933` aggregate / `163.397` active tok/s; acceptance `63.793%` |

The MTP3 service then ran a deterministic 195-token ASCII correctness oracle
with thinking disabled. C1, C4, C6, C8, C16, and C32 produced respectively
`2/2`, `12/12`, `18/18`, `24/24`, `16/16`, and `32/32` byte-identical
responses, with one unique output and no HTTP errors at every concurrency.
This directly covers the earlier failure mode where output diverged above four
parallel requests.

Reported KV capacity was 682,816 tokens for the seq=1 MTP0 gate and 641,792
tokens for the seq=32 MTP3 gate at `GPU_MEMORY_UTILIZATION=0.96`. Raw release
artifacts are under:

```text
/root/bench-results/glm52-v20-r5-clean-20260728/
```

### Retained 2026-07-26 MTP0 gate

| Gate | Configuration | Result |
|---|---|---:|
| Standard decode | Luke NVFP4, TP8/DCP1, A16, seq=1, graph=6 | `87.503 tok/s` aggregate; `88.310` active per user |
| Calibrated prefill | Luke NVFP4, TP8/DCP4, A16, exact 65,538-token prompt | `5,835 tok/s`; TTFT `11.231 s` |
| Hybrid decode | local NF3, TP4/DCP4, A16/Grid188 | `57.3 tok/s` |

All three correctness checks passed. DCP4 resolved the measured 24 MiB
lossless-DMA crossover, 8k query-split crossover, partial `2x2` indexer, owner
merge, and CKV prefetch depth 1. It is within 0.3% of the preceding 5,853 tok/s
run. DCP1 remains in the established `~87-88 tok/s` envelope and NF3 matches
the preceding 57.2 tok/s result.

Raw artifacts for that retained gate are under:

```text
/root/bench-results/glm52-v20-final-20260726/final-vllm0c79e41-sie603f74/
```

### Broader retained matrix

The tables below are the broader pre-calibration v20 topology campaign. They
remain useful cross-topology reference data; unchanged cells were not rerun
merely to publish the final image.

### TP8 Luke NVFP4 A16

| DCP | Indexer topology | Decode CC1 | Prefill 64k median | Prefill 400k | KV tokens |
|---:|---|---:|---:|---:|---:|
| 1 | sharded | 87.98 aggregate / 88.69 active | 6,149.9 exact | - | 559,616 |
| 2 | sharded | 73.2 | 5,866 | 5,197.2 | 1,040,128 |
| 4 | partial 2x2 | 72.7 | 5,834 | 5,106.1 | 1,984,256 |
| 8 | partial 2x4 | 67.6 | 5,741 | 5,025.5 | 3,964,928 |

DCP4 is the throughput profile. DCP8 retains 99.8% more KV capacity than DCP4
while remaining within 1.6% at 64k and 1.6% at 400k. DCP1 remains at the
established `~87-88 tok/s` decode baseline.

### Virtual TP6 AMD MXFP4 A8, online MXFP8 dense

| DCP | Decode CC1 | Prefill 64k median | Prefill 400k | KV tokens |
|---:|---:|---:|---:|---:|
| 1 | 83.5 | - | - | 312,000 |
| 2 | 68.5 | 4,249 | 3,697.8 | 562,944 |
| 3 | 66.14 | 3,614 | 3,359.3 | 842,753 |
| 6 | 51.03 | 2,379 | 2,343.2 | 1,661,337 |

The DCP1 result matches the historical online-MXFP8 control (`83.43 tok/s`).
DCP3 improved 64k and 400k prefill by 43.5% and 39.7% over the older path;
DCP2 and DCP6 remain at their prior performance envelope.

### TP4 NF3 hybrid A16

| DCP | Decode CC1 | Prefill 64k median | Prefill 400k | KV tokens |
|---:|---:|---:|---:|---:|
| 2 | 57.2 | 4,040 | 3,184.1 | - |
| 4 | 57.3 | 3,992 | 3,340.9 | 934,912 |

These are MTP0 numbers. The older `~104 tok/s` NF3 rows are MTP3 and must not
be compared directly to this table.

The release-gate corrected-reference KLD smoke for Luke A16 online MXFP8 was
`0.0662177` over 2,047 positions. It used the historical `kv_b_proj`-only
helper preset and is consistent with the new 20-run mean of `0.065398317`.
See [Online MXFP8 Attention Precision](#online-mxfp8-attention-precision) for
the later four-way campaign and the current recommendation.

## Xid 31 / cuBLAS Layout Fix

The old failure occurred when `_v_up_proj` consumed a strided BMM view backed
by a tightly sized DCP allocation. A guarded VMM reproduction proved that
cuBLAS can read through the next 64 KiB boundary for this shape. Normal
PyTorch allocator segments tolerate that read-ahead; a tight IPC allocation
can expose an unmapped page and produce `Xid 31 FAULT_PDE`.

The final fix does not clone every DCP output and does not reserve tail padding.
Instead, DCP producers write the same logical BHD tensor into physical
head-major storage. The transpose consumed by cuBLAS is therefore contiguous
and safe. This preserves DCP1 speed and avoids reducing KV capacity merely to
provide speculative read padding.

The exact reported production configuration was reproduced with TP8/DCP2,
MTP3, seq=16, graph=64, 8,192 batched tokens, A4, online MXFP8, ring DMA, and
FP8 KV. `expandable_segments` was deliberately disabled with:

```text
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

Two consecutive transitions passed:

| Run | Prompt tokens | Decode tokens | Server healthy | Kernel Xid/PDE |
|---|---:|---:|---:|---:|
| 1 | 300,068 | 512 | yes | none |
| 2 | 320,063 | 512 | yes | none |

The clean 2026-07-25 release tree was gated again with
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`, so expandable
segments were still disabled. A short request followed by a 301,244-token
prompt and 32 decode tokens completed; server logs and `dmesg` contained no
`Xid`, `FAULT_PDE`, or illegal-memory error. References to the "Xid gate" in
the validation logs name this regression test; they do not report a new Xid.

Reproduce the client side after the server is healthy:

```bash
python3 scripts/validate-glm52-xid31-long-prefill.py \
  --port 8000 \
  --model GLM-5.2-v20-xid31 \
  --target-tokens 300000 \
  --max-tokens 512 \
  --output xid31-run1.json
```

The helper still defaults to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
for ordinary serving. Native KV-offload deployments that cannot use expandable
segments may override it; correctness no longer depends on allocator slack.

## TP6 Corrections

Virtual TP6 pads 64 attention heads to 66 and exposes 11 heads per rank. Two
independent contracts needed correction:

1. A partial final DCP prefill chunk retains the pitch of the full borrowed
   workspace. vLLM now accepts only that exact shape/stride and compacts it
   before projection.
2. The logical W4A8 expert width is 352, while the repacked dynamic kernel
   executes the zero-padded N128 extent of 384. SparkInfer now uses 384 for
   tile/task geometry and scratch sizing while preserving logical N=352 in the
   public execution plan.

At `MAX_NUM_SEQS=16`, graph=64, batch=4096, and GMU=0.95, TP6/DCP3 exposes
`700,449` KV tokens. This is lower than older unsafe estimates because v20 also
accounts for MRV2 graph and sparse-DCP transient memory before allocating KV.

## Reproduce The Campaign

The release wrapper is
[`scripts/bench-glm52-v20-validation.sh`](../scripts/bench-glm52-v20-validation.sh).
It pins both the image tag and local image ID and delegates to the established
resumable v18/v19 runner. Install the benchmark client at
`/root/llm-inference-bench/llm_decode_bench.py`, or set `BENCH` to its path.

```bash
git clone https://github.com/local-inference-lab/rtx6kpro.git
cd rtx6kpro
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r5

# Complete 40-case historical-compatible campaign. Existing completed cases
# under RESULT_ROOT are skipped only when both summary.json and complete exist.
RESULT_ROOT=/root/bench-results/glm52-v20-full-estimate \
  TOKEN_TARGETING=estimate \
  scripts/bench-glm52-v20-validation.sh all

# Exact-token TP8 DCP2/DCP4/DCP8 prefill campaign in a separate result root.
RESULT_ROOT=/root/bench-results/glm52-v20-dcp-fast-exact \
  TOKEN_TARGETING=exact \
  scripts/bench-glm52-v20-validation.sh dcp-fast
```

Individual resumable groups are also available:

```bash
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh dcp1-mtp0
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh dcp1-mtp3
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh tp6-mtp3
TOKEN_TARGETING=estimate scripts/bench-glm52-v20-validation.sh nf3

# One or more explicit cases use: "case TP DCP MTP".
TOKEN_TARGETING=exact scripts/bench-glm52-v20-validation.sh configs \
  "nvfp4-a16-orig 8 4 0" \
  "mxfp4-a8-orig 6 3 3"
```

Default checkpoint locations are the tested Luke snapshot under the Hugging
Face cache, `/root/models/GLM-5.2-BF16-AMDMXFP4experts`, and
`/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`. Override `NVFP4_MODEL`,
`MXFP4_MODEL`, or `NF3_MODEL` when the same immutable checkpoints live
elsewhere. The runner defaults to GPU slots `0-7` and `8-15`, ports 8190/8191,
and CPU sets `0-31,64-95` and `32-63,96-127`; topology, ports, and CPU sets are
all explicit environment overrides.

Useful operational controls:

| Variable | Effect |
|---|---|
| `RESULT_ROOT` | Stable resumable output root. Use a different root for `estimate` and `exact`. |
| `FORCE_RERUN=1` | Ignore completion markers and rerun selected cases. |
| `KEEP_SERVERS=1` | Leave the last measured server pair running for manual inspection. |
| `SETTLE_SECONDS` | Delay after all paired servers become healthy; release default is 30. |
| `PREFILL_REPEATS` | Measured 64k repeats after warmup; release default is 3. |
| `CACHE_A`, `CACHE_B` | Persistent, isolated JIT caches for the two slots. |
| `CUDA_ALLOC_CONF` | Allocator setting passed as `PYTORCH_CUDA_ALLOC_CONF`. |

Each result root ends with aggregate `summary.json` and `summary.tsv`. Raw
case directories retain the exact command inputs, image/container inspection,
server logs, decode and prefill JSON, correctness response, thermal data, and
backend markers needed to audit an outlier.

The final validation artifacts on the release host are under:

```text
/root/bench-results/glm52-v20-r5-clean-20260728
/root/bench-results/glm52-v20-final-20260726/final-vllm0c79e41-sie603f74
/root/bench-results/glm52-v20-dcp8-query-owner-matrix-20260727
/root/bench-results/glm52-v20-release-auto-gate-20260727
/root/bench-results/glm52-v20-r4-calibration-20260727
/root/bench-results/glm52-v20-final-tp6-20260725
/root/bench-results/glm52-v20-final-xid-transition-20260725
/root/bench-results/glm52-v20-final2-clean-20260725
/root/bench-results/glm52-v20-final2-xid-transition-20260725
/root/bench-results/glm52-gg-dcp-topology-matrix-20260725
/root/bench-results/glm52-gg-dcp8-indexer-2x4-20260725
```
