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
  public source and exposes the Trellis MoE path without a binary ABI shim;
- r7 ships the opt-in, DCP-aware LMCache 0.5.2 prefix-offload path from its
  merged public source branch, with RAM-only and buffered-filesystem modes,
  while leaving ordinary serving unchanged.
- r8 builds XGrammar 0.2.5 from its pinned source commit and verifies GLM
  structural-tag handling for `tool_choice=required`: at least one tool call
  is required, while multiple calls and normal completion after a call remain
  valid.
- r9 adds an opt-in dynamic per-token NVFP4 MLA KV record ABI, exact adaptive
  sparse-indexer folding with a bounded temporary-workspace budget, and
  `pytest` in the deployed runtime. The default FP8 KV path is unchanged.
- r11 adds the reviewed DCP policy corrections, recoverable late KV-transfer
  handling, the GLM `tool_choice=required` single-call fix, opt-in CUDA-clean
  forkserver startup, and the complete DCP-aware LMCache durability stack.
  LMCache now also works through the unified `glm52-exl3` / `exl3` presets.
- r12 stabilizes EXL3/Trellis memory profiling around the repeatable
  post-warmup peak, uses the consolidated SparkInfer fused-MoE API, and fixes
  small-row planning for both target and MTP draft execution. The consolidated
  path raises matched TP4/DCP4 MTP0 decode from the stock r11 mean of 44.66 to
  48.48 tok/s (+8.56%) without a measurable prefill regression. MTP3 remains
  within acceptance-rate and run-to-run variance.
- r13 incorporates the final review corrections for EXL3's direct validation
  runner, Trellis arena accounting, and LMCache deadline handling. It also
  archives all three exact integration trees and verifies their launcher and
  dependency pins in the release gate. It retains r12's MTP0 decode uplift.
- r14 adds native mixed K3/K4 EXL3 expert execution for the 3.25 bpw
  checkpoint. Routes are packed once, activations are rotated once, and both
  packed bitrates execute in one cooperative Trellis grid without reconstructing
  or requantizing the checkpoint weights. The release also hardens all
  row-derived element extents and launch-capacity contracts around that path.
- r15 adds the maintained DeepSeek-V4-Flash-0731 DSpark profile to the shared
  image. r16 adds shared-region native CPU KV offload with corrected SWA/MTP
  retention; the DS4-specific recipes remain on [the r16 page](ds4dspark-v20.md).
- r17 makes mixed EXL3 execution shape-aware: K3/K4 decode at M<=32 stays in a
  single Trellis grid, while prefill uses serial tier launches with FP32
  accumulation and bounded workspace planning. It also isolates semantic PCIe
  graph owners and fixes compressed MLA page strides, FlashInfer pre-KV
  autotune, W4A16 capture planning, and SparkInfer wheel packaging for local
  runtime-JIT headers.
- r18 adds opt-in online MXFP8 conversion for eligible BF16 dense projections
  in EXL3 checkpoints. Serialized EXL3 routed experts remain unchanged; the
  default EXL3 mode remains native (`ONLINE_QUANT=none`). The validated default
  ignore list keeps `q_a_proj`, `kv_a_proj_with_mqa`, and `lm_head` in BF16.
- r19 consolidates the mixed K3/K4 execution, online-MXFP8 overlay, and EXL3
  rotation loader into one reviewable change. It remains backward compatible
  with legacy `per_expert_v1` checkpoints and adds the explicit `shared_h_v1`
  artifact contract. A calibrated shared-H encode stores gate/up `SUH` and
  down `SVH` once per layer and TP rank instead of once per expert, saving
  672.36 MiB/GPU at GLM-5.2 dimensions without expanding the rows during load.
  At r19 publication the full checkpoint was still pending; r28 closes those
  KLD and E2E gates. The exact encoder procedure is documented in
  [GLM-5.2 EXL3 shared-H quantization](glm5.2_exl3_shared_h_quantization.md).
- r20 is a strict r19 superset. It qualifies block-32 mixed K3/K4 Trellis
  prefill and adds opt-in online K6 conversion for eligible BF16 dense EXL3
  matrices. Converted K6 tensors are cached as atomic, per-rank safetensors
  under `/cache/exl3-online`; serialized K3/K4 expert weights are never
  rewritten. The exact r20 image passed cold-create, warm-cache, correctness,
  CUDA-graph, capacity, decode, and 8k/64k prefill gates on TP4.
- r25 adds runtime-dynamic mixed-Trellis expert partitions. The 3.36 bpw
  checkpoint uses 206 K3 + 50 K4 experts in layer 3 and 160 K3 + 96 K4 in
  layers 4-77. Those counts are now kernel launch data instead of compiled
  state, so one cached kernel is correct for both partitions. The exact r25
  image passed TP4/DCP4/MTP3 startup, correctness, CUDA graphs, CC1/CC8
  decode, 8k/64k prefill, and source/runtime-contract gates.
- r26 corrects the automatic TP4/DCP4 prefill policy. Because TP4/DCP4 has
  only one query partition, exact owner exchange adds transport without
  removing duplicate work. Auto mode now uses query split, full CKV gather,
  two indexer shards, no owner exchange, and measured depth-1 CKV prefetch.
  The exact release image passed lossless PCIe calibration, CUDA graphs,
  coherent MTP3 decode, and uncached 8k/64k prefill on a root-port host.
- r28 qualifies the complete 3.42 bpw `shared_h_v1` checkpoint. It passes the
  layer-dependent 206/50 and 148/108 K3/K4 partitions plus broadcast-H flags
  through the runtime ABI, retains legacy 192/64 loading, and auto-selects the
  measured block-32 prefill path only for known partitions. The exact image
  passed shard-integrity, CUDA-graph, DCP1/DCP4, MTP0/MTP3, c8/c16 correctness,
  8k/64k prefill, and three-run KLD gates.
- r33 restores the capture-safe K6 small-M dispatch with an explicit SM120
  capability gate and realigns mixed-Trellis execution with the QSRT ABI. The
  exact image passed the standard TP4/DCP1/MTP3 decode regression gate on the
  3.36 bpw checkpoint.

Historical comparison data remains on [v18](glm5.2_v18.md), while the DCP
optimization background remains on [v19](glm5.2_v19.md). This page is
self-contained for building, starting, operating, and validating v20; older
pages are provenance, not required setup instructions.

Canonical source merging and the required post-merge rebuild are tracked in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).
The image below is the exact measured release candidate. Open source deltas
remain independently reviewable; already-merged fixes are pinned through the
stated GG and B12X base commits.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12x06db0f4-fi1ac6942-cu132-20260809-r33
Docker manifest: sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
Local image ID: sha256:60944a4ea1fbb2d1f35d7972f685d8fb0b91e77dd5aeca1dcafa3bcc29846d12
```

This supersedes all earlier v20 candidates. The registry manifest is the exact
25,194,226,286-byte local image used for the final helper and runtime-contract
gates below; there was no rebuild between those gates and push. All three
source trees were composed from clean public bases plus exact public PR heads.
The generated integration patches and lockfiles are immutable release artifacts.

`b12x` identifies the current B12X project name. Image labels retain legacy
SparkInfer aliases for compatibility with older tooling.

Pinned source stack:

| Component | Ref / commit |
|---|---|
| Canonical GG base | `local-inference-lab/vllm dev/gilded-gnosis` @ `e2666d9a65f41fc376607531453cbd57c4c71016` |
| Composed vLLM tree | `fa13d334a2962756f9f7e9b562deb85387359f42` |
| B12X base | `local-inference-lab/b12x master` @ `9bbae67841e4818e7472e1edcdca8ebcbda68611` |
| Composed B12X tree | `06db0f4b27dbd19eb934da0da27eff7a7c49d8c4` |
| EXL3 extension | `brandonmmusic-max/exllamav3 a1-retile-sm120` @ `704aefd743b390af4bd0fb429d1906f9b964c7d8` |
| FlashInfer | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| CUTLASS C++ / DSL | `e6233cbac5d7c7a865c19c91cd684ceece19513c` / `4.6.0` |
| InstantTensor | `49b4010afc1cae0441e71fe0b0bffc24fa05e932` |
| LMCache | `local-inference-lab/LMCache release/v0.5.2-glm52-dcp-base` @ `9cebd405d0caf4bebe01d694b5a8bf4e3e354314`, composed tree `9a05c8818bae48d15b79c7e876418bb813c08cd0`, wheel `0.5.2+glm52dcp.4` |
| XGrammar | `mlc-ai/xgrammar v0.2.5` @ `2ea71da4ccb997a06928c9fb69b99f330da56697`, wheel `0.2.5` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| NCCL | local-inference `2.30.4` |
| PyTorch / CUDA / loaded cuDNN | `2.12.0+cu132` / `13.2.1` / `9.20.0.48` |
| CUDA system-base cuDNN packages | `9.22.0.52` |
| Launcher/runtime source embedded in the image | `local-inference-lab/blackwell-llm-docker` @ `47ac813334e094090d5fd85b317d13b2e932ef09` |
| Build and immutable reproduction tree | `local-inference-lab/blackwell-llm-docker` @ `426da51285d0666508003b03a75a442139fb7979` |

Image labels expose all base commits, PR heads, result trees, patch and lock
hashes, and a cache fingerprint derived from the pinned sources.

## Build It Exactly

The canonical build entry point is
[`build-gilded-gnosis-v20-final-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/426da51285d0666508003b03a75a442139fb7979/build-gilded-gnosis-v20-final-cu132.sh).
The explicit reproduction mode uses archived, hash-verified locks and patches,
then verifies that applying them to the pinned bases produces the exact trees
above. It validates runtime symbols, helper contracts, and image labels before
allowing an optional push.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 426da51285d0666508003b03a75a442139fb7979
VLLM_RELEASE_COMPOSITION=reproduce-r33 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

For a new release candidate, omit `VLLM_RELEASE_COMPOSITION`. The default
always resolves the current clean GG, SparkInfer, and LMCache bases, composes
the exact versioned PR manifests, and fails if a base or PR head moves during
the build. The clean composer and archived source artifacts are reviewed in
[blackwell-llm-docker #7](https://github.com/local-inference-lab/blackwell-llm-docker/pull/7);
the merged LMCache build, helper, tests, and r6/r7 reproduction modes are
reviewed in
[blackwell-llm-docker #8](https://github.com/local-inference-lab/blackwell-llm-docker/pull/8).
The exact r11 composition is reviewed in
[blackwell-llm-docker #10](https://github.com/local-inference-lab/blackwell-llm-docker/pull/10).
The r13 locks, reduced current-base manifests, EXL3/Trellis runtime gate, and
archived reproduction mode are reviewed in
[blackwell-llm-docker #11](https://github.com/local-inference-lab/blackwell-llm-docker/pull/11).
The r17 manifests, exact integration artifacts, and packaging regression gate
are reviewed in
[blackwell-llm-docker PR #14](https://github.com/local-inference-lab/blackwell-llm-docker/pull/14).
The r18 EXL3 online-MXFP8 helper policy, immutable source composition, and
`reproduce-r18` mode are reviewed in
[blackwell-llm-docker PR #15](https://github.com/local-inference-lab/blackwell-llm-docker/pull/15).
The r20 release is committed directly to Docker `main` at `8376fe35add980d028c9a12b5c6d5e48e40e836d`. Its
immutable `reproduce-r20` archive composes
[vLLM PR #228](https://github.com/local-inference-lab/vllm/pull/228),
[SparkInfer PR #112](https://github.com/local-inference-lab/sparkinfer/pull/112),
and [SparkInfer PR #113](https://github.com/local-inference-lab/sparkinfer/pull/113).
They supersede the now-closed vLLM #225/#226 and SparkInfer #105/#110/#111
heads without changing the retained r19 behavior.

The current r28 release is committed directly to Docker `main` at
[`d780c39`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/d780c393677eb0dd9dc5d2e09b98230313ec50cf).
Its `reproduce-r28` archive pins all three composed source trees, retains
[SparkInfer PR #117](https://github.com/local-inference-lab/sparkinfer/pull/117),
and embeds the reviewed TP4/DCP4 policy from launcher commit `6a61804`.
Remote validation is bound to the exact Docker image ID by
`validation/gilded-gnosis-v20-r28-remote-gpu.json`. Python wheel archives are
not bit-reproducible, so the release tool has an explicit
`USE_EXISTING_VALIDATED_IMAGE=1` path that rechecks source labels, runtime
contracts, helper expansion, and the receipt without rebuilding the validated
image before push.

The r20 build retains r8's XGrammar 0.2.5 pin and its required-tool semantics
test plus a real
GLM-5.2 tokenizer initialization under this image's Transformers 5 runtime.
XGrammar upstream caps Transformers below 5 because of tokenizer regressions
in other model families; r8 removed only that wheel metadata cap and recorded
the compatibility override in image labels. It also retains r9's paired
dynamic NVFP4 cache ABI and adaptive exact fold described below. These paths
do not alter the default FP8 KV serving configuration or the retained speed
and KLD tables.

The build deliberately excludes the separate weight-lifetime experiments in
vLLM PR #154, vLLM PR #157, and SparkInfer PR #62. It also excludes the
experimental sparse-CKV decode stack in vLLM PRs #159-#161 and SparkInfer PRs
#64-#65. It also excludes the later `bounded_compat` commits on build PR #5;
that selector policy was not part of this candidate or its validation.

## Source Changes

The cuBLAS/Xid correction is already in the pinned GG base through
[vLLM PR #147](https://github.com/local-inference-lab/vllm/pull/147) and
[SparkInfer PR #54](https://github.com/local-inference-lab/sparkinfer/pull/54).
The current bases contain the previously reviewed DCP, dynamic-NVFP4,
forkserver, XGrammar, EXL3, and runtime-lifetime foundations. The clean r28
manifests apply the following exact PR heads on top of those bases:

| Project | Review | Purpose |
|---|---|---|
| vLLM | [#145](https://github.com/local-inference-lab/vllm/pull/145) | Calibrated NVFP4 MLA KV outer-scale wiring. |
| vLLM | [#229](https://github.com/local-inference-lab/vllm/pull/229) | Harden compressed MLA physical cache, workspace, and physical-stride contracts. |
| vLLM | [#213](https://github.com/local-inference-lab/vllm/pull/213) | Skip pre-KV V2 attention during FlashInfer autotune. |
| vLLM | [#214](https://github.com/local-inference-lab/vllm/pull/214) | Add the DeepSeek-V4-Flash-0731 DSpark launch profile. |
| vLLM | [#217](https://github.com/local-inference-lab/vllm/pull/217) | Allocate native CPU KV offload from one process-shared region. |
| vLLM | [#218](https://github.com/local-inference-lab/vllm/pull/218) | Align SWA/MTP retention and shared-prefix tails under native offload. |
| vLLM | [#216](https://github.com/local-inference-lab/vllm/pull/216) | Isolate target, draft, profiling, production, and eager PCIe graph channels. |
| vLLM | [#228](https://github.com/local-inference-lab/vllm/pull/228) | Consolidate shape-aware mixed EXL3 execution, qualified block-32 prefill, legacy/shared-H rotation loading, and cached online K6 for eligible dense matrices. |
| vLLM | [#230](https://github.com/local-inference-lab/vllm/pull/230) | Keep broadcast mHC preprocessing behind the compile boundary. |
| vLLM | [#235](https://github.com/local-inference-lab/vllm/pull/235) | Align DeepSeek-V4-0731 reasoning and tool-prompt behavior. |
| SparkInfer | [#106](https://github.com/local-inference-lab/sparkinfer/pull/106) | Honor compressed MLA physical page stride in the backend. |
| SparkInfer | [#117](https://github.com/local-inference-lab/sparkinfer/pull/117) | Pass mixed K3/K4 counts and broadcast-H flags at launch time so one compiled Trellis kernel supports layer-dependent partitions and shared rotations. |
| LMCache | [#7-#17](https://github.com/local-inference-lab/LMCache/pulls) | Compose recoverable MP retrieval, prefix retention, eviction/lookup synchronization, native-FS durability and accounting, bounded diagnostics, and durable L1 writeback/prefetch. |

The release build itself does not merge canonical branches and does not consume
a precomposed integration branch. It generates all three integration patches
from the clean bases and manifests, verifies their result trees, and archives
the exact r28 vLLM, SparkInfer, and LMCache artifacts.

At publication time, r28 includes the recorded heads through immutable release
locks; it does not imply that every head was merged into GG, SparkInfer master,
or LMCache. The current merge state and dependency order are maintained in
issue #33.

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
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
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
    image: voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
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
      - VLLM_EXL3_PREFILL_CAPACITY
      - VLLM_EXL3_ONLINE_TRELLIS_BITS
      - VLLM_EXL3_ONLINE_CACHE_DIR
      - VLLM_EXL3_ONLINE_CACHE_MODE
      - KV_CACHE_DTYPE
      - KV_FP8_ROPE
      - VLLM_NVFP4_MLA_DYNAMIC_SCALE
      - VLLM_NVFP4_MLA_SCALES_FILE
      - SPARKINFER_INDEXER_TWO_LEVEL_FOLD
      - SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB
      - F8_DMA
      - B12X_PCIE_DMA
      - NF3_GRID188
      - LOAD_FORMAT
      - INSTANTTENSOR_BACKEND
      - LMCACHE_MODE
      - LMCACHE_L1_GB
      - LMCACHE_L1_INIT_GB
      - LMCACHE_L2_PATH
      - LMCACHE_L2_GB
      - LMCACHE_L2_WORKERS
      - LMCACHE_CHUNK_SIZE
      - LMCACHE_MAX_GPU_WORKERS
      - LMCACHE_MAX_CPU_WORKERS
      - LMCACHE_PORT
      - LMCACHE_HTTP_PORT
      - LMCACHE_PROMETHEUS_PORT
      - VLLM_WORKER_MULTIPROC_METHOD
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

# Add a 24 GiB host-RAM prefix cache. Ordinary GPU KV remains managed by vLLM.
LMCACHE_MODE=ram LMCACHE_L1_GB=24 MOE_MODE=a16 TP=8 DCP=4 \
  MTP=0 docker compose up -d

# RAM front tier plus persistent buffered filesystem/NVMe tier. The default
# path is already covered by the Compose /cache volume.
LMCACHE_MODE=disk LMCACHE_L1_GB=8 LMCACHE_L2_GB=512 \
  LMCACHE_L2_PATH=/cache/lmcache/8000 MOE_MODE=a16 TP=8 DCP=4 \
  MTP=0 docker compose up -d

# NF3 hybrid. MODEL_FAMILY selects its TP4/A16/NVFP4-KV defaults.
MODEL_FAMILY=glm52-hybrid DCP=4 MTP=3 docker compose up -d

# Community EXL3 profile. The helper pins its tested checkpoint revision and
# TP4/DCP4 defaults; r13 validates both MTP0 and MTP3 from the clean image.
MODEL_FAMILY=glm52-exl3 docker compose up -d

# Opt-in EXL3 + online MXFP8 dense projections. EXL3 routed experts stay in
# their serialized format; q_a_proj, kv_a_proj_with_mqa, and lm_head stay BF16.
MODEL_FAMILY=glm52-exl3 QUANTIZATION=exl3 ONLINE_QUANT=mxfp8 \
  TP=4 DCP=1 MTP=0 docker compose up -d

# Mixed K3/K4 experts plus cached online K6 for eligible BF16 dense matrices.
# Keep JIT_CACHE mounted at the same path across restarts and image updates.
MODEL_FAMILY=glm52-exl3 \
  MODEL=/root/models/GLM-5.2-EXL3-TR3-3.25bpw \
  SERVED_MODEL_NAME=GLM-5.2-EXL3-TR3-3.25bpw \
  QUANTIZATION=exl3 ONLINE_QUANT=exl3-b6 \
  GPUS=4,5,6,7 TP=4 DCP=1 MTP=0 \
  MAX_NUM_SEQS=1 GRAPH=6 MAX_MODEL_LEN=131072 \
  MAX_BATCHED_TOKENS=4096 GPU_MEMORY_UTILIZATION=0.95 \
  docker compose up -d

# Mixed K3/K4 EXL3 checkpoint. The helper discovers its per-expert bitrates;
# no conversion or online quantization flag is required.
MODEL_FAMILY=glm52-exl3 \
  MODEL=willfalco/GLM-5.2-EXL3-TR3-3.25bpw \
  MODEL_REVISION=d7d79c2d14599dfce7a5d12b85f7ad73f40e623d \
  SERVED_MODEL_NAME=GLM-5.2-EXL3-TR3-3.25bpw \
  TP=4 DCP=4 MTP=3 MAX_NUM_SEQS=8 GRAPH=32 \
  VLLM_EXL3_TRELLIS_MAX_M=32 docker compose up -d

# Exact r26 mixed-partition validation profile. Layer 3 has 206 K3 + 50 K4
# experts; later layers have 160 K3 + 96 K4. SparkInfer #117 discovers and
# passes both partitions at runtime.
MODEL_FAMILY=glm52-exl3 \
  MODEL=willfalco/GLM-5.2-EXL3-TR3-3.36bpw \
  MODEL_REVISION=8d9aa923a17502675ca23737349b67f2e66bb69d \
  SERVED_MODEL_NAME=GLM-5.2-EXL3-TR3-3.36bpw \
  GPUS=0,1,2,3 TP=4 DCP=4 MTP=3 \
  ONLINE_QUANT=exl3-b6 KV_CACHE_DTYPE=nvfp4_ds_mla \
  MTP_MOE_BACKEND=triton MTP_DRAFT_SAMPLE_METHOD=greedy \
  MAX_NUM_SEQS=8 GRAPH=32 MAX_MODEL_LEN=524288 \
  MAX_BATCHED_TOKENS=2048 GPU_MEMORY_UTILIZATION=0.96 \
  PCIE_CALIBRATION=auto docker compose up -d

# Exact r28 shared-H release profile. The loader detects shared_h_v1 from the
# checkpoint metadata; no layout switch is required.
MODEL_FAMILY=glm52-exl3 \
  MODEL=willfalco/GLM-5.2-EXL3-TR3-3.42bpw \
  MODEL_REVISION=ae68c65947efa90bea37308e15421872f124c46d \
  SERVED_MODEL_NAME=GLM-5.2-EXL3-TR3-3.42bpw \
  GPUS=0,1,2,3 TP=4 DCP=4 MTP=3 \
  ONLINE_QUANT=exl3-b6 KV_CACHE_DTYPE=nvfp4_ds_mla \
  MAX_NUM_SEQS=16 GRAPH=16 MAX_MODEL_LEN=262144 \
  MAX_BATCHED_TOKENS=4096 GPU_MEMORY_UTILIZATION=0.95 \
  PCIE_CALIBRATION=auto docker compose up -d

# Capacity-first EXL3 profile. This preserves MAX_BATCHED_TOKENS=2048 but
# reuses a 1024-row prefill arena in slices, returning VRAM to the KV cache.
# Expect roughly 7-12% lower prefill throughput than unrestricted capacity.
MODEL_FAMILY=glm52-exl3 \
  MODEL=willfalco/GLM-5.2-EXL3-TR3-3.25bpw \
  MODEL_REVISION=d7d79c2d14599dfce7a5d12b85f7ad73f40e623d \
  TP=4 DCP=4 MTP=3 MAX_BATCHED_TOKENS=2048 \
  VLLM_EXL3_PREFILL_CAPACITY=1024 docker compose up -d

# EXL3 with the same DCP-aware LMCache RAM connector.
MODEL_FAMILY=glm52-exl3 LMCACHE_MODE=ram DCP=2 docker compose up -d
```

The standalone current recipe is
[`docker-compose-glm52-exl3-v20-r28.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/d780c393677eb0dd9dc5d2e09b98230313ec50cf/examples/docker-compose-glm52-exl3-v20-r28.yml).
The r26 recipe remains available for historical reproduction.

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
| `VLLM_EXL3_TRELLIS_MAX_M` | Optional EXL3 routed-row/scratch capacity. It must cover the selected graph plan; use `6` for the validated CC1 MTP3/graph-6 profile or `32` for the validated seq-8/graph-32 profile. Execution fails closed above the planned arena. |
| `VLLM_EXL3_PREFILL_CAPACITY` | Optional EXL3 routed-expert prefill-arena bound. Unset/blank equals `MAX_BATCHED_TOKENS` and is the fastest default. `1024` is the measured capacity-first setting: larger scheduler batches are processed as contiguous slices through the same smaller persistent arena, typically trading 7-12% prefill throughput for more KV capacity. It does not cap prompt length, context length, or scheduler batch size. |
| `MAX_MODEL_LEN` | Recommended standard and NF3 default: `262144`. TP6 remains `128000`. Raise only within the KV capacity reported at startup. |
| `MAX_BATCHED_TOKENS` | Standard `8192`; NF3 `2048`. The validated virtual-TP6 profile uses `4096`. |
| `GPU_MEMORY_UTILIZATION` | Recommended TP8 and NF3 default: `0.96`; TP6 at most `0.95`. TP8 `0.98` boots but is unsafe for long-prefill runtime allocations. |
| `MOE_MODE` | `a4`, `a16`, or `force-a8-experimental`. |
| `ONLINE_QUANT` | `none`, `mxfp8`, `fp8`, `nf3-mxfp8`, `exl3-b6`, or `custom`. `exl3-b6` is valid only with `QUANTIZATION=exl3`. |
| `QUANTIZATION_CONFIG_JSON` | Explicit online quantization policy; overrides the helper preset. |
| `VLLM_EXL3_ONLINE_TRELLIS_BITS` | Fixed to `6` by the `exl3-b6` preset. Other values fail closed. |
| `VLLM_EXL3_ONLINE_CACHE_DIR` | `/cache/exl3-online`; persistent per-rank safetensors for online K6 conversion. |
| `VLLM_EXL3_ONLINE_CACHE_MODE` | `readwrite`; accepts explicit `readonly` or `off` diagnostic policies. Keep `readwrite` for normal serving. |
| `KV_CACHE_DTYPE` | Standard `fp8`; NF3 uses `nvfp4_ds_mla`. |
| `KV_FP8_ROPE` | `0`; set to `1` only with the r9 dynamic NVFP4 MLA cache mode below. |
| `VLLM_NVFP4_MLA_DYNAMIC_SCALE` | `0`; opt-in per-token NVFP4 outer scales. Requires `KV_CACHE_DTYPE=nvfp4_ds_mla`, `KV_FP8_ROPE=1`, and an empty static scales file. |
| `VLLM_NVFP4_MLA_SCALES_FILE` | Static calibrated outer-scale file. Leave empty in dynamic mode; static and dynamic scaling are mutually exclusive. |
| `F8_DMA` | Default `0` (lossless BF16 wire). `ag`, `ring`, `a2a`, `i8*`, and `mx*` are explicit compressed-wire experiments and are never auto-selected. |
| `PCIE_CALIBRATION` | `auto` uses a matching cached result or measures before model loading; `force` remeasures; `off` uses the conservative static/topology policy. |
| `PCIE_CALIBRATION_ONLY` | `1` prints the effective policy and exits without loading the model. |
| `PCIE_CALIBRATION_TIMEOUT` | Cold-probe limit in seconds; default `600`. A timeout terminates `torchrun` and every probe worker before serving can start. |
| `PCIE_CALIBRATION_CACHE_DIR` | Defaults below the active fingerprinted XDG cache, normally `/cache/jit/<fingerprint>/pcie-calibration`. |
| `PCIE_DMA_MIN_BYTES` | `auto`, `off`, or an explicit byte/KiB/MiB threshold for lossless BF16 PCIe DMA dispatch. |
| `DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS` | `auto` uses the measured crossover; an integer is an explicit minimum context. |
| `DCP_CKV_GATHER_MAX_TOKENS` | `140000`; maximum pure-prefill size eligible for transient full-CKV gather. Raise explicitly for longer prefills, accepting the documented workspace cost. |
| `LMCACHE_MODE` | `off`; `ram` enables host-RAM prefix offload and `disk` adds a buffered filesystem tier. Supported by `glm52`, `glm52-hybrid`, and `glm52-exl3` / `exl3`; not supported by DS4. |
| `LMCACHE_L1_GB` / `LMCACHE_L1_INIT_GB` | RAM-cache maximum and initial allocation, both `24` GiB by default when LMCache is enabled. |
| `LMCACHE_L2_PATH` / `LMCACHE_L2_GB` | Disk mode defaults to `/cache/lmcache/<PORT>` and `256` GiB. Keep `/cache` on persistent storage. |
| `LMCACHE_CHUNK_SIZE` | Auto: `384` for DCP3/DCP6 and `512` otherwise. Override only with a value aligned to every effective cache block. |
| `LMCACHE_MAX_GPU_WORKERS` | Defaults to `TP`; every TP rank is a GPU transfer client even when DCP is 1. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn`; set `forkserver` only to evaluate the opt-in CUDA-clean startup path. It is validated for correctness but did not show a consistent startup win. |
| `SPARKINFER_INDEXER_TWO_LEVEL_FOLD` | `auto`; use exact two-level folding when its temporary workspace fits the budget and exact streaming carry otherwise. `0` and `1` are diagnostic overrides. |
| `SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB` | `256`; temporary-workspace budget used by the adaptive exact indexer-fold planner. |

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

### LMCache Prefix Offload

LMCache is opt-in. With `LMCACHE_MODE=off`, the helper executes the ordinary
model command and does not start an LMCache process or change allocator
settings. `ram` starts one in-container LMCache MP server backed by host RAM.
`disk` adds the native filesystem adapter with `use_odirect=false`, so reads can
come from the Linux page cache when resident and from the underlying NVMe when
not resident.

The merged LMCache GLM/DCP changes are required because MLA+DCP does not have
TP-only cache geometry.
For example, TP8/DCP4 stores four sequence-shard objects, each consumed by two
query-parallel ranks. The connector now carries that exact reader count,
mirrors replicated and partially replicated KV groups, expands manager block
IDs into physical kernel blocks where required, and retains every asynchronous
chunk-store future and CUDA IPC event until completion. No FP8/I8 compression
is applied to LMCache data by this feature.

When enabled, the wrapper sets `PYTORCH_CUDA_ALLOC_CONF` to
`expandable_segments:False`: LMCache registers KV storage by virtual address,
and expandable segments are incompatible with that contract. Other allocator
options are preserved. Recheck the desired long-context memory budget when
enabling LMCache rather than assuming the ordinary-serving KV number is
unchanged.

The r13 helper accepts LMCache for `glm52`, `glm52-hybrid`, `nf3`,
`glm52-exl3`, and `exl3`. Release validation covers TP8/DCP1, DCP2, and DCP4,
plus virtual TP6/DCP3 and DCP6. Revalidate a new checkpoint or topology before
production deployment.

Startup must print `LMCache ready` before vLLM begins loading. A repeated exact
prompt reports the restored token count under
`usage.prompt_tokens_details.cached_tokens`. For multiple services, give each
one unique model, LMCache, HTTP, and Prometheus ports; services on standard
ports `8000+N` derive non-conflicting defaults automatically.

For a service on the default port, this is a minimal cold/hit check. The first
request must report zero cached tokens, the second must report a non-zero
chunk-aligned count, and the generated text must be identical:

```bash
curl -fsS http://127.0.0.1:8089/healthcheck | jq

PROMPT="$(printf 'LMCache r7 deterministic prefix verification. %.0s' {1..1024})"
jq -cn --arg model GLM-5.2-NVFP4 --arg prompt "$PROMPT" \
  '{model:$model,prompt:$prompt,max_tokens:8,temperature:0,seed:0}' \
  >/tmp/lmcache-request.json

curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' -d @/tmp/lmcache-request.json \
  >/tmp/lmcache-cold.json
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' -d @/tmp/lmcache-request.json \
  >/tmp/lmcache-hit.json

jq '{cached_tokens:(.usage.prompt_tokens_details.cached_tokens // 0)}' \
  /tmp/lmcache-cold.json /tmp/lmcache-hit.json
test "$(jq -r '.choices[0].text' /tmp/lmcache-cold.json)" = \
  "$(jq -r '.choices[0].text' /tmp/lmcache-hit.json)"
```

If `PORT` is `8000+N`, the default LMCache HTTP health port is `8089+N`.
Explicit `LMCACHE_HTTP_PORT` overrides it.

r11 additionally validated persistence rather than only a live-process hit:

| Gate | Cold | After restart / prior data |
|---|---:|---:|
| TP8/DCP2 disk, 12,800 tokens | `0` cached, `2.479 s` | `12,800` cached, `0.205 s` |
| Prior r6 DCP1 disk data | n/a | `12,800` cached, `0.198 s` |

Both restored runs produced the same output hash as their cold reference. The
prior-data test used a copy and did not modify the original r6 cache.

### Startup Method And Rollback

The default worker multiprocessing method remains `spawn`. The opt-in
`VLLM_WORKER_MULTIPROC_METHOD=forkserver` path starts a CUDA-clean forkserver
before CUDA-bearing imports and completed model load, graph capture, and
LMCache operation without inherited-CUDA failures. It did not produce a
consistent startup-time improvement on the release host, so it is not the
default.

Feature-level rollback does not require changing images:

```bash
# Disable LMCache and use ordinary vLLM KV handling.
LMCACHE_MODE=off docker compose up -d

# Force the established multiprocessing path.
VLLM_WORKER_MULTIPROC_METHOD=spawn docker compose up -d

# Bypass measured DCP policy for a diagnostic run.
PCIE_CALIBRATION=off DCP_QUERY_SPLIT=0 DCP_CKV_GATHER=0 \
  DCP_TOPK_OWNER_MERGE=0 docker compose up -d
```

The previous immutable release remains available for whole-image rollback:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm34f26c2-side7739a-fi801d57a-cu132-20260728-r9
sha256:8246024490670e43af6ccdc3df9c6dd0a084119f4507b7ac35a86f5a1c6c33c3
```

### Checkpoint And Quantization Modes

| Checkpoint | `QUANTIZATION` | `MOE_MODE` | Supported tested online mode |
|---|---|---|---|
| `lukealonso/GLM-5.2-NVFP4` | `modelopt_fp4` | `a4` or `a16` | `none` or `mxfp8` |
| `festr2/GLM-5.2-BF16-AMDMXFP4experts` | `mxfp4` | `force-a8-experimental` | `none`, `mxfp8`, or `fp8` |
| `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` | `nvfp4_nf3_hybrid` | `a16` | `nf3-mxfp8` |
| `brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` | `exl3` | `a16` / Trellis | `none` or `mxfp8` |
| `willfalco/GLM-5.2-EXL3-TR3-3.25bpw` | `exl3` | mixed K3/K4 Trellis | `none`, `mxfp8`, or `exl3-b6` |
| `willfalco/GLM-5.2-EXL3-TR3-3.36bpw` | `exl3` | layer-dependent mixed K3/K4 Trellis | `exl3-b6` validated in r25 |
| `willfalco/GLM-5.2-EXL3-TR3-3.42bpw` | `exl3` | shared-H, layer-dependent mixed K3/K4 Trellis | `none` and `exl3-b6` validated in r28 |

For Luke NVFP4, A4 and A16 select the routed-expert activation path; they do
not rewrite the NVFP4 checkpoint weights. A16 uses BF16 expert activations and
is the highest-accuracy tested mode. Force-A8 selects MXFP4 expert W4A8 and
applies to the AMD checkpoint, not Luke NVFP4. Generic online MXFP8 converts
eligible BF16 dense linears and does not rewrite existing NVFP4/MXFP4 routed
expert tensors.

For EXL3, `ONLINE_QUANT=mxfp8` converts only eligible BF16 dense projections.
It does not reinterpret, reconstruct, or requantize serialized EXL3 routed
expert weights. Native EXL3 remains the default. The helper's validated ignore
list is `q_a_proj`, `kv_a_proj_with_mqa`, and `lm_head`.

`ONLINE_QUANT=exl3-b6` instead encodes eligible aligned BF16 dense matrices as
native Trellis K6. Non-eligible or unaligned dense tensors retain the MXFP8
fallback selected by the preset, while serialized K3/K4 routed experts remain
bit-for-bit unchanged. The cache key covers checkpoint identity and revision,
source fingerprint, encoder identity, TP rank and geometry, tensor shape,
codebook/scales schema, K bits, and seed. Writes use a file lock and atomic
rename; an incomplete or invalid entry is ignored and rebuilt. The Compose
`JIT_CACHE` volume mounts `/cache`, so reuse that same host directory to avoid
re-encoding approximately 11.90 GB of artifacts on each cold start.

**Direct `vllm serve` launches (helper bypassed):** `ONLINE_QUANT` is read
only by the embedded helper; vLLM itself never reads it, so setting
`ONLINE_QUANT=exl3-b6` as a plain environment variable in a Compose file
whose entrypoint execs `vllm serve` directly is inert. Several community
reference composes currently carry it that way. For a direct launch the
activation contract is:

```bash
VLLM_EXL3_ONLINE_TRELLIS_BITS=6          # the actual MXFP8-vs-K6 switch (3-8)
VLLM_EXL3_ENCODER_SOURCE=/opt/exllamav3-python/exllamav3  # REQUIRED: no baked default in vLLM
VLLM_EXL3_ONLINE_CACHE_DIR=/cache/exl3-online              # optional (default VLLM_CACHE_ROOT/exl3_online)
VLLM_EXL3_ONLINE_CACHE_MODE=readwrite                      # optional (already the default)
```

together with `--quantization-config` selecting `mxfp8` for `linear` (and
optionally `shared_experts`) — that flag is the eligibility gate; the env
var upgrades eligible tensors from MXFP8 to K6. There is no `exl3-b6`
weight type in `--quantization-config`. Two related behaviors worth
knowing when comparing published KLD numbers: `lm_head` is excluded from
the online overlay in code regardless of the ignore list, and the ignore
list filters only the `linear` spec — a `shared_experts` spec is always
converted when present.

With `MTP>0`, the helper creates a same-checkpoint MTP draft using the same MoE
backend and probabilistic draft sampling. The target and draft share the
virtual 66-head layout at TP6. Acceptance must be read from the server log for
the exact measurement window; the client acceptance field is not the release
source of truth.

### Dynamic NVFP4 MLA KV

r9 contains the paired vLLM/SparkInfer implementation for per-token outer
scales in 368-byte NVFP4 MLA cache records. It avoids applying one static,
checkpoint-specific outer scale to every token. The cache ABI is explicitly
namespaced as `nvfp4_ds_mla:fp8-rope-368:dynamic-token-v1`, so a producer and
consumer cannot silently disagree about record geometry.

This remains opt-in and does not change standard `KV_CACHE_DTYPE=fp8` serving.
Enable all parts together:

```bash
KV_CACHE_DTYPE=nvfp4_ds_mla \
KV_FP8_ROPE=1 \
VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 \
VLLM_NVFP4_MLA_SCALES_FILE= \
  docker compose up -d
```

Do not provide a static scales file in this mode. To return to the established
default, set `KV_CACHE_DTYPE=fp8` and remove the three dynamic-mode variables.

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
| TP4 / DCP2 | eligible | on | on | `0` | measured `0/1` |
| TP4 / DCP4 | eligible | on | **off** | `2` | measured `0/1` |
| virtual TP6 / DCP1 | off | off | off | `0` | `0` |
| virtual TP6 / DCP2, DCP3, DCP6 | off | off | on | `0` | `0` |

`DCP_INDEXER_SHARDS=0` means the ordinary fully sharded indexer. At TP8/DCP4,
`2` creates a measured partial `2x2` topology; at TP8/DCP8, `4` creates `2x4`.
The CKV cache remains sharded by the full DCP size. The query-split flag at
DCP1 does not create inter-rank DCP traffic. TP4/DCP4 has one query partition;
owner exchange therefore cannot remove duplicate query work and is disabled
by default. The local exact merge uses one collective, whereas owner exchange
would add row routing plus an output all-gather.

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

The r26 root-port TP4/DCP4 release gate selected:

```text
VLLM_PCIE_DMA_MIN_BYTES=25165824
VLLM_DCP_QUERY_SPLIT=1
VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=8192
VLLM_B12X_MLA_CKV_GATHER=1
VLLM_DCP_TOPK_OWNER_MERGE=0
VLLM_DCP_INDEXER_SHARDS=2
VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1
```

Its full-phase query-split gain was 15.5% at 8k, 42.7% at 64k, and 43.7% at
131k. Depth-1 prefetch overlap won by 3.1-4.0%. Owner exchange is disabled by
the topology-independent TP4/DCP4 work-geometry rule, not by the PCIe probe.

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
# Mixed EXL3 TP4/DCP4 uses the same automatic policy interface.
MODEL_FAMILY=glm52-exl3 TP=4 DCP=4 PCIE_CALIBRATION=auto \
  docker compose up -d
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

### EXL3 Prefill Capacity

`VLLM_EXL3_PREFILL_CAPACITY` is a memory/performance policy for the EXL3
Trellis routed-expert prefill runtime. It is not a model-context limit and is
unrelated to `MAX_NUM_SEQS`. The value selects the maximum number of token rows
for which each persistent prefill plan and scratch arena is sized.

When the variable is unset or blank, capacity resolves to
`MAX_BATCHED_TOKENS`. A 2,048-row scheduler batch therefore uses a 2,048-row
arena and one prefill dispatch. With capacity 1,024, the scheduler contract
remains 2,048 rows, but the EXL3 integration dispatches the batch as two
contiguous 1,024-row slices. A 2,500-row batch would use `1024 + 1024 + 452`.
Inputs, route IDs, route weights, and output row placement are sliced together;
the same persistent plan and accumulator are reused for the exact short tail.

The setting changes neither serialized weights nor quantization. It does not
change the dense online-K6 path, the small-M/decode Trellis plan, prompt length,
`MAX_MODEL_LEN`, or the scheduler's `MAX_BATCHED_TOKENS` limit. Its only direct
cost is extra prefill dispatch and setup work when a live batch exceeds the
selected capacity. Decode does not directly use this arena. Tests cover exact
capacity boundaries, short tails, mixed and homogeneous expert layouts,
route/output preservation, invalid values, and CUDA-capture rejection.

The arena is persistent and reused across MoE layers, so its size is not
multiplied by the layer count. Target and MTP draft runtimes own separate
mutable arenas, however, which is why MTP profiles can recover substantially
more memory than MTP0 when the bound is reduced.

Original TP4 target-plus-MTP evidence for the capacity feature:

| Persistent arena | Unrestricted | Capacity 1024 | Recovered |
|---|---:|---:|---:|
| Target | 759.8 MiB | 279.7 MiB | 480.1 MiB |
| MTP draft | 1,054.2 MiB | 414.1 MiB | 640.1 MiB |
| **Total per rank/GPU** | **1,814.0 MiB** | **693.8 MiB** | **1,120.2 MiB** |

That profile increased exposed KV capacity from an estimated 110,080 tokens
to 257,024 tokens and passed an exact 126k retrieval test. The 1.1 GiB figure
is specific to this target-plus-draft geometry; it must not be generalized to
MTP0 or a different scheduler capacity.

A later matched TP4/DCP4/MTP0 shape-aware comparison used
`MAX_BATCHED_TOKENS=2048`:

| Mode | Persistent prefill buffers | KV tokens | Prefill 3k | Prefill 32k | Prefill 128k |
|---|---:|---:|---:|---:|---:|
| Default capacity 2048 | 783.3 MiB | 856,320 | 3,792 | 3,690 | 3,324 |
| Capacity 1024 | 438.7 MiB | 896,000 | 3,335 | 3,277 | 2,932 |

In that MTP0 profile, capacity 1024 recovered 344.6 MiB/GPU and exposed
39,680 additional KV tokens (`+4.6%`) while reducing prefill throughput by
`11.2-12.1%`. A separate combined MTP5 profile measured a `7-10%` prefill
cost and approximately `1-2%` generation-throughput variation. The supported
operator expectation is therefore a configuration-dependent **7-12% prefill
trade-off**, not 1-2%.

Operational guidance:

- leave the variable unset for maximum prefill throughput and lowest TTFT;
- use `1024` when KV capacity or successful long-context startup matters more;
- values must be positive and no greater than `MAX_BATCHED_TOKENS`; invalid
  values fail startup rather than being silently clamped;
- lower values create more slices and have not been qualified as release
  defaults;
- compare the startup-reported KV capacity and matched uncached prefill on the
  target topology before adopting a non-default value.

Do not confuse this setting with `VLLM_EXL3_TRELLIS_MAX_M`, which controls the
small-row/decode Trellis window and must cover the selected CUDA graph plan.
The final r20, r25, r26, and r28 release gates below left
`VLLM_EXL3_PREFILL_CAPACITY` unset, so their published 8k/64k prefill numbers
represent the unrestricted, fastest path.

## Release Gate

Every benchmark started only after all required model instances were healthy;
no benchmark overlapped another model load. The 2026-07-27 gate adds MTP3 and
batched correctness coverage for the final runtime-stride image. The retained
2026-07-26 comparison immediately below it is MTP0.

### Final 2026-08-09 r33 K6 and mixed-Trellis contract gate

The exact r33 image at the top of this page was tested without source mounts
on physical GPUs 4-7 of `192.168.0.69`. The checkpoint was
`willfalco/GLM-5.2-EXL3-TR3-3.36bpw` revision
`8d9aa923a17502675ca23737349b67f2e66bb69d`. The profile used TP4/DCP1/MTP3,
online EXL3 K6, NVFP4 DS-MLA KV, `MAX_NUM_SEQS=1`, graph cap 6, model length
131,072, GMU 0.95, greedy draft sampling, and InstantTensor BUFFERED.

| Gate | Result |
|---|---:|
| Source, image-label, launcher-hash, and helper contracts | pass |
| Focused SM120 K6 dispatch tests | 9 passed, 59 deselected |
| Full and piecewise CUDA graphs | pass |
| Model load | 80.11 GiB/GPU |
| Logical KV capacity | 202,304 tokens |
| Standard CC1 decode run 1 | 116.21 tok/s |
| Standard CC1 decode run 2 | 112.03 tok/s |
| Two-run median | **114.12 tok/s** |
| Historical matched profile | 113.40 tok/s |

The standard gate uses the default `llm_decode_bench` encyclopedia prompt and
temperature 1.0. A separate synthetic integer-sequence prompt reached about
144 tok/s because it is substantially more MTP-friendly. That result is not a
standard headline and is excluded from the regression comparison.

B12X #136 restores the K6 small-M path during CUDA graph replay and now checks
for exact SM120 capability before selecting it. B12X #137 fixes the mixed
Trellis QSRT argument contract. The full source identity and artifact hashes
are bound to the exact image in the
[r33 remote validation receipt](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/gilded-gnosis-v20-r33-remote-gpu.json).

### Final 2026-08-04 r28 shared-H gate

The exact r28 image at the top of this page was tested without source mounts on
physical GPUs 4-7 of the TP4 root-port host. The checkpoint was
`willfalco/GLM-5.2-EXL3-TR3-3.42bpw` revision
`ae68c65947efa90bea37308e15421872f124c46d`. All 79 model-shard hashes passed.
Metadata declares `shared_h_v1`; layer 3 uses 206 K3 + 50 K4 experts, layers
4-77 use 148 K3 + 108 K4, and layer 78 is all K3.

The physical shared-H representation removes 705,024,000 bytes
(672.36 MiB/GPU) for MTP0 and 714,424,320 bytes (681.33 MiB/GPU) when the MTP
layer is loaded. Runtime logs independently report 8.96 MiB saved for every
routed-expert layer and rank. No tensor is expanded back to per-expert H rows.

| Profile | CC1 ctx0 decode | Prefill 8k | Prefill 64k | Logical KV |
|---|---:|---:|---:|---:|
| TP4/DCP1/MTP0, default K6 | 53.25 / 53.33 tok/s | 3,586.81 | 3,386.11 | 241,216 |
| TP4/DCP1/MTP3, default K6 | 113.40 tok/s | - | - | 158,720 |
| TP4/DCP4/MTP3, default K6 | 93.76 tok/s | 3,488.76 | 3,337.05 | 163,840 |

The DCP4 profile used `MAX_NUM_SEQS=16`, graph 16, batch 4,096, and GMU 0.95.
Its aggregate decode was 240.17 tok/s at CC4 and 336.35 at CC8. Repeated batch
correctness passed 24/24 responses at c8 and 32/32 at c16. GMU 0.97-0.98 left
too little transient headroom for the first long prefill; 0.95 is the qualified
default. The legacy 3.25 bpw checkpoint also booted, selected its 192/64 path,
captured CUDA graphs, and returned the correct chat answer.

The KLD gate uses 2,047 teacher-forced positions from the published 2026-07-08
BF16 reference, TP4/DCP1/MTP0, eager execution, and three repeats for each
published profile. Lower is better. Run-to-run SD was zero for all three
repeated profiles; the no-online NVFP4 isolation row is a one-run control.

| Profile | KV format | Mean KLD | SD across positions |
|---|---|---:|---:|
| Checkpoint only; no online quantization | FP8, matched to BF16 reference capture | **0.074145973** | 0.292925745 |
| Online K6 isolation | FP8, matched to BF16 reference capture | **0.077949159** | 0.331558079 |
| No-online KV isolation | NVFP4 MLA | **0.107971445** | 0.432494372 |
| Release default; online K6 | NVFP4 MLA | **0.108828284** | 0.424119890 |

The first and last rows represent best matched checkpoint quality and the
actual release runtime. With FP8 KV held constant, K6 adds 0.003803186 mean
KLD (about 5.13% relative). With NVFP4 KV held constant, K6 adds 0.000856839
(about 0.79% relative). These interaction-dependent KLD deltas are not
additive. The KLD run loaded 84.08 GiB/GPU without online quantization and
79.47 GiB/GPU with K6, a further 4.61 GiB/GPU runtime saving.

The matched TP4/DCP1/MTP0 serving-capacity matrix at GMU 0.95, max sequences
1, and graph cap 6 is:

| Online mode | KV format | Model load/GPU | Logical KV tokens |
|---|---|---:|---:|
| none | FP8 | 83.85 GiB | **72,448** |
| K6 | FP8 | 79.25 GiB | **163,072** |
| none | NVFP4 MLA | 83.85 GiB | **107,136** |
| K6 | NVFP4 MLA | 79.25 GiB | **241,216** |

The no-online profiles cannot satisfy a 131,072-token request at GMU 0.95;
their exact block counts were therefore exposed with a 65,536 model limit.
The DCP4 value above is aggregate logical capacity: 40,960 local tokens per
CP rank multiplied by DCP4 equals 163,840.

The exact source trees, image ID, checkpoint audit, results, and KLD reference
are bound in
[`gilded-gnosis-v20-r28-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/32ebb47f564eb7085959b47293d794d92f14d9e3/validation/gilded-gnosis-v20-r28-remote-gpu.json).
The fully reproducible KLD commands are on the
[GLM-5.2 KLD page](../benchmarks/glm52-kld-evaluation.md).

### Final 2026-08-03 r26 TP4/DCP4 policy gate

The historical exact r26 image was copied to the separate root-port host and
tested on physical GPUs 4-7. No source mount or runtime
overlay was present. The first start ran the packaged lossless PCIe probe and
selected query split, full CKV gather, two indexer shards, no owner exchange,
and depth-1 CKV prefetch.

The matched MTP0 policy POC used the 3.25 bpw checkpoint. Values are median
uncached client prefill tok/s:

| Profile | 8k | 64k | 128k | Global KV budget |
|---|---:|---:|---:|---:|
| DCP1 reference | 3,642 | 3,517 | 3,475 | 331,136 |
| DCP4 old auto policy | 2,561 | 2,449 | 2,374 | - |
| DCP4 r26 auto policy | **3,533** | **3,384** | **3,286** | 1,216,000 |
| r26 uplift over old DCP4 | +37.9% | +38.2% | +38.4% | - |
| r26 gap to DCP1 | -3.0% | -3.8% | -5.4% | 3.67x DCP1 |

The release-image gate then used
`willfalco/GLM-5.2-EXL3-TR3-3.36bpw` revision
`8d9aa923a17502675ca23737349b67f2e66bb69d`, TP4/DCP4/MTP3, greedy draft
sampling, Triton draft MoE, graph cap 32, `MAX_NUM_SEQS=8`, batch 2,048,
GMU 0.94, dynamic NVFP4 MLA KV, and cached online K6 for eligible BF16 dense
matrices.

| Gate | Result |
|---|---:|
| Source and runtime contracts | pass |
| Automatic lossless PCIe calibration | pass; 24 MiB DMA crossover |
| Full and piecewise CUDA graphs | pass |
| Coherent decode | pass; 256 tokens |
| MTP draft acceptance | 164 / 276 = 59.4% |
| Logical KV capacity | 498,176 tokens |
| Uncached prefill 8k | **3,125 tok/s** |
| Uncached prefill 64k | **2,988 tok/s** |
| Gain over matched r25 release gate | **+29.8% / +48.1%** |

The first 3.36 bpw start created its online K6 cache and took 589 seconds to
load the model. This is one-time work when the Compose `/cache` volume is kept
persistent. The machine-readable receipt is
[`gilded-gnosis-v20-r26-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/fb58f5cb1df82dab841f6e73dd0995cbb20cd0f0/validation/gilded-gnosis-v20-r26-remote-gpu.json).
It binds all gates to image ID
`sha256:30a752f0b490a3841610dfbea8f5758eab66924537feaeb1cd06b8e6ac92cce4`.

### Final 2026-08-03 r25 dynamic mixed-Trellis gate

The r25 Docker image
`voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25`
with image ID
`sha256:7b2cdb7cb0e4298a5c0907d53c5bca25287ba10ca6707cb21ede1fb65076dea0`
was tested on physical GPUs 4-7 of the separate four-GPU root-port host. No
source mount or runtime overlay was present. The checkpoint was
`willfalco/GLM-5.2-EXL3-TR3-3.36bpw` at revision
`8d9aa923a17502675ca23737349b67f2e66bb69d`. Layer 3 contains 206 K3 + 50 K4
experts; layers 4-77 contain 160 K3 + 96 K4. SparkInfer #117 passes those
counts at launch time instead of capturing the first partition in compiled
kernel state.

The measured profile was TP4/DCP4/MTP3, greedy draft sampling, Triton draft
MoE, graph cap 32, `MAX_NUM_SEQS=8`, batch 2,048, GMU 0.96, dynamic NVFP4 MLA
KV, and cached online K6 for eligible BF16 dense matrices. The final image
contained vLLM tree `f5981f14b4d39979bc0d799c020d42002b707257` and SparkInfer
tree `978cdb3593367469abd16bc8bdbc4ed0ea2787da`.

| Gate | Result |
|---|---:|
| Source and runtime contracts | pass |
| Cold model startup and coherent output | pass |
| Full and piecewise CUDA graphs | pass |
| Logical KV capacity | 770,048 tokens |
| CC1 raw decode | 90.98 / 92.65 tok/s; median 91.82 |
| CC1 draft acceptance | 57.48% |
| CC1 normalized verifier rate | 33.70 steps/s vs 34.06 reference (-1.04%) |
| CC8 raw decode | 314.87 tok/s |
| CC8 draft acceptance | 60.56% |
| CC8 normalized verifier rate | 111.78 steps/s vs 112.00 reference (-0.20%) |
| Uncached prefill 8k | 2,407 tok/s vs 2,299 reference (+4.7%) |
| Uncached prefill 64k | 2,018 tok/s vs 2,030 reference (-0.6%) |

Raw speculative decode varies with draft acceptance, so the release comparison
also normalizes by verifier steps. Both CC1 and CC8 normalized rates are within
1.1% of the exact-source reference, while prefill is unchanged within noise or
faster. The machine-readable validation receipt is
[`gilded-gnosis-v20-r25-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/65c88fbef1bc9ac8f8fd8601431d2aef8fc17517/validation/gilded-gnosis-v20-r25-remote-gpu.json).
It binds all required gates to the exact local image ID. A later source-equal
rebuild produced a different ID because wheel archives are not bit-reproducible;
the gate rejected it, and only the original GPU-tested image was pushed.

### Final 2026-08-02 r20 mixed-EXL3 and online-K6 gate

The exact pushed r20 image was tested on a separate four-GPU root-port host.
Only physical GPUs 4-7 were exposed. The profile was TP4/DCP1/MTP0,
`MAX_NUM_SEQS=1`, graph cap 6, batch 4,096, maximum model length 131,072,
GMU 0.95, dynamic NVFP4 MLA KV, and the standard in-image helper. The local
checkpoint was `/root/models/GLM-5.2-EXL3-TR3-3.25bpw`; its serialized routed
experts remained 192 K3 plus 64 K4 tensors per MoE layer.

`ONLINE_QUANT=exl3-b6` selected block-32 mixed-expert prefill and native K6
for eligible dense matrices. The first start created 1,644 cache files totaling
11,897,961,792 bytes under `/cache/exl3-online`. The identical warm restart
reported 1,644 cache hits, no encode operations, and an unchanged cache
manifest SHA-256 of
`562715a08e1fea3aae0437e022d84ff424062ceba1f0d9ec32d0cf91b3a0a07a`.

| Gate | Result |
|---|---:|
| Cold create to `/health` | 827.04 s with both EXL3 and JIT caches empty |
| Warm cache restart to `/health` | 113.94 s |
| Warm model-loading phase | 26.05-27.17 s |
| Weight memory | 76.54 GiB/GPU cold; 76.38 GiB/GPU warm |
| Available KV memory | 11.25 GiB/GPU cold; 11.38 GiB/GPU warm |
| Logical KV capacity | 331,136 tokens cold; 334,848 warm |
| CUDA graph capture | full + piecewise pass |
| Correctness sanity | coherent, repeated answer 42 |
| MTP0 CC1 decode | 53.23 / 53.08 tok/s |
| Uncached prefill 8k | 3,635.67 tok/s |
| Uncached prefill 64k | 3,512.62 tok/s |

The preceding K6 POC measured 53.079/53.157 tok/s decode, 3,604.39 tok/s at
8k, and 3,490.25 tok/s at 64k on the same host class. The final r20 path is
therefore within run-to-run noise or 0.1-0.9% faster; no release regression was
observed. These root-port-host values are not interchangeable with the TP8
switch-host GLM tables elsewhere on this page.

### Final 2026-08-01 r19 legacy-EXL3 compatibility gate

The exact pushed r19 image was pulled by registry digest on a separate
eight-GPU host. Docker exposed only physical GPUs 4-7 by UUID; the test used
TP4/DCP1/MTP0, `MAX_NUM_SEQS=1`, graph cap 6, batch 2,048, 131,072 maximum
model length, GMU 0.95, FP8 KV, and the standard in-image helper. The unchanged
legacy checkpoint was `willfalco/GLM-5.2-EXL3-TR3-3.25bpw` revision
`d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`.

| Gate | Result |
|---|---:|
| Legacy layout | `per_expert_v1` (metadata absent) |
| Mixed expert tiers | 192 K3 + 64 K4 per MoE layer |
| Model load | 80.9 GiB/GPU, 25.9-27.8 s |
| Available KV memory | 7.57 GiB/GPU |
| Logical KV capacity | 150,720 tokens |
| CUDA graph capture | full + piecewise pass |
| Correctness response | exact `R19 EXL3 LEGACY OK` |
| Chunked-prefill smoke | 8,823 prompt tokens, pass |
| Runtime error scan | clean |

This proves that r19 does not require existing EXL3 checkpoints to be
rewritten. The `shared_h_v1` loader additionally passed 47 installed-package
unit tests, including legacy/shared-H mixed-tier parameterization. A complete
newly encoded shared-H checkpoint was not yet available at r19 publication.
The r28 gate above closes its full-model KLD and E2E validation.

### Final 2026-08-01 r18 EXL3 online-MXFP8 gate

The published r18 image was booted through its embedded helper with
`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` revision
`9297b9f1d53af5c67cffa01e30cc071a1ff7144b`, GPUs 0-3, TP4/DCP1/MTP0,
`MAX_NUM_SEQS=1`, graph cap 6, batch 4,096, maximum model length 131,072,
GMU 0.95, `QUANTIZATION=exl3`, and `ONLINE_QUANT=mxfp8`.

Startup confirmed direct MLA absorbed projections from the SparkInfer MXFP8
pack. A correctness request returned exactly `R18 EXL3 MXFP8 OK`; a 15-second
CC1 run reached `59.3 tok/s`, matching the pre-release candidate's
`59.52 tok/s` within measurement noise.

| TP4/DCP1/MTP0 | Native EXL3 | EXL3 + online MXFP8 | Delta |
|---|---:|---:|---:|
| Model VRAM per GPU | 75.79 GiB | 72.42 GiB | -3.37 GiB |
| Logical KV capacity | 340,480 | 439,424 | +29.1% |
| Decode CC1 | 54.26 tok/s | 59.52 tok/s | +9.7% |
| Prefill 8K | 4,045 tok/s | 3,985 tok/s | -1.5% |
| Prefill 64K | 3,975 tok/s | 4,210 tok/s | +5.9% (one long sample) |

The MTP3 candidate also returned coherent output, reached `121.42 tok/s` at
CC1 with 57.2% accepted draft tokens, and exposed 350,784 logical KV tokens.
The exact image was rebuilt from the archived r18 composition, passed its full
build/runtime policy gates, and was pushed without a subsequent rebuild.

The implementation is reviewed in
[vLLM PR #223](https://github.com/local-inference-lab/vllm/pull/223); the
immutable release composition is reviewed in
[blackwell-llm-docker PR #15](https://github.com/local-inference-lab/blackwell-llm-docker/pull/15)
and reproduced with `VLLM_RELEASE_COMPOSITION=reproduce-r18`.

### Final 2026-08-01 r17 EXL3 packaging and prefill gate

The published r17 image was validated with
`willfalco/GLM-5.2-EXL3-TR3-3.25bpw` revision
`d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`, TP4/DCP4/MTP0,
`MAX_NUM_SEQS=1`, graph cap 6, batch 2,048, maximum model length 262,144,
GMU 0.95, and NVFP4 MLA KV. It exposed 855,808 logical KV tokens.

The first candidate had the correct #222 EXL3 kernels but silently selected
PyNCCL because the SparkInfer wheel omitted `ipc_handle_registry.h`, a local
header included by the runtime-compiled PCIe source. SparkInfer #105 now
packages local headers and tests the wheel contract. The final image selected
`B12X_PCIE_ONESHOT_DMA` for the TP group and emitted no missing-header fallback.

| Final r17 standalone prefill | Client throughput |
|---|---:|
| 3K | `3,761` tok/s |
| 32K | `3,670` tok/s |
| 128K | `3,306` tok/s |

The matched post-fix A/B reference was 3,799 / 3,677 / 3,302 tok/s, so the
published image is at parity within run-to-run noise. A response sanity request
returned exactly `OK`. The earlier Discord result attributed to #219 is not a
missing release delta: #219 was superseded by #222, which retains one-grid
K3/K4 decode for M<=32 and serial FP32-accumulating prefill while adding bounded
prefill capacity.

The exact release composition and evidence are reviewed in
[blackwell-llm-docker PR #14](https://github.com/local-inference-lab/blackwell-llm-docker/pull/14)
and reproduced with `VLLM_RELEASE_COMPOSITION=reproduce-r17`.

### Final 2026-07-30 r14 mixed-EXL3 gates

The r14 image was booted through its embedded helper with no source mounts or
runtime patches. The gate used
`willfalco/GLM-5.2-EXL3-TR3-3.25bpw` revision
`d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`, GPUs 0-3, TP4/DCP4,
NVFP4 MLA KV, graph cap 6, and GMU 0.95. Startup identified every routed layer
as `tiers=((3, 192), (4, 64))`, completed compilation and CUDA graph capture,
and exposed 747,776 KV tokens.

| Clean r14 case | Aggregate decode |
|---|---:|
| MTP0 CC1, context 0 | `48.3` tok/s |
| MTP0 CC1, context 65,536 | `46.0` tok/s |

The same PR heads were validated before image assembly with MTP3 and a larger
graph plan:

| Mixed K3/K4 case | Aggregate decode |
|---|---:|
| MTP3 CC1 | `105.23` tok/s |
| MTP3 CC4 | `260.6` tok/s |
| MTP3 CC8 | `390.9` tok/s |

Eight concurrent requests returned 8/8 correct responses, and a 65,536-token
prompt followed by 256 generated tokens completed cleanly. Rank-0 Torch traces
showed the mixed decode kernel falling from `82.38` to `54.11 us/layer`
(`-34.3%`) and mixed prefill from `6.764` to `6.528 ms/layer` (`-3.5%`), with
no added host synchronization or copy. The final clean-image server and both
decode gates reported no traceback or GPU error.

The exact archive is reproducible with
`VLLM_RELEASE_COMPOSITION=reproduce-r14`.

### Final 2026-07-30 r13 gates

The published image was validated without source mounts using its embedded
helper. Package/ABI assertions, release composition checks, and the complete
build gate passed before push. The registry digest and pulled image ID match
the values in the release section above.

The EXL3 gate used the pinned
`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw` snapshot on GPUs 0-3, TP4/DCP4,
seq=1, graph=6, max model length 131,072, and GMU 0.90:

| Case | Stock r11 | r12 | r13 | r13 acceptance | r13 KV capacity |
|---|---:|---:|---:|---:|---:|
| MTP0 CC1 | `44.66` tok/s | `48.48` tok/s (`+8.56%`) | `48.61` tok/s (`+8.85%`) | - | `834,560` |
| MTP3 CC1 | `99.77` tok/s | `100.82` tok/s | `101.92` tok/s | `0.7117` | `485,888` |

Both cases returned the expected correctness answer with zero request errors.
The four matched MTP0 runs, including a GPU-group swap, show that the
consolidated r12 path provides a reproducible 8.56% decode improvement over
the actual stock r11 path; r13 retains it. An earlier r11-labelled development
run already had the consolidated code applied and must not be used as the
stock baseline. MTP3 differences are not claimed as a speedup because its
acceptance rate and run-to-run variance changed. The MTP3 capacity is
intentionally lower because the repeatable target and draft scratch peaks are
now both profiled before KV allocation instead of relying on a one-time
allocator observation.

The SparkInfer scratch/fused/W4A16 gates and the installed LMCache build and
integration suite passed. r13 retains the already validated
r11 DCP, long-prefill, concurrency, and LMCache behavior; the r13
delta changes EXL3/Trellis planning and memory accounting, not those paths.
The exact archive is reproducible with
`VLLM_RELEASE_COMPOSITION=reproduce-r13`.

### Final 2026-07-29 r11 gates

The pushed tag was removed locally and pulled back from Docker Hub. The pull
returned manifest
`sha256:eb4ece3757c03e10764f0900a1366ba4ef63c33560052c976d9ae08457482ff2`,
image ID `sha256:2d68887b1dcd42c62ad90596fcaef0c65496108a8dccdb72be67b699276a12c5`,
and the expected vLLM, SparkInfer, LMCache, and launcher labels.

The matched public-r9 A/B and r11 gate used GPUs 0-7, TP8/DCP1/MTP0, Luke
NVFP4 original dense weights, A16, seq=1, graph=6, and no concurrent model
load:

| Metric | r9 | r11 | Delta |
|---|---:|---:|---:|
| CC1 decode | `87.8` | `87.6` tok/s | `-0.23%` |
| 64k standalone prefill | `5,895` | `5,898` tok/s | `+0.05%` |

Both deltas are measurement noise. r11 therefore has no measurable DCP1
regression.

The lossless DCP auto policy selected query split, CKV gather, exact owner
merge, and a one-layer CKV prefetch. DCP4 selected two indexer shards and DCP8
selected four on the local-ring topology:

| Topology | 64k prefill | Additional gate | Aggregate KV tokens |
|---|---:|---:|---:|
| TP8/DCP1 | `5,898` | baseline | `569,408` |
| TP8/DCP4 | `5,824` | 127k: `5,655`; no collapse | `2,139,392` |
| TP8/DCP8 | `5,761` | safe local DCP rings | `4,264,960` |

TP8/DCP1/MTP3/A16 with online MXFP8 reached `156.4 tok/s` at CC1. Batched
correctness passed at C4, C6, C8, C16, and C32, including `64/64` valid C32
responses with no corrupted output or request failures.

LMCache r11 gates are documented in the prefix-offload section above. The
installed-wheel suite passed `218` tests with `131` skips; the vLLM DCP and
calibration suite passed `14` tests. Release-manifest, fail-fast, helper,
runtime-contract, and patched NCCL 2.30.4 checks also passed.

The exact archive is reproducible with
`VLLM_RELEASE_COMPOSITION=reproduce-r11`. Two review-only commits landed after
the image was measured: vLLM #179 adds test assertions and #194 adds a
docstring. Neither changes the validated runtime behavior.

### Final 2026-07-28 r9 gates

The immutable r9 image was rebuilt with `reproduce-r9`; the archived patches
reproduced the locked vLLM and SparkInfer trees, and all helper, source-tree,
runtime-symbol, XGrammar, LMCache, and image-label checks passed. The final
runtime contains `pytest 8.4.1`. Build-time ABI checks additionally verified
the dynamic NVFP4 writer's per-token scale argument, the exact cache ABI name,
and adaptive fold planning.

The opt-in dynamic NVFP4 path was then started through the embedded helper on
the local NF3 hybrid checkpoint with TP4/DCP1/MTP0, seq=1, graph=6, max model
length 262,144, and GMU 0.96. Startup exposed 273,344 GPU KV-cache tokens with
8.01 GiB available for KV. A 256-token completion and a request with 41,791
actual prompt tokens plus 64 completion tokens both completed successfully,
with no CJK contamination or server error. GMU 0.94 was also checked and was
correctly rejected by the capacity guard because only 6.03 GiB was available
against 7.68 GiB required; this was a configuration limit, not a kernel fault.

The adaptive fold remains exact in both branches. On the earlier matched
TP8/DCP4 capacity POC, `auto` selected two-level folding within the 256 MiB
budget and increased reported KV capacity by 136,704 tokens (`+5.60%`) without
a measurable throughput regression. Larger geometries fall back to exact
streaming carry instead of reserving an unbounded temporary tensor.

### Final 2026-07-28 r7 LMCache gates

The clean pushed r7 image was started through the embedded helper, first as
TP8/DCP1 and then as TP8/DCP4 on GPUs 0-7. The runs were sequential, so no
model load overlapped a measurement. Cold and hit responses were greedy and
produced the same SHA-256 output hash (`9d3dcd66...ba37`). The LMCache HTTP
healthcheck also returned `{"status":"healthy"}`.

| Gate | Cold result | Cache-hit result |
|---|---|---|
| TP8/DCP1 RAM, 8,192-token prompt | `0` cached, `2.161 s` | `8,192` cached, `0.138 s` |
| TP8/DCP4 RAM, 8,192-token prompt | `0` cached, `2.325 s` | `8,192` cached, `0.185 s` |

The preceding r6 gate additionally covered buffered-disk cold restarts for
DCP1 and DCP4 at 12,800 tokens, TP6/DCP3 and TP6/DCP6 at 12,288 tokens with
384-token chunks, and TP8/DCP1/MTP3 concurrent cold/hit requests. r7 changes
LMCache provenance from a build patch to its merged public commit and tightens
the physical-block strategy guard; the filesystem adapter and vLLM/SparkInfer
result trees are unchanged. Build-time tests ran against the installed CUDA
wheel, not an unbuilt source checkout, so the retained compute-performance
tables below remain the relevant speed baseline.

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
docker pull voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28

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
