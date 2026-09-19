# Jovian Judgement wheel-image deployment archive

Use one profile-enabled vLLM image for GLM-5.3-Flash, Qwen3.8-Flash-Next,
DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision and DeepSeek-V4.1-Flash. Select a
model profile; the image supplies its backend, graph, loader and cache defaults.
Do not copy another model's kernel environment or replace the entrypoint.

Status: **implemented** launch interface. The image below is published with
**qualified** native GPU smoke and 91 LMCache contract tests. Model-serving
measurements have separate, explicit image boundaries in
[performance evidence](#performance-evidence). Publication alone does not
qualify every mode, external-cache restore or arbitrary context length.

## Select the image

```bash
export LIL_IMAGE=ghcr.io/local-inference-lab/vllm:jovian-judgement-beta-20260917-9b2a25a581e55533
docker pull "$LIL_IMAGE"
```

This image contains CUDA 13.4.1/PyTorch 2.14 and 68 filesystem layers. Its
registry digest and complete component lock are recorded in the
[publication receipt](../benchmarks/prepared-b12x-contracts/checks/beta-runtime-container-release.json)
and [assembly manifest](../benchmarks/prepared-b12x-contracts/checks/beta-runtime-assembly.json).
The floating alias is `ghcr.io/local-inference-lab/vllm:jovian-judgement-beta`;
use the dated tag above when reproducing this configuration.

Two automated channels use the same Docker recipe and profile interface:

| Channel tag | vLLM source | B12X source |
|---|---|---|
| `ghcr.io/local-inference-lab/vllm:jovian-judgement` | `dev/jovian-judgement` | `master` |
| `ghcr.io/local-inference-lab/vllm:jovian-judgement-beta` | `integration/beta` | `integration/beta` |

The beta composition includes the open fixes in
[issue #773](https://github.com/local-inference-lab/vllm/issues/773).
The non-beta channel is not assumed to contain those fixes or to have the
same performance. A pull does not update an existing container: recreate it
in a maintenance window to use a different image.

Requirements: Linux x86-64, Docker with NVIDIA Container Toolkit, an NVIDIA
driver compatible with the CUDA runtime, sufficient host RAM/storage, and
available GPUs. These recipes target 96-GB RTX PRO 6000 Blackwell cards.
No command changes GPU clocks. Use stock clocks when comparing the measurements.

## Choose a model

Tensor parallelism (TP) is the number of GPUs for one model instance. Expose
exactly that many devices. Decode context parallelism (DCP) defaults to one.
Precision labels describe tensor formats: BF16 is 16-bit brain floating point,
FP8 is 8-bit floating point, NVFP4 is NVIDIA's 4-bit format, and MXFP8 is
microscaling FP8. Mixture-of-experts (MoE) kernels process the routed model
experts. See the [glossary](../GLOSSARY.md) for kernel and cache terminology.

| Model page | `PROFILE` | Default TP | Default speculation | API model name |
|---|---|---:|---|---|
| [GLM-5.3-Flash](../models/glm-5.3-flash.md) | `glm53-flash` | 4 | Off | `GLM-5.3-Flash-NVFP4` |
| [Qwen3.8-Flash-Next](../models/qwen38-flash-next.md) | `qwen38-flash-next` | 1 | MTP3 | `Qwen3.8-Flash-Next` |
| [DeepSeek V4 text](../models/deepseek-v4-flash.md) | `ds4-flash` | 2 | DSpark K5 | `DeepSeek-V4-Flash-0731` |
| [DeepSeek V4 Vision](../models/deepseek-v4-flash-vision.md) | `ds4-vision` | 2 | DSpark K3 | `DeepSeek-V4-Flash-Vision-Exp` |
| [DeepSeek V4.1](../models/deepseek-v4.1-flash.md) | `ds41-flash` | 4 | Adaptive DSpark K7 | `DeepSeek-V4.1-Flash` |

Profiles supply Hugging Face model names, so local checkpoint paths are not
required. `MODEL` overrides a checkpoint **within the selected architecture**;
it does not change `PROFILE`. Target and draft revisions are recorded in
benchmark receipts, not forced into the model names in these commands. DS4
text/Vision profiles also pin compatible checkpoint/remote-code revisions
internally; use `--print-config` to see the resolved revision.

## Start a server

Use Bash. First select the instance and optional serving arguments. This example
selects GLM with three-token Multi-Token Prediction (MTP):

```bash
PROFILE=glm53-flash
GPU_DEVICES=0,1,2,3
TP=4
PORT=8000
SERVE_ARGS=(--mode mtp --draft-tokens 3)
```

The model pages provide replacement values for this block. Then run the same
command for every profile:

```bash
DOCKER_PROFILE_ARGS=()
if [[ "$PROFILE" == ds41-flash ]]; then
  DOCKER_PROFILE_ARGS+=(--ulimit memlock=-1 --security-opt seccomp=unconfined)
fi

docker run -d --name "${PROFILE}-${PORT}" --init --restart unless-stopped \
  --gpus "\"device=${GPU_DEVICES}\"" --network host --ipc host \
  "${DOCKER_PROFILE_ARGS[@]}" \
  -v lil-huggingface:/root/.cache/huggingface \
  -v "${PROFILE}-${PORT}-runtime:/cache" \
  -e PROFILE="$PROFILE" -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP="$TP" -e PORT="$PORT" \
  -e VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=off \
  "$LIL_IMAGE" "${SERVE_ARGS[@]}"
```

The explicit two-shot override preserves the communication setting used in
the linked qualification. B12X one-shot and eligible DMA paths remain available,
with NCCL fallback. The hardware profile's unmodified two-shot threshold is
768 KB for GLM/DS4.1; do not silently substitute it when reproducing the tests.

The DS4.1 permissions allow pinned Engram memory and native io_uring disk
access. `seccomp=unconfined` relaxes Docker's syscall filter: use trusted
images/checkpoints. Other profiles do not need that override here.

The API listens on `0.0.0.0:$PORT`. It is unauthenticated unless configured
otherwise; keep it on a trusted network or behind an authenticated proxy.
Use separate GPUs, API ports, container names and runtime volumes for
simultaneous instances. The model-download volume may be shared.

```bash
docker logs -f "${PROFILE}-${PORT}"
curl -fsS "http://127.0.0.1:${PORT}/health"
curl -fsS "http://127.0.0.1:${PORT}/v1/models"
```

Model loading, preparation and graph capture can take several minutes. The
named volumes survive container removal. Do not delete volumes merely to
change a speculation mode. `--restart unless-stopped` restores the container
after a Docker/host restart if the same GPU and driver configuration remains
available; it does not restore GPU overclock settings.

## Select speculation

Put these arguments in `SERVE_ARGS`. Choose one mode; the launcher constructs
the model-specific speculative configuration. K is the maximum number of
proposed tokens, not a guarantee that every proposal is accepted.

| Profile | Arguments | Qualification scope |
|---|---|---|
| GLM | `--mode off` | Implemented; no no-spec cell in the six-profile wheel comparison |
| GLM | `--mode mtp --draft-tokens 3` | Qualified bounded TP4/DCP1 measurements |
| GLM | `--mode dflash2 --draft-tokens 7` | Qualified bounded TP4/DCP1 measurements; offline MXFP8 draft |
| Qwen | `--mode mtp --draft-tokens 3` | Qualified bounded TP1 text measurements |
| Qwen | `--mode off` | Implemented; not timed in that comparison |
| DS4 text | `--mode dspark --draft-tokens 5` | Qualified bounded TP2 measurements |
| DS4 Vision | `--mode dspark --draft-tokens 3` | Qualified bounded TP2 measurements |
| DS4.1 | `--mode dspark --draft-tokens 7` | Qualified bounded TP4, adaptive verification, RAM Engram |
| DS4 text/Vision or DS4.1 | `--mode off` | Implemented; not timed in that comparison |

Equivalent environment controls before the image name are `SPECULATOR=mtp`,
`MTP_DEPTH=3`, `DFLASH_DEPTH=7`, and `DSPARK_TOKENS=5`. Use only the depth
alias belonging to the selected mode; contradictory aliases fail validation.
`NUM_SPECULATIVE_TOKENS` is the common depth alias. There is no blanket claim
that larger values are supported or faster for every checkpoint.

DS4 standard MTP uses a different checkpoint contract from DSpark. Do not
switch only the mode on the DSpark checkpoint. The model page explains the
explicit, separately unqualified MTP alternative. Qwen and DS4.1 profiles do
not support DFlash2. Model-specific kernel variables remain in the image.

## Cache storage: GPU, LMCache or native offload

Key/value (KV) cache stores request state. It is separate from Qwen's PLE
lookup tables and DS4.1's Engram tables, which are model weights.

| Profile | GPU-local cache | LMCache host RAM / filesystem | Native KV offload |
|---|---|---|---|
| GLM | Default | Implemented, opt-in | Implemented, opt-in |
| DS4 text / Vision | Default | Implemented, opt-in | Unsupported by the profile |
| Qwen | Default | Unsupported by the profile | Unsupported by the profile |
| DS4.1 | Default | Unsupported by the profile | Unsupported by the profile |

The published image passes LMCache package/native contract tests. Full
model-level cold/RAM/filesystem/restart qualification is **not** repeated on
this digest. Historical community-image restore results retain their own
[GLM](../models/glm-5.3-flash-community-r35.md#lmcache-ram-and-filesystem-storage)
and [DS4](../models/ds4-jovian-judgement-r9.md) image identities. Do not present
those results as measurements of this CUDA 13.4 image.

The default `--cache-mode vram` needs no extra arguments or service. For
LMCache, append **one** of these alternatives to the chosen `SERVE_ARGS`:

```bash
# Host-RAM KV cache; 16 GiB maximum and 2 GiB initial allocation.
SERVE_ARGS+=(--cache-mode lmcache --cache-transfer-mode engine_driven
  --cache-l1-gib 16 --cache-l1-init-gib 2 --no-cache-l2-enabled)
```

```bash
# Host RAM plus a 256-GiB filesystem tier inside the persistent /cache volume.
SERVE_ARGS+=(--cache-mode lmcache --cache-transfer-mode engine_driven
  --cache-l1-gib 16 --cache-l1-init-gib 2
  --cache-l2-enabled --cache-l2-gib 256 --cache-directory /cache/lmcache)
```

These capacities are operator choices, not model benchmark conditions. With
`--ipc host`, the host `/dev/shm` capacity applies; Docker `--shm-size` does
not enlarge it. Leave room for the configured pool, staging buffers, model
loading and the operating system. Check host RAM, `/dev/shm` and disk capacity
before enabling a tier. Keep a separate writable cache directory per instance.

The launcher starts a CPU-only LMCache service and coordinates readiness and
shutdown. Engine workers perform the GPU transfers. Default service ports are
API port +10000/+10001/+10002; reserve all four ports and keep the resulting
numbers at or below 65535. Optional controls are `--cache-port`,
`--cache-http-port` and `--cache-metrics-port`.

Legacy environment aliases remain available: `CACHE_MODE=lmcache`,
`LMCACHE_TRANSFER_MODE=engine_driven`, `LMCACHE_L1_SIZE_GB`,
`LMCACHE_L1_INIT_SIZE_GB`, `LMCACHE_L2_ENABLED`, `LMCACHE_L2_MAX_CAPACITY_GB`
and `LMCACHE_L2_ROOT`. DS4 also accepts `LMCACHE_MODE=ram|disk|off`.
Prefer one interface per setting instead of mixing aliases.

GLM's native KV offload is a different backend, not LMCache:

```bash
SERVE_ARGS+=(--cache-mode native --cache-native-gib 64)
```

It remains implemented rather than model-level qualified for this artifact.
Checkpoint/runtime/layout identities namespace external objects; a different
identity must miss and recompute instead of importing incompatible state.

## Prefix-cache defaults

Prefix caching is enabled for all five profiles, including GPU-only mode.

| Model | Recurrent checkpoint policy | Retention interval |
|---|---|---|
| GLM | `request_boundaries` | Native default `0`; engine-driven LMCache explicitly selects `0` |
| Qwen | Native `auto`, selecting request boundaries where supported | Native default `0` |
| DS4 text / Vision | Attention-cache management, not the GLM/Qwen recurrent adapter | `4096` |
| DS4.1 | Attention-cache management, not the GLM/Qwen recurrent adapter | Native default `0`; no matched `0`/`4096` test recorded |

`--mamba-cache-mode align` is **not** the same setting as
`--recurrent-checkpoint-policy aligned`. GLM/Qwen need recurrent state as well
as attention KV. Request boundaries preserve exact supported instruction,
prompt and processed-response endpoints. The interval does not add periodic
recurrent checkpoints in that mode. Unsupported request/configuration cases
retain the existing cache behavior; the setting is not a universal per-token
cache promise.

Under the `aligned` checkpoint policy, a positive interval retains additional
periodic states, while `None` requests dense retention on the available block
grid. It is a token interval, **not** a cache timeout or scheduler batch size;
`--block-size 256` does not establish a 256-token recurrent checkpoint grid.
GLM aligned LMCache uses its cache-object token count, normally 4096. The
launcher also validates the associated layout; do not force 4096 on every model.

## Common overrides and configuration inspection

Use environment variables before the image name, or managed/native arguments
after it. Explicit CLI arguments override environment values. `--settings`
accepts a YAML file with `options` and `environment` mappings; CLI has higher
priority. A JSON option replaces its complete default object unless dotted
fields are used.

| Purpose | Environment | Argument in `SERVE_ARGS` |
|---|---|---|
| TP / DCP | `TP`, `DCP` | `--tensor-parallel-size`, `--decode-context-parallel-size` |
| Checkpoint / API name | `MODEL`, `SERVED_MODEL_NAME` | `--model`, `--served-model-name` |
| Target / DFlash revision | `MODEL_REVISION`, `DFLASH_MODEL_REVISION` | `--revision`, `--draft-revision` |
| Context / concurrent requests | `MAX_MODEL_LEN`, `MAX_NUM_SEQS` | `--max-model-len`, `--max-num-seqs` |
| Scheduler token budget | `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` |
| GPU allocation fraction | `GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` |
| Enable Qwen image/video path | `LANGUAGE_MODEL_ONLY=0` | `--no-language-model-only` |
| DS4.1 model tables | `ENGRAM_TABLE_MEMORY=ram|disk` | `--engram-table-memory ram` or `disk` |

For example, append `--max-model-len 65536 --max-num-seqs 8` to restrict
admission without copying a model's full command. Those limits change the
serving configuration; the benchmark receipts remain authoritative for speed
comparisons. `--env NAME=VALUE` is available for an explicit advanced runtime
override, but normal deployment does not need a list of kernel toggles.

To inspect the resolved command without loading a model or starting a cache
service, run this **instead of** the serving command:

```bash
docker run --rm -e PROFILE="$PROFILE" -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP="$TP" -e PORT="$PORT" -e VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=off \
  "$LIL_IMAGE" --print-config "${SERVE_ARGS[@]}"
```

Inspection reports value origins and redacts credentials. It does not need GPU
access or download checkpoints. `NCCL_GRAPH_FILE` must be unset unless it names
an existing topology file. Use native hardware policy on systems outside this
workstation PCIe profile; its NCCL/transport thresholds are not universal.

Raw `vllm serve`, `python` or `bash` commands bypass profile defaults. Legacy
shell-text `EXTRA_VLLM_ARGS` and ambiguous `BACKEND`/`MODE`/`DS4_*` controls
are rejected rather than silently interpreted. Do not transplant a community
image's full shell command and expect the unified profile to remain active.

The configuration source of truth is the image's versioned
[profile registry](https://github.com/local-inference-lab/blackwell-llm-docker/tree/b3fe0afe1621273059fb19dee1034e2272043a55/runtime/profiles),
[option aliases](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b3fe0afe1621273059fb19dee1034e2272043a55/runtime/options.yaml)
and [hardware profile](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b3fe0afe1621273059fb19dee1034e2272043a55/runtime/hardware/rtx-pro-6000-pcie.yaml).
This guide owns examples and evidence, not another copy of the kernel defaults.
The [complete generated parameter table](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b3fe0afe1621273059fb19dee1034e2272043a55/runtime/generated/parameters.md)
lists the managed controls for every profile. The
[Docker recipe](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b3fe0afe1621273059fb19dee1034e2272043a55/tools/jovian_wheel_runtime/Dockerfile.runtime)
packages that same registry rather than building a different launcher per model.

## Performance evidence

No serving or benchmark run was performed for this documentation update.
The recorded six-profile comparison uses a complete wheel-built image with
vLLM `1048fc5439d` and B12X `d9b572754a8`; its immutable local image identity
and all runtime arguments are in the
[qualification report](../benchmarks/prepared-b12x-serving/).

Stock RTX PRO 6000 Workstation GPUs, same physical cards per pair, temperature
1, context-zero decode median of three warmed 30-second runs. Concurrent
rates are aggregate: C1 means one active request, C8 eight, and C4 four.
Prefill is one sustained uncached nominal-32K window
after warmup, measured from client time to first token, not isolated GPU time.
Scheduler budget is 4096 except Qwen's 6019. LMCache is off. No +6000 numbers
are mixed into this table.

| Model / mode / parallelism | C1 tok/s, reference → wheel image | Concurrent tok/s, reference → wheel image | 32K prefill tok/s, reference → wheel image |
|---|---:|---:|---:|
| GLM MTP3 TP4/DCP1, R35 control | 247.12 → 249.33 (+0.89%) | C8 872.67 → 875.86 (+0.37%) | 15,332 → 15,580 (+1.62%) |
| GLM DFlash2 K7 TP4/DCP1, R35 control | 211.23 → 219.83 (+4.07%) | C8 673.26 → 717.84 (+6.62%) | 15,538 → 15,734 (+1.26%) |
| DS4.1 DSpark K7 TP4/DCP1, R38 control | 250.87 → 256.59 (+2.28%) | C8 808.17 → 830.82 (+2.80%) | 20,079 → 20,234 (+0.77%) |
| DS4 text DSpark K5 TP2/DCP1, R9 control | 191.35 → 190.12 (−0.64%) | C8 653.53 → 669.56 (+2.45%) | 13,527 → 13,863 (+2.48%) |
| DS4 Vision DSpark K3 TP2/DCP1, R9 control | 169.92 → 185.49 (+9.17%) | C4 411.49 → 423.11 (+2.83%) | 10,388 → 10,615 (+2.19%) |
| Qwen MTP3 TP1, R35 control | 172.92 → 190.11 (+9.94%) | C8 664.89 → 689.93 (+3.77%) | 15,387 → 15,162 (−1.46%) |

DS4 text/Vision use top-p 1 in these cells, not their profile's .95 default.
Qwen uses top-p .95/top-k 20 and an explicit eight-GiB KV allocation. DS4.1 uses
RAM Engram and max sequences 32 rather than the profile's disk/four-request
defaults. These are measurement conditions, not silently revised defaults.

GLM's five-run Sieve medians are 326.15 → 325.75 tok/s (−0.12%) for MTP3 and
458.14 → 461.42 (+0.72%) for DFlash2. Those results do not imply that all generated
programs were executed or correct. Speculative acceptance varies; emitted
tok/s is not the same as verifier execution rate.

The published image also contains the API, prepared-selection and PCIe lookup
changes recorded in the [component comparisons](../benchmarks/prepared-b12x-contracts/).
Those checks use separately identified image pairs; their measured cells are
listed on the model pages. They are not relabeled as a repeated six-model
matrix of the registry digest above. In particular, Qwen's three-window repeat
confirms **15,258 → 15,031 tok/s (−1.49%)** for prefill; the gap remains unresolved.

Historical community-image launches and clock-specific results remain in the
model pages' archive links. DCP4, alternate precision, arbitrary draft depth,
Qwen Vision, whole-model disk Engram throughput and long-context external
restore require their own qualification; absence of a measurement is not zero
regression.
