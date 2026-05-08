# GLM-5 on RTX PRO 6000 Blackwell (SM120)

> Current note, 2026-05-08: the older sections below document the original
> GLM-5/SGLang bring-up and historical vLLM limitations. The current working
> path for GLM-5.1 is vLLM with B12X sparse MLA, FlashInfer CUTLASS MoE,
> GLM-5.1 MTP, and the DCP coherence fixes described in this section.

## Current vLLM GLM-5.1 DCP Benchmark (2026-05-08)

### Docker image

Current pushed image:

```bash
voipmonitor/vllm:glm51-kimi-comm-20260508
```

The container entrypoint serves `lukealonso/GLM-5.1-NVFP4-MTP` as `GLM-5`.

This image includes the GLM-5.1 B12X sparse MLA port, DCP full-graph
block-table coherence fix, inline split-decode LSE support, FlashInfer CUTLASS
MoE backend, patched NCCL, and the communication selector used by the Kimi and
GLM tests.

### Recommended launch

This is the canonical launch wrapper used for the measurements below. Defaults
are DCP1, MTP enabled, vLLM C++ PCIe allreduce enabled.

```bash
cat >/tmp/glm <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:glm51-kimi-comm-20260508}"
NAME="${NAME:-glm51-vllm}"
PORT="${PORT:-5288}"
DCP_SIZE="${DCP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.865}"
GLM51_DISABLE_MTP="${GLM51_DISABLE_MTP:-0}"

if [[ -z "${VLLM_ENABLE_PCIE_ALLREDUCE+x}" ]]; then
  if [[ "${DCP_SIZE}" == "1" ]]; then
    VLLM_ENABLE_PCIE_ALLREDUCE=1
  else
    VLLM_ENABLE_PCIE_ALLREDUCE=0
  fi
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true

exec docker run -d \
  --gpus all \
  --ipc=host \
  --network host \
  --privileged \
  --name "${NAME}" \
  --entrypoint /usr/local/bin/run-glm51-vllm \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_IB_DISABLE=1 \
  -e USE_NCCL_XML=0 \
  -e PORT="${PORT}" \
  -e DCP_SIZE="${DCP_SIZE}" \
  -e MAX_MODEL_LEN=202752 \
  -e MAX_NUM_SEQS=64 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=256 \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e GLM51_DISABLE_MTP="${GLM51_DISABLE_MTP}" \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_B12X_MLA_DECODE_INLINE_LSE=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE}" \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=1 \
  -e VLLM_B12X_MLA_SPEC_SERIAL_DECODE=0 \
  -e VLLM_MTP_RETURN_NORMALIZED_HIDDEN=1 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_ACC=1.0 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE=1.0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e XDG_CACHE_HOME=/cache/jit \
  -e CUDA_CACHE_PATH=/cache/jit \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/vllm-glm51/jit:/cache/jit" \
  -v "${HOME}/.cache/vllm-glm51/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "${HOME}/.cache/vllm-glm51/triton:/root/.cache/triton" \
  -v "${HOME}/.cache/vllm-glm51/torchinductor:/root/.cache/torchinductor" \
  -v "${HOME}/.cache/vllm-glm51/vllm:/root/.cache/vllm" \
  "${IMAGE}"
EOF
chmod +x /tmp/glm
PORT=5264 DCP_SIZE=1 /tmp/glm
```

Useful variants:

```bash
# Disable MTP.
PORT=5264 DCP_SIZE=1 GLM51_DISABLE_MTP=1 /tmp/glm

# DCP4 baseline used in the table.
PORT=5264 DCP_SIZE=4 VLLM_ENABLE_PCIE_ALLREDUCE=0 /tmp/glm

# DCP4 explicit PCIe allreduce A/B.
PORT=5264 DCP_SIZE=4 VLLM_ENABLE_PCIE_ALLREDUCE=1 /tmp/glm

# DCP8 MTP on was measured at lower memory utilization.
PORT=5264 DCP_SIZE=8 GPU_MEMORY_UTILIZATION=0.82 /tmp/glm
```

### Benchmark method

Decode benchmark:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5264 \
  --model GLM-5 \
  --concurrency 1,2,4,8,16,32,64 \
  --contexts 0,16k,32k,64k,128k \
  --duration 10 \
  --decode-warmup-seconds 0 \
  --skip-prefill \
  --kv-budget <reported_gpu_kv_cache_tokens> \
  --display-mode plain \
  --no-hw-monitor
```

Before every measured decode run, a separate cc1 ctx0 30 second warmup was run.
Cold prefill was measured separately by sending unique prompts with
`max_tokens=1`, so prefix-cache hits do not contaminate TTFT.

Result directory:

```bash
/root/bench-results/glm5-vllm-20260508/dcp-matrix-20260508T040740Z
```

### KV cache budgets

| DCP | MTP | mem util | AR | GPU KV cache tokens |
|---:|:---:|---:|---:|---:|
| 1 | on | 0.865 | 1 | 371456 |
| 1 | off | 0.865 | 1 | 396288 |
| 2 | on | 0.865 | 0 | 743040 |
| 2 | off | 0.865 | 0 | 792576 |
| 4 | on | 0.865 | 0 | 1485824 |
| 4 | off | 0.865 | 0 | 1585152 |
| 8 | on | 0.820 | 0 | 2381824 |
| 8 | off | 0.865 | 0 | 3170304 |

### Decode throughput, MTP on

Columns are concurrency `1 / 2 / 4 / 8 / 16 / 32 / 64`.

#### DCP1, MTP on, AR=1, mem=0.865

| Context | tok/s |
|---:|---|
| 0 | 90.6 / 157.7 / 271.1 / 397.5 / 622.5 / 882.6 / 1165.1 |
| 16k | 85.9 / 154.2 / 227.8 / 342.3 / 468.7 / skip / skip |
| 32k | 83.9 / 143.8 / 216.7 / 289.2 / skip / skip / skip |
| 64k | 79.1 / 131.9 / 197.1 / skip / skip / skip / skip |
| 128k | 78.0 / 119.0 / skip / skip / skip / skip / skip |

#### DCP2, MTP on, AR=0, mem=0.865

| Context | tok/s |
|---:|---|
| 0 | 74.8 / 132.8 / 220.9 / 312.7 / 499.7 / 677.3 / 970.5 |
| 16k | 67.8 / 126.9 / 203.7 / 285.4 / 442.1 / 632.5 / skip |
| 32k | 75.3 / 121.6 / 182.6 / 266.1 / 418.5 / skip / skip |
| 64k | 69.4 / 119.6 / 174.5 / 256.4 / skip / skip / skip |
| 128k | 68.2 / 112.5 / 171.9 / skip / skip / skip / skip |

#### DCP4, MTP on, AR=0, mem=0.865

| Context | tok/s |
|---:|---|
| 0 | 71.3 / 122.6 / 195.4 / 280.1 / 396.1 / 549.9 / 753.2 |
| 16k | 72.6 / 115.2 / 176.0 / 268.2 / 370.8 / 526.5 / 701.7 |
| 32k | 72.1 / 122.5 / 166.2 / 243.8 / 369.2 / 526.3 / skip |
| 64k | 68.6 / 105.1 / 166.8 / 235.6 / 352.4 / skip / skip |
| 128k | 65.9 / 111.9 / 158.5 / 233.7 / skip / skip / skip |

#### DCP8, MTP on, AR=0, mem=0.820

| Context | tok/s |
|---:|---|
| 0 | 65.7 / 103.9 / 161.4 / 214.4 / 291.8 / 417.5 / 595.9 |
| 16k | 62.8 / 98.2 / 149.5 / 199.7 / 287.4 / 365.6 / 455.4 |
| 32k | 63.7 / 96.9 / 140.6 / 205.0 / 285.0 / 378.8 / 461.4 |
| 64k | 62.5 / 98.6 / 148.5 / 211.8 / 281.9 / 367.2 / skip |
| 128k | 56.9 / 95.8 / 156.8 / 200.6 / 273.4 / skip / skip |

### Decode throughput, MTP off

Columns are concurrency `1 / 2 / 4 / 8 / 16 / 32 / 64`.

#### DCP1, MTP off, AR=1, mem=0.865

| Context | tok/s |
|---:|---|
| 0 | 50.0 / 84.7 / 153.4 / 260.9 / 426.2 / 649.0 / 1068.7 |
| 16k | 48.1 / 81.4 / 145.5 / 244.0 / 378.5 / skip / skip |
| 32k | 47.6 / 79.5 / 141.5 / 234.4 / skip / skip / skip |
| 64k | 46.7 / 78.4 / 136.7 / skip / skip / skip / skip |
| 128k | 45.6 / 75.5 / skip / skip / skip / skip / skip |

#### DCP2, MTP off, AR=0, mem=0.865

| Context | tok/s |
|---:|---|
| 0 | 42.3 / 72.9 / 136.7 / 230.4 / 367.3 / 546.6 / 896.4 |
| 16k | 41.7 / 70.7 / 132.2 / 221.7 / 346.5 / 508.1 / skip |
| 32k | 41.3 / 70.1 / 129.9 / 219.4 / 340.2 / skip / skip |
| 64k | 40.9 / 68.5 / 127.1 / 209.8 / skip / skip / skip |
| 128k | 40.2 / 67.9 / 123.5 / skip / skip / skip / skip |

#### DCP4, MTP off, AR=0, mem=0.865

| Context | tok/s |
|---:|---|
| 0 | 42.0 / 71.3 / 131.0 / 215.3 / 327.3 / 483.0 / 762.5 |
| 16k | 41.3 / 69.1 / 126.8 / 206.6 / 309.7 / 464.0 / 705.2 |
| 32k | 41.1 / 68.7 / 125.6 / 202.6 / 308.2 / 451.4 / skip |
| 64k | 40.6 / 67.9 / 122.8 / 200.3 / 301.8 / skip / skip |
| 128k | 40.0 / 66.5 / 120.7 / 195.4 / skip / skip / skip |

#### DCP8, MTP off, AR=0, mem=0.865

This run is not valid for long-context reporting. ctx0 completed, but long
context cells produced benchmark errors/zero-token cells and the server refused
connections immediately afterward, so cold prefill did not run.

| Context | tok/s |
|---:|---|
| 0 | 40.2 / 66.7 / 117.5 / 184.3 / 278.0 / 403.5 / 584.8 |
| 16k | 39.3 / 64.5 / 112.9 / 176.4 / 266.9 / 381.4 / invalid |
| 32k | 39.1 / invalid / invalid / invalid / invalid / invalid / invalid |
| 64k | 39.0 / invalid / invalid / invalid / invalid / invalid / skip |
| 128k | 37.9 / invalid / invalid / invalid / invalid / skip / skip |

### Cold prefill throughput

Cold prefill sends unique prompts and reports prompt tokens per second. Columns
are contexts `8k / 16k / 32k / 64k / 128k`.

| DCP | MTP | AR | mem util | tok/s |
|---:|:---:|---:|---:|---|
| 1 | on | 1 | 0.865 | 4333 / 4296 / 3288 / 3711 / 3170 |
| 1 | off | 1 | 0.865 | 4398 / 4354 / 4191 / 3800 / 3224 |
| 2 | on | 0 | 0.865 | 3690 / 3637 / 3542 / 3396 / 3133 |
| 2 | off | 0 | 0.865 | 3762 / 3701 / 3602 / 3455 / 3191 |
| 4 | on | 0 | 0.865 | 2769 / 2570 / 2458 / 2382 / 2096 |
| 4 | off | 0 | 0.865 | 2818 / 2612 / 2499 / 2423 / 2138 |
| 8 | on | 0 | 0.820 | 1887 / 1678 / 1511 / 1435 / 1391 |
| 8 | off | 0 | 0.865 | invalid: server crashed/refused connections |

### DCP4 PCIe allreduce A/B

DCP4 was re-run with `VLLM_ENABLE_PCIE_ALLREDUCE=1` after the baseline matrix.
It is not a clear global win. ctx0 single-stream improves, cc32 is mixed, and
prefill is effectively unchanged.

| Mode | AR | ctx0 cc1 | ctx0 cc32 | ctx0 cc64 | 128k cc1 | Prefill 8k/128k |
|---|---:|---:|---:|---:|---:|---|
| DCP4 MTP on | 0 | 71.3 | 549.9 | 753.2 | 65.9 | 2769 / 2096 |
| DCP4 MTP on | 1 | 79.9 | 535.4 | 786.1 | 67.0 | 2766 / 2095 |
| DCP4 MTP off | 0 | 42.0 | 483.0 | 762.5 | 40.0 | 2818 / 2138 |
| DCP4 MTP off | 1 | 44.5 | 486.7 | 756.6 | 42.3 | 2814 / 2133 |

### Current interpretation

- `DCP1 + MTP` remains the fastest short-context decode profile.
- `DCP2` roughly doubles KV cache while preserving reasonable decode
  throughput.
- `DCP4` gives much larger KV cache, but decode and prefill cost is visible.
- `DCP8 + MTP` works and gives the largest valid long-context coverage tested
  here, but decode and prefill are substantially slower.
- `DCP8 + MTP off` at `mem=0.865` is not a stable long-context configuration
  in this run and needs separate debugging before reporting as supported.
- For DCP4, `AR=1` is useful for cc1 and sometimes cc64, but not a clear win
  for cc32. Keep `AR=0` as the DCP>1 default until a per-shape selector exists.

## Table of Contents

- [Current vLLM GLM-5.1 DCP Benchmark (2026-05-08)](#current-vllm-glm-51-dcp-benchmark-2026-05-08)
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [NVFP4 Quantization](#nvfp4-quantization)
- [Why SGLang Only (vLLM Does Not Work)](#why-sglang-only-vllm-does-not-work)
- [BF16 KV Cache Mandatory](#bf16-kv-cache-mandatory)
- [NCCL Environment Variables](#nccl-environment-variables)
- [Docker Images](#docker-images)
- [SGLang Launch Commands](#sglang-launch-commands)
- [MTP / Speculative Decoding](#mtp--speculative-decoding)
- [FlashInfer CUTLASS Race Condition Fix](#flashinfer-cutlass-race-condition-fix)
- [Power Consumption](#power-consumption)
- [Benchmark Results](#benchmark-results)
- [Memory Usage](#memory-usage)
- [TP/PP Configurations](#tppp-configurations)
- [SM120 Architecture Limitations](#sm120-architecture-limitations)
- [All Errors and Fixes](#all-errors-and-fixes)
- [Related PRs](#related-prs)

---

## Overview

| Parameter | Value |
|-----------|-------|
| Model | `zai-org/GLM-5` |
| Total parameters | 744B |
| Active parameters | 40B |
| Architecture | MoE with DeepSeek Sparse Attention (DSA), MLA-based |
| Experts | 256 total, 8 activated per token |
| MTP layer | Layer 78 (~19 GB in BF16 precision) |
| SWE-bench Verified | 77.8 (vs Qwen 72.0) |
| Inference engine | **SGLang only** (vLLM does not work on SM120) |
| Minimum GPUs | **8x RTX PRO 6000** (768 GB VRAM) |

GLM-5 is a 744B MoE model with DeepSeek Sparse Attention. On SM120 (RTX PRO 6000 Blackwell), SGLang bypasses all DSA backends and runs GLM-5 as if it were a DeepSeek V3.1 model -- using MLA kernels that ignore the sparsity mask. This is "backwards compatible" since the training-time indexer would have masked out irrelevant tokens, so computing full attention is slightly wasteful but not accuracy-degrading.

---

## Hardware Requirements

### Minimum: 8x RTX PRO 6000 (768 GB VRAM)

- NVFP4 weights: ~440 GB (57.06 GB per GPU across 8 GPUs)
- **Cannot fit on 4x RTX PRO 6000** (only 384 GB total VRAM)
- Minimum viable: **6x RTX PRO 6000** using `--tp 2 --pp 3`

### Reference configurations

| Component | Details |
|-----------|---------|
| GPUs | 8x NVIDIA RTX PRO 6000 Blackwell 96 GB (SM120) |
| Total VRAM | 768 GB |
| RAM | 1.5 TB recommended |
| CPU topology | 2x NUMA nodes: GPU0-3 on NUMA0, GPU4-7 on NUMA1 |
| Tested CPUs | Genoa (EPYC 9004) and Turin (EPYC 9005) |
| Driver | 590.48.01 (CUDA 13.1) |

---

## NVFP4 Quantization

### Available checkpoints

| Checkpoint | MTP | Disk Size | Notes |
|---|---|---|---|
| `lukealonso/GLM-5-NVFP4` | **No** | ~410 GB | Original quant, no MTP weights |
| `festr2/GLM-5-NVFP4-MTP` | **Yes** (BF16) | ~410 GB + 19 GB | MTP layer 78 restored from BF16 checkpoint |
| `QuantTrio/GLM-5-AWQ` | -- | ~420 GB | Fails with OOM during weight loading; NVFP4 is superior |

### Quantization details

- 4-bit blockwise with FP8 scales via NVIDIA Model Optimizer
- SGLang flag: `--quantization modelopt_fp4`
- VRAM for weights: ~57 GB per GPU on TP8
- MMLU accuracy: **0.873** (official BF16 benchmark: 0.877, gap of only -0.004)

### Accuracy concern at long context

~10% accuracy drops observed at 100K+ context lengths in MMLU testing.

---

## Why SGLang Only (vLLM Does Not Work)

**GLM-5 does NOT work on vLLM for SM120** as of 2026-03-08.

The error:
```
ValueError: No valid attention backend found for cuda with
AttentionSelectorConfig(head_size=576, dtype=torch.bfloat16, kv_cache_dtype=auto,
use_mla=True, use_sparse=True, ...)
```

Root causes:
1. No vLLM attention backend supports MLA + sparse attention + SM120 simultaneously
2. GLM-5 uses `qk_nope_head_dim == 192` (FlashInfer MLA requires 128)
3. NVFP4 support keeps breaking in vLLM

SGLang works by bypassing all DSA (DeepSeek Sparse Attention) backends entirely and running GLM-5 in non-DSA mode using FlashInfer FA2 MLA kernels that are SM120-compatible.

Grimulkan has a plan to port GLM-5 to vLLM:
1. Pull FlashInfer FA2 bf16 and XQA fp8 MLA kernels from SGLang into vLLM
2. Wire GLM-5 in non-DSA mode
3. Fix NVFP4 MoE GEMM + DCP compatibility
4. Use normal FA2 for prefill
5. Enable MTP head (already exists in vLLM)

---

## BF16 KV Cache Mandatory

**FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) does NOT work on SM120.** It produces garbled output or emits 1 token and stops.

The root cause is that luke had a local patch for KV scales in the FlashInfer backend (passing FP8 dequantization scales in the ragged+paged split path). Without those scales, the cached KV prefix is read back without undoing the scale division.

**Always use:**
```bash
--kv-cache-dtype bf16
```

This limits practical context to ~200K tokens (vs potentially more with FP8), but is the only working option.

---

## NCCL Environment Variables

### Required NCCL settings

```bash
export NCCL_IB_DISABLE=1               # No InfiniBand
export NCCL_P2P_LEVEL=SYS              # or PHB for same-NUMA only
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
```

### NCCL graph optimization (for Genoa/Turin with cross-NUMA)

```bash
wget https://www.voipmonitor.org/nccl_graph_opt.xml -O /mnt/nccl_graph_opt.xml
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
```

This tricks NCCL into using the low-latency (LL) protocol for small messages across NUMA nodes. Measured **+11% throughput improvement** on Genoa with 2 NUMA nodes and 4 GPUs per node.

Alternative (simpler but less optimal): `export NCCL_PROTO=LL`

### Other environment variables

```bash
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
export NVIDIA_TF32_OVERRIDE=1
## Do NOT set expandable_segments if using --enable-pcie-oneshot-allreduce (crashes IPC handles)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_CUMEM_HOST_ENABLE=0

# Critical for GLM-5:
export SGLANG_ENABLE_JIT_DEEPGEMM=0     # DeepGemm not supported on SM120
export SGLANG_ENABLE_DEEP_GEMM=0        # Fully disable DeepGemm fallback
export SGLANG_ENABLE_SPEC_V2=True       # MANDATORY for MTP (see MTP section)
```

### Kernel boot parameter

If NCCL P2P hangs occur:
```
iommu=pt
amd_iommu=pt    # on AMD platforms
```

---

## Docker Images

### Recommended: voipmonitor/sglang:cu130

```bash
docker pull voipmonitor/sglang:cu130
```

Pre-built image with all SM120 patches:
- SGLang with FlashInfer compiled from source (GDC fix PR #2913)
- PyTorch 2.11 cu130, CUTLASS MoE kernels
- b12x TP-only NVFP4 MoE/GEMM backend (lukealonso)
- PCIe oneshot allreduce (lukealonso)
- NCCL graph XML baked in at `/etc/nccl_graph_opt.xml`
- Pre-generated Triton MoE configs for RTX PRO 6000
- Model profiles in `/profiles/`

Source: [github.com/voipmonitor/blackwell-llm-docker](https://github.com/voipmonitor/blackwell-llm-docker)

### Docker run command

```bash
docker run --gpus all \
    --ipc=host --shm-size=8g --network=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v jit-cache:/cache/jit \
    voipmonitor/sglang:cu130 bash
```

### Alternative images

| Image | Notes |
|---|---|
| `lmsysorg/sglang:dev-cu13` | Official SGLang nightly, CUDA 13.0. May need `pip install transformers>=5.3,<5.4`. |

---

## SGLang Launch Commands

### Recommended: Docker run with MTP + b12x

```bash
docker run --gpus all \
  --ipc=host --shm-size=8g --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_ENABLE_DEEP_GEMM=0 \
  -e NCCL_GRAPH_FILE=/etc/nccl_graph_opt.xml \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  voipmonitor/sglang:cu130 \
  python3 -m sglang.launch_server \
    --model-path festr2/GLM-5-NVFP4-MTP \
    --served-model-name glm-5 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --tensor-parallel-size 8 \
    --quantization modelopt_fp4 \
    --kv-cache-dtype bf16 \
    --trust-remote-code \
    --cuda-graph-max-bs 32 \
    --max-running-requests 64 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 16384 \
    --host 0.0.0.0 --port 5000 \
    --disable-custom-all-reduce \
    --enable-metrics \
    --sleep-on-idle \
    --attention-backend flashinfer \
    --fp4-gemm-backend b12x \
    --moe-runner-backend b12x \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 4 \
    --speculative-num-draft-tokens 6 \
    --speculative-eagle-topk 1 \
    --json-model-override-args '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
```

For high-concurrency serving (4+ users), switch MoE backend:
```
--fp4-gemm-backend cutlass --moe-runner-backend cutlass
```

> **Note:** In TP-only mode (no expert parallel), `flashinfer_cutedsl` and `cutlass` both dispatch to the same `cutlass_moe_fp4()` code path. The cutedsl-specific kernel (`grouped_gemm_nt_masked`) is only used in EP/masked mode. Use `cutlass` for clarity.

### Launch parameter reference

| Parameter | Reason |
|-----------|--------|
| `--quantization modelopt_fp4` | Required for NVFP4 checkpoint |
| `--kv-cache-dtype bf16` | **Mandatory on SM120** -- fp8_e4m3 produces garbled output |
| `--tp 8` | All 8 GPUs required; model is 57 GB/GPU before KV cache |
| `--attention-backend flashinfer` | Architecture-independent; flashmla/trtllm are SM90/SM100 only |
| `--moe-runner-backend cutlass` | Fastest for MTP speculative decoding |
| `--disable-custom-all-reduce` | Required for 8 GPU cross-socket — PCIe oneshot hurts on cross-NUMA due to Infinity Fabric barrier latency. Use `--enable-pcie-oneshot-allreduce` only for ≤4 GPU same-NUMA setups. |
| `--mem-fraction-static 0.85-0.92` | Leave 7-15 GB for CUDA workspace per GPU |
| `SGLANG_ENABLE_JIT_DEEPGEMM=0` | DeepGemm not supported on SM120 |
| `SGLANG_ENABLE_DEEP_GEMM=0` | Fully disables DeepGemm fallback path |
| `SGLANG_ENABLE_SPEC_V2=True` | **Critical for MTP** -- without it, NEXTN falls back to EAGLE and loads model twice (OOM) |

---

## MTP / Speculative Decoding

### Configuration

```bash
# Environment variable (MANDATORY):
SGLANG_ENABLE_SPEC_V2=True

# Launch flags:
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-eagle-topk 1
```

**WARNING:** `SGLANG_ENABLE_SPEC_V2=True` is **mandatory**. Without it, SGLang silently converts NEXTN to EAGLE and loads the full model a second time as a draft model -- instant OOM (57 GB x 2 = 114 GB per GPU on a 96 GB card).

### MTP model checkpoint

- **Use:** `festr2/GLM-5-NVFP4-MTP` (HuggingFace)
- Created by Festr by restoring MTP heads from BF16 checkpoint to the NVFP4 quant
- MTP layer is layer 78, kept in BF16 precision (~19 GB)
- FP8 MTP is possible but not recommended (decreases accept rate)
- The original `lukealonso/GLM-5-NVFP4` does **not** include MTP weights

### Performance impact

MTP roughly **doubles throughput** over the non-MTP baseline:

- Accept rate: 0.55-0.94 (varies by context)
- Accept length: 2.19-2.80 tokens
- Without MTP: 35-50 tok/s
- With MTP: 70-105 tok/s

### MoE runner backend comparison

| Backend | Best for | Notes |
|---|---|---|
| `b12x` | conc 1-2 (single user) | Fused kernel, lowest latency. 95 tok/s at conc=1 with MTP. |
| `cutlass` | conc 4+ | Better batching efficiency. 962 tok/s at conc=32 with MTP. |
| `flashinfer_cutlass` | conc 4+ | FlashInfer's fused C++ kernel. Single call, auto tile tuning. |
| `deep_gemm` | -- | Falls back to cutlass (not supported on SM120) |

> **Note:** `flashinfer_cutedsl` and `cutlass` use the same `cutlass_moe_fp4()` code path in TP-only mode. The cutedsl masked kernel is only used with expert parallel.

---

## FlashInfer CUTLASS Race Condition Fix

A race condition in the FlashInfer CUTLASS FP4 GEMM kernel produces NaN values, causing crashes.

### Symptoms

```
/pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:112: _assert_async_cuda_kernel:
Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed.
```

Or: CUDA device-side assert triggered in `eagle_worker_v2.py:510 _zero_fill_draft_kv_for_cached_prefix`

### Root cause

FlashInfer CUTLASS FP4 GEMM kernel race condition. Fix: https://github.com/flashinfer-ai/flashinfer/pull/2716

### Workarounds (in order of preference)

1. **Upgrade to CUTLASS 4.4.1** and rebuild FlashInfer JIT cache (`rm -rf /cache/jit/*`). Use `voipmonitor/llm-pytorch-blackwell:nightly` which includes this fix.
2. Use `--fp4-gemm-backend flashinfer_cudnn` instead of flashinfer_cutlass
3. Use `--enable-nan-detection` (prevents crash but may produce garbage tokens)
4. Apply luke's sampler patch (validates/fixes probabilities before multinomial sampling)

**Important:** When upgrading Docker images, the old JIT kernel cache must be wiped for the fix to take effect:

```bash
rm -rf /cache/jit/*
```

---

## Power Consumption

GLM-5 draws significantly more power than other models:

| Phase | Power per Card | Notes |
|---|---|---|
| Decode | ~300W | Sustained |
| Prefill | 400-600W | Peaking at **640W** observed |
| Prefill (all 8 cards) | 600W each | All cards hit 600W simultaneously |

Plan cooling and PSU capacity accordingly. An 8-GPU setup draws up to **4,800W** from GPUs alone during prefill.

---

## Benchmark Results

### Decode throughput — MTP ON, 8x RTX PRO 6000, TP=8 (March 2026)

**b12x** (`--moe-runner-backend b12x`):

```
ctx\conc     1      2      4      8     16     32     64   128
   0      95.1  173.4  186.5  266.8  363.4  709.5  924.3    -
  16k     83.0  150.4  161.2  225.8   skip   skip   skip    -
  32k     77.7  130.3  143.9   skip   skip   skip   skip    -
  64k     73.4  120.5   skip   skip   skip   skip   skip    -
 128k     67.7   skip   skip   skip   skip   skip   skip    -
```

**cutlass** (`--moe-runner-backend cutlass`):

```
ctx\conc     1      2      4      8     16     32    64   128
   0      93.6  161.2  242.3  376.8  630.9  961.6     -    -
  16k     83.0  137.9  200.2  316.2   skip   skip     -    -
  32k     70.0  119.2  191.8   skip   skip   skip     -    -
  64k     73.2  112.4   skip   skip   skip   skip     -    -
 128k     67.6   skip   skip   skip   skip   skip     -    -
```

**cutlass run 2** (`--moe-runner-backend flashinfer_cutedsl` — same `cutlass_moe_fp4()` code path in TP-only):

```
ctx\conc     1      2      4      8     16     32      64   128
   0      93.7  156.9  243.9  392.4  590.1  994.5  1424.6    -
  16k     82.9  141.4  203.6  331.5   skip   skip    skip    -
  32k     73.4  122.7  186.5   skip   skip   skip    skip    -
  64k     66.3  112.6   skip   skip   skip   skip    skip    -
 128k     54.8   skip   skip   skip   skip   skip    skip    -
```

> Both "cutlass" and "cutedsl" benchmarks above execute the same kernel (`cutlass_moe_fp4`). Minor differences are from different input scale computation and run-to-run variance.

### Decode throughput — MTP OFF, 8x RTX PRO 6000, TP=8

**b12x:**

```
ctx\conc     1      2      4      8     16     32     64   128
   0      51.6   99.5  180.2  301.9  315.2  451.2  590.7    -
  16k     44.3   84.3  143.1  228.3   skip   skip   skip    -
  32k     42.4   73.6  126.4   skip   skip   skip   skip    -
  64k     36.2   64.2   skip   skip   skip   skip   skip    -
 128k     31.1   skip   skip   skip   skip   skip   skip    -
```

**cutlass:**

```
ctx\conc     1      2      4      8     16     32      64   128
   0      50.4   95.2  164.2  276.9  427.4  659.7  1010.6    -
  16k     43.6   81.1  131.6  209.4   skip   skip    skip    -
  32k     41.7   71.0  117.4   skip   skip   skip    skip    -
  64k     35.5   62.0   skip   skip   skip   skip    skip    -
 128k     30.6   skip   skip   skip   skip   skip    skip    -
```

### Summary — best backend by scenario

| Concurrency | MTP OFF | MTP ON |
|-------------|---------|--------|
| 1-2 users | b12x (52-100 tok/s) | **b12x (95-173 tok/s)** |
| 4-8 users | b12x (180-302) | cutlass (244-392) |
| 16-32 users | cutlass (427-660) | cutlass (590-995) |
| 64 users | cutlass (1011) | cutlass (1425) |

MTP doubles single-user throughput: 51→95 tok/s (+84%).

### PCIe Oneshot AllReduce Impact (MTP OFF, b12x, TP=8, conc=1, context=0)

| Config | tok/s | Notes |
|---|---|---|
| b12x + PCIe oneshot | **56.6** | +7.2% vs NCCL-only baseline |
| b12x (NCCL only) | 52.8 | Baseline without PCIe oneshot |

PCIe oneshot allreduce provides a consistent **+7% decode throughput** improvement, matching the gains seen on Qwen3.5 TP=4. Auto crossover: 48 KB on 8 GPUs.

### Prefill throughput

- Single batch prefill: **~4,000 tok/s**

### Startup time

| Phase | Duration |
|---|---|
| Model load (multithread, 16 threads) | ~36 seconds |
| CUDA graph capture | ~208 seconds |
| **Total startup** | **~7-8 minutes** |

![GLM-5 MTP at 0 context](../images/1477791305089421312_image.png)
![GLM-5 MTP at 100K context](../images/1477791594039083018_image.png)

---

## Memory Usage

### Per-GPU breakdown (8x TP8, NVFP4 + MTP)

| Component | Size |
|-----------|------|
| Weights (NVFP4) | 57.06 GB per GPU |
| KV Cache (bf16) | 29.32 GB per GPU |
| Total allocated | ~86.38 GB per GPU |
| Available after allocation | 7.43-7.53 GB per GPU |

### KV cache capacity

| mem-fraction-static | Total KV Tokens | Max Context |
|---|---|---|
| 0.92 | 314,304 | ~202,752 |
| 0.85 | Slightly less | ~190,000 |

BF16 KV cache limits practical context to ~200K tokens. FP8 KV cache would allow more but is broken on SM120.

---

## TP/PP Configurations

| GPUs | Configuration | Status |
|------|--------------|--------|
| 8x | `--tp 8` | **Primary configuration**, well tested |
| 6x | `--tp 2 --pp 3` | Reported viable, less tested |
| 4x | N/A | **Too large** -- NVFP4 weights alone are 440 GB |

---

## SM120 Architecture Limitations

### What SM120 lacks vs SM90/SM100

- **No TMEM** (Tensor Memory)
- **No TCGEN05** instructions
- **No WGMMA** instructions
- Shorter shared memory / register file
- Cannot run DeepGemm (requires WGMMA for SM90, TCGEN05 for SM100)
- Cannot run FlashAttention 3+ (based on TMEM/TCGEN05)
- Cannot run FlashMLA Sparse natively
- Limited to FlashAttention 2 via SM89 kernels

### How SGLang runs GLM-5 on SM120

SGLang bypasses all DSA backends and runs GLM-5 as a DeepSeek V3.1 model:
- Uses MLA kernels ignoring sparsity (FlashInfer FA2 variant)
- DSA indexer is not invoked
- Computes attention on all tokens (including those DSA would have masked)
- This is backwards compatible -- slightly wasteful but not accuracy-degrading

### Available SM120 MLA kernels in SGLang

1. FlashInfer FA-based BF16 MLA kernel (SM120 specific)
2. XQA FP8 MLA kernel (SM120 specific)

Neither is available in vLLM as of 2026-03-08.

---

## All Errors and Fixes

### Error 1: `deep_gemm.get_num_sms()` AttributeError

```
AttributeError: 'ImportError' object has no attribute 'get_num_sms'
```

**Fix:** Set `SGLANG_ENABLE_JIT_DEEPGEMM=0` and `SGLANG_ENABLE_DEEP_GEMM=0`.

### Error 2: NaN in probability tensor (CUTLASS race condition)

```
Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed.
```

**Fix:** See [FlashInfer CUTLASS Race Condition Fix](#flashinfer-cutlass-race-condition-fix) section above.

### Error 3: CUDA device-side assert (MTP + radix cache)

```
eagle_worker_v2.py:510 _zero_fill_draft_kv_for_cached_prefix
torch.AcceleratorError: CUDA error: device-side assert triggered
```

**Fix:** SGLang PR https://github.com/sgl-project/sglang/pull/19897. Root cause is the FlashInfer CUTLASS race condition.

### Error 4: NSA Backend unsupported architecture

```
RuntimeError: Assertion error (attention.hpp:159): Unsupported architecture
```

**Fix:** Override to FlashInfer backend. Set `nsa_prefill_backend = "flashinfer"` and `nsa_decode_backend = "flashinfer"` in server_args.py, or use `voipmonitor/llm-pytorch-blackwell:nightly` which includes this fix.

### Error 5: vLLM "No valid attention backend found"

```
ValueError: No valid attention backend found for cuda with ... use_mla=True, use_sparse=True
```

**Fix:** None. GLM-5 does not run on vLLM for SM120. Use SGLang.

### Error 6: FP8 KV cache garbled / stops after 1 token

**Fix:** Use `--kv-cache-dtype bf16`. FP8 KV is broken on SM120.

### Error 7: MTP missing from lukealonso/GLM-5-NVFP4

```
ValueError: MTP speculative decoding layer 78 weights missing from checkpoint.
```

**Fix:** Use `festr2/GLM-5-NVFP4-MTP` which includes the MTP layer.

### Error 8: Missing MoE Triton kernel configs

```
Config file not found at .../E=257,N=256,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json
```

**Fix:** Generate configs using `https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton` or use the `voipmonitor/llm-pytorch-blackwell:nightly` Docker which includes pre-generated configs.

---

## Related PRs

| PR | Description |
|---|---|
| [SGLang #19897](https://github.com/sgl-project/sglang/pull/19897) | Fix for radix cache + speculative decoding crash |
| [SGLang #19948](https://github.com/sgl-project/sglang/pull/19948) | DeepGemm SCALE_UE8M0 fix for NVFP4 on SM120 |
| [SGLang #19951](https://github.com/sgl-project/sglang/pull/19951) | Fix for broken latest SGLang |
| [SGLang #19963](https://github.com/sgl-project/sglang/pull/19963) | Compilation fixes |
| [SGLang #19428](https://github.com/sgl-project/sglang/pull/19428) | Performance improvement for GLM-5 |
| [SGLang #20043](https://github.com/sgl-project/sglang/issues/20043) | Bug report: NaN crash with speculative decoding |
| [FlashInfer #2708](https://github.com/flashinfer-ai/flashinfer/issues/2708) | FlashInfer FP4 CUTLASS race condition |
| [FlashInfer #2716](https://github.com/flashinfer-ai/flashinfer/pull/2716) | FlashInfer fix for the race condition |
