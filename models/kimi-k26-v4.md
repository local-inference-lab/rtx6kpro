# Kimi-K2.6 v4 on 8x RTX PRO 6000 Blackwell

Measured on 2026-05-15 on the local 8-GPU RTX PRO 6000 Blackwell host.

This page is the canonical Kimi-K2.6 recipe for the rebased GLM/Kimi vLLM
stack. It uses the current clean vLLM branch, canonical B12X source, FlashInfer
from git in JIT/editable mode, CUTE DSL 4.5.0, cuda-tile 1.3.0, and patched
NCCL PR2127 without an external XML topology.

## Image

```bash
voipmonitor/vllm:glm-kimi-canonical-rebase-layered-vllm68b3569f-b12xc929144-flashinfergit1a60071-cutedsl45-20260514
```

Runtime base layer:

```bash
voipmonitor/vllm:glm-kimi-runtimebase-flat-flashinfergit1a60071-cudatile130-cutedsl45-20260514
```

Source state:

| Component | Revision |
|---|---|
| vLLM branch | `codex/glm51-kimi-canonical-rebase-test-20260514` |
| vLLM commit | `68b3569f2` |
| vLLM upstream base | `f887aa1a53e273d90ac537fcd399504f70aff2c7` |
| B12X commit | `c929144c7689668b07ca65af10ceadf1c745165d` |
| FlashInfer | git editable, `1a60071` |
| CUTE DSL | `nvidia-cutlass-dsl==4.5.0` |
| cuda-tile | `cuda-tile==1.3.0` |
| NCCL | `/opt/libnccl-pr2127.so.2.30.3` via `LD_PRELOAD` and `VLLM_NCCL_SO_PATH` |

## Launch

Use these profile values:

| Profile | `DCP` | `MTP` | `GPU_MEM` | Notes |
|---|---:|---:|---:|---|
| DCP1 + MTP | 1 | 1 | 0.94 | fastest short-context MTP profile |
| DCP1 no-MTP | 1 | 0 | 0.94 | target-only baseline |
| DCP2 + MTP | 2 | 1 | 0.90 | leaves MTP graph/activation headroom |
| DCP2 no-MTP | 2 | 0 | 0.94 | larger KV cache |
| DCP4 + MTP | 4 | 1 | 0.90 | current balanced DCP profile |
| DCP4 no-MTP | 4 | 0 | 0.94 | larger KV cache, no draft overhead |
| DCP8 + MTP | 8 | 1 | 0.90 | largest MTP DCP profile |
| DCP8 no-MTP | 8 | 0 | 0.94 | maximum KV cache |

Start script:

```bash
cat >/tmp/run-kimi-k26-v4 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:glm-kimi-canonical-rebase-layered-vllm68b3569f-b12xc929144-flashinfergit1a60071-cutedsl45-20260514}"
NAME="${NAME:-kimi-k26-v4}"
PORT="${PORT:-5264}"
DCP="${DCP:-4}"
MTP="${MTP:-1}"
GPU_MEM="${GPU_MEM:-0.90}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm-kimi-k26-v4}"

mkdir -p \
  "${CACHE_ROOT}/cutlass_dsl" \
  "${CACHE_ROOT}/jit" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/torchinductor" \
  "${CACHE_ROOT}/vllm"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

mtp_disable=0
if [[ "${MTP}" == "0" ]]; then
  mtp_disable=1
fi

exec docker run -d --gpus all --ipc=host --network host --privileged \
  --name "${NAME}" \
  --entrypoint /bin/bash \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e NCCL_GRAPH_FILE= \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-pr2127.so.2.30.3 \
  -e LD_PRELOAD=/opt/libnccl-pr2127.so.2.30.3 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=8 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=0 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e PORT="${PORT}" \
  -e TP_SIZE=8 \
  -e DCP_SIZE="${DCP}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEM}" \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=128 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
  -e KV_CACHE_DTYPE=fp8 \
  -e ATTENTION_BACKEND=TRITON_MLA \
  -e KIMI_DISABLE_MTP="${mtp_disable}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
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
  -v "${CACHE_ROOT}/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "${CACHE_ROOT}/jit:/cache/jit" \
  -v "${CACHE_ROOT}/triton:/root/.cache/triton" \
  -v "${CACHE_ROOT}/torchinductor:/root/.cache/torchinductor" \
  -v "${CACHE_ROOT}/vllm:/root/.cache/vllm" \
  "${IMAGE}" \
  -lc 'exec /opt/vllm/scripts/run-kimi26-vllm --compilation-config "{\"pass_config\":{\"fuse_rope_kvcache_cat_mla\":true},\"splitting_ops\":[]}"'
EOF
chmod +x /tmp/run-kimi-k26-v4
```

Examples:

```bash
# DCP4 + MTP on, port 5264.
PORT=5264 DCP=4 MTP=1 GPU_MEM=0.90 /tmp/run-kimi-k26-v4

# DCP8 without MTP.
PORT=5264 DCP=8 MTP=0 GPU_MEM=0.94 /tmp/run-kimi-k26-v4
```

Expected startup checks:

```text
vLLM is using nccl==2.30.3
PCIe custom allreduce enabled via VLLM_ENABLE_PCIE_ALLREDUCE=1
Application startup complete.
```

## Benchmark Method

Before every measured profile, run a 128k context, concurrency 1 warmup for 30
seconds:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5264 \
  --model Kimi-K2.6 \
  --concurrency 1 \
  --contexts 128k \
  --duration 30 \
  --skip-prefill \
  --kv-budget <reported_gpu_kv_cache_tokens> \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor
```

Decode matrix:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5264 \
  --model Kimi-K2.6 \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --skip-prefill \
  --kv-budget <reported_gpu_kv_cache_tokens> \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor
```

Result directory:

```bash
/root/bench-results/kimi-v4-full-68b3569f-20260514
```

## Results

Results are generated from the benchmark JSON files after the full matrix
finishes.

<!-- KIMI_RESULTS_START -->
### KIMI DCP1 MTP on

KV cache budget: `418,016` tokens; 128k cc1 warmup: `52.4` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 122.0 | 176.6 | 337.7 | 492.6 | 754.1 | 1129.0 | 1675.0 | 2245.8 |
| 16k | 103.8 | 171.9 | 254.7 | 351.7 | 484.6 | skip | skip | skip |
| 32k | 92.9 | 136.9 | 207.8 | 265.8 | skip | skip | skip | skip |
| 64k | 74.6 | 115.8 | 155.1 | skip | skip | skip | skip | skip |
| 128k | 58.3 | 78.7 | skip | skip | skip | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.494 | 0.353 | 0.368 | 0.493 | 0.354 | 0.370 | 0.396 | 0.417 |
| 16k | 0.347 | 0.386 | 0.421 | 0.352 | 0.359 | skip | skip | skip |
| 32k | 0.317 | 0.462 | 0.344 | 0.349 | skip | skip | skip | skip |
| 64k | 0.581 | 0.347 | 0.441 | skip | skip | skip | skip | skip |
| 128k | 0.333 | 0.211 | skip | skip | skip | skip | skip | skip |

### KIMI DCP1 MTP off

KV cache budget: `478,992` tokens; 128k cc1 warmup: `46.1` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 92.4 | 151.3 | 239.5 | 350.9 | 636.2 | 957.7 | 1497.3 | 2222.6 |
| 16k | 81.5 | 134.9 | 207.2 | 298.9 | 489.5 | skip | skip | skip |
| 32k | 73.4 | 123.3 | 183.2 | 261.8 | skip | skip | skip | skip |
| 64k | 60.5 | 106.2 | 149.3 | skip | skip | skip | skip | skip |
| 128k | 46.1 | 83.6 | skip | skip | skip | skip | skip | skip |

### KIMI DCP2 MTP on

KV cache budget: `602,752` tokens; 128k cc1 warmup: `60.4` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 101.5 | 150.9 | 272.2 | 408.1 | 657.5 | 998.5 | 1444.7 | 1871.9 |
| 16k | 93.4 | 137.0 | 246.0 | 355.0 | 510.4 | 701.0 | skip | skip |
| 32k | 88.8 | 124.8 | 209.5 | 306.3 | 412.5 | skip | skip | skip |
| 64k | 80.6 | 113.2 | 187.5 | 233.3 | skip | skip | skip | skip |
| 128k | 68.1 | 97.6 | 133.8 | skip | skip | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.406 | 0.395 | 0.454 | 0.359 | 0.360 | 0.378 | 0.371 | 0.413 |
| 16k | 0.455 | 0.380 | 0.469 | 0.427 | 0.396 | 0.413 | skip | skip |
| 32k | 0.374 | 0.624 | 0.424 | 0.363 | 0.411 | skip | skip | skip |
| 64k | 0.491 | 0.301 | 0.275 | 0.375 | skip | skip | skip | skip |
| 128k | 0.656 | 0.470 | 0.389 | skip | skip | skip | skip | skip |

### KIMI DCP2 MTP off

KV cache budget: `953,088` tokens; 128k cc1 warmup: `55.7` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 81.3 | 132.3 | 222.2 | 312.2 | 559.9 | 847.7 | 1324.5 | 1983.7 |
| 16k | 76.1 | 123.3 | 204.5 | 283.2 | 478.3 | 677.2 | skip | skip |
| 32k | 72.0 | 118.4 | 189.6 | 265.0 | 423.4 | skip | skip | skip |
| 64k | 64.8 | 108.1 | 167.4 | 232.0 | skip | skip | skip | skip |
| 128k | 55.8 | 92.0 | 134.9 | skip | skip | skip | skip | skip |

### KIMI DCP4 MTP on

KV cache budget: `1,205,568` tokens; 128k cc1 warmup: `53.6` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 90.0 | 149.3 | 254.2 | 386.2 | 629.3 | 914.2 | 1304.0 | 1681.5 |
| 16k | 87.3 | 134.6 | 214.2 | 296.8 | 420.0 | 549.4 | 676.9 | skip |
| 32k | 82.2 | 111.7 | 174.4 | 248.8 | 311.2 | 384.8 | skip | skip |
| 64k | 70.7 | 100.3 | 139.4 | 173.5 | 204.3 | skip | skip | skip |
| 128k | 56.0 | 73.3 | 90.5 | 110.1 | skip | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.235 | 0.343 | 0.475 | 0.404 | 0.375 | 0.343 | 0.390 | 0.391 |
| 16k | 0.467 | 0.430 | 0.337 | 0.365 | 0.383 | 0.413 | 0.480 | skip |
| 32k | 0.441 | 0.309 | 0.325 | 0.410 | 0.410 | 0.404 | skip | skip |
| 64k | 0.354 | 0.303 | 0.406 | 0.442 | 0.420 | skip | skip | skip |
| 128k | 0.321 | 0.392 | 0.367 | 0.417 | skip | skip | skip | skip |

### KIMI DCP4 MTP off

KV cache budget: `1,906,240` tokens; 128k cc1 warmup: `55.8` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 77.8 | 125.0 | 206.8 | 315.7 | 543.4 | 809.5 | 1235.0 | 1813.0 |
| 16k | 73.0 | 116.4 | 187.4 | 270.1 | 420.2 | 580.2 | 784.9 | skip |
| 32k | 69.5 | 110.8 | 172.4 | 240.0 | 350.3 | 455.4 | skip | skip |
| 64k | 63.4 | 100.7 | 149.0 | 194.6 | 264.0 | skip | skip | skip |
| 128k | 55.8 | 84.3 | 115.7 | 140.0 | skip | skip | skip | skip |

### KIMI DCP8 MTP on

KV cache budget: `2,411,136` tokens; 128k cc1 warmup: `59.7` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 84.8 | 135.9 | 208.3 | 332.7 | 519.8 | 680.2 | 951.0 | 1121.7 |
| 16k | 83.1 | 143.3 | 202.2 | 301.3 | 444.5 | 555.4 | 759.4 | 887.6 |
| 32k | 81.9 | 119.9 | 182.0 | 274.7 | 377.9 | 466.2 | 614.9 | skip |
| 64k | 75.8 | 104.8 | 159.4 | 223.8 | 298.4 | 365.2 | skip | skip |
| 128k | 63.5 | 86.5 | 133.2 | 162.5 | 203.6 | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.252 | 0.446 | 0.347 | 0.360 | 0.371 | 0.355 | 0.427 | 0.455 |
| 16k | 0.333 | 0.617 | 0.384 | 0.392 | 0.408 | 0.481 | 0.440 | 0.449 |
| 32k | 0.447 | 0.470 | 0.341 | 0.419 | 0.404 | 0.411 | 0.439 | skip |
| 64k | 0.381 | 0.360 | 0.287 | 0.429 | 0.414 | 0.433 | skip | skip |
| 128k | 0.444 | 0.342 | 0.406 | 0.412 | 0.406 | skip | skip | skip |

### KIMI DCP8 MTP off

KV cache budget: `3,811,968` tokens; 128k cc1 warmup: `61.2` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 | cc128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 76.6 | 115.0 | 189.7 | 282.6 | 442.6 | 677.2 | 1026.0 | 1301.1 |
| 16k | 73.8 | 109.6 | 179.2 | 261.0 | 395.7 | 582.3 | 842.5 | 1053.7 |
| 32k | 71.5 | 106.2 | 170.6 | 242.9 | 366.3 | 516.2 | 716.7 | skip |
| 64k | 67.1 | 101.3 | 156.1 | 215.2 | 316.8 | 421.3 | skip | skip |
| 128k | 61.3 | 90.9 | 132.9 | 172.0 | 246.4 | skip | skip | skip |
<!-- KIMI_RESULTS_END -->

## Kimi Draft Acceptance Notes

The Kimi EAGLE3 draft acceptance rate is intentionally recorded separately
because it varies much more than GLM in this setup. The table below is generated
from both `decode-matrix.json` and the vLLM `SpecDecoding metrics` lines in
`final.log`.

<!-- KIMI_ACCEPTANCE_START -->
| Profile | JSON mean | JSON stdev | log avg min | log avg mean | log avg max | pos1 mean | pos2 mean | pos3 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KIMI DCP1 | 0.386 | 0.074 | 0.000 | 0.426 | 1.000 | 0.645 | 0.393 | 0.240 |
| KIMI DCP2 | 0.416 | 0.080 | 0.163 | 0.431 | 1.000 | 0.653 | 0.398 | 0.242 |
| KIMI DCP4 | 0.385 | 0.054 | 0.167 | 0.440 | 1.000 | 0.657 | 0.407 | 0.255 |
| KIMI DCP8 | 0.405 | 0.062 | 0.227 | 0.448 | 1.000 | 0.667 | 0.413 | 0.262 |
<!-- KIMI_ACCEPTANCE_END -->
