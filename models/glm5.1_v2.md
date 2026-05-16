# GLM-5.1 v2 on 8x RTX PRO 6000 Blackwell

Measured on 2026-05-15 on the local 8-GPU RTX PRO 6000 Blackwell host.

This page is the canonical GLM-5.1 recipe for the rebased GLM/Kimi vLLM stack.
It uses B12X sparse MLA, B12X FP4 MoE, W4A16 MoE by default, GLM MTP support,
patched NCCL PR2127 without an external XML topology, and the same base image as
the Kimi-K2.6 v4 recipe.

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
| DCP1 + MTP | 1 | 1 | 0.865 | fastest safe single-DCP profile |
| DCP1 no-MTP | 1 | 0 | 0.865 | target-only baseline |
| DCP2 + MTP | 2 | 1 | 0.865 | larger KV cache |
| DCP2 no-MTP | 2 | 0 | 0.865 | target-only DCP2 |
| DCP4 + MTP | 4 | 1 | 0.865 | current balanced GLM profile |
| DCP4 no-MTP | 4 | 0 | 0.865 | larger KV cache |
| DCP8 + MTP | 8 | 1 | 0.82 | leaves MTP graph/activation headroom |
| DCP8 no-MTP | 8 | 0 | 0.865 | maximum KV cache |

Start script:

```bash
cat >/tmp/run-glm51-v2 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:glm-kimi-canonical-rebase-layered-vllm68b3569f-b12xc929144-flashinfergit1a60071-cutedsl45-20260514}"
NAME="${NAME:-glm51-v2}"
PORT="${PORT:-5264}"
DCP="${DCP:-4}"
MTP="${MTP:-1}"
GPU_MEM="${GPU_MEM:-0.865}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm-glm51-v2}"

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
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_B12X_MLA_DECODE_INLINE_LSE=1 \
  -e VLLM_B12X_MLA_SPEC_SERIAL_DECODE=0 \
  -e VLLM_MTP_RETURN_NORMALIZED_HIDDEN=1 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_ACC=1.0 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE=1.0 \
  -e B12X_MOE_FORCE_A16=1 \
  -e PORT="${PORT}" \
  -e TP_SIZE=8 \
  -e DCP_SIZE="${DCP}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEM}" \
  -e MAX_MODEL_LEN=202752 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=64 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=256 \
  -e KV_CACHE_DTYPE=fp8 \
  -e ATTENTION_BACKEND=B12X_MLA_SPARSE \
  -e MOE_BACKEND=b12x \
  -e GLM51_DISABLE_MTP="${mtp_disable}" \
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
  -lc 'exec /opt/vllm/scripts/run-glm51-vllm'
EOF
chmod +x /tmp/run-glm51-v2
```

Examples:

```bash
# DCP4 + MTP on, port 5264.
PORT=5264 DCP=4 MTP=1 GPU_MEM=0.865 /tmp/run-glm51-v2

# DCP8 + MTP on.
PORT=5264 DCP=8 MTP=1 GPU_MEM=0.82 /tmp/run-glm51-v2
```

Expected startup checks:

```text
vLLM is using nccl==2.30.3
PCIe custom allreduce enabled via VLLM_ENABLE_PCIE_ALLREDUCE=1
MOE_BACKEND=b12x
B12X_MOE_FORCE_A16=1
Application startup complete.
```

## Benchmark Method

Before every measured profile, run a 128k context, concurrency 1 warmup for 30
seconds:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5264 \
  --model GLM-5 \
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
  --model GLM-5 \
  --concurrency 1,2,4,8,16,32,64 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --skip-prefill \
  --kv-budget <reported_gpu_kv_cache_tokens> \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor
```

Use the raw vLLM startup value from:

```text
GPU KV cache size: <N> tokens
```

Do not use the archived `kv-budget.txt` value as KV capacity. In this archived
run the benchmark driver used a conservative manual budget equal to
`GPU KV cache size / 4`; that value only affected benchmark fit/skip gating and
was previously mislabeled in this page.

Standalone prefill was not run for this v2 matrix. TTFT-derived prompt/token
rates are intentionally not reported here because prefix-cache state and request
admission can skew them heavily.

Result directory:

```bash
/root/bench-results/glm51-v2-full-68b3569f-20260514
```

Raw benchmark artifacts are checked into this repository:

```text
models/glm5.1/benchmarks/glm51-v2-full-68b3569f-20260514/
```

## PCIe / Communication State

The saved JSON and startup logs confirm the communication path used for every
profile. They do not contain per-profile measured PCIe GB/s counters; `p2pmark`
was not run in this matrix.

| Profile | DCP | MTP | GPU KV cache size | Archived `--kv-budget` used | P2P override | NCCL | PCIe allreduce path | C++ AR cutoff | PCIe GB/s |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| DCP1 + MTP | 1 | 1 | 319,232 | 79,808 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |
| DCP1 no-MTP | 1 | 0 | 341,504 | 85,376 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |
| DCP2 + MTP | 2 | 1 | 635,520 | 158,880 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |
| DCP2 no-MTP | 2 | 0 | 680,192 | 170,048 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |
| DCP4 + MTP | 4 | 1 | 1,271,040 | 317,760 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |
| DCP4 no-MTP | 4 | 0 | 1,360,384 | 340,096 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |
| DCP8 + MTP | 8 | 1 | 1,952,256 | 488,064 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |
| DCP8 no-MTP | 8 | 0 | 2,720,768 | 680,192 | effective | 2.30.3 PR2127 | vLLM C++ custom AR | 56KB, rows<=8 | not measured |

## Results

Results are generated from the benchmark JSON files after the full matrix
finishes.

<!-- GLM_RESULTS_START -->
### GLM DCP1 MTP on

GPU KV cache size: `319,232` tokens. Archived benchmark `--kv-budget` used:
`79,808` tokens.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 95.6 | 148.4 | 247.0 | 200.9 | 312.6 | 405.4 |
| 16k | 90.5 | 136.9 | 234.3 | skip | skip | skip |
| 32k | 89.0 | 135.6 | skip | skip | skip | skip |
| 64k | 86.1 | skip | skip | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.577 | 0.563 | 0.581 | 0.512 | 0.521 | 0.533 |
| 16k | 0.381 | 0.667 | 0.470 | skip | skip | skip |
| 32k | 0.552 | 0.564 | skip | skip | skip | skip |
| 64k | 0.559 | skip | skip | skip | skip | skip |

### GLM DCP1 MTP off

GPU KV cache size: `341,504` tokens. Archived benchmark `--kv-budget` used:
`85,376` tokens.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 59.0 | 92.4 | 164.2 | 251.4 | 428.7 | 317.8 |
| 16k | 57.0 | 88.2 | 155.7 | skip | skip | skip |
| 32k | 56.1 | 86.3 | skip | skip | skip | skip |
| 64k | 55.0 | skip | skip | skip | skip | skip |

### GLM DCP2 MTP on

GPU KV cache size: `635,520` tokens. Archived benchmark `--kv-budget` used:
`158,880` tokens; 128k cc1 warmup: `71.5` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 81.5 | 124.7 | 209.6 | 180.8 | 281.1 | 363.5 | 588.0 |
| 16k | 73.6 | 118.6 | 200.8 | 164.9 | skip | skip | skip |
| 32k | 72.3 | 114.0 | 198.0 | skip | skip | skip | skip |
| 64k | 74.5 | 115.5 | skip | skip | skip | skip | skip |
| 128k | 73.1 | skip | skip | skip | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.656 | 0.654 | 0.500 | 0.579 | 0.515 | 0.503 | 0.438 |
| 16k | 0.422 | 0.471 | 0.508 | 0.568 | skip | skip | skip |
| 32k | 0.379 | 0.565 | 0.535 | skip | skip | skip | skip |
| 64k | 0.321 | 0.508 | skip | skip | skip | skip | skip |
| 128k | 0.471 | skip | skip | skip | skip | skip | skip |

### GLM DCP2 MTP off

GPU KV cache size: `680,192` tokens. Archived benchmark `--kv-budget` used:
`170,048` tokens; 128k cc1 warmup: `49.3` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 53.6 | 82.6 | 146.6 | 226.4 | 375.4 | 296.5 | 479.9 |
| 16k | 51.6 | 79.7 | 140.6 | 213.3 | skip | skip | skip |
| 32k | 50.9 | 78.7 | 137.6 | skip | skip | skip | skip |
| 64k | 50.0 | 77.0 | skip | skip | skip | skip | skip |
| 128k | 49.3 | skip | skip | skip | skip | skip | skip |

### GLM DCP4 MTP on

GPU KV cache size: `1,271,040` tokens. Archived benchmark `--kv-budget` used:
`317,760` tokens; 128k cc1 warmup: `68.5` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 74.6 | 115.2 | 197.2 | 168.8 | 254.7 | 318.4 | 496.1 |
| 16k | 72.6 | 110.4 | 180.1 | 157.6 | 228.3 | skip | skip |
| 32k | 73.1 | 110.7 | 177.6 | 157.3 | skip | skip | skip |
| 64k | 68.9 | 104.8 | 166.8 | skip | skip | skip | skip |
| 128k | 68.0 | 106.0 | skip | skip | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.333 | 0.493 | 0.548 | 0.454 | 0.611 | 0.526 | 0.389 |
| 16k | 0.536 | 0.515 | 0.559 | 0.568 | 0.465 | skip | skip |
| 32k | 0.512 | 0.429 | 0.546 | 0.486 | skip | skip | skip |
| 64k | 0.571 | 0.405 | 0.521 | skip | skip | skip | skip |
| 128k | 0.519 | 0.444 | skip | skip | skip | skip | skip |

### GLM DCP4 MTP off

GPU KV cache size: `1,360,384` tokens. Archived benchmark `--kv-budget` used:
`340,096` tokens; 128k cc1 warmup: `48.2` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 52.1 | 80.3 | 140.4 | 208.0 | 336.5 | 277.3 | 447.9 |
| 16k | 50.1 | 76.9 | 132.8 | 197.3 | 308.2 | skip | skip |
| 32k | 49.9 | 76.6 | 132.4 | 194.4 | skip | skip | skip |
| 64k | 49.2 | 75.7 | 129.3 | skip | skip | skip | skip |
| 128k | 48.2 | 73.9 | skip | skip | skip | skip | skip |

### GLM DCP8 MTP on

GPU KV cache size: `1,952,256` tokens. Archived benchmark `--kv-budget` used:
`488,064` tokens; 128k cc1 warmup: `60.0` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 66.0 | 104.0 | 162.1 | 145.9 | 213.7 | 261.4 | 386.4 |
| 16k | 63.1 | 95.9 | 150.7 | 139.8 | 187.0 | skip | skip |
| 32k | 65.1 | 96.1 | 149.9 | 132.3 | skip | skip | skip |
| 64k | 62.4 | 98.9 | 148.7 | skip | skip | skip | skip |
| 128k | 60.2 | 94.5 | skip | skip | skip | skip | skip |

Spec acceptance rate:

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.407 | 0.650 | 0.656 | 0.516 | 0.554 | 0.351 | 0.492 |
| 16k | 0.622 | 0.535 | 0.561 | 0.542 | 0.421 | skip | skip |
| 32k | 0.641 | 0.528 | 0.517 | 0.493 | skip | skip | skip |
| 64k | 0.833 | 0.789 | 0.482 | skip | skip | skip | skip |
| 128k | 0.435 | 0.500 | skip | skip | skip | skip | skip |

### GLM DCP8 MTP off

GPU KV cache size: `2,720,768` tokens. Archived benchmark `--kv-budget` used:
`680,192` tokens; 128k cc1 warmup: `45.9` tok/s.

| Context | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 49.6 | 74.7 | 124.8 | 181.0 | 277.8 | 248.5 | 386.1 |
| 16k | 46.8 | 71.0 | 117.2 | 167.4 | 258.6 | 234.6 | skip |
| 32k | 46.7 | 70.9 | 116.8 | 166.1 | 257.0 | skip | skip |
| 64k | 46.6 | 70.3 | 116.6 | 164.8 | skip | skip | skip |
| 128k | 45.9 | 69.3 | 114.1 | skip | skip | skip | skip |
<!-- GLM_RESULTS_END -->

## GLM Draft Acceptance Notes

The GLM MTP acceptance table is recorded for comparison with Kimi. GLM normally
shows much steadier acceptance across the three draft positions than Kimi.

<!-- GLM_ACCEPTANCE_START -->
| Profile | JSON mean | JSON stdev | log avg min | log avg mean | log avg max | pos1 mean | pos2 mean | pos3 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLM DCP1 | 0.540 | 0.066 | 0.428 | 0.545 | 0.758 | 0.803 | 0.547 | 0.286 |
| GLM DCP2 | 0.505 | 0.085 | 0.419 | 0.537 | 0.691 | 0.798 | 0.535 | 0.278 |
| GLM DCP4 | 0.497 | 0.067 | 0.386 | 0.539 | 0.833 | 0.805 | 0.540 | 0.272 |
| GLM DCP8 | 0.549 | 0.115 | 0.412 | 0.554 | 1.000 | 0.806 | 0.561 | 0.297 |
<!-- GLM_ACCEPTANCE_END -->
