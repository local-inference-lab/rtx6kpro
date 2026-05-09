# Kimi-K2.6 v3 on 8x RTX PRO 6000 Blackwell

Measured on 2026-05-09 on the local 8-GPU RTX PRO 6000 Blackwell host.
This page reruns the Kimi-K2.6 decode matrix from [`kimi-k26-v2.md`](kimi-k26-v2.md)
with the current communicator baseline: patched NCCL without an external XML topology
file plus vLLM C++ PCIe allreduce only for small tensors.

## What Changed Versus v2

- Image: `voipmonitor/vllm:kimi-k26-v3-cpp56-20260509`.
- NCCL PR2127 is used from `/opt/libnccl-pr2127.so.2.30.3` via `VLLM_NCCL_SO_PATH` and `LD_PRELOAD`.
- No external NCCL XML topology is required: `USE_NCCL_XML=0` and `NCCL_GRAPH_FILE` is unset.
- PCIe custom allreduce uses the vLLM C++ backend only below a 56 KiB cutoff: `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB`.
- Larger reductions fall back to NCCL with `NCCL_PROTO=LL,LL128,Simple`.
- Experimental RTX6K fused allreduce-add is disabled: `VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0`.
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` is set explicitly; leaving it enabled reduced the reported DCP8 MTP KV cache from about 2.62M tokens to about 1.74M tokens at the same `GPU_MEMORY_UTILIZATION`.

A short DCP1 A/B check showed that patched NCCL/no-XML is effectively parity with XML for this setup. For example, DCP1+MTP ctx0/C32 was `1240.6 tok/s` with XML and `1221.2 tok/s` with patched NCCL/no-XML; DCP1 no-MTP ctx0/C32 was `963.4 tok/s` with XML and `962.8 tok/s` with no-XML. The v3 tables below use no-XML.

## Launch

Common variables:

```bash
export IMAGE=voipmonitor/vllm:kimi-k26-v3-cpp56-20260509
export PORT=5002
export DCP=8
export MTP=1
export GPU_MEM=0.90
export MAX_MODEL_LEN=262144
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=128
export MAX_CUDAGRAPH_CAPTURE_SIZE=512
export CACHE_ROOT=~/.cache/vllm-kimi-k26-v3
```

Use these profile values:

| Profile | `DCP` | `MTP` | `GPU_MEM` | Why |
|---|---:|---:|---:|---|
| DCP1 + MTP | 1 | 1 | 0.94 | fastest short-context MTP profile |
| DCP1 no-MTP | 1 | 0 | 0.94 | baseline target-only decode |
| DCP4 + MTP | 4 | 1 | 0.90 | leaves MTP graph/activation headroom |
| DCP4 no-MTP | 4 | 0 | 0.94 | larger KV cache, no MTP batch expansion |
| DCP8 + MTP | 8 | 1 | 0.90 | largest stable MTP DCP profile |
| DCP8 no-MTP | 8 | 0 | 0.94 | maximum KV cache |

Start command:

```bash
mkdir -p \
  "$CACHE_ROOT/cutlass_dsl" \
  "$CACHE_ROOT/jit" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/torchinductor" \
  "$CACHE_ROOT/vllm"

docker rm -f kimi-k26-v3 >/dev/null 2>&1 || true

MTP_ENV=(-e KIMI_DISABLE_MTP=0)
if [[ "$MTP" == "0" ]]; then
  MTP_ENV=(-e KIMI_DISABLE_MTP=1)
fi

docker run -d --gpus all --ipc=host --network host --privileged \
  --name kimi-k26-v3 \
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
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=0 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e PORT="$PORT" \
  -e TP_SIZE=8 \
  -e DCP_SIZE="$DCP" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEM" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="$MAX_CUDAGRAPH_CAPTURE_SIZE" \
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
  "${MTP_ENV[@]}" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE_ROOT/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "$CACHE_ROOT/jit:/cache/jit" \
  -v "$CACHE_ROOT/triton:/root/.cache/triton" \
  -v "$CACHE_ROOT/torchinductor:/root/.cache/torchinductor" \
  -v "$CACHE_ROOT/vllm:/root/.cache/vllm" \
  "$IMAGE" \
  run-kimi26-vllm
```

Expected startup checks:

```text
vLLM is using nccl==2.30.3
PCIe custom allreduce enabled via VLLM_ENABLE_PCIE_ALLREDUCE=1 (backend=cpp, using vLLM C++ custom allreduce).
Application startup complete.
```

## Benchmark Command

Decode matrix:

```bash
python3 /mnt/llm_decode_bench.py \
  --port ${PORT} \
  --model Kimi-K2.6 \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 0,16k,32k,64k,128k \
  --duration 10 \
  --skip-prefill \
  --display-mode plain \
  --output /root/bench-results/kimi-k26-v3-20260509/<profile>.json
```

Prefill sanity:

```bash
python3 /mnt/llm_decode_bench.py \
  --port ${PORT} \
  --model Kimi-K2.6 \
  --standalone-prefill \
  --prefill-only \
  --prefill-contexts 8k,16k,32k,64k,128k \
  --prefill-duration 10 \
  --display-mode plain \
  --output /root/bench-results/kimi-k26-v3-20260509/prefill_dcp1_mtp3.json
```

## KV Cache

| Config | GPU memory utilization | KV tokens | Max concurrency at 262144 |
|---|---:|---:|---:|
| DCP=1 + MTP=3 | 0.94 | 442,256 | 1.69x |
| DCP=1, No MTP | 0.94 | 491,200 | 1.87x |
| DCP=4 + MTP=3 | 0.90 | 1,312,192 | 5.01x |
| DCP=4, No MTP | 0.94 | 1,964,992 | 7.50x |
| DCP=8 + MTP=3 | 0.90 | 2,624,384 | 10.01x |
| DCP=8, No MTP | 0.94 | 3,929,600 | 14.99x |

## Decode Throughput

Aggregate decode tok/s from `llm_decode_bench.py --skip-prefill`.

### DCP=1 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 110.1 | 202.8 | 331.0 | 507.9 | 814.7 | 1221.2 | 1781.7 | 2406.5 |
| 16k | 104.3 | 175.6 | 268.9 | 354.1 | 490.4 | ∅ | ∅ | ∅ |
| 32k | 95.8 | 160.0 | 223.4 | 285.7 | ∅ | ∅ | ∅ | ∅ |
| 64k | 80.0 | 123.5 | 164.7 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 57.2 | 84.3 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

### DCP=1, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 91.4 | 151.9 | 240.4 | 406.9 | 641.8 | 962.8 | 1518.1 | 2310.2 |
| 16k | 81.1 | 135.6 | 206.6 | 339.1 | 487.5 | ∅ | ∅ | ∅ |
| 32k | 73.6 | 124.3 | 184.2 | 294.0 | ∅ | ∅ | ∅ | ∅ |
| 64k | 61.7 | 106.8 | 149.4 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 46.1 | 83.4 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

### DCP=4 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 99.8 | 162.8 | 262.7 | 411.7 | 647.1 | 979.7 | 1359.3 | 1678.5 |
| 16k | 97.2 | 158.8 | 233.8 | 311.0 | 424.2 | 542.4 | 690.2 | ∅ |
| 32k | 80.1 | 133.3 | 189.1 | 238.5 | 314.4 | 378.7 | ∅ | ∅ |
| 64k | 73.8 | 104.2 | 140.9 | 170.9 | 197.5 | ∅ | ∅ | ∅ |
| 128k | 62.6 | 82.9 | 93.8 | 106.4 | ∅ | ∅ | ∅ | ∅ |

### DCP=4, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76.5 | 124.9 | 208.5 | 355.1 | 544.8 | 822.9 | 1258.0 | 1867.0 |
| 16k | 72.8 | 116.7 | 187.5 | 298.7 | 425.7 | 590.9 | 832.7 | ∅ |
| 32k | 70.3 | 111.4 | 174.0 | 263.7 | 352.8 | 467.2 | ∅ | ∅ |
| 64k | 65.2 | 101.3 | 149.3 | 209.0 | 266.8 | ∅ | ∅ | ∅ |
| 128k | 55.8 | 84.6 | 116.0 | 149.4 | ∅ | ∅ | ∅ | ∅ |

### DCP=8 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 92.5 | 147.8 | 224.2 | 373.9 | 541.8 | 697.2 | 936.7 | 1151.1 |
| 16k | 83.9 | 145.9 | 210.4 | 310.2 | 441.6 | 567.1 | 773.5 | 1012.0 |
| 32k | 81.7 | 135.6 | 194.1 | 277.3 | 384.8 | 477.5 | 633.3 | ∅ |
| 64k | 73.5 | 120.9 | 174.7 | 235.4 | 298.7 | 367.8 | ∅ | ∅ |
| 128k | 65.4 | 99.7 | 130.0 | 160.0 | 205.4 | ∅ | ∅ | ∅ |

### DCP=8, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76.2 | 116.2 | 190.3 | 316.2 | 440.3 | 673.8 | 1042.5 | 1383.1 |
| 16k | 74.1 | 109.7 | 178.8 | 287.8 | 398.9 | 590.7 | 869.0 | 1131.7 |
| 32k | 72.3 | 106.8 | 171.2 | 269.4 | 368.2 | 524.5 | 737.3 | ∅ |
| 64k | 69.0 | 101.7 | 156.5 | 233.7 | 314.6 | 426.0 | ∅ | ∅ |
| 128k | 62.3 | 90.8 | 132.2 | 183.5 | 246.5 | ∅ | ∅ | ∅ |

## Prefill Sanity

DCP=1 + MTP=3, `GPU_MEM=0.94`, client-side prompt tokens divided by TTFT.

| ctx | prompt tokens | TTFT s | prefill tok/s | samples |
|---|---:|---:|---:|---:|
| 8k | 8,189 | 1.114 | 7,352 | 7 |
| 16k | 16,231 | 2.279 | 7,122 | 4 |
| 32k | 32,309 | 4.835 | 6,682 | 2 |
| 64k | 64,471 | 10.919 | 5,905 | 1 |
| 128k | 128,789 | 26.562 | 4,849 | 1 |

## Notes

- `∅` means the benchmark skipped or hid the cell because the requested total KV footprint does not fit the server-reported KV cache.
- DCP8 gives the largest useful long-context concurrency because it multiplies the effective KV budget, but DCP1 remains fastest for short-context MTP.
- DCP4/DCP8 startup with the XML topology and official NCCL 2.29.7 failed during DCP NCCL group initialization in this run. The patched NCCL PR2127 no-XML path started correctly and is the v3 baseline.
- DCP1 XML and no-XML were measured both ways; the results were parity within normal run noise, so no-XML is preferred to remove the external XML dependency.
- The generated JSON and logs for this run were stored locally under `/root/bench-results/kimi-k26-v3-20260509/`.
