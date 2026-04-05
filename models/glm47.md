# GLM-4.7 on RTX PRO 6000 Blackwell

## Quick Start

Runs on **4× RTX PRO 6000** (96 GB each) using FP8 quantization (compressed-tensors). Includes EAGLE speculative decoding with 3-step lookahead and hierarchical KV cache with CPU offload.

```bash
docker pull voipmonitor/sglang:cu130
```

```bash
docker run --gpus '"device=0,1,2,3"' \
  --ipc=host --shm-size=8g --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt:/mnt \
  -v jit-cache:/cache/jit \
  -e NCCL_P2P_LEVEL=4 \
  -e SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
  -e USE_TRITON_W8A8_FP8_KERNEL=1 \
  -e SGL_ENABLE_JIT_DEEPGEMM=0 \
  voipmonitor/sglang:cu130 \
  python3 -m sglang.launch_server \
    --model /mnt/zai-org/GLM-4.7-FP8 \
    --chat-template /mnt/zai-org/GLM-4.7-FP8/chat_template.jinja \
    --served-model-name glm-4.7-fp8 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --fp8-gemm-backend triton \
    --trust-remote-code \
    --context-length 200000 \
    --mem-fraction-static 0.95 \
    --sleep-on-idle \
    --max-running-requests 48 \
    --chunked-prefill-size 8192 \
    --enable-mixed-chunk \
    --cuda-graph-max-bs 8 \
    --attention-backend flashinfer \
    --disable-shared-experts-fusion \
    --schedule-conservativeness 0.3 \
    --enable-metrics \
    --enable-cache-report \
    --host 0.0.0.0 --port 4999 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
    --enable-hierarchical-cache --hicache-ratio 5 \
    --enable-flashinfer-allreduce-fusion
```

API endpoint: `http://localhost:4999/v1` (OpenAI-compatible).

If running as a **systemd service** instead of Docker, see [Systemd Service](#systemd-service) below.

---

## Overview

| Parameter | Value |
|-----------|-------|
| Model | `zai-org/GLM-4.7-FP8` |
| Architecture | `glm4_moe` (Mixture of Experts) |
| Hidden size | 5120 |
| Layers | 92 |
| Quantization | FP8 (compressed-tensors) |
| Model size on disk | ~338 GB (93 safetensor shards) |
| Context length | 200,000 tokens |
| Reasoning | Built-in thinking via `<think>` blocks |
| Tool calling | GLM-4.7 native format (`glm47` parser) |

---

## Hardware Requirements

| Configuration | Notes |
|---|---|
| **4× RTX PRO 6000 Blackwell** | TP=4, FP8, ~82 GB per GPU at max context |
| NVIDIA Driver | 590.48+ (tested with 595.58.03) |
| CUDA | 13.0+ |

---

## Key Configuration Details

### Chat Template

GLM-4.7 requires an explicit `--chat-template` pointing to the model's jinja file. Without it, vLLM/SGLang falls back to the template in `tokenizer_config.json` which may not match the expected format for reasoning and tool calling.

### Parsers

- `--reasoning-parser glm45` — Parses `<think>` blocks in GLM format
- `--tool-call-parser glm47` — Parses GLM-4.7 native tool call format

### Hierarchical Cache

`--enable-hierarchical-cache --hicache-ratio 5` enables CPU offload of KV cache, extending effective context by keeping cold KV entries in system RAM. The ratio of 5 means 5× more CPU cache than GPU cache.

### EAGLE Speculative Decoding

```
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

3-step EAGLE with top-1 selection, drafting 4 tokens per step.

### Shared Experts Fusion

`--disable-shared-experts-fusion` is required for stability with GLM-4.7 MoE architecture on Blackwell.

### FlashInfer Allreduce Fusion

`--enable-flashinfer-allreduce-fusion` fuses allreduce with FlashInfer attention kernels for better TP performance on PCIe systems.

---

## Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `NCCL_P2P_LEVEL` | `4` | PCIe P2P communication level |
| `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK` | `True` | Skip TP memory balance check (needed for FP8) |
| `USE_TRITON_W8A8_FP8_KERNEL` | `1` | Use Triton FP8 GEMM kernels |
| `SGL_ENABLE_JIT_DEEPGEMM` | `0` | Disable JIT DeepGemm (stability) |

---

## Systemd Service

When running natively (not in Docker), GLM-4.7 is managed via systemd:

```ini
# /etc/systemd/system/sglang4999.service
[Unit]
Description=SGLang Server on Port 4999
After=network.target

[Service]
Type=simple
ExecStart=/root/docker/llm-services/start-sglang4999.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

The wrapper script (`start-sglang4999.sh`) activates the `vllm6dev4` conda environment and launches SGLang with the same parameters as the Docker command above.

```bash
systemctl start sglang4999
systemctl status sglang4999
journalctl -u sglang4999 -f
```

---

## Docker Image

| Image | Purpose |
|---|---|
| **`voipmonitor/sglang:cu130`** | Recommended. SGLang with SM120 patches, FlashInfer from source, b12x backend, PCIe allreduce, NCCL graph. |

Source & Dockerfiles: [github.com/voipmonitor/blackwell-llm-docker](https://github.com/voipmonitor/blackwell-llm-docker)

---

## Known Issues

### PYTORCH_CUDA_ALLOC_CONF incompatibility

Do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when using `--enable-pcie-oneshot-allreduce` or `--enable-flashinfer-allreduce-fusion`. The expandable segments allocator breaks CUDA IPC memory handle exchange used by custom allreduce, causing `RuntimeError: invalid argument` at `pcie_allreduce.cu:321`.

### DeepGemm scale format

`DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0` — cosmetic warning. Disabled via `SGL_ENABLE_JIT_DEEPGEMM=0`.

### NCCL P2P deadlocks

If GPUs show 100% utilization at ~140W with no progress, try `NCCL_P2P_LEVEL=2` instead of `4`.
