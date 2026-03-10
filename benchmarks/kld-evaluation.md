# KLD Evaluation for Quantized Models

Measure how much quality is lost in quantized models (e.g. NVFP4) compared to a higher-precision reference (FP8) using KL divergence over full vocabulary logit distributions.

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [How It Works](#how-it-works)
- [Step-by-Step Guide](#step-by-step-guide)
- [Automation Script](#automation-script)
- [Interpreting Results](#interpreting-results)
- [Known Issues](#known-issues)

---

## Overview

Standard benchmarks (MMLU, HumanEval, etc.) are noisy and coarse. KL divergence measures the **exact difference in output probability distributions** between two models, giving a much more sensitive quality metric.

**Reference model:** `Qwen/Qwen3.5-397B-A17B-FP8` (TP8, 8× RTX 6000 Pro)
**Test model:** `nvidia/Qwen3.5-397B-A17B-NVFP4` (TP4, 4× RTX 6000 Pro)
**Dataset:** WikiText-2, 100 sliding windows (2048 tokens, stride 512), 204,800 total positions

## Results

```
KLD Evaluation Results (ref: Qwen3.5-397B-A17B-FP8, dataset: wikitext-2, 204,800 positions)
============================================================================================

Model                                      Mean KLD   Median KLD    P95 KLD    P99 KLD    Max KLD
------------------------------------------------------------------------------------------------
nvidia/Qwen3.5-397B-A17B-NVFP4            0.108530     0.027253   0.471153   1.409069    19.6018
```

### Interpretation

| Mean KLD | Quantization quality |
|----------|---------------------|
| < 0.01 | Near-lossless |
| 0.01 – 0.05 | Good, minimal quality loss |
| 0.05 – 0.1 | Noticeable quality loss |
| **> 0.1** | **Significant quality loss** |

The nvidia NVFP4 checkpoint shows:
- **Mean 0.108** — significant overall quality loss vs FP8
- **Median 0.027** (4× lower than mean) — most positions are OK, but the heavy tail drags the mean up
- **P95 = 0.47** — 5% of positions have substantial divergence
- **Max = 19.6** — some positions have completely different distributions

**Bottom line:** Usable for casual chat, but precision-sensitive tasks (code, math, long reasoning chains) will be noticeably degraded compared to FP8.

---

## How It Works

### Problem

SGLang only exposes top-k logprobs via its API, not full vocabulary logits. KLD needs full distributions over all 152,064 tokens.

### Solution

1. **Patch SGLang** at runtime to capture full `[N, vocab_size]` log-probability tensors during prefill
2. **Run reference model** (FP8) on sliding windows over WikiText-2, saving logits to disk as safetensors
3. **Run test model** (NVFP4) on the same windows, saving logits to disk
4. **Compute KLD** between reference and test logit distributions

### Architecture

```
Phase 1: FP8 Reference (TP8)          Phase 2: NVFP4 Test (TP4)
┌─────────────────────┐               ┌─────────────────────┐
│ SGLang Server       │               │ SGLang Server       │
│ + logit capture     │               │ + logit capture     │
│   patch             │               │   patch             │
└────────┬────────────┘               └────────┬────────────┘
         │ saves logits                        │ saves logits
         ▼                                     ▼
   /tmp/kld_ref/                         /tmp/kld_test/
   ├── 0.safetensors                     ├── 0.safetensors
   ├── 1.safetensors         ──KLD──►    ├── 1.safetensors
   └── ...99.safetensors                 └── ...99.safetensors
```

### Storage requirements

- Per window: 2048 × 152,064 × 2 bytes ≈ **594 MB**
- 100 windows = **~58 GB** per model
- Total for ref + one test: **~120 GB**
- Runtime: ~60s per phase (100 windows), KLD compute takes seconds

### What the patch does

The patch (`patches/sglang-kld-logit-capture.py`) modifies `LogitsProcessor.forward()` in SGLang to insert a `_kld_maybe_save()` hook:

```python
# BEFORE (in LogitsProcessor.forward, non-chunked path):
input_logits = logits[input_logprob_indices]
del logits
logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)

# AFTER:
input_logits = logits[input_logprob_indices]
del logits
_kld_maybe_save(input_logits)  # ← saves full [N, 152064] log-softmax
logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)
```

The hook:
- Is a no-op unless `SGLANG_KLD_SAVE_DIR` env var is set
- Only saves from TP rank 0 (avoids duplicate writes across tensor-parallel workers)
- Trims TP padding columns to actual `vocab_size` (controlled by `SGLANG_KLD_VOCAB_SIZE`, default 152064)
- Computes `log_softmax` in float32 for numerical stability, saves as float16 safetensors

---

## Step-by-Step Guide

### Prerequisites

- Docker image: [`llm-pytorch-blackwell:nightly`](https://github.com/voipmonitor/llm-pytorch-blackwell)
- 8× GPUs for FP8 reference (TP8), 4× GPUs for NVFP4 test (TP4)
- ~120 GB free disk space in `/tmp`
- Files from [`llm-pytorch-blackwell`](https://github.com/voipmonitor/llm-pytorch-blackwell) repo:
  - `patches/sglang-kld-logit-capture.py`
  - `scripts/sglang_kld_eval.py`

### Step 1: Start container

```bash
cd /path/to/llm-pytorch-blackwell

docker run --rm -it \
  --runtime nvidia --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 5000:5000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-nightly-jit:/cache/jit \
  -v /tmp/kld_ref:/tmp/kld_ref \
  -v /tmp/kld_test:/tmp/kld_test \
  -v $PWD/patches/sglang-kld-logit-capture.py:/workspace/sglang-kld-logit-capture.py:ro \
  -v $PWD/scripts/sglang_kld_eval.py:/workspace/sglang_kld_eval.py:ro \
  llm-pytorch-blackwell:nightly \
  bash
```

### Step 2: Apply the logit capture patch

Inside the container:

```bash
python /workspace/sglang-kld-logit-capture.py
```

Expected output:
```
OK: KLD logit capture patch applied to /opt/sglang/python/sglang/srt/layers/logits_processor.py
```

### Step 3: Run FP8 reference server

```bash
SGLANG_KLD_SAVE_DIR=/tmp/kld_ref \
SGLANG_KLD_VOCAB_SIZE=152064 \
SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0 \
python -m sglang.launch_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --tp 8 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.90 \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port 5000
```

**Important env vars:**

| Variable | Purpose |
|----------|---------|
| `SGLANG_KLD_SAVE_DIR` | Directory to save logit files. Hook is disabled if unset. |
| `SGLANG_KLD_VOCAB_SIZE` | Actual vocab size (default: 152064). Trims TP padding columns. |
| `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0` | Forces non-chunked logits path where the hook is inserted. |

### Step 4: Generate reference logits

From a **second terminal**, exec into the running container:

```bash
docker exec -it <container_id> \
  python /workspace/sglang_kld_eval.py --phase ref \
    --server-url http://localhost:5000 \
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
    --logits-dir /tmp/kld_ref
```

This sends 100 sliding windows (2048 tokens, stride 512) from WikiText-2 to the server. Each window triggers the logit capture hook, saving a `{i}.safetensors` file.

Expected output:
```
Window 100/100: 0.6s, saved 594 MB
Done. 100 windows in 61.6s
Files saved: 100
First file shape: torch.Size([2048, 152064])
```

### Step 5: Stop reference server, start test model

Ctrl+C the server in the first terminal, then start the test model:

```bash
# Clear test dir
rm -rf /tmp/kld_test/*

SGLANG_KLD_SAVE_DIR=/tmp/kld_test \
SGLANG_KLD_VOCAB_SIZE=152064 \
SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0 \
NCCL_P2P_LEVEL=SYS \
python -m sglang.launch_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tp 4 --trust-remote-code \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --mem-fraction-static 0.85 \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port 5000
```

> **Note:** Do NOT add `--speculative-*` flags. Speculative decoding changes the logit computation path and will produce incorrect results.

### Step 6: Generate test logits

From the second terminal:

```bash
docker exec -it <container_id> \
  python /workspace/sglang_kld_eval.py --phase test \
    --server-url http://localhost:5000 \
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
    --logits-dir /tmp/kld_test
```

> **Important:** Always use the **same tokenizer** for both ref and test phases to ensure identical sliding windows.

### Step 7: Compute KLD

```bash
python /workspace/sglang_kld_eval.py --phase compute \
  --ref-dir /tmp/kld_ref \
  --test-dirs /tmp/kld_test \
  --test-names "nvidia/NVFP4"
```

You can compare multiple test models at once:

```bash
python /workspace/sglang_kld_eval.py --phase compute \
  --ref-dir /tmp/kld_ref \
  --test-dirs /tmp/kld_test_nvidia /tmp/kld_test_sehyo \
  --test-names "nvidia/NVFP4" "Sehyo/NVFP4"
```

---

## Automation Script

For running the full pipeline with less manual work, use `scripts/kld_eval_pipeline.sh`:

```bash
# Generate FP8 reference logits
./scripts/kld_eval_pipeline.sh ref

# Generate test model logits
./scripts/kld_eval_pipeline.sh test nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --quantization modelopt_fp4 --attention-backend triton

# Compute KLD for all test models
./scripts/kld_eval_pipeline.sh compute

# Or run everything in one go
./scripts/kld_eval_pipeline.sh all nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --quantization modelopt_fp4 --attention-backend triton
```

Configuration via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `KLD_IMAGE` | `llm-pytorch-blackwell:nightly` | Docker image |
| `KLD_REF_MODEL` | `Qwen/Qwen3.5-397B-A17B-FP8` | Reference model |
| `KLD_REF_TP` | `8` | Reference TP size |
| `KLD_TEST_TP` | `4` | Test model TP size |
| `KLD_VOCAB_SIZE` | `152064` | Vocabulary size |
| `KLD_PORT` | `5000` | Server port |
| `KLD_BASE_DIR` | `/tmp/kld` | Base directory for logits |
| `KLD_HF_CACHE` | `/root/.cache/huggingface` | HuggingFace cache path |

---

## Interpreting Results

### KLD scale

| Mean KLD | Quantization quality |
|----------|---------------------|
| < 0.01 | Near-lossless |
| 0.01 – 0.05 | Good, minimal quality loss |
| 0.05 – 0.1 | Noticeable quality loss |
| > 0.1 | Significant quality loss |

### What the metrics mean

- **Mean KLD** — average divergence across all token positions. The primary quality metric.
- **Median KLD** — if much lower than mean, the distribution has a heavy right tail (a few positions are very wrong, most are fine).
- **P95 / P99** — tail behavior. High P95 means 5% of positions have substantially different predictions than the reference.
- **Max KLD** — worst single position. Values > 10 indicate completely broken predictions at some positions.

### KLD formula

For each token position, KLD is computed as:

```
KL(P_ref || Q_test) = Σ_x  P_ref(x) · (log P_ref(x) - log Q_test(x))
```

Where the sum is over all 152,064 vocabulary tokens. This measures how many bits of information are lost when using the test model's distribution instead of the reference.

---

## Known Issues

### Sehyo/Qwen3.5-397B-A17B-NVFP4 produces NaN on SGLang

Sehyo's checkpoint uses `compressed-tensors` quantization format. SGLang's `compressed-tensors` weight loader does not support `linear_attn` layers used by Qwen3.5-397B's mixed attention architecture (3 linear attention layers + 1 full attention, repeating). All `linear_attn` weights fail to load (`Parameter not found in params_dict`), leaving 45 out of 60 attention layers uninitialized, which produces 100% NaN logits.

The nvidia checkpoint works because it uses the `modelopt` format which has a dedicated loader that correctly maps `linear_attn` weights.

**Workaround:** None on current SGLang. vLLM may have better `compressed-tensors` support for this architecture.

### TP padding in logits

With tensor parallelism, SGLang pads the vocabulary dimension to a multiple of TP size. For Qwen3.5 (vocab_size=152,064) with TP8, logits are padded to 248,320 columns. The patch trims these padding columns via `SGLANG_KLD_VOCAB_SIZE` before computing log-softmax. Without trimming, the padding columns (containing garbage values) corrupt the probability distribution.

If you see logit shapes like `[2048, 248320]` instead of `[2048, 152064]`, set `SGLANG_KLD_VOCAB_SIZE=152064`.

### Chunked logits processing

The patch only hooks the non-chunked logits path. Set `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0` to ensure this path is used. With 2048-token windows this is fine — chunking is only needed for very large prefills.

### Speculative decoding

Do not enable speculative decoding (`--speculative-*` flags) during KLD evaluation. Speculative decoding uses a different logit computation path and the capture hook may save incorrect or extra logits.
