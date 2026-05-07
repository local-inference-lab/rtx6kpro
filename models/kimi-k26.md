# Kimi-K2.6 on 8x RTX PRO 6000 Blackwell

This is the current practical recipe for running `moonshotai/Kimi-K2.6` with
vLLM on 8x RTX PRO 6000 Blackwell / sm120. It uses the Kimi MLA path:

- target model: `moonshotai/Kimi-K2.6`
- draft model for MTP: `lightseekorg/kimi-k2.5-eagle3-mla`
- attention backend: `TRITON_MLA` for target and draft
- KV cache: `fp8`
- tensor parallel: `TP=8`
- decode context parallelism: `DCP=1`, `DCP=4`, or `DCP=8`
- runtime topology: PCIe P2P custom allreduce + NCCL XML graph file

> **Erratum, 2026-05-07:** the historical decode tables below label the longest
> context row as `128k`, but the benchmark run used by this page produced only
> about `82k-83k` prompt tokens for that row. The old `DCP=1, No MTP, 128k/C1`
> value `56.7 tok/s` is therefore an ~82k-context result. Re-measuring the old
> image on a true ~128.8k-token prompt gives `46.1 tok/s`; the newer v2 image
> gives `45.5 tok/s` on the same true 128k prompt. See
> [`kimi-k26-v2.md`](kimi-k26-v2.md) for the corrected rerun.

## Contents

- [Quick Start](#quick-start)
- [Choose A Profile](#choose-a-profile)
- [Launch: MTP Enabled](#launch-mtp-enabled)
- [Launch: Maximum KV Cache, No MTP](#launch-maximum-kv-cache-no-mtp)
- [Speed Vs Context Length](#speed-vs-context-length)
- [Expected Decode Throughput](#expected-decode-throughput)
- [8-GPU Topology Sanity Check](#8-gpu-topology-sanity-check)
- [Historical Decode Throughput: Marlin FP8 Forcing](#historical-decode-throughput-marlin-fp8-forcing)
- [Prefill Sanity Checks](#prefill-sanity-checks)
- [NCCL XML Status](#nccl-xml-status)
- [Why The MLA Draft](#why-the-mla-draft)
- [FP8 Tensor Draft Check](#fp8-tensor-draft-check)
- [Current Upstream Patch Stack](#current-upstream-patch-stack)
- [Legacy Kimi-K2.5 Community Image](#legacy-kimi-k25-community-image)

## Quick Start

Use this Docker image:

```bash
docker pull voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
```

The image is built from upstream vLLM plus the Kimi/K2.6 MTP patch stack tracked
in:

- vLLM issue: <https://github.com/vllm-project/vllm/issues/40608>
- vLLM draft PR: <https://github.com/vllm-project/vllm/pull/40750>
- detailed work log: [kimi-k26-mtp-long-ctx-wip/](kimi-k26-mtp-long-ctx-wip/README.md)

The current validated startup profile is `DCP=8 + MTP + XML`, without Marlin
FP8 activation forcing, on port `5002`:

```text
GPU KV cache size:           1,822,464 tokens
Maximum concurrency @262144: 6.95x
PCIe custom allreduce:       enabled on all 8 workers
TRITON_MLA:                  target and draft
Marlin FP8 activation force: disabled
```

## Choose A Profile

| Goal | DCP | MTP | max model len | Why |
|---|---:|---:|---:|---|
| Fastest single-stream / short-ctx speed | 1 | on | 150k or 262k | highest C=1 decode speed; smallest KV pool |
| Balanced long-ctx + concurrency | 4 | on | 262k | good speed with much larger KV pool |
| Maximum KV cache with MTP | 8 | on | 262k | best MTP capacity; slightly lower single-stream speed |
| Maximum KV cache / many short requests | 8 | off | 262k | no draft model VRAM; largest KV pool |

For most community validation start with `DCP=8 + MTP + --max-model-len 262144`.
For latency or single-stream comparisons also test `DCP=1 + MTP`. For maximum
batch capacity remove the speculative config and use `DCP=8`.

If you change this recipe to `DCP=4` while keeping `NCCL_GRAPH_FILE`, use a
build or runtime mount that includes the DCP/XML guard. Otherwise startup can
fail when the 8-GPU XML graph is applied to a 4-rank DCP subgroup.

## Launch: MTP Enabled

Set the profile variables first:

```bash
export IMAGE=voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
export PORT=5002
export DCP=8
export MAX_MODEL_LEN=262144
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=128
```

Then start vLLM:

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc "VLLM_ENABLE_PCIE_ALLREDUCE=1 \
NCCL_P2P_LEVEL=SYS \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_LOG_STATS_INTERVAL=1 \
VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.6 \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size ${DCP} \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization 0.94 \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --speculative-config '{\"model\":\"lightseekorg/kimi-k2.5-eagle3-mla\",\"method\":\"eagle3\",\"num_speculative_tokens\":3,\"draft_attention_backend\":\"TRITON_MLA\",\"draft_kv_cache_dtype\":\"fp8\",\"rejection_sample_method\":\"probabilistic\"}'"
```

Sanity check:

```bash
curl http://127.0.0.1:${PORT}/v1/models
```

Run the benchmark:

```bash
python3 /mnt/llm_decode_bench.py \
  --port ${PORT} \
  --model Kimi-K2.6 \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 0,16k,32k,64k,128k \
  --duration 10 \
  --skip-prefill
```

Do not set `VLLM_TEST_FORCE_FP8_MARLIN`,
`VLLM_MARLIN_INPUT_DTYPE=fp8`, or `VLLM_MARLIN_USE_ATOMIC_ADD` for the default
community recipe. The full matrix below showed no useful speed win, and forcing
target MoE activations to runtime FP8 is less quality-safe than leaving the
target path in its normal INT4-weight/BF16-activation mode.

## Launch: Maximum KV Cache, No MTP

Use this when you want the largest KV cache and many concurrent requests more
than single-stream MTP latency. It is the same command as above, but remove the
entire `--speculative-config ...` argument.

Recommended variables:

```bash
export DCP=8
export MAX_MODEL_LEN=262144
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=128
```

Expected KV cache at `--max-model-len 262144` from the current no-Marlin
benchmark matrix:

| Config | KV tokens | Max concurrency |
|---|---:|---:|
| DCP=1, MTP=3 | 343,664 | 1.31x |
| DCP=1, no MTP | 437,264 | 1.67x |
| DCP=4, MTP=3 | 1,152,960 | 4.40x |
| DCP=4, no MTP | 1,686,656 | 6.43x |
| DCP=8, MTP=3 | 1,822,464 | 6.95x |
| DCP=8, no MTP | 3,267,968 | 12.47x |

The DCP=4 rows were measured with the runtime DCP/XML guard described in
[NCCL XML Status](#nccl-xml-status).

## Speed Vs Context Length

`--max-model-len` is a real performance knob, not just a limit. Larger values
increase the captured block-table stride and cost decode throughput even for
short prompts.

Use this for short-context speed:

```bash
export MAX_MODEL_LEN=150000
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=256
```

Use this for full long-context flexibility:

```bash
export MAX_MODEL_LEN=262144
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=128
```

Measured isolation result from the long-context work: changing only
`--max-model-len` from `262144` to `150000` restored short-context
high-concurrency throughput from about `1397 tok/s` to `1527 tok/s` in the
DCP=8 no-MTP diagnostic run. The trade-off is obvious: lower max request length,
better short-context throughput.

## Expected Decode Throughput

Current quality-safe matrix, `llm_decode_bench.py --skip-prefill`, 10 seconds
per cell, aggregate tok/s. Common launch settings:

```text
image:                       voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
target model:                moonshotai/Kimi-K2.6
draft model when MTP is on:  lightseekorg/kimi-k2.5-eagle3-mla
TP / DCP:                    8 / 1, 4, or 8
attention backend:           TRITON_MLA
target KV cache dtype:       fp8
draft KV cache dtype:        fp8
max_model_len:               262144
max_num_batched_tokens:      8192
max_num_seqs:                128
gpu_memory_utilization:      0.94
NCCL graph file:             /mnt/nccl_graph_opt.xml
Marlin FP8 force envs:       disabled
disabled kernel:             MarlinFP8ScaledMMLinearKernel
```

DCP=4 was measured with the runtime DCP/XML guard that keeps XML for full-size
TP/world communicators but unsets `NCCL_GRAPH_FILE` for DCP subgroup
communicators.

### DCP=1 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 112.5 | 176.0 | 261.4 | 454.5 | 729.7 | 1104.0 | 1602.2 | 2044.3 |
| 16k | 99.4 | 163.9 | 227.4 | 384.3 | 542.3 | 730.9 | 905.9 | 1143.2 |
| 32k | 102.5 | 146.1 | 194.8 | 309.2 | 419.3 | 512.2 | 603.2 | 810.8 |
| 64k | 99.5 | 127.3 | 162.1 | 231.7 | 288.2 | 364.6 | 454.0 | 409.1 |
| 128k | 73.6 | 109.3 | 117.3 | 159.0 | 191.3 | 301.1 | 215.2 | 395.5 |

### DCP=1, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 91.5 | 151.3 | 242.6 | 350.0 | 492.7 | 858.8 | 1398.0 | 2164.7 |
| 16k | 84.6 | 141.3 | 218.7 | 318.2 | 429.1 | 700.0 | 1018.1 | 1397.9 |
| 32k | 79.6 | 133.3 | 198.9 | 286.1 | 381.5 | 573.1 | 763.9 | 1016.2 |
| 64k | 70.5 | 119.2 | 171.1 | 246.6 | 318.2 | 445.2 | 509.0 | 635.1 |
| 128k | 56.7 | 99.3 | 135.1 | 191.0 | 238.3 | 286.5 | 318.2 | 381.3 |

### DCP=4 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 101.5 | 156.0 | 219.8 | 381.9 | 597.2 | 925.7 | 1255.1 | 1596.0 |
| 16k | 96.3 | 143.1 | 194.7 | 330.1 | 465.2 | 648.6 | 840.9 | 1144.7 |
| 32k | 87.5 | 135.3 | 187.0 | 259.3 | 368.0 | 500.7 | 705.0 | 855.1 |
| 64k | 83.4 | 114.2 | 145.2 | 221.0 | 275.4 | 362.0 | 467.8 | 641.4 |
| 128k | 73.7 | 93.4 | 118.2 | 155.0 | 198.7 | 242.3 | 315.5 | 424.4 |

### DCP=4, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76.6 | 125.4 | 207.0 | 310.3 | 429.5 | 732.1 | 1145.5 | 1781.3 |
| 16k | 74.5 | 121.2 | 198.8 | 286.3 | 381.7 | 604.6 | 891.2 | 1270.2 |
| 32k | 73.6 | 117.1 | 182.9 | 262.5 | 350.0 | 509.4 | 700.0 | 890.7 |
| 64k | 68.6 | 107.3 | 163.1 | 222.8 | 286.2 | 413.5 | 509.0 | 636.4 |
| 128k | 61.6 | 93.5 | 139.1 | 174.9 | 206.6 | 286.0 | 317.9 | 381.4 |

### DCP=8 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 89.4 | 146.1 | 192.8 | 323.1 | 506.1 | 649.4 | 938.8 | 1168.9 |
| 16k | 88.5 | 132.4 | 184.9 | 299.2 | 463.1 | 599.8 | 892.0 | 1205.2 |
| 32k | 86.4 | 130.1 | 176.0 | 281.4 | 411.5 | 538.7 | 747.1 | 907.5 |
| 64k | 76.6 | 121.2 | 156.9 | 248.5 | 355.0 | 446.2 | 657.2 | 850.4 |
| 128k | 77.6 | 108.4 | 141.1 | 194.8 | 268.2 | 368.9 | 485.8 | 464.2 |

### DCP=8, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75.5 | 117.3 | 190.8 | 278.4 | 365.9 | 604.1 | 1017.7 | 1399.9 |
| 16k | 74.6 | 113.3 | 182.9 | 262.5 | 350.0 | 572.6 | 890.6 | 1144.9 |
| 32k | 74.5 | 109.3 | 175.1 | 254.1 | 333.7 | 509.2 | 826.6 | 1016.3 |
| 64k | 71.5 | 105.3 | 163.1 | 230.8 | 302.1 | 477.4 | 699.2 | 888.2 |
| 128k | 66.6 | 97.4 | 147.0 | 198.8 | 254.5 | 349.8 | 508.7 | 635.4 |

Interpretation:

- MTP is the default for single-stream and latency-sensitive decode. It wins
  C=1 in all DCP=1/DCP=4 cells and most DCP=8 cells.
- No-MTP is the default for maximum KV cache and high-concurrency throughput.
  It wins many C=128 cells because the target path is already saturated and
  speculative verification adds work.
- DCP=1 + MTP is the fastest single-stream profile, but has the smallest KV
  pool.
- DCP=8 no-MTP has the largest KV pool and best high-concurrency long-context
  throughput.
- DCP=8 + MTP is the best public "large KV plus speculation enabled" profile.
- `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000` is obsolete for this image. Do
  not use it for the Kimi-K2.6 MTP path.

## 8-GPU Topology Sanity Check

This is a narrow hardware sanity check from 2026-04-28, not the main public
throughput table above. It compares the same DCP=8 no-MTP recipe across two
hosts and across the two independent 8-GPU slices on the 16-GPU host.

Remote topology note: each tested 8-GPU slice is `8x RTX PRO 6000 Blackwell`
behind `2x C-Payne PCIe switches` with one uplink path. The 16-GPU host exposes
two such 8-GPU slices. The test keeps the 8-GPU NCCL XML enabled for these
8-rank runs.

Common settings:

```text
image:                  voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
model:                  moonshotai/Kimi-K2.6
TP / DCP:               8 / 8
MTP:                    off
attention backend:      TRITON_MLA
KV cache dtype:         fp8
max_model_len:          262144
max_num_batched_tokens: 8192
max_num_seqs:           128
gpu_memory_utilization: 0.94
NCCL graph file:        /mnt/nccl_graph_opt.xml
custom allreduce:       VLLM_ENABLE_PCIE_ALLREDUCE=1
benchmark:              llm_decode_bench.py --skip-prefill --duration 30
```

### Cross-Host 8-GPU No-MTP Check

| cell | budgetserver local | 10.229.14.14 GPU 0-7 | local delta |
|---|---:|---:|---:|
| 0 / 1 | 75.2 | 72.7 | +3.4% |
| 0 / 128 | 1245.9 | 1232.2 | +1.1% |
| 64k / 1 | 67.0 | 65.4 | +2.4% |
| 64k / 16 | 267.7 | 259.5 | +3.2% |

Interpretation: without MTP, the two hosts are close. The local host is only
about `+1%` to `+3%` faster on these targeted cells, so large MTP-only gaps
should not be interpreted as raw target-model or generic PCIe throughput gaps.

### 10.229.14.14 8-GPU Slice Check

| cell | GPU 0-7 | GPU 8-15 | second-slice delta |
|---|---:|---:|---:|
| 0 / 1 | 72.7 | 72.2 | -0.7% |
| 0 / 128 | 1232.2 | 1230.3 | -0.2% |
| 64k / 1 | 65.4 | 65.0 | -0.6% |
| 64k / 16 | 259.5 | 257.8 | -0.7% |

Interpretation: the two 8-GPU slices on `10.229.14.14` are effectively the
same for this no-MTP target path. Any large difference seen elsewhere is not
explained by simply choosing the first or second 8-GPU slice.

### 10.229.14.14 TP=16 / DCP=1 No-MTP Diagnostic

This is a separate diagnostic run on the same 16-GPU host using all 16 GPUs at
once. It does **not** use the 8-GPU XML file, because
`/mnt/nccl_graph_opt.xml` is an 8-rank graph (`dev 0..7`). vLLM also disables
the PCIe custom allreduce path for this run:

```text
Custom allreduce is disabled due to an unsupported world size: 16.
Supported world sizes: [2, 4, 6, 8].
```

Common settings:

```text
image:                  voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
model:                  moonshotai/Kimi-K2.6
TP / DCP:               16 / 1
MTP:                    off
attention backend:      TRITON_MLA
KV cache dtype:         fp8
max_model_len:          262144
max_num_batched_tokens: 8192
max_num_seqs:           128
gpu_memory_utilization: 0.94
NCCL graph file:        not used
allreduce path:         NCCL, because vLLM custom allreduce does not support world size 16
GPU KV cache size:      1,507,120 tokens
benchmark:              llm_decode_bench.py --standalone-prefill --duration 30
```

Prefill, C=1:

| ctx | prompt tokens | TTFT s | prefill tok/s | samples |
|---|---:|---:|---:|---:|
| 8k | 8,187 | 1.01 | 8,102 | 5 |
| 64k | 64,459 | 9.12 | 7,067 | 2 |
| 128k | 128,766 | 20.84 | 6,178 | 1 |

Decode, aggregate tok/s:

| ctx \ conc | 1 | 16 | 128 |
|---|---:|---:|---:|
| 0 | 75.9 | 780.5 | 2582.5 |
| 64k | 54.3 | 324.0 | does not fit |
| 128k | 42.1 | does not fit | does not fit |

The requested `128k / C=128` cell does not fit in the available KV cache. With
`max_tokens=2048`, it would require roughly `17.0M` total KV tokens, while this
TP16/DCP1 run has `1.51M` tokens available.

### 10.229.14.14 Turin Retest: K2.6 FP8 Tensor Draft

This 2026-04-30 retest used the same Turin host after the platform change, but
with a newly converted K2.6 Eagle3 MLA draft instead of the older K2.5 draft.
The purpose was a topology/runtime sanity check, not a full tuning sweep.

Common settings:

```text
image:                  voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
target model:           moonshotai/Kimi-K2.6
draft model:            /mnt/kimi-k2.6-eagle3-mla-fp8-tensor
draft source:           lightseekorg/kimi-k2.6-eagle3-mla
MTP:                    Eagle3, num_speculative_tokens=3
attention backend:      TRITON_MLA
KV cache dtype:         fp8
max_model_len:          262144
max_num_batched_tokens: 8192
max_num_seqs:           128
gpu_memory_utilization: 0.94
benchmark:              llm_decode_bench.py --skip-prefill --duration 10 --decode-warmup-seconds 20
```

The 8-GPU run used GPUs `0..7` and the existing 8-rank NCCL XML graph. The
16-GPU run used GPUs `0..15` and intentionally did not use
`/mnt/nccl_graph_opt.xml`, because that XML only contains `dev 0..7`. vLLM also
disables PCIe custom allreduce at world size 16 in this build.

| Run | TP/DCP | KV tokens | Result JSON |
|---|---:|---:|---|
| First 8 GPUs | 8 / 1 | 357,232 | `/mnt/kimi_k26_turin8_k26draft_fp8_tensor_decode_full_20260430.json` |
| All 16 GPUs | 16 / 1 | 1,402,656 | `/mnt/kimi_k26_turin16_k26draft_fp8_tensor_decode_keypoints_20260430.json` |

8-GPU full decode matrix, aggregate tok/s:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 125.0 | 215.6 | 353.3 | 528.3 | 805.3 | 1167.0 | 1887.8 | 2510.3 |
| 16k | 108.3 | 182.9 | 286.4 | 383.9 | 484.3 | — | — | — |
| 32k | 97.2 | 155.1 | 237.7 | 291.5 | — | — | — | — |
| 64k | 81.0 | 130.7 | 173.0 | — | — | — | — | — |
| 128k | 55.1 | 90.3 | — | — | — | — | — | — |

16-GPU key points, aggregate tok/s:

| ctx \ conc | 1 | 8 | 32 | 128 |
|---|---:|---:|---:|---:|
| 0 | 129.6 | 639.3 | 1399.5 | 2959.6 |
| 16k | 119.9 | 448.0 | 659.6 | — |
| 64k | 84.8 | 214.9 | — | — |
| 128k | 61.8 | 120.3 | — | — |

Server-reported speculative accept rate:

| Run | ctx/C1 | ctx0/C128 | 128k/C1 | 128k/C8 |
|---|---:|---:|---:|---:|
| 8 GPU | 0.316 | 0.430 | 0.309 | — |
| 16 GPU | 0.463 | 0.470 | 0.284 | 0.417 |

Interpretation: TP16 increases KV capacity and improves the long-context C=1
cells, but it is not a linear throughput scale-up. The world-size-16 run cannot
use the current PCIe custom allreduce path and was measured without the 8-rank
NCCL XML graph.

## Historical Decode Throughput: Marlin FP8 Forcing

Earlier matrix preserved for comparison. This used the old default recipe with
Marlin FP8 forcing envs enabled:

```text
VLLM_TEST_FORCE_FP8_MARLIN=1
VLLM_MARLIN_INPUT_DTYPE=fp8
VLLM_MARLIN_USE_ATOMIC_ADD=1
```

That path is no longer the recommended public recipe.

### DCP=1 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 114.1 | 178.9 | 243.7 | 417.4 | 659.0 | 1055.2 | 1522.0 | 1992.3 |
| 16k | 108.1 | 166.4 | 226.4 | 353.3 | 502.9 | 680.6 | 879.5 | 1080.1 |
| 32k | skip | 151.6 | 189.1 | 293.3 | 396.6 | 495.4 | 580.2 | 605.7 |
| 64k | 85.6 | 136.3 | 157.5 | 222.8 | 286.5 | 354.1 | 420.4 | 384.8 |
| 128k | 66.6 | 105.4 | 121.2 | 152.6 | 186.5 | 211.1 | 189.1 | 374.1 |

### DCP=1, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 88.5 | 148.4 | 233.8 | 340.7 | 475.2 | 823.6 | 1330.2 | 2026.3 |
| 16k | 82.6 | 139.1 | 214.6 | 310.2 | 413.8 | 667.9 | 954.0 | 1272.2 |
| 32k | 77.5 | 129.4 | 197.8 | 286.2 | 366.1 | 572.4 | 758.7 | 890.3 |
| 64k | 68.6 | 117.2 | 170.9 | 246.5 | 302.1 | 411.8 | 508.9 | 635.5 |
| 128k | 55.4 | 97.3 | 131.3 | 190.7 | 222.6 | 286.0 | 317.9 | 380.5 |

### DCP=4 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 98.8 | 160.0 | 211.4 | 363.3 | 576.5 | 853.2 | 1239.1 | 1506.2 |
| 16k | 96.4 | 145.1 | 190.0 | 311.0 | 453.4 | 603.1 | 741.0 | 891.9 |
| 32k | 88.5 | 130.2 | 180.8 | 266.2 | 367.5 | 451.9 | 575.0 | 597.2 |
| 64k | 76.5 | 114.3 | 151.9 | 205.4 | 267.5 | 303.4 | 404.9 | 432.4 |
| 128k | 67.6 | 87.4 | 112.4 | 152.0 | 175.8 | 213.5 | 247.7 | 383.9 |

### DCP=4, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.5 | 122.5 | 205.2 | 308.2 | 426.6 | 726.2 | 1136.1 | 1640.3 |
| 16k | 73.5 | 117.3 | 193.5 | 278.5 | 365.8 | 600.0 | 827.1 | 1145.0 |
| 32k | 71.5 | 113.5 | 182.8 | 254.7 | 334.0 | 509.1 | 694.1 | 882.4 |
| 64k | 66.7 | 105.3 | 163.0 | 222.7 | 270.3 | 379.0 | 508.8 | 629.3 |
| 128k | 60.3 | 92.8 | 135.1 | 174.7 | 206.7 | 254.6 | 318.0 | 379.7 |

### DCP=8 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.4 | 142.7 | 176.1 | 311.3 | 498.3 | 641.2 | 908.0 | 1137.0 |
| 16k | 85.4 | 131.2 | 186.9 | 297.1 | 466.0 | 590.7 | 744.4 | 861.0 |
| 32k | 87.4 | 121.1 | 163.9 | 275.8 | 400.2 | 520.8 | 699.6 | 828.2 |
| 64k | 79.3 | 114.2 | 156.1 | 237.3 | 340.5 | 432.9 | 560.7 | 581.7 |
| 128k | 75.4 | 99.4 | 135.2 | 197.9 | 252.2 | 312.5 | 417.6 | 416.2 |

### DCP=8, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.1 | 113.4 | 185.8 | 276.6 | 363.5 | 600.9 | 948.4 | 1260.6 |
| 16k | 73.6 | 111.2 | 178.7 | 262.2 | 333.9 | 571.3 | 826.7 | 1137.4 |
| 32k | 73.0 | 109.3 | 174.8 | 245.0 | 317.7 | 509.3 | 757.7 | 1008.6 |
| 64k | 69.7 | 104.7 | 162.9 | 229.1 | 301.8 | 445.7 | 632.2 | 757.0 |
| 128k | 65.6 | 95.5 | 146.8 | 197.5 | 252.9 | 350.0 | 476.8 | 635.3 |

## Prefill Sanity Checks

DCP=1 + MTP=3 prefill, C=1:

| ctx | prompt tokens | TTFT | prefill tok/s |
|---|---:|---:|---:|
| 8k | 5,312 | 0.71s | 7,892 |
| 16k | 10,480 | 1.44s | 7,478 |
| 32k | 20,822 | 2.94s | 7,177 |
| 64k | 41,505 | 6.48s | 6,449 |
| 128k | 82,845 | 14.82s | 5,604 |

If your numbers are much lower, first check:

- `NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml` is mounted and set.
- `VLLM_ENABLE_PCIE_ALLREDUCE=1` is present.
- logs say `PCIe custom allreduce enabled via VLLM_ENABLE_PCIE_ALLREDUCE=1`.
- logs say `Using AttentionBackendEnum.TRITON_MLA backend`.
- `--max-model-len` matches the profile you are comparing against.

## NCCL XML Status

The public recipe still uses:

```bash
NCCL_P2P_LEVEL=SYS
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
```

DCP=4 with XML needs the runtime DCP/XML guard used for the matrix above. The
guard keeps `NCCL_GRAPH_FILE` for full TP/world communicators, but temporarily
unsets it while creating and using DCP subgroup communicators. Without that
guard, NCCL can try to apply the full 8-GPU XML graph to a 4-rank DCP subgroup
and fail during communicator initialization. DCP=8 does not hit that partial
subgroup mismatch because the DCP group size matches TP size.

Without the XML, NCCL historically picked a much worse ring graph on this Turin
system. The bad no-XML path showed single cold 8k prefill around `876 tok/s`,
while XML was around `7478 tok/s`.

There is now an upstream NCCL draft PR for the no-XML topology issue:

- <https://github.com/NVIDIA/nccl/pull/2127>

With that NCCL fix, the targeted single cold 8k prefill reproducer reached
`7455 tok/s`, effectively matching XML. Until that fix is released in a normal
NCCL package, keep the XML file in the public recipe.

## Why The MLA Draft

Use `lightseekorg/kimi-k2.5-eagle3-mla` rather than a Llama-style Eagle draft
because it exercises the same MLA runtime path as Kimi:

- `TRITON_MLA` target and draft
- fp8 KV cache target and draft
- DCP correctness path
- Kimi tool-call and reasoning parsers

The non-MLA Llama draft is useful for separate debugging, but it is not the
representative public Kimi serving path.

## FP8 Tensor Draft Check

An experimental locally quantized draft was tested against the public
`lightseekorg/kimi-k2.5-eagle3-mla` draft to decide whether the FP8 tensor draft
should become the default recipe.

Common server settings:

```text
target model:              moonshotai/Kimi-K2.6
TP / DCP:                  8 / 1
attention backend:         TRITON_MLA for target and draft
target KV cache dtype:     fp8
draft KV cache dtype:      fp8
speculative tokens:        3
rejection sample method:   probabilistic
max_num_batched_tokens:    8192
gpu_memory_utilization:    0.97
NCCL graph file:           /mnt/nccl_graph_opt.xml
```

Only the draft model changed:

| Variant | Draft model |
|---|---|
| Public draft | `lightseekorg/kimi-k2.5-eagle3-mla` |
| Local FP8 tensor draft | `/mnt/kimi-k2.5-eagle3-mla-fp8-tensor` |

Benchmark command:

```bash
python3 /mnt/llm_decode_bench.py \
  --port 5002 \
  --model Kimi-K2.6 \
  --concurrency 1,16,64 \
  --contexts 0,16k,32k,64k,128k \
  --duration 20 \
  --max-tokens 512 \
  --skip-prefill
```

Manual KV budgets were set from the vLLM startup log:

| Variant | GPU KV cache size |
|---|---:|
| Public draft | 424,576 tokens |
| Local FP8 tensor draft | 431,328 tokens |

Decode result, aggregate tok/s:

| ctx / conc | Local FP8 tensor draft | Public draft | Delta |
|---|---:|---:|---:|
| 0 / 1 | 129.3 | 114.4 | +13.0% |
| 0 / 16 | 747.5 | 758.5 | -1.4% |
| 0 / 64 | 1603.3 | 1614.5 | -0.7% |
| 16k / 1 | 119.1 | 122.2 | -2.6% |
| 16k / 16 | 524.8 | 523.2 | +0.3% |
| 32k / 1 | 102.4 | 105.4 | -2.8% |
| 64k / 1 | 92.4 | 89.5 | +3.2% |
| 128k / 1 | 70.5 | 72.0 | -2.0% |

Conclusion: the local FP8 tensor draft is not a better default. It only clearly
wins the `0ctx / C=1` cell, is otherwise within noise or slightly slower, and
adds only about 1.6% more KV cache. The default community recipe should keep the
public `lightseekorg/kimi-k2.5-eagle3-mla` draft with `draft_kv_cache_dtype=fp8`.
The local FP8 tensor draft can remain an experimental option for local testing.

## Current Upstream Patch Stack

The current reconstruction branch includes:

- `vllm-project/vllm#39633`: explicit PCIe custom-allreduce opt-in
- `vllm-project/vllm#40609`: MLA + DCP + fp8 KV support
- `vllm-project/vllm#40610`: async proposer synchronization fix
- `vllm-project/vllm#40611`: draft-specific attention backend and KV dtype
- `vllm-project/vllm#40750`: TRITON_MLA full CUDA graphs, DCP correctness,
  batch-aware KV split selection, and sm120/fp8 tuning table
- runtime DCP/XML guard for partial DCP validation: unset `NCCL_GRAPH_FILE`
  only around DCP subgroup communicators when `1 < DCP < TP`

The draft PR is intentionally a runnable consolidation branch. It can be split
for upstream review after the end-to-end recipe is fully validated.

## Legacy Kimi-K2.5 Community Image

The older public Kimi-K2.5 image remains useful for historical comparison:

```bash
voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422
```

Do not use that image as the current Kimi-K2.6 recipe. It predates the final
Kimi-K2.6 MTP full-CG path, still documented the old speculation kill-switch,
and used older benchmark assumptions.
