# DeepSeek-V4-Flash DSpark — True 4-bit (NVFP4) KV Cache on 2× GPUs

True 4-bit E2M1 KV cache (`--kv-cache-dtype nvfp4_ds_mla`) for DeepSeek-V4-Flash
sparse-MLA on 2× RTX PRO 6000 (SM120), on top of the DSpark stack. Compared to
`fp8_ds_mla` on the same build: **1.47× more KV tokens per GiB, +25–34% decode**,
and start-position needle recall verified past **1M real tokens**. The fp8 path
is byte-identical and re-validated green on the same build (nvfp4 is a strict
addition).

Prior context: an earlier true-4-bit attempt in the DSpark line corrupted output
beyond ~411 prompt tokens. Root cause was page geometry — the GLM 512-wide layout
applied to DeepSeek-V4's 448-wide NoPE. This port implements the DSV4-native
layout instead, and reuses the fp8 tensor-core math path unchanged (every E2M1
magnitude is exactly representable in e4m3, so kernels expand nibbles in-place
and the existing block-scaled MMA + UE8M0 scales reproduce values losslessly).

## Table of Contents

- [Docker Image](#docker-image)
- [What's In The Patch Series](#whats-in-the-patch-series)
- [KV Page Layout](#kv-page-layout)
- [Launch Command — Standing Config (262K)](#launch-command--standing-config-262k)
- [Launch Command — 1M Context](#launch-command--1m-context)
- [Benchmarks](#benchmarks)
- [Quality](#quality)
- [Known Issues & Tips](#known-issues--tips)
- [Source & Credits](#source--credits)

---

## Docker Image

```text
danielwoz/vllm:dspark-nvfp4-cu132-20260708
danielwoz/vllm@sha256:4fc2e9ef3d0a489b892f330484078090d8c77125f292b21c560df649da7238eb
```

Built on the DSpark image with the patch series applied and the sparse-MLA
kernel recompiled into the AOT slot (no JIT at startup). The full series,
Dockerfile, and validation harnesses ship in the image at `/opt/nvfp4/` and at
the source repo below.

| Component | Pin |
|---|---|
| Base image | `fraserpricee/vllm:dspark-cu132-20260627` |
| FlashInfer | `0.6.13+cu132` (base image) + `01-flashinfer-cuda` series |
| vLLM | DSpark fork from base image + `03-vllm-nvfp4` series |
| Patch series | `https://github.com/danielwoz/vllm-dspark-nvfp4` @ `c633834` |
| CUDA | 13.2 runtime, arch `12.0f` |

## What's In The Patch Series

- `01-flashinfer-cuda/` — `ModelType::DSV4_NVFP4` + KV-cache traits, packed-byte
  TMA IO (byte-identical fp8 `arrive.expect_tx` protocol; fp8 is the offset-0
  case of the same path), in-place E2M1→fp8-e4m3 nibble expand in the decode and
  prefill consumers. ~180 lines.
- `02-flashinfer-python/` — dispatch gates, `kv_scale_format="nvfp4"`, page-size
  asserts relaxed to `{584, 360}`.
- `03-vllm-nvfp4/` — `nvfp4_ds_mla` dtype registration and plumbing, two Triton
  store kernels (SWA store + compressed-cache variant), a standalone q-transform
  Triton kernel (bit-exact vs the fused fp8 C++ op, difftest included), plus two
  standalone bugfixes that also apply to fp8 (`compressor.py` missing
  `cache_dtype_str`/`model_version` on `SlidingWindowMLASpec`; `sparse_swa.py`
  `get_kv_cache_shape` missing dtype branch).

## KV Page Layout

360 B/token (fp8_ds_mla is 584), mirroring the fp8 footer structure:

```text
data slot (stride 352): [0:224) 448× E2M1 nibbles (2/byte) | [224:352) 64× bf16 RoPE
footer @ pbs*352 + t*8: 7× per-64 UE8M0 scale + 1 pad     (footer identical to fp8)
scale: per-64 UE8M0 chosen so max|v| / 2^(e-127) ≤ 6.0 (E2M1 max), RTN via hardware cvt
```

## Launch Command — Standing Config (262K)

DSpark spec-decode + full cudagraphs. Note `--gpu-memory-utilization 0.96+`:
0.90 leaves too little KV headroom after graph capture with DSpark.

```bash
docker run --gpus '"device=0,1"' --ipc=host -p 8000:8000 \
  -v /path/to/hf-cache:/hf -e HF_HOME=/hf \
  danielwoz/vllm:dspark-nvfp4-cu132-20260708 \
    fraserprice/DeepSeek-V4-Flash-Abliterated-DSpark \
    --tensor-parallel-size 2 --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
    --tokenizer-mode deepseek_v4 --trust-remote-code \
    --speculative-config '{"method":"dspark","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
    --attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend=marlin \
    --max-model-len 262144 --gpu-memory-utilization 0.96 \
    --max-num-seqs 32 --max-num-batched-tokens 2048 \
    --disable-custom-all-reduce --async-scheduling \
    --enable-chunked-prefill --enable-prefix-caching \
    --max-cudagraph-capture-size 64 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --reasoning-parser deepseek_v4 --enable-auto-tool-choice --tool-call-parser deepseek_v4
```

Memory info (2× RTX PRO 6000, TP2):

```text
GPU KV cache size: 1,539,217 tokens
Maximum concurrency for 262,144 tokens per request: 5.87x
```

## Launch Command — 1M Context

Same stack with `--max-model-len 1048576 --enforce-eager` (cudagraphs off buys
the KV headroom) and `--gpu-memory-utilization 0.96`:

```text
GPU KV cache size: 2,935,863 tokens
```

Needle at position zero recalled at **1,030,039 real prompt tokens**.

## Benchmarks

Same build, same config, fp8 vs nvfp4 A/B (TP2, DSpark, cudagraphs, util 0.96,
262K max-len):

| Metric | fp8_ds_mla | nvfp4_ds_mla |
|---|---|---|
| KV footprint | 15.3 KB/token | **10.4 KB/token (1.47×)** |
| KV pool (same run config) | 591,951 tokens | 601,457 tokens |
| KV pool (tuned standing config) | — | **1,539,217 tokens** |
| Decode (DSpark accepted) | ~195 tok/s | **243–262 tok/s (+25–34%)** |
| DSpark acceptance | 3.55 / 4 | 3.67 / 4 |

Decode gain tracks the KV byte reduction — sparse-MLA decode gathers are
bandwidth-bound, and nvfp4 moves 1.47× fewer KV bytes.

## Quality

| Test | fp8 reference | nvfp4 |
|---|---|---|
| HumanEval-164 | 85.0% | **90.2%** |
| MBPP-100 | — | **94%** |
| Needle/garble sweep (100..8000 real toks) | all green | **all green** |
| Start-position needle @1,030,039 real toks | — | **recalled** |
| fp8 regression on the nvfp4 build | all green (zero-diff) | — |

Harnesses (`kvtest.py`, `qualbench.py`, `ctxbench.py`) ship in the image at
`/opt/nvfp4/validation/` and in the source repo.

## Known Issues & Tips

- **SM120 only.** The kernels are `sparse_mla_sm120`; nothing else is compiled.
- **util 0.90 fails with DSpark + cudagraphs** (KV squeeze after capture) — use
  0.96–0.985.
- **vLLM memory profile shows ~2.6 GiB more non-KV overhead vs fp8** (Triton
  JIT of the new store kernels + profiling-pass activation peak) — accounted
  for, not a leak.
- fp8 checkpoints/behavior are untouched: `--kv-cache-dtype fp8_ds_mla` on this
  image behaves exactly like the base image.

## Source & Credits

- Patch series + Dockerfile + validation: <https://github.com/danielwoz/vllm-dspark-nvfp4>
- Image: `danielwoz/vllm:dspark-nvfp4-cu132` (floating) / `:dspark-nvfp4-cu132-20260708`
- Built directly on tonyd2wild's DSpark stack and fraserpricee's image — the
  packed-KV IO protocol, spec-decode, and serving stack are theirs; this series
  adds the DSV4-native 4-bit page layout, the expand path, and the vLLM plumbing.
- vLLM and FlashInfer are Apache-2.0; full diffs vs the pristine base ship in
  the image under `/opt/nvfp4/patches/`.
