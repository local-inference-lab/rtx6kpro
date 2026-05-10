# MiMo-V2.5-Pro on RTX PRO 6000 Blackwell

This directory records the TP=8 MiMo-V2.5-Pro NVFP4/MXFP8 bring-up on 8x RTX PRO 6000 Blackwell PCIe with SGLang and B12X.

The validated artifact is a mixed ModelOpt checkpoint:

- routed expert MLPs quantized to NVFP4
- fused QKV attention tensors converted from the upstream TP-interleaved FP8 layout to MXFP8
- BF16 MTP draft weights provided through a small serving overlay
- serving target is SGLang + B12X with TP=8, MTP/EAGLE, B12X attention, B12X MoE, FP8 KV cache, and CUDA graph target verify

## Current artifacts

| Artifact | Value |
|---|---|
| Upstream source | `XiaomiMiMo/MiMo-V2.5-Pro` |
| Public checkpoint | `festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8` |
| Runtime image | `voipmonitor/sglang:mimo-v25-pro-tp8-microrecip-20260510` |
| Runtime digest | `sha256:b634da4afa79fb4bb8bb06c67938c8f3a816af96bb31d307605c2a56f35fa444` |
| Base image | `docker.io/lukealonso/sglang-cuda13-b12x:w4a16` |
| Target TP | `8` |
| TP=16 status | not validated |

## Documents

| Page | Scope |
|---|---|
| [Quantization pipeline](quantization.md) | Exact BF16 dequantization, BF16 smoke, NVFP4 quantization, MTP fixup, MXFP8 QKV conversion, and final mixed checkpoint assembly. |
| [SGLang/B12X patches](sglang-b12x-patches.md) | Runtime bugs found, patch files used, root-cause evidence, and why the final micro MoE reciprocal-scale path was needed. |
| [Running the model](running.md) | DockerHub image, exact `docker run` command, health/log/smoke commands, and current runtime caveats. |

## Validated runtime result

The production-style SGLang instance on port `30004` was run with MTP/EAGLE enabled and `--cuda-graph-max-bs 8`.

Smoke test:

- prompt required the exact sentence `The capital of France is Paris.`
- output included the exact required sentence
- output then produced a coherent checkpoint-integrity paragraph
- no CJK or replacement-token corruption was detected

Concurrent soak:

| Run | Result |
|---|---|
| first 8-way soak | 8/8 HTTP 200, 2,985 completion tokens, 83 s |
| warm 8-way soak | 8/8 HTTP 200, 4,042 completion tokens, 31 s |

The warm soak had no CJK characters, Unicode replacement characters, or multi-character non-ASCII corruption runs. One local checker warning was only a normal ASCII separator line in a generated incident-report heading.

## Important caveats

- TP=8 is the validated target. TP=16 may be possible at the SGLang partitioning layer, but the source MiMo Pro fused QKV tensors were TP-packed for TP=8 and TP=16 has not been proven.
- MiMo's tokenizer defaults to thinking mode. For normal chat smoke tests, pass `chat_template_kwargs: {"enable_thinking": false}`.
- The current runtime image preserves the compiled BF16 linear path after warmup with `SGLANG_DISABLE_AUTOTUNED_LINEAR_AFTER_WARMUP=0`. This was needed for graph/eager consistency during MTP target-verify debugging, but it can cause Torch Inductor autotune for new large prefill shapes. Treat that as a known runtime performance issue, not a checkpoint-quality issue.
- Earlier no-graph and no-MTP runs were useful gates, but the current validated image is the micro-MoE reciprocal-scale runtime described in this directory.
