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
| Runtime image | `voipmonitor/sglang:mimo-v25-pro-tp8-b12x3917cb2-20260510` |
| Runtime digest | `sha256:96aa5a10913cae3af6fe145e5c21238549971da271fc06b522ec4a6a9bd51c80` |
| Base image | `docker.io/lukealonso/sglang-cuda13-b12x:w4a16` |
| B12X source | `lukealonso/b12x@3917cb2fe5a2118eaab8b68f7710c71aad9e4b1c` |
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

Autotune-fix validation:

| Run | Result |
|---|---|
| short coherence prompt | `Paris` and `4`, HTTP 200, no request-time `AUTOTUNE mm` log lines |
| long context run B | 50,712 prompt tokens, returned `VEGA-8431` and `Paris`, HTTP 200, no request-time `AUTOTUNE mm` log lines |
| B12X 3917 A16 smoke | `Paris, 4`, HTTP 200 |
| B12X 3917 long context | returned `VEGA-8431, Paris` and `ORION-9265, Paris`, HTTP 200, no request-time `AUTOTUNE mm` log lines |

The warm soak had no CJK characters, Unicode replacement characters, or multi-character non-ASCII corruption runs. One local checker warning was only a normal ASCII separator line in a generated incident-report heading.

## Important caveats

- TP=8 is the validated target. TP=16 may be possible at the SGLang partitioning layer, but the source MiMo Pro fused QKV tensors were TP-packed for TP=8 and TP=16 has not been proven.
- MiMo's tokenizer defaults to thinking mode. For normal chat smoke tests, pass `chat_template_kwargs: {"enable_thinking": false}`.
- Startup CUDA graph capture still runs Torch Inductor autotune for small BF16 dense-linear shapes. The current image prevents new unquantized BF16 linear shapes from being compiled during live request handling, so request-time `AUTOTUNE mm(...)` log lines should not appear for normal short or long-context inference.
- Earlier no-graph and no-MTP runs were useful gates. The current recommended runtime uses the upstream B12X 3917 image with `B12X_MOE_FORCE_A16=1`; the older micro-MoE reciprocal-scale image remains documented as the previous validation point.
