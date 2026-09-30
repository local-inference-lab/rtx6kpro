# Kimi K3 EXL3 and original MXFP4 qualification

This is the historical qualification record for the 2026-09-29 frozen composition. Its PR-state table records that date, not the maintainer action list. Use [vLLM issue #906](https://github.com/local-inference-lab/vllm/issues/906) for merge decisions and [X4T CC8 instructions](../x4t-dspark-cc8.md) for that separate serving profile.

Kimi-K3 already loads on `dev/karmic-kraken`. The PRs below reproduce our **EXL3 TP9 + DSpark** and **official MXFP4 TP16** serving profiles without importing the shared beta branches.

**EXL3 TP9 is qualified:** 100.00 median decode tok/s versus 100.38 for the preserved runtime, with byte-identical output text, equal token counts and equal acceptance counters. The full-codeword decoder in B12X #436 is necessary to preserve that speed: without it, the same workload measures 82.56 tok/s.

**Official MXFP4 TP16 is also qualified:** 572.1 aggregate output tok/s with 32 requests, versus 560.2 for the preserved runtime at the same GPU settings; no OOM or preemption. The fractional `p2375` checkpoint is **unsupported**: it needs a separate decoder, not a TP10 setting.

## Remaining PRs to review and merge

### B12X → `master`

| PR | What it provides | Needed for |
| --- | --- | --- |
| [#384](https://github.com/local-inference-lab/b12x/pull/384) | Retains prepared launch owners and coordinates collective initialization. Conflicts with master are resolved while preserving its cache reclamation. | Both profiles; shared preparation infrastructure. |
| [#436](https://github.com/local-inference-lab/b12x/pull/436) | Exact full-codeword lookup for eligible K2 Trellis decode. Restores the measured 17.8% decode regression. | EXL3 decode performance; not native MXFP4. |
| [#441](https://github.com/local-inference-lab/b12x/pull/441) | FP32 dense-MLA split partials through the prepared API. BF16 remains the default. | The explicit FP32-partial setting in both profiles. |
| [#442](https://github.com/local-inference-lab/b12x/pull/442) | Carries compiled Trellis route-pack programs into the materialized plan, preventing kernel resolution during frozen serving/graph execution. | EXL3 correctness and CUDA Graph readiness. |

### vLLM → `dev/karmic-kraken`

| PR | What it provides | Needed for |
| --- | --- | --- |
| [#841](https://github.com/local-inference-lab/vllm/pull/841) | Bounded InstantTensor staging and ownership of deferred checkpoint tensors. CPU staging is not CPU inference offload. | Checkpoint loading. |
| [#858](https://github.com/local-inference-lab/vllm/pull/858) | Draft-specific projection precision/sharding and separate target/draft DCP cache geometry. Preserves explicit Markov-head quantization. | DSpark and DFlash2 integration; not a requirement for no-spec MXFP4. |
| [#937](https://github.com/local-inference-lab/vllm/pull/937) | `--kda-prefill-backend b12x` with reusable state-pool bindings; selector for FP32 MLA partials. | Prepared B12X attention in both profiles. |
| [#938](https://github.com/local-inference-lab/vllm/pull/938) | Releases profiling plans as one batch, avoiding repeated global compiler-cache scans and garbage collection. | Startup time; does not change inference arithmetic. |
| [#846](https://github.com/local-inference-lab/vllm/pull/846) | Verified native-wheel reuse and atomic FlashInfer autotune-cache writes. | Image builds and cache recovery; not model arithmetic. |

All PRs target their canonical branch directly. They can be reviewed and merged in one batch, then built once. The combined source trees have been assembled without conflicts. No intermediate serving rebuild is required between merges.

## Already merged or not needed

- **Merged:** B12X #433 common Trellis API, #411 uneven TP/DCP collectives, and vLLM #909 common EXL3 loader, #842 dense MLA, #857 Kimi projection/precision support, #924/#925 runtime memory and scheduling options.
- **Superseded:** B12X #415 is closed; vLLM #859 is closed in favor of merged #909 and B12X #433. Do not merge the EXL3-specific API alternative.
- **Not required:** B12X [#437](https://github.com/local-inference-lab/b12x/pull/437) stable route packing. Prefill matches the control without it; it remains an independent research-only option.
- B12X #292 uses the legacy MLA API and also changes split policy. #441 carries only partial precision into the prepared API; #292 is not an additional dependency.

This is not a TP16-only patch set. The measured EXL3 profile runs on TP9/DCP9; preparation, loader and precision interfaces are not hard-coded to either topology. Untested TP sizes are not claimed as qualified here.

## Measured EXL3 result

The checkpoint is the independently encoded uniform two-bit model at `/data/trellis-quant/b2-reproduction-20260925T113022Z/checkpoint`, not the lossless container repack of Luke's published QSRT model. Dense tensors retain their serialized precision.

Conditions: nine RTX PRO 6000 Blackwell 96 GB GPUs, identical devices/clocks for all arms; TP9/DCP9, Red Hat DSpark K5, BF16 activation arithmetic, FP8 target KV, FP32 MLA partials, B12X KDA prefill, scheduler chunk 4096, one request. Coding uses a fixed 183-token input, 4096 output tokens, temperature zero, seed 1, three measured runs.

| Runtime | Decode tok/s | Accepted proposals | Prefill 8k / 32k / 64k tok/s |
| --- | ---: | ---: | ---: |
| Preserved serving implementation | 100.380 | 42.628% | 2610 / 2622 / 2513 |
| Canonical composition without #436 | 82.556 | 42.628% | 2622 / 2632 / 2536 |
| Canonical composition with #436 | 100.004 | 42.628% | 2626 / 2635 / 2534 |

All nine complete outputs are identical; every run accepts 2790 of 6545 proposed tokens over 1309 target cycles. No output/acceptance change explains the throughput recovery. This is a runtime parity check, not a claim that two-bit quantization is lossless against official MXFP4 or that all tasks have identical quality.

## Official MXFP4 profile

Implemented and qualified for loading and serving: TP16/DCP4, native MXFP4 experts, BF16 dense projections and BF16 KV, FP32 MLA partials, B12X KDA prefill, no vision/draft, 8192-token request limit, scheduler chunk 1024, maximum 32 requests, 1.4 GB KV budget per GPU.

The physical cache holds **119,239 tokens**. Two measured batches of 32 requests, each with a 248-token prompt and 1024 generated tokens, complete without OOM or preemption at **571.72 and 572.53 aggregate output tok/s** (including prefill). Decode-window rates are 573.73 and 574.54 tok/s. Separate short chat responses are coherent in both runtimes but not byte-identical. This MXFP4 check qualifies loading, execution and throughput, not full-model numerical parity or task quality; no MXFP4 KLD comparison was run.

The preserved runtime measures 561.20 and 559.25 aggregate output tok/s under the same conditions. Means are **572.13 versus 560.23 tok/s**, an observed +2.1%; no throughput regression is found in this workload. The short CC1 decode-window checks are 48.77 versus 46.44 tok/s, one run each, not a statistical single-request benchmark.

Historical 612–614 tok/s used different GPU memory offsets and is not a matched software-regression comparison: the preserved runtime also runs below those figures here. GPUs 0–11 use 13,365 MHz memory and GPUs 12–15 use 16,365 MHz, with 450 W limits. This workload does not qualify 32 simultaneous 8192-token sequences; the physical cache is smaller than that total.

## Fractional p2375 on TP10

**Unsupported by the fetched canonical runtime.** `/data/trellis-quant/fractional-kld-20260923/checkpoints/p2375/fractional.json` declares `fractional-k3-v1` and the within-tile pattern `periodic-2-2-3-2-2-3-2-3`. The manifest itself says it is unsupported by EXL3/B12X.

It needs a matching GPU decoder, weight-preparation/checkpoint adapter, and numerical/graph/serving qualification. A metadata-only check inside the assembled image accepts the uniform two-bit EXL3 manifest but rejects the fractional directory because it has no EXL3 manifest or Hugging Face config (`evidence/checkpoint-format-probe.json`). Renaming the format or passing `--tp 10` cannot implement the missing codec. No weights were silently requantized, no CPU inference fallback was substituted, and TP10 serving is not claimed.

<details>
<summary>Source-locked image, reproduction and evidence</summary>

Canonical bases checked on 2026-09-29:

- vLLM: `502d6cb5acd2ba2a62ecf58497be558c9d86089f`.
- B12X: `b4b12bcf200a9979f40a3a9b7e660c6d83b91a73`.

The isolated [vLLM composition branch](https://github.com/local-inference-lab/vllm/tree/integration/kimi-review-composition-20260929) and [B12X composition branch](https://github.com/local-inference-lab/b12x/tree/integration/kimi-review-composition-20260929) contain only the table's review units plus release-description fragments. They are reproducibility branches, not extra PRs to merge wholesale.

- vLLM composition: `98a084492a19654530178dc79e70f00bf2aec1c9`.
- B12X composition: `dbf5b4072c446b2e01dfffc22fb5c5f47cc72524`.
- Local image: `local/kimi-k3:kk-review-98a084492a-dbf5b407`.
- Image ID: `sha256:f5c701d27818c33b5dd413073d343c32ea6e710c7a0b22819886031d5ceb1aa6`. This tag has not been published to Docker Hub.
- Runtime: CUDA 13.4, NVIDIA PyTorch 26.08 / Torch `2.14.0a0+4fdf77b940.nv26.08`, CUTLASS DSL 4.6.2. No Python serving overlays.

Local operator workspace: `/root/vllm/kimi/runtime-qualification-20260929/`.

```bash
cd /root/vllm/kimi/runtime-qualification-20260929
KIMI_IMAGE=local/kimi-k3:kk-review-98a084492a-dbf5b407 \
  KIMI_CONTAINER=kimi-k3-exl3-reviewed \
  bash start-exl3-tp9.sh candidate
# OR, after stopping the TP9 container and confirming all GPUs are free:
KIMI_IMAGE=local/kimi-k3:kk-review-98a084492a-dbf5b407 \
  KIMI_CONTAINER=kimi-k3-mxfp4-reviewed \
  bash start-mxfp4-tp16.sh
```

Launch scripts fail if their container name or port is already occupied. Use `KIMI_CONTAINER` for a distinct container name. The APIs bind to `0.0.0.0:8012` on the trusted local network.

`evidence/review-source-equivalence.json` proves that the review assembly differs from the measured TP9 source only in release-description JSON files. Runtime Python trees, native sources, tests and build scripts match exactly. Native wheel reuse separately verifies source/dependency/ABI identity and binary payload hashes. The preserved control and test images remain available.

The evidence index is `evidence/qualification.json`; its hashes cover checkpoint metadata, run outputs, logs, launch scripts and source-equivalence receipts. Raw paired runs: `evidence/reference/`, `evidence/candidate/`, `evidence/candidate-lut/`. Full MXFP4 receipts: `evidence/mxfp4/` and `evidence/mxfp4-reference/`. Component evidence includes real-weight compact/full LUT replay, exhaustive codeword equality, 24 EXL3 extent/graph cases, KDA equivalence, MLA high-page/graph coverage, five partial-precision cases, 53 warmup release cases, and the configuration/cache suites named in the PRs.

</details>
