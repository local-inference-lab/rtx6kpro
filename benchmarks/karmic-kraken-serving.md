# Karmic Kraken model-serving measurements

Status: **qualified** six-mode serving measurements under the conditions below.
The separate [TP4 memory-control comparison](tp4-memory-controls.md) records
isolated capacity/throughput trade-offs. No memory-saving control is promoted
to a shared default by this report.

## Conditions

RTX PRO 6000 Blackwell **Max-Q Workstation Edition**, **VRAM +6000**, automatic
graphics clocks and 325 W power limit. One server at a time on the same physical
GPU set: TP1 uses GPU5, TP2 uses GPU5/6, TP4 uses GPU5/6/7/8. These are not the
stock-clock Workstation measurements in the
[versioned guide archive](../archive/serving-guides/README.md).

Each cell is the median of **five warmed 30-second windows**. Decode uses
context zero and temperature 1. Prefill uses unique uncached nominal-32K
requests and client time to first token, which includes speculative
prompt-finalization work. C8/C4 rates are aggregate across requests. Output
depends on draft acceptance as well as execution cost; verifier steps/s are
reported separately and are not interchangeable with output tok/s.

Saved R9/R35/R38 measurements are read from their artifacts; those servers
were not restarted for the source-composition repeats. Checkpoint metadata
hashes and shard identities match within each pair. Shard payloads were not
rehash-read during serving. The
[machine-readable evidence](karmic-kraken-serving-samples.json) contains all
samples, configurations, fingerprints, images and benchmark commands.

| Model / mode | TP | Batch budget | Request slots | Context cap | Target sampling |
|---|---:|---:|---:|---:|---|
| DeepSeek V4 text, DSpark K5 | 2 | 4096 | 8 | 1,048,576 | Temperature 1, top-p 1 |
| DeepSeek V4 Vision, DSpark K3 | 2 | 4096 | 4 | 1,048,576 | Temperature 1, top-p 1 |
| Qwen3.8 Flash Next, MTP3 | 1 | 6019 | 16 | 262,144 | Temperature 1, top-p .95, top-k 20 |
| DeepSeek V4.1, adaptive DSpark K7 | 4 | 4096 | 32 | 131,072 | Temperature 1, top-p .95 |
| GLM-5.3 Flash, MTP3 | 4 | 4096 | 32 | 1,048,576 | Temperature 1, top-p .95 |
| GLM-5.3 Flash, DFlash2 K7 | 4 | 4096 | 32 | 1,048,576 | Temperature 1, top-p .95 |

All modes use DCP1 and GPU-only prefix storage. Qwen uses CPU PLE and a fixed
eight-GiB KV budget. DeepSeek V4.1 uses RAM Engram; its disk-table default has
separate functional checks, not these throughput measurements.

## Output and prefill

| Model / mode | Saved JJ C1 → KK, tok/s | Saved JJ concurrent → KK, tok/s | Saved JJ 32K prefill → KK, tok/s |
|---|---:|---:|---:|
| DeepSeek V4 text, DSpark K5 | 214.9 → 212.5 (−1.09%) | C8: 611.9 → 667.5 (+9.08%) | 11,167 → 11,392 (+2.01%) |
| DeepSeek V4 Vision, DSpark K3 | 169.2 → 193.9 (+14.60%) | C4: 394.7 → 401.4 (+1.68%) | 8,993 → 9,182 (+2.10%) |
| Qwen3.8 Flash Next, MTP3 | 157.6 → 173.2 (+9.91%) | C8: 674.2 → 698.4 (+3.58%) | 12,104 → 12,073 (−0.26%) |
| DeepSeek V4.1, adaptive DSpark K7 | 230.6 → 249.1 (+8.01%) | C8: 748.3 → 805.5 (+7.65%) | 17,503 → 17,982 (+2.74%) |
| GLM-5.3 Flash, MTP3 | 248.9 → 281.3 (+12.99%) | C8: 873.8 → 897.6 (+2.72%) | 13,514 → 13,816 (+2.23%) |
| GLM-5.3 Flash, DFlash2 K7 | 219.7 → 229.0 (+4.22%) | C8: 677.7 → 702.6 (+3.67%) | 13,840 → 13,955 (+0.83%) |

All six arms pass factual and repeated/changed-prefix checks. Every reported
window passes benchmark validity checks; no failed/looping decode or cached
prefill cells are discarded. Vision profiles pass their image checks. These
checks do not establish general checkpoint quality or arbitrary-context capacity.

## Verifier execution and acceptance

| Model / mode | Saved JJ C1 → KK, steps/s | Saved JJ concurrent → KK, steps/s | C1 accepted-length change |
|---|---:|---:|---:|
| DeepSeek V4 text, DSpark K5 | 73.24 → 75.45 (+3.02%) | C8: 231.69 → 249.68 (+7.77%) | −4.39% |
| DeepSeek V4 Vision, DSpark K3 | 79.29 → 87.18 (+9.96%) | C4: 185.65 → 190.53 (+2.63%) | +4.44% |
| Qwen3.8 Flash Next, MTP3 | 76.06 → 81.78 (+7.52%) | C8: 329.98 → 333.65 (+1.11%) | +1.88% |
| DeepSeek V4.1, adaptive DSpark K7 | 89.87 → 98.58 (+9.69%) | C8: 326.23 → 361.87 (+10.92%) | −0.32% |
| GLM-5.3 Flash, MTP3 | 99.39 → 112.14 (+12.84%) | C8: 351.88 → 364.76 (+3.66%) | +0.21% |
| GLM-5.3 Flash, DFlash2 K7 | 84.28 → 87.56 (+3.90%) | C8: 259.10 → 275.92 (+6.49%) | +0.33% |

DeepSeek V4 text retains a 1.09% negative C1 output delta despite faster
verifier execution. Accepted length is lower; an independent saved R9 repeat
measured 195.15 tok/s with 73.22 steps/s. Neither output sample is discarded.
Automatic MoE selection matches the explicitly pinned 128-CTA diagnostic
without a grid override and preserves prefill/C8 throughput. See the
[MoE tuning evidence](https://github.com/local-inference-lab/b12x/tree/d606d254998b33649fca41f6d8ac7fef971deb21/validation/serving/moe_overlap_tuning).

Qwen's prefill deficit against R35 is 0.26%. The source-composition startup's
81.78 C1 verifier steps/s was below a separate repaired-source arm's 84.06.
A **five-window cached restart of the identical image** measures **178.55 C1
tok/s / 83.96 steps/s** and **713.67 C8 tok/s / 339.13 steps/s**. Its C1
execution rate is within 0.12% of the repaired-source arm; a persistent 3%
regression is not reproduced. The first and repeated series are both retained
in [the restart evidence](qwen-cached-restart-samples.json), without replacing
the table with its best startup. The reason for the startup-dependent difference
is not established.

The four-iteration C1 profile confirms target and MTP kernel execution after
the unprofiled repeat. It contains 192 target dynamic-MoE launches, 12 MTP
W4A16 MoE launches and 12 CUDA graph launches. Kernel durations overlap and
profiling perturbs execution; they are not another throughput sample.

### GLM startup variability

A separate empty/populated compilation-cache startup pair retains five warmed
C1 windows per arm. MTP3 measures 277.03 then 265.69 tok/s (110.90 then
105.74 steps/s); DFlash2 measures 225.21 then 212.85 tok/s (87.21 then
85.05 steps/s). The latter output median is below the saved JJ 219.7 tok/s,
although verifier execution remains above JJ's 84.28 steps/s. The table above
is not replaced with a preferred startup, and these same-image pairs are not
a source-revision regression test.

[Startup samples, four-rank profiles and interpretation](glm-startup-consistency.md)
retain the slower results. Matching kernel selections and launch inventories
do not establish the cause of the execution difference. Neither clearing
compilation caches nor changing a hardware-queue limit is recommended solely
from these observations.

## KV capacity

These are shared-pool estimates at the configured context and speculation
settings, not a per-request context limit.

| Model / mode | Saved JJ logical tokens | KK logical tokens |
|---|---:|---:|
| DeepSeek V4 text, DSpark K5 | 1,192,702 | 1,301,500 |
| DeepSeek V4 Vision, DSpark K3 | 1,215,644 | 1,291,085 |
| Qwen3.8 Flash Next, MTP3, eight-GiB allocation | 517,581 | 434,258 |
| DeepSeek V4.1, adaptive DSpark K7 | 4,569,816 | 4,727,748 |
| GLM-5.3 Flash, MTP3 | 3,686,928 | 5,595,903 |
| GLM-5.3 Flash, DFlash2 K7 | 3,706,857 | 5,816,930 |

Qwen's physical eight-GiB pool is unchanged. Its capacity calculation now
reserves recurrent endpoints and a pinned restore source. The
[integer accounting](qwen-boundary-capacity-accounting.md) reconciles all
three recorded estimates exactly; omitting those reserves overstates capacity.

## Prefix cache checks

The [cache-recovery record](../archive/serving-guides/karmic-kraken-beta-20260919-cfc67a15ebc3daf7/benchmarks/karmic-kraken-serving.md#prefix-cache-checks)
preserves exact source/image boundaries for text RAM/disk recovery in Qwen,
GLM and DeepSeek, and image-aware recovery in DeepSeek V4 Vision/V4.1. GLM
and Qwen external recurrent checkpoints remain text-only. Their uncached
vision paths are available; image-bearing requests are recomputed.

### Spark TP2

The published image's Spark preset passes arithmetic, repeated/changed text
prefixes and image recognition on Max-Q GPUs 6/7. It uses TP2/DCP2, MTP3,
a 3072-token budget, four request slots, 3996 MiB KV per GPU and CPU-only
LMCache with 16 GiB RAM/64 GiB disk. It reports 899,435 logical KV tokens.

| Cache check | GPU hit tokens | External hit tokens | Loaded disk objects | Request time |
|---|---:|---:|---:|---:|
| Cold text request | 0 | 0 | 0 | 1.778 s |
| Native repeated prefix | 16,287 | 0 | 0 | 0.085 s |
| RAM restore | 0 | 16,287 | 0 | 0.106 s |
| RAM restore, changed suffix | 0 | 16,281 | 0 | 0.171 s |
| Disk restore after both processes restart, changed suffix | 0 | 16,287 | 28 | 0.165 s |

Each stage returns the expected factual answer at temperature 1. These are
single-request correctness checks, not throughput medians. The disk request
can reuse the suffix checkpoint retained by the RAM test. The
[public-image receipt](karmic-kraken-public-image.json) records identities,
native arguments and counter evidence. Cache metrics are reached through an
SSH tunnel because that service intentionally binds only to loopback.

The [GLM Spark TP2 guide](../models/glm-5.3-flash-spark-tp2.md) retains its
separate stock Workstation measurement: TP2/DCP2, 3072-token budget, C1
186.5 tok/s, C4 403.1 tok/s and 32K prefill 10,919 tok/s. That is one warmed
window per decode concurrency, not a row from this five-repeat Max-Q matrix.

## Source boundaries

Published image:
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b`,
digest `sha256:55e477ad62ae15a77c9b869e8fb8e2d958f6edcc4d95e306adfa89dee9ed19df`.
The [Actions release](https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/karmic-kraken-beta-443d9f815c57d23bfc3a5e780409fb26b0878d1c8b0a28c4c2818d6414af5454)
contains the component locks and packaging checks. Its 68 filesystem layers
contain vLLM `57a80980bbf4b40398de7ed851b23e55a3a4c50e` and B12X
`e9ce547767ff9ee6509faf294fa1b4e2380dfbf5` from the two integration branches.

Installed-source comparison against the timed composition below finds all
481 B12X Python files identical. Among 2,749 vLLM Python files, only generated
version metadata and the reviewed MoE docstring differ; executable Python ASTs
otherwise match. Torch, Triton, FlashInfer, LMCache, InstantTensor and ModelOpt
package versions match. The published image passes the Spark serving/cache
checks above; the six-mode throughput table was measured on the equivalent
source-composition image, not repeated on the registry digest.

The serving-composition image is
`sha256:d351927666613339608da3ec6c6f227048a60cb7c2a46baff2edf8a4b1dbb84c`.
It contains vLLM `cfdecfaa4739ee3141c9745997cd8a83f31745a4` and B12X
`f6d87176c1f5fe378d175f500ff652298c5af7ef`, installed in the shared runtime
without serving-source bind mounts or manual MoE grid pins. Its base is
`ghcr.io/local-inference-lab/vllm@sha256:5927520c447fdcbc0990567f9237756ff66a54e360438915a5f70cd5cf9d530d`,
CUDA 13.4.1/PyTorch 2.14. Native dependency sources are unchanged according
to the assembly receipt. This local image ID is not a registry digest.

Saved registry references:

- DeepSeek V4 text/Vision R9:
  `voipmonitor/vllm@sha256:5bea088597980b299a1df8a6f3fc6d2d22c723276088ea8583b456f27043c0cd`.
- Qwen/GLM R35:
  `localinferencelab/vllm@sha256:7a425c6864b951bbc368111490a4b0ac69d8cd0dd4987075b2c7d40b753b1bf5`.
- DeepSeek V4.1 R38:
  `localinferencelab/vllm@sha256:f41ca8bb10bb3a125a50340d70d39ad4b7f5605f3fcb661bc992ed0bc4701a00`.

The JSON evidence records model revisions, draft fingerprints, launch argv,
process environment, library hashes and benchmark script hashes. The
[integration checklist](https://github.com/local-inference-lab/vllm/issues/808)
identifies attributed source PRs. Floating deployment tags do not change
the identities of measurements recorded here.
