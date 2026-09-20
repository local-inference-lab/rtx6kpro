# TP4 runtime memory-control measurements

Status: qualified bounded TP4 measurements. No shared default was changed.

[All samples and effective launch arguments](tp4-memory-controls-samples.json).
The [five-window confirmations and four-rank profile summaries](tp4-memory-confirmation-samples.json)
retain every repeat, including reversals of the initial result.
The timed image and its relation to the published beta are identified in the
[serving source boundaries](karmic-kraken-serving.md#source-boundaries).
This comparison varies runtime storage settings on that same immutable image;
it does not compare CUDA releases or checkpoint quantization.

For complete default Compose files, including image/profile ENV and resolved
vLLM arguments, use the [expandable deployment reference](../docs/unified-vllm-docker.md#expand-the-complete-default-configurations).
Those are the user-facing defaults; the measurement JSON above retains the
TP4 and memory-control overrides specific to these experiments.

Four RTX PRO 6000 Blackwell Max-Q GPUs, VRAM +6000, automatic graphics
clocks, 325 W, TP4/DCP1. Each value is the median of at least three
warmed 30-second windows. Prefill is uncached 32K client TTFT; decode
uses context zero and temperature 1. C8 is aggregate; Vision uses C4.

Both sides start with populated compilation and tuning caches. These
controls are not the TP1 Qwen or TP2 DeepSeek rows in the model guide.

## Shared allocation fixes versus runtime settings

The image already applies allocation-lifetime fixes wherever the corresponding
operation is used: release attention preparation probes and stale owners
([vLLM #801](https://github.com/local-inference-lab/vllm/pull/801),
[#803](https://github.com/local-inference-lab/vllm/pull/803)), bound NVFP4
loader temporaries ([#805](https://github.com/local-inference-lab/vllm/pull/805)),
reuse caller-owned GLM DCP scratch
([#806](https://github.com/local-inference-lab/vllm/pull/806)), and account for
live graph/restore storage correctly
([#807](https://github.com/local-inference-lab/vllm/pull/807)). These are
operation-specific corrections, not Spark-only deployment switches.

The shared PyTorch schema-enumeration patch removes redundant Python work;
it does not free a GPU buffer. NGC PyTorch 2.14 already has direct mutation
tracking, so it does not receive the separate 2.13 mutation-metadata backport.
The [dependency contract](https://github.com/local-inference-lab/blackwell-llm-docker/blob/f0c4f9fe04267d7a83a896f1c81d390b1ece5fee/runtime/DEPENDENCIES.md)
identifies the build-owned patches and checks.

The controls tested below are different: allocator mapping granularity,
communication storage/parallelism, and cuBLAS workspace limits. They can alter
runtime behavior even though weights, activation precision and KV page
geometry remain unchanged. No combined memory preset is qualified here.

## Matched controls

Controls use 20-MiB expandable allocator segments, 16 NCCL channels
and 2-MiB NCCL buffers. Model, speculation, capture and request budgets
are unchanged within each comparison.

| Model / mode | 32K prefill, tok/s | C1 output / steps per second | Concurrent output / steps per second | Logical KV tokens |
|---|---:|---:|---:|---:|
| GLM-5.3 Flash / MTP3 | 13,725 | 265.62 / 106.61 | C8: 900.22 / 361.66 | 5,598,766 |
| GLM-5.3 Flash / DFlash2 K7 | 13,903 | 219.33 / 84.26 | C8: 717.89 / 271.70 | 5,822,262 |
| DeepSeek V4.1 / DSpark K7 | 18,004 | 253.82 / 98.99 | C8: 800.53 / 366.00 | 4,779,985 |
| DeepSeek V4 text / DSpark K5 | 14,257 | 254.77 / 88.60 | C8: 887.57 / 331.58 | 7,395,064 |
| DeepSeek V4 Vision / DSpark K3 | 10,357 | 238.66 / 107.23 | C4: 540.18 / 249.87 | 7,394,923 |
| Qwen3.8 Flash Next / MTP3 | 16,386 | 188.22 / 89.22 | C8: 921.44 / 441.81 | 959,805 |

## Isolated changes

Percentages are changes against the matched control, not speedup claims.
Output and execution rates are separate because draft acceptance varies.
The linked JSON retains every sample, range and accepted-length value.

| Model / mode | Control changed | Logged extra KV MiB¹ | KV tokens % | Prefill % | C1 output / steps % | Concurrent output / steps % |
|---|---|---:|---:|---:|---:|---:|
| GLM-5.3 Flash / MTP3 | `allocator12` | +430 | +1.12 | +0.02 | +0.65 / -0.82 | -2.10 / -0.63 |
| GLM-5.3 Flash / MTP3 | `nccl-buffer1m` | +0 | +0.00 | +0.14 | +1.80 / -0.41 | +0.35 / -0.21 |
| GLM-5.3 Flash / MTP3 | `nccl-channels2` | +215 | +0.55 | +0.12 | +0.39 / -1.19 | -2.17 / -1.70 |
| GLM-5.3 Flash / MTP3 | `cublas4m` | +205 | +0.53 | +0.90 | -1.28 / -0.00 | -0.48 / +0.08 |
| GLM-5.3 Flash / DFlash2 K7 | `allocator12` | +410 | +0.98 | +0.26 | -0.43 / +1.03 | -1.85 / +0.67 |
| GLM-5.3 Flash / DFlash2 K7 | `nccl-buffer1m` | +0 | +0.00 | +0.29 | -0.17 / +1.00 | -3.73 / +0.24 |
| GLM-5.3 Flash / DFlash2 K7 | `nccl-channels2` | +225 | +0.55 | +0.37 | -1.56 / +1.41 | -5.11 / -2.15 |
| GLM-5.3 Flash / DFlash2 K7 | `cublas4m` | +246 | +0.61 | +0.03 | +0.05 / +1.48 | -1.88 / +0.35 |
| DeepSeek V4.1 / DSpark K7 | `allocator12` | +164 | +1.39 | -0.51 | -3.93 / -1.45 | +0.81 / -0.58 |
| DeepSeek V4.1 / DSpark K7 | `nccl-buffer1m` | -20 | -0.17 | -0.39 | -0.36 / -2.94 | +1.30 / -2.83 |
| DeepSeek V4.1 / DSpark K7 | `nccl-channels2` | +225 | +1.88 | -0.25 | -1.18 / +1.13 | +1.04 / -0.37 |
| DeepSeek V4.1 / DSpark K7 | `cublas4m` | +246 | +2.01 | +0.06 | +1.72 / +0.37 | +1.26 / -2.14 |
| DeepSeek V4 text / DSpark K5 | `allocator12` | -10 | -0.03 | +0.33 | -1.64 / +3.81 | +1.20 / +1.55 |
| DeepSeek V4 text / DSpark K5 | `nccl-buffer1m` | +0 | +0.00 | -0.11 | +0.47 / +4.04 | +2.08 / +1.75 |
| DeepSeek V4 text / DSpark K5 | `nccl-channels2` | +225 | +0.45 | +0.23 | +2.55 / +3.71 | +0.62 / +0.65 |
| DeepSeek V4 text / DSpark K5 | `cublas4m` | +215 | +0.43 | -0.04 | +1.97 / +2.64 | +4.15 / +1.76 |
| DeepSeek V4 Vision / DSpark K3 | `allocator12` | +41 | +0.08 | -0.01 | -2.93 / -2.02 | +1.83 / -1.44 |
| DeepSeek V4 Vision / DSpark K3 | `nccl-buffer1m` | +0 | +0.00 | -0.14 | -3.20 / -0.85 | -1.27 / -0.97 |
| DeepSeek V4 Vision / DSpark K3 | `nccl-channels2` | +225 | +0.45 | -0.15 | +6.12 / +0.71 | -0.96 / -0.48 |
| DeepSeek V4 Vision / DSpark K3 | `cublas4m` | +246 | +0.48 | +0.14 | +1.20 / +1.09 | +1.90 / +0.78 |
| Qwen3.8 Flash Next / MTP3 | `allocator12` | fixed budget | +0.00 | -0.05 | -0.83 / -4.99 | -3.09 / -3.08 |
| Qwen3.8 Flash Next / MTP3 | `nccl-buffer1m` | fixed budget | +0.00 | +0.00 | -5.89 / -2.82 | -1.98 / -1.65 |
| Qwen3.8 Flash Next / MTP3 | `nccl-channels2` | fixed budget | +0.00 | +0.23 | +4.58 / +3.98 | +0.90 / +0.32 |
| Qwen3.8 Flash Next / MTP3 | `cublas4m` | fixed budget | +0.00 | +0.10 | +0.00 / -0.46 | -0.51 / -0.24 |

¹ Difference in available KV memory printed by the server for rank 0,
rounded to 0.01 GiB. Automatic admission spends savings on KV storage.
This is not an allocation census for all four GPUs or a measurement of
a specific allocator's reserved bytes. The JSON also retains separate
coarse device-memory samples for each of the four GPUs.
Qwen retains its explicit eight-GiB allocation even if workspace use falls.

### Environment variables

- `allocator12`: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:12`.
- `nccl-buffer1m`: `NCCL_BUFFSIZE=1048576`.
- `nccl-channels2`: `NCCL_MIN_NCHANNELS=2`; `NCCL_MAX_NCHANNELS=2`.
- `cublas4m`: `CUBLAS_WORKSPACE_CONFIG=:4096:1`.

## DFlash2 confirmation and collective profile

The two NCCL changes were repeated with **five** warmed C1/C8 windows and
three uncached 32K requests per arm. The control was restarted separately
from the three-window screen. All windows and both series are retained.

| Setting | C1 output, tok/s | C1 steps/s | C8 output, tok/s | C8 steps/s | 32K prefill, tok/s | Logical KV tokens |
|---|---:|---:|---:|---:|---:|---:|
| 16 channels, 2-MiB buffers | 216.84 | 84.38 | 702.40 | 271.06 | 13,922 | 5,822,262 |
| 16 channels, 1-MiB buffers | 223.46 | 85.57 | 704.28 | 273.24 | 13,918 | 5,822,262 |
| 2 channels, 2-MiB buffers | 223.38 | 85.35 | 684.00 | 265.17 | 13,906 | 5,854,253 |

The one-MiB buffer's initial C8 output deficit is not reproduced: the repeat
is +0.27% output and +0.81% execution, with unchanged KV capacity. It therefore
does not supply a measured capacity reason to change this configuration.

The two-channel C8 deficit **is reproduced**: −2.62% output and −2.17%
execution for +0.55% logical KV capacity. The initial execution deficit was
−2.15%. This is a capacity/throughput trade-off, not a universal improvement.

A separate four-iteration C8 trace contains 408 BF16 NCCL ring all-reduces
per rank in each arm. Rank 0's median duration grows from **48.06 to 54.72 μs**
when the launch grid falls from 16 to 2 CTAs; the collective count is unchanged.
The sum of NCCL kernel durations rises on all four ranks. Those durations
include rank waiting and can overlap other work, so their sum is not the
server's elapsed time. The unprofiled windows above establish the throughput
deficit; the profile identifies communication as an affected path.

## DeepSeek V4.1 allocator confirmation

Five warmed C1/C8 windows and three 32K requests repeat the allocator-only
comparison. DSpark retains adaptive draft depth in both arms.

| Allocator segment | C1 output, tok/s | C1 steps/s | C8 output, tok/s | C8 steps/s | 32K prefill, tok/s | Logical KV tokens |
|---|---:|---:|---:|---:|---:|---:|
| 20 MiB | 259.86 | 99.41 | 807.38 | 359.70 | 17,954 | 4,771,962 |
| 12 MiB | 250.84 | 96.75 | 800.96 | 354.80 | 18,018 | 4,846,611 |

The smaller segment increases logical KV capacity by 1.56% and logged
rank-zero available KV memory by approximately 184 MiB. C1 output falls
3.47%, with 2.67% fewer verifier steps/s and 0.58% lower accepted length.
C8 output falls 0.80%; prefill rises 0.36%. The negative C1 result from
the three-window screen therefore persists in the repeat. Adaptive DSpark
can change work per verifier step, so the step rate alone is not a kernel
latency comparison. The output result is sufficient not to promote this
allocator setting as a throughput-neutral default for DeepSeek V4.1.

## Qwen TP4 confirmations

Five warmed C1/C8 windows and three 32K requests repeat the allocator and
channel-count comparisons separately on TP4. The fixed eight-GiB KV pool
remains 959,805 logical tokens.

| Allocator segment / NCCL channels | C1 output, tok/s | C1 steps/s | C8 output, tok/s | C8 steps/s | 32K prefill, tok/s |
|---|---:|---:|---:|---:|---:|
| 20 MiB / 16 | 174.91 | 83.41 | 880.19 | 423.90 | 16,338 |
| 12 MiB / 16 | 184.46 | 88.65 | 904.15 | 439.49 | 16,387 |
| 20 MiB / 2 | 198.30 | 93.45 | 933.18 | 446.50 | 16,347 |

The smaller-segment restart uses approximately 296–306 MiB less median sampled
device memory across the four GPUs. Its C1 output/execution is +5.46%/+6.28%;
C8 is +2.72%/+3.68%. This reverses the negative execution deltas in the
three-window screen, so that screen does not demonstrate a repeatable allocator
regression. Nor does this repeat establish a reliable speedup: the identical
20-MiB control itself changes from 89.22 to 83.41 C1 steps/s between series.

The separate C8 traces retain matching kernel names, geometry and counts on
all four ranks, including 192 target dynamic-MoE calls and 420 NCCL calls per
rank. There is no observed backend-selection change. Node-level profiling
perturbs timing, so those traces are not substituted for unprofiled throughput.
The memory saving is repeatable; the direction of the speed change is not.

The two-channel arm uses approximately 786–1,232 MiB less median sampled
device memory across the four GPUs. It improves output in both series:
C1 +4.58% in the screen and +13.38% in the repeat; C8 +0.90% and +6.02%.
The repeat's verifier changes are +12.03% C1 and +5.33% C8. Because the
control startup itself varies, these measurements do not establish a universal
13% speedup. They support a model-specific TP4 MTP3 option, not changing
every model's channel count or extrapolating to TP1/TP2.

The C8 trace changes the NCCL ring all-reduce grid from 10 to 2 CTAs while
retaining 404 calls per rank. Rank 0's median collective duration increases
from 21.15 to 24.10 μs; the profile therefore does not show individually faster
all-reduce kernels. Whole-server throughput and overlapping, profiled kernel
durations measure different things. No single-kernel explanation is claimed.

## Deployment recommendation

Keep the shared TP4 NCCL defaults. In particular, do not copy Spark TP2's
two-channel setting into every model: the DFlash2 C8 confirmation shows why.
Retain the 20-MiB allocator setting for DeepSeek V4.1 unless its additional
KV capacity is worth the measured C1 trade-off.
For Qwen TP4/MTP3, two channels are a measured opt-in alternative with lower
device usage; retain the linked startup variability when quoting its benefit.

The four-MiB cuBLAS workspace is the most consistent isolated memory-saving
candidate in these six TP4 modes: the five automatic-KV profiles report
205–246 MiB more available rank-0 KV memory; fixed-budget Qwen instead uses
about 338–344 MiB less sampled device memory. Prefill changes range from
−0.04% to +0.90%, and Qwen C8 output changes by −0.51%. The complete table
retains the other mode deltas rather than implying every output sample improves.
This does not qualify a combination of the four controls or their use at TP1/TP2.

## Interpretation limits

A memory saving alone does not qualify a universal default. Repeat material
throughput deficits and test any proposed combination independently. These
isolated tests do not qualify lower fixed KV budgets, altered graph ladders,
different draft precision or arbitrary long-context/multimodal memory peaks.
Several identical-image restarts differ by multiple percent in execution rate.
The screen and confirmations remain separate; neither a single startup nor a
short node-level trace establishes the cause of that variability.
