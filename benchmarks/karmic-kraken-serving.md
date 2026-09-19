# Karmic Kraken model-serving measurements

Status: **qualified** bounded serving/cache checks; remaining performance
deficits are **research-only**. These are saved RTX PRO 6000 Workstation
measurements at stock clocks, without +6000 VRAM overclock. Updating launch
documentation does not rerun or reattribute them to another image digest.

## Decode and prefill

Context-zero decode, temperature 1, three warmed 30-second runs per concurrency.
Prefill uses unique uncached 32K requests over 30 seconds and client time to
first token. C8/C4 are aggregate output. All rows use DCP1 and a 4096-token
budget, except Qwen's matched 6019-token budget and explicit eight-GiB KV cache.

| Model / mode | GPUs | JJ → KK C1, tok/s | JJ → KK concurrent output, tok/s | JJ → KK 32K prefill, tok/s |
|---|---:|---:|---:|---:|
| GLM MTP3 | 4 | 249.33 → 255.71 (+2.56%) | C8: 875.86 → 878.74 (+0.33%) | 15,580 → 15,562 (−0.12%) |
| GLM DFlash2 K7 | 4 | 219.83 → 225.07 (+2.39%) | C8: 717.84 → 707.02 (−1.51%) | 15,734 → 15,798 (+0.41%) |
| DS4 text DSpark K5 | 2 | 190.12 → 205.54 (+8.11%) | C8: 669.56 → 667.18 (−0.36%) | 13,863 → 13,957 (+0.68%) |
| DS4 Vision DSpark K3 | 2 | 185.49 → 188.73 (+1.74%) | C4: 423.11 → 424.80 (+0.40%) | 10,615 → 10,691 (+0.72%) |
| DS4.1 DSpark K7 | 4 | 256.59 → 273.96 (+6.77%) | C8: 830.82 → 813.87 (−2.04%) | 20,234 → 19,973 (−1.29%) |
| Qwen MTP3 | 1 | 190.11 → 178.52 (−6.09%) | C8: 689.93 → 675.17 (−2.14%) | 15,162 → 14,915 (−1.63%) |

These output differences include stochastic draft acceptance; they are not
kernel-speedup percentages. DS4.1's matched throughput test retains the JJ
RAM-Engram placement and 131,072-token cap. Disk Engram and automatic context
admission are separate functional checks, not the throughput numbers above.

Follow-up measurements retain, rather than hide, the negative results:

- GLM DFlash2 with disjoint KDA scratch: six-run combined medians of 217.32
  C1 and 705.82 C8 tok/s, −1.14% and −1.67% against the saved JJ numbers.
  One repeated prefill window is 15,730 tok/s. Sieve varies independently;
  the five-request repeated median is 503.02 tok/s, not the llmbench C1 rate.
- DS4.1 repeated C8 median: 811.06 tok/s, −2.38%. Repeated prefill windows
  of 20,205 and 20,334 tok/s do not reproduce the initial prefill deficit.
- Qwen repeated C1: 179.87 tok/s, −5.38%; its 84.01 verifier steps/s are
  +0.65%, so output and execution rate do not move together. Repeated C8:
  687.77 tok/s, −0.31%. Prefill windows: 14,862 and 14,754 tok/s.

The [GLM Spark TP2 page](../models/glm-5.3-flash-spark-tp2.md) contains the
separate TP2/DCP2, 3072-budget result: C1 186.5, C4 403.1 and 32K prefill
10,919 tok/s. It is one warmed window per decode concurrency, not this
three-run TP4 comparison.

## Prefix cache checks

Factual requests have a unique cache salt. CPU restores follow a GPU-prefix
reset; persistent restores follow a restart of both serving and the cache
process. Correct answers and zero GPU hits are required for external recovery.
These are functionality checks, not external-cache throughput measurements.

| Model / request | Restored prefix | Persistent objects | Result |
|---|---:|---:|---|
| Qwen TP1, text | 16,301 tokens | 116 | Correct RAM and restart-disk answers |
| DS4 text, TP2 | 12,288 tokens | 14 | Correct RAM and restart-disk answers |
| DS4 Vision, text and image | 12,288 tokens | 14 | Correct restore; changed image misses and answers correctly |
| DS4.1, text | 12,288 tokens | 19 | Correct RAM and restart-disk answers |
| DS4.1, image before shared text | 16,384 tokens | 23 | Correct restore; changed-image negative control passes |
| GLM Spark TP2, public image | 16,285 tokens | 28 | Correct RAM and restart-disk answers, zero GPU hits |

GLM and Qwen external recurrent checkpoints are text-only. Vision remains
usable, but the connector recomputes those requests instead of reusing an
unauthenticated image checkpoint. DS4 Vision and DS4.1 use image-aware keys.

## Source boundaries

The initial KK GLM/DS4/Qwen matrix uses vLLM `bb709f4acee`, B12X
`eea3ced11fc1`, CUDA 13.4.1 and PyTorch 2.14. The GLM DFlash adaptation and
DS4.1 sampler warmup use vLLM `171c5b2c111`; the disjoint-scratch DFlash
follow-up uses `9872cfefe738`, with the same B12X revision. The saved JJ matrix
uses vLLM `1048fc5439d` and B12X `d9b572754a`.

The multimodal cache-isolation follow-up uses LMCache `688bee14e157`.
The public GLM Spark TP2 cache check uses vLLM `bd76814003bd`, B12X
`eea3ced11fc1`, that LMCache wheel, and image digest
`sha256:038ab1d937cfbd42272af364171396eef4fa1d12d4f66b6f55740cdabb3271eb`.
Its Max-Q validation host is not the stock Workstation speed-measurement host.

[Issue #808](https://github.com/local-inference-lab/vllm/issues/808) lists the
attributed PRs reconstructing the integrated serving sources. The
[JJ benchmark record](prepared-b12x-serving/) retains its raw reference data.
Changing a floating image tag does not change any of the measurement identities
above; use the corresponding release manifest when reproducing a result.
