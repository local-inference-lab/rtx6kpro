# NVFP4 shared-input prefill and filesystem eviction: R33 qualification

Status: **qualified for the bounded conditions below**. This is a historical
R32-to-R33 release report, not a general model-quality or all-topology claim.

Artifact: `localinferencelab/vllm:jovian-judgement-community-20260910-r33`.
Tested image ID:
`sha256:8a4aa9d80cd52d31a70702917c8efb5b54281b53910a20712e5a5c3b6f73b57c`.
The [registry receipt](fp4-prefill-filesystem-r33-registry.json) records its
immutable registry digest, successful pull-by-digest and two filesystem layers.
The [qualification record](fp4-prefill-filesystem-r33.json) retains raw prefill
samples, decode observations, source identities, checks and excluded failures.

## Changes from R32

- **Shared NVFP4 input quantization and separate expert projections.** B12X
  proves exact, finite, positive equality of immutable expert input scales
  once during weight preparation. Eligible prefills quantize each token once
  for its routed experts and use separate gate/up and down-projection kernels.
  The checkpoint, canonical scale vectors, target precision, draft-head choices
  and KV types do not change. Read-only custom-op schemas preserve the proof;
  mutable scale owners retain mutation protection.
- **Filesystem eviction accounting.** Deleting an already-absent object counts
  as successful cleanup, so byte accounting and the eviction index cannot retain
  a nonexistent file indefinitely. Pending stores are protected from eviction.
  Actual filesystem errors still propagate without retiring their accounting.
  Bounded long-key and legacy-path handling remain supported.
- **CPU-only native rebuild.** Only the LMCache filesystem library changes
  among 14 audited native binaries. CUDA, vLLM and FlashKDA binaries, launchers,
  sampling/history defaults, checkpoint policies and model profiles are retained.
  The source-locked build validates native-source compatibility before reuse.

Review units:
[B12X #353](https://github.com/local-inference-lab/b12x/pull/353) (L14nY1Wang's kernels),
[B12X #354](https://github.com/local-inference-lab/b12x/pull/354) (scale proof and mutation schemas),
[vLLM #727](https://github.com/local-inference-lab/vllm/pull/727) (inference-owner declaration),
[LMCache #67](https://github.com/local-inference-lab/LMCache/pull/67) (Derek Yates's eviction fix),
and [Docker recipe #31](https://github.com/local-inference-lab/blackwell-llm-docker/pull/31).
The original contributor commits and source-commit references are preserved.

## Source and packaging identity

| Component | Committed integration |
|---|---|
| vLLM | [`ae89131442359dc332d9c46009be3c1f8cdee0b4`](https://github.com/voipmonitor/vllm/tree/codex/jovian-nvfp4-split-r33-20260910) |
| B12X | [`59d51a36a942d56a9c36265855cdc7856fa7712e`](https://github.com/voipmonitor/b12x/tree/release/jovian-nvfp4-split-r33-20260910) |
| LMCache | [`29bc5a2efde737c436b04499eb62cd1776cebeec`](https://github.com/local-inference-lab/LMCache/tree/release/jovian-fp4-fs-ledger-r33-20260910) |

The [embedded source lock](fp4-prefill-filesystem-r33.source.lock) has SHA-256
`c4b1029eb355736f94b02efa19b10efe4d9680cac41ae38b952b05de9e4381c4`.
All three integrations retain R32 ancestry and clean Git trees. Package and
module version agree: `0.26.1rc0+glm53.r33.vllmae891314`.
The [independent artifact audit](fp4-prefill-filesystem-r33-artifact.json)
checks source-lock inputs, launcher hashes, layer count and native-library hashes.

R32 reference:
`localinferencelab/vllm@sha256:c9ad4a6ef4aa55232df9ed1a37e85d94eb8c7d5349561a6cbe828e72b61de83c`.
Reproduction uses the [portable source-locked recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/codex/glm53-source-locked-build/recipes/glm53)
with `--lmcache-native-mode cpu-rebuild-cuda-reuse`. Rebuilding from compatible
source does not promise identical OCI timestamps or archive metadata.

## GLM performance

All A/B arms run sequentially on **the same physical GPUs 4–7** of `aiserver`:
four RTX PRO 6000 Blackwell Workstation GPUs, 600 W limits, graphics offset
zero, **VRAM +6000**. They are not stock-clock or Max-Q results.
The target revision is `46aaae8a82032f77100f2f03e9cc11b391df3b4d`; the DFlash2
MXFP8 checkpoint is the unchanged locally staged `dc77ff1` conversion.
Shared settings: TP4/DCP1, FP8 target KV, GPU cache, 4096-token scheduler
budget, OMP1, 16 NCCL channels with 2 MiB buffers, and full-and-piecewise graphs.

Prefill sends exactly 32768 input token IDs and one output token, excludes one
warmup and measures for at least 30 seconds. A unique leading nonce and server
counters prove full local computation with zero prefix-cache reuse. The rate is
input tokens divided by client time to first token, not pure MoE kernel time.
No-spec uses temperature 0/top-p 1 to match its existing bracketed controls;
MTP3 and DFlash2 use temperature 1/top-p 0.95. Compare within a row.

| 32K prefill, input tok/s | R32 control | R33 packaged image | Change |
|---|---:|---:|---:|
| No speculation | 15,737 | 17,128 | +8.84% |
| MTP3 | 15,276 | 16,637 | +8.91% |
| DFlash2 K7 | 15,569 | 16,910 | +8.61% |

The other no-spec control is 15,842 tok/s, giving +8.12% instead of +8.84%.
A DFlash2 control repeat gives 15,505 tok/s. All control samples remain in the
receipt. A source-overlay prototype measured 17,152 no-spec input tok/s; it is
not substituted for the packaged R33 measurement.

Decode uses llm-inference-bench 0.6.1, context argument 0, temperature 1,
model-default top-p 0.95, respected EOS, ten-second warmup and 30-second cells.
C8 is aggregate throughput across eight clients. Multiple observations are
shown as ranges; an unavailable no-spec verifier metric is not a zero rate.

| Mode | C1 output tok/s, R32 → R33 | C1 verifier steps/s, R32 → R33 | C8 output tok/s, R32 → R33 | C8 verifier steps/s, R32 → R33 |
|---|---|---|---|---|
| No speculation | 176.80–189.66 → 177.05 | Not separately reported | Not rerun → 771.50 | Not separately reported |
| MTP3 | 281.16 → 275.45 (−2.03%) | 112.65 → 113.76 (+0.99%) | 1018.61 → 1002.93 (−1.54%) | 406.38 → 405.09 (−0.32%) |
| DFlash2 K7 | 250.88–263.54 → 244.10–260.56 | 96.65–101.02 → 97.27–97.32 | 783.71–785.44 → 785.79–789.48 | 303.56–308.39 → 309.69–310.72 |

MTP3 accepted length changes from 2.496 to 2.422 at C1 and 2.507 to 2.476
at C8. DFlash2 C1 acceptance is 2.596–2.609 in R32 and 2.510–2.678 in R33.
Output rate therefore cannot be interpreted independently of acceptance and
verifier execution. The initially lower DFlash2 verifier state also reproduces
on unmodified R32; no candidate-specific slow state is established. These
short observations do **not** justify a general decode speedup or statistical
equivalence claim. All reported cells have zero errors, loops, underfill and
warmup timeouts.

An earlier R32 MTP client was interrupted when the operator stopped its server
before the client finished. Its error cells are excluded explicitly, not
classified as a model regression. The replacement measurement completes normally.

## Qwen and DeepSeek scope

Qwen TP1/MTP3 uses one physical GPU4 with the same +6000/600 W settings,
FP8 KV, CPU PLE offload, BF16 target head, a private NVFP4 draft head with BF16
activations, OMP2, 6019-token budget and graph capture through 64 rows.
Temperature is 1, top-p 0.95 and top-k 20. The target revision is
`b797d2e1160b9596b2570e56c1d3590faa09d4ed`.

The independently measured source-overlay comparison gives 32K HTTP-wall
prefill **15,707.42 → 17,211.35 tok/s (+9.57%)**. It excludes two warmups
and retains five measured requests per arm. C1/C8 verifier changes are −0.83%
and −0.12%; no decode gain is claimed. The packaged-image cookbook is recorded
separately in the release JSON and on the [Qwen page](../../qwen38-flash-next.md).
The packaged R33 image independently gives **17,192.95 input tok/s (+9.46%)**,
199.69 C1 output tok/s and 732.85 C8 aggregate output tok/s. It checks
68 cold/repeated requests, six instruction-prefix cases, 38 post-prefill
logprob requests, C1/C8 decode and five cold 32K prefills.

Qwen TP1's intermediate width 640 meets the split tile's 128-element constraint.
TP2 width 320 does not; supported monolithic execution remains available and
no TP2 gain is claimed. The 512-expert target and W4A16 draft paths stay distinct.

DeepSeek V4 already uses its native shared-input MXFP4/MXFP8 split path.
A TP2/DCP1 reference trace verifies separate projections, but there is no DS4
runtime optimization attributed to these NVFP4 patches. R29 text/Vision and
cache qualification remains historical evidence, not a fresh R33 DS4 test matrix.

## Correctness and numerical limits

- **39 installed B12X CPU/GPU tests pass**, including mutation/version accounting,
  policy eligibility, independent operation oracles, graph replay, caller-owned
  allocations and the Qwen width-640 geometry.
- All three packaged GLM modes pass three approximately 32K literal-document
  lookups each, with nine exact values per mode and processed logprobs.
- The independent model-shaped operation oracles satisfy mean cosine ≥0.999
  and normalized RMSE ≤0.03. Split error is lower than monolithic error against
  those oracles. **Strict cross-kernel cosine 0.9999 fails** (approximately
  0.999889 GLM and 0.999874 Qwen). Bit-identical logits, identical generated
  sequences and broad target-distribution equivalence are not claimed.
- The [committed B12X evidence](https://github.com/voipmonitor/b12x/blob/perf/nvfp4-immutable-input-scales/docs/nvfp4-immutable-input-serving.md)
  includes commands, exact source boundaries, GPU UUIDs, warm/cold timings and
  real serving traces. GLM records 168 calls each to routing/FC1/FC2; Qwen 288 each.

## LMCache filesystem qualification

R32 reproduces stale eviction accounting for both short and bounded long keys
whose file was already absent. The rebuilt candidate passes **116 connector
and adapter tests**, including pending-store exclusion and real-I/O-error
preservation. The source-lock recipe passes **28 tests**.

A separate quartet, physical GPUs8–11, runs FP8 TP4/DCP1 MTP3 with a 4096-token
budget, 4 GiB RAM tier and native filesystem storage. This is a correctness
test, not a speed comparison with GPUs4–7. The [tier receipt](fp4-prefill-filesystem-r33-cache.json)
records all counters and answers:

| Request path | Prompt tokens | Recomputed tokens | Complete response time |
|---|---:|---:|---:|
| Cold | 54,641 | 54,641 | 7.904 s |
| GPU-local replay | 54,641 | 0 | 0.300 s |
| RAM restore | 54,641 | 0 | 0.260 s |
| Filesystem restore | 54,641 | 0 | 0.277 s |
| RAM replay of disk-retained objects | 54,641 | 0 | 0.256 s |
| Restore after workers and sidecar restart | 54,641 | 0 | 0.414 s |

Every requested value is exact. Filesystem measurements use a warm OS page
cache and include answer generation; they are not raw storage bandwidth.
Shared SYSTEM instructions restore 11,340 tokens with a different user turn;
changed SYSTEM instructions miss and produce the changed exact answer.
Retained RAM remains bounded and restore leases drain. The sidecar registers
four non-CUDA workers; GPU copies remain in the vLLM workers.

No additional DCP4, TP8, NVFP4 target-KV, long-run disk-pressure or Qwen/DS4
external-cache qualification is claimed. The pre-existing
[MTP constrained-JSON/LMCache failure #726](https://github.com/local-inference-lab/vllm/issues/726)
remains unresolved. R33 fixes filesystem accounting, not that grammar defect.
