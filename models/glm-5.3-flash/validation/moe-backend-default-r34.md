# B12X MoE deployment default: R34 qualification

Status: **qualified for the configuration and serving checks below**. This is
a historical release report; it does not claim a complete multi-model retest.

Image: `localinferencelab/vllm:jovian-judgement-community-20260910-r34`.
The [registry receipt](moe-backend-default-r34-registry.json) identifies the
immutable digest and verified pull. The [qualification record](moe-backend-default-r34.json)
includes raw prefill/decode samples; the [artifact audit](moe-backend-default-r34-artifact.json)
checks source and compiled-library identity.

## Changes from R33

- Native `vllm serve` and Python kernel configuration default to B12X MoE through
  image environment setting `VLLM_DEFAULT_MOE_BACKEND=b12x`. Omitting the target
  option no longer selects FlashInfer CUTLASS through the NVFP4 automatic oracle.
- Explicit CLI/API backend selection, including `auto`, remains authoritative.
  An omitted flat option preserves explicit nested kernel configuration.
  Invalid environment defaults fail validation. Outside this image, an unset
  environment retains portable vLLM's `auto` default.
- The GLM wrapper and Qwen Compose recipe already explicitly selected B12X in
  R33. Their backend selection is unchanged. Attention, linear and sampler
  backends and explicit speculative-model choices are separate settings.
- All R33 B12X and LMCache sources, launcher implementations, checkpoint
  policies, quantization and sampling defaults are retained. All 14 audited
  native binaries are byte-identical, including the corrected filesystem
  connector. The final image still has two filesystem layers.

Review units: [vLLM #728](https://github.com/local-inference-lab/vllm/pull/728)
and the existing [source-locked recipe #31](https://github.com/local-inference-lab/blackwell-llm-docker/pull/31).
No B12X PR is needed for this configuration change. R33's contributor
attribution and complete source ancestry are preserved; its unmerged review
units are not superseded merely by inclusion in R34.

## Configuration and artifact checks

Thirteen tests pass in the packaged R34 environment: unset and configured
defaults, normalization, invalid defaults, explicit `auto` and alternate
backends, nested configuration, and complete CLI-to-engine resolution. Tests
construct a tiny local model configuration without downloading weights.
Twenty-nine source-lock and native-artifact reuse tests pass. Repository
pre-commit checks, including Ruff and mypy, pass.

The [source lock](moe-backend-default-r34.source.lock) authenticates the complete
[vLLM integration](https://github.com/voipmonitor/vllm/tree/codex/jovian-b12x-default-r34-20260910)
at `c496604123b1f4441007b952a7ee37ab12c8f6ad`. B12X remains
`59d51a36a942d56a9c36265855cdc7856fa7712e`; LMCache remains
`29bc5a2efde737c436b04499eb62cd1776cebeec`. Package/module version agrees with
the lock: `0.26.1rc0+glm53.r34.vllmc4966041`.

## Qwen backend comparison

Both arms use the **R33 image**, the same physical RTX PRO 6000 Blackwell
Workstation GPU, 600 W, graphics offset zero and **VRAM +6000**. TP1/MTP3,
FP8 KV, CPU PLE offload, BF16 target head and a private NVFP4 draft head with
BF16 activations. OMP2, 6019-token budget, 16 sequences, full-and-piecewise
graphs through 64 rows. Temperature 1, top-p 0.95, top-k 20; EOS respected.
Target revision: `b797d2e1160b9596b2570e56c1d3590faa09d4ed`.

Only the target MoE option is omitted in the automatic arm, which the engine
log confirms selects FlashInfer CUTLASS. The draft remains B12X. Compiler-cache
namespaces are isolated; image, model, remaining arguments and mounts match.
C1 uses three independent runs with 10-second warmup and 30 measured seconds.
Prefill uses two warmups and five exact 32768-token requests with one output
token; every measured request reports zero cached tokens.

| Measurement | FlashInfer CUTLASS | B12X | Change |
|---|---:|---:|---:|
| C1 output tok/s, median | 182.970 | 204.039 | +11.52% |
| C1 output tok/s, min–max | 175.268–193.356 | 194.755–207.665 | — |
| C1 verifier steps/s, median | 86.784 | 96.915 | +11.67% |
| C1 verifier steps/s, min–max | 86.417–86.944 | 96.844–97.018 | — |
| Emitted tokens per verifier step, median | 2.1173 | 2.1031 | −0.67% |
| 32K input tok/s, HTTP-wall median | 16,958.185 | 17,268.178 | +1.83% |
| 32K input tok/s, engine-accounted median | 17,164.320 | 17,467.791 | +1.77% |

All six decode cells finish without errors, loops, underfill or warmup timeout.
The verifier difference exceeds the observed within-arm spread; the small
prefill difference is not evidence of a universal gain. No bit-exact output
equivalence between the two MoE implementations is claimed. A return control
was stopped during model loading before measurement and is not counted.

These are backend-selection results, **not R33-to-R34 speedups**: the published
R33 Compose recipe already selected B12X.

## GLM DFlash2 image control

R33 and R34 run sequentially on the same physical quartet of RTX PRO 6000
Workstation GPUs, **VRAM +6000**, 600 W, TP4/DCP1, FP8 target KV and GPU cache.
Both select B12X MoE. Target revision `46aaae8a82032f77100f2f03e9cc11b391df3b4d`,
offline MXFP8 DFlash2 conversion `dc77ff1`, seven speculative tokens, OMP1,
4096-token budget, 16 NCCL channels/2 MiB buffers, full-and-piecewise graphs.

Prefill uses exact 32768-token requests, one output token, one warmup and at
least 30 measured seconds; every sample computes all tokens locally. C1 uses
context argument zero, 15-second warmup and one 30-second cell. Temperature 1
and model-default top-p 0.95; prefill supplies top-p 0.95 explicitly.

| Measurement | R33, B12X | R34, B12X | Observed change |
|---|---:|---:|---:|
| 32K input tok/s, median | 16,961.14 | 16,997.00 | +0.21% |
| C1 output tok/s | 217.24 | 249.77 | +14.97% |
| C1 verifier steps/s | 92.47 | 97.40 | +5.33% |
| Emitted tokens per verifier step | 2.3493 | 2.5643 | +9.15% |

The short C1 observations differ in acceptance and execution rate. They show
no loss in this check, but do not establish a causal gain from the default
setting: both images use the same B12X sources and kernels. Do not advertise
the 14.97% observation as a general R34 decode speedup. R34 also passes three
deterministic document lookups with all nine requested values exact and finite
logprob checks. The API remains healthy after testing.

No fresh C8/Sieve, no-spec/MTP3 matrix, DCP4, TP8, NVFP4 KV, or Qwen/DeepSeek
external-cache qualification is claimed. Their existing evidence remains
attached to its measured artifact. The concurrent constrained-output/cache
failure tracked by [vLLM #726](https://github.com/local-inference-lab/vllm/issues/726)
is not fixed by this release.
