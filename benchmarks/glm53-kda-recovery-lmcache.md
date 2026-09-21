# GLM speculative KDA recovery with atomic LMCache checkpoints

## Behavior

Implemented: supported GLM-5.3 speculative decoding uses a single FP32 recurrent
checkpoint and compact per-token records instead of a full recurrent matrix for
every proposed token. B12X CuTe kernels verify the proposed tokens and commit
only the accepted updates. vLLM selects this automatically for MTP or DFlash
with one to seven draft tokens; `--no-use-replayssm` retains full speculative
states for diagnostic comparisons.

The automatic path also supports the atomic request-boundary LMCache connector.
After stop-token trimming, the runner first commits the accepted recurrent state
and compacts convolution history, then exports the checkpoint. The exporter must
not apply the speculative acceptance offset a second time. Without that ordering,
an external cache can preserve a stale recurrent state beside valid attention KV.

The connector already transports complete physical state pages and namespaces
them by model, runtime and cache layout. No LMCache source or disk-format change
is required. Full-checkpoint decoding keeps its existing capture-before-compaction
order. Generic aligned KV connectors and disaggregated prefill/decode are not
supported by this recovery integration.

## Precision and eligibility

The recurrent matrices and rank-one corrections remain FP32. Recorded keys and
forget gates preserve their original BF16 input values; this change adds no
quantization. Reconstructed arithmetic is checked against the FP32 recurrence
within numerical tolerance, not asserted bit-identical for every input.

Automatic selection requires GLM-5.3, BF16 activations, 128-wide KDA heads, FP32
recurrent state, an SM12x CUDA GPU, Model Runner V2, pipeline parallelism one,
and `none` or `align` Mamba cache mode. Stochastic recurrent-state rounding is
excluded. Unsupported automatic configurations retain their existing path;
explicitly forcing recovery in an unsupported configuration raises an error.
Other models remain opt-in.

The Mamba metadata backend is named `TRITON` in vLLM. That name does not identify
the GLM recovery compute kernels: speculative verification and grouped state
commit use B12X CuTe. Existing auxiliary normalization, metadata and convolution
compaction kernels remain in use.

The record/recovery contract derives from the Kimi-K3 integration by Jiangyun
Zhu, vLLM commit `70afdedc1081d28c3eaae53bece8292298484c86`. B12X implements the
recurrent computation and accepted-state reconstruction in CuTe. The changes
were developed with OpenAI Codex assistance.

## Source and test environment

The source PR branches are based on the canonical component branches:

| Component | Base | Tested feature revision |
|---|---|---|
| B12X master | `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68` | `28f023427b3a93ea72c17534802cd483cc0c3067` |
| vLLM dev/karmic-kraken | `af9e4dca109e0348323c0182e98a3aaf7282bfc3` | `4c954ffe6b97c32a64def88f12c27f66be36cb87` |

Review follow-ups document buffer contracts and reduce the large-offset test's
storage in B12X `b9ce2c05`. Follow-up [B12X #409](https://github.com/local-inference-lab/b12x/pull/409)
corrects binding diagnostics, defines final-state precedence for aliased outputs
and removes redundant static validation during execution. These changes do not
alter the recurrent GPU programs. Additional
runner coverage checks that logits-only restores reuse their existing checkpoint
without applying recovery records or exporting a duplicate. The caller supplies
no capture request in that mode; treating a synthetic logits-only/capture pair
as a serving failure would violate the caller's contract. The final-image checks
are recorded separately from the frozen throughput composition below.

Serving uses those changes composed with the integration branches. B12X
composition `ae981e79621a64ac3daabfd8b4875b55b86719a6` includes beta
`f6d8b8eb94cdeb4e652652f925a494c6fc86f101`; vLLM composition
`74102adcccd7aaae0c9e173ac6d926c2b5495b7a` includes beta
`22476af54c637cbb7c7d8193addd160da83a5ce3`. This preserves the beta cache-tail and
QSA fixes while testing the recovery feature.

Runtime image ID:
`sha256:d351927666613339608da3ec6c6f227048a60cb7c2a46baff2edf8a4b1dbb84c`.
Read-only source overlays replace B12X Python and the changed vLLM Python files;
installed native extensions are unchanged. This is a source-composition test,
not a claim that a published image already contains the feature.

Dependencies: PyTorch `2.14.0a0+4fdf77b940.nv26.8.63802676`, FlashInfer
`0.6.18+lil.cu134.sm120.g2206a14e4638`, LMCache
`0.5.3.dev365+lil.cu134.g688bee14e157`, NCCL 2.31.2.

Hardware: RTX PRO 6000 Blackwell Max-Q, remote GPUs 5–8. Memory clock observed
under load is 13,365 MHz; clocks were not changed for these tests. No throughput
comparison to a different GPU quartet or historical overclock is implied.

Common settings: B12X autotuning enabled, B12X target attention and MoE,
full-and-piecewise CUDA graphs, runner V2, FP8 MLA cache, two-shot all-reduce
disabled, LMCache 16 GiB RAM and 64 GiB disk with a separate volume per mode.
Functional requests use temperature 1 and top-p 0.95.

The complete launch recipe is
[start-glm53-recovery-cache-qualification.sh](glm53-kda-recovery-lmcache/repro/start-glm53-recovery-cache-qualification.sh).
The controller checks the container identity and assigned GPUs before restarting
only that container and its internal LMCache sidecar:
[qualify-glm-recovery-cache-mode.py](glm53-kda-recovery-lmcache/repro/qualify-glm-recovery-cache-mode.py).

## Kernel and runner validation

Qualified: **131 B12X tests passed, 29 skipped; 28 vLLM tests passed** on the
composed beta sources. Coverage includes windows of 1/4/8 tokens, FP32 recurrence
oracles, accepted and aligned-boundary states, unchanged verification checkpoints,
CUDA graph replay under frozen kernel resolution, caller-owned storage,
interleaved state fields, null entries, and pool offsets exceeding 2^31 elements.
Runner tests cover both capture orders and copying an already committed state
without speculative bias. Config tests cover automatic LMCache eligibility,
explicit opt-out and unsupported configurations.
The B12X suite also exercises capacity errors through public bindings, final
state precedence when destinations alias, and execution without repeated static
validation. Four CPU runner cases verify the logits-only checkpoint contract.

Maximum observed errors in the GPU suite: output relative L2 `7.995e-6`,
accepted-state absolute error `2.236e-8`, persistent-state absolute error after
16 windows `1.908e-6`.

Raw results:
[B12X XML](glm53-kda-recovery-lmcache/b12x-final-contract-tests.xml),
[vLLM XML](glm53-kda-recovery-lmcache/vllm-reviewed-tests.xml),
[CPU contract XML](glm53-kda-recovery-lmcache/logits-contract-cpu-tests.xml).

Changed-file lint and formatting checks pass. The full vLLM mypy hook reports an
existing `GPUModelRunner.jit_warmup_registry` attribute error, reproduced on the
untouched base. B12X's GDN test file also has an existing Ruff F821 warning for
a closure over `recurrent_state`, reproduced at its unmodified master base.
Neither existing failure is reported as a passing hook for this feature.

## TP2 MTP3 external-cache qualification

Checkpoint: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision
`a608241037e4c2565356bff7ca293f2133888f88`. TP2/DCP2, MTP3, batch budget 3072,
maximum four sequences, fixed KV allocation 4,190,109,696 bytes per rank.
LMCache chunks contain 3072 tokens and the launcher uses request-boundary
checkpoints. Reported logical capacity is 899,579 tokens; fitted context 897,024.

All external-restore checks clear native GPU prefix cache and require zero GPU
hits. Disk checks restart both the model server and LMCache sidecar, then require
an increase in disk-loaded objects. Lookup answers are compared to known answers
and cold controls, not judged solely from hit counters.

| Check | RAM external tokens | Restart/disk external tokens | Result |
|---|---:|---:|---|
| Identical prompt | 16,286 | 16,286 | Correct answer; 28 disk objects |
| Different user turn | 16,280 | Included in text probe | Correct changed answer |
| Changed tail of one long user prompt | 30,720 | 30,720 | Correct changed answer; 36 disk objects |
| Continue generated response | 14,451 | 14,466 | Restores beyond prompt; 26 disk objects |
| Edit early in the document | 0 | — | Correct cache miss and recomputation |

Qualified: text RAM/disk restore, generated-response continuation, native prefix
reuse, vision generation, and an overlapping 20,515-token prefill plus 2,048-token
decode. The existing atomic checkpoint connector explicitly excludes
image-bearing requests. Such requests still generate correctly and use native
GPU prefix caching, but **external multimodal checkpoint reuse is unsupported**;
the failed external-image hit probe is retained in the evidence directory.

Raw evidence is under
[glm53-recovery-lmcache-20260921](glm53-kda-recovery-lmcache/).

## TP2 throughput and limits

llm-inference-bench 0.6.2, context zero, temperature 1, 15-second decode warmup,
three 30-second measured windows, maximum 8192 output tokens, respecting EOS.
API prefill uses one warmed 30-second window of uncached approximately 32k
requests. The reported prefill metric includes time to first output and is not
isolated GPU kernel time.

| Measurement | Samples, tok/s | Median, tok/s |
|---|---|---:|
| C1 output | 186.997 / 166.891 / 166.415 | 166.891 |
| C4 aggregate output | 383.477 / 382.790 / 377.265 | 382.790 |
| 32k prefill | 9,678; 9 requests, 32,770 actual prompt tokens | 9,678 |

C1 verifier rates were 68.632 / 68.663 / 68.820 steps/s. The first C1 stream
ended with 269 repetitions of `OK. `; the benchmark marked it suspected rather
than confirmed looping. That sample is not clean output-quality evidence and
its higher token rate is not a performance claim. The other five decode cells
had no repetition flags. Functional requests passed, but these bounded checks
do not establish general model accuracy.

Research-only: comparing these rates with historical VRAM-only or autotuning-off
runs. Different cache geometry, autotuning, clocks and measurement windows prevent
attributing those differences to recovery.

The matched full-state control uses the same TP2 image, frozen sources, GPUs,
LMCache allocation and serving configuration, with `--no-use-replayssm` added.
All six control decode cells and all five serving checks passed without errors
or repetition flags. Both prefill windows contain nine uncached requests.

| Measurement | Full speculative states | Recovery | Recovery / full change |
|---|---:|---:|---:|
| C1 output, tok/s | 167.202 | 166.891 | −0.19% |
| C1 verifier, steps/s | 69.754 | 68.663 | −1.56% |
| C4 aggregate output, tok/s | 382.685 | 382.790 | +0.03% |
| C4 verifier, steps/s | 154.919 | 153.507 | −0.91% |
| 32k API prefill, tok/s | 9,539 | 9,678 | +1.46% |

These medians retain every measured window, including the flagged recovery
sample described above. Similar output throughput does not imply identical
execution cost or a general model-quality qualification.

## TP2 LMCache reservation accounting

An integer-planner comparison changes only the recurrent-state layout in the
3072-token LMCache configuration. It reproduces the serving capacity above.

| Per-request reservation, per rank | Full speculative states | Recovery records |
|---|---:|---:|
| Working recurrent blocks | 35 | 14 |
| Endpoint/restore blocks | 45 | 45 |
| Shared block bytes | 11,227,392 | 11,886,336 |
| Fixed request reservation, bytes | 898,191,360 | 701,293,824 |
| Fitted maximum context | 897,024 | 897,024 |
| Reported logical token capacity | 899,435 | 899,579 |

The fixed reservation decreases by **196,897,536 bytes (187.776 MiB), about
21.9% per request per rank**. This is a planner allowance, not bytes necessarily
resident for every idle conversation. Recovery records enlarge each physical
shared block; that reduces the number of blocks in the fixed pool from 373 to
352. Consequently this particular configuration saves fixed per-request memory
but does **not** increase the fitted maximum context of one long request. It
must not be advertised as a 21.9% increase in token capacity.

Reproducer: [account-glm-recovery-lmcache.py](glm53-kda-recovery-lmcache/repro/account-glm-recovery-lmcache.py).

## TP4 DFlash2 external-cache qualification

Checkpoint: `local-inference-lab/GLM-5.3-Flash-NVFP4`, revision
`520de24eabf507659eaef7c70f14fd584527facc`; prequantized MXFP8 draft
`local-inference-lab/GLM-5.3-Flash-DFlash2`, revision
`713226ab03bc38afdf955c7450436c2f7176f6f8`. TP4/DCP1, seven draft tokens,
4096-token batch/chunk budget, maximum 16 sequences, 131,072-token maximum
context, GPU memory utilization 0.93. Recovered recurrent state remains FP32;
MXFP8 refers to the draft weights, not recurrent state.

Qualified text RAM/disk restores use the same zero-GPU-hit controls as TP2:

| Check | RAM external tokens | Restart/disk external tokens |
|---|---:|---:|
| Identical prompt / different user turn | 16,287 / 16,281 | 16,287 |
| Changed tail of one long prompt | 28,672 | 28,672 |
| Continue generated response | 14,451 | 14,466 |

Disk loads transferred 80 / 112 / 80 objects respectively. All lookup answers
and five text/prefix/vision serving checks passed.

The matched full-state control uses the same image, source overlays, GPUs and
launch settings with only `--no-use-replayssm` added. Each arm has three warmed
30-second decode windows per concurrency and one warmed 30-second prefill window
containing ten uncached 32,770-token requests.

| Measurement | Full speculative states | Recovery | Recovery / full change |
|---|---:|---:|---:|
| C1 output, tok/s | 198.906 | 201.241 | +1.17% |
| C1 verifier, steps/s | 78.549 | 77.670 | −1.12% |
| C8 aggregate output, tok/s | 653.452 | 665.912 | +1.91% |
| C8 verifier, steps/s | 256.195 | 257.760 | +0.61% |
| 32k API prefill, tok/s | 12,725 | 12,714 | −0.09% |

All twelve decode cells had no errors or repetition flags. Output throughput
also reflects stochastic acceptance; these samples do not establish an isolated
speedup. The C1 execution delta is negative and retained explicitly. The bounded
comparison shows similar throughput while using the recovery layout.

## TP4 MTP3 external-cache qualification

The TP4 target and serving/cache settings match the DFlash2 arm, with MTP depth
three and the profile's Marlin draft MoE replacing the DFlash model. Recovery
is selected automatically; target attention and MoE remain B12X.

| Check | RAM external tokens | Restart/disk external tokens |
|---|---:|---:|
| Identical prompt / different user turn | 16,289 / 16,283 | 16,289 |
| Changed tail of one long prompt | 28,672 | 28,672 |
| Continue generated response | 14,456 | 14,471 |

All external probes had zero native GPU-cache hits and correct lookup answers.
Disk loads transferred 72 / 108 / 72 objects respectively. Text, prefix and
vision serving checks passed. Each throughput arm contains three warmed
30-second decode windows per concurrency. Each prefill window contains twelve
uncached 32,770-token requests.

| Measurement | Full speculative states | Recovery | Recovery / full change |
|---|---:|---:|---:|
| C1 output, tok/s | 247.130 | 239.376 | −3.14% |
| C1 verifier, steps/s | 99.014 | 96.770 | −2.27% |
| C8 aggregate output, tok/s | 839.817 | 824.829 | −1.78% |
| C8 verifier, steps/s | 337.155 | 334.615 | −0.75% |
| 32k API prefill, tok/s | 13,607 | 13,569 | −0.28% |

All twelve decode cells had no errors or repetition flags. This memory
optimization has a measured throughput cost for this MTP3 configuration; it is
not qualified as a speedup or as zero-regression. Output rates include acceptance
variation. Single under-load clock snapshots show P1, 13,365 MHz VRAM and event
mask `0x400` in both arms; observed SM clocks were 2,272–2,302 MHz for recovery
and 2,287–2,317 MHz for full states. The comparison is an end-to-end serving
measurement, not an isolated kernel timing claim.

## Container shutdown limitation

The original TP2 full-state startup refused an existing named LMCache RAM
arena after the preceding TP4 control exited with code 143. No process held
the arena; it was renamed, preserving its data, and the separately recorded
TP2 continuation completed all checks. The original failure receipt remains
in the evidence. This is a launcher shutdown-cleanup limitation, not a failed
recurrent-state calculation. The successful RAM and real-restart disk probes
do not establish recovery from arbitrary forced process termination.

## Publication

Luke merged [B12X #408](https://github.com/local-inference-lab/b12x/pull/408).
[B12X #409](https://github.com/local-inference-lab/b12x/pull/409) and
[vLLM #821](https://github.com/local-inference-lab/vllm/pull/821) remain non-draft
review PRs against `master` and `dev/karmic-kraken` respectively.

The `integration/karmic-kraken-beta` branches contain B12X
`cb56484ee2714bc8a00d4eccd0d53816dc62928f` and vLLM
`9e3eaac44013081b4a7859a628387503ed72e97c`. The component changelog fragments
travel with these commits; the vLLM fragment requires both B12X fragments.
The final B12X GPU suite used the equivalent runtime tree before the last
fragment-only commit. Throughput used the frozen composition described above;
the follow-up changes no recurrent GPU program. Verification of the generated
image and changelog remains pending.
