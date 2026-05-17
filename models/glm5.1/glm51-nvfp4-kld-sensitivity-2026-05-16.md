# GLM-5.1 NVFP4 KLD Sensitivity Resume - 2026-05-16

## Goal

Find which GLM-5.1 MoE expert layers are most responsible for NVFP4 KLD loss
against the FP8 reference, before attempting a new quantized checkpoint.

This was not a production checkpoint build. It started as a BF16 oracle sweep:
selected expert layers were dequantized from the FP8 reference to BF16 and left
unquantized, while the rest of the model stayed NVFP4.

Later validation repeated the best candidates with true BF16 expert weights
copied from the original `zai-org/GLM-5.1` checkpoint. Those BF16-direct results
are the cleaner signal and should be preferred over the older FP8-dequant
oracle when deciding which layers matter.

## Test Setup

- Base NVFP4 checkpoint:
  `/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/18175c33fa74faa199b5d462e02d12b9d3399295`
- FP8 reference checkpoint:
  `/root/.cache/huggingface/hub/models--zai-org--GLM-5.1-FP8/snapshots/a92f8155fe8574288534d41a6dbc1d72888ab2da`
- Reference logits:
  `/root/kld/current_cf2070d_b12xc929144_fp8_ref_wikitext_ctx2048_s512_w8_b12xmlasparse`
- Runner container:
  `glm51-kld-current-b12x`
- Runner image:
  `voipmonitor/vllm:glm51-canonical-git-vllmcf2070d-b12xc929144-cutedsl45-20260513`
- KLD windows:
  `w8`, context 2048, stride 512, 16376 scored positions
- Main results file:
  `/root/kld/oracle_sweep_results.tsv`
- Logs:
  `/root/kld/oracle_sweep_logs/`
- Sweep script:
  `/root/kld_ablation/run_bf16_oracle_sweep.py`
- Variant builder:
  `/root/kld_ablation/make_mixed_bf16_expert_variant.py`

Important runtime settings:

```bash
GLM51_DISABLE_MTP=1
B12X_MOE_FORCE_A16=1
ATTENTION_BACKEND=B12X_MLA_SPARSE
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_B12X_PAD_M_TO_POW2=1
--quantization modelopt_mixed
--moe-backend auto
--attention-backend B12X_MLA_SPARSE
--enforce-eager
--disable-custom-all-reduce
```

## Baselines

| checkpoint / mode | windows | KLD |
|---|---:|---:|
| NVFP4 W4A4 | 8 | 0.101372 |
| NVFP4 W4A16 | 8 | 0.065626 |
| NVFP4 W4A16 | 42 | 0.068724 |
| NVFP4 + `B12X_MOE_FORCE_A16=1`, BF16 reference | 1 | 0.053622 |
| config-only full-model `W4A16_NVFP4`, BF16 reference | 1 | 0.052546 |

W4A16 is the relevant production baseline here. W4A4 is diagnostic only.
The BF16-reference W1 numbers are not directly comparable to the FP8-reference
W8/W42 numbers, but they show that changing activation quantization alone is a
small effect: full-model W4A16 improved only about 2% versus the MoE-only A16
baseline.

## 2026-05-17 Local Re-Quant Export Fix

A local amax smoke re-quant initially looked worse than Luke's checkpoint. The
root cause was an export bug in quant-toolkit, not evidence that amax itself was
bad:

- `export_hf.py` tied GLM gate/up expert `weight_quantizer` amax values before
  export, but the old global pass only reached `17945` of `19200` expert pairs
  when layers were streamed/offloaded.
- The post-export metadata pass then tied all `weight_scale_2` pairs. For the
  missing pairs this made the checkpoint internally inconsistent: the packed
  FP4 weights were produced with one scale, while final `weight_scale_2` was
  overwritten to another scale.
- The fix ties GLM gate/up amax immediately before FP4 packing for each
  materialized streamed subtree. If ModelOpt has no calibrated weight amax, the
  exporter computes it from the materialized CPU weight and sets both gate/up
  quantizers before packing.

Quant-toolkit commits:

- `dd8ab31` - Fix GLM streaming gate-up FP4 scale tying
- `9f33cc9` - Compute GLM gate-up amax before FP4 export
- `dfbbc1c` - Avoid GPU amax recompute in GLM pre-export tie

Fixed smoke checkpoint:

`/root/kld/checkpoints/GLM-5.1-NVFP4-amax-smoke-prepacktie-modeloptf9d9a71-20260517`

This checkpoint is still only a small smoke-calibration export. It is useful to
prove the exporter is no longer corrupting GLM W13 scale metadata, but it is not
a production replacement for Luke's checkpoint.

| checkpoint / mode | reference | windows | KLD | positions |
|---|---|---:|---:|---:|
| Luke NVFP4 W4A16 | BF16 | 1 | 0.053622 | 2047 |
| local smoke before export fix | BF16 | 1 | 0.065833 | 2047 |
| local smoke after export fix | BF16 | 1 | 0.052286 | 2047 |
| Luke NVFP4 W4A16 | FP8 | 8 | 0.065626 | 16376 |
| local smoke after export fix | FP8 | 8 | 0.065739 | 16376 |
| Luke NVFP4 W4A16 | FP8 | 42 | 0.068724 | 85974 |
| local smoke after export fix | FP8 | 42 | 0.069499 | 85974 |

Interpretation: the exporter bug is fixed, and the bad local smoke regression is
gone. However, robust W8/W42 tests do not show a material gain over Luke's
current checkpoint. The next quality work should therefore focus on sensitive
expert layer precision, not on another blind full-model amax smoke run.

## BF16-Direct Validation

Selected expert layers were copied directly from the original BF16 checkpoint:

`/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d`

The rest of the checkpoint remained Luke's NVFP4 checkpoint. Runtime was still
`modelopt_mixed`; NVFP4 experts used B12X, while BF16 expert overrides used the
unquantized MoE path.

| selected BF16 expert layers | reference | windows | KLD | BF16 override size | note |
|---|---|---:|---:|---:|---|
| none, original NVFP4 + `B12X_MOE_FORCE_A16=1` | BF16 | 1 | 0.053622 | 0 GB | production-like baseline |
| config-only full-model `W4A16_NVFP4` | BF16 | 1 | 0.052546 | 0 GB | small activation-only gain |
| 51-62 | BF16 | 1 | 0.042128 | 216 GB | strong quality/size compromise |
| 45-47,51-62 | BF16 | 1 | 0.036973 | 270 GB | best BF16-direct result so far |
| 45-47,51-62 | FP8 | 8 | 0.046237 | 270 GB | robust W8 cross-check |

Direct BF16 experts `45-47,51-62` improved KLD by about 31% versus the
BF16-reference NVFP4 baseline. The same layer set also held up on the robust W8
FP8-reference test with KLD `0.046237`.

This points at expert weight precision as the main quality lever. It does not
look like an activation-calibration-only problem.

## Best BF16 Oracle Results

| selected BF16 expert layers | layers | BF16 override size | W8 KLD |
|---|---:|---:|---:|
| 45-47,51-62 | 15 | 270 GB | 0.043760 |
| 42-44,51-62 | 15 | 270 GB | 0.046492 |
| 51-62 | 12 | 216 GB | 0.047373 |
| 45-47,51-53,57-62 | 12 | 216 GB | 0.047636 |
| 51-53,57-62 | 9 | 162 GB | 0.051217 |
| 51-53,57-59 | 6 | 108 GB | 0.054575 |
| 51-53 | 3 | 54 GB | 0.059519 |

Relative to W4A16 baseline `0.065626`, the best tested oracle
`45-47,51-62` improves KLD by about 33%.

## Layer Sensitivity

Best three-layer contiguous blocks:

| layers | W8 KLD |
|---|---:|
| 51-53 | 0.059519 |
| 60-62 | 0.060748 |
| 57-59 | 0.060956 |
| 45-47 | 0.060972 |
| 54-56 | 0.061418 |
| 48-50 | 0.061513 |
| 63-65 | 0.061984 |
| 42-44 | 0.062177 |

Single-layer W8 validation was much weaker than block validation:

| layer | W8 KLD |
|---:|---:|
| 51 | 0.061860 |
| 53 | 0.062214 |
| 57 | 0.062744 |
| 32 | 0.063527 |
| 52 | 0.063646 |
| 46 | 0.063653 |

Conclusion: the gain is not from one magic layer. It is a multi-layer interaction
centered mainly on layers `51-62`, with `45-47` providing additional gain.

## Negative Findings

- Blind global scale tuning is not a good primary path.
- Some BF16 replacements make KLD worse or barely move it.
- Single-window W1 results are noisy and should only be used for screening.
- Directly mixing the current FP8 checkpoint into `modelopt_mixed` failed because
  the FP8 reference uses block-scale `weight_scale_inv`, while the current
  ModelOpt FP8 loader path expected a different FP8 scale format.

## Practical Interpretation

The current best evidence says the checkpoint should not be made better by
global scale tweaks. The high-leverage path is a mixed-precision checkpoint:

1. Keep most layers NVFP4.
2. Treat expert layers `51-62` as high priority for higher precision.
3. Add `45-47` if memory budget allows.
4. Test `45-47,51-62` first as the quality target.

BF16 oracle is too large for production, but it is a clean sensitivity probe.
Final production should be FP8 or a better NVFP4 re-quant for the sensitive
layers. MXFP8 should also be considered if the runtime/export path is available,
because Luke indicated it is more accurate than NVFP4.

## Next Quantization Plan

Recommended next steps:

1. Reproduce Luke's GLM-5.1 quant flow locally with real LFS calibration data,
   using `max`/amax calibration as requested by Luke. The LFS data is now
   present under `/root/quant-toolkit/data/text`.
2. Create a new quant config that preserves the current known-good exclusions:
   attention disabled, shared experts disabled, dense layers 0-2 disabled, MTP
   kept unquantized.
3. Treat the new amax checkpoint as a reproducibility/W4A4 check first. For
   W4A16, Luke indicated calibration should not be the primary factor.
4. Build a mixed checkpoint candidate:
   candidate A: layers `51-62` higher precision.
   candidate B: layers `45-47,51-62` higher precision.
5. If direct FP8 mixed export is blocked by ModelOpt format mismatch, add a
   conversion/export path for FP8 block-scaled expert weights instead of BF16.
6. Validate each candidate with KLD W8 first, then W42 only for the best one.

Do not spend more time on blind `input_scale` or `weight_scale_2` multipliers
unless used as a short diagnostic. The measured gain is layer-precision driven.
