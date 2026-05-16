# GLM-5.1 NVFP4 KLD Sensitivity Resume - 2026-05-16

## Goal

Find which GLM-5.1 MoE expert layers are most responsible for NVFP4 KLD loss
against the FP8 reference, before attempting a new quantized checkpoint.

This was not a production checkpoint build. It was a BF16 oracle sweep: selected
expert layers were dequantized from the FP8 reference to BF16 and left
unquantized, while the rest of the model stayed NVFP4.

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

W4A16 is the relevant production baseline here. W4A4 is diagnostic only.

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
layers.

## Next Quantization Plan

Recommended next steps:

1. Reproduce Luke's GLM-5.1 quant flow locally with real LFS calibration data.
   The LFS data is now present under `/root/quant-toolkit/data/text`.
2. Create a new quant config that preserves the current known-good exclusions:
   attention disabled, shared experts disabled, dense layers 0-2 disabled, MTP
   kept unquantized.
3. Build a mixed checkpoint candidate:
   candidate A: layers `51-62` higher precision.
   candidate B: layers `45-47,51-62` higher precision.
4. If direct FP8 mixed export is blocked by ModelOpt format mismatch, add a
   conversion/export path for FP8 block-scaled expert weights instead of BF16.
5. Validate each candidate with KLD W8 first, then W42 only for the best one.

Do not spend more time on blind `input_scale` or `weight_scale_2` multipliers
unless used as a short diagnostic. The measured gain is layer-precision driven.
