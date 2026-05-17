# GLM-5.1 NVFP4 KLD Sensitivity Resume - 2026-05-16

## TL;DR

The GLM-5.1 quality loss is primarily localized in MoE expert weights, not in a
global activation-scale bug. Luke's NVFP4 checkpoint in W4A16 mode is a solid
baseline, but selected expert layers are substantially more sensitive than the
rest of the model.

The best current production-shaped fix is a mixed checkpoint: keep Luke's
`GLM-5.1-NVFP4` as the base, keep most experts NVFP4/W4A16 with B12X, and
replace only selected sensitive expert layers with FP8 blockwise weight-only
(`FP8_PB_WO`) tensors from `zai-org/GLM-5.1-FP8`.

Current candidates:

| variant | FP8_PB_WO expert layers | W8 KLD | W42 KLD | model memory / rank | KV tokens |
|---|---|---:|---:|---:|---:|
| lower-memory | `51-62` | 0.049443 | 0.056918 | 45.30 GiB | 679872 |
| middle | `45-47,51-62` | 0.044499 | 0.054121 | 46.77 GiB | 654080 |
| middle+ | `42-47,51-62` | 0.043555 | 0.051939 | 48.25 GiB | 628224 |
| quality target | `42-62` | 0.040911 | 0.049504 | 49.73 GiB | 602432 |

For comparison, Luke NVFP4 W4A16 baseline was `0.065626` W8 and `0.068724`
W42. The heavy `42-62` variant improves robust W42 KLD by about 28%, but costs
about 4.43 GiB/rank more model memory than the `51-62` tradeoff.

The serving code requirement is vLLM support for `modelopt` mixed precision
MoE with `FP8_PB_WO` layer overrides. The original implementation commit was
`61eac2779 Support ModelOpt mixed FP8_PB_WO MoE`; it was ported to the active
canonical rebase branch as `166ea888a Support ModelOpt mixed FP8_PB_WO MoE`.

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

## 2026-05-17 FP8_PB_WO Mixed Checkpoint

The BF16 oracle result was converted into a production-shaped experiment:
selected expert layers are replaced with FP8 blockwise weight-only
(`FP8_PB_WO`) tensors from the official FP8 checkpoint, while the rest remains
Luke's NVFP4 W4A16 checkpoint.

This required adding ModelOpt mixed-precision MoE support for `FP8_PB_WO` in
vLLM. Canonical vLLM commit:

`61eac2779 Support ModelOpt mixed FP8_PB_WO MoE`

Checkpoint builder used for the local experiment:

`models/glm5.1/tools/make_glm51_mixed_fp8pbwo_moe.py`

Best current quality-target mixed checkpoint:

`/root/kld/checkpoints/GLM-5.1-NVFP4-mixed-fp8pbwo-L42-62-20260517`

Important implementation details:

- Old NVFP4 tensors for selected layers are physically removed from filtered
  safetensors shards. Updating `model.safetensors.index.json` alone is not
  enough because vLLM iterates safetensors files and would otherwise see
  duplicate tensor names.
- FP8 checkpoint `weight_scale_inv` keys are converted to ModelOpt
  `weight_scale` keys for the selected layers.
- NVFP4 layers still use B12X MoE. The FP8_PB_WO layers use the normal FP8 MoE
  backend selected by vLLM because B12X is an NVFP4 backend.
- The checkpoint has 100 safetensors files and 0 duplicate keys after
  validation.

Measured results:

| checkpoint / mode | reference | windows | KLD | positions | note |
|---|---|---:|---:|---:|---|
| Luke NVFP4 W4A16 | BF16 | 1 | 0.053622 | 2047 | baseline |
| FP8_PB_WO layer 51 only | BF16 | 1 | 0.050617 | 2047 | small but real gain |
| FP8_PB_WO layers 45-47,51-62 | BF16 | 1 | 0.038153 | 2047 | strong screening result |
| Luke NVFP4 W4A16 | FP8 | 8 | 0.065626 | 16376 | baseline |
| FP8_PB_WO layers 51-53,57-62 | FP8 | 8 | 0.051694 | 16376 | too much quality loss |
| FP8_PB_WO layers 45-47,51-53,57-62 | FP8 | 8 | 0.047075 | 16376 | short validation looked promising |
| FP8_PB_WO layers 51-62 | FP8 | 8 | 0.049443 | 16376 | cheaper tradeoff |
| FP8_PB_WO layers 45-47,51-62 | FP8 | 8 | 0.044499 | 16376 | robust short validation |
| FP8_PB_WO layers 42-47,51-62 | FP8 | 8 | 0.043555 | 16376 | best W8, heavier quality target |
| FP8_PB_WO layers 42-62 | FP8 | 8 | 0.040911 | 16376 | best W8, heavy quality target |
| Luke NVFP4 W4A16 | FP8 | 42 | 0.068724 | 85974 | baseline |
| FP8_PB_WO layers 51-62 | FP8 | 42 | 0.056918 | 85974 | cheaper tradeoff |
| FP8_PB_WO layers 45-47,51-53,57-62 | FP8 | 42 | 0.057432 | 85974 | worse than 51-62 at same runtime size |
| FP8_PB_WO layers 45-47,51-62 | FP8 | 42 | 0.054121 | 85974 | robust validation |
| FP8_PB_WO layers 42-47,51-62 | FP8 | 42 | 0.051939 | 85974 | best robust KLD so far |
| FP8_PB_WO layers 42-62 | FP8 | 42 | 0.049504 | 85974 | best robust KLD, heavy |

Runtime footprint:

| selected FP8_PB_WO layers | local dir size | model memory / rank | KV cache tokens | max 4096-token concurrency |
|---|---:|---:|---:|---:|
| 51-53,57-62 | 96 GB | 43.82 GiB | 705728 | 172.30x |
| 51-62 | 118 GB | 45.30 GiB | 679872 | 165.98x |
| 45-47,51-53,57-62 | 128 GB | 45.30 GiB | 679872 | 165.98x |
| 45-47,51-62 | 150 GB | 46.77 GiB | 654080 | 159.69x |
| 42-47,51-62 | 177 GB | 48.25 GiB | 628224 | 153.38x |
| 42-62 | 198 GB | 49.73 GiB | 602432 | 147.08x |

Interpretation: this is the first production-shaped result that matches the
BF16 oracle direction. The quality gap is primarily from selected expert weight
precision. Full-model A16 activation handling alone is not the high-leverage
fix. The `51-62` candidate is the better memory/KV tradeoff. Adding `45-47`
gives a strong middle quality target. Adding `42-47` improves quality further.
The current best robust KLD is the heavy `42-62` checkpoint: `0.049504` W42,
about 28% lower than the NVFP4 W4A16 W42 baseline `0.068724`, but it costs
about 2.96 GiB/rank more model memory than `45-47,51-62` and about 4.43 GiB/rank
more than `51-62`.

Two smaller variants were tested as quality/size probes. `51-53,57-62` saves
memory but gives up too much KLD. `45-47,51-53,57-62` looked better on W8, but
the robust W42 run was slightly worse than `51-62` while using the same runtime
memory/KV footprint. It is therefore not a preferred candidate.

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
- Directly mixing the current FP8 checkpoint into `modelopt_mixed` used to be
  blocked because the FP8 reference uses block-scale `weight_scale_inv`. This is
  now handled for MoE layers through the `FP8_PB_WO` conversion path above.

## Practical Interpretation

The current best evidence says the checkpoint should not be made better by
global scale tweaks. The high-leverage path is a mixed-precision checkpoint:

1. Keep most layers NVFP4.
2. Treat expert layers `51-62` as high priority for higher precision.
3. Add `45-47` if memory budget allows.
4. Test `45-47,51-62` first as the quality target.

BF16 oracle is too large for production, but it was a clean sensitivity probe.
The current production-shaped direction is mixed NVFP4 W4A16 plus FP8_PB_WO
expert layers for `45-47,51-62`. MXFP8 should still be considered if the
runtime/export path is available, because Luke indicated it is more accurate
than NVFP4.

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
4. Keep `42-62` as the current best quality-target FP8_PB_WO candidate.
5. Keep `51-62` as the main lower-memory tradeoff candidate. The smaller
   `51-53,57-62` and `45-47,51-53,57-62` probes did not beat it.
6. Keep `45-47,51-62` and `42-47,51-62` as middle tradeoffs if the full `42-62`
   set costs too much memory/KV in serving.
7. If MXFP8 export/runtime support becomes available, repeat the same selected
   layer sets with MXFP8 and compare KLD and memory footprint.

Do not spend more time on blind `input_scale` or `weight_scale_2` multipliers
unless used as a short diagnostic. The measured gain is layer-precision driven.

## Reproduction Map

Use these locations to reconstruct the work:

| artifact | path / repo | purpose |
|---|---|---|
| canonical vLLM git | `/root/vllm/src/vllm` | source for KLD runner and mixed `FP8_PB_WO` runtime support |
| vLLM support commit | `166ea888a Support ModelOpt mixed FP8_PB_WO MoE` on `codex/glm51-kimi-canonical-rebase-test-20260514` | adds ModelOpt mixed MoE `FP8_PB_WO` loading/runtime |
| mixed checkpoint builder | `models/glm5.1/tools/make_glm51_mixed_fp8pbwo_moe.py` | local materialized checkpoint builder |
| HF streaming uploader | `/root/kld/upload_glm51_mixed_fp8pbwo_to_hf.py` | public HF uploader that chains server-side repo duplicates and uploads only changed shards |
| KLD runner | `examples/offline_inference/score_mode_kld.py` in vLLM | two-phase reference-logit and KLD scoring runner |
| KLD work dir | `/root/kld` | logs, reference logits, local variants |
| FP8 reference logits W8 | `/root/kld/current_cf2070d_b12xc929144_fp8_ref_wikitext_ctx2048_s512_w8_b12xmlasparse` | reference logits for W8 KLD |
| FP8 reference logits W42 | `/root/kld/current_cf2070d_b12xc929144_fp8_ref_wikitext_ctx2048_s512_w42_b12xmlasparse` | reference logits for W42 KLD |
| BF16 reference logits W1 | `/root/kld/glm51_bf16_ref_vllm_wikitext_ctx2048_s512_w1_b12xmlasparse_20260516` | BF16 W1 screening reference |
| oracle sweep results | `/root/kld/oracle_sweep_results.tsv` | BF16/BF16-direct layer sensitivity sweep index |
| oracle sweep logs | `/root/kld/oracle_sweep_logs/` | per-run BF16 oracle logs |
| quant-toolkit patch repo | `/root/quant-toolkit` | local export fixes for GLM gate/up FP4 scale tying |

Source checkpoints used:

| role | checkpoint |
|---|---|
| NVFP4 base | `/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/18175c33fa74faa199b5d462e02d12b9d3399295` |
| FP8 reference/source | `/root/.cache/huggingface/hub/models--zai-org--GLM-5.1-FP8/snapshots/a92f8155fe8574288534d41a6dbc1d72888ab2da` |
| BF16 source | `/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d` |

## Exact KLD Method

All robust KLD numbers in this document use the vLLM score-mode KLD runner with
WikiText text, `context_length=2048`, `stride=512`, TP=8, eager mode, FP8 KV
cache, B12X sparse MLA attention, and one sequence per batch. The window count
is the number of sliding windows scored:

| window count | positions | purpose |
|---:|---:|---|
| 1 | 2047 | quick smoke/screening |
| 8 | 16376 | short robust check |
| 42 | 85974 | robust comparison used for final candidates |

Common runtime environment:

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
export SAFETENSORS_FAST_GPU=1
export B12X_MOE_FORCE_A16=1
export MOE_BACKEND=b12x
export ATTENTION_BACKEND=B12X_MLA_SPARSE
export GLM51_PATTERN='FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS'
```

Generate FP8 reference logits once:

```bash
cd /root/vllm/src/vllm
python3 examples/offline_inference/score_mode_kld.py \
  --reference-only \
  --reference-model /root/.cache/huggingface/hub/models--zai-org--GLM-5.1-FP8/snapshots/a92f8155fe8574288534d41a6dbc1d72888ab2da \
  --reference-logits /root/kld/current_cf2070d_b12xc929144_fp8_ref_wikitext_ctx2048_s512_w42_b12xmlasparse \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 \
  --stride 512 \
  --max-windows 42 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --cpu-offload-gb 16 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --attention-backend B12X_MLA_SPARSE \
  --hf-overrides "{\"index_topk_pattern\":\"${GLM51_PATTERN}\"}" \
  --trust-remote-code \
  --enforce-eager \
  --disable-custom-all-reduce
```

Measure Luke NVFP4 W4A16 against saved FP8 reference logits:

```bash
cd /root/vllm/src/vllm
python3 examples/offline_inference/score_mode_kld.py \
  --model /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/18175c33fa74faa199b5d462e02d12b9d3399295 \
  --reference-logits /root/kld/current_cf2070d_b12xc929144_fp8_ref_wikitext_ctx2048_s512_w42_b12xmlasparse \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 \
  --stride 512 \
  --max-windows 42 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --cpu-offload-gb 16 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --quantization modelopt_fp4 \
  --moe-backend b12x \
  --attention-backend B12X_MLA_SPARSE \
  --hf-overrides "{\"index_topk_pattern\":\"${GLM51_PATTERN}\"}" \
  --trust-remote-code \
  --enforce-eager \
  --disable-custom-all-reduce
```

Measure a mixed `FP8_PB_WO` variant:

```bash
cd /root/vllm/src/vllm
python3 examples/offline_inference/score_mode_kld.py \
  --model /root/kld/checkpoints/GLM-5.1-NVFP4-mixed-fp8pbwo-L42-62-20260517 \
  --reference-logits /root/kld/current_cf2070d_b12xc929144_fp8_ref_wikitext_ctx2048_s512_w42_b12xmlasparse \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --context-length 2048 \
  --stride 512 \
  --max-windows 42 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --cpu-offload-gb 16 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --quantization modelopt_mixed \
  --moe-backend b12x \
  --attention-backend B12X_MLA_SPARSE \
  --hf-overrides "{\"index_topk_pattern\":\"${GLM51_PATTERN}\"}" \
  --trust-remote-code \
  --enforce-eager \
  --disable-custom-all-reduce
```

Notes:

- `modelopt_fp4` is used for Luke's pure NVFP4 checkpoint.
- `modelopt_mixed` is used for mixed NVFP4 plus `FP8_PB_WO`.
- MTP is intentionally off for KLD. This isolates base model logits from draft
  model behavior.
- `--enforce-eager` was used to make the score-mode tests reproducible and to
  avoid unrelated CUDA graph/capture effects during quality measurement.

## Mixed FP8_PB_WO Checkpoint Build

The local materialized builder is:

`models/glm5.1/tools/make_glm51_mixed_fp8pbwo_moe.py`

Example:

```bash
python3 /root/rtx6kpro/models/glm5.1/tools/make_glm51_mixed_fp8pbwo_moe.py \
  --nvfp4-source /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/18175c33fa74faa199b5d462e02d12b9d3399295 \
  --fp8-source /root/.cache/huggingface/hub/models--zai-org--GLM-5.1-FP8/snapshots/a92f8155fe8574288534d41a6dbc1d72888ab2da \
  --dest /root/kld/checkpoints/GLM-5.1-NVFP4-mixed-fp8pbwo-L42-62-20260517 \
  --layers 42-62
```

What the builder does:

- Symlinks unchanged non-safetensors metadata from Luke's NVFP4 base.
- Rewrites every NVFP4 safetensors shard that contains selected expert tensors,
  dropping only the selected expert keys.
- Writes one new `model-mixed-fp8pbwo-layerXX.safetensors` file per selected
  layer.
- Converts FP8 source `weight_scale_inv` keys into ModelOpt `weight_scale`
  keys for the mixed checkpoint.
- Writes `config.json` with `quantization_config.quant_algo=MIXED_PRECISION`
  and per-layer `quantized_layers` entries.
- Writes a new `model.safetensors.index.json` so every tensor has exactly one
  owner file.

Do not only edit the index. The original NVFP4 keys must be physically removed
from changed shards, because vLLM can iterate actual safetensors keys and would
otherwise see duplicate tensor names.

## Public HF Upload Procedure

The upload is intentionally not a simple `upload_folder`, because unchanged
base files are too large. The current uploader is:

`/root/kld/upload_glm51_mixed_fp8pbwo_to_hf.py`

It creates public repos by server-side duplicating the previous repo, then
uploads only files that changed:

1. `l51-62` duplicates `lukealonso/GLM-5.1-NVFP4`.
2. `l45-47-l51-62` duplicates `l51-62` and adds layers `45-47`.
3. `l42-47-l51-62` duplicates `l45-47-l51-62` and adds layers `42-44`.
4. `l42-62` duplicates `l42-47-l51-62` and adds layers `48-50`.

Run:

```bash
HF_HUB_DISABLE_PROGRESS_BARS=1 \
HF_HUB_ENABLE_HF_TRANSFER=1 \
python3 /root/kld/upload_glm51_mixed_fp8pbwo_to_hf.py
```

The script uses the cached Hugging Face token if `HF_TOKEN` is not set. It is
restartable at file-commit granularity: already committed remote files are
skipped, while metadata files are force-updated.

Target public repos:

| variant | repo |
|---|---|
| `51-62` | `festr2/glm51-nvfp4-w4a16-fp8pbwo-l51-62-20260517` |
| `45-47,51-62` | `festr2/glm51-nvfp4-w4a16-fp8pbwo-l45-47-l51-62-20260517` |
| `42-47,51-62` | `festr2/glm51-nvfp4-w4a16-fp8pbwo-l42-47-l51-62-20260517` |
| `42-62` | `festr2/glm51-nvfp4-w4a16-fp8pbwo-l42-62-20260517` |

Upload verification on 2026-05-17:

| variant | files | expected layer shards | missing layer shards | extra layer shards | config/index check |
|---|---:|---:|---:|---:|---|
| `51-62` | 105 | 12 | 0 | 0 | pass |
| `45-47,51-62` | 108 | 15 | 0 | 0 | pass |
| `42-47,51-62` | 111 | 18 | 0 | 0 | pass |
| `42-62` | 114 | 21 | 0 | 0 | pass |

The verification checked `config.json` for `MIXED_PRECISION`, checked every
selected layer's `quantized_layers` entry for `FP8_PB_WO`, checked all other
expert layers for `NVFP4`, and sampled expert `0` and `255` weight/scale index
entries for every selected layer.

## Source Code State

vLLM:

- Original mixed `FP8_PB_WO` support was implemented in
  `/root/vllm/src/vllm` on branch
  `codex/glm51-kimi-canonical-a16-upstream-20260511`.
- The key commit is `61eac2779 Support ModelOpt mixed FP8_PB_WO MoE`.
- This commit was cherry-picked onto the current canonical rebase branch as
  `166ea888a Support ModelOpt mixed FP8_PB_WO MoE`.
- Pushed branch:
  `https://github.com/voipmonitor/vllm/commits/codex/glm51-kimi-canonical-rebase-test-20260514/`
- Do not merge the whole old `codex/glm51-kimi-canonical-a16-upstream-20260511`
  branch into the rebase branch. It diverges from the rebase branch and
  contains older runtime state.

quant-toolkit:

- Local repo: `/root/quant-toolkit`
- Pushed branch:
  `https://github.com/local-inference-lab/quant-toolkit/tree/codex/glm51-nvfp4-export-fixes-20260517`
- Relevant commits:
  - `dd8ab31 Fix GLM streaming gate-up FP4 scale tying`
  - `9f33cc9 Compute GLM gate-up amax before FP4 export`
  - `dfbbc1c Avoid GPU amax recompute in GLM pre-export tie`
- These commits fixed local export reproducibility, but did not produce a
  better full-model NVFP4 checkpoint than Luke's current checkpoint on robust
  W8/W42 tests.

## Operational Recommendations

- Use `51-62` first if KV capacity matters more than absolute KLD.
- Use `45-47,51-62` or `42-47,51-62` as the practical quality middle ground.
- Use `42-62` as the current best quality target when the extra model memory is
  acceptable.
- Keep W4A16 for NVFP4 serving (`B12X_MOE_FORCE_A16=1`) when evaluating these
  checkpoints. W4A4 remains useful as a diagnostic baseline but is not the
  quality target.
- Do not treat FP8 reference logits as perfect BF16 truth. They are practical
  enough for robust comparisons on these GPUs, but BF16 W1 was used as a
  separate cross-check where possible.
- If MXFP8 support becomes available in the runtime/export path, repeat the
  same layer sets. Luke indicated MXFP8 should be more accurate than NVFP4.
