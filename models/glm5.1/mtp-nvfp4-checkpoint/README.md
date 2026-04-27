# GLM-5.1 MTP Routed MoE NVFP4 Checkpoint

This note records how the local GLM-5.1 hybrid checkpoint was created:

```text
/mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink
```

The goal was to keep Luke Alonso's `GLM-5.1-NVFP4` checkpoint as the base model
and quantize only the appended MTP/NextN routed MoE experts from BF16 to
ModelOpt NVFP4.

No model weights are stored in this repository.  The scripts here recreate the
checkpoint as a symlink tree plus a small set of local MTP shard replacements.

## Summary

Input checkpoint:

```text
lukealonso/GLM-5.1-NVFP4
snapshot: 1b9f53ee1d11fcbb7ecefed115359afba02f104f
local path: /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/1b9f53ee1d11fcbb7ecefed115359afba02f104f
```

The checkpoint has:

```text
num_hidden_layers: 78
num_nextn_predict_layers: 1
MTP layer: model.layers.78
```

What was changed:

```text
model.layers.78.mlp.experts.*.{gate_proj,up_proj,down_proj}.weight
```

Only routed MTP experts were converted.  MTP shared experts, attention, norms,
embedding and lm head remain as in the base checkpoint.  In particular:

```text
model.layers.78.mlp.shared_experts.* stays BF16
```

The resulting routed expert tensor layout is:

```text
weight         uint8            shape example: (2048, 3072)
weight_scale   float8_e4m3fn    shape example: (2048, 384)
weight_scale_2 float32          scalar
input_scale    float32          scalar
```

The current checkpoint used the layer-77 routed expert `input_scale` tensors as
the proxy activation scale source:

```text
model.layers.77.mlp.experts.{expert}.{proj}.input_scale
```

An amax-based path is also supported by the builder, but the current measured
checkpoint was the proxy-scale variant because it was the best local result at
the time of this note.

## Files

```text
scripts/build_glm51_mtp_nvfp4_routed_checkpoint.py
scripts/merge_mtp_amax_safetensors.py
scripts/make_sglang_runtime_checkpoint.py
scripts/verify_glm51_mtp_nvfp4_checkpoint.py
patches/sglang_nextn_nvfp4_mtp_loader.patch
```

The main builder needs CUDA and ModelOpt because it calls:

```python
modelopt.torch.export.quant_utils.NVFP4QTensor.quantize
```

## Build

Install the needed Python packages in an environment that already has a CUDA
PyTorch build:

```bash
pip install safetensors nvidia-modelopt
```

Create the vLLM-oriented hybrid checkpoint:

```bash
python3 scripts/build_glm51_mtp_nvfp4_routed_checkpoint.py \
  --base-dir /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/1b9f53ee1d11fcbb7ecefed115359afba02f104f \
  --out-dir /mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink \
  --devices 0,1,2,3,4,5,6,7 \
  --workers 8
```

The output is mostly symlinks to the base checkpoint.  The only materialized
local safetensors are the filtered original MTP shard and the new MTP NVFP4
routed expert shards.  In the local run this added about 5.6 GiB of materialized
files while the full referenced checkpoint was about 421 GiB.

Expected builder result:

```text
converted expert projections: 768
```

## Optional Amax Path

The builder supports calibrated input scales:

```bash
python3 scripts/merge_mtp_amax_safetensors.py \
  /mnt/glm51_mtp_amax_calib_20260427 \
  --output /mnt/glm51_mtp_amax_real_merged.safetensors

python3 scripts/build_glm51_mtp_nvfp4_routed_checkpoint.py \
  --base-dir /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/1b9f53ee1d11fcbb7ecefed115359afba02f104f \
  --out-dir /mnt/glm51-luke-nvfp4-mtp-nvfp4routed-calibrated-symlink \
  --amax-file /mnt/glm51_mtp_amax_real_merged.safetensors \
  --devices 0,1,2,3,4,5,6,7 \
  --workers 8
```

The expected amax file contains per-expert tensors such as:

```text
model.layers.78.mlp.experts.{expert}.gate_proj.input_amax
model.layers.78.mlp.experts.{expert}.up_proj.input_amax
model.layers.78.mlp.experts.{expert}.down_proj.input_amax
```

or compact tensors:

```text
model.layers.78.mlp.experts.w13_input_amax
model.layers.78.mlp.experts.w2_input_amax
```

The builder converts amax to ModelOpt input scale with:

```text
input_scale = 1 / (input_amax * 448)
```

## SGLang Runtime View

For the SGLang test stack used here, the same tensors were exposed through a
second symlink view:

```bash
python3 scripts/make_sglang_runtime_checkpoint.py \
  --hybrid-dir /mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink \
  --base-dir /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/snapshots/1b9f53ee1d11fcbb7ecefed115359afba02f104f \
  --out-dir /mnt/glm51-luke-nvfp4-mtp-nvfp4routed-sglang-symlink
```

This keeps all weights and `model.safetensors.index.json` from the hybrid
checkpoint, but copies `config.json` from the base Luke checkpoint.  That matches
the SGLang loader behavior tested locally.  A SGLang source patch was still
needed so `DeepseekModelNextN` does not blindly disable `modelopt_fp4` for the
MTP routed MoE when layer-78 FP4 scales are present.

The minimal loader patch used for this is stored in:

```text
patches/sglang_nextn_nvfp4_mtp_loader.patch
```

## Verify

Run:

```bash
python3 scripts/verify_glm51_mtp_nvfp4_checkpoint.py \
  /mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink
```

Expected checks:

```text
counts: {'weight': 768, 'weight_scale': 768, 'weight_scale_2': 768, 'input_scale': 768}
model.layers.78.mlp.experts.0.gate_proj.weight torch.uint8 (2048, 3072)
model.layers.78.mlp.experts.0.gate_proj.weight_scale torch.float8_e4m3fn (2048, 384)
model.layers.78.mlp.experts.0.gate_proj.weight_scale_2 torch.float32 ()
model.layers.78.mlp.experts.0.gate_proj.input_scale torch.float32 ()
model.layers.78.mlp.shared_experts.gate_proj.weight torch.bfloat16 (2048, 6144)
PASS
```

## Measured Acceptance

These are local GLM-5.1 MTP=3 measurements using the same short generation
acceptance test.  The SGLang result used the SGLang runtime view above plus NSA
and b12x.

| Runtime | Checkpoint | accepted/draft | accepted/step | pos0 | pos1 | pos2 | tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| SGLang | NVFP4 MTP routed, b12x | 68.63% | 2.059 | 85.32% | 68.78% | 51.80% | 106.39 |
| vLLM | NVFP4 MTP routed current | 67.32% | 2.020 | 86.41% | 69.23% | 46.33% | 97.00 |
| vLLM | NVFP4 MTP routed old | 67.12% | 2.014 | 86.03% | 68.64% | 46.69% | 97.29 |
| vLLM | BF16 MTP | 63.44% | 1.903 | 89.31% | 66.33% | 34.69% | 92.83 |
| vLLM | FP8 MTP | 60.15% | 1.804 | 86.81% | 62.68% | 30.95% | 90.12 |

Local result files:

```text
/tmp/mtp_acceptance_glm51/bf16_result.json
/tmp/mtp_acceptance_glm51/fp8_result.json
/tmp/mtp_acceptance_glm51/nvfp4_result.json
/tmp/mtp_acceptance_glm51/nvfp4_singlealpha_current_result.json
/tmp/sglang_glm_acceptance_20260427/sglang_nvfp4_mtp_chattemplate_b12x_result.json
```

## Notes

The checkpoint is intentionally hybrid.  Do not assume every tensor in
`model.layers.78` is NVFP4.

The builder removes the original BF16 routed expert weights from
`model-00084-of-00084.safetensors` in the output tree.  This avoids duplicate
tensor collisions in loaders that scan all `.safetensors` files rather than only
following `model.safetensors.index.json`.

Gate and up projections are quantized with a paired `weight_scale_2` candidate.
The best local runtime behavior was obtained by keeping the generated per-proj
scales in the checkpoint and using the runtime path that handles fused W13
consistently.
