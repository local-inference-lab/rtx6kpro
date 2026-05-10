# MiMo-V2.5-Pro TP=8 quantization pipeline

Date: 2026-05-10

This page records the exact MiMo-V2.5-Pro TP=8 quantization path used to produce the public NVFP4/MXFP8 checkpoint.

## Source checkpoint and TP layout

Source model:

```text
XiaomiMiMo/MiMo-V2.5-Pro
```

Local source snapshot used during the run:

```text
/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro/snapshots/a75207db63de3c320950fe6fcfa9ff60f341b7a2
```

The upstream Pro fused QKV tensors are TP-packed with TP=8. This is the critical detail that differs from older MiMo 2.5 scripts that defaulted to TP=4.

Evidence from the shape check:

| Item | Value |
|---|---|
| Tensor | `model.layers.0.self_attn.qkv_proj.weight` |
| Weight shape | `[27136, 6144]` |
| FP8 scale shape | `[216, 48]` |
| TP=8 expected scale rows | `216` |
| TP=4 expected scale rows | `212` |
| TP=16 expected scale rows | `224` |

Conclusion: every Pro dequantize/deinterleave command must use:

```bash
--mimo-qkv-tp-size 8
```

## 1. Dequantize upstream FP8 to BF16

Wrapper:

```bash
scripts/dequantize_mimo_v25_pro.sh
```

Important defaults in the wrapper:

```bash
MODEL_ID=XiaomiMiMo/MiMo-V2.5-Pro
OUTPUT_DIR=/data/models/MiMo-V2.5-Pro-BF16-qkv-deinterleaved
WORKERS=8
OUTPUT_SHARD_SIZE_GIB=8
MIMO_QKV_TP_SIZE=8
```

Expanded command:

```bash
python3 tools/dequantize_fp8.py XiaomiMiMo/MiMo-V2.5-Pro \
  --output-dir /data/models/MiMo-V2.5-Pro-BF16-qkv-deinterleaved \
  --workers 8 \
  --output-shard-size-gib 8 \
  --mimo-deinterleave-qkv \
  --mimo-qkv-tp-size 8
```

Result:

- output: `/data/models/MiMo-V2.5-Pro-BF16-qkv-deinterleaved`
- output size: about 1.9 TB
- safetensors index present
- total files processed: 34
- total FP8 weights dequantized: 79,573
- total weights: 80,008

## 2. BF16 smoke validation

The useful BF16 gate was a 256-token test with enough input-token budget. An earlier one-token test only proved loading, and an earlier 256-token run with `--max-input-tokens 128` was invalid because the MiMo chat template was truncated.

Valid smoke command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
.venv/bin/python tools/mimo_forward_smoke.py \
  'What is the capital of France?' \
  --model mimo_v25_pro \
  --model-id /data/models/MiMo-V2.5-Pro-BF16-qkv-deinterleaved \
  --streaming \
  --gpu-capacity 90GiB \
  --cpu-capacity 1090GiB \
  --max-input-tokens 512 \
  --max-new-tokens 256 \
  --decode-mode kv-cache \
  --disable-thinking
```

Observed answer:

```text
The capital of France is **Paris**
```

This is the valid BF16 coherence gate.

## 3. Quantize routed experts to NVFP4

Base NVFP4 output:

```text
/data/models/MiMo-V2.5-Pro-NVFP4
```

Main command:

```bash
PYTHONUNBUFFERED=1 \
GPU_CAPACITY=84GiB \
AMAX_CHECKPOINT_INTERVAL=8 \
CLEANUP_INTERVAL=8 \
RESUME_AMAX=/data/models/MiMo-V2.5-Pro-NVFP4/amax_checkpoint.safetensors \
RESUME_BATCH=32 \
scripts/quantize_mimo_v25_pro.sh
```

Expanded command:

```bash
python3 quantize.py \
  --model mimo_v25_pro \
  --model-id /data/models/MiMo-V2.5-Pro-BF16-qkv-deinterleaved \
  --export-dir /data/models/MiMo-V2.5-Pro-NVFP4 \
  --calib-config configs/calib_mimo_v25_pro.toml \
  --save-amax /data/models/MiMo-V2.5-Pro-NVFP4/amax.safetensors \
  --batch-tokens 16384 \
  --streaming \
  --cpu-capacity 1090GiB \
  --gpu-capacity 84GiB \
  --amax-checkpoint-interval 8 \
  --cleanup-interval 8 \
  --floor-amaxes \
  --resume-amax /data/models/MiMo-V2.5-Pro-NVFP4/amax_checkpoint.safetensors \
  --resume-batch 32
```

Calibration config:

| Dataset | Batch size | Max length | Limit |
|---|---:|---:|---:|
| `text/agentic_coding_calib_v3.jsonl` | 16 | 1024 | 2048 |
| `text/diverse_calib.jsonl` | 16 | 1024 | 512 |
| `text/deep_calib.jsonl` | 1 | 4096 | 128 |

Total calibration plan: 288 batches.

Runtime changes during quantization:

- increased streaming `GPU_CAPACITY` from `70GiB` to `84GiB`
- increased amax checkpoint interval to 8
- increased cleanup interval to 8
- did not reduce calibration data, tensor formats, or quantization settings

Quantization completed and exported:

```text
/data/models/MiMo-V2.5-Pro-NVFP4
```

## 4. MTP fixup

The generic export emitted:

```text
No source index found, skipping MTP merge.
```

That was not enough for MiMo Pro serving with MTP/EAGLE. The separate fixup script was run:

```bash
scripts/fixup_mimo_v25_pro_nvfp4.sh
```

Expanded command:

```bash
python3 tools/fixup_mimo_nvfp4_export.py \
  --source-dir /data/models/MiMo-V2.5-Pro-BF16-qkv-deinterleaved \
  --export-dir /data/models/MiMo-V2.5-Pro-NVFP4
```

This copies draft-only MTP tensors into `model-mtp.safetensors` and updates the index. SGLang target-model loading later logs that these draft-only MTP weights are skipped by the target runner and loaded by the draft model runner, which is expected.

## 5. Convert MiMo fused QKV attention FP8 to MXFP8

Luke's correction was that the attention path should be TP-interleaved FP8 to MXFP8. The older `FP8_PB_WO` chimera was not the final path.

Wrapper:

```bash
env MODEL_DIR=/root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro/snapshots/a75207db63de3c320950fe6fcfa9ff60f341b7a2 \
  MIMO_QKV_TP_SIZE=8 \
  OUTPUT_FORMAT=mxfp8 \
  ON_INEXACT=requantize \
  OUTPUT_DIR=/data/models/MiMo-V2.5-Pro-MXFP8-qkv-deinterleaved \
  scripts/deinterleave_mimo_v25_pro_fp8_attn.sh
```

Expanded command:

```bash
python3 tools/deinterleave_mimo_fp8_qkv.py XiaomiMiMo/MiMo-V2.5-Pro \
  --model-dir /root/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro/snapshots/a75207db63de3c320950fe6fcfa9ff60f341b7a2 \
  --output-dir /data/models/MiMo-V2.5-Pro-MXFP8-qkv-deinterleaved \
  --mimo-qkv-tp-size 8 \
  --output-format mxfp8 \
  --on-inexact requantize \
  --unchanged-shards symlink \
  --force
```

Output:

```text
/data/models/MiMo-V2.5-Pro-MXFP8-qkv-deinterleaved
```

## 6. Build the mixed NVFP4/MXFP8 checkpoint

Wrapper:

```bash
env BASE=/data/models/MiMo-V2.5-Pro-NVFP4 \
  FP8_ATTN=/data/models/MiMo-V2.5-Pro-MXFP8-qkv-deinterleaved \
  OUTPUT=/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn \
  scripts/build_mimo_v25_pro_chimera.sh
```

Expanded commands:

```bash
python3 tools/build_mimo_nvfp4_fp8_attn_chimera.py \
  --base /data/models/MiMo-V2.5-Pro-NVFP4 \
  --fp8-attn /data/models/MiMo-V2.5-Pro-MXFP8-qkv-deinterleaved \
  --output /data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn \
  --attention-format mxfp8 \
  --unchanged-shards symlink \
  --force

python3 tools/write_mimo_chimera_mixed_quant_config.py \
  /data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn
```

Post-process summary:

- main QKV tensors replaced: 70
- MTP QKV tensors replaced: 3
- total QKV replacements: 73
- `FP8_PB_WO` QKV modules: 0
- MXFP8 QKV modules: 73
- total `quantized_layers` entries: 79,561

Final target checkpoint:

```text
/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn
```

MTP serving overlay:

```text
/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-BF16-MTP
```

The overlay includes BF16 MTP draft weights and symlinks the target-model shards/configs back to the mixed checkpoint.

## Size and upload

Symlink-dereferenced sizes:

| Path | Bytes | GiB |
|---|---:|---:|
| `/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn` | 596,967,489,918 | 555.969 |
| `/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-BF16-MTP` | 597,452,029,130 | 556.421 |

The smaller apparent local footprint is due to shard symlinks back into the base NVFP4 export.

Public Hugging Face upload:

```text
festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8
```

Upload completed at:

```text
2026-05-09T23:12:22.711028+00:00
```
