# Hybrid NVFP4 Assembly: BF16 Shared Expert

Replace the NVFP4-quantized shared expert in NVIDIA's checkpoint with full-precision BF16 weights from the original model. The shared expert runs on **every token** (unlike routed experts where only 10/512 activate), so its precision has outsized impact on output quality.

## What Changes vs Pure NVIDIA NVFP4

Only the shared expert weights change. Everything else is identical to `nvidia/Qwen3.5-397B-A17B-NVFP4`.

| Component | NVIDIA NVFP4 | Hybrid | Changed? |
|-----------|-------------|--------|----------|
| Routed experts (512/layer, 60 layers) | NVFP4 (uint8 packed) | NVFP4 (uint8 packed) | No |
| **Shared expert (1/layer, 60 layers)** | **NVFP4 (uint8 packed)** | **BF16 (full precision)** | **Yes** |
| Router / gate | BF16 | BF16 | No |
| Self-attention (15 layers) | BF16 | BF16 | No |
| Linear attention / GatedDeltaNet (45 layers) | BF16 | BF16 | No |
| KV cache scales (k_scale, v_scale) | FP8 | FP8 | No |
| Layer norms | BF16 | BF16 | No |
| Embeddings + lm_head | BF16 | BF16 | No |

**Size impact:** 234 GB vs 233 GB (+1 GB). The shared expert is only ~0.19% of total model parameters.

## NVIDIA NVFP4 Quantization Breakdown

For reference, here's what NVIDIA quantized and what they left in BF16:

**NVFP4 (quantized):**
- Routed expert weights — gate_proj, up_proj, down_proj x 512 experts x 60 layers (370,176 tensors)
- Shared expert weights — gate_proj, up_proj, down_proj x 60 layers (723 tensors including scales)

**FP8:**
- KV cache scales (k_scale, v_scale) — 30 tensors for the 15 full-attention layers

**BF16 (untouched by NVIDIA):**
- Router / gate (61 tensors)
- Shared expert gate (61 tensors)
- Self-attention layers (126 tensors)
- Linear attention / GatedDeltaNet layers (405 tensors)
- Layer norms (236 tensors)
- Embeddings (4 tensors)
- lm_head (1 tensor)

## Prerequisites

- HuggingFace access to both models (accept gated model agreements):
  - `nvidia/Qwen3.5-397B-A17B-NVFP4` (~233 GB)
  - `Qwen/Qwen3.5-397B-A17B` (~752 GB BF16)
- Both models downloaded to HF cache (`~/.cache/huggingface/hub/`)
- Python packages: `torch`, `safetensors`, `huggingface_hub`
- ~250 GB free disk space for output
- SGLang with CUDA 13 (e.g., `lmsysorg/sglang:dev-cu13`)

## Step 1: Assemble the Hybrid Checkpoint

Save as `assemble_hybrid.py` and run:

```python
#!/usr/bin/env python3
"""Assemble hybrid NVFP4 model: shared expert BF16, everything else from NVIDIA NVFP4."""

import json
import os
import re
import shutil
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

NVFP4_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"
BF16_MODEL = "Qwen/Qwen3.5-397B-A17B"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = Path("./hybrid-nvfp4")


def classify_key(key: str) -> str:
    """Classify tensor key to determine source model.

    Returns: 'nvfp4_expert', 'nvfp4_kv', 'bf16_shared', 'bf16'
    """
    if key.endswith(".k_scale") or key.endswith(".v_scale"):
        return "nvfp4_kv"
    if ".mlp.experts." in key and ".shared_expert" not in key:
        return "nvfp4_expert"
    if ".mlp.shared_expert." in key and ".shared_expert_gate" not in key:
        return "bf16_shared"
    return "bf16"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load weight map indices
    logger.info("Loading weight map indices...")
    nvfp4_idx = hf_hub_download(NVFP4_MODEL, "model.safetensors.index.json", token=HF_TOKEN)
    bf16_idx = hf_hub_download(BF16_MODEL, "model.safetensors.index.json", token=HF_TOKEN)
    with open(nvfp4_idx) as f:
        nvfp4_wm = json.load(f)["weight_map"]
    with open(bf16_idx) as f:
        bf16_wm = json.load(f)["weight_map"]
    logger.info("NVFP4 keys: %d, BF16 keys: %d", len(nvfp4_wm), len(bf16_wm))

    # Plan: which tensors from which source
    plan = {}
    for key in nvfp4_wm:
        cat = classify_key(key)
        if cat in ("nvfp4_expert", "nvfp4_kv"):
            # Routed experts + KV scales from NVIDIA
            plan[key] = {"source": "nvfp4", "file": nvfp4_wm[key], "category": cat}
        elif cat == "bf16_shared":
            # Shared expert: only .weight from BF16 (skip NVFP4 scale tensors)
            if key.endswith(".weight"):
                if key in bf16_wm:
                    plan[key] = {"source": "bf16", "file": bf16_wm[key], "category": cat}
            # else: skip .weight_scale, .weight_scale_2, .input_scale
        else:
            # Everything else (attention, norms, router, embeddings, lm_head)
            if key in bf16_wm:
                plan[key] = {"source": "bf16", "file": bf16_wm[key], "category": cat}
            elif not any(key.endswith(s) for s in (".input_scale", ".weight_scale", ".weight_scale_2")):
                plan[key] = {"source": "nvfp4", "file": nvfp4_wm[key], "category": "nvfp4_fallback"}

    # Summary
    by_cat = defaultdict(int)
    by_src = defaultdict(int)
    for info in plan.values():
        by_cat[info["category"]] += 1
        by_src[info["source"]] += 1
    logger.info("By source: %s", dict(by_src))
    logger.info("By category: %s", dict(by_cat))
    logger.info("Total tensors: %d", len(plan))

    # Collect needed source files
    nvfp4_files = {info["file"] for key, info in plan.items() if info["source"] == "nvfp4"}
    bf16_files = {info["file"] for key, info in plan.items() if info["source"] == "bf16"}
    logger.info("Need %d NVFP4 shards, %d BF16 shards", len(nvfp4_files), len(bf16_files))

    # Assemble with ~5GB output shards
    MAX_SHARD_BYTES = 5 * 1024**3
    all_tensors = {}
    weight_map = {}
    shard_idx = 0
    current_shard_bytes = 0

    def flush_shard():
        nonlocal all_tensors, shard_idx, current_shard_bytes
        if not all_tensors:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        shard_path = OUTPUT_DIR / shard_name
        logger.info("Writing shard %s (%d tensors, %.2f GB)...",
                     shard_name, len(all_tensors), current_shard_bytes / 1e9)
        save_file(all_tensors, str(shard_path))
        for k in all_tensors:
            weight_map[k] = shard_name
        all_tensors = {}
        current_shard_bytes = 0
        shard_idx += 1

    # Process NVFP4 tensors
    for nvf in sorted(nvfp4_files):
        logger.info("Processing NVFP4 shard: %s", nvf)
        local = hf_hub_download(NVFP4_MODEL, nvf, token=HF_TOKEN)
        with safe_open(local, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in plan and plan[key]["source"] == "nvfp4":
                    tensor = f.get_tensor(key)
                    all_tensors[key] = tensor
                    current_shard_bytes += tensor.nbytes
                    if current_shard_bytes >= MAX_SHARD_BYTES:
                        flush_shard()

    # Process BF16 tensors
    for bf in sorted(bf16_files):
        logger.info("Processing BF16 shard: %s", bf)
        local = hf_hub_download(BF16_MODEL, bf, token=HF_TOKEN)
        with safe_open(local, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in plan and plan[key]["source"] == "bf16":
                    tensor = f.get_tensor(key)
                    all_tensors[key] = tensor
                    current_shard_bytes += tensor.nbytes
                    if current_shard_bytes >= MAX_SHARD_BYTES:
                        flush_shard()

    flush_shard()

    # Fix shard names (replace XXXXX with actual count)
    total_shards = shard_idx
    final_wm = {}
    for key, sn in weight_map.items():
        final_wm[key] = sn.replace("XXXXX", f"{total_shards:05d}")
    for i in range(total_shards):
        old = OUTPUT_DIR / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUTPUT_DIR / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)

    # Write index
    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": final_wm}, f, indent=2)

    # Copy config files from NVIDIA checkpoint
    for cfg in ["config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "preprocessor_config.json",
                "processor_config.json", "video_preprocessor_config.json",
                "hf_quant_config.json"]:
        try:
            src = hf_hub_download(NVFP4_MODEL, cfg, token=HF_TOKEN)
            shutil.copy2(src, OUTPUT_DIR / cfg)
        except Exception:
            pass

    logger.info("=== Assembly complete: %s ===", OUTPUT_DIR)
    logger.info("Shards: %d, Total keys: %d", total_shards, len(final_wm))


if __name__ == "__main__":
    main()
```

Run:

```bash
HF_TOKEN=hf_your_token python3 assemble_hybrid.py
```

Takes ~2 minutes if both models are already in HF cache. Output: ~234 GB, ~47 shards, ~371K tensors.

## Step 2: Patch SGLang

SGLang's `modelopt_fp4` loader will try to load shared expert weights as NVFP4 (expecting uint8 packed tensors with scale factors). Since our hybrid has BF16 shared expert weights, we need to tell SGLang to load them as unquantized.

Edit `sglang/srt/layers/quantization/modelopt_quant.py`, find the `_get_quant_method` method in the `ModelOptFp4Config` class, and add the shared_expert check:

```python
def _get_quant_method(
    self,
    layer: torch.nn.Module,
    prefix: str,
    *,
    Linear: type[LinearMethodBase],
    Moe: type[FusedMoEMethodBase],
) -> Optional[QuantizeMethodBase]:
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

    if isinstance(layer, LinearBase):
        # Hybrid NVFP4: keep shared_expert in BF16
        if '.shared_expert.' in prefix and '.shared_expert_gate' not in prefix:
            return UnquantizedLinearMethod()
        if is_layer_skipped(
            prefix, self.exclude_modules, self.packed_modules_mapping
        ) or self.is_layer_excluded(prefix):
            return UnquantizedLinearMethod()
        return Linear(self)
    elif self.kv_cache_quant_algo and isinstance(layer, RadixAttention):
        return ModelOptFp8KVCacheMethod(self)
    elif isinstance(layer, FusedMoE):
        if self.is_layer_excluded(prefix):
            return None
        return Moe(self)
    return None
```

The key addition is lines 12-13: any `LinearBase` layer with `.shared_expert.` in its prefix (but not `.shared_expert_gate`) is forced to `UnquantizedLinearMethod()`, bypassing NVFP4 dequantization.

### Applying the patch

```bash
# Inside your SGLang container:
QUANT_FILE="/opt/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py"

# One-liner sed patch (adds the shared_expert check before the is_layer_skipped check):
python3 -c "
content = open('$QUANT_FILE').read()
old = '''        if isinstance(layer, LinearBase):
            if is_layer_skipped('''
new = '''        if isinstance(layer, LinearBase):
            # Hybrid NVFP4: keep shared_expert in BF16
            if '.shared_expert.' in prefix and '.shared_expert_gate' not in prefix:
                return UnquantizedLinearMethod()
            if is_layer_skipped('''
assert old in content, 'Patch target not found - SGLang version may differ'
content = content.replace(old, new)
open('$QUANT_FILE', 'w').write(content)
print('Patch applied successfully')
"
```

## Step 3: Launch SGLang

```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model /path/to/hybrid-nvfp4 \
  --served-model-name Qwen3.5 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --tensor-parallel-size 4 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --cuda-graph-max-bs 4 \
  --max-running-requests 4 \
  --context-length 262144 \
  --chunked-prefill-size 32768 \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 5000 \
  --disable-custom-all-reduce
```

You will see "not found in params_dict" warnings during loading — these are normal (TP sharding: each rank only loads 1/4 of experts, warnings appear for experts assigned to other ranks).

## Why Not BF16 Routed Experts on Layers 0 and 59?

We tested keeping routed experts on layers 0 (first) and 59 (last, affects logits) in BF16. These are the most quality-sensitive layers. However, SGLang's `FusedMoE` layer requires uniform tensor format across all experts — mixing NVFP4 packed (uint8, half-size) with BF16 (full-size) within the same FusedMoE causes shape mismatches at load time.

Making this work would require patching SGLang's FusedMoE to handle per-layer format switching, which is significantly more invasive than the shared_expert patch.

## Why exclude_modules Doesn't Work

SGLang has an `exclude_modules` mechanism in `hf_quant_config.json`, but it doesn't work for shared expert in this model because SGLang's `WeightsMapper` transforms the key prefixes during loading. The prefix seen by `_get_quant_method` doesn't match the patterns in `exclude_modules`. The hardcoded prefix check (`.shared_expert.` in prefix) bypasses this issue entirely.
