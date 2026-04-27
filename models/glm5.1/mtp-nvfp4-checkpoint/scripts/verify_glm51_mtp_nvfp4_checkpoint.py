#!/usr/bin/env python3
"""Verify a GLM-5.1 hybrid checkpoint with NVFP4 routed MTP experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


EXPERT_COUNT = 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(str(model_dir / weight_map[key]), framework="pt", device="cpu") as sf:
        return sf.get_tensor(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    model_dir = args.checkpoint
    with open(model_dir / "model.safetensors.index.json") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    expected = EXPERT_COUNT * len(PROJECTIONS)
    counts = {"weight": 0, "weight_scale": 0, "weight_scale_2": 0, "input_scale": 0}
    shards: set[str] = set()
    for expert_id in range(EXPERT_COUNT):
        for proj in PROJECTIONS:
            prefix = f"model.layers.78.mlp.experts.{expert_id}.{proj}"
            for suffix in counts:
                key = f"{prefix}.{suffix}"
                if key not in weight_map:
                    raise SystemExit(f"missing {key}")
                counts[suffix] += 1
                shards.add(weight_map[key])

    if any(value != expected for value in counts.values()):
        raise SystemExit(f"bad tensor counts: {counts}, expected each={expected}")

    sample_keys = [
        "model.layers.78.mlp.experts.0.gate_proj.weight",
        "model.layers.78.mlp.experts.0.gate_proj.weight_scale",
        "model.layers.78.mlp.experts.0.gate_proj.weight_scale_2",
        "model.layers.78.mlp.experts.0.gate_proj.input_scale",
        "model.layers.78.mlp.shared_experts.gate_proj.weight",
    ]
    print("checkpoint:", model_dir)
    print("counts:", counts)
    print("mtp nvfp4 shards:", sorted(shard for shard in shards if "mtp-nvfp4" in shard))
    for key in sample_keys:
        tensor = load_tensor(model_dir, weight_map, key)
        print(key, tensor.dtype, tuple(tensor.shape), weight_map[key])

    weight = load_tensor(model_dir, weight_map, sample_keys[0])
    scale = load_tensor(model_dir, weight_map, sample_keys[1])
    scale2 = load_tensor(model_dir, weight_map, sample_keys[2])
    input_scale = load_tensor(model_dir, weight_map, sample_keys[3])
    shared = load_tensor(model_dir, weight_map, sample_keys[4])
    assert weight.dtype == torch.uint8 and tuple(weight.shape) == (2048, 3072)
    assert scale.dtype == torch.float8_e4m3fn and tuple(scale.shape) == (2048, 384)
    assert scale2.dtype == torch.float32 and tuple(scale2.shape) == ()
    assert input_scale.dtype == torch.float32 and tuple(input_scale.shape) == ()
    assert shared.dtype == torch.bfloat16 and tuple(shared.shape) == (2048, 6144)
    print("PASS")


if __name__ == "__main__":
    main()
