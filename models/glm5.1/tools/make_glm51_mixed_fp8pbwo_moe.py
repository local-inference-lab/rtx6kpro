#!/usr/bin/env python3
"""Build a GLM-5.1 ModelOpt mixed checkpoint with selected MoE layers in FP8_PB_WO.

The source checkpoint stays symlinked where possible. Any original NVFP4 shard
that contains selected expert tensors is rewritten with only those tensors
removed, because vLLM iterates safetensors files and will otherwise see
duplicate names even if model.safetensors.index.json points elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERTS = range(256)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def parse_layers(spec: str) -> list[int]:
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            layers.update(range(int(first), int(last) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def is_selected_expert_key(key: str, layers: set[int]) -> bool:
    for layer in layers:
        if key.startswith(f"model.layers.{layer}.mlp.experts."):
            return True
    return False


def copy_or_link_metadata(src: Path, dst: Path) -> None:
    for child in src.iterdir():
        if child.name in {"config.json", "model.safetensors.index.json"}:
            continue
        target = dst / child.name
        if child.name.endswith(".safetensors"):
            continue
        if child.is_dir():
            os.symlink(child, target, target_is_directory=True)
        else:
            os.symlink(child, target)


def save_filtered_source_shards(
    src: Path,
    dst: Path,
    source_weight_map: dict[str, str],
    remove_keys: set[str],
) -> tuple[dict[str, str], set[str]]:
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in source_weight_map.items():
        keys_by_file[filename].append(key)

    changed_files = {source_weight_map[key] for key in remove_keys}
    new_weight_map: dict[str, str] = {}

    for filename, keys in sorted(keys_by_file.items()):
        src_file = src / filename
        dst_file = dst / filename
        if filename in changed_files:
            tensors = {}
            kept = 0
            dropped = 0
            with safe_open(src_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in remove_keys:
                        dropped += 1
                        continue
                    tensors[key] = f.get_tensor(key)
                    kept += 1
            save_file(tensors, dst_file)
            print(f"filtered {filename}: kept={kept} dropped={dropped}")
        else:
            os.symlink(src_file, dst_file)

        for key in keys:
            if key not in remove_keys:
                new_weight_map[key] = filename

    return new_weight_map, changed_files


def add_fp8_pb_wo_layers(
    fp8_src: Path,
    dst: Path,
    fp8_weight_map: dict[str, str],
    new_weight_map: dict[str, str],
    layers: list[int],
) -> None:
    for layer in layers:
        out_filename = f"model-mixed-fp8pbwo-layer{layer:02d}.safetensors"
        tensors = {}
        fp8_keys_by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for expert in EXPERTS:
            for projection in PROJECTIONS:
                base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                weight_key = f"{base}.weight"
                scale_inv_key = f"{base}.weight_scale_inv"
                scale_key = f"{base}.weight_scale"
                fp8_keys_by_file[fp8_weight_map[weight_key]].append(
                    (weight_key, weight_key)
                )
                fp8_keys_by_file[fp8_weight_map[scale_inv_key]].append(
                    (scale_inv_key, scale_key)
                )
                new_weight_map[weight_key] = out_filename
                new_weight_map[scale_key] = out_filename

        # Load only this layer into memory, even though source tensors span
        # several FP8 checkpoint shards.
        for filename, mappings in sorted(fp8_keys_by_file.items()):
            with safe_open(fp8_src / filename, framework="pt", device="cpu") as f:
                for src_key, dst_key in mappings:
                    tensors[dst_key] = f.get_tensor(src_key)

        save_file(tensors, dst / out_filename)
        print(f"wrote {out_filename}: tensors={len(tensors)}")


def write_mixed_config(src: Path, dst: Path, layers: list[int]) -> None:
    selected = set(layers)
    cfg = load_json(src / "config.json")
    qcfg = cfg.setdefault("quantization_config", {})
    qcfg["quant_method"] = "modelopt"
    qcfg["quant_algo"] = "MIXED_PRECISION"
    qcfg["quantized_layers"] = {
        f"model.layers.{layer}.mlp.experts": {
            "quant_algo": "FP8_PB_WO",
            "weight_block_size": [128, 128],
        }
        if layer in selected
        else {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
        for layer in range(3, 78)
    }
    write_json(dst / "config.json", cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvfp4-source", required=True, type=Path)
    parser.add_argument("--fp8-source", required=True, type=Path)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma/range list, e.g. 45-47,51-62",
    )
    args = parser.parse_args()

    src = args.nvfp4_source.resolve()
    fp8_src = args.fp8_source.resolve()
    dst = args.dest
    layers = parse_layers(args.layers)
    layer_set = set(layers)

    if dst.exists():
        raise SystemExit(f"destination already exists: {dst}")
    if not src.is_dir():
        raise SystemExit(f"missing NVFP4 source: {src}")
    if not fp8_src.is_dir():
        raise SystemExit(f"missing FP8 source: {fp8_src}")

    source_index = load_json(src / "model.safetensors.index.json")
    source_weight_map = source_index["weight_map"]
    fp8_weight_map = load_json(fp8_src / "model.safetensors.index.json")[
        "weight_map"
    ]

    remove_keys = {
        key
        for key in source_weight_map
        if is_selected_expert_key(key, layer_set)
    }

    dst.mkdir(parents=True)
    copy_or_link_metadata(src, dst)
    write_mixed_config(src, dst, layers)
    new_weight_map, _ = save_filtered_source_shards(
        src, dst, source_weight_map, remove_keys
    )
    add_fp8_pb_wo_layers(fp8_src, dst, fp8_weight_map, new_weight_map, layers)

    new_index = {
        "metadata": {
            "total_size": sum(
                (dst / filename).stat().st_size
                for filename in set(new_weight_map.values())
            )
        },
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    write_json(dst / "model.safetensors.index.json", new_index)

    print(f"created {dst}")
    print(f"layers={','.join(map(str, layers))}")
    print(f"weight_map keys={len(new_weight_map)}")


if __name__ == "__main__":
    main()
