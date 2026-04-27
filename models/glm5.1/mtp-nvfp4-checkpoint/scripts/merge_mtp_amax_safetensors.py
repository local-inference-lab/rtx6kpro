#!/usr/bin/env python3
"""Merge per-rank MTP MoE amax calibration dumps.

Each rank writes a safetensors file containing per-expert amax tensors.  This
script merges them into one file that can be passed to
build_glm51_mtp_nvfp4_routed_checkpoint.py --amax-file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def expand_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.safetensors")))
        else:
            files.append(path)
    return files


def merge_file(path: Path, merged: dict[str, torch.Tensor]) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            tensor = sf.get_tensor(key)
            if key not in merged:
                merged[key] = tensor.clone()
            elif key.endswith("sample_count"):
                merged[key] += tensor
            elif "amax" in key:
                merged[key] = torch.maximum(merged[key], tensor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/glm51_mtp_amax_merged.safetensors"),
    )
    args = parser.parse_args()

    files = [
        path
        for path in expand_inputs(args.inputs)
        if path.resolve() != args.output.resolve()
    ]
    if not files:
        raise SystemExit("no safetensors files found")

    merged: dict[str, torch.Tensor] = {}
    for path in files:
        merge_file(path, merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        merged,
        str(args.output),
        metadata={
            "source_files": ",".join(str(path) for path in files),
            "merge_rule": "amax=max, sample_count=sum",
        },
    )
    print(f"merged {len(files)} files -> {args.output}")
    print(f"tensors: {len(merged)}")


if __name__ == "__main__":
    main()
