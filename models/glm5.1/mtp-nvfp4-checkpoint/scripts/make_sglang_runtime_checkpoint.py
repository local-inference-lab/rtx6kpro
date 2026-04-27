#!/usr/bin/env python3
"""Create the SGLang runtime view of the hybrid GLM-5.1 MTP NVFP4 checkpoint.

The vLLM checkpoint uses a mixed-precision quantization_config that explicitly
lists layer-78 routed experts.  The SGLang test stack used during this work
loaded the same tensors through a standard ModelOpt NVFP4 config, plus a local
SGLang patch that allows ModelOpt FP4 for NextN routed MoE when the layer-78
scales are present.

This script creates a symlinked view and copies only config.json from the base
Luke checkpoint.  All weights and the model index continue to point at the
hybrid checkpoint.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.out_dir.exists():
        for dst in args.out_dir.iterdir():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.hybrid_dir.iterdir():
        dst = args.out_dir / src.name
        if src.name == "config.json":
            shutil.copy2(args.base_dir / "config.json", dst)
        else:
            os.symlink(src, dst)

    print(f"created SGLang runtime checkpoint view: {args.out_dir}")
    print("weights/index source:", args.hybrid_dir)
    print("config source:", args.base_dir / "config.json")


if __name__ == "__main__":
    main()
