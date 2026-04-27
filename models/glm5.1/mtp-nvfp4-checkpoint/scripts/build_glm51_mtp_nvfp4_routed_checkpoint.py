#!/usr/bin/env python3
"""Build a GLM-5.1 hybrid checkpoint with NVFP4 routed MTP experts.

Input:
  Luke Alonso's GLM-5.1-NVFP4 checkpoint.  It already contains the target model
  in NVFP4 and one BF16 MTP/NextN layer at model.layers.78.*.

Output:
  A symlinked checkpoint that reuses the base model files and replaces only
  model.layers.78.mlp.experts.*.{gate,up,down}.weight with ModelOpt NVFP4
  tensors plus weight_scale, weight_scale_2 and input_scale tensors.

Shared experts are intentionally left BF16, matching the base checkpoint's
*shared_experts* ignore rule.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.export.quant_utils import NVFP4QTensor


EXPERT_COUNT = 256
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
BLOCK_SIZE = 16
SHARD_LIMIT = int(4.5 * 1024**3)
ORIGINAL_MTP_SHARD = "model-00084-of-00084.safetensors"
NVFP4_ACT_SCALE_DENOM = 448.0


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def load_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(str(model_dir / weight_map[key]), framework="pt", device="cpu") as sf:
        return sf.get_tensor(key)


def load_amax_tensors(amax_file_str: str | None) -> dict[str, torch.Tensor]:
    if not amax_file_str:
        return {}

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(amax_file_str, framework="pt", device="cpu") as sf:
        for key in sf.keys():
            if key.endswith("input_amax") or key.endswith("_input_amax"):
                tensors[key] = sf.get_tensor(key).to(torch.float32)
    if not tensors:
        raise ValueError(f"{amax_file_str} does not contain input_amax tensors")
    return tensors


def find_amax_tensor(
    amax_tensors: dict[str, torch.Tensor], suffix: str
) -> torch.Tensor | None:
    for key in sorted(amax_tensors):
        if key.endswith(suffix):
            return amax_tensors[key]
    return None


def input_scale_from_amax(
    amax_tensors: dict[str, torch.Tensor],
    expert_id: int,
    proj: str,
) -> torch.Tensor | None:
    if not amax_tensors:
        return None

    exact = find_amax_tensor(amax_tensors, f".experts.{expert_id}.{proj}.input_amax")
    if exact is not None:
        amax = exact.reshape(()).to(torch.float32)
    else:
        compact_suffix = ".experts.w2_input_amax" if proj == "down_proj" else ".experts.w13_input_amax"
        compact = find_amax_tensor(amax_tensors, compact_suffix)
        if compact is None or compact.numel() <= expert_id:
            return None
        amax = compact[expert_id].reshape(()).to(torch.float32)

    if float(amax) <= 0:
        return None
    return torch.reciprocal(amax.clamp_min(1e-6) * NVFP4_ACT_SCALE_DENOM).cpu()


def resolve_input_scale(
    *,
    amax_tensors: dict[str, torch.Tensor],
    base_dir: Path,
    weight_map: dict[str, str],
    expert_id: int,
    proj: str,
) -> torch.Tensor:
    scale = input_scale_from_amax(amax_tensors, expert_id, proj)
    if scale is not None:
        return scale.reshape(())

    proxy_scale_key = f"model.layers.77.mlp.experts.{expert_id}.{proj}.input_scale"
    if proxy_scale_key in weight_map:
        return load_tensor(base_dir, weight_map, proxy_scale_key).to(torch.float32).reshape(())

    return torch.tensor(1.0 / (6.0 * NVFP4_ACT_SCALE_DENOM), dtype=torch.float32)


def symlink_or_copy_tree(base_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        for dst in out_dir.iterdir():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in base_dir.iterdir():
        dst = out_dir / src.name
        if src.name in {"config.json", "model.safetensors.index.json"}:
            shutil.copy2(src, dst)
        elif src.is_file():
            os.symlink(src, dst)


def filter_original_mtp_shard(
    base_dir: Path,
    out_dir: Path,
    excluded_keys: set[str],
) -> None:
    """Replace original MTP shard symlink with a local shard minus BF16 experts.

    fastsafetensors scans every .safetensors file in the directory, not only the
    index.  Leaving the original shard symlinked can collide with the new NVFP4
    routed expert tensors.
    """

    src = base_dir / ORIGINAL_MTP_SHARD
    dst = out_dir / ORIGINAL_MTP_SHARD
    kept: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            if key not in excluded_keys:
                kept[key] = sf.get_tensor(key)

    if dst.exists() or dst.is_symlink():
        dst.unlink()
    total_gib = sum(tensor_nbytes(t) for t in kept.values()) / 1024**3
    print(
        f"writing filtered {ORIGINAL_MTP_SHARD}: {len(kept)} tensors, "
        f"{total_gib:.2f} GiB",
        flush=True,
    )
    save_file(kept, str(dst))


def flush_shard(
    out_dir: Path,
    shard: dict[str, torch.Tensor],
    shard_idx: int,
    weight_map: dict[str, str],
    *,
    name_prefix: str,
) -> int:
    if not shard:
        return shard_idx
    name = f"{name_prefix}-{shard_idx:05d}.safetensors"
    total_gib = sum(tensor_nbytes(t) for t in shard.values()) / 1024**3
    print(f"writing {name}: {len(shard)} tensors, {total_gib:.2f} GiB", flush=True)
    save_file(shard, str(out_dir / name))
    for key in shard:
        weight_map[key] = name
    shard.clear()
    return shard_idx + 1


def quantize_weight(
    weight: torch.Tensor,
    *,
    weight_scale_2: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized, scale, scale_2 = NVFP4QTensor.quantize(
        weight.contiguous(),
        block_size=BLOCK_SIZE,
        weights_scaling_factor_2=weight_scale_2,
    )
    return (
        quantized._quantized_data.contiguous().cpu(),
        scale.contiguous().cpu(),
        scale_2.reshape(()).contiguous().cpu(),
    )


def worker_quantize(
    rank: int,
    device_id: int,
    base_dir_str: str,
    out_dir_str: str,
    amax_file_str: str | None,
    tasks: list[tuple[int, str]],
) -> tuple[dict[str, str], int]:
    base_dir = Path(base_dir_str)
    out_dir = Path(out_dir_str)
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    with open(base_dir / "model.safetensors.index.json") as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    amax_tensors = load_amax_tensors(amax_file_str)

    local_weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 1
    converted = 0

    for expert_id, proj in tasks:
        src_key = f"model.layers.78.mlp.experts.{expert_id}.{proj}.weight"
        if src_key not in weight_map:
            raise KeyError(f"missing BF16 MTP weight: {src_key}")

        weight = load_tensor(base_dir, weight_map, src_key)
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"{src_key} expected BF16, got {weight.dtype}")

        weight = weight.to(device, non_blocking=False)
        weight_scale_2 = None
        if proj in ("gate_proj", "up_proj"):
            other_proj = "up_proj" if proj == "gate_proj" else "gate_proj"
            other_key = f"model.layers.78.mlp.experts.{expert_id}.{other_proj}.weight"
            other_weight = load_tensor(base_dir, weight_map, other_key)
            if other_weight.dtype != torch.bfloat16:
                raise TypeError(f"{other_key} expected BF16, got {other_weight.dtype}")
            other_weight = other_weight.to(device, non_blocking=False)
            weight_scale_2 = torch.maximum(
                NVFP4QTensor.get_weights_scaling_factor_2(weight),
                NVFP4QTensor.get_weights_scaling_factor_2(other_weight),
            )
            del other_weight

        qweight, wscale, wscale_2 = quantize_weight(
            weight, weight_scale_2=weight_scale_2
        )
        del weight
        torch.cuda.empty_cache()

        out_prefix = src_key.removesuffix(".weight")
        tensors = {
            f"{out_prefix}.weight": qweight,
            f"{out_prefix}.weight_scale": wscale,
            f"{out_prefix}.weight_scale_2": wscale_2,
            f"{out_prefix}.input_scale": resolve_input_scale(
                amax_tensors=amax_tensors,
                base_dir=base_dir,
                weight_map=weight_map,
                expert_id=expert_id,
                proj=proj,
            ),
        }

        tensors_bytes = sum(tensor_nbytes(t) for t in tensors.values())
        if shard and shard_bytes + tensors_bytes > SHARD_LIMIT:
            shard_idx = flush_shard(
                out_dir,
                shard,
                shard_idx,
                local_weight_map,
                name_prefix=f"model-mtp-nvfp4-routed-gpu{rank}",
            )
            shard_bytes = 0

        shard.update(tensors)
        shard_bytes += tensors_bytes
        converted += 1

        if converted % 8 == 0:
            print(
                f"[gpu{rank}/cuda:{device_id}] converted "
                f"{converted}/{len(tasks)} projections",
                flush=True,
            )

    flush_shard(
        out_dir,
        shard,
        shard_idx,
        local_weight_map,
        name_prefix=f"model-mtp-nvfp4-routed-gpu{rank}",
    )
    return local_weight_map, converted


def build_mixed_precision_quant_config() -> dict:
    quantized_layers: dict[str, dict[str, int | str]] = {}
    for expert_id in range(EXPERT_COUNT):
        for proj in PROJECTIONS:
            layer_info = {"quant_algo": "NVFP4", "group_size": BLOCK_SIZE}
            quantized_layers[f"model.layers.78.mlp.experts.{expert_id}.{proj}"] = layer_info
            quantized_layers[
                f"model.layers.78.mtp_block.mlp.experts.{expert_id}.{proj}"
            ] = layer_info

    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": BLOCK_SIZE,
        "ignore": [
            "lm_head",
            "model.embed_tokens",
            "model.layers.78.self_attn*",
            "model.layers.78.self_attn.*",
            "model.layers.78.mtp_block.self_attn*",
            "model.layers.78.mtp_block.self_attn.*",
            "model.layers.78.mlp.shared_experts*",
            "model.layers.78.mlp.shared_experts.*",
            "model.layers.78.mtp_block.mlp.shared_experts*",
            "model.layers.78.mtp_block.mlp.shared_experts.*",
        ],
        "quantized_layers": quantized_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            "/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4/"
            "snapshots/1b9f53ee1d11fcbb7ecefed115359afba02f104f"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink"),
    )
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--amax-file",
        type=Path,
        default=None,
        help=(
            "Merged safetensors file from MTP amax calibration. If provided, "
            "input_scale is set to 1/(input_amax*448) per expert/projection."
        ),
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir
    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    workers = min(args.workers, len(devices), EXPERT_COUNT * len(PROJECTIONS))

    with open(base_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map: dict[str, str] = dict(index["weight_map"])

    symlink_or_copy_tree(base_dir, out_dir)

    all_tasks = [
        (expert_id, proj)
        for expert_id in range(EXPERT_COUNT)
        for proj in PROJECTIONS
    ]
    task_groups = [all_tasks[i::workers] for i in range(workers)]
    worker_args = [
        (
            rank,
            devices[rank % len(devices)],
            str(base_dir),
            str(out_dir),
            str(args.amax_file) if args.amax_file else None,
            task_group,
        )
        for rank, task_group in enumerate(task_groups)
    ]

    if workers == 1:
        results = [worker_quantize(*worker_args[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.starmap(worker_quantize, worker_args)

    converted = 0
    for local_weight_map, local_converted in results:
        weight_map.update(local_weight_map)
        converted += local_converted

    excluded_original_keys = {
        f"model.layers.78.mlp.experts.{expert_id}.{proj}.weight"
        for expert_id in range(EXPERT_COUNT)
        for proj in PROJECTIONS
    }
    for key in excluded_original_keys:
        if key not in weight_map:
            raise KeyError(f"missing converted weight map entry: {key}")

    filter_original_mtp_shard(base_dir, out_dir, excluded_original_keys)

    index["weight_map"] = dict(sorted(weight_map.items()))
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = sum(
        (out_dir / name).stat().st_size for name in set(weight_map.values())
    )
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    with open(out_dir / "config.json") as f:
        config = json.load(f)
    config.setdefault("_voipmonitor_hybrid", {})
    config["_voipmonitor_hybrid"].update(
        {
            "base_checkpoint": str(base_dir),
            "mtp_routed_experts": (
                "model.layers.78.mlp.experts.* quantized from BF16 to ModelOpt NVFP4"
            ),
            "mtp_shared_experts": (
                "left BF16 to match base checkpoint *shared_experts* ignore rule"
            ),
            "input_scale_source": str(args.amax_file)
            if args.amax_file
            else "model.layers.77.mlp.experts.{id}.{proj}.input_scale",
        }
    )
    config["quantization_config"] = build_mixed_precision_quant_config()
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"done: {out_dir}")
    print(f"converted expert projections: {converted}")
    print(f"workers: {workers}, devices: {devices}")


if __name__ == "__main__":
    main()
