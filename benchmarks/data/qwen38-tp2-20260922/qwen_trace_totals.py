"""Count rank-0 trace operations; nested CPU totals are not additive latency."""

import collections
import gzip
import json
import sys
from pathlib import Path


def summarize(path):
    with gzip.open(path) as source:
        events = json.load(source)["traceEvents"]
    selected = collections.defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    kernels = collections.defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    steps = []
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if name.startswith("ProfilerStep#"):
            steps.append(name)
        key = None
        if "qwen4_exp/nvidia/model_state.py" in name and name.endswith(": prepare_attn"):
            key = "qwen_attention_metadata_host_inclusive"
        elif "gdn_attn.py" in name and name.endswith(": update_block_table"):
            key = "gdn_group_refresh_host_inclusive"
        elif "b12x_gdn_metadata.py" in name and name.endswith(": refresh_state_indices"):
            key = "b12x_mixed_group_refresh_host_inclusive"
        elif name in {"aten::_foreach_copy_", "aten::fill_", "cudaLaunchKernel", "cudaGraphLaunch"}:
            key = name
        if key:
            selected[key]["calls"] += 1
            selected[key]["total_us"] += event.get("dur", 0)
        if event.get("cat") == "kernel":
            kernels[name]["calls"] += 1
            kernels[name]["total_us"] += event.get("dur", 0)
    return {
        "profile_steps": steps,
        "observed_target_metadata_preparations": selected.get(
            "qwen_attention_metadata_host_inclusive", {}
        ).get("calls", 0),
        "operations": dict(selected),
        "top_gpu_kernel_totals_not_wall_time": sorted(
            [{"name": name, **value} for name, value in kernels.items()],
            key=lambda item: -item["total_us"],
        )[:20],
    }


print(json.dumps({Path(path).name: summarize(path) for path in sys.argv[1:]}, indent=2))
