"""Compare TP2 Spark MTP3 cache reservations using the runtime's integer planner.

This performs no model execution. The recorded 3072-token LMCache geometry is
reconstructed from the TP2 serving log and checked against its reported capacity.
Changing only the recovery layout separates its capacity effect from cache policy.
"""

import argparse
import json
import math
import os
from types import SimpleNamespace
from pathlib import Path

import torch

from vllm.models.glm5next.nvidia.model import Glm5NextForCausalLM
from vllm.v1.core import kv_cache_utils as planner
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, MLAAttentionSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE"] = "1536"
    available = 4_190_109_696
    config = SimpleNamespace(
        use_request_boundary_checkpoints=True,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            max_model_len=1_048_576,
            hf_config=SimpleNamespace(
                linear_num_heads=64,
                linear_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
            use_kda_recoverssm=False,
            mamba_cache_mode="align",
        ),
        scheduler_config=SimpleNamespace(max_num_scheduled_tokens=3072),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, decode_context_parallel_size=2
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    attention = MLAAttentionSpec(
        block_size=1536,
        num_kv_heads=1,
        head_size=528,
        dtype=torch.uint8,
        cache_dtype_str="fp8",
        model_version="glm5_next",
        page_tail_bytes_per_token=33,
        alignment=64 * 132,
    )
    assert attention.page_size_bytes == 861_696
    rows = []
    for recovery in (False, True):
        config.model_config.max_model_len = 1_048_576
        config.cache_config.use_kda_recoverssm = recovery
        recurrent = MambaSpec(
            block_size=1536,
            shapes=Glm5NextForCausalLM.get_mamba_state_shape_from_config(config),
            dtypes=Glm5NextForCausalLM.get_mamba_state_dtype_from_config(config),
            mamba_cache_mode="align",
            num_speculative_blocks=0 if recovery else 3,
        )
        specs = {
            **{f"recurrent.{i}": recurrent for i in range(34)},
            **{f"attention.{i}": attention for i in range(12)},
        }
        groups = planner._get_weighted_shared_pool_kv_cache_groups(config, specs)
        stride = planner._pool_bytes_per_block(groups)
        blocks = available // stride
        fitted = planner._estimate_max_model_len_from_groups(
            config, groups, (blocks - 1) * stride
        )
        config.model_config.max_model_len = fitted
        cache = KVCacheConfig(
            num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=groups
        )
        capacity, _ = planner.get_kv_cache_capacity(config, cache)
        state_blocks = sum(
            math.ceil(
                g.kv_cache_spec.max_memory_usage_bytes(config)
                / g.kv_cache_spec.page_size_bytes
            )
            for g in groups
            if isinstance(g.kv_cache_spec, MambaSpec)
        )
        endpoints = planner._request_boundary_reserve_blocks(config, len(groups))
        rows.append(
            {
                "recovery": recovery,
                "recurrent_page_bytes": recurrent.page_size_bytes,
                "group_widths": [len(g.layer_names) for g in groups],
                "pool_block_bytes": stride,
                "pool_blocks": blocks,
                "recurrent_working_blocks": state_blocks,
                "endpoint_restore_blocks": endpoints,
                "fixed_per_request_bytes": (state_blocks + endpoints) * stride,
                "fitted_context": fitted,
                "logical_capacity": capacity,
            }
        )
    assert rows[1]["recurrent_page_bytes"] == 2_375_680
    assert rows[1]["logical_capacity"] == 899_579, rows
    assert rows[1]["fitted_context"] == 897_024, rows
    rendered = json.dumps({"allocation_bytes": available, "arms": rows}, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
