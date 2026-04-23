#!/usr/bin/env python3
"""Sweep (num_kv_splits, BLOCK_N, BLOCK_H, num_stages, num_warps) for Kimi MLA
decode attention (stage-1) on sm120 fp8. Runs per-GPU via CUDA_VISIBLE_DEVICES.

Usage (single GPU):
    python tune_triton_mla.py --rank 0 --world 8 --out /tmp/tune_gpu0.json

Parallel across 8 GPUs (shell driver):
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i python tune_triton_mla.py \
            --rank $i --world 8 --out /tmp/tune_gpu$i.json &
    done; wait
"""

import argparse
import itertools
import json
import math
import sys
import time
import traceback

import torch
import triton

from vllm.v1.attention.ops.triton_decode_attention import _fwd_grouped_kernel_stage1

# ---- Kimi MLA shapes (fixed) ----
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
LK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
LV = KV_LORA_RANK  # 512
BLOCK_DMODEL = 512
BLOCK_DPE = 64
BLOCK_DV = triton.next_power_of_2(LV)  # 512
PAGE_SIZE = 16
SHMEM_LIMIT = 101376  # sm120 dynamic shared mem cap

# ---- Sweep ranges ----
HEADS_LIST = [16, 64, 128]                # DCP=1, 4, 8 post-allgather
MAX_MODEL_LENS = [16000, 64000, 128000, 262144]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NUM_KV_SPLITS_LIST = [1, 2, 4, 8, 16, 32, 64]
BLOCK_N_LIST = [16, 32, 64, 128]
BLOCK_H_LIST = [8, 16, 32, 64]
STAGES_LIST = [1, 2, 3]
WARPS_LIST = [2, 4, 8]

# For each mml bucket we test at 5 fractions of mml so the picker gets a
# representative sample across the realistic seq_len range. Round 2 (4 points
# 5/30/60/95%) still regressed ctx=0 conc=128 by ~7% because real workload at
# ctx=0 has seq_len = just the output tokens (~50-100), i.e. ≪1% of mml. The
# 1% bucket gives tuning coverage of the short-seq regime.
#   mml=262144: seq_lens ≈ 2621, 13107, 78643, 157286, 249036
#   mml=16000:  seq_lens ≈    160,   800,   4800,   9600,  15200
SEQ_LEN_FRACTIONS = (0.01, 0.05, 0.30, 0.60, 0.95)

def test_seq_lens_for(mml: int) -> list[int]:
    return [max(256, int(mml * f)) for f in SEQ_LEN_FRACTIONS]

WARMUP = 2
TIMED = 5


def estimate_shmem(BLOCK_N: int, BLOCK_H: int, num_stages: int) -> int:
    # K stages (fp8 = 1 B)
    k_tile = num_stages * BLOCK_N * LK
    # Q tile (bf16 = 2 B)
    q_tile = BLOCK_H * LK * 2
    # QK accumulator (fp32)
    acc = BLOCK_H * BLOCK_N * 4
    # plus overhead ~4 KB for softmax/mask
    return k_tile + q_tile + acc + 4096


def analytic_prefilter(BLOCK_N, BLOCK_H, num_stages):
    return estimate_shmem(BLOCK_N, BLOCK_H, num_stages) <= SHMEM_LIMIT


def make_tensors(heads: int, max_seq_len: int, B: int, max_splits: int,
                 device: torch.device):
    """Allocate synthetic tensors SIZED FOR max_seq_len (reusable across
    smaller seq_lens — the kernel reads up to b_seqlen per request)."""
    # Q: (B, heads, LK)  bf16
    q = torch.randn(B, heads, LK, dtype=torch.bfloat16, device=device) * 0.1

    # KV cache: small pool (10k unique blocks) with random indexing.
    pool_blocks = 10000
    kv_cache_bf = (torch.randn(pool_blocks, PAGE_SIZE, 1, LK,
                               dtype=torch.bfloat16, device=device) * 0.1)
    kv_cache = kv_cache_bf.to(torch.float8_e4m3fn)
    del kv_cache_bf

    # req_to_tokens sized for the MAX seq_len we'll test at.
    max_bpr = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    req_to_tokens = torch.randint(
        0, pool_blocks, (B, max_bpr), dtype=torch.int32, device=device)
    # b_seqlen starts at max; caller mutates in-place per sub-seq_len.
    b_seqlen = torch.full((B,), max_seq_len, dtype=torch.int32, device=device)

    # att_out pre-allocated at MAX splits; we pass a slice per call.
    att_out_full = torch.empty(B, heads, max_splits, LV + 1,
                               dtype=torch.float32, device=device)

    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)

    return q, kv_cache, req_to_tokens, b_seqlen, att_out_full, k_scale, v_scale


def run_one_multi_seq(q, kv_cache, req_to_tokens, b_seqlen, att_out_full,
                      k_scale, v_scale, heads, B, num_kv_splits,
                      BLOCK_N, BLOCK_H, num_stages, num_warps,
                      seq_lens: list[int]):
    """Attempt one config at each of seq_lens. Return list of avg ms per
    seq_len, or None if any launch failed."""
    att_out = att_out_full[:, :, :num_kv_splits, :]
    kv_group_num = heads
    grid = (B, triton.cdiv(heads, min(BLOCK_H, kv_group_num)), num_kv_splits)
    sm_scale = 1.0 / math.sqrt(LK)

    def launch():
        _fwd_grouped_kernel_stage1[grid](
            q, kv_cache, kv_cache, sm_scale,
            req_to_tokens, b_seqlen, att_out,
            req_to_tokens.stride(0),
            q.stride(0), q.stride(1),
            kv_cache.stride(-3), kv_cache.stride(-2),
            kv_cache.stride(-3), kv_cache.stride(-2),
            att_out.stride(0), att_out.stride(1), att_out.stride(2),
            k_scale, v_scale,
            kv_group_num=kv_group_num, q_head_num=heads,
            BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, NUM_KV_SPLITS=num_kv_splits,
            PAGE_SIZE=PAGE_SIZE, logit_cap=0.0,
            num_warps=num_warps, num_stages=num_stages,
            Lk=LK, Lv=LV, IS_MLA=True,
        )

    # Warmup at max seq_len (worst case for shmem scheduling)
    b_seqlen.fill_(max(seq_lens))
    try:
        for _ in range(WARMUP):
            launch()
        torch.cuda.synchronize()
    except Exception:
        return None

    ms_per_seq: list[float] = []
    for seq_len in seq_lens:
        b_seqlen.fill_(seq_len)
        # one more warmup at this seq_len to stabilize caches
        try:
            launch()
            torch.cuda.synchronize()
        except Exception:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            launch()
        end.record()
        end.synchronize()
        ms_per_seq.append(start.elapsed_time(end) / TIMED)
    return ms_per_seq


def iter_outer_points(rank: int, world: int):
    """Yield (heads, mml, B) triples for this rank. Round-robin across ranks."""
    i = 0
    for heads in HEADS_LIST:
        for mml in MAX_MODEL_LENS:
            for B in BATCHES:
                if i % world == rank:
                    yield heads, mml, B
                i += 1


def iter_inner_configs():
    """Yield all (num_kv_splits, BLOCK_N, BLOCK_H, stages, warps) combos that
    pass shmem pre-filter."""
    for (num_kv_splits, BLOCK_N, BLOCK_H, stages, warps) in itertools.product(
            NUM_KV_SPLITS_LIST, BLOCK_N_LIST, BLOCK_H_LIST,
            STAGES_LIST, WARPS_LIST):
        if not analytic_prefilter(BLOCK_N, BLOCK_H, stages):
            continue
        # BLOCK_H > kv_group_num (=heads) is wasted but not illegal.
        # num_warps must divide some threadblock constraint — triton handles.
        yield (num_kv_splits, BLOCK_N, BLOCK_H, stages, warps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda", 0)  # CUDA_VISIBLE_DEVICES maps this
    torch.cuda.set_device(device)

    outer_points = list(iter_outer_points(args.rank, args.world))
    inner_configs = list(iter_inner_configs())
    total = len(outer_points) * len(inner_configs)

    print(f"[rank {args.rank}] {len(outer_points)} outer × "
          f"{len(inner_configs)} inner = {total} configs",
          file=sys.stderr, flush=True)

    results = []
    t0 = time.time()
    done = 0

    for heads, mml, B in outer_points:
        seq_lens = test_seq_lens_for(mml)
        max_seq_len = max(seq_lens)
        # Alloc tensors sized for the max seq_len (reused across all sub
        # seq_lens and all inner configs).
        try:
            tensors = make_tensors(heads, max_seq_len, B,
                                   max(NUM_KV_SPLITS_LIST), device)
        except torch.cuda.OutOfMemoryError:
            print(f"[rank {args.rank}] OOM alloc heads={heads} mml={mml} B={B}",
                  file=sys.stderr, flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[rank {args.rank}] alloc fail: {e}",
                  file=sys.stderr, flush=True)
            continue

        best = None  # (geomean_ms, ms_per_seq, nsp, BN, BH, ns, nw)
        for (nsp, BN, BH, ns, nw) in inner_configs:
            ms_list = run_one_multi_seq(
                *tensors, heads, B, nsp, BN, BH, ns, nw, seq_lens)
            done += 1
            if ms_list is not None:
                # Geometric mean: minimises sum of log(ms) across seq_lens.
                # Balanced picker that doesn't let one seq_len dominate.
                logsum = 0.0
                for v in ms_list:
                    logsum += math.log(max(v, 1e-6))
                geomean = math.exp(logsum / len(ms_list))
                if best is None or geomean < best[0]:
                    best = (geomean, ms_list, nsp, BN, BH, ns, nw)
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else -1
                print(f"[rank {args.rank}] {done}/{total} "
                      f"elapsed={elapsed:.0f}s rate={rate:.1f}/s "
                      f"eta={eta:.0f}s", file=sys.stderr, flush=True)

        if best is not None:
            results.append({
                "heads": heads, "max_model_len": mml, "B": B,
                "seq_lens": seq_lens,
                "ms_per_seq": best[1],
                "geomean_ms": best[0],
                "num_kv_splits": best[2],
                "BLOCK_N": best[3],
                "BLOCK_H": best[4],
                "num_stages": best[5],
                "num_warps": best[6],
            })
            msstr = "/".join(f"{m:.2f}" for m in best[1])
            print(f"[rank {args.rank}] WIN heads={heads} mml={mml} B={B} "
                  f"geomean={best[0]:.3f} ms={msstr} splits={best[2]} "
                  f"BN={best[3]} BH={best[4]} stages={best[5]} "
                  f"warps={best[6]}",
                  file=sys.stderr, flush=True)

        del tensors
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[rank {args.rank}] done, wrote {len(results)} winners to {args.out}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
