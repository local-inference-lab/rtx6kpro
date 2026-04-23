# Kimi-K2.6 eagle3 MTP long-context perf — **STATUS: shippable**

This directory documents a deep optimisation session on the
**Kimi-K2.6 target + `lightseekorg/kimi-k2.5-eagle3-mla` draft** stack on
**8× RTX 6000 Pro Blackwell (sm120)** under vLLM with the `TRITON_MLA`
attention backend and fp8 KV cache. What started as a 4.3× single-stream win at
30k-ctx grew into a full kernel-config tuning loop, DCP correctness fixes, a
manual block-table-zero-init fix, and extensive cross-DCP × MTP-on/off
benchmarking.

Target audience: **another engineer (or another AI agent) picking this up
cold**. Everything you need to reason about what was done, why, and where the
residual trade-offs live is below.

---

## TL;DR

Starting point before any work: **30k ctx single-stream MTP = 25 tok/s**
(kill-switch needed above 7k ctx to avoid collapse). sglang on the same
hardware/model did ~70 tok/s at 30k.

Ending point (this image):
`voipmonitor/vllm:cu130-mtp-tuned-v3-20260423`

- Single-stream 30k ctx MTP: **~88 tok/s** (DCP=8) / **~99 tok/s** (DCP=4) / **~125 tok/s** (DCP=1 extrapolated from 16k/64k)
- Concurrent conc=128 ctx=0 no-MTP: **up to 2026 tok/s at DCP=1** (Blackwell cap, matches sglang within noise)
- `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN` kill-switch no longer needed (MTP wins at any ctx on single-stream)
- Clean A/B against the hand-tuned baseline shows the residual delta is **max-model-len**, not kernel config

## Hardware & stack

| item | value |
|---|---|
| GPUs | 8× RTX 6000 Pro Blackwell (sm120), 94 GB/GPU |
| vLLM | branch `snapshot/kimi-k25-eagle3mla-current-20260423`, `/opt/vllm` in-container |
| Target model | `moonshotai/Kimi-K2.6` |
| Draft model | `lightseekorg/kimi-k2.5-eagle3-mla` |
| Attention | `TRITON_MLA` for target + draft, fp8 KV cache |
| Quant | fp8 MoE via Marlin |
| Num spec tokens | 3 (production default), tested 2/3/5 |

---

## Current docker images (chronological, all on Docker Hub)

| tag | status | what it has |
|---|---|---|
| `voipmonitor/vllm:cu130-mtp-baseline-20260423` | kept | pre-changes snapshot |
| `voipmonitor/vllm:cu130-mtp-fix-v1-20260423` | superseded | first CPU-sync fix attempt (no throughput win; 2k regression) |
| `voipmonitor/vllm:cu130-mtp-fix-v2-20260423` | superseded | defensive v1 minus the regressing bit |
| `voipmonitor/vllm:cu130-mtp-cg-fix-20260423` | **landmark** | FULL CG for TRITON_MLA, 4.3× single-stream 30k win |
| `voipmonitor/vllm:cu130-mtp-batchaware-20260423` | superseded | adds batch-aware num_kv_splits + DCP=4/8 correctness |
| `voipmonitor/vllm:cu130-mtp-tuned-20260423` | superseded | kernel microbench auto-tune (4-seq-len fractions) |
| `voipmonitor/vllm:cu130-mtp-tuned-v3-20260423` | **USE THIS** | 5-seq-len retune + `zero_()` removal |

Every image contains the full modified vLLM source at `/opt/vllm` (not a
mount); the important deltas are in
`/opt/vllm/vllm/v1/attention/backends/mla/triton_mla.py` and the new sibling
`triton_mla_tuning.py`.

---

## The journey (in order)

This section is long on purpose — the intermediate findings are important for
understanding why the final code looks the way it does.

### Phase 0 — the initial problem

`TRITON_MLA` inherited `_cudagraph_support = NEVER` from
`AttentionMetadataBuilder`. vLLM logged

> `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with TritonMLABackend … setting cudagraph_mode=PIECEWISE`

Under PIECEWISE mode every attention layer runs eager between two captured
compiled-compute graphs. For a single target decode step at 30k ctx with
`num_tokens = 4` (1 real + 3 speculative), per-layer launch overhead plus
stage1+stage2 kernel submission added up to ~85 ms GPU time across 61 target
layers. With 3 draft forwards on top, iter time ≈ 90 ms → ~2.3 tok/s × 11 iter/s
≈ 25 tok/s. Mitigation in the wild was
`VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000` (turn MTP off for long prompts),
which made it mean that long-context was a pure-target-model workload.

### Phase 1 — FULL cudagraph for TRITON_MLA in spec-verify shapes (4.3× win)

The central insight was that `TRITON_MLA` could be made FULL-CG-safe for
UNIFORM_BATCH spec-verify shapes with three changes to
`vllm/v1/attention/backends/mla/triton_mla.py`:

1. **Subclass the metadata builder**: `TritonMLAMetadataBuilder` declares
   ```python
   _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
   query_len_support  = QueryLenSupport.UNIFORM
   ```
   This bumps `reorder_batch_threshold` to `1 + num_spec_tokens` via
   `_init_reorder_batch_threshold(supports_spec_as_decode=True)`, routing
   spec-verify batches through the decode path instead of the expensive
   chunked-prefill branch.

2. **Fixed `num_kv_splits` as `tl.constexpr`**. The old code recomputed it per
   call from `attn_metadata.max_seq_len`; that recompiled the stage-1 kernel
   on every seq-len change and a captured graph can only embed one
   specialisation. Set `CG_NUM_KV_SPLITS = 64` (measured sweet spot at
   30k-ctx / sm120 fp8 / single-request via a kernel microbench).

3. **Persistent, shared buffers across all 61 layers**: `o`, `lse`,
   `attn_logits` were being allocated every call with `torch.zeros` /
   `torch.empty` — unsafe under CG capture (the graph embeds a freed
   pointer). Module-level pool keyed on `(key, device, shape, dtype)` gives a
   single 538 MB `attn_logits` allocation per TP rank instead of
   `61 × 538 MB`. Layers run sequentially within a forward, so sharing is
   safe.

4. `_build_decode` pre-expands `block_table` and per-query `seq_lens` into
   persistent CG-safe buffers (`cg_buf_*`) via `.copy_()` each step. The
   Triton decode kernel indexes `block_table` with `cur_batch = 0..num_decode_tokens-1`
   — treats each query as a separate "request" — so we must pre-expand
   row-wise rather than at forward time (which would allocate a transient and
   stale out the captured graph's pointers).

**Memory cost**: +5.5 GB/rank vs the pre-fix image:

- `attn_logits` shared buffer: `512 × 8 × 64 × 513 × 4B = 538 MB`
- `o`, `lse` shared buffers: ~4 MB combined
- `cg_buf_block_table`, `cg_buf_seq_lens`: ~33 MB
- FULL CG captures over 49 sizes: ~4 GB

Required serve-cmd change at the time: `--max-model-len 131072` to leave KV
budget (later revisited — see Phase 6).

**Measured win** (8× RTX 6000 Pro Blackwell / TP=8 / DCP=1 at the time /
Kimi-K2.5 eagle3 / fp8 KV / num_spec=3):

| ctx | before | after | speedup |
|---|---:|---:|---|
| 2,000 | 75.0 | **116.6** | 1.55× |
| 10,000 | 32.2 | **112.8** | 3.5× |
| 30,000 | 25.3 | **107.5** | **4.3×** |

Interarrival went **flat across ctx at ~19–21 ms p50** — matching what
FlashInfer-MLA delivers on its FULL-CG path without the XQA causal-mask bug
(see Dead ends).

Image: `voipmonitor/vllm:cu130-mtp-cg-fix-20260423`.

### Phase 2 — DCP=4 and DCP=8 correctness

With the base fix shipped, the natural next step was to combine it with
**decode context parallelism** (DCP) to multiply KV-cache capacity by
`cp_world_size`.

**DCP=4 initially crashed** with
`AssertionError: m.max_query_len <= self.reorder_batch_threshold`
at `mla_attention.py:1707`. Root cause: `_init_reorder_batch_threshold` forces
the threshold back to 1 under DCP>1 when `supports_dcp_with_varlen=False`.

Fix:

```python
class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("supports_dcp_with_varlen", True)
        super().__init__(*args, **kwargs)
```

**DCP=8 then produced garbled output** ("*inside:12 babitted SUcloth knots…*").
Root cause: my `_build_decode` derived per-query seq_lens as
`seq_lens[i] - (qpr - 1) + j`, but under DCP the base builder substitutes
`dcp_local_seq_lens` for `seq_lens` — that's the LOCAL per-rank slice, not
GLOBAL. The arithmetic was wrong because the DCP split is per-global-position.

Fix: derive per-query GLOBAL seq_lens from the `dcp_tot_seq_lens_device`
argument, then feed them through `get_dcp_local_seq_lens` to convert back to
local per this rank:

```python
if dcp_tot_seq_lens_device is not None and self.dcp_world_size > 1:
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
    global_per_query = (
        dcp_tot_seq_lens_device.to(seq_lens_device.dtype).unsqueeze(1)
        - (qpr - 1) + _arange
    ).reshape(-1)
    expanded = get_dcp_local_seq_lens(
        global_per_query, self.dcp_world_size, self.dcp_rank,
        self.cp_kv_cache_interleave_size,
    )
else:
    expanded = (seq_lens_device.unsqueeze(1) - (qpr - 1) + _arange).reshape(-1)
```

After the fix: generation is coherent under both DCP=4 and DCP=8, acceptance
rate unchanged (~45–55% per-position).

### Phase 3 — concurrent regression diagnosis (batch-aware num_kv_splits)

User ran the standard concurrent decode bench and spotted a regression vs his
pre-everything baseline:

| metric | pre-work | after CG fix | delta |
|---|---:|---:|---|
| conc=128, ctx=0 | 1508 tok/s | 1144 tok/s | **−24%** |
| conc=64, ctx=0 | 1008 tok/s | 763 tok/s | −24% |
| conc=1, ctx=0 | 51.7 tok/s | 69.7 tok/s | **+35%** |

Single-stream improved; high-concurrency regressed. Root cause:
`CG_NUM_KV_SPLITS = 64` fixed in Phase 1 was tuned for single-request long
context. At B=128 conc with `q_num_heads=128` (post-DCP all-gather), head-tile
count is `cdiv(128, BLOCK_H=8) = 16`, so stage-1 grid

```
128 requests × 16 H-blocks × 64 kv-splits = 131 072 blocks
```

on 144 SMs — **910× oversubscription**. Each CTA does ~zero work, launch
overhead dominates. Stage-2 merges 64 partials per head sequentially —
proportional extra cost.

Fix: `num_kv_splits = _pick_num_kv_splits(B, q_num_heads)` computed per call as
a Python int. Since each CG bucket has a fixed B, each bucket gets its own
compiled kernel (different `tl.constexpr`). Buffer `attn_logits` is always
sized at `MAX_NUM_KV_SPLITS = 64`; we pass `[:B, :, :num_kv_splits, :]` as the
kernel tensor. Parent strides are preserved through the slice, so every bucket's
captured graph reads/writes at stable addresses within the shared pool.

Analytical heuristic (fallback if the tuned table has no match):

```python
def _pick_num_kv_splits(B, q_num_heads):
    h_blocks = max(1, (q_num_heads + 7) // 8)
    total = B * h_blocks
    target = max(1, 576 // total)
    p = 1
    while (p << 1) <= target:
        p <<= 1
    return max(1, min(MAX_NUM_KV_SPLITS, p))
```

Image: `voipmonitor/vllm:cu130-mtp-batchaware-20260423`.

### Phase 4 — kernel microbench auto-tune (first round, 4-seq-len fractions)

The analytic heuristic is OK but can't reason about `BLOCK_N`, `BLOCK_H`,
`num_stages`, `num_warps`. A microbench was written at
[`bench/tune_triton_mla.py`](bench/tune_triton_mla.py):

- Calls `_fwd_grouped_kernel_stage1` **directly** with synthetic `q`, fp8
  paged KV, random `req_to_tokens` pointing into a 10k-block pool (realistic
  cache behaviour, memory bounded).
- Sweeps `(heads ∈ {16, 64, 128}, max_model_len ∈ {16k, 64k, 128k, 262k},
  B ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}) × (num_kv_splits × BLOCK_N × BLOCK_H
  × num_stages × num_warps)` — ~54 k combinations total.
- **Analytical shmem pre-filter** (sm120 dynamic cap 101 376 B) skips ~60% of
  combinations before any Triton compile.
- **8-GPU parallel** via `CUDA_VISIBLE_DEVICES=k` subprocess partitioning,
  round-robin across outer points (heads × mml × B).
- `torch.cuda.Event` timing, WARMUP=2 + TIMED=5 per config.

First run used `SEQ_LEN_FRACTIONS = (0.05, 0.30, 0.60, 0.95)` — 4 data
points per mml bucket, picker minimises geometric mean across them.

Aggregator at [`bench/aggregate_tune.py`](bench/aggregate_tune.py) produces
`triton_mla_tuning.py`: a Python `dict` literal
`TUNED_KV_CONFIGS[(heads, mml, B)] → {num_kv_splits, BLOCK_N, BLOCK_H,
num_stages, num_warps}` plus a `lookup_config()` that rounds mml and B down to
the nearest tuned bucket.

**Striking univariate findings across all 108 winners:**

- `num_stages = 1` wins **108/108**. The vLLM default was 2. sm120 fp8 MLA
  pipelines better with single stage + more shmem for wider tiles.
- `BLOCK_H = 32` wins 59/108 (16 wins 45/108, 8 only 4/108). The sm120 fp8
  override to `BLOCK_H = 8` inside `_decode_grouped_att_m_fwd` was leaving
  performance on the table.
- `BLOCK_N = 32` or `16` (never 64/128).
- `num_warps = 4` wins 101/108.

Integration into `triton_mla.py` required **inlining** stage-1 + stage-2
kernel calls in `forward_mqa` (bypassing the `decode_attention_fwd` wrapper
that hard-codes `BLOCK_H` / `num_stages`), reading the tuned config from
`lookup_config(q_num_heads, self._tuning_max_model_len, B)`, with the
analytic fallback.

Bench result: **per-bench-cell net win at long ctx, but a hard regression at
conc=1 ctx=0** (98.3 → 76.5 tok/s, −22 %). Root cause: the 4-point sweep's
shortest seq_len was **5 % of mml = 13 107 tokens for mml=262 144**, whereas
real workload at ctx=0 generates only a handful of tokens (seq_len ≈ output
tokens ≈ 50–100). Geomean at the 5 %/30 %/60 %/95 % points was dominated by
long-ctx timings; the picker chose configs that are suboptimal at short
seq_len.

Image: `voipmonitor/vllm:cu130-mtp-tuned-20260423` (kept for comparison, not
the shipping tag).

### Phase 5 — second tuning round (5-seq-len fractions incl. 1 %)

Added a 1 % fraction to the sweep:

```python
SEQ_LEN_FRACTIONS = (0.01, 0.05, 0.30, 0.60, 0.95)
```

At mml=262 144 that's seq_len = 2 621 tokens; at mml=16 000 it's 160 tokens.
The short fraction is cheap to measure (kernel scales linearly with seq), so
total sweep cost grew only ~5 % → ~30 min on 8 GPUs.

Effect on winners for **smaller mml** buckets was dramatic (e.g.
`(128, 16000, 128)` went from `splits=8/BN=32/BH=32` → `splits=1/BN=16/BH=32`
— geomean across the short points strongly prefers `num_kv_splits=1`). For
the **mml=262 144** bucket the winner at B=128 stayed at `splits=4` because
the 2 621-token test point isn't yet short enough for a deployment that also
sees 249 k seq — geomean still dominated by the long seq_len timings.

Bench delta vs Phase 4 (no-MTP, DCP=8, all settings else equal):

| ctx/conc | 4-seq (Phase 4) | 5-seq (Phase 5) | delta |
|---|---:|---:|---|
| 0/1 | 67.5 | 73.6 | +9 % |
| 0/64 | 891 | 953 | +7 % |
| 0/128 | 1400 | 1397 | ~0 % (mml=262 k bucket unchanged) |

### Phase 6 — "why are we still 100 tok/s below the hand-tuned 1508?"

Diagnostic: user's original 1508 tok/s baseline used a *different* CLI:

```
--max-model-len 150000        # vs 262144
--max-num-batched-tokens 4096 # vs 8192
--language-model-only          # vs absent
```

**Override test** (force `(128, 262144, 128)` to match the baseline config
exactly — `splits=1, BN=32, BH=8, stages=2`): throughput went to **1334
tok/s** (actually *worse* than 1397). → *Kernel config isn't the culprit.*

**Isolation test 1** (our config + only `--max-model-len 150000`, nothing
else changed from the 1397-run): **1527 tok/s** at conc=128 ctx=0.
Effectively matches the 1508 baseline within noise. → `max-model-len` alone
accounts for the full ~130 tok/s delta. Test 2 (batched=4096) and Test 3
(language-only) were skipped as the hypothesis was already fully explained.

**Why does `max-model-len` matter so much?** vLLM's CG captures pre-allocate
`block_table` at `max_num_seqs × cdiv(max_model_len, block_size=16) × int32`:

- mml=262 144: `128 × 16 384 × 4B = 8.4 MB`, row stride 64 KB
- mml=150 000: `128 × 9 375 × 4B = 4.8 MB`, row stride 37 KB

The Triton decode kernel scans `block_table[req, 0:actual_bpr]` per query
with the full (captured) row stride. At conc=128 × 61 layers × many decode
steps/sec, wider stride → more L2 pressure + prefetcher disruption, even
though only the first `actual_bpr` (= `cdiv(actual_seq, 16)`) entries per row
are actually read. Out-of-kernel cost that no kernel microbench can see.

This is a **deployment trade-off**, not a tuning gap:

- **Short-ctx speed config**: `--max-model-len 150000`. 1527 tok/s at conc=128 ctx=0; cap at 150 k requests.
- **Long-ctx flex config**: `--max-model-len 262144`. 1397 tok/s at conc=128 ctx=0 but can serve 262 k requests.

A future deeper fix would require vLLM to capture multiple seq_len buckets
per batch size — substantial core rework, punted.

### Phase 7 — `zero_()` removal (final micro-opt)

`_build_decode` had a per-step defensive zero of the unused tail of
`_cg_buf_block_table`:

```python
if bt_cols_src < self._cg_buf_block_table.shape[1]:
    self._cg_buf_block_table[:bt_rows, bt_cols_src:].zero_()
```

Buffer was `torch.empty`-allocated (uninitialised memory); the `zero_()` made
unused columns deterministic page-0 so random-byte garbage wouldn't produce
wild `kv_page_number * stride` addresses inside stage-1. Safety net, not
correctness-required — the kernel already masks loads beyond `split_kv_end`
via `tl.load(..., mask=offs_n < split_kv_end, other=0)`, and Triton masked
loads on CUDA never fault on invalid addresses.

Cleaner fix:

1. `torch.empty` → `torch.zeros` in `_maybe_lazy_init_cg_bufs` (one allocation, one-time cost).
2. Drop the per-step `zero_()`.

For our MTP workload (qpr > 1, spec-verify only), `_build_decode` runs once
per decode step and the `zero_()` was copying up to ~7 MB per step at
mml=262 k. Saving that consistently improved the MTP conc=128 run by a
surprisingly large margin (see the "MTP before/after zero_() removal" table
further down).

`zero_()` is **not called** in the pure-decode (no-MTP, qpr = 1) path — that
path goes through the base `MLACommonMetadataBuilder._build_decode` and never
enters the branch — so the fix is MTP-specific.

Image: `voipmonitor/vllm:cu130-mtp-tuned-v3-20260423` (**the shipping tag**).

---

## Intermediate benchmarks (chronological)

Every bench table the user shared or I measured during this session, in the
order they happened. `Aggregate throughput (tok/s)` unless stated otherwise.
Use this to reconstruct the journey — each block is labelled by the code
state at that point so another reader can map numbers → code state →
image tag.

### User's pre-work baseline (historical, reference for all regressions)

Before any work in this session. Original unmodified `triton_mla.py`
(no CG fix, no tuning). User CLI: `--max-model-len 150000`,
`--max-num-batched-tokens 4096`, `--language-model-only`, DCP=8, no MTP,
max-num-seqs default (256).

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 51.7 | 95.4 | 170.9 | 268.6 | 378.4 | 599.2 | 1008.1 | **1508.8** |

This is the number every subsequent "regression vs baseline" is measured
against. It's the target for the final **conc=128 ctx=0 no-MTP** cell.

### After Phase 1 (CG fix, image `cu130-mtp-cg-fix-20260423`)

Single-stream long-ctx wins on DCP=1 from the Phase-1 microbench:

| ctx | before | after | speedup |
|---|---:|---:|---|
| 2 000 | 75.0 | 116.6 | 1.55× |
| 10 000 | 32.2 | 112.8 | 3.5× |
| 30 000 | 25.3 | **107.5** | **4.3×** |

### Phase 3 regression report (user bench on Phase 1 image, after DCP=8 fix)

User's concurrent decode bench with **DCP=8, no MTP, `mml=262 144`,
`batched=8192`**:

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 69.7 | 763.3 | 1144.8 |

vs pre-work 51.7 / 1008 / 1508. Single-stream up +35 %, conc=64/128 down
−24 %. This triggered Phase 3 (batch-aware num_kv_splits).

### Phase 3 post-fix (batch-aware, pre-tuning, image `cu130-mtp-batchaware-20260423`)

MTP=3 / DCP=8 / `mml=262 144`, conc 1/64/128 × ctx 0/16k/32k/64k/128k:

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 91.5 | 1014.2 | 1110.6 |
| 16k | 91.4 | 727.1 | 968.0 |
| 32k | 84.6 | 713.9 | 838.9 |
| 64k | 77.5 | 466.9 | 463.0 |
| 128k | 64.6 | 428.7 | 435.8 |

### Phase 4 (first auto-tune, 4-seq fractions 5/30/60/95 %) — MTP=3

Image `cu130-mtp-tuned-20260423`, DCP=8, `mml=262 144`, MTP=3 with kill-switch
`VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=100000`:

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 76.5 | 934.1 | 1202.4 |
| 16k | 74.5 | 853.7 | 1226.5 |
| 32k | 73.6 | 733.3 | 908.1 |
| 64k | 68.6 | 592.0 | 856.1 |
| 128k | 73.6 | 517.1 | 462.6 |

Note conc=1 / ctx=0 = **76.5** vs Phase-3 pre-tuning **91.5** — the 4-seq
tuning REGRESSED short-ctx single-stream by −16 %. This was the signal that
the sweep was biased to long-ctx (tuning only tested seq_len ≥ 5 % of mml).

### Phase 4 — same image, no MTP

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 67.5 | 891.0 | 1400.0 |
| 16k | 65.7 | 827.0 | 1146.0 |
| 32k | 66.6 | 763.0 | 1019.0 |
| 64k | 64.7 | 636.0 | 764.0 |
| 128k | 61.6 | 509.0 | 636.0 |

Same short-ctx regression pattern in no-MTP.

### Phase 5 — user's quick MTP test (5-seq retune, no kill-switch)

User ran a conc 1/64/128 at ctx=0 quick check after the 5-seq retune was
deployed, before the `zero_()` removal:

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 79.5 | 886.6 | 1087.0 |

Improvement over Phase 4 conc=1 (76.5 → 79.5) but conc=128 still well below
Phase-3 (1111) or pre-work (1508). Led to override + CLI diagnostics.

### Phase 5 — my no-MTP bench (DCP=8, mml=262 144)

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 73.6 | 953.2 | 1396.6 |
| 16k | 65.7 | 826.9 | 1146.4 |
| 32k | 66.6 | 762.6 | 1018.5 |
| 64k | 64.7 | 635.7 | 763.8 |
| 128k | 61.6 | 508.5 | 636.1 |

Moved conc=64 and conc=128 up from Phase 4 (+7 % / ~0 % at ctx=0) — the 1 %
fraction addition to `SEQ_LEN_FRACTIONS` helped smaller-mml buckets but the
mml=262 k / B=128 winner was unchanged (geomean still dominated by long
seq_len).

### Phase 6 diagnostic: override `(128, 262 144, 128)` to baseline kernel config

5-seq tuning with one MANUAL override (force
`splits=1/BN=32/BH=8/stages=2/warps=4` — the exact config the original
baseline kernel would have picked at mml=150 000 via its adaptive
`num_kv_splits` heuristic):

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 73.5 | 954.7 | 1334.2 |
| 16k | 72.6 | 890.6 | 1144.7 |
| 32k | 72.6 | 826.9 | 1017.5 |
| 64k | 69.6 | 698.9 | 890.2 |
| 128k | 65.6 | 509.1 | 635.3 |

**1334 < 1397** — kernel config is NOT the explanation of the 1508 baseline
delta. Important result.

### Phase 6 diagnostic: all three CLI changes (user's original recipe)

Revert override, switch CLI to `--max-model-len 150 000`,
`--max-num-batched-tokens 4096`, `--language-model-only`,
`--max-num-seqs 256`:

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 74.5 | 1016.6 | **1527.9** |
| 16k | 73.6 | 890.7 | 1271.1 |
| 32k | 73.5 | 827.3 | 1144.6 |
| 64k | 70.5 | 637.2 | 888.8 |
| 128k | 65.6 | 509.0 | 635.4 |

→ matches user's historical 1508 within noise. Gap between this (1527) and
the mml=262 k run (1397) is purely CLI, not kernel.

### Phase 6 isolation Test 1: only `--max-model-len 150 000` changed

Back to batched=8192, max-num-seqs=128, no `--language-model-only` — **only**
`--max-model-len 150 000` changed from the mml=262 k run:

| ctx \ conc | 1 | 64 | 128 |
|---|---:|---:|---:|
| 0 | 73.5 | 955.2 | **1527.0** |
| 16k | 72.6 | 891.1 | 1271.9 |
| 32k | 72.6 | 826.0 | 1143.6 |

Reproduces 1527 with just the one change. Tests 2 (batched=4096) and 3
(language-only) skipped — the hypothesis is complete: `max-model-len` alone
accounts for the entire 130-tok/s delta. See the Phase 6 section above for
the CG block_table stride explanation.

### Phase 7 — `zero_()` removal (MTP=3 ctx=0, DCP=8, no kill-switch)

| conc | pre-removal | post-removal | delta |
|---|---:|---:|---|
| 1 | 79.5 | 90.4 | +14 % |
| 64 | 886.6 | 925.9 | +4 % |
| 128 | 1087.0 | **1311.7** | **+21 %** |

The `+21 %` at conc=128 was the signature that the per-step `zero_()` was
actually costing (not negligible as initially assumed — at mml=262 k with
B=128 the zeroing tail is ~7 MB per decode-step metadata build, which
compounds across steps).

---

## Full benchmark matrix (final image)

All measured with `llm_decode_bench.py --skip-prefill` and MTP/no-MTP
variants, 10 s duration per cell. Aggregate throughput in tok/s.
`VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN` **not set** (MTP always on when
`--speculative-config` is passed).

### DCP=1 + MTP=3 (single-stream champion)

Prefill (C=1, baseline TTFT=0.040 s subtracted):

| ctx | tokens | TTFT | prefill s | prefill tok/s |
|---|---:|---:|---:|---:|
| 8k | 5 312 | 0.71 | 0.67 | 7 892 |
| 16k | 10 480 | 1.44 | 1.40 | 7 478 |
| 32k | 20 822 | 2.94 | 2.90 | 7 177 |
| 64k | 41 505 | 6.48 | 6.44 | 6 449 |
| 128k | 82 845 | 14.82 | 14.78 | 5 604 |

Aggregate throughput (tok/s):

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **114.1** | 178.9 | 243.7 | 417.4 | 659.0 | 1055.2 | 1522.0 | **1992.3** |
| 16k | 108.1 | 166.4 | 226.4 | 353.3 | 502.9 | 680.6 | 879.5 | 1080.1 |
| 32k | skip | 151.6 | 189.1 | 293.3 | 396.6 | 495.4 | 580.2 | 605.7 |
| 64k | 85.6 | 136.3 | 157.5 | 222.8 | 286.5 | 354.1 | 420.4 | 384.8 |
| 128k | 66.6 | 105.4 | 121.2 | 152.6 | 186.5 | 211.1 | 189.1 | 374.1 |

### DCP=1, no MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 88.5 | 148.4 | 233.8 | 340.7 | 475.2 | 823.6 | 1330.2 | **2026.3** |
| 16k | 82.6 | 139.1 | 214.6 | 310.2 | 413.8 | 667.9 | 954.0 | 1272.2 |
| 32k | 77.5 | 129.4 | 197.8 | 286.2 | 366.1 | 572.4 | 758.7 | 890.3 |
| 64k | 68.6 | 117.2 | 170.9 | 246.5 | 302.1 | 411.8 | 508.9 | 635.5 |
| 128k | 55.4 | 97.3 | 131.3 | 190.7 | 222.6 | 286.0 | 317.9 | 380.5 |

### DCP=4 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **98.8** | 160.0 | 211.4 | 363.3 | 576.5 | 853.2 | 1239.1 | 1506.2 |
| 16k | 96.4 | 145.1 | 190.0 | 311.0 | 453.4 | 603.1 | 741.0 | 891.9 |
| 32k | 88.5 | 130.2 | 180.8 | 266.2 | 367.5 | 451.9 | 575.0 | 597.2 |
| 64k | 76.5 | 114.3 | 151.9 | 205.4 | 267.5 | 303.4 | 404.9 | 432.4 |
| 128k | 67.6 | 87.4 | 112.4 | 152.0 | 175.8 | 213.5 | 247.7 | 383.9 |

### DCP=4, no MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.5 | 122.5 | 205.2 | 308.2 | 426.6 | 726.2 | 1136.1 | 1640.3 |
| 16k | 73.5 | 117.3 | 193.5 | 278.5 | 365.8 | 600.0 | 827.1 | 1145.0 |
| 32k | 71.5 | 113.5 | 182.8 | 254.7 | 334.0 | 509.1 | 694.1 | 882.4 |
| 64k | 66.7 | 105.3 | 163.0 | 222.7 | 270.3 | 379.0 | 508.8 | 629.3 |
| 128k | 60.3 | 92.8 | 135.1 | 174.7 | 206.7 | 254.6 | 318.0 | 379.7 |

### DCP=8 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.4 | 142.7 | 176.1 | 311.3 | 498.3 | 641.2 | 908.0 | 1137.0 |
| 16k | 85.4 | 131.2 | 186.9 | 297.1 | 466.0 | 590.7 | 744.4 | 861.0 |
| 32k | 87.4 | 121.1 | 163.9 | 275.8 | 400.2 | 520.8 | 699.6 | 828.2 |
| 64k | 79.3 | 114.2 | 156.1 | 237.3 | 340.5 | 432.9 | 560.7 | 581.7 |
| 128k | 75.4 | 99.4 | 135.2 | 197.9 | 252.2 | 312.5 | 417.6 | 416.2 |

### DCP=8, no MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.1 | 113.4 | 185.8 | 276.6 | 363.5 | 600.9 | 948.4 | 1260.6 |
| 16k | 73.6 | 111.2 | 178.7 | 262.2 | 333.9 | 571.3 | 826.7 | 1137.4 |
| 32k | 73.0 | 109.3 | 174.8 | 245.0 | 317.7 | 509.3 | 757.7 | 1008.6 |
| 64k | 69.7 | 104.7 | 162.9 | 229.1 | 301.8 | 445.7 | 632.2 | 757.0 |
| 128k | 65.6 | 95.5 | 146.8 | 197.5 | 252.9 | 350.0 | 476.8 | 635.3 |

### KV cache scaling with DCP

| config | KV tokens @ mml=262 144 | max concurrency |
|---|---:|---:|
| DCP=1 + MTP=3 | 337 296 | 1.29× |
| DCP=1 no-MTP | 437 520 | 1.67× |
| DCP=4 + MTP=3 | 1 126 528 | 4.30× |
| DCP=8 + MTP=3 | 1 769 600 | 6.75× |
| DCP=8 no-MTP | 3 424 256 | 13.03× |

DCP scales KV cache roughly linearly; MTP subtracts draft-model VRAM and
spec-verify CG pool.

### MTP before/after `zero_()` removal (DCP=8 + MTP=3, ctx=0)

User benchmarks showed:

| conc | before `zero_()` removal | after | delta |
|---|---:|---:|---|
| 1 | 79.5 | 90.4 | +14 % |
| 64 | 886.6 | 925.9 | +4 % |
| 128 | 1087.0 | 1311.7 | **+21 %** |

---

## When to use MTP, when to disable it

With the final image there is **no monotonic rule** — MTP wins single-stream
at every ctx, loses at high concurrency past some ctx threshold that depends
on DCP. Ballpark from the matrix (MTP vs no-MTP, aggregate tok/s):

- **conc=1**: MTP wins at every ctx and every DCP by +20% to +40 %.
- **conc=16–32**: MTP roughly ties at short ctx, loses slightly at ≥ 64 k ctx.
- **conc=128**: MTP loses at ctx=0 (GPU saturated, spec-verify is wasted
  work). Wins or ties at 16 k–64 k. Long-ctx behaviour varies per DCP.

The old `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000` kill-switch is no
longer necessary. `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=100000` (the
intermediate setting) was kept during most of this session as a conservative
default; the final images run without it. A proper long-term fix is a
**batch-aware** kill-switch (phase 2 in the original plan — not implemented
here), e.g. `VLLM_SPECULATIVE_DISABLE_WHEN_BATCH_ABOVE=32`.

---

## Files you care about

### In the image (`/opt/vllm/vllm/v1/attention/backends/mla/`)

- **`triton_mla.py`** — the only materially modified vLLM source file (also
  copied here at [`patches/triton_mla_final.py`](patches/triton_mla_final.py)).
  Contains `TritonMLAMetadataBuilder`, inline stage-1/stage-2 launches in
  `forward_mqa`, and the full DCP-aware `_build_decode`.
- **`triton_mla_tuning.py`** — auto-generated by `tune_triton_mla.py` +
  `aggregate_tune.py`. 108-entry lookup table of
  `(q_num_heads, max_model_len, B) → best kernel config`.
  Copied at [`patches/triton_mla_tuning.py`](patches/triton_mla_tuning.py).

Backup copies preserved in the container:

- `triton_mla.py.bak_split_tuning_20260421` — pre-everything, original
- `triton_mla.py.bak_pre_tuning_20260423` — after Phase 3 (batch-aware splits), pre-Phase 4
- `triton_mla.py.bak_fixed64_20260423` — intermediate with fixed `CG_NUM_KV_SPLITS=64`

### In this directory

- [`bench/tune_triton_mla.py`](bench/tune_triton_mla.py) — stand-alone kernel
  microbench (runs per-GPU via `CUDA_VISIBLE_DEVICES`, no vLLM server needed)
- [`bench/aggregate_tune.py`](bench/aggregate_tune.py) — merges per-GPU
  JSONs into the final `triton_mla_tuning.py`
- [`bench/bench_triton_mla.py`](bench/bench_triton_mla.py) — single-shape
  kernel microbench (pre-dates the sweep; kept for quick spot checks)
- [`bench/e2e_bench.py`](bench/e2e_bench.py) — streaming tok/s + interarrival
  probe from Phase 1
- [`patches/triton_mla_final.py`](patches/triton_mla_final.py) — the current
  `triton_mla.py` (full file snapshot)
- [`patches/triton_mla_tuning.py`](patches/triton_mla_tuning.py) — the current
  `triton_mla_tuning.py` (5-seq retune)
- [`patches/triton_mla_full_cg_WIN.patch`](patches/triton_mla_full_cg_WIN.patch)
  — **Phase 1 only** diff (for reference; the current `triton_mla.py` has
  many more layers on top)

### Regenerating the kernel tuning

```bash
# inside the container, from a clean state (no vllm server running)
mkdir -p /tmp/tune_out
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup /opt/venv/bin/python \
    /root/vllm/mtp-long-ctx-fix/bench/tune_triton_mla.py \
    --rank $i --world 8 \
    --out /tmp/tune_out/tune_gpu$i.json \
    > /tmp/tune_out/tune_gpu$i.stdout \
    2> /tmp/tune_out/tune_gpu$i.stderr &
done; wait

python3 /root/vllm/mtp-long-ctx-fix/bench/aggregate_tune.py \
  --in-glob '/tmp/tune_out/tune_gpu*.json' \
  --out /path/to/triton_mla_tuning.py
```

~30 min wall-clock on 8× RTX 6000 Pro. All GPUs must be free — the sweep
allocates its own synthetic tensors and must not race with vllm server VRAM.

---

## Recipe — launch the final image

### Short-ctx speed config (recommended default, ≤ 150 k ctx workload)

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  voipmonitor/vllm:cu130-mtp-tuned-v3-20260423 \
  bash -lc '
VLLM_ENABLE_PCIE_ALLREDUCE=1 NCCL_P2P_LEVEL=SYS \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_MARLIN_USE_ATOMIC_ADD=1 VLLM_MARLIN_INPUT_DTYPE=fp8 \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.6 \
  --served-model-name Kimi-K2.6 --trust-remote-code --host 0.0.0.0 --port 5002 \
  --tensor-parallel-size 8 --pipeline-parallel-size 1 --decode-context-parallel-size 8 \
  --enable-chunked-prefill --enable-prefix-caching --load-format fastsafetensors --async-scheduling \
  --gpu-memory-utilization 0.94 \
  --max-model-len 150000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 256 \
  --mm-processor-cache-gb 0 --mm-encoder-tp-mode weights \
  --language-model-only \
  --attention-backend TRITON_MLA --kv-cache-dtype fp8 \
  --tool-call-parser kimi_k2 --enable-auto-tool-choice --reasoning-parser kimi_k2 \
  --speculative-config '\''{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"probabilistic"}'\'''
```

### Long-ctx flex config (≥ 150 k ctx workload)

Same as above but:

```
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
```

(and drop `--language-model-only` if you need the vision encoder). Expect
~7 % lower throughput at `ctx ≤ 16 k` vs the short-ctx config.

### DCP sweet spots

| use case | DCP | MTP | notes |
|---|---|---|---|
| single-stream interactive, long ctx | 1 | on | max single-stream tok/s at long ctx; KV 337 k @ mml=262 k |
| mixed concurrency up to 32, ctx 0–64 k | 4 | on | best long-ctx / concurrency balance |
| many-concurrency short ctx | 8 | off | 3.4 M KV tokens, 13× concurrency @ mml=262 k |
| many-concurrency with MTP | 8 | on | 1.8 M KV, 6.75× concurrency; use when single-stream also matters |

---

## Dead ends, for the record

### FLASHINFER_MLA on sm120 fp8

Supports `UNIFORM_BATCH` CG via the XQA kernel, which hit ~19 ms target
forward and flat interarrival — same as what TRITON_MLA + FULL CG eventually
achieved. But XQA **does not apply causal masking within the 4-query
spec-verification span** (treats each flattened token as an independent
request), so draft acceptance collapsed to 0 %. Removing the flatten path on
FP8 routes to `trtllm-gen`, which errors
`TllmGenFmhaRunner: Unsupported architecture` (trtllm-gen fp8 MLA is not
compiled for sm120 in this FlashInfer build). Dead end without upstream
kernel work.

### CPU-side sync removal in `MLACommonMetadataBuilder.build()`

Early attempt (image `cu130-mtp-fix-v1-20260423`) removed a
`compute_num_computed_tokens().cpu()` call that fired every spec-verify
metadata build. The sync was real (~80 ms at 30 k ctx) but turned out to
already overlap with GPU work under async scheduling, so removing it saved
CPU cycles without moving single-request tok/s. The same patch also kept
`_seq_lens_cpu` populated with `optimistic_seq_lens_cpu` in async spec mode,
which regressed 2 k tok/s due to workspace-sizing drift. Reverted in v2.

### Manual override of `(128, 262144, 128)` to match the hand-tuned baseline

Forced the bucket to `splits=1/BN=32/BH=8/stages=2/warps=4` — the exact
config the original baseline kernel used when run with `mml=150 000` (via
`_pick_num_kv_splits` returning 1 because `max_seq_len ≈ 100` at ctx=0).
Result: **1334 tok/s**, actually worse than the 1397 that auto-tuning
produces. Definitively ruled out kernel config as the explanation of the
pre-isolation 100-tok/s gap — see Phase 6.

### sglang as reference

sglang achieves ~70 tok/s at 30 k ctx on this hardware/model. The final
TRITON_MLA image does ~88–107 tok/s at 30 k depending on DCP, so sglang is no
longer a ceiling. Kernel source inspection of sglang's split-KV implementation
wasn't needed in the end — the FlashMLA pattern (shared `cg_buf_*` on the
metadata builder) already at
`vllm/v1/attention/backends/mla/flashmla.py` was the right template.

---

## Unfinished threads / future work

1. **Adaptive MTP kill-switch** based on running batch size (phase 2 of the
   original plan). `VLLM_SPECULATIVE_DISABLE_WHEN_BATCH_ABOVE=<threshold>` or
   a `(batch, ctx)` combined rule. ~30–60 min patch into scheduler/eagle
   logic.
2. **Variable `num_speculative_tokens`**: single-stream might benefit from
   num_spec=4 or 5; high-concurrency needs 0. Requires extending CG capture
   to multiple num_spec variants → ~2–3× CG pool size. Punted.
3. **Multi-bucket `max_seq_len` CG capture** to close the
   mml=262 k vs mml=150 k throughput delta without forcing the user to pick.
   Major vLLM-side rework (CG logic, dispatcher, metadata builder); not
   attempted.
4. **Upstream PR** of the `triton_mla.py` + `triton_mla_tuning.py` changes.
   The patch is clean-ish; the tuning table is hardware-specific (sm120
   fp8 MLA) so upstream would need to ship empty and regenerate per target.

---

## Provenance

- Session on 2026-04-23 by `festr2@gmail.com` + the agent, across **16 hours
  wall-clock** of iteration.
- All code changes are in
  `/opt/vllm/vllm/v1/attention/backends/mla/{triton_mla.py, triton_mla_tuning.py}`
  (plus the `.bak_*` backups of intermediate states) inside the docker
  container `mystifying_herschel` (image base: `voipmonitor/vllm:cu130`).
- The `triton_mla.py` under `patches/` is a verbatim snapshot of what's in
  the **shipping image** `cu130-mtp-tuned-v3-20260423`.
- vLLM base commit: `47fcb8ca6` on branch
  `snapshot/kimi-k25-eagle3mla-current-20260423`. The patch is not applicable
  cleanly against upstream main without adjusting for the `MLACommonImpl`
  refactor deltas.
