# MTP long-context debugging: FIXED

## TL;DR — 4.3× speedup at 30k ctx

Enabling FULL CUDA graph capture for `TRITON_MLA` in spec-verification
shapes turned the 30k-context MTP workload from **25 tok/s** to **107.5 tok/s**
on 8× RTX 6000 Pro Blackwell / Kimi-K2.5-eagle3-mla / fp8 KV — beating
sglang's ~70 tok/s on the same hardware / model.

| ctx    | before | after   | speedup |
|--------|-------:|--------:|---------|
| 2 000  |  75.0  | **116.6** | 1.55× |
| 10 000 |  32.2  | **112.8** | 3.5×  |
| 30 000 |  25.3  | **107.5** | **4.3×** |

Interarrival is now **flat across ctx at ~19–21 ms p50**, matching what
FlashInfer-MLA delivers on its FULL-CG path. The only benefit FlashInfer had
over TRITON_MLA on sm120 fp8 was FULL cudagraph capture; TRITON_MLA's
correctness with eagle3 spec verify is already good (45–55 % per-position
acceptance).

**Docker image**:
`voipmonitor/vllm:cu130-mtp-cg-fix-20260423`

**Patch**: [`patches/triton_mla_full_cg_WIN.patch`](patches/triton_mla_full_cg_WIN.patch)
against vLLM `47fcb8ca6` (or the parent commit).

---

## What changed, and why it works

### Before

`TRITON_MLA` inherited `_cudagraph_support = NEVER` from `AttentionMetadataBuilder`.
vLLM logged

> `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with TritonMLABackend … setting cudagraph_mode=PIECEWISE`

and PIECEWISE-only mode means every layer's attention runs eager between two
captured compiled-compute graphs. For a single target decode step at 30k ctx
with `num_tokens=4` (1 real + 3 speculative), the per-layer launch overhead
and stage-1 + stage-2 kernel submission added up to ~85 ms GPU time across
61 target layers. With 3 draft forwards on top, iter time ≈ 90 ms →
~2.3 tok/s × 11 iter/s ≈ 25 tok/s.

### Root cause of the old NEVER

`forward_mqa` had three shape-dependent / allocation-per-call behaviours that
made it incompatible with graph capture:

1. **`num_kv_splits` chosen at runtime** from `attn_metadata.max_seq_len`.
   It's a `tl.constexpr` in `_fwd_grouped_kernel_stage1`, so each value
   produces a different compiled kernel and a differently-sized stage-1
   temporary. A captured graph can only embed one.
2. **`o`, `lse`, `attn_logits` allocated every call** with `torch.zeros` /
   `torch.empty`. Allocation during capture produces a fresh tensor address
   that the graph embeds, but at replay that memory has been freed → UB.
3. **UNIFORM spec-verify shape not supported.** With `query_len_support =
   SINGLE_ONLY` and `reorder_batch_threshold = 1`, any spec-verify batch
   (max_query_len > 1) was routed to the prefill branch and never went
   through `forward_mqa` at all — so it ran eager + prefill-kernel every
   decode step.

### The fix

All in `vllm/v1/attention/backends/mla/triton_mla.py` — see
`patches/triton_mla_full_cg_WIN.patch`:

1. New subclass `TritonMLAMetadataBuilder(MLACommonMetadataBuilder)` with

       _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
       query_len_support  = QueryLenSupport.UNIFORM

   This does two things: it tells vLLM it's safe to capture FULL CG at any
   batch size, and it bumps `reorder_batch_threshold` to `1 + num_spec_tokens`
   via `_init_reorder_batch_threshold(supports_spec_as_decode=True)`, so
   spec-verify batches go through the decode path instead of the expensive
   chunked-prefill branch.

2. `_build_decode` override that pre-expands `block_table` and per-query
   `seq_lens` into **persistent CG-safe buffers** (`cg_buf_block_table`,
   `cg_buf_seq_lens`) via `.copy_()`. Needed because the Triton decode
   kernel does `cur_batch_req_idx = cur_batch` — it treats each query as a
   separate "request" — so we pre-expand row-wise rather than expanding at
   forward time (which would allocate a fresh transient and stale out the
   captured graph's data pointer).

3. `forward_mqa` uses a **fixed `CG_NUM_KV_SPLITS = 64`** (measured sweet
   spot at 30k ctx on sm120 fp8 from `bench/bench_triton_mla.py`). Slightly
   suboptimal at very short ctx, but the flat-graph replay wins dominate.

4. Module-level **shared persistent output buffers** keyed by
   `(device, shape, dtype)` for `o`, `lse`, `attn_logits`. Sharing across
   all 61 target layers + 1 draft layer turns `61 × 538 MB` of attn_logits
   into a single `538 MB` allocation per TP rank. Layers run sequentially,
   so sharing is safe.

5. Persistent buffers allocated once to the max CG capture size
   (`min(max_num_seqs * (1 + num_spec_tokens), max_cudagraph_capture_size)`),
   so every captured shape uses a slice of the same underlying storage —
   required for CUDA-graph replay to read/write at stable addresses.

### Memory impact

+5.5 GB per TP rank beyond the pre-fix image:
- `attn_logits` shared buffer: 512 × 8 × 64 × 513 × 4 B = **538 MB**
- `o`, `lse` shared buffers: ~4 MB combined
- `cg_buf_block_table`, `cg_buf_seq_lens` in builder: ~33 MB
- FULL CG captures over 49 sizes: ~4 GB

On this 94 GB/GPU box the original config with `max_model_len=262144` no
longer fits; the demo recipe now uses `--max-model-len 131072` (comfortable
for 30k-ctx workloads). Either that, or drop `--gpu-memory-utilization` from
0.90 to ~0.86 would also free enough. Keeping `max_model_len` default at
`131072` is the safer recommendation.

### Correctness

Draft acceptance verified against pre-fix baseline: 45–55 % per-position
acceptance (same as before), mean acceptance length 2.0–2.68 tokens. Sample
generation produces coherent text (verified with "write a haiku about
speculative decoding" → correct reasoning).

The `block_table` expansion preserves causal semantics because every query
of request `i` reads from a duplicated row of the same `(num_blocks)`
physical page list, and the per-query seq_lens we derive
(`seq_lens[i] − (qpr − 1) + j`) give each query exactly its causal KV
prefix.

---

## Reproduction

Launch:

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  voipmonitor/vllm:cu130-mtp-cg-fix-20260423 \
  bash -lc '
VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=100000 \
VLLM_ENABLE_PCIE_ALLREDUCE=1 \
NCCL_P2P_LEVEL=SYS \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
VLLM_MARLIN_INPUT_DTYPE=fp8 \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.6 \
  --served-model-name Kimi-K2.5 \
  --trust-remote-code --host 0.0.0.0 --port 5002 \
  --tensor-parallel-size 8 --pipeline-parallel-size 1 \
  --enable-chunked-prefill --enable-prefix-caching \
  --load-format fastsafetensors --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 --max-num-seqs 128 \
  --max-model-len 131072 \
  --attention-backend TRITON_MLA --kv-cache-dtype fp8 \
  --tool-call-parser kimi_k2 --enable-auto-tool-choice --reasoning-parser kimi_k2 \
  --speculative-config '\''{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"probabilistic"}'\'''
```

Compared to the pre-fix recipe:
- **removed** `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (no longer needed)
- **removed** `VLLM_LOG_STATS_INTERVAL=1` (just for debugging, not required)
- **changed** `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000` → `100000` (MTP
  now pays off at any reasonable context)
- **added** `--max-model-len 131072` (FULL CG captures eat ~4 GB of KV cache
  budget; bump `--gpu-memory-utilization` if you need 262k)

Benchmark:

```bash
python /root/vllm/mtp-long-ctx-fix/bench/e2e_bench.py \
  --port 5002 -c 30000 --max-tokens 300 --label my-test
```

Expected: ~107 tok/s at 30k ctx, ~116 tok/s at 2k ctx.

---

## History / dead ends for the record

### Before the CG fix

First attempt (`cu130-mtp-fix-v1-20260423` image, now superseded): tried to
remove a `compute_num_computed_tokens().cpu()` sync in
`MLACommonMetadataBuilder.build()` that fired on every spec-verify metadata
build and stalled on pending target GPU work. The sync was real (~80 ms at
30k ctx) but turned out to already overlap with unrelated GPU work under
async scheduling, so removing it saved CPU cycles without moving
single-request tok/s. That patch also had an overeager
`gpu_model_runner.py` change (keep `_seq_lens_cpu` populated with
`optimistic_seq_lens_cpu` in async spec mode) which regressed 2k tok/s due
to workspace-sizing drift; reverted in v2.

### FLASHINFER_MLA path (not used)

FlashInfer-MLA supports `UNIFORM_BATCH` CG on sm120 FP8 via the XQA
kernel, which gave target forward ~19 ms and flat interarrival across ctx —
exactly what we see now with the CG fix. But XQA does not apply causal
masking within the 4-query spec-verification span (it treats each flattened
token as an independent request), so draft acceptance dropped to 0 %.
Removing the flatten path on FP8 routes to `trtllm-gen`, which errors
`TllmGenFmhaRunner: Unsupported architecture` — trtllm-gen FP8 MLA is not
compiled for sm120 in this FlashInfer build. Dead end without upstream
kernel work. TRITON_MLA + FULL CG (this fix) is strictly better: same FLAT
interarrival, correct acceptance, no upstream dependency.

### sglang sources

Not needed in the end — the FlashMLA pattern (shared `cg_buf_*` on the
metadata builder) was the right template and it's already in tree at
`vllm/v1/attention/backends/mla/flashmla.py`.

---

## Files modified

- `vllm/v1/attention/backends/mla/triton_mla.py` — all changes
  (see `patches/triton_mla_full_cg_WIN.patch`, patched file saved as
  `patches/triton_mla_final.py`).

No other vLLM files need changes.

## Benchmark scripts

- `bench/bench_triton_mla.py` — kernel-level microbench (decode fwd)
- `bench/e2e_bench.py` — streaming tok/s + interarrival probe

## Docker images

| tag | status | description |
|-----|--------|-------------|
| `voipmonitor/vllm:cu130-mtp-baseline-20260423` | ✅ kept | pre-changes snapshot |
| `voipmonitor/vllm:cu130-mtp-fix-v1-20260423` | ❌ superseded | v1 sync-fix (no throughput win; 2k regression) |
| `voipmonitor/vllm:cu130-mtp-fix-v2-20260423` | ❌ superseded | defensive v1 minus the regressing bit |
| `voipmonitor/vllm:cu130-mtp-cg-fix-20260423` | ✅ **USE THIS** | FULL CG for TRITON_MLA, 4.3× at 30k |
