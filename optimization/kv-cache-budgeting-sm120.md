# KV Cache Budgeting On SM120

Weights and the key/value (KV) cache come out of the same 94.97 GiB per card,
so on a checkpoint that nearly fills the GPUs the usable context window is set
by weight precision and by two flags that are easy to get backwards. This page
records measurements from one model served four ways on 8x RTX PRO 6000
Blackwell (SM120, PCIe, no NVLink), and the ordering rule behind them.

The short version:

- Requantising FP8 to NVFP4 freed 35.7 GiB per GPU and took the servable
  context from 49,152 to 589,824 tokens on the same checkpoint.
- `flashinfer_autotune` allocates **after** the KV cache and is **not** in the
  profiling budget, so vLLM's own "fully utilize gpu memory" suggestion
  overshoots.
- That allocation **scales with `--max-num-batched-tokens`**. Raising chunk
  size therefore requires *lowering* `--gpu-memory-utilization`, which is the
  opposite of the intuition.

## Table of Contents

- [Why Precision Sets The Context Window](#why-precision-sets-the-context-window)
- [The Allocation Ordering Trap](#the-allocation-ordering-trap)
- [Autotune Scales With Chunk Size](#autotune-scales-with-chunk-size)
- [Measured Configurations](#measured-configurations)
- [Tuning Procedure](#tuning-procedure)
- [Tensor Parallelism Does Not Buy Context Under MLA](#tensor-parallelism-does-not-buy-context-under-mla)
- [Reading Parameter Counts From Packed NVFP4 Repositories](#reading-parameter-counts-from-packed-nvfp4-repositories)

## Why Precision Sets The Context Window

`GLM-5.3` (architecture `glm_moe_dsa`, 78 layers, Multi-head Latent Attention
(MLA) with `kv_lora_rank` 512 and Dynamic Sparse Attention (DSA) at
`index_topk` 2048) declares `max_position_embeddings` of 1,048,576. What is
actually servable depends entirely on what the weights leave behind:

| Precision | Weights per GPU | KV available | Pool | Servable context |
|---|---|---|---|---|
| FP8 | 90.2 GiB | 2.5 GiB | 49,728 | 49,152 |
| NVFP4 | 54.52 GiB | 30.23 GiB | 604,672 | 589,824 |

Same checkpoint, same 8 GPUs, a 12x difference in context. The FP8 build is not
a short-context model; it is a normal-context model with no room left to cache.
This architecture measures at roughly 54 KB per token of KV, so every GiB
recovered from weights is worth about 19,000 tokens of window.

If a model's context looks disappointing, check the weight footprint before
concluding the checkpoint is limited.

## The Allocation Ordering Trap

During startup vLLM profiles the model, sizes the KV cache to fill the
`--gpu-memory-utilization` budget, allocates it, and *then* runs kernel warmup.
`flashinfer_autotune` performs a dummy forward pass at that last step, and its
allocation was never part of the profiling budget. It is spent from whatever
utilization left unreserved.

This makes the suggestion vLLM prints during profiling misleading:

```
Replace gpu_memory_utilization config with `--kv-cache-memory=3340740608`
(3.11 GiB) to fit into requested memory, or `--kv-cache-memory=34647019008`
(32.27 GiB) to fully utilize gpu memory.
```

Following the "fully utilize" figure produces a clean KV sizing and then a
failure in warmup:

```
compile_or_warm_up_model -> kernel_warmup -> flashinfer_autotune -> _dummy_run
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
GPU 0 has a total capacity of 94.97 GiB of which 441.44 MiB is free.
```

Leave real slack instead of taking that number literally.

## Autotune Scales With Chunk Size

The size of the autotune allocation is not a constant. It tracks
`--max-num-batched-tokens`:

| `--max-num-batched-tokens` | Autotune allocation | Peak activation |
|---|---|---|
| 1024 | ~512 MiB | 0.54 GiB |
| 8192 | 1.54 GiB | 2.53 GiB |

Both numbers matter and they stack. Peak activation is inside the profiling
budget and shrinks the KV pool directly; the autotune allocation is outside it
and must fit in the unreserved remainder.

The practical consequence is counterintuitive: **raising
`--max-num-batched-tokens` requires lowering `--gpu-memory-utilization`.** At
8192 chunks, utilization 0.98 leaves 1.37 GiB free against a 1.54 GiB request
and dies in warmup; 0.97 leaves roughly 2.8 GiB and starts.

## Measured Configurations

Same checkpoint and hardware throughout; only the two flags move.

| Chunks | Util | Peak activation | KV per GPU | Pool | `--max-model-len` | Result |
|---|---|---|---|---|---|---|
| 1024 | 0.95 | 0.54 GiB | 30.21 GiB | 607,872 | 589,824 | starts |
| 8192 | 0.95 | 2.53 GiB | 28.48 GiB | 569,984 | 557,056 | starts, 5.6% less context |
| 8192 | 0.98 | 2.53 GiB | 31.48 GiB | 620,863 | 618,496 | **OOM in warmup** |
| 8192 | 0.97 | 2.53 GiB | 30.23 GiB | 604,672 | 589,824 | starts |

The last row is the useful one: utilization 0.97 buys back exactly what the
larger chunk size costs, landing on the same context as the 1024 configuration.

Measured prefill improvement from the 8x chunk raise was **+11%** (1,988 to
2,210 tokens/s on an identical ~550,000 token prompt), which is modest. On this
SM120 sparse-MLA path prefill does not appear to be limited by kernel launch
overhead, so there is little reason to push chunk size further. It was worth
taking only because at utilization 0.97 it costs no context.

Throughput figures taken while `--enable-prefix-caching` is on are not a clean
benchmark: vLLM counts cache hits toward `Avg prompt throughput`, and runs
sharing a prefix reported 46,000-57,000 tokens/s against a measured 2,210. Check
the reported `Prefix cache hit rate` before quoting any prefill number.

## Tuning Procedure

Measure rather than estimate. An estimate from 54 KB/token was 7% high, which
is enough to prevent startup.

1. Boot with a deliberately conservative `--max-model-len`.
2. Read the engine's own line:
   `GPU KV cache size: N tokens, Maximum concurrency for M tokens per request`.
3. Raise `--max-model-len` toward `N`, aligned down to a multiple of 4096, with
   a few percent of margin. The pool varies slightly between boots on identical
   configuration, so a value sitting exactly on the ceiling can start once and
   fail later.
4. Never set `--max-model-len` above the pool. vLLM refuses with a `ValueError`
   naming the real ceiling, roughly 90 seconds into profiling and before the
   weight load completes. This is a fast, informative failure and is the
   cheapest way to find the true limit.

## Tensor Parallelism Does Not Buy Context Under MLA

Under MLA the KV cache is **replicated per tensor-parallel rank**, not sharded.
Raising `--tensor-parallel-size` provides room for *weights* and leaves the
per-GPU cache budget unchanged, so it does not extend the context window.

Decode Context Parallelism (DCP) would shard it, but the SM120 sparse MLA
implementation derives from `MLAAttentionImpl` rather than `SparseMLACommonImpl`
and never defines `dcp_world_size`, so that path is unavailable on this backend.

The corollary is that the only lever on context for a weight-heavy checkpoint is
the weight footprint itself, which is what makes the NVFP4 requantisation above
worth the effort.

## Reading Parameter Counts From Packed NVFP4 Repositories

The Hugging Face API reports *packed* element counts for U8-packed NVFP4
repositories, so such a repository shows roughly half its true parameter count
and derives to around 9 bits per weight. A genuinely 4-bit checkpoint can
therefore look heavier than a 4.57 bits-per-weight baseline when it is lighter.

Compare the total safetensors byte size rather than any bits-per-weight figure
derived from the reported parameter count.
