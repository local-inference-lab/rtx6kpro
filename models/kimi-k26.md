# Kimi K2.6 on RTX PRO 6000 Blackwell

## Overview

This page tracks the current public community recipe for the Kimi MLA serving stack on RTX PRO 6000 Blackwell systems.

The reproducible public demo path currently uses:
- target model: `moonshotai/Kimi-K2.5`
- speculative draft: `lightseekorg/kimi-k2.5-eagle3-mla`
- serving engine: `vLLM`
- attention backend: `TRITON_MLA`
- KV cache: `fp8`
- decode context parallelism: `DCP=4` or `DCP=8`

That setup is used here because it exercises the same MLA serving path that matters for Kimi K2.x, is public, and is significantly more representative for Kimi than the non-MLA Llama-style EAGLE draft.

## Community Image

Docker image:

```bash
docker pull voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422
```

What is inside:
- vLLM `0.19.2rc1.dev48+g47fcb8ca6.d20260420`
- patched NCCL `2.29.7`
- Kimi MLA runtime path: `TRITON_MLA + fp8 KV`
- DCP=4 XML-scoped workaround included in the image
- community-tested `Kimi-K2.5 + eagle3-mla` launch path

## Why This Uses `eagle3-mla` Instead of the Llama Draft

The MLA draft is the right public demo for Kimi because:
- it is MLA-aware, so it uses the same `TRITON_MLA + fp8 KV + DCP` stack as the target model
- it is a better proxy for real Kimi serving than the non-MLA Llama-style draft
- in practical Kimi testing it behaved better than the Llama draft for this serving path
- it keeps the benchmark focused on the Kimi MLA runtime instead of mixing in a different draft backend/runtime path

## Recommended Launch Commands

### Fastest known: DCP=4

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422 \
  bash -lc '
VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000 \
VLLM_ENABLE_PCIE_ALLREDUCE=1 \
NCCL_P2P_LEVEL=SYS \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_LOG_STATS_INTERVAL=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
VLLM_MARLIN_INPUT_DTYPE=fp8 \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.5 \
  --served-model-name Kimi-K2.5 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 5000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 128 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --decode-context-parallel-size 4 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --speculative-config '\''{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"probabilistic"}'\''
'
```

### Reference alternative: DCP=8

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422 \
  bash -lc '
VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000 \
VLLM_ENABLE_PCIE_ALLREDUCE=1 \
NCCL_P2P_LEVEL=SYS \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_LOG_STATS_INTERVAL=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
VLLM_MARLIN_INPUT_DTYPE=fp8 \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.5 \
  --served-model-name Kimi-K2.5 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 5000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 128 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --decode-context-parallel-size 8 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --speculative-config '\''{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"probabilistic"}'\''
'
```

## DCP=4 vs DCP=8

From `python3 /mnt/llm_decode_bench.py --port 5000` on the same serving stack:

| Scenario | DCP=4 | DCP=8 |
|---|---:|---:|
| Decode, ctx=0, concurrency=1 | 85.5 tok/s | 78.6 tok/s |
| Decode, ctx=16k, concurrency=1 | 52.7 tok/s | 47.7 tok/s |
| Decode, ctx=32k, concurrency=128 | 889.8 tok/s | 763.5 tok/s |
| Decode, ctx=64k, concurrency=64 | 508.4 tok/s | 445.2 tok/s |

At the moment `DCP=4` is the recommended community setting and `DCP=8` is the reference comparison point.

## NCCL XML Note

`NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml` is still recommended for the fastest known configuration.

Patched NCCL without XML was functional, but prefill was still slower than the XML-based path, especially at low concurrency. The XML path remains the best known public recipe for now.

## Minimal Discord Summary

> Community image: `voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422`
>
> The public demo path uses `moonshotai/Kimi-K2.5` + `lightseekorg/kimi-k2.5-eagle3-mla` because that is the most representative MLA setup for Kimi on vLLM: same `TRITON_MLA + fp8 KV + DCP` serving path, and better behavior than the non-MLA Llama draft.
>
> Current recommendation: `DCP=4` with `NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml`. `DCP=8` remains the reference alternative.
