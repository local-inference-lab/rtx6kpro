# Inference Throughput: Qwen3.5-397B-A17B Quantizations

Decode throughput benchmark comparing quantized Qwen3.5-397B-A17B checkpoints on 4x RTX PRO 6000 Blackwell (TP4), with and without MTP speculative decoding.

## Test Environment

| Parameter | Value |
|-----------|-------|
| **GPUs** | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB each) |
| **Engine** | vLLM 0.17.0rc1 (TP4) |
| **Container** | llm-pytorch-blackwell:nightly-cuda132 |
| **Benchmark tool** | llm_decode_bench.py |
| **Date** | 2026-03-14 |
| **Max tokens** | 512 per request |
| **Concurrency** | 1, 2, 4, 8, 16, 32, 64, 128 |
| **Context lengths** | 0, 16k, 32k, 64k, 128k |

### vLLM server config (common)

```
--tensor-parallel-size 4
--gpu-memory-utilization 0.9
--max-num-batched-tokens 8192
--max-num-seqs 128
--enable-prefix-caching
--enable-chunked-prefill
```

MTP: `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`

---

## AWQ MTP Long-Context Degradation

**AWQ + MTP is catastrophically slow at long context.** This is the single most important finding and fundamentally changes the recommendation for agentic/long-context workloads.

### Per-request decode speed at C=1 (tok/s) — the agentic use case

This table shows what a single user experiences. No queuing effects, pure compute performance.

| Model | MTP | ctx=0 | 16k | 32k | 64k | 128k | Degradation |
|-------|-----|-------|-----|-----|-----|------|-------------|
| **AWQ** | **ON** | **147** | 110 | 88 | 61 | **40** | **-73%** |
| AWQ | OFF | 104 | 104 | 103 | 100 | 96 | -8% |
| lukealonso NVFP4 | ON | 127 | 127 | 127 | 128 | 122 | -4% |
| lukealonso NVFP4 | OFF | 81 | 80 | 80 | 78 | 77 | -5% |
| nvidia NVFP4 | ON | 121 | 122 | 120 | 125 | 123 | -2% (stable) |
| nvidia NVFP4 | OFF | 79 | 78 | 78 | 76 | 75 | -5% |

**The crossover point:**
- **ctx=0:** AWQ MTP is 16% faster than NVFP4 MTP (147 vs 127)
- **ctx=16k:** NVFP4 MTP is already 15% faster (127 vs 110)
- **ctx=32k:** AWQ MTP (88) drops below AWQ *without* MTP (103) — **MTP becomes actively harmful**
- **ctx=64k:** NVFP4 MTP is 2.1x faster than AWQ MTP (128 vs 61)
- **ctx=128k:** NVFP4 MTP is 3.1x faster than AWQ MTP (122 vs 40)

### Root cause

The root cause is not yet fully understood. All three models share identical `vocab_size=248320` and architecture (`Qwen3_5MoeForConditionalGeneration` VLM). The degradation is specific to AWQ INT4 quantization interacting with vLLM's MTP speculative decoding at long contexts. Likely factors:
- AWQ INT4 GEMM kernels (group_size=128, per-channel scaling metadata) may have higher memory overhead than NVFP4's hardware-native E2M1 format when combined with MTP's additional KV cache for speculative tokens
- vLLM's MTP implementation may allocate KV cache differently for AWQ vs NVFP4 quantization backends
- AWQ's speculative token acceptance rate may degrade at long context, wasting KV cache budget on rejected tokens

At C=128 ctx=128k, queue utilization reaches ~81%, confirming requests are being serialized rather than batched.

### Agentic / Low-Concurrency Recommendation

For agentic workloads (tool-calling agents, RAG with long documents, multi-turn conversations with growing context):

- **Use NVFP4 (lukealonso) + MTP.** Delivers 122 tok/s at 128k context — stable across all lengths.
- **Do NOT use AWQ + MTP** if context may exceed 16k. At 32k+, MTP actively hurts AWQ performance.
- If AWQ is required for quality reasons, **disable MTP** for long-context requests — AWQ without MTP (96 tok/s at 128k) is 2.4x faster than AWQ with MTP (40 tok/s).

---

## Summary

### Decode throughput at context=0 (tok/s)

| Model | MTP | C=1 | C=8 | C=16 | C=32 | C=64 | C=128 |
|-------|-----|-----|-----|------|------|------|-------|
| **AWQ** | **ON** | **147** | **767** | **1163** | **1679** | **2622** | **3519** |
| lukealonso NVFP4 | ON | 127 | 615 | 934 | 1441 | 2283 | 3220 |
| nvidia NVFP4 | ON | 121 | 577 | 918 | 1418 | 2252 | 3232 |
| AWQ | OFF | 104 | 509 | 843 | 1272 | 1909 | 2796 |
| lukealonso NVFP4 | OFF | 81 | 414 | 668 | 987 | 1590 | 2291 |
| nvidia NVFP4 | OFF | 79 | 406 | 652 | 987 | 1590 | 2294 |

### Decode throughput at context=64k (tok/s)

| Model | MTP | C=1 | C=8 | C=16 | C=32 | C=64 | C=128 |
|-------|-----|-----|-----|------|------|------|-------|
| lukealonso NVFP4 | ON | 128 | 525 | 904 | 1295 | **1905** | **2183** |
| nvidia NVFP4 | ON | 125 | 581 | 877 | 1271 | **1912** | **2159** |
| **AWQ** | **ON** | **61** | **389** | **680** | **1074** | 1747 | 2303 |
| AWQ | OFF | 100 | 477 | 748 | 1080 | 1464 | 1909 |
| lukealonso NVFP4 | OFF | 78 | 398 | 636 | 922 | 1338 | 1907 |
| nvidia NVFP4 | OFF | 76 | 390 | 621 | 891 | 1339 | 1783 |

### MTP speedup

| Model | C=1 | C=8 | C=32 | C=64 | C=128 |
|-------|-----|-----|------|------|-------|
| AWQ | +41% | +51% | +32% | +37% | +26% |
| lukealonso NVFP4 | +57% | +49% | +46% | +44% | +41% |
| nvidia NVFP4 | +53% | +42% | +44% | +42% | +41% |

### Key findings

1. **AWQ + MTP degrades catastrophically at long context:** per-request decode drops from 147 tok/s (ctx=0) to 40 tok/s (ctx=128k) — a 73% degradation. At ctx=32k, MTP becomes slower than no-MTP for AWQ. At ctx=128k, NVFP4 MTP is 3.1x faster than AWQ MTP. See [AWQ MTP Long-Context Degradation](#awq-mtp-long-context-degradation) above.
2. **NVFP4 MTP is stable across all context lengths:** lukealonso delivers 122-128 tok/s at C=1 regardless of context, with only 4% degradation from ctx=0 to ctx=128k
3. **AWQ is fastest only at short context (ctx=0)**, outperforming NVFP4 by 9-16% with MTP ON
4. **AWQ collapses at 128k/C=128 MTP ON**: throughput drops to 646 tok/s (queue utilization ~81%) while NVFP4 holds 2157 tok/s
5. **Without MTP, all three models converge at 128k/C=128 to ~1527 tok/s** — the gap disappears entirely
6. **MTP gives 26-57% speedup at ctx=0** depending on model and concurrency — but only NVFP4 sustains this at long context
7. **lukealonso and nvidia NVFP4 have identical throughput** without MTP; lukealonso is ~5% faster with MTP at low concurrency
8. **AWQ MTP at C=128 ctx=0 peaks at 3519 tok/s** — highest measured throughput in short-context workloads

---

## Full Results: AWQ (QuantTrio/Qwen3.5-397B-A17B-AWQ)

### AWQ MTP ON — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 147 | 269 | 411 | 767 | 1163 | 1679 | 2622 | 3519 |
| 16k | 110 | 205 | 319 | 602 | 980 | 1435 | 2231 | 2135 |
| 32k | 88 | 163 | 269 | 507 | 851 | 1284 | 2044 | 2024 |
| 64k | 61 | 115 | 206 | 389 | 680 | 1074 | 1747 | 2303 |
| 128k | 40 | 76 | 136 | 264 | 477 | 812 | 1326 | 646 |

### AWQ MTP OFF — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 104 | 167 | 298 | 509 | 843 | 1272 | 1909 | 2796 |
| 16k | 104 | 167 | 294 | 501 | 812 | 1209 | 1781 | 2542 |
| 32k | 103 | 165 | 290 | 493 | 796 | 1146 | 1654 | 2290 |
| 64k | 100 | 159 | 282 | 477 | 748 | 1080 | 1464 | 1909 |
| 128k | 96 | 151 | 267 | 445 | 684 | 923 | 1209 | 1526 |

---

## Full Results: lukealonso/Qwen3.5-397B-A17B-NVFP4

### lukealonso MTP ON — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 127 | 219 | 350 | 615 | 934 | 1441 | 2283 | 3220 |
| 16k | 127 | 220 | 354 | 610 | 881 | 1439 | 2206 | 2902 |
| 32k | 127 | 214 | 345 | 601 | 933 | 1377 | 2075 | 2574 |
| 64k | 128 | 215 | 344 | 525 | 904 | 1295 | 1905 | 2183 |
| 128k | 122 | 214 | 344 | 570 | 838 | 1185 | 1633 | 2157 |

### lukealonso MTP OFF — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 81 | 141 | 251 | 414 | 668 | 987 | 1590 | 2291 |
| 16k | 80 | 139 | 250 | 406 | 653 | 986 | 1465 | 2164 |
| 32k | 80 | 137 | 247 | 406 | 652 | 954 | 1463 | 2037 |
| 64k | 78 | 135 | 243 | 398 | 636 | 922 | 1338 | 1907 |
| 128k | 77 | 133 | 239 | 389 | 604 | 859 | 1209 | 1527 |

---

## Full Results: nvidia/Qwen3.5-397B-A17B-NVFP4

### nvidia MTP ON — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 121 | 212 | 340 | 577 | 918 | 1418 | 2252 | 3232 |
| 16k | 122 | 207 | 340 | 598 | 922 | 1390 | 2167 | 2624 |
| 32k | 120 | 206 | 340 | 589 | 909 | 1340 | 2065 | 2502 |
| 64k | 125 | 209 | 341 | 581 | 877 | 1271 | 1912 | 2159 |
| 128k | 123 | 203 | 334 | 554 | 806 | 1164 | 1620 | 2138 |

### nvidia MTP OFF — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 79 | 135 | 243 | 406 | 652 | 987 | 1590 | 2294 |
| 16k | 78 | 133 | 239 | 398 | 652 | 955 | 1464 | 2164 |
| 32k | 78 | 131 | 239 | 398 | 637 | 954 | 1462 | 2036 |
| 64k | 76 | 131 | 235 | 390 | 621 | 891 | 1339 | 1783 |
| 128k | 75 | 127 | 231 | 382 | 589 | 858 | 1209 | 1527 |

---

## Analysis

### AWQ MTP long-context collapse

The most striking result is AWQ MTP's per-request performance degradation across context lengths. At C=1, where there is no queuing — only pure decode compute — AWQ MTP drops from 147 tok/s at ctx=0 to just 40 tok/s at ctx=128k, a 73% decline. In contrast, NVFP4 MTP maintains 122-128 tok/s across all context lengths (only 4% decline).

The crossover where MTP *hurts* AWQ is between ctx=16k and ctx=32k:
- At ctx=32k: AWQ MTP (88 tok/s) is slower than AWQ no-MTP (103 tok/s)
- This means MTP speculative decoding is generating tokens that get rejected, consuming KV cache budget without delivering throughput gains

At high concurrency (C=128) with ctx=128k, this compounds into total collapse: 646 tok/s vs NVFP4's 2157 tok/s. Queue utilization hits 81%, meaning requests are being serialized rather than batched.

**Note on root cause:** All three models share identical `vocab_size=248320` and the same VLM architecture (`Qwen3_5MoeForConditionalGeneration`). The degradation is specific to AWQ INT4 quantization interacting with vLLM's MTP speculative decoding. Likely factors include AWQ's per-channel scaling metadata overhead (group_size=128), different KV cache allocation paths in vLLM for AWQ vs NVFP4, and potentially lower speculative token acceptance rates for AWQ at long context.

### AWQ vs NVFP4 throughput

The winner depends on context length and whether MTP is enabled:

**Short context (ctx=0), MTP ON:** AWQ is fastest, outperforming NVFP4 by 9-16% across all concurrency levels. AWQ peaks at 3519 tok/s vs NVFP4's 3220-3232 tok/s at C=128.

**Long context (64k+), MTP ON:** NVFP4 dominates. At ctx=64k/C=1, NVFP4 is 2.1x faster (128 vs 61 tok/s). At ctx=128k/C=1, it's 3.1x faster (122 vs 40 tok/s). Even at high concurrency, NVFP4 holds steady while AWQ collapses.

**Without MTP:** AWQ is moderately faster at short context (3-33%), but all three models converge tightly at 128k/C=128 to approximately 1527 tok/s. The context-length penalty is steepest for AWQ with MTP.

AWQ's advantage at short context despite NVFP4 having dedicated FP4 Tensor Cores is explained by:
1. AWQ uses mature INT4 GEMM kernels with better scheduling
2. NVFP4's E2M1 format (8 unique values) vs AWQ's 16 levels means slightly different effective quantization density per bit
3. AWQ's per-channel scaling and salient weight protection reduce quantization error without runtime overhead

### MTP impact on throughput

MTP provides substantial speedup at short context across all models:
- **NVFP4 models:** consistent 40-57% speedup at low concurrency, 41% at C=128 — **sustained at all context lengths**
- **AWQ:** 41-51% at low concurrency at ctx=0, but **MTP becomes harmful at ctx=32k+** (AWQ MTP is slower than AWQ no-MTP)

MTP is more effective for NVFP4 because the base decode speed is slower, giving more room for speculative acceleration. Critically, NVFP4 sustains the MTP benefit at all context lengths, while for AWQ the benefit reverses.

### Context length impact

Throughput degrades with longer contexts due to KV cache memory pressure:
- **Without MTP:** All models degrade gracefully. At C=128, ctx=128k vs ctx=0 reduces throughput by ~31-40%. All three models land at ~1527 tok/s at 128k/C=128.
- **With MTP, NVFP4:** Moderate degradation at high context. 128k/C=128 yields ~2138-2157 tok/s, still well above the no-MTP baseline.
- **With MTP, AWQ:** At C=1 — 73% degradation from ctx=0 to ctx=128k. At C=128 — complete collapse at 128k (646 tok/s, below no-MTP baseline of 1526 tok/s).

The practical recommendation: use AWQ for short-context batch workloads (ctx < 16k); prefer NVFP4 for long-context or mixed-context deployments. If AWQ is needed for quality, disable MTP when context exceeds 16k.

### NVFP4: lukealonso vs nvidia

Without MTP, the two NVFP4 checkpoints have **identical throughput** (within measurement noise). With MTP, lukealonso is ~5% faster at C=1 (127 vs 121 tok/s) but converges at high concurrency. The difference likely comes from minor weight distribution differences affecting MTP acceptance rate.

---

## Legacy: SGLang Results (2026-03-11)

Previous measurements on SGLang 0.5.9 (TP4) with MTP ON only. These used a different benchmark method (Prometheus `sglang:gen_throughput` metric) and are not directly comparable to the vLLM results above.

```
SGLang MTP ON — Aggregate decode throughput (tok/s), context=0
=========================================================================

Model                                 C=1    C=8    C=16    C=32    C=64
------------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ      152    665     976    1516    1662
lukealonso/Qwen3.5-397B-A17B-NVFP4   132    581     852    1191    1202
```

Note: SGLang numbers are lower at high concurrency because `--max-running-requests 64` was used vs 128 for vLLM. The relative ranking (AWQ > NVFP4) is consistent across both engines.
