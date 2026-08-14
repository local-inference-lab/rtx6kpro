# Daily Summaries

Automated daily summaries of the [RTX6kPRO Discord](https://discord.gg/X54jjmcxWJ) community activity.
Each highlight links directly to the relevant Discord message.

*Auto-generated daily at 08:07 UTC. Source: Discord bot → Claude → Discord + wiki.*


## March 2026

| Date | Highlights |
|------|------------|
| [2026-08-13](2026-08/2026-08-13.md) | - DeepSeek-V4-Pro-0813 (GA) launched; mixed reviews — strong benchmarks but users found it slower/ |
| [2026-08-12](2026-08/2026-08-12.md) | - RTX PRO 6000 prices spiking hard — Nvidia's own store jumped to $16K, Microcenter went $11.8K→ |
| [2026-08-11](2026-08/2026-08-11.md) | B12X kernels heading upstream: Luke opened an official vLLM PR for native B12X support on SM120 —  |
| [2026-08-10](2026-08/2026-08-10.md) | B12X owner-reduce patch for GLM-5.2 TP8: John's 3-level PCIe topology-aware all-reduce cut cross-swi |
| [2026-08-09](2026-08/2026-08-09.md) | - New PCIe custom communication (ported from GLM tp8 work) pushes GLM-5.2 nvfp4 to 205 tok/s sustain |
| [2026-08-08](2026-08/2026-08-08.md) | DS4-Flash rapid-fire releases (r29→r31): fixed compressed-MLA verifier corruption above C24, nativ |
| [2026-08-07](2026-08/2026-08-07.md) | ormandj's SGLang DS4-Flash build fixes looping/garbage-output bug that plagued DSpark; GSM8K bench n |
| [2026-08-06](2026-08/2026-08-06.md) | - ormandj found & patched a major SGLang bug causing SWA/KV-cache retract loops on long generations  |
| [2026-08-05](2026-08/2026-08-05.md) | DS4-Flash reasoning bug fixed: HIGH was silently rendering as MAX; r27 fix confirmed — LOW or MAX  |
| [2026-08-04](2026-08/2026-08-04.md) | Critical DS4-Flash encoder bugs found: reasoning_effort MAX was mislabeled as HIGH (HIGH was a no-op |
| [2026-08-03](2026-08/2026-08-03.md) | DS4-Flash-0731 "Gilded Gnosis" r24 published — fixes the long-standing "Workspace is locked" b12x  |
| [2026-08-02](2026-08/2026-08-02.md) | DeepSeek-V4-Flash-0731 (GA) ships with a serious quality regression: multiple users (ormandj, RelayM |
| [2026-08-01](2026-08/2026-08-01.md) | DeepSeek-V4-Flash-0731 officially released (successor to preview), same DSpark speculative-decode ar |
| [2026-07-31](2026-07/2026-07-31.md) | DeepSeek V4 Flash-0731 dropped (API-only, weights pending) with post-trained-only update — claims  |
| [2026-07-30](2026-07/2026-07-30.md) | MMLU-Pro accuracy showdown: NF3/EXL3 quants match or beat NVFP4 in several categories (business, bio |
| [2026-07-29](2026-07/2026-07-29.md) | - GLM-5.2 "Gilded Gnosis" v20 saw rapid-fire docker releases (r5→r9) adding forked LMCache 0.5.2 w |
| [2026-07-28](2026-07/2026-07-28.md) | Kimi-K3 released (753B total / 104B active params, 1.56TB weights, MXFP4-trained) — massive commun |
| [2026-07-27](2026-07/2026-07-27.md) | Kimi-K3 released on HuggingFace; community immediately starts MoE kernel/quant work [(jump)](https:/ |
| [2026-07-26](2026-07/2026-07-26.md) | - GLM-5.2 v20 image ships with DCP prefill collapse fixed — auto-policy composes query-split/CKV-g |
| [2026-07-25](2026-07/2026-07-25.md) | DCP4 collapse root-caused & fixed: koush spotted GLM-5.2 vLLM fork was gathering topk scores/idx it  |
| [2026-07-24](2026-07/2026-07-24.md) | - New GLM-5.2 vLLM image fixes DCP2/4/8 crashes (illegal memory access, tensor layout stride bug); u |
| [2026-07-23](2026-07/2026-07-23.md) | - GLM-5.2 v20 Docker image ships, fixing the cuBLAS/Xid DCP read-ahead crash and adding an auto-enab |
| [2026-07-22](2026-07/2026-07-22.md) | Poolside Laguna S 2.1 (118B MoE, 8B active, 1M ctx, single-GPU fit) launched but failed community te |
| [2026-07-21](2026-07/2026-07-21.md) | - GLM-5.2 NVFP4 KV-cache bug found (outer scale unused) — fixed, KLD improved, merged as vLLM PR # |
| [2026-07-20](2026-07/2026-07-20.md) | - b12x fused-indexer IMA crash (long-ctx + spec-decode) root-caused, fix merged [(jump)](https://dis |
| [2026-07-19](2026-07/2026-07-19.md) | - New vLLM image v18→v19 (gilded-gnosis) adds CKV gather + fp8 KV-rope, pushing GLM-5.2 hybrid col |
| [2026-07-18](2026-07/2026-07-18.md) | - koush's DCP CKV-gather patch (a general fix for any DSA-based model) delivers massive prefill gain |
| [2026-07-17](2026-07/2026-07-17.md) | Kimi K3 (Moonshot) unveiled: 2.8T params, 896 experts (16 active), native MXFP4/MXFP8 training, Atte |
| [2026-07-16](2026-07/2026-07-16.md) | Bug: DSpark v10 (lucifer-cutlass) shows a big regression vs v8 — JSON schema output error rate spi |
| [2026-07-15](2026-07/2026-07-15.md) | - vLLM KLD benchmarks compromised: PR #26682's fused/combo kernels cause run-to-run rounding varianc |
| [2026-07-14](2026-07/2026-07-14.md) | GLM-5.2 vLLM v1.3 released (davidsyoung/vllm-glm52): guarded B12X A2A decode +9-11%, hitting 83 tok/ |
| [2026-07-13](2026-07/2026-07-13.md) | ds4dspark v10 released — fixes draft/acceptance randomness independence, prefix-cache token exclus |
| [2026-07-12](2026-07/2026-07-12.md) | DS4-Flash "dspark" is 50%+ faster than MTP2 but unstable — random Chinese-language token spam unde |
| [2026-07-11](2026-07/2026-07-11.md) | - New GLM-5.2-MXFP8-NVFP4-NF3-Hybrid v3.6 checkpoint hits 91.92% GPQA Diamond (up from 88.38% baseli |
| [2026-07-10](2026-07/2026-07-10.md) | - GLM-5.2-MXFP8-NVFP4-NF3-Hybrid "healed" weights (KLD 0.15→0.12, GPQA +0.5) jumped aider-polyglot |
| [2026-07-09](2026-07/2026-07-09.md) | GLM-5.2 MXFP8-NVFP4-NF3-Hybrid quant hits GPQA-Diamond 88.38% and beats REAP-504B on aider polyglot  |
| [2026-07-08](2026-07/2026-07-08.md) | GLM-5.2 v14 image: new hybrid a2a backend fixes DCP decode regression — +10% prefill / +13% decode |
| [2026-07-07](2026-07/2026-07-07.md) | GLM-5.2 v14 docker image released with new B12X backend (beats "lucifer" cutlass), instanttensor loa |
| [2026-07-06](2026-07/2026-07-06.md) | - DeepSeek-V4-Flash-DSpark added FP4 indexer cache (7-token draft) — small KV gain (+18.6k tokens) |
| [2026-07-05](2026-07/2026-07-05.md) | - jetha built a native NVFP4 attention kernel for sm120/121, speeding up Gemma 4 decode; wheels avai |
| [2026-07-04](2026-07/2026-07-04.md) | DS4-Flash v9 released — new B12X A16/A8 modes beat "lucifer-cutlass" on prefill and decode; A8 fas |
| [2026-07-03](2026-07/2026-07-03.md) | ACS redirect on c-payne PCIe switches nearly doubled token-gen throughput — clearing `ReqRedir/Cmp |
| [2026-07-02](2026-07/2026-07-02.md) | DSpark v8 (lucifer+cutlass) hits >1M context on TP2: 500 tok/s aggregate gen, 9000 tok/s prefill, 20 |
| [2026-07-01](2026-07/2026-07-01.md) | DS4-Flash v8 Docker released with b12x all-reduce enabled; real-world prefill on TP=2 lucifer-cutlas |
| [2026-06-30](2026-06/2026-06-30.md) | vLLM v8 incoming with TP=4 fix for DS4F+DSpark — Festr fixed the `--tp4` crash and reported 374– |
| [2026-06-29](2026-06/2026-06-29.md) | DSpark v7 released with prefix cache support — Festr published the v7 Docker image with DeepSpark  |
| [2026-06-28](2026-06/2026-06-28.md) | DeepSeek releases DSpark / DeepSpec: New speculative decoding paper + code claiming **57–78% faste |
| [2026-06-27](2026-06/2026-06-27.md) | New Docker image released: `eldritch-final-v1e8d565-b12x284a2ea-cu132-20260626` with all regressions |
| [2026-06-26](2026-06/2026-06-26.md) | DS4-Flash v5 recipe is broken — regressions across all kernels, was accidentally published before  |
| [2026-06-25](2026-06/2026-06-25.md) | DS4F v5 accidentally pushed — has a known crash bug in the b12x path; lucifer path unaffected. Do  |
| [2026-06-24](2026-06/2026-06-24.md) | GLM-5.2 REAP 504B Aider Polyglot final score: 89.8% pass_rate_2 after rerunning context-exhausted ca |
| [2026-06-23](2026-06/2026-06-23.md) | New Docker v12 fixes A16=1 VRAM bug, freeing 5–8GB/GPU; TP4 REAP KV cache jumps 400k → 1M tokens |
| [2026-06-22](2026-06/2026-06-22.md) | GLM-5.2 v12 Docker released with DCP+MTP fixes, VLLM_DCP_SHARD_DRAFT=1 now default — peaks **160 t |
| [2026-06-21](2026-06/2026-06-21.md) | GLM-5.2 REAP 504B aider polyglot benchmark: `pass_rate_2: 78.7` at "High" reasoning effort (225 test |
| [2026-06-20](2026-06/2026-06-20.md) | GLM 5.2 DCP+MTP root cause fixed: MTP layers were replicated per DCP rank instead of sharded, breaki |
| [2026-06-19](2026-06/2026-06-19.md) | GLM-5.2 NVFP4 on TP8 benchmarks: prefill 2,684 tok/s @ 8k ctx, 1,824 tok/s @ 128k ctx (200W power-li |
| [2026-06-18](2026-06/2026-06-18.md) | GLM-5.2 NVFP4 quant by luke uploaded (`lukealonso/GLM-5.2-NVFP4`); Aider benchmark shows pass_rate_1 |
| [2026-06-17](2026-06/2026-06-17.md) | GLM-5.2 released by zai-org: new IndexShare architecture (mandatory in 5.2, was experimental in 5.1) |
| [2026-06-16](2026-06/2026-06-16.md) | b12x decode speed regression fixed — a Codex-introduced bug in `expected_m` kernel selection cause |
| [2026-06-15](2026-06/2026-06-15.md) | Fable 5 scores 99.1% on Aider (223/225 after retry, 86.7% first-pass); Kimi K2.7 scores 91.6% for co |
| [2026-06-14](2026-06/2026-06-14.md) | KV cache bug: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` must be set in vLLM or usable KV cache tops out |
| [2026-06-13](2026-06/2026-06-13.md) | MiniMax M3 dropped (428B params, 26B active, multimodal with vision/video, 1M context, BF16 weights) |
| [2026-06-12](2026-06/2026-06-12.md) | DFlash cracked in vLLM — Festr solved vLLM dflash with help from mythos; MiMo Pro 2.5 (1T model) n |
| [2026-06-11](2026-06/2026-06-11.md) | DS4F empty-response bug confirmed reproducible: Model occasionally skips closing `</think>` tag, ret |
| [2026-06-10](2026-06/2026-06-10.md) | CUDA 13.3 released — fixes critical WGMMA data race bug causing silent result corruption in B12X N |
| [2026-06-09](2026-06/2026-06-09.md) | New Lucifer Docker image `hg436/vllm-public:lucifer-9d9a0a0` released: vLLM main + FlashInfer #3395  |
| [2026-06-08](2026-06/2026-06-08.md) | a8 vs a16 accuracy test (900 runs): Jon ran Estonia, hotel-lights, and lavd benchmarks; a8 won or ti |
| [2026-06-07](2026-06/2026-06-07.md) | DSV4-Flash prefix cache bug confirmed: On TP=4, the cache was evicting early and only utilizing ~200 |
| [2026-06-06](2026-06/2026-06-06.md) | New Docker image `voipmonitor/vllm:abyssal-abjuration-611a842` released with rebased vllm + latest f |
| [2026-06-05](2026-06/2026-06-05.md) | New image: `lavd/vllm:b12x-nameless-ascent-6-4-13.2-2` — Luke's latest `nameless-ascent` branch, d |
| [2026-06-04](2026-06/2026-06-04.md) | New Docker image `voipmonitor/dsv4-flash:lucifer-mxfp4-cutlass-20260603` released — lucifer base w |
| [2026-06-03](2026-06/2026-06-03.md) | DS4F prefix cache fix merged to vllm main ([PR #44082](https://github.com/vllm-project/vllm/pull/440 |
| [2026-06-02](2026-06/2026-06-02.md) | luke's `apotheosis` vLLM branch hits 250.3 tok/s on DS4-Flash TP=2 c=1 with full AOT compilation, si |
| [2026-06-01](2026-06/2026-06-01.md) | MiniMax M3 launched — multimodal (text/image/video→text), 1M-token context, MiniMax Sparse Atten |
| [2026-05-31](2026-05/2026-05-31.md) | DS V4 Flash on tp=4 hits ~200 t/s decode, 94.6% prefix cache hit rate, ~3M KV cache with `cstechdev/ |
| [2026-05-30](2026-05/2026-05-30.md) | DSv4 Flash hitting 245-250 tok/s on TP4 SM120 with luke's unholy-fusion branch + b12x MoE + MTP; luc |
| [2026-05-29](2026-05/2026-05-29.md) | DSV4-Flash "unholy-fusion" branch hits 245 t/s decode (MTP off ~130, MTP on approaching 200): Luke m |
| [2026-05-28](2026-05/2026-05-28.md) | DSV4-Flash SM120 breakthrough: `lucifer1004/dsv4-flash-sm120:latest` achieves full DeepGEMM prefill/ |
| [2026-05-27](2026-05/2026-05-27.md) | Luke fixed a long-context bug in both sglang and vllm; took the jasl branch, stripped its kernels, s |
| [2026-05-26](2026-05/2026-05-26.md) | DSv4-Flash docker `jasl-dsv4-5-23-13.2` (cu132): gen throughput ~35→55 tok/s; Lavd expects 70+ onc |
| [2026-05-25](2026-05/2026-05-25.md) | b12x DS V4 Flash sglang push: luke released new b12x build with MTP support; early results show **~4 |
| [2026-05-24](2026-05/2026-05-24.md) | DSV4 Flash: 82 tok/s C=1 MTP=OFF on sglang+b12x (TP=2), 79 tok/s at 128K context — C=2 near-linear |
| [2026-05-23](2026-05/2026-05-23.md) | DSv4 Flash: 1400 t/s on GB10 at 128k context using w4a8 cutlass; sglang+b12x shows zero decode degra |
| [2026-05-22](2026-05/2026-05-22.md) | RTX 6000 Pro price hike ~20% worldwide attributed to DDR7 memory costs; still selling [(jump)](https |
| [2026-05-21](2026-05/2026-05-21.md) | vLLM PR #40082 merged — FlashInfer b12x backends for SM120/SM121 (RTX Pro 6000 Blackwell / DGX Spa |
| [2026-05-20](2026-05/2026-05-20.md) | RTX Pro 6000 prices up 30-37% in South Korea and Europe vs. last week; members panic-buying at $9,99 |
| [2026-05-19](2026-05/2026-05-19.md) | DGX Spark DSv4: 700-800 prefill t/s at 128-200k ctx, 32 t/s decode C=1 MTP=2 [(jump)](https://discor |
| [2026-05-18](2026-05/2026-05-18.md) | DSv4-Flash MTP accuracy regression fixed in `lavd/vllm:jasl-dsv4-5-16-26`; prefix caching confirmed  |
| [2026-05-17](2026-05/2026-05-17.md) | DeepSeek V4 Flash on 2× RTX PRO 6000: pangfather shared full vLLM config (TP=2, 393K ctx, MTP=2, FP |
| [2026-05-16](2026-05/2026-05-16.md) | DSV4 Flash: 120 tok/s on TP=2; new `lavd/vllm:jasl-dsv4-5-15-26` image fixes cpasync import error fr |
| [2026-05-15](2026-05/2026-05-15.md) | Qwen 27B runaway thinking discovery: unbounded thinking on GPQA Diamond gives only 40.4% accuracy be |
| [2026-05-14](2026-05/2026-05-14.md) | infernix's GPT-5.5 autoresearch branch for DSv4 Flash on SM120 claims 4.8k prefill/s and 27/30 Eston |
| [2026-05-13](2026-05/2026-05-13.md) | DSV4 Flash MTP2: 35→120 tok/s after jasl vllm fixes; new image `lavd/vllm:jasl-dsv4-5-12-26` hits  |
| [2026-05-12](2026-05/2026-05-12.md) | repne/vllm:v3 released with tinygemm_bf16; Qwen3.6-27B FP8 TP=1 benchmarks: prefill 9,057 tok/s @8k  |
| [2026-05-11](2026-05/2026-05-11.md) | MiniMax M2.7 reaches 145 tok/s (TP=2) with latest b12x 0.13.5 + `B12X_MOE_FORCE_A16=1`, double the o |
| [2026-05-10](2026-05/2026-05-10.md) | DS4 Flash on 2x RTX Pro 6000 (FP8+FP4): 39 tok/s single, 79 tok/s dual. MTP + CUDA graph conflict � |
| [2026-05-09](2026-05/2026-05-09.md) | GLM-5.1 DCP=4 fixed: `voipmonitor/vllm:glm51-kimi-comm-20260508` eliminates garbage output at DCP=4; |
| [2026-05-08](2026-05/2026-05-08.md) | MiMo-V2.5 looping fix found: `do_sample=false` in generation_config.json disabled sampling params; f |
| [2026-05-07](2026-05/2026-05-07.md) | MiMo V2.5 thought loops confirmed as model-level bug: Luke ruled out quantization, kernels, and MTP  |
| [2026-05-06](2026-05/2026-05-06.md) | Gemma-4 MTP spec decode massive uplift: PR #41745 cherry-picked into vLLM nightly; single-stream 37  |
| [2026-05-05](2026-05/2026-05-05.md) | GLM-5.1 MTP corruption fixed in new Docker image (`voipmonitor/vllm:glm51-mtp-b12xsparse-ficutlass-t |
| [2026-05-04](2026-05/2026-05-04.md) | Luke cracked MiMo V2.5 MTP — root cause was TP tensors internally shuffled in a "dastardly rearran |
| [2026-05-03](2026-05/2026-05-03.md) | GLM-5.1 OOV loop bug: token IDs ≥154870 unmapped in tokenizer → infinite decode loops at temp=0; |
| [2026-05-02](2026-05/2026-05-02.md) | MiMo-V2.5 stream-race bug fixed: b12x/SGLang schedule/forward stream race on `req_to_token`; `jumper |
| [2026-05-01](2026-05/2026-05-01.md) | MiMo-V2.5 NVFP4 now runnable on TP=2 via luke's b12x 0.11.0 + sglang fork; ~80 tok/s single-batch, f |
| [2026-04-25](2026-04/2026-04-25.md) | DS V4/Flash blocked on SM120: DeepGEMM explicitly confirmed no SM120 support planned. Both DS V4 Pro |
| [2026-04-24](2026-04/2026-04-24.md) | DeepSeek V4 dropped (MIT): V4-Pro 1.6T params/49B active, Flash 284B. Flash fits 2× RTX 6000 Pro (~ |
| [2026-04-23](2026-04/2026-04-23.md) | Qwen 3.6 27B + FP8 released — 40–130 tok/s on single RTX 6k with MTP3, 70–90% acceptance rate; |
| [2026-04-22](2026-04/2026-04-22.md) | Kimi K2.6 hits 120–130 tok/s on 8×RTX 6000 Pro with DCP8 + Eagle3 speculative decoding + 3.6M fp8 |
| [2026-04-21](2026-04/2026-04-21.md) | Kimi K2.6 released (1.1T params, same arch as K2.5, vision included): SWE-Bench Pro 58.6%, HLE beati |
| [2026-04-20](2026-04/2026-04-20.md) | GLM-5.1 vLLM OOM bug fixed — new Docker `voipmonitor/vllm:glm51-tp8-nodcp-mtp3-tritondraft-b12x095 |
| [2026-04-19](2026-04/2026-04-19.md) | PCIe P2P allreduce config for ~10% throughput gain: Force-enable P2P via modprobe `options nvidia NV |
| [2026-04-18](2026-04/2026-04-18.md) | EXL3 benchmark breakthrough: mratsim's Qwen3.5-397B-A17B EXL3 quant achieves **1500 pp/s and 50+ tg/ |
| [2026-04-17](2026-04/2026-04-17.md) | Qwen3.6-35B-A3B released (MoE); community upset the poll-winner 27B wasn't chosen first — speculat |
| [2026-04-16](2026-04/2026-04-16.md) | b12x FP4 GEMM kernel merged into FlashInfer ([jump](https://discord.com/channels/1466898002793857221 |
| [2026-04-15](2026-04/2026-04-15.md) | Introspective Diffusion (I-DLM-8B) matches AR model quality, beats LLaDA-2.1-mini (16B) by +26 AIME- |
| [2026-04-14](2026-04/2026-04-14.md) | GLM 5.1 with native NSA/DSA attention now working on TP=8 with NVFP4 weights + FP8 KV cache via cust |
| [2026-04-13](2026-04/2026-04-13.md) | MiniMax M2.7 NVFP4 quant released by luke, then updated mid-day incorporating Jon's calibration data |
| [2026-04-12](2026-04/2026-04-12.md) | MiniMax M2.7 dropped (~1am) with same architecture as M2.5 — just a weight update. luke confirmed  |
| [2026-04-11](2026-04/2026-04-11.md) | vLLM MTP reaches sglang parity after a week of work by Festr; key insight was using sglang's eager-m |
| [2026-04-10](2026-04/2026-04-10.md) | GLM-5.1 NVFP4 quant live on 8x RTX Pro 6000: 131 tok/s with MTP, 95 tok/s via flashinfer cutlass. Bu |
| [2026-04-09](2026-04/2026-04-09.md) | GLM-5.1 NVFP4 upload by luke — 52 tok/s single request on 8x RTX 6000 PRO (NVFP4, b12x, sglang), t |
| [2026-04-08](2026-04/2026-04-08.md) | b12x hits 220 t/s single-user decode on 2x RTX PRO 6000; 198 t/s on Qwen3.5-122B vs 131 t/s baseline |
| [2026-04-07](2026-04/2026-04-07.md) | GLM-5 NVFP4 on 8×RTX PRO 6000: SGLang+MTP leads at 99 tok/s single-user (0 ctx), 249.8 tok/s at C=4 |
| [2026-04-06](2026-04/2026-04-06.md) | Qwen3.5 397B on 4×RTX PRO 6000: sglang record 108 tok/s (no MTP) / 180 tok/s (with MTP); vLLM now 9 |
| [2026-04-05](2026-04/2026-04-05.md) | vLLM 0.19 MoE TP regression identified and fixed — PR vllm-project/vllm#38990 awaiting merge; bran |
| [2026-04-04](2026-04/2026-04-04.md) | Qwen3.5-397B NVFP4 benchmarks on 4× RTX PRO 6000 (sglang b12x 0.7.2): MTP on = 180 tok/s single-use |
| [2026-04-03](2026-04/2026-04-03.md) | b12x 0.7.1 fixes OOM and >8 concurrency crashes on SM120; attention backend now launches for all bat |
| [2026-04-02](2026-04/2026-04-02.md) | b12x v0.7.0 released: fixes OOM on prefill, adds attention backend support (bs=1 only for now); new  |
| [2026-04-01](2026-04/2026-04-01.md) | PCIe oneshot AllReduce + fusion gives +7-8% decode throughput on SM120 across all tested models (Qwe |
| [2026-03-31](2026-03/2026-03-31.md) | b12x fused MoE kernel launched for SM120: 29–37% faster than cutlass at conc 1–8 on Qwen3.5-397B |
| [2026-03-30](2026-03/2026-03-30.md) | b12x benchmark (4x RTX 6000, TP4): 1.32x vs cutlass at conc=1-8; cutlass wins above conc=16; vLLM pe |
| [2026-03-29](2026-03/2026-03-29.md) | SGLang PR #21601: NVFP4 KV cache for Blackwell — ~2x memory vs FP8, no accuracy loss on GSM8K |
| [2026-03-28](2026-03/2026-03-28.md) | FlashInfer CuteDSL NVFP4 MoE backend landed (PR #2838); available in SGLang as `--moe-runner-backend |
| [2026-03-27](2026-03/2026-03-27.md) | GLM 5.1 announced by z.ai; no open weights yet; hosted API tested by Unoid (96 tool calls/2 min, "fe |
| [2026-03-26](2026-03/2026-03-26.md) | vLLM ModelRunner v2 still Top-K only logprobs — 1% accuracy miss; Phaelon's fork remains the only fu |
| [2026-03-25](2026-03/2026-03-25.md) | TurboQuant channel created; Google Research re-announced RaBitQ KV cache compression (ICLR 2026 post |
| [2026-03-24](2026-03/2026-03-24.md) | Festr publishes Dockerfiles to GitHub: `voipmonitor/blackwell-llm-docker` — community repo for SM120 |
| [2026-03-23](2026-03/2026-03-23.md) | GLM-5 hardware requirements: 6 cards + PP = no MTP, 35 tok/s; 8 cards + MTP = 140 tok/s; single node |
| [2026-03-22](2026-03/2026-03-22.md) | MiniMax M2.7 confirmed NOT releasing open weights; community disappointment; M2.5 remains the open o |
| [2026-03-21](2026-03/2026-03-21.md) | c-payne.com (Chris) joins: gen5 100-lane PCIe switches in stock; gen6 160-lane planned — Microchip c |
| [2026-03-20](2026-03/2026-03-20.md) | b12x SM120 MoE/FP4 kernels first Docker release: `voipmonitor/sglang:test-cu130`, 168 tok/s bench, 2 |
| [2026-03-19](2026-03/2026-03-19.md) | M2.7 weights still not out; M3 may go closed-source; GLM dropped AIR license, Qwen AI lead forced ou |
| [2026-03-18](2026-03/2026-03-18.md) | MiniMax M2.7 announced; weights expected Thu-Fri; worry about Chinese models closing source |
| [2026-03-17](2026-03/2026-03-17.md) | Flashinfer attention patch in Festr's docker (`voipmonitor/llm-pytorch-blackwell:nightly`) — sglang  |
| [2026-03-16](2026-03/2026-03-16.md) | `--attention-backend triton` causes 20 tok/s at 135k context on Qwen3.5; `flashinfer` fixes it but n |
| [2026-03-15](2026-03/2026-03-15.md) | brandonmusic's flashinfer K=64 patch confirmed by multiple independent testers to produce zero perfo |
| [2026-03-14](2026-03/2026-03-14.md) | Custom allreduce: 3.3x faster than NCCL at 256B–1KB (8 GPUs); Turin shows 3.8–5.8x at small sizes |
| [2026-03-13](2026-03/2026-03-13.md) | Festr's `llm-inference-bench`: single Python script for standardized LLM throughput testing |
| [2026-03-12](2026-03/2026-03-12.md) | KLD: AWQ lower divergence than NVFP4 for Qwen3.5; Unsloth dropped NVFP4 — poor quality and not faste |
| [2026-03-11](2026-03/2026-03-11.md) | vLLM: Qwen3.5 NVFP4 TP4+EP4+MTP3 — 150+ tok/s single, 1773 tok/s at 64 concurrent |
| [2026-03-10](2026-03/2026-03-10.md) | Kimi K2.5 EAGLE3 draft models on HF (`AQ-MedAI/Kimi-K25-eagle3`); vLLM PR #35966 adds support |
| [2026-03-09](2026-03/2026-03-09.md) | Bot posted comprehensive GLM-5 wiki pages: benchmark results, architecture overview, hardware requir |
| [2026-03-08](2026-03/2026-03-08.md) | Luke's custom PCIe allreduce beats NCCL ~2x at small batch; ~300 tok/s on Qwen3.5 NVFP4 8 GPUs |
| [2026-03-07](2026-03/2026-03-07.md) | PinchBench launched at pinchbench.com: agentic coding benchmark; GLM-5 and MiniMax M2.5 both under 4 |
| [2026-03-06](2026-03/2026-03-06.md) | GLM-5 crash: `flashinfer_cutlass` fp4-gemm triggers NaN; fix: `SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_E |
| [2026-03-05](2026-03/2026-03-05.md) | `--moe-runner-backend deep_gemm` boosts GLM-5 EAGLE by 20–30 tok/s; actually falls back to `cutlass` |
| [2026-03-04](2026-03/2026-03-04.md) | GLM-5 crash narrowed: `--fp4-gemm-backend flashinfer_cutlass` causes NaN/assertion failures; `flashi |
| [2026-03-03](2026-03/2026-03-03.md) | MTP changes model behavior: enabling MTP on Qwen3.5-NVFP4 activates different MoE experts; PR #35936 |
| [2026-03-02](2026-03/2026-03-02.md) | FlashInfer PRs #2460 and #2650: NVFP4 sm120 improvements; marginal gain over compiled vLLM (65 vs 67 |
| [2026-03-01](2026-03/2026-03-01.md) | Festr releases `claude-relay`: routes Claude Code to local models with cache sanitization |

## February 2026

| Date | Highlights |
|------|------------|
| [2026-08-13](2026-08/2026-08-13.md) | - DeepSeek-V4-Pro-0813 (GA) launched; mixed reviews — strong benchmarks but users found it slower/ |
| [2026-08-12](2026-08/2026-08-12.md) | - RTX PRO 6000 prices spiking hard — Nvidia's own store jumped to $16K, Microcenter went $11.8K→ |
| [2026-08-11](2026-08/2026-08-11.md) | B12X kernels heading upstream: Luke opened an official vLLM PR for native B12X support on SM120 —  |
| [2026-08-10](2026-08/2026-08-10.md) | B12X owner-reduce patch for GLM-5.2 TP8: John's 3-level PCIe topology-aware all-reduce cut cross-swi |
| [2026-08-09](2026-08/2026-08-09.md) | - New PCIe custom communication (ported from GLM tp8 work) pushes GLM-5.2 nvfp4 to 205 tok/s sustain |
| [2026-08-08](2026-08/2026-08-08.md) | DS4-Flash rapid-fire releases (r29→r31): fixed compressed-MLA verifier corruption above C24, nativ |
| [2026-08-07](2026-08/2026-08-07.md) | ormandj's SGLang DS4-Flash build fixes looping/garbage-output bug that plagued DSpark; GSM8K bench n |
| [2026-08-06](2026-08/2026-08-06.md) | - ormandj found & patched a major SGLang bug causing SWA/KV-cache retract loops on long generations  |
| [2026-08-05](2026-08/2026-08-05.md) | DS4-Flash reasoning bug fixed: HIGH was silently rendering as MAX; r27 fix confirmed — LOW or MAX  |
| [2026-08-04](2026-08/2026-08-04.md) | Critical DS4-Flash encoder bugs found: reasoning_effort MAX was mislabeled as HIGH (HIGH was a no-op |
| [2026-08-03](2026-08/2026-08-03.md) | DS4-Flash-0731 "Gilded Gnosis" r24 published — fixes the long-standing "Workspace is locked" b12x  |
| [2026-08-02](2026-08/2026-08-02.md) | DeepSeek-V4-Flash-0731 (GA) ships with a serious quality regression: multiple users (ormandj, RelayM |
| [2026-08-01](2026-08/2026-08-01.md) | DeepSeek-V4-Flash-0731 officially released (successor to preview), same DSpark speculative-decode ar |
| [2026-07-31](2026-07/2026-07-31.md) | DeepSeek V4 Flash-0731 dropped (API-only, weights pending) with post-trained-only update — claims  |
| [2026-07-30](2026-07/2026-07-30.md) | MMLU-Pro accuracy showdown: NF3/EXL3 quants match or beat NVFP4 in several categories (business, bio |
| [2026-07-29](2026-07/2026-07-29.md) | - GLM-5.2 "Gilded Gnosis" v20 saw rapid-fire docker releases (r5→r9) adding forked LMCache 0.5.2 w |
| [2026-07-28](2026-07/2026-07-28.md) | Kimi-K3 released (753B total / 104B active params, 1.56TB weights, MXFP4-trained) — massive commun |
| [2026-07-27](2026-07/2026-07-27.md) | Kimi-K3 released on HuggingFace; community immediately starts MoE kernel/quant work [(jump)](https:/ |
| [2026-07-26](2026-07/2026-07-26.md) | - GLM-5.2 v20 image ships with DCP prefill collapse fixed — auto-policy composes query-split/CKV-g |
| [2026-07-25](2026-07/2026-07-25.md) | DCP4 collapse root-caused & fixed: koush spotted GLM-5.2 vLLM fork was gathering topk scores/idx it  |
| [2026-07-24](2026-07/2026-07-24.md) | - New GLM-5.2 vLLM image fixes DCP2/4/8 crashes (illegal memory access, tensor layout stride bug); u |
| [2026-07-23](2026-07/2026-07-23.md) | - GLM-5.2 v20 Docker image ships, fixing the cuBLAS/Xid DCP read-ahead crash and adding an auto-enab |
| [2026-07-22](2026-07/2026-07-22.md) | Poolside Laguna S 2.1 (118B MoE, 8B active, 1M ctx, single-GPU fit) launched but failed community te |
| [2026-07-21](2026-07/2026-07-21.md) | - GLM-5.2 NVFP4 KV-cache bug found (outer scale unused) — fixed, KLD improved, merged as vLLM PR # |
| [2026-07-20](2026-07/2026-07-20.md) | - b12x fused-indexer IMA crash (long-ctx + spec-decode) root-caused, fix merged [(jump)](https://dis |
| [2026-07-19](2026-07/2026-07-19.md) | - New vLLM image v18→v19 (gilded-gnosis) adds CKV gather + fp8 KV-rope, pushing GLM-5.2 hybrid col |
| [2026-07-18](2026-07/2026-07-18.md) | - koush's DCP CKV-gather patch (a general fix for any DSA-based model) delivers massive prefill gain |
| [2026-07-17](2026-07/2026-07-17.md) | Kimi K3 (Moonshot) unveiled: 2.8T params, 896 experts (16 active), native MXFP4/MXFP8 training, Atte |
| [2026-07-16](2026-07/2026-07-16.md) | Bug: DSpark v10 (lucifer-cutlass) shows a big regression vs v8 — JSON schema output error rate spi |
| [2026-07-15](2026-07/2026-07-15.md) | - vLLM KLD benchmarks compromised: PR #26682's fused/combo kernels cause run-to-run rounding varianc |
| [2026-07-14](2026-07/2026-07-14.md) | GLM-5.2 vLLM v1.3 released (davidsyoung/vllm-glm52): guarded B12X A2A decode +9-11%, hitting 83 tok/ |
| [2026-07-13](2026-07/2026-07-13.md) | ds4dspark v10 released — fixes draft/acceptance randomness independence, prefix-cache token exclus |
| [2026-07-12](2026-07/2026-07-12.md) | DS4-Flash "dspark" is 50%+ faster than MTP2 but unstable — random Chinese-language token spam unde |
| [2026-07-11](2026-07/2026-07-11.md) | - New GLM-5.2-MXFP8-NVFP4-NF3-Hybrid v3.6 checkpoint hits 91.92% GPQA Diamond (up from 88.38% baseli |
| [2026-07-10](2026-07/2026-07-10.md) | - GLM-5.2-MXFP8-NVFP4-NF3-Hybrid "healed" weights (KLD 0.15→0.12, GPQA +0.5) jumped aider-polyglot |
| [2026-07-09](2026-07/2026-07-09.md) | GLM-5.2 MXFP8-NVFP4-NF3-Hybrid quant hits GPQA-Diamond 88.38% and beats REAP-504B on aider polyglot  |
| [2026-07-08](2026-07/2026-07-08.md) | GLM-5.2 v14 image: new hybrid a2a backend fixes DCP decode regression — +10% prefill / +13% decode |
| [2026-07-07](2026-07/2026-07-07.md) | GLM-5.2 v14 docker image released with new B12X backend (beats "lucifer" cutlass), instanttensor loa |
| [2026-07-06](2026-07/2026-07-06.md) | - DeepSeek-V4-Flash-DSpark added FP4 indexer cache (7-token draft) — small KV gain (+18.6k tokens) |
| [2026-07-05](2026-07/2026-07-05.md) | - jetha built a native NVFP4 attention kernel for sm120/121, speeding up Gemma 4 decode; wheels avai |
| [2026-07-04](2026-07/2026-07-04.md) | DS4-Flash v9 released — new B12X A16/A8 modes beat "lucifer-cutlass" on prefill and decode; A8 fas |
| [2026-07-03](2026-07/2026-07-03.md) | ACS redirect on c-payne PCIe switches nearly doubled token-gen throughput — clearing `ReqRedir/Cmp |
| [2026-07-02](2026-07/2026-07-02.md) | DSpark v8 (lucifer+cutlass) hits >1M context on TP2: 500 tok/s aggregate gen, 9000 tok/s prefill, 20 |
| [2026-07-01](2026-07/2026-07-01.md) | DS4-Flash v8 Docker released with b12x all-reduce enabled; real-world prefill on TP=2 lucifer-cutlas |
| [2026-06-30](2026-06/2026-06-30.md) | vLLM v8 incoming with TP=4 fix for DS4F+DSpark — Festr fixed the `--tp4` crash and reported 374– |
| [2026-06-29](2026-06/2026-06-29.md) | DSpark v7 released with prefix cache support — Festr published the v7 Docker image with DeepSpark  |
| [2026-06-28](2026-06/2026-06-28.md) | DeepSeek releases DSpark / DeepSpec: New speculative decoding paper + code claiming **57–78% faste |
| [2026-06-27](2026-06/2026-06-27.md) | New Docker image released: `eldritch-final-v1e8d565-b12x284a2ea-cu132-20260626` with all regressions |
| [2026-06-26](2026-06/2026-06-26.md) | DS4-Flash v5 recipe is broken — regressions across all kernels, was accidentally published before  |
| [2026-06-25](2026-06/2026-06-25.md) | DS4F v5 accidentally pushed — has a known crash bug in the b12x path; lucifer path unaffected. Do  |
| [2026-06-24](2026-06/2026-06-24.md) | GLM-5.2 REAP 504B Aider Polyglot final score: 89.8% pass_rate_2 after rerunning context-exhausted ca |
| [2026-06-23](2026-06/2026-06-23.md) | New Docker v12 fixes A16=1 VRAM bug, freeing 5–8GB/GPU; TP4 REAP KV cache jumps 400k → 1M tokens |
| [2026-06-22](2026-06/2026-06-22.md) | GLM-5.2 v12 Docker released with DCP+MTP fixes, VLLM_DCP_SHARD_DRAFT=1 now default — peaks **160 t |
| [2026-06-21](2026-06/2026-06-21.md) | GLM-5.2 REAP 504B aider polyglot benchmark: `pass_rate_2: 78.7` at "High" reasoning effort (225 test |
| [2026-06-20](2026-06/2026-06-20.md) | GLM 5.2 DCP+MTP root cause fixed: MTP layers were replicated per DCP rank instead of sharded, breaki |
| [2026-06-19](2026-06/2026-06-19.md) | GLM-5.2 NVFP4 on TP8 benchmarks: prefill 2,684 tok/s @ 8k ctx, 1,824 tok/s @ 128k ctx (200W power-li |
| [2026-06-18](2026-06/2026-06-18.md) | GLM-5.2 NVFP4 quant by luke uploaded (`lukealonso/GLM-5.2-NVFP4`); Aider benchmark shows pass_rate_1 |
| [2026-06-17](2026-06/2026-06-17.md) | GLM-5.2 released by zai-org: new IndexShare architecture (mandatory in 5.2, was experimental in 5.1) |
| [2026-06-16](2026-06/2026-06-16.md) | b12x decode speed regression fixed — a Codex-introduced bug in `expected_m` kernel selection cause |
| [2026-06-15](2026-06/2026-06-15.md) | Fable 5 scores 99.1% on Aider (223/225 after retry, 86.7% first-pass); Kimi K2.7 scores 91.6% for co |
| [2026-06-14](2026-06/2026-06-14.md) | KV cache bug: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` must be set in vLLM or usable KV cache tops out |
| [2026-06-13](2026-06/2026-06-13.md) | MiniMax M3 dropped (428B params, 26B active, multimodal with vision/video, 1M context, BF16 weights) |
| [2026-06-12](2026-06/2026-06-12.md) | DFlash cracked in vLLM — Festr solved vLLM dflash with help from mythos; MiMo Pro 2.5 (1T model) n |
| [2026-06-11](2026-06/2026-06-11.md) | DS4F empty-response bug confirmed reproducible: Model occasionally skips closing `</think>` tag, ret |
| [2026-06-10](2026-06/2026-06-10.md) | CUDA 13.3 released — fixes critical WGMMA data race bug causing silent result corruption in B12X N |
| [2026-06-09](2026-06/2026-06-09.md) | New Lucifer Docker image `hg436/vllm-public:lucifer-9d9a0a0` released: vLLM main + FlashInfer #3395  |
| [2026-06-08](2026-06/2026-06-08.md) | a8 vs a16 accuracy test (900 runs): Jon ran Estonia, hotel-lights, and lavd benchmarks; a8 won or ti |
| [2026-06-07](2026-06/2026-06-07.md) | DSV4-Flash prefix cache bug confirmed: On TP=4, the cache was evicting early and only utilizing ~200 |
| [2026-06-06](2026-06/2026-06-06.md) | New Docker image `voipmonitor/vllm:abyssal-abjuration-611a842` released with rebased vllm + latest f |
| [2026-06-05](2026-06/2026-06-05.md) | New image: `lavd/vllm:b12x-nameless-ascent-6-4-13.2-2` — Luke's latest `nameless-ascent` branch, d |
| [2026-06-04](2026-06/2026-06-04.md) | New Docker image `voipmonitor/dsv4-flash:lucifer-mxfp4-cutlass-20260603` released — lucifer base w |
| [2026-06-03](2026-06/2026-06-03.md) | DS4F prefix cache fix merged to vllm main ([PR #44082](https://github.com/vllm-project/vllm/pull/440 |
| [2026-06-02](2026-06/2026-06-02.md) | luke's `apotheosis` vLLM branch hits 250.3 tok/s on DS4-Flash TP=2 c=1 with full AOT compilation, si |
| [2026-06-01](2026-06/2026-06-01.md) | MiniMax M3 launched — multimodal (text/image/video→text), 1M-token context, MiniMax Sparse Atten |
| [2026-05-31](2026-05/2026-05-31.md) | DS V4 Flash on tp=4 hits ~200 t/s decode, 94.6% prefix cache hit rate, ~3M KV cache with `cstechdev/ |
| [2026-05-30](2026-05/2026-05-30.md) | DSv4 Flash hitting 245-250 tok/s on TP4 SM120 with luke's unholy-fusion branch + b12x MoE + MTP; luc |
| [2026-05-29](2026-05/2026-05-29.md) | DSV4-Flash "unholy-fusion" branch hits 245 t/s decode (MTP off ~130, MTP on approaching 200): Luke m |
| [2026-05-28](2026-05/2026-05-28.md) | DSV4-Flash SM120 breakthrough: `lucifer1004/dsv4-flash-sm120:latest` achieves full DeepGEMM prefill/ |
| [2026-05-27](2026-05/2026-05-27.md) | Luke fixed a long-context bug in both sglang and vllm; took the jasl branch, stripped its kernels, s |
| [2026-05-26](2026-05/2026-05-26.md) | DSv4-Flash docker `jasl-dsv4-5-23-13.2` (cu132): gen throughput ~35→55 tok/s; Lavd expects 70+ onc |
| [2026-05-25](2026-05/2026-05-25.md) | b12x DS V4 Flash sglang push: luke released new b12x build with MTP support; early results show **~4 |
| [2026-05-24](2026-05/2026-05-24.md) | DSV4 Flash: 82 tok/s C=1 MTP=OFF on sglang+b12x (TP=2), 79 tok/s at 128K context — C=2 near-linear |
| [2026-05-23](2026-05/2026-05-23.md) | DSv4 Flash: 1400 t/s on GB10 at 128k context using w4a8 cutlass; sglang+b12x shows zero decode degra |
| [2026-05-22](2026-05/2026-05-22.md) | RTX 6000 Pro price hike ~20% worldwide attributed to DDR7 memory costs; still selling [(jump)](https |
| [2026-05-21](2026-05/2026-05-21.md) | vLLM PR #40082 merged — FlashInfer b12x backends for SM120/SM121 (RTX Pro 6000 Blackwell / DGX Spa |
| [2026-05-20](2026-05/2026-05-20.md) | RTX Pro 6000 prices up 30-37% in South Korea and Europe vs. last week; members panic-buying at $9,99 |
| [2026-05-19](2026-05/2026-05-19.md) | DGX Spark DSv4: 700-800 prefill t/s at 128-200k ctx, 32 t/s decode C=1 MTP=2 [(jump)](https://discor |
| [2026-05-18](2026-05/2026-05-18.md) | DSv4-Flash MTP accuracy regression fixed in `lavd/vllm:jasl-dsv4-5-16-26`; prefix caching confirmed  |
| [2026-05-17](2026-05/2026-05-17.md) | DeepSeek V4 Flash on 2× RTX PRO 6000: pangfather shared full vLLM config (TP=2, 393K ctx, MTP=2, FP |
| [2026-05-16](2026-05/2026-05-16.md) | DSV4 Flash: 120 tok/s on TP=2; new `lavd/vllm:jasl-dsv4-5-15-26` image fixes cpasync import error fr |
| [2026-05-15](2026-05/2026-05-15.md) | Qwen 27B runaway thinking discovery: unbounded thinking on GPQA Diamond gives only 40.4% accuracy be |
| [2026-05-14](2026-05/2026-05-14.md) | infernix's GPT-5.5 autoresearch branch for DSv4 Flash on SM120 claims 4.8k prefill/s and 27/30 Eston |
| [2026-05-13](2026-05/2026-05-13.md) | DSV4 Flash MTP2: 35→120 tok/s after jasl vllm fixes; new image `lavd/vllm:jasl-dsv4-5-12-26` hits  |
| [2026-05-12](2026-05/2026-05-12.md) | repne/vllm:v3 released with tinygemm_bf16; Qwen3.6-27B FP8 TP=1 benchmarks: prefill 9,057 tok/s @8k  |
| [2026-05-11](2026-05/2026-05-11.md) | MiniMax M2.7 reaches 145 tok/s (TP=2) with latest b12x 0.13.5 + `B12X_MOE_FORCE_A16=1`, double the o |
| [2026-05-10](2026-05/2026-05-10.md) | DS4 Flash on 2x RTX Pro 6000 (FP8+FP4): 39 tok/s single, 79 tok/s dual. MTP + CUDA graph conflict � |
| [2026-05-09](2026-05/2026-05-09.md) | GLM-5.1 DCP=4 fixed: `voipmonitor/vllm:glm51-kimi-comm-20260508` eliminates garbage output at DCP=4; |
| [2026-05-08](2026-05/2026-05-08.md) | MiMo-V2.5 looping fix found: `do_sample=false` in generation_config.json disabled sampling params; f |
| [2026-05-07](2026-05/2026-05-07.md) | MiMo V2.5 thought loops confirmed as model-level bug: Luke ruled out quantization, kernels, and MTP  |
| [2026-05-06](2026-05/2026-05-06.md) | Gemma-4 MTP spec decode massive uplift: PR #41745 cherry-picked into vLLM nightly; single-stream 37  |
| [2026-05-05](2026-05/2026-05-05.md) | GLM-5.1 MTP corruption fixed in new Docker image (`voipmonitor/vllm:glm51-mtp-b12xsparse-ficutlass-t |
| [2026-05-04](2026-05/2026-05-04.md) | Luke cracked MiMo V2.5 MTP — root cause was TP tensors internally shuffled in a "dastardly rearran |
| [2026-05-03](2026-05/2026-05-03.md) | GLM-5.1 OOV loop bug: token IDs ≥154870 unmapped in tokenizer → infinite decode loops at temp=0; |
| [2026-05-02](2026-05/2026-05-02.md) | MiMo-V2.5 stream-race bug fixed: b12x/SGLang schedule/forward stream race on `req_to_token`; `jumper |
| [2026-05-01](2026-05/2026-05-01.md) | MiMo-V2.5 NVFP4 now runnable on TP=2 via luke's b12x 0.11.0 + sglang fork; ~80 tok/s single-batch, f |
| [2026-04-25](2026-04/2026-04-25.md) | DS V4/Flash blocked on SM120: DeepGEMM explicitly confirmed no SM120 support planned. Both DS V4 Pro |
| [2026-04-24](2026-04/2026-04-24.md) | DeepSeek V4 dropped (MIT): V4-Pro 1.6T params/49B active, Flash 284B. Flash fits 2× RTX 6000 Pro (~ |
| [2026-04-23](2026-04/2026-04-23.md) | Qwen 3.6 27B + FP8 released — 40–130 tok/s on single RTX 6k with MTP3, 70–90% acceptance rate; |
| [2026-04-22](2026-04/2026-04-22.md) | Kimi K2.6 hits 120–130 tok/s on 8×RTX 6000 Pro with DCP8 + Eagle3 speculative decoding + 3.6M fp8 |
| [2026-04-21](2026-04/2026-04-21.md) | Kimi K2.6 released (1.1T params, same arch as K2.5, vision included): SWE-Bench Pro 58.6%, HLE beati |
| [2026-04-20](2026-04/2026-04-20.md) | GLM-5.1 vLLM OOM bug fixed — new Docker `voipmonitor/vllm:glm51-tp8-nodcp-mtp3-tritondraft-b12x095 |
| [2026-04-19](2026-04/2026-04-19.md) | PCIe P2P allreduce config for ~10% throughput gain: Force-enable P2P via modprobe `options nvidia NV |
| [2026-04-18](2026-04/2026-04-18.md) | EXL3 benchmark breakthrough: mratsim's Qwen3.5-397B-A17B EXL3 quant achieves **1500 pp/s and 50+ tg/ |
| [2026-04-17](2026-04/2026-04-17.md) | Qwen3.6-35B-A3B released (MoE); community upset the poll-winner 27B wasn't chosen first — speculat |
| [2026-04-16](2026-04/2026-04-16.md) | b12x FP4 GEMM kernel merged into FlashInfer ([jump](https://discord.com/channels/1466898002793857221 |
| [2026-04-15](2026-04/2026-04-15.md) | Introspective Diffusion (I-DLM-8B) matches AR model quality, beats LLaDA-2.1-mini (16B) by +26 AIME- |
| [2026-04-14](2026-04/2026-04-14.md) | GLM 5.1 with native NSA/DSA attention now working on TP=8 with NVFP4 weights + FP8 KV cache via cust |
| [2026-04-13](2026-04/2026-04-13.md) | MiniMax M2.7 NVFP4 quant released by luke, then updated mid-day incorporating Jon's calibration data |
| [2026-04-12](2026-04/2026-04-12.md) | MiniMax M2.7 dropped (~1am) with same architecture as M2.5 — just a weight update. luke confirmed  |
| [2026-04-11](2026-04/2026-04-11.md) | vLLM MTP reaches sglang parity after a week of work by Festr; key insight was using sglang's eager-m |
| [2026-04-10](2026-04/2026-04-10.md) | GLM-5.1 NVFP4 quant live on 8x RTX Pro 6000: 131 tok/s with MTP, 95 tok/s via flashinfer cutlass. Bu |
| [2026-04-09](2026-04/2026-04-09.md) | GLM-5.1 NVFP4 upload by luke — 52 tok/s single request on 8x RTX 6000 PRO (NVFP4, b12x, sglang), t |
| [2026-04-08](2026-04/2026-04-08.md) | b12x hits 220 t/s single-user decode on 2x RTX PRO 6000; 198 t/s on Qwen3.5-122B vs 131 t/s baseline |
| [2026-04-07](2026-04/2026-04-07.md) | GLM-5 NVFP4 on 8×RTX PRO 6000: SGLang+MTP leads at 99 tok/s single-user (0 ctx), 249.8 tok/s at C=4 |
| [2026-04-06](2026-04/2026-04-06.md) | Qwen3.5 397B on 4×RTX PRO 6000: sglang record 108 tok/s (no MTP) / 180 tok/s (with MTP); vLLM now 9 |
| [2026-04-05](2026-04/2026-04-05.md) | vLLM 0.19 MoE TP regression identified and fixed — PR vllm-project/vllm#38990 awaiting merge; bran |
| [2026-04-04](2026-04/2026-04-04.md) | Qwen3.5-397B NVFP4 benchmarks on 4× RTX PRO 6000 (sglang b12x 0.7.2): MTP on = 180 tok/s single-use |
| [2026-04-03](2026-04/2026-04-03.md) | b12x 0.7.1 fixes OOM and >8 concurrency crashes on SM120; attention backend now launches for all bat |
| [2026-04-02](2026-04/2026-04-02.md) | b12x v0.7.0 released: fixes OOM on prefill, adds attention backend support (bs=1 only for now); new  |
| [2026-04-01](2026-04/2026-04-01.md) | PCIe oneshot AllReduce + fusion gives +7-8% decode throughput on SM120 across all tested models (Qwe |
| [2026-02-28](2026-02/2026-02-28.md) | GLM-5 NVFP4 on 8 cards: 44 tok/s @ 0 ctx, 30 @ 150k; 4000 tok/s prefill at 400W |
| [2026-02-27](2026-02/2026-02-27.md) | Kimi K2.5 decode on Turin (8 cards, DCP=8): 65 tok/s @ 0 ctx, 36 @ 100k, 27 @ 200k |
| [2026-02-26](2026-02/2026-02-26.md) | Power benchmark: 300W→500W gives up to 30% more throughput at 64 concurrency for MiniMax M2.5 NVFP4 |
| [2026-02-25](2026-02/2026-02-25.md) | vLLM PR #34424: FP8 GEMM sm120 optimizations with smaller M-dimension kernels |
| [2026-02-24](2026-02/2026-02-24.md) | NCCL tuning on AMD XGMI: `NCCL_P2P_DISABLE=1` alone insufficient; needs full combo: `NCCL_MIN_NCHANN |
| [2026-02-23](2026-02/2026-02-23.md) | NCCL v2.28.3 bug: hardcodes AMD_BW=16 GB/s for all AMD CPUs — 12-16× underestimate for Turin xGMI3;  |
| [2026-02-22](2026-02/2026-02-22.md) | `NCCL_P2P_LEVEL=SYS` env vars close speed gap between 2-CPU and switch setups: 70 tok/s without DCP  |
| [2026-02-21](2026-02/2026-02-21.md) | Root cause of Kimi K2.5 speed gap on 2-CPU AMD system: PCIe P2P traffic must cross CPU SMP interconn |
| [2026-02-20](2026-02/2026-02-20.md) | 16-GPU Kimi K2.5 (TP16) achieves 40 tok/s single-prompt and ~1400 tok/s aggregate; TP16 beats PP2+TP |
| [2026-02-19](2026-02/2026-02-19.md) | Qwen 3.5-397B FP8 running on 8×RTX with SGLang: 75-125 tok/s, MTP speculative decoding confirmed wor |
| [2026-02-18](2026-02/2026-02-18.md) | GLM-5 NVFP4 running but only 35-36 tok/s; expected ~50 tok/s from luke's setup — investigating discr |
| [2026-02-17](2026-02/2026-02-17.md) | SGLang FP8 KV cache bug: scales applied on write but not un-applied on read — gibberish on cache reu |
| [2026-02-16](2026-02/2026-02-16.md) | Qwen 3.5 released at 807 GB weights — too large for FP8 on 8×RTX; waiting for official FP8/NVFP4 qua |
| [2026-02-15](2026-02/2026-02-15.md) | GLM 4.7 FP8 SGLang launch command shared with triton FP8 kernel, 4-GPU TP, 200K context |
| [2026-02-14](2026-02/2026-02-14.md) | GLM-5 unsupported on SM120 entirely in both vLLM and SGLang — NVFP4 the only future path |
| [2026-02-13](2026-02/2026-02-13.md) | MiniMax M2.5 FP8 confirmed running on 8x RTX 6000 Pro: 70 tok/s single, 122 tok/s dual connection, 7 |
| [2026-02-12](2026-02/2026-02-12.md) | MiniMax M2.1 working on 4x RTX 6000 Pro with official vLLM FP8 (8-bit), 96G×4 supports ~400K KV cach |
