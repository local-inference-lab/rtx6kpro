# GLM-5.2 Runbook Hub

Use this page as the stable entry point for GLM-5.2 on RTX PRO 6000
Blackwell. Versioned pages preserve reproducible serving contracts and
benchmark evidence.

## Recommended Deployment

| Need | Page |
|---|---|
| Serve NVFP4 or projection-mixed EXL3 on CUDA 13.3 | [GLM-5.2 Infernal Invocation r11](glm5.2-infernal-invocation-r11.md) |
| Configure NIXL Prefill/Decode disaggregation | [GLM-5.2 NIXL P/D Disaggregation](glm5.2_nixl_pd.md) |
| Run the MXFP4 expert checkpoint | [GLM-5.2 FP8 + MXFP4 Experts](glm5.2_mxfp4.md) |
| Compare KLD and quant quality | [GLM-5.2 KLD Evaluation](../benchmarks/glm52-kld-evaluation.md) |
| Reproduce GGUF/BF16 dequant KLD | [GGUF to BF16 Dequant KLD Audit](glm5.2/glm52-gguf-bf16-dequant-kld-2026-07-08.md) |
| Reproduce unsloth-style prefill KLD | [Unsloth-Style Prefill KLD Reproduction](glm5.2/glm52-unsloth-style-prefill-kld-2026-07-07.md) |

## Serving Contract

| Area | Guidance |
|---|---|
| Recommended image | Infernal Invocation r11, documented in [its qualified runbook](glm5.2-infernal-invocation-r11.md) |
| Main checkpoint | `lukealonso/GLM-5.2-NVFP4` |
| Main backend | B12X sparse MLA, B12X MoE, B12X dense/FP8 path unless a page says otherwise |
| DCP | [GLM-5.2 v18](glm5.2_v18.md) contains broad DCP1/2/4/8 measurements; Infernal Invocation r11 is qualified only for DCP1 |
| Spec decode | Infernal Invocation r11 is qualified with MTP3 for the listed NVFP4 and EXL3 profiles |
| Quality checks | KLD, long-context decode, coding probes, acceptance rates |

## Version Map

| Page | Status | Why keep it |
|---|---|---|
| [GLM-5.2 Infernal Invocation r11](glm5.2-infernal-invocation-r11.md) | Qualified | CUDA 13.3 NVFP4 TP8 and projection-mixed EXL3 R7 TP4 serving contracts. |
| [GLM-5.2 v20](glm5.2_v20.md) | Historical Gilded Gnosis reference | Gilded Gnosis/SparkInfer image, cuBLAS-safe DCP output, and TP6 release evidence. |
| [GLM-5.2 v19](glm5.2_v19.md) | Historical Gilded Gnosis predecessor | Canonical GG migration, deterministic CuTe cache keys, and DCP optimization background. |
| [GLM-5.2 v18](glm5.2_v18.md) | Broad benchmark reference | Complete Gilded Gnosis DCP, TP6, MTP3, NF3, KLD, and checkpoint tables. |
| [GLM-5.2 v17](glm5.2_v17.md) | Historical Gilded Gnosis predecessor | Earlier fast-DCP integration and validation history. |
| [GLM-5.2 v16](glm5.2_v16.md) | Historical Gilded Gnosis predecessor | First unified GG release and TP6 validation history. |
| [GLM-5.2 v15](glm5.2_v15.md) | Historical Fathomless baseline | Fathomless validation and Docker/source pins. |
| [GLM-5.2 v14](glm5.2_v14.md) | Historical Eldritch predecessor | Online FP8-MXFP8 overlay, KLD, DCP and TP6 validation history. |
| [GLM-5.2 v13](glm5.2_v13.md) | Historical Eldritch baseline | Earlier B12X/DCP/MTP production recipe and speed references. |
| [GLM-5.2 v12](glm5.2_v12.md) | Historical Dark Devotion baseline | Useful for comparing older DCP/MTP acceptance behavior. |
| [GLM-5.2 v11](glm5.2_v11.md) | Historical quant comparison | NVFP4, FP8, MXFP8 checkpoint comparison work. |
| [GLM-5.2 v14 pending PRs](glm5.2_v14_pending_prs.md) | Temporary tracker | PR carry-over notes from the v14 development cycle. |

## Operational Reminders

- Keep the exact `index_topk_pattern` from the selected runbook. A truncated
  pattern can silently degrade GLM output.
- For DCP runs, use the DCP policy documented by the selected image runbook.
- For NIXL P/D runs, set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` on every service. When
  DCP is 4, also set `DCP_INDEXER_SHARDS=4` on both Prefill and Decode.
- Keep the P/D execution modes asymmetric: Prefill uses `--enforce-eager` and
  Decode uses `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
- Do not compare throughput across pages unless the image, DCP mode, MTP mode,
  graph capture size, batch tokens, and GPU placement match.
- KLD is a regression and quantization sanity check, not a full task-quality
  metric.
