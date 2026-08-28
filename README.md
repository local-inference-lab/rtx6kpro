# RTX PRO 6000 Blackwell LLM Wiki

This repository is a field wiki for running frontier LLMs on NVIDIA RTX PRO
6000 Blackwell / SM120 PCIe systems. It is more than a few launch snippets: it
contains reproducible Docker builds, exact vLLM and SGLang runbooks, benchmark
tables, KLD quality checks, quantization notes, DCP/MTP/DSpark/DFlash debugging,
PCIe topology work, and regression history.

> Community workbench for RTX PRO 6000 / SM120 serving:
> https://discord.gg/X54jjmcxWJ

## Start Here

If you just want to run a model, use these stable hub pages first:

| Model family | Start here | Scope |
|---|---|---|
| GLM-5.2 | [GLM-5.2 Runbook Hub](models/glm-5.2.md) | Fathomless vLLM, NVFP4, online FP8/MXFP8, B12X, DCP, MTP, KLD. |
| DeepSeek-V4-Flash / DSpark | [DeepSeek-V4-Flash Runbook Hub](models/deepseek-v4-flash.md) | Standard checkpoint, MTP, DSpark, B12X, Lucifer, CUTLASS. |
| Kimi | [Kimi Runbook Hub](models/kimi.md) | Kimi-K2.7-Code, DFlash, parser/tool-call runtime. |
| Xiaomi MiMo | [MiMo Runbook Hub](models/mimo.md) | MiMo V2.5 Pro FP4-DFlash. |
| Qwen3.8-27B | [Qwen3.8-27B on RTX PRO 6000 Blackwell](models/qwen38-27b.md), [readable QSRT K5 training result](models/qwen38-qsrt-k5-training-result.md), [exact QSRT K5 specification](models/qwen38-qsrt-k5-r16.md) | TP1, TP2, and TP4 throughput evidence plus the QSRT K5 training interpretation, artifact, fidelity, runtime, and source contract. |
| Qwen3.8-Flash-Next | [Qwen3.8-Flash-Next on two RTX PRO 6000 Blackwell GPUs](models/qwen38-flash-next.md) | vLLM TP2 FP8 and NVFP4 launch recipes, PLE offload, and NVFP4 loader patch. |
| GLM-5.1 | [GLM-5.1 Runbook Hub](models/glm-5.1.md) | Historical GLM-5.1, KLD methodology, older B12X/SGLang work. |
| Legacy / secondary models | [Legacy Model Runbooks](models/legacy.md) | DeepSeek-V4-Pro, GLM-4.7, Qwen, MiniMax, older Kimi pages. |

Need the complete map of every Markdown page?

| Index | Use |
|---|---|
| [Full Wiki Index](INDEX.md) | Complete generated catalog of every page in this repository. |
| [Glossary And Acronym Guide](GLOSSARY.md) | Acronym expansions and writing rules for newcomer-friendly docs. |
| [Newcomer Onboarding](docs/newcomer-onboarding.md) | How to ask useful questions without lowering the technical signal. |

## What Is In This Repository?

| Need | Where |
|---|---|
| Copy/paste production launch commands | Model hubs and current versioned model pages. |
| Rebuild the Docker image | [Eldritch Docker](models/eldritch-enlightenment-docker.md), current model image sections, and build scripts in [scripts](scripts/). |
| Compare backend speed | Model benchmark tables plus [Benchmark Results](benchmarks/results.md). |
| Check quantization quality | [GLM-5.2 KLD](benchmarks/glm52-kld-evaluation.md), [KLD Evaluation](benchmarks/kld-evaluation.md), and model-specific KLD sections. |
| Understand MTP, DSpark, or DFlash | [Speculative Decoding](optimization/speculative-decoding.md), DS4/Kimi/MiMo pages. |
| Debug topology or PCIe behavior | [Topology](hardware/topology.md), [PCIe Bandwidth](hardware/pcie-bandwidth.md), [GPU Configurations](hardware/gpu-configs.md). |
| Avoid known runtime footguns | [Common Issues](troubleshooting/common-issues.md), model caveats, and daily summaries. |
| Understand old measurements | Historical versioned pages and [Daily Summaries](daily-summaries/). |

## Recommended Production-Style Pages

| Area | Page | Why it matters |
|---|---|---|
| GLM-5.2 serving stack | [GLM-5.2 Infernal Invocation r18](models/glm5.2-infernal-invocation-r18.md) | Source-qualified CUDA 13.3 profiles with sparse-prefill row validation, projection-mixed EXL3 TP4, online MCG K6, and NVFP4 TP8. |
| GLM-5.2 MXFP4 | [GLM-5.2 FP8 + MXFP4 Experts](models/glm5.2_mxfp4.md) | Native MXFP4 expert checkpoint path and A8 serving notes. |
| DS4 serving profile | [DS4 DSpark Infernal Invocation r18](models/ds4dspark-infernal-invocation-r18.md) | 0731 checkpoint, fixed K5, FULL graph dispatch, strict-tool concurrency, and native plus LMCache filesystem replay. |
| DS4 full reference | [DS4 DSpark v9](models/ds4dspark-v9.md) | Full DSpark and standard MTP sweep reference. |
| Kimi-K2.7-Code | [Kimi-K2.7-Code v3](models/kimi-k27-code_v3.md) | Fathomless Kimi DFlash validation. |
| MiMo FP4-DFlash | [MiMo FP4-DFlash v3](models/xiaomi-mimo-v2.5-pro-fp4-dflash_v3.md) | MiMo DFlash validation and fix notes. |

Older pages are intentionally preserved. Prefer the hub page for each model
family unless you are reproducing a specific old result.

## Core Topics

| Topic | Page |
|---|---|
| Docker images and release lines | [Docker Images](optimization/docker-images.md) |
| PCIe oneshot all-reduce | [PCIe oneshot all-reduce](optimization/pcie-oneshot-allreduce.md) |
| NCCL tuning and empty graph-file failures | [NCCL tuning](optimization/nccl-tuning.md) |
| Speculative decoding | [Speculative decoding](optimization/speculative-decoding.md) |
| NVFP4 quantization | [NVFP4 quantization](optimization/nvfp4-quantization.md) |
| Hybrid NVFP4 assembly | [Hybrid NVFP4 assembly](optimization/hybrid-nvfp4-assembly.md) |
| B12X FP8 / DeepGEMM comparison | [B12X dense FP8 GEMM vs DeepGEMM](optimization/b12x-dense-fp8-gemm-vs-deepgemm.md) |
| B12X W4A8 tiny decode | [B12X W4A8 MX tiny decode](optimization/b12x-w4a8mx-tiny-decode-kernel.md) |
| DSpark upstream consolidation | [DSpark upstream consolidation](optimization/dspark-upstream-consolidation.md) |
| I/O tuning | [I/O tuning](optimization/io-tuning.md) |

## Benchmarks And Quality

| Area | Page |
|---|---|
| Consolidated throughput | [Benchmark Results](benchmarks/results.md) |
| vLLM vs SGLang throughput | [Inference throughput](benchmarks/inference-throughput/README.md) |
| GLM-5.2 KLD and quant quality | [GLM-5.2 KLD Evaluation](benchmarks/glm52-kld-evaluation.md) |
| General KLD methodology | [KLD Evaluation](benchmarks/kld-evaluation.md) |
| MTP quality checks | [MTP Quality Evaluation](benchmarks/mtp-quality-evaluation.md) |
| NVFP4 quantization comparison | [NVFP4 Quantization Comparison](benchmarks/nvfp4-quantization-comparison.md) |

KLD is a regression and quantization-sanity tool, not a complete quality metric.
Use it together with long-context decode, coding probes, acceptance-rate checks,
and task-level benchmarks.

## Hardware And Topology

Most current measurements target RTX PRO 6000 Blackwell / GB202 / SM120 cards:
96 GB GDDR7 per GPU, PCIe 5.0 x16, no NVLink, usually 4-GPU, 8-GPU, or 16-GPU
PCIe-switch systems.

| Area | Page |
|---|---|
| SM120 vs SM100 | [SM120 vs SM100 Architecture](hardware/sm120-vs-sm100-architecture.md) |
| PCIe topology | [Topology](hardware/topology.md) |
| PCIe bandwidth | [PCIe Bandwidth](hardware/pcie-bandwidth.md) |
| GPU configs | [GPU Configurations](hardware/gpu-configs.md) |
| ASUS ESC8000A-E13P | [ASUS ESC8000A-E13P + Broadcom Switches](hardware/asus-esc8000a-e13p-broadcom-switches.md) |
| ASRockRack Turin 16 GPU | [ASRockRack + EPYC Turin + 4x c-payne](hardware/asrockrack-turin-cpayne-16gpu.md) |
| ASRock WRX90 16 GPU | [ASRock WRX90 + 4x c-payne](hardware/wrx90-cpayne-16gpu-4switch.md) |
| Power tuning | [Blackwell power limit sweep](hardware/blackwell-power-limit-sweep.md) |

## Inference Engines

| Engine | Page | Current role |
|---|---|---|
| vLLM | [vLLM](inference-engines/vllm.md) | Primary runtime for current GLM-5.2, DS4, Kimi, and MiMo pages. |
| FlashInfer | [FlashInfer](inference-engines/flashinfer.md) | SM120 sparse MLA, CUTLASS MoE, sampler, and kernel integration notes. |
| SGLang | [SGLang](inference-engines/sglang.md) | Historical and alternate runtime notes, especially older GLM/MiMo paths. |

## Acronym Policy

The wiki uses many acronyms: DCP, MTP, DSpark, DFlash, MLA, MoE, KLD, TP, CC,
P2P, NVFP4, MXFP8, and more. To make pages readable for newcomers:

- Expand important acronyms on first use: `Decode Context Parallelism (DCP)`.
- Do not expand acronyms inside commands, Docker tags, environment variables,
  JSON, file paths, or raw logs.
- Use [Glossary And Acronym Guide](GLOSSARY.md) as the source of truth.
- Run `python3 scripts/check-acronyms.py` before polishing a major page.

## Keeping The Community Useful

The wiki should reduce accidental gatekeeping: newcomers should be able to find
the right runbook, decode acronyms, and reproduce a known-good launch without
needing tribal knowledge. That does not mean every Discord question can be
answered from memory.

Use [Newcomer Onboarding](docs/newcomer-onboarding.md) as the support contract:
bring the model page, Docker image, full launch command, GPU layout, TP/DCP/MTP
or DSpark settings, client command, and logs. That keeps the server welcoming
without turning it into an unstructured support queue.

## Common Operational Rules

- Do not launch with `NCCL_GRAPH_FILE=` set to an empty string. Unset it if no
  real XML graph file is used.
- Reuse cache directories while debugging; otherwise TileLang, Triton, CuTe,
  and FlashInfer rebuilds dominate iteration time.
- For quick smoke tests, use small `MAX_NUM_SEQS` and graph caps. For published
  tables, use the graph sizes documented in the model page.
- For DFlash and DSpark, confirm backend markers and acceptance rates before
  trusting throughput numbers.
- For GLM-5.2, keep the exact `index_topk_pattern` and DCP policy from the
  relevant runbook; a truncated pattern can silently degrade output.

## Maintaining The Wiki

When adding a page:

- Link it from the relevant model hub.
- Add a short status block if it is a current runbook.
- Keep exact Docker image tags, source commits, model snapshot IDs, GPU layout,
  backend choices, and benchmark commands.
- Regenerate the full index:

```bash
python3 scripts/generate-wiki-index.py > INDEX.md
```

For performance claims, include both the server launch config and the client
command so results can be reproduced on another PCIe-only Blackwell host.

Maintained from community Discord experiments through July 2026.
