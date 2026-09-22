# Kimi-K3 on RTX PRO 6000 Blackwell

For nine GPUs, use [QSRT-K2 with Red Hat DSpark K5](qsrt-tp9-dspark.md).
It includes a complete Docker launch with TP9/DCP9, vision, native CPU KV
offload and a source-locked CUDA 13.4 image.

For the official checkpoint on sixteen GPUs, see
[Kimi-K3 MXFP4 Runtime](production-runtime.md), which documents the CUDA 13.3
and PyTorch 2.13 deployment. Its measurements apply to that image, not the
QSRT-K2 profile.

## Runtime and Evaluation Documents

| Document | Purpose | Status |
|---|---|---|
| [QSRT-K2 TP9 with Red Hat DSpark](qsrt-tp9-dspark.md) | Docker launch, configuration, benchmarks and source reconstruction | qualified |
| [QSRT TP9 source and validation report](qsrt-tp9-dspark-qualification.md) | Karmic/B12X merge list, exact source composition, numerical checks and profile evidence | qualified |
| [Kimi-K3 MXFP4 Runtime](production-runtime.md) | Source-locked Docker, DSpark, DFlash, target-only decode, native host KV offload, vision, LLMConduit, tools, and Oh My Pi | qualified |
| [Native host KV offload](native-host-kv-offload.md) | Process-shared RAM KV reuse for official MXFP4 with DSpark on TP16/DCP16 | qualified |
| [Full MXFP4 4096-token prefill](full-mxfp4-p4096-prefill.md) | Exact 4096-token scheduler chunks with physical 1M KV capacity on TP16/DCP16 | research-only |
| [Source-locked serving receipt](validation/source-locked-runtime-20260816.json) | No-speculation, DSpark, and DFlash source composition and runtime evidence | qualified |
| [Distribution-fidelity reference](distribution-fidelity-1024x2048.md) | Teacher-forced hidden-state and KLD comparison over 1,024 contexts | implemented |
| [AA-LCR reproduction](aa-lcr-reproduction.md) | Reproducible capability comparison protocol | qualified |
| [Official MXFP4 versus QSRT K2](aa-lcr-official-mxfp4-vs-qsrt-k2.md) | Paired AA-LCR comparison | qualified |
| [QSRT K2 TP16/DCP8](qsrt-k2-tp16-dcp8.md) | Target-only and DSpark serving for the QSRT K2 checkpoint | qualified |
| [Red Hat DSpark DCP16](redhat-dspark-dcp16.md) | RedHatAI BF16 draft compatibility | qualified |

Machine-readable receipts are stored under [`validation/`](validation/).
Repository tools used by the evaluation documents are stored under
[`tools/`](tools/).
