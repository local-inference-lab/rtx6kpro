# RTX PRO 6000 Blackwell Wiki Sources

Use the [https://github.com/local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) field wiki to select a documented serving configuration, find prior methodology, and understand topology-sensitive results. Execute new measurements with the pinned benchmark tool and preserve new raw output.

## Lookup order

1. Resolve the current `master` commit and record its full 40-character SHA.
2. Start from [`README.md#start-here`](https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md#start-here), then open the current model hub/runbook.
3. Use [`INDEX.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/INDEX.md) for discovery only when the front page or hub does not link the needed material.
4. Use [`GLOSSARY.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/GLOSSARY.md) for metric and runtime terminology.
5. Publish commit-pinned wiki links rather than moving `master` links.

## Benchmark sources

- Current model runbook: source for the documented server launch, checkpoint, image, TP/DCP, cache, speculation, graph, scheduler, startup markers, and stated limitations.
- [`benchmarks/results.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/results.md): consolidated result discovery and historical context. Do not use an aggregate table as a control unless its exact artifact and conditions match.
- [`benchmarks/inference-throughput/README.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/inference-throughput/README.md): example of a context × concurrency matrix with explicit environment, server configuration, MTP on/off controls, raw files, and bounded interpretation.
- [`benchmarks/glm52-kld-evaluation.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/glm52-kld-evaluation.md): current GLM-5.2/vLLM KLD route.
- [`benchmarks/kld-evaluation.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/kld-evaluation.md): general and older Qwen/SGLang KLD workflow; it explicitly defers current GLM-5.2 reproduction to the GLM-specific page.
- [`benchmarks/mtp-quality-evaluation.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/mtp-quality-evaluation.md): with/without-MTP quality comparison pattern and exact configuration reporting.
- [`benchmarks/nvfp4-quantization-comparison.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/benchmarks/nvfp4-quantization-comparison.md): quantization comparison context.
- [`optimization/speculative-decoding.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/speculative-decoding.md): MTP/DSpark/DFlash background and caveats.
- [`hardware/topology.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/topology.md): required context for PCIe, NUMA, P2P, and collective-sensitive claims.
- [`hardware/blackwell-power-limit-sweep.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/blackwell-power-limit-sweep.md): methodology context when a claim depends on power limit, clocks, thermals, or performance per watt.
- [`optimization/nccl-tuning.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/nccl-tuning.md) and [`optimization/pcie-oneshot-allreduce.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md): transport context when changing collective configuration; validate on the target topology rather than inheriting a result.

A wiki result is prior evidence, not the output of the current run. Record it as a comparison source only after confirming image/source/model identities, client command, hardware, topology, cache state, contexts, concurrency, duration or output limit, and aggregation.
