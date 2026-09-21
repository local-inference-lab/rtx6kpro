# Karmic Kraken public-report validation

The published beta fixes the GLM TP4/DCP4/MTP3 startup assertion and includes
an alias-safe B12X KDA decode binding. Text-prefix LMCache restores pass on
the Spark TP2 preset. The exact nearly-full-cache admission reported by a
community user remains unqualified.

## Tested image and conditions

- Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260921-22cb9b8298caa06e`.
- Digest: `sha256:7ee79a417df23504779130d3acddb0854395a307d1a217c590e18d4f51269910`.
- vLLM: `64b752317aecc2199d030a730f7a3db0bd147684`.
- B12X: `6b80ac55b93bb3684fc35ff3f2dc34424e9f9eab`.
- Hardware: RTX PRO 6000 Blackwell Max-Q, stock clocks. No overclock or
  local source mounts were used for the GLM checks below.
- Sampling: temperature 1, top-p 0.95. Short correctness requests respect EOS.

## Startup routing

[vLLM #822](https://github.com/local-inference-lab/vllm/pull/822) distinguishes
a long prefill carrying a zero-draft marker from speculative decode. The
reported 4,096-token profile batch previously entered a path prepared for
128 decode tokens. This was a capacity/routing assertion, not GPU exhaustion.

Qualified: the published image starts GLM-5.3-Flash-NVFP4 with TP4/DCP4/MTP3,
captures full/piecewise CUDA graphs, answers a short arithmetic request and
accepts a 39,026-token input. The checkpoint revision is
`520de24eabf507659eaef7c70f14fd584527facc`.

## B12X gate/output alias and text restore

[vLLM #824](https://github.com/local-inference-lab/vllm/pull/824) preserves
the binding contract that the mutable output must not overlap the live
read-only GDN gate. Only that exact alias case computes into a separate
destination and copies the result back after the gate has been consumed.
Other binding errors are not swallowed; the normal path is unchanged.

The serving checks use `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision
`a608241037e4c2565356bff7ca293f2133888f88`, TP2/DCP2/MTP3, batch budget 3072,
four maximum sequences, and 4,190,109,696 bytes of GPU KV per rank. LMCache
uses 16 GiB RAM, 64 GiB disk and 3072-token chunks.

| Check | Observation | Conclusion |
|---|---|---|
| 59,542-token prompt, clear native GPU prefix, repeat | All 59,542 tokens restored externally; answer remains 13. Initial cold/restore times were 6.409/0.305 s. A later restore took 1.421 s and also passed. | Qualified text restore; these are individual latency observations, not a speedup benchmark. |
| 104,538-token restore after starting three long requests | Full external hit, answer 13, healthy engine; measured KV peak 88.9%. One pressure request had finished before the restore was admitted. | Qualified bounded concurrent restore, not four simultaneously resident requests. |
| Heavier pressure, 98.6% KV | Fourth request remained deferred; canceling pressure clients restored service. | Does not reproduce the reporter's successful admission at 97.8% occupancy. |
| Admission attempted near 80% KV while three streams were being scheduled | Final request recomputed with zero external hits after waiting; correct answer and healthy engine. | Correct response but no external-restore coverage. |

The reporter used eight sequences, a 4096-token batch, 3.6 GiB KV per rank
and 1.5 GiB vision-weight CPU offload. Those differences are not hidden by the
preset checks. Implementation is included in beta; qualification is limited
to the conditions in the table.

Image-bearing requests remain excluded from GLM's external checkpoint
connector because image identity is not yet safe for external reuse. Native
GPU prefix reuse and image generation are separate supported paths.
The [recurrent recovery report](glm53-kda-recovery-lmcache.md) records the
RAM/disk restart tests and memory-saving implementation.

## Qwen CPU offload investigation

A local port of upstream vLLM #54743 excludes non-prefix-cacheable QSA
scratch groups while preserving the scheduler/worker group-index contract.
Its CPU tests pass, but the composed GPU path is **research-only**: with
TP2/MTP3, a real 2,880-token external hit changed correct arithmetic into
repetitive, incorrect output. This source overlay is not in the published
beta and is not a qualified workaround for the community offload report.

Native-prefix or zero-hit responses do not validate CPU restore. The
investigation separately checks no-speculation restore and host-memory
registration; neither successful cold responses nor larger CPU allocation
alone close the correctness issue.
