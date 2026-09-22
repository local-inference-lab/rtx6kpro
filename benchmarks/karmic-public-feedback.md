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

[vLLM #821](https://github.com/local-inference-lab/vllm/pull/821), including
the alias correction consolidated from closed #824, preserves
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
GPU prefix reuse and ordinary vision inference are separate supported paths.
The [recurrent recovery report](glm53-kda-recovery-lmcache.md) records the
RAM/disk restart tests and memory-saving implementation.

## Qwen CPU offload investigation

The maintained `SimpleCPUOffloadConnector` passes genuine external-restore
checks with TP2/DCP1/MTP3. [vLLM #834](https://github.com/local-inference-lab/vllm/pull/834)
propagates target CuMem permission to the draft configuration; it preserves
connector safety checks rather than disabling the expandable-segments guard.
The allocator fix is attributed to llitz's Qwen serving bundle.
[Launcher #63](https://github.com/local-inference-lab/blackwell-llm-docker/pull/63)
exposes this path through `CACHE_MODE=native`, without an LMCache service.

Qualified conditions: Qwen NVFP4 QAD snapshot
`7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`, two stock Max-Q GPUs,
TP2/DCP1/MTP3, context8192, four sequences, 1 GiB GPU KV per rank and 4 GiB
total CPU cache. Temperature 1 and top-p .95. The serving image is the pinned
beta above plus the indicated source changes; registry validation is separate.

| Check | Result |
|---|---|
| Unmodified beta, SimpleCPU connector, no CuMem, explicit native-cache reset | Two real 2880-token external restores, zero native hits; arithmetic and prefix-code recall pass. |
| Allocator fix, CuMem plus expandable segments, explicit native-cache reset | Two real 2880-token external restores, zero native hits; arithmetic and prefix-code recall pass. |
| Unified launcher with `CACHE_MODE=native`, CuMem enabled, ten intervening prompts | A 6551-token request restores 2880 tokens externally with zero native hits; recalls TEAL-428 and answers 13. Full/piecewise capture and engine health pass. |
| Published `karmic-kraken-beta-20260922-97197f5085f4798b`, no source mounts or edits, ten intervening prompts | A 6883-token request restores 2880 tokens externally with zero native hits, recalls TEAL-428 and answers 13; healthy engine. |
| Published `karmic-kraken-beta-20260922-04a3c00a18b9d45f`, no source changes, ten intervening prompts with distinct access codes | The 6883-token original request restores 2880 tokens externally with zero native hits; it recovers TEAL-428 rather than an intervening code and answers 13. All twelve requests pass; engine remains healthy. |
| Published `karmic-kraken-beta-20260922-4d9905b931656635`, including reviewed QSA guards | All twelve distinct-code requests pass; the original prompt restores 2880 external tokens with zero GPU hits and returns TEAL-428 and 13. |
| Published `karmic-kraken-beta-20260922-3221ccacf71002ea`, canonical draft/PLE synchronization and tuning-result compatibility | All twelve distinct-code requests pass; 2880 external tokens, zero GPU hits, original TEAL-428 and sum 13, healthy engine. The registry digest resolves to the exact tested image config. |

The registry confirmation uses digest
`sha256:9d2431fa46c10b429fd98eeeb8aea013ba12e2010303402203c569ea09ab7e51`,
vLLM `96b79aa073c2059462b613876366cf693d1c0c27`, B12X
`6b80ac55b93bb3684fc35ff3f2dc34424e9f9eab` and runtime
`f67ab3f21c5a21472bd8aebb2eeb76f5b5c73eb6`.
[Raw registry restore responses and cache counters](data/karmic-merge-audit-20260922/qwen-registry-native-restore.jsonl).

The distinct-code confirmations retain
[responses from `04a3c00a18b9d45f`](data/karmic-merge-audit-20260922/qwen-final-registry-native-restore.jsonl)
and [responses from `4d9905b931656635`](data/karmic-merge-audit-20260922/qwen-qsa-guards-registry-native-restore.jsonl)
separately. Their [image receipts and conditions](data/karmic-merge-audit-20260922/README.md)
identify the exact runtime for each run.

Five intervening prompts were insufficient to evict the prefix in a preceding
control: that response was a GPU hit and is not counted as CPU-restore evidence.
The native CPU cache is volatile across restart and is not the persistent
LMCache implementation. DCP2/4 native Qwen offload is unsupported by the launcher.
[Raw restore responses and counter deltas](data/qwen38-tp2-20260922/README.md)
distinguish external hits from native GPU hits.

A local port of upstream vLLM #54743 excludes non-prefix-cacheable QSA
scratch groups while preserving the scheduler/worker group-index contract.
Its CPU tests pass, but the composed GPU path is **research-only**: with
TP2/MTP3, a real 2,880-token external hit changed correct arithmetic into
repetitive, incorrect output. This source overlay is not in the published
beta and is not a qualified workaround for the community offload report.

Native-prefix or zero-hit responses do not validate CPU restore. The
generic path remains unsupported by the Qwen launcher; neither successful cold
responses nor larger CPU allocation alone close its correctness issue. The
working SimpleCPU path does not validate the generic connector.
