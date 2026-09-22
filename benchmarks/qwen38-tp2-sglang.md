# Qwen TP2 decode: community SGLang and LIL vLLM

Five warmed runs of each deployed stack use the same QAD weights and two
RTX PRO 6000 Blackwell Max-Q GPUs at stock clocks. Removing unused attention
metadata work improves vLLM output by 7.52% at concurrency 1 and 4.72% at
concurrency 8. The resulting vLLM arm exceeds SGLang at concurrency 1 and
approximately matches it at concurrency 8. The comparison measures actual
output tokens, not verifier rates alone.

Implementation: [vLLM #835](https://github.com/local-inference-lab/vllm/pull/835).
[Raw measurements, request checks, and rank-0 profiles](data/qwen38-tp2-20260922/README.md)
include both engines and the vLLM before/after pair.

## Conditions

- Checkpoint: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, snapshot
  `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd` (QAD weights).
- TP2, DCP1, three draft tokens plus the verifier bonus token; context limit
  524288. vLLM uses MTP3; SGLang uses NEXTN3, top-k 1, four draft-tree tokens.
- Same physical GPU pair, sequential engine runs, no other test workload
  during scored windows. Memory clock under load: 13365 MHz; no VRAM overclock.
- Temperature 1, top-p .95, top-k disabled, reasoning effort `medium`.
  Both servers tokenize the context-0 chat prompt to 77 input tokens.
- llm-inference-bench 0.6.2, commit `bdc96c125b522ec65ef29f01570f443fffae1cdc`:
  30-second windows, 10-second decode warmup, maximum 32768 output tokens,
  concurrency 1 and 8, context 0, loop detection enabled. An HTTP wrapper
  sends the explicit sampling settings above to both servers. Five runs per arm.
- The benchmark's local hardware telemetry describes the client, not the
  remote server. Server clocks were checked separately; do not use the JSON's
  client GPU inventory as the measured hardware.

## Source and deployment identities

vLLM image:
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260921-22cb9b8298caa06e`,
digest `sha256:7ee79a417df23504779130d3acddb0854395a307d1a217c590e18d4f51269910`.
It contains vLLM `64b752317aecc2199d030a730f7a3db0bd147684` and B12X
`6b80ac55b93bb3684fc35ff3f2dc34424e9f9eab` on CUDA 13.4.1. The Qwen profile
uses B12X MoE, dense projections, GDN and PCIe all-reduce, FP32 recurrent
state, a private NVFP4 draft vocabulary head, full/piecewise graphs,
6019-token batch budget, 16 maximum sequences and 8 GiB GPU KV per rank.

The SGLang recipe is `kanadaj/sglang-qwen38fn-sm120-turbo`, commit
`524f248296d7b5266373fa6bbede5a4350c13ecd`, `Dockerfile.hicache-wip`, built
without source edits. Foundation digest:
`sha256:f2859d1ccf824a5295088cf578eba89b0f3eeefff6ae7679c3f5d64af0689458`.
Resulting image ID:
`sha256:152d2a70b426634b74c9fb0003d69aa7037837adb5da35eb2327d61649a34479`.
Its recipe uses CUDA 13.0.3, FlashInfer QSA/GDN, BF16 recurrent state,
NCCL, a private NVFP4 draft head, HiCache 45 GB per rank, a 6144-token
prefill chunk and 64 maximum sequences.

These are deployed-stack comparisons, not an isolated vLLM-versus-SGLang
kernel comparison: runtime versions, recurrent state precision and cache
implementations differ. Neither arm changes target checkpoint weights.

## Repeated decode measurements

Qualified under the conditions above; medians of five runs. C8 is aggregate
output. The accepted-length column is output per engine step over the measured
window, not SGLang's final instantaneous acceptance gauge.

| Arm | C1 output tok/s | C1 steps/s | C1 accepted length | C8 output tok/s | C8 steps/s | C8 accepted length |
|---|---:|---:|---:|---:|---:|---:|
| Community SGLang | 200.83 | 84.33 | 2.377 | 891.67 | 383.67 | 2.323 |
| Published vLLM, B12X all-reduce | 211.00 | 88.25 | 2.381 | 866.35 | 371.70 | 2.331 |
| Same vLLM, NCCL control | 205.79 | 88.41 | 2.328 | 853.02 | 367.30 | 2.322 |
| Same vLLM, uniform-graph metadata elision | 226.87 | 95.85 | 2.369 | 907.28 | 387.74 | 2.340 |

Published vLLM versus SGLang: **+5.07% C1, −2.84% C8**. No scored cell
reported a loop or request failure. NCCL does not close the C8 gap; its C1
step rate is essentially unchanged and its lower output includes acceptance
variation. This does not establish a universal B12X/NCCL ranking.

The metadata-only implementation gains **7.52% C1 and 4.72% C8** against the
published vLLM arm. Against the community SGLang arm it measures **+12.97%
C1 and +1.75% C8**: C8 is approximately matched, not a large advantage.
This arm uses the published image plus commit `b2dffa26658` rebased onto
the image's vLLM source as `ab89b608b74`. It retains the same B12X all-reduce,
weights, quantization, state precision and launch parameters.

The matched nominal-32k prefill control uses a 30-second uncached window
after warmup, 12 samples per arm. Published beta: **14,665 tok/s**; metadata
elision: **14,822 tok/s**, +1.07%, with no observed regression. These are
median prompt-tokens/client-TTFT results, not isolated GPU prefill time.
Actual median input lengths are 32,120 and 32,119 tokens because the unique
cache-busting prefix includes the run identifier. The batch budget remains
6019 in both arms.

## Profile finding

An eight-step rank-0 C8 trace contains 840 foreach-copy calls and 280 GDN
cache-group refreshes. The Qwen attention metadata wrapper accounts for
50.85 ms of traced host time, including nested calls. B12X's mixed
prefill/decode worklists are copied for each recurrent group even when the
selected uniform decode graph does not read them. Nested totals must not be
added together or treated as unprofiled inference latency.

The NCCL control has the same metadata work and approximately the same
traced host duration. Thus replacing the collective does not remove this
host-side preparation. A narrowly scoped implementation carries the selected
graph contract into the builder and retains complete staging for mixed
graphs, including mixed graphs captured with uniform-looking dummy rows.
The explicit contract removes this work without changing live speculative
state inputs. In a matched C8 capture with eight target metadata preparations:

| Operation | Published beta | Metadata elision |
|---|---:|---:|
| Foreach-copy calls | 840 | 0 |
| B12X mixed-group refresh calls | 288 | 0 |
| Host `cudaLaunchKernel` calls | 3136 | 464 |
| Qwen metadata host time, inclusive | 50.85 ms | 23.21 ms |

The profiler adds overhead; these host durations are diagnostic, not the
unprofiled speedup. The five-run table above is the end-to-end result.

The SGLang C8 trace is also included for independent host/GPU comparison.
Differences in state precision and dependency versions prevent attributing
every cross-engine kernel-time difference to the framework itself.

Correctness qualification: 59 CPU/GPU attention-metadata tests pass over the
beta composition, including live accepted-state updates, graph storage,
padded/mixed batches and existing GLM classification regressions. Sixteen
TP2/MTP3 requests at C4, including inputs up to 31,242 tokens and eight JSON
schema requests, correctly preserve a prefix access code and compute a sum.
All five scored C1/C8 repeats finish without loop detection or request errors.
