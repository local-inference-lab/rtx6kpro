# Warmup buffer reuse and host-RAM retention: R31 qualification

Status: **qualified for the bounded conditions below**. This report records
the historical R30-to-R31 changes; it does not extend qualification to other
hardware, target-KV formats or concurrency levels.

## Artifact and source

Image: `localinferencelab/vllm:jovian-judgement-community-20260909-r31`.
Tested image ID:
`sha256:b6f31960019f2808418f52f9f60e445f9347d6d3bf7ce3f9d52e287f736ed087`.
The [source lock](warmup-retention-r31.source.lock) has SHA-256
`f2c30d3703e82148d7ba5d1e3da03d876b3ba535c6de36d11612b0588eadbec1`.
The [registry receipt](warmup-retention-r31.registry.json) records the manifest
digest and verifies that a pull by digest reproduces this tested image.

| Component | Complete source |
|---|---|
| vLLM | [55769270](https://github.com/voipmonitor/vllm/tree/5576927057cf71b6ec61d120932338b333efa089) |
| B12X | [3edbcbce](https://github.com/voipmonitor/b12x/tree/release/jovian-judgement-20260909-r29), unchanged from R30 |
| LMCache | [f868580a](https://github.com/local-inference-lab/LMCache/tree/f868580a475b7e6f7fd81e586526481d4dbead0e) |
| Source-locked recipe | [09002f791](https://github.com/local-inference-lab/blackwell-llm-docker/tree/09002f791/recipes/glm53) |

There are exactly two filesystem layers: a flattened runtime and one committed
source installation. Fourteen audited native libraries are byte-identical to
R30. CUDA, PyTorch, FlashInfer, FlashKDA and B12X are not replaced. LMCache's
Python wheel reuses the digest-pinned, ABI-matched compiled payload after
checking native source/build identities. R30 is a build-stage artifact source,
not the final image's parent. Installed Git trees, displayed version and source
lock agree: `0.26.1rc0+glm53.r31.vllm55769270`.

## Changelog from R30

- [vLLM #724](https://github.com/local-inference-lab/vllm/pull/724) reuses MoE
  warmup input/output, routing weights and scratch across int32/int64 route-ID
  passes. Joninco's authorship is preserved; logprobz extracted the review PR.
  The change does not replace a serving kernel or change model quantization.
- Filesystem-loaded LMCache objects remain reusable in the bounded host-RAM
  pool by default: `LMCACHE_L2_PREFETCH_POLICY=retain`. This incorporates Tim
  Rice's [launcher proposal](https://github.com/local-inference-lab/rtx6kpro/pull/100)
  into the maintained source recipe. It is unrelated to GPU hardware L2
  prefetching.
- [LMCache #66](https://github.com/local-inference-lab/LMCache/pull/66) makes
  room for retained filesystem restores in a write-through cache, using
  bounded ownership-safe LRU eviction. Active readers and writers remain
  protected. The launcher enables the existing emergency-eviction option for
  retained filesystem objects without enabling writeback.
- `LMCACHE_SERVER_EXTRA_ARGS` forwards literal whitespace-separated arguments.
  It rejects line breaks and overrides of launcher-owned identity, geometry,
  transport and listener options. It does not evaluate shell expressions or
  expand wildcards.

Temperature 1/top-p 0.95, high reasoning, preserved thinking history, the
BF16 target head/private NVFP4 MTP head, full-and-piecewise graphs, checkpoint
policies, GLM backends and separate Qwen/DS4 profiles are retained.

## Matched performance

Hardware: the same four RTX PRO 6000 **Max-Q Workstation**, 300 W per GPU,
with an existing **+6000 VRAM offset**. These are not stock 600 W Workstation
results and must not be compared directly to that hardware's 14K prefill table.

Configuration: TP4/DCP1, MTP3, FP8 target KV, GPU cache, 4096 scheduled tokens,
16 sequences, maximum length 1,048,576, GPU utilization 0.93, OMP1,
NCCL 16 channels/2 MiB, full-and-piecewise graphs, temperature 1/top-p 0.95.
Decode uses 15 seconds of warmup followed by a 30-second context-zero C1 cell.
Prefill uses one warmup and ten cold samples in a 30-second nominal 32K bucket,
with 32,314–32,315 actual prompt tokens. Input rate is prompt tokens divided
by client TTFT; it includes first-output work.

| Measurement | R30 | R31 source | R31 source repeat |
|---|---:|---:|---:|
| 32K prefill, input tok/s | 11,151 | 11,068 (−0.74%) | Not repeated |
| C1 output, tok/s | 264.61 | 248.51 (−6.09%) | 259.26 (−2.02%) |
| C1 verifier, steps/s | 103.406 | 103.571 (+0.16%) | 103.389 (−0.02%) |
| Mean emitted tokens per speculative step | 2.5595 | 2.3999 | 2.5081 |
| Logical GPU KV capacity, tokens | 3,780,444 | 3,780,444 | Unchanged |

The prefill and verifier measurements show essentially unchanged execution
speed. Short output cells differ with speculative acceptance; neither a
throughput gain nor exact stochastic acceptance parity is established. The
lower first output sample is retained, not discarded.
The PR author's approximately 0.9 GiB TP8 capacity recovery is not a measured
TP4 result: this quartet's logical KV pool did not grow.

[Machine-readable cells and methodology](warmup-retention-r31.measurements.json)
preserve the measurements and both image identities. Performance cells used
image `602361fa`, before the LMCache Python headroom fix. The final image has
identical vLLM, B12X and native code; LMCache was disabled in these cells.
The exact final image was used for every cache check below.

## RAM retention and restore correctness

Conditions: the same TP4/DCP1 MTP3 server, FP8 target KV, CPU-only engine-driven
sidecar, 4 GiB bounded L1 RAM, filesystem L2 and retained prefetch objects.
The literal lookup fixture does not reveal answer values in its output example.
All answers are exact.

| Request | API elapsed | Prompt tokens recomputed |
|---|---:|---:|
| Cold 54,643-token literal lookup | 6.334 s | 54,643 |
| GPU prefix-cache reuse | 0.270 s | 0 |
| External RAM restore | 0.278 s | 0 |
| Filesystem restore after clearing RAM | 0.293 s | 0 |
| Reuse of disk-loaded objects retained in RAM | 0.271 s | 0 |
| Filesystem restore after restarting both services | 0.410 s | 0 |

After filesystem restore, **128 objects / 1,764,753,408 bytes** remain resident
with zero temporary objects. Repeating the request preserves that RAM inventory
and performs no prompt recomputation. Shared SYSTEM instructions reuse 11,340
tokens with different user continuations, including after restart; a changed
SYSTEM prompt correctly misses. Filesystem times include the OS page cache,
not cold-device storage performance.

### Bounded RAM pressure

Six distinct 32,768-token cold prompts write **6,617,825,280 durable bytes**,
exceeding the 4 GiB RAM capacity. The oldest replay must restore all prompt
tokens after a GPU prefix-cache reset.

- Without the headroom fix, L2 lookup succeeds but RAM allocation fails and
  all 32,768 tokens are recomputed. The answer remains exact. This failed
  admission result is preserved in the evidence.
- With the fix, the same test restores all 32,768 tokens in **0.170 s**, with
  zero recompute and the identical greedy answer. RAM remains bounded, allocator
  health passes and all read/write reservations are released.

This is not a promise that arbitrary oversubscription always hits. Active
readers/writers may prevent eviction; in that case restore can safely miss.
Retention uses RAM capacity and ordinary LRU eviction, not permanent pinning.

[Cache measurements and checks](warmup-retention-r31.cache.json) preserve both
the failed admission control and the qualified image's tier/pressure results.

## CPU and compatibility gates

18 warmup/lifetime tests, 166 source-recipe tests and 60 LMCache eviction/
prefetch-policy tests pass. The standalone dev-based LMCache #66 also passes
all 60. Three minimal cases fail without that fix. Writeback durability,
deadlines, isolated LRU and read/write ownership remain covered. Ruff, shell
syntax, dependency checks, complete Git/source-lock checks and native-library
identity checks pass.

No fresh no-spec, DFlash2, DCP4, C8/C64, Sieve, TP8 or NVFP4-KV performance
matrix is claimed. Their retained source behavior and earlier qualified
conditions remain documented separately. Qwen and DS4 GPU tests are not
repeated for this release. Use a separate external-cache volume when adopting
the image; cross-release cache compatibility is not newly qualified.

## Controls

LMCache remains opt-in through `CACHE_MODE=lmcache`. Filesystem-backed runs
default to retained RAM objects. `LMCACHE_L2_PREFETCH_POLICY=default` selects
the temporary-object policy, which releases prefetched objects after readers
finish. `LMCACHE_L2_ENABLED=0` selects RAM-only storage.

For an additional supported server option, for example:

```bash
-e LMCACHE_SERVER_EXTRA_ARGS='--max-cpu-workers 4'
```

The value is split on whitespace; embedded shell quoting is not interpreted.
Use dedicated launcher variables for cache identity, geometry and ports.
See the [model runbook](../../glm-5.3-flash.md) for complete serving commands.
