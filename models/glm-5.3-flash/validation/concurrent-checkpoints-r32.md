# Concurrent immutable checkpoint publication: R32 qualification

Status: **qualified for immutable-page admission and the bounded cache tests
below**. Constrained JSON generation has a separate failed diagnostic; it is
not included in the qualification claim.

This report records the R31-to-R32 change to LMCache admission. It does not
represent a replacement of the serving kernels or a full model-performance
qualification matrix.

## Artifact and sources

Tested image ID:
`sha256:84a2c85d9f889bf0a4b0d08f33630544e1366deb3546d2b26046b64e3e1f364a`.
Published tag: `localinferencelab/vllm:jovian-judgement-community-20260909-r32`.
The [registry readback](concurrent-checkpoints-r32.registry.json) verifies that
pulling by digest returns that tested image with exactly two filesystem layers.
The [source lock](concurrent-checkpoints-r32.source.lock) has SHA-256
`16bba062c83574820d3a097414d672deef8d6079b1abe1c7f989980424654901`.

| Component | Source |
|---|---|
| vLLM | [Warmup-buffer composition](https://github.com/voipmonitor/vllm/tree/5576927057cf71b6ec61d120932338b333efa089), unchanged from R31 |
| B12X | [Qualified kernel composition](https://github.com/voipmonitor/b12x/tree/release/jovian-judgement-20260909-r29), unchanged from R31 |
| LMCache | [Concurrent publication composition](https://github.com/local-inference-lab/LMCache/tree/35ad809fdddd430c3970777a2ff3d984bf8d1963) |
| Docker recipe | [Source-locked build](https://github.com/local-inference-lab/blackwell-llm-docker/tree/9bb70f35c/recipes/glm53) |

The [artifact audit](concurrent-checkpoints-r32.audit.json) verifies two
filesystem layers, clean Git trees, consistent installed versions, unchanged
launchers/build inputs and byte-identical contents of all 14 audited native
libraries. The LMCache diff contains five Python implementation files and one
test file from Derek Yates's [LMCache #65](https://github.com/local-inference-lab/LMCache/pull/65).
No CUDA, PyTorch, FlashInfer, FlashKDA, B12X or native LMCache component changes.

## Changelog from R31

Request-boundary checkpoints can share immutable attention pages. If one
generation is still copying a shared page to RAM, another generation must
wait for that writer rather than treating the page as an invalid store.
R31 could discard the waiting generation, causing unnecessary prompt
recomputation on a later restore. This was a cache-availability failure, not
evidence of cross-tenant data leakage or mixed checkpoint bytes.

R32 returns a metadata-only busy admission and retries from the worker-owned
background transfer. The model thread and LMCache RPC handler remain free.
After the owner commits, the waiting generation reuses the page without another
copy. After the owner aborts, the waiting generation reserves and writes it.
Backoff grows from 1 to 10 ms within a bounded busy-retry interval; individual
RPC deadlines and lease reconciliation remain unchanged. Shutdown and terminal
errors fail closed. Publication still requires every rank's acknowledgement.

All R31 changes remain, including retained filesystem-to-RAM objects,
ownership-safe restore headroom in [LMCache #66](https://github.com/local-inference-lab/LMCache/pull/66),
MoE warmup-buffer reuse, GLM/Qwen/DS4 launch profiles and two-layer packaging.
Sampling, reasoning/history policy, BF16 target head/private NVFP4 MTP head,
4096-token scheduler budget and full-and-piecewise graphs are unchanged.

## CPU regression evidence

- Two independent real-storage cases fail on the published R31 package:
  owner commit and owner abort both discard a waiting generation.
- Both cases pass with R32. Additional deadline and shutdown cases return a
  failed admission without exposing copy slots or releasing another owner's lease.
- All 124 checkpoint storage, identity, durable index, admission, prefetch
  policy and RAM-eviction tests pass against the composition and again against
  the packages installed in the immutable R32 image.
- The real shared-memory/message-queue regression holds one producer's copy,
  verifies another producer remains pending, then publishes both generations.
- Changed-file SPDX, device guards, isort, Ruff, codespell and mypy checks pass.
  Rust hooks are explicitly excluded because no Rust code changes.

## Serving qualification

Conditions: one TP4/DCP1 MTP3 engine, FP8 target KV, 4096-token scheduler budget,
full-and-piecewise graphs, four RTX PRO 6000 Max-Q Workstation GPUs at 300 W,
VRAM +6000 and unchanged graphics offsets. LMCache uses a CPU-only sidecar,
4 GiB RAM and filesystem storage. The target snapshot is
`46aaae8a82032f77100f2f03e9cc11b391df3b4d`; no model downloads or source mounts
are used. GPU KV capacity is 3,780,444 tokens, matching R31.

The literal lookup contains 54,643 prompt tokens and does not supply the
expected values in its answer example. Every row produces the four exact
recorded values; all restored rows recompute zero prompt tokens.

| Request source | Complete request latency | Prompt tokens restored |
|---|---:|---:|
| Cold computation | 5.537 s | 0 |
| GPU prefix cache | 0.231 s | 54,643 |
| LMCache RAM | 0.272 s | 54,643 |
| Filesystem after RAM eviction | 0.294 s | 54,643 |
| RAM retained after filesystem restore | 0.275 s | 54,643 |
| Filesystem after worker and sidecar restart | 0.409 s | 54,643 |

These are single correctness-probe requests, including the 64-token answer,
not pure memory-copy latency or a matched performance A/B. Filesystem reads
may use the OS page cache; no cold-device bandwidth claim is made.

The shared-instruction probe restores an 11,340-token SYSTEM endpoint with
different user questions from GPU, RAM and restarted filesystem state. Only
the 11-token user suffix is recomputed. Changing a recorded value in SYSTEM
produces a complete cache miss and the changed answer.

The RAM-pressure probe writes 6,948,716,544 durable payload bytes into a
4 GiB RAM cache. Its oldest 32,768-token prompt restores exactly with zero
recompute in 0.082 s. RAM remains bounded, all read/write leases drain and
the allocator remains healthy. Latency includes waiting for stores to settle.

Raw [tier results](concurrent-checkpoints-r32.tiers.json) and
[RAM-pressure results](concurrent-checkpoints-r32.pressure.json) retain the
source counters, ownership checks and request outcomes.

### Concurrent requests and formatting limitations

Three rounds each submit four overlapping 24,275–24,276-token prompts. All
12 subsequent full-prompt replays restore through LMCache with zero recompute.
All three different-salt probes remain full cold misses. Pending generations,
copy leases and RAM locks drain; the server stays healthy.

Every one of the 27 cold/replay/isolated responses contains the correct
literal value. However, the strict bare-value formatting test **fails** for
two responses: `Record 777 has value VALUE_5450` instead of `VALUE_5450`.
The [original failed result](concurrent-checkpoints-r32.requests.json) is
preserved, not relabelled as a passing output-format test.

A separate JSON-schema arm has two complete rounds with exact answers and
eight zero-recompute replays. Its third round fails with HTTP 500 after
XGrammar rejects generated tokens. This is not a qualified constrained-output
run. The worker remains healthy. A matched R31 LMCache control reproduces the
same failure in its third cold round (one HTTP 500 among 22 cold/replay/isolation
requests). A GPU-only R31 control passes 32 cold requests. The defect therefore
predates #65, but its root cause is not established. It is tracked separately
in [vLLM #726](https://github.com/local-inference-lab/vllm/issues/726).

[R31 LMCache results](concurrent-checkpoints-r32.grammar-r31-lmcache.json),
[R31 GPU-only results](concurrent-checkpoints-r32.grammar-r31-vram.json),
[R32 partial results](concurrent-checkpoints-r32.grammar-r32-partial.json) and
[R32 failure record](concurrent-checkpoints-r32.grammar-r32-failure.json) preserve
the observations. The [reproducer](checkpoint-json-grammar/probe_mtp_json_grammar.py)
records HTTP bodies as well as answer correctness. This cache release does
not qualify strict JSON-schema generation with MTP and LMCache.

## Performance and compatibility scope

No R32 kernel-speedup claim is made. The vLLM/B12X source and native libraries
are identical to R31. [R31 measurements](warmup-retention-r31.md#matched-performance)
remain historical measurements, not relabelled R32 results: TP4/DCP1 MTP3 on
RTX PRO 6000 Max-Q 300 W GPUs at VRAM +6000 measured 11,068 input tok/s for
32K prefill, and C1 output 248.51/259.26 tok/s with 103.571/103.389 verifier
steps/s in the first/repeated cells. Those cells disabled LMCache.

Use matching worker and sidecar images because busy admission is a protocol
response understood by both. Cache salts and immutable namespace checks remain
unchanged. This release does not promise interoperability between arbitrary
source versions or multiple independent manifest indices sharing a disk path.
Use a dedicated external-cache namespace when upgrading.

Multi-replica TP4/DCP4 qualification behind HAProxy is independently reported by
[Derek Yates](https://github.com/local-inference-lab/LMCache/pull/65#issuecomment-5609669700).
It is separate evidence, not a topology reproduced by the single-quartet tests
reported here. No additional TP8, NVFP4-KV, Qwen or DS4 GPU matrix is claimed.

## Usage

There is no additional retry switch. Enable external cache through
`CACHE_MODE=lmcache`; the paired worker and CPU-only sidecar use the correction
automatically. Without LMCache, serving behavior is unchanged.
See the [model runbook](../../glm-5.3-flash.md) for complete commands.
