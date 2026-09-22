# Kimi-K3 QSRT-K2 TP9: source composition and validation

QSRT-K2 with Red Hat DSpark K5 serves on nine RTX PRO 6000 Blackwell GPUs.
Shared-expert overlap and L2 prefetch improve paired coding decode by **7.72%**,
without changing the generated text or draft acceptance in the measured runs.
Reusable prefill projection/attention buffers reduce peak live allocation by
**1.015 GiB per rank** at 256k input tokens; their isolated throughput effect is
small. [Launch instructions](qsrt-tp9-dspark.md) are separate from this report.

This report identifies the `20260922-r1` control image. The
[split KDA projection report](kda-projection-qualification.md) covers the
additional opt-in decode overlap in `20260922-r2` and the updated #843 head.

The source changes are **implemented** and the TP9 configuration below is
**qualified**. Other tensor-parallel sizes, higher request concurrency and
other checkpoints are not covered by these measurements.

## What to merge

Merge the shared dependencies first: vLLM
[#798](https://github.com/local-inference-lab/vllm/pull/798) and B12X
[#384](https://github.com/local-inference-lab/b12x/pull/384).
Then merge the following PRs in the same integration batch. All vLLM PRs target
`dev/karmic-kraken`; all B12X PRs target `master`. No branch retargeting or
intermediate Docker rebuild is required. Build and qualify the combined tree.

| vLLM PR | Behavior supplied |
| --- | --- |
| [#841](https://github.com/local-inference-lab/vllm/pull/841) | Bounded InstantTensor staging, deferred tensor ownership and vision-first loading |
| [#842](https://github.com/local-inference-lab/vllm/pull/842) | Prepared dense MLA, byte-preserving FP8 DCP transport, overlapping transfers and bounded prefill buffers |
| [#843](https://github.com/local-inference-lab/vllm/pull/843) | Kimi vision/projection memory management, uneven projection shards, shared-expert overlap and optional L2 hints |
| [#844](https://github.com/local-inference-lab/vllm/pull/844) | QSRT-K2 expert loading into prepared B12X A16 kernels |
| [#845](https://github.com/local-inference-lab/vllm/pull/845) | Parallel-draft precision, mixed target/draft cache geometry and native host-KV compatibility |
| [#846](https://github.com/local-inference-lab/vllm/pull/846) | Reuse of identical native build payloads and recovery from incomplete autotune caches |

| B12X PR | Behavior supplied |
| --- | --- |
| [#411](https://github.com/local-inference-lab/b12x/pull/411) | Prepared PCIe collectives and DCP operations for uneven groups |
| [#412](https://github.com/local-inference-lab/b12x/pull/412) | Prepared QSRT A16 experts, source-aligned TP shards and route tables |
| [#413](https://github.com/local-inference-lab/b12x/pull/413) | Live varlen-attention row counts within fixed prepared capacities |

vLLM #798 is common ancestry of the six vLLM review units. Until #798 merges,
GitHub also displays its changes in those PRs; each PR links its feature-only
commit. Source reconstruction verifies the combined tree. B12X's FA2-compatible
normalization from #400 is already in the base and is not duplicated.

## Runtime and precision

- Hardware: nine RTX PRO 6000 Blackwell 96 GiB GPUs on the local PCIe host,
  NVIDIA driver 615.71.09, 600 W power limits. Clock configuration was not
  changed; clocks were not sampled during timing, so these are not
  fixed-frequency measurements.
- QSRT-K2 target, TP9/DCP9, one request, five draft proposals, 4,096-token
  scheduler budget, 950,000-token request limit. Target graphs capture five
  and six rows; the draft captures five rows.
- Target expert activations stay BF16/A16. Dense weights retain the checkpoint
  representation. Draft MXFP8 linears use W8A16; draft QKV and Markov weights
  remain BF16. Target KV is FP8, draft KV is BF16.
- FP8 transport copies existing cache bytes and scales; it does not quantize
  BF16 collective inputs. Bounded buffers preserve projection input rank,
  strides, context-chunk order and accumulation order.
- B12X MLA decode/prefill, Triton KDA prefill, native 32 GiB host KV offload,
  vision enabled in data-parallel encoder mode, InstantTensor loader.

The packaged image reports **3,262,264 physical token slots**, distinct from
the configured 950,000-token per-request limit. Cache allocation is automatic
at `--gpu-memory-utilization 0.970`, not a claim that arbitrary concurrent
requests or vision shapes fit.

## Decode measurements

The coding request uses 183 stored input tokens, temperature zero, seed one,
4,096 output tokens and one concurrent request. Decode timing excludes TTFT.
The paired sequence is control, optimized, optimized, control, control,
optimized. Both arms run in the same loaded model with unchanged weights.

| Configuration | Median decode | Draft acceptance |
| --- | ---: | ---: |
| Paired control: no shared-expert overlap or L2 hints | 100.69 tok/s | 46.0645% |
| Paired shared-expert overlap + L2 hints | 108.47 tok/s | 46.0645% |
| Source-installed qualification | 109.73 tok/s | 46.0645% |
| Packaged production profile, repeated measurement | 109.72 tok/s | 46.0645% |

Every full coding run emits the same text: SHA-256
`20968e50f3e10c53f16eea582b794f8fbfa7729ba1e377e68f63703c03ac4ffd`.
Each run proposes 6,200 draft tokens in 1,240 cycles and accepts 2,856; mean
emitted tokens per target cycle is 3.30323. The optimization improves cycle
throughput, not the number of accepted proposals.

The packaged profile's first three runs immediately after startup measured
104.74 tok/s median. A repetition without a restart or configuration change
measured 109.62, 109.77 and 109.72 tok/s. The initial timing difference has no
proven cause; both sets are retained. Use the resident paired result for the
causal speedup claim, not differences between separate launches.

## Profile attribution and PCIe dispatch

Four-iteration Torch captures separate graph kernels from request timing.
Rank 0's paired target graph changes from 31.184 to 28.882 ms:

| Interval classification | Control | Overlap + L2 |
| --- | ---: | ---: |
| Communication without concurrent compute | 8.418 ms | 6.013 ms |
| Compute overlapping communication | 0.027 ms | 2.898 ms |

The packaged image independently shows 2.915 ms of overlap, a 27.497 ms target
graph and a 1.005 ms draft graph. These short traces are attribution evidence,
not replacements for E2E timing. Routed experts still account for roughly
10.5 ms of the paired target trace: decode is **not solely communication-bound**.

L2 hints prefetch immutable dense/shared weights using existing CuTeDSL
kernels: bounded 24 MiB and 48 MiB windows, at most eight rows, 184 prepared
plans, no persisting-L2 reservation. Shared experts execute on their existing
auxiliary stream. Neither change alters arithmetic.

The nine-rank collective microbenchmark uses the maximum rank latency and
changed-input CUDA Graph replay. For BF16 `[1,7168]`, B12X takes 15.24 us versus
NCCL 19.78 us. For `[6,7168]`, B12X takes 34.52 us versus NCCL 21.94 us.
Therefore the **32 KiB custom-all-reduce ceiling is retained**: the 84 KiB
DSpark target reductions correctly use NCCL. B12X DCP exchange remains active
and is a separate operation. Widening the cutoff would slow this workload;
the two algorithms also use different floating-point reduction orders.

## Prefill and numerical checks

The prefill adapter reuses caller-owned projection, compact-value and
attention buffers. Its 975,192,064-byte requirement fits the existing
1,152,310,528-byte target workspace lane; it adds no second workspace pool.
Prepared capacity is fixed at 76,032 context rows. Live counts do not become
compilation keys or pad the mathematical attention domain.

| Input length | Control | Bounded buffers | Peak live-allocation reduction |
| --- | ---: | ---: | ---: |
| 65,536 | 2,502.73 tok/s | 2,520.70 tok/s | 0.506 GiB/rank |
| 262,144 | 2,078.82 tok/s | 2,080.98 tok/s | 1.015 GiB/rank |

These are ABBA measurements. The allocator's reserved-memory total need not
fall when live buffers are reused. The packaged profile also completes cold
8k/32k/64k requests at 2,329/2,585/2,504 tok/s respectively, one sample each.

Qualification includes:

- Exact BF16 pre-LM-head captures for the stored 32,768- and 524,288-token
  protocols, including prefill-tail samples and 192 teacher-forced decode
  positions. Capture receipts record the actual 2,816-row prefill steps.
  This is exact parity on the recorded inputs, not a universal quality claim.
- A resident optimized capacity check with 949,872 input + 128 output tokens.
  It completes in 739.1 s without reducing KV allocation. The packaged image
  is separately checked at 524k through its identical installed source tree;
  the 950k check is not represented as a second packaged-image run.
- Four images totaling 65,232 patches, yielding 16,448 prompt tokens; this
  checks allocation and request execution, not vision-answer accuracy.
- Native host-KV replay at 78,114 prompt tokens: 2,102,132,736 bytes restored,
  identical 64-token output, 31.128 s cold versus 1.951 s restored, including
  output generation.
- 225 selected CPU tests passed (25 skipped); GPU suites: 11 bounded-prefill,
  110 route/LUT, 16 MoE capacity, 18 InstantTensor and 160 model-memory tests
  passed. A nine-GPU collective replay test passed. Existing broader B12X
  fixture failures also reproduce on the unmodified base; this is not an
  all-repository-test-pass assertion.

## Source and artifact identities

The published image is
`voipmonitor/vllm:kimi-k3-kk-cu134-tp9-dspark-k5-20260922-r1`.
Its registry manifest digest is
`sha256:50752698ae56e2bfe033226c51df71c283213bbf98eefff60d2e5a08d0424b07`.
The remote configuration digest matches the tested local image ID:
`sha256:8550fc2f2b85fb730d2969e7d847b7c6391ed751854f9a585362b93d358c0767`.

The image installs wheels, not source overlays. The source composition is:

| Component | Base | Composed commit |
| --- | --- | --- |
| vLLM | `9e5d1793fa34db4e672664711690e8a68d90fcd3` | `9367f3712c2290cc9c6df1b12f4c5a5b7bc082e0` |
| B12X | `c2dc1cf02295b8241fc6a7728be7b6e8c23dda2f` | `93c1eef568cd90eb315df687bc3f3950b91b7526` |
| Docker recipe | — | `53c52e71e74fd81387acaafa9d0b69ebfde12cfe` |

The [source reconstruction script](tools/reproduce-qsrt-tp9-sources.sh) merges
the pinned PR heads onto the bases and asserts the exact Git trees. Both
composition orders with dependencies first have been verified. The composed
source commits are published in their respective repositories.

The reconstruction commands were run against fresh GitHub clones. Rebuilding
from the component archive reproduces the runtime manifest and all **5,689
checked vLLM, B12X and launcher source/native files** exactly. The
`reproduced-runtime-payload.json` release asset contains the per-file hashes.
The rebuilt image has a different image ID; the equality claim covers those
installed payloads and the manifest, not every filesystem byte or image timestamp.

Runtime: CUDA 13.4.1, NVIDIA PyTorch 26.08
`2.14.0a0+4fdf77b940.nv26.8.63802676`, NCCL 2.31.2, CuTeDSL 4.6.2 and
FlashInfer 0.6.18. All twelve vLLM native payloads match the reused, ABI-checked
native wheel; FlashInfer and NCCL were not recompiled for Python-only changes.
The runtime manifest SHA-256 is
`cea03a81beadf4c0e69de904aa817656a2f42d9bee55c334667c5c2de364402c`.

The [component archive and validation receipts](https://github.com/local-inference-lab/rtx6kpro/releases/tag/kimi-k3-qsrt-tp9-dspark-20260922)
contain pinned wheel bundles, checksums, benchmark tools and source manifests.
They contain no model checkpoint. Full local evidence is retained at
`/mnt/luke/kimi-k3-runs/kk-integration-20260919/tp9-dspark-bounded-prefill`.
Benchmark reproduction and image assembly commands are in the
[runbook](qsrt-tp9-dspark.md).
