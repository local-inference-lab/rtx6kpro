# Kimi-K3 split projection overlap and recurrent prefix checkpoints

The split-projection stream implementation overlaps independent MXFP8 Q/K/V
and BF16 gate/factor/beta work without replacing projection kernels. It improves
paired QSRT-K2 TP9 coding decode by **1.15%**, with identical generated text and
DSpark acceptance. Its implementation is in vLLM commit
`e777269b4045256d5b4ec2e690d73c2c0aa0000c`.

The FP32 recurrent prefix-checkpoint implementation is **research-only**. Its
checkpoint exports pass component tests, but enabling scheduler reuse changes
prefill partitioning and does not preserve the full model's hidden states.
It is excluded from the projection-only source and image.

## Configuration and evidence

Evidence is stored under
`/mnt/luke/kimi-k3-runs/kk-integration-20260919/tp9-kda-stream-prefix`.
The control uses vLLM `9367f3712c2290cc9c6df1b12f4c5a5b7bc082e0` and B12X
`93c1eef568cd90eb315df687bc3f3950b91b7526`, packaged as
`voipmonitor/vllm:kimi-k3-kk-cu134-tp9-dspark-k5-20260922-r1`.

Hardware is nine RTX PRO 6000 Blackwell 96 GiB GPUs, 600 W power limits and
driver 615.71.09. Clock policy is unchanged; frequencies were not sampled
during timing. The runtime uses QSRT-K2 TP9/DCP9, Red Hat DSpark MXFP8 K5,
one concurrent request, A16 target expert activations, FP32 recurrent state,
4,096-token scheduler budget and a 950,000-token context limit. No component
GPU benchmarks run concurrently with serving measurements.

## Split-projection stream overlap

`VLLM_KIMI_KDA_PROJECTION_STREAM_TOKEN_THRESHOLD=6` enables the measured path;
the source default is zero. Only CUDA graph capture at one through six rows
uses the auxiliary stream. Eager execution, larger batches and batch-invariant
mode keep serial dispatch. The gate/factor branch runs on the main stream,
Q/K/V on the existing auxiliary stream, and an event join precedes convolution
and recurrent-state consumption. Weight formats, GEMM implementations, tensor
layouts and reduction order remain unchanged.

Upstream vLLM #54697 is already in the control's ancestry. It implements
packed, unquantized BF16 TP8 projection overlap. The split MXFP8/BF16 path
measured here is outside that selector; this change does not duplicate its
skinny GEMM kernels. Open #54151 addresses large-M sequence-parallel AG-GEMM,
not this split low-M path. The implementation reuses the existing stream helper.

**Qualified component evidence:** 30 real-checkpoint cases cover layers 1, 45
and 90, ranks 0 and 8, and row counts 1, 5, 6, 12 and 16. Changed-input graph
replay is bit-exact. Six-row latency is 18.43 versus 14.34 microseconds with
warm weights and 42.46 versus 38.38 microseconds with cache eviction. Five
additional pytest cases check graph replay and serial/batch-invariant dispatch.
Receipts: `projection-layer-rank-sweep.json`, `projection-serving-dispatch.json`
and `projection-only-tests.log`.

**Qualified paired E2E evidence:** `projection-decode-abba-r2/receipt.json`
records three 4,096-token coding runs per arm, 183 input tokens, temperature
zero and seed one. Median decode is 104.7808 versus 105.9812 tok/s (+1.1456%).
Every run accepts 2,856 of 6,200 proposals over 1,240 target cycles
(46.064516%; 3.30323 emitted tokens per cycle). All output text hashes equal
`20968e50f3e10c53f16eea582b794f8fbfa7729ba1e377e68f63703c03ac4ffd`.

Both graphs remain resident during this A/B; those absolute rates are not the
single-graph production rate. A separate single-graph build with projection
overlap measures 110.77 tok/s median, but also contains the prefix experiment.
The short coding prompt does not exercise internal prefix coalescing. Use the
paired result for the causal gain, not a comparison between separate launches.

The projection-only installed image is **qualified** at 32,768 and 524,288
input tokens: all sampled prefill tails, scheduler boundaries and 192
teacher-forced decode positions match the control bit-for-bit. Its initial
three coding runs measure 107.45 tok/s median, with the same text and acceptance
above. These post-startup rates are retained independently of the paired A/B.

Native host-KV restores 2,102,132,736 bytes for a 78,114-token prompt. The
64-token output is identical; cold completion takes 31.029 seconds and replay
1.938 seconds. Four images totaling 65,232 patches complete as a 16,448-token
request. This is an allocation/execution check, not a vision accuracy score.
Receipts are in `projection-qualified`.

The ordinary serving worker, without diagnostic RPCs or source mounts,
measures **109.67 tok/s median** over three repeated 4,096-token coding runs.
Its first three post-startup runs measured 105.93 tok/s; both sets retain the
same output and acceptance. The r1 repeated absolute rate was 109.72 tok/s,
so the independent server launches do **not** demonstrate a material absolute
throughput gain. The controlled resident A/B is the evidence for the small
1.15% incremental improvement. No cause is assigned to the startup variation.

Cold prefill measures 2,123 / 2,584 / 2,501 tok/s at 8k / 32k / 64k, one
sample per size. The 8k request includes logged JIT activity for
`l2norm_fwd_kernel2` and `layer_norm_gated_fwd_kernel`; its rate must not be
presented as warmed kernel throughput. The 32k/64k results are essentially
unchanged from r1. No prefill speedup is claimed for the decode-only stream gate.
Three repeated cold-cache 8k requests after kernel warmup measure
**2,588 tok/s median** (2,579–2,592 tok/s); host-KV and prefix-cache reuse are
disabled for every request. Receipt: `production-prefill-8k-repeat.json`.

The public image's ordinary runtime passes a 404 check for `/collective_rpc`.
Both source reconstruction from fresh GitHub clones and runtime-bundle
reassembly reproduce the tested source tree and manifest exactly. This does
not claim a second byte-identical Docker image build. See the
[launch and reconstruction guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/kimi-k3/qsrt-tp9-dspark.md)
and [artifact release](https://github.com/local-inference-lab/rtx6kpro/releases/tag/kimi-k3-qsrt-tp9-kda-stream-20260922).

The source is included in [vLLM #843](https://github.com/local-inference-lab/vllm/pull/843)
as commit `537b4020ec3b47a75e9dbb954ceb72d1e99da2e7`. Merging that head with
the other pinned review units produces tree
`d707a4125f953444e8024afd3cf88ee4239eaf65`, identical to the tested integration.
No B12X source change is required. The image is
`voipmonitor/vllm:kimi-k3-kk-cu134-tp9-dspark-k5-20260922-r2`; enable the stream
threshold explicitly as shown above. Its local image ID is
`sha256:8f71983a5d8f8e59688d0caff1b59c86a38a604a125f95bbf8d12f0fd4423dd3`,
and runtime manifest SHA-256 is
`6e996e7914bbc2a9dd516e4967134e7ce3f1021d391c319fa9a23349e555da32`.
All 12 native vLLM payloads are reused under identical native-input/ABI checks.

## FP32 recurrent prefix checkpoints

Research source:
`perf/kimi-split-kda-overlap@ed2df537bfde5609b89e023a1c25ec7bfdc0a3c8`.
`VLLM_KIMI_KDA_TRITON_PREFIX_CHECKPOINTS=1` exports internal 64-token states
from the FP32 Triton accumulator. DSpark scheduler eligibility is also needed;
without it the exported states do not eliminate terminal forward execution.
Component tests pass: 47 checkpoint/projection cases, 7 packaging cases,
174 cache cases and 59 scheduler cases.

At 32,768 input tokens, the first 11 common prefill-tail captures are exact.
The control's final 1,408- and 384-row steps become a single 1,792-row step.
The final prefill tail and all 192 scored decode rows are finite but are not
bit-identical to the control. The exact arithmetic source of the rechunking
difference has not been localized; a passing checkpoint-export kernel test
does not qualify the changed model execution schedule.

The frozen full-vocabulary LM-head comparator reports:

| Measure over 192 teacher-forced positions | Result |
| --- | ---: |
| Mean reference-to-candidate KL | 0.0004548302 |
| p99 KL | 0.0161625 |
| Maximum KL | 0.0235598 |
| Top-1 agreement | 192 / 192 |

This is a runtime-schedule comparison of the same QSRT checkpoint, not a
quantization comparison against official MXFP4. The comparator, LM-head and
capture hashes are in `checkpoint-rechunking-drift.json`.

Nine cold/append requests at approximately 12k, 32k and 64k input tokens
complete with identical 32-token outputs. Cold TTFT improves by approximately
0.2 seconds; prefix-hit TTFT remains essentially unchanged at 0.42–0.74 seconds.
There is only one sample per case. Receipts are in `control-prefix` and
`dspark-qualified/prefix`. This evidence does not justify enabling the path
under the requirement to preserve model numerics.

The research implementation is retained for component and scheduler work.
Production qualification must keep it disabled. The earlier `native-32k`
receipt from commit `70172a93f` predates DSpark scheduler eligibility and
must not be cited as evidence for active prefix checkpoint reuse.
