# Measuring quantization distribution fidelity in vLLM

Kullback-Leibler Divergence (KLD) measures how far a candidate model's
next-token distribution moves from a designated reference distribution on
exactly aligned, teacher-forced token sequences. This page specifies the
Local Inference Lab protocol for full-vocabulary KLD measurements, including
the route-controlled procedure required to compare quantization formats on
Mixture-of-Experts (MoE) models.

KLD is a sensitive distribution-fidelity measurement. It is not, by itself, a
measure of correctness, capability, generation stability, or user preference.

## Status

| Component | Status | Evidence or limitation |
|---|---|---|
| Measurement and artifact contract on this page | implemented | The contract defines aligned inputs, full-vocabulary computation, controls, statistics, and receipts |
| Kimi K3 post-normalization hidden-state replay | qualified | The [1,024-context Kimi K3 artifact](../models/kimi-k3/distribution-fidelity-1024x2048.md) includes live-logit replay checks, runtime repeats, hashes, and candidate receipts |
| Kimi K3 natural-route candidate comparison | qualified | The published comparison measures the official MXFP4 reference against QSRT K2 under each checkpoint's natural routes |
| Generic exact route-controlled replay in vLLM | unsupported | The Local Inference Lab vLLM source at commit [`47ccf6c`](https://github.com/local-inference-lab/vllm/tree/47ccf6c57d92f03630ebcbad3809450545825488) can capture selected expert IDs, but it does not capture consumed route weights or provide an exact route-replay interface |
| GLM-5.3-Flash route-controlled KLD result | research-only | The [BF16-to-NVFP4 four-cell report](glm-5.3-flash-bf16-nvfp4.md) publishes analysis and held-out results; it has no acceptance threshold and measures `FLASHINFER_CUTLASS`, not the nondeterministic B12X NVFP4 MoE path |
| GLM-5.3-Flash QAD step-1,750 checkpoint comparison | research-only | The [QAD step 1,750 report](glm-5.3-flash-qad-step1750.md) finds lower aggregate BF16-to-candidate KLD than the published NVFP4 comparator under pre-specified fixed-route and natural-route criteria; the gain is heterogeneous and concentrated in dialogue/instruction data |
| GLM-5.3-Flash QAD step-2,500 checkpoint progression | research-only | The [QAD step 2,500 report](glm-5.3-flash-qad-step2500.md) finds lower KLD than published NVFP4 in both routing conditions; relative to QAD step 1,750, natural-route KLD improves while exact-BF16-route KLD regresses |
| GLM-5.3-Flash QAD TV-nucleus step-2,500 comparison | research-only | The [QAD TV-nucleus step 2,500 report](glm-5.3-flash-qad-tvn-step2500.md) finds lower natural-route KLD than published NVFP4, an inconclusive exact-BF16-route change, and higher KLD than QAD step 2,500 under both routing conditions |

The status of one model artifact does not qualify the same capture path,
runtime, dataset, or acceptance threshold for another model.

## Measurement contract

For reference logits \(z_t^B\), candidate logits \(z_t^Q\), and full
vocabulary \(V\), compute the forward divergence at predicted-token position
\(t\):

```text
p_t = softmax(z_t^B)
q_t = softmax(z_t^Q)
KL_t(B || Q) = sum over v in V of p_t[v] * (log p_t[v] - log q_t[v])
```

`B` identifies the baseline operand and `Q` identifies the candidate operand.
Use the label `BF16` only when the baseline checkpoint and measured compute are
actually Brain Floating Point 16-bit (BF16). If the reference is FP8, MXFP4, or
another format, reports must name that format and may use `R` instead of `B`.

The primary value is the arithmetic mean of `KL_t(B || Q)` in nats across all
scored positions. A sequence of `T` token IDs normally has `T - 1` scored
next-token positions. The implementation must:

- compare every vocabulary entry; top-k log probabilities are insufficient;
- use identical token IDs and identical position alignment for both operands;
- compute `log_softmax` or `logsumexp` in at least FP32 and accumulate summary
  sums in FP64;
- preserve the reference-to-candidate direction, because KLD is asymmetric;
- record vocabulary identity and reject a shape or token-ID mismatch;
- report negative values beyond a declared numerical tolerance as errors
  instead of silently clamping them.

Jensen–Shannon divergence and top-1 agreement are useful secondary metrics,
but neither replaces the forward KLD.

## Define the estimand before capture

A result is interpretable only when the report says which difference it is
intended to measure.

| Estimand | Required comparison | Meaning |
|---|---|---|
| End-to-end serving divergence | Natural reference versus natural candidate | Includes checkpoint, quantized tensors, kernels, router changes, and any other unfrozen runtime difference |
| Quantized transformer-body divergence | Shared canonical language-model head over reference and candidate final hidden states | Excludes candidate language-model-head error and includes all upstream runtime effects |
| Codec or expert-compute divergence on a fixed MoE path | Baseline routes and route weights replayed through candidate compute (`B×Q`) | Conditions the candidate on the baseline routing mediator |
| Route-mediated effect | All four routing and compute cells | Distinguishes route changes from compute changes and exposes their interaction |

Every non-treatment variable must be frozen. This includes the source
checkpoint lineage, tokenizer, chat template, attention and Key-Value (KV)
cache dtype, tensor and expert parallel topology, backend selection, sequence
length, batching, and numerical flags. If a variable cannot be frozen, name it
as part of the treatment rather than attributing the result only to a codec.

## Capture modes

### Full-logit capture

Capture the complete reference and candidate logits after the canonical
language-model head. This measures the head and transformer body together and
supports direct KLD computation. It is the simplest semantic contract but can
require hundreds of gibibytes per operand.

Exact streaming is allowed: a comparator may process vocabulary chunks and
retain sufficient statistics instead of storing a full tensor, provided that
it computes the same full-vocabulary `logsumexp` and weighted sum as the dense
formula. A top-k approximation is not an exact streaming implementation.

### Post-normalization hidden-state replay

Capture each operand's final transformer state after the model's final
normalization and immediately before the language-model head. Reconstruct both
logit distributions with one frozen, higher-precision language-model head:

```text
z_t^B = head(h_t^B)
z_t^Q = head(h_t^Q)
```

This contract substantially reduces capture storage and isolates body/runtime
divergence. It deliberately excludes candidate head quantization. The report
must identify the shared head, its bias or tied-embedding semantics, tensor
hash, dtype, and matrix multiplication precision.

Hidden-state replay is qualified separately for each architecture and capture
point. Compare replayed logits against live logits on a frozen subset and
publish mean, tail, maximum, and top-1 discrepancies before using replayed KLD.
The [Kimi K3 artifact](../models/kimi-k3/distribution-fidelity-1024x2048.md)
is one qualified implementation, not proof that every vLLM model has the same
capture boundary.

## Evaluation data

Three datasets have different roles and must not be conflated:

1. The **quantizer calibration corpus** estimates scales, clipping, rotations,
   or reconstruction parameters while constructing the candidate checkpoint.
2. The **KLD analysis partition** supports implementation debugging, codec
   tuning, and threshold selection.
3. The **KLD qualification partition** is read only after the candidate
   configuration and decision rule are frozen.

There is no universal quantizer-calibration corpus or universal KLD corpus.
For an MoE quantizer, sparse expert activation makes calibration coverage a
first-class concern. [MoEQuant](https://proceedings.mlr.press/v267/chen25aa.html)
uses expert-balanced sampling for this purpose; that is guidance for quantizer
calibration, not a prescription for the KLD evaluation distribution.

The KLD suite should be a broad, frozen, stratified sample of intended use. It
should include factual prose, technical material, dialogue and instructions,
code, formal reasoning, supported languages, and structured or tool-shaped
text in declared proportions. Each context must retain:

- source dataset and immutable revision;
- source document or repository cluster;
- extraction rule, deterministic offset, and representation type;
- exact token IDs, tokenizer revision, token hash, and content hash;
- allocation stratum and analysis or qualification partition.

Use source-document clusters as the independent sampling units. Contexts cut
from the same document or software repository are correlated and must remain
in the same partition.

Check contamination at more than one granularity. Exact document hashes do not
detect copied passages, benchmark questions embedded in longer documents, or
near duplicates. Publish exact-content checks plus token or character n-gram
overlap and an approximate shingle method such as MinHash. Scan overlap between
the quantizer calibration corpus, both KLD partitions, and capability
benchmarks. If the calibration corpus is unavailable, record its identity as
unknown instead of asserting independence.

The Kimi K3 suite is a qualified example with 1,024 distinct 2,048-token
contexts, ten content strata, source-cluster partitioning, 768 analysis
contexts, 256 qualification contexts, and 64 runtime-repeat sentinels. Its
allocation is a reusable starting point, not a universal population model.

## MoE route control

Hard top-k routing is discontinuous. A small upstream numerical difference can
change an expert selection, after which later hidden states and later routes
follow a different trajectory. Natural-route KLD can therefore obscure a
codec comparison: two codecs may have similar deployed-path divergence even
when their error on the same expert path differs substantially.

### Four-cell intervention

The left symbol names the source of expert IDs and route weights. The right
symbol names the compute/checkpoint precision. The notation assumes an actual
BF16 baseline; use `R×R`, `R×Q`, `Q×R`, and `Q×Q` when the reference is not
BF16.

| Cell | Expert IDs and consumed weights | Compute | Interpretation |
|---|---|---|---|
| `B×B` | Natural baseline routes | Baseline | Reference and runtime-repeat floor |
| `B×Q` | Baseline routes replayed verbatim | Candidate | Candidate compute error conditional on the baseline path |
| `Q×B` | Natural candidate routes transplanted verbatim | Baseline | Diagnostic effect of the candidate routing schedule under baseline compute |
| `Q×Q` | Natural candidate routes | Candidate | Deployed-path candidate divergence |

For codec ranking, Local Inference Lab uses `B×Q` as the headline estimand
because candidates are compared on the same route decisions. Report `Q×Q`
alongside it as deployed-path divergence and `Q×B` as a routing diagnostic.
This is an estimand choice, not a claim that route changes are irrelevant to
deployment.

Do not infer that `Q×Q` equals the sum of independent compute and routing
damage. The two factors interact, and KLD itself is nonlinear. The fourth cell
permits a factorial interaction contrast, but labels such as "percentage of
damage caused by routing" require a declared causal estimand, denominator, and
confidence interval. The four-run intervention and non-additive interaction
have direct scholarly precedent in Parvel Gu's
[causal route-mediated damage study](https://arxiv.org/abs/2608.11212).

### Exact route trace contract

An exact pinned-route run must capture and replay:

- logical expert IDs before Expert Parallel Load Balancing (EPLB) remapping;
- the exact post-selection route weights consumed by the expert kernel,
  including normalization and correction-bias semantics;
- `(context_id, predicted_token_index, layer_id, route_slot)` identity;
- top-k width, expert count, route ordering, dtype, shape, and hashes;
- the checkpoint and run from which the route trace originated.

Replaying only expert IDs while recomputing candidate weights is an
**ID-pinned** run, not exact `B×Q`. Report it separately. Reordering weights,
renormalizing them, substituting physical EPLB expert indices, or accepting a
missing token/layer entry invalidates the intervention. The runtime must fail
closed on any mismatch.

At vLLM commit [`47ccf6c`](https://github.com/local-inference-lab/vllm/tree/47ccf6c57d92f03630ebcbad3809450545825488),
[`RoutedExpertsCapturer`](https://github.com/local-inference-lab/vllm/blob/47ccf6c57d92f03630ebcbad3809450545825488/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py)
retains selected expert IDs. It does not retain the exact consumed route
weights and there is no generic route-transplant interface. Consequently,
exact `B×Q` and `Q×B` are unsupported in that revision; a paper-quality result
requires a reviewed and qualified implementation rather than an inference
from the existing expert-ID capture.

## Runtime controls

For every operand and repeat:

- load stored token IDs directly; do not retokenize source text;
- disable Multi-Token Prediction (MTP) and other speculative paths unless the
  speculative mechanism is the declared treatment;
- keep batch shape, chunked-prefill policy, sequence order, context length,
  attention backend, KV-cache dtype, parallel topology, and graph mode fixed;
- capture only the target model's canonical output, not draft-model heads;
- capture after tensor-parallel assembly and write once from global rank zero;
- record source commit, container digest, model revision, tensor hashes,
  hardware, driver, CUDA, PyTorch, vLLM, kernels, environment variables, and
  launch arguments;
- preserve raw logs that prove the selected loader, quantization, and kernels;
- repeat a stratified sentinel subset in fresh processes to estimate runtime
  and reduction-order variation.

The baseline must match the intended claim. A vendor-provided FP8 or MXFP4
checkpoint can be a useful reference, but a result against it is not
`KL(BF16 || candidate)`. If baseline and candidate are assembled from different
model lineages, the result mixes quantization error with checkpoint changes.

## Required statistics

Publish at least:

- micro mean, median, p95, p99, p99.9, and maximum per-token KLD;
- macro mean over the frozen allocation strata;
- Jensen–Shannon divergence and top-1 agreement;
- estimates by allocation stratum, semantic class, and context-depth bucket;
- the highest-KLD and most top-1-discordant contexts with source identities;
- paired candidate differences and source-cluster bootstrap 95% confidence
  intervals with the resampling seed and count;
- baseline and candidate runtime-repeat results;
- for MoE models, all available four-cell values plus route-set disagreement,
  layer-level flip rates, expert replacements, and router-margin summaries.

Bootstrap source clusters, not individual tokens. Millions of token positions
do not constitute millions of independent samples. When comparing candidates,
use paired resampling of the same source clusters and token positions.

No model-independent KLD bands such as "below 0.01 is near-lossless" are
supported. Set an acceptance rule per model and artifact before reading the
qualification partition. Base it on the runtime/replay floor, paired
uncertainty, practical task evidence, and the smallest difference the study is
designed to resolve. A difference comparable to the repeat floor is
inconclusive without additional repeats.

## Minimum artifact receipt

A published result must make every operand and transformation identifiable:

```text
suite-manifest.json          corpus, strata, partitions, token identities
source-registry.json         source revisions and cluster identities
tokens/                      exact per-context token IDs
reference-runtime.json       checkpoint, source, image, hardware, launch
candidate-runtime.json       checkpoint, source, image, hardware, launch
reference-capture/           logits or post-normalization hidden states
candidate-capture/           logits or post-normalization hidden states
routes/reference/            logical IDs and consumed weights when controlled
routes/candidate/            logical IDs and consumed weights when controlled
comparison.json              metrics, numerical settings, and input hashes
paired-comparison.json       paired source-cluster estimates when applicable
validation/                  replay, repeat, overlap, and structural checks
checksums.txt                file integrity
```

Equivalent layouts are acceptable if the same identities and checks are
present. Publish immutable repository or dataset revisions rather than mutable
branch links for measurement artifacts.

## Procedure

1. State whether the target is end-to-end fidelity, shared-head body fidelity,
   fixed-route codec comparison, or causal route decomposition.
2. Pin the reference and candidate checkpoints, tokenizer, model source,
   container, runtime configuration, and treatment boundary.
3. Freeze the quantizer calibration corpus identity and document its overlap
   with evaluation data.
4. Build and freeze source-clustered analysis and qualification partitions;
   store exact token IDs and hashes.
5. Qualify the capture boundary, comparator, and optional hidden-state replay
   against live full-vocabulary logits.
6. Capture baseline repeats and establish the process/runtime floor.
7. For an MoE codec comparison, capture exact baseline routes and weights,
   then run `B×Q`; also run natural `Q×Q`, and run the complete four-cell
   design when making claims about routing.
8. Tune only on the analysis partition. Freeze codec parameters, exclusions,
   aggregation weights, hypotheses, and acceptance rules.
9. Read the qualification partition once, produce paired cluster-bootstrap
   intervals, validate all hashes, and publish the receipts with limitations.

## Limitations

Teacher-forced KLD at one context length does not measure error accumulation
during free generation. Text-only suites do not cover vision or audio inputs.
KLD can reveal a distribution shift without establishing whether the shift is
helpful or harmful on a task. It must be accompanied by capability,
long-context, structured-output, tool-use, and free-generation evaluations
selected for the checkpoint's intended deployment.

## Scientific and technical provenance

The protocol builds on established and independently published work:

- Dutta et al.,
  [*Accuracy is Not All You Need*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e0e956681b04ac126679e8c7dd706b2e-Abstract-Conference.html),
  established KLD and answer flips as complementary measurements for compressed
  language models.
- Xin et al.,
  [*Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*](https://arxiv.org/abs/2601.20088),
  uses a full-precision teacher and a full-vocabulary forward-KL objective for
  quantized students.
- The Poolside team describes final-teacher-hidden-state caching and shared
  frozen-head logit reconstruction in the
  [*Laguna M.1/XS.2 Technical Report*](https://arxiv.org/abs/2605.27605).
- Gu's
  [four-run causal apparatus](https://arxiv.org/abs/2608.11212) directly
  precedes scholarly claims about clean-route pinning, route transplantation,
  and route-by-compute interaction.

Within Local Inference Lab, the contribution history is:

- Phaelon's full-vocabulary vLLM score-mode work, documented in
  [vLLM pull request 35961](https://github.com/vllm-project/vllm/pull/35961),
  was the direct engineering inspiration for the lab's initial KLD captures.
- Luke Alonso requested and methodologically directed the Kimi K3 fidelity
  program, including the teacher/candidate comparison and shared-head
  hidden-state workflow. He later established the lab's requirement to
  separate fixed-baseline-route `B×Q` from natural-route `Q×Q` when ranking MoE
  quantization formats.
- Martin Vit (`Festr`) constructed and validated the Kimi K3 corpus,
  implemented the capture, replay, comparison, and receipt tooling, executed
  the qualification runs, and assembled and published the artifact and its
  technical documentation. AI tools assisted corpus construction and
  documentation; dataset choices, verification, implementation, measurements,
  and publication remained human-directed and human-reviewed.

These credits describe contributions to the Local Inference Lab implementation
and artifact. They do not assign invention of KLD, hidden-state caching, or the
four-run causal design to the lab.
