# GLM-5.3-Flash NVFP4 QAD step 2,500 distribution fidelity

Status: **research-only**.

This report evaluates the Local Inference Lab
GLM-5.3-Flash-NVFP4-QAD-step2500 checkpoint against the pinned Brain
Floating Point 16-bit (BF16) reference, the published NVIDIA 4-bit
floating-point (NVFP4) comparator, and the
GLM-5.3-Flash-NVFP4-QAD-step1750 checkpoint. The candidate is a complete
serving materialization produced by the Quatrain quantization-aware
distillation (QAD) workflow.

The pre-specified primary endpoint passes. On 524,020 held-out positions under
exact BF16-route replay, forward Kullback-Leibler Divergence (KLD) decreases
from 0.067476 nats/token for the published NVFP4 comparator to 0.064066 for
QAD step 2,500. The paired change is -0.003410, a 5.054% reduction. Its
allocation-stratified source-cluster 95% bootstrap interval is -0.005093 to
-0.001953, entirely below zero.

The pre-specified secondary natural-route endpoint also passes. Held-out KLD
decreases from 0.162164 to 0.129126 nats/token. The paired change is
-0.033038, a 20.373% reduction, with a 95% interval of -0.041590 to
-0.025780.

Checkpoint progression is not uniformly favorable. Relative to QAD step
1,750, QAD step 2,500 improves held-out natural-route KLD by 10.728%, but
worsens exact-BF16-route KLD by 5.498%. Both paired intervals exclude zero.
The natural-route improvement and fixed-route regression are concentrated in
the dialogue, instruction, and assistance stratum. The result supports QAD
step 2,500 as the more faithful end-to-end checkpoint under its own routing,
but not as the more faithful expert computation conditional on BF16 routes.

The separate [verifier-backed behavioral-fidelity report](../models/glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md)
compares published NVFP4 with QAD step 2,500 under the R30
temperature-1/top-p-0.95 serving contract. Each checkpoint produced three
responses for each of 7,168 tasks. QAD increases the primary semantic point
estimate by 0.0311 percentage points; the paired 95% interval is -0.2748 to
+0.3380 points and lies inside the predeclared ±1-point
practical-equivalence band. The program-execution family changes by +0.7952
points with an interval from -1.1300 to +2.7297 points. Improved distribution
fidelity therefore accompanies preserved aggregate behavioral fidelity, not a
resolved aggregate or programming improvement. The evidence does not establish
a causal link from lower KLD to task quality.

## Checkpoint identity and contents

| Role | Artifact identity |
|---|---|
| BF16 distribution reference | zai-org/GLM-5.3-Flash-BF16@61f77a1e1a67c410650ce5017411337da0dcd11a |
| Published NVFP4 comparator | local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62 |
| QAD progression comparator | GLM-5.3-Flash-NVFP4-QAD-step1750; checkpoint-manifest SHA-256 ed76ada3ee9e4bf10d74554fb62a8d0e1767e8d5b22ee15f4940cc10cbf2da5c |
| QAD candidate | GLM-5.3-Flash-NVFP4-QAD-step2500; checkpoint-manifest SHA-256 d783ff38cacd712bd29f7f7f31129b8633928c4bb782acf2e75b336cb90a743d |

The candidate materialization manifest has SHA-256
962f7905587be9e4377b8bebbcd8e93b49c3c49f3bda6d818260da518dcd5e0a.
Its weight index has SHA-256
b43d25a280d02bfd2a58c046386e24baad78fcce355ea2d48cc0c4c78671686b,
and its model configuration has SHA-256
676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996.
All 44 indexed files were checked against the sizes and SHA-256 values in the
materialization manifest before inference. They contain 148,498 indexed
tensors and 198,042,331,512 indexed tensor bytes.

The checkpoint contract declares the following trained components:

- routed-expert matrices;
- shared-expert matrices;
- dense multilayer-perceptron matrices;
- decoder and final Root Mean Square Normalization weights;
- router matrices; and
- controller-managed routing-correction buffers.

Attention, hyper-connections, embeddings, and the language-model head are
frozen by the training contract. Routed experts are serialized as NVFP4 E2M1
values with E4M3 block-16 scales. The materialization also contains
checkpoint-specific native-route activation scales. The input-scale artifact
has SHA-256
547c4a704f8489d4b0a9a96ddef1326bb3ea0be643835798b18c4a91be179cb0.

The tokenizer JSON has SHA-256
19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d
and is identical to the QAD step-1,750 tokenizer. The candidate chat template
has SHA-256
0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5,
while the step-1,750 template has a different identity. Prompt formatting
cannot affect this comparison because every evaluation operand consumes the
same stored token IDs; no checkpoint template is applied during capture.

The source checkpoint records 293,312,834 completed training tokens at step
2,500. Its trainable objective combines full-vocabulary forward KLD from a
BF16 teacher with mean-squared error on the final hidden state entering the
frozen language-model head. The recorded best independent
eval/student_routes/loss is 0.082397918 at step 2,500. That resident probe has
a different dataset, execution topology, route intervention, and aggregation
contract, so its value is not compared numerically with the KLD values in this
report.

## Evaluation design

### Token suite

The evaluation suite has durable identifier
glm-5.3-flash-kimi-k3-source-fidelity-1024x-max2048-v1. It contains 1,024
GLM-tokenized contexts with at most 2,048 prompt tokens across ten allocation
strata. Dialogue sources use a pinned GLM chat template. Source clusters do
not cross the partition boundary.

| Partition | Contexts | Scored next-token positions | Role |
|---|---:|---:|---|
| analysis | 768 | 1,571,435 | Development-facing measurement and hypothesis formation |
| qualification | 256 | 524,020 | Held-out evaluation read after the decision rules were frozen |

The suite manifest SHA-256 is
50520bdba81a9447b769e72da58720fb8468bb6dd19f641493c9110b04f9972b.
The ordered token hash is
e2c541bce4a3213f697cd5236eaa60393d044dc054ea3ec2cc35d79dcb089f9b.
Source selection comes from the public
[festr2/kimi-k3-distribution-fidelity-1024x2048-v1](https://huggingface.co/datasets/festr2/kimi-k3-distribution-fidelity-1024x2048-v1)
artifact at revision 402919ae70d61396087571b63fe9185d95491afb. The token
artifacts are GLM-specific; Kimi-K3 hidden states or logits are not used as a
GLM reference.

### Distribution and route conditions

Every captured final hidden state is projected through the same BF16
language-model head. Its safetensors SHA-256 is
ace852f317fb56d240f539e4dbffc0883827cfc1f58b3df5ba2d389289801869.
The comparator reconstructs all 154,880 logits at every scored position and
computes full-vocabulary forward KL(BF16 || candidate) without top-k or top-p
truncation.

| Condition | Candidate execution | Estimand |
|---|---|---|
| Natural routes | Every checkpoint selects and weights its own experts | End-to-end distribution fidelity of the executed checkpoint |
| Exact BF16 routes | Every candidate replays the BF16 logical expert IDs and consumed route weights | Checkpoint fidelity conditional on one shared expert path |

The exact route trace covers the selected top-eight experts and weights for
every prompt token and every Mixture-of-Experts (MoE) layer from layer 3
through layer 44. Replay occurs after top-k selection and before Expert
Parallel Load Balancing remapping. Natural-route and exact-route KLD are
distinct estimands, not additive components of one error.

### Runtime and topology

The BF16 reference capture uses Tensor Parallelism 8 (TP8). The published
NVFP4, QAD step-1,750, and QAD step-2,500 checkpoints use TP4 on the same four
NVIDIA RTX PRO 6000 Blackwell GPUs. Pairing therefore compares candidates
against exactly the same BF16 distribution while matching candidate topology
and runtime.

Candidate inference uses eager execution, disabled Torch compilation,
sequential shared-expert execution, NVIDIA Collective Communications Library
tensor-parallel reductions, disabled custom all-reduce, and the reproducible
FLASHINFER_CUTLASS MoE backend. The instrumented image is
local/glm53-kld-r18:dev-20260903, with image ID
sha256:11a0ad1530f8232050e7bff18350860421e5815e645026905135cde3e1cfff73.

The B12X NVFP4 MoE backend is **unsupported for this measurement** because its
real layer-3 kernel is nondeterministic under the available deterministic
controls. No KLD value in this report characterizes the B12X serving path.

### Frozen decision rules

The qualification contract was written before QAD step 2,500 executed any
suite context. Its SHA-256 is
5468d33309fd8570308936fdd39631ecfbf95475db9cb31801988f03014535b7.

The primary endpoint is the paired step-2,500-minus-published-NVFP4 change in
token-weighted KLD under exact BF16-route replay. The secondary endpoint is
the same paired change under natural routes. Each directional claim passes
only when the upper endpoint of a 10,000-sample paired bootstrap interval is
below zero.

The contract also pre-specifies paired step-2,500-minus-step-1,750 progression
comparisons. A negative interval favors step 2,500, a positive interval favors
step 1,750, and an interval containing zero is inconclusive. The bootstrap
resamples (dataset, source_cluster_id) units within each allocation stratum,
preserves observed stratum position weights, and uses seed 1. No absolute KLD
pass threshold is defined.

## Paired fidelity results

Negative changes favor QAD step 2,500. The interval is the pre-specified
paired, allocation-stratified source-cluster micro interval.

| Partition | Routing | Reference | Reference KLD | Step-2,500 KLD | Change | Relative change | Paired 95% interval | Conclusion |
|---|---|---|---:|---:|---:|---:|---:|---|
| analysis | Natural | Published NVFP4 | 0.175257 | 0.137753 | -0.037504 | -21.400% | -0.043533 to -0.031881 | Development-facing evidence |
| analysis | Exact BF16 routes | Published NVFP4 | 0.070514 | 0.067025 | -0.003489 | -4.948% | -0.004589 to -0.002313 | Development-facing evidence |
| analysis | Natural | QAD step 1,750 | 0.156552 | 0.137753 | -0.018799 | -12.008% | -0.022259 to -0.015620 | Step 2,500 lower |
| analysis | Exact BF16 routes | QAD step 1,750 | 0.062700 | 0.067025 | +0.004325 | +6.898% | +0.003463 to +0.005288 | Step 2,500 higher |
| qualification | Natural | Published NVFP4 | 0.162164 | 0.129126 | -0.033038 | -20.373% | -0.041590 to -0.025780 | Secondary criterion passes |
| qualification | Exact BF16 routes | Published NVFP4 | 0.067476 | 0.064066 | -0.003410 | -5.054% | -0.005093 to -0.001953 | Primary criterion passes |
| qualification | Natural | QAD step 1,750 | 0.144644 | 0.129126 | -0.015518 | -10.728% | -0.019451 to -0.011882 | Step 2,500 lower |
| qualification | Exact BF16 routes | QAD step 1,750 | 0.060727 | 0.064066 | +0.003339 | +5.498% | +0.002313 to +0.004437 | Step 2,500 higher |

The signs and magnitudes replicate between the analysis and qualification
partitions. QAD step 2,500 therefore improves the deployed natural-route
estimand while regressing the route-conditioned progression estimand relative
to QAD step 1,750.

The descriptive natural-minus-fixed KLD gaps on qualification are 0.094689
for published NVFP4, 0.083916 for QAD step 1,750, and 0.065060 for QAD step
2,500. These gaps compare different counterfactual executions and cannot be
interpreted as an additive or causal quantity attributable only to routing.

### Held-out secondary metrics and tails

| Routing | Checkpoint | Mean KLD | Mean JS | Top-1 agreement with BF16 | KLD p99 | KLD p99.9 | Maximum KLD |
|---|---|---:|---:|---:|---:|---:|---:|
| Natural | Published NVFP4 | 0.162164 | 0.023748 | 90.663% | 3.009845 | 14.564972 | 32.544923 |
| Natural | QAD step 1,750 | 0.144644 | 0.022990 | 90.788% | 2.595301 | 12.483213 | 30.669209 |
| Natural | QAD step 2,500 | 0.129126 | 0.022964 | 90.837% | 2.284655 | 9.532325 | 26.766065 |
| Exact BF16 routes | Published NVFP4 | 0.067476 | 0.013388 | 93.034% | 1.118111 | 4.900694 | 33.261937 |
| Exact BF16 routes | QAD step 1,750 | 0.060727 | 0.012904 | 93.217% | 1.010593 | 3.650674 | 27.426561 |
| Exact BF16 routes | QAD step 2,500 | 0.064066 | 0.013954 | 93.082% | 1.073902 | 3.742166 | 24.833613 |

Natural-route secondary metrics and reported tails improve monotonically
across the three checkpoints. Under exact BF16 routes, QAD step 2,500 remains
better than the published NVFP4 comparator in mean KLD and high-KLD tails,
but it is worse than QAD step 1,750 in mean KLD, mean Jensen-Shannon
divergence, top-1 agreement, p99, and p99.9. The maximum fixed-route KLD is
lower for step 2,500. These metrics are descriptive and are not additional
pass criteria.

At the individual-position level, QAD step 2,500 has lower KLD than the
published NVFP4 comparator on 47.884% of natural-route positions and 45.104%
of exact-route positions. Relative to QAD step 1,750, the fractions are
47.973% and 45.026%. Median paired changes are approximately zero. Aggregate
gains therefore come from error magnitudes and tails, not a majority of
individual token positions.

## Allocation-stratum heterogeneity

The table reports held-out step-2,500-minus-reference KLD changes and
independent within-stratum source-cluster 95% intervals. Negative values favor
QAD step 2,500.

| Allocation stratum | Natural vs published NVFP4 | Exact routes vs published NVFP4 | Natural vs QAD step 1,750 | Exact routes vs QAD step 1,750 |
|---|---:|---:|---:|---:|
| Chinese | -0.004683 (-0.018028 to +0.007894) | -0.007840 (-0.015303 to -0.002041) | +0.000709 (-0.015660 to +0.017266) | +0.000432 (-0.002345 to +0.003572) |
| Code, tests, documentation, and issues | -0.002555 (-0.006217 to +0.001325) | -0.001574 (-0.003008 to -0.000307) | -0.000280 (-0.003145 to +0.003026) | -0.000916 (-0.002005 to +0.000094) |
| Dialogue, instruction, and assistance | -0.236102 (-0.306744 to -0.180620) | -0.009006 (-0.020979 to +0.000351) | -0.116122 (-0.145833 to -0.091065) | +0.030934 (+0.023660 to +0.038506) |
| Encyclopedic and factual | -0.008942 (-0.014071 to -0.004318) | -0.003434 (-0.006403 to -0.000550) | -0.000973 (-0.004706 to +0.003150) | -0.000142 (-0.001767 to +0.001594) |
| Literary, narrative, and creative | -0.001751 (-0.003772 to +0.000300) | -0.001796 (-0.003419 to -0.000562) | -0.002292 (-0.004587 to -0.000283) | -0.001116 (-0.002838 to +0.000151) |
| News, history, economics, legal, and essays | -0.003043 (-0.004531 to -0.001703) | -0.001016 (-0.002014 to -0.000165) | -0.003377 (-0.007375 to -0.000471) | +0.000303 (-0.000078 to +0.000809) |
| Other multilingual | -0.002240 (-0.005382 to +0.000721) | -0.000997 (-0.003647 to +0.002048) | +0.000566 (-0.003339 to +0.005042) | +0.000757 (-0.002610 to +0.004685) |
| Scientific and technical | -0.001584 (-0.004831 to +0.001649) | -0.000850 (-0.001893 to -0.000022) | -0.001527 (-0.004488 to +0.001230) | -0.000237 (-0.000795 to +0.000241) |
| Structured data, tools, APIs, and tables | -0.003754 (-0.006552 to -0.000867) | -0.002674 (-0.005581 to -0.000738) | -0.001018 (-0.003020 to +0.001275) | -0.001201 (-0.003986 to +0.000626) |
| Worked mathematics, science, and formal reasoning | -0.005763 (-0.007533 to -0.002916) | -0.003049 (-0.004618 to -0.000972) | -0.001347 (-0.002819 to -0.000042) | -0.002479 (-0.003720 to -0.001140) |

Dialogue, instruction, and assistance contains 12.5% of held-out positions but
accounts arithmetically for 89.33% of the natural-route improvement over the
published NVFP4 comparator. It accounts for 93.54% of the natural-route
improvement over QAD step 1,750.

Under exact BF16 routes, the same dialogue stratum accounts for 115.82% of the
net regression against QAD step 1,750; improvements in other strata partially
offset it. Worked mathematics, science, and formal reasoning improves
significantly even in that fixed-route progression comparison. The aggregate
checkpoint ranking must therefore not be generalized as a uniform domain
improvement.

## Natural-route behavior

Route metrics cover every scored token and all 42 routed MoE layers. Route
weight Total Variation (TV) is computed after independently normalizing each
top-eight weight vector.

| Partition | Candidate paired with BF16 | Top-1 expert agreement | Ordered slot agreement | Mean top-8 set overlap | Exact top-8 set agreement | Mean weight TV |
|---|---|---:|---:|---:|---:|---:|
| analysis | Published NVFP4 | 89.375% | 55.442% | 87.251% | 36.613% | 0.112382 |
| analysis | QAD step 1,750 | 89.470% | 55.583% | 87.348% | 36.771% | 0.111505 |
| analysis | QAD step 2,500 | 89.498% | 55.359% | 87.333% | 36.380% | 0.111565 |
| qualification | Published NVFP4 | 89.308% | 55.462% | 87.172% | 36.641% | 0.113143 |
| qualification | QAD step 1,750 | 89.409% | 55.623% | 87.270% | 36.822% | 0.112229 |
| qualification | QAD step 2,500 | 89.429% | 55.366% | 87.255% | 36.407% | 0.112298 |

QAD step 2,500 has the highest top-1 expert agreement with BF16, but lower
ordered-slot and exact-set agreement than QAD step 1,750. Set overlap and
route-weight TV are nearly unchanged. The natural-route KLD gain is therefore
not explained by simply copying more BF16 expert IDs.

The combination of a natural-route improvement and a dialogue-specific
fixed-route regression is compatible with model-route co-adaptation, but the
two execution conditions do not identify a causal decomposition. This report
does not attribute the difference to router matrices, correction biases,
expert weights, activation scales, or their interactions individually.

## Direct checkpoint-to-checkpoint distance

Forward KLD is asymmetric. The table uses the named comparator checkpoint as
the reference and QAD step 2,500 as the candidate. These distances describe
distribution change and are not quality rankings.

| Partition | Routing | Reference checkpoint | KL(reference || step 2,500) | Mean JS | Vocabulary top-1 agreement |
|---|---|---|---:|---:|---:|
| analysis | Natural | Published NVFP4 | 0.140969 | 0.023647 | 90.465% |
| analysis | Exact BF16 routes | Published NVFP4 | 0.069634 | 0.014354 | 92.748% |
| analysis | Natural | QAD step 1,750 | 0.130903 | 0.022293 | 90.645% |
| analysis | Exact BF16 routes | QAD step 1,750 | 0.062191 | 0.012906 | 92.932% |
| qualification | Natural | Published NVFP4 | 0.133598 | 0.022959 | 90.608% |
| qualification | Exact BF16 routes | Published NVFP4 | 0.066404 | 0.013835 | 92.810% |
| qualification | Natural | QAD step 1,750 | 0.123014 | 0.021573 | 90.775% |
| qualification | Exact BF16 routes | QAD step 1,750 | 0.059893 | 0.012528 | 93.051% |

## Training-evaluation separation

The retained QAD step-2,500 replay store has durable identifier
glm53-native-chat-200m-total8192-v1. Its manifest SHA-256 is
6fde914a014abf229f682eb0620b1a4a0aba203025d9dd02b2bb7174f0e8d49f.
It contains 60,763 documents and 200,002,758 stored tokens. The first
124,940,998 tokens are the replay corpus used by QAD step 1,750; the remaining
75,061,760 tokens are the extension consumed by the step-2,500 training run.

An exact-token audit scanned the complete replay store against all 1,024
evaluation contexts. It found zero candidate matches for the first 256 tokens
of an evaluation context and zero exact full-context matches. The overlap
receipt has SHA-256
5ee06cae84de79c0ea7953e40ffc2c9ea3a5d5327a0262cbecbcbf53ae2a885c.

The audit verifies exact token-sequence separation under its declared
algorithm. It does not detect semantic similarity, paraphrases, or shared
spans shorter than 256 tokens and therefore is not evidence of complete
contamination absence.

## Validation

- Independent candidate server starts produced identical canonical tensor
  payloads for one 2,048-token context. The final-hidden-state payload SHA-256
  is 673270beaf187eab17a239a89cfe488f5e366d2a7c86d4ff3c5be751e642f803,
  and the 42-layer logical-route payload SHA-256 is
  ee55ecf21b106d9b256e12721131114c7df4c00f77015f1d4f08d7761aebae28.
  The independent-start receipt SHA-256 is
  a73f93156f34b56d4a72bee77edaa21665adca4e2a1c64b1931c1a400a45126f.
- Each candidate routing condition contains exactly 768 analysis contexts and
  256 qualification contexts, covering 1,571,435 and 524,020 scored positions
  respectively.
- Every hidden-state source hash was verified during all full-vocabulary
  comparisons. Every route source hash was verified during all route
  comparisons.
- The comparator uses a full-vocabulary row-maximum pass, float32 shifted
  exponentials, float64 mass accumulation, and float64 weighted reductions.
  Its SHA-256 is
  54b1ab1c24ca5d671c00fcd27f693f8a489024a55521b34f8bfeb12d27f20b7b.
- The paired comparator SHA-256 is
  8a1056da99c216861459caf92f2216d0db1b4d225981b67bc3fcafc887863087.
  The route comparator SHA-256 is
  85ad17d50f0f0b81f68a1f97061c70bc2bf8236085097857621ffcfff0282879.
  The exact-overlap scanner SHA-256 is
  5e8eb3954ea8f030235d93c3cf8989e37d1b326e807055ea1fd965334a97f160.
- No scored comparison produced a negative KLD or Jensen-Shannon value before
  the configured numerical roundoff clamp.
- All candidate capture and comparison processes exited. Local physical GPUs
  12 through 15 are free. The 16-GPU source host has no remaining candidate
  capture or scoring process.

## Interpretation and limitations

The evidence supports a constrained conclusion: for the pinned artifacts,
token suite, shared BF16 language-model head, TP4 candidate topology, and
reproducible FLASHINFER_CUTLASS runtime, QAD step 2,500 has lower aggregate
forward KLD to BF16 than the published NVFP4 comparator under both natural
routing and exact BF16-route replay. Both pre-specified directional claims
replicate on the held-out qualification partition.

Relative to QAD step 1,750, the evidence instead supports two condition-specific
conclusions: QAD step 2,500 is closer to BF16 under its natural routing, while
QAD step 1,750 is closer to BF16 when both checkpoints are forced through the
BF16 routes. Choosing a checkpoint for deployment therefore requires natural
generation and capability evaluation in addition to distribution fidelity;
the fixed-route result should be treated as a diagnostic, not as the deployed
checkpoint ranking.

The evidence does not establish task accuracy, preference quality, safety,
serving throughput, or B12X behavior. Additional limitations are:

- Teacher-forced next-token KLD does not measure divergence after models
  sample different autoregressive continuations.
- A common BF16 language-model head intentionally removes independently
  rounded output-head effects.
- The BF16 operand uses TP8 while all candidates use TP4. This is controlled
  for paired candidate rankings but limits topology-independent
  interpretation of absolute KLD.
- The suite is a declared ten-stratum sample rather than all possible GLM
  deployment traffic, and the largest natural-route gain is concentrated in
  dialogue/instruction contexts.
- The exact-token audit does not cover semantic or short-span overlap.
- Full captures and per-position arrays are retained by Local Inference Lab
  and are not part of the documentation repository.

## Receipt identities

| Receipt | Analysis SHA-256 | Qualification SHA-256 |
|---|---|---|
| BF16 to QAD step 2,500, natural routes | 3c9a31cdc7dfb2733eeaf226b206365c5412710084fdbe2490ffe6b70a506d05 | 95d87b018c6d1cdf913a311b8df2f254fe6f95b22102edb30c14ad72bd68d88b |
| BF16 to QAD step 2,500, exact BF16 routes | 44763ed0cab211e1c28d77972fe186db19c0003582550d5a030e93b648a40bd5 | b7435b2d432538a79ceaa21d111db9e4cb4e8d4c2f44edad8f0517a9196d73d7 |
| Paired step 2,500 minus published NVFP4, natural routes | 33a45305eea66571acdc41405ceeafbf8173111063b67f78088a20888fdc40c1 | b4d9bbbadc3ba111fa6bc97f5bcd93d21f0b193821c5ee4be5cf68c9686fcc30 |
| Paired step 2,500 minus published NVFP4, exact BF16 routes | 66ae4782ac5e6e8dcea937e608f9f29bdbf4c3352c139c0a645c648171834815 | 76c9943e7733021253bfe13846b6efc6aa844819f53e9f5d13739c31483f8398 |
| Paired step 2,500 minus step 1,750, natural routes | 5e6b110f9c11273d30a060f1c1269e03af10d4123f8af3a7377ef6f1c6fef8c0 | dd3cfeaa3215347f22a5863c4a08eebf17729f57c4d15b5985d3538c767ba8aa |
| Paired step 2,500 minus step 1,750, exact BF16 routes | 3e94c3ea8b323deb2a3b9139d34ea1fe0fe38c23eee40d0c6bccf3a631f9396d | 5607c7160864a6a3224c07738336d3363eb091179760bc77354ad439f12aaf3d |
| Published NVFP4 to step 2,500, natural routes | 0eb99afe4c00e0a063a2d74cdbaad7fe4b13ba1d04e687679767c5385023bf01 | a16ccbdb549935753f0bdd74864c082540a6d41808405ea8154b3e09f88f96d7 |
| Published NVFP4 to step 2,500, exact BF16 routes | c38ab3ebed280ede0dea08d6bad77e252a56afdaa1770b0014201116ba09b446 | 43715636411f39d17f10d08c7117a367751f155294c2f6f50c6ec35acb7a5110 |
| QAD step 1,750 to step 2,500, natural routes | b3eedbb177c5ce75d68af2b0868f8857413a25ec29fb0891ffe59dcf41125e6a | 5d48e4156fe9e164afd7c02a9623f49cfbeacd33169cff6d512bfc6920485178 |
| QAD step 1,750 to step 2,500, exact BF16 routes | fdf1110ab413aab8887ac5beb3a12b7892614df92f095bed534c3eb862376cfa | c514713df0e64909a6992b93a927cfd88957fc545728d07543e1d938950bd92b |
| Natural routes, BF16 versus step 2,500 | 7069b1b9773653ae9b69e4374e62fb669d8db3021202254a4d392381bc2e997f | ffe579389a7415a3edce9b50d3da3e423e66ac3bb753b9928fd699d66dbd788a |
| Natural routes, published NVFP4 versus step 2,500 | afe82bf1a8b99dbfd0d74295d5e0ad8a3a093675cc550ab73e593beee87cfed6 | 07771fb669127ef833329a59c679098bfe480b71aa720c4a440b45924d0b7604 |
| Natural routes, step 1,750 versus step 2,500 | ff7603c196ce70601446e149481cd8f036afe23c841f9a39abd0078424d47af9 | 62492e9b5f8c9d596f492230732846cf6196968ed07ef47ec833a3c2cdfa0d3e |

## Contribution record

The Local Inference Lab
[general KLD protocol](https://github.com/local-inference-lab/rtx6kpro/blob/master/kld/README.md#scientific-and-technical-provenance)
records the scientific precedents and contribution history. Phaelon's
full-vocabulary vLLM inference-engine score-mode work provided the lab's
initial engineering inspiration. Luke Alonso directed the shared-head fidelity
program and defined the requirement to rank MoE quantization checkpoints under
fixed baseline routes as well as natural routes. Martin Vit (Festr)
constructed and validated the GLM token suite, implemented and qualified the
capture, exact route replay, distribution comparison, pairing, overlap audit,
and receipt tooling, executed the measurements, and assembled this report.

The evaluated checkpoint is credited to the Local Inference Lab Quatrain
training and materialization workflow; the retained artifact manifest does not
assign individual authorship. AI tools assisted implementation and
documentation under human direction and review. These credits do not claim
invention of KLD, quantization-aware distillation, hidden-state caching, or
route-controlled causal designs.
