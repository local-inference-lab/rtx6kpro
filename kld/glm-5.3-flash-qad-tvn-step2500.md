# GLM-5.3-Flash NVFP4 QAD TV-nucleus step 2,500 distribution fidelity

Status: **research-only**.

This report evaluates the materialized
`GLM-5.3-Flash-NVFP4-QAD-TVN-step2500` checkpoint against the pinned Brain
Floating Point 16-bit (BF16) reference, the published Local Inference Lab
NVIDIA 4-bit floating-point (NVFP4) checkpoint, and the QAD step-2,500
checkpoint trained without the TV-nucleus objective profile.

The pre-specified primary endpoint is inconclusive. On 524,020 held-out
positions under exact BF16-route replay, forward Kullback-Leibler Divergence
(KLD) changes from 0.067476 nats/token for published NVFP4 to 0.066875 for QAD
TV-nucleus step 2,500. The paired change is -0.000601, a 0.890% reduction. Its
allocation-stratified source-cluster 95% bootstrap interval is -0.002002 to
+0.000684 and crosses zero.

The pre-specified secondary natural-route endpoint passes. Held-out KLD
decreases from 0.162164 to 0.145654 nats/token. The paired change is
-0.016510, a 10.181% reduction, with a 95% interval from -0.021337 to
-0.012230.

Relative to QAD step 2,500 without the TV-nucleus objective, both endpoints
regress. Natural-route KLD increases by 0.016528, or 12.800%, with a paired
95% interval from +0.012799 to +0.020763. Exact-BF16-route KLD increases by
0.002809, or 4.385%, with an interval from +0.002065 to +0.003583. The
non-TV-nucleus QAD step-2,500 checkpoint is therefore closer to BF16 under
both declared KLD estimands.

The natural-route improvement over published NVFP4 is concentrated in the
dialogue, instruction, and assistance allocation stratum. The absence of a
fixed-route improvement supports no claim that the TV-nucleus checkpoint has
more faithful expert computation conditional on BF16 routes. Behavioral task
quality is measured separately; KLD alone is not a capability ranking.

The separately qualified
[temperature-1/top-p-0.95 behavioral comparison](../models/glm-5.3-flash/qad-tvn-step2500-verifier-backed-behavioral-fidelity.md)
finds the three served checkpoints practically equivalent on the primary
Verifier-Backed Behavioral Fidelity score. The TV-nucleus point estimate is
0.0603 percentage points above published NVFP4 and 0.0292 points above QAD
step 2,500; both paired intervals lie inside the declared ±1-point
equivalence band and cross zero.

## Checkpoint identity and contents

| Role | Artifact identity |
|---|---|
| BF16 distribution reference | `zai-org/GLM-5.3-Flash-BF16@61f77a1e1a67c410650ce5017411337da0dcd11a` |
| Published NVFP4 comparator | `local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62` |
| QAD comparator | `GLM-5.3-Flash-NVFP4-QAD-step2500`; materialization-manifest SHA-256 `962f7905587be9e4377b8bebbcd8e93b49c3c49f3bda6d818260da518dcd5e0a` |
| TV-nucleus candidate | `GLM-5.3-Flash-NVFP4-QAD-TVN-step2500`; materialization-manifest SHA-256 `01a1ea703cef6e0e44cf788a774504ca430bad60beddde44779cf607743769c1` |

The TV-nucleus candidate has Quatrain revision
`5c318b73571f53022f0729e539a0a9a6698b60c5` and Quatrain checkpoint-manifest
SHA-256
`d0e37d9be7db72cd3e9ca8c19597d9b10905d8c526af044a1fb984b99af3b874`.
Its weight index has SHA-256
`b43d25a280d02bfd2a58c046386e24baad78fcce355ea2d48cc0c4c78671686b`,
and its model configuration has SHA-256
`676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996`.
The 44 indexed files contain 148,498 tensors totaling 198,062,285,560 bytes.
Every indexed file was verified against the size and SHA-256 recorded in the
materialization manifest after transfer from
`10.66.66.14:/data/models/GLM-5.3-Flash-NVFP4-QAD-TVN-step2500`.

The content manifests are the candidate's canonical identity. No accessible
Hugging Face revision is part of the evaluation identity.

The checkpoint records 2,500 optimizer steps and 293,312,834 training tokens
under objective profile `nucleus_tv_hidden_stop_logodds_v1`. It retains the
published NVFP4 input-scale sidecar with SHA-256
`4255779f031450572af8548c610fd9abfe7df89704985d18595c21788593cd05`;
the materialization does not represent step-2,500 activation recalibration.
Training-path KLD and Total Variation values are provenance only and are not
numerically comparable with the held-out endpoints in this report.

## Evaluation design

### Token suite

The evaluation suite has durable identifier
`glm-5.3-flash-kimi-k3-source-fidelity-1024x-max2048-v1`. It contains 1,024
GLM-tokenized contexts with at most 2,048 prompt tokens across ten allocation
strata. Source clusters do not cross the partition boundary.

| Partition | Contexts | Scored next-token positions | Role |
|---|---:|---:|---|
| analysis | 768 | 1,571,435 | Development-facing measurement |
| qualification | 256 | 524,020 | Held-out decision evaluation |

The suite manifest SHA-256 is
`50520bdba81a9447b769e72da58720fb8468bb6dd19f641493c9110b04f9972b`.
The ordered token hash is
`e2c541bce4a3213f697cd5236eaa60393d044dc054ea3ec2cc35d79dcb089f9b`.
Source selection comes from the public
[festr2/kimi-k3-distribution-fidelity-1024x2048-v1](https://huggingface.co/datasets/festr2/kimi-k3-distribution-fidelity-1024x2048-v1)
artifact at revision `402919ae70d61396087571b63fe9185d95491afb`. The stored
token artifacts are GLM-specific; Kimi-K3 hidden states and logits are not
used as a GLM reference.

### Distribution and routing conditions

Every captured final hidden state is projected through one shared BF16
language-model head. Its safetensors SHA-256 is
`ace852f317fb56d240f539e4dbffc0883827cfc1f58b3df5ba2d389289801869`.
The comparator reconstructs all 154,880 logits at every scored position and
computes full-vocabulary forward `KL(BF16 || candidate)` without top-k or top-p
truncation.

| Condition | Candidate execution | Estimand |
|---|---|---|
| Natural routes | Each checkpoint selects and weights its own experts | End-to-end distribution fidelity of the executed checkpoint |
| Exact BF16 routes | The candidate consumes BF16 top-eight expert IDs and route weights | Checkpoint fidelity conditional on one shared expert path |

The exact route trace covers every prompt token and every Mixture-of-Experts
layer from layer 3 through layer 44. Replay occurs after logical top-k
selection and before Expert Parallel Load Balancing remapping. Natural-route
and exact-route KLD are distinct counterfactual estimands; their difference is
not an additive routing-error decomposition.

### Runtime and validation contract

The BF16 reference capture uses Tensor Parallelism 8 (TP8). Candidate capture
uses four independent TP4 replicas on physical GPU groups `0–3`, `4–7`,
`8–11`, and `12–15`. The route-control implementation requires one scheduled
sequence per replica so every route artifact has one unambiguous owner. Four
disjoint 256-context ranges execute concurrently.

Candidate inference uses BF16 activations, the `fp8_ds_mla` KV cache, eager
execution, disabled Torch compilation, sequential shared experts, NVIDIA
Collective Communications Library tensor-parallel reductions, disabled custom
all-reduce, disabled FlashInfer autotuning, and the reproducible
FLASHINFER_CUTLASS MoE path. The instrumented image is
`local/glm53-kld-r18:dev-20260903`, image ID
`sha256:11a0ad1530f8232050e7bff18350860421e5815e645026905135cde3e1cfff73`.

The B12X NVFP4 MoE path is **unsupported for this measurement** because its
real layer-3 kernel does not satisfy the available deterministic controls. No
KLD value in this report characterizes B12X serving.

Full-vocabulary comparison uses 16 independent single-GPU shards. Each shard
covers a disjoint ordered 128-context range. Raw per-position KLD and
Jensen-Shannon arrays are concatenated in suite order, after which every
aggregate and 10,000-sample bootstrap statistic is recomputed.

The qualification contract was recorded before candidate inference. Its
SHA-256 is
`cd399cf185cba41ddcd4f65d19298da961040b6f264cb85bdd917264462633be`.
The primary exact-route and secondary natural-route criteria require the upper
endpoint of the paired allocation-stratified source-cluster 95% interval to be
below zero. No absolute KLD threshold is defined.

## Paired fidelity results

Negative changes favor the TV-nucleus candidate. The interval is the
pre-specified allocation-stratified source-cluster micro interval.

| Partition | Routing | Baseline | Baseline KLD | TV-nucleus KLD | Change | Relative change | Paired 95% interval | Conclusion |
|---|---|---|---:|---:|---:|---:|---:|---|
| analysis | Natural | Published NVFP4 | 0.175257 | 0.157116 | -0.018141 | -10.351% | -0.021088 to -0.015190 | Development-facing evidence |
| analysis | Exact BF16 routes | Published NVFP4 | 0.070514 | 0.070703 | +0.000189 | +0.268% | -0.000663 to +0.001042 | Inconclusive |
| analysis | Natural | QAD step 2,500 | 0.137753 | 0.157116 | +0.019364 | +14.057% | +0.015969 to +0.023197 | TV-nucleus higher |
| analysis | Exact BF16 routes | QAD step 2,500 | 0.067025 | 0.070703 | +0.003678 | +5.487% | +0.002739 to +0.004522 | TV-nucleus higher |
| qualification | Natural | Published NVFP4 | 0.162164 | 0.145654 | -0.016510 | -10.181% | -0.021337 to -0.012230 | Secondary criterion passes |
| qualification | Exact BF16 routes | Published NVFP4 | 0.067476 | 0.066875 | -0.000601 | -0.890% | -0.002002 to +0.000684 | Primary criterion inconclusive |
| qualification | Natural | QAD step 2,500 | 0.129126 | 0.145654 | +0.016528 | +12.800% | +0.012799 to +0.020763 | TV-nucleus higher |
| qualification | Exact BF16 routes | QAD step 2,500 | 0.064066 | 0.066875 | +0.002809 | +4.385% | +0.002065 to +0.003583 | TV-nucleus higher |

The analysis and qualification partitions agree in sign for all four
comparisons. The exact-route comparison with published NVFP4 remains
inconclusive in both partitions. The TV-nucleus checkpoint improves the
natural-route deployment estimand over published NVFP4 but does not match the
lower KLD of QAD step 2,500 without the TV-nucleus objective.

### Held-out secondary metrics and tails

| Routing | Checkpoint | Mean KLD | Mean JS | Top-1 agreement with BF16 | KLD p99 | KLD p99.9 | Maximum KLD |
|---|---|---:|---:|---:|---:|---:|---:|
| Natural | Published NVFP4 | 0.162164 | 0.023748 | 90.663% | 3.009845 | 14.564972 | 32.544923 |
| Natural | QAD step 2,500 | 0.129126 | 0.022964 | 90.837% | 2.284655 | 9.532325 | 26.766065 |
| Natural | QAD TV-nucleus step 2,500 | 0.145654 | 0.022435 | 90.980% | 2.630759 | 12.596392 | 30.699043 |
| Exact BF16 routes | Published NVFP4 | 0.067476 | 0.013388 | 93.034% | 1.118111 | 4.900694 | 33.261937 |
| Exact BF16 routes | QAD step 2,500 | 0.064066 | 0.013954 | 93.082% | 1.073902 | 3.742166 | 24.833613 |
| Exact BF16 routes | QAD TV-nucleus step 2,500 | 0.066875 | 0.013273 | 93.157% | 1.160447 | 4.593327 | 29.562607 |

Mean Jensen-Shannon divergence and vocabulary top-1 agreement do not rank the
checkpoints in the same order as forward KLD. This is not a numerical
contradiction: forward KLD weights low-probability tail errors differently,
whereas Jensen-Shannon divergence is bounded and symmetric. None of these
secondary metrics has a pass threshold.

## Allocation-stratum heterogeneity

The table reports held-out candidate-minus-baseline KLD changes with
independent within-stratum source-cluster 95% intervals. Negative values favor
the TV-nucleus candidate.

| Allocation stratum | Natural vs published NVFP4 | Exact routes vs published NVFP4 | Natural vs QAD step 2,500 | Exact routes vs QAD step 2,500 |
|---|---:|---:|---:|---:|
| Chinese | -0.009178 (-0.020656 to -0.000499) | -0.005237 (-0.012177 to +0.000643) | -0.004496 (-0.012020 to +0.001996) | +0.002603 (-0.000851 to +0.006083) |
| Code, tests, documentation, and issues | -0.002161 (-0.005732 to +0.001169) | -0.000238 (-0.001169 to +0.000788) | +0.000394 (-0.002674 to +0.003208) | +0.001336 (+0.000362 to +0.002463) |
| Dialogue, instruction, and assistance | -0.113563 (-0.152595 to -0.081797) | +0.003215 (-0.006357 to +0.011242) | +0.122539 (+0.094001 to +0.156530) | +0.012221 (+0.008328 to +0.016550) |
| Encyclopedic and factual | -0.006109 (-0.009022 to -0.003317) | -0.001701 (-0.004672 to +0.001351) | +0.002832 (-0.001929 to +0.007438) | +0.001732 (-0.001065 to +0.004532) |
| Literary, narrative, and creative | -0.000279 (-0.001282 to +0.000823) | -0.000081 (-0.000944 to +0.000810) | +0.001472 (-0.000196 to +0.003203) | +0.001716 (+0.000651 to +0.003133) |
| News, history, economics, legal, and essays | -0.000914 (-0.004487 to +0.004267) | -0.001082 (-0.001643 to -0.000015) | +0.002129 (-0.000536 to +0.006809) | -0.000066 (-0.001005 to +0.000600) |
| Other multilingual | -0.001041 (-0.003526 to +0.000994) | -0.000103 (-0.002168 to +0.001363) | +0.001198 (-0.000963 to +0.003319) | +0.000894 (-0.002614 to +0.003788) |
| Scientific and technical | +0.001156 (-0.001320 to +0.004276) | -0.000053 (-0.000979 to +0.000771) | +0.002740 (-0.000544 to +0.006803) | +0.000798 (+0.000309 to +0.001381) |
| Structured data, tools, APIs, and tables | -0.001807 (-0.005228 to +0.001715) | -0.002115 (-0.004912 to -0.000285) | +0.001947 (-0.000383 to +0.004725) | +0.000559 (-0.000340 to +0.001360) |
| Worked mathematics, science, and formal reasoning | -0.002552 (-0.005372 to +0.001587) | -0.000397 (-0.001748 to +0.001653) | +0.003211 (+0.001550 to +0.005613) | +0.002652 (+0.001608 to +0.004106) |

Dialogue, instruction, and assistance contains 12.5% of held-out positions but
accounts arithmetically for 86.0% of the natural-route KLD reduction relative
to published NVFP4. It accounts for 92.7% of the natural-route regression
relative to QAD step 2,500. The aggregate natural-route rankings therefore do
not describe a uniform improvement across traffic classes.

## Natural-route behavior

Route metrics cover every scored token and all 42 routed layers. Route-weight
Total Variation is computed after independently normalizing each top-eight
weight vector.

| Partition | Route reference | Top-1 expert agreement | Ordered slot agreement | Mean top-8 set overlap | Exact top-8 set agreement | Mean weight TV |
|---|---|---:|---:|---:|---:|---:|
| analysis | BF16 | 89.565% | 55.476% | 87.403% | 36.521% | 0.111099 |
| analysis | Published NVFP4 | 89.448% | 55.482% | 87.316% | 36.606% | 0.111914 |
| analysis | QAD step 2,500 | 89.780% | 55.874% | 87.672% | 37.052% | 0.107887 |
| qualification | BF16 | 89.501% | 55.487% | 87.323% | 36.533% | 0.111863 |
| qualification | Published NVFP4 | 89.414% | 55.587% | 87.273% | 36.750% | 0.112312 |
| qualification | QAD step 2,500 | 89.721% | 55.904% | 87.599% | 37.081% | 0.108568 |

The TV-nucleus candidate has slightly higher top-1 expert agreement with BF16
than published NVFP4 and QAD step 2,500 when each is separately compared with
BF16. Direct candidate-to-checkpoint agreement remains about 89–90%, so the
checkpoints still choose different top experts in roughly one token-layer case
out of ten. These route statistics do not identify whether router matrices,
correction biases, expert weights, scales, or their interactions caused the
distribution changes.

## Direct checkpoint-to-checkpoint distance

Forward KLD is asymmetric. The table treats the named comparator checkpoint
as the reference distribution and TV-nucleus step 2,500 as the candidate.
Unlike the BF16-referenced endpoints, these values measure the direct
distribution change between two quantized checkpoints. They are descriptive,
have no predeclared pass threshold, and are not quality rankings.

| Partition | Routing | Reference checkpoint | KL(reference || TV-nucleus) | Mean JS | Vocabulary top-1 agreement |
|---|---|---|---:|---:|---:|
| analysis | Natural | Published NVFP4 | 0.163711 | 0.023828 | 90.480% |
| analysis | Natural | QAD step 2,500 | 0.137058 | 0.021042 | 90.973% |
| analysis | Exact BF16 routes | Published NVFP4 | 0.075618 | 0.014075 | 92.814% |
| analysis | Exact BF16 routes | QAD step 2,500 | 0.066790 | 0.012521 | 93.142% |
| qualification | Natural | Published NVFP4 | 0.152936 | 0.022958 | 90.592% |
| qualification | Natural | QAD step 2,500 | 0.129817 | 0.020492 | 91.085% |
| qualification | Exact BF16 routes | Published NVFP4 | 0.072871 | 0.013720 | 92.853% |
| qualification | Exact BF16 routes | QAD step 2,500 | 0.063962 | 0.012149 | 93.230% |

Both partitions show the same ordering: TV-nucleus is closer to QAD step
2,500 than to published NVFP4, and exact BF16-route replay reduces the direct
distance. That geometric relationship does not imply that TV-nucleus lies
between the other checkpoints in capability or along a single training axis.

## Validation

- Four independent natural-route TP4 starts produced identical final-hidden
  tensor payload SHA-256
  `9310ca09b54ddc990a89d6ad0daab0de6f8799b8e7c0cebd6e5c929124cfee25`
  and logical-route payload SHA-256
  `33ead542b572bc2da55e414dfc8d756bbffdedc1639630898b018ca3174ed881`.
- Four independent exact-route TP4 starts produced final-hidden tensor payload
  SHA-256
  `582e8bcdec2665d31f431b2c4c300e81396a306a6d144dc0d3416d1153e7b4ef`.
- Natural and exact-route merged captures each contain exactly 1,024 unique
  context indices. Their manifest SHA-256 values are
  `6c09ee5197dd5ed33ce6771192feb692764e28fb867f360d3bf471c03421d056`
  and
  `e5a7a733988c6ad658acf0ede82f144e0f5b56ca149c75d733efa5abfd77b369`.
- Every hidden-state and route source-file hash was verified. Every scoring
  shard covered its declared ordered range, and the merged receipts cover
  exactly 1,571,435 analysis positions and 524,020 qualification positions.
- No comparison produced a negative KLD or Jensen-Shannon value before the
  configured numerical roundoff clamp.
- Direct checkpoint-to-checkpoint scoring covers the same 1,571,435 analysis
  and 524,020 qualification positions for every comparator and routing
  condition. It produced no negative-roundoff event.
- The full-vocabulary comparator SHA-256 is
  `54b1ab1c24ca5d671c00fcd27f693f8a489024a55521b34f8bfeb12d27f20b7b`.
  The paired comparator SHA-256 is
  `8a1056da99c216861459caf92f2216d0db1b4d225981b67bc3fcafc887863087`.
  The route comparator SHA-256 is
  `85ad17d50f0f0b81f68a1f97061c70bc2bf8236085097857621ffcfff0282879`.

The retained published-NVFP4 receipts use shorter display labels for the same
BF16 reference manifest. Paired interoperability receipts normalize only the
reference display label; the manifest SHA-256, context records, numeric
metrics, and per-position arrays are unchanged.

## Interpretation and limitations

The evidence supports one constrained directional claim: under the candidate's
natural routing, QAD TV-nucleus step 2,500 is closer to the pinned BF16
distribution than published NVFP4 on the declared held-out suite. The
fixed-BF16-route primary endpoint does not establish an improvement over
published NVFP4, and both routing conditions favor QAD step 2,500 without the
TV-nucleus objective.

The qualified R30 behavioral result supplies a separate conclusion: all three
checkpoints are practically equivalent on the primary
temperature-1/top-p-0.95 VBF score. The KLD ranking is therefore not a
resolved sampled-task ranking. Top-p does not alter the reported KLD because
the comparator evaluates complete teacher-forced next-token distributions
before decoding.

Additional limitations are:

- Teacher-forced next-token KLD does not measure divergence after models
  generate different autoregressive continuations.
- A common BF16 language-model head intentionally removes independently
  rounded output-head effects.
- The BF16 operand uses TP8 while candidates use TP4. Candidate rankings pair
  against the same BF16 tensor artifacts, but absolute values are not
  topology-independent measurements.
- The suite is a declared ten-stratum sample rather than all possible GLM
  deployment traffic.
- KLD, Jensen-Shannon divergence, and vocabulary top-1 agreement weight
  distribution changes differently and need not produce identical rankings.
- The result does not establish task accuracy, preference quality, safety,
  serving throughput, or causal attribution to one trained component.
- Full captures and per-position arrays are retained by Local Inference Lab
  and are not included in the documentation repository.

## Receipt identities

| Receipt | Analysis SHA-256 | Qualification SHA-256 |
|---|---|---|
| BF16 to TV-nucleus step 2,500, natural routes | `fe5b6fae4216c88de2f8dee8d151b1632cefa9bd03953e4cf4c25d15bab0d28e` | `c1c0dac036404671a83def494637eaa60eb83fdd3ec58295fa68587e747078b4` |
| BF16 to TV-nucleus step 2,500, exact BF16 routes | `010a2081510fabddc5ff604fa8b1c5ceaa571a02b516bf98f6118f14e8d57b7e` | `7302f5abdfa2c80a5168118a944ed8f885a0aa8691b9a9024f04c85eeb12e6e5` |
| Paired TV-nucleus step 2,500 minus published NVFP4, natural routes | `2c721b6f662528d6b4afd57b8a91299bf9ac3929947e83948f8160a95a77b4ac` | `fa421286fbe2fc19acb44fdf9d321f67159862f1d56109b48256967198fb344f` |
| Paired TV-nucleus step 2,500 minus published NVFP4, exact BF16 routes | `a851a5809499a1b1bf6a5a06724af3d6dce6cfdc18ad14fdd64f6365f23389cd` | `4d0a0aadfab5a62250d85cabfdcc0cc9d7440f6ac50452a7b0326fd6eb64c173` |
| Paired TV-nucleus step 2,500 minus QAD step 2,500, natural routes | `df7843a1c011410e6e4aa9c494933a8d9bc0817dca4eac6e407e66c3907d674d` | `762c1ce8b0f90aaf0d30374bbe6be3edbf94767481ef649d73ae4260137cc3dc` |
| Paired TV-nucleus step 2,500 minus QAD step 2,500, exact BF16 routes | `573970f3173cf82b3c31ce9cc3095150f3235b74eced952901339bc33ac241b7` | `913c8efc37f064b47f75cf2f125c51ed9671ef580046f84ea9aed771b94caac6` |

Direct-distance JSON/NPZ receipt SHA-256 pairs are:

| Reference and routing | Analysis JSON / NPZ | Qualification JSON / NPZ |
|---|---|---|
| Published NVFP4, natural routes | f1786d87fc7d2ff855bf9dadee892cd0a79c149e152e1ac6c638a6c96bc3e5cd / 63ff6aed0d5e0f590a0ec28777e7300f28991fa2463ad0619f1f36a0fb1623a9 | 2eb5220c07e039d927e320d5424e69bcd2480c90ad1e6397fd63582f0a6128bd / 6ea1edef1884ac9fb30ae8e8814bca2b90710efd2c39df0fee93c06320f21746 |
| QAD step 2,500, natural routes | 448a4f4b79d086f12cda19310dad43e229a322fa6b9b7246e67b8736c99bad64 / 2ff36068966e3d402ad517deffa8ab2e7b1949ebedeaa451c7457c683b30f123 | 91348af43de71ecaf5543a2c0c494eb3b2639ca31bd9c23a51cad0d75800a2c2 / fe25f2062ae24a63ba67c6a7599cb51738bed9839b61823324e0059f01e2cc5b |
| Published NVFP4, exact BF16 routes | dae72c6517a3255fba21bdb9a92e52c07416949e90137f9f566e80cdcebd3e59 / fc03af88ba22a7f1a4a48f9e3d3be14dfca353ae1adede2c6d3b1307451a2710 | f0ffa665166456bdf37cce58998369071908ad232cf785c1e74e3d1030afe825 / 70f2d509806786c90566717cb565f50a4fcd974d3cb08e4ae318dd644294ea94 |
| QAD step 2,500, exact BF16 routes | 91413345d67f3dabe5a5875a8c8d91f790f2d5ff1774e4f88a6a683412ba969b / 073292ba0c4484e0193f04ce64756d8fd670d178c2ea20861a915153484c31a1 | 567f9f4c2aee4da72dbd7cf432b0e874397a4dba6cdd5ef7ec1adae51572893d / ac61a776327a0c9f1d44f375cf647ad7dc6eb139b8df7f9fa56efeb6e47eecca |

## Contribution record

The Local Inference Lab
[general KLD protocol](README.md#scientific-and-technical-provenance)
records the scientific precedents and contribution history. Phaelon's
full-vocabulary vLLM inference-engine score-mode work provided the lab's
initial engineering inspiration. Luke Alonso directed the shared-head
fidelity program and defined the requirement to rank Mixture-of-Experts
quantization checkpoints under fixed baseline routes as well as natural
routes. Martin Vit (Festr) constructed and validated the GLM token suite,
implemented and qualified the capture, exact route replay, distribution
comparison, pairing, and receipt tooling, executed the measurements, and
assembled this report.

The checkpoint is credited to the Local Inference Lab Quatrain training and
materialization workflow; its retained manifest does not assign individual
authorship. AI tools assisted implementation and documentation under human
direction and review. These credits do not claim invention of KLD,
quantization-aware distillation, hidden-state caching, or route-controlled
causal designs.
