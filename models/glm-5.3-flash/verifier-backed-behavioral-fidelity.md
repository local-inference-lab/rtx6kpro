# GLM-5.3-Flash verifier-backed behavioral fidelity

Status: **implemented** as a benchmark method. The three-checkpoint R30
nucleus-sampling comparison documented here is **qualified**.

Verifier-Backed Behavioral Fidelity (VBF) is a Local Inference Lab regression
benchmark for measuring whether checkpoint changes alter objectively
verifiable answers. It complements next-token distribution metrics such as
Kullback-Leibler Divergence (KLD): KLD measures probability-distribution
movement, while VBF measures whether generated answers cross a specified
correctness boundary.

VBF is a project-specific benchmark with an explicit operation contract. It is
not an external leaderboard or a universal model-quality score.

## Qualified GLM-5.3-Flash result

The qualified evaluation compares three NVIDIA 4-bit floating-point
(NVFP4)-family checkpoints on 7,168 procedurally generated tasks. Each
checkpoint produced three fixed-seed
sampling repeats per task, for 21,504 responses per checkpoint and 64,512
responses overall. Generation used temperature 1.0, nucleus sampling with
top-p 0.95, maximum reasoning effort, and a 32,768-token completion limit.

| Checkpoint | Semantic score | Exact responses | Protocol-valid responses | Length-limited responses |
|---|---:|---:|---:|---:|
| Published NVFP4 | 94.5386% | 19,285/21,504 (89.6810%) | 21,383/21,504 (99.4373%) | 73 |
| QAD step 2,500 | 94.5698% | 19,112/21,504 (88.8765%) | 21,439/21,504 (99.6977%) | 46 |
| QAD TV-nucleus step 2,500 | 94.5989% | 19,338/21,504 (89.9275%) | 21,436/21,504 (99.6838%) | 52 |

The semantic score is the mean fraction of correct required fields. Repeats
belonging to the same task remain in one statistical cluster.

| Candidate minus reference | Score change | Paired 95% interval | Predeclared decision |
|---|---:|---:|---|
| QAD step 2,500 minus published NVFP4 | +0.0311 points | -0.2748 to +0.3380 points | practically equivalent |
| QAD TV-nucleus step 2,500 minus published NVFP4 | +0.0603 points | -0.2581 to +0.3785 points | practically equivalent |
| QAD TV-nucleus step 2,500 minus QAD step 2,500 | +0.0292 points | -0.2739 to +0.3325 points | practically equivalent |

Every interval lies completely inside the predeclared ±1-percentage-point
equivalence band and crosses zero. The result establishes practical
equivalence on the primary VBF endpoint under this serving and decoding
contract. It does not establish that one checkpoint is better.

The checkpoint-specific reports contain the complete comparisons:

- [QAD step 2,500 behavioral fidelity](qad-step2500-verifier-backed-behavioral-fidelity.md)
- [QAD TV-nucleus step 2,500 behavioral fidelity](qad-tvn-step2500-verifier-backed-behavioral-fidelity.md)

## Secondary outcome diagnostics

An exact task cluster counts as correct only when all three sampled responses
are exact. Published NVFP4 has 5,540 such tasks, QAD step 2,500 has 5,377, and
QAD TV-nucleus step 2,500 has 5,587.

The paired exact-task diagnostic favors published NVFP4 over QAD step 2,500
(909 published-only versus 746 QAD-only tasks; exact McNemar
p=0.0000674). It does not distinguish TV-nucleus from published NVFP4
(787 published-only versus 834 TV-nucleus-only; p=0.253). It favors
TV-nucleus over QAD step 2,500 (738 QAD-only versus 948 TV-nucleus-only;
p=0.000000346). These are predeclared secondary diagnostics and do not replace
the semantic-score decision.

No task-family comparison excludes zero. The widest and least certain family
is program execution:

| Candidate minus reference | Program-execution change | Paired 95% interval |
|---|---:|---:|
| QAD step 2,500 minus published NVFP4 | +0.7952 points | -1.1300 to +2.7297 points |
| QAD TV-nucleus step 2,500 minus published NVFP4 | +0.2372 points | -1.7764 to +2.2554 points |
| QAD TV-nucleus step 2,500 minus QAD step 2,500 | -0.5580 points | -2.4786 to +1.3486 points |

The 7,168-task suite already contains 1,024 program-execution tasks and 3,072
program responses per checkpoint. A separate 2,048-task suite containing more
instances of the same integer-program template is not part of this
qualification set.

Length-limit rates are low under nucleus sampling: 0.3395% for published
NVFP4, 0.2139% for QAD step 2,500, and 0.2418% for TV-nucleus. Paired
finish-reason accounting shows no TV-nucleus looping penalty in this
configuration. Finish-reason groups are outcome-selected diagnostics rather
than counterfactual estimates of another token budget.

## What VBF measures

Python generators create each prompt and its executable answer key. No
language model writes prompts, computes expected answers, or judges responses.
A master seed, task family, and item number determine every generated value.

The suite contains 1,024 tasks from each family:

| Task family | Verifiable behavior |
|---|---|
| Record reconciliation | Apply ordered corrections, joins, filters, and exact aggregates. |
| Event-sourced state | Reconstruct mutable state from an ordered event stream. |
| Dependency graph | Compute reachability, path properties, ancestry, and mandatory nodes. |
| Constraint assignment | Solve a generated one-to-one ordering problem with a verified unique solution. |
| Program execution | Execute specified integer control flow; no model-produced code is executed. |
| Policy application | Apply explicit rules with priority, precedence, and boundary operators. |
| Evidence-chain retrieval | Follow corrected relationships through distractor-heavy supplied context. |

VBF measures deterministic reasoning and instruction execution when the prompt
supplies all required facts. It does not measure free-form writing, subjective
usefulness, factual knowledge outside the prompt, safety, repository work,
tool selection, or arbitrary deployment traffic.

## Scoring contract

Every prompt requests one strict JSON object with named fields and explicit
types. The scorer removes separately returned reasoning and inline think
blocks, then parses the last complete JSON object in visible answer content.
It does not repair malformed JSON, coerce types, ignore list order, or infer
missing values.

Two scores are retained:

1. Semantic score is the fraction of required top-level fields whose values
   and types exactly match the executable answer key. Every task receives
   equal aggregate weight.
2. Exact-response accuracy is one only when every required field is correct
   and the object has exactly the required keys.

For stochastic evaluation, the primary score first averages the three repeats
within each task and then averages the 7,168 task values. The paired percentile
bootstrap resamples whole task identifiers and keeps all three repeats
together. It uses 100,000 samples.

The practical decision margin is one semantic-score percentage point:

| Decision | Required paired 95% interval for candidate minus reference |
|---|---|
| worse | Entire interval is below -1 point. |
| better | Entire interval is above +1 point. |
| practically equivalent | Entire interval lies between -1 and +1 points. |
| not worse | Harm beyond -1 point is excluded, but two-sided equivalence is not established. |
| inconclusive | The interval still includes harm beyond -1 point. |
| unsupported | Complete pairing, provenance, or required execution conditions are absent. |

The one-point margin is a declared deployment tolerance, not a mathematical
constant.

## Checkpoints and serving contract

| Role | Durable identity |
|---|---|
| Published NVFP4 | local-inference-lab/GLM-5.3-Flash-NVFP4 at revision 378ca54585c46542bad1f3cb3ed0d73ae51cdb62; model-index SHA-256 0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb |
| QAD step 2,500 | Materialization-manifest SHA-256 962f7905587be9e4377b8bebbcd8e93b49c3c49f3bda6d818260da518dcd5e0a |
| QAD TV-nucleus step 2,500 | Materialization-manifest SHA-256 01a1ea703cef6e0e44cf788a774504ca430bad60beddde44779cf607743769c1 |

All checkpoints use the R30 serving image
localinferencelab/vllm at digest
sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131.
Each checkpoint is served sequentially by four Tensor Parallelism 4 (TP4),
Decode Context Parallelism 1 (DCP1) replicas on physical GPU groups 0-3, 4-7,
8-11, and 12-15. Each replica schedules at most 96
sequences, captures CUDA graphs through 96 sequences, and reserves 28 GiB of
8-bit floating-point (FP8) key/value cache per GPU. Aggregate client
concurrency is 384.

The runtime uses Brain Floating Point 16-bit (BF16) activations, the B12X
NVFP4 matrix-kernel backend, a 65,536-token model limit, a 4,096-token
scheduler budget, the FlashKDA recurrent-prefill kernel, prefix caching, and
no Multi-Token Prediction (MTP) or other speculative decoding. Exact
container, source-lock, checkpoint, GPU, and endpoint metadata are retained
with every run.

A +6000 MHz memory-clock offset was present for every QAD response and for the
last portion of the published-NVFP4 run. It was not present uniformly during
the first portion of the published run. Clock frequency is not a scoring or
decoding input, and no GPU error was observed, but the asymmetry means these
receipts must not be used for a checkpoint throughput comparison.

## Verification and reproducibility

All 64,512 declared responses completed without API errors. An independent
verifier executed all 1,024 program answer keys and recomputed every receipt
digest and score. It found no missing or duplicate keys, invalid digests,
unexpected responses, or score mismatches.

The [machine-readable public summary](validation/r30-top-p095-vbf-three-checkpoint-20260910.json)
records full-precision primary values and durable evidence identities.

| Evidence | Durable identifier |
|---|---|
| Suite file | SHA-256 38fc5741abb98cc188a5ecd80d2c7b0f328e9615725317b54d8b32c663fc0b42 |
| Canonical task records | SHA-256 7c5fc36de45e8b388c98279c1666aba22e3daa8f8273a11d2378794913c76f43 |
| Effective qualification-scope amendment | SHA-256 05982dff9f4eae75c58367942b27cff689aee9e8a942ceb18911ea07b3350284 |
| Evaluation contract | SHA-256 503680dbe53a4bd034c628b17e3091e65c7522a7100b2b1bd5c55112f93f7c41 |
| Published run / summary / verifier | fa179fa98e9f6fa6b669a2748316ede093a18dd1a410050a29014071baae8994 / 5d68a0aeaba1d2ae1f09757c43b36514a676e017dfd465b9305a11587de138cc / b9cf8274d51fb20632e9555423445895fe957a9a8f90b8802b28909527660b28 |
| QAD run / summary / verifier | 28e6a51b08c4f0bc2a35948e292c1e8029d4230d9b91797e1bee721820f5a201 / b9997e328c8172e2802011949d3282de0ca12f4ae98b8fbf3028ad0ebeafe2b7 / 343e65261acb1d9d7bf30c0db2182ae8380ee72ba68191c9ca7ae3a6577ec040 |
| TV-nucleus run / summary / verifier | 2f826f67590d10ddca6e53222bb40716aeafd494a53e2a4b825f2880447c3c12 / bc70065f436a7ce8a99484f53d17b597713b6b0272d0a3fdf8d9170d4a2fda54 / 494613616554d80c7ab1c5dbb296d46fdc5d8385c2d31b4e77590943965b4e50 |
| Published-reference comparison | SHA-256 a94be5c4220a17de848dd40db79ce31380079b1a2ea4a66af202bda318fd130d |
| TV-nucleus versus QAD comparison | SHA-256 f042ee1f1d5dbfac68bce630721c1e646cb33db9b7a9d34d1cdf7e3bcb6ce44c |
| Consolidated diagnostic | SHA-256 e3653cecd14e6d4dd8db37c77640eea2d7a18e44c748c0eb000c17f5ecee40af |
| Public machine-readable summary | SHA-256 b5479a38814a08b564c0264d37c102727d0b6313eac2e2d80bae3699a4de958c |
| Complete artifact checksum manifest | SHA-256 972f8b6edc767a57557ad3cd444cd1290e9763e330da4662b81e30c9ccd75f7b |
| Retained artifact root | /mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/r30-temperature1-top-p095-three-checkpoint-20260909 |

Qualification certifies the declared scope, provenance, pairing, execution
completeness, and verification. It does not turn VBF into a universal quality
measure.

## Relationship to KLD and decoding

On the held-out distribution-fidelity suite, natural-route forward KLD from
BF16 is 0.162164 nats/token for published NVFP4, 0.129126 for QAD step 2,500,
and 0.145654 for QAD TV-nucleus step 2,500. Under exact BF16-route replay the
values are 0.067476, 0.064066, and 0.066875. The distribution reports provide
the uncertainty intervals and route interpretation:

- [QAD step 2,500 distribution fidelity](../../kld/glm-5.3-flash-qad-step2500.md)
- [QAD TV-nucleus step 2,500 distribution fidelity](../../kld/glm-5.3-flash-qad-tvn-step2500.md)

Both trained checkpoints improve the natural-route KLD point estimate relative
to published NVFP4, while all three are practically equivalent on the
qualified nucleus-sampling VBF endpoint. QAD step 2,500 has lower KLD than the
TV-nucleus checkpoint, but no resolved semantic-score advantage. KLD is
therefore useful as a distribution-fidelity measurement, not as a standalone
capability objective.

A separate temperature-zero comparison produced semantic scores of 90.959%
for published NVFP4, 91.436% for QAD step 2,500, and 88.850% for TV-nucleus.
Under that contract TV-nucleus was worse by 2.109 points versus published
NVFP4 and 2.586 points versus QAD step 2,500. Those executions also differed
in runtime image, scheduler capacity, and comparator concurrency. Their
cross-contract comparison with the R30 nucleus-sampling result is
**research-only**. It shows that the greedy served-system regression does not
replicate under the matched R30 temperature-1/top-p-0.95 contract, but it does
not isolate top-p as the sole cause.

KLD reconstructs the full teacher-forced next-token distribution before token
selection. Changing temperature or top-p does not change the already captured
KLD value. Decoding parameters can nevertheless change autoregressive
trajectories, stopping behavior, and the practical effect of distribution
errors, which is why the deployed decoding contract must be tested separately.

## Attribution

The deterministic task generators, executable answer keys, strict scoring
contract, durable receipt format, runtime-verification records, executions,
paired analysis, and report are Local Inference Lab work. Paired bootstrap
intervals and McNemar's exact test are established statistical methods rather
than claimed mathematical inventions. AI tools assisted implementation and
documentation under human direction and review.
