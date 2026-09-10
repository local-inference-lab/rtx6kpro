# GLM-5.3-Flash QAD step 2,500 verifier-backed behavioral fidelity

Status: **qualified** for the declared R30 temperature-1/top-p-0.95
served-system comparisons; **research-only** as evidence about general model
quality or a weight-only checkpoint effect.

This report compares the Quantization-Aware Distillation (QAD) artifact
GLM-5.3-Flash-NVFP4-QAD-step2500 with the published Local Inference Lab
NVIDIA 4-bit floating-point (NVFP4) checkpoint and the QAD artifact trained
with a Total-Variation (TV) nucleus objective,
GLM-5.3-Flash-NVFP4-QAD-TVN-step2500. Verifier-Backed Behavioral Fidelity
(VBF) uses procedurally generated tasks, executable answer keys, strict JSON
scoring, and paired whole-task uncertainty estimates.

The qualified result uses temperature 1.0 and top-p 0.95. Temperature-zero
results are reported separately as research-only diagnostics and do not
contribute to the qualified result.

## Qualified result

Each checkpoint answered the same 7,168 tasks three times, producing 21,504
responses per checkpoint. Generation used temperature 1.0, top-p 0.95, fixed
task/repeat seeds, maximum reasoning effort, and a 32,768-token completion
limit.

| Comparator | Comparator score | QAD score | QAD change | Paired 95% interval | Decision |
|---|---:|---:|---:|---:|---|
| Published NVFP4 | 94.5386% | 94.5698% | **+0.0311 points** | **-0.2748 to +0.3380 points** | **practically equivalent** |
| QAD TV-nucleus step 2,500 | 94.5989% | 94.5698% | **-0.0292 points** | **-0.3325 to +0.2739 points** | **practically equivalent** |

Both 100,000-sample paired task-cluster bootstrap intervals lie completely
inside the predeclared ±1-point practical-equivalence band and cross zero.
QAD step 2,500 is therefore practically equivalent to both comparators on the
primary endpoint. Neither comparison establishes that QAD is better or worse.

## Outcome diagnostics

| Metric | Published NVFP4 | QAD step 2,500 | QAD TV-nucleus step 2,500 |
|---|---:|---:|---:|
| Semantic score | 94.5386% | 94.5698% | **94.5989%** |
| Exact responses | 19,285/21,504 (89.6810%) | 19,112/21,504 (88.8765%) | **19,338/21,504 (89.9275%)** |
| Field micro-accuracy | 95.9058% | 95.9176% | **95.9586%** |
| Protocol-valid responses | 21,383/21,504 (99.4373%) | **21,439/21,504 (99.6977%)** | 21,436/21,504 (99.6838%) |
| Length-limited responses | 73 | **46** | 52 |
| Completion tokens | 109,256,432 | 104,649,137 | **104,275,806** |

An exact task cluster counts as correct only when all three responses are
exact. Published NVFP4 has 5,540 such tasks, QAD step 2,500 has 5,377, and
TV-nucleus has 5,587.

Against published NVFP4, both checkpoints solve 4,631 exact clusters,
published NVFP4 alone solves 909, QAD alone solves 746, and neither solves
882. The exact McNemar p-value is 0.0000674 and favors published NVFP4 on this
stricter secondary criterion. Against TV-nucleus, both solve 4,639, QAD alone
solves 738, TV-nucleus alone solves 948, and neither solves 843. The exact
McNemar p-value is 0.000000346 and favors TV-nucleus on the same secondary
criterion.

At field-occurrence level, QAD has 5,778 regressions and 5,803 recoveries
versus published NVFP4, a net 25 recoveries across 14,031 value disagreements.
It has 5,813 regressions and 5,726 recoveries versus TV-nucleus, a net 87
regressions across 13,868 value disagreements. Fields and repeats belonging to
one task are dependent and are not independent samples.

## Task-family diagnostics

Each family contains 1,024 task clusters and 3,072 responses per checkpoint.
No family interval excludes zero. These are secondary diagnostics without a
multiple-comparison correction.

| Task family | QAD minus published NVFP4 | Paired 95% interval | QAD minus TV-nucleus | Paired 95% interval |
|---|---:|---:|---:|---:|
| Constraint assignment | -0.0911 points | -0.3041 to +0.1153 | -0.0223 points | -0.2511 to +0.1962 |
| Dependency graph | -0.1163 points | -0.4650 to +0.2232 | -0.1256 points | -0.4464 to +0.1814 |
| Event-sourced state | -0.1099 points | -0.4395 to +0.2279 | -0.1058 points | -0.4272 to +0.2116 |
| Evidence-chain retrieval | -0.0380 points | -0.2550 to +0.1872 | -0.1139 points | -0.3255 to +0.0949 |
| Policy application | -0.0592 points | -0.1827 to +0.0639 | -0.0934 points | -0.2085 to +0.0122 |
| Program execution | +0.7952 points | -1.1300 to +2.7297 | +0.5580 points | -1.3486 to +2.4786 |
| Record reconciliation | -0.1628 points | -0.8830 to +0.5615 | -0.3011 points | -1.0213 to +0.4313 |

The program-execution point estimates are unresolved and support no claim
about programming, code generation, debugging, or tool use.

## Length-limit diagnostic

QAD reaches the completion limit in 46 of 21,504 responses, compared with 73
for published NVFP4 and 52 for TV-nucleus.

| Comparator/QAD finish condition | Response pairs | Contribution to QAD-minus-comparator score |
|---|---:|---:|
| Published reaches limit; QAD stops | 71 | +0.3084 points |
| Published stops; QAD reaches limit | 44 | -0.1928 points |
| Both stop normally, QAD versus published | 21,387 | -0.0845 points |
| TV-nucleus reaches limit; QAD stops | 51 | +0.2151 points |
| TV-nucleus stops; QAD reaches limit | 45 | -0.1987 points |
| Both stop normally, QAD versus TV-nucleus | 21,407 | -0.0455 points |

Two published/QAD pairs and one TV-nucleus/QAD pair reach the limit in both
arms and contribute zero. The low and balanced limit rates show no QAD
looping penalty under the qualified nucleus-sampling contract. Finish-reason
groups are outcome-selected diagnostics and do not estimate a different
stopping policy.

## Evaluation and serving contract

The suite contains 1,024 generated tasks from each of constraint assignment,
dependency graph, event-sourced state, evidence-chain retrieval, policy
application, program execution, and record reconciliation. The [VBF method
page](verifier-backed-behavioral-fidelity.md) defines scoring and decision
rules.

| Role | Durable identity |
|---|---|
| Published NVFP4 | local-inference-lab/GLM-5.3-Flash-NVFP4 at revision 378ca54585c46542bad1f3cb3ed0d73ae51cdb62; model-index SHA-256 0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb |
| QAD step 2,500 | Materialization-manifest SHA-256 962f7905587be9e4377b8bebbcd8e93b49c3c49f3bda6d818260da518dcd5e0a; model-index SHA-256 b43d25a280d02bfd2a58c046386e24baad78fcce355ea2d48cc0c4c78671686b |
| QAD TV-nucleus step 2,500 | Materialization-manifest SHA-256 01a1ea703cef6e0e44cf788a774504ca430bad60beddde44779cf607743769c1; Quatrain checkpoint-manifest SHA-256 d0e37d9be7db72cd3e9ca8c19597d9b10905d8c526af044a1fb984b99af3b874 |

All checkpoints use R30 image localinferencelab/vllm at digest
sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131.
Each checkpoint uses four Tensor Parallelism 4 (TP4), Decode Context
Parallelism 1 (DCP1) replicas on GPU groups 0-3, 4-7, 8-11, and 12-15, with
96 scheduled sequences, CUDA graphs through 96 sequences, and 28 GiB of 8-bit
floating-point (FP8) key/value cache per GPU. Aggregate client concurrency is
384. Multi-Token Prediction (MTP) and speculative decoding are disabled.

A +6000 MHz memory-clock offset was present for all QAD and TV-nucleus
responses and for the last portion of the published-NVFP4 run, but not
uniformly for the first portion of the published run. No GPU error was
observed. Clock frequency is not a generation or scoring input, but this
asymmetry excludes a valid throughput comparison between checkpoints.

## Distribution-fidelity relationship

The [QAD step-2,500 distribution-fidelity report](../../kld/glm-5.3-flash-qad-step2500.md)
finds that held-out natural-route forward Kullback-Leibler Divergence (KLD)
from the Brain Floating Point 16-bit (BF16) reference is 0.129126 nats/token
for QAD, compared with 0.162164 for published NVFP4 and 0.145654 for
TV-nucleus. QAD improves on published NVFP4 by 20.373%. Its KLD is 0.016528
nats/token lower than TV-nucleus, with a paired interval from -0.020763 to
-0.012799 nats/token. Both comparisons exclude zero.

Under exact BF16-route replay, QAD KLD is 0.064066, compared with 0.067476 for
published NVFP4 and 0.066875 for TV-nucleus. QAD improves on published NVFP4
by 5.054%. Its KLD is 0.002809 nats/token lower than TV-nucleus, with a paired
interval from -0.003583 to -0.002065 nats/token. Both comparisons exclude
zero.

The KLD ranking therefore does not produce a resolved ranking on the
temperature-1/top-p-0.95 VBF semantic score. Teacher-forced KLD measures
next-token distribution fidelity before decoding; top-p does not change the
KLD calculation.

## Research-only greedy diagnostic

A separate temperature-zero comparison on the same 7,168 task identities
produced semantic scores of 90.959% for published NVFP4, 91.436% for QAD step
2,500, and 88.850% for TV-nucleus. QAD was higher than published NVFP4 by
0.476 points, with a paired 95% interval from -0.276 to +1.226 points. That
interval supports a not-worse decision under the declared one-point margin,
but not practical equivalence. QAD was higher than TV-nucleus by 2.586 points,
with an interval from +1.789 to +3.389 points.

The TV-nucleus greedy run used a different runtime and serving capacity from
the QAD greedy run. The QAD-versus-TV-nucleus result is therefore a
served-system comparison rather than a weight-only checkpoint comparison.

A separate temperature-zero evaluation pooled 9,856 non-overlapping tasks and
measured a QAD-minus-published semantic-score change of +0.332 points with a
paired 95% interval from -0.317 to +0.982 points. It was practically equivalent
under its own declared one-point margin.

A separately declared temperature-zero suite of 2,048 unseen numeric
instances from one deterministic integer-program template measured a QAD gain
of 8.371 points with a paired 95% interval from +5.985 to +10.742 points. The
R30 temperature-1/top-p-0.95 program family instead measures +0.795 points
with an interval from -1.130 to +2.730 points. The narrow greedy result does
not establish a general programming advantage. Cross-contract comparisons
are research-only because runtime image, scheduler geometry, decoding, and
repeat count are not all identical.

## Verification and reproducibility

All 64,512 responses across the three checkpoints completed without API
errors. An independent verifier executed all 1,024 program answer keys and
recomputed every receipt digest and score. It found no missing, duplicate, or
unexpected keys and no score mismatch.

The [machine-readable public summary](validation/r30-top-p095-vbf-three-checkpoint-20260910.json)
records full-precision values.

| Evidence | Durable identifier |
|---|---|
| Suite file | SHA-256 38fc5741abb98cc188a5ecd80d2c7b0f328e9615725317b54d8b32c663fc0b42 |
| Canonical task records | SHA-256 7c5fc36de45e8b388c98279c1666aba22e3daa8f8273a11d2378794913c76f43 |
| Evaluation contract | SHA-256 503680dbe53a4bd034c628b17e3091e65c7522a7100b2b1bd5c55112f93f7c41 |
| Published run / summary / verifier | fa179fa98e9f6fa6b669a2748316ede093a18dd1a410050a29014071baae8994 / 5d68a0aeaba1d2ae1f09757c43b36514a676e017dfd465b9305a11587de138cc / b9cf8274d51fb20632e9555423445895fe957a9a8f90b8802b28909527660b28 |
| QAD run / summary / verifier | 28e6a51b08c4f0bc2a35948e292c1e8029d4230d9b91797e1bee721820f5a201 / b9997e328c8172e2802011949d3282de0ca12f4ae98b8fbf3028ad0ebeafe2b7 / 343e65261acb1d9d7bf30c0db2182ae8380ee72ba68191c9ca7ae3a6577ec040 |
| TV-nucleus run / summary / verifier | 2f826f67590d10ddca6e53222bb40716aeafd494a53e2a4b825f2880447c3c12 / bc70065f436a7ce8a99484f53d17b597713b6b0272d0a3fdf8d9170d4a2fda54 / 494613616554d80c7ab1c5dbb296d46fdc5d8385c2d31b4e77590943965b4e50 |
| Published-reference comparison | SHA-256 a94be5c4220a17de848dd40db79ce31380079b1a2ea4a66af202bda318fd130d |
| TV-nucleus versus QAD comparison | SHA-256 f042ee1f1d5dbfac68bce630721c1e646cb33db9b7a9d34d1cdf7e3bcb6ce44c |
| Consolidated diagnostics | SHA-256 e3653cecd14e6d4dd8db37c77640eea2d7a18e44c748c0eb000c17f5ecee40af |
| Retained R30 artifact | /mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/r30-temperature1-top-p095-three-checkpoint-20260909 |

## Scope and limitations

The qualified claim concerns the task-weighted semantic score on seven
generated deterministic families under the named served-system contract. It
does not establish equivalence for every exact-task criterion, individual
family, free-form chat, factual recall, safety, coding work, or tool calls.
Stochastic served-system evidence does not isolate checkpoint weights from
every runtime interaction.

## Attribution

The deterministic task generators, executable answer keys, strict scoring
contract, durable receipt format, runtime-verification records, executions,
paired analysis, and report are Local Inference Lab work. Paired bootstrap
intervals and McNemar's exact test are established statistical methods rather
than claimed mathematical inventions. AI tools assisted implementation and
documentation under human direction and review.
