# GLM-5.3-Flash QAD step 2,500 verifier-backed behavioral fidelity

Status: **qualified** for the declared R30 temperature-1/top-p-0.95
served-system comparison; **research-only** as evidence about general model
quality or a weight-only checkpoint effect.

This report compares the published Local Inference Lab GLM-5.3-Flash NVIDIA
4-bit floating-point (NVFP4) checkpoint with the Quantization-Aware
Distillation (QAD) artifact GLM-5.3-Flash-NVFP4-QAD-step2500.
Verifier-Backed Behavioral Fidelity (VBF) uses procedurally generated tasks,
executable answer keys, strict JSON scoring, and paired whole-task uncertainty
estimates.

## Qualified result

Each checkpoint answered the same 7,168 tasks three times. Generation used
temperature 1.0, top-p 0.95, fixed task/repeat seeds, maximum reasoning effort,
and a 32,768-token completion limit.

| Published NVFP4 | QAD step 2,500 | QAD minus published NVFP4 | Paired 95% interval | Decision |
|---:|---:|---:|---:|---|
| 94.5386% | 94.5698% | **+0.0311 points** | **-0.2748 to +0.3380 points** | **practically equivalent** |

The complete 100,000-sample paired task-cluster bootstrap interval lies inside
the predeclared ±1-point practical-equivalence band. The interval crosses
zero, so the result does not establish that QAD is better.

## Outcome diagnostics

| Metric | Published NVFP4 | QAD step 2,500 | Difference |
|---|---:|---:|---:|
| Semantic score | 94.5386% | **94.5698%** | +0.0311 points |
| Exact responses | **19,285/21,504 (89.6810%)** | 19,112/21,504 (88.8765%) | -0.8045 points |
| Field micro-accuracy | 95.9058% | **95.9176%** | +0.0118 points |
| Protocol-valid responses | 21,383/21,504 (99.4373%) | **21,439/21,504 (99.6977%)** | +0.2604 points |
| Length-limited responses | 73 | **46** | -27 |
| Completion tokens | 109,256,432 | **104,649,137** | -4,607,295 |

An exact task cluster counts as correct only when all three responses are
exact. Both checkpoints solve 4,631 clusters; published NVFP4 alone solves
909, QAD alone solves 746, and neither solves 882. The two-sided exact McNemar
p-value is 0.0000674 and favors published NVFP4 on this stricter secondary
criterion.

At field-occurrence level, QAD has 5,778 harmful regressions and 5,803
recoveries, a net 25 recoveries across 14,031 value disagreements. Fields and
repeats within one task are dependent and are not independent samples.

The three repeat-level semantic-score changes are +0.3876, -0.0009, and
-0.2933 percentage points. Their changing signs illustrate why the primary
analysis clusters repeats and why one sampled response is not enough to
characterize a stochastic checkpoint.

## Task-family diagnostics

Every family contains 1,024 task clusters and 3,072 responses per checkpoint.
No family interval excludes zero. Family results are secondary diagnostics;
seven intervals were inspected without a multiple-comparison correction.

| Task family | Published NVFP4 | QAD step 2,500 | QAD change | Paired 95% interval |
|---|---:|---:|---:|---:|
| Constraint assignment | 99.8056% | 99.7145% | -0.0911 points | -0.3041 to +0.1153 |
| Dependency graph | 99.1025% | 98.9862% | -0.1163 points | -0.4650 to +0.2232 |
| Event-sourced state | 98.5107% | 98.4009% | -0.1099 points | -0.4395 to +0.2279 |
| Evidence-chain retrieval | 99.0533% | 99.0153% | -0.0380 points | -0.2550 to +0.1872 |
| Policy application | 99.6454% | 99.5862% | -0.0592 points | -0.1827 to +0.0639 |
| Program execution | 69.6894% | 70.4846% | +0.7952 points | -1.1300 to +2.7297 |
| Record reconciliation | 95.9635% | 95.8008% | -0.1628 points | -0.8830 to +0.5615 |

The program-execution point estimate is positive but unresolved. It does not
establish a programming, code-generation, debugging, or tool-use advantage.

## Finish-reason diagnostic

QAD reaches the completion limit in 46 of 21,504 responses, compared with 73
for published NVFP4. Paired finish-reason groups account arithmetically for
the aggregate score change:

| Published/QAD finish condition | Response pairs | Contribution to QAD-minus-published score |
|---|---:|---:|
| Published reaches limit; QAD stops | 71 | +0.3084 points |
| Published stops; QAD reaches limit | 44 | -0.1928 points |
| Both stop normally | 21,387 | -0.0845 points |
| Both reach limit | 2 | 0.0000 points |

These groups were selected from observed outcomes. They diagnose the served
system but do not estimate performance under a different completion limit.

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

Both checkpoints use R30 image
localinferencelab/vllm at digest
sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131.
Each checkpoint uses four independent Tensor Parallelism 4 (TP4), Decode
Context Parallelism 1 (DCP1) replicas on GPU groups 0-3, 4-7, 8-11, and
12-15, with 96 scheduled sequences and 28 GiB of 8-bit floating-point (FP8)
key/value cache per GPU. Aggregate client concurrency is 384. Multi-Token
Prediction (MTP) and speculative decoding are disabled.

A +6000 MHz memory-clock offset was present for every QAD response and for the
last portion of the published-NVFP4 run, but not uniformly for the first
portion of the published run. No GPU error was observed. Clock frequency is
not a generation or scoring input, but this asymmetry excludes a valid
throughput comparison between checkpoints.

## Distribution-fidelity relationship

The [QAD step-2,500 distribution-fidelity report](../../kld/glm-5.3-flash-qad-step2500.md)
finds that QAD reduces held-out forward Kullback-Leibler Divergence (KLD) from
the Brain Floating Point 16-bit (BF16) reference by 20.373% under natural
routing and 5.054% under exact BF16-route replay relative to published NVFP4.
Both predeclared KLD intervals exclude zero.

That measurable distribution improvement accompanies practical equivalence,
not a resolved improvement, on the R30 nucleus-sampling VBF semantic score.
Lower teacher-forced KLD means closer next-token distribution matching under
the KLD contract. It does not guarantee better sampled task answers.

## Research-only cross-contract evidence

A separate temperature-zero evaluation pooled 9,856 non-overlapping tasks and
measured a QAD-minus-published semantic-score change of +0.332 points with a
paired 95% interval from -0.317 to +0.982 points. It was practically equivalent
under its declared one-point margin.

A separately declared temperature-zero suite of 2,048 unseen numeric
instances from one deterministic integer-program template measured a QAD gain
of 8.371 points with a paired 95% interval from +5.985 to +10.742 points. That
result is qualified for the narrow template and does not represent general
programming ability.

The R30 temperature-1/top-p-0.95 program family measures +0.795 points with an
interval from -1.130 to +2.730 points. The narrow greedy gain therefore does
not provide a general capability conclusion for the deployment-aligned
sampling contract. Cross-contract comparisons are research-only because the
runtime image, scheduler geometry, decoding, and repeated-sampling design are
not all identical.

## Verification and reproducibility

All 43,008 declared responses across the two checkpoints completed. An
independent verifier executed all 1,024 program answer keys and recomputed all
21,504 receipt digests and scores for each checkpoint with zero mismatches.
The [machine-readable public summary](validation/r30-top-p095-vbf-three-checkpoint-20260910.json)
records full-precision values.

| Evidence | Durable identifier |
|---|---|
| Suite file | SHA-256 38fc5741abb98cc188a5ecd80d2c7b0f328e9615725317b54d8b32c663fc0b42 |
| Canonical task records | SHA-256 7c5fc36de45e8b388c98279c1666aba22e3daa8f8273a11d2378794913c76f43 |
| Evaluation contract | SHA-256 503680dbe53a4bd034c628b17e3091e65c7522a7100b2b1bd5c55112f93f7c41 |
| Published run / summary / verifier | fa179fa98e9f6fa6b669a2748316ede093a18dd1a410050a29014071baae8994 / 5d68a0aeaba1d2ae1f09757c43b36514a676e017dfd465b9305a11587de138cc / b9cf8274d51fb20632e9555423445895fe957a9a8f90b8802b28909527660b28 |
| QAD run / summary / verifier | 28e6a51b08c4f0bc2a35948e292c1e8029d4230d9b91797e1bee721820f5a201 / b9997e328c8172e2802011949d3282de0ca12f4ae98b8fbf3028ad0ebeafe2b7 / 343e65261acb1d9d7bf30c0db2182ae8380ee72ba68191c9ca7ae3a6577ec040 |
| Comparison JSON | SHA-256 a94be5c4220a17de848dd40db79ce31380079b1a2ea4a66af202bda318fd130d |
| Retained R30 artifact | /mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/r30-temperature1-top-p095-three-checkpoint-20260909 |

## Scope and limitations

The qualified claim concerns the task-weighted semantic score on seven
generated deterministic families under the named served-system contract. It
does not establish equivalence for exact-task accuracy, every family,
free-form chat, factual recall, safety, coding work, or tool calls. VBF
responses are sampled outcomes, so the result does not isolate checkpoint
weights from every runtime interaction.

## Attribution

The deterministic task generators, executable answer keys, strict scoring
contract, durable receipt format, runtime-verification records, executions,
paired analysis, and report are Local Inference Lab work. Paired bootstrap
intervals and McNemar's exact test are established statistical methods rather
than claimed mathematical inventions. AI tools assisted implementation and
documentation under human direction and review.
