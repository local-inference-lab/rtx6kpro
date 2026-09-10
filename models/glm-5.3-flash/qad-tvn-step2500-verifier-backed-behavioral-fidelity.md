# GLM-5.3-Flash QAD TV-nucleus step 2,500 verifier-backed behavioral fidelity

Status: **qualified** for the declared R30 temperature-1/top-p-0.95
served-system comparisons; **research-only** as evidence about general model
quality or a weight-only checkpoint effect.

This report compares
GLM-5.3-Flash-NVFP4-QAD-TVN-step2500 with the published Local Inference Lab
NVIDIA 4-bit floating-point (NVFP4) checkpoint and Quantization-Aware
Distillation (QAD) step 2,500 trained without the Total-Variation (TV)
nucleus objective. Verifier-Backed Behavioral Fidelity (VBF) uses
procedurally generated tasks, executable answer keys, strict JSON scoring, and
paired whole-task uncertainty estimates.

The TV-nucleus training objective targets teacher and student probability mass
inside their temperature-1/top-p-0.95 nuclei and includes a
continuation-versus-stop log-odds term. The generation contract therefore
tests the distribution regime targeted by the objective.

## Qualified result

Each checkpoint answered the same 7,168 tasks three times, producing 21,504
responses per checkpoint. Generation used temperature 1.0, top-p 0.95, fixed
task/repeat seeds, maximum reasoning effort, and a 32,768-token completion
limit.

| Comparator | Comparator score | TV-nucleus score | TV-nucleus change | Paired 95% interval | Decision |
|---|---:|---:|---:|---:|---|
| Published NVFP4 | 94.5386% | 94.5989% | **+0.0603 points** | **-0.2581 to +0.3785 points** | **practically equivalent** |
| QAD step 2,500 | 94.5698% | 94.5989% | **+0.0292 points** | **-0.2739 to +0.3325 points** | **practically equivalent** |

Both 100,000-sample paired task-cluster bootstrap intervals lie completely
inside the predeclared ±1-point practical-equivalence band and cross zero.
The TV-nucleus checkpoint is therefore practically equivalent to both
comparators on the primary endpoint. Neither comparison establishes that it is
better.

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

Against published NVFP4, both checkpoints solve 4,753 exact clusters,
TV-nucleus alone solves 834, published alone solves 787, and neither solves
794. The exact McNemar p-value is 0.253. Against QAD step 2,500, both solve
4,639, TV-nucleus alone solves 948, QAD alone solves 738, and neither solves
843. The exact McNemar p-value is 0.000000346 and favors TV-nucleus on this
stricter secondary criterion.

At field-occurrence level, TV-nucleus has 5,732 regressions and 5,844
recoveries versus published NVFP4, a net 112 recoveries. It has 5,726
regressions and 5,813 recoveries versus QAD step 2,500, a net 87 recoveries.
Fields and repeats belonging to one task are dependent and are not independent
samples.

## Task-family diagnostics

Each family contains 1,024 task clusters and 3,072 responses per checkpoint.
No family interval excludes zero. These are secondary diagnostics without a
multiple-comparison correction.

| Task family | TV-nucleus minus published NVFP4 | Paired 95% interval | TV-nucleus minus QAD | Paired 95% interval |
|---|---:|---:|---:|---:|
| Constraint assignment | -0.0688 points | -0.2623 to +0.1228 | +0.0223 points | -0.1962 to +0.2511 |
| Dependency graph | +0.0093 points | -0.2976 to +0.3255 | +0.1256 points | -0.1814 to +0.4464 |
| Event-sourced state | -0.0041 points | -0.3459 to +0.3418 | +0.1058 points | -0.2116 to +0.4272 |
| Evidence-chain retrieval | +0.0760 points | -0.1302 to +0.2930 | +0.1139 points | -0.0949 to +0.3255 |
| Policy application | +0.0342 points | -0.0692 to +0.1475 | +0.0934 points | -0.0122 to +0.2085 |
| Program execution | +0.2372 points | -1.7764 to +2.2554 | -0.5580 points | -2.4786 to +1.3486 |
| Record reconciliation | +0.1383 points | -0.6226 to +0.8911 | +0.3011 points | -0.4313 to +1.0213 |

The program-execution point estimates are unresolved and support no claim
about programming, code generation, debugging, or tool use.

## Length-limit diagnostic

The TV-nucleus checkpoint reaches the completion limit in 52 of 21,504
responses, compared with 73 for published NVFP4 and 46 for QAD step 2,500.

| Comparator/TV-nucleus finish condition | Response pairs | Contribution to TV-nucleus-minus-comparator score |
|---|---:|---:|
| Published reaches limit; TV-nucleus stops | 72 | +0.3167 points |
| Published stops; TV-nucleus reaches limit | 51 | -0.2231 points |
| Both stop normally | 21,380 | -0.0333 points |
| QAD reaches limit; TV-nucleus stops | 45 | +0.1987 points |
| QAD stops; TV-nucleus reaches limit | 51 | -0.2151 points |
| Both stop normally, TV-nucleus versus QAD | 21,407 | +0.0455 points |

One published/TV-nucleus pair and one QAD/TV-nucleus pair reach the limit in
both arms and contribute zero. The low and balanced limit rates show no
TV-nucleus looping penalty under the qualified nucleus-sampling contract.
Finish-reason groups are outcome-selected diagnostics and do not estimate a
different stopping policy.

## Evaluation and serving contract

The suite contains 1,024 generated tasks from each of constraint assignment,
dependency graph, event-sourced state, evidence-chain retrieval, policy
application, program execution, and record reconciliation. The [VBF method
page](verifier-backed-behavioral-fidelity.md) defines scoring and decision
rules.

| Role | Durable identity |
|---|---|
| Published NVFP4 | local-inference-lab/GLM-5.3-Flash-NVFP4 at revision 378ca54585c46542bad1f3cb3ed0d73ae51cdb62; model-index SHA-256 0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb |
| QAD step 2,500 | Materialization-manifest SHA-256 962f7905587be9e4377b8bebbcd8e93b49c3c49f3bda6d818260da518dcd5e0a |
| QAD TV-nucleus step 2,500 | Materialization-manifest SHA-256 01a1ea703cef6e0e44cf788a774504ca430bad60beddde44779cf607743769c1; Quatrain checkpoint-manifest SHA-256 d0e37d9be7db72cd3e9ca8c19597d9b10905d8c526af044a1fb984b99af3b874 |

All checkpoints use R30 image
localinferencelab/vllm at digest
sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131.
Each checkpoint uses four Tensor Parallelism 4 (TP4), Decode Context
Parallelism 1 (DCP1) replicas on GPU groups 0-3, 4-7, 8-11, and 12-15, with
96 scheduled sequences, CUDA graphs through 96 sequences, and 28 GiB of 8-bit
floating-point (FP8) key/value cache per GPU. Aggregate client concurrency is
384. Multi-Token Prediction (MTP) and speculative decoding are disabled.

A +6000 MHz memory-clock offset was present for all TV-nucleus and QAD
responses and for the last portion of the published-NVFP4 run, but not
uniformly for the first portion of the published run. No GPU error was
observed. Clock frequency is not a generation or scoring input, but this
asymmetry excludes a valid throughput comparison between checkpoints.

## Distribution-fidelity relationship

The [TV-nucleus distribution-fidelity report](../../kld/glm-5.3-flash-qad-tvn-step2500.md)
finds that held-out natural-route forward Kullback-Leibler Divergence (KLD)
from the Brain Floating Point 16-bit (BF16) reference is 0.145654 nats/token
for TV-nucleus, compared with 0.162164 for published NVFP4 and 0.129126 for
QAD step 2,500. The TV-nucleus checkpoint improves on published NVFP4 by
10.181% but regresses against QAD step 2,500 by 12.800%; both paired intervals
exclude zero.

Under exact BF16-route replay, TV-nucleus KLD is 0.066875, compared with
0.067476 for published NVFP4 and 0.064066 for QAD. The published comparison is
inconclusive; QAD has significantly lower KLD.

The KLD ranking therefore does not produce a resolved ranking on the
temperature-1/top-p-0.95 VBF semantic score. Teacher-forced KLD measures
next-token distribution fidelity before decoding; top-p does not change the
KLD calculation.

## Research-only greedy diagnostic

A separate temperature-zero comparison produced semantic scores of 90.959%
for published NVFP4, 91.436% for QAD step 2,500, and 88.850% for TV-nucleus.
TV-nucleus was lower than published NVFP4 by 2.109 points (95% interval
-2.922 to -1.308) and lower than QAD by 2.586 points (interval -3.389 to
-1.789).

That TV-nucleus regression does not replicate under the matched R30
temperature-1/top-p-0.95 contract. The greedy TV-nucleus run reached the
length limit in 411 of 7,168 responses; the R30 nucleus-sampling run reached
it in 52 of 21,504 responses. Against published NVFP4, asymmetric greedy
length-limit outcomes account arithmetically for the complete aggregate
disadvantage.

The two evaluations also differ in runtime image, serving capacity, task
assignment, and repeat count. Their contrast shows that the deployed result is
sensitive to the complete decoding and serving contract; it cannot attribute
the change solely to top-p or checkpoint weights.

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
