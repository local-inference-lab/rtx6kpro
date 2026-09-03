# Experimental Method for Inference Benchmarks

Use this method for candidate-versus-control, regression, attribution, tail-latency, stability, and multi-factor comparisons. It adapts general experimental-design principles to inference systems; it is not a separate scientific workflow.

## Keep the objects distinct

| Object | Meaning |
|---|---|
| Question | The decision the benchmark must inform |
| Observation | The measured behavior that motivated the test, before interpretation |
| Experimental unit | The independently prepared unit that can vary on its own |
| Candidate | The configuration being evaluated |
| Control | The directly comparable reference configuration |
| Changed variable | The named difference intended to explain the result |
| Nuisance variable | A non-target difference that can move the measurement |
| Rival explanation | Another cause consistent with the observation |
| Prediction | The result expected before the target run is inspected |
| Falsification condition | A result that would make the claim unsupported |
| Measurement | The command, quantity, units, and aggregation |
| Evidence | Raw observations that bear on the claim |
| Conclusion | The narrow interpretation supported by the evidence |

A plausible mechanism is not evidence. A difference observed after several settings changed does not identify which setting caused it.

## Plan before running

Record the following before inspecting the target result:

1. **Question and observation.** State what was observed without causal language and name the decision the test must answer.
2. **Experimental unit.** Define the level of independent preparation. When server state is relevant, one independently started and warmed server instance is usually one unit. Requests within that instance are subsamples, not independent server replications.
3. **Candidate, control, and factors.** Pin every identity. Name every changed factor and every setting held constant.
4. **Nuisance variables.** Include machine and topology, thermal and clock state, JIT and cache state, request scheduling, model sampling, background load, and run order when relevant.
5. **Rivals.** List credible alternatives before choosing tests: ordinary run-to-run variation, an unintended configuration difference, MTP acceptance changes, cache or compilation state, topology, measurement error, or an anomalous sample.
6. **Predictions and falsification.** State what candidate and rival explanations predict, what measurement separates them, and what result would leave the claim unsupported.
7. **Primary measurement and stopping rule.** Predeclare the primary outcome, run count, exclusions, aggregation, failure handling, and when collection stops. Label later analyses exploratory.

## Controls, replication, and order

- Establish a noise floor with repeated control-versus-control runs when the expected candidate delta is small.
- Replicate at the experimental-unit level. Five requests from one server process are not five independent server runs.
- Alternate or randomize candidate/control order with a recorded schedule. Balanced sequences such as `ABBA` and `BAAB` reduce monotonic warmup, thermal, cache, and clock drift without pretending to remove them.
- Block comparisons by machine, GPU topology, model revision, or other unavoidable nuisance factor. Compare within blocks before combining them.
- Preserve failed, cancelled, truncated, and negative runs. Do not replace them silently.
- Repeat the complete comparison on an independently prepared unit when the claim is intended to generalize beyond one server state.

## One factor or several

For causal attribution, change one factor and hold the rest fixed. If the question concerns several interacting switches, use an explicit factorial matrix rather than unrelated configuration combinations. A factorial result may estimate main effects and interactions; a fastest observed combination alone cannot assign causality to one component.

Example factors might include activation precision, MTP depth, custom all-reduce, and DCP width. Record the complete matrix, randomized or balanced run order, repeated cells, and any combinations that are invalid or untested.

## Minimum comparison record

```text
Question:
Observation:
Experimental unit:
Candidate:
Control:
Changed variable(s):
Held constant:
Nuisance variables:
Rival explanations:
Design and run order:
Primary measurement:
Secondary measurements:
Stopping and exclusion rules:
Prediction:
Falsification condition:
Exploratory analyses:
```

## Evidence review

Before publishing a conclusion, ask:

- Does the design measure the question it claims to answer?
- Can a nuisance variable or rival explanation account for the result?
- Is the effect larger than repeated-control variation?
- Do independent repetitions agree, and are incompatible conditions reported separately?
- Are effect magnitude, uncertainty, failures, and limitations visible?
- Does the conclusion stop at the tested hardware, topology, image, model, workload, and run protocol?

Apply the same standard to favorable and unfavorable results. Classify validity threats as `critical`, `important`, or `minor`; do not let a long list of cosmetic limitations obscure one fatal confounder.

## Attribution

This inference-specific method adapts principles from K-Dense AI's MIT-licensed `hypothesis-generation`, `experimental-design`, and `scientific-critical-thinking` skills at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f), including rival explanations, falsification, experimental units, replication, blocking, run-order control, and validity review.
