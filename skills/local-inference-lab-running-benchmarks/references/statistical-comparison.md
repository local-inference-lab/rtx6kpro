# Statistical Comparison for Inference Results

Use this guide after the comparison design is fixed. Analysis cannot repair confounding, pseudoreplication, or an uncontrolled change.

## Analyze the right unit

Match the analysis to the experimental unit declared in the plan. Requests from one server instance describe within-run variation; they do not become independent server replications because there are many of them. Preserve both levels:

- per-request or per-sample observations for distribution shape and tail behavior;
- per-run summaries for variation across independently prepared runs.

Do not pool observations across machines, topologies, server starts, or protocols and report the pooled sample count as independent replication.

## Inspect before summarizing

For candidate and control, report:

- every independent run and its sample count;
- failures, missing values, cancellations, truncations, and exclusions;
- raw distributions or quantiles appropriate to the measurement;
- warmup and collection boundaries;
- repeated-control variation when available.

A single mean or best run is insufficient for a distributional or small-delta claim. Tail claims must name the sampled quantity and show whether the tail appears within each run or only after pooling unlike runs.

## Compare magnitude and uncertainty

Report the absolute and relative effect with units. Add an uncertainty interval or the full independent-run distribution appropriate to the design; do not rely on a thresholded p-value or one point estimate.

Use paired comparisons when candidate and control runs are deliberately paired within the same block and order schedule. Use independent comparisons when they are genuinely independent. With few independent runs, emphasize raw run values, effect magnitude, repeated-control noise, and the limited precision rather than making asymptotic claims the design cannot support.

A difference is not operationally meaningful merely because it is detectable. Define the practical regression or improvement threshold before looking at the target result when the decision requires one.

## Assumptions and multiplicity

- Check whether the selected summary and comparison match skewed, heavy-tailed, censored, or bounded data.
- Keep prefill, sustained decode, burst/end-to-end throughput, speculative decode, completion-token distributions, accuracy, and qualitative observations separate.
- Predeclare the primary outcome. Label additional metrics, subgroups, and post-hoc cuts exploratory.
- If many configurations or outcomes are searched, report the complete search space and account for multiplicity; the winning cell is not an unbiased estimate of its advantage.
- Preserve the stopping rule. Extending collection only when the result is unfavorable or unclear invalidates the original decision rule.

## Synthesize evidence without erasing heterogeneity

Evidence from compatible independent runs may be summarized, but retain per-run and per-block results. Treat a different machine, topology, image digest, model revision, workload, or protocol as a separate condition unless the design explicitly models that factor.

When combining evidence:

1. verify that every result answers the same question and uses comparable measurements;
2. group results by material condition and experimental unit;
3. report disagreement and plausible moderators instead of averaging it away;
4. weight conclusions by design validity and independence, not by the number of request rows;
5. distinguish reproducibility on the same setup from replication on an independently prepared setup.

Several correlated measurements from one run are one line of evidence, not a consensus. A broad claim requires convergence across independent units and the boundaries named in the claim.

## Claim review

Publish each claim as:

```text
Conditions: exact candidate, control, blocks, and held-constant settings
Measurement: command, experimental units, sample counts, run order, exclusions, and aggregation
Result: absolute and relative effects with units, uncertainty, failures, and raw evidence
Conclusion: supported scope, rival explanations not excluded, and limitations
Status: confirmatory or exploratory
```

The conclusion is unsupported when the observed advantage falls within control-versus-control variation, does not survive independent repetition, depends on an undeclared exclusion, or is confounded with another changed factor.

## Attribution

This inference-specific guide adapts principles from K-Dense AI's MIT-licensed `statistical-analysis` and `scientific-critical-thinking` skills, including question-first analysis, distribution inspection, assumption checks, effect magnitude, uncertainty, consistent validity standards, and evidence-bounded conclusions, at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f).
