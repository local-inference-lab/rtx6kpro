# Controlled Qualitative Comparisons

Use this guide when a prompt evaluation compares models, revisions, images, quants, or serving configurations. It applies core experimental principles without turning the evaluation into a scientific-specific workflow or inventing a universal score.

## Freeze the protocol

Before generating candidate outputs, record:

- the exact question the comparison should answer;
- prompt text and hash;
- harness, system prompt, tools, reasoning effort, sampling settings, token limit, and execution mode;
- the experimental unit, candidate, and control identities;
- changed, held-constant, and nuisance variables;
- observable rubric, primary decision rule, and effect measure;
- candidate and rival predictions plus the falsification condition;
- independent repetition count, run order, stopping rule, exclusions, and treatment of failures;
- whether the comparison is exploratory or confirmatory.

Manual repair, prompt editing, retries, and tool interventions are separate conditions, not invisible cleanup.

## Define independence and control order

One generation is one evaluation unit for output-level observations. Repeated inspection of the same generated artifact does not create independent repetitions. When evaluating a stable tendency rather than one-shot capability, use independently generated outputs and preserve every output.

Randomize or balance candidate/control order when practical so evaluator fatigue, changing runtime state, or a learned expectation does not align with one candidate. Hide candidate labels from the evaluator when practical. Record the order, any available sampling seed, and whether blinding was possible.

Keep model, prompt, harness, tools, and evaluation environment fixed except for the named factor. When several factors change, label the result a system comparison rather than attributing the difference to one component.

## Separate observation and interpretation

Record direct observations first: build result, runtime behavior, console errors, visible rubric behaviors, and artifact evidence. Then state the interpretation and plausible rivals, such as sampling variance, evaluator expectations, token-limit differences, tool access, or a manual intervention.

Apply the same rubric and severity standard to favored and disfavored candidates. Preserve negative, failed, and ambiguous outputs. A qualitative observation may complement throughput or accuracy evidence but does not replace it.

Report the candidate-control effect for each decision-bearing rubric item, using outcome counts or another protocol-specific magnitude rather than an invented universal score. Report uncertainty across independent outputs. When repetitions are too few to support an interval, show every independent outcome and state that precision is limited; a one-shot difference has no independent-run uncertainty.

## Bounded conclusion

A result may support:

- one-shot capability under the exact preserved protocol;
- a repeated tendency under the exact protocol when independent repetitions agree;
- an exploratory difference that needs a stronger controlled comparison.

It does not by itself establish general model quality, production reliability, causal attribution to one component, or performance.

## Attribution

This inference-evaluation guide adapts protocol control, independent replication, run-order control, blinding, observation/interpretation separation, and validity-review principles from K-Dense AI's MIT-licensed `experimental-design` and `scientific-critical-thinking` skills at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f).
