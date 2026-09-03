# Evidence-Bounded Diagnosis

Use this method to reduce a failure without locking onto the first plausible component. The output is still an inference-runtime bug report, not a general scientific report.

## Separate observation from explanation

Freeze the observable failure first:

- exact input and expected versus actual behavior;
- first known working and failing identities;
- frequency and timing;
- experimental unit, repetitions, and sample counts;
- complete environment and launch state;
- raw logs and artifacts.

Then list candidate causes. Keep a mechanism, component name, or correlation labeled as a hypothesis until a discriminating test bears on it.

## Generate rivals before routing

Include materially different explanations when plausible:

- source regression or patch interaction;
- image, package, or entrypoint difference;
- launch or scheduler configuration;
- model or quantization revision;
- MTP/DFlash acceptance or draft behavior;
- JIT, cache, graph-capture, or warmup state;
- hardware topology, link health, power, clocks, thermal state, or background load;
- request, parser, sampling, or client behavior;
- measurement error or ordinary run-to-run variation.

Do not force a fixed number. The point is to prevent confirmation bias and choose tests that separate live alternatives.

## Choose discriminating tests

For each candidate cause, state:

1. the result it predicts;
2. a rival that predicts a different result;
3. the smallest test that separates them;
4. the exact control and changed variable;
5. a result that weakens or falsifies the candidate;
6. an indeterminate result and the next test it requires.

Prefer matched one-variable comparisons. Use the current recommended image as the distribution control for custom-image failures, and use the immediate base or source checkout when isolating an introduced delta. Repeat a control against itself when nondeterminism or state drift could explain a small difference.

## Evidence ledger

Maintain this table while reducing the failure:

| Claim or candidate cause | Evidence for | Evidence against | Discriminating test | Status |
|---|---|---|---|---|
| UNKNOWN — needs verification | UNKNOWN — needs verification | UNKNOWN — needs verification | UNKNOWN — needs verification | unresolved |

Allowed statuses are:

- `candidate`: plausible but not yet discriminated;
- `strengthened`: new evidence is more likely under this explanation than named rivals;
- `weakened`: contrary evidence exists but is not decisive;
- `falsified`: the declared falsification condition occurred under a valid test;
- `unresolved`: evidence or control is insufficient.

Record negative and contrary evidence as eagerly as supporting evidence. A surviving candidate is not proven merely because the tested rivals failed.

## Review validity before escalation

Classify issues as:

- `critical`: the comparison cannot identify the claimed cause;
- `important`: the issue materially limits interpretation;
- `minor`: worth recording but does not change the main routing decision.

Apply the same standard to the suspected source and the control. Route ownership only when the minimal reproducer or matched comparison isolates the source boundary. Otherwise report the unresolved candidates and keep support with the artifact owner defined by the support policy.

## Attribution

This inference-specific method adapts rival-hypothesis, falsification, evidence-ledger, and validity-review principles from K-Dense AI's MIT-licensed `hypothesis-generation` and `scientific-critical-thinking` skills at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f).
