---
name: local-inference-lab-evaluating-prompts
description: Run repeatable one-shot qualitative coding comparisons while preserving exact prompts and generated artifacts. Use when evaluating a model with Tetris, the single-page platformer, Flamingo, or another supplied prompt.
license: MIT
compatibility: Requires the model endpoint and a workspace capable of building or previewing generated web applications.
metadata:
  author: local-inference-lab-community
  version: "1.5"
---

# Evaluating Models with Qualitative Prompts

Run repeatable one-shot coding evaluations separately from throughput, reasoning, and accuracy benchmarks.

The governing word is **unaltered**: preserve the exact prompt, protocol, harness conditions, generated files, and manual intervention record.

## Procedure

1. Read [references/rtx6kpro-wiki.md](references/rtx6kpro-wiki.md) and record the current serving runbook context without importing its results into this prompt evaluation.
2. Select one prompt and read its complete protocol:
   - Tetris: [references/tetris.md](references/tetris.md)
   - Single-page platformer: [references/platformer.md](references/platformer.md)
   - Flamingo: [references/flamingo.md](references/flamingo.md)
3. When comparing candidates, read [references/comparison-method.md](references/comparison-method.md) and predeclare the evaluation unit, candidate, control, changed and nuisance variables, primary rubric, run order, independent repetitions, stopping rule, exclusions, and exploratory or confirmatory status.
4. Save the exact prompt as `prompt.txt` and calculate its SHA-256.
5. Record the model/revision, quantization, image/engine, harness/system prompt, available tools, reasoning effort, temperature, token limit, and execution mode.
6. Run one prompt in one turn without repair by default. For repeated comparisons, use the declared balanced or randomized order and blind candidate labels when practical. Use an agentic or iterative protocol only when explicitly selected, and label it separately.
7. Preserve every generated file before changing it. Build and run the output using the documented commands.
8. Complete `assets/qualitative-prompt-run.template.md` with direct observations before interpretation, rubric findings, console/build errors, evidence, candidate order, manual changes, and comparison effect magnitude and uncertainty.
9. Report the result separately from performance or accuracy measurements.

Use these values rather than guessing:

- `UNKNOWN — needs verification` for an unresolved fact.
- `Not tested` for an unrun behavior or platform.
- `N/A` when a field genuinely does not apply.

## Verification

Finish only when:

- the prompt text/hash and protocol are exact;
- the original generated output is preserved;
- build and runtime commands/results are recorded;
- every manual repair is separated from the one-shot result;
- the evaluation unit, exact control, changed and nuisance variables, run order, independent repetitions, stopping rule, exclusions, failures, and blinding status are recorded when candidates are compared;
- direct observations are separated from interpretation and plausible rival explanations;
- observations follow the prompt-specific rubric without inventing a universal score;
- comparative claims report prompt-specific effect magnitude and uncertainty or the full independent-run outcomes;
- the conclusion is limited to the preserved protocol, and proprietary source assets were not copied into the generated platformer.
