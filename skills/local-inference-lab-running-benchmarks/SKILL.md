---
name: local-inference-lab-running-benchmarks
description: Run and report controlled Local Inference Lab model, quant, image, and runtime benchmarks. Use for target-only decode, Estonia, LAVD, Hotel Lights, prefill, speculative, accuracy, regression, or candidate-versus-baseline testing.
license: MIT
compatibility: Requires Python 3.10+, Git, an OpenAI-compatible endpoint, and access to the target inference hardware.
metadata:
  author: local-inference-lab-community
  version: "1.5"
---

# Running Local Inference Lab Benchmarks

Run reproducible performance, stability, and quality comparisons for models, quantizations, images, runtime configurations, and patches.

The governing word is **controlled**: pin every identity, declare the decision being tested, change only named variables, and preserve the raw result files.

## Procedure

1. Read [references/rtx6kpro-wiki.md](references/rtx6kpro-wiki.md), select the current model runbook and relevant methodology pages, and record commit-pinned references.
2. Read [references/experimental-method.md](references/experimental-method.md). Freeze the observation and question; define the experimental unit, candidate, control, changed and nuisance variables, rival explanations, predictions, falsification condition, repetitions, run order, exclusions, and stopping rule before inspecting the target result.
3. Read [references/baseline-suites.md](references/baseline-suites.md) and select the smallest suite that answers the decision.
4. Read [references/llm-inference-bench.md](references/llm-inference-bench.md), pin the benchmark repository commit, inspect its current `README.md` and `--help`, and capture the serving endpoint configuration.
5. Run the target-only control before any speculative configuration. Verify that MTP, external draft models, n-gram speculation, and other speculative paths are disabled.
6. Run the selected profiles in the declared balanced or randomized order. Replicate at the experimental-unit level and preserve every raw result, including failures and contrary evidence.
7. Read [references/statistical-comparison.md](references/statistical-comparison.md). Report effect magnitude, uncertainty, per-run distributions, repeated-control variation, and exploratory versus confirmatory analyses without treating subsamples as independent runs.
8. Read [references/reporting.md](references/reporting.md), preserve raw JSON, and produce a result record from `assets/benchmark-run-record.template.md` plus a concise report from `assets/benchmark-report.template.md`.

Use these values rather than guessing:

- `UNKNOWN — needs verification` for a fact that should exist but has not been established.
- `Not tested` for a profile or configuration that was not run.
- `N/A` when a field genuinely does not apply.

## Default for a new model or quant

Unless the user selects another purpose, run:

1. target-only decode with speculation disabled;
2. Estonia and LAVD long-reasoning checks;
3. Hotel Lights only for the full suite, when resources permit, or when other results disagree;
4. qualitative coding prompts only as a separate evaluation, never as throughput data.

## Verification

Finish only when:

- the benchmark repository, engine, image, model, and component revisions are immutable;
- the hardware, topology, runtime, launch command, graph/cache/scheduler settings, reasoning effort, concurrency, and run count are recorded;
- candidate and control differ only by declared factors, or the result is labeled a system comparison;
- target-only and speculative results are reported separately;
- raw results have hashes and every skipped profile is marked `Not tested`;
- the experimental unit, repeated-control noise, run order, stopping rule, effect magnitude, uncertainty, and independent repetitions are visible;
- the conclusion addresses rival explanations and is no broader than the tested conditions.
