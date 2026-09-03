# Focused GitHub Workflow

## Confirm the route

A package, patch, benchmark report, or Discord thread is a complete contribution. Use GitHub only when the user explicitly asks for an issue, PR, review, or upstream submission.

## Inspect before editing

Read the target repository's `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, issue/PR templates, active target branch, open issues, and open PRs. Resolve the full target commit and merge base.

Search for work answering the same technical question. Join or build on the existing owner when the objective is the same. A parallel PR is justified only for a genuinely different implementation, engine, topology, or qualification matrix; state that distinction.

## Focus the change

A focused issue or PR states:

- the stand-alone technical problem;
- exact target/base and comparison range;
- resulting observable behavior;
- technical reason or invariant;
- compatibility impact;
- inherited work and human attribution;
- validation commands and results;
- untested and unsupported cases;
- intentionally excluded follow-up work.

Remove drive-by cleanup, unrelated formatting, broad renames, and dependency changes that enlarge the review surface.

## Convert a portable package

1. Treat its manifest as the provenance source.
2. Apply it to a clean branch at the recorded base.
3. Preserve original commit authorship where possible.
4. Split unrelated changes.
5. Link the package as evidence, but keep the issue/PR body stand-alone.
6. Run the repository's checks.

## Multi-source composition

For every included or reviewed source, record:

- repository and full commit;
- PR number and head commit when applicable;
- semantic purpose;
- human authors;
- disposition: included, alternative, experimental, superseded, rejected, deferred, or out-of-scope;
- technical reason.

Record the integration base, composition strategy, result tree/commit, patch hash, conflicts, and validation. Consolidation does not erase attribution.

## Claim discipline

Keep demonstrated behavior, hypotheses, and speculation distinct. A technical mechanism or plausible source boundary remains a hypothesis until a controlled test bears on it. For experimental claims, preserve the observation, control, experimental unit, changed and nuisance variables, rival explanations, prediction, falsification condition, repetitions, run order, stopping rule, exclusions, effect magnitude, uncertainty, raw evidence, and critical validity threats.

Weight overlapping evidence by independence and design validity rather than row count. Keep heterogeneous hardware, topologies, images, models, workloads, and protocols separate unless the design explicitly supports synthesis. State confirmatory versus exploratory status and limit the conclusion to the tested conditions.

## Repository prose

Describe the system as it exists after the change. State what, why, how, compatibility impact, and validation. Write from the repository state and linked evidence; put chronology or failed attempts only in an explicitly historical document.

These claim-review principles adapt K-Dense AI's MIT-licensed `scientific-critical-thinking`, `hypothesis-generation`, and `statistical-analysis` skills at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f) to Local Inference Lab contributions.
