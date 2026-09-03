# Merge Reconciliation

Use for an in-progress merge, rebase, cherry-pick, or a planned integration where branches overlap semantically.

## Recover intent before editing

Record the current `rtx6kpro` wiki commit and any commit-pinned model runbook, source contract, or historical investigation that bears on the integration. A current runbook establishes present support intent; a historical page establishes only the exact result it documents.

Inspect:

```bash
git status
git merge-base <side-a> <side-b>
git log --oneline <base>..<side-a>
git log --oneline <base>..<side-b>
git diff <base>..<side-a>
git diff <base>..<side-b>
```

Read the primary sources for each side: issue/spec, PR body, commits, design documents, tests, and comments that explain invariants. Do not infer intent from the conflict markers alone.

## Classify every conflict

- `disjoint-intent` — both changes can coexist; preserve both.
- `same-question-different-answer` — choose the answer matching the declared integration goal and record the trade-off.
- `superseded` — keep the surviving design and state why the other premise no longer applies.
- `generated-output` — resolve the source and regenerate the derived file.

Do not invent a hybrid neither side requested.

## Discriminate incompatible answers

For `same-question-different-answer`, do not choose the cleaner-looking implementation by default. State:

- the observable question both sides answer;
- each candidate's mechanism and prediction;
- rival explanations for any apparent advantage;
- the control, experimental unit, changed variables, and nuisance variables;
- the smallest test where the candidates predict different outcomes;
- the falsification condition, stopping rule, effect magnitude, uncertainty, and raw evidence;
- critical validity threats and unresolved alternatives.

Change one material factor where possible. If several design differences remain coupled, record a system comparison rather than attributing the result to one implementation detail. Apply the same evidence standard to both sides and preserve contrary or failed runs.

When compatible independent tests disagree, keep results separated by environment, topology, workload, or other material condition. Do not average heterogeneous evidence into a false consensus.

## Neutrality and attribution

When reconciling two contributors or parallel agents, use a neutral reviewer or third agent where practical. Supply both diffs and both intent sources. Do not favor the side that invoked the reconciler.

Preserve human authorship through commits, trailers, or the integration record.

## Validate and finish

Discover and run the repository checks, typically type checking, focused tests, broader tests, then formatting. Fix only breakage caused by the integration.

Stage the resolution and finish the merge or rebase. Do not abort merely because the conflict is difficult.

For non-trivial decisions, complete `assets/merge-resolution-record.template.md` with:

- conflicting interface/file;
- intent on each side;
- classification;
- chosen resolution;
- preserved and dropped behavior;
- compatibility impact;
- tests;
- final result commit/tree.

The discriminating-test and validity-review principles above adapt K-Dense AI's MIT-licensed `hypothesis-generation`, `experimental-design`, and `scientific-critical-thinking` skills at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f) to implementation reconciliation.

## Done when

- No conflict markers or unmerged paths remain.
- Every semantic conflict has a recorded decision.
- Generated files were regenerated from resolved sources.
- Checks pass or failures are explicitly bounded.
- The merge/rebase is complete and authorship is preserved.
