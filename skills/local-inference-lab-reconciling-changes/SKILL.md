---
name: local-inference-lab-reconciling-changes
description: Reconcile Local Inference Lab merges, rebases, cherry-picks, and multi-source integrations while preserving intent, attribution, and tested behavior. Use when Git reports conflicts or overlapping changes must be composed.
license: MIT
compatibility: Requires Git and access to the repository's issues, pull requests, specifications, and test commands where available.
metadata:
  author: local-inference-lab-community
  version: "1.5"
---

# Reconciling Local Inference Lab Changes

Resolve merges, rebases, cherry-picks, and multi-source integrations without losing intent, attribution, or tested behavior.

The governing word is **intent**: conflict markers show text overlap; primary sources and tests show what each side was trying to preserve.

## Procedure

1. Read [references/rtx6kpro-wiki.md](references/rtx6kpro-wiki.md), recover relevant current source contracts and named historical evidence, and record commit-pinned sources.
2. Read [references/reconciliation-method.md](references/reconciliation-method.md).
3. Inspect the merge/rebase state, merge base, both histories, both diffs, conflicting files, issues/specifications, PRs, tests, and design documents.
4. State each side's intent in repository-local language before resolving a hunk.
5. Classify each conflict as `disjoint-intent`, `same-question-different-answer`, `superseded`, or `generated-output`.
6. Preserve both compatible intents. When answers are incompatible, use the discriminating-test method: state candidate predictions and rivals, run the smallest controlled comparison that separates them, record falsification conditions and contrary evidence, and choose only the answer supported for the declared integration goal. Do not invent behavior neither side requested.
7. Preserve authorship through commits, trailers, or the integration record.
8. Resolve sources before generated outputs; regenerate lockfiles, manifests, generated code, and patches.
9. Run the repository's focused and broad checks, finish the merge/rebase, and complete `assets/merge-resolution-record.template.md` for non-trivial decisions.

Use `UNKNOWN — needs verification`, `Not tested`, and `N/A` rather than guessing or dropping fields.

## Verification

Finish only when:

- no conflict markers or unmerged paths remain;
- every semantic conflict has a documented classification and decision;
- surviving and deliberately dropped behavior are explicit;
- incompatible answers were resolved from controlled evidence, or their unresolved validity limits are recorded;
- generated outputs derive from resolved sources;
- checks pass or failures are precisely bounded;
- the merge/rebase is complete and authorship is preserved.
