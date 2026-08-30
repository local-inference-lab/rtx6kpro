# RTX PRO 6000 Blackwell Wiki Sources

Use the [https://github.com/local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) field wiki to recover repository-local intent, source contracts, qualified invariants, and historical regression evidence before reconciling overlapping runtime changes.

## Lookup order

1. Resolve and record the current wiki `master` commit.
2. Start from [`README.md#start-here`](https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md#start-here), then open the current model hub/runbook.
3. Use the runbook's source contract, backend selection, cache invariants, startup evidence, qualification conditions, and limitations as intent sources.
4. Use [`INDEX.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/INDEX.md) to find exact historical pages when commit messages, issues, or PRs name a prior release.
5. Publish commit-pinned wiki links in the resolution record.

[`optimization/dspark-upstream-consolidation.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/dspark-upstream-consolidation.md) is a useful reconciliation-method example: it compares parallel implementations with disjoint histories, separates end-to-end evidence from code-level differences, identifies strengths and blockers on both sides, proposes directional ports, and records measured decisions. Reuse that method, not its conclusions, unless the same branches, commits, hardware, and runtime conditions are being reconciled.

Versioned pages and daily summaries are historical primary evidence for a named result. They do not override the current model hub when deciding the present supported configuration.
