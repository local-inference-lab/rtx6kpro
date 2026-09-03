---
name: local-inference-lab-reporting-bugs
description: Create reproducible Local Inference Lab bug reports and route support ownership from exact artifacts and controls. Use when an image, model, quant, engine, patch, or configuration fails or regresses.
license: MIT
compatibility: Requires access to the failing artifact, its base or recommended image, and enough runtime information to create a minimal reproducer.
metadata:
  author: local-inference-lab-community
  version: "1.5"
---

# Reporting Local Inference Lab Bugs

Create a support package or tracker report that ties a failure to exact artifacts, configurations, and source ownership.

The governing word is **reproducer**: route a problem only after reducing it to the smallest input and exact identities that still fail.

## Procedure

1. Read [references/rtx6kpro-wiki.md](references/rtx6kpro-wiki.md), identify the current runbook, check documented failure signatures, and record a commit-pinned runbook URL.
2. Read [references/support-routing.md](references/support-routing.md).
3. Read [references/diagnostic-method.md](references/diagnostic-method.md). Freeze the observation before interpretation, list candidate and rival causes, define discriminating tests and falsification conditions, and maintain the evidence ledger.
4. Capture the exact image digest or package ID, base/recommended image, source commits, model revision, patches, hardware, topology, runtime, launch command, cache state, expected behavior, and actual behavior.
5. Run matched controls that change one named variable where possible. Record supporting, contrary, negative, and indeterminate evidence without routing from component assumptions.
6. Reduce the failure to the smallest deterministic request, script, or sequence that preserves it.
7. For a custom image, keep first-line support in its dedicated thread.
8. Escalate upstream only when the problem reproduces on the current recommended image under an equivalent configuration, or a minimal reproducer independently identifies the responsible source change.
9. Complete `assets/bug-report.template.md`, attach logs/evidence with hashes, and use the delivery route selected by the user. A portable package is sufficient; GitHub is optional.

Use these values rather than guessing:

- `UNKNOWN — needs verification` for an unresolved fact.
- `Not tested` for an unrun reproduction/configuration.
- `N/A` when a field does not apply.

## Verification

Finish only when:

- exact failing and comparison identities are recorded;
- the observation, candidate causes, rival explanations, falsification conditions, and evidence ledger are present;
- the smallest available reproducer is included;
- custom, recommended, and source-level outcomes are distinguished;
- the selected owner follows from a discriminating test rather than engine/component assumptions;
- unresolved alternatives and critical validity threats remain explicit;
- credentials, private hostnames, personal paths, and sensitive logs are absent.
