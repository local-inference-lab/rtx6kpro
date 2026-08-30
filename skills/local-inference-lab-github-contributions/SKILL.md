---
name: local-inference-lab-github-contributions
description: Prepare focused Local Inference Lab GitHub issues and pull requests with exact identities, attribution, and validation. Use only when the user explicitly chooses GitHub as the delivery route.
license: MIT
compatibility: Requires Git and access to the selected public GitHub repository. GitHub delivery must be explicitly selected by the user.
metadata:
  author: local-inference-lab-community
  version: "1.5"
---

# Preparing Local Inference Lab GitHub Contributions

Prepare focused GitHub issues and pull requests only when the user chooses GitHub as the delivery route.

The governing word is **focused**: one technical question, exact source identities, a small review surface, and validation that matches the resulting behavior.

## Procedure

1. Read [references/rtx6kpro-wiki.md](references/rtx6kpro-wiki.md), inspect current model hubs and prior work, and record commit-pinned documentation sources.
2. Read [references/github-workflow.md](references/github-workflow.md).
3. Inspect repository guidance, current target branch, merge base, open issues, open PRs, and overlapping branches before editing.
4. State the problem, observable result, scope, non-goals, exact base, compatibility impact, and validation plan in `assets/coordination-note.template.md` when coordination is useful.
5. Apply or split the change so the issue/PR answers one technical question.
6. Preserve human attribution and record included, alternative, superseded, experimental, rejected, deferred, or out-of-scope sources.
7. Run repository checks and write a stand-alone issue or `assets/pull-request-description.template.md`. Label each material statement as demonstrated behavior, hypothesis, or speculation; for experimental claims include the control, experimental unit, repetitions, effect magnitude, uncertainty, rival explanations, validity threats, and raw evidence.
8. Offer a portable package for Discord testing, but do not open additional PRs or issues unless the user requests them.

Use `UNKNOWN — needs verification`, `Not tested`, and `N/A` instead of guessing or removing fields.

## Verification

Finish only when:

- GitHub was explicitly selected;
- overlapping work was inspected and linked rather than duplicated;
- the contribution is narrow and based on exact commits;
- every demonstrated claim has a command, result, and evidence, while hypotheses and speculation remain explicitly labeled;
- experimental conclusions disclose effect magnitude, uncertainty, rivals, critical validity threats, and tested boundaries;
- attribution and reviewed exclusions are preserved;
- the issue/PR text makes sense with only the current repository state.
