---
name: local-inference-lab-sharing-changes
description: Package Local Inference Lab patches, changed files, scripts, reproducers, and evidence into a traceable portable archive. Use when sharing work directly through Discord or another channel without requiring GitHub.
license: MIT
compatibility: Requires Python 3.10+; Git is optional but recommended for patch creation.
metadata:
  version: "1.5"
  author: local-inference-lab-community
---

# Sharing Local Inference Lab Changes

Package patches, modified files, scripts, experiments, reproducers, or evidence so another person can inspect and apply the work without requiring a GitHub issue or pull request.

The governing word is **traceable**: identify the exact base, isolate the introduced delta, preserve human attribution, include exact application and reversal steps, and retain the evidence that supports every claim.

## Procedure

1. Read [references/rtx6kpro-wiki.md](references/rtx6kpro-wiki.md), locate the current model runbook when relevant, and record a commit-pinned runbook reference.
2. Identify the contribution kind, intended result, exact base, introduced delta, recipient, and requested delivery route. For experiments, also preserve the question, observation, experimental unit, candidate/control, changed and nuisance variables, rival explanations, prediction, falsification condition, run order, stopping rule, and exploratory or confirmatory status.
3. Recover durable identities from Git history, image metadata, manifests, benchmark JSON, and attached files before asking the user.
4. Read [references/package-format.md](references/package-format.md) and choose the smallest artifact that represents the work completely.
5. Read [references/community-writing.md](references/community-writing.md) and rewrite every package document so it stands alone.
6. Create the package from [assets/package/](assets/package/) or run `scripts/change_package.py init`.
7. Apply the package to a clean copy of the exact base and run the declared checks. Preserve raw observations, failed and contrary runs, independent-run boundaries, exclusions, effect magnitude, uncertainty, and the evidence needed to reproduce the conclusion.
8. Run `finalize`, `validate --strict`, and `archive`; then write the concise Discord summary from the validated manifest.

Use these values rather than guessing:

- `UNKNOWN — needs verification` for a fact that should exist but has not been established.
- `Not tested` for a test or configuration that was not run.
- `N/A` when a field genuinely does not apply.

Do not turn a package-only request into GitHub work. A complete ZIP is a valid final contribution.

## Commands

Resolve paths relative to this skill directory:

```bash
python3 scripts/change_package.py init ./my-change --package-id my-change-r1
python3 scripts/change_package.py finalize ./my-change
python3 scripts/change_package.py validate ./my-change --strict
python3 scripts/change_package.py archive ./my-change --output ./my-change-r1.zip
```

## Verification

Finish only when:

- the exact base and every included artifact have durable identities;
- the package applies or installs from a clean base;
- `README.md`, `MANIFEST.json`, `TESTING.md`, and `SHA256SUMS` agree;
- every claim is bounded by exact conditions, independent evidence, uncertainty, rival explanations, and a conclusion no broader than the tested setup;
- no credentials, personal paths, private hostnames, or undeclared files remain;
- the Discord message is short and points to the complete package.
