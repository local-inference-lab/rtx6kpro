# Portable Change Package

## Contents

- Package layout
- Delta formats
- Manifest requirements
- Application safety
- Validation
- Share message

## Community runbook reference

Record the current `local-inference-lab/rtx6kpro` commit, the relevant repository-relative runbook path, a commit-pinned URL, and the runbook's relationship to the package. The runbook supplies context; the package's own base identities, deltas, and evidence remain authoritative.

## Package layout

```text
<package-id>/
├── README.md
├── MANIFEST.json
├── TESTING.md
├── patches/
├── files/
├── evidence/
└── SHA256SUMS
```

Include only directories that carry content. Keep every path relative and portable.

- `README.md`: resulting behavior, exact base, application, compatibility, limits, and support expectation.
- `MANIFEST.json`: machine-readable identities, artifacts, attribution, validation, and limitations.
- `TESTING.md`: exact commands, conditions, results, and evidence paths.
- `patches/`: Git patches, diffs, or bundles.
- `files/`: replacement or additional files only when a patch is unsuitable.
- `evidence/`: raw JSON, logs, screenshots, reproducers, and reports.
- `SHA256SUMS`: digest for every package file except itself.

## Delta formats

Prefer the first complete representation:

| Situation | Artifact |
|---|---|
| One or more commits; preserve authorship | `git format-patch --full-index --binary` |
| Uncommitted working tree | `git diff --full-index --binary <base> --` |
| Branch history must travel | `git bundle` |
| Non-Git files | Replacement files with old/new SHA-256 and exact paths |
| Container overlay | Exact base digest plus patch/files and build/application instructions |

```bash
git format-patch --full-index --binary --stdout <base>..HEAD > patches/change.patch
git diff --full-index --binary <base> -- > patches/change.diff
git bundle create patches/change.bundle <base>..HEAD
```

Verify the artifact against a clean checkout of the exact base. Prefer an inspectable patch over an unexplained filesystem overlay.

## Manifest requirements

Record:

- package ID, title, summary, status, and contribution kind;
- human authors or submitter;
- exact base repository and commit, image digest, archive hash, or file hashes;
- each artifact path, kind, purpose, and SHA-256;
- inherited and introduced behavior;
- application and reversal commands;
- compatibility impact;
- validation commands, conditions, results, and evidence;
- known, untested, and unsupported cases;
- support contact or a one-off/no-support statement.

Public URLs are optional when the package itself contains the evidence.

For experimental evidence, `TESTING.md` additionally preserves the question, observation, experimental unit, candidate/control, changed and nuisance variables, rival explanations, prediction, falsification condition, independent repetitions, run order, stopping and exclusion rules, effect magnitude, uncertainty, negative or contrary runs, and confirmatory or exploratory status. Keep raw observations separate from interpretation.

## Application safety

Provide non-destructive inspection before application:

```bash
git apply --stat patches/change.diff
git apply --check patches/change.diff
git apply patches/change.diff
```

For mail-formatted commits:

```bash
git apply --stat patches/change.patch
git apply --check patches/change.patch
git am patches/change.patch
```

For replacement files, list destination, old hash, new hash, backup, and reversal. Never write outside the intended repository or workspace.

## Validation

Apply only package contents to a clean base, run the declared checks, and retain raw output under `evidence/` where practical.

```bash
python3 scripts/change_package.py finalize path/to/package
python3 scripts/change_package.py validate path/to/package --strict
python3 scripts/change_package.py archive path/to/package --output contribution.zip
```

The experiment-record principles above adapt K-Dense AI's MIT-licensed `hypothesis-generation`, `experimental-design`, `statistical-analysis`, and `scientific-critical-thinking` skills at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f) to portable inference change packages.

## Share message

```text
[Change package] <semantic title>

Base: <repository/image/archive identity>
Change: <one or two sentences>
Status: <implemented | qualified | research-only | unsupported>
Tested: <short result or Not tested>
Limitations: <short boundary>
Package: <attachment or link>
Support: <contact or one-off/no-support statement>
```
