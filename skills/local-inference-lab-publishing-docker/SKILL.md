---
name: local-inference-lab-publishing-docker
description: Prepare and validate Local Inference Lab Docker image publications, support threads, provenance records, qualification evidence, and lifecycle updates. Use when publishing a recommended or custom image, revising it, supporting it, or marking it superseded.
license: MIT
compatibility: Requires Docker or registry metadata access for digest verification; Python 3.10+ runs the bundled validator.
metadata:
  version: "1.5"
  author: local-inference-lab-community
---

# Publishing Local Inference Lab Docker Images

Prepare a Docker image publication that keeps one clear recommended runtime per model family while allowing custom experiments with explicit provenance, lifecycle, and support ownership.

Classify two independent dimensions:

- `distribution_role`: `recommended` or `custom`.
- `qualification_status`: `implemented`, `qualified`, `research-only`, or `unsupported`.

A custom image is an ephemeral test build unless its author explicitly accepts maintenance and support. Only designated maintainers may publish the recommended image.

## Procedure

1. Read [references/rtx6kpro-wiki.md](references/rtx6kpro-wiki.md), locate the current model-family runbook, and record its wiki commit and commit-pinned URL.
2. Read [references/image-policy.md](references/image-policy.md) and classify the model family, distribution role, release class, qualification, and maintenance status.
3. Read [references/provenance.md](references/provenance.md) and recover the final/base digests, public recipe, exact source commits, PRs, patches, overlays, package changes, build flags, defaults, and entrypoint changes.
4. Read [references/qualification.md](references/qualification.md), classify evidence as `exploratory`, `confirmatory`, or `qualification-evidence`, and verify only the exact tested hardware, topology, runtime, model revision, launch command, controls, experimental units, independent repetitions, effect magnitude, uncertainty, and benchmark claims.
5. Read [references/support-lifecycle.md](references/support-lifecycle.md) and create the required support route, upstream escalation boundary, and supersession plan.
6. Copy [assets/image-record.template.json](assets/image-record.template.json), fill every field, and use only `UNKNOWN — needs verification`, `Not tested`, or `N/A` for unavailable values.
7. Run non-strict validation and render a draft thread. Create the dedicated custom-image thread or recommended-image support route, upload the machine-readable record, and capture their public URLs.
8. Replace every unresolved `UNKNOWN — needs verification` required for publication and run `validate --strict` until it passes.
9. Render the final complete thread. For a custom image, render at most one short model-channel link with `render-main-link`. For a recommended image, render the maintainer release announcement with `render-recommended`.

Never fabricate a digest, commit, PR, benchmark result, validation result, or compatibility claim. A moving tag is not an image identity.

## Commands

```bash
python3 scripts/image_record.py validate image-record.json
python3 scripts/image_record.py render-thread image-record.json --output support-thread-draft.md

# After creating the support route and uploading the record:
python3 scripts/image_record.py validate image-record.json --strict
python3 scripts/image_record.py render-thread image-record.json --strict --output support-thread.md
python3 scripts/image_record.py render-main-link image-record.json --strict --output model-channel-post.md
python3 scripts/image_record.py render-recommended image-record.json --strict --output recommended-release.md
```

## Verification

Finish only when:

- final and base images use `image@sha256:<digest>` identities;
- the build is reproducible from a public recipe and exact source identities;
- inherited behavior is separate from image-specific changes;
- every validation or performance claim names the exact command, control, experimental unit, changed variables, repetitions, run order, effect magnitude, uncertainty, and evidence class;
- a critical confounder or unresolved rival explanation prevents the affected claim from using `qualification-evidence`;
- unperformed work is marked `Not tested`, not passed;
- the custom-image support thread exists before its model-channel link is posted;
- the author owns first-line support for a custom derivative;
- no secret, token, private hostname, personal path, or identifying log remains;
- the rendered Markdown contains every required section.
