# Community Writing and Evidence Rules

## Exact identities

Use full Git commits and result trees, immutable image digests, model repository plus immutable revision, PR head commits, and SHA-256 for patches, archives, raw results, and evidence. Branch names, tags, dates, and labels are attributes rather than identities.

## Attribution

Separate inherited behavior from the introduced delta. Preserve human authorship through commits, patch metadata, manifest fields, or explicit credit. A coding agent is tooling, not the human author.

## Evidence boundary

State each result as:

```text
Conditions:
Measurement:
Result:
Conclusion:
Limitations:
```

A screenshot or summary table is a view of evidence, not the evidence identity. Retain the raw result.

## Status vocabulary

- `implemented`: the change exists, but the claimed matrix is not fully validated.
- `qualified`: the listed validation passed under every exact stated condition.
- `research-only`: useful evidence or experiment without a maintenance recommendation.
- `unsupported`: known not to work or outside the support boundary.

## Stand-alone prose

Every README, manifest, test record, comment, commit, report, and announcement must make sense to a technically capable reader who has the artifact and linked public sources.

- Describe the resulting system and its purpose.
- Introduce semantic roles before internal labels.
- State present behavior, technical reason, compatibility impact, and validation.
- Put chronology only in explicitly historical material.
- Write comments for invariants and non-obvious constraints.
- Give TODOs a missing condition and removal criterion.

Rewrite any sentence that depends on unstated history.
