# Support, Escalation, and Supersession

## Custom-image thread

A custom image needs a dedicated support thread containing the complete required template. Keep launch help, bug reports, benchmark updates, and revisions in that thread. The model channel receives at most one short link to it.

The image author owns first-line support. State either:

- `author-supported`: owner, public contact, issue route, and maintenance commitment; or
- `ephemeral`: no maintenance promise, but the thread still records reports and exact revisions.

## Upstream escalation

Escalate a custom-image problem only when either:

1. it reproduces on the current recommended image under an equivalent configuration; or
2. a minimal reproducer identifies the responsible source change independently of the custom image.

Otherwise keep the problem with the custom image. Testing only another derivative is not sufficient.

A report must identify the exact custom and recommended digests, model revision, launch configuration, and outcome. Normalize configurations before attributing cause.

## Supersession

When the relevant change enters the recommended image:

1. set `maintenance_status: superseded`;
2. mark the support thread `superseded`;
3. identify the replacing recommended image digest;
4. stop directing new users to the custom image;
5. retain the thread as an auditable record.

A newer tag alone does not prove supersession; use the exact replacing digest and source evidence.
