# Docker Image Publishing Policy

## Contents

- Recommended image
- Custom images
- Status fields
- Required publication gate
- Missing-information vocabulary

## Recommended image

Each model family has one recommended, maintainer-supported image.

- Main announcements and the server's automated image listing or bot point to this image.
- It must pass the project's correctness, stability, and performance regression gates.
- It uses `distribution_role: recommended`, `maintenance_status: maintainer-supported`, `qualification_status: qualified`, and a maintainer-owned approval source.
- Only designated maintainers may replace the recommended image.

## Custom images

Custom images are welcome and use `distribution_role: custom`.

- Create one dedicated support thread in the corresponding forum or model section before wider linking. If the correct location is unclear, ask a server maintainer before posting.
- Post at most one link to that thread in the relevant model channel.
- Keep reports concerning the image in its support thread.
- Treat the image as `ephemeral` unless the author explicitly commits to maintaining and supporting it.
- When its relevant changes enter the recommended image, mark the image and thread `superseded` and identify the replacing recommended digest.
- An image with insufficient provenance may be withheld or removed from main-channel linking until its documentation is corrected.

Using another community image as the base is acceptable. Credit the base, document every change, and support the derivative.

## Status fields

### Release class

- `experimental`: exploratory image without a release guarantee.
- `community-derivative`: community image derived from an exact named base.
- `official`: maintainer-owned release with a durable approval source.

### Distribution role

- `recommended`: the single supported general-user image for the model family.
- `custom`: an independent experiment or derivative with its own support route.

### Qualification status

- `implemented`: image exists; full claimed matrix is incomplete.
- `qualified`: all listed gates passed under the exact stated conditions.
- `research-only`: evidence or experiment, not a recommendation.
- `unsupported`: known not to work or outside the support boundary.

### Maintenance status

- `maintainer-supported`: maintained by the project maintainers.
- `author-supported`: custom-image author accepts maintenance and first-line support.
- `ephemeral`: test build with no maintenance promise.
- `superseded`: replaced by an exact recommended image digest.

## Required publication gate

Before publication confirm:

- final and base image digests;
- public Dockerfile or build script and complete build command;
- exact vLLM/SGLang, B12X, and other component identities;
- all PRs, patches, overlays, package changes, flags, defaults, and entrypoints;
- inherited versus introduced functionality;
- tested hardware, topology, drivers, runtime, model, quant, parallelism, caches, graphs, scheduler, and launch command;
- validation commands and raw evidence;
- digest-pinned baseline and exact benchmark commands for performance claims;
- limitations and untested configurations;
- support thread and owner;
- security screening.

## Missing-information vocabulary

Use exactly:

- `UNKNOWN — needs verification`: the fact should exist but has not been established.
- `Not tested`: the test or configuration was not run.
- `N/A`: the field genuinely does not apply.

Do not omit fields or substitute guesses.
