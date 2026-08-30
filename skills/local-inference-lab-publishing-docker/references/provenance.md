# Image Provenance and Reproducibility

Record the current `local-inference-lab/rtx6kpro` wiki commit and commit-pinned model runbook alongside the image record. The runbook identifies community guidance; verify the image digest, source composition, and build evidence directly.

## Required identities

Record:

- final image repository, tag, and immutable digest;
- base image repository, tag, and immutable digest;
- public Dockerfile or build script URL and exact recipe commit;
- complete build command;
- vLLM or SGLang repository and full commit or verified release reference;
- B12X and every other relevant component commit/version;
- every included PR with URL, head commit, title, and human authors;
- every patch with URL/path, SHA-256, purpose, and human authors;
- every file overlay with source, destination, old/new hash, and purpose;
- package additions/removals/upgrades;
- build arguments, environment defaults, and entrypoint changes;
- resulting source tree and integration patch hash for composed trees;
- model repository and immutable revision used for validation.

## Build requirements

A publication build should:

1. Start from `FROM <base>@sha256:<digest>`.
2. Fail on a dirty checkout or mismatched source identity.
3. Fetch or copy only declared files.
4. Verify archive and patch hashes before application.
5. Record component commits and resulting trees during the build.
6. Avoid moving branch heads and unpinned downloads.
7. Preserve recipe logs or a machine-readable build receipt.

After pushing, resolve the registry digest and publish `repository:tag@sha256:<digest>`. A local image ID is supporting evidence, not the public identity.

## Change separation

Write three lists:

- **Inherited**: behavior already present in the exact base; credit it clearly.
- **Introduced**: only source commits, patches, overlays, package changes, flags, defaults, profiles, or validation assets added by this image.
- **Compatibility impact**: changed CLI flags, environment variables, model formats, caches, graph capture, topology, runtime requirements, volumes, entrypoints, or upgrade/downgrade behavior.

Do not list inherited functionality as an image-specific change.

## Composition record

For each included or reviewed source record the repository, full commit, PR head, authors, purpose, disposition, and technical reason. Record the integration base, composition strategy, resulting commit/tree, patch hash, conflicts, and tests. Consolidation does not erase attribution.
