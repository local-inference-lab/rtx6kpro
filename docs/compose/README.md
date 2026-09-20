# Full Compose configurations from shared model profiles

Choose and expand a deployment in the
[shared Docker guide](../unified-vllm-docker.md#expand-the-complete-default-configurations).
The same blocks appear on each model page. Download the corresponding
`.compose.yaml`, select the GPU IDs, and start it with `docker compose -f
FILE.compose.yaml up -d`. Use Docker Compose 2.23.1 or newer.

The expanded file contains all non-secret image/profile environment values
and resolved launch options. It calls the image's `runtime.explicit` entrypoint,
which keeps image validation and CUDA/NCCL bootstrap behavior. It does not load
another copy of the model defaults. GPU-only prefix caching is selected; inactive
LMCache settings remain visible but do not start a cache service.

Normal short commands remain the preferred interface when changing tensor
parallelism, speculation or external cache, because the shared resolver updates
dependent settings. Expanded exports are tied to the image release they describe.
Generate them again when changing such controls or upgrading the image.
Do not store Hugging Face credentials in a published Compose file.

## Regenerate the documentation

Offline regeneration uses the committed snapshot and needs no configuration
service. To refresh that snapshot for another release, maintainers provide the
image-aware catalog API used by the Docker configurator. Its URL is an operator
input, not a public website dependency. The API verifies the public release,
registry metadata and matching
[shared runtime sources](https://github.com/local-inference-lab/blackwell-llm-docker/tree/main/runtime)
before resolving the image's profiles. With that service available locally,
select a published release ID:

```bash
uv run --no-project --with pyyaml python scripts/generate-compose-reference.py \
  --catalog-url http://127.0.0.1:4173 --release-id 392295614 --write
```

`karmic-kraken-beta/catalog.json` retains the selected release, source identities
and every resolved response. It is evidence for reproduction, not a second
registry of model defaults. The script's deployment list contains only the
documented model/preset/mode selections. Hardware, memory, graph, cache and
backend settings come from the shared image resolver.

Offline freshness check:

```bash
uv run --no-project --with pyyaml python scripts/generate-compose-reference.py
LIL_RUNTIME_SOURCE=/path/to/blackwell-llm-docker \
  uv run --no-project --with pyyaml --with pytest \
  python -m pytest -q scripts/test_generate_compose_reference.py
```

The tests validate YAML with the real Compose parser, complete environment
coverage, GPU selection, shell quoting and native-argument parity. They do not
start model servers. The generator updates marked sections in existing pages;
handwritten commands, measurements and historical archives stay intact.

## Validate against the packaged image

With the matching image already downloaded, compare its profile and explicit
configuration paths without GPUs or model loading:

```bash
uv run --no-project --with pyyaml python scripts/validate-compose-image.py \
  docs/compose/karmic-kraken-beta --output /tmp/compose-validation.json
```

The validator uses `--runtime runc --network none`, one CPU, a one-GiB memory
limit and `--print-config`. No model/cache volumes or GPU device requests are
passed. `--ssh-target` and `--ssh-host-key-alias` select another Docker host.

[Packaged-image validation](karmic-kraken-beta/validation.json) records exact
image/file identities and all nine passing comparisons. This is configuration
parity, not another GPU performance measurement.
