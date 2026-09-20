# One Docker image for GLM, Qwen and DeepSeek

> Historical recipe snapshot. Use the [model guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/docs/unified-vllm-docker.md) for the recommended deployment. This snapshot pins the September 19 beta image; performance tables retain their original image and hardware identities. [Archive manifest](https://github.com/local-inference-lab/rtx6kpro/blob/master/archive/serving-guides/karmic-kraken-beta-20260919-cfc67a15ebc3daf7/manifest.json).

Choose a model profile and GPU IDs. The shared image supplies that model's
backend, graph, loader and cache defaults. You do not need a model-specific
entrypoint or a copied block of kernel variables.

## Select the image

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7
docker pull "$IMAGE"
```

This is the Karmic Kraken integration channel: CUDA 13.4.1, PyTorch 2.14,
vLLM, B12X, FlashInfer and LMCache. Linux x86-64, Docker, NVIDIA Container
Toolkit and a compatible NVIDIA driver are required. The examples target
96-GB RTX PRO 6000 GPUs. No command changes clocks.

All channels use the same build recipe and launch interface:

| Image tag after `ghcr.io/local-inference-lab/vllm:` | vLLM branch | B12X branch |
|---|---|---|
| `karmic-kraken-beta` | `integration/karmic-kraken-beta` | `integration/karmic-kraken-beta` |
| `karmic-kraken` | `dev/karmic-kraken` | `master` |
| `jovian-judgement-beta` | `integration/beta` | `integration/beta` |
| `jovian-judgement` | `dev/jovian-judgement` | `master` |

Use the beta channel for the integrated fixes listed in
[issue #808](https://github.com/local-inference-lab/vllm/issues/808).
Published image versions and source manifests are available in the
[container releases](https://github.com/local-inference-lab/blackwell-llm-docker/releases).
Pulling an image does not change a running container; recreate that container
in a maintenance window to update it.

## Choose a model

Each model page has a complete copyable command:

| Model | Selector before the image | GPUs | Default speculation |
|---|---|---:|---|
| [GLM-5.3-Flash](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash.md) | `-e PROFILE=glm53-flash` | 4 | Off; the page starts MTP3 explicitly |
| [GLM Spark TP2](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash-spark-tp2.md) | `-e PRESET=glm53-spark-tp2` | 2 | MTP3 |
| [Qwen3.8 Flash Next](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/qwen38-flash-next.md) | `-e PROFILE=qwen38-flash-next` | 1 or 2 | MTP3 |
| [DeepSeek V4 text](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/deepseek-v4-flash.md) | `-e PROFILE=ds4-flash` | 2 | DSpark K5 |
| [DeepSeek V4 Vision](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/deepseek-v4-flash-vision.md) | `-e PROFILE=ds4-vision` | 2 | DSpark K3 |
| [DeepSeek V4.1](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/deepseek-v4.1-flash.md) | `-e PROFILE=ds41-flash` | 4 | Adaptive DSpark K7 |

Profiles provide Hugging Face checkpoint names. `MODEL` overrides the checkpoint
within that architecture; it does not select another architecture. Spark TP2
is a separate memory configuration, not the TP4 recipe with only `TP=2` changed.

## Start a server

Example: GLM, four GPUs, MTP3, API port 8000:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7
docker pull "$IMAGE"
docker run -d --name glm53 --init --restart unless-stopped \
  --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v glm53-runtime:/cache \
  -e PROFILE=glm53-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=4 -e PORT=8000 "$IMAGE" --mode mtp --draft-tokens 3
```

Change Docker options and `-e` variables **before** the image name.
Native serving arguments go **after** it. For example:

| Change | Before the image | Equivalent native argument after the image |
|---|---|---|
| Two model shards | `-e TP=2` | `--tensor-parallel-size 2` |
| DCP2 | `-e DCP=2` | `--decode-context-parallel-size 2` |
| API port | `-e PORT=8001` | `--port 8001` |
| Maximum active requests | `-e MAX_NUM_SEQS=16` | `--max-num-seqs 16` |
| Prefill token budget | `-e MAX_NUM_BATCHED_TOKENS=4096` | `--max-num-batched-tokens 4096` |
| Context limit | `-e MAX_MODEL_LEN=131072` | `--max-model-len 131072` |
| Fixed eight-GiB KV allocation | `-e KV_CACHE_MEMORY_BYTES=8589934592` | `--kv-cache-memory-bytes 8589934592` |

Expose as many GPU IDs as TP requires. DCP must divide TP and be supported
by the selected model. Changing memory/graph settings can reduce capacity or
prevent startup; the model pages list the measured configurations.

```bash
docker logs -f glm53
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
docker stop glm53
docker start glm53
```

The API binds `0.0.0.0`; keep it on a trusted network or configure authentication.
For simultaneous servers use different GPU IDs, ports, names and runtime volumes.
The HF volume can be shared. Named volumes survive container removal.
`--restart unless-stopped` restores the container after Docker/host startup,
but does not restore overclock settings.

## Select speculation

Put one choice after `"$IMAGE"`:

| Choice | Arguments | Models |
|---|---|---|
| Target only | `--mode off` | All profiles |
| MTP3 | `--mode mtp --draft-tokens 3` | GLM, Qwen |
| DFlash2 K7 | `--mode dflash2 --draft-tokens 7` | GLM |
| DSpark K5 | `--mode dspark --draft-tokens 5` | DS4 text |
| DSpark K3 | `--mode dspark --draft-tokens 3` | DS4 Vision |
| Adaptive DSpark K7 | `--mode dspark --draft-tokens 7` | DS4.1 |

The draft count is a maximum proposal length, not the number guaranteed to
be accepted. GLM DFlash2 uses the offline MXFP8 draft
`local-inference-lab/GLM-5.3-Flash-DFlash2`. DS4 standard MTP needs a different
checkpoint contract; use its model page rather than changing only the mode.

## Cache storage: GPU, LMCache or native offload

GPU-only prefix caching is the default. LMCache adds host RAM and optional
disk storage for reusable request prefixes. It does not make the active
context larger and is separate from PLE/Engram model-table offload.

Add these variables before the image for **RAM cache**:

```bash
-e CACHE_MODE=lmcache -e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2
```

For **RAM plus disk**, add:

```bash
-e CACHE_MODE=lmcache -e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2 \
-e LMCACHE_L2_ENABLED=1 -e LMCACHE_L2_GB=64
```

The image starts and supervises the CPU-only cache service. Its persistent
directory is inside `/cache`; service ports derive from the model API port.
LMCache reserves API-port + 10000, + 10001 and + 10002. For simultaneous
instances, space API ports at least three apart, for example 8000 and 8003,
or explicitly choose non-overlapping cache-service ports.
The GLM Spark TP2 page includes the bounded worker settings for that deployment.
Check host RAM, `/dev/shm` and disk capacity. With `--ipc host`, Docker's
`--shm-size` does not enlarge host shared memory.

| Model | RAM/disk prefix restore | Image-bearing requests |
|---|---|---|
| GLM, including Spark TP2 | Text supported | Vision runs, but external recurrent restore is skipped |
| Qwen | Text supported | Vision runs, but external recurrent restore is skipped |
| DS4 text | Supported | Use the Vision model for images |
| DS4 Vision | Supported | Image-keyed restore supported |
| DS4.1 | Supported | Image-keyed restore supported |

See [cache test results](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/karmic-kraken-serving.md#prefix-cache-checks).
GLM additionally exposes native KV offload through `--cache-mode native`;
that is not the LMCache path measured here. Other profiles reject that choice.

### Prefix cache defaults

Leave prefix policy to the model profile. GLM keeps exact recurrent checkpoints
at request boundaries. Qwen's native automatic policy selects its supported
recurrent handling. DeepSeek uses its own attention-cache structure.

Do not add one global `--prefix-cache-retention-interval 4096` to every model.
That parameter changes recurrent checkpoint retention, not the scheduling batch
or a cache expiry time. The profile and cache adapter select compatible geometry.

## Qwen PLE and DS4.1 Engram placement

These learned n-gram tables are model weights, not prefix KV or draft speculators.

| Model | RAM | Disk |
|---|---|---|
| Qwen | `-e VLLM_PLE_TABLE_MEMORY=ram` before the image | `-e VLLM_PLE_TABLE_MEMORY=disk` before the image |
| DS4.1 | `--engram-table-memory ram` after the image | `--engram-table-memory disk` after the image |

Qwen's default is CPU PLE offload; keep it enabled for the one-GPU recipe.
DS4.1 defaults to disk Engram. Disk paths need fast local storage and still use
host working memory. DS4.1's command includes `--ulimit memlock=-1` and
`--security-opt seccomp=unconfined` for pinned memory and io_uring. Use trusted
images when relaxing that syscall filter. Placement alternatives do not carry
an unmeasured speed guarantee.

## Inspect or override the launch

Inspect without loading a model or allocating GPUs:

```bash
docker run --rm --runtime runc --network none \
  -e PRESET=glm53-spark-tp2 "$IMAGE" --print-config
```

The output includes native vLLM arguments, relevant environment, the source of
each setting and the cache-service plan. Secrets are redacted. Explicit native
arguments override environment aliases, which override preset/model defaults.

For an advanced setting, use a native argument after the image or `-e NAME=value`
before it. Keep the image entrypoint. Do not paste shell text into
`EXTRA_VLLM_ARGS`; the launcher forwards argument arrays without evaluating a shell.
The [complete parameter reference](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/runtime/generated/parameters.md)
lists supported aliases. Profiles and deployment presets live in the same
[runtime directory](https://github.com/local-inference-lab/blackwell-llm-docker/tree/main/runtime).

## Performance evidence

The [Karmic Kraken model table](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/karmic-kraken-serving.md) records
stock-clock decode, prefill and cache checks. It also retains negative deltas;
not every model is faster than JJ. The
[JJ deployment archive](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/docs/unified-vllm-docker-jj-archive.md) preserves its separate
image identities and measurements. No benchmark is rerun merely to update this guide.
