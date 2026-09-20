# GLM-5.3-Flash on two 96-GB GPUs

> Historical recipe snapshot. Use the [model guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash-spark-tp2.md) for the recommended deployment. This snapshot pins the September 19 beta image; performance tables retain their original image and hardware identities. [Archive manifest](https://github.com/local-inference-lab/rtx6kpro/blob/master/archive/serving-guides/karmic-kraken-beta-20260919-cfc67a15ebc3daf7/manifest.json).

Run `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` with the shared Karmic
Kraken beta image. The Spark preset supplies TP2/DCP2, MTP3, B12X backends
and memory settings for two RTX PRO 6000 cards. “Spark” identifies this
checkpoint, not ARM-based DGX Spark hardware.

## Start the server

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7
docker pull "$IMAGE"
docker run -d --name glm-spark-tp2 --init --restart unless-stopped \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  --security-opt seccomp=unconfined \
  -v lil-huggingface:/root/.cache/huggingface \
  -v glm-spark-runtime:/cache \
  -e PRESET=glm53-spark-tp2 -e PORT=8000 \
  "$IMAGE"
```

Change `device=0,1` to your two GPU IDs. The checkpoint downloads into the
Hugging Face volume; no host model path is required. To use a saved HF token,
mount your existing HF cache or pass `--env HF_TOKEN` after exporting it.
The image does not change clocks. This CUDA 13.4.1 image needs a compatible
NVIDIA driver; native driver 615.71.09 is tested.

The API model name is `GLM-5.3-Flash` on port 8000:

```bash
docker logs -f glm-spark-tp2
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Keep the API on a trusted network or add authentication. First startup includes
model loading, kernel preparation and graph capture and can take several minutes.

## RAM and disk prefix cache

GPU-only prefix caching is enabled by default. For LMCache, add these arguments
**before `"$IMAGE"`** in the command above:

```bash
-e CACHE_MODE=lmcache \
-e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2 \
-e LMCACHE_L2_ENABLED=1 -e LMCACHE_L2_GB=64 \
-e LMCACHE_MAX_CPU_WORKERS=4 -e LMCACHE_MAX_GPU_WORKERS=2
```

This starts the CPU cache service automatically: up to 16 GiB RAM plus a
64-GiB disk tier in the persistent `/cache` volume. To use only RAM, also add
`-e LMCACHE_L2_ENABLED=0`. LMCache reserves `PORT + 10000`, `PORT + 10001`
and `PORT + 10002`. Use API ports at least three apart and separate runtime volumes
for independent cache instances.

Text prefixes restore from RAM and from disk after both services restart.
Image-bearing requests run normally, but their recurrent checkpoints
are not restored through external cache. CPU and disk caches reuse prior
requests; they do not enlarge the active context or GPU KV allocation.

For an edit near the end of a long user message, the server also keeps a
checkpoint before the final prefill chunk. It can restore that unchanged
prefix and recompute roughly one to two chunks. An edit before that checkpoint
still needs an earlier matching prefix or a full recomputation. This costs one
additional checkpoint bundle, about 3% context capacity in this preset.

With the CPU/disk settings above, a tested suffix edit in a 520k-token prompt
restored 516,096 tokens in 1.66 seconds instead of recomputing the full prompt
in 70.07 seconds. Restoration also passes after restarting both the server
and cache process. These are request-latency checks, separate from the
Workstation throughput measurements below.

## Capacity and controls

| Setting | Preset value | Change before the image name |
|---|---|---|
| GPUs / parallelism | TP2, DCP2 | Keep this pair for the tested memory configuration |
| Speculation | MTP3 | `-e MTP_DEPTH=0` disables speculation |
| Request slots | 4 | `-e MAX_NUM_SEQS=4` |
| Prefill budget | 3072 tokens | `-e MAX_NUM_BATCHED_TOKENS=3072` |
| KV allocation | 3996 MiB per GPU | `-e KV_CACHE_MEMORY_BYTES=3758096384` selects 3.5 GiB |
| Context capacity | About 897k with the LMCache configuration above | Startup reports the resolved limit for your cache mode |
| Vision | Enabled, no one-image admission cap | Image size/count still consume memory |
| Prefix policy | Request-boundary checkpoints | No retention-interval parameter needed |

Capacity is shared by requests and includes generated output. The preset has
a tight VRAM budget: long contexts together with large images can trigger
allocator retries. Lower the explicit KV allocation if your workload needs
more working memory. Increasing slots or changing speculation also changes
graph memory; do not assume the same capacity.

Target KV is FP8 and recurrent state is FP32. The target vocabulary head
remains BF16; MTP uses a private NVFP4 draft head. Full-and-piecewise CUDA
graphs cover 1, 2, 4, 8, 12 and 16 verifier rows. Two-shot all-reduce is off.

## Measured speed

Two RTX PRO 6000 **Workstation** cards at **stock clocks, without +6000 VRAM
overclock**; TP2/DCP2 MTP3, 3072-token prefill budget, four slots, temperature 1
and top-p .95. Decode uses one warmed 30-second measurement per concurrency;
prefill uses a 30-second window of uncached 32K requests. These are not medians
of repeated benchmark runs.

| Measurement | Speed |
|---|---:|
| C1 decode, context 0 | **186.5 output tok/s** |
| C4 decode, context 0 | **403.1 output tok/s total** |
| 32K prefill | **10,919 input tok/s** |

The separate Max-Q server validates startup and cache recovery, not the speeds
above. Source composition and test details are listed in
[vLLM issue #808](https://github.com/local-inference-lab/vllm/issues/808).
The [R2 archive](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash-spark-tp2-r2-archive.md) retains the separate
CUDA 13.3 release, its measurements and its one-image limit.

For four GPUs use the [GLM TP4 recipe](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash.md). For other models and
common controls see the [shared Docker guide](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/docs/unified-vllm-docker.md).
