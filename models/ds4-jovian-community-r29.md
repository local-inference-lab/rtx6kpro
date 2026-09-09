# DeepSeek V4 on the shared Jovian Judgement image

The shared Docker image serves GLM-5.3-Flash, Qwen3.8-Flash-Next and DeepSeek
V4 using one installed vLLM/B12X/LMCache source composition. Each model uses
its own launcher profile; GLM precision, scheduler and cache settings are not
substituted for the DS4 profile.

```text
localinferencelab/vllm:jovian-judgement-community-20260909-r31
```

Status: **qualified for the bounded TP2/DCP1 FP8 checks below**.
The source includes the clustered BF16-router barrier correction and immutable
LMCache gather metadata. Two concurrent Vision/LMCache tests of 600 seconds
complete without the previously reproduced launch failure. This does not
qualify long-duration filesystem-pressure behavior. Other TP/DCP topologies
and NVFP4 target KV are not qualified by these checks.

The bounded GPU evidence below is from R29. R31 retains those model/native
components and the launcher sampling defaults of temperature 1/top-p 0.95,
with request and native-CLI overrides. DS4 inference is not rerun for R31.
See the [R31 changelog and qualification scope](glm-5.3-flash/validation/warmup-retention-r31.md).

## Start text or Vision

Choose two available 96 GiB RTX PRO 6000 Blackwell GPUs. The example uses
model names and named Docker volumes; no source or absolute model-path mounts
are required. It downloads weights on first launch. Docker needs NVIDIA
Container Toolkit and a CUDA 13.3-compatible driver.

```bash
IMAGE=localinferencelab/vllm:jovian-judgement-community-20260909-r31
GPU_DEVICES=0,1
PORT=8000
VARIANT=text
NAME=ds4-jovian-text
docker pull "$IMAGE"
docker run -d --name "$NAME" --init \
  --gpus "\"device=${GPU_DEVICES}\"" --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ds4-model-cache:/root/.cache/huggingface \
  -v ds4-r30-runtime-cache:/cache \
  -e DS4_MODEL_VARIANT="$VARIANT" -e TP_SIZE=2 -e DCP_SIZE=1 \
  -e PORT="$PORT" -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e SERVED_MODEL_NAME=DeepSeek-V4-Flash \
  -e LMCACHE_MODE=off \
  --entrypoint /usr/local/bin/serve-ds4-jovian.sh "$IMAGE"
docker logs -f "$NAME"
```

For Vision, set `VARIANT=vision` and `NAME=ds4-jovian-vision` before running
the command. Stop the text container first if reusing its GPUs or API port.
The entrypoint selects the corresponding Hugging Face repository automatically.
An explicit `MODEL` overrides it; `MODEL_REVISION` pins a revision when needed.

| Profile | Text | Vision |
|---|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` |
| DSpark draft tokens | Fixed K5 | Fixed K3 |
| Sampling / rejection | Probabilistic / standard | Probabilistic / standard |
| Request slots | 8 | 4 |
| CUDA graph cap | 48 rows | 16 rows |
| TP / DCP | 2 / 1 | 2 / 1 |
| Scheduler budget | 4096 tokens | 4096 tokens |

Both profiles select B12X attention and W4A8 mixture-of-experts kernels, with
DeepGEMM dense projections (`BACKEND=b12x-a8-dglin`). `BACKEND=b12x-a8`
selects B12X dense projections for independent testing; the table below does
not measure that alternative. Loading uses buffered InstantTensor. Graph mode
is `FULL_AND_PIECEWISE`; OMP uses two threads. DS4-specific graph overrides
are `DS4_MAX_CUDAGRAPH_CAPTURE_SIZE` and `DS4_CUDAGRAPH_CAPTURE_SIZES`.

The API model name is `DeepSeek-V4-Flash`. Host networking exposes the API
without authentication; use a trusted network or authenticated proxy.
No example changes GPU clocks.

## Optional RAM and filesystem caching

Replace `-e LMCACHE_MODE=off` with `-e LMCACHE_MODE=ram` or `disk`.
The default L1 capacity is 24 GiB of host shared memory, configurable with
`LMCACHE_L1_GB`. Disk mode also uses `LMCACHE_L2_GB` and `LMCACHE_L2_PATH`;
put the path under the persistent `/cache` volume. For example:

```bash
# Docker environment arguments before the image name:
-e LMCACHE_MODE=disk -e LMCACHE_L1_GB=24 \
-e LMCACHE_L2_GB=256 -e LMCACHE_L2_PATH=/cache/lmcache/ds4
```

Transfers use asynchronous pinned shared memory owned by vLLM workers. The
LMCache sidecar is CPU-only and does not create its own CUDA context. Keep
enough host RAM for the L1 pool, transfer metadata, weights and other services.
For simultaneous model services, choose distinct cache names, directories and
`LMCACHE_PORT`/`LMCACHE_HTTP_PORT` values.

Start with an empty external-cache volume or L2 directory. The R29 LMCache
source fixes asynchronous reuse of pinned block-ID metadata, which could export
bytes from the wrong GPU pages. The correction cannot repair previously written
payloads, and DS4 filesystem keys do not automatically reject them on an image
update. The example therefore uses a separate runtime-cache volume. Preserve
existing volumes until their owner chooses to remove them.

GPU-only utilization defaults to 0.975. Engine-driven TP2 with a model limit
of at least one million tokens uses 0.970. These are measured 96 GiB GPU
profiles, not guarantees for different hardware. Native filesystem offload is
not the supported external-cache path for these profiles.

## Measured performance

Two stock-clock RTX PRO 6000 Blackwell Workstation GPUs, physical GPUs0/1,
temperature 1, warmed C1 context-zero decode, 30-second measured windows,
nominal 32K cold prefill with 12 samples per image. Prefill is input tokens
divided by client time to first token, not an isolated attention-kernel timer.
The control is the published [DS4 r9](ds4-jovian-judgement-r9.md).

| Profile | C1 output, control → shared tok/s | Verifier, control → shared steps/s | 32K prefill, control → shared tok/s |
|---|---:|---:|---:|
| Text K5 | 201.62 → 212.30 (+5.30%) | 72.49 → 75.59 (+4.28%) | 14,028 → 14,220 (+1.37%) |
| Vision K3 | 173.01 → 170.40 (−1.51%) | 80.54 → 80.47 (−0.09%) | 11,027 → 11,074 (+0.43%) |

The text row uses the return control. Its first control measured 190.18 tok/s
and 70.94 steps/s; that variation prevents treating the single candidate cell
as a proven speedup. These checks establish runnable profiles and bounded
performance observations, not a universal gain. Vision output varies with
speculative acceptance even when verifier rate is almost unchanged.

These performance cells use the shared DS4 integration before the independent
LMCache metadata and BF16-router barrier corrections. They are not an exact
performance retest of the final image. Final-image evidence is the 136-test
router suite and the concurrent Vision/LMCache serving check below.

## Cache and model checks

Text and Vision pass an 810K-token admission test, ordinary answers, tool/image
smokes, and GPU-prefix continuations. RAM and restart-filesystem checks compare
the actual copied bytes of all eight cache groups before model computation.
The replicated MLA cache is stored by rank0 and restored by both TP ranks.

| Profile / tier | External tokens restored | Local prefix hit | Local tail computed | Byte equality / literal answer | Elapsed |
|---|---:|---:|---:|---|---:|
| Text / RAM | 32,768 | 0 | 3,349 | Pass / exact | 0.74 s |
| Text / filesystem after both services restart | 32,768 | 0 | 3,349 | Pass / exact | 1.93 s |
| Vision / RAM | 32,768 | 0 | 3,458 | Pass / exact | 1.34 s |
| Vision / filesystem after both services restart | 32,768 | 0 | 3,458 | Pass / exact | 1.00 s |

Elapsed time includes byte-hashing diagnostics and answer generation. Filesystem
pages may remain in the OS page cache; these are not cold-disk bandwidth
measurements. All sidecars remain CPU-only.

A fixed-token logprob probe also showed cold/warm differences with GPU-only
caching, without LMCache. Those failed exact-logprob observations are retained;
they are not used as a cache-copy oracle or hidden by relaxing its tolerance.
Byte equality specifically qualifies transfer integrity, not arbitrary model
numerical determinism or answer quality.

### Concurrent Vision and cache transfers

The supplied four-client, four-image streaming workload reproduced two
`CTA Not Present` failures before the router correction, after 259 and 55
seconds. The BF16 router contained consumer-only whole-block barriers and
incomplete cluster shared-memory lifetime synchronization. The correction in
[vLLM #723](https://github.com/local-inference-lab/vllm/pull/723) gives
consumer reductions their own barrier and keeps all peer CTAs alive through
the remote writes. This is separate from the LMCache metadata defect.

With both corrections, two 600-second runs complete 529 and 501 HTTP 200
requests, with zero HTTP errors and a healthy engine. The 501-request run uses
the baked image without source mounts, exception-wait instrumentation or
overclocking. GPU memory transfers remain asynchronous and CUDA graphs remain
enabled. The supplied client does not independently validate SSE error events
or model-answer quality; this is a bounded serving check, not an unrestricted
stability guarantee. The [qualification report](glm-5.3-flash/validation/shared-serving-r29.md)
also retains the unresolved sanitizer diagnostic and its CUDA-only reproducer.

## Source and qualification evidence

The image has two filesystem layers, complete committed component histories,
and an embedded `/opt/glm53-flash/source.lock`. The
[R31 artifact report](glm-5.3-flash/validation/warmup-retention-r31.md) records
the deployed source identity. The historical
[R29 qualification](glm-5.3-flash/validation/shared-serving-r29.md) records
the GLM/Qwen/DS4 integration, immutable identities, test counts and limits.
The [source-locked recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/codex/glm53-source-locked-build/recipes/glm53)
builds the same model profiles without chaining community images.
