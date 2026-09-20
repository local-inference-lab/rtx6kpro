# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke for Local Inference Lab.</em></p>

Serve `local-inference-lab/GLM-5.3-Flash-NVFP4` through the `glm53-flash`
profile in the [shared vLLM Docker guide](../docs/unified-vllm-docker.md).
The same image serves Qwen and DeepSeek; a model-specific image or entrypoint
is not required. The shared guide owns the image tag, launch command, LMCache
configuration and general option reference.

## Start the server

This starts MTP3 on four GPUs with the shared Karmic Kraken beta image:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name glm53 --init --restart unless-stopped \
  --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v glm53-runtime:/cache \
  -e PROFILE=glm53-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=4 -e PORT=8000 "$IMAGE" --mode mtp --draft-tokens 3
```

The API is on port 8000 with model name `GLM-5.3-Flash-NVFP4`.
Change `-e PORT=8000` to choose a port and `device=0,1,2,3` to choose GPUs.
Check readiness with `docker logs -f glm53` and
`curl -fsS http://127.0.0.1:8000/health`.

Replace the arguments **after `"$IMAGE"`** to select another mode:

| Mode | Arguments after the image |
|---|---|
| No speculation | `--mode off` |
| MTP3 | `--mode mtp --draft-tokens 3` |
| DFlash2 K7 | `--mode dflash2 --draft-tokens 7` |

DFlash2 downloads `local-inference-lab/GLM-5.3-Flash-DFlash2`, an offline
MXFP8 checkpoint, into the shared HF volume. Add
`-e SERVED_MODEL_NAME=GLM-5.3-Flash` before the image for that shorter API name.
For two GPUs use the [Spark TP2 recipe](glm-5.3-flash-spark-tp2.md); its
checkpoint and memory budget differ from this four-GPU configuration.

Measured Karmic Kraken results and saved JJ comparisons are in the
[model benchmark table](../benchmarks/karmic-kraken-serving.md).

## Common settings

Keep the profile defaults unless your workload needs a different capacity.
Place the following `-e` overrides **before `"$IMAGE"`**:

| Setting | Default / recommendation | Example override |
|---|---|---|
| GPUs | TP4; expose four GPU IDs | `-e TP=4` |
| Context parallelism | DCP1 for the speed table | `-e DCP=4` for full-CKV gather; DCP must divide TP |
| Active requests | 32 | `-e MAX_NUM_SEQS=16` |
| Prefill budget | 4096 tokens | `-e MAX_NUM_BATCHED_TOKENS=4096` |
| Per-request context limit | 1,048,576, subject to available KV | `-e MAX_MODEL_LEN=131072` |
| GPU memory fraction | 0.93 | `-e GPU_MEMORY_UTILIZATION=0.90` to leave more working space |
| External prefix storage | GPU-only | `-e CACHE_MODE=lmcache` with the RAM/disk settings below |

TP2 needs the [Spark checkpoint and preset](glm-5.3-flash-spark-tp2.md),
not just `-e TP=2` on this command. Changing slots or draft length can change
graph memory and KV capacity.

### Backend and request defaults

The target uses ModelOpt NVFP4, B12X attention, B12X KDA prefill, B12X MoE
and dense kernels, FP8 KV, and full-and-piecewise CUDA graphs. Two-shot
all-reduce is off. MTP uses Marlin draft MoE and a private NVFP4 vocabulary
head; the target head remains BF16. DFlash2 uses offline MXFP8 weights,
B12X dense kernels, FLASH_ATTN draft attention and automatic draft KV dtype.

GPU prefix caching uses `request_boundaries` with aligned recurrent states.
Keep this default: no manual `--prefix-cache-retention-interval` is needed.
Vision has no artificial one/two-image profile cap; image resolution and
context still consume memory.

Request defaults are temperature 1, top-p .95, reasoning `high` and
`clear_thinking=false`. For example, a request can select
`"chat_template_kwargs":{"reasoning_effort":"high"}`.

Explicit request sampling and template options override their corresponding
server defaults. If replacing the complete template-default JSON, retain
`clear_thinking=false` when preserved assistant reasoning is required.

The [shared cache section](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload)
documents GPU-local, LMCache RAM/filesystem and native KV offload. LMCache is
opt-in; text prefixes support CPU restore and persistent restart recovery.
External recurrent restore does not apply to image-bearing requests.

`-e DCP=4` selects DCP4 and automatic full-CKV gather in both MTP and DFlash2.
The speed table uses DCP1; it does not predict DCP4 speed. For an explicit
NVFP4 target-cache experiment, append `--kv-cache-dtype nvfp4_ds_mla` after
the image. That changes the numerical and memory configuration and is not
the FP8 measurement below.

## Measured performance

Four RTX PRO 6000 **Max-Q**, **VRAM +6000**, automatic graphics clocks;
TP4/DCP1, 4096-token budget, 32 slots, GPU-only FP8 target cache,
full-and-piecewise graphs and temperature 1/top-p .95. Five warmed
30-second windows per cell; context-zero decode and uncached nominal-32K
prefill measured from client TTFT. C8 is aggregate.

| Mode | C1 output | C8 aggregate output | 32K prefill | Change from saved R35: C1 / C8 / prefill |
|---|---:|---:|---:|---:|
| MTP3 | 281.3 tok/s | 897.6 tok/s | 13,816 tok/s | +12.99% / +2.72% / +2.23% |
| DFlash2 K7 | 229.0 tok/s | 702.6 tok/s | 13,955 tok/s | +4.22% / +3.67% / +0.83% |

The corresponding C1 verifier rates are 112.14 steps/s for MTP3 and 87.56
steps/s for DFlash2. Startup reports 5,595,903 and 5,816,930 logical KV tokens,
respectively, shared across requests. Both modes pass arithmetic and
repeated/changed-prefix checks.
[Exact images, settings and all samples](../benchmarks/karmic-kraken-serving.md).

Sieve, no-spec and DCP4 were not remeasured in this matrix. Their preceding
results, including stock Workstation measurements, remain in the
[versioned guide archive](../archive/serving-guides/README.md).

## Quality evaluation and historical releases

Runtime throughput does not establish checkpoint quality. Retain the exact
runtime/checkpoint boundaries of these independent reports:

- [BF16, published NVFP4 and QAD AA-LCR comparison](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md)
  and [reproduction method](glm-5.3-flash/aa-lcr-reproduction.md).
- [Verifier-backed behavioral fidelity](glm-5.3-flash/verifier-backed-behavioral-fidelity.md),
  [QAD step-2500](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md)
  and [QAD TV-nucleus](glm-5.3-flash/qad-tvn-step2500-verifier-backed-behavioral-fidelity.md).
- [BF16/NVFP4 distribution fidelity](../kld/glm-5.3-flash-bf16-nvfp4.md)
  and [QAD quantization reports](../kld/glm-5.3-flash-qad-step2500.md).
- [Community R35 deployment and measurement archive](glm-5.3-flash-community-r35.md):
  release-specific launchers, DCP and no-spec matrices, +6000 measurements,
  source locks, historical LMCache restores and reported constrained-output limits.

Source review and integration checklist: [issue #808](https://github.com/local-inference-lab/vllm/issues/808).
