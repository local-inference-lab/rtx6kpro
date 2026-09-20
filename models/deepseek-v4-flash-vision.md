# DeepSeek-V4-Flash Vision

Serve `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` using the `ds4-vision`
profile in the [shared vLLM Docker guide](../docs/unified-vllm-docker.md).
It shares the image and launcher with GLM, Qwen and DeepSeek text, but has
its own checkpoint, Vision defaults and fixed DSpark K3 configuration.

## Start the server

This starts TP2/DCP1 with DSpark K3:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name ds4-vision --init --restart unless-stopped \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v ds4-vision-runtime:/cache \
  -e PROFILE=ds4-vision -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=2 -e PORT=8000 "$IMAGE"
```

The API model is `DeepSeek-V4-Flash-Vision-Exp` on port 8000.
Change `device=0,1`, `-e TP=2` and `-e PORT=8000` to select the deployment.
Check readiness with `docker logs -f ds4-vision`.

For target-only serving append `--mode off` **after `"$IMAGE"`**.
Model and compiler caches stay in named volumes. The image profile selects
compatible model/code revisions; no absolute checkpoint path is required.
The [Karmic Kraken benchmark table](../benchmarks/karmic-kraken-serving.md)
records the image comparison and exact measurement settings.

## Common settings

The command uses TP2/DCP1 and DSpark K3. Put `-e` settings before the image
and native arguments after it:

| Setting | Default / recommendation | Example override |
|---|---|---|
| GPU count | TP2; expose two GPU IDs | `-e TP=4` with `device=0,1,2,3` |
| Active requests | 4 | `-e MAX_NUM_SEQS=8` if your memory budget allows |
| Prefill budget | 4096 tokens | `-e MAX_NUM_BATCHED_TOKENS=4096` |
| Context | Automatic, `-1` | `-e MAX_MODEL_LEN=131072` |
| GPU memory fraction | 0.975 | `-e GPU_MEMORY_UTILIZATION=0.95` for more image working space |
| Speculation | DSpark K3 | `--mode dspark --draft-tokens 3` after the image; `--mode off` disables it |

B12X attention and W4A8 MoE, FP8 CLI cache mode and prefix caching are enabled.
Keep the model-specific cache retention default. Sampling defaults are
temperature 1/top-p .95, thinking enabled and reasoning `high`. The speed test
below uses top-p 1 explicitly.

The profile imposes no artificial one/two-image cap. Image resolution, count
and context consume memory; the text prefill figure is not an image-encoding
benchmark. Reducing the GPU memory fraction leaves more temporary image space.

The profile intentionally leaves `--linear-backend` unspecified. This delegates
dense selection to the model/runtime; it is not evidence that every dense
operation uses a particular DeepGEMM kernel. Do not paste GLM dense settings
into this profile without a separate comparison.

GPU-only cache is the default. LMCache host-RAM and filesystem modes are
implemented as documented in the
[shared cache section](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload).
Text and image-prefix recovery, restart recovery and different-image isolation
are checked in the [Karmic Kraken cache record](../benchmarks/karmic-kraken-serving.md#prefix-cache-checks).
Native KV offload is unsupported by this profile.

## Measured performance

Two RTX PRO 6000 **Max-Q**, **VRAM +6000**, automatic graphics clocks;
TP2/DCP1, DSpark K3, 4096-token budget, four slots and FP8 GPU cache.
Five warmed 30-second windows per cell. Decode uses context zero and temperature
1/top-p **1**, not the profile's .95 default. C4 is aggregate, not C8.
32K prefill is uncached text input measured from client TTFT.

| Metric | Saved JJ R9 | Karmic Kraken | Change |
|---|---:|---:|---:|
| C1 output | 169.2 tok/s | 193.9 tok/s | +14.60% |
| C4 aggregate output | 394.7 tok/s | 401.4 tok/s | +1.68% |
| 32K text prefill | 8,993 tok/s | 9,182 tok/s | +2.10% |
| C1 verifier rate | 79.29 steps/s | 87.18 steps/s | +9.96% |

Text, image, repeated-prefix and changed-prefix checks pass. The KK server
reports 1,291,085 logical KV tokens with a 1,048,576 per-request context cap.
[Image versions, configuration and all samples](../benchmarks/karmic-kraken-serving.md).

## Related model and historical releases

- [DeepSeek V4 text](deepseek-v4-flash.md) uses the separate `ds4-flash` profile.
- [DeepSeek V4.1](deepseek-v4.1-flash.md) uses native Engram placement and is not
  selected by changing only the checkpoint in this Vision profile.
- [Community R9 text/Vision record](ds4-jovian-judgement-r9.md),
  [Vision R3 record](ds4-vision-jovian-judgement-r3.md) and
  [shared community-runtime record](ds4-jovian-community-r29.md) preserve their
  release-specific commands and qualification. They are not the unified image.
- [Archived recipes and measurements](../archive/serving-guides/README.md)
  preserve the preceding guides and stock Workstation figures.
- [Source review and limits](https://github.com/local-inference-lab/vllm/issues/808).
