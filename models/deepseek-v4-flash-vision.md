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
records the shared-image comparison. The JJ results below are separate.

## Serving defaults and cache

| Setting | Profile behavior |
|---|---|
| Parallelism / speculation | TP2/DCP1, fixed DSpark K3, probabilistic proposals, standard rejection |
| Backends | B12X attention and W4A8 MoE; native dense selection |
| KV / prefix cache | FP8 CLI mode, prefix cache enabled, retention interval 4096 |
| Graphs | Full-and-piecewise; default graph cap 16 |
| Scheduler | 4096 tokens, four sequences |
| Context / GPU fraction | Native automatic context admission, `max-model-len=-1`; GPU fraction .975 |
| Sampling / reasoning | Temperature 1/top-p .95, thinking enabled, `high` |
| Images | No artificial image1/image2 cap in the profile; encoder and memory limits remain |

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

Same stock RTX PRO 6000 Workstation pair, TP2/DCP1, fixed K3, 4096-token
budget, four sequences, FP8 KV, GPU-only cache and configured context limit
1,048,576. Decode uses temperature 1/top-p **1**, not the profile's .95 default,
with three warmed 30-second context-zero runs. C4 is aggregate; it is not C8.
32K prefill is uncached text input measured from client TTFT.

| Metric | Community R9 → wheel image | Change |
|---|---:|---:|
| C1 output | 169.92 → 185.49 tok/s | +9.17% |
| C4 output | 411.49 → 423.11 tok/s | +2.83% |
| 32K text prefill | 10,388 → 10,615 tok/s | +2.19% |
| Logical KV tokens | 1,215,925 → 1,285,174 | +5.70% |

Five API checks including image input and all six decode cells pass. C1 ranges
are 165.67–174.26 versus 185.04–192.78 tok/s. Results are bounded serving
measurements, not a model-quality score or an actual million-token request.
[Image identities, parameters and raw samples](../benchmarks/prepared-b12x-serving/).

## Related model and historical releases

- [DeepSeek V4 text](deepseek-v4-flash.md) uses the separate `ds4-flash` profile.
- [DeepSeek V4.1](deepseek-v4.1-flash.md) uses native Engram placement and is not
  selected by changing only the checkpoint in this Vision profile.
- [Community R9 text/Vision record](ds4-jovian-judgement-r9.md),
  [Vision R3 record](ds4-vision-jovian-judgement-r3.md) and
  [shared community-runtime record](ds4-jovian-community-r29.md) preserve their
  release-specific commands and qualification. They are not the unified image.
- [Source review and limits](https://github.com/local-inference-lab/vllm/issues/773).
