# DeepSeek-V4-Flash

Serve `deepseek-ai/DeepSeek-V4-Flash-0731` using the `ds4-flash` profile in
the [shared vLLM Docker guide](../docs/unified-vllm-docker.md). The image is
shared with GLM, Qwen and DeepSeek Vision; its model profile selects the
DSpark and B12X defaults.

## Start the server

This starts TP2/DCP1 with DSpark K5:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name ds4 --init --restart unless-stopped \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v ds4-runtime:/cache \
  -e PROFILE=ds4-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=2 -e PORT=8000 "$IMAGE"
```

The API model is `DeepSeek-V4-Flash-0731` on port 8000.
Change `device=0,1`, `-e TP=2` and `-e PORT=8000` to select the deployment.
Check readiness with `docker logs -f ds4`.

For target-only serving append `--mode off` **after `"$IMAGE"`**.
Model and compiler caches stay in named volumes. The image profile selects
compatible model/code revisions; no absolute checkpoint path is required.
The [Karmic Kraken benchmark table](../benchmarks/karmic-kraken-serving.md)
records the image comparison and exact measurement settings.

## Common settings

The command uses TP2/DCP1, DSpark K5, probabilistic proposals and standard
rejection. Put `-e` settings before the image and native arguments after it:

| Setting | Default / recommendation | Example override |
|---|---|---|
| GPU count | TP2; expose two GPU IDs | `-e TP=4` with `device=0,1,2,3` |
| Active requests | 8 | `-e MAX_NUM_SEQS=16` |
| Prefill budget | 4096 tokens | `-e MAX_NUM_BATCHED_TOKENS=4096` |
| Context | Automatic, `-1` | `-e MAX_MODEL_LEN=131072` |
| GPU memory fraction | 0.975 | `-e GPU_MEMORY_UTILIZATION=0.95` for more working space |
| Speculation | DSpark K5 | `--mode dspark --draft-tokens 5` after the image; `--mode off` disables it |

The profile selects B12X attention, W4A8 MoE and FP8 compressed attention KV.
Prefix caching is on; keep its model-specific retention setting. Breakable
prefill is off. Sampling defaults are temperature 1/top-p .95 with thinking
enabled and reasoning `high`. The speed test below uses top-p 1 explicitly.

The profile leaves `--linear-backend` unspecified; that is native selection,
not a guarantee that every projection uses a particular kernel. Do not copy
GLM's complete backend environment into this recipe.

DSpark and standard Multi-Token Prediction (MTP) use different checkpoint
contracts. The following standard-MTP alternative is not covered by the DSpark
speed measurements on this page. Replace the serving
command's final line with:

```bash
"$IMAGE" --model deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash --mode mtp --draft-tokens 3
```

GPU-only KV is the default. LMCache RAM/filesystem offload is opt-in through
the [shared cache controls](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload).
CPU and disk-prefix recovery have whole-model checks in the
[Karmic Kraken cache record](../benchmarks/karmic-kraken-serving.md#prefix-cache-checks).
Native KV offload is unsupported by this profile. DeepSeek V4.1's Engram RAM/SSD
controls are unrelated to it.

## Measured performance

Two RTX PRO 6000 **Max-Q**, **VRAM +6000**, automatic graphics clocks;
TP2/DCP1, DSpark K5, 4096-token budget, eight slots and FP8 GPU cache.
Five warmed 30-second windows per cell. Decode uses context zero and temperature
1/top-p **1**, not the profile's .95 default. C8 is aggregate. Prefill is
uncached nominal 32K measured from client time to first token.

| Metric | Saved JJ R9 | Karmic Kraken | Change |
|---|---:|---:|---:|
| C1 output | 214.9 tok/s | 212.5 tok/s | −1.09% |
| C8 aggregate output | 611.9 tok/s | 667.5 tok/s | +9.08% |
| 32K prefill | 11,167 tok/s | 11,392 tok/s | +2.01% |
| C1 verifier rate | 73.24 steps/s | 75.45 steps/s | +3.02% |

C1 output includes variable draft acceptance: median accepted length is
2.944 versus 2.815, despite faster verification. The KK run reports 1,301,500
logical KV tokens with a 1,048,576 per-request context cap. That is cache
capacity, not a million-token performance test.
[Image versions, configuration and all samples](../benchmarks/karmic-kraken-serving.md).

## Historical deployment and measurement records

- [Archived recipes and measurements](../archive/serving-guides/README.md)
  retain complete preceding guides, including stock Workstation results.
- [JJ R9](ds4-jovian-judgement-r9.md) contains the release-specific comparison
  recipe. Its environment variables are not shared-image defaults.
- [Empty reasoning before tool calls](ds4f-empty-think/README.md) and
  [PCIe transport calibration](ds4f-b12x-pcie-autotune.md).
- [Source review](https://github.com/local-inference-lab/vllm/issues/808).
