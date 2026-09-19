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
records the shared-image comparison. The JJ results below are separate.

## Serving defaults and alternatives

| Setting | Profile behavior |
|---|---|
| Parallelism / speculation | TP2/DCP1, fixed DSpark K5, probabilistic proposals, standard rejection |
| Backends | B12X attention and W4A8 MoE; native dense selection |
| KV / prefix cache | FP8 compressed attention KV, prefix caching enabled, retention interval 4096 |
| Graphs | Full-and-piecewise, default graph cap 48; breakable prefill off |
| Scheduler | 4096 tokens, eight sequences |
| Context / GPU fraction | Native automatic context admission, `max-model-len=-1`; GPU fraction .975 |
| Sampling / reasoning | Temperature 1/top-p .95, thinking enabled, `high` |

The profile leaves `--linear-backend` unspecified; that is native selection,
not a guarantee that every projection uses a particular kernel. Do not copy
GLM's complete backend environment into this recipe.

DSpark and standard Multi-Token Prediction (MTP) use different checkpoint
contracts. The following standard-MTP alternative is **implemented**, but
not qualified by the DSpark measurements on this page. Replace the serving
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

Same stock RTX PRO 6000 Workstation pair, TP2/DCP1, fixed K5, 4096-token
budget, eight sequences, FP8 KV, GPU-only cache and configured context limit
1,048,576. Decode uses temperature 1/top-p **1**, not the profile's .95 default,
with three warmed 30-second context-zero runs. C8 is aggregate.
Prefill is uncached nominal 32K, measured from client time to first token.

| Metric | Community R9 → wheel image | Change |
|---|---:|---:|
| C1 output | 191.35 → 190.12 tok/s | −0.64% |
| C8 output | 653.53 → 669.56 tok/s | +2.45% |
| 32K prefill | 13,527 → 13,863 tok/s | +2.48% |
| Logical KV tokens | 1,192,983 → 1,293,619 | +8.44% |

All four API checks and six decode cells pass. C1 request-verifier throughput
rises 1.92%, while accepted length changes from 2.662 to 2.593; the small
output decrease is retained, not called a regression-free result. Configuring
a million-token limit is not a million-token request test.
[Exact image identities, parameters and raw samples](../benchmarks/prepared-b12x-serving/).

## Historical deployment and measurement records

These pages retain their image-specific launchers, source locks and results.
Their environment variables are not a second configuration source for the
unified image.

| Need | Record |
|---|---|
| Text K5 / Vision K3, GPU KV and text LMCache restore | [Jovian Judgement R9](ds4-jovian-judgement-r9.md) |
| Shared community GLM/Qwen/DS4 image | [Shared community-runtime record](ds4-jovian-community-r29.md) |
| Qualified Infernal source composition | [Infernal Invocation R21](ds4dspark-infernal-invocation-r21.md) |
| Target-only 2.11M-KV capacity study | [Infernal Invocation R19](ds4dspark-infernal-invocation-r19.md) |
| 0731 checkpoint deployment | [Infernal Invocation R18](ds4dspark-infernal-invocation-r18.md) |
| Topology-calibrated transport | [B12X PCIe transport calibration](ds4f-b12x-pcie-autotune.md) |
| Gilded source composition | [Gilded Gnosis R33](ds4dspark-v20-r33.md) |
| Fathomless TP2/TP4 sweep | [Fathomless validation](ds4dspark-v10.md) |
| DSpark and standard-MTP sweep | [DSpark/MTP reference](ds4dspark-v9.md) |
| Empty reasoning before tool calls | [Troubleshooting](ds4f-empty-think/README.md) |

### Release namespace map

| Source line | Revision namespace | Serving specification |
|---|---|---|
| `dev/jovian-judgement` | Jovian Judgement `r*` | [DS4 text and Vision r9](ds4-jovian-judgement-r9.md) |
| `dev/infernal-invocation` | Infernal Invocation `r*` | [r21 qualified source composition](ds4dspark-infernal-invocation-r21.md), [r19 capacity study](ds4dspark-infernal-invocation-r19.md) |
| `dev/gilded-gnosis` | Gilded Gnosis `v20-r*` | [Gilded Gnosis r33](ds4dspark-v20-r33.md) |
| Fathomless Firmament | `v9` and `v10` | [v10](ds4dspark-v10.md), [v9](ds4dspark-v9.md) |
| Eldritch Enlightenment | DS4 Flash `v1-v6` | [v6](ds4-flash-v6.md), [v5](ds4-flash-v5.md), [v4](ds4-flash-v4.md), [v3](ds4-flash-v3.md), [v2](ds4-flash-v2.md), [v1](ds4-flash-v1.md) |

Revision numbers belong to their source line, not to one global sequence.
Source review and unresolved items:
[issue #773](https://github.com/local-inference-lab/vllm/issues/773).
