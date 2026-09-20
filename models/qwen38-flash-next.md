# Qwen3.8-Flash-Next

Serve `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` using the
`qwen38-flash-next` profile in the [shared Docker guide](../docs/unified-vllm-docker.md).
The image and launcher are shared with GLM and DeepSeek; no GLM entrypoint
bypass or copied kernel environment is needed. This is not
[Qwen3.8-27B](qwen38-27b.md).

## Start on one GPU: TP1

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name qwen38 --init --restart unless-stopped \
  --gpus '"device=0"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v qwen38-runtime:/cache \
  -e PROFILE=qwen38-flash-next -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=1 -e PORT=8000 "$IMAGE"
```

The profile selects MTP3 and CPU PLE tables. The API model is
`Qwen3.8-Flash-Next` on port 8000. It downloads the checkpoint by name and
does not change GPU clocks. Check startup with `docker logs -f qwen38`.

## Start on two GPUs: TP2

Use the same command with `--gpus '"device=0,1"'` and `-e TP=2`.
Change the container name or stop the overlapping instance before starting it.
The checkpoint and PLE placement stay the same.
Keep DCP at 1: Qwen's QSA attention backend rejects context parallelism.
TP2 splits model weights across two GPUs; it does not require DCP2.
TP2 with MTP3 and vision passes text/image checks and text-prefix recovery
from RAM and disk after restart. These are functionality checks; the measured
speed table below uses TP1.

Optional arguments go **after `"$IMAGE"`**:

| Choice | Arguments |
|---|---|
| MTP3 | Default, or `--mode mtp --draft-tokens 3` |
| No speculation | `--mode off` |
| Vision | `--no-language-model-only` |
| Eight-GiB KV budget used by the comparison | `--kv-cache-memory-bytes 8589934592` |

Use `-e PORT=8001` before the image to change the API port.
The [Compose example](qwen38-flash-next/qwen38-flash-next.compose.yml)
provides TP1 and TP2 services using these same image profiles:

```bash
export LIL_IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
curl -fLO https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/qwen38-flash-next/qwen38-flash-next.compose.yml
GPU=0 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp1 up -d
```

For two GPUs select `--profile tp2` and set `GPU0`/`GPU1`.
The [Karmic Kraken benchmark table](../benchmarks/karmic-kraken-serving.md)
records the TP1 comparison against saved JJ measurements.

## Precision, model tables and cache

Put these `-e` overrides **before `"$IMAGE"`**:

| Setting | Default / recommendation | Example override |
|---|---|---|
| Active requests | 16 | `-e MAX_NUM_SEQS=8` |
| Prefill budget | 6019 tokens | `-e MAX_NUM_BATCHED_TOKENS=4096` |
| Context limit | 262,144 | `-e MAX_MODEL_LEN=131072` |
| GPU memory fraction | 0.96 | `-e GPU_MEMORY_UTILIZATION=0.93` to reserve more working space |
| PLE table placement | Host RAM | `-e VLLM_PLE_TABLE_MEMORY=disk` for native disk loading |
| Prefix storage | GPU-only | `-e CACHE_MODE=lmcache` with [RAM/disk controls](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload) |

The checkpoint uses mixed ModelOpt NVFP4, not four-bit storage for every
tensor. The target vocabulary head stays BF16; MTP has a private NVFP4 head
with BF16 activations. B12X handles MoE, dense kernels and GDN decode, with
native Qwen attention. The V2 runner uses full-and-piecewise graphs.

Attention KV is FP8, while recurrent state follows the model's native contract.
Prefix caching is on; the native `auto` policy selects exact recurrent request
boundaries where supported. Leave this policy to the model profile.

PLE is a learned embedding table, not request KV and not an n-gram speculator.
Historical startup accounting records about 26.82 GiB of mapped host tables;
leave additional host RAM for loading and the server. Keep offload enabled for
the one-96-GB-GPU recipe. Disk mode requires fast local storage and still uses
host working memory; the performance table uses RAM, not disk placement.

The shared guide explains [prefix retention](../docs/unified-vllm-docker.md#prefix-cache-defaults).
Do not add a global `--prefix-cache-retention-interval 4096` override.
For vision, append `--no-language-model-only` after the image name. Image-bearing
requests do not restore recurrent checkpoints through external LMCache; their
uncached vision path remains available.

Request sampling in the recorded tests is temperature 1/top-p .95/top-k 20.
The profile leaves checkpoint generation configuration authoritative rather
than claiming all checkpoint revisions have identical server defaults.
For a non-thinking request, use
`"chat_template_kwargs":{"enable_thinking":false}`; it is a different
workload from a reasoning benchmark.

## Measured performance

One RTX PRO 6000 **Max-Q**, **VRAM +6000**, automatic graphics clocks;
TP1/MTP3, CPU PLE, FP8 KV, 6019-token budget, 16 slots and explicit eight-GiB
KV allocation. Temperature 1/top-p .95/top-k 20; five warmed 30-second
windows per cell. Decode uses context zero; uncached nominal-32K prefill
uses client TTFT. C8 is aggregate.

| Metric | Saved JJ R35 | Karmic Kraken | Change |
|---|---:|---:|---:|
| C1 output | 157.6 tok/s | 173.2 tok/s | +9.91% |
| C8 aggregate output | 674.2 tok/s | 698.4 tok/s | +3.58% |
| 32K prefill | 12,104 tok/s | 12,073 tok/s | −0.26% |
| C1 verifier rate | 76.06 steps/s | 81.78 steps/s | +7.52% |

Arithmetic and repeated/changed-prefix checks pass. Output includes draft
acceptance as well as execution speed.
[Exact image versions, configuration, repeats and samples](../benchmarks/karmic-kraken-serving.md).

A separate five-run restart with the same image and prepared cache measures
178.5 C1 and 713.7 C8 tok/s. Both startup series are retained in the report.

The eight-GiB allocation reports **434,258 logical KV tokens**, shared across
requests. It now includes safe recurrent endpoint and restore reservations;
the R35 report's 517,581-token estimate omitted those reserves. The physical
eight-GiB budget did not shrink. See the
[capacity accounting](../benchmarks/qwen-boundary-capacity-accounting.md).

### TP4 memory option

For TP4/MTP3, use four GPU IDs and `-e TP=4` in the launch command. The
hardware profile's default is sixteen NCCL channels. A measured alternative
is to add `-e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2` before `"$IMAGE"`.
It reduced sampled device memory by about 0.8–1.2 GiB per GPU and improved
decode in both TP4 test series. The gain varies between server starts;
the [TP4 results](../benchmarks/tp4-memory-controls.md#qwen-tp4-confirmations)
retain both series. This option was not measured at TP1/TP2 and should not
be copied into every model's configuration.

## Quality evaluation and historical releases

- [Published NVFP4 versus QAD AA-LCR](qwen38-flash-next/aa-lcr-nvfp4-vs-qad.md).
- [Direct-answer arithmetic stability](qwen38-flash-next/direct-arithmetic-stability-nvfp4-vs-qad.md).
- [Community R35 deployment and measurement archive](qwen38-flash-next-community-r35.md):
  +6000-clock results, SGLang/Sieve comparisons, older capacity measurements
  and exact recipe boundaries.
- [Versioned guide archive](../archive/serving-guides/README.md): preceding
  stock Workstation tables and complete launch instructions.

Sieve and TP2 speed were not remeasured in the Max-Q matrix.
Source review and integration checklist: [issue #808](https://github.com/local-inference-lab/vllm/issues/808).
