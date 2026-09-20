# Qwen3.8-Flash-Next

> Historical recipe snapshot. Use the [model guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/qwen38-flash-next.md) for the recommended deployment. This snapshot pins the September 19 beta image; performance tables retain their original image and hardware identities. [Archive manifest](https://github.com/local-inference-lab/rtx6kpro/blob/master/archive/serving-guides/karmic-kraken-beta-20260919-cfc67a15ebc3daf7/manifest.json).

Serve `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` using the
`qwen38-flash-next` profile in the [shared Docker guide](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/docs/unified-vllm-docker.md).
The image and launcher are shared with GLM and DeepSeek; no GLM entrypoint
bypass or copied kernel environment is needed. This is not
[Qwen3.8-27B](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/qwen38-27b.md).

## Start on one GPU: TP1

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7
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
The [Compose example](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/qwen38-flash-next/qwen38-flash-next.compose.yml)
provides TP1 and TP2 services using these same image profiles:

```bash
export LIL_IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7
curl -fLO https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/qwen38-flash-next/qwen38-flash-next.compose.yml
GPU=0 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp1 up -d
```

For two GPUs select `--profile tp2` and set `GPU0`/`GPU1`.
The [Karmic Kraken benchmark table](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/karmic-kraken-serving.md)
records the TP1 comparison; the JJ results below remain a separate measurement.

## Precision, model tables and cache

| Setting | Profile behavior |
|---|---|
| Weight format | Mixed ModelOpt NVFP4; the repository name does not mean every tensor is four-bit |
| PLE n-gram tables | CPU-mapped host RAM, `VLLM_PLE_CPU_OFFLOAD=1` |
| Vocabulary projections | BF16 target; private NVFP4 MTP head with BF16 activations |
| Kernels | B12X MoE, dense and GDN decode; native Qwen attention selection |
| KV / recurrent state | FP8 attention KV; recurrent state follows the native model contract |
| Runner / graphs | V2, full-and-piecewise, graph cap 64 |
| Scheduler / context | 6019 tokens, 16 sequences, maximum context 262,144 |
| Prefix cache | Enabled; native `auto` selects exact recurrent request boundaries where supported |
| Vision | Text-only by default; `--no-language-model-only` enables the model path |
| LMCache | Optional CPU/disk restore for text; separate from PLE offload |

PLE is a learned embedding table, not request KV and not an n-gram speculator.
Historical startup accounting records about 26.82 GiB of mapped host tables;
leave additional host RAM for loading and the server. Keep offload enabled for
the one-96-GB-GPU recipe. A device-resident PLE alternative requires a separate
memory and correctness qualification.

The shared guide explains [prefix retention](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/docs/unified-vllm-docker.md#prefix-cache-defaults).
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

Stock RTX PRO 6000 Workstation, TP1/MTP3, CPU PLE, FP8 KV, 6019-token budget,
16 sequences, explicit eight-GiB KV allocation, context-zero decode and
temperature 1/top-p .95/top-k 20. Three warmed 30-second decode runs; uncached
nominal-32K prefill uses client time to first token. C8 is aggregate.

| Metric | Community R35 → wheel image | Change |
|---|---:|---:|
| C1 output | 172.92 → 190.11 tok/s | +9.94% |
| C8 output | 664.89 → 689.93 tok/s | +3.77% |
| 32K prefill, one sustained window | 15,387 → 15,162 tok/s | −1.46% |
| C1 request-verifier rate | 81.90 → 83.47 steps/s | +1.92% |

All four API checks and six decode cells pass. Acceptance changes from 2.105
to 2.280, so the C1 output gain is not an isolated kernel speedup.
[Exact image boundary, commands and raw samples](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/prepared-b12x-serving).

A separate three-window prefill repeat on the indexed-PCIe-plan integration
image measures R35 **15,258** versus **15,031 tok/s**, a **−1.49%** median
difference. The windows are independent warmed measurements within one
startup per image, not three independent startups. The small gap remains
**unresolved**, without assigning it to a particular kernel or PR.
[Repeat conditions and receipts](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/prepared-b12x-contracts/#qwen-prefill-repeat).

The eight-GiB comparison has 517,581 logical KV tokens. A separate automatic
KV-sizing startup passes graph capture and API checks with 773,216 tokens;
it is not another throughput measurement. These are shared-pool capacities,
not the context limit of an individual request. Use engine-reported logical
capacity rather than physical-block count times page size.

## Quality evaluation and historical releases

- [Published NVFP4 versus QAD AA-LCR](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/qwen38-flash-next/aa-lcr-nvfp4-vs-qad.md).
- [Direct-answer arithmetic stability](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/qwen38-flash-next/direct-arithmetic-stability-nvfp4-vs-qad.md).
- [Community R35 deployment and measurement archive](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/qwen38-flash-next-community-r35.md):
  +6000-clock results, SGLang/Sieve comparisons, older capacity measurements
  and exact recipe boundaries. These are not measurements of the wheel image.

No Qwen Sieve rerun or TP2 speed is claimed for the wheel comparison.
Source review and unresolved items: [issue #773](https://github.com/local-inference-lab/vllm/issues/773).
