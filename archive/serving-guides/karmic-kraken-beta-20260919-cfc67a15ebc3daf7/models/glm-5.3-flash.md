# GLM-5.3-Flash

> Historical recipe snapshot. Use the [model guide](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md) for the recommended deployment. This snapshot pins the September 19 beta image; performance tables retain their original image and hardware identities. [Archive manifest](https://github.com/local-inference-lab/rtx6kpro/blob/master/archive/serving-guides/karmic-kraken-beta-20260919-cfc67a15ebc3daf7/manifest.json).

<p align="center">
  <img src="https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/76595579cc04c245c13537583a93d525060b8ce0/images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke for Local Inference Lab.</em></p>

Serve `local-inference-lab/GLM-5.3-Flash-NVFP4` through the `glm53-flash`
profile in the [shared vLLM Docker guide](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/docs/unified-vllm-docker.md).
The same image serves Qwen and DeepSeek; a model-specific image or entrypoint
is not required. The shared guide owns the image tag, launch command, LMCache
configuration and general option reference.

## Start the server

This starts MTP3 on four GPUs with the shared Karmic Kraken beta image:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7
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
For two GPUs use the [Spark TP2 recipe](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash-spark-tp2.md); its
checkpoint and memory budget differ from this four-GPU configuration.

Measured Karmic Kraken results and saved JJ comparisons are in the
[model benchmark table](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/karmic-kraken-serving.md).
The separate JJ measurements below retain their original image boundaries.

## Serving defaults and alternatives

| Setting | Profile behavior |
|---|---|
| Parallelism | TP4/DCP1; four 96-GB GPUs in the measured configuration |
| Speculation when omitted | Off; the explicit examples select MTP3 or DFlash2 K7 |
| Target precision | ModelOpt NVFP4; B12X MoE and dense backends |
| Attention | B12X sparse attention/selection and B12X KDA prefill |
| MTP | B12X attention, Marlin draft MoE, private NVFP4 draft vocabulary head; BF16 target head |
| DFlash2 | Offline MXFP8 weights, B12X dense path, FLASH_ATTN draft attention, automatic draft KV dtype |
| Target KV | FP8; `--kv-cache-dtype nvfp4_ds_mla` is an explicit, separately unqualified option for this artifact |
| Graphs | Full-and-piecewise target/draft decode graphs, capture sizes through 256 |
| Scheduler | 4096 tokens, 32 sequences, one prefill lane, compute share 0.4 |
| Context / GPU fraction | 1,048,576 configured tokens / 0.93; configuration is not a million-token test |
| Prefix policy | `request_boundaries` with `mamba-cache-mode=align`; no manual retention interval needed |
| Sampling / reasoning | Temperature 1, top-p .95, reasoning `high`, `clear_thinking=false` |
| Vision | No artificial one/two-image profile cap; native encoder/context/memory limits remain |

Explicit request sampling and template options override their corresponding
server defaults. If replacing the complete template-default JSON, retain
`clear_thinking=false` when preserved assistant reasoning is required.

The [shared cache section](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload)
documents GPU-local, LMCache RAM/filesystem and native KV offload. LMCache is
opt-in; text prefixes support CPU restore and persistent restart recovery.
External recurrent restore does not apply to image-bearing requests.

`--decode-context-parallel-size 4` selects DCP4 and automatic full-CKV gather;
this is not specific to DFlash2. DCP4 and TP8 are **implemented**, but the
six-profile wheel comparison below qualifies DCP1, not those alternatives.
The [Spark TP2 recipe](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash-spark-tp2.md) has a different checkpoint
and capacity contract; it is not this four-GPU profile with TP changed to two.

## Measured performance

Stock RTX PRO 6000 Workstation quartet, TP4/DCP1, 4096-token budget, GPU-only
FP8 target cache, full-and-piecewise graphs, temperature 1/top-p .95. Decode:
context zero, medians of three warmed 30-second runs. Prefill: sustained
uncached nominal-32K requests, client time to first token. C8 is aggregate.
Image identities, runtime arguments and raw samples are in the
[wheel-image qualification](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/prepared-b12x-serving).

| Mode | C1 tok/s, R35 → wheel image | C8 tok/s, R35 → wheel image | 32K prefill tok/s, R35 → wheel image | Sieve tok/s, R35 → wheel image |
|---|---:|---:|---:|---:|
| MTP3 | 247.12 → 249.33 (+0.89%) | 872.67 → 875.86 (+0.37%) | 15,332 → 15,580 (+1.62%) | 326.15 → 325.75 (−0.12%) |
| DFlash2 K7 | 211.23 → 219.83 (+4.07%) | 673.26 → 717.84 (+6.62%) | 15,538 → 15,734 (+1.26%) | 458.14 → 461.42 (+0.72%) |

Sieve uses five measured requests and is not a coding-correctness evaluation.
No no-spec measurement exists in this wheel-image matrix. Historical no-spec,
DCP4 and +6000-clock values remain in the archive, not substituted here.

### Prepared B12X plan integration

A separate same-GPU component comparison qualifies the B12X master
reconciliation and vLLM #789, which retains GLM's prepared selection plan:

| Metric | Control → prepared-plan image | Change |
|---|---:|---:|
| MTP3 C8 output | 856.05 → 857.27 tok/s | +0.14% |
| MTP3 32K prefill | 15,486 → 15,531 tok/s | +0.29% |
| DFlash2 C1 output | 214.31 → 212.08 tok/s | −1.04% |
| DFlash2 C8 output | 695.88 → 708.64 tok/s | +1.83% |
| DFlash2 32K prefill | 15,702 → 15,702 tok/s | 0.00% |

DFlash C1 verifier throughput rises 0.22% while acceptance changes; negative
output deltas are retained rather than called zero regression. The MTP control
contains C8 only. Read the
[component receipts](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/benchmarks/prepared-b12x-contracts/#retained-pooled-selection-glm)
separately from the whole-image table. Two-shot is disabled in both comparisons.

Functional checks include arithmetic, repeated/changed prefix requests,
2/8/16 small images and image history. These are bounded checks, not general
language quality, arbitrary image resolution or million-token qualification.

## Quality evaluation and historical releases

Runtime throughput does not establish checkpoint quality. Retain the exact
runtime/checkpoint boundaries of these independent reports:

- [BF16, published NVFP4 and QAD AA-LCR comparison](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md)
  and [reproduction method](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash/aa-lcr-reproduction.md).
- [Verifier-backed behavioral fidelity](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash/verifier-backed-behavioral-fidelity.md),
  [QAD step-2500](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md)
  and [QAD TV-nucleus](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash/qad-tvn-step2500-verifier-backed-behavioral-fidelity.md).
- [BF16/NVFP4 distribution fidelity](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/kld/glm-5.3-flash-bf16-nvfp4.md)
  and [QAD quantization reports](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/kld/glm-5.3-flash-qad-step2500.md).
- [Community R35 deployment and measurement archive](https://github.com/local-inference-lab/rtx6kpro/blob/76595579cc04c245c13537583a93d525060b8ce0/models/glm-5.3-flash-community-r35.md):
  release-specific launchers, DCP and no-spec matrices, +6000 measurements,
  source locks, historical LMCache restores and reported constrained-output limits.

Source review and unresolved items: [issue #773](https://github.com/local-inference-lab/vllm/issues/773).
