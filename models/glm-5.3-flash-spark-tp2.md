# GLM-5.3-Flash Spark on two GPUs

Release status: **research-only**, intended for experimental community testing.
This deployment targets two 96 GiB NVIDIA RTX PRO
6000 Blackwell Workstation GPUs with Tensor Parallelism 2 (TP2) and Decode
Context Parallelism 2 (DCP2). It uses the Spark checkpoint, three-token
Multi-Token Prediction (MTP3), GPU-local FP8 KV cache and one image per prompt.
It does not replace the [four-GPU community deployment](glm-5.3-flash.md).
The image platform is `linux/amd64`. “Spark” names the checkpoint variant;
these measurements do not qualify ARM-based DGX Spark hardware.

## Start serving

```bash
docker run -d --name glm-spark-tp2 --init \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  -v hf-cache:/root/.cache/huggingface \
  -v glm-spark-tp2-cache:/cache \
  localinferencelab/vllm:jovian-judgement-community-tp2-experimental-20260912-r1
```

The API listens on **0.0.0.0:8000**, including `/v1/chat/completions`. Use
`GLM-5.3-Flash-NVFP4-Spark` as the API model name. Set `-e PORT=5056` before
the image name to select another port. Change the two device IDs to select
other GPUs; both must be free of competing GPU processes.

The launcher downloads `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` into the
named Hugging Face cache. No DFlash checkpoint, host source directory, or
absolute checkpoint path is required. An existing checkpoint can instead be
mounted read-only and selected through `MODEL`. `MODEL_REVISION` optionally
pins a checkpoint revision; otherwise Hugging Face `main` is used.

The image does **not** change GPU clocks. Report whether memory overclocking
was enabled when sharing performance or correctness results.

## Default serving contract

| Setting | Value |
|---|---|
| Parallelism | TP2 / DCP2 |
| Speculation | MTP3; probabilistic proposals and standard rejection |
| Context limit | 1,048,576 tokens, including generated output |
| KV allocation | 4 GiB per rank; shared between requests, not four independent 1M contexts |
| Scheduler | 3,072 target tokens per iteration; at most four requests |
| Vision | One image at the native 8,000-feature ceiling; video disabled |
| Target / MTP MoE and dense projections | B12X |
| Attention | B12X sparse MLA; B12X KDA prefill, automatic KDA decode selection; full-image FlashAttention for vision |
| Target KV / recurrent state | FP8 / FP32 |
| Vocabulary heads | BF16 target verifier; private NVFP4 MTP proposal copy |
| CUDA graphs | Full and piecewise; capture sizes 1, 2, 4, 8, 12, 16 |
| Prefix policy | Request boundaries; one prefill lane, compute share 0.4 |
| Sampling defaults | Temperature 1.0, top-p 0.95; reasoning high, clear_thinking false |
| NCCL | Two channels, 1 MiB buffer |

Client-supplied sampling parameters take precedence. Explicit vLLM arguments
can override launcher arguments; changed image counts, scheduler budgets,
context sizes and concurrency limits require independent memory validation.
`DRY_RUN=1` prints the complete command without loading weights.

## Memory implementation

The implementation releases unused packed-weight copies, reuses disjoint
attention and KDA scratch, and allows consumed graph-pool outputs to be
reclaimed while retaining required graph resources. Token-local vision
projections run in 4,096-row chunks; full-image attention and image resolution
are unchanged. The serial target and MTP forwards share temporary indexer
buffers, but not model weights, KV contents or recurrent state.

These changes do not further quantize the checkpoint or offload weights.
The serial-sharing guard requires matching buffer geometry, a single MTP
layer, pipeline parallelism 1 and no overlapping microbatches.

The memory mechanisms can also apply to TP4, but their magnitude depends on
sharding and workload. **No TP4 memory or performance improvement is qualified
by the TP2 measurements.** The TP2 NCCL configuration is not a TP4 recommendation.

## Qualification scope

Status: **qualified** for the following same-process sequence on physical GPUs
2 and 3, measured on 2026-09-12:

- Cold 32,768-token image prompt, then fresh 8K text and two exact text replays.
- Cold 1,048,320-token image prompt with a 256-token output allowance: correct
  visual answer in 146.0 seconds, with zero prefix and image-cache hits.
- Fresh 8K text and two exact replays after the near-limit image request.
- Three active decoders plus a cold 32K image; all decoders made progress and
  the visual answer was correct.
- Fifteen text checks covering exact prompts, shared instructions, response
  continuation and tool history.
- C1/C4 decode and cold 32K prefill, with zero errors and no server restart.

The image is intended for bounded community testing, not unrestricted shapes
or concurrency. GPU-local text prefix replay and native-image capacity are
separate checks: exact multimodal endpoint/token replay is not supported by
the request-boundary adapter. TP2 LMCache, DFlash, multiple images, video and
arbitrary GPU topologies are not qualified by this deployment.

Physical free VRAM can approach zero because the allocator retains reusable
segments. A small physical-free counter alone does not establish an OOM, but
this profile should not share either GPU with another process.

## Measured performance

Two RTX PRO 6000 Workstation GPUs, **VRAM +6000**, graphics offset zero and
automatic graphics clocks; TP2/DCP2 MTP3 with the defaults above. These are not
stock-clock measurements. The benchmark uses temperature 1.0 and top-p 0.95.
Decode context is 0; each benchmark starts with a 15-second warmup and uses
30-second measured cells. The table retains all three samples per concurrency.
Prefill uses one warmup followed by 30 seconds of unique, uncached 32K requests.

| Measurement | Median | Range / individual samples |
|---|---:|---|
| 32K prefill | **10,765.9 input tok/s** | Ten cold samples; every prompt token computed locally |
| C1 MTP3 output | **195.7 tok/s** | 184.5 / 202.3 / 195.7 |
| C1 verifier | **78.75 steps/s** | 75.50 / 78.87 / 78.75 |
| C4 aggregate output | **455.1 tok/s** | 446.2 / 464.6 / 455.1 |
| C4 aggregate verifier | **183.55 steps/s** | 178.84 / 187.24 / 183.55 |

The same-GPU serial-indexer diagnostic reference recorded 10,864.7 prefill
tok/s, a three-run C1 median of 196.8 tok/s and 78.67 steps/s, and one C4
sample of 475.6 tok/s and 185.35 steps/s. The packaged image differs by
−0.91% prefill, −0.54% C1 output / +0.10% C1 steps, and −4.31% C4 output /
−0.97% C4 steps. C4 acceptance also differed. This sequential packaging check
is not a controlled speedup experiment; neither the slower first sample nor
the faster repeat is discarded, and no isolated performance gain is claimed.
The [qualification record](glm-5.3-flash/tp2-experimental-qualification.json)
identifies both image artifacts and retains the raw measurements and checks.

## Source and packaging

Published image digest, verified by pulling from DockerHub:

```text
localinferencelab/vllm@sha256:723159dff669c259d32fbe59e2887016baa4c5d3a67a61dba49c8c456286af5f
```

The image has two filesystem layers: a fixed runtime foundation and a complete
source installation. Its TP2 entrypoint is a metadata-only specialization;
CUDA, FlashKDA and B12X native artifacts are reused rather than rebuilt.

`/opt/glm53-flash/source.lock` identifies exact component commits, trees and
build-input hashes. Complete Git histories are included at
`/opt/glm53-flash/vllm`, `/opt/glm53-flash/b12x` and `/opt/lmcache/source`.
The source lock, not an assumption about a mutable branch, defines the image.
