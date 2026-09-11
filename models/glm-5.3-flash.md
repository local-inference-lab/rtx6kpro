# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke for
Local Inference Lab.</em></p>

This page specifies the qualified GLM-5.3-Flash deployment for four NVIDIA RTX
PRO 6000 Blackwell Workstation Edition GPUs. The runtime serves the
`local-inference-lab/GLM-5.3-Flash-NVFP4` target checkpoint without
speculation, with three-token Multi-Token Prediction (MTP), or with the
`local-inference-lab/GLM-5.3-Flash-DFlash2` draft checkpoint.

The commands use Hugging Face repository names and named Docker volumes. They
do not require checkpoint paths or source-code bind mounts.

## Status

| Capability | Status |
|---|---|
| Tensor parallelism of four with one decode-context rank | **qualified** for no speculation, MTP depth 3, and DFlash2 depth 7 |
| Tensor parallelism of four with four decode-context ranks | **qualified** for the same three serving modes, including complete-KV prefill |
| Two decode-context ranks | **implemented**; not independently performance-qualified for this artifact |
| Tensor parallelism of eight | **implemented**; not independently hardware-qualified for this artifact |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4`; Hugging Face `main` unless `MODEL_REVISION` is set |
| QAD step-1,750 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step1750`](../kld/glm-5.3-flash-qad-step1750.md); distribution fidelity, verifier-backed behavior, and AA-LCR are measured, but the checkpoint is not a qualified serving target |
| QAD step-2,500 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step2500`](../kld/glm-5.3-flash-qad-step2500.md); distribution fidelity is measured, and the [R30 nucleus-sampling VBF report](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md) qualifies practical equivalence on its primary semantic score; production serving remains unqualified |
| QAD TV-nucleus step-2,500 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-TVN-step2500`](../kld/glm-5.3-flash-qad-tvn-step2500.md); distribution fidelity is measured, and the [R30 nucleus-sampling VBF report](glm-5.3-flash/qad-tvn-step2500-verifier-backed-behavioral-fidelity.md) qualifies practical equivalence to published NVFP4 and QAD step 2,500; production serving remains unqualified |
| AA-LCR capability evaluation | **qualified** for the exact BF16, published-NVFP4, and QAD checkpoint-and-runtime configurations in the [three-configuration report](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) |
| Verifier-backed behavioral fidelity | **qualified** practical equivalence among published NVFP4, QAD step 2,500, and QAD TV-nucleus step 2,500 under the [R30 temperature-1/top-p-0.95 three-checkpoint contract](glm-5.3-flash/verifier-backed-behavioral-fidelity.md) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2`; Hugging Face `main` unless `DFLASH_MODEL_REVISION` is set |
| Target routed experts | ModelOpt NVFP4 using B12X 4-bit weights and 4-bit activations; eligible prefills share input quantization and use separate expert projections |
| DFlash2 weights | Offline-serialized ModelOpt MXFP8; no online weight quantization |
| Target KV cache | **qualified** FP8; packed NVFP4 is implemented but not qualified for R35 |
| MTP proposal vocabulary head | NVFP4 draft-only copy by default; the target verifier vocabulary head remains BF16 |
| GPU prefix cache | **qualified** request/SYSTEM boundaries in all six TP4 mode/DCP combinations; fine aligned retention is selectable |
| Native DRAM offload | **implemented** and opt-in with `CACHE_MODE=native`; not independently requalified for R34 |
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache`; asynchronous engine-driven pinned shared memory is the default transfer path |
| CUDA graphs | **qualified** with launcher default `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` for target and speculative decode |
| Scheduler | 4,096 target tokens per step; fixed prefill compute share 0.4; interval 1; one prefill lane by default, optional bounded interleaving |
| Root filesystem | Two layers: flattened runtime foundation and committed source installation |
| FlashKDA numerical stability | **qualified** with the stable FP32 forward-substitution inverse |
| Qwen3.8-Flash-Next serving | **qualified** separately for TP1/MTP3 text, GPU prefix cache and bounded performance; see the [Qwen deployment page](qwen38-flash-next.md) for launch, PLE offload, clock conditions, TP2 limitations and results |
| DeepSeek V4 serving | **qualified** for bounded TP2/DCP1 FP8 text and Vision checks; see the [DS4 runbook](ds4-jovian-community-r29.md) |
| Qualification date | 2026-09-11; mode-specific evidence retains its measured release identity |

R35 qualifies the public source composition with GLM DFlash2 TP4/DCP1 serving,
C1/C8, cold 32K prefill, prefix restoration and focused GPU correctness tests.
The capability table also records qualification inherited from the linked
mode-specific reports; it does not imply that every matrix was repeated on R35.
No complete MTP3/DCP4/LMCache restart or Qwen/DeepSeek serving retest is claimed.

**Upgrade GLM NVFP4 deployments from R33/R34:** their split MoE prefill omits
the model's SwiGLU clamp. R35 applies the merged B12X correction without
disabling the fast split path. The defect and arithmetic fix are reproduced;
this is not proof that every reported runaway generation has the same cause.

The [BF16-to-NVFP4 distribution-fidelity report](../kld/glm-5.3-flash-bf16-nvfp4.md),
[QAD step 1,750 comparison](../kld/glm-5.3-flash-qad-step1750.md), and
[QAD step 2,500 progression comparison](../kld/glm-5.3-flash-qad-step2500.md),
and [QAD TV-nucleus step 2,500 comparison](../kld/glm-5.3-flash-qad-tvn-step2500.md)
are research-only. They measure a reproducible FlashInfer CUTLASS path rather
than the B12X serving path specified here.

The [AA-LCR result](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) qualifies the BF16,
published NVFP4, and QAD configurations on 100 long-context questions with
three independent generations each. The published checkpoint scores 74.00%,
QAD scores 73.00%, and BF16 scores 71.67%; paired evidence does not distinguish
the three complete configurations. The accompanying
[reproduction specification](glm-5.3-flash/aa-lcr-reproduction.md) fixes the
dataset, prompt, sampling, runtime, equality checker, and receipt validation.

The [Verifier-Backed Behavioral Fidelity report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md)
compares published NVFP4, QAD step 2,500, and QAD TV-nucleus step 2,500 on
7,168 deterministic tasks with executable answer keys and no language-model
judge. Each checkpoint produces three fixed-seed responses per task under the
R30 temperature-1/top-p-0.95 serving contract. Their primary semantic scores
are 94.5386%, 94.5698%, and 94.5989%, respectively. Every paired 95% interval
lies inside the predeclared ±1-point equivalence band and crosses zero. The
result establishes practical equivalence on the primary endpoint, not a
superior checkpoint. Exact-task and family diagnostics remain separate
secondary outcomes.

## Docker artifact

```text
localinferencelab/vllm:jovian-judgement-community-20260911-r35
```

The image contains two filesystem layers: a flattened CUDA 13.3/PyTorch 2.13
runtime foundation and one installation of complete committed vLLM, B12X and
LMCache sources. FlashInfer, the DS4-compatible native vLLM operator and the
authenticated FlashKDA extension are source-locked. It is not built by adding
layers to a preceding community release.

The [embedded source lock](glm-5.3-flash/validation/swiglu-reviewed-composition-r35.source.lock)
identifies the installed components and build inputs. The
[R35 qualification and R34-to-R35 changelog](glm-5.3-flash/validation/swiglu-reviewed-composition-r35.md)
records artifact identity, measurements and qualification limits; the
[registry receipt](glm-5.3-flash/validation/swiglu-reviewed-composition-r35-registry.json)
contains the immutable digest and verified pull result.
Eligible GLM and Qwen prefills share quantized input across routed experts and
use separate expert projections. LMCache filesystem eviction retires missing
objects from byte accounting while protecting pending writes and preserving
actual I/O errors. Checkpoint policies, concurrent publication, model precision,
sampling/history defaults and model launch profiles are preserved. The separate
standalone LMCache wrapper also provisions named SHM for explicit engine-driven
transfer. Native libraries match the qualified source-composition image; no
CUDA, FlashInfer or FlashKDA replacement is involved. The image sets
`VLLM_DEFAULT_MOE_BACKEND=b12x`, so direct
`vllm serve` also selects B12X when `--moe-backend` is omitted. An explicit
backend, including `auto`, remains authoritative. The GLM wrapper additionally
accepts `MOE_BACKEND`; its default is B12X. This changes MoE selection, not
attention or sampler selection.

The same installed runtime supports
[Qwen3.8-Flash-Next](qwen38-flash-next.md) and
[DeepSeek V4 text/Vision](ds4-jovian-community-r29.md) through separate launch
profiles. DS4 backend defaults do not replace the GLM settings below.

Known limitation: concurrent MTP3 requests with strict JSON-schema output and
LMCache can fail grammar validation with HTTP 500. The failure is reproduced
on both R31 and R32; [vLLM #726](https://github.com/local-inference-lab/vllm/issues/726)
tracks it. R35 does not claim to fix that constrained-output defect.

## Runtime backends

| Operation | Implementation |
|---|---|
| Target sparse attention and C4 index selection | B12X |
| Target recurrent prefill | FlashKDA with packed checkpoint exports |
| Target recurrent decode | B12X when eligible, with the supported Triton path otherwise |
| Target routed experts | B12X NVFP4 W4A4; shared-input split projections for eligible prefills |
| Target dense projections | B12X |
| Tensor-parallel all-reduce | B12X PCIe one-shot/two-shot for supported sizes; PyNCCL for the remaining sizes |
| MTP attention / experts | B12X / Marlin |
| MTP vocabulary projection | Private NVFP4 draft copy; target vocabulary projection remains BF16 |
| DFlash2 weights and linear projections | Offline MXFP8 checkpoint and B12X |
| DFlash2 local attention | Graph-safe split-KV FlashAttention |
| Sampling | FlashInfer-compatible probability dispatch and standard rejection |
| External cache | LMCache through worker-owned asynchronous pinned SHM |

The target uses FP8 KV cache. DFlash's local attention uses its compatible
automatic cache dtype; it is not advertised as an MXFP8 KV format.
DeepGEMM and TileLang are installed dependencies but are not selected for the
qualified GLM target, MTP or DFlash2 hot paths. FlashKDA is the prefill default;
`GLM53_KDA_PREFILL_BACKEND=b12x` selects the retained B12X alternative. The
performance table below uses FlashKDA, not that alternative.

## Measured performance

### R35 source-composition qualification

Same physical quartet of RTX PRO 6000 Workstation GPUs, **VRAM +6000**, 600 W,
TP4/DCP1, DFlash2 K7, FP8 KV, B12X MoE, 4096-token budget, OMP1 and
full-and-piecewise graphs. Sampling uses temperature 1 and top-p 0.95.

| Metric | R34 | R35 component composition | Change |
|---|---:|---:|---:|
| Cold 32K prefill tok/s | 16,872 | 16,694 | −1.05% |
| C1 output tok/s | 254.85 | 255.05 | +0.08% |
| C1 verifier steps/s | 97.37 | 97.71 | +0.34% |
| C8 aggregate output tok/s | 803.70 | 792.84 | −1.35% |
| C8 aggregate verifier steps/s | 308.74 | 310.74 | +0.65% |

These are bounded screening measurements, not a general speedup claim. R35's
installed component sources and native libraries match the measured composition;
only release metadata and the separately tested standalone LMCache wrapper
differ. [Conditions, artifact proof and final-image checks](glm-5.3-flash/validation/swiglu-reviewed-composition-r35.md).

### R34 deployment-default qualification

GLM DFlash2 K7, TP4/DCP1, FP8 KV, the same physical RTX PRO 6000 Workstation
quartet with **VRAM +6000**, 4096-token budget and full-and-piecewise graphs:
32K prefill **16,961 → 16,997 tok/s (+0.21%)**, R33 → R34. The R34 C1 cell gives
**249.77 output tok/s and 97.40 verifier steps/s**. Both images use B12X;
this is not a B12X-versus-FlashInfer comparison. The short C1 control has
different speculative acceptance, so no general decode gain is claimed.
[Conditions, control values and raw samples](glm-5.3-flash/validation/moe-backend-default-r34.md).

### R33 shared-input NVFP4 prefill

Same physical quartet of **RTX PRO 6000 Blackwell Workstation, 600 W,
VRAM +6000**, used sequentially for R32 and R33. TP4/DCP1, FP8 target KV,
GPU cache, 4096-token scheduler budget, OMP1, 16 NCCL channels/2 MiB buffers
and full-and-piecewise graphs. These are **not stock-clock measurements**.

Prefill sends exactly 32,768 input tokens and one output token, excludes one
warmup and measures for at least 30 seconds. Server counters confirm zero
prefix-cache reuse. The rate includes first-output work. No-spec uses
temperature 0/top-p 1; MTP3 and DFlash2 use temperature 1/top-p 0.95.

| Mode | 32K input tok/s, R32 → R33 | Change | R33 C1 output tok/s | R33 C8 aggregate output tok/s |
|---|---:|---:|---:|---:|
| No speculation | 15,737 → 17,128 | **+8.84%** | 177.05 | 771.50 |
| MTP3 | 15,276 → 16,637 | **+8.91%** | 275.45 | 1002.93 |
| DFlash2 K7 | 15,569 → 16,910 | **+8.61%** | 244.10–260.56 | 785.79–789.48 |

C1 and C8 mean one and eight concurrent clients. Decode uses context 0,
temperature 1/top-p 0.95, ten-second warmup and 30-second measured cells.
MTP3 output changes −2.03%/−1.54% at C1/C8 while verifier rate changes
+0.99%/−0.32%; acceptance differs. DFlash2 repeats overlap the reference's
observed verifier states. A general decode speedup or statistical equivalence
is **not established**. The [complete comparison](glm-5.3-flash/validation/fp4-prefill-filesystem-r33.md#glm-performance)
retains every reference and repeat, acceptance, numerical limits and exact
answer checks. Sieve was not rerun for R33.

LMCache passes exact 54,641-token GPU/RAM/filesystem/restart restores with zero
recomputed prompt tokens. Filesystem restore takes 0.277 seconds and restore
after worker/sidecar restart 0.414 seconds, including answer generation, with
a warm OS page cache. These are bounded correctness measurements, not a
storage-bandwidth or one-million-token qualification. See the
[cache evidence](glm-5.3-flash/validation/fp4-prefill-filesystem-r33.md#lmcache-filesystem-qualification).

### Historical R31 warmup and retained-RAM update

Same physical quartet of **RTX PRO 6000 Max-Q Workstation, 300 W, VRAM +6000**;
TP4/DCP1 MTP3, FP8 KV, 4096-token budget, full-and-piecewise graphs,
temperature 1/top-p 0.95. LMCache is disabled for these performance cells.

| Measurement | R30 | R31 source | R31 source repeat |
|---|---:|---:|---:|
| Cold 32K prefill | 11,151 tok/s | 11,068 tok/s (−0.74%) | Not repeated |
| C1 output | 264.61 tok/s | 248.51 tok/s (−6.09%) | 259.26 tok/s (−2.02%) |
| C1 verifier | 103.406 steps/s | 103.571 (+0.16%) | 103.389 (−0.02%) |
| Mean emitted tokens per step | 2.5595 | 2.3999 | 2.5081 |

Prefill and verifier execution are essentially unchanged. Output varies with
acceptance; no throughput improvement is established. GPU KV capacity remains
3,780,444 logical tokens. The [report](glm-5.3-flash/validation/warmup-retention-r31.md)
preserves all cells, the source/final-image boundary and exact cache tests.
Do not compare this Max-Q table directly to stock 600 W Workstation results.

### Historical R29-to-R30 source comparison

The [R30 source comparison](glm-5.3-flash/validation/shared-serving-r30.md#matched-performance)
uses DFlash2 K7, TP4/DCP4 and engine-driven LMCache on the same stock quartet:

| Measurement | R29 → R30 source composition | Change |
|---|---:|---:|
| Cold 32K prefill | 13,294 → 13,285 input tok/s | −0.07% |
| C1 output, 30-second cell | 201.73 → 212.33 tok/s | +5.26% |
| C1 verifier | 81.04 → 81.26 steps/s | +0.27% |
| Sieve C1 output, five-run median | 256.40 → 336.69 tok/s | +31.31% observed, not an established speedup |
| Sieve output min–max | 242.57–331.69 → 229.21–415.05 tok/s | Broad overlapping ranges |
| Sieve verifier, median | 77.46 → 77.76 steps/s | +0.39% |

Prefill and verifier execution are effectively unchanged. Speculative
acceptance explains the variation in emitted tok/s; these samples do not
establish a repeatable 31% gain. Sieve uses temperature 1/top-p 0.95 and
4096 output tokens. The short duration cells retain top-p 1 to compare serving
source; they do not measure the R30 default sampling change. All artifact and
measurement boundaries are in the linked report. R30 does not repeat C8/C64
throughput; historical results below remain labelled with their measured image.

### Historical TP4/DCP1 comparison: R28.1 and R29

Stock RTX PRO 6000 Blackwell Workstation GPUs, TP4/DCP1, FP8 target KV,
4096-token scheduler budget, OMP1, NCCL 16 channels/2 MiB buffers and
`FULL_AND_PIECEWISE` graphs. The R28.1 control and shared-runtime candidate
were measured on the same physical GPUs3/12/13/14. C1 is context-zero decode,
temperature 1, 15 seconds of warmup and 30 seconds measured.

Prefill is the client TTFT-derived input rate for a cold nominal 32K bucket
(about 32,315 actual tokens), with 12 samples per image. It includes
first-output work rather than timing only an attention kernel.

| Mode | 32K prefill, R28.1 → shared tok/s | C1 output, R28.1 → shared tok/s | C1 verifier, R28.1 → shared steps/s |
|---|---:|---:|---:|
| No-spec | 14,650 → 14,733 (+0.57%) | 169.38 → 169.35 (−0.02%) | Same as output rate |
| MTP3 | 14,279 → 14,320 (+0.29%) | 258.02 → 265.19 (+2.78%) | 102.39 → 108.69 (+6.15%) |
| DFlash2 K7 | 14,519 → 14,529 (+0.07%) | 226.01 → 230.98 (+2.20%) | 89.40 → 89.55 (+0.16%) |

These are bounded observations, not proof of a universal speedup. MTP accepted
length changes 2.520 → 2.440; DFlash changes 2.528 → 2.579. The MTP verifier
difference was not isolated to a particular change, and host-side concurrent
work differed between cells. Source/image boundaries are explicit in the
[qualification report](glm-5.3-flash/validation/shared-serving-r29.md): these
performance cells precede the LMCache metadata and DS4 BF16-router corrections;
their GLM GPU-cache hot path is unchanged by those corrections.

C8/C64 and Sieve were not rerun for R29. The
[R28.1 report](glm-5.3-flash/validation/scheduler-serving-r28.1.md) retains its
C8/C64 results and the short DFlash C8 decrease without relabelling them as R29.
The [R28 report](glm-5.3-flash/validation/fp8-serving-r28.md) retains the
six-mode/DCP matrix and Sieve results. None of the table above uses a +6000
VRAM offset.

## Start the server

Use model names and named volumes; source-code mounts are not needed.
The defaults already select full-and-piecewise graphs, the B12X paths,
FlashInfer sampling, NCCL 16 channels/2 MiB and OMP1.

```bash
IMAGE=localinferencelab/vllm:jovian-judgement-community-20260911-r35
GPU_DEVICES=0,1,2,3
PORT=8000
docker pull "$IMAGE"
```

Choose one mode. No speculation:

```bash
NAME=jovian-judgement-nospec
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=0)
```

MTP3:

```bash
NAME=jovian-judgement-mtp3
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=3)
```

DFlash2 with seven draft tokens:

```bash
NAME=jovian-judgement-dflash2
MODE_ARGS=(-e SPECULATOR=dflash2 -e DFLASH_DEPTH=7
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2)
```

Run the common command after assigning the chosen mode's variables:

```bash
docker run -d --name "$NAME" --init \
  --gpus "\"device=${GPU_DEVICES}\"" --network host --ipc host \
  -v jovian-judgement-r33-runtime-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4 \
  -e CACHE_MODE=vram -e KV_CACHE_QUANT=fp8_ds_mla \
  -e TP=4 -e DCP=1 -e PORT="$PORT" \
  -e MAX_MODEL_LEN=1048576 -e MAX_NUM_SEQS=32 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=1 -e PREFILL_COMPUTE_SHARE=0.4 \
  -e GPU_MEMORY_UTILIZATION=0.93 \
  "${MODE_ARGS[@]}" "$IMAGE"
```

For DCP4, change `-e DCP=1` to `-e DCP=4`. Full-CKV prefill is selected
automatically; it is not DFlash-only. DCP2 and TP8 are implemented but are not
independently qualified by the TP4/DCP1/DCP4 measurements on this page.

Model names resolve Hugging Face `main` at startup. Optional `MODEL_REVISION`
and `DFLASH_MODEL_REVISION` variables provide reproducible model selection.
The runtime authenticates the resolved revisions before external checkpoint
reuse; changing a model or source identity produces safe cache misses.

### Scheduler and cache geometry

`PREFILL_COMPUTE_SHARE=0.4` targets 40% of measured execution time for prefill
only while prefill and decode contend. `PREFILL_SCHEDULE_INTERVAL=1` is required.
Set `FAIRNESS_ENGINE=none` to disable this policy. Model execution is indivisible,
so the realized share can oscillate over short windows. External-cache transfers
do not consume local-prefill compute credit or a prefill lane while waiting.

The scheduler launcher exposes these controls. Explicit native CLI arguments
after the image name override corresponding environment values without duplicate
flags.

| Environment | Native CLI option | Values and default |
|---|---|---|
| `PREFILL_COMPUTE_SHARE` | `--prefill-compute-share` | Finite number strictly between 0 and 1, or `auto`; launcher default `0.4` |
| `PREFILL_COMPUTE_HALF_LIFE` | `--prefill-compute-half-life` | `smooth`, `responsive`, or positive finite seconds; valid only with share `auto` |
| `MAX_PARALLEL_PREFILLS` | `--max-parallel-prefills` | Positive integer or `auto`; default `1`; `auto` selects at most four lanes, capped by `MAX_NUM_SEQS` |
| `PREFILL_POLICY` | `--prefill-policy` | `round-robin` (default) or `decode-aware` |
| `DECODE_REFILL_TARGET` | `--decode-refill-target` | Positive integer or `auto` (default); automatic target equals the effective lane count |

The lane count is independent of attention pages, recurrent checkpoints and
LMCache object size. All lanes share the same global 4096-token scheduler budget;
four lanes do not multiply that budget by four.

For concurrent long prefills and latency-sensitive short requests, opt into:

```bash
# Add before "$IMAGE" in the common docker run command:
-e MAX_PARALLEL_PREFILLS=auto \
-e PREFILL_POLICY=decode-aware \
-e DECODE_REFILL_TARGET=auto
```

Keep fixed share `0.4` initially. Interleaving can bring a short request to decode
sooner by distributing service among long requests; it can increase their
individual time to first token. One lane remains the image default. Automatic
compute share is implemented for experiments, not selected as the production
default. `FAIRNESS_ENGINE=micro_slicing` is rejected.

MTP uses a private NVFP4
proposal vocabulary head by default, costing 85.08 MiB per TP4 rank. The target
vocabulary head remains BF16; `VLLM_GLM53_MTP_DRAFT_HEAD=bf16` selects the
unquantized draft head for independent comparison.

The launcher owns page geometry. GPU-local serving keeps 2048-token attention
pages; fine recurrent retention is selected separately below. LMCache derives
per-rank pages from 4096-token storage objects and the DCP width. The public
vLLM attention block argument remains 256. Weighted allocation groups layers
by their actual cache cost, with a bounded number of groups, to reduce padding.
Normal deployments do not need to override these layout settings.

### Reasoning effort

The GLM image launcher defaults chat reasoning to **high**, not max. An API
request can override it with `"reasoning_effort":"max"` or
`"reasoning_effort":"low"`. High still enables reasoning; it is not a
no-thinking mode. To change the server default, pass
`--default-chat-template-kwargs '{"reasoning_effort":"max"}'` after the image
name. Running `vllm serve` directly bypasses the image launcher's default.

The agent profile sets **`clear_thinking=false`**: historical reasoning supplied
by the client remains in the rendered conversation. Clients must return it in
the corresponding assistant messages. Explicit request template options
override the profile. Setting `clear_thinking=true` deliberately removes
completed-turn reasoning and changes the token prefix; it is not a matched-input
repair for generation problems. When replacing the complete server template
JSON, include `"clear_thinking":false` to retain this agent behavior.

### Sampling defaults

The GLM launcher supplies **temperature 1 and top-p 0.95**, matching the
[publisher generation configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/generation_config.json),
even if the quantized checkpoint metadata omits them. Fixed top-k is disabled
(vLLM effective `top_k=0`); GLM does not inherit Qwen's top-k 20.

Per-request `temperature`, `top_p` and `top_k` override individual defaults.
Explicit native `--generation-config`, `--override-generation-config` or
`--config` arguments replace the launcher's sampling preset, allowing an
operator-owned policy. Direct `vllm serve` invocation bypasses the launcher.
The [Hugging Face metadata PR](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/discussions/4)
proposes the same defaults; Docker does not depend on its merge.

Five no-spec and three DFlash2 full 839,815-token preserved-history requests
showed no degeneration at top-p 0.95. That finite result supports the profile,
not a universal numerical-repair claim. Top-p 1 remains available explicitly.

### Recurrent checkpoint policy

The default `auto` policy selects request-boundary retention for the qualified
GLM no-spec/MTP3/DFlash2 configurations, including DCP4. It preserves exact
request endpoints and leading SYSTEM/developer instruction endpoints. It does
not promise a hit at every arbitrary byte or token inside a changed user turn.

For arbitrary shared token prefixes, GPU-local aligned retention is selectable.
Add the following environment setting before the image name and arguments
after the image name:

```bash
# Before "$IMAGE":
-e GLM53_MAMBA_BLOCK_SIZE=256

# After "$IMAGE":
--recurrent-checkpoint-policy aligned --prefix-match-unit 256
```

Attention pages remain 2048 tokens. Packed exports retain interior recurrent
states without forcing a target forward per checkpoint. This trades additional
retained recurrent state for finer prefix reuse. Request-boundary retention
stores fewer states for ordinary shared-instruction and turn-boundary reuse.
The fine-aligned MTP3/DCP4 interval comparison measured −0.20% prefill,
+0.32% C1 verifier and +0.35% C8 verifier rate; that comparison precedes the
disjoint-MLA projection change. It is not a fresh speed claim for every mode.

## LMCache RAM and filesystem storage

LMCache is opt-in. In the common command, replace `-e CACHE_MODE=vram` with:

```bash
-e CACHE_MODE=lmcache \
-e LMCACHE_TRANSFER_MODE=engine_driven \
-e LMCACHE_L1_SIZE_GB=64 \
-e LMCACHE_L2_ENABLED=1 \
-v jovian-judgement-r33-lmcache-l2:/lmcache-l2
```

The host shared-memory filesystem must have at least 96 GiB available for the
default 64 GiB RAM pool and transfer buffers. With `--ipc host`, the host's
`/dev/shm` capacity applies; `--shm-size` does not enlarge it. The sidecar runs
inside the container and has no CUDA context. Existing vLLM workers perform
GPU gather/scatter through asynchronous pinned SHM. Do not manually remove a
shared-memory pool while either service is using it.

The launcher chooses geometry compatible with `LMCACHE_CHUNK_SIZE=4096` and
`LMCACHE_TARGET_TOKEN_BUDGET=4096`. Semantic target, recurrent and draft state
are published only as complete all-rank bundles. Payload locks protect active
copies, and incompatible source/model/layout identities safely miss. RAM
pressure evicts unlocked payloads; filesystem storage remains reusable after
both services restart. Incomplete or incompatible semantic generations are
recomputed, never partially imported. Version-1 semantic payload files do not
match the version-2 storage keys used here.

Set `LMCACHE_L2_ENABLED=0` for RAM-only operation. If multiple instances share
the host network, give each distinct API and LMCache HTTP/MP/metrics ports.
Do not share a writable cache directory across independent sidecars.

`LMCACHE_L2_PREFETCH_POLICY=retain` is the default: disk-loaded objects remain
reusable in the bounded host-RAM L1 after readers finish. They are not pinned
forever; LRU can evict objects without active readers or writers. If a retained
filesystem restore needs RAM, the launcher enables bounded emergency eviction
without enabling writeback. Active owners can still force a safe cache miss.
This is host-memory caching, not GPU hardware L2 prefetching.

`LMCACHE_L2_PREFETCH_POLICY=default` selects temporary prefetched objects that
are released when readers finish. `LMCACHE_SERVER_EXTRA_ARGS` accepts literal
whitespace-separated server options, for example `--max-cpu-workers 4`.
Shell expressions and embedded quoting are not evaluated; use dedicated
variables for identity, geometry, transport and listener settings.

R31's 4 GiB RAM-pressure test writes 6.62 GB of durable objects, then restores
an evicted 32K prompt in 0.170 s with zero recompute. A 54,643-token literal
lookup restores from RAM in 0.278 s, filesystem in 0.293 s and after both
services restart in 0.410 s. All answers are exact. After filesystem load,
128 objects / 1.76 GB remain reusable in RAM. These are bounded TP4/DCP1 MTP3
checks with a warm OS page cache; see the
[retention report](glm-5.3-flash/validation/warmup-retention-r31.md#ram-retention-and-restore-correctness).

`LMCACHE_HTTP_HOST` defaults to `127.0.0.1`. Wildcard or IPv6 binds have matching
readiness addresses. This interface exposes administrative operations; remote
access requires a trusted network or authenticated proxy.

Identical semantic checkpoints avoid duplicate payload publication. Growing
histories share complete attention pages between immutable endpoints, while
partial tails and recurrent/draft state remain endpoint-specific. A 32,768-token
DFlash2/DCP4 cold store writes 80 objects; exact local/RAM/filesystem replays
add none. Appending 256 tokens reuses the complete prefix and writes 40 objects.
The [R30 report](glm-5.3-flash/validation/shared-serving-r30.md#cache-and-source-correctness)
records three-mode all-rank byte comparisons and cancellation/eviction evidence.

The retained recurrent-checkpoint architecture has
[all-six FP8 mode/DCP million-token qualification](glm-5.3-flash/validation/fp8-serving-r28.md#checkpoint-storage).
The exact R28.1 image separately passes MTP3/DCP4 with four prefill lanes:

| Mode / DCP | 1M cold, seconds | RAM restore, seconds | Restore after both services restart, seconds |
|---|---:|---:|---:|
| MTP3 / 4, four lanes | 99.284 | 0.855 | 0.970 |

Each restore attributes all one million prompt tokens to external storage and
zero to local compute. Times include API/first-output work. The OS page cache
was not flushed; restart results are not cold-device storage benchmarks.
The R28.1 image also passes 54K literal lookup answers across cache tiers,
shared-SYSTEM reuse, C4 all-rank bytes and C8 cancellation/live-read eviction.
These tests qualify storage correctness, not universal bitwise generation
equivalence across different floating-point prefill partitions.

A separate one-observation, same-quartet R28/R28.1 RAM comparison records
0.725 → 0.813 seconds for 1M tokens. That 88 ms difference is retained in the
report; it is insufficient to establish steady-state transfer-speed equivalence.

Native DRAM offload remains implemented through `CACHE_MODE=native`; it is not
independently requalified for R31. The qualified external path above is LMCache.
Packed NVFP4 target KV and Qwen LMCache are outside this release's qualification.

R29 additionally qualifies DFlash2/DCP4 after the paged-gather metadata fix:
54K cold/RAM/restart-filesystem answers, shared SYSTEM reuse, exact comparison
of 3,639,803,904 transferred bytes across four ranks, and three C8 cancellation
and live-read eviction rounds. This bounded check is not a repeat of the
one-million-token timing matrix.

Use an empty external-cache namespace when adopting R32. Immutable pinned block-ID
snapshots prevent asynchronous gathers from copying a later batch's pages.
The correction cannot repair payloads written without that guarantee. Atomic
GLM checkpoint identities reject incompatible sources; fresh named volumes
also isolate ordinary external-cache objects. Preserve existing volumes until
their owner chooses to remove them.

For independent native-offload testing, use `-e CACHE_MODE=native` and
`-e NATIVE_KV_OFFLOADING_SIZE_GB=64` in the common command. The launcher selects
the required shareable allocator. These settings do not enable LMCache.

## Source and review contract

The [portable source-locked build recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/main/recipes/glm53)
includes the native FlashKDA build, source-bundle verification and CPU build
tests. It lists the exact source revisions; no chain of preceding community
images is needed. Runtime ABI dependencies are supplied by its pinned base.

Complete Git mirrors preserve authorship and integration resolutions:
[vLLM](https://github.com/voipmonitor/vllm/tree/integration/jovian-reviewed-sources-20260911),
[B12X](https://github.com/voipmonitor/b12x/tree/integration/jovian-reviewed-sources-20260911),
[LMCache](https://github.com/local-inference-lab/LMCache/tree/release/jovian-fp4-fs-ledger-r33-20260910).
The [open merge checklist](https://github.com/local-inference-lab/vllm/issues/731)
describes each PR and integration caveat. Source locks, not tag-name inference,
identify the measured packages. The timing matrix and exact packaged storage
checks have their source boundaries recorded in the validation evidence.

Git histories retain the original contributions and author attribution,
including Luke Alonso, MadeBy561, Giancarlo Delfin, Derek Yates, Thien Tran,
logprobz, Apple FCU Fleet, Martin Vit, Codex and the other recorded authors.
