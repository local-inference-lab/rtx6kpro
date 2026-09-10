# Qwen3.8-Flash-Next

Run [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
with the Jovian Judgement community image. Tensor Parallelism (TP) of one means
**one GPU**, not four. The qualified text-serving configuration uses three-token
Multi-Token Prediction (MTP), abbreviated MTP3 here, and offloads the per-layer
n-gram embedding (PLE) table to host RAM. It is a different model from
[Qwen3.8-27B](qwen38-27b.md).

```text
localinferencelab/vllm:jovian-judgement-community-20260910-r33
```

The image contains the same vLLM/B12X runtime as [GLM-5.3-Flash](glm-5.3-flash.md),
but Qwen needs its own launch arguments. The Compose recipe below bypasses the
image's GLM entrypoint. No source mounts or absolute checkpoint paths are needed.
R33 retains temperature 1/top-p 0.95/top-k 20 and enables shared-input NVFP4
split prefill for eligible TP1 expert shapes. The packaged image passes the
TP1/MTP3 answer, prefix-cache, logprob and performance checks below. R29 and
R28.1 comparisons retain their measured artifact and clock conditions.

## Start on one GPU: TP1

Status: **qualified** for text, MTP3, 8-bit floating-point (FP8) key/value (KV)
cache, GPU prefix reuse and the performance measurements below on one 96 GB RTX PRO 6000 Blackwell
Workstation Edition GPU. The recipe does not change GPU clocks.

Download the [Compose file](qwen38-flash-next/qwen38-flash-next.compose.yml):

```bash
curl -fL -o qwen38-flash-next.compose.yml \
  https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/qwen38-flash-next/qwen38-flash-next.compose.yml

GPU=0 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp1 up -d
docker compose -f qwen38-flash-next.compose.yml --profile tp1 logs -f qwen-tp1
```

`GPU` selects the physical card for TP1; `GPU0` and `GPU1` select the pair for
TP2. Choose unused GPUs and a port. Requirements are Linux, Docker Compose with
GPU reservations/profiles, NVIDIA Container Toolkit and a driver compatible
with the image's CUDA 13.3 runtime. Model weights occupy approximately 98.5 GB
on disk, in addition to the Docker image and compiler cache. A first launch
downloads the model by repository name; later launches reuse named volumes.
Model loading and CUDA graph capture take several minutes even with local weights.

The API model name is **`Qwen3.8-Flash-Next`**:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Explain what a hash table is."}],"temperature":1,"top_p":0.95,"top_k":20,"max_tokens":2048}'
```

Use `http://SERVER:8000/v1` in an OpenAI-compatible client. The recipe binds all
host interfaces without authentication; expose it only on a trusted network
or behind an authenticated proxy. For a non-thinking request, add
`"chat_template_kwargs":{"enable_thinking":false}` to the JSON body.
That is a different workload from the reasoning benchmark below.

## Start on two GPUs: TP2

Status: **implemented**, with a statically checked recipe; TP2 serving and
performance have **not been qualified on the shared R33 image**. Measurements from
other Qwen-specific images are not substituted for that missing result.

Select two distinct available GPUs. Stop the TP1 service before switching
profiles, because both profiles use the same API port:

```bash
docker compose -f qwen38-flash-next.compose.yml --profile tp1 down
GPU0=0 GPU1=1 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp2 up -d
docker compose -f qwen38-flash-next.compose.yml --profile tp2 logs -f qwen-tp2
```

The TP2 profile changes both the exposed device pair and
`--tensor-parallel-size 2`. MTP3, PLE offload, precision, token budget and graph
settings remain identical. B12X PCIe all-reduce is enabled for eligible
collectives, with the NVIDIA Collective Communications Library (NCCL)
available for other sizes; this requires working GPU peer access. At TP1 no
multi-GPU collective is executed.

Do not activate both profiles together. The named model/cache volumes survive
`down`; do not add `--volumes` unless deleting those caches is intended.

## N-gram / PLE offload

**`VLLM_PLE_CPU_OFFLOAD=1` is enabled in both recipes.** It puts the model's
large n-gram lookup table in CUDA-mapped host RAM. GPU kernels read the required
entries over PCIe; the rest of the model still runs on the selected GPUs.
Offloading changes storage placement, not the checkpoint's quantization.
It is neither LMCache nor n-gram speculative decoding.

The qualified TP1 startup records **26.82 GiB of mapped host RAM for PLE** and
**74.64 GiB of GPU model-loading memory**. Host RAM is also needed for the server
and loading buffers; 26.82 GiB is not a complete host-memory requirement.
Keep offload enabled for this one-96-GB-GPU recipe. Moving that table back into
GPU memory would exceed its available budget before allocating KV cache.

With two GPUs and sufficient memory, device-resident PLE can be selected for a
separate experiment:

```bash
VLLM_PLE_CPU_OFFLOAD=0 GPU0=0 GPU1=1 PORT=8000 \
  docker compose -f qwen38-flash-next.compose.yml --profile tp2 up -d
```

Status of device-resident PLE: **implemented, not qualified**. No
offload-on/off speed comparison is claimed here. Changing this environment
setting recreates the serving container; use it only in a maintenance window.
The native override `--additional-config '{"ple_table_memory":"mapped_host"}'`
or `'{"ple_table_memory":"device"}'` takes precedence over the environment
setting when supplied to vLLM.

## Precision, backends and cache

The checkpoint combines NVIDIA 4-bit floating-point weights (NVFP4) with
other precisions. BF16 denotes bfloat16; W4A16 denotes 4-bit weights with
16-bit activations. Qwen sparse attention is abbreviated QSA. The precision
switches below control the vocabulary projection, not the entire model.

| Component | Recipe setting |
|---|---|
| Checkpoint | Mixed ModelOpt quantization; the NVFP4 name does not mean every tensor uses 4-bit precision |
| Target vocabulary head | BF16, preserved by `VLLM_MXFP8_LM_HEAD=0` |
| MTP draft vocabulary head | Private NVFP4 weight copy, selected by `VLLM_MTP_NVFP4_LM_HEAD=1` |
| Draft-head activations | BF16, enforced by `VLLM_LM_HEAD_A16=1`: W4A16, not W4A4 |
| Linear layers / mixture-of-experts layers / recurrent decode | B12X |
| Qwen sparse attention | Complete B12X QSA operation |
| Gated DeltaNet recurrent prefill | FlashInfer, not GLM's FlashKDA backend |
| Attention KV / recurrent state | FP8 / FP32 |
| Model runner / CUDA graphs | V2 / `FULL_AND_PIECEWISE`, capture through 64 rows |
| Scheduling | 6,019 batched tokens, 16 sequences, OMP2 |
| Maximum request context | 262,144 tokens, including generated tokens |
| GPU prefix cache | Enabled; repeated prompts and shared SYSTEM/developer prefixes qualified |
| LMCache RAM/disk restore | Installed in the image, but **not enabled or qualified for Qwen** |
| Image/video inputs | Disabled by this text-only recipe; not qualified here |

**Keep `VLLM_LM_HEAD_A16=1`.** Both the shared image and the Compose recipe
select it. The alternative value zero produced near-zero MTP draft acceptance
at C8/C16 in three diagnostic sweeps; value one restored acceptance without
changing the target vocabulary head. The W4A4 kernel-level cause is not
established, so W4A16 remains the qualified configuration.

R33 TP1 qualification reports **857,047 usable logical KV tokens**, versus
859,808 in its R32 control (−0.32%, including boot allocation variation).
This is a pool shared by requests,
not the maximum context of one request. A nominal physical-blocks-times-page
calculation reported about 13.1 million; that is **not usable capacity** for
this hybrid cache. Trust the engine's logical-capacity startup line. The
requested block size is 64, but the measured effective hybrid pages are 3,008
tokens; the recipe does not manually force a different geometry.

## Measured llmbench and Sieve performance

### R33 NVFP4 prefill: one GPU with VRAM +6000

Status: **qualified for these bounded cells**. The same physical RTX PRO 6000
Blackwell Workstation GPU, 600 W limit, graphics offset zero and **VRAM +6000**
is used sequentially for both arms. TP1/MTP3, FP8 KV, CPU PLE offload, OMP2,
6019-token budget, 16 sequences and full-and-piecewise graphs through 64 rows.
Sampling is **temperature 1, top-p 0.95, top-k 20**, normal reasoning, EOS
respected. These are not the stock-clock/xhigh Sieve results below.

| Measurement | R32 | R33 packaged image | Change |
|---|---:|---:|---:|
| Uncached 32K prefill, HTTP-wall input tok/s | 15,707.42 | 17,192.95 | **+9.46%** |
| Uncached 32K prefill, engine-accounted input tok/s | 15,865.89 | 17,390.95 | +9.61% |
| C1/context 0 output tok/s | 195.06 | 199.69 | +2.37% |
| C1 verifier steps/s | 97.45 | 96.79 | −0.67% |
| C8/context 0 aggregate output tok/s | 736.33 | 732.85 | −0.47% |
| C8 aggregate verifier steps/s | 378.79 | 384.38 | +1.48% |

Prefill excludes two warmups and retains five requests of exactly 32768 input
tokens and one output token, all with zero cache hits. Decode uses a ten-second
warmup and one thirty-second cell per concurrency. Acceptance changes with
sampling, so these short decode observations are not a general speedup or
statistical equivalence claim. A source-overlay prototype gave 17,211 input
tok/s; the table uses the packaged image's independently repeated result.

The image passes 68 cold/repeated requests, six shared-instruction cases and
38 post-prefill logprob requests. No checkpoint values change. B12X proves
exact equality of immutable expert input scales and shares input quantization
across routed experts before separate projections. Target head BF16 and the
private NVFP4 draft head remain unchanged. TP2's width-320 expert partition is
not split-tile eligible; no TP2 gain is claimed. LMCache remains unqualified
for Qwen. [Source, raw samples, numerical limits and complete R33 changelog](glm-5.3-flash/validation/fp4-prefill-filesystem-r33.md).

### Historical stock-clock engine and Sieve comparisons

The shared R29 composition passes a same-GPU TP1/MTP3 comparison against
R28.1: three warmed C1 repeats have median output **173.66 → 177.27 tok/s
(+2.08%)** and verifier rate **85.21 → 85.37 steps/s (+0.18%)**. C8 and
32K prefill differ by less than 1%. These bounded measurements do not establish
a general speedup; an initial lower C1 sample is retained in the
[shared-image qualification report](glm-5.3-flash/validation/shared-serving-r29.md).
The engine comparison and ten-run Sieve table below are explicitly R28.1
measurements, not repeated or relabelled R29 results.

Status: **qualified for the measured vLLM and SGLang cells**, with deployment
differences stated below. The mratsim turbo column is **research-only,
operator-reported evidence**, not an independently qualified comparison.

The measured vLLM and SGLang deployments each use **TP1: one 96 GB RTX PRO 6000
Blackwell Workstation GPU**, a 600 W limit, **stock graphics and VRAM offsets
of zero**, the same NVFP4 checkpoint, three speculative draft tokens, FP8
attention KV and host PLE offload. These are **not +6000 results**.

**Decode and Sieve sampling: temperature 1, top-p 0.95, top-k 20,
reasoning `xhigh`, EOS respected.** The
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
0.6.1 decode sweep uses context `0`: the same short chat prompt renders to
119 input tokens on both servers. Each cell has 15 seconds of warmup followed
by one 30-second measurement. C2 and above report **aggregate output across
clients**, not speed per chat.

| Measurement, tok/s | vLLM R28.1 / MTP3 | SGLang FlashInfer / NEXTN3 | SGLang mratsim turbo, reported |
|---|---:|---:|---:|
| C1, context 0 | **172.8** | 152.5 | 154.4 |
| C2, context 0 | **304.3** | 267.5 | Not measured |
| C4, context 0 | **485.4** | 446.8 | Not measured |
| C8, context 0 | **632.4** | Not measured | Not measured |
| C16, context 0 | **944.5** | Not measured | Not measured |
| Uncached 32K prefill, median input rate | 14,813 | **15,583** | 15,281 |
| Sieve coding, median of 10 runs | **239.8** | 205.3 | Not measured |
| Sieve coding, minimum | 222.1 | 176.2 | Not measured |
| Sieve coding, maximum | 265.1 | 232.6 | Not measured |

The measured SGLang configuration is capped at four active requests; C8/C16
were not measured or inferred from queued clients. vLLM runs on physical GPU0
and SGLang on GPU1, without swapping cards. Their differing recurrent-state
precision, draft-head precision, token budgets and cache settings are recorded
in the [engine-comparison report](qwen38-flash-next/validation/tp1-engine-comparison.md).
The table compares those deployments, not isolated engine implementations.

Numerically, vLLM C1 is **13.3% higher** than measured SGLang FlashInfer and
**11.9% higher** than the reported turbo result. SGLang FlashInfer prefill is
**5.2% higher** than vLLM; the reported turbo prefill is **3.2% higher**.
Turbo clocks, sampling, hardware, source identity and benchmark repetition
count were not independently checked. Its two values were supplied by the
serving operator on 2026-09-08; they are not a matched A/B or a significance claim.

Cold prefill uses exactly 32,768 identical token IDs and one output token,
two discarded warmups and five measurements, all with zero cache hits.
The table uses the complete HTTP request time, including first-token work;
it is **not pure GPU prefill time**. vLLM's separate engine-accounted median
is 14,960.7 input tok/s and must not replace the comparable HTTP metric.

Sieve uses one discarded warmup and ten sequential C1 requests with a
2,000-output-token cap. Its rate includes reasoning and answer tokens,
excluding time to first text. All twenty measured requests finish by EOS.
Generated Python was not executed: this is throughput evidence, not code-quality
validation. The [measurement summary and Sieve samples](qwen38-flash-next/validation/tp1-engine-comparison.json)
retain every measured decode cell and all ten Sieve rates per engine.

TP2, no-MTP and PLE-offload-disabled throughput remain **unqualified**
for this image. The separate
[TP1 qualification report](qwen38-flash-next/validation/r28.1-tp1.md) preserves
the three-repeat same-GPU image comparison and bounded cache/output checks.
Those samples are not pooled with the engine-comparison measurements above.

### Run the TP1 qualification benchmark

The pinned command below reproduces the configuration of the separate
TP1 qualification report. Use an otherwise idle server with Python and `uv`:

```bash
curl -fL -o llm_decode_bench.py \
  https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py

for QWEN_REPEAT in 1 2 3; do
  uv run --no-project --with httpx --with rich --with psutil python llm_decode_bench.py \
    --host 127.0.0.1 --port 8000 --model Qwen3.8-Flash-Next \
    --contexts 0 --concurrency 1,8,16 --duration 30 \
    --decode-warmup-seconds 10 --max-tokens 32768 \
    --temperature 1 --respect-eos --skip-prefill \
    --display-mode plain --no-resume --output "qwen-decode-$QWEN_REPEAT.json" < /dev/null
done
```

Use a fresh output directory to preserve previous samples. Keep the same GPU,
clock offsets, model revision, launch arguments and benchmark revision for an
A/B comparison. Check errors, filled concurrency and loop flags before using
a throughput number. The source revision in the download URL freezes the
measured benchmark; the serving model remains an ordinary Hugging Face name.
