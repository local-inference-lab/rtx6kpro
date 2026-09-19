# GLM Spark TP2: experimental R2 archive

Release status: **research-only**, intended for experimental community testing.
This deployment targets two 96 GiB NVIDIA RTX PRO
6000 Blackwell Workstation GPUs with Tensor Parallelism 2 (TP2) and Decode
Context Parallelism 2 (DCP2). It uses the Spark checkpoint, three-token
Multi-Token Prediction (MTP3), FP8 KV cache and one image per prompt. GPU-local
caching is the default; engine-driven LMCache is optional.
It does not replace the [four-GPU community deployment](glm-5.3-flash.md).
The image platform is `linux/amd64`. “Spark” names the checkpoint variant;
these measurements do not qualify ARM-based DGX Spark hardware.

## Start serving

```bash
docker run -d --name glm-spark-tp2 --init \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  -v hf-cache:/root/.cache/huggingface \
  -v glm-spark-tp2-cache:/cache \
  localinferencelab/vllm:jovian-judgement-community-tp2-experimental-20260913-r2
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
| cuBLAS workspace | `CUBLAS_WORKSPACE_CONFIG=:4096:1`, 4 MiB per handle |

Client-supplied sampling parameters take precedence. Explicit vLLM arguments
can override launcher arguments; changed image counts, scheduler budgets,
context sizes and concurrency limits require independent memory validation.
`DRY_RUN=1` prints the complete command without loading weights.

## Optional RAM and filesystem cache

Add these options **before the image name** in the serving command:

```bash
-e CACHE_MODE=lmcache \
-e LMCACHE_L1_SIZE_GB=64 \
-e LMCACHE_L2_ROOT=/cache/lmcache-l2
```

LMCache uses engine-driven asynchronous shared-memory transfers. The vLLM
workers own GPU gather/scatter; the sidecar is CPU-only and creates no separate
GPU context. The profile preserves FP8 target KV, the 1M context limit and the
4 GiB GPU KV budget. GPU-only caching remains the default.

The 64 GiB L1 arena is preallocated pinned host shared memory, not a lazy
maximum. Rank mappings share that physical arena. With `--ipc host`, ensure
host `/dev/shm` has capacity for the arena plus transfer buffers; Docker's
`--shm-size` setting does not resize host shared memory. Choose a smaller
`LMCACHE_L1_SIZE_GB` if necessary. Set `LMCACHE_L2_ENABLED=0` for RAM-only
storage. Otherwise the `/cache` volume preserves filesystem objects across
container replacement. Administrative sidecar listeners default to loopback.

LMCache objects cover 4,096 global tokens; attention pages contain 2,048 tokens
per DCP rank. Request-boundary bundles carry their recurrent endpoint, so the
3,072-token scheduler budget need not equal the storage-object size. Aligned
transfers retain their alignment and budget checks.

Persistent checkpoints authenticate model content, serving source and layout.
Incompatible objects are safe misses. Locally mounted checkpoint files are
hashed at startup; this can delay the first startup log while weights are
read. Keep checkpoint files immutable while serving.

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

The NVFP4 MTP experts use B12X W4A16; this profile does not claim W4A4 for
every operator. BF16 projections can use cuBLAS.

The default explicitly requests `MAX_MODEL_LEN=1048576` and
`KV_CACHE_MEMORY_BYTES=4294967296` per GPU. It does not silently shorten the
context. `GPU_MEMORY_UTILIZATION` does not resize an explicit KV allocation.
Automatic profiling requires `KV_CACHE_MEMORY_BYTES=auto`; accepting a shorter
context additionally requires `MAX_MODEL_LEN=-1`.

Bounding cuBLAS workspace to 4 MiB saves 140 MiB of live allocation per rank
in the stock-clock comparison below. It does not guarantee that every GPU or
request shape fits. Community Max-Q allocation failures and a separate
dense-projection cuBLAS failure were not reproduced on these Workstation GPUs.
Reporter-hardware confirmation remains necessary; R2 is not a confirmed fix
for every reported cuBLAS crash.

One image means one image across the complete API request, including previous
conversation turns. Native `--limit-mm-per-prompt` overrides can increase that
count, but multiple maximum-size images combined with 1M context are outside
this qualification.

The memory mechanisms can also apply to TP4, but their magnitude depends on
sharding and workload. **No TP4 memory or performance improvement is qualified
by the TP2 measurements.** The TP2 NCCL configuration is not a TP4 recommendation.

## Qualification scope

Status: **qualified** for the following bounded correctness checks on the R2
image, two stock-clock RTX PRO 6000 Workstation GPUs, TP2/DCP2 MTP3, 4 GiB
KV per rank and the defaults above. Both cache profiles reported 1,051,958
usable KV tokens after hybrid-state accounting.

| Check | Result |
|---|---|
| GPU-local cold image input, 1,048,320 tokens including native image features | Correct pigeon answer in 143.0 s; no prefix/image-cache hits; minimum sampled free VRAM 232.31 MiB per GPU |
| Three decoders plus cold 32K image after the long image input | Correct image answer; all decoders progressed; maximum observed gap 0.552 s; no errors |
| Post-image text cache, 9,751/9,752/9,753-token inputs and 536-token continuation | 27 checks passed |
| LMCache cold text, 1,048,320 tokens | 139.687 s |
| RAM restore of that text | 0.808 s; all 1,048,320 tokens restored; zero recompute |
| Filesystem restore after both services restart | 1.279 s; all 1,048,320 tokens restored; zero recompute |
| Literal-document answers and shared-instruction reuse across cache tiers | All 16 checks passed; shared instruction endpoint restores 11,340 tokens and computes only the 11-token suffix |
| Packaging and source tests | Two layers; clean Git sources; authenticated launchers; 216 recipe tests passed |

The million-token synthetic text checks transfer attribution and output
equality, not language quality. Literal-document checks separately require
exact answers. Filesystem restore includes the OS page cache and must not be
interpreted as cold physical-disk throughput.

Exact multimodal request-endpoint restore remains **unsupported** by the
semantic adapter. Aligned fallback may restore most tokens while recomputing
a tail. Native vision capacity is not a claim of zero-recompute image replay.
DFlash, video, multiple maximum-size images, other GPU models and extended
soak testing are outside this experimental qualification.

### R2 performance against a same-session R1 control

Both immutable images were run sequentially on the same physical GPU pair,
stock VRAM/graphics offsets, TP2/DCP2 MTP3 and a 3,072-token budget.
Temperature was 1.0 and top-p 0.95. Prefill used one warmup and 30 seconds of
unique cold 32K inputs. C1 used three independent 30-second cells, each with
15 seconds of warmup. Development cache-reset endpoints were enabled on both
test servers; they are not enabled in the public command above.

| Measurement | R1 control | R2 | Change |
|---|---:|---:|---:|
| 32K prefill, input tok/s | 10,605.8 | **10,648.9** | **+0.41%** |
| C1 MTP3 median output, tok/s | 163.38 | **170.77** | **+4.52%** |
| C1 median verifier, steps/s | 67.171 | **67.547** | **+0.56%** |
| C1 output samples, tok/s | 163.38 / 161.37 / 168.20 | 167.30 / 170.77 / 172.87 | — |
| C1 verifier samples, steps/s | 67.171 / 67.069 / 67.535 | 67.326 / 67.547 / 67.701 | — |

Output throughput depends on stochastic proposal acceptance. These three
samples establish no sustained kernel-speedup claim; verifier ranges overlap.
The comparison passed the bounded 2% prefill/verifier regression gate.
An earlier stock-clock control measured 70.224 steps/s, but that rate was not
reproduced by either image in this session. It is not used as evidence of an
R2-specific loss. No operating-condition cause was established.

The [R2 qualification record](glm-5.3-flash/tp2-experimental-r2-qualification.json)
retains image identities, samples, cache checks and limitations. The
[artifact audit](glm-5.3-flash/tp2-experimental-r2-artifact-audit.json)
confirms unchanged component implementations and authenticated launcher files.

## Historical R1 performance

The R1 publication used a different physical pair with **VRAM +6000**,
TP2/DCP2 MTP3, context 0, temperature 1.0 and top-p 0.95. These are not
stock-clock R2 results or a matched R2 comparison. The raw
[R1 qualification record](glm-5.3-flash/tp2-experimental-qualification.json)
retains all three decode samples, image/cache checks and artifact identities.

| Measurement | Median | Range / individual samples |
|---|---:|---|
| 32K prefill | **10,765.9 input tok/s** | Ten cold samples; every prompt token computed locally |
| C1 MTP3 output | **195.7 tok/s** | 184.5 / 202.3 / 195.7 |
| C1 verifier | **78.75 steps/s** | 75.50 / 78.87 / 78.75 |
| C4 aggregate output | **455.1 tok/s** | 446.2 / 464.6 / 455.1 |
| C4 aggregate verifier | **183.55 steps/s** | 178.84 / 187.24 / 183.55 |

## Source and packaging

Published R2 digest, verified by pulling from DockerHub:

```text
localinferencelab/vllm@sha256:c549afc8dc065fa63246618761ec2563af78ff70dae6409ed5c72acdc850f795
```

The [registry verification](glm-5.3-flash/tp2-experimental-r2-registry.json)
matches the tested image ID, source-lock hash and two filesystem layers.
The [source manifest](glm-5.3-flash/tp2-experimental-r2-source.lock) is also
available without pulling the image.

The image has two filesystem layers: a fixed runtime foundation and a complete
source installation. Its TP2 entrypoint is a metadata-only specialization;
CUDA, FlashKDA and B12X native artifacts are reused rather than rebuilt.

`/opt/glm53-flash/source.lock` identifies exact component commits, trees and
build-input hashes. Complete Git histories are included at
`/opt/glm53-flash/vllm`, `/opt/glm53-flash/b12x` and `/opt/lmcache/source`.
The source lock, not an assumption about a mutable branch, defines the image.
R2 additionally authenticates the recipe commit/tree separately from component
provenance. The [TP2 recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/fix/glm-spark-tp2-community-launcher/recipes/glm53)
contains the build and launcher sources.

## Stock-clock workspace comparison

The fixed-capacity diagnostic builds were compared on the same two RTX PRO
6000 Workstation GPUs, stock offsets, TP2/DCP2 MTP3, context 0, temperature 1.0
and top-p 0.95. Each of three independent decode cells had a 15-second warmup
and 30 seconds of measurement. Prefill used unique uncached 32K inputs.

| Measurement | 32 MiB workspace control | 4 MiB workspace | Change |
|---|---:|---:|---:|
| 32K prefill, input tok/s | 10,643.94 | 10,642.33 | −0.02% |
| C1 MTP3 output median, tok/s | 173.52 | 173.89 | +0.21% |
| C1 verifier median, steps/s | 70.224 | 70.176 | −0.07% |
| C1 output range, tok/s | 171.27–176.39 | 171.55–176.54 | — |

These overlapping samples do not establish a speedup. The R1 release's 195.7
tok/s result used VRAM +6000 on a different pair and is not a stock-clock
regression baseline. Final-image measurements are reported separately.

## R2 changes relative to experimental R1

- TP2 supports opt-in LMCache through the installed cache launcher, with a
  CPU-only sidecar and worker-owned asynchronous transfers.
- cuBLAS workspace is bounded to 4 MiB per handle without shrinking the 1M
  context or changing precision.
- Semantic checkpoint transfers support a 3,072-token model budget with
  4,096-token storage objects and 2,048-token per-rank attention pages.
- Explicit native context arguments are emitted once. Automatic KV/context
  fitting is opt-in rather than a silent capacity reduction.
- Source metadata authenticates recipe and launcher provenance independently.
  vLLM, B12X and LMCache implementations and checkpoint weights are unchanged.
