# DeepSeek-V4.1-Flash

Serve `deepseek-ai/DeepSeek-V4.1-Flash` using the `ds41-flash` profile in the
[shared Docker guide](../docs/unified-vllm-docker.md). It supports native
text/vision input, B12X kernels and the checkpoint's embedded DSpark draft.
It is distinct from [DeepSeek V4 text](deepseek-v4-flash.md) and
[V4 Vision](deepseek-v4-flash-vision.md).

## Start the server

This starts TP4/DCP1 with adaptive DSpark K7 and disk-backed Engram tables:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name ds41 --init --restart unless-stopped \
  --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -v lil-huggingface:/root/.cache/huggingface -v ds41-runtime:/cache \
  -e PROFILE=ds41-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=4 -e PORT=8000 "$IMAGE"
```

The API model is `DeepSeek-V4.1-Flash` on port 8000. The checkpoint downloads
into the shared HF volume. Check readiness with `docker logs -f ds41`.
The io_uring loader needs the syscall permission above; use a trusted image.

Add native options **after `"$IMAGE"`**:

| Choice | Arguments |
|---|---|
| Keep n-gram tables in RAM | `--engram-table-memory ram` |
| Read tables from SSD | `--engram-table-memory disk` (default) |
| No speculation | `--mode off` |
| Restrict context | `--max-model-len 131072` |
| Change request slots | `--max-num-seqs 16` |

The default is 32 slots and an automatically sized context. Change GPU IDs,
`TP` and `PORT` before the image name. MTP/DFlash2 are not DS4.1's DSpark mode.
See the [Karmic Kraken benchmark table](../benchmarks/karmic-kraken-serving.md)
for the comparison with RAM Engram; disk placement has separate functional
cache checks and is not the speed measured below.

## RAM versus SSD ngrams

Engram contains learned n-gram model tables. It is **not** KV cache, prefix
caching or an LMCache tier. Target/draft transformer weights remain on GPU.

| Placement | Behavior | Capacity requirement |
|---|---|---|
| `disk` | Native io_uring reads selected checkpoint rows from local SSD into bounded staging buffers | Fast local SSD/NVMe; loader, OS cache and staging still need RAM |
| `ram` | Complete tables in pinned, GPU-mapped host RAM | Historical TP4 checkpoint accounting: 188.83 GiB for tables, plus loader/server/OS memory |

RAM allocation failure does not silently choose disk. Generic CPU model
offload remains disabled. Optional `--engram-disk-resident-scales` and
`--engram-projection-tp` are not enabled by default and have no whole-model
speed claim in this qualification. `CACHE_MODE=lmcache` independently enables
CPU/disk prefix storage; selecting RAM Engram does not enable it.

## Common settings

Keep TP4/DCP1 and adaptive DSpark K7 for this recipe. Additional capacity
settings go **before `"$IMAGE"`**:

| Setting | Default / recommendation | Example override |
|---|---|---|
| Active requests | 32 | `-e MAX_NUM_SEQS=16` |
| Prefill budget | 4096 tokens | `-e MAX_NUM_BATCHED_TOKENS=4096` |
| Context | Automatic, `-1` | `-e MAX_MODEL_LEN=131072` |
| GPU memory fraction | 0.95 | `-e GPU_MEMORY_UTILIZATION=0.93` for more working space |
| JIT monitor | `warn` for serving | `-e JIT_MONITOR_MODE=error` only for strict warmup diagnostics |
| External prefix storage | GPU-only | `-e CACHE_MODE=lmcache` with [RAM/disk settings](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload) |

B12X handles attention, MoE and dense kernels. Full-and-piecewise decode
graphs are enabled; breakable prefill is off. Prefix caching is on even though
the default retention interval is `0`. Keep the model's own cache geometry.

Draft proposals are greedy with standard rejection; target requests still
sample at temperature 1/top-p .95. Default reasoning is `high`, with the
checkpoint's `low=50`, `high=75`, `max=100` budget mapping.

The native cache is heterogeneous despite the CLI's `fp8` label: MXFP8
sliding-window payloads, NVFP4 indexed payloads and index/state groups. Do not
describe it as a uniform FP8 cache. The GLM/Qwen request-boundary recurrent
adapter does not apply to DeepSeek's attention-cache structure.

Use `--adaptive-verification=false` to disable DSpark trimming, not DSpark
itself. `--adaptive-verification-cost-scale` changes trimming cost; neither
alternative is timed here. A request can override the reasoning budget with
`"chat_template_kwargs":{"reasoning_effort":50}`.

## Measured performance

Four RTX PRO 6000 **Max-Q**, **VRAM +6000**, automatic graphics clocks;
TP4/DCP1, RAM Engram, adaptive DSpark K7, 4096-token budget, 32 slots,
131,072 context cap, GPU-only cache and temperature 1/top-p .95.
Five warmed 30-second windows per cell; context-zero decode and uncached
nominal-32K prefill measured from client TTFT. C8 is aggregate.

| Metric | Saved JJ R38 | Karmic Kraken | Change |
|---|---:|---:|---:|
| C1 output | 230.6 tok/s | 249.1 tok/s | +8.01% |
| C8 aggregate output | 748.3 tok/s | 805.5 tok/s | +7.65% |
| 32K prefill | 17,503 tok/s | 17,982 tok/s | +2.74% |
| C1 verifier rate | 89.87 steps/s | 98.58 steps/s | +9.69% |

Text, image and repeated/changed-prefix checks pass. The KK server reports
4,727,748 logical KV tokens shared by requests, not a per-request context
limit. These are RAM-Engram results; selecting disk can change throughput.
[Image versions, configuration and all samples](../benchmarks/karmic-kraken-serving.md).

### Breakable prefill

`-e VLLM_USE_BREAKABLE_CUDAGRAPH=1` before the image captures graph segments around dynamic
attention/cache operations, which still execute outside those segments. It is
not a single FULL graph for all prefill work. Decode graphs remain available
when this option is off.

It remains off by default: additional graph memory can reduce KV capacity.
The [separate graph-memory comparison](../benchmarks/prepared-b12x-contracts/#breakable-prefill-ds41)
records that trade-off on its own image; it is not the throughput table above.

## Historical releases and source review

- [Archived recipes and measurements](../archive/serving-guides/README.md)
  preserve the preceding guides, API checks and stock Workstation tables.
- [Community R38 deployment and measurement archive](deepseek-v4.1-flash-community-r38.md):
  R37/R38 measurements, Sieve, source/dependency locks and exact checkpoint scope.
- [R37 record](deepseek-v4.1-flash/r37/release.md) and
  [R36 record](deepseek-v4.1-flash/r36/release.md), including separately labelled clocks.
- [Shared runtime dependency corrections](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b3fe0afe1621273059fb19dee1034e2272043a55/runtime/DEPENDENCIES.md):
  PyTorch/CuTe corrections belong to image assembly, not model startup patches.
- [Issue #808](https://github.com/local-inference-lab/vllm/issues/808):
  open source PRs, integration status and qualification limits.
