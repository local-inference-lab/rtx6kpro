# SGLang and B12X changes for MiMo-V2.5-Pro

Date: 2026-05-10

This page records the runtime changes needed to make the TP=8 MiMo-V2.5-Pro NVFP4/MXFP8 checkpoint run coherently with SGLang, B12X, MTP/EAGLE, and CUDA graph target verify.

## Runtime baseline

Base image:

```text
docker.io/lukealonso/sglang-cuda13-b12x:w4a16
```

The validated image with patches baked in:

```text
voipmonitor/sglang:mimo-v25-pro-tp8-microrecip-20260510
sha256:b634da4afa79fb4bb8bb06c67938c8f3a816af96bb31d307605c2a56f35fa444
```

The image is not a generic upstream SGLang image. It contains targeted SGLang and B12X overlays listed below.

## Patch inventory

| File in image | Why it was changed |
|---|---|
| `/opt/sglang/python/sglang/srt/models/mimo_v2.py` | MiMo-V2.5-Pro architecture handling, MTP target/draft loading behavior, and debug instrumentation for layer/MoE state comparisons. |
| `/opt/sglang/python/sglang/srt/speculative/multi_layer_eagle_worker.py` | MTP/EAGLE runtime adjustments used by the MiMo serving path. |
| `/opt/sglang/python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` | Target-verify graph/eager comparison instrumentation and ordering controls used to isolate corruption. |
| `/opt/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py` | CUDA graph replay fixes, including restaging MiMo hybrid/SWA cache locations for target verify. |
| `/opt/sglang/python/sglang/srt/layers/attention/b12x_backend.py` | B12X attention metadata/workspace alignment for graph replay versus eager decode. |
| `/opt/sglang/python/sglang/srt/layers/quantization/unquant.py` | BF16 unquantized linear path stabilization for graph/eager consistency. |
| `/opt/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py` | ModelOpt mixed NVFP4/MXFP8 loading, B12X MoE output-buffer handling, and split-topk proof-of-cause hooks. |
| `/opt/sglang/python/sglang/srt/layers/communicator.py` | Debug context propagation through communicator/allreduce boundaries. |
| `/opt/sglang/python/sglang/srt/layers/layernorm.py` | Native RMSNorm isolation switch used during graph/eager debugging. |
| `/opt/sglang/python/sglang/srt/layers/sampler.py` | PyTorch sampler path used to avoid the earlier sampling corruption path. |
| `/opt/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py` | MiMo chat/serving behavior used by the OpenAI-compatible endpoint. |
| `/opt/venv/lib/python3.12/site-packages/b12x/integration/tp_moe.py` | B12X MoE integration instrumentation and deterministic split-topk proof-of-cause path. |
| `/opt/venv/lib/python3.12/site-packages/b12x/moe/fused/micro.py` | Production fix candidate: allow the direct micro MoE kernel with reciprocal scales and invert scales inside the kernel. |

## Required runtime settings

These settings were part of the validated runtime:

| Setting | Value |
|---|---|
| `--tp-size` | `8` |
| `--json-model-override-args` | `{"architectures":["MiMoV2FlashForCausalLM"]}` |
| `--page-size` | `64` |
| `--quantization` | `modelopt_mixed` |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--attention-backend` | `b12x` |
| `--moe-runner-backend` | `b12x` |
| `--fp4-gemm-backend` | `cutlass` |
| `--fp8-gemm-backend` | `flashinfer_cutlass` |
| `--sampling-backend` | `pytorch` |
| `--enable-multi-layer-eagle` | enabled |
| `--speculative-algorithm` | `EAGLE` |
| `--speculative-num-steps` | `3` |
| `--speculative-eagle-topk` | `1` |
| `--speculative-num-draft-tokens` | `4` |
| `--cuda-graph-max-bs` | `8` |
| `--disable-piecewise-cuda-graph` | enabled |

B12X attention requires `--page-size 64`. With page size 1, startup fails with:

```text
ValueError: b12x attention backend requires page_size=64, got 1
```

## Debugging timeline

### 1. Checkpoint correctness came first

The BF16 checkpoint was validated before quantization with `tools/mimo_forward_smoke.py` for at least 256 generated tokens. The corrected BF16 smoke answered:

```text
The capital of France is **Paris**
```

The final NVFP4/MXFP8 checkpoint was then validated without MTP/EAGLE and with CUDA graph modes disabled before target-verify graph work continued.

### 2. Sampling corruption was isolated

Early long generations could start coherently and later degrade into multilingual or garbled text. BF16 KV cache did not explain it. Switching to the PyTorch sampler path made greedy and controlled sampled tests coherent under the validated configuration.

Current runtime uses:

```bash
--sampling-backend pytorch
```

### 3. BF16 dense linear graph/eager mismatch

MiMo has unquantized BF16 dense paths in addition to the routed FP4 MoE paths. During CUDA graph capture, SGLang used a compiled/autotuned BF16 linear path. After warmup, eager target-verify could fall back to `F.linear`, meaning graph replay and eager comparison were not executing the same BF16 linear implementation.

The runtime therefore sets:

```bash
SGLANG_DISABLE_AUTOTUNED_LINEAR_AFTER_WARMUP=0
```

This keeps the method stable across graph capture/replay/eager. The downside is that new large dynamic prefill shapes can trigger Torch Inductor autotune during inference. That is a performance caveat, not a checkpoint-integrity failure.

### 4. B12X attention graph metadata alignment

The target-verify compare originally showed graph/eager B12X attention metadata mismatches. The B12X attention overlay aligns graph-capture workspace metadata with the eager decode path, including KV chunk sizing and workspace capacity for the dense eager-aligned replay plan.

This is why the runtime image includes:

```text
/opt/sglang/python/sglang/srt/layers/attention/b12x_backend.py
```

### 5. SWA cache-location replay fix

In MiMo hybrid/sliding-window attention, graph replay must restage both full KV and SWA KV cache locations. The SGLang CUDA graph runner patch restages the SWA output cache locations used by the target-verify graph.

This is why the runtime image includes:

```text
/opt/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py
```

### 6. B12X NVFP4 MoE accumulation mismatch

After attention metadata and cache-location issues were addressed, graph/eager differences remained in the B12X FP4 MoE expert-output path.

Evidence:

- router inputs matched
- router logits matched
- top-k ids matched
- top-k weights matched
- first nonzero difference appeared at MoE expert output before TP allreduce
- forcing `B12X_MOE_FORCE_A16=1` made graph/eager exact
- SGLang-side split-topk made graph/eager exact
- B12X-side split-topk made graph/eager exact

The B12X static/dynamic FP4 MoE path scatters routed expert outputs using BF16 atomic add. Multiple top-k expert contributions can therefore be order-dependent between graph replay and eager execution.

### 7. Preferred fix: B12X direct micro MoE with reciprocal scales

B12X already had a direct micro MoE kernel for small shapes such as target-verify/decode batches. That kernel accumulates top-k contributions deterministically inside one kernel instead of using the static/dynamic BF16 atomic scatter path.

It was not selected for this checkpoint because the micro-kernel support check rejected:

```text
input_scales_are_reciprocal=True
```

SGLang's ModelOpt NVFP4 call uses reciprocal input scales, so the B12X micro MoE patch:

- allows the micro kernel when `input_scales_are_reciprocal=True`
- inverts the relevant input/down scales inside the micro kernel
- keeps A16 off
- avoids the split-topk multi-launch fallback

The important patched file is:

```text
/opt/venv/lib/python3.12/site-packages/b12x/moe/fused/micro.py
```

With the micro reciprocal-scale patch only, target-verify graph/eager compare became exact:

| Compare | Result |
|---|---|
| logits | `max_abs=0.0`, `mean_abs=0.0`, `top_mismatch=0/4` |
| hidden | `max_abs=0.0` |
| layer stats/residual/hidden | `all_match max_abs=0.0` |

## Validated production-style result

The patched production-style container was run on port `30004` with:

- TP=8
- MTP/EAGLE enabled
- target CUDA graphs enabled for `bs=[1..8]`
- piecewise CUDA graph disabled
- A16 MoE disabled
- no graph/eager debug compare env
- no split-topk env

Smoke:

- HTTP 200
- exact sentence `The capital of France is Paris.`
- coherent checkpoint-integrity paragraph
- no CJK/replacement-token corruption

Warm 8-way soak:

- 8/8 requests returned HTTP 200
- 4,042 completion tokens
- 6,304 total tokens
- 31 s wall time
- health remained HTTP 200

## What not to confuse with real failures

During startup and new dynamic-shape prefill, Torch Inductor may log lines like:

```text
No valid triton configs. OutOfMemoryError: out of resource
```

These are rejected autotune candidates, not necessarily fatal GPU OOMs. A real failure should be correlated with process exit, traceback, CUDA device-side assert, NaN/Inf detection, scheduler shutdown, or `/health` failure.
