# Qwen3.8-Flash-Next On Two RTX PRO 6000 Blackwell GPUs

This runbook provides two vLLM recipes for two 96 GiB RTX PRO 6000 Blackwell
GPUs connected through PCIe, using Tensor Parallelism (TP) 2:

| Recipe | Checkpoint | Image path | Status |
|---|---|---|---|
| [FP8](#fp8-recipe) | `Qwen/Qwen3.8-Flash-Next-FP8` | Official preview image | Boot and API smoke test validated |
| [NVFP4](#nvfp4-recipe) | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | Same image plus local PLE patch | Boot and API smoke test validated |

These are reproducible launch configurations, not throughput benchmarks.
They were validated with vLLM image
`vllm/vllm-openai:qwen38-flash-next`, reported by the server as
`0.1.dev20073+g8e685d198`. Revalidate both recipes after changing the image,
model revision, driver, or vLLM release line.

## Shared Requirements

- Two NVIDIA RTX PRO 6000 Blackwell GPUs with 96 GiB VRAM each.
- Docker with the NVIDIA Container Toolkit and access to GPUs 0 and 1.
- At least 62 GiB of container memory available for the Position Learning
  Enhancement (PLE) CPU offload worker. The N-gram embedding is materialized
  in host RAM, not paged to an SSD by this recipe.
- A Hugging Face cache mount. Authenticate before the first download when the
  model repository requires it, for example with `huggingface-cli login`.

Both recipes publish port 18005 and use both GPUs. Do not run them at the same
time.

## FP8 Recipe

The official FP8 checkpoint selects vLLM's native FP8 PLE embedding path. It
does not need a Python source override or a checkpoint conversion.

```bash
docker compose -f compose/qwen38-flash-next-fp8-tp2.yml up -d
```

The recipe intentionally retains three operational settings:

- `VLLM_PLE_CPU_OFFLOAD=1` keeps the large PLE N-gram table in a dedicated CPU
  worker so the GPUs retain usable KV cache space.
- `SYS_PTRACE` allows the PyTorch CUDA IPC registration used by that worker.
  Without it this environment fails with `pidfd_getfd: Operation not permitted`.
- `--disable-custom-all-reduce` uses NCCL/PyNCCL. The custom all-reduce kernel
  fails during CUDA Graph capture on this TP2 SM120 configuration.

The recipe leaves Torch Inductor compilation and FlashInfer autotuning enabled.
On the validation system, default `torch.compile`, FlashInfer autotune, and
CUDA Graph capture completed successfully.

## NVFP4 Recipe

The RadixArk NVFP4 checkpoint is hybrid: its outer quantization is ModelOpt
NVFP4, while the PLE N-gram embedding remains FP8 and includes
`ngram_embedding.weight_scale`. The preview image chooses the FP8 PLE loader
only from the outer `Fp8Config`, so it otherwise creates only
`ngram_embedding.weight` and fails during checkpoint loading.

The recipe builds a tiny derived image that applies
[qwen38-flash-next-nvfp4-ple-fp8.patch](../patches/qwen38-flash-next-nvfp4-ple-fp8.patch).
The patch uses `ple_embedding_dtype=float8_e4m3fn` to select the FP8 PLE
embedding method and register the required scale parameter.

```bash
docker compose -f compose/qwen38-flash-next-nvfp4-tp2.yml build
docker compose -f compose/qwen38-flash-next-nvfp4-tp2.yml up -d
```

This recipe keeps the conservative compilation configuration that was required
for the tested NVFP4 checkpoint: Inductor compilation is disabled while decode
CUDA Graphs remain enabled. It also disables FlashInfer autotuning. Those two
settings are specific to this NVFP4 path; do not copy them into the FP8 recipe
without reproducing an issue.

## Verify The Server

```bash
curl -fsS http://127.0.0.1:18005/health
```

An OpenAI-compatible smoke test is:

```bash
curl -fsS http://127.0.0.1:18005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash",
    "messages": [{"role": "user", "content": "Reply with OK only."}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Operational Notes

- `PYTORCH_ALLOC_CONF=expandable_segments:True` is a low-risk CUDA allocator
  fragmentation guard. It is not required for an initial boot, but helps avoid
  failures where aggregate free VRAM exists without a sufficiently large
  contiguous allocation.
- The recipes use `max-model-len=262144`, `max-num-seqs=4`, MTP with three
  speculative tokens, prefix caching, Qwen reasoning parsing, and Qwen Coder
  tool-call parsing. Lower these limits before assuming a different host has
  equivalent capacity.
- The NVFP4 patch is tied to the preview image file layout. Rebuild and review
  the patch when upstream model support changes.
