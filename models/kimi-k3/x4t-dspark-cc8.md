# Kimi K3 X4T with DSpark K7

This profile serves the lossless Kimi-K3 X4T checkpoint on 16 RTX PRO 6000 Blackwell 96 GB GPUs with eight concurrent requests, DCP16, FP8 KV and Red Hat DSpark K7. It uses selected online MXFP8 KDA/draft projections; vision and prefix caching are disabled. The KV pool holds **4,284,552 tokens** with a **1,000,000-token per-request limit**.

The [reproduction bundle](https://github.com/local-inference-lab/rtx6kpro/releases/tag/kimi-k3-x4t-dspark-k7-cc8-20260930) contains the exact serving wheels, dependency locks, launch scripts and source-reconstruction recipe. It does not contain model weights. The [merge list](https://github.com/local-inference-lab/vllm/issues/906) identifies each required vLLM/B12X PR.

## Build the image

Requires Linux x86-64, Docker with Buildx, NVIDIA Container Toolkit, a driver supporting CUDA 13.4, GitHub CLI, Python 3.12, and access to `nvcr.io/nvidia/pytorch`. Run these commands in an empty working directory:

```bash
gh release download kimi-k3-x4t-dspark-k7-cc8-20260930 \
  --repo local-inference-lab/rtx6kpro \
  --pattern 'kimi-k3-x4t-dspark-k7-cc8-20260930.tar.zst*'
sha256sum -c kimi-k3-x4t-dspark-k7-cc8-20260930.tar.zst.sha256
tar --zstd -xf kimi-k3-x4t-dspark-k7-cc8-20260930.tar.zst
cd kimi-k3-x4t-dspark-k7-cc8-20260930
sha256sum -c SHA256SUMS

# Apply pinned PR changes and verify both complete source-tree hashes.
python3 assemble-sources.py --output sources

git clone https://github.com/local-inference-lab/blackwell-llm-docker.git docker-recipe
git -C docker-recipe checkout b3fe0afe1621273059fb19dee1034e2272043a55
docker buildx create --name kimi-x4t-rebuild --driver docker-container
BUILDX_BUILDER=kimi-x4t-rebuild python3 \
  docker-recipe/tools/jovian_wheel_runtime/build_runtime_image.py \
  runtime-bundle local/kimi-k3:x4t-dspark-k7-cc8-reproduced-20260930

docker run --rm -i --entrypoint /opt/venv/bin/python \
  local/kimi-k3:x4t-dspark-k7-cc8-reproduced-20260930 - \
  < verify-image.py > rebuilt-payload.json
diff -u evidence/running-payload.json rebuilt-payload.json
```

This builds from a pinned NVIDIA foundation and complete wheels, not local parent-image tags or Python file overlays. FlashInfer does not need recompilation. The resulting Docker image ID can differ from the preserved image because layer/metadata identities differ; the installed serving source, native binaries and listed dependency versions must match the payload receipt.

The frozen recipe uses **CUTLASS DSL 4.6.2**. Do not replace that pin with 4.7.1 and assume the same E2E qualification. The PRs are compatible with the 4.7.1 master at the component-test level; replacing the serving image requires a separate full-model check.

## Prepare the checkpoints

On Frank2 the X4T checkpoint is `/data/ssd2/Kimi-K3-X4T`. It is a lossless representation of the official MXFP4 expert weights, not a two-bit EXL3 model. The checkpoint directory must contain `manifest.json`, `build-contract.json`, `metadata/` and `tensors/`.

From the extracted bundle directory:

```bash
export KIMI_CHECKPOINT=/data/ssd2/Kimi-K3-X4T
python3 launch/prepare-view.py \
  --checkpoint "$KIMI_CHECKPOINT" --output launch/checkpoint-view

# Omit this download if these exact draft files already exist in launch/drafts.
docker run --rm --network host \
  -v "$PWD/launch:/task" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /opt/venv/bin/python \
  local/kimi-k3:x4t-dspark-k7-cc8-reproduced-20260930 \
  -c 'from huggingface_hub import snapshot_download; snapshot_download("RedHatAI/Kimi-K3-speculator.dspark", revision="38a88101e0d46bb22134b9da340f381b954d40d4", local_dir="/task/drafts/redhat-dspark-38a88101")'
```

The draft source is BF16 and approximately 9.49 GB on disk. The launch quantizes its selected linear projections online to MXFP8, leaving QKV and the Markov head BF16. Target embeddings are shared. Creating `checkpoint-view` only writes serving metadata; it does not modify model weights.

## Start on Frank2

Confirm the 16 selected GPUs and port 8012 are free before starting another instance. The script refuses to reuse an existing container name or occupied port. It does not stop any service, change GPU clocks or reset hardware.

```bash
export KIMI_CHECKPOINT=/data/ssd2/Kimi-K3-X4T
KIMI_CONTAINER=kimi-k3-x4t-dspark-k7-cc8 \
  bash launch/start-x4t-dspark-k7-cc8.sh
docker logs -f kimi-k3-x4t-dspark-k7-cc8
```

After startup, use `http://10.66.66.14:8012/v1` and model name `Kimi-K3`. It binds `0.0.0.0` without authentication in this profile; use only on a trusted network or configure an authenticated gateway before exposing it publicly.

The default GPU UUID list selects Frank2 GPUs 0–13, 15 and 16, excluding GPU14/PCI C5:00.0, which has an x4 link. On another host, set `KIMI_GPUS` to a comma-separated list of **16 appropriate GPU UUIDs**.

## Settings

| Setting | Default and recommendation | Override example |
| --- | --- | --- |
| `KIMI_IMAGE` | `local/kimi-k3:x4t-dspark-k7-cc8-reproduced-20260930` | `KIMI_IMAGE=your-registry/your-tag` |
| `KIMI_PORT` | `8012` | `KIMI_PORT=8015` |
| `KIMI_CHECKPOINT` | `/data/ssd2/Kimi-K3-X4T` | `KIMI_CHECKPOINT=/models/Kimi-K3-X4T` |
| `KIMI_KV_BYTES` | `5600000000` bytes/GPU | `KIMI_KV_BYTES=5400000000` for more free VRAM |
| `KIMI_MAX_MODEL_LEN` | `1000000` | `KIMI_MAX_MODEL_LEN=262144` |
| `KIMI_KERNEL_CACHE` | `/mnt/luke/kimi-k3-cache/kk-components` | `KIMI_KERNEL_CACHE=/data/kimi-kernel-cache` |
| `KIMI_RUN_ROOT` | Directory containing the launch scripts | `KIMI_RUN_ROOT=/data/kimi-serving` |

CC8 is fixed by the wrapper (`--max-num-seqs 8`). Keep DSpark K7, `FULL_DECODE_ONLY` and graph rows `7,14,28,56,64` together: the target verifies eight rows per request, while the draft produces seven. The allocator is `expandable_segments:True,large_segment_size_mb:12,graph_capture_record_stream_reuse:True`. Removing its graph-reuse option or increasing concurrency changes memory requirements.

The request limit is not a promise that eight simultaneous 1M-token requests fit. Their combined cache would exceed the 4.28M-token pool. CC16 is not supported by this memory qualification; at the same KV budget it OOMs during lazy NCCL initialization.

## Measurements and limits

Measurements use the preserved source-identical image on Frank2, 16 RTX PRO 6000 Blackwell GPUs at 600 W and +6000 memory offset, approximately 16365 MHz loaded memory. The launcher does not set those clocks. Timing is research-only because the driver reported Reliability event mask `0x400`.

| Workload | Measured rate |
| --- | ---: |
| Game coding, 183 input / 4096 output, median of 3 | 116.72 decode tok/s |
| Python proxy coding, 143 input / 2048 output, median of 3 | 94.38 decode tok/s |
| Uncached 32k prefill, median of 3 | 3740.41 tok/s |
| Eight concurrent requests, 512 output each | 249.32 aggregate E2E tok/s |

Game-coding acceptance was 34.55% median; individual rates ranged from 110.54 to 151.56 tok/s. Do not treat 151.56 as the representative speed. Eight concurrent 32768-input/256-output requests passed without OOM, preemption or restart. Full-model KLD and generation to the 1M limit were not tested.

For the exact source/image hashes, PR boundaries, GPU/component checks and serving receipts, see the [integration issue](https://github.com/local-inference-lab/vllm/issues/906) and the bundle's `source-lock.json` and `evidence/`. The original MXFP4 and uniform EXL3 results are recorded separately in the [historical qualification report](validation/exl3-mxfp4-canonical-20260929.md).
