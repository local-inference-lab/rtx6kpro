#!/usr/bin/env bash
set -euo pipefail

# Source-unmodified image qualification, matching the TP2 decode comparison.
registry_image=${QWEN_REGISTRY_IMAGE:?Set the immutable registry image}
registry_container=${QWEN_REGISTRY_CONTAINER:-feedback-qwen-registry-perf-20260922}
docker run -d --name "$registry_container" \
  --init --gpus '"device=5,6"' --network host --ipc host --shm-size 32g \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v feedback-qwen-runtime-cache:/cache \
  -e PROFILE=qwen38-flash-next -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e MODEL=/root/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \
  -e TP=2 -e DCP=1 -e PORT=5071 -e MAX_MODEL_LEN=524288 \
  -e KV_CACHE_MEMORY_BYTES=8589934592 -e HF_HUB_OFFLINE=1 \
  "$registry_image" --mode mtp --draft-tokens 3 \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir "/cache/profiles/${registry_container}" \
  --profiler-config.torch_profiler_with_stack true \
  --profiler-config.torch_profiler_record_shapes false \
  --profiler-config.torch_profiler_with_memory false \
  --profiler-config.torch_profiler_with_flops false \
  --profiler-config.torch_profiler_use_gzip true \
  --profiler-config.ignore_frontend true \
  --profiler-config.max_iterations 8 \
  --profiler-config.warmup_iterations 0 \
  --profiler-config.active_iterations 8 \
  --profiler-config.wait_iterations 0
