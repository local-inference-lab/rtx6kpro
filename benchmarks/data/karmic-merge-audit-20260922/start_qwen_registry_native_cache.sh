#!/usr/bin/env bash
set -euo pipefail

# Bounded external-restore qualification; no runtime source overlays.
registry_image=${QWEN_REGISTRY_IMAGE:?Set the immutable registry image}
registry_container=${QWEN_REGISTRY_CONTAINER:-feedback-qwen-registry-native-20260922}
docker run -d --name "$registry_container" \
  --init --gpus '"device=7,8"' --network host --ipc host --shm-size 32g \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v feedback-qwen-runtime-cache:/cache \
  -e PROFILE=qwen38-flash-next -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e MODEL=/root/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \
  -e TP=2 -e DCP=1 -e PORT=5075 -e MAX_MODEL_LEN=8192 \
  -e KV_CACHE_MEMORY_BYTES=1073741824 -e MAX_NUM_SEQS=4 \
  -e CACHE_MODE=native -e NATIVE_KV_OFFLOADING_SIZE_GB=4 \
  -e HF_HUB_OFFLINE=1 \
  "$registry_image" --mode mtp --draft-tokens 3
