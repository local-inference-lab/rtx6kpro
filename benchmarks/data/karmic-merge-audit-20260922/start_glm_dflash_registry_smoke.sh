#!/usr/bin/env bash
set -euo pipefail

# Bounded text/vision check of the packaged DFlash loader and serving path.
registry_image=${GLM_REGISTRY_IMAGE:?Set the immutable registry image}
registry_container=${GLM_REGISTRY_CONTAINER:?Set a unique container name}
[[ "$registry_image" =~ ^ghcr.io/local-inference-lab/vllm@sha256:[0-9a-f]{64}$ ]]
docker run -d --name "$registry_container" \
  --init --gpus '"device=5,6,7,8"' --network host --ipc host --shm-size 32g \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v feedback-glm-registry-cache:/cache \
  -e PROFILE=glm53-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e MODEL=/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/snapshots/520de24eabf507659eaef7c70f14fd584527facc \
  -e DFLASH_MODEL=/root/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-DFlash2/snapshots/713226ab03bc38afdf955c7450436c2f7176f6f8 \
  -e TP=4 -e DCP=1 -e PORT=5075 -e MAX_MODEL_LEN=65536 \
  -e MAX_NUM_SEQS=8 -e KV_CACHE_MEMORY_BYTES=12884901888 \
  -e HF_HUB_OFFLINE=1 \
  "$registry_image" --mode dflash2 --draft-tokens 7
