#!/usr/bin/env bash
set -euo pipefail

collective=${QWEN_COLLECTIVE:-b12x}
arm=${QWEN_ARM:-$collective}
extra=()
mounts=()
if [[ $collective == pynccl ]]; then
  extra+=(--disable-custom-all-reduce)
elif [[ $collective != b12x ]]; then
  echo 'QWEN_COLLECTIVE must be b12x or pynccl' >&2
  exit 2
fi

if [[ -n ${QWEN_METADATA_SOURCE:-} ]]; then
  for source_file in \
    vllm/models/qwen4_exp/nvidia/model_state.py \
    vllm/v1/attention/backend.py \
    vllm/v1/attention/backends/gdn_attn.py \
    vllm/v1/worker/gpu/attn_utils.py \
    vllm/v1/worker/gpu/cudagraph_utils.py \
    vllm/v1/worker/gpu/input_batch.py \
    vllm/v1/worker/gpu/model_runner.py; do
    mounts+=(-v "$QWEN_METADATA_SOURCE/$source_file:/opt/venv/lib/python3.12/site-packages/$source_file:ro")
  done
fi

docker run -d --name "feedback-vllm-qwen-tp2-${arm}-20260922" \
  --init --gpus '"device=5,6"' --network host --ipc host --shm-size 32g \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v feedback-qwen-runtime-cache:/cache \
  -e PROFILE=qwen38-flash-next -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e MODEL=/root/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd \
  -e TP=2 -e DCP=1 -e PORT=5071 -e MAX_MODEL_LEN=524288 \
  -e KV_CACHE_MEMORY_BYTES=8589934592 -e HF_HUB_OFFLINE=1 \
  "${mounts[@]}" \
  ghcr.io/local-inference-lab/vllm@sha256:7ee79a417df23504779130d3acddb0854395a307d1a217c590e18d4f51269910 \
  --mode mtp --draft-tokens 3 \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --profiler-config.profiler torch \
  --profiler-config.torch_profiler_dir "/cache/profiles/qwen-vllm-tp2-${arm}" \
  --profiler-config.torch_profiler_with_stack true \
  --profiler-config.torch_profiler_record_shapes false \
  --profiler-config.torch_profiler_with_memory false \
  --profiler-config.torch_profiler_with_flops false \
  --profiler-config.torch_profiler_use_gzip true \
  --profiler-config.ignore_frontend true \
  --profiler-config.max_iterations 8 \
  --profiler-config.warmup_iterations 0 \
  --profiler-config.active_iterations 8 \
  --profiler-config.wait_iterations 0 \
  "${extra[@]}"
