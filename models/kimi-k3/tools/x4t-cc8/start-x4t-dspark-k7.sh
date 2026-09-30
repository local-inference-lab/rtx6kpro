#!/usr/bin/env bash
# Serve Kimi X4T with MXFP8 KDA and Red Hat DSpark seven-token speculation.
set -euo pipefail
unset NCCL_GRAPH_FILE
readonly root=${KIMI_RUN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
readonly image=${KIMI_IMAGE:-local/kimi-k3:x4t-tp16-prepared-20260930}
readonly name=${KIMI_CONTAINER:-kimi-k3-x4t-redhat-dspark-k7-dcp16-20260930}
readonly port=${KIMI_PORT:-8012}
readonly kv=${KIMI_KV_BYTES:-4000000000}
readonly uuids=${KIMI_GPUS:-GPU-a9fe82bb-7ed7-b3c0-59d0-db83e310170a,GPU-68bb0500-3f8a-6ace-f44b-dc0a0f90e83f,GPU-912e2e2b-0663-b6b1-8b7d-c9ceb647c4c7,GPU-1b99fbf6-7038-118d-1577-d3e1f7f0dc4f,GPU-06d95f5c-aef4-b457-c903-987e0dff8dac,GPU-ffe45a62-880f-f315-e161-5b9b41427222,GPU-c01cd5df-48b9-8096-7e7b-3b9b76fd9cdc,GPU-483c98d5-a286-c09a-c035-4493569da84d,GPU-3b1fa5cd-abdd-3dd4-9276-0bb169c427fc,GPU-6c085e45-350a-4034-662e-2a05d3ebd3f4,GPU-aabd6d9e-a55c-2919-aca3-4631a65e0ef3,GPU-bdc68f99-45c4-3351-0377-5b62ae7b893a,GPU-e65b4b33-5d3e-f05a-a74e-cdf904084496,GPU-c3cbeed4-063a-e159-d7d6-c92c5a4ee100,GPU-81eb3f31-7d0a-64bd-5492-680ee32f00c5,GPU-6a53f238-5e95-007b-65f9-37ee07cecd3e}
sequences=${KIMI_MAX_SEQS:-1}
context=${KIMI_MAX_MODEL_LEN:-1000000}
graph_mode=${KIMI_CUDA_GRAPH_MODE:-FULL_AND_PIECEWISE}
capture_sizes=${KIMI_GRAPH_TOKENS:-}
allocator=${KIMI_ALLOCATOR_CONFIG:-expandable_segments:True,large_segment_size_mb:12}
profile_args=(--language-model-only --no-enable-prefix-caching)
if ! [[ "$sequences" =~ ^[1-9][0-9]*$ ]]; then
  echo 'KIMI_MAX_SEQS must be a positive integer' >&2
  exit 2
fi
graph_sizes=()
if [[ -n "$capture_sizes" ]]; then
  IFS=, read -ra token_sizes <<< "$capture_sizes"
  for size in "${token_sizes[@]}"; do
    if ! [[ "$size" =~ ^[1-9][0-9]*$ ]] || ((size > sequences*8)); then
      echo 'KIMI_GRAPH_TOKENS must contain row counts in 1..8*KIMI_MAX_SEQS' >&2
      exit 2
    fi
    graph_sizes+=("$size")
  done
  # Target verification uses eight rows; this DSpark draft uses seven.
  if [[ ",${capture_sizes}," != *",$((sequences*8)),"* ]] || [[ ",${capture_sizes}," != *",$((sequences*7)),"* ]]; then
    echo 'KIMI_GRAPH_TOKENS must cover target and draft at KIMI_MAX_SEQS' >&2
    exit 2
  fi
else
  for ((i=1; i<=sequences; i++)); do
    graph_sizes+=("$((i*7))" "$((i*8))")
  done
fi
mapfile -t graph_sizes < <(printf '%s\n' "${graph_sizes[@]}" | sort -nu)
graphs="[$(IFS=,; echo "${graph_sizes[*]}")]"
if docker inspect "$name" >/dev/null 2>&1; then
  echo "Container name already exists: $name" >&2
  exit 1
fi
if [[ -n $(ss -H -ltn "sport = :$port") ]]; then
  echo "Port $port is occupied" >&2
  exit 1
fi
readonly evidence="$root/evidence/$name"
mkdir -p "$root/cache/hf-modules" "$evidence"
docker run -d --init --name "$name" --gpus all --network host --ipc host \
  --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$root/cache/hf-modules:/root/.cache/huggingface/modules" \
  -v "$root:/task" -v "${KIMI_CHECKPOINT:-/data/ssd2/Kimi-K3-X4T}:/x4t:ro" \
  -v "${KIMI_KERNEL_CACHE:-/mnt/luke/kimi-k3-cache/kk-components}:/cache/kimi-k3" \
  -e HF_HUB_OFFLINE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="$uuids" \
  -e OMP_NUM_THREADS=16 -e PYTORCH_CUDA_ALLOC_CONF="$allocator" \
  -e NCCL_DEBUG=WARN \
  -e VLLM_DEBUG_GRAPH_MEMORY_ACCOUNTING=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e VLLM_K3_DENSE_MLA_PARTIAL_DTYPE=fp32 \
  -e VLLM_K3_KV_GROUP_SIZE=6 -e VLLM_KV_CACHE_LAYOUT=BLHNC \
  -e VLLM_DSPARK_DRAFT_KV_WINDOW=32768 -e VLLM_DSPARK_COMPACT_ROPE=1 \
  -e VLLM_DSPARK_SHARD_MARKOV_HEAD=1 -e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
  -e VLLM_KIMI_K3_B12X_DSPARK_ARGMAX=1 \
  -e VLLM_DFLASH_SHARD_AUX_PROJECTION=1 -e VLLM_DFLASH_AUX_BF16_STAGING=1 \
  -e VLLM_DFLASH_AUX_MXFP8_STREAMING=0 -e VLLM_DFLASH_COMPACT_ROPE=1 \
  -e VLLM_B12X_MXFP8_ACTIVATION_MODE=a16 \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096 -e B12X_W4A16_PREFILL_FUSED_SUM=1 \
  -e B12X_W4A16_STABLE_ROUTE_PACK=1 -e B12X_W4A16_SMALL_M_HOST_BARRIER_RESET=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_SINGLE_CHANNEL=1 -e VLLM_USE_B12X_DCP_A2A=1 \
  -e B12X_PCIE_ALLREDUCE_ALGORITHM=island_rs \
  -e B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION=1 \
  -e B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER=0 -e B12X_PCIE_HIERARCHICAL_THREADS=256 \
  -e B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES=24 \
  -e B12X_PCIE_HIERARCHICAL_BF16X2=1 -e B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS=7168 \
  -e B12X_PCIE_DCP_THREADS=512 -e B12X_PCIE_DCP_BLOCK_LIMIT=8 \
  -e B12X_PCIE_KIMI_TOPK_THREADS=384 -e VLLM_KIMI_SHARD_QKV_A=1 \
  -e VLLM_KIMI_SHARD_AUXILIARY_PROJECTIONS=1 -e VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1 \
  -e VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE=4096 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e INSTANTTENSOR_COPY=0 -e INSTANTTENSOR_BUFFER_SIZE=536870912 \
  -e INSTANTTENSOR_IO_DEPTH=16 -e INSTANTTENSOR_BACKEND=AIO -e INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.6 \
  -e TRITON_CACHE_DIR=/cache/kimi-k3/triton -e CUTE_DSL_CACHE_DIR=/cache/kimi-k3/cute \
  -e B12X_COMPILE_CACHE_DIR=/cache/kimi-k3/b12x-compile \
  -e VLLM_CACHE_ROOT=/cache/kimi-k3/vllm -e B12X_COMPILE_WORKERS=4 \
  --entrypoint /usr/local/bin/lil-entrypoint "$image" \
  /opt/venv/bin/lil-runtime-bootstrap /opt/venv/bin/python \
  -m vllm.entrypoints.cli.main serve /task/checkpoint-view --served-model-name Kimi-K3 \
  --trust-remote-code --host 0.0.0.0 --port "$port" \
  --tensor-parallel-size 16 --decode-context-parallel-size 16 \
  --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 1 \
  --load-format exact_mxfp4 --quantization kimi_x4t \
  --quantization-config '{"linear":"mxfp8","ignore":["re:^(?!.*(?:self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj|in_proj_qkvgfab)|vision_tower\\..*|mm_projector\\..*)$).*$"]}' \
  --speculative-config '{"method":"dspark","model":"/task/drafts/redhat-dspark-38a88101","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto","draft_sample_method":"greedy","rejection_sample_method":"block","draft_load_config":{"load_format":"auto"},"quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*qkv_proj$","re:.*markov_head.*"]}}' \
  --moe-backend b12x --linear-backend b12x \
  --attention-backend B12X --kda-prefill-backend b12x \
  --dtype bfloat16 --kv-cache-dtype fp8 --kv-cache-memory-bytes "$kv" \
  --max-model-len "$context" --max-num-batched-tokens 4096 --max-num-scheduled-tokens 4096 \
  --max-num-seqs "$sequences" --enable-chunked-prefill \
  "${profile_args[@]}" \
  --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"$graph_mode\",\"cudagraph_capture_sizes\":$graphs,\"pass_config\":{\"fuse_allreduce_rms\":true}}" \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --structured-outputs-config.backend xgrammar
docker inspect "$name" > "$evidence/container.json"
