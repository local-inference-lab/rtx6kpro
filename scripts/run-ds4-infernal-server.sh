#!/usr/bin/env bash
set -euo pipefail

# Launch one source-locked DeepSeek-V4-Flash-0731 benchmark server. PROFILE is
# the public benchmark identity; MODE, DSpark depth, and draft count are the
# corresponding serve-ds4-flash.sh interfaces.

IMAGE=${IMAGE:-voipmonitor/vllm:infernal-invocation-vllm7ed814e-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r7}
NAME=${NAME:-ds4-infernal-benchmark}
PORT=${PORT:-5000}
GPUS=${GPUS:-0,1}
TP=${TP:-${TP_SIZE:-2}}
PROFILE=${PROFILE:-${MODE:-dspark-k5}}
BACKEND=${BACKEND:-b12x-a8}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_BATCHED=${MAX_BATCHED:-${MAX_NUM_BATCHED_TOKENS:-8192}}
GPU_MEM=${GPU_MEM:-${GPU_MEMORY_UTILIZATION:-0.975}}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}
ALLREDUCE_MODE=${ALLREDUCE_MODE:-auto}
LOAD_FORMAT=${LOAD_FORMAT:-instanttensor}
INSTANTTENSOR_BACKEND=${INSTANTTENSOR_BACKEND:-BUFFERED}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
CACHE=${CACHE:-/root/.cache/ds4-infernal-benchmark}
CONTAINER_TMP=${CONTAINER_TMP:-${CACHE}/tmp/${NAME}}
SHM_SIZE=${SHM_SIZE:-32g}

case "${PROFILE}" in
  dspark-mtp0)
    helper_mode=dspark-mtp0
    dspark_tokens=0
    dspark_depth=fixed
    ;;
  dspark-k5)
    helper_mode=dspark
    dspark_tokens=5
    dspark_depth=fixed
    ;;
  dspark-k7)
    helper_mode=dspark
    dspark_tokens=7
    dspark_depth=fixed
    ;;
  dspark-k7-dynamic)
    helper_mode=dspark
    dspark_tokens=7
    dspark_depth=dynamic
    ;;
  *)
    printf 'PROFILE must be dspark-mtp0, dspark-k5, dspark-k7, or dspark-k7-dynamic; got %s\n' \
      "${PROFILE}" >&2
    exit 2
    ;;
esac

IFS=, read -r -a gpu_list <<<"${GPUS}"
if (( ${#gpu_list[@]} != TP )); then
  printf 'GPUS=%s exposes %s devices but TP=%s\n' "${GPUS}" "${#gpu_list[@]}" "${TP}" >&2
  exit 2
fi
if [[ "${LOAD_FORMAT}" != instanttensor ]]; then
  printf 'The qualified benchmark contract requires LOAD_FORMAT=instanttensor\n' >&2
  exit 2
fi

mkdir -p "${CACHE}" "${CONTAINER_TMP}"

helper_env=(
  -e MODE="${helper_mode}"
  -e BACKEND="${BACKEND}"
  -e PORT="${PORT}"
  -e TP_SIZE="${TP}"
  -e DCP_SIZE=1
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_BATCHED}"
  -e GPU_MEMORY_UTILIZATION="${GPU_MEM}"
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"
  -e ALLREDUCE_MODE="${ALLREDUCE_MODE}"
  -e LOAD_FORMAT="${LOAD_FORMAT}"
  -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND}"
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
  -e DSPARK_DEPTH_MODE="${dspark_depth}"
  -e DRAFT_SAMPLE_METHOD=probabilistic
  -e PYTHONHASHSEED=0
)
if (( dspark_tokens > 0 )); then
  helper_env+=(-e DSPARK_TOKENS="${dspark_tokens}")
fi

optional_env=(
  GRAPH CUDAGRAPH_CAPTURE_SIZES ENABLE_FLASHINFER_AUTOTUNE
  DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE DSPARK_SPS_CURVE
  DSPARK_CONFIDENCE_THRESHOLD DSPARK_BUDGET_FRAC
  DSPARK_CONFIDENCE_TEMPERATURE DSPARK_SPS_OVERHEAD_MS
  VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE B12X_PCIE_DMA
  KV_OFFLOADING_SIZE NATIVE_L2_GB NATIVE_L2_PATH
  LMCACHE_MODE LMCACHE_L1_GB LMCACHE_L1_INIT_GB LMCACHE_L2_GB
  LMCACHE_L2_PATH LMCACHE_L2_WORKERS LMCACHE_CHUNK_SIZE
  LMCACHE_MAX_GPU_WORKERS LMCACHE_MAX_CPU_WORKERS LMCACHE_PORT
  LMCACHE_HTTP_PORT LMCACHE_PROMETHEUS_PORT LMCACHE_START_TIMEOUT
  VLLM_SERVER_DEV_MODE EXTRA_VLLM_ARGS DRY_RUN
)
for key in "${optional_env[@]}"; do
  if [[ -n "${!key+x}" ]]; then helper_env+=(-e "${key}=${!key}"); fi
done

docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc host \
  --shm-size "${SHM_SIZE}" \
  --network host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "${CACHE}:/cache:rw" \
  -v "${CONTAINER_TMP}:/container-tmp:rw" \
  -e CUDA_VISIBLE_DEVICES="${GPUS}" \
  -e TMPDIR=/container-tmp \
  "${helper_env[@]}" \
  --entrypoint /usr/local/bin/lmcache-mp-wrapper.sh \
  "${IMAGE}" \
  /usr/local/bin/serve-ds4-flash.sh

printf '%s profile=%s backend=%s allreduce=%s tp=%s gpus=%s port=%s max_seqs=%s cache=%s\n' \
  "${NAME}" "${PROFILE}" "${BACKEND}" "${ALLREDUCE_MODE}" "${TP}" \
  "${GPUS}" "${PORT}" "${MAX_NUM_SEQS}" "${CACHE}"
