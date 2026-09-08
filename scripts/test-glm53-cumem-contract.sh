#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/compose/glm53-flash-nvfp4-jovian-cumem.yml"
LAUNCHER="${ROOT_DIR}/scripts/run-glm53-flash-jovian-cumem-compose.sh"
README="${ROOT_DIR}/models/glm53-flash/README.md"
IMAGE_ID="sha256:67967ddac438bec8393a2822d00c532f1417fc34eb7652c46ec9a2040559a05a"

for path in "${COMPOSE_FILE}" "${LAUNCHER}" "${README}"; do
  [[ -f "${path}" ]] || {
    printf 'missing required path: %s\n' "${path}" >&2
    exit 1
  }
done

rendered="$(mktemp)"
trap 'rm -f "${rendered}"' EXIT

IMAGE="${IMAGE_ID}" \
MODEL_DIR=/tmp/glm53-model \
CACHE_DIR=/tmp/glm53-cache \
docker compose --profile cumem -f "${COMPOSE_FILE}" config >"${rendered}"

require_rendered() {
  local value=$1
  grep -Fq -- "${value}" "${rendered}" || {
    printf 'rendered Compose is missing: %s\n' "${value}" >&2
    exit 1
  }
}

for value in \
  "${IMAGE_ID}" \
  'restart: unless-stopped' \
  'user: "0:0"' \
  'name: glm53-flash-jovian-cumem-broker' \
  'target: /run/lmcache-cumem' \
  'LMCACHE_MP_TRANSFER_MODE: lmcache_driven' \
  'VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS: "60"' \
  'CUDA_VISIBLE_DEVICES: 0,1,2,3' \
  '--separate-object-groups' \
  '--supported-transfer-mode' \
  '--chunk-size' \
  '--l1-size-gb' \
  '--served-model-name' \
  'glm-5.3-flash' \
  '--tensor-parallel-size' \
  '--decode-context-parallel-size' \
  '--max-model-len' \
  '1048576' \
  '--max-num-batched-tokens' \
  '--max-num-seqs' \
  '--kv-cache-memory' \
  '33554432000' \
  '--kv-cache-dtype' \
  'fp8' \
  '--gpu-memory-utilization' \
  '0.945' \
  '--max-cudagraph-capture-size' \
  '128' \
  '--enable-cumem-allocator' \
  '--shutdown-timeout' \
  '--enable-prefix-caching' \
  '--enable-prompt-tokens-details' \
  '"num_speculative_tokens":3' \
  '"lmcache.mp.mp_transfer_mode":"lmcache_driven"'; do
  require_rendered "${value}"
done

[[ "$(grep -Fc -- 'restart: unless-stopped' "${rendered}")" == 2 ]]
[[ "$(grep -Fc -- "image: ${IMAGE_ID}" "${rendered}")" == 2 ]]

grep -Fq -- "${IMAGE_ID}" "${LAUNCHER}"
grep -Fq -- '--profile cumem' "${LAUNCHER}"
grep -Fq -- 'stop -t 90 server' "${LAUNCHER}"
grep -Fq -- 'stop -t 90 lmcache' "${LAUNCHER}"
grep -Fq -- 'fb167fed58b5f3b4d3e050efcffcea9b5b70f715' "${README}"
grep -Fq -- 'b39d501b26' "${README}"
grep -Fq -- '88cff3d5a9' "${README}"
grep -Fq -- '129/129' "${README}"
grep -Fq -- '116/116' "${README}"

printf 'GLM-5.3 cuMem Compose contract passed\n'
