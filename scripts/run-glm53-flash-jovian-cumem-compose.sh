#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${ROOT_DIR}/compose/glm53-flash-nvfp4-jovian-cumem.yml}"
EXPECTED_IMAGE_ID="sha256:1d43855573a38e90215b785fb158498bb3654d75c45cef258c512e08c0036ffb"

IMAGE="${IMAGE:?set IMAGE to a local tag or ID for the fleet-qualified image}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the downloaded GLM-5.3 checkpoint}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR to a writable JIT cache directory}"
NAME="${NAME:-glm53-flash-jovian-cumem}"
BROKER_VOLUME="${BROKER_VOLUME:-glm53-flash-jovian-cumem-broker}"
PORT="${PORT:-8001}"
ACTION="${1:-up}"

export IMAGE MODEL_DIR CACHE_DIR NAME BROKER_VOLUME PORT
compose=(docker compose --profile cumem -p "${NAME}" -f "${COMPOSE_FILE}")

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

verify_image() {
  local actual_image_id
  actual_image_id="$(docker image inspect "${IMAGE}" --format '{{.Id}}')" ||
    die "IMAGE is not present locally: ${IMAGE}"
  [[ "${actual_image_id}" == "${EXPECTED_IMAGE_ID}" ]] ||
    die "IMAGE resolves to ${actual_image_id}; expected ${EXPECTED_IMAGE_ID}"
}

prepare_launch() {
  [[ -f "${MODEL_DIR}/config.json" ]] ||
    die "MODEL_DIR does not look like a local Hugging Face checkpoint: ${MODEL_DIR}"
  mkdir -p "${CACHE_DIR}"
  verify_image

  docker volume create "${BROKER_VOLUME}" >/dev/null
  local broker_owner
  broker_owner="$(
    docker run --rm \
      --user 0:0 \
      --volume "${BROKER_VOLUME}:/run/lmcache-cumem" \
      --entrypoint stat \
      "${IMAGE}" \
      -c '%u:%g' /run/lmcache-cumem
  )"
  [[ "${broker_owner}" == "0:0" ]] ||
    die "${BROKER_VOLUME} must be root-owned; found ${broker_owner}"
}

start_stack() {
  prepare_launch
  "${compose[@]}" up -d lmcache
  "${compose[@]}" up -d server
}

stop_stack() {
  # Preserve graceful automatic unregister: workers stop before the sidecar.
  "${compose[@]}" stop -t 90 server
  "${compose[@]}" stop -t 90 lmcache
}

case "${ACTION}" in
  up|start)
    start_stack
    ;;
  restart)
    stop_stack
    start_stack
    ;;
  down|stop)
    stop_stack
    "${compose[@]}" down
    ;;
  config|ps)
    "${compose[@]}" "${ACTION}"
    ;;
  logs)
    "${compose[@]}" logs -f --tail=200
    ;;
  *)
    die "usage: $0 [up|restart|down|logs|ps|config]"
    ;;
esac

printf '%s\n' \
  "profile=cumem" \
  "image=${IMAGE}" \
  "expected_image_id=${EXPECTED_IMAGE_ID}" \
  "model_dir=${MODEL_DIR}" \
  "cache_dir=${CACHE_DIR}" \
  "broker_volume=${BROKER_VOLUME}" \
  "served_model_name=glm-5.3-flash" \
  "port=${PORT}"
