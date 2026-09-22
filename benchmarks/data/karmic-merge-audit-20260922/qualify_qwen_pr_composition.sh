#!/usr/bin/env bash
set -euo pipefail

# Validate an identified image on the assigned Max-Q GPU pairs.
# GPUs 5/6 and 7/8 and ports 5071/5075 must be free before invocation.
registry_image=${1:?Pass ghcr.io/local-inference-lab/vllm@sha256:DIGEST}
run_id=${2:?Pass a unique lowercase evidence prefix}
if [[ -n ${QWEN_BUILT_IMAGE_ID:-} ]]; then
  # A LAN-transferred build can be checked while it uploads. Publication
  # requires a subsequent registry pull with the identical image config ID.
  [[ "$registry_image" =~ ^ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-[0-9]{8}-[0-9a-f]{16}$ ]]
  [[ "$QWEN_BUILT_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]]
else
  [[ "$registry_image" =~ ^ghcr.io/local-inference-lab/vllm@sha256:[0-9a-f]{64}$ ]]
fi
[[ "$run_id" =~ ^[a-z0-9-]+$ ]]
qualification_root=/root/vllm/qwen38next
cd "$qualification_root"
evidence=results/integration-audit-20260922
python_exec=tmp/unified-runtime-tests/bin/python
remote_host=root@192.168.0.115
ssh_args=(-o BatchMode=yes -o HostKeyAlias=192.168.66.4)
perf_container="${run_id}-perf"
native_container="${run_id}-native"
test ! -e "$evidence/${run_id}-matched-1.json"
if [[ -n ${QWEN_BUILT_IMAGE_ID:-} ]]; then
  actual_image_id=$(ssh "${ssh_args[@]}" "$remote_host" \
    "docker image inspect --format '{{.Id}}' $registry_image")
  test "$actual_image_id" = "$QWEN_BUILT_IMAGE_ID"
fi

scp "${ssh_args[@]}" scripts/start_qwen_registry_perf.sh \
  scripts/start_qwen_registry_native_cache.sh "$remote_host:/tmp/"
ssh "${ssh_args[@]}" "$remote_host" \
  "QWEN_REGISTRY_IMAGE=$registry_image QWEN_REGISTRY_CONTAINER=$perf_container QWEN_REGISTRY_GPUS=5,6 QWEN_REGISTRY_PORT=5071 bash /tmp/start_qwen_registry_perf.sh"
ssh "${ssh_args[@]}" "$remote_host" \
  "QWEN_REGISTRY_IMAGE=$registry_image QWEN_REGISTRY_CONTAINER=$native_container QWEN_REGISTRY_GPUS=7,8 QWEN_REGISTRY_PORT=5075 bash /tmp/start_qwen_registry_native_cache.sh"

wait_healthy() {
  local container=$1 port=$2 deadline=$((SECONDS + 1800)) startup_tail
  while ! curl --noproxy '*' -fsS --max-time 3 "http://192.168.0.115:$port/health" >/dev/null 2>&1; do
    if [[ $(ssh "${ssh_args[@]}" "$remote_host" "docker inspect --format '{{.State.Running}}' $container") != true ]]; then
      ssh "${ssh_args[@]}" "$remote_host" "docker logs $container" > "$evidence/$container-startup.log" 2>&1
      return 1
    fi
    startup_tail=$(ssh "${ssh_args[@]}" "$remote_host" "docker logs --tail 40 $container 2>&1")
    if [[ "$startup_tail" == *"EngineCore failed to start."* ||
          "$startup_tail" == *"Engine core initialization failed."* ]]; then
      ssh "${ssh_args[@]}" "$remote_host" "docker logs $container" > "$evidence/$container-startup.log" 2>&1
      return 1
    fi
    if ((SECONDS >= deadline)); then
      return 1
    fi
    sleep 5
  done
}
wait_healthy "$perf_container" 5071
wait_healthy "$native_container" 5075
"$python_exec" scripts/glm53-runtime-audit.py \
  --ssh-host root@192.168.66.4 --container "$perf_container" \
  --container "$native_container" --output "$evidence/${run_id}-deployments.json"
printf 'Both endpoints healthy; testing distinct-code CPU restore.\n'
STRICT_RECALL=1 EVICT_COUNT=10 PRIME_MARKER=zeta EVICT_PREFIX=eta \
  PROBE_QUESTION='Return the access code and the result of 11 plus 2.' \
  OFFLOAD_BASE=http://192.168.0.115:5075 \
  "$python_exec" /tmp/qwen-generic-offload-probe-20260921.py \
  > "$evidence/${run_id}-native-restore.jsonl"
ssh "${ssh_args[@]}" "$remote_host" "docker stop $native_container"

# The other test pair stays idle during throughput measurements.
ssh "${ssh_args[@]}" "$remote_host" \
  'nvidia-smi -i 5,6 --query-gpu=timestamp,index,uuid,name,clocks.current.sm,clocks.current.memory,memory.used,utilization.gpu,clocks_throttle_reasons.active --format=csv -l 1' \
  > "$evidence/${run_id}-gpu-clocks.csv" &
clock_monitor_pid=$!
trap 'kill "$clock_monitor_pid" 2>/dev/null || true' EXIT
for repeat in 1 2 3 4 5; do
  "$python_exec" /tmp/run-qwen-matched-llmbench-20260922.py \
    --host 192.168.0.115 --port 5071 --model Qwen3.8-Flash-Next \
    --contexts 0 --concurrency 1,8 --duration 30 --decode-warmup-seconds 10 \
    --max-tokens 32768 --temperature 1 --skip-prefill --display-mode plain \
    --output "$evidence/${run_id}-matched-${repeat}.json" \
    > "$evidence/${run_id}-matched-${repeat}.log" 2>&1
  jq -e '[.results[] | (.aggregate_tps > 0 and
    (.failure_reason == null or .failure_reason == "") and
    ((.num_errors // 0) == 0) and (.loop_detected|not) and
    (.underfilled|not) and (.warmup_timed_out|not) and (.capacity_limited|not))] | all' \
    "$evidence/${run_id}-matched-${repeat}.json"
done
for repeat in 1 2 3; do
  "$python_exec" /tmp/run-qwen-matched-llmbench-20260922.py \
    --host 192.168.0.115 --port 5071 --model Qwen3.8-Flash-Next \
    --standalone-prefill --prefill-only --prefill-contexts 32k --prefill-duration 30 \
    --contexts 0 --concurrency 1 --duration 30 --decode-warmup-seconds 10 \
    --max-tokens 32768 --temperature 1 --display-mode plain \
    --output "$evidence/${run_id}-prefill32k-${repeat}.json" \
    > "$evidence/${run_id}-prefill32k-${repeat}.log" 2>&1
done
QWEN_BASE=http://192.168.0.115:5071 "$python_exec" scripts/qwen_mtp_metadata_smoke.py \
  > "$evidence/${run_id}-mixed.jsonl"
"$python_exec" scripts/summarize_qwen_canonical.py "$evidence" "${run_id}-matched" \
  --minimum-runs 5 > "$evidence/${run_id}-summary.json"
printf 'Registry serving qualification complete; performance endpoint retained.\n'
