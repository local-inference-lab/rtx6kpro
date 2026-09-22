#!/usr/bin/env bash
set -euo pipefail

# Share the assigned four-GPU partition without overlapping performance work.
qwen_controller_pid=${1:?Pass the running Qwen qualification controller PID}
[[ "$qwen_controller_pid" =~ ^[0-9]+$ ]]
cd /root/vllm/qwen38next
evidence=results/integration-audit-20260922
deadline=$((SECONDS + 3600))
while kill -0 "$qwen_controller_pid" 2>/dev/null; do
  ((SECONDS < deadline)) || exit 1
  sleep 5
done
test -s "$evidence/qwen-tuning-registry-summary.json"
jq -e '."qwen-tuning-registry-matched".files | length == 5' \
  "$evidence/qwen-tuning-registry-summary.json" >/dev/null
ssh_args=(-o BatchMode=yes -o HostKeyAlias=192.168.66.4)
remote_host=root@192.168.0.115
# Keep a fixed post-prefill confirmation series separate from the initial five
# decode windows. Do not replace a lower valid result or pool startup states.
test ! -e "$evidence/qwen-tuning-registry-confirmation-1.json"
ssh "${ssh_args[@]}" "$remote_host" \
  'nvidia-smi -i 5,6 --query-gpu=timestamp,index,uuid,name,clocks.current.sm,clocks.current.memory,memory.used,utilization.gpu,clocks_throttle_reasons.active --format=csv -l 1' \
  > "$evidence/qwen-tuning-registry-confirmation-gpu-clocks.csv" &
clock_monitor_pid=$!
trap 'kill "$clock_monitor_pid" 2>/dev/null || true' EXIT
for repeat in 1 2 3 4 5; do
  tmp/unified-runtime-tests/bin/python /tmp/run-qwen-matched-llmbench-20260922.py \
    --host 192.168.0.115 --port 5071 --model Qwen3.8-Flash-Next \
    --contexts 0 --concurrency 1,8 --duration 30 --decode-warmup-seconds 10 \
    --max-tokens 32768 --temperature 1 --skip-prefill --display-mode plain \
    --output "$evidence/qwen-tuning-registry-confirmation-${repeat}.json" \
    > "$evidence/qwen-tuning-registry-confirmation-${repeat}.log" 2>&1
  jq -e '[.results[] | (.aggregate_tps > 0 and
    (.failure_reason == null or .failure_reason == "") and
    ((.num_errors // 0) == 0) and (.loop_detected|not) and
    (.underfilled|not) and (.warmup_timed_out|not) and (.capacity_limited|not))] | all' \
    "$evidence/qwen-tuning-registry-confirmation-${repeat}.json"
done
tmp/unified-runtime-tests/bin/python scripts/summarize_qwen_canonical.py "$evidence" \
  qwen-tuning-registry-confirmation --minimum-runs 5 \
  > "$evidence/qwen-tuning-registry-confirmation-summary.json"
kill "$clock_monitor_pid" 2>/dev/null || true
wait "$clock_monitor_pid" 2>/dev/null || true
trap - EXIT
ssh "${ssh_args[@]}" "$remote_host" 'docker stop qwen-tuning-registry-perf'
scp "${ssh_args[@]}" scripts/start_glm_dflash_registry_smoke.sh "$remote_host:/tmp/"
ssh "${ssh_args[@]}" "$remote_host" \
  'GLM_REGISTRY_IMAGE=ghcr.io/local-inference-lab/vllm@sha256:bf2c8f5da6f1de82b345367102b528692231a23eadc1b49c180c82c397c77522 GLM_REGISTRY_CONTAINER=glm-tuning-registry-smoke bash /tmp/start_glm_dflash_registry_smoke.sh'
tmp/unified-runtime-tests/bin/python scripts/qualify-model-endpoint.py \
  --container glm-tuning-registry-smoke --docker-ssh-host root@192.168.66.4 \
  --base http://192.168.0.115:5075 --model GLM-5.3-Flash-NVFP4 \
  --vision --vision-counts 2 4 --smoke-only \
  --expected-method dflash --expected-draft-tokens 7 \
  --output-dir "$evidence/glm-tuning-registry-smoke"
jq -e '.all_smoke_passed == true' "$evidence/glm-tuning-registry-smoke/qualification.json"
