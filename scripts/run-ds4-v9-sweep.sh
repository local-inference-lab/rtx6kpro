#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SWEEP_SCRIPT=${SWEEP_SCRIPT:-$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")}
LAUNCHER=${LAUNCHER:-$SCRIPT_DIR/run-ds4-v9-server.sh}
BENCH=${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}
RESULT_RENDERER=${RESULT_RENDERER:-}
LAYOUT_PLANNER=${LAYOUT_PLANNER:-$SCRIPT_DIR/plan-ds4-sweep-layout.py}
OUT=${OUT:-/root/bench-results/ds4-v9-$(date -u +%Y%m%d-%H%M%S)}
IMAGE=${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704}
PROGRESS_FILE=${PROGRESS_FILE:-/root/vllm/prubezne_vysledky}

TPS=${TPS:-2,4}
BACKENDS=${BACKENDS:-b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass}
MODES=${MODES:-standard-mtp0,standard-mtp2,standard-mtp3,dspark}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY:-1,16,32,64}
DECODE_CONTEXTS=${DECODE_CONTEXTS:-0}
DECODE_DURATION=${DECODE_DURATION:-30}
DECODE_MAX_TOKENS=${DECODE_MAX_TOKENS:-8192}
DECODE_TOKEN_BUDGET=${DECODE_TOKEN_BUDGET:-2000000}
PREFILL_CONTEXTS=${PREFILL_CONTEXTS:-8k,64k,128k}
PREFILL_DURATION=${PREFILL_DURATION:-10}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
MAX_BATCHED=${MAX_BATCHED:-}
GPU_MEM=${GPU_MEM:-}
ALLREDUCE_MODE=${ALLREDUCE_MODE:-}
RUN_DECODE=${RUN_DECODE:-1}
RUN_PREFILL=${RUN_PREFILL:-1}
QUALIFICATION_ROLE=${QUALIFICATION_ROLE:-combined}
PORT_BASE=${PORT_BASE:-7100}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-2400}
SYNC_WAVE_READY=${SYNC_WAVE_READY:-1}
ENABLE_TOPO_PIN=${ENABLE_TOPO_PIN:-1}
POST_READY_SETTLE_SECONDS=${POST_READY_SETTLE_SECONDS:-30}
RUNTIME_WARMUP=${RUNTIME_WARMUP:-1}
RUNTIME_WARMUP_DECODE_DURATION=${RUNTIME_WARMUP_DECODE_DURATION:-3}
RUNTIME_WARMUP_SPARSE_CONTEXT=${RUNTIME_WARMUP_SPARSE_CONTEXT:-}
RUNTIME_WARMUP_SPARSE_DURATION=${RUNTIME_WARMUP_SPARSE_DURATION:-2}
RUNTIME_WARMUP_PREFILL_DURATION=${RUNTIME_WARMUP_PREFILL_DURATION:-1}
POST_WARMUP_SETTLE_SECONDS=${POST_WARMUP_SETTLE_SECONDS:-30}
RESUME=${RESUME:-1}
VLLM_PATCH_FILE=${VLLM_PATCH_FILE:-/root/vllm/blackwell-llm-docker/patches/vllm-b12x-indexer-warmup-fallback-20260704.patch}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-ds4-v9}
SHARED_CACHE=${SHARED_CACHE:-}
GPU_GROUPS_TP2=${GPU_GROUPS_TP2:-"0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15"}
GPU_GROUPS_TP4=${GPU_GROUPS_TP4:-"0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15"}

mkdir -p "$OUT"

record_repro_artifacts() {
  local -a script_hash_inputs
  mkdir -p "$OUT/repro"
  cp "$LAUNCHER" "$OUT/repro/$(basename "$LAUNCHER")"
  cp "$SWEEP_SCRIPT" "$OUT/repro/$(basename "$SWEEP_SCRIPT")"
  script_hash_inputs=("$LAUNCHER" "$SWEEP_SCRIPT")
  if [[ "$SWEEP_SCRIPT" != "$SCRIPT_DIR/run-ds4-v9-sweep.sh" ]]; then
    cp "$SCRIPT_DIR/run-ds4-v9-sweep.sh" \
      "$OUT/repro/run-ds4-v9-sweep.sh"
    script_hash_inputs+=("$SCRIPT_DIR/run-ds4-v9-sweep.sh")
  fi
  if [[ -f "$SCRIPT_DIR/render-ds4-v9-results.py" ]]; then
    cp "$SCRIPT_DIR/render-ds4-v9-results.py" "$OUT/repro/render-ds4-v9-results.py"
    script_hash_inputs+=("$SCRIPT_DIR/render-ds4-v9-results.py")
  fi
  if [[ -f "$SCRIPT_DIR/validate-ds4-sweep-case.py" ]]; then
    cp "$SCRIPT_DIR/validate-ds4-sweep-case.py" \
      "$OUT/repro/validate-ds4-sweep-case.py"
    script_hash_inputs+=("$SCRIPT_DIR/validate-ds4-sweep-case.py")
  fi
  if [[ -f "$SCRIPT_DIR/validate-ds4-runtime-log.py" ]]; then
    cp "$SCRIPT_DIR/validate-ds4-runtime-log.py" \
      "$OUT/repro/validate-ds4-runtime-log.py"
    script_hash_inputs+=("$SCRIPT_DIR/validate-ds4-runtime-log.py")
  fi
  if [[ -f "$LAYOUT_PLANNER" ]]; then
    cp "$LAYOUT_PLANNER" "$OUT/repro/$(basename "$LAYOUT_PLANNER")"
    script_hash_inputs+=("$LAYOUT_PLANNER")
  fi
  if [[ -n "$RESULT_RENDERER" && -f "$RESULT_RENDERER" ]]; then
    cp "$RESULT_RENDERER" "$OUT/repro/$(basename "$RESULT_RENDERER")"
    script_hash_inputs+=("$RESULT_RENDERER")
  fi
  if [[ -f "$VLLM_PATCH_FILE" ]]; then
    cp "$VLLM_PATCH_FILE" "$OUT/repro/$(basename "$VLLM_PATCH_FILE")"
    sha256sum "$VLLM_PATCH_FILE" > "$OUT/repro/vllm-patch.sha256" || true
  fi
  if [[ -f "$BENCH" ]]; then
    cp "$BENCH" "$OUT/repro/$(basename "$BENCH")"
    sha256sum "$BENCH" > "$OUT/repro/bench.sha256" || true
  fi
  sha256sum "${script_hash_inputs[@]}" > "$OUT/repro/scripts.sha256" || true
  if git -C "$SCRIPT_DIR/.." rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$SCRIPT_DIR/.." rev-parse HEAD > "$OUT/repro/rtx6kpro-head.txt" || true
    git -C "$SCRIPT_DIR/.." status --short > "$OUT/repro/rtx6kpro-status.txt" || true
  fi
  docker image inspect "$IMAGE" > "$OUT/repro/image-inspect.json" 2>/dev/null || true
  docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' > "$OUT/repro/image-labels.json" 2>/dev/null || true
  nvidia-smi topo -m > "$OUT/repro/nvidia-topo.txt" 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit --format=csv > "$OUT/repro/nvidia-gpus.csv" 2>/dev/null || true
}

on_exit() {
  local status=$?
  printf '%s sweep_process_exit role=%s status=%s out=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$QUALIFICATION_ROLE" "$status" "$OUT" \
    >> "$PROGRESS_FILE"
}
trap on_exit EXIT

split_csv() {
  local input=$1
  input=${input//,/ }
  printf '%s\n' $input
}

gpu_groups_for_tp() {
  case "$1" in
    2) printf '%s\n' $GPU_GROUPS_TP2 ;;
    4) printf '%s\n' $GPU_GROUPS_TP4 ;;
    *) echo "Unsupported TP=$1" >&2; return 2 ;;
  esac
}

model_name_for_mode() {
  case "$1" in
    standard-*) printf '%s\n' "${STANDARD_SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}" ;;
    dspark|dspark-*) printf '%s\n' "${DSPARK_SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}" ;;
    *) return 2 ;;
  esac
}

wait_for_server() {
  local name=$1
  local port=$2
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      docker logs --tail 240 "$name" || true
      return 1
    fi
    sleep 5
  done
  docker logs --tail 300 "$name" || true
  echo "Timed out waiting for $name on port $port" >&2
  return 1
}

append_case_summary() {
  local status=$1 label=$2 case_dir=$3
  python3 - "$status" "$label" "$case_dir" "$PROGRESS_FILE" <<'PY'
import datetime as dt
import json
import math
import pathlib
import sys

status, label, case_dir, progress_file = sys.argv[1:]
case_path = pathlib.Path(case_dir)
now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def load(name):
    path = case_path / name
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)

def fmt(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.1f}"

decode = load("decode.json")
prefill = load("prefill.json")

cc = {}
for row in decode.get("results", []):
    if int(row.get("context_tokens", -1)) == 0:
        cc[int(row.get("concurrency", 0))] = row.get("aggregate_tps")

coding_summary = decode.get("coding_peak", {}).get("summary", {})
pref = prefill.get("prefill", {})

def prefill_rate(target):
    candidates = []
    for key, row in pref.items():
        try:
            actual = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= target - actual <= 256:
            candidates.append((actual, row))
    if not candidates:
        return None
    return max(candidates)[1].get("tok_per_sec")

line = (
    f"{now} {status} {label} "
    f"decode cc1={fmt(cc.get(1))} cc16={fmt(cc.get(16))} "
    f"cc32={fmt(cc.get(32))} cc64={fmt(cc.get(64))} "
    f"coding_median={fmt(coding_summary.get('median_generation_tok_s'))} "
    f"cjk={coding_summary.get('cjk_runs', 'NA')} "
    f"prefill 8k={fmt(prefill_rate(8192))} "
    f"64k={fmt(prefill_rate(65536))} "
    f"128k={fmt(prefill_rate(131072))} "
    f"dir={case_path}"
)
with open(progress_file, "a", encoding="utf-8") as f:
    f.write(line + "\n")
print(line)
PY
}

validate_case_results() {
  local case_dir=$1
  local -a args
  args=(
    "$SCRIPT_DIR/validate-ds4-sweep-case.py"
    "$case_dir"
    --decode-concurrency "$DECODE_CONCURRENCY"
    --prefill-contexts "$PREFILL_CONTEXTS"
  )
  [[ "$RUN_DECODE" == "1" ]] && args+=(--run-decode)
  [[ "$RUN_PREFILL" == "1" ]] && args+=(--run-prefill)
  if [[ "$RUN_DECODE" == "1" && -n "$RUNTIME_WARMUP_SPARSE_CONTEXT" ]]; then
    args+=(--sparse-warmup-context "$RUNTIME_WARMUP_SPARSE_CONTEXT")
  fi
  python3 "${args[@]}"
}

validate_runtime_log() {
  local server_log=$1
  local measurement_start_utc=$2
  python3 "$SCRIPT_DIR/validate-ds4-runtime-log.py" \
    "$server_log" "$measurement_start_utc"
}

validate_reusable_case() {
  local case_dir=$1
  local server_log="$case_dir/${QUALIFICATION_ROLE}-server.log"
  local start_file="$case_dir/${QUALIFICATION_ROLE}-measurement-start-utc.txt"
  if [[ "$QUALIFICATION_ROLE" == "combined" ]]; then
    server_log="$case_dir/server.log"
    start_file="$case_dir/measurement-start-utc.txt"
  fi
  [[ -f "$server_log" && -f "$start_file" ]] || return 1
  validate_case_results "$case_dir" >/dev/null 2>&1 || return 1
  validate_runtime_log "$server_log" "$start_file" \
    >/dev/null 2>&1
}

settle_after_ready() {
  if (( POST_READY_SETTLE_SECONDS > 0 )); then
    echo "==> post-ready settle ${POST_READY_SETTLE_SECONDS}s"
    sleep "$POST_READY_SETTLE_SECONDS"
  fi
}

warm_runtime_case() {
  local port=$1 model_name=$2 case_dir=$3
  if [[ "$RUNTIME_WARMUP" != "1" ]]; then
    return 0
  fi

  if [[ "$RUN_DECODE" == "1" ]]; then
    if [[ -n "$RUNTIME_WARMUP_SPARSE_CONTEXT" ]]; then
      echo "==> runtime warmup sparse decode context=${RUNTIME_WARMUP_SPARSE_CONTEXT}"
      python3 "$BENCH" \
        --host localhost \
        --port "$port" \
        --model "$model_name" \
        --concurrency "$DECODE_CONCURRENCY" \
        --contexts "$RUNTIME_WARMUP_SPARSE_CONTEXT" \
        --duration "$RUNTIME_WARMUP_SPARSE_DURATION" \
        --decode-warmup-seconds 0 \
        --max-tokens 64 \
        --max-total-tokens "$DECODE_TOKEN_BUDGET" \
        --skip-prefill \
        --display-mode plain \
        --no-hw-monitor \
        --output "$case_dir/warmup-sparse-decode.json" \
        > "$case_dir/warmup-sparse-decode.log" 2>&1
    fi

    echo "==> runtime warmup decode ${DECODE_CONCURRENCY}"
    python3 "$BENCH" \
      --host localhost \
      --port "$port" \
      --model "$model_name" \
      --concurrency "$DECODE_CONCURRENCY" \
      --contexts "$DECODE_CONTEXTS" \
      --duration "$RUNTIME_WARMUP_DECODE_DURATION" \
      --decode-warmup-seconds 0 \
      --max-tokens "$DECODE_MAX_TOKENS" \
      --max-total-tokens "$DECODE_TOKEN_BUDGET" \
      --skip-prefill \
      --display-mode plain \
      --no-hw-monitor \
      --output "$case_dir/warmup-decode.json" \
      > "$case_dir/warmup-decode.log" 2>&1
  fi

  if [[ "$RUN_PREFILL" == "1" ]]; then
    echo "==> runtime warmup prefill ${PREFILL_CONTEXTS}"
    python3 "$BENCH" \
      --host localhost \
      --port "$port" \
      --model "$model_name" \
      --concurrency 1 \
      --contexts 0 \
      --prefill-only \
      --standalone-prefill \
      --prefill-contexts "$PREFILL_CONTEXTS" \
      --prefill-duration "$RUNTIME_WARMUP_PREFILL_DURATION" \
      --max-tokens "$DECODE_MAX_TOKENS" \
      --max-total-tokens "$DECODE_TOKEN_BUDGET" \
      --display-mode plain \
      --no-hw-monitor \
      --output "$case_dir/warmup-prefill.json" \
      > "$case_dir/warmup-prefill.log" 2>&1
  fi

  if (( POST_WARMUP_SETTLE_SECONDS > 0 )); then
    echo "==> post-warmup settle ${POST_WARMUP_SETTLE_SECONDS}s"
    sleep "$POST_WARMUP_SETTLE_SECONDS"
  fi
}

launch_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir model_name cache_dir
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  case_dir="$OUT/$label"
  cache_dir=${SHARED_CACHE:-$case_dir/cache}
  model_name=$(model_name_for_mode "$mode")
  mkdir -p "$case_dir"

  echo "==> launch $label on GPUs=$gpus port=$port"
  if ! IMAGE="$IMAGE" \
    NAME="$name" \
    PORT="$port" \
    GPUS="$gpus" \
    TP="$tp" \
    BACKEND="$backend" \
    MODE="$mode" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_BATCHED="$MAX_BATCHED" \
    GPU_MEM="$GPU_MEM" \
    ALLREDUCE_MODE="$ALLREDUCE_MODE" \
    ENABLE_TOPO_PIN="$ENABLE_TOPO_PIN" \
    CACHE="$cache_dir" \
    CONTAINER_TMP="$case_dir/tmp" \
    "$LAUNCHER" 2>&1 | tee "$case_dir/launch.log"; then
    return 1
  fi
}

bench_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir model_name
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  case_dir="$OUT/$label"
  model_name=$(model_name_for_mode "$mode")
  mkdir -p "$case_dir"

  curl -fsS "http://127.0.0.1:${port}/version" > "$case_dir/version.json" || true
  curl -fsS "http://127.0.0.1:${port}/v1/models" > "$case_dir/models.json" || return 1

  local server_log="$case_dir/${QUALIFICATION_ROLE}-server.log"
  local warmup_server_log="$case_dir/${QUALIFICATION_ROLE}-warmup-server.log"
  local runtime_start_file="$case_dir/${QUALIFICATION_ROLE}-measurement-start-utc.txt"
  if [[ "$QUALIFICATION_ROLE" == "combined" ]]; then
    server_log="$case_dir/server.log"
    warmup_server_log="$case_dir/warmup-server.log"
    runtime_start_file="$case_dir/measurement-start-utc.txt"
  fi

  warm_runtime_case "$port" "$model_name" "$case_dir" || return 1
  docker logs "$name" > "$warmup_server_log" 2>&1 || true
  date -u +%Y-%m-%dT%H:%M:%SZ > "$runtime_start_file"

  if [[ "$RUN_DECODE" == "1" ]]; then
    echo "==> decode $label"
    if ! python3 "$BENCH" \
      --host localhost \
      --port "$port" \
      --model "$model_name" \
      --concurrency "$DECODE_CONCURRENCY" \
      --contexts "$DECODE_CONTEXTS" \
      --duration "$DECODE_DURATION" \
      --max-tokens "$DECODE_MAX_TOKENS" \
      --max-total-tokens "$DECODE_TOKEN_BUDGET" \
      --skip-prefill \
      --display-mode plain \
      --no-hw-monitor \
      --coding-peak \
      --coding-peak-runs 5 \
      --output "$case_dir/decode.json" \
      2>&1 | tee "$case_dir/decode.log"; then
      return 1
    fi
  fi

  if [[ "$RUN_PREFILL" == "1" ]]; then
    echo "==> prefill $label"
    if ! python3 "$BENCH" \
      --host localhost \
      --port "$port" \
      --model "$model_name" \
      --concurrency 1 \
      --contexts 0 \
      --prefill-only \
      --standalone-prefill \
      --prefill-contexts "$PREFILL_CONTEXTS" \
      --prefill-duration "$PREFILL_DURATION" \
      --max-tokens "$DECODE_MAX_TOKENS" \
      --max-total-tokens "$DECODE_TOKEN_BUDGET" \
      --display-mode plain \
      --no-hw-monitor \
      --output "$case_dir/prefill.json" \
      2>&1 | tee "$case_dir/prefill.log"; then
      return 1
    fi
  fi

  curl -fsS "http://127.0.0.1:${port}/metrics" > "$case_dir/final-metrics.prom" || true
  docker logs "$name" > "$server_log" 2>&1 || true
  validate_case_results "$case_dir" || return 1
  validate_runtime_log \
    "$server_log" \
    "$runtime_start_file" || return 1
  docker rm -f "$name" >/dev/null 2>&1 || true
  append_case_summary DONE "$label" "$case_dir"
}

fail_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name case_dir
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  case_dir="$OUT/$label"
  mkdir -p "$case_dir"
  docker logs "$name" > "$case_dir/${QUALIFICATION_ROLE}-server.log" 2>&1 || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  append_case_summary FAILED "$label" "$case_dir" || true
}

run_case() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  local label name
  label="tp${tp}-${backend}-${mode}"
  name="${CONTAINER_PREFIX}-${label}"
  launch_case "$tp" "$backend" "$mode" "$gpus" "$port" || return 1
  wait_for_server "$name" "$port" || return 1
  settle_after_ready
  bench_case "$tp" "$backend" "$mode" "$gpus" "$port"
}

run_case_guarded() {
  local tp=$1 backend=$2 mode=$3 gpus=$4 port=$5
  if run_case "$tp" "$backend" "$mode" "$gpus" "$port"; then
    return 0
  fi
  fail_case "$tp" "$backend" "$mode" "$gpus" "$port"
  return 1
}

run_tp_matrix() {
  local tp=$1
  local -a cases planned wave_pids
  local gpu_groups
  gpu_groups="$(gpu_groups_for_tp "$tp" | paste -sd' ' -)"
  mapfile -t planned < <(
    python3 "$LAYOUT_PLANNER" \
      --tp "$tp" --backends "$BACKENDS" --modes "$MODES" \
      --gpu-groups "$gpu_groups" --port-base "$PORT_BASE"
  )
  cases=()
  local planned_case
  for planned_case in "${planned[@]}"; do
    local c_tp backend mode ordinal wave slot gpus port
    IFS=$'\t' read -r c_tp backend mode ordinal wave slot gpus port \
      <<<"$planned_case"
    local label="tp${c_tp}-${backend}-${mode}"
    if [[ "$RESUME" == "1" ]] \
      && validate_reusable_case "$OUT/$label"; then
      append_case_summary REUSED "$label" "$OUT/$label"
    else
      cases+=("$planned_case")
    fi
  done

  local idx=0 failures=0
  while (( idx < ${#cases[@]} )); do
    wave_pids=()
    local -a ready_cases bench_pids
    ready_cases=()
    bench_pids=()
    local c_tp c_backend c_mode c_ordinal c_wave c_slot c_gpus c_port
    IFS=$'\t' read -r c_tp c_backend c_mode c_ordinal c_wave c_slot c_gpus c_port \
      <<<"${cases[$idx]}"
    local active_wave=$c_wave
    while (( idx < ${#cases[@]} )); do
      IFS=$'\t' read -r c_tp c_backend c_mode c_ordinal c_wave c_slot c_gpus c_port \
        <<<"${cases[$idx]}"
      [[ "$c_wave" == "$active_wave" ]] || break
      if [[ "$SYNC_WAVE_READY" == "1" ]]; then
        local spec="$c_tp:$c_backend:$c_mode:$c_gpus:$c_port"
        if launch_case "$c_tp" "$c_backend" "$c_mode" "$c_gpus" "$c_port" \
          && wait_for_server \
            "${CONTAINER_PREFIX}-tp${c_tp}-${c_backend}-${c_mode}" "$c_port"; then
          # Load and fully initialize each server before starting the next one.
          # Benchmark clients are launched only after this whole wave is ready.
          ready_cases+=("$spec")
        else
          fail_case "$c_tp" "$c_backend" "$c_mode" "$c_gpus" "$c_port"
          failures=$((failures + 1))
        fi
      else
        run_case_guarded "$c_tp" "$c_backend" "$c_mode" "$c_gpus" "$c_port" &
        wave_pids+=("$!")
      fi
      idx=$((idx + 1))
    done
    if [[ "$SYNC_WAVE_READY" == "1" ]]; then
      settle_after_ready
      for spec in "${ready_cases[@]}"; do
        IFS=: read -r c_tp c_backend c_mode c_gpus c_port <<<"$spec"
        bench_case "$c_tp" "$c_backend" "$c_mode" "$c_gpus" "$c_port" &
        bench_pids+=("$!")
      done
      for i in "${!bench_pids[@]}"; do
        IFS=: read -r c_tp c_backend c_mode c_gpus c_port <<<"${ready_cases[$i]}"
        if ! wait "${bench_pids[$i]}"; then
          fail_case "$c_tp" "$c_backend" "$c_mode" "$c_gpus" "$c_port"
          failures=$((failures + 1))
        fi
      done
    else
      for pid in "${wave_pids[@]}"; do
        if ! wait "$pid"; then
          failures=$((failures + 1))
        fi
      done
    fi
  done

  if (( failures > 0 )); then
    echo "TP${tp} completed with $failures failed case(s)" >&2
    return 1
  fi
}

chmod +x "$LAUNCHER"
record_repro_artifacts
{
  printf 'image=%s\nout=%s\nprogress_file=%s\nlauncher=%s\nbench=%s\n' \
    "$IMAGE" "$OUT" "$PROGRESS_FILE" "$LAUNCHER" "$BENCH"
  printf 'result_renderer=%s\n' "${RESULT_RENDERER:-none}"
  printf 'max_num_seqs=%s\nport_base=%s\nstartup_timeout=%s\n' \
    "$MAX_NUM_SEQS" "$PORT_BASE" "$STARTUP_TIMEOUT"
  printf 'qualification_role=%s\nrun_decode=%s\nrun_prefill=%s\n' \
    "$QUALIFICATION_ROLE" "$RUN_DECODE" "$RUN_PREFILL"
  printf 'max_model_len=%s\nmax_batched=%s\ngpu_mem=%s\nallreduce_mode=%s\n' \
    "${MAX_MODEL_LEN:-launcher-default}" "${MAX_BATCHED:-launcher-default}" \
    "${GPU_MEM:-launcher-default}" "${ALLREDUCE_MODE:-launcher-default}"
  printf 'post_ready_settle_seconds=%s\n' "$POST_READY_SETTLE_SECONDS"
  printf 'runtime_warmup=%s\nruntime_warmup_decode_duration=%s\n' \
    "$RUNTIME_WARMUP" "$RUNTIME_WARMUP_DECODE_DURATION"
  printf 'runtime_warmup_sparse_context=%s\nruntime_warmup_sparse_duration=%s\n' \
    "${RUNTIME_WARMUP_SPARSE_CONTEXT:-disabled}" "$RUNTIME_WARMUP_SPARSE_DURATION"
  printf 'runtime_warmup_prefill_duration=%s\npost_warmup_settle_seconds=%s\n' \
    "$RUNTIME_WARMUP_PREFILL_DURATION" "$POST_WARMUP_SETTLE_SECONDS"
  printf 'decode_concurrency=%s\ndecode_contexts=%s\ndecode_duration=%s\n' \
    "$DECODE_CONCURRENCY" "$DECODE_CONTEXTS" "$DECODE_DURATION"
  printf 'decode_max_tokens=%s\ndecode_token_budget=%s\n' \
    "$DECODE_MAX_TOKENS" "$DECODE_TOKEN_BUDGET"
  printf 'prefill_contexts=%s\nprefill_duration=%s\n' \
    "$PREFILL_CONTEXTS" "$PREFILL_DURATION"
  printf 'backends=%s\nmodes=%s\ntps=%s\nsync_wave_ready=%s\nenable_topo_pin=%s\nvllm_patch_file=%s\n' \
    "$BACKENDS" "$MODES" "$TPS" "$SYNC_WAVE_READY" "$ENABLE_TOPO_PIN" "$VLLM_PATCH_FILE"
  printf 'container_prefix=%s\ngpu_groups_tp2=%s\ngpu_groups_tp4=%s\nresume=%s\n' \
    "$CONTAINER_PREFIX" "$GPU_GROUPS_TP2" "$GPU_GROUPS_TP4" "$RESUME"
  printf 'shared_cache=%s\n' "${SHARED_CACHE:-disabled}"
} | tee "$OUT/run-config-${QUALIFICATION_ROLE}.txt"
printf '%s sweep_start role=%s image=%s out=%s backends=%s modes=%s tps=%s sync_wave_ready=%s enable_topo_pin=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$QUALIFICATION_ROLE" "$IMAGE" "$OUT" "$BACKENDS" "$MODES" "$TPS" "$SYNC_WAVE_READY" "$ENABLE_TOPO_PIN" \
  >> "$PROGRESS_FILE"
for tp in $(split_csv "$TPS"); do
  run_tp_matrix "$tp"
done
printf '%s sweep_done role=%s out=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$QUALIFICATION_ROLE" "$OUT" \
  >> "$PROGRESS_FILE"
