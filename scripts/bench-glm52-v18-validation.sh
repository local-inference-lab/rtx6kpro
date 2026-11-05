#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v18-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718}"
EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:a1202a5b9f712910306b684de05c4bbea05c37609c0eb0e9394fd7d980fa49ac}"
BENCH="${BENCH:-/root/llm-inference-bench/llm_decode_bench.py}"
NVFP4_MODEL="${NVFP4_MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}"
MXFP4_MODEL="${MXFP4_MODEL:-/root/models/GLM-5.2-BF16-AMDMXFP4experts}"
NF3_MODEL="${NF3_MODEL:-/root/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v18-validation-$(date -u +%Y%m%dT%H%M%SZ)}"
RELEASE_LABEL="${RELEASE_LABEL:-v18}"
NAME_PREFIX="${NAME_PREFIX:-glm52-v18-validation}"

PORT_A="${PORT_A:-8190}"
PORT_B="${PORT_B:-8191}"
GPU_A_TP8="${GPU_A_TP8:-0,1,2,3,4,5,6,7}"
GPU_B_TP8="${GPU_B_TP8:-8,9,10,11,12,13,14,15}"
GPU_A_TP6="${GPU_A_TP6:-0,1,2,3,4,5}"
GPU_B_TP6="${GPU_B_TP6:-8,9,10,11,12,13}"
GPU_A_TP4="${GPU_A_TP4:-0,1,2,3}"
GPU_B_TP4="${GPU_B_TP4:-8,9,10,11}"
CPU_A="${CPU_A:-0-31,64-95}"
CPU_B="${CPU_B:-32-63,96-127}"
CACHE_A="${CACHE_A:-/root/.cache/vllm-glm52-v18/validation/slot-a}"
CACHE_B="${CACHE_B:-/root/.cache/vllm-glm52-v18/validation/slot-b}"
TMP_ROOT="${TMP_ROOT:-/root/vllm/tmp/glm52-v18-validation}"

SETTLE_SECONDS="${SETTLE_SECONDS:-30}"
DECODE_DURATION="${DECODE_DURATION:-30}"
PREFILL_DURATION="${PREFILL_DURATION:-10}"
PREFILL_WARMUP_RUNS="${PREFILL_WARMUP_RUNS:-1}"
PREFILL_REPEATS="${PREFILL_REPEATS:-3}"
BETWEEN_RUNS_SECONDS="${BETWEEN_RUNS_SECONDS:-10}"
TOKEN_TARGETING="${TOKEN_TARGETING:-estimate}"
NEEDLE_GATE="${NEEDLE_GATE:-1}"
NEEDLE_SIZES="${NEEDLE_SIZES:-8,64,96}"
NEEDLE_REQUIRED_CLEAN_K="${NEEDLE_REQUIRED_CLEAN_K:-96}"
SPARSE_MLA_FORCE_MQA="${SPARSE_MLA_FORCE_MQA:-0}"
CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF:-expandable_segments:True}"
TP8_MAX_MODEL_LEN="${TP8_MAX_MODEL_LEN:-131072}"
TP6_MAX_MODEL_LEN="${TP6_MAX_MODEL_LEN:-128000}"
TP4_MAX_MODEL_LEN="${TP4_MAX_MODEL_LEN:-131072}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
ONLINE_MXFP8_CONFIG_JSON="${ONLINE_MXFP8_CONFIG_JSON:-}"

NAME_A="${NAME_PREFIX}-a"
NAME_B="${NAME_PREFIX}-b"
PROGRESS_FILE="${RESULT_ROOT}/progress.log"

mkdir -p "${RESULT_ROOT}" "${CACHE_A}" "${CACHE_B}" "${TMP_ROOT}"
[[ -f "${BENCH}" ]] || { echo "missing benchmark client: ${BENCH}" >&2; exit 2; }
[[ "${TOKEN_TARGETING}" =~ ^(estimate|exact)$ ]] || {
  echo "TOKEN_TARGETING must be estimate or exact" >&2
  exit 2
}
[[ "$(docker image inspect "${IMAGE}" --format '{{.Id}}')" == "${EXPECTED_IMAGE_ID}" ]] || {
  echo "image ID does not match the validated ${RELEASE_LABEL} image" >&2
  exit 2
}

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${PROGRESS_FILE}"
}

safe_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' |
    sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//; s/-+$//'
}

case_vars() {
  local key="$1"
  MODEL_PATH= DISPLAY_NAME= MOE_MODE= QUANTIZATION= ONLINE_QUANT=
  QUANTIZATION_CONFIG_JSON=
  MODEL_FAMILY=glm52 KV_CACHE_DTYPE=fp8
  case "${key}" in
    nvfp4-a4-orig)
      MODEL_PATH="${NVFP4_MODEL}"; DISPLAY_NAME="Luke NVFP4 A4 original"
      MOE_MODE=a4; QUANTIZATION=modelopt_fp4; ONLINE_QUANT=none ;;
    nvfp4-a4-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"; DISPLAY_NAME="Luke NVFP4 A4 online MXFP8"
      MOE_MODE=a4; QUANTIZATION=modelopt_fp4; ONLINE_QUANT=mxfp8
      QUANTIZATION_CONFIG_JSON="${ONLINE_MXFP8_CONFIG_JSON}" ;;
    nvfp4-a16-orig)
      MODEL_PATH="${NVFP4_MODEL}"; DISPLAY_NAME="Luke NVFP4 A16 original"
      MOE_MODE=a16; QUANTIZATION=modelopt_fp4; ONLINE_QUANT=none ;;
    nvfp4-a16-online-mxfp8)
      MODEL_PATH="${NVFP4_MODEL}"; DISPLAY_NAME="Luke NVFP4 A16 online MXFP8"
      MOE_MODE=a16; QUANTIZATION=modelopt_fp4; ONLINE_QUANT=mxfp8
      QUANTIZATION_CONFIG_JSON="${ONLINE_MXFP8_CONFIG_JSON}" ;;
    mxfp4-a8-orig)
      MODEL_PATH="${MXFP4_MODEL}"; DISPLAY_NAME="AMD MXFP4 experts A8 original"
      MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4; ONLINE_QUANT=none ;;
    mxfp4-a8-online-mxfp8)
      MODEL_PATH="${MXFP4_MODEL}"; DISPLAY_NAME="AMD MXFP4 experts A8 online MXFP8"
      MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4; ONLINE_QUANT=mxfp8
      QUANTIZATION_CONFIG_JSON="${ONLINE_MXFP8_CONFIG_JSON}" ;;
    mxfp4-a8-online-fp8)
      MODEL_PATH="${MXFP4_MODEL}"; DISPLAY_NAME="AMD MXFP4 experts A8 online FP8"
      MOE_MODE=force-a8-experimental; QUANTIZATION=mxfp4; ONLINE_QUANT=fp8 ;;
    nf3-hybrid-a16)
      MODEL_PATH="${NF3_MODEL}"; DISPLAY_NAME="NVFP4/NF3 hybrid A16"
      MODEL_FAMILY=glm52-hybrid; MOE_MODE=a16; QUANTIZATION=nvfp4_nf3_hybrid
      ONLINE_QUANT=nf3-mxfp8; KV_CACHE_DTYPE=nvfp4_ds_mla ;;
    *) echo "unknown case: ${key}" >&2; return 2 ;;
  esac
}

stop_servers() {
  docker rm -f "${NAME_A}" "${NAME_B}" >/dev/null 2>&1 || true
}

cleanup() {
  if [[ "${KEEP_SERVERS}" != 1 ]]; then stop_servers; fi
}
trap cleanup EXIT INT TERM

config_path() {
  local key="$1" tp="$2" dcp="$3" mtp="$4"
  printf '%s/tp%s-dcp%s-mtp%s/%s' "${RESULT_ROOT}" "${tp}" "${dcp}" "${mtp}" "${key}"
}

result_complete() {
  [[ "${FORCE_RERUN}" != 1 && -s "$1/summary.json" && -f "$1/complete" ]]
}

start_case() {
  local spec="$1" slot="$2"
  read -r key tp dcp mtp <<<"${spec}"
  case_vars "${key}"
  [[ -d "${MODEL_PATH}" ]] || { echo "missing model: ${MODEL_PATH}" >&2; return 2; }

  local name port cpus gpus cache tmp out served
  if [[ "${slot}" == a ]]; then
    name="${NAME_A}"; port="${PORT_A}"; cpus="${CPU_A}"; cache="${CACHE_A}"
    case "${tp}" in 8) gpus="${GPU_A_TP8}";; 6) gpus="${GPU_A_TP6}";; 4) gpus="${GPU_A_TP4}";; esac
  else
    name="${NAME_B}"; port="${PORT_B}"; cpus="${CPU_B}"; cache="${CACHE_B}"
    case "${tp}" in 8) gpus="${GPU_B_TP8}";; 6) gpus="${GPU_B_TP6}";; 4) gpus="${GPU_B_TP4}";; esac
  fi
  out="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}")"
  served="GLM-5.2-${RELEASE_LABEL}-$(safe_name "${key}-tp${tp}-dcp${dcp}-mtp${mtp}")"
  tmp="${TMP_ROOT}/slot-${slot}"
  mkdir -p "${out}" "${cache}" "${tmp}"
  rm -f "${out}/complete"
  docker rm -f "${name}" >/dev/null 2>&1 || true

  local max_len="${TP8_MAX_MODEL_LEN}" max_seqs=32 graph=128 batched=8192 gmu=0.90
  if [[ "${tp}" == 6 ]]; then
    max_len="${TP6_MAX_MODEL_LEN}"; max_seqs=16; graph=64; batched=4096; gmu=0.950
  elif [[ "${tp}" == 4 ]]; then
    max_len="${TP4_MAX_MODEL_LEN}"; max_seqs=8; graph=64; batched=3072; gmu=0.960
  fi

  progress "START slot=${slot} case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} gpus=${gpus}"
  docker run -d --name "${name}" --network host --ipc host --privileged --init \
    --gpus all --cpuset-cpus "${cpus}" --shm-size 32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /root/models:/root/models:ro \
    -v /root/.cache/huggingface:/root/.cache/huggingface:rw \
    -v "${cache}:/cache:rw" -v "${tmp}:/container-tmp:rw" \
    -e MODEL_FAMILY="${MODEL_FAMILY}" -e GPUS="${gpus}" -e MODEL="${MODEL_PATH}" \
    -e SERVED_MODEL_NAME="${served}" -e PORT="${port}" -e TP="${tp}" -e DCP="${dcp}" \
    -e DCP_BACKEND=a2a -e DCP_A2A_MAX_TOKENS=64 -e DCP_A2A_LARGE_BACKEND=ag_rs \
    -e DCP_PREFILL_WORKSPACE=auto -e MTP="${mtp}" -e MAX_NUM_SEQS="${max_seqs}" \
    -e GRAPH="${graph}" -e MAX_MODEL_LEN="${max_len}" -e MAX_BATCHED_TOKENS="${batched}" \
    -e GPU_MEMORY_UTILIZATION="${gmu}" -e MOE_MODE="${MOE_MODE}" -e MOE_BACKEND=b12x \
    -e LINEAR_BACKEND=auto -e QUANTIZATION="${QUANTIZATION}" -e ONLINE_QUANT="${ONLINE_QUANT}" \
    -e QUANTIZATION_CONFIG_JSON="${QUANTIZATION_CONFIG_JSON}" \
    -e SPARSE_MLA_FORCE_MQA="${SPARSE_MLA_FORCE_MQA}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${CUDA_ALLOC_CONF}" \
    -e F8_DMA=0 -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
    -e LOAD_FORMAT=instanttensor -e INSTANTTENSOR_BACKEND=BUFFERED \
    --entrypoint /usr/local/bin/serve-gilded-gnosis.sh "${IMAGE}" > "${out}/container.id"
  docker inspect "${name}" > "${out}/container.inspect.json"
  [[ "$(docker inspect "${name}" --format '{{.Image}}')" == "${EXPECTED_IMAGE_ID}" ]]
  if docker inspect "${name}" --format '{{range .Mounts}}{{println .Destination}}{{end}}' |
      grep -Eq '^(/opt/vllm|/opt/venv|.*/site-packages)(/|$)'; then
    echo "source overlay detected in ${name}" >&2
    return 3
  fi
  printf '%s\n' "${name}" > "${out}/container.name"
  printf '%s\n' "${port}" > "${out}/port"
  printf '%s\n' "${served}" > "${out}/served-model.name"
  printf '%s\n' "${DISPLAY_NAME}" > "${out}/display.name"
}

wait_ready() {
  local spec="$1"
  read -r key tp dcp mtp <<<"${spec}"
  local out name port
  out="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}")"
  name="$(cat "${out}/container.name")"; port="$(cat "${out}/port")"
  for _ in $(seq 1 1800); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      docker logs "${name}" > "${out}/server.ready.log" 2>&1
      grep -Fq 'Loading safetensors using InstantTensor loader' "${out}/server.ready.log"
      grep -Fq 'vLLM is using nccl==2.30.4' "${out}/server.ready.log"
      progress "READY case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} port=${port}"
      return 0
    fi
    if [[ "$(docker inspect "${name}" --format '{{.State.Status}}' 2>/dev/null || true)" != running ]]; then
      docker logs "${name}" > "${out}/server.failed.log" 2>&1 || true
      progress "FAILED case=${key} log=${out}/server.failed.log"
      return 1
    fi
    sleep 2
  done
  docker logs "${name}" > "${out}/server.timeout.log" 2>&1 || true
  return 1
}

run_case() {
  local spec="$1"
  read -r key tp dcp mtp <<<"${spec}"
  case_vars "${key}"
  local out name port served
  out="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}")"
  name="$(cat "${out}/container.name")"; port="$(cat "${out}/port")"
  served="$(cat "${out}/served-model.name")"

  progress "CORRECTNESS case=${key} tp=${tp} dcp=${dcp} mtp=${mtp}"
  python3 - "${port}" "${served}" "${out}/correctness.json" <<'PY'
import json, sys, urllib.request
port, model, output = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Write a short Python function that merges two sorted integer lists."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=900) as response:
    result = json.load(response)
text = result["choices"][0]["message"].get("reasoning") or result["choices"][0]["message"].get("content") or ""
assert text.strip() and text.count("!") < 16
open(output, "w").write(json.dumps(result, indent=2) + "\n")
PY

  if [[ "${NEEDLE_GATE}" == 1 ]]; then
    progress "NEEDLE case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} sizes=${NEEDLE_SIZES}"
    python3 "$(dirname "${BASH_SOURCE[0]}")/needle_gate.py" \
      --host 127.0.0.1 --port "${port}" --model "${served}" \
      --sizes "${NEEDLE_SIZES}" --required-clean-k "${NEEDLE_REQUIRED_CLEAN_K}" \
      --output "${out}/needle-gate.json" > "${out}/needle-gate.log" 2>&1 \
      || { cat "${out}/needle-gate.log"; echo "needle gate FAILED for case=${key}"; exit 1; }
  fi

  progress "DECODE case=${key} tp=${tp} dcp=${dcp} mtp=${mtp}"
  docker logs "${name}" > "${out}/server.before-decode.log" 2>&1
  local log_lines_before
  log_lines_before="$(wc -l < "${out}/server.before-decode.log")"
  python3 "${BENCH}" --host 127.0.0.1 --port "${port}" --model "${served}" \
    --skip-prefill --contexts 0 --concurrency 1 --duration "${DECODE_DURATION}" \
    --max-tokens 2048 --no-hw-monitor --display-mode plain \
    --output "${out}/decode-cc1.json" > "${out}/decode-cc1.log" 2>&1
  docker logs "${name}" > "${out}/server.after-decode.log" 2>&1
  tail -n "+$((log_lines_before + 1))" "${out}/server.after-decode.log" \
    > "${out}/server.decode-window.log"

  progress "PREFILL_WARMUP case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} runs=${PREFILL_WARMUP_RUNS}"
  local warmup
  for warmup in $(seq 1 "${PREFILL_WARMUP_RUNS}"); do
    python3 "${BENCH}" --host 127.0.0.1 --port "${port}" --model "${served}" \
      --prefill-only --standalone-prefill --prefill-contexts 64k \
      --prefill-duration "${PREFILL_DURATION}" --token-targeting "${TOKEN_TARGETING}" \
      --max-tokens 1 --no-hw-monitor --display-mode plain \
      --output "${out}/prefill-warmup${warmup}.json" > "${out}/prefill-warmup${warmup}.log" 2>&1
    sleep "${BETWEEN_RUNS_SECONDS}"
  done

  progress "PREFILL case=${key} tp=${tp} dcp=${dcp} mtp=${mtp} repeats=${PREFILL_REPEATS}"
  local run
  for run in $(seq 1 "${PREFILL_REPEATS}"); do
    python3 "${BENCH}" --host 127.0.0.1 --port "${port}" --model "${served}" \
      --prefill-only --standalone-prefill --prefill-contexts 64k \
      --prefill-duration "${PREFILL_DURATION}" --token-targeting "${TOKEN_TARGETING}" \
      --max-tokens 1 --no-hw-monitor --display-mode plain \
      --output "${out}/prefill-run${run}.json" > "${out}/prefill-run${run}.log" 2>&1
    sleep "${BETWEEN_RUNS_SECONDS}"
  done
  docker logs "${name}" > "${out}/server.final.log" 2>&1
  if [[ "${MODEL_FAMILY}" == glm52 ]]; then
    case "${MOE_MODE}" in
      a16) grep -Fq 'B12X MoE force-A16 enabled: using quant_mode=w4a16' "${out}/server.final.log" ;;
      force-a8-experimental)
        grep -Fq 'B12X MoE force-A8 enabled: using quant_mode=w4a8_mx' "${out}/server.final.log" ;;
    esac
  else
    grep -Fq 'nvfp4_nf3_hybrid: armed exact TP4 Grid188 one-grid decode' "${out}/server.final.log"
    grep -Fq 'nvfp4_nf3_hybrid: executing TP4 Grid188 one-grid decode' "${out}/server.final.log"
    grep -Fq 'kv_cache_dtype=nvfp4_ds_mla' "${out}/server.final.log"
  fi
  case "${ONLINE_QUANT}" in
    mxfp8)
      grep -Fq 'ScaleDesc(dtype=torch.uint8, static=False' "${out}/server.final.log" ;;
    fp8)
      grep -Fq 'ScaleDesc(dtype=torch.float32, static=True' "${out}/server.final.log" ;;
  esac

  python3 - "${out}" "${tp}" "${dcp}" "${mtp}" "${PREFILL_REPEATS}" "${TOKEN_TARGETING}" <<'PY'
import json, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
tp, dcp, mtp, repeats = map(int, sys.argv[2:6])
token_targeting = sys.argv[6]
decode = json.loads((root / "decode-cc1.json").read_text())
row = decode["results"][0]
prefill = [
    float(json.loads((root / f"prefill-run{i}.json").read_text())["prefill"]["65536"]["tok_per_sec"])
    for i in range(1, repeats + 1)
]
log = (root / "server.final.log").read_text(errors="replace")
decode_log = (root / "server.decode-window.log").read_text(errors="replace")
kv = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)
accept_lengths = [
    float(value)
    for value in re.findall(r"Mean acceptance length: ([0-9.]+)", decode_log)
]
accept_rates = [
    float(value) / 100.0
    for value in re.findall(r"Avg Draft acceptance rate: ([0-9.]+)%", decode_log)
]
client_accept_length = float(row.get("server_spec_accept_length") or 0)
client_accept_rate = float(row.get("server_spec_accept_rate") or 0)
summary = {
    "decode_cc1": float(row["aggregate_tps"]),
    "prefill_64k_runs": prefill,
    "prefill_64k_median": statistics.median(prefill),
    "kv_tokens": int(kv.group(1).replace(",", "")) if kv else None,
    "mean_acceptance_length": (
        statistics.median(accept_lengths) if accept_lengths else client_accept_length
    ),
    "draft_acceptance_rate": (
        statistics.median(accept_rates) if accept_rates else client_accept_rate
    ),
    "client_acceptance_length": client_accept_length,
    "client_draft_acceptance_rate": client_accept_rate,
    "token_targeting": token_targeting,
    "fast_dcp_path": (
        "Using transient full-CKV gather for B12X sparse MLA prefill" in log
        and "Keeping local query heads for transient full-CKV B12X sparse MLA prefill" in log
    ),
}
fast_expected = (tp == 4 and dcp in (2, 4)) or (tp == 8 and dcp in (2, 4, 8))
if fast_expected:
    assert summary["fast_dcp_path"], "fast DCP path was not observed"
if tp == 6 and dcp > 1:
    assert not summary["fast_dcp_path"], "virtual TP6 unexpectedly used full-CKV"
if mtp:
    assert (
        summary["mean_acceptance_length"] > 1.0
        or summary["draft_acceptance_rate"] > 0.0
    ), "MTP accepted no draft tokens"
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(root, json.dumps(summary, sort_keys=True))
PY
  touch "${out}/complete"
  progress "DONE case=${key} tp=${tp} dcp=${dcp} mtp=${mtp}"
}

run_pair() {
  local a="$1" b="${2:-}"
  local key tp dcp mtp out_a out_b=
  read -r key tp dcp mtp <<<"${a}"; out_a="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}")"
  if [[ -n "${b}" ]]; then
    read -r key tp dcp mtp <<<"${b}"; out_b="$(config_path "${key}" "${tp}" "${dcp}" "${mtp}")"
  fi
  if result_complete "${out_a}" && { [[ -z "${out_b}" ]] || result_complete "${out_b}"; }; then
    progress "SKIP pair=${a}|${b}"
    return
  fi
  stop_servers
  if ! result_complete "${out_a}"; then start_case "${a}" a; fi
  if [[ -n "${b}" ]] && ! result_complete "${out_b}"; then start_case "${b}" b; fi
  local pids=() pid
  if ! result_complete "${out_a}"; then wait_ready "${a}" & pids+=("$!"); fi
  if [[ -n "${b}" ]] && ! result_complete "${out_b}"; then wait_ready "${b}" & pids+=("$!"); fi
  for pid in "${pids[@]}"; do wait "${pid}"; done
  progress "ALL_READY settle=${SETTLE_SECONDS} pair=${a}|${b}"
  sleep "${SETTLE_SECONDS}"
  if ! result_complete "${out_a}"; then run_case "${a}"; fi
  if [[ -n "${b}" ]] && ! result_complete "${out_b}"; then run_case "${b}"; fi
  if [[ "${KEEP_SERVERS}" != 1 ]]; then stop_servers; fi
}

declare -a CONFIGS=()
add_dcp1_mtp0() {
  CONFIGS+=(
    "nvfp4-a4-orig 8 1 0" "nvfp4-a4-online-mxfp8 8 1 0"
    "nvfp4-a16-orig 8 1 0" "nvfp4-a16-online-mxfp8 8 1 0"
    "mxfp4-a8-orig 8 1 0" "mxfp4-a8-online-mxfp8 8 1 0"
    "mxfp4-a8-online-fp8 8 1 0"
  )
}
add_dcp1_mtp3() {
  CONFIGS+=(
    "nvfp4-a4-orig 8 1 3" "nvfp4-a4-online-mxfp8 8 1 3"
    "nvfp4-a16-orig 8 1 3" "nvfp4-a16-online-mxfp8 8 1 3"
    "mxfp4-a8-orig 8 1 3" "mxfp4-a8-online-mxfp8 8 1 3"
  )
}
add_dcp_fast_mtp0() {
  local key
  for key in \
    nvfp4-a4-orig nvfp4-a4-online-mxfp8 \
    nvfp4-a16-orig nvfp4-a16-online-mxfp8 \
    mxfp4-a8-orig mxfp4-a8-online-mxfp8 mxfp4-a8-online-fp8; do
    CONFIGS+=("${key} 8 2 0" "${key} 8 4 0" "${key} 8 8 0")
  done
}
add_tp6_mtp3() {
  CONFIGS+=(
    "mxfp4-a8-orig 6 3 3" "mxfp4-a8-online-mxfp8 6 3 3"
    "mxfp4-a8-orig 6 6 3" "mxfp4-a8-online-mxfp8 6 6 3"
  )
}
add_nf3() {
  CONFIGS+=("nf3-hybrid-a16 4 4 0" "nf3-hybrid-a16 4 4 3")
}

mode="${1:-all}"
case "${mode}" in
  dcp1-mtp0) add_dcp1_mtp0 ;;
  dcp1-mtp3) add_dcp1_mtp3 ;;
  dcp-fast) add_dcp_fast_mtp0 ;;
  tp6-mtp3) add_tp6_mtp3 ;;
  nf3) add_nf3 ;;
  all) add_dcp1_mtp0; add_dcp1_mtp3; add_dcp_fast_mtp0; add_tp6_mtp3; add_nf3 ;;
  configs) shift; CONFIGS=("$@") ;;
  *) echo "usage: $0 {all|dcp1-mtp0|dcp1-mtp3|dcp-fast|tp6-mtp3|nf3|configs ...}" >&2; exit 2 ;;
esac

for ((i=0; i<${#CONFIGS[@]}; i+=2)); do
  run_pair "${CONFIGS[i]}" "${CONFIGS[i+1]:-}"
done

python3 - "${RESULT_ROOT}" "${TOKEN_TARGETING}" <<'PY'
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
token_targeting = sys.argv[2]
rows = []
for path in sorted(root.glob("tp*-dcp*-mtp*/*/summary.json")):
    topology, case = path.parts[-3:-1]
    match = re.fullmatch(r"tp(\d+)-dcp(\d+)-mtp(\d+)", topology)
    if not match:
        raise ValueError(f"invalid result topology: {topology}")
    tp, dcp, mtp = map(int, match.groups())
    data = json.loads(path.read_text())
    data.setdefault("token_targeting", token_targeting)
    rows.append({"case": case, "tp": tp, "dcp": dcp, "mtp": mtp, **data})
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (root / "summary.tsv").open("w") as f:
    f.write("case\ttp\tdcp\tmtp\tdecode_cc1\tprefill_64k\tkv_tokens\taccept_len\taccept_rate\tfast_dcp\n")
    for r in rows:
        f.write("\t".join(str(r.get(k, "")) for k in (
            "case", "tp", "dcp", "mtp", "decode_cc1", "prefill_64k_median",
            "kv_tokens", "mean_acceptance_length", "draft_acceptance_rate", "fast_dcp_path"
        )) + "\n")
print((root / "summary.tsv").read_text())
PY
progress "VALIDATION_COMPLETE root=${RESULT_ROOT}"
