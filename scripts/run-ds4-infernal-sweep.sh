#!/usr/bin/env bash
set -euo pipefail

# Run the DeepSeek-V4-Flash-0731 Infernal Invocation qualification matrix on a
# 16-GPU RTX PRO 6000 Blackwell host. The generic scheduler starts each server
# in a wave serially so its shared JIT cache is populated without concurrent
# writers, then benchmarks disjoint GPU groups concurrently.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GENERIC_SWEEP_SCRIPT=${GENERIC_SWEEP_SCRIPT:-${SCRIPT_DIR}/run-ds4-v9-sweep.sh}

export IMAGE="${IMAGE:-voipmonitor/vllm:infernal-invocation-vllm7ed814e-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r7}"
export OUT="${OUT:-/root/bench-results/ds4-infernal-invocation-r7-$(date -u +%Y%m%d-%H%M%S)}"
export PROGRESS_FILE="${PROGRESS_FILE:-${OUT}/progress.log}"
export LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/run-ds4-infernal-server.sh}"
export SWEEP_SCRIPT="${SWEEP_SCRIPT:-${SCRIPT_DIR}/run-ds4-infernal-sweep.sh}"
export RESULT_RENDERER="${RESULT_RENDERER:-${SCRIPT_DIR}/render-ds4-infernal-results.py}"
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-/root/rtx6kpro/.source-lock-has-no-runtime-overlay}"
export CONTAINER_PREFIX="${CONTAINER_PREFIX:-ds4-infernal-invocation-r7}"
export STANDARD_SERVED_MODEL_NAME=DeepSeek-V4-Flash-0731
export DSPARK_SERVED_MODEL_NAME=DeepSeek-V4-Flash-0731

export GPU_GROUPS_TP2="${GPU_GROUPS_TP2:-0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15}"
export GPU_GROUPS_TP4="${GPU_GROUPS_TP4:-0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15}"
export TPS="${TPS:-2,4}"
export BACKENDS="${BACKENDS:-b12x-a16,b12x-a8,b12x-a8-dglin}"
export MODES="${MODES:-dspark-mtp0,dspark-k5,dspark-k7,dspark-k7-dynamic}"
export MAX_BATCHED="${MAX_BATCHED:-8192}"
export GPU_MEM="${GPU_MEM:-0.975}"
export ALLREDUCE_MODE="${ALLREDUCE_MODE:-auto}"
export SHARED_CACHE="${SHARED_CACHE:-/root/.cache/ds4-infernal-cu133-sweep}"
export SYNC_WAVE_READY=1
export ENABLE_TOPO_PIN=0
export PORT_BASE="${PORT_BASE:-5000}"
# Release qualification measures the latency-sensitive single-user path by
# default. Operators can still request a concurrency matrix explicitly.
export DECODE_CONCURRENCY="${DECODE_CONCURRENCY:-1}"
export DECODE_CONTEXTS="${DECODE_CONTEXTS:-0}"
export DECODE_DURATION="${DECODE_DURATION:-30}"
export PREFILL_CONTEXTS="${PREFILL_CONTEXTS:-8k,64k,131008}"
export PREFILL_DURATION="${PREFILL_DURATION:-10}"
export POST_READY_SETTLE_SECONDS="${POST_READY_SETTLE_SECONDS:-30}"
export RUNTIME_WARMUP_DECODE_DURATION="${RUNTIME_WARMUP_DECODE_DURATION:-5}"
# Start one warmup request beyond the sparse-attention threshold so long-decode
# kernels are compiled before the benchmark log boundary.
export RUNTIME_WARMUP_SPARSE_CONTEXT="${RUNTIME_WARMUP_SPARSE_CONTEXT:-4096}"
export RUNTIME_WARMUP_SPARSE_DURATION="${RUNTIME_WARMUP_SPARSE_DURATION:-2}"
export RUNTIME_WARMUP_PREFILL_DURATION="${RUNTIME_WARMUP_PREFILL_DURATION:-2}"
export POST_WARMUP_SETTLE_SECONDS="${POST_WARMUP_SETTLE_SECONDS:-30}"

# Decode reserves the CC1 FULL-graph envelope and enough KV for an 8192-token
# completion. GRAPH=auto still includes every target and draft row reachable by
# one request. Prefill uses MNS16 and a larger memory budget so TP2 K7 profiles
# can reserve the 131072-token model contract. The 131008-token prefill target
# leaves output space while remaining in the nominal 128k benchmark class.
RUN_DECODE=1 \
RUN_PREFILL=0 \
QUALIFICATION_ROLE=decode \
MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-1}" \
MAX_MODEL_LEN="${DECODE_MAX_MODEL_LEN:-10240}" \
GPU_MEM="${DECODE_GPU_MEM:-${GPU_MEM}}" \
"${GENERIC_SWEEP_SCRIPT}" "$@"

RUN_DECODE=0 \
RUN_PREFILL=1 \
QUALIFICATION_ROLE=prefill \
MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-16}" \
MAX_MODEL_LEN="${PREFILL_MAX_MODEL_LEN:-131072}" \
GPU_MEM="${PREFILL_GPU_MEM:-0.98}" \
"${GENERIC_SWEEP_SCRIPT}" "$@"

exec python3 "${RESULT_RENDERER}" "${OUT}"
