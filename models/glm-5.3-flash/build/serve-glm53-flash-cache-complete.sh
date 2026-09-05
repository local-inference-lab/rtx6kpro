#!/usr/bin/env bash
set -euo pipefail

readonly cache_launcher=/usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh

# With no cache-mode selection, preserve the image's serving command. Selecting
# CACHE_MODE enables the cache-mode contract, including explicit split-cache
# geometry and optional external cache connectors.
if [[ -z ${CACHE_MODE+x} && -z ${LMCACHE_ENABLED+x} && \
  ${CACHE_CONFIG_DRY_RUN:-0} != 1 ]]; then
  exec "${cache_launcher}" "$@"
fi

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

require_positive_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[1-9][0-9]*$ ]] ||
    fail "${name} must be a positive integer; got ${value}"
}

require_nonnegative_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[0-9]+$ ]] ||
    fail "${name} must be a non-negative integer; got ${value}"
}

sanitize_namespace_component() {
  local value=$1
  value=${value//[^A-Za-z0-9._-]/_}
  [[ -n ${value} ]] || value=unset
  printf '%s' "${value}"
}

# Reject command-line values for settings owned by this launcher's environment
# contract. Emitting a setting twice lets argument order silently select a cache
# layout that does not match the persistent-object namespace.
for arg in "$@"; do
  case "${arg}" in
    --kv-cache-dtype | --kv-cache-dtype=* | \
      --kv-offloading-backend | --kv-offloading-backend=* | \
      --kv-offloading-size | --kv-offloading-size=* | \
      --kv-transfer-config | --kv-transfer-config=* | \
      --enable-cumem-allocator | \
      --prefix-cache-retention-interval | \
      --prefix-cache-retention-interval=*)
      fail "${arg%%=*} is managed by CACHE_MODE and the cache environment variables"
      ;;
  esac
done

cache_mode=${CACHE_MODE:-}
if [[ -z ${cache_mode} ]]; then
  if [[ ${LMCACHE_ENABLED:-0} == 1 ]]; then
    cache_mode=lmcache
  else
    cache_mode=vram
  fi
fi
case "${cache_mode}" in
  vram | native | lmcache) ;;
  *) fail "CACHE_MODE must be vram, native, or lmcache; got ${cache_mode}" ;;
esac

tp=${TP:-4}
dcp=${DCP:-1}
require_positive_integer TP "${tp}"
require_positive_integer DCP "${dcp}"
case "${tp}" in
  4 | 8) ;;
  *) fail "The cache-complete launcher supports TP=4 or TP=8; got ${tp}" ;;
esac
if ((tp % dcp != 0)); then
  fail "DCP must divide TP; got TP=${tp} DCP=${dcp}"
fi

interleave=${CP_KV_CACHE_INTERLEAVE_SIZE:-4}
require_positive_integer CP_KV_CACHE_INTERLEAVE_SIZE "${interleave}"
gather_selector=${DCP_CKV_GATHER:-auto}
case "${gather_selector}" in
  auto)
    if ((dcp > 1)); then
      resolved_gather=1
    else
      resolved_gather=0
    fi
    ;;
  0 | 1) resolved_gather=${DCP_CKV_GATHER} ;;
  *) fail "DCP_CKV_GATHER must be auto, 0, or 1; got ${gather_selector}" ;;
esac

kv_cache_quant=${KV_CACHE_QUANT:-}
if [[ -z ${kv_cache_quant} ]]; then
  configured_kv_cache_dtype=${KV_CACHE_DTYPE:-fp8}
  case "${configured_kv_cache_dtype}" in
    fp8 | fp8_e4m3 | fp8_ds_mla) kv_cache_quant=fp8_ds_mla ;;
    nvfp4_ds_mla) kv_cache_quant=nvfp4_ds_mla ;;
    *)
      fail "KV_CACHE_DTYPE is not representable by KV_CACHE_QUANT: ${configured_kv_cache_dtype}"
      ;;
  esac
fi
case "${kv_cache_quant}" in
  fp8_ds_mla)
    vllm_kv_cache_dtype=fp8
    lmcache_kv_cache_dtype=fp8_ds_mla
    ;;
  nvfp4_ds_mla)
    vllm_kv_cache_dtype=nvfp4_ds_mla
    lmcache_kv_cache_dtype=nvfp4_ds_mla
    ;;
  *)
    fail "KV_CACHE_QUANT must be fp8_ds_mla or nvfp4_ds_mla; got ${kv_cache_quant}"
    ;;
esac

# LMCache objects are global sequence ranges. DCP shards each target cache
# block across ranks, so automatic geometry resolves one complete per-rank
# page inside the configured retention interval. GPU-only and native-offload
# modes use the capacity-efficient 2,048-token page unless overridden.
if [[ -n ${GLM53_TARGET_BLOCK_SIZE:-} ]]; then
  target_block_size=${GLM53_TARGET_BLOCK_SIZE}
elif [[ ${cache_mode} == lmcache ]]; then
  target_block_size=auto
else
  target_block_size=2048
fi
mamba_block_size=${GLM53_MAMBA_BLOCK_SIZE:-auto}
if [[ ${target_block_size} != auto ]]; then
  require_positive_integer GLM53_TARGET_BLOCK_SIZE "${target_block_size}"
  ((target_block_size % 64 == 0)) ||
    fail "GLM53_TARGET_BLOCK_SIZE must be a multiple of 64; got ${target_block_size}"
fi
if [[ ${mamba_block_size} != auto ]]; then
  require_positive_integer GLM53_MAMBA_BLOCK_SIZE "${mamba_block_size}"
  ((mamba_block_size % 64 == 0)) ||
    fail "GLM53_MAMBA_BLOCK_SIZE must be a multiple of 64; got ${mamba_block_size}"
  if [[ ${target_block_size} != auto ]] &&
    ((mamba_block_size % target_block_size != 0 && target_block_size % mamba_block_size != 0)); then
    fail "GLM53_MAMBA_BLOCK_SIZE must be a multiple or divisor of GLM53_TARGET_BLOCK_SIZE"
  fi
fi

speculator=${SPECULATOR:-mtp}
case "${speculator}" in
  mtp)
    spec_depth=${MTP_DEPTH:-${MTP:-${NUM_SPECULATIVE_TOKENS:-0}}}
    require_nonnegative_integer MTP_DEPTH "${spec_depth}"
    ((spec_depth <= 5)) || fail "MTP_DEPTH must be between 0 and 5; got ${spec_depth}"
    ;;
  dflash | dflash2)
    speculator=dflash2
    spec_depth=${DFLASH_DEPTH:-${NUM_SPECULATIVE_TOKENS:-7}}
    require_positive_integer DFLASH_DEPTH "${spec_depth}"
    ((spec_depth <= 7)) ||
      fail "DFLASH_DEPTH must be between 1 and 7; got ${spec_depth}"
    ;;
  *) fail "SPECULATOR must be mtp or dflash2; got ${speculator}" ;;
esac

export TP=${tp}
export DCP=${dcp}
export SPECULATOR=${speculator}
export NUM_SPECULATIVE_TOKENS=${spec_depth}
export KV_CACHE_DTYPE=${vllm_kv_cache_dtype}
export LMCACHE_VLLM_KV_CACHE_DTYPE=${vllm_kv_cache_dtype}
export LMCACHE_KV_CACHE_DTYPE=${lmcache_kv_cache_dtype}
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=${target_block_size}
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=${mamba_block_size}

case "${cache_mode}" in
  vram)
    export LMCACHE_ENABLED=0
    ;;
  native)
    export LMCACHE_ENABLED=0
    native_offload_size=${NATIVE_KV_OFFLOADING_SIZE_GB:-64}
    [[ ${native_offload_size} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
      fail "NATIVE_KV_OFFLOADING_SIZE_GB must be a positive number; got ${native_offload_size}"
    [[ ! ${native_offload_size} =~ ^0+([.]0+)?$ ]] ||
      fail "NATIVE_KV_OFFLOADING_SIZE_GB must be greater than zero"
    set -- "$@" \
      --kv-offloading-backend native \
      --kv-offloading-size "${native_offload_size}" \
      --enable-cumem-allocator
    ;;
  lmcache)
    export LMCACHE_ENABLED=1
    lmcache_transfer_mode=${LMCACHE_TRANSFER_MODE:-engine_driven}
    case "${lmcache_transfer_mode}" in
      engine_driven | lmcache_driven | auto) ;;
      *)
        fail "LMCACHE_TRANSFER_MODE must be engine_driven, lmcache_driven, or auto; got ${lmcache_transfer_mode}"
        ;;
    esac
    export LMCACHE_TRANSFER_MODE=${lmcache_transfer_mode}
    if [[ ${lmcache_transfer_mode} == engine_driven && \
      -z ${GPU_MEMORY_UTILIZATION+x} ]]; then
      export GPU_MEMORY_UTILIZATION=0.950
    fi
    lmcache_chunk_size=${LMCACHE_CHUNK_SIZE:-4096}
    target_token_budget=${LMCACHE_TARGET_TOKEN_BUDGET:-4096}
    require_positive_integer LMCACHE_CHUNK_SIZE "${lmcache_chunk_size}"
    require_positive_integer LMCACHE_TARGET_TOKEN_BUDGET "${target_token_budget}"
    if [[ ${lmcache_chunk_size} != "${target_token_budget}" ]]; then
      fail "LMCACHE_CHUNK_SIZE and LMCACHE_TARGET_TOKEN_BUDGET must match; got ${lmcache_chunk_size} and ${target_token_budget}"
    fi
    if [[ ${target_block_size} != auto ]] &&
      ((lmcache_chunk_size % (target_block_size * dcp) != 0)); then
      fail "LMCACHE_CHUNK_SIZE must be a multiple of GLM53_TARGET_BLOCK_SIZE times DCP; got chunk ${lmcache_chunk_size}, block ${target_block_size}, and DCP ${dcp}"
    fi
    export LMCACHE_CHUNK_SIZE=${lmcache_chunk_size}
    export LMCACHE_TARGET_TOKEN_BUDGET=${target_token_budget}

    case "${LMCACHE_L2_ENABLED:-1}" in
      0 | 1) ;;
      *) fail "LMCACHE_L2_ENABLED must be 0 or 1; got ${LMCACHE_L2_ENABLED}" ;;
    esac
    if [[ ${LMCACHE_L2_ENABLED:-1} == 1 ]]; then
      l2_root=${LMCACHE_L2_ROOT:-${LMCACHE_L2_PATH:-/lmcache-l2}}
      [[ ${l2_root} == /* && ${l2_root} != *\"* && \
        ${l2_root} != *\\* && ${l2_root} != *\'* ]] ||
        fail "LMCACHE_L2_ROOT must be an absolute path without quotes or backslashes"

      schema=$(sanitize_namespace_component "${LMCACHE_SCHEMA_REVISION:-glm53-r18-cache-v1}")
      model_id=$(sanitize_namespace_component "${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4}")
      model_revision=$(sanitize_namespace_component "${LMCACHE_MODEL_REVISION_ID:-${MODEL_REVISION:-huggingface-main}}")
      draft_revision=none
      if [[ ${speculator} == dflash2 ]]; then
        draft_revision=$(sanitize_namespace_component "${LMCACHE_DFLASH_REVISION_ID:-${DFLASH_MODEL_REVISION:-huggingface-main}}")
      fi
      namespace="${schema}/${model_id}-${model_revision}/kv-${kv_cache_quant}/tp-${tp}-dcp-${dcp}-i-${interleave}-g-${resolved_gather}/spec-${speculator}-${spec_depth}-draft-${draft_revision}/chunk-${lmcache_chunk_size}"
      export LMCACHE_L2_PATH="${l2_root%/}/${namespace}"
      export LMCACHE_INSTANCE_ID=${LMCACHE_INSTANCE_ID:-"glm53-${schema}-tp${tp}-dcp${dcp}-${kv_cache_quant}"}
    fi
    ;;
esac

if [[ ${CACHE_CONFIG_DRY_RUN:-0} == 1 ]]; then
  printf 'CACHE_MODE=%q\n' "${cache_mode}"
  printf 'KV_CACHE_QUANT=%q\n' "${kv_cache_quant}"
  printf 'TP=%q\nDCP=%q\nDCP_CKV_GATHER=%q\n' \
    "${tp}" "${dcp}" "${resolved_gather}"
  printf 'SPECULATOR=%q\nSPECULATIVE_DEPTH=%q\n' \
    "${speculator}" "${spec_depth}"
  printf 'VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=%q\n' \
    "${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE}"
  printf 'VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=%q\n' \
    "${VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE}"
  if [[ ${cache_mode} == lmcache ]]; then
    printf 'LMCACHE_TRANSFER_MODE=%q\n' "${LMCACHE_TRANSFER_MODE}"
    printf 'GPU_MEMORY_UTILIZATION=%q\n' \
      "${GPU_MEMORY_UTILIZATION:-launcher-default}"
    printf 'LMCACHE_L2_PATH=%q\n' "${LMCACHE_L2_PATH:-disabled}"
  fi
  printf 'ARGV:'
  if (($# > 0)); then
    printf ' %q' "$@"
  fi
  printf '\n'
  exit 0
fi

exec "${cache_launcher}" "$@"
