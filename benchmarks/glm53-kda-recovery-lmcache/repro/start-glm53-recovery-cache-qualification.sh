#!/usr/bin/env bash
# Run on the assigned host. Source overlays preserve the installed CUDA extensions.
set -euo pipefail

mode=${1:?Use tests, mtp2, mtp4 or dflash4}
stage=/root/glm53-recovery-lmcache-qualification
site=/opt/venv/lib/python3.12/site-packages
image=local/karmic-head-sync:cfdecfaa4739-f6d87176c1f5
mounts=(-v "$stage:/workspace" -v "$stage/b12x/b12x:$site/b12x:ro")
while IFS= read -r source; do
  mounts+=(-v "$source:$site/${source#"$stage/vllm/"}:ro")
done < <(rg --files "$stage/vllm/vllm" -g '*.py')

case "$mode" in
  tests) devices=5 ;;
  mtp2) devices=5,6 ;;
  mtp4|dflash4) devices=5,6,7,8 ;;
  *) exit 2 ;;
esac
while read -r used; do
  (( used <= 200 )) || { echo "Qualification GPU is occupied" >&2; exit 1; }
done < <(nvidia-smi --id="$devices" --query-gpu=memory.used --format=csv,noheader,nounits)

name=glm53-recovery-cache-$mode
volume=$name-data
recovery_layout=${RECOVERY_LAYOUT:-auto}
case "$recovery_layout" in
  auto) ;;
  full) name+=-full ;;
  *) echo 'RECOVERY_LAYOUT must be auto or full' >&2; exit 2 ;;
esac
common=(docker run -d --name "$name" --init --gpus "\"device=$devices\""
  --label lil.qualification=glm53-kda-recovery-lmcache
  --network host --ipc host --shm-size 32g --memory 192g --memory-swap 192g
  --ulimit memlock=-1 --ulimit stack=67108864:67108864
  --security-opt seccomp=unconfined "${mounts[@]}" -e OMP_NUM_THREADS=1)
if [[ "$mode" == tests ]]; then
  "${common[@]}" --entrypoint /bin/bash "$image" -lc 'sleep infinity'
  exit
fi

hub=/root/.cache/huggingface/hub
model=$hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/snapshots/520de24eabf507659eaef7c70f14fd584527facc
tp=4; dcp=1; batch=4096; seqs=16; captures=1,2,4,8,16,32,40,48,64,96,128
max_len=131072; gmu=0.93; extra=()
if [[ "$mode" == mtp2 ]]; then
  model=$hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4-Spark/snapshots/a608241037e4c2565356bff7ca293f2133888f88
  tp=2; dcp=2; batch=3072; seqs=4; captures=1,2,4,8,12,16
  max_len=-1; gmu=0.985
  extra=(-- --kv-cache-memory-bytes 4190109696)
fi
if [[ "$recovery_layout" == full ]]; then
  [[ ${#extra[@]} != 0 ]] || extra=(--)
  extra+=(--no-use-replayssm)
fi
spec=mtp
[[ "$mode" != dflash4 ]] || spec=dflash2
envs=(-e MODEL="$model" -e PROFILE=glm53-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie
  -e SERVED_MODEL_NAME=GLM-5.3-Flash -e PORT=5571 -e TP="$tp" -e DCP="$dcp"
  -e SPECULATOR="$spec"
  -e MAX_MODEL_LEN="$max_len" -e MAX_NUM_SEQS="$seqs" -e MAX_NUM_BATCHED_TOKENS="$batch"
  -e GPU_MEMORY_UTILIZATION="$gmu" -e CACHE_MODE="${CACHE_KIND:-lmcache}"
  -e LMCACHE_L2_ENABLED=1 -e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2
  -e LMCACHE_L2_GB=64 -e LMCACHE_MAX_CPU_WORKERS=4 -e LMCACHE_MAX_GPU_WORKERS=2
  -e LMCACHE_CHUNK_SIZE="$batch" -e VLLM_SERVER_DEV_MODE=1
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${captures##*,}" -e CUDAGRAPH_CAPTURE_SIZES="$captures"
  -e LOAD_FORMAT=safetensors -e KV_CACHE_DTYPE=fp8_ds_mla
  -e B12X_AUTOTUNE="${B12X_AUTOTUNE:-1}"
  -e ADDITIONAL_CONFIG='{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}'
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=off
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo -e NCCL_NET_PLUGIN=none
  -e NCCL_TUNER_PLUGIN=none)
if [[ "$mode" == dflash4 ]]; then
  envs+=(-e DFLASH_DEPTH=7)
else
  envs+=(-e MTP_DEPTH=3)
fi
if [[ "$mode" == mtp2 ]]; then
  envs+=(-e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2 -e NCCL_BUFFSIZE=1048576
    -e CUBLAS_WORKSPACE_CONFIG=:4096:1 -e VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=65536
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:12
    -e SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X"}')
fi
"${common[@]}" -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$volume:/cache" "${envs[@]}" "$image" "${extra[@]}"
