#!/usr/bin/env bash
set -euo pipefail
export CUTE_DSL_ARCH=sm_120a
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_SPEC_V2=True
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
python3 -m sglang.launch_server   --model-path lukealonso/GLM-5.1-NVFP4   --served-model-name GLM-5   --reasoning-parser glm45   --tool-call-parser glm47   --tensor-parallel-size 8   --quantization modelopt_fp4   --kv-cache-dtype fp8_e4m3   --trust-remote-code   --disable-shared-experts-fusion   --nsa-prefill-backend b12x   --nsa-decode-backend b12x   --page-size 64   --attention-backend nsa   --moe-runner-backend b12x   --fp4-gemm-backend b12x   --cuda-graph-max-bs 4   --enable-pcie-oneshot-allreduce   --speculative-algorithm EAGLE   --speculative-num-steps 3   --speculative-num-draft-tokens 4   --speculative-eagle-topk 1   --chunked-prefill-size 8192   --max-running-requests 4   --mem-fraction-static 0.76   --host 0.0.0.0   --port 8001   --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'   --json-model-override-args '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'   --preferred-sampling-params '{"temperature": 1.0, "top_p": 0.95}'   --enable-metrics
