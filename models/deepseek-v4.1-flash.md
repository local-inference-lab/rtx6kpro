# DeepSeek-V4.1-Flash

Serve `deepseek-ai/DeepSeek-V4.1-Flash` using the `ds41-flash` profile in the
[shared Docker guide](../docs/unified-vllm-docker.md). It supports native
text/vision input, B12X kernels and the checkpoint's embedded DSpark draft.
It is distinct from [DeepSeek V4 text](deepseek-v4-flash.md) and
[V4 Vision](deepseek-v4-flash-vision.md).

## Start the server

This starts TP4/DCP1 with adaptive DSpark K7 and disk-backed Engram tables:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name ds41 --init --restart unless-stopped \
  --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -v lil-huggingface:/root/.cache/huggingface -v ds41-runtime:/cache \
  -e PROFILE=ds41-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=4 -e PORT=8000 "$IMAGE"
```

The API model is `DeepSeek-V4.1-Flash` on port 8000. The checkpoint downloads
into the shared HF volume. Check readiness with `docker logs -f ds41`.
The io_uring loader needs the syscall permission above; use a trusted image.

Add native options **after `"$IMAGE"`**:

| Choice | Arguments |
|---|---|
| Keep n-gram tables in RAM | `--engram-table-memory ram` |
| Read tables from SSD | `--engram-table-memory disk` (default) |
| No speculation | `--mode off` |
| Restrict context | `--max-model-len 131072` |
| Change request slots | `--max-num-seqs 16` |

The default is 32 slots and an automatically sized context. Change GPU IDs,
`TP` and `PORT` before the image name. MTP/DFlash2 are not DS4.1's DSpark mode.
See the [Karmic Kraken benchmark table](../benchmarks/karmic-kraken-serving.md)
for the comparison with RAM Engram; disk placement has separate functional
cache checks and is not the speed measured below.

<!-- BEGIN LIL-COMPOSE-REFERENCE -->
## Expand the complete default configurations

Each block contains a runnable release-tagged Compose file, all image/profile
ENV settings and the resolved vLLM command. Requires Docker Compose 2.23.1+.
These snapshots use GPU-only prefix caching; inactive cache options do not
start LMCache. They are deployment defaults, not benchmark-only overrides.

For ordinary TP, speculation or cache changes, use the short commands above
so dependent settings are recalculated. See the
[expanded-file editing guide](../docs/unified-vllm-docker.md#expand-the-complete-default-configurations)
before modifying a frozen configuration. No credentials are included.

<details>
<summary>DeepSeek V4.1: TP4, adaptive DSpark K7, disk Engram: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/ds41-flash-tp4.compose.yaml). Save it as `ds41-flash-tp4.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f ds41-flash-tp4.compose.yaml up -d
```

Logs: `docker compose -f ds41-flash-tp4.compose.yaml logs -f model`.
Stop: `docker compose -f ds41-flash-tp4.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: ds41-flash-tp4
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-4d9905b931656635
    container_name: ds41-flash-tp4
    init: true
    network_mode: host
    ipc: host
    shm_size: 32g
    restart: unless-stopped
    ulimits:
      memlock:
        soft: -1
        hard: -1
      stack:
        soft: 67108864
        hard: 67108864
    volumes:
    - lil-huggingface:/root/.cache/huggingface
    - ds41-flash-tp4-runtime:/cache
    entrypoint:
    - /opt/venv/bin/python
    - -m
    - runtime.explicit
    command:
    - --config
    - /etc/lil-launch.yaml
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            device_ids:
            - '0'
            - '1'
            - '2'
            - '3'
            capabilities:
            - gpu
    security_opt:
    - seccomp=unconfined
    environment:
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/b12x/cute
      BASH_ENV: /etc/bash.bashrc
      CCCL_VERSION: 13.3.4.2.1
      COCOAPI_VERSION: 2.0+nv0.8.1
      CUBLASMP_VERSION: 0.10.0.3695
      CUBLAS_VERSION: 13.7.0.27
      CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0
      CUDA_BINARY_LOADER_THREAD_COUNT: '8'
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/cuda
      CUDA_COMPONENT_LIST: crt nvrtc driver-dev culibos-dev cudart cudart-dev nvcc tileiras cupti
      CUDA_DEVICE_MAX_CONNECTIONS: '32'
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_DRIVER_VERSION: 615.65.02
      CUDA_HOME: /usr/local/cuda
      CUDA_MODULE_LOADING: LAZY
      CUDA_VERSION: 13.4.1.012
      CUDLA_VERSION: 13.4.49
      CUDNN_FRONTEND_VERSION: 1.27.0
      CUDNN_VERSION: 9.25.0.28
      CUFFT_VERSION: 12.4.0.34
      CUFILE_VERSION: 1.19.0.109
      CURAND_VERSION: 10.4.4.49
      CUSOLVERMP_VERSION: 0.9.0.6427
      CUSOLVER_VERSION: 12.3.2.15
      CUSPARSELT_VERSION: 0.9.1.1
      CUSPARSE_VERSION: 12.8.6.49
      CUTE_DSL_ARCH: sm_120a
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/cute-dsl
      CUTILE_PYTHON_VERSION: 1.5.0
      CUTLASS_DSL_VERSION: 4.6.2
      DALI_BUILD: ''
      DALI_URL_SUFFIX: '130'
      DALI_VERSION: 2.2.0
      DOCA_VERSION: 3.5.0
      EFA_VERSION: 1.48.0
      ENV: /etc/shinit_v2
      GDRCOPY_VERSION: 2.5.1
      HF_HOME: /root/.cache/huggingface
      HPCX_VERSION: '2.50'
      INSTANTTENSOR_BACKEND: BUFFERED
      JUPYTER_PORT: '8888'
      LC_ALL: C.UTF-8
      LD_LIBRARY_PATH: /usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
      LIBRARY_PATH: '/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64/stubs:'
      MAXSMVER: ''
      MODEL_OPT_VERSION: 0.45.0
      MOFED_VERSION: 5.4-rdmacore63.0
      NCCL_BUFFSIZE: '2097152'
      NCCL_IB_DISABLE: '1'
      NCCL_MAX_NCHANNELS: '16'
      NCCL_MIN_NCHANNELS: '16'
      NCCL_NET_PLUGIN: spcx
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_VERSION: 2.30.7+cuda13.3
      NIXL_VERSION: 1.3.0
      NPP_VERSION: 13.2.0.35
      NSIGHT_COMPUTE_VERSION: 2026.3.0.13
      NSIGHT_SYSTEMS_VERSION: 2026.5.1.18
      NVFATBIN_VERSION: 13.4.49
      NVFUSER_BUILD_VERSION: 0.1.4a0+nvidia
      NVFUSER_VERSION: ''
      NVIDIA_BUILD_ID: '406036884'
      NVIDIA_DRIVER_CAPABILITIES: compute,utility,video
      NVIDIA_PRODUCT_NAME: PyTorch
      NVIDIA_PYTORCH_VERSION: '26.08'
      NVIDIA_REQUIRE_CUDA: cuda>=9.0
      NVIDIA_VISIBLE_DEVICES: 0,1,2,3
      NVJITLINK_VERSION: 13.4.52
      NVJPEG_VERSION: 13.2.2.35
      NVPL_LAPACK_MATH_MODE: PEDANTIC
      NVPTXCOMPILER_VERSION: 13.4.59
      NVRX_VERSION: 0.6.0
      NVSHMEM_VERSION: 3.7.1
      NVVM_VERSION: 13.4.59
      OMPI_MCA_coll_hcoll_enable: '0'
      OMP_NUM_THREADS: '8'
      OPAL_PREFIX: /usr/local/mpi
      OPENMPI_VERSION: 5.0.10
      OPENUCX_VERSION: 1.21.0
      PATH: /opt/venv/bin:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin
      PIP_BREAK_SYSTEM_PACKAGES: '1'
      PIP_CONSTRAINT: /etc/pip/constraint.txt
      PIP_DEFAULT_TIMEOUT: '100'
      POLYGRAPHY_VERSION: 0.53.3
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: python
      PYTHONIOENCODING: utf-8
      PYTORCH_BUILD_NUMBER: '0'
      PYTORCH_BUILD_VERSION: 2.14.0a0+4fdf77b
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      PYTORCH_HOME: /opt/pytorch/pytorch
      PYTORCH_VERSION: 2.14.0a0+4fdf77b
      RDMACORE_VERSION: '63.0'
      SAFETENSORS_FAST_GPU: '1'
      SHELL: /bin/bash
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/triton
      TRITON_CUDACRT_PATH: /usr/local/cuda/include
      TRITON_CUDART_PATH: /usr/local/cuda/include
      TRITON_CUOBJDUMP_PATH: /usr/local/cuda/bin/cuobjdump
      TRITON_CUPTI_INCLUDE_PATH: /usr/local/cuda/include
      TRITON_CUPTI_LIB_PATH: /usr/local/cuda/lib64
      TRITON_NVDISASM_PATH: /usr/local/cuda/bin/nvdisasm
      TRITON_PTXAS_PATH: /usr/local/cuda/bin/ptxas
      TRTOSS_VERSION: ''
      TRT_VERSION: 11.2.1.2+cuda13.3
      UCC_CL_BASIC_TLS: ^sharp
      UCC_EC_CUDA_EXEC_NUM_THREADS: '256'
      VIRTUAL_ENV: /opt/venv
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526/vllm
      VLLM_ENABLE_PCIE_ALLREDUCE: '1'
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '1'
      VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: '1024'
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_DMA_MIN_BYTES: 6MB
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 96KB
      VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE: 96KB
      VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE: 'off'
      VLLM_USE_AOT_COMPILE: '1'
      VLLM_USE_BREAKABLE_CUDAGRAPH: '0'
      VLLM_USE_FLASHINFER_SAMPLER: '1'
      VLLM_USE_MEGA_AOT_ARTIFACT: '1'
      VLLM_USE_STANDALONE_COMPILE: '1'
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/ds41-flash-a1764596b72ae526
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  ds41-flash-tp4-runtime:
    name: ds41-flash-tp4-runtime
  lil-huggingface:
    name: lil-huggingface
configs:
  lil-launch:
    content: |
      schema: lil-explicit-launch/v1
      profile: ds41-flash
      hardware: rtx-pro-6000-pcie
      options:
        cache-transfer-mode: engine_driven
        cache-l1-gib: 24.0
        cache-l1-init-gib: 2
        cache-l2-gib: 256.0
        cache-l2-enabled: false
        cache-cpu-workers: 4
        cache-l2-workers: 4
        cache-directory: /cache/lmcache
        cache-object-tokens: 4096
        cache-native-gib: 64.0
        cache-host: 127.0.0.1
        cache-http-host: 127.0.0.1
        cache-start-timeout: 120.0
        cache-prefetch-policy: retain
        cache-broker-directory: /cache/lmcache-cumem
        cache-load-failure-policy: recompute
        host: 0.0.0.0
        port: 8000
        pipeline-parallel-size: 1
        decode-context-parallel-size: 1
        dtype: bfloat16
        kv-cache-dtype: fp8
        load-format: instanttensor
        enable-prefix-caching: true
        enable-prompt-tokens-details: true
        enable-chunked-prefill: true
        enable-auto-tool-choice: true
        cache-mode: vram
        model: deepseek-ai/DeepSeek-V4.1-Flash
        served-model-name: DeepSeek-V4.1-Flash
        tensor-parallel-size: 4
        mode: dspark
        prefill-compute-share: '0.4'
        prefill-schedule-interval: 1
        max-parallel-prefills: 1
        max-model-len: -1
        max-num-seqs: 32
        max-num-batched-tokens: 4096
        gpu-memory-utilization: 0.95
        block-size: 256
        swa-block-size: 128
        max-cudagraph-capture-size: 128
        compilation-config:
          cudagraph_mode: FULL_AND_PIECEWISE
        attention-backend: B12X
        moe-backend: b12x
        linear-backend: b12x
        async-scheduling: true
        scheduler-reserve-full-isl: false
        prefill-policy: round-robin
        decode-refill-target: auto
        generation-config: vllm
        override-generation-config:
          temperature: 1.0
          top_p: 0.95
        engram-table-memory: disk
        engram-disk-resident-scales: false
        engram-projection-tp: false
        adaptive-verification: true
        adaptive-verification-cost-scale: 1.0
        safetensors-load-strategy: lazy
        jit-monitor-mode: warn
        tokenizer-mode: deepseek_v41
        reasoning-parser: deepseek_v41
        tool-call-parser: deepseek_v41
        enable-force-include-usage: true
        enable-request-id-headers: true
        default-chat-template-kwargs:
          thinking: true
          reasoning_effort: high
        speculative-config:
          method: dspark
          attention_backend: B12X
          draft_sample_method: greedy
          rejection_sample_method: standard
          num_speculative_tokens: 7
          draft_tensor_parallel_size: 4
          enable_adaptive_verification: true
          adaptive_verification_cost_scale: 1.0
        draft-tokens: 7
        engram-config:
          cpu_offload: false
          table_memory: disk
          disk_resident_scales: false
          projection_tp: false
      passthrough: []
      vllm_defaults:
      - additional-config
      - code-revision
      - cp-kv-cache-interleave-size
      - cudagraph-capture-sizes
      - dcp-kv-cache-interleave-size
      - disable-custom-all-reduce
      - enable-flashinfer-autotune
      - gdn-decode-kernel
      - hf-overrides
      - kv-cache-memory-bytes
      - language-model-only
      - mamba-cache-mode
      - mamba-ssm-cache-dtype
      - mm-encoder-tp-mode
      - mm-processor-cache-gb
      - prefill-compute-half-life
      - prefix-cache-retention-interval
      - prefix-match-unit
      - quantization
      - recurrent-checkpoint-policy
      - revision
      - trust-remote-code
      runtime_bindings:
        UNBOUND-RUNTIME: Runtime lock from the selected immutable image
        checkpoint_identity: Verified target/draft revisions before opening external storage
      environment_keys:
      - B12X_COMPILE_CACHE_DIR
      - B12X_CUTE_COMPILE_CACHE_DIR
      - BASH_ENV
      - CCCL_VERSION
      - COCOAPI_VERSION
      - CUBLASMP_VERSION
      - CUBLAS_VERSION
      - CUDA_ARCH_LIST
      - CUDA_BINARY_LOADER_THREAD_COUNT
      - CUDA_CACHE_PATH
      - CUDA_COMPONENT_LIST
      - CUDA_DEVICE_MAX_CONNECTIONS
      - CUDA_DEVICE_ORDER
      - CUDA_DRIVER_VERSION
      - CUDA_HOME
      - CUDA_MODULE_LOADING
      - CUDA_VERSION
      - CUDLA_VERSION
      - CUDNN_FRONTEND_VERSION
      - CUDNN_VERSION
      - CUFFT_VERSION
      - CUFILE_VERSION
      - CURAND_VERSION
      - CUSOLVERMP_VERSION
      - CUSOLVER_VERSION
      - CUSPARSELT_VERSION
      - CUSPARSE_VERSION
      - CUTE_DSL_ARCH
      - CUTE_DSL_CACHE_DIR
      - CUTILE_PYTHON_VERSION
      - CUTLASS_DSL_VERSION
      - DALI_BUILD
      - DALI_URL_SUFFIX
      - DALI_VERSION
      - DOCA_VERSION
      - EFA_VERSION
      - ENV
      - GDRCOPY_VERSION
      - HF_HOME
      - HPCX_VERSION
      - INSTANTTENSOR_BACKEND
      - JUPYTER_PORT
      - LC_ALL
      - LD_LIBRARY_PATH
      - LIBRARY_PATH
      - MAXSMVER
      - MODEL_OPT_VERSION
      - MOFED_VERSION
      - NCCL_BUFFSIZE
      - NCCL_IB_DISABLE
      - NCCL_MAX_NCHANNELS
      - NCCL_MIN_NCHANNELS
      - NCCL_NET_PLUGIN
      - NCCL_P2P_LEVEL
      - NCCL_PROTO
      - NCCL_VERSION
      - NIXL_VERSION
      - NPP_VERSION
      - NSIGHT_COMPUTE_VERSION
      - NSIGHT_SYSTEMS_VERSION
      - NVFATBIN_VERSION
      - NVFUSER_BUILD_VERSION
      - NVFUSER_VERSION
      - NVIDIA_BUILD_ID
      - NVIDIA_DRIVER_CAPABILITIES
      - NVIDIA_PRODUCT_NAME
      - NVIDIA_PYTORCH_VERSION
      - NVIDIA_REQUIRE_CUDA
      - NVIDIA_VISIBLE_DEVICES
      - NVJITLINK_VERSION
      - NVJPEG_VERSION
      - NVPL_LAPACK_MATH_MODE
      - NVPTXCOMPILER_VERSION
      - NVRX_VERSION
      - NVSHMEM_VERSION
      - NVVM_VERSION
      - OMPI_MCA_coll_hcoll_enable
      - OMP_NUM_THREADS
      - OPAL_PREFIX
      - OPENMPI_VERSION
      - OPENUCX_VERSION
      - PATH
      - PIP_BREAK_SYSTEM_PACKAGES
      - PIP_CONSTRAINT
      - PIP_DEFAULT_TIMEOUT
      - POLYGRAPHY_VERSION
      - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
      - PYTHONIOENCODING
      - PYTORCH_BUILD_NUMBER
      - PYTORCH_BUILD_VERSION
      - PYTORCH_CUDA_ALLOC_CONF
      - PYTORCH_HOME
      - PYTORCH_VERSION
      - RDMACORE_VERSION
      - SAFETENSORS_FAST_GPU
      - SHELL
      - SPARKINFER_COMPILE_CACHE_DIR
      - TENSORBOARD_PORT
      - TORCHAO_BUILD_VERSION
      - TORCHINDUCTOR_CACHE_DIR
      - TORCHINDUCTOR_CUTLASS_DIR
      - TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION
      - TORCHTITAN_BUILD_VERSION
      - TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
      - TORCH_CUDA_ARCH_LIST
      - TORCH_NCCL_USE_COMM_NONBLOCKING
      - TRANSFORMER_ENGINE_VERSION
      - TRITON_CACHE_DIR
      - TRITON_CUDACRT_PATH
      - TRITON_CUDART_PATH
      - TRITON_CUOBJDUMP_PATH
      - TRITON_CUPTI_INCLUDE_PATH
      - TRITON_CUPTI_LIB_PATH
      - TRITON_NVDISASM_PATH
      - TRITON_PTXAS_PATH
      - TRTOSS_VERSION
      - TRT_VERSION
      - UCC_CL_BASIC_TLS
      - UCC_EC_CUDA_EXEC_NUM_THREADS
      - VIRTUAL_ENV
      - VLLM_CACHE_DIR
      - VLLM_CACHE_ROOT
      - VLLM_ENABLE_PCIE_ALLREDUCE
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
      - VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
      - VLLM_PCIE_ALLREDUCE_BACKEND
      - VLLM_PCIE_DMA_MIN_BYTES
      - VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE
      - VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE
      - VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE
      - VLLM_USE_AOT_COMPILE
      - VLLM_USE_BREAKABLE_CUDAGRAPH
      - VLLM_USE_FLASHINFER_SAMPLER
      - VLLM_USE_MEGA_AOT_ARTIFACT
      - VLLM_USE_STANDALONE_COMPILE
      - VLLM_USE_V2_MODEL_RUNNER
      - VLLM_WORKER_MULTIPROC_METHOD
      - XDG_CACHE_HOME
      - _CUDA_COMPAT_PATH
```

The explicit runner passes these native arguments to vLLM through the image's
CUDA/NCCL bootstrap. This command is shown for inspection; the Compose file
above also supplies its environment and persistent volumes.

```bash
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve deepseek-ai/DeepSeek-V4.1-Flash \
  --async-scheduling \
  --attention-backend B12X \
  --block-size 256 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --decode-context-parallel-size 1 \
  --decode-refill-target auto \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-force-include-usage \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-request-id-headers \
  --engram-config '{"cpu_offload":false,"table_memory":"disk","disk_resident_scales":false,"projection_tp":false}' \
  --generation-config vllm \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 \
  --jit-monitor-mode warn \
  --kv-cache-dtype fp8 \
  --linear-backend b12x \
  --load-format instanttensor \
  --max-cudagraph-capture-size 128 \
  --max-model-len -1 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --max-parallel-prefills 1 \
  --moe-backend b12x \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --pipeline-parallel-size 1 \
  --port 8000 \
  --prefill-compute-share 0.4 \
  --prefill-policy round-robin \
  --prefill-schedule-interval 1 \
  --reasoning-parser deepseek_v41 \
  --safetensors-load-strategy lazy \
  --no-scheduler-reserve-full-isl \
  --served-model-name DeepSeek-V4.1-Flash \
  --speculative-config '{"method":"dspark","attention_backend":"B12X","draft_sample_method":"greedy","rejection_sample_method":"standard","num_speculative_tokens":7,"draft_tensor_parallel_size":4,"enable_adaptive_verification":true,"adaptive_verification_cost_scale":1.0}' \
  --swa-block-size 128 \
  --tensor-parallel-size 4 \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41
```

</details>
<!-- END LIL-COMPOSE-REFERENCE -->

## RAM versus SSD ngrams

Engram contains learned n-gram model tables. It is **not** KV cache, prefix
caching or an LMCache tier. Target/draft transformer weights remain on GPU.

| Placement | Behavior | Capacity requirement |
|---|---|---|
| `disk` | Native io_uring reads selected checkpoint rows from local SSD into bounded staging buffers | Fast local SSD/NVMe; loader, OS cache and staging still need RAM |
| `ram` | Complete tables in pinned, GPU-mapped host RAM | Historical TP4 checkpoint accounting: 188.83 GiB for tables, plus loader/server/OS memory |

RAM allocation failure does not silently choose disk. Generic CPU model
offload remains disabled. Optional `--engram-disk-resident-scales` and
`--engram-projection-tp` are not enabled by default and have no whole-model
speed claim in this qualification. `CACHE_MODE=lmcache` independently enables
CPU/disk prefix storage; selecting RAM Engram does not enable it.

## Common settings

Keep TP4/DCP1 and adaptive DSpark K7 for this recipe. Additional capacity
settings go **before `"$IMAGE"`**:

| Setting | Default / recommendation | Example override |
|---|---|---|
| Active requests | 32 | `-e MAX_NUM_SEQS=16` |
| Prefill budget | 4096 tokens | `-e MAX_NUM_BATCHED_TOKENS=4096` |
| Context | Automatic, `-1` | `-e MAX_MODEL_LEN=131072` |
| GPU memory fraction | 0.95 | `-e GPU_MEMORY_UTILIZATION=0.93` for more working space |
| JIT monitor | `warn` for serving | `-e JIT_MONITOR_MODE=error` only for strict warmup diagnostics |
| External prefix storage | GPU-only | `-e CACHE_MODE=lmcache` with [RAM/disk settings](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload) |

B12X handles attention, MoE and dense kernels. Full-and-piecewise decode
graphs are enabled; breakable prefill is off. Prefix caching is on even though
the default retention interval is `0`. Keep the model's own cache geometry.

Draft proposals are greedy with standard rejection; target requests still
sample at temperature 1/top-p .95. Default reasoning is `high`, with the
checkpoint's `low=50`, `high=75`, `max=100` budget mapping.

The native cache is heterogeneous despite the CLI's `fp8` label: MXFP8
sliding-window payloads, NVFP4 indexed payloads and index/state groups. Do not
describe it as a uniform FP8 cache. The GLM/Qwen request-boundary recurrent
adapter does not apply to DeepSeek's attention-cache structure.

Use `--adaptive-verification=false` to disable DSpark trimming, not DSpark
itself. `--adaptive-verification-cost-scale` changes trimming cost; neither
alternative is timed here. A request can override the reasoning budget with
`"chat_template_kwargs":{"reasoning_effort":50}`.

## Measured performance

Four RTX PRO 6000 **Max-Q**, **VRAM +6000**, automatic graphics clocks;
TP4/DCP1, RAM Engram, adaptive DSpark K7, 4096-token budget, 32 slots,
131,072 context cap, GPU-only cache and temperature 1/top-p .95.
Five warmed 30-second windows per cell; context-zero decode and uncached
nominal-32K prefill measured from client TTFT. C8 is aggregate.

| Metric | Saved JJ R38 | Karmic Kraken | Change |
|---|---:|---:|---:|
| C1 output | 230.6 tok/s | 249.1 tok/s | +8.01% |
| C8 aggregate output | 748.3 tok/s | 805.5 tok/s | +7.65% |
| 32K prefill | 17,503 tok/s | 17,982 tok/s | +2.74% |
| C1 verifier rate | 89.87 steps/s | 98.58 steps/s | +9.69% |

Text, image and repeated/changed-prefix checks pass. The KK server reports
4,727,748 logical KV tokens shared by requests, not a per-request context
limit. These are RAM-Engram results; selecting disk can change throughput.
[Image versions, configuration and all samples](../benchmarks/karmic-kraken-serving.md).

### Breakable prefill

`-e VLLM_USE_BREAKABLE_CUDAGRAPH=1` before the image captures graph segments around dynamic
attention/cache operations, which still execute outside those segments. It is
not a single FULL graph for all prefill work. Decode graphs remain available
when this option is off.

It remains off by default: additional graph memory can reduce KV capacity.
The [separate graph-memory comparison](../benchmarks/prepared-b12x-contracts/#breakable-prefill-ds41)
records that trade-off on its own image; it is not the throughput table above.

## Historical releases and source review

- [Archived recipes and measurements](../archive/serving-guides/README.md)
  preserve the preceding guides, API checks and stock Workstation tables.
- [Community R38 deployment and measurement archive](deepseek-v4.1-flash-community-r38.md):
  R37/R38 measurements, Sieve, source/dependency locks and exact checkpoint scope.
- [R37 record](deepseek-v4.1-flash/r37/release.md) and
  [R36 record](deepseek-v4.1-flash/r36/release.md), including separately labelled clocks.
- [Shared runtime dependency corrections](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b3fe0afe1621273059fb19dee1034e2272043a55/runtime/DEPENDENCIES.md):
  PyTorch/CuTe corrections belong to image assembly, not model startup patches.
- [Issue #808](https://github.com/local-inference-lab/vllm/issues/808):
  open source PRs, integration status and qualification limits.
