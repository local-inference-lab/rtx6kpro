# GLM-5.3-Flash on two 96-GB GPUs

Run `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` with the shared Karmic
Kraken beta image. The Spark preset supplies TP2/DCP2, MTP3, B12X backends
and memory settings for two RTX PRO 6000 cards. “Spark” identifies this
checkpoint, not ARM-based DGX Spark hardware.

## Start the server

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name glm-spark-tp2 --init --restart unless-stopped \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  --security-opt seccomp=unconfined \
  -v lil-huggingface:/root/.cache/huggingface \
  -v glm-spark-runtime:/cache \
  -e PRESET=glm53-spark-tp2 -e PORT=8000 \
  "$IMAGE"
```

Change `device=0,1` to your two GPU IDs. The checkpoint downloads into the
Hugging Face volume; no host model path is required. To use a saved HF token,
mount your existing HF cache or pass `--env HF_TOKEN` after exporting it.
The image does not change clocks. This CUDA 13.4.1 image needs a compatible
NVIDIA driver; native driver 615.71.09 is tested.

The API model name is `GLM-5.3-Flash` on port 8000:

```bash
docker logs -f glm-spark-tp2
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Keep the API on a trusted network or add authentication. First startup includes
model loading, kernel preparation and graph capture and can take several minutes.

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
<summary>GLM Spark: TP2/DCP2, MTP3: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/glm53-spark-tp2.compose.yaml). Save it as `glm53-spark-tp2.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f glm53-spark-tp2.compose.yaml up -d
```

Logs: `docker compose -f glm53-spark-tp2.compose.yaml logs -f model`.
Stop: `docker compose -f glm53-spark-tp2.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: glm53-spark-tp2
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-4d9905b931656635
    container_name: glm53-spark-tp2
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
    - glm53-spark-tp2-runtime:/cache
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
            capabilities:
            - gpu
    environment:
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/b12x/cute
      B12X_DYNAMIC_DIRECT_EXPERT_SCALES: '1'
      B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET: '1'
      B12X_DYNAMIC_SPLIT_COMPUTE_MAC: '224'
      B12X_DYNAMIC_SPLIT_FAST_PREPARE: '1'
      B12X_DYNAMIC_SPLIT_LOW_SMEM: '1'
      B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE: '1'
      B12X_DYNAMIC_WORK_SOURCE: persistent_grid
      B12X_MHC_PDL: '1'
      B12X_PCIE_ONESHOT_BLOCK_LIMIT: '4'
      B12X_PCIE_ONESHOT_PDL: '1'
      B12X_PCIE_ONESHOT_THREADS: '512'
      BASH_ENV: /etc/bash.bashrc
      CCCL_VERSION: 13.3.4.2.1
      COCOAPI_VERSION: 2.0+nv0.8.1
      CUBLASMP_VERSION: 0.10.0.3695
      CUBLAS_VERSION: 13.7.0.27
      CUBLAS_WORKSPACE_CONFIG: :4096:1
      CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0
      CUDA_BINARY_LOADER_THREAD_COUNT: '8'
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/cuda
      CUDA_COMPONENT_LIST: crt nvrtc driver-dev culibos-dev cudart cudart-dev nvcc tileiras cupti
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/cute-dsl
      CUTILE_PYTHON_VERSION: 1.5.0
      CUTLASS_DSL_VERSION: 4.6.2
      DALI_BUILD: ''
      DALI_URL_SUFFIX: '130'
      DALI_VERSION: 2.2.0
      DOCA_VERSION: 3.5.0
      EFA_VERSION: 1.48.0
      ENV: /etc/shinit_v2
      GDRCOPY_VERSION: 2.5.1
      GLOO_SOCKET_IFNAME: lo
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
      NCCL_BUFFSIZE: '1048576'
      NCCL_IB_DISABLE: '1'
      NCCL_MAX_NCHANNELS: '2'
      NCCL_MIN_NCHANNELS: '2'
      NCCL_NET_PLUGIN: none
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      NCCL_SOCKET_IFNAME: lo
      NCCL_TUNER_PLUGIN: none
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
      NVIDIA_VISIBLE_DEVICES: 0,1
      NVJITLINK_VERSION: 13.4.52
      NVJPEG_VERSION: 13.2.2.35
      NVPL_LAPACK_MATH_MODE: PEDANTIC
      NVPTXCOMPILER_VERSION: 13.4.59
      NVRX_VERSION: 0.6.0
      NVSHMEM_VERSION: 3.7.1
      NVVM_VERSION: 13.4.59
      OMPI_MCA_coll_hcoll_enable: '0'
      OMP_NUM_THREADS: '1'
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
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True,large_segment_size_mb:12
      PYTORCH_HOME: /opt/pytorch/pytorch
      PYTORCH_VERSION: 2.14.0a0+4fdf77b
      RDMACORE_VERSION: '63.0'
      SAFETENSORS_FAST_GPU: '1'
      SHELL: /bin/bash
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/triton
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
      VLLM_B12X_MLA_CKV_GATHER: '1'
      VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS: '65536'
      VLLM_B12X_MOE_FP4_FORCE_A16: '0'
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac/vllm
      VLLM_CAUSAL_CONV1D_UPDATE_HOIST: '1'
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      VLLM_DISABLE_SHARED_EXPERTS_STREAM: '0'
      VLLM_ENABLE_PCIE_ALLREDUCE: '1'
      VLLM_GLM53_DFLASH_ATTN: '1'
      VLLM_GLM53_KDA_GATE_SIDE_STREAM: '1'
      VLLM_GLM53_L2_PREFETCH: '1'
      VLLM_GLM53_L2_PREFETCH_PERSIST_MB: '0'
      VLLM_GLM53_MTP_DRAFT_HEAD: nvfp4
      VLLM_GLM53_ONLINE_DENSE_MXFP8: '0'
      VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE: auto
      VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE: '2048'
      VLLM_LM_HEAD_A16: '1'
      VLLM_MTP_NVFP4_LM_HEAD: '0'
      VLLM_MXFP8_LM_HEAD: '0'
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_DMA_MIN_BYTES: 'off'
      VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE: 'off'
      VLLM_PLUGINS: ''
      VLLM_USE_FLASHINFER_SAMPLER: '1'
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/glm53-flash-f997c73b160489ac
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  glm53-spark-tp2-runtime:
    name: glm53-spark-tp2-runtime
  lil-huggingface:
    name: lil-huggingface
configs:
  lil-launch:
    content: |
      schema: lil-explicit-launch/v1
      profile: glm53-flash
      hardware: rtx-pro-6000-pcie
      options:
        cache-transfer-mode: engine_driven
        cache-l1-gib: 64.0
        cache-l1-init-gib: 2
        cache-l2-gib: 512.0
        cache-l2-enabled: true
        cache-cpu-workers: 16
        cache-l2-workers: 8
        cache-directory: /cache/lmcache
        cache-object-tokens: 3072
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
        decode-context-parallel-size: 2
        dtype: bfloat16
        kv-cache-dtype: fp8
        load-format: safetensors
        enable-prefix-caching: true
        enable-prompt-tokens-details: true
        enable-chunked-prefill: true
        enable-auto-tool-choice: true
        cache-mode: vram
        cache-gpu-workers: 8
        model: local-inference-lab/GLM-5.3-Flash-NVFP4-Spark
        served-model-name: GLM-5.3-Flash
        tensor-parallel-size: 2
        mode: mtp
        prefill-compute-share: '0.4'
        prefill-schedule-interval: 1
        max-parallel-prefills: 1
        max-model-len: -1
        max-num-seqs: 4
        max-num-batched-tokens: 3072
        gpu-memory-utilization: 0.985
        block-size: 256
        target-page-size: '2048'
        recurrent-page-size: auto
        recurrent-checkpoint-policy: request_boundaries
        mamba-cache-mode: align
        cp-kv-cache-interleave-size: 4
        dcp-kv-cache-interleave-size: 4
        dcp-ckv-gather: auto
        attention-backend: B12X
        moe-backend: b12x
        linear-backend: b12x
        quantization: modelopt_mixed
        enable-flashinfer-autotune: false
        max-cudagraph-capture-size: 16
        cudagraph-capture-sizes:
        - 1
        - 2
        - 4
        - 8
        - 12
        - 16
        compilation-config:
          cudagraph_mode: FULL_AND_PIECEWISE
        additional-config:
          glm53_kda_decode_backend: auto
          kda_prefill_backend: b12x
        reasoning-parser: glm45
        tool-call-parser: glm47
        default-chat-template-kwargs:
          reasoning_effort: high
          clear_thinking: false
        override-generation-config:
          temperature: 1.0
          top_p: 0.95
        kv-cache-memory-bytes: 4190109696
        speculative-config:
          method: mtp
          draft_sample_method: probabilistic
          rejection_sample_method: standard
          moe_backend: b12x
          attention_backend: B12X
          num_speculative_tokens: 3
        draft-tokens: 3
      passthrough: []
      vllm_defaults:
      - async-scheduling
      - code-revision
      - decode-refill-target
      - disable-custom-all-reduce
      - enable-force-include-usage
      - enable-request-id-headers
      - engram-config
      - gdn-decode-kernel
      - generation-config
      - hf-overrides
      - jit-monitor-mode
      - language-model-only
      - mamba-ssm-cache-dtype
      - mm-encoder-tp-mode
      - mm-processor-cache-gb
      - prefill-compute-half-life
      - prefill-policy
      - prefix-cache-retention-interval
      - prefix-match-unit
      - revision
      - safetensors-load-strategy
      - scheduler-reserve-full-isl
      - swa-block-size
      - tokenizer-mode
      - trust-remote-code
      runtime_bindings:
        UNBOUND-RUNTIME: Runtime lock from the selected immutable image
        checkpoint_identity: Verified target/draft revisions before opening external storage
      environment_keys:
      - B12X_COMPILE_CACHE_DIR
      - B12X_CUTE_COMPILE_CACHE_DIR
      - B12X_DYNAMIC_DIRECT_EXPERT_SCALES
      - B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET
      - B12X_DYNAMIC_SPLIT_COMPUTE_MAC
      - B12X_DYNAMIC_SPLIT_FAST_PREPARE
      - B12X_DYNAMIC_SPLIT_LOW_SMEM
      - B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE
      - B12X_DYNAMIC_WORK_SOURCE
      - B12X_MHC_PDL
      - B12X_PCIE_ONESHOT_BLOCK_LIMIT
      - B12X_PCIE_ONESHOT_PDL
      - B12X_PCIE_ONESHOT_THREADS
      - BASH_ENV
      - CCCL_VERSION
      - COCOAPI_VERSION
      - CUBLASMP_VERSION
      - CUBLAS_VERSION
      - CUBLAS_WORKSPACE_CONFIG
      - CUDA_ARCH_LIST
      - CUDA_BINARY_LOADER_THREAD_COUNT
      - CUDA_CACHE_PATH
      - CUDA_COMPONENT_LIST
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
      - GLOO_SOCKET_IFNAME
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
      - NCCL_SOCKET_IFNAME
      - NCCL_TUNER_PLUGIN
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
      - VLLM_B12X_MLA_CKV_GATHER
      - VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS
      - VLLM_B12X_MOE_FP4_FORCE_A16
      - VLLM_CACHE_DIR
      - VLLM_CACHE_ROOT
      - VLLM_CAUSAL_CONV1D_UPDATE_HOIST
      - VLLM_DISABLED_KERNELS
      - VLLM_DISABLE_SHARED_EXPERTS_STREAM
      - VLLM_ENABLE_PCIE_ALLREDUCE
      - VLLM_GLM53_DFLASH_ATTN
      - VLLM_GLM53_KDA_GATE_SIDE_STREAM
      - VLLM_GLM53_L2_PREFETCH
      - VLLM_GLM53_L2_PREFETCH_PERSIST_MB
      - VLLM_GLM53_MTP_DRAFT_HEAD
      - VLLM_GLM53_ONLINE_DENSE_MXFP8
      - VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE
      - VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE
      - VLLM_LM_HEAD_A16
      - VLLM_MTP_NVFP4_LM_HEAD
      - VLLM_MXFP8_LM_HEAD
      - VLLM_PCIE_ALLREDUCE_BACKEND
      - VLLM_PCIE_DMA_MIN_BYTES
      - VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE
      - VLLM_PLUGINS
      - VLLM_USE_FLASHINFER_SAMPLER
      - VLLM_USE_V2_MODEL_RUNNER
      - VLLM_WORKER_MULTIPROC_METHOD
      - XDG_CACHE_HOME
      - _CUDA_COMPAT_PATH
```

The explicit runner passes these native arguments to vLLM through the image's
CUDA/NCCL bootstrap. This command is shown for inspection; the Compose file
above also supplies its environment and persistent volumes.

```bash
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve local-inference-lab/GLM-5.3-Flash-NVFP4-Spark \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --attention-backend B12X \
  --block-size 256 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --cp-kv-cache-interleave-size 4 \
  --cudagraph-capture-sizes 1 2 4 8 12 16 \
  --dcp-kv-cache-interleave-size 4 \
  --decode-context-parallel-size 2 \
  --default-chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":false}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.985 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 4190109696 \
  --linear-backend b12x \
  --load-format safetensors \
  --mamba-cache-mode align \
  --max-cudagraph-capture-size 16 \
  --max-model-len -1 \
  --max-num-batched-tokens 3072 \
  --max-num-seqs 4 \
  --max-parallel-prefills 1 \
  --moe-backend b12x \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --pipeline-parallel-size 1 \
  --port 8000 \
  --prefill-compute-share 0.4 \
  --prefill-schedule-interval 1 \
  --quantization modelopt_mixed \
  --reasoning-parser glm45 \
  --recurrent-checkpoint-policy request_boundaries \
  --served-model-name GLM-5.3-Flash \
  --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X","num_speculative_tokens":3}' \
  --tensor-parallel-size 2 \
  --tool-call-parser glm47
```

</details>
<!-- END LIL-COMPOSE-REFERENCE -->

## RAM and disk prefix cache

GPU-only prefix caching is enabled by default. For LMCache, add these arguments
**before `"$IMAGE"`** in the command above:

```bash
-e CACHE_MODE=lmcache \
-e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2 \
-e LMCACHE_L2_ENABLED=1 -e LMCACHE_L2_GB=64 \
-e LMCACHE_MAX_CPU_WORKERS=4 -e LMCACHE_MAX_GPU_WORKERS=2
```

This starts the CPU cache service automatically: up to 16 GiB RAM plus a
64-GiB disk tier in the persistent `/cache` volume. To use only RAM, also add
`-e LMCACHE_L2_ENABLED=0`. LMCache reserves `PORT + 10000`, `PORT + 10001`
and `PORT + 10002`. Use API ports at least three apart and separate runtime volumes
for independent cache instances.

Text prefixes restore from RAM and from disk after both services restart.
Image-bearing requests run normally, but their recurrent checkpoints
are not restored through external cache. CPU and disk caches reuse prior
requests; they do not enlarge the active context or GPU KV allocation.

For an edit near the end of a long user message, the server also keeps a
checkpoint before the final prefill chunk. It can restore that unchanged
prefix and recompute roughly one to two chunks. An edit before that checkpoint
still needs an earlier matching prefix or a full recomputation. This costs one
additional checkpoint bundle, about 3% context capacity in this preset.

With the CPU/disk settings above, a tested suffix edit in a 520k-token prompt
restored 516,096 tokens in 1.66 seconds instead of recomputing the full prompt
in 70.07 seconds. Restoration also passes after restarting both the server
and cache process. These are request-latency checks, separate from the
Workstation throughput measurements below.

## Capacity and controls

| Setting | Preset value | Change before the image name |
|---|---|---|
| GPUs / parallelism | TP2, DCP2 | Keep this pair for the tested memory configuration |
| Speculation | MTP3 | `-e MTP_DEPTH=0` disables speculation |
| Request slots | 4 | `-e MAX_NUM_SEQS=4` |
| Prefill budget | 3072 tokens | `-e MAX_NUM_BATCHED_TOKENS=3072` |
| KV allocation | 3996 MiB per GPU | `-e KV_CACHE_MEMORY_BYTES=3758096384` selects 3.5 GiB |
| Context capacity | About 899k with the LMCache configuration above | Startup reports the resolved limit for your cache mode |
| Vision | Enabled, no one-image admission cap | Image size/count still consume memory |
| Prefix policy | Request-boundary checkpoints | No retention-interval parameter needed |

Capacity is shared by requests and includes generated output. The preset has
a tight VRAM budget: long contexts together with large images can trigger
allocator retries. Lower the explicit KV allocation if your workload needs
more working memory. Increasing slots or changing speculation also changes
graph memory; do not assume the same capacity.

Target KV is FP8 and recurrent state is FP32. The target vocabulary head
remains BF16; MTP uses a private NVFP4 draft head. Full-and-piecewise CUDA
graphs cover 1, 2, 4, 8, 12 and 16 verifier rows. Two-shot all-reduce is off.

## Measured speed

Two RTX PRO 6000 **Workstation** cards at **stock clocks, without +6000 VRAM
overclock**; TP2/DCP2 MTP3, 3072-token prefill budget, four slots, temperature 1
and top-p .95. Decode uses one warmed 30-second measurement per concurrency;
prefill uses a 30-second window of uncached 32K requests. These are not medians
of repeated benchmark runs.

| Measurement | Speed |
|---|---:|
| C1 decode, context 0 | **186.5 output tok/s** |
| C4 decode, context 0 | **403.1 output tok/s total** |
| 32K prefill | **10,919 input tok/s** |

The separate Max-Q server validates startup and cache recovery, not the speeds
above. Source composition and test details are listed in
[vLLM issue #808](https://github.com/local-inference-lab/vllm/issues/808).
The [R2 archive](glm-5.3-flash-spark-tp2-r2-archive.md) retains the separate
CUDA 13.3 release, its measurements and its one-image limit.
The [versioned guide archive](../archive/serving-guides/README.md) preserves
the preceding beta recipe and cache measurements.

For four GPUs use the [GLM TP4 recipe](glm-5.3-flash.md). For other models and
common controls see the [shared Docker guide](../docs/unified-vllm-docker.md).
