# Qwen3.8-Flash-Next

Serve `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` using the
`qwen38-flash-next` profile in the [shared Docker guide](../docs/unified-vllm-docker.md).
The image and launcher are shared with GLM and DeepSeek; no GLM entrypoint
bypass or copied kernel environment is needed. This is not
[Qwen3.8-27B](qwen38-27b.md).

## Start on one GPU: TP1

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name qwen38 --init --restart unless-stopped \
  --gpus '"device=0"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v qwen38-runtime:/cache \
  -e PROFILE=qwen38-flash-next -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=1 -e PORT=8000 "$IMAGE"
```

The profile selects MTP3 and CPU PLE tables. The API model is
`Qwen3.8-Flash-Next` on port 8000. It downloads the checkpoint by name and
does not change GPU clocks. Check startup with `docker logs -f qwen38`.

## Start on two GPUs: TP2

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name qwen38-tp2 --init --restart unless-stopped \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v qwen38-runtime:/cache \
  -e PROFILE=qwen38-flash-next -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=2 -e DCP=1 -e PORT=8000 "$IMAGE"
```

Stop an overlapping TP1 instance first. The checkpoint and PLE placement stay
the same. TP2 splits model weights across two GPUs; it does not require DCP2.
Use DCP1 for the measured configuration and for external prefix-cache modes.
TP2 with MTP3 and vision passes text/image checks and text-prefix recovery
from RAM and disk after restart. TP1 and TP2 speed measurements below identify
their GPU clocks and image separately.

Optional arguments go **after `"$IMAGE"`**:

| Choice | Arguments |
|---|---|
| MTP3 | Default, or `--mode mtp --draft-tokens 3` |
| No speculation | `--mode off` |
| Vision | `--no-language-model-only` |
| Eight-GiB KV budget used by the comparison | `--kv-cache-memory-bytes 8589934592` |

Use `-e PORT=8001` before the image to change the API port.
The [Compose example](qwen38-flash-next/qwen38-flash-next.compose.yml)
provides TP1 and TP2 services using these same image profiles:

```bash
export LIL_IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
curl -fLO https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/qwen38-flash-next/qwen38-flash-next.compose.yml
GPU=0 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp1 up -d
```

For two GPUs select `--profile tp2` and set `GPU0`/`GPU1`.
The [Karmic Kraken benchmark table](../benchmarks/karmic-kraken-serving.md)
records the TP1 comparison against saved JJ measurements.

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
<summary>Qwen3.8 Flash Next: TP1, MTP3: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/qwen38-tp1.compose.yaml). Save it as `qwen38-tp1.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f qwen38-tp1.compose.yaml up -d
```

Logs: `docker compose -f qwen38-tp1.compose.yaml logs -f model`.
Stop: `docker compose -f qwen38-tp1.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: qwen38-tp1
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-3221ccacf71002ea
    container_name: qwen38-tp1
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
    - qwen38-tp1-runtime:/cache
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
            capabilities:
            - gpu
    environment:
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/b12x/cute
      B12X_DENSE_SPLITK_TURBO: '1'
      B12X_DYNAMIC_DETERMINISTIC_OUTPUT: '0'
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
      CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0
      CUDA_BINARY_LOADER_THREAD_COUNT: '8'
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/cute-dsl
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
      NCCL_CUMEM_ENABLE: '0'
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
      NVIDIA_VISIBLE_DEVICES: '0'
      NVJITLINK_VERSION: 13.4.52
      NVJPEG_VERSION: 13.2.2.35
      NVPL_LAPACK_MATH_MODE: PEDANTIC
      NVPTXCOMPILER_VERSION: 13.4.59
      NVRX_VERSION: 0.6.0
      NVSHMEM_VERSION: 3.7.1
      NVVM_VERSION: 13.4.59
      OMPI_MCA_coll_hcoll_enable: '0'
      OMP_NUM_THREADS: '2'
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/triton
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
      VLLM_B12X_DENSE_ACTIVATION_MODE: auto
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04/vllm
      VLLM_CAUSAL_CONV1D_UPDATE_HOIST: '1'
      VLLM_COMPUTE_NANS_IN_LOGITS: '0'
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      VLLM_ENABLE_PCIE_ALLREDUCE: '1'
      VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH: '1'
      VLLM_LM_HEAD_A16: '1'
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '0'
      VLLM_MTP_NVFP4_LM_HEAD: '1'
      VLLM_MXFP8_LM_HEAD: '0'
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PLE_CPU_OFFLOAD: '1'
      VLLM_QWEN3_8_FLASH_NEXT_HC_TP: '0'
      VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT: '1'
      VLLM_QWEN3_8_FLASH_NEXT_OVERLAP: '1'
      VLLM_SSM_CONV_STATE_LAYOUT: DS
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-ab35f5750b0aca04
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  qwen38-tp1-runtime:
    name: qwen38-tp1-runtime
  lil-huggingface:
    name: lil-huggingface
configs:
  lil-launch:
    content: |
      schema: lil-explicit-launch/v1
      profile: qwen38-flash-next
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
        model: local-inference-lab/Qwen3.8-Flash-Next-NVFP4
        served-model-name: Qwen3.8-Flash-Next
        tensor-parallel-size: 1
        mode: mtp
        max-model-len: 262144
        max-num-seqs: 16
        max-num-batched-tokens: 6019
        gpu-memory-utilization: 0.96
        block-size: 64
        max-cudagraph-capture-size: 64
        compilation-config:
          cudagraph_mode: FULL_AND_PIECEWISE
          custom_ops:
          - all
        mamba-cache-mode: align
        mamba-ssm-cache-dtype: auto
        async-scheduling: true
        quantization: modelopt_mixed
        moe-backend: b12x
        linear-backend: b12x
        gdn-decode-kernel: b12x
        enable-flashinfer-autotune: false
        mm-encoder-tp-mode: data
        mm-processor-cache-gb: 0.0
        language-model-only: false
        reasoning-parser: qwen3
        tool-call-parser: qwen3_xml
        speculative-config:
          method: mtp
          moe_backend: b12x
          num_speculative_tokens: 3
        draft-tokens: 3
      passthrough: []
      vllm_defaults:
      - additional-config
      - attention-backend
      - code-revision
      - cp-kv-cache-interleave-size
      - cudagraph-capture-sizes
      - dcp-kv-cache-interleave-size
      - decode-refill-target
      - default-chat-template-kwargs
      - disable-custom-all-reduce
      - enable-force-include-usage
      - enable-request-id-headers
      - engram-config
      - generation-config
      - hf-overrides
      - jit-monitor-mode
      - kv-cache-memory-bytes
      - max-parallel-prefills
      - override-generation-config
      - prefill-compute-half-life
      - prefill-compute-share
      - prefill-policy
      - prefill-schedule-interval
      - prefix-cache-retention-interval
      - prefix-match-unit
      - recurrent-checkpoint-policy
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
      - B12X_DENSE_SPLITK_TURBO
      - B12X_DYNAMIC_DETERMINISTIC_OUTPUT
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
      - NCCL_CUMEM_ENABLE
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
      - VLLM_B12X_DENSE_ACTIVATION_MODE
      - VLLM_CACHE_DIR
      - VLLM_CACHE_ROOT
      - VLLM_CAUSAL_CONV1D_UPDATE_HOIST
      - VLLM_COMPUTE_NANS_IN_LOGITS
      - VLLM_DISABLED_KERNELS
      - VLLM_ENABLE_PCIE_ALLREDUCE
      - VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH
      - VLLM_LM_HEAD_A16
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
      - VLLM_MTP_NVFP4_LM_HEAD
      - VLLM_MXFP8_LM_HEAD
      - VLLM_PCIE_ALLREDUCE_BACKEND
      - VLLM_PLE_CPU_OFFLOAD
      - VLLM_QWEN3_8_FLASH_NEXT_HC_TP
      - VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT
      - VLLM_QWEN3_8_FLASH_NEXT_OVERLAP
      - VLLM_SSM_CONV_STATE_LAYOUT
      - VLLM_USE_V2_MODEL_RUNNER
      - VLLM_WORKER_MULTIPROC_METHOD
      - XDG_CACHE_HOME
      - _CUDA_COMPAT_PATH
```

The explicit runner passes these native arguments to vLLM through the image's
CUDA/NCCL bootstrap. This command is shown for inspection; the Compose file
above also supplies its environment and persistent volumes.

```bash
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --async-scheduling \
  --block-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --decode-context-parallel-size 1 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --gdn-decode-kernel b12x \
  --gpu-memory-utilization 0.96 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --no-language-model-only \
  --linear-backend b12x \
  --load-format instanttensor \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype auto \
  --max-cudagraph-capture-size 64 \
  --max-model-len 262144 \
  --max-num-batched-tokens 6019 \
  --max-num-seqs 16 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 0.0 \
  --moe-backend b12x \
  --pipeline-parallel-size 1 \
  --port 8000 \
  --quantization modelopt_mixed \
  --reasoning-parser qwen3 \
  --served-model-name Qwen3.8-Flash-Next \
  --speculative-config '{"method":"mtp","moe_backend":"b12x","num_speculative_tokens":3}' \
  --tensor-parallel-size 1 \
  --tool-call-parser qwen3_xml
```

</details>

<details>
<summary>Qwen3.8 Flash Next: TP2, MTP3: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/qwen38-tp2.compose.yaml). Save it as `qwen38-tp2.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f qwen38-tp2.compose.yaml up -d
```

Logs: `docker compose -f qwen38-tp2.compose.yaml logs -f model`.
Stop: `docker compose -f qwen38-tp2.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: qwen38-tp2
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-3221ccacf71002ea
    container_name: qwen38-tp2
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
    - qwen38-tp2-runtime:/cache
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
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/b12x/cute
      B12X_DENSE_SPLITK_TURBO: '1'
      B12X_DYNAMIC_DETERMINISTIC_OUTPUT: '0'
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
      CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0
      CUDA_BINARY_LOADER_THREAD_COUNT: '8'
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/cute-dsl
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
      NCCL_CUMEM_ENABLE: '0'
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
      NVIDIA_VISIBLE_DEVICES: 0,1
      NVJITLINK_VERSION: 13.4.52
      NVJPEG_VERSION: 13.2.2.35
      NVPL_LAPACK_MATH_MODE: PEDANTIC
      NVPTXCOMPILER_VERSION: 13.4.59
      NVRX_VERSION: 0.6.0
      NVSHMEM_VERSION: 3.7.1
      NVVM_VERSION: 13.4.59
      OMPI_MCA_coll_hcoll_enable: '0'
      OMP_NUM_THREADS: '2'
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/triton
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
      VLLM_B12X_DENSE_ACTIVATION_MODE: auto
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c/vllm
      VLLM_CAUSAL_CONV1D_UPDATE_HOIST: '1'
      VLLM_COMPUTE_NANS_IN_LOGITS: '0'
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      VLLM_ENABLE_PCIE_ALLREDUCE: '1'
      VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH: '1'
      VLLM_LM_HEAD_A16: '1'
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '0'
      VLLM_MTP_NVFP4_LM_HEAD: '1'
      VLLM_MXFP8_LM_HEAD: '0'
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PLE_CPU_OFFLOAD: '1'
      VLLM_QWEN3_8_FLASH_NEXT_HC_TP: '0'
      VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT: '1'
      VLLM_QWEN3_8_FLASH_NEXT_OVERLAP: '1'
      VLLM_SSM_CONV_STATE_LAYOUT: DS
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-c5051345ff7dfc2c
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  qwen38-tp2-runtime:
    name: qwen38-tp2-runtime
  lil-huggingface:
    name: lil-huggingface
configs:
  lil-launch:
    content: |
      schema: lil-explicit-launch/v1
      profile: qwen38-flash-next
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
        model: local-inference-lab/Qwen3.8-Flash-Next-NVFP4
        served-model-name: Qwen3.8-Flash-Next
        tensor-parallel-size: 2
        mode: mtp
        max-model-len: 262144
        max-num-seqs: 16
        max-num-batched-tokens: 6019
        gpu-memory-utilization: 0.96
        block-size: 64
        max-cudagraph-capture-size: 64
        compilation-config:
          cudagraph_mode: FULL_AND_PIECEWISE
          custom_ops:
          - all
        mamba-cache-mode: align
        mamba-ssm-cache-dtype: auto
        async-scheduling: true
        quantization: modelopt_mixed
        moe-backend: b12x
        linear-backend: b12x
        gdn-decode-kernel: b12x
        enable-flashinfer-autotune: false
        mm-encoder-tp-mode: data
        mm-processor-cache-gb: 0.0
        language-model-only: false
        reasoning-parser: qwen3
        tool-call-parser: qwen3_xml
        speculative-config:
          method: mtp
          moe_backend: b12x
          num_speculative_tokens: 3
        draft-tokens: 3
      passthrough: []
      vllm_defaults:
      - additional-config
      - attention-backend
      - code-revision
      - cp-kv-cache-interleave-size
      - cudagraph-capture-sizes
      - dcp-kv-cache-interleave-size
      - decode-refill-target
      - default-chat-template-kwargs
      - disable-custom-all-reduce
      - enable-force-include-usage
      - enable-request-id-headers
      - engram-config
      - generation-config
      - hf-overrides
      - jit-monitor-mode
      - kv-cache-memory-bytes
      - max-parallel-prefills
      - override-generation-config
      - prefill-compute-half-life
      - prefill-compute-share
      - prefill-policy
      - prefill-schedule-interval
      - prefix-cache-retention-interval
      - prefix-match-unit
      - recurrent-checkpoint-policy
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
      - B12X_DENSE_SPLITK_TURBO
      - B12X_DYNAMIC_DETERMINISTIC_OUTPUT
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
      - NCCL_CUMEM_ENABLE
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
      - VLLM_B12X_DENSE_ACTIVATION_MODE
      - VLLM_CACHE_DIR
      - VLLM_CACHE_ROOT
      - VLLM_CAUSAL_CONV1D_UPDATE_HOIST
      - VLLM_COMPUTE_NANS_IN_LOGITS
      - VLLM_DISABLED_KERNELS
      - VLLM_ENABLE_PCIE_ALLREDUCE
      - VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH
      - VLLM_LM_HEAD_A16
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
      - VLLM_MTP_NVFP4_LM_HEAD
      - VLLM_MXFP8_LM_HEAD
      - VLLM_PCIE_ALLREDUCE_BACKEND
      - VLLM_PLE_CPU_OFFLOAD
      - VLLM_QWEN3_8_FLASH_NEXT_HC_TP
      - VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT
      - VLLM_QWEN3_8_FLASH_NEXT_OVERLAP
      - VLLM_SSM_CONV_STATE_LAYOUT
      - VLLM_USE_V2_MODEL_RUNNER
      - VLLM_WORKER_MULTIPROC_METHOD
      - XDG_CACHE_HOME
      - _CUDA_COMPAT_PATH
```

The explicit runner passes these native arguments to vLLM through the image's
CUDA/NCCL bootstrap. This command is shown for inspection; the Compose file
above also supplies its environment and persistent volumes.

```bash
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --async-scheduling \
  --block-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --decode-context-parallel-size 1 \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --gdn-decode-kernel b12x \
  --gpu-memory-utilization 0.96 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --no-language-model-only \
  --linear-backend b12x \
  --load-format instanttensor \
  --mamba-cache-mode align \
  --mamba-ssm-cache-dtype auto \
  --max-cudagraph-capture-size 64 \
  --max-model-len 262144 \
  --max-num-batched-tokens 6019 \
  --max-num-seqs 16 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-gb 0.0 \
  --moe-backend b12x \
  --pipeline-parallel-size 1 \
  --port 8000 \
  --quantization modelopt_mixed \
  --reasoning-parser qwen3 \
  --served-model-name Qwen3.8-Flash-Next \
  --speculative-config '{"method":"mtp","moe_backend":"b12x","num_speculative_tokens":3}' \
  --tensor-parallel-size 2 \
  --tool-call-parser qwen3_xml
```

</details>
<!-- END LIL-COMPOSE-REFERENCE -->

## Precision, model tables and cache

Put these `-e` overrides **before `"$IMAGE"`**:

| Setting | Default / recommendation | Example override |
|---|---|---|
| Active requests | 16 | `-e MAX_NUM_SEQS=8` |
| Prefill budget | 6019 tokens | `-e MAX_NUM_BATCHED_TOKENS=4096` |
| Context limit | 262,144 | `-e MAX_MODEL_LEN=131072` |
| GPU memory fraction | 0.96 | `-e GPU_MEMORY_UTILIZATION=0.93` to reserve more working space |
| PLE table placement | Host RAM | `-e VLLM_PLE_TABLE_MEMORY=disk` for native disk loading |
| Prefix storage | GPU-only | `-e CACHE_MODE=lmcache` with [RAM/disk controls](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload) |

The checkpoint uses mixed ModelOpt NVFP4, not four-bit storage for every
tensor. The target vocabulary head stays BF16; MTP has a private NVFP4 head
with BF16 activations. B12X handles MoE, dense kernels and GDN decode, with
native Qwen attention. The V2 runner uses full-and-piecewise graphs.

Attention KV is FP8, while recurrent state follows the model's native contract.
Prefix caching is on; the native `auto` policy selects exact recurrent request
boundaries where supported. Leave this policy to the model profile.

PLE is a learned embedding table, not request KV and not an n-gram speculator.
Historical startup accounting records about 26.82 GiB of mapped host tables;
leave additional host RAM for loading and the server. Keep offload enabled for
the one-96-GB-GPU recipe. Disk mode requires fast local storage and still uses
host working memory; the performance table uses RAM, not disk placement.

Native disk reads need io_uring permission. The [Docker configurator](https://local-inference-lab.ai/docker)
adds the required setting when you select disk tables. When editing a command
manually, also add `--security-opt seccomp=unconfined` before the image, or use
a custom seccomp profile that permits io_uring. The unconfined option disables
Docker's default syscall filter for that container; RAM placement does not need it.

The shared guide explains [prefix retention](../docs/unified-vllm-docker.md#prefix-cache-defaults).
Do not add a global `--prefix-cache-retention-interval 4096` override.
For vision, append `--no-language-model-only` after the image name. Image-bearing
requests do not restore recurrent checkpoints through external LMCache; their
uncached vision path remains available.

The TP1 measurements use temperature 1/top-p .95/top-k 20; the TP2 comparison
uses temperature 1/top-p .95 with top-k disabled.
The profile leaves checkpoint generation configuration authoritative rather
than claiming all checkpoint revisions have identical server defaults.
For a non-thinking request, use
`"chat_template_kwargs":{"enable_thinking":false}`; it is a different
workload from a reasoning benchmark.

## CPU prefix cache without LMCache

To reuse evicted text prefixes from host RAM, start Qwen with vLLM's built-in
CPU offload. This is separate from the PLE weight tables and from LMCache:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name qwen38-cpu-cache --init --restart unless-stopped \
  --gpus '"device=0,1"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v qwen38-runtime:/cache \
  -e PROFILE=qwen38-flash-next -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=2 -e DCP=1 -e PORT=8000 \
  -e CACHE_MODE=native -e NATIVE_KV_OFFLOADING_SIZE_GB=64 "$IMAGE"
```

The 64 GiB CPU allocation is total across ranks; reserve additional host RAM
for PLE tables, loading and the OS. MTP3 stays enabled. The launcher selects
the SimpleCPU connector and aligned checkpoints. Keep DCP1 and do not override
`VLLM_USE_SIMPLE_KV_OFFLOAD=0`: the generic Qwen offload path has an unresolved
restore-correctness problem. CPU prefixes disappear when the server stops;
use `CACHE_MODE=lmcache` when you need the separate persistent RAM/disk service.
[CPU-restore checks](../benchmarks/karmic-public-feedback.md#qwen-cpu-offload-investigation).

## Measured performance

### One GPU, VRAM +6000

One RTX PRO 6000 **Max-Q**, **VRAM +6000**, automatic graphics clocks;
TP1/MTP3, CPU PLE, FP8 KV, 6019-token budget, 16 slots and explicit eight-GiB
KV allocation. Temperature 1/top-p .95/top-k 20; five warmed 30-second
windows per cell. Decode uses context zero; uncached nominal-32K prefill
uses client TTFT. C8 is aggregate.

| Metric | Saved JJ R35 | Karmic Kraken | Change |
|---|---:|---:|---:|
| C1 output | 157.6 tok/s | 173.2 tok/s | +9.91% |
| C8 aggregate output | 674.2 tok/s | 698.4 tok/s | +3.58% |
| 32K prefill | 12,104 tok/s | 12,073 tok/s | −0.26% |
| C1 verifier rate | 76.06 steps/s | 81.78 steps/s | +7.52% |

Arithmetic and repeated/changed-prefix checks pass. Output includes draft
acceptance as well as execution speed.
[Exact image versions, configuration, repeats and samples](../benchmarks/karmic-kraken-serving.md).

A separate five-run restart with the same image and prepared cache measures
178.5 C1 and 713.7 C8 tok/s. Both startup series are retained in the report.

The eight-GiB allocation reports **434,258 logical KV tokens**, shared across
requests. It now includes safe recurrent endpoint and restore reservations;
the R35 report's 517,581-token estimate omitted those reserves. The physical
eight-GiB budget did not shrink. See the
[capacity accounting](../benchmarks/qwen-boundary-capacity-accounting.md).

### Two GPUs, stock clocks

Two RTX PRO 6000 **Max-Q**, **no overclock**, TP2/DCP1/MTP3, CPU PLE,
FP8 KV, 6019-token batch budget, 16 slots and eight GiB KV per rank.
Temperature 1/top-p .95, top-k disabled, reasoning effort `medium`.
Decode is the median of five warmed 30-second context-zero runs; C8 is
aggregate output. The 32k figure is the median of three uncached windows,
twelve requests each, using client TTFT.

| Measurement | vLLM | Community SGLang |
|---|---:|---:|
| C1 output | 222.3 tok/s | 200.8 tok/s |
| C8 aggregate output | 906.8 tok/s | 891.7 tok/s |
| 32k prefill | 14,570 tok/s | Not measured in this comparison |

vLLM image: `karmic-kraken-beta-20260922-3221ccacf71002ea`, without local
source changes. This is +10.7% C1 and +1.7% C8; C8 is approximately matched.
Against the preceding measured beta, output is −3.0% C1 / +0.5% C8 and
prefill is −0.75%; verifier rates differ by −0.6% / +0.2%.
A separate five-run confirmation gives 222.6 / 905.1 tok/s at C1/C8.
The three prefill windows span 14,547–14,586 tok/s. Both engines
use the same QAD checkpoint, but their runtime versions and recurrent-state
precision differ. [Configuration and repeated comparison](../benchmarks/qwen38-tp2-sglang.md)
and [published-image confirmation](../benchmarks/karmic-integration-merge-audit.md#qwen-tp2-performance-gate).

One additional C8 test on image `04a3c00a18b9d45f`, with EOS stopping disabled, triggered the loop
guard; that failed cell is not a speed result. Two controls with normal EOS
handling passed, but the cause is not established. The report preserves the
failure alongside the successful five-run series.

### TP4 memory option

For TP4/MTP3, use four GPU IDs and `-e TP=4` in the launch command. The
hardware profile's default is sixteen NCCL channels. A measured alternative
is to add `-e NCCL_MIN_NCHANNELS=2 -e NCCL_MAX_NCHANNELS=2` before `"$IMAGE"`.
It reduced sampled device memory by about 0.8–1.2 GiB per GPU and improved
decode in both TP4 test series. The gain varies between server starts;
the [TP4 results](../benchmarks/tp4-memory-controls.md#qwen-tp4-confirmations)
retain both series. This option was not measured at TP1/TP2 and should not
be copied into every model's configuration.

## Quality evaluation and historical releases

- [Published NVFP4 versus QAD AA-LCR](qwen38-flash-next/aa-lcr-nvfp4-vs-qad.md).
- [Direct-answer arithmetic stability](qwen38-flash-next/direct-arithmetic-stability-nvfp4-vs-qad.md).
- [Community R35 deployment and measurement archive](qwen38-flash-next-community-r35.md):
  +6000-clock results, SGLang/Sieve comparisons, older capacity measurements
  and exact recipe boundaries.
- [Versioned guide archive](../archive/serving-guides/README.md): preceding
  stock Workstation tables and complete launch instructions.

Sieve was not remeasured in the Max-Q matrix.
Source review and integration checklist: [issue #808](https://github.com/local-inference-lab/vllm/issues/808).
