# One Docker image for GLM, Qwen and DeepSeek

Choose a model profile and GPU IDs. The shared image supplies that model's
backend, graph, loader and cache defaults. You do not need a model-specific
entrypoint or a copied block of kernel variables.

## Select the image

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
```

This is the Karmic Kraken integration channel: CUDA 13.4.1, PyTorch 2.14,
vLLM, B12X, FlashInfer and LMCache. Linux x86-64, Docker, NVIDIA Container
Toolkit and a compatible NVIDIA driver are required. For native CUDA JIT,
`nvidia-smi` must report CUDA 13.4 or higher; installing a host CUDA toolkit
does not upgrade the driver. The examples target 96-GB RTX PRO 6000 GPUs.
No command changes clocks.

All channels use the same build recipe and launch interface:

| Image tag after `ghcr.io/local-inference-lab/vllm:` | vLLM branch | B12X branch |
|---|---|---|
| `karmic-kraken-beta` | `integration/karmic-kraken-beta` | `integration/karmic-kraken-beta` |
| `karmic-kraken` | `dev/karmic-kraken` | `master` |
| `jovian-judgement-beta` | `integration/beta` | `integration/beta` |
| `jovian-judgement` | `dev/jovian-judgement` | `master` |

Use the beta channel for the integrated fixes listed in
[issue #808](https://github.com/local-inference-lab/vllm/issues/808).
Published image versions and source manifests are available in the
[container releases](https://github.com/local-inference-lab/blackwell-llm-docker/releases).
Pulling an image does not change a running container; recreate that container
in a maintenance window to update it.

The [recipe archive](../archive/serving-guides/README.md) preserves preceding
commands and measurements with fixed image versions. Use a dated release tag
from that archive when reproducing a result; the short channel tags move when
another build is published.

## Choose a model

Each model page has a complete copyable command:

To inspect everything passed to vLLM, use the
[expandable full Compose configurations](#expand-the-complete-default-configurations)
below or the equivalent block on each model page.

| Model | Selector before the image | GPUs | Default speculation |
|---|---|---:|---|
| [GLM-5.3-Flash](../models/glm-5.3-flash.md) | `-e PROFILE=glm53-flash` | 4 | Off; the page starts MTP3 explicitly |
| [GLM Spark TP2](../models/glm-5.3-flash-spark-tp2.md) | `-e PRESET=glm53-spark-tp2` | 2 | MTP3 |
| [Qwen3.8 Flash Next](../models/qwen38-flash-next.md) | `-e PROFILE=qwen38-flash-next` | 1 or 2 | MTP3 |
| [DeepSeek V4 text](../models/deepseek-v4-flash.md) | `-e PROFILE=ds4-flash` | 2 | DSpark K5 |
| [DeepSeek V4 Vision](../models/deepseek-v4-flash-vision.md) | `-e PROFILE=ds4-vision` | 2 | DSpark K3 |
| [DeepSeek V4.1](../models/deepseek-v4.1-flash.md) | `-e PROFILE=ds41-flash` | 4 | Adaptive DSpark K7 |

Profiles provide Hugging Face checkpoint names. `MODEL` overrides the checkpoint
within that architecture; it does not select another architecture. Spark TP2
is a separate memory configuration, not the TP4 recipe with only `TP=2` changed.

## Start a server

Example: GLM, four GPUs, MTP3, API port 8000:

```bash
IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
docker pull "$IMAGE"
docker run -d --name glm53 --init --restart unless-stopped \
  --gpus '"device=0,1,2,3"' --network host --ipc host --shm-size 32g \
  -v lil-huggingface:/root/.cache/huggingface -v glm53-runtime:/cache \
  -e PROFILE=glm53-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=4 -e PORT=8000 "$IMAGE" --mode mtp --draft-tokens 3
```

Change Docker options and `-e` variables **before** the image name.
Native serving arguments go **after** it. For example:

| Change | Before the image | Equivalent native argument after the image |
|---|---|---|
| Two model shards | `-e TP=2` | `--tensor-parallel-size 2` |
| DCP2 | `-e DCP=2` | `--decode-context-parallel-size 2` |
| API port | `-e PORT=8001` | `--port 8001` |
| Maximum active requests | `-e MAX_NUM_SEQS=16` | `--max-num-seqs 16` |
| Prefill token budget | `-e MAX_NUM_BATCHED_TOKENS=4096` | `--max-num-batched-tokens 4096` |
| Context limit | `-e MAX_MODEL_LEN=131072` | `--max-model-len 131072` |
| Fixed eight-GiB KV allocation | `-e KV_CACHE_MEMORY_BYTES=8589934592` | `--kv-cache-memory-bytes 8589934592` |

Expose as many GPU IDs as TP requires. DCP must divide TP and be supported
by the selected model. Changing memory/graph settings can reduce capacity or
prevent startup; the model pages list the measured configurations.

```bash
docker logs -f glm53
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
docker stop glm53
docker start glm53
```

The API binds `0.0.0.0`; keep it on a trusted network or configure authentication.
For simultaneous servers use different GPU IDs, ports, names and runtime volumes.
The HF volume can be shared. Named volumes survive container removal.
`--restart unless-stopped` restores the container after Docker/host startup,
but does not restore overclock settings.

## Select speculation

Put one choice after `"$IMAGE"`:

| Choice | Arguments | Models |
|---|---|---|
| Target only | `--mode off` | All profiles |
| MTP3 | `--mode mtp --draft-tokens 3` | GLM, Qwen |
| DFlash2 K7 | `--mode dflash2 --draft-tokens 7` | GLM |
| DSpark K5 | `--mode dspark --draft-tokens 5` | DS4 text |
| DSpark K3 | `--mode dspark --draft-tokens 3` | DS4 Vision |
| Adaptive DSpark K7 | `--mode dspark --draft-tokens 7` | DS4.1 |

The draft count is a maximum proposal length, not the number guaranteed to
be accepted. GLM DFlash2 uses the offline MXFP8 draft
`local-inference-lab/GLM-5.3-Flash-DFlash2`. DS4 standard MTP needs a different
checkpoint contract; use its model page rather than changing only the mode.

## Cache storage: GPU, LMCache or native offload

GPU-only prefix caching is the default. LMCache adds host RAM and optional
disk storage for reusable request prefixes. It does not make the active
context larger and is separate from PLE/Engram model-table offload.

Add these variables before the image for **RAM cache**:

```bash
-e CACHE_MODE=lmcache -e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2 \
-e LMCACHE_L2_ENABLED=0
```

For **RAM plus disk**, add:

```bash
-e CACHE_MODE=lmcache -e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2 \
-e LMCACHE_L2_ENABLED=1 -e LMCACHE_L2_GB=64
```

The image starts and supervises the CPU-only cache service. Its persistent
directory is inside `/cache`; service ports derive from the model API port.
Set `LMCACHE_L2_ENABLED` explicitly when choosing RAM-only or RAM-plus-disk;
model profiles can have different disk-tier defaults.
LMCache reserves API-port + 10000, + 10001 and + 10002. For simultaneous
instances, space API ports at least three apart, for example 8000 and 8003,
or explicitly choose non-overlapping cache-service ports.
The GLM Spark TP2 page includes the bounded worker settings for that deployment.
Check host RAM, `/dev/shm` and disk capacity. With `--ipc host`, Docker's
`--shm-size` does not enlarge host shared memory.

| Model | RAM/disk prefix restore | Image-bearing requests |
|---|---|---|
| GLM, including Spark TP2 | Text supported | Vision runs, but external recurrent restore is skipped |
| Qwen | Text supported | Vision runs, but external recurrent restore is skipped |
| DS4 text | Supported | Use the Vision model for images |
| DS4 Vision | Supported | Image-keyed restore supported |
| DS4.1 | Supported | Image-keyed restore supported |

See [cache test results](../benchmarks/karmic-kraken-serving.md#prefix-cache-checks).
GLM additionally exposes native KV offload through `--cache-mode native`;
that is not the LMCache path measured here. Other profiles reject that choice.

### Prefix cache defaults

Leave prefix policy to the model profile. GLM keeps exact recurrent checkpoints
at request boundaries. Qwen's native automatic policy selects its supported
recurrent handling. DeepSeek uses its own attention-cache structure.

Do not add one global `--prefix-cache-retention-interval 4096` to every model.
That parameter changes recurrent checkpoint retention, not the scheduling batch
or a cache expiry time. The profile and cache adapter select compatible geometry.

## Qwen PLE and DS4.1 Engram placement

These learned n-gram tables are model weights, not prefix KV or draft speculators.

| Model | RAM | Disk |
|---|---|---|
| Qwen | `-e VLLM_PLE_TABLE_MEMORY=ram` before the image | `-e VLLM_PLE_TABLE_MEMORY=disk` before the image |
| DS4.1 | `--engram-table-memory ram` after the image | `--engram-table-memory disk` after the image |

Qwen's default is CPU PLE offload; keep it enabled for the one-GPU recipe.
DS4.1 defaults to disk Engram. Disk paths need fast local storage and still use
host working memory. DS4.1's command includes `--ulimit memlock=-1` and
`--security-opt seccomp=unconfined` for pinned memory and io_uring. Use trusted
images when relaxing that syscall filter. Placement alternatives do not carry
an unmeasured speed guarantee.

<!-- BEGIN LIL-COMPOSE-REFERENCE -->
## Expand the complete default configurations

These are runnable, release-tagged snapshots of the shared profiles, including
all non-secret image/profile ENV values and resolved serving options. They use
GPU-only prefix caching. Inactive `cache-*` options do not start LMCache.
They are deployment defaults, not copies of benchmark-only overrides.

Requires [Docker Compose 2.23.1 or newer](https://docs.docker.com/reference/compose-file/configs/).
`services.model.environment` is applied to vLLM; `configs.lil-launch.content`
contains the resolved options. The image's explicit runner retains bootstrap,
validation and cache supervision without re-reading model defaults.

Change GPU IDs in both `device_ids` and `NVIDIA_VISIBLE_DEVICES`. Only start one
example at a time unless GPU IDs, API ports and container names do not overlap.
Named volumes keep checkpoints and compiled kernels across container restarts.

For ordinary changes to TP, speculation or LMCache, prefer the short launch
commands on this page: they recompute dependent settings. The expanded files
freeze those settings; do not change only a derived field or image tag. The
`UNBOUND-RUNTIME` cache-path marker is filled by the image at startup.
`vllm_defaults` lists options left to native vLLM, not hidden profile overrides.
Host-injected credentials, container IDs and driver-provided variables are not
predicted by this static snapshot.
The image's `*_VERSION` ENV entries are build metadata; changing those strings
does not install another CUDA, NCCL or Python package.

The quick-start commands follow the moving beta channel. These expanded
files keep the release tag whose settings they display. Every model page has
the corresponding expandable block as well.

<details>
<summary>GLM-5.3 Flash: TP4, no speculation: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/glm53-tp4-off.compose.yaml). Save it as `glm53-tp4-off.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f glm53-tp4-off.compose.yaml up -d
```

Logs: `docker compose -f glm53-tp4-off.compose.yaml logs -f model`.
Stop: `docker compose -f glm53-tp4-off.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: glm53-tp4-off
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
    container_name: glm53-tp4-off
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
    - glm53-tp4-off-runtime:/cache
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
    environment:
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/cute
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
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/cute-dsl
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
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      PYTORCH_HOME: /opt/pytorch/pytorch
      PYTORCH_VERSION: 2.14.0a0+4fdf77b
      RDMACORE_VERSION: '63.0'
      SAFETENSORS_FAST_GPU: '1'
      SHELL: /bin/bash
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/triton
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
      VLLM_B12X_MLA_CKV_GATHER: '0'
      VLLM_B12X_MOE_FP4_FORCE_A16: '0'
      VLLM_B12X_NVFP4_ACTIVATION_MODE: quantized
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/vllm
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
      VLLM_PCIE_DMA_MIN_BYTES: 6MB
      VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE: 'off'
      VLLM_PLUGINS: ''
      VLLM_USE_FLASHINFER_SAMPLER: '1'
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  glm53-tp4-off-runtime:
    name: glm53-tp4-off-runtime
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
        enable-chunked-prefill: true
        enable-auto-tool-choice: true
        cache-mode: vram
        cache-gpu-workers: 8
        model: local-inference-lab/GLM-5.3-Flash-NVFP4
        served-model-name: GLM-5.3-Flash-NVFP4
        tensor-parallel-size: 4
        mode: 'off'
        prefill-compute-share: '0.4'
        prefill-schedule-interval: 1
        max-parallel-prefills: 1
        max-model-len: 1048576
        max-num-seqs: 32
        max-num-batched-tokens: 4096
        gpu-memory-utilization: 0.93
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
        max-cudagraph-capture-size: 256
        cudagraph-capture-sizes:
        - 1
        - 2
        - 4
        - 8
        - 16
        - 32
        - 40
        - 48
        - 64
        - 96
        - 128
        - 192
        - 256
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
        draft-tokens: 0
      passthrough: []
      vllm_defaults:
      - async-scheduling
      - code-revision
      - decode-refill-target
      - disable-custom-all-reduce
      - enable-force-include-usage
      - enable-prompt-tokens-details
      - enable-request-id-headers
      - engram-config
      - gdn-decode-kernel
      - generation-config
      - jit-monitor-mode
      - kv-cache-memory-bytes
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
      - speculative-config
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
      - VLLM_B12X_MLA_CKV_GATHER
      - VLLM_B12X_MOE_FP4_FORCE_A16
      - VLLM_B12X_NVFP4_ACTIVATION_MODE
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
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --attention-backend B12X \
  --block-size 256 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --cp-kv-cache-interleave-size 4 \
  --cudagraph-capture-sizes 1 2 4 8 16 32 40 48 64 96 128 192 256 \
  --dcp-kv-cache-interleave-size 4 \
  --decode-context-parallel-size 1 \
  --default-chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":false}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.93 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --linear-backend b12x \
  --load-format instanttensor \
  --mamba-cache-mode align \
  --max-cudagraph-capture-size 256 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
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
  --served-model-name GLM-5.3-Flash-NVFP4 \
  --tensor-parallel-size 4 \
  --tool-call-parser glm47
```

</details>

<details>
<summary>GLM-5.3 Flash: TP4, MTP3: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/glm53-tp4-mtp3.compose.yaml). Save it as `glm53-tp4-mtp3.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f glm53-tp4-mtp3.compose.yaml up -d
```

Logs: `docker compose -f glm53-tp4-mtp3.compose.yaml logs -f model`.
Stop: `docker compose -f glm53-tp4-mtp3.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: glm53-tp4-mtp3
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
    container_name: glm53-tp4-mtp3
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
    - glm53-tp4-mtp3-runtime:/cache
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
    environment:
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/cute
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
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/cute-dsl
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
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      PYTORCH_HOME: /opt/pytorch/pytorch
      PYTORCH_VERSION: 2.14.0a0+4fdf77b
      RDMACORE_VERSION: '63.0'
      SAFETENSORS_FAST_GPU: '1'
      SHELL: /bin/bash
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/triton
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
      VLLM_B12X_MLA_CKV_GATHER: '0'
      VLLM_B12X_MOE_FP4_FORCE_A16: '0'
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/vllm
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
      VLLM_PCIE_DMA_MIN_BYTES: 6MB
      VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE: 'off'
      VLLM_PLUGINS: ''
      VLLM_USE_FLASHINFER_SAMPLER: '1'
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  glm53-tp4-mtp3-runtime:
    name: glm53-tp4-mtp3-runtime
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
        enable-chunked-prefill: true
        enable-auto-tool-choice: true
        cache-mode: vram
        cache-gpu-workers: 8
        model: local-inference-lab/GLM-5.3-Flash-NVFP4
        served-model-name: GLM-5.3-Flash-NVFP4
        tensor-parallel-size: 4
        mode: mtp
        prefill-compute-share: '0.4'
        prefill-schedule-interval: 1
        max-parallel-prefills: 1
        max-model-len: 1048576
        max-num-seqs: 32
        max-num-batched-tokens: 4096
        gpu-memory-utilization: 0.93
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
        max-cudagraph-capture-size: 256
        cudagraph-capture-sizes:
        - 1
        - 2
        - 4
        - 8
        - 16
        - 32
        - 40
        - 48
        - 64
        - 96
        - 128
        - 192
        - 256
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
        draft-tokens: 3
        speculative-config:
          method: mtp
          draft_sample_method: probabilistic
          rejection_sample_method: standard
          moe_backend: marlin
          attention_backend: B12X
          num_speculative_tokens: 3
      passthrough: []
      vllm_defaults:
      - async-scheduling
      - code-revision
      - decode-refill-target
      - disable-custom-all-reduce
      - enable-force-include-usage
      - enable-prompt-tokens-details
      - enable-request-id-headers
      - engram-config
      - gdn-decode-kernel
      - generation-config
      - jit-monitor-mode
      - kv-cache-memory-bytes
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
      - VLLM_B12X_MLA_CKV_GATHER
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
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --attention-backend B12X \
  --block-size 256 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --cp-kv-cache-interleave-size 4 \
  --cudagraph-capture-sizes 1 2 4 8 16 32 40 48 64 96 128 192 256 \
  --dcp-kv-cache-interleave-size 4 \
  --decode-context-parallel-size 1 \
  --default-chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":false}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.93 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --linear-backend b12x \
  --load-format instanttensor \
  --mamba-cache-mode align \
  --max-cudagraph-capture-size 256 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
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
  --served-model-name GLM-5.3-Flash-NVFP4 \
  --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"marlin","attention_backend":"B12X","num_speculative_tokens":3}' \
  --tensor-parallel-size 4 \
  --tool-call-parser glm47
```

</details>

<details>
<summary>GLM-5.3 Flash: TP4, DFlash2 K7: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/glm53-tp4-dflash2.compose.yaml). Save it as `glm53-tp4-dflash2.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f glm53-tp4-dflash2.compose.yaml up -d
```

Logs: `docker compose -f glm53-tp4-dflash2.compose.yaml logs -f model`.
Stop: `docker compose -f glm53-tp4-dflash2.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: glm53-tp4-dflash2
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
    container_name: glm53-tp4-dflash2
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
    - glm53-tp4-dflash2-runtime:/cache
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
    environment:
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/cute
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
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/cute-dsl
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
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      PYTORCH_HOME: /opt/pytorch/pytorch
      PYTORCH_VERSION: 2.14.0a0+4fdf77b
      RDMACORE_VERSION: '63.0'
      SAFETENSORS_FAST_GPU: '1'
      SHELL: /bin/bash
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/triton
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
      VLLM_B12X_MLA_CKV_GATHER: '0'
      VLLM_B12X_MOE_FP4_FORCE_A16: '0'
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5/vllm
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
      VLLM_PCIE_DMA_MIN_BYTES: 6MB
      VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE: 'off'
      VLLM_PLUGINS: ''
      VLLM_USE_FLASHINFER_SAMPLER: '1'
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/glm53-flash-9a07df7f681265c5
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  glm53-tp4-dflash2-runtime:
    name: glm53-tp4-dflash2-runtime
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
        enable-chunked-prefill: true
        enable-auto-tool-choice: true
        cache-mode: vram
        cache-gpu-workers: 8
        model: local-inference-lab/GLM-5.3-Flash-NVFP4
        served-model-name: GLM-5.3-Flash-NVFP4
        tensor-parallel-size: 4
        mode: dflash2
        prefill-compute-share: '0.4'
        prefill-schedule-interval: 1
        max-parallel-prefills: 1
        max-model-len: 1048576
        max-num-seqs: 32
        max-num-batched-tokens: 4096
        gpu-memory-utilization: 0.93
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
        max-cudagraph-capture-size: 256
        cudagraph-capture-sizes:
        - 1
        - 2
        - 4
        - 8
        - 16
        - 32
        - 40
        - 48
        - 64
        - 96
        - 128
        - 192
        - 256
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
        draft-tokens: 7
        speculative-config:
          method: dflash
          model: local-inference-lab/GLM-5.3-Flash-DFlash2
          draft_sample_method: probabilistic
          rejection_sample_method: standard
          attention_backend: FLASH_ATTN
          kv_cache_dtype: auto
          num_speculative_tokens: 7
      passthrough: []
      vllm_defaults:
      - async-scheduling
      - code-revision
      - decode-refill-target
      - disable-custom-all-reduce
      - enable-force-include-usage
      - enable-prompt-tokens-details
      - enable-request-id-headers
      - engram-config
      - gdn-decode-kernel
      - generation-config
      - jit-monitor-mode
      - kv-cache-memory-bytes
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
      - VLLM_B12X_MLA_CKV_GATHER
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
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
  --attention-backend B12X \
  --block-size 256 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --cp-kv-cache-interleave-size 4 \
  --cudagraph-capture-sizes 1 2 4 8 16 32 40 48 64 96 128 192 256 \
  --dcp-kv-cache-interleave-size 4 \
  --decode-context-parallel-size 1 \
  --default-chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":false}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.93 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --linear-backend b12x \
  --load-format instanttensor \
  --mamba-cache-mode align \
  --max-cudagraph-capture-size 256 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
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
  --served-model-name GLM-5.3-Flash-NVFP4 \
  --speculative-config '{"method":"dflash","model":"local-inference-lab/GLM-5.3-Flash-DFlash2","draft_sample_method":"probabilistic","rejection_sample_method":"standard","attention_backend":"FLASH_ATTN","kv_cache_dtype":"auto","num_speculative_tokens":7}' \
  --tensor-parallel-size 4 \
  --tool-call-parser glm47
```

</details>

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
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
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
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/b12x/cute
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
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/cute-dsl
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/triton
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
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79/vllm
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
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/glm53-flash-a8cefca1ce25fc79
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
      - enable-prompt-tokens-details
      - enable-request-id-headers
      - engram-config
      - gdn-decode-kernel
      - generation-config
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
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
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
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/b12x/cute
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
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/cute-dsl
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/triton
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
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67/vllm
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
      VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT: '1'
      VLLM_QWEN3_8_FLASH_NEXT_OVERLAP: '1'
      VLLM_SSM_CONV_STATE_LAYOUT: DS
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-be6d1a6af3105c67
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
        language-model-only: true
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
      - enable-prompt-tokens-details
      - enable-request-id-headers
      - engram-config
      - generation-config
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
  --gdn-decode-kernel b12x \
  --gpu-memory-utilization 0.96 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --language-model-only \
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
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
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
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/b12x/cute
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
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/cute-dsl
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/triton
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
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c/vllm
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
      VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT: '1'
      VLLM_QWEN3_8_FLASH_NEXT_OVERLAP: '1'
      VLLM_SSM_CONV_STATE_LAYOUT: DS
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/qwen38-flash-next-f4cd7c441ed6d33c
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
        language-model-only: true
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
      - enable-prompt-tokens-details
      - enable-request-id-headers
      - engram-config
      - generation-config
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
  --gdn-decode-kernel b12x \
  --gpu-memory-utilization 0.96 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --language-model-only \
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

<details>
<summary>DeepSeek V4 text: TP2, DSpark K5: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/ds4-flash-tp2.compose.yaml). Save it as `ds4-flash-tp2.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f ds4-flash-tp2.compose.yaml up -d
```

Logs: `docker compose -f ds4-flash-tp2.compose.yaml logs -f model`.
Stop: `docker compose -f ds4-flash-tp2.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: ds4-flash-tp2
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
    container_name: ds4-flash-tp2
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
    - ds4-flash-tp2-runtime:/cache
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
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/b12x/cute
      BASH_ENV: /etc/bash.bashrc
      CCCL_VERSION: 13.3.4.2.1
      COCOAPI_VERSION: 2.0+nv0.8.1
      CUBLASMP_VERSION: 0.10.0.3695
      CUBLAS_VERSION: 13.7.0.27
      CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0
      CUDA_BINARY_LOADER_THREAD_COUNT: '8'
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/cute-dsl
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/triton
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
      VLLM_B12X_MOE_FP4_FORCE_A16: '0'
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da/vllm
      VLLM_ENABLE_PCIE_ALLREDUCE: '1'
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '1'
      VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: '1024'
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      VLLM_USE_AOT_COMPILE: '1'
      VLLM_USE_BREAKABLE_CUDAGRAPH: '0'
      VLLM_USE_FLASHINFER_SAMPLER: '1'
      VLLM_USE_MEGA_AOT_ARTIFACT: '1'
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/ds4-flash-6f6f0f3e881879da
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  ds4-flash-tp2-runtime:
    name: ds4-flash-tp2-runtime
  lil-huggingface:
    name: lil-huggingface
configs:
  lil-launch:
    content: |
      schema: lil-explicit-launch/v1
      profile: ds4-flash
      hardware: rtx-pro-6000-pcie
      options:
        cache-transfer-mode: engine_driven
        cache-l1-gib: 24.0
        cache-l1-init-gib: 24
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
        enable-chunked-prefill: true
        enable-auto-tool-choice: true
        cache-mode: vram
        model: deepseek-ai/DeepSeek-V4-Flash-0731
        served-model-name: DeepSeek-V4-Flash-0731
        tensor-parallel-size: 2
        mode: dspark
        max-model-len: -1
        max-num-seqs: 8
        max-num-batched-tokens: 4096
        gpu-memory-utilization: 0.975
        block-size: 256
        max-cudagraph-capture-size: 48
        compilation-config:
          cudagraph_mode: FULL_AND_PIECEWISE
          custom_ops:
          - all
        attention-backend: B12X
        moe-backend: b12x
        trust-remote-code: true
        prefix-cache-retention-interval: 4096
        async-scheduling: true
        scheduler-reserve-full-isl: false
        enable-flashinfer-autotune: true
        tokenizer-mode: deepseek_v4
        reasoning-parser: deepseek_v4
        tool-call-parser: deepseek_v4
        enable-prompt-tokens-details: true
        enable-force-include-usage: true
        enable-request-id-headers: true
        default-chat-template-kwargs:
          thinking: true
          reasoning_effort: high
        override-generation-config:
          temperature: 1.0
          top_p: 0.95
        revision: 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
        code-revision: 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
        speculative-config:
          method: dspark
          draft_sample_method: probabilistic
          rejection_sample_method: standard
          num_speculative_tokens: 5
          model: deepseek-ai/DeepSeek-V4-Flash-0731
          revision: 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
        draft-tokens: 5
      passthrough: []
      vllm_defaults:
      - additional-config
      - cp-kv-cache-interleave-size
      - cudagraph-capture-sizes
      - dcp-kv-cache-interleave-size
      - decode-refill-target
      - disable-custom-all-reduce
      - engram-config
      - gdn-decode-kernel
      - generation-config
      - jit-monitor-mode
      - kv-cache-memory-bytes
      - language-model-only
      - linear-backend
      - mamba-cache-mode
      - mamba-ssm-cache-dtype
      - max-parallel-prefills
      - mm-encoder-tp-mode
      - mm-processor-cache-gb
      - prefill-compute-half-life
      - prefill-compute-share
      - prefill-policy
      - prefill-schedule-interval
      - prefix-match-unit
      - quantization
      - recurrent-checkpoint-policy
      - safetensors-load-strategy
      - swa-block-size
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
      - VLLM_B12X_MOE_FP4_FORCE_A16
      - VLLM_CACHE_DIR
      - VLLM_CACHE_ROOT
      - VLLM_ENABLE_PCIE_ALLREDUCE
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
      - VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
      - VLLM_PCIE_ALLREDUCE_BACKEND
      - VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE
      - VLLM_USE_AOT_COMPILE
      - VLLM_USE_BREAKABLE_CUDAGRAPH
      - VLLM_USE_FLASHINFER_SAMPLER
      - VLLM_USE_MEGA_AOT_ARTIFACT
      - VLLM_USE_V2_MODEL_RUNNER
      - VLLM_WORKER_MULTIPROC_METHOD
      - XDG_CACHE_HOME
      - _CUDA_COMPAT_PATH
```

The explicit runner passes these native arguments to vLLM through the image's
CUDA/NCCL bootstrap. This command is shown for inspection; the Compose file
above also supplies its environment and persistent volumes.

```bash
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --async-scheduling \
  --attention-backend B12X \
  --block-size 256 \
  --code-revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --decode-context-parallel-size 1 \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --enable-force-include-usage \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-request-id-headers \
  --gpu-memory-utilization 0.975 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --load-format instanttensor \
  --max-cudagraph-capture-size 48 \
  --max-model-len -1 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --moe-backend b12x \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --pipeline-parallel-size 1 \
  --port 8000 \
  --prefix-cache-retention-interval 4096 \
  --reasoning-parser deepseek_v4 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --no-scheduler-reserve-full-isl \
  --served-model-name DeepSeek-V4-Flash-0731 \
  --speculative-config '{"method":"dspark","draft_sample_method":"probabilistic","rejection_sample_method":"standard","num_speculative_tokens":5,"model":"deepseek-ai/DeepSeek-V4-Flash-0731","revision":"9e165c30e2704aec5d9d593cce3eebd58bbef1cb"}' \
  --tensor-parallel-size 2 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --trust-remote-code
```

</details>

<details>
<summary>DeepSeek V4 Vision: TP2, DSpark K3: full Compose, ENV and vLLM command</summary>

[Download the complete Compose file](https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/docs/compose/karmic-kraken-beta/ds4-vision-tp2.compose.yaml). Save it as `ds4-vision-tp2.compose.yaml`,
choose GPU IDs, then run:

```bash
docker compose -f ds4-vision-tp2.compose.yaml up -d
```

Logs: `docker compose -f ds4-vision-tp2.compose.yaml logs -f model`.
Stop: `docker compose -f ds4-vision-tp2.compose.yaml down` (keeps model/cache volumes).

```yaml
# Generated from the selected image's shared runtime profiles.
# Requires Docker Compose 2.23.1 or newer. No HF credentials are embedded.
name: ds4-vision-tp2
services:
  model:
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
    container_name: ds4-vision-tp2
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
    - ds4-vision-tp2-runtime:/cache
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
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/b12x/cute
      BASH_ENV: /etc/bash.bashrc
      CCCL_VERSION: 13.3.4.2.1
      COCOAPI_VERSION: 2.0+nv0.8.1
      CUBLASMP_VERSION: 0.10.0.3695
      CUBLAS_VERSION: 13.7.0.27
      CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0
      CUDA_BINARY_LOADER_THREAD_COUNT: '8'
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/cute-dsl
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/triton
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
      VLLM_B12X_MOE_FP4_FORCE_A16: '0'
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb/vllm
      VLLM_ENABLE_PCIE_ALLREDUCE: '1'
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '1'
      VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: '1024'
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      VLLM_USE_AOT_COMPILE: '1'
      VLLM_USE_BREAKABLE_CUDAGRAPH: '0'
      VLLM_USE_FLASHINFER_SAMPLER: '1'
      VLLM_USE_MEGA_AOT_ARTIFACT: '1'
      VLLM_USE_V2_MODEL_RUNNER: '1'
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/ds4-vision-1d5a0da78eb5eefb
      _CUDA_COMPAT_PATH: /usr/local/cuda/compat
    configs:
    - source: lil-launch
      target: /etc/lil-launch.yaml
volumes:
  ds4-vision-tp2-runtime:
    name: ds4-vision-tp2-runtime
  lil-huggingface:
    name: lil-huggingface
configs:
  lil-launch:
    content: |
      schema: lil-explicit-launch/v1
      profile: ds4-vision
      hardware: rtx-pro-6000-pcie
      options:
        cache-transfer-mode: engine_driven
        cache-l1-gib: 24.0
        cache-l1-init-gib: 24
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
        enable-chunked-prefill: true
        enable-auto-tool-choice: true
        cache-mode: vram
        model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
        served-model-name: DeepSeek-V4-Flash-Vision-Exp
        tensor-parallel-size: 2
        mode: dspark
        max-model-len: -1
        max-num-seqs: 4
        max-num-batched-tokens: 4096
        gpu-memory-utilization: 0.975
        block-size: 256
        max-cudagraph-capture-size: 16
        compilation-config:
          cudagraph_mode: FULL_AND_PIECEWISE
          custom_ops:
          - all
        attention-backend: B12X
        moe-backend: b12x
        trust-remote-code: true
        prefix-cache-retention-interval: 4096
        async-scheduling: true
        scheduler-reserve-full-isl: false
        enable-flashinfer-autotune: true
        tokenizer-mode: deepseek_v4
        reasoning-parser: deepseek_v4
        tool-call-parser: deepseek_v4
        enable-prompt-tokens-details: true
        enable-force-include-usage: true
        enable-request-id-headers: true
        default-chat-template-kwargs:
          thinking: true
          reasoning_effort: high
        override-generation-config:
          temperature: 1.0
          top_p: 0.95
        revision: 6821d6ad3681a4b137b066b76094fa82ebd0a380
        code-revision: 6821d6ad3681a4b137b066b76094fa82ebd0a380
        speculative-config:
          method: dspark
          draft_sample_method: probabilistic
          rejection_sample_method: standard
          num_speculative_tokens: 3
          model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
          revision: 6821d6ad3681a4b137b066b76094fa82ebd0a380
        draft-tokens: 3
      passthrough: []
      vllm_defaults:
      - additional-config
      - cp-kv-cache-interleave-size
      - cudagraph-capture-sizes
      - dcp-kv-cache-interleave-size
      - decode-refill-target
      - disable-custom-all-reduce
      - engram-config
      - gdn-decode-kernel
      - generation-config
      - jit-monitor-mode
      - kv-cache-memory-bytes
      - language-model-only
      - linear-backend
      - mamba-cache-mode
      - mamba-ssm-cache-dtype
      - max-parallel-prefills
      - mm-encoder-tp-mode
      - mm-processor-cache-gb
      - prefill-compute-half-life
      - prefill-compute-share
      - prefill-policy
      - prefill-schedule-interval
      - prefix-match-unit
      - quantization
      - recurrent-checkpoint-policy
      - safetensors-load-strategy
      - swa-block-size
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
      - VLLM_B12X_MOE_FP4_FORCE_A16
      - VLLM_CACHE_DIR
      - VLLM_CACHE_ROOT
      - VLLM_ENABLE_PCIE_ALLREDUCE
      - VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
      - VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
      - VLLM_PCIE_ALLREDUCE_BACKEND
      - VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE
      - VLLM_USE_AOT_COMPILE
      - VLLM_USE_BREAKABLE_CUDAGRAPH
      - VLLM_USE_FLASHINFER_SAMPLER
      - VLLM_USE_MEGA_AOT_ARTIFACT
      - VLLM_USE_V2_MODEL_RUNNER
      - VLLM_WORKER_MULTIPROC_METHOD
      - XDG_CACHE_HOME
      - _CUDA_COMPAT_PATH
```

The explicit runner passes these native arguments to vLLM through the image's
CUDA/NCCL bootstrap. This command is shown for inspection; the Compose file
above also supplies its environment and persistent volumes.

```bash
/opt/venv/bin/python -m vllm.entrypoints.cli.main serve deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --async-scheduling \
  --attention-backend B12X \
  --block-size 256 \
  --code-revision 6821d6ad3681a4b137b066b76094fa82ebd0a380 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --decode-context-parallel-size 1 \
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --enable-force-include-usage \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-request-id-headers \
  --gpu-memory-utilization 0.975 \
  --host 0.0.0.0 \
  --kv-cache-dtype fp8 \
  --load-format instanttensor \
  --max-cudagraph-capture-size 16 \
  --max-model-len -1 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --moe-backend b12x \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --pipeline-parallel-size 1 \
  --port 8000 \
  --prefix-cache-retention-interval 4096 \
  --reasoning-parser deepseek_v4 \
  --revision 6821d6ad3681a4b137b066b76094fa82ebd0a380 \
  --no-scheduler-reserve-full-isl \
  --served-model-name DeepSeek-V4-Flash-Vision-Exp \
  --speculative-config '{"method":"dspark","draft_sample_method":"probabilistic","rejection_sample_method":"standard","num_speculative_tokens":3,"model":"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp","revision":"6821d6ad3681a4b137b066b76094fa82ebd0a380"}' \
  --tensor-parallel-size 2 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --trust-remote-code
```

</details>

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
    image: ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260920-443d9f815c57d23b
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
      B12X_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/b12x/compile
      B12X_CUTE_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/b12x/cute
      BASH_ENV: /etc/bash.bashrc
      CCCL_VERSION: 13.3.4.2.1
      COCOAPI_VERSION: 2.0+nv0.8.1
      CUBLASMP_VERSION: 0.10.0.3695
      CUBLAS_VERSION: 13.7.0.27
      CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0
      CUDA_BINARY_LOADER_THREAD_COUNT: '8'
      CUDA_CACHE_PATH: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/cuda
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
      CUTE_DSL_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/cute-dsl
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
      SPARKINFER_COMPILE_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/b12x/compile
      TENSORBOARD_PORT: '6006'
      TORCHAO_BUILD_VERSION: +gitdd0efc75
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/torchinductor
      TORCHINDUCTOR_CUTLASS_DIR: /opt/pytorch/pytorch/third_party/cutlass
      TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION: '0'
      TORCHTITAN_BUILD_VERSION: 0.2.2+gitbadf21a1
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE: '1'
      TORCH_CUDA_ARCH_LIST: 7.5 8.0 8.6 9.0 10.0 12.0+PTX
      TORCH_NCCL_USE_COMM_NONBLOCKING: '0'
      TRANSFORMER_ENGINE_VERSION: '2.18'
      TRITON_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/triton
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
      VLLM_CACHE_DIR: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/vllm
      VLLM_CACHE_ROOT: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6/vllm
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
      XDG_CACHE_HOME: /cache/jit/UNBOUND-RUNTIME/ds41-flash-c39ccd7dc267f5d6
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
        enable-prompt-tokens-details: true
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

## Inspect or override the launch

Inspect without loading a model or allocating GPUs:

```bash
docker run --rm --runtime runc --network none \
  -e PRESET=glm53-spark-tp2 "$IMAGE" --print-config
```

The output includes native vLLM arguments, relevant environment, the source of
each setting and the cache-service plan. Secrets are redacted. Explicit native
arguments override environment aliases, which override preset/model defaults.

For an advanced setting, use a native argument after the image or `-e NAME=value`
before it. Keep the image entrypoint. Do not paste shell text into
`EXTRA_VLLM_ARGS`; the launcher forwards argument arrays without evaluating a shell.
The [complete parameter reference](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/runtime/generated/parameters.md)
lists supported aliases. Profiles and deployment presets live in the same
[runtime directory](https://github.com/local-inference-lab/blackwell-llm-docker/tree/main/runtime).

### Memory tuning

Keep the model or preset's allocator and NCCL settings unless you need to
trade throughput for capacity. Spark TP2's memory reductions are not universal
TP4 defaults: smaller NCCL channel counts reduce DFlash2 C8 throughput, and
smaller allocator segments reduce DeepSeek V4.1 C1 throughput in the
[matched TP4 tests](../benchmarks/tp4-memory-controls.md).

The most consistent isolated saving in that comparison is the cuBLAS workspace
limit. It is **opt-in**, not a shared default. To try it, add
`-e CUBLAS_WORKSPACE_CONFIG=:4096:1` before `"$IMAGE"` in a launch command.
Five automatic-KV TP4 profiles made roughly 205–246 MiB more KV memory
available on rank 0; Qwen's fixed-budget profile instead used less device memory.
This does not establish the same result at TP1/TP2 or with combined overrides.
Removing that override restores the image/preset's workspace policy.

## Performance evidence

The [Karmic Kraken model table](../benchmarks/karmic-kraken-serving.md) records
decode, prefill, clocks, image versions and cache checks. Compare rows with the
same hardware, mode and sampling settings. Output speed includes speculative
acceptance and is not a kernel-only timing.

The [recipe archive](../archive/serving-guides/README.md) and
[JJ deployment archive](unified-vllm-docker-jj-archive.md) preserve preceding
instructions and their measurements. Canonical model URLs continue to show the
recommended beta recipe, so users do not have to choose among competing guides.
