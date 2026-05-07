# vLLM communicator experiments for GLM-5.1 and Kimi-K2.6

Date: 2026-05-07

This page records only the communicator work from the GLM/Kimi vLLM session:
which allreduce/DCP communication paths were tested, which patches were used,
what was fastest, what regressed, and what should be treated as experimental.

The goal is reproducibility. Numbers below are tied to concrete Docker images,
environment flags, and raw result files when available. PCIe rx/tx values are
the benchmark's coarse NVML live diagnostics in MB/s, not a per-kernel profiler
counter.

## Executive summary

Current stable recommendation for GLM-5.1 DCP=1 MTP:

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:glm51-kimi-comm-20260507` |
| NCCL | Patched NCCL PR #2127, `/opt/libnccl-pr2127.so.2.30.3` |
| XML | Not required; launch should `unset NCCL_GRAPH_FILE` |
| NCCL env | `NCCL_P2P_LEVEL=SYS`, `NCCL_PROTO=LL,LL128,Simple` |
| vLLM allreduce backend | `VLLM_PCIE_ALLREDUCE_BACKEND=cpp` |
| vLLM C++ cutoff | `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB` |
| RTX6K fused add | Disabled for GLM MTP: `VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0` |
| Reason | Fused allreduce+add without an end barrier corrupts GLM MTP hidden states; adding the barrier fixes correctness but loses cc32 throughput. |

Best stable GLM DCP=1 MTP result from the final no-fused-add run:

| Concurrency | tok/s runs | Acceptance runs | Raw files |
|---:|---:|---:|---|
| 1 | `84.3`, `93.5` | `0.4608`, `0.5000` | `/tmp/glm51_final_no_fusedadd_c1.json`, `/tmp/glm51_final_no_fusedadd_c1_r2.json` |
| 32 | `855.6`, `853.0` | `0.5443`, `0.6051` | `/tmp/glm51_final_no_fusedadd_c32.json`, `/tmp/glm51_final_no_fusedadd_c32_r2.json` |

These GLM numbers were measured with
`voipmonitor/vllm:glm51-kimi-main-pciearselect-20260505` plus bind-mounted
overlay files. `voipmonitor/vllm:glm51-kimi-comm-20260507` packages the same
overlay files into the image; its contents were verified by SHA256, but this
document does not claim a separate full model rerun after packaging.

Best stable Kimi-K2.6 v2 result from the published Kimi v2 matrix:

| Profile | C1 ctx0 | C32 ctx0 | Source |
|---|---:|---:|---|
| DCP=1, MTP=3 | `117.1 tok/s` | `1149.7 tok/s` | `models/kimi-k26-v2.md` |
| DCP=1, no MTP | `86.9 tok/s` | `857.0 tok/s` | `models/kimi-k26-v2.md` |

Important negative result:

- The RTX6K fused allreduce+add/RMS prototypes are not the default for GLM MTP.
- They are useful research code and passed microbench correctness in isolation,
  but the no-end-barrier fused path corrupted GLM MTP under the real vLLM CUDA
  graph/fusion path.
- For now, the safe production path is patched NCCL + vLLM C++ selector, not
  the fused RTX6K add path.

## Hardware and common runtime

Unless explicitly stated otherwise, measurements were local 8x RTX PRO 6000
Blackwell PCIe.

Common runtime choices:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
OMP_NUM_THREADS=16
CUTE_DSL_ARCH=sm_120a
CUDA_DEVICE_MAX_CONNECTIONS=32
NCCL_P2P_LEVEL=SYS
NCCL_IB_DISABLE=1
SAFETENSORS_FAST_GPU=1
```

Important cache mounts used in long-running tests:

```bash
XDG_CACHE_HOME=/cache/jit
CUDA_CACHE_PATH=/cache/jit
VLLM_CACHE_DIR=/cache/jit/vllm
TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi
FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer
VLLM_CACHE_ROOT=/root/.cache/vllm
TRITON_CACHE_DIR=/root/.cache/triton
TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions
CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl
```

## Docker images involved

| Image | Role |
|---|---|
| `voipmonitor/vllm:glm51-main-20260505` | Main GLM/Kimi vLLM rebased image. Labels show vLLM upstream commit `844df542694089045589ceaad52cba69ab58527a`, FlashInfer `0.6.10rc1`, NCCL PR #2127 library, and vLLM PR #41654. |
| `voipmonitor/vllm:glm51-kimi-main-pciearselect-20260505` | Current shared GLM/Kimi image with the PCIe allreduce selector work. Docker labels include NCCL PR #2127 and FlashInfer `0.6.10rc1`; `vllm.upstream.commit` label is `unknown` on this tag but it derives from the same main-port line. |
| `voipmonitor/vllm:glm51-kimi-comm-20260507` | Follow-up packaged communicator image. It bakes the host overlay files for `custom_all_reduce.py`, `allreduce_rms_fusion.py`, and `/opt/rtx6k-pcie-fused-allreduce` into the image so GLM/Kimi can be launched without bind-mounting the local experiment tree. Digest: `sha256:ce62d09208adc31bf89a431376943ece7201df996edc4dd155684325f16eb9d4`. |
| `voipmonitor/vllm:glm51-dcp-nccl2127-noxml-b12x0111-20260504` | Earlier GLM DCP image used to validate patched NCCL no-XML behavior for DCP1/2/4/8. |
| `voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424` | Older Kimi reference image from the first Kimi page; used as a historical comparison point. |

The current images preload patched NCCL with:

```bash
VLLM_NCCL_SO_PATH=/opt/libnccl-pr2127.so.2.30.3
LD_PRELOAD=/opt/libnccl-pr2127.so.2.30.3
```

Docker labels verified on the current GLM/Kimi images:

| Label | Value |
|---|---|
| `nccl.pr` | `https://github.com/NVIDIA/nccl/pull/2127` |
| `nccl.commit` | `6b72eea218cc5bc6f1632dc0fd09000237bdb98b` |
| `vllm.pr41654` | `https://github.com/vllm-project/vllm/pull/41654` |
| `vllm.pr41654.commit` | `2fd929ab3d15000f72d7bd980394dc76cb70841d` |
| `flashinfer.version` | `0.6.10rc1` on the current main/pciearselect images |

Additional labels on `voipmonitor/vllm:glm51-kimi-comm-20260507`:

| Label | Value |
|---|---|
| `voipmonitor.communicator.base` | `voipmonitor/vllm:glm51-kimi-main-pciearselect-20260505` |
| `voipmonitor.communicator.default` | `patched-nccl-pr2127+cpp-selector+fusedadd-off` |
| `voipmonitor.communicator.custom_all_reduce_sha256` | `7154f2ce49b5bdb1e2028219d39f189eaf1d35fc520344b1376f593736f6c23e` |
| `voipmonitor.communicator.allreduce_rms_fusion_sha256` | `44e8dbb86da1a68effe2577e4200933169b3958c127c58a3d276bb3b1cb60d7f` |
| `voipmonitor.communicator.rtx6k_pcie_allreduce_cu_sha256` | `1126e00c8d401c14138fcfb9247e5ab6c726f59443bc5416816bd817ae9bfab9` |
| `voipmonitor.communicator.rtx6k_pcie_allreduce_py_sha256` | `76cce6b2ca1adedd8d8ba875a7db911943278272d0c9f982b816c9102f3b9ed2` |

The packaged image also includes helper launchers:

```text
/usr/local/bin/run-glm51-vllm
/usr/local/bin/run-kimi26-vllm
```

These launchers unset `NCCL_GRAPH_FILE` by default, unless `USE_NCCL_XML=1` is
explicitly set.

## Communicator types tested

### 1. Official NCCL plus hand-written XML

This was the older DCP topology workaround. It used:

```bash
NCCL_P2P_LEVEL=SYS
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
```

It was useful for DCP8 before the NCCL PR #2127 image existed. It is no longer
the preferred path because patched NCCL PR #2127 matched the DCP8 XML result
without requiring the external XML.

Measured DCP8 comparison from `glm51-vllm-dcp-nccl2127-noxml-2026-05-04.md`:

| Stack | XML | AR | C32 tok/s | Raw file |
|---|---|---:|---:|---|
| Official NCCL | yes | 0 | `310.8` | `/tmp/dcp8-official-xml-ar0_cc32.json` |
| Official NCCL | yes | 1 | `250.5` | `/tmp/dcp8-official-xml-ar1_cc32.json` |
| Patched NCCL PR #2127 | no | 0 | `305.3` | `/tmp/dcp8-nccl2127-noxml-ar0_cc32.json` |
| Patched NCCL PR #2127 | no | 1 | `245.3` | `/tmp/dcp8-nccl2127-noxml-ar1_cc32.json` |

Conclusion: patched NCCL no-XML is a practical replacement for the XML in this
workload. It does not by itself make DCP8 fast.

### 2. Patched NCCL PR #2127, no XML

This is the current base. The patch is loaded by `LD_PRELOAD` and vLLM's
`VLLM_NCCL_SO_PATH`; launch should unset `NCCL_GRAPH_FILE` when validating the
no-XML path.

What it fixed operationally:

- The hand-written topology XML is no longer needed for local 8x RTX6K PCIe.
- DCP8 no-XML throughput is on par with official NCCL plus XML.
- It gives us one Docker image that can run on machines without copying the XML.

What it did not fix:

- DCP2/DCP4/DCP8 still carry decode-context-parallel communication overhead.
- DCP1 is still the fastest GLM decode mode in the measured cc32 tests.

GLM no-XML DCP matrix from the 2026-05-04 image:

| DCP | `VLLM_ENABLE_PCIE_ALLREDUCE` | C1 tok/s | C32 tok/s | Raw files |
|---:|---:|---:|---:|---|
| 1 | 0 | `94.3` | `677.1` | `/tmp/dcp1-nccl2127-noxml-ar0-cc1cc32-cc1.json`, `/tmp/dcp1-nccl2127-noxml-ar0-cc1cc32-cc32.json` |
| 1 | 1 | `95.5` | `683.2` | `/tmp/dcp1-nccl2127-noxml-ar1-cc1cc32-cc1.json`, `/tmp/dcp1-nccl2127-noxml-ar1-cc1cc32-cc32.json` |
| 2 | 0 | `71.6` | `535.4` | `/tmp/dcp2-nccl2127-noxml-ar0-cc1cc32-cc1.json`, `/tmp/dcp2-nccl2127-noxml-ar0-cc1cc32-cc32.json` |
| 4 | 0 | `69.6` | `562.2` | `/tmp/dcp4-nccl2127-noxml-ar0-cc1cc32-cc1.json`, `/tmp/dcp4-nccl2127-noxml-ar0-cc1cc32-cc32.json` |
| 8 | 0 | not recorded in that table | `357.4` repeat | `/tmp/dcp8-nccl2127-noxml-ar0-repeat_cc32.json` |

### 3. vLLM C++ custom allreduce selector

Patch concept:

- Add `VLLM_PCIE_ALLREDUCE_BACKEND=cpp` to force the vLLM C++ custom allreduce
  path for this PCIe platform instead of b12x oneshot.
- Add `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF`, usually `56KB`.
- In `custom_all_reduce.py`, `should_custom_ar()` returns false when tensor
  byte size is above the cutoff, so large messages fall back to NCCL.

Relevant local patched source path:

```text
/root/benchmarks/vllm-dcp1-stage2-selector-20260506/vllm/distributed/device_communicators/custom_all_reduce.py
```

Relevant code locations in that file:

| Area | Lines from local file |
|---|---|
| Backend selector | `_get_pcie_allreduce_backend()` around lines 37-44 |
| b12x runtime loader | `_load_b12x_pcie_oneshot_runtime()` around lines 83-89 |
| cpp cutoff env | `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF` around lines 475-477 |
| large-message NCCL fallback | `inp_size > self._cpp_ar_cutoff_size: return False` around lines 668-675 |
| fused-add capability hooks | `VLLM_RTX6K_FUSED_ALLREDUCE_ADD*` around lines 487-551 |

Kimi C32 MTP communication sweep:

| Variant | C32 tok/s | PCIe rx/tx MB/s | Raw log |
|---|---:|---:|---|
| cpp default / force stage2 | `1046.4` | `~75,821 / 74,584` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_mtp3_cpp_default_c32_ctx0_run1.log` |
| force 1-stage | `643.1` | `~168,815 / 168,214` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_mtp3_cpp_force1stage_c32_ctx0_run1.log` |
| pure NCCL default protocols | `1003.5`, `998.6` | `~120,388 / 119,926`, `~121,296 / 120,269` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_mtp3_cpp_cutoff0_purenccl_c32_ctx0_run1.log`, run2 |
| pure NCCL with `NCCL_PROTO=LL,LL128,Simple` | `1127.6` | `~76,175 / 76,381` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_mtp3_cpp_purenccl_proto_all_c32_ctx0_run1.log` |
| pure NCCL with `NCCL_PROTO=LL128` | `1131.6` | `~76,508 / 75,485` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_mtp3_cpp_purenccl_proto_ll128_c32_ctx0_run1.log` |
| pure NCCL with `NCCL_PROTO=LL` | `998.1` | `~123,263 / 123,670` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_mtp3_cpp_purenccl_proto_ll_c32_ctx0_run1.log` |
| pure NCCL with `NCCL_PROTO=Simple` | `1108.9` | `~69,709 / 69,399` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_mtp3_cpp_purenccl_proto_simple_c32_ctx0_run1.log` |

Interpretation:

- For Kimi MTP C32, forcing only one-stage custom allreduce was clearly bad.
- NCCL protocol selection matters. `LL` alone was high traffic and slower.
- `LL128` and `Simple` were the useful protocols in this run.
- The current stable launch keeps `NCCL_PROTO=LL,LL128,Simple` so NCCL can pick
  the better protocol while preserving fallback options.

Kimi no-MTP C1/C32 selected rows:

| Variant | C1 tok/s | C32 tok/s | PCIe rx/tx at C32 | Raw files |
|---|---:|---:|---:|---|
| stage1<=56KB else NCCL | `87.5` | `945.7` | `~61,078 / 60,339` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_nomtp_stage1_56k_nccl_cc1_ctx0.json`, `..._cc32_run1_cc32_ctx0.json` |
| pure NCCL, no custom AR | `80.1` | not in this row | not measured in same row | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_nomtp_fresh_purenccl_20260506_cc1_ctx0.json` |
| FlashInfer AR fusion | not measured in same row | `952.5` | `~61,337 / 60,815` | `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_nomtp_flashinfer_ar_fusion_cc32_20260506_cc32_ctx0.json` |

Conclusion: vLLM C++ selector with small-message custom AR and NCCL fallback is
the best safe general-purpose shape tested for Kimi no-MTP. For Kimi MTP C32,
NCCL protocol tuning had the biggest gain.

### 4. b12x PCIe oneshot allreduce

This path came from b12x. In vLLM it was selected by the older default:

```bash
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
```

The vLLM port can load `b12x.distributed.PCIeOneshotAllReduce`. It also has
autotune/crossover logic in the local patched `custom_all_reduce.py`.

What worked:

- Kimi no-MTP DCP1 single batch was competitive:
  - `86.6 tok/s` from `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_nomtp_b12x_default_run1_cc1_ctx0.json`.
  - `86.8 tok/s` from `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_nomtp_b12x_56k_run1_cc1_ctx0.json`.
- Kimi no-MTP C32 with split barrier was also competitive:
  - `944.8 tok/s`, PCIe `~61,089 / 60,496 MB/s` from `/root/bench-results/vllm-cpp-ar-20260505/kimi_dcp1_nomtp_b12x_splitbarrier_56k_cc32_run1_cc32_ctx0.json`.

What did not work well enough for the final GLM path:

- The b12x/no-end-barrier assumptions did not transfer safely into every vLLM
  graph/fusion path.
- SGLang can avoid some barriers via double-buffering. In vLLM we saw real
  hidden-state corruption when a no-end-barrier fused path was used under GLM
  MTP. That made correctness, not only speed, the deciding issue.
- For DCP>1, `VLLM_ENABLE_PCIE_ALLREDUCE=1` was neutral or worse in the GLM
  DCP matrix. DCP8 AR=1 was slower than AR=0 in both XML and no-XML comparisons.

### 5. RTX6K fused allreduce+add prototype

Prototype paths:

```text
/root/benchmarks/rtx6k-pcie-fused-allreduce-20260506-134604/pcie_allreduce.cu
/root/benchmarks/vllm-dcp1-stage2-selector-20260506/rtx6k-pcie-fused-allreduce/pcie_allreduce.cu
```

Prototype README:

```text
/root/benchmarks/rtx6k-pcie-fused-allreduce-20260506-134604/README.md
```

Implemented CUDA extension operations:

| Operation | Purpose |
|---|---|
| `all_reduce` | Existing one-stage PCIe allreduce with rotated peer reads and graph IPC registration. |
| `all_reduce_add` | Fused one-stage allreduce plus local residual add epilog. |
| `all_reduce_hier_add` | Experimental 4+4 hierarchical fused allreduce plus add. |
| fused add RMS hooks | Experimental path to fuse allreduce, residual add, and RMSNorm. |

Important env flags:

```bash
VLLM_RTX6K_PCIE_FUSED_ALLREDUCE_PATH=/opt/rtx6k-pcie-fused-allreduce
VLLM_RTX6K_FUSED_ALLREDUCE_ADD=1
VLLM_RTX6K_FUSED_ALLREDUCE_ADD_1STAGE_MAX_SIZE=56KB
VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=1
VLLM_RTX6K_FUSED_ALLREDUCE_ADD_STAGE2=1
VLLM_RTX6K_FUSED_ALLREDUCE_ADD_RMS=1
VLLM_RTX6K_NCCL_RESIDUAL_ADD=1
VLLM_RTX6K_NCCL_PREADD_FUSED_RMS=1
```

Standalone microbench observations from the prototype README:

| Shape | Result |
|---|---|
| 14,336 B graph replay | fused one-stage `~11.8 us` vs NCCL+add `~56.2 us` |
| 172,032 B graph replay | fused one-stage `~59.4 us` vs NCCL+add `~65.7 us` |
| 458,752 B cc32-equivalent | fused one-stage `~147.3 us`, no-end hierarchy `~133.7 us`, NCCL+add `~93.0 us` |

Interpretation:

- Fused one-stage is attractive for small decode messages.
- It does not beat NCCL for cc32-size messages because read-all one-stage
  traffic scales poorly.
- A custom communicator that beats NCCL at large messages needs a real
  ring/reduce-scatter/allgather-like protocol or deeper fusion into producers
  and consumers.

### 6. vLLM allreduce+RMS fusion and RTX6K fused add

Local patched source:

```text
/root/benchmarks/vllm-dcp1-stage2-selector-20260506/vllm/compilation/passes/fusion/allreduce_rms_fusion.py
```

Patch concept:

- vLLM already has compiler passes that can fuse `allreduce -> residual add ->
  rmsnorm`.
- We added RTX6K-specific branches controlled by:

```bash
VLLM_RTX6K_FUSED_ALLREDUCE_ADD=1
VLLM_RTX6K_FUSED_ALLREDUCE_ADD_RMS=1
VLLM_RTX6K_NCCL_RESIDUAL_ADD=1
VLLM_RTX6K_NCCL_PREADD_FUSED_RMS=1
```

Relevant local code locations:

| Area | Lines from local file |
|---|---|
| env gates | `_RTX6K_USE_FUSED_ADD_RMS`, `_RTX6K_USE_FUSED_ADD`, `_RTX6K_USE_NCCL_RESIDUAL_ADD` around lines 260-266 |
| fused add call | `ca_comm.fused_all_reduce_add(allreduce_in, residual)` around lines 399-433 |

GLM MTP correctness bisect:

| Variant | C1 tok/s | C32 tok/s | Acceptance | Outcome | Raw files |
|---|---:|---:|---:|---|---|
| Image only, no overlay | `86.2` | not measured | `0.6061` | OK | `/tmp/glm51_regress_pcie_image_only_c1.json` |
| cpp/no-XML no overlay | `88.2` | not measured | `0.4608` | OK | `/tmp/glm51_regress_pcie_cpp_nooverlay_c1.json` |
| selector overlay passive | `89.8` | not measured | `0.4510` | OK | `/tmp/glm51_regress_overlay_passive_c1.json` |
| fused add only, no RMS fusion | `89.9` | `~810.0` stable | `0.5784` C1 | Correct, but not fastest C32 | `/tmp/glm51_regress_fusedadd_only_c1.json`, `/tmp/glm51_final_candidate_fusedadd_no_rms_c32*.json` |
| RMS fusion without RTX6K fused add | `93.3` | not measured | `0.6566` | OK | `/tmp/glm51_regress_rmsfusion_no_fusedadd_c1.json` |
| fused add + RMS fusion, no end barrier | `38.1` | not used | `0.0588` | Broken hidden states / MTP corruption | `/tmp/glm51_regress_fusedadd_rmsfusion_c1.json` |
| fused add + RMS fusion, end barrier | `92.3` | `804.2` | `0.6970` C1, `0.4896` C32 | Correct but C32 slower | `/tmp/glm51_regress_fusedadd_rmsfusion_barrier_c1.json`, `/tmp/glm51_regress_fusedadd_rmsfusion_barrier_c32.json` |
| final no fused add | `84.3`, `93.5` | `855.6`, `853.0` | `0.4608`, `0.5000` C1; `0.5443`, `0.6051` C32 | Best stable GLM default | `/tmp/glm51_final_no_fusedadd_*.json` |

Root cause interpretation:

- GLM MTP C1 hidden shape `(4, 6144)` is bf16, i.e. `49,152` bytes. With a
  `56KB` limit it selects the small-message one-stage fused add path.
- The no-end-barrier path depends on safe double-buffering and reuse ordering.
  That was not guaranteed in the vLLM compiler-fusion/CUDA graph replay path.
- The failure mode was not a clean crash. It was silent hidden-state corruption:
  acceptance collapsed and output became bad.
- Adding the end barrier restored correctness but reduced high-concurrency
  throughput enough that it was not the final GLM recommendation.

Final policy:

```bash
VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0
```

The fused add path remains useful as an experiment, but it should be opt-in and
must be validated with MTP acceptance and content tests before release.

## DCP-specific findings

GLM DCP results from the 2026-05-04 no-XML DCP image:

| DCP | Best measured C32 in that matrix | Notes |
|---:|---:|---|
| 1 | `683.2 tok/s` with `AR=1`; `677.1` with `AR=0` | DCP1 fastest in that specific DCP image. |
| 2 | `535.4 tok/s` with `AR=0` | `AR=1` was slower at `510.5`. |
| 4 | `562.2 tok/s` with `AR=0` | Slightly better than DCP2 in that run. |
| 8 | `357.4 tok/s` repeat with `AR=0` | DCP8 gives more KV capacity but much lower short-context C32 decode. |

DCP conclusions:

- DCP increases KV capacity but adds attention-path communication.
- For GLM short-context throughput, DCP1 remained best.
- For DCP>1, `VLLM_ENABLE_PCIE_ALLREDUCE=1` was not a universal win and was
  often worse.
- Patched NCCL PR #2127 removes XML dependency but does not remove DCP overhead.

Kimi v2 DCP capacity and throughput from `models/kimi-k26-v2.md`:

| Profile | KV tokens | ctx0/C1 | ctx0/C32 |
|---|---:|---:|---:|
| DCP=1, MTP=3 | `442,192` | `117.1` | `1149.7` |
| DCP=1, no MTP | `491,200` | `86.9` | `857.0` |
| DCP=4, MTP=3 | `1,311,936` | `88.3` | `892.7` |
| DCP=4, no MTP | `1,964,992` | `73.9` | `724.5` |
| DCP=8, MTP=3 | `2,623,872` | `81.9` | `659.3` |
| DCP=8, no MTP | `3,929,600` | `72.7` | `616.7` |

Kimi conclusion:

- DCP4/DCP8 are useful when context/KV capacity matters.
- For ctx0 throughput, DCP1 is still fastest.
- Kimi did not expose the same GLM MTP corruption signature in the final
  no-fused-add path, but that does not make the RTX6K fused add path generally
  safe.

## NCCL protocol tuning

The largest Kimi MTP C32 communication improvement came from NCCL protocol
selection, not from a custom fused communicator.

Observed Kimi MTP C32 protocol results:

| `NCCL_PROTO` | C32 tok/s | PCIe rx/tx MB/s | Raw log |
|---|---:|---:|---|
| default in pure NCCL run | `1003.5`, `998.6` | `~120GB/s` each direction | `kimi_dcp1_mtp3_cpp_cutoff0_purenccl_c32_ctx0_run*.log` |
| `LL` | `998.1` | `~123GB/s` each direction | `kimi_dcp1_mtp3_cpp_purenccl_proto_ll_c32_ctx0_run1.log` |
| `LL128` | `1131.6` | `~76GB/s` each direction | `kimi_dcp1_mtp3_cpp_purenccl_proto_ll128_c32_ctx0_run1.log` |
| `Simple` | `1108.9` | `~70GB/s` each direction | `kimi_dcp1_mtp3_cpp_purenccl_proto_simple_c32_ctx0_run1.log` |
| `LL,LL128,Simple` | `1127.6` | `~76GB/s` each direction | `kimi_dcp1_mtp3_cpp_purenccl_proto_all_c32_ctx0_run1.log` |

Microbench note from `/root/bench-results/vllm-cpp-ar-20260505/pynccl_vs_torch_nccl_microbench.log`:

| Size | torch dist NCCL us | vLLM PyNCCL us | Winner |
|---:|---:|---:|---|
| 16KB | `20.88` | `19.84` | PyNCCL |
| 32KB | `28.66` | `21.81` | PyNCCL |
| 56KB | `25.07` | `24.49` | PyNCCL |
| 64KB | `21.55` | `22.12` | torch |
| 256KB | `39.46` | `39.45` | tie |
| 512KB | `61.28` | `59.50` | PyNCCL |

Interpretation:

- Microbench differences are small around the cutoff and do not fully predict
  end-to-end throughput.
- End-to-end decode is sensitive to protocol choice, CUDA graph scheduling,
  message shape mix, and whether MTP expands the effective decode batch.
- `NCCL_PROTO=LL,LL128,Simple` is the safest measured setting because it allows
  NCCL to avoid the bad LL-only behavior while retaining protocol choices.

## Launch snippets for reconstruction

### Final safe GLM DCP1 MTP communicator launch wrapper

This is the final safe no-fused-add variant used for the current GLM numbers:

```bash
VLLM_IMAGE=voipmonitor/vllm:glm51-kimi-comm-20260507 \
USE_NCCL_XML=0 \
DCP_SIZE=1 \
PORT=5261 \
MAX_MODEL_LEN=202752 \
ENABLE_RTX6K_FUSED_ADD=0 \
/root/benchmarks/glm5-vllm-20260507/run_glm_variant_used.sh final-no-fusedadd-test
```

The wrapper sets the important communicator flags:

```bash
NCCL_P2P_LEVEL=SYS
NCCL_PROTO=LL,LL128,Simple
VLLM_NCCL_SO_PATH=/opt/libnccl-pr2127.so.2.30.3
LD_PRELOAD=/opt/libnccl-pr2127.so.2.30.3
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=cpp
VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB
VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0
```

The same GLM server can also be started directly from the packaged image:

```bash
docker run -d --gpus all --ipc=host --network host --privileged \
  --name glm51-comm-dcp1 \
  --entrypoint /usr/local/bin/run-glm51-vllm \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm-glm51-comm/jit:/cache/jit \
  -v ~/.cache/vllm-glm51-comm/cutlass_dsl:/root/.cache/cutlass_dsl \
  -v ~/.cache/vllm-glm51-comm/triton:/root/.cache/triton \
  -v ~/.cache/vllm-glm51-comm/torchinductor:/root/.cache/torchinductor \
  -v ~/.cache/vllm-glm51-comm/vllm:/root/.cache/vllm \
  -e PORT=5261 \
  -e DCP_SIZE=1 \
  -e MAX_MODEL_LEN=202752 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  voipmonitor/vllm:glm51-kimi-comm-20260507
```

### Kimi v2 current communicator launch

Use the Kimi v2 page as the full command:

```text
models/kimi-k26-v2.md
```

Key communicator flags from that launch:

```bash
NCCL_P2P_LEVEL=SYS
NCCL_PROTO=LL,LL128,Simple
VLLM_NCCL_SO_PATH=/opt/libnccl-pr2127.so.2.30.3
LD_PRELOAD=/opt/libnccl-pr2127.so.2.30.3
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=cpp
unset NCCL_GRAPH_FILE
```

The packaged image has an equivalent helper:

```bash
docker run -d --gpus all --ipc=host --network host --privileged \
  --name kimi26-comm-dcp1 \
  --entrypoint /usr/local/bin/run-kimi26-vllm \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm-kimi26-comm/jit:/cache/jit \
  -v ~/.cache/vllm-kimi26-comm/cutlass_dsl:/root/.cache/cutlass_dsl \
  -v ~/.cache/vllm-kimi26-comm/triton:/root/.cache/triton \
  -v ~/.cache/vllm-kimi26-comm/torchinductor:/root/.cache/torchinductor \
  -v ~/.cache/vllm-kimi26-comm/vllm:/root/.cache/vllm \
  -e PORT=5002 \
  -e DCP_SIZE=1 \
  -e MAX_MODEL_LEN=262144 \
  voipmonitor/vllm:glm51-kimi-comm-20260507
```

## Patch inventory

### Patched NCCL PR #2127

Implemented in Docker images, not as a local vLLM source file:

| Item | Value |
|---|---|
| PR | `https://github.com/NVIDIA/nccl/pull/2127` |
| Commit | `6b72eea218cc5bc6f1632dc0fd09000237bdb98b` |
| Library | `/opt/libnccl-pr2127.so.2.30.3` |
| Activation | `LD_PRELOAD=/opt/libnccl-pr2127.so.2.30.3`, `VLLM_NCCL_SO_PATH=/opt/libnccl-pr2127.so.2.30.3` |
| Purpose | Make the local PCIe topology work without external `NCCL_GRAPH_FILE` XML. |

### vLLM PCIe backend selector

Local source path:

```text
/root/benchmarks/vllm-dcp1-stage2-selector-20260506/vllm/distributed/device_communicators/custom_all_reduce.py
```

Patch behavior:

- `VLLM_PCIE_ALLREDUCE_BACKEND=b12x|cpp` chooses b12x oneshot or vLLM C++ custom AR.
- `VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB` disables custom AR above the cutoff so
  NCCL handles larger messages.
- b12x runtime is loaded only if available from `b12x.distributed`.
- Cross-NUMA guard can be controlled by `VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA`.

### RTX6K fused allreduce extension

Local source paths:

```text
/root/benchmarks/vllm-dcp1-stage2-selector-20260506/rtx6k-pcie-fused-allreduce/pcie_allreduce.cu
/root/benchmarks/rtx6k-pcie-fused-allreduce-20260506-134604/pcie_allreduce.cu
```

Patch behavior:

- Adds fused `all_reduce_add`.
- Adds experimental hierarchical 4+4 fused add.
- Adds optional end barrier controlled by `VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER`.
- Provides hooks for fused add RMS experiments.

Status:

- Microbench useful.
- Not the final GLM default.
- Must be treated as unsafe without end-barrier validation in real vLLM MTP.

### vLLM allreduce/RMS fusion pass hooks

Local source path:

```text
/root/benchmarks/vllm-dcp1-stage2-selector-20260506/vllm/compilation/passes/fusion/allreduce_rms_fusion.py
```

Patch behavior:

- Adds RTX6K-specific branches to replace `allreduce -> residual add -> rmsnorm`
  with custom fused operations when env flags are set.
- Adds optional NCCL residual-add experiments.

Status:

- `RMS fusion without RTX6K fused add` was correct and fast enough.
- `RTX6K fused add + RMS fusion + no end barrier` corrupted GLM MTP.
- `RTX6K fused add + RMS fusion + end barrier` was correct but slower at C32.

### Local launcher guard patch

Local launchers were updated so `ENABLE_RTX6K_FUSED_ADD=auto` no longer enables
the unsafe path by default for GLM:

```text
/root/benchmarks/glm5-vllm-20260507/run_glm_variant_used.sh
/root/benchmarks/vllm-dcp1-stage2-selector-20260506/run_glm_variant.sh
```

Final safe launch explicitly uses:

```bash
ENABLE_RTX6K_FUSED_ADD=0
```

## What to avoid

| Avoid | Why |
|---|---|
| GLM MTP with `VLLM_RTX6K_FUSED_ALLREDUCE_ADD=1` and no end barrier | Silent hidden-state corruption; acceptance collapsed to about `0.0588` in `/tmp/glm51_regress_fusedadd_rmsfusion_c1.json`. |
| Force one-stage custom allreduce for large C32 messages | Kimi MTP C32 dropped to `643.1 tok/s` and PCIe traffic rose to `~168GB/s` each direction. |
| `NCCL_PROTO=LL` alone for Kimi MTP C32 | Slower and higher PCIe traffic than LL128/Simple. |
| Treat patched NCCL no-XML as a DCP speed fix | It removes XML dependency; it does not remove DCP communication overhead. |
| Enable `VLLM_ENABLE_PCIE_ALLREDUCE=1` blindly for DCP>1 | In GLM DCP2/DCP4/DCP8, AR=1 was neutral or slower in the recorded matrix. |

## Open work

- Build an end-to-end validated communicator that can safely beat NCCL for
  larger C32-size messages. The current one-stage algorithm cannot do that
  because traffic scales badly.
- If continuing the RTX6K fused add path, preserve the end-barrier correctness
  guard or prove double-buffer ownership in the exact vLLM CUDA graph/fusion
  path.
- Re-evaluate a real ring/reduce-scatter/allgather-style custom communicator
  if the goal is to outperform NCCL at large messages.
- For DCP>1, profile the attention-side DCP communication directly. The current
  allreduce changes do not explain all DCP overhead.

## Raw data index

Primary documentation:

| Path | Contents |
|---|---|
| `models/glm5.1/glm51-vllm-dcp-nccl2127-noxml-2026-05-04.md` | GLM DCP1/2/4/8 no-XML NCCL PR #2127 matrix. |
| `models/kimi-k26-v2.md` | Current Kimi v2 local 8-GPU matrix. |
| `models/glm5.1/glm51-sglang-vllm-cc32-state-2026-05-03.md` | Earlier SGLang/vLLM GLM cc32 state and MTP acceptance context. |

Local raw result directories:

| Path | Contents |
|---|---|
| `/root/bench-results/vllm-cpp-ar-20260505/` | Kimi communicator sweeps: cpp selector, b12x, NCCL protocol sweeps, fused experiments. |
| `/root/bench-results/vllm-dcp1-stage2-selector-20260506/` | Kimi no-MTP selector/fused-stage sweeps. |
| `/root/bench-results/kimi-k26-v2-20260507/` | Kimi v2 matrix results. |
| `/tmp/glm51_final_no_fusedadd_*.json` | Final safe GLM no-fused-add MTP checks. |
| `/tmp/glm51_regress_*.json` | GLM MTP corruption bisect for fused add/RMS variants. |
| `/tmp/dcp*-nccl2127-*.json` | GLM DCP no-XML NCCL PR #2127 measurements. |
