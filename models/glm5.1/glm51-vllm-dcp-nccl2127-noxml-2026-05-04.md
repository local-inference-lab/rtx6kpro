# GLM-5.1 vLLM DCP with NCCL PR #2127 no-XML topology fix

Date: 2026-05-04

This page records the vLLM GLM-5.1 DCP test state that uses a patched NCCL build from NVIDIA NCCL PR #2127. The purpose was to verify whether the custom NCCL topology fix can replace the hand-written NCCL graph XML while preserving DCP throughput.

All measurements on this page use local 8-GPU RTX PRO 6000 Blackwell PCIe unless explicitly marked as an external comparison. Remote host `10.229.14.14` was not used for these results.

## Docker image

The same image was pushed under two tags:

| Tag | DockerHub digest |
|---|---|
| `voipmonitor/vllm:glm51-dcp-nccl2127-noxml-b12x0111-20260504` | `sha256:bd10fbacbe2b4cb7720f62d8c33791521f5d4e9995b867fd311e66f23cc48479` |
| `voipmonitor/vllm:glm51-mtp-b12xsparse-ficutlass-dcpfix-nccl2127-b12x0111-cg256-20260504` | `sha256:bd10fbacbe2b4cb7720f62d8c33791521f5d4e9995b867fd311e66f23cc48479` |

Image provenance:

| Component | Value |
|---|---|
| Base image | `voipmonitor/vllm:glm51-mtp-b12xsparse-ficutlass-dcpfix-b12x0111-cg256-20260504` |
| Patched NCCL PR | `https://github.com/NVIDIA/nccl/pull/2127` |
| Patched NCCL source commit | `6b72eea218cc5bc6f1632dc0fd09000237bdb98b` |
| Patched NCCL library in image | `/opt/libnccl-pr2127.so.2.30.3` |
| vLLM PR #41654 | `https://github.com/vllm-project/vllm/pull/41654`, commit `2fd929ab3d15000f72d7bd980394dc76cb70841d` |
| `ncclGetVersion()` | `23003` |
| NCCL version string | `NCCL version 2.30.3 compiled with CUDA 13.2` |
| NCCL library SHA256 | `4fc3b875b06c70921dbdb8a9aa5c88d99ddeadd09cefe1aea0827374bd76dca3` |

Runtime package versions verified inside the pushed image:

| Package | Version |
|---|---|
| `vllm` | `0.0.0+local` |
| `b12x` | `0.11.1` |
| `torch` | `2.11.0+cu130` |
| `flashinfer-python` | `0.6.8` |
| `transformers` | `5.3.0` |
| `fastsafetensors` | `0.2.2` |

Important effective image settings:

| Setting | Value |
|---|---|
| `LD_PRELOAD` | `/opt/libnccl-pr2127.so.2.30.3` |
| `VLLM_NCCL_SO_PATH` | `/opt/libnccl-pr2127.so.2.30.3` |
| vLLM sparse attention backend | `B12X_MLA_SPARSE` |
| MoE backend used in the final DCP benchmarks | `flashinfer_cutlass` |
| Speculative method | GLM-5.1 MTP, `num_speculative_tokens=3` |
| MTP acceptance thresholds | `VLLM_SPEC_ACCEPT_THRESHOLD_ACC=1.0`, `VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE=1.0` |

## Launch command

Use this command for every DCP variant. Change only `DCP`, `AR`, `PORT`, `GPU_MEMORY_UTILIZATION`, or `MAX_NUM_SEQS` as needed.

`AR=0` means `VLLM_ENABLE_PCIE_ALLREDUCE=0`. `AR=1` means `VLLM_ENABLE_PCIE_ALLREDUCE=1`.

The command intentionally unsets `NCCL_GRAPH_FILE` inside the container. This is the no-XML path enabled by the patched NCCL library.

```bash
export DCP=1
export AR=0
export PORT=5261
export GPU_MEMORY_UTILIZATION=0.76
export MAX_NUM_SEQS=32
export MAX_CUDAGRAPH_CAPTURE_SIZE=32
export CACHE_ROOT="$HOME/.cache/vllm-glm51-dcp-nccl2127"

mkdir -p \
  "$HOME/.cache/huggingface" \
  "$CACHE_ROOT/jit" \
  "$CACHE_ROOT/cutlass_dsl" \
  "$CACHE_ROOT/torchinductor" \
  "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/vllm"

docker rm -f "vllm-glm51-dcp${DCP}-ar${AR}-nccl2127" 2>/dev/null || true

docker run -d --name "vllm-glm51-dcp${DCP}-ar${AR}-nccl2127" \
  --gpus all \
  --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --entrypoint /bin/bash \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$CACHE_ROOT/jit:/cache/jit" \
  -v "$CACHE_ROOT/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "$CACHE_ROOT/torchinductor:/root/.cache/torchinductor" \
  -v "$CACHE_ROOT/triton:/root/.cache/triton" \
  -v "$CACHE_ROOT/vllm:/root/.cache/vllm" \
  -e DCP="${DCP}" \
  -e AR="${AR}" \
  -e PORT="${PORT}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl \
  -e CUDA_CACHE_PATH=/cache/jit \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e HF_HUB_OFFLINE=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e VLLM_MTP_PROB_DIAG=0 \
  -e VLLM_B12X_MLA_SPEC_SERIAL_DECODE=0 \
  -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 \
  -e VLLM_MTP_DRAFT_PROBS=0 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_ACC=1.0 \
  -e VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE=1.0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="${AR}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  voipmonitor/vllm:glm51-dcp-nccl2127-noxml-b12x0111-20260504 \
  -lc 'set -euo pipefail
      unset NCCL_GRAPH_FILE
      echo "LD_PRELOAD=${LD_PRELOAD:-}"
      echo "VLLM_NCCL_SO_PATH=${VLLM_NCCL_SO_PATH:-}"
      echo "NCCL_GRAPH_FILE=${NCCL_GRAPH_FILE:-<unset>}"
      echo "DCP=${DCP} AR=${AR}"
      exec /opt/venv/bin/vllm serve lukealonso/GLM-5.1-NVFP4-MTP \
        --host 0.0.0.0 --port "${PORT}" \
        --served-model-name GLM-5 \
        --trust-remote-code \
        --tensor-parallel-size 8 \
        --decode-context-parallel-size "${DCP}" \
        --dcp-comm-backend ag_rs \
        --dcp-kv-cache-interleave-size 1 \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len 65536 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
        --kv-cache-dtype fp8 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --reasoning-parser glm45 \
        --tool-call-parser glm47 \
        --quantization modelopt_fp4 \
        --attention-backend B12X_MLA_SPARSE \
        --moe-backend flashinfer_cutlass \
        --hf-overrides '"'"'{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'"'"' \
        --speculative-config '"'"'{"model":"lukealonso/GLM-5.1-NVFP4-MTP","method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"probabilistic","moe_backend":"flashinfer_cutlass","use_local_argmax_reduction":true}'"'"''
```

Recommended variants from the measurements below:

| Goal | `DCP` | `AR` | Notes |
|---|---:|---:|---|
| Fastest measured cc32 decode | `1` | `1` | `683.2 tok/s`; `AR=0` was almost identical at `677.1 tok/s`. |
| DCP>1 without XML | `4` | `0` | Best DCP>1 cc32 run in this matrix: `562.2 tok/s`. |
| DCP8 noXML | `8` | `0` | Patched NCCL noXML roughly matches official NCCL+XML, but DCP8 is still much slower than DCP1. |
| Avoid | `8` | `1` | PCIe allreduce hurt DCP8 in both patched noXML and official XML comparisons. |

## Benchmark command

The primary decode measurements used this benchmark shape:

```bash
python3 /mnt/llm_decode_bench_temp0.py \
  --port 5261 \
  --model GLM-5 \
  --skip-prefill \
  --concurrency 32 \
  --contexts 0 \
  --duration 30 \
  --max-tokens 1024 \
  --output /tmp/dcp1-nccl2127-noxml-ar0-cc1cc32-cc32.json
```

For the single-request rows, only `--concurrency 1` and output path changed.

## Primary measurements on patched NCCL noXML Docker

These rows were measured on `voipmonitor/vllm:glm51-dcp-nccl2127-noxml-b12x0111-20260504` with `NCCL_GRAPH_FILE` unset and patched NCCL loaded from `/opt/libnccl-pr2127.so.2.30.3`.

| DCP | AR | Concurrency | tok/s | Avg TTFT | Completed/errors | Raw result file |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 1 | `94.3` | `0.13s` | `1/0` | `/tmp/dcp1-nccl2127-noxml-ar0-cc1cc32-cc1.json` |
| 1 | 0 | 32 | `677.1` | `0.98s` | `32/0` | `/tmp/dcp1-nccl2127-noxml-ar0-cc1cc32-cc32.json` |
| 1 | 1 | 1 | `95.5` | `0.14s` | `1/0` | `/tmp/dcp1-nccl2127-noxml-ar1-cc1cc32-cc1.json` |
| 1 | 1 | 32 | `683.2` | `0.97s` | `32/0` | `/tmp/dcp1-nccl2127-noxml-ar1-cc1cc32-cc32.json` |
| 2 | 0 | 1 | `71.6` | `0.18s` | `1/0` | `/tmp/dcp2-nccl2127-noxml-ar0-cc1cc32-cc1.json` |
| 2 | 0 | 32 | `535.4` | `0.52s` | `32/0` | `/tmp/dcp2-nccl2127-noxml-ar0-cc1cc32-cc32.json` |
| 2 | 1 | 1 | `71.5` | `0.16s` | `1/0` | `/tmp/dcp2-nccl2127-noxml-ar1-cc1cc32-cc1.json` |
| 2 | 1 | 32 | `510.5` | `1.35s` | `32/0` | `/tmp/dcp2-nccl2127-noxml-ar1-cc1cc32-cc32.json` |
| 4 | 0 | 1 | `69.6` | `0.17s` | `1/0` | `/tmp/dcp4-nccl2127-noxml-ar0-cc1cc32-cc1.json` |
| 4 | 0 | 32 | `562.2` | `1.23s` | `32/0` | `/tmp/dcp4-nccl2127-noxml-ar0-cc1cc32-cc32.json` |
| 4 | 1 | 1 | `68.6` | `0.19s` | `1/0` | `/tmp/dcp4-nccl2127-noxml-ar1-cc1cc32-cc1.json` |
| 4 | 1 | 32 | `558.3` | `1.27s` | `32/0` | `/tmp/dcp4-nccl2127-noxml-ar1-cc1cc32-cc32.json` |
| 8 | 0 | 32 | `305.3` | `1.44s` | `32/0` | `/tmp/dcp8-nccl2127-noxml-ar0_cc32.json` |
| 8 | 1 | 32 | `245.3` | `1.11s` | `32/0` | `/tmp/dcp8-nccl2127-noxml-ar1_cc32.json` |
| 8 | 0 | 32 repeat | `357.4` | `1.94s` | `32/0` | `/tmp/dcp8-nccl2127-noxml-ar0-repeat_cc32.json` |
| 8 | 1 | 32 repeat | `335.9` | `1.55s` | `32/0` | `/tmp/dcp8-nccl2127-noxml-ar1-repeat_cc32.json` |

Interpretation:

- Patched NCCL noXML works for DCP1, DCP2, DCP4, and DCP8.
- `AR=1` is neutral/slightly positive for DCP1, negative for DCP2, neutral/slightly negative for DCP4, and negative for DCP8.
- DCP2/DCP4/DCP8 reduce single-request throughput compared with DCP1. That is expected from extra decode-context-parallel communication overhead; `AR=1` does not remove it.
- DCP8 noXML has variability between the short run and repeat. The stable conclusion is directionally clear: DCP8 remains much slower than DCP1, and `AR=1` is not the fix.

## DCP8 topology comparison

This comparison exists only to answer whether patched NCCL replaces the XML. The official NCCL+XML rows were not run from the final patched Docker image.

| NCCL stack | XML | AR | Concurrency | tok/s | Avg TTFT | Completed/errors | Raw result file |
|---|---|---:|---:|---:|---:|---:|---|
| Official NCCL | yes | 0 | 32 | `310.8` | `1.21s` | `32/0` | `/tmp/dcp8-official-xml-ar0_cc32.json` |
| Official NCCL | yes | 1 | 32 | `250.5` | `1.11s` | `32/0` | `/tmp/dcp8-official-xml-ar1_cc32.json` |
| Patched NCCL PR #2127 | no | 0 | 32 | `305.3` | `1.44s` | `32/0` | `/tmp/dcp8-nccl2127-noxml-ar0_cc32.json` |
| Patched NCCL PR #2127 | no | 1 | 32 | `245.3` | `1.11s` | `32/0` | `/tmp/dcp8-nccl2127-noxml-ar1_cc32.json` |

Conclusion from this table: patched NCCL without XML is effectively on par with official NCCL plus the hand-written XML for DCP8 in this workload. The practical benefit is removing the XML dependency. The patch does not by itself make DCP8 faster than DCP1.

## Earlier DCP merge baseline

These rows were measured before adding the patched NCCL library. They are included so the DCP impact can be separated from the NCCL PR #2127 impact.

| Stack | DCP | AR | XML | Concurrency | tok/s | Avg TTFT | Completed/errors | Raw result file |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Fast DCP1 reference image | 1 | 0 | no | 32 | `704.1` | `1.59s` | `32/0` | `/tmp/dcpmerge_baseline_dcp1_cc32.json` |
| DCP merge image before patched NCCL | 1 | 0 | no | 32 | `678.4` | `3.12s` | `32/0` | `/tmp/dcpmerge_new_dcp1_cc32.json` |
| DCP merge image before patched NCCL | 2 | 0 | no | 32 | `525.1` | `0.94s` | `32/0` | `/tmp/dcpmerge_new_dcp2_cc32_unsetxml.json` |
| DCP merge image before patched NCCL | 4 | 0 | no | 32 | `554.5` | `0.98s` | `32/0` | `/tmp/dcpmerge_new_dcp4_cc32_unsetxml.json` |
| DCP merge image before patched NCCL | 8 | 0 | no | 32 | `257.8` | `1.12s` | `32/0` | `/tmp/dcpmerge_new_dcp8_cc32_unsetxml.json` |
| DCP merge image before patched NCCL, b12x MoE | 8 | 0 | no | 32 | `186.3` | `2.48s` | `32/0` | `/tmp/dcpmerge_new_dcp8_b12xmoe_cc32.json` |

## Current conclusions

- The pushed NCCL PR #2127 image is usable as the no-XML DCP test base.
- The main verified value of the patched NCCL is topology robustness: it removes the need for `/mnt/nccl_graph_opt.xml` while matching official NCCL+XML DCP8 throughput in the measured workload.
- For this GLM-5.1 vLLM stack, DCP1 is still the fastest measured decode mode at cc32.
- For DCP>1 on this patched noXML image, use `VLLM_ENABLE_PCIE_ALLREDUCE=0` unless a new benchmark proves otherwise.
- FlashInfer CUTLASS MoE remains materially faster than b12x MoE in the DCP8 test path measured here.

## Open follow-up

The remaining performance question is not the XML anymore. The next useful work is to profile why DCP communication overhead dominates DCP8 decode and whether the `ag_rs` path, PCIe allreduce path, or MoE/attention scheduling can be fused or reduced enough to make DCP useful without losing too much single-request and cc32 throughput.
