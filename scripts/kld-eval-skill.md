---
name: kld-eval
description: Run KLD (KL divergence) evaluation for quantized models. Measures quality loss vs FP8 reference using full vocabulary logit distributions on WikiText-2. Use when the user asks to measure KLD, compare model quality, or evaluate quantization.
user-invocable: true
allowed-tools: Bash, Read, Write, Grep, Glob, Agent
argument-hint: [action] [model-or-backend]
---

# KLD Evaluation Skill

Measures KL divergence between a quantized model and an FP8 reference using full vocabulary logit distributions captured during prefill on WikiText-2.

## Arguments

`$ARGUMENTS` can be:
- `ref` — create FP8 reference logits
- `test <model-or-backend>` — capture test model logits
- `compute` — compute KLD from existing logits
- `full` — run full pipeline (ref + test + compute)
- empty — show status of existing logit captures

## Docker Image

Use `voipmonitor/sglang:test-cu132` which has the KLD patch baked in. If using a different container, apply the patch first:

```bash
python /workspace/sglang-kld-logit-capture.py  # if baked into image
# OR copy and run from host:
docker cp patches/sglang-kld-logit-capture.py <container>:/tmp/
docker exec <container> python /tmp/sglang-kld-logit-capture.py
```

## Procedure

### Finding or Creating a Container

Look for a running container with the sglang image:
```bash
docker ps --format "{{.ID}} {{.Image}} {{.Names}}" | grep -i sglang
```

If none exists, start one:
```bash
docker run -d --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit \
  -v /mnt:/mnt \
  -p 5000:5000 \
  --name kld-eval \
  voipmonitor/sglang:test-cu132 \
  sleep infinity
```

### Verifying the KLD Patch

```bash
docker exec <container> grep -c "_kld_maybe_save" /opt/sglang/python/sglang/srt/layers/logits_processor.py
```
If 0, apply the patch:
```bash
docker exec <container> python /tmp/sglang-kld-logit-capture.py
```

### Phase 1: FP8 Reference (TP8, 8 GPUs)

```bash
docker exec <container> mkdir -p /mnt/kld_ref
docker exec <container> rm -f /mnt/kld_ref/*.safetensors

# Start FP8 server
docker exec -d <container> bash -c 'SGLANG_KLD_SAVE_DIR=/mnt/kld_ref \
  NCCL_P2P_LEVEL=SYS python3 -m sglang.launch_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 8 --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-custom-all-reduce \
  --attention-backend triton \
  --host 0.0.0.0 --port 5000 \
  2>&1 | tee /tmp/kld_server.log'

# Wait for server, then generate logits
docker exec <container> bash -c 'cd /workspace && python3 sglang_kld_eval.py \
  --phase ref --server-url http://localhost:5000 \
  --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
  --logits-dir /mnt/kld_ref --num-windows 100'
```

Expected: 100 files, each ~1188 MB (float32), shape [2048, 152064].

### Phase 2: Test Model (TP4, 4 GPUs)

Kill the ref server first (use `docker top` + `kill` specific PIDs, NOT `pkill`).

#### lukealonso/NVFP4 (flashinfer_cutlass)

```bash
docker exec <container> mkdir -p /mnt/kld_test_luke
docker exec <container> rm -f /mnt/kld_test_luke/*.safetensors

docker exec -d <container> bash -c 'SGLANG_KLD_SAVE_DIR=/mnt/kld_test_luke \
  NCCL_P2P_LEVEL=SYS SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model lukealonso/Qwen3.5-397B-A17B-NVFP4 --served-model-name Qwen3.5 \
  --tensor-parallel-size 4 --trust-remote-code \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cutlass \
  --attention-backend triton \
  --mem-fraction-static 0.90 \
  --disable-custom-all-reduce \
  --chunked-prefill-size 4096 \
  --speculative-algo NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 64 --max-running-requests 64 \
  --host 0.0.0.0 --port 5000 \
  2>&1 | tee /tmp/kld_server.log'
```

#### nvidia/NVFP4

Same as above but `--model nvidia/Qwen3.5-397B-A17B-NVFP4`.

#### QuantTrio/AWQ

```bash
docker exec -d <container> bash -c 'SGLANG_KLD_SAVE_DIR=/mnt/kld_test_awq \
  NCCL_P2P_LEVEL=SYS SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model QuantTrio/Qwen3.5-397B-A17B-AWQ --served-model-name Qwen3.5 \
  --tensor-parallel-size 4 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --mem-fraction-static 0.90 \
  --disable-custom-all-reduce \
  --chunked-prefill-size 4096 \
  --speculative-algo NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 64 --max-running-requests 64 \
  --host 0.0.0.0 --port 5000 \
  2>&1 | tee /tmp/kld_server.log'
```

#### Generate test logits (same for all models)

```bash
docker exec <container> bash -c 'cd /workspace && python3 sglang_kld_eval.py \
  --phase test --server-url http://localhost:5000 \
  --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
  --logits-dir /mnt/kld_test_luke --num-windows 100'
```

Expected: 100 files (MTP heads are auto-filtered by the patch).

### Phase 3: Compute KLD

Kill the server. Use a free GPU (e.g., GPU 4-7 if server used 0-3):

```bash
docker exec <container> bash -c 'CUDA_VISIBLE_DEVICES=4 python3 /workspace/sglang_kld_eval.py \
  --phase compute \
  --ref-dir /mnt/kld_ref \
  --test-dirs /mnt/kld_test_luke \
  --test-names "lukealonso/NVFP4"'
```

Or compare multiple:
```bash
docker exec <container> bash -c 'CUDA_VISIBLE_DEVICES=4 python3 /workspace/sglang_kld_eval.py \
  --phase compute \
  --ref-dir /mnt/kld_ref \
  --test-dirs /mnt/kld_test_luke /mnt/kld_test_nvidia /mnt/kld_test_awq \
  --test-names "lukealonso/NVFP4" "nvidia/NVFP4" "QuantTrio/AWQ"'
```

## Important Notes

### Server Management
- **NEVER use `pkill -f sglang`** inside `docker exec` — it kills the bash session itself.
- Use `docker top <container>` to find PIDs, then `docker exec <container> kill -9 <pid>`.
- Or use `nvidia-smi --query-compute-apps=pid --format=csv,noheader` to find GPU processes.

### MTP and KLD
- MTP speculative decoding is safe during KLD — the patch auto-filters MTP head logits.
- Verify correct file count: should be exactly `num-windows` (default 100), NOT 200.

### Reference Compatibility
- Reference and test logits MUST be from the same SGLang/torch setup.
- Different setups produce different FP8 reference logits (float16 rounding, different kernels).
- Always use the same tokenizer for ref and test (`Qwen/Qwen3.5-397B-A17B-FP8`).

### File Alignment
- If ref has N files numbered 0..N-1 and test has N files numbered 0..N-1, they align directly.
- If ref was captured with old patch (200 files, MTP contaminated), use even files only:
  `ref file i*2` corresponds to `test file i`.

### Expected Results (Qwen3.5-397B-A17B)

| Model | Expected Mean KLD |
|-------|------------------|
| QuantTrio/AWQ (INT4) | ~0.024 |
| nvidia/NVFP4 | ~0.035 |
| lukealonso/NVFP4 | ~0.036 |

### Interpretation

| Mean KLD | Quality |
|----------|---------|
| < 0.01 | Near-lossless |
| 0.01 - 0.05 | Good, minimal loss |
| 0.05 - 0.1 | Noticeable loss |
| > 0.1 | Significant loss |

## Patching SGLang (if not using voipmonitor/sglang:test-cu132)

If the container doesn't have the KLD patch, apply it:

```bash
# Copy patch into container
docker cp /root/docker/llm-services/pytorch/patches/sglang-kld-logit-capture.py <container>:/tmp/

# Apply
docker exec <container> python3 /tmp/sglang-kld-logit-capture.py

# Copy eval script
docker cp /root/docker/llm-services/pytorch/scripts/sglang_kld_eval.py <container>:/workspace/

# Install datasets (needed for wikitext)
docker exec <container> pip install datasets
```
