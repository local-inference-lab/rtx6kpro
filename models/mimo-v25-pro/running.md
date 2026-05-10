# Running MiMo-V2.5-Pro TP=8 on SGLang/B12X

This is the exact runbook for the validated MiMo-V2.5-Pro TP=8 runtime.

## Image

Use the DockerHub image with the SGLang/B12X patch overlay baked in:

```bash
voipmonitor/sglang:mimo-v25-pro-tp8-b12x3917cb2-20260510
```

Digest:

```text
sha256:96aa5a10913cae3af6fe145e5c21238549971da271fc06b522ec4a6a9bd51c80
```

The `mimo-v25-pro-tp8-b12x3917cb2-latest` tag points to the same digest at the time this page was written, but the dated tag is preferred for reproducibility.

This image layers `lukealonso/b12x@3917cb2fe5a2118eaab8b68f7710c71aad9e4b1c` over the previously validated SGLang autotune-fix image. The validated launch below uses `B12X_MOE_FORCE_A16=1`.

## Required model directories

The command below assumes these paths exist on the host:

```text
/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-BF16-MTP
/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn
/data/models/MiMo-V2.5-Pro-NVFP4
/models/.cache/huggingface
/models/.vllm_cache/triton
/models/.vllm_cache/sglang-generated
```

The public checkpoint is:

```text
festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8
```

The MTP overlay directory is expected to contain the BF16 draft weights and symlink the target-model files back to the mixed NVFP4/MXFP8 checkpoint.

## Launch command

This command runs TP=8 on GPUs 0-7 and serves the OpenAI-compatible API on port `30004`.

```bash
docker run -d --name mimo-v25-pro-tp8-b12x3917-a16-30004 \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --ipc=host --network host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e CUTE_DSL_ARCH="sm_120a" \
  -e SGLANG_PREVENT_THOUGHT_LOOPS=1 \
  -e B12X_MOE_FORCE_A16=1 \
  -e B12X_ENABLE_DYNAMIC_DOWN_SCALE=0 \
  -e B12X_MOE_EAGER_EXACT_DYNAMIC=0 \
  -e SGLANG_DISABLE_AUTOTUNED_LINEAR_AFTER_WARMUP=1 \
  -e SGLANG_UNQUANT_AUTOTUNED_LINEAR_MAX_TOKENS=128 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e NCCL_NET_GDR_LEVEL=SYS \
  -v /data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-BF16-MTP:/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-BF16-MTP:ro \
  -v /data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn:/data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn:ro \
  -v /data/models/MiMo-V2.5-Pro-NVFP4:/data/models/MiMo-V2.5-Pro-NVFP4:ro \
  -v /models/.cache/huggingface:/root/.cache/huggingface \
  -v /models/.vllm_cache/triton:/root/.triton \
  -v /models/.vllm_cache/sglang-generated:/root/.cache/sglang-generated \
  voipmonitor/sglang:mimo-v25-pro-tp8-b12x3917cb2-20260510 \
  python3 -m sglang.launch_server \
    --model-path /data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-BF16-MTP \
    --tokenizer-path /data/models/MiMo-V2.5-Pro-NVFP4-MXFP8-attn \
    --served-model-name mimo-v2.5-pro \
    --tp-size 8 \
    --json-model-override-args '{"architectures":["MiMoV2FlashForCausalLM"]}' \
    --page-size 64 \
    --host 0.0.0.0 \
    --port 30004 \
    --kv-cache-dtype fp8_e4m3 \
    --quantization modelopt_mixed \
    --mem-fraction-static 0.85 \
    --swa-full-tokens-ratio 0.3 \
    --chunked-prefill-size 8192 \
    --context-length 131072 \
    --max-running-requests 8 \
    --enable-multi-layer-eagle \
    --reasoning-parser mimo \
    --tool-call-parser mimo \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --moe-runner-backend b12x \
    --attention-backend b12x \
    --mm-attention-backend b12x \
    --fp4-gemm-backend cutlass \
    --fp8-gemm-backend flashinfer_cutlass \
    --sampling-backend pytorch \
    --cuda-graph-max-bs 8 \
    --disable-piecewise-cuda-graph \
    --enable-metrics
```

## Health and logs

Health:

```bash
curl -sS http://127.0.0.1:30004/health
```

Follow logs:

```bash
docker logs -f mimo-v25-pro-tp8-b12x3917-a16-30004
```

Last 200 lines:

```bash
docker logs --tail 200 mimo-v25-pro-tp8-b12x3917-a16-30004
```

Recent hard errors:

```bash
docker logs --since 10m mimo-v25-pro-tp8-b12x3917-a16-30004 2>&1 \
  | rg -i 'traceback|device-side|scheduler hit|cuda error|exception|nan|\binf\b'
```

## Smoke test

MiMo's tokenizer defaults to thinking mode. For normal chat tests, pass:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

Smoke request:

```bash
curl -sS http://127.0.0.1:30004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mimo-v2.5-pro","messages":[{"role":"user","content":"First write exactly this sentence: The capital of France is Paris. Then write one coherent paragraph explaining why checkpoint integrity matters for large language model deployment."}],"temperature":0,"max_tokens":384,"chat_template_kwargs":{"enable_thinking":false}}'
```

Expected properties:

- HTTP 200
- output includes `The capital of France is Paris.`
- no CJK or replacement-character corruption
- `finish_reason` should normally be `stop`

## Runtime notes

`--page-size 64` is mandatory for B12X attention.

`--fp4-gemm-backend cutlass` was used in the validated image. Earlier runs with other FP4 paths were useful for isolation, but this is the current documented runtime.

`B12X_MOE_FORCE_A16=1` is the current recommended setting for the B12X 3917 image. It keeps the routed MoE execution on the A16 path while retaining the mixed NVFP4/MXFP8 checkpoint and B12X attention. The previous `B12X_MOE_FORCE_A16=0` micro reciprocal-scale path remains a useful validation artifact, but it was not the path used for the B12X 3917 long-context checks below.

`--disable-piecewise-cuda-graph` is intentional. The validated path uses normal CUDA graph target verify with `--cuda-graph-max-bs 8`; piecewise CUDA graph was disabled after earlier runtime recapture failures.

`SGLANG_DISABLE_AUTOTUNED_LINEAR_AFTER_WARMUP=1` is intentional in the autotune-fix image. The SGLang overlay now allows new unquantized BF16 dense-linear `torch.compile` keys only while CUDA graph warmup/capture is explicitly running. After startup, live requests can reuse already captured compiled keys, but unknown eager prefill shapes fall back to `F.linear` instead of triggering Torch Inductor autotune during inference.

`SGLANG_UNQUANT_AUTOTUNED_LINEAR_MAX_TOKENS=128` is a secondary guard for the unquantized BF16 dense-linear fast path. It keeps graph/decode-size shapes eligible for the compiled path and keeps large dynamic prefill chunks off that path.

Startup logs can still contain `AUTOTUNE mm(...)` during CUDA graph capture. The validated behavior is that request windows do not add new `AUTOTUNE mm(...)` lines. Check a request window with:

```bash
SINCE=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
# run a request here
docker logs --since "$SINCE" mimo-v25-pro-tp8-b12x3917-a16-30004 2>&1 \
  | rg 'AUTOTUNE mm|Prefill batch|Decode batch|HTTP/1.1'
```

## Validated result

The production-style container tested before publishing this page was:

```text
mimo-v25-pro-tp8-b12x3917-a16-30004
```

Results:

| Test | Result |
|---|---|
| smoke | coherent Paris + checkpoint-integrity answer, HTTP 200 |
| first 8-way concurrent soak | 8/8 HTTP 200, 2,985 completion tokens, 83 s |
| warm 8-way concurrent soak | 8/8 HTTP 200, 4,042 completion tokens, 31 s |
| short autotune-fix request | coherent `Paris` + `4`, HTTP 200, no request-time `AUTOTUNE mm` |
| long autotune-fix request | 50,712 prompt tokens, returned `VEGA-8431` + `Paris`, HTTP 200, no request-time `AUTOTUNE mm` |
| B12X 3917 smoke | returned `Paris, 4`, HTTP 200 |
| B12X 3917 long context A | returned `VEGA-8431, Paris`, HTTP 200, no request-time `AUTOTUNE mm` |
| B12X 3917 long context B | returned `ORION-9265, Paris`, HTTP 200, no request-time `AUTOTUNE mm` |
| health after soak | HTTP 200 |
