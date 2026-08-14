# GLM-5.2 Infernal Invocation r11

**Status: qualified for the NVFP4 TP8 and EXL3 R7 TP4 profiles specified on
this page.** The image serves GLM-5.2 on RTX PRO 6000 Blackwell GPUs with
B12X sparse MLA and MoE kernels, CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, and
InstantTensor loading.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm908522a-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r11` |
| Registry digest | `sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971` |
| Image ID | `sha256:f226a6fd788bb4af345a17b768654f1e5a7487a812746ccb117aa9b040a82294` |
| vLLM base | `dev/infernal-invocation@ce5f50f6d01b02336c4207f11277fd7bedacb4d6` |
| vLLM integration tree | `908522a320ecc26582926228c9644af085f5a86c` |
| B12X integration tree | `5d648d944a047d4fac5c2035309c207b3faebd9c` |
| LMCache integration tree | `5fdf59cfa184bc15dc5414df0bd633da9e49aaae` |
| Docker source | [`6f92a9ecf`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/6f92a9ecff35f0dabd6f444b05daab4024b49257) |
| Qualification receipt | [`infernal-invocation-r11-local-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/6f92a9ecff35f0dabd6f444b05daab4024b49257/validation/infernal-invocation-r11-local-gpu.json) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The source locks contain no private source patch. Each lock records its base
commit, ordered pull-request heads, resulting Git tree, and patch digest.

## NVFP4 TP8 Profile

Download the immutable Compose specification and start the server:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/6f92a9ecff35f0dabd6f444b05daab4024b49257/examples/docker-compose-glm52-nvfp4-infernal-invocation-r11.yml
docker compose -f docker-compose-glm52-nvfp4-infernal-invocation-r11.yml up -d
```

The profile serves `lukealonso/GLM-5.2-NVFP4` revision
`8a1f4a13204acf2b7ac840375efaed64c231c522` with this contract:

| Setting | Value |
|---|---|
| Tensor/decode context parallelism | TP8 / DCP1 |
| Speculative decoding | MTP3 |
| Scheduler | `MAX_NUM_SEQS=32`, `MAX_BATCHED_TOKENS=8192` |
| CUDA graph cap | 128 rows |
| Context limit | 262,144 tokens |
| GPU memory utilization | 0.95 |
| Routed experts | Serialized NVFP4, B12X W4A16 execution |
| Dense projections | Online MXFP8 without ignored projections |
| KV cache | FP8 MLA |
| Loading | InstantTensor `BUFFERED` |
| FP8 transport | `F8_DMA=ring` |

Override deployment values before `docker compose up` when required:

```bash
PORT=8001 GPUS=8,9,10,11,12,13,14,15 \
JIT_CACHE=/srv/cache/glm52-nvfp4-ii-r11 \
docker compose -f docker-compose-glm52-nvfp4-infernal-invocation-r11.yml up -d
```

Use one writable JIT directory for one image, checkpoint revision, and serving
profile. Sharing a JIT directory between NVFP4 and EXL3 processes is
unsupported.

## EXL3 R7 TP4 Profile

Download and start the projection-mixed EXL3 R7 profile:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/6f92a9ecff35f0dabd6f444b05daab4024b49257/examples/docker-compose-glm52-exl3-infernal-invocation-r11.yml
docker compose -f docker-compose-glm52-exl3-infernal-invocation-r11.yml up -d
```

The profile serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f` with this contract:

| Setting | Value |
|---|---|
| Tensor/decode context parallelism | TP4 / DCP1 |
| Speculative decoding | MTP3, greedy draft selection |
| Scheduler | `MAX_NUM_SEQS=8`, `MAX_BATCHED_TOKENS=2048` |
| CUDA graph cap | 32 rows |
| Context limit | 65,536 tokens |
| GPU memory utilization | 0.98 |
| Routed experts | Checkpoint-native Trellis K3/K4/K5 |
| Dense and shared projections | Online EXL3 K6 |
| KV cache | NVFP4 DS-MLA |
| Loading | InstantTensor `BUFFERED`, `INSTANTTENSOR_COPY=0` |
| Decode graphs | FULL |

The first load encodes eligible BF16 tensors as K6 and stores the derived
payload under `/cache`. Reuse the same checkpoint revision and persistent JIT
volume to avoid repeating that work.

InstantTensor borrowed buffers may be reused after the loader iterator
advances. A layerwise online quantizer therefore clones only marked borrowed
tensors that must survive into deferred layer processing. The copy is released
after the layer completes. This ownership rule prevents retained weights from
changing while avoiding the checkpoint-wide memory cost of
`INSTANTTENSOR_COPY=1`; it is implemented by
[vLLM PR #305](https://github.com/local-inference-lab/vllm/pull/305).

## Qualified Results

All measurements used the published image without source bind mounts.

| Profile | Condition | Aggregate decode | Active-user decode | Errors |
|---|---|---:|---:|---:|
| NVFP4 TP8/DCP1/MTP3 | CC1, context 0, 5 s warmup, 30 s measurement, temperature 0 | 182.54 tok/s | 183.38 tok/s | 0 |
| EXL3 R7 TP4/DCP1/MTP3 | CC1, context 0, 3 s warmup, 20 s measurement | 131.45 tok/s | 129.07 tok/s | 0 |
| EXL3 R7 TP4/DCP1/MTP3 | repeated 20 s measurement | 127.22 tok/s | 127.46 tok/s | 0 |

The NVFP4 profile allocated 625,856 KV tokens. The EXL3 R7 profile allocated
72,384 KV tokens under its 65,536-token qualification configuration. Both
profiles returned `703` for a deterministic `37 * 19` request. The EXL3
checkpoint loaded 322 GiB of source weights and prepared every mixed K3/K4/K5
routed-expert layer with borrowed InstantTensor buffers enabled.

## Source Responsibilities

| Repository | Pull requests | Responsibility |
|---|---|---|
| vLLM | [#300](https://github.com/local-inference-lab/vllm/pull/300) | Validate and load projection-mixed EXL3 R7 checkpoints; cache online K6 payloads. |
| vLLM | [#301](https://github.com/local-inference-lab/vllm/pull/301) | Enforce GLM-5.2 B12X sparse MLA layout, decode-extension, DCP metadata, and virtual-TP contracts. |
| vLLM | [#305](https://github.com/local-inference-lab/vllm/pull/305) | Own borrowed weights retained by deferred layerwise online processing. |
| B12X | [#148](https://github.com/local-inference-lab/b12x/pull/148) | Execute projection-mixed Trellis K3/K4/K5 routed experts. |

Issue [#67](https://github.com/local-inference-lab/rtx6kpro/issues/67)
records the complete vLLM, B12X, and LMCache merge contract for the image.

## Qualification Boundary

- **Qualified:** NVFP4 TP8/DCP1/MTP3 and EXL3 R7 TP4/DCP1/MTP3 startup,
  deterministic arithmetic, FULL decode graph capture where configured, and
  the CC1/context-zero measurements above.
- **Implemented:** GLM DCP policy, TP6 virtual sharding, LMCache, and EXL3
  schemas already supported by the shared launcher.
- **Unqualified:** GLM prefill, DCP greater than one, TP6, higher concurrency,
  alternate EXL3 checkpoints, hybrid NF3 checkpoints, and long-context
  quality for this source composition.

Community test reports must include the image digest, checkpoint repository
and revision, TP/DCP/MTP values, physical GPU order, PCIe topology, context
length, concurrency, resolved helper-policy lines, and the relevant engine log
excerpt. Throughput comparisons require identical checkpoint, graph,
scheduler, quantization, and GPU placement settings.
