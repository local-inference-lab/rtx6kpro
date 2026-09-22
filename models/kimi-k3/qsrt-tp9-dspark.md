# Kimi-K3 QSRT-K2 with Red Hat DSpark on nine GPUs

Run `lukealonso/Kimi-K3-QSRT-K2` with Red Hat DSpark K5 on **nine RTX PRO 6000
Blackwell 96 GiB GPUs**, using TP9/DCP9. The profile enables vision,
InstantTensor loading, native CPU KV offload, 4,096-token prefill scheduling,
shared-expert overlap and L2 prefetch. No source checkout is mounted into the
container.

## Start the server

The Hugging Face cache must contain both checkpoints at the revisions below.
If needed, download them with the Hugging Face CLI; allow sufficient disk space
for the complete QSRT-K2 checkpoint:

```bash
hf download lukealonso/Kimi-K3-QSRT-K2 \
  --revision 3b98114115f1d41ce7963ba346c3fca19918b0bd
hf download RedHatAI/Kimi-K3-speculator.dspark \
  --revision 38a88101e0d46bb22134b9da340f381b954d40d4
```

Use Docker with the NVIDIA Container Toolkit and a CUDA 13.4-capable driver;
this profile was tested with driver 615.71.09. The following command uses the
first nine CUDA devices in PCI bus order and binds port 8012 on all interfaces:

```bash
docker run -d --init --name kimi-k3-qsrt-tp9 --gpus all \
  --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/kimi-k3-cache/tp9:/cache/kimi-k3 \
  -e HF_HUB_OFFLINE=1 \
  -e KIMI_QUANT_FORMAT=qsrt_k2 \
  -e KIMI_CHECKPOINT=/root/.cache/huggingface/hub/models--lukealonso--Kimi-K3-QSRT-K2/snapshots/3b98114115f1d41ce7963ba346c3fca19918b0bd \
  -e KIMI_SPECULATOR=dspark-redhat \
  -e KIMI_TP=9 -e KIMI_DCP=9 -e KIMI_MAX_SEQS=1 \
  -e KIMI_KV_BYTES=0 -e KIMI_PORT=8012 \
  voipmonitor/vllm:kimi-k3-kk-cu134-tp9-dspark-k5-20260922-r1 \
  bash /opt/lil/runtime/serve-kimi-k3.sh
```

Before exposing the port outside a trusted network, add
`--env-file /path/to/api-key.env` to `docker run`; that file must define
`VLLM_API_KEY`. Clients then send `Authorization: Bearer <key>`. Do not put the
key in shell arguments or logs. This launch does not enable development RPCs.

Watch startup with `docker logs -f kimi-k3-qsrt-tp9`. The API is
`http://HOST:8012/v1`, model name **`Kimi-K3-QSRT-K2`**. Keep the cache volume
between launches so unchanged CuTeDSL/Triton kernels need not compile again.

## Settings

The launch above selects the tested values explicitly where a global launcher
default differs. Other settings belong to the `dspark-redhat` TP9 profile:

| Setting | Default / recommended value | Example override and consequence |
| --- | --- | --- |
| `KIMI_TP`, `KIMI_DCP` | QSRT defaults to TP10 and DCP=TP; use **9/9** | `-e KIMI_TP=9 -e KIMI_DCP=9`; both are required for this profile |
| `KIMI_MAX_SEQS` | **1** | `-e KIMI_MAX_SEQS=2` changes admission and captured graph sizes; extra concurrency is not qualified here |
| `KIMI_MAX_MODEL_LEN` | **950000** | `-e KIMI_MAX_MODEL_LEN=262144` limits input + output, not physical cache bytes |
| `KIMI_KV_BYTES` | Required for QSRT; **0** selects automatic sizing | `-e KIMI_KV_BYTES=6000000000` fixes per-rank GPU KV bytes; capacity must still cover the selected context |
| `KIMI_GPU_MEMORY_UTILIZATION` | **0.970** for this TP9 profile | `-e KIMI_GPU_MEMORY_UTILIZATION=0.960` leaves more headroom when `KIMI_KV_BYTES=0` |
| `KIMI_NATIVE_KV_GIB` | **32** | `-e KIMI_NATIVE_KV_GIB=16` halves native CPU offload capacity |
| `VLLM_KIMI_L2_PREFETCH` | **1** in this profile; source default is 0 | `-e VLLM_KIMI_L2_PREFETCH=0` disables cache hints for an A/B test |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | **0** in this profile | `-e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` disables shared-expert overlap |

DSpark uses five proposals, with target CUDA graphs for five/six rows and a
five-row draft graph. The draft retains a 32,768-token tail. The target's
routed experts use A16 activations; the draft uses MXFP8 weights with BF16 QKV
and Markov projections. These optimizations do not add weight or activation
quantization to the selected configuration.

Vision is enabled without an image-count limit. The processor limits an image
to 40,960 patches and 512 patches per side. Keep sufficient transient memory
for large image requests. The TP9 profile uses data-parallel vision encoding;
do not substitute tensor-parallel vision settings without requalification.

The server reported **3,262,264 physical token slots** with the tested cache
configuration; its per-request context limit remains **950,000**. Physical
slots do not mean that several 950k requests can be admitted with
`KIMI_MAX_SEQS=1`.

## Measured performance

| Workload | Result |
| --- | ---: |
| Single coding request, 183 input / 4,096 output tokens | **109.72 tok/s median decode** |
| Draft acceptance on that coding request | **46.0645%**, 3.303 emitted tokens per target cycle |
| Cold prefill, 8k / 32k / 64k inputs | **2,329 / 2,585 / 2,504 tok/s** |
| Native host-KV replay, 78,114-token input + 64-token output | **1.95 s**, versus 31.13 s cold |

Measured on nine RTX PRO 6000 Blackwell GPUs, 600 W power limits, driver
615.71.09, using the image and settings above. Clock policy was left unchanged;
frequency was not recorded during timing. Decode is the median of three
repeated runs; prefill numbers are one cold sample per length. The initial
post-startup decode measurement was 104.74 tok/s. The controlled resident A/B
shows **+7.72%** from overlap + L2 hints. See the
[qualification report](qsrt-tp9-dspark-qualification.md) for both measurements,
exact parity checks and validation limits.

## Reproduce the benchmark

Download and extract the validation archive from the
[TP9 artifact release](https://github.com/local-inference-lab/rtx6kpro/releases/tag/kimi-k3-qsrt-tp9-dspark-20260922).
It includes the stored coding token IDs and the exact benchmark client.
With no other requests or GPU jobs active, run:

```bash
python3 validation/tools/benchmark-kimi-k3-dspark-decode.py \
  --url http://127.0.0.1:8012 --model Kimi-K3-QSRT-K2 \
  --token-file validation/inputs/coding-game-token-ids.json \
  --prompt-tokens 183 --max-tokens 4096 --warmups 1 --runs 3 \
  --output-dir coding-tp9-results
```

The client records decode timing, output hashes and acceptance counters.
Server logs' periodically averaged generation throughput is not a substitute
for this request-level measurement.

## Reconstruct sources and build the image

Install Git, Docker Buildx, Python 3.12, `jq`, `zstd`, and `uv`. Clone this wiki
repository, then reconstruct the pinned source composition into an absent
directory:

```bash
git clone https://github.com/local-inference-lab/rtx6kpro.git
bash rtx6kpro/models/kimi-k3/tools/reproduce-qsrt-tp9-sources.sh ./kimi-sources
```

The script starts from pinned `dev/karmic-kraken` and B12X `master` commits,
merges the PR heads, and verifies that the resulting Git trees are exactly
those used in the image. It does not change either remote mainline.

Download `kimi-k3-tp9-components.tar.zst` and its `.sha256` file from the
[artifact release](https://github.com/local-inference-lab/rtx6kpro/releases/tag/kimi-k3-qsrt-tp9-dspark-20260922).
The archive contains the exact component wheels and manifests, not model
weights. Assemble and build them without recompiling unchanged dependencies:

```bash
sha256sum --check kimi-k3-tp9-components.tar.zst.sha256
tar --zstd -xf kimi-k3-tp9-components.tar.zst
bash rtx6kpro/models/kimi-k3/tools/assemble-qsrt-tp9-runtime.sh \
  ./kimi-sources ./kimi-k3-tp9-components ./kimi-runtime-bundle \
  local/kimi-k3:qsrt-tp9-reproduced
```

This verifies every component checksum and the composed runtime manifest
before building. For source changes, build replacement wheels using
`kimi-sources/vllm/tools/jovian_wheel_release/build_bundle.sh` and
`kimi-sources/b12x/ci/lil_wheels/build_bundle.sh`. Their pinned runtime locks
require the Buildx builder `lil-wheel-cu134-sm120`. vLLM supports
`VLLM_PRECOMPILED_BUNDLE` only when all native inputs and ABI fields match;
otherwise it must rebuild its native extensions. Requalify changed sources;
the exact-manifest assembly script intentionally rejects a different build.
