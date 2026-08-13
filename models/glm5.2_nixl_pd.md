# GLM-5.2 NIXL Prefill/Decode Disaggregation

Status: **locally validated** on a two-node RTX PRO 6000 Blackwell deployment
using RoCEv2. This is an operational recipe, not a qualification receipt for
the current GLM-5.2 release image.

This page documents the compatibility fix and runtime contract used to run a
GLM-5.2 Gilded Gnosis r28-derived image with vLLM's `NixlConnector`. The base
release is the r28 image documented in the
[GLM-5.2 v20 release history](glm5.2_v20_history.md#start-the-server). Prefill
and decode run as separate services and transfer CUDA KV-cache buffers over
UCX/RDMA.

## Reference image

The validated local image was:

```text
local/vllm-glm52-v20-r28-kvtransfer:20260810
```

It is an overlay of the history page's immutable r28 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
```

The `local/` name is a build receipt, not a publicly pullable registry image.
Build the overlay on every node, publish it to an accessible registry, or
replace the name with an equivalent immutable image before using the command
templates below.

The r28 base already contains the CUDA 13 NIXL binary distribution
(`nixl-cu13`), but it does not contain the small `nixl` Python package that
exposes the import namespace required by connector discovery. The validated
overlay installs `nixl==1.3.2` without replacing the base image's vLLM, PyTorch,
or native CUDA 13 stack:

```dockerfile
ARG BASE_IMAGE=voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
FROM ${BASE_IMAGE}

USER root

ARG NIXL_VERSION=1.3.2

RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps \
      "nixl==${NIXL_VERSION}" \
    && /opt/venv/bin/python -c \
      "import importlib.util as u; assert u.find_spec('nixl')"
```

The locally validated image also carried Mooncake compatibility packages for
an alternative connector, but they are not required by the NIXL configuration
on this page and are intentionally omitted from the minimal overlay above. If
Mooncake is added to a CUDA 13 image, use `mooncake-transfer-engine-cuda13`
rather than adding a CUDA 12 runtime compatibility path.

## Mandatory runtime contract

Apply these rules to both Prefill and Decode:

1. **Every NIXL P/D configuration must disable expandable CUDA allocator
   segments, including DCP1/2/4/8 and both Prefill and Decode:**

   ```bash
   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
   ```

   Do not copy the common PyTorch OOM suggestion to enable expandable segments
   into this recipe. NIXL registers CUDA KV-cache allocations for transfer;
   the validated configuration requires the non-expandable allocator path.

2. **When DCP is 4, set four indexer shards on both services:**

   ```bash
   -e DCP=4 \
   -e DCP_INDEXER_SHARDS=4
   ```

   This page validates DCP4 only. Treat other DCP layouts as separate profiles
   and validate their indexer geometry before deployment.

3. Give each container host networking, IPC access, an unlimited memlock, and
   access to the selected RDMA device. The two services must use compatible
   model, TP/DCP, KV-cache, and block-size settings.

4. **The execution modes are intentionally asymmetric:** Prefill uses
   `--enforce-eager`, while Decode uses
   `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`. Do not
   collapse these into one shared set of trailing arguments.

## Network prerequisites

- Configure routable RoCEv2 addresses on the Prefill and Decode data-plane
  interfaces. Do not use a management address for the KV path.
- Verify link state, lossless-network policy, MTU, and RDMA connectivity before
  starting vLLM.
- Set `UCX_NET_DEVICES` to the local RDMA device and port on each node. Do not
  assume that Linux Ethernet names and RDMA device names are identical.
- `VLLM_NIXL_SIDE_CHANNEL_HOST` is the **local** data-plane address on each
  service, not the remote peer address.

The validated UCX transport selection was:

```bash
-e UCX_TLS=rc,cuda_copy
```

## Container privilege

The templates retain `--privileged` because that is the exact container mode
used by the validated deployment. It gives the container access to all host
devices and capabilities, so it is **not** a least-privilege recommendation.
Use it only with a trusted image on a controlled inference node.

For a hardened deployment, replace `--privileged` with explicit mappings for
the required `/dev/infiniband` devices and only the capabilities required by
the local RDMA, GPUDirect, and container-runtime configuration. Device nodes
vary by host, and that reduced-privilege profile was not validated by this
receipt; re-run RDMA registration and end-to-end KV-transfer validation after
changing the security boundary.

## Prefill template

The Prefill worker is the KV producer and runs eager. The following is the
complete P/D-specific container shape; fill in the model paths and keep the
remaining model and scheduler values aligned with the matching GLM-5.2
runbook.

```bash
docker run \
  --gpus all \
  --ipc=host \
  --network host \
  --privileged \
  --ulimit memlock=-1 \
  -e UCX_TLS=rc,cuda_copy \
  -e UCX_NET_DEVICES=<PREFILL_RDMA_DEVICE>:1 \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=<PREFILL_RDMA_IP> \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=5557 \
  -e DCP=4 \
  -e DCP_INDEXER_SHARDS=4 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  local/vllm-glm52-v20-r28-kvtransfer:20260810 \
  --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
  --enforce-eager
```

## Decode template

The Decode worker is the KV consumer. It keeps full CUDA graphs for decode
only instead of inheriting Prefill's eager setting:

```bash
docker run \
  --gpus all \
  --ipc=host \
  --network host \
  --privileged \
  --ulimit memlock=-1 \
  -e UCX_TLS=rc,cuda_copy \
  -e UCX_NET_DEVICES=<DECODE_RDMA_DEVICE>:1 \
  -e VLLM_NIXL_SIDE_CHANNEL_HOST=<DECODE_RDMA_IP> \
  -e VLLM_NIXL_SIDE_CHANNEL_PORT=5558 \
  -e DCP=4 \
  -e DCP_INDEXER_SHARDS=4 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  local/vllm-glm52-v20-r28-kvtransfer:20260810 \
  --kv-transfer-config \
    '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

`kv_load_failure_policy=fail` is intentional: a missing or incompatible KV
transfer should fail visibly instead of silently changing the request path.

## Validation

Before model startup, confirm that connector discovery succeeds:

```bash
/opt/venv/bin/python -c \
  "import importlib.util as u; assert u.find_spec('nixl')"
```

After both workers and the P/D router are healthy, send an end-to-end request
that performs Prefill on the producer and Decode on the consumer. The validated
deployment logged all of the following:

```text
NIXL compatibility check passed
Backend UCX was instantiated
Registering KV_Caches ... kv_buffer_device: cuda, use_host_buffer: False
Two-stage processing completed successfully
```

## Troubleshooting

| Symptom | Check |
|---|---|
| `NixlConnector` is unavailable or `import nixl` fails | Confirm that the small `nixl==1.3.2` namespace package is present in addition to `nixl-cu13`. |
| Startup or registration fails after copying another allocator recipe | Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` on **both** services. |
| DCP4 indexer or KV geometry differs between P and D | Confirm `DCP=4` and `DCP_INDEXER_SHARDS=4` on **both** services. |
| Prefill and Decode use the same execution-mode flags | Restore `--enforce-eager` on Prefill and `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` on Decode; the profiles are intentionally different. |
| UCX selects the wrong interface or cannot reach the peer | Check each node's local `UCX_NET_DEVICES`, RoCE address, GID, MTU, routing, and firewall. |
| Requests succeed without evidence of KV transfer | Require producer/consumer selection in the router and verify the NIXL/UCX registration and transfer logs. |

Do not infer a working NIXL data path from service health alone. The acceptance
gate is a successful two-stage request plus connector registration and transfer
evidence on the workers.
