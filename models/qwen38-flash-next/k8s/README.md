# Kubernetes example: N TP1 replicas, one shared PLE table

Status: **example**. These manifests show the mechanism described in the
[recipe page](../../qwen38-flash-next.md#multiple-tp1-replicas-with-one-shared-ple-table)
with plain `nvidia.com/gpu` scheduling and no site-specific GPU pinning. The
qualified single-GPU evidence is in the
[validation report](../validation/shared-ple-r35-20260916.md); the operator
manifests that produced it pin GPUs by UUID through DRA and are not part of
this example.

- `deployment.yaml`: ConfigMap, Service and a Deployment with `replicas: 2`
  that co-locates every replica on one node (`podAffinity`), mounts the host's
  `/dev/shm` with `hostIPC: true`, and sets `VLLM_PLE_TABLE_MEMORY=shared`.
  Replace the model-cache PVC, the API key secret and the image tag as needed.
  The 80Gi memory limit is required on every replica because any of them may
  populate the 26.82 GiB table after a reboot; requests follow the attacher's
  measured footprint (7.9 GiB) with headroom.
- `prune-job.yaml`: removes stale table directories after a checkpoint or
  revision bump. Set `KEEP` to the key the running replicas log, run with
  `--dry-run` first (the default), then drop the flag.

```bash
kubectl apply -f deployment.yaml
kubectl logs deploy/qwen3-8-flash-next-vllm | grep "shared PLE table"
```
