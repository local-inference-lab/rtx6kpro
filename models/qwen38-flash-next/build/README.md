# Qwen3.8-Flash-Next shared-PLE-table overlay on R35

Status: **implemented; overlay on the R35 community digest**.

This directory builds a derived image in which several independent TP1
Qwen3.8-Flash-Next NVFP4 vLLM processes on one host serve **one** CPU-resident
copy of the PLE n-gram table (26.82 GiB NVFP4 at TP1) instead of each pinning a
private copy. The change is a pure Python source overlay on the qualified
`localinferencelab/vllm:jovian-judgement-community-20260911-r35` image; b12x
and every native artifact are untouched. Serving profile, launch arguments
and checkpoint are those of the [TP1 recipe](../../qwen38-flash-next.md).

| Role | Identity |
|---|---|
| Base image | `docker.io/localinferencelab/vllm@sha256:7a425c6864b951bbc368111490a4b0ac69d8cd0dd4987075b2c7d40b753b1bf5` (`jovian-judgement-community-20260911-r35`, vLLM `0.26.1rc0+glm53.r35.vllmde982a50`, b12x 1.3.0 @ `98086604`) |
| Base source commit | `de982a50c6a3e4718e5cf9f00423a92192718da1` (the checkout at `/opt/glm53-flash/vllm` inside the image) |
| Overlay source | [`renehonig/vllm` branch `build/qwen38-shared-ple-r35`](https://github.com/renehonig/vllm/tree/build/qwen38-shared-ple-r35), commit `028b1c4921f92c61889b5b8485f1ef606a32b41c` (branch head: the five overlay commits plus the tmpfs-message, test-double and review-hardening follow-ups, all on top of `de982a50`) |
| PR branch | `feat/qwen38-shared-ple-table`, targeting `local-inference-lab/vllm` `dev/jovian-judgement` |
| Overlaid files | `vllm/envs.py`, `vllm/models/qwen3_8_flash_next/{model.py,ple_layer.py,ple_shared_table.py}`, `tests/models/test_qwen3_8_flash_next_ple_shared_table.py` |
| Published image | `ghcr.io/renehonig/vllm:jovian-r35-shared-ple-028b1c4921f9` = `ghcr.io/renehonig/vllm@sha256:65cec7a0dc6a702ed4911339365cbf5356cf8f14311714adbfee5e8f12bb60e1` |
| Source lock | [`qwen38-flash-next-shared-ple-r35.source.lock`](qwen38-flash-next-shared-ple-r35.source.lock): the image's `/opt/glm53-flash/source.lock` plus `overlay.*` keys (repository, commit, tree, per-file sha256) |
| Validation | [`../validation/shared-ple-r35-20260916.md`](../validation/shared-ple-r35-20260916.md) |

## How the overlay works

The R35 image is a source-overlay image: `PYTHONPATH=/opt/glm53-flash/vllm`
puts a git checkout of the fork ahead of the venv, and the native extensions
live inside that checkout. Replacing Python files in the tree is therefore
the sanctioned composition path. The `Dockerfile` copies the five files, then
refuses to build unless the tree is at `de982a50`, every copied file matches
the sha256 passed as a build argument, `git status` lists exactly those files,
the new module imports, and its CLI runs. `/opt/glm53-flash/overlay.identity`
records repository, branch, commit and base commit; OCI labels carry the same.

## Reproduce

```bash
git clone --filter=blob:none https://github.com/renehonig/vllm.git jovian-vllm
git -C jovian-vllm fetch origin build/qwen38-shared-ple-r35
cd rtx6kpro/models/qwen38-flash-next/build
VLLM_FORK=../../../../jovian-vllm ./build-overlay.sh 028b1c4921f92c61889b5b8485f1ef606a32b41c
# Optional GPU-side check (serve --help needs a visible GPU):
CHECK_GPU=<gpu-index> VLLM_FORK=... ./build-overlay.sh 028b1c4921f92c61889b5b8485f1ef606a32b41c
docker tag localinferencelab/vllm:jovian-r35-shared-ple-028b1c4921f9 <registry>/vllm:jovian-r35-shared-ple-028b1c4921f9
docker push <registry>/vllm:jovian-r35-shared-ple-028b1c4921f9   # pin the RepoDigest it prints
```

`build-overlay.sh` builds from a `git archive` of the commit, never from a
working tree, and aborts if the commit changes any file the Dockerfile does
not copy. The built tag is a local name only; the image was not published
under `localinferencelab/`.

## Run

Set `VLLM_PLE_TABLE_MEMORY=shared` instead of `VLLM_PLE_CPU_OFFLOAD=1`, mount
the host's `/dev/shm` at `/dev/shm`, keep `ipc: host` / `hostIPC: true` and
`IPC_LOCK`. Everything else is the TP1 recipe. See the "Multiple TP1 replicas
with one shared PLE table" section of the [recipe page](../../qwen38-flash-next.md),
the [Compose profile](../qwen38-flash-next.compose.yml) (`--profile tp1-shared`)
and the [Kubernetes example](../k8s/).
