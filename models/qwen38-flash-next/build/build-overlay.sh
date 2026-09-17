#!/usr/bin/env bash
# Build the shared-PLE-table overlay image from a git archive of the fork
# branch, never from a working tree.  Usage:
#
#   VLLM_FORK=/path/to/jovian-vllm ./build-overlay.sh [<commit-ish>]
#
# The image is tagged localinferencelab/vllm:jovian-r35-shared-ple-<short-sha>
# (override with IMAGE_TAG).  Push or `docker save` it afterwards; the digest
# to pin in manifests is printed at the end.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fork=${VLLM_FORK:-"$here/../../../../jovian-vllm"}
ref=${1:-build/qwen38-shared-ple-r35}
base_commit=de982a50c6a3e4718e5cf9f00423a92192718da1
files=(
  vllm/envs.py
  vllm/models/qwen3_8_flash_next/model.py
  vllm/models/qwen3_8_flash_next/ple_layer.py
  vllm/models/qwen3_8_flash_next/ple_shared_table.py
  tests/models/test_qwen3_8_flash_next_ple_shared_table.py
)

commit=$(git -C "$fork" rev-parse --verify "${ref}^{commit}")
short=$(git -C "$fork" rev-parse --short=12 "$commit")
tag=${IMAGE_TAG:-"localinferencelab/vllm:jovian-r35-shared-ple-${short}"}

# The overlay must touch exactly the files copied by the Dockerfile.
changed=$(git -C "$fork" diff --name-only "$base_commit" "$commit" | sort)
expected=$(printf '%s\n' "${files[@]}" | sort)
if [ "$changed" != "$expected" ]; then
  echo "overlay $commit changes unexpected files relative to $base_commit:" >&2
  diff <(echo "$expected") <(echo "$changed") >&2 || true
  exit 1
fi

sha() { git -C "$fork" show "$commit:$1" | sha256sum | cut -d' ' -f1; }

context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
git -C "$fork" archive --format=tar "$commit" -- "${files[@]}" | tar -x -C "$context"
cp "$here/Dockerfile" "$context/Dockerfile"

DOCKER_BUILDKIT=1 docker build \
  --build-arg VLLM_OVERLAY_COMMIT="$commit" \
  --build-arg VLLM_OVERLAY_BRANCH="$ref" \
  --build-arg VLLM_BASE_COMMIT="$base_commit" \
  --build-arg ENVS_SHA256="$(sha vllm/envs.py)" \
  --build-arg MODEL_SHA256="$(sha vllm/models/qwen3_8_flash_next/model.py)" \
  --build-arg PLE_LAYER_SHA256="$(sha vllm/models/qwen3_8_flash_next/ple_layer.py)" \
  --build-arg PLE_SHARED_TABLE_SHA256="$(sha vllm/models/qwen3_8_flash_next/ple_shared_table.py)" \
  --build-arg TEST_SHA256="$(sha tests/models/test_qwen3_8_flash_next_ple_shared_table.py)" \
  -t "$tag" "$context"

echo "built $tag from $commit"
docker image inspect "$tag" --format 'image id: {{.Id}}'

# The serve CLI needs a visible GPU to build its argument parser; check it on
# a GPU host so a broken overlay never reaches a manifest.
if [ -n "${CHECK_GPU:-}" ]; then
  docker run --rm --gpus "device=${CHECK_GPU}" --entrypoint /bin/bash "$tag" -c \
    'test "$(/opt/venv/bin/python -m vllm.entrypoints.cli.main serve --help | grep -c additional-config)" -gt 0 \
     && git -C /opt/glm53-flash/vllm status --short && cat /opt/glm53-flash/overlay.identity'
fi
echo "push it (docker push $tag) and pin the RepoDigest it reports, or"
echo "docker save $tag | sudo k3s ctr images import - on the node."
