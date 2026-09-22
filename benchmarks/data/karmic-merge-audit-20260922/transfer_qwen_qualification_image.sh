#!/usr/bin/env bash
set -euo pipefail

# Transfer the exact automatic-build artifact while registry upload proceeds.
# A later registry pull must reproduce the recorded image config ID.
evidence_dir=${1:?Set an existing evidence directory}
vllm_commit=${2:?Set the expected vLLM commit}
b12x_commit=${3:?Set the expected B12X commit}
recipe_commit=${4:?Set the expected recipe commit}
for source_commit in "$vllm_commit" "$b12x_commit" "$recipe_commit"; do
  [[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
done
test -d "$evidence_dir"
test ! -e "$evidence_dir/transferred-image-id.txt"
builder_ssh=(ssh -o BatchMode=yes -o HostKeyAlias=192.168.66.14 root@10.66.66.14)
server_ssh=(ssh -o BatchMode=yes -o HostKeyAlias=192.168.66.4 root@192.168.0.115)
builder_docker='runuser -u github-flashinfer -- env DOCKER_HOST=unix:///run/lil-flashinfer-docker/docker.sock docker'
assembly_path=/var/lib/github-flashinfer/work-blackwell-llm-docker/_temp/community-assembly-karmic-kraken-beta.json
deadline=$((SECONDS + 1800))
while true; do
  if "${builder_ssh[@]}" "test -f $assembly_path && cat $assembly_path" > "$evidence_dir/observed-assembly.json" &&
    jq -e --arg v "$vllm_commit" --arg b "$b12x_commit" --arg r "$recipe_commit" \
      '.components.vllm.source_commit == $v and .components.b12x.source_commit == $b and .recipe_commit == $r' \
      "$evidence_dir/observed-assembly.json" >/dev/null; then
    break
  fi
  ((SECONDS < deadline)) || exit 1
  sleep 5
done
image=$(jq -r .image "$evidence_dir/observed-assembly.json")
[[ "$image" =~ ^ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-[0-9]{8}-[0-9a-f]{16}$ ]]
while ! "${builder_ssh[@]}" "$builder_docker image inspect $image" \
  > "$evidence_dir/builder-image-inspect.json" 2>/dev/null; do
  ((SECONDS < deadline)) || exit 1
  sleep 5
done
builder_id=$(jq -r '.[0].Id' "$evidence_dir/builder-image-inspect.json")
[[ "$builder_id" =~ ^sha256:[0-9a-f]{64}$ ]]
config_id=$builder_id
# The builder's containerd store identifies an image by its OCI manifest;
# classic Docker identifies it by the manifest's config object instead.
builder_blob=/var/lib/github-flashinfer/.local/share/docker/containerd/daemon/io.containerd.content.v1.content/blobs/sha256/${builder_id#sha256:}
if "${builder_ssh[@]}" "test -f $builder_blob"; then
  "${builder_ssh[@]}" "cat $builder_blob" > "$evidence_dir/builder-oci-object.json"
  test "$(sha256sum "$evidence_dir/builder-oci-object.json" | cut -d' ' -f1)" = "${builder_id#sha256:}"
  if jq -e '.config.digest != null' "$evidence_dir/builder-oci-object.json" >/dev/null; then
    config_id=$(jq -r .config.digest "$evidence_dir/builder-oci-object.json")
  fi
fi
[[ "$config_id" =~ ^sha256:[0-9a-f]{64}$ ]]
printf 'Transferring %s (%s)\n' "$image" "$config_id"
"${builder_ssh[@]}" "$builder_docker image save $image" | "${server_ssh[@]}" 'docker image load'
remote_id=$("${server_ssh[@]}" "docker image inspect --format '{{.Id}}' $image")
test "$remote_id" = "$config_id"
"${server_ssh[@]}" "docker image inspect $image" > "$evidence_dir/server-image-inspect.json"
jq -e -n --slurpfile builder "$evidence_dir/builder-image-inspect.json" \
  --slurpfile server "$evidence_dir/server-image-inspect.json" \
  '$builder[0][0].RootFS == $server[0][0].RootFS' >/dev/null
printf '%s\n' "$remote_id" > "$evidence_dir/transferred-image-id.txt"
printf 'Exact build artifact transferred. Registry digest verification remains required.\n'
