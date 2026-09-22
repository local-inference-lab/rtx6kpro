#!/usr/bin/env bash
# Assemble verified component wheels without recompiling FlashInfer or NCCL.
set -euo pipefail
readonly sources=$(realpath "${1:?Provide the verified source directory}")
readonly artifacts=$(realpath "${2:?Provide the extracted component archive}")
readonly output=${3:?Provide an absent runtime bundle directory}
readonly image=${4:-local/kimi-k3:qsrt-tp9-reproduced}
readonly python=${PYTHON:-python3}
test ! -e "$output"
test "$(git -C "$sources/docker" rev-parse HEAD)" = 53c52e71e74fd81387acaafa9d0b69ebfde12cfe
test "$(git -C "$sources/vllm" rev-parse 'HEAD^{tree}')" = 311a3e608e0a217ddcd88ac92aed5abb98556c2c
test "$(git -C "$sources/b12x" rev-parse 'HEAD^{tree}')" = 3e05ed2429855ed818afcb3ef8d605f6eca6539e
for component in nccl flashinfer b12x vllm lmcache instanttensor; do
  (cd "$artifacts/components/$component" && sha256sum --check SHA256SUMS)
done
readonly recipe=$sources/docker/tools/jovian_wheel_runtime
"$python" "$recipe/assemble_qwen38_runtime_bundle.py" \
  --output "$output" --qwen-lock "$recipe/qwen38-runtime.lock" \
  --ngc-foundation-lock "$recipe/foundation.lock" \
  --nccl-bundle "$artifacts/components/nccl" \
  --flashinfer-bundle "$artifacts/components/flashinfer" \
  --b12x-bundle "$artifacts/components/b12x" \
  --vllm-bundle "$artifacts/components/vllm" \
  --lmcache-bundle "$artifacts/components/lmcache" \
  --instanttensor-bundle "$artifacts/components/instanttensor" \
  --auxiliary-cache "$artifacts/auxiliary-cache"
test "$(sha256sum "$output/manifest.json" | awk '{print $1}')" = cea03a81beadf4c0e69de904aa817656a2f42d9bee55c334667c5c2de364402c
"$python" "$recipe/build_runtime_image.py" "$output" "$image"
