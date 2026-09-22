#!/usr/bin/env bash
# Assemble verified component wheels without recompiling FlashInfer or NCCL.
set -euo pipefail
readonly sources=$(realpath "${1:?Provide the verified source directory}")
readonly artifacts=$(realpath "${2:?Provide the extracted component archive}")
readonly output=${3:?Provide an absent runtime bundle directory}
readonly image=${4:-local/kimi-k3:qsrt-tp9-reproduced}
readonly python=${PYTHON:-python3}
readonly vllm_bundle=$(realpath "${5:-$artifacts/components/vllm}")
test ! -e "$output"
test "$(git -C "$sources/docker" rev-parse HEAD)" = 53c52e71e74fd81387acaafa9d0b69ebfde12cfe
case "$(git -C "$sources/vllm" rev-parse 'HEAD^{tree}')" in
  311a3e608e0a217ddcd88ac92aed5abb98556c2c)
    expected_manifest=cea03a81beadf4c0e69de904aa817656a2f42d9bee55c334667c5c2de364402c ;;
  d707a4125f953444e8024afd3cf88ee4239eaf65)
    expected_manifest=6e996e7914bbc2a9dd516e4967134e7ce3f1021d391c319fa9a23349e555da32 ;;
  *) echo 'The vLLM source tree is not a pinned TP9 composition.' >&2; exit 1 ;;
esac
test "$(git -C "$sources/b12x" rev-parse 'HEAD^{tree}')" = 3e05ed2429855ed818afcb3ef8d605f6eca6539e
for component in nccl flashinfer b12x lmcache instanttensor; do
  (cd "$artifacts/components/$component" && sha256sum --check SHA256SUMS)
done
(cd "$vllm_bundle" && sha256sum --check SHA256SUMS)
readonly recipe=$sources/docker/tools/jovian_wheel_runtime
"$python" "$recipe/assemble_qwen38_runtime_bundle.py" \
  --output "$output" --qwen-lock "$recipe/qwen38-runtime.lock" \
  --ngc-foundation-lock "$recipe/foundation.lock" \
  --nccl-bundle "$artifacts/components/nccl" \
  --flashinfer-bundle "$artifacts/components/flashinfer" \
  --b12x-bundle "$artifacts/components/b12x" \
  --vllm-bundle "$vllm_bundle" \
  --lmcache-bundle "$artifacts/components/lmcache" \
  --instanttensor-bundle "$artifacts/components/instanttensor" \
  --auxiliary-cache "$artifacts/auxiliary-cache"
test "$(sha256sum "$output/manifest.json" | awk '{print $1}')" = "$expected_manifest"
case "${KIMI_BUILD_IMAGE:-1}" in
  1) "$python" "$recipe/build_runtime_image.py" "$output" "$image" ;;
  0) printf 'Verified runtime bundle: %s\n' "$output" ;;
  *) echo 'KIMI_BUILD_IMAGE must be 0 or 1.' >&2; exit 2 ;;
esac
