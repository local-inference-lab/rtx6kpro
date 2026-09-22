#!/usr/bin/env bash
# Reconstruct the reviewed source trees and verify their immutable identities.
set -euo pipefail
readonly destination=${1:?Provide an absent destination directory}
if [[ -e $destination ]]; then
  echo 'The source destination must not exist; existing checkouts are never modified.' >&2
  exit 1
fi
mkdir -p "$destination"
readonly root=$(realpath "$destination")

compose() {
  local name=$1 branch=$2 base=$3 expected=$4 canonical=$5
  shift 5
  local repository=https://github.com/local-inference-lab/$name.git
  git clone --filter=blob:none --no-checkout --single-branch --branch "$branch" \
    "$repository" "$root/$name"
  git -C "$root/$name" fetch origin "$base" "$canonical" "$@"
  git -C "$root/$name" checkout --detach "$base"
  local commit
  for commit in "$@"; do
    git -C "$root/$name" -c user.name='Source composition verifier' \
      -c user.email='source-verifier@localhost' -c commit.gpgsign=false \
      merge --no-ff --no-edit "$commit"
  done
  test "$(git -C "$root/$name" rev-parse 'HEAD^{tree}')" = "$expected"
  test "$(git -C "$root/$name" rev-parse "$canonical^{tree}")" = "$expected"
  # The tree is already verified. Pin the published commit for wheel versioning.
  git -C "$root/$name" checkout --detach "$canonical"
  test -z "$(git -C "$root/$name" status --porcelain)"
  printf '%s: verified tree %s\n' "$name" "$expected"
}

# vLLM #798 supplies caller-owned workspace lanes. The other PRs are disjoint.
compose vllm dev/karmic-kraken \
  9e5d1793fa34db4e672664711690e8a68d90fcd3 \
  311a3e608e0a217ddcd88ac92aed5abb98556c2c \
  9367f3712c2290cc9c6df1b12f4c5a5b7bc082e0 \
  3e7b8f223d9adfac967377f3e578d90d76c0b81e \
  ff32297843360093447d671f08b7b4e522aa2c4c \
  c49ef2e733bd6cbfdae71881e053d6539e75d33f \
  1cb34efae87ba351fdbf892da9c895b174d2fc3e \
  15b7b3c68de54ebfe7121829a38c26fa1e7727ee \
  120805fb4c0e69573181bf08c4d1a50df413e430 \
  3e3c24a8cee83e672a94560b162b8f0e4e64a71d

# B12X #384 retains prepared programs; #411, #412 and #413 supply Kimi paths.
compose b12x master \
  c2dc1cf02295b8241fc6a7728be7b6e8c23dda2f \
  3e05ed2429855ed818afcb3ef8d605f6eca6539e \
  93c1eef568cd90eb315df687bc3f3950b91b7526 \
  129458b2c2576979ecc5f4160bfd6610236ca578 \
  e0bebdc6cc24532325e8cac8dd61402b1d4efacb \
  9c3d35d44887c233857826e997fad592595c690d \
  d0ab281d0bb84ea41af57581e93b18de52ab6c95

git clone --filter=blob:none --no-checkout --single-branch \
  --branch agent/kk-kimi-prefill-overlap \
  https://github.com/local-inference-lab/blackwell-llm-docker.git "$root/docker"
git -C "$root/docker" checkout --detach 53c52e71e74fd81387acaafa9d0b69ebfde12cfe
printf 'Sources ready in %s\n' "$root"
