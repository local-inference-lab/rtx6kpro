#!/usr/bin/env bash
# hf-dedup-xet-blobs.sh
#
# Hardlink XET (content-addressed) blobs from one HF Hub repo cache into another,
# so that `hf download` skips files the two repos share instead of re-downloading.
#
# Background:
#   HF Hub stores blobs per-repo in models--<org>--<name>/blobs/. Even when two
#   repos share files (a quantization plus its MTP head, a fork, ...), the client
#   has no cross-repo lookup, so it pulls them again. With XET storage the blob
#   filename IS the SHA256 of the content -- identical files have identical names
#   across repos, which makes safe hardlinking trivial.
#
# Usage:
#   hf-dedup-xet-blobs.sh [--dry-run] [--datasets] <source-repo-id> <target-repo-id>
#
# Example:
#   hf-dedup-xet-blobs.sh lukealonso/GLM-5.1-NVFP4 lukealonso/GLM-5.1-NVFP4-MTP
#   hf download lukealonso/GLM-5.1-NVFP4-MTP   # now only the truly new files

set -euo pipefail

DRY_RUN=0
KIND="models"
SCRIPT_NAME=$(basename "$0")

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] <source-repo-id> <target-repo-id>

Hardlink XET blobs from <source> repo cache into <target> repo cache so that
'hf download <target>' reuses the shared files instead of pulling them again.

Repos must be in <org>/<name> form. Both caches must already be on the same
filesystem (hardlinks cannot cross filesystems). Source must already be
downloaded; target may be empty or partially downloaded.

OPTIONS:
  --dry-run      Show what would happen without changing anything.
  --datasets     Operate on the datasets cache (datasets--*) instead of models.
  -h, --help     Show this help.

ENVIRONMENT:
  HF_HUB_CACHE   Cache root. Default: \${HF_HOME:-~/.cache/huggingface}/hub

WHAT IT DOES:
  1. For every 64-char-hex (XET / SHA256) blob in source/blobs that is missing
     in target/blobs, create a hardlink. Existing target blobs are not touched.
  2. For every <hash>.incomplete in target/blobs that now has a final blob,
     remove the orphan .incomplete file.
  3. Non-XET blobs (40-char git-LFS SHA1, configs, README, .gitattributes) are
     skipped -- their content typically differs between repos.

EXAMPLE:
  $SCRIPT_NAME lukealonso/GLM-5.1-NVFP4 lukealonso/GLM-5.1-NVFP4-MTP
  hf download lukealonso/GLM-5.1-NVFP4-MTP
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --datasets) KIND="datasets"; shift ;;
    -h|--help)  usage; exit 0 ;;
    --)         shift; break ;;
    -*)         echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)          break ;;
  esac
done

[[ $# -eq 2 ]] || { usage >&2; exit 2; }

SRC_ID="$1"
DST_ID="$2"

[[ "$SRC_ID" == "$DST_ID" ]] && die "source and target are the same repo"

CACHE_ROOT="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

repo_to_dir() {
  local id="$1"
  [[ "$id" == */* ]] || die "repo id must be <org>/<name>: $id"
  local org="${id%%/*}"
  local repo="${id#*/}"
  echo "$CACHE_ROOT/${KIND}--${org}--${repo}"
}

SRC_DIR="$(repo_to_dir "$SRC_ID")"
DST_DIR="$(repo_to_dir "$DST_ID")"
SRC_BLOBS="$SRC_DIR/blobs"
DST_BLOBS="$DST_DIR/blobs"

[[ -d "$SRC_BLOBS" ]] || die "source blobs dir not found: $SRC_BLOBS
(have you run 'hf download $SRC_ID' yet?)"

if [[ ! -d "$DST_BLOBS" ]]; then
  echo "Target $DST_BLOBS missing -- will create it."
  if (( DRY_RUN == 0 )); then
    mkdir -p "$DST_BLOBS"
  fi
fi

# Hardlinks cannot cross filesystems. Check before doing anything.
if [[ -d "$DST_BLOBS" ]]; then
  src_dev=$(stat -c '%d' "$SRC_BLOBS")
  dst_dev=$(stat -c '%d' "$DST_BLOBS")
  [[ "$src_dev" == "$dst_dev" ]] || die "source and target are on different filesystems (dev $src_dev vs $dst_dev) -- hardlinks impossible"
fi

XET_RE='^[0-9a-f]{64}$'

linked=0
already=0
skipped_nonxet=0
removed_inc=0
total_bytes_dedup=0

# 1. Hardlink missing XET blobs from source -> target.
while IFS= read -r -d '' blob; do
  name=$(basename "$blob")
  if ! [[ "$name" =~ $XET_RE ]]; then
    skipped_nonxet=$((skipped_nonxet+1))
    continue
  fi
  if [[ -e "$DST_BLOBS/$name" ]]; then
    already=$((already+1))
    continue
  fi
  size=$(stat -c '%s' "$blob")
  if (( DRY_RUN )); then
    printf '[dry-run] ln %s  (%s)\n' "$name" "$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size B")"
  else
    ln "$blob" "$DST_BLOBS/$name"
  fi
  linked=$((linked+1))
  total_bytes_dedup=$((total_bytes_dedup+size))
done < <(find "$SRC_BLOBS" -maxdepth 1 -type f ! -name '*.incomplete' -print0)

# 2. Remove orphan <hash>.incomplete files in target that now have a final blob.
while IFS= read -r -d '' inc; do
  h=$(basename "$inc" .incomplete)
  if [[ -f "$DST_BLOBS/$h" ]]; then
    if (( DRY_RUN )); then
      printf '[dry-run] rm %s\n' "$(basename "$inc")"
    else
      rm "$inc"
    fi
    removed_inc=$((removed_inc+1))
  fi
done < <(find "$DST_BLOBS" -maxdepth 1 -type f -name '*.incomplete' -print0)

human=$(numfmt --to=iec --suffix=B "$total_bytes_dedup" 2>/dev/null || echo "$total_bytes_dedup B")

echo
echo "==================================================="
(( DRY_RUN )) && echo "DRY RUN -- nothing changed."
echo "Source : $SRC_ID"
echo "Target : $DST_ID"
echo "Cache  : $CACHE_ROOT"
echo "---------------------------------------------------"
echo "Hardlinked         : $linked blobs ($human saved)"
echo "Already in target  : $already"
echo "Removed orphan .incomplete : $removed_inc"
echo "Skipped non-XET    : $skipped_nonxet (configs, README, ...)"
echo "==================================================="
if (( DRY_RUN )); then
  echo "Re-run without --dry-run to apply."
else
  echo "Next: hf download $DST_ID"
fi
