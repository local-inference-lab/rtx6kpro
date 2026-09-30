#!/usr/bin/env bash
# Serve eight DSpark K7 requests with 5.6 GB of KV storage per GPU.
set -euo pipefail
export KIMI_IMAGE=${KIMI_IMAGE:-local/kimi-k3:x4t-dspark-k7-cc8-reproduced-20260930}
export KIMI_CONTAINER=${KIMI_CONTAINER:-kimi-k3-x4t-dspark-k7-cc8-reproduced}
export KIMI_CUDA_GRAPH_MODE=${KIMI_CUDA_GRAPH_MODE:-FULL_DECODE_ONLY}
export KIMI_GRAPH_TOKENS=${KIMI_GRAPH_TOKENS:-7,14,28,56,64}
export KIMI_ALLOCATOR_CONFIG=${KIMI_ALLOCATOR_CONFIG:-expandable_segments:True,large_segment_size_mb:12,graph_capture_record_stream_reuse:True}
export KIMI_MAX_SEQS=8
export KIMI_KV_BYTES=${KIMI_KV_BYTES:-5600000000}
exec bash "$(dirname "${BASH_SOURCE[0]}")/start-x4t-dspark-k7.sh"
