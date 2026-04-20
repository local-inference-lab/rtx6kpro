#!/usr/bin/env bash
set -euo pipefail
python3 scripts/benchmark_glm_variants.py --variants dense_mla nsa --runs 30
