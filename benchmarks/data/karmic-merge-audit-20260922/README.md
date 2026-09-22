# Canonical merge and registry validation evidence

These artifacts support the [canonical PR audit](../../karmic-integration-merge-audit.md)
and the [public-report validation](../../karmic-public-feedback.md).

## Source composition

`vllm-expected.json` and `b12x-expected.json` contain canonical commits,
actual PR heads, ordered merge simulations and complete-tree comparisons.
The `qwen-canonical-matched-*` cells use sharded HyperConnection projections;
`qwen-replicated-matched-*` uses the same composition with replication enabled.
The matching rank-0 traces capture eight C8 target steps. Their source overlays
are diagnostic, not a published image.

`vllm-metadata-refresh-check.json` and `b12x-moe-refresh-check.json` record
the composition after importing canonical commits `a18246b0626` and
`b294e69d`. These are the source-equality checks for published integration
heads `47cb3450b11` and `c4349457`. The corresponding focused test log records
61 GDN and 183 B12X tests, including CUDA execution. The standalone
`audit_integration_prs.py` script requires Python, Git and authenticated `gh`;
it creates audit objects without changing the checkout or any branch.

`qwen-numa-matched-*` uses the same complete source composition with replicated
projections and GPU-local CPU/host-memory binding at startup. Five repeats,
mixed-request checks and prefill are retained. `qwen-remote-cpu-matched-*` moves
only the loaded engine's CPU threads to the remote NUMA node; host memory stays
GPU-local. `qwen-reference-hc-matched-*` is a separate two-run file-substitution
diagnostic, not an implementation proposed for merge. These controls must not
be silently combined into one benchmark configuration.

The `qwen-local-allocation-matched-*` control constrains CPU affinity only
during the mapped PLE allocation and restores it immediately. The
`qwen-membind-matched-*` control instead uses Docker `--cpuset-mems 1` with
unrestricted CPU scheduling. Neither restores the fast rate. Both use the
same `4717b12198a` / `2879cb0234` Python sources.

The `qwen-membind-cpu-node1-matched-1.json` cell then constrains the engine
and worker threads on that same loaded server. The `cpu-all`, `cpu-0`,
`cpu-split` and `thread-split` C1 files record subsequent affinity controls;
they must not be interpreted as separate source changes or as repeated
independent startups. The complete NUMA startup recipe remains distinct.

## Published image

The `qwen-registry-*` files use the unmodified image
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-97197f5085f4798b`,
digest `sha256:9d2431fa46c10b429fd98eeeb8aea013ba12e2010303402203c569ea09ab7e51`.
No model source bind mounts or in-container source edits were used.

- `qwen-registry-matched-{1..5}.json`: five warmed C1/C8 decode repeats.
- `qwen-registry-prefill32k.json`: uncached 32k prefill, 30-second window.
- `qwen-registry-mixed-smoke.jsonl`: 16 text/JSON-schema checks at C4.
- `qwen-registry-native-restore.jsonl`: real CPU-prefix restore after ten
  other prompts; 2880 external hits, zero native GPU hits, correct code and sum.
- `start_qwen_registry_perf.sh` and `start_qwen_registry_native_cache.sh`:
  launch configurations, parameterized by the immutable registry image.

Hardware is the same pair of RTX PRO 6000 Blackwell Max-Q GPUs at stock clocks.
Decode uses TP2/DCP1/MTP3, temperature 1, top-p .95, top-k disabled and reasoning
effort medium. Native-cache correctness uses a separate, smaller cache budget
and the checkpoint's top-k default. It is not a cache performance benchmark.
The JSON's hardware inventory belongs to the benchmark client; server clocks
and devices are specified in the linked reports.
