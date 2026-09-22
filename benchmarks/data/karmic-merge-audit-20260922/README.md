# Canonical merge and registry validation evidence

These artifacts support the [canonical PR audit](../../karmic-integration-merge-audit.md)
and the [public-report validation](../../karmic-public-feedback.md).

## Registry composition with distributed tuning compatibility

The `qwen-tuning-registry-*` files identify
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-3221ccacf71002ea`,
digest `sha256:bf2c8f5da6f1de82b345367102b528692231a23eadc1b49c180c82c397c77522`.
The official assembly, publication receipt, manifest and generated changelog
are under `tuning-registry-release/`. They contain vLLM `622912b9b2c`,
B12X `3113aa0b859` and recipe `338546b0e79`.

The exact image was transferred from the automatic builder while its registry
upload ran. `tuning-registry-pull-identity.log` records the subsequent immutable
registry pull and matching config ID in both serving containers.
`builder-oci-manifest.json` maps the OCI digest to Docker config
`sha256:5cb7de41ecce821b2c33880b2787254c602aed86637c64ee58c77e701fd68db9`.
All 68 rootfs layers match; the different image IDs reported by containerd and
classic Docker refer to the manifest and its config, not different images.
No serving sources were mounted or edited.

The five `qwen-tuning-registry-matched-*.json` repeats give initial medians
222.330/906.795 output tok/s and 95.738/389.077 verifier steps/s at C1/C8.
Every cell passes its error, loop and occupancy checks. Three uncached prefill
windows contain twelve requests each, measuring 14547/14570/14586 tok/s.
All sixteen mixed text/JSON-schema requests and twelve distinct-code cache
requests pass. The original prompt restores 2880 external tokens with zero
native hits, returning TEAL-428 and the sum 13.

The fixed five-run `qwen-tuning-registry-confirmation-*` series follows the
prefill and correctness checks on the unchanged server. It is recorded
separately rather than replacing initial lower valid samples or pooling two
different startup states. `qualify_glm_after_qwen.sh` defines this ordering
and stops Qwen before the four-GPU GLM text/vision check.
All ten confirmation cells pass; medians are 222.565/905.134 output tok/s
and 95.809/389.319 verifier steps/s. Prefill warmup does not remove the lower
C1 output observation relative to the preceding measured image.

`glm-tuning-registry-smoke/` contains seven passing text/prefix/vision checks
on the same immutable digest. GLM uses TP4/DCP1/DFlash2 K7, a 65,536-token
context, eight sequences and twelve GiB GPU KV per rank. The process audit
verifies the probabilistic draft/standard rejection configuration and the
checkpoint snapshots. The startup log records graph capture. This is not
a GLM throughput benchmark or external-LMCache test.

`vllm-handoff-pr-status.json` and `b12x-handoff-pr-status.json` confirm that the
ten vLLM and four B12X PRs remain open, non-draft, correctly targeted and at
the exact heads used in the complete-tree proofs. The corresponding canonical
branches also remained at the audited commits during this confirmation.

## Distributed tuning result compatibility

`qwen-ple-registry-*-startup.log` records failed TP2/MTP3 starts of image
`e5fef1d4361a77ea`, whose immutable receipts are in `ple-registry-release/`.
The image's package tests passed, but actual serving found that canonical
vLLM expects `TuningRequirement.rejected_count` and B12X did not expose it.
These files are failure evidence, not throughput measurements.

B12X #414 supplies a validated default-zero result field without changing
kernel arithmetic or winner selection. The `tuning-result-contract-before.log`
contains five failing contract tests. The separate B12X and vLLM `*-after.log`
files contain 61 and 29 passing host tests. The `tuning-package-gate-*` records
show the installed-package mismatch failing, both compatible field protocols
passing, and 506 publisher/launcher tests passing with one skip.

`b12x-tuning-contract-final-check.json` adds #414 to the ordered PR composition.
Its integration commit is `3113aa0b8596fe55a96a28d58efd40f4dfa9955c`; the exact
tree is `67c96e44da899e4660c38905724cb1943f8ab768`. The matching vLLM audit is
`vllm-handoff-final-check.json`. Both have zero residual differences.

## Registry composition with QSA pre-launch guards

The `qwen-qsa-guards-registry-*` artifacts use
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-4d9905b931656635`,
digest `sha256:9625bbc047d3140ce55bb244fdd3ddf1e2911cc5d8ae109e8a5c1a6675dbb575`.
Its image configuration is
`sha256:ff88c64f7b7a222f6043e2bd753503a09c02bbaf613b9090b3ff64773e06fdd2`.
The complete release receipts are under `qsa-guards-registry-release/`.
Deployment and GPU-monitor files record the serving process and physical pair;
no source overlays or CPU-affinity overrides are used.

`qwen-qsa-guards-registry-native-restore.jsonl` records twelve successful
distinct-code checks. Restoration returns 2880 external tokens with zero GPU
hits and recovers the original code and sum. Its cache server is stopped before
decode measurements on the other pair begin.

`qwen-qsa-guards-registry-matched-{1..5}.json` records five valid warmed
C1/C8 repeats: medians 229.269/902.425 output tok/s and 96.309/388.140 verifier
steps/s. The three `prefill32k-{1..3}.json` windows contain twelve requests
each and measure 15000, 14680 and 14544 tok/s. The mixed-request file records
16 correct text/JSON-schema responses. The benchmark client's hardware
inventory is not the remote server's inventory; use the separate server GPU
clock record for hardware conditions.

The publication-order reproducer and validation logs are
`changelog-publication-order-{before,after}.log` and
`changelog-publication-order-live-check.json`. Three regressions fail before
the publisher correction; 85 tests pass after it. The read-only live check
selects the 05:06 publication, not the 02:35 publication returned first by the
release API. [Publisher PR #65](https://github.com/local-inference-lab/blackwell-llm-docker/pull/65)
does not change inference code or rewrite published manifests. The existing
release's changelog remains accurate against its explicitly named, broader
comparison baseline.

## Registry composition with canonical metadata and MoE updates

The `qwen-final-registry-*` files use
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-04a3c00a18b9d45f`,
digest `sha256:f30243cff149bdde2cc7eb43e90e2839a6c89c0d30f9364935267570a145fc45`.
`canonical-registry-release/` contains its build receipt, component manifests
and generated changelog. Neither serving process uses source overlays or
CPU-affinity overrides; the configured Torch profiler was never triggered.

- `qwen-final-registry-unprofiled-{1..5}.json`: five warmed C1/C8 repeats.
  Median output is 228.803/909.260 tok/s, with no failed cells in this series.
- `qwen-final-registry-restart-{1,2}.json`: separate restart checks. The
  second C8 cell is **ERROR: loop**, not a valid speed sample. Retain it when
  assessing correctness; it must not disappear from a successful-run median.
- `qwen-final-registry-eos-c8-{1,2}.json`: two separate controls respecting
  EOS. Both pass, but their different stopping semantics and small sample
  do not establish the cause of the failed restart cell.
- `qwen-final-registry-prefill32k.json` and
  `qwen-final-registry-restart-prefill32k-{1,2}.json`: three uncached windows,
  twelve requests each; rates 14,502, 14,619 and 14,674 tok/s.
- `qwen-final-registry-mixed.jsonl`: 16 correct text/JSON-schema requests.
- `qwen-final-registry-native-restore.jsonl`: strict prefix recall after ten
  intervening prompts with distinct access codes; 2880 external tokens,
  zero native GPU hits, correct original code and sum.
- `qwen-final-registry-deployment.txt`: actual image identity, mounts and
  selected communication backend. `qwen-final-unprofiled-gpu-clocks.csv`
  records server GPUs, independently of the benchmark client's inventory.

`vllm-registry-final-check.json` and `b12x-registry-final-check.json` are
the full-tree source proofs for this registry image. To repeat a proof, use
the included `audit_integration_prs.py` with a local component checkout,
`--github-repo local-inference-lab/vllm` or `local-inference-lab/b12x`,
`--canonical` and `--integration` set to the recorded commits, and `--prs`
set to the ordered list in the report. The script uses `git merge-tree` and
creates audit objects; it does not change a checkout or push a branch.

The strict native restore reproducer is `probe_native_cpu_restore.py`:

```bash
OFFLOAD_BASE=http://SERVER:5075 STRICT_RECALL=1 EVICT_COUNT=10 \
PRIME_MARKER=zeta EVICT_PREFIX=eta \
PROBE_QUESTION='Return the access code and the result of 11 plus 2.' \
python probe_native_cpu_restore.py
```

Use the separate native-cache launcher and its small cache budget. A GPU
prefix hit does not pass this test. The script asserts distinct-code recall,
positive external hits and zero GPU hits on the final restore.

## Source composition

`vllm-canonical-85a78-final-check.json` freezes canonical vLLM
`85a78f57a0e4d825f19b9cff243068d9a3aac7b2` and integration
`622912b9b2cfd04784da5b0dfc69ce8aa5aee02d`. The ordered ten PRs produce
the identical whole tree `c38e7c35300f96d016ddd2611019bf55a435b624`, with
no conflicting merges or residual files. PR #798 preserves independent DFlash
context-projection ownership while adopting canonical PLE and draft loading.

`canonical-85a78-serving-contracts.log` records 222 passing component tests.
Five cases initially fail because the test container exposes one GPU and the
default Docker seccomp policy blocks disk I/O. They are not serving failures:
`canonical-85a78-gpu-policy-controls.log` reruns them with two assigned GPUs and
disk-I/O permission, plus eight MHC graph-replay cases; all 13 pass. This gives
235 distinct passing tests without changing expectations or skipping a failure.
The separate four-case DFlash ownership log repeats a subset of that coverage.
These source tests do not substitute for registry serving checks.

`vllm-final-review-check.json` and `b12x-final-review-check.json` record the
earlier PR reconstruction after the QSA boundary review. Both have identical
simulation/integration trees, no conflicting PRs and no residual files. The
associated `*-merge-pr-status.json` files record the open, non-draft PRs and
their canonical target branches. The B12X integration commit is
`d16e71c3da2a13c2d73370fba5b1149c95428fbd`; its tree is
`895617e7f076eb6b099f450a947f36627d42356f`.

The QSA boundary correction rejects zero planned local score capacity, invalid
query-position geometry before expansion, and the DCP-only reuse API called
with DCP1. Valid live requests may still leave ranks empty.
`b12x-qsa-review-before.log` records seven failing negative reproducers against
the unmodified implementation. `b12x-qsa-review-after-gpu.log` records 19 passing
geometry/capacity checks after the correction, including the CUDA numerical
oracle with live-empty ranks. `b12x-qsa-review-graph-reuse.log` records four
passing graph/pool/reuse tests with the state-owned capacity references.
These are qualified contract checks, not a full Qwen DCP serving matrix.

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

## Metadata-fix registry image

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

## Historical source-overlay diagnostics

`qwen-moe-refresh-matched-{1,2}.json` and
`qwen-moe-refresh-after-profile-{1,2}.json` record one unchanged source-overlay
process before and after both an uncached prefill and a Torch capture.
Its C1 verifier rate changes from about 88 to 96 steps/s. Because both
interventions occurred, these files do not isolate the profiler as the cause.
The `cuda-profiler-graph-state.jsonl` microprobe does not reproduce a speedup
from profiler construction or stop. These observations are not a shipped
optimization and do not justify a NUMA/PLE allocator workaround.

The source-unmodified registry measurements above begin at about 96 steps/s
without either intervention and retain that rate after a container restart.
The repetition failure in the restart series is a separate correctness
observation; verifier throughput cannot exonerate a repeated output.
