# Karmic integration: canonical merge and registry qualification

The ordered PR composition below reproduces the complete integration
trees: **zero file differences** for vLLM and B12X, including tests, CI,
documentation and release fragments. Serving measurements retain their exact
image identities; a source reconstruction is not itself a generation test.

Status: **implemented** and **qualified** for the source reconstruction and
bounded checks described here. The source-overlay investigations are retained
as [historical diagnostics](data/karmic-merge-audit-20260922/README.md);
they are not substituted for the published-image measurements.

## Artifact boundary

The source audit below targets vLLM integration `622912b9b2c` after the
canonical draft/PLE update. The registry measurements in this section use
the preceding `47cb3450b11` integration; their identities are not interchangeable.

Image:
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-4d9905b931656635`

Digest:
`sha256:9625bbc047d3140ce55bb244fdd3ddf1e2911cc5d8ae109e8a5c1a6675dbb575`

| Component | Published revision |
|---|---|
| vLLM integration | `47cb3450b11aac466f03a933de576a37f970da53` |
| B12X integration | `d16e71c3da2a13c2d73370fba5b1149c95428fbd` |
| Shared runtime recipe | `2057ca2708f9cfdc3d45814ad84a7ced742df05b` |

Both source branches are `integration/karmic-kraken-beta`; the branch named
`integration/beta` belongs to Jovian Judgement and is not the audit target.
The [registry receipt and component manifest](data/karmic-merge-audit-20260922/qsa-guards-registry-release/)
bind the complete native dependencies and runtime to this image.

The beta publication job in [build 35694181770](https://github.com/local-inference-lab/blackwell-llm-docker/actions/runs/35694181770)
succeeded, including 94 cache/packaging tests and native GPU smoke. The separate canonical KK publication failed because canonical
B12X does not yet include #406. Merge the listed dependency before treating
the canonical channel as equivalent to this beta.

## Frozen inputs and merge order

| Component | Canonical input | PR order |
|---|---|---|
| vLLM | `dev/karmic-kraken`, `85a78f57a0e4d825f19b9cff243068d9a3aac7b2` | #798, #805, #813, #821, #822, #837, #834, #835, #836, #838 |
| B12X | `master`, `b294e69d8eba2ea56d2aed7cc359c0df4bcaa57d` | #393, #406, #409 |

Ordinary merge-commit composition preserves contributor histories and produces
these complete Git trees:

| Component | Canonical + PR tree, identical to integration | Remaining differences |
|---|---|---:|
| vLLM | `c38e7c35300f96d016ddd2611019bf55a435b624` | 0 |
| B12X | `895617e7f076eb6b099f450a947f36627d42356f` | 0 |

The [vLLM](data/karmic-merge-audit-20260922/vllm-canonical-85a78-final-check.json) and
[B12X](data/karmic-merge-audit-20260922/b12x-final-review-check.json) audit
records include every actual PR head, author, canonical base and intermediate
merge tree. All merges are conflict-free at these revisions. Commit IDs need
not match because the integration histories differ; all tracked file contents
do match. Changed PR heads or canonical tips require rerunning the audit.
Squash/rebase is not the merge operation simulated here.

### Consolidation and attribution

- vLLM #807 is closed: merged #809 already supplies its graph-accounting and
  recurrent-state ownership changes.
- vLLM #824 is closed and included in #821: compact KDA recovery and the
  gate/output alias correction share one branch. Merge #798's caller-owned
  workspace contracts first; Kimi retains caller-owned recovery records.
- vLLM #777 is replaced by #837, a direct KK port retaining Naadir Jeewa's
  positional-override implementation and attribution.
- vLLM #836 retains Zhewen Li's SimpleCPU multi-group/QSA scratch exclusion.
- vLLM #838 and B12X #393 include the integration release-fragment policy.
  Canonical-sync fragments describe canonical changes without modifying
  previously published fragments.
- B12X #406 uses master's schema version 7 and silicon-based tuning identity,
  while preserving the context-parallel QSA capacity contract. Its #405
  prerequisite is already merged. Reviewed guards reject invalid planned
  capacity and query metadata before scratch writes; DCP1 uses `run()` rather
  than the DCP-only `attend_reuse()` entry point.
- vLLM #798 retains canonical release-before-rebind coverage and the
  integration checks for selector-plan invalidation and live index-cache views.
  Its canonical synchronization preserves checkpoint-aware PLE placement,
  pooled draft loading and mixed-page cache layouts. The fused DFlash context
  projection still owns its quantization method independently of the query
  projection. Both loader selections have regression coverage.

Luke Alonso's metadata-copy fusion, Qwen projection-sharding capability,
NVFP4 decode tiles, runtime grid tuning and IQ2_XS support remain present.
The metadata fusion handles mixed batches; #835 skips unused worklists only
when a uniform decode graph is selected. These are complementary contracts.
No separate MiMo PR or independent experiment is added. MiMo support already
in the frozen canonical base remains present.

## Focused validation

| Conditions | Measurement | Result |
|---|---|---|
| PR #798 synchronized with canonical `85a78f57a0e`: DFlash, PLE, configuration, warmup, mixed-page packing and MHC replay | 235 distinct tests | Pass; five require two GPUs or disk-I/O permission and pass with those capabilities |
| Canonical noncausal/mixed/asymmetric attention, runtime layout and pooled-weight paths | 14 selected tests | Pass |
| Published composition: GDN metadata and graph contracts | 61 tests | Pass |
| Published composition: B12X tuning and swapped NVFP4, including CUDA | 183 tests | Pass |
| QSA boundary correction: invalid capacity, pre-write validation and live-empty DCP ranks | Seven negative reproducers fail before; 19 contract/oracle tests pass after | Qualified |
| QSA graph replay, draft-tail reuse, invalid-input rejection and smaller runtime pool | Four CUDA tests | Pass |
| Prior composition: configuration, SimpleCPU and GLM recovery | 81 tests | Pass |
| Prior composition: QSA/GDN/KDA alias and shared counts | 112 tests, nine skips | Pass |
| Prior composition: B12X geometry, preparation and RoCE | 63 tests | Pass |
| Prior composition: contiguous normalization, including graph replay | Eight CUDA tests | Pass |
| Prior composition: GLM selector release and rebinding | Two tests | Pass |
| Registry comparison image `04a3c00a18b9d45f`, TP2/MTP3, mixed text and JSON-schema requests at C4 | 16 requests, inputs up to 31,242 tokens | All codes and sums correct |
| Registry comparison image `04a3c00a18b9d45f`, SimpleCPU cache after ten GPU-evicting prompts with distinct codes | 2,880 externally restored tokens, zero GPU-prefix hits | Original code and sum correct; server healthy |
| Image-aware Compose exports for all nine documented deployments | 22 parser/runtime-parity tests | Pass |

The prior composition is vLLM `4717b12198a` / B12X `2879cb0234`.
Those focused checks are not represented as reruns on the final image.
The 61 GDN and 183 tuning/NVFP4 checks use vLLM `47cb3450` and B12X
`c4349457`; the subsequent 23 QSA checks cover the boundary/test correction
in B12X `d16e71c3`.
The registry serving tests use no source mounts or in-container code edits.

SimpleCPU validation uses TP2/MTP3, an 8192-token context, one GiB GPU KV per
rank and four GiB CPU offload. The original prompt contains 6,883 tokens.
Each intervening prompt has a different access code; restoration must recover
`TEAL-428` and the sum 13, rather than reuse an intervening request's state.
This qualifies live CPU restoration, not persistence after server shutdown.

## Qwen TP2 performance gate

Same physical pair of RTX PRO 6000 Blackwell Max-Q GPUs at stock clocks,
memory clock 13365 MHz, QAD checkpoint
`7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`, TP2/DCP1/MTP3,
temperature 1, top-p .95, top-k disabled and reasoning effort `medium`.
vLLM uses a 6019-token batch budget, 16 slots, eight GiB KV per rank and a
524288-token context limit. Decode uses the same 77-token input and the median
of five warmed 30-second windows. C8 is aggregate output.

| Deployed stack | C1 output tok/s | C1 steps/s | C8 output tok/s | C8 steps/s |
|---|---:|---:|---:|---:|
| Community SGLang, matched saved runs | 200.83 | 84.33 | 891.67 | 383.67 |
| Metadata-fix registry image `97197f5085f4798b` | 225.17 | 95.29 | 898.41 | 387.12 |
| Canonical-update registry image `04a3c00a18b9d45f` | 228.80 | 96.35 | 909.26 | 387.84 |
| QSA-guard registry image `4d9905b931656635` | 229.27 | 96.31 | 902.43 | 388.14 |

The QSA-guard image measures **+14.16% C1 / +1.21% C8** against SGLang.
All five C1/C8 repeats pass the error, loop and occupancy guards. Three uncached
32k windows measure 15,000, 14,680 and 14,544 tok/s: **median 14,680 tok/s**,
0.4% above the canonical-update image's three-window median. Sixteen mixed
requests and all twelve distinct-code CPU-restore requests pass. Client-side
hardware inventory in llmbench describes the client host; the separate GPU
monitor records the actual remote serving pair.

The canonical-update image measures **+13.93% C1 and +1.97% C8** against SGLang,
and **+1.61% C1 / +1.21% C8** against the metadata-fix registry image.
C8 is approximately matched to SGLang, not a large advantage. The five C1
step rates range from 96.29 to 96.40; output variation also reflects draft
acceptance. This five-run series has no request errors, loops or underfilled
concurrency. The Torch profiler was never started in this registry process,
and no CPU/NUMA binding was applied.

The canonical-update comparison uses digest
`sha256:f30243cff149bdde2cc7eb43e90e2839a6c89c0d30f9364935267570a145fc45`,
the same vLLM/runtime revisions as the artifact boundary above, and B12X
`c4349457b13905f9f1689240fcda5991e133de44`. The QSA boundary guards, associated
tests and release-policy wording distinguish the B12X revisions; no valid
DCP1 kernel arithmetic was changed by those guards.

Three 30-second uncached 32k prefill windows measure 14,502, 14,619 and
14,674 tok/s, twelve requests per window. The median is **14,619 tok/s**;
median input lengths are 32,118–32,119 tokens. This is **−1.59%** against
the preceding registry's single 14,855 window and **−0.31%** against the
initial comparison image's 14,665 window. The small residual difference is
not claimed as zero regression or hidden by decode gains. These are
prompt-tokens/client-TTFT measurements, not isolated GPU prefill time.

After a container restart, C1 retains 96.57/96.76 steps/s with
221.27/226.12 output tok/s. The first C8 repeat measures 894.22 tok/s and
389.47 steps/s. The second C8 repeat is **invalid**, not a speed result:
one stream repeats a six-character numeric sequence 705 times, and the
loop guard aborts the cell. Both repeats use temperature 1; the benchmark
suppresses EOS by default. Two separate C8 controls respecting EOS complete
without a loop, but this does not establish EOS suppression as the cause or
prove the checkpoint is loop-free. The failed cell is retained in the raw
evidence rather than replaced by a successful retry.

Both engines use the same target checkpoint and private NVFP4 draft head.
SGLang uses CUDA 13.0.3 and BF16 recurrent state; vLLM uses CUDA 13.4.1 and
FP32 recurrent state. This compares deployed stacks, not framework code with
identical arithmetic and dependencies.
[Request methodology, SGLang recipe and profiles](qwen38-tp2-sglang.md).

### Why replicated projections are the PCIe default

Tensor-parallel sharding of Qwen's small residual-mixing projections reduces
weight storage but introduces collectives at each layer. An eight-step C8
trace records 1,728 NCCL all-gathers with sharding versus 32 with replicated
projections; direct-copy kernels fall from 3,808 to 1,264. The shared Docker
recipe sets `VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0` for this hardware profile.
Setting it to 1 remains an explicit memory-saving option.

The separate uniform-graph optimization removes 840 foreach-copy calls and
reduces host kernel launches from 3136 to 464 in the matched eight-step
trace. It preserves mixed-batch preparation and live accepted-state updates.
Trace durations are diagnostic and are not added to infer serving latency.

Source-overlay startup controls have an unresolved state-sensitive performance
difference. They do not justify a PLE allocator, CPU-affinity or profiler
workaround; none is shipped. The unmodified registry qualification above is
the deployment evidence.

## Compatibility limits and merge handoff

- Compact GLM recovery #821 reduces TP2 recovery storage from 856.58 to
  668.81 MiB per rank. Its recorded TP4 MTP output cost is about 3.1% at
  C1 and 1.8% at C8; this tradeoff remains visible.
- Generic Qwen OffloadingConnector/MTP restore is **research-only** and not
  included. The qualified alternative is SimpleCPU via `CACHE_MODE=native`.
- Image-bearing GLM external LMCache reuse is **unsupported** because its
  checkpoint identity does not include the image payload. Ordinary vision
  and native GPU prefix caching work.
- The separately reported structured-tool assertion was not reproduced in
  the bounded test; a reporter payload is still needed. It is not marked fixed.

[Luke's merge checklist](https://github.com/local-inference-lab/vllm/issues/808)
lists the purpose and order of all 13 PRs.
[Raw evidence and reproducers](data/karmic-merge-audit-20260922/) retain exact
image identities, individual cells, source audits and historical controls.
