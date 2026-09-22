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

The source audit targets vLLM integration `622912b9b2c` and B12X
`3113aa0b859`, packaged together after the canonical draft/PLE update and
tuning-result repair. Historical comparison rows retain their own image
identities; they are not interchangeable with this artifact.

Image:
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-3221ccacf71002ea`

Digest:
`sha256:bf2c8f5da6f1de82b345367102b528692231a23eadc1b49c180c82c397c77522`

| Component | Published revision |
|---|---|
| vLLM integration | `622912b9b2cfd04784da5b0dfc69ce8aa5aee02d` |
| B12X integration | `3113aa0b8596fe55a96a28d58efd40f4dfa9955c` |
| Shared runtime recipe | `338546b0e79ff765dd3031b671bef5e476b4e883` |

Both source branches are `integration/karmic-kraken-beta`; the branch named
`integration/beta` belongs to Jovian Judgement and is not the audit target.
The [registry receipt and component manifest](data/karmic-merge-audit-20260922/tuning-registry-release/)
bind the complete native dependencies and runtime to this image.

The beta publication job in [build 35703163862](https://github.com/local-inference-lab/blackwell-llm-docker/actions/runs/35703163862)
succeeded, including 94 cache/packaging tests and native GPU smoke. The separate canonical KK publication failed because canonical
B12X does not yet include the required integration contracts. Merge the listed dependencies before treating
the canonical channel as equivalent to this beta.

The serving host received the exact builder artifact over LAN while the
registry upload completed. Pulling the immutable registry digest then returned
the identical Docker config
`sha256:5cb7de41ecce821b2c33880b2787254c602aed86637c64ee58c77e701fd68db9`,
also used by both Qwen containers. The archived OCI manifest maps the registry
digest to that config; all 68 filesystem layers match. This verifies artifact
identity across containerd and classic Docker stores, which expose different
objects through their image-ID fields. No serving source overlays were used.

## Frozen inputs and merge order

| Component | Canonical input | PR order |
|---|---|---|
| vLLM | `dev/karmic-kraken`, `85a78f57a0e4d825f19b9cff243068d9a3aac7b2` | #798, #805, #813, #821, #822, #837, #834, #835, #836, #838 |
| B12X | `master`, `b294e69d8eba2ea56d2aed7cc359c0df4bcaa57d` | #393, #406, #409, #414 |

Ordinary merge-commit composition preserves contributor histories and produces
these complete Git trees:

| Component | Canonical + PR tree, identical to integration | Remaining differences |
|---|---|---:|
| vLLM | `c38e7c35300f96d016ddd2611019bf55a435b624` | 0 |
| B12X | `67c96e44da899e4660c38905724cb1943f8ab768` | 0 |

The [vLLM](data/karmic-merge-audit-20260922/vllm-canonical-85a78-final-check.json) and
[B12X](data/karmic-merge-audit-20260922/b12x-tuning-contract-final-check.json) audit
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
- B12X #414 adds the rejection-count field consumed by canonical vLLM's
  distributed tuning coordinator. The five-argument result constructor remains
  valid. No candidate skipping, kernel arithmetic or winner-selection policy
  changes; failed launch candidates are not silently discarded.

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
| B12X tuning-result contract and canonical vLLM rank coordination | Five focused tests fail before; 61 B12X and 29 vLLM tests pass after | Qualified host protocol |
| Installed-package tuning exchange and publisher gate | Mismatched packages fail; compatible five/six-field pairs pass; 506 publisher/launcher tests pass, one skip | Qualified without a GPU or model |
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

### Distributed-startup failure caught by serving qualification

Image `e5fef1d4361a77ea`, digest
`sha256:37a1aeeaddad12f395eec8735e56b62443a0c091a0657cc97bda22e55d276d88`,
contains vLLM `622912b9b2c`, B12X `d16e71c3da2` and recipe `95cc14ef046`.
Both TP2/MTP3 deployments fail during distributed autotuning: canonical vLLM
reads `TuningRequirement.rejected_count`, absent from the selected B12X package.
This is a startup failure, not a valid benchmark arm. The published release is
marked with the failure; no throughput is assigned to it.

[B12X #414](https://github.com/local-inference-lab/b12x/pull/414) supplies the
validated result field. [Publisher #66](https://github.com/local-inference-lab/blackwell-llm-docker/pull/66)
executes the installed packages' host-side exchange before publication, so an
import-only package check can no longer miss this schema mismatch. The source
proof above includes #414. Both rebuilt Qwen TP2/MTP3 servers start, capture
graphs and pass the distinct-code CPU-restore check. These serving results
remain separate from the 90 passing host tests. Complete receipts and failed
startup logs are retained in the raw evidence directory.

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
| Audited PR composition `3221ccacf71002ea`, initial series | 222.33 | 95.74 | 906.80 | 389.08 |
| Same `3221ccacf71002ea` server, fixed post-prefill confirmation | 222.57 | 95.81 | 905.13 | 389.32 |

The audited PR composition measures **+10.71% C1 / +1.70% C8** against the
saved SGLang deployment. Relative to the QSA-guard image, measured output is
**−3.03% C1 / +0.48% C8**, while verifier rate is **−0.59% / +0.24%**.
The C1 accepted-length median changes from 2.380 to 2.321. The lower output
result is retained, not relabeled as zero regression; five stochastic runs do
not isolate a source-induced acceptance change. C1 output ranges from 213.03
to 234.24 tok/s, whereas verifier rates remain within 95.70–95.78 steps/s.

All ten initial decode cells pass the error, loop and occupancy guards.
Three uncached prefill windows measure **14,547 / 14,570 / 14,586 tok/s**,
each with twelve requests: **median 14,570**, −0.75% against the QSA-guard
image. Sixteen mixed text/JSON-schema requests and all twelve distinct-code
CPU-restore requests pass. Restoration returns 2,880 external tokens and zero
native GPU hits. The profiler was configured but never activated.

A second, fixed five-run series follows the prefill and mixed-request checks
without changing or restarting the server. All ten additional decode cells
pass, with medians **222.57 / 905.13 output tok/s** and **95.81 / 389.32
steps/s**. C1 accepted length is 2.324. This confirms the lower C1 observation;
prefill warmup does not restore the preceding image's output median. The
two series remain separate, and neither invalidates or replaces the initial
results. The verifier difference remains below 1%, but an acceptance-related
output difference must not be described as zero throughput regression.

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
vLLM `47cb3450b11aac466f03a933de576a37f970da53`, recipe
`2057ca2708f9cfdc3d45814ad84a7ced742df05b`, and B12X
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

### GLM DFlash2 loader and vision check

The same immutable `3221ccacf71002ea` image passes all seven bounded checks
on four stock Max-Q GPUs: arithmetic, cold/repeated/changed text prefixes,
single-image shape recognition, and ordered two/four-image color lookup.
The actual process uses DFlash2 K7 with probabilistic drafts and standard
rejection, TP4/DCP1, B12X target serving and full/piecewise graph capture.
The draft attention backend is `FLASH_ATTN` with automatic cache dtype.

Target snapshot: `520de24eabf507659eaef7c70f14fd584527facc`;
MXFP8 draft snapshot: `713226ab03bc38afdf955c7450436c2f7176f6f8`.
The bounded launch sets a 65,536-token context, eight sequences and twelve GiB
GPU KV per rank; the scheduler budget remains 4096. No source overlays are
used. These are correctness/startup checks, not a repeated GLM throughput or
external-LMCache qualification on this digest.
[Launch, process audit, responses and server log](data/karmic-merge-audit-20260922/glm-tuning-registry-smoke/)
record the conditions independently of the Qwen measurements.

### Outstanding limits

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
lists the purpose and order of all 14 PRs.
[Raw evidence and reproducers](data/karmic-merge-audit-20260922/) retain exact
image identities, individual cells, source audits and historical controls.
