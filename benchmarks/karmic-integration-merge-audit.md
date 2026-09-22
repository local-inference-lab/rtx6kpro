# Karmic integration: reproducible canonical PR composition

The published vLLM and B12X integration branches can be reconstructed from the canonical
branches plus the PRs below. This audit compares complete Git trees, including
tests, CI, documentation and immutable release fragments; commit ancestry alone
is not considered evidence of source equivalence.

Status: **implemented** and **qualified** for source reconstruction and focused
tests. Serving qualification is recorded separately because canonical Qwen
projection sharding changes the measured TP2 execution path.

Both `integration/karmic-kraken-beta` branches contain the audited trees:
vLLM `47cb3450b11aac466f03a933de576a37f970da53` and B12X
`c4349457b13905f9f1689240fcda5991e133de44`. The corresponding container is
building. Source publication is not qualification of that registry artifact.

## Frozen inputs and merge order

| Component | Canonical input | PR order |
|---|---|---|
| vLLM | `dev/karmic-kraken`, `a18246b0626414b5a198bc63ffffb15625208bb7` | #798, #805, #813, #821, #822, #837, #834, #835, #836, #838 |
| B12X | `master`, `b294e69d8eba2ea56d2aed7cc359c0df4bcaa57d` | #393, #406, #409 |

The machine-readable audit records every PR head, author, base branch, merge
tree and residual diff. It performs ordinary merge-commit composition, preserving
contributor histories. A squash/rebase workflow may require equivalent conflict
resolution; it is not the operation simulated here. Any changed canonical tip
or PR head requires another audit.

## Consolidation decisions

- vLLM #807 is closed: merged #809 already contains its graph accounting and
  recurrent-state ownership changes. Merging #807 again would restore stale code.
- vLLM #824 is closed and included in #821: compact KDA recovery and its gate/output
  alias correction are one reviewed branch. Kimi still receives caller-owned
  recovery records. Merge the workspace contracts in #798 first.
- vLLM #777 is closed in favor of #837: the direct KK port retains Naadir Jeewa's
  original implementation and attribution and resolves adjacent test changes.
- vLLM #836 materializes the SimpleCPU multi-prefix-group/QSA scratch exclusion
  already used by beta, retaining Zhewen Li's upstream contribution.
- vLLM #838 carries the integration release-fragment policy. B12X #393 includes
  the matching policy and KK wheel workflow. Canonical-sync fragments describe
  existing canonical behavior without changing published fragments.
- B12X #406 is refreshed against master. Its schema test expects version 7:
  the context-parallel capacity contract adds ABI fields. Contributor history
  and master's silicon-based tuning identity are both preserved.
- vLLM #798 retains both canonical release-before-rebind coverage and the beta
  checks that clear selector plans and rebind live index-cache views.

## Tree equality

| Component | Canonical + PR tree | Remaining file differences |
|---|---|---:|
| vLLM | `2c3073c5554acaf6265f23151602bb034320cd8d` | 0 |
| B12X | `aa4ab95732b1eef1f63e5daee515d67b03b2df20` | 0 |

The reconstruction and published integration trees are identical. A fresh audit
of all actual PR heads found no conflicts or residual files immediately before
publication. Registry validation is recorded separately from this equality check.

The integration refresh keeps canonical Qwen tensor-parallel HyperConnection
projections, per-head GDN warmup and QSA capacity compatibility. B12X retains
contiguous-attention normalization, silicon-based tuning identity and switched
RoCE traffic-class handling from master. None is silently reverted to obtain
source equality or a benchmark number.

The composition also retains Luke Alonso's cross-group GDN metadata fusion,
compact NVFP4 decode tiles, expanded runtime grid tuning and IQ2_XS support.
The GDN fusion complements #835: mixed batches refresh worklists in one launch;
uniform graph decode skips worklists it does not consume. Both changes coexist
without a conflict. Their release fragments preserve the canonical author's
attribution.

## Focused validation

The composed Python sources run inside the CUDA 13.4.1 wheel-runtime image,
retaining its compiled extensions. This is a source-composition diagnostic,
not validation of an unpublished container digest.

| Conditions | Measurement | Result |
|---|---|---|
| Configuration, SimpleCPU scheduler, GLM recovery | 81 tests, three unrelated selections excluded | Pass |
| QSA, GDN metadata/warmup, KDA alias and shared counts | 112 tests, nine skips | Pass |
| B12X preparation, DCP geometry, RoCE contracts, evidence validation | 63 tests | Pass |
| B12X contiguous normalization on CUDA, including graph replay | 8 tests | Pass |
| GLM selector release and rebinding, both pool lifetimes | 2 tests | Pass |
| Qwen TP2/MTP3, temperature 1, mixed text/JSON-schema requests at C4 | 16 requests, inputs up to approximately 31k tokens | All answers correct |

The matrix above was collected on the preceding source composition
`4717b12198a` / `2879cb0234`. After importing the canonical metadata/MoE changes,
the published `47cb3450b11` / `c4349457` composition passes 61 GDN tests and
183 B12X tuning/swapped-NVFP4 tests, including CUDA execution on the assigned
Max-Q test GPU. Those focused results do not substitute for the pending
source-unmodified container serving check.

The source audit does not establish that every open PR in the repositories is
required by beta. MiMo-specific changes and independent experiments outside
these reconstructed trees are not included in the merge list.

## Compatibility limits

Compact GLM recovery is a memory-saving feature with a recorded TP4 MTP output
cost of about 3.1% at C1 and 1.8% at C8. Generic Qwen OffloadingConnector MTP
restore remains research-only; the qualified alternative is SimpleCPU through
the unified launcher's native CPU-cache mode. External GLM image-bearing
LMCache reuse remains unsupported because checkpoint identity does not include
the image payload. Source equality does not remove these documented limits.

## Qwen TP2 performance gate

Conditions match the [Qwen TP2 comparison](qwen38-tp2-sglang.md): the same
Max-Q GPU pair at stock clocks, QAD checkpoint, TP2/DCP1/MTP3, temperature 1,
top-p .95, reasoning medium, 6019-token batch budget and 8 GiB KV per rank.
Each decode cell is the median of five warmed 30-second windows. The serving
composition in this table is `4717b12198a` / `2879cb0234`, before the additional
canonical metadata/MoE commits. It uses Python source overlays
on the comparison's CUDA 13.4.1 image, preserving compiled native extensions.

| Source/configuration | C1 output tok/s | C1 steps/s | C8 output tok/s | C8 steps/s |
|---|---:|---:|---:|---:|
| Uniform-graph metadata fix on the comparison image | 226.87 | 95.85 | 907.28 | 387.74 |
| Published metadata-fix image, no source overlays | 225.17 | 95.29 | 898.41 | 387.12 |
| Canonical + audited PRs, sharded residual projections | 187.73 | 82.30 | 805.65 | 347.40 |
| Same sources, replicated projections (`VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0`) | 204.82 | 87.66 | 855.43 | 369.39 |
| Same sources, replicated projections, GPU-local CPU/host-memory binding at startup | 222.49 | 95.97 | 903.23 | 389.22 |

Disabling projection sharding improves C1 output by 9.1% and C8 by 6.2% against
the same-source sharded control. It does **not** recover the full earlier
throughput. Both source-composed arms pass 16 mixed text/JSON-schema requests
at concurrency four; all 20 decode cells have no request failures or loop flags.

An eight-step rank-0 C8 trace records 1,728 NCCL all-gathers with sharding versus
32 with replication. Direct-copy kernels fall from 3,808 to 1,264. Replication
restores the comparison image's collective and copy counts. The uniform-graph
metadata fix remains active in both arms: zero foreach-copy calls and 464 host
kernel launches. These profile counts establish the extra work; nested trace
durations are not end-to-end latency.

The published metadata-fix image is
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-97197f5085f4798b`,
digest `sha256:9d2431fa46c10b429fd98eeeb8aea013ba12e2010303402203c569ea09ab7e51`.
It contains vLLM `96b79aa073c2059462b613876366cf693d1c0c27`, B12X
`6b80ac55b93bb3684fc35ff3f2dc34424e9f9eab` and recipe
`f67ab3f21c5a21472bd8aebb2eeb76f5b5c73eb6`. Five warmed repeats and 16 mixed
requests pass without source mounts or edits. Uncached 32k prefill measures
14,855 tok/s over 12 samples, with 32,120 median input tokens.

The restart-sensitive performance difference is under investigation. Two-run controls
with the earlier tuning-cache identity (203.32/858.02 tok/s at C1/C8) and with
the comparison image's untouched B12X (200.21/851.56) do not recover it. These
controls do not establish B12X or its cache identity as the cause. The
source-unmodified registry result retains the earlier verifier rate, narrowing
the investigation to the source-composed refresh and its execution environment.
The complete canonical composition with replication and GPU-local NUMA binding
passes five warmed repeats and 16 mixed requests without reverting any canonical
source. It restores the verifier rate. C1 output ranges from 213.54 to 236.89
tok/s as effective accepted length varies from 2.227 to 2.468, while the step
rate stays between 95.88 and 96.00. Its 32k prefill is 14,589 tok/s, 1.79% below
the published-image confirmation; this difference remains visible rather than
being described as an unconditional no-regression result.

Moving the already-loaded engine and both workers together to the remote CPU
NUMA node retains 95.66 C1 and 388.62 C8 steps/s. This compares two constrained
CPU placements, not constrained versus unrestricted placement; it does not
rule out CPU scheduling as the cause. The host allocations stay GPU-local.

Two allocation controls do not recover the rate: placing only the 13.41-GiB
mapped PLE tables per rank on the GPU-local node yields 87.00/366.55 C1/C8
steps/s; restricting all container memory to that node yields 88.08/369.44.
Neither allocation experiment is included as an implementation change.

On that same memory-restricted running server, constraining the engine and both
workers' CPU threads to one NUMA node, without restarting or changing weights,
kernels or cache, yields 219.87/903.82 output tok/s and 95.89/388.66 steps/s.
Moving the constrained group together to the other CPU node retains 95.67 C1
steps/s. Releasing the affinity mask alone also retains 95.58 C1 steps/s; it
does not force threads to migrate. Explicitly splitting the workers across
CPU nodes retains 95.72 C1 steps/s; splitting their main and auxiliary threads
retains 95.67. The affinity changes restore this running instance, but these
controls do not fully localize the startup-sensitive mechanism. The measured
GPU-local NUMA startup configuration and the unmodified registry image are
qualified separately. No PLE placement patch is proposed. A separate
two-run substitution of the preceding HyperConnection file is not evidence
that its replicated arithmetic is faster: CPU placement was not controlled.

[Raw decode cells, source audits and diagnostic traces](data/karmic-merge-audit-20260922/)
are separate from the earlier comparison's measurements.
