# SwiGLU-correct public source composition: R35 release

Status: **qualified for the artifact, arithmetic and bounded GLM serving checks
specified here**. This historical release report compares R35 with R34; it is
not a complete multi-model or production-runaway qualification.

Image: `localinferencelab/vllm:jovian-judgement-community-20260911-r35`.
The [registry receipt](swiglu-reviewed-composition-r35-registry.json) identifies
the immutable manifest and image configuration. The [embedded source lock](swiglu-reviewed-composition-r35.source.lock)
authenticates the installed components, compatible native artifacts and launcher
inputs. The image has **two filesystem layers**, not an overlay on R34.

## R34-to-R35 changelog

- **Correct model activation semantics in split NVFP4 MoE prefill.** The gate
  projection is clamped above the model's SwiGLU limit, and the up projection is
  clamped on both sides before the activation product. GLM declares a limit of
  10. R33/R34 omitted this operation in the split phase while the monolithic
  implementation applied it. The fast split path remains enabled.
- **Use the public reviewed source composition.** Pinned Jovian Judgement,
  B12X master and LMCache bases plus the ordered review heads reproduce complete
  component trees without a private integration patch. Original contributor
  history is retained. The [merge checklist](https://github.com/local-inference-lab/vllm/issues/731)
  separates source review from image publication.
- **Isolate model-specific native imports.** Quantization configuration lookup
  does not import an unrelated DeepSeek V4.1 native backend. Selecting that
  backend still requires its actual dependencies; this is not a fallback or a
  claim of DeepSeek V4.1 serving support.
- **Provision named SHM in the standalone LMCache wrapper.** Explicit
  `engine_driven` uses a validated, preallocated named L1 arena; other modes keep
  lazy allocation. The normal cache-complete community launcher already had
  that contract. This correction does not change its transfer protocol.
- **Preserve deployment behavior.** Target B12X MoE, existing M8/MHC controls,
  FlashKDA prefill, full-and-piecewise graphs, 4096-token scheduling, FP8 KV,
  request-boundary/aligned checkpoints, private NVFP4 MTP proposal head, MXFP8
  DFlash2 and Qwen/DeepSeek V4 launch profiles remain included. GLM defaults
  remain temperature 1, top-p 0.95, high reasoning and `clear_thinking=false`.
- **Expose consistent provenance.** The package reports
  `0.26.1rc0+glm53.r35.vllmde982a50`, consistent with the source lock and image
  labels. CUDA, PyTorch, FlashInfer and FlashKDA native libraries are unchanged.

The B12X correction is already merged through [#353](https://github.com/local-inference-lab/b12x/pull/353).
R35 includes merge commit `8648fae3bc19c164b46098c1e97882efd8443876` through its
verified descendant. No additional B12X correction PR is required.

## Source identities and preservation

| Component | Attributed source mirror | Commit |
|---|---|---|
| vLLM | [voipmonitor/vllm](https://github.com/voipmonitor/vllm/tree/integration/jovian-reviewed-sources-20260911) | `de982a50c6a3e4718e5cf9f00423a92192718da1` |
| B12X | [voipmonitor/b12x](https://github.com/voipmonitor/b12x/tree/integration/jovian-reviewed-sources-20260911) | `98086604c86ec1e78977e5023ce282ecb97ab8a7` |
| LMCache | [local-inference-lab/LMCache](https://github.com/local-inference-lab/LMCache/tree/release/jovian-fp4-fs-ledger-r33-20260910) | `29bc5a2efde737c436b04499eb62cd1776cebeec` |

The [composition manifests](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/recipes/glm53/review-composition.md)
pin 32 vLLM, three B12X and nine LMCache review units and their destination
branches. Source inclusion does not mean those PRs have merged.

The independent [R34 preservation audit](https://github.com/local-inference-lab/blackwell-llm-docker/blob/9275b68ca51e79218bb28f38f37ca69eb9a0f6a3/recipes/glm53/r34-preservation.md)
checks required behavior rather than asserting identical R34 Git trees.
No required R34 serving patch is missing from the corrected composition.

The [R35 installed-artifact check](swiglu-reviewed-composition-r35-artifact.json)
compares R35 with the measured composition image. All **20,616 tracked-file and
mode checks** across both images pass. Native overrides, serving configuration
and environment are identical. Of 825 differing bytecode files, 824 contain
identical executable code and one records the release version. Other runtime
differences are release/build metadata and the tested standalone LMCache wrapper.
Git provenance and generated compiler caches are excluded from executable-source
equivalence; the complete upper-layer manifests remain in the local audit.

## Arithmetic correctness

Derek Yates (D-Rock) identified the missing activation clamp. The independent
GPU reproducer executes the production dispatch using 512 token rows, eight
experts, K256, N128, top-k two and seed 42. A limit of two makes the clamping
branch observable deterministically on this synthetic input.

| Arm | Result against the clamped reference |
|---|---|
| Immutable R34 | Fails: cosine 0.929721, RMSE 0.365815, maximum absolute error 3.11328125 |
| R34 plus only the two-file, 13-line clamp correction | Passes |
| Unmodified public composition | Passes |
| Packaged R35 | All 16 split-dispatch policy/GPU tests pass, including the clamp regression |

The [R35 test receipt](swiglu-reviewed-composition-r35-split-dispatch.log)
includes supported split dispatch, CUDA-graph/scratch reuse and policy cases.
No arithmetic tolerance was weakened. Docker launcher/build tests pass: 192
tests plus two subtests. Five standalone-wrapper SHM tests failed before the
launcher correction and pass afterward.

**Consequence and limitation:** a missing clamp changes model computation even
when every value remains finite. Attention/recurrent state can therefore be
wrong and subsequently stored faithfully by LMCache. This is a plausible cause
of unstable generation, but neither this arithmetic test nor the bounded serving
checks prove that every reported 20K–168K runaway had that single cause.
The correction is not a change to KV-cache precision or sampling defaults.

For persistent request-boundary caches, the launcher includes the complete
source-lock digest in checkpoint identity. R33/R34 entries therefore miss under
R35's automatically derived identity. Do not force a shared custom checkpoint
identity across these artifacts. An independently managed external cache should
use a distinct namespace rather than replaying state computed by the faulty path.

## Matched R34 comparison

The comparison uses the same physical RTX PRO 6000 Workstation quartet, GPUs
4–7, 600 W, graphics offset zero and **VRAM +6000 MHz**. It is not stock-clock
data. Both arms use TP4/DCP1, DFlash2 K7 probabilistic proposals with standard
rejection, temperature 1/top-p 0.95, FP8 target KV, B12X target backends and
all-reduce, FlashKDA prefill, V2 runner, 4096-token budget, OMP1, NCCL16/2MiB
and `FULL_AND_PIECEWISE` graphs.

The target revision is `46aaae8a82032f77100f2f03e9cc11b391df3b4d`; the offline
MXFP8 DFlash2 conversion is identified as `dc77ff1` in the recorded launch.
Both checkpoints are local read-only mounts during qualification.

| Metric | R34 | R35 component composition | Change |
|---|---:|---:|---:|
| Cold 32K prefill tok/s | 16,872.108 | 16,694.304 | −1.05% |
| C1 output tok/s | 254.852 | 255.054 | +0.08% |
| C1 verifier steps/s | 97.374 | 97.708 | +0.34% |
| C8 aggregate output tok/s | 803.703 | 792.843 | −1.35% |
| C8 aggregate verifier steps/s | 308.737 | 310.736 | +0.65% |

Decode uses llm-decode-bench 0.4.29, zero initial context, a 15-second warmup,
one 30-second cell per concurrency and an 8192-output-token limit. There are
no API errors. C8 emitted tokens per verifier step differ, 2.603 versus 2.552;
output rate alone is not an execution-speed oracle.

Prefill uses exact 32,768-token prompts, one output token and at least 30
measured seconds after warmup. All 17 R34 and 16 composition samples compute
32,768 tokens locally with no cached/external tokens. These observations meet
the bounded 2% screening gate; they do not establish statistical equivalence or
a general speedup. [Raw comparison and conditions](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/recipes/glm53/review-qualification.json).

## Packaged-image serving and limits

The R35 Docker artifact runs on physical GPUs 8–11 at `0.0.0.0:5051`, with
the same model/configuration and VRAM +6000. Seventeen prefix/conversation checks
pass: three exact 8192-token repeats, shared SYSTEM instructions, four concurrent
document lookups, response continuation, tool history and changed instructions.
Repeated prompts restore the complete input endpoint and preserve deterministic
output token IDs. [Condensed receipt](swiglu-reviewed-composition-r35-prefix.json).

The separately serialized final-image performance check on GPUs 8–11 gives:

| Metric | R35 on GPUs 8–11 |
|---|---:|
| Exact cold 32K prefill, median | 16,343 tok/s |
| Exact cold 32K prefill, range across 15 measured prompts | 16,278–16,407 tok/s |
| C1 output / verifier | 263.57 tok/s / 97.04 steps/s |
| C8 aggregate output / verifier | 813.61 tok/s / 310.24 steps/s |

The [exact-prefill receipt](swiglu-reviewed-composition-r35-prefill.json) verifies
32,768 locally computed tokens and zero cached/external tokens for every request,
with one excluded warmup and 30 measured seconds. The [decode receipt](swiglu-reviewed-composition-r35-serving.json)
uses llm-inference-bench 0.6.1, temperature 1, model-default top-p 0.95, a
15-second warmup and 30-second C1/C8 cells. Both decode cells have no errors,
loops, underfill or warmup timeout. Its separate text-prompt prefill observation
is not used as the exact token-ID prefill result above. Other model services run
on other GPU quartets of the host. These measurements are an operational check,
not a second matched R34 A/B; neither the GPU quartet nor benchmark revision
matches the preceding comparison.

The full no-spec/MTP3 performance matrix, DCP4 external-cache restart matrix,
TP8, NVFP4 target KV and Qwen/DeepSeek serving matrix were not repeated for this
release. Existing model-specific results retain their original artifact identity.
[vLLM #726](https://github.com/local-inference-lab/vllm/issues/726), concerning
strict JSON output with concurrent MTP/LMCache, remains unresolved. The
pre-existing two-element B12X MHC BF16 oracle discrepancy remains documented in
the preservation evidence; R35 neither introduces nor fixes it.

## Build and review ownership

The [two-layer recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/main/recipes/glm53)
is reviewed in [Docker #31](https://github.com/local-inference-lab/blackwell-llm-docker/pull/31).
The standalone-wrapper correction is recipe commit
`e3f72dec76d5af4e90da64e3e029d7df70e29765`.
Docker #31 merged into `main` after publication, at
`cedd4e07aa6bcac78c4d72c2672c6a076aaf2807`. All 44 component PRs remain open
at the manifest-pinned heads in the publication audit.
Docker publication and Docker-repository integration do not merge the vLLM,
B12X or LMCache review units. Their ordered maintainer merge actions remain
listed separately in [issue #731](https://github.com/local-inference-lab/vllm/issues/731).
