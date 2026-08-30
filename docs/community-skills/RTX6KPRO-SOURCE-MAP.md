# rtx6kpro Source Map for the Community Skills

Repository: `https://github.com/local-inference-lab/rtx6kpro`  
Release-audit commit: `4126fc06692c6ab042b0cd37d5062893fa402f47`

The audit commit records what this skill release reviewed. Agents must still resolve the current `master` commit whenever a skill runs.

## Source-selection rule

1. Start with `README.md#start-here`.
2. Use the current model-family hub/runbook.
3. Use `INDEX.md` only when the front page or model hub does not link the needed page.
4. Use `GLOSSARY.md` for terminology.
5. Prefer current model hubs over versioned pages, aggregate result tables, and daily summaries.
6. Use historical material only to reproduce a named result or locate a regression boundary.
7. Publish commit-pinned links: `https://github.com/local-inference-lab/rtx6kpro/blob/<commit>/<path>`.

## Shared entry points

| Path | Role |
|---|---|
| `README.md` | Current landing page and Start Here map. |
| `INDEX.md` | Generated discovery index for pages not yet linked from a hub. |
| `GLOSSARY.md` | Community vocabulary and acronym expansions. |
| `docs/newcomer-onboarding.md` | Minimum information required for useful support and reproduction. |
| `daily-summaries/` | Chronology and historical evidence, not current defaults. |

## Model hubs and current runbooks

Use the model-family hub selected by the current `README.md`. At the audited commit, prominent entry points included:

- `models/glm-5.3-flash-dflash2.md`
- `models/glm-5.2.md`
- `models/deepseek-v4-flash.md`
- `models/kimi.md`
- `models/mimo.md`
- `models/qwen38-27b.md`
- `models/glm-5.1.md`
- `models/legacy.md`

A current model runbook may provide an immutable image, source contract, launch command, startup evidence, qualification status, limitations, and links to scripts or compose files. Verify every artifact identity directly before publication.

## Skill mapping

| Skill | Primary rtx6kpro sources |
|---|---|
| `local-inference-lab-sharing-changes` | Current model hub, `README.md`, `INDEX.md`, `GLOSSARY.md`; record the wiki commit and runbook in the package. |
| `local-inference-lab-publishing-docker` | Current model runbook; `models/glm-5.3-flash-dflash2.md` as a publication-structure example; `compose/`; `scripts/`; `optimization/docker-images.md`; hardware topology; common issues. |
| `local-inference-lab-running-benchmarks` | Current runbook; `benchmarks/results.md`; `benchmarks/inference-throughput/README.md`; GLM-specific and general KLD pages; MTP-quality and NVFP4 comparisons; speculative-decoding and topology pages. |
| `local-inference-lab-evaluating-prompts` | Current runbook only for serving context. The prompt run remains the source of its qualitative result. |
| `local-inference-lab-reconciling-changes` | Current source contract and qualification invariants; `optimization/dspark-upstream-consolidation.md` as a reconciliation-method example. |
| `local-inference-lab-reporting-bugs` | Current runbook; `docs/newcomer-onboarding.md`; `troubleshooting/common-issues.md`; topology and bandwidth pages. |
| `local-inference-lab-github-contributions` | Current model hubs, `INDEX.md`, `GLOSSARY.md`, existing investigations, `scripts/check-acronyms.py`, and `scripts/generate-wiki-index.py`. |

## Benchmark and quality sources

| Path | Appropriate use |
|---|---|
| `benchmarks/results.md` | Discover prior results and candidate evidence; confirm exact identities and conditions before comparison. |
| `benchmarks/inference-throughput/README.md` | Context-by-concurrency methodology, explicit server configuration, MTP controls, raw results, and bounded interpretation. |
| `benchmarks/glm52-kld-evaluation.md` | Current GLM-5.2/vLLM KLD workflow at the audited repository state. |
| `benchmarks/kld-evaluation.md` | General and older Qwen/SGLang KLD workflow; follow its redirect for current GLM-5.2 reproduction. |
| `benchmarks/mtp-quality-evaluation.md` | With/without-MTP quality comparison and statistical reporting pattern. |
| `benchmarks/nvfp4-quantization-comparison.md` | Quantization comparison evidence and controls. |
| `optimization/speculative-decoding.md` | Speculative-decoding concepts, configurations, known issues, and model-specific notes; verify against current runbooks. |

## Hardware and troubleshooting sources

| Path | Appropriate use |
|---|---|
| `hardware/topology.md` | PCIe, NUMA, switch, P2P, and collective-sensitive interpretation. |
| `hardware/pcie-bandwidth.md` | Transport measurements and topology-specific performance context. |
| `hardware/gpu-configs.md` | GPU layouts and deployment context. |
| `hardware/blackwell-power-limit-sweep.md` | Power-limit, clock, thermal, and performance-per-watt methodology. |
| `optimization/nccl-tuning.md` | NCCL and graph-file tuning context; validate against the target topology and current runtime. |
| `optimization/pcie-oneshot-allreduce.md` | PCIe collective implementation and comparison context. |
| `troubleshooting/common-issues.md` | Known signatures and existing workarounds; treat each entry as a hypothesis until reproduced. |
| `troubleshooting/pcie-link-speed-flapping-cpayne.md` | c-payne Switchtec link-flapping diagnosis and verify-then-lock procedure; not a generic fix. |
| `scripts/pcie-link-supervisor.py` | Companion supervisor implementation for that scoped hardware procedure. |

## Reconciliation source

`optimization/dspark-upstream-consolidation.md` demonstrates a useful method for reconciling parallel implementations: establish exact histories, measure end-to-end behavior, compare architecture and source deltas, identify strengths and blockers on both sides, define directional ports, and record measured decisions. Its findings apply only to the exact revisions and environment it documents.

## Wiki contribution checks

When changing `rtx6kpro` itself:

```bash
python3 scripts/check-acronyms.py
python3 scripts/generate-wiki-index.py > INDEX.md
```

Link new pages from the relevant model hub and label their status clearly.
