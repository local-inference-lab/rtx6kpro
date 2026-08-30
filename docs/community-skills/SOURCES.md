# Design Sources

The collection follows these public specifications and implementation patterns:

1. **Agent Skills specification** — one skill per directory, required `SKILL.md`, permitted frontmatter, optional `scripts/`, `references/`, and `assets/`, relative file references, progressive disclosure, and `skills-ref` validation.
2. **Anthropic Agent Skills documentation and repository** — concise descriptions containing what and when, focused skill names, direct one-level references, sub-500-line `SKILL.md` bodies, and one directory per installable skill.
3. **OpenAI Build skills and Package plugins documentation plus `openai/plugins`** — focused skill folders, optional `agents/openai.yaml`, implicit/explicit invocation boundaries, and a `.codex-plugin/plugin.json` manifest when distributing several skills as one installable plugin.
4. **Hermes Skills System** — `skills/<skill-name>/SKILL.md` tap layout, independently installable skill paths, slash-command invocation, skill stacking, personal and project-local skill directories, and optional `skills.sh.json` groupings.
5. **K-Dense `scientific-agent-skills`** — evidence-method concepts adapted from the exact MIT-licensed sources recorded below.
6. **mattpocock/skills** — granular engineering skills, explicit invocation semantics, and focused merge-conflict resolution based on primary intent sources.
7. **Local Inference Lab `rtx6kpro` field wiki** — reviewed at `4126fc06692c6ab042b0cd37d5062893fa402f47`. The collection uses its front-page model hubs as current-entry points; `INDEX.md` for discovery; `GLOSSARY.md` for terminology; current model pages for launch/source/qualification contracts; benchmark pages for throughput, KLD, MTP-quality, and quantization methodology; hardware pages for topology-sensitive claims; newcomer onboarding and troubleshooting pages for support. Historical version pages and daily summaries remain reproduction evidence rather than current defaults.
8. **Local Inference Lab `llm-inference-bench` and source repositories** — immutable source compositions, integration locks, qualification receipts, and the separation of prefill, sustained decode, finite burst/end-to-end, completion statistics, and dataset accuracy.
9. **Community Docker Image Publishing and Support Policy supplied with this package** — one recommended image per model family, dedicated custom-image threads, support ownership, upstream escalation boundaries, supersession, immutable provenance, exact benchmark controls, and fixed missing-information markers.

## K-Dense evidence-method provenance

Source repository: [`K-Dense-AI/scientific-agent-skills`](https://github.com/K-Dense-AI/scientific-agent-skills)  
Source commit: [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f)

The adaptation is attributed under the [MIT License at `LICENSE.md`](https://github.com/K-Dense-AI/scientific-agent-skills/blob/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f/LICENSE.md), copyright (c) 2025 K-Dense Inc. Exact source paths at that commit:

| Source capability | Skill and reference-material paths |
|---|---|
| Hypothesis generation | `skills/hypothesis-generation/SKILL.md`; `skills/hypothesis-generation/references/concepts_and_workflow.md`; `skills/hypothesis-generation/references/hypothesis_quality_criteria.md`; `skills/hypothesis-generation/references/experimental_design_patterns.md`; `skills/hypothesis-generation/references/causal_inference_and_claims.md`; `skills/hypothesis-generation/references/preregistration_and_open_science.md` |
| Experimental design | `skills/experimental-design/SKILL.md`; `skills/experimental-design/references/design_types.md`; `skills/experimental-design/references/factorial_and_doe.md`; `skills/experimental-design/references/randomization_and_blocking.md`; `skills/experimental-design/references/sequential_and_adaptive.md` |
| Statistical analysis | `skills/statistical-analysis/SKILL.md`; `skills/statistical-analysis/references/assumptions_and_diagnostics.md`; `skills/statistical-analysis/references/effect_sizes_and_power.md`; `skills/statistical-analysis/references/reporting_standards.md`; `skills/statistical-analysis/references/test_selection_guide.md` |
| Scientific critical thinking | `skills/scientific-critical-thinking/SKILL.md`; `skills/scientific-critical-thinking/references/common_biases.md`; `skills/scientific-critical-thinking/references/core_capabilities.md`; `skills/scientific-critical-thinking/references/evidence_hierarchy.md`; `skills/scientific-critical-thinking/references/experimental_design.md`; `skills/scientific-critical-thinking/references/scientific_method.md`; `skills/scientific-critical-thinking/references/statistical_pitfalls.md` |

These concepts were rewritten for and incorporated into the existing Local Inference Lab workflows for change sharing, image qualification, benchmarks, prompt evaluation, reconciliation, bug diagnosis, and contribution review. They were not added as a science-specific skill or as a general scientific-analysis workflow.

Contribution examples and generated image records are fictional and do not identify real contributors. Public repository links are included only as documentation and provenance references.
