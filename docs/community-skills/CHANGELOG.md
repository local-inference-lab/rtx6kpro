# Changelog

## 1.5

- Released the repository-native skill bundle as `1.5` with plugin version `1.5.0`.
- Adapted evidence-bounded hypothesis, experimental-design, statistical-analysis, and critical-review concepts into the seven existing inference workflows; no science-specific skill was added.
- Added shared inference vocabulary for observations, experimental units, candidate/control comparisons, changed and nuisance variables, rivals, prediction/falsification, independent repetitions, order, stopping/exclusions, evidence status, effect magnitude, uncertainty, validity threats, and bounded conclusions.
- Upgraded Docker image records to schema version `4`; generated image-record Markdown is rendered from the schema-v4 example JSON.

## 1.4

- Added a direct `rtx6kpro` wiki reference to every independently installable skill.
- Added a current-source lookup order: resolve the wiki commit, start from the front-page model hubs, use `INDEX.md` for discovery, use `GLOSSARY.md` for terminology, and treat historical pages as exact-reproduction evidence.
- Added commit-pinned model-runbook fields to patch packages, Docker image records, benchmark reports, qualitative evaluations, merge records, bug reports, and GitHub coordination notes.
- Added skill-specific pointers to current model runbooks, benchmark/KLD/MTP methodology, hardware topology, newcomer onboarding, common issues, source contracts, and the DSpark consolidation study.
- Added scoped pointers for power-limit methodology, NCCL/PCIe collective analysis, inference-engine context, and c-payne PCIe link-speed flapping with its verify-then-lock supervisor.
- Upgraded the Docker publication record to schema version 3 and added validated `community_wiki` provenance.
- Added community runbook links to full image threads, one-line custom-image posts, and recommended-image releases.
- Added tests that reject moving `master` runbook URLs in strict Docker publication records.

## 1.3

- Organized the collection as seven independently installable skill directories.
- Kept every `SKILL.md` focused and moved branch-specific detail into direct, one-level references.
- Added `agents/openai.yaml` metadata to each skill and Hermes tap grouping metadata in `skills.sh.json`.
- Added a current skill-only ChatGPT/Codex plugin manifest at `.codex-plugin/plugin.json`.
- Added recommended-versus-custom Docker distribution roles.
- Added custom-image support threads, one-link model-channel routing, ephemeral/maintained/superseded lifecycle, and recommended-image correctness, stability, and performance gates.
- Added exact `UNKNOWN — needs verification`, `Not tested`, and `N/A` semantics.
- Added Docker record validation and renderers for support threads, custom-image links, and recommended releases.
- Kept direct patch and file packages as first-class contributions without requiring GitHub.
- Separated benchmarking and qualitative prompt evaluation into independent skills.
- Added fictional strict-validation examples and repository-level tests.
