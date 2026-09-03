# Local Inference Lab Community Skills

These repository-native Agent Skills cover sharing source changes, publishing and supporting Docker images, running comparable benchmarks, evaluating models with qualitative prompts, reconciling overlapping work, reporting bugs, and preparing explicitly requested GitHub contributions.

Current release: skill bundle `1.5`, plugin `1.5.0`, and Docker image-record schema `4`.

## Why this is a collection

The Agent Skills format defines one skill as one directory containing its own `SKILL.md`. The seven installable capabilities live at this repository's root under `skills/<skill-name>/`; there is intentionally no root `SKILL.md`. The repository exposes those directories as a Hermes tap and through its ChatGPT/Codex plugin manifest while retaining the field wiki, runbooks, and benchmarks that the skills use.

These workflows have different triggers, outputs, support boundaries, and tools. Independent skill directories improve discovery and prevent Docker policy, benchmark methodology, or GitHub instructions from loading during an unrelated patch-sharing task.

Each skill may contain:

```text
skills/<skill-name>/
├── SKILL.md                  # required skill metadata and procedure
├── agents/openai.yaml        # optional ChatGPT/Codex UI metadata
├── references/               # loaded on demand
├── scripts/                  # deterministic helpers, when needed
└── assets/                   # output templates, when needed
```

At the repository root, `.codex-plugin/plugin.json` exposes `./skills/` for ChatGPT and Codex, and `skills.sh.json` provides Hermes/skills.sh categorization. Tests, examples, this documentation, and release notes stay outside individual skill directories.

## Installed skills

| Slash command | Purpose |
|---|---|
| [`/local-inference-lab-sharing-changes`](../../skills/local-inference-lab-sharing-changes/SKILL.md) | Package patches, files, scripts, experiments, reproducers, and evidence for direct sharing without requiring GitHub. |
| [`/local-inference-lab-publishing-docker`](../../skills/local-inference-lab-publishing-docker/SKILL.md) | Publish recommended or custom Docker images with immutable provenance, qualification, support-thread routing, and supersession. |
| [`/local-inference-lab-running-benchmarks`](../../skills/local-inference-lab-running-benchmarks/SKILL.md) | Run target-only decode, Estonia, LAVD, Hotel Lights, speculative, prefill, and accuracy comparisons with pinned controls. |
| [`/local-inference-lab-evaluating-prompts`](../../skills/local-inference-lab-evaluating-prompts/SKILL.md) | Run repeatable one-shot Tetris, platformer, and future Flamingo qualitative evaluations. |
| [`/local-inference-lab-reconciling-changes`](../../skills/local-inference-lab-reconciling-changes/SKILL.md) | Resolve merges, rebases, cherry-picks, and multi-source compositions without losing intent or attribution. |
| [`/local-inference-lab-reporting-bugs`](../../skills/local-inference-lab-reporting-bugs/SKILL.md) | Build minimal reproducible reports and route custom-image problems through the recommended-image control. |
| [`/local-inference-lab-github-contributions`](../../skills/local-inference-lab-github-contributions/SKILL.md) | Prepare focused GitHub issues or pull requests only when GitHub is explicitly selected. |

The skills are independently installable and may be combined when a task genuinely spans workflows.

## Hermes installation

### Install the collection as a tap

Add this repository as the tap:

```bash
hermes skills tap add local-inference-lab/rtx6kpro
hermes skills search local-inference-lab
```

Install individual skills as needed:

```bash
hermes skills install local-inference-lab/rtx6kpro/skills/local-inference-lab-sharing-changes
hermes skills install local-inference-lab/rtx6kpro/skills/local-inference-lab-publishing-docker
hermes skills install local-inference-lab/rtx6kpro/skills/local-inference-lab-running-benchmarks
```

### Local personal installation

Run from the root of a `local-inference-lab/rtx6kpro` checkout:

```bash
mkdir -p ~/.hermes/skills
cp -R skills/* ~/.hermes/skills/
```

### Project-local installation

```bash
mkdir -p .agents/skills
cp -R skills/* .agents/skills/
```

Hermes registers each installed skill as its own slash command. Several may be stacked when needed, for example:

```text
/local-inference-lab-publishing-docker /local-inference-lab-running-benchmarks qualify this custom image and prepare its support thread
```


## ChatGPT and Codex installation

From the root of a `local-inference-lab/rtx6kpro` checkout, copy the individual skill folders into a location ChatGPT/Codex scans:

```bash
# User-level skills
mkdir -p ~/.agents/skills
cp -R skills/* ~/.agents/skills/

# Or repository-scoped skills
mkdir -p .agents/skills
cp -R skills/* .agents/skills/
```

For reusable distribution, `.codex-plugin/plugin.json` exposes the repository's seven skill directories through the current ChatGPT/Codex plugin workflow rather than the deprecated standalone OpenAI skills catalog.

## Common routes

### Share a patch without GitHub

```text
/local-inference-lab-sharing-changes Package these commits and test evidence as a validated ZIP for Discord. Do not create a PR.
```

### Publish a custom image

```text
/local-inference-lab-publishing-docker Prepare a dedicated support thread, provenance record, and one short model-channel link for this custom image.
```

### Publish the recommended image

```text
/local-inference-lab-publishing-docker Validate this maintainer-supported image against correctness, stability, and performance regression gates and prepare the recommended-image announcement.
```

### Benchmark a new model or quant

```text
/local-inference-lab-running-benchmarks Run the standard baseline: target-only decode with speculation disabled, Estonia, and LAVD. Keep Hotel Lights optional and preserve raw JSON.
```

### Report a custom-image failure

```text
/local-inference-lab-reporting-bugs Reduce this failure, compare it against the current recommended image, and prepare the support-thread report.
```

## RTX PRO 6000 Blackwell wiki integration

Every skill points to the public `local-inference-lab/rtx6kpro` field wiki for the parts relevant to its workflow. Agents resolve the current wiki commit, begin with the front-page **Start Here** model hubs, and publish commit-pinned runbook links rather than relying on moving `master` URLs.

The skills use the wiki as follows:

| Skill | Wiki use |
|---|---|
| Sharing changes | Record the current runbook context and package a commit-pinned evidence link. |
| Publishing Docker | Find the current model-family runbook, recommended control, source contract, launch, qualification evidence, limitations, topology, and known issues. |
| Running benchmarks | Select a documented server configuration and consult throughput, KLD, MTP-quality, speculative-decoding, and topology methodology without treating old tables as new results. |
| Evaluating prompts | Record serving context only; qualitative results still come from the preserved prompt and generated artifact. |
| Reconciling changes | Recover source contracts and use consolidation studies as a comparison method, not as reusable conclusions. |
| Reporting bugs | Follow the newcomer support contract, current runbook, common-issue index, topology guidance, and scoped hardware procedures such as the c-payne link supervisor when applicable. |
| GitHub contributions | Find existing work and apply the wiki's hub, acronym, status, and generated-index conventions. |

The repository was reviewed for this release at `4126fc06692c6ab042b0cd37d5062893fa402f47`. That audit identity documents this package build; agents still resolve the current wiki commit whenever they use a skill.

See [`RTX6KPRO-SOURCE-MAP.md`](RTX6KPRO-SOURCE-MAP.md) for the complete human-readable mapping.

## Docker distribution policy

Each model family has one `recommended` maintainer-supported image for general users. Main announcements and automated listings point to it. A recommended image must be `official`, `qualified`, `maintainer-supported`, and pass correctness, stability, and performance regression gates.

Custom images remain welcome. Each custom image:

- receives a dedicated support thread;
- receives at most one link from the relevant model channel;
- is `ephemeral` unless its author accepts maintenance and first-line support;
- keeps image-specific reports in its thread;
- escalates upstream only after reproduction on the current recommended image or a minimal source reproducer identifies the responsible change;
- becomes `superseded` when the relevant work enters an exact recommended image digest.

The Docker skill contains a machine-readable record schema, validator, full support-thread renderer, one-link renderer, recommended-image renderer, and pre-publication checklist.

## Missing information

All skills use the same exact vocabulary:

- `UNKNOWN — needs verification`: a fact should exist but has not been established.
- `Not tested`: a test or configuration was not run.
- `N/A`: the field genuinely does not apply.

Unknown values are never guessed, and unperformed tests are never reported as passed.

## Validation

Run the repository test suite:

```bash
python3 -m unittest discover -s tests -v
```

Validate each skill with the open reference validator when it is available:

```bash
for skill in skills/*; do
  skills-ref validate "$skill"
done
```

The included tests additionally check directory/name matching, frontmatter constraints, direct reference links, line limits, helper behavior, Docker policy invariants, exact prompt text, security scans, and deterministic archives.

## Examples

`examples/` contains fictional contribution records. They may point to public project documentation solely to demonstrate commit-pinned source references, and do not identify real contributors.
