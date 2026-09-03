# Community Skills Validation

Release target: skill bundle `1.5`, plugin `1.5.0`, Docker image-record schema `4`.

This page is the validation checklist, not a record of completed checks. No release-1.5 results are recorded here. The prior standalone bundle's pass counts do not describe this integrated repository release; record results only after running the checks against the complete integrated tree.

## Required invariants

Validation must confirm:

- exactly seven independently installable skills exist under repository-root `skills/<skill-name>/`, each with its own `SKILL.md` and no repository-root `SKILL.md`;
- skill names, frontmatter, direct one-level references, `agents/openai.yaml` metadata, and `skills.sh.json` groupings satisfy the collection rules;
- `.codex-plugin/plugin.json` points to `./skills/` and declares version `1.5.0`;
- every skill links its task-specific `references/rtx6kpro-wiki.md`, resolves the current wiki commit at use time, and publishes commit-pinned runbook URLs;
- Docker image records use schema version `4`, preserve the validated `community_wiki` provenance, and reject moving `master` runbook URLs in strict mode;
- generated image-record Markdown is produced by the existing renderer from the schema-v4 example JSON rather than maintained as an independent source;
- portable change packages remain schema version `2` with their validated `community_wiki` block;
- evidence records distinguish observations from interpretations and carry the inference evidence vocabulary, repetition/order/stopping controls, uncertainty, validity threats, and bounded conclusions;
- examples remain fictional and security checks reject credentials, access tokens, non-public hostnames, personal filesystem paths, unresolved required publication facts, and identifying logs.

## Checks to run

From the repository root:

```bash
python3 -m unittest discover -s tests -v
```

When the official reference validator is available:

```bash
for skill in skills/*; do
  skills-ref validate "$skill"
done
```

The integrated validation record should capture the actual command output for structure, schema and renderer behavior, local links, JSON/YAML parsing, helper behavior, security scans, and deterministic archives. Absence of a result means not run, never passed.
