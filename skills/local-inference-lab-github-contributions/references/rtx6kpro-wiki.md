# RTX PRO 6000 Blackwell Wiki Sources

Use the [https://github.com/local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) field wiki to find current runbooks, prior investigations, terminology, and repository-local documentation requirements before opening a focused issue or pull request.

## Source lookup

1. Resolve and record the current `master` commit.
2. Start from [`README.md#start-here`](https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md#start-here), then inspect the current model hub and related pages.
3. Use [`INDEX.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/INDEX.md) to find prior investigations and named historical releases.
4. Use [`GLOSSARY.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/GLOSSARY.md) and expand important acronyms on first prose use; leave commands, tags, variables, JSON, paths, and raw logs unchanged.
5. Link commit-pinned pages in coordination notes and PR descriptions.

## When contributing to rtx6kpro itself

- Link a new page from the relevant model hub.
- Mark a runbook or result as `current`, `historical`, `reduced validation`, or `experimental` as appropriate.
- Include exact Docker tag/digest, model snapshot, launch/client commands, TP/DCP/speculation settings, GPU layout, validation, and limitations.
- Run `python3 scripts/check-acronyms.py` when polishing a major page.
- Regenerate the index with `python3 scripts/generate-wiki-index.py > INDEX.md`.
- Treat the environment and current repository scripts as the source of truth; inspect their current `--help` or content rather than copying stale command syntax from a historical page.

Do not create a duplicate wiki page or issue when an existing model hub, investigation, or PR already answers the same technical question.
