# RTX PRO 6000 Blackwell Wiki Sources

Use the [https://github.com/local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) field wiki to identify the community runbook, exact serving context, and known evidence related to a shared change.

## Lookup order

1. Resolve the current `master` commit and record its full 40-character SHA.
2. Start from [`README.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md) and its **Start Here** table.
3. Open the current model-family hub or runbook. Use [`INDEX.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/INDEX.md) only when the page is not linked from the front page.
4. Use [`GLOSSARY.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/GLOSSARY.md) for community terminology.
5. Treat versioned model pages and [`daily-summaries/`](https://github.com/local-inference-lab/rtx6kpro/tree/master/daily-summaries) as historical evidence unless the package reproduces that exact result.
6. In a published package, replace moving `master` links with commit-pinned links in the form `https://github.com/local-inference-lab/rtx6kpro/blob/<wiki-commit>/<path>`.

## Package use

Record the wiki repository, wiki commit, relevant runbook path, commit-pinned runbook URL, and relationship to the contribution. Use the current runbook to identify the documented base configuration and terminology, but derive source identities and changed files from the actual artifact being packaged.

Do not copy an image tag, launch command, benchmark number, or support conclusion from the wiki without checking the current model hub and confirming that it applies to the package's exact model, image, engine, topology, and source revisions.
