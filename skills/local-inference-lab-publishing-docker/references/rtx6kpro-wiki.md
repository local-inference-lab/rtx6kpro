# RTX PRO 6000 Blackwell Wiki Sources

Use the [https://github.com/local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) field wiki as the community runbook index for current model-family guidance, qualified launch configurations, historical regressions, and supporting evidence.

## Lookup order

1. Resolve the current `master` commit and record its full 40-character SHA.
2. Start from [`README.md#start-here`](https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md#start-here) and select the model-family hub or current runbook.
3. Use [`INDEX.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/INDEX.md) only when the required page is not linked from the front page or model hub.
4. Use [`GLOSSARY.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/GLOSSARY.md) for community terminology.
5. Treat versioned pages and [`daily-summaries/`](https://github.com/local-inference-lab/rtx6kpro/tree/master/daily-summaries) as historical evidence unless reproducing an exact named release.
6. Publish a commit-pinned URL: `https://github.com/local-inference-lab/rtx6kpro/blob/<wiki-commit>/<runbook-path>`.

## Docker publication sources

- Current model hubs and runbooks: source of the documented recommended launch, status, source contract, startup checks, qualification evidence, and limitations.
- [`models/glm-5.3-flash-dflash2.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash-dflash2.md): structural example of a status block, immutable image identity, source contract, exact launch, startup verification, qualification evidence, and limitations. Use its structure, not its model-specific claims, for unrelated images.
- [`optimization/docker-images.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/docker-images.md): background container guidance and historical image notes. A current model runbook outranks this page for image selection.
- [`compose/`](https://github.com/local-inference-lab/rtx6kpro/tree/master/compose) and [`scripts/`](https://github.com/local-inference-lab/rtx6kpro/tree/master/scripts): reproducible launch/build/qualification artifacts when a current model page points to them.
- [`hardware/topology.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/topology.md), [`hardware/pcie-bandwidth.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/pcie-bandwidth.md), and [`hardware/gpu-configs.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/gpu-configs.md): topology context for hardware-bound performance claims.
- [`troubleshooting/common-issues.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/troubleshooting/common-issues.md): known failure signatures and existing fixes; verify status against the current runbook and source repositories.
- [`inference-engines/vllm.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/inference-engines/vllm.md) and [`inference-engines/sglang.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/inference-engines/sglang.md): engine-specific background. The selected current model runbook remains authoritative for the image being published.

Add the wiki commit and runbook to the image record. The wiki is contextual evidence, not proof of the image's own digest, build recipe, source composition, or test result. Verify those directly.
