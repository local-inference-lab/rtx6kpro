# RTX PRO 6000 Blackwell Wiki Sources

Use the [https://github.com/local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) field wiki to start from a known-good runbook, check documented failure signatures, and attach the exact context needed for useful support.

## Required lookup

1. Resolve and record the current wiki `master` commit.
2. Open [`README.md#start-here`](https://github.com/local-inference-lab/rtx6kpro/blob/master/README.md#start-here) and the current model-family runbook.
3. Follow [`docs/newcomer-onboarding.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/docs/newcomer-onboarding.md): collect the exact runbook, image tag and digest, full launch/compose, GPU layout, TP/DCP/MTP/DSpark settings, the last relevant server log lines, client command, and observed output.
4. Search [`troubleshooting/common-issues.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/troubleshooting/common-issues.md) for the exact symptom or signature.
5. For transport, memory, or scaling failures, consult [`hardware/topology.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/topology.md), [`hardware/pcie-bandwidth.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/pcie-bandwidth.md), and the applicable hardware page.
6. For link-speed flapping behind c-payne Switchtec fabrics, consult [`troubleshooting/pcie-link-speed-flapping-cpayne.md`](https://github.com/local-inference-lab/rtx6kpro/blob/master/troubleshooting/pcie-link-speed-flapping-cpayne.md) and inspect [`scripts/pcie-link-supervisor.py`](https://github.com/local-inference-lab/rtx6kpro/blob/master/scripts/pcie-link-supervisor.py). Preserve its platform scope and verify-then-lock ordering: never lock a degraded link below its maximum.
7. Record a commit-pinned runbook URL in the report.

A troubleshooting entry is a lead, not proof that the current failure has the same cause. Reproduce against the current documented runbook and reduce the failing input before assigning ownership. Historical pages are appropriate when locating first-known-working and first-known-failing identities.
