# Local Inference Lab Contribution Playbook

This playbook keeps rapid experimentation composable. It does not require every contribution to become a pull request, and it does not make every experimental image a general-user recommendation.

## Choose the route

| Work being shared | Skill | Normal output |
|---|---|---|
| Patch, changed files, script, reproducer, or evidence | `local-inference-lab-sharing-changes` | Validated portable ZIP and concise post |
| Docker image | `local-inference-lab-publishing-docker` | Image record, required support thread, and appropriate announcement |
| Model/quant/runtime test | `local-inference-lab-running-benchmarks` | Raw JSON, run record, and bounded report |
| One-shot coding comparison | `local-inference-lab-evaluating-prompts` | Exact prompt, generated files, rubric observations, and intervention record |
| Merge/rebase/composition | `local-inference-lab-reconciling-changes` | Completed integration and resolution record |
| Failure or regression | `local-inference-lab-reporting-bugs` | Minimal reproducer and evidence-based owner route |
| Explicit issue or PR request | `local-inference-lab-github-contributions` | Focused stand-alone issue/PR |

## Use the community runbook first

For RTX PRO 6000 Blackwell and SM120 work, resolve the current commit of `local-inference-lab/rtx6kpro`, start from its front-page model-family hubs, and record the relevant commit-pinned runbook URL. Use current hubs for recommended launch and support context. Use versioned pages, benchmark histories, and daily summaries only when reproducing the exact named result.

The wiki supplements direct evidence; it does not replace image digests, source commits, launch commands, raw benchmark output, logs, or minimal reproducers.

## Portable packages are first-class contributions

A contributor may share a patch, Git bundle, file set, script, reproducer, benchmark result, or experimental evidence directly. A complete package identifies its exact base, introduced delta, authorship, application and reversal commands, validation, limitations, and support expectation. GitHub is optional.

Prefer the smallest complete representation:

- committed work: mail-formatted Git patches;
- uncommitted work: full-index binary-capable diff;
- branch history: Git bundle;
- non-Git work: files with old/new hashes and exact destinations;
- container overlay: immutable base digest plus every applied patch/file and build step.

## Evidence method

Use the same compact method for benchmarks, prompt comparisons, regressions, image qualification, and competing implementation choices:

1. Separate the direct observation from its interpretation, then state one answerable question.
2. Declare the independent experimental unit, candidate, control, changed variables, held-constant variables, and nuisance variables.
3. List credible rival explanations. For each candidate explanation, state a prediction and the result that would falsify it.
4. Predeclare the primary measurement, independent repetitions, balanced or randomized run order, stopping rule, exclusions, and confirmatory or exploratory status. Preserve failed and contrary runs.
5. Prefer one-factor comparisons. When several factors intentionally change, call it a system comparison; use a factorial design only when interactions matter and the run budget supports it.
6. Report raw observations before interpretation, the absolute and relative effect, uncertainty, repeated-control variation, and critical validity threats. Bound the conclusion to the tested matrix.

Independent submeasurements inside one server run, host, model response, or other experimental unit are not independent replications. Treat cache/JIT state, thermal or clock drift, request scheduling, run order, and configuration drift as possible nuisances. A qualification claim needs controlled, repeated evidence; an exploratory result can direct the next test but cannot establish general support.

When evidence is combined, weight it by design validity and independence rather than row count. Keep materially different hardware, topologies, images, models, workloads, and protocols separate unless the comparison justifies synthesis. Report heterogeneity and unresolved alternatives instead of averaging disagreement into a false consensus.

This method adapts the MIT-licensed `hypothesis-generation`, `experimental-design`, `statistical-analysis`, and `scientific-critical-thinking` skills from K-Dense AI's [`scientific-agent-skills`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f) at commit `f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f` to local-inference work; it is not a separate science-specific workflow.

## Docker images

Images with incomplete provenance may be withheld or removed from main model-channel linking until corrected.

### Recommended image

Each model family has one recommended, maintainer-supported image. It is the general-user default and the target for main announcements and the server's automated image listing or bot. Replacement requires maintainer approval and passed correctness, stability, and performance regression gates.

### Custom image

A custom image is an independent experiment or derivative. It has its own support thread and may receive one short link from the model channel. The thread owns launch help, revisions, image-specific benchmarks, and reports.

A custom image starts as an ephemeral test build. Its author may explicitly accept maintenance and first-line support. When the relevant work is incorporated into the recommended image, mark the custom image and thread `superseded` and name the exact replacing digest.

### Provenance and publication

Publish:

- final and base `image@sha256:<digest>` identities;
- public Dockerfile/build script and exact recipe commit;
- complete build command;
- exact engine, B12X, and other component commits/releases;
- every PR, patch, overlay, package change, build argument, default, and entrypoint change;
- inherited versus introduced behavior;
- complete tested hardware/configuration and launch command;
- validation commands, results, evidence, limitations, and support owner;
- directly comparable baseline and exact commands for every performance claim.

An image without sufficient provenance should not be linked from main channels until corrected.

## Support and escalation

Custom-image reports remain in the custom support thread. Escalate only when:

1. the problem reproduces on the current recommended image under an equivalent configuration; or
2. a minimal reproducer independently identifies the responsible source change.

Testing another custom derivative is not a replacement for the recommended control.

## Benchmark baseline

For a new model, quant, or material runtime change, the normal baseline is:

1. target-only decode with all speculative paths disabled;
2. Estonia;
3. LAVD;
4. Hotel Lights for the full or stronger suite;
5. optional qualitative coding prompts in a separate report.

`MTP=0` does not prove target-only operation when an external speculative configuration remains active. Restart without every draft/speculative path and verify the running server.

Performance comparisons pin candidate, control, benchmark commit, hardware, launch/configuration, contexts, concurrency, duration/output limit, repetitions, aggregation, and changed variables. Tail claims normally use at least five independent repetitions.

## Communication standard

Keep the public post short. Put commands, source identities, full tables, and raw evidence in the package or support thread.

Every technical document must make sense from the current artifact and repository state. State resulting behavior, technical reason, compatibility impact, validation, and limitations. Do not require readers to reconstruct an earlier conversation.
