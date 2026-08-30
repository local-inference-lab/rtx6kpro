# Image Qualification and Performance Claims

Use the current commit-pinned `local-inference-lab/rtx6kpro` model runbook to select the documented launch and expected qualification routes. Re-run the checks; a prior wiki result is not proof that the candidate passed.

## Tested configuration

For every qualified profile state:

- GPU model/count, interconnect, topology, NUMA placement, power, and clocks;
- driver, CUDA/runtime, PyTorch, NCCL, serving-engine commit, and kernel backends;
- model repository and immutable revision;
- quantization and weight/activation format;
- TP, DCP/CP, EP, DP, and other parallelism;
- target and draft KV-cache formats and capacities;
- speculative method, draft checkpoint/revision, and lookahead, or `disabled`;
- graph mode and captured shapes;
- model length, batched/scheduled tokens, sequences, and chunks;
- JIT/cache volume policy;
- exact launch or Compose command.

Distinct graph or generated-kernel contracts require distinct writable caches and distinct qualification entries.

## Validation gate

Run and record:

1. clean reproducible build;
2. image-history and installed-package/source identity inspection;
3. import/startup smoke tests;
4. focused regression tests for introduced behavior;
5. correctness and stability tests;
6. performance regression tests where claimed;
7. comparison with the exact base or source-identical control;
8. immutable push-digest resolution.

Use `Not tested` for any omitted test. `qualified` is unavailable until the exact advertised profile passes every required gate with `qualification-evidence`.

## Evidence strength

Classify every result before using it in publication:

- `exploratory`: protocol, primary measurement, comparison, or stopping rule was selected or changed after inspecting results;
- `confirmatory`: the question, experimental unit, candidate/control, changed variables, primary measurement, repetitions, order, exclusions, stopping rule, and falsification condition were declared before the target result;
- `qualification-evidence`: confirmatory evidence that also covers the exact advertised profile and all required correctness, stability, compatibility, and regression gates.

An implemented feature, a successful smoke test, or an exploratory speed result is not `qualification-evidence`.

For candidate/control performance comparisons:

- define the independent experimental unit and do not count requests within one server process as independent server replications;
- list rival explanations and nuisance variables, including topology, clocks, thermals, cache/JIT state, scheduling, and run order;
- change one factor for causal attribution, or label the result a system comparison;
- alternate or randomize run order and block by machine/topology where practical;
- record repeated-control variation for every qualified performance claim and use enough repeated controls to resolve a small claimed delta;
- report absolute and relative effect magnitude, uncertainty, failures, exclusions, and every independent run;
- keep heterogeneous machines, topologies, model revisions, workloads, and protocols separate unless the design explicitly models those factors.

Apply the same validity standard to favorable and unfavorable results. A critical confounder makes the associated claim unavailable even when every command completed successfully.

## Performance claim requirements

Every claim includes:

- candidate and baseline image digests;
- benchmark repository and exact commit;
- exact candidate and baseline commands;
- hardware, topology, runtime, model revision, quant, and server configuration;
- concurrency, input lengths, output length or duration, warmup, experimental unit, independent repetitions, run order, and aggregation;
- raw result files and hashes, including failed or contrary runs;
- explicitly changed variables, nuisance variables, rival explanations, and repeated-control noise;
- absolute and relative effect magnitude with uncertainty;
- `confirmatory`, `exploratory`, or `qualification-evidence` classification;
- narrow conclusion and limitations.

Keep prefill, sustained decode, finite Burst/E2E, accuracy, and speculative results separate. For target-only decode, verify all speculative paths are disabled in the running server. An environment value such as `MTP=0` is insufficient when an external speculative configuration remains active.

The experimental-design and evidence-review principles above adapt K-Dense AI's MIT-licensed `hypothesis-generation`, `experimental-design`, `statistical-analysis`, and `scientific-critical-thinking` skills at commit [`f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f`](https://github.com/K-Dense-AI/scientific-agent-skills/tree/f6fcafeb1cc8c82eca0160a18bc41c38427b8e0f) to inference image qualification.
