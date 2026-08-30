# Baseline Suite Selection

Use one suite per decision. Running more tests is not automatically better; a smaller controlled suite is preferable to a larger mixed comparison.

## Selection table

| Decision | Required suite | Optional additions |
|---|---|---|
| Early exploratory change | Target-only decode, startup smoke, one of Estonia or LAVD | Short qualitative prompt |
| New model or quant for community testing | Target-only decode, Estonia, LAVD, exact launch/config record | Hotel Lights, pinned dataset accuracy |
| Stronger correctness/stability qualification | Standard suite plus Hotel Lights and one repeated control | Qualitative coding prompt, long-context accuracy |
| Quant or kernel A/B | Target-only candidate/control, five independent repetitions for tail claims, same-stack reasoning checks | Dataset paired A/B |
| Speculative decoding | Target-only control, then each MTP/DFlash/DSpark mode separately | Acceptance-normalized comparison |
| Prefill/context scaling | Explicit context sweep with decode reported separately | Fabric diagnostics |
| Dataset accuracy | Pinned dataset profile and paired per-item A/B | Repeated self-control to estimate noise |

## Quick baseline

Use when GPU time is limited or the contribution remains exploratory:

1. Target-only decode with every speculative path disabled.
2. Estonia or LAVD at a declared reasoning effort, concurrency, and run count.
3. Startup/import smoke test and one representative request.

State why the selected long-reasoning profile is sufficient and mark the other profiles `Not tested`.

## Standard model or quant baseline

Use for a new model, new quant, or materially changed runtime intended for community testing:

1. Target-only decode matrix with no speculative path.
2. Estonia at the agreed reasoning effort. The commonly requested stress form is concurrency 30 with 30 measured runs.
3. LAVD under the same declared reasoning effort and a stated concurrency/run count.
4. Exact model, quant, image, source, and server-launch identities.
5. Raw JSON and concise report.

Hotel Lights is more resource-intensive. Run it for the full suite, when Estonia and LAVD disagree, or when stronger qualification is required.

## Quant or runtime A/B

Keep these fixed unless the comparison explicitly studies them:

- GPU hardware, count, topology, power limits, and clocks;
- driver, CUDA, PyTorch, NCCL, and container base;
- engine/component commits and model revision;
- TP/DCP/CP/EP/DP, graph mode, scheduler, and KV cache;
- benchmark commit, prompts, contexts, concurrency, output limit, and duration;
- JIT/cache warmup and persistent cache policy.

Change only the declared variable. When several variables change, label the result a **system comparison**, not a component attribution.

For p50/p95/p99, outlier, or small-regression claims:

- run at least five independent repetitions unless the selected protocol requires more;
- publish every per-run result;
- state whether percentiles are pooled, averaged, or calculated per run then aggregated;
- compare the control against itself when interpreting small accuracy differences.

## Target-only versus speculative

1. Run target-only decode.
2. Run each MTP, DFlash, DSpark, or other draft configuration separately.
3. Keep the target stack fixed.
4. Report raw tokens/second, acceptance, and acceptance-normalized steps/second separately.
5. State whether the objective is low-latency C1, high-concurrency throughput, or both.

## Selection record

Before expensive work, record:

```text
Decision being tested:
Selected suite:
Candidate:
Control:
Changed variable(s):
Profiles included:
Profiles skipped:
Reasoning effort:
Concurrency and runs:
Expected support boundary:
```

Skipped profiles are `Not tested`, never passed.
