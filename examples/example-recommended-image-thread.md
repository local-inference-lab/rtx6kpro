# GLM example recommended image r5

Fictional maintainer-supported image that incorporates the tested scheduler bound and replaces r4.

## Identity and status

- Release class: `official`
- Distribution role: `recommended`
- Qualification: `qualified`
- Maintenance: `maintainer-supported`
- Model family: GLM example family
- Image and digest: `registry.example.org/community/vllm:glm-example-recommended-r5@sha256:9999999999999999999999999999999999999999999999999999999999999999`
- Recommended image/control: `registry.example.org/community/vllm:glm-example-recommended-r5@sha256:9999999999999999999999999999999999999999999999999999999999999999`

## Community runbook

- Wiki: https://github.com/local-inference-lab/rtx6kpro @ `4126fc06692c6ab042b0cd37d5062893fa402f47`
- Runbook: [README.md](https://github.com/local-inference-lab/rtx6kpro/blob/4126fc06692c6ab042b0cd37d5062893fa402f47/README.md)
- Relationship: Community wiki landing page used to locate current model-family guidance; this record remains a fictional format example.

## Based on

- Base image and digest: `registry.example.org/community/vllm:glm-example-recommended-r4@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
- Base credit: Fictional Local Inference Lab maintainers and contributors

## Build recipe

- Public recipe: https://example.org/lil-images/glm-example-recommended/Dockerfile
- Recipe commit: `8888888888888888888888888888888888888888`
- Complete build command:
```bash
docker build --pull=false -f Dockerfile -t registry.example.org/community/vllm:glm-example-recommended-r5 .
```

## Source commits, PRs, patches, and overlays

- **vllm:** https://github.com/example/vllm @ `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`; release `N/A`
  - PR: https://github.com/example/vllm/pull/42 @ `3333333333333333333333333333333333333333` — Bound the fictional scheduler diagnostic queue; authors: Example Contributor
- **b12x:** https://github.com/example/b12x @ `4444444444444444444444444444444444444444`; release `N/A`

### Package changes
- N/A

### Build arguments
- VLLM_SOURCE_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
- B12X_SOURCE_COMMIT=4444444444444444444444444444444444444444

### Environment defaults
- MAX_NUM_SEQS=30
- CUDAGRAPH_MODE=FULL

### Entrypoint changes
- N/A

- Result tree: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
- Integration patch: `sha256:abababababababababababababababababababababababababababababababab`

## Changes from the base image

### Inherited
- The preceding recommended image supplies the fictional GLM profile and qualified B12X runtime.

### Introduced
- The tested scheduler diagnostic bound is incorporated into the recommended source composition.

### Compatibility impact
- Diagnostic clients receive at most 1,024 pending entries; serving defaults remain unchanged.

## Tested configuration

### Recommended four-GPU qualification
- Hardware: 4 x fictional SM120 GPUs with 96 GiB each
- Topology: Single-socket PCIe topology; peer access enabled
- Power/clocks: 300 W per GPU; stock clocks
- Driver/runtime: Example driver 999.1; CUDA 13.2; PyTorch 2.12.0; NCCL 2.30.0
- Engine: example/vllm@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa with example/b12x@4444444444444444444444444444444444444444
- Model/quant: example/glm-model@6666666666666666666666666666666666666666; NVFP4 weights; BF16 activations
- Parallelism: TP4, DCP1, EP1, DP1
- KV/speculation: FP8 target KV; 262,144-token model limit; Disabled for target-only control; separately qualified MTP3 profile recorded in evidence
- Graph/scheduler: FULL CUDA graph; max_num_seqs=30; max_num_batched_tokens=4096
- Cache/JIT: Dedicated fresh JIT volume for every compared profile
- Launch command:
```bash
docker run --rm --gpus all --network host -e TP=4 -e MAX_NUM_SEQS=30 -e CUDAGRAPH_MODE=FULL registry.example.org/community/vllm:glm-example-recommended-r5@sha256:9999999999999999999999999999999999999999999999999999999999999999
```

## Validation results

### Commands
- `python -m pytest tests/test_scheduler.py -q`
- `python3 llm_decode_bench.py --port 8000 --test-profile estonia --profile-concurrency 30 --profile-runs 30 --reasoning-effort high --output estonia.json`
- `python3 llm_decode_bench.py --port 8000 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output candidate.json`
- `python3 llm_decode_bench.py --port 8001 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output baseline.json`

### Results
- **correctness / Recommended correctness gate:** passed. Conditions: Clean recommended image and listed model revision.. Measurement: Focused unit tests plus Estonia 30 by 30 high-reasoning run.. Result: Focused tests passed and Estonia scored 29 of 30.. Conclusion: The listed correctness gate passed under the qualified configuration.. Evidence: https://example.org/evidence/recommended-r5-correctness.json (`sha256:1212121212121212121212121212121212121212121212121212121212121212`).
- **stability / Recommended stability gate:** passed. Conditions: Thirty concurrent requests with fresh cache and three repetitions.. Measurement: Ninety requests plus server health and fatal-signature monitoring.. Result: 90 of 90 completed; zero restarts and zero fatal signatures.. Conclusion: The listed stability gate passed.. Evidence: https://example.org/evidence/recommended-r5-stability.json (`sha256:3434343434343434343434343434343434343434343434343434343434343434`).
- **performance / Recommended performance regression gate:** passed. Conditions: r5 and r4 on identical hardware and fresh dedicated JIT volumes.. Measurement: Five independent target-only decode sweeps.. Result: No tested cell regressed by more than 1.5%; C30 improved by 0.8%.. Conclusion: The recommended performance regression gate passed for the tested matrix.. Evidence: https://example.org/evidence/recommended-r5-performance.json (`sha256:5656565656565656565656565656565656565656565656565656565656565656`).
- **smoke / Image startup:** passed. Conditions: Fresh container and JIT volume.. Measurement: Readiness and representative generation request.. Result: Healthy startup and successful request.. Conclusion: The image starts under the published launch configuration.. Evidence: https://example.org/evidence/recommended-r5-startup.txt (`sha256:7878787878787878787878787878787878787878787878787878787878787878`).

## Performance claims

### Recommended r5 target-only regression comparison
- Candidate: `registry.example.org/community/vllm:glm-example-recommended-r5@sha256:9999999999999999999999999999999999999999999999999999999999999999`
- Control: `registry.example.org/community/vllm:glm-example-recommended-r4@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
- Benchmark: https://github.com/local-inference-lab/llm-inference-bench @ `7777777777777777777777777777777777777777`
- Hardware/model: 4 x fictional SM120 GPUs; identical power and clocks; example/glm-model@6666666666666666666666666666666666666666
- Concurrency/lengths: 1,4,16,30; 0,8k,32k tokens; 30 seconds per cell
- Evidence class: `qualification-evidence`
- Experimental unit: One independently started and warmed server process on the fixed four-GPU host
- Independent repetitions: 5
- Run order: Predeclared randomized ten-run order with five candidate and five control server starts
- Stopping rule: Five independently started server runs per image
- Exclusions: None; failed runs would remain in the raw evidence
- Aggregation: Median aggregate tokens/second per cell; every run retained
- Changed variables: Scheduler diagnostic snapshot bound incorporated in the recommended source composition
- Nuisance variables: Server warmup and cache state, Thermal and clock drift, Request scheduling variation
- Rival explanations: Ordinary control run variation, Run-order drift, An unintended launch difference
- Falsification condition: A reproducible regression beyond the declared gate and repeated-control variation
- Repeated-control variation: Per-cell repeated-control ranges were 0.8% to 1.7%
- Absolute effect: Candidate minus control ranged from -18 to +11 aggregate tokens/second across tested cells
- Relative effect: Candidate minus control ranged from -1.5% to +0.8%; C30 improved by 0.8%
- Uncertainty: Across five independent runs per image, per-cell median absolute deviation was 0.4% to 0.9%; no population-level interval was claimed
- Candidate command:
```bash
python3 llm_decode_bench.py --port 8000 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output candidate.json
```
- Control command:
```bash
python3 llm_decode_bench.py --port 8001 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output baseline.json
```
- Raw results: https://example.org/evidence/recommended-r5-performance.json (`sha256:5656565656565656565656565656565656565656565656565656565656565656`)
- Result: No cell regressed by more than 1.5%; C30 improved by 0.8%.
- Conclusion: r5 passed the target-only performance regression gate on the tested matrix.

## Known limitations

- This is a fictional documentation example and is not a runnable image.

## Untested configurations

- Other GPU counts, SGLang, and models outside the example family were Not tested.

## Unsupported configurations

- Production use of this fictional image is unsupported.

## Support and issue routing

- Support owner: Example Maintainer Team
- Contact: https://example.org/community/maintainers
- Support commitment: `maintained`
- Support thread: https://discord.com/channels/111111111111111111/222222222222222222/666666666666666666
- Thread status: `active`
- Issue tracker: https://example.org/issues/glm-example-recommended
- Upstream escalation: Use the recommended-image support thread for first-line reports and route source issues only after a minimal reproducer identifies the responsible component.
- Superseded by: `N/A`

## Publication record

- Machine-readable record: https://example.org/records/example-glm-recommended-r5.json
- Main-channel link: https://discord.com/channels/111111111111111111/777777777777777777/888888888888888888
- Automated listing: `listed`
- Maintainer approval: https://example.org/releases/example-glm-recommended-r5
