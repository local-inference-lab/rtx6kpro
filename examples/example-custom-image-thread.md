# GLM example scheduler experiment

Fictional custom image that tests one scheduler-bound change against the recommended community image.

## Identity and status

- Release class: `community-derivative`
- Distribution role: `custom`
- Qualification: `qualified`
- Maintenance: `author-supported`
- Model family: GLM example family
- Image and digest: `registry.example.org/community/vllm:glm-example-scheduler-r1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
- Recommended image/control: `registry.example.org/community/vllm:glm-example-recommended-r4@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`

## Community runbook

- Wiki: https://github.com/local-inference-lab/rtx6kpro @ `4126fc06692c6ab042b0cd37d5062893fa402f47`
- Runbook: [README.md](https://github.com/local-inference-lab/rtx6kpro/blob/4126fc06692c6ab042b0cd37d5062893fa402f47/README.md)
- Relationship: Community wiki landing page used to locate current model-family guidance; this record remains a fictional format example.

## Based on

- Base image and digest: `registry.example.org/community/vllm:glm-example-recommended-r4@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
- Base credit: Fictional Local Inference Lab maintainers and contributors

## Build recipe

- Public recipe: https://example.org/lil-images/glm-example-scheduler/Dockerfile
- Recipe commit: `1111111111111111111111111111111111111111`
- Complete build command:
```bash
docker build --pull=false -f Dockerfile -t registry.example.org/community/vllm:glm-example-scheduler-r1 .
```

## Source commits, PRs, patches, and overlays

- **vllm:** https://github.com/example/vllm @ `2222222222222222222222222222222222222222`; release `N/A`
  - PR: https://github.com/example/vllm/pull/42 @ `3333333333333333333333333333333333333333` — Bound the fictional scheduler diagnostic queue; authors: Example Contributor
- **b12x:** https://github.com/example/b12x @ `4444444444444444444444444444444444444444`; release `N/A`

### Package changes
- N/A

### Build arguments
- VLLM_SOURCE_COMMIT=2222222222222222222222222222222222222222
- B12X_SOURCE_COMMIT=4444444444444444444444444444444444444444

### Environment defaults
- MAX_NUM_SEQS=30
- CUDAGRAPH_MODE=FULL

### Entrypoint changes
- N/A

- Result tree: `5555555555555555555555555555555555555555`
- Integration patch: `sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc`

## Changes from the base image

### Inherited
- The recommended base supplies the fictional GLM serving profile, B12X kernels, and target-only launcher.

### Introduced
- The scheduler diagnostic snapshot is capped at 1,024 entries.

### Compatibility impact
- Diagnostic clients receive no more than 1,024 pending entries; serving admission behavior is unchanged.

## Tested configuration

### Four fictional Blackwell GPUs, target-only
- Hardware: 4 x fictional SM120 GPUs with 96 GiB each
- Topology: Single-socket PCIe topology; peer access enabled
- Power/clocks: 300 W per GPU; stock clocks
- Driver/runtime: Example driver 999.1; CUDA 13.2; PyTorch 2.12.0; NCCL 2.30.0
- Engine: example/vllm@2222222222222222222222222222222222222222 with example/b12x@4444444444444444444444444444444444444444
- Model/quant: example/glm-model@6666666666666666666666666666666666666666; NVFP4 weights; BF16 activations
- Parallelism: TP4, DCP1, EP1, DP1
- KV/speculation: FP8 target KV; 262,144-token model limit; Disabled; no MTP, external draft, or n-gram speculation
- Graph/scheduler: FULL CUDA graph; max_num_seqs=30; max_num_batched_tokens=4096
- Cache/JIT: Fresh dedicated JIT volume for candidate and a separate fresh volume for baseline
- Launch command:
```bash
docker run --rm --gpus all --network host -e TP=4 -e MAX_NUM_SEQS=30 -e CUDAGRAPH_MODE=FULL registry.example.org/community/vllm:glm-example-scheduler-r1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
```

## Validation results

### Commands
- `python -m pytest tests/test_scheduler.py -q`
- `python3 llm_decode_bench.py --port 8000 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output candidate.json`
- `python3 llm_decode_bench.py --port 8001 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output baseline.json`

### Results
- **smoke / Image startup:** passed. Conditions: Fresh container and dedicated JIT volume.. Measurement: Health request after server readiness.. Result: Server reached healthy state and completed one request.. Conclusion: The tested image starts under the listed configuration.. Evidence: https://example.org/evidence/custom-r1-startup.txt (`sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd`).
- **correctness / Scheduler regression checks:** passed. Conditions: Clean source tree at the listed result tree.. Measurement: Three deterministic scheduler unit tests.. Result: 3 passed.. Conclusion: The diagnostic bound preserves tested short-queue behavior.. Evidence: https://example.org/evidence/custom-r1-correctness.txt (`sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`).
- **stability / Thirty-request stability run:** passed. Conditions: Target-only profile with max_num_seqs=30.. Measurement: Thirty concurrent requests repeated three times.. Result: 90 of 90 requests completed; zero restarts.. Conclusion: No instability was observed in the tested stress configuration.. Evidence: https://example.org/evidence/custom-r1-stability.json (`sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff`).
- **performance / Target-only decode regression:** passed. Conditions: Candidate and recommended control on the same hardware with separate fresh JIT volumes.. Measurement: Five independent 30-second runs at concurrency 1, 4, 16, and 30.. Result: Candidate aggregate decode remained within 1.2% of the recommended control across tested cells.. Conclusion: No material target-only decode regression was observed under the listed configuration.. Evidence: https://example.org/evidence/custom-r1-performance.json (`sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef`).

## Performance claims

### Target-only decode parity
- Candidate: `registry.example.org/community/vllm:glm-example-scheduler-r1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
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
- Changed variables: Scheduler diagnostic snapshot bound
- Nuisance variables: Server warmup and cache state, Thermal and clock drift, Request scheduling variation
- Rival explanations: Ordinary control run variation, Run-order drift, An unintended launch difference
- Falsification condition: A reproducible per-cell regression outside repeated-control variation
- Repeated-control variation: Per-cell repeated-control ranges were 0.7% to 1.4%
- Absolute effect: Candidate minus control ranged from -14 to +9 aggregate tokens/second across tested cells
- Relative effect: Candidate minus control ranged from -1.2% to +0.7%; every tested cell was within 1.2% of control
- Uncertainty: Across five independent runs per image, per-cell median absolute deviation was 0.3% to 0.8%; no population-level interval was claimed
- Candidate command:
```bash
python3 llm_decode_bench.py --port 8000 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output candidate.json
```
- Control command:
```bash
python3 llm_decode_bench.py --port 8001 --skip-prefill --contexts 0,8k,32k --concurrency 1,4,16,30 --duration 30 --output baseline.json
```
- Raw results: https://example.org/evidence/custom-r1-performance.json (`sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef`)
- Result: All tested cells were within 1.2% of the control.
- Conclusion: The diagnostic-only change did not materially affect target-only decode on the tested system.

## Known limitations

- This is a fictional documentation example and is not a runnable image.

## Untested configurations

- Other GPU counts, SGLang, speculative decoding, and models outside the example family were Not tested.

## Unsupported configurations

- Production use of this fictional image is unsupported.

## Support and issue routing

- Support owner: Example Image Maintainer
- Contact: https://example.org/community/example-image-maintainer
- Support commitment: `maintained`
- Support thread: https://discord.com/channels/111111111111111111/222222222222222222/333333333333333333
- Thread status: `active`
- Issue tracker: https://example.org/issues/glm-example-scheduler
- Upstream escalation: Keep reports in this thread. Escalate upstream only after reproduction on the current recommended image under an equivalent configuration or a minimal reproducer identifies the responsible source change.
- Superseded by: `N/A`

## Publication record

- Machine-readable record: https://example.org/records/example-glm-custom-r1.json
- Main-channel link: https://discord.com/channels/111111111111111111/444444444444444444/555555555555555555
- Automated listing: `not-applicable`
- Maintainer approval: N/A
