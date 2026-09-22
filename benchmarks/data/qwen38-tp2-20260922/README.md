# Qwen TP2 measurement and trace artifacts

These files support the [Qwen TP2 comparison](../../qwen38-tp2-sglang.md)
and [CPU-cache restore checks](../../karmic-public-feedback.md).
The reports define checkpoint, hardware, clocks, source revisions, sampling,
and limits. `SHA256SUMS` records the artifact checksums.

## Decode and prefill

- `sglang-tp2-matched-{1..5}.json`: community SGLang.
- `vllm-b12x-tp2-matched-{1..5}.json`: published vLLM beta.
- `vllm-pynccl-tp2-matched-{1..5}.json`: collective-only diagnostic control.
- `vllm-gdn-uniform-tp2-matched-{1..5}.json`: the same beta with uniform-graph
  attention-metadata staging removed.
- `qwen-tp2-decode-summary.json`: medians and individual samples.
- `*-tp2-prefill32k.json`: warmed, uncached 30-second prefill controls.

The `hardware` data in benchmark JSON describes the benchmark client's local
machine, **not** the remote two-GPU server. Use the server conditions in the
report. `server_accept_len_effective` measures output per engine step during
the scored window; SGLang's final instantaneous acceptance gauge does not.

The client is the working copy based on benchmark commit
`bdc96c125b522ec65ef29f01570f443fffae1cdc` with
`llmbench-error-cells.patch` applied. The patch labels failed streams as errors
and changes the displayed version to 0.6.2; it does not change successful
throughput calculation. All compared arms used this same client. The resulting
`llm_decode_bench.py` SHA-256 is
`053989edff8c9c93e2b96e61342b2ffbd9851e03deba17e6d3fc96fcd6694c1e`.

`run-qwen-matched-llmbench-20260922.py` runs that benchmark while
setting temperature 1, top-p .95, top-k disabled and medium reasoning in both
engines' chat requests. Install the benchmark dependencies and set
`LLM_BENCH_PATH` to its `llm_decode_bench.py` when it is not at the recorded
`/root/llm-inference-bench` location. Example, after starting one server:

```bash
LLM_BENCH_PATH=/path/to/llm-inference-bench/llm_decode_bench.py \
python run-qwen-matched-llmbench-20260922.py \
  --host SERVER --port PORT --model Qwen3.8-Flash-Next \
  --contexts 0 --concurrency 1,8 --duration 30 \
  --decode-warmup-seconds 10 --max-tokens 32768 --temperature 1 \
  --skip-prefill --output measurement.json
```

Run five times per arm on the same GPU pair. The original launcher scripts
record the exact tested arguments and source-overlay diagnostic; their
machine-specific GPU IDs, model paths, names and ports must be adapted before
reuse. They are evidence, not universal production defaults.

## Profiles and correctness

`qwen-{sglang,vllm-b12x,vllm-gdn-uniform}-tp2-c8-rank0-20260922.trace.json.gz`
contains eight decode steps at concurrency 8. Open a decompressed trace in a
Chrome-trace-compatible viewer. `qwen_trace_totals.py` produces the included
operation summary; nested CPU durations and summed GPU kernels are not
end-to-end latency.

`qwen-gdn-uniform-mixed-smoke.jsonl` records 16 concurrent text/structured-output
checks. `qwen-simple-cumem-mtp-restore-{a,b}.jsonl` records explicit native-cache
reset followed by genuine CPU restores. `qwen-native-profile-pressure-restore.jsonl`
uses the unified launcher and ten intervening prompts instead of a reset.
The restore must have positive external-hit deltas, zero native-hit deltas,
and the correct answer; a fast GPU prefix hit is not CPU-cache evidence.

All request content here is synthetic validation data.
