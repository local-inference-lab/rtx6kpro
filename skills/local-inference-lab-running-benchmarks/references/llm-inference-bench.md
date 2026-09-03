# Running `local-inference-lab/llm-inference-bench`

The repository changes frequently. Treat its selected commit, `README.md`, `CHANGELOG.md`, and `python3 llm_decode_bench.py --help` output as the command source of truth for each run.

## Pin and inspect the tool

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench.git
cd llm-inference-bench
git status --short
git rev-parse HEAD
python3 llm_decode_bench.py --help
```

When using an existing checkout, inspect local modifications before updating it. Do not discard user work. Record the full benchmark commit in every result package and keep result files associated with that commit.

Install only dependencies documented by the selected commit.

## Capture the serving endpoint

Record before benchmarking:

- engine, component repositories, and full commits;
- immutable image reference when containerized;
- model repository and immutable revision;
- quantization and weight/activation formats;
- GPU model/count, topology, power limits, clocks, driver, CUDA, PyTorch, and NCCL;
- TP/DCP/CP/EP/DP and communication backend;
- KV-cache format/capacity, graph mode, scheduler limits, and maximum context;
- exact launch command and relevant environment variables;
- JIT/cache volume, warmup state, and cache reuse policy.

Run a readiness request and verify that the endpoint reports the intended model.

## Verify target-only decode

A raw target control disables every speculative path:

- MTP;
- DFlash or DSpark external drafts;
- n-gram speculation;
- any other draft checkpoint or speculative configuration.

`MTP=0` is insufficient when an external `--speculative-config` remains active. Remove or disable that configuration, restart the server, and verify the server metadata or logs show target-only decoding.

Example shape, adjusted to the endpoint capacity and the selected benchmark commit:

```bash
python3 llm_decode_bench.py \
  --port <port> \
  --model <served-model-name> \
  --skip-prefill \
  --contexts 0,8k,16k,32k \
  --concurrency 1,2,4,8,16,32 \
  --duration 30 \
  --output target-only-decode.json
```

Record capacity skips and effective concurrency. Do not silently remove requested cells.

## Community reasoning profiles

Resolve profile names and arguments from the selected commit's `--help`. Common profiles include:

- `estonia` and `estonia-long` for repeated long-answer reasoning and completion-token statistics;
- `lavd-test` or an alias such as `lavd` for long structured-context consistency;
- `hotel-lights` or aliases such as `hotel` and `lights` for a compact reasoning consistency test.

Examples after confirming the selected commit supports them:

```bash
python3 llm_decode_bench.py \
  --port <port> \
  --model <served-model-name> \
  --test-profile estonia \
  --profile-concurrency 30 \
  --profile-runs 30 \
  --reasoning-effort high \
  --output estonia-high.json

python3 llm_decode_bench.py \
  --port <port> \
  --model <served-model-name> \
  --test-profile lavd-test \
  --profile-concurrency 30 \
  --profile-runs 30 \
  --reasoning-effort high \
  --output lavd-high.json

python3 llm_decode_bench.py \
  --port <port> \
  --model <served-model-name> \
  --test-profile hotel-lights \
  --profile-concurrency 30 \
  --profile-runs 30 \
  --reasoning-effort high \
  --output hotel-lights-high.json
```

The commonly requested Estonia community stress convention is concurrency 30, 30 measured runs, and at least 28 correct results. Record it as a community qualification convention rather than a universal scientific threshold. Publish the actual score, completion-token percentiles, truncations, failures, and conditions.

Use the scorer and expected values embedded in the selected benchmark commit for LAVD and Hotel Lights. Do not reimplement or silently alter their scoring.

## Dataset accuracy

When the selected commit supports pinned dataset profiles such as GSM8K, MMLU-Pro, or GPQA Diamond:

- use the repository-provided pinned inputs and scorer;
- preserve per-item results;
- keep temperature, prompt format, output limits, and concurrency fixed;
- use paired per-item comparison where supported;
- distinguish truncation from unparseable output;
- run the control against itself to estimate batching/non-determinism noise before interpreting small deltas.

## Speculative decoding

Run speculation only after target-only control. For every mode, record:

- draft model/revision or in-checkpoint method;
- requested speculative token count;
- draft and target KV formats;
- raw aggregate and per-request throughput;
- acceptance length/rate and per-position acceptance when available;
- acceptance-normalized engine steps/second;
- graph, cache, and scheduler differences from the target-only control.

Do not present raw speculative tokens/second as target engine speed.

## Prefill and context scaling

Keep prefill and decode in separate tables. Record prompt tokens, TTFT, prefill tokens/second, cache state, contexts, repetitions, and whether the measurement is client- or server-derived.

## Output integrity

For every JSON result:

```bash
sha256sum <result>.json
```

Preserve relevant server logs and hash them. Never publish tokens, credentials, private hostnames, or personal filesystem paths.
