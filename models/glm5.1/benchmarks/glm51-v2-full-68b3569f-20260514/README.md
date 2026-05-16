# GLM-5.1 v2 Full Matrix Raw Artifacts

Raw benchmark artifacts for the GLM-5.1 v2 page:

```text
models/glm5.1_v2.md
```

Measured on 2026-05-15 with:

```text
voipmonitor/vllm:glm-kimi-canonical-rebase-layered-vllm68b3569f-b12xc929144-flashinfergit1a60071-cutedsl45-20260514
```

Each `glm-dcp*-mtp*` directory contains:

| File | Meaning |
|---|---|
| `decode-matrix.json` | Machine-readable decode matrix output from `llm_decode_bench.py`. |
| `decode-matrix.log` | Console rendering of the decode matrix. |
| `warmup-128k-cc1.json` | 128k/cc1 warmup output, when the KV budget could fit it. |
| `warmup-128k-cc1.log` | Console rendering of the 128k/cc1 warmup. |
| `startup.log` | vLLM startup log, including NCCL, DCP, CUDA graph, and allreduce path. |
| `final.log` | Full server log captured after the benchmark run. |
| `docker-run.env` | Minimal launch metadata captured by the driver. |
| `kv-budget.txt` | Manual `llm_decode_bench.py --kv-budget` value used by the archived driver. In this run it is `GPU KV cache size / 4`, not the vLLM KV capacity. |
| `container.id` | Docker container ID used for that profile. |

Top-level files:

| File | Meaning |
|---|---|
| `driver.log` | Full benchmark driver output across all profiles. |
| `matrix.log` | Matrix generation log. |
| `watcher.log` | Runtime watcher log. |
| `driver.exit` | Driver exit status. |

Important caveat: this run used `--skip-prefill`. Standalone prefill-only
throughput was not measured. TTFT-derived prompt/token rates from these JSON
files should not be treated as prefill throughput because prefix-cache state and
request admission can skew them heavily.

KV capacity caveat: use `startup.log` / `final.log` line `GPU KV cache size:
<N> tokens` as the authoritative vLLM KV cache capacity. The archived
`kv-budget.txt` values were conservative benchmark fit/skip gates and are not
directly comparable to vLLM startup logs.

PCIe caveat: the JSON startup diagnostics record that NVIDIA P2P override was
effective, patched NCCL PR2127 was used, and vLLM C++ PCIe custom allreduce was
enabled. They do not contain per-profile PCIe GB/s measurements; `p2pmark` was
not run in this matrix.
