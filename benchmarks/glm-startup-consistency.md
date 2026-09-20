# GLM serving throughput across compilation-cache restarts

Status: **qualified** bounded serving measurements; the cause of the
startup-dependent execution difference is **research-only**.

## Conditions

GLM-5.3 Flash NVFP4, TP4/DCP1, four RTX PRO 6000 Blackwell Max-Q GPUs,
VRAM +6000, automatic graphics clocks, 325 W. Both modes use the same
[source-composition image](karmic-kraken-serving.md#source-boundaries),
temperature 1/top-p .95, a 4096-token batch budget and 32 request slots.
Decode context is zero. Each reported decode median contains five warmed
30-second windows. The 32K prefill check has **one** uncached request per
startup and is not a five-window prefill qualification.

[All windows, effective settings, four-rank trace summaries and device telemetry](glm-startup-consistency-samples.json)
retain the image/checkpoint identities and trace digests.

The first start has an empty writable compilation/tuning cache. The second
reuses that same cache after the first server exits. Model checkpoints,
memory controls and native serving arguments remain identical, apart from
the destination for a separate four-iteration Torch trace. Throughput windows
run with profiling inactive; traces follow those windows.

## Measurements

| Mode | Writable cache at startup | C1 output, tok/s | Verifier steps/s | Accepted length | 32K prefill, tok/s |
|---|---|---:|---:|---:|---:|
| MTP3 | Empty | 277.03 | 110.90 | 2.479 | 13,947 |
| MTP3 | Populated | 265.69 | 105.74 | 2.525 | 13,888 |
| DFlash2 K7 | Empty | 225.21 | 87.21 | 2.597 | 14,106 |
| DFlash2 K7 | Populated | 212.85 | 85.05 | 2.498 | 13,989 |

MTP3's populated-cache restart is **−4.10% output and −4.65% execution**,
despite higher accepted length. DFlash2 is **−5.49% output and −2.47%
execution**; its accepted length is also 3.81% lower. These are startup pairs
within one image, not source-revision A/Bs. They do not establish that storing
compilation files itself causes slower decode.

The main six-mode table retains its independent 281.26/228.95 tok/s MTP3/
DFlash2 series. The DFlash2 populated-cache median here is below the saved JJ
219.7 tok/s sample, while verifier execution remains above JJ's 84.28 steps/s.
Neither the slower series nor its acceptance difference is discarded.

## What the profiles establish

- Kernel names, launch geometry and invocation counts match on every rank:
  145 distinct entries for MTP3 and 150 for DFlash2. MTP3's 36 recorded B12X
  preparation selections also match. No observed backend or grid change
  explains the difference.
- In the populated-cache traces, rank 3 has a median signed same-stream
  graph separation around 0.449 μs, versus about 0.128 μs on all ranks in the
  empty-cache traces. Programmatic dependent launches can overlap, producing
  negative separations; summing these values does not measure total GPU idle time.
- MTP3's collective-arrival skew grows from about 2.2 to 6.8 μs, while the
  completion tail after the final rank arrives stays around 7.5 μs. DFlash2
  shows about 2.3 versus 6.6–7.0 μs arrival skew, with an approximately
  11-μs completion tail. A longer observed collective can include waiting for
  another rank; it is not by itself proof of a slower communication kernel.
- Cell-matched device telemetry retains P1 and 16,365-MHz memory clocks.
  Corresponding graphics-clock medians differ by at most approximately 8 MHz.
  UTC benchmark events are joined to timezone-aware GPU telemetry.

The evidence retains all captured steps, including profiler-start skew.
Node-level profiling can perturb launch timing, so these traces locate a
candidate path rather than replacing the unprofiled throughput measurements.

## Deployment decision

Do not clear a user's compilation cache as a speed recommendation. Cache
deletion adds startup compilation and has not been established as a reliable
runtime fix. No queue-capacity or memory-control default is changed by this
startup observation.

## CUDA connection-capacity diagnostic

Two additional MTP3 starts use copies of the same populated cache. The first
requests `CUDA_DEVICE_MAX_CONNECTIONS=128`; the second leaves the setting
unset. Each has five warmed C1 windows, one uncached 32K request, and a separate
four-rank trace. The launch requests sixteen NCCL channels in both arms.
CUDA hardware connections and NCCL channels are different controls.

| CUDA connection setting | C1 output, tok/s | Verifier steps/s | Accepted length | 32K prefill, tok/s |
|---|---:|---:|---:|---:|
| Unset | 265.55 | 106.45 | 2.510 | 13,845 |
| 128 | 264.67 | 105.99 | 2.498 | 13,859 |

The explicit 128 setting changes output by **−0.33%** and verifier rate by
**−0.44%**, with unchanged logical KV capacity. It does not recover the
110.90-step/s empty-cache result. Keep the CUDA connection default; this
diagnostic supplies no measured benefit and does not resolve the startup cause.

NVIDIA documents graph-internal stream expansion and potential hardware-channel
serialization in its
[CUDA graph troubleshooting guide](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html).
That motivated the diagnostic, not a presumption that serialization caused the
observed difference.

### Environment observation limit

The launch receipt and API-process snapshot retain the requested NCCL minimum,
maximum and buffer size. All four worker snapshots retain the expected CUDA
connection value and NCCL maximum. They do not retain the NCCL minimum or
buffer-size entry. These are `/proc/<pid>/environ` observations, not live
`getenv` calls: Linux exposes initial environment memory, which subsequent
environment changes do not necessarily update. See
[the Linux interface contract](https://www.man7.org/linux/man-pages/man5/proc_pid_environ.5.html).

A CPU-only reproducer in the same image confirms another relevant limitation:
after `setproctitle("VLLM::Worker_TP0")`, those entries disappear from `/proc`,
while Python's environment and C `getenv` retain all four requested values.
The linked samples include the before/after observations. Missing entries are
therefore recorded as unknown; they are not evidence that NCCL lost its settings.
The two serving arms have matching observations, and present conflicting values
are rejected by the comparison check. No live-worker `getenv` claim is made.
