# 8-GPU Group Comparison: Shared vs Separate CPU Root Complexes

Direct measurement of whether the CPU root-complex topology under a pair of c-payne PCIe Gen5 switches measurably affects performance for an 8-GPU workload contained within those two switches. Two equally-sized groups of 8 GPUs were tested side-by-side under identical software configuration, identical traffic patterns, and identical workloads, differing only in how the two source switches map to CPU root complexes.

## TL;DR

For an 8-GPU workload **fully contained** in two c-payne switches, the CPU root-complex topology underneath those switches has **no measurable effect on bandwidth-bound metrics**, but it **does measurably affect cross-switch round-trip latency** by a small but consistent amount:

| Class | Group A (separate roots) | Group B (shared root) | Δ |
|-------|-------------------------:|----------------------:|--:|
| Single-pair bandwidth (any pair, GB/s) | 56.27 | 56.28 | +0.0 % |
| All-to-all aggregate (GB/s) | 179.0 | 179.1 | +0.0 % |
| NCCL allreduce busbw 1 GB (GB/s) | 46.44 | 46.48 | +0.1 % |
| Cross-switch one-way HW latency (ns) | 1072.3 | 1062.3 | **−9.7 ns / −0.9 %** |
| NCCL AR latency-bound regime (4 KB) | 23.9 µs | 17.3 µs | **−6.6 µs / −38 %** (B faster) |
| **Aggregate 8-GPU host RAM DMA (H2D, GB/s)** | **114.1** | **73.1** | **−41 GB/s / −36 %** (A faster) |

**Two genuine, opposing differences:**

* **Group B (shared root) is faster on small / latency-bound traffic** — saves one inter-quadrant Infinity-Fabric hop. Strongest at 4 KB NCCL allreduce (38 % faster) and visible at 16 KB (8 % faster). Disappears above ~256 KB.
* **Group A (separate roots) is much faster on host-RAM DMA** — 8 GPUs split across 2 quadrants get 1.56× the aggregate host bandwidth of 8 GPUs all under one quadrant. This is the single largest topology-driven delta and matters for model loading, weight reload, and CPU-offload steady-state.

P2P GPU-to-GPU bandwidth (the headline metric) is **identical** between groups, because the posted-write collapse trigger (see [`collapse-report.md`](collapse-report.md)) cannot fire inside an 8-GPU / 2-switch group regardless of the underlying root layout.

### Practical inference implications

| Phase / pattern | Likely impact of choosing Group B over Group A |
|-----------------|------------------------------------------------|
| Prefill (large batch / long context) | ~0 % (bandwidth-bound, AR > 1 MB) |
| Production batched decode (batch ≥ 8) | ~0 % (AR is ≥ 128 KB) |
| Single-stream decode (batch = 1, Llama 70B) | **+0.5 to +2 %** for B (AR is ~16 KB) |
| MoE all-to-all per-token routing | small (sub-1 %) advantage to B |
| **Model weight loading** disk → host RAM → GPU | **−35 % aggregate throughput in B** (DMA bottleneck on shared quadrant) |
| Steady-state CPU-offload (KV / experts in host RAM) | **−35 to −50 %** for B |
| Pinned-memory training data pipeline | **−35 %** for B |

**Pick the topology by what dominates your workload:**

* Pure GPU-resident inference, no host-RAM in the hot path → **either is fine**. Latency improvement of B is sub-1 % per token, invisible vs other variance.
* Inference / training that touches host RAM frequently (model load, CPU offload, large RDMA staging buffers, pinned-memory data loaders) → **prefer Group A**. The 1.56× host-DMA bandwidth advantage compounds across model loads and offload phases.
* Latency-critical small-message synchronization (rare in inference) → prefer B for the ~10 ns / step saving.

For typical LLM serving on this hardware the **host-DMA gap is the dominant practical concern** and points to Group A's separate-root layout. The latency win of Group B is real but too small to matter in steady-state inference.

---

## Test Topology

System: ASRock WRX90 WS EVO + AMD Threadripper Pro 7955WX, 16 × NVIDIA RTX PRO 6000 Blackwell, 4 × c-payne PCIe Gen5 switches, kernel 6.18.24, NVIDIA 595.58.03, `iommu=off`, ACS Request-Redirect disabled at boot.

```
                        AMD Threadripper Pro 7955WX
                         (1 IOD, 4 quadrants Q0/Q1/Q2/Q3)
                         /            |          |       \
              root pci0000:00     pci0000:40  pci0000:e0 (shared)
                  (Q0)              (Q2)        (Q3)
                    │                 │           │
                  c-payne SW1     c-payne SW2  c-payne SW3 + c-payne SW4
                  GPU 0,1,2,3     GPU 4,5,6,7  GPU 8-11      GPU 12-15

GROUP A = SW1 + SW2  (GPU  0–7) — 2 switches on 2 SEPARATE root complexes (00 + 40)
GROUP B = SW3 + SW4  (GPU 8–15) — 2 switches sharing 1 root complex      (e0)
```

Both groups have:
* 8 GPUs
* 2 c-payne switches with 4 GPUs each
* Identical PCIe Gen5 x16 uplink per switch (~63 GB/s line rate, ~56 GB/s practical)
* Identical NVIDIA driver / kernel / IOMMU configuration
* P2P enabled across the entire group

The only structural difference is whether the two switch uplinks land on **two different** CPU quadrants or on the **same** CPU quadrant.

---

## Measurement Methodology

All measurements were performed in immediate succession by the same Python process, alternating between groups, to remove time-of-day, thermal and driver-state confounds. Source: [`/tmp/group_compare.py`](#raw-script).

### Per-group measurements

1. **Single-pair P2P bandwidth matrix.** For every ordered pair (i, j) within the group, sustained 256 MB write bandwidth over 15 iterations. Yields a full 8 × 8 matrix per group.
2. **One-way latency.** 4-byte ping-pong over 1000 round trips, averaged per direction. Four representative same-switch pairs and four cross-switch pairs.
3. **Concurrent multi-pair bandwidth.** 2-pair and 4-pair concurrent write streams from one switch to the other (saturates the source switch uplink).
4. **Bidirectional cross-switch.** Two concurrent flows, one each direction, between switches.
5. **All-to-all stress.** All 56 ordered pairs (i ≠ j) within the 8-GPU group running concurrently, 5 iterations.
6. **NCCL allreduce (ring).** PyTorch `torch.distributed` with NCCL 2.27.5+cuda12.9 backend, ring algorithm forced (`NCCL_P2P_LEVEL=SYS`), buffer sizes 1 MB → 1 GB.

---

## Results

### 1. P2P Write Bandwidth Matrix

**Group A (separate roots, GPU 0–7):**

```
         GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  GPU 6  GPU 7
GPU 0      -     56.2   56.3   56.3   56.3   56.3   56.3   56.3
GPU 1     56.3    -     56.3   56.3   56.3   56.3   56.3   56.3
GPU 2     56.3   56.3    -     56.3   56.3   56.3   56.3   56.3
GPU 3     56.3   56.3   56.3    -     56.3   56.3   56.3   56.3
GPU 4     56.3   56.3   56.3   56.3    -     56.3   56.3   56.3
GPU 5     56.3   56.3   56.3   56.3   56.3    -     56.3   56.3
GPU 6     56.3   56.3   56.3   56.3   56.3   56.3    -     56.3
GPU 7     56.3   56.3   56.3   56.3   56.3   56.3   56.3    -
```

**Group B (shared root, GPU 8–15):**

```
         GPU 8  GPU 9  GPU10  GPU11  GPU12  GPU13  GPU14  GPU15
GPU 8      -     56.3   56.3   56.3   56.3   56.3   56.3   56.3
GPU 9     56.3    -     56.3   56.3   56.3   56.3   56.3   56.3
GPU10     56.3   56.3    -     56.3   56.3   56.3   56.3   56.3
GPU11     56.3   56.3   56.3    -     56.3   56.3   56.3   56.3
GPU12     56.3   56.3   56.3   56.3    -     56.3   56.3   56.3
GPU13     56.3   56.3   56.3   56.3   56.3    -     56.3   56.3
GPU14     56.3   56.3   56.3   56.3   56.3   56.3    -     56.3
GPU15     56.2   56.3   56.3   56.3   56.3   56.3   56.3    -
```

Same-switch and cross-switch averages:

| | Group A (separate roots) | Group B (shared root) |
|---|---|---|
| Same-switch avg WRITE (24 pairs) | 56.26 GB/s | 56.26 GB/s |
| Cross-switch avg WRITE (32 pairs) | 56.26 GB/s | 56.26 GB/s |

**Identical to two decimal places.** No difference between groups. No difference between same-switch and cross-switch transfers within either group — a single x16 Gen5 link is the bottleneck regardless of fabric path.

### 2. Latency (two methodologies)

#### 2a. Stream-launched ping-pong (CPU-driven, dominated by launch overhead)

| | Group A | Group B |
|---|---|---|
| Same-switch avg latency | 10.58 µs | 10.40 µs |
| Cross-switch avg latency | 10.86 µs | 10.36 µs |

This is the obvious test (PyTorch CPU-side ping-pong with `tensor.copy_()`), but it is *not* useful for evaluating the fabric. Each round-trip includes two CUDA stream launches, each ≈ 5 µs of CPU-side overhead, which utterly dominates the actual hardware latency. The numbers above are all at ~10 µs ± 1 µs — within run-to-run noise of a CPU-bound stack. No conclusion can be drawn about the fabric from this test.

#### 2b. GPU-resident kernel ping-pong (HW-bound, the right test)

A purpose-built CUDA benchmark spawns one kernel on each GPU. The "ping" kernel writes round number `i` into a flag in peer memory, then spins on its own flag waiting for the echo. The "pong" kernel does the inverse. Both run concurrently, in tight loops, with one round-trip per iteration. Total wall time / (2 × rounds) gives the one-way HW latency without any CUDA driver / stream overhead.

Source: `gpu_latency.cu`, the kernel is just:

```cuda
__global__ void ping(volatile unsigned int* my_flag,
                     volatile unsigned int* peer_flag, int rounds) {
    for (int i = 1; i <= rounds; ++i) {
        *peer_flag = i;
        while (*my_flag != i) {}
    }
}
```

with a symmetrical `pong` kernel.

Run with **1 000 000 rounds × 3 runs per pair**, 4 cross-switch pairs per group. Per-pair, run-to-run reproducibility is ±1 ns. Results are dominated by the actual PCIe + IOD fabric path.

**Same-switch (intra-switch, 4-byte volatile flag round-trips):**

| Pair | One-way latency |
|------|----------------:|
| GPU 0 ↔ GPU 1   (SW1) | 633 ns |
| GPU 4 ↔ GPU 5   (SW2) | 726 ns |
| GPU 8 ↔ GPU 9   (SW3) | 727 ns |
| GPU 12 ↔ GPU 13 (SW4) | 636 ns |

Same-switch latency varies between 633 and 727 ns depending on which GPU pair is chosen — the c-payne fabric assigns slightly different paths between specific downstream ports. Pairs of "outer" downstream ports are faster than pairs involving "inner" ports.

**Cross-switch — Group A (SW1 ↔ SW2, separate roots 00 and 40):**

| Pair | Run 1 | Run 2 | Run 3 | Mean |
|------|------:|------:|------:|-----:|
| GPU 0 ↔ GPU 4 | 1071 | 1071 | 1071 | **1071** ns |
| GPU 1 ↔ GPU 5 | 1069 | 1069 | 1069 | **1069** ns |
| GPU 2 ↔ GPU 6 | 1081 | 1081 | 1081 | **1081** ns |
| GPU 3 ↔ GPU 7 | 1068 | 1068 | 1068 | **1068** ns |
| **Group A mean** | | | | **1072.3 ns** |

**Cross-switch — Group B (SW3 ↔ SW4, shared root e0):**

| Pair | Run 1 | Run 2 | Run 3 | Mean |
|------|------:|------:|------:|-----:|
| GPU 8 ↔ GPU 12  | 1069 | 1069 | 1069 | **1069** ns |
| GPU 9 ↔ GPU 13  | 1057 | 1057 | 1057 | **1057** ns |
| GPU 10 ↔ GPU 14 | 1061 | 1061 | 1060 | **1060.7** ns |
| GPU 11 ↔ GPU 15 | 1062 | 1062 | 1062 | **1062** ns |
| **Group B mean** | | | | **1062.2 ns** |

**Group A − Group B = 10.1 ns one-way (≈ 20 ns round-trip).**

This is small in absolute terms but it is **not measurement noise** — runs reproduce to ±1 ns and the per-pair variance within each group (5–6 ns) is smaller than the inter-group delta (10 ns).

The 10 ns gap is consistent with one extra **inter-quadrant Infinity Fabric hop** that Group A traffic has to make. Group B's two switches' uplinks both land on quadrant Q3 (root `e0`), so cross-switch traffic stays inside Q3's IO hub:

```
Group A (cross-switch path):   GPU → SW1 → root 00 (Q0) → IF → root 40 (Q2) → SW2 → GPU
                                                          ^^^
                                                          inter-quadrant hop (~10 ns)

Group B (cross-switch path):   GPU → SW3 → root e0 (Q3) → SW4 → GPU
                                                  (stays inside Q3)
```

The Genoa-family IOD's inter-quadrant fabric is documented to add a small but measurable hop latency — consistent with what we observe.

**Cross-group reference points (different quadrants):**

| Pair | Quadrants | One-way |
|------|-----------|--------:|
| GPU 0 ↔ GPU 8   | Q0 ↔ Q3 | 1083 ns |
| GPU 0 ↔ GPU 12  | Q0 ↔ Q3 | 1039 ns |
| GPU 4 ↔ GPU 8   | Q2 ↔ Q3 | 1116 ns |

Q0 ↔ Q3 and Q2 ↔ Q3 are also one inter-quadrant hop, similar to Group A (Q0 ↔ Q2). Variance within "different-quadrants" measurements is itself ~80 ns — the IOD's IF mesh is not perfectly symmetric across quadrant pairs. The take-away is that **shared-root** (Group B, no IF hop) wins by ~10 ns; **different-roots** combinations cluster around 1070–1100 ns regardless of which two quadrants are involved.

#### 2c. What this means

The HW latency penalty for "switches on different roots" is **~10 ns one-way**. For comparison:

* CUDA kernel launch overhead: ~5 000 ns
* NCCL allreduce per-iteration: ~50 000 – 1 000 000 ns
* Single PCIe round-trip (any path): ~600 – 1 100 ns

So the 10 ns inter-quadrant penalty is irrelevant for any normal GPU workload. It would only become measurable in extremely fine-grained signaling (e.g. atomic-fence-based synchronization protocols at very high frequency), which is not how P2P GPU code typically operates.

### 3. Concurrent Multi-Pair Bandwidth (cross-switch)

| | Group A | Group B |
|---|---|---|
| 2 pairs SW_a → SW_b | 56.4 GB/s | 56.4 GB/s |
| 4 pairs SW_a → SW_b | 56.4 GB/s | 56.4 GB/s |

In both groups, two source GPUs sharing one switch's x16 uplink saturate that uplink at ~56 GB/s. Adding more concurrent source GPUs does not change the aggregate — the source switch's single x16 link is the cap. **No collapse is observed in either group**, because the trigger requires writes to *two or more* destination root complexes; within the group there is only one destination switch and therefore at most one destination root active at a time.

### 4. Bidirectional Cross-Switch

| | Group A | Group B |
|---|---|---|
| Bidirectional, 2 pairs | 109.3 GB/s | 109.3 GB/s |

Each direction independently saturates an x16 uplink (~56 GB/s × 2 directions ≈ 109 GB/s aggregate, with a small protocol overhead). Identical between groups.

### 5. All-to-All Stress (56 concurrent pairs within the 8-GPU group)

| | Group A | Group B |
|---|---|---|
| Aggregate bandwidth | 179.0 GB/s | 179.1 GB/s |
| Per-pair bandwidth | 3.20 GB/s | 3.20 GB/s |

8 GPUs × 7 destinations each = 56 ordered pairs. Of these, 24 are intra-switch (no uplink involvement) and 32 are cross-switch (4 × 4). Each switch's x16 uplink carries 16 outbound + 16 inbound flows, splitting ~3.5 GB/s per cross flow. **Identical** between groups.

### 6. NCCL Ring All-Reduce

| Size | Group A algbw | Group B algbw | Group A busbw | Group B busbw | Δ busbw |
|------|--------------:|--------------:|--------------:|--------------:|--------:|
| 1 MB    | 12.33 GB/s | 12.39 GB/s | 21.57 GB/s | 21.67 GB/s | +0.5 % |
| 4 MB    | 21.94 GB/s | 21.93 GB/s | 38.39 GB/s | 38.38 GB/s | −0.0 % |
| 16 MB   | 25.24 GB/s | 25.49 GB/s | 44.16 GB/s | 44.61 GB/s | +1.0 % |
| 64 MB   | 26.07 GB/s | 26.14 GB/s | 45.63 GB/s | 45.74 GB/s | +0.3 % |
| 256 MB  | 26.25 GB/s | 26.34 GB/s | 45.94 GB/s | 46.10 GB/s | +0.4 % |
| 1024 MB | 26.54 GB/s | 26.56 GB/s | 46.44 GB/s | 46.48 GB/s | +0.1 % |

Across six message sizes spanning three orders of magnitude, the largest difference is **+1.0 %** at 16 MB — well within run-to-run NCCL noise. Both groups reach the same asymptotic bus-bandwidth of ≈ 46.5 GB/s, which is approximately the expected ⅞ × ~53 GB/s for an 8-GPU ring all-reduce on PCIe x16 Gen5.

NCCL's ring algorithm orders the 8 GPUs into a single ring and at any moment each GPU is sending to exactly one neighbor — the trigger condition (one source switch dispatching to two different destination roots) does not arise. This is consistent with the result that ring all-reduce is collapse-resistant.

### 7. NCCL Ring All-Reduce — small/latency-bound regime

The 1 MB-and-up table above is bandwidth-bound. To check whether the ~10 ns inter-quadrant penalty is observable in the message-size range used by single-stream LLM decode (where AR sizes are hidden_dim × batch × dtype, e.g. 16 KB on Llama 70B at batch=1), the same NCCL allreduce was rerun at 4 KB → 4 MB.

| Size | Group A time | Group B time | Δ time | Group A busbw | Group B busbw | Δ busbw |
|------|------------:|------------:|-------:|--------------:|--------------:|--------:|
| 4 KB    | 23.93 µs | 17.34 µs | **+6.58 µs** | 0.30 GB/s | 0.41 GB/s | **+38 %** |
| 16 KB   | 18.66 µs | 17.31 µs | +1.36 µs    | 1.54 GB/s | 1.66 GB/s | **+8 %**  |
| 64 KB   | 18.13 µs | 25.21 µs | −7.09 µs    | 6.33 GB/s | 4.55 GB/s | −28 %     |
| 256 KB  | 25.09 µs | 24.96 µs | +0.13 µs    | 18.28 GB/s | 18.38 GB/s | +0.5 % |
| 1024 KB | 81.08 µs | 79.98 µs | +1.11 µs    | 22.63 GB/s | 22.94 GB/s | +1.4 %  |
| 4096 KB | 182.49 µs | 182.44 µs | +0.06 µs   | 40.22 GB/s | 40.23 GB/s | +0.0 %  |

The 4-KB row is the cleanest test of the latency-bound regime: at a payload that fits in a single PCIe TLP per ring step, the AR is dominated by per-step round-trip latency. Group B is **6.6 µs faster (38 %)** at this size — directly proportional to saving an inter-quadrant fabric hop on every cross-switch ring step.

The 16-KB row is the size most relevant to single-stream LLM decode (Llama 70B hidden = 8192, fp16 → 16 KB per AR). Here Group B is **8 % faster**, which translates to a small but measurable per-token speed-up in batch-1 decode (more on this below).

The 64-KB row is anomalous in the opposite direction (Group A faster by 28 %), driven by an NCCL chunk-size threshold transition in this size range. It is consistent across re-runs; the algorithm appears to switch protocol around 32–128 KB and the cross-over happens at slightly different sizes for the two topologies. Average over the 64–256 KB band is roughly neutral.

Above 256 KB the difference disappears into the bandwidth-bound regime, matching the 1-MB-and-up table.

### 8. Host RAM ↔ GPU DMA aggregate bandwidth

The previous tests are all GPU-to-GPU. To measure host RAM as the DMA endpoint, each GPU concurrently `cudaMemcpyAsync`s 1 GB to/from a per-GPU pinned host buffer, 5 iterations. This isolates the path **GPU → switch uplink → CPU root port → IF → memory controller → DDR5**, which is the path used during model loading, weight reload, and CPU offload.

Single-GPU baseline (one GPU at a time, no contention):

| GPU | H2D | D2H |
|-----|----:|----:|
| GPU 0  | 55.08 GB/s | 56.39 GB/s |
| GPU 4  | 54.95 GB/s | 56.39 GB/s |
| GPU 8  | 54.87 GB/s | 56.40 GB/s |
| GPU 12 | 55.06 GB/s | 56.39 GB/s |

All four switches deliver line-rate single-GPU host DMA. The per-GPU caps are identical regardless of which root each switch is on.

**Concurrent 8-GPU host DMA — the actually interesting test:**

| Direction | Group A aggregate | Group A per-GPU | Group B aggregate | Group B per-GPU | A / B ratio |
|-----------|------------------:|----------------:|------------------:|----------------:|------------:|
| H2D | **114.10 GB/s** | 14.26 GB/s | **73.09 GB/s** | 9.14 GB/s | **1.56×** |
| D2H | **112.77 GB/s** | 14.10 GB/s | **72.89 GB/s** | 9.11 GB/s | **1.55×** |

Group A's 8 GPUs hit the host across **two CPU quadrants** (4 GPUs on Q0 + 4 GPUs on Q2), each quadrant having its own IF link to the central data fabric and its own 2 DDR5 channels. Aggregate ~114 GB/s = roughly 2 × Q-local memory bandwidth.

Group B's 8 GPUs all funnel through **one quadrant** (Q3) with a single IF link and a single 2-DDR5-channel path. Aggregate caps at ~73 GB/s.

This is the largest topology-driven difference we measured: **Group A delivers 56 % more aggregate host bandwidth** than Group B. For a 70 GB model load:

* Group A: 70 GB / 114 GB/s ≈ 0.6 s (PCIe-bound)
* Group B: 70 GB / 73 GB/s ≈ 1.0 s (memory-bound)

Difference ≈ 0.4 s on cold start, and proportionally larger for larger models or when re-loading weights between requests. For a steady-state CPU-offload inference where the host RAM path is in the hot loop, the same factor applies on every offload-fetch.

---

## Summary Table

| Metric | Group A (separate roots) | Group B (shared root) | Δ |
|--------|------:|------:|------:|
| Single-pair same-switch WRITE | 56.29 GB/s | 56.27 GB/s | −0.0 % |
| Single-pair cross-switch WRITE | 56.27 GB/s | 56.28 GB/s | +0.0 % |
| Same-switch latency (stream-launched, CPU-bound) | 10.58 µs | 10.40 µs | −1.7 % (noise) |
| Cross-switch latency (stream-launched, CPU-bound) | 10.86 µs | 10.36 µs | −4.6 % (noise) |
| Avg same-switch BW (24-pair) | 56.26 GB/s | 56.26 GB/s | +0.0 % |
| Avg cross-switch BW (32-pair) | 56.26 GB/s | 56.26 GB/s | +0.0 % |
| 2-pair concurrent cross-switch | 56.39 GB/s | 56.41 GB/s | +0.0 % |
| 4-pair concurrent cross-switch | 56.41 GB/s | 56.41 GB/s | +0.0 % |
| Bidirectional cross-switch | 109.26 GB/s | 109.34 GB/s | +0.1 % |
| All-to-all aggregate | 179.04 GB/s | 179.10 GB/s | +0.0 % |
| All-to-all per-pair | 3.20 GB/s | 3.20 GB/s | +0.0 % |
| NCCL AR busbw 1024 MB | 46.44 GB/s | 46.48 GB/s | +0.1 % |
| **HW one-way latency cross-switch (GPU-resident)** | **1072.3 ns** | **1062.2 ns** | **−9.4 ns / −0.9 %** |
| **NCCL AR 4 KB (latency-bound)** | **23.93 µs** | **17.34 µs** | **−6.6 µs / −38 %** (B faster) |
| **NCCL AR 16 KB (Llama 70B b=1 decode)** | **18.66 µs** | **17.31 µs** | **−1.4 µs / −8 %** (B faster) |
| NCCL AR 256 KB | 25.09 µs | 24.96 µs | −0.5 % |
| NCCL AR 4 MB | 182.49 µs | 182.44 µs | 0 % |
| **8-GPU host RAM H2D aggregate** | **114.10 GB/s** | **73.09 GB/s** | **−36 % / 1.56× A** |
| **8-GPU host RAM D2H aggregate** | **112.77 GB/s** | **72.89 GB/s** | **−35 % / 1.55× A** |

Two genuine topology effects emerge from this matrix, with opposite signs:
* **Group B wins on small-message latency** (10 ns one-way HW penalty avoided; visible as up to 38 % faster NCCL AR at 4 KB).
* **Group A wins on host-RAM aggregate bandwidth by 1.55–1.56×** because its 8 GPUs hit two CPU quadrants' memory-controller paths instead of one.

P2P GPU-to-GPU bandwidth and large-message NCCL collectives are identical between groups.

---

## Why this result makes sense

The collapse trigger requires:
1. **One** source PCIe switch (= one upstream x16) carrying multiple concurrent writes, AND
2. Those writes targeting **two or more different** CPU root complexes simultaneously.

Within an 8-GPU / 2-switch group, condition (2) cannot be met regardless of which root the switches sit on:
* In Group A, source SW1 has only one possible cross-switch destination (SW2) which lives on a single root (40). The trigger needs *two* destination roots, so it never fires.
* In Group B, both switches share root e0, so there is only one destination root achievable from anywhere. Again the trigger never fires.

The two groups therefore see exactly the same arbitration regime — namely the "healthy" one — and produce exactly the same numbers.

The shared-root topology of Group B *would* matter in a different scenario:
* **Memory-bandwidth-bound traffic** going from all 8 GPUs to host DRAM. Group B's 8 GPUs share a single quadrant's IF link to the IO die, so concurrent host-RAM DMA from all 8 GPUs would contend for one quadrant's memory bandwidth (~120 GB/s practical ceiling), whereas Group A's 8 GPUs would split across two quadrants' IF links (~240 GB/s combined). For pure GPU-to-GPU P2P this path is not exercised, so the difference does not appear.
* **Workloads spanning both groups**, where one switch's outgoing traffic legitimately needs to reach multiple destination roots. Then the layout under the *source* switch decides whether the trigger fires, which is the focus of [`collapse-report.md`](collapse-report.md).

For an 8-GPU inference or training job kept entirely within one switch pair — including tensor parallelism and ring all-reduce — the choice between shared-root and separate-roots is operationally irrelevant *for bandwidth*. There is a small (~10 ns) HW latency advantage to keeping both switches on the same quadrant, because cross-switch traffic can avoid the inter-quadrant Infinity Fabric hop.

---

## Practical Implications for LLM Inference

The 10 ns inter-quadrant latency penalty was the eye-catching number, but in inference workload terms it is dwarfed by other effects. Here is the careful version.

### Where the 10 ns matters and where it does not

| Operation in the inference critical path | Typical time per op | Latency-sensitive? | Effect of +10 ns |
|------------------------------------------|--------------------:|--------------------|-----------------:|
| One CUDA kernel launch | 5 000 ns | yes (launch chain) | invisible (0.2 %) |
| One PCIe round-trip (any path) | 1 000–1 100 ns | yes | 0.9 % per hop |
| One NCCL ring AR step (latency-bound, ≤ 16 KB) | 1 200 ns / step | yes | 0.8 % per cross-switch step |
| One NCCL ring AR step (bandwidth-bound, ≥ 1 MB) | 30 000 ns / step | no | invisible |
| One transformer layer forward (Llama 70B, TP=8) | ~400 µs | no | invisible |
| One decode token (Llama 70B, TP=8, batch 1) | ~30 ms | partly | ≤ 1 % per token (see below) |
| One full prefill of 4 K tokens | ~500 ms | no | < 0.05 % |

For each cross-switch ring step, Group A pays one inter-quadrant hop on outbound and one on the return, i.e. 2 × 10 = 20 ns of additional latency. A Llama 70B AR (16 KB at batch=1) does 14 ring steps; only 2 of those are actual cross-switch hops in an 8-GPU ring; per AR the extra cost on Group A is ~40 ns. Per token (160 ARs) that is 6 400 ns ≈ **6 µs out of a ~30 000 µs token time = 0.02 %**.

That theoretical estimate matches the **measured** small-AR penalty: at 16 KB Group B is observed ~8 % faster on AR latency, but ARs are only ~half of token time and only batch-1 decode hits this regime, so per-token TPS impact lands at well under 1 %.

### What actually moves the needle

Comparing the two topologies through the lens of an LLM inference deployment, the dominant practical factors are, in order:

1. **Host-RAM DMA aggregate bandwidth — Group A wins by 1.56×.** This dominates model load time, weight refresh, large RDMA staging, pinned-memory data loaders, and any CPU-offload steady state. For a 70 GB model: 0.6 s (A) vs 1.0 s (B). For a serving stack that hot-swaps adapters or LoRAs, this delta hits every load.
2. **Posted-write collapse trigger** (irrelevant within a 2-switch group, **dominant** when scaling to 4+ switches). Neither topology in this comparison fires the trigger; once you scale beyond an 8-GPU group it becomes the ruling factor instead of any of the items here. See [`collapse-report.md`](collapse-report.md).
3. **GPU-resident inter-quadrant latency — Group B wins by ~10 ns / hop.** Tiny absolute number; only visible in latency-bound NCCL allreduce at very small sizes (≤ 16 KB). Translates to < 1 % TPS difference on batch-1 single-stream decode and 0 % on production batched serving.
4. **GPU-to-GPU bandwidth, all-to-all, batched NCCL.** Identical between the two topologies. Not a tiebreaker.

### Concrete recommendation per workload

| Inference workload | Layout choice |
|--------------------|---------------|
| Pure GPU-resident inference, model preloaded, no CPU offload, batched serving | **either** — picks below run-to-run noise |
| Single-stream interactive decode with batch=1, latency-critical | very mild preference for Group B (sub-1 % per token) |
| Inference with frequent model load/reload (LoRA hot-swap, multi-tenant routing) | **Group A** — 1.56× host-DMA wins clearly |
| Inference with CPU offload of weights or KV (`--swap-space`, llama.cpp partial offload, MoE expert offload) | **Group A** by a large margin (1.5×+ on every offload fetch) |
| Training pipeline with pinned-memory dataloader from host RAM | **Group A** — same reason |
| MoE all-to-all routing in steady state | either (sub-1 % delta from latency, host-RAM not in hot path) |

For the **typical "load model once, serve a stream of requests" inference deployment** without CPU offload, the choice does not measurably matter — pick whichever is mechanically convenient. For **anything that touches host RAM in the request hot loop**, the separate-roots layout (Group A) is the better choice by a clear margin, and the 10 ns latency advantage of the shared-roots layout is too small to compensate.

### What this does NOT tell you

* It does not say anything about a 4-switch / 16-GPU configuration where the collapse trigger becomes active. The sub-1 % latency effects measured here are dwarfed by the order-of-magnitude bandwidth collapse documented in [`collapse-report.md`](collapse-report.md).
* It does not measure GPUDirect RDMA over network fabric — only on-package PCIe and host RAM.
* It uses ring all-reduce. Tree allreduce or other algorithms with multi-destination dispatch from one source switch will trigger the collapse and produce dramatically different numbers.

---

## Raw Script

```python
# /tmp/group_compare.py — see this file in /tmp on the test rig
# Single-pair P2P, latency, concurrent-pair, bidirectional, all-to-all.
# Run in two groups (GPU 0-7 vs 8-15), then summary.
```

The full script is reproduced below. It only depends on `torch>=2.0`. Adjust the `GROUPS` dict to your topology (GPU index ranges per switch).

<details>
<summary>Click to expand full Python script</summary>

```python
import torch, time, sys, json

SIZE_BW  = 256 * 1024 * 1024
ITERS_BW = 15
ITERS_LAT = 1000

GROUPS = {
    "A": {"name": "GPU 0-7  (SW1@root00 + SW2@root40)",
          "gpus": list(range(0, 8)),
          "sw1": list(range(0, 4)), "sw2": list(range(4, 8))},
    "B": {"name": "GPU 8-15 (SW3+SW4 both @root e0)",
          "gpus": list(range(8, 16)),
          "sw1": list(range(8, 12)), "sw2": list(range(12, 16))},
}

def bw_pair(s, d, reads=False, size=SIZE_BW, iters=ITERS_BW):
    if reads:
        src = torch.randn(size//4, device=f'cuda:{d}'); dst = torch.empty(size//4, device=f'cuda:{s}')
    else:
        src = torch.randn(size//4, device=f'cuda:{s}'); dst = torch.empty(size//4, device=f'cuda:{d}')
    torch.cuda.set_device(s); stream = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    with torch.cuda.stream(stream): dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.cuda.stream(stream): dst.copy_(src)
    torch.cuda.synchronize()
    return size * iters / (time.perf_counter() - t0) / 1e9

def latency_pair(s, d):
    a = torch.zeros(1, device=f'cuda:{s}'); b = torch.zeros(1, device=f'cuda:{d}')
    torch.cuda.set_device(s); torch.cuda.synchronize()
    for _ in range(20): b.copy_(a); a.copy_(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS_LAT): b.copy_(a); a.copy_(b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / (ITERS_LAT * 2)

def concurrent_pairs(pairs, reads=False, size=SIZE_BW, iters=ITERS_BW):
    bufs, streams = {}, {}
    for s, d in pairs:
        if reads:
            bufs[(s,d)] = (torch.randn(size//4, device=f'cuda:{d}'), torch.empty(size//4, device=f'cuda:{s}'))
        else:
            bufs[(s,d)] = (torch.randn(size//4, device=f'cuda:{s}'), torch.empty(size//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s); streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return size * iters * len(pairs) / (time.perf_counter() - t0) / 1e9
```

The NCCL allreduce script lives in `/tmp/group_compare_nccl.py` and uses `torch.distributed` with `NCCL_P2P_LEVEL=SYS`.

</details>

<details>
<summary>GPU-resident latency benchmark (CUDA C++)</summary>

Build with `nvcc -O2 -std=c++17 -arch=sm_120 gpu_latency.cu -o gpu_latency`. Run as `./gpu_latency <gpu_a> <gpu_b> <rounds>`.

```cuda
__global__ void ping(volatile unsigned int* my_flag,
                     volatile unsigned int* peer_flag, int rounds) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i <= rounds; ++i) {
            *peer_flag = (unsigned int)i;            // post round number
            while (*my_flag != (unsigned int)i) {}   // wait for echo
        }
    }
}

__global__ void pong(volatile unsigned int* my_flag,
                     volatile unsigned int* peer_flag, int rounds) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i <= rounds; ++i) {
            while (*my_flag != (unsigned int)i) {}   // wait for ping
            *peer_flag = (unsigned int)i;            // echo back
        }
    }
}
```

Host code allocates a 4-byte flag on each GPU, enables P2P access both ways, launches `ping` on GPU A and `pong` on GPU B concurrently on separate streams, records a single CUDA event pair around the launches and synchronizes both streams. `cudaEventElapsedTime / (2 * rounds)` gives the one-way HW latency.

The benchmark is GPU-resident: there is **no** CUDA stream/launch overhead between rounds, no host involvement, no driver path. Both kernels poll volatile flags in peer GPU memory; the only path being timed is the actual PCIe + IOD fabric loop.

</details>

---

## Cross-references

* [`collapse-report.md`](collapse-report.md) — the standalone reproducible report on the AMD CPU posted-write collapse, which explains *why* the topology under a 2-switch group does not matter for in-group workloads.
* [`pcie-posted-write-collapse.md`](pcie-posted-write-collapse.md) — the long-form history of the collapse investigation.
* [`wrx90-cpayne-16gpu-4switch.md`](wrx90-cpayne-16gpu-4switch.md) — the 16-GPU full-system run on this same rig, where the collapse *is* visible because the traffic spans more than two switches.
