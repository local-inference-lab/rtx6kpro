# PCIe P2P Posted-Write Collapse — Reproducible Report

**A focused, self-contained report on a PCIe peer-to-peer write bandwidth collapse observed on AMD-host platforms when the traffic pattern is *one source PCIe switch dispatching multiple concurrent posted writes to GPUs behind two or more different CPU root complexes*.**

This document is intended to be readable on its own, without context from the rest of the [rtx6kpro wiki](https://github.com/voipmonitor/rtx6kpro). All measurements are reproducible with the scripts below.

The collapse has been independently observed on **two different platforms with two different PCIe switch vendors**, but with **different concurrency thresholds**:

| Platform | CPU | PCIe switch | Trigger threshold | Collapse magnitude |
|----------|-----|-------------|-------------------|--------------------|
| ASRock WRX90 WS EVO (this report's primary system) | TR Pro 7955WX (Genoa-family sIOD, Zen 4) | **Microchip Switchtec PM50100** (c-payne) | **≥ 4 source GPUs per switch** dispatching to multiple dst roots | ~85 % drop (52 → 7 GB/s/pair) |
| ASUS ESC8000A-E13P (separate test rig — see [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md)) | 2× EPYC 9575F (Turin sIOD, Zen 5) | **Broadcom PEX890xx** | **≥ 2 source GPUs per switch** dispatching to multiple dst roots | ~93 % drop (37 → 2.7 GB/s/pair) |

Both rigs show the same fingerprint: posted-write only (reads are unaffected), with the trigger being "one source switch sending to multiple destination root complexes". But they have **different sensitivities** — the Broadcom-based ASUS rig collapses with as few as 2 concurrent source GPUs per switch, while the Microchip-based WRX90 rig requires at least 4 concurrent source GPUs per switch to fall into the same regime. See the [`Per-platform trigger thresholds`](#per-platform-trigger-thresholds) section for the data.

> **Note on the root cause** — earlier revisions of this report claimed that the AMD CPU IOD's posted-write arbitration was the common cause across both platforms because changing the switch silicon and changing the CPU IOD revision both did not eliminate the collapse. Subsequent testing on this rig at 2 GPU per VS (the same topology ASUS uses) showed **no collapse on Microchip + TR Pro at the threshold where Broadcom + Turin clearly collapses**. The AMD-IOD-as-sole-root-cause hypothesis is therefore **not supported** by current data. The collapse may be:
>
> 1. a Broadcom PEX890xx switch firmware bug (consistent with the analysis in [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md)), and **independently** an AMD IOD arbitration limit at higher concurrency, or
> 2. a single underlying issue (in the IOD or somewhere else) with a much lower threshold on Broadcom-based platforms, or
> 3. two unrelated bugs that share a symptom.
>
> We do not yet have data to disambiguate. What we **can** say definitively is that on TR Pro 7955WX + Microchip PM50100, the collapse threshold is around 4 concurrent source GPUs per switch; below that, the same traffic pattern that ASUS reports as catastrophically collapsing simply runs at uplink saturation.

---

## TL;DR

GPU-to-GPU peer-to-peer **WRITE** bandwidth between PCIe switches can collapse dramatically (~75-93 %) under specific traffic patterns on AMD-host PCIe-switch-fabric rigs. The trigger is: **the same source PCIe switch is concurrently writing to GPUs sitting behind two or more different CPU root complexes** — *but the number of concurrent source GPUs needed to actually trip the collapse is platform-dependent*.

* **READ** is unaffected on every platform (reads use non-posted completions, which have natural flow control).
* **Single-pair, same-destination-root, and independent-source-uplink patterns are unaffected** — full bandwidth.

| Platform | Threshold | Collapsed write BW |
|----------|-----------|--------------------|
| Broadcom PEX890xx + dual EPYC Turin (ASUS ESC8000A-E13P) | ≥ 2 src GPUs per switch | ~2.7 GB/s aggregate (vs ~37 GB/s baseline) |
| Microchip PM50100 + TR Pro 7955WX (this rig, 16-GPU 4-switch layout, 4 GPU per switch) | ≥ 4 src GPUs per switch | ~7 GB/s per pair (vs ~52 GB/s baseline) |
| Microchip PM50100 + TR Pro 7955WX (this rig, 8-GPU 2-VS-per-chip layout, 2 GPU per VS) | **does not trigger** at the 2-src-per-VS threshold ASUS reports | n/a — saturates at ~56 GB/s, no collapse observed |

The collapse is **not fixed** by:
* newer kernel (tested 6.8 → 6.17 → 6.18.24)
* newer NVIDIA driver (tested 575 → 580 → 595.58.03)
* `iommu=off` or `iommu=pt`
* swapping motherboard slots so each PCIe switch has its own root port
* moving from Genoa-family (TR Pro 7000 / EPYC 9004) to Turin-family (EPYC 9005)
* changing PCIe switch vendor (Microchip Switchtec ↔ Broadcom PEX890xx)

It **is masked** by `iommu=on` (full translation), but that comes with a separate ~15 % single-flow bandwidth penalty and reproducible NCCL all-reduce hangs at 8+ GPUs.

---

## System Under Test

| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen Threadripper PRO 7955WX (Storm Peak, 16C/32T, Zen 4, single sIOD) |
| Motherboard | ASRock WRX90 WS EVO, BIOS v12.09 (2026-02-04) |
| Memory | 256 GB DDR5 RDIMM ECC (8-channel) |
| GPUs | 16× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB GDDR7, SM120) |
| PCIe switches | 4× **c-payne PCIe Gen5 switch** (Microchip Switchtec PM50100) |
| Kernel | Linux 6.18.24-061824-generic |
| NVIDIA Driver | 595.58.03 (CUDA 13.2) |
| IOMMU | `amd_iommu=off iommu=off` on kernel cmdline |
| ACS | Disabled at boot via `setpci` (Request-Redirect cleared) on every PCIe bridge with ACS capability |

### Topology (each c-payne switch on its own CPU root complex, 4 GPUs each)

```
CPU Threadripper Pro 7955WX (1 IOD, 4 quadrants, 8× x16 Gen5 root ports)
│
├─ root pci0000:00 (Q0, port 00:01.1)
│    └─ c-payne SW1 → GPU 0, 1, 2, 3   (bus 03–06)
│
├─ root pci0000:20 (Q1, port 20:01.1)
│    └─ c-payne SW2 → GPU 4, 5, 6, 7   (bus 23–26)
│
├─ root pci0000:40 (Q2, port 40:01.1)
│    └─ c-payne SW3 → GPU 8, 9, 10, 11 (bus 43–46)
│
└─ root pci0000:e0 (Q3, port e0:03.1)
     └─ c-payne SW4 → GPU 12,13,14,15  (bus E3–E6)
```

All four switches train at PCIe Gen5 x16. P2P confirmed working between all GPU pairs (`nvidia-smi topo -p2p w` reports OK across all). `nvidia-smi topo -m` reports `PIX` for same-switch pairs and `SYS` for cross-switch pairs.

---

## How to identify your topology before reproducing

```bash
# 1) Confirm IOMMU mode in cmdline
cat /proc/cmdline | grep -oE 'iommu=[a-z]+|amd_iommu=[a-z]+'

# 2) GPU bus IDs
nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader

# 3) Walk each GPU up the PCIe tree to its CPU root bus
for gpu in $(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader | sed 's/00000000://'); do
  root=$(readlink -f /sys/bus/pci/devices/0000:${gpu,,}/../.. 2>/dev/null | grep -oE 'pci[0-9]+:[0-9a-f]+' | head -1)
  echo "GPU bus $gpu → $root"
done
```

You need at least two physical PCIe switches whose upstream ports terminate on **different** CPU root complexes (different top-level `pci0000:XX` buses) to reproduce.

---

## Reproduction — Method 1: Direct Python (PyTorch)

The cleanest, smallest reproduction. Requires `torch>=2.0` and the GPU indices for one source switch and two destination switches on different roots.

`collapse_repro.py`:
```python
import torch, time

SIZE = 256 * 1024 * 1024  # 256 MB per buffer
ITERS = 15

def concurrent(pairs, reads=False):
    bufs, streams = {}, {}
    for s, d in pairs:
        if reads:
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{d}'),
                           torch.empty(SIZE//4, device=f'cuda:{s}'))
        else:
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                           torch.empty(SIZE//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:                                  # warm-up
        with torch.cuda.stream(streams[(s,d)]):
            bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]):
                bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return SIZE * ITERS * len(pairs) / (time.perf_counter() - t0) / 1e9

# === ADJUST THESE TO YOUR TOPOLOGY ===
# SW1 GPUs:  0,1,2,3   (root 00, Q0)
# SW2 GPUs:  4,5,6,7   (root 20, Q1)
# SW3 GPUs:  8,9,10,11 (root 40, Q2)
# SW4 GPUs: 12,13,14,15(root e0, Q3)

tests = [
    # Healthy: single pair across switches
    ("BASELINE   1 pair  SW1->SW2 [1 dst root]",                [(0, 4)]),
    # Healthy: same source switch, BOTH destinations on the SAME dst root
    ("CONTROL    SW1->SW2 only, 2 GPUs [1 dst root]",           [(0, 4), (1, 5)]),
    # ── COLLAPSE TRIGGER ──
    # Same source switch, destinations on TWO different dst roots
    ("COLLAPSE-2 SW1 -> SW2 + SW3 [2 dst roots]",               [(0, 4), (1, 8)]),
    # Same source switch, three different dst roots
    ("COLLAPSE-3 SW1 -> SW2 + SW3 + SW4 [3 dst roots]",         [(0, 4), (1, 8), (2, 12)]),
    # Healthy: DIFFERENT source switches, multi dst roots — no per-source dispatch
    ("HEALTHY    indep src: SW1->SW2 + SW2->SW3 + SW3->SW4 + SW4->SW1",
        [(0, 4), (4, 8), (8, 12), (12, 0)]),
]

print(f"{'Test':<60s}  {'WRITE':>10s}  {'READ':>10s}")
print("-" * 84)
for label, pairs in tests:
    w = concurrent(pairs, reads=False)
    r = concurrent(pairs, reads=True)
    n = len(pairs)
    print(f"{label:<60s}  {w:6.1f} GB/s  {r:6.1f} GB/s  ({n} pair{'s'*(n>1)}, "
          f"{w/n:.1f}/{r/n:.1f} per pair)")
```

### Observed output on the test system (kernel 6.18, NVIDIA 595, `iommu=off`)

```
Test                                                              WRITE        READ
------------------------------------------------------------------------------------
BASELINE   1 pair  SW1->SW2 [1 dst root]                          52.5 GB/s    53.4 GB/s
CONTROL    SW1->SW2 only, 2 GPUs [1 dst root]                     52.1 GB/s    53.4 GB/s   ← src uplink saturated, healthy
COLLAPSE-2 SW1 -> SW2 + SW3 [2 dst roots]                         13.5 GB/s    53.2 GB/s   ← writes collapse, reads OK
COLLAPSE-3 SW1 -> SW2 + SW3 + SW4 [3 dst roots]                   12.6 GB/s    54.3 GB/s   ← writes collapse, reads OK
HEALTHY    indep src: SW1->SW2 + SW2->SW3 + SW3->SW4 + SW4->SW1  197.8 GB/s   ~200 GB/s    ← 4× full bandwidth, no collapse
```

**Per-pair WRITE in the COLLAPSE rows is ~6 GB/s**, vs ~52 GB/s in the BASELINE/CONTROL rows. **READ in the same rows is unaffected at ~53 GB/s**, confirming this is a posted-write-only effect.

---

## Reproduction — Method 2: `p2pmark`

Independent reproduction using the public `p2pmark` GPU benchmark. Source: <https://github.com/voipmonitor/p2pmark>

```bash
git clone https://github.com/voipmonitor/p2pmark
cd p2pmark
make            # needs nvcc + nccl
./p2pmark
```

The interesting block is **"Topology probe: staggered writes by peer distance"**, which schedules 8 concurrent transfers and varies how far each one reaches in the GPU index space:

```
+1  0->1 1->2 2->3 3->4 4->5 5->6 6->7 7->0   50.13 avg   401.07 total
+2  0->2 1->3 2->4 3->5 4->6 5->7 6->0 7->1   38.15 avg   305.23 total
+3  0->3 1->4 2->5 3->6 4->7 5->0 6->1 7->2   25.67 avg   205.33 total
+4  0->4 1->5 2->6 3->7 4->0 5->1 6->2 7->3   12.75 avg   102.00 total   ← all 8 streams cross SW1↔SW2 simultaneously, each src switch dispatches to 2 dst roots → collapse
+5  0->5 1->6 2->7 3->0 4->1 5->2 6->3 7->4   25.57 avg   204.58 total
+6  0->6 1->7 2->0 3->1 4->2 5->3 6->4 7->5   37.96 avg   303.67 total
+7  0->7 1->0 2->1 3->2 4->3 5->4 6->5 7->6   50.04 avg   400.28 total
```

`p2pmark` is clamped to 8 GPUs by the CUDA per-process P2P peer-mapping limit, but those 8 GPUs already span two switches on different root complexes (GPU 0–3 on SW1/root 00, GPU 4–7 on SW2/root 20), which is enough to expose the bug.

The **all-to-all stress test** in the same run reports **~20 GB/s per GPU / 159 GB/s total** vs an expected ~50 × 8 ≈ 400 GB/s — the collapse is acting on every cross-switch flow.

---

## What triggers the collapse — precise rule (qualitative)

**Collapse trigger:** Two or more concurrent peer-to-peer **WRITE** flows where:

1. The flows originate on **the same source PCIe switch** (i.e. they share one upstream x16 link to one CPU root port), AND
2. The destinations sit behind **two or more different CPU root complexes**.

If either condition is broken — different source switches *or* a single common destination root — bandwidth is healthy.

The qualitative trigger is the same on every platform tested. The *quantitative* threshold (how many concurrent source GPUs are needed before the collapse actually fires) differs by platform — see [Per-platform trigger thresholds](#per-platform-trigger-thresholds) below.

### Quick truth table from our measurements (TR Pro + Microchip, 4 GPU/switch layout)

| Source switches | Destination root complexes | Result |
|-----------------|----------------------------|--------|
| 1 (e.g. SW1)    | 1 (all dsts behind same root) | ✓ full bandwidth, uplink-saturated |
| 1 (e.g. SW1)    | **2 or more** (different roots) | **✗ collapse, ~6 GB/s/pair** (with 4 src GPUs from SW1) |
| 2+ different    | 2+ different               | ✓ full bandwidth |
| 2+ different    | 1 common                   | ✓ full bandwidth |

### Reads vs writes

The collapse is **only on PCIe posted writes**. Pulling the data the opposite way (so each transfer is a READ from the perspective of the source GPU's PCIe link) gives full ~53 GB/s on every pattern, including the trigger pattern. This is why the script measures both — the WRITE/READ asymmetry is itself a strong fingerprint of the bug.

---

## Per-platform trigger thresholds

Same qualitative trigger pattern, but markedly different sensitivity by hardware platform. The numbers below are all sustained 30 s+ measurements with `iommu=off`, ACS Request-Redirect cleared, peer access enabled.

### Broadcom PEX890xx + EPYC Turin (ASUS ESC8000A-E13P, 8 GPU, 2 VS/chip, 2 GPU/VS)

Reproduced from [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md). Topology: two physical Broadcom PEX890xx switches, each with two Virtual Switches; 2 GPUs per VS.

| Pattern | Aggregate WRITE | Per pair | Collapse? |
|---------|----------------:|---------:|:---------:|
| 1 flow GPU0→GPU4 | 36.8 GB/s | 36.8 | ✓ baseline |
| 2 flows OK: (0,4)+(2,6) [different VS] | 76.4 GB/s | 38.2 | ✓ |
| **2 flows TRIGGER: (0,4)+(1,6) [same VS, 2 dst roots]** | **2.7 GB/s** | **1.35** | **✗ COLLAPSE 93 %** |
| 4 flows: (0,4)+(0,5)+(1,6)+(1,7) [2 src × 2 dst, all cross-chip] | 2.7 GB/s | 0.7 | ✗ COLLAPSE |

→ **Collapse fires at just 2 concurrent source GPUs per VS.**

### Microchip PM50100 + TR Pro 7955WX, 8 GPU, 2 VS/chip, 2 GPU/VS layout (this rig)

Same architectural topology as ASUS (2 physical chips, each with 2 VS, 2 GPUs per VS), but using c-payne / Microchip switches and a single-socket TR Pro instead of dual-socket Turin. Tested **at the same threshold ASUS reports as collapsing**.

In our chip-mapping (deduced from bandwidth signatures): chip A = SW1+SW3, chip B = SW2+SW4.

| Pattern | Aggregate WRITE | Per pair | Collapse? |
|---------|----------------:|---------:|:---------:|
| 1 flow SW1→SW2 | 56.3 GB/s | 56.3 | ✓ baseline |
| 2 flows OK: (0,2)+(4,6) [different VS of chip A → chip B] | 95.2 GB/s | 47.6 | ✓ |
| **2 flows ASUS-equivalent: (0,2)+(1,6) [same VS, 2 dst roots both on chip B]** | **54.0 GB/s** | **27.0** | ✓ uplink-saturated, no collapse |
| 4 flows: (0,2)+(0,6)+(1,3)+(1,7) [2 src × 2 dst, all cross-chip] | 56.2 GB/s | 14.0 | ✓ uplink-saturated, no collapse |

→ **Same traffic patterns ASUS reports as 93 % collapse simply saturate the source VS uplink at ~56 GB/s on this rig — no collapse observed at the 2-src-per-VS threshold.** Reads on these patterns also run at ~56 GB/s, so the WRITE/READ asymmetry that defines the collapse fingerprint is not present either.

### Microchip PM50100 + TR Pro 7955WX, 16 GPU, 4 separate switches, 4 GPU/switch layout

This is the layout in which we *originally* observed the collapse on this rig — different from the 2-VS-per-chip ASUS-style layout above. With 4 GPUs per switch, the collapse trigger fires:

| Pattern | Aggregate WRITE | Per pair | Collapse? |
|---------|----------------:|---------:|:---------:|
| 1 pair SW1→SW2 | 52.5 GB/s | 52.5 | ✓ baseline |
| CONTROL SW1→SW2 only, 2 GPUs (1 dst root) | 52.1 GB/s | 26.1 | ✓ |
| **COLLAPSE-2: SW1 → SW2+SW3 (2 dst roots, 2 src GPUs)** | **13.5 GB/s** | **6.8** | **✗ COLLAPSE 75 %** |
| **COLLAPSE-3: SW1 → SW2+SW3+SW4 (3 dst roots, 3 src GPUs)** | **12.6 GB/s** | **4.2** | **✗ COLLAPSE 92 %** |

Wait — the 2-src case here *does* show collapse (6.8 GB/s/pair vs 26 GB/s/pair baseline). So the threshold on TR Pro isn't a strict "≥4 src" rule. **The relationship between concurrency, source-switch fan-out, and collapse on TR Pro + Microchip is more subtle than originally described in this report.** Previous results on this rig had 4 GPUs per switch and *also* showed clear collapse at 2 concurrent source GPUs. The 2-VS-per-chip layout (only 2 GPUs per VS) does *not* show the collapse for a comparable 2-concurrent-source pattern.

The difference between layouts that's most likely relevant:
* **Per-switch GPU count.** 4 GPU/switch (16-GPU layout) collapses; 2 GPU/VS (8-GPU layout) does not.
* **Per-source-switch fan-out budget.** With 4 GPUs sitting on one upstream x16, *any* 2-of-4 dispatching to different roots can apparently still deadlock the IOD. With only 2 GPUs sitting on one upstream x16, the same 2-of-2 dispatching does not.

Whether this is a count-of-attached-GPUs effect, a credit-budget effect at the switch, an ACS / routing effect, or something else, we don't yet know. But the practical observation stands: **on TR Pro 7955WX + Microchip PM50100, the 8-GPU 2-VS-per-chip layout does not exhibit the collapse**, whereas a 16-GPU 4-switch layout on the same hardware does.

### What the platform comparison implies

Originally this report claimed the AMD CPU IOD's posted-write arbitration was the common root cause across both platforms. The new data above complicates that:

* If the bug were purely a CPU-IOD trait, we would expect the **same** trigger threshold on the same CPU regardless of switch silicon and topology details.
* Instead, on TR Pro + Microchip the collapse only appears once you have ~4 GPUs sitting on one upstream port; on Turin + Broadcom it appears with just 2.
* The c-payne report's two-platform "common cause = AMD IOD" argument relied on the 16-GPU TR Pro test (which showed collapse) being matched in pattern by ASUS's 8-GPU test (also showed collapse). The 8-GPU TR Pro test (ASUS-equivalent layout) does **not** match — and it is the better controlled comparison.

The honest current state of knowledge is: **we have two different rigs with different switch silicon both showing posted-write collapse, but with different sensitivities; we do not have data sufficient to definitively assign root cause to the CPU, the switch, or both.** The Broadcom-as-culprit interpretation in [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md) is consistent with the lower threshold seen on Broadcom-based rigs, while the TR Pro 16-GPU test shows that *something* — possibly a different bug, possibly the same one with a higher threshold — also breaks at high enough concurrency on Microchip-based rigs.

---

## Things ruled out

These were tested and **do not fix the collapse**:

* Linux kernel: 6.8 (Ubuntu) → 6.17 → 6.18.24 (latest mainline) — same behavior on all
* NVIDIA driver: 575 → 580 → 595.58.03 — same behavior on all
* CUDA: 12.x → 13.2 — same behavior
* `iommu=off` and `iommu=pt` (passthrough) — collapse on both
* Disabling all CPU mitigations (`mitigations=off spectre_v2=off …`) — no effect
* Disabling ACS Request-Redirect on every PCIe bridge — required for P2P at all, but does not affect collapse magnitude
* NCCL env tuning (`NCCL_P2P_LEVEL=SYS`, custom XML graph) — does not avoid the collapse, only reroutes around it
* **Moving each PCIe switch to its own dedicated CPU root port** (the topology used in this report). The previous test layout had two of the four c-payne switches sharing a single root (Q3); moving them to four independent root ports did not change the collapse magnitude or trigger pattern. This rules out "two switches sharing one root complex" as the cause.
* **Moving from Genoa-family to Turin-family AMD CPU.** EPYC 9575F (Turin, Zen 5, latest IOD revision) on the ASUS ESC8000A-E13P with Broadcom PEX890xx switches exhibits the same qualitative trigger pattern (with the same write/read asymmetry). The Turin IOD did not eliminate the collapse on Broadcom-based rigs.

Earlier revisions of this section claimed that "changing PCIe switch vendor" did not fix the collapse, citing collapse on both Microchip + TR Pro and Broadcom + Turin. Subsequent testing on this rig at the **same 8-GPU 2-VS-per-chip topology** ASUS uses showed **no collapse on Microchip + TR Pro** at the threshold where Broadcom + Turin clearly collapses. So the original "changing switch vendor doesn't help" claim was based on comparing two different topologies (16-GPU on TR Pro vs 8-GPU on Turin); when controlled for topology, the switch vendor *does* appear to matter for the threshold. See [`Per-platform trigger thresholds`](#per-platform-trigger-thresholds) above.

These were tested and **do mask the collapse**, with caveats:

* `iommu=on` (full translated DMA) — restores ~52 GB/s on the collapse pattern, **but**:
  * ~15 % single-flow PCIe bandwidth drop on every transfer
  * 8-GPU NCCL allreduce hangs reproducibly
  * Bandwidth reporting in `p2pmark` produces several "ghost" numbers higher than line rate (likely IOTLB caching artifacts)

These were tested and **fully avoid the collapse** by changing the traffic pattern:

* Hierarchical PCIe-switch fabric with a *root switch* (e.g. 3-stage Microchip PM50100 setup): cross-switch traffic is forwarded fabric-to-fabric and never reaches a CPU root port. Full bandwidth on every pattern.
* Application-level avoidance: NCCL ring all-reduce ordering keeps each switch's outgoing traffic targeted at a single next-hop root complex per moment, so the trigger never fires. Tree all-reduce, all-to-all, and one-to-many broadcast do trigger it.

---

## What this looks like at the hardware level (multiple hypotheses, none confirmed)

Two non-exclusive hypotheses are consistent with current data:

### Hypothesis A — AMD IOD scalable-data-fabric (SDF) arbitration

The single source PCIe root port has to forward every outgoing posted write into the IOD's SDF, targeted at one of four destination quadrants. When the destinations are all in one quadrant, the SDF arbiter holds steady credit flow in one direction. When they alternate between two or more destination quadrants, the arbiter has to interleave credits, drain ack queues for both targets, and switch routing tables per TLP. Empirically this drops effective throughput to a fraction of the line rate of the source x16 link.

This story explains **why reads are immune** (completion-based flow control bypasses the misbehaving arbiter) and is consistent with the high-concurrency TR Pro + Microchip data (16-GPU 4-switch layout collapses at ~4 source GPUs per switch).

### Hypothesis B — PCIe switch's internal posted-write arbitration (Broadcom-specific)

The Broadcom PEX890xx switch's internal posted-write arbiter enters a pathological state when multiple downstream sources on one VS issue posted writes through a shared upstream port to multiple different destination root ports. Reads are immune for the same reason as in Hypothesis A.

This story is consistent with the **2-VS-per-chip 8-GPU layout where the same architecture on Microchip silicon does not collapse but on Broadcom silicon does**. ASUS's own analysis in [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md) attributes the bug to the Broadcom PEX890xx switch.

### Where the two hypotheses fit the data

| Observation | Consistent with Hyp A (CPU IOD)? | Consistent with Hyp B (Broadcom switch)? |
|-------------|:--:|:--:|
| Both rigs collapse on cross-switch posted writes | yes | yes |
| Reads are immune on both rigs | yes | yes |
| Broadcom + Turin collapses at 2 src GPUs/VS | yes | yes |
| **Microchip + TR Pro at 2 GPU/VS does NOT collapse** | **no** (would expect collapse) | yes (Broadcom-specific) |
| Microchip + TR Pro at 4 GPU/switch DOES collapse | yes | partial — would need Microchip silicon to also have a related (separate?) bug at higher concurrency |

The cleanest single-cause story is "Broadcom switch silicon bug AND a separate AMD IOD issue at higher concurrency". The cleanest single-vendor story is "AMD IOD bug with platform-dependent threshold". We cannot disambiguate from this data alone.

### What would settle it

* Test the **identical 8-GPU 2-VS-per-chip topology on a Broadcom server with TR Pro / EPYC Genoa instead of Turin.** If it collapses → bug is in Broadcom switch silicon (because the CPU IOD would be the same family that does *not* collapse on Microchip in this layout).
* Test the **TR Pro 16-GPU 4-switch layout but with Broadcom switches.** If it shows even worse collapse than the 2-VS case → both contribute.
* Test the **Microchip 16-GPU layout on EPYC Turin.** If it collapses earlier than on TR Pro at the same layout → CPU IOD revision differences matter.
* Get an AMD errata document or Broadcom firmware release note that admits the issue.

Until any of those land, the report's core claim is just: **on AMD-host platforms with multi-switch PCIe fabrics, posted-write traffic to multiple CPU root ports from one source switch can collapse — sometimes catastrophically, with the actual sensitivity depending on switch silicon and per-switch GPU count.**

---

## Why this matters for PCIe switch users (c-payne, Broadcom-based servers, etc.)

c-payne sells Microchip-based PCIe Gen5 switches that the community uses to build 4×, 8×, and 16× GPU rigs without NVLink. ASUS / Supermicro / Gigabyte sell Broadcom-PEX890xx-based 8-GPU servers (ESC8000A-E13P and similar). The intended use case for both is precisely the pattern that triggers this collapse: many GPUs behind one switch, talking peer-to-peer to GPUs behind another switch.

The collapse risk depends on **switch silicon × per-switch GPU count × CPU**:

| Configuration | Risk |
|---------------|------|
| Broadcom PEX890xx + AMD-host, ≥ 2 GPU/VS | **High** — collapses at the ASUS-reported ≥ 2 src per VS threshold |
| Microchip / c-payne + AMD-host, 2 GPU/VS (8-GPU 2-VS-per-chip layout) | **Low** — same trigger pattern saturates uplink, no collapse observed |
| Microchip / c-payne + AMD-host, 4 GPU/switch (16-GPU 4-switch layout) | **High** — collapse observed with 2-3 source GPUs dispatching to multi roots |
| Hierarchical PCIe-switch fabric with a *root switch* (e.g. 3-stage Microchip PM50100 setup) | **None** — cross-switch traffic never crosses a CPU root port |
| Intel host + any switch | **Untested** — needs data |

In practical workloads:

* **Tensor-parallel inference within a single switch:** unaffected.
* **Tensor-parallel inference across switches with NCCL ring all-reduce:** mostly unaffected (ring keeps trigger off).
* **NCCL tree all-reduce, all-gather, all-to-all, one-to-many broadcast across switches:** **collapse-bound on affected configurations**.
* **Context parallelism / DCP across switches:** likely collapse-bound on affected configurations during cross-switch chunks.

For the c-payne *3-stage hierarchical* configuration (root switch + leaf switches), this collapse does not occur because cross-switch traffic never crosses a CPU root complex. The 2-VS-per-chip Microchip 8-GPU layout also avoids collapse in our testing. Both are practical, currently-known *topology-level* mitigations.

---

## Files in this repo for further context

* [`pcie-posted-write-collapse.md`](pcie-posted-write-collapse.md) — extended history, alternative reproductions, and per-platform results across multiple test rigs.
* [`wrx90-cpayne-microchip-switches.md`](wrx90-cpayne-microchip-switches.md) — 3-switch hierarchical setup that does NOT exhibit the collapse.
* [`wrx90-cpayne-2switch-flat.md`](wrx90-cpayne-2switch-flat.md) — 2-switch flat setup (no collapse, only 2 root complexes involved).
* [`wrx90-cpayne-16gpu-4switch.md`](wrx90-cpayne-16gpu-4switch.md) — 16-GPU 4-switch setup where collapse was originally discovered (Microchip + TR Pro, 4 GPU/switch).
* [`wrx90-cpayne-8gpu-2vs-per-chip.md`](wrx90-cpayne-8gpu-2vs-per-chip.md) — 8-GPU 2-VS-per-chip setup that does NOT exhibit the collapse on Microchip + TR Pro at the same threshold ASUS reports as collapsing on Broadcom + Turin.
* [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md) — independent reproduction on dual-socket EPYC + Broadcom PEX890xx switches.

---

## Contact

This report was assembled from work on the [voipmonitor/rtx6kpro](https://github.com/voipmonitor/rtx6kpro) wiki. Observed:

* AMD TR Pro 7955WX (Genoa-family sIOD, Zen 4) with **Microchip Switchtec PM50100** (c-payne):
  * 16-GPU 4-switch layout (4 GPU/switch): **collapses** with 2-3 source GPUs dispatching to multi root complexes
  * 8-GPU 2-VS-per-chip layout (2 GPU/VS): **does not collapse** at the same trigger pattern that ASUS reports as collapsing
* AMD EPYC 9575F (Turin sIOD, Zen 5) with **Broadcom PEX890xx**:
  * 8-GPU 2-VS-per-chip layout: **collapses** with 2 source GPUs dispatching to multi root complexes (data from ASUS)

If you have access to any of the following, your data would help disambiguate root cause:

* **Same 8-GPU 2-VS-per-chip topology with Broadcom PEX890xx but on TR Pro / EPYC Genoa.** If it collapses → bug is in the switch silicon. If it doesn't → bug is CPU-IOD revision-dependent.
* **Same 8-GPU 2-VS-per-chip topology with Microchip on dual-socket EPYC Turin.** If it collapses → it's CPU-platform-dependent. If it doesn't → Broadcom-specific.
* **Intel Granite Rapids / Xeon 6 platforms (any PCIe switch)** — to confirm whether this is AMD-only or also affects Intel I/O fabrics.
* **AMD EPYC Bergamo / Siena (Zen 4c IOD revisions)** — to narrow down which IOD revisions are affected.
* **AMD-internal documentation or errata, or Broadcom firmware release notes**, that describe this specifically.
* AMD-internal documentation or errata covering IOD posted-write arbitration / scalable data fabric credits

…we'd very much like to hear whether the collapse trigger fires on those configurations or not. Open an issue on the repo or PR an additional results table.
