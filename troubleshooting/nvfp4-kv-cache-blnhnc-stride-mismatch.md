# NVFP4 KV Cache Produces Coherent-Nothing: The BLNHC Layout / Writer Stride Mismatch

Running a QSA (Qwen3.8-Flash-Next) model with `--kv-cache-dtype nvfp4` on
vLLM 0.30.0 produces **incoherent output at every context length** — short
prompts included. It is silent: no exception, no crash, no Xid, no AER. The
server stays up and returns plausible-looking token streams, which is exactly
why this does not look like a hardware or OOM problem and is easy to waste a
day on.

This page documents the root cause (a stride mismatch between the stock
nvfp4 KV **write** kernel and the engine's default KV-cache **layout**), the
one-line workaround, and a self-contained repro. It was found and verified on
the 2× RTX PRO 5000 Blackwell (SM120) box in
[`hardware/`](../hardware/), running
[`models/qwen38-flash-next.md`](../models/qwen38-flash-next.md) with
`modelopt_mixed` NVFP4 weights.

Related pages:

- [`models/qwen38-flash-next.md`](../models/qwen38-flash-next.md) — the model
  this page is about; the FP8-KV runbook.
- [`troubleshooting/common-issues.md`](common-issues.md#nvfp4-kv-cache-incoherent-output-qsahybrid) —
  the short symptom → fix entry that points here.
- [`troubleshooting/pcie-link-speed-flapping-cpayne.md`](pcie-link-speed-flapping-cpayne.md) —
  a different silent-failure shape on the same box class (a working path that
  quietly degrades with no error); useful contrast.

Upstream: reported as
[vllm-project/vllm#61225](https://github.com/vllm-project/vllm/issues/61225).

## Table of Contents

- [System Under Test](#system-under-test)
- [Symptoms](#symptoms)
- [What Was Ruled Out](#what-was-ruled-out)
- [Root Cause](#root-cause)
- [Evidence](#evidence)
- [The Fix: Force A Head-Major Layout](#the-fix-force-a-head-major-layout)
- [Is The Fix Worth It: Quality, Speed, Capacity](#is-the-fix-worth-it-quality-speed-capacity)
- [Minimal Reproduction](#minimal-reproduction)
- [Pitfalls](#pitfalls)

## System Under Test

| Component | Configuration |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 5000 Blackwell Workstation, 48 GB, SM120, PCIe Gen4 ×16 |
| vLLM | 0.30.0 (`vllm/vllm-openai:v0.30.0`) |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` (Qwen4Exp / QSA sparse-attention, `modelopt_mixed`) |
| Parallelism | TP2, expert-parallel on |
| Flag | `--kv-cache-dtype nvfp4` |
| Engine log | `INFO ... Using BLNHC KV cache layout.` (default) |

## Symptoms

Generation is incoherent at **all** lengths with the default layout. The tell is
that deterministic, trivial sanity checks fail — a healthy model answers these
reflexively:

| Probe | Expected | Got (nvfp4 / BLNHC) |
|---|---|---|
| "what is 17 × 23, number only" | `391` | unrelated arithmetic prose |
| "capital of Japan" | `Tokyo` | "The user is asking a simple question: …" |
| "spell 'quantization' backwards" | `noitazitnauq` | meta-narration, no answer |
| ~253k-token needle retrieval | the needle code | 0 of N needles found |

The output is *well-formed* English that never actually answers — it is a model
reading back a corrupted KV cache, not a model that has quantization noise.
That distinction is the whole diagnosis: quantization noise would degrade a
correct answer; this produces a wrong answer even on a two-token question.

## What Was Ruled Out

Before reaching the layout, the usual suspects were each closed with a
negative result. They are listed because each one *looks* like the culprit and
is not:

- **The QSA read kernel** (the sparse paged GQA Triton kernel from
  [vllm#54846](https://github.com/vllm-project/vllm/pull/54846)): numerically
  correct. Writer→reader round-trips at production geometry — multi-block
  `block_table`, multi-split/K + merge, multi-request, 4096-row prefill
  chunks, `indexer_budget` top-k 2048 — match a bf16 reference on the
  dequantized cache to ≤6e-5.
- **MTP / speculative decoding**: the same garbage is produced with MTP fully
  off, so the draft path is not involved.
- **Per-layer quant scales**: the `modelopt_mixed` checkpoint carries no
  per-layer K/V/q scale tensors; stock `set_default_quant_scales` leaves
  `_k_scale`/`_v_scale` = 1.0, which matches every passing round-trip.

Each of those being clean points the diagnosis at the one component the
round-trip tests *don't* exercise in the engine's real configuration: the
physical layout the allocator chooses for the cache.

## Root Cause

The nvfp4 per-layer buffer is `[B, 2H, N, full_dim]`, where
`full_dim = head_size//2 + head_size//16` (E2M1 data + E4M3 block scales) and K
and V occupy separate head-slot halves. The write kernel
(`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`,
`reshape_and_cache_nvfp4_dispatch`) and the standard read helper
(`vllm/utils/torch_utils.py: nvfp4_split_data_scale`) make **different
assumptions** about that buffer's physical order:

| Component | How it gets head/token offsets |
|---|---|
| **Write kernel** | Detects HND vs NHD from `stride(2) > stride(1)`, then uses **internal head-major (HND) formulas** for the head/data-token strides and for the data\|scale region split (`data_per_kv = H·N·data_dim`). |
| **Read helper** | **Scales the actual tensor strides** by `data_dim/full_dim`. |

The two agree only when the physical page is **head-major**. The engine's
layout resolver (`vllm/v1/attention/backends/utils.py: resolve_kv_cache_layout`)
picks the **first** entry of the backend's `supported_kv_cache_layouts()`. For
the QSA backend that list is `(BLNHC, BLHNC)`, so it resolves to
**BLNHC** — a *token-major* layout, where each token is a row of `2H` head
slots × `full_dim` bytes and K/V slots are interleaved per token, each slot
packing data+scale. Under that layout the write kernel's internal HND formulas
place head `h`, token `t`, and the scale region at byte offsets that do not
match where the reader looks. Every K/V write (and its block scales) lands in
the wrong bytes; the reader then faithfully reads the garbage back.

The asymmetry that makes this easy to miss: a write kernel run against a
contiguous or head-major buffer (e.g. any test that builds `torch.zeros(...)`
and passes `[B, H, N, C]`) is self-consistent and **passes**. The token-major
page is only ever produced by the real allocator, so round-trip tests that do
not recreate the engine's physical layout never see it. That is why the kernel
is fine and the model is not.

## Evidence

Driving the stock writer and the standard reader against both physical layouts
at production geometry (B=2, N=5568, H=2, D=256, `full_dim`=144) and measuring
the writer→reader round-trip (dequantized cache vs. the original bf16 K/V):

| physical layout | K relerr | V relerr | verdict |
|---|---:|---:|---|
| token-major (BLNHC — engine default) | **1.4644** | **1.2185** | corrupt |
| head-major (BLHNC) | **0.0895** | **0.0895** | correct (pure nvfp4 rounding) |

0.0895 is the expected nvfp4 quantization error (E2M1 + E4M3 block scale) —
the same value a known-good round-trip produces. 1.46 / 1.22 means the
dequantized cache is unrelated to the input.

## The Fix: Force A Head-Major Layout

Set the layout the write kernel expects:

```bash
VLLM_KV_CACHE_LAYOUT=BLHNC
```

(equivalently `--kv-cache-layout BLHNC` where the engine exposes it). With
this, the same server is coherent: deterministic sanity checks all pass,
~253k-context needle retrieval is 5/6, and C=3 full-context concurrency is
clean with no OOM or preemption. `BLHNC` is block-compact (L and B outermost),
so it also satisfies the hybrid / KV-group layout constraints this model has —
it is a valid layout, not a hack.

Add it to the model's launch command alongside `--kv-cache-dtype nvfp4`.

## Is The Fix Worth It: Quality, Speed, Capacity

With the layout fixed, nvfp4 KV is quality-safe but not speed-free. On this
box, same protocol, MTP on:

| KV dtype | quality | 253k cold TTFT | 253k decode TPOT | KV pool | full-ctx conc |
|---|---|---:|---:|---:|---|
| bf16 | 7/7 | 29.4 s | 4.5–4.6 ms | 1.17× | C=1 |
| **fp8** (default) | 7/7 | 27.8 s | 6.9–7.8 ms | 2.00× | C=2 (C=3 degrades) |
| **nvfp4 + BLHNC** | 7/7, needle 5/6 | 47.2 s | 7.5 ms | 2.50× | C=3 clean |

Two takeaways:

- **Quality: no measurable loss, with one truncation caveat.** All three arms
  pass 7/7 on the deterministic quality gate. On the ~253k-context
  needle-retrieval protocol (`needle_multi.py`, SEED=42): fp8 scores 6/6,
  nvfp4+BLHNC scores 5/6. The single nvfp4 miss is **not** a retrieval
  failure — all five secret codes are found and the sum (3799) is computed
  correctly in the model's reasoning; the final answer line is cut
  mid-word (`…XEPSILON-19`) by the bench's `max_tokens=512` cap because the
  nvfp4 run's reasoning trace was longer and hit the generation limit.
  It is a harness truncation artifact, not a KV-quality degradation.
  The cost is asymmetric: nvfp4 is ~1.7× slower in **prefill** (TTFT) but
  nearly unchanged in **decode** (TPOT). The choice is driven by prefill
  time and capacity, not accuracy.
- **Capacity is the only real win.** The 2.50× pool is what makes C=3 at full
  context survive, where fp8 degrades. If your workload is single long
  sessions, fp8 is the better default (faster prefill, no layout flag needed).
  Reach for nvfp4+BLHNC when you need to hold more concurrent full-context
  sessions.

## Minimal Reproduction

Self-contained, one GPU, ~300 MB VRAM at production geometry. `REPRO_B` /
`REPRO_N` shrink it on a busy box — the mismatch is a layout property, not a
size property. Needs any vLLM build with the nvfp4 KV write kernel
(0.30.0 works).

```python
#!/usr/bin/env python3
import os, torch
from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
from vllm.utils.torch_utils import nvfp4_split_data_scale

DEV = "cuda"
B = int(os.environ.get("REPRO_B", 2))
H, D = 2, 256
N = int(os.environ.get("REPRO_N", 5568))
HD, FD = 2 * H, D // 2 + D // 16      # K+V slots; nvfp4 full_dim = 144
PAGE, SD = HD * N * FD, D // 16
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def _e4m3(bits):
    p = bits.to(torch.int32) & 0x7F
    e, m = (p >> 3) & 0x0F, p & 0x7
    v = torch.where(e == 0, m.float() / 512.0,
                    torch.pow(2.0, e.float() - 7.0) * (1.0 + m.float() / 8.0))
    return torch.where(p == 0, torch.zeros_like(v), v)

def _scale_coords(n, swizzled, dev):
    t, s = (torch.arange(n, device=dev).view(-1, 1),
            torch.arange(SD, device=dev).view(1, -1))
    if not swizzled:
        return t.expand(n, SD), s.expand(n, SD)
    g = SD // 4
    return (t // 4) * 4 + (s // g), (s % g) * 4 + (t % 4)

def dequant_side(side, swizzled, device):
    data, scales = nvfp4_split_data_scale(side)
    data, scales = data.contiguous(), scales.view(torch.uint8)
    b, n, h, _ = data.shape
    table = torch.tensor(_E2M1 + [-v for v in _E2M1],
                         dtype=torch.float32, device=device)
    nib = torch.stack((data & 0x0F, (data >> 4) & 0x0F), dim=-1)
    vals = table[nib.reshape(b, n, h, D).long()]
    ti, si = _scale_coords(n, swizzled, device)
    picked = scales[:, ti.reshape(-1), :, :].reshape(b, n, SD, h, SD)
    sel = si.view(1, n, SD, 1, 1).expand(b, n, SD, h, 1)
    picked = torch.gather(picked, 4, sel).squeeze(4).permute(0, 1, 3, 2)
    return (vals * _e4m3(picked).repeat_interleave(16, dim=3)).to(torch.bfloat16)

def relerr(a, b):
    a, b = a.to(torch.float64), b.to(torch.float64)
    return ((a - b).abs().mean() / b.abs().mean().clamp(min=1e-12)).item()

def run(order):
    raw = torch.zeros(B * PAGE, dtype=torch.uint8, device=DEV)
    view = (torch.as_strided(raw, (B, HD, N, FD), (PAGE, FD, HD * FD, 1))
            if order == "token-major"
            else torch.as_strided(raw, (B, HD, N, FD), (PAGE, N * FD, FD, 1)))
    g = torch.Generator(device=DEV); g.manual_seed(0)
    key = (torch.randn(B * N, H, D, device=DEV, generator=g) * 0.5).bfloat16()
    val = (torch.randn(B * N, H, D, device=DEV, generator=g) * 0.5).bfloat16()
    scale = torch.tensor(1.0, dtype=torch.float32, device=DEV)
    slots = torch.arange(B * N, dtype=torch.int64, device=DEV)
    reshape_and_cache_flash(key, val, view[:, :H].transpose(1, 2),
                            view[:, H:].transpose(1, 2), slots, "nvfp4",
                            scale, scale)
    torch.cuda.synchronize()
    k_dq = dequant_side(view[:, :H].transpose(1, 2), False, DEV)
    v_dq = dequant_side(view[:, H:].transpose(1, 2), True, DEV)
    kr, vr = (relerr(k_dq.reshape(B*N, H, D), key),
              relerr(v_dq.reshape(B*N, H, D), val))
    print(f"{order:13s}  K relerr={kr:.4f}  V relerr={vr:.4f}  "
          f"-> {'CORRUPT' if max(kr, vr) > 0.3 else 'OK (nvfp4 rounding)'}")

run("token-major")   # engine default for QSA (BLNHC)
run("head-major")    # VLLM_KV_CACHE_LAYOUT=BLHNC
```

Measured output:

```
token-major    K relerr=1.4644  V relerr=1.2185  -> CORRUPT
head-major     K relerr=0.0895  V relerr=0.0895  -> OK (nvfp4 rounding)
```

## Pitfalls

**A passing round-trip does not clear the layout.** Any test that builds the
cache from a plain buffer or a head-major view exercises only the HND branch of
the write kernel and will pass. To test the nvfp4 write path meaningfully you
must recreate the engine's *physical* per-layer strides — which is what the
repro above does by building the raw page and applying `as_strided` with the
BLNHC vs BLHNC stride orders.

**The corruption is silent and total, so do not debug it as a model problem.**
Incoherent output at *every* length (including two-token prompts) with a clean
log is the signature of a corrupt KV read/write path, not a quantization or
weight problem. A model with a genuinely bad checkpoint or a broken read kernel
usually still gets the short, deterministic prompts right.

**BLHNC is not a workaround that trades one layout bug for another.** It is
block-compact (L and B outermost), which is exactly the property the hybrid
KV-group allocator needs, and it is in the QSA backend's declared layout set.
Forcing it is a valid configuration, not a forced fallback.

**Do not confuse this with the GLM-5 FP8 garbled-output bug.**
[common-issues.md](common-issues.md) has a separate "FP8 KV Cache Garbled
Output (GLM-5)" entry — that one is missing dequant scales in the FlashInfer
backend, a different root cause on a different model. Both present as garbled
output, but the fix and the backend are unrelated.
