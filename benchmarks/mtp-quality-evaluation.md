# MTP Quality Evaluation

Does MTP (Multi-Token Prediction) speculative decoding affect model accuracy? We ran GPQA Diamond and GSM8K with and without MTP across two NVFP4 checkpoints to find out.

## Table of Contents

- [Summary](#summary)
- [Test Environment](#test-environment)
- [GPQA Diamond Results](#gpqa-diamond-results)
  - [Per-Run Scores](#per-run-scores)
  - [Aggregate Statistics](#aggregate-statistics)
  - [MTP Impact per Checkpoint](#mtp-impact-per-checkpoint)
  - [Checkpoint Comparison](#checkpoint-comparison)
- [GSM8K Results](#gsm8k-results)
- [Hard Math Test](#hard-math-test)
- [Conclusions](#conclusions)
- [Raw Data](#raw-data)

---

## Summary

| Configuration | GPQA Mean | GSM8K (thinking) | GSM8K (no thinking) | Wall Time (GPQA) |
|:---|:---:|:---:|:---:|:---:|
| lukealonso/NVFP4 + MTP | **88.26%** | **99.0%** | **44%** | ~1h 29m |
| lukealonso/NVFP4, no MTP | 87.50% | — | — | ~1h 48m |
| nvidia/NVFP4 + MTP | 87.44% | 97.5% | 39% | ~1h 43m |
| nvidia/NVFP4, no MTP | 86.55% | — | — | ~2h 15m |

**Key findings:**
1. **MTP does not degrade quality** — differences are within statistical noise
2. **MTP provides 18-24% faster inference** — a free speedup
3. **lukealonso checkpoint consistently outperforms nvidia** across all benchmarks

---

## Test Environment

| Parameter | Value |
|:---|:---|
| **GPU** | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB each) |
| **Engine** | SGLang 0.5.9 |
| **Container** | `voipmonitor/llm-pytorch-blackwell:nightly-cuda132` |
| **Date** | 2026-03-11 |
| **Eval framework** | [simple-evals](https://github.com/openai/simple-evals) (ChatCompletionSampler) |

### Server config (common)

```
--tensor-parallel-size 8
--quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3
--attention-backend triton
--moe-runner-backend flashinfer_cutlass
--fp4-gemm-backend flashinfer_cudnn
--cuda-graph-max-bs 128
--max-running-requests 128
--context-length 262144
--chunked-prefill-size 32768
--mem-fraction-static 0.80
--disable-custom-all-reduce
--disable-shared-experts-fusion
--schedule-conservativeness 0.1
```

### MTP-specific flags (only for MTP ON tests)

```
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6
```

### Known warning in all runs

```
DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0.
This might cause accuracy degradation on Blackwell.
```

### GPQA eval config

| Parameter | Value |
|:---|:---|
| **Benchmark** | GPQA Diamond (198 questions) |
| **Repeats** | 8 per configuration |
| **Parallel workers** | 8 |
| **Temperature** | 0.0 |
| **Max tokens** | 64,000 |
| **Thinking mode** | Enabled (`chat_template_kwargs: {thinking: True}`) |

---

## GPQA Diamond Results

### Per-Run Scores

| Run | lukealonso MTP | lukealonso No MTP | nvidia MTP | nvidia No MTP |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.889 | 0.864 | 0.859 | 0.864 |
| 2 | 0.879 | 0.874 | 0.904 | 0.859 |
| 3 | 0.869 | 0.894 | 0.859 | 0.869 |
| 4 | 0.884 | 0.869 | 0.884 | 0.864 |
| 5 | 0.904 | 0.889 | 0.879 | 0.848 |
| 6 | 0.884 | 0.864 | 0.859 | 0.864 |
| 7 | 0.879 | 0.874 | 0.874 | 0.869 |
| 8 | 0.874 | 0.874 | 0.879 | 0.889 |

### Aggregate Statistics

| Metric | lukealonso MTP | lukealonso No MTP | nvidia MTP | nvidia No MTP |
|:---|:---:|:---:|:---:|:---:|
| **Mean** | **0.8826** | **0.8750** | **0.8744** | **0.8655** |
| Std (across runs) | 0.0106 | 0.0109 | 0.0155 | 0.0117 |
| Min | 0.869 | 0.864 | 0.859 | 0.848 |
| Max | 0.904 | 0.894 | 0.904 | 0.889 |
| Median | 0.8815 | 0.874 | 0.877 | 0.864 |
| Avg response length (chars) | 2550 | 2512 | 2436 | — |
| Wall time | ~1h 29m | ~1h 48m | ~1h 43m | ~2h 15m |

### MTP Impact per Checkpoint

| Checkpoint | MTP ON | MTP OFF | Delta | Speed Improvement |
|:---|:---:|:---:|:---:|:---:|
| lukealonso | 88.26% | 87.50% | +0.76pp (within noise) | ~18% faster |
| nvidia | 87.44% | 86.55% | +0.89pp (within noise) | ~24% faster |

MTP vs no-MTP Welch's t-test (lukealonso): t=1.41, p=0.18 — **not statistically significant** at α=0.05.

Both checkpoints show a small positive delta with MTP, but this is within run-to-run variance. MTP is theoretically lossless: the target model verifies all speculated tokens and rejects incorrect ones, so output should match standard autoregressive decoding.

### Checkpoint Comparison

| Benchmark | lukealonso | nvidia | Delta |
|:---|:---:|:---:|:---:|
| **GPQA** (MTP, thinking) | **88.26%** | 87.44% | **+0.82pp** |
| **GPQA** (no MTP, thinking) | **87.50%** | 86.55% | **+0.95pp** |
| **GSM8K** (thinking) | **99.0%** | 97.5% | **+1.5pp** |
| **GSM8K** (no thinking, 5-shot) | **44%** | 39% | **+5.0pp** |
| **Hard Math** (no thinking) | **89.5%** | 84.2% | **+5.3pp** |

lukealonso wins every benchmark. The gap widens without thinking mode (+5pp on GSM8K, +5.3pp on Hard Math) because chain-of-thought can compensate for quantization errors. This is consistent with lukealonso's lower KLD (0.035 vs 0.109, see [KLD evaluation](kld-evaluation.md)) and community reports of nvidia NVFP4 accuracy issues (vLLM Issue #36094).

---

## GSM8K Results

### With thinking mode

| Model | Score | Std | Config |
|:---|:---:|:---:|:---|
| lukealonso | **99.0%** | 0.099 | 200 examples, max-tokens 16000 |
| nvidia | 97.5% | 0.156 | 200 examples, max-tokens 16000 |

lukealonso not only scores higher but has lower variance (std 0.099 vs 0.156), suggesting more stable outputs.

### Without thinking (5-shot)

| Model | Score | Config |
|:---|:---:|:---|
| lukealonso | **44%** | 200 examples, max-tokens 2048 |
| nvidia | 39% | 200 examples, max-tokens 2048 |

Without chain-of-thought reasoning, the quantization quality gap is much more pronounced (+5pp).

---

## Hard Math Test

19 custom math questions, no thinking mode.

| Q# | Question | lukealonso | nvidia |
|:---:|:---|:---:|:---:|
| 1 | (37×43)−(29×51)+17 | FAIL (139) | FAIL (10) |
| 2 | 123² − 113² | OK | OK |
| 3 | 2³¹ mod 7 | FAIL (4) | FAIL (1) |
| 4 | log₂(x)=5.5, x=? | OK | OK |
| 5 | P(2 aces in row) | OK | OK |
| 6 | LCM(12,18,20) | OK | OK |
| 7 | Primes < 50 | OK | OK |
| 8 | Sum primes < 30 | OK | OK |
| 9 | (root1+1)(root2+1) for x²−7x+12=0 | **OK (20)** | **FAIL (30)** |
| 10 | 2ᵃ×3ᵇ=72, a+b=? | OK | OK |
| 11 | Altitude to hypotenuse | OK | OK |
| 12 | 10th Fibonacci | OK | OK |
| 13 | Geometric sequence 8th term | OK | OK |
| 14 | MISSISSIPPI arrangements | OK | OK |
| 15 | C(8,3) | OK | OK |
| 16 | Infinite geometric series sum | OK | OK |
| 17 | 2×2 determinant | OK | OK |
| 18 | Sum 1 to 100 | OK | OK |
| 19 | 13³ | OK | OK |

**Result:** lukealonso 17/19 (89.5%), nvidia 16/19 (84.2%). Key difference: Q9 (algebraic manipulation) — lukealonso correctly computes 20, nvidia incorrectly answers 30.

---

## Conclusions

### 1. Enable MTP for production serving

MTP provides 18-24% inference speedup with no measurable accuracy degradation. The verification mechanism in speculative decoding guarantees output fidelity.

### 2. Use lukealonso over nvidia checkpoint

lukealonso/Qwen3.5-397B-A17B-NVFP4 consistently outperforms nvidia/Qwen3.5-397B-A17B-NVFP4 across all benchmarks (+0.8pp to +5.3pp). The advantage is especially pronounced without thinking mode. This aligns with KLD measurements (0.035 vs 0.109) and community reports.

### 3. Recommended production config

```bash
# Model
--model lukealonso/Qwen3.5-397B-A17B-NVFP4

# MTP (speculative decoding)
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6

# Bug mitigations
--disable-shared-experts-fusion
--disable-custom-all-reduce
```

---

## Raw Data

### GPQA — lukealonso MTP

```json
{
  "chars": 2549.5454545454545,
  "chars:std": 510.686619847691,
  "score:std": 0.3321451120698461,
  "scores": ["0.889", "0.879", "0.869", "0.884", "0.904", "0.884", "0.879", "0.874"],
  "mean_score": 0.8825757575757577
}
```

### GPQA — lukealonso No MTP

```json
{
  "chars": 2512.040404040404,
  "chars:std": 470.0423023987109,
  "score:std": 0.3321451120698461,
  "scores": ["0.864", "0.874", "0.894", "0.869", "0.889", "0.864", "0.874", "0.874"],
  "mean_score": 0.875
}
```

### GPQA — nvidia MTP

```json
{
  "chars": 2435.686868686869,
  "chars:std": 526.6866921099777,
  "score:std": 0.32637362467481845,
  "scores": ["0.859", "0.904", "0.859", "0.884", "0.879", "0.859", "0.874", "0.879"],
  "mean_score": 0.8743686868686869
}
```

### GPQA — nvidia No MTP

```
Scores: ['0.864', '0.859', '0.869', '0.864', '0.848', '0.864', '0.869', '0.889']
Mean: 0.8655
```
