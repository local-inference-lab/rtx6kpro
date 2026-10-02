# GLM-5.3-Flash: greedy checksum stability and MoE activation precision

Measured 2026-10-02 on the Karmic Kraken beta. The test is
[`needle-checksum`](https://github.com/local-inference-lab/llm-inference-bench#needle-checksum-greedy-numerical-stability)
in llm-inference-bench 0.7.6.

## Summary

| Question | Answer |
|---|---|
| Does today's default (QAD checkpoint, b12x W4A4) still fail? | Yes. 5.9% of identical greedy requests answer 3330 (41/700). |
| What fixes it? | b12x W4A16 activations (`VLLM_B12X_MOE_FP4_FORCE_A16=1`): 0/500. |
| What does W4A16 cost? | Prefill −10 to −11%. Decode unchanged. KV −2%. |
| FP32 router weights (`B12X_W4A16_FP32_TOPK_WEIGHTS=1`)? | No measurable cost. On QAD they were not needed (0/500 either way). On the pre-QAD checkpoint: 19.5% → 4.0%. |
| W4A8 (MXFP8 activations)? | 0.6% wrong, but slower than W4A16 in both prefill and decode, and 16% less KV. It needs a vLLM change (not in any image). |
| b12x vs Marlin (the 2026-09-17 report said 72% vs 28%)? | Not reproduced: 19.5% vs 20.5% on the same checkpoint. |
| Root cause | A near-tie in the model on one digit, combined with run-to-run nondeterminism of the serving stack. |

## Test principle

| Item | Value |
|---|---|
| Origin | Checksum probe shared by logprobz on the Local Inference Lab Discord, 2026-09-17 |
| Prompt | 8,192 tokens: 291 inert `REC` lines and three `CANONICAL FACT` lines (ALPHA=478, BETA=788, GAMMA=426) |
| Task | `CHECKSUM = ALPHA + 2*BETA + 3*GAMMA`, submitted in one `submit_context_check` tool call |
| Correct answer | 3332 (= 478 + 1576 + 1278) |
| Typical wrong answer | 3330 |
| Request | temperature 0, top_p 0.95, seed 275001, max_tokens 4096, thinking with `reasoning_effort: max` |
| Repetition | The same request 500 times at concurrency 8 |
| Meaning | Identical greedy requests should give identical answers. The wrong-answer rate measures numerical noise at a near-tie of the model. |
| Scope | One prompt and one failure mode. This is not a general quality benchmark. |

### Where the error comes from

| Reasoning form the model writes | Next token | logprob("2") − logprob("0") |
|---|---|---|
| One-step sum `478 + 1576 + 1278 = 333?` | near-tie | mean −1.8 to +4.2 nats depending on configuration; single requests range from −9.1 to +9.1 |
| Stepwise `2054 + 1278 = 333?` | safe | +12 to +15 nats |

- Retrieval never failed: ALPHA, BETA and GAMMA were always correct.
- A first "3330" is often corrected later in the reasoning ("Wait, let me recompute"). A run fails only when it is not corrected.

### The tool call is required

| Prompt variant | QAD W4A4 | QAD W4A16 + FP32 weights | pre-QAD W4A16 |
|---|---|---|---|
| Original, answer through a tool call | 6.0% wrong | 0% | 18.0% |
| Plain answer line, no tool | 0% | 0% | 0% |

- Without the tool schema the model computes stepwise and never reaches the near-tie.

## Setup

| Item | Value |
|---|---|
| GPUs | 4× RTX PRO 6000 Blackwell Max-Q Workstation, 325 W, memory 13,365 MHz, PCIe Gen5 x16, driver 615.71.09 |
| Runtime | Content-equivalent to `karmic-kraken-beta-20261001-020df706`: vLLM `4a379ed4`, b12x `b557d878`, CUTLASS DSL 4.7.1 |
| Profile | `glm53-flash`, TP4, MTP 3 tokens (draft MoE on Marlin), b12x MoE/attention/linear, FP8 KV, 32 sequences |
| QAD checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` main `175ae8ce` (2026-09-16, current default) |
| pre-QAD checkpoint | same repository, revision `46aaae8a` (2026-09-04) |

| Variant | How it was selected |
|---|---|
| W4A4 | Profile default |
| W4A16 | `VLLM_B12X_MOE_FP4_FORCE_A16=1` |
| FP32 router weights | `B12X_W4A16_FP32_TOPK_WEIGHTS=1` (W4A16 only) |
| W4A8 (MXFP8) | Experimental vLLM patch: requests `kMxfp8Dynamic` for NVFP4 experts, which runs b12x `w4a8_nvfp4` |
| Marlin | `MOE_BACKEND=marlin` |
| No MTP | `SPECULATOR=off` |

## Results: wrong answers

Each cell is the share of requests answering anything other than 3332. In the "original script" column, `n` is the number of requests.

| Checkpoint | MoE | Router weights | Original script, C8 | Bench profile, C8 | Bench profile, C30 |
|---|---|---|---|---|---|
| QAD | b12x W4A4 (default) | — | **5.9%** (n=700) | **6.0%** | 9.0% |
| QAD | b12x W4A8 (experimental) | — | 0.6% (n=500) | — | — |
| QAD | b12x W4A16 | BF16 | — | **0%** | 0% |
| QAD | b12x W4A16 | FP32 | 0% (n=500) | 0% | 0% |
| pre-QAD | b12x W4A4 | — | 9.5% (n=200) | — | — |
| pre-QAD | b12x W4A4, no MTP | — | 22.0% (n=200) | — | — |
| pre-QAD | b12x W4A16 | BF16 | 19.5% (n=200) | 18.0% | 23.2% |
| pre-QAD | b12x W4A16 | FP32 | 4.0% (n=200) | — | — |
| pre-QAD | Marlin W4A16 | BF16 | 20.5% (n=200) | — | — |

- Higher concurrency adds batch-composition noise. Compare runs only at the same concurrency.
- With n=200, the uncertainty is about ±3 percentage points.

### Which path the model takes

The table classifies the first decision point of each run.

| Configuration | Wrong | Runs that start with the one-step sum | Mean (median) margin there | First digit "0" there |
|---|---|---|---|---|
| pre-QAD W4A4 | 9.5% | 42% | −1.77 (−2.12) | 58 of 85 |
| pre-QAD W4A4, no MTP | 22.0% | 60% | −1.19 (−1.19) | 74 of 120 |
| pre-QAD W4A16 | 19.5% | 98% | +0.77 (+1.12) | 67 of 195 |
| pre-QAD Marlin | 20.5% | 94% | +0.17 (+0.38) | 88 of 189 |
| pre-QAD W4A16 + FP32 weights | 4.0% | 96% | +1.40 (+1.63) | 41 of 192 |
| QAD W4A4 | 6.6% | 86% | +2.63 (+3.13) | 78 of 428 |
| QAD W4A8 | 0.6% | 57% | +4.16 (+4.50) | 13 of 285 |
| QAD W4A16 + FP32 weights | 0% | 3% | +1.36 (+0.87) | 4 of 15 |

- Numerics change both which path the model takes and the margin on the near-tie.
- QAD W4A16 almost always computes stepwise, so the near-tie never comes up.

## Results: speed (QAD, TP4, MTP)

Decode is aggregate output tok/s at context 0 with 30 s windows (llm_decode_bench 0.7.3). Prefill is standalone. Mean MTP acceptance is about 2.5 tokens/step in every row.

| Variant | C1 | C8 | C16 | Prefill 8K | Prefill 32K | KV tokens |
|---|---|---|---|---|---|---|
| W4A4 (default) | 240 | 821 | 1197 | 12,502 | 13,410 | 5.62M |
| W4A8 (experimental) | 220 (−8.5%) | 749 (−8.8%) | 1119 (−6.5%) | 9,779 (−21.8%) | 10,610 (−20.9%) | 4.70M (−16.4%) |
| W4A16, BF16 router weights | 241 (+0.5%) | 819 (−0.2%) | 1182 (−1.3%) | 11,217 (−10.3%) | 11,936 (−11.0%) | 5.51M (−2.0%) |
| W4A16, FP32 router weights | 242 (+0.6%) | 806 (−1.9%)* | 1184 (−1.1%) | 11,175 (−10.6%) | 11,907 (−11.2%) | 5.51M (−2.0%) |

- Percentages are relative to W4A4.
- The two W4A16 rows were measured back-to-back in A/B/B/A order, 4 decode samples each.
- \* The C8 difference matches lower MTP acceptance in that sample (2.49 vs 2.53). Per decode step it is −0.2%.
- FP32 router weights move the router-weight multiply from the FC2 epilogue into the FP32 top-k sum. The amount of work stays the same.

## Results: determinism

Pre-QAD checkpoint, W4A4, identical requests sent one after another (no batching).

| Test | Result |
|---|---|
| 8 identical requests, sequential | 6 distinct outputs (5 without MTP) |
| Fresh prefill each time (unique `cache_salt`) | First generated token: chosen-token logprob differs by 0.03–0.34; top-5 alternatives by up to 1.6–3.0 nats |
| Prompt restored from the prefix cache | First token identical. From token 2 on, chosen tokens differ by up to 0.29 and alternatives by up to 2.7 nats |

| Configuration (each also without MTP) | Still nondeterministic |
|---|---|
| Profile defaults | yes |
| KDA gate and shared-expert side streams off | yes |
| Side streams off, Marlin MoE instead of b12x | yes |
| Side streams off, Marlin, b12x PCIe all-reduce off | yes |
| Side streams off, Marlin, linear backend `auto` instead of b12x | yes |

Not yet tested:
- b12x GLM indexer top-k (`run_paged_topk`)
- sparse MLA
- KDA kernels
- FP8 KV writer

Related observation: vLLM's `persistent_topk` and `top_k_per_row_decode` return the selected indices in a different order on every call (195–200 distinct orders in 200 calls). When values tie at the k-th position, the selected set changes too. GLM-5.3-Flash uses the b12x top-k instead; its determinism is not yet tested.

## Side findings

| Item | Observation |
|---|---|
| `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` | GLM fails during b12x preparation: `planned dynamic direct routing is unsupported for this launch shape`. `_heuristic_dynamic_route_mode` plans with `deterministic_output=False`, and another path also selects direct routing. |
| `ENABLE_PREFIX_CACHING=0` | GLM fails at startup: `Split GLM-5.3 target and recurrent-state pages require mamba_cache_mode='align'` |
| W4A8 for NVFP4 | b12x supports it (`w4a8_nvfp4`). vLLM's ModelOpt NVFP4 MoE path never requests MXFP8 activations. |
| `VLLM_B12X_NVFP4_ACTIVATION_MODE` | Defined in vLLM but never read. The GLM profile's derived value has no effect. |

## Options

| Change | Wrong answers (QAD) | Speed |
|---|---|---|
| Keep W4A4 (today) | 5.9% | baseline |
| `VLLM_B12X_MOE_FP4_FORCE_A16=1` | 0% | prefill −10 to −11%, decode unchanged |
| plus `B12X_W4A16_FP32_TOPK_WEIGHTS=1` | 0% | no extra cost |
| W4A8 (needs a vLLM change) | 0.6% | decode −7 to −9%, prefill −21%, KV −16% |

Open: find the nondeterministic kernel (candidates: b12x indexer top-k, KDA kernels).

## Reproduce

| Step | Command |
|---|---|
| Serve GLM-5.3-Flash with tool calling | any `glm53-flash` profile of the Karmic Kraken image |
| Run the probe | `python3 llm_decode_bench.py --port PORT --model MODEL --profile needle-checksum` |
| Change concurrency or runs | `--profile-concurrency 30`, `--profile-runs 1000` |
| Labels | `EXACT`, `NEAR_MISS` (3330), `WRONG_SUM`, `WRONG_FACTS`, `FAIL`, `TRUNC` |
| Duration | about 70 s at C8 on GLM TP4 |
