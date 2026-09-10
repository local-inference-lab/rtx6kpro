# Czech Tutoring Workload Quality: DeepSeek-V4-Flash vs V4.1-Flash vs Qwen3.8-Flash-Next vs GLM-5.3-Flash

Date: 2026-09-10. Measured by Festr with Claude (Fable 5.1) on the RTX PRO 6000 boxes.
Scope: **quality only** (no throughput). All four models serve a real product workload:
an AI tutor for Czech entrance exams (CERMAT), pupils aged 11 to 15. The application
keeps the truth in code (answer keys, grading, state); the model has four narrow jobs.
This page measures those jobs, in Czech, at the sampling settings the models are shipped with.

## Table of Contents

- [TL;DR: who is good for what](#tldr-who-is-good-for-what)
- [Setup](#setup)
- [What was measured](#what-was-measured)
- [Results](#results)
  - [Exam solving](#exam-solving)
  - [Message interpretation](#message-interpretation-60-messages)
  - [Error diagnosis](#error-diagnosis-20-cases)
  - [Reply formulation](#reply-formulation-20-briefs)
  - [Pairwise judge](#pairwise-judge-wl-judge--ds4-flash-thinking--judge--glm-high)
  - [End-to-end tutoring sessions](#end-to-end-tutoring-sessions-24-identical-seeds-per-model)
  - [Human review notes](#human-review-notes)
- [Caveats](#caveats)
- [How to reproduce](#how-to-reproduce)

---

## TL;DR: who is good for what

| Job | Best | Also fine | Avoid | Why |
|---|---|---|---|---|
| Writing the tutor's replies in Czech | **DeepSeek-V4.1-Flash** (`reasoning_effort` low or high) | DeepSeek-V4-Flash (thinking off), GLM-5.3-Flash (`high`) | Qwen3.8 `medium`, Qwen3.8 `none` | V4.1 and V4 write clean, natural Czech; GLM is lively and fastest but 2 % of its words are broken and it slid into Slovak once in 45 replies; Qwen xhigh is correct but dry and thinks 750 tokens per reply; Qwen medium has grammar errors and an English word |
| Classifying a pupil's chat message (JSON) | **DeepSeek-V4.1-Flash** (100 % intent, even with thinking off) | GLM-5.3-Flash `high` (98 %, 37 reasoning tokens), DS4-Flash thinking, Qwen xhigh/medium (98 %) | DS4-Flash and Qwen with thinking off (88 to 90 %) | thinking matters for V4 and Qwen, not for V4.1 |
| Diagnosing why a wrong answer went wrong | any | | | 95 to 100 % everywhere |
| Solving Czech-language exam tasks | **DS4-Flash, DS4.1-Flash** (40/40) | | GLM, Qwen (35 to 37/40) | GLM and Qwen miss idioms and i/y agreement |
| Solving math exam tasks | any (100 %) | | | with a complete assignment every model solves them |
| Cheapest thinking | GLM `high` (about 50 tokens per task) | DS4.1 low (60 to 200) | Qwen xhigh (runaway to 64k seen once) | |

Two things the first attempt at this comparison got wrong, kept here as a warning:
GLM had been served at `reasoning_effort=max` with `top_p=1` and produced nonsense, and a
12 000-token output cap silently cut off up to 40 % of reasoning runs. Both rounds were thrown
away. Everything below is the second round with fixed settings.

## Setup

| Model | Build | Engine | GPUs | Reasoning control | Notes |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash | BF16/FP8 vLLM checkpoint | vLLM, `--reasoning-parser deepseek_v4` | TP2 | thinking on by default; off via `chat_template_kwargs: {"thinking": false}` | no effort levels; reasoning token count from `reasoning` field |
| DeepSeek-V4.1-Flash | `deepseek-v4.1-flash` | SGLang, reasoning parser `deepseek-v41` | remote box, 8 concurrent | **thinking off by default**; `reasoning_effort` = `none` / `low` / `high` / `max` or a float 0 to 0.99 | reasoning returned in `reasoning_content`; SGLang does not report reasoning tokens, counts here are estimated from text length |
| Qwen3.8-Flash-Next | NVFP4 (`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`) | vLLM, `--reasoning-parser qwen3` | TP1, `--max-num-seqs 4` | `reasoning_effort` = `xhigh` (default) / `medium` / `low` / `none`; `high` and `max` return HTTP 400 | |
| GLM-5.3-Flash | `GLM-5.3-Flash-NVFP4-QAD-TVN-step2500` | vLLM, `--reasoning-parser glm45`, `--default-chat-template-kwargs {"reasoning_effort":"high"}` | TP4 | `reasoning_effort` = `high` only in this comparison (the lab's choice) | on this build `high` behaves adaptively and thinks almost nothing (0 to 50 tokens), while `medium` behaves like `max` (767 vs 1 763 reasoning tokens on the same prompt) |

Sampling for every call: `temperature 1.0`, `top_p 0.95`, `max_tokens 64000`. Nothing was
truncated except one Qwen xhigh run on a Czech task that reasoned for the full 64 000 tokens.

## What was measured

1. **Exam solving.** 32 math and 40 Czech tasks from real CERMAT tests with official keys.
   Only tasks whose complete assignment is in the text (no figure, table or base text from a
   neighbouring item); every task was reviewed by hand. Answers are compared by a unit-aware
   normalizer (`1 400 metrů` equals `1400 m`, `0,75` equals `3/4`).
2. **Message interpretation.** 60 real chat messages from simulated pupils, hand-labeled with
   intent (11 classes: answer, worked procedure, question about the task, hint request,
   explanation request, new topic, skip, stop, off-topic, social, other), the answer the pupil
   commits to, and emotion. Output constrained by a JSON schema.
3. **Error diagnosis.** 20 written wrong procedures; pick the cause from 6 candidates (JSON enum).
4. **Reply formulation.** 20 briefs (facts computed by code, the goal of the reply, forbidden
   values, length limit) → the tutor's reply. Deterministic checks (forbidden value present,
   mandatory question missing, length, copying the brief, talking about "the system"),
   typo rate by `hunspell -d cs_CZ`, and a pairwise LLM judge: two judges (DS4-Flash with
   thinking, GLM `high`), both orders, 20 briefs, so 40 judgments per judge per pair.
5. **End-to-end tutoring sessions.** 24 simulated sessions (8 scenarios × 3 seeds, identical
   seeds for every model), simulated pupils and the session judge always on DS4-Flash, the
   tutor's four LLM jobs all on the model under test. Code checks: answer key leaked, answer
   options written by the model, meta-talk, repeated sentences, fallback after failed checks.
6. **Human review.** 64 replies read by the author, every Czech-task miss checked against the key.

## Results

### Exam solving

| Model, mode | Math 32 | Czech 40 | Median reasoning tokens math / Czech |
|---|---|---|---|
| DS4-Flash, thinking off | 97 % | 100 % | 0 / 0 |
| DS4-Flash, thinking on | 100 % | 100 % | 232 / 544 |
| DS4.1-Flash `none` | 100 % | 95 % | 0 / 0 |
| DS4.1-Flash `low` | 100 % | 100 % | 153 / 288 |
| DS4.1-Flash `high` | 100 % | 98 % | 186 / 354 |
| DS4.1-Flash `max` | 100 % | 100 % | 207 / 552 |
| Qwen3.8 `none` | 100 % | 88 % | 0 / 0 |
| Qwen3.8 `medium` | 100 % | 92 % | 248 / 811 |
| Qwen3.8 `xhigh` | 100 % | 88 % | 235 / 1 130 (max 64 000) |
| GLM-5.3-Flash `high` | 100 % | 88 % | 48 / 81 |

Every Czech miss was checked by hand: the key was right each time. GLM and Qwen fail on
idioms ("jde mu to jako psovi pastva", "chleba o dvou kůrkách") and on i/y agreement
("švihadly"); DeepSeek models do not.

### Message interpretation (60 messages)

| Model, mode | Intent | Answer | Emotion | Median reasoning tokens |
|---|---|---|---|---|
| DS4-Flash, thinking off | 90 % | 87 % | 68 % | 0 |
| DS4-Flash, thinking on | 98 % | 98 % | 77 % | 276 |
| DS4.1-Flash `none` | **100 %** | 92 % | 82 % | 0 |
| DS4.1-Flash `low` | **100 %** | 98 % | 83 % | 198 |
| DS4.1-Flash `high` | **100 %** | **100 %** | 83 % | 245 |
| DS4.1-Flash `max` | **100 %** | 97 % | 83 % | 200 |
| Qwen3.8 `none` | 88 % | 92 % | 70 % | 0 |
| Qwen3.8 `medium` | 98 % | 98 % | 75 % | 339 |
| Qwen3.8 `xhigh` | 98 % | 97 % | 72 % | 332 |
| GLM-5.3-Flash `high` | 98 % | 95 % | 73 % | 37 |

Emotion labels are subjective; treat the column as relative. Each model's intent misses are
borderline cases ("hele já už musím jít, čau" labeled social instead of stop).

### Error diagnosis (20 cases)

All modes of all models: 100 %, except DS4-Flash with thinking off: 95 %.

### Reply formulation (20 briefs)

| Model, mode | Typos per 100 words | Failed hard checks | Median reasoning tokens |
|---|---|---|---|
| DS4-Flash, thinking off | 0.72 | 0 | 0 |
| DS4-Flash, thinking on | 0.78 | 0 | 285 |
| DS4.1-Flash `none` | 0.96 | 0 | 0 |
| DS4.1-Flash `low` | 0.64 | 0 | 58 |
| DS4.1-Flash `high` | 0.75 | 0 | 66 |
| DS4.1-Flash `max` | **0.43** | 0 | 96 |
| Qwen3.8 `none` | 1.11 | 1 forbidden value, 1 meta-talk | 0 |
| Qwen3.8 `medium` | 2.63 | 0 | 339 |
| Qwen3.8 `xhigh` | 0.41 | 0 | 757 (max 3 226) |
| GLM-5.3-Flash `high` | 0.87 | 1 meta-talk | 0 |

Typo rate counts words unknown to hunspell after a name and emoji whitelist; colloquial
Czech ("blbej", "týhle") inflates DS4-Flash's number, real defects ("učiteloví", "cibuly",
"zkusenost", Slovak forms) inflate GLM's. Qwen xhigh's low rate comes with the driest prose.

### Pairwise judge (W:L, judge = DS4-Flash thinking / judge = GLM high)

| Pair | Result | Order-swap consistency |
|---|---|---|
| Qwen `medium` vs Qwen `xhigh` | medium 27:13 / 23:14 | 0.75 |
| Qwen `xhigh` vs DS4-Flash off | xhigh 25:15 / 24:13 | 0.62 |
| GLM `high` vs DS4-Flash off | 20:20 / GLM 27:13 | 0.82 |
| Qwen `xhigh` vs GLM `high` | Qwen 25:14 / GLM 22:18 (judges disagree) | 0.65 |
| DS4-Flash on vs off | on 26:14 / off 22:18 (judges disagree) | 0.85 |
| DS4.1 `low` vs DS4-Flash off | **DS4.1 22:16 / 24:13** | 0.68 |
| DS4.1 `low` vs Qwen `xhigh` | **DS4.1 21:19 / 26:11** | 0.68 |
| DS4.1 `low` vs GLM `high` | DS4.1 23:15 / GLM 20:18 (judges disagree) | 0.65 |
| DS4.1 `none` vs `low` | none 23:17 / 20:16 | 0.68 |
| DS4.1 `low` vs `max` | **max 31:9 / 23:17** | 0.65 |

Read the judges with care: they reward liveliness and barely penalize grammar. They prefer
Qwen `medium` over `xhigh` although medium makes six times more typos and once dropped an
English word ("Klidně continuing") into a Czech reply. Consistency is the share of briefs
where a judge picked the same winner in both orders.

### End-to-end tutoring sessions (24 identical seeds per model)

| Tutor model, mode | Problem sessions | Code-check findings | Judge findings |
|---|---|---|---|
| DS4.1-Flash `low` | **4 / 24** | 1 (fallback after a check failed) | 6 |
| DS4.1-Flash `high` | **4 / 24** | 0 | 6 |
| GLM-5.3-Flash `high` | 5 / 24 | 0 | 9 (one reply fully in Slovak) |
| Qwen3.8 `xhigh` | 7 / 24 | 1 (false positive) | 11 |
| DS4-Flash | 9 / 24 | 3 (2 fallbacks, 1 repeated sentence) | 13 |

Most judge findings are noise (a correct answer confirmed after the pupil answered, "no
problem found" essays). Real ones: GLM's Slovak reply, DS4-Flash inventing the content of a
missing figure, Qwen ignoring a request three times.

### Human review notes

- **DeepSeek-V4.1-Flash**: natural, warm, correct Czech in every mode; right genders and
  vocatives; reacts to what the pupil wrote. `none` is slightly rough ("žádný dnešní selhání
  neexistuje"), `max` is the most polished. Same family voice as V4-Flash, a bit crisper.
- **DeepSeek-V4-Flash**: cleanest Czech among the locally served models, the most human touches
  (a concrete story about boots on sale for the "why do I need percentages" question),
  sometimes one question too many.
- **GLM-5.3-Flash `high`**: lively, pedagogically sound, fastest; but every ~50th word is
  defective (missing diacritics, wrong forms, word fragments) and one whole reply came out in
  Slovak. For a Czech tutor that undermines what it teaches.
- **Qwen3.8 `xhigh`**: factually right and disciplined, dry, hundreds to thousands of reasoning
  tokens for a two-sentence reply. `medium`: grammar errors ("Dneska ti nechceš"), invented
  words, one English word; not usable for Czech prose.

## Caveats

- Sample sizes are small (20 to 72 items per task); differences of a few points are noise.
- Latency is not compared: servers were shared and one model sat behind a VPN link.
- The judges are LLMs with 0.6 to 0.85 order-swap consistency; the human review is the tie-breaker.
- GLM's effort levels on this QAD build are not monotonic; only `high` was measured on request.
- DS4.1 reasoning token counts are estimates (SGLang reports none).

## How to reproduce

Every call is a plain OpenAI-compatible chat completion. Reasoning control per model:

```json
{"model": "DeepSeek-V4-Flash", "messages": [...], "temperature": 1.0, "top_p": 0.95, "max_tokens": 64000,
 "chat_template_kwargs": {"thinking": false}}
{"model": "deepseek-v4.1-flash", "messages": [...], "temperature": 1.0, "top_p": 0.95, "max_tokens": 64000,
 "reasoning_effort": "low"}
{"model": "Qwen3.8-Flash-Next", "messages": [...], "temperature": 1.0, "top_p": 0.95, "max_tokens": 64000,
 "reasoning_effort": "xhigh"}
{"model": "GLM-5.3-Flash-NVFP4-QAD-TVN-step2500", "messages": [...], "temperature": 1.0, "top_p": 0.95,
 "max_tokens": 64000, "reasoning_effort": "high"}
```

Reasoning tokens: `usage.completion_tokens_details.reasoning_tokens` on vLLM; on SGLang estimate
`completion_tokens - len(content) / 3.3`. Structured tasks use
`response_format: {"type": "json_schema", ...}`; on vLLM with reasoning parsers the grammar is
applied after the reasoning block, so leave thinking on for GLM.

The three public task sets and a standalone runner live next to this page in
[czech-tutor-eval/](czech-tutor-eval/): `interpretace.json` (60 labeled messages),
`diagnoza.json` (20 diagnosis cases), `formulace.json` (20 briefs) and `czech_tutor_eval.py`,
which runs interpretation, diagnosis and formulation against any OpenAI-compatible endpoint
and prints the same metrics as above (typo rate needs `hunspell` with the `cs_CZ` dictionary).
The exam-solving set is not published because CERMAT test texts are not freely redistributable;
the selection rule was: official key is a letter or a numeric value, the assignment is
complete in the text, no reference to a figure, table or another item's base text.
The end-to-end harness is part of the PureMentor application and is not public.
