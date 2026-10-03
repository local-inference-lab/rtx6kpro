# GLM-5.2 coupled SQG K96Tail on Infernal Invocation r11

**Status: locally qualified; immutable publication pending.** The assembled
checkpoint, both exact-r11 KLD methods, TP4/DCP4/MTP3 native-SQG serving
evidence, five Estonia runs, five LAVD runs, and the combined quality receipt
are sealed locally. The immutable Hugging Face revision and upstream merge
remain pending. The only acceptable final serving/qualification base is the newest
pinned Infernal Invocation r11/MTP3 lineage identified below.

## Identity and lineage

The final runtime lineage is exact Infernal Invocation **r11/MTP3**. Infernal
Invocation r13 contributes only the native GLM SQG loader and kernel feature
heads. It is not a serving base, an image identity, or a source of results for
K96Tail. The v20 image family is not used for code, serving, or results.

| Item | Exact value | State |
|---|---|---|
| Base release | Infernal Invocation r11 | pinned |
| Base image | `voipmonitor/vllm:infernal-invocation-vllm908522a-b12x5d648d9-fi1ac6942-cu133-torch213-20260813-r11` | pinned |
| Base registry digest | `sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971` | pinned |
| Base image ID | `sha256:f226a6fd788bb4af345a17b768654f1e5a7487a812746ccb117aa9b040a82294` | pinned |
| KLD/hidden-replay tpfix image | `verdictai/glm52-k96-ii-r11:20260815-tpfix` | sealed KLD receipts only |
| KLD/hidden-replay image ID | `sha256:79dfbf6e697a1081016a3256ff5c96b1c2d6dcddd6f04662c9c693881310fa87` | measured |
| Final serving candidate | `verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix` | locally qualified; public receipt link pending |
| Final serving candidate image ID | `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def` | measured image identity |
| vLLM base | `ce5f50f6d01b02336c4207f11277fd7bedacb4d6` | r11 |
| vLLM integration tree | `908522a320ecc26582926228c9644af085f5a86c` | r11 |
| B12X integration tree | `5d648d944a047d4fac5c2035309c207b3faebd9c` | r11 |
| vLLM native-SQG donor | PR #315, head `ca966847` | donor only |
| B12X native-SQG donor | PR #197, head `b234532` | donor only |
| Complete vLLM overlay patch | `a357dc85bdb306927a4e6cf4a572a284688f86a62ee250fbcc0b33f2435e42f2` | ordered patch 1 |
| Supplemental MTP3 loader patch | `658cbdd678774b0a1167c6244ff78d627b77613709f610e26e8d0e1933cfa03e` | ordered patch 2 |
| Exact vLLM package | `0.26.1rc0+infernal.invocation.cu133.r11.vllm908522a.b12x5d648d9` | measured in KLD receipt |
| Target checkpoint repository | `brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail` | public immutable revision pending |

The r11 release page records the base identity and its qualified MTP3 profiles
in [`glm5.2-infernal-invocation-r11.md`](glm5.2-infernal-invocation-r11.md).
The donor release is documented separately in
[`glm5.2-infernal-invocation-r13-sqg.md`](glm5.2-infernal-invocation-r13-sqg.md).
Its TP4/DCP1/MTP0 result is not a result for this checkpoint or runtime. The
NVFP4 and EXL3 results on the r11 base-release page also belong to different
checkpoints; they establish base-release behavior, not K96Tail quality or
performance.

The local derived-image contract is
`/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/Dockerfile`, SHA-256
`39d4786eedf2a5f1ea469809374cfb70330f2e1c0feb7349199e4c7e5aa20ec5`.
The publication-closure provenance manifest is
`/home/brandonmusic/KLC_SANDBOXES/glm52-sqg-mcg-experiments-github/coupled_hadamard_k96tail/runtime/exact-ii-r11/PROVENANCE.json`,
SHA-256
`0c6c4694461e9d42e87ec048cc4a20ad115327487e0e8b89d31278f527d5f931`.
That manifest explicitly forbids r13 as the base and forbids v20 lineage.

The final vLLM source composition is ordered. On a clean archive of base
`ce5f50f6d01b02336c4207f11277fd7bedacb4d6`, the complete vLLM patch passed
`git apply --check` and applied; the supplemental MTP3 loader patch then passed
`git apply --check` and applied. `py_compile`, Ruff, and diff-check also passed.
No passing pytest result is claimed for the new MTP test.

### Result-provenance firewall

Only results whose receipts identify the K96Tail checkpoint and their exact
named exact-r11 candidate image may be promoted on this page. The sealed KLD
receipts remain tied to the `tpfix` image above; final serving receipts must be
tied to `tpfix-mtpfix`. Release-family similarity, shared donor commits, or
matching TP count is insufficient.

| Evidence source | May be claimed for K96Tail? | Reason |
|---|---|---|
| Sealed exact-r11 K96Tail TP4/DCP1 KLD receipt below | **Yes, within its stated DCP1 quality geometry only** | Exact checkpoint, image ID, token/reference hashes, and runtime version are recorded. |
| Sealed exact-r11 K96Tail pre-LM-head replay receipt below | **Yes, within its one-context replay scope and disclosed post-observation v2 gate only** | Exact capture, LM head, token, reference, native-repeat, and receipt hashes are recorded. |
| Infernal Invocation r11 NVFP4 or EXL3 release results | **No** | Correct base family, but different checkpoints and profiles. |
| Infernal Invocation r13 SQG KLD/throughput | **No** | r13 is donor-code lineage only and its qualified runtime is TP4/DCP1/MTP0 on another checkpoint. |
| Any v20 result or image | **No** | v20 is outside the final runtime and code lineage. |
| Pre-tpfix K96Tail mean KLD `8.607420359327334` | **No** | Rejected global/local TP-closure diagnostic. |
| TP-fixed diagnostic without sealed image identity | **No as a final claim** | Useful corroboration only; the sealed exact-r11 receipt is authoritative. |

No result may be copied from the r11, r13, or v20 release pages into the
K96Tail result tables. Required measurements must be rerun and sealed against
this page's checkpoint and derived-image identities.

## Checkpoint profile

This is a coupled-Hadamard re-encode of the frozen GLM-5.2 SQG checkpoint. It
does not download or reconstruct the original BF16 model for the encode. The
saved BMM-law calibration/Hessian dataset is
`brandonmusic/GLM-5.2-BMM-Law-SQG-Hessians` revision
`a05b3b92d749f6a641af5cfd52de2b4720380dfd`.

The coupled transform closes the GLM routed-expert function as follows:

1. normalized block-Hadamard H512 at the residual boundary;
2. H128 over the coupled gate/up preactivation coordinates;
3. exact `silu(gate) * up` activation;
4. H128 at the postactivation/down-projection boundary;
5. expert-static draw 0 or 6 signs, with local H13 alpha 0.25;
6. candidate-conditioned downstream H2 selection.

The stored bitrate profile is mixed by layer and tensor:

| Tensor class | Stored profile |
|---|---|
| Routed layer 3 | 720 K3 + 48 K4 matrices; 3.0625 bpw |
| Routed layers 4-77 | 672 K3 + 96 K4 matrices per layer; 3.125 bpw |
| MTP routed layer 78 | Preserved source payload: 384 K3 + 384 K4; 3.5 bpw |
| Eligible non-routed matrices | 380 K6 matrices |
| Remaining tensors | BF16 |

This is not a uniform-K96 or uniform-3.0625-bpw checkpoint. K96 applies to the
96 K4 assignments in routed layers 4-77; layer 3 is the sealed K48 exception,
and MTP78 remains unchanged.

The assembled local manifest is
`/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail/COUPLED_REENCODE_MANIFEST.json`,
SHA-256 `5b4309289c69dc618da03d34e10f826b8a6e7ba3fc67969582cf5b81b825a123`.
It records `complete=true`, all target routed layers 3-77 coupled, and
`mtp_layer_78_policy=preserve_source_unchanged`.

The quick codec census is
`/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/full_coupled_k96tail_no_shortcut_model_codec_quick.json`,
SHA-256 `a2d0648abc05e883e90d1c0ba0bbbb4464bbe8658fcd77ea5400849a8582bf14`.
It passes with 177,613 indexed tensors, 465 shards, 76 routed layers, 75
coupled layers, 380 K6 matrices, and all 768 MTP78 routed tensors preserved.

## Native runtime contract

B12X decodes serialized K3/K4 atoms directly into the routed W4A8 execution
path and executes non-routed K6 natively. The vLLM `exl3` name is the loader
interface for the checkpoint schema; it does not mean that routed weights are
served through an ExLlama GEMM fallback.

Every accepted TP rank must emit one complete
`glm52-native-sqg-w4a8-rank-evidence-v1` record with:

- routed layers 3-78 present in both `loaded_layers` and `executed_layers`
  (coupled targets 3-77 plus preserved source-SQG MTP78);
- `activation_endpoint=full-w4a8` and scope `routed_experts`;
- `allow_a16_fallback=false`;
- no BF16/dequantized routed-weight fallback;
- the expected K3, K4, and K6 endpoints.

The coupled TP4 runtime must also preserve the global transform geometry. Sign
vectors are generated over the global `2I` preactivation and `I`
postactivation extents, then sliced using TP-rank offsets. Gate/up
preactivations are all-gathered, restored from rank-major to global
projection-major order, and only then closed through H128 and
`silu(gate) * up`. The static image gate is
`/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/verify_runtime.py`,
SHA-256 `a0890ec9c92b82b1a441b7b823b9cdedfdced9aaee8ba9b7d405f9fc5607e9c0`.

This tpfix contract matters: the earlier TP-local implementation produced a
full-model mean KLD of `8.607420359327334`. That value is a rejected runtime
diagnostic, not checkpoint quality.

## Sealed full-vocabulary KLD

**Measured result: complete for the exact-r11 TP4/DCP1 quality geometry.** It is
not a TP4/DCP4/MTP3 serving result.

| Field | Value |
|---|---:|
| Direction | `KL(BF16 reference || K96Tail candidate)` |
| Token input | fixed 2,048-token GLM sequence |
| Token-sequence SHA-256 | `d0be87a4909ad00c311e36fc5c81c7d6942216c16a8fbbbf77778edb273346f1` |
| Scored positions | 2,047 |
| Vocabulary | 154,880, full distribution |
| Geometry | TP4 / PP1 / DCP1; MTP excluded |
| Candidate KV | FP8 |
| Image ID | `sha256:79dfbf6e697a1081016a3256ff5c96b1c2d6dcddd6f04662c9c693881310fa87` |
| Mean KLD | **0.1401771516114036** |
| Median KLD | 0.0015440876595675945 |
| p95 | 0.6778242588043213 |
| p99 | 2.480538845062256 |
| Worst-1% CVaR | 4.160942645300002 over 21 positions |
| Maximum | 8.928885459899902 |
| Nonfinite positions | 0 |
| Trim fraction | 0.0 |

The sealed receipt is
`/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/coupled-k96tail-full-exact-ii-r11-tpfix/kld/kld_exact_ii_r11_tp4dcp1.json`,
SHA-256 `7979c9c8b0c81714cd38e225646e42a88b2cb8eb03232be271373255c506a408`.
It includes all 2,047 per-position values, the reference manifest/logits hashes,
runtime topology, kernel source hashes, image identity, and exact r11 vLLM
version. It does not contain a p99.9 field; do not invent one.

### Honest source comparison

The frozen source-control checkpoint measured mean KLD
`0.07583317451217256` with the same fixed token/reference artifacts and
2,047-position TP4/DCP1 harness. Its receipt used the older r3 source-control
runtime image, not the exact-r11 tpfix image, so this is an informative source
comparison rather than a same-image A/B. Its receipt is
`/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/coupled-tail-source-control-tp4dcp1-routes-v2/kld/kld_sm120_tp4dcp1.json`,
SHA-256 `5d8aedb462658c693f1ce790f48ce5ed3cd6876897b1367a5cd36e42c0e2d434`.

K96Tail is therefore `0.06434397709923104` higher in mean KLD, or 1.8485x the
source mean (84.85% higher). The tpfix result is simultaneously 61.40x below
the rejected `8.607420359327334` runtime diagnostic. The repair removed the
systemic TP closure error; it did not make K96Tail equal to or better than the
source checkpoint on mean KLD. KLD is one fixed-context distribution-fidelity
measurement and does not replace task or long-context evaluation.

## Hidden-state replay

**Measured result: complete for one fixed 2,048-token compatibility context.**
This is not the Kimi K3 1,024-context qualification suite and is not a
TP4/DCP4/MTP3 serving result.

The cross-check captures the final normalized hidden state immediately
before the unchanged BF16 LM head on the same 2,048 token IDs. GLM-5.2 hidden
width is 6,144, so the auditable raw BF16 capture is `[2048,6144]`, not the
Kimi-derived `[2048,7168]` geometry. Rows 0-2046 are scored through the shared
BF16 `lm_head.weight` of shape `[154880,6144]`; the terminal row is retained but
not counted as a next-token position.

| Field | Measured value |
|---|---:|
| Receipt schema | `glm52-hidden-replay-one-context-kld-v2` |
| `complete` / `qualification_pass` | `true` / `true` |
| Direction | `KL(BF16 reference || K96Tail candidate)` |
| Scored positions | 2,047 |
| Capture | `[2048,6144]` BF16; 2,048 raw rows |
| BF16 LM head | `[154880,6144]`; source/candidate byte-identical |
| Mean KLD | **0.1401762649458023** |
| Median / p95 / p99 KLD | `0.0015491123620071697` / `0.6775412922257588` / `2.4739678400154723` |
| Worst-1% CVaR / maximum KLD | `4.160923965131938` / `8.928887865112092` |
| Mean JS | `0.02756684929459105` |
| Top-1 agreement | **0.9174401563263312** |

The sealed receipt is
`/home/brandonmusic/KLC_SANDBOXES/glm52_sqg_w4a8_sm120_local_acceptance_20260812/RESULTS/coupled-k96tail-hidden-replay-exact-ii-r11/hidden-replay-kld.json`,
SHA-256 `34f14cada0424ddb1387fec78a96a16ebe2109ddf2c063862d52108d0450e6b2`.
The capture manifest is
`raw-20260815T075846Z/hidden/manifest.json`, SHA-256
`12412b014f730a89890f7b9416c6498976139d960498d4cdf60058df2f639cc1`.
The hidden-state file SHA-256 is
`50781f1192645e13899d11f2181717ea05e028688db0201ede6d7e88360ee874`.
The LM-head file and raw-tensor SHA-256 values are respectively
`f56e1020446f4eaf73b418dd62ef853eb51e4f5c502d09ec0301086a90557969`
and `a012be05e7716292407d418b408222de256d4dbe2fe2143a44d27d8e3553bfba`.

### v2 post-observation compatibility gate

These bounds were selected **after observing** the deterministic BF16 offline
LM-head reduction-order delta distribution. They are operational replay
compatibility bounds, not a preregistered scientific model-quality gate.

| Comparison | Limit | Observed | Pass |
|---|---:|---:|---:|
| Native-versus-replay mean absolute KLD delta | `<=5e-5` | `8.866656012740393e-7` | yes |
| Per-position absolute KLD delta, p99 | `<=1e-4` | `6.126456900251877e-5` | yes |
| Per-position absolute KLD delta, maximum | `<=5e-3` | `0.0035557552471569` | yes |

The independently repeated native full-logit receipt is
`raw-20260815T075846Z/capture-full-kld.json`, SHA-256
`a67e896ffc1f879cdb3320042c50981532ce7837e0375fe34bcb0729fe685fc7`.
Its mean delta and maximum per-position delta versus the baseline native receipt
are both `0.0`: the repeat is exact at every scored position. The replay mean is
`8.866656012740393e-7` below the native mean `0.1401771516114036` under this
one-context comparison. This validates the replay implementation under the
disclosed gate; it does not expand the evidence to task quality, long context,
or target serving topology.

## Candidate TP4/DCP4/MTP3 serving profile

**Status: locally qualified under the sealed receipt described below.** This
profile ran on the derived exact-r11 image above. Substituting the r13 SQG
image or any v20 image invalidates the qualification. The candidate profile is:

| Setting | Candidate value |
|---|---|
| Physical GPUs | four SM120 GPUs |
| Tensor/decode context parallelism | TP4 / DCP4 |
| Speculative decoding | MTP3, greedy draft selection |
| Scheduler | `MAX_NUM_SEQS=8`, `MAX_BATCHED_TOKENS=2048` |
| CUDA graph cap | 32 rows |
| Context limit | 262,144 tokens |
| GPU memory utilization | 0.96 |
| Routed experts | native B12X SQG W4A8 |
| Non-routed SQG | native B12X K6 |
| KV cache | NVFP4 DS-MLA |
| Loading | InstantTensor `BUFFERED`, `INSTANTTENSOR_COPY=0` |
| DCP | A2A, threshold 16, `ag_rs` large backend, global top-k |
| Draft sharding | `VLLM_DCP_SHARD_DRAFT=1` |
| Runtime attestation | `tp4dcp4mtp3` |

The local Compose source is
`/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/deploy/compose.yaml`,
SHA-256 `c31f89f090e7f046bbd347a3b28d0c4f4486ad9b8e862645c4fcae1caa8c9c58`.
The launch wrapper is
`/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/deploy/serve.sh`, SHA-256
`b5b491535b6d61bdac81c7a7888b2d446831d96748ad5ca1f87829992b4aa904`.

The wrapper fails closed unless the selected image has these exact labels:

```text
ai.verdict.infernal-invocation.base=r11
ai.verdict.target.topology=tp4-dcp4-mtp3
ai.verdict.coupled.tp4-preactivation=all-gather-reassembly-v1
```

Local qualification operators may launch the candidate only from the pinned
source bundle:

```bash
cd /home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime
sha256sum deploy/compose.yaml deploy/serve.sh
MODEL_DIR=/home/brandonmusic/models/GLM-5.2-SQG-Coupled-H512-H128-K96Tail \
JIT_CACHE=/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/cache \
RESULTS_DIR=/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/results \
./deploy/serve.sh
```

This is the pinned reproduction command; the local qualification claim comes
from the sealed receipt, not from the command itself.

The final `tpfix-mtpfix` candidate first reached a four-rank loader observation:
all four ranks reported the `288G` main checkpoint and `4.28G` MTP payload
loaded, with `80.28 GiB` model memory per rank. The later sealed four-rank
execution and task receipt is the qualification evidence; loader memory alone
is not promoted as proof.

## Serving and task acceptance

TP4/DCP4/MTP3 acceptance requires all of the following on the exact image and
checkpoint revision:

- engine logs proving TP4, DCP4, and MTP3, including shard-draft behavior;
- four complete native-SQG rank-evidence records;
- healthy startup and required graph capture;
- a DCP1-versus-DCP4 numerical discriminator capable of rejecting a
  finite-but-wrong DCP merge;
- deterministic smoke output and exact runtime/environment receipt;
- no CUDA, NCCL, Xid, OOM, worker, or fallback errors in the accepted window.

Five sequential Estonia requests at concurrency 1 and five LAVD requests at
concurrency 5 were then preserved with raw benchmark JSONs/logs,
image/container inspections, summary, and `SHA256SUMS`. The local runner is
`/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/deploy/run-quality-5x.sh`,
SHA-256 `54eaf5a7f4ef986e3b927a8624a6865db76466bacf92ecc690855b509d4a5784`.
The idempotent post-task sealer SHA-256 is
`74f8070f081bade7180b9a20d18611174c092362c5d43d03c79621dd9f1af118`.
It pins `local-inference-lab/llm-inference-bench` v0.4.29 commit
`0b4185b5b435e948b199c9077a00b084864aa963`, script SHA-256
`59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7`,
requires result metadata version `0.4.29`, and writes the benchmark provenance
into the sealed receipt.
The accepted bundle must include `server.log` and all four complete native-SQG
rank receipts; their absence is a hard failure.
Receipt discovery is bound to the current container's `.State.StartedAt` and
copies only rank-evidence files newer than that boundary. This prevents four
files from a previous server from satisfying the gate. The sealed receipt
selected exactly current rank PIDs `767`, `825`, `926`, and `1030`. All four
are complete, schema `glm52-native-sqg-w4a8-rank-evidence-v1`, TP4,
`full-w4a8`, routed-expert, no-A16-fallback receipts that loaded and executed
layers 3--78. The fatal-audit file contains zero matches.
The hidden-replay-to-quality waiter is
`/home/brandonmusic/KLC_SANDBOXES/ii-r11-k96-runtime/deploy/wait-and-run-quality-exact-r11.sh`,
SHA-256 `944952a43a520e265d5b381f71c0e448b3f883d5c60c50ec11009ced083cf88b`.

Current results:

| Gate | Result |
|---|---|
| Hidden-state replay | **COMPLETE locally; public link pending** |
| Four-rank main + MTP loader observation | **COMPLETE; corroborating only** |
| TP4/DCP4/MTP3 native-SQG serving receipt | **SEALED LOCALLY; public link pending** |
| Estonia | **SEALED LOCALLY: 5/5 valid and correct; public link pending** |
| LAVD | **SEALED LOCALLY: 5/5 valid and correct, 4 exact + 1 near; public link pending** |
| Estonia decode metrics | **COMPLETE locally** |
| Cold/uncached Estonia prefill | **NOT MEASURED** |

The first Estonia request was sent while the server used
`MAX_MODEL_LEN=131072` and was rejected HTTP 400 because the complete frozen
request envelope exceeded that ceiling. With the exact checkpoint tokenizer
and chat template, Estonia measures 133,186 input tokens and a 193,186-token
total ceiling with 60,000 output; LAVD measures 20,310 input tokens and a
100,310-token ceiling with 80,000 output. The existing 140,000-token DCP
full-CKV gather capacity covers Estonia prefill, so no CKV-capacity change or
reload is required. No valid generation or Estonia score was produced, so this
is a non-model request-envelope diagnostic, not a quality result. The corrected
default is `262144`.

The nominal runs ending `084104Z` and `084128Z` overlapped benchmark clients,
invalidating the intended concurrency-1 Estonia condition. They are rejected
diagnostics and contribute zero qualification runs. Only those two client units
were stopped; the exact-r11 server remained loaded and healthy and returned to
zero running/zero waiting requests. Sole clean supervised run `084427Z`
completed its v0.4.29 Estonia stage 5/5 valid and correct, pass/fail `5/0`, with
zero truncations. Its result JSON SHA-256 is
`62551513dabdfdb0c389edc15220e03d3d95d12e5f85e26125ca0ba8962b6e92`.
Aggregate generation was `42.67293371871177 tok/s`; mean per-request generation
was `43.5170233617816 tok/s`. The reported `148,287 tok/s` prefill scout reused
a prefix cache warmed by the rejected overlapping diagnostics and must not be
reported as cold or uncached Estonia prefill.

The same supervised run completed LAVD 5/5 valid and correct: four exact
answers of `72,46` and one near answer of `71,45.75` (count delta `-1`, hours
delta `-0.25`). It had zero failures and zero truncations. The raw JSON
SHA-256 is
`2a9e2772049be73e59d14a31f1f47b17dbcd871f536af6a7f09febc8ab29b4e1`.
Aggregate generation was `22.13984480814618 tok/s`; mean per-request
generation was `22.087665860284208 tok/s`; mean elapsed time was
`768.5865516199963 s`; and mean TTFT was `15.050551386599546 s`.

The combined quality summary SHA-256 is
`4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57`;
the accepted run-manifest SHA-256 is
`bf57161cec69b9e8e8ebb5e705c1c5c6ed88c80046364c2a07a8882e60d0e3a9`.
`qualification.complete` is present and all 16 manifest entries verify. The
reuse-capable finalizer records layers 3--78 and gathers MTP counters without
rerunning either task suite, correcting the earlier runner assertion that
stopped at 77 even though preserved source-SQG MTP78 had correctly loaded and
executed.

The MTP summary SHA-256 is
`d8a7f22f6da05423a00d972e186deb17ba9f36133aae692c2477d53cd4f0f4ff`.
Across 106,204 generated tokens, the runtime drafted 88,704 tokens and accepted
76,631: aggregate acceptance `0.863895652958153`; per-position acceptance
`0.9242762445887446`, `0.8623511904761905`, and `0.8050595238095238`.

## Sealed-receipt insertion points

The blocks below are intentionally explicit. Replace **PENDING** only after the
named artifact exists, passes its gates, and has a recorded SHA-256. Do not
insert estimates, log excerpts without receipts, proxy measurements, or results
from another image/checkpoint.

### Hidden-replay receipt

<!-- Insert only a sealed exact-r11 hidden-replay receipt here. -->

| Required field | Value |
|---|---|
| Status | **SEALED LOCALLY; PUBLIC LINK PENDING** |
| Receipt path | `RESULTS/coupled-k96tail-hidden-replay-exact-ii-r11/hidden-replay-kld.json` |
| Receipt SHA-256 | `34f14cada0424ddb1387fec78a96a16ebe2109ddf2c063862d52108d0450e6b2` |
| `complete` / `qualification_pass` | `true` / `true` |
| Scored positions | `2047` |
| Capture shape/dtype | `[2048,6144]` BF16 |
| Capture manifest SHA-256 | `12412b014f730a89890f7b9416c6498976139d960498d4cdf60058df2f639cc1` |
| Capture file SHA-256 | `50781f1192645e13899d11f2181717ea05e028688db0201ede6d7e88360ee874` |
| BF16 LM-head file/tensor SHA-256 | `f56e1020446f4eaf73b418dd62ef853eb51e4f5c502d09ec0301086a90557969` / `a012be05e7716292407d418b408222de256d4dbe2fe2143a44d27d8e3553bfba` |
| Mean replay KLD / JS / top-1 | `0.1401762649458023` / `0.02756684929459105` / `0.9174401563263312` |
| Paired delta versus native KLD | mean `8.866656012740393e-7`; p99 `6.126456900251877e-5`; max `0.0035557552471569` |
| Gate provenance | **v2 post-observation**, limits disclosed above |
| Repeated native receipt | `a67e896ffc1f879cdb3320042c50981532ce7837e0375fe34bcb0729fe685fc7`; exact per position |

### Exact TP4/DCP4/MTP3 startup and correctness receipt

<!-- Insert only evidence from verdictai/glm52-k96-ii-r11:20260815-tpfix-mtpfix or its exact immutable digest. -->

| Required field | Value |
|---|---|
| Status | **SEALED LOCALLY; IMMUTABLE PUBLIC URL PENDING** |
| Acceptance receipt path or immutable URL | `evidence/final-exact-ii-r11/qualification-5x/exact-r11-tp4dcp4mtp3-20260815T084427Z/quality-summary.json` |
| Receipt SHA-256 | `4f19ba5e4a8676c80bc49e89d346b0985faa209f14bdd6d9713e9ee6c4397f57` |
| Run-manifest SHA-256 | `bf57161cec69b9e8e8ebb5e705c1c5c6ed88c80046364c2a07a8882e60d0e3a9` |
| MTP summary SHA-256 | `d8a7f22f6da05423a00d972e186deb17ba9f36133aae692c2477d53cd4f0f4ff` |
| Checkpoint immutable revision | **PENDING** |
| Image digest | `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def` |
| Engine-attested topology | `TP4/DCP4/MTP3` |
| Native-SQG rank evidence | `4/4` complete; `full-w4a8`; no A16 fallback; loaded/executed 3--78 |
| Error/fallback audit | `0` fatal matches |
| Separate graph/discriminator/smoke claim | **Not asserted by this quality-summary receipt** |

### Estonia five-run receipt

<!-- Local result JSON complete; insert immutable public link after publication. -->

| Field | Value |
|---|---|
| Status | **SEALED LOCALLY (5/5); PUBLIC LINK PENDING** |
| Rejected envelope diagnostic | HTTP 400 at `MAX_MODEL_LEN=131072`; **not a quality run** |
| Corrected request ceiling | `MAX_MODEL_LEN=262144` |
| Benchmark provenance | v0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963`, script SHA-256 `59dd767c933e06f9724a84a8883d2aac156252dbbc279ce155658005d27424d7` |
| Local result path | `results/qualification-5x/exact-r11-tp4dcp4mtp3-20260815T084427Z/estonia-c1-r5.json` |
| Immutable public URL | **PENDING** |
| Result JSON SHA-256 | `62551513dabdfdb0c389edc15220e03d3d95d12e5f85e26125ca0ba8962b6e92` |
| Image/topology identity | `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def`; TP4/DCP4/MTP3 |
| Concurrency / requested runs | `1 / 5` |
| Run 1 | `PASS`, correct, 1,397 completion tokens, 30.974969591989066 s |
| Run 2 | `PASS`, correct, 2,783 completion tokens, 64.26823912700638 s |
| Run 3 | `PASS`, correct, 2,980 completion tokens, 69.5352602010098 s |
| Run 4 | `PASS`, correct, 5,947 completion tokens, 146.53665814099077 s |
| Run 5 | `PASS`, correct, 3,751 completion tokens, 88.25072755399742 s |
| Completion-token distribution | average `3371.6`; p50 `2980`; p90 `5068.6`; p99 `5859.16` |
| Generation throughput | aggregate `42.67293371871177 tok/s`; mean per request `43.5170233617816 tok/s` |
| Latency | mean elapsed `79.91317092299869 s`; mean TTFT `0.9025994309980888 s` |
| Prefill qualification | Reported scout `148,287 tok/s` is cache-warmed by rejected overlap diagnostics; **no cold/uncached prefill claim** |
| Aggregate interpretation | 5/5 valid and correct; zero failures and zero truncations |

### LAVD five-run receipt

<!-- Insert only the sealed LAVD JSON produced after target-topology acceptance. -->

| Field | Value |
|---|---|
| Status | **SEALED LOCALLY (5/5); PUBLIC LINK PENDING** |
| Receipt path or immutable URL | `results/qualification-5x/exact-r11-tp4dcp4mtp3-20260815T084427Z/lavd-c5-r5.json`; immutable public URL pending |
| Receipt SHA-256 | `2a9e2772049be73e59d14a31f1f47b17dbcd871f536af6a7f09febc8ab29b4e1` |
| Image/checkpoint/topology identity | `sha256:ab6bd60716b0a8e453b6345cb10e43e79726d92729b1f29058e31d7cc1c67def`; K96Tail; TP4/DCP4/MTP3 |
| Concurrency / requested runs | `5 / 5` |
| Run 1 | `EXACT`, correct, `72,46`, 11,984 completion tokens |
| Run 2 | `EXACT`, correct, `72,46`, 17,146 completion tokens |
| Run 3 | `EXACT`, correct, `72,46`, 18,463 completion tokens |
| Run 4 | `EXACT`, correct, `72,46`, 17,738 completion tokens |
| Run 5 | `NEAR`, correct, `71,45.75`, 18,080 completion tokens; deltas `-1`, `-0.25` |
| Completion-token distribution | average `16682.2`; p50 `17738`; p90 `18309.8`; p99 `18447.68` |
| Generation throughput | aggregate `22.13984480814618 tok/s`; mean per request `22.087665860284208 tok/s` |
| Latency | mean elapsed `768.5865516199963 s`; mean TTFT `15.050551386599546 s` |
| Dataset / prompt SHA-256 | `612f8041bbca048c044dd77ebd58964afded85b0d76513c013715a401b09dc34` / `5c83674d5f0fd2a727bf11c521a765f8be4a13087714a2151fc96078018c4aa0` |
| Aggregate interpretation | 5/5 valid and correct; 4 exact + 1 near; zero failures/truncations |

### Final public Hugging Face revision

<!-- Insert only after the public revision is immutable and anonymously verified. -->

| Field | Value |
|---|---|
| Repository | `brandonmusic/GLM-5.2-SQG-Coupled-H512-H128-K96Tail` |
| Attempted routed-upload helper | `deploy/accelerate-hf-routed-upload.sh`, SHA-256 `7fcf04c7762caf14b9beb9ea94ee0e4ae466ea02940b4139e69ca9a7460374f5` |
| Attempt result | First attempt about `24 Mb/s` versus original `20--24 Mb/s`; stopped at 04:56; targeted P8 first stage restarted at 05:03; no throughput gain or full completeness inferred |
| Canonical full-folder uploader | `deploy/complete-final-model-upload.sh`, SHA-256 `f6b3352db4a288793254bd0c2656e31c805e3718b23fe2239a6649fcd7646d46` |
| Persistent model-upload unit | `glm52-k96tail-final-model-upload.service`, SHA-256 `f73b3a35718ff43955c37f4a9f9425b9f117576b260fba684130ce6b8ed7e1e8`; enabled/active |
| Final release publisher | `deploy/wait-and-publish-hf-release.sh`, SHA-256 `13430d53b309d3cfaa5b68cf5a062b1982660e3a0eba86fdddd2762cd7a25110` |
| Persistent publication unit | `glm52-k96tail-final-hf-publication.service`, SHA-256 `ff033d7d3573bedf282b1535065aaf9f1a9d719cd53f0cd1fc89fb74be4eda62`; enabled/active |
| Publication gate | `results/hf-publication.ready` deliberately absent pending final local card/provenance/manifests review |
| Publication status | **PENDING** |
| Immutable revision | **PENDING** |
| Top-level `SHA256SUMS` | **PENDING** |
| Model-card KLD receipt link/hash | **PENDING** |
| Hidden-replay link/hash | **PENDING** |
| TP4/DCP4/MTP3 receipt link/hash | **PENDING** |
| Estonia/LAVD raw-result links/hashes | **PENDING** |
| Anonymous download verification | **PENDING** |

## Publication checklist

- [x] Assembled checkpoint manifest and codec census pass locally.
- [x] Exact-r11 TP4/DCP1 full-vocabulary KLD receipt is sealed.
- [x] Hidden-state capture/replay receipt and paired comparison are sealed
      locally under the disclosed post-observation v2 gate.
- [x] TP4/DCP4/MTP3 quality-summary and four native-SQG rank receipts are
      sealed locally; no separate graph/discriminator/smoke claim is made.
- [x] Estonia 1-5 local result JSON is complete and byte-hashed.
- [x] LAVD 1-5 and the combined quality receipt are sealed locally.
- [ ] Checkpoint, scripts, manifests, receipts, and regenerated `SHA256SUMS` are
      uploaded to the public Hugging Face repository.
- [ ] The public immutable Hugging Face revision is recorded here.
- [ ] The deployment artifacts and result receipts are committed to
      `local-inference-lab/rtx6kpro`.
- [ ] Public files are anonymously downloaded and reverified by hash.

## Qualification boundary

- **Measured:** assembled mixed-rate checkpoint census; exact-r11 tpfix
  TP4/DCP1 full-vocabulary KLD `0.1401771516114036` with FP8 KV; one-context
  hidden-replay KLD `0.1401762649458023` and top-1 agreement
  `0.9174401563263312` under the disclosed post-observation v2 gate; exact-r11
  TP4/DCP4/MTP3 four-rank native full-W4A8 execution through layers 3--78;
  Estonia correctness 5/5; and LAVD correctness 5/5 with four exact and one
  near result, all with the metrics above.
- **Implemented/configured:** native SQG K3/K4/K6 execution, coupled H512/H128
  TP closure, supplemental MTP3 loader fix, label-gated r11 image, candidate
  TP4/DCP4/MTP3 Compose, and four-rank loaded-state observation.
- **Pending/not claimed:** immutable public model and receipt revisions,
  anonymous public verification, public experiments commit, cold/uncached
  Estonia prefill, and any separate graph/discriminator/smoke assertion not
  represented by the sealed quality summary.
- **Not used:** Infernal Invocation r13 as a runtime base and every v20 image.

Any published result must name the checkpoint revision, image digest,
TP/DCP/MTP, KV dtype, graph cap, token/reference hashes, physical GPU order,
PCIe topology, raw receipt path, and receipt SHA-256. A result from another
checkpoint, KV format, DCP mode, MTP count, or image is not interchangeable.
