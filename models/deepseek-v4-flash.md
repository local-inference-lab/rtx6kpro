# DeepSeek-V4-Flash Runbook Hub

Use this page as the stable entry point for DeepSeek-V4-Flash and DSpark on RTX
PRO 6000 Blackwell. Release pages are immutable measurement and deployment
specifications.

## Recommended Deployment

| Need | Page |
|---|---|
| Serve the 0731 DSpark checkpoint | [DeepSeek-V4-Flash-0731 Infernal Invocation r10](ds4dspark-infernal-invocation-r10.md) |
| Inspect the Gilded Gnosis baseline | [DeepSeek-V4-Flash-0731 Gilded Gnosis r33](ds4dspark-v20-r33.md) |
| Inspect the Fathomless TP2/TP4 sweep | [DeepSeek-V4-Flash v10 Fathomless Validation](ds4dspark-v10.md) |
| Inspect the full DSpark and standard-MTP sweep | [DeepSeek-V4-Flash and DSpark v9](ds4dspark-v9.md) |
| Diagnose empty reasoning before tool calls | [DS4 empty-think troubleshooting](ds4f-empty-think/README.md) |

## Serving Contracts

| Area | Specification |
|---|---|
| Recommended image line | Infernal Invocation r10 for `deepseek-ai/DeepSeek-V4-Flash-0731` |
| DSpark checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Standard-MTP checkpoint | `deepseek-ai/DeepSeek-V4-Flash` |
| Archived DSpark checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| General-purpose DSpark profile | fixed probabilistic K5, InstantTensor BUFFERED, B12X W4A8, FP8 compressed MLA KV |
| KV offload | Native vLLM CPU/filesystem tiers or LMCache L1/filesystem L2; use one ownership model per server |
| Backend family | B12X; archived pages also describe SparkInfer, Lucifer, and CUTLASS implementations |
| Speculative decoding | Standard MTP and DSpark use different checkpoint, graph, and verifier contracts |

## Release Namespace Map

| Source line | Revision namespace | Serving specification |
|---|---|---|
| `dev/infernal-invocation` | Infernal Invocation `r*` | [Infernal Invocation r10](ds4dspark-infernal-invocation-r10.md) |
| `dev/gilded-gnosis` | Gilded Gnosis `v20-r*` | [Gilded Gnosis r33](ds4dspark-v20-r33.md) |
| Fathomless Firmament | `v9` and `v10` | [v10](ds4dspark-v10.md), [v9](ds4dspark-v9.md) |
| Eldritch Enlightenment | DS4 Flash `v1-v6` | [v6](ds4-flash-v6.md), [v5](ds4-flash-v5.md), [v4](ds4-flash-v4.md), [v3](ds4-flash-v3.md), [v2](ds4-flash-v2.md), [v1](ds4-flash-v1.md) |

Gilded Gnosis pages remain historical specifications. Infernal Invocation
revision numbers do not continue the Gilded Gnosis `v20-r*` sequence because
the source branches have different identities and merge contracts.

## Operational Invariants

- Standard MTP and DSpark are not interchangeable.
- Confirm model revision, backend markers, graph coverage, and source-tree
  labels before comparing performance.
- Keep `NCCL_GRAPH_FILE` unset unless it names an existing NCCL XML topology
  file.
- Reuse release-scoped JIT caches. CuTe and FlashInfer compilation can dominate
  the first startup for an uncovered shape.
