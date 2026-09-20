# Qwen request-boundary cache capacity

Status: **qualified accounting reconciliation** for the recorded TP1/MTP3
configuration. The smaller reported token capacity does not mean that the
eight-GiB GPU cache allocation shrank. It includes blocks reserved for safe
recurrent checkpoint ownership that the R35 capacity report omitted.

## Conditions and measurement

Checkpoint: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, revision
`b797d2e1160b9596b2570e56c1d3590faa09d4ed`. TP1, MTP3, FP8 attention cache,
262,144-token context, 6,019-token prefill budget and
`--kv-cache-memory-bytes 8589934592`. Both comparison arms use native
request-boundary checkpoints.

The matrix records 517,581 logical tokens in R35 and 434,258 in the serving
composition containing vLLM `cfdecfaa4739ee3141c9745997cd8a83f31745a4`.
The preceding integration with endpoint-reserve accounting reports 448,705.
These are capacity estimates for the shared pool, not a per-request context
limit and not a count of physical blocks multiplied by page width.

## Reconciliation

The 48 target layers contain 12 full-attention and 36 recurrent GDN layers.
One MTP full-attention layer and one PLE recurrent state produce 13 attention
groups and 37 recurrent groups. A physical page spans 3,008 tokens. It stores
1,024 bytes per token of FP8 K/V and 64 bytes per token of BF16 compressed
selector keys: **3,272,704 bytes**. The eight-GiB budget fits **2,624 pages**, or
8,587,575,296 bytes. This physical-pool calculation is unchanged across the
three reported capacities.

A maximum-length request needs `13 × ceil(262144 / 3008) + 37 × 5 = 1329`
live blocks. Each private endpoint or pinned restore adds a bundle of
`50 groups + 1 auxiliary block = 51` blocks.

| Capacity accounting | Additional bundles | Blocks per maximum-length request | Reported logical tokens |
|---|---:|---:|---:|
| R35 report omitting boundary reserves | 0 | 1,329 | 517,581 |
| Three endpoints and one pinned restore | 4 | 1,533 | 448,705 |
| Four endpoints and one pinned restore | 5 | 1,584 | 434,258 |

Every token result is exactly `floor(262144 × 2624 / blocks_per_request)`.
The fourth private endpoint retains a checkpoint before a prefill tail, so a
slightly extended long prompt can reuse its common prefix. It adds 51 blocks
per maximum-length request and reduces the 448,705 estimate by 3.22%.
Removing that reserve would overstate supported concurrency and undermine the
checkpoint ownership bound; it would not recover an unrelated leaked buffer.

## Source and validation

- [Per-request endpoint and restore reservation](https://github.com/local-inference-lab/vllm/blob/cfdecfaa4739ee3141c9745997cd8a83f31745a4/vllm/v1/core/kv_cache_utils.py)
  is included in `get_max_concurrency_for_kv_cache_config`.
- [Recurrent page requirements](https://github.com/local-inference-lab/vllm/blob/cfdecfaa4739ee3141c9745997cd8a83f31745a4/vllm/v1/kv_cache_interface.py)
  account for live, speculative and prefill-checkpoint state.
- [QSA main-page and selector-tail geometry](https://github.com/local-inference-lab/vllm/blob/cfdecfaa4739ee3141c9745997cd8a83f31745a4/vllm/models/qwen4_exp/common/b12x_qsa_cache.py)
  preserves the BF16 selector tail alongside FP8 K/V.
- The calculation below independently reconstructs all three integers;
  both endpoint values match the [exported serving matrix](karmic-kraken-serving-samples.json).

This is a source/log accounting check, not a GPU allocator trace or a promise
that arbitrary request mixtures fit the maximum-concurrency estimate. It does
not justify changing model context, page geometry or checkpoint ownership.

```python
page_bytes = 3008 * (2 * 2 * 256 + 128 * 2 // 4)
pool_blocks = (8 * 1024**3) // page_bytes
live_blocks = 13 * ((262144 + 3007) // 3008) + 37 * 5
for bundles in (0, 4, 5):
    print(262144 * pool_blocks // (live_blocks + bundles * 51))
# 517581, 448705, 434258
```
