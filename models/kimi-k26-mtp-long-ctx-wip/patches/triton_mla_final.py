# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)

SM120_FP8_SINGLE_REQ_MAX_KV_SPLITS = 128
# With UNIFORM_BATCH CUDA graph support we cannot vary num_kv_splits between
# captures (it is a Triton `tl.constexpr` in the stage-1 kernel, and changing
# it both recompiles the kernel and resizes the stage-2 merge temporary). Pick
# a single value that is optimal at long context and acceptable at short — 64
# is the measured sweet spot at 30k ctx on sm120 fp8 (microbench in the WIP
# report). 128 is slightly faster at 50k+ but the attn_logits temporary doubles
# in size and we can't afford 1 GB per layer * 61 layers.
CG_NUM_KV_SPLITS = 64

# Persistent CUDA-graph-safe buffers SHARED across all TritonMLAImpl layer
# instances. Layers run sequentially in a forward pass so one buffer is enough
# per (device, shape, dtype) signature. Keyed so a mixed-dtype / multi-device
# deployment still works.  Size at 30k ctx with spec=3 on sm120 TP=8:
#   attn_logits: (512, 8, 64, 513) float32 = 538 MB  (vs 61× that per-layer)
#   o:           (512, 8, 512)    bf16    = 4 MB
#   lse:         (512, 8)         bf16    = 8 KB
_SHARED_CG_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_shared_cg_buffer(
    key_prefix: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (key_prefix, device, shape, dtype)
    buf = _SHARED_CG_BUFFERS.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _SHARED_CG_BUFFERS[key] = buf
    return buf


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    # FULL cudagraphs are safe to capture per-batch-size as long as every
    # replay uses the same num_kv_splits / output buffer addresses. We enforce
    # both in TritonMLAImpl.forward_mqa (fixed CG_NUM_KV_SPLITS + persistent
    # buffers) and here (per-query block_table / seq_lens written in-place into
    # persistent cg_buf_*).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # UNIFORM = all requests in the batch have the same query_len. Combined
    # with _init_reorder_batch_threshold(supports_spec_as_decode=True), this
    # makes spec-verify batches (query_len = 1 + num_spec_tokens per request)
    # route through the decode path instead of the expensive prefill branch.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Persistent CUDA-graph-safe buffers for the spec-verify case where the
        # Triton decode kernel needs per-query block_table rows and seq_lens.
        # These are populated in _build_decode via `.copy_()` so the addresses
        # seen by a captured graph stay stable across replays.
        #
        # Size: max_num_seqs * reorder_batch_threshold covers the largest
        # captured cudagraph shape. reorder_batch_threshold was just bumped by
        # `_init_reorder_batch_threshold(supports_spec_as_decode=True)` to
        # 1 + num_spec_tokens, so this is (max_num_seqs * (1+num_spec)).
        max_num_tokens = (
            self.vllm_config.scheduler_config.max_num_seqs
            * max(1, int(self.reorder_batch_threshold))
        )
        block_size = self.kv_cache_spec.block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        max_blocks_per_req = (max_model_len + block_size - 1) // block_size

        self._cg_buf_block_table: torch.Tensor | None = None
        self._cg_buf_seq_lens: torch.Tensor | None = None
        self._cg_max_num_tokens = max_num_tokens
        self._cg_max_blocks_per_req = max_blocks_per_req
        self._cg_enabled = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

    def _maybe_lazy_init_cg_bufs(
        self, device: torch.device, block_table_dtype: torch.dtype,
        seq_lens_dtype: torch.dtype,
    ) -> None:
        if self._cg_buf_block_table is None:
            self._cg_buf_block_table = torch.empty(
                (self._cg_max_num_tokens, self._cg_max_blocks_per_req),
                dtype=block_table_dtype,
                device=device,
            )
            self._cg_buf_seq_lens = torch.empty(
                (self._cg_max_num_tokens,),
                dtype=seq_lens_dtype,
                device=device,
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> MLACommonDecodeMetadata:
        num_reqs = seq_lens_device.shape[0]
        # For pure decode (num_decode_tokens == num_reqs, query_len==1 per req)
        # block_table already has the right shape and seq_lens is one per query.
        # The interesting case is spec-verify (num_decode_tokens == num_reqs * qpr
        # with qpr > 1): the Triton decode kernel indexes block_table with
        # `cur_batch` (0..num_decode_tokens-1) treating each query as an
        # independent "request", and loads seq_lens at the same index. So we
        # expand block_table by repeating each row `qpr` times and derive per-
        # query seq_lens from the post-step seq_lens_device.
        if num_decode_tokens > num_reqs and num_reqs > 0:
            qpr = num_decode_tokens // num_reqs
            # num_decode_tokens must be exactly num_reqs * qpr under UNIFORM
            # query_len_support. Assert to catch builder misconfiguration
            # early instead of silently indexing OOB in the kernel.
            assert num_decode_tokens == num_reqs * qpr, (
                "TritonMLA decode-path expects uniform query_len per request; "
                f"got num_decode_tokens={num_decode_tokens}, num_reqs={num_reqs}"
            )

            self._maybe_lazy_init_cg_bufs(
                device=block_table_tensor.device,
                block_table_dtype=block_table_tensor.dtype,
                seq_lens_dtype=seq_lens_device.dtype,
            )

            # Widen block_table_tensor in case it has fewer columns than our
            # persistent buffer (happens if this step only has small prefixes).
            bt_rows = num_decode_tokens
            bt_cols_src = block_table_tensor.shape[1]
            assert bt_rows <= self._cg_buf_block_table.shape[0], (
                f"spec-verify num_decode_tokens={bt_rows} exceeds CG buffer "
                f"capacity {self._cg_buf_block_table.shape[0]}"
            )
            assert bt_cols_src <= self._cg_buf_block_table.shape[1], (
                f"block_table has {bt_cols_src} columns > CG buffer "
                f"{self._cg_buf_block_table.shape[1]}"
            )

            # Expand in-place into CG-stable buffer: row i*qpr+j -> request
            # i's block_table row (same for every j in 0..qpr-1).
            for j in range(qpr):
                # Each strided slice is contiguous along dim 0 with stride qpr
                self._cg_buf_block_table[j:bt_rows:qpr, :bt_cols_src].copy_(
                    block_table_tensor
                )
            # Any columns beyond bt_cols_src are unused by the kernel (it
            # bounds accesses with seq_lens) but zero them for safety.
            if bt_cols_src < self._cg_buf_block_table.shape[1]:
                self._cg_buf_block_table[:bt_rows, bt_cols_src:].zero_()

            # Per-query seq_lens: for request i, queries at positions
            # p, p+1, ..., p+qpr-1 need seq_lens ctx+1, ctx+2, ..., ctx+qpr.
            # attn_metadata.seq_lens already stores the post-step per-request
            # length = ctx+qpr (assuming all spec tokens are going to be
            # written before attention runs, which is the case with the fused
            # unified_mla_kv_cache_update splitting op). So per-query:
            # seq_lens_per_query[i*qpr + j] = seq_lens_device[i] - (qpr-1) + j
            _arange = torch.arange(
                qpr,
                device=seq_lens_device.device,
                dtype=seq_lens_device.dtype,
            )
            expanded = (
                seq_lens_device.unsqueeze(1) - (qpr - 1) + _arange
            ).reshape(-1)
            self._cg_buf_seq_lens[:bt_rows].copy_(expanded)

            return MLACommonDecodeMetadata(
                block_table=self._cg_buf_block_table[:bt_rows],
                seq_lens=self._cg_buf_seq_lens[:bt_rows],
                dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            )

        # Fast path: query_len==1 per request, no expansion needed.
        return MLACommonDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()

        # CG-safe persistent buffers are pulled from the SHARED module-level
        # pool on first forward_mqa call, so all 61 target layers + 1 draft
        # layer reuse the same attn_logits / o / lse storage instead of each
        # owning a 538 MB buffer.
        vllm_cfg = get_current_vllm_config_or_none()
        if vllm_cfg is not None:
            scheduler_cfg = vllm_cfg.scheduler_config
            # max_num_seqs * (1 + num_spec_tokens) covers spec verify.
            spec_cfg = vllm_cfg.speculative_config
            qpr_max = 1 + (spec_cfg.num_speculative_tokens if spec_cfg is not None
                           and spec_cfg.num_speculative_tokens is not None
                           else 0)
            self._cg_max_tokens: int = scheduler_cfg.max_num_seqs * qpr_max
            # Also honour the actual cudagraph_capture_sizes max if compilation
            # config is available (the scheduler max can exceed what CG
            # actually captures, see max_cudagraph_capture_size).
            try:
                cg_max = vllm_cfg.compilation_config.max_cudagraph_capture_size
                if cg_max is not None:
                    self._cg_max_tokens = min(self._cg_max_tokens, cg_max)
            except AttributeError:
                pass
        else:
            # conservative fallback; should not happen in a normal serve path
            self._cg_max_tokens = 512

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        return super()._flash_attn_varlen_diff_headdims(
            q,
            k,
            v,
            return_softmax_lse=return_softmax_lse,
            softmax_scale=softmax_scale,
            **kwargs,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]

        # Fixed num_kv_splits — see CG_NUM_KV_SPLITS docstring. For
        # VLLM_BATCH_INVARIANT mode, still use 1 for determinism.
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            num_kv_splits = CG_NUM_KV_SPLITS

        # Pull CG-safe persistent buffers from the shared pool (one set per
        # (device, dtype, shape) tuple across ALL layers).
        assert B <= self._cg_max_tokens, (
            f"forward_mqa: B={B} exceeds CG capture max {self._cg_max_tokens}"
        )
        o_buf = _get_shared_cg_buffer(
            "o",
            (self._cg_max_tokens, q_num_heads, self.kv_lora_rank),
            q.dtype,
            q.device,
        )
        lse_buf = _get_shared_cg_buffer(
            "lse",
            (self._cg_max_tokens, q_num_heads),
            q.dtype,
            q.device,
        )
        attn_logits_buf = _get_shared_cg_buffer(
            "attn_logits",
            (
                self._cg_max_tokens,
                q_num_heads,
                num_kv_splits,
                # +1 stores the LSE that stage2 uses to merge partial outs
                self.kv_lora_rank + 1,
            ),
            torch.float32,
            q.device,
        )
        o = o_buf[:B]
        lse = lse_buf[:B]
        attn_logits = attn_logits_buf[:B]

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # block_table and seq_lens are already per-query for spec verify
        # (expanded by TritonMLAMetadataBuilder._build_decode into CG-safe
        # buffers) or per-request for pure decode.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse
