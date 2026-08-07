from __future__ import annotations

import torch
import torch.nn as nn
from flash_attn.cute import flash_attn_varlen_func


class Model(nn.Module):
    """Fixed native-paged ABI used to bring up and then optimize FA4."""

    def forward(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        del (
            cu_seqlens_k,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )
        page_size = key_cache.shape[1]
        if page_size not in (64, 128):
            raise ValueError(
                f"FA4 native-paged recipe requires page size 64 or 128, got {page_size}"
            )

        head_dim = q.shape[-1]
        output = flash_attn_varlen_func(
            q,
            key_cache,
            value_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=block_table.shape[1] * page_size,
            seqused_k=seq_lens,
            page_table=block_table,
            softmax_scale=head_dim**-0.5,
            causal=True,
            num_splits=1,
            pack_gqa=True,
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.to(torch.bfloat16)
