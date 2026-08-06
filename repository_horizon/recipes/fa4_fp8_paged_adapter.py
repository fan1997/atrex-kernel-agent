from __future__ import annotations

import torch
import torch.nn as nn
from flash_attn.cute import flash_attn_func


class Model(nn.Module):
    """Auditable official-FA4 dense fallback for paged FP8 KV."""

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
        del cu_seqlens_k, paged_kv_indptr, paged_kv_indices
        del paged_kv_last_page_len
        page_size = key_cache.shape[1]
        if page_size not in (64, 128):
            raise ValueError(
                f"FA4 repository recipe requires page_size 64 or 128, got {page_size}"
            )
        head_dim = q.shape[-1]
        num_kv_heads = key_cache.shape[2]
        query_bounds = [int(value) for value in cu_seqlens_q.tolist()]
        outputs = []
        for request in range(len(query_bounds) - 1):
            query_start = query_bounds[request]
            query_end = query_bounds[request + 1]
            kv_length = int(seq_lens[request].item())
            num_pages = (kv_length + page_size - 1) // page_size
            page_ids = block_table[request, :num_pages].long()
            key = key_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[:kv_length]
            value = value_cache[page_ids].reshape(-1, num_kv_heads, head_dim)[
                :kv_length
            ]
            output = flash_attn_func(
                q[query_start:query_end].unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                softmax_scale=head_dim**-0.5,
                causal=True,
            )
            if isinstance(output, (tuple, list)):
                output = output[0]
            outputs.append(output.squeeze(0).to(torch.bfloat16))
        return torch.cat(outputs, dim=0)
