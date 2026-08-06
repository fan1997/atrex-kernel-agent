import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(
        self,
        q,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        block_table,
        seq_lens,
    ):
        del cu_seqlens_k, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len
        num_query_heads = q.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = q.shape[2]
        repeat = num_query_heads // num_kv_heads
        output = torch.empty_like(q, dtype=torch.bfloat16)
        bounds = [int(value) for value in cu_seqlens_q.tolist()]
        for request in range(len(bounds) - 1):
            q_start, q_end = bounds[request : request + 2]
            q_len = q_end - q_start
            kv_len = int(seq_lens[request])
            pages = (kv_len + key_cache.shape[1] - 1) // key_cache.shape[1]
            ids = block_table[request, :pages].long()
            key = key_cache[ids].reshape(-1, num_kv_heads, head_dim)[:kv_len]
            value = value_cache[ids].reshape(-1, num_kv_heads, head_dim)[:kv_len]
            query = q[q_start:q_end].transpose(0, 1).float()
            key = key.repeat_interleave(repeat, dim=1).transpose(0, 1).float()
            value = value.repeat_interleave(repeat, dim=1).transpose(0, 1).float()
            scores = query @ key.transpose(-1, -2) * (head_dim**-0.5)
            query_positions = torch.arange(q_len, device=q.device) + kv_len - q_len
            key_positions = torch.arange(kv_len, device=q.device)
            mask = key_positions[None, :] <= query_positions[:, None]
            probability = torch.softmax(
                scores.masked_fill(~mask.unsqueeze(0), float("-inf")), dim=-1
            )
            output[q_start:q_end] = (
                (probability @ value).transpose(0, 1).to(torch.bfloat16)
            )
        return output
