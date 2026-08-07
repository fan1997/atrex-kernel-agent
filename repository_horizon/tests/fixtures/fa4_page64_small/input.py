import torch

FP8_DTYPE = torch.float8_e4m3fn


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _fp8_random(shape: tuple[int, ...]) -> torch.Tensor:
    values = torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * 0.1
    return values.clamp(-448.0, 448.0).to(FP8_DTYPE)


def _make_inputs(
    q, query_start_loc, seq_lens, num_kv_heads, page_size, kv_layout, role, stage
):
    del role, stage
    query_start_loc = _ints(query_start_loc)
    seq_lens = _ints(seq_lens)
    query_tokens, num_query_heads, head_dim = q
    if page_size != 64 or kv_layout != "NHD":
        raise ValueError("fixture requires NHD page_size=64")
    pages_per_request = [(value + page_size - 1) // page_size for value in seq_lens]
    total_pages = sum(pages_per_request)
    key_cache = torch.zeros(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=FP8_DTYPE,
        device="cuda",
    )
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.full(
        (len(seq_lens), max(pages_per_request)),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    page_indptr = [0]
    cursor = 0
    for request, (seq_len, pages) in enumerate(
        zip(seq_lens, pages_per_request, strict=True)
    ):
        ids = torch.arange(cursor, cursor + pages, dtype=torch.int32, device="cuda")
        block_table[request, :pages] = ids
        key_cache[cursor : cursor + pages].view(-1, num_kv_heads, head_dim)[
            :seq_len
        ].copy_(_fp8_random((seq_len, num_kv_heads, head_dim)))
        value_cache[cursor : cursor + pages].view(-1, num_kv_heads, head_dim)[
            :seq_len
        ].copy_(_fp8_random((seq_len, num_kv_heads, head_dim)))
        cursor += pages
        page_indptr.append(cursor)
    kv_cumulative = [0]
    for seq_len in seq_lens:
        kv_cumulative.append(kv_cumulative[-1] + seq_len)
    return {
        "q": _fp8_random((query_tokens, num_query_heads, head_dim)),
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cu_seqlens_q": torch.tensor(query_start_loc, dtype=torch.int32, device="cuda"),
        "cu_seqlens_k": torch.tensor(kv_cumulative, dtype=torch.int32, device="cuda"),
        "paged_kv_indptr": torch.tensor(page_indptr, dtype=torch.int32, device="cuda"),
        "paged_kv_indices": torch.arange(total_pages, dtype=torch.int32, device="cuda"),
        "paged_kv_last_page_len": torch.tensor(
            [value % page_size or page_size for value in seq_lens],
            dtype=torch.int32,
            device="cuda",
        ),
        "block_table": block_table,
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
    }
