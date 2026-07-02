from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - local environment may not have Triton installed
    triton = None
    tl = None


def _attention_and_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        scores = torch.where(mask, scores, torch.full_like(scores, -1e6))

    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v)
    lse = torch.logsumexp(scores, dim=-1)
    return output, lse


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("TritonFlashAttentionFunction requires the `triton` package to be installed.")


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        output, lse = _attention_and_lse(q, k, v, is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, lse)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, _lse = ctx.saved_tensors

        with torch.enable_grad():
            q_recomputed = q.detach().requires_grad_(True)
            k_recomputed = k.detach().requires_grad_(True)
            v_recomputed = v.detach().requires_grad_(True)
            output, _ = _attention_and_lse(q_recomputed, k_recomputed, v_recomputed, ctx.is_causal)

        dq, dk, dv = torch.autograd.grad(
            outputs=output,
            inputs=(q_recomputed, k_recomputed, v_recomputed),
            grad_outputs=grad_output,
        )
        return dq, dk, dv, None


if triton is not None:

    @triton.jit
    def _flash_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        stride_q_batch,
        stride_q_seq,
        stride_q_dim,
        stride_k_batch,
        stride_k_seq,
        stride_k_dim,
        stride_v_batch,
        stride_v_seq,
        stride_v_dim,
        stride_o_batch,
        stride_o_seq,
        stride_o_dim,
        stride_lse_batch,
        stride_lse_seq,
        n_queries,
        n_keys,
        d_model,
        scale,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        query_block_idx = tl.program_id(1)

        query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        key_offsets = tl.arange(0, BLOCK_N)

        q_mask = (query_offsets[:, None] < n_queries) & (dim_offsets[None, :] < d_model)
        q_ptrs = (
            q_ptr
            + batch_idx * stride_q_batch
            + query_offsets[:, None] * stride_q_seq
            + dim_offsets[None, :] * stride_q_dim
        )
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        max_scores = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for key_start in range(0, n_keys, BLOCK_N):
            current_key_offsets = key_start + key_offsets
            k_mask = (current_key_offsets[:, None] < n_keys) & (dim_offsets[None, :] < d_model)
            k_ptrs = (
                k_ptr
                + batch_idx * stride_k_batch
                + current_key_offsets[:, None] * stride_k_seq
                + dim_offsets[None, :] * stride_k_dim
            )
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                causal_mask = query_offsets[:, None] >= current_key_offsets[None, :]
                scores = tl.where(causal_mask, scores, float("-inf"))

            scores = tl.where(query_offsets[:, None] < n_queries, scores, float("-inf"))
            current_max = tl.max(scores, axis=1)
            new_max = tl.maximum(max_scores, current_max)

            exp_scale = tl.exp(max_scores - new_max)
            probs = tl.exp(scores - new_max[:, None])
            sum_exp = sum_exp * exp_scale + tl.sum(probs, axis=1)
            acc = acc * exp_scale[:, None]

            v_mask = (current_key_offsets[:, None] < n_keys) & (dim_offsets[None, :] < d_model)
            v_ptrs = (
                v_ptr
                + batch_idx * stride_v_batch
                + current_key_offsets[:, None] * stride_v_seq
                + dim_offsets[None, :] * stride_v_dim
            )
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            acc += tl.dot(probs.to(v.dtype), v)
            max_scores = new_max

        output = acc / sum_exp[:, None]
        lse = max_scores + tl.log(sum_exp)

        out_mask = (query_offsets[:, None] < n_queries) & (dim_offsets[None, :] < d_model)
        out_ptrs = (
            o_ptr
            + batch_idx * stride_o_batch
            + query_offsets[:, None] * stride_o_seq
            + dim_offsets[None, :] * stride_o_dim
        )
        tl.store(out_ptrs, output.to(tl.float32), mask=out_mask)

        lse_ptrs = lse_ptr + batch_idx * stride_lse_batch + query_offsets * stride_lse_seq
        tl.store(lse_ptrs, lse, mask=query_offsets < n_queries)

    @triton.jit
    def _flash_attention_backward_d_kernel(
        o_ptr,
        do_ptr,
        d_ptr,
        stride_o_batch,
        stride_o_seq,
        stride_o_dim,
        stride_do_batch,
        stride_do_seq,
        stride_do_dim,
        stride_d_batch,
        stride_d_seq,
        n_queries,
        d_model,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        query_block_idx = tl.program_id(1)

        query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)

        mask = (query_offsets[:, None] < n_queries) & (dim_offsets[None, :] < d_model)
        o_ptrs = (
            o_ptr
            + batch_idx * stride_o_batch
            + query_offsets[:, None] * stride_o_seq
            + dim_offsets[None, :] * stride_o_dim
        )
        do_ptrs = (
            do_ptr
            + batch_idx * stride_do_batch
            + query_offsets[:, None] * stride_do_seq
            + dim_offsets[None, :] * stride_do_dim
        )
        o = tl.load(o_ptrs, mask=mask, other=0.0)
        do = tl.load(do_ptrs, mask=mask, other=0.0)
        d_values = tl.sum(o * do, axis=1)

        d_ptrs = d_ptr + batch_idx * stride_d_batch + query_offsets * stride_d_seq
        tl.store(d_ptrs, d_values, mask=query_offsets < n_queries)

    @triton.jit
    def _flash_attention_backward_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        d_ptr,
        dk_ptr,
        dv_ptr,
        stride_q_batch,
        stride_q_seq,
        stride_q_dim,
        stride_k_batch,
        stride_k_seq,
        stride_k_dim,
        stride_v_batch,
        stride_v_seq,
        stride_v_dim,
        stride_do_batch,
        stride_do_seq,
        stride_do_dim,
        stride_lse_batch,
        stride_lse_seq,
        stride_d_batch,
        stride_d_seq,
        stride_dk_batch,
        stride_dk_seq,
        stride_dk_dim,
        stride_dv_batch,
        stride_dv_seq,
        stride_dv_dim,
        n_queries,
        n_keys,
        d_model,
        scale,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        key_block_idx = tl.program_id(1)

        key_offsets = key_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)
        query_offsets_base = tl.arange(0, BLOCK_M)

        k_mask = (key_offsets[:, None] < n_keys) & (dim_offsets[None, :] < d_model)
        k_ptrs = (
            k_ptr
            + batch_idx * stride_k_batch
            + key_offsets[:, None] * stride_k_seq
            + dim_offsets[None, :] * stride_k_dim
        )
        v_ptrs = (
            v_ptr
            + batch_idx * stride_v_batch
            + key_offsets[:, None] * stride_v_seq
            + dim_offsets[None, :] * stride_v_dim
        )
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

        for query_start in range(0, n_queries, BLOCK_M):
            query_offsets = query_start + query_offsets_base
            q_mask = (query_offsets[:, None] < n_queries) & (dim_offsets[None, :] < d_model)
            q_ptrs = (
                q_ptr
                + batch_idx * stride_q_batch
                + query_offsets[:, None] * stride_q_seq
                + dim_offsets[None, :] * stride_q_dim
            )
            do_ptrs = (
                do_ptr
                + batch_idx * stride_do_batch
                + query_offsets[:, None] * stride_do_seq
                + dim_offsets[None, :] * stride_do_dim
            )
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                scores = tl.where(causal_mask, scores, float("-inf"))
            scores = tl.where(query_offsets[:, None] < n_queries, scores, float("-inf"))

            lse_ptrs = lse_ptr + batch_idx * stride_lse_batch + query_offsets * stride_lse_seq
            d_row_ptrs = d_ptr + batch_idx * stride_d_batch + query_offsets * stride_d_seq
            lse = tl.load(lse_ptrs, mask=query_offsets < n_queries, other=0.0)
            d_row = tl.load(d_row_ptrs, mask=query_offsets < n_queries, other=0.0)

            probs = tl.exp(scores - lse[:, None])
            dp = tl.dot(do, tl.trans(v))
            ds = probs * (dp - d_row[:, None])
            dv += tl.dot(tl.trans(probs.to(do.dtype)), do)
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q) * scale

        dk_ptrs = (
            dk_ptr
            + batch_idx * stride_dk_batch
            + key_offsets[:, None] * stride_dk_seq
            + dim_offsets[None, :] * stride_dk_dim
        )
        dv_ptrs = (
            dv_ptr
            + batch_idx * stride_dv_batch
            + key_offsets[:, None] * stride_dv_seq
            + dim_offsets[None, :] * stride_dv_dim
        )
        tl.store(dk_ptrs, dk, mask=k_mask)
        tl.store(dv_ptrs, dv, mask=k_mask)

    @triton.jit
    def _flash_attention_backward_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        d_ptr,
        dq_ptr,
        stride_q_batch,
        stride_q_seq,
        stride_q_dim,
        stride_k_batch,
        stride_k_seq,
        stride_k_dim,
        stride_v_batch,
        stride_v_seq,
        stride_v_dim,
        stride_do_batch,
        stride_do_seq,
        stride_do_dim,
        stride_lse_batch,
        stride_lse_seq,
        stride_d_batch,
        stride_d_seq,
        stride_dq_batch,
        stride_dq_seq,
        stride_dq_dim,
        n_queries,
        n_keys,
        d_model,
        scale,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        query_block_idx = tl.program_id(1)

        query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        key_offsets_base = tl.arange(0, BLOCK_N)

        q_mask = (query_offsets[:, None] < n_queries) & (dim_offsets[None, :] < d_model)
        q_ptrs = (
            q_ptr
            + batch_idx * stride_q_batch
            + query_offsets[:, None] * stride_q_seq
            + dim_offsets[None, :] * stride_q_dim
        )
        do_ptrs = (
            do_ptr
            + batch_idx * stride_do_batch
            + query_offsets[:, None] * stride_do_seq
            + dim_offsets[None, :] * stride_do_dim
        )
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)
        lse_ptrs = lse_ptr + batch_idx * stride_lse_batch + query_offsets * stride_lse_seq
        d_row_ptrs = d_ptr + batch_idx * stride_d_batch + query_offsets * stride_d_seq
        lse = tl.load(lse_ptrs, mask=query_offsets < n_queries, other=0.0)
        d_row = tl.load(d_row_ptrs, mask=query_offsets < n_queries, other=0.0)

        dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for key_start in range(0, n_keys, BLOCK_N):
            key_offsets = key_start + key_offsets_base
            kv_mask = (key_offsets[:, None] < n_keys) & (dim_offsets[None, :] < d_model)
            k_ptrs = (
                k_ptr
                + batch_idx * stride_k_batch
                + key_offsets[:, None] * stride_k_seq
                + dim_offsets[None, :] * stride_k_dim
            )
            v_ptrs = (
                v_ptr
                + batch_idx * stride_v_batch
                + key_offsets[:, None] * stride_v_seq
                + dim_offsets[None, :] * stride_v_dim
            )
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                scores = tl.where(causal_mask, scores, float("-inf"))
            scores = tl.where(query_offsets[:, None] < n_queries, scores, float("-inf"))

            probs = tl.exp(scores - lse[:, None])
            dp = tl.dot(do, tl.trans(v))
            ds = probs * (dp - d_row[:, None])
            dq += tl.dot(ds.to(k.dtype), k) * scale

        dq_ptrs = (
            dq_ptr
            + batch_idx * stride_dq_batch
            + query_offsets[:, None] * stride_dq_seq
            + dim_offsets[None, :] * stride_dq_dim
        )
        tl.store(dq_ptrs, dq, mask=q_mask)


def _triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton()
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("Triton FlashAttention expects CUDA tensors.")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Expected q, k, v to have shape [batch, seq, dim].")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Batch dimensions of q, k, v must match.")
    if k.shape[-2] != v.shape[-2] or q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("Incompatible q, k, v shapes.")

    batch_size, n_queries, d_model = q.shape
    n_keys = k.shape[-2]
    output = torch.empty_like(q)
    lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

    block_d = triton.next_power_of_2(d_model)
    block_m = 32
    block_n = 32
    grid = (batch_size, triton.cdiv(n_queries, block_m))
    scale = 1.0 / math.sqrt(d_model)
    _flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        n_queries,
        n_keys,
        d_model,
        scale,
        is_causal=is_causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return output, lse


def _triton_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton()
    if not all(t.is_cuda for t in (q, k, v, output, lse, grad_output)):
        raise RuntimeError("Triton FlashAttention backward expects CUDA tensors.")

    batch_size, n_queries, d_model = q.shape
    n_keys = k.shape[-2]
    scale = 1.0 / math.sqrt(d_model)
    block_d = triton.next_power_of_2(d_model)
    block_m = 32
    block_n = 32

    d = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    d_grid = (batch_size, triton.cdiv(n_queries, block_m))
    _flash_attention_backward_d_kernel[d_grid](
        output,
        grad_output,
        d,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        d.stride(0),
        d.stride(1),
        n_queries,
        d_model,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )

    kv_grid = (batch_size, triton.cdiv(n_keys, block_n))
    _flash_attention_backward_dkdv_kernel[kv_grid](
        q,
        k,
        v,
        grad_output,
        lse,
        d,
        dk,
        dv,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        lse.stride(0),
        lse.stride(1),
        d.stride(0),
        d.stride(1),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        n_queries,
        n_keys,
        d_model,
        scale,
        is_causal=is_causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )

    q_grid = (batch_size, triton.cdiv(n_queries, block_m))
    _flash_attention_backward_dq_kernel[q_grid](
        q,
        k,
        v,
        grad_output,
        lse,
        d,
        dq,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        lse.stride(0),
        lse.stride(1),
        d.stride(0),
        d.stride(1),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        n_queries,
        n_keys,
        d_model,
        scale,
        is_causal=is_causal,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return dq, dk, dv


class TritonFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        output, lse = _triton_forward(q, k, v, is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, output, lse)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = _triton_backward(q, k, v, output, lse, grad_output, ctx.is_causal)
        return dq, dk, dv, None
