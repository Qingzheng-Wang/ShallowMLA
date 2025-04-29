import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_L": 32, "BLOCK_T": 32, "BLOCK_K": 32, "BLOCK_R": 32
            }, num_warps=4
        ),
        triton.Config(
            {
                "BLOCK_L": 64, "BLOCK_T": 64, "BLOCK_K": 64, "BLOCK_R": 64
            }, num_warps=4
        ),
        triton.Config(
            {
                "BLOCK_L": 128, "BLOCK_T": 64, "BLOCK_K": 64, "BLOCK_R": 64
            }, num_warps=8
        ),
        triton.Config(
            {
                "BLOCK_L": 64, "BLOCK_T": 128, "BLOCK_K": 64, "BLOCK_R": 64
            }, num_warps=8
        ),
        triton.Config(
            {
                "BLOCK_L": 128, "BLOCK_T": 128, "BLOCK_K": 64, "BLOCK_R": 64
            }, num_warps=8
        ),
        triton.Config(
            {
                "BLOCK_L": 64, "BLOCK_T": 64, "BLOCK_K": 128, "BLOCK_R": 64
            }, num_warps=8
        ),
        triton.Config(
            {
                "BLOCK_L": 64, "BLOCK_T": 64, "BLOCK_K": 64, "BLOCK_R": 128
            }, num_warps=8
        ),
        triton.Config(
            {
                "BLOCK_L": 256, "BLOCK_T": 64, "BLOCK_K": 64, "BLOCK_R": 64
            }, num_warps=16
        ),
        triton.Config(
            {
                "BLOCK_L": 64, "BLOCK_T": 256, "BLOCK_K": 64, "BLOCK_R": 64
            }, num_warps=16
        )
    ],
    key=['L', 'T', 'K', 'R'] # re-sesearch when changing these values
)
@triton.jit
def fused_qk_attention_kernel(
    q_nrope_absorb_ptr, # [B, L, H, K]
    q_rope_ptr, # [B, L, H, R]
    kv_latent_cache_ptr, # [B, T, K]
    k_rope_cache_ptr, # [B, T, R]
    out_ptr, 
    B, L, H, K, R, T,
    stride_qn_b, stride_qn_l, stride_qn_h, stride_qn_k, 
    stride_qr_b, stride_qr_l, stride_qr_h, stride_qr_r,
    stride_kv_b, stride_kv_t, stride_kv_k,
    stride_kr_b, stride_kr_t, stride_kr_r,
    stride_out_b, stride_out_l, stride_out_h, stride_out_t,
    softmax_scale,
    BLOCK_L: tl.constexpr, # block size for seq_len_q
    BLOCK_T: tl.constexpr, # block size for seq_len_k
    BLOCK_K: tl.constexpr, # block size for kv_latent_rank
    BLOCK_R: tl.constexpr, # block size for qk_rope_head_dim
    dtype: tl.constexpr,
):
    """
    This kernel focuses on the QK^T in the MLA.
    MLA splits the QK^T into two parts:
        1. Q_{nrope}C^T, where C is the latent kv cache.
        2. Q_{rope}K_{rope}^T, where Q_{rope} and K_{rope} are the rope embeddings.
    This is represented as the following torch code:
    ```python
        torch.einsum(
            "blhk,btk->blht", q_nrope_absorb, self.kv_latent_cache[:batch_size, :end_pos]
        ) + 
        torch.einsum(
            "blhr,btr->blht", q_rope, self.k_rope_cache[:batch_size, :end_pos]
        )
    ```
    The common form of these two operation is: 
    [batch_size (b), seq_len_q (l), num_heads (h), kv_latent_rank (k) or qk_rope_head_dim (r)], 
    [batch_size (b), seq_len_k (t), kv_latent_rank (k) or qk_rope_head_dim (r)] ->
    [batch_size (b), seq_len_q (l), num_heads (h), seq_len_k (t)]

    b: batch_size
    l: seq_len_q
    h: num_heads
    k: kv_latent_rank
    r: qk_rope_head_dim
    t: seq_len_k

    """

    pid_l = tl.program_id(0) # q position
    pid_t = tl.program_id(1) # k position
    pid_b = tl.program_id(2) # batch position

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L) # one pid process one l or t
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K) # one pid reverse on the k or r dimension
    offs_r = tl.arange(0, BLOCK_R)

    mask_l = offs_l < L
    mask_t = offs_t < T

    for h in range(H):
        accumulator = tl.zeros((BLOCK_L, BLOCK_T), dtype=dtype)

        # q_nrope_absorb kv_latent_cache
        for k_off in range(0, K, BLOCK_K):
            mask_k = k_off + offs_k < K
            q_nrope_absorb = tl.load(
                q_nrope_absorb_ptr + 
                pid_b * stride_qn_b + 
                offs_l[:, None] * stride_qn_l + # [:, None] + [None, :] is (l, k), get a l * k matrix
                h * stride_qn_h + 
                (k_off + offs_k)[None, :] * stride_qn_k, 
                mask=mask_l[:, None] & mask_k[None, :],
                other=0.0
            )
            kv_latent_cache = tl.load(
                kv_latent_cache_ptr + 
                pid_b * stride_kv_b + 
                offs_t[None, :] * stride_kv_t + 
                (k_off + offs_k)[:, None] * stride_kv_k, # make k prior to t, it is like transpose
                mask=mask_t[None, :] & mask_k[:, None],
                other=0.0
            )
            accumulator += tl.dot(q_nrope_absorb, kv_latent_cache)

        for r_off in range(0, R, BLOCK_R):
            mask_r = r_off + offs_r < R
            q_rope = tl.load(
                q_rope_ptr + 
                pid_b * stride_qr_b + 
                offs_l[:, None] * stride_qr_l + 
                h * stride_qr_h + 
                (r_off + offs_r)[None, :] * stride_qr_r,
                mask=mask_l[:, None] & mask_r[None, :],
                other=0.0
            )
            k_rope_cache = tl.load(
                k_rope_cache_ptr +
                pid_b * stride_kr_b + 
                offs_t[None, :] * stride_kr_t +
                (r_off + offs_r)[:, None] * stride_kr_r,
                mask=mask_t[None, :] & mask_r[:, None],
                other=0.0
            )
            accumulator += tl.dot(q_rope, k_rope_cache)

        accumulator *= softmax_scale
        tl.store(
            out_ptr + 
            pid_b * stride_out_b + 
            offs_l[:, None] * stride_out_l +
            h * stride_out_h + 
            offs_t[None, :] * stride_out_t, 
            accumulator, 
            mask=mask_l[:, None] & mask_t[None, :],
        )


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_L": 32, "BLOCK_T": 32, "BLOCK_K": 32, "BLOCK_R": 32
            }, num_warps=4
        ),
        triton.Config(
            {
                "BLOCK_L": 64, "BLOCK_T": 64, "BLOCK_K": 64, "BLOCK_R": 64
            }, num_warps=4
        ),
    ],
    key=['L', 'T', 'K', 'R'] # re-sesearch when changing these values
)
@triton.jit
def fused_qk_attention_kernel_2(
    q_nrope_absorb_ptr, # [B, L, H, K]
    q_rope_ptr, # [B, L, H, R]
    kv_latent_cache_ptr, # [B, T, K]
    k_rope_cache_ptr, # [B, T, R]
    out_ptr, 
    B, L, H, K, R, T,
    stride_qn_b, stride_qn_l, stride_qn_h, stride_qn_k, 
    stride_qr_b, stride_qr_l, stride_qr_h, stride_qr_r,
    stride_kv_b, stride_kv_t, stride_kv_k,
    stride_kr_b, stride_kr_t, stride_kr_r,
    stride_out_b, stride_out_l, stride_out_h, stride_out_t,
    softmax_scale,
    BLOCK_L: tl.constexpr, # block size for seq_len_q
    BLOCK_T: tl.constexpr, # block size for seq_len_k
    BLOCK_K: tl.constexpr, # block size for kv_latent_rank
    BLOCK_R: tl.constexpr, # block size for qk_rope_head_dim
    dtype: tl.constexpr,
):
    """
    Version 2 of the fused qk attention kernel.
    THIS KERNEL PARALLIZE ON HEADS!!!

    This kernel focuses on the QK^T in the MLA.
    MLA splits the QK^T into two parts:
        1. Q_{nrope}C^T, where C is the latent kv cache.
        2. Q_{rope}K_{rope}^T, where Q_{rope} and K_{rope} are the rope embeddings.
    This is represented as the following torch code:
    ```python
        torch.einsum(
            "blhk,btk->blht", q_nrope_absorb, self.kv_latent_cache[:batch_size, :end_pos]
        ) + 
        torch.einsum(
            "blhr,btr->blht", q_rope, self.k_rope_cache[:batch_size, :end_pos]
        )
    ```
    The common form of these two operation is: 
    [batch_size (b), seq_len_q (l), num_heads (h), kv_latent_rank (k) or qk_rope_head_dim (r)], 
    [batch_size (b), seq_len_k (t), kv_latent_rank (k) or qk_rope_head_dim (r)] ->
    [batch_size (b), seq_len_q (l), num_heads (h), seq_len_k (t)]

    b: batch_size
    l: seq_len_q
    h: num_heads
    k: kv_latent_rank
    r: qk_rope_head_dim
    t: seq_len_k

    """

    pid_l = tl.program_id(0) # q position on the seq_len_q
    pid_t = tl.program_id(1) # k position on the seq_len_k
    pid_bh = tl.program_id(2) # batch and head index

    b = pid_bh // H
    h = pid_bh % H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L) # one pid process one l or t
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K) # one pid reverse on the k or r dimension
    offs_r = tl.arange(0, BLOCK_R)

    mask_l = offs_l < L
    mask_t = offs_t < T

    accumulator = tl.zeros((BLOCK_L, BLOCK_T), dtype=tl.float32)

    # q_nrope_absorb kv_latent_cache
    for k_off in range(0, K, BLOCK_K):
        mask_k = k_off + offs_k < K
        q_nrope_absorb = tl.load(
            q_nrope_absorb_ptr + 
            b * stride_qn_b + 
            offs_l[:, None] * stride_qn_l + # [:, None] + [None, :] is (l, k), get a l * k matrix
            h * stride_qn_h + 
            (k_off + offs_k)[None, :] * stride_qn_k, 
            mask=mask_l[:, None] & mask_k[None, :],
            other=0.0
        )
        kv_latent_cache = tl.load(
            kv_latent_cache_ptr + 
            b * stride_kv_b + 
            offs_t[None, :] * stride_kv_t + 
            (k_off + offs_k)[:, None] * stride_kv_k, # make k prior to t, it is like transpose
            mask=mask_t[None, :] & mask_k[:, None],
            other=0.0
        )
        accumulator += tl.dot(q_nrope_absorb, kv_latent_cache)

    for r_off in range(0, R, BLOCK_R):
        mask_r = r_off + offs_r < R
        q_rope = tl.load(
            q_rope_ptr + 
            b * stride_qr_b + 
            offs_l[:, None] * stride_qr_l + 
            h * stride_qr_h + 
            (r_off + offs_r)[None, :] * stride_qr_r,
            mask=mask_l[:, None] & mask_r[None, :],
            other=0.0
        )
        k_rope_cache = tl.load(
            k_rope_cache_ptr +
            b * stride_kr_b + 
            offs_t[None, :] * stride_kr_t +
            (r_off + offs_r)[:, None] * stride_kr_r,
            mask=mask_t[None, :] & mask_r[:, None],
            other=0.0
        )
        accumulator += tl.dot(q_rope, k_rope_cache)

    accumulator *= softmax_scale
    tl.store(
        out_ptr + 
        b * stride_out_b + 
        offs_l[:, None] * stride_out_l +
        h * stride_out_h + 
        offs_t[None, :] * stride_out_t, 
        accumulator, 
        mask=mask_l[:, None] & mask_t[None, :],
    )

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_D": 32
            }, num_warps=4
        ),
        triton.Config(
            {
                "BLOCK_D": 64
            }, num_warps=4
        ),
    ],
    key=['D'] # re-sesearch when changing these values
)
@triton.jit
def fused_apply_rotary_emb_kernel(
    x_ptr, # [B, L, H, D]
    freq_cis_ptr, # [L, D // 2, 2]
    out_ptr, # [B, L, H, D]
    B, L, H, D, 
    stride_x_b, stride_x_l, stride_x_h, stride_x_d,
    stride_freq_cis_l, stride_freq_cis_d, stride_freq_cis_2,
    stride_out_b, stride_out_l, stride_out_h, stride_out_d,
    BLOCK_D: tl.constexpr, # one block process half D
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    d_half = D // 2

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d_half

    x_real = tl.load(
        x_ptr + 
        pid_b * stride_x_b +
        pid_l * stride_x_l +
        pid_h * stride_x_h +
        offs_d * 2 * stride_x_d, # real part
        mask=mask_d,
        other=0.0
    )
    x_imag = tl.load(
        x_ptr +
        pid_b * stride_x_b +
        pid_l * stride_x_l +
        pid_h * stride_x_h +
        offs_d * 2 * stride_x_d + stride_x_d, # image part
        mask=mask_d,
        other=0.0
    )

    freqs_cos = tl.load(
        freq_cis_ptr + 
        pid_l * stride_freq_cis_l +
        offs_d * stride_freq_cis_d +
        0 * stride_freq_cis_2, # cos part
        mask=mask_d,
        other=0.0
    )
    freqs_sin = tl.load(
        freq_cis_ptr + 
        pid_l * stride_freq_cis_l +
        offs_d * stride_freq_cis_d +
        1 * stride_freq_cis_2, # sin part
        mask=mask_d,
        other=0.0
    )

    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos

    # save real and imag, the real and imag is interleaved
    tl.store(
        out_ptr + 
        pid_b * stride_out_b +
        pid_l * stride_out_l +
        pid_h * stride_out_h +
        offs_d * 2 * stride_out_d, # real part
        out_real,
        mask_d
    )
    tl.store(
        out_ptr +
        pid_b * stride_out_b +
        pid_l * stride_out_l +
        pid_h * stride_out_h +
        offs_d * 2 * stride_out_d + stride_out_d, # image part
        out_imag,
        mask_d
    )

def fused_qk_attention_forward(
    q_nrope_absorb: torch.Tensor, 
    q_rope: torch.Tensor, 
    kv_latent_cache: torch.Tensor, 
    k_rope_cache: torch.Tensor,
    softmax_scale: float,
    kernel_version: int = 1,
    dtype: tl.constexpr = tl.float32,
): 

    B, L, H, K = q_nrope_absorb.shape
    _, T, R = k_rope_cache.shape
    out = torch.empty((B, L, H, T), dtype=q_nrope_absorb.dtype, device=q_nrope_absorb.device)

    if kernel_version == 1:
        grid = lambda meta: (
            triton.cdiv(L, meta["BLOCK_L"]), 
            triton.cdiv(T, meta["BLOCK_T"]),
            B, # batch size
        )
        kernel_func = fused_qk_attention_kernel
    elif kernel_version == 2:
        grid = lambda meta: (
            triton.cdiv(L, meta["BLOCK_L"]), 
            triton.cdiv(T, meta["BLOCK_T"]),
            B * H, # batch size * num_heads
        )
        kernel_func = fused_qk_attention_kernel_2

    kernel_func[grid](
        q_nrope_absorb,
        q_rope,
        kv_latent_cache,
        k_rope_cache,
        out,
        B, L, H, K, R, T,
        *q_nrope_absorb.stride(),
        *q_rope.stride(),
        *kv_latent_cache.stride(),
        *k_rope_cache.stride(),
        *out.stride(),
        softmax_scale, 
        # BLOCK_L=32,
        # BLOCK_T=32,
        # BLOCK_K=32,
        # BLOCK_R=32,
        dtype=dtype,
    ) # the block sizes is tuned by autotune

    return out

class FusedQKAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_nrope_absorb, q_rope, kv_latent_cache, k_rope_cache,
                softmax_scale, kernel_version, dtype):
        scores = fused_qk_attention_forward(
            q_nrope_absorb, q_rope,
            kv_latent_cache, k_rope_cache,
            softmax_scale, kernel_version=kernel_version, dtype=dtype
        )
        
        ctx.save_for_backward(q_nrope_absorb, q_rope, kv_latent_cache, k_rope_cache)
        ctx.softmax_scale = softmax_scale
        
        return scores
    
    @staticmethod
    def backward(ctx, grad_scores):
        """
        Backward pass: Computes gradients for the inputs.
        dL/dQ_n = (dL/dScores * scale) @ K_l
        dL/dK_l = Q_n.T @ (dL/dScores * scale)
        dL/dQ_r = (dL/dScores * scale) @ K_r
        dL/dK_r = Q_r.T @ (dL/dScores * scale)
        """
        # Retrieve saved tensors and parameters
        q_nrope_absorb, q_rope, kv_latent_cache, k_rope_cache = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale

        scaled_grad_scores = grad_scores * softmax_scale

        grad_q_nrope_absorb = torch.einsum("blht,btk->blhk", scaled_grad_scores, kv_latent_cache)

        grad_kv_latent_cache = torch.einsum("blhk,blht->btk", q_nrope_absorb, scaled_grad_scores)

        grad_q_rope = torch.einsum("blht,btr->blhr", scaled_grad_scores, k_rope_cache)

        grad_k_rope_cache = torch.einsum("blhr,blht->btr", q_rope, scaled_grad_scores)

        return (
            grad_q_nrope_absorb,
            grad_q_rope,
            grad_kv_latent_cache,
            grad_k_rope_cache,
            None,  # Gradient for softmax_scale
            None,  # Gradient for kernel_version
            None,  # Gradient for dtype
        )
        
def fused_qk_attention(
    q_nrope_absorb: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent_cache: torch.Tensor,
    k_rope_cache: torch.Tensor,
    softmax_scale: float,
    kernel_version: int = 1,
    dtype: tl.constexpr = tl.float16,
):
    return FusedQKAttentionFunction.apply(
        q_nrope_absorb, q_rope, kv_latent_cache, k_rope_cache,
        softmax_scale, kernel_version, dtype
    )
        
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 64}, num_warps=4),
    ],
    key=["T"] # Autotune based on sequence length T
)
@triton.jit
def fused_mask_softmax_kernel_out_of_place(
    scores_ptr,
    mask_ptr,
    output_ptr,
    B, L, H, T,
    stride_scores_b, stride_scores_l, stride_scores_h, stride_scores_t,
    stride_mask_l, stride_mask_t,
    stride_out_b, stride_out_l, stride_out_h, stride_out_t,
    BLOCK_T: tl.constexpr,
):
    # Each program processes one row, corresponding to (b, l, h)
    pid = tl.program_id(0)
    # Decode b, l, h from the grid size (B*L*H)
    tmp = pid
    b = tmp // (L * H)
    tmp = tmp % (L * H)
    l = tmp // H
    h = tmp % H

    # Starting address of the current row in scores
    row_scores_ptr = scores_ptr + b * stride_scores_b + l * stride_scores_l + h * stride_scores_h
    
    row_output_ptr = output_ptr + b * stride_out_b + l * stride_out_l + h * stride_out_h

    # First Pass : Compute max
    row_max = -float('inf') # Use float('inf') instead of 1e-9 for better generality
    offs = tl.arange(0, BLOCK_T)
    for start in range(0, T, BLOCK_T):
        offs_t = start + offs
        mask_valid = offs_t < T
        scores_tile = tl.load(
            row_scores_ptr + offs_t * stride_scores_t,
            mask=mask_valid,
            other=-float('inf') # Use -inf
        )
        mask_tile = tl.load(
            mask_ptr + l * stride_mask_l + offs_t * stride_mask_t,
            mask=mask_valid,
            other=0.0
        )
        tile = scores_tile + mask_tile
        
        current_max = tl.max(tl.where(mask_valid, tile, -float('inf')), axis=0)
        row_max = tl.maximum(row_max, current_max)

    # Second Pass: Compute sum
    row_sum = 0.0
    for start in range(0, T, BLOCK_T):
        offs_t = start + offs
        mask_valid = offs_t < T
        scores_tile = tl.load(
            row_scores_ptr + offs_t * stride_scores_t,
            mask=mask_valid,
            other=-float('inf') # changed from 1e-9
        )
        mask_tile = tl.load(
            mask_ptr + l * stride_mask_l + offs_t * stride_mask_t,
            mask=mask_valid,
            other=0.0
        )
        tile = scores_tile + mask_tile
        # Subtract max for stability. Ensure invalid elements don't contribute to sum.
        tile = tl.where(mask_valid, tile - row_max, -float('inf'))
        exp_tile = tl.exp(tile) # exp(-inf) = 0
        # Accumulate sum. Ensure only valid elements contribute.
        row_sum += tl.sum(exp_tile, axis=0) # exp_tile is already 0 where mask_valid is false

    # Third Pass: Normalize and write to OUTPUT
    for start in range(0, T, BLOCK_T):
        offs_t = start + offs
        mask_valid = offs_t < T
        scores_tile = tl.load(
            row_scores_ptr + offs_t * stride_scores_t,
            mask=mask_valid,
            other=-float('inf') # Or same value as in Pass 2
        )
        mask_tile = tl.load(
            mask_ptr + l * stride_mask_l + offs_t * stride_mask_t,
            mask=mask_valid,
            other=0.0
        )
        tile = scores_tile + mask_tile
        tile = tl.where(mask_valid, tile - row_max, -float('inf'))
        exp_tile = tl.exp(tile)
        softmax_tile = exp_tile / row_sum
        tl.store(row_output_ptr + offs_t * stride_out_t, softmax_tile, mask=mask_valid)
    
def fused_mask_softmax_forward_out_of_place(scores: torch.Tensor, mask: torch.Tensor): # Renamed
    """
    (Out-of-place version)
    fused_mask_softmax: Apply mask to scores and perform softmax normalization
                        along the last dimension, writing to a new tensor.
    scores: [B, L, H, T] (Input)
    mask: [1, L, 1, T] (Input)
    Returns: [B, L, H, T] (Output - Softmax result)
    """
    B, L, H, T = scores.shape
    # Create the output tensor
    # Ensure output tensor has the same dtype and device as scores
    output = torch.empty_like(scores)

    stride_scores_b, stride_scores_l, stride_scores_h, stride_scores_t = scores.stride()

    _, stride_mask_l, _, stride_mask_t = mask.stride()
    stride_out_b, stride_out_l, stride_out_h, stride_out_t = output.stride()

    grid = lambda meta: (B * L * H,)

    fused_mask_softmax_kernel_out_of_place[grid](
        scores,         # Input scores
        mask,           # Input mask
        output,         # Output tensor
        B, L, H, T,
        stride_scores_b, stride_scores_l, stride_scores_h, stride_scores_t, # Input scores strides
        stride_mask_l, stride_mask_t,                                       # Input mask strides
        stride_out_b, stride_out_l, stride_out_h, stride_out_t,             # Output strides
        # BLOCK_T is autotuned by the kernel decorator
    )
    
    return output
    
class FusedMaskSoftmaxFunction(torch.autograd.Function):
    """
    PyTorch autograd Function wrapper for the fused masked softmax Triton kernel.
    Computes: output = softmax(scores + mask, dim=-1)
    """

    @staticmethod
    def forward(ctx, scores, mask):
        softmax_output = fused_mask_softmax_forward_out_of_place(scores, mask)
        
        ctx.save_for_backward(softmax_output)

        return softmax_output

    @staticmethod
    def backward(ctx, grad_softmax_output):
        """
        Backward pass: Computes gradient for scores. Gradient for mask is None.
        dL/dx = y * (dL/dy - sum(dL/dy * y)), where y = softmax(x)
        Here x = scores + mask. Since d(scores+mask)/dscores = 1,
        dL/dscores = dL/dx.
        """
        softmax_output, = ctx.saved_tensors
        sum_term = torch.sum(grad_softmax_output * softmax_output, dim=-1, keepdim=True)
        grad_scores = softmax_output * (grad_softmax_output - sum_term)

        return grad_scores, None
    
def fused_mask_softmax(scores: torch.Tensor, mask: torch.Tensor):
    return FusedMaskSoftmaxFunction.apply(scores, mask)

def fused_apply_rotary_emb(
    x: torch.Tensor,
    freq_cis: torch.Tensor
):
    B, L, H, D = x.shape
    out = torch.empty((B, L, H, D), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        B, # batch size
        L, # seq_len_q
        H, # num_heads
    )

    fused_apply_rotary_emb_kernel[grid](
        x, freq_cis, out, 
        B, L, H, D,
        *x.stride(),
        *freq_cis.stride(),
        *out.stride(),
        # BLOCK_D=32 # auto tune
    )

    return out

DEFAULT_BLOCK_SIZE = 1024

@triton.jit
def _optimized_rms_norm_kernel(
    x_ptr, out_ptr, weight_ptr,
    batch_size, normalized_dim,
    stride_x_batch, stride_x_dim,
    stride_out_batch, stride_out_dim,
    stride_weight,
    epsilon,
    BLOCK_SIZE: tl.constexpr = DEFAULT_BLOCK_SIZE,
):
    """
    Every program deals with one row on the x (assuming the last dim as the dim to normalized) 
    """
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return

    # assuming the storage of x is contiguous
    row_ptr = x_ptr + pid * stride_x_batch
    out_row_ptr = out_ptr + pid * stride_out_batch

    # calculate the squared mean
    sum_squares = 0.0
    offs = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, normalized_dim, BLOCK_SIZE):
        block_offs = block_start + offs
        mask = block_offs < normalized_dim
        x_block = tl.load(row_ptr + block_offs * stride_x_dim, mask=mask, other=0.0)
        sum_squares += tl.sum(x_block * x_block * mask, axis=0)

    inv_rms = 1.0 / tl.sqrt(sum_squares / normalized_dim + epsilon)

    # normalize and multiplied by scale（weight）
    for block_start in range(0, normalized_dim, BLOCK_SIZE):
        block_offs = block_start + offs
        mask = block_offs < normalized_dim
        x_block = tl.load(row_ptr + block_offs * stride_x_dim, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + block_offs * stride_weight, mask=mask, other=1.0)
        out_block = x_block * inv_rms * weight_block
        tl.store(out_row_ptr + block_offs * stride_out_dim, out_block, mask=mask)

def fused_rms_norm(x, normalized_shape, weight, epsilon=1e-6):
    """
    Fused RMSNorm

    Require:
      - the last dim's shape should equals normalized_shape[0]
    """
    if weight.dtype != x.dtype:
        weight = weight.to(x.dtype)

    normalized_dim = normalized_shape[0]
    
    if not x.is_contiguous():
        x = x.contiguous()

    # batchsize_by_seqlen = batchsize * seqlen
    batchsize_by_seqlen = x.numel() // normalized_dim

    # create output tensor
    out = torch.empty_like(x)

    stride_x_batch = normalized_dim  # stride per row
    stride_x_dim = 1

    stride_out_batch = normalized_dim
    stride_out_dim = 1

    # assuming the weight to be 1-D tensor
    stride_weight = weight.stride(0) if weight.dim() > 0 else 0

    grid = (batchsize_by_seqlen,)

    _optimized_rms_norm_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        weight_ptr=weight,
        batch_size=batchsize_by_seqlen,
        normalized_dim=normalized_dim,
        stride_x_batch=stride_x_batch,
        stride_x_dim=stride_x_dim,
        stride_out_batch=stride_out_batch,
        stride_out_dim=stride_out_dim,
        stride_weight=stride_weight,
        epsilon=epsilon,
    )

    return out