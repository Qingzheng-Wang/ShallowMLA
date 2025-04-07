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

    accumulator = tl.zeros((BLOCK_L, BLOCK_T), dtype=dtype)

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

def fused_qk_attention(
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
        BLOCK_D=32
    )

    return out
