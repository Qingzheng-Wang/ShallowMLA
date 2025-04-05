import torch
import math

from torch import nn
from typing import Optional
from torch.nn import functional as F


def precompute_freqs_cis(
    qk_rope_head_dim: int, 
    seq_len: int, 
    seq_len_train: int, 
    beta_fast: int, 
    beta_slow: int, 
    rope_theta: float, 
    rope_factor: float
) -> torch.Tensor:
    """
    This function is adapted from 
    https://github.com/deepseek-ai/DeepSeek-V3/blob/a878eada08ea6913f5a2ae80a43afeffdef082ef/inference/model.py#L294

    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        seq_len: the seq len used during inference
        seq_len_train: the seq len used during training

        rope_theta: the base theta


    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # theta_i = 1 / (rope_theta ^ (2i / d)) i in {0, 2, ... d - 2}
    freqs = 1.0 / (rope_theta ** (torch.arange(0, qk_rope_head_dim, 2, dtype=torch.float32) / qk_rope_head_dim))
    if seq_len > seq_len_train: # if inference seq_len > seq_len_train
        low, high = find_correction_range(beta_fast, beta_slow, qk_rope_head_dim, rope_theta, seq_len_train)
        smooth = 1 - linear_ramp_factor(low, high, qk_rope_head_dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs) # outer product, freqs[t, i] = theta_i * t
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # exp(j * theta_i * t)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLATorch(nn.Module):
    def __init__(
        self,
        dim: int, # model dim
        kv_latent_rank: int, # rank of the cached compressed kv (c_kv)
        q_latent_rank: int, # rank of the cached compressed q (c_q)
        num_heads: int, # number of heads
        qk_nrope_head_dim: int, # dim of the q and k heads
        v_head_dim: int, # dim of the v head
        qk_rope_head_dim: int, # dim of the q and k rotary embeddings
        max_batch_size: int, # max batch size
        max_seq_len: int, # max sequence length
    ):
        super().__init__()
        self.qk_head_dim = qk_nrope_head_dim + qk_rope_head_dim

        self.proj_kv_down = nn.Linear(dim, kv_latent_rank + qk_rope_head_dim, bias=False)
        self.rms_norm_kv = nn.RMSNorm(kv_latent_rank)
        self.proj_kv_up = nn.Linear(kv_latent_rank, num_heads * (qk_nrope_head_dim + v_head_dim), bias=False)

        self.proj_q_down = nn.Linear(dim, q_latent_rank, bias=False)
        self.rms_norm_q = nn.RMSNorm(q_latent_rank)
        self.proj_q_up = nn.Linear(q_latent_rank, num_heads * self.qk_head_dim, bias=False)

        self.proj_out = nn.Linear(num_heads * v_head_dim, dim, bias=False)

        self.softmax_scale = 1.0 / self.qk_head_dim ** 0.5
        self.num_heads = num_heads
        self.kv_latent_rank = kv_latent_rank
        self.q_latent_rank = q_latent_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nrope_head_dim = qk_nrope_head_dim
        self.v_head_dim = v_head_dim

        self.register_buffer(
            "kv_latent_cache", 
            torch.zeros(max_batch_size, max_seq_len, kv_latent_rank), 
            persistent=False # the buffer will not be saved in the state dict
        )
        self.register_buffer(
            "k_rope_cache", 
            torch.zeros(max_batch_size, max_seq_len, qk_rope_head_dim),
            persistent=False
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, # the position of the x to be placed on the cache
        freq_cis: torch.Tensor, # the precomputed freq cis for rotary embeddings
        mask: Optional[torch.Tensor] = None, # the mask for the attention, [seq_len_q, seq_len_k]
    ):
        batch_size, seq_len, dim = x.shape
        
        q = self.proj_q_up(self.rms_norm_q(self.proj_q_down(x)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_nrope, q_rope = q.split(
            [self.qk_nrope_head_dim, self.qk_rope_head_dim], dim=-1
        ) # q_nrope: [batch_size, seq_len, num_heads, qk_nrope_head_dim], 
        # q_rope: [batch_size, seq_len, num_heads, qk_rope_head_dim]
        q_rope = apply_rotary_emb(q_rope, freq_cis)
        q = torch.cat([q_nrope, q_rope], dim=-1)
        
        kv_latent, k_rope = self.proj_kv_down(x).split(
            [self.kv_latent_rank, self.qk_rope_head_dim], dim=-1
        ) # kv_latent: [batch_size, seq_len, kv_latent_rank], k_rope: [batch_size, seq_len, qk_rope_head_dim]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freq_cis).squeeze(2) # [batch_size, seq_len, qk_rope_head_dim]

        end_pos = start_pos + seq_len
        self.kv_latent_cache[:batch_size, start_pos:end_pos] = kv_latent
        self.k_rope_cache[:batch_size, start_pos:end_pos] = k_rope

        # reshape the kv up weight, to make it be absorbed into the q
        proj_kv_up_weight = self.proj_kv_up.weight # [num_heads * (qk_nrope_head_dim + v_head_dim, kv_latent_rank]
        proj_kv_up_weight = proj_kv_up_weight.view(
            self.num_heads, self.qk_nrope_head_dim + self.v_head_dim, self.kv_latent_rank
        )
        # q_nrope absorb the kv_up weight, make q_nrope could directly matmul with kv_latent, 
        # and kv_latent can be directly get from the cache
        proj_kv_up_weight_q_nrope_absorbed = proj_kv_up_weight[:, :self.qk_nrope_head_dim, :]
        q_nrope_absorb = torch.einsum(
            "blhd,hdk->blhk", q_nrope, proj_kv_up_weight_q_nrope_absorbed
        ) # [batch_size, seq_len, num_heads, kv_latent_rank]
        scores = (
            torch.einsum(
                "blhk,btk->blht", q_nrope_absorb, self.kv_latent_cache[:batch_size, :end_pos]
            ) + 
            torch.einsum(
                "blhr,btr->blht", q_rope, self.k_rope_cache[:batch_size, :end_pos]
            )
        ) * self.softmax_scale # [batch_size, seq_len_q, num_heads, seq_len_k]

        # mask the scores
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(0) # [1, seq_len_q, 1, seq_len_k]
            scores += mask # [batch_size, seq_len_q, num_heads, seq_len_k]

        scores = scores.softmax(dim=-1)
        x = torch.einsum(
            "blht,btk->blhk", scores, self.kv_latent_cache[:batch_size, :end_pos]
        )
        proj_kv_up_weight_v = proj_kv_up_weight[:, -self.v_head_dim:, :]
        x = torch.einsum(
            "blhk,hdk->blhd", x, proj_kv_up_weight_v
        ) # [batch_size, seq_len, num_heads, v_head_dim]

        x = x.flatten(start_dim=2) # [batch_size, seq_len, num_heads * v_head_dim]
        x = self.proj_out(x)

        return x

class FFN(nn.Module):
    def __init__(
        self, 
        dim: int, # model dim
        ffn_hidden_dim: int, # hidden dim of the ffn
    ):
        super().__init__()
        self.proj_up_1 = nn.Linear(dim, ffn_hidden_dim, bias=False)
        self.proj_up_2 = nn.Linear(dim, ffn_hidden_dim, bias=False)
        self.proj_down = nn.Linear(ffn_hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.proj_down(F.silu(self.proj_up_1(x) + self.proj_up_2(x)))
    
class Layer(nn.Module):
    def __init__(
        self, 
        dim: int, # model dim
        kv_latent_rank: int, # rank of the cached compressed kv (c_kv)
        q_latent_rank: int, # rank of the cached compressed q (c_q)
        num_heads: int, # number of heads
        qk_nrope_head_dim: int, # dim of the q and k heads
        v_head_dim: int, # dim of the v head
        qk_rope_head_dim: int, # dim of the q and k rotary embeddings
        max_batch_size: int, # max batch size
        max_seq_len: int, # max sequence length
        ffn_hidden_dim: int, # hidden dim of the ffn
    ):
        super().__init__()
        self.mla = MLATorch(
            dim, kv_latent_rank, q_latent_rank, num_heads, qk_nrope_head_dim, 
            v_head_dim, qk_rope_head_dim, max_batch_size, max_seq_len
        )
        self.ffn = FFN(dim, ffn_hidden_dim)
        self.mla_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

    def forward(
            self, 
            x: torch.Tensor, 
            start_pos: int, 
            freq_cis: torch.Tensor, 
            mask: Optional[torch.Tensor] = None
        ):
        # use prenorm
        x = x + self.mla(self.mla_norm(x), start_pos, freq_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))

class Transformer(nn.Module):
    def __init__(
        self, 
        dim: int, # model dim
        kv_latent_rank: int, # rank of the cached compressed kv (c_kv)
        q_latent_rank: int, # rank of the cached compressed q (c_q)
        num_heads: int, # number of heads
        qk_nrope_head_dim: int, # dim of the q and k heads
        v_head_dim: int, # dim of the v head
        qk_rope_head_dim: int, # dim of the q and k rotary embeddings
        max_batch_size: int, # max batch size
        max_seq_len: int, # max sequence length
        max_seq_len_train: int, # max seq len used during training
        ffn_hidden_dim: int, # hidden dim of the ffn
        num_layers: int, # number of layers
        vocab_size: int, # vocabulary size
    ): 
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            Layer(
                dim, kv_latent_rank, q_latent_rank, num_heads, 
                qk_nrope_head_dim, v_head_dim, qk_rope_head_dim, 
                max_batch_size, max_seq_len, ffn_hidden_dim, 
            ) for _ in range(num_layers)
        ])
        self.register_buffer(
            "freq_cis", 
            precompute_freqs_cis(
                qk_rope_head_dim, max_seq_len, max_seq_len_train, 
                beta_fast=32, beta_slow=1, rope_theta=10000.0, 
                rope_factor=40.0
            )
        )
        self.final_norm = nn.RMSNorm(dim)
        self.final_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self, 
        tokens: torch.Tensor, # [batch_size, seq_len]
        start_pos: int = 0,
    ):
        x = self.embedding(tokens)
        seq_len = tokens.shape[1]
        end_pos = start_pos + seq_len
        freq_cis = self.freq_cis[start_pos:end_pos]
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device).triu_(1)
        
        for layer in self.layers:
            x = layer(x, start_pos, freq_cis, mask)
        
        x = self.final_norm(x)[:, -1] # use the last as the prediction
        logits = self.final_proj(x)
        return logits


