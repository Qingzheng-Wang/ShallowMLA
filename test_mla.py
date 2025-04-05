import torch
from mla import Transformer, precompute_freqs_cis, apply_rotary_emb, MLATorch
import time
import torch.profiler as profiler


def test_transformer_forward():
    torch.manual_seed(42)

    dim = 64
    kv_latent_rank = 32
    q_latent_rank = 32
    num_heads = 4
    qk_nrope_head_dim = 24
    v_head_dim = 16
    qk_rope_head_dim = 40
    max_batch_size = 2
    max_seq_len = 16
    max_seq_len_train = 8
    ffn_hidden_dim = 128
    num_layers = 2
    vocab_size = 100

    model = Transformer(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_seq_len_train=max_seq_len_train,
        ffn_hidden_dim=ffn_hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size
    )

    model.eval()
    batch_size = 2
    seq_len = 5
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(tokens, start_pos=0)

    print("Input shape:", tokens.shape)               # [2, 5]
    print("Logits shape:", logits.shape)              # [2, vocab_size]
    assert logits.shape == (batch_size, vocab_size), "Output shape mismatch"
    print("Test passed.")


def test_continuous_inference():
    print("Running test_continuous_inference...")
    torch.manual_seed(0)

    dim = 64
    kv_latent_rank = 32
    q_latent_rank = 32
    num_heads = 4
    qk_nrope_head_dim = 24
    v_head_dim = 16
    qk_rope_head_dim = 40
    max_batch_size = 1
    max_seq_len = 16
    max_seq_len_train = 8
    ffn_hidden_dim = 128
    num_layers = 2
    vocab_size = 50

    model = Transformer(
        dim, kv_latent_rank, q_latent_rank, num_heads,
        qk_nrope_head_dim, v_head_dim, qk_rope_head_dim,
        max_batch_size, max_seq_len, max_seq_len_train,
        ffn_hidden_dim, num_layers, vocab_size
    )

    model.eval()

    full_input = torch.full((1, 16), dtype=torch.long, fill_value=1)
    first_chunk = full_input[:, :3]  # [1, 2, 3]
    next_token = full_input[:, 3:]   # [4]

    with torch.no_grad():
        logits_full = model(full_input)

    with torch.no_grad():
        model(
            first_chunk,
            start_pos=0,
        )
        logits_incremental = model(
            next_token,
            start_pos=3
        )

    diff = torch.norm(logits_full - logits_incremental).item()
    print(f"logits diff: {diff:.6f}")
    assert diff < 1e-4, "Incremental logits mismatch!"
    print("✅ Continuous inference test passed.\n")


def test_rope_interpolation():
    print("Running test_rope_interpolation...")
    qk_rope_head_dim = 40
    seq_len_train = 8
    seq_len_long = 16
    rope_theta = 10000.0
    rope_factor = 40.0
    beta_fast = 32
    beta_slow = 1

    freqs_train = precompute_freqs_cis(
        qk_rope_head_dim, seq_len_train, seq_len_train, beta_fast, beta_slow, rope_theta, rope_factor
    )

    freqs_interp = precompute_freqs_cis(
        qk_rope_head_dim, seq_len_long, seq_len_train, beta_fast, beta_slow, rope_theta, rope_factor
    )

    # 取一个 fake 输入应用 rotary embedding，看是否不同
    dummy_input = torch.randn(1, seq_len_long, 1, qk_rope_head_dim)

    out_train = apply_rotary_emb(dummy_input[:, :seq_len_train], freqs_train)
    out_interp = apply_rotary_emb(dummy_input[:, :seq_len_train], freqs_interp[:seq_len_train])

    diff = torch.norm(out_train - out_interp).item()
    print(f"Interpolated RoPE output difference: {diff:.6f}")
    assert diff > 1e-4, "RoPE interpolation didn't take effect!"
    print("✅ RoPE interpolation test passed.\n")


def benchmark_mla():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    dim = 2048
    num_heads = 16
    kv_latent_rank = 512
    q_latent_rank = 512
    qk_nrope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    max_batch_size = 8
    max_seq_len = 4096 * 4

    mla = MLATorch(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    ).to(device)

    batch_size = 8
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, dim).to(device)

    start_pos = 0
    freq_cis = precompute_freqs_cis(
        qk_rope_head_dim, max_seq_len, seq_len, beta_fast=32, beta_slow=1,
        rope_theta=10000.0, rope_factor=40.0
    ).to(device)[start_pos: start_pos + seq_len]

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)

    # warmup
    for _ in range(5):
        _ = mla(x, start_pos, freq_cis, mask)

    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    # torch profiler benchmark
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_dir"),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for _ in range(5):
            _ = mla(x, start_pos, freq_cis, mask)
            prof.step()

    torch.cuda.synchronize() if device == "cuda" else None

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    end = time.time()

    avg_time = (end - start) / 10
    print(f"MLA forward average time: {avg_time:.4f} seconds")
    print(f"Throughput: {(batch_size * seq_len) / avg_time:.2f} tokens/sec")

if __name__ == "__main__":
    # test_transformer_forward()
    # test_continuous_inference()
    # test_rope_interpolation()
    benchmark_mla()
