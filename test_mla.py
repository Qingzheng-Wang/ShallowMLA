import torch
from mla import (
    Transformer, precompute_freqs_cis, 
    apply_rotary_emb, MLA
)
import time
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from kernel import fused_qk_attention, fused_mask_softmax, fused_apply_rotary_emb


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

def test_mla_triton():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    torch.manual_seed(42)
    dtype = torch.float16
    
    dim = 2048
    num_heads = 16
    kv_latent_rank = 512
    q_latent_rank = 512
    qk_nrope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    max_batch_size = 8
    max_seq_len = 4096 * 4

    mla_torch = MLA(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        optim_type="torch",
    ).to(device)

    mla_triton = MLA(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        optim_type="triton",
    ).to(device)

    mla_triton.load_state_dict(mla_torch.state_dict())

    batch_size = 8
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, dim, dtype=dtype).to(device)

    start_pos = 0
    freq_cis = precompute_freqs_cis(
        qk_rope_head_dim, max_seq_len, seq_len, beta_fast=32, beta_slow=1,
        rope_theta=10000.0, rope_factor=40.0
    ).to(device)[start_pos: start_pos + seq_len]

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)

    result_torch = mla_torch(x, start_pos, freq_cis, mask)
    result_triton = mla_triton(x, start_pos, freq_cis, mask)

    diff = torch.norm(result_torch - result_triton).item()
    if diff < 1e-4:
        print("✅ Triton and Torch results match.")
    else:
        print(f"❌ Triton and Torch results differ: {diff:.6f}")

def test_fused_qk_attention():

    B = 8         # batch size
    L = 1024      # query len
    T = 1024      # key len
    H = 16        # num heads
    K = 512       # latent rank
    R = 64        # rope dim
    softmax_scale = 1.0 / (K + R) ** 0.5  # match with model

    q_nrope_absorb = torch.randn(B, L, H, K, dtype=torch.float32, device='cuda')
    q_rope = torch.randn(B, L, H, R, dtype=torch.float32, device='cuda')
    kv_latent_cache = torch.randn(B, T, K, dtype=torch.float32, device='cuda')
    k_rope_cache = torch.randn(B, T, R, dtype=torch.float32, device='cuda')

    scores_torch = (
        torch.einsum("blhk,btk->blht", q_nrope_absorb, kv_latent_cache) +
        torch.einsum("blhr,btr->blht", q_rope, k_rope_cache)
    ) * softmax_scale

    scores_triton = fused_qk_attention(
        q_nrope_absorb, q_rope, kv_latent_cache, k_rope_cache, softmax_scale, 
        kernel_version=2
    )

    print("Max abs diff:", (scores_torch - scores_triton).abs().max().item())
    print("Mean diff:", (scores_torch - scores_triton).abs().mean().item())
    print("Scores equal:", torch.allclose(scores_torch, scores_triton, rtol=1e-3, atol=1e-3))
    
def test_fused_mask_softmax():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} for fused_mask_softmax test")

    torch.manual_seed(42)

    B = 8   # batch
    L = 256 # seq_len_q
    H = 16  # heads
    T = 256 # seq_len_k

    scores = torch.randn(B, L, H, T, dtype=torch.float32, device=device)

    base_mask = torch.full((L, T), float('-inf'), device=device)
    base_mask = torch.triu(base_mask, diagonal=1)
    mask = base_mask.view(1, L, 1, T)

    # Torch
    scores_torch = scores.clone()
    scores_torch += mask
    scores_torch = torch.softmax(scores_torch, dim=-1)

    # Triton
    scores_triton = scores.clone()
    fused_mask_softmax(scores_triton, mask)

    abs_diff = (scores_torch - scores_triton).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print(f"Max abs diff: {max_abs_diff:.6f}")
    print(f"Mean abs diff: {mean_abs_diff:.6f}")

def test_fused_apply_rotary_emb():
    B = 8         # batch size
    L = 1024      # query len
    H = 16        # num heads
    D = 64        # rope dim

    x = torch.randn(B, L, H, D, dtype=torch.float32, device='cuda')
    freqs_cis = torch.randn(L, D // 2, 2, dtype=torch.float32, device='cuda')
    out_torch = apply_rotary_emb(x, freqs_cis)
    out_triton = fused_apply_rotary_emb(x, freqs_cis)

    if torch.allclose(out_torch, out_triton, rtol=1e-3, atol=1e-3):
        print("✅ fused apply rotary emb test passed.")
        print("Max abs diff:", (out_torch - out_triton).abs().max().item())
        print("Mean diff:", (out_torch - out_triton).abs().mean().item())
        print("Outputs equal:", torch.allclose(out_torch, out_triton, rtol=1e-3, atol=1e-3))
    else:
        print("❌ fused apply rotary emb test failed.")
        print("Max abs diff:", (out_torch - out_triton).abs().max().item())
        print("Mean diff:", (out_torch - out_triton).abs().mean().item())
        print("Outputs equal:", torch.allclose(out_torch, out_triton, rtol=1e-3, atol=1e-3))

def test_rope():
    # test if two version of apply_rotary_emb are the same
    # the origin is input with freqs_cis as complex
    # the new one is input with freqs as real and imag
    x = torch.randn(2, 4, 4, 4)
    freqs = torch.randn(4, 2, 2)
    freqs_complex = torch.view_as_complex(freqs)
    from mla import apply_rotary_emb, apply_rotary_emb_origin
    out = apply_rotary_emb(x, freqs)
    out_origin = apply_rotary_emb_origin(x, freqs_complex)
    if torch.allclose(out, out_origin):
        print("✅ RoPE test passed.")

def test_mla_cache_manager_triton():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    torch.manual_seed(42)
    dtype = torch.float16
    
    dim = 2048
    num_heads = 16
    kv_latent_rank = 512
    q_latent_rank = 512
    qk_nrope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    max_batch_size = 8
    max_seq_len = 4096 * 4

    mla_triton = MLA(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        optim_type="triton", # ablation:rmsnorm:rope:qk_attention
        use_page_cache=True,
        use_page_cache_triton=False,
        page_size=1024,
    ).to(device)

    mla_triton_optim_page_cache = MLA(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        optim_type="triton", # ablation:rmsnorm:rope:qk_attention
        use_page_cache=True,
        use_page_cache_triton=True,
        page_size=1024,
    ).to(device)

    mla_triton_optim_page_cache.load_state_dict(mla_triton.state_dict())

    batch_size = 8
    seq_len = 1024
    x = torch.randn(batch_size, seq_len, dim, dtype=dtype).to(device)

    start_pos = 0
    freq_cis = precompute_freqs_cis(
        qk_rope_head_dim, max_seq_len, seq_len, beta_fast=32, beta_slow=1,
        rope_theta=10000.0, rope_factor=40.0
    ).to(device)[start_pos: start_pos + seq_len]

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)

    result_triton = mla_triton(x, start_pos, freq_cis, mask)
    result_triton_optim_page_cache = mla_triton_optim_page_cache(x, start_pos, freq_cis, mask)

    diff = torch.norm(result_triton - result_triton_optim_page_cache).item()
    if diff < 1e-4:
        print("✅ Triton and Triton with optim cache results match.")
    else:
        print(f"❌ Triton and Triton with optim cache results differ: {diff:.6f}")
        
def benchmark_mla(batch_size=8, seq_len=1024, use_profile=False, dtype=torch.float32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    dim = 2048
    num_heads = 16
    kv_latent_rank = 512
    q_latent_rank = 512
    qk_nrope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    max_batch_size = 32
    max_seq_len = 4096 * 4

    mla_torch = MLA(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        optim_type="torch",
        use_page_cache=True,
        page_size=1024,
    ).to(device)
    mla_torch.eval()

    x = torch.randn(batch_size, seq_len, dim, dtype=dtype).to(device)

    start_pos = 0
    freq_cis = precompute_freqs_cis(
        qk_rope_head_dim, max_seq_len, seq_len, beta_fast=32, beta_slow=1,
        rope_theta=10000.0, rope_factor=40.0
    ).to(device)[start_pos: start_pos + seq_len]

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype).triu_(1)

    torch.cuda.empty_cache() if device == "cuda" else None
    # warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = mla_torch(x, start_pos, freq_cis, mask)

    torch.cuda.empty_cache() if device == "cuda" else None
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    with torch.inference_mode():
        for _ in range(10):
            _ = mla_torch(x, start_pos, freq_cis, mask)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()
    avg_time_torch = (end - start) / 10

    # ========== MLA with paged cache, but not triton optimized =========

    mla_triton = MLA(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        optim_type="triton",
        use_page_cache=True,
        use_page_cache_triton=False,
        page_size=1024,
    ).to(device)

    mla_triton.load_state_dict(mla_torch.state_dict()) # ensure weights are the same
    mla_triton.eval()

    torch.cuda.empty_cache() if device == "cuda" else None
    # warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = mla_triton(x, start_pos, freq_cis, mask)

    torch.cuda.empty_cache() if device == "cuda" else None
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    # torch profiler benchmark
    with torch.inference_mode():
        for _ in range(10):
            _ = mla_triton(x, start_pos, freq_cis, mask)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()
    avg_time_triton = (end - start) / 10

    if use_profile:
        print("\n🔍 Profiler for MLA Torch")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3),
            on_trace_ready=tensorboard_trace_handler(f"logdir/torch_{int(time.time())}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                _ = mla_torch(x, start_pos, freq_cis, mask)
                prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        print("\n🔍 Profiler for MLA Triton")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3),
            on_trace_ready=tensorboard_trace_handler(f"log_dir/triton_{int(time.time())}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                _ = mla_triton(x, start_pos, freq_cis, mask)
                prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # ========== MLA with triton optimized paged cache =========

    mla_triton_optim_page_cache = MLA(
        dim=dim,
        kv_latent_rank=kv_latent_rank,
        q_latent_rank=q_latent_rank,
        num_heads=num_heads,
        qk_nrope_head_dim=qk_nrope_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        optim_type="triton",
        use_page_cache=True,
        use_page_cache_triton=True,
        page_size=1024,
    ).to(device)

    mla_triton_optim_page_cache.load_state_dict(mla_torch.state_dict()) # ensure weights are the same
    mla_triton_optim_page_cache.eval()

    torch.cuda.empty_cache() if device == "cuda" else None
    # warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = mla_triton_optim_page_cache(x, start_pos, freq_cis, mask)

    torch.cuda.empty_cache() if device == "cuda" else None
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    # torch profiler benchmark
    with torch.inference_mode():
        for _ in range(10):
            _ = mla_triton_optim_page_cache(x, start_pos, freq_cis, mask)

    torch.cuda.synchronize() if device == "cuda" else None
    end = time.time()
    avg_time_triton_optim_page_cache = (end - start) / 10

    print(f"MLA Torch forward average time: {avg_time_torch:.4f} seconds")
    print(f"MLA Triton forward average time: {avg_time_triton:.4f} seconds")
    print(f"MLA Triton optimized page cache forward average time: {avg_time_triton_optim_page_cache:.4f} seconds")
    throughput_torch = (batch_size * seq_len) / avg_time_torch
    throughput_triton = (batch_size * seq_len) / avg_time_triton
    throughput_triton_optim_page_cache = (batch_size * seq_len) / avg_time_triton_optim_page_cache
    print(f"Throughput Torch: {throughput_torch:.2f} tokens/sec")
    print(f"Throughput Triton: {throughput_triton:.2f} tokens/sec")
    print(f"Throughput Triton optimized page cache: {throughput_triton_optim_page_cache:.2f} tokens/sec")
    if mla_torch.use_page_cache:
        print("=========MLA Torch page cache stats=========")
        print(f"Page cache size: {mla_torch.page_size}")
        for key, value in mla_torch.cache_manager.get_memory_usage().items():
            print(f"{key}: {value}")
    if mla_triton.use_page_cache:
        print("=========MLA Triton page cache stats (Non-Triton page cache)=========")
        print(f"Page cache size: {mla_triton.page_size}")
        for key, value in mla_triton.cache_manager.get_memory_usage().items():
            print(f"{key}: {value}")
    if mla_triton_optim_page_cache.use_page_cache:
        print("=========MLA Triton page cache stats (Triton page cache)=========")
        print(f"Page cache size: {mla_triton_optim_page_cache.page_size}")
        for key, value in mla_triton_optim_page_cache.cache_manager.get_memory_usage().items():
            print(f"{key}: {value}")

    return avg_time_torch, avg_time_triton, throughput_torch, throughput_triton

def benchmark():
    batch_sizes = [4]
    seq_lens = [4096]
    results = {}
    torch.manual_seed(88)

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
            (
                avg_time_torch, 
                avg_time_triton, 
                throughput_torch,
                throughput_triton
            ) = benchmark_mla(
                batch_size=batch_size, 
                seq_len=seq_len, 
                use_profile=False, 
                dtype=torch.float16
            )
            results[(batch_size, seq_len)] = (
                avg_time_torch, 
                avg_time_triton, 
                throughput_torch,
                throughput_triton
            )
    
    import matplotlib.pyplot as plt

    # Extract data for plotting
    batch_sizes = sorted(set(k[0] for k in results.keys()))
    seq_lens = sorted(set(k[1] for k in results.keys()))

    torch_throughputs = [
        [results[(batch_size, seq_len)][2] for seq_len in seq_lens]
        for batch_size in batch_sizes
    ]
    triton_throughputs = [
        [results[(batch_size, seq_len)][3] for seq_len in seq_lens]
        for batch_size in batch_sizes
    ]

    # Plot throughput comparison
    plt.figure(figsize=(12, 8))
    for i, batch_size in enumerate(batch_sizes):
        plt.plot(seq_lens, torch_throughputs[i], label=f"Torch (Batch {batch_size})", marker='o')
        plt.plot(seq_lens, triton_throughputs[i], label=f"Triton (Batch {batch_size})", marker='x')

    plt.xlabel("Sequence Length")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Throughput Comparison: Torch vs Triton")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("throughput_comparison.png")

if __name__ == "__main__":
    # test_transformer_forward()
    # test_continuous_inference()
    # test_rope_interpolation()
    # benchmark_mla()
    benchmark()
    # test_mla_triton()
    # test_fused_qk_attention()
    # test_rope()
    # test_fused_apply_rotary_emb()
    # test_fused_mask_softmax()
    # test_mla_cache_manager_triton()