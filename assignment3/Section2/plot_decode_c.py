#!/usr/bin/env python3
"""Benchmark: Decode attention memory bandwidth, batch=1, varying context length c."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import flashinfer

DEVICE = "cuda"
DTYPE = torch.float16
NUM_WARMUP = 10
NUM_ITERS = 100

# Model configurations
CONFIGS = [
    {"name": "LLaMA3-1B", "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8},
    {"name": "LLaMA3-3B", "hidden_size": 3072, "num_attention_heads": 24, "num_key_value_heads": 8},
    {"name": "LLaMA3-8B", "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8},
]

BATCH_SIZE = 1
CONTEXT_LENGTHS = 2 ** np.arange(7, 16)  # 128 to 32768


def compute_decode_bytes(batch, num_qo_heads, num_kv_heads, context_len, head_dim):
    """Total bytes read/written for decode attention (fp16 = 2 bytes per element)."""
    q_bytes = batch * num_qo_heads * head_dim * 2
    k_bytes = batch * num_kv_heads * context_len * head_dim * 2
    v_bytes = batch * num_kv_heads * context_len * head_dim * 2
    o_bytes = batch * num_qo_heads * head_dim * 2
    return q_bytes + k_bytes + v_bytes + o_bytes


def benchmark_fn(fn, num_warmup=NUM_WARMUP, num_iters=NUM_ITERS):
    """Run fn repeatedly and return average elapsed time in seconds."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters / 1000.0


def repeat_kv(x, n_rep):
    """Expand KV heads for SDPA: (B, H_kv, S, D) -> (B, H_kv*n_rep, S, D)."""
    if n_rep == 1:
        return x
    bs, h, s, d = x.shape
    return x[:, :, None, :, :].expand(bs, h, n_rep, s, d).reshape(bs, h * n_rep, s, d)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, config in enumerate(CONFIGS):
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        head_dim = config["hidden_size"] // num_heads
        n_rep = num_heads // num_kv_heads

        sdpa_gbps_list = []
        fi_gbps_list = []

        for context_len in CONTEXT_LENGTHS:
            context_len = int(context_len)
            total_bytes = compute_decode_bytes(BATCH_SIZE, num_heads, num_kv_heads, context_len, head_dim)

            # --- PyTorch SDPA benchmark ---
            # Decode: q_len=1, kv_len=context_len
            q = torch.randn(BATCH_SIZE, num_heads, 1, head_dim, dtype=DTYPE, device=DEVICE)
            k = torch.randn(BATCH_SIZE, num_kv_heads, context_len, head_dim, dtype=DTYPE, device=DEVICE)
            v = torch.randn(BATCH_SIZE, num_kv_heads, context_len, head_dim, dtype=DTYPE, device=DEVICE)
            k_exp = repeat_kv(k, n_rep)
            v_exp = repeat_kv(v, n_rep)

            t = benchmark_fn(lambda: F.scaled_dot_product_attention(q, k_exp, v_exp))
            sdpa_gbps_list.append(total_bytes / t / 1e9)

            del q, k, v, k_exp, v_exp
            torch.cuda.empty_cache()

            # --- FlashInfer benchmark (single decode) ---
            q_fi = torch.randn(num_heads, head_dim, dtype=DTYPE, device=DEVICE)
            k_fi = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
            v_fi = torch.randn(context_len, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)

            t = benchmark_fn(lambda: flashinfer.single_decode_with_kv_cache(q_fi, k_fi, v_fi))
            fi_gbps_list.append(total_bytes / t / 1e9)

            del q_fi, k_fi, v_fi
            torch.cuda.empty_cache()

            print(f"  {config['name']}, c={context_len}: SDPA={sdpa_gbps_list[-1]:.2f}, FlashInfer={fi_gbps_list[-1]:.2f} GB/s")

        axs[idx].plot(CONTEXT_LENGTHS, sdpa_gbps_list, label='PyTorch SDPA', marker='o')
        axs[idx].plot(CONTEXT_LENGTHS, fi_gbps_list, label='FlashInfer', marker='x')
        axs[idx].set_xscale('log', base=2)
        axs[idx].set_title(config['name'])
        axs[idx].set_xlabel('c (context length)')
        axs[idx].set_xticks(CONTEXT_LENGTHS)
        axs[idx].set_xticklabels([str(int(c)) for c in CONTEXT_LENGTHS], rotation=45)
        axs[idx].legend()
        axs[idx].grid(True, which='both')

    axs[0].set_ylabel('Memory Bandwidth Utilization (GB/s)')
    fig.suptitle('Decode Attention Memory Bandwidth (batch=1, varying c)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('decode_attention_c.png', dpi=300)
    print("Saved decode_attention_c.png")


if __name__ == '__main__':
    main()
