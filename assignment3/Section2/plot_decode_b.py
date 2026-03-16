#!/usr/bin/env python3
"""Benchmark: Decode attention memory bandwidth, c=1024, varying batch size."""

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

CONTEXT_LEN = 1024
BATCH_SIZES = 2 ** np.arange(0, 7)  # 1 to 64
PAGE_SIZE = 16


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


def setup_paged_kv(batch_size, seq_len, num_kv_heads, head_dim, page_size=PAGE_SIZE):
    """Create paged KV cache tensors for FlashInfer (HND layout)."""
    pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size

    k_data = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=DTYPE, device=DEVICE)
    v_data = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=DTYPE, device=DEVICE)

    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=DEVICE) * pages_per_seq
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=DEVICE)

    last_page_len = seq_len % page_size
    if last_page_len == 0:
        last_page_len = page_size
    kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device=DEVICE)

    return k_data, v_data, kv_indptr, kv_indices, kv_last_page_len


def main():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, config in enumerate(CONFIGS):
        num_heads = config["num_attention_heads"]
        num_kv_heads = config["num_key_value_heads"]
        head_dim = config["hidden_size"] // num_heads
        n_rep = num_heads // num_kv_heads

        sdpa_gbps_list = []
        fi_gbps_list = []

        for batch_size in BATCH_SIZES:
            batch_size = int(batch_size)
            total_bytes = compute_decode_bytes(batch_size, num_heads, num_kv_heads, CONTEXT_LEN, head_dim)

            # --- PyTorch SDPA benchmark ---
            q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=DTYPE, device=DEVICE)
            k = torch.randn(batch_size, num_kv_heads, CONTEXT_LEN, head_dim, dtype=DTYPE, device=DEVICE)
            v = torch.randn(batch_size, num_kv_heads, CONTEXT_LEN, head_dim, dtype=DTYPE, device=DEVICE)
            k_exp = repeat_kv(k, n_rep)
            v_exp = repeat_kv(v, n_rep)

            t = benchmark_fn(lambda: F.scaled_dot_product_attention(q, k_exp, v_exp))
            sdpa_gbps_list.append(total_bytes / t / 1e9)

            del q, k, v, k_exp, v_exp
            torch.cuda.empty_cache()

            # --- FlashInfer benchmark (batch decode with paged KV) ---
            q_fi = torch.randn(batch_size, num_heads, head_dim, dtype=DTYPE, device=DEVICE)
            k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = setup_paged_kv(
                batch_size, CONTEXT_LEN, num_kv_heads, head_dim
            )

            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND", use_tensor_cores=True)
            wrapper.plan(
                kv_indptr, kv_indices, kv_last_page_len,
                num_heads, num_kv_heads, head_dim, PAGE_SIZE,
            )

            t = benchmark_fn(lambda: wrapper.run(q_fi, (k_data, v_data)))
            fi_gbps_list.append(total_bytes / t / 1e9)

            del q_fi, k_data, v_data, workspace, wrapper
            torch.cuda.empty_cache()

            print(f"  {config['name']}, batch={batch_size}: SDPA={sdpa_gbps_list[-1]:.2f}, FlashInfer={fi_gbps_list[-1]:.2f} GB/s")

        axs[idx].plot(BATCH_SIZES, sdpa_gbps_list, label='PyTorch SDPA', marker='o')
        axs[idx].plot(BATCH_SIZES, fi_gbps_list, label='FlashInfer', marker='x')
        axs[idx].set_xscale('log', base=2)
        axs[idx].set_title(config['name'])
        axs[idx].set_xlabel('Batch size')
        axs[idx].set_xticks(BATCH_SIZES)
        axs[idx].set_xticklabels([str(int(b)) for b in BATCH_SIZES])
        axs[idx].legend()
        axs[idx].grid(True, which='both')

    axs[0].set_ylabel('Memory Bandwidth Utilization (GB/s)')
    fig.suptitle('Decode Attention Memory Bandwidth (c=1024, varying batch size)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('decode_attention_b.png', dpi=300)
    print("Saved decode_attention_b.png")


if __name__ == '__main__':
    main()
