#!/usr/bin/env python3
"""Benchmark: Decode attention memory bandwidth with FlashInfer, batch=128, c=1024, varying page size."""

import torch
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

BATCH_SIZE = 128
CONTEXT_LEN = 1024
PAGE_SIZES = [1, 2, 4, 8, 16]


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


def setup_paged_kv(batch_size, seq_len, num_kv_heads, head_dim, page_size):
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

        total_bytes = compute_decode_bytes(BATCH_SIZE, num_heads, num_kv_heads, CONTEXT_LEN, head_dim)

        fi_gbps_list = []

        for page_size in PAGE_SIZES:
            # --- FlashInfer benchmark (batch decode with paged KV) ---
            q_fi = torch.randn(BATCH_SIZE, num_heads, head_dim, dtype=DTYPE, device=DEVICE)
            k_data, v_data, kv_indptr, kv_indices, kv_last_page_len = setup_paged_kv(
                BATCH_SIZE, CONTEXT_LEN, num_kv_heads, head_dim, page_size
            )

            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND", use_tensor_cores=True)
            wrapper.plan(
                kv_indptr, kv_indices, kv_last_page_len,
                num_heads, num_kv_heads, head_dim, page_size,
            )

            t = benchmark_fn(lambda: wrapper.run(q_fi, (k_data, v_data)))
            fi_gbps_list.append(total_bytes / t / 1e9)

            del q_fi, k_data, v_data, workspace, wrapper
            torch.cuda.empty_cache()

            print(f"  {config['name']}, page_size={page_size}: FlashInfer={fi_gbps_list[-1]:.2f} GB/s")

        axs[idx].plot(PAGE_SIZES, fi_gbps_list, label='FlashInfer', marker='o', color='tab:orange')
        axs[idx].set_title(config['name'])
        axs[idx].set_xlabel('Page size')
        axs[idx].set_xticks(PAGE_SIZES)
        axs[idx].set_xticklabels([str(p) for p in PAGE_SIZES])
        axs[idx].legend()
        axs[idx].grid(True, which='both')

    axs[0].set_ylabel('Memory Bandwidth Utilization (GB/s)')
    fig.suptitle('Decode Attention Memory Bandwidth (batch=128, c=1024, varying page size)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('decode_attention_page.png', dpi=300)
    print("Saved decode_attention_page.png")


if __name__ == '__main__':
    main()
