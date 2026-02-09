import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import the Engine classes from the respective files
from no_kv import Engine as NoKVEngine
from single_batch import Engine as SingleBatchEngine


def benchmark_no_kv(engine, input_ids_list, output_length):
    """
    Benchmarks the no_kv engine.
    `input_ids_list` should be a python list of ints.
    """
    output_ids = input_ids_list.copy()

    torch.cuda.synchronize()
    start_time = time.time()

    # The `no_kv.py` implementation re-processes the entire sequence for each new token.
    new_token = engine.run(output_ids, prefill=True)
    output_ids.append(new_token)

    for _ in range(output_length - 1):
        new_token = engine.run(output_ids, prefill=True)
        output_ids.append(new_token)

    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def benchmark_single_batch(engine, input_ids_tensor, output_length):
    """
    Benchmarks the single_batch engine (with KV cache).
    `input_ids_tensor` should be a torch tensor on the correct device.
    """
    torch.cuda.synchronize()
    start_time = time.time()

    # Prefill step
    new_token = engine.run(input_ids_tensor, prefill=True)

    # Decode step
    for _ in range(output_length - 1):
        # Pass only the last generated token
        new_token = engine.run([new_token], prefill=False)

    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


def main():
    """
    Main function to run the benchmark and plot results.
    """
    input_length = 1024
    output_lengths = list(range(128, 2049, 128))

    torch.manual_seed(42)
    input_ids_tensor = torch.randint(0, 32000, (input_length,), device="cuda")
    input_ids_list = input_ids_tensor.tolist()

    print("Initializing engines (this may take a moment)...")
    no_kv_engine = NoKVEngine()
    single_batch_engine = SingleBatchEngine()
    print("Engines initialized.")

    no_kv_times = []
    single_batch_times = []

    print("\nStarting benchmark...")
    print(f"Input length: {input_length} tokens")
    print("Output lengths to test:", output_lengths)
    print("-" * 30)

    for i, out_len in enumerate(output_lengths):
        print(
            f"[{i + 1}/{len(output_lengths)}] Benchmarking for output length: {out_len}"
        )

        # Benchmark no_kv
        t = benchmark_no_kv(no_kv_engine, input_ids_list, out_len)
        no_kv_times.append(t)
        print(f"  - No KV Cache: {t:.4f} seconds")

        # Benchmark single_batch with KV cache
        t = benchmark_single_batch(single_batch_engine, input_ids_tensor, out_len)
        single_batch_times.append(t)
        print(f"  - With KV Cache: {t:.4f} seconds")

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 7))

    plt.plot(
        output_lengths,
        no_kv_times,
        "o-",
        label="no_kv.py (Without KV Cache)",
        color="crimson",
    )
    plt.plot(
        output_lengths,
        single_batch_times,
        "s-",
        label="single_batch.py (With KV Cache)",
        color="steelblue",
    )

    plt.title(
        f"KV Cache Benchmark (Input Length: {input_length} tokens)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Number of Generated Tokens (Output Length)", fontsize=12)
    plt.ylabel("End-to-end Generation Time (seconds)", fontsize=12)
    plt.legend(fontsize=11)
    plt.xticks(np.arange(128, 2049, 128), rotation=45)
    plt.tight_layout()

    plot_filename = "kv_cache_benchmark.png"
    plt.savefig(plot_filename)

    print("-" * 30)
    print(
        f"\nBenchmark complete. Plot saved to '{plot_filename}' in the current directory."
    )

    print("-" * 30)
    print(
        f"\nBenchmark complete. Plot saved to '{plot_filename}' in the current directory."
    )

    print(f"\nBenchmark complete. Plot saved to '{plot_filename}' in the current directory.")

if __name__ == "__main__":
    main()
