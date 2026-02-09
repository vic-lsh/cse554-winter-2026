import time

import matplotlib.pyplot as plt
import torch
from uniform_prefill import Engine

def benchmark():
    batch_sizes = [2**i for i in range(7)]
    generation_times = []
    throughputs = []
    input_len = 512
    output_len = 128

    print("Loading engine...")
    engine = Engine()
    print("Engine loaded.")

    for bs in batch_sizes:
        print(f"Benchmarking with batch size: {bs}")
        # Create dummy input
        input_ids = torch.ones((bs, input_len), dtype=torch.long, device="cuda")

        # Warmup before timing
        _ = engine.run(input_ids, prefill=True)
        for _ in range(5):
            _ = engine.run(
                torch.ones((bs, 1), dtype=torch.long, device="cuda"), prefill=False
            )

        # Timed run
        torch.cuda.synchronize()
        start_time = time.time()

        # Prefill
        next_token = engine.run(input_ids, prefill=True)

        # Decode
        for _ in range(output_len - 1):
            next_token = engine.run(next_token, prefill=False)

        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        generation_times.append(total_time)

        # Throughput: (total tokens generated) / time
        # Total tokens = batch_size * output_len
        throughput = (bs * output_len) / total_time
        throughputs.append(throughput)

        print(
            f"Batch size: {bs}, Time: {total_time:.3f}s, Throughput: {throughput:.2f} tokens/s"
        )

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, generation_times, marker="o")
    plt.title("Generation Time vs. Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Generation Time (s)")
    plt.xscale("log", base=2)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, throughputs, marker="o")
    plt.title("Throughput vs. Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (tokens/s)")
    plt.xscale("log", base=2)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("bench_2_plots.png")
    print("Plots saved to assignment2/bench_2_plots.png")

    plt.savefig("bench_2_plots.png")
    print("Plots saved to assignment2/bench_2_plots.png")

if __name__ == "__main__":
    benchmark()
