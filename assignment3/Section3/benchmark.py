import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append(str(Path(__file__).resolve().parent.parent))
from flashinfer_pipeline import Request, Engine  # noqa: E402


# -----------------------------
# Helpers
# -----------------------------
def safe_reset_cache(engine: Engine) -> None:
    """Use engine.reset_cache() if present, otherwise clear kv_cache_map manually."""
    if hasattr(engine, "reset_cache") and callable(engine.reset_cache):
        engine.reset_cache()
        torch.cuda.synchronize()
        return

    if hasattr(engine, "kv_cache_map"):
        for kv in list(engine.kv_cache_map.values()):
            if hasattr(kv, "release") and callable(kv.release):
                kv.release()
        engine.kv_cache_map.clear()

    torch.cuda.synchronize()


def make_requests(engine: Engine, batch_size: int, prefill_len: int, decode_len: int) -> List[Request]:
    vocab_size = getattr(engine.tokenizer, "vocab_size", None) or 128256
    reqs: List[Request] = []
    for i in range(batch_size):
        dummy_ids = torch.randint(0, vocab_size, (prefill_len,), dtype=torch.long)
        reqs.append(Request(i, dummy_ids, decode_len))
    return reqs


def append_dummy_token(requests: List[Request]) -> None:
    """Append one CPU token so the next decode step sees a longer history."""
    for r in requests:
        tok = torch.zeros(
            1,
            dtype=r.output_token_ids.dtype,
            device=r.output_token_ids.device,
        )
        r.output_token_ids = torch.cat([r.output_token_ids, tok], dim=0)


def time_one_run(
    engine: Engine,
    requests: List[Request],
    num_decode_req: int,
    collect_breakdown: bool = False,
) -> Tuple[float, torch.Tensor, Optional[Dict[str, float]]]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = engine.run(requests, num_decode_req=num_decode_req, enable_timing=collect_breakdown)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    breakdown = dict(getattr(engine, "last_run_breakdown", {})) if collect_breakdown else None
    return ms, out, breakdown


def warmup_engine(engine: Engine) -> None:
    """Warm up CUDA / kernels once before benchmarking."""
    safe_reset_cache(engine)
    reqs = make_requests(engine, batch_size=1, prefill_len=8, decode_len=2)

    _, _, _ = time_one_run(engine, reqs, num_decode_req=0)
    append_dummy_token(reqs)

    _, _, _ = time_one_run(engine, reqs, num_decode_req=1)
    torch.cuda.synchronize()
    safe_reset_cache(engine)


# -----------------------------
# Timed benchmark (used for plots)
# -----------------------------
def benchmark_case(
    engine: Engine,
    batch_size: int,
    prefill_len: int,
    decode_len: int,
    collect_prefill_breakdown: bool = False,
):
    """
    Measures:
      - prefill time (one prefill pass)
      - total decode time (sum of all decode passes after prefill)
      - e2e time = prefill + decode
    Assumes decode_len is the total number of generated tokens requested.
    The prefill pass generates the first token, so decode runs decode_len - 1 times.
    """
    safe_reset_cache(engine)
    requests = make_requests(engine, batch_size, prefill_len, decode_len)

    # Prefill
    prefill_ms, _, prefill_breakdown = time_one_run(
        engine,
        requests,
        num_decode_req=0,
        collect_breakdown=collect_prefill_breakdown,
    )
    append_dummy_token(requests)

    # Decode
    total_decode_ms = 0.0
    for _ in range(max(decode_len - 1, 0)):
        step_ms, _, _ = time_one_run(engine, requests, num_decode_req=batch_size)
        total_decode_ms += step_ms
        append_dummy_token(requests)

    e2e_ms = prefill_ms + total_decode_ms
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": total_decode_ms,
        "e2e_ms": e2e_ms,
        "prefill_breakdown": prefill_breakdown,
    }


# -----------------------------
# Profiling helpers (used only for breakdown inspection)
# -----------------------------
def profile_prefill_breakdown(engine: Engine, batch_size: int, prefill_len: int, row_limit: int = 15) -> None:
    safe_reset_cache(engine)
    requests = make_requests(engine, batch_size=batch_size, prefill_len=prefill_len, decode_len=1)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function(f"prefill_bs{batch_size}_len{prefill_len}"):
            engine.run(requests, num_decode_req=0)

    torch.cuda.synchronize()
    print(f"\n[Prefill breakdown] batch_size={batch_size}, prefill_len={prefill_len}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))


def profile_last_decode_breakdown(
    engine: Engine,
    batch_size: int,
    prefill_len: int,
    decode_len: int,
    row_limit: int = 15,
) -> None:
    if decode_len < 2:
        return

    safe_reset_cache(engine)
    requests = make_requests(engine, batch_size=batch_size, prefill_len=prefill_len, decode_len=decode_len)

    # Run prefill (first generated token)
    _, _, _ = time_one_run(engine, requests, num_decode_req=0)
    append_dummy_token(requests)

    # Run decode_len - 2 decode steps normally
    for _ in range(decode_len - 2):
        _, _, _ = time_one_run(engine, requests, num_decode_req=batch_size)
        append_dummy_token(requests)

    # Profile only the final decode step
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with record_function(f"last_decode_bs{batch_size}_prefill{prefill_len}_decode{decode_len}"):
            engine.run(requests, num_decode_req=batch_size)

    torch.cuda.synchronize()
    print(f"\n[Last decode breakdown] batch_size={batch_size}, prefill_len={prefill_len}, decode_len={decode_len}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))


# -----------------------------
# Plot helpers
# -----------------------------
def save_target1_plot(decode_lengths, prefill_ms, decode_ms, e2e_ms, out_path="target1_decode_scaling.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(decode_lengths, prefill_ms, marker="o", label="Prefill time")
    plt.plot(decode_lengths, decode_ms, marker="o", label="Total decode time")
    plt.plot(decode_lengths, e2e_ms, marker="o", label="End-to-end time")
    plt.xscale("log", base=2)
    plt.xlabel("Decode length (log2 scale)")
    plt.ylabel("Time (ms)")
    plt.title("Target 1: Decode length scaling")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_target2_plot(prefill_lengths, prefill_ms, out_path="target2_prefill_scaling.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(prefill_lengths, prefill_ms, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Prefill length (log2 scale)")
    plt.ylabel("Prefill time (ms)")
    plt.title("Target 2: Prefill length scaling")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_target3_time_plot(batch_sizes, e2e_ms, out_path="target3_batch_scaling_time.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(batch_sizes, e2e_ms, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Batch size (log2 scale)")
    plt.ylabel("End-to-end time (ms)")
    plt.title("Target 3: Batch size scaling (E2E time)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_target3_throughput_plot(batch_sizes, throughput, out_path="target3_batch_scaling_throughput.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(batch_sizes, throughput, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Batch size (log2 scale)")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Target 3: Batch size scaling (throughput)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------
def print_stage_breakdown(breakdown: Optional[Dict[str, float]], top_k: int = 6) -> None:
    if not breakdown:
        return

    total_ms = breakdown.get("total", 0.0)
    rows = [(k, v) for k, v in breakdown.items() if k != "total"]
    rows.sort(key=lambda kv: kv[1], reverse=True)
    rows = rows[:top_k]

    print(f"  stage breakdown (top {top_k}):")
    for name, ms in rows:
        pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
        print(f"    {name:20s} {ms:9.2f} ms ({pct:5.1f}%)")
    print(f"    {'total(run stages)':20s} {total_ms:9.2f} ms")


def run_target1(engine: Engine, with_profiler: bool = False) -> None:
    print("\n--- Target 1: Decode Length Scaling ---")
    decode_lengths = [2**i for i in range(5, 11)]
    t1_prefill_ms = []
    t1_decode_ms = []
    t1_e2e_ms = []

    for dl in decode_lengths:
        stats = benchmark_case(engine, batch_size=32, prefill_len=256, decode_len=dl)
        t1_prefill_ms.append(stats["prefill_ms"])
        t1_decode_ms.append(stats["decode_ms"])
        t1_e2e_ms.append(stats["e2e_ms"])
        print(
            f"decode_len={dl:4d} | "
            f"prefill={stats['prefill_ms']:.2f} ms | "
            f"total_decode={stats['decode_ms']:.2f} ms | "
            f"e2e={stats['e2e_ms']:.2f} ms"
        )

    save_target1_plot(decode_lengths, t1_prefill_ms, t1_decode_ms, t1_e2e_ms)

    if with_profiler:
        for dl in decode_lengths:
            profile_last_decode_breakdown(engine, batch_size=32, prefill_len=256, decode_len=dl)


def run_target2(engine: Engine, with_profiler: bool = False, with_stage_breakdown: bool = True) -> None:
    print("\n--- Target 2: Prefill Length Scaling ---")
    prefill_lengths = [2**i for i in range(8, 15)]
    t2_prefill_ms = []

    for pl in prefill_lengths:
        stats = benchmark_case(
            engine,
            batch_size=1,
            prefill_len=pl,
            decode_len=1,
            collect_prefill_breakdown=with_stage_breakdown,
        )
        t2_prefill_ms.append(stats["prefill_ms"])
        print(f"prefill_len={pl:5d} | prefill={stats['prefill_ms']:.2f} ms")
        if with_stage_breakdown:
            print_stage_breakdown(breakdown=stats["prefill_breakdown"])

    save_target2_plot(prefill_lengths, t2_prefill_ms)

    if with_profiler:
        for pl in prefill_lengths:
            profile_prefill_breakdown(engine, batch_size=1, prefill_len=pl)


def run_target3(engine: Engine) -> None:
    print("\n--- Target 3: Batch Size Scaling ---")
    batch_sizes = [2**i for i in range(0, 9)]
    t3_e2e_ms = []
    t3_throughput = []

    prefill_len = 128
    decode_len = 128

    for bs in batch_sizes:
        stats = benchmark_case(engine, batch_size=bs, prefill_len=prefill_len, decode_len=decode_len)

        total_tokens = bs * (prefill_len + decode_len)
        total_time_s = stats["e2e_ms"] / 1000.0
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

        t3_e2e_ms.append(stats["e2e_ms"])
        t3_throughput.append(throughput)

        print(
            f"batch_size={bs:3d} | "
            f"e2e={stats['e2e_ms']:.2f} ms | "
            f"throughput={throughput:.2f} tok/s"
        )

    save_target3_time_plot(batch_sizes, t3_e2e_ms)
    save_target3_throughput_plot(batch_sizes, t3_throughput)


def parse_args():
    parser = argparse.ArgumentParser(description="Section 3 benchmark runner")
    parser.add_argument(
        "--target",
        choices=["1", "2", "3", "all"],
        default="2",
        help="Which target to run. Default runs only Target 2.",
    )
    parser.add_argument(
        "--with-profiler",
        action="store_true",
        help="Enable torch.profiler breakdown runs for selected target(s).",
    )
    parser.add_argument(
        "--no-stage-breakdown",
        action="store_true",
        help="Disable run()-stage timing breakdown for Target 2.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    engine = Engine()
    warmup_engine(engine)

    if args.target in ("1", "all"):
        run_target1(engine, with_profiler=args.with_profiler)
    if args.target in ("2", "all"):
        run_target2(
            engine,
            with_profiler=args.with_profiler,
            with_stage_breakdown=not args.no_stage_breakdown,
        )
    if args.target in ("3", "all"):
        run_target3(engine)

    print("\nSaved:")
    if args.target in ("1", "all"):
        print("  target1_decode_scaling.png")
    if args.target in ("2", "all"):
        print("  target2_prefill_scaling.png")
    if args.target in ("3", "all"):
        print("  target3_batch_scaling_time.png")
        print("  target3_batch_scaling_throughput.png")
