from typing import Dict, List, Optional, Tuple

import torch

from chunked_engine import Engine as ChunkEngine
from chunked_scheduler import ChunkScheduler, InputRequest as ChunkInputRequest
from continous_engine import Engine as CBEngine
from continous_scheduler import CBScheduler, InputRequest as CBInputRequest


NUM_REQS = 100
INPUT_LEN = 512
OUTPUT_LEN = 512
TOKEN_BATCH_SIZE = 512
CB_REQ_BATCH_SIZE = 10


def gen_fixed_data(
    num_reqs: int = NUM_REQS,
    input_len: int = INPUT_LEN,
    output_len: int = OUTPUT_LEN,
) -> Tuple[List[str], List[int]]:
    prompt = "q " * input_len
    return [prompt for _ in range(num_reqs)], [output_len for _ in range(num_reqs)]


def safe_reset_cache(engine) -> None:
    if hasattr(engine, "kv_cache_map"):
        for kv in list(engine.kv_cache_map.values()):
            if hasattr(kv, "release") and callable(kv.release):
                kv.release()
        engine.kv_cache_map.clear()
    torch.cuda.synchronize()


def chunked_time(
    engine: ChunkEngine,
    sample_inputs: List[str],
    sample_output_lengths: List[int],
    collect_iteration_breakdown: bool = False,
) -> Tuple[float, Optional[Dict[str, float]]]:
    scheduler = ChunkScheduler(engine, token_batch_size=TOKEN_BATCH_SIZE)

    for prompt, output_len in zip(sample_inputs, sample_output_lengths):
        scheduler.add_req(ChunkInputRequest(prompt, output_len=output_len))

    total_breakdown: Dict[str, float] = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    while not scheduler.finished():
        scheduler.run(enable_timing=collect_iteration_breakdown)
        if collect_iteration_breakdown:
            for name, ms in getattr(engine, "last_run_breakdown", {}).items():
                total_breakdown[name] = total_breakdown.get(name, 0.0) + ms
    end.record()

    torch.cuda.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0
    return elapsed_s, total_breakdown if collect_iteration_breakdown else None


def cb_time(
    engine: CBEngine,
    sample_inputs: List[str],
    sample_output_lengths: List[int],
) -> float:
    scheduler = CBScheduler(engine, req_batch_size=CB_REQ_BATCH_SIZE)

    for prompt, output_len in zip(sample_inputs, sample_output_lengths):
        scheduler.add_req(CBInputRequest(prompt, output_len=output_len))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    while not scheduler.finished():
        scheduler.run()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0


def print_stage_breakdown(
    breakdown: Optional[Dict[str, float]],
    title: str,
    top_k: Optional[int] = None,
) -> None:
    if not breakdown:
        return

    total_ms = breakdown.get("total", 0.0)
    rows = [(name, ms) for name, ms in breakdown.items() if name != "total"]
    rows.sort(key=lambda item: item[1], reverse=True)
    if top_k is not None:
        rows = rows[:top_k]

    print(title)
    for name, ms in rows:
        pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
        print(f"  {name:20s} {ms:9.2f} ms ({pct:5.1f}%)")
    print(f"  {'total':20s} {total_ms:9.2f} ms")


def main() -> None:
    sample_inputs, sample_output_lengths = gen_fixed_data()

    chunk_engine = ChunkEngine()
    safe_reset_cache(chunk_engine)
    chunked_time_val, chunked_breakdown = chunked_time(
        chunk_engine,
        sample_inputs,
        sample_output_lengths,
        collect_iteration_breakdown=True,
    )

    print(
        f"Chunked scheduler total time: {chunked_time_val:.2f} seconds "
        f"for {NUM_REQS} requests, input_len={INPUT_LEN}, output_len={OUTPUT_LEN}"
    )
    print_stage_breakdown(
        chunked_breakdown,
        title="\n[Chunked scheduler stage breakdown]",
    )

    del chunk_engine
    torch.cuda.empty_cache()

    cb_engine = CBEngine()
    safe_reset_cache(cb_engine)
    cb_time_val = cb_time(cb_engine, sample_inputs, sample_output_lengths)

    print(f"\nContinuous batching total time: {cb_time_val:.2f} seconds")
    ratio = cb_time_val / chunked_time_val if chunked_time_val > 0 else float("inf")
    print(f"Continuous batching / chunked ratio: {ratio:.2f}")


if __name__ == "__main__":
    main()
