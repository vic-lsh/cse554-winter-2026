import random
import torch

from continous_engine import Engine
from continous_scheduler import CBScheduler, InputRequest, NaiveScheduler

# Input and output length gen
NUM_REQS = 100
def gen_data():
    sample_inputs = []
    sample_output_lengths = []
    for _ in range(NUM_REQS):
        input_len = random.randint(1, 10)
        output_len = random.randint(1, 128)

        prompt = "q " * input_len
        sample_inputs.append(prompt)
        sample_output_lengths.append(output_len)
    return sample_inputs, sample_output_lengths

def naive_time(engine, sample_inputs, sample_output_lengths):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    naive_scheduler = NaiveScheduler(engine, req_batch_size=10)
    for prompt, output_len in zip(sample_inputs, sample_output_lengths):
        naive_scheduler.add_req(InputRequest(prompt, output_len=output_len))

    start.record()
    while not naive_scheduler.finished():
        naive_scheduler.run()

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000  # Convert milliseconds to seconds

def cb_time(engine, sample_inputs, sample_output_lengths):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    scheduler = CBScheduler(engine, req_batch_size=10)
    start.record()
    for prompt, output_len in zip(sample_inputs, sample_output_lengths):
        scheduler.add_req(InputRequest(prompt, output_len=output_len))

    # Drain remaining requests
    start.record()

    while not scheduler.finished():
        scheduler.run()

    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000  # Convert milliseconds to seconds

if __name__ == "__main__":
    engine = Engine()
    sample_inputs, sample_output_lengths = gen_data()
    cb_time_val = cb_time(engine, sample_inputs, sample_output_lengths)

    del engine 
    torch.cuda.empty_cache()  # Clear GPU memory before running naive scheduler
    engine = Engine()  # Re-initialize engine for fair timing
    naive_time_val = naive_time(engine, sample_inputs, sample_output_lengths)
    print(f"Total time taken for continuous scheduling: {cb_time_val:.2f} seconds")
    print(f"Total time taken for naive scheduling: {naive_time_val:.2f} seconds")
    speedup = naive_time_val / cb_time_val if cb_time_val > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}")