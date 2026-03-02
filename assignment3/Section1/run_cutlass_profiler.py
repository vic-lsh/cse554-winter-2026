#!/usr/bin/env python3
"""Run CUTLASS profiler for all GEMM shapes and merge results with gemm_perf.csv."""

import subprocess
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILER = os.path.join(SCRIPT_DIR, "cutlass", "build", "tools", "profiler", "cutlass_profiler")
# CUTLASS profiler inserts ".gemm." before the extension, so we use this base name
RAW_CSV_BASE = os.path.join(SCRIPT_DIR, "cutlass_raw.csv")
RAW_CSV = os.path.join(SCRIPT_DIR, "cutlass_raw.gemm.csv")  # actual file written by profiler
PERF_CSV = os.path.join(SCRIPT_DIR, "gemm_perf.csv")
GPU_DEVICE = 1  # use a free GPU

SHAPES = [
    (512, 512),
    (4096, 4096),
    (14336, 4096),
    (4096, 1024),
    (1024, 4096),
]

SPLIT_K_VALUES = [1, 2, 4, 8]
M_RANGE = "128:2048:128"

def run_profiler():
    # Remove old raw CSV if exists
    for f in [RAW_CSV_BASE, RAW_CSV]:
        if os.path.exists(f):
            os.remove(f)

    total_runs = len(SHAPES) * len(SPLIT_K_VALUES)
    run_idx = 0

    for N, K in SHAPES:
        for sk in SPLIT_K_VALUES:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] N={N}, K={K}, split_k={sk}...")

            cmd = [
                PROFILER,
                "--providers=cutlass",
                "--operation=gemm",
                "--gemm_kind=universal",
                f"--m={M_RANGE}",
                f"--n={N}",
                f"--k={K}",
                "--A=f16:column",
                "--B=f16:column",
                "--C=f16:column",
                "--D=f16:column",
                "--accum=f32",
                "--alpha=1",
                "--beta=0",
                "--split_k_mode=serial",
                f"--split_k_slices={sk}",
                "--profiling-iterations=100",
                "--warmup-iterations=5",
                f"--output={RAW_CSV_BASE}",
                "--append=true",
                f"--device={GPU_DEVICE}",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR (rc={result.returncode})")
                print(result.stderr[-500:] if result.stderr else "")
            else:
                print(f"  Done")

def parse_and_merge():
    print("\nParsing CUTLASS results...")
    raw = pd.read_csv(RAW_CSV)

    # Print columns for debugging
    print(f"  Raw CSV columns: {list(raw.columns)}")
    print(f"  Raw CSV rows: {len(raw)}")

    # The CUTLASS profiler CSV uses columns like: m, n, k, GFLOPs, Runtime
    # Find the GFLOPs column (may have varying names)
    gflops_col = None
    for col in raw.columns:
        if 'gflop' in col.lower() and 'byte' not in col.lower():
            gflops_col = col
            break

    if gflops_col is None:
        print("  ERROR: Could not find GFLOPs column in output!")
        print(f"  Available columns: {list(raw.columns)}")
        return

    print(f"  Using GFLOPs column: '{gflops_col}'")

    # Filter out failed runs (Status != success or GFLOPs == 0)
    if 'Status' in raw.columns:
        raw = raw[raw['Status'].str.lower() == 'success']
    raw = raw[raw[gflops_col] > 0]

    # Group by (m, n, k) and take the best GFLOPs across all kernels and split_k values
    best = raw.groupby(['m', 'n', 'k'])[gflops_col].max().reset_index()
    best['tflops'] = best[gflops_col] / 1000.0
    best['library'] = 'cutlass'
    best = best.rename(columns={'m': 'M', 'n': 'N', 'k': 'K'})

    print(f"  Best results: {len(best)} unique (M, N, K) shapes")
    print(f"  TFLOPS range: {best['tflops'].min():.2f} - {best['tflops'].max():.2f}")

    # Load existing gemm_perf.csv and append
    existing = pd.read_csv(PERF_CSV)
    # Remove any previous cutlass rows
    existing = existing[existing['library'] != 'cutlass']

    combined = pd.concat([existing, best[['M', 'N', 'K', 'library', 'tflops']]], ignore_index=True)
    combined.to_csv(PERF_CSV, index=False)
    print(f"\n  Written {len(combined)} rows to {PERF_CSV}")
    print(f"    cublas: {len(combined[combined['library'] == 'cublas'])} rows")
    print(f"    cutlass: {len(combined[combined['library'] == 'cutlass'])} rows")

if __name__ == "__main__":
    run_profiler()
    parse_and_merge()
