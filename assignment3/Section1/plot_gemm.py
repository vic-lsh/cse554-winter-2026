import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
# M,N,K,library,tflops
df = pd.read_csv('gemm_perf.csv')

# Get all unique (N, K) shapes
shapes = df[['N', 'K']].drop_duplicates().values.tolist()

# Plot for each shape
for N, K in shapes:
    shape_df = df[(df['N'] == N) & (df['K'] == K)]

    plt.figure(figsize=(8, 5))

    for lib in shape_df['library'].unique():
        lib_df = shape_df[shape_df['library'] == lib].sort_values('M')
        plt.plot(lib_df['M'], lib_df['tflops'], marker='o', label=lib.upper())

    plt.title(f'GEMM Performance: N={int(N)}, K={int(K)}')
    plt.xlabel('M')
    plt.ylabel('TFLOPS')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'gemm_perf_N{int(N)}_K{int(K)}.png', dpi=300)
    plt.close()

print("Plots saved.")
