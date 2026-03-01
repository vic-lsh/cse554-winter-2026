
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(err) << std::endl;      \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status = (call);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__   \
                      << " - status: " << status << std::endl;               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

static const int WARMUP_ITERS = 100;
static const int PROFILE_ITERS = 100;
static const size_t L2_FLUSH_SIZE = 50 * 1024 * 1024; // 50 MB

double profile_gemm(cublasHandle_t handle, int M, int N, int K,
                    int* clear_l2_buffer) {
    size_t size_A = (size_t)M * K * sizeof(__half);
    size_t size_B = (size_t)K * N * sizeof(__half);
    size_t size_C = (size_t)M * N * sizeof(__half);

    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Initialize with zeros (content doesn't affect timing)
    CUDA_CHECK(cudaMemset(d_A, 0, size_A));
    CUDA_CHECK(cudaMemset(d_B, 0, size_B));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, CUDA_R_16F, M,
            d_B, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Profiling with L2 cache flush
    float total_ms = 0;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < PROFILE_ITERS; ++i) {
        CUDA_CHECK(cudaMemset(clear_l2_buffer, 0, L2_FLUSH_SIZE));
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, CUDA_R_16F, M,
            d_B, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    double avg_ms = total_ms / PROFILE_ITERS;
    double tflops = (2.0 * M * N * K) / (avg_ms * 1e-3) / 1e12;
    return tflops;
}

int main() {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int* clear_l2_buffer;
    CUDA_CHECK(cudaMalloc(&clear_l2_buffer, L2_FLUSH_SIZE));

    // M values: 128 to 2048, step 128
    std::vector<int> M_values;
    for (int m = 128; m <= 2048; m += 128) {
        M_values.push_back(m);
    }

    // (N, K) shapes
    struct Shape { int N, K; };
    std::vector<Shape> shapes = {
        {512, 512}, {4096, 4096}, {14336, 4096}, {4096, 1024}, {1024, 4096}
    };

    std::ofstream csv("gemm_perf.csv");
    csv << "M,N,K,library,tflops" << std::endl;

    for (const auto& shape : shapes) {
        for (int M : M_values) {
            std::cout << "Profiling M=" << M
                      << ", N=" << shape.N
                      << ", K=" << shape.K << "..." << std::flush;

            double tflops = profile_gemm(handle, M, shape.N, shape.K,
                                         clear_l2_buffer);

            csv << M << "," << shape.N << "," << shape.K
                << ",cublas," << tflops << std::endl;

            std::cout << " " << tflops << " TFLOPS" << std::endl;
        }
    }

    csv.close();
    CUDA_CHECK(cudaFree(clear_l2_buffer));
    CUBLAS_CHECK(cublasDestroy(handle));

    std::cout << "Results written to gemm_perf.csv" << std::endl;
    return 0;
}
