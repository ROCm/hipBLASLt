#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <chrono>

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_HIPBLASLT_ERROR(error) \
    if (error != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLASLt error: '%d' at %s:%d\n", error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

template <typename Ta, typename Tb, typename Tc, typename Tcompute, typename Tscale>
class ScaleRunner
{
public:
    hipblasLtHandle_t handle;
    int64_t m, n, k;
    int64_t batch_count;
    Tcompute alpha, beta;
    int64_t max_workspace_size;

    Ta *d_a;
    Tb *d_b;
    Tc *d_c, *d_d;
    void *d_workspace;

    std::vector<Ta> h_a;
    std::vector<Tb> h_b;
    std::vector<Tc> h_c, h_d, h_cpu_result;

    hipStream_t stream;

    hipblasLtHalf h_b_scale;
    hipblasLtHalf* d_b_scale;

    ScaleRunner(int64_t m, int64_t n, int64_t k, int64_t batch_count,
                Tcompute alpha, Tcompute beta, int64_t max_workspace_size)
        : m(m), n(n), k(k), batch_count(batch_count), alpha(alpha), beta(beta),
          max_workspace_size(max_workspace_size)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

        size_t size_a = m * k * batch_count;
        size_t size_b = k * n * batch_count;
        size_t size_c = m * n * batch_count;

        h_a.resize(size_a, static_cast<Ta>(1));
        h_b.resize(size_b, static_cast<Tb>(1));
        h_c.resize(size_c, static_cast<Tc>(0));
        h_d.resize(size_c);
        h_cpu_result.resize(size_c);

        CHECK_HIP_ERROR(hipMalloc(&d_a, size_a * sizeof(Ta)));
        CHECK_HIP_ERROR(hipMalloc(&d_b, size_b * sizeof(Tb)));
        CHECK_HIP_ERROR(hipMalloc(&d_c, size_c * sizeof(Tc)));
        CHECK_HIP_ERROR(hipMalloc(&d_d, size_c * sizeof(Tc)));
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
        CHECK_HIP_ERROR(hipMalloc(&d_b_scale, sizeof(hipblasLtHalf)));

        CHECK_HIP_ERROR(hipMemcpy(d_a, h_a.data(), size_a * sizeof(Ta), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_b, h_b.data(), size_b * sizeof(Tb), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_c, h_c.data(), size_c * sizeof(Tc), hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(hipStreamCreate(&stream));

        h_b_scale = static_cast<hipblasLtHalf>(2.0f);
        CHECK_HIP_ERROR(hipMemcpy(d_b_scale, &h_b_scale, sizeof(hipblasLtHalf), hipMemcpyHostToDevice));

        std::cout << "Initialized B scale factor: " << static_cast<float>(h_b_scale) << std::endl;
        std::cout << "----------------------------" << std::endl;
    }

    ~ScaleRunner()
    {
        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_c));
        CHECK_HIP_ERROR(hipFree(d_d));
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIP_ERROR(hipFree(d_b_scale));

        CHECK_HIP_ERROR(hipStreamDestroy(stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    }

    void run(std::function<void()> gpu_func)
    {
        auto start = std::chrono::high_resolution_clock::now();
        gpu_func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "GPU Computation Time: " << diff.count() << " s" << std::endl;

        CHECK_HIP_ERROR(hipMemcpy(h_d.data(), d_d, h_d.size() * sizeof(Tc), hipMemcpyDeviceToHost));

        start = std::chrono::high_resolution_clock::now();
        cpu_gemm();
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "CPU Computation Time: " << diff.count() << " s" << std::endl;

        verify();

        std::cout << "GPU Result (first 10 elements): ";
        for (int i = 0; i < std::min(10, static_cast<int>(h_d.size())); ++i)
        {
            std::cout << std::setprecision(6) << static_cast<float>(h_d[i]) << " ";
        }
        std::cout << std::endl;

        std::cout << "CPU Result (first 10 elements): ";
        for (int i = 0; i < std::min(10, static_cast<int>(h_cpu_result.size())); ++i)
        {
            std::cout << std::setprecision(6) << static_cast<float>(h_cpu_result[i]) << " ";
        }
        std::cout << std::endl;
    }

    void cpu_gemm()
    {
        for (int64_t i = 0; i < m; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                double temp = 0.0;
                for (int64_t l = 0; l < k; ++l)
                {
                    temp += static_cast<double>(h_a[i * k + l]) * static_cast<double>(h_b[l * n + j]);
                }
                temp *= static_cast<double>(h_b_scale);
                h_cpu_result[i * n + j] = static_cast<Tc>(alpha * temp + beta * static_cast<double>(h_c[i * n + j]));
            }
        }
    }

    void verify()
    {
        double max_error = 0.0;
        double avg_error = 0.0;
        int num_large_errors = 0;
        const double tolerance = 1e-3;

        for (int64_t i = 0; i < m * n; ++i)
        {
            double expected = static_cast<double>(h_cpu_result[i]);
            double actual = static_cast<double>(h_d[i]);
            double error = std::abs(expected - actual);
            max_error = std::max(max_error, error);
            avg_error += error;

            if (error > tolerance)
            {
                num_large_errors++;
                if (num_large_errors <= 10)
                {
                    std::cout << "Large error at index " << i << ": Expected " << expected 
                              << ", Actual " << actual << ", Error " << error << std::endl;
                }
            }
        }

        avg_error /= (m * n);

        std::cout << "Max error: " << max_error << std::endl;
        std::cout << "Average error: " << avg_error << std::endl;
        std::cout << "Number of errors exceeding tolerance: " << num_large_errors << std::endl;

        if (max_error > tolerance)
        {
            std::cout << "Verification failed: Max error exceeds tolerance." << std::endl;
        }
        else if (num_large_errors > 0)
        {
            std::cout << "Verification failed: " << num_large_errors << " errors exceed tolerance." << std::endl;
        }
        else
        {
            std::cout << "Verification passed." << std::endl;
        }
    }
};

void printMatrix(const void* d_matrix, int64_t rows, int64_t cols, hipStream_t stream)
{
    std::vector<hipblasLtHalf> h_matrix(rows * cols);
    CHECK_HIP_ERROR(hipMemcpyAsync(h_matrix.data(), d_matrix,
                                   rows * cols * sizeof(hipblasLtHalf),
                                   hipMemcpyDeviceToHost, stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    int print_count = std::min(rows * cols, int64_t(10));
    for(int64_t i = 0; i < print_count; ++i)
    {
        std::cout << std::setprecision(6) << static_cast<float>(h_matrix[i]) << " ";
    }
    std::cout << std::endl;
}

void simpleGemmWithBScale(hipblasLtHandle_t  handle,
                          hipblasOperation_t trans_a,
                          hipblasOperation_t trans_b,
                          int64_t            m,
                          int64_t            n,
                          int64_t            k,
                          int64_t            batch_count,
                          float&             alpha,
                          float&             beta,
                          void*              d_a,
                          void*              d_b,
                          void*              d_c,
                          void*              d_d,
                          void*              d_workspace,
                          int64_t            max_workspace_size,
                          hipStream_t        stream,
                          hipblasLtHalf*     d_b_scale)
{
    hipblasLtHalf h_b_scale_verify;
    CHECK_HIP_ERROR(hipMemcpy(&h_b_scale_verify, d_b_scale, sizeof(hipblasLtHalf), hipMemcpyDeviceToHost));
    std::cout << "B Scale factor on device: " << static_cast<float>(h_b_scale_verify) << std::endl;
    std::cout << "----------------------------" << std::endl;

    std::cout << "Matrix A before GEMM:" << std::endl;
    printMatrix(d_a, m, k, stream);
    std::cout << "Matrix B before GEMM:" << std::endl;
    printMatrix(d_b, k, n, stream);

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(hipblasLtHalf*)));

    hipblasLtHalf* scale_ptr_b = nullptr;
    size_t scale_ptr_size = sizeof(hipblasLtHalf*);
    size_t size_written;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescGetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &scale_ptr_b, scale_ptr_size, &size_written));
    std::cout << "B Scale pointer set: " << (scale_ptr_b == d_b_scale ? "Yes" : "No") << std::endl;
    std::cout << "----------------------------" << std::endl;

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmul,
                                          &alpha, d_a, matA, d_b, matB,
                                          &beta, d_c, matC, d_d, matD,
                                          nullptr, d_workspace, max_workspace_size, stream));

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    std::cout << "Matrix D after GEMM:" << std::endl;
    printMatrix(d_d, m, n, stream);
    std::cout << "----------------------------" << std::endl;

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
}

int main()
{
    ScaleRunner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hipblasLtHalf> runner(
        128, 128, 128, 1, 1.f, 0.f, 32 * 1024 * 1024);

    runner.run([&runner] {
        simpleGemmWithBScale(runner.handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                             runner.m, runner.n, runner.k, runner.batch_count,
                             runner.alpha, runner.beta,
                             runner.d_a, runner.d_b, runner.d_c, runner.d_d,
                             runner.d_workspace, runner.max_workspace_size,
                             runner.stream, runner.d_b_scale);
    });

    return 0;
}