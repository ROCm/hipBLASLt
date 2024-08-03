#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>

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

template <typename Ta,
          typename Tb,
          typename Tc,
          typename Tcompute,
          typename Tscale>
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
    std::vector<Tc> h_c, h_d;

    hipStream_t stream;

    hipblasLtHalf h_a_scale;
    hipblasLtHalf* d_a_scale;
    hipblasLtHalf h_b_scale;
    hipblasLtHalf* d_b_scale;

    ScaleRunner(int64_t m,
           int64_t n,
           int64_t k,
           int64_t batch_count,
           Tcompute alpha,
           Tcompute beta,
           int64_t max_workspace_size)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

        size_t size_a = m * k * batch_count;
        size_t size_b = k * n * batch_count;
        size_t size_c = m * n * batch_count;

        h_a.resize(size_a);
        h_b.resize(size_b);
        h_c.resize(size_c);
        h_d.resize(size_c);

        // Initialize host matrices
        std::fill(h_a.begin(), h_a.end(), static_cast<Ta>(1));
        std::fill(h_b.begin(), h_b.end(), static_cast<Tb>(1));
        std::fill(h_c.begin(), h_c.end(), static_cast<Tc>(0));
        std::fill(h_d.begin(), h_d.end(), static_cast<Tc>(0));

        CHECK_HIP_ERROR(hipMalloc(&d_a, size_a * sizeof(Ta)));
        CHECK_HIP_ERROR(hipMalloc(&d_b, size_b * sizeof(Tb)));
        CHECK_HIP_ERROR(hipMalloc(&d_c, size_c * sizeof(Tc)));
        CHECK_HIP_ERROR(hipMalloc(&d_d, size_c * sizeof(Tc)));
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
        CHECK_HIP_ERROR(hipMalloc(&d_a_scale, sizeof(hipblasLtHalf)));
        CHECK_HIP_ERROR(hipMalloc(&d_b_scale, sizeof(hipblasLtHalf)));

        CHECK_HIP_ERROR(hipMemcpy(d_a, h_a.data(), size_a * sizeof(Ta), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_b, h_b.data(), size_b * sizeof(Tb), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_c, h_c.data(), size_c * sizeof(Tc), hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(hipStreamCreate(&stream));

        // Initialize scale factors
        h_a_scale = static_cast<hipblasLtHalf>(1.0f);
        h_b_scale = static_cast<hipblasLtHalf>(1.0f);
        CHECK_HIP_ERROR(hipMemcpy(d_a_scale, &h_a_scale, sizeof(hipblasLtHalf), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_b_scale, &h_b_scale, sizeof(hipblasLtHalf), hipMemcpyHostToDevice));

        std::cout << "Initialized A scale factor: " << static_cast<float>(h_a_scale) << std::endl;
        std::cout << "Initialized B scale factor: " << static_cast<float>(h_b_scale) << std::endl;
    }

    ~ScaleRunner()
    {
        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_c));
        CHECK_HIP_ERROR(hipFree(d_d));
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIP_ERROR(hipFree(d_a_scale));
        CHECK_HIP_ERROR(hipFree(d_b_scale));

        CHECK_HIP_ERROR(hipStreamDestroy(stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    }

    void run(std::function<void()> func)
    {
        // Run the GEMM operation
        func();

        // Copy result back to host
        CHECK_HIP_ERROR(hipMemcpy(h_d.data(), d_d, h_d.size() * sizeof(Tc), hipMemcpyDeviceToHost));

        // Verify the result
        verify();

        // Print the first few elements of the result
        std::cout << "Result (first 10 elements): ";
        for (int i = 0; i < std::min(10, static_cast<int>(h_d.size())); ++i)
        {
            std::cout << static_cast<float>(h_d[i]) << " ";
        }
        std::cout << std::endl;
    }

    void verify()
    {
        double max_error = 0.0;
        for (int64_t i = 0; i < m; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                double expected = 0.0;
                for (int64_t l = 0; l < k; ++l)
                {
                    expected += static_cast<double>(h_a[i * k + l]) * static_cast<double>(h_b[l * n + j]);
                }
                expected *= static_cast<double>(h_a_scale) * static_cast<double>(h_b_scale);
                expected = alpha * expected + beta * static_cast<double>(h_c[i * n + j]);
                
                double actual = static_cast<double>(h_d[i * n + j]);
                double error = std::abs(expected - actual);
                max_error = std::max(max_error, error);

                if (i < 5 && j < 5) {
                    std::cout << "Element (" << i << "," << j << "): "
                              << "Expected " << expected << ", Actual " << actual << std::endl;
                }
            }
        }

        std::cout << "Max error: " << max_error << std::endl;
        if (max_error > 1e-3)
        {
            std::cerr << "Verification failed!" << std::endl;
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
    CHECK_HIP_ERROR(hipMemcpyAsync(h_matrix.data(),
                                   d_matrix,
                                   rows * cols * sizeof(hipblasLtHalf),
                                   hipMemcpyDeviceToHost,
                                   stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    int print_count = std::min(rows * cols, int64_t(10));
    for(int64_t i = 0; i < print_count; ++i)
    {
        std::cout << static_cast<float>(h_matrix[i]) << " ";
    }
    std::cout << std::endl;
}