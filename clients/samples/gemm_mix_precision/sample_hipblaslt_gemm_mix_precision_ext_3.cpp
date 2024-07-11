/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <iostream>
#include <random>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

using hipblaslt_rng_t = std::mt19937;

inline hipblaslt_rng_t get_seed()
{
    auto tid = std::this_thread::get_id();
    return hipblaslt_rng_t(std::hash<std::thread::id>{}(tid));
}

thread_local hipblaslt_rng_t t_hipblaslt_rng = get_seed();

template <typename T>
void init_hpl(T* A, size_t size)
{
    for(size_t i = 0; i < size; i++)
        A[i] = static_cast<T>(std::uniform_real_distribution<double>(-0.5, 0.5)(t_hipblaslt_rng));
}

template <typename Tci, typename Ti>
float getValue(Ti value, float scale)
{
    if (sizeof(Tci) < sizeof(Ti))
        return float(Tci(scale * float(value)));
    else
        return scale * float(value);
}

template <typename TypeA, typename TypeB, typename TypeCD, typename ComputeInputType, typename AlphaType, typename BetaType>
int CpuGemmExtWithAmaxScaleAB(int64_t            M,
                              int64_t            N,
                              int64_t            K,
                              int64_t            B,
                              void*              a,
                              void*              b,
                              void*              c,
                              void*              ref,
                              AlphaType          alpha,
                              BetaType           beta,
                              float              scaleA,
                              float              scaleB)
{
    const auto    batchStrideA = M * K;
    const auto    batchStrideB = K * N;
    const auto    batchStrideC = M * N;
    const auto    batchStrideD = M * N;
    const TypeA*  aPtr         = reinterpret_cast<const TypeA*>(a);
    const TypeB*  bPtr         = reinterpret_cast<const TypeB*>(b);
    const TypeCD* cPtr         = reinterpret_cast<const TypeCD*>(c);
          TypeCD* refPtr       = reinterpret_cast<      TypeCD*>(ref);

    for(int64_t b = 0; b < B; ++b)
    {
        for(int64_t i = 0; i < M; ++i)
        {
            for(int64_t j = 0; j < N; ++j)
            {
                float tmpRef = 0.0f;
                for(int64_t k = 0; k < K; ++k)
                {
                    float valueA = getValue<ComputeInputType, TypeA>(aPtr[batchStrideA * b + M * k + i], scaleA);
                    float valueB = getValue<ComputeInputType, TypeB>(bPtr[batchStrideB * b + K * j + k], scaleB);
                    tmpRef += (valueA * valueB);
                }

                float valueC = float(cPtr[batchStrideC * b + j * M + i]);
                refPtr[batchStrideD * b + j * M + i] = TypeCD(alpha * tmpRef + beta * valueC);
            }
        }
    }

    return 0;
}

template <typename TypeCD>
int validate(int64_t m,
             int64_t n,
             int64_t batch_count,
             void*   gpu,
             void*   ref)
{
    const auto batchStrideD = m * n;
    const TypeCD* gpuPtr = reinterpret_cast<const TypeCD*>(gpu);
    const TypeCD* refPtr = reinterpret_cast<const TypeCD*>(ref);

    for(int64_t b = 0; b < batch_count; ++b)
    {
        for(int64_t i = 0; i < m; ++i)
        {
            for(int64_t j = 0; j < n; ++j)
            {
                const auto lhs = float(gpuPtr[batchStrideD * b + j * m + i]);
                const auto rhs = float(refPtr[batchStrideD * b + j * m + i]);

                if(std::abs(lhs - rhs) > 1e-5)
                {
                    std::cout << lhs << " vs " << rhs << '\n';
                    return -1;
                }
            }
        }
    }

    return 0;
}

template <typename InTypeA, typename InTypeB, typename OutType, typename ComputeInputType, typename AlphaType, typename BetaType>
void GemmExtWithAmaxScaleAB(int64_t    m,
                            int64_t    n,
                            int64_t    k,
                            int64_t    batch_count,
                            AlphaType  alpha,
                            BetaType   beta,
                            float      cvtMax,
                            int64_t    max_workspace_size)
{
    hipStream_t       stream;
    hipblasLtHandle_t handle;
    void *a, *b, *c, *d, *ref;
    void *d_a, *d_b, *d_c, *d_d, *d_scaleA, *d_scaleB, *d_workspace, *d_sync; // device
    float scaleA, scaleB;

    // create stream and handle
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    // create device memory
    CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipMalloc(&d_scaleA, sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scaleB, sizeof(float)));
    if(max_workspace_size > 0)
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
    CHECK_HIP_ERROR(hipMalloc(&d_sync, sizeof(int32_t)));

    // create host memory
    CHECK_HIP_ERROR(hipHostMalloc(&a,   m * k * batch_count * sizeof(InTypeA)));
    CHECK_HIP_ERROR(hipHostMalloc(&b,   n * k * batch_count * sizeof(InTypeB)));
    CHECK_HIP_ERROR(hipHostMalloc(&c,   m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&d,   m * n * batch_count * sizeof(OutType)));
    CHECK_HIP_ERROR(hipHostMalloc(&ref, m * n * batch_count * sizeof(OutType)));

    // initialize data
    init_hpl<InTypeA>(reinterpret_cast<InTypeA*>(a), m * k * batch_count);
    init_hpl<InTypeB>(reinterpret_cast<InTypeB*>(b), n * k * batch_count);
    init_hpl<OutType>(reinterpret_cast<OutType*>(c), m * n * batch_count);

    // copy data to device
    CHECK_HIP_ERROR(hipMemset(d_sync, 0, sizeof(std::uint32_t)));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_a, a, m * k * batch_count * sizeof(InTypeA), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_b, b, n * k * batch_count * sizeof(InTypeB), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));

    // scale B is 240/AMax  and scaleA will be AMax/240
    CHECK_HIPBLASLT_ERROR(hipblasltExtFastValueDevidedByAMaxWithRcp(HIP_R_16F, HIP_R_32F, d_scaleB, d_scaleA, d_b, d_workspace, d_sync, m, n, cvtMax, stream));

    // hipblaslt setProblem API
    hipblaslt_ext::Gemm gemm(handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             HIP_R_8F_E4M3_FNUZ,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmInputs inputs;
    hipblaslt_ext::GemmEpilogue epilogue;
    hipblaslt_ext::GemmPreference gemmPref;

    inputs.a      = d_a;
    inputs.b      = d_b;
    inputs.c      = d_c;
    inputs.d      = d_d;
    inputs.alpha  = &alpha;
    inputs.beta   = &beta;
    inputs.scaleA = d_scaleA;
    inputs.scaleB = d_scaleB;

    gemmPref.setMaxWorkspaceBytes(max_workspace_size);

    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    // hipblaslt algoGetHeuristic API
    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));

    if(!heuristicResult.empty())
    {
        // hipblaslt initialize, run and sync
        CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
        CHECK_HIPBLASLT_ERROR(gemm.run(stream));

        // get device data back to host
        CHECK_HIP_ERROR(hipMemcpyAsync(&scaleA, d_scaleA, sizeof(float), hipMemcpyDeviceToHost, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(&scaleB, d_scaleB, sizeof(float), hipMemcpyDeviceToHost, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(d, d_d, m * n * batch_count * sizeof(OutType), hipMemcpyDeviceToHost, stream));

        // sync to get result to host
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        // get ref result
        CpuGemmExtWithAmaxScaleAB<InTypeA, InTypeB, OutType, ComputeInputType, AlphaType, BetaType>(m, n, k, batch_count, a, b, c, ref, alpha, beta, scaleA, scaleB);

        if (validate<OutType>(m, n, batch_count, d, ref) == 0)
            std::cout << "PASS" << std::endl;
    }
    else
    {
        std::cerr << "No valid solution found!" << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));
    CHECK_HIP_ERROR(hipFree(d_scaleA));
    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIP_ERROR(hipFree(d_sync));

    CHECK_HIP_ERROR(hipFree(a));
    CHECK_HIP_ERROR(hipFree(b));
    CHECK_HIP_ERROR(hipFree(c));
    CHECK_HIP_ERROR(hipFree(d));
    
    return;
}

int main(int argc, char** argv)
{
    if (argc != 8)
    {
        std::cout << argv[0] << " [M] [N] [K] [B] [alpha] [beta] [cvtMax]" << std::endl;
        return 0;
    }

    size_t m      = std::atoi(argv[1]);
    size_t n      = std::atoi(argv[2]);
    size_t k      = std::atoi(argv[3]);
    size_t b      = std::atoi(argv[4]);
    float  alpha  = std::atof(argv[5]);
    float  beta   = std::atof(argv[6]);
    float  cvtMax = std::atof(argv[7]);

    GemmExtWithAmaxScaleAB<hipblaslt_f8_fnuz, hipblasLtHalf, hipblasLtHalf, hipblaslt_f8_fnuz, float, float>(m, n, k, b, alpha, beta, cvtMax, 32 * 1024 * 1024);

    return 0;
}

