/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>

#include "helper.h"

void simpleGemmMixPrecisionExt(hipblasLtHandle_t  handle,
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
                               hipStream_t        stream);

template<typename TypeA, typename TypeB, typename TypeC, typename TypeD, typename ScalarType>
int validate(const Runner<TypeA, TypeB, TypeC, TypeD, ScalarType> &runner)
{
    std::vector<float> ref(runner.m * runner.n * runner.batch_count, 0);

    for (int64_t b = 0; b < runner.batch_count; ++b) {
        const auto batchStrideA = runner.m * runner.k;
        const auto batchStrideB = runner.k * runner.n;
        const auto batchStrideC = runner.m * runner.n;
        const auto batchStrideD = runner.m * runner.n;
        const hipblaslt_f8 *aPtr = reinterpret_cast<const hipblaslt_f8 *>(runner.a);
        const hipblasLtHalf *bPtr = reinterpret_cast<const hipblasLtHalf *>(runner.b);
        const float *cPtr = reinterpret_cast<const float *>(runner.c);
        for (int64_t i = 0; i < runner.m ; ++i) {
            for (int64_t j = 0; j < runner.n; ++j) {
                for (int64_t k = 0; k < runner.k; ++k) {
                    ref[batchStrideD * b + j * runner.m + i] += float(aPtr[batchStrideA * b + runner.m * k + i]) * float(bPtr[batchStrideB * b + runner.k * j + k]);
                }

                ref[batchStrideD * b + j * runner.m + i] *= runner.alpha;
                ref[batchStrideD * b + j * runner.m + i] += runner.beta * cPtr[batchStrideC * b + j * runner.m + i];
            }
        }
    }

    std::vector<float> gpuResult(runner.m * runner.n * runner.batch_count, 0);
    hipMemcpyDtoH(gpuResult.data(), runner.d_d, runner.batch_count * runner.m * runner.n * sizeof(float));

    for (int64_t b = 0; b < runner.batch_count; ++b) {
        const auto batchStrideD = runner.m * runner.n;
        for (int64_t i = 0; i < runner.m ; ++i) {
            for (int64_t j = 0; j < runner.n; ++j) {
                const auto lhs = ref[batchStrideD * b + j * runner.m + i];
                const auto rhs = gpuResult[batchStrideD * b + j * runner.m + i];

                if (std::abs(lhs - rhs) > 1e-5) {
                    // std::cout << lhs << " vs " << rhs << '\n';
                    // assert(ref[batchStrideD * b + j * runner.m + i] == gpuResult[batchStrideD * b + j * runner.m + i]);
                    return -1;
                }
            }
        }
    }

    return 0;
}

int main()
{
    /** This is an example using hipblaslt extension API.
     *  This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    Runner<hipblaslt_f8, hipblasLtHalf, float, float, float> runner(
        1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);

    runner.run([&runner] {
        simpleGemmMixPrecisionExt(runner.handle,
                                  HIPBLAS_OP_N,
                                  HIPBLAS_OP_N,
                                  runner.m,
                                  runner.n,
                                  runner.k,
                                  runner.batch_count,
                                  runner.alpha,
                                  runner.beta,
                                  runner.d_a,
                                  runner.d_b,
                                  runner.d_c,
                                  runner.d_d,
                                  runner.d_workspace,
                                  runner.max_workspace_size,
                                  runner.stream);
    });

    if (validate(runner)) {
        std::cerr << "Validation failed\n";
    }

    return 0;
}

void simpleGemmMixPrecisionExt(hipblasLtHandle_t  handle,
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
                               hipStream_t        stream)
{
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(handle,
                             trans_a,
                             trans_b,
                             HIPBLASLT_R_8F_E4M3,
                             HIPBLASLT_R_16F,
                             HIPBLASLT_R_32F,
                             HIPBLASLT_R_32F,
                             HIPBLASLT_COMPUTE_F32_FAST_F16);

    // Copy scaleA to device memory
    float scaleA   = 2.f;
    void* d_scaleA = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_scaleA, sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_scaleA, &scaleA, sizeof(float), hipMemcpyHostToDevice));

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.a     = d_a;
    inputs.b     = d_b;
    inputs.c     = d_c;
    inputs.d     = d_d;
    inputs.alpha = &alpha;
    inputs.beta  = &beta;
    // inputs.bias  = d_bias; // Add bias here if needed.
    inputs.scaleA = d_scaleA; // Add scaleA, this is a device pointer.
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));

    if(heuristicResult.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        CHECK_HIP_ERROR(hipFree(d_scaleA));
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, calculate the needed workspace_size and allocate d_workspace here
    // uint64_t workspace_size = 0;
    // for(int i = 0; i < returnedAlgoCount; i++)
    //     workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

    // Make sure to initialize everytime the algo changes
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
    CHECK_HIPBLASLT_ERROR(gemm.run(stream));

    CHECK_HIP_ERROR(hipFree(d_scaleA));
    return;
}
