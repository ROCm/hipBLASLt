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
#include "helper.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>

void simpleGemmScaleABVec(hipblasLtHandle_t  handle,
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

int main()
{
    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float, float> runner(
        128, 128, 128, 1, 1.f, 1.f, 32 * 1024 * 1024);

    std::cout << "Running with Scale A and Scale B alternating between 0.5 and 2.0" << std::endl;

    runner.run([&runner] {
        simpleGemmScaleABVec(runner.handle,
                           HIPBLAS_OP_T,
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

    return 0;
}

void simpleGemmScaleABVec(hipblasLtHandle_t  handle,
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
    std::vector<float> h_scale_a(128);
    std::vector<float> h_scale_b(128);

    // Initialize scale_a and scale_b to alternate between 0.5 and 2.0
    for (int i = 0; i < 128; ++i) {
        if (i % 2 == 0) {
            h_scale_a[i] = 0.5f;
            h_scale_b[i] = 0.5f;
        } else {
            h_scale_a[i] = 2.0f;
            h_scale_b[i] = 2.0f;
        }
    }

    float* d_scale_a;
    float* d_scale_b;

    // Allocate device memory for scale vectors
    CHECK_HIP_ERROR(hipMalloc(&d_scale_a, h_scale_a.size() * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_b, h_scale_b.size() * sizeof(float)));

    // Copy scale vectors from host to device
    CHECK_HIP_ERROR(hipMemcpyAsync(d_scale_a, h_scale_a.data(), h_scale_a.size() * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(hipMemcpyAsync(d_scale_b, h_scale_b.data(), h_scale_b.size() * sizeof(float), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_8F_E4M3_FNUZ, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_8F_E4M3_FNUZ, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16BF, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16BF, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    // Set A and B matrix scale vectors
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT, &d_scale_a, sizeof(float*)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT, &d_scale_b, sizeof(float*)));

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    const int                        request_solutions = 5;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int                              returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                          matmul,
                                                          matA,
                                                          matB,
                                                          matC,
                                                          matD,
                                                          pref,
                                                          request_solutions,
                                                          heuristicResult,
                                                          &returnedAlgoCount));

    if(returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        CHECK_HIP_ERROR(hipFree(d_scale_a));
        CHECK_HIP_ERROR(hipFree(d_scale_b));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
        return;
    }

    uint64_t workspace_size = max_workspace_size;
    for(int i = 0; i < returnedAlgoCount; i++)
        workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);

    // Perform matrix multiplication
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                          matmul,
                                          &alpha,
                                          d_a,
                                          matA,
                                          d_b,
                                          matB,
                                          &beta,
                                          d_c,
                                          matC,
                                          d_d,
                                          matD,
                                          &heuristicResult[0].algo,
                                          d_workspace,
                                          workspace_size,
                                          stream));

    // Clean up resources
    CHECK_HIP_ERROR(hipFree(d_scale_a));
    CHECK_HIP_ERROR(hipFree(d_scale_b));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));

    std::cout << "Matrix multiplication completed successfully." << std::endl;
}