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
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>

#include "helper.h"

void simpleGemmScaleABCExt(hipblasLtHandle_t  handle,
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
                          float              h_scale_a,
                          float              h_scale_b,
                          float              h_scale_c);

int main()
{
    // This is an example using hipblaslt extension API: ScaleA & ScaleB & ScaleC
    Runner<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float, float> runner(
        128, 128, 128, 1, 1.f, 1.f, 32 * 128 * 128);

    float scale_a = 0.5f; // scale A setting
    float scale_b = 2.0f; // scale B setting
    float scale_c = 2.0f; // scale C setting
    std::cout << "Running with Scale A = " << scale_a << ", Scale B = " << scale_b << ", and Scale C = " << scale_c << std::endl;

    runner.run([&runner, scale_a, scale_b, scale_c] {
        simpleGemmScaleABCExt(runner.handle,
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
                             runner.stream,
                             scale_a,
                             scale_b,
                             scale_c);
    });

    return 0;
}

void simpleGemmScaleABCExt(hipblasLtHandle_t  handle,
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
                          float              h_scale_a,
                          float              h_scale_b,
                          float              h_scale_c)
{
    hipblaslt_ext::GemmPreferenceV2 gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(handle,
                             trans_a,
                             trans_b,
                             HIP_R_8F_E4M3_FNUZ,
                             HIP_R_8F_E4M3_FNUZ,
                             HIP_R_8F_E4M3_FNUZ,
                             HIP_R_8F_E4M3_FNUZ,
                             HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogueV2
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputsV2 inputs;
    float*                      d_scale_a;
    float*                      d_scale_b;
    float*                      d_scale_c;
    CHECK_HIP_ERROR(hipMalloc(&d_scale_a, sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_b, sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_scale_c, sizeof(float)));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_a, &h_scale_a, sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_b, &h_scale_b, sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_HIP_ERROR(
        hipMemcpyAsync(d_scale_c, &h_scale_c, sizeof(float), hipMemcpyHostToDevice, stream));

    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setScaleA(d_scale_a);
    inputs.setScaleB(d_scale_b);
    inputs.setScaleC(d_scale_c);
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));

    if(heuristicResult.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, calculate the needed workspace_size and allocate d_workspace here
    // uint64_t workspace_size = 0;
    // for(int i = 0; i < returnedAlgoCount; i++)
    //     workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

    // Make sure to initialize every time when algo changes
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
    CHECK_HIPBLASLT_ERROR(gemm.run(stream));
    return;
}
