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

void simpleGroupedGemmExt(hipblasLtHandle_t     handle,
                          hipblasOperation_t    trans_a,
                          hipblasOperation_t    trans_b,
                          std::vector<int64_t>& m,
                          std::vector<int64_t>& n,
                          std::vector<int64_t>& k,
                          std::vector<int64_t>& batch_count,
                          std::vector<float>&   alpha,
                          std::vector<float>&   beta,
                          std::vector<void*>&   d_a,
                          std::vector<void*>&   d_b,
                          std::vector<void*>&   d_c,
                          std::vector<void*>&   d_d,
                          void*                 d_workspace,
                          int64_t               max_workspace_size,
                          hipStream_t           stream);

int main()
{
    /** This is an example using hipblaslt extension API.
     *  This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    std::vector<int64_t>                                  m           = {1024, 512};
    std::vector<int64_t>                                  n           = {512, 512};
    std::vector<int64_t>                                  k           = {1920, 128};
    std::vector<int64_t>                                  batch_count = {1, 1};
    std::vector<float>                                    alpha       = {1.0f, 1.0f};
    std::vector<float>                                    beta        = {1.0f, 1.0f};
    RunnerVec<hipblasLtHalf, hipblasLtHalf, float, float> runner(
        m, n, k, batch_count, alpha, beta, 32 * 1024 * 1024);

    runner.run([&runner] {
        simpleGroupedGemmExt(runner.handle,
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

    return 0;
}

void simpleGroupedGemmExt(hipblasLtHandle_t     handle,
                          hipblasOperation_t    trans_a,
                          hipblasOperation_t    trans_b,
                          std::vector<int64_t>& m,
                          std::vector<int64_t>& n,
                          std::vector<int64_t>& k,
                          std::vector<int64_t>& batch_count,
                          std::vector<float>&   alpha,
                          std::vector<float>&   beta,
                          std::vector<void*>&   d_a,
                          std::vector<void*>&   d_b,
                          std::vector<void*>&   d_c,
                          std::vector<void*>&   d_d,
                          void*                 d_workspace,
                          int64_t               max_workspace_size,
                          hipStream_t           stream)
{
    // Get all algo doesn't require a gemm instance.
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(
        hipblaslt_ext::getAllAlgos(handle,
                                   hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                                   trans_a,
                                   trans_a,
                                   HIP_R_16F,
                                   HIP_R_16F,
                                   HIP_R_16F,
                                   HIP_R_16F,
                                   HIPBLAS_COMPUTE_32F,
                                   heuristicResult));

    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::GroupedGemm groupedgemm(handle,
                                           trans_a,
                                           trans_b,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIPBLAS_COMPUTE_32F);

    std::vector<hipblaslt_ext::GemmEpilogue> epilogue{
        hipblaslt_ext::
            GemmEpilogue()}; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    std::vector<hipblaslt_ext::GemmInputs> inputs(m.size());
    for(int i = 0; i < m.size(); i++)
    {
        inputs[i].a     = d_a[i];
        inputs[i].b     = d_b[i];
        inputs[i].c     = d_c[i];
        inputs[i].d     = d_d[i];
        inputs[i].alpha = &alpha[i];
        inputs[i].beta  = &beta[i];
    }
    // hipblaslt_ext::GemmEpilogue supports broadcasting
    groupedgemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    uint64_t            workspace_size = 0;
    std::vector<size_t> validIdx;
    for(size_t i = 0; i < heuristicResult.size(); i++)
    {
        size_t workspaceSizeInBytes = 0;
        if(groupedgemm.isAlgoSupported(heuristicResult[i].algo, workspaceSizeInBytes)
           == HIPBLAS_STATUS_SUCCESS)
        {
            if(workspaceSizeInBytes <= max_workspace_size)
            {
                workspace_size = max(workspace_size, workspaceSizeInBytes);
                validIdx.push_back(i);
            }
        }
    }

    if(validIdx.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, allocate d_workspace here
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

    // Get the default values from the grouepdgemm object
    hipblaslt_ext::UserArguments* userArgs;
    CHECK_HIP_ERROR(hipHostMalloc(&userArgs, m.size() * sizeof(hipblaslt_ext::UserArguments)));
    groupedgemm.getDefaultValueForDeviceUserArguments(userArgs);
    // Copy them to device memory
    hipblaslt_ext::UserArguments* d_userArgs;
    CHECK_HIP_ERROR(hipMalloc(&d_userArgs, m.size() * sizeof(hipblaslt_ext::UserArguments)));
    CHECK_HIP_ERROR(hipMemcpy(d_userArgs,
                              userArgs,
                              m.size() * sizeof(hipblaslt_ext::UserArguments),
                              hipMemcpyHostToDevice));

    // Make sure to initialize everytime the algo changes
    CHECK_HIPBLASLT_ERROR(groupedgemm.initialize(heuristicResult[validIdx[0]].algo, d_workspace));
    CHECK_HIPBLASLT_ERROR(groupedgemm.run(d_userArgs, stream));

    CHECK_HIP_ERROR(hipFree(userArgs));
    CHECK_HIP_ERROR(hipFree(d_userArgs));
    return;
}
