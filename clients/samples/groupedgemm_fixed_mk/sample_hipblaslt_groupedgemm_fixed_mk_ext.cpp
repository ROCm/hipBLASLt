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

void simpleGroupedGemmFixedMKExt(hipblasLtHandle_t     handle,
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
        simpleGroupedGemmFixedMKExt(runner.handle,
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

__global__ void kernelUpdateN(uint32_t gemm_count, void* userArgs, int64_t* sizes_n)
{
    uint64_t id = hipBlockIdx_x * 256 + hipThreadIdx_x;

    if(id >= gemm_count)
        return;

    hipblaslt_ext::UserArguments* d_userArgs = static_cast<hipblaslt_ext::UserArguments*>(userArgs);
    d_userArgs[id].n                         = sizes_n[id];
}

void simpleGroupedGemmFixedMKExt(hipblasLtHandle_t     handle,
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
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::GroupedGemm groupedgemm(handle,
                                           trans_a,
                                           trans_b,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIP_R_16F,
                                           HIPBLASLT_COMPUTE_F32);

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

    // When n is free and m, k is fixed, we'll need sum of n to work
    // 1. collect sum of N
    // 2. set problem to {Ms, {sum of N, 1, 1, 1, ...}, Ks}
    int                  sum_of_n = 0;
    std::vector<int64_t> sum_of_n_vec;
    for(int i = 0; i < n.size(); i++)
    {
        sum_of_n += n[i];
        sum_of_n_vec.push_back(1);
    }
    sum_of_n_vec[0] = sum_of_n;

    // Copy the N vector to device memory.
    int64_t* d_n = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_n, m.size() * sizeof(int64_t)));
    CHECK_HIP_ERROR(hipMemcpy(d_n, n.data(), m.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    // hipblaslt_ext::GemmEpilogue supports broadcasting
    groupedgemm.setProblem(m, sum_of_n_vec, k, batch_count, epilogue, inputs);

    // Get the default hipblaslt_ext::UserArguments aafter setProblem
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

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(
        groupedgemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));

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

    // Make sure to initialize everytime the algo changes
    CHECK_HIPBLASLT_ERROR(groupedgemm.initialize(heuristicResult[0].algo, d_workspace));

    // Then you can change the N in the previous kernel to whatever you want, just make sure the sum of N does not exceed the setup.
    int threads = 256;
    int blocks  = ceil((double)m.size() / threads);
    hipLaunchKernelGGL(
        kernelUpdateN, dim3(blocks), dim3(threads), 0, stream, (uint32_t)m.size(), d_userArgs, d_n);
    CHECK_HIPBLASLT_ERROR(groupedgemm.run(d_userArgs, stream));

    CHECK_HIP_ERROR(hipFree(userArgs));
    CHECK_HIP_ERROR(hipFree(d_userArgs));
    return;
}
