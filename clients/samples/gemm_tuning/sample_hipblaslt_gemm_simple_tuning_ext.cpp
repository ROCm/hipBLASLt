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
#include <chrono>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>

#include "helper.h"

void simpleGemmTuningSplitKExt(hipblasLtHandle_t  handle,
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
                               void*              d_bias,
                               void*              d_workspace,
                               int64_t            max_workspace_size,
                               hipStream_t        stream);

int main()
{
    /** This is an example using hipblaslt extension API.
     *  This is a NN example with
     *  a = (m, k). lda = m
     *  b = (k, n). ldb = k
     *  c = d = (m, n). ldc = ldd = m
     */
    Runner<hipblaslt_f8_fnuz, hipblasLtHalf, hipblasLtHalf, float, float> runner(
        1024, 16, 8192, 1, 1.f, 1.f, 32 * 1024 * 1024);

    runner.run([&runner] {
        simpleGemmTuningSplitKExt(runner.handle,
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
                                  runner.d_bias,
                                  runner.d_workspace,
                                  runner.max_workspace_size,
                                  runner.stream);
    });

    return 0;
}

double getTimeUSSync(hipStream_t stream)
{
    if(hipDeviceSynchronize() != hipSuccess)
    {
        std::cerr << "Synchronizing device failed" << std::endl;
    }

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

void simpleGemmTuningSplitKExt(hipblasLtHandle_t  handle,
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
                               void*              d_bias,
                               void*              d_workspace,
                               int64_t            max_workspace_size,
                               hipStream_t        stream)
{
    // Get all algo doesn't require a gemm instance.
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                     trans_a,
                                                     trans_a,
                                                     HIP_R_8F_E4M3_FNUZ,
                                                     HIP_R_16F,
                                                     HIP_R_16F,
                                                     HIP_R_16F,
                                                     HIPBLAS_COMPUTE_32F_FAST_16F,
                                                     heuristicResult));

    // Here you can setup wgm, splitk combinations for later tuning.
    std::vector<unsigned char>             splitkVec = {0, 2, 4, 8, 12, 16, 20, 24};
    std::vector<hipblaslt_ext::GemmTuning> tunings;
    tunings.resize(splitkVec.size());
    for(size_t i = 0; i < splitkVec.size(); i++)
    {
        tunings[i].splitK = splitkVec[i];
    }

    // Input setup
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(handle,
                             trans_a,
                             trans_b,
                             HIP_R_8F_E4M3_FNUZ,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F_FAST_16F);

    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.mode           = HIPBLASLT_EPILOGUE_BIAS;
    epilogue.bias_data_type = HIP_R_32F;
    epilogue.aux_ld         = m;
    epilogue.aux_stride     = m;
    hipblaslt_ext::GemmInputs inputs;
    inputs.a     = d_a;
    inputs.b     = d_b;
    inputs.c     = d_c;
    inputs.d     = d_d;
    inputs.alpha = &alpha;
    inputs.beta  = &beta;
    inputs.bias  = d_bias;
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    // Not all the solutions supports GemmTuning, if you create a
    // hipblaslt_ext::GemmTuning without changing any default values,
    // the effect is same as calling API
    // isAlgoSupported(algo, returnedWorkspaceSize)
    uint64_t                         workspace_size = 0;
    std::vector<std::vector<size_t>> validIdx;
    validIdx.resize(tunings.size());
    for(size_t j = 0; j < heuristicResult.size(); j++)
    {
        for(size_t i = 0; i < tunings.size(); i++)
        {
            size_t workspaceSizeInBytes = 0;
            // If tuning is given, the API will not return success if the solution cannot
            // accept an user tuning parameter.
            if(gemm.isAlgoSupported(heuristicResult[j].algo, tunings[i], workspaceSizeInBytes)
               == HIPBLAS_STATUS_SUCCESS)
            {
                if(workspaceSizeInBytes <= max_workspace_size)
                {
                    workspace_size = max(workspace_size, workspaceSizeInBytes);
                    validIdx[i].push_back(j);
                }
            }
        }
    }

    for(size_t i = 0; i < validIdx.size(); i++)
    {
        if(validIdx[i].empty())
        {
            std::cerr << "No valid solution found for splitk " << tunings[i].splitK << "!"
                      << std::endl;
            return;
        }
    }
    // Note that different Tuning configurations will get different
    // amounts of validIdx.

    void* ws_ptr = nullptr;
    // Changing GSU might require more workspace_size.
    if(workspace_size > max_workspace_size)
    {
        static_cast<void>(hipMalloc(&ws_ptr, workspace_size));
    }
    else
    {
        ws_ptr = d_workspace;
    }

    size_t totalSolutions = 0;
    for(size_t i = 0; i < validIdx.size(); i++)
    {
        totalSolutions += validIdx[i].size();
    }
    std::cout << "Total solutions: " << totalSolutions << std::endl;

    size_t solIndex = -1, splitIndex = -1;
    double bestTime = std::numeric_limits<double>::max();
    int    run      = 1000;
    for(size_t i = 0; i < validIdx.size(); i++)
    {
        std::cout << "Solutions for tuning parameters: splitk: " << (int)tunings[i].splitK
                  << ", wgm: " << (int)tunings[i].wgm << ": " << validIdx[i].size() << std::endl
                  << "[Solution Index][SplitK][WGM]" << std::endl;
        for(size_t idx = 0; idx < validIdx[i].size(); idx++)
        {
            // Make sure to initialize every time when algo changes
            // If tuning is given, the API will not return success if the solution cannot accept an user tuning parameter.
            CHECK_HIPBLASLT_ERROR(
                gemm.initialize(heuristicResult[validIdx[i][idx]].algo, tunings[i], ws_ptr));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            auto gpuTime = getTimeUSSync(stream);
            for(size_t t = 0; t < run; t++)
                static_cast<void>(gemm.run(stream));
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            gpuTime = (getTimeUSSync(stream) - gpuTime) / run;
            std::cout << "[" << validIdx[i][idx] << "][" << (int)tunings[i].splitK << "]["
                      << (int)tunings[i].wgm << "] "
                      << "Time: " << gpuTime << "us" << std::endl;
            if(gpuTime < bestTime)
            {
                bestTime   = gpuTime;
                solIndex   = validIdx[i][idx];
                splitIndex = i;
            }
        }
    }

    if(solIndex != -1)
    {
        std::cout << "0 in SplitK and WGM means use solutions' default value." << std::endl;
        std::cout << "Best solution: " << solIndex << std::endl
                  << "SplitK: " << (int)tunings[splitIndex].splitK << std::endl
                  << "Wgm: " << (int)tunings[splitIndex].wgm << std::endl
                  << "Time: " << bestTime << "us" << std::endl;
    }

    if(workspace_size > max_workspace_size)
    {
        static_cast<void>(hipFree(ws_ptr));
    }
    return;
}
