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

#include "hipblaslt-ext.hpp"
#include "exceptions.hpp"
#include "hipblaslt_internal.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt_float8.h>
#include <iostream>
#include <rocblaslt.h>

namespace hipblaslt_ext
{
    template <typename SrcType, typename DstType>
    __global__ void datatypeConversion(const SrcType* src, DstType* dst, std::size_t numElements)
    {
        const auto tId        = threadIdx.x;
        const auto bId        = blockIdx.x;
        const auto blockSize  = blockDim.x * blockDim.y * blockDim.z;
        const auto elemOffset = bId * blockSize + tId;

        if(elemOffset < numElements)
        {
            dst[elemOffset] = DstType(src[elemOffset]);
        }
    }

    template <typename SrcType, typename DstType>
    void datatypeConversionCpu(const SrcType* src, DstType* dst, std::size_t numElements)
    {
        for(std::size_t i = 0; i < numElements; ++i)
        {
            dst[i] = DstType(src[i]);
        }
    }

    auto NullDeleter = [](void*) { return hipSuccess; };

    HipBufferPtr makeHipBuffer(std::size_t numBytes)
    {
        if(!numBytes)
        {
            return HipBufferPtr(nullptr, NullDeleter);
        }

        void* ptr = nullptr;
        auto  err = hipMalloc(&ptr, numBytes);

        if(err != hipSuccess)
        {
            return HipBufferPtr(nullptr, NullDeleter);
        }

        return HipBufferPtr(ptr, &hipFree);
    }

    void GemmPreference::setMaxWorkspaceBytes(size_t workspaceBytes)
    {
        m_workspace_bytes = workspaceBytes;
    }

    const size_t GemmPreference::getMaxWorkspaceBytes() const
    {
        return m_workspace_bytes;
    }

    GemmInstance::GemmInstance(hipblasLtHandle_t handle, GemmType type)
        : m_gemm_type(type)
        , m_handle(handle)
    {
    }

    GemmType GemmInstance::getGemmType()
    {
        return m_gemm_type;
    }

    size_t GemmInstance::getGemmCount()
    {
        return m_gemm_count;
    }

    hipblasStatus_t GemmInstance::algoGetHeuristic(
        const int                                      requestedAlgoCount,
        const GemmPreference&                          pref,
        std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_algo_get_heuristic_cpp((rocblaslt_handle)m_handle,
                                             gemmType,
                                             m_data,
                                             pref.getMaxWorkspaceBytes(),
                                             requestedAlgoCount,
                                             *results));
    }

    hipblasStatus_t GemmInstance::isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                  size_t&                workspaceSizeInBytes)
    try
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo  = reinterpret_cast<rocblaslt_matmul_algo*>(&algo);
        return RocBlasLtStatusToHIPStatus(rocblaslt_is_algo_supported_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data, *rocalgo, workspaceSizeInBytes));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::initialize(const hipblasLtMatmulAlgo_t& algo,
                                             void*                        workspace,
                                             bool                         useUserArgs,
                                             hipStream_t                  stream)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo  = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        return RocBlasLtStatusToHIPStatus(rocblaslt_makeArgument_cpp((rocblaslt_handle)m_handle,
                                                                     gemmType,
                                                                     *rocalgo,
                                                                     workspace,
                                                                     useUserArgs,
                                                                     stream,
                                                                     m_data));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::run(hipStream_t stream)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;

        if(m_auxiliary_conversion_buffers.size())
        {
            for(auto& conversions : m_auxiliary_conversion_buffers)
            {
                for(auto& conversion : conversions)
                {
                    //Only check dst since the existence garantees requirement of fp8/bf8 -> fp16 conversion
                    if(auto& dst = std::get<1>(conversion))
                    {
                        auto src = std::get<0>(conversion);

                        if(src)
                        {
                            auto           srcType           = std::get<2>(conversion);
                            const auto     numElements       = std::get<4>(conversion);
                            constexpr auto numWorkitemsPerWg = 256;
                            const auto     numWg             = (numElements / numWorkitemsPerWg)
                                               + !!(numElements % numWorkitemsPerWg);

                            if(srcType == HIPBLASLT_R_8F_E4M3)
                            {
                                datatypeConversion<hipblaslt_f8, hipblasLtHalf>
                                    <<<numWg, numWorkitemsPerWg, 0, stream>>>(
                                        (const hipblaslt_f8*)src,
                                        (hipblasLtHalf*)dst.get(),
                                        numElements);

                                // hipStreamSynchronize(stream);
                                // std::vector<hipblaslt_f8> cpuSrc(numElements);
                                // std::vector<hipblasLtHalf> cpuDst(numElements);
                                // hipMemcpyDtoH(cpuSrc.data(), src, numElements);
                                // hipMemcpyDtoH(cpuDst.data(), dst.get(), numElements * 2);

                                // for (size_t i = 0; i < numElements; ++i) {
                                //     auto a = float(cpuSrc[i]);
                                //     auto b = float(cpuDst[i]);
                                //     if (std::abs(a - b) > 1e-5) {
                                //         std::cout << "values mismatched at " << i << " (a, b) == (" << a << ", " << b << ")\n";
                                //     }
                                // }
                            }
                            else if(srcType == HIPBLASLT_R_8F_E5M2)
                            {
                                datatypeConversion<hipblaslt_bf8, hipblasLtHalf>
                                    <<<numWg, numWorkitemsPerWg, 0, stream>>>(
                                        (const hipblaslt_bf8*)src,
                                        (hipblasLtHalf*)dst.get(),
                                        numElements);
                            }
                        }
                    }
                }
            }
        }

        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_run_cpp((rocblaslt_handle)m_handle, gemmType, m_data, stream));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    Gemm::Gemm(hipblasLtHandle_t      handle,
               hipblasOperation_t     opA,
               hipblasOperation_t     opB,
               hipblasltDatatype_t    typeA,
               hipblasltDatatype_t    typeB,
               hipblasltDatatype_t    typeC,
               hipblasltDatatype_t    typeD,
               hipblasLtComputeType_t typeCompute)
        : GemmInstance(handle, GemmType::HIPBLASLT_GEMM)
    {
        m_problem_types.push_back({opA, opB, typeA, typeB, typeC, typeD, typeCompute});
        rocblaslt_init_gemmData((rocblaslt_handle)m_handle,
                                static_cast<rocblaslt::RocGemmType>(m_gemm_type),
                                opA,
                                opB,
                                typeA,
                                typeB,
                                typeC,
                                typeD,
                                (rocblaslt_compute_type)typeCompute,
                                0,
                                m_data);
    }

    Gemm::Gemm(hipblasLtHandle_t       handle,
               hipblasLtMatmulDesc_t   matmul_descr,
               const void*             alpha,
               const void*             A,
               hipblasLtMatrixLayout_t matA,
               const void*             B,
               hipblasLtMatrixLayout_t matB,
               const void*             beta,
               const void*             C,
               hipblasLtMatrixLayout_t matC,
               void*                   D,
               hipblasLtMatrixLayout_t matD)
        : GemmInstance(handle, GemmType::HIPBLASLT_GEMM)
    {
        auto status = setProblem(matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
    }

    hipblasStatus_t Gemm::setProblem(int64_t       m,
                                     int64_t       n,
                                     int64_t       k,
                                     int64_t       batch_count,
                                     GemmEpilogue& epilogue,
                                     GemmInputs&   inputs)
    {
        if(n == 0 || m == 0)
        {
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        int lda     = m_problem_types[0].op_a == HIPBLAS_OP_N ? m : k;
        int ldb     = m_problem_types[0].op_b == HIPBLAS_OP_N ? k : n;
        int ldc     = m;
        int strideA = m * k;
        int strideB = n * k;
        int strideC = m * n;
        return setProblem(m,
                          n,
                          k,
                          batch_count,
                          lda,
                          ldb,
                          ldc,
                          ldc,
                          strideA,
                          strideB,
                          strideC,
                          strideC,
                          epilogue,
                          inputs,
                          m_problem_types[0]);
    }

    hipblasStatus_t Gemm::setProblem(int64_t          m,
                                     int64_t          n,
                                     int64_t          k,
                                     int64_t          batch_count,
                                     int64_t          lda,
                                     int64_t          ldb,
                                     int64_t          ldc,
                                     int64_t          ldd,
                                     int64_t          strideA,
                                     int64_t          strideB,
                                     int64_t          strideC,
                                     int64_t          strideD,
                                     GemmEpilogue&    epilogue,
                                     GemmInputs&      inputs,
                                     GemmProblemType& problemtype)
    {
#ifndef __gfx940__
        constexpr auto numGemms = 1;

        if(m_auxiliary_conversion_buffers.size() != numGemms)
        {
            m_auxiliary_conversion_buffers.resize(numGemms);

            for(std::size_t j = 0; j < m_auxiliary_conversion_buffers.size(); ++j)
            {
                const std::vector<std::int64_t> sizes{strideA, strideB};
                const std::vector<void*>       gemmInputs{inputs.a, inputs.b};
                auto&                          conversions = m_auxiliary_conversion_buffers.at(j);
                auto&                          problem     = m_problem_types.at(j);
                const std::vector<hipblasltDatatype_t> dtypes{
                    problem.type_a, problem.type_b, problem.type_c};

                for(std::size_t i = 0; i < sizes.size(); ++i)
                {
                    auto dtype = dtypes.at(i);

                    if(dtype == HIPBLASLT_R_8F_E4M3 || dtype == HIPBLASLT_R_8F_E5M2)
                    {
                        const auto numElements = sizes.at(i) * batch_count;
                        auto       numBytes    = numElements * sizeof(hipblasLtHalf);
                        conversions.emplace_back(std::make_tuple(gemmInputs.at(i),
                                                                 std::move(makeHipBuffer(numBytes)),
                                                                 dtype,
                                                                 HIPBLASLT_R_16F,
                                                                 numElements));
                    }
                    else
                    {
                        conversions.emplace_back(std::make_tuple(gemmInputs.at(i),
                                                                 std::move(makeHipBuffer(0)),
                                                                 dtype,
                                                                 HIPBLASLT_R_16F,
                                                                 0));
                    }
                }
            }
        }

        //Shallow copy
        GemmInputs gemmInputs = inputs;
        GemmProblemType gemmProblemType = problemtype;
        auto&      problem    = m_problem_types.at(0);

        if(auto& a = std::get<1>(m_auxiliary_conversion_buffers.at(0).at(0)))
        {
            gemmInputs.a   = a.get();
            gemmProblemType.type_a = HIPBLASLT_R_16F;
        }

        if(auto& b = std::get<1>(m_auxiliary_conversion_buffers.at(0).at(1)))
        {
            gemmInputs.b   = b.get();
            gemmProblemType.type_b = HIPBLASLT_R_16F;
        }
#else
        GemmInputs& gemmInputs = inputs;
        GemmProblemType &gemmProblemType = problemtype;
#endif
        auto rocepilogue    = reinterpret_cast<rocblaslt::RocGemmEpilogue*>(&epilogue);
        auto rocepinputs    = reinterpret_cast<rocblaslt::RocGemmInputs*>(&gemmInputs);
        auto rocproblemtype = reinterpret_cast<rocblaslt::RocGemmProblemType*>(&gemmProblemType);
        auto status         = RocBlasLtStatusToHIPStatus(rocblaslt_gemm_create_cpp(m,
                                                                           n,
                                                                           batch_count,
                                                                           k,
                                                                           lda,
                                                                           ldb,
                                                                           ldc,
                                                                           ldd,
                                                                           strideA,
                                                                           strideB,
                                                                           strideC,
                                                                           strideD,
                                                                           *rocepilogue,
                                                                           *rocepinputs,
                                                                           *rocproblemtype,
                                                                           m_data,
                                                                           m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types[0] = problemtype;
        }
        return status;
    }

    hipblasStatus_t Gemm::setProblem(hipblasLtMatmulDesc_t   matmul_descr,
                                     const void*             alpha,
                                     const void*             A,
                                     hipblasLtMatrixLayout_t matA,
                                     const void*             B,
                                     hipblasLtMatrixLayout_t matB,
                                     const void*             beta,
                                     const void*             C,
                                     hipblasLtMatrixLayout_t matC,
                                     void*                   D,
                                     hipblasLtMatrixLayout_t matD)
    {
        auto rocproblemtypes
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemType>*>(&m_problem_types);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_gemm_create_cpp((rocblaslt_matmul_desc)matmul_descr,
                                      alpha,
                                      A,
                                      (rocblaslt_matrix_layout)matA,
                                      B,
                                      (rocblaslt_matrix_layout)matB,
                                      beta,
                                      C,
                                      (rocblaslt_matrix_layout)matC,
                                      D,
                                      (rocblaslt_matrix_layout)matD,
                                      (*rocproblemtypes)[0],
                                      m_data,
                                      m_gemm_count));
    }

    GemmProblemType Gemm::getProblemTypes()
    {
        return m_problem_types[0];
    }

    HIPBLASLT_EXPORT GroupedGemm::GroupedGemm(hipblasLtHandle_t      handle,
                                              hipblasOperation_t     opA,
                                              hipblasOperation_t     opB,
                                              hipblasltDatatype_t    typeA,
                                              hipblasltDatatype_t    typeB,
                                              hipblasltDatatype_t    typeC,
                                              hipblasltDatatype_t    typeD,
                                              hipblasLtComputeType_t typeCompute)
        : GemmInstance(handle, GemmType::HIPBLASLT_GROUPED_GEMM)
    {
        m_problem_types.push_back({opA, opB, typeA, typeB, typeC, typeD, typeCompute});
        rocblaslt_init_gemmData((rocblaslt_handle)m_handle,
                                static_cast<rocblaslt::RocGemmType>(m_gemm_type),
                                opA,
                                opB,
                                typeA,
                                typeB,
                                typeC,
                                typeD,
                                (rocblaslt_compute_type)typeCompute,
                                0,
                                m_data);
    }

    HIPBLASLT_EXPORT GroupedGemm::GroupedGemm(hipblasLtHandle_t                     handle,
                                              std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                              std::vector<void*>&                   alpha,
                                              std::vector<void*>&                   A,
                                              std::vector<hipblasLtMatrixLayout_t>& matA,
                                              std::vector<void*>&                   B,
                                              std::vector<hipblasLtMatrixLayout_t>& matB,
                                              std::vector<void*>&                   beta,
                                              std::vector<void*>&                   C,
                                              std::vector<hipblasLtMatrixLayout_t>& matC,
                                              std::vector<void*>&                   D,
                                              std::vector<hipblasLtMatrixLayout_t>& matD)
        : GemmInstance(handle, GemmType::HIPBLASLT_GROUPED_GEMM)
    {
        auto status = setProblem(matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&      m,
                                            std::vector<int64_t>&      n,
                                            std::vector<int64_t>&      k,
                                            std::vector<int64_t>&      batch_count,
                                            std::vector<GemmEpilogue>& epilogue,
                                            std::vector<GemmInputs>&   inputs)
    {
        std::vector<int64_t> lda;
        std::vector<int64_t> ldb;
        std::vector<int64_t> ldc;
        std::vector<int64_t> ldd;
        std::vector<int64_t> strideA;
        std::vector<int64_t> strideB;
        std::vector<int64_t> strideC;
        std::vector<int64_t> strideD;
        for(size_t i = 0; i < m.size(); i++)
        {
            size_t iIdx = m_problem_types.size() == 1 ? 0 : i;
            lda.push_back(m_problem_types[iIdx].op_a == HIPBLAS_OP_N ? m[i] : k[i]);
            ldb.push_back(m_problem_types[iIdx].op_b == HIPBLAS_OP_N ? k[i] : n[i]);
            ldc.push_back(m[i]);
            ldd.push_back(m[i]);
            strideA.push_back(m[i] * k[i]);
            strideB.push_back(m[i] * k[i]);
            strideC.push_back(m[i] * k[i]);
            strideD.push_back(m[i] * k[i]);
        }
        return setProblem(m,
                          n,
                          k,
                          batch_count,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          strideA,
                          strideB,
                          strideC,
                          strideD,
                          epilogue,
                          inputs,
                          m_problem_types[0]);
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&      m,
                                            std::vector<int64_t>&      n,
                                            std::vector<int64_t>&      k,
                                            std::vector<int64_t>&      batch_count,
                                            std::vector<int64_t>&      lda,
                                            std::vector<int64_t>&      ldb,
                                            std::vector<int64_t>&      ldc,
                                            std::vector<int64_t>&      ldd,
                                            std::vector<int64_t>&      strideA,
                                            std::vector<int64_t>&      strideB,
                                            std::vector<int64_t>&      strideC,
                                            std::vector<int64_t>&      strideD,
                                            std::vector<GemmEpilogue>& epilogue,
                                            std::vector<GemmInputs>&   inputs,
                                            GemmProblemType&           problemtype)
    {
        auto rocepilogue = reinterpret_cast<std::vector<rocblaslt::RocGemmEpilogue>*>(&epilogue);
        auto rocinputs   = reinterpret_cast<std::vector<rocblaslt::RocGemmInputs>*>(&inputs);
        std::vector<GemmProblemType> tmptype = {problemtype};
        auto                         rocproblemtype
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemType>*>(&tmptype);
        auto status = RocBlasLtStatusToHIPStatus(rocblaslt_groupedgemm_create_cpp(m,
                                                                                  n,
                                                                                  batch_count,
                                                                                  k,
                                                                                  lda,
                                                                                  ldb,
                                                                                  ldc,
                                                                                  ldd,
                                                                                  strideA,
                                                                                  strideB,
                                                                                  strideC,
                                                                                  strideD,
                                                                                  *rocepilogue,
                                                                                  *rocinputs,
                                                                                  *rocproblemtype,
                                                                                  m_data,
                                                                                  m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types = tmptype;
        }
        return status;
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                            std::vector<void*>&                   alpha,
                                            std::vector<void*>&                   A,
                                            std::vector<hipblasLtMatrixLayout_t>& matA,
                                            std::vector<void*>&                   B,
                                            std::vector<hipblasLtMatrixLayout_t>& matB,
                                            std::vector<void*>&                   beta,
                                            std::vector<void*>&                   C,
                                            std::vector<hipblasLtMatrixLayout_t>& matC,
                                            std::vector<void*>&                   D,
                                            std::vector<hipblasLtMatrixLayout_t>& matD)
    {
        auto matmul_descr_groupedGemm
            = reinterpret_cast<std::vector<rocblaslt_matmul_desc>*>(&matmul_descr);
        auto matA_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matA);
        auto matB_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matB);
        auto matC_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matC);
        auto matD_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matD);
        auto A_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&A);
        auto B_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&B);
        auto C_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&C);
        auto alpha_groupedGemm = reinterpret_cast<std::vector<const void*>*>(&alpha);
        auto beta_groupedGemm  = reinterpret_cast<std::vector<const void*>*>(&beta);
        auto rocproblemtypes
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemType>*>(&m_problem_types);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_groupedgemm_create_cpp(*matmul_descr_groupedGemm,
                                             *alpha_groupedGemm,
                                             *A_groupedGemm,
                                             *matA_groupedGemm,
                                             *B_groupedGemm,
                                             *matB_groupedGemm,
                                             *beta_groupedGemm,
                                             *C_groupedGemm,
                                             *matC_groupedGemm,
                                             D,
                                             *matD_groupedGemm,
                                             (*rocproblemtypes),
                                             m_data,
                                             m_gemm_count));
    }

    std::vector<GemmProblemType> GroupedGemm::getProblemTypes()
    {
        return m_problem_types;
    }

    HIPBLASLT_EXPORT hipblasStatus_t
        GroupedGemm::getDefaultValueForDeviceUserArguments(void* hostDeviceUserArgs)
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return RocBlasLtStatusToHIPStatus(rocblaslt_get_default_user_args(
            (rocblaslt_handle)m_handle, gemmType, m_data, hostDeviceUserArgs));
    }

    HIPBLASLT_EXPORT hipblasStatus_t GroupedGemm::run(void* deviceUserArgs, hipStream_t stream)
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return RocBlasLtStatusToHIPStatus(rocblaslt_run_user_args_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data, deviceUserArgs, stream));
    }

    hipblasStatus_t matmulIsAlgoSupported(hipblasLtHandle_t       handle,
                                          hipblasLtMatmulDesc_t   matmulDesc,
                                          const void*             alpha,
                                          hipblasLtMatrixLayout_t Adesc,
                                          hipblasLtMatrixLayout_t Bdesc,
                                          const void*             beta,
                                          hipblasLtMatrixLayout_t Cdesc,
                                          hipblasLtMatrixLayout_t Ddesc,
                                          hipblasLtMatmulAlgo_t&  algo,
                                          size_t&                 workspaceSizeInBytes)
    try
    {
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_matmul_is_algo_supported((rocblaslt_handle)handle,
                                               (rocblaslt_matmul_desc)matmulDesc,
                                               alpha,
                                               (rocblaslt_matrix_layout)Adesc,
                                               (rocblaslt_matrix_layout)Bdesc,
                                               beta,
                                               (rocblaslt_matrix_layout)Cdesc,
                                               (rocblaslt_matrix_layout)Ddesc,
                                               (rocblaslt_matmul_algo*)&algo,
                                               &workspaceSizeInBytes));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    std::string gemmType2String(GemmType type)
    {
        switch(type)
        {
        case GemmType::HIPBLASLT_GEMM:
            return "gemm";
        case GemmType::HIPBLASLT_GROUPED_GEMM:
            return "grouped gemm";
        }
    }

    hipblasStatus_t getAllAlgos(hipblasLtHandle_t                              handle,
                                GemmType                                       typeGemm,
                                hipblasOperation_t                             opA,
                                hipblasOperation_t                             opB,
                                hipblasltDatatype_t                            typeA,
                                hipblasltDatatype_t                            typeB,
                                hipblasltDatatype_t                            typeC,
                                hipblasltDatatype_t                            typeD,
                                hipblasLtComputeType_t                         typeCompute,
                                std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    try
    {
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_matmul_get_all_algos_cpp((rocblaslt_handle)handle,
                                               static_cast<rocblaslt::RocGemmType>(typeGemm),
                                               opA,
                                               opB,
                                               typeA,
                                               typeB,
                                               typeC,
                                               typeD,
                                               (rocblaslt_compute_type)typeCompute,
                                               *results));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo)
    {
        int* algo_ptr = (int*)algo.data;
        if(*algo_ptr < 0)
        {
            return -1;
        }
        return *algo_ptr;
    }

    hipblasStatus_t
        getAlgosFromIndex(hipblasLtHandle_t                              handle,
                          std::vector<int>&                              algoIndex,
                          std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    {
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_get_algos_from_index_cpp(
            (rocblaslt_handle)handle, algoIndex, *results));
    }

} // End of namespace hipblasltext
