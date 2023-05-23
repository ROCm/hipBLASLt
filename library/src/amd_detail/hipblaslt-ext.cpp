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
#include <iostream>
#include <rocblaslt.h>

namespace hipblaslt_ext
{
    void GemmPreference::setMaxWorkspaceBytes(size_t workspaceBytes)
    {
        m_workspace_bytes = workspaceBytes;
    }

    const size_t GemmPreference::getMaxWorkspaceBytes() const
    {
        return m_workspace_bytes;
    }

    template <GemmType GemmTypeT>
    HIPBLASLT_EXPORT Gemm<GemmTypeT>::Gemm(hipblasLtHandle_t      handle,
                                           hipblasOperation_t     opA,
                                           hipblasOperation_t     opB,
                                           hipblasDatatype_t      typeA,
                                           hipblasDatatype_t      typeB,
                                           hipblasDatatype_t      typeC,
                                           hipblasDatatype_t      typeD,
                                           hipblasLtComputeType_t typeCompute)
        : m_gemm_type(GemmTypeT)
        , m_handle(handle)
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

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    HIPBLASLT_EXPORT Gemm<GemmTypeT>::Gemm(hipblasLtHandle_t       handle,
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
        : m_gemm_type(GemmTypeT)
        , m_handle(handle)
    {
        auto status = setProblemFromhipBlasLt(
            matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
    }

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    HIPBLASLT_EXPORT Gemm<GemmTypeT>::Gemm(hipblasLtHandle_t                     handle,
                                           std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                           std::vector<float>&                   alpha,
                                           std::vector<void*>&                   A,
                                           std::vector<hipblasLtMatrixLayout_t>& matA,
                                           std::vector<void*>&                   B,
                                           std::vector<hipblasLtMatrixLayout_t>& matB,
                                           std::vector<float>&                   beta,
                                           std::vector<void*>&                   C,
                                           std::vector<hipblasLtMatrixLayout_t>& matC,
                                           std::vector<void*>&                   D,
                                           std::vector<hipblasLtMatrixLayout_t>& matD)
        : m_gemm_type(GemmTypeT)
        , m_handle(handle)
    {
        auto status = setProblemFromhipBlasLt(
            matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
    }

    template <GemmType GemmTypeT>
    GemmType Gemm<GemmTypeT>::getGemmType()
    {
        return m_gemm_type;
    }

    template <GemmType GemmTypeT>
    size_t Gemm<GemmTypeT>::getGemmCount()
    {
        return m_gemm_count;
    }

    template <GemmType GemmTypeT>
    std::vector<GemmProblemType> Gemm<GemmTypeT>::getProblemTypes()
    {
        return m_problem_types;
    }

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    hipblasStatus_t Gemm<GemmTypeT>::setProblem(int64_t       m,
                                                int64_t       n,
                                                int64_t       k,
                                                int64_t       batch_count,
                                                GemmEpilogue& epilogue,
                                                GemmInputs&   inputs)
    {
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

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    hipblasStatus_t Gemm<GemmTypeT>::setProblem(int64_t          m,
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
        auto rocepilogue    = reinterpret_cast<rocblaslt::RocGemmEpilogue*>(&epilogue);
        auto rocepinputs    = reinterpret_cast<rocblaslt::RocGemmInputs*>(&inputs);
        auto rocproblemtype = reinterpret_cast<rocblaslt::RocGemmProblemType*>(&problemtype);
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

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    hipblasStatus_t Gemm<GemmTypeT>::setProblem(std::vector<int64_t>&      m,
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
                          m_problem_types);
    }

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    hipblasStatus_t Gemm<GemmTypeT>::setProblem(std::vector<int64_t>&         m,
                                                std::vector<int64_t>&         n,
                                                std::vector<int64_t>&         k,
                                                std::vector<int64_t>&         batch_count,
                                                std::vector<int64_t>&         lda,
                                                std::vector<int64_t>&         ldb,
                                                std::vector<int64_t>&         ldc,
                                                std::vector<int64_t>&         ldd,
                                                std::vector<int64_t>&         strideA,
                                                std::vector<int64_t>&         strideB,
                                                std::vector<int64_t>&         strideC,
                                                std::vector<int64_t>&         strideD,
                                                std::vector<GemmEpilogue>&    epilogue,
                                                std::vector<GemmInputs>&      inputs,
                                                std::vector<GemmProblemType>& problemtype)
    {
        auto rocepilogue = reinterpret_cast<std::vector<rocblaslt::RocGemmEpilogue>*>(&epilogue);
        auto rocinputs   = reinterpret_cast<std::vector<rocblaslt::RocGemmInputs>*>(&inputs);
        auto rocproblemtype
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemType>*>(&problemtype);
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
            m_problem_types = problemtype;
        }
        return status;
    }

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    hipblasStatus_t Gemm<GemmTypeT>::setProblemFromhipBlasLt(hipblasLtMatmulDesc_t   matmul_descr,
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
        rocblaslt::RocGemm gemm;
        gemm.setGemmType(static_cast<rocblaslt::RocGemmType>(m_gemm_type));
        gemm.setHandle((rocblaslt_handle)m_handle);
        auto status = RocBlasLtStatusToHIPStatus(
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
                                      gemm));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            auto hipGemm    = reinterpret_cast<Gemm<T>*>(&gemm);
            m_gemm_type     = hipGemm->getGemmType();
            m_gemm_count    = hipGemm->getGemmCount();
            m_data          = hipGemm->m_data;
            m_problem_types = hipGemm->getProblemTypes();
        }
        return status;
    }

    template <GemmType GemmTypeT>
    template <GemmType T, typename>
    hipblasStatus_t
        Gemm<GemmTypeT>::setProblemFromhipBlasLt(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                                 std::vector<float>&                   alpha,
                                                 std::vector<void*>&                   A,
                                                 std::vector<hipblasLtMatrixLayout_t>& matA,
                                                 std::vector<void*>&                   B,
                                                 std::vector<hipblasLtMatrixLayout_t>& matB,
                                                 std::vector<float>&                   beta,
                                                 std::vector<void*>&                   C,
                                                 std::vector<hipblasLtMatrixLayout_t>& matC,
                                                 std::vector<void*>&                   D,
                                                 std::vector<hipblasLtMatrixLayout_t>& matD)
    {
        auto matmul_descr_groupedGemm
            = reinterpret_cast<std::vector<rocblaslt_matmul_desc>*>(&matmul_descr);
        auto matA_groupedGemm = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matA);
        auto matB_groupedGemm = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matB);
        auto matC_groupedGemm = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matC);
        auto matD_groupedGemm = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matD);
        auto A_groupedGemm    = reinterpret_cast<std::vector<const void*>*>(&A);
        auto B_groupedGemm    = reinterpret_cast<std::vector<const void*>*>(&B);
        auto C_groupedGemm    = reinterpret_cast<std::vector<const void*>*>(&C);
        std::vector<const void*> alpha_groupedGemm, beta_groupedGemm;
        for(int i = 0; i < matmul_descr.size(); i++)
        {
            alpha_groupedGemm.push_back((const void*)(&(alpha[i])));
            beta_groupedGemm.push_back((const void*)(&(beta[i])));
        }
        rocblaslt::RocGemm gemm;
        gemm.setGemmType(static_cast<rocblaslt::RocGemmType>(m_gemm_type));
        gemm.setHandle((rocblaslt_handle)m_handle);
        auto status
            = RocBlasLtStatusToHIPStatus(rocblaslt_groupedgemm_create_cpp(*matmul_descr_groupedGemm,
                                                                          alpha_groupedGemm,
                                                                          *A_groupedGemm,
                                                                          *matA_groupedGemm,
                                                                          *B_groupedGemm,
                                                                          *matB_groupedGemm,
                                                                          beta_groupedGemm,
                                                                          *C_groupedGemm,
                                                                          *matC_groupedGemm,
                                                                          D,
                                                                          *matD_groupedGemm,
                                                                          gemm));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            auto hipGemm    = reinterpret_cast<Gemm<T>*>(&gemm);
            m_gemm_type     = hipGemm->getGemmType();
            m_gemm_count    = hipGemm->getGemmCount();
            m_data          = hipGemm->m_data;
            m_problem_types = hipGemm->getProblemTypes();
        }
        return status;
    }

    template <GemmType GemmTypeT>
    hipblasStatus_t Gemm<GemmTypeT>::algoGetHeuristic(
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

    template <GemmType GemmTypeT>
    hipblasStatus_t Gemm<GemmTypeT>::isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                     size_t&                workspaceSizeInBytes)
    try
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo  = reinterpret_cast<rocblaslt_matmul_algo*>(&algo);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_is_algo_supported_cpp(gemmType, m_data, *rocalgo, workspaceSizeInBytes));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    template <GemmType GemmTypeT>
    hipblasStatus_t Gemm<GemmTypeT>::initialize(const hipblasLtMatmulAlgo_t& algo,
                                                void*                        workspace,
                                                hipStream_t                  stream)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo  = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_makeArgument_cpp(gemmType, *rocalgo, workspace, stream, m_data));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    template <GemmType GemmTypeT>
    hipblasStatus_t Gemm<GemmTypeT>::run(hipStream_t stream)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_run_cpp((rocblaslt_handle)m_handle, gemmType, m_data, stream));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
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
                                hipblasDatatype_t                              typeA,
                                hipblasDatatype_t                              typeB,
                                hipblasDatatype_t                              typeC,
                                hipblasDatatype_t                              typeD,
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

    template class HIPBLASLT_EXPORT           Gemm<GemmType::HIPBLASLT_GEMM>;
    template class HIPBLASLT_EXPORT           Gemm<GemmType::HIPBLASLT_GROUPED_GEMM>;
    template HIPBLASLT_EXPORT hipblasStatus_t Gemm<GemmType::HIPBLASLT_GEMM>::setProblem(
        int64_t m, int64_t n, int64_t b, int64_t k, GemmEpilogue& epilogue, GemmInputs& inputs);
    template HIPBLASLT_EXPORT hipblasStatus_t
        Gemm<GemmType::HIPBLASLT_GEMM>::setProblem(int64_t          m,
                                                   int64_t          n,
                                                   int64_t          b,
                                                   int64_t          k,
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
                                                   GemmProblemType& problemtype);
    template HIPBLASLT_EXPORT hipblasStatus_t
        Gemm<GemmType::HIPBLASLT_GEMM>::setProblemFromhipBlasLt(hipblasLtMatmulDesc_t matmul_descr,
                                                                const void*           alpha,
                                                                const void*           A,
                                                                hipblasLtMatrixLayout_t matA,
                                                                const void*             B,
                                                                hipblasLtMatrixLayout_t matB,
                                                                const void*             beta,
                                                                const void*             C,
                                                                hipblasLtMatrixLayout_t matC,
                                                                void*                   D,
                                                                hipblasLtMatrixLayout_t matD);
    template HIPBLASLT_EXPORT hipblasStatus_t
        Gemm<GemmType::HIPBLASLT_GROUPED_GEMM>::setProblem(std::vector<int64_t>&      m,
                                                           std::vector<int64_t>&      n,
                                                           std::vector<int64_t>&      b,
                                                           std::vector<int64_t>&      k,
                                                           std::vector<GemmEpilogue>& epilogue,
                                                           std::vector<GemmInputs>&   inputs);
    template HIPBLASLT_EXPORT hipblasStatus_t Gemm<GemmType::HIPBLASLT_GROUPED_GEMM>::setProblem(
        std::vector<int64_t>&         m,
        std::vector<int64_t>&         n,
        std::vector<int64_t>&         b,
        std::vector<int64_t>&         k,
        std::vector<int64_t>&         lda,
        std::vector<int64_t>&         ldb,
        std::vector<int64_t>&         ldc,
        std::vector<int64_t>&         ldd,
        std::vector<int64_t>&         strideA,
        std::vector<int64_t>&         strideB,
        std::vector<int64_t>&         strideC,
        std::vector<int64_t>&         strideD,
        std::vector<GemmEpilogue>&    epilogue,
        std::vector<GemmInputs>&      inputs,
        std::vector<GemmProblemType>& problemtype);
    template HIPBLASLT_EXPORT hipblasStatus_t
        Gemm<GemmType::HIPBLASLT_GROUPED_GEMM>::setProblemFromhipBlasLt(
            std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
            std::vector<float>&                   alpha,
            std::vector<void*>&                   A,
            std::vector<hipblasLtMatrixLayout_t>& matA,
            std::vector<void*>&                   B,
            std::vector<hipblasLtMatrixLayout_t>& matB,
            std::vector<float>&                   beta,
            std::vector<void*>&                   C,
            std::vector<hipblasLtMatrixLayout_t>& matC,
            std::vector<void*>&                   D,
            std::vector<hipblasLtMatrixLayout_t>& matD);
} // End of namespace hipblasltext
