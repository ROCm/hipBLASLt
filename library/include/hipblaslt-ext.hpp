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

/*! \file
 *  \brief hipblaslt-ext.hpp provides general matrix-matrix operations with
 *  C++ style flexible API to let user set attributes for solution selection.
 */

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled
//! unmodified through either AMD HCC or NVCC. Key features tend to be in the
//! spirit and terminology of CUDA, but with a portable path to other
//! accelerators as well.
//

#pragma once
#include "hipblaslt/hipblaslt.h"

#include <memory>
#include <vector>

namespace hipblaslt_ext
{

    /*! \ingroup types_module
     *  \brief It is an enumerated type used to specific the type of the gemm problem in hipblasLtExt APIs.
     */
    enum class GemmType
    {
        HIPBLASLT_GEMM             = 1,
        HIPBLASLT_GROUPED_GEMM     = 2,
        HIPBLASLT_GEMMTYPE_UNKNOWN = 3,
    };

    /*! \ingroup types_module
     *  \brief hipblasLt extension instance for gemm problems.
     *
     * \details The instance can be used to create arguments to compute the matrix
     * multiplication of matrices A and B to produce the output matrix D, according
     * to the following operation: \p D = \p alpha*( \p A *\p B) + \p beta*( \p C ),
     * where \p A, \p B, and \p C are input matrices, and \p alpha and \p beta are
     * input scalars.
     */
    class Gemm
    {
    public:
        // Use when calling bridge layer APIs
        HIPBLASLT_EXPORT explicit Gemm(hipblasLtHandle_t handle, size_t maxWorkspaceBytes);

        /*! \ingroup library_module
        *  \brief Set the gemm problem from hipblasLt structures
        *
        *  \details
        *  This function set the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  matmulDesc              Handle to a previously created matrix multiplication
        * descriptor of type \ref hipblasLtMatmulDesc_t .
        *  @param[in]
        *  alpha,beta              Pointers to the scalars used in the multiplication.
        *  @param[in]
        *  matA,matB,matC,matD Handles to the previously created matrix layout
        * descriptors of the type \ref hipblasLtMatrixLayout_t .
        *  @param[in]
        *  A,B,C                   Pointers to the GPU memory associated with the
        * corresponding descriptors \p matA, \p matB and \p matC .
        *  @param[out]
        *  D                       Pointer to the GPU memory associated with the
        * descriptor \p matD .
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t setProblemFromhipBlasLt(hipblasLtMatmulDesc_t   matmul_descr,
                                                const void*             alpha,
                                                const void*             A,
                                                hipblasLtMatrixLayout_t matA,
                                                const void*             B,
                                                hipblasLtMatrixLayout_t matB,
                                                const void*             beta,
                                                const void*             C,
                                                hipblasLtMatrixLayout_t matC,
                                                void*                   D,
                                                hipblasLtMatrixLayout_t matD);

        /*! \ingroup library_module
        *  \brief Set the grouped gemm problem from hipblasLt structures
        *
        *  \details
        *  This function set the problem from hipblasLt structures. For more information
        * about the structures, see hipblasLtMatmul for more information.
        *
        *  @param[in]
        *  matmulDesc              Vectors of handle to a previously created matrix
        * multiplication descriptor of type \ref hipblasLtMatmulDesc_t .
        *  @param[in]
        *  alpha,beta              Vectors of float used in the multiplication.
        *  @param[in]
        *  matA,matB,matC,matD Vectors of handle to the previously created matrix
        * layout descriptors of the type \ref hipblasLtMatrixLayout_t .
        *  @param[in]
        *  A,B,C                   Vectors of pointer to the GPU memory associated
        * with the corresponding descriptors \p matA, \p matB and \p matC .
        *  @param[out]
        *  D                       Vector of pointer to the GPU memory associated with
        * the descriptor \p matD .
        *
        *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
        * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
        * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
        * the configured operation cannot be run using the selected device. \retval
        * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
        * selected device doesn't support the configured operation. \retval
        * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
        * conflict or in an impossible configuration.
        *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
        * initialized.
        */
        HIPBLASLT_EXPORT
        hipblasStatus_t
            setGroupedProblemFromhipBlasLt(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
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

        HIPBLASLT_EXPORT GemmType getGemmType();
        HIPBLASLT_EXPORT size_t   getGemmCount();
        HIPBLASLT_EXPORT size_t   getWorkspaceBytes();

    private:
        GemmType m_gemm_type       = GemmType::HIPBLASLT_GEMMTYPE_UNKNOWN;
        size_t   m_gemm_count      = 0;
        size_t   m_workspace_bytes = 0;

        hipblasLtHandle_t     m_handle;
        std::shared_ptr<void> m_data;
    };

    /*******************************************************************************
     * Ext APIs
     ******************************************************************************/

    HIPBLASLT_EXPORT std::string gemmType2String(GemmType type);

    /*! \ingroup library_module
     *  \brief Retrieve the possible algorithms
     *
     *  \details
     *  This function retrieves the possible algorithms for the matrix multiply
     * operation hipblasLtMatmul() function with the given data and compute tpye.
     * The output is placed in heuristicResult in the order of increasing
     * estimated compute time.
     *
     *  @param[in]
     *  handle                  Pointer to the allocated hipBLASLt handle for the
     * hipBLASLt context. See \ref hipblasLtHandle_t .
     *  @param[in]
     *  hipblasLtExtGemmTypeEnum_t Gemm type. ex. GEMM, GROUPED_GEMM.
     *  @param[in]
     *  opA, opB Transpose settings of A, B.
     *  @param[in]
     *  typeA,typeB,typeC,typeD The data type of matrix A, B, C, D.
     *  @param[in]
     *  typeCompute             The compute type.
     *  @param[out]
     *  heuristicResult The algorithm heuristic vector.
     *
     *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. Inspect
     * returnedAlgoCount > 0.state for the status of the
     * results. \retval HIPBLAS_STATUS_NOT_SUPPORTED     If no heuristic function
     * available for current configuration. \retval HIPBLAS_STATUS_INVALID_VALUE If
     * no solution is found.
     */
    HIPBLASLT_EXPORT
    hipblasStatus_t getAllAlgos(hipblasLtHandle_t                              handle,
                                GemmType                                       typeGemm,
                                hipblasOperation_t                             opA,
                                hipblasOperation_t                             opB,
                                hipblasDatatype_t                              typeA,
                                hipblasDatatype_t                              typeB,
                                hipblasDatatype_t                              typeC,
                                hipblasDatatype_t                              typeD,
                                hipblasLtComputeType_t                         typeCompute,
                                std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

    /*! \ingroup library_module
     *  \brief Check if the algorithm supports the problem. (For hipblasLt API)
     *
     *  \details
     *  This function updates the problem saved inside the algorithm if the problem is
     * supported. The required workspaceSizeInBytes is also returned.
     *
     *  @param[in]
     *  handle                  Pointer to the allocated hipBLASLt handle for the
     * hipBLASLt context. See \ref hipblasLtHandle_t .
     *  @param[in]
     *  matmulDesc              Handle to a previously created matrix multiplication
     * descriptor of type \ref hipblasLtMatmulDesc_t .
     *  @param[in]
     *  alpha,beta              Pointers to the scalars used in the multiplication.
     *  @param[in]
     *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
     * descriptors of the type \ref hipblasLtMatrixLayout_t .
     *  @param[in]
     *  algo The algorithm heuristic.
     *  @param[out]
     *  workspaceSizeInBytes Return the required workspace size.
     *
     *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. The problem is
     * supported by the algorithm.
     * results. \retval HIPBLAS_STATUS_INVALID_VALUE     The problem is not supported.
     */
    HIPBLASLT_EXPORT
    hipblasStatus_t matmulIsAlgoSupported(hipblasLtHandle_t       handle,
                                          hipblasLtMatmulDesc_t   matmulDesc,
                                          const void*             alpha,
                                          hipblasLtMatrixLayout_t Adesc,
                                          hipblasLtMatrixLayout_t Bdesc,
                                          const void*             beta,
                                          hipblasLtMatrixLayout_t Cdesc,
                                          hipblasLtMatrixLayout_t Ddesc,
                                          hipblasLtMatmulAlgo_t&  algo,
                                          size_t&                 workspaceSizeInBytes);

    /*! \ingroup library_module
    *  \brief Check if the algorithm supports the problem. (For hipblaslt extension API)
    *
    *  \details
    *  This function updates the problem saved inside the algorithm if the problem is
    * supported. The required workspaceSizeInBytes is also returned.
    *
    *  @param[in]
    *  gemm The hipblasLt extension instance.
    *  @param[in]
    *  algo The algorithm heuristic.
    *  @param[out]
    *  workspaceSizeInBytes Return the required workspace size.
    *
    *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. The problem is
    * supported by the algorithm.
    * results. \retval HIPBLAS_STATUS_INVALID_VALUE     The problem is not supported.
    */
    HIPBLASLT_EXPORT
    hipblasStatus_t
        isAlgoSupported(Gemm& gemm, hipblasLtMatmulAlgo_t& algo, size_t& workspaceSizeInBytes);

    HIPBLASLT_EXPORT
    hipblasStatus_t
        algoGetHeuristic(Gemm&                                          gemm,
                         const int                                      requestedAlgoCount,
                         std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

    /*! \ingroup library_module
    *  \brief Create kernel arguments from a given hipblaslt_ext::Gemm instance.
    *
    *  \details
    *  This function creates kernel arguemtns from a given hipblaslt_ext::Gemm instance
    * then saves the arguments inside the instance.
    *
    *  @param[in]
    *  gemm                   The hipblaslt_ext::Gemm instance.
    *  @param[in]
    *  algo                    Handle for matrix multiplication algorithm to be
    * used. See \ref hipblasLtMatmulAlgo_t. When NULL, an implicit heuritics query
    * with default search preferences will be performed to determine actual
    * algorithm to use.
    *  @param[in]
    *  workspace               Pointer to the workspace buffer allocated in the GPU
    * memory. Pointer must be 16B aligned (that is, lowest 4 bits of address must
    * be 0).
    *  @param[in]
    *  stream                  The HIP stream where all the GPU work will be
    * submitted.
    *
    *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
    * successfully. \retval HIPBLAS_STATUS_INVALID_VALUE If the gemm_count = 0.
    */
    HIPBLASLT_EXPORT
    hipblasStatus_t makeArgument(Gemm&                        gemm,
                                 const hipblasLtMatmulAlgo_t& algo,
                                 void*                        workspace,
                                 hipStream_t                  stream);

    /*! \ingroup library_module
    *  \brief Execute the kernel arguments stored inside the hipblaslt_ext::Gemm
    * instance.
    *
    *  @param[in]
    *  gemm                   The hipblaslt_ext::Gemm instance.
    *  @param[in]
    *  stream                  The HIP stream where all the GPU work will be
    * submitted.
    *
    *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
    * successfully.
    */
    HIPBLASLT_EXPORT
    hipblasStatus_t run(Gemm& gemm, hipStream_t stream);
} // End of namespace hipblasltext
