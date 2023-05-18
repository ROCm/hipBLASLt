/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief rocblaslt-functions.h provides Linear Algebra Subprograms
 *  of Level 3, using HIP optimized for AMD GPU hardware.
 */

#pragma once
#ifndef _ROCBLASLT_FUNCTIONS_H_
#define _ROCBLASLT_FUNCTIONS_H_

#include "rocblaslt-types.h"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 *    GENERAL Matrix Multiplication
 * ===========================================================================
 */

/*! \ingroup rocblaslt_module
 *  \brief Matrix multiplication
 *
 *  \details
 *  \p rocblaslt_matmul computes the matrix multiplication of matrices A and B
 to
 *  produce the output matrix D, according to the following operation:
 *  \f[
 *    D := Activation(\alpha \cdot op(A) \cdot op(B) + \beta \cdot C)
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to
 the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only supports the case where D has the same shape of C.
 *
 *  @param[out]
 *  d_D
 *
 *  @param[in]
 *  handle      handle to the rocblaslt library context queue.
 *  @param[in]
 *  matmul_descr
 *  @param[in]
 *  pref
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  A
 *  @param[in]
 *  B
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[in]
 *  C
 *  @param[in]
 *  D
 *  @param[in]
 *  stream
 *  \retval     rocblaslt_status_success the operation completed successfully.
 *  \retval     rocblaslt_status_invalid_handle the library context was not
 initialized.
 *  \retval     rocblaslt_status_invalid_pointer \p alpha, \p A, \p B, \p beta,
 \p C or \p D
 *              pointer is invalid.
 *  \retval     rocblaslt_status_arch_mismatch the device is not supported.
 *  \retval     rocblaslt_status_not_implemented
 */

rocblaslt_status rocblaslt_matmul(rocblaslt_handle             handle,
                                  rocblaslt_matmul_desc        matmul_descr,
                                  const void*                  alpha,
                                  const void*                  A,
                                  rocblaslt_matrix_layout      matA,
                                  const void*                  B,
                                  rocblaslt_matrix_layout      matB,
                                  const void*                  beta,
                                  const void*                  C,
                                  rocblaslt_matrix_layout      matC,
                                  void*                        D,
                                  rocblaslt_matrix_layout      matD,
                                  const rocblaslt_matmul_algo* algo,
                                  void*                        workspace,
                                  size_t                       workspaceSizeInBytes,
                                  hipStream_t                  stream);

rocblaslt_status rocblaslt_gemm_create_cpp(rocblaslt_matmul_desc   matmul_descr,
                                           const void*             alpha,
                                           const void*             A,
                                           rocblaslt_matrix_layout matA,
                                           const void*             B,
                                           rocblaslt_matrix_layout matB,
                                           const void*             beta,
                                           const void*             C,
                                           rocblaslt_matrix_layout matC,
                                           void*                   D,
                                           rocblaslt_matrix_layout matD,
                                           rocblaslt::RocGemm&     gemm);

rocblaslt_status rocblaslt_groupedgemm_create_cpp(std::vector<rocblaslt_matmul_desc>& matmul_descr,
                                                  std::vector<const void*>&           alpha,
                                                  std::vector<const void*>&           A,
                                                  std::vector<rocblaslt_matrix_layout>& matA,
                                                  std::vector<const void*>&             B,
                                                  std::vector<rocblaslt_matrix_layout>& matB,
                                                  std::vector<const void*>&             beta,
                                                  std::vector<const void*>&             C,
                                                  std::vector<rocblaslt_matrix_layout>& matC,
                                                  std::vector<void*>&                   D,
                                                  std::vector<rocblaslt_matrix_layout>& matD,
                                                  rocblaslt::RocGemm&                   gemm);

rocblaslt_status rocblaslt_makeArgument_cpp(const rocblaslt::RocGemmType gemmType,
                                            const rocblaslt_matmul_algo& algo,
                                            void*                        workspace,
                                            hipStream_t                  stream,
                                            std::shared_ptr<void>        gemmData);

rocblaslt_status rocblaslt_run_cpp(rocblaslt_handle       handle,
                                   rocblaslt::RocGemmType gemmType,
                                   std::shared_ptr<void>  gemmData,
                                   hipStream_t            stream);

#ifdef __cplusplus
}
#endif

#endif /* _ROCBLASLT_FUNCTIONS_H_ */
