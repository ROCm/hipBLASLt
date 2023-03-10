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

#include "rocblaslt_mat.hpp"
#include "definitions.h"
#include "handle.h"
#include "rocblaslt_mat_utils.hpp"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief
 *******************************************************************************/

rocblaslt_status rocblaslt_matmul_impl(const rocblaslt_handle       handle,
                                       const rocblaslt_matmul_desc  matmul_descr,
                                       const void*                  A,
                                       const void*                  B,
                                       const void*                  C,
                                       void*                        D,
                                       rocblaslt_matrix_layout      matA,
                                       rocblaslt_matrix_layout      matB,
                                       rocblaslt_matrix_layout      matC,
                                       rocblaslt_matrix_layout      matD,
                                       const void*                  alpha,
                                       const void*                  beta,
                                       const rocblaslt_matmul_algo* algo,
                                       void*                        workspace,
                                       size_t                       workspaceSizeInBytes,
                                       hipStream_t                  stream)
{
    hipblasOperation_t     opA          = matmul_descr->op_A;
    hipblasOperation_t     opB          = matmul_descr->op_B;
    rocblaslt_compute_type compute_type = matmul_descr->compute_type;
    const void*            bias         = nullptr;
    hipblasDatatype_t      bias_type
        = matmul_descr->bias_type == 0 ? matD->type : matmul_descr->bias_type;
    rocblaslt_epilogue epilogue = matmul_descr->epilogue;
    if(is_bias_enabled(epilogue))
        bias = matmul_descr->bias;

    const void* scaleD = nullptr;
    if(matmul_descr->scaleD)
        scaleD = matmul_descr->scaleD;

    // matrix A
    int64_t           num_rows_a     = matA->m;
    int64_t           num_cols_a     = matA->n;
    int64_t           lda            = matA->ld;
    int64_t           batch_stride_a = matA->batch_stride;
    int               num_batches_a  = matA->batch_count;
    hipblasDatatype_t type_a         = matA->type;

    // matrix B
    // int64_t num_rows_b = matB->m;
    // int64_t num_cols_b = matB->n;
    int64_t           ldb            = matB->ld;
    int64_t           batch_stride_b = matB->batch_stride;
    int               num_batches_b  = matB->batch_count;
    hipblasDatatype_t type_b         = matB->type;

    // matrix C
    // int64_t num_rows_c = matC->m;
    // int64_t num_cols_c = matC->n;
    int64_t           ldc            = matC->ld;
    int64_t           batch_stride_c = matC->batch_stride;
    int               num_batches_c  = matC->batch_count;
    hipblasDatatype_t type_c         = matC->type;

    // matrix D
    int64_t           num_rows_d     = matD->m;
    int64_t           num_cols_d     = matD->n;
    int64_t           ldd            = matD->ld;
    int64_t           batch_stride_d = matD->batch_stride;
    int               num_batches_d  = matD->batch_count;
    hipblasDatatype_t type_d         = matD->type;

    int64_t m = num_rows_d;
    int64_t n = num_cols_d;
    int64_t k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

    auto validArgs = validateMatmulArgs(handle,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        B,
                                        beta,
                                        C,
                                        D,
                                        num_batches_a,
                                        num_batches_b,
                                        num_batches_c,
                                        num_batches_d,
                                        batch_stride_a,
                                        batch_stride_b,
                                        batch_stride_c,
                                        batch_stride_d);
    if(validArgs != rocblaslt_status_continue)
        return validArgs;

        // float alpha_f = *(reinterpret_cast<const float *>(alpha));
        // float beta_f = *(reinterpret_cast<const float *>(beta));

#define EX_PARM                                                                          \
    handle, opA, opB, m, n, k, alpha, A, type_a, lda, batch_stride_a, 0, B, type_b, ldb, \
        batch_stride_b, 0, beta, C, type_c, ldc, batch_stride_c, 0, D, type_d, ldd,      \
        batch_stride_d, 0, num_batches_a, true, compute_type, algo, workspace,           \
        workspaceSizeInBytes, bias, scaleD, bias_type, epilogue, stream

    return rocblaslt_matmul_template(EX_PARM);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
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
                                  hipStream_t                  stream)

{
    // Check if handle is valid
    if(handle == nullptr || matmul_descr == nullptr || matA == nullptr || matB == nullptr
       || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return rocblaslt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr
       || D == nullptr)
    {
        log_error(__func__, "invalid data pointer");
        return rocblaslt_status_invalid_pointer;
    }
    if(workspace == nullptr && workspaceSizeInBytes > 0)
    {
        log_error(__func__, "invalid workspace pointer");
        return rocblaslt_status_invalid_pointer;
    }

    if(matA->type != matB->type || matA->type != matC->type || matA->type != matD->type)
    {
        log_error(__func__, "invalid  matrix datatype");
        return rocblaslt_status_type_mismatch;
    }

    if(get_logger_layer_mode() != rocblaslt_layer_mode_none)
    {
        log_api(__func__,
                "A",
                A,
                "Adesc",
                matA,
                "B",
                B,
                "Bdesc",
                matB,
                "C",
                C,
                "Cdesc",
                matC,
                "D",
                D,
                "Ddesc",
                matD,
                "computeDesc",
                matmul_descr,
                "algo",
                algo,
                "workSpace",
                workspace,
                "workSpaceSizeInBytes",
                workspaceSizeInBytes,
                "stream",
                stream);
    }

    if(get_logger_layer_mode() != rocblaslt_layer_mode_none)
    {
        log_trace(__func__,
                  "A",
                  A,
                  "Adesc",
                  rocblaslt_matrix_layout_to_string(matA),
                  "B",
                  B,
                  "Bdesc",
                  rocblaslt_matrix_layout_to_string(matB),
                  "C",
                  C,
                  "Cdesc",
                  rocblaslt_matrix_layout_to_string(matC),
                  "D",
                  D,
                  "Ddesc",
                  rocblaslt_matrix_layout_to_string(matD),
                  "computeDesc",
                  rocblaslt_matmul_desc_to_string(matmul_descr),
                  "workSpace",
                  workspace,
                  "workSpaceSizeInBytes",
                  workspaceSizeInBytes,
                  "alpha",
                  *(reinterpret_cast<const float*>(alpha)),
                  "beta",
                  *(reinterpret_cast<const float*>(beta)),
                  "stream",
                  stream);
    }
    return rocblaslt_matmul_impl(handle,
                                 matmul_descr,
                                 A,
                                 B,
                                 C,
                                 D,
                                 matA,
                                 matB,
                                 matC,
                                 matD,
                                 alpha,
                                 beta,
                                 algo,
                                 workspace,
                                 workspaceSizeInBytes,
                                 stream);
}

#ifdef __cplusplus
}
#endif
