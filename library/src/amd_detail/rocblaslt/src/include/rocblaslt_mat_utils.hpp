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

#pragma once
#ifndef ROCBLASLT_UTILS_HPP
#define ROCBLASLT_UTILS_HPP
#include "handle.h"
#include "utility.hpp"

inline rocblaslt_status getOriginalSizes(hipblasOperation_t opA,
                                         hipblasOperation_t opB,
                                         int64_t            num_rows_a,
                                         int64_t            num_cols_a,
                                         int64_t            num_rows_b,
                                         int64_t            num_cols_b,
                                         int64_t&           m,
                                         int64_t&           n,
                                         int64_t&           k)
{
    // values of num_* are values after been transposed, redirect to before which
    // been transposed. initialized m,n,k by NN.
    m = num_rows_a, n = num_cols_b, k = num_cols_a;
    if(opA == HIPBLAS_OP_T)
    {
        m = num_cols_a;
        k = num_rows_a;
    }
    if(opB == HIPBLAS_OP_T)
    {
        n = num_rows_b;
        if(k != num_cols_b)
        {
            std::cerr << "A, B matrix size are not matched" << std::endl;
            return rocblaslt_status_invalid_size;
        }
    }
    else if(k != num_rows_b)
    {
        std::cerr << "A, B matrix size are not matched" << std::endl;
        return rocblaslt_status_invalid_size;
    }

    return rocblaslt_status_success;
}

/*******************************************************************************
 * Validate Matmul Descr. init Arguments - matrix init.
 ******************************************************************************/
inline rocblaslt_status validateMatmulDescrArgs(rocblaslt_handle       handle,
                                                hipblasOperation_t     opA,
                                                hipblasOperation_t     opB,
                                                int64_t                num_rows_a,
                                                int64_t                num_cols_a,
                                                int64_t                lda,
                                                int64_t                num_rows_b,
                                                int64_t                num_cols_b,
                                                int64_t                ldb,
                                                int64_t                num_rows_c,
                                                int64_t                num_cols_c,
                                                int64_t                ldc,
                                                int64_t                num_rows_d,
                                                int64_t                num_cols_d,
                                                int64_t                ldd,
                                                hipblasltDatatype_t    type_a,
                                                hipblasltDatatype_t    type_b,
                                                hipblasltDatatype_t    type_c,
                                                hipblasltDatatype_t    type_d,
                                                rocblaslt_compute_type compute_type)
{
    // handle must be valid
    if(!handle)
        return rocblaslt_status_invalid_handle;

    // sizes of matrics A,B,C,D must fulfill the matrix multiplication rule.
    // D = A x B + C
    // values of num_* are values after been transposed, redirect to before which
    // been transposed.
    int64_t m, n, k;
    auto    status
        = getOriginalSizes(opA, opB, num_rows_a, num_cols_a, num_rows_b, num_cols_b, m, n, k);
    if(status != rocblaslt_status_success)
        return status;

    if(m != num_rows_c || m != num_rows_d || n != num_cols_c || n != num_cols_d)
    {
        std::cerr << " matrix size is not valid" << std::endl;
        return rocblaslt_status_invalid_size;
    }

    // data type of matrics must be the same
    if(type_a != type_b || type_a != type_c || type_a != type_c)
        return rocblaslt_status_invalid_value;

    switch(type_a)
    {
    case HIPBLASLT_R_32F:
        if(compute_type != rocblaslt_compute_f32)
            return rocblaslt_status_invalid_value;
        break;
    case HIPBLASLT_R_32I:
        if(compute_type != rocblaslt_compute_i32)
            return rocblaslt_status_invalid_value;
        break;
    default:
        return rocblaslt_status_invalid_value;
        break;
    }

    return rocblaslt_status_success;
}

/*******************************************************************************
 * Validate Matmul Arguments
 ******************************************************************************/
inline rocblaslt_status validateMatmulArgs(int64_t     m,
                                           int64_t     n,
                                           int64_t     k,
                                           const void* alpha,
                                           const void* a,
                                           const void* b,
                                           const void* beta,
                                           const void* c,
                                           const void* d,
                                           int         num_batches_a  = 1,
                                           int         num_batches_b  = 1,
                                           int         num_batches_c  = 1,
                                           int         num_batches_d  = 1,
                                           int64_t     batch_stride_a = 0,
                                           int64_t     batch_stride_b = 0,
                                           int64_t     batch_stride_c = 0,
                                           int64_t     batch_stride_d = 0)
{
    // sizes must not be negative
    if(batch_stride_a < 0 || batch_stride_b < 0 || batch_stride_c < 0 || batch_stride_d < 0)
    {
        std::cerr << "matrix and stride size must be positive" << std::endl;
        return rocblaslt_status_invalid_size;
    }

    // number of batches of matrics A,B,C,D must be the same and negative
    if(num_batches_a != num_batches_b || num_batches_a != num_batches_c
       || num_batches_a != num_batches_d || num_batches_a < 1)
    {
        std::cerr << " number of batches of matrics A,B,C,D must be the same and negative"
                  << std::endl;
        return rocblaslt_status_invalid_size;
    }

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by
    // beta
    // Note: we don't check n here since grouped gemm accept some n == 0
    if(!m || !num_batches_a)
        return rocblaslt_status_success;

    if(!beta)
        return rocblaslt_status_invalid_pointer;

    // pointers must be valid
    if(n && ((k && (!a || !b || !alpha)) || !c || !d))
        return rocblaslt_status_invalid_pointer;

    return rocblaslt_status_continue;
}

inline rocblaslt_status rocblaslt_epilogue_valid_args(const rocblaslt_epilogue&  epilogue,
                                                      const int64_t&             num_rows_e,
                                                      const int64_t&             num_cols_e,
                                                      const hipblasltDatatype_t& d_type,
                                                      const hipblasltDatatype_t& original_bias_type,
                                                      const void*                e_ptr,
                                                      const int64_t&             original_lde,
                                                      const int64_t&             original_stride_e,
                                                      const void*                original_bias,
                                                      const void*          original_scaleAlphaVec,
                                                      const void*          alpha,
                                                      void*&               E,
                                                      int64_t&             lde,
                                                      int64_t&             batch_stride_e,
                                                      void*&               bias,
                                                      hipblasltDatatype_t& bias_type,
                                                      void*&               scaleAlphaVec,
                                                      bool&                gradient)
{
    // Set status
    rocblaslt_status status = rocblaslt_status_continue;
    // External update args
    bias_type = original_bias_type == 0 ? d_type : original_bias_type;
    gradient  = is_grad_enabled(epilogue);

    if(is_bias_enabled(epilogue))
        bias = (void*)original_bias;
    else
        bias = nullptr;

    if(original_scaleAlphaVec)
        scaleAlphaVec = (void*)original_scaleAlphaVec; //pointer mode alpha vector pass by alpha
    else
        scaleAlphaVec = nullptr;

    // matrix E
    E = nullptr;
    if(is_e_enabled(epilogue))
    {
        if(e_ptr == nullptr)
            status = rocblaslt_status_invalid_pointer;
        E = (void*)e_ptr;
    }
    lde            = original_lde > 0 ? original_lde : num_rows_e;
    batch_stride_e = original_stride_e > 0 ? original_stride_e : original_lde * num_cols_e;
    if(E != nullptr && ((lde < num_rows_e) || (batch_stride_e < (num_cols_e * num_rows_e))))
        status = rocblaslt_status_invalid_value;
    return status;
}

inline rocblaslt_status rocblaslt_matmul_valid_args(const rocblaslt_matmul_desc matmul_descr,
                                                    const void*                 A,
                                                    const void*                 B,
                                                    const void*                 C,
                                                    const void*                 D,
                                                    rocblaslt_matrix_layout     matA,
                                                    rocblaslt_matrix_layout     matB,
                                                    rocblaslt_matrix_layout     matC,
                                                    rocblaslt_matrix_layout     matD,
                                                    const void*                 alpha,
                                                    const void*                 beta,
                                                    int64_t&                    m,
                                                    int64_t&                    n,
                                                    int64_t&                    k,
                                                    int64_t&                    lda,
                                                    int64_t&                    batch_stride_a,
                                                    int64_t&                    ldb,
                                                    int64_t&                    batch_stride_b,
                                                    int64_t&                    ldc,
                                                    int64_t&                    batch_stride_c,
                                                    int64_t&                    ldd,
                                                    int64_t&                    batch_stride_d,
                                                    int64_t&                    lde,
                                                    int64_t&                    batch_stride_e,
                                                    void*&                      bias,
                                                    hipblasltDatatype_t&        bias_type,
                                                    void*&                      scaleAlphaVec,
                                                    void*&                      E,
                                                    bool&                       gradient,
                                                    rocblaslt_compute_type&     compute_type)
{
    // Internal assign
    hipblasOperation_t opA = matmul_descr->op_A;

    // matrix A
    int64_t num_rows_a    = matA->m;
    int64_t num_cols_a    = matA->n;
    int     num_batches_a = matA->batch_count;
    lda                   = matA->ld;
    batch_stride_a        = matA->batch_stride;

    // matrix B
    int num_batches_b = matB->batch_count;
    ldb               = matB->ld;
    batch_stride_b    = matB->batch_stride;

    // matrix C
    int num_batches_c = matC->batch_count;
    ldc               = matC->ld;
    batch_stride_c    = matC->batch_stride;

    // matrix D
    int64_t num_rows_d    = matD->m;
    int64_t num_cols_d    = matD->n;
    int     num_batches_d = matD->batch_count;
    ldd                   = matD->ld;
    batch_stride_d        = matD->batch_stride;

    compute_type = matmul_descr->compute_type;

    m = num_rows_d;
    n = num_cols_d;
    k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

    auto status = validateMatmulArgs(m,
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

    if(status == rocblaslt_status_continue)
    {
        if(!(matA->type == HIPBLASLT_R_32F && matB->type == HIPBLASLT_R_32F
             && matC->type == HIPBLASLT_R_32F && matD->type == HIPBLASLT_R_32F)
           && compute_type == rocblaslt_compute_f32_fast_xf32)
            status = rocblaslt_status_not_implemented;
        if(!(matA->type == HIPBLASLT_R_8I && matB->type == HIPBLASLT_R_8I
             && matC->type == HIPBLASLT_R_32I && matD->type == HIPBLASLT_R_32I)
           && compute_type == rocblaslt_compute_i32)
            status = rocblaslt_status_not_implemented;
    }
    const void* alphaVecPtr = matmul_descr->pointermode ? alpha : nullptr;
    if(status == rocblaslt_status_continue)
        status = rocblaslt_epilogue_valid_args(matmul_descr->epilogue,
                                               num_rows_d,
                                               num_cols_d,
                                               matD->type,
                                               matmul_descr->bias_type,
                                               matmul_descr->e,
                                               matmul_descr->lde,
                                               matmul_descr->stride_e,
                                               matmul_descr->bias,
                                               alphaVecPtr,
                                               alpha,
                                               E,
                                               lde,
                                               batch_stride_e,
                                               bias,
                                               bias_type,
                                               scaleAlphaVec,
                                               gradient);
    return status;
}

template <typename TiA, typename TiB, typename To, typename Tc>
inline int rocblaslt_get_matmul_alg_config_max_id(hipblasOperation_t opA, hipblasOperation_t opB)
{
    // TODO
    return true;
}
#endif
