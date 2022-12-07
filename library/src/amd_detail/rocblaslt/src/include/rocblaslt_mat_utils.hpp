/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
                                                hipblasDatatype_t      type_a,
                                                hipblasDatatype_t      type_b,
                                                hipblasDatatype_t      type_c,
                                                hipblasDatatype_t      type_d,
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
    case HIPBLAS_R_32F:
        if(compute_type != rocblaslt_compute_f32)
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
inline rocblaslt_status validateMatmulArgs(rocblaslt_handle handle,
                                           int64_t          m,
                                           int64_t          n,
                                           int64_t          k,
                                           const void*      alpha,
                                           const void*      a,
                                           const void*      b,
                                           const void*      beta,
                                           const void*      c,
                                           const void*      d,
                                           int              num_batches_a  = 1,
                                           int              num_batches_b  = 1,
                                           int              num_batches_c  = 1,
                                           int              num_batches_d  = 1,
                                           int64_t          batch_stride_a = 0,
                                           int64_t          batch_stride_b = 0,
                                           int64_t          batch_stride_c = 0,
                                           int64_t          batch_stride_d = 0)
{
    // handle must be valid
    if(!handle)
        return rocblaslt_status_invalid_handle;

    // sizes must not be negative
    if(batch_stride_a < 0 || batch_stride_b < 0 || batch_stride_c < 0 || batch_stride_d < 0)
    {
        std::cerr << "matrix and stride size must be posstive" << std::endl;
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
    if(!m || !n || !num_batches_a)
        return rocblaslt_status_success;

    if(!beta)
        return rocblaslt_status_invalid_pointer;

    // pointers must be valid
    if((k && (!a || !b || !alpha)) || !c || !d)
        return rocblaslt_status_invalid_pointer;

    return rocblaslt_status_continue;
}

template <typename Ti, typename To, typename Tc>
inline int rocblaslt_get_matmul_alg_config_max_id(hipblasOperation_t opA, hipblasOperation_t opB)
{
    // TODO
    return true;
}
#endif
