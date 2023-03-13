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
#ifndef ROCBLASLT_MAT_HPP
#define ROCBLASLT_MAT_HPP

#include "handle.h"
#include "utility.hpp"

#include "tensile_host.hpp"

template <typename Ti, typename To, typename Tc>
rocblaslt_status rocblaslt_batched_template(rocblaslt_handle             handle,
                                            hipblasOperation_t           trans_a,
                                            hipblasOperation_t           trans_b,
                                            int64_t                      m,
                                            int64_t                      n,
                                            int64_t                      k,
                                            const Tc*                    alpha,
                                            const Ti*                    a,
                                            int64_t                      ld_a,
                                            int64_t                      batch_stride_a,
                                            const Ti*                    b,
                                            int64_t                      ld_b,
                                            int64_t                      batch_stride_b,
                                            const Tc*                    beta,
                                            const To*                    c,
                                            int64_t                      ld_c,
                                            int64_t                      batch_stride_c,
                                            To*                          d,
                                            int64_t                      ld_d,
                                            int64_t                      batch_stride_d,
                                            int64_t                      batch_count,
                                            bool                         strided_batch,
                                            const rocblaslt_matmul_algo* algo,
                                            void*                        workspace,
                                            size_t                       workspaceSizeInBytes,
                                            const void*                  bias,
                                            const Tc*                    scaleD,
                                            hipblasDatatype_t            bias_type,
                                            rocblaslt_epilogue           epilogue,
                                            hipStream_t                  stream)
{
    workspaceSizeInBytes = min(workspaceSizeInBytes, algo->max_workspace_bytes);
    RocblasltContractionProblem<Ti, To, Tc> problem{handle,
                                                    trans_a,
                                                    trans_b,
                                                    m,
                                                    n,
                                                    k,
                                                    alpha,
                                                    a,
                                                    nullptr,
                                                    ld_a,
                                                    batch_stride_a,
                                                    b,
                                                    nullptr,
                                                    ld_b,
                                                    batch_stride_b,
                                                    beta,
                                                    c,
                                                    nullptr,
                                                    ld_c,
                                                    batch_stride_c,
                                                    d,
                                                    nullptr,
                                                    ld_d,
                                                    batch_stride_d,
                                                    batch_count,
                                                    strided_batch,
                                                    bias,
                                                    scaleD,
                                                    bias_type,
                                                    epilogue,
                                                    workspace,
                                                    workspaceSizeInBytes,
                                                    stream};
    return runContractionProblem(algo, problem);
}

template <typename Ti, typename To = Ti, typename Tc = To>
rocblaslt_status rocblaslt_matmul_typecasting(rocblaslt_handle             handle,
                                              hipblasOperation_t           trans_a,
                                              hipblasOperation_t           trans_b,
                                              int64_t                      m,
                                              int64_t                      n,
                                              int64_t                      k,
                                              const void*                  alpha,
                                              const void*                  a,
                                              int64_t                      ld_a,
                                              int64_t                      batch_stride_a,
                                              const void*                  b,
                                              int64_t                      ld_b,
                                              int64_t                      batch_stride_b,
                                              const void*                  beta,
                                              const void*                  c,
                                              int64_t                      ld_c,
                                              int64_t                      batch_stride_c,
                                              void*                        d,
                                              int64_t                      ld_d,
                                              int64_t                      batch_stride_d,
                                              int64_t                      batch_count,
                                              bool                         strided_batch,
                                              const rocblaslt_matmul_algo* algo,
                                              void*                        workspace,
                                              size_t                       workspaceSizeInBytes,
                                              const void*                  bias,
                                              const void*                  scaleD,
                                              hipblasDatatype_t            bias_type,
                                              rocblaslt_epilogue           epilogue,
                                              hipStream_t                  stream)
{
    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(Ti))
       || !isAligned(d, sizeof(To)))
    {
        std::cerr << "memmory is not aligned" << std::endl;
        return rocblaslt_status_invalid_size;
    }
    return rocblaslt_batched_template(handle,
                                      trans_a,
                                      trans_b,
                                      m,
                                      n,
                                      k,
                                      reinterpret_cast<const Tc*>(alpha),
                                      reinterpret_cast<const Ti*>(a),
                                      ld_a,
                                      batch_stride_a,
                                      reinterpret_cast<const Ti*>(b),
                                      ld_b,
                                      batch_stride_b,
                                      reinterpret_cast<const Tc*>(beta),
                                      reinterpret_cast<const To*>(c),
                                      ld_c,
                                      batch_stride_c,
                                      (To*)d,
                                      ld_d,
                                      batch_stride_d,
                                      batch_count,
                                      strided_batch,
                                      algo,
                                      workspace,
                                      workspaceSizeInBytes,
                                      reinterpret_cast<const void*>(bias),
                                      reinterpret_cast<const Tc*>(scaleD),
                                      bias_type,
                                      epilogue,
                                      stream);
}

inline rocblaslt_status rocblaslt_matmul_template(rocblaslt_handle             handle,
                                                  hipblasOperation_t           trans_a,
                                                  hipblasOperation_t           trans_b,
                                                  int64_t                      m,
                                                  int64_t                      n,
                                                  int64_t                      k,
                                                  const void*                  alpha,
                                                  const void*                  a,
                                                  hipblasDatatype_t            a_type,
                                                  int64_t                      ld_a,
                                                  int64_t                      batch_stride_a,
                                                  const void*                  b,
                                                  hipblasDatatype_t            b_type,
                                                  int64_t                      ld_b,
                                                  int64_t                      batch_stride_b,
                                                  const void*                  beta,
                                                  const void*                  c,
                                                  hipblasDatatype_t            c_type,
                                                  int64_t                      ld_c,
                                                  int64_t                      batch_stride_c,
                                                  void*                        d,
                                                  hipblasDatatype_t            d_type,
                                                  int64_t                      ld_d,
                                                  int64_t                      batch_stride_d,
                                                  int64_t                      batch_count,
                                                  bool                         strided_batch,
                                                  rocblaslt_compute_type       compute_type,
                                                  const rocblaslt_matmul_algo* algo,
                                                  void*                        workspace,
                                                  size_t                       workspaceSizeInBytes,
                                                  const void*                  bias,
                                                  const void*                  scaleD,
                                                  hipblasDatatype_t            bias_type,
                                                  rocblaslt_epilogue           epilogue,
                                                  hipStream_t                  stream)
{
    rocblaslt_status rs_status = rocblaslt_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                       \
    handle, trans_a, trans_b, m, n, k, alpha, a, ld_a, batch_stride_a, b, ld_b, batch_stride_b,   \
        beta, c, ld_c, batch_stride_c, d, ld_d, batch_stride_d, batch_count, strided_batch, algo, \
        workspace, workspaceSizeInBytes, bias, scaleD, bias_type, epilogue, stream

    if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F)
    {
        if(c_type == HIPBLAS_R_32F && d_type == HIPBLAS_R_32F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_matmul_typecasting<float, float, float>(EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F)
    {
        if(c_type == HIPBLAS_R_16F && d_type == HIPBLAS_R_16F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_matmul_typecasting<rocblaslt_half, rocblaslt_half, float>(
                    EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B)
    {
        if(c_type == HIPBLAS_R_16B && d_type == HIPBLAS_R_16B)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status
                    = rocblaslt_matmul_typecasting<rocblaslt_bfloat16, rocblaslt_bfloat16, float>(
                        EX_TYPECASTING_PARM);
            }
        }
    }
    else
    {
        rs_status = rocblaslt_status_not_implemented;
    }

    return rs_status;
}
#endif
