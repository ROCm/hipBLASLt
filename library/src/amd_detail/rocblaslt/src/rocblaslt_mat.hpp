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
                                            Tc*                          e,
                                            int64_t                      ld_e,
                                            int64_t                      batch_stride_e,
                                            int64_t                      batch_count,
                                            bool                         strided_batch,
                                            bool                         grouped_gemm,
                                            bool                         gradient,
                                            const rocblaslt_matmul_algo* algo,
                                            void*                        workspace,
                                            size_t                       workspaceSizeInBytes,
                                            const void*                  bias,
                                            const Tc*                    scaleDVec,
                                            hipblasDatatype_t            bias_type,
                                            rocblaslt_epilogue           epilogue,
                                            std::shared_ptr<void>        gemmData,
                                            hipStream_t                  stream)
{
    workspaceSizeInBytes = min(workspaceSizeInBytes, algo->max_workspace_bytes);
    RocblasltContractionProblem<Ti, To, Tc> problem{trans_a,
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
                                                    e,
                                                    nullptr,
                                                    ld_e,
                                                    batch_stride_e,
                                                    batch_count,
                                                    strided_batch,
                                                    grouped_gemm,
                                                    gradient,
                                                    bias,
                                                    scaleDVec,
                                                    bias_type,
                                                    epilogue,
                                                    workspace,
                                                    workspaceSizeInBytes,
                                                    stream};
    return runContractionProblem(handle, algo, problem, gemmData);
}

template <typename Ti, typename To, typename Tc>
rocblaslt_status rocblaslt_gemm_create_batched_template(hipblasOperation_t     trans_a,
                                                        hipblasOperation_t     trans_b,
                                                        int64_t                m,
                                                        int64_t                n,
                                                        int64_t                k,
                                                        const Tc*              alpha,
                                                        const Ti*              a,
                                                        int64_t                ld_a,
                                                        int64_t                batch_stride_a,
                                                        const Ti*              b,
                                                        int64_t                ld_b,
                                                        int64_t                batch_stride_b,
                                                        const Tc*              beta,
                                                        const To*              c,
                                                        int64_t                ld_c,
                                                        int64_t                batch_stride_c,
                                                        To*                    d,
                                                        int64_t                ld_d,
                                                        int64_t                batch_stride_d,
                                                        Tc*                    e,
                                                        int64_t                ld_e,
                                                        int64_t                batch_stride_e,
                                                        int64_t                batch_count,
                                                        bool                   strided_batch,
                                                        bool                   grouped_gemm,
                                                        bool                   gradient,
                                                        const void*            bias,
                                                        const Tc*              scaleDVec,
                                                        hipblasDatatype_t      bias_type,
                                                        rocblaslt_epilogue     epilogue,
                                                        std::shared_ptr<void>& gemmData,
                                                        size_t&                gemmCount)
{
    RocblasltContractionProblem<Ti, To, Tc> problem{trans_a,
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
                                                    e,
                                                    nullptr,
                                                    ld_e,
                                                    batch_stride_e,
                                                    batch_count,
                                                    strided_batch,
                                                    grouped_gemm,
                                                    gradient,
                                                    bias,
                                                    scaleDVec,
                                                    bias_type,
                                                    epilogue,
                                                    nullptr,
                                                    0,
                                                    0};
    return gemmCreate(problem, gemmData, gemmCount);
}

template <typename Ti, typename To, typename Tc>
rocblaslt_status
    rocblaslt_groupedgemm_create_batched_template(hipblasOperation_t               trans_a,
                                                  hipblasOperation_t               trans_b,
                                                  std::vector<int64_t>&            m,
                                                  std::vector<int64_t>&            n,
                                                  std::vector<int64_t>&            k,
                                                  std::vector<const Tc*>&          alpha,
                                                  std::vector<const Ti*>&          a,
                                                  std::vector<int64_t>&            ld_a,
                                                  std::vector<int64_t>&            batch_stride_a,
                                                  std::vector<const Ti*>&          b,
                                                  std::vector<int64_t>&            ld_b,
                                                  std::vector<int64_t>&            batch_stride_b,
                                                  std::vector<const Tc*>&          beta,
                                                  std::vector<const To*>&          c,
                                                  std::vector<int64_t>&            ld_c,
                                                  std::vector<int64_t>&            batch_stride_c,
                                                  std::vector<To*>&                d,
                                                  std::vector<int64_t>&            ld_d,
                                                  std::vector<int64_t>&            batch_stride_d,
                                                  std::vector<int64_t>&            batch_count,
                                                  bool                             strided_batch,
                                                  bool                             grouped_gemm,
                                                  std::vector<const void*>&        bias,
                                                  std::vector<const Tc*>&          scaleDVec,
                                                  std::vector<hipblasDatatype_t>&  bias_type,
                                                  std::vector<rocblaslt_epilogue>& epilogue,
                                                  std::shared_ptr<void>&           gemmData,
                                                  size_t&                          gemmCount)
{
    std::vector<RocblasltContractionProblem<Ti, To, Tc>> problems;

    for(int i = 0; i < m.size(); i++)
    {
        problems.push_back(RocblasltContractionProblem<Ti, To, Tc>{trans_a,
                                                                   trans_b,
                                                                   m[i],
                                                                   n[i],
                                                                   k[i],
                                                                   alpha[i],
                                                                   a[i],
                                                                   nullptr,
                                                                   ld_a[i],
                                                                   batch_stride_a[i],
                                                                   b[i],
                                                                   nullptr,
                                                                   ld_b[i],
                                                                   batch_stride_b[i],
                                                                   beta[i],
                                                                   c[i],
                                                                   nullptr,
                                                                   ld_c[i],
                                                                   batch_stride_c[i],
                                                                   d[i],
                                                                   nullptr,
                                                                   ld_d[i],
                                                                   batch_stride_d[i],
                                                                   nullptr,
                                                                   nullptr,
                                                                   0,
                                                                   0,
                                                                   batch_count[i],
                                                                   strided_batch,
                                                                   grouped_gemm,
                                                                   false,
                                                                   bias[i],
                                                                   scaleDVec[i],
                                                                   bias_type[i],
                                                                   epilogue[i],
                                                                   nullptr,
                                                                   0,
                                                                   0});
    }
    return groupedGemmCreate(problems, gemmData, gemmCount);
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
                                              void*                        e,
                                              int64_t                      ld_e,
                                              int64_t                      batch_stride_e,
                                              int64_t                      batch_count,
                                              bool                         strided_batch,
                                              bool                         grouped_gemm,
                                              bool                         gradient,
                                              const rocblaslt_matmul_algo* algo,
                                              void*                        workspace,
                                              size_t                       workspaceSizeInBytes,
                                              const void*                  bias,
                                              const void*                  scaleDVec,
                                              hipblasDatatype_t            bias_type,
                                              rocblaslt_epilogue           epilogue,
                                              std::shared_ptr<void>        gemmData,
                                              hipStream_t                  stream)
{
    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
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
                                      (Tc*)e,
                                      ld_e,
                                      batch_stride_e,
                                      batch_count,
                                      strided_batch,
                                      grouped_gemm,
                                      gradient,
                                      algo,
                                      workspace,
                                      workspaceSizeInBytes,
                                      reinterpret_cast<const void*>(bias),
                                      reinterpret_cast<const Tc*>(scaleDVec),
                                      bias_type,
                                      epilogue,
                                      gemmData,
                                      stream);
}

template <typename Ti, typename To = Ti, typename Tc = To>
rocblaslt_status rocblaslt_gemm_create_typecasting(hipblasOperation_t     trans_a,
                                                   hipblasOperation_t     trans_b,
                                                   int64_t                m,
                                                   int64_t                n,
                                                   int64_t                k,
                                                   const void*            alpha,
                                                   const void*            a,
                                                   int64_t                ld_a,
                                                   int64_t                batch_stride_a,
                                                   const void*            b,
                                                   int64_t                ld_b,
                                                   int64_t                batch_stride_b,
                                                   const void*            beta,
                                                   const void*            c,
                                                   int64_t                ld_c,
                                                   int64_t                batch_stride_c,
                                                   void*                  d,
                                                   int64_t                ld_d,
                                                   int64_t                batch_stride_d,
                                                   void*                  e,
                                                   int64_t                ld_e,
                                                   int64_t                batch_stride_e,
                                                   int64_t                batch_count,
                                                   bool                   strided_batch,
                                                   bool                   grouped_gemm,
                                                   bool                   gradient,
                                                   const void*            bias,
                                                   const void*            scaleDVec,
                                                   hipblasDatatype_t      bias_type,
                                                   rocblaslt_epilogue     epilogue,
                                                   std::shared_ptr<void>& gemmData,
                                                   size_t&                gemmCount)
{
    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
       || !isAligned(d, sizeof(To)))
    {
        std::cerr << "memmory is not aligned" << std::endl;
        return rocblaslt_status_invalid_size;
    }
    return rocblaslt_gemm_create_batched_template(trans_a,
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
                                                  (Tc*)e,
                                                  ld_e,
                                                  batch_stride_e,
                                                  batch_count,
                                                  strided_batch,
                                                  grouped_gemm,
                                                  gradient,
                                                  reinterpret_cast<const void*>(bias),
                                                  reinterpret_cast<const Tc*>(scaleDVec),
                                                  bias_type,
                                                  epilogue,
                                                  gemmData,
                                                  gemmCount);
}

template <typename Ti, typename To = Ti, typename Tc = To>
rocblaslt_status rocblaslt_groupedgemm_create_typecasting(hipblasOperation_t        trans_a,
                                                          hipblasOperation_t        trans_b,
                                                          std::vector<int64_t>&     m,
                                                          std::vector<int64_t>&     n,
                                                          std::vector<int64_t>&     k,
                                                          std::vector<const void*>& alpha,
                                                          std::vector<const void*>& a,
                                                          std::vector<int64_t>&     ld_a,
                                                          std::vector<int64_t>&     batch_stride_a,
                                                          std::vector<const void*>& b,
                                                          std::vector<int64_t>&     ld_b,
                                                          std::vector<int64_t>&     batch_stride_b,
                                                          std::vector<const void*>& beta,
                                                          std::vector<const void*>& c,
                                                          std::vector<int64_t>&     ld_c,
                                                          std::vector<int64_t>&     batch_stride_c,
                                                          std::vector<void*>&       d,
                                                          std::vector<int64_t>&     ld_d,
                                                          std::vector<int64_t>&     batch_stride_d,
                                                          std::vector<int64_t>&     batch_count,
                                                          bool                      strided_batch,
                                                          bool                      grouped_gemm,
                                                          std::vector<const void*>& bias,
                                                          std::vector<const void*>& scaleDVec,
                                                          std::vector<hipblasDatatype_t>& bias_type,
                                                          std::vector<rocblaslt_epilogue>& epilogue,
                                                          std::shared_ptr<void>&           gemmData,
                                                          size_t& gemmCount)
{
    std::vector<const Tc*> groupedAlpha, groupedBeta;
    std::vector<const Ti*> groupedA, groupedB;
    std::vector<const To*> groupedC;
    std::vector<To*>       groupedD;
    std::vector<const Tc*> groupedScaleDVec;

    for(int i = 0; i < alpha.size(); i++)
    {
        groupedAlpha.push_back(reinterpret_cast<const Tc*>(alpha[i]));
        groupedBeta.push_back(reinterpret_cast<const Tc*>(beta[i]));
        groupedA.push_back(reinterpret_cast<const Ti*>(a[i]));
        groupedB.push_back(reinterpret_cast<const Ti*>(b[i]));
        groupedC.push_back(reinterpret_cast<const To*>(c[i]));
        groupedD.push_back(reinterpret_cast<To*>(d[i]));
        groupedScaleDVec.push_back(reinterpret_cast<const Tc*>(scaleDVec[i]));
    }

    return rocblaslt_groupedgemm_create_batched_template(trans_a,
                                                         trans_b,
                                                         m,
                                                         n,
                                                         k,
                                                         groupedAlpha,
                                                         groupedA,
                                                         ld_a,
                                                         batch_stride_a,
                                                         groupedB,
                                                         ld_b,
                                                         batch_stride_b,
                                                         groupedBeta,
                                                         groupedC,
                                                         ld_c,
                                                         batch_stride_c,
                                                         groupedD,
                                                         ld_d,
                                                         batch_stride_d,
                                                         batch_count,
                                                         strided_batch,
                                                         grouped_gemm,
                                                         bias,
                                                         groupedScaleDVec,
                                                         bias_type,
                                                         epilogue,
                                                         gemmData,
                                                         gemmCount);
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
                                                  void*                        e,
                                                  int64_t                      ld_e,
                                                  int64_t                      batch_stride_e,
                                                  int64_t                      batch_count,
                                                  bool                         strided_batch,
                                                  bool                         grouped_gemm,
                                                  bool                         gradient,
                                                  rocblaslt_compute_type       compute_type,
                                                  const rocblaslt_matmul_algo* algo,
                                                  void*                        workspace,
                                                  size_t                       workspaceSizeInBytes,
                                                  const void*                  bias,
                                                  const void*                  scaleDVec,
                                                  hipblasDatatype_t            bias_type,
                                                  rocblaslt_epilogue           epilogue,
                                                  std::shared_ptr<void>        gemmData,
                                                  hipStream_t                  stream)
{
    rocblaslt_status rs_status = rocblaslt_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                        \
    handle, trans_a, trans_b, m, n, k, alpha, a, ld_a, batch_stride_a, b, ld_b, batch_stride_b,    \
        beta, c, ld_c, batch_stride_c, d, ld_d, batch_stride_d, e, ld_e, batch_stride_e,           \
        batch_count, strided_batch, grouped_gemm, gradient, algo, workspace, workspaceSizeInBytes, \
        bias, scaleDVec, bias_type, epilogue, gemmData, stream

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
        else if(c_type == HIPBLAS_R_32F && d_type == HIPBLAS_R_32F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_matmul_typecasting<rocblaslt_half, float, float>(
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

inline rocblaslt_status rocblaslt_gemm_create_template_cpp(hipblasOperation_t     trans_a,
                                                           hipblasOperation_t     trans_b,
                                                           int64_t                m,
                                                           int64_t                n,
                                                           int64_t                k,
                                                           const void*            alpha,
                                                           const void*            a,
                                                           hipblasDatatype_t      a_type,
                                                           int64_t                ld_a,
                                                           int64_t                batch_stride_a,
                                                           const void*            b,
                                                           hipblasDatatype_t      b_type,
                                                           int64_t                ld_b,
                                                           int64_t                batch_stride_b,
                                                           const void*            beta,
                                                           const void*            c,
                                                           hipblasDatatype_t      c_type,
                                                           int64_t                ld_c,
                                                           int64_t                batch_stride_c,
                                                           void*                  d,
                                                           hipblasDatatype_t      d_type,
                                                           int64_t                ld_d,
                                                           int64_t                batch_stride_d,
                                                           void*                  e,
                                                           int64_t                ld_e,
                                                           int64_t                batch_stride_e,
                                                           int64_t                batch_count,
                                                           bool                   strided_batch,
                                                           bool                   grouped_gemm,
                                                           bool                   gradient,
                                                           rocblaslt_compute_type compute_type,
                                                           const void*            bias,
                                                           const void*            scaleDVec,
                                                           hipblasDatatype_t      bias_type,
                                                           rocblaslt_epilogue     epilogue,
                                                           std::shared_ptr<void>& gemmData,
                                                           size_t&                gemmCount)
{
    rocblaslt_status rs_status = rocblaslt_status_not_implemented;

#define EX_TYPECASTING_PARM_GEMM_CPP                                                             \
    trans_a, trans_b, m, n, k, alpha, a, ld_a, batch_stride_a, b, ld_b, batch_stride_b, beta, c, \
        ld_c, batch_stride_c, d, ld_d, batch_stride_d, e, ld_e, batch_stride_e, batch_count,     \
        strided_batch, grouped_gemm, gradient, bias, scaleDVec, bias_type, epilogue, gemmData,   \
        gemmCount

    if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F)
    {
        if(c_type == HIPBLAS_R_32F && d_type == HIPBLAS_R_32F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_gemm_create_typecasting<float, float, float>(
                    EX_TYPECASTING_PARM_GEMM_CPP);
            }
        }
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F)
    {
        if(c_type == HIPBLAS_R_16F && d_type == HIPBLAS_R_16F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status
                    = rocblaslt_gemm_create_typecasting<rocblaslt_half, rocblaslt_half, float>(
                        EX_TYPECASTING_PARM_GEMM_CPP);
            }
        }
        else if(c_type == HIPBLAS_R_32F && d_type == HIPBLAS_R_32F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status
                    = rocblaslt_gemm_create_typecasting<rocblaslt_half, float, float>(
                        EX_TYPECASTING_PARM_GEMM_CPP);
            }
        }
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B)
    {
        if(c_type == HIPBLAS_R_16B && d_type == HIPBLAS_R_16B)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_gemm_create_typecasting<rocblaslt_bfloat16,
                                                              rocblaslt_bfloat16,
                                                              float>(EX_TYPECASTING_PARM_GEMM_CPP);
            }
        }
    }
    else
    {
        rs_status = rocblaslt_status_not_implemented;
    }

    return rs_status;
}

inline rocblaslt_status
    rocblaslt_groupedgemm_create_template_cpp(hipblasOperation_t               trans_a,
                                              hipblasOperation_t               trans_b,
                                              std::vector<int64_t>&            m,
                                              std::vector<int64_t>&            n,
                                              std::vector<int64_t>&            k,
                                              std::vector<const void*>&        alpha,
                                              std::vector<const void*>&        a,
                                              hipblasDatatype_t                a_type,
                                              std::vector<int64_t>&            ld_a,
                                              std::vector<int64_t>&            batch_stride_a,
                                              std::vector<const void*>&        b,
                                              hipblasDatatype_t                b_type,
                                              std::vector<int64_t>&            ld_b,
                                              std::vector<int64_t>&            batch_stride_b,
                                              std::vector<const void*>&        beta,
                                              std::vector<const void*>&        c,
                                              hipblasDatatype_t                c_type,
                                              std::vector<int64_t>&            ld_c,
                                              std::vector<int64_t>&            batch_stride_c,
                                              std::vector<void*>&              d,
                                              hipblasDatatype_t                d_type,
                                              std::vector<int64_t>&            ld_d,
                                              std::vector<int64_t>&            batch_stride_d,
                                              std::vector<int64_t>&            batch_count,
                                              bool                             strided_batch,
                                              bool                             grouped_gemm,
                                              rocblaslt_compute_type           compute_type,
                                              std::vector<const void*>&        bias,
                                              std::vector<const void*>&        scaleDVec,
                                              std::vector<hipblasDatatype_t>&  bias_type,
                                              std::vector<rocblaslt_epilogue>& epilogue,
                                              std::shared_ptr<void>&           gemmData,
                                              size_t&                          gemmCount)
{
    rocblaslt_status rs_status = rocblaslt_status_not_implemented;

#define EX_TYPECASTING_PARM_GroupedGemm_CPP                                                      \
    trans_a, trans_b, m, n, k, alpha, a, ld_a, batch_stride_a, b, ld_b, batch_stride_b, beta, c, \
        ld_c, batch_stride_c, d, ld_d, batch_stride_d, batch_count, strided_batch, grouped_gemm, \
        bias, scaleDVec, bias_type, epilogue, gemmData, gemmCount

    if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F)
    {
        if(c_type == HIPBLAS_R_32F && d_type == HIPBLAS_R_32F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_groupedgemm_create_typecasting<float, float, float>(
                    EX_TYPECASTING_PARM_GroupedGemm_CPP);
            }
        }
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F)
    {
        if(c_type == HIPBLAS_R_16F && d_type == HIPBLAS_R_16F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_groupedgemm_create_typecasting<rocblaslt_half,
                                                                     rocblaslt_half,
                                                                     float>(
                    EX_TYPECASTING_PARM_GroupedGemm_CPP);
            }
        }
        else if(c_type == HIPBLAS_R_32F && d_type == HIPBLAS_R_32F)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_groupedgemm_create_typecasting<rocblaslt_half,
                                                                     float,
                                                                     float>(
                    EX_TYPECASTING_PARM_GroupedGemm_CPP);
            }
        }
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B)
    {
        if(c_type == HIPBLAS_R_16B && d_type == HIPBLAS_R_16B)
        {
            if(compute_type == rocblaslt_compute_f32)
            {
                rs_status = rocblaslt_groupedgemm_create_typecasting<rocblaslt_bfloat16,
                                                                     rocblaslt_bfloat16,
                                                                     float>(
                    EX_TYPECASTING_PARM_GroupedGemm_CPP);
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
