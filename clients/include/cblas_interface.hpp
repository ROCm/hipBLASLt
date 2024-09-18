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

#pragma once

#include "cblas.h"
#include "datatype_interface.hpp"
#include "hipblaslt_ostream.hpp"
#include <hipblaslt/hipblaslt.h>
#include <type_traits>

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
 */

// gemm
template <typename Tc>
void cblas_gemm(hipblasOperation_t       transA,
                hipblasOperation_t       transB,
                int64_t                  m,
                int64_t                  n,
                int64_t                  k,
                Tc                       alpha,
                const void*              A,
                int64_t                  lda,
                const void*              B,
                int64_t                  ldb,
                Tc                       beta,
                std::add_pointer_t<void> C,
                int64_t                  ldc,
                const Tc*                AlphaVec,
                const Tc*                scaleA,
                const Tc*                scaleB,
                Tc                       scaleD,
                bool                     isScaleAVec,
                bool                     isScaleBVec,
                hipDataType              tiA,
                hipDataType              tiB,
                hipDataType              to,
                hipDataType              tc,
                hipDataType              tciA,
                hipDataType              tciB,
                bool                     alt = false);

inline void cblas_gemm(hipblasOperation_t       transA,
                       hipblasOperation_t       transB,
                       int64_t                  m,
                       int64_t                  n,
                       int64_t                  k,
                       computeTypeInterface     alpha,
                       const void*              A,
                       int64_t                  lda,
                       const void*              B,
                       int64_t                  ldb,
                       computeTypeInterface     beta,
                       std::add_pointer_t<void> C,
                       int64_t                  ldc,
                       const void*              AlphaVec,
                       const void*              scaleA,
                       const void*              scaleB,
                       void*                    scaleD,
                       bool                     isScaleAVec,
                       bool                     isScaleBVec,
                       hipDataType              tiA,
                       hipDataType              tiB,
                       hipDataType              to,
                       hipDataType              tc,
                       hipDataType              tciA,
                       hipDataType              tciB,
                       bool                     alt = false)
{
    switch(tc)
    {
    case HIP_R_32F:
        cblas_gemm<float>(transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha.f32,
                          A,
                          lda,
                          B,
                          ldb,
                          beta.f32,
                          C,
                          ldc,
                          (const float*)AlphaVec,
                          (const float*)scaleA,
                          (const float*)scaleB,
                          *(float*)scaleD,
                          isScaleAVec,
                          isScaleBVec,
                          tiA,
                          tiB,
                          to,
                          tc,
                          tciA,
                          tciB,
                          alt);
        return;
    case HIP_R_64F:
        cblas_gemm<double>(transA,
                           transB,
                           m,
                           n,
                           k,
                           alpha.f64,
                           A,
                           lda,
                           B,
                           ldb,
                           beta.f64,
                           C,
                           ldc,
                           (const double*)AlphaVec,
                           (const double*)scaleA,
                           (const double*)scaleB,
                           *(double*)scaleD,
                           isScaleAVec,
                           isScaleBVec,
                           tiA,
                           tiB,
                           to,
                           tc,
                           tciA,
                           tciB,
                           alt);
        return;
    case HIP_R_32I:
        cblas_gemm<int32_t>(transA,
                            transB,
                            m,
                            n,
                            k,
                            alpha.i32,
                            A,
                            lda,
                            B,
                            ldb,
                            beta.i32,
                            C,
                            ldc,
                            (const int32_t*)AlphaVec,
                            (const int32_t*)scaleA,
                            (const int32_t*)scaleB,
                            *(int32_t*)scaleD,
                            isScaleAVec,
                            isScaleBVec,
                            tiA,
                            tiB,
                            to,
                            tc,
                            tciA,
                            tciB,
                            alt);
        return;
    default:
        hipblaslt_cerr << "Error type in cblas_gemm()" << std::endl;
        return;
    }
}
