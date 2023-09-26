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

#pragma once

#include "cblas.h"
#include <hipblaslt/hipblaslt.h>
#include <type_traits>

/*!\file
 * \brief provide template functions interfaces to CBLAS C89 interfaces, it is only used for testing
 * not part of the GPU library
 */

// gemm
template <typename TiA, typename TiB, typename To, typename Tc, typename Tci = TiA>
void cblas_gemm(hipblasOperation_t     transA,
                hipblasOperation_t     transB,
                int64_t                m,
                int64_t                n,
                int64_t                k,
                Tc                     alpha,
                const TiA*             A,
                int64_t                lda,
                const TiB*             B,
                int64_t                ldb,
                Tc                     beta,
                std::add_pointer_t<To> C,
                int64_t                ldc,
                const Tc*              AlphaVec,
                Tc                     scaleA,
                Tc                     scaleB,
                Tc                     scaleD,
                bool                   alt = false);
