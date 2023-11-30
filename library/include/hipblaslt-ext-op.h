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
 *  \brief hipblaslt-ext-op.h provides extension operations with
 *  C-style API.
 */

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled
//! unmodified through either AMD HCC or NVCC. Key features tend to be in the
//! spirit and terminology of CUDA, but with a portable path to other
//! accelerators as well.
//

#pragma once

#include <hipblaslt/hipblaslt.h>

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup library_module
 *  \brief Perform softmax on given tensor.
 *
 *  \details
 *  This function computes softmax on given 2D-tensor along specified dimension.
 *
 *  @param[in]
 *  datatype Datatype of input/output tensor, currently support HIP_R_32F only.
 *
 *  @param[in]
 *  m The first dimension of input/output tensor.
 *
 *  @param[in]
 *  n The second dimension of input/output tensor. Currently only values less than or equal to 256 are supported.
 *
 *  @param[in]
 *  dim Specified dimension to perform softmax on. Currently 1 is the only valid value.
 *
 *  @param[in]
 *  input input tensor buffer.
 *
 *  @param[in]
 *  stream The HIP stream where all the GPU work will be submitted.
 *
 *  @param[out]
 *  output output tensor buffer.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If it runs successfully.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE If \p n is greater than 256.
 *  \retval HIPBLAS_STATUS_NOT_SUPPORTED If \p dim is not 1 or \p datatype is not HIP_R_32F.
 */
HIPBLASLT_EXPORT hipblasStatus_t hipblasltExtSoftmax(hipDataType datatype,
                                                     uint32_t    m,
                                                     uint32_t    n,
                                                     uint32_t    dim,
                                                     void*       output,
                                                     void*       input,
                                                     hipStream_t stream);

/*! \ingroup library_module
 *  \brief Perform layernorm on given tensor.
 *
 *  \details
 *  This function computes layernorm on given 2D-tensor.
 *
 *  @param[in]
 *  datatype Datatype of input/output tensor, currently support HIP_R_32F only.
 *
 *  @param[out]
 *  output output tensor buffer. can't be nullptr.
 *
 *  @param[out]
 *  mean tensor buffer. can't be nullptr.
 *
 *  @param[out]
 *  invvar tensor buffer. 1 / sqrt(std).  can't be nullptr.
 *
 *  @param[in]
 *  input tensor buffer. can't be nullptr.
 *
 *  @param[in]
 *  m The first dimension of input/output tensor.
 *
 *  @param[in]
 *  n The second dimension of input/output tensor.
 *
 *  @param[in]
 *  eps for sqrt to avoid inf value.
 *
 *  @param[in]
 *  gamma tensor buffer. nullptr means calculation doesn't involve gamma.
 *
 *  @param[in]
 *  beta tensor buffer. nullptr means calculation doesn't involve beta.
 *
 *  @param[in]
 *  stream The HIP stream where all the GPU work will be submitted.
 *
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If it runs successfully.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE If \p m is greater than 4096.
 *  \retval HIPBLAS_STATUS_NOT_SUPPORTED if \p datatype is not HIP_R_32F.
 */
HIPBLASLT_EXPORT hipblasStatus_t hipblasltExtLayerNorm(hipDataType datatype,
                                                       void*       output,
                                                       void*       mean,
                                                       void*       invvar,
                                                       void*       input,
                                                       uint32_t    m,
                                                       uint32_t    n,
                                                       float       eps,
                                                       void*       gamma,
                                                       void*       beta,
                                                       hipStream_t stream);
#ifdef __cplusplus
}
#endif
