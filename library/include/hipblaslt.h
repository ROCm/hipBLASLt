/*******************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled
//! unmodified through either AMD HCC or NVCC. Key features tend to be in the
//! spirit and terminology of CUDA, but with a portable path to other
//! accelerators as well.
//!
//! This is the master include file for hipBLASLt, wrapping around rocBLASLt and
//! cuBLASLt.
//

#pragma once
#ifndef _HIPBLASLT_H_
#define _HIPBLASLT_H_

#include "hipblaslt-export.h"
#include "hipblaslt-version.h"
#include <hipblas.h>

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#if defined(__HIP_PLATFORM_HCC__)
#include "hipblaslt-types.h"
#endif

typedef enum {
  HIPBLASLT_EPILOGUE_DEFAULT = 1,
  HIPBLASLT_EPILOGUE_RELU = 2,
  HIPBLASLT_EPILOGUE_BIAS = 4,
  HIPBLASLT_EPILOGUE_RELU_BIAS = 6,
  HIPBLASLT_EPILOGUE_GELU = 32,
  HIPBLASLT_EPILOGUE_GELU_BIAS = 36
} hipblasLtEpilogue_t;

typedef enum {
  HIPBLASLT_COMPUTE_F32 = 300,
} hipblasLtComputeType_t;

typedef enum {
  HIPBLASLT_MATRIX_LAYOUT_TYPE = 0,
  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 1,
  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 2
} hipblasLtMatrixLayoutAttribute_t;

typedef enum {
  HIPBLASLT_MATMUL_DESC_TRANSA = 0,
  HIPBLASLT_MATMUL_DESC_TRANSB = 1,
  HIPBLASLT_MATMUL_DESC_EPILOGUE = 2,
  HIPBLASLT_MATMUL_DESC_BIAS_POINTER = 3,
  HIPBLASLT_MATMUL_DESC_MAX = 4
} hipblasLtMatmulDescAttributes_t;

typedef enum {
  HIPBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
  HIPBLASLT_MATMUL_PREF_MAX = 2
} hipblasLtMatmulPreferenceAttributes_t;

/* Opaque structures holding information */
// clang-format off

#if defined(__HIP_PLATFORM_HCC__)
typedef struct {
  uint64_t data[4];
} hipblasLtMatmulDescOpaque_t;
typedef struct {
  uint64_t data[4];
} hipblasLtMatrixLayoutOpaque_t;
typedef struct {
  uint64_t data[5];
} hipblasLtMatmulPreferenceOpaque_t;
typedef void* hipblasLtHandle_t;
typedef hipblasLtMatmulDescOpaque_t* hipblasLtMatmulDesc_t;
typedef hipblasLtMatrixLayoutOpaque_t* hipblasLtMatrixLayout_t;
typedef hipblasLtMatmulPreferenceOpaque_t* hipblasLtMatmulPreference_t;
typedef struct _hipblasLtMatmulAlgo_t{
  uint32_t solutionIdx = 0;
  size_t max_workspace_bytes = 0;
} hipblasLtMatmulAlgo_t;
typedef struct _hipblasLtMatmulHeuristicResult_t{
  hipblasLtMatmulAlgo_t algo;
  size_t workspaceSize = 0;
  hipblasStatus_t state = HIPBLAS_STATUS_SUCCESS;
  float wavesCount = 1.0;
  int reserved[4];
} hipblasLtMatmulHeuristicResult_t;
#elif defined(__HIP_PLATFORM_NVCC__)
#endif
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle);

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t *matDescr,
                                            hipblasDatatype_t valueType,
                                            uint64_t rows, uint64_t cols,
                                            int64_t ld);

HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatrixLayoutDestory(const hipblasLtMatrixLayout_t descr);

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t *matmulDesc,
                                          hipblasLtComputeType_t computeType,
                                          hipblasDatatype_t scaleType);

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t descr);

HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc,
                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                const void *buf, size_t sizeInBytes);

HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t matmulDesc,
                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                void *buf, size_t sizeInBytes,
                                size_t *sizeWritten);

HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t *pref);

HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref);

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attribute, const void *data,
    size_t dataSize);

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulPreferenceGetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attribute, void *data,
    size_t dataSize);

HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc,
    hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc,
    hipblasLtMatmulPreference_t pref, int requestedAlgoCount,
    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount);

HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmul_descr,
                const void *alpha, const void *A, hipblasLtMatrixLayout_t matA,
                const void *B, hipblasLtMatrixLayout_t matB, const void *beta,
                const void *C, hipblasLtMatrixLayout_t matC, void *D,
                hipblasLtMatrixLayout_t matD, const hipblasLtMatmulAlgo_t *algo,
                void *workspace, size_t workspaceSizeInBytes,
                hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // _HIPBLASLT_H_
