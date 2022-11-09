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

#include "hipblaslt.h"
#include "exceptions.hpp"

#include <hip/hip_runtime_api.h>
#include <rocblaslt.h>
#include <stdio.h>
#include <stdlib.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIPBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)                      \
  {                                                                            \
    hipblasStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;             \
    if (TMP_STATUS_FOR_CHECK != HIPBLAS_STATUS_SUCCESS) {                      \
      return TMP_STATUS_FOR_CHECK;                                             \
    }                                                                          \
  }

#define RETURN_IF_ROCSPARSELT_ERROR(INPUT_STATUS_FOR_CHECK)                    \
  {                                                                            \
    rocblaslt_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;            \
    if (TMP_STATUS_FOR_CHECK != rocblaslt_status_success) {                    \
      return RocBlasLtStatusToHIPStatus(TMP_STATUS_FOR_CHECK);                 \
    }                                                                          \
  }

hipblasStatus_t hipErrorToHIPBLASStatus(hipError_t status) {
  switch (status) {
  case hipSuccess:
    return HIPBLAS_STATUS_SUCCESS;
  case hipErrorMemoryAllocation:
  case hipErrorLaunchOutOfResources:
    return HIPBLAS_STATUS_ALLOC_FAILED;
  case hipErrorInvalidDevicePointer:
    return HIPBLAS_STATUS_INVALID_VALUE;
  case hipErrorInvalidDevice:
  case hipErrorInvalidResourceHandle:
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  case hipErrorInvalidValue:
    return HIPBLAS_STATUS_INVALID_VALUE;
  case hipErrorNoDevice:
  case hipErrorUnknown:
    return HIPBLAS_STATUS_INTERNAL_ERROR;
  default:
    return HIPBLAS_STATUS_INTERNAL_ERROR;
  }
}

hipblasStatus_t RocBlasLtStatusToHIPStatus(rocblaslt_status_ status) {
  switch (status) {
  case rocblaslt_status_success:
    return HIPBLAS_STATUS_SUCCESS;
  case rocblaslt_status_invalid_handle:
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  case rocblaslt_status_not_implemented:
    return HIPBLAS_STATUS_INTERNAL_ERROR;
  case rocblaslt_status_invalid_pointer:
    return HIPBLAS_STATUS_INVALID_VALUE;
  case rocblaslt_status_invalid_size:
    return HIPBLAS_STATUS_INVALID_VALUE;
  case rocblaslt_status_memory_error:
    return HIPBLAS_STATUS_ALLOC_FAILED;
  case rocblaslt_status_internal_error:
    return HIPBLAS_STATUS_INTERNAL_ERROR;
  case rocblaslt_status_invalid_value:
    return HIPBLAS_STATUS_INVALID_VALUE;
  case rocblaslt_status_arch_mismatch:
    return HIPBLAS_STATUS_ARCH_MISMATCH;
  default:
    throw HIPBLAS_STATUS_INVALID_ENUM;
  }
}

hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle) try {
  // Check if handle is valid
  if (handle == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  int deviceId;
  hipError_t err;
  hipblasStatus_t retval = HIPBLAS_STATUS_SUCCESS;

  err = hipGetDevice(&deviceId);
  if (err == hipSuccess) {
    retval = RocBlasLtStatusToHIPStatus(
        rocblaslt_create((rocblaslt_handle *)handle));
  }
  return retval;
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle) try {
  return RocBlasLtStatusToHIPStatus(
      rocblaslt_destroy((const rocblaslt_handle)handle));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t *matDescr,
                                            hipblasDatatype_t valueType,
                                            uint64_t rows, uint64_t cols,
                                            int64_t ld) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matrix_layout_create(
      (rocblaslt_matrix_layout *)matDescr, valueType, rows, cols, ld));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t
hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t descr) try {
  return RocBlasLtStatusToHIPStatus(
      rocblaslt_matrix_layout_destory((const rocblaslt_matrix_layout)descr));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t *matmulDesc,
                                          hipblasLtComputeType_t computeType,
                                          hipblasDatatype_t scaleType) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_desc_create(
      (rocblaslt_matmul_desc *)matmulDesc, (rocblaslt_compute_type)computeType,
      scaleType));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t
hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t descr) try {
  return RocBlasLtStatusToHIPStatus(
      rocblaslt_matmul_desc_destroy((const rocblaslt_matmul_desc)descr));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t
hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc,
                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                const void *buf, size_t sizeInBytes) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_desc_set_attribute(
      (rocblaslt_matmul_desc)matmulDesc,
      (rocblaslt_matmul_desc_attributes)matmulAttr, buf, sizeInBytes));
} catch (...) {
  return exception_to_hipblas_status();
}
hipblasStatus_t
hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t matmulDesc,
                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                void *buf, size_t sizeInBytes,
                                size_t *sizeWritten) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_desc_get_attribute(
      (rocblaslt_matmul_desc)matmulDesc,
      (rocblaslt_matmul_desc_attributes)matmulAttr, buf, sizeInBytes,
      sizeWritten));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t
hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t *pref) try {
  return RocBlasLtStatusToHIPStatus(
      rocblaslt_matmul_preference_create((rocblaslt_matmul_preference *)pref));
} catch (...) {
  return exception_to_hipblas_status();
}
hipblasStatus_t
hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_preference_destroy(
      (const rocblaslt_matmul_preference)pref));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attribute, const void *data,
    size_t dataSize) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_preference_set_attribute(
      (rocblaslt_matmul_preference)pref,
      (rocblaslt_matmul_preference_attributes)attribute, data, dataSize));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulPreferenceGetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attribute, void *data,
    size_t sizeInBytes, size_t *sizeWritten) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_preference_get_attribute(
      (rocblaslt_matmul_preference)pref,
      (rocblaslt_matmul_preference_attributes)attribute, data, sizeInBytes,
      sizeWritten));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc,
    hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc,
    hipblasLtMatmulPreference_t pref, int requestedAlgoCount,
    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_algo_get_heuristic(
      (rocblaslt_handle)handle, (rocblaslt_matmul_desc)matmulDesc,
      (rocblaslt_matrix_layout)Adesc, (rocblaslt_matrix_layout)Bdesc,
      (rocblaslt_matrix_layout)Cdesc, (rocblaslt_matrix_layout)Ddesc,
      (rocblaslt_matmul_preference)pref, requestedAlgoCount,
      (rocblaslt_matmul_heuristic_result *)heuristicResultsArray,
      returnAlgoCount));
} catch (...) {
  return exception_to_hipblas_status();
}

hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmul_descr,
                const void *alpha, const void *A, hipblasLtMatrixLayout_t matA,
                const void *B, hipblasLtMatrixLayout_t matB, const void *beta,
                const void *C, hipblasLtMatrixLayout_t matC, void *D,
                hipblasLtMatrixLayout_t matD, const hipblasLtMatmulAlgo_t *algo,
                void *workspace, size_t workspaceSizeInBytes,
                hipStream_t stream) try {
  return RocBlasLtStatusToHIPStatus(rocblaslt_matmul(
      (rocblaslt_handle)handle, (rocblaslt_matmul_desc)matmul_descr, alpha, A,
      (rocblaslt_matrix_layout)matA, B, (rocblaslt_matrix_layout)matB, beta, C,
      (rocblaslt_matrix_layout)matC, D, (rocblaslt_matrix_layout)matD,
      (const rocblaslt_matmul_algo *)algo, workspace, workspaceSizeInBytes,
      stream));
} catch (...) {
  return exception_to_hipblas_status();
}

// Other Utilities
hipblasStatus_t hipblasLtGetVersion(hipblasLtHandle_t handle,
                                    int *version) try {
  if (handle == nullptr) {
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  }

  *version = HIPBLASLT_VERSION_MAJOR * 100000 + HIPBLASLT_VERSION_MINOR * 100 +
             HIPBLASLT_VERSION_PATCH;

  return HIPBLAS_STATUS_SUCCESS;
} catch (...) {
  return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtGetGitRevision(hipblasLtHandle_t handle,
                                        char *rev) try {
  // Get hipSPARSE revision
  if (handle == nullptr) {
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  }

  if (rev == nullptr) {
    return HIPBLAS_STATUS_INVALID_VALUE;
  }

  static constexpr char v[] = TO_STR(HIPBLASLT_VERSION_TWEAK);

  memcpy(rev, v, sizeof(v));

  return HIPBLAS_STATUS_SUCCESS;
} catch (...) {
  return exception_to_hipblas_status();
}
// TODO
// hipblasStatus_t hipblasLtGetArchName(char** archName)
// try
//{
//    *archName        = nullptr;
//    std::string arch = rocblaslt_internal_get_arch_name();
//    *archName        = (char*)malloc(arch.size() * sizeof(char));
//    strncpy(*archName, arch.c_str(), arch.size());
//    return HIPBLAS_STATUS_SUCCESS;
//}
// catch(...)
//{
//    if(archName != nullptr)
//    {
//        free(*archName);
//        *archName = nullptr;
//    }
//    return exception_to_hipblas_status();
//}

#ifdef __cplusplus
}
#endif
