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

/*! \file
 *  \brief hipblaslt.h provides general matrix-matrix operations with
 *  flexible API to let user set attributes for solution selection.
 */

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

#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#if defined(__HIP_PLATFORM_HCC__)
#include "hipblaslt-types.h"
#endif

/* Opaque structures holding information */
// clang-format off

/*! \ingroup types_module
 *  \brief Specify the enum type to set the postprocessing options for the epilogue.
 */
typedef enum {
  HIPBLASLT_EPILOGUE_DEFAULT = 1,    /**<No special postprocessing, just scale and quantize the results if necessary.*/
  HIPBLASLT_EPILOGUE_RELU = 2,       /**<Apply ReLU point-wise transform to the results:(x:=max(x, 0))*/
  HIPBLASLT_EPILOGUE_BIAS = 4,       /**<Apply (broadcast) bias from the bias vector. Bias vector length must match matrix D rows, and it must be packed (such as stride between vector elements is 1). Bias vector is broadcast to all columns and added before applying the final postprocessing.*/
  HIPBLASLT_EPILOGUE_RELU_BIAS = 6,  /**<Apply bias and then ReLU transform.*/
  HIPBLASLT_EPILOGUE_GELU = 32,      /**<Apply GELU point-wise transform to the results (x:=GELU(x)).*/
  HIPBLASLT_EPILOGUE_GELU_BIAS = 36  /**<Apply Bias and then GELU transform.*/
} hipblasLtEpilogue_t;

/*! \ingroup types_module
 *  \brief Specify the compute precision modes of the matrix
 */
typedef enum {
  HIPBLASLT_COMPUTE_F32 = 300,     /**<32-bit floating-point precision.*/
} hipblasLtComputeType_t;

/*! \ingroup types_module
 *  \brief Specify the attributes that define the details of the matrix operation.
 */
typedef enum {
  HIPBLASLT_MATRIX_LAYOUT_TYPE = 0,                /**<Specifies the data precision type. See \p hipblasDataType_t. Data Type: uint32_t*/
  HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 1,         /**<Number of matmul operations to perform in the batch. Default value is 1. Data Type: int32_t*/
  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 2 /**<Stride (in elements) to the next matrix for the strided batch operation. Default value is 0. Data Type: int64_t*/
} hipblasLtMatrixLayoutAttribute_t;

/*! \ingroup types_module
 *  \brief Specify the attributes that define the specifics of the matrix multiply operation.
 */
typedef enum {
  HIPBLASLT_MATMUL_DESC_TRANSA = 0,               /**<Specifies the type of transformation operation that should be performed on matrix A. Default value is HIPBLAS_OP_N (for example, non-transpose operation). See hipblasOperation_t. Data Type:int32_t*/
  HIPBLASLT_MATMUL_DESC_TRANSB = 1,               /**<Specifies the type of transformation operation that should be performed on matrix B. Default value is HIPBLAS_OP_N (for example, non-transpose operation). See hipblasOperation_t. Data Type:int32_t*/
  HIPBLASLT_MATMUL_DESC_EPILOGUE = 2,             /**<Epilogue function. See hipblasLtEpilogue_t. Default value is: HIPBLASLT_EPILOGUE_DEFAULT. Data Type: uint32_t*/
  HIPBLASLT_MATMUL_DESC_BIAS_POINTER = 3,         /**<Bias or Bias gradient vector pointer in the device memory. Data Type:void *\/ const void**/
  HIPBLASLT_MATMUL_DESC_MAX = 4
} hipblasLtMatmulDescAttributes_t;

/*! \ingroup types_module
 *  \brief It is an enumerated type used to apply algorithm search preferences while fine-tuning the heuristic function.
 */
typedef enum {
  HIPBLASLT_MATMUL_PREF_SEARCH_MODE = 0,          /**<Search mode. Data Type: uint32_t*/
  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,  /**<Maximum allowed workspace memory. Default is 0 (no workspace memory allowed). Data Type: uint64_t*/
  HIPBLASLT_MATMUL_PREF_MAX = 2
} hipblasLtMatmulPreferenceAttributes_t;

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
/*! \ingroup types_module
 *  \brief Handle to the hipBLASLt library context queue
 *
 *  \details
 *  The hipblasLtHandle_t type is a pointer type to an opaque structure holding the hipBLASLt library context. Use the following functions to manipulate this library context:
 *
 *  \ref hipblasLtCreate():
 *  To initialize the hipBLASLt library context and return a handle to an opaque structure holding the hipBLASLt library context.
 *  \ref hipblasLtDestroy():
 *  To destroy a previously created hipBLASLt library context descriptor and release the resources.
 */
typedef void* hipblasLtHandle_t;
/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication operation
 *
 *  \details
 *  This is a pointer to an opaque structure holding the description of the matrix multiplication operation \ref hipblasLtMatmul().
 *  Use the following functions to manipulate this descriptor:
 *  \ref hipblasLtMatmulDescCreate(): To create one instance of the descriptor.
 *  \ref hipblasLtMatmulDescDestroy(): To destroy a previously created descriptor and release the resources.
 */
typedef hipblasLtMatmulDescOpaque_t* hipblasLtMatmulDesc_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix layout
 *
 *  \details
 *  This is a pointer to an opaque structure holding the description of a matrix layout.
 *  Use the following functions to manipulate this descriptor:
 *  \ref hipblasLtMatrixLayoutCreate(): To create one instance of the descriptor.
 *  \ref hipblasLtMatrixLayoutDestroy(): To destroy a previously created descriptor and release the resources.
 */
typedef hipblasLtMatrixLayoutOpaque_t* hipblasLtMatrixLayout_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication preference
 *
 *  \details
 *  This is a pointer to an opaque structure holding the description of the preferences for \ref hipblasLtMatmulAlgoGetHeuristic() configuration.
 *  Use the following functions to manipulate this descriptor:
 *  \ref hipblasLtMatmulPreferenceCreate(): To create one instance of the descriptor.
 *  \ref hipblasLtMatmulPreferenceDestroy(): To destroy a previously created descriptor and release the resources.
 */
typedef hipblasLtMatmulPreferenceOpaque_t* hipblasLtMatmulPreference_t;

/*! \ingroup types_module
 *  \brief Description of the matrix multiplication algorithm
 *
 *  \details
 *  This is an opaque structure holding the description of the matrix multiplication algorithm.
 *  This structure can be trivially serialized and later restored for use with the same version of hipBLASLt library to save on selecting the right configuration again.
 */
typedef struct _hipblasLtMatmulAlgo_t{
  uint32_t solutionIdx = 0;
  size_t max_workspace_bytes = 0;
} hipblasLtMatmulAlgo_t;

/*! \ingroup types_module
 *  \brief Description of the matrix multiplication algorithm
 *
 *  \details
 *  This is a descriptor that holds the configured matrix multiplication algorithm descriptor and its runtime properties.
 *  This structure can be trivially serialized and later restored for use with the same version of hipBLASLt library to save on selecting the right configuration again.
 */
typedef struct _hipblasLtMatmulHeuristicResult_t{
  hipblasLtMatmulAlgo_t algo;                      /**<Algo struct*/
  size_t workspaceSize = 0;                        /**<Actual size of workspace memory required.*/
  hipblasStatus_t state = HIPBLAS_STATUS_SUCCESS;  /**<Result status. Other fields are valid only if, after call to hipblasLtMatmulAlgoGetHeuristic(), this member is set to HIPBLAS_STATUS_SUCCESS..*/
  float wavesCount = 1.0;                          /**<Waves count is a device utilization metric. A wavesCount value of 1.0f suggests that when the kernel is launched it will fully occupy the GPU.*/
  int reserved[4];                                 /**<Reserved.*/
} hipblasLtMatmulHeuristicResult_t;
#elif defined(__HIP_PLATFORM_NVCC__)
#endif
// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup library_module
 *  \brief Create a hipblaslt handle
 *
 *  \details
 *  This function initializes the hipBLASLt library and creates a handle to an
 * opaque structure holding the hipBLASLt library context. It allocates light
 * hardware resources on the host and device, and must be called prior to making
 * any other hipBLASLt library calls. The hipBLASLt library context is tied to
 * the current CUDA device. To use the library on multiple devices, one
 * hipBLASLt handle should be created for each device.
 *
 *  @param[out]
 *  handle  Pointer to the allocated hipBLASLt handle for the created hipBLASLt
 * context.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS The allocation completed successfully.
 *  \retval HIPBLAS_STATUS_INVALID_VALUE \p handle == NULL.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t *handle);

/*! \ingroup library_module
 *  \brief Destory a hipblaslt handle
 *
 *  \details
 *  This function releases hardware resources used by the hipBLASLt library.
 *  This function is usually the last call with a particular handle to the
 * hipBLASLt library. Because hipblasLtCreate() allocates some internal
 * resources and the release of those resources by calling hipblasLtDestroy()
 * will implicitly call hipDeviceSynchronize(), it is recommended to minimize
 * the number of hipblasLtCreate()/hipblasLtDestroy() occurrences.
 *
 *  @param[in]
 *  handle  Pointer to the hipBLASLt handle to be destroyed.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS The hipBLASLt context was successfully
 * destroyed. \retval HIPBLAS_STATUS_NOT_INITIALIZED The hipBLASLt library was
 * not initialized. \retval HIPBLAS_STATUS_INVALID_VALUE \p handle == NULL.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);

/*! \ingroup library_module
 *  \brief Create a matrix layout descriptor
 *
 *  \details
 *  This function creates a matrix layout descriptor by allocating the memory
 * needed to hold its opaque structure.
 *
 *  @param[out]
 *  matLayout Pointer to the structure holding the matrix layout descriptor
 * created by this function. see \ref hipblasLtMatrixLayout_t .
 *  @param[in]
 *  type Enumerant that specifies the data precision for the matrix layout
 * descriptor this function creates. See hipblasDataType_t.
 *  @param[in]
 *  rows Number of rows of the matrix.
 *  @param[in]
 *  cols Number of columns of the matrix.
 *  @param[in]
 *  ld The leading dimension of the matrix. In column major layout, this is the
 * number of elements to jump to reach the next column. Thus ld >= m (number of
 * rows).
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If the descriptor was created successfully.
 *  \retval HIPBLAS_STATUS_ALLOC_FAILED If the memory could not be allocated.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t *matLayout,
                                            hipblasDatatype_t type,
                                            uint64_t rows, uint64_t cols,
                                            int64_t ld);

/*! \ingroup library_module
 *  \brief Destory a matrix layout descriptor
 *
 *  \details
 *  This function destroys a previously created matrix layout descriptor object.
 *
 *  @param[in]
 *  matLayout Pointer to the structure holding the matrix layout descriptor that
 * should be destroyed by this function. see \ref hipblasLtMatrixLayout_t .
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If the operation was successful.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t matLayout);

/*! \ingroup library_module
 *  \brief Create a matrix multiply descriptor
 *
 *  \details
 *  This function creates a matrix multiply descriptor by allocating the memory
 * needed to hold its opaque structure.
 *
 *  @param[out]
 *  matmulDesc  Pointer to the structure holding the matrix multiply descriptor
 * created by this function. See \ref hipblasLtMatmulDesc_t .
 *  @param[in]
 *  computeType  Enumerant that specifies the data precision for the matrix
 * multiply descriptor this function creates. See \ref hipblasLtComputeType_t .
 *  @param[in]
 *  scaleType  Enumerant that specifies the data precision for the matrix
 * transform descriptor this function creates. See hipblasDataType_t.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If the descriptor was created successfully.
 *  \retval HIPBLAS_STATUS_ALLOC_FAILED If the memory could not be allocated.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t *matmulDesc,
                                          hipblasLtComputeType_t computeType,
                                          hipblasDatatype_t scaleType);

/*! \ingroup library_module
 *  \brief Destory a matrix multiply descriptor
 *
 *  \details
 *  This function destroys a previously created matrix multiply descriptor
 * object.
 *
 *  @param[in]
 *  matmulDesc  Pointer to the structure holding the matrix multiply descriptor
 * that should be destroyed by this function. See \ref hipblasLtMatmulDesc_t .
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If operation was successful.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t matmulDesc);

/*! \ingroup library_module
 *  \brief  Set attribute to a matrix multiply descriptor
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 * previously created matrix multiply descriptor.
 *
 *  @param[in]
 *  matmulDesc  Pointer to the previously created structure holding the matrix
 * multiply descriptor queried by this function. See \ref hipblasLtMatmulDesc_t
 * .
 *  @param[in]
 *  attr  	The attribute that will be set by this function. See \ref
 * hipblasLtMatmulDescAttributes_t.
 *  @param[in]
 *  buf  The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of buf buffer (in bytes) for verification.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If the attribute was set successfully..
 *  \retval HIPBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 * doesn't match the size of the internal storage for the selected attribute.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc,
                                hipblasLtMatmulDescAttributes_t attr,
                                const void *buf, size_t sizeInBytes);

/*! \ingroup library_module
 *  \brief Query attribute from a matrix multiply descriptor
 *
 *  \details
 *  This function returns the value of the queried attribute belonging to a
 * previously created matrix multiply descriptor.
 *
 *  @param[in]
 *  matmulDesc  Pointer to the previously created structure holding the matrix
 * multiply descriptor queried by this function. See \ref hipblasLtMatmulDesc_t
 * .
 *  @param[in]
 *  attr  	    The attribute that will be retrieved by this function. See
 * \ref hipblasLtMatmulDescAttributes_t.
 *  @param[out]
 *  buf         Memory address containing the attribute value retrieved by this
 * function.
 *  @param[in]
 *  sizeInBytes Size of \p buf buffer (in bytes) for verification.
 *  @param[out]
 *  sizeWritten Valid only when the return value is HIPBLAS_STATUS_SUCCESS. If
 * sizeInBytes is non-zero: then sizeWritten is the number of bytes actually
 * written; if sizeInBytes is 0: then sizeWritten is the number of bytes needed
 * to write full contents.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS       If attribute's value was successfully
 * written to user memory. \retval HIPBLAS_STATUS_INVALID_VALUE If \p
 * sizeInBytes is 0 and \p sizeWritten is NULL, or if \p sizeInBytes is non-zero
 * and \p buf is NULL, or \p sizeInBytes doesn't match size of internal storage
 * for the selected attribute.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t matmulDesc,
                                hipblasLtMatmulDescAttributes_t attr, void *buf,
                                size_t sizeInBytes, size_t *sizeWritten);

/*! \ingroup library_module
 *  \brief Create a preference descriptor
 *
 *  \details
 *  This function creates a matrix multiply heuristic search preferences
 * descriptor by allocating the memory needed to hold its opaque structure.
 *
 *  @param[out]
 *  pref  Pointer to the structure holding the matrix multiply preferences
 * descriptor created by this function. see \ref hipblasLtMatmulPreference_t .
 *
 *  \retval HIPBLAS_STATUS_SUCCESS         If the descriptor was created
 * successfully. \retval HIPBLAS_STATUS_ALLOC_FAILED    If memory could not be
 * allocated.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t *pref);

/*! \ingroup library_module
 *  \brief Destory a preferences descriptor
 *
 *  \details
 *  This function destroys a previously created matrix multiply preferences
 * descriptor object.
 *
 *  @param[in]
 *  pref  Pointer to the structure holding the matrix multiply preferences
 * descriptor that should be destroyed by this function. See \ref
 * hipblasLtMatmulPreference_t .
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If operation was successful.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref);

/*! \ingroup library_module
 *  \brief Set attribute to a preference descriptor
 *
 *  \details
 *  This function sets the value of the specified attribute belonging to a
 * previously created matrix multiply preferences descriptor.
 *
 *  @param[in]
 *  pref        Pointer to the previously created structure holding the matrix
 * multiply preferences descriptor queried by this function. See \ref
 * hipblasLtMatmulPreference_t
 *  @param[in]
 *  attr  	    The attribute that will be set by this function. See \ref
 * hipblasLtMatmulPreferenceAttributes_t.
 *  @param[in]
 *  buf         The value to which the specified attribute should be set.
 *  @param[in]
 *  sizeInBytes Size of \p buf buffer (in bytes) for verification.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS If the attribute was set successfully..
 *  \retval HIPBLAS_STATUS_INVALID_VALUE If \p buf is NULL or \p sizeInBytes
 * doesn't match the size of the internal storage for the selected attribute.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attr, const void *buf,
    size_t sizeInBytes);

/*! \ingroup library_module
 *  \brief Query attribute from a preference descriptor
 *
 *  \details
 *  This function returns the value of the queried attribute belonging to a
 * previously created matrix multiply heuristic search preferences descriptor.
 *
 *  @param[in]
 *  pref        Pointer to the previously created structure holding the matrix
 * multiply heuristic search preferences descriptor queried by this function.
 * See \ref hipblasLtMatmulPreference_t.
 *  @param[in]
 *  attr  	    The attribute that will be retrieved by this function. See
 * \ref hipblasLtMatmulPreferenceAttributes_t.
 *  @param[out]
 *  buf         Memory address containing the attribute value retrieved by this
 * function.
 *  @param[in]
 *  sizeInBytes Size of \p buf buffer (in bytes) for verification.
 *  @param[out]
 *  sizeWritten Valid only when the return value is HIPBLAS_STATUS_SUCCESS. If
 * sizeInBytes is non-zero: then sizeWritten is the number of bytes actually
 * written; if sizeInBytes is 0: then sizeWritten is the number of bytes needed
 * to write full contents.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS       If attribute's value was successfully
 * written to user memory. \retval HIPBLAS_STATUS_INVALID_VALUE If \p
 * sizeInBytes is 0 and \p sizeWritten is NULL, or if \p sizeInBytes is non-zero
 * and \p buf is NULL, or \p sizeInBytes doesn't match size of internal storage
 * for the selected attribute.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulPreferenceGetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attr, void *buf, size_t sizeInBytes,
    size_t *sizeWritten);

/*! \ingroup library_module
 *  \brief Retrieve the possible algorithms
 *
 *  \details
 *  This function retrieves the possible algorithms for the matrix multiply
 * operation hipblasLtMatmul() function with the given input matrices A, B and
 * C, and the output matrix D. The output is placed in heuristicResultsArray[]
 * in the order of increasing estimated compute time.
 *
 *  @param[in]
 *  handle                  Pointer to the allocated hipBLASLt handle for the
 * hipBLASLt context. See \ref hipblasLtHandle_t .
 *  @param[in]
 *  matmulDesc              Handle to a previously created matrix multiplication
 * descriptor of type \ref hipblasLtMatmulDesc_t .
 *  @param[in]
 *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
 * descriptors of the type \ref hipblasLtMatrixLayout_t .
 *  @param[in]
 *  pref                    Pointer to the structure holding the heuristic
 * search preferences descriptor. See \ref hipblasLtMatmulPreference_t .
 *  @param[in]
 *  requestedAlgoCount      Size of the \p heuristicResultsArray (in elements).
 * This is the requested maximum number of algorithms to return.
 *  @param[out]
 *  heuristicResultsArray[] Array containing the algorithm heuristics and
 * associated runtime characteristics, returned by this function, in the order
 * of increasing estimated compute time.
 *  @param[out]
 *  returnAlgoCount         Number of algorithms returned by this function. This
 * is the number of \p heuristicResultsArray elements written.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS           If query was successful. Inspect
 * heuristicResultsArray[0 to (returnAlgoCount -1)].state for the status of the
 * results. \retval HIPBLAS_STATUS_NOT_SUPPORTED     If no heuristic function
 * available for current configuration. \retval HIPBLAS_STATUS_INVALID_VALUE If
 * \p requestedAlgoCount is less or equal to zero.
 */
HIPBLASLT_EXPORT
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc,
    hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc,
    hipblasLtMatmulPreference_t pref, int requestedAlgoCount,
    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int *returnAlgoCount);

/*! \ingroup library_module
 *  \brief Retrieve the possible algorithms
 *
 *  \details
 *  This function computes the matrix multiplication of matrices A and B to
 * produce the output matrix D, according to the following operation: \p D = \p
 * alpha*( \p A *\p B) + \p beta*( \p C ), where \p A, \p B, and \p C are input
 * matrices, and \p alpha and \p beta are input scalars. Note: This function
 * supports both in-place matrix multiplication (C == D and Cdesc == Ddesc) and
 * out-of-place matrix multiplication (C != D, both matrices must have the same
 * data type, number of rows, number of columns, batch size, and memory order).
 * In the out-of-place case, the leading dimension of C can be different from
 * the leading dimension of D. Specifically the leading dimension of C can be 0
 * to achieve row or column broadcast. If Cdesc is omitted, this function
 * assumes it to be equal to Ddesc.
 *
 *  @param[in]
 *  handle                  Pointer to the allocated hipBLASLt handle for the
 * hipBLASLt context. See \ref hipblasLtHandle_t .
 *  @param[in]
 *  matmulDesc              Handle to a previously created matrix multiplication
 * descriptor of type \ref hipblasLtMatmulDesc_t .
 *  @param[in]
 *  alpha,beta              Pointers to the scalars used in the multiplication.
 *  @param[in]
 *  Adesc,Bdesc,Cdesc,Ddesc Handles to the previously created matrix layout
 * descriptors of the type \ref hipblasLtMatrixLayout_t .
 *  @param[in]
 *  A,B,C                   Pointers to the GPU memory associated with the
 * corresponding descriptors \p Adesc, \p Bdesc and \p Cdesc .
 *  @param[out]
 *  D                       Pointer to the GPU memory associated with the
 * descriptor \p Ddesc .
 *  @param[in]
 *  algo                    Handle for matrix multiplication algorithm to be
 * used. See \ref hipblasLtMatmulAlgo_t. When NULL, an implicit heuritics query
 * with default search preferences will be performed to determine actual
 * algorithm to use.
 *  @param[in]
 *  workspace               Pointer to the workspace buffer allocated in the GPU
 * memory. Pointer must be 16B aligned (that is, lowest 4 bits of address must
 * be 0).
 *  @param[in]
 *  workspaceSizeInBytes    Size of the workspace.
 *  @param[in]
 *  stream                  The HIP stream where all the GPU work will be
 * submitted.
 *
 *  \retval HIPBLAS_STATUS_SUCCESS           If the operation completed
 * successfully. \retval HIPBLAS_STATUS_EXECUTION_FAILED  If HIP reported an
 * execution error from the device. \retval HIPBLAS_STATUS_ARCH_MISMATCH     If
 * the configured operation cannot be run using the selected device. \retval
 * HIPBLAS_STATUS_NOT_SUPPORTED     If the current implementation on the
 * selected device doesn't support the configured operation. \retval
 * HIPBLAS_STATUS_INVALID_VALUE     If the parameters are unexpectedly NULL, in
 * conflict or in an impossible configuration. For example, when
 * workspaceSizeInBytes is less than workspace required by the configured algo.
 *  \retval HIBLAS_STATUS_NOT_INITIALIZED    If hipBLASLt handle has not been
 * initialized.
 */
HIPBLASLT_EXPORT
hipblasStatus_t
hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc,
                const void *alpha, const void *A, hipblasLtMatrixLayout_t Adesc,
                const void *B, hipblasLtMatrixLayout_t Bdesc, const void *beta,
                const void *C, hipblasLtMatrixLayout_t Cdesc, void *D,
                hipblasLtMatrixLayout_t Ddesc,
                const hipblasLtMatmulAlgo_t *algo, void *workspace,
                size_t workspaceSizeInBytes, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // _HIPBLASLT_H_
