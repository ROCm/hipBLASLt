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

/*! \file
 *  \brief rocblaslt-auxiliary.h provides auxilary functions in rocblaslt
 */

#pragma once
#ifndef _ROCBLASLT_AUXILIARY_H_
#define _ROCBLASLT_AUXILIARY_H_

#include <stdint.h>

#include "rocblaslt-types.h"

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a rocblaslt handle
 *
 *  \details
 *  \p rocblaslt_create creates the rocBLASLt library context. It must be
 *  initialized before any other rocBLASLt API function is invoked and must be
 * passed to all subsequent library function calls. The handle should be
 * destroyed at the end using rocblaslt_destroy_handle().
 *
 *  @param[out]
 *  handle  the pointer to the handle to the rocBLASLt library context.
 *
 *  \retval rocblaslt_status_success the initialization succeeded.
 *  \retval rocblaslt_status_invalid_handle \p handle pointer is invalid.
 *  \retval rocblaslt_status_internal_error an internal error occurred.
 */
rocblaslt_status rocblaslt_create(rocblaslt_handle *handle);

/*! \ingroup aux_module
 *  \brief Destroy a rocblaslt handle
 *
 *  \details
 *  \p rocblaslt_destroy destroys the rocBLASLt library context and releases all
 *  resources used by the rocBLASLt library.
 *
 *  @param[in]
 *  handle  the handle to the rocBLASLt library context.
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_handle \p handle is invalid.
 *  \retval rocblaslt_status_internal_error an internal error occurred.
 */
rocblaslt_status rocblaslt_destroy(const rocblaslt_handle handle);

/*! \ingroup aux_module
 *  \brief Create a descriptor for matrix
 *  \details
 *  \p rocblaslt_matrix_layout_create creates a matrix descriptor It initializes
 *  It should be destroyed at the end using rocblaslt_matrix_layout_destory().
 *
 *  @param[out]
 *  matDescr   the pointer to the matrix descriptor
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_pointer \p descr pointer is invalid.
 *  \retval rocblaslt_status_invalid_value
 */
rocblaslt_status
rocblaslt_matrix_layout_create(rocblaslt_matrix_layout *matDescr,
                               hipblasDatatype_t valueType, uint64_t rows,
                               uint64_t cols, int64_t ld);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p rocblaslt_matrix_layout_destory destroys a matrix descriptor and releases
 * all resources used by the descriptor
 *
 *  @param[in]
 *  descr   the matrix descriptor
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_pointer \p descr is invalid.
 */
rocblaslt_status
rocblaslt_matrix_layout_destory(const rocblaslt_matrix_layout descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix attribute of a matrix descriptor
 *
 *  \details
 *  \p rocblaslt_matrix_layout_set_attribute sets the value of the specified
 * attribute belonging to matrix descr such as number of batches and their
 * stride.
 *
 *  @param[inout]
 *  matDescr        the matrix descriptor
 *  @param[in]
 *  handle          the rocblaslt handle
 *  matAttribute    \ref rocblaslt_mat_num_batches, \ref
 * rocblaslt_mat_batch_stride. data            pointer to the value to which the
 * specified attribute will be set. dataSize        size in bytes of the
 * attribute value used for verification.
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_handle \p handle or \p descr pointer is
 * invalid. \retval rocblaslt_status_invalid_pointer \p data pointer is invalid.
 *  \retval rocblaslt_status_invalid_value \p rocblaslt_matrix_layout_attribute
 * is invalid.
 */
rocblaslt_status
rocblaslt_matmul_desc_create(rocblaslt_matmul_desc *matmulDesc,
                             rocblaslt_compute_type computeType,
                             hipblasDatatype_t scaleType);

/*! \ingroup aux_module
 *  \brief Destroy a matrix multiplication descriptor
 *
 *  \details
 *  \p rocblaslt_matrix_layout_destory destroys a multiplication matrix descr.
 *
 *  @param[in]
 *  descr   the matrix multiplication descriptor
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_value \p descr is invalid.
 */
rocblaslt_status
rocblaslt_matmul_desc_destroy(const rocblaslt_matmul_desc descr);

/*! \ingroup aux_module
 *  \brief Specify the attribute of a matrix multiplication descriptor
 *
 *  \details
 *  \p rocblaslt_matmul_desc_set_attribute sets the value of the specified
 * attribute belonging to matrix multiplication descriptor.
 *
 *  @param[inout]
 *  matmulDesc    the matrix multiplication descriptor
 *  @param[in]
 *  attribute
 *  buf            pointer to the value to which the specified attribute will
 * be set. dataSize        size in bytes of the attribute value used for
 * verification.
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_handle \p handle or \p pref pointer
 * is invalid. \retval rocblaslt_status_invalid_pointer \p data pointer is
 * invalid. \retval rocblaslt_status_invalid_value \p
 * rocblaslt_matmul_desc_attributes is invalid.
 */
rocblaslt_status
rocblaslt_matmul_desc_set_attribute(rocblaslt_matmul_desc matmulDesc,
                                    rocblaslt_matmul_desc_attributes matmulAttr,
                                    const void *buf, size_t sizeInBytes);
/*! \ingroup aux_module
 *  \brief Get the specific attribute from matrix multiplication descriptor
 *
 *  \details
 *  \p rocblaslt_matmul_preference_get_attribute returns the value of the
 * queried attribute belonging to matrix multiplication descriptor.
 *
 *  @param[inout]
 *  buf            the memory address containing the attribute value retrieved
 * by this function
 *
 *  @param[in]
 *  matmulDesc     the matrix multiplication descriptor
 *  sizeInBytes    size in bytes of the attribute value used for verification.
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_handle \p handle or \p pref pointer
 * is invalid. \retval rocblaslt_status_invalid_pointer \p data pointer is
 * invalid. \retval rocblaslt_status_invalid_value \p
 * rocblaslt_matmul_desc_attributes is invalid.
 */
rocblaslt_status
rocblaslt_matmul_desc_get_attribute(rocblaslt_matmul_desc matmulDesc,
                                    rocblaslt_matmul_desc_attributes matmulAttr,
                                    void *buf, size_t sizeInBytes,
                                    size_t *sizeWritten);

/*! \ingroup aux_module
 *  \brief Initializes the algorithm selection descriptor
 *  \details
 *  \p rocblaslt_matmul_preference_create creates a algorithm selection
 * descriptor. It should be destroyed at the end using
 * rocblaslt_matmul_preference_destroy().
 *
 *  @param[out]
 *  pref the pointer to the algorithm selection descriptor
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_pointer \p pref pointer is invalid.
 *  \retval rocblaslt_status_invalid_value
 */
rocblaslt_status
rocblaslt_matmul_preference_create(rocblaslt_matmul_preference *pref);
/*! \ingroup aux_module
 *  \brief Destroy a algorithm selection descriptor
 *
 *  \details
 *  \p rocblaslt_matmul_preference_destroy destroys a algorithm selection
 * descriptor and releases all resources used by the descriptor
 *
 *  @param[in]
 *  pref   the algorithm selection descriptor
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_pointer \p pref is invalid.
 */
rocblaslt_status
rocblaslt_matmul_preference_destroy(const rocblaslt_matmul_preference pref);

/*! \ingroup aux_module
 *  \brief Specify the algorithm attribute of a algorithm selection descriptor
 *
 *  \details
 *  \p rocblaslt_matmul_preference_set_attribute sets the value of the specified
 * attribute belonging to algorithm selection descriptor.
 *
 *  @param[inout]
 *  pref    the algorithm selection descriptor
 *  @param[in]
 *  attribute
 *  data            pointer to the value to which the specified attribute will
 * be set. dataSize        size in bytes of the attribute value used for
 * verification.
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_handle \p handle or \p pref pointer
 * is invalid. \retval rocblaslt_status_invalid_pointer \p data pointer is
 * invalid. \retval rocblaslt_status_invalid_value \p
 * rocblaslt_matmul_preference_attributes is invalid.
 */
rocblaslt_status rocblaslt_matmul_preference_set_attribute(
    rocblaslt_matmul_preference pref,
    rocblaslt_matmul_preference_attributes attribute, const void *data,
    size_t dataSize);

/*! \ingroup aux_module
 *  \brief Get the specific algorithm attribute from algorithm selection
 * descriptor
 *
 *  \details
 *  \p rocblaslt_matmul_preference_get_attribute returns the value of the
 * queried attribute belonging to algorithm selection descriptor.
 *
 *  @param[inout]
 *  data            the memory address containing the attribute value retrieved
 * by this function
 *
 *  @param[in]
 *  pref    the algorithm selection descriptor
 *  dataSize        size in bytes of the attribute value used for verification.
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_handle \p handle or \p pref pointer
 * is invalid. \retval rocblaslt_status_invalid_pointer \p data pointer is
 * invalid. \retval rocblaslt_status_invalid_value \p
 * rocblaslt_matmul_preference_attributes is invalid.
 */
rocblaslt_status rocblaslt_matmul_preference_get_attribute(
    rocblaslt_matmul_preference pref,
    rocblaslt_matmul_preference_attributes attribute, void *data,
    size_t dataSize);

/*! \ingroup aux_module
 *  \brief Get the specific algorithm attribute from algorithm selection
 * descriptor
 *
 *  \details
 *  \p rocblaslt_matmul_algo_get_heuristic returns the possible algorithms for
 * the matrix multiply operation rocblaslt_matmul() function with the given
 * input matrices A, B and C, and the output matrix D. The output is placed in
 * heuristicResultsArray[] in the order of increasing estimated compute time.
 *
 *  @param[out]
 *  heuristicResultsArray
 *  returnAlgoCount
 *
 *  @param[in]
 *  pref    the algorithm selection descriptor
 *
 *  \retval rocblaslt_status_success the operation completed successfully.
 *  \retval rocblaslt_status_invalid_handle \p handle or \p pref pointer
 * is invalid.
 */
rocblaslt_status rocblaslt_matmul_algo_get_heuristic(
    rocblaslt_handle handle, rocblaslt_matmul_desc matmulDesc,
    rocblaslt_matrix_layout Adesc, rocblaslt_matrix_layout Bdesc,
    rocblaslt_matrix_layout Cdesc, rocblaslt_matrix_layout Ddesc,
    rocblaslt_matmul_preference pref, int requestedAlgoCount,
    rocblaslt_matmul_heuristic_result heuristicResultsArray[],
    int *returnAlgoCount);

#ifdef __cplusplus
}
#endif

#endif /* _ROCBLASLT_AUXILIARY_H_ */
