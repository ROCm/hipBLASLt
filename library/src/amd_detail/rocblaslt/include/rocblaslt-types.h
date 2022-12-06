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
 * \brief rocblaslt-types.h defines data types used by rocblaslt
 */

#pragma once
#ifndef _ROCBLASLT_TYPES_H_
#define _ROCBLASLT_TYPES_H_

#include <hip/hip_bfloat16.h>
#include <hipblas.h>
#include <memory>
#include <stddef.h>
#include <stdint.h>

#define ROCBLASLT_KERNEL __global__
#define ROCBLASLT_DEVICE_ILF __device__

/*! \ingroup types_module
 *  \brief Specifies whether int32 or int64 is used.
 */
#if defined(rocblaslt_ILP64)
typedef int64_t rocblaslt_int;
#else
typedef int32_t rocblaslt_int;
#endif

/* Forward declaration of hipStream_t */
typedef struct ihipStream_t* hipStream_t;

/*! \ingroup types_module
 *  \brief Handle to the rocBLASLt library context queue.
 *
 *  \details
 *  The rocBLASLt handle is a structure holding the rocBLASLt library context.
 * It must be initialized using \ref rocblaslt_create_handle and the returned
 * handle must be passed to all subsequent library function calls. It should be
 * destroyed at the end using \ref rocblaslt_destroy_handle.
 */
typedef struct _rocblaslt_handle* rocblaslt_handle;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The rocBLASLt matrix descriptor is a structure holding all properties of a
 * matrix. It must be initialized using \ref rocblaslt_create_mat_descr and the
 * returned descriptor must be passed to all subsequent library calls that
 * involve the matrix. It should be destroyed at the end using \ref
 * rocblaslt_destroy_mat_descr.
 */
typedef struct _rocblaslt_matrix_layout* rocblaslt_matrix_layout;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication operation
 *
 *  \details
 *  The rocBLASLt matrix multiplication descriptor is a structure holding
 *  the description of the matrix multiplication operation.
 *  It is initialized with \ref rocblaslt_matmul_desc_create function.
 */
typedef struct _rocblaslt_matmul_desc* rocblaslt_matmul_desc;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication preference.
 *
 *  \details
 *  It is initialized with \ref rocblaslt_matmul_preference_create function.
 */
typedef struct _rocblaslt_matmul_preference* rocblaslt_matmul_preference;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix multiplication algorithm.
 *
 *  \details
 *  It is initialized with \ref rocblaslt_matmul_algo_get_heuristic function.
 */
// typedef struct _rocblaslt_matmul_algo rocblaslt_matmul_algo;

/*! \ingroup types_module
 *  \brief Descriptor of the heristic result.
 *
 *  \details
 *  It is initialized with \ref rocblaslt_matmul_algo_get_heuristic function.
 */
// typedef struct _rocblaslt_matmul_heuristic_result
// rocblaslt_matmul_heuristic_result;
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Single precision floating point type */
typedef float rocblaslt_float;

typedef hip_bfloat16 rocblaslt_bfloat16;

#ifdef ROCM_USE_FLOAT16
typedef _Float16 rocblaslt_half;
#else
/*! \brief Structure definition for rocblaslt_half */
typedef struct rocblaslt_half
{
    uint16_t data;
} rocblaslt_half;
#endif

/*! \ingroup types_module
 *  \brief Specify the postprocessing options for the epilogue
 *
 *  \details
 *  The \ref rocblaslt_epilogue indicates the postprocessing for the epilogue.
 */
typedef enum rocblaslt_epilogue_
{
    ROCBLASLT_EPILOGUE_DEFAULT   = 1,
    ROCBLASLT_EPILOGUE_RELU      = 2,
    ROCBLASLT_EPILOGUE_BIAS      = 4,
    ROCBLASLT_EPILOGUE_RELU_BIAS = 6,
    ROCBLASLT_EPILOGUE_GELU      = 32,
    ROCBLASLT_EPILOGUE_GELU_BIAS = 36
} rocblaslt_epilogue;

/*! \ingroup types_module
 *  \brief Indicates if the pointer is device pointer or host pointer.
 *
 *  \details
 *  The \ref rocblaslt_pointer_mode indicates whether scalar values are passed
 * by reference on the host or device. The \ref rocblaslt_pointer_mode can be
 * changed by rocblaslt_set_pointer_mode(). The currently used pointer mode can
 * be obtained by rocblaslt_get_pointer_mode().
 */
typedef enum rocblaslt_pointer_mode_
{
    rocblaslt_pointer_mode_host   = 0, /**< scalar pointers are in host memory. */
    rocblaslt_pointer_mode_device = 1 /**< scalar pointers are in device memory. */
} rocblaslt_pointer_mode;

/*! \ingroup types_module
 *  \brief Indicates if layer is active with bitmask.
 *
 *  \details
 *  The \ref rocblaslt_layer_mode bit mask indicates the logging
 * characteristics.
 */
typedef enum rocblaslt_layer_mode
{
    rocblaslt_layer_mode_none      = 0, /**< layer is not active. */
    rocblaslt_layer_mode_log_error = 1, /**< layer is in error mode. */
    rocblaslt_layer_mode_log_trace = 2, /**< layer is in trace mode. */
    rocblaslt_layer_mode_log_hints = 4, /**< layer is in hints mode. */
    rocblaslt_layer_mode_log_info  = 8, /**< layer is in info mode. */
    rocblaslt_layer_mode_log_api   = 16, /**< layer is in api mode. */
} rocblaslt_layer_mode;

/*! \ingroup types_module
 *  \brief Indicates if layer is active with level.
 *
 *  \details
 *  The \ref rocblaslt_layer_level number indicates the logging characteristics.
 *  A higher log level will show logs including the lower log level.
 */
typedef enum rocblaslt_layer_level
{
    rocblaslt_layer_level_none      = 0, /**< layer is not active. */
    rocblaslt_layer_level_log_error = 1, /**< layer is in error mode. */
    rocblaslt_layer_level_log_trace = 2, /**< layer is in trace mode. */
    rocblaslt_layer_level_log_hints = 3, /**< layer is in hints mode. */
    rocblaslt_layer_level_log_info  = 4, /**< layer is in info mode. */
    rocblaslt_layer_level_log_api   = 5, /**< layer is in api mode. */
} rocblaslt_layer_level;

/*! \ingroup types_module
 *  \brief List of rocblaslt status codes definition.
 *
 *  \details
 *  This is a list of the \ref rocblaslt_status types that are used by the
 * rocBLASLt library.
 */
typedef enum rocblaslt_status_
{
    rocblaslt_status_success                 = 0, /**< success. */
    rocblaslt_status_invalid_handle          = 1, /**< handle not initialized, invalid or null. */
    rocblaslt_status_not_implemented         = 2, /**< function is not implemented. */
    rocblaslt_status_invalid_pointer         = 3, /**< invalid pointer parameter. */
    rocblaslt_status_invalid_size            = 4, /**< invalid size parameter. */
    rocblaslt_status_memory_error            = 5, /**< failed memory allocation, copy, dealloc. */
    rocblaslt_status_internal_error          = 6, /**< other internal library failure. */
    rocblaslt_status_invalid_value           = 7, /**< invalid value parameter. */
    rocblaslt_status_arch_mismatch           = 8, /**< device arch is not supported. */
    rocblaslt_status_zero_pivot              = 9, /**< encountered zero pivot. */
    rocblaslt_status_not_initialized         = 10, /**< descriptor has not been initialized. */
    rocblaslt_status_type_mismatch           = 11, /**< index types do not match. */
    rocblaslt_status_requires_sorted_storage = 12, /**< sorted storage required. */
    rocblaslt_status_continue                = 13 /**< nothing preventing function to proceed. */
} rocblaslt_status;

/*! \ingroup types_module
 *  \brief Specify the compute precision modes of the matrix
 *
 *  \details
 */
typedef enum rocblaslt_compute_type_
{
    rocblaslt_compute_f32 = 300, /**< 32-bit floating-point precision. */
} rocblaslt_compute_type;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix descriptor
 *
 *  \details
 *  The rocblaslt_matrix_layout_attribute is used in the
 *  \ref rocblaslt_matrix_layout_set_attribute and \ref
 * rocblaslt_matrix_layout_get_attribute functions
 */
typedef enum rocblaslt_matrix_layout_attribute_
{
    ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 0, /**< number of matrices in a batch. */
    ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
    = 1, /**< stride between consecutive matrices in a batch expressed in terms
            of matrix elements. */
    ROCBLASLT_MATRIX_LAYOUT_MAX = 2
} rocblaslt_matrix_layout_attribute;

/*! \ingroup types_module
 *  \brief Specify the additional attributes of a matrix multiplication
 * descriptor
 *
 *  \details
 *  The rocblaslt_matmul_descr_attribute_ is used in the
 *  \ref rocblaslt_matmul_descr_set_attribute and \ref
 * rocblaslt_matmul_descr_get_attribute functions
 */
typedef enum rocblaslt_matmul_desc_attributes_
{
    ROCBLASLT_MATMUL_DESC_TRANSA         = 0,
    ROCBLASLT_MATMUL_DESC_TRANSB         = 1,
    ROCBLASLT_MATMUL_DESC_EPILOGUE       = 2,
    ROCBLASLT_MATMUL_DESC_BIAS_POINTER   = 3,
    ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 4,
    ROCBLASLT_MATMUL_DESC_MAX            = 5
} rocblaslt_matmul_desc_attributes;

/*! \ingroup types_module
 *  \brief Specify the matrix multiplication preference attributes.
 *
 *  \details
 *  The \ref rocblaslt_matmul_preference_attributes is used in the
 *  \ref rocblaslt_matmul_preference_get_attribute and \ref
 * rocblaslt_matmul_preference_set_attribute functions.
 */
typedef enum rocblaslt_matmul_preference_attributes_
{
    ROCBLASLT_MATMUL_PREF_SEARCH_MODE         = 0,
    ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    ROCBLASLT_MATMUL_PREF_MAX                 = 2
} rocblaslt_matmul_preference_attributes;

/********************************************************************************
 * \brief rocblaslt_matmul_algo holds the description of the matrix
 * multiplication algorithm.
 *******************************************************************************/
typedef struct _rocblaslt_matmul_algo
{
    union u
    {
        std::shared_ptr<void> ptr;
        uint8_t               data[48];
    } data;
    size_t max_workspace_bytes = 0;
} rocblaslt_matmul_algo;

/********************************************************************************
 * \brief rocblaslt_matmul_heuristic holds the configured matrix
 * multiplication algorithm descriptor and its runtime properties.
 *******************************************************************************/
typedef struct _rocblaslt_matmul_heuristic_result
{
    rocblaslt_matmul_algo algo;
    size_t                workspaceSize = 0;
    rocblaslt_status      state         = rocblaslt_status_success;
    float                 wavesCount    = 1.0;
    int                   reserved[4];
} rocblaslt_matmul_heuristic_result;
#ifdef __cplusplus
}
#endif

#endif /* _ROCBLASLT_TYPES_H_ */
