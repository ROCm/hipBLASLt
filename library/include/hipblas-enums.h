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

/** \file
 *  \brief hipblas_enums.h defines hipblas compatible enums imported from hipblas
 */

#pragma once
#ifndef _HIPBLAS_ENUMS_H_
#define _HIPBLAS_ENUMS_H_

/*! \brief hipblas status codes definition */
typedef enum
{
    HIPBLAS_STATUS_SUCCESS           = 0, /**< Function succeeds */
    HIPBLAS_STATUS_NOT_INITIALIZED   = 1, /**< HIPBLAS library not initialized */
    HIPBLAS_STATUS_ALLOC_FAILED      = 2, /**< resource allocation failed */
    HIPBLAS_STATUS_INVALID_VALUE     = 3, /**< unsupported numerical value was passed to function */
    HIPBLAS_STATUS_MAPPING_ERROR     = 4, /**< access to GPU memory space failed */
    HIPBLAS_STATUS_EXECUTION_FAILED  = 5, /**< GPU program failed to execute */
    HIPBLAS_STATUS_INTERNAL_ERROR    = 6, /**< an internal HIPBLAS operation failed */
    HIPBLAS_STATUS_NOT_SUPPORTED     = 7, /**< function not implemented */
    HIPBLAS_STATUS_ARCH_MISMATCH     = 8, /**< architecture mismatch */
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9, /**< hipBLAS handle is null pointer */
    HIPBLAS_STATUS_INVALID_ENUM      = 10, /**<  unsupported enum value was passed to function */
    HIPBLAS_STATUS_UNKNOWN           = 11, /**<  back-end returned an unsupported status code */
} hipblasStatus_t;

#ifndef HIPBLAS_OPERATION_DECLARED
#define HIPBLAS_OPERATION_DECLARED
/*! \brief Used to specify whether the matrix is to be transposed or not. */
typedef enum
{
    HIPBLAS_OP_N = 111, /**<  Operate with the matrix. */
    HIPBLAS_OP_T = 112, /**<  Operate with the transpose of the matrix. */
    HIPBLAS_OP_C = 113 /**< Operate with the conjugate transpose of the matrix. */
} hipblasOperation_t;

#elif __cplusplus >= 201103L
static_assert(HIPBLAS_OP_N == 111, "Inconsistent declaration of HIPBLAS_OP_N");
static_assert(HIPBLAS_OP_T == 112, "Inconsistent declaration of HIPBLAS_OP_T");
static_assert(HIPBLAS_OP_C == 113, "Inconsistent declaration of HIPBLAS_OP_C");
#endif // HIPBLAS_OPERATION_DECLARED

#endif // _HIPBLASLT_H_
