/*! \file */
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
#ifndef HANDLE_H
#define HANDLE_H

#include "rocblaslt.h"
//#include "rocblaslt_ostream.hpp"
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

struct _rocblaslt_attribute
{
    _rocblaslt_attribute(){};

    ~_rocblaslt_attribute();

    void clear();

    const void* data();

    size_t length();

    size_t get(void* out, size_t size);

    template <typename T>
    size_t get(T* out)
    {
        return get(out, sizeof(T));
    }

    void set(const void* in, size_t size);

    template <typename T>
    void set(const T* in)
    {
        set(in, sizeof(T));
    }

private:
    void*  _data      = nullptr;
    size_t _data_size = 0;
};

/********************************************************************************
 * \brief rocblaslt_handle is a structure holding the rocblaslt library context.
 * It must be initialized using rocblaslt_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocblaslt_destroy_handle().
 *******************************************************************************/
struct _rocblaslt_handle
{
    // constructor
    _rocblaslt_handle();
    // destructor
    ~_rocblaslt_handle() = default;

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    int wavefront_size;
    // asic revision
    int asic_rev;

    // pointer mode ; default mode is host
    rocblaslt_pointer_mode pointer_mode = rocblaslt_pointer_mode_host;
};

/********************************************************************************
 * \brief rocblaslt_matrix_layout is a structure holding the rocblaslt matrix
 * content. It must be initialized using rocblaslt_matrix_layout_create()
 * and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocblaslt_matrix_layout_destory().
 *******************************************************************************/
struct _rocblaslt_matrix_layout
{
    // constructor
    _rocblaslt_matrix_layout(){};
    // destructor
    ~_rocblaslt_matrix_layout(){};

    // num rows
    uint64_t m = 0;
    // num cols
    uint64_t n = 0;
    // leading dimension
    int64_t ld = 0;
    // data type of the matrix
    hipblasDatatype_t type;
    int32_t           batch_count  = 1;
    int64_t           batch_stride = 0;
};

/********************************************************************************
 * \brief rocblaslt_matmul_desc holds the description of the matrix
 *multiplication operation. It is initialized and destroyed with
 *rocblaslt_matmul_desc_create() and rocblaslt_matmul_desc_destroy() functions
 *respectively.
 *******************************************************************************/
struct _rocblaslt_matmul_desc
{
    // constructor
    _rocblaslt_matmul_desc(){};
    // destructor
    ~_rocblaslt_matmul_desc(){};

    // operation applied to the matrix A
    hipblasOperation_t op_A = HIPBLAS_OP_N;
    // operation applied to the matrix B
    hipblasOperation_t op_B = HIPBLAS_OP_N;
    // epilogue operation
    rocblaslt_epilogue epilogue = ROCBLASLT_EPILOGUE_DEFAULT;
    // bias vector pointer
    void*             bias      = nullptr;
    void*             scaleD    = nullptr;
    hipblasDatatype_t bias_type = static_cast<hipblasDatatype_t>(-1);
    //
    rocblaslt_compute_type compute_type;
    hipblasDatatype_t      scale_type;
};

/********************************************************************************
 * \brief rocblaslt_matmul_preference holds the description of the matrix
 * multiplication preference.
 * It is initialized and destroyed with rocblaslt_matmul_preference_create()
 * and rocblaslt_matmul_preference_destroy() functions respectively.
 *******************************************************************************/
struct _rocblaslt_matmul_preference
{
    // constructor
    _rocblaslt_matmul_preference(){};
    // destructor
    ~_rocblaslt_matmul_preference(){};
    //
    uint32_t search_mode         = 0;
    uint64_t max_workspace_bytes = 0;

    int64_t alg_config_id     = 0;
    int64_t alg_max_id        = 0;
    int64_t search_iterations = 0;
};

#endif // HANDLE_H
