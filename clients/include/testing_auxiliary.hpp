/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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

#include "flops.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <hipblaslt/hipblaslt.h>

void testing_aux_handle_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_handle_destroy_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_handle(const Arguments& arg)
{
    hipblasLtHandle_t handle;
    EXPECT_HIPBLAS_STATUS(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_mat_init_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblasLtMatrixLayout_t m_descr;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutCreate(nullptr, arg.a_type, row, col, ld),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_init(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_handle        handle{arg};
    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_mat_destroy_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutDestroy(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_set_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    int     data;
    int64_t data64;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, sizeof(int)),
                          HIPBLAS_STATUS_INVALID_VALUE);

    data = 1;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, 1),
        HIPBLAS_STATUS_INVALID_VALUE);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutSetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, nullptr, sizeof(int64_t)),
        HIPBLAS_STATUS_INVALID_VALUE);

    data64 = ld * col;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, 1),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_mat_get_attr_bad_arg(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    int     data;
    int64_t data64;
    size_t  sizeWritten;

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            nullptr, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, nullptr, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutGetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, 1, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(mat,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          nullptr,
                                          sizeof(int64_t),
                                          &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    // test sizeWritten is nullptr, and the return state should be success
    data = 0;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(int), nullptr),
        HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(data, 1);
}

void testing_aux_mat_set_get_attr(const Arguments& arg)
{
    const int64_t row = 128;
    const int64_t col = 128;
    const int64_t ld  = 128;

    hipblaslt_local_matrix_layout mat(row, col, ld, arg.a_type);
    EXPECT_HIPBLAS_STATUS(mat.status(), HIPBLAS_STATUS_SUCCESS);

    int32_t data, data_r;
    size_t  sizeWritten;

    data = 2;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatrixLayoutSetAttribute(
                              mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data, sizeof(data)),
                          HIPBLAS_STATUS_SUCCESS);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatrixLayoutGetAttribute(
            mat, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &data_r, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data_r == data);

    std::vector<int64_t> data64_v = {0, ld * col};
    int64_t              data64_r = 0;
    for(int64_t data64 : data64_v)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                mat, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &data64, sizeof(int64_t)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutGetAttribute(mat,
                                              HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                              &data64_r,
                                              sizeof(int64_t),
                                              &sizeWritten),
            HIPBLAS_STATUS_SUCCESS);
        ASSERT_TRUE(data64_r == data64);
    }
}

void testing_aux_matmul_init_bad_arg(const Arguments& arg)
{
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescCreate(nullptr, arg.compute_type, arg.scale_type),
                          HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_init(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);
}

void testing_aux_matmul_set_attr_bad_arg(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              nullptr, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data)),
                          HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, nullptr, sizeof(data)),
                          HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, 1),
        HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_get_attr_bad_arg(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data = HIPBLASLT_EPILOGUE_RELU;
    size_t              sizeWritten;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            nullptr, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_NOT_INITIALIZED);

    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, nullptr, sizeof(data), &sizeWritten),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, 1, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);

    void* dBias = nullptr;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescGetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, 4, &sizeWritten),
                          HIPBLAS_STATUS_INVALID_VALUE);
    // test sizeWritten is nullptr, and the return state should be success
    data = HIPBLASLT_EPILOGUE_RELU;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data), nullptr),
        HIPBLAS_STATUS_SUCCESS);
    // test return buffer value equals HIPBLASLT_EPILOGUE_DEFAULT
    EXPECT_EQ(data, HIPBLASLT_EPILOGUE_DEFAULT);
}

void testing_aux_matmul_set_get_attr(const Arguments& arg)
{
    const hipblasOperation_t opA = HIPBLAS_OP_T;
    const hipblasOperation_t opB = HIPBLAS_OP_N;

    hipblaslt_local_matmul_descr matmul(opA, opB, arg.compute_type, arg.scale_type);
    EXPECT_HIPBLAS_STATUS(matmul.status(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtEpilogue_t data   = HIPBLASLT_EPILOGUE_DEFAULT;
    hipblasLtEpilogue_t data_r = HIPBLASLT_EPILOGUE_RELU;
    size_t              sizeWritten;
    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(
                              matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data, sizeof(data)),
                          HIPBLAS_STATUS_SUCCESS);
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulDescGetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &data_r, sizeof(data_r), &sizeWritten),
        HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(data == data_r);
}

void testing_aux_matmul_alg_init_bad_arg(const Arguments& arg) {}

void testing_aux_matmul_alg_init(const Arguments& arg) {}

void testing_aux_matmul_alg_set_attr_bad_arg(const Arguments& arg) {}

void testing_aux_matmul_alg_get_attr_bad_arg(const Arguments& arg) {}

void testing_aux_matmul_pref_init_bad_arg(const Arguments& arg)
{
    hipblasLtMatmulPreference_t pref;
    size_t                      workspace_size = 0;

    EXPECT_HIPBLAS_STATUS(hipblasLtMatmulPreferenceCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

void testing_aux_matmul_pref_init(const Arguments& arg)
{
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(pref.status(), HIPBLAS_STATUS_SUCCESS);
}
