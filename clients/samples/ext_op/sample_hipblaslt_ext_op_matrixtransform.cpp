/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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
#include "../common/helper.h"
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

template <bool RowMaj>
int64_t getLeadingDimSize(int64_t numRows, int64_t numCols)
{
    return RowMaj ? numCols : numRows;
}

void simpleTransformF32(int64_t           m,
                        int64_t           n,
                        bool              rowMajA,
                        bool              rowMajB,
                        bool              rowMajC,
                        bool              transA,
                        bool              transB,
                        float*            a,
                        float*            b,
                        float*            c,
                        hipblasLtHandle_t handle,
                        hipStream_t       stream)
{
    int32_t                     batchSize   = 1;
    int64_t                     batchStride = m * n;
    float                       alpha       = 1.f;
    float                       beta        = 0.f;
    hipDataType                 datatype{HIP_R_32F};
    hipDataType                 scaleDatatype{HIP_R_32F};
    hipblasLtOrder_t            orderA = rowMajA ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    hipblasLtOrder_t            orderB = rowMajB ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    hipblasLtOrder_t            orderC = rowMajC ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
    auto                        tA     = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    auto                        tB     = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = transA ? n : m;
    shapeA.second = transA ? m : n;
    shapeB.first  = transB ? n : m;
    shapeB.second = transB ? m : n;

    hipblasLtMatrixTransformDesc_t desc;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype));
    hipblasLtPointerMode_t pMode = HIPBLASLT_POINTER_MODE_HOST;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode)));

    auto ldA = rowMajA ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                       : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    auto ldB = rowMajB ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                       : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    auto ldC = rowMajC ? getLeadingDimSize<true>(m, n) : getLeadingDimSize<false>(m, n);

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &tA, sizeof(tA)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &tB, sizeof(tB)));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride)));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransform(
        handle, desc, &alpha, a, layoutA, &beta, b, layoutB, c, layoutC, stream));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixTransformDescDestroy(desc));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutC));
}

int main(int argc, char** argv)
{
    //example for converting A from row-major to col-major
    int64_t                         m = 4;
    int64_t                         n = 8;
    OptMatrixTransformRunner<float> runner(4, 8, false, false, true, false, false);
    runner.run([&runner]() {
        simpleTransformF32(runner.m,
                           runner.n,
                           runner.rowMajA,
                           runner.rowMajB,
                           runner.rowMajC,
                           runner.transA,
                           runner.transB,
                           runner.da,
                           runner.db,
                           runner.dc,
                           runner.handle,
                           runner.stream);
    });

    return EXIT_SUCCESS;
}
