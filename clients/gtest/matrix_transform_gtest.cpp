/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023 Advanced Micro Devices, Inc.
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
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
    struct MatrixTransformIO
    {
        MatrixTransformIO()                 = default;
        virtual ~MatrixTransformIO()        = default;
        virtual void*  getBuf(size_t i)     = 0;
        virtual size_t elemNumBytes() const = 0;

    private:
        MatrixTransformIO(const MatrixTransformIO&)            = delete;
        MatrixTransformIO(MatrixTransformIO&&)                 = delete;
        MatrixTransformIO& operator=(const MatrixTransformIO&) = delete;
        MatrixTransformIO& operator=(MatrixTransformIO&&)      = delete;
    };

    template <typename DType>
    struct TypedMatrixTransformIO : public MatrixTransformIO
    {
        TypedMatrixTransformIO(int64_t m, int64_t n, int64_t b)
        {
            auto err = hipMalloc(&this->a, m * n * b * sizeof(DType));
            EXPECT_EQ(err, hipSuccess);
            err = hipMalloc(&this->b, m * n * b * sizeof(DType));
            EXPECT_EQ(err, hipSuccess);
            err = hipMalloc(&this->c, m * n * b * sizeof(DType));
            EXPECT_EQ(err, hipSuccess);
            init(this->a, m * n * b);
            init(this->b, m * n * b);
        }

        ~TypedMatrixTransformIO() override
        {
            auto err = hipFree(a);
            err      = hipFree(b);
            err      = hipFree(c);
            EXPECT_EQ(err, hipSuccess);
        }

        void* getBuf(size_t i) override
        {
            void* buf[] = {a, b, c};
            return buf[i];
        }

        size_t elemNumBytes() const override
        {
            return sizeof(DType);
        }

    private:
        void init(DType* buf, size_t len)
        {
            srand(time(nullptr));
            std::vector<DType> ref(len);

            for(auto& i : ref)
            {
                i = DType(rand() % 7 - 3);
            }

            auto err = hipMemcpy(buf, ref.data(), len * sizeof(DType), hipMemcpyHostToDevice);
            ASSERT_EQ(err, hipSuccess);
        }

    private:
        DType* a{};
        DType* b{};
        DType* c{};
    };

    using MatrixTransformIOPtr = std::unique_ptr<MatrixTransformIO>;
    MatrixTransformIOPtr
        makeMatrixTransformIOPtr(hipDataType datatype, int64_t m, int64_t n, int64_t b)
    {
        if(datatype == HIP_R_32F)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtFloat>>(m, n, b);
        }
        else if(datatype == HIP_R_16F)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtHalf>>(m, n, b);
        }
        else if(datatype == HIP_R_16BF)
        {
            return std::make_unique<TypedMatrixTransformIO<hipblasLtBfloat16>>(m, n, b);
        }
        else if(datatype == HIP_R_8I)
        {
            return std::make_unique<TypedMatrixTransformIO<int8_t>>(m, n, b);
        }
        else if(datatype == HIP_R_32I)
        {
            return std::make_unique<TypedMatrixTransformIO<int32_t>>(m, n, b);
        }
        return nullptr;
    }

    template <bool RowMaj>
    int64_t getLeadingDimSize(int64_t numRows, int64_t numCols)
    {
        return RowMaj ? numCols : numRows;
    }

    template <bool RowMaj>
    uint32_t getOffset(uint32_t row, uint32_t col, uint32_t ld)
    {
        if constexpr(RowMaj)
        {
            return ld * row + col;
        }
        else
        {
            return ld * col + row;
        }
    }

    template <typename DType, typename ScaleType, bool RowMajA, bool RowMajB, bool RowMajC>
    void cpuTransform(DType*       c,
                      const DType* a,
                      const DType* b,
                      ScaleType    alpha,
                      ScaleType    beta,
                      bool         transA,
                      bool         transB,
                      uint32_t     m,
                      uint32_t     n,
                      uint32_t     ldA,
                      uint32_t     ldB,
                      uint32_t     ldC,
                      uint32_t     batchSize,
                      uint32_t     batchStride)
    {
        for(uint32_t k = 0; k < batchSize; ++k)
        {
            const int64_t batchOffset = k * int64_t(batchStride);

            for(uint32_t i = 0; i < m; ++i)
            {
                for(uint32_t j = 0; j < n; ++j)
                {
                    const auto offsetA
                        = transA ? getOffset<RowMajA>(j, i, ldA) : getOffset<RowMajA>(i, j, ldA);
                    const auto offsetB
                        = transB ? getOffset<RowMajB>(j, i, ldB) : getOffset<RowMajB>(i, j, ldB);
                    const auto offsetC = getOffset<RowMajC>(i, j, ldC);
                    c[batchOffset + offsetC]
                        = a[batchOffset + offsetA] * alpha + b[batchOffset + offsetB] * beta;
                }
            }
        }
    }

    template <typename DType>
    void validation(void*    c,
                    void*    a,
                    void*    b,
                    float    alpha,
                    float    beta,
                    uint32_t m,
                    uint32_t n,
                    uint32_t ldA,
                    uint32_t ldB,
                    uint32_t ldC,
                    uint32_t batchSize,
                    uint32_t batchStride,
                    bool     rowMajA,
                    bool     rowMajB,
                    bool     rowMajC,
                    bool     transA,
                    bool     transB)
    {
        using std::begin;
        using std::end;
        std::vector<float> hC(m * n * batchSize, 0);
        std::vector<float> hA(m * n * batchSize, 0);
        std::vector<float> hB(m * n * batchSize, 0);
        std::vector<float> cpuRef(m * n * batchSize, 0);
        std::vector<DType> dA(m * n * batchSize);
        std::vector<DType> dB(m * n * batchSize);
        std::vector<DType> dC(m * n * batchSize);
        auto               err = hipMemcpyDtoH(dA.data(), a, m * n * batchSize * sizeof(DType));
        err                    = hipMemcpyDtoH(dB.data(), b, m * n * batchSize * sizeof(DType));
        err                    = hipMemcpyDtoH(dC.data(), c, m * n * batchSize * sizeof(DType));

        ASSERT_EQ(err, hipSuccess);

        std::transform(begin(dC), end(dC), begin(hC), [](auto i) { return float(i); });

        std::transform(begin(dA), end(dA), begin(hA), [](auto i) { return float(i); });

        std::transform(begin(dB), end(dB), begin(hB), [](auto i) { return float(i); });

        if(rowMajA && rowMajB && rowMajC)
        {
            cpuTransform<float, float, true, true, true>(cpuRef.data(),
                                                         hA.data(),
                                                         hB.data(),
                                                         alpha,
                                                         beta,
                                                         transA,
                                                         transB,
                                                         m,
                                                         n,
                                                         ldA,
                                                         ldB,
                                                         ldC,
                                                         batchSize,
                                                         batchStride);
        }
        else if(!rowMajA && rowMajB && rowMajC)
        {
            cpuTransform<float, float, false, true, true>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(rowMajA && !rowMajB && rowMajC)
        {
            cpuTransform<float, float, true, false, true>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(rowMajA && rowMajB && !rowMajC)
        {
            cpuTransform<float, float, true, true, false>(cpuRef.data(),
                                                          hA.data(),
                                                          hB.data(),
                                                          alpha,
                                                          beta,
                                                          transA,
                                                          transB,
                                                          m,
                                                          n,
                                                          ldA,
                                                          ldB,
                                                          ldC,
                                                          batchSize,
                                                          batchStride);
        }
        else if(!rowMajA && !rowMajB && rowMajC)
        {
            cpuTransform<float, float, false, false, true>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(!rowMajA && rowMajB && !rowMajC)
        {
            cpuTransform<float, float, false, true, false>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(rowMajA && !rowMajB && !rowMajC)
        {
            cpuTransform<float, float, true, false, false>(cpuRef.data(),
                                                           hA.data(),
                                                           hB.data(),
                                                           alpha,
                                                           beta,
                                                           transA,
                                                           transB,
                                                           m,
                                                           n,
                                                           ldA,
                                                           ldB,
                                                           ldC,
                                                           batchSize,
                                                           batchStride);
        }
        else if(!rowMajA && !rowMajB && !rowMajC)
        {
            cpuTransform<float, float, false, false, false>(cpuRef.data(),
                                                            hA.data(),
                                                            hB.data(),
                                                            alpha,
                                                            beta,
                                                            transA,
                                                            transB,
                                                            m,
                                                            n,
                                                            ldA,
                                                            ldB,
                                                            ldC,
                                                            batchSize,
                                                            batchStride);
        }

        for(size_t i = 0; i < cpuRef.size(); ++i)
        {
            ASSERT_FLOAT_EQ(cpuRef[i], hC[i]);
        }
    }
}

class MatrixTransformTest : public ::testing::TestWithParam<std::tuple<hipDataType,
                                                                       hipDataType,
                                                                       hipblasOperation_t,
                                                                       hipblasOperation_t,
                                                                       hipblasLtOrder_t,
                                                                       hipblasLtOrder_t,
                                                                       hipblasLtOrder_t>>
{
};

TEST_P(MatrixTransformTest, Basic)
{
    int64_t                     m             = 1024;
    int64_t                     n             = 1024;
    int32_t                     batchSize     = 1;
    auto                        datatype      = std::get<0>(GetParam());
    auto                        scaleDatatype = std::get<1>(GetParam());
    auto                        opA           = std::get<2>(GetParam());
    auto                        opB           = std::get<3>(GetParam());
    auto                        orderA        = std::get<4>(GetParam());
    auto                        orderB        = std::get<5>(GetParam());
    auto                        orderC        = std::get<6>(GetParam());
    float                       alpha         = 1;
    float                       beta          = 1;
    int64_t                     batchStride   = m * n;
    std::pair<int64_t, int64_t> shapeA;
    std::pair<int64_t, int64_t> shapeB;
    shapeA.first  = opA == HIPBLAS_OP_T ? n : m;
    shapeA.second = opA == HIPBLAS_OP_T ? m : n;
    shapeB.first  = opB == HIPBLAS_OP_T ? n : m;
    shapeB.second = opB == HIPBLAS_OP_T ? m : n;
    uint32_t ldA  = (orderA == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeA.first, shapeA.second)
                        : getLeadingDimSize<false>(shapeA.first, shapeA.second);
    uint32_t ldB  = (orderB == HIPBLASLT_ORDER_ROW)
                        ? getLeadingDimSize<true>(shapeB.first, shapeB.second)
                        : getLeadingDimSize<false>(shapeB.first, shapeB.second);
    uint32_t ldC  = (orderC == HIPBLASLT_ORDER_ROW) ? getLeadingDimSize<true>(m, n)
                                                    : getLeadingDimSize<false>(m, n);

    auto  inputs = makeMatrixTransformIOPtr(datatype, m, n, batchSize);
    void* dA     = inputs->getBuf(0);
    void* dB     = inputs->getBuf(1);
    void* dC     = inputs->getBuf(2);

    hipblasLtMatrixTransformDesc_t desc;
    auto                   hipblasLtErr = hipblasLtMatrixTransformDescCreate(&desc, scaleDatatype);
    hipblasLtPointerMode_t pMode        = HIPBLASLT_POINTER_MODE_HOST;
    hipblasLtErr                        = hipblasLtMatrixTransformDescSetAttribute(
        desc,
        hipblasLtMatrixTransformDescAttributes_t::HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pMode,
        sizeof(pMode));

    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);

    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opA, sizeof(opA));
    hipblasLtErr = hipblasLtMatrixTransformDescSetAttribute(
        desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opB, sizeof(opB));
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutA, datatype, shapeA.first, shapeA.second, ldA);
    hipblasLtErr
        = hipblasLtMatrixLayoutCreate(&layoutB, datatype, shapeB.first, shapeB.second, ldB);
    hipblasLtErr = hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, ldC);
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderA,
        sizeof(orderA));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderB,
        sizeof(orderB));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_ORDER,
        &orderC,
        sizeof(orderC));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batchSize,
        sizeof(batchSize));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutA,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutB,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtErr = hipblasLtMatrixLayoutSetAttribute(
        layoutC,
        hipblasLtMatrixLayoutAttribute_t::HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batchStride,
        sizeof(batchStride));
    hipblasLtHandle_t handle{};
    hipblasLtErr = hipblasLtCreate(&handle);
    hipblasLtErr = hipblasLtMatrixTransform(
        handle, desc, &alpha, dA, layoutA, &beta, dB, layoutB, dC, layoutC, nullptr);
    ASSERT_EQ(hipblasLtErr, HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    auto rowMajA = (orderA == HIPBLASLT_ORDER_ROW);
    auto rowMajB = (orderB == HIPBLASLT_ORDER_ROW);
    auto rowMajC = (orderC == HIPBLASLT_ORDER_ROW);
    auto transA  = (opA == HIPBLAS_OP_T);
    auto transB  = (opB == HIPBLAS_OP_T);

    if(datatype == HIP_R_32F)
    {
        validation<float>(dC,
                          dA,
                          dB,
                          alpha,
                          beta,
                          m,
                          n,
                          ldA,
                          ldB,
                          ldC,
                          batchSize,
                          batchStride,
                          rowMajA,
                          rowMajB,
                          rowMajC,
                          transA,
                          transB);
    }
    else if(datatype == HIP_R_16F)
    {
        validation<hipblasLtHalf>(dC,
                                  dA,
                                  dB,
                                  alpha,
                                  beta,
                                  m,
                                  n,
                                  ldA,
                                  ldB,
                                  ldC,
                                  batchSize,
                                  batchStride,
                                  rowMajA,
                                  rowMajB,
                                  rowMajC,
                                  transA,
                                  transB);
    }
    else if(datatype == HIP_R_16BF)
    {
        validation<hipblasLtBfloat16>(dC,
                                      dA,
                                      dB,
                                      alpha,
                                      beta,
                                      m,
                                      n,
                                      ldA,
                                      ldB,
                                      ldC,
                                      batchSize,
                                      batchStride,
                                      rowMajA,
                                      rowMajB,
                                      rowMajC,
                                      transA,
                                      transB);
    }
    else if(datatype == HIP_R_8I)
    {
        validation<int8_t>(dC,
                           dA,
                           dB,
                           alpha,
                           beta,
                           m,
                           n,
                           ldA,
                           ldB,
                           ldC,
                           batchSize,
                           batchStride,
                           rowMajA,
                           rowMajB,
                           rowMajC,
                           transA,
                           transB);
    }
    else if(datatype == HIP_R_32I)
    {
        validation<int32_t>(dC,
                            dA,
                            dB,
                            alpha,
                            beta,
                            m,
                            n,
                            ldA,
                            ldB,
                            ldC,
                            batchSize,
                            batchStride,
                            rowMajA,
                            rowMajB,
                            rowMajC,
                            transA,
                            transB);
    }

    hipblasLtErr = hipblasLtMatrixTransformDescDestroy(desc);
    hipblasLtErr = hipblasLtDestroy(handle);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtErr = hipblasLtMatrixLayoutDestroy(layoutC);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    MatrixTransformTest,
    ::testing::Combine(::testing::ValuesIn({HIP_R_32F, HIP_R_16F, HIP_R_16BF, HIP_R_8I, HIP_R_32I}),
                       ::testing::ValuesIn({HIP_R_32F}),
                       ::testing::ValuesIn({HIPBLAS_OP_N, HIPBLAS_OP_T}),
                       ::testing::ValuesIn({HIPBLAS_OP_N, HIPBLAS_OP_T}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL}),
                       ::testing::ValuesIn({HIPBLASLT_ORDER_ROW, HIPBLASLT_ORDER_COL})));
