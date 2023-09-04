/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace amd_detail
{
    template <typename DType, size_t VectorWidth>
    struct VectorIOType
    {
        using vtype = void;
        constexpr static size_t vw{VectorWidth};
    };

    template <>
    struct VectorIOType<int8_t, 4>
    {
        using vtype = char4;
    };

    template <>
    struct VectorIOType<int32_t, 4>
    {
        using vtype = int4;
    };

    template <>
    struct VectorIOType<float, 4>
    {
        using vtype = float4;
    };

    template <bool RowMaj>
    __device__ uint32_t getOffset(uint32_t row, uint32_t col, uint32_t ld)
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

    template <bool RowMaj, uint32_t TileM, uint32_t TileN, uint32_t VectorWidth>
    __device__ uint32_t getThreadLocalRowIdx(uint32_t tId)
    {
        constexpr auto numComponentsM
            = RowMaj ? TileM : TileM / VectorWidth + !!(TileM % VectorWidth);
        constexpr auto v = RowMaj ? 1 : VectorWidth;
        return (tId % numComponentsM) * v;
    }

    template <bool RowMaj, uint32_t TileM, uint32_t TileN, uint32_t VectorWidth>
    __device__ uint32_t getThreadLocalColIdx(uint32_t tId)
    {
        constexpr auto numComponentsM
            = RowMaj ? TileM : TileM / VectorWidth + !!(TileM % VectorWidth);
        constexpr auto v = RowMaj ? VectorWidth : 1;
        return (tId / numComponentsM) * v;
    }

    template <bool RowMaj>
    uint32_t getLeadingDimSize(uint32_t numRows, uint32_t numCols)
    {
        return RowMaj ? numCols : numRows;
    }

    template <typename DType,
              typename ScaleType,
              bool     RowMajA,
              bool     RowMajB,
              bool     RowMajC,
              uint32_t NumThreadsM,
              uint32_t NumThreadsN,
              uint32_t VectorWidth>
    __global__ void __launch_bounds__(256, 4) transform(DType*       c,
                                                        const DType* a,
                                                        const DType* b,
                                                        ScaleType    alpha,
                                                        ScaleType    beta,
                                                        uint32_t     numRows,
                                                        uint32_t     numCols,
                                                        uint32_t     ldA,
                                                        uint32_t     ldB,
                                                        uint32_t     ldC,
                                                        uint32_t     batchStride,
                                                        bool         transA,
                                                        bool         transB)
    {
        constexpr auto TileM              = RowMajC ? NumThreadsM : NumThreadsM * VectorWidth;
        constexpr auto TileN              = RowMajC ? NumThreadsN * VectorWidth : NumThreadsN;
        const auto     tId                = threadIdx.x;
        const auto     bId                = blockIdx.x;
        const auto     numThreadsPerBlock = blockDim.x;
        assert(TileM * TileN == numThreadsPerBlock * VectorWidth);
        const auto numTilesM = numRows / TileM + !!(numRows % TileM);
        const auto batchIdx  = blockIdx.z;
        const auto blockRow  = (bId % numTilesM) * TileM;
        const auto blockCol  = (bId / numTilesM) * TileN;
        const auto tRow      = getThreadLocalRowIdx<RowMajC, TileM, TileN, VectorWidth>(tId);
        const auto tCol      = getThreadLocalColIdx<RowMajC, TileM, TileN, VectorWidth>(tId);
        const auto row       = blockRow + tRow;
        const auto col       = blockCol + tCol;

        if(row >= numRows || col >= numCols)
        {
            return;
        }

        const auto batchOffset = batchIdx * batchStride;

        if constexpr(VectorWidth == 1)
        {
            const auto offsetA
                = (transA ? getOffset<RowMajA>(col, row, ldA) : getOffset<RowMajA>(row, col, ldA))
                  + batchOffset;
            const auto offsetB
                = (transB ? getOffset<RowMajB>(col, row, ldB) : getOffset<RowMajB>(row, col, ldB))
                  + batchOffset;
            const ScaleType aData = static_cast<ScaleType>(a[offsetA]);
            const ScaleType bData = static_cast<ScaleType>(b[offsetB]);
            const DType     cData = static_cast<DType>(aData * alpha + bData * beta);
            const auto      offsetC
                = getOffset<RowMajC>(tRow + blockRow, tCol + blockCol, ldC) + batchOffset;
            c[offsetC] = cData;
        }
        else
        {
            const auto vectorWriteDirSize = RowMajC ? numCols : numRows;
            const auto blockVectorWriteEndBound
                = RowMajC ? (col + VectorWidth) : (row + VectorWidth);
            const auto vectorShift = blockVectorWriteEndBound > vectorWriteDirSize
                                         ? (blockVectorWriteEndBound - vectorWriteDirSize)
                                         : 0;
            ScaleType  aData[VectorWidth];
            ScaleType  bData[VectorWidth];

#pragma unroll
            for(uint32_t i = 0; i < VectorWidth; ++i)
            {
                uint32_t offsetA, offsetB;

                if constexpr(RowMajC)
                {
                    offsetA = (transA ? getOffset<RowMajA>(col + i - vectorShift, row, ldA)
                                      : getOffset<RowMajA>(row, col + i - vectorShift, ldA))
                              + batchOffset;
                    offsetB = (transB ? getOffset<RowMajB>(col + i - vectorShift, row, ldB)
                                      : getOffset<RowMajB>(row, col + i - vectorShift, ldB))
                              + batchOffset;
                }
                else
                {
                    offsetA = (transA ? getOffset<RowMajA>(col, row + i - vectorShift, ldA)
                                      : getOffset<RowMajA>(row + i - vectorShift, col, ldA))
                              + batchOffset;
                    offsetB = (transB ? getOffset<RowMajB>(col, row + i - vectorShift, ldB)
                                      : getOffset<RowMajB>(row + i - vectorShift, col, ldB))
                              + batchOffset;
                }

                aData[i] = static_cast<ScaleType>(a[offsetA]);
                bData[i] = static_cast<ScaleType>(b[offsetB]);
            }

            //only begin index is required, since vector write always along continuous direction
            uint32_t cOffset{};

            if constexpr(RowMajC)
            {
                cOffset = getOffset<RowMajC>(tRow + blockRow, tCol + blockCol - vectorShift, ldC)
                          + batchOffset;
            }
            else
            {
                cOffset = getOffset<RowMajC>(tRow + blockRow - vectorShift, tCol + blockCol, ldC)
                          + batchOffset;
            }

            using VectorType = typename VectorIOType<DType, VectorWidth>::vtype;

            if constexpr(std::is_same<VectorType, void>::value)
            {
                DType cData[VectorWidth];
#pragma unroll
                for(uint32_t i = 0; i < VectorWidth; ++i)
                {
                    cData[i] = static_cast<DType>(alpha * aData[i] + beta * bData[i]);
                }

                if(!vectorShift)
                {
#pragma unroll
                    for(uint32_t i = 0; i < VectorWidth; ++i)
                    {
                        c[cOffset + i] = cData[i];
                    }
                }
                else
                {
                    for(uint32_t i = vectorShift; i < VectorWidth; ++i)
                    {
                        c[cOffset + i] = cData[i];
                    }
                }
            }
            else
            {
                VectorType cData;
#pragma unroll
                for(uint32_t i = 0; i < VectorWidth; ++i)
                {
                    cData.data[i] = static_cast<DType>(alpha * aData[i] + beta * bData[i]);
                }

                if(!vectorShift)
                {
#pragma unroll
                    for(uint32_t i = 0; i < VectorWidth; ++i)
                    {
                        c[cOffset + i] = cData.data[i];
                    }
                }
                else
                {
                    for(uint32_t i = vectorShift; i < VectorWidth; ++i)
                    {
                        c[cOffset + i] = cData.data[i];
                    }
                }
            }
        }
    }
}
