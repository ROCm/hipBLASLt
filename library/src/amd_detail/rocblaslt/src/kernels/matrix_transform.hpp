#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace amd_detail
{

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
    __global__ void __launch_bounds__(256, 4) transform(
                              DType*       c,
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
            ScaleType aData[VectorWidth];
            ScaleType bData[VectorWidth];

#pragma unroll
            for(uint32_t i = 0; i < VectorWidth; ++i)
            {
                uint32_t offsetA, offsetB;

                if constexpr(RowMajC)
                {
                    offsetA = (transA ? getOffset<RowMajA>(col + i, row, ldA)
                                      : getOffset<RowMajA>(row, col + i, ldA))
                              + batchOffset;
                    offsetB = (transB ? getOffset<RowMajB>(col + i, row, ldB)
                                      : getOffset<RowMajB>(row, col + i, ldB))
                              + batchOffset;
                }
                else
                {
                    offsetA = (transA ? getOffset<RowMajA>(col, row + i, ldA)
                                      : getOffset<RowMajA>(row + i, col, ldA))
                              + batchOffset;
                    offsetB = (transB ? getOffset<RowMajB>(col, row + i, ldB)
                                      : getOffset<RowMajB>(row + i, col, ldB))
                              + batchOffset;
                }

                aData[i] = static_cast<ScaleType>(a[offsetA]);
                bData[i] = static_cast<ScaleType>(b[offsetB]);
            }

            DType    cData[VectorWidth];
            uint32_t offsetC;

#pragma unroll
            for(uint32_t i = 0; i < VectorWidth; ++i)
            {
                if constexpr(RowMajC)
                {
                    offsetC = getOffset<RowMajC>(tRow + blockRow, tCol + blockCol + i, ldC)
                              + batchOffset;
                }
                else
                {
                    offsetC = getOffset<RowMajC>(tRow + blockRow + i, tCol + blockCol, ldC)
                              + batchOffset;
                }

                cData[i]   = static_cast<DType>(alpha * aData[i] + beta * bData[i]);
                c[offsetC] = cData[i];
            }
        }
    }
}
