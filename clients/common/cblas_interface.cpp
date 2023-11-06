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
#include "cblas_interface.hpp"
#include "hipblaslt_vector.hpp"
#include "utility.hpp"
#include <bitset>
#include <iostream>
#include <omp.h>

CBLAS_TRANSPOSE HIPOperationToCBLASTanspose(hipblasOperation_t trans)
{
    switch(trans)
    {
    case HIPBLAS_OP_N:
        return CblasNoTrans;
    case HIPBLAS_OP_T:
        return CblasTrans;
    case HIPBLAS_OP_C:
        return CblasConjTrans;
    }
}

template <typename T>
class customVector
{
public:
    void initialize(size_t size)
    {
        m_data.resize(size);
        m_pointer = m_data.data();
    }
    void initialize(const void* buffer)
    {
        m_pointer = const_cast<void*>(buffer);
    }

    operator T*()
    {
        return (T*)m_pointer;
    }

    operator const T*() const
    {
        return (const T*)m_pointer;
    }

    T& operator[](std::size_t i)
    {
        return ((T*)m_pointer)[i];
    }

private:
    std::vector<T> m_data;
    void*          m_pointer = nullptr;
};

template <typename TiA, typename TiB, typename To, typename Tc, typename Tci>
void cblas_gemm(hipblasOperation_t     transA,
                hipblasOperation_t     transB,
                int64_t                m,
                int64_t                n,
                int64_t                k,
                Tc                     alpha,
                const TiA*             A,
                int64_t                lda,
                const TiB*             B,
                int64_t                ldb,
                Tc                     beta,
                std::add_pointer_t<To> C,
                int64_t                ldc,
                const Tc*              AlphaVec,
                Tc                     scaleA,
                Tc                     scaleB,
                Tc                     scaleD,
                bool                   alt)
{
    using TcCast  = std::conditional_t<std::is_same<Tc, int32_t>::value, double, Tc>;
    using TciCast = std::conditional_t<std::is_same<Tci, int32_t>::value, double, Tci>;

    // cblas does not support hipblasLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    customVector<TcCast> A_Tc, B_Tc, C_Tc;
    if(AlphaVec != nullptr)
    {
        A_Tc.initialize(sizeA);
        if constexpr(sizeof(TiA) > sizeof(TciCast))
        {
            if(transA == HIPBLAS_OP_N)
            {
                for(size_t i = 0; i < sizeA; i++)
                    A_Tc[i] = static_cast<TcCast>(static_cast<TciCast>(A[i] / scaleA))
                              * AlphaVec[i % m];
            }
            else
            {
                for(size_t i = 0; i < sizeA; i++)
                    A_Tc[i] = static_cast<TcCast>(static_cast<TciCast>(A[i] / scaleA))
                              * AlphaVec[i / k];
            }
        }
        else
        {
            if(transA == HIPBLAS_OP_N)
            {
                for(size_t i = 0; i < sizeA; i++)
                    A_Tc[i] = static_cast<TcCast>(A[i]) * AlphaVec[i % m];
            }
            else
            {
                for(size_t i = 0; i < sizeA; i++)
                    A_Tc[i] = static_cast<TcCast>(A[i]) * AlphaVec[i / k];
            }
        }
    }
    else if constexpr(std::is_same<TiA, TcCast>::value && std::is_same<TciCast, TcCast>::value)
    {
        A_Tc.initialize(A);
    }
    else
    {
        A_Tc.initialize(sizeA);
        if constexpr(sizeof(TiA) > sizeof(TciCast))
        {
            for(size_t i = 0; i < sizeA; i++)
            {
                A_Tc[i] = static_cast<TcCast>(static_cast<TciCast>(A[i] / scaleA));
            }
        }
        else
        {
            for(size_t i = 0; i < sizeA; i++)
            {
                A_Tc[i] = static_cast<TcCast>(A[i]);
            }
        }
    }

    if constexpr(std::is_same<TiB, TcCast>::value && std::is_same<TciCast, TcCast>::value)
    {
        B_Tc.initialize(B);
    }
    else
    {
        B_Tc.initialize(sizeB);
        if constexpr(sizeof(TiB) > sizeof(TciCast))
        {
            for(size_t i = 0; i < sizeB; i++)
            {
                B_Tc[i] = static_cast<TcCast>(static_cast<TciCast>(B[i] / scaleB));
            }
        }
        else
        {
            for(size_t i = 0; i < sizeB; i++)
            {
                B_Tc[i] = static_cast<TcCast>(B[i]);
            }
        }
    }

    if constexpr(std::is_same<To, TcCast>::value)
    {
        C_Tc.initialize(C);
    }
    else
    {
        C_Tc.initialize(sizeC);
        for(size_t i = 0; i < sizeC; i++)
        {
            C_Tc[i] = static_cast<TcCast>(C[i]);
        }
    }

    alpha *= scaleA * scaleB;

    TcCast alphaCast = (TcCast)alpha;
    TcCast betaCast  = (TcCast)beta;

    // just directly cast, since transA, transB are integers in the enum
    //printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    if constexpr(std::is_same<TcCast, float>::value)
    {
        cblas_sgemm(CblasColMajor,
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alphaCast,
                    A_Tc,
                    lda,
                    B_Tc,
                    ldb,
                    betaCast,
                    C_Tc,
                    ldc);
    }
    else if constexpr(std::is_same<TcCast, double>::value)
    {
        cblas_dgemm(CblasColMajor,
                    HIPOperationToCBLASTanspose(transA),
                    HIPOperationToCBLASTanspose(transB),
                    m,
                    n,
                    k,
                    alphaCast,
                    A_Tc,
                    lda,
                    B_Tc,
                    ldb,
                    betaCast,
                    C_Tc,
                    ldc);
    }

    if(scaleD != 1)
    {
        for(size_t i = 0; i < sizeC; i++)
            C[i] = static_cast<To>(C_Tc[i] * scaleD);
    }
    else
    {
        if constexpr(!std::is_same<To, TcCast>::value)
        {
            for(size_t i = 0; i < sizeC; i++)
                C[i] = static_cast<To>(C_Tc[i]);
        }
    }
}

#define CREATEFUNCTION(TiA, TiB, To, Tc, Tci)                                        \
    template void cblas_gemm<TiA, TiB, To, Tc, Tci>(hipblasOperation_t     transA,   \
                                                    hipblasOperation_t     transB,   \
                                                    int64_t                m,        \
                                                    int64_t                n,        \
                                                    int64_t                k,        \
                                                    Tc                     alpha,    \
                                                    const TiA*             A,        \
                                                    int64_t                lda,      \
                                                    const TiB*             B,        \
                                                    int64_t                ldb,      \
                                                    Tc                     beta,     \
                                                    std::add_pointer_t<To> C,        \
                                                    int64_t                ldc,      \
                                                    const Tc*              AlphaVec, \
                                                    Tc                     scaleA,   \
                                                    Tc                     scaleB,   \
                                                    Tc                     scaleD,   \
                                                    bool                   alt);

CREATEFUNCTION(hip_bfloat16, hip_bfloat16, hip_bfloat16, float, hip_bfloat16)
CREATEFUNCTION(hip_bfloat16, hip_bfloat16, float, float, hip_bfloat16)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, float, float, hipblaslt_bf8_fnuz)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, float, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float, hipblaslt_bf8_fnuz)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, hipblasLtHalf, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hipblasLtHalf)
CREATEFUNCTION(hipblasLtHalf, hipblasLtHalf, float, float, hipblasLtHalf)
CREATEFUNCTION(float, float, float, float, float)
CREATEFUNCTION(double, double, double, double, double)
CREATEFUNCTION(int8_t, int8_t, int32_t, int32_t, int8_t)
// Mix precision
// FP16FP8 mix FP16 in MFMA
CREATEFUNCTION(hipblasLtHalf, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float, hipblasLtHalf)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblasLtHalf, hipblaslt_f8_fnuz, float, hipblasLtHalf)
CREATEFUNCTION(hipblasLtHalf, hipblaslt_f8_fnuz, hipblasLtHalf, float, hipblasLtHalf)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblasLtHalf, hipblasLtHalf, float, hipblasLtHalf)
CREATEFUNCTION(hipblasLtHalf, hipblaslt_f8_fnuz, float, float, hipblasLtHalf)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblasLtHalf, float, float, hipblasLtHalf)
// Mix precision
// FP16FP8 mix FP8 in MFMA
CREATEFUNCTION(hipblasLtHalf, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblasLtHalf, hipblaslt_f8_fnuz, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblasLtHalf, hipblaslt_f8_fnuz, hipblasLtHalf, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblasLtHalf, hipblasLtHalf, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblasLtHalf, hipblaslt_f8_fnuz, float, float, hipblaslt_f8_fnuz)
CREATEFUNCTION(hipblaslt_f8_fnuz, hipblasLtHalf, float, float, hipblaslt_f8_fnuz)
