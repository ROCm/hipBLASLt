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

// gemm
template <>
void cblas_gemm<hip_bfloat16, hip_bfloat16, float>(hipblasOperation_t  transA,
                                                   hipblasOperation_t  transB,
                                                   int64_t             m,
                                                   int64_t             n,
                                                   int64_t             k,
                                                   float               alpha,
                                                   const hip_bfloat16* A,
                                                   int64_t             lda,
                                                   const hip_bfloat16* B,
                                                   int64_t             ldb,
                                                   float               beta,
                                                   hip_bfloat16*       C,
                                                   int64_t             ldc,
                                                   bool                alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<hip_bfloat16>(C_float[i]);
}

template <>
void cblas_gemm<hip_bfloat16, float, float>(hipblasOperation_t  transA,
                                            hipblasOperation_t  transB,
                                            int64_t             m,
                                            int64_t             n,
                                            int64_t             k,
                                            float               alpha,
                                            const hip_bfloat16* A,
                                            int64_t             lda,
                                            const hip_bfloat16* B,
                                            int64_t             ldb,
                                            float               beta,
                                            float*              C,
                                            int64_t             ldc,
                                            bool                alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<hipblaslt_f8, float, float>(hipblasOperation_t  transA,
                                            hipblasOperation_t  transB,
                                            int64_t             m,
                                            int64_t             n,
                                            int64_t             k,
                                            float               alpha,
                                            const hipblaslt_f8* A,
                                            int64_t             lda,
                                            const hipblaslt_f8* B,
                                            int64_t             ldb,
                                            float               beta,
                                            float*              C,
                                            int64_t             ldc,
                                            bool                alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
        A_float[i] = static_cast<float>(A[i]);
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<hipblasLtHalf, hipblasLtHalf, float>(hipblasOperation_t   transA,
                                                     hipblasOperation_t   transB,
                                                     int64_t              m,
                                                     int64_t              n,
                                                     int64_t              k,
                                                     float                alpha,
                                                     const hipblasLtHalf* A,
                                                     int64_t              lda,
                                                     const hipblasLtHalf* B,
                                                     int64_t              ldb,
                                                     float                beta,
                                                     hipblasLtHalf*       C,
                                                     int64_t              ldc,
                                                     bool                 alt)
{
    // cblas does not support hipblasLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = float_to_bfloat16_truncate(float(A[i]));
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = float_to_bfloat16_truncate(float(B[i]));
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = float_to_bfloat16_truncate(float(C[i]));
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = A[i];
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = C[i];
    }

    // just directly cast, since transA, transB are integers in the enum
    //printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = hipblasLtHalf(C_float[i]);
}

template <>
void cblas_gemm<hipblasLtHalf, float, float>(hipblasOperation_t   transA,
                                             hipblasOperation_t   transB,
                                             int64_t              m,
                                             int64_t              n,
                                             int64_t              k,
                                             float                alpha,
                                             const hipblasLtHalf* A,
                                             int64_t              lda,
                                             const hipblasLtHalf* B,
                                             int64_t              ldb,
                                             float                beta,
                                             float*               C,
                                             int64_t              ldc,
                                             bool                 alt)
{
    // cblas does not support hipblasLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);

    host_vector<float> A_float(sizeA), B_float(sizeB);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = float_to_bfloat16_truncate(float(A[i]));
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = float_to_bfloat16_truncate(float(B[i]));
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = A[i];
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<float, float, float>(hipblasOperation_t transA,
                                     hipblasOperation_t transB,
                                     int64_t            m,
                                     int64_t            n,
                                     int64_t            k,
                                     float              alpha,
                                     const float*       A,
                                     int64_t            lda,
                                     const float*       B,
                                     int64_t            ldb,
                                     float              beta,
                                     float*             C,
                                     int64_t            ldc,
                                     bool               alt)
{

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

// AlphaVec gemm
template <>
void cblas_gemm_alphascale<hip_bfloat16, hip_bfloat16, float>(hipblasOperation_t  transA,
                                                              hipblasOperation_t  transB,
                                                              int64_t             m,
                                                              int64_t             n,
                                                              int64_t             k,
                                                              float               alpha,
                                                              const hip_bfloat16* A,
                                                              int64_t             lda,
                                                              const hip_bfloat16* B,
                                                              int64_t             ldb,
                                                              float               beta,
                                                              hip_bfloat16*       C,
                                                              int64_t             ldc,
                                                              const float*        AlphaVec,
                                                              bool                alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = static_cast<float>(A[i]);
        A_float[i] *= AlphaVec[i % m];
    }
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);
    for(size_t i = 0; i < sizeC; i++)
        C_float[i] = static_cast<float>(C[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = static_cast<hip_bfloat16>(C_float[i]);
}

template <>
void cblas_gemm_alphascale<hip_bfloat16, float, float>(hipblasOperation_t  transA,
                                                       hipblasOperation_t  transB,
                                                       int64_t             m,
                                                       int64_t             n,
                                                       int64_t             k,
                                                       float               alpha,
                                                       const hip_bfloat16* A,
                                                       int64_t             lda,
                                                       const hip_bfloat16* B,
                                                       int64_t             ldb,
                                                       float               beta,
                                                       float*              C,
                                                       int64_t             ldc,
                                                       const float*        AlphaVec,
                                                       bool                alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = static_cast<float>(A[i]);
        A_float[i] *= AlphaVec[i % m];
    }
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm_alphascale<hipblaslt_f8, float, float>(hipblasOperation_t  transA,
                                                       hipblasOperation_t  transB,
                                                       int64_t             m,
                                                       int64_t             n,
                                                       int64_t             k,
                                                       float               alpha,
                                                       const hipblaslt_f8* A,
                                                       int64_t             lda,
                                                       const hipblaslt_f8* B,
                                                       int64_t             ldb,
                                                       float               beta,
                                                       float*              C,
                                                       int64_t             ldc,
                                                       const float*        AlphaVec,
                                                       bool                alt)
{
    // cblas does not support hip_bfloat16, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = static_cast<float>(A[i]);
        A_float[i] *= AlphaVec[i % m];
    }
    for(size_t i = 0; i < sizeB; i++)
        B_float[i] = static_cast<float>(B[i]);

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm_alphascale<hipblasLtHalf, hipblasLtHalf, float>(hipblasOperation_t   transA,
                                                                hipblasOperation_t   transB,
                                                                int64_t              m,
                                                                int64_t              n,
                                                                int64_t              k,
                                                                float                alpha,
                                                                const hipblasLtHalf* A,
                                                                int64_t              lda,
                                                                const hipblasLtHalf* B,
                                                                int64_t              ldb,
                                                                float                beta,
                                                                hipblasLtHalf*       C,
                                                                int64_t              ldc,
                                                                const float*         AlphaVec,
                                                                bool                 alt)
{
    // cblas does not support hipblasLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);
    size_t sizeC = n * size_t(ldc);

    host_vector<float> A_float(sizeA), B_float(sizeB), C_float(sizeC);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = float_to_bfloat16_truncate(float(A[i]));
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = float_to_bfloat16_truncate(float(B[i]));
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = float_to_bfloat16_truncate(float(C[i]));
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
        {
            A_float[i] = A[i];
            A_float[i] *= AlphaVec[i % m];
        }
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
        for(size_t i = 0; i < sizeC; i++)
            C_float[i] = C[i];
    }

    // just directly cast, since transA, transB are integers in the enum
    //printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C_float,
                ldc);

    for(size_t i = 0; i < sizeC; i++)
        C[i] = hipblasLtHalf(C_float[i]);
}

template <>
void cblas_gemm_alphascale<hipblasLtHalf, float, float>(hipblasOperation_t   transA,
                                                        hipblasOperation_t   transB,
                                                        int64_t              m,
                                                        int64_t              n,
                                                        int64_t              k,
                                                        float                alpha,
                                                        const hipblasLtHalf* A,
                                                        int64_t              lda,
                                                        const hipblasLtHalf* B,
                                                        int64_t              ldb,
                                                        float                beta,
                                                        float*               C,
                                                        int64_t              ldc,
                                                        const float*         AlphaVec,
                                                        bool                 alt)
{
    // cblas does not support hipblasLtHalf, so convert to higher precision float
    // This will give more precise result which is acceptable for testing

    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);

    host_vector<float> A_float(sizeA), B_float(sizeB);

    if(alt)
    {
        for(size_t i = 0; i < sizeA; i++)
            A_float[i] = float_to_bfloat16_truncate(float(A[i]));
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = float_to_bfloat16_truncate(float(B[i]));
    }
    else
    {
        for(size_t i = 0; i < sizeA; i++)
        {
            A_float[i] = A[i];
            A_float[i] *= AlphaVec[i % m];
        }
        for(size_t i = 0; i < sizeB; i++)
            B_float[i] = B[i];
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A_float,
                lda,
                B_float,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm_alphascale<float, float, float>(hipblasOperation_t transA,
                                                hipblasOperation_t transB,
                                                int64_t            m,
                                                int64_t            n,
                                                int64_t            k,
                                                float              alpha,
                                                const float*       A,
                                                int64_t            lda,
                                                const float*       B,
                                                int64_t            ldb,
                                                float              beta,
                                                float*             C,
                                                int64_t            ldc,
                                                const float*       AlphaVec,
                                                bool               alt)
{
    size_t sizeA = (transA == HIPBLAS_OP_N ? k : m) * size_t(lda);
    size_t sizeB = (transB == HIPBLAS_OP_N ? n : k) * size_t(ldb);

    host_vector<float> A_float(sizeA);

    for(size_t i = 0; i < sizeA; i++)
    {
        A_float[i] = A[i];
        A_float[i] *= AlphaVec[i % m];
    }

    // just directly cast, since transA, transB are integers in the enum
    // printf("transA: hipblaslt =%d, cblas=%d\n", transA, HIPOperationToCBLASTanspose(transA) );
    cblas_sgemm(CblasColMajor,
                HIPOperationToCBLASTanspose(transA),
                HIPOperationToCBLASTanspose(transB),
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}
