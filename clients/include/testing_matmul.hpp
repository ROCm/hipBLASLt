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

#pragma once

#include "cblas_interface.hpp"
#include "flops.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include "hipblaslt_vector.hpp"
#include "near.hpp"
#include "norm.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <cstddef>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <omp.h>

template <typename Ti, typename Tc, typename To, typename Tbias, typename Tact, typename F>
void epilogue_func(int64_t m,
                   int64_t n,
                   int64_t ld,
                   Ti*     in,
                   To*     out,
                   Tc*     out_raw,
                   To*     e,
                   Tc      scaleD,
                   Tc      scaleE,
                   bool    enable_bias,
                   Tbias*  bias,
                   Tact    arg1,
                   Tact    arg2,
                   F&      act_func,
                   bool    gradient)
{
    auto saturate_o = [](Tact val) { return static_cast<To>(val); };

    for(int i = 0; i < m; i++)
    {
        Ti bias_data = enable_bias ? static_cast<Ti>(*(bias + i)) : 0;
#pragma omp parallel for
        for(int j = 0; j < n; j++)
        {
            auto pos     = j * ld + i;
            auto in_Tact = static_cast<Tact>(*(in + pos)) + bias_data;
            if(e && !gradient)
            {
                *(e + pos) = static_cast<To>(in_Tact * scaleE);
            }
            Tact in_Tact_act = 0;
            if(gradient)
                in_Tact_act
                    = act_func(static_cast<Tact>(*(e + pos)), arg1, arg2) * in_Tact * scaleD;
            else
                in_Tact_act = act_func(in_Tact, arg1, arg2) * scaleD;
            *(out + pos)     = saturate_o(in_Tact_act);
            *(out_raw + pos) = static_cast<Tc>(in_Tact_act);
        }
    }
}
template <typename Ti, typename Tc, typename To, typename Tbias>
void epilogue_func(int64_t m,
                   int64_t n,
                   int64_t ld,
                   Ti*     in,
                   To*     out,
                   Tc*     out_raw,
                   To*     e,
                   Tc      scaleD,
                   Tc      scaleE,
                   bool    enable_bias,
                   Tbias*  bias,
                   bool    gradient)
{
    auto saturate_o = [](Ti val) { return static_cast<To>(val); };

    for(int i = 0; i < m; i++)
    {
        Ti bias_data = enable_bias ? static_cast<Ti>(*(bias + i)) : 0;
#pragma omp parallel for
        for(int j = 0; j < n; j++)
        {
            auto pos  = j * ld + i;
            auto temp = static_cast<Ti>(*(in + pos)) + bias_data;
            if(e)
            {
                *(e + pos) = static_cast<To>(temp * scaleE);
            }
            temp *= scaleD;
            *(out + pos)     = saturate_o(temp);
            *(out_raw + pos) = static_cast<Tc>(temp);
        }
    }
}

template <bool SumLd, typename Tc, typename Ti, typename To>
void reduction_func(
    Ti* workspace, To* bias, int length, int k, int s1, int s2, int s3, int batch_count)
{
    assert(batch_count == 1);
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int i1 = 0; i1 < length; i1++)
        {
            Tc sum = 0;
            for(int i2 = 0; i2 < k; i2++)
            {
                if constexpr(SumLd)
                {
                    sum += static_cast<Tc>(workspace[i1 * s2 + i2 * s1 + batch * s3]);
                }
                else
                {
                    sum += static_cast<Tc>(workspace[i1 * s1 + i2 * s2 + batch * s3]);
                }
            }
            bias[i1] = static_cast<To>(sum);
        }
    }
}

auto _relu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    return static_cast<decltype(in)>(std::max(static_cast<decltype(in)>(0), in));
};

auto _gelu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.7978845608028654);
    constexpr auto k1    = static_cast<Tc>(0.044715);
    Tc             in_Tc = static_cast<Tc>(in);

    return static_cast<decltype(in)>(
        0.5f * (in_Tc * (1.f + std::tanh(k0 * (in_Tc * (1.f + k1 * (in_Tc * in_Tc)))))));
};

auto _dgelu = [](auto in, auto /*arg1*/, auto /*arg2*/) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.0535161);
    constexpr auto k1    = static_cast<Tc>(0.398942);
    constexpr auto k2    = static_cast<Tc>(0.0356774);
    constexpr auto k3    = static_cast<Tc>(0.797885);
    Tc             in_Tc = static_cast<Tc>(in);

    Tc pow3 = in_Tc * in_Tc * in_Tc;
    Tc x1   = k0 * pow3 + k1 * in_Tc;
    Tc xx   = k2 * pow3 + k3 * in_Tc;
    Tc x2   = 4 / pow(exp(-xx) + exp(xx), 2);
    Tc tmp  = 0.5 * tanh(xx) + x1 * x2 + 0.5;
    return static_cast<decltype(in)>(0.5f * tanh(xx) + x1 * x2 + 0.5f);
};

template <typename TiA, typename TiB, typename To, typename Tc, typename Tci>
void testing_matmul_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipblasOperation_t transA = HIPBLAS_OP_T;
    const hipblasOperation_t transB = HIPBLAS_OP_N;

    // allocate memory on device
    device_vector<TiA> dA(safe_size / 2);
    device_vector<TiB> dB(safe_size);
    device_vector<To>  dC(safe_size);
    device_vector<To>  dD(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());

    hipblaslt_local_handle        handle{arg};
    hipblaslt_local_matrix_layout matA(M, K, lda, arg.a_type);
    hipblaslt_local_matrix_layout matB(K, N, ldb, arg.b_type);
    hipblaslt_local_matrix_layout matC(M, N, ldc, arg.c_type);
    hipblaslt_local_matrix_layout matD(M, N, ldc, arg.d_type);
    hipblaslt_local_matmul_descr  matmul(transA, transB, arg.compute_type, arg.scale_type);

    size_t                     workspace_size = 0;
    hipblaslt_local_preference pref;

    void* workspace = nullptr;
    float alpha = 1.0, beta = 0.0;

    hipStream_t stream = nullptr;
}

template <typename TiA, typename TiB, typename To, typename Tc, typename Tci>
void testing_matmul(const Arguments& arg)
{
    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used          = 0.0;
    double                 hipblaslt_error = 0.0;
    bool                   HMM             = arg.HMM;
    hipblaslt_local_handle handle{arg};
    hipStream_t            stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    hipblasOperation_t transA(char_to_hipblas_operation(arg.transA));
    hipblasOperation_t transB(char_to_hipblas_operation(arg.transB));

    using Talpha = Tc;

    bool    do_grouped_gemm = arg.grouped_gemm > 0;
    int32_t gemm_count      = std::max(1, arg.grouped_gemm);

    std::vector<int64_t> M(gemm_count), N(gemm_count), K(gemm_count), lda(gemm_count),
        ldb(gemm_count), ldc(gemm_count), ldd(gemm_count), lde(gemm_count);
    std::vector<Talpha>  h_alpha(gemm_count), h_beta(gemm_count);
    std::vector<int64_t> A_row(gemm_count), A_col(gemm_count), B_row(gemm_count), B_col(gemm_count);
    std::vector<int64_t> stride_a(gemm_count), stride_b(gemm_count), stride_c(gemm_count),
        stride_d(gemm_count), stride_e(gemm_count);
    std::vector<bool> do_batched(gemm_count), epilogue_on(gemm_count, false),
        change_bias_type(gemm_count, false);
    std::vector<int>    num_batches(gemm_count);
    std::vector<size_t> size_A(gemm_count), size_B(gemm_count), size_C(gemm_count),
        size_D(gemm_count), size_D_copy(gemm_count), size_E(gemm_count), size_bias(gemm_count),
        size_scaleAlphaVec(gemm_count);

    std::vector<hipblasLtMatrixLayout_t> matA(gemm_count), matB(gemm_count), matC(gemm_count),
        matD(gemm_count);
    std::vector<hipblasLtMatmulDesc_t> matmul(gemm_count);
    std::vector<hipblasLtEpilogue_t>   epilogue(gemm_count, HIPBLASLT_EPILOGUE_DEFAULT);

    std::vector<device_vector<TiA>*>    dA(gemm_count);
    std::vector<device_vector<TiB>*>    dB(gemm_count);
    std::vector<device_vector<To>*>     dC(gemm_count), dD(gemm_count), dBias(gemm_count);
    std::vector<device_vector<Talpha>*> dScaleAlphaVec(gemm_count), dBias_C(gemm_count),
        dScaleA(gemm_count), dScaleB(gemm_count), dScaleC(gemm_count), dScaleD(gemm_count),
        dScaleE(gemm_count);
    std::vector<device_vector<To>*> dE(gemm_count);

    std::vector<host_vector<TiA>*> hA(gemm_count);
    std::vector<host_vector<TiB>*> hB(gemm_count);
    std::vector<host_vector<To>*>  hC(gemm_count), hD_gold(gemm_count), hD_1(gemm_count),
        hBias(gemm_count), hBias_gold(gemm_count);
    std::vector<host_vector<Talpha>*> hD_gold_epl(gemm_count), hScaleAlphaVec(gemm_count),
        hD_gold_ScaleAlpha(gemm_count), hBias_C(gemm_count), hBias_gold_C(gemm_count),
        hBias_gold_epl(gemm_count), hScaleA(gemm_count), hScaleB(gemm_count), hScaleC(gemm_count),
        hScaleD(gemm_count), hScaleE(gemm_count);
    std::vector<host_vector<To>*> hE(gemm_count, nullptr), hE_gold(gemm_count, nullptr);
    std::vector<void*>            alpha_in(gemm_count);

    for(int i = 0; i < gemm_count; i++)
    {
        M[i]       = arg.M[i];
        N[i]       = arg.N[i];
        K[i]       = arg.K[i];
        h_alpha[i] = arg.get_alpha<Talpha>();
        h_beta[i]  = arg.get_beta<Talpha>();
        lda[i]     = arg.lda[i];
        ldb[i]     = arg.ldb[i];
        ldc[i]     = arg.ldc[i];
        ldd[i]     = arg.ldd[i];
        lde[i]     = arg.lde[i];

        A_row[i] = transA == HIPBLAS_OP_N ? M[i] : K[i];
        A_col[i] = transA == HIPBLAS_OP_N ? K[i] : M[i];
        B_row[i] = transB == HIPBLAS_OP_N ? K[i] : N[i];
        B_col[i] = transB == HIPBLAS_OP_N ? N[i] : K[i];

        do_batched[i]  = (arg.batch_count > 1);
        num_batches[i] = (do_batched[i] ? arg.batch_count : 1);

        stride_a[i] = do_batched[i] ? arg.stride_a[i] : lda[i] * A_col[i];
        stride_b[i] = do_batched[i] ? arg.stride_b[i] : ldb[i] * B_col[i];
        stride_c[i] = do_batched[i] ? arg.stride_c[i] : ldc[i] * N[i];
        stride_d[i] = do_batched[i] ? arg.stride_c[i] : ldd[i] * N[i];
        stride_e[i] = do_batched[i] ? arg.stride_e[i] : lde[i] * N[i];

        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matA[i]), arg.a_type, A_row[i], A_col[i], lda[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matB[i]), arg.b_type, B_row[i], B_col[i], ldb[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matC[i]), arg.c_type, M[i], N[i], ldc[i]));
        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatrixLayoutCreate(&(matD[i]), arg.d_type, M[i], N[i], ldc[i]));

        if(do_batched[i])
        {
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matA[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matB[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matC[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(
                    matD[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &(num_batches[i]), sizeof(int)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matA[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_a[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matB[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_b[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matC[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_c[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatrixLayoutSetAttribute(matD[i],
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &(stride_d[i]),
                                                  sizeof(int64_t)),
                HIPBLAS_STATUS_SUCCESS);
        }

        CHECK_HIPBLASLT_ERROR(
            hipblasLtMatmulDescCreate(&(matmul[i]), arg.compute_type, arg.scale_type));

        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[i], HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[i], HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));

        if(arg.bias_vector)
        {
            epilogue_on[i] = true;
            switch(arg.activation_type)
            {
            case hipblaslt_activation_type::relu:
                epilogue[i] = HIPBLASLT_EPILOGUE_RELU_BIAS;
                break;
            case hipblaslt_activation_type::gelu:
                epilogue[i] = HIPBLASLT_EPILOGUE_GELU_BIAS;
                break;
            default:
                epilogue[i] = HIPBLASLT_EPILOGUE_BIAS;
                break;
            }
        }
        else
        {
            switch(arg.activation_type)
            {
            case hipblaslt_activation_type::relu:
                epilogue[i]    = HIPBLASLT_EPILOGUE_RELU;
                epilogue_on[i] = true;
                break;
            case hipblaslt_activation_type::gelu:
                epilogue[i]    = HIPBLASLT_EPILOGUE_GELU;
                epilogue_on[i] = true;
                break;
            default:
                break;
            }
        }
        if(arg.gradient)
        {
            switch(epilogue[i])
            {
            case HIPBLASLT_EPILOGUE_BIAS:
            {
                switch(arg.bias_source)
                {
                case hipblaslt_bias_source::a:
                    epilogue[i] = HIPBLASLT_EPILOGUE_BGRADA;
                    break;
                case hipblaslt_bias_source::b:
                    epilogue[i] = HIPBLASLT_EPILOGUE_BGRADB;
                    break;
                default:
                    break;
                }
            }
            break;
            case HIPBLASLT_EPILOGUE_GELU:
                CHECK_SUCCESS(arg.use_e && "Must enable use e if gradient is enabled with gelu.");
                epilogue[i] = HIPBLASLT_EPILOGUE_DGELU;
                break;
            case HIPBLASLT_EPILOGUE_GELU_BIAS:
                CHECK_SUCCESS(arg.use_e && "Must enable use e if gradient is enabled with gelu.");
                epilogue[i] = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
                break;
            default:
                break;
            }
        }
        if(arg.use_e)
        {
            switch(epilogue[i])
            {
            case HIPBLASLT_EPILOGUE_GELU:
                epilogue[i] = HIPBLASLT_EPILOGUE_GELU_AUX;
                break;
            case HIPBLASLT_EPILOGUE_GELU_BIAS:
                epilogue[i] = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
                break;
            default:
                break;
            }
        }

        if(arg.scaleAlpha_vector)
        {
            epilogue_on[i] = true;
        }

        size_A[i]
            = stride_a[i] == 0 ? lda[i] * A_col[i] * num_batches[i] : stride_a[i] * num_batches[i];
        size_B[i]
            = stride_b[i] == 0 ? ldb[i] * B_col[i] * num_batches[i] : stride_b[i] * num_batches[i];
        size_C[i]
            = stride_c[i] == 0 ? ldc[i] * N[i] * num_batches[i] : stride_c[i] * num_batches[i];
        size_D[i]
            = stride_d[i] == 0 ? ldd[i] * N[i] * num_batches[i] : stride_d[i] * num_batches[i];
        size_E[i]
            = stride_e[i] == 0 ? lde[i] * N[i] * num_batches[i] : stride_e[i] * num_batches[i];
        size_D_copy[i]        = arg.unit_check || arg.norm_check ? size_D[i] : 0;
        size_scaleAlphaVec[i] = arg.scaleAlpha_vector ? M[i] : 1;
        if(arg.bias_vector)
        {
            if(arg.bias_source == hipblaslt_bias_source::a
               || arg.bias_source == hipblaslt_bias_source::d)
                size_bias[i] = M[i];
            else if(arg.bias_source == hipblaslt_bias_source::b)
                size_bias[i] = N[i];
        }
        else
        {
            size_bias[i] = 1;
        }

        // allocate memory on device
        dA[i]             = new device_vector<TiA>(size_A[i], 1, HMM);
        dB[i]             = new device_vector<TiB>(size_B[i], 1, HMM);
        dC[i]             = new device_vector<To>(size_C[i], 1, HMM);
        dD[i]             = new device_vector<To>(size_D[i], 1, HMM);
        dBias[i]          = new device_vector<To>(size_bias[i], 1, HMM);
        dScaleAlphaVec[i] = new device_vector<Talpha>(size_scaleAlphaVec[i], 1, HMM);
        dBias_C[i]        = new device_vector<Talpha>(size_bias[i], 1, HMM);

        CHECK_DEVICE_ALLOCATION(dA[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dB[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dC[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dD[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dBias[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dScaleAlphaVec[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dBias_C[i]->memcheck());
        if(arg.scaleA)
        {
            dScaleA[i] = new device_vector<Talpha>(1, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dScaleA[i]->memcheck());
        }
        if(arg.scaleB)
        {
            dScaleB[i] = new device_vector<Talpha>(1, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dScaleB[i]->memcheck());
        }
        if(arg.scaleC)
        {
            dScaleC[i] = new device_vector<Talpha>(1, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dScaleC[i]->memcheck());
        }
        if(arg.scaleD)
        {
            dScaleD[i] = new device_vector<Talpha>(1, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dScaleD[i]->memcheck());
        }
        if(arg.scaleE)
        {
            dScaleB[i] = new device_vector<Talpha>(1, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dScaleE[i]->memcheck());
        }

        if(arg.use_e)
        {
            dE[i] = new device_vector<To>(size_E[i], 1, HMM);
            CHECK_DEVICE_ALLOCATION(dE[i]->memcheck());
        }
        else
        {
            dE[i] = nullptr;
        }

        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
        hA[i]                 = new host_vector<TiA>(size_A[i]);
        hB[i]                 = new host_vector<TiB>(size_B[i]);
        hC[i]                 = new host_vector<To>(size_C[i]);
        hD_gold[i]            = new host_vector<To>(size_D_copy[i]);
        hD_gold_epl[i]        = new host_vector<Talpha>(size_D_copy[i]);
        hD_gold_ScaleAlpha[i] = new host_vector<Talpha>(size_D_copy[i]);
        hD_1[i]               = new host_vector<To>(size_D_copy[i]);
        hBias[i]              = new host_vector<To>(size_bias[i]);
        hBias_gold_epl[i]     = new host_vector<Talpha>(size_D_copy[i]); // Reduction for matrix D
        hBias_gold[i]         = new host_vector<To>(size_bias[i]);
        hBias_gold_C[i]       = new host_vector<Talpha>(size_bias[i]);
        hScaleAlphaVec[i]     = new host_vector<Talpha>(size_scaleAlphaVec[i]);
        hBias_C[i]            = new host_vector<Talpha>(size_bias[i]);

        if(arg.scaleA)
            hScaleA[i] = new host_vector<Talpha>(1);
        if(arg.scaleB)
            hScaleB[i] = new host_vector<Talpha>(1);
        if(arg.scaleC)
            hScaleC[i] = new host_vector<Talpha>(1);
        if(arg.scaleD)
            hScaleD[i] = new host_vector<Talpha>(1);
        if(arg.scaleE)
            hScaleE[i] = new host_vector<Talpha>(1);

        if(arg.use_e)
        {
            hE[i] = new host_vector<To>(size_E[i]);
            if(!arg.gradient)
                hE_gold[i] = new host_vector<To>(size_E[i]);
        }

        hipblaslt_seedrand();

        // Initial Data on CPU
        if(arg.alpha_isnan<Tc>())
        {
            hipblaslt_init_nan<TiA>(
                *hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
            hipblaslt_init_nan<TiB>(
                *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
        }
        else
        {
            if(arg.initialization == hipblaslt_initialization::rand_int)
            {
                hipblaslt_init<TiA>(
                    *hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
                hipblaslt_init_alternating_sign<TiB>(
                    *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
            }
            else if(arg.initialization == hipblaslt_initialization::trig_float)
            {
                hipblaslt_init_sin<TiA>(
                    *hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
                hipblaslt_init_cos<TiB>(
                    *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
            }
            else if(arg.initialization == hipblaslt_initialization::hpl)
            {
                hipblaslt_init_hpl<TiA>(
                    *hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
                hipblaslt_init_hpl<TiB>(
                    *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
            }
            else if(arg.initialization == hipblaslt_initialization::special)
            {
                hipblaslt_init_alt_impl_big<TiA>(
                    *hA[i], A_row[i], A_col[i], lda[i], num_batches[i]);
                hipblaslt_init_alt_impl_small<TiB>(
                    *hB[i], B_row[i], B_col[i], ldb[i], num_batches[i]);
            }
        }

        if(arg.beta_isnan<Tc>())
        {
            hipblaslt_init_nan<To>(*hC[i], M[i], N[i], ldc[i], stride_c[i], num_batches[i]);
        }
        else
        {
            if(arg.initialization == hipblaslt_initialization::rand_int)
                hipblaslt_init<To>(*hC[i], M[i], N[i], ldc[i], stride_c[i], num_batches[i]);
            else if(arg.initialization == hipblaslt_initialization::trig_float)
                hipblaslt_init_sin<To>(*hC[i], M[i], N[i], ldc[i], stride_c[i], num_batches[i]);
            else if(arg.initialization == hipblaslt_initialization::hpl)
                hipblaslt_init_hpl<To>(*hC[i], M[i], N[i], ldc[i], stride_c[i], num_batches[i]);
            else if(arg.initialization == hipblaslt_initialization::special)
                hipblaslt_init<To>(*hC[i], M[i], N[i], ldc[i], stride_c[i], num_batches[i]);
        }

        if(arg.gradient && arg.use_e)
        {
            hipblaslt_init<To>(*hE[i], M[i], N[i], lde[i], stride_e[i], num_batches[i]);
        }

        if(arg.bias_vector)
        {
            hipblaslt_init<To>(*hBias[i], M[i], 1, M[i]);
            hipblaslt_init<Talpha>(*hBias_C[i], M[i], 1, M[i]);
        }

        if(arg.scaleA)
            hipblaslt_init<Talpha>(*hScaleA[i], 1, 1, 1);

        if(arg.scaleB)
            hipblaslt_init<Talpha>(*hScaleB[i], 1, 1, 1);

        if(arg.scaleC)
        {
            if constexpr(std::is_same<To, hipblaslt_f8_fnuz>::value
                         || std::is_same<To, hipblaslt_bf8_fnuz>::value)
            {
                hipblaslt_init_small<Talpha>(*hScaleC[i], 1, 1, 1);
            }
            else
            {
                hipblaslt_init<Talpha>(*hScaleC[i], 1, 1, 1);
            }
        }

        if(arg.scaleD)
        {
            if constexpr(std::is_same<To, hipblaslt_f8_fnuz>::value
                         || std::is_same<To, hipblaslt_bf8_fnuz>::value)
            {
                hipblaslt_init_small<Talpha>(*hScaleD[i], 1, 1, 1);
            }
            else
            {
                hipblaslt_init<Talpha>(*hScaleD[i], 1, 1, 1);
            }
        }

        if(arg.scaleE)
            hipblaslt_init<Talpha>(*hScaleE[i], 1, 1, 1);

        if(arg.scaleAlpha_vector)
            hipblaslt_init<Talpha>(*hScaleAlphaVec[i], M[i], 1, M[i]);

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA[i]->transfer_from(*hA[i]));
        CHECK_HIP_ERROR(dB[i]->transfer_from(*hB[i]));
        CHECK_HIP_ERROR(dC[i]->transfer_from(*hC[i]));
        if(arg.gradient && arg.use_e)
        {
            CHECK_HIP_ERROR(dE[i]->transfer_from(*hE[i]));
        }
        if(!arg.gradient && arg.bias_vector)
        {
            CHECK_HIP_ERROR(dBias[i]->transfer_from(*hBias[i]));
            CHECK_HIP_ERROR(dBias_C[i]->transfer_from(*hBias_C[i]));
        }

        if(arg.scaleA)
            CHECK_HIP_ERROR(dScaleA[i]->transfer_from(*hScaleA[i]));

        if(arg.scaleB)
            CHECK_HIP_ERROR(dScaleB[i]->transfer_from(*hScaleB[i]));

        if(arg.scaleC)
            CHECK_HIP_ERROR(dScaleC[i]->transfer_from(*hScaleC[i]));

        if(arg.scaleD)
            CHECK_HIP_ERROR(dScaleD[i]->transfer_from(*hScaleD[i]));

        if(arg.scaleE)
            CHECK_HIP_ERROR(dScaleE[i]->transfer_from(*hScaleE[i]));

        if(arg.scaleAlpha_vector)
        {
            CHECK_HIP_ERROR(dScaleAlphaVec[i]->transfer_from(*hScaleAlphaVec[i]));
            alpha_in[i] = *(dScaleAlphaVec[i]);
            h_alpha[i]  = 1.0; // use dScaleAlphaVec instead, original alpha = 1.0 for verify
        }
        else
            alpha_in[i] = &(h_alpha[i]);

        if(size_D_copy[i])
        {
            if(epilogue_on[i])
            {
                std::transform(hC[i]->begin(),
                               hC[i]->end(),
                               hD_gold_epl[i]->begin(),
                               [](To c) -> Talpha { return static_cast<Talpha>(c); });
            }
            else
            {
                std::copy(hC[i]->begin(), hC[i]->end(), hD_gold[i]->begin());
            }
        }

        if(epilogue_on[i])
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(
                    matmul[i], HIPBLASLT_MATMUL_DESC_EPILOGUE, &(epilogue[i]), sizeof(epilogue[i])),
                HIPBLAS_STATUS_SUCCESS);

        if(arg.use_e)
        {
            void* e_addr = *dE[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[i], HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &e_addr, sizeof(void*)));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[i], HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &lde[i], sizeof(int64_t)));
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[i],
                                                HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                                &stride_e[i],
                                                sizeof(int64_t)));
        }

        if(arg.bias_vector)
        {
            const void* bias_addr;
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[i],
                                                HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &arg.bias_type,
                                                sizeof(hipDataType)),
                HIPBLAS_STATUS_SUCCESS);
            if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
            {
                bias_addr           = *dBias_C[i];
                change_bias_type[i] = true;
            }
            else
            {
                bias_addr = *dBias[i];
            }

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(
                    matmul[i], HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_addr, sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);
        }

        if(arg.scaleA)
        {
            void* scaleA_addr = *dScaleA[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[i], HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA_addr, sizeof(void*)));
        }

        if(arg.scaleB)
        {
            void* scaleB_addr = *dScaleB[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[i], HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB_addr, sizeof(void*)));
        }

        if(arg.scaleC)
        {
            void* scaleC_addr = *dScaleC[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[i], HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &scaleC_addr, sizeof(void*)));
        }

        if(arg.scaleD)
        {
            void* scaleD_addr = *dScaleD[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[i], HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleD_addr, sizeof(void*)));
        }

        if(arg.scaleE)
        {
            void* scaleE_addr = *dScaleE[i];
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[i],
                                                HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
                                                &scaleE_addr,
                                                sizeof(void*)));
        }

        if(arg.scaleAlpha_vector)
        {
            hipblasLtPointerMode_t scale_mode
                = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(
                    matmul[i], HIPBLASLT_MATMUL_DESC_POINTER_MODE, &scale_mode, sizeof(scale_mode)),
                HIPBLAS_STATUS_SUCCESS);
        }
    }

    // set preference
    size_t                     max_workspace_size = 32 * 1024 * 1024;
    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)),
        HIPBLAS_STATUS_SUCCESS);

    // set workspace
    device_vector<unsigned char>* dWorkspace     = nullptr;
    size_t                        workspace_size = 0;

    // set user args
    hipblaslt_ext::UserArguments* userArgs   = nullptr;
    hipblaslt_ext::UserArguments* d_userArgs = nullptr;

    // Get Heuristic results
    int32_t requestAlgoCount  = arg.requested_solution_num < 0 ? std::numeric_limits<int32_t>::max()
                                                               : arg.requested_solution_num;
    int     returnedAlgoCount = 0;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(requestAlgoCount);

    // grouped gemm
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(
        handle, transA, transB, arg.a_type, arg.b_type, arg.c_type, arg.d_type, arg.compute_type);
    hipblaslt_ext::GroupedGemm groupedGemm(
        handle, transA, transB, arg.a_type, arg.b_type, arg.c_type, arg.d_type, arg.compute_type);
    std::vector<void*> da(gemm_count), db(gemm_count), dc(gemm_count), dd(gemm_count);
    std::vector<hipblaslt_ext::GemmEpilogue> extepilogue;
    std::vector<hipblaslt_ext::GemmInputs>   extinputs;
    hipblaslt_ext::GemmProblemType           extproblemtype;
    if(arg.use_ext_setproblem)
    {
        extepilogue.resize(gemm_count);
        extinputs.resize(gemm_count);

        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            auto  bias_type = HIPBLASLT_DATATYPE_INVALID;
            void* bias_addr = nullptr;
            if(arg.bias_vector)
            {
                bias_type = arg.bias_type;
                if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
                {
                    bias_addr = (void*)*dBias_C[gemmIdx];
                }
                else
                {
                    bias_addr = (void*)*dBias[gemmIdx];
                }
            }
            extepilogue[gemmIdx].mode           = epilogue[gemmIdx];
            extepilogue[gemmIdx].bias_data_type = bias_type;
            extepilogue[gemmIdx].aux_ld         = lde[gemmIdx];
            extepilogue[gemmIdx].aux_stride     = stride_e[gemmIdx];

            extinputs[gemmIdx].a        = *dA[gemmIdx];
            extinputs[gemmIdx].b        = *dB[gemmIdx];
            extinputs[gemmIdx].c        = *dC[gemmIdx];
            extinputs[gemmIdx].d        = *dD[gemmIdx];
            extinputs[gemmIdx].alpha    = &h_alpha[gemmIdx];
            extinputs[gemmIdx].beta     = &h_beta[gemmIdx];
            extinputs[gemmIdx].bias     = bias_addr;
            extinputs[gemmIdx].scaleA   = arg.scaleA ? *dScaleA[gemmIdx] : nullptr;
            extinputs[gemmIdx].scaleB   = arg.scaleB ? *dScaleB[gemmIdx] : nullptr;
            extinputs[gemmIdx].scaleC   = arg.scaleC ? *dScaleC[gemmIdx] : nullptr;
            extinputs[gemmIdx].scaleD   = arg.scaleD ? *dScaleD[gemmIdx] : nullptr;
            extinputs[gemmIdx].scaleAux = arg.scaleE ? *dScaleE[gemmIdx] : nullptr;
            if(arg.scaleAlpha_vector)
                extinputs[gemmIdx].scaleAlphaVec = *dScaleAlphaVec[gemmIdx];
        }
        extproblemtype.op_a         = transA;
        extproblemtype.op_b         = transB;
        extproblemtype.type_a       = arg.a_type;
        extproblemtype.type_b       = arg.b_type;
        extproblemtype.type_c       = arg.c_type;
        extproblemtype.type_d       = arg.d_type;
        extproblemtype.type_compute = arg.compute_type;
    }

    hipblaslt_ext::GemmType gemmType = do_grouped_gemm
                                           ? hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM
                                           : hipblaslt_ext::GemmType::HIPBLASLT_GEMM;

    if(arg.algo_method == 2)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
        heuristicResult.clear();

        int algoIndexCount = 0;
        int algoIndexInc   = 100;
        while(1)
        {
            // Get algos by index
            // In real cases, the user can use the saved algo index to get the algorithm.
            // isAlgoSupported is not necessary if the user is sure that the algo supports the problem.
            std::vector<int> algoIndex(algoIndexInc);
            std::iota(std::begin(algoIndex), std::end(algoIndex), algoIndexCount);
            algoIndexCount += algoIndexInc;
            std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
            CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpAlgo));
            returnedAlgoCount = tmpAlgo.size();

            bool foundAlgo = false;

            if(!do_grouped_gemm)
            {
                if(arg.use_ext)
                {
                    if(arg.use_ext_setproblem)
                    {
                        CHECK_HIPBLASLT_ERROR(gemm.setProblem(M[0],
                                                              N[0],
                                                              K[0],
                                                              num_batches[0],
                                                              lda[0],
                                                              ldb[0],
                                                              ldc[0],
                                                              ldd[0],
                                                              stride_a[0],
                                                              stride_b[0],
                                                              stride_c[0],
                                                              stride_d[0],
                                                              extepilogue[0],
                                                              extinputs[0],
                                                              extproblemtype));
                    }
                    else
                    {
                        CHECK_HIPBLASLT_ERROR(gemm.setProblem(matmul[0],
                                                              alpha_in[0],
                                                              *(dA[0]),
                                                              matA[0],
                                                              *(dB[0]),
                                                              matB[0],
                                                              &h_beta[0],
                                                              *(dC[0]),
                                                              matC[0],
                                                              *(dD[0]),
                                                              matD[0]));
                    }
                    for(int j = 0; j < returnedAlgoCount; j++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(gemm.isAlgoSupported(tmpAlgo[j].algo, tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            heuristicResult.push_back(tmpAlgo[j]);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                            foundAlgo      = true;
                            break;
                        }
                    }
                }
                else
                {
                    for(int j = 0; j < returnedAlgoCount; j++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                                matmul[0],
                                                                alpha_in[0],
                                                                matA[0],
                                                                matB[0],
                                                                &h_beta[0],
                                                                matC[0],
                                                                matD[0],
                                                                tmpAlgo[j].algo,
                                                                tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            heuristicResult.push_back(tmpAlgo[j]);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                            foundAlgo      = true;
                            break;
                        }
                    }
                }
            }
            else
            {
                if(arg.use_ext_setproblem)
                {
                    auto num_batches_64
                        = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
                    CHECK_HIPBLASLT_ERROR(groupedGemm.setProblem(M,
                                                                 N,
                                                                 K,
                                                                 num_batches_64,
                                                                 lda,
                                                                 ldb,
                                                                 ldc,
                                                                 ldd,
                                                                 stride_a,
                                                                 stride_b,
                                                                 stride_c,
                                                                 stride_d,
                                                                 extepilogue,
                                                                 extinputs,
                                                                 extproblemtype));
                }
                else
                {
                    for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
                    {
                        da[gemmIdx] = *dA[gemmIdx];
                        db[gemmIdx] = *dB[gemmIdx];
                        dc[gemmIdx] = *dC[gemmIdx];
                        dd[gemmIdx] = *dD[gemmIdx];
                    }

                    std::vector<void*> h_alpha_void, h_beta_void;
                    for(size_t i = 0; i < h_alpha.size(); i++)
                    {
                        h_alpha_void.push_back(&h_alpha[i]);
                        h_beta_void.push_back(&h_beta[i]);
                    }

                    CHECK_HIPBLASLT_ERROR(groupedGemm.setProblem(
                        matmul, h_alpha_void, da, matA, db, matB, h_beta_void, dc, matC, dd, matD));
                }

                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    size_t tmpWorkspaceSize = 0;
                    if(groupedGemm.isAlgoSupported(tmpAlgo[j].algo, tmpWorkspaceSize)
                       == HIPBLAS_STATUS_SUCCESS)
                    {
                        heuristicResult.push_back(tmpAlgo[j]);
                        workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                        foundAlgo      = true;
                        break;
                    }
                }
            }

            if(foundAlgo || (tmpAlgo.size() == 0))
            {
                break;
            }
        }
    }
    else if(arg.algo_method == 1)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
        EXPECT_HIPBLAS_STATUS(hipblaslt_ext::getAllAlgos(handle,
                                                         gemmType,
                                                         transA,
                                                         transB,
                                                         arg.a_type,
                                                         arg.b_type,
                                                         arg.c_type,
                                                         arg.d_type,
                                                         arg.compute_type,
                                                         tmpAlgo),
                              HIPBLAS_STATUS_SUCCESS);
        returnedAlgoCount = tmpAlgo.size();
        heuristicResult.clear();
        int requestCount = 0;
        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                if(arg.use_ext_setproblem)
                {
                    CHECK_HIPBLASLT_ERROR(gemm.setProblem(M[0],
                                                          N[0],
                                                          K[0],
                                                          num_batches[0],
                                                          lda[0],
                                                          ldb[0],
                                                          ldc[0],
                                                          ldd[0],
                                                          stride_a[0],
                                                          stride_b[0],
                                                          stride_c[0],
                                                          stride_d[0],
                                                          extepilogue[0],
                                                          extinputs[0],
                                                          extproblemtype));
                }
                else
                {
                    CHECK_HIPBLASLT_ERROR(gemm.setProblem(matmul[0],
                                                          alpha_in[0],
                                                          *(dA[0]),
                                                          matA[0],
                                                          *(dB[0]),
                                                          matB[0],
                                                          &h_beta[0],
                                                          *(dC[0]),
                                                          matC[0],
                                                          *(dD[0]),
                                                          matD[0]));
                }
                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    size_t tmpWorkspaceSize = 0;
                    if(gemm.isAlgoSupported(tmpAlgo[j].algo, tmpWorkspaceSize)
                       == HIPBLAS_STATUS_SUCCESS)
                    {
                        requestCount++;
                        heuristicResult.push_back(tmpAlgo[j]);
                        workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                    }
                    if(requestCount >= requestAlgoCount)
                    {
                        break;
                    }
                }
            }
            else
            {
                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    size_t tmpWorkspaceSize = 0;
                    if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                            matmul[0],
                                                            alpha_in[0],
                                                            matA[0],
                                                            matB[0],
                                                            &h_beta[0],
                                                            matC[0],
                                                            matD[0],
                                                            tmpAlgo[j].algo,
                                                            tmpWorkspaceSize)
                       == HIPBLAS_STATUS_SUCCESS)
                    {
                        requestCount++;
                        heuristicResult.push_back(tmpAlgo[j]);
                        workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                    }
                    if(requestCount >= requestAlgoCount)
                    {
                        break;
                    }
                }
            }
        }
        else
        {
            if(arg.use_ext_setproblem)
            {
                auto num_batches_64 = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
                CHECK_HIPBLASLT_ERROR(groupedGemm.setProblem(M,
                                                             N,
                                                             K,
                                                             num_batches_64,
                                                             lda,
                                                             ldb,
                                                             ldc,
                                                             ldd,
                                                             stride_a,
                                                             stride_b,
                                                             stride_c,
                                                             stride_d,
                                                             extepilogue,
                                                             extinputs,
                                                             extproblemtype));
            }
            else
            {
                for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
                {
                    da[gemmIdx] = *dA[gemmIdx];
                    db[gemmIdx] = *dB[gemmIdx];
                    dc[gemmIdx] = *dC[gemmIdx];
                    dd[gemmIdx] = *dD[gemmIdx];
                }

                std::vector<void*> h_alpha_void, h_beta_void;
                for(size_t i = 0; i < h_alpha.size(); i++)
                {
                    h_alpha_void.push_back(&h_alpha[i]);
                    h_beta_void.push_back(&h_beta[i]);
                }

                CHECK_HIPBLASLT_ERROR(groupedGemm.setProblem(
                    matmul, h_alpha_void, da, matA, db, matB, h_beta_void, dc, matC, dd, matD));
            }

            for(int j = 0; j < returnedAlgoCount; j++)
            {
                size_t tmpWorkspaceSize = 0;
                if(groupedGemm.isAlgoSupported(tmpAlgo[j].algo, tmpWorkspaceSize)
                   == HIPBLAS_STATUS_SUCCESS)
                {
                    requestCount++;
                    heuristicResult.push_back(tmpAlgo[j]);
                    workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                }
                if(requestCount >= requestAlgoCount)
                {
                    break;
                }
            }
        }
    }
    else
    {
        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                if(arg.use_ext_setproblem)
                {
                    CHECK_HIPBLASLT_ERROR(gemm.setProblem(M[0],
                                                          N[0],
                                                          K[0],
                                                          num_batches[0],
                                                          lda[0],
                                                          ldb[0],
                                                          ldc[0],
                                                          ldd[0],
                                                          stride_a[0],
                                                          stride_b[0],
                                                          stride_c[0],
                                                          stride_d[0],
                                                          extepilogue[0],
                                                          extinputs[0],
                                                          extproblemtype));
                }
                else
                {
                    CHECK_HIPBLASLT_ERROR(gemm.setProblem(matmul[0],
                                                          alpha_in[0],
                                                          *(dA[0]),
                                                          matA[0],
                                                          *(dB[0]),
                                                          matB[0],
                                                          &h_beta[0],
                                                          *(dC[0]),
                                                          matC[0],
                                                          *(dD[0]),
                                                          matD[0]));
                }
                CHECK_HIPBLASLT_ERROR(
                    gemm.algoGetHeuristic(requestAlgoCount, gemmPref, heuristicResult));
                returnedAlgoCount = heuristicResult.size();
            }
            else
            {
                EXPECT_HIPBLAS_STATUS((hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                       matmul[0],
                                                                       matA[0],
                                                                       matB[0],
                                                                       matC[0],
                                                                       matD[0],
                                                                       pref,
                                                                       requestAlgoCount,
                                                                       heuristicResult.data(),
                                                                       &returnedAlgoCount)),
                                      HIPBLAS_STATUS_SUCCESS);
            }

            for(int i = 0; i < returnedAlgoCount; i++)
                workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);
        }
        else
        {
            if(arg.use_ext_setproblem)
            {
                auto num_batches_64 = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
                CHECK_HIPBLASLT_ERROR(groupedGemm.setProblem(M,
                                                             N,
                                                             K,
                                                             num_batches_64,
                                                             lda,
                                                             ldb,
                                                             ldc,
                                                             ldd,
                                                             stride_a,
                                                             stride_b,
                                                             stride_c,
                                                             stride_d,
                                                             extepilogue,
                                                             extinputs,
                                                             extproblemtype));
            }
            else
            {
                // grouped gemm
                for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
                {
                    da[gemmIdx] = *dA[gemmIdx];
                    db[gemmIdx] = *dB[gemmIdx];
                    dc[gemmIdx] = *dC[gemmIdx];
                    dd[gemmIdx] = *dD[gemmIdx];
                }

                std::vector<void*> h_alpha_void, h_beta_void;
                for(size_t i = 0; i < h_alpha.size(); i++)
                {
                    h_alpha_void.push_back(&h_alpha[i]);
                    h_beta_void.push_back(&h_beta[i]);
                }

                CHECK_HIPBLASLT_ERROR(groupedGemm.setProblem(
                    matmul, h_alpha_void, da, matA, db, matB, h_beta_void, dc, matC, dd, matD));
            }

            CHECK_HIPBLASLT_ERROR(
                groupedGemm.algoGetHeuristic(requestAlgoCount, gemmPref, heuristicResult));
            returnedAlgoCount = heuristicResult.size();

            workspace_size = max_workspace_size;
        }
    }

    dWorkspace = new device_vector<unsigned char>(workspace_size, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dWorkspace->memcheck());

    if(arg.print_solution_found)
        hipblaslt_cout << "Is supported " << heuristicResult.size()
                       << " / Total solutions: " << returnedAlgoCount << std::endl;
    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    if(arg.unit_check || arg.norm_check)
    {
        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, *dWorkspace));

                CHECK_HIPBLASLT_ERROR(gemm.run(stream));
            }
            else
            {
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                      matmul[0],
                                                      alpha_in[0],
                                                      *(dA[0]),
                                                      matA[0],
                                                      *(dB[0]),
                                                      matB[0],
                                                      &(h_beta[0]),
                                                      *(dC[0]),
                                                      matC[0],
                                                      *(dD[0]),
                                                      matD[0],
                                                      &heuristicResult[0].algo,
                                                      *dWorkspace,
                                                      workspace_size,
                                                      stream),
                                      HIPBLAS_STATUS_SUCCESS);
            }
        }
        else
        {
            if(arg.use_user_args)
            {
                //grouped gemm
                CHECK_HIPBLASLT_ERROR(groupedGemm.initialize(heuristicResult[0].algo, *dWorkspace));
                CHECK_HIP_ERROR(
                    hipHostMalloc(&userArgs, gemm_count * sizeof(hipblaslt_ext::UserArguments)));
                groupedGemm.getDefaultValueForDeviceUserArguments(userArgs);
                // Copy them to device memory
                CHECK_HIP_ERROR(
                    hipMalloc(&d_userArgs, gemm_count * sizeof(hipblaslt_ext::UserArguments)));
                CHECK_HIP_ERROR(hipMemcpy(d_userArgs,
                                          userArgs,
                                          gemm_count * sizeof(hipblaslt_ext::UserArguments),
                                          hipMemcpyHostToDevice));

                CHECK_HIPBLASLT_ERROR(groupedGemm.run(d_userArgs, stream));
            }
            else
            {
                //grouped gemm
                CHECK_HIPBLASLT_ERROR(
                    groupedGemm.initialize(heuristicResult[0].algo, *dWorkspace, false, stream));

                CHECK_HIPBLASLT_ERROR(groupedGemm.run(stream));
            }
        }
    }

    // get CPU result
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        // For the xf32 xdl math op, cast type of A/B from float to xfloat32 .
        if constexpr(std::is_same<TiA, float>{} && std::is_same<TiB, float>{}
                     && std::is_same<To, float>{} && std::is_same<Tc, float>{})
            if(arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
            {
                for(int i = 0; i < gemm_count; i++)
                {
                    type_to_xdl_math_op_type<hipblasLtXfloat32, float>(hA[i]->data(), size_A[i]);
                    type_to_xdl_math_op_type<hipblasLtXfloat32, float>(hB[i]->data(), size_B[i]);
                }
            }

#define epilogue_param                                                                  \
    M[gemmIdx], N[gemmIdx], ldd[gemmIdx], *(hD_gold_epl[gemmIdx]) + pos,                \
        *(hD_gold[gemmIdx]) + pos, *(hBias_gold_epl[gemmIdx]) + pos, ePos, scaleDValue, \
        scaleEValue, applyBias
        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            auto alpha    = h_alpha[gemmIdx];
            auto betaTemp = h_beta[gemmIdx];
            if(arg.scaleC)
                betaTemp *= (*hScaleC[gemmIdx])[0];
            auto scaleAValue = arg.scaleA ? (*hScaleA[gemmIdx])[0] : 1;
            auto scaleBValue = arg.scaleB ? (*hScaleB[gemmIdx])[0] : 1;
            auto scaleDValue = arg.scaleD ? (*hScaleD[gemmIdx])[0] : 1;
            auto scaleEValue = arg.scaleE ? (*hScaleE[gemmIdx])[0] : 1;

            for(int batchIdx = 0; batchIdx < num_batches[gemmIdx]; batchIdx++)
            {
                if(epilogue_on[gemmIdx])
                {
                    if(arg.scaleAlpha_vector)
                    {
                        cblas_gemm<TiA, TiB, Talpha, Talpha, Tci>(
                            transA,
                            transB,
                            M[gemmIdx],
                            N[gemmIdx],
                            K[gemmIdx],
                            alpha,
                            *(hA[gemmIdx]) + stride_a[gemmIdx] * batchIdx,
                            lda[gemmIdx],
                            *(hB[gemmIdx]) + stride_b[gemmIdx] * batchIdx,
                            ldb[gemmIdx],
                            betaTemp,
                            *(hD_gold_epl[gemmIdx]) + stride_d[gemmIdx] * batchIdx,
                            ldd[gemmIdx],
                            *(hScaleAlphaVec[gemmIdx]) + 0,
                            scaleAValue,
                            scaleBValue,
                            1,
                            false);
                    }
                    else
                    {
                        cblas_gemm<TiA, TiB, Talpha, Talpha, Tci>(
                            transA,
                            transB,
                            M[gemmIdx],
                            N[gemmIdx],
                            K[gemmIdx],
                            alpha,
                            *(hA[gemmIdx]) + stride_a[gemmIdx] * batchIdx,
                            lda[gemmIdx],
                            *(hB[gemmIdx]) + stride_b[gemmIdx] * batchIdx,
                            ldb[gemmIdx],
                            betaTemp,
                            *(hD_gold_epl[gemmIdx]) + stride_d[gemmIdx] * batchIdx,
                            ldd[gemmIdx],
                            nullptr,
                            scaleAValue,
                            scaleBValue,
                            1,
                            false);
                    }
                    auto pos    = stride_d[gemmIdx] * batchIdx;
                    auto hEInst = arg.gradient ? hE : hE_gold;
                    auto ePos = (hEInst[gemmIdx] == nullptr) ? nullptr : (*(hEInst[gemmIdx]) + pos);
                    auto applyBias = arg.gradient ? false : arg.bias_vector;

                    if(change_bias_type[gemmIdx] == false)
                    {
                        switch(arg.activation_type)
                        {
                        case hipblaslt_activation_type::gelu:
                            if(arg.gradient)
                                epilogue_func(epilogue_param,
                                              *(hBias[gemmIdx]) + 0,
                                              arg.activation_arg1,
                                              arg.activation_arg2,
                                              ::_dgelu,
                                              true);
                            else
                                epilogue_func(epilogue_param,
                                              *(hBias[gemmIdx]) + 0,
                                              arg.activation_arg1,
                                              arg.activation_arg2,
                                              ::_gelu,
                                              false);
                            break;
                        case hipblaslt_activation_type::relu:
                            epilogue_func(epilogue_param,
                                          *(hBias[gemmIdx]) + 0,
                                          arg.activation_arg1,
                                          arg.activation_arg2,
                                          ::_relu,
                                          arg.gradient);
                            break;
                        default:
                            epilogue_func(epilogue_param, *(hBias[gemmIdx]) + 0, false);
                            break;
                        }
                    }
                    else
                    {
                        switch(arg.activation_type)
                        {
                        case hipblaslt_activation_type::gelu:
                            if(arg.gradient)
                                epilogue_func(epilogue_param,
                                              *(hBias_C[gemmIdx]) + 0,
                                              arg.activation_arg1,
                                              arg.activation_arg2,
                                              ::_dgelu,
                                              true);
                            else
                                epilogue_func(epilogue_param,
                                              *(hBias_C[gemmIdx]) + 0,
                                              arg.activation_arg1,
                                              arg.activation_arg2,
                                              ::_gelu,
                                              false);
                            break;
                        case hipblaslt_activation_type::relu:
                            epilogue_func(epilogue_param,
                                          *(hBias_C[gemmIdx]) + 0,
                                          arg.activation_arg1,
                                          arg.activation_arg2,
                                          ::_relu,
                                          arg.gradient);
                            break;
                        default:
                        {
                            epilogue_func(epilogue_param, *(hBias_C[gemmIdx]) + 0, false);
                        }
                        break;
                        }
                    }
                    if(arg.gradient && arg.bias_vector && batchIdx == num_batches[gemmIdx] - 1)
                    {
                        if(arg.bias_source == hipblaslt_bias_source::d)
                        {
                            if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
                                reduction_func<false, float>(*(hBias_gold_epl[gemmIdx]) + pos,
                                                             *(hBias_gold_C[gemmIdx]) + 0,
                                                             M[gemmIdx],
                                                             N[gemmIdx],
                                                             1,
                                                             ldd[gemmIdx],
                                                             stride_d[gemmIdx],
                                                             num_batches[gemmIdx]);
                            else
                                reduction_func<false, float>(*(hBias_gold_epl[gemmIdx]) + pos,
                                                             *(hBias_gold[gemmIdx]) + 0,
                                                             M[gemmIdx],
                                                             N[gemmIdx],
                                                             1,
                                                             ldd[gemmIdx],
                                                             stride_d[gemmIdx],
                                                             num_batches[gemmIdx]);
                        }
                        else
                        {
                            // *(hA[gemmIdx]) + stride_a[gemmIdx] * batchIdx
                            bool sumLd = false;
                            int  s1 = 1, s2 = 1, s3 = 1;

                            auto reduc = [&sumLd,
                                          &s1,
                                          &s2,
                                          &s3,
                                          &hBias_gold_C,
                                          &hBias_gold,
                                          &size_bias,
                                          &K,
                                          &num_batches,
                                          &gemmIdx,
                                          &arg]<typename Ti>(Ti* ptr) {
                                if(sumLd)
                                {
                                    if(arg.d_type != arg.scale_type
                                       && arg.bias_type == arg.scale_type)
                                        reduction_func<true, float>(ptr,
                                                                    *(hBias_gold_C[gemmIdx]) + 0,
                                                                    size_bias[gemmIdx],
                                                                    K[gemmIdx],
                                                                    s1,
                                                                    s2,
                                                                    s3,
                                                                    num_batches[gemmIdx]);
                                    else
                                        reduction_func<true, float>(ptr,
                                                                    *(hBias_gold[gemmIdx]) + 0,
                                                                    size_bias[gemmIdx],
                                                                    K[gemmIdx],
                                                                    s1,
                                                                    s2,
                                                                    s3,
                                                                    num_batches[gemmIdx]);
                                }
                                else
                                {
                                    if(arg.d_type != arg.scale_type
                                       && arg.bias_type == arg.scale_type)
                                        reduction_func<false, float>(ptr,
                                                                     *(hBias_gold_C[gemmIdx]) + 0,
                                                                     size_bias[gemmIdx],
                                                                     K[gemmIdx],
                                                                     s1,
                                                                     s2,
                                                                     s3,
                                                                     num_batches[gemmIdx]);
                                    else
                                        reduction_func<false, float>(ptr,
                                                                     *(hBias_gold[gemmIdx]) + 0,
                                                                     size_bias[gemmIdx],
                                                                     K[gemmIdx],
                                                                     s1,
                                                                     s2,
                                                                     s3,
                                                                     num_batches[gemmIdx]);
                                }
                            };

                            if(arg.bias_source == hipblaslt_bias_source::a)
                            {
                                TiA* ptr = *(hA[gemmIdx]);
                                s2       = lda[gemmIdx];
                                s3       = stride_a[gemmIdx];
                                sumLd    = transA == HIPBLAS_OP_N ? false : true;
                                reduc(ptr);
                            }
                            else if(arg.bias_source == hipblaslt_bias_source::b)
                            {
                                TiB* ptr = *(hB[gemmIdx]);
                                s2       = ldb[gemmIdx];
                                s3       = stride_b[gemmIdx];
                                sumLd    = transB == HIPBLAS_OP_N ? true : false;
                                reduc(ptr);
                            }
                        }
                    }
                }
                else
                {
                    cblas_gemm<TiA, TiB, To, Talpha, Tci>(
                        transA,
                        transB,
                        M[gemmIdx],
                        N[gemmIdx],
                        K[gemmIdx],
                        alpha,
                        *(hA[gemmIdx]) + stride_a[gemmIdx] * batchIdx,
                        lda[gemmIdx],
                        *(hB[gemmIdx]) + stride_b[gemmIdx] * batchIdx,
                        ldb[gemmIdx],
                        betaTemp,
                        *(hD_gold[gemmIdx]) + stride_d[gemmIdx] * batchIdx,
                        ldd[gemmIdx],
                        nullptr,
                        scaleAValue,
                        scaleBValue,
                        scaleDValue,
                        false);
                }
            }
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // fetch GPU
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            CHECK_HIP_ERROR(hD_1[gemmIdx]->transfer_from(*(dD[gemmIdx])));
            if(!arg.gradient && arg.use_e)
                CHECK_HIP_ERROR(hE[gemmIdx]->transfer_from(*(dE[gemmIdx])));
            if(arg.gradient && arg.bias_vector)
            {
                CHECK_HIP_ERROR(hBias[gemmIdx]->transfer_from(*(dBias[gemmIdx])));
                CHECK_HIP_ERROR(hBias_C[gemmIdx]->transfer_from(*(dBias_C[gemmIdx])));
            }
            if(arg.unit_check)
            {
                unit_check_general<To>(M[gemmIdx],
                                       N[gemmIdx],
                                       ldd[gemmIdx],
                                       stride_d[gemmIdx],
                                       *(hD_gold[gemmIdx]),
                                       *(hD_1[gemmIdx]),
                                       num_batches[gemmIdx]);
                if(!arg.gradient && arg.use_e)
                    unit_check_general<To>(M[gemmIdx],
                                           N[gemmIdx],
                                           lde[gemmIdx],
                                           stride_e[gemmIdx],
                                           *(hE_gold[gemmIdx]),
                                           *(hE[gemmIdx]),
                                           num_batches[gemmIdx]);
                if(arg.gradient && arg.bias_vector)
                {
                    if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
                        unit_check_general<Talpha>(size_bias[gemmIdx],
                                                   1,
                                                   size_bias[gemmIdx],
                                                   size_bias[gemmIdx],
                                                   *(hBias_gold_C[gemmIdx]),
                                                   *(hBias_C[gemmIdx]),
                                                   num_batches[gemmIdx]);

                    else
                        unit_check_general<To>(size_bias[gemmIdx],
                                               1,
                                               size_bias[gemmIdx],
                                               size_bias[gemmIdx],
                                               *(hBias_gold[gemmIdx]),
                                               *(hBias[gemmIdx]),
                                               num_batches[gemmIdx]);
                }
            }

            if(arg.norm_check)
            {
                double norm_error = std::abs(norm_check_general<To>('F',
                                                                    M[gemmIdx],
                                                                    N[gemmIdx],
                                                                    ldd[gemmIdx],
                                                                    stride_d[gemmIdx],
                                                                    *(hD_gold[gemmIdx]),
                                                                    *(hD_1[gemmIdx]),
                                                                    num_batches[gemmIdx]));
                hipblaslt_error += norm_error;
                if(arg.norm_check_assert)
                    CHECK_SUCCESS(norm_check<To>(norm_error));

                if(!arg.gradient && arg.use_e)
                {
                    double norm_error = std::abs(norm_check_general<To>('F',
                                                                        M[gemmIdx],
                                                                        N[gemmIdx],
                                                                        lde[gemmIdx],
                                                                        stride_e[gemmIdx],
                                                                        *(hE_gold[gemmIdx]),
                                                                        *(hE[gemmIdx]),
                                                                        num_batches[gemmIdx]));
                    hipblaslt_error += norm_error;
                    if(arg.norm_check_assert)
                        CHECK_SUCCESS(norm_check<To>(norm_error));
                }
                if(arg.gradient && arg.bias_vector)
                {
                    if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
                    {
                        double norm_error
                            = std::abs(norm_check_general<Talpha>('F',
                                                                  M[gemmIdx],
                                                                  1,
                                                                  M[gemmIdx],
                                                                  M[gemmIdx],
                                                                  *(hBias_gold_C[gemmIdx]),
                                                                  *(hBias_C[gemmIdx]),
                                                                  num_batches[gemmIdx]));
                        hipblaslt_error += norm_error;
                        if(arg.norm_check_assert)
                            CHECK_SUCCESS(norm_check<Talpha>(norm_error));
                    }
                    else
                    {
                        double norm_error = std::abs(norm_check_general<To>('F',
                                                                            M[gemmIdx],
                                                                            1,
                                                                            M[gemmIdx],
                                                                            M[gemmIdx],
                                                                            *(hBias_gold[gemmIdx]),
                                                                            *(hBias[gemmIdx]),
                                                                            num_batches[gemmIdx]));
                        hipblaslt_error += norm_error;
                        if(arg.norm_check_assert)
                            CHECK_SUCCESS(norm_check<To>(norm_error));
                    }
                }
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(size_t sol = 0; sol < heuristicResult.size(); sol++)
        {
            if(!do_grouped_gemm)
            {

                if(arg.use_ext)
                {
                    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[sol].algo, *dWorkspace));
                    for(int i = 0; i < number_cold_calls; i++)
                        CHECK_HIPBLASLT_ERROR(gemm.run(stream));
                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    gpu_time_used = get_time_us_sync(stream); // in microseconds

                    for(int i = 0; i < number_hot_calls; i++)
                        CHECK_HIPBLASLT_ERROR(gemm.run(stream));
                }
                else
                {
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                              matmul[0],
                                                              alpha_in[0],
                                                              *(dA[0]),
                                                              matA[0],
                                                              *(dB[0]),
                                                              matB[0],
                                                              &(h_beta[0]),
                                                              *(dC[0]),
                                                              matC[0],
                                                              *(dD[0]),
                                                              matD[0],
                                                              &heuristicResult[sol].algo,
                                                              *dWorkspace,
                                                              workspace_size,
                                                              stream),
                                              HIPBLAS_STATUS_SUCCESS);
                    }

                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    gpu_time_used = get_time_us_sync(stream); // in microseconds
                    for(int i = 0; i < number_hot_calls; i++)
                    {
                        EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                              matmul[0],
                                                              alpha_in[0],
                                                              *(dA[0]),
                                                              matA[0],
                                                              *(dB[0]),
                                                              matB[0],
                                                              &(h_beta[0]),
                                                              *(dC[0]),
                                                              matC[0],
                                                              *(dD[0]),
                                                              matD[0],
                                                              &heuristicResult[sol].algo,
                                                              *dWorkspace,
                                                              workspace_size,
                                                              stream),
                                              HIPBLAS_STATUS_SUCCESS);
                    }
                }
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
            }
            else
            {
                if(arg.use_user_args)
                {
                    //grouped gemm
                    CHECK_HIPBLASLT_ERROR(
                        groupedGemm.initialize(heuristicResult[sol].algo, *dWorkspace));
                    if(userArgs != nullptr)
                        CHECK_HIP_ERROR(hipHostMalloc(
                            &userArgs, gemm_count * sizeof(hipblaslt_ext::UserArguments)));
                    groupedGemm.getDefaultValueForDeviceUserArguments(userArgs);
                    // Copy them to device memory
                    if(d_userArgs != nullptr)
                        CHECK_HIP_ERROR(hipMalloc(
                            &d_userArgs, gemm_count * sizeof(hipblaslt_ext::UserArguments)));
                    CHECK_HIP_ERROR(hipMemcpy(d_userArgs,
                                              userArgs,
                                              gemm_count * sizeof(hipblaslt_ext::UserArguments),
                                              hipMemcpyHostToDevice));

                    for(int i = 0; i < number_cold_calls; i++)
                        CHECK_HIPBLASLT_ERROR(groupedGemm.run(d_userArgs, stream));

                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    gpu_time_used = get_time_us_sync(stream); // in microseconds

                    for(int i = 0; i < number_hot_calls; i++)
                        CHECK_HIPBLASLT_ERROR(groupedGemm.run(d_userArgs, stream));

                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
                }
                else
                {
                    //grouped gemm
                    CHECK_HIPBLASLT_ERROR(groupedGemm.initialize(
                        heuristicResult[sol].algo, *dWorkspace, false, stream));

                    for(int i = 0; i < number_cold_calls; i++)
                        CHECK_HIPBLASLT_ERROR(groupedGemm.run(stream));

                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    gpu_time_used = get_time_us_sync(stream); // in microseconds

                    for(int i = 0; i < number_hot_calls; i++)
                        CHECK_HIPBLASLT_ERROR(groupedGemm.run(stream));

                    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                    gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
                }
            }

            double flops = 0;
            for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
            {
                flops += gemm_gflop_count<Tc>(M[gemmIdx], N[gemmIdx], K[gemmIdx]);
                switch(arg.activation_type)
                {
                case hipblaslt_activation_type::relu:
                    flops += relu_gflop_count<Tc>(M[gemmIdx], N[gemmIdx]);
                    break;
                case hipblaslt_activation_type::gelu:
                    flops += gelu_gflop_count<Tc>(M[gemmIdx], N[gemmIdx]);
                    break;
                default:
                    break;
                }
            }

#define argument_param                                                                             \
    e_transA, e_transB, e_grouped_gemm, e_batch_count, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a,  \
        e_beta, e_ldb, e_stride_b, e_ldc, e_stride_c, e_ldd, e_stride_d, e_d_type, e_compute_type, \
        e_activation_type, e_bias_vector

            ArgumentModel<argument_param>{}.log_args<Tc>(hipblaslt_cout,
                                                         sol,
                                                         arg,
                                                         gpu_time_used,
                                                         flops,
                                                         ArgumentLogging::NA_value,
                                                         cpu_time_used,
                                                         hipblaslt_error);
        }
    }

    if(dWorkspace != nullptr)
        delete dWorkspace;
    if(userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(userArgs));
    if(d_userArgs != nullptr)
        CHECK_HIP_ERROR(hipFree(d_userArgs));

    for(int i = 0; i < gemm_count; i++)
    {
        delete hA[i];
        delete hB[i];
        delete hC[i];
        delete hD_gold[i];
        delete hD_gold_epl[i];
        delete hD_gold_ScaleAlpha[i];
        delete hD_1[i];
        delete hBias[i];
        delete hBias_gold_epl[i];
        delete hBias_gold[i];
        delete hBias_gold_C[i];
        delete hScaleAlphaVec[i];
        delete hBias_C[i];
        delete dA[i];
        delete dB[i];
        delete dC[i];
        delete dD[i];
        delete dBias[i];
        delete dScaleAlphaVec[i];
        delete dBias_C[i];
        if(arg.scaleA)
        {
            delete hScaleA[i];
            delete dScaleA[i];
        }
        if(arg.scaleB)
        {
            delete hScaleB[i];
            delete dScaleB[i];
        }
        if(arg.scaleC)
        {
            delete hScaleC[i];
            delete dScaleC[i];
        }
        if(arg.scaleD)
        {
            delete hScaleD[i];
            delete dScaleD[i];
        }
        if(arg.scaleE)
        {
            delete hScaleE[i];
            delete dScaleE[i];
        }
        if(arg.use_e)
        {
            delete dE[i];
            delete hE[i];
        }
    }

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
