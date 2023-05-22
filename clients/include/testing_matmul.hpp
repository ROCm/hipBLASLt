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

template <typename Ti,
          typename Tc,
          typename To,
          typename Tbias,
          typename Talpha,
          typename Tact,
          typename F>
void epilogue_func(int64_t m,
                   int64_t n,
                   int64_t ld,
                   Ti*     in,
                   To*     out,
                   Tc*     out_raw,
                   Tc*     e,
                   bool    enable_bias,
                   bool    enable_scaleD,
                   Talpha* scaleD,
                   Tbias*  bias,
                   Tact    arg1,
                   Tact    arg2,
                   F&      act_func,
                   bool    gradient)
{
    auto saturate_o = [](Tact val) { return static_cast<To>(val); };

    for(int i = 0; i < m; i++)
    {
        Ti bias_data   = enable_bias ? static_cast<Ti>(*(bias + i)) : 0;
        Ti scaleD_data = enable_scaleD ? static_cast<Ti>(*(scaleD + i)) : 1;
#pragma omp parallel for
        for(int j = 0; j < n; j++)
        {
            auto pos     = j * ld + i;
            auto in_Tact = static_cast<Tact>(*(in + pos)) + bias_data;
            if(e && !gradient)
            {
                *(e + pos) = static_cast<Tc>(in_Tact);
            }
            Tact in_Tact_act = 0;
            if(gradient)
                in_Tact_act = act_func(static_cast<Tact>(*(e + pos)), arg1, arg2) * in_Tact;
            else
                in_Tact_act = act_func(in_Tact, arg1, arg2);
            in_Tact_act *= scaleD_data;
            *(out + pos)     = saturate_o(in_Tact_act);
            *(out_raw + pos) = static_cast<Tc>(in_Tact_act);
        }
    }
}
template <typename Ti, typename Tc, typename To, typename Tbias, typename Talpha>
void epilogue_func(int64_t m,
                   int64_t n,
                   int64_t ld,
                   Ti*     in,
                   To*     out,
                   Tc*     out_raw,
                   Tc*     e,
                   bool    enable_bias,
                   bool    enable_scaleD,
                   Talpha* scaleD,
                   Tbias*  bias,
                   bool    gradient)
{
    auto saturate_o = [](Ti val) { return static_cast<To>(val); };

    for(int i = 0; i < m; i++)
    {
        Ti bias_data   = enable_bias ? static_cast<Ti>(*(bias + i)) : 0;
        Ti scaleD_data = enable_scaleD ? static_cast<Ti>(*(scaleD + i)) : 1;
#pragma omp parallel for
        for(int j = 0; j < n; j++)
        {
            auto pos  = j * ld + i;
            auto temp = static_cast<Ti>(*(in + pos)) + bias_data;
            if(e)
            {
                *(e + pos) = static_cast<Tc>(temp);
            }
            temp *= scaleD_data;
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

template <typename Ti, typename To, typename Talpha>
void scaleD_func(
    int64_t m, int64_t n, int64_t ld, Ti* in, To* out, bool enable_scaleD, Talpha* scaleD)
{
    auto saturate_o = [](Ti val) { return static_cast<To>(val); };

    for(int i = 0; i < m; i++)
    {
        Ti scaleD_data = enable_scaleD ? static_cast<Ti>(*(scaleD + i)) : 1;
#pragma omp parallel for
        for(int j = 0; j < n; j++)
        {
            auto pos     = j * ld + i;
            auto temp    = static_cast<Ti>(*(in + pos)) * scaleD_data;
            *(out + pos) = saturate_o(temp);
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

template <typename Ti, typename To, typename Tc>
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
    device_vector<Ti> dA(safe_size / 2);
    device_vector<Ti> dB(safe_size);
    device_vector<Ti> dC(safe_size);
    device_vector<Ti> dD(safe_size);
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

template <typename Ti, typename To, typename Tc>
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

    using Talpha = float;

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
        size_scaleD(gemm_count);

    std::vector<hipblasLtMatrixLayout_t> matA(gemm_count), matB(gemm_count), matC(gemm_count),
        matD(gemm_count);
    std::vector<hipblasLtMatmulDesc_t> matmul(gemm_count);
    std::vector<hipblasLtEpilogue_t>   epilogue(gemm_count, HIPBLASLT_EPILOGUE_DEFAULT);

    std::vector<device_vector<Ti>*>     dA(gemm_count), dB(gemm_count);
    std::vector<device_vector<To>*>     dC(gemm_count), dD(gemm_count), dBias(gemm_count);
    std::vector<device_vector<Talpha>*> dScaleD(gemm_count), dBias_C(gemm_count);
    std::vector<device_vector<Tc>*>     dE(gemm_count);

    std::vector<host_vector<Ti>*> hA(gemm_count), hB(gemm_count);
    std::vector<host_vector<To>*> hC(gemm_count), hD_gold(gemm_count), hD_1(gemm_count),
        hBias(gemm_count), hBias_gold(gemm_count);
    std::vector<host_vector<Talpha>*> hD_gold_epl(gemm_count), hScaleD(gemm_count),
        hBias_C(gemm_count), hBias_gold_C(gemm_count), hBias_gold_epl(gemm_count);
    std::vector<host_vector<Tc>*> hE(gemm_count, nullptr), hE_gold(gemm_count, nullptr);

    for(int i = 0; i < gemm_count; i++)
    {
        M[i]       = arg.M;
        N[i]       = arg.N;
        K[i]       = arg.K;
        h_alpha[i] = arg.get_alpha<Talpha>();
        h_beta[i]  = arg.get_beta<Talpha>();
        lda[i]     = arg.lda;
        ldb[i]     = arg.ldb;
        ldc[i]     = arg.ldc;
        ldd[i]     = arg.ldd;
        lde[i]     = arg.lde;

        A_row[i] = transA == HIPBLAS_OP_N ? M[i] : K[i];
        A_col[i] = transA == HIPBLAS_OP_N ? K[i] : M[i];
        B_row[i] = transB == HIPBLAS_OP_N ? K[i] : N[i];
        B_col[i] = transB == HIPBLAS_OP_N ? N[i] : K[i];

        do_batched[i]  = (arg.batch_count > 1);
        num_batches[i] = (do_batched[i] ? arg.batch_count : 1);

        stride_a[i] = do_batched[i] ? arg.stride_a : lda[i] * A_col[i];
        stride_b[i] = do_batched[i] ? arg.stride_b : ldb[i] * B_col[i];
        stride_c[i] = do_batched[i] ? arg.stride_c : ldc[i] * N[i];
        stride_d[i] = do_batched[i] ? arg.stride_c : ldd[i] * N[i];
        stride_e[i] = do_batched[i] ? arg.stride_e : lde[i] * N[i];

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

        if(arg.scaleD_vector)
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
        size_D_copy[i] = arg.unit_check || arg.norm_check ? size_D[i] : 0;
        size_scaleD[i] = arg.scaleD_vector ? M[i] : 1;
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
        dA[i]      = new device_vector<Ti>(size_A[i], 1, HMM);
        dB[i]      = new device_vector<Ti>(size_B[i], 1, HMM);
        dC[i]      = new device_vector<To>(size_C[i], 1, HMM);
        dD[i]      = new device_vector<To>(size_D[i], 1, HMM);
        dBias[i]   = new device_vector<To>(size_bias[i], 1, HMM);
        dScaleD[i] = new device_vector<Talpha>(size_scaleD[i], 1, HMM);
        dBias_C[i] = new device_vector<Talpha>(size_bias[i], 1, HMM);

        CHECK_DEVICE_ALLOCATION(dA[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dB[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dC[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dD[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dBias[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dScaleD[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dBias_C[i]->memcheck());
        if(arg.use_e)
        {
            dE[i] = new device_vector<Tc>(size_E[i], 1, HMM);
            CHECK_DEVICE_ALLOCATION(dE[i]->memcheck());
        }
        else
        {
            dE[i] = nullptr;
        }

        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
        hA[i]             = new host_vector<Ti>(size_A[i]);
        hB[i]             = new host_vector<Ti>(size_B[i]);
        hC[i]             = new host_vector<To>(size_C[i]);
        hD_gold[i]        = new host_vector<To>(size_D_copy[i]);
        hD_gold_epl[i]    = new host_vector<Talpha>(size_D_copy[i]);
        hD_1[i]           = new host_vector<To>(size_D_copy[i]);
        hBias[i]          = new host_vector<To>(size_bias[i]);
        hBias_gold_epl[i] = new host_vector<Talpha>(size_D_copy[i]); // Reduction for matrix D
        hBias_gold[i]     = new host_vector<To>(size_bias[i]);
        hBias_gold_C[i]   = new host_vector<Talpha>(size_bias[i]);
        hScaleD[i]        = new host_vector<Talpha>(size_scaleD[i]);
        hBias_C[i]        = new host_vector<Talpha>(size_bias[i]);

        if(arg.use_e)
        {
            hE[i] = new host_vector<Tc>(size_E[i]);
            if(!arg.gradient)
                hE_gold[i] = new host_vector<Tc>(size_E[i]);
        }

        hipblaslt_seedrand();

        // Initial Data on CPU
        if(arg.alpha_isnan<Tc>())
        {
            hipblaslt_init_nan<Ti>(*hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
            hipblaslt_init_nan<Ti>(*hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
        }
        else
        {
            if(arg.initialization == hipblaslt_initialization::rand_int)
            {
                hipblaslt_init<Ti>(*hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
                hipblaslt_init_alternating_sign<Ti>(
                    *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
            }
            else if(arg.initialization == hipblaslt_initialization::trig_float)
            {
                hipblaslt_init_sin<Ti>(
                    *hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
                hipblaslt_init_cos<Ti>(
                    *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
            }
            else if(arg.initialization == hipblaslt_initialization::hpl)
            {
                hipblaslt_init_hpl<Ti>(
                    *hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
                hipblaslt_init_hpl<Ti>(
                    *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
            }
            else if(arg.initialization == hipblaslt_initialization::special)
            {
                hipblaslt_init_alt_impl_big<Ti>(*hA[i], A_row[i], A_col[i], lda[i], num_batches[i]);
                hipblaslt_init_alt_impl_small<Ti>(
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
            hipblaslt_init<Tc>(*hE[i], M[i], N[i], lde[i], stride_e[i], num_batches[i]);
        }

        if(arg.bias_vector)
        {
            hipblaslt_init<To>(*hBias[i], M[i], 1, M[i]);
            hipblaslt_init<Talpha>(*hBias_C[i], M[i], 1, M[i]);
        }

        if(arg.scaleD_vector)
            hipblaslt_init<Talpha>(*hScaleD[i], M[i], 1, M[i]);

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

        if(arg.scaleD_vector)
            CHECK_HIP_ERROR(dScaleD[i]->transfer_from(*hScaleD[i]));

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
            if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
            {
                bias_addr           = *dBias_C[i];
                change_bias_type[i] = true;
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(matmul[i],
                                                    HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                    &arg.bias_type,
                                                    sizeof(hipblasDatatype_t)),
                    HIPBLAS_STATUS_SUCCESS);
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

        if(arg.scaleD_vector)
        {
            const void* scaleD_addr = *dScaleD[i];
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(
                    matmul[i], HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleD_addr, sizeof(void*)),
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
    device_vector<unsigned char>* dWorkspace;
    size_t                        workspace_size = 0;

    // Get Heuristic results
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult{1};
    int                                           requestAlgoCount  = 1;
    int                                           returnedAlgoCount = 0;

    // grouped gemm
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm<hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM> groupedGemm(handle);
    std::vector<void*> da(gemm_count), db(gemm_count), dc(gemm_count), dd(gemm_count);

    if(!do_grouped_gemm)
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

        for(int i = 0; i < returnedAlgoCount; i++)
            workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);
        dWorkspace = new device_vector<unsigned char>(workspace_size, 1, HMM);
        CHECK_DEVICE_ALLOCATION(dWorkspace->memcheck());
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
        CHECK_HIPBLASLT_ERROR(groupedGemm.setProblemFromhipBlasLt(
            matmul, h_alpha, da, matA, db, matB, h_beta, dc, matC, dd, matD));

        CHECK_HIPBLASLT_ERROR(
            groupedGemm.algoGetHeuristic(requestAlgoCount, gemmPref, heuristicResult));
        returnedAlgoCount = heuristicResult.size();

        dWorkspace = new device_vector<unsigned char>(max_workspace_size, 1, HMM);
        CHECK_DEVICE_ALLOCATION(dWorkspace->memcheck());
    }

    if(arg.unit_check || arg.norm_check)
    {
        if(!do_grouped_gemm)
        {
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                  matmul[0],
                                                  &(h_alpha[0]),
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
        else
        {
            //grouped gemm
            CHECK_HIPBLASLT_ERROR(
                groupedGemm.initialize(heuristicResult[0].algo, *dWorkspace, stream));

            CHECK_HIPBLASLT_ERROR(groupedGemm.run(stream));
        }
    }

    // get CPU result
    if(arg.unit_check || arg.norm_check)
    {
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

#define epilogue_param                                                                \
    M[gemmIdx], N[gemmIdx], ldd[gemmIdx], *(hD_gold_epl[gemmIdx]) + pos,              \
        *(hD_gold[gemmIdx]) + pos, *(hBias_gold_epl[gemmIdx]) + pos, ePos, applyBias, \
        arg.scaleD_vector, *(hScaleD[gemmIdx]) + 0
        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            for(int batchIdx = 0; batchIdx < num_batches[gemmIdx]; batchIdx++)
            {
                if(epilogue_on[gemmIdx])
                {
                    cblas_gemm<Ti, Talpha, Talpha>(transA,
                                                   transB,
                                                   M[gemmIdx],
                                                   N[gemmIdx],
                                                   K[gemmIdx],
                                                   h_alpha[gemmIdx],
                                                   *(hA[gemmIdx]) + stride_a[gemmIdx] * batchIdx,
                                                   lda[gemmIdx],
                                                   *(hB[gemmIdx]) + stride_b[gemmIdx] * batchIdx,
                                                   ldb[gemmIdx],
                                                   h_beta[gemmIdx],
                                                   *(hD_gold_epl[gemmIdx])
                                                       + stride_d[gemmIdx] * batchIdx,
                                                   ldd[gemmIdx],
                                                   false);
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
                            Ti*  ptr = nullptr;
                            if(arg.bias_source == hipblaslt_bias_source::a)
                            {
                                ptr   = *(hA[gemmIdx]);
                                s2    = lda[gemmIdx];
                                s3    = stride_a[gemmIdx];
                                sumLd = transA == HIPBLAS_OP_N ? false : true;
                            }
                            else if(arg.bias_source == hipblaslt_bias_source::b)
                            {
                                ptr   = *(hA[gemmIdx]);
                                s2    = ldb[gemmIdx];
                                s3    = stride_b[gemmIdx];
                                sumLd = transB == HIPBLAS_OP_N ? true : false;
                            }
                            if(sumLd)
                            {
                                if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
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
                                if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
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
                        }
                    }
                }
                else
                {
                    cblas_gemm<Ti, To, Talpha>(transA,
                                               transB,
                                               M[gemmIdx],
                                               N[gemmIdx],
                                               K[gemmIdx],
                                               h_alpha[gemmIdx],
                                               *(hA[gemmIdx]) + stride_a[gemmIdx] * batchIdx,
                                               lda[gemmIdx],
                                               *(hB[gemmIdx]) + stride_b[gemmIdx] * batchIdx,
                                               ldb[gemmIdx],
                                               h_beta[gemmIdx],
                                               *(hD_gold[gemmIdx]) + stride_d[gemmIdx] * batchIdx,
                                               ldd[gemmIdx],
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
                    unit_check_general<Tc>(M[gemmIdx],
                                           N[gemmIdx],
                                           lde[gemmIdx],
                                           stride_e[gemmIdx],
                                           *(hE_gold[gemmIdx]),
                                           *(hE[gemmIdx]),
                                           num_batches[gemmIdx]);
                if(arg.gradient && arg.bias_vector)
                {
                    if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
                        unit_check_general<Talpha>(M[gemmIdx],
                                                   1,
                                                   M[gemmIdx],
                                                   M[gemmIdx],
                                                   *(hBias_gold_C[gemmIdx]),
                                                   *(hBias_C[gemmIdx]),
                                                   num_batches[gemmIdx]);
                    else
                        unit_check_general<To>(M[gemmIdx],
                                               1,
                                               M[gemmIdx],
                                               M[gemmIdx],
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
                    double norm_error = std::abs(norm_check_general<Tc>('F',
                                                                        M[gemmIdx],
                                                                        N[gemmIdx],
                                                                        lde[gemmIdx],
                                                                        stride_e[gemmIdx],
                                                                        *(hE_gold[gemmIdx]),
                                                                        *(hE[gemmIdx]),
                                                                        num_batches[gemmIdx]));
                    hipblaslt_error += norm_error;
                    if(arg.norm_check_assert)
                        CHECK_SUCCESS(norm_check<Tc>(norm_error));
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
                            CHECK_SUCCESS(norm_check<Tc>(norm_error));
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
                            CHECK_SUCCESS(norm_check<Tc>(norm_error));
                    }
                }
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        if(!do_grouped_gemm)
        {
            for(int i = 0; i < number_cold_calls; i++)
            {
                EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                      matmul[0],
                                                      &(h_alpha[0]),
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

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            gpu_time_used = get_time_us_sync(stream); // in microseconds
            for(int i = 0; i < number_hot_calls; i++)
            {
                EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                      matmul[0],
                                                      &(h_alpha[0]),
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
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        }
        else
        {
            //grouped gemm
            CHECK_HIPBLASLT_ERROR(
                groupedGemm.initialize(heuristicResult[0].algo, *dWorkspace, stream));

            for(int i = 0; i < number_cold_calls; i++)
                CHECK_HIPBLASLT_ERROR(groupedGemm.run(stream));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            gpu_time_used = get_time_us_sync(stream); // in microseconds

            for(int i = 0; i < number_hot_calls; i++)
                CHECK_HIPBLASLT_ERROR(groupedGemm.run(stream));

            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
            gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        }

        double flops = 0;
        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            flops += gemm_gflop_count<float>(M[gemmIdx], N[gemmIdx], K[gemmIdx]);
            switch(arg.activation_type)
            {
            case hipblaslt_activation_type::relu:
                flops += relu_gflop_count<float>(M[gemmIdx], N[gemmIdx]);
                break;
            case hipblaslt_activation_type::gelu:
                flops += gelu_gflop_count<float>(M[gemmIdx], N[gemmIdx]);
                break;
            default:
                break;
            }
        }

#define argument_param                                                                             \
    e_transA, e_transB, e_grouped_gemm, e_batch_count, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a,  \
        e_beta, e_ldb, e_stride_b, e_ldc, e_stride_c, e_ldd, e_stride_d, e_d_type, e_compute_type, \
        e_activation_type, e_bias_vector

        ArgumentModel<argument_param>{}.log_args<float>(hipblaslt_cout,
                                                        arg,
                                                        gpu_time_used,
                                                        flops,
                                                        ArgumentLogging::NA_value,
                                                        cpu_time_used,
                                                        hipblaslt_error);
    }

    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
