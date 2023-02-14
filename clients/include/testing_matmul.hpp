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
#include <hipblaslt/hipblaslt.h>
#include <omp.h>

template <typename Ti, typename To, typename Tbias, typename Talpha, typename Tact, typename F>
void epilogue_func(int64_t m,
                   int64_t n,
                   int64_t ld,
                   Ti*     in,
                   To*     out,
                   bool    enable_bias,
                   bool    enable_scaleD,
                   Talpha* scaleD,
                   Tbias*  bias,
                   Tact    arg1,
                   Tact    arg2,
                   F&      act_func)
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
            in_Tact *= scaleD_data;
            *(out + pos) = saturate_o(act_func(in_Tact, arg1, arg2));
        }
    }
}
template <typename Ti, typename To, typename Tbias, typename Talpha>
void epilogue_func(int64_t m,
                   int64_t n,
                   int64_t ld,
                   Ti*     in,
                   To*     out,
                   bool    enable_bias,
                   bool    enable_scaleD,
                   Talpha* scaleD,
                   Tbias*  bias)
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
            temp *= scaleD_data;
            *(out + pos) = saturate_o(temp);
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
    hipblasOperation_t transA = char_to_hipblas_operation(arg.transA);
    hipblasOperation_t transB = char_to_hipblas_operation(arg.transB);

    using Talpha = float;

    int64_t M       = arg.M;
    int64_t N       = arg.N;
    int64_t K       = arg.K;
    Talpha  h_alpha = arg.get_alpha<Talpha>();
    Talpha  h_beta  = arg.get_beta<Talpha>();
    int64_t lda     = arg.lda;
    int64_t ldb     = arg.ldb;
    int64_t ldc     = arg.ldc;
    int64_t ldd     = arg.ldd;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used          = 0.0;
    double                 hipblaslt_error = 0.0;
    bool                   HMM             = arg.HMM;
    hipblaslt_local_handle handle{arg};
    hipStream_t            stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    int64_t A_row = transA == HIPBLAS_OP_N ? M : K;
    int64_t A_col = transA == HIPBLAS_OP_N ? K : M;
    int64_t B_row = transB == HIPBLAS_OP_N ? K : N;
    int64_t B_col = transB == HIPBLAS_OP_N ? N : K;

    int64_t stride_1_a = transA == HIPBLAS_OP_N ? 1 : lda;
    int64_t stride_2_a = transA == HIPBLAS_OP_N ? lda : 1;

    bool    do_batched  = (arg.batch_count > 1);
    int     num_batches = (do_batched ? arg.batch_count : 1);
    int64_t stride_a    = do_batched ? arg.stride_a : lda * A_col;
    int64_t stride_b    = do_batched ? arg.stride_b : ldb * B_col;
    int64_t stride_c    = do_batched ? arg.stride_c : ldc * N;
    int64_t stride_d    = do_batched ? arg.stride_c : ldd * N;

    hipblaslt_local_matrix_layout matA(A_row, A_col, lda, arg.a_type);
    hipblaslt_local_matrix_layout matB(B_row, B_col, ldb, arg.b_type);
    hipblaslt_local_matrix_layout matC(M, N, ldc, arg.c_type);
    hipblaslt_local_matrix_layout matD(M, N, ldc, arg.d_type);

    if(do_batched)
    {
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_batches, sizeof(int)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_batches, sizeof(int)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_batches, sizeof(int)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_batches, sizeof(int)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(int64_t)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(int64_t)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(int64_t)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatrixLayoutSetAttribute(
                matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(int64_t)),
            HIPBLAS_STATUS_SUCCESS);
    }

    hipblaslt_local_matmul_descr matmul(transA, transB, arg.compute_type, arg.scale_type);
    bool                         epilogue_on = false;
    hipblasLtEpilogue_t          epilogue    = HIPBLASLT_EPILOGUE_DEFAULT;
    if(arg.bias_vector)
    {
        epilogue_on = true;
        switch(arg.activation_type)
        {
        case hipblaslt_activation_type::relu:
            epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
            break;
        case hipblaslt_activation_type::gelu:
            epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
            break;
        default:
            epilogue = HIPBLASLT_EPILOGUE_BIAS;
            break;
        }
    }
    else
    {
        switch(arg.activation_type)
        {
        case hipblaslt_activation_type::relu:
            epilogue    = HIPBLASLT_EPILOGUE_RELU;
            epilogue_on = true;
            break;
        case hipblaslt_activation_type::gelu:
            epilogue    = HIPBLASLT_EPILOGUE_GELU;
            epilogue_on = true;
            break;
        default:
            break;
        }
    }

    size_t max_workspace_size = 32 * 1024 * 1024;

    hipblaslt_local_preference pref;
    EXPECT_HIPBLAS_STATUS(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)),
        HIPBLAS_STATUS_SUCCESS);

    const size_t size_A = stride_a == 0 ? lda * A_col * num_batches : stride_a * num_batches;

    const size_t size_B      = stride_b == 0 ? ldb * B_col * num_batches : stride_b * num_batches;
    const size_t size_C      = stride_c == 0 ? ldc * N * num_batches : stride_c * num_batches;
    const size_t size_D      = stride_d == 0 ? ldd * N * num_batches : stride_d * num_batches;
    const size_t size_D_copy = arg.unit_check || arg.norm_check ? size_D : 0;
    const size_t size_bias   = arg.bias_vector ? M : 1;
    const size_t size_scaleD = arg.scaleD_vector ? M : 1;

    // allocate memory on device
    device_vector<Ti>     dA(size_A, 1, HMM);
    device_vector<Ti>     dB(size_B, 1, HMM);
    device_vector<To>     dC(size_C, 1, HMM);
    device_vector<To>     dD(size_D, 1, HMM);
    device_vector<To>     dBias(size_bias, 1, HMM);
    device_vector<Talpha> dScaleD(size_scaleD, 1, HMM);
    device_vector<Talpha> dBias_C(size_bias, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    CHECK_DEVICE_ALLOCATION(dBias.memcheck());
    CHECK_DEVICE_ALLOCATION(dScaleD.memcheck());
    CHECK_DEVICE_ALLOCATION(dBias_C.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti>     hA(size_A);
    host_vector<Ti>     hB(size_B);
    host_vector<To>     hC(size_C);
    host_vector<To>     hD_gold(size_D_copy);
    host_vector<Talpha> hD_gold_epl(size_D_copy);
    host_vector<To>     hD_1(size_D_copy);
    host_vector<To>     hBias(size_bias);
    host_vector<Talpha> hScaleD(size_scaleD);
    host_vector<Talpha> hBias_C(size_bias);

    hipblaslt_seedrand();

    // Initial Data on CPU
    if(arg.alpha_isnan<Tc>())
    {
        hipblaslt_init_nan<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
        hipblaslt_init_nan<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
    }
    else
    {
        if(arg.initialization == hipblaslt_initialization::rand_int)
        {
            hipblaslt_init<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipblaslt_init_alternating_sign<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipblaslt_initialization::trig_float)
        {
            hipblaslt_init_sin<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipblaslt_init_cos<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipblaslt_initialization::hpl)
        {
            hipblaslt_init_hpl<Ti>(hA, A_row, A_col, lda, stride_a, num_batches);
            hipblaslt_init_hpl<Ti>(hB, B_row, B_col, ldb, stride_b, num_batches);
        }
        else if(arg.initialization == hipblaslt_initialization::special)
        {
            hipblaslt_init_alt_impl_big<Ti>(hA, A_row, A_col, lda, num_batches);
            hipblaslt_init_alt_impl_small<Ti>(hB, B_row, B_col, ldb, num_batches);
        }
    }

    if(arg.beta_isnan<Tc>())
    {
        hipblaslt_init_nan<To>(hC, M, N, ldc, stride_c, num_batches);
    }
    else
    {
        if(arg.initialization == hipblaslt_initialization::rand_int)
            hipblaslt_init<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipblaslt_initialization::trig_float)
            hipblaslt_init_sin<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipblaslt_initialization::hpl)
            hipblaslt_init_hpl<To>(hC, M, N, ldc, stride_c, num_batches);
        else if(arg.initialization == hipblaslt_initialization::special)
            hipblaslt_init<To>(hC, M, N, ldc, stride_c, num_batches);
    }

    if(arg.bias_vector)
    {
        hipblaslt_init<To>(hBias, M, 1, M);
        hipblaslt_init<Talpha>(hBias_C, M, 1, M);
    }

    if(arg.scaleD_vector)
        hipblaslt_init<Talpha>(hScaleD, M, 1, M);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));
    if(arg.bias_vector)
    {
        CHECK_HIP_ERROR(dBias.transfer_from(hBias));
        CHECK_HIP_ERROR(dBias_C.transfer_from(hBias_C));
    }

    if(arg.scaleD_vector)
        CHECK_HIP_ERROR(dScaleD.transfer_from(hScaleD));

    if(size_D_copy)
    {
        if(epilogue_on)
        {
            std::transform(hC.begin(), hC.end(), hD_gold_epl.begin(), [](To c) -> Talpha {
                return static_cast<Talpha>(c);
            });
        }
        else if(arg.scaleD_vector)
        {
            std::transform(hC.begin(), hC.end(), hD_gold_epl.begin(), [](To c) -> Talpha {
                return static_cast<Talpha>(c);
            });
        }
        else
        {
            std::copy(hC.begin(), hC.end(), hD_gold.begin());
        }
    }

    if(epilogue_on)
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)),
            HIPBLAS_STATUS_SUCCESS);

    bool change_bias_type = false;
    if(arg.bias_vector)
    {
        const void* bias_addr;
        if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
        {
            bias_addr        = dBias_C;
            change_bias_type = true;
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul,
                                                HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &arg.bias_type,
                                                sizeof(hipblasDatatype_t)),
                HIPBLAS_STATUS_SUCCESS);
        }
        else
        {
            bias_addr = dBias;
        }

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_addr, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);
    }

    if(arg.scaleD_vector)
    {
        const void* scaleD_addr = dScaleD;
        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleD_addr, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);
    }

    // Get Heuristic results
    hipblasLtMatmulHeuristicResult_t heuristicResult[3] = {0};
    int                              returnedAlgoCount  = 0;
    EXPECT_HIPBLAS_STATUS(
        (hipblasLtMatmulAlgoGetHeuristic(
            handle, matmul, matA, matB, matC, matD, pref, 3, heuristicResult, &returnedAlgoCount)),
        HIPBLAS_STATUS_SUCCESS);

    size_t                       workspace_size = heuristicResult[0].workspaceSize;
    device_vector<unsigned char> dWorkspace(workspace_size, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dWorkspace.memcheck());

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                              matmul,
                                              &h_alpha,
                                              dA,
                                              matA,
                                              dB,
                                              matB,
                                              &h_beta,
                                              dC,
                                              matC,
                                              dD,
                                              matD,
                                              &heuristicResult[0].algo,
                                              dWorkspace,
                                              workspace_size,
                                              stream),
                              HIPBLAS_STATUS_SUCCESS);
        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

#define epilogue_param \
    M, N, ldd, hD_gold_epl + pos, hD_gold + pos, arg.bias_vector, arg.scaleD_vector, hScaleD + 0
        for(int i = 0; i < num_batches; i++)
        {
            if(epilogue_on)
            {
                cblas_gemm<Ti, Talpha, Talpha>(transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               h_alpha,
                                               hA + stride_a * i,
                                               lda,
                                               hB + stride_b * i,
                                               ldb,
                                               h_beta,
                                               hD_gold_epl + stride_d * i,
                                               ldd,
                                               false);
                auto pos = stride_d * i;

                if(change_bias_type == false)
                {
                    switch(arg.activation_type)
                    {
                    case hipblaslt_activation_type::gelu:
                        epilogue_func(epilogue_param,
                                      hBias + 0,
                                      arg.activation_arg1,
                                      arg.activation_arg2,
                                      ::_gelu);
                        break;
                    case hipblaslt_activation_type::relu:
                        epilogue_func(epilogue_param,
                                      hBias + 0,
                                      arg.activation_arg1,
                                      arg.activation_arg2,
                                      ::_relu);
                        break;
                    default:
                        epilogue_func(epilogue_param, hBias + 0);
                        break;
                    }
                }
                else
                {
                    switch(arg.activation_type)
                    {
                    case hipblaslt_activation_type::gelu:
                        epilogue_func(epilogue_param,
                                      hBias_C + 0,
                                      arg.activation_arg1,
                                      arg.activation_arg2,
                                      ::_gelu);
                        break;
                    case hipblaslt_activation_type::relu:
                        epilogue_func(epilogue_param,
                                      hBias_C + 0,
                                      arg.activation_arg1,
                                      arg.activation_arg2,
                                      ::_relu);
                        break;
                    default:
                        epilogue_func(epilogue_param, hBias_C + 0);
                        break;
                    }
                }
            }
            else if(arg.scaleD_vector)
            {
                cblas_gemm<Ti, Talpha, Talpha>(transA,
                                               transB,
                                               M,
                                               N,
                                               K,
                                               h_alpha,
                                               hA + stride_a * i,
                                               lda,
                                               hB + stride_b * i,
                                               ldb,
                                               h_beta,
                                               hD_gold_epl + stride_d * i,
                                               ldd,
                                               false);
#define scaleD_param M, N, ldd, hD_gold_epl + pos, hD_gold + pos, arg.scaleD_vector, hScaleD + 0
                auto pos = stride_d * i;
                scaleD_func(scaleD_param);
            }
            else
            {
                cblas_gemm<Ti, To, Talpha>(transA,
                                           transB,
                                           M,
                                           N,
                                           K,
                                           h_alpha,
                                           hA + stride_a * i,
                                           lda,
                                           hB + stride_b * i,
                                           ldb,
                                           h_beta,
                                           hD_gold + stride_d * i,
                                           ldd,
                                           false);
            }
        }

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // fetch GPU
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hD_1.transfer_from(dD));

        if(arg.unit_check)
        {
            unit_check_general<To>(M, N, ldd, stride_d, hD_gold, hD_1, num_batches);
        }

        if(arg.norm_check)
        {
            hipblaslt_error = std::abs(
                norm_check_general<To>('F', M, N, ldd, stride_d, hD_gold, hD_1, num_batches));
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;
        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                  matmul,
                                                  &h_alpha,
                                                  dA,
                                                  matA,
                                                  dB,
                                                  matB,
                                                  &h_beta,
                                                  dC,
                                                  matC,
                                                  dD,
                                                  matD,
                                                  &heuristicResult[0].algo,
                                                  dWorkspace,
                                                  workspace_size,
                                                  stream),
                                  HIPBLAS_STATUS_SUCCESS);
        }

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                  matmul,
                                                  &h_alpha,
                                                  dA,
                                                  matA,
                                                  dB,
                                                  matB,
                                                  &h_beta,
                                                  dC,
                                                  matC,
                                                  dD,
                                                  matD,
                                                  &heuristicResult[0].algo,
                                                  dWorkspace,
                                                  workspace_size,
                                                  stream),
                                  HIPBLAS_STATUS_SUCCESS);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
        auto flops    = gemm_gflop_count<float>(M, N, K);
        switch(arg.activation_type)
        {
        case hipblaslt_activation_type::relu:
            flops += relu_gflop_count<float>(M, N);
            break;
        case hipblaslt_activation_type::gelu:
            flops += gelu_gflop_count<float>(M, N);
            break;
        default:
            break;
        }
#define argument_param_nb                                                                     \
    e_transA, e_transB, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a, e_beta, e_ldb, e_stride_b, \
        e_ldc, e_stride_c, e_ldd, e_stride_d, e_d_type, e_compute_type, e_activation_type,    \
        e_bias_vector
#define argument_param argument_param_nb, e_batch_count

        if(do_batched)
            ArgumentModel<argument_param>{}.log_args<float>(hipblaslt_cout,
                                                            arg,
                                                            gpu_time_used,
                                                            flops,
                                                            ArgumentLogging::NA_value,
                                                            cpu_time_used,
                                                            hipblaslt_error);
        else
            ArgumentModel<argument_param_nb>{}.log_args<float>(hipblaslt_cout,
                                                               arg,
                                                               gpu_time_used,
                                                               flops,
                                                               ArgumentLogging::NA_value,
                                                               cpu_time_used,
                                                               hipblaslt_error);
    }
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
