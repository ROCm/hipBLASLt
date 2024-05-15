/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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
#include <cstdlib>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <omp.h>
#include <set>

extern "C" __global__ void flush_icache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

template <typename T>
T absoluteValue(T in)
{
    return (float(in) > 0) ? in : T(-float(in));
}

template <typename T>
T maxValue(T sr1, T sr2)
{
    return (float(sr1) > float(sr2)) ? sr1 : sr2;

}

template <typename Ti, typename To>
void cpuAMax(To* out, Ti* in, std::uint32_t length)
{
    // calculate amax
    Ti m = Ti(0);
    for(int j = 0; j < length; j++)
    {
        m = maxValue(m, absoluteValue(in[j]));
    }
    out[0] = To(m);
}

template <typename Ti, typename To>
void cpuValueDividedByAMax(To* out, Ti* in, std::uint32_t length, float value)
{
    // calculate amax
    Ti m = Ti(0);
    for(int j = 0; j < length; j++)
    {
        m = maxValue(m, absoluteValue(in[j]));
    }
    out[0] = To(value/float(m));
}

template <typename Ti, typename Tc, typename To, typename Tbias, typename Tact, typename F>
void epilogue_func(int64_t m,
                   int64_t n,
                   int64_t ld,
                   Ti*     in,
                   To*     out,
                   Tc*     out_raw,
                   Tc*     amaxD,
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
        if(amaxD != nullptr)
        {
            for(int j = 0; j < n; j++)
            {
                auto pos = j * ld + i;
                *amaxD   = *amaxD > fabs(static_cast<Tc>(*(in + pos)))
                               ? *amaxD
                               : fabs(static_cast<Tc>(*(in + pos)));
            }
        }
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
                   Tc*     amaxD,
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
        if(amaxD != nullptr)
        {
            for(int j = 0; j < n; j++)
            {
                auto pos = j * ld + i;
                *amaxD   = *amaxD > fabs(static_cast<Tc>(*(in + pos)))
                               ? *amaxD
                               : fabs(static_cast<Tc>(*(in + pos)));
            }
        }
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

template <typename TiA,
          typename TiB,
          typename To,
          typename Tc,
          typename TciA = TiA,
          typename TciB = TiB>
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
    hipblaslt_local_matmul_descr  matmul(transA,
                                        transB,
                                        arg.compute_type,
                                        arg.scale_type,
                                        arg.compute_input_typeA,
                                        arg.compute_input_typeB);

    size_t workspace_size = 0;
    hipblaslt_local_preference pref;

    void* workspace = nullptr;
    float alpha = 1.0, beta = 0.0;

    hipStream_t stream = nullptr;
}

template <typename T>
void copy_gemm_to_host(hipStream_t                     stream,
                       const uint32_t&                 gemm_count,
                       std::vector<host_vector<T>*>&   hDst,
                       std::vector<device_vector<T>*>& dSrc)
{

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
    {
        CHECK_HIP_ERROR(hDst[gemmIdx]->transfer_from(*(dSrc[gemmIdx])));
    }
}

template <typename To, typename Tc, typename Tbias>
void check(hipStream_t                         stream,
           const Arguments&                    arg,
           const uint32_t&                     gemm_count,
           const std::vector<int64_t>&         M,
           const std::vector<int64_t>&         N,
           const std::vector<int64_t>&         ldd,
           const std::vector<int64_t>&         lde,
           const std::vector<int64_t>&         stride_d,
           const std::vector<int64_t>&         stride_e,
           const std::vector<int>&             num_batches,
           const std::vector<size_t>&          size_bias,
           std::vector<host_vector<To>*>&      hD_gold,
           std::vector<host_vector<To>*>&      hD_1,
           std::vector<device_vector<To>*>&    dD,
           std::vector<host_vector<Tc>*>&      hAmaxD_gold,
           std::vector<host_vector<Tc>*>&      hAmaxD,
           std::vector<device_vector<Tc>*>&    dAmaxD,
           std::vector<host_vector<To>*>&      hE_gold,
           std::vector<host_vector<To>*>&      hE,
           std::vector<device_vector<To>*>&    dE,
           std::vector<host_vector<Tbias>*>&   hBias_gold,
           std::vector<host_vector<Tbias>*>&   hBias,
           std::vector<device_vector<Tbias>*>& dBias,
           std::vector<double>&                tol,
           double&                             hipblaslt_error)
{
    // fetch GPU
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
    {
        if(!arg.gradient && arg.use_e)
            CHECK_HIP_ERROR(hE[gemmIdx]->transfer_from(*(dE[gemmIdx])));
        if(arg.amaxD)
        {
            CHECK_HIP_ERROR(hAmaxD[gemmIdx]->transfer_from(*(dAmaxD[gemmIdx])));
        }
        if(arg.gradient && arg.bias_vector)
        {
            CHECK_HIP_ERROR(hBias[gemmIdx]->transfer_from(*(dBias[gemmIdx])));
        }
        if(arg.unit_check)
        {
            if(tol[gemmIdx] != 0)
            {
                near_check_general<To>(M[gemmIdx],
                                       N[gemmIdx],
                                       ldd[gemmIdx],
                                       stride_d[gemmIdx],
                                       *(hD_gold[gemmIdx]),
                                       *(hD_1[gemmIdx]),
                                       num_batches[gemmIdx],
                                       tol[gemmIdx]);
            }
            else
            {
                unit_check_general<To>(M[gemmIdx],
                                       N[gemmIdx],
                                       ldd[gemmIdx],
                                       stride_d[gemmIdx],
                                       *(hD_gold[gemmIdx]),
                                       *(hD_1[gemmIdx]),
                                       num_batches[gemmIdx]);
            }
            if(arg.amaxD)
            {
                if(tol[gemmIdx] != 0)
                {
                    near_check_general<Tc>(1,
                                           1,
                                           1,
                                           1,
                                           *(hAmaxD_gold[gemmIdx]),
                                           *(hAmaxD[gemmIdx]),
                                           num_batches[gemmIdx],
                                           tol[gemmIdx]);
                }
                else
                {
                    unit_check_general<Tc>(1,
                                           1,
                                           1,
                                           1,
                                           *(hAmaxD_gold[gemmIdx]),
                                           *(hAmaxD[gemmIdx]),
                                           num_batches[gemmIdx]);
                }
            }
            if(!arg.gradient && arg.use_e)
            {
                if(tol[gemmIdx] != 0)
                {
                    near_check_general<To>(M[gemmIdx],
                                           N[gemmIdx],
                                           lde[gemmIdx],
                                           stride_e[gemmIdx],
                                           *(hE_gold[gemmIdx]),
                                           *(hE[gemmIdx]),
                                           num_batches[gemmIdx],
                                           tol[gemmIdx]);
                }
                else
                {
                    unit_check_general<To>(M[gemmIdx],
                                           N[gemmIdx],
                                           lde[gemmIdx],
                                           stride_e[gemmIdx],
                                           *(hE_gold[gemmIdx]),
                                           *(hE[gemmIdx]),
                                           num_batches[gemmIdx]);
                }
            }
            if(arg.gradient && arg.bias_vector)
            {
                if(tol[gemmIdx] != 0)
                {
                    near_check_general<Tbias>(size_bias[gemmIdx],
                                              1,
                                              size_bias[gemmIdx],
                                              size_bias[gemmIdx],
                                              *(hBias_gold[gemmIdx]),
                                              *(hBias[gemmIdx]),
                                              num_batches[gemmIdx],
                                              tol[gemmIdx]);
                }
                else
                {
                    unit_check_general<Tbias>(size_bias[gemmIdx],
                                              1,
                                              size_bias[gemmIdx],
                                              size_bias[gemmIdx],
                                              *(hBias_gold[gemmIdx]),
                                              *(hBias[gemmIdx]),
                                              num_batches[gemmIdx]);
                }
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

            if(arg.amaxD)
            {
                double norm_error = std::abs(norm_check_general<Tc>('F',
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    *(hAmaxD_gold[gemmIdx]),
                                                                    *(hAmaxD[gemmIdx]),
                                                                    num_batches[gemmIdx]));
                hipblaslt_error += norm_error;
                if(arg.norm_check_assert)
                    CHECK_SUCCESS(norm_check<Tc>(norm_error));
            }
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
                double norm_error = std::abs(norm_check_general<Tbias>('F',
                                                                       M[gemmIdx],
                                                                       1,
                                                                       M[gemmIdx],
                                                                       M[gemmIdx],
                                                                       *(hBias_gold[gemmIdx]),
                                                                       *(hBias[gemmIdx]),
                                                                       num_batches[gemmIdx]));
                hipblaslt_error += norm_error;
                if(arg.norm_check_assert)
                    CHECK_SUCCESS(norm_check<Tbias>(norm_error));
            }
        }
    }
}

// A function to determing the default bias_type
hipDataType derive_unset_bias_type(const Arguments& arg)
{
    // TODO: confirm if HIP_R_64F, HIP_R_32I are neccessary for biastype
    static const std::set<hipDataType> supported_bias_types
        = {HIP_R_32F, HIP_R_16F, HIP_R_16BF, HIP_R_64F, HIP_R_32I};

    hipDataType real_bias_type = arg.bias_type;

    // when bias type is unset.
    if(arg.bias_type == HIPBLASLT_DATATYPE_INVALID)
    {
        if(arg.compute_type == HIPBLAS_COMPUTE_32I)
        {
            real_bias_type = HIP_R_32I;
        }
        else if(arg.compute_type == HIPBLAS_COMPUTE_32F_FAST_TF32)
        {
            real_bias_type = HIP_R_32F;
        }
        else if((arg.a_type == HIP_R_8F_E4M3_FNUZ || arg.a_type == HIP_R_8F_E5M2_FNUZ)
                && (arg.b_type == HIP_R_8F_E4M3_FNUZ || arg.b_type == HIP_R_8F_E5M2_FNUZ))
        {
            if(arg.d_type == HIP_R_32F || arg.d_type == HIP_R_16BF)
                real_bias_type = HIP_R_16BF;
            else if(arg.d_type == HIP_R_16F)
                real_bias_type = HIP_R_16F;
            else //more default cases once support C != D
                real_bias_type = HIP_R_16F;
        }
        else
        {
            real_bias_type = arg.d_type;
        }
    }

    if(supported_bias_types.count(real_bias_type) == 0)
        throw std::invalid_argument("Invalid bias type "
                                    + std::string(hip_datatype_to_string(real_bias_type)));

    return real_bias_type;
}

template <typename TiA,
          typename TiB,
          typename To,
          typename Tc,
          typename TciA = TiA,
          typename TciB = TiB>
void testing_matmul(const Arguments& arg)
{
    // after this, real bias type should not be invalid
    hipDataType real_bias_type = derive_unset_bias_type(arg);

    // for all f8/bf8 cases including mix mode
    if constexpr((sizeof(TiA) == 1 || sizeof(TiB) == 1) && !std::is_same<Tc, int32_t>::value)
    {
        if constexpr(std::is_same<To, hip_bfloat16>::value || std::is_same<To, float>::value)
        {
          if(real_bias_type == HIP_R_16BF)
          {
              return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, hip_bfloat16>(arg);
          }
          else
          {
              return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, float>(arg);
          }
        }
        else
        {
          if(real_bias_type == HIP_R_16F)
          {
              return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, hipblasLtHalf>(arg);
          }
          else
          {
              return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, float>(arg);
          }
        }
    }
    else if constexpr(std::is_same<To, hipblasLtHalf>::value)
    {
        if(real_bias_type == HIP_R_16F)
        {
            return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, hipblasLtHalf>(arg);
        }
        else
        {
            return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, float>(arg);
        }
    }
    else if constexpr(std::is_same<To, hip_bfloat16>::value)
    {
        if(real_bias_type == HIP_R_16BF)
        {
            return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, hip_bfloat16>(arg);
        }
        else
        {
            return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, float>(arg);
        }
    }
    else if constexpr(std::is_same<To, float>::value)
    {
        return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, float>(arg);
    }
    else if constexpr(std::is_same<To, int32_t>::value)
    {
        return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, int32_t>(arg);
    }
    else if constexpr(std::is_same<To, double>::value)
    {
        return testing_matmul_with_bias<TiA, TiB, To, Tc, TciA, TciB, double>(arg);
    }
    // shouldn't arrive here
    CHECK_SUCCESS(false);
    return;
}

template <typename TiA,
          typename TiB,
          typename To,
          typename Tc,
          typename TciA,
          typename TciB,
          typename Tbias>
void testing_matmul_with_bias(const Arguments& arg)
{
    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used = 0.0;
    bool                   HMM    = arg.HMM;
    hipblaslt_local_handle handle{arg};
    hipStream_t            stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    hipEvent_t event_gpu_time_start, event_gpu_time_end;
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventCreate(&event_gpu_time_end));

    hipblasOperation_t transA(char_to_hipblas_operation(arg.transA));
    hipblasOperation_t transB(char_to_hipblas_operation(arg.transB));

    hipDataType tciA = arg.compute_input_typeA;
    hipDataType tciB = arg.compute_input_typeB;

    using Talpha = Tc;

    char*   case2           = getenv("CASE2");
    bool    do_grouped_gemm = arg.grouped_gemm > 0;
    int32_t gemm_count      = std::max(1, arg.grouped_gemm);
    int64_t rotating        = arg.rotating * 1024 * 1024;

    std::vector<int64_t> M(gemm_count), N(gemm_count), K(gemm_count), lda(gemm_count),
        ldb(gemm_count), ldc(gemm_count), ldd(gemm_count), lde(gemm_count);
    std::vector<Talpha>  h_alpha(gemm_count), h_beta(gemm_count);
    std::vector<int64_t> A_row(gemm_count), A_col(gemm_count), B_row(gemm_count), B_col(gemm_count);
    std::vector<int64_t> stride_a(gemm_count), stride_b(gemm_count), stride_c(gemm_count),
        stride_d(gemm_count), stride_e(gemm_count);
    std::vector<bool>   do_batched(gemm_count), epilogue_on(gemm_count, false);
    std::vector<int>    num_batches(gemm_count);
    std::vector<size_t> size_A(gemm_count), size_B(gemm_count), size_C(gemm_count),
        size_D(gemm_count), size_D_copy(gemm_count), size_E(gemm_count), size_bias(gemm_count),
        size_scaleAlphaVec(gemm_count);

    std::vector<hipblasLtMatrixLayout_t> matA(gemm_count), matB(gemm_count), matC(gemm_count),
        matD(gemm_count);
    std::vector<std::vector<hipblasLtMatmulDesc_t>> matmul;
    std::vector<hipblasLtEpilogue_t> epilogue(gemm_count, HIPBLASLT_EPILOGUE_DEFAULT);

    std::vector<device_vector<TiA>*>    dA(gemm_count);
    std::vector<device_vector<TiB>*>    dB(gemm_count);
    std::vector<device_vector<To>*>     dC(gemm_count), dD(gemm_count);
    std::vector<device_vector<Talpha>*> dScaleAlphaVec(gemm_count), dScaleA(gemm_count),
        dScaleB(gemm_count), dScaleC(gemm_count), dScaleD(gemm_count), dScaleE(gemm_count),
        dAmaxD(gemm_count);
    std::vector<device_vector<To>*>    dE(gemm_count);
    std::vector<device_vector<Tbias>*> dBias(gemm_count);
    std::vector<device_vector<TiA>*> dWorkSpaceA(gemm_count);
    std::vector<device_vector<std::int32_t>*> dSyncA(gemm_count);
    std::vector<device_vector<TiB>*> dWorkSpaceB(gemm_count);
    std::vector<device_vector<std::int32_t>*> dSyncB(gemm_count);

    std::vector<host_vector<TiA>*>    hA(gemm_count);
    std::vector<host_vector<TiB>*>    hB(gemm_count);
    std::vector<host_vector<To>*>     hC(gemm_count), hD_gold(gemm_count), hD_1(gemm_count);
    std::vector<host_vector<Talpha>*> hD_gold_epl(gemm_count), hScaleAlphaVec(gemm_count),
        hD_gold_ScaleAlpha(gemm_count), hBias_gold_epl(gemm_count), hScaleA(gemm_count),
        hScaleB(gemm_count), hScaleC(gemm_count), hScaleD(gemm_count), hScaleE(gemm_count),
        hAmaxD_gold(gemm_count), hAmaxD(gemm_count);

    std::vector<host_vector<To>*>    hE(gemm_count, nullptr), hE_gold(gemm_count, nullptr);
    std::vector<void*>               alpha_in(gemm_count);
    std::vector<host_vector<Tbias>*> hBias(gemm_count), hBias_gold(gemm_count);

    // Need to split into two for loop to calculate the rotating buffer
    int64_t totalRotatingSizeNeeded = 0;
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

        size_A[i]
            = stride_a[i] == 0 ? lda[i] * A_col[i] * num_batches[i] : stride_a[i] * num_batches[i];
        size_B[i]
            = stride_b[i] == 0 ? ldb[i] * B_col[i] * num_batches[i] : stride_b[i] * num_batches[i];
        size_C[i]
            = stride_c[i] == 0 ? ldc[i] * N[i] * num_batches[i] : stride_c[i] * num_batches[i];
        size_D[i]
            = stride_d[i] == 0 ? ldd[i] * N[i] * num_batches[i] : stride_d[i] * num_batches[i];

        size_E[i] = arg.use_e ? (stride_e[i] == 0 ? lde[i] * N[i] * num_batches[i]
                                                  : stride_e[i] * num_batches[i])
                              : 0;
        if(arg.c_equal_d)
        {
            ldd[i]      = arg.ldc[i];
            stride_d[i] = stride_c[i];
            size_D[i]   = size_C[i];
        }

        size_D_copy[i]        = arg.unit_check || arg.norm_check ? size_D[i] : 0;
        size_scaleAlphaVec[i] = arg.scaleAlpha_vector ? M[i] : 0;
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
            size_bias[i] = 0;
        }
        auto    biasSize = size_bias[i] * sizeof(Tbias);
        int64_t sizeC    = h_beta[i] == 0 ? 0 : size_C[i] * sizeof(To);
        totalRotatingSizeNeeded += size_A[i] * sizeof(TiA) + size_B[i] * sizeof(TiB) + sizeC
                                   + size_D[i] * sizeof(To) + size_E[i] * sizeof(To) + biasSize
                                   + size_scaleAlphaVec[i] * sizeof(Talpha);
    }

    // Calculating block count
    int32_t max_iters   = max(arg.cold_iters, arg.iters);
    int32_t block_count = max(1, min(max_iters, ceil((float)rotating / totalRotatingSizeNeeded)));
    if(rotating > 0)
    {
        hipblaslt_cout << "Rotating buffer " << rotating / (1024 * 1024) << " MiB. "
                       << "Needed Size: " << totalRotatingSizeNeeded / (1024 * 1024) << " MiB. "
                       << "Needed block count: " << block_count
                       << " (Capped to max iters: " << max_iters << ")" << std::endl;
    }
    // Calculating block count end
    matmul.resize(block_count, std::vector<hipblasLtMatmulDesc_t>(gemm_count));

    for(int i = 0; i < gemm_count; i++)
    {
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
            hipblasLtMatmulDescCreate(&(matmul[0][i]), arg.compute_type, arg.scale_type));

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &tciA, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);

        EXPECT_HIPBLAS_STATUS(
            hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &tciB, sizeof(void*)),
            HIPBLAS_STATUS_SUCCESS);

        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[0][i], HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul[0][i], HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));

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

        // allocate memory on device
        dA[i] = new device_vector<TiA>(size_A[i] * block_count, 1, HMM);
        dB[i] = new device_vector<TiB>(size_B[i] * block_count, 1, HMM);
        dC[i] = new device_vector<To>(size_C[i] * block_count, 1, HMM);
        if(!arg.c_equal_d)
            dD[i] = new device_vector<To>(size_D[i] * block_count, 1, HMM);
        else
            dD[i] = dC[i];
        dBias[i]          = new device_vector<Tbias>(size_bias[i] * block_count, 1, HMM);
        dScaleAlphaVec[i] = new device_vector<Talpha>(size_scaleAlphaVec[i] * block_count, 1, HMM);
        CHECK_DEVICE_ALLOCATION(dA[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dB[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dC[i]->memcheck());
        if(!arg.c_equal_d)
            CHECK_DEVICE_ALLOCATION(dD[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dBias[i]->memcheck());
        CHECK_DEVICE_ALLOCATION(dScaleAlphaVec[i]->memcheck());
        if(arg.use_e)
        {
            dE[i] = new device_vector<To>(size_E[i] * block_count, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dE[i]->memcheck());
        }
        else
        {
            dE[i] = nullptr;
        }

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
        if(arg.amaxD)
        {
            epilogue_on[i] = true;
            dAmaxD[i]      = new device_vector<Talpha>(1, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dAmaxD[i]->memcheck());
        }
        if(arg.scaleE)
        {
            dScaleE[i] = new device_vector<Talpha>(1, 1, HMM);
            CHECK_DEVICE_ALLOCATION(dScaleE[i]->memcheck());
        }

        // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
        hA[i]                 = new host_vector<TiA>(size_A[i]);
        hB[i]                 = new host_vector<TiB>(size_B[i]);
        hC[i]                 = new host_vector<To>(size_C[i]);
        hD_gold[i]            = new host_vector<To>(size_D_copy[i]);
        hD_gold_epl[i]        = new host_vector<Talpha>(size_D_copy[i]);
        hD_gold_ScaleAlpha[i] = new host_vector<Talpha>(size_D_copy[i]);
        hD_1[i]               = new host_vector<To>(size_D_copy[i]);
        hBias[i]              = new host_vector<Tbias>(size_bias[i]);
        hBias_gold[i]         = new host_vector<Tbias>(size_bias[i]);
        hBias_gold_epl[i]     = new host_vector<Talpha>(size_D_copy[i]); // Reduction for matrix D
        hScaleAlphaVec[i]     = new host_vector<Talpha>(size_scaleAlphaVec[i]);

        if(arg.scaleA)
            hScaleA[i] = new host_vector<Talpha>(1);
        if(arg.scaleB)
            hScaleB[i] = new host_vector<Talpha>(1);
        if(arg.scaleC)
            hScaleC[i] = new host_vector<Talpha>(1);
        if(arg.scaleD)
            hScaleD[i] = new host_vector<Talpha>(1);
        if(arg.amaxD)
        {
            hAmaxD_gold[i] = new host_vector<Talpha>(1);
            hAmaxD[i]      = new host_vector<Talpha>(1);
        }
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
            else if(arg.initialization == hipblaslt_initialization::zero)
            {
                hipblaslt_init_zero<TiA>(
                    *hA[i], A_row[i], A_col[i], lda[i], stride_a[i], num_batches[i]);
                hipblaslt_init_zero<TiB>(
                    *hB[i], B_row[i], B_col[i], ldb[i], stride_b[i], num_batches[i]);
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
            else if(arg.initialization == hipblaslt_initialization::zero)
                hipblaslt_init_zero<To>(*hC[i], M[i], N[i], ldc[i], stride_c[i], num_batches[i]);
        }

        if(arg.gradient && arg.use_e)
        {
            hipblaslt_init<To>(*hE[i], M[i], N[i], lde[i], stride_e[i], num_batches[i]);
        }

        if(arg.bias_vector)
        {
            hipblaslt_init<Tbias>(*hBias[i], M[i], 1, M[i]);
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

        if(arg.amaxD)
            hipblaslt_init_zero<Talpha>(*hAmaxD_gold[i], 1, 1, 1);

        if(arg.scaleE)
            hipblaslt_init<Talpha>(*hScaleE[i], 1, 1, 1);

        if(arg.scaleAlpha_vector)
            hipblaslt_init<Talpha>(*hScaleAlphaVec[i], M[i], 1, M[i]);

        // copy data from CPU to device
        CHECK_HIP_ERROR(dA[i]->transfer_from(*hA[i], block_count));
        CHECK_HIP_ERROR(dB[i]->transfer_from(*hB[i], block_count));
        CHECK_HIP_ERROR(dC[i]->transfer_from(*hC[i], block_count));
        if(arg.gradient && arg.use_e)
        {
            CHECK_HIP_ERROR(dE[i]->transfer_from(*hE[i], block_count));
        }
        if(!arg.gradient && arg.bias_vector)
        {
            CHECK_HIP_ERROR(dBias[i]->transfer_from(*hBias[i], block_count));
        }

        if(arg.scaleAlpha_vector)
        {
            CHECK_HIP_ERROR(dScaleAlphaVec[i]->transfer_from(*hScaleAlphaVec[i], block_count));
            alpha_in[i] = *(dScaleAlphaVec[i]);
            h_alpha[i]  = 1.0; // use dScaleAlphaVec instead, original alpha = 1.0 for verify
        }
        else
            alpha_in[i] = &(h_alpha[i]);

        if(arg.scaleA)
        {
            if(arg.amaxScaleA && (arg.a_type == HIP_R_32F || arg.a_type == HIP_R_16F))
                if (arg.isScaleAmaxDivisorA)
                    cpuValueDividedByAMax((*hScaleA[i]).data(), (*hA[i]).data(), A_row[i] * A_col[i], arg.amaxDividendA);
                else
                    cpuAMax((*hScaleA[i]).data(), (*hA[i]).data(), A_row[i] * A_col[i]);
            else
                CHECK_HIP_ERROR(dScaleA[i]->transfer_from(*hScaleA[i]));
        }

        if(arg.scaleB)
        {
            bool  amaxScaleB          = arg.amaxScaleB;
            bool  isScaleAmaxDivisorB = arg.isScaleAmaxDivisorB;
            float amaxDividendB       = arg.amaxDividendB;

            if(amaxScaleB && (arg.b_type == HIP_R_32F || arg.b_type == HIP_R_16F))
                if (isScaleAmaxDivisorB) {
                    cpuValueDividedByAMax((*hScaleB[i]).data(), (*hB[i]).data(), B_row[i] * B_col[i], amaxDividendB);
                }
                else
                    cpuAMax((*hScaleB[i]).data(), (*hB[i]).data(), B_row[i] * B_col[i]);
            else
                CHECK_HIP_ERROR(dScaleB[i]->transfer_from(*hScaleB[i]));
        }

        if(arg.scaleC)
            CHECK_HIP_ERROR(dScaleC[i]->transfer_from(*hScaleC[i]));

        if(arg.scaleD)
            CHECK_HIP_ERROR(dScaleD[i]->transfer_from(*hScaleD[i]));

        if(arg.scaleE)
            CHECK_HIP_ERROR(dScaleE[i]->transfer_from(*hScaleE[i]));

        //// copy data from CPU to device end

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
            EXPECT_HIPBLAS_STATUS(hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                                  HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                                  &(epilogue[i]),
                                                                  sizeof(epilogue[i])),
                                  HIPBLAS_STATUS_SUCCESS);

        if(arg.use_e)
        {
            void* e_addr = *dE[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &e_addr, sizeof(void*)));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &lde[i], sizeof(int64_t)));
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                                &stride_e[i],
                                                sizeof(int64_t)));
        }

        if(arg.bias_vector)
        {
            const void* bias_addr;
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &arg.bias_type,
                                                sizeof(hipDataType)),
                HIPBLAS_STATUS_SUCCESS);
            bias_addr = *dBias[i];

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_addr, sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);
        }

        if(arg.scaleA)
        {
            void* scaleA_addr = *dScaleA[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA_addr, sizeof(void*)));

            if (arg.amaxScaleA)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], HIPBLASLT_MATMUL_DESC_AMAX_SCALE_A, &arg.amaxScaleA, sizeof(bool)));
                if(arg.isScaleAmaxDivisorA)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[0][i], HIPBLASLT_MATMUL_DESC_IS_SCALE_AMAX_DIVISOR_A, &arg.isScaleAmaxDivisorA, sizeof(bool)));
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[0][i], HIPBLASLT_MATMUL_DESC_AMAX_DIVIDED_A, &arg.amaxDividendA, sizeof(float)));

                }
            }
        }

        if(arg.scaleB)
        {
            void* scaleB_addr = *dScaleB[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB_addr, sizeof(void*)));

            if (arg.amaxScaleB)
            {
                CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                    matmul[0][i], HIPBLASLT_MATMUL_DESC_AMAX_SCALE_B, &arg.amaxScaleB, sizeof(bool)));
                if(arg.isScaleAmaxDivisorB)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[0][i], HIPBLASLT_MATMUL_DESC_IS_SCALE_AMAX_DIVISOR_B, &arg.isScaleAmaxDivisorB, sizeof(bool)));
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                        matmul[0][i], HIPBLASLT_MATMUL_DESC_AMAX_DIVIDED_B, &arg.amaxDividendB, sizeof(float)));

                }
            }
        }

        if(arg.scaleC)
        {
            void* scaleC_addr = *dScaleC[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &scaleC_addr, sizeof(void*)));
        }

        if(arg.scaleD)
        {
            void* scaleD_addr = *dScaleD[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleD_addr, sizeof(void*)));
        }

        if(arg.amaxD)
        {
            void* amaxD_addr = *dAmaxD[i];
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul[0][i], HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, &amaxD_addr, sizeof(void*)));
        }

        if(arg.scaleE)
        {
            void* scaleE_addr = *dScaleE[i];
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
                                                &scaleE_addr,
                                                sizeof(void*)));
        }

        if(arg.scaleAlpha_vector)
        {
            hipblasLtPointerMode_t scale_mode
                = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[0][i],
                                                HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &scale_mode,
                                                sizeof(scale_mode)),
                HIPBLAS_STATUS_SUCCESS);
        }

        for(int32_t b = 1; b < matmul.size(); b++)
        {
            CHECK_HIPBLASLT_ERROR(
                hipblasLtMatmulDescCreate(&(matmul[b][i]), arg.compute_type, arg.scale_type));
            CHECK_HIPBLASLT_ERROR(hipblaslt_ext::copyMatmul(matmul[0][i], matmul[b][i]));

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
                                                &tciA,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            EXPECT_HIPBLAS_STATUS(
                hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
                                                &tciB,
                                                sizeof(void*)),
                HIPBLAS_STATUS_SUCCESS);

            // Update bias, E
            if(arg.bias_vector)
            {
                const void* bias_addr = (const void*)((*dBias[i]) + b * size_bias[i]);
                EXPECT_HIPBLAS_STATUS(
                    hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                    HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                    &bias_addr,
                                                    sizeof(void*)),
                    HIPBLAS_STATUS_SUCCESS);
            }
            if(arg.use_e)
            {
                void* e_addr = (*dE[i]) + b * size_E[i];
                CHECK_HIPBLASLT_ERROR(
                    hipblasLtMatmulDescSetAttribute(matmul[b][i],
                                                    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                    &e_addr,
                                                    sizeof(void*)));
            }
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
    int32_t requestAlgoCount = arg.requested_solution_num < 0 ? HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM
                                                              : arg.requested_solution_num;
    int     returnedAlgoCount = 0;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    std::vector<size_t>                           heuristicTuningIndex;

    // Cpp API
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    std::vector<hipblaslt_ext::Gemm>                    gemmVec;
    std::vector<hipblaslt_ext::GroupedGemm>             groupedGemmVec;
    std::vector<std::vector<hipblaslt_ext::GemmInputs>> extinputs;

    // C to Cpp API for GG
    std::vector<std::vector<void*>> da(block_count, std::vector<void*>(gemm_count));
    std::vector<std::vector<void*>> db(block_count, std::vector<void*>(gemm_count));
    std::vector<std::vector<void*>> dc(block_count, std::vector<void*>(gemm_count));
    std::vector<std::vector<void*>> dd(block_count, std::vector<void*>(gemm_count));

    for(int32_t b = 0; b < block_count; b++)
    {
        gemmVec.push_back(hipblaslt_ext::Gemm(handle,
                                              transA,
                                              transB,
                                              arg.a_type,
                                              arg.b_type,
                                              arg.c_type,
                                              arg.d_type,
                                              arg.compute_type));
        groupedGemmVec.push_back(hipblaslt_ext::GroupedGemm(handle,
                                                            transA,
                                                            transB,
                                                            arg.a_type,
                                                            arg.b_type,
                                                            arg.c_type,
                                                            arg.d_type,
                                                            arg.compute_type));
    }

    std::vector<hipblaslt_ext::GemmEpilogue> extepilogue;
    hipblaslt_ext::GemmProblemType           extproblemtype;
    if(arg.use_ext_setproblem)
    {
        extinputs.resize(block_count, std::vector<hipblaslt_ext::GemmInputs>(gemm_count));
        extepilogue.resize(gemm_count);

        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            auto  bias_type = HIPBLASLT_DATATYPE_INVALID;
            void* bias_addr = nullptr;
            for(int32_t b = 0; b < block_count; b++)
            {
                if(arg.bias_vector)
                {
                    bias_type = arg.bias_type;
                    bias_addr = (void*)((*dBias[gemmIdx]) + b * size_bias[gemmIdx]);
                }
                if(b == 0)
                {
                    extepilogue[gemmIdx].mode           = epilogue[gemmIdx];
                    extepilogue[gemmIdx].bias_data_type = bias_type;
                    extepilogue[gemmIdx].aux_ld         = lde[gemmIdx];
                    extepilogue[gemmIdx].aux_stride     = stride_e[gemmIdx];
                }

                extinputs[b][gemmIdx].a        = (void*)((*dA[gemmIdx]) + b * size_A[gemmIdx]);
                extinputs[b][gemmIdx].b        = (void*)((*dB[gemmIdx]) + b * size_B[gemmIdx]);
                extinputs[b][gemmIdx].c        = (void*)((*dC[gemmIdx]) + b * size_C[gemmIdx]);
                extinputs[b][gemmIdx].d        = (void*)((*dD[gemmIdx]) + b * size_D[gemmIdx]);
                extinputs[b][gemmIdx].alpha    = &h_alpha[gemmIdx];
                extinputs[b][gemmIdx].beta     = &h_beta[gemmIdx];
                extinputs[b][gemmIdx].bias     = bias_addr;
                extinputs[b][gemmIdx].scaleA   = arg.scaleA ? *dScaleA[gemmIdx] : nullptr;
                extinputs[b][gemmIdx].scaleB   = arg.scaleB ? *dScaleB[gemmIdx] : nullptr;
                extinputs[b][gemmIdx].scaleC   = arg.scaleC ? *dScaleC[gemmIdx] : nullptr;
                extinputs[b][gemmIdx].scaleD   = arg.scaleD ? *dScaleD[gemmIdx] : nullptr;
                extinputs[b][gemmIdx].scaleAux = arg.scaleE ? *dScaleE[gemmIdx] : nullptr;
                if(arg.use_e)
                    extinputs[b][gemmIdx].aux = (void*)((*dE[gemmIdx]) + b * size_E[gemmIdx]);
                if(arg.scaleAlpha_vector)
                    extinputs[b][gemmIdx].scaleAlphaVec
                        = (void*)((*dScaleAlphaVec[gemmIdx]) + b * size_scaleAlphaVec[gemmIdx]);
            }
        }
        extproblemtype.op_a         = transA;
        extproblemtype.op_b         = transB;
        extproblemtype.type_a       = arg.a_type;
        extproblemtype.type_b       = arg.b_type;
        extproblemtype.type_c       = arg.c_type;
        extproblemtype.type_d       = arg.d_type;
        extproblemtype.type_compute = arg.compute_type;
    }
    else if(arg.grouped_gemm)
    {
        for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
        {
            for(int32_t b = 0; b < block_count; b++)
            {
                da[b][gemmIdx] = (void*)((*dA[gemmIdx]) + b * size_A[gemmIdx]);
                db[b][gemmIdx] = (void*)((*dB[gemmIdx]) + b * size_B[gemmIdx]);
                dc[b][gemmIdx] = (void*)((*dC[gemmIdx]) + b * size_C[gemmIdx]);
                dd[b][gemmIdx] = (void*)((*dD[gemmIdx]) + b * size_D[gemmIdx]);
            }
        }
    }

    hipblaslt_ext::GemmType gemmType = do_grouped_gemm
                                           ? hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM
                                           : hipblaslt_ext::GemmType::HIPBLASLT_GEMM;

    // Remove duplicate
    std::vector<uint32_t> gsu_vector;
    std::vector<uint32_t> wgm_vector;
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.gsu_vector[i] == -1)
            break;
        gsu_vector.push_back(arg.gsu_vector[i]);
    }
    for(int32_t i = 0; i < MAX_SUPPORTED_NUM_PROBLEMS; i++)
    {
        if(arg.wgm_vector[i] == -1)
            break;
        wgm_vector.push_back(arg.wgm_vector[i]);
    }
    std::set<uint32_t> remove_duplicate(gsu_vector.begin(), gsu_vector.end());
    gsu_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    remove_duplicate = std::set<uint32_t>(wgm_vector.begin(), wgm_vector.end());
    wgm_vector.assign(remove_duplicate.begin(), remove_duplicate.end());
    std::vector<hipblaslt_ext::GemmTuning> tuningVec;
    if(arg.use_ext)
    {
        for(size_t wgm = 0; wgm < wgm_vector.size(); wgm++)
            for(size_t gsu = 0; gsu < gsu_vector.size(); gsu++)
            {
                hipblaslt_ext::GemmTuning tuning;
                tuning.splitK = gsu_vector[gsu];
                tuning.wgm    = wgm_vector[wgm];
                tuningVec.push_back(tuning);
            }
    }
    else
    {
        // C API does not support
        tuningVec.push_back(hipblaslt_ext::GemmTuning());
    }

    if(arg.algo_method == 2)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
        heuristicResult.clear();
        heuristicTuningIndex.clear();

        int algoIndexCount = 0;
        int algoIndexInc   = 100;
        while(1)
        {
            std::vector<int>                              algoIndex;
            std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;
            bool                                          foundAlgo = false;
            if(arg.solution_index == -1)
            {
                // Get algos by index
                // In real cases, the user can use the saved algo index to get the algorithm.
                // isAlgoSupported is not necessary if the user is sure that the algo supports the problem.
                algoIndex.resize(algoIndexInc);
                std::iota(std::begin(algoIndex), std::end(algoIndex), algoIndexCount);
                algoIndexCount += algoIndexInc;
            }
            else
            {
                // Specify the index
                algoIndex.resize(1);
                algoIndex[0] = arg.solution_index;
            }
            if(HIPBLAS_STATUS_INVALID_VALUE
               == hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, tmpAlgo))
            {
                break;
            }
            returnedAlgoCount = tmpAlgo.size();

            if(!do_grouped_gemm)
            {
                if(arg.use_ext)
                {
                    if(arg.use_ext_setproblem)
                    {
                        for(int32_t b = 0; b < block_count; b++)
                            CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(M[0],
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
                                                                        extinputs[b][0],
                                                                        extproblemtype));
                    }
                    else
                    {
                        for(int32_t b = 0; b < block_count; b++)
                            CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(matmul[b][0],
                                                                        alpha_in[0],
                                                                        *(dA[0]) + b * size_A[0],
                                                                        matA[0],
                                                                        *(dB[0]) + b * size_B[0],
                                                                        matB[0],
                                                                        &h_beta[0],
                                                                        *(dC[0]) + b * size_C[0],
                                                                        matC[0],
                                                                        *(dD[0]) + b * size_D[0],
                                                                        matD[0]));
                    }
                    for(int j = 0; j < returnedAlgoCount; j++)
                    {
                        for(size_t t = 0; t < tuningVec.size(); t++)
                        {
                            size_t tmpWorkspaceSize = 0;
                            if(gemmVec[0].isAlgoSupported(
                                   tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                               == HIPBLAS_STATUS_SUCCESS)
                            {
                                heuristicResult.push_back(tmpAlgo[j]);
                                heuristicTuningIndex.push_back(t);
                                workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                                foundAlgo      = true;
                            }
                        }
                        if(foundAlgo)
                            break;
                    }
                }
                else
                {
                    for(int j = 0; j < returnedAlgoCount; j++)
                    {
                        for(size_t t = 0; t < 1; t++) // CAPI not supported yet
                        {
                            size_t tmpWorkspaceSize = 0;
                            if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                                    matmul[0][0],
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
                                heuristicTuningIndex.push_back(t);
                                workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                                foundAlgo      = true;
                                break;
                            }
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
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(M,
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
                                                                           extinputs[b],
                                                                           extproblemtype));
                }
                else
                {
                    std::vector<void*> h_alpha_void, h_beta_void;
                    for(size_t i = 0; i < h_alpha.size(); i++)
                    {
                        h_alpha_void.push_back(&h_alpha[i]);
                        h_beta_void.push_back(&h_beta[i]);
                    }
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(matmul[b],
                                                                           h_alpha_void,
                                                                           da[b],
                                                                           matA,
                                                                           db[b],
                                                                           matB,
                                                                           h_beta_void,
                                                                           dc[b],
                                                                           matC,
                                                                           dd[b],
                                                                           matD));
                }

                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    for(size_t t = 0; t < tuningVec.size(); t++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(groupedGemmVec[0].isAlgoSupported(
                               tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                            foundAlgo      = true;
                        }
                    }
                    if(foundAlgo)
                        break;
                }
            }

            if(arg.solution_index != -1)
            {
                CHECK_SOLUTION_FOUND(foundAlgo);
                foundAlgo = true;
            }
            if(foundAlgo)
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
        heuristicTuningIndex.clear();
        int requestCount = 0;
        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                if(arg.use_ext_setproblem)
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(M[0],
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
                                                                    extinputs[b][0],
                                                                    extproblemtype));
                }
                else
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(matmul[b][0],
                                                                    alpha_in[0],
                                                                    *(dA[0]) + b * size_A[0],
                                                                    matA[0],
                                                                    *(dB[0]) + b * size_B[0],
                                                                    matB[0],
                                                                    &h_beta[0],
                                                                    *(dC[0]) + b * size_C[0],
                                                                    matC[0],
                                                                    *(dD[0]) + b * size_D[0],
                                                                    matD[0]));
                }
                for(int j = 0; j < returnedAlgoCount; j++)
                {
                    int addRequest = 0;
                    for(size_t t = 0; t < tuningVec.size(); t++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(gemmVec[0].isAlgoSupported(
                               tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            addRequest = 1;
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                        }
                    }
                    requestCount += addRequest;
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
                    int addRequest = 0;
                    for(size_t t = 0; t < 1; t++) // C API not supported yet
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                                matmul[0][0],
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
                            addRequest = 1;
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                        }
                    }
                    requestCount += addRequest;
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
                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(M,
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
                                                                       extinputs[b],
                                                                       extproblemtype));
            }
            else
            {
                std::vector<void*> h_alpha_void, h_beta_void;
                for(size_t i = 0; i < h_alpha.size(); i++)
                {
                    h_alpha_void.push_back(&h_alpha[i]);
                    h_beta_void.push_back(&h_beta[i]);
                }

                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(matmul[b],
                                                                       h_alpha_void,
                                                                       da[b],
                                                                       matA,
                                                                       db[b],
                                                                       matB,
                                                                       h_beta_void,
                                                                       dc[b],
                                                                       matC,
                                                                       dd[b],
                                                                       matD));
            }

            for(int j = 0; j < returnedAlgoCount; j++)
            {
                int    addRequest       = 0;
                size_t tmpWorkspaceSize = 0;
                for(size_t t = 0; t < tuningVec.size(); t++)
                {
                    if(groupedGemmVec[0].isAlgoSupported(
                           tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                       == HIPBLAS_STATUS_SUCCESS)
                    {
                        addRequest = 1;
                        heuristicResult.push_back(tmpAlgo[j]);
                        heuristicTuningIndex.push_back(t);
                        workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                    }
                }
                requestCount += addRequest;
                if(requestCount >= requestAlgoCount)
                {
                    break;
                }
            }
        }
    }
    else
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo;

        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                if(arg.use_ext_setproblem)
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(M[0],
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
                                                                    extinputs[b][0],
                                                                    extproblemtype));
                }
                else
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(gemmVec[b].setProblem(matmul[b][0],
                                                                    alpha_in[0],
                                                                    *(dA[0]) + b * size_A[0],
                                                                    matA[0],
                                                                    *(dB[0]) + b * size_B[0],
                                                                    matB[0],
                                                                    &h_beta[0],
                                                                    *(dC[0]) + b * size_C[0],
                                                                    matC[0],
                                                                    *(dD[0]) + b * size_D[0],
                                                                    matD[0]));
                }
                CHECK_HIPBLASLT_ERROR(
                    gemmVec[0].algoGetHeuristic(requestAlgoCount, gemmPref, tmpAlgo));
                heuristicResult.clear();
                heuristicTuningIndex.clear();
                for(int j = 0; j < tmpAlgo.size(); j++)
                {
                    for(size_t t = 0; t < tuningVec.size(); t++)
                    {
                        size_t tmpWorkspaceSize = 0;
                        if(gemmVec[0].isAlgoSupported(
                               tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                           == HIPBLAS_STATUS_SUCCESS)
                        {
                            heuristicResult.push_back(tmpAlgo[j]);
                            heuristicTuningIndex.push_back(t);
                            workspace_size = std::max(workspace_size, tmpWorkspaceSize);
                        }
                    }
                }
                returnedAlgoCount = heuristicResult.size();
            }
            else
            {
                std::vector<hipblasLtMatmulHeuristicResult_t> tmpAlgo(requestAlgoCount);
                EXPECT_HIPBLAS_STATUS((hipblasLtMatmulAlgoGetHeuristic(handle,
                                                                       matmul[0][0],
                                                                       matA[0],
                                                                       matB[0],
                                                                       matC[0],
                                                                       matD[0],
                                                                       pref,
                                                                       requestAlgoCount,
                                                                       tmpAlgo.data(),
                                                                       &returnedAlgoCount)),
                                      HIPBLAS_STATUS_SUCCESS);
                heuristicResult.clear();
                for(int32_t i = 0; i < returnedAlgoCount; i++)
                {
                    heuristicResult.push_back(tmpAlgo[i]);
                }
                heuristicTuningIndex.resize(heuristicResult.size(), 0); // C API not supported yet
            }

            for(int i = 0; i < returnedAlgoCount; i++)
                workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);
        }
        else
        {
            if(arg.use_ext_setproblem)
            {
                auto num_batches_64 = std::vector<int64_t>{num_batches.begin(), num_batches.end()};
                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(M,
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
                                                                       extinputs[b],
                                                                       extproblemtype));
            }
            else
            {
                std::vector<void*> h_alpha_void, h_beta_void;
                for(size_t i = 0; i < h_alpha.size(); i++)
                {
                    h_alpha_void.push_back(&h_alpha[i]);
                    h_beta_void.push_back(&h_beta[i]);
                }
                for(int32_t b = 0; b < block_count; b++)
                    CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].setProblem(matmul[b],
                                                                       h_alpha_void,
                                                                       da[b],
                                                                       matA,
                                                                       db[b],
                                                                       matB,
                                                                       h_beta_void,
                                                                       dc[b],
                                                                       matC,
                                                                       dd[b],
                                                                       matD));
            }

            CHECK_HIPBLASLT_ERROR(
                groupedGemmVec[0].algoGetHeuristic(requestAlgoCount, gemmPref, tmpAlgo));
            heuristicResult.clear();
            heuristicTuningIndex.clear();
            for(int j = 0; j < tmpAlgo.size(); j++)
            {
                for(size_t t = 0; t < tuningVec.size(); t++)
                {
                    size_t tmpWorkspaceSize = 0;
                    if(groupedGemmVec[0].isAlgoSupported(
                           tmpAlgo[j].algo, tuningVec[t], tmpWorkspaceSize)
                       == HIPBLAS_STATUS_SUCCESS)
                    {
                        heuristicResult.push_back(tmpAlgo[j]);
                        heuristicTuningIndex.push_back(t);
                    }
                }
            }
            workspace_size = max_workspace_size;
        }
    }

    returnedAlgoCount = heuristicResult.size();

    if(returnedAlgoCount == 0)
    {
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
            delete hScaleAlphaVec[i];
            delete dA[i];
            delete dB[i];
            delete dC[i];
            if(!arg.c_equal_d)
                delete dD[i];
            delete dBias[i];
            delete dScaleAlphaVec[i];
            if(arg.scaleA)
            {
                delete hScaleA[i];
                delete dScaleA[i];
            }
            if(arg.scaleB)
            {
                delete hScaleB[i];
            }
            if(arg.scaleB)
            {
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
            if(arg.amaxD)
            {
                delete hAmaxD_gold[i];
                delete hAmaxD[i];
                delete dAmaxD[i];
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
        int             deviceId;
        hipDeviceProp_t deviceProperties;
        static_cast<void>(hipGetDevice(&deviceId));
        static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
        //workaround before known_bug work
        if(gpu_arch_match(deviceProperties.gcnArchName, "11?")
           && (arg.gradient || arg.grouped_gemm || arg.a_type == HIP_R_32F
               || arg.b_type == HIP_R_32F || arg.a_type == HIP_R_64F
               || arg.b_type
                      == HIP_R_64F)) //arg.activation_type == gelu || arg.bias_source == a || arg.bias_source == b)
        {
            hipblaslt_cerr << "No Solution Found!!" << std::endl;
            return;
        }
    }

    CHECK_SOLUTION_FOUND(returnedAlgoCount);

    dWorkspace = new device_vector<unsigned char>(workspace_size * block_count, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dWorkspace->memcheck());

    if(arg.use_user_args)
    {
        CHECK_HIP_ERROR(
            hipHostMalloc(&userArgs, gemm_count * sizeof(hipblaslt_ext::UserArguments)));
        CHECK_HIP_ERROR(hipMalloc(&d_userArgs,
                                  block_count * gemm_count * sizeof(hipblaslt_ext::UserArguments)));
    }

    auto ptrs = benchmark_allocation();

    if(arg.print_solution_found)
        hipblaslt_cout << "Is supported " << heuristicResult.size()
                       << " / Total solutions: " << returnedAlgoCount * tuningVec.size()
                       << std::endl;

    if(heuristicResult.size() != heuristicTuningIndex.size())
    {
        hipblaslt_cerr << "Internal error, heuristicResult.size() != heuristicTuningIndex.size() "
                       << heuristicResult.size() << " != " << heuristicTuningIndex.size()
                       << std::endl;
        exit(EXIT_FAILURE);
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

#define epilogue_param                                                                     \
    M[gemmIdx], N[gemmIdx], ldd[gemmIdx], *(hD_gold_epl[gemmIdx]) + pos,                   \
        *(hD_gold[gemmIdx]) + pos, *(hBias_gold_epl[gemmIdx]) + pos,                       \
        arg.amaxD ? *(hAmaxD_gold[gemmIdx]) + 0 : nullptr, ePos, scaleDValue, scaleEValue, \
        applyBias
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
                    cblas_gemm<TiA, TiB, Talpha, Talpha, TciA, TciB>(
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
                        arg.scaleAlpha_vector ? *(hScaleAlphaVec[gemmIdx]) + 0 : nullptr,
                        scaleAValue,
                        scaleBValue,
                        1,
                        false);
                    auto pos    = stride_d[gemmIdx] * batchIdx;
                    auto hEInst = arg.gradient ? hE : hE_gold;
                    auto ePos = (hEInst[gemmIdx] == nullptr) ? nullptr : (*(hEInst[gemmIdx]) + pos);
                    auto applyBias = arg.gradient ? false : arg.bias_vector;

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
                    if(arg.gradient && arg.bias_vector && batchIdx == num_batches[gemmIdx] - 1)
                    {
                        if(arg.bias_source == hipblaslt_bias_source::d)
                        {
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
                                          &hBias_gold,
                                          &size_bias,
                                          &K,
                                          &num_batches,
                                          &gemmIdx,
                                          &arg]<typename Ti>(Ti* ptr) {
                                if(sumLd)
                                {
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
                    cblas_gemm<TiA, TiB, To, Talpha, TciA, TciB>(
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
    }

    if(!arg.timing)
    {
        if(!do_grouped_gemm)
        {
            if(arg.use_ext)
            {
                CHECK_HIPBLASLT_ERROR(gemmVec[0].initialize(
                    heuristicResult[0].algo, tuningVec[heuristicTuningIndex[0]], *dWorkspace));

                CHECK_HIPBLASLT_ERROR(gemmVec[0].run(stream));
            }
            else
            {
                CHECK_HIP_ERROR(hipStreamSynchronize(stream));
                EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                      matmul[0][0],
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
                CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].initialize(
                    heuristicResult[0].algo, tuningVec[heuristicTuningIndex[0]], *dWorkspace));
                groupedGemmVec[0].getDefaultValueForDeviceUserArguments(userArgs);
                // Copy them to device memory
                CHECK_HIP_ERROR(hipMemcpy(d_userArgs,
                                          userArgs,
                                          gemm_count * sizeof(hipblaslt_ext::UserArguments),
                                          hipMemcpyHostToDevice));

                CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(d_userArgs, stream));
            }
            else
            {
                //grouped gemm
                CHECK_HIPBLASLT_ERROR(
                    groupedGemmVec[0].initialize(heuristicResult[0].algo,
                                                 tuningVec[heuristicTuningIndex[0]],
                                                 *dWorkspace,
                                                 false,
                                                 stream));

                CHECK_HIPBLASLT_ERROR(groupedGemmVec[0].run(stream));
            }
        }

        double              hipblaslt_error = 0.0;
        std::vector<double> tol(gemm_count);
        if(arg.unit_check && hipblaslt_get_arch_major() == 11 && sizeof(TiA) == 2
           && sizeof(TiB) == 2)
        {
            for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
            {
                tol[gemmIdx] = K[gemmIdx] * sum_error_tolerance_for_gfx11<Tc, TiA, To>;
            }
        }
        if(arg.unit_check || arg.norm_check)
        {
            copy_gemm_to_host(stream, gemm_count, hD_1, dD);
            check(stream,
                  arg,
                  gemm_count,
                  M,
                  N,
                  ldd,
                  lde,
                  stride_d,
                  stride_e,
                  num_batches,
                  size_bias,
                  hD_gold,
                  hD_1,
                  dD,
                  hAmaxD_gold,
                  hAmaxD,
                  dAmaxD,
                  hE_gold,
                  hE,
                  dE,
                  hBias_gold,
                  hBias,
                  dBias,
                  tol,
                  hipblaslt_error);
        }
    }
    else
    {
        // Get device information
        hipDeviceProp_t deviceProps;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProps, 0));
        int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

        size_t      best_sol      = -1;
        double      best_flops    = 0.0;
        double      best_gpu_time = std::numeric_limits<double>::max();
        std::string best_s_name   = "";
        std::string best_k_name   = "";
        int         number_cold_calls
            = ((arg.unit_check || arg.norm_check) && arg.cold_iters == 0) ? 1 : arg.cold_iters;
        int number_hot_calls = arg.iters;

        int    flush_iter      = 100000;
        double flush_time_used = 0;
        if(arg.flush)
        {
            for(int i = 0; i < flush_iter; i++)
                hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);

            if(arg.use_gpu_timer)
                CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
            else
            {
                flush_time_used = get_time_us_sync(stream);
            }
            for(int i = 0; i < flush_iter; i++)
                hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
            if(arg.use_gpu_timer)
            {
                CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
                CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
                float gpu_time_ms;
                CHECK_HIP_ERROR(
                    hipEventElapsedTime(&gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
                flush_time_used = gpu_time_ms * 1000; // ms to us
            }
            else
            {
                flush_time_used = get_time_us_sync(stream) - flush_time_used;
            }
            flush_time_used /= flush_iter;
        }

        for(size_t sol = 0; sol < heuristicResult.size(); sol++)
        {
            if(!do_grouped_gemm)
            {
                if(arg.use_ext)
                {
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(
                            gemmVec[b].initialize(heuristicResult[sol].algo,
                                                  tuningVec[heuristicTuningIndex[sol]],
                                                  *dWorkspace));
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(gemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, dD);
                    }
                    if(arg.use_gpu_timer)
                        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
                    else
                    {
                        gpu_time_used = get_time_us_sync(stream);
                    }

                    for(int i = 0; i < number_hot_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(gemmVec[i % block_count].run(stream));
                        if(arg.flush)
                            hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                    }
                }
                else
                {
                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        TiA* ptr_dA     = *(dA[0]) + (i % block_count) * size_A[0];
                        TiB* ptr_dB     = *(dB[0]) + (i % block_count) * size_B[0];
                        To*  ptr_dC     = *(dC[0]) + (i % block_count) * size_C[0];
                        To*  ptr_dD     = *(dD[0]) + (i % block_count) * size_D[0];
                        auto ptr_matmul = matmul[i % block_count][0];
                        auto ptr_alpha
                            = arg.scaleAlpha_vector
                                  ? *(dScaleAlphaVec[0]) + (i % block_count) * size_scaleAlphaVec[0]
                                  : alpha_in[0];
                        EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                              ptr_matmul,
                                                              ptr_alpha,
                                                              ptr_dA,
                                                              matA[0],
                                                              ptr_dB,
                                                              matB[0],
                                                              &(h_beta[0]),
                                                              ptr_dC,
                                                              matC[0],
                                                              ptr_dD,
                                                              matD[0],
                                                              &heuristicResult[sol].algo,
                                                              *dWorkspace,
                                                              workspace_size,
                                                              stream),
                                              HIPBLAS_STATUS_SUCCESS);
                        if(i == 0 && (arg.unit_check || arg.norm_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, dD);
                    }

                    if(arg.use_gpu_timer)
                        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
                    else
                    {
                        gpu_time_used = get_time_us_sync(stream);
                    }
                    for(int i = 0; i < number_hot_calls; i++)
                    {
                        TiA* ptr_dA     = *(dA[0]) + (i % block_count) * size_A[0];
                        TiB* ptr_dB     = *(dB[0]) + (i % block_count) * size_B[0];
                        To*  ptr_dC     = *(dC[0]) + (i % block_count) * size_C[0];
                        To*  ptr_dD     = *(dD[0]) + (i % block_count) * size_D[0];
                        auto ptr_matmul = matmul[i % block_count][0];
                        auto ptr_alpha
                            = arg.scaleAlpha_vector
                                  ? *(dScaleAlphaVec[0]) + (i % block_count) * size_scaleAlphaVec[0]
                                  : alpha_in[0];
                        EXPECT_HIPBLAS_STATUS(hipblasLtMatmul(handle,
                                                              ptr_matmul,
                                                              ptr_alpha,
                                                              ptr_dA,
                                                              matA[0],
                                                              ptr_dB,
                                                              matB[0],
                                                              &(h_beta[0]),
                                                              ptr_dC,
                                                              matC[0],
                                                              ptr_dD,
                                                              matD[0],
                                                              &heuristicResult[sol].algo,
                                                              *dWorkspace,
                                                              workspace_size,
                                                              stream),
                                              HIPBLAS_STATUS_SUCCESS);
                        if(arg.flush)
                            hipLaunchKernelGGL(flush_icache, dim3(gpu_block3), dim3(64), 0, stream);
                    }
                }
                if(arg.use_gpu_timer)
                {
                    CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
                    CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
                    float gpu_time_ms;
                    CHECK_HIP_ERROR(hipEventElapsedTime(
                        &gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
                    gpu_time_used = gpu_time_ms * 1000; // ms to us
                }
                else
                {
                    gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
                }
            }
            else
            {
                if(arg.use_user_args)
                {
                    std::vector<unsigned char*> d_userArgsVec(block_count);
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size)));
                        groupedGemmVec[b].getDefaultValueForDeviceUserArguments(userArgs);
                        d_userArgsVec[b] = (unsigned char*)d_userArgs
                                           + b * gemm_count * sizeof(hipblaslt_ext::UserArguments);
                        // Copy them to device memory
                        CHECK_HIP_ERROR(hipMemcpy(d_userArgsVec[b],
                                                  userArgs,
                                                  gemm_count * sizeof(hipblaslt_ext::UserArguments),
                                                  hipMemcpyHostToDevice));
                    }

                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(
                            d_userArgsVec[i % block_count], stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, dD);
                    }
                    if(arg.use_gpu_timer)
                        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
                    else
                    {
                        gpu_time_used = get_time_us_sync(stream);
                    }

                    for(int i = 0; i < number_hot_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(
                            d_userArgsVec[i % block_count], stream));
                    }

                    if(arg.use_gpu_timer)
                    {
                        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
                        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
                        float gpu_time_ms;
                        CHECK_HIP_ERROR(hipEventElapsedTime(
                            &gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
                        gpu_time_used = gpu_time_ms * 1000; // ms to us
                    }
                    else
                    {
                        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
                    }
                }
                else
                {
                    //grouped gemm
                    for(int32_t b = 0; b < block_count; b++)
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[b].initialize(
                            heuristicResult[sol].algo,
                            tuningVec[heuristicTuningIndex[sol]],
                            ((unsigned char*)(*dWorkspace) + b * workspace_size),
                            false,
                            stream));

                    for(int i = 0; i < number_cold_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(stream));
                        if(i == 0 && (arg.unit_check || arg.norm_check))
                            copy_gemm_to_host(stream, gemm_count, hD_1, dD);
                    }
                    if(arg.use_gpu_timer)
                        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_start, stream));
                    else
                    {
                        gpu_time_used = get_time_us_sync(stream);
                    }

                    for(int i = 0; i < number_hot_calls; i++)
                    {
                        CHECK_HIPBLASLT_ERROR(groupedGemmVec[i % block_count].run(stream));
                    }

                    if(arg.use_gpu_timer)
                    {
                        CHECK_HIP_ERROR(hipEventRecord(event_gpu_time_end, stream));
                        CHECK_HIP_ERROR(hipEventSynchronize(event_gpu_time_end));
                        float gpu_time_ms;
                        CHECK_HIP_ERROR(hipEventElapsedTime(
                            &gpu_time_ms, event_gpu_time_start, event_gpu_time_end));
                        gpu_time_used = gpu_time_ms * 1000; // ms to us
                    }
                    else
                    {
                        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;
                    }
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

            double              hipblaslt_error = 0.0;
            std::vector<double> tol(gemm_count);
            if(arg.unit_check && hipblaslt_get_arch_major() == 11 && sizeof(TiA) == 2
               && sizeof(TiB) == 2)
            {
                for(int gemmIdx = 0; gemmIdx < gemm_count; gemmIdx++)
                {
                    tol[gemmIdx] = K[gemmIdx] * sum_error_tolerance_for_gfx11<Tc, TiA, To>;
                }
            }
            if(arg.unit_check || arg.norm_check)
                check(stream,
                      arg,
                      gemm_count,
                      M,
                      N,
                      ldd,
                      lde,
                      stride_d,
                      stride_e,
                      num_batches,
                      size_bias,
                      hD_gold,
                      hD_1,
                      dD,
                      hAmaxD_gold,
                      hAmaxD,
                      dAmaxD,
                      hE_gold,
                      hE,
                      dE,
                      hBias_gold,
                      hBias,
                      dBias,
                      tol,
                      hipblaslt_error);

#define argument_param                                                                             \
    e_transA, e_transB, e_grouped_gemm, e_batch_count, e_M, e_N, e_K, e_alpha, e_lda, e_stride_a,  \
        e_beta, e_ldb, e_stride_b, e_ldc, e_stride_c, e_ldd, e_stride_d, e_d_type, e_compute_type, \
        e_activation_type, e_bias_vector, e_rotating

            int32_t     solutionIndex = -1;
            std::string solutionName  = "";
            std::string kernelName    = "";
            if(arg.print_kernel_info)
            {
                if(arg.use_ext)
                {
                    if(!do_grouped_gemm)
                    {
                        solutionName = gemmVec[0].getSolutionName();
                        kernelName   = gemmVec[0].getKernelName();
                    }
                    else
                    {
                        solutionName = groupedGemmVec[0].getSolutionName();
                        kernelName   = groupedGemmVec[0].getKernelName();
                    }
                }
                else
                {
                    solutionName
                        = hipblaslt_ext::getSolutionNameFromAlgo(handle, heuristicResult[sol].algo);
                    kernelName
                        = hipblaslt_ext::getKernelNameFromAlgo(handle, heuristicResult[sol].algo);
                }
                solutionIndex = hipblaslt_ext::getIndexFromAlgo(heuristicResult[sol].algo);
            }
            ArgumentModel<argument_param>{}.log_args<Tc>(
                hipblaslt_cout,
                sol,
                solutionIndex,
                solutionName,
                kernelName,
                arg,
                (uint32_t)tuningVec[heuristicTuningIndex[sol]].splitK,
                (uint32_t)tuningVec[heuristicTuningIndex[sol]].wgm,
                gpu_time_used,
                flush_time_used,
                flops,
                ArgumentLogging::NA_value,
                cpu_time_used,
                hipblaslt_error);
            if(best_gpu_time > gpu_time_used)
            {
                best_sol      = sol;
                best_flops    = flops;
                best_gpu_time = gpu_time_used;
                best_s_name   = solutionName;
                best_k_name   = kernelName;
            }
        }

        if(heuristicResult.size() > 1)
        {
            int32_t     solutionIndex = -1;
            std::string solutionName  = "";
            std::string kernelName    = "";
            if(arg.print_kernel_info)
            {
                solutionIndex = hipblaslt_ext::getIndexFromAlgo(heuristicResult[best_sol].algo);
                solutionName  = best_s_name;
                kernelName    = best_k_name;
            }

            hipblaslt_cout << "Winner: " << std::endl;
            ArgumentModel<argument_param>{}.log_args<Tc>(
                hipblaslt_cout,
                best_sol,
                solutionIndex,
                solutionName,
                kernelName,
                arg,
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].splitK,
                (uint32_t)tuningVec[heuristicTuningIndex[best_sol]].wgm,
                best_gpu_time,
                flush_time_used,
                best_flops,
                ArgumentLogging::NA_value,
                cpu_time_used,
                0.0);
        }
    }

    for(auto it : ptrs)
    {
        CHECK_HIP_ERROR(hipFree(it));
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
        delete hScaleAlphaVec[i];
        delete dA[i];
        delete dB[i];
        delete dC[i];
        if(!arg.c_equal_d)
            delete dD[i];
        delete dBias[i];
        delete dScaleAlphaVec[i];
        if(arg.scaleA)
        {
            delete hScaleA[i];
            delete dScaleA[i];
        }
        if(arg.scaleB)
        {
            delete hScaleB[i];
        }
        if(arg.scaleB)
        {
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
        if(arg.amaxD)
        {
            delete hAmaxD_gold[i];
            delete hAmaxD[i];
            delete dAmaxD[i];
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
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_start));
    CHECK_HIP_ERROR(hipEventDestroy(event_gpu_time_end));
}
