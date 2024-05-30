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

#include "hipblaslt_arguments.hpp"
#include <hipblaslt/hipblaslt.h>

template <typename T>
constexpr auto hipblaslt_type2datatype()
{
    if(std::is_same<T, hipblasLtHalf>{})
        return HIP_R_16F;
    if(std::is_same<T, hip_bfloat16>{})
        return HIP_R_16BF;
    if(std::is_same<T, float>{})
        return HIP_R_32F;
    if(std::is_same<T, hipblaslt_f8_fnuz>{})
        return HIP_R_8F_E4M3_FNUZ;
    if(std::is_same<T, hipblaslt_bf8_fnuz>{})
        return HIP_R_8F_E5M2_FNUZ;
    if(std::is_same<T, int32_t>{})
        return HIP_R_32I;
    if(std::is_same<T, hipblasLtInt8>{})
        return HIP_R_8I;

    return HIP_R_16F; // testing purposes we default to f32 ex
}

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto hipblaslt_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case HIP_R_16F:
        return TEST<hipblasLtHalf>{}(arg);
    case HIP_R_16BF:
        return TEST<hip_bfloat16>{}(arg);
    case HIP_R_32F:
        return TEST<float>{}(arg);
    case HIP_R_8F_E4M3_FNUZ:
        return TEST<hipblaslt_f8_fnuz>{}(arg);
    case HIP_R_8F_E5M2_FNUZ:
        return TEST<hipblaslt_bf8_fnuz>{}(arg);
    case HIP_R_8I:
        return TEST<hipblasLtInt8>{}(arg);
    case HIP_R_32I:
        return TEST<int32_t>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// gemm functions
template <template <typename...> class TEST>
auto hipblaslt_matmul_dispatch(const Arguments& arg)
{
    const auto TiA = arg.a_type;
    const auto TiB = arg.b_type;
    auto       To  = arg.c_type;
    auto       Tc  = arg.compute_type;
    auto       TciA = arg.compute_input_typeA;
    auto       TciB = arg.compute_input_typeB;
    char*      case2 = getenv("CASE2");

    if(arg.d_type == To)
    {
        if(TiA == To && TiB == To && To == HIP_R_16F && Tc == HIPBLAS_COMPUTE_32F)
        {
            if(TciA == TciB && TciA == HIP_R_16F)
                return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hipblasLtHalf, hipblasLtHalf>{}(arg);
            else if(TciA == TciB && TciA == HIP_R_16BF)
                return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hip_bfloat16, hip_bfloat16>{}(arg);
            else if(TciA == TciB && TciA == HIP_R_8F_E4M3_FNUZ)
                return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz>{}(arg);
            else if(TciA == HIP_R_8F_E4M3_FNUZ && TciB == HIP_R_8F_E5M2_FNUZ)
                return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz>{}(arg);
            else if(TciA == HIP_R_8F_E5M2_FNUZ && TciB == HIP_R_8F_E4M3_FNUZ)
                return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz>{}(arg);
            else
                return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIP_R_16F && Tc == HIPBLAS_COMPUTE_32F_FAST_16BF)
        {
            return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, hip_bfloat16, hip_bfloat16>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIP_R_16BF && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hip_bfloat16, hip_bfloat16, hip_bfloat16, float>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIP_R_32F
                && (Tc == HIPBLAS_COMPUTE_32F || Tc == HIPBLAS_COMPUTE_32F_FAST_TF32))
        {
            return TEST<float, float, float, float>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIP_R_64F && (Tc == HIPBLAS_COMPUTE_64F))
        {
            return TEST<double, double, double, double>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_16F && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblasLtHalf, hipblasLtHalf, float, float>{}(arg);
        }
        else if(TiA == HIP_R_16BF && TiB == HIP_R_16BF && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hip_bfloat16, hip_bfloat16, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblasLtBfloat16, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz, hipblasLtBfloat16, float>{}(arg);
        }
        // FP8/BF8 input-output combination
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_8F_E5M2_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_8F_E5M2_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_8F_E5M2_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_8F_E5M2_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz, hipblaslt_bf8_fnuz, float>{}(arg);
        }
        // end of FP8/BF8 combinations
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2_FNUZ && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8_fnuz, hipblaslt_f8_fnuz, hipblasLtBfloat16, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_8F_E5M2_FNUZ && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblaslt_bf8_fnuz, hipblasLtBfloat16, float>{}(arg);
        }
        /*
        else if(Ti == HIP_R_8I && To == HIP_R_8I && Tc == HIPBLAS_COMPUTE_32I)
        {
            return TEST<hipblasLtInt8, hipblasLtInt8, int32_t>{}(arg);
        }
        */
        else if(TiA == HIP_R_8I && To == HIP_R_32I && Tc == HIPBLAS_COMPUTE_32I)
        {
            return TEST<hipblasLtInt8, hipblasLtInt8, int32_t, int32_t>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblaslt_f8_fnuz,
                        hipblasLtHalf,
                        hipblaslt_f8_fnuz,
                        float,
                        hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblasLtHalf,
                        hipblaslt_f8_fnuz,
                        hipblaslt_f8_fnuz,
                        float,
                        hipblasLtHalf,
                        hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F
                && To == HIP_R_16F && Tc == HIPBLAS_COMPUTE_32F_FAST_16F && case2 != nullptr)
        {
		return TEST<hipblaslt_f8_fnuz, hipblasLtHalf, hipblasLtHalf, float, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz>{}(
                    arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
		return TEST<hipblaslt_f8_fnuz, hipblasLtHalf, hipblasLtHalf, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8_fnuz, hipblasLtHalf, float, hipblasLtHalf, hipblasLtHalf>{}(
                arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblasLtHalf, float, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8_fnuz, float, float, hipblasLtHalf, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz,
                        hipblasLtHalf,
                        hipblaslt_f8_fnuz,
                        float,
                        hipblaslt_f8_fnuz,
                        hipblaslt_f8_fnuz>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_8F_E4M3_FNUZ
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblasLtHalf,
                        hipblaslt_f8_fnuz,
                        hipblaslt_f8_fnuz,
                        float,
                        hipblaslt_f8_fnuz>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz,
                        hipblasLtHalf,
                        hipblasLtHalf,
                        float,
                        hipblaslt_f8_fnuz,
                        hipblaslt_f8_fnuz>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblasLtHalf,
                        hipblaslt_f8_fnuz,
                        hipblasLtHalf,
                        float,
                        hipblaslt_f8_fnuz>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblasLtHalf, float, float, hipblaslt_f8_fnuz, hipblaslt_f8_fnuz>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8_fnuz, float, float, hipblaslt_f8_fnuz>{}(arg);
        }
    }
    return TEST<void>{}(arg);
}
