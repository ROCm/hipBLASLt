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

#include "datatype_interface.hpp"
#include "hipblaslt_arguments.hpp"
#include <hipblaslt/hipblaslt.h>

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
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        return TEST<hipblaslt_f8>{}(arg);
    case HIP_R_8F_E5M2:
        return TEST<hipblaslt_bf8>{}(arg);
#endif
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
    const auto TiA  = arg.a_type;
    const auto TiB  = arg.b_type;
    auto       To   = arg.c_type;
    auto       Tc   = arg.compute_type;
    auto       TciA = arg.compute_input_typeA;
    auto       TciB = arg.compute_input_typeB;

    // setting compute_type to f16_r will automatically fallback to f32_r
    if(Tc == HIPBLAS_COMPUTE_16F) {
        Tc = HIPBLAS_COMPUTE_32F;
    }

    if(arg.d_type == To)
    {
        if(TiA == To && TiB == To && To == HIP_R_16F && Tc == HIPBLAS_COMPUTE_32F)
        {
            if(TciA == TciB && TciA == HIP_R_16F)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hipblasLtHalf,
                            hipblasLtHalf>{}(arg);
            else if(TciA == TciB && TciA == HIP_R_16BF)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hip_bfloat16,
                            hip_bfloat16>{}(arg);
            else if(TciA == TciB && TciA == HIP_R_8F_E4M3_FNUZ)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hipblaslt_f8_fnuz,
                            hipblaslt_f8_fnuz>{}(arg);
            else if(TciA == HIP_R_8F_E4M3_FNUZ && TciB == HIP_R_8F_E5M2_FNUZ)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hipblaslt_f8_fnuz,
                            hipblaslt_bf8_fnuz>{}(arg);
            else if(TciA == HIP_R_8F_E5M2_FNUZ && TciB == HIP_R_8F_E4M3_FNUZ)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hipblaslt_bf8_fnuz,
                            hipblaslt_f8_fnuz>{}(arg);
#ifdef ROCM_USE_FLOAT8
            else if(TciA == TciB && TciA == HIP_R_8F_E4M3)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hipblaslt_f8,
                            hipblaslt_f8>{}(arg);
            else if(TciA == HIP_R_8F_E4M3 && TciB == HIP_R_8F_E5M2)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hipblaslt_f8,
                            hipblaslt_bf8>{}(arg);
            else if(TciA == HIP_R_8F_E5M2 && TciB == HIP_R_8F_E4M3)
                return TEST<hipblasLtHalf,
                            hipblasLtHalf,
                            hipblasLtHalf,
                            float,
                            hipblaslt_bf8,
                            hipblaslt_f8>{}(arg);
#endif
            else
                return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIP_R_16F && Tc == HIPBLAS_COMPUTE_32F_FAST_16BF)
        {
            return TEST<hipblasLtHalf,
                        hipblasLtHalf,
                        hipblasLtHalf,
                        float,
                        hip_bfloat16,
                        hip_bfloat16>{}(arg);
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
#ifdef ROCM_USE_FLOAT8
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E4M3 && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E4M3 && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_f8, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E5M2 && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_bf8, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E4M3 && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E4M3 && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, hipblasLtBfloat16, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E5M2 && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_bf8, float, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E5M2 && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_bf8, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E5M2 && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_bf8, hipblasLtBfloat16, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E4M3 && To == HIP_R_8F_E4M3
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, hipblaslt_f8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E4M3 && To == HIP_R_8F_E5M2
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, hipblaslt_bf8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E5M2 && To == HIP_R_8F_E4M3
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_bf8, hipblaslt_f8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E5M2 && To == HIP_R_8F_E5M2
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_bf8, hipblaslt_bf8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E4M3 && To == HIP_R_8F_E4M3
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_f8, hipblaslt_f8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E4M3 && To == HIP_R_8F_E5M2
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_f8, hipblaslt_bf8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E5M2 && To == HIP_R_8F_E4M3
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_bf8, hipblaslt_f8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E5M2 && To == HIP_R_8F_E5M2
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_bf8, hipblaslt_bf8, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E4M3 && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_f8, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E5M2 && TiB == HIP_R_8F_E4M3 && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_bf8, hipblaslt_f8, hipblasLtBfloat16, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E5M2 && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_bf8, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3 && TiB == HIP_R_8F_E5M2 && To == HIP_R_16BF
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblaslt_f8, hipblaslt_bf8, hipblasLtBfloat16, float>{}(arg);
        }
#endif
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
        else if(TiA == HIP_R_8I && TiB == HIP_R_8I && To == HIP_R_8I && Tc == HIPBLAS_COMPUTE_32I)
        {
            return TEST<hipblasLtInt8, hipblasLtInt8, hipblasLtInt8, int32_t>{}(arg);
        }
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
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblasLtHalf, hipblasLtHalf, float, hipblasLtHalf>{}(
                arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_16F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblasLtHalf,
                        hipblaslt_f8_fnuz,
                        hipblasLtHalf,
                        float,
                        hipblasLtHalf,
                        hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIP_R_8F_E4M3_FNUZ && TiB == HIP_R_16F && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblaslt_f8_fnuz, hipblasLtHalf, float, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F_FAST_16F)
        {
            return TEST<hipblasLtHalf,
                        hipblaslt_f8_fnuz,
                        float,
                        float,
                        hipblasLtHalf,
                        hipblasLtHalf>{}(arg);
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
            return TEST<hipblaslt_f8_fnuz,
                        hipblasLtHalf,
                        float,
                        float,
                        hipblaslt_f8_fnuz,
                        hipblaslt_f8_fnuz>{}(arg);
        }
        else if(TiA == HIP_R_16F && TiB == HIP_R_8F_E4M3_FNUZ && To == HIP_R_32F
                && Tc == HIPBLAS_COMPUTE_32F)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8_fnuz, float, float, hipblaslt_f8_fnuz>{}(arg);
        }
    }
    return TEST<void>{}(arg);
}
