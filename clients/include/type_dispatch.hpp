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

#include "hipblaslt_arguments.hpp"
#include <hipblaslt/hipblaslt.h>

template <typename T>
constexpr auto hipblaslt_type2datatype()
{
    if(std::is_same<T, hipblasLtHalf>{})
        return HIPBLASLT_R_16F;
    if(std::is_same<T, hip_bfloat16>{})
        return HIPBLASLT_R_16B;
    if(std::is_same<T, float>{})
        return HIPBLASLT_R_32F;
    if(std::is_same<T, hipblaslt_f8>{})
        return HIPBLASLT_R_8F_E4M3;
    if(std::is_same<T, hipblaslt_bf8>{})
        return HIPBLASLT_R_8F_E5M2;
    if(std::is_same<T, int32_t>{})
        return HIPBLASLT_R_32I;
    if(std::is_same<T, hipblasLtInt8>{})
        return HIPBLASLT_R_8I;

    return HIPBLASLT_R_16F; // testing purposes we default to f32 ex
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
    case HIPBLASLT_R_16F:
        return TEST<hipblasLtHalf>{}(arg);
    case HIPBLASLT_R_16B:
        return TEST<hip_bfloat16>{}(arg);
    case HIPBLASLT_R_32F:
        return TEST<float>{}(arg);
    case HIPBLASLT_R_8F_E4M3:
        return TEST<hipblaslt_f8>{}(arg);
    case HIPBLASLT_R_8F_E5M2:
        return TEST<hipblaslt_bf8>{}(arg);
    case HIPBLASLT_R_8I:
        return TEST<hipblasLtInt8>{}(arg);
    case HIPBLASLT_R_32I:
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

    if(arg.d_type == To)
    {
        if(TiA == To && TiB == To && To == HIPBLASLT_R_16F && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIPBLASLT_R_16B && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hip_bfloat16, hip_bfloat16, hip_bfloat16, float>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIPBLASLT_R_32F
                && (Tc == HIPBLASLT_COMPUTE_F32 || Tc == HIPBLASLT_COMPUTE_F32_FAST_XF32))
        {
            return TEST<float, float, float, float>{}(arg);
        }
        else if(TiA == To && TiB == To && To == HIPBLASLT_R_64F && (Tc == HIPBLASLT_COMPUTE_F64))
        {
            return TEST<double, double, double, double>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_16F && TiB == HIPBLASLT_R_16F && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblasLtHalf, hipblasLtHalf, float, float>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, float, float>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E5M2 && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_bf8, hipblaslt_f8, float, float>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_8F_E5M2 && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblaslt_bf8, float, float>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_16F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_8F_E4M3
                && To == HIPBLASLT_R_8F_E4M3 && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblaslt_f8, hipblaslt_f8, float>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E5M2 && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_16F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_bf8, hipblaslt_f8, hipblasLtHalf, float>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_8F_E5M2 && To == HIPBLASLT_R_16F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblaslt_bf8, hipblasLtHalf, float>{}(arg);
        }
        /*
        else if(Ti == HIPBLASLT_R_8I && To == HIPBLASLT_R_8I && Tc == HIPBLASLT_COMPUTE_I32)
        {
            return TEST<hipblasLtInt8, hipblasLtInt8, int32_t>{}(arg);
        }
        */
        else if(TiA == HIPBLASLT_R_8I && To == HIPBLASLT_R_32I && Tc == HIPBLASLT_COMPUTE_I32)
        {
            return TEST<hipblasLtInt8, hipblasLtInt8, int32_t, int32_t>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_16F && To == HIPBLASLT_R_8F_E4M3
                && Tc == HIPBLASLT_COMPUTE_F32_FAST_F16)
        {
            return TEST<hipblaslt_f8, hipblasLtHalf, hipblaslt_f8, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_16F && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_8F_E4M3
                && Tc == HIPBLASLT_COMPUTE_F32_FAST_F16)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8, hipblaslt_f8, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_16F && To == HIPBLASLT_R_16F
                && Tc == HIPBLASLT_COMPUTE_F32_FAST_F16)
        {
            return TEST<hipblaslt_f8, hipblasLtHalf, hipblasLtHalf, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_16F && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_16F
                && Tc == HIPBLASLT_COMPUTE_F32_FAST_F16)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8, hipblasLtHalf, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_16F && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32_FAST_F16)
        {
            return TEST<hipblaslt_f8, hipblasLtHalf, float, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_16F && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32_FAST_F16)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8, float, float, hipblasLtHalf>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_16F && To == HIPBLASLT_R_8F_E4M3
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblasLtHalf, hipblaslt_f8, float, hipblaslt_f8>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_16F && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_8F_E4M3
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8, hipblaslt_f8, float, hipblaslt_f8>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_16F && To == HIPBLASLT_R_16F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblasLtHalf, hipblasLtHalf, float, hipblaslt_f8>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_16F && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_16F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8, hipblasLtHalf, float, hipblaslt_f8>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_8F_E4M3 && TiB == HIPBLASLT_R_16F && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblaslt_f8, hipblasLtHalf, float, float, hipblaslt_f8>{}(arg);
        }
        else if(TiA == HIPBLASLT_R_16F && TiB == HIPBLASLT_R_8F_E4M3 && To == HIPBLASLT_R_32F
                && Tc == HIPBLASLT_COMPUTE_F32)
        {
            return TEST<hipblasLtHalf, hipblaslt_f8, float, float, hipblaslt_f8>{}(arg);
        }
    }
    return TEST<void>{}(arg);
}
