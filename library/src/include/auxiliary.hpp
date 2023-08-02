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

#include <hipblaslt/hipblaslt.h>

HIPBLASLT_EXPORT
constexpr const char* hipblas_status_to_string(hipblasStatus_t status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(HIPBLAS_STATUS_SUCCESS);
        CASE(HIPBLAS_STATUS_NOT_INITIALIZED);
        CASE(HIPBLAS_STATUS_ALLOC_FAILED);
        CASE(HIPBLAS_STATUS_INVALID_VALUE);
        CASE(HIPBLAS_STATUS_MAPPING_ERROR);
        CASE(HIPBLAS_STATUS_EXECUTION_FAILED);
        CASE(HIPBLAS_STATUS_INTERNAL_ERROR);
        CASE(HIPBLAS_STATUS_NOT_SUPPORTED);
        CASE(HIPBLAS_STATUS_ARCH_MISMATCH);
        CASE(HIPBLAS_STATUS_INVALID_ENUM);
        CASE(HIPBLAS_STATUS_UNKNOWN);
        CASE(HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are
    // missing from our switch. If the value is not a valid hipblasStatus_t, we
    // return this string.
    return "<undefined hipblasStatus_t value>";
}

HIPBLASLT_EXPORT
constexpr const char* hipblas_operation_to_string(hipblasOperation_t value)
{
    switch(value)
    {
    case HIPBLAS_OP_N:
        return "N";
    case HIPBLAS_OP_T:
        return "T";
    case HIPBLAS_OP_C:
        return "C";
    }
    return "invalid";
}

HIPBLASLT_EXPORT
constexpr hipblasOperation_t char_to_hipblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return HIPBLAS_OP_N;
    case 'T':
    case 't':
        return HIPBLAS_OP_T;
    case 'C':
    case 'c':
        return HIPBLAS_OP_C;
    default:
        return static_cast<hipblasOperation_t>(0);
    }
}

// return precision string for hipblasltDatatype_t
HIPBLASLT_EXPORT
constexpr const char* hipblaslt_datatype_to_string(hipblasltDatatype_t type)
{
    switch(type)
    {
    case HIPBLASLT_R_32F:
        return "f32_r";
    case HIPBLASLT_R_16F:
        return "f16_r";
    case HIPBLASLT_R_16B:
        return "bf16_r";
    case HIPBLASLT_R_8I:
        return "i8_r";
    case HIPBLASLT_R_8F_E4M3:
        return "f8_r";
    case HIPBLASLT_R_8F_E5M2:
        return "bf8_r";
    default:
        return "non-supported type";
    }
    return "invalid";
}

// return precision string for hipblasltDatatype_t
HIPBLASLT_EXPORT
constexpr const char* hipblaslt_computetype_to_string(hipblasLtComputeType_t type)
{
    switch(type)
    {
    case HIPBLASLT_COMPUTE_F32:
        return "f32_r";
    case HIPBLASLT_COMPUTE_F32_FAST_XF32:
        return "xf32_r";
    }
    return "invalid";
}

// clang-format off
HIPBLASLT_EXPORT
constexpr hipblasltDatatype_t string_to_hipblaslt_datatype(const std::string& value)
{
    return
        value == "f32_r" || value == "s" ? HIPBLASLT_R_32F  :
        value == "f16_r" || value == "h" ? HIPBLASLT_R_16F  :
        value == "bf16_r"                ? HIPBLASLT_R_16B  :
        value == "f8_r"                ? HIPBLASLT_R_8F_E4M3  :
        value == "bf8_r"                ? HIPBLASLT_R_8F_E5M2  :
        static_cast<hipblasltDatatype_t>(0);
}

HIPBLASLT_EXPORT
constexpr hipblasLtComputeType_t string_to_hipblaslt_computetype(const std::string& value)
{
    return
        value == "f32_r" || value == "s" ? HIPBLASLT_COMPUTE_F32  :
        value == "xf32_r" || value == "x" ? HIPBLASLT_COMPUTE_F32_FAST_XF32 :
        static_cast<hipblasLtComputeType_t>(0);
}
// clang-format on

/*********************************************************************************************************
 * \brief The main structure for Numerical checking to detect numerical
 *abnormalities such as NaN/zero/Inf
 *********************************************************************************************************/
typedef struct hipblaslt_check_numerics_s
{
    // Set to true if there is a NaN in the vector/matrix
    bool has_NaN = false;

    // Set to true if there is a zero in the vector/matrix
    bool has_zero = false;

    // Set to true if there is an Infinity in the vector/matrix
    bool has_Inf = false;
} hipblaslt_check_numerics_t;

/*******************************************************************************
 * \brief  returns true if arg is NaN
 ********************************************************************************/
template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipblaslt_isnan(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipblaslt_isnan(T arg)
{
    return std::isnan(arg);
}

__host__ __device__ inline bool hipblaslt_isnan(hipblasLtHalf arg)
{
    union
    {
        hipblasLtHalf fp;
        uint16_t      data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) != 0;
}

/*******************************************************************************
 * \brief  returns true if arg is Infinity
 ********************************************************************************/

template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipblaslt_isinf(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool hipblaslt_isinf(T arg)
{
    return std::isinf(arg);
}

__host__ __device__ inline bool hipblaslt_isinf(hipblasLtHalf arg)
{
    union
    {
        hipblasLtHalf fp;
        uint16_t      data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) == 0;
}

/*******************************************************************************
 * \brief  returns true if arg is zero
 ********************************************************************************/

template <typename T>
__host__ __device__ inline bool hipblaslt_iszero(T arg)
{
    return arg == 0;
}
