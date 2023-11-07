/*! \file */
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
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "handle.h"
#include "logging.h"
#include <algorithm>
#include <exception>

#pragma STDC CX_LIMITED_RANGE ON

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

// return precision string for rocblaslt_datatype
constexpr const char* rocblaslt_datatype_string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_16F:
        return "f16_r";
    case HIP_R_32F:
        return "f32_r";
    case HIP_R_16BF:
        return "b16_r";
    case HIP_R_32I:
        return "i32_r";
    case HIP_R_8I:
        return "i8_r";
    default:
        return "invalidType";
    }
}

// return precision string for rocblaslt_compute_type
constexpr const char* rocblaslt_compute_type_string(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f32:
        return "f32";
    case rocblaslt_compute_f32_fast_xf32:
        return "xf32";
    case rocblaslt_compute_i32:
        return "i32";
    default:
        return "invalidType";
    }
}

constexpr const char* rocblaslt_transpose_letter(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return "N";
    case HIPBLAS_OP_T:
        return "T";
    default:
        return "invalidTranspose";
    }
}
// Convert rocblaslt_status to string
constexpr const char* rocblaslt_status_to_string(rocblaslt_status status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(rocblaslt_status_success);
        CASE(rocblaslt_status_invalid_handle);
        CASE(rocblaslt_status_not_implemented);
        CASE(rocblaslt_status_invalid_pointer);
        CASE(rocblaslt_status_invalid_size);
        CASE(rocblaslt_status_memory_error);
        CASE(rocblaslt_status_internal_error);
        CASE(rocblaslt_status_invalid_value);
        CASE(rocblaslt_status_arch_mismatch);
        CASE(rocblaslt_status_zero_pivot);
        CASE(rocblaslt_status_not_initialized);
        CASE(rocblaslt_status_type_mismatch);
        CASE(rocblaslt_status_requires_sorted_storage);
        CASE(rocblaslt_status_continue);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are
    // missing from our switch. If the value is not a valid rocblaslt_status, we
    // return this string.
    return "<undefined rocblaslt_status value>";
}
template <typename>
static constexpr char rocblaslt_precision_string[] = "invalid";
template <>
static constexpr char rocblaslt_precision_string<rocblaslt_bfloat16>[] = "bf16_r";
template <>
static constexpr char rocblaslt_precision_string<rocblaslt_half>[] = "f16_r";
template <>
static constexpr char rocblaslt_precision_string<float>[] = "f32_r";
template <>
static constexpr char rocblaslt_precision_string<double>[] = "f64_r";
template <>
static constexpr char rocblaslt_precision_string<int8_t>[] = "i8_r";
template <>
static constexpr char rocblaslt_precision_string<uint8_t>[] = "u8_r";
template <>
static constexpr char rocblaslt_precision_string<int32_t>[] = "i32_r";
template <>
static constexpr char rocblaslt_precision_string<uint32_t>[] = "u32_r";

std::string prefix(const char* layer, const char* caller);

const char* hipDataType_to_string(hipDataType type);

const char* rocblaslt_compute_type_to_string(rocblaslt_compute_type type);

const char* rocblaslt_matrix_layout_attributes_to_string(rocblaslt_matrix_layout_attribute_ type);

const char* rocblaslt_matmul_desc_attributes_to_string(rocblaslt_matmul_desc_attributes type);

const char* hipblasOperation_to_string(hipblasOperation_t op);

const char* rocblaslt_layer_mode2string(rocblaslt_layer_mode layer_mode);

const char* rocblaslt_epilogue_to_string(rocblaslt_epilogue epilogue);

std::string rocblaslt_matrix_layout_to_string(rocblaslt_matrix_layout mat);

std::string rocblaslt_matmul_desc_to_string(rocblaslt_matmul_desc matmul_desc);

// Return the leftmost significant bit position
#if defined(rocblaslt_ILP64)
static inline rocblaslt_int rocblaslt_clz(rocblaslt_int n)
{
    return 64 - __builtin_clzll(n);
}
#else
static inline rocblaslt_int rocblaslt_clz(rocblaslt_int n)
{
    return 32 - __builtin_clz(n);
}
#endif
std::ostream* get_logger_os();
uint32_t      get_logger_layer_mode();

template <typename H, typename... Ts>
void log_base(rocblaslt_layer_mode layer_mode, const char* func, H head, Ts&&... xs)
{
    if(get_logger_layer_mode() & layer_mode)
    {
        std::string comma_separator = " ";

        std::ostream* os = get_logger_os();

        std::string prefix_str = prefix(rocblaslt_layer_mode2string(layer_mode), func);

        log_arguments(*os, comma_separator, prefix_str, head, std::forward<Ts>(xs)...);
    }
}

template <typename H, typename... Ts>
void log_error(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_error, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_trace, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_hints) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_hints(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_hints, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_info) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_info(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_info, func, head, std::forward<Ts>(xs)...);
}

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_api) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_api(const char* func, H head, Ts&&... xs)
{
    log_base(rocblaslt_layer_mode_log_api, func, head, std::forward<Ts>(xs)...);
}
// if bench logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocblaslt-bench.
template <typename H, typename... Ts>
void log_bench(const char* func, H head, std::string precision, Ts&&... xs)
{
    // TODO
    // if(nullptr != handle && nullptr != handle->log_bench_os)
    //{
    //    if(handle->log_bench)
    //    {
    //        std::string comma_separator = " ";

    //        std::ostream* os = handle->log_bench_os;

    //        std::string prefix_str = prefix("Bench", func);

    //        log_arguments(*os, comma_separator, prefix_str, head,
    //        std::forward<Ts>(xs)...);
    //    }
    //}
}
// Convert the current C++ exception to rocblaslt_status
// This allows extern "C" functions to return this function in a catch(...)
// block while converting all C++ exceptions to an equivalent rocblaslt_status
// here
inline rocblaslt_status exception_to_rocblaslt_status(std::exception_ptr e
                                                      = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return rocblaslt_status_success;
}
catch(const rocblaslt_status& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return rocblaslt_status_memory_error;
}
catch(...)
{
    return rocblaslt_status_internal_error;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(const T* xp)
{
    return *xp;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(T x)
{
    return static_cast<T>(0);
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(const T* xp)
{
    return static_cast<T>(0);
}

//
// Provide some utility methods for enums.
//
struct rocblaslt_enum_utils
{
    template <typename U>
    static inline bool is_invalid(U value_);
};

template <>
inline bool rocblaslt_enum_utils::is_invalid(rocblaslt_compute_type value_)
{
    switch(value_)
    {
    case rocblaslt_compute_f32:
    case rocblaslt_compute_f32_fast_xf32:
    case rocblaslt_compute_i32:
        return false;
    default:
        return true;
    }
};

template <>
inline bool rocblaslt_enum_utils::is_invalid(rocblaslt_matmul_preference_attributes value_)
{
    switch(value_)
    {
    case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
    case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
        return false;
    default:
        return true;
    }
};

inline bool is_grad_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_DGELU:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
    case ROCBLASLT_EPILOGUE_BGRADA:
    case ROCBLASLT_EPILOGUE_BGRADB:
        return true;
    default:
        return false;
    }
};

inline bool is_e_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_DGELU:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
    case ROCBLASLT_EPILOGUE_GELU_AUX:
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
        return true;
    default:
        return false;
    }
};

inline bool is_bias_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_BIAS:
    case ROCBLASLT_EPILOGUE_RELU_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
    case ROCBLASLT_EPILOGUE_BGRADA:
    case ROCBLASLT_EPILOGUE_BGRADB:
        return true;
    default:
        return false;
    }
};

inline bool is_act_enabled(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_RELU:
    case ROCBLASLT_EPILOGUE_RELU_BIAS:
    case ROCBLASLT_EPILOGUE_GELU:
    case ROCBLASLT_EPILOGUE_GELU_BIAS:
    case ROCBLASLT_EPILOGUE_GELU_AUX:
    case ROCBLASLT_EPILOGUE_GELU_AUX_BIAS:
    case ROCBLASLT_EPILOGUE_DGELU:
    case ROCBLASLT_EPILOGUE_DGELU_BGRAD:
        return true;
    case ROCBLASLT_EPILOGUE_DEFAULT:
    case ROCBLASLT_EPILOGUE_BIAS:
    default:
        return false;
    }
};

inline bool is_biasSrc_AB(rocblaslt_epilogue value_)
{
    switch(value_)
    {
    case ROCBLASLT_EPILOGUE_BGRADA:
    case ROCBLASLT_EPILOGUE_BGRADB:
        return true;
    default:
        return false;
    }
};

template <typename T>
struct floating_traits
{
    using data_t = T;
};

template <typename T>
using floating_data_t = typename floating_traits<T>::data_t;

// Internal use, whether Tensile supports ldc != ldd
// We assume true if the value is greater than or equal to 906
bool rocblaslt_internal_tensile_supports_ldc_ne_ldd(rocblaslt_handle handle);

// for internal use during testing, fetch arch name
//std::string rocblaslt_internal_get_arch_name();

#endif // UTILITY_H
