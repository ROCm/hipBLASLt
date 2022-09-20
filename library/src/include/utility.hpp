/*! \file */
/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "handle.h"
#include "logging.h"
#include <algorithm>
#include <exception>

#pragma STDC CX_LIMITED_RANGE ON

inline bool isAligned(const void *pointer, size_t byte_count) {
  return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

// return precision string for rocblaslt_datatype
constexpr const char *rocblaslt_datatype_string(hipDataType type) {
  switch (type) {
  case HIP_R_16F:
    return "f16_r";
  case HIP_R_32F:
    return "f32_r";
  default:
    return "invalidType";
  }
}

// return precision string for rocblaslt_compute_type
constexpr const char *
rocblaslt_compute_type_string(rocblaslt_compute_type type) {
  switch (type) {
  case rocblaslt_compute_f32:
    return "f32";
  default:
    return "invalidType";
  }
}

constexpr const char *rocblaslt_transpose_letter(rocblaslt_operation op) {
  switch (op) {
  case ROCBLASLT_OP_N:
    return "N";
  case ROCBLASLT_OP_T:
    return "T";
  default:
    return "invalidTranspose";
  }
}
// Convert rocblaslt_status to string
constexpr const char *rocblaslt_status_to_string(rocblaslt_status status) {
#define CASE(x)                                                                \
  case x:                                                                      \
    return #x
  switch (status) {
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
// template <> constexpr char rocblaslt_precision_string<rocblaslt_bfloat16 >[]
// = "bf16_r";
template <>
static constexpr char rocblaslt_precision_string<rocblaslt_half>[] = "f16_r";
template <> static constexpr char rocblaslt_precision_string<float>[] = "f32_r";
template <>
static constexpr char rocblaslt_precision_string<double>[] = "f64_r";
template <> static constexpr char rocblaslt_precision_string<int8_t>[] = "i8_r";
template <>
static constexpr char rocblaslt_precision_string<uint8_t>[] = "u8_r";
template <>
static constexpr char rocblaslt_precision_string<int32_t>[] = "i32_r";
template <>
static constexpr char rocblaslt_precision_string<uint32_t>[] = "u32_r";

// Return the leftmost significant bit position
#if defined(rocblaslt_ILP64)
static inline rocblaslt_int rocblaslt_clz(rocblaslt_int n) {
  return 64 - __builtin_clzll(n);
}
#else
static inline rocblaslt_int rocblaslt_clz(rocblaslt_int n) {
  return 32 - __builtin_clz(n);
}
#endif

// if trace logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_trace) == true
// then
// log_function will call log_arguments to log function
// arguments with a comma separator
template <typename H, typename... Ts>
void log_trace(rocblaslt_handle handle, H head, Ts &&...xs) {
  if (nullptr != handle) {
    if (handle->layer_mode & rocblaslt_layer_mode_log_trace) {
      std::string comma_separator = ",";

      std::ostream *os = handle->log_trace_os;
      log_arguments(*os, comma_separator, head, std::forward<Ts>(xs)...);
    }
  }
}

// if bench logging is turned on with
// (handle->layer_mode & rocblaslt_layer_mode_log_bench) == true
// then
// log_bench will call log_arguments to log a string that
// can be input to the executable rocblaslt-bench.
template <typename H, typename... Ts>
void log_bench(rocblaslt_handle handle, H head, std::string precision,
               Ts &&...xs) {
  if (nullptr != handle) {
    if (handle->layer_mode & rocblaslt_layer_mode_log_bench) {
      std::string space_separator = " ";

      std::ostream *os = handle->log_bench_os;
      log_arguments(*os, space_separator, head, precision,
                    std::forward<Ts>(xs)...);
    }
  }
}

// Trace log scalar values pointed to by pointer
template <typename T> T log_trace_scalar_value(const T *value) {
  return value ? *value : std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
T log_trace_scalar_value(rocblaslt_handle handle, const T *value) {
  T host;
  if (value && handle->pointer_mode == rocblaslt_pointer_mode_device) {
    hipMemcpy(&host, value, sizeof(host), hipMemcpyDeviceToHost);
    value = &host;
  }
  return log_trace_scalar_value(value);
}

#define LOG_TRACE_SCALAR_VALUE(handle, value)                                  \
  log_trace_scalar_value(handle, value)

// Bench log scalar values pointed to by pointer
template <typename T> T log_bench_scalar_value(const T *value) {
  return (value ? *value : std::numeric_limits<T>::quiet_NaN());
}

template <typename T>
T log_bench_scalar_value(rocblaslt_handle handle, const T *value) {
  T host;
  if (value && handle->pointer_mode == rocblaslt_pointer_mode_device) {
    hipMemcpy(&host, value, sizeof(host), hipMemcpyDeviceToHost);
    value = &host;
  }
  return log_bench_scalar_value(value);
}

#define LOG_BENCH_SCALAR_VALUE(handle, name)                                   \
  log_bench_scalar_value(handle, name)

// replaces X in string with s, d, c, z or h depending on typename T
template <typename T> std::string replaceX(std::string input_string) {
  if (std::is_same<T, float>::value) {
    std::replace(input_string.begin(), input_string.end(), 'X', 's');
  } else if (std::is_same<T, double>::value) {
    std::replace(input_string.begin(), input_string.end(), 'X', 'd');
  }
  return input_string;
}

//
// These macros can be redefined if the developer includes src/include/debug.h
//
#define ROCBLASLT_DEBUG_VERBOSE(msg__) (void)0
#define ROCBLASLT_RETURN_STATUS(token__) return rocblaslt_status_##token__

// Convert the current C++ exception to rocblaslt_status
// This allows extern "C" functions to return this function in a catch(...)
// block while converting all C++ exceptions to an equivalent rocblaslt_status
// here
inline rocblaslt_status exception_to_rocblaslt_status(
    std::exception_ptr e = std::current_exception()) try {
  if (e)
    std::rethrow_exception(e);
  return rocblaslt_status_success;
} catch (const rocblaslt_status &status) {
  return status;
} catch (const std::bad_alloc &) {
  return rocblaslt_status_memory_error;
} catch (...) {
  return rocblaslt_status_internal_error;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(T x) {
  return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar_device_host(const T *xp) {
  return *xp;
}

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(T x) {
  return static_cast<T>(0);
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T zero_scalar_device_host(const T *xp) {
  return static_cast<T>(0);
}

//
// Provide some utility methods for enums.
//
struct rocblaslt_enum_utils {
  template <typename U> static inline bool is_invalid(U value_);
};

template <>
inline bool rocblaslt_enum_utils::is_invalid(rocblaslt_compute_type value_) {
  switch (value_) {
  case rocblaslt_compute_f32:
    return false;
  default:
    return true;
  }
};

template <>
inline bool rocblaslt_enum_utils::is_invalid(
    rocblaslt_matmul_preference_attributes value_) {
  switch (value_) {
  case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
  case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
    return false;
  default:
    return true;
  }
};

template <typename T> struct floating_traits { using data_t = T; };

template <typename T>
using floating_data_t = typename floating_traits<T>::data_t;

// Internal use, whether Tensile supports ldc != ldd
// We assume true if the value is greater than or equal to 906
bool rocblaslt_internal_tensile_supports_ldc_ne_ldd(rocblaslt_handle handle);

// for internal use during testing, fetch arch name
std::string rocblaslt_internal_get_arch_name();

#endif // UTILITY_H
