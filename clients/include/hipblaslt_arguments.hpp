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

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_math.hpp"
#include "hipblaslt_ostream.hpp"
#include <cstddef>
#include <hipblaslt/hipblaslt.h>
#include <istream>
#include <map>
#include <ostream>
#include <tuple>

#define HIPBLASLT_MAX_REQUESTED_SOLUTION_NUM 65536

// Predeclare enumerator
enum hipblaslt_argument : int;

/*! \brief device matches pattern */
bool gpu_arch_match(const std::string& gpu_arch, const char pattern[4]);

/***************************************************************************
 *! \brief Class used to parse command arguments in both client & gtest    *
 * WARNING: If this data is changed, then hipblaslt_common.yaml must also be *
 * changed.                                                                *
 ***************************************************************************/
constexpr std::size_t MAX_SUPPORTED_NUM_PROBLEMS{32};
struct Arguments
{
    /*************************************************************************
     *                    Beginning Of Arguments                             *
     *************************************************************************/

    char function[64];
    char name[64];
    char category[64];
    char known_bug_platforms[64];

    // 32bit
    float alpha;
    float beta;

    int64_t stride_a[MAX_SUPPORTED_NUM_PROBLEMS]; //  stride_a > transA == 'N' ? lda * K : lda * M
    int64_t stride_b[MAX_SUPPORTED_NUM_PROBLEMS]; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    int64_t stride_c[MAX_SUPPORTED_NUM_PROBLEMS]; //  stride_c > ldc * N
    int64_t stride_d[MAX_SUPPORTED_NUM_PROBLEMS]; //  stride_d > ldd * N
    int64_t stride_e[MAX_SUPPORTED_NUM_PROBLEMS]; //  stride_e > lde * N

    size_t user_allocated_workspace;

    int64_t M[MAX_SUPPORTED_NUM_PROBLEMS];
    int64_t N[MAX_SUPPORTED_NUM_PROBLEMS];
    int64_t K[MAX_SUPPORTED_NUM_PROBLEMS];

    int64_t lda[MAX_SUPPORTED_NUM_PROBLEMS];
    int64_t ldb[MAX_SUPPORTED_NUM_PROBLEMS];
    int64_t ldc[MAX_SUPPORTED_NUM_PROBLEMS];
    int64_t ldd[MAX_SUPPORTED_NUM_PROBLEMS];
    int64_t lde[MAX_SUPPORTED_NUM_PROBLEMS];

    int32_t batch_count;

    int32_t iters;
    int32_t cold_iters;

    uint32_t algo;
    int32_t  solution_index;
    int32_t  requested_solution_num;

    hipblasltDatatype_t    a_type;
    hipblasltDatatype_t    b_type;
    hipblasltDatatype_t    c_type;
    hipblasltDatatype_t    d_type;
    hipblasLtComputeType_t compute_type;
    hipblasltDatatype_t    scale_type;

    hipblaslt_initialization initialization;

    // the gpu arch string after "gfx" for which the test is valid
    // '?' is wildcard char, empty string is default as valid on all
    char gpu_arch[4];

    // memory padding for testing write out of bounds
    uint32_t pad;
    int32_t  grouped_gemm;

    // 16 bit
    uint16_t threads;
    uint16_t streams;

    // bytes
    uint8_t devices;

    int8_t norm_check;
    int8_t unit_check;
    int8_t timing;

    char transA;
    char transB;

    hipblaslt_activation_type activation_type;
    float                     activation_arg1; // threshold when activation type is relu
    float                     activation_arg2; // upperbound when activation type is relu

    hipblasltDatatype_t   bias_type;
    hipblaslt_bias_source bias_source;
    bool                  bias_vector;
    bool                  scaleA;
    bool                  scaleB;
    bool                  scaleC;
    bool                  scaleD;
    bool                  scaleE;
    bool                  scaleAlpha_vector;
    bool                  amaxScaleA;
    bool                  amaxScaleB;
    bool                  c_noalias_d;
    bool                  HMM;
    bool                  use_e;
    bool                  gradient;
    bool                  norm_check_assert;

    // API related
    bool    use_ext;
    bool    use_ext_setproblem;
    int     algo_method; // 0 for getheuristic, 1 for get all algos, 2 for algo index
    bool    use_user_args;
    int32_t rotating;
    bool use_gpu_timer;

    // tuning
    int32_t gsu_vector[MAX_SUPPORTED_NUM_PROBLEMS]; // This is for client
    int32_t wgm_vector[MAX_SUPPORTED_NUM_PROBLEMS]; // This is for client

    // print
    bool print_solution_found;
    bool print_kernel_info;

    /*************************************************************************
     *                     End Of Arguments                                  *
     *************************************************************************/

    // we don't have a constructor as the python generated data is used for memory initializer for testing
    // thus this is for other use where we want defaults to match those specified in hipblaslt_common.yaml
    void init();

    // clang-format off

// Generic macro which operates over the list of arguments in order of declaration
#define FOR_EACH_ARGUMENT(OPER, SEP) \
    OPER(function) SEP               \
    OPER(name) SEP                   \
    OPER(category) SEP               \
    OPER(known_bug_platforms) SEP    \
    OPER(alpha) SEP                  \
    OPER(beta) SEP                   \
    OPER(stride_a) SEP               \
    OPER(stride_b) SEP               \
    OPER(stride_c) SEP               \
    OPER(stride_d) SEP               \
    OPER(stride_e) SEP               \
    OPER(user_allocated_workspace) SEP \
    OPER(M) SEP                      \
    OPER(N) SEP                      \
    OPER(K) SEP                      \
    OPER(lda) SEP                    \
    OPER(ldb) SEP                    \
    OPER(ldc) SEP                    \
    OPER(ldd) SEP                    \
    OPER(lde) SEP                    \
    OPER(batch_count) SEP            \
    OPER(iters) SEP                  \
    OPER(cold_iters) SEP             \
    OPER(algo) SEP                   \
    OPER(solution_index) SEP         \
    OPER(requested_solution_num) SEP \
    OPER(a_type) SEP                 \
    OPER(b_type) SEP                 \
    OPER(c_type) SEP                 \
    OPER(d_type) SEP                 \
    OPER(compute_type) SEP           \
    OPER(scale_type) SEP             \
    OPER(initialization) SEP         \
    OPER(gpu_arch) SEP               \
    OPER(pad) SEP                    \
    OPER(grouped_gemm) SEP           \
    OPER(threads) SEP                \
    OPER(streams) SEP                \
    OPER(devices) SEP                \
    OPER(norm_check) SEP             \
    OPER(unit_check) SEP             \
    OPER(timing) SEP                 \
    OPER(transA) SEP                 \
    OPER(transB) SEP                 \
    OPER(activation_type) SEP        \
    OPER(activation_arg1) SEP        \
    OPER(activation_arg2) SEP        \
    OPER(bias_type) SEP              \
    OPER(bias_source) SEP            \
    OPER(bias_vector) SEP            \
    OPER(scaleA) SEP                 \
    OPER(scaleB) SEP                 \
    OPER(scaleC) SEP                 \
    OPER(scaleD) SEP                 \
    OPER(scaleE) SEP                 \
    OPER(scaleAlpha_vector) SEP      \
    OPER(amaxScaleA) SEP             \
    OPER(amaxScaleB) SEP             \
    OPER(c_noalias_d) SEP            \
    OPER(HMM) SEP                    \
    OPER(use_e) SEP                  \
    OPER(gradient) SEP               \
    OPER(norm_check_assert) SEP      \
    OPER(use_ext) SEP                \
    OPER(use_ext_setproblem) SEP     \
    OPER(algo_method) SEP            \
    OPER(use_user_args) SEP          \
    OPER(rotating) SEP               \
    OPER(use_gpu_timer) SEP          \
    OPER(gsu_vector) SEP             \
    OPER(wgm_vector) SEP             \
    OPER(print_solution_found) SEP   \
    OPER(print_kernel_info) SEP

    // clang-format on

    // Validate input format.
    static void validate(std::istream& ifs);

    // Function to print Arguments out to stream in YAML format
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& str,
                                                  const Arguments&            arg);

    // Google Tests uses this with std:ostream automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg);

    // Function to read Arguments data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg);

#ifdef WIN32
    // Clang specific code
    template <typename T>
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  std::pair<char const*, T>   p);

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream&                 os,
                                                  std::pair<char const*, hipblasltDatatype_t> p);

    friend hipblaslt_internal_ostream&
        operator<<(hipblaslt_internal_ostream&                      os,
                   std::pair<char const*, hipblaslt_initialization> p);

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream&  os,
                                                  std::pair<char const*, bool> p);
// End of Clang specific code
#endif

    // Convert (alpha, alphai) and (beta, betai) to a particular type
    // Return alpha, beta adjusted to 0 for when they are NaN
    template <typename T>
    T get_alpha() const
    {
        return alpha_isnan<T>() ? T(0) : convert_alpha_beta<T>(alpha);
    }

    template <typename T>
    T get_beta() const
    {
        return beta_isnan<T>() ? T(0) : convert_alpha_beta<T>(beta);
    }

    template <typename T>
    bool alpha_isnan() const
    {
        return hipblaslt_isnan(alpha);
    }

    template <typename T>
    bool beta_isnan() const
    {
        return hipblaslt_isnan(beta);
    }

private:
    template <typename T, typename U>
    static T convert_alpha_beta(U r)
    {
        return T(r);
    }
};

// We make sure that the Arguments struct is C-compatible
static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

// Arguments enumerators
// Create
//     enum hipblaslt_argument : int {e_M, e_N, e_K, e_KL, ... };
// There is an enum value for each case in FOR_EACH_ARGUMENT.
//
#define CREATE_ENUM(NAME) e_##NAME,
enum hipblaslt_argument : int
{
    FOR_EACH_ARGUMENT(CREATE_ENUM, )
};
#undef CREATE_ENUM

#if __clang__
#define HIPBLASLT_CLANG_STATIC static
#else
#define HIPBLASLT_CLANG_STATIC
#endif

// ArgumentsHelper contains a templated lambda apply<> where there is a template
// specialization for each line in the CPP macro FOR_EACH_ARGUMENT. For example,
// the first lambda is:  apply<e_M> = [](auto&& func, const Arguments& arg, auto){func("M", arg.m);};
// This lambda can be used to print "M" and arg.m.
//
// alpha and beta are specialized separately, because they need to use get_alpha() or get_beta().
// To prevent multiple definitions of specializations for alpha and beta, the hipblaslt_argument
// enum for alpha and beta are changed to hipblaslt_argument(-1) and hipblaslt_argument(-2) during
// the FOR_EACH_ARGUMENT loop. Those out-of-range enum values are not used except here, and are
// only used so that the FOR_EACH_ARGUMENT loop can be used to loop over all of the arguments.

#if __cplusplus >= 201703L
// C++17
// ArgumentsHelper contains a templated lambda apply<> where there is a template
// specialization for each line in the CPP macro FOR_EACH_ARGUMENT. For example,
// the first lambda is:  apply<e_M> = [](auto&& func, const Arguments& arg, auto){func("M", arg.m)}
// This lambda can be used to print "M" and arg.m
namespace ArgumentsHelper
{
    template <hipblaslt_argument>
    static constexpr auto apply = nullptr;

    // Macro defining specializations for specific arguments
    // e_alpha and e_beta get turned into negative sentinel value specializations
    // clang-format off
#define APPLY(NAME)                                                                         \
    template <>                                                                             \
    HIPBLASLT_CLANG_STATIC constexpr auto                                                   \
        apply<e_##NAME == e_M ? hipblaslt_argument(-1) :                                    \
              e_##NAME == e_N ? hipblaslt_argument(-2) :                                    \
              e_##NAME == e_K ? hipblaslt_argument(-3) :                                    \
              e_##NAME == e_lda ? hipblaslt_argument(-4) :                                  \
              e_##NAME == e_stride_a ? hipblaslt_argument(-5) :                             \
              e_##NAME == e_ldb ? hipblaslt_argument(-6) :                                  \
              e_##NAME == e_stride_b ? hipblaslt_argument(-7) :                             \
              e_##NAME == e_ldc ? hipblaslt_argument(-8) :                                  \
              e_##NAME == e_stride_c ? hipblaslt_argument(-9) :                             \
              e_##NAME == e_ldd ? hipblaslt_argument(-10) :                                 \
              e_##NAME == e_stride_d ? hipblaslt_argument(-11) :                            \
              e_##NAME == e_lde ? hipblaslt_argument(-12) :                                 \
              e_##NAME == e_stride_e ? hipblaslt_argument(-13) :                            \
              e_##NAME == e_alpha ? hipblaslt_argument(-14) :                               \
              e_##NAME == e_beta ? hipblaslt_argument(-15) :                                \
              e_##NAME == e_rotating ? hipblaslt_argument(-16) : e_##NAME> = \
            [](auto&& func, const Arguments& arg, auto) { func(#NAME, arg.NAME); }

    // Specialize apply for each Argument
    FOR_EACH_ARGUMENT(APPLY, ;);

    // Specialization for e_M
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_M> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("m", arg.M[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.M[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.M[i]);
                }
                s += ")";
                func("m", s.c_str());
            }
        };

    // Specialization for e_N
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_N> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("n", arg.N[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.N[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.N[i]);
                }
                s += ")";
                func("n", s.c_str());
            }
        };

    // Specialization for e_K
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_K> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("k", arg.K[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.K[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.K[i]);
                }
                s += ")";
                func("k", s.c_str());
            }
        };

    // Specialization for e_lda
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_lda> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("lda", arg.lda[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.lda[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.lda[i]);
                }
                s += ")";
                func("lda", s.c_str());
            }
        };

    // Specialization for e_stride_a
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_stride_a> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("stride_a", arg.stride_a[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.stride_a[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.stride_a[i]);
                }
                s += ")";
                func("stride_a", s.c_str());
            }
        };

    // Specialization for e_ldb
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_ldb> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("ldb", arg.ldb[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.ldb[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.ldb[i]);
                }
                s += ")";
                func("ldb", s.c_str());
            }
        };

    // Specialization for e_stride_b
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_stride_b> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("stride_b", arg.stride_b[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.stride_b[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.stride_b[i]);
                }
                s += ")";
                func("stride_b", s.c_str());
            }
        };

    // Specialization for e_ldc
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_ldc> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("ldc", arg.ldc[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.ldc[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.ldc[i]);
                }
                s += ")";
                func("ldc", s.c_str());
            }
        };

    // Specialization for e_stride_c
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_stride_c> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("stride_c", arg.stride_c[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.stride_c[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.stride_c[i]);
                }
                s += ")";
                func("stride_c", s.c_str());
            }
        };

    // Specialization for e_ldd
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_ldd> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("ldd", arg.ldd[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.ldd[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.ldd[i]);
                }
                s += ")";
                func("ldd", s.c_str());
            }
        };

    // Specialization for e_stride_d
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_stride_d> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("stride_d", arg.stride_d[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.stride_d[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.stride_d[i]);
                }
                s += ")";
                func("stride_d", s.c_str());
            }
        };

    // Specialization for e_lde
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_lde> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("lde", arg.lde[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.lde[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.lde[i]);
                }
                s += ")";
                func("lde", s.c_str());
            }
        };

    // Specialization for e_stride_e
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_stride_e> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.grouped_gemm <= 1)
            {
                func("stride_e", arg.stride_e[0]);
            }
            else
            {
                std::string s = "(" + std::to_string(arg.stride_e[0]);
                for(size_t i = 1; i < arg.grouped_gemm; i++)
                {
                    s += "," + std::to_string(arg.stride_e[i]);
                }
                s += ")";
                func("stride_e", s.c_str());
            }
        };

    // Specialization for e_alpha
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_alpha> =
        [](auto&& func, const Arguments& arg, auto T) {
            func("alpha", arg.get_alpha<decltype(T)>());
        };

    // Specialization for e_beta
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_beta> =
        [](auto&& func, const Arguments& arg, auto T) {
            func("beta", arg.get_beta<decltype(T)>());
        };

    // Specialization for e_rotating
    template <>
    HIPBLASLT_CLANG_STATIC constexpr auto apply<e_rotating> =
        [](auto&& func, const Arguments& arg, auto T) {
            if(arg.rotating > 0)
                func("rotating_buffer", arg.rotating);
        };
};
// clang-format on

#else
#error "Unsupported C++ version"
#endif

#undef APPLY
