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

#include "hipblaslt_math.hpp"
#include <cinttypes>
#include <hipblaslt/hipblaslt.h>
#include <random>
#include <thread>
#include <type_traits>

/* ============================================================================================ */
// Random number generator
using hipblaslt_rng_t = std::mt19937;

extern hipblaslt_rng_t g_hipblaslt_seed;
extern std::thread::id g_main_thread_id;

extern thread_local hipblaslt_rng_t t_hipblaslt_rng;
extern thread_local int             t_hipblaslt_rand_idx;

// optimized helper
float hipblaslt_uniform_int_1_10();

void hipblaslt_uniform_int_1_10_run_float(float* ptr, size_t num);
void hipblaslt_uniform_int_1_10_run_double(double* ptr, size_t num);

// For the main thread, we use g_hipblaslt_seed; for other threads, we start with a different seed but
// deterministically based on the thread id's hash function.
inline hipblaslt_rng_t get_seed()
{
    auto tid = std::this_thread::get_id();
    return tid == g_main_thread_id ? g_hipblaslt_seed
                                   : hipblaslt_rng_t(std::hash<std::thread::id>{}(tid));
}

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void hipblaslt_seedrand()
{
    t_hipblaslt_rng      = get_seed();
    t_hipblaslt_rand_idx = 0;
}

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */
class hipblaslt_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union
        {
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(t_hipblaslt_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(t_hipblaslt_rng);
    }

    // Random signed char
    explicit operator signed char()
    {
        return static_cast<signed char>(std::uniform_int_distribution<int>{}(t_hipblaslt_rng));
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half
    explicit operator hipblasLtHalf()
    {
        return random_nan_data<hipblasLtHalf, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator hip_bfloat16()
    {
        return random_nan_data<hip_bfloat16, uint16_t, 7, 8>();
    }

    // Single NaN float8...
    explicit operator hipblaslt_f8()
    {
        union
        {
            uint8_t      bits;
            hipblaslt_f8 value;
        } x;
        x.bits = 0x80;
        return x.value;
    }

    // Single NaN bfloat8...
    explicit operator hipblaslt_bf8()
    {
        union
        {
            uint8_t       bits;
            hipblaslt_bf8 value;
        } x;
        x.bits = 0x80;
        return x.value;
    }
};

/* ============================================================================================ */
/*! \brief  Random number generator which generates Inf values */
class hipblaslt_inf_rng
{
    // Generate random Inf values
    static unsigned rand2()
    {
        return std::uniform_int_distribution<unsigned>(0, 1)(t_hipblaslt_rng);
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return rand2() ? std::numeric_limits<T>::min : std::numeric_limits<T>::max;
    }
    // Random float
    template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return T(rand2() ? -std::numeric_limits<double>::infinity()
                         : std::numeric_limits<double>::infinity());
    }
};

/* ============================================================================================ */
/*! \brief  Random number generator which generates zero values */
class hipblaslt_zero_rng
{
    // Generate random zero values
    static unsigned rand2()
    {
        return std::uniform_int_distribution<unsigned>(0, 1)(t_hipblaslt_rng);
    }

public:
    // Random integer
    template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return 0;
    }
    // Random float
    template <typename T, std::enable_if_t<!std::is_integral<T>{}, int> = 0>
    explicit operator T()
    {
        return T(rand2() ? -0.0 : 0.0);
    }
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random NaN number */
template <typename T>
inline T random_nan_generator()
{
    return T(hipblaslt_nan_rng{});
}

/*! \brief  generate a random Inf number */
template <typename T>
inline T random_inf_generator()
{
    return T(hipblaslt_inf_rng{});
}

/*! \brief  generate a random Inf number */
template <typename T>
inline T random_zero_generator()
{
    return T(hipblaslt_zero_rng{});
}

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return T(hipblaslt_uniform_int_1_10());
}

template <>
inline float random_generator()
{
    return hipblaslt_uniform_int_1_10();
}

// for hipblasLtHalf, generate float, and convert to hipblasLtHalf
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline hipblasLtHalf random_generator<hipblasLtHalf>()
{
#if defined(__HIP_PLATFORM_HCC__)
#define CAST
#else
#define CAST static_cast<float>
#endif
    return hipblasLtHalf(CAST(std::uniform_int_distribution<int>(-2, 2)(t_hipblaslt_rng)));
};

// for hip_bfloat16, generate float, and convert to hip_bfloat16
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline hip_bfloat16 random_generator<hip_bfloat16>()
{
#if defined(__HIP_PLATFORM_HCC__)
#define CAST
#else
#define CAST static_cast<float>
#endif
    return hip_bfloat16(CAST(std::uniform_int_distribution<int>(-2, 2)(t_hipblaslt_rng)));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return static_cast<int8_t>(
        std::uniform_int_distribution<unsigned short>(1, 3)(t_hipblaslt_rng));
};

/*! \brief  generate a sequence of random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline void random_run_generator(T* ptr, size_t num)
{
    for(size_t i = 0; i < num; i++)
    {
        ptr[i] = random_generator<T>();
    }
}

template <>
inline void random_run_generator<float>(float* ptr, size_t num)
{
    hipblaslt_uniform_int_1_10_run_float(ptr, num);
};

template <>
inline void random_run_generator<double>(double* ptr, size_t num)
{
    hipblaslt_uniform_int_1_10_run_double(ptr, num);
};

// HPL

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return static_cast<T>(std::uniform_real_distribution<double>(-0.5, 0.5)(t_hipblaslt_rng));
}

/*! \brief  generate a random number in [-1.0,1.0] doubles  */
template <>
inline int8_t random_hpl_generator()
{
    auto saturate = [](double val) {
        auto _val = std::nearbyint(val);
        _val      = _val > 127.f ? 127.f : _val < -128.f ? -128.f : _val;
        return _val;
    };

    return static_cast<int8_t>(
        saturate(std::uniform_real_distribution<double>(-1.0, 1.0)(t_hipblaslt_rng)));
}

// for hip_bfloat16, generate float, and convert to hip_bfloat16
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
inline hip_bfloat16 random_hpl_generator()
{
    return hip_bfloat16(std::uniform_real_distribution<float>(-0.5, 0.5)(t_hipblaslt_rng));
}

/*! \brief  generate a random ASCII string of up to length n */
inline std::string random_string(size_t n)
{
    std::string str;
    if(n)
    {
        size_t len = std::uniform_int_distribution<size_t>(1, n)(t_hipblaslt_rng);
        str.reserve(len);
        for(size_t i = 0; i < len; ++i)
            str.push_back(static_cast<char>(
                std::uniform_int_distribution<unsigned short>(0x20, 0x7E)(t_hipblaslt_rng)));
    }
    return str;
}
