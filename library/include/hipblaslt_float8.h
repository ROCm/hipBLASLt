/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc.
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

#ifndef _HIPBLASLT_FLOAT8_H_
#define _HIPBLASLT_FLOAT8_H_

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))
/*! \brief Struct to represent a 8 bit floating-point number. */
typedef struct
{
    uint8_t data;
} hipblaslt_f8_fnuz;

typedef struct
{
    uint8_t data;
} hipblaslt_bf8_fnuz;

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// We are clipping in down conversion by default
#define hipblaslt_F8_downcast_clipping 1

namespace hipblaslt_hip_f8_impl
{

    template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
    HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

    template <int wm, int we, typename T, bool negative_zero_nan>
    HIP_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace hipblaslt_hip_f8_impl

#include "hipblaslt_hip_f8_impl.h"

static __device__ bool hipblaslt_hip_f8_bias_mode_bit_device = true;
static bool            hipblaslt_hip_f8_bias_mode_bit_host   = true;

struct HIPBLASLT_EXPORT hipblaslt_f8_fnuz
{
    uint8_t data;
    enum class hipblaslt_hip_f8_rounding_mode
    {
        standard,
        stochastic
    };

    // default constructor
    HIP_HOST_DEVICE hipblaslt_f8_fnuz() = default;

#if defined(__gfx940__)
    // device specific optimized F8 down-conversion code

    template <bool stochastic_rounding = false>
    static HIP_DEVICE uint8_t cast_to_f8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;
        union
        {
            float    fval;
            uint32_t i32val;
            uint8_t  i8val[4]; // NOTE: not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

#ifdef hipblaslt_F8_downcast_clipping
        if((val.i32val & 0x7F800000) != 0x7F800000) /// propagate NAN/INF, no clipping
            val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
#endif
        if(stochastic_rounding)
        {
            ival       = __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0); // 0 pos
            val.i32val = ival;
            i8data     = val.i8val[0]; // little endian
        }
        else // RNE CVT
        {
            ival = __builtin_amdgcn_cvt_pk_fp8_f32(
                val.fval, val.fval, ival, false); // false -> WORD0
            val.i32val = ival;
            i8data     = val.i8val[0];
        }
        return i8data;
    }

#endif // __gfx940__

    // constructor from float
#if defined(__gfx940__)

    // NOTE: ON-DEVICE... always optimal bias
    explicit HIP_DEVICE hipblaslt_f8_fnuz(float                          v,
                                          hipblaslt_hip_f8_rounding_mode rm
                                          = hipblaslt_hip_f8_rounding_mode::standard,
                                          uint32_t rng = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == hipblaslt_hip_f8_rounding_mode::stochastic)
            data = cast_to_f8_from_f32<true>(v, rng);
        else
            data = cast_to_f8_from_f32<false>(v);
    }

    // Host only implementation using s/w simulation
    explicit HIP_HOST
#else
    // both Host and DEVICE for non-gfx940 using s/w simulation
    explicit HIP_HOST_DEVICE
#endif
        hipblaslt_f8_fnuz(float                          v,
                          hipblaslt_hip_f8_rounding_mode rm
                          = hipblaslt_hip_f8_rounding_mode::standard,
                          uint32_t rng = 0)
    {
#ifdef hipblaslt_F8_downcast_clipping
        data = hipblaslt_hip_f8_impl::
            cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
                v, (rm == hipblaslt_hip_f8_rounding_mode::stochastic), rng);
#else // hipblaslt_F8_downcast_clipping
        data = hipblaslt_hip_f8_impl::
            cast_to_f8<3, 4, float, true /*negative_zero_nan*/, false /*clip*/>(
                v, (rm == hipblaslt_hip_f8_rounding_mode::stochastic), rng);
#endif // hipblaslt_F8_downcast_clipping
    }

    // Constructor from half
    explicit HIP_HOST_DEVICE hipblaslt_f8_fnuz(_Float16                       v,
                                               hipblaslt_hip_f8_rounding_mode rm
                                               = hipblaslt_hip_f8_rounding_mode::standard,
                                               uint32_t rng = 0)
        : hipblaslt_f8_fnuz((float)v, rm, rng)
    {
    }
    // constructor from bfloat16
    explicit HIP_HOST_DEVICE hipblaslt_f8_fnuz(hip_bfloat16                   v,
                                               hipblaslt_hip_f8_rounding_mode rm
                                               = hipblaslt_hip_f8_rounding_mode::standard,
                                               uint32_t rng = 0)
        : hipblaslt_f8_fnuz((float)v, rm, rng)
    {
    }
    // constructor from int
    explicit HIP_HOST_DEVICE hipblaslt_f8_fnuz(int                            v,
                                               hipblaslt_hip_f8_rounding_mode rm
                                               = hipblaslt_hip_f8_rounding_mode::standard,
                                               uint32_t rng = 0)
        : hipblaslt_f8_fnuz((float)v, rm, rng)
    {
    }
    // constructor from double
    explicit HIP_HOST_DEVICE hipblaslt_f8_fnuz(double                         v,
                                               hipblaslt_hip_f8_rounding_mode rm
                                               = hipblaslt_hip_f8_rounding_mode::standard,
                                               uint32_t rng = 0)
        : hipblaslt_f8_fnuz((float)v, rm, rng)
    {
    }

    // convert to float
#if defined(__gfx940__)
    // upcast using device specific intrinsic
    explicit inline HIP_DEVICE operator float() const
    {
        float    fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

        return fval;
    }

    explicit inline HIP_HOST operator float() const
#else // non gfx940
    explicit inline HIP_HOST_DEVICE operator float() const
#endif
    {
        return hipblaslt_hip_f8_impl::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(data);
    }

    // convert to half
    explicit inline HIP_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to bfloat16
    explicit inline HIP_HOST_DEVICE operator hip_bfloat16() const
    {
        return hip_bfloat16(float(*this)); // convert to float, then convert to f16
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return data == 0x00;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return data == 0x80;
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return data == 0x80;
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ hipblaslt_f8_fnuz& operator=(const hipblaslt_f8_fnuz& a)
    {
        data = a.data;
        return *this;
    }
};

struct HIPBLASLT_EXPORT hipblaslt_bf8_fnuz
{
    uint8_t data;
    enum class hipblaslt_hip_f8_rounding_mode
    {
        standard,
        stochastic
    };

    // default constructor
    HIP_HOST_DEVICE hipblaslt_bf8_fnuz() = default;

#if defined(__gfx940__)
    // device specific optimized F8 down-conversion code

    template <bool stochastic_rounding = false>
    static HIP_DEVICE uint8_t cast_to_bf8_from_f32(float v, uint32_t rng = 0)
    {
        uint8_t i8data;
        union
        {
            float    fval;
            uint32_t i32val;
            uint8_t  i8val[4]; // NOTE: not endian independent
        } val;

        uint32_t ival = 0;
        val.fval      = v;

#ifdef hipblaslt_F8_downcast_clipping
        if((val.i32val & 0x7F800000) != 0x7F800000) // propagate NAN/INF, no clipping
            val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
#endif
        if(stochastic_rounding)
        {
            ival       = __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
            val.i32val = ival;
            i8data     = val.i8val[0]; // little endian
        }
        else // RNE CVT
        {
            ival = __builtin_amdgcn_cvt_pk_bf8_f32(
                val.fval, val.fval, ival, false); // false -> WORD0
            val.i32val = ival;
            i8data     = val.i8val[0];
        }
        return i8data;
    }

#endif // __gfx940__

    // constructor from float
#if defined(__gfx940__)

    // NOTE: ON-DEVICE... always optimal bias
    explicit HIP_DEVICE hipblaslt_bf8_fnuz(float                          v,
                                           hipblaslt_hip_f8_rounding_mode rm
                                           = hipblaslt_hip_f8_rounding_mode::standard,
                                           uint32_t rng = 0)
    {
        // runtime branch, use cast_to_f8_from_f32 if want to avoid it
        if(rm == hipblaslt_hip_f8_rounding_mode::stochastic)
            data = cast_to_bf8_from_f32<true>(v, rng);
        else
            data = cast_to_bf8_from_f32<false>(v);
    }

    // Host only implementation using s/w simulation
    explicit HIP_HOST
#else
    // both Host and DEVICE for non-gfx940 using s/w simulation
    explicit HIP_HOST_DEVICE
#endif
        hipblaslt_bf8_fnuz(float                          v,
                           hipblaslt_hip_f8_rounding_mode rm
                           = hipblaslt_hip_f8_rounding_mode::standard,
                           uint32_t rng = 0)
    {
#ifdef hipblaslt_F8_downcast_clipping
        data = hipblaslt_hip_f8_impl::
            cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
                v, (rm == hipblaslt_hip_f8_rounding_mode::stochastic), rng);
#else
        data = hipblaslt_hip_f8_impl::
            cast_to_f8<2, 5, float, true /*negative_zero_nan*/, false /*clip*/>(
                v, (rm == hipblaslt_hip_f8_rounding_mode::stochastic), rng);
#endif // hipblaslt_F8_downcast_clipping
    }

    // Constructor from half
    explicit HIP_HOST_DEVICE hipblaslt_bf8_fnuz(_Float16                       v,
                                                hipblaslt_hip_f8_rounding_mode rm
                                                = hipblaslt_hip_f8_rounding_mode::standard,
                                                uint32_t rng = 0)
        : hipblaslt_bf8_fnuz((float)v, rm, rng)
    {
    }
    // constructor from bfloat16
    explicit HIP_HOST_DEVICE hipblaslt_bf8_fnuz(hip_bfloat16                   v,
                                                hipblaslt_hip_f8_rounding_mode rm
                                                = hipblaslt_hip_f8_rounding_mode::standard,
                                                uint32_t rng = 0)
        : hipblaslt_bf8_fnuz((float)v, rm, rng)
    {
    }
    // constructor from int
    explicit HIP_HOST_DEVICE hipblaslt_bf8_fnuz(int                            v,
                                                hipblaslt_hip_f8_rounding_mode rm
                                                = hipblaslt_hip_f8_rounding_mode::standard,
                                                uint32_t rng = 0)
        : hipblaslt_bf8_fnuz((float)v, rm, rng)
    {
    }
    // constructor from double
    explicit HIP_HOST_DEVICE hipblaslt_bf8_fnuz(double                         v,
                                                hipblaslt_hip_f8_rounding_mode rm
                                                = hipblaslt_hip_f8_rounding_mode::standard,
                                                uint32_t rng = 0)
        : hipblaslt_bf8_fnuz((float)v, rm, rng)
    {
    }

    // convert to float
#if defined(__gfx940__)
    // upcast using device specific intrinsic
    explicit inline HIP_DEVICE operator float() const
    {
        float    fval;
        uint32_t i32val = static_cast<uint32_t>(data);

        // upcast
        asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));

        return fval;
    }

    explicit inline HIP_HOST operator float() const
#else // non gfx940
    explicit inline HIP_HOST_DEVICE operator float() const
#endif
    {
        return hipblaslt_hip_f8_impl::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(data);
    }

    // convert to half
    explicit inline HIP_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to bfloat16
    explicit inline HIP_HOST_DEVICE operator hip_bfloat16() const
    {
        return hip_bfloat16(float(*this)); // convert to float, then convert to f16
    }

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        return data == 0x00;
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        return data == 0x80;
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        return data == 0x80;
    }

    // assignment overloading only from the same F8 types
    inline __host__ __device__ hipblaslt_bf8_fnuz& operator=(const hipblaslt_bf8_fnuz& a)
    {
        data = a.data;
        return *this;
    }
};

namespace std
{
    inline hipblaslt_f8_fnuz sin(hipblaslt_f8_fnuz a)
    {
        return hipblaslt_f8_fnuz(sinf(float(a)));
    }
    inline hipblaslt_f8_fnuz cos(hipblaslt_f8_fnuz a)
    {
        return hipblaslt_f8_fnuz(cosf(float(a)));
    }
    inline hipblaslt_bf8_fnuz sin(hipblaslt_bf8_fnuz a)
    {
        return hipblaslt_bf8_fnuz(sinf(float(a)));
    }
    inline hipblaslt_bf8_fnuz cos(hipblaslt_bf8_fnuz a)
    {
        return hipblaslt_bf8_fnuz(cosf(float(a)));
    }
    __device__ __host__ constexpr hipblaslt_f8_fnuz real(const hipblaslt_f8_fnuz& a)
    {
        return a;
    }
    __device__ __host__ constexpr hipblaslt_bf8_fnuz real(const hipblaslt_bf8_fnuz& a)
    {
        return a;
    }
}

// Special operator overloading
inline std::ostream& operator<<(std::ostream& os, const hipblaslt_f8_fnuz& f8)
{
    return os << float(f8);
}

inline std::ostream& operator<<(std::ostream& os, const hipblaslt_bf8_fnuz& bf8)
{
    return os << float(bf8);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns float
inline __host__ __device__ float operator+(const float fa, hipblaslt_f8_fnuz b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(const float fa, hipblaslt_bf8_fnuz b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(hipblaslt_f8_fnuz a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(hipblaslt_bf8_fnuz a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(hipblaslt_f8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ float operator+(hipblaslt_bf8_fnuz a, hipblaslt_f8_fnuz b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ hipblaslt_f8_fnuz operator+(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return hipblaslt_f8_fnuz(float(a) + float(b));
}

inline __host__ __device__ hipblaslt_bf8_fnuz operator+(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return hipblaslt_bf8_fnuz(float(a) + float(b));
}

inline __host__ __device__ hipblaslt_f8_fnuz& operator+=(hipblaslt_f8_fnuz& a, hipblaslt_f8_fnuz b)
{
    return a = hipblaslt_f8_fnuz(float(a) + float(b));
}

inline __host__ __device__ hipblaslt_bf8_fnuz& operator+=(hipblaslt_bf8_fnuz& a,
                                                          hipblaslt_bf8_fnuz  b)
{
    return a = hipblaslt_bf8_fnuz(float(a) + float(b));
}

// overloading multiplication, always returns float,
inline __host__ __device__ float operator*(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, hipblaslt_f8_fnuz b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f8_fnuz a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, hipblaslt_f8_fnuz b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, hipblaslt_f8_fnuz b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, hipblaslt_bf8_fnuz b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf8_fnuz a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(int32_t a, hipblaslt_bf8_fnuz b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(double a, hipblaslt_bf8_fnuz b)
{
    return ((float)a * float(b));
}

// overloading for mixed f8 and bf8 types
inline __host__ __device__ float operator*(hipblaslt_f8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(hipblaslt_bf8_fnuz a, hipblaslt_f8_fnuz b)
{
    return float(a) * float(b);
}

// overloading for compare
inline __host__ __device__ bool operator==(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return (a.data == b.data);
}
inline __host__ __device__ bool operator==(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return (a.data == b.data);
}

inline __host__ __device__ bool operator!=(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return (a.data != b.data);
}
inline __host__ __device__ bool operator!=(hipblaslt_bf8_fnuz a, hipblaslt_bf8_fnuz b)
{
    return (a.data != b.data);
}

inline __host__ __device__ bool operator>=(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline __host__ __device__ bool operator>(hipblaslt_f8_fnuz a, hipblaslt_f8_fnuz b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}
// ================ Explicit downcasting to support different rounding (RNE, SR) ===============
// NOTE: we going to remove all assignment operator overloading from other types and enforce
// this explicit_downcast function to make any roudning behavior default
// We have to explicitly call this function with SR flag
/*
template <typename T,
          typename Ta,
          bool stochastic_rounding,
          typename std::enable_if<std::is_same<T, Ta>{}, int>::type = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng = 0)
{
    // same type, no conversion
    return a;
}

// Use h/w intrinsic and optimized version when __gfx940__
template <
    typename T,
    typename Ta,
    bool stochastic_rounding,
    typename std::enable_if<(!(std::is_same<T, Ta>{})
                             && (std::is_same<T, hipblaslt_f8_fnuz>{} || std::is_same<T, hipblaslt_bf8_fnuz>{})),
                            int>::type
    = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng)
{
#if defined(__gfx940__)
    // NOTE: we are directly calling cast_to_f8_from_f32 instead of constructor to optimize away one runtime branch
    T val;
    if(std::is_same<T, hipblaslt_f8_fnuz>::value)
        val.data = hipblaslt_f8_fnuz::cast_to_f8_from_f32<stochastic_rounding>(float(a), rng);
    else
        val.data = hipblaslt_bf8_fnuz::cast_to_bf8_from_f32<stochastic_rounding>(float(a), rng);
    return val;
#else // non gfx940
    return T(float(a),
             stochastic_rounding ? T::hipblaslt_hip_f8_rounding_mode::stochastic
                                 : T::hipblaslt_hip_f8_rounding_mode::standard,
             rng);
#endif // __gfx940__
}

// NOTE NOTE: The above code is good if we don't consider HIP-GEMM code and only consider the quantization
// However, if we need HIP-GEMM for fall-back, we would need explicit_cast handles Tacc=f32 to To=f16/bf16 conversion
template <
    typename T,
    typename Ta,
    bool stochastic_rounding,
    typename std::enable_if<(!(std::is_same<T, Ta>{})
                             && !(std::is_same<T, hipblaslt_f8_fnuz>{} || std::is_same<T, hipblaslt_bf8_fnuz>{})),
                            int>::type
    = 0>
inline __host__ __device__ T explicit_downcast(Ta a, uint32_t rng)
{
    // the return type is not a F8 types, no SR for those types
    // not sure if we have direct conversion, so converting to float first
    // no effect if the input type is float
    return T(float(a));
}
*/
// =================================================================================================

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // _HIPBLASLT_FLOAT8_H_
