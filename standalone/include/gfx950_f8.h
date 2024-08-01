/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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

#ifndef _GFX950_FLOAT8_IMPL_H_
#define _GFX950_FLOAT8_IMPL_H_

#include "gfx950_common.h"

enum class f8_format
{
    FP8_FMT,
    BF8_FMT
};

namespace gfx950_f8_impl
{
    /***************/
    /* p4v version */
    /***************/
    bool is_fp8_inf(uint8_t in, f8_format f8_src_fmt, bool f8_bias)
    {
        bool result = false;
        switch((uint32_t)f8_bias)
        {
        case 0:
            result = (f8_src_fmt == f8_format::BF8_FMT)
                         ? ((BEXP_BF8_GFX950(in) == 0x1f) && (MANT_BF8_GFX950(in) == 0x0))
                         : false;
            break;
        case 1:
            result = (in == 0x80);
            break;
        default:
            break;
        }
        return result;
    }

    bool is_fp8_nan(uint8_t in, f8_format f8_src_fmt, bool f8_bias)
    {
        bool result = false;
        switch((uint32_t)f8_bias)
        {
        case 0:
            result = (f8_src_fmt == f8_format::BF8_FMT)
                         ? (BEXP_BF8_GFX950(in) == 0x1f)
                         : ((BEXP_FP8_GFX950(in) == 0xf) && (MANT_FP8_GFX950(in) == 0x7));
            break;
        case 1:
            result = (in == 0x80);
            break;
        default:
            break;
        }
        return result;
    }

    bool is_fp8_zero(uint8_t in, bool f8_bias)
    {
        bool result = false;
        switch((uint32_t)f8_bias)
        {
        case 0:
            result = ((in & 0x7f) == 0x0);
            break;
        case 1:
            result = (in == 0x0);
            break;
        default:
            break;
        }
        return result;
    }

    bool is_fp8_denorm(uint8_t in, f8_format f8_src_fmt, bool f8_bias)
    {
        bool result = false;
        switch((uint32_t)f8_bias)
        {
        case 0:
        case 1:
            result = (f8_src_fmt == f8_format::BF8_FMT)
                         ? ((BEXP_BF8_GFX950(in) == 0x0) && (MANT_BF8_GFX950(in) != 0x0))
                         : ((BEXP_FP8_GFX950(in) == 0x0) && (MANT_FP8_GFX950(in) != 0x0));
            break;
        default:
            break;
        }
        return result;
    }

    uint32_t round_fp32_significand_rne(bool& is_significand_ovf, uint32_t trail_sig_fp32)
    {
        is_significand_ovf = false;
        // trail_sig_fp32 is of the form 1.31
        uint32_t trail_significand = (trail_sig_fp32 >> 8) & 0x7fffff;
        uint32_t ulp_half_ulp      = (trail_sig_fp32 >> 7) & 0x3; // 1.31 >> 7 = 1.24
        uint32_t or_remain         = (trail_sig_fp32 >> 0) & 0x7f;
        switch(ulp_half_ulp)
        {
        case 0:
        case 2:
            break;
        case 1:
            if(or_remain)
            {
                trail_significand += 1;
            }
            break;
        case 3:
            trail_significand += 1;
            break;
        default:
            break;
        }
        is_significand_ovf = (((trail_significand >> 23) & 0x1) == 0x1);
        return (trail_significand & 0x7fffff); // trail_significand is of the form .23
    }

    float fp8_to_f32(uint8_t in, uint32_t scale_exp_f32, f8_format f8_src_fmt, bool f8_bias)
    {
        // Based on f8_src_fmt and f8_bias, get sign, unbiased_exponent, trailing significand
        // If number is subnormal, normalize the number as well

        uint32_t sign_fp8
            = (f8_src_fmt == f8_format::BF8_FMT) ? SIGN_BF8_GFX950(in) : SIGN_FP8_GFX950(in);
        uint32_t trailing_significand_fp8
            = (f8_src_fmt == f8_format::BF8_FMT) ? MANT_BF8_GFX950(in) : MANT_FP8_GFX950(in);
        int32_t exp_fp8
            = (f8_src_fmt == f8_format::BF8_FMT) ? BEXP_BF8_GFX950(in) : BEXP_FP8_GFX950(in);
        uint32_t exp_bias_fp8     = (f8_src_fmt == f8_format::BF8_FMT) ? ((f8_bias == 0) ? 15 : 16)
                                                                       : ((f8_bias == 0) ? 7 : 8);
        int32_t  unbiased_exp_fp8 = exp_fp8 - exp_bias_fp8;
        bool     is_fp8_pre_scale_inf    = is_fp8_inf(in, f8_src_fmt, f8_bias);
        bool     is_fp8_pre_scale_nan    = is_fp8_nan(in, f8_src_fmt, f8_bias);
        bool     is_fp8_pre_scale_zero   = is_fp8_zero(in, f8_bias);
        bool     is_fp8_pre_scale_denorm = is_fp8_denorm(in, f8_src_fmt, f8_bias);
        uint32_t mantissa_bits           = (f8_src_fmt == f8_format::BF8_FMT) ? 2 : 3;

        // normalize subnormal number
        if(is_fp8_pre_scale_denorm)
        {
            unbiased_exp_fp8 = (f8_src_fmt == f8_format::BF8_FMT) ? ((f8_bias == 0) ? -14 : -15)
                                                                  : ((f8_bias == 0) ? -6 : -7);
            for(uint32_t mB = (mantissa_bits - 1); mB >= 0; mB--)
            {
                if((trailing_significand_fp8 >> mB) != 0)
                {
                    trailing_significand_fp8 = (trailing_significand_fp8 << (mantissa_bits - mB))
                                               & ((1 << mantissa_bits) - 1);
                    unbiased_exp_fp8 = unbiased_exp_fp8 - (mantissa_bits - mB);
                    break;
                }
            }
        }
        // at this point, leading significand bit is always 1 for non-zero input

        // apply scale
        unbiased_exp_fp8 += (scale_exp_f32 - 127);

        // at this point the exponent range is the output exponent range

        uint32_t f32 = 0;

        if(is_fp8_pre_scale_inf)
        {
            f32 = (sign_fp8 << 31) | 0x7f800000;
        }
        else if(is_fp8_pre_scale_nan || (scale_exp_f32 == 0xff))
        {
            f32 = (sign_fp8 << 31) | 0x7f800000
                  | (trailing_significand_fp8 << (23 - mantissa_bits));
        }
        else if(is_fp8_pre_scale_zero)
        {
            f32 = (sign_fp8 << 31) | 0x0;
        }
        else
        {
            if(unbiased_exp_fp8 < -149)
            {
                // scaled number is less than f32 min subnorm; output 0
                f32 = ((sign_fp8 << 31) | 0x0);
            }
            else if(unbiased_exp_fp8 < -126)
            {
                // scaled number is in f32 subnorm range,
                //  adjust mantissa such that unbiased_exp_fp8 is -126 and apply rne
                int32_t exp_shift        = -126 - unbiased_exp_fp8;
                int32_t unbiased_exp_f32 = unbiased_exp_fp8 + exp_shift;
                assert(unbiased_exp_f32 == -126);
                uint32_t trail_sig_fp32
                    = (1 << 31) | (trailing_significand_fp8 << (31 - mantissa_bits));
                trail_sig_fp32 >>= exp_shift;
                bool is_sig_ovf = false;
                trail_sig_fp32  = round_fp32_significand_rne(is_sig_ovf, trail_sig_fp32);
                f32             = (sign_fp8 << 31) | ((is_sig_ovf ? 0x01 : 0x00) << 23)
                      | (trail_sig_fp32 & 0x7fffff);
            }
            else if(unbiased_exp_fp8 < +128)
            {
                // scaled number is in f32 normal range
                //  apply rne
                uint32_t biased_exp_f32 = unbiased_exp_fp8 + 127;
                uint32_t trail_sig_fp32
                    = (1 << 31) | (trailing_significand_fp8 << (31 - mantissa_bits));
                bool is_sig_ovf = false;
                trail_sig_fp32  = round_fp32_significand_rne(is_sig_ovf, trail_sig_fp32);
                biased_exp_f32 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f32 == +255)
                {
                    f32 = (sign_fp8 << 31) | 0x7f800000;
                }
                else
                {
                    f32 = (sign_fp8 << 31) | ((biased_exp_f32 & 0xff) << 23)
                          | (trail_sig_fp32 & 0x7fffff);
                }
            }
            else
            {
                // scaled number is greater than f32 max normL output +/- inf
                f32 = (sign_fp8 << 31) | 0x7f800000;
            }
        }

        return *((float*)(&f32));
    }

    uint8_t get_fp8_inf(f8_format f8_src_fmt, bool f8_bias, bool clamp)
    {
        uint8_t result = 0;
        if(clamp == 0)
        {
            switch((uint32_t)f8_bias)
            {
            case 0:
                result = (f8_src_fmt == f8_format::BF8_FMT) ? 0x7c : 0x7f;
                break;
            case 1:
                result = 0x80;
                break;
            default:
                break;
            }
        }
        else
        {
            switch((uint32_t)f8_bias)
            {
            case 0:
                result = (f8_src_fmt == f8_format::BF8_FMT) ? 0x7b : 0x7e;
                break;
            case 1:
                result = 0x80;
                break;
            default:
                break;
            }
        }
        return result;
    }

    uint8_t get_fp8_nan(f8_format f8_src_fmt, bool f8_bias)
    {
        uint8_t result = 0;
        switch((uint32_t)f8_bias)
        {
        case 0:
            result = 0x7f;
            break;
        case 1:
            result = 0x80;
            break;
        default:
            break;
        }
        return result;
    }

    uint8_t get_fp8_exp_bias(f8_format f8_src_fmt, bool f8_bias)
    {
        uint8_t result = 0;
        switch((uint32_t)f8_bias)
        {
        case 0:
            result = (f8_src_fmt == f8_format::BF8_FMT) ? 15 : 7;
            break;
        case 1:
            result = (f8_src_fmt == f8_format::BF8_FMT) ? 16 : 8;
            break;
        default:
            break;
        }
        return result;
    }

    uint32_t round_f8_significand_rne(bool&     is_significand_ovf,
                                      f8_format f8_src_fmt,
                                      uint32_t  trail_sig_f8)
    {
        is_significand_ovf = false;
        // trail_sig_f8 is of the form 1.31
        uint32_t mantissa_bits_f8 = (f8_src_fmt == f8_format::BF8_FMT) ? 2 : 3;
        uint32_t trail_significand
            = (trail_sig_f8 >> (31 - mantissa_bits_f8)) & ((1 << mantissa_bits_f8) - 1);
        uint32_t ulp_half_ulp = (trail_sig_f8 >> (31 - mantissa_bits_f8 - 1))
                                & 0x3; // 1.31 >> (31-mantissa_bits_f8-1)
        uint32_t or_remain = (trail_sig_f8 >> 0) & ((1 << (31 - mantissa_bits_f8 - 1)) - 1);
        switch(ulp_half_ulp)
        {
        case 0:
        case 2:
            break;
        case 1:
            if(or_remain)
            {
                trail_significand += 1;
            }
            break;
        case 3:
            trail_significand += 1;
            break;
        default:
            break;
        }
        is_significand_ovf = (((trail_significand >> mantissa_bits_f8) & 0x1) == 0x1);
        // trail_significand is of the form .mantissa_bits_f8
        return (trail_significand & ((1 << mantissa_bits_f8) - 1));
    }

    uint8_t f32_to_fp8(uint32_t  in,
                       uint32_t  scale_exp_f32,
                       f8_format f8_src_fmt,
                       bool      f8_bias,
                       bool      clamp,
                       bool      stochastic_round,
                       uint32_t  in1)
    {
        uint32_t sign_f32                 = SIGN32(in);
        uint32_t trailing_significand_f32 = MAN32(in);
        int32_t  exp_f32                  = BEXP32(in);
        int32_t  unbiased_exp_f32         = exp_f32 - 127;
        bool     is_f32_pre_scale_inf     = (exp_f32 == 0xff) && (trailing_significand_f32 == 0);
        bool     is_f32_pre_scale_nan     = (exp_f32 == 0xff) && (trailing_significand_f32 != 0);
        bool     is_f32_pre_scale_zero    = ((in & 0x7fffffff) == 0);
        bool     is_f32_pre_scale_denorm  = (exp_f32 == 0x00) && (trailing_significand_f32 != 0);
        // stochastic rounding
        // copied from existing f8_math.cpp
        if(stochastic_round)
        {
            trailing_significand_f32
                += ((f8_src_fmt == f8_format::BF8_FMT) ? ((in1 & 0xfffff800) >> 11)
                                                       : ((in1 & 0xfffff000) >> 12));
        }

        // normalize subnormal number
        if(is_f32_pre_scale_denorm)
        {
            unbiased_exp_f32 = -126;
            for(uint32_t mB = 22; mB >= 0; mB--)
            {
                if((trailing_significand_f32 >> mB) != 0)
                {
                    trailing_significand_f32 = (trailing_significand_f32 << (23 - mB)) & 0x7fffff;
                    unbiased_exp_f32         = unbiased_exp_f32 - (23 - mB);
                    break;
                }
            }
        }
        // at this point, leading significand bit is always 1 for non-zero input

        // apply scale
        unbiased_exp_f32 -= (scale_exp_f32 - 127);

        // at this point the exponent is the output exponent range

        uint8_t fp8 = 0;

        if(is_f32_pre_scale_inf)
        {
            fp8 = (sign_f32 << 7) | get_fp8_inf(f8_src_fmt, f8_bias, clamp);
        }
        else if(is_f32_pre_scale_nan || (scale_exp_f32 == 0xff))
        {
            fp8 = (sign_f32 << 7) | get_fp8_nan(f8_src_fmt, f8_bias);
        }
        else if(is_f32_pre_scale_zero)
        {
            fp8 = (f8_src_fmt == f8_format::BF8_FMT) ? 0x0 : ((sign_f32 << 7) | 0x0);
        }
        else
        {
            int32_t  min_subnorm_uexp_f8 = (f8_src_fmt == f8_format::BF8_FMT) ? -16 : -9;
            int32_t  max_subnorm_uexp_f8 = (f8_src_fmt == f8_format::BF8_FMT) ? -14 : -6;
            int32_t  max_norm_uexp_f8    = (f8_src_fmt == f8_format::BF8_FMT) ? +15 : +8;
            uint32_t mantissa_bits_f8    = (f8_src_fmt == f8_format::BF8_FMT) ? 2 : 3;
            uint32_t exponent_bits_f8    = (f8_src_fmt == f8_format::BF8_FMT) ? 5 : 4;
            if(unbiased_exp_f32 < min_subnorm_uexp_f8)
            {
                // scaled number is less than f8 min subnorm; output 0
                fp8 = (sign_f32 << 7) | 0x0;
            }
            else if(unbiased_exp_f32 < max_subnorm_uexp_f8)
            {
                // scaled number is in f8 subnorm range,
                //  adjust mantissa such that unbiased_exp_f32 is
                //  max_subnorm_uexp_f8 and apply rne
                int32_t exp_shift       = max_subnorm_uexp_f8 - unbiased_exp_f32;
                int32_t unbiased_exp_f8 = unbiased_exp_f32 + exp_shift;
                assert(unbiased_exp_f8 == max_subnorm_uexp_f8);
                uint32_t trail_sig_f8 = (1 << 31) | (trailing_significand_f32 << 8);
                trail_sig_f8 >>= exp_shift;
                bool is_sig_ovf = false;
                trail_sig_f8    = round_f8_significand_rne(is_sig_ovf, f8_src_fmt, trail_sig_f8);
                fp8 = (sign_f32 << 7) | ((uint8_t)((is_sig_ovf ? 0x01 : 0x00) << mantissa_bits_f8))
                      | (trail_sig_f8 & ((1 << mantissa_bits_f8) - 1));
            }
            else if(unbiased_exp_f32 <= max_norm_uexp_f8)
            {
                // scaled number is in f8 normal range
                //  apply rne
                uint32_t biased_exp_f8 = unbiased_exp_f32 + get_fp8_exp_bias(f8_src_fmt, f8_bias);
                uint32_t trail_sig_f8  = (1 << 31) | (trailing_significand_f32 << 8);
                bool     is_sig_ovf    = false;
                trail_sig_f8 = round_f8_significand_rne(is_sig_ovf, f8_src_fmt, trail_sig_f8);
                biased_exp_f8 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f8
                   == (uint32_t)(max_norm_uexp_f8 + get_fp8_exp_bias(f8_src_fmt, f8_bias) + 1))
                {
                    fp8 = (sign_f32 << 7) | get_fp8_inf(f8_src_fmt, f8_bias, clamp);
                }
                else if((f8_src_fmt != f8_format::BF8_FMT) && (biased_exp_f8 == 0xf)
                        && (trail_sig_f8 == 0x7))
                {
                    fp8 = (sign_f32 << 7) | get_fp8_inf(f8_src_fmt, f8_bias, clamp);
                }
                else
                {
                    fp8 = (sign_f32 << 7)
                          | ((biased_exp_f8 & ((1 << exponent_bits_f8) - 1)) << mantissa_bits_f8)
                          | (trail_sig_f8 & ((1 << mantissa_bits_f8) - 1));
                }
            }
            else
            {
                // scaled number is greater than f8 max normal output
                //  clamp to f8 flt_max/inf based on clamp control
                fp8 = (sign_f32 << 7) | get_fp8_inf(f8_src_fmt, f8_bias, clamp);
            }
        }

        return fp8;
    }

    /*
     *  OCP FP8 s/w conversion.. default for gfx950
     *  Only for temporary use, will switch to standard header when available
     *  Reference : https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_e4m3fn.h
     *  License: https://github.com/pytorch/pytorch?tab=License-1-ov-file
     */
    inline int clz(uint32_t x)
    {
        return __builtin_clz(x);
    }

    inline uint8_t cast_to_f8_e4m3(float _x, bool stoch, uint32_t rng)
    {
        uint32_t fp8_max     = uint32_t(1087) << 20;
        uint32_t denorm_mask = uint32_t(141) << 23;
        uint32_t f_bits      = reinterpret_cast<uint32_t&>(_x);

        uint8_t        result = 0u;
        const uint32_t sign   = f_bits & 0x80000000;

        f_bits ^= sign;

        if(f_bits >= fp8_max)
        {
            result = 0x7f;
        }
        else
        {
            if(f_bits < (uint32_t(121) << 23))
            {
                float f_bits_f32      = reinterpret_cast<float&>(f_bits);
                float denorm_mask_f32 = reinterpret_cast<float&>(denorm_mask);

                /* TODO: Implement SR support for subnormal values! */
                /* FP32's rounding is used for RNE, need to split*/
                float sum = f_bits_f32 + denorm_mask_f32;

                f_bits = reinterpret_cast<uint32_t&>(sum);
                result = static_cast<uint8_t>(f_bits - denorm_mask);
            }
            else
            {
                if(!stoch) // RNE
                {
                    uint8_t mant_odd = (f_bits >> 20) & 1;
                    f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
                    f_bits += mant_odd;
                }
                else // SR
                    f_bits += ((uint32_t)(7 - 127) << 23) + (rng >> 12);

                result = static_cast<uint8_t>(f_bits >> 20);
            }
        }

        result |= static_cast<uint8_t>(sign >> 24);
        return result;
    }

    inline uint8_t cast_to_f8_e5m2(float _x, bool stoch, uint32_t rng)
    {
        uint32_t fp32_inf    = uint32_t(255) << 23;
        uint32_t fp8_max     = uint32_t(143) << 23;
        uint32_t denorm_mask = uint32_t(134) << 23;

        uint32_t f_bits = reinterpret_cast<uint32_t&>(_x);
        uint8_t  result = 0u;

        const uint32_t sign = f_bits & 0x80000000;

        f_bits ^= sign;

        if(f_bits >= fp8_max)
        {
            result = f_bits > fp32_inf ? uint32_t(0x7F) : uint32_t(0x7C);
        }
        else
        {
            if(f_bits < (uint32_t(113) << 23))
            {
                float f_bits_f32      = reinterpret_cast<float&>(f_bits);
                float denorm_mask_f32 = reinterpret_cast<float&>(denorm_mask);

                /* TODO: Implement SR support for subnormal values! */
                /* FP32's rounding is used for RNE, need to split*/
                float sum = f_bits_f32 + denorm_mask_f32;

                f_bits = reinterpret_cast<uint32_t&>(sum);
                result = static_cast<uint8_t>(f_bits - denorm_mask);
            }
            else
            {
                if(!stoch) // RNE
                {
                    uint32_t mant_odd = (f_bits >> 21) & 1;
                    f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;

                    // rounding bias part 2
                    f_bits += mant_odd;
                }
                else // SR
                    f_bits += ((uint32_t)(15 - 127) << 23) + (rng >> 11);

                result = static_cast<uint8_t>(f_bits >> 21);
            }
        }

        result |= static_cast<uint8_t>(sign >> 24);
        return result;
    }

    inline float cast_from_f8_e4m3(uint8_t x)
    {
        const uint32_t w            = (uint32_t)x << 24;
        const uint32_t sign         = w & UINT32_C(0x80000000);
        const uint32_t nonsign      = w & UINT32_C(0x7FFFFFFF);
        uint32_t       renorm_shift = clz(nonsign);
        renorm_shift                = renorm_shift > 4 ? renorm_shift - 4 : 0;

        const int32_t inf_nan_mask = ((int32_t)(nonsign + 0x01000000) >> 8) & INT32_C(0x7F800000);
        const int32_t zero_mask    = (int32_t)(nonsign - 1) >> 31;
        uint32_t      result
            = sign
              | ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) | inf_nan_mask)
                 & ~zero_mask);

        return reinterpret_cast<const float&>(result);
    }

    inline float cast_from_f8_e5m2(uint8_t x)
    {
        uint16_t half_representation = x;
        half_representation <<= 8;
        _Float16 half_value = reinterpret_cast<const _Float16&>(half_representation);
        return float(half_value);
    }
}

struct gfx950_fp8
{
    uint8_t data;

    // default constructor
    gfx950_fp8() = default;

    explicit gfx950_fp8(float v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
    {
        bool SR = (rm == rounding_mode::stochastic);

        // data = gfx950_f8_impl::f32_to_fp8( *((uint32_t*)(&v)), 127, f8_format::FP8_FMT, false, true, SR, rng);
        data = gfx950_f8_impl::cast_to_f8_e4m3(v, SR, rng);
    }

    // Constructor from half
    explicit gfx950_fp8(_Float16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_fp8((float)v, rm, rng)
    {
    }

    // constructor from bfloat16
    explicit gfx950_fp8(bfloat16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_fp8((float)v, rm, rng)
    {
    }

    explicit inline operator float() const
    {
        // return gfx950_f8_impl::fp8_to_f32(data, 127, f8_format::FP8_FMT, false); // p4v-version
        return gfx950_f8_impl::cast_from_f8_e4m3(data);
    }

    // convert to half
    explicit inline operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to bfloat16
    explicit inline operator bfloat16() const
    {
        return bfloat16(float(*this)); // convert to float, then convert to f16
    }

    // check for zero
    inline bool is_zero() const
    {
        return gfx950_f8_impl::is_fp8_zero(data, false);
    }

    // check for nan
    inline bool is_nan() const
    {
        return gfx950_f8_impl::is_fp8_nan(data, f8_format::FP8_FMT, false);
        return data == 0x80;
    }

    // check for inf
    inline bool is_inf() const
    {
        return gfx950_f8_impl::is_fp8_inf(data, f8_format::FP8_FMT, false);
    }

    // assignment overloading only from the same F8 types
    inline gfx950_fp8& operator=(const gfx950_fp8& a)
    {
        data = a.data;
        return *this;
    }
};

struct gfx950_bf8
{
    uint8_t data;

    // default constructor
    gfx950_bf8() = default;

    explicit gfx950_bf8(float v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
    {
        bool SR = (rm == rounding_mode::stochastic);

        // data = gfx950_f8_impl::f32_to_fp8(*((uint32_t*)(&v)), 127, f8_format::BF8_FMT, false, true, SR, rng);
        data = gfx950_f8_impl::cast_to_f8_e5m2(v, SR, rng);
    }

    // Constructor from half
    explicit gfx950_bf8(_Float16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_bf8((float)v, rm, rng)
    {
    }

    // constructor from bfloat16
    explicit gfx950_bf8(bfloat16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_bf8((float)v, rm, rng)
    {
    }

    explicit inline operator float() const
    {
        // return gfx950_f8_impl::fp8_to_f32(data, 127, f8_format::BF8_FMT, false);
        return gfx950_f8_impl::cast_from_f8_e5m2(data);
    }

    // convert to half
    explicit inline operator _Float16() const
    {
        return _Float16(float(*this)); // convert to float, then convert to f16
    }

    // convert to bfloat16
    explicit inline operator bfloat16() const
    {
        return bfloat16(float(*this)); // convert to float, then convert to f16
    }

    // check for zero
    inline bool is_zero() const
    {
        return gfx950_f8_impl::is_fp8_zero(data, false);
    }

    // check for nan
    inline bool is_nan() const
    {
        return gfx950_f8_impl::is_fp8_nan(data, f8_format::BF8_FMT, false);
    }

    // check for inf
    inline bool is_inf() const
    {
        return gfx950_f8_impl::is_fp8_inf(data, f8_format::BF8_FMT, false);
    }

    // assignment overloading only from the same F8 types
    inline gfx950_bf8& operator=(const gfx950_bf8& a)
    {
        data = a.data;
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const gfx950_fp8& fp8)
{
    return os << float(fp8);
}

inline std::ostream& operator<<(std::ostream& os, const gfx950_bf8& bf8)
{
    return os << float(bf8);
}

#endif // _GFX950_FLOAT8_IMPL_H_
