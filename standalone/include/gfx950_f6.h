
#ifndef _GFX950_FLOAT6_IMPL_H_
#define _GFX950_FLOAT6_IMPL_H_

#include "gfx950_common.h"

enum class f6_format
{
    FP6_FMT = 0,
    BF6_FMT
};

namespace gfx950_f6_impl
{
    bool is_fp6_dnrm(uint6_t in, f6_format f6_src_fmt)
    {
        bool result = false;
        switch(f6_src_fmt)
        {
        case f6_format::BF6_FMT:
            result = (BEXP_BF6(in.val) == 0) && (MANT_BF6(in.val) != 0);
            break;
        case f6_format::FP6_FMT:
            result = (BEXP_FP6(in.val) == 0) && (MANT_FP6(in.val) != 0);
            break;
        default:
            break;
        }
        return result;
    }

    uint32_t round_fp32_f6_significand_rne(bool& is_significand_ovf, uint32_t trail_sig_fp32)
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

    float fp6_to_f32(uint6_t in, uint32_t scale_exp_f32, f6_format f6_src_fmt)
    {
        uint32_t sign_fp6
            = (f6_src_fmt == f6_format::BF6_FMT) ? SIGN_BF6(in.val) : SIGN_FP6(in.val);
        uint32_t trailing_significand_fp6
            = (f6_src_fmt == f6_format::BF6_FMT) ? MANT_BF6(in.val) : MANT_FP6(in.val);
        int32_t  exp_fp6 = (f6_src_fmt == f6_format::BF6_FMT) ? BEXP_BF6(in.val) : BEXP_FP6(in.val);
        uint32_t exp_bias_fp6          = (f6_src_fmt == f6_format::BF6_FMT) ? 3 : 1;
        int32_t  unbiased_exp_fp6      = exp_fp6 - exp_bias_fp6;
        bool     is_fp6_pre_scale_zero = ((in.val & 0x1f) == 0x0);
        bool     is_fp6_pre_scale_dnrm = is_fp6_dnrm(in, f6_src_fmt);
        uint32_t mantissa_bits         = (f6_src_fmt == f6_format::BF6_FMT) ? 2 : 3;

        // normalize subnormal number
        if(is_fp6_pre_scale_dnrm)
        {
            unbiased_exp_fp6 = (f6_src_fmt == f6_format::BF6_FMT) ? -2 : 0;
            for(uint32_t mB = (mantissa_bits - 1); mB >= 0; mB--)
            {
                if((trailing_significand_fp6 >> mB) != 0)
                {
                    trailing_significand_fp6 = (trailing_significand_fp6 << (mantissa_bits - mB))
                                               & ((1 << mantissa_bits) - 1);
                    unbiased_exp_fp6 = unbiased_exp_fp6 - (mantissa_bits - mB);
                    break;
                }
            }
        }
        // at this point, leading significand bit is always 1 for non-zero input

        // apply scale
        unbiased_exp_fp6 += (scale_exp_f32 - 127);

        // at this point the exponent range is the output exponent range

        uint32_t f32 = 0;

        if(scale_exp_f32 == 0xff)
        {
            f32 = (sign_fp6 << 31) | 0x7f8c0000
                  | (trailing_significand_fp6 << (23 - mantissa_bits));
        }
        else if(is_fp6_pre_scale_zero)
        {
            f32 = (sign_fp6 << 31) | 0x0;
        }
        else
        {
            if(unbiased_exp_fp6 < -149)
            {
                // scaled number is less than f32 min subnorm; output 0
                f32 = ((sign_fp6 << 31) | 0x0);
            }
            else if(unbiased_exp_fp6 < -126)
            {
                // scaled number is in f32 subnorm range,
                //  adjust mantissa such that unbiased_exp_fp6 is -126 and apply rne
                int32_t exp_shift        = -126 - unbiased_exp_fp6;
                int32_t unbiased_exp_f32 = unbiased_exp_fp6 + exp_shift;
                assert(unbiased_exp_f32 == -126);
                uint32_t trail_sig_fp32
                    = (1 << 31) | (trailing_significand_fp6 << (31 - mantissa_bits));
                trail_sig_fp32 >>= exp_shift;
                bool is_sig_ovf = false;
                trail_sig_fp32  = round_fp32_f6_significand_rne(is_sig_ovf, trail_sig_fp32);
                f32             = (sign_fp6 << 31) | ((is_sig_ovf ? 0x01 : 0x00) << 23)
                      | (trail_sig_fp32 & 0x7fffff);
            }
            else if(unbiased_exp_fp6 < +128)
            {
                // scaled number is in f32 normal range
                //  apply rne
                uint32_t biased_exp_f32 = unbiased_exp_fp6 + 127;
                uint32_t trail_sig_fp32
                    = (1 << 31) | (trailing_significand_fp6 << (31 - mantissa_bits));
                bool is_sig_ovf = false;
                trail_sig_fp32  = round_fp32_f6_significand_rne(is_sig_ovf, trail_sig_fp32);
                biased_exp_f32 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f32 == +255)
                {
                    f32 = (sign_fp6 << 31) | 0x7f800000;
                }
                else
                {
                    f32 = (sign_fp6 << 31) | ((biased_exp_f32 & 0xff) << 23)
                          | (trail_sig_fp32 & 0x7fffff);
                }
            }
            else
            {
                // scaled number is greater than f32 max normL output +/- inf
                f32 = (sign_fp6 << 31) | 0x7f800000;
            }
        }

        return *((float*)(&f32));
    }

    uint32_t round_f6_significand_rne(bool&     is_significand_ovf,
                                      f6_format f6_src_fmt,
                                      uint32_t  trail_sig_f6)
    {
        is_significand_ovf = false;
        // trail_sig_f6 is of the form 1.31
        uint32_t mantissa_bits_f6 = (f6_src_fmt == f6_format::BF6_FMT) ? 2 : 3;
        uint32_t trail_significand
            = (trail_sig_f6 >> (31 - mantissa_bits_f6)) & ((1 << mantissa_bits_f6) - 1);
        uint32_t ulp_half_ulp = (trail_sig_f6 >> (31 - mantissa_bits_f6 - 1))
                                & 0x3; // 1.31 >> (31-mantissa_bits_f6-1)
        uint32_t or_remain = (trail_sig_f6 >> 0) & ((1 << (31 - mantissa_bits_f6 - 1)) - 1);
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
        is_significand_ovf = (((trail_significand >> mantissa_bits_f6) & 0x1) == 0x1);
        // trail_significand is of the form .mantissa_bits_f6
        return (trail_significand & ((1 << mantissa_bits_f6) - 1));
    }

    uint6_t f32_to_fp6(uint32_t  in,
                       uint32_t  scale_exp_f32,
                       f6_format f6_src_fmt,
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
                += ((f6_src_fmt == f6_format::BF6_FMT) ? ((in1 & 0xfffff800) >> 11)
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

        uint6_t fp6 = {0};

        if(is_f32_pre_scale_inf || is_f32_pre_scale_nan || (scale_exp_f32 == 0xff))
        {
            fp6.val = (sign_f32 << 5) | 0x1f;
        }
        else if(is_f32_pre_scale_zero)
        {
            fp6.val = (sign_f32 << 5) | 0x0;
        }
        else
        {
            int32_t  min_subnorm_uexp_f6 = (f6_src_fmt == f6_format::BF6_FMT) ? -4 : -3;
            int32_t  max_subnorm_uexp_f6 = (f6_src_fmt == f6_format::BF6_FMT) ? -2 : 0;
            int32_t  max_norm_uexp_f6    = (f6_src_fmt == f6_format::BF6_FMT) ? +4 : +2;
            uint32_t mantissa_bits_f6    = (f6_src_fmt == f6_format::BF6_FMT) ? 2 : 3;
            uint32_t exponent_bits_f6    = (f6_src_fmt == f6_format::BF6_FMT) ? 3 : 2;
            if(unbiased_exp_f32 < min_subnorm_uexp_f6)
            {
                // scaled number is less than f6 min subnorm; output 0
                fp6.val = (sign_f32 << 5) | 0x0;
            }
            else if(unbiased_exp_f32 < max_subnorm_uexp_f6)
            {
                // scaled number is in f6 subnorm range,
                //  adjust mantissa such that unbiased_exp_f32 is
                //  max_subnorm_uexp_f6 and apply rne
                int32_t exp_shift       = max_subnorm_uexp_f6 - unbiased_exp_f32;
                int32_t unbiased_exp_f6 = unbiased_exp_f32 + exp_shift;
                assert(unbiased_exp_f6 == max_subnorm_uexp_f6);
                uint32_t trail_sig_f6 = (1 << 31) | (trailing_significand_f32 << 8);
                trail_sig_f6 >>= exp_shift;
                bool is_sig_ovf = false;
                trail_sig_f6    = round_f6_significand_rne(is_sig_ovf, f6_src_fmt, trail_sig_f6);
                fp6.val         = (sign_f32 << 5)
                          | ((uint8_t)((is_sig_ovf ? 0x01 : 0x00) << mantissa_bits_f6))
                          | (trail_sig_f6 & ((1 << mantissa_bits_f6) - 1));
            }
            else if(unbiased_exp_f32 <= max_norm_uexp_f6)
            {
                // scaled number is in f6 normal range
                //  apply rne
                int32_t biased_exp_f6
                    = unbiased_exp_f32 + ((f6_src_fmt == f6_format::BF6_FMT) ? 3 : 1);
                uint32_t trail_sig_f6 = (1 << 31) | (trailing_significand_f32 << 8);
                bool     is_sig_ovf   = false;
                trail_sig_f6 = round_f6_significand_rne(is_sig_ovf, f6_src_fmt, trail_sig_f6);
                biased_exp_f6 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f6
                   == (max_norm_uexp_f6 + ((f6_src_fmt == f6_format::BF6_FMT) ? 3 : 1) + 1))
                {
                    fp6.val = (sign_f32 << 5) | 0x1f;
                }
                else
                {
                    fp6.val
                        = (sign_f32 << 5)
                          | ((biased_exp_f6 & ((1 << exponent_bits_f6) - 1)) << mantissa_bits_f6)
                          | (trail_sig_f6 & ((1 << mantissa_bits_f6) - 1));
                }
            }
            else
            {
                // scaled number is greater than f6 max normal output
                //  clamp to f6 flt_max
                fp6.val = (sign_f32 << 5) | 0x1f;
            }
        }

        return fp6;
    }
}

struct gfx950_fp6
{
    uint6_t data;

    // default constructor
    gfx950_fp6() = default;

    explicit gfx950_fp6(float v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
    {
        bool SR = (rm == rounding_mode::stochastic);
        data    = gfx950_f6_impl::f32_to_fp6(*((uint32_t*)(&v)), 127, f6_format::FP6_FMT, SR, rng);
    }

    // Constructor from half
    explicit gfx950_fp6(_Float16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_fp6((float)v, rm, rng)
    {
    }

    // constructor from bfloat16
    explicit gfx950_fp6(bfloat16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_fp6((float)v, rm, rng)
    {
    }

    explicit inline operator float() const
    {
        return gfx950_f6_impl::fp6_to_f32(data, 127, f6_format::FP6_FMT);
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

    // assignment overloading only from the same F6 types
    inline gfx950_fp6& operator=(const gfx950_fp6& a)
    {
        data = a.data;
        return *this;
    }
};

struct gfx950_bf6
{
    uint6_t data;

    // default constructor
    gfx950_bf6() = default;

    explicit gfx950_bf6(float v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
    {
        bool SR = (rm == rounding_mode::stochastic);
        data    = gfx950_f6_impl::f32_to_fp6(*((uint32_t*)(&v)), 127, f6_format::BF6_FMT, SR, rng);
    }

    // Constructor from half
    explicit gfx950_bf6(_Float16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_bf6((float)v, rm, rng)
    {
    }

    // constructor from bfloat16
    explicit gfx950_bf6(bfloat16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_bf6((float)v, rm, rng)
    {
    }

    explicit inline operator float() const
    {
        return gfx950_f6_impl::fp6_to_f32(data, 127, f6_format::BF6_FMT);
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

    // assignment overloading only from the same F6 types
    inline gfx950_bf6& operator=(const gfx950_bf6& a)
    {
        data = a.data;
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const gfx950_fp6& fp6)
{
    return os << float(fp6);
}

inline std::ostream& operator<<(std::ostream& os, const gfx950_bf6& bf6)
{
    return os << float(bf6);
}

// get uint8 at row i, column j from matrix of m rows, n columns
// static uint6_t getFp6(cl_char* buffer, int m, int n, int i, int j, int rowmajor)
// {
//     int index = 0;
//     if(rowmajor == 1)
//     {
//         index = i * n + j;
//     }
//     else
//     {
//         index = j * m + i;
//     }
//     int p1, p2, cp1;
//     p1  = index / 4;
//     p2  = index % 4;
//     cp1 = p1 * 3;

//     char temp1 = 0;
//     char temp2 = 0;

//     uint6_t ret = {0};
//     switch(p2)
//     {
//     case 0:
//         temp1   = buffer[cp1];
//         ret.val = temp1 & 0x3f;
//         break;
//     case 1:
//         temp1   = buffer[cp1];
//         temp2   = buffer[cp1 + 1];
//         ret.val = ((temp1 & 0xc0) >> 6) | ((temp2 & 0xf) << 2);
//         break;
//     case 2:
//         temp1   = buffer[cp1 + 1];
//         temp2   = buffer[cp1 + 2];
//         ret.val = ((temp1 & 0xf0) >> 4) | ((temp2 & 0x3) << 4);
//         break;
//     case 3:
//         temp1   = buffer[cp1 + 2];
//         ret.val = (temp1 & 0xfc) >> 2;
//         break;
//     }

//     return ret;
// }

// static void setFp6(cl_char* buffer, uint6_t value, int m, int n, int i, int j, int rowmajor)
// {
//     int index = 0;
//     if(rowmajor == 1)
//     {
//         index = i * n + j;
//     }
//     else
//     {
//         index = j * m + i;
//     }
//     int p1, p2, cp1;
//     p1  = index / 4;
//     p2  = index % 4;
//     cp1 = p1 * 3;

//     char temp1 = 0;
//     char temp2 = 0;
//     char save  = value.val;
//     switch(p2)
//     {
//     case 0:
//         temp1       = buffer[cp1];
//         buffer[cp1] = (temp1 & 0xc0) | save;
//         break;
//     case 1:
//         temp1           = buffer[cp1];
//         temp2           = buffer[cp1 + 1];
//         buffer[cp1]     = ((save & 0x3) << 6) | (temp1 & 0x3f);
//         buffer[cp1 + 1] = (temp2 & 0xf) | ((save & 0x3c) >> 2);
//         break;
//     case 2:
//         temp1           = buffer[cp1 + 1];
//         temp2           = buffer[cp1 + 2];
//         buffer[cp1 + 1] = ((save & 0xf) << 4) | (temp1 & 0xf);
//         buffer[cp1 + 2] = ((save & 0x30) >> 4) | (temp2 & 0x3);
//         break;
//     case 3:
//         temp1           = buffer[cp1 + 2];
//         buffer[cp1 + 2] = (save << 2) | (temp1 & 0x3);
//         break;
//     }
// }

// //buffer is row major, m, n is size of uint6
// static void initFp6MatrixInRowMajor(cl_char* buffer,
//                                     int      m,
//                                     int      n,
//                                     int      init_pattern,
//                                     uint32_t   scale_exp_f32    = 127,
//                                     uint32_t   f6_src_fmt       = FP6_FMT,
//                                     bool     stochastic_round = false,
//                                     uint32_t   in1              = 0)
// {
//     for(int i = 0; i < m; i++)
//     {
//         for(int j = 0; j < n; j++)
//         {
//             int      offset = i * n + j;
//             uint32_t temp_var;
//             switch(init_pattern)
//             {
//             case 1:
//                 temp_var = FloatMapToInt(cos(offset));
//                 break;
//             case 2:
//                 temp_var = FloatMapToInt(sin(offset));
//                 break;
//             case 3:
//                 temp_var = FloatMapToInt(cos(offset) + sin(offset));
//                 break;
//             case 4:
//                 temp_var = i;
//                 break;
//             case 5:
//                 temp_var = j;
//                 break;
//             case 6:
//                 temp_var = FloatMapToInt(i);
//                 break;
//             case 7:
//                 temp_var = FloatMapToInt(j);
//                 break;
//             case 8:
//                 temp_var = FloatMapToInt(1.0);
//                 break;
//             case 9:
//                 temp_var = FloatMapToInt(0.5);
//                 break;
//             case 10:
//                 temp_var = FloatMapToInt(0.25);
//                 break;
//             default:
//                 temp_var = 0;
//                 break;
//             }
//             if((init_pattern == 4) || (init_pattern == 5)) //test mode
//             {
//                 uint6_t value;
//                 value.val = temp_var;
//                 setFp6(buffer, value, m, n, i, j, 1);
//             }
//             else
//             {
//                 uint6_t value
//                     = f32_to_fp6(temp_var, scale_exp_f32, f6_src_fmt, stochastic_round, in1);
//                 setFp6(buffer, value, m, n, i, j, 1);
//             }
//         }
//     }
// }

// //dump function
// static void dumpFp6MatrixInHex(cl_char* data, const char* fileName, int m, int n, int rowmajor)
// {
//     FILE* file = fopen(fileName, "w+t");
//     for(int i = 0; i < m; i++)
//     {
//         for(int j = 0; j < n; j++)
//         {
//             uint6_t value = getFp6(data, m, n, i, j, rowmajor);
//             uint    ret   = value.val;
//             fprintf(file, "%02x ", ret);
//         }
//         fprintf(file, "\n");
//     }

//     fclose(file);
// }

// static void dumpFp6MatrixInFloat(cl_char*    data,
//                                  const char* fileName,
//                                  int         m,
//                                  int         n,
//                                  int         rowmajor,
//                                  uint32_t      scale_exp_f32 = 127,
//                                  uint32_t      f8_src_fmt    = FP6_FMT)
// {
//     FILE* file = fopen(fileName, "w+t");
//     for(int i = 0; i < m; i++)
//     {
//         for(int j = 0; j < n; j++)
//         {
//             float value = IntMapToFloat(
//                 fp6_to_f32(getFp6(data, m, n, i, j, rowmajor), scale_exp_f32, f8_src_fmt));
//             fprintf(file, "%f ", value);
//         }
//         fprintf(file, "\n");
//     }

//     fclose(file);
// }

// static void printFp6Matrix(cl_char* buffer,
//                            int      m,
//                            int      n,
//                            int      rowmajor,
//                            uint32_t   scale_exp_f32 = 127,
//                            uint32_t   f8_src_fmt    = FP6_FMT)
// {
//     for(int i = 0; i < m; i++)
//     {
//         for(int j = 0; j < n; j++)
//         {
//             float value = IntMapToFloat(
//                 fp6_to_f32(getFp6(buffer, m, n, i, j, rowmajor), scale_exp_f32, f8_src_fmt));
//             printf("%f ", value);
//         }
//         printf("\n");
//     }
// }

// //do transpose for f6
// //input is M x K, output is K x M
// //m.k is the pixel size for the matrix, should be align to 4
// static int transposeFp6(cl_char* des, cl_char* src, int m, int k, int mode)
// {
//     //check the size alignment
//     if((m & 0x3) || (k & 0x3))
//     {
//         printf("unsupport matrix size %d x %d\n", m, k);
//         printf("matrix size should align to 4\n");
//         return 1;
//     }

//     //start to get data
//     for(int i = 0; i < m; i++)
//     {

//         for(int j = 0; j < k; j += 4)
//         {

//             //fetch data
//             uint6_t data[4] = {0};
//             for(int t = 0; t < 4; t++)
//                 data[t] = getFp6(src, m, k, i, j + t, 1);

//             //save data
//             for(int tt = 0; tt < 4; tt++)
//                 setFp6(des, data[tt], k, m, j + tt, i, 1);
//         }
//     }
//     return 0;
// }

#endif
