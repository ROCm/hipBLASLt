#ifndef _GFX950_FLOAT4_IMPL_H_
#define _GFX950_FLOAT4_IMPL_H_

#include "gfx950_common.h"

namespace gfx950_f4_impl
{
    int32_t round_fp32_f4_significand_rne(bool& is_significand_ovf, uint32_t trail_sig_fp32)
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

    float fp4_to_f32(uint4_t in, uint32_t scale_exp_f32)
    {
        uint32_t sign_fp4                 = SIGN_FP4(in.val);
        uint32_t trailing_significand_fp4 = MANT_FP4(in.val);
        int32_t  unbiased_exp_fp4         = BEXP_FP4(in.val) - 1;
        bool     is_fp4_pre_scale_zero    = ((in.val & 0x7) == 0x0);
        bool     is_fp4_pre_scale_dnrm    = (SIGN_FP4(in.val) == 0) && (MANT_FP4(in.val) != 0);

        // normalize subnormal number
        if(is_fp4_pre_scale_dnrm)
        {
            trailing_significand_fp4 = 0;
            unbiased_exp_fp4         = -1;
        }
        // at this point, leading significand bit is always 1 for non-zero input

        // apply scale
        unbiased_exp_fp4 += (scale_exp_f32 - 127);

        // at this point the exponent range is the output exponent range

        uint32_t f32 = 0;

        if(scale_exp_f32 == 0xff)
        {
            f32 = (sign_fp4 << 31) | 0x7fc00000 | (trailing_significand_fp4 << 22);
        }
        else if(scale_exp_f32 == 0x7f)
        {
            // Scale is 1.0; Direct conversion
            switch(in.val & 0x7)
            {
            case 0:
                f32 = 0x00000000;
                break; // +-0.0
            case 1:
                f32 = 0x3f000000;
                break; // +-0.5
            case 2:
                f32 = 0x3f800000;
                break; // +-1.0
            case 3:
                f32 = 0x3fc00000;
                break; // +-1.5
            case 4:
                f32 = 0x40000000;
                break; // +-2.0
            case 5:
                f32 = 0x40400000;
                break; // +-3.0
            case 6:
                f32 = 0x40800000;
                break; // +-4.0
            case 7:
                f32 = 0x40c00000;
                break; // +-6.0
            default:
                f32 = 0;
                break;
            }
            f32 |= (sign_fp4 << 31);
        }
        else if(is_fp4_pre_scale_zero)
        {
            f32 = (sign_fp4 << 31) | 0x0;
        }
        else
        {
            if(unbiased_exp_fp4 < -149)
            {
                // scaled number is less than f32 min subnorm; output 0
                f32 = ((sign_fp4 << 31) | 0x0);
            }
            else if(unbiased_exp_fp4 < -126)
            {
                // scaled number is in f32 subnorm range,
                //  adjust mantissa such that unbiased_exp_fp4 is -126 and apply rne
                int32_t exp_shift        = -126 - unbiased_exp_fp4;
                int32_t unbiased_exp_f32 = unbiased_exp_fp4 + exp_shift;
                assert(unbiased_exp_f32 == -126);
                uint32_t trail_sig_fp32 = (1 << 31) | (trailing_significand_fp4 << 30);
                trail_sig_fp32 >>= exp_shift;
                bool is_sig_ovf = false;
                trail_sig_fp32  = round_fp32_f4_significand_rne(is_sig_ovf, trail_sig_fp32);
                f32             = (sign_fp4 << 31) | ((is_sig_ovf ? 0x01 : 0x00) << 23)
                      | (trail_sig_fp32 & 0x7fffff);
            }
            else if(unbiased_exp_fp4 < +128)
            {
                // scaled number is in f32 normal range
                //  apply rne
                uint32_t biased_exp_f32 = unbiased_exp_fp4 + 127;
                uint32_t trail_sig_fp32 = (1 << 31) | (trailing_significand_fp4 << 30);
                bool     is_sig_ovf     = false;
                trail_sig_fp32          = round_fp32_f4_significand_rne(is_sig_ovf, trail_sig_fp32);
                biased_exp_f32 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f32 == +255)
                {
                    f32 = (sign_fp4 << 31) | 0x7f800000;
                }
                else
                {
                    f32 = (sign_fp4 << 31) | ((biased_exp_f32 & 0xff) << 23)
                          | (trail_sig_fp32 & 0x7fffff);
                }
            }
            else
            {
                // scaled number is greater than f32 max normL output +/- inf
                f32 = (sign_fp4 << 31) | 0x7f800000;
            }
        }

        return *((float*)(&f32));
    }

    uint32_t round_f4_significand_rne(bool& is_significand_ovf, uint32_t trail_sig_f4)
    {
        is_significand_ovf = false;
        // trail_sig_f4 is of the form 1.31
        uint32_t trail_significand = (trail_sig_f4 >> 30) & 0x1;
        uint32_t ulp_half_ulp      = (trail_sig_f4 >> 29) & 0x3; // 1.31 >> (31-1-1)
        uint32_t or_remain         = (trail_sig_f4 >> 0) & ((1 << 29) - 1);
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
        is_significand_ovf = (((trail_significand >> 1) & 0x1) == 0x1);
        // trail_significand is of the form .1
        return (trail_significand & 0x1);
    }

    uint4_t f32_to_fp4(uint32_t in, uint32_t scale_exp_f32, bool stochastic_round, uint32_t in1)
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
            trailing_significand_f32 += ((in1 & 0xfffff000) >> 12);
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

        uint4_t fp4 = {0};

        if(is_f32_pre_scale_inf || is_f32_pre_scale_nan || (scale_exp_f32 == 0xff))
        {
            fp4.val = (sign_f32 << 3) | 0x7;
        }
        else if(is_f32_pre_scale_zero)
        {
            fp4.val = (sign_f32 << 3) | 0x0;
        }
        else
        {
            int32_t  min_subnorm_uexp_f4 = -1;
            int32_t  max_subnorm_uexp_f4 = 0;
            int32_t  max_norm_uexp_f4    = +2;
            uint32_t mantissa_bits_f4    = 1;
            uint32_t exponent_bits_f4    = 2;
            if(unbiased_exp_f32 < min_subnorm_uexp_f4)
            {
                // scaled number is less than f4 min subnorm; output 0
                fp4.val = (sign_f32 << 3) | 0x0;
            }
            else if(unbiased_exp_f32 < max_subnorm_uexp_f4)
            {
                // scaled number is in f4 subnorm range,
                //  adjust mantissa such that unbiased_exp_f32 is
                //  max_subnorm_uexp_f4 and apply rne
                int32_t exp_shift       = max_subnorm_uexp_f4 - unbiased_exp_f32;
                int32_t unbiased_exp_f4 = unbiased_exp_f32 + exp_shift;
                assert(unbiased_exp_f4 == max_subnorm_uexp_f4);
                uint32_t trail_sig_f4 = (1 << 31) | (trailing_significand_f32 << 8);
                trail_sig_f4 >>= exp_shift;
                bool is_sig_ovf = false;
                trail_sig_f4    = round_f4_significand_rne(is_sig_ovf, trail_sig_f4);
                fp4.val         = (sign_f32 << 3)
                          | ((uint8_t)((is_sig_ovf ? 0x01 : 0x00) << mantissa_bits_f4))
                          | (trail_sig_f4 & ((1 << mantissa_bits_f4) - 1));
            }
            else if(unbiased_exp_f32 <= max_norm_uexp_f4)
            {
                // scaled number is in f4 normal range
                //  apply rne
                uint32_t biased_exp_f4 = unbiased_exp_f32 + 1;
                uint32_t trail_sig_f4  = (1 << 31) | (trailing_significand_f32 << 8);
                bool     is_sig_ovf    = false;
                trail_sig_f4           = round_f4_significand_rne(is_sig_ovf, trail_sig_f4);
                biased_exp_f4 += (is_sig_ovf ? 1 : 0);
                if(biased_exp_f4 == (uint32_t)(max_norm_uexp_f4 + 1 + 1))
                {
                    fp4.val = (sign_f32 << 3) | 0x7;
                }
                else
                {
                    fp4.val
                        = (sign_f32 << 3)
                          | ((biased_exp_f4 & ((1 << exponent_bits_f4) - 1)) << mantissa_bits_f4)
                          | (trail_sig_f4 & ((1 << mantissa_bits_f4) - 1));
                }
            }
            else
            {
                // scaled number is greater than f4 max normal output
                //  clamp to f4 flt_max
                fp4.val = (sign_f32 << 3) | 0x7;
            }
        }

        return fp4;
    }
}

struct gfx950_fp4
{
    uint4_t data;

    // default constructor
    gfx950_fp4() = default;

    explicit gfx950_fp4(float v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
    {
        bool SR = (rm == rounding_mode::stochastic);
        data    = gfx950_f4_impl::f32_to_fp4(*((uint32_t*)(&v)), 127, SR, rng);
    }

    // Constructor from half
    explicit gfx950_fp4(_Float16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_fp4((float)v, rm, rng)
    {
    }

    // constructor from bfloat16
    explicit gfx950_fp4(bfloat16 v, rounding_mode rm = rounding_mode::standard, uint32_t rng = 0)
        : gfx950_fp4((float)v, rm, rng)
    {
    }

    explicit inline operator float() const
    {
        return gfx950_f4_impl::fp4_to_f32(data, 127);
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

    // // check for zero
    // inline bool is_zero() const
    // {
    //     return gfx950_f4_impl::is_fp8_zero(data, false);
    // }

    // // check for nan
    // inline bool is_nan() const
    // {
    //     return gfx950_f4_impl::is_fp8_nan(data, f8_format::FP8_FMT, false);
    //     return data == 0x80;
    // }

    // // check for inf
    // inline bool is_inf() const
    // {
    //     return gfx950_f4_impl::is_fp8_inf(data, f8_format::FP8_FMT, false);
    // }

    // assignment overloading only from the same F4 types
    inline gfx950_fp4& operator=(const gfx950_fp4& a)
    {
        data = a.data;
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const gfx950_fp4& fp4)
{
    return os << float(fp4);
}

// static uint4_t getlow(cl_char twoFp4) {
//     uint ret = twoFp4 & 0xf;
//     uint4_t fp4 = {0};
//     fp4.val = ret;
//     return fp4;
// }

// static uint4_t gethigh(cl_char twoFp4) {
//     uint ret = (twoFp4 >> 4) & 0xf;
//     uint4_t fp4 = {0};
//     fp4.val = ret;
//     return fp4;
// }

// static uint4_t getFp4(cl_char twoFp4, int high) {
//     if (high == 1) {
//         return gethigh(twoFp4);
//     } else {
//         return getlow(twoFp4);
//     }
// }

// static void setlow(cl_char* twoFp4, uint4_t fp4) {
//     uint value = fp4.val;
//     *twoFp4 = *twoFp4 & 0xf0;
//     value = value & 0xf;
//    *twoFp4 = *twoFp4 | value;
// }

// static void sethigh(cl_char* twoFp4, uint4_t fp4) {
//     uint value = fp4.val;
//     *twoFp4 = *twoFp4 & 0x0f;
//     value = value & 0xf;
//     value = value << 4;
//     *twoFp4 = *twoFp4 | value;
// }

// static void setFp4(cl_char* twoFp4, uint4_t value, int high) {
//     if (high == 1) {
//         sethigh(twoFp4, value);
//     } else {
//         setlow(twoFp4, value);
//     }
// }

// // get uint4 at row i, column j from matrix of m rows, n columns
// static uint4_t getFp4(cl_char* buffer, int m, int n, int i, int j, int rowmajor, bool debug=false) {
//     int index = 0;
//     if (rowmajor == 1) {
//         index = i*n + j;
//     } else {
//         index = j*m + i;
//     }
//     int high = index % 2;

//     cl_char twoFp4 = buffer[index/2];
//     uint4_t ret = getFp4(twoFp4, high);
//     if (debug) {
//         printf("m:%d, n:%d, i:%d, j:%d, index:%d, rowmajor:%d, ret:%01x\n",m,n,i,j,index,rowmajor,ret.val);
//     }
//     return ret;
// }

// static void setFp4(cl_char* buffer, uint4_t value, int m, int n, int i, int j, int rowmajor) {
//     int index = 0;
//     if (rowmajor == 1) {
//         index = i*n + j;
//     } else {
//         index = j*m + i;
//     }
//     int high = index % 2;

//     setFp4(buffer+index/2, value, high);
// }

// //buffer is row major, m, n is size of uint4
// static void initFp4MatrixInRowMajor(cl_char* buffer, int m, int n, int init_pattern) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             int offset = i*n + j;
//             float temp_var;
//             switch (init_pattern)
//             {
//                 case 1:
//                     temp_var = cos(offset);
//                     break;
//                 case 2:
//                     temp_var = sin(offset);
//                     break;
//                 case 3:
//                     temp_var = cos(offset) + sin(offset);
//                     break;
//                 case 10:
//                     temp_var = 0.25;
//                     break;
//                 default:
//                     temp_var = 0;
//                     break;
//             }
//             uint4_t value = f32_to_fp4(TACL_gemm_ns::FloatMapToInt(temp_var), 127, false, 0);
//             setFp4(buffer, value, m, n, i, j, 1);
//         }
//     }
// }

// //rowin is row major without padding; columnout is column major with padding
// static void padFp4RowToColumn(cl_char *columnout, cl_char *rowin, int m, int n, int pad) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             uint4_t value = getFp4(rowin, m, n, i, j, 1);
//             setFp4(columnout, value, m+pad, n, i, j, 0);
//         }
//     }
// }

// //rowin is row major without padding; rowout is row major with padding
// static void padFp4RowToRow(cl_char *rowout, cl_char *rowin, int m, int n, int pad) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             uint4_t value = getFp4(rowin, m, n, i, j, 1);
//             setFp4(rowout, value, m, n+pad, i, j, 1);
//         }
//     }
// }

// //columnin is column major with padding; rowout is row major without padding
// static void unpadFp4ColumnToRow(cl_char *rowout, cl_char *columnin, int m, int n, int pad) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             uint4_t value = getFp4(columnin, m+pad, n, i, j, 0);
//             setFp4(rowout, value, m, n, i, j, 1);
//         }
//     }
// }

// //rowin is row major with padding; rowout is row major without padding
// static void unpadFp4RowToRow(cl_char *rowout, cl_char *rowin, int m, int n, int pad) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             uint4_t value = getFp4(rowin, m, n+pad, i, j, 1);
//             setFp4(rowout, value, m, n, i, j, 1);
//         }
//     }
// }

// static void printFp4Matrix(cl_char* buffer, int m, int n, int rowmajor) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float value = TACL_gemm_ns::IntMapToFloat(fp4_to_f32(getFp4(buffer, m, n, i, j, rowmajor), 127));
//             printf("%f ", value);
//         }
//         printf("\n");
//     }
// }

// static void dumpFp4MatrixInHex(cl_char *data, const char *fileName, int m, int n, int rowmajor) {
//     FILE *file = fopen(fileName, "w+t");
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             uint value = getFp4(data, m, n, i, j, rowmajor).val;
//             fprintf(file, "%01x ", value);
//         }
//         fprintf(file, "\n");
//     }

//     fclose(file);
// }

// static void dumpFp4MatrixInFloat(cl_char *data, const char *fileName, int m, int n, int rowmajor) {
//     FILE *file = fopen(fileName, "w+t");
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float value = TACL_gemm_ns::IntMapToFloat(fp4_to_f32(getFp4(data, m, n, i, j, rowmajor), 127));
//             fprintf(file, "%f ", value);
//         }
//         fprintf(file, "\n");
//     }

//     fclose(file);
// }

// /*
// //matrix all row major, a: mxk, b: kxn, c:mxn
// static void matrix_mul_Fp4_nopad(cl_char *a, cl_char *b, cl_int *c, int m, int n, int k) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             for (int p = 0; p < k; p++) {
//                 uint value_a = getFp4(a, m, k, i, p, 1);
//                 uint value_b = getFp4(b, k, n, p, j, 1);
//                 c[i*n+j] += value_a * value_b;
//             }
//         }
//     }
// }*/

// static uint32_t check_fp4_result(cl_char *result, cl_char *golden, int Mdim, int Ndim, int pad, int c_col_major)
// {
//     uint32_t SameResult = 1;
//     uint32_t strideC;

//     printf("\n\n>>>>>>>>>>>>> \033[1m\033[;34m  CHECKING RESULTS  \033[0m  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

//     if(c_col_major == 1) {
//         strideC = Mdim + pad;
//     }
//     else {
//         strideC = Ndim + pad;
//     }

//     for(int j = 0; j < Mdim; j++) {
//         for(int i = 0; i < Ndim; i++) {
//             uint4_t rlt, gld;
//             if (c_col_major) {
//                 rlt = getFp4(result, strideC, Ndim, j, i, 0);
//             } else {
//                 rlt = getFp4(result, Mdim, strideC, j, i, 1);
//             }
//             gld = getFp4(golden, Mdim, Ndim, j, i, 1);

//             if (rlt.val != gld.val) {
//                 SameResult = 0;

//                 printf("#######  diff index is [y = %d, x = %d]\n", j, i);
//                 //printf("#######  gpu results      = %f   \n", (c_col_major ? result[i * strideC + j] : result[j * strideC + i]));
//                 //printf("#######  golden           = %f   \n", golden[j * Ndim + i]);

//                 printf("#######  gpu results hex  = %08x    \n", rlt.val);
//                 printf("#######  golden      hex  = %08x    \n", gld.val);

//                 goto PRINT_RESULT;
//             }
//         }
//     }

// PRINT_RESULT:
//     if (SameResult) {
//         printf("\n<<<<<<<<<<<<<<<   \033[1m\033[;32m #TEST PASSED \033[0m                      <<<<<<<<<<<<<<<<<<<<\n\n");
//     }
//     else {
//         printf("\n<<<<<<<<<<<<<<<   \033[1m\033[;35m #TEST FAILED \033[0m!!!!!!!!!! !!!!!!!!!! <<<<<<<<<<<<<<<<<<<<\n\n");
//     }
//     return SameResult;
// }

// static void printFp32Matrix(cl_float* data, int m, int n, int rowmajor) {
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             cl_float *valueP;
//             if (rowmajor == 1) {
//                 valueP = data + (i * n + j);
//             } else {
//                 valueP = data + (j * m + i);
//             }

//             printf("%f ", *valueP);
//         }
//         printf("\n");
//     }
// }

// static void dumpFp32MatrixInHex(cl_int *data, const char *fileName, int m, int n, int rowmajor) {
//     FILE *file = fopen(fileName, "w+t");
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             cl_int *valueP;
//             if (rowmajor == 1) {
//                 valueP = data + (i * n + j);
//             } else {
//                 valueP = data + (j * m + i);
//             }

//             fprintf(file, "%08x ", *valueP);
//         }
//         fprintf(file, "\n");
//     }

//     fclose(file);
// }

// static void dumpFp32MatrixInFloat(cl_float *data, const char *fileName, int m, int n, int rowmajor) {
//     FILE *file = fopen(fileName, "w+t");
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             cl_float *valueP;
//             if (rowmajor == 1) {
//                 valueP = data + (i * n + j);
//             } else {
//                 valueP = data + (j * m + i);
//             }

//             fprintf(file, "%f ", *valueP);
//         }
//         fprintf(file, "\n");
//     }

//     fclose(file);
// }

// static void Dot32FloatScale(float srcA[], float srcB[], float srcC, float & dest, float scaleA, float scaleB, bool debug=false)
// {
//     double result = 0.0f;
//     for (int i = 0; i < 32; i++) {
//         result += (double)srcA[i] * (double)srcB[i];
//         if (debug)
//             printf("+%08x*%08x ", *((uint32_t*)(&srcA[i])), *((uint32_t*)(&srcB[i])));
//     }
//     result *= (double)scaleA;
//     result *= (double)scaleB;
//     result += (double)srcC;
//     if (debug) {
//         printf(" *=%08x *=%08x +=%08x", *((uint32_t*)(&scaleA)),*((uint32_t*)(&scaleB)),*((uint32_t*)(&srcC)));
//     }
//     dest = (float)result;
//     if (debug) {
//         printf("\n   result=%08x\n",*((uint32_t*)(&dest)));
//     }
// }

#endif
