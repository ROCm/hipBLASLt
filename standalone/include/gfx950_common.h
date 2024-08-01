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

#ifndef _GFX950_F468_COMMON_H_
#define _GFX950_F468_COMMON_H_

#include "bfloat16.h"
#include <float.h>
#include <stdint.h>

#define BEXP32(x) ((x & 0x7f800000) >> 23)
#define MAN32(x) (x & 0x7fffff)
#define SIGN32(x) (x >> 31)

// fp8 (1, 4, 3)
#define BEXP_FP8_GFX950(x) ((x & 0x78) >> 3)
#define MANT_FP8_GFX950(x) (x & 0x7)
#define SIGN_FP8_GFX950(x) ((x >> 7) & 0x1)
#define FP8_GFX950(s, e, m) ((s << 7) | (e << 3) | (m & 0x7))

// bf8 (1, 5, 2)
#define BEXP_BF8_GFX950(x) ((x & 0x7c) >> 2)
#define MANT_BF8_GFX950(x) (x & 0x3)
#define SIGN_BF8_GFX950(x) ((x >> 7) & 0x1)
#define BF8_GFX950(s, e, m) ((s << 7) | (e << 2) | (m & 0x3))

// fp6 (1, 2, 3)
#define BEXP_FP6(x) ((x & 0x18) >> 3)
#define MANT_FP6(x) (x & 0x7)
#define SIGN_FP6(x) ((x >> 5) & 0x1)
#define FP6(s, e, m) ((s << 5) | (e << 3) | (m & 0x7))

// bf6 (1, 3, 2)
#define BEXP_BF6(x) ((x & 0x1c) >> 2)
#define MANT_BF6(x) (x & 0x3)
#define SIGN_BF6(x) ((x >> 5) & 0x1)
#define BF6(s, e, m) ((s << 5) | (e << 2) | (m & 0x3))

// fp4 (1, 2, 1)
#define BEXP_FP4(x) ((x & 0x6) >> 1)
#define MANT_FP4(x) (x & 0x1)
#define SIGN_FP4(x) ((x >> 3) & 1)
#define FP4(s, e, m) ((s << 3) | (e << 1) | (m & 0x1))

//added for fp4
typedef struct
{
    uint32_t val : 4;
} uint4_t;

typedef struct
{
    uint32_t val : 6;
} uint6_t;

enum class rounding_mode
{
    standard,
    stochastic
};

#endif
