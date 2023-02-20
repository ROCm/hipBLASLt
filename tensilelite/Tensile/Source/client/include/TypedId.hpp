/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/DataTypes.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    constexpr uint32_t GemmTypeId(DataType aType,
                                  DataType bType,
                                  DataType cType,
                                  DataType dType,
                                  DataType alphaType,
                                  DataType betaType)
    {
        static_assert(BitFieldGenerator::ElementWidth((uint32_t)DataType::Count) * 6
                          <= BitFieldGenerator::maxBitFieldWidth,
                      "Max bitfield width exceeded");

        return BitFieldGenerator::GenerateBitField(
            BitFieldGenerator::ElementWidth((uint32_t)DataType::Count),
            (uint32_t)aType,
            (uint32_t)bType,
            (uint32_t)cType,
            (uint32_t)dType,
            (uint32_t)alphaType,
            (uint32_t)betaType);
    }

    template <typename A     = float,
              typename B     = A,
              typename C     = A,
              typename D     = C,
              typename Alpha = D,
              typename Beta  = D>
    struct TypedGemm
    {
        using AType     = A;
        using BType     = B;
        using CType     = C;
        using DType     = D;
        using AlphaType = Alpha;
        using BetaType  = Beta;

        constexpr static uint32_t TypeId()
        {
            return GemmTypeId(TypeInfo<A>::Enum,
                              TypeInfo<B>::Enum,
                              TypeInfo<C>::Enum,
                              TypeInfo<D>::Enum,
                              TypeInfo<Alpha>::Enum,
                              TypeInfo<Beta>::Enum);
        }
    };

    // Commonly used type groupings
    // Naming: _[Ti_To_Tc]_:
    // S=float, D=double, C=complex<float>, Z=complex<double>,
    // H=Half, B=BF16, I8x4=Int8x4, I32=int32_t
    using TypedGemm_S_S_S = TypedGemm<float>;
    using TypedGemm_D_D_D = TypedGemm<double>;
    using TypedGemm_C_C_C = TypedGemm<std::complex<float>>;
    using TypedGemm_Z_Z_Z = TypedGemm<std::complex<double>>;
#ifdef TENSILE_USE_HALF
    using TypedGemm_H_H_H = TypedGemm<Half>;
    using TypedGemm_H_H_S = TypedGemm<Half, Half, Half, Half, float, float>;
    using TypedGemm_H_S_S = TypedGemm<Half, Half, float, float>;
#endif // TENSILE_USE_HALF
    using TypedGemm_I8x4_I32_I32 = TypedGemm<Int8x4, Int8x4, int32_t, int32_t>;
    using TypedGemm_I8_I8_I32    = TypedGemm<int8_t, int8_t, int8_t, int8_t, int32_t, int32_t>;
    using TypedGemm_I8_I32_I32   = TypedGemm<int8_t, int8_t, int32_t, int32_t>;
    using TypedGemm_I8_I32_S     = TypedGemm<int8_t, int8_t, int32_t, int32_t, float, float>;
    using TypedGemm_I8_I8_S      = TypedGemm<int8_t, int8_t, int8_t, int8_t, float, float>;
    using TypedGemm_I32_I32_I32  = TypedGemm<int32_t>;
#ifdef TENSILE_USE_BF16
    using TypedGemm_B_B_S = TypedGemm<BFloat16, BFloat16, BFloat16, BFloat16, float, float>;
    using TypedGemm_B_S_S = TypedGemm<BFloat16, BFloat16, float, float>;
#endif // TENSILE_USE_BF16
} // namespace Tensile
