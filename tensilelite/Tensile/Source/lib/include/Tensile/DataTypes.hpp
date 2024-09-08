/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>

#include <Tensile/Comparison.hpp>

#include <Tensile/DataTypes_BFloat16.hpp>
#include <Tensile/DataTypes_Float8_BFloat8.hpp>
#include <Tensile/DataTypes_Half.hpp>
#include <Tensile/DataTypes_Int8.hpp>
#include <Tensile/DataTypes_Int8x4.hpp>
#include <Tensile/DataTypes_XFloat32.hpp>

namespace Tensile
{
    /**
 * \ingroup Tensile
 * \defgroup DataTypes Data Type Info
 *
 * @brief Definitions and metadata on supported data types.
 */

    /**
 * \ingroup DataTypes
 * @{
 */

    /**
 * Data Type
 */
    enum class DataType : int
    {
        Float,
        Double,
        ComplexFloat,
        ComplexDouble,
        Half,
        Int8x4,
        Int32,
        BFloat16,
        Int8,
        Float8,
        BFloat8,
        XFloat32,
        Float8BFloat8,
        BFloat8Float8,
        Count,
        None = Count
    };

    std::string   ToString(DataType d);
    std::string   TypeAbbrev(DataType d);
    size_t        GetElementSize(DataType d);
    std::ostream& operator<<(std::ostream& stream, DataType const& t);
    std::istream& operator>>(std::istream& stream, DataType& t);

    /**
 * \ingroup DataTypes
 * \brief Runtime accessible data type metadata
 */
    struct DataTypeInfo
    {
        static DataTypeInfo const& Get(int index);
        static DataTypeInfo const& Get(DataType t);
        static DataTypeInfo const& Get(std::string const& str);

        DataType    dataType;
        std::string name;
        std::string abbrev;

        size_t elementSize;
        size_t packing;
        size_t segmentSize;

        bool isComplex;
        bool isIntegral;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <typename T>
        static void registerTypeInfo();

        static void addInfoObject(DataTypeInfo const& info);

        static std::map<DataType, DataTypeInfo>* getData();
        static std::map<std::string, DataType>*  getTypeNames();
    };

    /**
 * \ingroup DataTypes
 * \brief Compile-time accessible data type metadata.
 */
    template <typename T>
    struct TypeInfo
    {
    };

    template <typename T, DataType T_Enum, int T_Packing, bool T_IsComplex, bool T_IsIntegral>
    struct BaseTypeInfo
    {
        constexpr static DataType Enum = T_Enum;

        /// Bytes of one element.  May contain multiple segments.
        constexpr static size_t ElementSize = sizeof(T);
        /// Segments per element.
        constexpr static size_t Packing = T_Packing;
        /// Bytes per segment.
        constexpr static size_t SegmentSize = ElementSize / Packing;

        constexpr static bool IsComplex  = T_IsComplex;
        constexpr static bool IsIntegral = T_IsIntegral;

        static inline std::string Name()
        {
            return ToString(Enum);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Enum);
        }
    };

    template <typename T, DataType T_Enum, int T_Packing, bool T_IsComplex, bool T_IsIntegral>
    constexpr DataType BaseTypeInfo<T, T_Enum, T_Packing, T_IsComplex, T_IsIntegral>::Enum;
    template <typename T, DataType T_Enum, int T_Packing, bool T_IsComplex, bool T_IsIntegral>
    constexpr size_t BaseTypeInfo<T, T_Enum, T_Packing, T_IsComplex, T_IsIntegral>::ElementSize;
    template <typename T, DataType T_Enum, int T_Packing, bool T_IsComplex, bool T_IsIntegral>
    constexpr size_t BaseTypeInfo<T, T_Enum, T_Packing, T_IsComplex, T_IsIntegral>::Packing;
    template <typename T, DataType T_Enum, int T_Packing, bool T_IsComplex, bool T_IsIntegral>
    constexpr size_t BaseTypeInfo<T, T_Enum, T_Packing, T_IsComplex, T_IsIntegral>::SegmentSize;

    template <typename T, DataType T_Enum, int T_Packing, bool T_IsComplex, bool T_IsIntegral>
    constexpr bool BaseTypeInfo<T, T_Enum, T_Packing, T_IsComplex, T_IsIntegral>::IsComplex;
    template <typename T, DataType T_Enum, int T_Packing, bool T_IsComplex, bool T_IsIntegral>
    constexpr bool BaseTypeInfo<T, T_Enum, T_Packing, T_IsComplex, T_IsIntegral>::IsIntegral;

    template <>
    struct TypeInfo<float> : public BaseTypeInfo<float, DataType::Float, 1, false, false>
    {
    };
    template <>
    struct TypeInfo<double> : public BaseTypeInfo<double, DataType::Double, 1, false, false>
    {
    };
    template <>
    struct TypeInfo<std::complex<float>>
        : public BaseTypeInfo<std::complex<float>, DataType::ComplexFloat, 1, true, false>
    {
    };
    template <>
    struct TypeInfo<std::complex<double>>
        : public BaseTypeInfo<std::complex<double>, DataType::ComplexDouble, 1, true, false>
    {
    };

    template <>
    struct TypeInfo<Int8x4> : public BaseTypeInfo<Int8x4, DataType::Int8x4, 4, false, true>
    {
    };

    template <>
    struct TypeInfo<int32_t> : public BaseTypeInfo<int32_t, DataType::Int32, 1, false, true>
    {
    };

    template <>
    struct TypeInfo<Half> : public BaseTypeInfo<Half, DataType::Half, 1, false, false>
    {
    };
    template <>
    struct TypeInfo<BFloat16> : public BaseTypeInfo<BFloat16, DataType::BFloat16, 1, false, false>
    {
    };

    // Enum DataType::Int8 maps to int8_t, struct Tensile::Int8 is only used for LogTensor now
    template <>
    struct TypeInfo<int8_t> : public BaseTypeInfo<int8_t, DataType::Int8, 1, false, true>
    {
    };

    template <>
    struct TypeInfo<Float8> : public BaseTypeInfo<Float8, DataType::Float8, 1, false, false>
    {
    };

    template <>
    struct TypeInfo<BFloat8> : public BaseTypeInfo<BFloat8, DataType::BFloat8, 1, false, false>
    {
    };

    template <>
    struct TypeInfo<XFloat32> : public BaseTypeInfo<XFloat32, DataType::XFloat32, 1, false, false>
    {
    };

    template <>
    struct TypeInfo<Float8BFloat8>
        : public BaseTypeInfo<Float8BFloat8, DataType::Float8BFloat8, 1, false, false>
    {
    };

    template <>
    struct TypeInfo<BFloat8Float8>
        : public BaseTypeInfo<BFloat8Float8, DataType::BFloat8Float8, 1, false, false>
    {
    };

    // Variant for constants
    using ConstantVariant = std::variant<float,
                                         double,
                                         std::complex<float>,
                                         std::complex<double>,
                                         Half,
                                         Int8x4,
                                         int32_t,
                                         BFloat16,
                                         Float8,
                                         BFloat8,
                                         int8_t>;

    // Convert variants to type T
    template <typename T>
    typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value
                                || std::is_same<Half, T>::value || std::is_same<int32_t, T>::value
                                || std::is_same<BFloat16, T>::value
                                || std::is_same<int8_t, T>::value || std::is_same<Float8, T>::value
                                || std::is_same<BFloat8, T>::value,
                            T>::type
        constVariantCast(const ConstantVariant& val)
    {
        switch(val.index())
        {
        case static_cast<int>(DataType::Float):
            return static_cast<T>(*std::get_if<float>(&val));
        case static_cast<int>(DataType::Double):
            return static_cast<T>(*std::get_if<double>(&val));
        case static_cast<int>(DataType::Half):
            return static_cast<T>(*std::get_if<Half>(&val));
        case static_cast<int>(DataType::Int32):
            return static_cast<T>(*std::get_if<int32_t>(&val));
        case static_cast<int>(DataType::BFloat16):
            return static_cast<T>(*std::get_if<BFloat16>(&val));
        case static_cast<int>(DataType::Int8):
            return static_cast<T>(*std::get_if<int8_t>(&val));
        case static_cast<int>(DataType::Float8):
            return static_cast<T>(*std::get_if<Float8>(&val));
        case static_cast<int>(DataType::BFloat8):
            return static_cast<T>(*std::get_if<BFloat8>(&val));
        default:
            throw std::runtime_error("Unsupported variant cast type.");
        }
    }

    template <typename T>
    typename std::enable_if<std::is_same<std::complex<double>, T>::value
                                || std::is_same<std::complex<float>, T>::value,
                            T>::type
        constVariantCast(const ConstantVariant& val)
    {
        switch(val.index())
        {
        case static_cast<int>(DataType::ComplexFloat):
            return static_cast<T>(*std::get_if<std::complex<float>>(&val));
        case static_cast<int>(DataType::ComplexDouble):
            return static_cast<T>(*std::get_if<std::complex<double>>(&val));
        default:
            throw std::runtime_error("Unsupported variant cast type.");
        }
    }

    template <typename T>
    typename std::enable_if<std::is_same<Int8x4, T>::value, T>::type
        constVariantCast(const ConstantVariant& val)
    {
        return static_cast<T>(*std::get_if<Int8x4>(&val));
    }

    std::string ToString(ConstantVariant d);
    bool        CompareValue(const ConstantVariant& d, double value);

    /**
 * @}
 */
} // namespace Tensile
