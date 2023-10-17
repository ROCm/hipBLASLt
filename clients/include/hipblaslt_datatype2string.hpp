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

#include "auxiliary.hpp"
#include "hipblaslt_ostream.hpp"
#include <hipblaslt/hipblaslt.h>
#include <string>

enum class hipblaslt_initialization
{
    rand_int   = 111,
    trig_float = 222,
    hpl        = 333,
    special    = 444,
};

typedef enum class _hipblaslt_activation_type
{
    none = 1,
    relu = 2,
    gelu = 3,
} hipblaslt_activation_type;

typedef enum class _hipblaslt_bias_source
{
    a = 1,
    b = 2,
    d = 3,
} hipblaslt_bias_source;

inline hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                              hipblaslt_activation_type   act)
{
    switch(act)
    {
    case hipblaslt_activation_type::none:
        os << "none";
        break;
    case hipblaslt_activation_type::relu:
        os << "relu";
        break;
    case hipblaslt_activation_type::gelu:
        os << "gelu";
        break;
    }
    return os;
}
inline hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                              hipblaslt_bias_source       bias)
{
    switch(bias)
    {
    case hipblaslt_bias_source::a:
        os << "a";
        break;
    case hipblaslt_bias_source::b:
        os << "b";
        break;
    case hipblaslt_bias_source::d:
        os << "d";
        break;
    }
    return os;
}
constexpr auto hipblaslt_initialization2string(hipblaslt_initialization init)
{
    switch(init)
    {
    case hipblaslt_initialization::rand_int:
        return "rand_int";
    case hipblaslt_initialization::trig_float:
        return "trig_float";
    case hipblaslt_initialization::hpl:
        return "hpl";
    case hipblaslt_initialization::special:
        return "special";
    }
    return "invalid";
}

inline hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                              hipblaslt_initialization    init)
{
    return os << hipblaslt_initialization2string(init);
}

// clang-format off
inline hipblaslt_initialization string2hipblaslt_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? hipblaslt_initialization::rand_int   :
        value == "trig_float" ? hipblaslt_initialization::trig_float :
        value == "hpl"        ? hipblaslt_initialization::hpl        :
        value == "special"    ? hipblaslt_initialization::special    :
        static_cast<hipblaslt_initialization>(0);
}
// clang-format on
inline const hipblaslt_activation_type string_to_hipblaslt_activation_type(const std::string& value)
{
    return value == "none"   ? hipblaslt_activation_type::none
           : value == "gelu" ? hipblaslt_activation_type::gelu
           : value == "relu" ? hipblaslt_activation_type::relu
                             : static_cast<hipblaslt_activation_type>(0);
}

inline const hipblaslt_bias_source string_to_hipblaslt_bias_source(const std::string& value)
{
    return value == "a"   ? hipblaslt_bias_source::a
           : value == "b" ? hipblaslt_bias_source::b
           : value == "d" ? hipblaslt_bias_source::d
                          : static_cast<hipblaslt_bias_source>(0);
}

// Convert hipblaslt_activation_type to string
inline const char* hipblaslt_activation_type_to_string(hipblaslt_activation_type type)
{
    switch(type)
    {
    case hipblaslt_activation_type::gelu:
        return "gelu";
    case hipblaslt_activation_type::relu:
        return "relu";
    case hipblaslt_activation_type::none:
        return "none";
    default:
        return "invalid";
    }
}

// Convert hipblaslt_bias_source to string
inline const char* hipblaslt_bias_source_to_string(hipblaslt_bias_source type)
{
    switch(type)
    {
    case hipblaslt_bias_source::a:
        return "a";
    case hipblaslt_bias_source::b:
        return "b";
    case hipblaslt_bias_source::d:
        return "d";
    default:
        return "invalid";
    }
}
