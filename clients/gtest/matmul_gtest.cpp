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
#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "testing_matmul.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // matmul
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename TiA,
              typename TiB = TiA,
              typename To  = TiB,
              typename Tc  = To,
              typename Tci = TiA,
              typename     = void>
    struct matmul_testing : hipblaslt_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename TiA, typename TiB, typename To, typename Tc, typename Tci>
    struct matmul_testing<
        TiA,
        TiB,
        To,
        Tc,
        Tci,
        std::enable_if_t<
            (std::is_same<TiA, hipblasLtHalf>{} && std::is_same<TiB, hipblasLtHalf>{})
            || (std::is_same<TiA, hip_bfloat16>{} && std::is_same<TiB, hip_bfloat16>{})
            || (std::is_same<TiA, float>{} && std::is_same<TiB, float>{})
            || (std::is_same<TiA, hipblaslt_f8_fnuz>{} && std::is_same<TiB, hipblaslt_f8_fnuz>{})
            || (std::is_same<TiA, hipblaslt_bf8_fnuz>{} && std::is_same<TiB, hipblaslt_f8_fnuz>{})
            || (std::is_same<TiA, hipblaslt_f8_fnuz>{} && std::is_same<TiB, hipblaslt_bf8_fnuz>{})
            || (std::is_same<TiA, double>{} && std::is_same<TiB, double>{})
            || (std::is_same<TiA, hipblasLtInt8>{} && std::is_same<TiB, hipblasLtInt8>{})
            || (std::is_same<TiA, hipblaslt_f8_fnuz>{} && std::is_same<TiB, hipblasLtHalf>{})
            || (std::is_same<TiA, hipblasLtHalf>{} && std::is_same<TiB, hipblaslt_f8_fnuz>{})>>
        : hipblaslt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "matmul"))
                testing_matmul<TiA, TiB, To, Tc, Tci>(arg);
            else if(!strcmp(arg.function, "matmul_bad_arg"))
                testing_matmul_bad_arg<TiA, TiB, To, Tc, Tci>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct matmul_test : RocBlasLt_Test<matmul_test, matmul_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipblaslt_matmul_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "matmul") || !strcmp(arg.function, "matmul_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBlasLt_TestName<matmul_test> name(arg.name);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "bad_arg";
            }
            else
            {
                name << hip_datatype_to_string(arg.a_type) << hip_datatype_to_string(arg.b_type)
                     << hip_datatype_to_string(arg.c_type) << hip_datatype_to_string(arg.d_type)
                     << hipblaslt_computetype_to_string(arg.compute_type);

                if(arg.activation_type != hipblaslt_activation_type::none)
                {
                    name << '_' << hipblaslt_activation_type_to_string(arg.activation_type);
                }

                if(arg.bias_vector)
                {
                    name << "_BIAS" << hipblaslt_bias_source_to_string(arg.bias_source);
                    if(arg.d_type != arg.scale_type && arg.bias_type == arg.scale_type)
                        name << hip_datatype_to_string(arg.bias_type);
                }

                if(arg.gradient)
                {
                    if(arg.use_e)
                    {
                        name << "_GRAD";
                    }
                }
                else
                {
                    if(arg.use_e)
                    {
                        name << "_AUX";
                    }
                }

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M[0] << '_' << arg.N[0] << '_' << arg.K[0] << '_' << arg.alpha
                     << '_' << arg.lda[0] << '_' << arg.ldb[0] << '_' << arg.beta << '_'
                     << arg.ldc[0] << '_' << arg.ldd[0];

                if(arg.use_e)
                {
                    name << '_' << arg.lde[0];
                }

                name << '_' << arg.batch_count;

                if(arg.scaleA)
                    name << "_SA";

                if(arg.scaleB)
                    name << "_SB";

                if(arg.scaleC)
                    name << "_SC";

                if(arg.scaleD)
                    name << "_SD";

                if(arg.scaleE)
                    name << "_SAux";

                if(arg.scaleAlpha_vector)
                    name << "_SAV";

                if(arg.grouped_gemm > 0)
                    name << "_GG" << arg.grouped_gemm;

                // grouped gemm only supports ext
                if(arg.use_ext || arg.grouped_gemm > 0)
                    name << "_APIExt";
                if(arg.use_ext_setproblem)
                    name << "_APIExtSet";
                if(arg.algo_method == 2)
                    name << "_APIAlgoIndex";
                else if(arg.algo_method == 1)
                    name << "_APIFindAllAlgo";
                if(arg.use_user_args)
                    name << "_UserArgs";
            }

            return std::move(name);
        }
    };

    TEST_P(matmul_test, matmul)
    {
        RUN_TEST_ON_THREADS_STREAMS(hipblaslt_matmul_dispatch<matmul_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(matmul_test);

} // namespace
