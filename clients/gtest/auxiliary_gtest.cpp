/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include "testing_auxiliary.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // aux
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
    struct aux_testing : hipblaslt_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename Ti, typename To, typename Tc>
    struct aux_testing<
        Ti,
        To,
        Tc,
        std::enable_if_t<std::is_same<Ti, hipblasLtHalf>{} || std::is_same<Ti, hip_bfloat16>{}
                         || std::is_same<Ti, float>{}>> : hipblaslt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "aux_handle_init_bad_arg"))
                testing_aux_handle_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_handle_destroy_bad_arg"))
                testing_aux_handle_destroy_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_handle"))
                testing_aux_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_init_bad_arg"))
                testing_aux_mat_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_init_arg"))
                testing_aux_mat_init(arg);
            else if(!strcmp(arg.function, "aux_mat_destroy_bad_arg"))
                testing_aux_mat_destroy_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg"))
                testing_aux_mat_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg"))
                testing_aux_mat_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_set_get_attr"))
                testing_aux_mat_set_get_attr(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg"))
                testing_aux_matmul_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_init"))
                testing_aux_matmul_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg"))
                testing_aux_matmul_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg"))
                testing_aux_matmul_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr"))
                testing_aux_matmul_set_get_attr(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg"))
                testing_aux_matmul_alg_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init"))
                testing_aux_matmul_alg_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg"))
                testing_aux_matmul_alg_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg"))
                testing_aux_matmul_alg_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg"))
                testing_aux_matmul_pref_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init"))
                testing_aux_matmul_pref_init(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct aux_test : RocBlasLt_Test<aux_test, aux_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipblaslt_matmul_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "aux_handle_init_bad_arg")
                   || !strcmp(arg.function, "aux_handle_destroy_bad_arg")
                   || !strcmp(arg.function, "aux_handle")
                   || !strcmp(arg.function, "aux_mat_init_bad_arg")
                   || !strcmp(arg.function, "aux_mat_init_arg")
                   || !strcmp(arg.function, "aux_mat_destroy_bad_arg")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_mat_set_get_attr")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_init")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_set_get_attr")
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_alg_init")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_plan_init");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocBlasLt_TestName<aux_test> name(arg.name);

            name << hipblas_datatype_to_string(arg.a_type);

            return std::move(name);
        }
    };

    TEST_P(aux_test, conversion)
    {
        RUN_TEST_ON_THREADS_STREAMS(hipblaslt_matmul_dispatch<aux_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(aux_test);

} // namespace
