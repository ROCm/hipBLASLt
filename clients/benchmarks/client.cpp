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

#include "program_options.hpp"

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_parse_data.hpp"
#include "type_dispatch.hpp"
#include "utility.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "testing_matmul.hpp"

#include "type_dispatch.hpp"
#include "utility.hpp"
#include <algorithm>
#undef I

using namespace roc; // For emulated program_options
using namespace std::literals; // For std::string literals of form "str"s

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(const Arguments&), str_less>;

// Run a function by using map to map arg.function to function
void run_function(const func_map& map, const Arguments& arg, const std::string& msg = "")
{
    auto match = map.find(arg.function);
    if(match == map.end())
        throw std::invalid_argument("Invalid combination --function "s + arg.function
                                    + " --a_type "s + hipblas_datatype_to_string(arg.a_type) + msg);
    match->second(arg);
}

// Template to dispatch testing_matmul for performance tests
// the test is marked invalid when (Ti, To, Tc) not in (H/H/S, B/B/S)
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_matmul : hipblaslt_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_matmul<
    Ti,
    To,
    Tc,
    std::enable_if_t<
#ifdef __HIP_PLATFORM_HCC__
        (std::is_same<Ti, To>{}
         && (std::is_same<Ti, hipblasLtHalf>{} || std::is_same<Ti, hip_bfloat16>{}
             || std::is_same<Ti, float>{})
         && std::is_same<Tc, float>{})
#else
        (std::is_same<Ti, To>{}
         && ((std::is_same<Ti, hipblasLtHalf>{} && std::is_same<Tc, hipblasLtHalf>{})
             || (std::is_same<Ti, hip_bfloat16>{} && std::is_same<Tc, hip_bfloat16>{})))
#endif
        || (std::is_same<Ti, To>{} && (std::is_same<Ti, int8_t>{}) && std::is_same<Tc, int32_t>{})>>
    : hipblaslt_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {{"matmul", testing_matmul<Ti, To, Tc>}};
        run_function(map, arg);
    }
};

int run_bench_test(Arguments& arg, const std::string& filter, bool any_stride, bool yaml = false)
{
    hipblaslt_cout << std::setiosflags(std::ios::fixed)
                   << std::setprecision(7); // Set precision to 7 digits

    // disable unit_check in client benchmark, it is only used in gtest unit test
    arg.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    arg.timing = 1;

    // One stream and one thread (0 indicates to use default behavior)
    arg.streams = 0;
    arg.threads = 0;

    // Skip past any testing_ prefix in function
    static constexpr char prefix[] = "testing_";
    const char*           function = arg.function;
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
        function += sizeof(prefix) - 1;

    if(yaml && strstr(function, "_bad_arg"))
        return 0;
    if(!filter.empty())
    {
        if(!strstr(function, filter.c_str()))
            return 0;
    }

    // adjust dimension for GEMM routines
    int64_t min_lda = arg.transA == 'N' ? arg.M : arg.K;
    int64_t min_ldb = arg.transB == 'N' ? arg.K : arg.N;
    int64_t min_ldc = arg.M;
    int64_t min_ldd = arg.M;
    int64_t min_lde = arg.M;
    if(arg.lda < min_lda)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
        arg.lda = min_lda;
    }
    if(arg.ldb < min_ldb)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
        arg.ldb = min_ldb;
    }
    if(arg.ldc < min_ldc)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
        arg.ldc = min_ldc;
    }
    if(arg.ldd < min_ldd)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
        arg.ldd = min_ldd;
    }
    if(arg.lde < min_lde)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: lde < min_lde, set lde = " << min_lde << std::endl;
        arg.lde = min_lde;
    }
    int64_t min_stride_a = arg.lda * (arg.transA == 'N' ? arg.K : arg.M);
    int64_t min_stride_b = arg.ldb * (arg.transB == 'N' ? arg.N : arg.K);
    int64_t min_stride_c = arg.ldc * arg.N;
    int64_t min_stride_d = arg.ldd * arg.N;
    int64_t min_stride_e = arg.lde * arg.N;
    if(!any_stride && arg.stride_a < min_stride_a)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: stride_a < min_stride_a, set stride_a = "
        //               << min_stride_a << std::endl;
        arg.stride_a = min_stride_a;
    }
    if(!any_stride && arg.stride_b < min_stride_b)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: stride_b < min_stride_b, set stride_b = "
        //               << min_stride_b << std::endl;
        arg.stride_b = min_stride_b;
    }
    if(!any_stride && arg.stride_c < min_stride_c)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: stride_c < min_stride_c, set stride_c = "
        //               << min_stride_c << std::endl;
        arg.stride_c = min_stride_c;
    }
    if(!any_stride && arg.stride_d < min_stride_d)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: stride_d < min_stride_d, set stride_d = "
        //               << min_stride_d << std::endl;
        arg.stride_d = min_stride_d;
    }
    if(!any_stride && arg.stride_e < min_stride_e)
    {
        //hipblaslt_cout << "hipblaslt-bench INFO: stride_e < min_stride_e, set stride_e = "
        //               << min_stride_e << std::endl;
        arg.stride_e = min_stride_e;
    }

    hipblaslt_matmul_dispatch<perf_matmul>(arg);
    return 0;
}

int hipblaslt_bench_datafile(const std::string& filter, bool any_stride)
{
    int ret = 0;
    for(Arguments arg : HipBlasLt_TestData())
        ret |= run_bench_test(arg, filter, any_stride, true);
    test_cleanup::cleanup();
    return ret;
}

// Replace --batch with --batch_count for backward compatibility
void fix_batch(int argc, char* argv[])
{
    static char b_c[] = "--batch_count";
    for(int i = 1; i < argc; ++i)
        if(!strcmp(argv[i], "--batch"))
        {
            static int once
                = (hipblaslt_cerr << argv[0]
                                  << " warning: --batch is deprecated, and --batch_count "
                                     "should be used instead."
                                  << std::endl,
                   0);
            argv[i] = b_c;
        }
}

int main(int argc, char* argv[])
try
{
    fix_batch(argc, argv);
    Arguments   arg;
    std::string function;
    std::string precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;
    std::string scale_type;
    std::string bias_type;
    std::string bias_source;
    std::string initialization;
    std::string filter;
    std::string activation_type;
    int         device_id;
    int         flags             = 0;
    bool        datafile          = hipblaslt_parse_data(argc, argv);
    bool        log_function_name = false;
    bool        any_stride        = false;

    arg.init(); // set all defaults

    options_description desc("hipblaslt-bench command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<int64_t>(&arg.M)->default_value(128),
         "Specific matrix size: the number of rows or columns in matrix.")

        ("sizen,n",
         value<int64_t>(&arg.N)->default_value(128),
         "Specific matrix the number of rows or columns in matrix")

        ("sizek,k",
         value<int64_t>(&arg.K)->default_value(128),
         "Specific matrix size: the number of columns in A and rows in B.")

        ("lda",
         value<int64_t>(&arg.lda),
         "Leading dimension of matrix A.")

        ("ldb",
         value<int64_t>(&arg.ldb),
         "Leading dimension of matrix B.")

        ("ldc",
         value<int64_t>(&arg.ldc),
         "Leading dimension of matrix C.")

        ("ldd",
         value<int64_t>(&arg.ldd),
         "Leading dimension of matrix D.")

        ("lde",
         value<int64_t>(&arg.lde),
         "Leading dimension of matrix E.")

        ("any_stride",
         value<bool>(&any_stride)->default_value(false),
         "Do not modify input strides based on leading dimensions")

        ("stride_a",
         value<int64_t>(&arg.stride_a),
         "Specific stride of strided_batched matrix A, second dimension * leading dimension.")

        ("stride_b",
         value<int64_t>(&arg.stride_b),
         "Specific stride of strided_batched matrix B, second dimension * leading dimension.")

        ("stride_c",
         value<int64_t>(&arg.stride_c),
         "Specific stride of strided_batched matrix C, second dimension * leading dimension.")

        ("stride_d",
         value<int64_t>(&arg.stride_d),
         "Specific stride of strided_batched matrix D, second dimension * leading dimension.")

        ("stride_e",
         value<int64_t>(&arg.stride_e),
         "Specific stride of strided_batched matrix E, second dimension * leading dimension.")

        ("alpha",
          value<float>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<float>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         value<std::string>(&function)->default_value("matmul"), "BLASLt function to test. "
         "Options: matmul")

        ("precision,r",
         value<std::string>(&precision)->default_value("f16_r"), "Precision of matrix A,B,C,D  "
         "Options: f32_r,f16_r,bf16_r")

/*TODO: Enable individual matrix type option once input/output can support different data type.
        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
        "Options: f32_r,f16_r,bf16_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
        "Options: f32_r,f16_r,bf16_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: f32_r,f16_r,bf16_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
        "Options: f32_r,f16_r,bf16_r")
*/
        ("compute_type",
         value<std::string>(&compute_type)->default_value("f32_r"), "Precision of computation. "
         "Options: s,f32_r,x,xf32_r")

        ("scale_type",
         value<std::string>(&scale_type), "Precision of scalar. "
        "Options: f16_r,bf16_r")

        ("initialization",
         value<std::string>(&initialization)->default_value("hpl"),
         "Intialize matrix data."
         "Options: rand_int, trig_float, hpl(floating)")

        ("transA",
         value<char>(&arg.transA)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transB",
         value<char>(&arg.transB)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("batch_count",
         value<int32_t>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched and strided_batched routines")

        ("HMM",
         value<bool>(&arg.HMM)->default_value(false),
         "Parameter requesting the use of HipManagedMemory")

        ("verify,v",
         value<int8_t>(&arg.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<int32_t>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("cold_iters,j",
         value<int32_t>(&arg.cold_iters)->default_value(2),
         "Cold Iterations to run before entering the timing loop")

        ("algo",
         value<uint32_t>(&arg.algo)->default_value(0),
         "Reserved.")

        ("solution_index",
         value<int32_t>(&arg.solution_index)->default_value(0),
         "Reserved.")

        ("activation_type",
         value<std::string>(&activation_type)->default_value("none"),
         "Options: None, gelu, relu")

        ("activation_arg1",
         value<float>(&arg.activation_arg1)->default_value(0),
         "Reserved.")

        ("activation_arg2",
         value<float>(&arg.activation_arg2)->default_value(std::numeric_limits<float>::infinity()),
         "Reserved.")

        ("bias_type",
         value<std::string>(&bias_type), "Precision of bias vector."
        "Options: f16_r,bf16_r,f32_r,default(same with D type)")

        ("bias_source",
         value<std::string>(&bias_source)->default_value("d"),
         "Choose bias source: a, b, d")

        ("bias_vector",
         bool_switch(&arg.bias_vector)->default_value(false),
         "Apply bias vector")

        ("scaleD_vector",
         bool_switch(&arg.scaleD_vector)->default_value(false),
         "Apply scaleD vector")

        ("scaleAlpha_vector",
         bool_switch(&arg.scaleAlpha_vector)->default_value(false),
         "Apply scaleAlpha vector")

        ("use_e",
         bool_switch(&arg.use_e)->default_value(false),
         "Apply AUX output/ gradient input")

        ("gradient",
         bool_switch(&arg.gradient)->default_value(false),
         "Enable gradient")

        ("grouped_gemm",
         value<int32_t>(&arg.grouped_gemm)->default_value(0),
         "Use grouped_gemm if non-zero. Number of gemms to run")

        ("device",
         value<int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("c_noalias_d",
         bool_switch(&arg.c_noalias_d)->default_value(false),
         "C and D are stored in separate memory")

        ("workspace",
         value<size_t>(&arg.user_allocated_workspace)->default_value(0),
         "Set fixed workspace memory size instead of using hipblaslt managed memory")

        ("log_function_name",
         bool_switch(&log_function_name)->default_value(false),
         "Function name precedes other itmes.")

        ("function_filter",
         value<std::string>(&filter),
         "Simple strstr filter on function name only without wildcards")

        ("help,h", "produces this help message")

        ("version", "Prints the version number");
    // clang-format on

    // parse command line into arg structure and stack variables using desc
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if((argc <= 1 && !datafile) || vm.count("help"))
    {
        hipblaslt_cout << desc << std::endl;
        return 0;
    }

    if(vm.find("version") != vm.end())
    {
        int                    version;
        hipblaslt_local_handle handle;
        hipblasLtGetVersion(handle, &version);
        hipblaslt_cout << "hipBLASLt version: " << version << std::endl;
        return 0;
    }

    // transfer local variable state
    ArgumentModel_set_log_function_name(log_function_name);

    // Device Query
    int64_t device_count = query_device_property();

    hipblaslt_cout << std::endl;
    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    set_device(device_id);

    if(datafile)
        return hipblaslt_bench_datafile(filter, any_stride);

    // single bench run

    // validate arguments

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string_to_hipblas_datatype(precision);
    if(prec == static_cast<hipblasDatatype_t>(0))
        throw std::invalid_argument("Invalid value for --precision " + precision);

    arg.a_type = a_type == "" ? prec : string_to_hipblas_datatype(a_type);
    if(arg.a_type == static_cast<hipblasDatatype_t>(0))
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    arg.b_type = b_type == "" ? prec : string_to_hipblas_datatype(b_type);
    if(arg.b_type == static_cast<hipblasDatatype_t>(0))
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    arg.c_type = c_type == "" ? prec : string_to_hipblas_datatype(c_type);
    if(arg.c_type == static_cast<hipblasDatatype_t>(0))
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    arg.d_type = d_type == "" ? prec : string_to_hipblas_datatype(d_type);
    if(arg.d_type == static_cast<hipblasDatatype_t>(0))
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    bool is_f16      = arg.a_type == HIPBLAS_R_16F || arg.a_type == HIPBLAS_R_16B;
    bool is_f32      = arg.a_type == HIPBLAS_R_32F;
    arg.compute_type = compute_type == "" ? (HIPBLASLT_COMPUTE_F32)
                                          : string_to_hipblaslt_computetype(compute_type);
    if(arg.compute_type == static_cast<hipblasLtComputeType_t>(0))
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    if(string_to_hipblas_datatype(bias_type) == static_cast<hipblasDatatype_t>(0) && bias_type != ""
       && bias_type != "default")
        throw std::invalid_argument("Invalid value for --bias_type " + bias_type);
    else
        arg.bias_type = string_to_hipblas_datatype(bias_type);

    arg.initialization = string2hipblaslt_initialization(initialization);
    if(arg.initialization == static_cast<hipblaslt_initialization>(0))
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    arg.activation_type = string_to_hipblaslt_activation_type(activation_type);
    if(arg.activation_type == static_cast<hipblaslt_activation_type>(0))
        throw std::invalid_argument("Invalid value for --activation_type " + activation_type);

    arg.bias_source = string_to_hipblaslt_bias_source(bias_source);

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));

    int copied = snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());
    if(copied <= 0 || copied >= sizeof(arg.function))
        throw std::invalid_argument("Invalid value for --function");

    arg.norm_check_assert = false;
    return run_bench_test(arg, filter, any_stride);
}
catch(const std::invalid_argument& exp)
{
    hipblaslt_cerr << exp.what() << std::endl;
    return -1;
}
