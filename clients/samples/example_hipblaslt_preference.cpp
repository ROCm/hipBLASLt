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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

// default sizes
#define DIM1 1024
#define DIM2 1024
#define DIM3 1024
#define BATCH_COUNT 1
#define ALPHA 2
#define BETA 3
#define BENCH_LOOP_COUNT 3
#define COLD_LOOP_COUNT 0
#define GPU_TIMER 0

typedef enum class _ActivationType
{
    NONE = 0,
    RELU = 1,
    GELU = 2,
} ActivationType;

typedef enum class _BiasSrc
{
    NONE = 0,
    A    = 1,
    B    = 2,
    D    = 3,
} BiasSrc;

inline const char* ToString(ActivationType act)
{
    switch(act)
    {
    case ActivationType::NONE:
        return "none";
    case ActivationType::RELU:
        return "relu";
    case ActivationType::GELU:
        return "gelu";
    default:
        return "[Unknown Activation Type]";
    }
}

inline const char* ToString(BiasSrc bias)
{
    switch(bias)
    {
    case BiasSrc::NONE:
        return "none";
    case BiasSrc::A:
        return "A";
    case BiasSrc::B:
        return "B";
    case BiasSrc::D:
        return "D";
    default:
        return "[Unknown Bias Type]";
    }
}

auto _relu = [](auto in) -> decltype(in) {
    return static_cast<decltype(in)>(std::max(static_cast<decltype(in)>(0), in));
};

auto _gelu = [](auto in) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.7978845608028654);
    constexpr auto k1    = static_cast<Tc>(0.044715);
    Tc             in_Tc = static_cast<Tc>(in);

    return static_cast<decltype(in)>(
        0.5f * (in_Tc * (1.f + std::tanh(k0 * (in_Tc * (1.f + k1 * (in_Tc * in_Tc)))))));
};

auto _dgelu = [](auto in) -> decltype(in) {
    using Tc = float;

    constexpr auto k0    = static_cast<Tc>(0.0535161);
    constexpr auto k1    = static_cast<Tc>(0.398942);
    constexpr auto k2    = static_cast<Tc>(0.0356774);
    constexpr auto k3    = static_cast<Tc>(0.797885);
    Tc             in_Tc = static_cast<Tc>(in);

    Tc pow3 = in_Tc * in_Tc * in_Tc;
    Tc x1   = k0 * pow3 + k1 * in_Tc;
    Tc xx   = k2 * pow3 + k3 * in_Tc;
    Tc x2   = 4 / pow(exp(-xx) + exp(xx), 2);
    Tc tmp  = 0.5 * tanh(xx) + x1 * x2 + 0.5;
    return static_cast<decltype(in)>(0.5f * tanh(xx) + x1 * x2 + 0.5f);
};

template <typename T>
inline bool AlmostEqual(T a, T b)
{
    T absA = (a > 0) ? a : -a;
    T absB = (b > 0) ? b : -b;
    // this avoids NaN when inf is compared against inf in the alternative code
    // path
    if(static_cast<float>(absA) == std::numeric_limits<float>::infinity()
       || // numeric_limits is yet to
       // support _Float16 type
       // properly;
       static_cast<float>(absB)
           == std::numeric_limits<float>::infinity()) // however promoting it to
    // float works just as fine
    {
        return a == b;
    }
    T absDiff = (a - b > 0) ? a - b : b - a;
    return absDiff / (absA + absB + 1) < 0.001;
}

template <typename T>
void print_strided_batched(
    const char* name, T* A, int64_t n1, int64_t n2, int64_t n3, int64_t s1, int64_t s2, int64_t s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    printf("---------- %s (MxN=%ldx%ld, batch=%ld, stride=%ld"
           ", batch stride=%ld)----------\n",
           name,
           n1,
           n2,
           n3,
           s2,
           s3);
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                printf("[%ld]\t%8.3f\t",
                       (i1 * s1) + (i2 * s2) + (i3 * s3),
                       static_cast<float>(A[(i1 * s1) + (i2 * s2) + (i3 * s3)]));
            }
            printf("\n");
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            printf("\n");
    }
}

template <typename Ti, typename To, typename Tc>
void mat_mul_bias_activation(Tc             alpha,
                             Tc             beta,
                             int            M,
                             int            N,
                             int            K,
                             int            batch_count,
                             const Ti*      A,
                             int            As1,
                             int            As2,
                             int            As3,
                             const Ti*      B,
                             int            Bs1,
                             int            Bs2,
                             int            Bs3,
                             const To*      C,
                             int            Cs1,
                             int            Cs2,
                             int            Cs3,
                             To*            D,
                             int            Ds1,
                             int            Ds2,
                             int            Ds3,
                             Tc*            E,
                             int            Es1,
                             int            Es2,
                             int            Es3,
                             To*            bias,
                             Tc*            scaleD,
                             bool           gradient,
                             ActivationType actType,
                             Tc*            workspace)
{
    std::function<Tc(Tc)> actFunc;
    if(actType == ActivationType::RELU)
        actFunc = _relu;
    else if(actType == ActivationType::GELU && gradient)
        actFunc = _dgelu;
    else if(actType == ActivationType::GELU)
        actFunc = _gelu;

    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int i1 = 0; i1 < M; i1++)
        {
            for(int i2 = 0; i2 < N; i2++)
            {
                Tc t = static_cast<Tc>(0);
                for(int i3 = 0; i3 < K; i3++)
                {
                    t += static_cast<Tc>(A[i1 * As1 + i3 * As2 + batch * As3])
                         * static_cast<Tc>(B[i3 * Bs1 + i2 * Bs2 + batch * Bs3]);
                }
                t = beta * static_cast<Tc>(C[i1 * Cs1 + i2 * Cs2 + batch * Cs3]) + alpha * t
                    + (bias == nullptr ? 0 : bias[i1]);
                if(E != nullptr && !gradient)
                {
                    E[i1 * Es1 + i2 * Es2 + batch * Es3] = static_cast<Tc>(t);
                }
                if(actType != ActivationType::NONE)
                {
                    if(gradient)
                        t = actFunc(static_cast<Tc>(E[i1 * Es1 + i2 * Es2 + batch * Es3])) * t;
                    else
                        t = actFunc(t);
                }
                t = t * (scaleD == nullptr ? 1.0 : scaleD[i1]);
                if(workspace)
                    workspace[i1 + i2 * M + batch * M * N] = static_cast<Tc>(t);
                D[i1 * Ds1 + i2 * Ds2 + batch * Ds3] = static_cast<To>(t);
            }
        }
    }
}

template <typename Tc, typename Ti, typename To>
void reduction(Ti* workspace, To* bias, int length, int k, int s1, int s2, int s3, int batch_count)
{
    assert(batch_count == 1);
    for(int batch = 0; batch < batch_count; batch++)
    {
        for(int i1 = 0; i1 < length; i1++)
        {
            Tc sum = 0;
            for(int i2 = 0; i2 < k; i2++)
            {
                sum += static_cast<Tc>(workspace[i1 * s1 + i2 * s2 + batch * s3]);
            }
            bias[i1] = static_cast<To>(sum);
        }
    }
}

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

// cppcheck-suppress constParameter
static void show_usage(char* argv[])
{
    std::cerr << "Usage: " << argv[0] << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\t\tShow this help message\n"
              << "\t-v, --verbose\t\t\t\tVerbose output\n"
              << "\t-V, --validate\t\t\t\tVerify results\n"
              << "\t-s, --request_solutions\t\t\tNumber of solutions to run (default is 1)\n"
              << "\t-m \t\t\tm\t\tGEMM_STRIDED argument m\n"
              << "\t-n \t\t\tn\t\tGEMM_STRIDED argument n\n"
              << "\t-k \t\t\tk \t\tGEMM_STRIDED argument k\n"
              << "\t--lda \t\t\tlda \t\tGEMM_STRIDED argument lda\n"
              << "\t--ldb \t\t\tldb \t\tGEMM_STRIDED argument ldb\n"
              << "\t--ldc \t\t\tldc \t\tGEMM_STRIDED argument ldc\n"
              << "\t--ldd \t\t\tldd \t\tGEMM_STRIDED argument ldd\n"
              << "\t--trans_a \t\ttrans_a \tGEMM_STRIDED argument trans_a (N, T)\n"
              << "\t--trans_b \t\ttrans_b \tGEMM_STRIDED argument trans_b (N, T)\n"
              << "\t--datatype \t\tdatatype \tGEMM_STRIDED argument in out "
                 "datatype:fp32,fp16,bf16\n"
              << "\t--stride_a \t\tstride_a \tGEMM_STRIDED argument stride_a\n"
              << "\t--stride_b \t\tstride_b \tGEMM_STRIDED argument stride_b\n"
              << "\t--stride_c \t\tstride_c \tGEMM_STRIDED argument stride_c\n"
              << "\t--stride_d \t\tstride_d \tGEMM_STRIDED argument stride_d\n"
              << "\t--alpha \t\talpha \t\tGEMM_STRIDED argument alpha\n"
              << "\t--beta \t\t\tbeta \t\tGEMM_STRIDED argument beta\n"
              << "\t--batch_count \t\tbatch \t\tGEMM_STRIDED argument batch count\n"
              << "\t--act \t\t\tact \t\tGEMM_STRIDED set activation type: relu "
                 "or gelu\n"
              << "\t--grad \t\tgrad \t\tGEMM_STRIDED enable grad: 0 or 1 "
                 "(default is 0)\n"
              << "\t--use_e \t\tuse_e \t\tGEMM_STRIDED enable use_e: 0 or 1 "
                 "(default is 0)\n"
              << "\t--bias \t\t\tbias \t\tGEMM_STRIDED enable bias and choose bias src: A, B, D\n"
              << "\t--scaleD \t\tscaleD \t\tGEMM_STRIDED enable scaleD: 0 or 1 "
                 "(default is 0)\n"
              << "\t--header \t\theader \t\tPrint header for output (default is "
                 "enabled)\n"
              << "\t--timing \t\ttiming \t\tBechmark GPU kernel performance:0 or "
                 "1 (default is 1)\n"
              << "\t--bench_count\t\tbench_count \t\tNumber of benchmark runs (default is 3)\n"
              << "\t--sync_count\t\tsync_count \t\tNumber of sync runs (default is 1)\n"
              << "\t--cold_iters \t\tcold_iters \tCold Iterations to run "
                 "before entering the timing loop (default is 0)\n"
              << "\t--ext \t\t\text \tuse Ext API\n"
              << std::endl;
}

static int parse_arguments(int                 argc,
                           char*               argv[],
                           hipblasDatatype_t&  in_out_datatype,
                           int64_t&            m,
                           int64_t&            n,
                           int64_t&            k,
                           int64_t&            lda,
                           int64_t&            ldb,
                           int64_t&            ldc,
                           int64_t&            ldd,
                           int64_t&            lde,
                           int64_t&            stride_a,
                           int64_t&            stride_b,
                           int64_t&            stride_c,
                           int64_t&            stride_d,
                           int64_t&            stride_e,
                           int32_t&            batch_count,
                           float&              alpha,
                           float&              beta,
                           hipblasOperation_t& trans_a,
                           hipblasOperation_t& trans_b,
                           bool&               enable_grad,
                           bool&               enable_e,
                           BiasSrc&            biasSrc,
                           bool&               enable_scaleD,
                           ActivationType&     actType,
                           bool&               header,
                           bool&               verbose,
                           bool&               validate,
                           bool&               timing,
                           int32_t&            request_solutions,
                           int32_t&            sync_loop_count,
                           int32_t&            bench_loop_count,
                           int32_t&            cold_loop_count,
                           bool&               useExt)
{
    if(argc >= 2)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
            {
                if((arg == "-h") || (arg == "--help"))
                {
                    return EXIT_FAILURE;
                }
                if((arg == "-v") || (arg == "--verbose"))
                {
                    verbose = true;
                }
                else if((arg == "-V") || (arg == "--validate"))
                {
                    validate = true;
                }
                else if((arg == "-s") || arg == "--request_solutions")
                {
                    request_solutions = atoi(argv[++i]);
                    if(request_solutions == 0)
                    {
                        std::cerr << "Error: request_solutions is " << request_solutions
                                  << std::endl;
                        return EXIT_FAILURE;
                    }
                }
                else if(arg == "--header")
                {
                    header = true;
                }
                else if(arg == "--timing")
                {
                    timing = atoi(argv[++i]);
                }
                else if((arg == "-m") && (i + 1 < argc))
                {
                    m = atoi(argv[++i]);
                }
                else if((arg == "-n") && (i + 1 < argc))
                {
                    n = atoi(argv[++i]);
                }
                else if((arg == "-k") && (i + 1 < argc))
                {
                    k = atoi(argv[++i]);
                }
                else if((arg == "--batch_count") && (i + 1 < argc))
                {
                    batch_count = atoi(argv[++i]);
                }
                else if((arg == "--lda") && (i + 1 < argc))
                {
                    lda = atoi(argv[++i]);
                }
                else if((arg == "--ldb") && (i + 1 < argc))
                {
                    ldb = atoi(argv[++i]);
                }
                else if((arg == "--ldc") && (i + 1 < argc))
                {
                    ldc = atoi(argv[++i]);
                }
                else if((arg == "--ldd") && (i + 1 < argc))
                {
                    ldd = atoi(argv[++i]);
                }
                else if((arg == "--lde") && (i + 1 < argc))
                {
                    lde = atoi(argv[++i]);
                }
                else if((arg == "--stride_a") && (i + 1 < argc))
                {
                    stride_a = atoi(argv[++i]);
                }
                else if((arg == "--stride_b") && (i + 1 < argc))
                {
                    stride_b = atoi(argv[++i]);
                }
                else if((arg == "--stride_c") && (i + 1 < argc))
                {
                    stride_c = atoi(argv[++i]);
                }
                else if((arg == "--stride_d") && (i + 1 < argc))
                {
                    stride_d = atoi(argv[++i]);
                }
                else if((arg == "--stride_e") && (i + 1 < argc))
                {
                    stride_e = atoi(argv[++i]);
                }
                else if((arg == "--alpha") && (i + 1 < argc))
                {
                    alpha = atof(argv[++i]);
                }
                else if((arg == "--beta") && (i + 1 < argc))
                {
                    beta = atof(argv[++i]);
                }
                else if((arg == "--grad") && (i + 1 < argc))
                {
                    enable_grad = atoi(argv[++i]);
                }
                else if((arg == "--use_e") && (i + 1 < argc))
                {
                    enable_e = atoi(argv[++i]);
                }
                else if((arg == "--bias") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "A", 1) == 0 || strncmp(argv[i], "a", 1) == 0)
                    {
                        biasSrc = BiasSrc::A;
                    }
                    else if(strncmp(argv[i], "B", 1) == 0 || strncmp(argv[i], "b", 1) == 0)
                    {
                        biasSrc = BiasSrc::B;
                    }
                    else if(strncmp(argv[i], "D", 1) == 0 || strncmp(argv[i], "d", 1) == 0)
                    {
                        biasSrc = BiasSrc::D;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--scaleD") && (i + 1 < argc))
                {
                    enable_scaleD = atoi(argv[++i]);
                }
                else if((arg == "--act") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "relu", 4) == 0)
                        actType = ActivationType::RELU;
                    else if(strncmp(argv[i], "gelu", 4) == 0)
                        actType = ActivationType::GELU;
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--trans_a") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_a = HIPBLAS_OP_N;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_a = HIPBLAS_OP_T;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--trans_b") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "N", 1) == 0 || strncmp(argv[i], "n", 1) == 0)
                    {
                        trans_b = HIPBLAS_OP_N;
                    }
                    else if(strncmp(argv[i], "T", 1) == 0 || strncmp(argv[i], "t", 1) == 0)
                    {
                        trans_b = HIPBLAS_OP_T;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--datatype") && (i + 1 < argc))
                {
                    ++i;
                    if(strncmp(argv[i], "fp32", 4) == 0)
                    {
                        in_out_datatype = HIPBLAS_R_32F;
                    }
                    else if(strncmp(argv[i], "fp16", 4) == 0)
                    {
                        in_out_datatype = HIPBLAS_R_16F;
                    }
                    else if(strncmp(argv[i], "bf16", 4) == 0)
                    {
                        in_out_datatype = HIPBLAS_R_16B;
                    }
                    else
                    {
                        std::cerr << "error with " << arg << std::endl;
                        std::cerr << "do not recognize value " << argv[i];
                        return EXIT_FAILURE;
                    }
                }
                else if((arg == "--sync_count") && (i + 1 < argc))
                {
                    sync_loop_count = atoi(argv[++i]);
                }
                else if((arg == "--bench_count") && (i + 1 < argc))
                {
                    bench_loop_count = atoi(argv[++i]);
                }
                else if((arg == "--cold_iters") && (i + 1 < argc))
                {
                    cold_loop_count = atoi(argv[++i]);
                }
                else if((arg == "--ext") && (i + 1 < argc))
                {
                    useExt = atoi(argv[++i]);
                }
                else
                {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return EXIT_FAILURE;
                }
            }
            else
            {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

bool bad_argument(hipblasOperation_t trans_a,
                  hipblasOperation_t trans_b,
                  int64_t            m,
                  int64_t            n,
                  int64_t            k,
                  int64_t            lda,
                  int64_t            ldb,
                  int64_t            ldc,
                  int64_t            ldd,
                  int64_t            lde,
                  int64_t            stride_a,
                  int64_t            stride_b,
                  int64_t            stride_c,
                  int64_t            stride_d,
                  int64_t            stride_e,
                  int32_t            batch_count)
{
    bool argument_error = false;
    if((trans_a == HIPBLAS_OP_N) && (lda < m))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << m << std::endl;
    }
    if((trans_a == HIPBLAS_OP_T) && (lda < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lda = " << lda << " < " << k << std::endl;
    }
    if((trans_b == HIPBLAS_OP_N) && (ldb < k))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << k << std::endl;
    }
    if((trans_b == HIPBLAS_OP_T) && (ldb < n))
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldb = " << ldb << " < " << n << std::endl;
    }
    if(stride_a < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_a < 0" << std::endl;
    }
    if(stride_b < 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_b < 0" << std::endl;
    }
    if(ldc < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldc = " << ldc << " < " << m << std::endl;
    }
    if(stride_c < n * ldc)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_c = " << stride_c << " < " << n * ldc << std::endl;
    }
    if(ldd < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument ldc = " << ldd << " < " << m << std::endl;
    }
    if(stride_d < n * ldd)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_c = " << stride_d << " < " << n * ldd << std::endl;
    }
    if(lde < m)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument lde = " << lde << " < " << m << std::endl;
    }
    if(stride_e < n * lde)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument stride_e = " << stride_e << " < " << n * lde << std::endl;
    }
    if(batch_count == 0)
    {
        argument_error = true;
        std::cerr << "ERROR: bad argument batch_count = " << batch_count << std::endl;
    }

    return argument_error;
}

bool epilogue_bad_argument(
    bool enable_grad, bool enable_e, BiasSrc biasSrc, bool enable_scaleD, ActivationType actType)
{
    bool argument_error = false;
    if(biasSrc == BiasSrc::A || biasSrc == BiasSrc::B)
    {
        if(!enable_grad)
        {
            argument_error = true;
            std::cerr << "ERROR: Must enable gradient when bias = A or B" << std::endl;
        }
        if(actType != ActivationType::NONE)
        {
            argument_error = true;
            std::cerr
                << "ERROR: Activation type has to be none when bias = A or B. Bad argument act = "
                << ToString(actType) << std::endl;
        }
    }
    else if(biasSrc == BiasSrc::D && enable_grad)
    {
        if(!enable_e)
        {
            argument_error = true;
            std::cerr << "ERROR: Must enable e when gradient = true and bias = D" << std::endl;
        }
    }

    return argument_error;
}

template <typename T>
void initialize_a_b_c_e_bias(std::vector<T>&     ha,
                             int64_t             size_a,
                             std::vector<T>&     hb,
                             int64_t             size_b,
                             std::vector<T>&     hc,
                             int64_t             size_c,
                             std::vector<float>& he,
                             int64_t             size_e,
                             std::vector<T>&     h_bias,
                             int64_t             size_bias,
                             std::vector<float>& h_scaleD,
                             int64_t             size_scaleD)
{
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = static_cast<T>((rand() % 7) - 3);
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = static_cast<T>((rand() % 7) - 3);
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = static_cast<T>((rand() % 7) - 3);
    }
    for(int i = 0; i < size_e; ++i)
    {
        he[i] = static_cast<float>((rand() % 7) - 3);
    }
    for(int i = 0; i < size_bias; ++i)
    {
        h_bias[i] = static_cast<T>((rand() % 7) - 3);
    }
    for(int i = 0; i < size_scaleD; ++i)
    {
        h_scaleD[i] = static_cast<float>((rand() % 7) - 3);
    }
}

template <typename T>
void test_hipblaslt(hipblasDatatype_t  in_out_datatype,
                    hipblasOperation_t trans_a,
                    hipblasOperation_t trans_b,
                    int64_t            m,
                    int64_t            n,
                    int64_t            k,
                    int64_t            lda,
                    int64_t            ldb,
                    int64_t            ldc,
                    int64_t            ldd,
                    int64_t            lde,
                    int64_t            stride_a,
                    int64_t            stride_b,
                    int64_t            stride_c,
                    int64_t            stride_d,
                    int64_t            stride_e,
                    int32_t            batch_count,
                    float              alpha,
                    float              beta,
                    bool               enable_grad,
                    bool               enable_e,
                    BiasSrc            biasSrc,
                    bool               enable_scaleD,
                    ActivationType     actType,
                    bool               validate,
                    bool               verbose,
                    bool               timing,
                    int32_t            request_solutions,
                    int32_t            sync_loop_count,
                    int32_t            bench_loop_count,
                    int32_t            cold_loop_count,
                    bool               useExt)
{
    int64_t     a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    int64_t     row_a, col_a, row_b, col_b, row_c, col_c;
    int         size_a1, size_b1, size_c1 = ldc * n, size_d1 = ldd * n, size_e1 = lde * n;
    std::string trans_string;
    if(trans_a == HIPBLAS_OP_N)
    {
        trans_string += "N";
        row_a      = m;
        col_a      = k;
        a_stride_1 = 1;
        a_stride_2 = lda;
        size_a1    = lda * k;
    }
    else
    {
        trans_string += "T";
        row_a      = k;
        col_a      = m;
        a_stride_1 = lda;
        a_stride_2 = 1;
        size_a1    = lda * m;
    }
    if(trans_b == HIPBLAS_OP_N)
    {
        trans_string += "N, ";
        row_b      = k;
        col_b      = n;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        size_b1    = ldb * n;
    }
    else
    {
        trans_string += "T, ";
        row_b      = n;
        col_b      = k;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        size_b1    = ldb * k;
    }
    row_c = m;
    col_c = n;

    int size_a      = batch_count == 0 ? size_a1 : size_a1 + stride_a * (batch_count - 1);
    int size_b      = batch_count == 0 ? size_b1 : size_b1 + stride_b * (batch_count - 1);
    int size_c      = batch_count == 0 ? size_c1 : size_c1 + stride_c * (batch_count - 1);
    int size_d      = batch_count == 0 ? size_d1 : size_d1 + stride_d * (batch_count - 1);
    int size_e      = batch_count == 0 ? size_e1 : size_e1 + stride_e * (batch_count - 1);
    int size_scaleD = enable_scaleD ? m : 0;
    int size_bias   = 0;
    switch(biasSrc)
    {
    case BiasSrc::A:
    case BiasSrc::D:
        size_bias = m;
        break;
    case BiasSrc::B:
        size_bias = n;
        break;
    default:
        break;
    }

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<T>     ha(size_a);
    std::vector<T>     hb(size_b);
    std::vector<T>     hc(size_c);
    std::vector<T>     hd(size_c);
    std::vector<T>     hd_gold(size_d);
    std::vector<float> he(size_e);
    std::vector<float> he_gold(size_e);
    std::vector<T>     h_bias(size_bias);
    std::vector<T>     h_bias_gold(size_bias);
    std::vector<float> h_scaleD(size_scaleD);

    // initial data on host
    initialize_a_b_c_e_bias(
        ha, size_a, hb, size_b, hc, size_c, he, size_e, h_bias, size_bias, h_scaleD, size_scaleD);

    // allocate memory on device
    void *      da, *db, *dc, *dd, *de, *d_bias, *d_scaleD;
    int         num_streams = 1;
    hipStream_t stream      = nullptr;

    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dd, size_d * sizeof(T)));
    if(enable_e)
        CHECK_HIP_ERROR(hipMalloc(&de, size_e * sizeof(float)));
    if(biasSrc != BiasSrc::NONE)
        CHECK_HIP_ERROR(hipMalloc(&d_bias, size_bias * sizeof(T)));
    if(enable_scaleD)
        CHECK_HIP_ERROR(hipMalloc(&d_scaleD, size_scaleD * sizeof(float)));
    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(T) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(T) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(T) * size_c, hipMemcpyHostToDevice));
    if(enable_grad && enable_e)
        CHECK_HIP_ERROR(hipMemcpy(de, he.data(), sizeof(float) * size_e, hipMemcpyHostToDevice));
    if(!enable_grad && biasSrc != BiasSrc::NONE)
        CHECK_HIP_ERROR(
            hipMemcpy(d_bias, h_bias.data(), sizeof(T) * size_bias, hipMemcpyHostToDevice));
    if(enable_scaleD)
        CHECK_HIP_ERROR(hipMemcpy(
            d_scaleD, h_scaleD.data(), sizeof(float) * size_scaleD, hipMemcpyHostToDevice));

    hipblasLtHandle_t           handle;
    hipblasLtMatrixLayout_t     matA, matB, matC, matD;
    hipblasLtMatmulDesc_t       matmul;
    hipblasLtMatmulPreference_t pref;
    uint64_t                    max_workspace_size = 32 * 1024 * 1024;
    void*                       d_workspace        = nullptr;

    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, in_out_datatype, row_a, col_a, lda));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, in_out_datatype, row_b, col_b, ldb));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, in_out_datatype, row_c, col_c, ldc));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, in_out_datatype, row_c, col_c, ldd));
    if(batch_count > 1)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_d, sizeof(stride_d)));
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLASLT_COMPUTE_F32, HIPBLAS_R_32F));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue;
    if(enable_grad)
    {
        if(enable_e && biasSrc == BiasSrc::NONE && actType == ActivationType::GELU)
            epilogue = HIPBLASLT_EPILOGUE_DGELU;
        else if(!enable_e && biasSrc == BiasSrc::A && actType == ActivationType::NONE)
            epilogue = HIPBLASLT_EPILOGUE_BGRADA;
        else if(!enable_e && biasSrc == BiasSrc::B && actType == ActivationType::NONE)
            epilogue = HIPBLASLT_EPILOGUE_BGRADB;
        else if(enable_e && biasSrc == BiasSrc::D && actType == ActivationType::GELU)
            epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
    }
    else
    {
        if(!enable_e && biasSrc == BiasSrc::D && actType == ActivationType::NONE)
            epilogue = HIPBLASLT_EPILOGUE_BIAS;
        else if(!enable_e && biasSrc == BiasSrc::D && actType == ActivationType::RELU)
            epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
        else if(!enable_e && biasSrc == BiasSrc::D && actType == ActivationType::GELU)
            epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
        else if(!enable_e && biasSrc == BiasSrc::NONE && actType == ActivationType::NONE)
            epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
        else if(!enable_e && biasSrc == BiasSrc::NONE && actType == ActivationType::RELU)
            epilogue = HIPBLASLT_EPILOGUE_RELU;
        else if(!enable_e && biasSrc == BiasSrc::NONE && actType == ActivationType::GELU)
            epilogue = HIPBLASLT_EPILOGUE_GELU;
        else if(enable_e && biasSrc == BiasSrc::D && actType == ActivationType::GELU)
            epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
        else if(enable_e && actType == ActivationType::GELU)
            epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;
    }
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    if(enable_e)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &de, sizeof(void*)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &lde, sizeof(int64_t)));
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, &stride_e, sizeof(int64_t)));
    }
    if(biasSrc != BiasSrc::NONE)
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(void*)));
    if(enable_scaleD)
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scaleD, sizeof(void*)));

    // Set User Preference attributes
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_HIPBLASLT_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    hipblasLtExtGemm_t handleExt;

    // Get Heuristic results
    hipblasLtMatmulHeuristicResult_t* heuristicResult
        = new hipblasLtMatmulHeuristicResult_t[request_solutions]{0};
    int returnedAlgoCount = 0;
    if(!useExt)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                              matmul,
                                                              matA,
                                                              matB,
                                                              matC,
                                                              matD,
                                                              pref,
                                                              request_solutions,
                                                              heuristicResult,
                                                              &returnedAlgoCount));
    }
    else
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtExtGemmCreate(
            &handleExt, handle, matmul, &alpha, da, matA, db, matB, &beta, dc, matC, dd, matD));

        CHECK_HIPBLASLT_ERROR(hipblasLtExtAlgoGetHeuristic(
            handleExt, pref, request_solutions, heuristicResult, &returnedAlgoCount));
    }
    if(returnedAlgoCount != request_solutions && returnedAlgoCount >= 1)
        std::cout << "Less Solution found! request: " << request_solutions
                  << ", found: " << returnedAlgoCount << std::endl;
    if(returnedAlgoCount == 0)
    {
        std::cerr << "No Solution found! request: " << request_solutions << std::endl;
        return;
    }
    uint64_t workspace_size = 0;
    for(int sol = 0; sol < returnedAlgoCount; sol++)
        workspace_size = max(workspace_size, heuristicResult[sol].workspaceSize);
    if(workspace_size > 0)
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));

    // Solve problem  // call gen function
    if(!useExt)
    {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                              matmul,
                                              &alpha,
                                              da,
                                              matA,
                                              db,
                                              matB,
                                              &beta,
                                              dc,
                                              matC,
                                              dd,
                                              matD,
                                              &heuristicResult[0].algo,
                                              d_workspace,
                                              workspace_size,
                                              stream));
    }
    else
    {
        //grouped gemm
        CHECK_HIPBLASLT_ERROR(
            hipblasLtExtMakeArgument(handleExt, &heuristicResult[0].algo, d_workspace, stream));

        CHECK_HIPBLASLT_ERROR(hipblasLtExtRun(handleExt, stream));
    }

    hipStreamSynchronize(stream);
    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hd.data(), dd, sizeof(T) * size_c, hipMemcpyDeviceToHost));
    if(enable_grad && biasSrc != BiasSrc::NONE)
        CHECK_HIP_ERROR(
            hipMemcpy(h_bias.data(), d_bias, sizeof(T) * size_bias, hipMemcpyDeviceToHost));
    if(!enable_grad && enable_e)
        CHECK_HIP_ERROR(hipMemcpy(he.data(), de, sizeof(float) * size_e, hipMemcpyDeviceToHost));

    double   bestMs = std::numeric_limits<double>::max();
    double   bestTflops;
    uint32_t bestIdx = 0;
    for(int sol = 0; sol < returnedAlgoCount; sol++)
    {
        std::string timing_string;
        if(timing)
        {
            hipEvent_t start, stop;
            hipEventCreate(&start);
            hipEventCreate(&stop);
            double eventMs = 0;

            if(!useExt)
            {
                for(int loop = 0; loop < cold_loop_count; loop++)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                                          matmul,
                                                          &alpha,
                                                          da,
                                                          matA,
                                                          db,
                                                          matB,
                                                          &beta,
                                                          dc,
                                                          matC,
                                                          dd,
                                                          matD,
                                                          &heuristicResult[sol].algo,
                                                          d_workspace,
                                                          workspace_size,
                                                          stream));
                }

                if(GPU_TIMER)
                    hipEventRecord(start, stream);
                else
                    eventMs = get_time_us_sync(stream); // in microseconds

                for(int sync_loop = 0; sync_loop < sync_loop_count; sync_loop++)
                {
                    for(int loop = 0; loop < bench_loop_count; loop++)
                    {
                        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle,
                                                              matmul,
                                                              &alpha,
                                                              da,
                                                              matA,
                                                              db,
                                                              matB,
                                                              &beta,
                                                              dc,
                                                              matC,
                                                              dd,
                                                              matD,
                                                              &heuristicResult[sol].algo,
                                                              d_workspace,
                                                              workspace_size,
                                                              stream));
                    }
                    hipDeviceSynchronize();
                }

                if(GPU_TIMER)
                {
                    hipEventRecord(stop, stream);
                    hipEventSynchronize(stop);
                    float temp;
                    hipEventElapsedTime(&temp, start, stop);
                    eventMs = double(temp);
                    hipEventDestroy(start);
                    hipEventDestroy(stop);
                    eventMs /= (bench_loop_count * sync_loop_count);
                    eventMs = eventMs * 1000;
                }
                else
                {
                    eventMs = get_time_us_sync(stream) - eventMs;
                    eventMs /= (bench_loop_count * sync_loop_count);
                    eventMs = eventMs / 1000;
                }
            }
            else
            {
                for(int loop = 0; loop < cold_loop_count; loop++)
                {
                    CHECK_HIPBLASLT_ERROR(hipblasLtExtRun(handleExt, stream));
                }

                if(GPU_TIMER)
                    hipEventRecord(start, stream);
                else
                    eventMs = get_time_us_sync(stream); // in microseconds

                for(int sync_loop = 0; sync_loop < sync_loop_count; sync_loop++)
                {
                    for(int loop = 0; loop < bench_loop_count; loop++)
                    {
                        CHECK_HIPBLASLT_ERROR(hipblasLtExtRun(handleExt, stream));
                    }
                    hipDeviceSynchronize();
                }

                if(GPU_TIMER)
                {
                    hipEventRecord(stop, stream);
                    hipEventSynchronize(stop);
                    float temp;
                    hipEventElapsedTime(&temp, start, stop);
                    eventMs = double(temp);
                    hipEventDestroy(start);
                    hipEventDestroy(stop);
                    eventMs /= (bench_loop_count * sync_loop_count);
                    eventMs = eventMs * 1000;
                }
                else
                {
                    eventMs = get_time_us_sync(stream) - eventMs;
                    eventMs /= (bench_loop_count * sync_loop_count);
                    eventMs = eventMs / 1000;
                }
            }
            bestMs        = std::min(bestMs, eventMs);
            double flops  = 2 * m * n * k * batch_count;
            double tflops = flops / eventMs / 1000000000;

            if(bestMs == eventMs)
            {
                bestTflops = tflops;
                bestIdx    = sol;
            }

            if(returnedAlgoCount > 1)
            {
                std::cout << std::setprecision(3) << "  Solution" << std::setw(3) << sol << ":"
                          << std::setw(10) << eventMs * 1000 << " us, " << std::setw(7) << tflops
                          << " Tflops";

                if(bestMs == eventMs)
                {
                    std::cout << " *" << std::endl;
                }
                else
                    std::cout << std::endl;
            }
        }
    }

    std::cout << trans_string << m << ", " << n << ", " << k << ", " << lda << ", " << ldb << ", "
              << ldc << ", " << stride_a << ", " << stride_b << ", " << stride_c << ", "
              << batch_count << ", " << alpha << ", " << beta << ", " << enable_e << ", "
              << ToString(biasSrc) << ", " << enable_scaleD << ", " << ToString(actType);
    if(timing)
    {
        std::cout << ", " << bestMs * 1000 << ", " << bestTflops;
    }
    if(request_solutions > 1)
        std::cout << ", solution " << bestIdx;

    std::cout << std::endl;

    // calculate golden or correct result
    if(validate)
    {
        auto*  a_ptr = &ha[0];
        auto*  b_ptr = &hb[0];
        auto*  c_ptr = &hc[0];
        auto*  d_ptr = &hd_gold[0];
        T*     bias_ptr;
        float* e_ptr;
        if(!enable_grad && enable_e)
            e_ptr = &he_gold[0];
        else if(enable_grad && enable_e)
            e_ptr = &he[0];
        else
            e_ptr = nullptr;
        if(biasSrc != BiasSrc::NONE && !enable_grad)
            bias_ptr = &h_bias[0];
        else
            bias_ptr = nullptr;
        float* scaleD_ptr;
        if(enable_scaleD)
            scaleD_ptr = &h_scaleD[0];
        else
            scaleD_ptr = nullptr;
        void* workspace = nullptr;
        if(enable_grad && biasSrc == BiasSrc::D)
            workspace = (void*)malloc(workspace_size);
        mat_mul_bias_activation<T, T, float>(alpha,
                                             beta,
                                             m,
                                             n,
                                             k,
                                             batch_count,
                                             a_ptr,
                                             a_stride_1,
                                             a_stride_2,
                                             stride_a,
                                             b_ptr,
                                             b_stride_1,
                                             b_stride_2,
                                             stride_b,
                                             c_ptr,
                                             1,
                                             ldc,
                                             stride_c,
                                             d_ptr,
                                             1,
                                             ldd,
                                             stride_d,
                                             e_ptr,
                                             1,
                                             lde,
                                             stride_e,
                                             bias_ptr,
                                             scaleD_ptr,
                                             enable_grad,
                                             actType,
                                             (float*)workspace);
        if(enable_grad && biasSrc != BiasSrc::NONE)
        {
            bias_ptr = &h_bias_gold[0];
            // D case
            if(biasSrc == BiasSrc::D)
            {
                assert(m == size_bias);
                reduction<float>((float*)workspace, bias_ptr, m, n, 1, m, (m * n), batch_count);
            }
            else if(biasSrc == BiasSrc::A)
            {
                assert(m == size_bias);
                reduction<float>(
                    a_ptr, bias_ptr, m, k, a_stride_1, a_stride_2, stride_a, batch_count);
            }
            else if(biasSrc == BiasSrc::B)
            {
                assert(n == size_bias);
                reduction<float>(
                    b_ptr, bias_ptr, n, k, b_stride_1, b_stride_2, stride_b, batch_count);
            }
        }

        if(workspace)
            free(workspace);

        bool passed = true;
        for(int i = 0; i < size_c; i++)
        {
            if(!AlmostEqual(hd_gold[i], hd[i]))
            {
                printf("[D] Err: Index %d: %f vs %f\n",
                       i,
                       static_cast<float>(hd_gold[i]),
                       static_cast<float>(hd[i]));
                passed = false;
            }
        }
        if(!enable_grad && enable_e)
        {
            for(int i = 0; i < size_e; i++)
            {
                if(!AlmostEqual(he_gold[i], he[i]))
                {
                    printf("[E] Err: Index %d: %f vs %f\n",
                           i,
                           static_cast<float>(he_gold[i]),
                           static_cast<float>(he[i]));
                    passed = false;
                }
            }
        }
        if(enable_grad && biasSrc != BiasSrc::NONE)
        {
            for(int i = 0; i < size_bias; i++)
            {
                if(!AlmostEqual(h_bias_gold[i], h_bias[i]))
                {
                    printf("[B] Err: Index %d: %f vs %f\n",
                           i,
                           static_cast<float>(h_bias_gold[i]),
                           static_cast<float>(h_bias[i]));
                    passed = false;
                }
            }
        }

        if(!passed)
        {
            std::cout << "FAIL";
        }
        else
        {
            std::cout << "PASS";
        }
        if(returnedAlgoCount > 1)
            std::cout << " (Only validate first solution)";
        std::cout << std::endl;
    }

    if(verbose)
    {
        printf("\n");
        if(trans_a == HIPBLAS_OP_N)
        {
            print_strided_batched("ha initial", &ha[0], m, k, batch_count, 1, lda, stride_a);
        }
        else
        {
            print_strided_batched("ha initial", &ha[0], m, k, batch_count, lda, 1, stride_a);
        }
        if(trans_b == HIPBLAS_OP_N)
        {
            print_strided_batched("hb initial", &hb[0], k, n, batch_count, 1, ldb, stride_b);
        }
        else
        {
            print_strided_batched("hb initial", &hb[0], k, n, batch_count, ldb, 1, stride_b);
        }
        print_strided_batched("hc initial", &hc[0], m, n, batch_count, 1, ldc, stride_c);
        if(enable_e)
        {
            if(enable_grad)
                print_strided_batched("he initial", &he[0], m, n, batch_count, 1, lde, stride_e);
            else
            {
                print_strided_batched("he_gold", &he_gold[0], m, n, batch_count, 1, lde, stride_e);
                print_strided_batched("he device", &he[0], m, n, batch_count, 1, lde, stride_e);
            }
        }
        if(biasSrc != BiasSrc::NONE)
        {
            if(enable_grad)
                print_strided_batched("h_bias gold", &h_bias_gold[0], m, 1, 1, 1, m, 0);
            print_strided_batched("h_bias", &h_bias[0], m, 1, 1, 1, m, 0);
        }
        if(enable_scaleD)
            print_strided_batched("h_scaleD", &h_scaleD[0], m, 1, 1, 1, m, 0);
        print_strided_batched("hd_gold", &hd_gold[0], m, n, batch_count, 1, ldc, stride_c);
        print_strided_batched("hd device", &hd[0], m, n, batch_count, 1, ldc, stride_c);
    }

    delete[] heuristicResult;
    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIP_ERROR(hipFree(dd));
    CHECK_HIP_ERROR(hipFree(d_workspace));
    if(enable_e)
        CHECK_HIP_ERROR(hipFree(de));
    if(biasSrc != BiasSrc::NONE)
        CHECK_HIP_ERROR(hipFree(d_bias));
    if(enable_scaleD)
        CHECK_HIP_ERROR(hipFree(d_scaleD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));

    return;
}

int main(int argc, char* argv[])
{
    int deviceId = 0;
    CHECK_HIP_ERROR(hipSetDevice(deviceId));
    // initialize parameters with default values
    hipblasOperation_t trans_a         = HIPBLAS_OP_N;
    hipblasOperation_t trans_b         = HIPBLAS_OP_N;
    hipblasDatatype_t  in_out_datatype = HIPBLAS_R_32F;

    int64_t invalid_int   = std::numeric_limits<int64_t>::min() + 1;
    float   invalid_float = std::numeric_limits<float>::quiet_NaN();

    // initialize to invalid value to detect if values not specified on command
    // line
    int64_t m = invalid_int, lda = invalid_int, stride_a = invalid_int;
    int64_t n = invalid_int, ldb = invalid_int, stride_b = invalid_int;
    int64_t k = invalid_int, ldc = invalid_int, stride_c = invalid_int;
    int64_t ldd = invalid_int, stride_d = invalid_int;
    int64_t lde = invalid_int, stride_e = invalid_int;

    int32_t batch_count       = BATCH_COUNT;
    int32_t sync_loop_count   = 1;
    int32_t bench_loop_count  = BENCH_LOOP_COUNT;
    int32_t cold_loop_count   = COLD_LOOP_COUNT;
    int32_t request_solutions = 1;

    bool useExt = false;

    float          alpha         = ALPHA;
    float          beta          = BETA;
    bool           enable_grad   = false;
    bool           enable_e      = false;
    BiasSrc        biasSrc       = BiasSrc::NONE;
    bool           enable_scaleD = false;
    ActivationType actType       = ActivationType::NONE;

    bool verbose  = false;
    bool header   = true;
    bool validate = false;
    bool timing   = true;

    if(parse_arguments(argc,
                       argv,
                       in_out_datatype,
                       m,
                       n,
                       k,
                       lda,
                       ldb,
                       ldc,
                       ldd,
                       lde,
                       stride_a,
                       stride_b,
                       stride_c,
                       stride_d,
                       stride_e,
                       batch_count,
                       alpha,
                       beta,
                       trans_a,
                       trans_b,
                       enable_grad,
                       enable_e,
                       biasSrc,
                       enable_scaleD,
                       actType,
                       header,
                       verbose,
                       validate,
                       timing,
                       request_solutions,
                       sync_loop_count,
                       bench_loop_count,
                       cold_loop_count,
                       useExt))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    // when arguments not specified, set to default values
    if(m == invalid_int)
        m = DIM1;
    if(n == invalid_int)
        n = DIM2;
    if(k == invalid_int)
        k = DIM3;
    if(lda == invalid_int)
        lda = trans_a == HIPBLAS_OP_N ? m : k;
    if(ldb == invalid_int)
        ldb = trans_b == HIPBLAS_OP_N ? k : n;
    if(ldc == invalid_int)
        ldc = m;
    if(ldd == invalid_int)
        ldd = m;
    if(lde == invalid_int)
        lde = m;
    if(stride_a == invalid_int)
        stride_a = trans_a == HIPBLAS_OP_N ? lda * k : lda * m;
    if(stride_b == invalid_int)
        stride_b = trans_b == HIPBLAS_OP_N ? ldb * n : ldb * k;
    if(stride_c == invalid_int)
        stride_c = ldc * n;
    if(stride_d == invalid_int)
        stride_d = ldd * n;
    if(stride_e == invalid_int)
        stride_e = lde * n;
    if(alpha != alpha)
        alpha = ALPHA; // check for alpha == invalid_float == NaN
    if(beta != beta)
        beta = BETA; // check for beta == invalid_float == NaN
    if(batch_count == invalid_int)
        batch_count = BATCH_COUNT;

    if(bad_argument(trans_a,
                    trans_b,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                    ldd,
                    lde,
                    stride_a,
                    stride_b,
                    stride_c,
                    stride_d,
                    stride_e,
                    batch_count))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    if(epilogue_bad_argument(enable_grad, enable_e, biasSrc, enable_scaleD, actType))
    {
        show_usage(argv);
        return EXIT_FAILURE;
    }

    if(header)
    {
        std::cout << "transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, "
                     "stride_c, batch_count, alpha, beta, use_e, bias, scaleD, activationType";
        if(timing)
            std::cout << ", us, tflops";
        if(request_solutions > 1)
            std::cout << ", best solution";
        std::cout << std::endl;
    }

    if(in_out_datatype == HIPBLAS_R_32F)
        test_hipblaslt<hipblasLtFloat>(in_out_datatype,
                                       trans_a,
                                       trans_b,
                                       m,
                                       n,
                                       k,
                                       lda,
                                       ldb,
                                       ldc,
                                       ldd,
                                       lde,
                                       stride_a,
                                       stride_b,
                                       stride_c,
                                       stride_d,
                                       stride_e,
                                       batch_count,
                                       alpha,
                                       beta,
                                       enable_grad,
                                       enable_e,
                                       biasSrc,
                                       enable_scaleD,
                                       actType,
                                       validate,
                                       verbose,
                                       timing,
                                       request_solutions,
                                       sync_loop_count,
                                       bench_loop_count,
                                       cold_loop_count,
                                       useExt);
    else if(in_out_datatype == HIPBLAS_R_16F)
        test_hipblaslt<hipblasLtHalf>(in_out_datatype,
                                      trans_a,
                                      trans_b,
                                      m,
                                      n,
                                      k,
                                      lda,
                                      ldb,
                                      ldc,
                                      ldd,
                                      lde,
                                      stride_a,
                                      stride_b,
                                      stride_c,
                                      stride_d,
                                      stride_e,
                                      batch_count,
                                      alpha,
                                      beta,
                                      enable_grad,
                                      enable_e,
                                      biasSrc,
                                      enable_scaleD,
                                      actType,
                                      validate,
                                      verbose,
                                      timing,
                                      request_solutions,
                                      sync_loop_count,
                                      bench_loop_count,
                                      cold_loop_count,
                                      useExt);
    else if(in_out_datatype == HIPBLAS_R_16B)
        test_hipblaslt<hipblasLtBfloat16>(in_out_datatype,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          lda,
                                          ldb,
                                          ldc,
                                          ldd,
                                          lde,
                                          stride_a,
                                          stride_b,
                                          stride_c,
                                          stride_d,
                                          stride_e,
                                          batch_count,
                                          alpha,
                                          beta,
                                          enable_grad,
                                          enable_e,
                                          biasSrc,
                                          enable_scaleD,
                                          actType,
                                          validate,
                                          verbose,
                                          timing,
                                          request_solutions,
                                          sync_loop_count,
                                          bench_loop_count,
                                          cold_loop_count,
                                          useExt);

    return EXIT_SUCCESS;
}
