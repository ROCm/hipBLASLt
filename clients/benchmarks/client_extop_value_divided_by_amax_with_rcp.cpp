/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
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

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt_datatype2string.hpp>
#include <hipblaslt_init.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

void printUsage(char* programName)
{
    std::cout << "Usage: " << programName << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\tShow this help message\n"
              << "\t-t, --type\t\t\tType of problem, default is S.\n"
              << "\t-d, --dtype\t\t\tDest Type of problem, default is S.\n"
              << "\t-m, --m\t\t\t\tSize of dim 0, default is 64\n"
              << "\t-n, --n\t\t\t\tSize of dim 1, default is 64\n"
              << "\t-v, --n\t\t\t\tValue is divided by AMax, default is 240\n"
              << "\t-i, --i\t\t\t\titeration\n"
              << "\t--initialization \t\tInitialize matrix data. Options: rand_int, trig_float, "
                 "hpl(floating), special, zero. (default is hpl)\n";
}

template <typename T>
T abs(T a)
{
    return (double(a) > 0) ? a : -a;
}

template <typename T>
T max(T a, T b)
{
    return (double(a) > double(b)) ? a : b;
}

template <typename Ti, typename To>
void cpuAMax(To* out, To* outRcp, Ti* in, std::uint32_t length, float value)
{
    // calculate amax
    Ti m = 0;
    for(int j = 0; j < length; j++)
    {
        m = max(m, abs(in[j]));
    }
    float tmp = value / float(m);
    out[0] = To(tmp);
    outRcp[0] = To(1.0f/tmp);
}

int parseArgs(int                       argc,
              char**                    argv,
              std::string&              type,
              std::string&              dtype,
              size_t&                   m,
              size_t&                   n,
              float&                    v,
              size_t&                   iter,
              hipblaslt_initialization& init)
{
    if(argc <= 1)
    {
        return EXIT_SUCCESS;
    }

    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-')))
        {
            if((arg == "-h") || (arg == "--help"))
            {
                return EXIT_FAILURE;
            }
            else if(arg == "-t" || arg == "--type")
            {
                type = argv[++i];
            }
            else if(arg == "-d" || arg == "--dtype")
            {
                dtype = argv[++i];
            }
            else if(arg == "-m" || arg == "--m")
            {
                m = std::stoul(argv[++i]);
            }
            else if(arg == "-n" || arg == "--n")
            {
                n = std::stoul(argv[++i]);
            }
            else if(arg == "-v" || arg == "--value")
            {
                v = std::stof(argv[++i]);
            }
            else if(arg == "-i" || arg == "--i")
            {
                iter = std::stoul(argv[++i]);
            }
            else if(arg == "--initialization" || arg == "--init")
            {
                const std::string initStr{argv[++i]};

                if(initStr != "rand_int" && initStr != "trig_float" && initStr != "hpl" && initStr != "special" && initStr != "zero")
                {
                    std::cerr << "Invalid initialization type: " << initStr << '\n';
                    return EXIT_FAILURE;
                }

                init = string2hipblaslt_initialization(initStr);
            }
        }
        else
        {
            std::cerr << "error with " << arg << std::endl;
            std::cerr << "option must start with - or --" << std::endl << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

template <typename Dtype>
void dumpBuffer(const char* title, Dtype* data, int N)
{
    std::cout << "----- " << title << "----- " << std::endl;
    for(int n = 0; n < N; n++)
    {
        std::cout << float(data[n]) << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

template <typename T>
void compare(const char* title, const std::vector<T>& cpuOutput, const std::vector<T>& refOutput)
{
    T maxErr = 0.0;
    for(int i = 0; i < cpuOutput.size(); i++)
    {
        T err  = abs(refOutput[i] - cpuOutput[i]);
        maxErr = max(maxErr, err);
    }

    std::cout << title << " max error : " << float(maxErr) << std::endl;
}

template <typename DType>
void initData(DType* data, std::size_t numElements, hipblaslt_initialization initMethod)
{
    switch(initMethod)
    {
    case hipblaslt_initialization::rand_int:
        hipblaslt_init<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::trig_float:
        hipblaslt_init_cos<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::hpl:
        hipblaslt_init_hpl<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::special:
        hipblaslt_init_alt_impl_big<DType>(data, numElements, 1, 1);
        break;
    case hipblaslt_initialization::zero:
        hipblaslt_init_zero<DType>(data, numElements, 1, 1);
        break;
    default:
        break;
    }
}

template <typename Ti, typename To>
int AmaxTest(hipDataType type, hipDataType dtype, int m, int n, float v, int iter, hipblaslt_initialization& init)
{
    int         numElements = m * n;

    To* gpuOutput{nullptr};
    To* gpuOutputRcp{nullptr};
    Ti* gpuInput{nullptr};
    Ti* gpuWorkSpace{nullptr};
    std::uint32_t* gpuSync{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, sizeof(To));
    hipErr      = hipMalloc(&gpuOutputRcp, sizeof(To));
    hipErr      = hipMalloc(&gpuInput, m * n * sizeof(Ti));

    hipErr = hipMalloc(&gpuWorkSpace, 4096 * sizeof(Ti));
    hipErr = hipMalloc(&gpuSync, sizeof(std::uint32_t));

    std::vector<To>            cpuOutput(1, 0.f);
    std::vector<To>            cpuOutputRcp(1, 0.f);
    std::vector<Ti>            cpuInput(m * n, 0.f);
    std::vector<Ti>            cpuWorkSpace(4096, 0.f);
    std::vector<std::uint32_t> cpuSync(1, 0.f);

    std::vector<To>            refOutput(1, 0.f);
    std::vector<To>            refOutputRcp(1, 0.f);

    initData(cpuInput.data(), numElements, init);

    cpuAMax(refOutput.data(), refOutputRcp.data(), cpuInput.data(), m * n, v);

    hipErr = hipMemset(gpuOutput, 0, sizeof(To));
    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), m * n * sizeof(Ti));

    hipErr = hipMemset(gpuWorkSpace, 0, 4096 * sizeof(Ti));
    hipErr = hipMemset(gpuSync, 0, sizeof(std::uint32_t));

    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    //warmup
    hipblasStatus_t hipblasltErr;
    hipblasltErr = hipblasltExtFastValueDividedByAMaxWithRcp(type, dtype, gpuOutput, gpuOutputRcp, gpuInput, gpuWorkSpace, gpuSync, m, n, v, stream);

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(To));
    hipErr = hipMemcpyDtoH(cpuOutputRcp.data(), gpuOutputRcp, sizeof(To));

    hipErr = hipStreamSynchronize(stream);

    compare("Output", cpuOutput, refOutput);
    compare("OutputRcp", cpuOutputRcp, refOutputRcp);

    // dumpBuffer("Input", cpuInput.data(), m * n);
    // dumpBuffer("GPU", cpuOutput.data(), 1);
    // dumpBuffer("CPU", refOutput.data(), 1);

    // warm up
    for(int i = 0; i < iter; ++i)
    {
        hipblasltErr = hipblasltExtFastValueDividedByAMaxWithRcp(type, dtype, gpuOutput, gpuOutputRcp, gpuInput, gpuWorkSpace, gpuSync, m, n, v, stream);
    }
    hipErr = hipStreamSynchronize(stream);

    hipEvent_t beg, end;
    hipErr      = hipEventCreate(&beg);
    hipErr      = hipEventCreate(&end);
    hipErr      = hipEventRecord(beg, stream);

    for(int i = 0; i < iter; ++i)
    {
        hipblasltErr = hipblasltExtFastValueDividedByAMaxWithRcp(type, dtype, gpuOutput, gpuOutputRcp, gpuInput, gpuWorkSpace, gpuSync, m, n, v, stream);
    }

    hipErr = hipEventRecord(end, stream);
    hipErr = hipEventSynchronize(end);
    hipErr = hipStreamSynchronize(stream);

    float dur{};
    hipErr = hipEventElapsedTime(&dur, beg, end);
    std::cout << "Time elapsed: " << std::to_string(dur * 1000 / iter) << " us\n";

    hipErr = hipEventDestroy(beg);
    hipErr = hipEventDestroy(end);

    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuOutputRcp);
    hipErr = hipFree(gpuInput);
    hipErr = hipFree(gpuWorkSpace);
    hipErr = hipFree(gpuSync);

    return 0;
}

int main(int argc, char** argv)
{
    std::string type{"S"};
    std::string dtype{"S"};
    std::size_t m{64};
    std::size_t n{64};
    float       v{240.f};
    std::size_t i{200};

    hipblaslt_initialization init{hipblaslt_initialization::hpl};

    if(auto err = parseArgs(argc, argv, type, dtype, m, n, v, i, init))
    {
        std::cout << "m " << m << " n " << n << " v " << v << " i " << i << std::endl;
        printUsage(argv[0]);
        return err;
    }

    std::cout << "m " << m << " n " << n << " i " << i << std::endl;

    if((type == "S" || type == "s") && (type == dtype))
        return AmaxTest<float, float>(HIP_R_32F, HIP_R_32F, m, n, v, i, init);
    else if((type == "S" || type == "s") && (dtype == "H" || dtype == "H"))
        return AmaxTest<float, hipblasLtHalf>(HIP_R_32F, HIP_R_16F, m, n, v, i, init);
    else if((type == "H" || type == "h") && (type == dtype))
        return AmaxTest<hipblasLtHalf, hipblasLtHalf>(HIP_R_16F, HIP_R_16F, m, n, v, i, init);
    else if((type == "H" || type == "h") && (dtype == "S" || dtype == "s"))
        return AmaxTest<hipblasLtHalf, float>(HIP_R_16F, HIP_R_32F, m, n, i, v, init);
    else
        std::cout << "Unsupported data type " << type << std::endl;

    return 0;
}
