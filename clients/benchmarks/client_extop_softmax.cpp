#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <numeric>
#include <vector>

void printUsage(char* programName)
{
    std::cout << "Usage: " << programName << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\tShow this help message\n"
              << "\t-m, --m\t\t\t\tSize of dim 0, default is 1335\n"
              << "\t-n, --n\t\t\t\tSize of dim 1, default is 16\n";
}

int parseArgs(int argc, char** argv, size_t* m, size_t* n)
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

            if(arg == "-m" || arg == "--m")
            {
                *m = std::stoul(argv[++i]);
            }
            else if(arg == "-n" || arg == "--n")
            {
                *n = std::stoul(argv[++i]);
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

int main(int argc, char** argv)
{
    std::size_t m{1335};
    std::size_t n{16};

    if(auto err = parseArgs(argc, argv, &m, &n))
    {
        printUsage(argv[0]);
        return err;
    }

    std::size_t numElements     = m * n;
    std::size_t elementNumBytes = sizeof(float);
    float*      input{};
    float*      output{};
    auto        hipErr = hipMalloc(&input, numElements * elementNumBytes);
    hipErr             = hipMalloc(&output, numElements * elementNumBytes);
    std::vector<float> data(numElements, 0.f);
    std::iota(begin(data), end(data), 0.f);
    hipErr = hipMemcpyHtoD(input, data.data(), numElements * elementNumBytes);
    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    //warmup
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, output, input, stream);

    if(hipblasltErr)
    {
        std::cerr << "Invalid shape (" << m << ", " << n << "), currently support n <= 256\n";
        hipFree(input);
        hipFree(output);
        hipStreamDestroy(stream);
        return EXIT_FAILURE;
    }

    hipEvent_t beg, end;
    hipErr      = hipEventCreate(&beg);
    hipErr      = hipEventCreate(&end);
    int numRuns = 50;
    hipErr      = hipEventRecord(beg, stream);

    for(int i = 0; i < numRuns; ++i)
    {
        hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, output, input, stream);
    }

    hipErr = hipEventRecord(end, stream);
    hipErr = hipEventSynchronize(end);
    hipErr = hipStreamSynchronize(stream);
    float dur{};
    hipErr = hipEventElapsedTime(&dur, beg, end);
    std::cout << "Time elapsed: " << std::to_string(dur / numRuns) << " ms\n";
    std::vector<float> gpuOutput(numElements, 0.f);
    hipErr = hipEventDestroy(beg);
    hipErr = hipEventDestroy(end);
    hipErr = hipMemcpyDtoH(gpuOutput.data(), output, numElements * elementNumBytes);
    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(input);
    hipErr = hipFree(output);
    return 0;
}
