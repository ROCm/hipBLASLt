#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt_init.hpp>
#include <limits>
#include <numeric>
#include <vector>

#include "../include/hipblaslt_random.hpp"
#include "../include/unit.hpp"
#include "hipblaslt_arguments.hpp"

namespace
{
    template <typename DType>
    void cpuSoftmax(DType* m, DType* a, std::uint32_t numRows, std::uint32_t numCols)
    {
        for(std::uint32_t i = 0; i < numRows; ++i)
        {
            const auto rowMax = *std::max_element(a + i * numCols, a + i * numCols + numCols);
            auto       rowSum = 0.f;
            std::transform(a + i * numCols,
                           a + i * numCols + numCols,
                           m + i * numCols,
                           [&rowSum, rowMax](auto v) {
                               const auto u = std::exp(v - rowMax);
                               rowSum += u;
                               return u;
                           });

            std::transform(m + i * numCols,
                           m + i * numCols + numCols,
                           m + i * numCols,
                           [rowSum](auto v) { return v / rowSum; });
        }
    }

    template <typename DType>
    void cpuLayerNorm(DType*        out,
                      DType*        mean,
                      DType*        invvar,
                      DType*        in,
                      std::uint32_t batch,
                      std::uint32_t length,
                      DType         eps   = 1e-05,
                      DType*        gamma = nullptr,
                      DType*        beta  = nullptr)
    {
        // calculate mean
        for(int i = 0; i < batch; i++)
        {
            int    count = 0;
            DType* inC   = in + i * length;
            DType* outC  = out + i * length;

            for(int j = 0; j < length; j++)
            {
                count        = count + 1;
                float delta  = inC[j] - mean[i];
                mean[i]      = mean[i] + delta / count;
                float delta2 = inC[j] - mean[i];
                invvar[i]    = invvar[i] + delta * delta2;
            }
            invvar[i] = 1 / std::sqrt((invvar[i] / length) + eps);

            // calculate invvar
            for(int j = 0; j < length; j++)
            {
                outC[j] = (inC[j] - mean[i]) * invvar[i];

                if(gamma != nullptr)
                    outC[j] = outC[j] * gamma[j];

                if(beta != nullptr)
                    outC[j] = outC[j] + beta[j];
            }
        }
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
    void cpuAMax(To* out, Ti* in, std::uint32_t length)
    {
        // calculate amax
        Ti m = 0;
        for(int j = 0; j < length; j++)
        {
            m = max(m, abs(in[j]));
        }
        out[0] = To(m);
    }

    template <typename Ti, typename To>
    void cpuValueDividedbyAMaxWithRcp(To* out, To* outRcp, Ti* in, std::uint32_t length, float value)
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

    template <typename Ti, typename To, typename Ts>
    void cpuAMaxWithScale(To* out, Ts* outD, Ti* in, float* in_scale, std::uint32_t length)
    {
        // calculate amax
        Ti m = 0;
        for(int j = 0; j < length; j++)
        {
            m       = max(m, abs(in[j]));
            outD[j] = static_cast<Ts>(in[j] * in_scale[0]);
        }
        out[0] = To(m);
    }

    template<typename Ti, typename To>
    void cpuAMax2D(To* out, const Ti* in, const std::uint32_t M, const std::uint32_t N, const std::uint32_t B, const std::uint32_t LD, const std::uint32_t Stride)
    {
        Ti tmp = Ti(-999);
        for (int b=0; b<B; b++)
        {
            for(int n=0; n<N; n++)
            {
                for(int m=0; m<M; m++)
                {
                    tmp = max(tmp, abs(in[m + n * LD + b * Stride]));
                }
            }
        }

        out[0] = To(tmp);
    }
}

enum class amaxInitMethod
{
    hpl = 111,
    nan = 222,
    max = 333,
    min = 444
};

struct AMaxTestData
{
    hipDataType type;
    hipDataType dtype;
    uint32_t    m;
    uint32_t    n;
    bool hasWorkspace;
};

struct AMaxWithScaleTestData
{
    hipDataType    type;
    hipDataType    dtype;
    hipDataType    scaleType;
    amaxInitMethod initMethod;
    uint32_t       m;
    uint32_t       n;
    bool hasWorkspace;
};

struct ValueDividedbyAMaxWithRcpTestData
{
    hipDataType type;
    hipDataType dtype;
    uint32_t m;
    uint32_t n;
    float value;
};

struct AMax2DTestData
{
    hipDataType type;
    hipDataType dtype;
    uint32_t    m;
    uint32_t    n;
    uint32_t    ld;
};

class ExtOpSoftmaxTest : public testing::TestWithParam<uint32_t>
{
};
class ExtOpSoftmaxUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpLayerNormTest : public testing::TestWithParam<uint32_t>
{
};
class ExtOpLayerNormUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpAMaxTest : public testing::TestWithParam<AMaxTestData>
{
};
class ExtOpAMaxUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpAMaxWithScaleTest : public testing::TestWithParam<AMaxWithScaleTestData>
{
};
class ExtOpAMaxWithScaleUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpValueDividedbyAMaxWithRcpTest : public testing::TestWithParam<ValueDividedbyAMaxWithRcpTestData>
{
};
class ExtOpValueDividedbyAMaxWithRcpUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

class ExtOpAMax2DTest : public testing::TestWithParam<AMax2DTestData>
{
};
class ExtOpAMax2DUnsupportedDatatypeTest : public testing::TestWithParam<hipDataType>
{
};

TEST_P(ExtOpSoftmaxTest, softmaxSuccess)
{
    uint32_t           m = GetParam();
    uint32_t           n = 16;
    std::vector<float> input(m * n, 0.f);
    std::vector<float> output(m * n, 0.f);
    hipblaslt_uniform_int_1_10_run_float(input.data(), input.size());
    float* gpuInput{};
    float* gpuOutput{};

    int             deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    if(gpu_arch_match(deviceProperties.gcnArchName, "11?"))
        return;

    auto err          = hipMalloc(&gpuInput, m * n * sizeof(float));
    err               = hipMalloc(&gpuOutput, m * n * sizeof(float));
    err               = hipMemcpyHtoD(gpuInput, input.data(), m * n * sizeof(float));
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, m, n, 1, gpuOutput, gpuInput, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    err = hipDeviceSynchronize();
    EXPECT_EQ(err, hipSuccess);
    std::vector<float> cpuRef(m * n, 0.f);
    cpuSoftmax(cpuRef.data(), input.data(), m, n);
    err = hipMemcpyDtoH(output.data(), gpuOutput, m * n * sizeof(float));

    for(std::size_t i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(output[i], cpuRef[i], 1e-5);
    }

    err = hipFree(gpuInput);
    err = hipFree(gpuOutput);
}

TEST_P(ExtOpLayerNormTest, layernormSuccess)
{
    uint32_t m = GetParam();
    uint32_t n = 16;

    std::vector<float> output(m * n, 0.f);
    std::vector<float> mean(m, 0.f);
    std::vector<float> invvar(m, 0.f);
    std::vector<float> input(m * n, 0.f);
    std::vector<float> gamma(n, 1.f);
    std::vector<float> beta(n, 0.f);

    hipblaslt_init_hpl(input, n, m, n);
    hipblaslt_init_hpl(gamma, n, 1, n);
    hipblaslt_init_hpl(beta, n, 1, n);

    float* gpuOutput{};
    float* gpuMean{};
    float* gpuInvvar{};
    float* gpuInput{};
    float* gpuGamma{};
    float* gpuBeta{};

    int             deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    if(gpu_arch_match(deviceProperties.gcnArchName, "11?"))
        return;

    auto err = hipMalloc(&gpuOutput, m * n * sizeof(float));
    err      = hipMalloc(&gpuMean, m * sizeof(float));
    err      = hipMalloc(&gpuInvvar, m * sizeof(float));
    err      = hipMalloc(&gpuInput, m * n * sizeof(float));
    err      = hipMalloc(&gpuGamma, n * sizeof(float));
    err      = hipMalloc(&gpuBeta, n * sizeof(float));

    err = hipMemcpyHtoD(gpuInput, input.data(), m * n * sizeof(float));
    err = hipMemcpyHtoD(gpuGamma, gamma.data(), n * sizeof(float));
    err = hipMemcpyHtoD(gpuBeta, beta.data(), n * sizeof(float));

    auto hipblasltErr = hipblasltExtLayerNorm(HIP_R_32F,
                                              gpuOutput,
                                              gpuMean,
                                              gpuInvvar,
                                              gpuInput,
                                              m,
                                              n,
                                              1e-05,
                                              gpuGamma,
                                              gpuBeta,
                                              nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_SUCCESS);
    err = hipDeviceSynchronize();
    EXPECT_EQ(err, hipSuccess);

    std::vector<float> cpuRef(m * n, 0.0f);
    std::vector<float> cpuMean(m, 0.0f);
    std::vector<float> cpuInvvar(m, 0.0f);
    cpuLayerNorm<float>(cpuRef.data(),
                        cpuMean.data(),
                        cpuInvvar.data(),
                        input.data(),
                        m,
                        n,
                        1e-05,
                        gamma.data(),
                        beta.data());

    err = hipMemcpyDtoH(output.data(), gpuOutput, m * n * sizeof(float));
    err = hipMemcpyDtoH(mean.data(), gpuMean, m * sizeof(float));
    err = hipMemcpyDtoH(invvar.data(), gpuInvvar, m * sizeof(float));

    for(std::size_t i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(output[i], cpuRef[i], 1e-5);
    }
    for(std::size_t i = 0; i < m; ++i)
    {
        EXPECT_NEAR(mean[i], cpuMean[i], 1e-5);
    }
    for(std::size_t i = 0; i < m; ++i)
    {
        EXPECT_NEAR(invvar[i], cpuInvvar[i], 1e-5);
    }

    err = hipFree(gpuOutput);
    err = hipFree(gpuMean);
    err = hipFree(gpuInvvar);
    err = hipFree(gpuInput);
}

template <typename Ti, typename To>
void AMaxTest(hipDataType type, hipDataType dtype, std::size_t m, std::size_t n, bool hasWorkspace)
{
    std::size_t numElements = m * n;

    To* gpuOutput{nullptr};
    Ti* gpuInput{nullptr};
    Ti* gpuWorkSpace{nullptr};
    std::int32_t* gpuSync{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, sizeof(To));
    hipErr      = hipMalloc(&gpuInput, m * n * sizeof(Ti));
    if (hasWorkspace)
    {
        hipErr      = hipMalloc(&gpuWorkSpace, 4096 * sizeof(Ti));
        hipErr      = hipMalloc(&gpuSync, sizeof(std::int32_t));
    }

    std::vector<To> cpuOutput(1, 0.f);
    std::vector<Ti> cpuInput(m * n, 0.f);
    std::vector<To> refOutput(1, 0.f);

    hipblaslt_init_hpl(cpuInput, m * n, 1, m * n);

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), m * n * sizeof(Ti));
    if (hasWorkspace)
    {
        hipErr = hipMemset(gpuWorkSpace, 0, m * n * sizeof(Ti));
        hipErr = hipMemset(gpuSync, 0, sizeof(std::int32_t));
    }

    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    hipblasStatus_t hipblasltErr;
    if (hasWorkspace)
        hipblasltErr = hipblasltExtFastAMax(type, dtype, gpuOutput, gpuInput, gpuWorkSpace, gpuSync, m, n, stream);
    else
        hipblasltErr = hipblasltExtAMax(type, dtype, gpuOutput, gpuInput, m, n, stream);

    hipErr = hipDeviceSynchronize();

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(To));

    cpuAMax(refOutput.data(), cpuInput.data(), m * n);

    EXPECT_NEAR(float(refOutput[0]), float(cpuOutput[0]), 1e-5);

    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuInput);
}

template <typename Ti, typename To, typename Ts>
void AMaxTestWithScale(hipDataType    type,
                       hipDataType    dtype,
                       hipDataType    scaleType,
                       amaxInitMethod initMethod,
                       std::size_t    m,
                       std::size_t    n,
                       bool           hasWorkspace)
{
    int             deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    if(!gpu_arch_match(deviceProperties.gcnArchName, "94?"))
        return;

    std::size_t numElements   = m * n;

    To*           gpuOutput{nullptr};
    Ti*           gpuInput{nullptr};
    Ts*           gpuOutputD;
    float*        gpuInputScale;
    Ti*           gpuWorkSpace{nullptr};
    std::int32_t* gpuSync{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, sizeof(To));
    hipErr      = hipMalloc(&gpuInput, m * n * sizeof(Ti));
    hipErr      = hipMalloc(&gpuOutputD, m * n * sizeof(Ts));
    hipErr      = hipMalloc(&gpuInputScale, 1 * sizeof(float));

    if (hasWorkspace)
    {
        hipErr      = hipMalloc(&gpuWorkSpace, 4096 * sizeof(Ti));
        hipErr      = hipMalloc(&gpuSync, sizeof(std::int32_t));
    }

    std::vector<To>    cpuOutput(1, 0.f);
    std::vector<Ti>    cpuInput(m * n, 0.f);
    std::vector<Ts>    cpuOutputD(m * n);
    std::vector<float> cpuInputScale(1);
    std::vector<To>    refOutput(1, 0.f);
    std::vector<Ts>    refOutputD(m * n);

    switch(initMethod)
    {
    case amaxInitMethod::hpl:
        hipblaslt_init_hpl<Ti>(cpuInput, m * n, 1, m * n);
        break;
    case amaxInitMethod::nan:
        std::fill(cpuInput.begin(), cpuInput.end(), std::numeric_limits<Ti>::infinity());
        break;
    case amaxInitMethod::max:
        std::fill(cpuInput.begin(), cpuInput.end(), std::numeric_limits<Ti>::max());
        break;
    case amaxInitMethod::min:
        std::fill(cpuInput.begin(), cpuInput.end(), -std::numeric_limits<Ti>::max());
        break;
    default:
        break;
    }
    cpuInputScale[0] = (float)0.5;

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), m * n * sizeof(Ti));
    hipErr = hipMemcpyHtoD(gpuInputScale, cpuInputScale.data(), 1 * sizeof(float));
    if (hasWorkspace)
    {
        hipErr = hipMemset(gpuWorkSpace, 0, 4096 * sizeof(Ti));
        hipErr = hipMemset(gpuSync, 0, 4096 * sizeof(std::int32_t));
    }

    hipStream_t stream{};
    hipErr            = hipStreamCreate(&stream);
    hipblasStatus_t hipblasltErr;
    if (hasWorkspace)
        hipblasltErr = hipblasltExtFastAMaxWithScale(type, dtype, scaleType, gpuOutput, gpuOutputD, gpuInput, gpuInputScale, gpuWorkSpace, gpuSync, m, n, stream);
    else
        hipblasltErr = hipblasltExtAMaxWithScale(type, dtype, scaleType, gpuOutput, gpuOutputD, gpuInput, gpuInputScale, m, n, stream);

    hipErr = hipDeviceSynchronize();

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(To));
    hipErr = hipMemcpyDtoH(cpuOutputD.data(), gpuOutputD, m * n * sizeof(Ts));

    cpuAMaxWithScale(
        refOutput.data(), refOutputD.data(), cpuInput.data(), cpuInputScale.data(), m * n);

    unit_check_general<To>(1, 1, 1, refOutput.data(), (const To*)cpuOutput.data());
    unit_check_general<Ts>(m, n, 1, refOutputD.data(), (const Ts*)cpuOutputD.data());

    hipErr = hipStreamDestroy(stream);

    if (hasWorkspace)
    {
        hipErr = hipFree(gpuSync);
        hipErr = hipFree(gpuWorkSpace);
    }

    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuInput);
    hipErr = hipFree(gpuOutputD);
    hipErr = hipFree(gpuInputScale);
}

template <typename Ti, typename To>
void ValueDividedbyAMaxWithRcpTest(hipDataType type, hipDataType dtype, std::size_t m, std::size_t n, size_t value)
{
    std::size_t numElements = m * n;

    To* gpuOutput{nullptr};
    To* gpuOutputRcp{nullptr};
    Ti* gpuInput{nullptr};
    Ti* gpuWorkSpace{nullptr};
    std::int32_t* gpuSync{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, sizeof(To));
    hipErr      = hipMalloc(&gpuOutputRcp, sizeof(To));
    hipErr      = hipMalloc(&gpuInput, m * n * sizeof(Ti));
    hipErr      = hipMalloc(&gpuWorkSpace, 4096 * sizeof(Ti));
    hipErr      = hipMalloc(&gpuSync, sizeof(std::int32_t));

    std::vector<To> refOutput(1, -999.f);
    std::vector<To> refOutputRcp(1, -999.f);
    std::vector<To> cpuOutput(1, -999.f);
    std::vector<To> cpuOutputRcp(1, -999.f);
    std::vector<Ti> cpuInput(m * n, 0.f);

    hipblaslt_init_hpl(cpuInput, m * n, 1, m * n);

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), m * n * sizeof(Ti));
    hipErr = hipMemset(gpuWorkSpace, 0, 4096 * sizeof(Ti));
    hipErr = hipMemset(gpuSync, 0, sizeof(std::int32_t));

    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    auto hipblasltErr = hipblasltExtFastValueDividedByAMaxWithRcp(type, dtype, gpuOutput, gpuOutputRcp, gpuInput, gpuWorkSpace, gpuSync, m, n, value, stream);
    hipErr = hipDeviceSynchronize();

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(To));
    hipErr = hipMemcpyDtoH(cpuOutputRcp.data(), gpuOutputRcp, sizeof(To));

    cpuValueDividedbyAMaxWithRcp(refOutput.data(), refOutputRcp.data(), cpuInput.data(), m * n, value);

    EXPECT_NEAR(float(refOutput[0]), float(cpuOutput[0]), 1e-5);
    EXPECT_NEAR(float(refOutputRcp[0]), float(cpuOutputRcp[0]), 1e-5);

    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuOutputRcp);
    hipErr = hipFree(gpuInput);
    hipErr = hipFree(gpuWorkSpace);
    hipErr = hipFree(gpuSync);
}

template <typename Ti, typename To>
void AMax2DTest(hipDataType type, hipDataType dtype, std::size_t m, std::size_t n, std::size_t ld)
{
    std::size_t numAllocIn  = m + (n-1) * ld;
    std::size_t numElements = m * n;

    To* gpuOutput{nullptr};
    Ti* gpuInput{nullptr};
    Ti* gpuWorkSpace{nullptr};
    std::int32_t* gpuSync{nullptr};

    auto hipErr = hipMalloc(&gpuOutput, sizeof(To));
    hipErr      = hipMalloc(&gpuInput, numAllocIn * sizeof(Ti));
    hipErr      = hipMalloc(&gpuWorkSpace, 4096 * sizeof(Ti));
    hipErr      = hipMalloc(&gpuSync, sizeof(std::int32_t));

    std::vector<To> refOutput(1, -999.f);
    std::vector<To> refOutputRcp(1, -999.f);
    std::vector<To> cpuOutput(1, -999.f);
    std::vector<To> cpuOutputRcp(1, -999.f);
    std::vector<Ti> cpuInput(numAllocIn, 999.f);

    hipblaslt_init_hpl(cpuInput, m, n, ld);

    hipErr = hipMemcpyHtoD(gpuInput, cpuInput.data(), numAllocIn * sizeof(Ti));
    hipErr = hipMemset(gpuWorkSpace, 0, 4096 * sizeof(Ti));
    hipErr = hipMemset(gpuSync, 0, sizeof(std::int32_t));

    hipStream_t stream{};
    hipErr = hipStreamCreate(&stream);
    auto hipblasltErr = hipblasltExtFastAMax2D(type, dtype, gpuOutput, gpuInput, gpuWorkSpace, gpuSync, m, n, ld, stream);
    hipErr = hipDeviceSynchronize();

    hipErr = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(To));

    cpuAMax2D(refOutput.data(), cpuInput.data(), m, n, 1, ld, numAllocIn);

    EXPECT_NEAR(float(refOutput[0]), float(cpuOutput[0]), 1e-5);

    hipErr = hipStreamDestroy(stream);
    hipErr = hipFree(gpuOutput);
    hipErr = hipFree(gpuInput);
    hipErr = hipFree(gpuWorkSpace);
    hipErr = hipFree(gpuSync);
}

TEST_P(ExtOpAMaxTest, amaxSuccess)
{
    AMaxTestData    testdata = GetParam();
    int             deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    if(gpu_arch_match(deviceProperties.gcnArchName, "11?"))
        return;

    if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_32F)
    {
        AMaxTest<float, float>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.hasWorkspace);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_16F)
    {
        AMaxTest<float, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.hasWorkspace);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_32F)
    {
        AMaxTest<hipblasLtHalf, float>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.hasWorkspace);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_16F)
    {
        AMaxTest<hipblasLtHalf, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.hasWorkspace);
    }
}

TEST_P(ExtOpAMaxWithScaleTest, amaxSuccess)
{
    AMaxWithScaleTestData testdata = GetParam();
    if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_32F
       && testdata.scaleType == HIP_R_8F_E4M3_FNUZ)
    {
        AMaxTestWithScale<float, float, hipblaslt_f8_fnuz>(testdata.type,
                                                           testdata.dtype,
                                                           testdata.scaleType,
                                                           testdata.initMethod,
                                                           testdata.m,
                                                           testdata.n,
                                                           testdata.hasWorkspace);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_32F
            && testdata.scaleType == HIP_R_8F_E5M2_FNUZ)
    {
        AMaxTestWithScale<float, float, hipblaslt_bf8_fnuz>(testdata.type,
                                                            testdata.dtype,
                                                            testdata.scaleType,
                                                            testdata.initMethod,
                                                            testdata.m,
                                                            testdata.n,
                                                            testdata.hasWorkspace);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_16F
            && testdata.scaleType == HIP_R_8F_E4M3_FNUZ)
    {
        AMaxTestWithScale<float, hipblasLtHalf, hipblaslt_f8_fnuz>(testdata.type,
                                                                   testdata.dtype,
                                                                   testdata.scaleType,
                                                                   testdata.initMethod,
                                                                   testdata.m,
                                                                   testdata.n,
                                                                   testdata.hasWorkspace);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_16F
            && testdata.scaleType == HIP_R_8F_E5M2_FNUZ)
    {
        AMaxTestWithScale<float, hipblasLtHalf, hipblaslt_bf8_fnuz>(testdata.type,
                                                                    testdata.dtype,
                                                                    testdata.scaleType,
                                                                    testdata.initMethod,
                                                                    testdata.m,
                                                                    testdata.n,
                                                                    testdata.hasWorkspace);
    }
}

TEST_P(ExtOpValueDividedbyAMaxWithRcpTest, amaxSuccess)
{
    ValueDividedbyAMaxWithRcpTestData testdata = GetParam();
    int deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    if(gpu_arch_match(deviceProperties.gcnArchName, "11?"))
        return;

    if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_32F)
    {
        ValueDividedbyAMaxWithRcpTest<float, float>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.value);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_16F)
    {
        ValueDividedbyAMaxWithRcpTest<float, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.value);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_32F)
    {
        ValueDividedbyAMaxWithRcpTest<hipblasLtHalf, float>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.value);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_16F)
    {
        ValueDividedbyAMaxWithRcpTest<hipblasLtHalf, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.value);
    }
}

TEST_P(ExtOpAMax2DTest, amaxSuccess)
{
    AMax2DTestData testdata = GetParam();
    int deviceId;
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    if(gpu_arch_match(deviceProperties.gcnArchName, "11?"))
        return;

    if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_32F)
    {
        AMax2DTest<float, float>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.ld);
    }
    else if(testdata.type == HIP_R_32F && testdata.dtype == HIP_R_16F)
    {
        AMax2DTest<float, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.ld);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_32F)
    {
        AMax2DTest<hipblasLtHalf, float>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.ld);
    }
    else if(testdata.type == HIP_R_16F && testdata.dtype == HIP_R_16F)
    {
        AMax2DTest<hipblasLtHalf, hipblasLtHalf>(testdata.type, testdata.dtype, testdata.m, testdata.n, testdata.ld);
    }
}

TEST_P(ExtOpSoftmaxUnsupportedDatatypeTest, softmaxFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtSoftmax(GetParam(), 16, 16, 1, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, softmaxFailureUnsupportedShapeOrReductionDim)
{
    auto hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, 16, 512, 1, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
    hipblasltErr = hipblasltExtSoftmax(HIP_R_32F, 16, 16, 0, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST_P(ExtOpLayerNormUnsupportedDatatypeTest, layernormFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtLayerNorm(
        GetParam(), nullptr, nullptr, nullptr, nullptr, 16, 1024, 1e-05, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, layernormFailureInvalidValue)
{
    auto hipblasltErr = hipblasltExtLayerNorm(
        HIP_R_32F, nullptr, nullptr, nullptr, nullptr, 16, 1024, 1e-05, nullptr, nullptr, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_P(ExtOpAMaxUnsupportedDatatypeTest, amaxFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtAMax(GetParam(), GetParam(), nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, amaxFailureInvalidValue)
{
    auto hipblasltErr = hipblasltExtAMax(HIP_R_32F, HIP_R_32F, nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_P(ExtOpAMaxWithScaleUnsupportedDatatypeTest, amaxWithScaleFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtAMaxWithScale(
        GetParam(), GetParam(), GetParam(), nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST(ExtOpTest, amaxWithScaleFailureInvalidValue)
{
    auto hipblasltErr = hipblasltExtAMaxWithScale(
        HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_P(ExtOpValueDividedbyAMaxWithRcpUnsupportedDatatypeTest, valueDividedbyAMaxWithRcpFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtFastValueDividedByAMaxWithRcp(GetParam(), GetParam(), nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

TEST_P(ExtOpAMax2DUnsupportedDatatypeTest, amax2DFailureUnsupportedDatatype)
{
    auto hipblasltErr = hipblasltExtFastAMax2D(GetParam(), GetParam(), nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr);
    EXPECT_EQ(hipblasltErr, HIPBLAS_STATUS_NOT_SUPPORTED);
}

INSTANTIATE_TEST_SUITE_P(ExtOpTest, ExtOpSoftmaxTest, testing::Values<uint32_t>(1, 16, 1335));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpSoftmaxUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16F, HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpLayerNormTest,
                         testing::Values<uint32_t>(1, 16, 1335, 6666));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpLayerNormUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16F, HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(
    ExtOpTest,
    ExtOpAMaxTest,
    testing::Values<AMaxTestData>(AMaxTestData{HIP_R_32F, HIP_R_32F, 1, 1, false},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 100, 213, false},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 1335, 6666, false},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1, 1, false},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 100, 213, false},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1335, 6666, false},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1, 1, false},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 100, 213, false},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1335, 6666, false},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1, 1, false},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 100, 213, false},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1335, 6666, false},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 1, 1, true},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 100, 213, true},
                                  AMaxTestData{HIP_R_32F, HIP_R_32F, 1335, 6666, true},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1, 1, true},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 100, 213, true},
                                  AMaxTestData{HIP_R_32F, HIP_R_16F, 1335, 6666, true},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1, 1, true},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 100, 213, true},
                                  AMaxTestData{HIP_R_16F, HIP_R_32F, 1335, 6666, true},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1, 1, true},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 100, 213, true},
                                  AMaxTestData{HIP_R_16F, HIP_R_16F, 1335, 6666, true}));
INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpAMaxUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(
    ExtOpTest,
    ExtOpAMaxWithScaleTest,
    testing::Values<AMaxWithScaleTestData>(
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 16, 16, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 16, 16, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1335, 6666, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1335, 6666, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 16, 16, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 16, 16, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1335, 6666, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1335, 6666, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::nan, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::nan, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::max, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::max, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::min, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::min, 1, 1, false},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 16, 16, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 16, 16, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1335, 6666, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1335, 6666, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 16, 16, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 16, 16, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::hpl, 1335, 6666, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_16F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::hpl, 1335, 6666, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::nan, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::nan, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::max, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::max, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E4M3_FNUZ, amaxInitMethod::min, 1, 1, true},
        AMaxWithScaleTestData{HIP_R_32F, HIP_R_32F, HIP_R_8F_E5M2_FNUZ, amaxInitMethod::min, 1, 1, true}));

INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpAMaxWithScaleUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpValueDividedbyAMaxWithRcpUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(
    ExtOpTest,
    ExtOpValueDividedbyAMaxWithRcpTest,
    testing::Values<ValueDividedbyAMaxWithRcpTestData>(
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_32F, HIP_R_32F, 1, 1, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_32F, HIP_R_32F, 100, 213, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_32F, HIP_R_32F, 1335, 6666, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_32F, HIP_R_16F, 1, 1, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_32F, HIP_R_16F, 100, 213, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_32F, HIP_R_16F, 1335, 6666, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_16F, HIP_R_32F, 1, 1, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_16F, HIP_R_32F, 100, 213, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_16F, HIP_R_32F, 1335, 6666, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_16F, HIP_R_16F, 1, 1, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_16F, HIP_R_16F, 100, 213, 10},
                                  ValueDividedbyAMaxWithRcpTestData{HIP_R_16F, HIP_R_16F, 1335, 6666, 10}
    ));

INSTANTIATE_TEST_SUITE_P(ExtOpTest,
                         ExtOpAMax2DUnsupportedDatatypeTest,
                         testing::Values<hipDataType>(HIP_R_16BF));

INSTANTIATE_TEST_SUITE_P(
    ExtOpTest,
    ExtOpAMax2DTest,
    testing::Values<AMax2DTestData>(
                                  AMax2DTestData{HIP_R_32F, HIP_R_32F, 8, 8, 8},
                                  AMax2DTestData{HIP_R_32F, HIP_R_32F, 100, 213, 100},
                                  AMax2DTestData{HIP_R_32F, HIP_R_32F, 1335, 6666, 1335},
                                  AMax2DTestData{HIP_R_32F, HIP_R_16F, 8, 8, 8},
                                  AMax2DTestData{HIP_R_32F, HIP_R_16F, 100, 213, 100},
                                  AMax2DTestData{HIP_R_32F, HIP_R_16F, 1335, 6666, 1335},
                                  AMax2DTestData{HIP_R_16F, HIP_R_32F, 8, 8, 8},
                                  AMax2DTestData{HIP_R_16F, HIP_R_32F, 100, 213, 100},
                                  AMax2DTestData{HIP_R_16F, HIP_R_32F, 1335, 6666, 1335},
                                  AMax2DTestData{HIP_R_16F, HIP_R_16F, 8, 8, 8},
                                  AMax2DTestData{HIP_R_16F, HIP_R_16F, 100, 213, 100},
                                  AMax2DTestData{HIP_R_16F, HIP_R_16F, 1335, 6666, 1335},
                                  AMax2DTestData{HIP_R_32F, HIP_R_32F, 8, 8, 128},
                                  AMax2DTestData{HIP_R_32F, HIP_R_32F, 100, 213, 256},
                                  AMax2DTestData{HIP_R_32F, HIP_R_32F, 1335, 6666, 2048},
                                  AMax2DTestData{HIP_R_32F, HIP_R_16F, 8, 8, 128},
                                  AMax2DTestData{HIP_R_32F, HIP_R_16F, 100, 213, 256},
                                  AMax2DTestData{HIP_R_32F, HIP_R_16F, 1335, 6666, 2048},
                                  AMax2DTestData{HIP_R_16F, HIP_R_32F, 8, 8, 128},
                                  AMax2DTestData{HIP_R_16F, HIP_R_32F, 100, 213, 256},
                                  AMax2DTestData{HIP_R_16F, HIP_R_32F, 1335, 6666, 2048},
                                  AMax2DTestData{HIP_R_16F, HIP_R_16F, 8, 8, 128},
                                  AMax2DTestData{HIP_R_16F, HIP_R_16F, 100, 213, 256},
                                  AMax2DTestData{HIP_R_16F, HIP_R_16F, 1335, 6666, 2048}
    ));
