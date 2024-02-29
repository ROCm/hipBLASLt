/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipblaslt-ext-op.h"
#include "hipblaslt-ext-op-internal.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>
#include <libgen.h>
#include <memory>
#include <rocblaslt-auxiliary.h>
#include <sstream>
#include <string>
#include <tensile_host.hpp>
#include <unordered_map>
#include <vector>

hipblasStatus_t hipblasltSoftmaxRun(hipDataType datatype,
                                    uint32_t    m,
                                    uint32_t    n,
                                    uint32_t    dim,
                                    void*       output,
                                    void*       input,
                                    hipStream_t stream);

hipblasStatus_t hipblasltExtSoftmax(hipDataType datatype,
                                    uint32_t    m,
                                    uint32_t    n,
                                    uint32_t    dim,
                                    void*       output,
                                    void*       input,
                                    hipStream_t stream)
{
    return hipblasltSoftmaxRun(datatype, m, n, dim, output, input, stream);
}

hipblasStatus_t hipblasltLayerNormRun(hipDataType datatype,
                                      void*       output,
                                      void*       mean,
                                      void*       invvar,
                                      void*       input,
                                      uint32_t    m,
                                      uint32_t    n,
                                      float       eps,
                                      void*       gamma,
                                      void*       beta,
                                      hipStream_t stream);

hipblasStatus_t hipblasltExtLayerNorm(hipDataType datatype,
                                      void*       output,
                                      void*       mean,
                                      void*       invvar,
                                      void*       input,
                                      uint32_t    m,
                                      uint32_t    n,
                                      float       eps,
                                      void*       gamma,
                                      void*       beta,
                                      hipStream_t stream)
{
    return hipblasltLayerNormRun(
        datatype, output, mean, invvar, input, m, n, eps, gamma, beta, stream);
}

hipblasStatus_t hipblasltAMaxRun(const hipDataType datatype,
                                 const hipDataType outDatatype,
                                 void*             output,
                                 void*             input,
                                 uint32_t          m,
                                 uint32_t          n,
                                 hipStream_t       stream);

hipblasStatus_t hipblasltAMaxWithScaleRun(const hipDataType datatype,
                                          const hipDataType outDatatype,
                                          const hipDataType scaleDatatype,
                                          void*             output,
                                          void*             outputD,
                                          void*             input,
                                          void*             inputScale,
                                          uint32_t          m,
                                          uint32_t          n,
                                          hipStream_t       stream);

hipblasStatus_t hipblasltExtAMax(const hipDataType datatype,
                                 const hipDataType outDatatype,
                                 void*             output,
                                 void*             input,
                                 uint32_t          m,
                                 uint32_t          n,
                                 hipStream_t       stream)
{
    return hipblasltAMaxRun(datatype, outDatatype, output, input, m, n, stream);
}

hipblasStatus_t hipblasltExtAMaxWithScale(const hipDataType datatype,
                                          const hipDataType outDatatype,
                                          const hipDataType scaleDatatype,
                                          void*             output,
                                          void*             outputD,
                                          void*             input,
                                          void*             inputScale,
                                          uint32_t          m,
                                          uint32_t          n,
                                          hipStream_t       stream)
{
    return hipblasltAMaxWithScaleRun(
        datatype, outDatatype, scaleDatatype, output, outputD, input, inputScale, m, n, stream);
}

namespace
{
    constexpr char DEFAULT_EXT_OP_LIBRARY_PATH[]
        = "/opt/rocm/lib/hipblaslt/library/hipblasltExtOpLibrary.dat";
    constexpr uint32_t SUPPORTED_MAX_N = 256;
    constexpr uint32_t WORKGROUP_SIZE  = 256;

    std::string trimArchName(const std::string& archName)
    {
        auto pos = archName.find(':');

        if(pos != std::string::npos)
        {
            return archName.substr(0, pos);
        }

        return archName;
    }

    std::string getExtOpLibraryPath()
    {
        if(auto libPath = std::getenv("HIPBLASLT_EXT_OP_LIBRARY_PATH"))
        {
            return libPath;
        }

        auto        soPath = rocblaslt_internal_get_so_path("hipblaslt");
        std::string libPath(dirname(&soPath[0]));

        if(rocblaslt_internal_test_path(libPath + "/../Tensile/library"))
            libPath += "/../Tensile/library";
        else if(rocblaslt_internal_test_path(libPath + "library"))
            libPath += "/library";
        else
            libPath += "/hipblaslt/library";

        libPath += "/hipblasltExtOpLibrary.dat";

        if(rocblaslt_internal_test_path(libPath))
        {
            return libPath;
        }

        return DEFAULT_EXT_OP_LIBRARY_PATH;
    }

    std::string hipDataTypeo_char(hipDataType type)
    {
        if(type == HIP_R_16F)
            return std::string("H");
        else if(type == HIP_R_32F)
            return std::string("S");
        return std::string("S");
    }

    inline uint32_t getSoftmaxNumWorkgroups(uint32_t m, uint32_t tileM)
    {
        return (m / tileM) + !!(m % tileM);
    }

    inline uint32_t getSoftmaxBestKernelTileN(uint32_t n)
    {
        const uint32_t exponent = std::ceil(std::log2(n));
        return 1 << exponent;
    }

    inline uint32_t getSoftmaxKernelTileM(uint32_t tileM)
    {
        return WORKGROUP_SIZE / tileM;
    }

    uint32_t elementNumBytes(hipDataType type)
    {
        if(type == HIP_R_16F)
            return 2;
        else if(type == HIP_R_32F)
            return 4;
        return 4;
    }

    inline uint32_t getLdsUsageByte(hipDataType datatype, uint32_t tileM, uint32_t tileN)
    {
        return elementNumBytes(datatype) * tileM * tileN;
    }

    static const ExtOpMasterLibrary& getExtOpMasterLibrary()
    {
        static ExtOpMasterLibrary lib(getExtOpLibraryPath());
        return lib;
    }

    std::vector<std::unique_ptr<Tensile::hip::SolutionAdapter>>& extOpLibraries()
    {
        static std::vector<std::unique_ptr<Tensile::hip::SolutionAdapter>> adapters;

        if(adapters.size())
        {
            return adapters;
        }

        int  numDevices{};
        auto err = hipGetDeviceCount(&numDevices);

        for(std::size_t i = 0; i < numDevices; ++i)
        {
            adapters.emplace_back(std::make_unique<Tensile::hip::SolutionAdapter>());
        }

        int currentDevice{};
        err = hipGetDevice(&currentDevice);

        try
        {
            auto& lib = getExtOpMasterLibrary();

            for(auto& adapter : adapters)
            {
                //setup code object root only, ignore the error
                err = adapter->initializeLazyLoading("", lib.getLibraryFolder());
            }
        }
        catch(const std::runtime_error& e)
        {
            rocblaslt_log_error("extOpLibraries", "ExtOpLibPath", getExtOpLibraryPath().c_str());
        }

        return adapters;
    }
}

hipblasStatus_t hipblasltSoftmaxRun(hipDataType datatype,
                                    uint32_t    m,
                                    uint32_t    n,
                                    uint32_t    dim,
                                    void*       output,
                                    void*       input,
                                    hipStream_t stream)
{
    if(datatype != HIP_R_32F)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if(dim != 1)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    const auto tileN = getSoftmaxBestKernelTileN(n);
    const auto tileM = getSoftmaxKernelTileM(tileN);

    if(tileN > SUPPORTED_MAX_N)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int         currentDeviceId{};
    auto        err       = hipGetDevice(&currentDeviceId);
    auto&       adapter   = extOpLibraries().at(currentDeviceId);
    auto        gpu       = Tensile::hip::GetCurrentDevice();
    const auto  archName  = trimArchName(gpu->archName());
    auto&       masterLib = getExtOpMasterLibrary();
    const auto& lib
        = masterLib
              .getLibrary(archName, SoftmaxSolutionLibrary::opName, hipDataTypeo_char(datatype))
              ->as<SoftmaxSolutionLibrary>();
    auto sol
        = lib.findBestSolution(SoftmaxProblem(m, n, hipDataType_to_tensile_type(datatype)), *gpu);
    const auto kernelName = sol->name();
    err                   = adapter->initKernel(kernelName);
    Tensile::KernelArguments kArgs(false);
    kArgs.append("input", input);
    kArgs.append("output", output);
    kArgs.append("m", m);
    kArgs.append("n", n);
    const auto                numWorkgroups = getSoftmaxNumWorkgroups(m, tileM);
    Tensile::KernelInvocation invocation{kernelName,
                                         sol->getCodeObjectPath(),
                                         {WORKGROUP_SIZE, 1, 1},
                                         {numWorkgroups, 1, 1},
                                         {numWorkgroups * WORKGROUP_SIZE, 1, 1},
                                         getLdsUsageByte(datatype, tileM, tileN),
                                         kArgs};

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if(err)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasltLayerNormRun(hipDataType datatype,
                                      void*       output,
                                      void*       mean,
                                      void*       invvar,
                                      void*       input,
                                      uint32_t    m,
                                      uint32_t    n,
                                      float       eps,
                                      void*       gamma,
                                      void*       beta,
                                      hipStream_t stream)
{
    if(datatype != HIP_R_32F)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if(output == nullptr || mean == nullptr || invvar == nullptr || input == nullptr || m == 0
       || n == 0)
        return HIPBLAS_STATUS_INVALID_VALUE;

    int         currentDeviceId{};
    auto        err       = hipGetDevice(&currentDeviceId);
    auto&       adapter   = extOpLibraries().at(currentDeviceId);
    auto        gpu       = Tensile::hip::GetCurrentDevice();
    const auto  archName  = trimArchName(gpu->archName());
    auto&       masterLib = getExtOpMasterLibrary();
    const auto& lib
        = masterLib
              .getLibrary(archName, LayerNormSolutionLibrary::opName, hipDataTypeo_char(datatype))
              ->as<LayerNormSolutionLibrary>();
    auto sol
        = lib.findBestSolution(LayerNormProblem(m, n, hipDataType_to_tensile_type(datatype)), *gpu);
    const auto kernelName    = sol->name();
    err                      = adapter->initKernel(kernelName);
    const auto numWorkgroups = m;

    Tensile::KernelInvocation invocation;
    invocation.kernelName      = kernelName;
    invocation.codeObjectFile  = sol->getCodeObjectPath();
    invocation.workGroupSize.x = sol->getNumWorkitems();
    invocation.workGroupSize.y = 1;
    invocation.workGroupSize.z = 1;
    invocation.numWorkGroups.x = 1;
    invocation.numWorkGroups.y = numWorkgroups;
    invocation.numWorkGroups.z = 1;
    invocation.numWorkItems.x  = sol->getNumWorkitems();
    invocation.numWorkItems.y  = numWorkgroups;
    invocation.numWorkItems.z  = 1;
    invocation.sharedMemBytes  = 32 * sizeof(float);
    invocation.args            = Tensile::KernelArguments(false);
    invocation.args.reserve(60, 9);
    invocation.args.append("output", output);
    invocation.args.append("mean", mean);
    invocation.args.append("invvar", invvar);
    invocation.args.append("input", input);
    invocation.args.append("gamma", gamma);
    invocation.args.append("beta", beta);
    invocation.args.append("m", m);
    invocation.args.append("n", n);
    invocation.args.append("eps", eps);

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if(err)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasltAMaxRun(const hipDataType datatype,
                                 const hipDataType outDatatype,
                                 void*             output,
                                 void*             input,
                                 uint32_t          m,
                                 uint32_t          n,
                                 hipStream_t       stream)
{
    if(datatype != HIP_R_32F && datatype != HIP_R_16F)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if(output == nullptr || input == nullptr || m == 0 || n == 0)
        return HIPBLAS_STATUS_INVALID_VALUE;

    uint32_t    len = m * n;
    int         currentDeviceId{};
    auto        err       = hipGetDevice(&currentDeviceId);
    auto&       adapter   = extOpLibraries().at(currentDeviceId);
    auto        gpu       = Tensile::hip::GetCurrentDevice();
    const auto  archName  = trimArchName(gpu->archName());
    auto&       masterLib = getExtOpMasterLibrary();
    const auto& lib
        = masterLib.getLibrary(archName, AMaxSolutionLibrary::opName, hipDataTypeo_char(datatype))
              ->as<AMaxSolutionLibrary>();
    auto       sol        = lib.findBestSolution(AMaxProblem(len,
                                                hipDataType_to_tensile_type(datatype),
                                                hipDataType_to_tensile_type(outDatatype)),
                                    *gpu);
    const auto kernelName = sol->name();
    err                   = adapter->initKernel(kernelName);

    Tensile::KernelInvocation invocation;
    invocation.kernelName      = kernelName;
    invocation.codeObjectFile  = sol->getCodeObjectPath();
    invocation.workGroupSize.x = sol->getNumWorkitems();
    invocation.workGroupSize.y = 1;
    invocation.workGroupSize.z = 1;
    invocation.numWorkGroups.x = 1;
    invocation.numWorkGroups.y = 1;
    invocation.numWorkGroups.z = 1;
    invocation.numWorkItems.x  = sol->getNumWorkitems();
    invocation.numWorkItems.y  = 1;
    invocation.numWorkItems.z  = 1;
    invocation.sharedMemBytes  = 32 * sizeof(float);
    invocation.args            = Tensile::KernelArguments(false);
    invocation.args.reserve(20, 3);
    invocation.args.append("output", output);
    invocation.args.append("input", input);
    invocation.args.append("length", len);

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if(err)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasltAMaxWithScaleRun(const hipDataType datatype,
                                          const hipDataType outDatatype,
                                          const hipDataType scaleDatatype,
                                          void*             output,
                                          void*             outputD,
                                          void*             input,
                                          void*             inputScale,
                                          uint32_t          m,
                                          uint32_t          n,
                                          hipStream_t       stream)
{
    if(datatype != HIP_R_32F
       || scaleDatatype != HIP_R_8F_E4M3_FNUZ && scaleDatatype != HIP_R_8F_E5M2_FNUZ)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if(!output || !outputD || !input || !inputScale || !m || !n)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    uint32_t    len = m * n;
    int         currentDeviceId{};
    auto        err       = hipGetDevice(&currentDeviceId);
    auto&       adapter   = extOpLibraries().at(currentDeviceId);
    auto        gpu       = Tensile::hip::GetCurrentDevice();
    const auto  archName  = trimArchName(gpu->archName());
    auto&       masterLib = getExtOpMasterLibrary();
    const auto& lib
        = masterLib.getLibrary(archName, AMaxSolutionLibrary::opName, hipDataTypeo_char(datatype))
              ->as<AMaxSolutionLibrary>();
    auto sol = lib.findBestSolution(AMaxProblem(len,
                                                hipDataType_to_tensile_type(datatype),
                                                hipDataType_to_tensile_type(outDatatype),
                                                hipDataType_to_tensile_type(scaleDatatype),
                                                true),
                                    *gpu);

    if(!sol)
    {
        std::cerr << "AMaxWithScale: No valid solution found!" << std::endl;
        return HIPBLAS_STATUS_SUCCESS;
    }

    const auto kernelName = sol->name();
    err                   = adapter->initKernel(kernelName);

    Tensile::KernelInvocation invocation;
    invocation.kernelName      = kernelName;
    invocation.codeObjectFile  = sol->getCodeObjectPath();
    invocation.workGroupSize.x = sol->getNumWorkitems();
    invocation.workGroupSize.y = 1;
    invocation.workGroupSize.z = 1;
    invocation.numWorkGroups.x = 1;
    invocation.numWorkGroups.y = 1;
    invocation.numWorkGroups.z = 1;
    invocation.numWorkItems.x  = sol->getNumWorkitems();
    invocation.numWorkItems.y  = 1;
    invocation.numWorkItems.z  = 1;
    invocation.sharedMemBytes  = 32 * sizeof(float);
    invocation.args            = Tensile::KernelArguments(false);
    invocation.args.reserve(20, 3);
    invocation.args.append("output", output);
    invocation.args.append("outputD", outputD);
    invocation.args.append("input", input);
    invocation.args.append("inputScae", inputScale);
    invocation.args.append("length", len);

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if(err)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}
