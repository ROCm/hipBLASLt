/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <tensile_host.hpp>
#include <rocblaslt-auxiliary.h>
#include <libgen.h>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <unordered_map>

hipblasStatus_t hipblasltSoftmaxRun(hipblasDatatype_t datatype, uint32_t m, uint32_t n, uint32_t dim,
                                    void *output, void *input, hipStream_t stream);

hipblasStatus_t hipblasltExtSoftmax(hipblasDatatype_t datatype, uint32_t m, uint32_t n, uint32_t dim,
    void *output, void *input, hipStream_t stream) {
    return hipblasltSoftmaxRun(datatype, m, n, dim, output, input, stream);
}

namespace {
    constexpr char DEFAULT_EXT_OP_LIBRARY_PATH[] = "/opt/rocm/lib/hipblaslt/library/hipblasltExtOpLibrary.dat";
    constexpr uint32_t SUPPORTED_MAX_N = 256;
    constexpr uint32_t WORKGROUP_SIZE = 256;

    std::string trimArchName(const std::string &archName) {
        auto pos = archName.find(':');

        if (pos != std::string::npos) {
            return archName.substr(0, pos);
        }

        return archName;
    }

    std::string getExtOpLibraryPath() {
        if (auto libPath = std::getenv("HIPBLASLT_EXT_OP_LIBRARY_PATH")) {
            return libPath;
        }

        auto soPath = rocblaslt_internal_get_so_path("hipblaslt");
        std::string libPath(dirname(&soPath[0]));

        if(rocblaslt_internal_test_path(libPath + "/../Tensile/library"))
            libPath += "/../Tensile/library";
        else if(rocblaslt_internal_test_path(libPath + "library"))
            libPath += "/library";
        else
            libPath += "/hipblaslt/library";

        libPath += "/hipblasltExtOpLibrary.dat";

        if (rocblaslt_internal_test_path(libPath)) {
            return libPath;
        }
        
        return DEFAULT_EXT_OP_LIBRARY_PATH;

    }

    inline uint32_t getNumWorkgroups(uint32_t m, uint32_t tileM) {
        return (m / tileM) + !!(m % tileM);
    }

    const std::string kernelFuncName(uint32_t tileM, uint32_t tileN) {
        std::stringstream ss;
        ss << "Softmax_DT_S_MT_" << tileM << "_" << tileN;
        return ss.str();
    }

    inline uint32_t getBestKernelTileN(uint32_t n) {
        const uint32_t exponent = std::ceil(std::log2(n));
        return 1 << exponent;
    }

    inline uint32_t getKernelTileM(uint32_t tileM) {
        return WORKGROUP_SIZE / tileM;
    }

    uint32_t elementNumBytes(hipblasDatatype_t type) {
        assert(type == HIPBLAS_R_32F);

        if (type == HIPBLAS_R_32F) {
            return 4;
        }

        return 1;
    }

    inline uint32_t getLdsUsageByte(hipblasDatatype_t datatype, uint32_t tileM, uint32_t tileN) {
        return elementNumBytes(datatype) * tileM * tileN;
    }

    static const ExtOpMasterLibrary &getExtOpMasterLibrary() {
        static ExtOpMasterLibrary lib(getExtOpLibraryPath());
        return lib;
    }

    static auto extOpLibraries = []() {
        std::vector<std::unique_ptr<Tensile::hip::SolutionAdapter>> adapters;
        int numDevices{};
        auto err = hipGetDeviceCount(&numDevices);

        for (std::size_t i = 0; i < numDevices; ++i) {
            adapters.emplace_back(std::make_unique<Tensile::hip::SolutionAdapter>());
        }

        int currentDevice{};
        err = hipGetDevice(&currentDevice);
        auto &lib = getExtOpMasterLibrary();

        for (auto &adapter : adapters) {
            //setup code object root only, ignore the error
            err = adapter->initializeLazyLoading("", lib.getLibraryFolder());
        }

        return adapters;
    }();
}

hipblasStatus_t hipblasltSoftmaxRun(hipblasDatatype_t datatype, uint32_t m, uint32_t n, uint32_t dim,
                                    void *output, void *input, hipStream_t stream) {
    if (datatype != HIPBLAS_R_32F) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if (dim != 1) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    const auto tileN = getBestKernelTileN(n);
    const auto tileM = getKernelTileM(tileN);

    if (tileN > SUPPORTED_MAX_N) {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int currentDeviceId{};
    auto err = hipGetDevice(&currentDeviceId);
    auto &adapter = extOpLibraries.at(currentDeviceId);
    auto gpu = Tensile::hip::GetCurrentDevice();
    const auto archName = trimArchName(gpu->archName());
    auto &masterLib = getExtOpMasterLibrary();
    const auto &lib = masterLib.getLibrary(archName, SoftmaxSolutionLibrary::opName)->as<SoftmaxSolutionLibrary>();
    auto sol = lib.findBestSolution(SoftmaxProblem(m, n, hipblasDatatype_to_tensile_type(datatype)), *gpu);
    const auto kernelName = sol->name();
    err = adapter->initKernel(kernelName);
    Tensile::KernelArguments kArgs;
    kArgs.append("input", input);
    kArgs.append("output", output);
    kArgs.append("m", m);
    kArgs.append("n", n);
    const auto numWorkgroups = getNumWorkgroups(m, tileM);
    Tensile::KernelInvocation invocation{
        kernelName,
        sol->getCodeObjectPath(),
        {WORKGROUP_SIZE, 1, 1},
        {numWorkgroups, 1, 1},
        {numWorkgroups * WORKGROUP_SIZE, 1, 1},
        getLdsUsageByte(datatype, tileM, tileN),
        kArgs
    };

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if (err) {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}
