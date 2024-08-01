/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

// #include <Tensile/Contractions.hpp>
// #include <Tensile/EmbeddedLibrary.hpp>
// #include <Tensile/MasterSolutionLibrary.hpp>
// #include <Tensile/Tensile.hpp>
// #include <Tensile/hip/HipHardware.hpp>
// #include <Tensile/hip/HipSolutionAdapter.hpp>
// #include <Tensile/hip/HipUtils.hpp>
// #include "Utility.hpp"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/program_options.hpp>

#include <cstddef>
#include <fstream>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "bfloat16.h"
#include "gfx950_f4.h"
#include "gfx950_f6.h"
#include "gfx950_f8.h"
#include "hipblaslt_float8.h"

#define HIP_CHECK_EXC(expr)                                                                       \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (expr);                                                                    \
        if(e)                                                                                     \
        {                                                                                         \
            const char*        errName = hipGetErrorName(e);                                      \
            const char*        errMsg  = hipGetErrorString(e);                                    \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl                                                             \
                << errMsg << std::endl;                                                           \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

#define HIP_CHECK_EXC_MESSAGE(expr, message)                                                      \
    do                                                                                            \
    {                                                                                             \
        hipError_t e = (expr);                                                                    \
        if(e)                                                                                     \
        {                                                                                         \
            const char*        errName = hipGetErrorName(e);                                      \
            const char*        errMsg  = hipGetErrorString(e);                                    \
            std::ostringstream msg;                                                               \
            msg << "Error " << e << "(" << errName << ") " << __FILE__ << ":" << __LINE__ << ": " \
                << std::endl                                                                      \
                << #expr << std::endl                                                             \
                << errMsg << std::endl                                                            \
                << (message) << std::endl;                                                        \
            throw std::runtime_error(msg.str());                                                  \
        }                                                                                         \
    } while(0)

#define HIP_CHECK_RETURN(expr) \
    do                         \
    {                          \
        hipError_t e = (expr); \
        if(e)                  \
            return e;          \
    } while(0)

#define HIP_CHECK_PRINT(expr)                             \
    {                                                     \
        hipError_t e = (expr);                            \
        if(e)                                             \
            std::cout << "Error code " << e << std::endl; \
    }

// randomly pick a value from [begin, last]
static int random_int(int begin, int last)
{
    if(begin == last)
        return last;

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distr(begin, last); // define the range

    int picked = distr(gen);
    return picked;
}

// randomly small float [0.1, 0.2, 0.3,...., 1.0]
static float random_small_float()
{
    return (float)random_int(1, 10) / 10.0f;
}

// Simple RAII class for GPU buffers.  T is the type of pointer that
// data() returns
template <class T = void>
class gpubuf_t
{
public:
    gpubuf_t() {}
    // buffers are movable but not copyable
    gpubuf_t(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(bsize, other.bsize);
    }
    gpubuf_t& operator=(gpubuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(bsize, other.bsize);
        return *this;
    }
    gpubuf_t(const gpubuf_t&)            = delete;
    gpubuf_t& operator=(const gpubuf_t&) = delete;

    ~gpubuf_t()
    {
        free();
    }

    hipError_t alloc(const size_t size)
    {
        bsize = size;
        free();
        auto ret = hipMalloc(&buf, bsize);
        if(ret != hipSuccess)
        {
            buf   = nullptr;
            bsize = 0;
        }
        return ret;
    }

    hipError_t set_zero()
    {
        if(bsize > 0)
        {
            auto ret = hipMemset(buf, 0, bsize);
            return ret;
        }

        return hipSuccess;
    }

    size_t size() const
    {
        return bsize;
    }

    void free()
    {
        if(buf != nullptr)
        {
            (void)hipFree(buf);
            buf = nullptr;
        }
    }

    T* data() const
    {
        return static_cast<T*>(buf);
    }

    // equality/bool tests
    bool operator==(std::nullptr_t n) const
    {
        return buf == n;
    }
    bool operator!=(std::nullptr_t n) const
    {
        return buf != n;
    }
    operator bool() const
    {
        return buf;
    }

private:
    // The GPU buffer
    void*  buf   = nullptr;
    size_t bsize = 0;
};

// copy a host vector to the device
template <typename T>
gpubuf_t<> host_vec_to_dev(const std::vector<T>& hvec)
{
    gpubuf_t<> ret;
    if(ret.alloc(sizeof(T) * hvec.size()) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemcpy(ret.data(), hvec.data(), sizeof(T) * hvec.size(), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to memcpy");
    return ret;
}

template <typename T>
class KernelArgumentsContainer
{
public:
    void setPointer(void* pointer, size_t size)
    {
        m_data     = (T*)pointer;
        m_dataSize = size;
    }

    void reserve(size_t maxSize)
    {
        m_maxSize = maxSize;
        if(!m_data)
        {
            m_vec_data.reserve(maxSize);
        }
    }

    void insert(size_t startPos, size_t size, T value)
    {
        if(!m_data)
        {
            m_vec_data.insert(m_vec_data.end(), size, value);
            m_currentLocation = m_vec_data.size();
            return;
        }
        else if(startPos + size < m_dataSize)
        {
            // We don't insert 0 here because we'll copy data later.
            // Adding this API is to compatible with vector insert.
            // for(size_t i = startPos; i < startPos + size; i++)
            // {
            //     m_data[i] = value;
            // }
            m_currentLocation += size;
        }
    }

    size_t size() const
    {
        return m_currentLocation;
    }

    size_t end() const
    {
        return m_currentLocation;
    }

    const uint8_t* data() const
    {
        if(!m_data)
        {
            return m_vec_data.data();
        }
        return (const uint8_t*)m_data;
    }

    uint8_t* rawdata()
    {
        if(!m_data)
        {
            T* ptr = m_vec_data.data();
            return ptr;
        }
        return (uint8_t*)m_data;
    }

    const T& operator[](unsigned int i) const
    {
        if(!m_data)
        {
            return m_vec_data[i];
        }
        return m_data[i];
    }

    T& operator[](unsigned int i)
    {
        if(!m_data)
        {
            return m_vec_data[i];
        }
        return m_data[i];
    }

private:
    size_t         m_maxSize         = 0;
    size_t         m_currentLocation = 0;
    T*             m_data            = nullptr;
    size_t         m_dataSize;
    std::vector<T> m_vec_data;
};

class KernelArguments
{
public:
    KernelArguments(bool log = true);
    virtual ~KernelArguments();

    void reserve(size_t bytes, size_t count);

    template <typename T>
    void append(std::string const& name, T value);

    void const* data() const;
    uint8_t*    rawdata();
    size_t      size() const;

private:
    template <typename T>
    void append(std::string const& name, T value, bool bound);

    template <typename T>
    std::string stringForValue(T value, bool bound);

    // void appendRecord(std::string const& name, Arg info);

    template <typename T>
    void writeValue(size_t offset, T value);

    KernelArgumentsContainer<uint8_t> m_data;

    std::vector<std::string> m_names;

    bool m_log;
};

template <typename T>
inline void KernelArguments::append(std::string const& name, T value)
{
    append(name, value, true);
}

template <typename T>
inline std::string KernelArguments::stringForValue(T value, bool bound)
{
    if(!m_log)
        return "";

    if(!bound)
        return "<unbound>";

    using castType = std::conditional_t<std::is_pointer<T>::value, void const*, T>;

    std::ostringstream msg;
    msg << static_cast<castType>(value);
    return msg.str();
}

template <typename T>
inline void KernelArguments::append(std::string const& name, T value, bool bound)
{
    size_t offset = m_data.size();
    size_t size   = sizeof(T);

    m_data.insert(m_data.end(), sizeof(value), 0);
    writeValue(offset, value);
}

template <typename T>
inline void KernelArguments::writeValue(size_t offset, T value)
{
    if(offset + sizeof(T) > m_data.size())
    {
        throw std::runtime_error("Value exceeds allocated bounds.");
    }

    std::memcpy(&m_data[offset], &value, sizeof(T));
}

KernelArguments::KernelArguments(bool log)
    : m_log(log)
{
}

KernelArguments::~KernelArguments() {}

void KernelArguments::reserve(size_t bytes, size_t count)
{
    m_data.reserve(bytes);
    m_names.reserve(count);
}

void const* KernelArguments::data() const
{
    return reinterpret_cast<void const*>(m_data.data());
}

uint8_t* KernelArguments::rawdata()
{
    return m_data.rawdata();
}

size_t KernelArguments::size() const
{
    return m_data.size();
}

struct KernelInvocation
{
public:
    std::string kernelName;
    std::string codeObjectFile; //Code object file kernel is located in

    dim3   blockDim; // threads in a block
    dim3   gridDim; // num of block
    dim3   totalItemDim; // total threads numbers
    size_t sharedMemBytes = 0;

    KernelArguments args;
};

class SolutionAdapter
{
public:
    SolutionAdapter();
    SolutionAdapter(bool debug);
    SolutionAdapter(bool debug, std::string const& name);
    ~SolutionAdapter();

    virtual std::string name() const
    {
        return m_name;
    }

    hipError_t loadCodeObjectFile(std::string const& path);

    hipError_t launchKernel(KernelInvocation const& kernel);
    hipError_t launchKernel(KernelInvocation const& kernel,
                            hipStream_t             stream,
                            hipEvent_t              startEvent,
                            hipEvent_t              stopEvent);

    hipError_t initKernel(std::string const& name);

private:
    hipError_t getKernel(hipFunction_t& rv, std::string const& name);

    std::mutex m_access;

    std::vector<hipModule_t>                       m_modules;
    std::unordered_map<std::string, hipFunction_t> m_kernels;
    bool                                           m_debug           = false;
    bool                                           m_debugSkipLaunch = false;
    std::string                                    m_name            = "HipSolutionAdapter";
    std::string                                    m_codeObjectDirectory;

    std::vector<std::string>        m_loadedModuleNames;
    std::unordered_set<std::string> m_loadedCOFiles;
};

SolutionAdapter::SolutionAdapter() {}

SolutionAdapter::SolutionAdapter(bool debug)
    : m_debug(debug)
{
}

SolutionAdapter::SolutionAdapter(bool debug, std::string const& name)
    : m_debug(debug)
    , m_name(name)
{
}

SolutionAdapter::~SolutionAdapter()
{
    for(auto module : m_modules)
        HIP_CHECK_PRINT(hipModuleUnload(module));
}

std::string removeXnack(std::string coFilename)
{
    std::string xnackVersion = "xnack"; //Extra character before and after xnack
    size_t      loc          = coFilename.find(xnackVersion);
    if(loc != std::string::npos)
        coFilename.replace(loc - 1, xnackVersion.length() + 2, "");

    return coFilename;
}

hipError_t SolutionAdapter::loadCodeObjectFile(std::string const& path)
{
    hipModule_t module;

    HIP_CHECK_RETURN(hipModuleLoad(&module, path.c_str()));

    if(m_debug)
        std::cout << "loaded code object " << path << std::endl;

    {
        std::lock_guard<std::mutex> guard(m_access);
        m_modules.push_back(module);
        m_loadedModuleNames.push_back("File " + path);

        //Isolate filename
        size_t start = path.rfind('/');
        start        = (start == std::string::npos) ? 0 : start + 1;
        m_loadedCOFiles.insert(removeXnack(std::string(path.begin() + start, path.end())));
    }
    return hipSuccess;
}

hipError_t SolutionAdapter::initKernel(std::string const& name)
{
    hipFunction_t function;
    return getKernel(function, name);
}

hipError_t SolutionAdapter::getKernel(hipFunction_t& rv, std::string const& name)
{
    std::unique_lock<std::mutex> guard(m_access);
    hipError_t                   err = hipErrorNotFound;

    auto it = m_kernels.find(name);
    if(it != m_kernels.end())
    {
        rv = it->second;
        return hipSuccess;
    }

    for(auto module : m_modules)
    {
        err = hipModuleGetFunction(&rv, module, name.c_str());

        if(err == hipSuccess)
        {
            m_kernels[name] = rv;
            return err;
        }
        else if(err != hipErrorNotFound)
        {
            return err;
        }
    }

    return err;
}

hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel)
{
    return launchKernel(kernel, nullptr, nullptr, nullptr);
}

hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel,
                                         hipStream_t             stream,
                                         hipEvent_t              startEvent,
                                         hipEvent_t              stopEvent)
{
    if(!kernel.codeObjectFile.empty())
    {
        //If required code object file hasn't yet been loaded, load it now
        m_access.lock();
        bool loaded
            = m_loadedCOFiles.find(removeXnack(kernel.codeObjectFile)) != m_loadedCOFiles.end();
        std::string codeObjectDir = m_codeObjectDirectory;
        m_access.unlock();

        if(!loaded)
        {
            //Try other xnack versions
            size_t     loc = kernel.codeObjectFile.rfind('.');
            hipError_t err;

            for(auto ver : {"", "-xnack-", "-xnack+"})
            {
                std::string modifiedCOName = kernel.codeObjectFile;
                modifiedCOName.insert(loc, ver);
                err = loadCodeObjectFile(codeObjectDir + modifiedCOName);

                if(err == hipSuccess)
                    break;
            }
        }
    }

    if(m_debug)
    {
        std::cout << "Kernel " << kernel.kernelName << std::endl;
        std::cout << " l[" << kernel.blockDim.x << "," << kernel.blockDim.y << ","
                  << kernel.blockDim.z << "] x g[" << kernel.gridDim.x << "," << kernel.gridDim.y
                  << "," << kernel.gridDim.z << "] = [" << kernel.totalItemDim.x << ","
                  << kernel.totalItemDim.y << "," << kernel.totalItemDim.z << "]" << std::endl;
    }
    if(m_debugSkipLaunch)
    {
        std::cout << "DEBUG: Skip kernel execution" << std::endl;
        if(startEvent != nullptr)
            HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));
        if(stopEvent != nullptr)
            HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
        return hipSuccess;
    }

    hipFunction_t function;
    HIP_CHECK_RETURN(getKernel(function, kernel.kernelName));

    void*  kernelArgs = const_cast<void*>(kernel.args.data());
    size_t argsSize   = kernel.args.size();

    void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               kernelArgs,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &argsSize,
                               HIP_LAUNCH_PARAM_END};

    if(startEvent != nullptr)
        HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));
    HIP_CHECK_RETURN(hipExtModuleLaunchKernel(function,
                                              kernel.totalItemDim.x,
                                              kernel.totalItemDim.y,
                                              kernel.totalItemDim.z,
                                              kernel.blockDim.x,
                                              kernel.blockDim.y,
                                              kernel.blockDim.z,
                                              kernel.sharedMemBytes, // sharedMem
                                              stream, // stream
                                              nullptr,
                                              (void**)&hipLaunchParams,
                                              nullptr, // event
                                              nullptr // event
                                              ));
    if(stopEvent != nullptr)
        HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
    return hipSuccess;
}

namespace po = boost::program_options;

template <typename T>
po::typed_value<T>* value_default(std::string const& desc)
{
    return po::value<T>()->default_value(T(), desc);
}

template <typename T>
po::typed_value<T>* value_default()
{
    return po::value<T>()->default_value(T());
}

template <typename T>
po::typed_value<std::vector<T>>* vector_default_empty()
{
    return value_default<std::vector<T>>("[]");
}

po::options_description all_options()
{
    po::options_description options("Tensile client options");

    // clang-format off
            options.add_options()
                ("help,h", "Show help message.")

                ("config-file",              vector_default_empty<std::string>(), "INI config file(s) to read.")

                ("code-object,c",            vector_default_empty<std::string>(), "Code object file with kernel(s).  If none are "
                                                                                  "specified, we will use the embedded code "
                                                                                  "object(s) if available.")

                ("asm-file",                 po::value<std::string>(), ".s filename to compile and run, not used in cpp, used in python")

                ("kernelname",               po::value<std::string>(), "kernel name in the asm mete")

                ("workgroup-size",           vector_default_empty<std::string>(), "Num-Threads in a block.  Comma-separated list of "
                                                                                   "sizes, in the order of the Einstein notation.")

                ("num-workgroups",           vector_default_empty<std::string>(), "How man blocks launched.  Comma-separated list of "
                                                                                   "sizes, in the order of the Einstein notation.")

                ("input-bytes",              po::value<size_t>()->default_value(0), "bytes of input (only for LDS_TEST)")

                ("ext-lds-bytes",            po::value<size_t>()->default_value(0), "extern lds sizes (Should be 0 if the asm has hardcoded lds-usage)")

                ("matB_N",                   po::value<size_t>()->default_value(0), "size of matrix B: N direction (row)")

                ("matB_K",                   po::value<size_t>()->default_value(0), "size of matrix B: K direction (colume)")

                ("gemm_M",                   po::value<uint32_t>()->default_value(0), "Dim M for gemm")

                ("gemm_N",                   po::value<uint32_t>()->default_value(0), "Dim N for gemm")

                ("gemm_K",                   po::value<uint32_t>()->default_value(0), "Dim K for gemm")

                ("alpha",                    po::value<float>()->default_value(1.0f), "Alpha")

                ("beta",                     po::value<float>()->default_value(0.0f), "Beta")

                ("test-tag",                 po::value<std::string>(), "test-tag to find kernel arg and validation funcs")
                ;
    // clang-format on

    return options;
}

int GetHardware(po::variables_map const& args)
{
    int deviceCount = 0;
    HIP_CHECK_EXC(hipGetDeviceCount(&deviceCount));

    int deviceIdx = 0;

    if(deviceIdx >= deviceCount)
        throw std::runtime_error("Invalid device index " + std::to_string(deviceIdx) + " ("
                                 + std::to_string(deviceCount) + " total found.)");

    HIP_CHECK_EXC(hipSetDevice(deviceIdx));
    return deviceIdx;
}

hipStream_t GetStream(po::variables_map const& args)
{
    if(true)
        return 0;

    hipStream_t stream;
    HIP_CHECK_EXC(hipStreamCreate(&stream));
    return stream;
}

void LoadCodeObjects(po::variables_map const& args, SolutionAdapter& adapter)
{
    auto const& filenames = args["code-object"].as<std::vector<std::string>>();

    if(filenames.empty())
    {
        throw;
    }
    else
    {
        //only trigger exception when failed to load all code objects.
        bool       loaded   = false;
        hipError_t retError = hipSuccess;

        for(auto const& filename : filenames)
        {
            hipError_t ret;

            std::cout << "Loading " << filename << std::endl;
            ret = adapter.loadCodeObjectFile(filename);

            if(ret == hipSuccess)
                loaded = true;
            else
                retError = ret;
        }

        if(!loaded)
            HIP_CHECK_EXC(retError);
    }
}

template <typename T>
std::vector<T> split_nums(std::string const& value)
{
    std::vector<std::string> parts;
    boost::split(parts, value, boost::algorithm::is_any_of(",;"));

    std::vector<T> rv;
    rv.reserve(parts.size());

    for(auto const& part : parts)
        if(part != "")
            rv.push_back(boost::lexical_cast<T>(part));

    return rv;
}

template <typename T>
void parse_arg_nums(po::variables_map& args, std::string const& name)
{
    auto inValue = args[name].as<std::vector<std::string>>();

    std::vector<std::vector<T>> outValue;
    outValue.reserve(inValue.size());
    for(auto const& str : inValue)
        outValue.push_back(split_nums<T>(str));

    boost::any v(outValue);

    args.at(name).value() = v;
}

void parse_arg_ints(po::variables_map& args, std::string const& name)
{
    parse_arg_nums<size_t>(args, name);
}

void parse_arg_double(po::variables_map& args, std::string const& name)
{
    parse_arg_nums<double>(args, name);
}

po::variables_map parse_args(int argc, const char* argv[])
{
    auto options = all_options();

    po::variables_map args;
    po::store(po::parse_command_line(argc, argv, options), args);
    po::notify(args);

    if(args.count("help"))
    {
        std::cout << options << std::endl;
        exit(1);
    }

    if(args.count("config-file"))
    {
        auto configFiles = args["config-file"].as<std::vector<std::string>>();
        for(auto filename : configFiles)
        {
            std::cout << "loading config file " << filename << std::endl;
            std::ifstream file(filename.c_str());
            if(file.bad())
                throw std::runtime_error("Could not open " + filename);
            po::store(po::parse_config_file(file, options), args);
        }
    }

    parse_arg_ints(args, "workgroup-size");
    parse_arg_ints(args, "num-workgroups");

    return args;
}

class AsmRunnerAndValidator
{
public:
    AsmRunnerAndValidator(){};
    virtual ~AsmRunnerAndValidator(){};
    virtual void
        LaunchKernel(SolutionAdapter& adapter, KernelInvocation& kernelInvoc, hipStream_t stream)
    {
        HIP_CHECK_EXC(adapter.launchKernel(kernelInvoc, stream, nullptr, nullptr));
    }

    virtual void SetupKernelArgs(po::variables_map& args, KernelInvocation& kernelInvoc) = 0;
    virtual bool Validation()                                                            = 0;
};

class AMAXRunner : public AsmRunnerAndValidator
{
private:
    std::vector<float> inputH;
    std::vector<float> outputH;
    gpubuf_t<float>    inputD;
    gpubuf_t<float>    outputD;

    void cpuAMax(float* out, float* in, size_t length)
    {
        // calculate amax
        float m = 0;
        for(int j = 0; j < length; j++)
        {
            m = max(m, abs(in[j]));
        }
        out[0] = m;
    }

public:
    AMAXRunner()
        : AsmRunnerAndValidator()
    {
        inputH  = std::vector<float>(256, 0.0f);
        outputH = std::vector<float>(1);

        for(float& elem : inputH)
            elem = static_cast<float>((rand() % 201) - 100);

        HIP_CHECK_EXC(inputD.alloc(sizeof(float) * inputH.size()));
        HIP_CHECK_EXC(outputD.alloc(sizeof(float)));
    }

    virtual void SetupKernelArgs(po::variables_map& args, KernelInvocation& kernelInvoc) override
    {
        HIP_CHECK_EXC(hipMemcpy(
            inputD.data(), inputH.data(), sizeof(float) * inputH.size(), hipMemcpyHostToDevice));

        KernelArguments& kernelArg = kernelInvoc.args;
        kernelArg.append("output", (void*)(outputD.data()));
        kernelArg.append("input", (void*)(inputD.data()));
        kernelArg.append("length", 256);
    }

    virtual bool Validation() override
    {
        std::cout << std::endl << "Validation:" << std::endl;

        // fetch result from gpu, notice that for gpubuf_t .size() mean the bytes, not vector length
        std::vector<float> gpuOutput(1);
        HIP_CHECK_EXC(
            hipMemcpy(gpuOutput.data(), outputD.data(), outputD.size(), hipMemcpyDeviceToHost));

        // CPU reference result
        cpuAMax(outputH.data(), inputH.data(), 256);

        // compare
        std::cout << "AMAX from kernel:" << gpuOutput[0] << ", AMAX from cpu: " << outputH[0]
                  << std::endl;
        return gpuOutput[0] == outputH[0];
    }
};

class FastAMAXRunner : public AsmRunnerAndValidator
{
private:
    std::vector<_Float16> inputH;
    std::vector<float>    outputH;

    gpubuf_t<_Float16> inputD;
    gpubuf_t<float>    outputD;
    gpubuf_t<_Float16> workspaceD;
    gpubuf_t<uint32_t> syncD;

    uint32_t total_elems;

    void cpuAMax(float* out, _Float16* in, size_t length)
    {
        // calculate amax
        float m = 0;
        for(int j = 0; j < length; j++)
        {
            m = max(m, abs((float)in[j]));
        }
        out[0] = m;
    }

    template <typename T>
    bool print(const std::vector<T>& gpuOutput, uint32_t numRow, uint32_t numCol)
    {
        for(int i = 0; i < numRow; i++)
        {
            for(int j = 0; j < numCol; j++)
            {
                auto  id    = i + j * numRow;
                float value = (float)(gpuOutput[id]);
                std::cout << value << ", ";
            }
            std::cout << std::endl;
        }
        return true;
    }

public:
    FastAMAXRunner()
        : AsmRunnerAndValidator()
    {
    }

    virtual void SetupKernelArgs(po::variables_map& args, KernelInvocation& kernelInvoc) override
    {
        size_t N    = args["matB_N"].as<size_t>();
        size_t K    = args["matB_K"].as<size_t>();
        total_elems = N * K;
        inputH      = std::vector<_Float16>(total_elems);
        outputH     = std::vector<float>(1);

        for(_Float16& elem : inputH)
            elem = static_cast<_Float16>((double)rand() / RAND_MAX);
        inputH.back() = 65280.0; // make last one be the largest....

        HIP_CHECK_EXC(inputD.alloc(sizeof(_Float16) * inputH.size()));
        HIP_CHECK_EXC(hipMemcpy(
            inputD.data(), inputH.data(), sizeof(_Float16) * inputH.size(), hipMemcpyHostToDevice));

        HIP_CHECK_EXC(outputD.alloc(sizeof(float)));
        HIP_CHECK_EXC(workspaceD.alloc(sizeof(_Float16) * 4096));
        HIP_CHECK_EXC(syncD.alloc(sizeof(uint32_t)));
        HIP_CHECK_EXC(workspaceD.set_zero());
        HIP_CHECK_EXC(syncD.set_zero());

        // based on benchmarks
        int workSize;
        int amax_gsu = 128;
        if(total_elems <= 32768)
        {
            workSize = max(total_elems, 16384);
            amax_gsu = 1;
        }
        else if(total_elems <= 1048576)
            workSize = 16384;
        else if(amax_gsu <= 134217728)
            workSize = 32768;
        else
            workSize = 65536;

        int numGroups = amax_gsu;
        numGroups     = min(int((total_elems + workSize - 1) / workSize), int(numGroups));

        kernelInvoc.gridDim      = dim3(numGroups, 1, 1); // num oF WG
        kernelInvoc.totalItemDim = dim3(kernelInvoc.blockDim.x * kernelInvoc.gridDim.x,
                                        kernelInvoc.blockDim.y * kernelInvoc.gridDim.y,
                                        kernelInvoc.blockDim.z * kernelInvoc.gridDim.z);

        kernelInvoc.args           = KernelArguments(false);
        KernelArguments& kernelArg = kernelInvoc.args;
        kernelArg.reserve(64, 9);
        kernelArg.append("output", outputD.data());
        kernelArg.append("input", inputD.data());
        kernelArg.append("workSpace", workspaceD.data());
        kernelArg.append("sync", syncD.data());
        kernelArg.append("length", total_elems);
        kernelArg.append("is_div", 0);
        kernelArg.append("div", 0);
        kernelArg.append("workSize", workSize);
        kernelArg.append("numGroups", numGroups);
    }

    virtual bool Validation() override
    {
        std::cout << std::endl << "Validation:" << std::endl;

        // fetch result from gpu, notice that for gpubuf_t .size() mean the bytes, not vector length
        std::vector<float> gpuOutput(1);
        HIP_CHECK_EXC(
            hipMemcpy(gpuOutput.data(), outputD.data(), outputD.size(), hipMemcpyDeviceToHost));

        // CPU reference result
        cpuAMax(outputH.data(), inputH.data(), total_elems);

        std::vector<_Float16> worksapce(4096);
        uint32_t              sync;
        HIP_CHECK_EXC(hipMemcpy(
            worksapce.data(), workspaceD.data(), workspaceD.size(), hipMemcpyDeviceToHost));
        HIP_CHECK_EXC(hipMemcpy(&sync, syncD.data(), syncD.size(), hipMemcpyDeviceToHost));
        print(worksapce, 64, 64);
        std::cout << "sync: " << sync << std::endl;

        // compare
        std::cout << "AMAX from kernel:" << gpuOutput[0] << ", AMAX from cpu: " << outputH[0]
                  << std::endl;
        return gpuOutput[0] == outputH[0];
    }
};

class GemmAmaxDRunner : public AsmRunnerAndValidator
{
private:
    std::vector<hipblaslt_f8_fnuz> inputA_h;
    std::vector<hipblaslt_f8_fnuz> inputB_h;
    std::vector<hipblaslt_f8_fnuz> inputC_h;
    std::vector<hipblaslt_f8_fnuz> outputD_h;

    gpubuf_t<hipblaslt_f8_fnuz> inputA_d;
    gpubuf_t<hipblaslt_f8_fnuz> inputB_d;
    gpubuf_t<hipblaslt_f8_fnuz> inputC_d;
    gpubuf_t<hipblaslt_f8_fnuz> outputD_d;

    std::vector<float> scales_h; // scaleA,B,C,D
    gpubuf_t<float>    scales_d;

    uint32_t M;
    uint32_t N;
    uint32_t K;
    float    alpha;
    float    beta;

    // amax D
    float              amaxD_h;
    gpubuf_t<float>    amaxD_d;
    gpubuf_t<float>    workspaceD;
    gpubuf_t<uint32_t> syncD;

    // D1 = leading dimension, D2 = the other
    size_t Coord2Idx_ColMaj(uint32_t D1,
                            uint32_t D2,
                            size_t   x,
                            size_t   y) // x = moving along d2, y = moving along d1
    {
        size_t idx = x * D1 + y;
        return idx;
    }

    template <typename T>
    void cpuGEMM_ColMaj(T* A, T* B, T* C, T* D, float& amaxD)
    {
        float sA = scales_h[0];
        float sB = scales_h[1];
        float sC = scales_h[2];
        float sD = scales_h[3];

        amaxD = 0;

        for(auto dx = 0; dx < N; ++dx)
        {
            for(auto dy = 0; dy < M; ++dy)
            {
                float d = 0;
                for(auto dk = 0; dk < K; ++dk)
                {
                    auto idx_A = Coord2Idx_ColMaj(M, K, dk, dy);
                    auto idx_B = Coord2Idx_ColMaj(K, N, dx, dk);
                    d += alpha * (sA * A[idx_A]) * (sB * B[idx_B]);
                }
                auto idx_CD = Coord2Idx_ColMaj(M, N, dx, dy);
                d += beta * (sC * C[idx_CD]);

                if(abs(d) > amaxD)
                    amaxD = abs(d);

                D[idx_CD] = (T)(d * sD);
            }
        }
    }

    template <typename T>
    bool print(const std::vector<T>& gpuOutput, uint32_t numRow, uint32_t numCol)
    {
        for(int i = 0; i < numRow; i++)
        {
            for(int j = 0; j < numCol; j++)
            {
                auto  id    = i + j * numRow;
                float value = (float)(gpuOutput[id]);
                std::cout << value << ", ";
            }
            std::cout << std::endl;
        }
        return true;
    }

public:
    GemmAmaxDRunner()
        : AsmRunnerAndValidator()
    {
    }

    virtual void SetupKernelArgs(po::variables_map& args, KernelInvocation& kernelInvoc) override
    {
        M     = args["gemm_M"].as<uint32_t>();
        N     = args["gemm_N"].as<uint32_t>();
        K     = args["gemm_K"].as<uint32_t>();
        alpha = args["alpha"].as<float>();
        beta  = args["beta"].as<float>();

        // init A,B,C
        inputA_h  = std::vector<hipblaslt_f8_fnuz>(M * K, (hipblaslt_f8_fnuz)1.0f);
        inputB_h  = std::vector<hipblaslt_f8_fnuz>(N * K, (hipblaslt_f8_fnuz)1.0f);
        inputC_h  = std::vector<hipblaslt_f8_fnuz>(M * N, (hipblaslt_f8_fnuz)0.0f);
        outputD_h = std::vector<hipblaslt_f8_fnuz>(M * N, (hipblaslt_f8_fnuz)0.0f);

        for(auto& a : inputA_h)
            a = (hipblaslt_f8_fnuz)random_small_float();
        for(auto& b : inputB_h)
            b = (hipblaslt_f8_fnuz)random_small_float();
        for(auto& c : inputC_h)
            c = (hipblaslt_f8_fnuz)random_small_float();

        scales_h = {1.0f, 1.0f, 1.0f, 1.0f};

        HIP_CHECK_EXC(inputA_d.alloc(sizeof(hipblaslt_f8_fnuz) * M * K));
        HIP_CHECK_EXC(inputB_d.alloc(sizeof(hipblaslt_f8_fnuz) * N * K));
        HIP_CHECK_EXC(inputC_d.alloc(sizeof(hipblaslt_f8_fnuz) * M * N));
        HIP_CHECK_EXC(outputD_d.alloc(sizeof(hipblaslt_f8_fnuz) * M * N));
        HIP_CHECK_EXC(scales_d.alloc(sizeof(float) * 4));

        HIP_CHECK_EXC(amaxD_d.alloc(sizeof(float)));
        HIP_CHECK_EXC(workspaceD.alloc(sizeof(float) * 4096));
        HIP_CHECK_EXC(workspaceD.set_zero());
        HIP_CHECK_EXC(syncD.alloc(sizeof(uint32_t)));
        HIP_CHECK_EXC(syncD.set_zero());

        // copy to device
        HIP_CHECK_EXC(hipMemcpy(inputA_d.data(),
                                inputA_h.data(),
                                sizeof(hipblaslt_f8_fnuz) * inputA_h.size(),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(inputB_d.data(),
                                inputB_h.data(),
                                sizeof(hipblaslt_f8_fnuz) * inputB_h.size(),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(inputC_d.data(),
                                inputC_h.data(),
                                sizeof(hipblaslt_f8_fnuz) * inputC_h.size(),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(scales_d.data(),
                                scales_h.data(),
                                sizeof(float) * scales_h.size(),
                                hipMemcpyHostToDevice));

        kernelInvoc.args           = KernelArguments(false);
        KernelArguments& kernelArg = kernelInvoc.args;

        kernelArg.reserve(1024, 128);
        kernelArg.append("Gemm-info", (uint32_t)1);
        kernelArg.append("kernel-info0", (uint32_t)52428801);
        kernelArg.append("kernel-info1", (uint32_t)8);
        kernelArg.append("numWG", (uint32_t)(kernelInvoc.gridDim.x));
        kernelArg.append("SizesFree0", M); // M
        kernelArg.append("SizesFree1", N); // N
        kernelArg.append("SizesFree2", (uint32_t)1); // B
        kernelArg.append("SizesSum0", K); // K

        kernelArg.append("D", outputD_d.data());
        kernelArg.append("C", inputC_d.data());
        kernelArg.append("A", inputA_d.data());
        kernelArg.append("B", inputB_d.data());

        kernelArg.append("strideD0", M);
        kernelArg.append("strideD1", (M * N));
        kernelArg.append("strideC0", M);
        kernelArg.append("strideC1", (M * N));
        kernelArg.append("strideA0", M);
        kernelArg.append("strideA1", (M * K));
        kernelArg.append("strideB0", K);
        kernelArg.append("strideB1", (K * N));

        kernelArg.append("alpha", alpha);
        kernelArg.append("beta", beta);

        kernelArg.append("AddressScaleA", &(scales_d.data()[0]));
        kernelArg.append("AddressScaleB", &(scales_d.data()[1]));
        kernelArg.append("AddressScaleC", &(scales_d.data()[2]));
        kernelArg.append("AddressScaleD", &(scales_d.data()[3]));

        kernelArg.append("AmaxDOutput", amaxD_d.data());
        kernelArg.append("Workspace", workspaceD.data());
        kernelArg.append("Sync", syncD.data());
        // kernelArg.append("NumGroup", (uint32_t)(kernelInvoc.gridDim.x));
    }

    template <typename T>
    bool compare(std::vector<T>& gpuOutput, std::vector<T>& ref, float amaxD_gpu, float amaxD_ref)
    {
        float maxErr = 0.0;
        for(int i = 0; i < ref.size(); i++)
        {
            // std::cout << "kernel out : " << gpuOutput[i] << ", ref : " << ref[i] << std::endl;
            float refV = (float)(ref[i]);
            float gpuV = (float)(gpuOutput[i]);
            float err  = refV - gpuV;
            maxErr     = max(maxErr, abs(err));
        }

        std::cout << "max error : " << maxErr << std::endl;
        std::cout << "amaxD kernel out: " << amaxD_gpu << ", ref : " << amaxD_ref << std::endl;
        return (maxErr < 0.125f) && (amaxD_gpu == amaxD_ref);
    }

    virtual bool Validation() override
    {
        std::cout << std::endl << "Validation:" << std::endl;

        // CPU gemm
        cpuGEMM_ColMaj(
            inputA_h.data(), inputB_h.data(), inputC_h.data(), outputD_h.data(), amaxD_h);
        // print(outputD_h, M, N);
        std::cout << "amaxD_h: " << amaxD_h << std::endl;

        // compare
        std::vector<hipblaslt_f8_fnuz> gpuOutput(M * N);
        float                          amaxD_gpuOut;
        HIP_CHECK_EXC(
            hipMemcpy(gpuOutput.data(), outputD_d.data(), outputD_d.size(), hipMemcpyDeviceToHost));
        HIP_CHECK_EXC(
            hipMemcpy(&amaxD_gpuOut, amaxD_d.data(), amaxD_d.size(), hipMemcpyDeviceToHost));
        std::cout << "amaxD_gpu: " << amaxD_gpuOut << std::endl;
        // print(gpuOutput, M, N);

        std::vector<float> worksapce(4096);
        uint32_t           sync;
        HIP_CHECK_EXC(hipMemcpy(
            worksapce.data(), workspaceD.data(), workspaceD.size(), hipMemcpyDeviceToHost));
        HIP_CHECK_EXC(hipMemcpy(&sync, syncD.data(), syncD.size(), hipMemcpyDeviceToHost));
        // print(worksapce, 64, 64);
        // std::cout << "sync: " << sync << std::endl;

        return compare(gpuOutput, outputD_h, amaxD_gpuOut, amaxD_h);

        // // compare
        // std::cout << "AMAX from kernel:" << gpuOutput[0] << ", AMAX from cpu: " << outputH[0]
        //           << std::endl;
        // return gpuOutput[0] == outputH[0];
    }
};

AsmRunnerAndValidator* CreateTypedRunner(const std::string& test_tag)
{
    if(test_tag == "amax")
        return new AMAXRunner();
    else if(test_tag == "fastAmax")
        return new FastAMAXRunner();
    else if(test_tag == "gemm_amaxD")
        return new GemmAmaxDRunner();
    else
    {
        std::cout << "haven't implemented the SetupKernelArgs for test-tag:" << test_tag
                  << std::endl;
        return nullptr;
    }
}

int main(int argc, const char* argv[])
{
    auto args = parse_args(argc, argv);

    // Set srand
    // unsigned int seed = args["init-seed"].as<unsigned int>();
    // if(seed == 0)
    // {
    //     seed = time(NULL);
    // }

    // unsigned int seed = time(NULL);
    unsigned int seed = 2567;
    std::cout << std::endl << "srand seed is set to " << seed << std::endl << std::endl;
    srand(seed);

    auto        deviceID = GetHardware(args);
    hipStream_t stream   = GetStream(args);

    SolutionAdapter adapter(true);
    LoadCodeObjects(args, adapter);

    std::vector<size_t> wgs = args["workgroup-size"].as<std::vector<std::vector<size_t>>>().front();
    std::vector<size_t> numWGS
        = args["num-workgroups"].as<std::vector<std::vector<size_t>>>().front();

    KernelInvocation kernelInvoc;

    // launch param
    kernelInvoc.kernelName   = args["kernelname"].as<std::string>();
    kernelInvoc.blockDim     = dim3(wgs[0], wgs[1], wgs[2]); // size of a WK (threads)
    kernelInvoc.gridDim      = dim3(numWGS[0], numWGS[1], numWGS[2]); // num oF WG
    kernelInvoc.totalItemDim = dim3(wgs[0] * numWGS[0], wgs[1] * numWGS[1], wgs[2] * numWGS[2]);
    kernelInvoc.sharedMemBytes
        = args["ext-lds-bytes"].as<size_t>(); // should be zero for hand-written assembly

    // branch to different test cases
    std::string test_tag = args["test-tag"].as<std::string>();

    auto runnerClass = CreateTypedRunner(test_tag);
    if(!runnerClass)
    {
        std::cout << "haven't implemented the SetupKernelArgs for test-tag:" << test_tag
                  << std::endl;
        return 1;
    }

    // kernal arguments
    runnerClass->SetupKernelArgs(args, kernelInvoc);

    // launch kernel
    runnerClass->LaunchKernel(adapter, kernelInvoc, stream);

    // validation
    if(runnerClass->Validation())
        std::cout << "validation succeeded!" << std::endl;
    else
        std::cout << "validation failed!" << std::endl;

    return 0;
}
