/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// The implementation of the rocblaslt<->Tensile interface layer.

#include "rocblaslt.h"

/*****************************************************************************
 * This is the only file in rocblaslt which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

#include "rocblaslt-types.h"
#include "rocblaslt_mat_utils.hpp"
#include "tensile_host.hpp"
//#include "utility.hpp"

//#include <Tensile/AMDGPU.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/EmbeddedLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <atomic>
#include <complex>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <glob.h>
#include <libgen.h>
#include <link.h>
#include <unistd.h>

#define HIPBLASLT_LIB_PATH "/opt/rocm/hipblaslt/lib"

namespace
{
#ifndef WIN32
    std::string hipblaslt_so_path;

    int hipblaslt_dl_iterate_phdr_callback(struct dl_phdr_info* hdr_info, size_t size, void* data)
    {
        // uncomment to see all dependent .so files
        // fprintf(stderr, "hipblaslt so file: %s\n", hdr_info->dlpi_name);
        if(hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, "libhipblaslt"))
        {
            hipblaslt_so_path = hdr_info->dlpi_name;
        }
        return 0;
    }
#endif
    /******************************************************
 * Map a hipblas data type to a corresponding Tensile type *
 ******************************************************/
    inline Tensile::DataType hipblasDatatype_to_tensile_type(hipblasDatatype_t type)
    {
        switch(type)
        {
        case HIPBLAS_R_16F:
            return Tensile::DataType::Half;
        case HIPBLAS_R_32F:
            return Tensile::DataType::Float;
        case HIPBLAS_R_16B:
            return Tensile::DataType::BFloat16;
        default:
            assert(!"hipblasDatatype_to_tensile_type: non-supported type");
            return Tensile::DataType::None;
        }
    }

    /******************************************************
 * Map a rocblaslt type to a corresponding Tensile type *
 ******************************************************/
    template <typename T>
    struct rocblaslt_to_tensile_type
    {
        using tensile_type = T;
    };

    template <>
    struct rocblaslt_to_tensile_type<rocblaslt_half>
    {
        using tensile_type = Tensile::Half;
    };

    template <>
    struct rocblaslt_to_tensile_type<rocblaslt_bfloat16>
    {
        using tensile_type = Tensile::BFloat16;
    };

    /********************************************************************
 * Variable template to map a rocblaslt type into a Tensile::DataType *
 ********************************************************************/
    template <typename>
    constexpr auto tensile_datatype = nullptr;

    template <>
    constexpr auto tensile_datatype<rocblaslt_half> = Tensile::DataType::Half;

    template <>
    constexpr auto tensile_datatype<float> = Tensile::DataType::Float;

    template <>
    constexpr auto tensile_datatype<rocblaslt_bfloat16> = Tensile::DataType::BFloat16;

    /*************************************************************************
 * Class for converting alpha and beta between rocblaslt and Tensile types *
 * By default, alpha and beta are the same type as Tc compute_type       *
 *************************************************************************/
    template <typename Ti, typename To = Ti, typename Tc = To>
    struct AlphaBeta
    {
        using tensile_type = typename rocblaslt_to_tensile_type<Tc>::tensile_type;
        static void copy(tensile_type* dst, const Tc* src)
        {
            static_assert(sizeof(*src) == sizeof(*dst),
                          "Tensile and rocblaslt types are not the same size");
            static_assert(std::is_standard_layout<tensile_type>{} && std::is_standard_layout<Tc>{},
                          "Tensile or rocblaslt types are not standard layout types");
            memcpy(dst, src, sizeof(*dst));
        }
    };

    /****************************************************************
 * Construct a Tensile Problem from a RocblasltContractionProblem *
 ****************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto ConstructTensileProblem(const RocblasltContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile DataTypes corresponding to rocblaslt data types
        static constexpr Tensile::DataType Tensile_Ti = tensile_datatype<Ti>;
        static constexpr Tensile::DataType Tensile_To = tensile_datatype<To>;
        static constexpr Tensile::DataType Tensile_Tc = tensile_datatype<Tc>;

        // Tensor descriptors for a, b
        Tensile::TensorDescriptor a, b;

        // Tensor ops for matrices, like complex conjugate
        Tensile::TensorOps aops, bops, cops, dops;

        // Tensile Indices for contraction problem
        Tensile::ContractionProblem::FreeIndices  freeIndex(2);
        Tensile::ContractionProblem::BoundIndices boundIndex(1);
        Tensile::ContractionProblem::BatchIndices batchIndex{{2, 2, 2, 2}};

        // Set up GEMM indices
        freeIndex[0].isA = true;
        freeIndex[1].isA = false;
        freeIndex[0].c = freeIndex[0].d = 0;
        freeIndex[1].c = freeIndex[1].d = 1;

        // We set K=0 when alpha==0.
        // This makes alpha==0 a change in the problem, and not just a change in the
        // inputs. It optimizes all problems with alpha==0 into K=0 and alpha=(don't
        // care)
        auto k = prob.k && *prob.alpha ? prob.k : 0;

        // clang-format off

        // If A is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_a != HIPBLAS_OP_N)
        {
            a = {
                    Tensile_Ti,
                    {k, prob.m, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a},
                    prob.buffer_offset_a
                };
            freeIndex[0].i  = 1;
            boundIndex[0].a = 0;
        }
        else
        {
            a = {
                    Tensile_Ti,
                    {prob.m, k, prob.batch_count},
                    {prob.row_stride_a, prob.col_stride_a, prob.batch_stride_a},
                    prob.buffer_offset_a
                };
            freeIndex[0].i  = 0;
            boundIndex[0].a = 1;
        }

        // If B is transposed, swap the free and bound dimensions and their ranks
        if(prob.trans_b != HIPBLAS_OP_N)
        {
            b = {
                    Tensile_Ti,
                    {prob.n, k, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b},
                    prob.buffer_offset_b
                };
            freeIndex[1].i  = 0;
            boundIndex[0].b = 1;
        }
        else
        {
            b = {
                    Tensile_Ti,
                    {k, prob.n, prob.batch_count},
                    {prob.row_stride_b, prob.col_stride_b, prob.batch_stride_b},
                    prob.buffer_offset_b
                };
            freeIndex[1].i  = 1;
            boundIndex[0].b = 0;
        }

        // clang-format on

        // Descriptor for input matrix C
        Tensile::TensorDescriptor c{Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c},
                                    prob.buffer_offset_c};

        // Descriptor for output matrix D
        Tensile::TensorDescriptor d{Tensile_To,
                                    {prob.m, prob.n, prob.batch_count},
                                    {prob.row_stride_d, prob.col_stride_d, prob.batch_stride_d},
                                    prob.buffer_offset_d};

        // The ContractionProblem
        Tensile::ContractionProblem tensileProblem{a,
                                                   aops,
                                                   b,
                                                   bops,
                                                   c,
                                                   cops,
                                                   d,
                                                   dops,
                                                   freeIndex,
                                                   batchIndex,
                                                   boundIndex,
                                                   *prob.beta,
                                                   prob.workspaceSize};

        tensileProblem.setAlphaType(Tensile_Tc);
        tensileProblem.setBetaType(Tensile_Tc);

        // HPA is active iff sizeof(compute type) > sizeof(input type)
        tensileProblem.setHighPrecisionAccumulate(sizeof(Tc) > sizeof(Ti));

        // Environment variable to force use of VALU for double precision gemm
        static bool force_valu_for_dgemm = std::getenv("ROCBLASLT_INTERNAL_FORCE_VALU_FOR_DGEMM");
        if(std::is_same<Ti, double>::value && std::is_same<To, double>::value
           && std::is_same<Tc, double>::value && force_valu_for_dgemm)
        {
            tensileProblem.setArithmeticUnit(Tensile::ArithmeticUnit::VALU);
        }

        // set batch mode
        tensileProblem.setStridedBatched(prob.strided_batch);

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // tensileAlpha=0 Not positive if this is necessary here as well
        typename AlphaBeta<Ti, To, Tc>::tensile_type tensileAlpha;
        if(prob.k)
            AlphaBeta<Ti, To, Tc>::copy(&tensileAlpha, prob.alpha);
        else
            memset(&tensileAlpha, 0, sizeof(tensileAlpha));
        tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(tensileAlpha));

        // Add problem predicates for CEqualsD
        tensileProblem.setCEqualsD(prob.C == prob.D);

        // set bias mode
        tensileProblem.setUseBias(true);
        tensileProblem.setBiasType(hipblasDatatype_to_tensile_type(prob.bias_type));

        // set ScaleD mode
        tensileProblem.setUseScaleD(true);

        // set Actvation
        tensileProblem.setActivationType(Tensile::ActivationType::All);
        tensileProblem.setActivationHPA(sizeof(Tc) > sizeof(Ti));
        Tensile::ActivationType tensileAct = Tensile::ActivationType::None;
        switch(prob.epilogue)
        {
        case ROCBLASLT_EPILOGUE_RELU:
        case ROCBLASLT_EPILOGUE_RELU_BIAS:
            tensileAct = Tensile::ActivationType::Relu;
            break;
        case ROCBLASLT_EPILOGUE_GELU:
        case ROCBLASLT_EPILOGUE_GELU_BIAS:
            tensileAct = Tensile::ActivationType::Gelu;
            break;
        case ROCBLASLT_EPILOGUE_BIAS:
        case ROCBLASLT_EPILOGUE_DEFAULT:
            break;
        }
        tensileProblem.setActivationEnumArg(tensileAct);

        return tensileProblem;
    }

    /***************************************************************
 * Construct the inputs to a Tensile ContractionProblem        *
 ***************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto GetTensileInputs(const RocblasltContractionProblem<Ti, To, Tc>& prob)
    {
        // Tensile types corresponding to Ti, To, Tc
        using Tensile_Ti          = typename rocblaslt_to_tensile_type<Ti>::tensile_type;
        using Tensile_To          = typename rocblaslt_to_tensile_type<To>::tensile_type;
        using Tensile_Talpha_beta = typename AlphaBeta<Ti, To, Tc>::tensile_type;

        // Make sure rocblaslt and Tensile types are compatible
        // (Even if Ti=rocblaslt_int8x4, Tensile_Ti=Int8x4, they are both 32-byte)
        static_assert(sizeof(Tensile_Ti) == sizeof(Ti) && sizeof(Tensile_To) == sizeof(To),
                      "Tensile and rocblaslt types are not the same size");

        static_assert(std::is_standard_layout<Ti>{} && std::is_standard_layout<Tensile_Ti>{}
                          && std::is_standard_layout<To>{} && std::is_standard_layout<Tensile_To>{},
                      "Tensile or rocblaslt types are not standard layout types");

        // Structure describing the inputs (A, B, C, D, alpha, beta)
        Tensile::TypedContractionInputs<Tensile_Ti,
                                        Tensile_Ti,
                                        Tensile_To,
                                        Tensile_To,
                                        Tensile_Talpha_beta,
                                        Tensile_Talpha_beta>
            inputs;

        // Set the A, B, C, D matrices pointers in Tensile
        inputs.a = reinterpret_cast<const Tensile_Ti*>(prob.A);
        inputs.b = reinterpret_cast<const Tensile_Ti*>(prob.B);
        inputs.c = reinterpret_cast<const Tensile_To*>(prob.C);
        inputs.d = reinterpret_cast<Tensile_To*>(prob.D);

        inputs.batchA = reinterpret_cast<Tensile_Ti const* const*>(prob.batch_A);
        inputs.batchB = reinterpret_cast<Tensile_Ti const* const*>(prob.batch_B);
        inputs.batchC = reinterpret_cast<Tensile_To const* const*>(prob.batch_C);
        inputs.batchD = reinterpret_cast<Tensile_To* const*>(prob.batch_D);

        // Set the GSU workspace
        inputs.ws = prob.workspace;

        // set bias vector
        inputs.bias   = reinterpret_cast<const Tensile_To*>(prob.bias);
        inputs.scaleD = reinterpret_cast<const Tensile_Talpha_beta*>(prob.scaleD);

        // push 2 activation arguments
        inputs.activationArgs.push_back(static_cast<Tensile_To>(0.0f));
        inputs.activationArgs.push_back(static_cast<Tensile_To>(0.0f));

        // alpha and beta are stored by value in Tensile::TypedContractionInputs
        // alpha and beta are copied from host to Tensile::TypedContractionInputs
        // If k==0, we do not need to dereference prob.alpha and can set
        // inputs.alpha=0
        if(prob.k)
            AlphaBeta<Ti, To, Tc>::copy(&inputs.alpha, prob.alpha);
        else
            memset(&inputs.alpha, 0, sizeof(inputs.alpha));
        AlphaBeta<Ti, To, Tc>::copy(&inputs.beta, prob.beta);

        return inputs;
    }

    /**************************************************
 * The TensileHost struct interfaces with Tensile *
 **************************************************/
    class TensileHost
    {
        // The library object
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> m_library;
        std::shared_ptr<hipDeviceProp_t>                                             m_deviceProp;

        // The adapter object. mutable is used to allow adapters to be modified
        // even when they are stored in a const vector which is immutable in size
        struct adapter_s
        {
            mutable std::atomic<Tensile::hip::SolutionAdapter*> adapter{nullptr};
            mutable std::mutex                                  mutex;
        };

        // Each device contains an adapter
        std::vector<adapter_s> const m_adapters;

    public:
        TensileHost()
            : m_adapters(GetDeviceCount())
        {
            // We mark TensileHost as initialized. This is so that CI tests can
            // verify that the initialization occurs in the "multiheaded" tests
            rocblaslt_internal_tensile_is_initialized() = true;
        }

        // TensileHost is not copyable or assignable
        TensileHost(const TensileHost&) = delete;
        TensileHost& operator=(const TensileHost&) = delete;

        // Get the number of devices
        static int GetDeviceCount()
        {
            int count;
            if(hipGetDeviceCount(&count) != hipSuccess)
            {
                std::cerr << "\nrocblaslt error: Could not initialize Tensile host: No "
                             "devices found"
                          << std::endl;
                // rocblaslt_abort();
            }
            return count;
        }

        ~TensileHost()
        {
            for(auto& a : m_adapters)
                delete a.adapter;
        }

        auto& get_library() const
        {
            return m_library;
        }

        auto& get_device_property() const
        {
            return m_deviceProp;
        }

        auto& get_adapters() const
        {
            return m_adapters;
        }

        /*******************************************************
   * Testpath() tests that a path exists and is readable *
   *******************************************************/
        static bool TestPath(const std::string& path)
        {
#ifdef WIN32
            return ((_access(path.c_str(), 4) != -1) || (_access(path.c_str(), 6) != -1));
#else
            return access(path.c_str(), R_OK) == 0;
#endif
        }

        /*********************************************************************
   * Initialize adapter and library according to environment variables *
   * and default paths based on librocblaslt.so location and GPU         *
   *********************************************************************/
        void initialize(Tensile::hip::SolutionAdapter& adapter, int32_t deviceId)
        {
            std::string path;
#ifndef WIN32
            path.reserve(PATH_MAX);
#endif

            // The name of the current GPU platform
            std::string processor = rocblaslt_internal_get_arch_name();

            const char* env = getenv("HIPBLASLT_TENSILE_LIBPATH");
            if(env)
            {
                path = env;
            }
            else
            {
                path = HIPBLASLT_LIB_PATH;

                // Find the location of librocblaslt.so
                // Fall back on hard-coded path if static library or not found

#ifndef HIPBLASLT_STATIC_LIB
                dl_iterate_phdr(hipblaslt_dl_iterate_phdr_callback, NULL);
                if(hipblaslt_so_path.size())
                    path = std::string{dirname(&hipblaslt_so_path[0])};
#endif // ifndef HIPBLASLT_STATIC_LIB

                // Find the location of the libraries
                if(TestPath(path + "/../Tensile/library"))
                    path += "/../Tensile/library";
                else if(TestPath(path + "library"))
                    path += "/library";
                else
                    path += "/hipblaslt/library";

                if(TestPath(path + "/" + processor))
                    path += "/" + processor;
            }

            // only load modules for the current architecture
            auto dir = path + "/*" + processor + "*co";

            bool no_match = false;
#ifdef WIN32
            std::replace(dir.begin(), dir.end(), '/', '\\');
            WIN32_FIND_DATAA finddata;
            HANDLE           hfine = FindFirstFileA(dir.c_str(), &finddata);
            if(hfine != INVALID_HANDLE_VALUE)
            {
                do
                {
                    std::string codeObjectFile = path + "\\" + finddata.cFileName;
                    adapter.loadCodeObjectFile(codeObjectFile.c_str());
                } while(FindNextFileA(hfine, &finddata));
            }
            else
            {
                no_match = true;
            }
            FindClose(hfine);
#else
            glob_t glob_result{};
            int    g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
            if(!g)
            {
                for(size_t i = 0; i < glob_result.gl_pathc; ++i)
                    adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
            }
            else if(g == GLOB_NOMATCH)
            {
                no_match = true;
            }
            else
            {
#if 0
                // clang-format off
                static std::ostream& once = std::cerr
                                    << "\nrocblaslt warning: glob(\"" << dir << "\", ...) returned "
                                    << (g == GLOB_ABORTED ? "GLOB_ABORTED"
                                                          : g == GLOB_NOSPACE ? "GLOB_NOSPACE"
                                                                              : "an unknown error")
                                    << "." << std::endl;
                // clang-format on
#endif
            }
            globfree(&glob_result);
#endif
            if(no_match)
            {
                // static rocblaslt_internal_ostream& once
                //    = rocblaslt_cerr
                std::cerr << "\nrocblaslt warning: No paths matched " << dir
                          << ". Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly."
                          << std::endl;
            }

            // We initialize a local static variable with a lambda function call to
            // avoid race conditions when multiple threads with different device IDs try
            // to initialize library. This ensures that only one thread initializes
            // library, and other threads trying to initialize library wait for it to
            // complete.
            static int once = [&] {
#ifdef TENSILE_YAML
                path += "/TensileLibrary.yaml";
#else
                path += "/TensileLibrary.dat";
#endif
                if(!TestPath(path))
                {
                    std::cerr << "\nrocblaslt error: Cannot read " << path << ": "
                              << strerror(errno) << std::endl;
                    // rocblaslt_abort();
                }

                auto lib = Tensile::LoadLibraryFile<Tensile::ContractionProblem>(path);
                if(!lib)
                    std::cerr << "\nrocblaslt error: Could not load " << path << std::endl;
                else
                {
                    using MSL = Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>;
                    m_library = std::dynamic_pointer_cast<MSL>(lib);
                }
                return 0;
            }();

            if(!m_library)
            {
                std::cerr << "\nrocblaslt error: Could not initialize Tensile library" << std::endl;
                // rocblaslt_abort();
            }

            hipDeviceProp_t prop;
            HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));

            m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);
        }
    };

    // Return the library and adapter for the current HIP device
    auto& get_library_and_adapter(
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>* library
        = nullptr,
        std::shared_ptr<hipDeviceProp_t>* deviceProp = nullptr,
        int                               device     = -1)
    try
    {
        // TensileHost is initialized on the first call
        static TensileHost host;

        if(device == -1)
            hipGetDevice(&device);

        // Adapter entry for the current HIP device ID
        auto& a       = host.get_adapters().at(device);
        auto* adapter = a.adapter.load(std::memory_order_acquire);

        // Once set, a.adapter contains the adapter for the current HIP device ID
        if(!adapter)
        {
            // Lock so that only one thread performs initialization of the adapter
            std::lock_guard<std::mutex> lock(a.mutex);

            adapter = a.adapter.load(std::memory_order_relaxed);
            if(!adapter)
            {
                // Allocate a new adapter using the current HIP device
                adapter = new Tensile::hip::SolutionAdapter;

                // Initialize the adapter and possibly the library
                host.initialize(*adapter, device);

                // Atomically change the adapter stored for this device ID
                a.adapter.store(adapter, std::memory_order_release);
            }
        }

        // If an adapter is found, it is assumed that the library is initialized
        if(library)
            *library = host.get_library();
        if(deviceProp)
            *deviceProp = host.get_device_property();

        return *adapter;
    }
    catch(const std::exception& e)
    {
        std::cerr << "\nrocblaslt error: Could not initialize Tensile host:\n"
                  << e.what() << std::endl;
        // rocblaslt_abort();
    }
    catch(...)
    {
        std::cerr << "\nrocblaslt error: Could not initialize Tensile host:\nUnknown "
                     "exception thrown"
                  << std::endl;
        // rocblaslt_abort();
    }

#if 0
    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_once(const std::ostream& msg)
    {
        if(rocblaslt_suppress_tensile_error_messages())
            return;
        static constexpr char varname[] = "ROCBLASLT_VERBOSE_TENSILE_ERROR";
        static const char*    verbose   = getenv(varname);
        if(!verbose)
        {
            static auto& once = std::cerr
                                << msg
                                << "\nThis message will be only be displayed once, unless the "
                                << varname << " environment variable is set." << std::endl;
        }
        else
            std::cerr << msg << std::endl;
    }
#endif

} // namespace

/******************************************************************************
 * runContractionProblem calls Tensile to run a contraction problem described *
 * by RocblasltContractionProblem *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblaslt_status runContractionProblem(const rocblaslt_matmul_algo*                   algo,
                                       const RocblasltContractionProblem<Ti, To, Tc>& prob)
{
    rocblaslt_status status = rocblaslt_status_internal_error;
    try
    {
        std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
        std::shared_ptr<hipDeviceProp_t>                                             deviceProp;
        std::shared_ptr<Tensile::Hardware>                                           hardware;

        auto& adapter = get_library_and_adapter(&library, &deviceProp, prob.handle->device);

        hardware          = Tensile::hip::GetDevice(*deviceProp);
        auto tensile_prob = ConstructTensileProblem(prob);
        if(algo->fallback && prob.bias == nullptr && prob.scaleD == nullptr
           && tensile_prob.activationEnumArg() == Tensile::ActivationType::None)
        {
            tensile_prob.setUseBias(false);
            tensile_prob.setActivationType(Tensile::ActivationType::None);
            tensile_prob.setActivationHPA(false);
            tensile_prob.setUseScaleD(false);
        }
        std::shared_ptr<Tensile::ContractionSolution> solution
            = std::static_pointer_cast<Tensile::ContractionSolution>(algo->data.ptr);

        if(!solution)
        {
#if 0
            std::ostream msg;
            print_once(msg << "\nrocblaslt error: No Tensile solution found for " << prob);
#endif
            status = rocblaslt_status_not_implemented;
        }
        else
        {
            adapter.launchKernels(solution->solve(tensile_prob, GetTensileInputs(prob), *hardware),
                                  prob.stream,
                                  nullptr,
                                  nullptr);
            status = rocblaslt_status_success;
        }
    }
    catch(const std::exception& e)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
    }
    catch(...)
    {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
    }

    return status;
}

/******************************************************************************
 * ConstructRocblasltProblem creates RocblasltContractionProblem from mat     *
 * layout and descriptor for Tensile's findTopSolutions.                      *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
RocblasltContractionProblem<Ti, To, Tc>
    ConstructRocblasltProblem(const rocblaslt_handle      handle,
                              const rocblaslt_matmul_desc matmul_descr,
                              rocblaslt_matrix_layout     matA,
                              rocblaslt_matrix_layout     matB,
                              rocblaslt_matrix_layout     matC,
                              rocblaslt_matrix_layout     matD,
                              const Tc*                   alpha,
                              const Tc*                   beta,
                              size_t                      maxWorkSpaceBytes)
{
    hipblasOperation_t opA       = matmul_descr->op_A;
    hipblasOperation_t opB       = matmul_descr->op_B;
    const void*        bias      = nullptr;
    hipblasDatatype_t  bias_type = matmul_descr->bias_type == static_cast<hipblasDatatype_t>(0)
                                       ? matD->type
                                       : matmul_descr->bias_type;
    rocblaslt_epilogue epilogue  = matmul_descr->epilogue;
    if(is_bias_enabled(epilogue))
        bias = (const void*)matmul_descr->bias;

    const Tc* scaleD = nullptr;
    if(matmul_descr->scaleD)
        scaleD = (const Tc*)matmul_descr->scaleD;

    // matrix A
    int64_t num_rows_a     = matA->m;
    int64_t num_cols_a     = matA->n;
    int64_t lda            = matA->ld;
    int64_t batch_stride_a = matA->batch_stride;
    int     num_batches_a  = matA->batch_count;

    // matrix B
    int64_t ldb            = matB->ld;
    int64_t batch_stride_b = matB->batch_stride;
    int     num_batches_b  = matB->batch_count;

    // matrix C
    int64_t ldc            = matC->ld;
    int64_t batch_stride_c = matC->batch_stride;
    int     num_batches_c  = matC->batch_count;

    // matrix D
    int64_t num_rows_d     = matD->m;
    int64_t num_cols_d     = matD->n;
    int64_t ldd            = matD->ld;
    int64_t batch_stride_d = matD->batch_stride;
    int     num_batches_d  = matD->batch_count;

    int64_t m = num_rows_d;
    int64_t n = num_cols_d;
    int64_t k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

    int8_t      dummy;
    const void* dummy_ptr = &dummy;
    auto        validArgs = validateMatmulArgs(handle,
                                        m,
                                        n,
                                        k,
                                        dummy_ptr,
                                        dummy_ptr,
                                        dummy_ptr,
                                        dummy_ptr,
                                        dummy_ptr,
                                        dummy_ptr,
                                        num_batches_a,
                                        num_batches_b,
                                        num_batches_c,
                                        num_batches_d,
                                        batch_stride_a,
                                        batch_stride_b,
                                        batch_stride_c,
                                        batch_stride_d);
    if(validArgs != rocblaslt_status_continue)
    {
        m = 0;
        n = 0;
        k = 0;
    }
    RocblasltContractionProblem<Ti, To, Tc> problem{handle,
                                                    opA,
                                                    opB,
                                                    m,
                                                    n,
                                                    k,
                                                    alpha,
                                                    nullptr,
                                                    nullptr,
                                                    lda,
                                                    batch_stride_a,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    ldb,
                                                    batch_stride_b,
                                                    0,
                                                    beta,
                                                    nullptr,
                                                    nullptr,
                                                    ldc,
                                                    batch_stride_c,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    ldd,
                                                    batch_stride_d,
                                                    0,
                                                    num_batches_a,
                                                    true,
                                                    bias,
                                                    scaleD,
                                                    bias_type,
                                                    epilogue,
                                                    nullptr,
                                                    maxWorkSpaceBytes,
                                                    nullptr};
    return problem;
}

/******************************************************************************
 * getBestSolutions calls Tensile's findTopSolutions and converts to          *
 * rocblaslt_matmul_heuristic_result.                                         *
 ******************************************************************************/

void _convertToHeuristicResultArray(
    std::vector<std::shared_ptr<Tensile::ContractionSolution>>& solutions,
    int                                                         requestedAlgoCount,
    rocblaslt_matmul_heuristic_result                           heuristicResultsArray[],
    int*                                                        returnAlgoCount,
    size_t                                                      maxWorkSpaceBytes,
    const Tensile::ContractionProblem&                          problem,
    size_t                                                      fallbackCount)
{
    *returnAlgoCount = std::min((int)solutions.size(), requestedAlgoCount);
    for(size_t i = 0; i < *returnAlgoCount; i++)
    {
        auto solution                          = solutions[i];
        heuristicResultsArray[i].algo.data.ptr = std::static_pointer_cast<void>(solution);
        heuristicResultsArray[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResultsArray[i].algo.fallback            = fallbackCount-- > 0 ? true : false;
        heuristicResultsArray[i].state                    = rocblaslt_status_success;
        heuristicResultsArray[i].workspaceSize = solution->requiredWorkspaceSize(problem);
    }
    for(size_t i = *returnAlgoCount; i < requestedAlgoCount; i++)
    {
        heuristicResultsArray[i].state = rocblaslt_status_invalid_value;
    }
}

template <typename Ti, typename To, typename Tc>
rocblaslt_status getBestSolutions(RocblasltContractionProblem<Ti, To, Tc> prob,
                                  int                                     requestedAlgoCount,
                                  rocblaslt_matmul_heuristic_result       heuristicResultsArray[],
                                  int*                                    returnAlgoCount,
                                  size_t                                  maxWorkSpaceBytes)
{
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>> library;
    std::shared_ptr<hipDeviceProp_t>                                             deviceProp;
    std::shared_ptr<Tensile::Hardware>                                           hardware;

    // auto &adapter =
    get_library_and_adapter(&library, &deviceProp, prob.handle->device);

    hardware          = Tensile::hip::GetDevice(*deviceProp);
    auto tensile_prob = ConstructTensileProblem(prob);
    std::vector<std::shared_ptr<Tensile::ContractionSolution>> solutions_fallback;
    // Fallback to original kernels
    if(prob.scaleD == nullptr && prob.bias == nullptr
       && tensile_prob.activationEnumArg() == Tensile::ActivationType::None)
    {
        auto useBias   = tensile_prob.useBias();
        auto actType   = tensile_prob.activationType();
        auto actHPA    = tensile_prob.activationHPA();
        auto useScaleD = tensile_prob.useScaleD();
        tensile_prob.setUseBias(false);
        tensile_prob.setActivationType(Tensile::ActivationType::None);
        tensile_prob.setActivationHPA(false);
        tensile_prob.setUseScaleD(false);
        solutions_fallback = library->findTopSolutions(tensile_prob, *hardware, requestedAlgoCount);
        // restore
        tensile_prob.setUseBias(useBias);
        tensile_prob.setActivationType(actType);
        tensile_prob.setActivationHPA(actHPA);
        tensile_prob.setUseScaleD(useScaleD);
    }

    auto solutions = library->findTopSolutions(tensile_prob, *hardware, requestedAlgoCount);
    if(solutions_fallback.size() > 0)
    {
        solutions.insert(solutions.begin(), solutions_fallback.begin(), solutions_fallback.end());
    }
    _convertToHeuristicResultArray(solutions,
                                   requestedAlgoCount,
                                   heuristicResultsArray,
                                   returnAlgoCount,
                                   maxWorkSpaceBytes,
                                   tensile_prob,
                                   solutions_fallback.size());

    return rocblaslt_status_success;
}

/***************************************************************
 * ! \brief  Initialize rocblaslt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocblaslt_createialize()
{
    get_library_and_adapter();
}

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocblaslt dependencies. This file's template functions are not defined in a *
 * header file, in order to keep Tensile and rocblaslt separate. *
 ******************************************************************************/

// types
#define CREATEFUNCTION(Ti, To, Tc)                                                          \
    template rocblaslt_status runContractionProblem<Ti, To, Tc>(                            \
        const rocblaslt_matmul_algo* algo, const RocblasltContractionProblem<Ti, To, Tc>&); \
    template RocblasltContractionProblem<Ti, To, Tc> ConstructRocblasltProblem<Ti, To, Tc>( \
        const rocblaslt_handle      handle,                                                 \
        const rocblaslt_matmul_desc matmul_descr,                                           \
        rocblaslt_matrix_layout     matA,                                                   \
        rocblaslt_matrix_layout     matB,                                                   \
        rocblaslt_matrix_layout     matC,                                                   \
        rocblaslt_matrix_layout     matD,                                                   \
        const Tc*                   alpha,                                                  \
        const Tc*                   beta,                                                   \
        size_t                      maxWorkSpaceBytes);                                                          \
    template rocblaslt_status getBestSolutions<Ti, To, Tc>(                                 \
        RocblasltContractionProblem<Ti, To, Tc> prob,                                       \
        int                                     requestedAlgoCount,                         \
        rocblaslt_matmul_heuristic_result       heuristicResultsArray[],                    \
        int*                                    returnAlgoCount,                            \
        size_t                                  maxWorkSpaceBytes);

CREATEFUNCTION(float, float, float)
CREATEFUNCTION(rocblaslt_half, rocblaslt_half, float)
CREATEFUNCTION(rocblaslt_bfloat16, rocblaslt_bfloat16, float)

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for
 *testing) *
 ***********************************************************************************/
std::atomic_bool& rocblaslt_internal_tensile_is_initialized()
{
    static std::atomic_bool init;
    return init;
}
