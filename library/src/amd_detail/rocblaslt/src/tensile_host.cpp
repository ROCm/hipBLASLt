/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************/

// The implementation of the rocblaslt<->Tensile interface layer.

#include "rocblaslt.h"

/*****************************************************************************
 * This is the only file in rocblaslt which should #include Tensile headers    *
 * or reference Tensile identifiers. tensile_host.hpp defines the interface. *
 *****************************************************************************/

#include "rocblaslt-types.h"
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

namespace {
#ifndef WIN32
std::string hipblaslt_so_path;

int hipblaslt_dl_iterate_phdr_callback(struct dl_phdr_info *hdr_info,
                                       size_t size, void *data) {
  // uncomment to see all dependent .so files
  // fprintf(stderr, "hipblaslt so file: %s\n", hdr_info->dlpi_name);
  if (hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, "hipblaslt.")) {
    hipblaslt_so_path = hdr_info->dlpi_name;
  }
  return 0;
}
#endif

/******************************************************
 * Map a rocblaslt type to a corresponding Tensile type *
 ******************************************************/
template <typename T> struct rocblaslt_to_tensile_type {
  using tensile_type = T;
};

template <> struct rocblaslt_to_tensile_type<rocblaslt_half> {
  using tensile_type = Tensile::Half;
};

/********************************************************************
 * Variable template to map a rocblaslt type into a Tensile::DataType *
 ********************************************************************/
template <typename> constexpr auto tensile_datatype = nullptr;

template <>
constexpr auto tensile_datatype<rocblaslt_half> = Tensile::DataType::Half;

template <> constexpr auto tensile_datatype<float> = Tensile::DataType::Float;

/*************************************************************************
 * Class for converting alpha and beta between rocblaslt and Tensile types *
 * By default, alpha and beta are the same type as Tc compute_type       *
 *************************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To> struct AlphaBeta {
  using tensile_type = typename rocblaslt_to_tensile_type<Tc>::tensile_type;
  static void copy(tensile_type *dst, const Tc *src) {
    static_assert(sizeof(*src) == sizeof(*dst),
                  "Tensile and rocblaslt types are not the same size");
    static_assert(std::is_standard_layout<tensile_type>{} &&
                      std::is_standard_layout<Tc>{},
                  "Tensile or rocblaslt types are not standard layout types");
    memcpy(dst, src, sizeof(*dst));
  }
};

/****************************************************************
 * Construct a Tensile Problem from a RocblasltContractionProblem *
 ****************************************************************/
template <typename Ti, typename To, typename Tc>
auto ConstructTensileProblem(
    const RocblasltContractionProblem<Ti, To, Tc> &prob) {
  // Tensile DataTypes corresponding to rocblaslt data types
  static constexpr Tensile::DataType Tensile_Ti = tensile_datatype<Ti>;
  static constexpr Tensile::DataType Tensile_To = tensile_datatype<To>;
  static constexpr Tensile::DataType Tensile_Tc = tensile_datatype<Tc>;

  // Tensor descriptors for a, b
  Tensile::TensorDescriptor a, b;

  // Tensor ops for matrices, like complex conjugate
  Tensile::TensorOps aops, bops, cops, dops;

  // Tensile Indices for contraction problem
  Tensile::ContractionProblem::FreeIndices freeIndex(2);
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
  Tensile::TensorDescriptor c{
      Tensile_To,
      {prob.m, prob.n, prob.batch_count},
      {prob.row_stride_c, prob.col_stride_c, prob.batch_stride_c},
      prob.buffer_offset_c};

  // Descriptor for output matrix D
  Tensile::TensorDescriptor d{
      Tensile_To,
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
  static bool force_valu_for_dgemm =
      std::getenv("ROCBLASLT_INTERNAL_FORCE_VALU_FOR_DGEMM");
  if (std::is_same<Ti, double>::value && std::is_same<To, double>::value &&
      std::is_same<Tc, double>::value && force_valu_for_dgemm) {
    tensileProblem.setArithmeticUnit(Tensile::ArithmeticUnit::VALU);
  }

  // set batch mode
  tensileProblem.setStridedBatched(prob.strided_batch);

  // alpha and beta are stored by value in Tensile::TypedContractionInputs
  // alpha and beta are copied from host to Tensile::TypedContractionInputs
  // If k==0, we do not need to dereference prob.alpha and can set
  // tensileAlpha=0 Not positive if this is necessary here as well
  typename AlphaBeta<Ti, To, Tc>::tensile_type tensileAlpha;
  if (prob.k)
    AlphaBeta<Ti, To, Tc>::copy(&tensileAlpha, prob.alpha);
  else
    memset(&tensileAlpha, 0, sizeof(tensileAlpha));
  tensileProblem.setAlphaRestriction(Tensile::toScalarValueEnum(tensileAlpha));

  // Add problem predicates for CEqualsD
  tensileProblem.setCEqualsD(prob.C == prob.D);

  // set bias mode
  tensileProblem.setUseBias(true);

  // set Actvation
  tensileProblem.setActivationType(Tensile::ActivationType::All);
  tensileProblem.setActivationHPA(sizeof(Tc) > sizeof(Ti));
  Tensile::ActivationType tensileAct = Tensile::ActivationType::None;
  switch (prob.epilogue) {
  case ROCBLASLT_EPILOGUE_RELU:
  case ROCBLASLT_EPILOGUE_RELU_BIAS:
    tensileAct = Tensile::ActivationType::Relu;
    break;
  case ROCBLASLT_EPILOGUE_GELU:
  case ROCBLASLT_EPILOGUE_GELU_BIAS:
    tensileAct = Tensile::ActivationType::Gelu;
    break;
  }
  tensileProblem.setActivationEnumArg(tensileAct);

  return tensileProblem;
}

/***************************************************************
 * Construct the inputs to a Tensile ContractionProblem        *
 ***************************************************************/
template <typename Ti, typename To, typename Tc>
auto GetTensileInputs(const RocblasltContractionProblem<Ti, To, Tc> &prob) {
  // Tensile types corresponding to Ti, To, Tc
  using Tensile_Ti = typename rocblaslt_to_tensile_type<Ti>::tensile_type;
  using Tensile_To = typename rocblaslt_to_tensile_type<To>::tensile_type;
  using Tensile_Talpha_beta = typename AlphaBeta<Ti, To, Tc>::tensile_type;

  // Make sure rocblaslt and Tensile types are compatible
  // (Even if Ti=rocblaslt_int8x4, Tensile_Ti=Int8x4, they are both 32-byte)
  static_assert(sizeof(Tensile_Ti) == sizeof(Ti) &&
                    sizeof(Tensile_To) == sizeof(To),
                "Tensile and rocblaslt types are not the same size");

  static_assert(std::is_standard_layout<Ti>{} &&
                    std::is_standard_layout<Tensile_Ti>{} &&
                    std::is_standard_layout<To>{} &&
                    std::is_standard_layout<Tensile_To>{},
                "Tensile or rocblaslt types are not standard layout types");

  // Structure describing the inputs (A, B, C, D, alpha, beta)
  Tensile::TypedContractionInputs<Tensile_Ti, Tensile_Ti, Tensile_To,
                                  Tensile_To, Tensile_Talpha_beta,
                                  Tensile_Talpha_beta>
      inputs;

  // Set the A, B, C, D matrices pointers in Tensile
  inputs.a = reinterpret_cast<const Tensile_Ti *>(prob.A);
  inputs.b = reinterpret_cast<const Tensile_Ti *>(prob.B);
  inputs.c = reinterpret_cast<const Tensile_To *>(prob.C);
  inputs.d = reinterpret_cast<Tensile_To *>(prob.D);

  inputs.batchA = reinterpret_cast<Tensile_Ti const *const *>(prob.batch_A);
  inputs.batchB = reinterpret_cast<Tensile_Ti const *const *>(prob.batch_B);
  inputs.batchC = reinterpret_cast<Tensile_To const *const *>(prob.batch_C);
  inputs.batchD = reinterpret_cast<Tensile_To *const *>(prob.batch_D);

  // Set the GSU workspace
  inputs.ws = prob.workspace;

  // set bias vector
  inputs.bias = prob.bias;

  // push 2 activation arguments
  inputs.activationArgs.push_back(0);
  inputs.activationArgs.push_back(0);

  // alpha and beta are stored by value in Tensile::TypedContractionInputs
  // alpha and beta are copied from host to Tensile::TypedContractionInputs
  // If k==0, we do not need to dereference prob.alpha and can set
  // inputs.alpha=0
  if (prob.k)
    AlphaBeta<Ti, To, Tc>::copy(&inputs.alpha, prob.alpha);
  else
    memset(&inputs.alpha, 0, sizeof(inputs.alpha));
  AlphaBeta<Ti, To, Tc>::copy(&inputs.beta, prob.beta);

  return inputs;
}

/**************************************************
 * The TensileHost struct interfaces with Tensile *
 **************************************************/
class TensileHost {
  // The library object
  std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>
      m_library;
  std::shared_ptr<hipDeviceProp_t> m_deviceProp;

  // The adapter object. mutable is used to allow adapters to be modified
  // even when they are stored in a const vector which is immutable in size
  struct adapter_s {
    mutable std::atomic<Tensile::hip::SolutionAdapter *> adapter{nullptr};
    mutable std::mutex mutex;
  };

  // Each device contains an adapter
  std::vector<adapter_s> const m_adapters;

public:
  TensileHost() : m_adapters(GetDeviceCount()) {
    // We mark TensileHost as initialized. This is so that CI tests can
    // verify that the initialization occurs in the "multiheaded" tests
    rocblaslt_internal_tensile_is_initialized() = true;
  }

  // TensileHost is not copyable or assignable
  TensileHost(const TensileHost &) = delete;
  TensileHost &operator=(const TensileHost &) = delete;

  // Get the number of devices
  static int GetDeviceCount() {
    int count;
    if (hipGetDeviceCount(&count) != hipSuccess) {
      std::cerr << "\nrocblaslt error: Could not initialize Tensile host: No "
                   "devices found"
                << std::endl;
      // rocblaslt_abort();
    }
    return count;
  }

  ~TensileHost() {
    for (auto &a : m_adapters)
      delete a.adapter;
  }

  auto &get_library() const { return m_library; }

  auto &get_device_property() const { return m_deviceProp; }

  auto &get_adapters() const { return m_adapters; }

  /*******************************************************
   * Testpath() tests that a path exists and is readable *
   *******************************************************/
  static bool TestPath(const std::string &path) {
#ifdef WIN32
    return ((_access(path.c_str(), 4) != -1) ||
            (_access(path.c_str(), 6) != -1));
#else
    return access(path.c_str(), R_OK) == 0;
#endif
  }

  /*********************************************************************
   * Initialize adapter and library according to environment variables *
   * and default paths based on librocblaslt.so location and GPU         *
   *********************************************************************/
  void initialize(Tensile::hip::SolutionAdapter &adapter, int32_t deviceId) {
    std::string path;
#ifndef WIN32
    path.reserve(PATH_MAX);
#endif

    // The name of the current GPU platform
    std::string processor = rocblaslt_internal_get_arch_name();

    const char *env = getenv("HIPBLASLT_TENSILE_LIBPATH");
    if (env) {
      path = env;
    } else {
      path = HIPBLASLT_LIB_PATH;

      // Find the location of librocblaslt.so
      // Fall back on hard-coded path if static library or not found

#ifndef HIPBLASLT_STATIC_LIB
      dl_iterate_phdr(hipblaslt_dl_iterate_phdr_callback, NULL);
      if (hipblaslt_so_path.size())
        path = std::string{dirname(&hipblaslt_so_path[0])};
#endif // ifndef HIPBLASLT_STATIC_LIB

      // Find the location of the libraries
      if (TestPath(path + "/../Tensile/library"))
        path += "/../Tensile/library";
      else if (TestPath(path + "library"))
        path += "/library";
      else
        path += "/hipblaslt/library";

      if (TestPath(path + "/" + processor))
        path += "/" + processor;
    }

    // only load modules for the current architecture
    auto dir = path + "/*" + processor + "*co";

    bool no_match = false;
#ifdef WIN32
    std::replace(dir.begin(), dir.end(), '/', '\\');
    WIN32_FIND_DATAA finddata;
    HANDLE hfine = FindFirstFileA(dir.c_str(), &finddata);
    if (hfine != INVALID_HANDLE_VALUE) {
      do {
        std::string codeObjectFile = path + "\\" + finddata.cFileName;
        adapter.loadCodeObjectFile(codeObjectFile.c_str());
      } while (FindNextFileA(hfine, &finddata));
    } else {
      no_match = true;
    }
    FindClose(hfine);
#else
    glob_t glob_result{};
    int g = glob(dir.c_str(), GLOB_NOSORT, nullptr, &glob_result);
    if (!g) {
      for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        adapter.loadCodeObjectFile(glob_result.gl_pathv[i]);
    } else if (g == GLOB_NOMATCH) {
      no_match = true;
    } else {
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
    if (no_match) {
      // static rocblaslt_internal_ostream& once
      //    = rocblaslt_cerr
      std::cerr
          << "\nrocblaslt warning: No paths matched " << dir
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
      if (!TestPath(path)) {
        std::cerr << "\nrocblaslt error: Cannot read " << path << ": "
                  << strerror(errno) << std::endl;
        // rocblaslt_abort();
      }

      auto lib = Tensile::LoadLibraryFile<Tensile::ContractionProblem>(path);
      if (!lib)
        std::cerr << "\nrocblaslt error: Could not load " << path << std::endl;
      else {
        using MSL = Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>;
        m_library = std::dynamic_pointer_cast<MSL>(lib);
      }
      return 0;
    }();

    if (!m_library) {
      std::cerr << "\nrocblaslt error: Could not initialize Tensile library"
                << std::endl;
      // rocblaslt_abort();
    }

    hipDeviceProp_t prop;
    HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceId));

    m_deviceProp = std::make_shared<hipDeviceProp_t>(prop);
  }
};

// Return the library and adapter for the current HIP device
auto &get_library_and_adapter(
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>
        *library = nullptr,
    std::shared_ptr<hipDeviceProp_t> *deviceProp = nullptr,
    int device = -1) try {
  // TensileHost is initialized on the first call
  static TensileHost host;

  if (device == -1)
    hipGetDevice(&device);

  // Adapter entry for the current HIP device ID
  auto &a = host.get_adapters().at(device);
  auto *adapter = a.adapter.load(std::memory_order_acquire);

  // Once set, a.adapter contains the adapter for the current HIP device ID
  if (!adapter) {
    // Lock so that only one thread performs initialization of the adapter
    std::lock_guard<std::mutex> lock(a.mutex);

    adapter = a.adapter.load(std::memory_order_relaxed);
    if (!adapter) {
      // Allocate a new adapter using the current HIP device
      adapter = new Tensile::hip::SolutionAdapter;

      // Initialize the adapter and possibly the library
      host.initialize(*adapter, device);

      // Atomically change the adapter stored for this device ID
      a.adapter.store(adapter, std::memory_order_release);
    }
  }

  // If an adapter is found, it is assumed that the library is initialized
  if (library)
    *library = host.get_library();
  if (deviceProp)
    *deviceProp = host.get_device_property();

  return *adapter;
} catch (const std::exception &e) {
  std::cerr << "\nrocblaslt error: Could not initialize Tensile host:\n"
            << e.what() << std::endl;
  // rocblaslt_abort();
} catch (...) {
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
rocblaslt_status
runContractionProblem(const RocblasltContractionProblem<Ti, To, Tc> &prob) {
  rocblaslt_status status = rocblaslt_status_internal_error;
  std::shared_ptr<Tensile::ContractionSolution> solution;

  try {
    std::shared_ptr<Tensile::MasterSolutionLibrary<Tensile::ContractionProblem>>
        library;
    std::shared_ptr<hipDeviceProp_t> deviceProp;
    std::shared_ptr<Tensile::Hardware> hardware;

    auto &adapter =
        get_library_and_adapter(&library, &deviceProp, prob.handle->device);

    hardware = Tensile::hip::GetDevice(*deviceProp);
    auto tensile_prob = ConstructTensileProblem(prob);
    auto handle = prob.handle;

    solution = library->findBestSolution(tensile_prob, *hardware);

    if (!solution) {
#if 0
            std::ostream msg;
            print_once(msg << "\nrocblaslt error: No Tensile solution found for " << prob);
#endif
      status = rocblaslt_status_not_implemented;
    } else {
      adapter.launchKernels(
          solution->solve(tensile_prob, GetTensileInputs(prob), *hardware),
          prob.stream, nullptr, nullptr);
      status = rocblaslt_status_success;
    }
  } catch (const std::exception &e) {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but exception thrown for " << prob << e.what());
#endif
  } catch (...) {
#if 0
        std::ostream msg;
        print_once(msg << "\nrocblaslt error: " << (solution ? "" : "No ")
                       << "Tensile solution found, but unknown exception thrown for " << prob);
#endif
  }

  return status;
}

/***************************************************************
 * ! \brief  Initialize rocblaslt for the current HIP device, to *
 * avoid costly startup time at the first call on that device. *
 ***************************************************************/
extern "C" void rocblaslt_createialize() { get_library_and_adapter(); }

/******************************************************************************
 * Intantiate the cases of runContractionProblem which are needed to satisfy  *
 * rocblaslt dependencies. This file's template functions are not defined in a *
 * header file, in order to keep Tensile and rocblaslt separate. *
 ******************************************************************************/

// types
template rocblaslt_status
runContractionProblem(const RocblasltContractionProblem<float, float, float> &);
template rocblaslt_status runContractionProblem(
    const RocblasltContractionProblem<rocblaslt_half, rocblaslt_half, float> &);

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for
 *testing) *
 ***********************************************************************************/
std::atomic_bool &rocblaslt_internal_tensile_is_initialized() {
  static std::atomic_bool init;
  return init;
}
