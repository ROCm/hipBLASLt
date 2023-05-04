#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <ostream>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/noncopyable.hpp>

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

namespace {
    using CUMaskComponent = std::uint32_t;
    using CUMask = std::vector<CUMaskComponent>;
    constexpr auto CUMaskComponentSize = sizeof(CUMaskComponent) * 2;
    CUMask toCUMask(const std::string &maskString) {
        auto beg = maskString.find("0x");

        if (beg != std::string::npos) {
            beg += 2;
        }

        CUMask mask;

        for (auto i = beg; i < maskString.size(); i += CUMaskComponentSize) {
            auto substr = maskString.substr(i, CUMaskComponentSize);
            mask.push_back(std::stoul(substr, nullptr, 16));
        }

        return mask;
    }

    CUMask inversedCUMask(const CUMask &mask) {
        CUMask ret;

        std::transform(begin(mask), end(mask), std::back_inserter(ret), [] (auto i) {
            return ~i;
        });

        return ret;
    }

    std::ostream &operator<<(std::ostream &os, const CUMask &mask) {
        for (std::size_t i = 0; i + 1 < mask.size(); ++i) {
            os << std::hex << std::setfill('0') << std::setw(4) << mask[i] << ' ';
        }

        os << mask.back() << '\n';
        os << std::dec;
        return os;
    }

    std::size_t numCUOfDevice(int deviceId = 0) {
        hipDevice_t dev;
        auto err = hipDeviceGet(&dev, deviceId);
        hipDeviceProp_t deviceProp;
        err = hipGetDeviceProperties(&deviceProp, deviceId);
        return static_cast<std::size_t>(deviceProp.multiProcessorCount);
    }

    bool isCUMaskValid(std::size_t numCU, const CUMask &mask) {
        auto numBits = std::accumulate(begin(mask), end(mask), 0ull, [](auto l, auto r){
            return l + sizeof(r) * 8;
        });

        return numBits >= numCU;
    }

    template<typename T>
    std::size_t numActiveBits(T v) {
        static_assert(std::is_integral<T>::value, "T must be integral");
        std::size_t n{};

        for (size_t i = 0; i < sizeof(T) * 8; ++i) {
            n += ((v >> i) & 1);
        }

        return n;
    }

    std::size_t numActiveCUs(const CUMask &mask) {
        return std::accumulate(begin(mask), end(mask), 0ull, [] (auto l, auto r) {
            return l += numActiveBits(r);
        });
    }

    struct hipAutoEvent : boost::noncopyable {
        explicit hipAutoEvent() {
            CHECK_HIP_ERROR(hipEventCreate(&ev));
        }

        ~hipAutoEvent() {
            CHECK_HIP_ERROR(hipEventDestroy(ev));
        }

        operator hipEvent_t() {
            return ev;
        }

    private:
        hipEvent_t ev;
    };

    enum {
        M = 1000000,
        G = 1000000000,
        T = 1000000000000
    };

    template<std::size_t OpScale = T>
    float flops(std::size_t numOps, float timeMs) {
        return (numOps * 1e3 / timeMs) / OpScale;
    }

    template<typename DType>
    struct hipAutoBuffer : boost::noncopyable {
        explicit hipAutoBuffer(std::size_t numElements)
            : numElements(numElements) {
            CHECK_HIP_ERROR(hipMalloc(&buffer, numElements * sizeof(DType)));
        }

        ~hipAutoBuffer() {
            CHECK_HIP_ERROR(hipFree(buffer));
        }

        DType *data() {
            return buffer;
        }

        const DType *data() const {
            return buffer;
        }

        std::size_t numBytes() const {
            return numElements * sizeof(DType);
        }

    private:
        DType *buffer{};
        std::size_t numElements;
    };

    template<typename DType>
    constexpr hipblasDatatype_t tohipblasDataType() {
        if constexpr (std::is_same<DType, hipblasLtFloat>::value) {
            return HIPBLAS_R_32F;
        } else if constexpr (std::is_same<DType, hipblasLtHalf>::value) {
            return HIPBLAS_R_16F;
        } else if constexpr (std::is_same<DType, hipblasLtBfloat16>::value) {
            return HIPBLAS_R_16B;
        }
        return HIPBLAS_R_32F;
    }

    class KernelRunner {
    public:
        virtual void run(hipStream_t, bool sync = false) = 0;
        virtual std::size_t profile(hipStream_t, std::size_t, std::size_t) { return 0; }
        virtual ~KernelRunner() = default;
        virtual std::size_t numElementsProcess() const = 0;
    };

    template<typename DType, std::size_t MaxWorkspaceSize = (32ull << 20)>
    class GEMMRunner : public KernelRunner {
        static constexpr auto datatype = tohipblasDataType<DType>();
    public:
        GEMMRunner(std::size_t m, std::size_t n, std::size_t k)
        : bufA(m * k), bufB(n * k), bufC(m * n), bufD(m * n), workspace(MaxWorkspaceSize) {
            CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmulDesc, HIPBLASLT_COMPUTE_F32, HIPBLAS_R_32F));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&matmulPref));
            uint64_t maxWorkspaceSize = MaxWorkspaceSize;
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(matmulPref,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &maxWorkspaceSize, sizeof(maxWorkspaceSize)));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutA, datatype, m, k, m));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutB, datatype, k, n, k));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutC, datatype, m, n, m));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&layoutD, datatype, m, n, m));
            solve(1);
        }

        ~GEMMRunner() {
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutA));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutB));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutC));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(layoutD));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmulDesc));
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(matmulPref));
            CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        }

        void run(hipStream_t stream, bool sync) override {
            float alpha = 1.f;
            float beta = 1.f;
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmulDesc, &alpha,
                bufA.data(), layoutA,
                bufB.data(), layoutB,
                &beta,
                bufC.data(), layoutC,
                bufD.data(), layoutD,
                &algos.at(bestAlgoIdx).algo,
                workspace.data(),
                workspace.numBytes(),
                stream));

            if (sync) {
                CHECK_HIP_ERROR(hipDeviceSynchronize());
            }
        }

        std::size_t profile(hipStream_t stream, std::size_t numBench, std::size_t numSync) override {
            float alpha = 1.f;
            float beta = 1.f;
            float bestDur = std::numeric_limits<float>::max();
            std::size_t bestIdx;
            hipAutoEvent beg, end;

            for (std::size_t algoIdx = 0; algoIdx < algos.size(); ++algoIdx) {
                hipEventRecord(beg);
                for (std::size_t i = 0; i < numSync; ++i) {
                    for (std::size_t j = 0; j < numBench; ++j) {
                        run(stream, false);
                    }
                    hipDeviceSynchronize();
                }
                hipEventRecord(end);
                float dur{};
                hipEventElapsedTime(&dur, beg, end);

                if (dur < bestDur) {
                    bestDur = dur;
                    bestIdx = algoIdx;
                }
            }

            bestAlgoIdx = bestIdx;
            return bestIdx;
        }

        std::size_t numElementsProcess() const override {
            return 0;
        }

    private:
        void solve(int numSols) {
            int retNumSols{};
            algos.resize(numSols);
            CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutC, layoutD, matmulPref, numSols, algos.data(), &retNumSols));

            if (retNumSols != numSols) {
                algos.resize(retNumSols);
            }
        }

    private:
        std::size_t m{};
        std::size_t n{};
        std::size_t k{};
        hipblasLtHandle_t handle;
        hipblasLtMatmulDesc_t matmulDesc;
        hipblasLtMatmulPreference_t matmulPref;
        hipblasLtMatrixLayout_t layoutA;
        hipblasLtMatrixLayout_t layoutB;
        hipblasLtMatrixLayout_t layoutC;
        hipblasLtMatrixLayout_t layoutD;
        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        hipAutoBuffer<DType> bufA;
        hipAutoBuffer<DType> bufB;
        hipAutoBuffer<DType> bufC;
        hipAutoBuffer<DType> bufD;
        hipAutoBuffer<uint8_t> workspace;
        std::size_t bestAlgoIdx{};
    };

    template<typename DType>
    __global__ void memoryBoundKernel(DType* input, DType* output, const int N) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        while (tid < N) {
            output[tid] = std::max(input[tid], (DType)0);
            tid += gridDim.x * blockDim.x;
        }
    }

    template<typename DType>
    class MemoryBoundKernelRunner : public KernelRunner {
    public:
        explicit MemoryBoundKernelRunner(std::size_t numElements)
        : numElements(numElements), in(numElements), out(numElements) {

        }

        void run(hipStream_t stream, bool sync) override {
            memoryBoundKernel<DType><<<numElements / 1024, 1024, 0, stream>>>(in.data(), out.data(), numElements);

            if (sync) {
                hipDeviceSynchronize();
            }
        }

        std::size_t profile(hipStream_t stream, std::size_t numBench, std::size_t numSync) override {
            return KernelRunner::profile(stream, numBench, numSync);
        }

        std::size_t numElementsProcess() const override {
            return numElements;
        }

    private:
        std::size_t numElements;
        hipAutoBuffer<DType> in;
        hipAutoBuffer<DType> out;
    };

    std::unique_ptr<KernelRunner> makeGEMMRunner(std::size_t m, std::size_t n ,std::size_t k, hipblasDatatype_t datatype) {
        if (datatype == HIPBLAS_R_32F) {
            return std::make_unique<GEMMRunner<hipblasLtFloat>>(m, n, k);
        } else if (datatype == HIPBLAS_R_16F) {
            return std::make_unique<GEMMRunner<hipblasLtHalf>>(m, n, k);
        } else if (datatype == HIPBLAS_R_16B) {
            return std::make_unique<GEMMRunner<hipblasLtBfloat16>>(m, n, k);
        }
        return nullptr;
    }

    std::unique_ptr<KernelRunner> makeMemoryBoundKernelRunner(std::size_t numElements, hipblasDatatype_t datatype) {
        if (datatype == HIPBLAS_R_32F) {
            return std::make_unique<MemoryBoundKernelRunner<hipblasLtFloat>>(numElements);
        } else if (datatype == HIPBLAS_R_16F) {
            return std::make_unique<MemoryBoundKernelRunner<hipblasLtHalf>>(numElements);
        } else if (datatype == HIPBLAS_R_16B) {
            return std::make_unique<MemoryBoundKernelRunner<hipblasLtBfloat16>>(numElements);
        }
        return nullptr;
    }

    hipblasDatatype_t toHipblasDatatype(const std::string &s) {
        if (s == "f32") {
            return HIPBLAS_R_32F;
        } else if (s == "f16") {
            return HIPBLAS_R_16F;
        } else if (s == "b16") {
            return HIPBLAS_R_16B;
        }

        return HIPBLAS_R_32F;
    }

    bool isDatatypeStringValid(const std::string &s) {
        return std::set<std::string>{"f32", "f16", "b16"}.count(s);
    }
}

namespace po = boost::program_options;

int main(int argc, char **argv) {
    try {
        po::options_description desc("hipBLASLt CU-masked stream example");
        desc.add_options()
            ("help,h", "Help screen")
            ("datatype,d", po::value<std::string>()->default_value("f32"), "Data type for GEMM")
            ("verbose,v", po::value<bool>()->default_value(false)->zero_tokens(), "Verbose output")
            ("m,m", po::value<std::size_t>()->default_value(1024), "M dimension of GEMM")
            ("n,n", po::value<std::size_t>()->default_value(1024), "N dimension of GEMM")
            ("k,k", po::value<std::size_t>()->default_value(1024), "K dimension of GEMM")
            ("num_bench", po::value<std::size_t>()->default_value(1), "# of benchmark run")
            ("num_sync", po::value<std::size_t>()->default_value(1), "# of synchronized run")
            ("cu_mask,c", po::value<std::string>(), "CU-Mask for hipStream of GEMM, a string with hex digits, e.g. 0xffffffffffffffffffffffff00000000")
            ("m_cu_mask", po::value<std::string>(), "CU-Mask for hipStream of memory-bound kernel, a string with hex digits, e.g. 0x000000000000000000000000ffffffff");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        const auto verbose = vm.at("verbose").as<bool>();
        hipblasDatatype_t datatype = HIPBLAS_R_32F;

        if (vm.count("datatype")) {
            const auto datatypeStr = vm.at("datatype").as<std::string>();

            if (!isDatatypeStringValid(datatypeStr)) {
                std::cout << "Invalid datatype: " << datatypeStr << '\n';
                return EXIT_FAILURE;
            }

            datatype = toHipblasDatatype(datatypeStr);
        }

        bool useCUMask{};
        CUMask cuMask;

        if (vm.count("cu_mask")) {
            auto &maskStr = vm.at("cu_mask").as<std::string>();
            cuMask = toCUMask(maskStr);
            useCUMask = true;

            if (verbose) {
                std::cout << "CU Mask: " << cuMask;
            }
        }

        auto numCU = numCUOfDevice();

        if (useCUMask && !isCUMaskValid(numCU, cuMask)) {
            std::cout << "Invalid CU mask, # of bits(" << numActiveCUs(cuMask) << ")" << "< # of CU(" << numCU << ")\n";
            return EXIT_FAILURE;
        }

        auto m = vm.at("m").as<std::size_t>();
        auto n = vm.at("n").as<std::size_t>();
        auto k = vm.at("k").as<std::size_t>();

        if (verbose) {
            std::cout << "GEMM sizes: " << m << ", " << n << ", " << k << '\n';
        }

        auto numBench = vm.at("num_bench").as<std::size_t>();
        auto numSync = vm.at("num_sync").as<std::size_t>();

        hipStream_t stream{};
        hipStream_t mStream{};

        if (useCUMask) {
            CHECK_HIP_ERROR(hipExtStreamCreateWithCUMask(&stream, cuMask.size(), cuMask.data()));
            auto mMask = toCUMask(vm.at("m_cu_mask").as<std::string>());
            CHECK_HIP_ERROR(hipExtStreamCreateWithCUMask(&mStream, mMask.size(), mMask.data()));
        } else {
            hipStreamCreate(&stream);
            hipStreamCreate(&mStream);
        }

        auto runner = makeGEMMRunner(m, n, k, datatype);

        if (verbose) {
            std::cout << "Profiling GEMM solutions...\n";
        }

        auto bestIdx = runner->profile(stream, numBench, numSync);

        if (verbose) {
            std::cout << "Best algo idx: " << bestIdx << '\n';
        }

        if (verbose) {
            std::cout << "Warmup run for GEMM\n";
        }

        // warmup
        for (size_t i = 0; i < 10; ++i) {
            runner->run(stream, true);
        }

        auto mRunner = makeMemoryBoundKernelRunner(1024 * 48, datatype);

        if (verbose) {
            std::cout << "Warmup run for memory bound kernel\n";
        }

        // warmup
        for (size_t i = 0; i < 10; ++i) {
            mRunner->run(mStream, true);
        }

        auto profileRun = [numSync, numBench, &runner, &mRunner, m, n, k] (hipStream_t stream, hipStream_t mStream) {
            hipAutoEvent beg, end;
            CHECK_HIP_ERROR(hipEventRecord(beg, stream));

            for (std::size_t j = 0; j < numSync; ++j) {
                for (std::size_t i = 0; i < numBench; ++i) {
                    runner->run(stream);
                    mRunner->run(mStream);
                }
                CHECK_HIP_ERROR(hipDeviceSynchronize());
            }

            CHECK_HIP_ERROR(hipEventRecord(end, stream));
            CHECK_HIP_ERROR(hipEventSynchronize(end));
            float dur{};
            CHECK_HIP_ERROR(hipEventElapsedTime(&dur, beg, end));
            const auto numRuns = numBench * numSync;
            std::cout << "\tPerf: " << std::to_string(flops<T>(numRuns * (2 * m * n * k + mRunner->numElementsProcess()), dur )) << " Tflops, " << std::to_string(dur / numRuns) << " ms\n";
            return dur;
        };

        std::cout << "Run with CU-mask\n";
        auto withMaskDur = profileRun(stream, mStream);
        std::cout << "Run without CU-mask\n";
        hipStream_t streams[2];
        hipStreamCreate(&streams[0]);
        hipStreamCreate(&streams[1]);
        auto withoutMaskDur = profileRun(streams[0], streams[1]);
        std::cout << (withoutMaskDur / withMaskDur) * 100 << "\%\n";
        hipStreamDestroy(streams[0]);
        hipStreamDestroy(streams[1]);

        if (stream) {
            CHECK_HIP_ERROR(hipStreamDestroy(stream));
        }

        if (mStream) {
            CHECK_HIP_ERROR(hipStreamDestroy(mStream));
        }
    } catch (const po::error &e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
