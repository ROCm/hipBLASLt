/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#include "hipblaslt-ext.hpp"
#include "exceptions.hpp"
#include "hipblaslt_internal.hpp"
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt_float8.h>
#include <iostream>
#include <rocblaslt.h>

namespace hipblaslt_ext
{
    class GemmProblemTypeV2::GemmProblemTypeImpl
    {
    public:
        hipblasOperation_t   op_a; //!< The A martix transpose
        hipblasOperation_t   op_b; //!< The B matrix transpose
        hipDataType          type_a; //!< The A matrix datatype.
        hipDataType          type_b; //!< The B matrix datatype.
        hipDataType          type_c; //!< The C matrix datatype.
        hipDataType          type_d; //!< The D matrix datatype.
        hipblasComputeType_t type_compute; //!< The compute datatype.
    };

    GemmProblemTypeV2::GemmProblemTypeV2()
        : pimpl(std::make_unique<GemmProblemTypeImpl>())
    {
    }

    GemmProblemTypeV2::GemmProblemTypeV2(hipblasOperation_t opA,
                                         hipblasOperation_t opB,
                                         hipDataType typeA,
                                         hipDataType typeB,
                                         hipDataType typeC,
                                         hipDataType typeD,
                                         hipblasComputeType_t typeCompute)
        : pimpl(std::make_unique<GemmProblemTypeImpl>())
    {
        pimpl->op_a = opA;
        pimpl->op_b = opB;
        pimpl->type_a = typeA;
        pimpl->type_b = typeB;
        pimpl->type_c = typeC;
        pimpl->type_d = typeD;
        pimpl->type_compute = typeCompute;
    }

    GemmProblemTypeV2::~GemmProblemTypeV2() = default;

    GemmProblemTypeV2::GemmProblemTypeV2(const GemmProblemTypeV2& type)
        : pimpl(std::make_unique<GemmProblemTypeImpl>(*type.pimpl))
    {
    }

    GemmProblemTypeV2& GemmProblemTypeV2::operator=(const GemmProblemTypeV2& type)
    {
        *pimpl = *type.pimpl;
        return *this;
    }

    GemmProblemTypeV2::GemmProblemTypeV2(GemmProblemTypeV2&& type)            = default;
    GemmProblemTypeV2& GemmProblemTypeV2::operator=(GemmProblemTypeV2&& type) = default;

    void GemmProblemTypeV2::setOpA(hipblasOperation_t op)
    {
        pimpl->op_a = op;
    }

    void GemmProblemTypeV2::setOpB(hipblasOperation_t op)
    {
        pimpl->op_b = op;
    }

    void GemmProblemTypeV2::setTypeA(hipDataType type)
    {
        pimpl->type_a = type;
    }

    void GemmProblemTypeV2::setTypeB(hipDataType type)
    {
        pimpl->type_b = type;
    }

    void GemmProblemTypeV2::setTypeC(hipDataType type)
    {
        pimpl->type_c = type;
    }

    void GemmProblemTypeV2::setTypeD(hipDataType type)
    {
        pimpl->type_d = type;
    }

    void GemmProblemTypeV2::setTypeCompute(hipblasComputeType_t type)
    {
        pimpl->type_compute = type;
    }

    hipblasOperation_t GemmProblemTypeV2::getOpA() const
    {
        return pimpl->op_a;
    }

    hipblasOperation_t GemmProblemTypeV2::getOpB() const
    {
        return pimpl->op_b;
    }

    hipDataType GemmProblemTypeV2::getTypeA() const
    {
        return pimpl->type_a;
    }

    hipDataType GemmProblemTypeV2::getTypeB() const
    {
        return pimpl->type_b;
    }

    hipDataType GemmProblemTypeV2::getTypeC() const
    {
        return pimpl->type_c;
    }

    hipDataType GemmProblemTypeV2::getTypeD() const
    {
        return pimpl->type_d;
    }

    hipblasComputeType_t GemmProblemTypeV2::getTypeCompute() const
    {
        return pimpl->type_compute;
    }

    class GemmEpilogueV2::GemmEpilogueImpl
    {
    public:
        hipblasLtEpilogue_t mode           = HIPBLASLT_EPILOGUE_DEFAULT;
        hipDataType         bias_data_type = HIPBLASLT_DATATYPE_INVALID;
        int                 aux_ld         = 0;
        int                 aux_stride     = 0;
        int                 scaling_a_type = 0;
        int                 scaling_b_type = 0;
    };

    GemmEpilogueV2::GemmEpilogueV2()
        : pimpl(std::make_unique<GemmEpilogueImpl>())
    {
    }

    GemmEpilogueV2::~GemmEpilogueV2() = default;

    GemmEpilogueV2::GemmEpilogueV2(const GemmEpilogueV2& epilogue)
        : pimpl(std::make_unique<GemmEpilogueImpl>(*epilogue.pimpl))
    {
    }

    GemmEpilogueV2& GemmEpilogueV2::operator=(const GemmEpilogueV2& epilogue)
    {
        *pimpl = *epilogue.pimpl;
        return *this;
    }

    GemmEpilogueV2::GemmEpilogueV2(GemmEpilogueV2&& epilogue)            = default;
    GemmEpilogueV2& GemmEpilogueV2::operator=(GemmEpilogueV2&& epilogue) = default;

    void GemmEpilogueV2::setMode(hipblasLtEpilogue_t mode)
    {
        pimpl->mode = mode;
    }

    void GemmEpilogueV2::setBiasDataType(hipDataType bias_data_type)
    {
        pimpl->bias_data_type = bias_data_type;
    }

    void GemmEpilogueV2::setAuxLeadingDimension(int aux_ld)
    {
        pimpl->aux_ld = aux_ld;
    }

    void GemmEpilogueV2::setAuxBatchStride(int aux_stride)
    {
        pimpl->aux_stride = aux_stride;
    }

    void GemmEpilogueV2::setScalingAType(int scaling_a_type)
    {
        pimpl->scaling_a_type = scaling_a_type;
    }

    void GemmEpilogueV2::setScalingBType(int scaling_b_type)
    {
        pimpl->scaling_b_type = scaling_b_type;
    }

    hipblasLtEpilogue_t GemmEpilogueV2::getMode() const
    {
        return pimpl->mode;
    }

    hipDataType GemmEpilogueV2::getBiasDataType() const
    {
        return pimpl->bias_data_type;
    }

    int GemmEpilogueV2::getAuxLeadingDimension() const
    {
        return pimpl->aux_ld;
    }

    int GemmEpilogueV2::getAuxBatchStride() const
    {
        return pimpl->aux_stride;
    }

    int GemmEpilogueV2::getScalingAType() const
    {
        return pimpl->scaling_a_type;
    }

    int GemmEpilogueV2::getScalingBType() const
    {
        return pimpl->scaling_b_type;
    }

    class GemmInputsV2::GemmInputsImpl
    {
    public:
        const void* a     = nullptr;
        const void* b     = nullptr;
        const void* c     = nullptr;
        const void* d     = nullptr;
        const void* alpha = nullptr;
        const void* beta  = nullptr;
        // Epilogue inputs
        const void* bias          = nullptr;
        const void* scaleA        = nullptr;
        const void* scaleB        = nullptr;
        const void* scaleC        = nullptr;
        const void* scaleD        = nullptr;
        const void* scaleAux      = nullptr;
        const void* scaleAlphaVec = nullptr;
        const void* aux           = nullptr;
        const void* amaxD         = nullptr;
    };

    GemmInputsV2::GemmInputsV2()
        : pimpl(std::make_unique<GemmInputsImpl>())
    {
    }

    GemmInputsV2::~GemmInputsV2() = default;

    GemmInputsV2::GemmInputsV2(const GemmInputsV2& input)
        : pimpl(std::make_unique<GemmInputsImpl>(*input.pimpl))
    {
    }

    GemmInputsV2& GemmInputsV2::operator=(const GemmInputsV2& input)
    {
        *pimpl = *input.pimpl;
        return *this;
    }

    GemmInputsV2::GemmInputsV2(GemmInputsV2&& input)            = default;
    GemmInputsV2& GemmInputsV2::operator=(GemmInputsV2&& input) = default;

    void GemmInputsV2::setA(const void* a)
    {
        pimpl->a = a;
    }

    void GemmInputsV2::setB(const void* b)
    {
        pimpl->b = b;
    }

    void GemmInputsV2::setC(const void* c)
    {
        pimpl->c = c;
    }

    void GemmInputsV2::setD(const void* d)
    {
        pimpl->d = d;
    }

    void GemmInputsV2::setAlpha(const void* alpha)
    {
        pimpl->alpha = alpha;
    }

    void GemmInputsV2::setBeta(const void* beta)
    {
        pimpl->beta = beta;
    }

    void GemmInputsV2::setBias(const void* bias)
    {
        pimpl->bias = bias;
    }

    void GemmInputsV2::setScaleA(const void* scaleA)
    {
        pimpl->scaleA = scaleA;
    }

    void GemmInputsV2::setScaleB(const void* scaleB)
    {
        pimpl->scaleB = scaleB;
    }

    void GemmInputsV2::setScaleC(const void* scaleC)
    {
        pimpl->scaleC = scaleC;
    }

    void GemmInputsV2::setScaleD(const void* scaleD)
    {
        pimpl->scaleD = scaleD;
    }

    void GemmInputsV2::setScaleAux(const void* scaleAux)
    {
        pimpl->scaleAux = scaleAux;
    }

    void GemmInputsV2::setScaleAlphaVec(const void* scaleAlphaVec)
    {
        pimpl->scaleAlphaVec = scaleAlphaVec;
    }

    void GemmInputsV2::setAux(const void* aux)
    {
        pimpl->aux = aux;
    }

    void GemmInputsV2::setAmaxD(const void* amaxD)
    {
        pimpl->amaxD = amaxD;
    }

    const void* GemmInputsV2::getA() const
    {
        return pimpl->a;
    }

    const void* GemmInputsV2::getB() const
    {
        return pimpl->b;
    }

    const void* GemmInputsV2::getC() const
    {
        return pimpl->c;
    }

    const void* GemmInputsV2::getD() const
    {
        return pimpl->d;
    }

    const void* GemmInputsV2::getAlpha() const
    {
        return pimpl->alpha;
    }

    const void* GemmInputsV2::getBeta() const
    {
        return pimpl->beta;
    }

    const void* GemmInputsV2::getBias() const
    {
        return pimpl->bias;
    }

    const void* GemmInputsV2::getScaleA() const
    {
        return pimpl->scaleA;
    }

    const void* GemmInputsV2::getScaleB() const
    {
        return pimpl->scaleB;
    }

    const void* GemmInputsV2::getScaleC() const
    {
        return pimpl->scaleC;
    }

    const void* GemmInputsV2::getScaleD() const
    {
        return pimpl->scaleD;
    }

    const void* GemmInputsV2::getScaleAux() const
    {
        return pimpl->scaleAux;
    }

    const void* GemmInputsV2::getScaleAlphaVec() const
    {
        return pimpl->scaleAlphaVec;
    }

    const void* GemmInputsV2::getAux() const
    {
        return pimpl->aux;
    }

    const void* GemmInputsV2::getAmaxD() const
    {
        return pimpl->amaxD;
    }

    // End of pimpl classes
    /////////////////////////////////////////////////////

    bool currentArchSupportsFp8()
    {
        using std::begin;
        using std::end;

        static const std::string fp8Archs[] = {"gfx940", "gfx941", "gfx942"};
        const auto               archName   = rocblaslt_internal_get_arch_name();
        return std::find(begin(fp8Archs), end(fp8Archs), archName) != end(fp8Archs);
    }

    template <typename SrcType, typename DstType, typename ScaleType = float>
    __global__ void datatypeConversion(const SrcType*   src,
                                       DstType*         dst,
                                       const ScaleType* scale,
                                       std::size_t      numElements)
    {
        const auto tId        = threadIdx.x;
        const auto bId        = blockIdx.x;
        const auto blockSize  = blockDim.x * blockDim.y * blockDim.z;
        const auto elemOffset = bId * blockSize + tId;
        const auto scaleValue = scale ? *scale : 1.f;

        if(elemOffset < numElements)
        {
            dst[elemOffset] = DstType(float(src[elemOffset]) * scaleValue);
        }
    }

    template <typename SrcType, typename DstType>
    void datatypeConversionCpu(const SrcType* src, DstType* dst, std::size_t numElements)
    {
        for(std::size_t i = 0; i < numElements; ++i)
        {
            dst[i] = DstType(src[i]);
        }
    }

    auto NullDeleter = [](void*) { return hipSuccess; };

    HipBufferPtr makeHipBuffer(std::size_t numBytes)
    {
        if(!numBytes)
        {
            return HipBufferPtr(nullptr, NullDeleter);
        }

        void* ptr = nullptr;
        auto  err = hipMalloc(&ptr, numBytes);

        if(err != hipSuccess)
        {
            return HipBufferPtr(nullptr, NullDeleter);
        }

        return HipBufferPtr(ptr, &hipFree);
    }

    void GemmPreference::setMaxWorkspaceBytes(size_t workspaceBytes)
    {
        m_workspace_bytes = workspaceBytes;
    }

    const size_t GemmPreference::getMaxWorkspaceBytes() const
    {
        return m_workspace_bytes;
    }

    GemmInstance::GemmInstance(hipblasLtHandle_t handle, GemmType type)
        : m_gemm_type(type)
        , m_handle(handle)
    {
    }

    GemmInstance::GemmInstance(GemmInstance&& rhs) noexcept            = default;
    GemmInstance& GemmInstance::operator=(GemmInstance&& rhs) noexcept = default;

    GemmType GemmInstance::getGemmType()
    {
        return m_gemm_type;
    }

    size_t GemmInstance::getGemmCount()
    {
        return m_gemm_count;
    }

    hipblasStatus_t GemmInstance::algoGetHeuristic(
        const int                                      requestedAlgoCount,
        const GemmPreference&                          pref,
        std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_algo_get_heuristic_cpp((rocblaslt_handle)m_handle,
                                             gemmType,
                                             m_data,
                                             pref.getMaxWorkspaceBytes(),
                                             requestedAlgoCount,
                                             *results));
    }

    hipblasStatus_t GemmInstance::isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                  size_t&                workspaceSizeInBytes)
    try
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo  = reinterpret_cast<rocblaslt_matmul_algo*>(&algo);
        return RocBlasLtStatusToHIPStatus(rocblaslt_is_algo_supported_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data, *rocalgo, nullptr, workspaceSizeInBytes));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::isAlgoSupported(hipblasLtMatmulAlgo_t& algo,
                                                  GemmTuning&            tuning,
                                                  size_t&                workspaceSizeInBytes)
    try
    {
        auto gemmType  = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo   = reinterpret_cast<rocblaslt_matmul_algo*>(&algo);
        auto roctuning = reinterpret_cast<rocblaslt::RocTuning*>(&tuning);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_is_algo_supported_cpp((rocblaslt_handle)m_handle,
                                            gemmType,
                                            m_data,
                                            *rocalgo,
                                            roctuning,
                                            workspaceSizeInBytes));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::initialize(const hipblasLtMatmulAlgo_t& algo,
                                             void*                        workspace,
                                             bool                         useUserArgs,
                                             hipStream_t                  stream)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo  = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        return RocBlasLtStatusToHIPStatus(rocblaslt_makeArgument_cpp((rocblaslt_handle)m_handle,
                                                                     gemmType,
                                                                     *rocalgo,
                                                                     nullptr,
                                                                     workspace,
                                                                     useUserArgs,
                                                                     stream,
                                                                     m_data));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::initialize(const hipblasLtMatmulAlgo_t& algo,
                                             GemmTuning&                  tuning,
                                             void*                        workspace,
                                             bool                         useUserArgs,
                                             hipStream_t                  stream)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType  = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto rocalgo   = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        auto roctuning = reinterpret_cast<const rocblaslt::RocTuning*>(&tuning);
        return RocBlasLtStatusToHIPStatus(rocblaslt_makeArgument_cpp((rocblaslt_handle)m_handle,
                                                                     gemmType,
                                                                     *rocalgo,
                                                                     roctuning,
                                                                     workspace,
                                                                     useUserArgs,
                                                                     stream,
                                                                     m_data));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    hipblasStatus_t GemmInstance::run(hipStream_t stream, hipEvent_t start, hipEvent_t stop)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;

        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto status   = RocBlasLtStatusToHIPStatus(
            rocblaslt_run_cpp((rocblaslt_handle)m_handle, gemmType, m_data, stream, start, stop));

        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    std::string GemmInstance::getSolutionName()
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return rocblaslt_get_solution_name_from_data_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data);
    }

    std::string GemmInstance::getKernelName()
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return rocblaslt_get_kernel_name_from_data_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data);
    }

    Gemm::Gemm(hipblasLtHandle_t    handle,
               hipblasOperation_t   opA,
               hipblasOperation_t   opB,
               hipDataType          typeA,
               hipDataType          typeB,
               hipDataType          typeC,
               hipDataType          typeD,
               hipblasComputeType_t typeCompute)
        : GemmInstance(handle, GemmType::HIPBLASLT_GEMM)
    {
        m_problem_types.push_back({opA, opB, typeA, typeB, typeC, typeD, typeCompute});
        rocblaslt_init_gemmData((rocblaslt_handle)m_handle,
                                static_cast<rocblaslt::RocGemmType>(m_gemm_type),
                                opA,
                                opB,
                                typeA,
                                typeB,
                                typeC,
                                typeD,
                                (rocblaslt_compute_type)typeCompute,
                                0,
                                m_data);
    }

    Gemm::Gemm(hipblasLtHandle_t       handle,
               hipblasLtMatmulDesc_t   matmul_descr,
               const void*             alpha,
               const void*             A,
               hipblasLtMatrixLayout_t matA,
               const void*             B,
               hipblasLtMatrixLayout_t matB,
               const void*             beta,
               const void*             C,
               hipblasLtMatrixLayout_t matC,
               void*                   D,
               hipblasLtMatrixLayout_t matD)
        : GemmInstance(handle, GemmType::HIPBLASLT_GEMM)
    {
        auto status = setProblem(matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
    }

    Gemm::Gemm(Gemm&&) noexcept            = default;
    Gemm& Gemm::operator=(Gemm&&) noexcept = default;

    hipblasStatus_t Gemm::setProblem(int64_t       m,
                                     int64_t       n,
                                     int64_t       k,
                                     int64_t       batch_count,
                                     GemmEpilogue& epilogue,
                                     GemmInputs&   inputs)
    {
        if(n == 0 || m == 0)
        {
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        int64_t lda     = m_problem_types[0].op_a == HIPBLAS_OP_N ? m : k;
        int64_t ldb     = m_problem_types[0].op_b == HIPBLAS_OP_N ? k : n;
        int64_t ldc     = m;
        int64_t strideA = m * k;
        int64_t strideB = n * k;
        int64_t strideC = m * n;
        return setProblem(m,
                          n,
                          k,
                          batch_count,
                          lda,
                          ldb,
                          ldc,
                          ldc,
                          strideA,
                          strideB,
                          strideC,
                          strideC,
                          epilogue,
                          inputs,
                          m_problem_types[0]);
    }

    hipblasStatus_t Gemm::setProblem(int64_t         m,
                                     int64_t         n,
                                     int64_t         k,
                                     int64_t         batch_count,
                                     GemmEpilogueV2& epilogue,
                                     GemmInputsV2&   inputs)
    {
        if(n == 0 || m == 0)
        {
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        int64_t lda     = m_problem_types[0].op_a == HIPBLAS_OP_N ? m : k;
        int64_t ldb     = m_problem_types[0].op_b == HIPBLAS_OP_N ? k : n;
        int64_t ldc     = m;
        int64_t strideA = m * k;
        int64_t strideB = n * k;
        int64_t strideC = m * n;
        GemmProblemTypeV2 prob(m_problem_types[0].op_a,
                               m_problem_types[0].op_b,
                               m_problem_types[0].type_a,
                               m_problem_types[0].type_b,
                               m_problem_types[0].type_c,
                               m_problem_types[0].type_d,
                               m_problem_types[0].type_compute);
        return setProblem(m,
                          n,
                          k,
                          batch_count,
                          lda,
                          ldb,
                          ldc,
                          ldc,
                          strideA,
                          strideB,
                          strideC,
                          strideC,
                          epilogue,
                          inputs,
                          prob);
    }

    hipblasStatus_t Gemm::setProblem(int64_t            m,
                                     int64_t            n,
                                     int64_t            k,
                                     int64_t            batch_count,
                                     int64_t            lda,
                                     int64_t            ldb,
                                     int64_t            ldc,
                                     int64_t            ldd,
                                     int64_t            strideA,
                                     int64_t            strideB,
                                     int64_t            strideC,
                                     int64_t            strideD,
                                     GemmEpilogue&      epilogue,
                                     GemmInputs&        inputs,
                                     GemmProblemType&   problemtype)
    {
        GemmInputs      gemmInputs      = inputs;
        GemmProblemType gemmProblemType = problemtype;
        auto            rocepilogue     = reinterpret_cast<rocblaslt::RocGemmEpilogue*>(&epilogue);
        auto            rocepinputs     = reinterpret_cast<rocblaslt::RocGemmInputs*>(&gemmInputs);
        auto rocproblemtype = reinterpret_cast<rocblaslt::RocGemmProblemType*>(&gemmProblemType);
        auto status
            = RocBlasLtStatusToHIPStatus(rocblaslt_gemm_create_cpp((rocblaslt_handle)m_handle,
                                                                   m,
                                                                   n,
                                                                   batch_count,
                                                                   k,
                                                                   lda,
                                                                   ldb,
                                                                   ldc,
                                                                   ldd,
                                                                   strideA,
                                                                   strideB,
                                                                   strideC,
                                                                   strideD,
                                                                   *rocepilogue,
                                                                   *rocepinputs,
                                                                   *rocproblemtype,
                                                                   m_data,
                                                                   m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types[0] = problemtype;
        }
        return status;
    }

    hipblasStatus_t Gemm::setProblem(int64_t            m,
                                     int64_t            n,
                                     int64_t            k,
                                     int64_t            batch_count,
                                     int64_t            lda,
                                     int64_t            ldb,
                                     int64_t            ldc,
                                     int64_t            ldd,
                                     int64_t            strideA,
                                     int64_t            strideB,
                                     int64_t            strideC,
                                     int64_t            strideD,
                                     GemmEpilogueV2&    epilogue,
                                     GemmInputsV2&      inputs,
                                     GemmProblemTypeV2& problemtype)
    {
        auto rocepilogue    = reinterpret_cast<rocblaslt::RocGemmEpilogueV2*>(epilogue.pimpl.get());
        auto rocepinputs    = reinterpret_cast<rocblaslt::RocGemmInputsV2*>(inputs.pimpl.get());
        auto rocproblemtype = reinterpret_cast<rocblaslt::RocGemmProblemTypeV2*>(problemtype.pimpl.get());
        auto status
            = RocBlasLtStatusToHIPStatus(rocblaslt_gemm_create_cpp((rocblaslt_handle)m_handle,
                                                                   m,
                                                                   n,
                                                                   batch_count,
                                                                   k,
                                                                   lda,
                                                                   ldb,
                                                                   ldc,
                                                                   ldd,
                                                                   strideA,
                                                                   strideB,
                                                                   strideC,
                                                                   strideD,
                                                                   *rocepilogue,
                                                                   *rocepinputs,
                                                                   *rocproblemtype,
                                                                   m_data,
                                                                   m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types[0] = GemmProblemType{problemtype.getOpA(),
                                                 problemtype.getOpB(),
                                                 problemtype.getTypeA(),
                                                 problemtype.getTypeB(),
                                                 problemtype.getTypeC(),
                                                 problemtype.getTypeD(),
                                                 problemtype.getTypeCompute()};
        }
        return status;
    }

    hipblasStatus_t Gemm::setProblem(hipblasLtMatmulDesc_t   matmul_descr,
                                     const void*             alpha,
                                     const void*             A,
                                     hipblasLtMatrixLayout_t matA,
                                     const void*             B,
                                     hipblasLtMatrixLayout_t matB,
                                     const void*             beta,
                                     const void*             C,
                                     hipblasLtMatrixLayout_t matC,
                                     void*                   D,
                                     hipblasLtMatrixLayout_t matD)
    {
        auto rocproblemtypes
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemType>*>(&m_problem_types);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_gemm_create_cpp((rocblaslt_handle)m_handle,
                                      (rocblaslt_matmul_desc)matmul_descr,
                                      alpha,
                                      A,
                                      (rocblaslt_matrix_layout)matA,
                                      B,
                                      (rocblaslt_matrix_layout)matB,
                                      beta,
                                      C,
                                      (rocblaslt_matrix_layout)matC,
                                      D,
                                      (rocblaslt_matrix_layout)matD,
                                      (*rocproblemtypes)[0],
                                      m_data,
                                      m_gemm_count));
    }

    GemmProblemType Gemm::getProblemTypes()
    {
        return m_problem_types[0];
    }

    GemmProblemTypeV2 Gemm::getProblemTypesV2()
    {
        GemmProblemTypeV2 problemtype(m_problem_types[0].op_a,
                                      m_problem_types[0].op_b,
                                      m_problem_types[0].type_a,
                                      m_problem_types[0].type_b,
                                      m_problem_types[0].type_c,
                                      m_problem_types[0].type_d,
                                      m_problem_types[0].type_compute);
        return problemtype;
    }

    HIPBLASLT_EXPORT GroupedGemm::GroupedGemm(hipblasLtHandle_t    handle,
                                              hipblasOperation_t   opA,
                                              hipblasOperation_t   opB,
                                              hipDataType          typeA,
                                              hipDataType          typeB,
                                              hipDataType          typeC,
                                              hipDataType          typeD,
                                              hipblasComputeType_t typeCompute)
        : GemmInstance(handle, GemmType::HIPBLASLT_GROUPED_GEMM)
    {
        m_problem_types.push_back({opA, opB, typeA, typeB, typeC, typeD, typeCompute});
        rocblaslt_init_gemmData((rocblaslt_handle)m_handle,
                                static_cast<rocblaslt::RocGemmType>(m_gemm_type),
                                opA,
                                opB,
                                typeA,
                                typeB,
                                typeC,
                                typeD,
                                (rocblaslt_compute_type)typeCompute,
                                0,
                                m_data);
    }

    GroupedGemm::GroupedGemm(GroupedGemm&&) noexcept            = default;
    GroupedGemm& GroupedGemm::operator=(GroupedGemm&&) noexcept = default;

    HIPBLASLT_EXPORT GroupedGemm::GroupedGemm(hipblasLtHandle_t                     handle,
                                              std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                              std::vector<void*>&                   alpha,
                                              std::vector<void*>&                   A,
                                              std::vector<hipblasLtMatrixLayout_t>& matA,
                                              std::vector<void*>&                   B,
                                              std::vector<hipblasLtMatrixLayout_t>& matB,
                                              std::vector<void*>&                   beta,
                                              std::vector<void*>&                   C,
                                              std::vector<hipblasLtMatrixLayout_t>& matC,
                                              std::vector<void*>&                   D,
                                              std::vector<hipblasLtMatrixLayout_t>& matD)
        : GemmInstance(handle, GemmType::HIPBLASLT_GROUPED_GEMM)
    {
        auto status = setProblem(matmul_descr, alpha, A, matA, B, matB, beta, C, matC, D, matD);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cout << "Failed to create instance " << status << std::endl;
        }
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&      m,
                                            std::vector<int64_t>&      n,
                                            std::vector<int64_t>&      k,
                                            std::vector<int64_t>&      batch_count,
                                            std::vector<GemmEpilogue>& epilogue,
                                            std::vector<GemmInputs>&   inputs)
    {
        std::vector<int64_t> lda;
        std::vector<int64_t> ldb;
        std::vector<int64_t> ldc;
        std::vector<int64_t> ldd;
        std::vector<int64_t> strideA;
        std::vector<int64_t> strideB;
        std::vector<int64_t> strideC;
        std::vector<int64_t> strideD;
        for(size_t i = 0; i < m.size(); i++)
        {
            size_t iIdx = m_problem_types.size() == 1 ? 0 : i;
            lda.push_back(m_problem_types[iIdx].op_a == HIPBLAS_OP_N ? m[i] : k[i]);
            ldb.push_back(m_problem_types[iIdx].op_b == HIPBLAS_OP_N ? k[i] : n[i]);
            ldc.push_back(m[i]);
            ldd.push_back(m[i]);
            strideA.push_back(m[i] * k[i]);
            strideB.push_back(m[i] * k[i]);
            strideC.push_back(m[i] * k[i]);
            strideD.push_back(m[i] * k[i]);
        }
        return setProblem(m,
                          n,
                          k,
                          batch_count,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          strideA,
                          strideB,
                          strideC,
                          strideD,
                          epilogue,
                          inputs,
                          m_problem_types[0]);
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&        m,
                                            std::vector<int64_t>&        n,
                                            std::vector<int64_t>&        k,
                                            std::vector<int64_t>&        batch_count,
                                            std::vector<GemmEpilogueV2>& epilogue,
                                            std::vector<GemmInputsV2>&   inputs)
    {
        std::vector<int64_t> lda;
        std::vector<int64_t> ldb;
        std::vector<int64_t> ldc;
        std::vector<int64_t> ldd;
        std::vector<int64_t> strideA;
        std::vector<int64_t> strideB;
        std::vector<int64_t> strideC;
        std::vector<int64_t> strideD;
        for(size_t i = 0; i < m.size(); i++)
        {
            size_t iIdx = m_problem_types.size() == 1 ? 0 : i;
            lda.push_back(m_problem_types[iIdx].op_a == HIPBLAS_OP_N ? m[i] : k[i]);
            ldb.push_back(m_problem_types[iIdx].op_b == HIPBLAS_OP_N ? k[i] : n[i]);
            ldc.push_back(m[i]);
            ldd.push_back(m[i]);
            strideA.push_back(m[i] * k[i]);
            strideB.push_back(m[i] * k[i]);
            strideC.push_back(m[i] * k[i]);
            strideD.push_back(m[i] * k[i]);
        }
        GemmProblemTypeV2 prob(m_problem_types[0].op_a,
                               m_problem_types[0].op_b,
                               m_problem_types[0].type_a,
                               m_problem_types[0].type_b,
                               m_problem_types[0].type_c,
                               m_problem_types[0].type_d,
                               m_problem_types[0].type_compute);
        return setProblem(m,
                          n,
                          k,
                          batch_count,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          strideA,
                          strideB,
                          strideC,
                          strideD,
                          epilogue,
                          inputs,
                          prob);
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&      m,
                                            std::vector<int64_t>&      n,
                                            std::vector<int64_t>&      k,
                                            std::vector<int64_t>&      batch_count,
                                            std::vector<int64_t>&      lda,
                                            std::vector<int64_t>&      ldb,
                                            std::vector<int64_t>&      ldc,
                                            std::vector<int64_t>&      ldd,
                                            std::vector<int64_t>&      strideA,
                                            std::vector<int64_t>&      strideB,
                                            std::vector<int64_t>&      strideC,
                                            std::vector<int64_t>&      strideD,
                                            std::vector<GemmEpilogue>& epilogue,
                                            std::vector<GemmInputs>&   inputs,
                                            GemmProblemType&           problemtype)
    {
        auto rocepilogue = reinterpret_cast<std::vector<rocblaslt::RocGemmEpilogue>*>(&epilogue);
        auto rocinputs   = reinterpret_cast<std::vector<rocblaslt::RocGemmInputs>*>(&inputs);
        std::vector<GemmProblemType> tmptype = {problemtype};
        auto                         rocproblemtype
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemType>*>(&tmptype);
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_groupedgemm_create_cpp((rocblaslt_handle)m_handle,
                                             m,
                                             n,
                                             batch_count,
                                             k,
                                             lda,
                                             ldb,
                                             ldc,
                                             ldd,
                                             strideA,
                                             strideB,
                                             strideC,
                                             strideD,
                                             *rocepilogue,
                                             *rocinputs,
                                             *rocproblemtype,
                                             m_data,
                                             m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types = tmptype;
        }
        return status;
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<int64_t>&        m,
                                            std::vector<int64_t>&        n,
                                            std::vector<int64_t>&        k,
                                            std::vector<int64_t>&        batch_count,
                                            std::vector<int64_t>&        lda,
                                            std::vector<int64_t>&        ldb,
                                            std::vector<int64_t>&        ldc,
                                            std::vector<int64_t>&        ldd,
                                            std::vector<int64_t>&        strideA,
                                            std::vector<int64_t>&        strideB,
                                            std::vector<int64_t>&        strideC,
                                            std::vector<int64_t>&        strideD,
                                            std::vector<GemmEpilogueV2>& epilogue,
                                            std::vector<GemmInputsV2>&   inputs,
                                            GemmProblemTypeV2&           problemtype)
    {
        std::vector<rocblaslt::RocGemmEpilogueV2> rocepilogue;
        for(auto& e : epilogue)
        {
            rocepilogue.push_back(*reinterpret_cast<rocblaslt::RocGemmEpilogueV2*>(e.pimpl.get()));
        }

        std::vector<rocblaslt::RocGemmInputsV2> rocinputs;
        for(auto& i : inputs)
        {
            rocinputs.push_back(*reinterpret_cast<rocblaslt::RocGemmInputsV2*>(i.pimpl.get()));
        }
        GemmProblemTypeV2 tmp = problemtype;
        std::vector<rocblaslt::RocGemmProblemTypeV2> rocproblemtype = {*reinterpret_cast<rocblaslt::RocGemmProblemTypeV2*>(tmp.pimpl.get())};
        auto status = RocBlasLtStatusToHIPStatus(
            rocblaslt_groupedgemm_create_cpp((rocblaslt_handle)m_handle,
                                             m,
                                             n,
                                             batch_count,
                                             k,
                                             lda,
                                             ldb,
                                             ldc,
                                             ldd,
                                             strideA,
                                             strideB,
                                             strideC,
                                             strideD,
                                             rocepilogue,
                                             rocinputs,
                                             rocproblemtype,
                                             m_data,
                                             m_gemm_count));
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            m_problem_types[0] = GemmProblemType{problemtype.getOpA(),
                                                 problemtype.getOpB(),
                                                 problemtype.getTypeA(),
                                                 problemtype.getTypeB(),
                                                 problemtype.getTypeC(),
                                                 problemtype.getTypeD(),
                                                 problemtype.getTypeCompute()};
        }
        return status;
    }

    hipblasStatus_t GroupedGemm::setProblem(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                            std::vector<void*>&                   alpha,
                                            std::vector<void*>&                   A,
                                            std::vector<hipblasLtMatrixLayout_t>& matA,
                                            std::vector<void*>&                   B,
                                            std::vector<hipblasLtMatrixLayout_t>& matB,
                                            std::vector<void*>&                   beta,
                                            std::vector<void*>&                   C,
                                            std::vector<hipblasLtMatrixLayout_t>& matC,
                                            std::vector<void*>&                   D,
                                            std::vector<hipblasLtMatrixLayout_t>& matD)
    {
        auto matmul_descr_groupedGemm
            = reinterpret_cast<std::vector<rocblaslt_matmul_desc>*>(&matmul_descr);
        auto matA_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matA);
        auto matB_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matB);
        auto matC_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matC);
        auto matD_groupedGemm  = reinterpret_cast<std::vector<rocblaslt_matrix_layout>*>(&matD);
        auto A_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&A);
        auto B_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&B);
        auto C_groupedGemm     = reinterpret_cast<std::vector<const void*>*>(&C);
        auto alpha_groupedGemm = reinterpret_cast<std::vector<const void*>*>(&alpha);
        auto beta_groupedGemm  = reinterpret_cast<std::vector<const void*>*>(&beta);
        auto rocproblemtypes
            = reinterpret_cast<std::vector<rocblaslt::RocGemmProblemType>*>(&m_problem_types);
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_groupedgemm_create_cpp((rocblaslt_handle)m_handle,
                                             *matmul_descr_groupedGemm,
                                             *alpha_groupedGemm,
                                             *A_groupedGemm,
                                             *matA_groupedGemm,
                                             *B_groupedGemm,
                                             *matB_groupedGemm,
                                             *beta_groupedGemm,
                                             *C_groupedGemm,
                                             *matC_groupedGemm,
                                             D,
                                             *matD_groupedGemm,
                                             (*rocproblemtypes),
                                             m_data,
                                             m_gemm_count));
    }

    std::vector<GemmProblemType> GroupedGemm::getProblemTypes()
    {
        return m_problem_types;
    }

    std::vector<GemmProblemTypeV2> GroupedGemm::getProblemTypesV2()
    {
        std::vector<GemmProblemTypeV2> problemtype(m_problem_types.size());
        for(size_t i = 0; i < problemtype.size(); i++)
        {
            problemtype[i] = GemmProblemTypeV2(m_problem_types[i].op_a,
                                               m_problem_types[i].op_b,
                                               m_problem_types[i].type_a,
                                               m_problem_types[i].type_b,
                                               m_problem_types[i].type_c,
                                               m_problem_types[i].type_d,
                                               m_problem_types[i].type_compute);
        }
        return problemtype;
    }

    HIPBLASLT_EXPORT hipblasStatus_t
        GroupedGemm::getDefaultValueForDeviceUserArguments(void* hostDeviceUserArgs)
    {
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return RocBlasLtStatusToHIPStatus(rocblaslt_get_default_user_args(
            (rocblaslt_handle)m_handle, gemmType, m_data, hostDeviceUserArgs));
    }

    HIPBLASLT_EXPORT hipblasStatus_t GroupedGemm::run(void* deviceUserArgs, hipStream_t stream)
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        return RocBlasLtStatusToHIPStatus(rocblaslt_run_user_args_cpp(
            (rocblaslt_handle)m_handle, gemmType, m_data, deviceUserArgs, stream));
    }

    hipblasStatus_t matmulIsAlgoSupported(hipblasLtHandle_t       handle,
                                          hipblasLtMatmulDesc_t   matmulDesc,
                                          const void*             alpha,
                                          hipblasLtMatrixLayout_t Adesc,
                                          hipblasLtMatrixLayout_t Bdesc,
                                          const void*             beta,
                                          hipblasLtMatrixLayout_t Cdesc,
                                          hipblasLtMatrixLayout_t Ddesc,
                                          hipblasLtMatmulAlgo_t&  algo,
                                          size_t&                 workspaceSizeInBytes)
    try
    {
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_matmul_is_algo_supported((rocblaslt_handle)handle,
                                               (rocblaslt_matmul_desc)matmulDesc,
                                               alpha,
                                               (rocblaslt_matrix_layout)Adesc,
                                               (rocblaslt_matrix_layout)Bdesc,
                                               beta,
                                               (rocblaslt_matrix_layout)Cdesc,
                                               (rocblaslt_matrix_layout)Ddesc,
                                               (rocblaslt_matmul_algo*)&algo,
                                               &workspaceSizeInBytes));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    std::string gemmType2String(GemmType type)
    {
        switch(type)
        {
        case GemmType::HIPBLASLT_GEMM:
            return "gemm";
        case GemmType::HIPBLASLT_GROUPED_GEMM:
            return "grouped gemm";
        }
    }

    hipblasStatus_t getAllAlgos(hipblasLtHandle_t                              handle,
                                GemmType                                       typeGemm,
                                hipblasOperation_t                             opA,
                                hipblasOperation_t                             opB,
                                hipDataType                                    typeA,
                                hipDataType                                    typeB,
                                hipDataType                                    typeC,
                                hipDataType                                    typeD,
                                hipblasComputeType_t                           typeCompute,
                                std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    try
    {
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_matmul_get_all_algos_cpp((rocblaslt_handle)handle,
                                               static_cast<rocblaslt::RocGemmType>(typeGemm),
                                               opA,
                                               opB,
                                               typeA,
                                               typeB,
                                               typeC,
                                               typeD,
                                               (rocblaslt_compute_type)typeCompute,
                                               *results));
    }
    catch(...)
    {
        return exception_to_hipblas_status();
    }

    int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo)
    {
        int* algo_ptr = (int*)algo.data;
        if(*algo_ptr < 0)
        {
            return -1;
        }
        return *algo_ptr;
    }

    std::string getSolutionNameFromAlgo(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo)
    {
        int* algo_ptr = (int*)algo.data;
        if(*algo_ptr < 0)
        {
            return "";
        }
        auto rocalgo = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        return rocblaslt_get_solution_name_from_algo((rocblaslt_handle)handle, *rocalgo);
    }

    std::string getKernelNameFromAlgo(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo)
    {
        int* algo_ptr = (int*)algo.data;
        if(*algo_ptr < 0)
        {
            return "";
        }
        auto rocalgo = reinterpret_cast<const rocblaslt_matmul_algo*>(&algo);
        return rocblaslt_get_kernel_name_from_algo((rocblaslt_handle)handle, *rocalgo);
    }

    hipblasStatus_t
        getAlgosFromIndex(hipblasLtHandle_t                              handle,
                          std::vector<int>&                              algoIndex,
                          std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults)
    {
        auto results
            = reinterpret_cast<std::vector<rocblaslt_matmul_heuristic_result>*>(&heuristicResults);
        results->clear();
        return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_get_algos_from_index_cpp(
            (rocblaslt_handle)handle, algoIndex, *results));
    }

    hipblasStatus_t copyMatmul(hipblasLtMatmulDesc_t src, hipblasLtMatmulDesc_t dst)
    {
        return RocBlasLtStatusToHIPStatus(
            rocblaslt_copy_matmul((rocblaslt_matmul_desc)src, (rocblaslt_matmul_desc)dst));
    }

    int matmulIsTuned(hipblasLtHandle_t       handle,
                      hipblasLtMatmulDesc_t   matmulDesc,
                      hipblasLtMatrixLayout_t Adesc,
                      hipblasLtMatrixLayout_t Bdesc,
                      hipblasLtMatrixLayout_t Cdesc,
                      hipblasLtMatrixLayout_t Ddesc)
    {
        return rocblaslt_matmul_is_tuned((rocblaslt_handle)handle,
            (rocblaslt_matmul_desc)matmulDesc,
            (rocblaslt_matrix_layout)Adesc,
            (rocblaslt_matrix_layout)Bdesc,
            (rocblaslt_matrix_layout)Cdesc,
            (rocblaslt_matrix_layout)Ddesc);
    }

} // End of namespace hipblasltext
