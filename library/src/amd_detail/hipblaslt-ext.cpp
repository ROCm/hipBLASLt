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

    struct GemmInstance::ConversionHelper
    {
        using Conversions = std::tuple<HipBufferPtr, //src
                                       HipBufferPtr, //dst
                                       hipDataType, //srcType
                                       hipDataType, //dstType
                                       std::size_t, //numElements
                                       HipBufferPtr>; //scale
        std::vector<std::vector<Conversions>> m_auxiliary_conversion_buffers;
        ConversionHelper(const std::vector<GemmProblemType>& problemTypes,
                         GemmInputs&                         inputs,
                         hipDataType                         conversionDType,
                         int64_t                             batchSize,
                         int64_t                             strideA,
                         int64_t                             strideB,
                         int64_t                             strideC,
                         int64_t                             strideD)
        {
            const auto numGemms = problemTypes.size();
            if(m_auxiliary_conversion_buffers.size() != numGemms)
            {
                m_auxiliary_conversion_buffers.resize(numGemms);
            }

            for(std::size_t j = 0; j < m_auxiliary_conversion_buffers.size(); ++j)
            {
                const std::vector<std::int64_t> sizes{strideA, strideB, strideC};
                const std::vector<void*>        gemmInputs{inputs.a, inputs.b, inputs.c};
                const std::vector<void*>        scales{inputs.scaleA, inputs.scaleB, inputs.scaleC};
                auto&                           conversions = m_auxiliary_conversion_buffers.at(j);
                auto&                           problem     = problemTypes.at(j);
                const std::vector<hipDataType>  dtypes{
                    problem.type_a, problem.type_b, problem.type_c};

                //a, b and c
                for(std::size_t i = 0; i < sizes.size(); ++i)
                {
                    auto       dtype       = dtypes.at(i);
                    const auto numElements = sizes.at(i);

                    if(dtype == HIP_R_8F_E4M3_FNUZ || dtype == HIP_R_8F_E5M2_FNUZ)
                    {
                        const auto numBytes = numElements * 2;
                        conversions.emplace_back(
                            std::make_tuple(std::move(HipBufferPtr(gemmInputs.at(i), NullDeleter)),
                                            std::move(makeHipBuffer(numBytes)),
                                            dtype,
                                            conversionDType,
                                            numElements,
                                            std::move(HipBufferPtr(scales.at(i), NullDeleter))));
                    }
                    else
                    {
                        conversions.emplace_back(
                            std::make_tuple(std::move(HipBufferPtr(gemmInputs.at(i), NullDeleter)),
                                            std::move(makeHipBuffer(0)),
                                            dtype,
                                            conversionDType,
                                            numElements,
                                            std::move(HipBufferPtr(scales.at(i), NullDeleter))));
                    }
                }

                //for d
                auto       output      = inputs.d;
                const auto numElements = strideD * batchSize;

                if(problem.type_d == HIP_R_8F_E4M3_FNUZ || problem.type_d == HIP_R_8F_E5M2_FNUZ)
                {
                    auto numBytes = numElements * 2;
                    conversions.emplace_back(
                        std::make_tuple(std::move(makeHipBuffer(numBytes)),
                                        std::move(HipBufferPtr(output, NullDeleter)),
                                        conversionDType,
                                        problem.type_d,
                                        numElements,
                                        std::move(HipBufferPtr(inputs.scaleD, NullDeleter))));
                }
                else
                {
                    conversions.emplace_back(
                        std::make_tuple(std::move(makeHipBuffer(0)),
                                        std::move(HipBufferPtr(output, NullDeleter)),
                                        conversionDType,
                                        problem.type_d,
                                        numElements,
                                        std::move(HipBufferPtr(inputs.scaleD, NullDeleter))));
                }
            }
        }

        ~ConversionHelper()                               = default;
        ConversionHelper(ConversionHelper&& rhs) noexcept = default;
        //force move
        ConversionHelper(const ConversionHelper& rhs)        = delete;
        ConversionHelper& operator=(const ConversionHelper&) = delete;
        ConversionHelper& operator=(ConversionHelper&&)      = default;

        void convertInputs(hipStream_t stream)
        {
            if(m_auxiliary_conversion_buffers.size())
            {
                for(auto& conversions : m_auxiliary_conversion_buffers)
                {
                    for(size_t i = 0; i < 3; ++i)
                    {
                        auto& conversion = conversions.at(i);
                        auto& dst        = std::get<1>(conversion);
                        auto& src        = std::get<0>(conversion);

                        if(src && dst)
                        {
                            auto           srcType           = std::get<2>(conversion);
                            auto           dstType           = std::get<3>(conversion);
                            const auto     numElements       = std::get<4>(conversion);
                            auto&          scale             = std::get<5>(conversion);
                            constexpr auto numWorkitemsPerWg = 256;
                            const auto     numWg             = (numElements / numWorkitemsPerWg)
                                               + !!(numElements % numWorkitemsPerWg);

                            if(srcType == HIP_R_8F_E4M3_FNUZ)
                            {
                                datatypeConversion<hipblaslt_f8_fnuz, hipblasLtHalf>
                                    <<<numWg, numWorkitemsPerWg, 0, stream>>>(
                                        (const hipblaslt_f8_fnuz*)src.get(),
                                        (hipblasLtHalf*)dst.get(),
                                        (const float*)scale.get(),
                                        numElements);
                            }
                            else if(srcType == HIP_R_8F_E5M2_FNUZ)
                            {
                                datatypeConversion<hipblaslt_bf8_fnuz, hipblasLtHalf>
                                    <<<numWg, numWorkitemsPerWg, 0, stream>>>(
                                        (const hipblaslt_bf8_fnuz*)src.get(),
                                        (hipblasLtHalf*)dst.get(),
                                        (const float*)scale.get(),
                                        numElements);
                            }
                        }
                    }
                }
            }
        }

        void convertOutputs(hipStream_t stream)
        {
            if(m_auxiliary_conversion_buffers.size())
            {
                for(auto& conversions : m_auxiliary_conversion_buffers)
                {
                    if(conversions.size() > 3)
                    {
                        auto&          conversion        = conversions.at(3);
                        auto&          src               = std::get<0>(conversion);
                        auto&          dst               = std::get<1>(conversion);
                        auto           srcType           = std::get<2>(conversion);
                        auto           dstType           = std::get<3>(conversion);
                        const auto     numElements       = std::get<4>(conversion);
                        auto&          scale             = std::get<5>(conversion);
                        constexpr auto numWorkitemsPerWg = 256;
                        const auto     numWg             = (numElements / numWorkitemsPerWg)
                                           + !!(numElements % numWorkitemsPerWg);
                        //indicates d needs datatype conversion
                        if(src && dst)
                        {
                            if(dstType == HIP_R_8F_E4M3_FNUZ)
                            {
                                datatypeConversion<hipblasLtHalf, hipblaslt_f8_fnuz>
                                    <<<numWg, numWorkitemsPerWg, 0, stream>>>(
                                        (const hipblasLtHalf*)src.get(),
                                        (hipblaslt_f8_fnuz*)dst.get(),
                                        (const float*)scale.get(),
                                        numElements);
                            }
                            else if(dstType == HIP_R_8F_E5M2_FNUZ)
                            {
                                datatypeConversion<hipblasLtHalf, hipblaslt_bf8_fnuz>
                                    <<<numWg, numWorkitemsPerWg, 0, stream>>>(
                                        (const hipblasLtHalf*)src.get(),
                                        (hipblaslt_bf8_fnuz*)dst.get(),
                                        (const float*)scale.get(),
                                        numElements);
                            }
                        }
                    }
                }
            }
        }
    };

    GemmInstance::GemmInstance(hipblasLtHandle_t handle, GemmType type)
        : m_gemm_type(type)
        , m_handle(handle)
    {
    }

    GemmInstance::~GemmInstance() {}
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

    hipblasStatus_t GemmInstance::run(hipStream_t stream)
    try
    {
        if(m_gemm_count == 0)
            return HIPBLAS_STATUS_INVALID_VALUE;

        if(m_conversion_helper)
        {
            m_conversion_helper->convertInputs(stream);
        }

        auto gemmType = static_cast<rocblaslt::RocGemmType>(m_gemm_type);
        auto status   = RocBlasLtStatusToHIPStatus(
            rocblaslt_run_cpp((rocblaslt_handle)m_handle, gemmType, m_data, stream));

        if(m_conversion_helper)
        {
            m_conversion_helper->convertOutputs(stream);
        }

        return status;
    }
    catch(...)
    {
        return exception_to_hipblas_status();
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

        int lda     = m_problem_types[0].op_a == HIPBLAS_OP_N ? m : k;
        int ldb     = m_problem_types[0].op_b == HIPBLAS_OP_N ? k : n;
        int ldc     = m;
        int strideA = m * k;
        int strideB = n * k;
        int strideC = m * n;
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

    hipblasStatus_t Gemm::setProblem(int64_t          m,
                                     int64_t          n,
                                     int64_t          k,
                                     int64_t          batch_count,
                                     int64_t          lda,
                                     int64_t          ldb,
                                     int64_t          ldc,
                                     int64_t          ldd,
                                     int64_t          strideA,
                                     int64_t          strideB,
                                     int64_t          strideC,
                                     int64_t          strideD,
                                     GemmEpilogue&    epilogue,
                                     GemmInputs&      inputs,
                                     GemmProblemType& problemtype)
    {
        constexpr auto conversionDType = HIP_R_16F;
        auto           needConversion  = [&problemtype]() -> bool {
            using std::begin;
            using std::end;
            const auto types = {problemtype.type_a, problemtype.type_b, problemtype.type_c};
            auto       mixedPrecision
                = end(types) != std::adjacent_find(begin(types), end(types), std::not_equal_to<>());
            return mixedPrecision && !currentArchSupportsFp8();
        }();

        if(needConversion)
        {
            m_conversion_helper = std::make_unique<ConversionHelper>(m_problem_types,
                                                                     inputs,
                                                                     conversionDType,
                                                                     batch_count,
                                                                     strideA,
                                                                     strideB,
                                                                     strideC,
                                                                     strideD);
        }

        //Shallow copy
        GemmInputs      gemmInputs      = inputs;
        GemmProblemType gemmProblemType = problemtype;
        auto&           problem         = m_problem_types.at(0);

        if(needConversion)
        {
            if(auto& a
               = std::get<1>(m_conversion_helper->m_auxiliary_conversion_buffers.at(0).at(0)))
            {
                gemmInputs.a           = a.get();
                gemmProblemType.type_a = conversionDType;
            }

            if(auto& b
               = std::get<1>(m_conversion_helper->m_auxiliary_conversion_buffers.at(0).at(1)))
            {
                gemmInputs.b           = b.get();
                gemmProblemType.type_b = conversionDType;
            }

            if(auto& c
               = std::get<1>(m_conversion_helper->m_auxiliary_conversion_buffers.at(0).at(2)))
            {
                gemmInputs.c           = c.get();
                gemmProblemType.type_c = conversionDType;
            }

            if(auto& d
               = std::get<0>(m_conversion_helper->m_auxiliary_conversion_buffers.at(0).at(3)))
            {
                gemmInputs.d           = d.get();
                gemmProblemType.type_d = conversionDType;
            }
        }

        auto rocepilogue    = reinterpret_cast<rocblaslt::RocGemmEpilogue*>(&epilogue);
        auto rocepinputs    = reinterpret_cast<rocblaslt::RocGemmInputs*>(&gemmInputs);
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

} // End of namespace hipblasltext
