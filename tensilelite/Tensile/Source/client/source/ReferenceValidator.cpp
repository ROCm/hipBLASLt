/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ReferenceValidator.hpp"
#include "ResultComparison.hpp"
#include "ResultReporter.hpp"

#include "Reference.hpp"

#include <Tensile/DataTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        ReferenceValidator::ReferenceValidator(po::variables_map const&            args,
                                               std::shared_ptr<DataInitialization> dataInit)
            : m_dataInit(dataInit)
        {
            m_elementsToValidate = args["num-elements-to-validate"].as<int>();
            m_printValids        = args["print-valids"].as<bool>();
            m_printMax           = args["print-max"].as<int>();

            m_printTensorA   = args["print-tensor-a"].as<bool>();
            m_printTensorB   = args["print-tensor-b"].as<bool>();
            m_printTensorC   = args["print-tensor-c"].as<bool>();
            m_printTensorD   = args["print-tensor-d"].as<bool>();
            m_printTensorRef = args["print-tensor-ref"].as<bool>();

            m_printAny = m_printTensorA || m_printTensorB || m_printTensorC || m_printTensorD
                         || m_printTensorRef;

            m_enabled = m_elementsToValidate != 0 || m_printAny;
        }

        bool ReferenceValidator::needMoreBenchmarkRuns() const
        {
            if(m_enabled && m_numBenchmarkRuns == 0)
                return true;

            return false;
        }

        void ReferenceValidator::preBenchmarkRun() {}

        void ReferenceValidator::postBenchmarkRun()
        {
            m_numBenchmarkRuns++;
        }

        void ReferenceValidator::preProblem(ContractionProblemGemm const& problem)
        {
            if(m_enabled)
            {
                m_problem          = problem;
                m_referenceInputs  = m_dataInit->prepareCPUInputs(problem);
                m_validationStride = 1;
                if(m_elementsToValidate > 0
                   && m_elementsToValidate < problem.d().totalLogicalElements())
                    m_validationStride
                        = NextPrime(problem.d().totalAllocatedElements() / m_elementsToValidate);

                SolveCPU(problem, *m_referenceInputs, m_validationStride);
            }
        }

        void
            ReferenceValidator::preProblemGroupedGemm(ContractionProblemGroupedGemm const& problems)
        {
            if(m_enabled)
            {
                m_problems    = problems;
                m_groupedGemm = true;
                SolveCPUGroupedGemm(problems.gemms, *m_referenceInputs, m_validationStride);
            }
        }

        void ReferenceValidator::preSolution(ContractionSolution const& solution)
        {
            m_validatedSolution = false;
            m_errorInSolution   = false;
        }

        bool ReferenceValidator::needMoreRunsInSolution() const
        {
            if(m_enabled && !m_validatedSolution)
                return true;

            return false;
        }

        size_t ReferenceValidator::numWarmupRuns()
        {
            if(m_enabled && !m_validatedSolution)
                return 1;

            return 0;
        }

        void ReferenceValidator::setNumWarmupRuns(size_t count) {}

        void ReferenceValidator::preWarmup() {}

        void ReferenceValidator::postWarmup() {}

        bool ReferenceValidator::validateSolution(std::shared_ptr<ContractionInputs> inputs)
        {
            // retreive alpha/beta type set via setAlpha/BetaType()
            auto alphaType = m_problem.alphaType();
            auto betaType  = m_problem.betaType();
            auto biasType  = m_problem.biasType();

            // Backward-compatible: when setAlpha/BetaType() wasn't called, use the old way
            // Could remove after rocBLAS is updated
            if(alphaType == DataType::None)
            {
                alphaType = m_problem.a().dataType() == DataType::BFloat16
                                ? DataType::Float
                                : m_problem.d().dataType();
            }
            if(betaType == DataType::None)
            {
                betaType = alphaType;
            }

            auto const& typedReference = dynamic_cast<ContractionInputs const&>(*m_referenceInputs);
            auto const& typedResult    = dynamic_cast<ContractionInputs const&>(*inputs);

            auto rv = validate(typedReference, typedResult);

            return rv;
        }

        void ReferenceValidator::validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                                 TimingEvents const&                startEvents,
                                                 TimingEvents const&                stopEvents)
        {
            if(m_enabled && !m_validatedSolution)
            {
                validateSolution(inputs);
                m_validatedSolution = true;
            }
        }

        bool ReferenceValidator::validateTyped(TensorDescriptor const& tensor,
                                               void const*             refPtr,
                                               void const*             resPtr,
                                               size_t                  maxElements,
                                               bool                    isgpu)
        {
            bool rv = false;
            switch(tensor.dataType())
            {
            case DataType::Float:
            {
                rv = checkResultsTyped(
                    tensor, (float const*)refPtr, (float const*)resPtr, maxElements, isgpu);
            }
            break;
            case DataType::Double:
            {
                rv = checkResultsTyped(
                    tensor, (double const*)refPtr, (double const*)resPtr, maxElements, isgpu);
            }
            break;
            case DataType::ComplexFloat:
            {
                rv = checkResultsTyped(tensor,
                                       (std::complex<float> const*)refPtr,
                                       (std::complex<float> const*)resPtr,
                                       maxElements,
                                       isgpu);
            }
            break;
            case DataType::ComplexDouble:
            {
                rv = checkResultsTyped(tensor,
                                       (std::complex<double> const*)refPtr,
                                       (std::complex<double> const*)resPtr,
                                       maxElements,
                                       isgpu);
            }
            break;
            case DataType::Half:
            {
                rv = checkResultsTyped(
                    tensor, (Half const*)refPtr, (Half const*)resPtr, maxElements, isgpu);
            }
            break;
            case DataType::Int8x4:
            {
                throw std::runtime_error("Unsupported validator data type Int8x4 for output.");
            }
            break;
            case DataType::Int32:
            {
                rv = checkResultsTyped(
                    tensor, (int32_t const*)refPtr, (int32_t const*)resPtr, maxElements, isgpu);
            }
            break;
            case DataType::BFloat16:
            {
                rv = checkResultsTyped(
                    tensor, (BFloat16 const*)refPtr, (BFloat16 const*)resPtr, maxElements, isgpu);
            }
            break;
            case DataType::Int8:
            {
                rv = checkResultsTyped(
                    tensor, (int8_t const*)refPtr, (int8_t const*)resPtr, maxElements, isgpu);
            }
            break;
            default:
                throw std::runtime_error("Unsupported validator data type");
            }
            if(rv)
            {
                std::cout << "Check failed in output tensor: " << tensor << std::endl;
            }
            return rv;
        }

        bool ReferenceValidator::validate(ContractionInputs const& reference,
                                          ContractionInputs const& result)
        {
            bool rv = false;
            if(!m_enabled)
                return rv;

            if(m_printAny)
                printTensors(reference, result);

            if(m_elementsToValidate != 0)
            {
                // TODO: Combine ContractionProblemGroupedGemm and ContractionProblemGemm
                if(m_groupedGemm)
                {
                    for(size_t j = 0; j < m_problems.gemms.size(); j++)
                    {
                        for(size_t i = 0; i < m_problems.gemms[j].tensors().size(); i++)
                        {
                            auto& tensor = m_problems.gemms[j].tensors()[i];
                            if(!tensor.isOutput())
                                continue;

                            void const* refPtr = nullptr;
                            void const* resPtr = nullptr;
                            switch(static_cast<ContractionProblemGemm::TENSOR>(i))
                            {
                            case ContractionProblemGemm::TENSOR::A:
                            {
                                refPtr = reference.groupedA[j];
                                resPtr = result.groupedA[j];
                            }
                            break;
                            case ContractionProblemGemm::TENSOR::B:
                            {
                                refPtr = reference.groupedB[j];
                                resPtr = result.groupedB[j];
                            }
                            break;
                            case ContractionProblemGemm::TENSOR::C:
                            {
                                refPtr = reference.groupedC[j];
                                resPtr = result.groupedC[j];
                            }
                            break;
                            case ContractionProblemGemm::TENSOR::D:
                            {
                                refPtr = reference.groupedD[j];
                                resPtr = result.groupedD[j];
                            }
                            break;
                            case ContractionProblemGemm::TENSOR::E:
                            {
                                refPtr = reference.groupedE[j];
                                resPtr = result.groupedE[j];
                            }
                            break;
                            case ContractionProblemGemm::TENSOR::BIAS:
                            {
                                refPtr = reference.groupedBias[j];
                                resPtr = result.groupedBias[j];
                            }
                            break;
                            case ContractionProblemGemm::TENSOR::SCALED:
                            {
                                refPtr = reference.groupedScaleD[j];
                                resPtr = result.groupedScaleD[j];
                            }
                            break;
                            default:
                                throw std::runtime_error("Unrecognized output tensor.");
                            }

                            rv = validateTyped(tensor,
                                               refPtr,
                                               resPtr,
                                               result.groupedMaxElements[j][i],
                                               result.gpu);
                        }
                    }
                    return rv;
                }

                for(size_t i = 0; i < m_problem.tensors().size(); i++)
                {
                    auto& tensor = m_problem.tensors()[i];
                    if(!tensor.isOutput())
                        continue;

                    void const* refPtr = nullptr;
                    void const* resPtr = nullptr;
                    if(dynamic_cast<ContractionProblemGemm*>(&m_problem))
                    {
                        switch(static_cast<ContractionProblemGemm::TENSOR>(i))
                        {
                        case ContractionProblemGemm::TENSOR::A:
                        {
                            refPtr = reference.a;
                            resPtr = result.a;
                        }
                        break;
                        case ContractionProblemGemm::TENSOR::B:
                        {
                            refPtr = reference.b;
                            resPtr = result.b;
                        }
                        break;
                        case ContractionProblemGemm::TENSOR::C:
                        {
                            refPtr = reference.c;
                            resPtr = result.c;
                        }
                        break;
                        case ContractionProblemGemm::TENSOR::D:
                        {
                            refPtr = reference.d;
                            resPtr = result.d;
                        }
                        break;
                        case ContractionProblemGemm::TENSOR::E:
                        {
                            refPtr = reference.e;
                            resPtr = result.e;
                        }
                        break;
                        case ContractionProblemGemm::TENSOR::BIAS:
                        {
                            refPtr = reference.bias;
                            resPtr = result.bias;
                        }
                        break;
                        case ContractionProblemGemm::TENSOR::SCALED:
                        {
                            refPtr = reference.scaleD;
                            resPtr = result.scaleD;
                        }
                        break;
                        default:
                            throw std::runtime_error("Unrecognized output tensor.");
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Failed to cast problem to one of the tensors.");
                    }

                    rv = validateTyped(tensor, refPtr, resPtr, result.maxElements[i], result.gpu);
                }
            }

            return rv;
        }

        void ReferenceValidator::allocateResultBuffer(size_t bytes)
        {
            if(m_cpuResultBufferSize == bytes)
                return;
            m_cpuResultBuffer.reset();

            uint8_t* buffer;
            HIP_CHECK_EXC(hipHostMalloc(&buffer, bytes, 0));
            m_cpuResultBuffer.reset(buffer, hipFree);
            m_cpuResultBufferSize = bytes;
        }

        void ReferenceValidator::printTensors(ContractionInputs const& reference,
                                              ContractionInputs const& result)
        {
            size_t requiredBufferSize = 0;

            std::cout << "reference alpha: " << ToString(reference.alpha)
                      << ", beta: " << ToString(reference.beta) << std::endl;
            std::cout << "result    alpha: " << ToString(result.alpha)
                      << ", beta: " << ToString(result.beta) << std::endl;

            if(m_printTensorA)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.a().totalAllocatedBytes());
            if(m_printTensorB)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.b().totalAllocatedBytes());
            if(m_printTensorC)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.c().totalAllocatedBytes());
            if(m_printTensorD)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.d().totalAllocatedBytes());
            if(m_printTensorRef)
                requiredBufferSize
                    = std::max(requiredBufferSize, m_problem.d().totalAllocatedBytes());

            if(m_cpuResultBufferSize < requiredBufferSize)
                allocateResultBuffer(requiredBufferSize);

            if(m_printTensorA)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.a,
                                        m_problem.a().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(
                    LogLevel::Verbose, "A", m_cpuResultBuffer.get(), m_problem.a(), result.a);
            }

            if(m_printTensorB)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.b,
                                        m_problem.b().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(
                    LogLevel::Verbose, "B", m_cpuResultBuffer.get(), m_problem.b(), result.b);
            }

            if(result.c == result.d && (m_printTensorC || m_printTensorD))
            {
                // If the pointers are the same, only print the buffer once.
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                        result.c,
                                        m_problem.c().totalAllocatedBytes(),
                                        hipMemcpyDeviceToHost));
                m_reporter->logTensor(
                    LogLevel::Verbose, "C_D", m_cpuResultBuffer.get(), m_problem.c(), result.c);
            }
            else
            {
                if(m_printTensorC)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                            result.c,
                                            m_problem.c().totalAllocatedBytes(),
                                            hipMemcpyDeviceToHost));
                    m_reporter->logTensor(
                        LogLevel::Verbose, "C", m_cpuResultBuffer.get(), m_problem.c(), result.c);
                }

                if(m_printTensorD)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.get(),
                                            result.d,
                                            m_problem.d().totalAllocatedBytes(),
                                            hipMemcpyDeviceToHost));
                    m_reporter->logTensor(
                        LogLevel::Verbose, "D", m_cpuResultBuffer.get(), m_problem.d(), result.d);
                }
            }

            if(m_printTensorRef)
            {
                m_reporter->logTensor(
                    LogLevel::Verbose, "Ref", reference.d, m_problem.d(), reference.d);
            }
        }

        template <typename ValidType>
        bool ReferenceValidator::checkResultsTyped(TensorDescriptor const& tensor,
                                                   ValidType const*        reference,
                                                   ValidType const*        result,
                                                   size_t                  maxElement,
                                                   bool                    isgpu)
        {
            PointwiseComparison<ValidType> compareValid(m_printValids, m_printMax, m_printMax > 0);
            InvalidComparison<ValidType>   compareInvalid(m_printMax, m_printMax > 0);

            size_t elementsToCopy       = tensor.totalAllocatedElements();
            size_t elementsOffsetToCopy = 0;
            size_t elementsBeforeData   = 0;
            size_t elementsAfterData    = 0;

            BoundsCheckMode boundsCheck = m_dataInit->getCurBoundsCheck();
            if(boundsCheck == BoundsCheckMode::NaN)
                elementsToCopy = maxElement;
            size_t bytesToCopy = elementsToCopy * sizeof(ValidType);

            if(m_cpuResultBufferSize < bytesToCopy)
                allocateResultBuffer(bytesToCopy);

            if(boundsCheck == BoundsCheckMode::GuardPageBack)
                elementsOffsetToCopy = (maxElement - tensor.totalAllocatedElements());

            auto copykind = isgpu ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;

            HIP_CHECK_EXC(hipMemcpy(
                m_cpuResultBuffer.get(), result + elementsOffsetToCopy, bytesToCopy, copykind));

            if(boundsCheck == BoundsCheckMode::NaN)
            {
                ptrdiff_t bPadding = maxElement - tensor.totalAllocatedElements();
                elementsBeforeData = bPadding / 2;
                elementsAfterData
                    = elementsToCopy - (tensor.totalAllocatedElements() + elementsBeforeData);
            }
            // If there was extra data allocated before the tensor to do bounds
            // checking, resultBuffer is the whole allocation, while resultData
            // points directly to the result.
            ValidType const* resultBuffer
                = reinterpret_cast<ValidType const*>(m_cpuResultBuffer.get());
            ValidType const* resultData      = resultBuffer + elementsBeforeData;
            ValidType const* resultAfterData = resultData + tensor.totalAllocatedElements();

            size_t boundsCheckElements = 0;

            for(ptrdiff_t i = 0; i < elementsBeforeData; i++)
            {
                boundsCheckElements++;
                compareInvalid.before(resultBuffer[i], i, elementsBeforeData);
            }

            if(m_validationStride == 1)
            {
                std::vector<size_t> coord(tensor.dimensions());
                size_t outerCount = CoordCount(tensor.sizes().begin() + 1, tensor.sizes().end());

                size_t       prevBaseIndex = 0;
                const size_t innerDimSize  = tensor.sizes()[0];
                const size_t initialStride = tensor.strides()[0];

                for(size_t i = 0; i < outerCount; i++)
                {
                    CoordNumbered(i,
                                  coord.begin() + 1,
                                  coord.end(),
                                  tensor.sizes().begin() + 1,
                                  tensor.sizes().end());
                    size_t baseElemIndex = tensor.index(coord);

                    if(boundsCheck == BoundsCheckMode::NaN && baseElemIndex != 0
                       && baseElemIndex != prevBaseIndex + innerDimSize)
                    {
                        for(auto innerIndex = prevBaseIndex + innerDimSize;
                            innerIndex < baseElemIndex;
                            innerIndex++)
                        {
                            compareInvalid.inside(
                                resultData[innerIndex], innerIndex, baseElemIndex);
                        }
                    }

                    prevBaseIndex = baseElemIndex;

                    for(size_t j = 0; j < innerDimSize; j++)
                    {
                        size_t elemIndex = baseElemIndex + (j * initialStride);

                        ValidType referenceValue = reference[elemIndex];
                        ValidType resultValue    = resultData[elemIndex];

                        compareValid(
                            referenceValue, resultValue, elemIndex, (i * tensor.sizes()[0]) + j);
                    }
                }
            }
            else
            {
                std::vector<size_t> coord(tensor.dimensions());
                for(size_t elemNumber = 0; elemNumber < tensor.totalLogicalElements();
                    elemNumber += m_validationStride)
                {
                    CoordNumbered(elemNumber,
                                  coord.begin(),
                                  coord.end(),
                                  tensor.sizes().begin(),
                                  tensor.sizes().end());
                    size_t elemIndex = tensor.index(coord);

                    ValidType referenceValue = reference[elemIndex];
                    ValidType resultValue    = resultData[elemIndex];

                    compareValid(referenceValue, resultValue, elemIndex, elemNumber);
                }
            }

            for(ptrdiff_t i = 0; i < elementsAfterData; i++)
            {
                compareInvalid.after(resultAfterData[i], i, elementsAfterData);
            }

            if(boundsCheckElements > 0)
                std::cout << "Performed bounds check on " << boundsCheckElements << " elements ("
                          << elementsBeforeData << " before data)" << std::endl;

            compareValid.report();
            compareInvalid.report();

            if(compareValid.error() || compareInvalid.error())
            {
                m_errorInSolution = true;
                m_error           = true;

                return true;
            }

            return false;
        }

        void ReferenceValidator::postSolution()
        {
            if(m_enabled && !m_validatedSolution)
                return;

            if(m_elementsToValidate != 0)
            {
                if(m_errorInSolution)
                {
                    m_errorsReported++;
                    m_reporter->report(ResultKey::Validation, "FAILED");
                }
                else
                    m_reporter->report(ResultKey::Validation, "PASSED");
            }
            else
            {
                m_reporter->report(ResultKey::Validation, "NO_CHECK");
            }

            m_errorInSolution = false;
        }

        void ReferenceValidator::postProblem() {}

        void ReferenceValidator::finalizeReport() {}

        int ReferenceValidator::error() const
        {
            return m_errorsReported;
        }
    } // namespace Client
} // namespace Tensile
