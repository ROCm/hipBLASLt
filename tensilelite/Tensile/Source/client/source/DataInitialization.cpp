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

#include "DataInitialization.hpp"
// #include "DataInitializationTyped.hpp"

#include <Tensile/Utils.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>

namespace Tensile
{
    namespace Client
    {
        std::string ToString(InitMode mode)
        {
            switch(mode)
            {
            case InitMode::Zero:
                return "Zero";
            case InitMode::One:
                return "One";
            case InitMode::Two:
                return "Two";
            case InitMode::Random:
                return "Random";
            case InitMode::NaN:
                return "NaN";
            case InitMode::Inf:
                return "Inf";
            case InitMode::BadInput:
                return "BadInput";
            case InitMode::BadOutput:
                return "BadOutput";
            case InitMode::SerialIdx:
                return "SerialIdx";
            case InitMode::SerialDim0:
                return "SerialDim0";
            case InitMode::SerialDim1:
                return "SerialDim1";
            case InitMode::Identity:
                return "Identity";
            case InitMode::TrigSin:
                return "TrigSin";
            case InitMode::TrigCos:
                return "TrigCos";
            case InitMode::TrigAbsSin:
                return "TrigAbsSin";
            case InitMode::TrigAbsCos:
                return "TrigAbsCos";
            case InitMode::RandomNarrow:
                return "RandomNarrow";
            case InitMode::NegOne:
                return "NegOne";
            case InitMode::Max:
                return "Max";
            case InitMode::DenormMin:
                return "DenormMin";
            case InitMode::DenormMax:
                return "DenormMax";
            case InitMode::RandomNegPosLimited:
                return "RandomNegPosLimited";
            case InitMode::Free:
                return "Free";

            case InitMode::Count:
                break;
            }

            throw std::runtime_error(
                concatenate("Invalid InitMode value: ", static_cast<int>(mode)));
        }

        std::ostream& operator<<(std::ostream& stream, InitMode const& mode)
        {
            return stream << ToString(mode);
        }

        std::istream& operator>>(std::istream& stream, InitMode& mode)
        {
            std::string strValue;
            stream >> strValue;

            if(strValue == ToString(InitMode::Zero))
                mode = InitMode::Zero;
            else if(strValue == ToString(InitMode::One))
                mode = InitMode::One;
            else if(strValue == ToString(InitMode::Two))
                mode = InitMode::Two;
            else if(strValue == ToString(InitMode::Random))
                mode = InitMode::Random;
            else if(strValue == ToString(InitMode::NaN))
                mode = InitMode::NaN;
            else if(strValue == ToString(InitMode::Inf))
                mode = InitMode::Inf;
            else if(strValue == ToString(InitMode::BadInput))
                mode = InitMode::BadInput;
            else if(strValue == ToString(InitMode::BadOutput))
                mode = InitMode::BadOutput;
            else if(strValue == ToString(InitMode::SerialIdx))
                mode = InitMode::SerialIdx;
            else if(strValue == ToString(InitMode::SerialDim0))
                mode = InitMode::SerialDim0;
            else if(strValue == ToString(InitMode::SerialDim1))
                mode = InitMode::SerialDim1;
            else if(strValue == ToString(InitMode::Identity))
                mode = InitMode::Identity;
            else if(strValue == ToString(InitMode::TrigSin))
                mode = InitMode::TrigSin;
            else if(strValue == ToString(InitMode::TrigCos))
                mode = InitMode::TrigCos;
            else if(strValue == ToString(InitMode::TrigAbsSin))
                mode = InitMode::TrigAbsSin;
            else if(strValue == ToString(InitMode::TrigAbsCos))
                mode = InitMode::TrigAbsCos;
            else if(strValue == ToString(InitMode::RandomNarrow))
                mode = InitMode::RandomNarrow;
            else if(strValue == ToString(InitMode::NegOne))
                mode = InitMode::NegOne;
            else if(strValue == ToString(InitMode::Max))
                mode = InitMode::Max;
            else if(strValue == ToString(InitMode::DenormMin))
                mode = InitMode::DenormMin;
            else if(strValue == ToString(InitMode::DenormMax))
                mode = InitMode::DenormMax;
            else if(strValue == ToString(InitMode::RandomNegPosLimited))
                mode = InitMode::RandomNegPosLimited;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(InitMode::Count))
                    mode = static_cast<InitMode>(value);
                else
                    throw std::runtime_error(
                        concatenate("Can't convert ", strValue, " to InitMode."));
            }
            else
            {
                throw std::runtime_error(concatenate("Can't convert ", strValue, " to InitMode."));
            }

            return stream;
        }

        std::ostream& operator<<(std::ostream& stream, BoundsCheckMode const& mode)
        {
            std::string strValue;

            if(mode == BoundsCheckMode::Disable)
                strValue = "Disable";
            else if(mode == BoundsCheckMode::NaN)
                strValue = "NaN";
            else if(mode == BoundsCheckMode::GuardPageFront)
                strValue = "GuardPageFront";
            else if(mode == BoundsCheckMode::GuardPageBack)
                strValue = "GuardPageBack";
            else if(mode == BoundsCheckMode::GuardPageAll)
                strValue = "GuardPageAll";
            else
                throw std::runtime_error(
                    concatenate("Invalid BoundsCheckMode value: ", static_cast<int>(mode)));

            return stream << strValue;
        }

        std::istream& operator>>(std::istream& stream, BoundsCheckMode& mode)
        {
            std::string strValue;
            stream >> strValue;

            if(strValue == "Disable")
                mode = BoundsCheckMode::Disable;
            else if(strValue == "NaN")
                mode = BoundsCheckMode::NaN;
            else if(strValue == "GuardPageFront")
                mode = BoundsCheckMode::GuardPageFront;
            else if(strValue == "GuardPageBack")
                mode = BoundsCheckMode::GuardPageBack;
            else if(strValue == "GuardPageAll")
                mode = BoundsCheckMode::GuardPageAll;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(BoundsCheckMode::MaxMode))
                    mode = static_cast<BoundsCheckMode>(value);
                else
                    throw std::runtime_error(
                        concatenate("Can't convert ", strValue, " to BoundsCheckMode."));
            }
            else
            {
                throw std::runtime_error(
                    concatenate("Can't convert ", strValue, " to BoundsCheckMode."));
            }

            return stream;
        }

        template <typename T>
        std::shared_ptr<T> allocNewGPUBuffer(const char* title, size_t size)
        {
            static const int sizew = 10;
            T*               ptr   = nullptr;
            HIP_CHECK_EXC(hipMalloc(&ptr, size));
            auto p = std::shared_ptr<T>(ptr, hipFree);
            if(Debug::Instance().printTensorInfo())
                std::cout << "info: allocate " << title << " " << std::setw(sizew) << size
                          << " bytes at " << static_cast<void*>(ptr) << "\n";
            return p;
        }

        void initGPUBatchedInput(void*                      base,
                                 void**                     array,
                                 TensorDescriptor const&    tensor,
                                 const std::vector<size_t>& batchIdx)
        {
            std::vector<size_t> batchSizes;
            std::vector<size_t> batchStrides;
            for(auto& idx : batchIdx)
            {
                batchSizes.push_back(tensor.sizes().at(idx));
                batchStrides.push_back(tensor.strides().at(idx));
            }
            std::vector<size_t> coord(batchSizes.size(), 0);

            auto      count    = CoordCount(batchSizes.begin(), batchSizes.end());
            uint8_t** cpuArray = (uint8_t**)std::malloc(count * sizeof(void*));
            for(size_t idx = 0; idx < count; idx++)
            {
                CoordNumbered(
                    idx, coord.begin(), coord.end(), batchSizes.begin(), batchSizes.end());
                cpuArray[idx] = (uint8_t*)base;
                for(size_t i = 0; i < batchSizes.size(); i++)
                {
                    cpuArray[idx] += coord[i] * batchStrides[i];
                }
            }

            HIP_CHECK_EXC(hipMemcpy(array, cpuArray, count * sizeof(void*), hipMemcpyHostToDevice));

            std::free(cpuArray);
        }

        void* copyBadInputBuffers(const TensorDescriptor& descriptor,
                                  void*                   dst,
                                  void*                   src,
                                  void*                   bad,
                                  size_t                  totalElements,
                                  hipMemcpyKind           kind)
        {
            HIP_CHECK_EXC(
                hipMemcpy(dst,
                          src,
                          DataTypeInfo::Get(descriptor.dataType()).elementSize * totalElements,
                          kind));
            ptrdiff_t dPadding = totalElements - descriptor.totalAllocatedElements();
            dPadding *= descriptor.elementBytes();
            void* dstOffset = (void*)((uint8_t*)dst + dPadding / 2);
            Tensile::hip::CopyTensorVoid(dstOffset, src, descriptor, kind);
            return dstOffset;
        }

        void* copyNaNInputBuffers(const TensorDescriptor& descriptor,
                                  void*                   dst,
                                  void*                   src,
                                  size_t                  totalElements,
                                  hipMemcpyKind           kind)
        {
            ptrdiff_t dPadding  = totalElements - descriptor.totalAllocatedElements();
            uint8_t*  dstOffset = (uint8_t*)dst + (dPadding * descriptor.elementBytes());
            HIP_CHECK_EXC(hipMemcpy(dstOffset,
                                    src,
                                    descriptor.elementBytes() * descriptor.totalAllocatedElements(),
                                    kind));
            return dstOffset;
        }

        void* copyInputBuffers(const TensorDescriptor& descriptor,
                               void*                   dst,
                               void*                   src,
                               size_t                  totalElements,
                               hipMemcpyKind           kind)
        {
            HIP_CHECK_EXC(hipMemcpy(dst, src, descriptor.elementBytes() * totalElements, kind));
            return dst;
        }

        double DataInitialization::GetRepresentativeBetaValue(po::variables_map const& args)
        {
            auto argValue = args["init-beta"].as<int>();

            if(argValue == 0)
                return 0.0;

            if(argValue == 1)
                return 1.0;

            return 1.5;
        }

        DataInitialization::DataInitialization(po::variables_map const&    args,
                                               ClientProblemFactory const& problemFactory)
            : m_maxBatch(0)
            , m_stridedBatched(args["strided-batched"].as<bool>())
            , m_cEqualsD(args["c-equal-d"].as<bool>())
            , m_elementsToValidate(args["num-elements-to-validate"].as<int>())
            , m_keepPristineCopyOnGPU(args["pristine-on-gpu"].as<bool>())
            , m_workspaceSize(problemFactory.workspaceSize())
        {
            m_boundsCheck    = args["bounds-check"].as<BoundsCheckMode>();
            m_curBoundsCheck = m_boundsCheck;

            if(m_boundsCheck == BoundsCheckMode::GuardPageAll)
            {
                //GuardPageAll needs 2 runs per solution.
                //First run perform front side guard page checking.
                m_curBoundsCheck     = BoundsCheckMode::GuardPageFront;
                m_numRunsPerSolution = 2;
            }

            std::vector<std::vector<double>> activationAdditionalArgs;
            if(args.count("activation-additional-args"))
                activationAdditionalArgs
                    = args["activation-additional-args"].as<std::vector<std::vector<double>>>();

            if(problemFactory.problems().empty())
            {
                throw std::runtime_error("No problems in ProblemFactory.");
            }

            // Add switch cases here if needed. ex. GEMM, GEMM+GEMM

            // Get tensor info from problem factory.
            // TODO: Let ContractionProblemGroupedGemm use the same API as ContractionProblemGemm if possible.
            {
                auto const& p = problemFactory.problems()[0];
                if(auto ptr = dynamic_cast<ContractionProblemGroupedGemm const*>(p.get()))
                {
                    const ContractionProblemGroupedGemm& grouped = (*ptr);
                    if(m_problemDependentData)
                    {
                        throw std::runtime_error(
                            "Currently does not support dependent data with grouped gemm.");
                    }
                    if(problemFactory.problems().size() != 1)
                    {
                        throw std::runtime_error("Currently only supports one ContractionProblem "
                                                 "if grouped gemm is found in the ProblemFactory.");
                    }
                    m_vdata.resize(grouped.gemms[0].tensors().size());
                    m_cdata.resize(grouped.gemms[0].constants().size());
                }
                else
                {
                    m_vdata.resize(problemFactory.problems()[0]->tensors().size());
                    m_cdata.resize(problemFactory.problems()[0]->constants().size());
                }
            }

            for(auto const& p : problemFactory.problems())
            {
                if(auto ptr = dynamic_cast<ContractionProblemGemm const*>(p.get()))
                {
                    const ContractionProblemGemm& problem = (*ptr);
                    for(size_t i = 0; i < problem.tensors().size(); i++)
                    {
                        auto dataType = problem.tensors()[i].dataType();
                        if(m_vdata[i].pristine.find(dataType) == m_vdata[i].pristine.end())
                        {
                            m_vdata[i].pristine[dataType]             = PristineUnit();
                            m_vdata[i].pristine[dataType].maxElements = 0;
                        }
                        auto& pristine       = m_vdata[i].pristine[dataType];
                        pristine.maxElements = std::max(
                            pristine.maxElements, problem.tensors()[i].totalAllocatedElements());
                        if(m_vdata[i].name.empty())
                        {
                            m_vdata[i].name = problem.tensors()[i].getName();
                        }
                        else if(m_vdata[i].name != problem.tensors()[i].getName())
                        {
                            std::string s = "Input tensor name " + problem.tensors()[i].getName()
                                            + " not match the pristine name " + m_vdata[i].name
                                            + " at index " + std::to_string(i) + ".";
                            throw std::runtime_error(s.c_str());
                        }
                    }
                    auto constants = problem.constants();
                    for(size_t i = 0; i < constants.size(); i++)
                    {
                        if(m_cdata[i].name.empty())
                        {
                            m_cdata[i].name = constants[i].name;
                        }
                        else if(m_cdata[i].name != constants[i].name)
                        {
                            std::string s = "Input constant name " + constants[i].name
                                            + " not match the pristine name " + m_cdata[i].name
                                            + " at index " + std::to_string(i) + ".";
                            throw std::runtime_error(s.c_str());
                        }
                    }

                    size_t numOfBatch = 1;
                    for(size_t i = 0; i < problem.batchIndices().size(); i++)
                        numOfBatch *= problem.batchSize(i);
                    m_maxBatch = std::max(m_maxBatch, numOfBatch);
                }
                else if(auto ptr = dynamic_cast<ContractionProblemGroupedGemm const*>(p.get()))
                {
                    const ContractionProblemGroupedGemm& problems = (*ptr);

                    struct gElement
                    {
                        size_t              maxElements;
                        std::vector<size_t> offsets;
                    };

                    auto gElements = std::vector<std::map<DataType, gElement>>(m_vdata.size());
                    for(auto const& problem : problems.gemms)
                    {
                        for(size_t i = 0; i < problem.tensors().size(); i++)
                        {
                            auto dataType = problem.tensors()[i].dataType();
                            if(m_vdata[i].pristine.find(dataType) == m_vdata[i].pristine.end())
                            {
                                m_vdata[i].pristine[dataType]             = PristineUnit();
                                m_vdata[i].pristine[dataType].maxElements = 0;
                            }
                            if(gElements[i].find(dataType) == gElements[i].end())
                            {
                                gElements[i][dataType].maxElements = 0;
                            }
                            auto& pristine = m_vdata[i].pristine[dataType];
                            gElements[i][dataType].maxElements
                                += problem.tensors()[i].totalAllocatedElements();
                            gElements[i][dataType].offsets.push_back(
                                problem.tensors()[i].totalAllocatedElements());
                            if(m_vdata[i].name.empty())
                            {
                                m_vdata[i].name = problem.tensors()[i].getName();
                            }
                            else if(m_vdata[i].name != problem.tensors()[i].getName())
                            {
                                std::string s = "Input tensor name "
                                                + problem.tensors()[i].getName()
                                                + " not match the pristine name " + m_vdata[i].name
                                                + " at index " + std::to_string(i) + ".";
                                throw std::runtime_error(s.c_str());
                            }
                        }
                        auto constants = problem.constants();
                        for(size_t i = 0; i < constants.size(); i++)
                        {
                            if(m_cdata[i].name.empty())
                            {
                                m_cdata[i].name = constants[i].name;
                            }
                            else if(m_cdata[i].name != constants[i].name)
                            {
                                std::string s = "Input constant name " + constants[i].name
                                                + " not match the pristine name " + m_cdata[i].name
                                                + " at index " + std::to_string(i) + ".";
                                throw std::runtime_error(s.c_str());
                            }
                        }

                        size_t numOfBatch = 1;
                        for(size_t i = 0; i < problem.batchIndices().size(); i++)
                            numOfBatch *= problem.batchSize(i);
                        m_maxBatch = std::max(m_maxBatch, numOfBatch);
                    }

                    // Update maxElements
                    for(size_t i = 0; i < gElements.size(); i++)
                    {
                        for(auto it : gElements[i])
                        {
                            auto& pristine = m_vdata[i].pristine[it.first];
                            pristine.maxElements
                                = std::max(pristine.maxElements, it.second.maxElements);
                            if(pristine.groupedGemmOffsets.empty())
                            {
                                pristine.groupedGemmOffsets = it.second.offsets;
                            }
                            else
                            {
                                if(pristine.groupedGemmOffsets.size() != it.second.offsets.size())
                                {
                                    throw std::runtime_error(
                                        "Unable to update groupedGemmOffsets.");
                                }
                                for(size_t j = 0; j < it.second.offsets.size(); j++)
                                {
                                    pristine.groupedGemmOffsets[j] = std::max(
                                        pristine.groupedGemmOffsets[j], it.second.offsets[j]);
                                }
                            }
                        }
                    }
                }
            }

            // Init tensors
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                std::string initName = "init-" + m_vdata[i].name;
                std::string typeName = m_vdata[i].name + "-type";
                if(args.count(initName))
                {
                    m_vdata[i].init = args[initName].as<InitMode>();
                }
                else
                {
                    m_vdata[i].init = InitMode::Zero;
                }

                for(auto p = m_vdata[i].pristine.begin(); p != m_vdata[i].pristine.end();)
                {
                    // Remove pristine with maxElements = 0
                    if(p->second.maxElements == 0)
                    {
                        p = m_vdata[i].pristine.erase(p);
                        continue;
                    }

                    size_t dataTypeSize = DataTypeInfo::Get(p->first).elementSize;
                    if(m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        p->second.maxElements += 1024;
                    }
                    else if(m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                            || m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                    {
                        unsigned int roundUpSize = pageSize / dataTypeSize;
                        p->second.maxElements
                            = RoundUpToMultiple<unsigned int>(p->second.maxElements, roundUpSize);
                        // No bias page guard
                    }
                    ++p;
                }
                std::cout << "Tensor name " << m_vdata[i].name << " init mode "
                          << ToString(m_vdata[i].init) << std::endl;
            }

            // Init contants
            for(size_t i = 0; i < m_cdata.size(); i++)
            {
                std::string initName = "init-" + m_cdata[i].name;
                m_cdata[i].dataType  = DataType::None;
                // FIXME: Currently hardcoded
                if(m_cdata[i].name.find("activation") != std::string::npos)
                {
                    double value = 0.0;
                    if(activationAdditionalArgs.empty())
                    {
                        value = getValueWithUpperLowerBoundFP<double>(2.0, -2.0);
                    }
                    else
                    {
                        std::string            name   = m_cdata[i].name;
                        std::string            prefix = "activation-";
                        std::string::size_type pos    = name.find(prefix);

                        size_t index = -1;
                        if(pos != std::string::npos)
                        {
                            name.erase(pos, prefix.length());
                            index = greekToIndex(name);
                        }
                        // FIXME: Valgrind error: Invalid read of size 8
                        const auto& actArgs = activationAdditionalArgs[0];
                        value = (index >= actArgs.size()) ? actArgs[actArgs.size() - 1]
                                                          : actArgs[index];
                    }
                    m_cdata[i].freeValue = value;
                    m_cdata[i].init      = InitMode::Free;
                }
                else if(args.count(initName))
                {
                    m_cdata[i].init = args[initName].as<InitMode>();
                }
                else
                {
                    m_cdata[i].init = InitMode::Zero;
                }
                std::cout << "constant name " << m_cdata[i].name << " init mode "
                          << ToString(m_cdata[i].init) << std::endl;
            }

            // Need refactor, gemm a, b, c, d only
            m_problemDependentData = 0;
            for(size_t i = 0; i < 4; i++)
            {
                m_problemDependentData
                    = m_problemDependentData || IsProblemDependent(m_vdata[i].init);
            }

            allocNewCPUInputs();
            allocNewGPUInputs();

            for(auto& it : m_vdata)
            {
                for(auto& p : it.pristine)
                {
                    auto  dataTypeSize = DataTypeInfo::Get(p.first).elementSize;
                    auto& pUnit        = p.second;
                    // Init and copy valid from cpu to gpu, only copies when != dependent data
                    if(!m_problemDependentData)
                    {

                        initArray(p.first, it.init, pUnit.cpuInput.valid.get(), pUnit.maxElements);
                        HIP_CHECK_EXC(hipMemcpy(pUnit.gpuInput.valid.get(),
                                                pUnit.cpuInput.valid.get(),
                                                dataTypeSize * pUnit.maxElements,
                                                hipMemcpyHostToDevice));
                    }
                    // Init and copy bad from cpu to gpu
                    if(pUnit.gpuInput.bad && pUnit.cpuInput.bad)
                    {
                        initArray(p.first,
                                  InitMode::BadOutput,
                                  pUnit.cpuInput.bad.get(),
                                  pUnit.maxElements);
                        HIP_CHECK_EXC(hipMemcpy(pUnit.gpuInput.bad.get(),
                                                pUnit.cpuInput.bad.get(),
                                                dataTypeSize * pUnit.maxElements,
                                                hipMemcpyHostToDevice));
                    }
                }
            }
        }

        void DataInitialization::allocNewCPUInputs()
        {
            for(auto& it : m_vdata)
            {
                for(auto& p : it.pristine)
                {
                    auto&  pUnit = p.second;
                    size_t size  = DataTypeInfo::Get(p.first).elementSize * pUnit.maxElements;
                    if(size <= 0)
                    {
                        throw std::runtime_error("Size not exists.");
                    }

                    std::stringstream ss;
                    ss << "Failed to allocate cpu input " << it.name << " type("
                       << DataTypeInfo::Get(p.first).abbrev
                       << "), element size: " << DataTypeInfo::Get(p.first).elementSize
                       << ", element length: " << pUnit.maxElements;

                    if(!pUnit.cpuInput.current)
                    {
                        auto ptr = std::shared_ptr<void>(std::malloc(size), std::free);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[input]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.cpuInput.current = ptr;
                    }
                    if(!pUnit.cpuInput.valid)
                    {
                        auto ptr = std::shared_ptr<void>(std::malloc(size), std::free);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[valid]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.cpuInput.valid = ptr;
                    }
                    if(!pUnit.cpuInput.bad && m_curBoundsCheck == BoundsCheckMode::NaN)
                    {
                        auto ptr = std::shared_ptr<void>(std::malloc(size), std::free);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[bad]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.cpuInput.bad = ptr;
                    }
                }
            }
            return;
        }

        void DataInitialization::allocNewGPUInputs()
        {
            std::vector<std::shared_ptr<void>> guardPage;
            void*                              guardPagePtr;
            bool enableGuardPage = (m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                                    || m_curBoundsCheck == BoundsCheckMode::GuardPageBack);

            for(auto& it : m_vdata)
            {
                for(auto& p : it.pristine)
                {
                    auto&  pUnit = p.second;
                    size_t size  = DataTypeInfo::Get(p.first).elementSize * pUnit.maxElements;

                    std::stringstream ss;
                    ss << "Failed to allocate cpu input " << it.name << " type("
                       << DataTypeInfo::Get(p.first).abbrev
                       << "), element size: " << DataTypeInfo::Get(p.first).elementSize
                       << ", element length: " << pUnit.maxElements;

                    if(!pUnit.gpuInput.current)
                    {
                        if(enableGuardPage)
                        {
                            HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                            guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                        }
                        auto ptr = allocNewGPUBuffer<void>(it.name.c_str(), size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[input]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.gpuInput.current = ptr;
                        std::string n          = "batch" + it.name;
                        auto        batch_ptr
                            = allocNewGPUBuffer<void*>(n.c_str(), sizeof(uint8_t*) * m_maxBatch);
                        if(batch_ptr == nullptr)
                            throw std::runtime_error("out of batch gpu memory");
                        pUnit.gpuInput.batch = batch_ptr;
                    }
                    if(!pUnit.gpuInput.valid)
                    {
                        if(enableGuardPage)
                        {
                            HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                            guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                        }
                        auto ptr = allocNewGPUBuffer<void>(it.name.c_str(), size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[valid]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.gpuInput.valid = ptr;
                    }
                    if(!pUnit.gpuInput.bad)
                    {
                        if(enableGuardPage)
                        {
                            HIP_CHECK_EXC(hipMalloc(&guardPagePtr, pageSize));
                            guardPage.push_back(std::shared_ptr<void>(guardPagePtr, hipFree));
                        }
                        auto ptr = allocNewGPUBuffer<void>(it.name.c_str(), size);
                        if(ptr == nullptr)
                        {
                            std::stringstream s;
                            s << "[bad]" << ss.str();
                            throw std::runtime_error(s.str().c_str());
                        }
                        pUnit.gpuInput.bad = ptr;
                    }
                }
            }

            if(!m_workspacePristine)
            {
                std::shared_ptr<void> ptr = nullptr;
                if(m_workspaceSize > 0)
                {
                    ptr = allocNewGPUBuffer<void>("ws", m_workspaceSize);
                    if(ptr == nullptr)
                        throw std::runtime_error(
                            "out of gpu memory while allocating workspace size");
                }
                m_workspacePristine = ptr;
            }

            // allocate remaining memory to prevend other user use GPU when benchmarking
            if(Debug::Instance().getBenchmark())
            {
                void*           extra = nullptr;
                size_t          remainingSize;
                hipDeviceProp_t hipProps;
                HIP_CHECK_EXC(hipGetDeviceProperties(&hipProps, 0));
                remainingSize = size_t(hipProps.totalGlobalMem);
                printf("Trying to allocate all GPU memory to prevend other user use GPU when "
                       "benchmarking \n");
                while(1)
                {
                    if(hipSuccess == hipMalloc(&extra, remainingSize))
                    {
                        printf("LOCAL: GPU benchmark protect, allocate %zu MB Success \n",
                               remainingSize / (1024 * 1024));
                    }
                    else
                    {
                        printf("LOCAL: GPU benchmark protect, allocate %zu MB Fail \n",
                               remainingSize / (1024 * 1024));
                    }
                    remainingSize = remainingSize / 2;
                    if(remainingSize <= 0)
                    {
                        break;
                    }
                };
            }
            return;
        }

        void DataInitialization::initializeGPUBatchedInputs(ContractionProblemGemm const& problem)
        {
            auto batchIdxs = problem.batchIndices();
            // FIXME: batch not supported for bias and scaleD
            for(size_t i = 0; i < 4 /*m_vdata.size()*/; i++)
            {
                auto&               pUnit = m_vdata[i].pristine[problem.tensors()[i].dataType()];
                std::vector<size_t> batchIdx(batchIdxs.size(), 0);
                ptrdiff_t           padding = 0;
                for(size_t j = 0; j < batchIdxs.size(); j++)
                {
                    switch(i)
                    {
                    case 0:
                        batchIdx[j] = batchIdxs[j].a;
                        break;
                    case 1:
                        batchIdx[j] = batchIdxs[j].b;
                        break;
                    case 2:
                        batchIdx[j] = batchIdxs[j].c;
                        break;
                    case 3:
                        batchIdx[j] = batchIdxs[j].d;
                        break;
                    }
                }
                if(m_curBoundsCheck == BoundsCheckMode::NaN)
                {
                    padding
                        = (pUnit.maxElements - problem.tensors()[i].totalAllocatedElements()) / 2;
                }
                else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
                {
                    padding = pUnit.maxElements - problem.tensors()[i].totalAllocatedElements();
                }
                padding *= DataTypeInfo::Get(problem.tensors()[i].dataType()).elementSize;
                uint8_t* offset = (uint8_t*)pUnit.gpuInput.current.get();
                initGPUBatchedInput((void*)(offset + padding),
                                    pUnit.gpuInput.batch.get(),
                                    problem.tensors()[i],
                                    batchIdx);
            }
        }

        void DataInitialization::initializeCPUInputs(ContractionProblemGemm const& problem)
        {
            auto& tensors = problem.tensors();
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                if(m_problemDependentData)
                {
                    // Should this m_cEqualsD set in ContractionProblem or boost args?
                    for(auto& p : m_vdata[i].pristine)
                    {
                        // Only update when the descriptor changed
                        if(p.second.initDescriptor != tensors[i])
                        {
                            p.second.initDescriptor = tensors[i];
                            initArray(p.first,
                                      m_vdata[i].init,
                                      p.second.cpuInput.valid.get(),
                                      tensors[i]);
                        }
                    }
                }
            }
        }

        void DataInitialization::initializeConstantInputs(ContractionProblemGemm const& problem)
        {
            // Update constants if needed
            for(size_t i = 0; i < problem.constants().size(); i++)
            {
                auto& prop = m_cdata[i];
                if(prop.dataType != problem.constants()[i].dataType)
                {
                    prop.dataType = problem.constants()[i].dataType;
                    switch(prop.dataType)
                    {
                    case DataType::Float:
                        prop.value = getValue<float>(prop.init, prop.freeValue);
                        break;
                    case DataType::Double:
                        prop.value = getValue<double>(prop.init, prop.freeValue);
                        break;
                    case DataType::Half:
                        prop.value = getValue<Half>(prop.init, prop.freeValue);
                        break;
                    case DataType::Int32:
                        prop.value = getValue<int32_t>(prop.init, prop.freeValue);
                        break;
                    case DataType::BFloat16:
                        prop.value = getValue<BFloat16>(prop.init, prop.freeValue);
                        break;
                    case DataType::Int8:
                        prop.value = getValue<int8_t>(prop.init, prop.freeValue);
                        break;
                    case DataType::ComplexFloat:
                        prop.value = getValue<std::complex<float>>(prop.init, prop.freeValue);
                        break;
                    case DataType::ComplexDouble:
                        prop.value = getValue<std::complex<double>>(prop.init, prop.freeValue);
                        break;
                    case DataType::Int8x4:
                        prop.value = getValue<Int8x4>(prop.init, prop.freeValue);
                        break;
                    case DataType::Count:;
                    }
                }
                if(Debug::Instance().printTensorInfo() && prop.dataType != DataType::None)
                    std::cout << "Constant " << m_cdata[i].name << ". Type "
                              << DataTypeInfo::Get(prop.dataType).abbrev << std::endl;
            }
            return;
        }

        void DataInitialization::copyInputs(std::vector<void*>&               ptrs,
                                            std::vector<void**>&              batchPtrs,
                                            std::vector<size_t>&              maxElements,
                                            std::vector<std::vector<size_t>>& offsets,
                                            ContractionProblemGemm const&     problem,
                                            hipMemcpyKind                     kind)
        {
            ptrs.clear();
            batchPtrs.clear();
            maxElements.clear();
            if(m_curBoundsCheck == BoundsCheckMode::NaN)
            {
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    void* ptr  = nullptr;
                    auto& desc = problem.tensors()[i];
                    auto  it   = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto& p = it->second;
                        if(kind == hipMemcpyHostToHost)
                            ptr = copyBadInputBuffers(desc,
                                                      p.cpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.cpuInput.bad.get(),
                                                      p.maxElements,
                                                      kind);
                        else if(kind == hipMemcpyHostToDevice)
                            ptr = copyBadInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.cpuInput.bad.get(),
                                                      p.maxElements,
                                                      kind);
                        else if(kind == hipMemcpyDeviceToDevice)
                            ptr = copyBadInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.gpuInput.valid.get(),
                                                      p.gpuInput.bad.get(),
                                                      p.maxElements,
                                                      kind);
                        ptrs.push_back(ptr);
                        batchPtrs.push_back(p.getInputByKind(kind).batch.get());
                        maxElements.push_back(p.maxElements);
                        offsets.push_back(p.groupedGemmOffsets);
                    }
                    else
                    {
                        ptrs.push_back(nullptr);
                        batchPtrs.push_back(nullptr);
                        maxElements.push_back(0);
                        offsets.push_back(std::vector<size_t>());
                    }
                }
            }
            else if(m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
            {
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    void* ptr  = nullptr;
                    auto& desc = problem.tensors()[i];
                    auto  it   = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto& p = it->second;
                        if(kind == hipMemcpyHostToHost)
                            ptr = copyNaNInputBuffers(desc,
                                                      p.cpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.maxElements,
                                                      kind);
                        else if(kind == hipMemcpyHostToDevice)
                            ptr = copyNaNInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.cpuInput.valid.get(),
                                                      p.maxElements,
                                                      kind);
                        else if(kind == hipMemcpyDeviceToDevice)
                            ptr = copyNaNInputBuffers(desc,
                                                      p.gpuInput.current.get(),
                                                      p.gpuInput.valid.get(),
                                                      p.maxElements,
                                                      kind);
                        ptrs.push_back(ptr);
                        batchPtrs.push_back(p.getInputByKind(kind).batch.get());
                        maxElements.push_back(p.maxElements);
                        offsets.push_back(p.groupedGemmOffsets);
                    }
                    else
                    {
                        ptrs.push_back(nullptr);
                        batchPtrs.push_back(nullptr);
                        maxElements.push_back(0);
                        offsets.push_back(std::vector<size_t>());
                    }
                }
            }
            else
            {
                for(size_t i = 0; i < m_vdata.size(); i++)
                {
                    void* ptr  = nullptr;
                    auto& desc = problem.tensors()[i];
                    auto  it   = m_vdata[i].pristine.find(desc.dataType());
                    if(it != m_vdata[i].pristine.end())
                    {
                        auto& p = it->second;
                        if(kind == hipMemcpyHostToHost)
                            ptr = copyInputBuffers(desc,
                                                   p.cpuInput.current.get(),
                                                   p.cpuInput.valid.get(),
                                                   p.maxElements,
                                                   kind);
                        else if(kind == hipMemcpyHostToDevice)
                            ptr = copyInputBuffers(desc,
                                                   p.gpuInput.current.get(),
                                                   p.cpuInput.valid.get(),
                                                   p.maxElements,
                                                   kind);
                        else if(kind == hipMemcpyDeviceToDevice)
                            ptr = copyInputBuffers(desc,
                                                   p.gpuInput.current.get(),
                                                   p.gpuInput.valid.get(),
                                                   p.maxElements,
                                                   kind);
                        if(ptr == nullptr)
                        {
                            std::runtime_error("output ptr is null when copy input");
                        }
                        ptrs.push_back(ptr);
                        batchPtrs.push_back(p.getInputByKind(kind).batch.get());
                        maxElements.push_back(p.maxElements);
                        offsets.push_back(p.groupedGemmOffsets);
                    }
                    else
                    {
                        ptrs.push_back(nullptr);
                        batchPtrs.push_back(nullptr);
                        maxElements.push_back(0);
                        offsets.push_back(std::vector<size_t>());
                    }
                }
            }
        }

        void DataInitialization::resetOutput(std::vector<void*>&               ptrs,
                                             std::vector<void**>&              batchPtrs,
                                             std::vector<size_t>&              maxElements,
                                             std::vector<std::vector<size_t>>& offsets,
                                             ContractionProblemGemm const&     problem,
                                             hipMemcpyKind                     kind)
        {
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                void* ptr  = nullptr;
                auto& desc = problem.tensors()[i];
                if(!desc.isOutput()) // Need init first
                    continue;
                auto it = m_vdata[i].pristine.find(desc.dataType());
                if(it != m_vdata[i].pristine.end())
                {
                    auto& p = it->second;
                    if(kind == hipMemcpyHostToHost)
                        ptr = copyInputBuffers(desc,
                                               p.cpuInput.current.get(),
                                               p.cpuInput.valid.get(),
                                               p.maxElements,
                                               kind);
                    else if(kind == hipMemcpyHostToDevice)
                        ptr = copyInputBuffers(desc,
                                               p.gpuInput.current.get(),
                                               p.cpuInput.valid.get(),
                                               p.maxElements,
                                               kind);
                    else if(kind == hipMemcpyDeviceToDevice)
                        ptr = copyInputBuffers(desc,
                                               p.gpuInput.current.get(),
                                               p.gpuInput.valid.get(),
                                               p.maxElements,
                                               kind);
                    if(ptr == nullptr)
                    {
                        std::runtime_error("output ptr is null when copy input");
                    }
                    ptrs[i]        = ptr;
                    batchPtrs[i]   = p.getInputByKind(kind).batch.get();
                    maxElements[i] = p.maxElements;
                    offsets[i]     = p.groupedGemmOffsets;
                }
                else
                {
                    ptrs[i]        = nullptr;
                    batchPtrs[i]   = nullptr;
                    maxElements[i] = 0;
                    offsets[i].clear();
                }
            }
        }

        void DataInitialization::copyValidToGPUBuffer(ContractionProblemGemm const& problem)
        {
            for(size_t i = 0; i < m_vdata.size(); i++)
            {
                void* ptr  = nullptr;
                auto& desc = problem.tensors()[i];
                auto& p    = m_vdata[i].pristine[desc.dataType()];
                ptr        = copyInputBuffers(desc,
                                       p.gpuInput.valid.get(),
                                       p.cpuInput.valid.get(),
                                       p.maxElements,
                                       hipMemcpyHostToDevice);
                if(ptr == nullptr)
                    std::__throw_runtime_error("error");
            }
        }

        template <typename T>
        void DataInitialization::setContractionInputs(std::vector<T*>&     ptrs,
                                                      std::vector<void**>& batchPtrs,
                                                      void*                ws,
                                                      std::vector<ConstDataInitProperties>& cdata,
                                                      std::vector<size_t> maxElements,
                                                      bool                isGPU,
                                                      ContractionInputs*  inputs)
        {
            inputs->a      = (void*)ptrs[ContractionProblemGemm::TENSOR::A];
            inputs->b      = (void*)ptrs[ContractionProblemGemm::TENSOR::B];
            inputs->c      = (void*)ptrs[ContractionProblemGemm::TENSOR::C];
            inputs->d      = (void*)ptrs[ContractionProblemGemm::TENSOR::D];
            inputs->e      = (void*)ptrs[ContractionProblemGemm::TENSOR::E];
            inputs->bias   = (void*)ptrs[ContractionProblemGemm::TENSOR::BIAS];
            inputs->scaleD = (void*)ptrs[ContractionProblemGemm::TENSOR::SCALED];

            inputs->batchA = (void**)batchPtrs[ContractionProblemGemm::TENSOR::A];
            inputs->batchB = (void**)batchPtrs[ContractionProblemGemm::TENSOR::B];
            inputs->batchC = (void**)batchPtrs[ContractionProblemGemm::TENSOR::C];
            inputs->batchD = (void**)batchPtrs[ContractionProblemGemm::TENSOR::D];

            inputs->gpu = isGPU;

            inputs->ws             = (void*)ws;
            inputs->alpha          = cdata[ContractionProblemGemm::CONST::ALPHA].value;
            inputs->beta           = cdata[ContractionProblemGemm::CONST::BETA].value;
            inputs->activationArgs = {cdata[ContractionProblemGemm::CONST::ACTALPHA].value,
                                      cdata[ContractionProblemGemm::CONST::ACTBETA].value};

            inputs->maxElements = maxElements;
        }

        void DataInitialization::setContractionGroupedInputs(
            std::vector<void*>&                     ptrs,
            std::vector<void**>&                    batchPtrs,
            void*                                   ws,
            std::vector<ConstDataInitProperties>&   cdata,
            bool                                    isGPU,
            ContractionProblemGemm const&           problem,
            std::vector<std::vector<size_t>> const& offsets,
            ContractionGroupedInputs*               inputs)
        {
            auto aBuffer      = (uint8_t*)ptrs[ContractionProblemGemm::TENSOR::A];
            auto bBuffer      = (uint8_t*)ptrs[ContractionProblemGemm::TENSOR::B];
            auto cBuffer      = (uint8_t*)ptrs[ContractionProblemGemm::TENSOR::C];
            auto dBuffer      = (uint8_t*)ptrs[ContractionProblemGemm::TENSOR::D];
            auto biasBuffer   = (uint8_t*)ptrs[ContractionProblemGemm::TENSOR::BIAS];
            auto scaleDBuffer = (uint8_t*)ptrs[ContractionProblemGemm::TENSOR::SCALED];

            std::vector<uint8_t*> u8Ptr;
            for(auto p : ptrs)
            {
                u8Ptr.push_back((uint8_t*)p);
            }

            for(int idx = 0; idx < offsets[0].size(); idx++)
            {
                ContractionInputs   unit;
                std::vector<size_t> maxElements;
                for(size_t j = 0; j < offsets.size(); j++)
                {

                    if(offsets[j].size() != 0)
                    {
                        maxElements.push_back(offsets[j][idx]);
                    }
                    else
                    {
                        maxElements.push_back(0);
                    }
                }
                setContractionInputs(u8Ptr, batchPtrs, ws, cdata, maxElements, isGPU, &unit);
                inputs->grouped.push_back(unit);

                u8Ptr[ContractionProblemGemm::TENSOR::A]
                    += offsets[ContractionProblemGemm::TENSOR::A][idx] * problem.a().elementBytes();
                u8Ptr[ContractionProblemGemm::TENSOR::B]
                    += offsets[ContractionProblemGemm::TENSOR::B][idx] * problem.b().elementBytes();
                u8Ptr[ContractionProblemGemm::TENSOR::C]
                    += offsets[ContractionProblemGemm::TENSOR::C][idx] * problem.c().elementBytes();
                u8Ptr[ContractionProblemGemm::TENSOR::D]
                    += offsets[ContractionProblemGemm::TENSOR::D][idx] * problem.d().elementBytes();
                if(u8Ptr[ContractionProblemGemm::TENSOR::BIAS] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::BIAS]
                        += offsets[ContractionProblemGemm::TENSOR::BIAS][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::BIAS].elementBytes();
                }
                if(u8Ptr[ContractionProblemGemm::TENSOR::SCALED] != nullptr)
                {
                    u8Ptr[ContractionProblemGemm::TENSOR::SCALED]
                        += offsets[ContractionProblemGemm::TENSOR::SCALED][idx]
                           * problem.tensors()[ContractionProblemGemm::TENSOR::SCALED]
                                 .elementBytes();
                }
            }
        }

        // For GEMM only
        std::shared_ptr<ProblemInputs>
            DataInitialization::ConvertToProblemInputs(ContractionProblemGemm const& problem,
                                                       bool                          isGPU)
        {
            std::shared_ptr<ProblemInputs> result;
            if(m_groupedOffsets[0].empty())
            {
                auto inputs = new ContractionInputs();
                if(isGPU)
                    setContractionInputs(m_gpuPtrs,
                                         m_gpuBatchPtrs,
                                         m_workspacePristine.get(),
                                         m_cdata,
                                         m_maxElements,
                                         isGPU,
                                         inputs);
                else
                {
                    auto dummyBatchPtrs = std::vector<void**>(
                        ContractionProblemGemm::TENSOR::TENSOR_COUNT, nullptr);
                    setContractionInputs(m_cpuPtrs,
                                         dummyBatchPtrs,
                                         m_workspacePristine.get(),
                                         m_cdata,
                                         m_maxElements,
                                         isGPU,
                                         inputs);
                }
                result = static_pointer_cast<ProblemInputs>(
                    std::shared_ptr<ContractionInputs>(inputs));
            }
            else
            {
                auto inputs = new ContractionGroupedInputs();
                // Currently grouped gemm does not support batch, so we use a dummy batch vector here.
                auto dummyBatchPtrs
                    = std::vector<void**>(ContractionProblemGemm::TENSOR::TENSOR_COUNT, nullptr);
                if(isGPU)
                    setContractionGroupedInputs(m_gpuPtrs,
                                                dummyBatchPtrs,
                                                m_workspacePristine.get(),
                                                m_cdata,
                                                isGPU,
                                                problem,
                                                m_groupedOffsets,
                                                inputs);
                else
                    setContractionGroupedInputs(m_cpuPtrs,
                                                dummyBatchPtrs,
                                                m_workspacePristine.get(),
                                                m_cdata,
                                                isGPU,
                                                problem,
                                                m_groupedOffsets,
                                                inputs);
                result = static_pointer_cast<ProblemInputs>(
                    std::shared_ptr<ContractionGroupedInputs>(inputs));
            }
            return result;
        }

        DataInitialization::~DataInitialization() {}
    } // namespace Client
} // namespace Tensile
