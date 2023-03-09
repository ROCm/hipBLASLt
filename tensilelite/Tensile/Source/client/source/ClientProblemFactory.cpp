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

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        ClientProblemFactory::ClientProblemFactory(po::variables_map const& args)
            : m_problemSizes(args["problem-size"].as<std::vector<std::vector<size_t>>>())
            , m_stridedBatched(args["strided-batched"].as<bool>())
            , m_groupedGemm(args["grouped-gemm"].as<bool>())
            , m_highPrecisionAccumulate(args["high-precision-accumulate"].as<bool>())
            , m_kernelLanguage(args["kernel-language"].as<KernelLanguage>())
            , m_performanceMetric(args["performance-metric"].as<PerformanceMetric>())
            , m_deterministicMode(args["deterministic-mode"].as<bool>())
            , m_cEqualsD(args["c-equal-d"].as<bool>())
            , m_biasTypeArgs(std::vector<DataType>(1, DataType::Float))
            , m_activationType(ActivationType::None)
            , m_activationHPA(false)
            , m_activationGuard(false)
            , m_activationEnumArg(std::vector<ActivationType>(1, ActivationType::None))
        {
            std::vector<bool> isComplex;
            if(args.count("problem-identifier"))
            {
                ContractionProblemGemm::IdentifierToIndices(
                    args["problem-identifier"].as<std::string>(),
                    m_freeIndices,
                    m_batchIndices,
                    m_boundIndices,
                    isComplex);

                for(size_t i = 0; i < isComplex.size(); i++)
                {
                    if(isComplex[i])
                    {
                        std::runtime_error("Complex is not supported.");
                    }
                }
            }
            else
            {
                std::runtime_error("Currently only accepts identifier as input.");
            }

            // Default datatype
            DataType type = DataType::None;
            if(args.count("type"))
            {
                type = args["type"].as<DataType>();
            }

            // Should add problem type in ClientParamters.ini
            auto dummy     = ContractionProblemGemm::GetDummy();
            auto tensors   = dummy.tensors();
            auto constants = dummy.constants();
            m_tensorTypes.resize(tensors.size());
            m_tensorStrides.resize(tensors.size());
            m_constantTypes.resize(constants.size());
            m_constantValues.resize(constants.size());
            // Get types and values from the information from ContractionProblem
            // May contain useless information for ClientProblemFactory
            // Get tensor types
            for(size_t i = 0; i < tensors.size(); i++)
            {
                std::string typeName = tensors[i].getName() + "-type";
                if(args.count(typeName))
                {
                    m_tensorTypes[i] = args[typeName].as<DataType>();
                }
                else
                {
                    m_tensorTypes[i] = type;
                }
                std::string strideName = tensors[i].getName() + "-strides";
                if(args.count(strideName))
                {
                    m_tensorStrides[i] = args[strideName].as<std::vector<std::vector<size_t>>>();
                }
                else
                {
                    m_tensorStrides[i] = std::vector<std::vector<size_t>>();
                }
            }
            // Get constant types
            for(size_t i = 0; i < constants.size(); i++)
            {
                std::string typeName = constants[i].name + "-type";
                if(args.count(typeName))
                {
                    m_constantTypes[i] = args[typeName].as<DataType>();
                }
                else
                {
                    m_constantTypes[i] = type;
                }
                std::string valueName = "init-" + constants[i].name;
                if(args.count(valueName))
                {
                    m_constantValues[i]
                        = DataInitialization::getValue<double>(args[valueName].as<InitMode>());
                }
                else
                {
                    m_constantValues[i] = 0;
                }
            }

            if(args.count("use-e"))
                m_useE = args["use-e"].as<bool>();

            if(args.count("use-gradient"))
                m_useGradient = args["use-gradient"].as<bool>();

            if(args.count("bias-type-args"))
                m_biasTypeArgs = args["bias-type-args"].as<std::vector<DataType>>();

            if(args.count("activation-type"))
                m_activationType = args["activation-type"].as<ActivationType>();
            if(args.count("activation-hpa"))
                m_activationHPA = args["activation-hpa"].as<bool>();
            if(args.count("activation-guard"))
                m_activationGuard = args["activation-guard"].as<bool>();
            if(args.count("activation-enum-args"))
                m_activationEnumArg
                    = args["activation-enum-args"].as<std::vector<ActivationType>>();
            if(args.count("use-bias"))
                m_useBias = args["use-bias"].as<bool>();
            if(args.count("use-scaleD"))
                m_useScaleD = args["use-scaleD"].as<bool>();
            if(args.count("max-workspace-size"))
                m_maxWorkspaceSize = args["max-workspace-size"].as<size_t>();

            if(m_groupedGemm)
            {
                auto problems = std::make_shared<ContractionProblemGroupedGemm>();
                createProblems(problems->gemms);
                m_problems.push_back(static_pointer_cast<ContractionProblem>(problems));
            }
            else
            {
                std::vector<ContractionProblemGemm> v;
                createProblems(v);
                for(auto& it : v)
                {
                    auto problem     = std::make_shared<ContractionProblemGemm>();
                    (*problem.get()) = it;
                    m_problems.push_back(static_pointer_cast<ContractionProblem>(problem));
                }
            }
        }

        ClientProblemFactory::~ClientProblemFactory() = default;

        std::vector<std::shared_ptr<ContractionProblem>> const&
            ClientProblemFactory::problems() const
        {
            return m_problems;
        }

        void ClientProblemFactory::createProblems(std::vector<ContractionProblemGemm>& rv)
        {
            rv.clear();
            int biasSize       = std::max(1, (int)m_biasTypeArgs.size());
            int activationSize = std::max(1, (int)m_activationEnumArg.size());
            rv.reserve(m_problemSizes.size() * activationSize);

            std::vector<size_t> aStrides, bStrides, cStrides, dStrides, eStrides;

            if(m_tensorStrides[ContractionProblemGemm::TENSOR::A].size() == 1)
                aStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::A][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::B].size() == 1)
                bStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::B][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::C].size() == 1)
                cStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::C][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::D].size() == 1)
                dStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::D][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::E].size() == 1)
                eStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::E][0];

            for(int k = 0; k < biasSize; k++)
            {
                for(int j = 0; j < activationSize; j++)
                {
                    for(int i = 0; i < m_problemSizes.size(); i++)
                    {
                        if(m_tensorStrides[ContractionProblemGemm::TENSOR::A].size()
                           == m_problemSizes.size())
                            aStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::A][i];
                        if(m_tensorStrides[ContractionProblemGemm::TENSOR::B].size()
                           == m_problemSizes.size())
                            bStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::B][i];
                        if(m_tensorStrides[ContractionProblemGemm::TENSOR::C].size()
                           == m_problemSizes.size())
                            cStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::C][i];
                        if(m_tensorStrides[ContractionProblemGemm::TENSOR::D].size()
                           == m_problemSizes.size())
                            dStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::D][i];
                        if(m_tensorStrides[ContractionProblemGemm::TENSOR::E].size()
                           == m_problemSizes.size())
                            eStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::E][i];

                        rv.push_back(ContractionProblemGemm::FromIndexSizes(
                            m_freeIndices,
                            m_batchIndices,
                            m_boundIndices,
                            m_problemSizes[i],
                            m_tensorTypes[ContractionProblemGemm::TENSOR::A],
                            aStrides,
                            m_tensorTypes[ContractionProblemGemm::TENSOR::B],
                            bStrides,
                            m_tensorTypes[ContractionProblemGemm::TENSOR::C],
                            cStrides,
                            m_tensorTypes[ContractionProblemGemm::TENSOR::D],
                            dStrides,
                            m_constantValues[ContractionProblemGemm::CONST::BETA]));

                        rv.back().setAlphaRestriction(toScalarValueEnum(
                            m_constantValues[ContractionProblemGemm::CONST::ALPHA]));
                        rv.back().setCEqualsD(m_cEqualsD);
                        rv.back().setAlphaType(
                            m_constantTypes[ContractionProblemGemm::CONST::ALPHA]);
                        rv.back().setBetaType(m_constantTypes[ContractionProblemGemm::CONST::BETA]);
                        rv.back().setStridedBatched(m_stridedBatched);
                        rv.back().setHighPrecisionAccumulate(m_highPrecisionAccumulate);
                        rv.back().setUseGradient(m_useGradient);
                        rv.back().setUseBias(m_useBias);
                        rv.back().setUseE(m_useE);
                        rv.back().setKernelLanguage(m_kernelLanguage);
                        rv.back().setPerformanceMetric(m_performanceMetric);
                        rv.back().setDeterministicMode(m_deterministicMode);
                        rv.back().setFp16AltImpl(m_fp16AltImpl);
                        rv.back().setActivationType(m_activationType);
                        rv.back().setWorkspaceSize(m_maxWorkspaceSize);
                        if(k < m_biasTypeArgs.size())
                        {
                            rv.back().setBias(m_biasTypeArgs[k], rv.back().d().sizes()[0]);
                        }
                        else
                        {
                            rv.back().setBias(DataType::None, 0);
                        }
                        if(m_useE)
                        {
                            bool isEOutput = true;
                            if(m_useGradient)
                                isEOutput = false;
                            rv.back().setE(m_tensorTypes[ContractionProblemGemm::TENSOR::E],
                                           rv.back().d().sizes(),
                                           eStrides,
                                           isEOutput);
                        }
                        if(j < m_activationEnumArg.size())
                        {
                            rv.back().setActivationEnumArg(m_activationEnumArg[j]);
                        }
                        else
                        {
                            rv.back().setActivationType(m_activationType);
                        }
                        rv.back().setActivationHPA(m_activationHPA);
                        rv.back().setActivationGuard(m_activationGuard);
                        rv.back().setUseScaleD(m_useScaleD);
                        rv.back().setScaleD(m_constantTypes[ContractionProblemGemm::CONST::ALPHA],
                                            rv.back().d().sizes()[0]);

                        rv.back().setGroupedGemm(m_groupedGemm);
                    }
                }
            }
        }
    } // namespace Client
} // namespace Tensile
