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
            : m_freeIndices(args["free"].as<ContractionProblem::FreeIndices>())
            , m_batchIndices(args["batch"].as<ContractionProblem::BatchIndices>())
            , m_boundIndices(args["bound"].as<ContractionProblem::BoundIndices>())
            , m_problemSizes(args["problem-size"].as<std::vector<std::vector<size_t>>>())
            , m_aType(DataType::Float)
            , m_bType(DataType::Float)
            , m_cType(DataType::Float)
            , m_dType(DataType::Float)
            , m_alphaType(DataType::Float)
            , m_betaType(DataType::Float)
            , m_stridedBatched(args["strided-batched"].as<bool>())
            , m_groupedGemm(args["grouped-gemm"].as<bool>())
            , m_highPrecisionAccumulate(args["high-precision-accumulate"].as<bool>())
            , m_kernelLanguage(args["kernel-language"].as<KernelLanguage>())
            , m_performanceMetric(args["performance-metric"].as<PerformanceMetric>())
            , m_deterministicMode(args["deterministic-mode"].as<bool>())
            , m_cEqualsD(args["c-equal-d"].as<bool>())
            , m_arithmeticUnit(args["arithmetic-unit"].as<ArithmeticUnit>())
            , m_aStrides(args["a-strides"].as<std::vector<std::vector<size_t>>>())
            , m_bStrides(args["b-strides"].as<std::vector<std::vector<size_t>>>())
            , m_cStrides(args["c-strides"].as<std::vector<std::vector<size_t>>>())
            , m_dStrides(args["d-strides"].as<std::vector<std::vector<size_t>>>())
            , m_aOps(args["a-ops"].as<TensorOps>())
            , m_bOps(args["b-ops"].as<TensorOps>())
            , m_cOps(args["c-ops"].as<TensorOps>())
            , m_dOps(args["d-ops"].as<TensorOps>())
            , m_aOffset(args["offset-a"].as<size_t>())
            , m_bOffset(args["offset-b"].as<size_t>())
            , m_cOffset(args["offset-c"].as<size_t>())
            , m_dOffset(args["offset-d"].as<size_t>())
            , m_biasTypeArgs(std::vector<DataType>(1, DataType::Float))
            , m_activationType(ActivationType::None)
            , m_activationHPA(false)
            , m_activationEnumArg(std::vector<ActivationType>(1, ActivationType::None))
        {
            if(args.count("problem-identifier"))
                ContractionProblem::IdentifierToIndices(
                    args["problem-identifier"].as<std::string>(),
                    m_freeIndices,
                    m_batchIndices,
                    m_boundIndices,
                    m_aOps,
                    m_bOps,
                    m_cOps,
                    m_dOps);

            if(args.count("type"))
            {
                m_aType = m_bType = m_cType = m_dType = m_alphaType = m_betaType
                    = args["type"].as<DataType>();
            }

            if(args.count("a-type"))
                m_aType = args["a-type"].as<DataType>();
            if(args.count("b-type"))
                m_bType = args["b-type"].as<DataType>();
            if(args.count("c-type"))
                m_cType = args["c-type"].as<DataType>();
            if(args.count("d-type"))
                m_dType = args["d-type"].as<DataType>();
            if(args.count("alpha-type"))
                m_alphaType = args["alpha-type"].as<DataType>();
            if(args.count("beta-type"))
                m_betaType = args["beta-type"].as<DataType>();

            m_beta  = DataInitialization::getValue<double>(args["init-beta"].as<InitMode>());
            m_alpha = DataInitialization::getValue<double>(args["init-alpha"].as<InitMode>());

            if(args.count("bias-type-args"))
                m_biasTypeArgs = args["bias-type-args"].as<std::vector<DataType>>();

            if(args.count("activation-type"))
                m_activationType = args["activation-type"].as<ActivationType>();
            if(args.count("activation-hpa"))
                m_activationHPA = args["activation-hpa"].as<bool>();
            if(args.count("activation-enum-args"))
                m_activationEnumArg
                    = args["activation-enum-args"].as<std::vector<ActivationType>>();
            if(args.count("use-bias"))
                m_useBias = args["use-bias"].as<bool>();
            if(args.count("use-scaleD"))
                m_useScaleD = args["use-scaleD"].as<bool>();
            if(args.count("max-workspace-size"))
                m_maxWorkspaceSize = args["max-workspace-size"].as<size_t>();
            m_problems = createProblems();
        }

        ClientProblemFactory::~ClientProblemFactory() = default;

        std::vector<ContractionProblem> const& ClientProblemFactory::problems() const
        {
            return m_problems;
        }

        std::vector<ContractionProblem> ClientProblemFactory::createProblems()
        {
            std::vector<ContractionProblem> rv;
            int                             biasSize = std::max(1, (int)m_biasTypeArgs.size());
            int activationSize                       = std::max(1, (int)m_activationEnumArg.size());
            rv.reserve(m_problemSizes.size() * activationSize);

            std::vector<size_t> aStrides, bStrides, cStrides, dStrides;

            if(m_aStrides.size() == 1)
                aStrides = m_aStrides[0];
            if(m_bStrides.size() == 1)
                bStrides = m_bStrides[0];
            if(m_cStrides.size() == 1)
                cStrides = m_cStrides[0];
            if(m_dStrides.size() == 1)
                dStrides = m_dStrides[0];
            for(int k = 0; k < biasSize; k++)
            {
                for(int j = 0; j < activationSize; j++)
                {
                    for(int i = 0; i < m_problemSizes.size(); i++)
                    {
                        if(m_aStrides.size() == m_problemSizes.size())
                            aStrides = m_aStrides[i];
                        if(m_bStrides.size() == m_problemSizes.size())
                            bStrides = m_bStrides[i];
                        if(m_cStrides.size() == m_problemSizes.size())
                            cStrides = m_cStrides[i];
                        if(m_dStrides.size() == m_problemSizes.size())
                            dStrides = m_dStrides[i];

                        rv.push_back(ContractionProblem::FromIndexSizes(m_freeIndices,
                                                                        m_batchIndices,
                                                                        m_boundIndices,
                                                                        m_problemSizes[i],
                                                                        m_aType,
                                                                        aStrides,
                                                                        m_aOps,
                                                                        m_aOffset,
                                                                        m_bType,
                                                                        bStrides,
                                                                        m_bOps,
                                                                        m_bOffset,
                                                                        m_cType,
                                                                        cStrides,
                                                                        m_cOps,
                                                                        m_cOffset,
                                                                        m_dType,
                                                                        dStrides,
                                                                        m_dOps,
                                                                        m_dOffset,
                                                                        m_beta));

                        rv.back().setAlphaRestriction(toScalarValueEnum(m_alpha));
                        rv.back().setCEqualsD(m_cEqualsD);
                        rv.back().setAlphaType(m_alphaType);
                        rv.back().setBetaType(m_betaType);
                        rv.back().setStridedBatched(m_stridedBatched);
                        rv.back().setGroupedGemm(m_groupedGemm);
                        rv.back().setHighPrecisionAccumulate(m_highPrecisionAccumulate);
                        rv.back().setUseBias(m_useBias);
                        rv.back().setUseScaleD(m_useScaleD);
                        rv.back().setKernelLanguage(m_kernelLanguage);
                        rv.back().setPerformanceMetric(m_performanceMetric);
                        rv.back().setDeterministicMode(m_deterministicMode);
                        rv.back().setArithmeticUnit(m_arithmeticUnit);
                        rv.back().setFp16AltImpl(m_fp16AltImpl);
                        rv.back().setActivationType(m_activationType);
                        rv.back().setWorkspaceSize(m_maxWorkspaceSize);
                        if(k < m_biasTypeArgs.size())
                        {
                            rv.back().setBiasType(m_biasTypeArgs[k]);
                        }
                        else
                        {
                            rv.back().setBiasType(DataType::None);
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
                    }
                }
            }

            return rv;
        }
    } // namespace Client
} // namespace Tensile
