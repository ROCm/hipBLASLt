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

#pragma once

#include <Tensile/Activation.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/Tensile.hpp>

#include <boost/program_options.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {

        namespace po = boost::program_options;

        class ClientProblemFactory
        {
        public:
            ClientProblemFactory(po::variables_map const& args);
            ~ClientProblemFactory();

            std::vector<std::shared_ptr<ContractionProblem>> const& problems() const;

            size_t workspaceSize() const
            {
                return m_maxWorkspaceSize;
            }

        private:
            void createProblems(std::vector<ContractionProblemGemm>& rv);

            std::vector<std::shared_ptr<ContractionProblem>> m_problems;

            ContractionProblemGemm::FreeIndices  m_freeIndices;
            ContractionProblemGemm::BatchIndices m_batchIndices;
            ContractionProblemGemm::BoundIndices m_boundIndices;

            std::vector<DataType>                         m_tensorTypes;
            std::vector<std::vector<std::vector<size_t>>> m_tensorStrides;

            std::vector<DataType> m_constantTypes;
            std::vector<double>   m_constantValues;

            bool m_stridedBatched;
            bool m_groupedGemm;
            bool m_highPrecisionAccumulate;
            bool m_deterministicMode;
            bool m_cEqualsD;
            bool m_useBias;
            int  m_biasSrc;
            bool m_useScaleAB;
            bool m_useScaleCD;
            bool m_useScaleAlphaVec;
            bool m_useE;
            bool m_useGradient = false;

            int                              m_sparse;
            KernelLanguage                   m_kernelLanguage;
            PerformanceMetric                m_performanceMetric;
            bool                             m_fp16AltImpl;
            ActivationType                   m_activationType;
            std::vector<DataType>            m_biasTypeArgs;
            bool                             m_activationNoGuard;
            std::vector<ActivationType>      m_activationEnumArg;
            size_t                           m_maxWorkspaceSize = 0;
            DataType                         m_computeInputType;
            DataType                         m_f32XdlMathOp;
            DataType                         m_activationComputeType;
            std::vector<std::vector<size_t>> m_problemSizes;
            bool                             m_useUserArgs;
            int                              m_debugBreakPoints;
        };

    } // namespace Client
} // namespace Tensile
