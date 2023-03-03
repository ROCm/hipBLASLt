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

#include <Tensile/Comparison.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

namespace Tensile
{
    template <>
    struct Comparison<ContractionProblemGemm::FreeIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm::FreeIndex const& lhs,
                           ContractionProblemGemm::FreeIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d, lhs.c, rhs.c, lhs.i, rhs.i, lhs.isA, rhs.isA);
        }
    };

    template <>
    struct Comparison<ContractionProblemGemm::BatchIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm::BatchIndex const& lhs,
                           ContractionProblemGemm::BatchIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d, lhs.c, rhs.c, lhs.a, rhs.a, lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblemGemm::BoundIndex>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm::BoundIndex const& lhs,
                           ContractionProblemGemm::BoundIndex const& rhs)
        {
            return LexicographicCompare(lhs.a, rhs.a, lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblemGemm>
    {
        enum
        {
            implemented = true
        };

        static int compare(ContractionProblemGemm const& lhs, ContractionProblemGemm const& rhs)
        {
            return LexicographicCompare(lhs.operationIdentifier(),
                                        rhs.operationIdentifier(),
                                        lhs.highPrecisionAccumulate(),
                                        rhs.highPrecisionAccumulate(),
                                        lhs.kernelLanguage(),
                                        rhs.kernelLanguage(),
                                        lhs.deterministicMode(),
                                        rhs.deterministicMode(),
                                        lhs.a(),
                                        rhs.a(),
                                        lhs.b(),
                                        rhs.b(),
                                        lhs.c(),
                                        rhs.c(),
                                        lhs.d(),
                                        rhs.d(),
                                        lhs.workspaceSize(),
                                        rhs.workspaceSize(),
                                        lhs.stridedBatched(),
                                        rhs.stridedBatched(),
                                        lhs.groupedGemm(),
                                        rhs.groupedGemm(),
                                        lhs.performanceMetric(),
                                        rhs.performanceMetric(),
                                        lhs.fp16AltImpl(),
                                        rhs.fp16AltImpl(),
                                        lhs.activationType(),
                                        rhs.activationType(),
                                        lhs.activationHPA(),
                                        rhs.activationHPA(),
                                        lhs.useBias(),
                                        rhs.useBias(),
                                        lhs.useE(),
                                        rhs.useE(),
                                        lhs.useScaleD(),
                                        rhs.useScaleD());
        }
    };
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::ContractionProblemGemm>
    {
        inline size_t operator()(Tensile::ContractionProblemGemm const& problem) const
        {
            return Tensile::hash_combine(problem.operationIdentifier(),
                                         problem.a(),
                                         problem.b(),
                                         problem.c(),
                                         problem.d(),
                                         problem.highPrecisionAccumulate(),
                                         problem.kernelLanguage(),
                                         problem.deterministicMode(),
                                         problem.workspaceSize(),
                                         problem.stridedBatched(),
                                         problem.groupedGemm(),
                                         problem.performanceMetric(),
                                         problem.fp16AltImpl(),
                                         problem.activationType(),
                                         problem.activationHPA(),
                                         problem.useBias(),
                                         problem.useE(),
                                         problem.useScaleD());
        }
    };

    template <>
    struct hash<std::vector<Tensile::ContractionProblemGemm>>
    {
        inline size_t operator()(std::vector<Tensile::ContractionProblemGemm> const& problems) const
        {
            size_t hash = 0;
            for(int idx = 0; idx < problems.size(); idx++)
            {
                auto problem = problems[idx];
                hash += Tensile::hash_combine(problem.operationIdentifier(),
                                              problem.a(),
                                              problem.b(),
                                              problem.c(),
                                              problem.d(),
                                              problem.highPrecisionAccumulate(),
                                              problem.kernelLanguage(),
                                              problem.deterministicMode(),
                                              problem.workspaceSize(),
                                              problem.stridedBatched(),
                                              problem.groupedGemm(),
                                              problem.performanceMetric(),
                                              problem.fp16AltImpl(),
                                              problem.activationType(),
                                              problem.activationHPA(),
                                              problem.useBias(),
                                              problem.useScaleD());
            }
            return hash;
        }
    };

} // namespace std
