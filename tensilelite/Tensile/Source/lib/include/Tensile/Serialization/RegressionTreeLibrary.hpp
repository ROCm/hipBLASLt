/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Debug.hpp>
#include <Tensile/RegressionTreeLibrary.hpp>

#include <cstddef>
#include <unordered_set>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename Key, typename ReturnValue, typename IO>
        struct MappingTraits<RegressionTree::Tree<Key, ReturnValue>, IO>
        {
            using Tree = RegressionTree::Tree<Key, ReturnValue>;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Tree& tree)
            {
                iot::mapRequired(io, "tree", tree.tree);
            }
            const static bool flow = false;
        };

        template <typename Key, typename ReturnValue, typename IO>
        struct MappingTraits<RegressionTree::BasicForest<Key, ReturnValue>, IO>
        {
            using Forest = RegressionTree::BasicForest<Key, ReturnValue>;
            using iot    = IOTraits<IO>;

            static void mapping(IO& io, Forest& lib)
            {
                int32_t index = -1;
                iot::mapRequired(io, "trees", lib.trees);
            }

            const static bool flow = false;
        };

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<RegressionTreeLibrary<MyProblem, MySolution>, IO>
        {
            using Library = RegressionTreeLibrary<MyProblem, MySolution>;
            using iot = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "RegressionTreeLibrary requires that context be "
                                  "set to a SolutionMap.");
                }
                std::vector<int> mappingIndices;
                if(iot::outputting(io))
                {
                    mappingIndices.reserve(lib.solutionmap.size());

                    for(auto const& pair : lib.solutionmap)
                        mappingIndices.push_back(pair.first);

                    iot::mapRequired(io, "table", mappingIndices);
                }
                else
                {
                    iot::mapRequired(io, "table", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "RegressionTreeLibrary requires non empty "
                                      "mapping index set.");

                    for(int index : mappingIndices)
                    {
                        auto slnIter = ctx->solutions->find(index);
                        if(slnIter == ctx->solutions->end())
                        {
                            iot::setError(
                                io,
                                concatenate("[RegressionTreeLibrary] Invalid solution index: ",
                                            index));
                        }
                        else
                        {
                            auto solution = slnIter->second;
                            lib.solutionmap.insert(std::make_pair(index, solution));
                        }
                    }
                }

                using Forest = RegressionTree::BasicForest<std::vector<float>, float>;
                std::shared_ptr<Forest> forest;
                if(iot::outputting(io))
                {
                    forest = std::dynamic_pointer_cast<Forest>(lib.forest);
                }
                else
                {
                    forest     = std::make_shared<Forest>();
                    lib.forest = forest;
                }
                MappingTraits<Forest, IO>::mapping(io, *forest);

                using SolutionFeatures
                    = std::vector<std::shared_ptr<MLFeatures::MLFeature<MySolution>>>;
                SolutionFeatures solFeatures;
                if(iot::outputting(io))
                {
                    solFeatures = lib.solFeatures;
                }
                iot::mapOptional(io, "solutionFeatures", solFeatures);
                lib.solFeatures = solFeatures;

                using ProblemFeatures
                    = std::vector<std::shared_ptr<MLFeatures::MLFeature<MyProblem>>>;
                ProblemFeatures probFeatures;
                if(iot::outputting(io))
                {
                    probFeatures = lib.probFeatures;
                }
                iot::mapOptional(io, "problemFeatures", probFeatures);
                lib.probFeatures = probFeatures;
            }
            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<RegressionTree::Node, IO>
        {
            using Node = typename RegressionTree::Node;
            using iot  = IOTraits<IO>;

            static void mapping(IO& io, Node& node)
            {
                iot::mapRequired(io, "featureIdx", node.featureIdx);
                iot::mapRequired(io, "threshold", node.threshold);
                iot::mapRequired(io, "nextIdxLTE", node.nextIdxLTE);
                iot::mapRequired(io, "nextIdxGT", node.nextIdxGT);
            }

            const static bool flow = true;
        };

    } // namespace Serialization
} // namespace TensileLite
