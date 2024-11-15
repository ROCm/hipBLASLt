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

#include <array>
#include <functional>
#include <vector>

namespace TensileLite
{
    /**
     * \ingroup Tensile
     * \defgroup RegressionTree Regression Tree
     *
     * @brief Regression model to estimate efficiencies
     *
     * Group of trees used to estimate efficiency values for solutions in the
     * library. Used for RegressionTreeLibrary.
     */

    /**
     * \ingroup RegressionTree
     */
    namespace RegressionTree
    {
        struct Node
        {
            int   featureIdx; // Index into feature array
            float threshold; // Decision threshold value
            float nextIdxLTE; // Next node index if val <= threshold, may be leaf value
            float nextIdxGT; // Next node index if val > threshold, may be leaf value
        };

        /**
         * @brief Generic regression tree
         *
         * @tparam Key type used for deciding
         * @tparam ReturnValue type returned by tree
         */
        template <typename Key, typename ReturnValue>
        struct Tree
        {

            Tree() = default;
            Tree(std::vector<Node> tree)
                : tree(std::move(tree))
            {
            }

            float predict(Key const& key) const
            {
                int   nodeIdx  = 0;
                int   treeSize = tree.size();
                Node  currentNode;
                float treeValue;

                while(nodeIdx < treeSize)
                {
                    currentNode    = tree[nodeIdx];
                    bool branchLTE = key[currentNode.featureIdx] <= currentNode.threshold;
                    treeValue      = branchLTE ? currentNode.nextIdxLTE : currentNode.nextIdxGT;

                    if(treeValue < 1.0f)
                        return treeValue;
                    else
                        nodeIdx = static_cast<int>(treeValue);
                }

                throw std::runtime_error("Regression Tree out of bounds error.");
                return false;
            }

            bool valid(bool verbose = false) const
            {
                size_t treeSize = tree.size();
                Node   currentNode;
                bool   valid = true;

                if(treeSize == 0)
                {
                    if(verbose)
                    {
                        std::cout << "Tree invalid: no nodes." << std::endl;
                    }
                    return false;
                }

                if(treeSize > ((size_t)std::numeric_limits<signed int>::max() + 1))
                {
                    /* Restrict size to +ve int range, -ve idxs for reserved values */
                    if(verbose)
                    {
                        std::cout << "Tree invalid: too many nodes." << std::endl;
                    }
                    return false;
                }

                // Check for any invalid nodes
                for(int nodeIdx = 0; nodeIdx < treeSize; nodeIdx++)
                {
                    currentNode = tree[nodeIdx];

                    // Avoid OOB on feature array
                    if((currentNode.featureIdx < 0)
                       || (currentNode.featureIdx >= std::tuple_size<Key>::value))
                    {
                        if(verbose)
                        {
                            std::cout << "Node " << std::to_string(nodeIdx)
                                      << " invalid: Unrecognised type '"
                                      << std::to_string(currentNode.featureIdx) << "'" << std::endl;
                        }
                        valid = false;
                    }
                }
                return valid;
            }

            std::vector<Node> tree;
        };

        /**
         * @brief Abstract base class for a group of regression trees
         *
         * @tparam Key used to query trees
         * @tparam ReturnValue type returned by trees
         */
        template <typename Key, typename ReturnValue>
        struct Forest
        {
            Forest() = default;

            virtual ~Forest() = default;

            virtual ReturnValue computeEfficiency(Key const& inputParameters) const = 0;

            virtual std::string description() const = 0;
        };

        /**
         * @brief Forest that returns value from trees
         * @tparam Key used to query trees
         * @tparam ReturnValue type returned by trees
         */
        template <typename Key, typename ReturnValue>
        struct BasicForest : public Forest<Key, ReturnValue>
        {
            using Base = Forest<Key, ReturnValue>;
            using Tree = Tree<Key, ReturnValue>;

            BasicForest() {}

            virtual ReturnValue computeEfficiency(Key const& inputParameters) const override
            {
                bool debug = Debug::Instance().getSolutionSelectionTrace();
                if(debug)
                {
                    std::cout << "Forest " << this->description() << std::endl;
                    std::cout << "Entering solution selection evaluation loop. Searching forest."
                              << std::endl;
                }

                ReturnValue rv      = 0;
                size_t      treenum = 0;
                for(Tree const& tree : trees)
                {

                    if(debug)
                        std::cout << "Running predict tree: " << treenum << std::endl;

                    ReturnValue result = tree.predict(inputParameters);
                    rv += result;
                    if(debug)
                    {
                        std::cout << "Tree " << treenum << " predicts: " << result << std::endl;
                        std::cout << "Accummulated value: " << rv << std::endl;
                    }
                }
                return rv;
            }

            virtual std::string description() const override
            {
                return concatenate("RegressionTree Forest: Number of trees ", trees.size());
            }

            std::vector<Tree> trees;
        };
    } // namespace RegressionTree
} // namespace TensileLite
