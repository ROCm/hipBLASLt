/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <functional>
#include <iomanip>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/ProblemKey.hpp>
#include <Tensile/Properties.hpp>
#include <Tensile/Utils.hpp>

// #ifdef ENABLE_ROCTX
// #include <roctracer/roctx.h>
// #endif

namespace Tensile
{
    /**
     * \ingroup Tensile
     * \defgroup PropertyMatching Property Matching
     *
     * @brief Distance-based matching of Property values to a table.
     *
     * Generic algorithm for comparing an object to a table of predefined
     * values based on Property objects and a Distance function. Used for
     * MatchingLibrary.
     */

    /**
     * \ingroup PropertyMatching
     */
    namespace Matching
    {
        template <typename Key, typename Value>
        struct MatchingTableEntry
        {
            Key    key;
            Value  value;
            double speed;
        };

        template <typename Value>
        struct KEntry
        {
            int    k;
            Value  value;
        };

        template <typename Object, typename Value, typename ReturnValue>
        struct MatchingTable
        {
            using Properties = std::vector<std::shared_ptr<Property<Object>>>;
            using Transform  = std::function<ReturnValue(Value)>;

            MatchingTable() = default;
            MatchingTable(Properties const& properties)
                : properties(properties)
            {
            }

            virtual ~MatchingTable() = default;

            virtual std::tuple<ReturnValue, double> findBestMatch(Object const& object,
                                                                  Transform     transform) const
                = 0;

            virtual std::vector<ReturnValue>
                findTopMatch(Object const& object, Transform transform, int numSolutions) const = 0;

            virtual ReturnValue findBestEvaluationSolution(Object const&   object,
                                                           Hardware const& hardware,
                                                           Transform       transform) const
                = 0;

            virtual std::vector<Value> matchesInOrder(Object const& object) const = 0;

            virtual std::vector<Value> GetAll() const = 0;

            virtual std::string description() const = 0;

            virtual std::string distanceType() const = 0;

            Properties properties;
        };

        struct Point {
            int32_t x;
            int32_t y;
        };
        class KDTree {
        private:
            struct Node {
                Point point;
                Node* left;
                Node* right;
                Node(const Point& p) : point(p), left(nullptr), right(nullptr) {}
            };

            Node* root;
            uint32_t maxM, maxN;

            int64_t distance(const Point& p1, const Point& p2) const
            {
                int64_t dx = p1.x - p2.x;
                int64_t dy = p1.y - p2.y;
                return dx * dx + dy * dy;
            }

            void insertNode(Node*& node, const Point& point, int depth)
            {
                if (node == nullptr) {
                    node = new Node(point);
                } else {
                    int cd = depth % 2;
                    if (cd == 0) {
                        if (point.x < node->point.x) {
                            insertNode(node->left, point, depth + 1);
                        } else {
                            insertNode(node->right, point, depth + 1);
                        }
                    } else {
                        if (point.y < node->point.y) {
                            insertNode(node->left, point, depth + 1);
                        } else {
                            insertNode(node->right, point, depth + 1);
                        }
                    }
                }
            }

            struct ComparePoints {
                bool operator()(const std::pair<double, Point>& p1, const std::pair<double, Point>& p2) {
                    return p1.first > p2.first;
                }
            };

            inline void pushToResult(Node* node, const Point& target, int k, std::priority_queue<std::pair<int64_t, Point>, std::vector<std::pair<int64_t, Point>>, ComparePoints>& result, double dist) const
            {
                    if (result.size() < k) {
                        result.push(std::make_pair(dist, node->point));
                    } else {
                        std::pair<int64_t, Point> p = result.top();
                        int64_t maxDist = p.first;
                        if (dist < maxDist) {
                            result.pop();
                            result.push(std::make_pair(dist, node->point));
                        }
                    }
            }

            template<int T>
            void findNearestPoints(Node* node, const Point& target, int k, std::priority_queue<std::pair<int64_t, Point>, std::vector<std::pair<int64_t, Point>>, ComparePoints>& result, int depth) const
            {
                if (node == nullptr) {
                    return;
                }

                int64_t dist = distance(node->point, target);

                if constexpr(T == 0)
                {
                    if (node->point.x >= target.x && node->point.y >= target.y) {
                        pushToResult(node, target, k, result, dist);
                    }
                }
                else if constexpr(T == 1)
                {
                    if (node->point.y >= target.y) {
                        pushToResult(node, target, k, result, dist);
                    }
                }
                else if constexpr(T == 2)
                {
                    if (node->point.x >= target.x) {
                        pushToResult(node, target, k, result, dist);
                    }
                }
                else if constexpr(T == 3)
                {
                    pushToResult(node, target, k, result, dist);
                }

                int cd = depth % 2;
                if (cd == 0) {
                    if (target.x < node->point.x) {
                        findNearestPoints<T>(node->left, target, k, result, depth + 1);
                        if (result.size() < k || std::abs(target.x - node->point.x) < dist) {
                            findNearestPoints<T>(node->right, target, k, result, depth + 1);
                        }
                    } else {
                        findNearestPoints<T>(node->right, target, k, result, depth + 1);
                        if (result.size() < k || std::abs(target.x - node->point.x) < dist) {
                            findNearestPoints<T>(node->left, target, k, result, depth + 1);
                        }
                    }
                } else {
                    if (target.y < node->point.y) {
                        findNearestPoints<T>(node->left, target, k, result, depth + 1);
                        if (result.size() < k || std::abs(target.y - node->point.y) < dist) {
                            findNearestPoints<T>(node->right, target, k, result, depth + 1);
                        }
                    } else {
                        findNearestPoints<T>(node->right, target, k, result, depth + 1);
                        if (result.size() < k || std::abs(target.y - node->point.y) < dist) {
                            findNearestPoints<T>(node->left, target, k, result, depth + 1);
                        }
                    }
                }
            }

        public:
            KDTree() : root(nullptr), maxM(0), maxN(0) {}

            void insert(const int32_t x, const int32_t y)
            {
                if(maxM < x)
                    maxM = x;
                if(maxN < y)
                    maxN = y;
                Point point = {static_cast<int32_t>(x), static_cast<int32_t>(y)};
                insertNode(root, point, 0);
            }

            std::vector<Point> findKNearestPoints(const int32_t x, const int32_t y, int k) const
            {
                Point target = {static_cast<int32_t>(x), static_cast<int32_t>(y)};

                // Create a new priority queue with the updated comparison functor
                std::priority_queue<std::pair<int64_t, Point>, std::vector<std::pair<int64_t, Point>>, ComparePoints> result;
                if(x <= maxM && y <= maxN)
                    findNearestPoints<0>(root, target, k, result, 0);
                else if(x > maxM && y < maxN)
                    findNearestPoints<1>(root, target, k, result, 0);
                else if(x < maxM && y > maxN)
                    findNearestPoints<2>(root, target, k, result, 0);
                else
                    findNearestPoints<3>(root, target, k, result, 0);

                std::vector<Point> points;
                while(!result.empty())
                {
                    points.push_back(result.top().second);
                    result.pop();
                }
                return points;
            }
        };

        /**
         * Shared code between the generic DistanceMatchingTable and the specialization
         * for the special Equality distance
         */
        template <typename Key,
                  typename Object,
                  typename Value,
                  typename ReturnValue,
                  typename Distance>
        struct DistanceMatchingCommon : public MatchingTable<Object, Value, ReturnValue>
        {
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using KEntry     = KEntry<Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;

            DistanceMatchingCommon(ReturnValue nullValue = ReturnValue())
                : nullValue(nullValue)
            {
            }

            DistanceMatchingCommon(Properties const& properties,
                                   ReturnValue       nullValue = ReturnValue())
                : Base(properties)
                , nullValue(nullValue)
            {
            }

            DistanceMatchingCommon(Distance const&   distance,
                                   Properties const& properties,
                                   ReturnValue       nullValue = ReturnValue())
                : Base(properties)
                , nullValue(nullValue)
                , distance(distance)
            {
            }

            virtual std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                                     Transform  transform) const
                = 0;

            virtual std::tuple<ReturnValue, double>
                findBestMatch(Object const& object, Transform transform) const override
            {
                return findBestKeyMatch(
                    ProblemKey::keyForProblem<Key, Object>(object, this->properties), transform);
            }

            virtual std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const = 0;

            virtual std::vector<ReturnValue> findTopMatch(Object const& object,
                                                          Transform     transform,
                                                          int           numSolutions) const override
            {
                return findTopKeyMatch(
                    ProblemKey::keyForProblem<Key, Object>(object, this->properties),
                    transform,
                    numSolutions);
            }

            virtual ReturnValue findBestEvaluationSolution(Object const&   object,
                                                           Hardware const& hardware,
                                                           Transform       transform) const override
            {
                double bestDistance = std::numeric_limits<double>::max();

                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return this->nullValue;

                ReturnValue theMatch = transform(iter->value);

                ReturnValue bestMatch = theMatch;
                if(theMatch != nullptr)
                {
                    size_t model_M          = iter->key[0];
                    size_t model_N          = iter->key[1];
                    size_t model_K          = 1;
                    size_t model_NumBatches = 1;

                    if(iter->key.size() > 3)
                    {
                        model_K          = iter->key[3];
                        model_NumBatches = iter->key[2];
                    }
                    else
                    {
                        model_K = iter->key[2];
                    }
                    bestDistance = theMatch->computeTAMScore(object,
                                                             hardware,
                                                             (double)model_M,
                                                             (double)model_N,
                                                             (double)model_K,
                                                             (double)model_NumBatches);
                }

                iter++;

                while(iter != this->table.end())
                {
                    auto nextMatch = transform(iter->value);

                    if(nextMatch != nullptr)
                    {
                        size_t model_M          = iter->key[0];
                        size_t model_N          = iter->key[1];
                        size_t model_K          = 1;
                        size_t model_NumBatches = 1;

                        if(iter->key.size() > 3)
                        {
                            model_K          = iter->key[3];
                            model_NumBatches = iter->key[2];
                        }
                        else
                        {
                            model_K = iter->key[2];
                        }
                        double nextDistance = theMatch->computeTAMScore(object,
                                                                        hardware,
                                                                        (double)model_M,
                                                                        (double)model_N,
                                                                        (double)model_K,
                                                                        (double)model_NumBatches);

                        if(nextDistance < bestDistance)
                        {
                            bestMatch    = nextMatch;
                            bestDistance = nextDistance;
                        }
                    }

                    ++iter;
                }

                return bestMatch;
            }

            virtual std::vector<Value> matchesInOrder(Object const& object) const override
            {
                return keyMatchesInOrder(
                    ProblemKey::keyForProblem<Key, Object>(object, this->properties));
            }

            std::vector<Value> keyMatchesInOrder(Key const& key) const
            {
                std::vector<std::pair<double, size_t>> indices(this->table.size());

                for(size_t i = 0; i < this->table.size(); i++)
                    indices[i] = std::make_pair(distance(key, this->table[i].key), i);

                std::sort(indices.begin(), indices.end());

                std::vector<Value> result;
                result.reserve(this->table.size());

                for(auto const& entry : indices)
                    result.push_back(this->table[entry.second].value);

                return result;
            }

            std::vector<Value> GetAll() const override
            {
                std::vector<Value> result;
                result.reserve(this->table.size());

                for(auto& data : this->table)
                    result.push_back(data.value);

                return result;
            }

            virtual std::string description() const override
            {
                std::string rv = concatenate(
                    "Table: Properties: ", this->properties, ", ", table.size(), " row(s), ");

                rv += concatenate("Distance: ", Distance::Type());

                return rv;
            }

            virtual std::string distanceType() const override
            {
                return Distance::Type();
            }

            std::vector<Entry> table;
            Distance           distance;

            ReturnValue nullValue;

            KDTree kdTree;
            std::map<std::tuple<int32_t, int32_t>, std::vector<KEntry>> kSolutionMap;
        };

        /**
         * Generic version of DistanceMatchingTable
         */
        template <typename Key,
                  typename Object,
                  typename Value,
                  typename ReturnValue,
                  typename Distance>
        struct DistanceMatchingTable
            : public DistanceMatchingCommon<Key, Object, Value, ReturnValue, Distance>
        {
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;
            using Common     = DistanceMatchingCommon<Key, Object, Value, ReturnValue, Distance>;
            using Common::distance;
            using Common::nullValue;
            using Common::table;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Common(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(properties, nullValue)
            {
            }

            DistanceMatchingTable(Distance const&   distance,
                                  Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(distance, properties, nullValue)
            {
            }

            std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                             Transform  transform) const
            {
                const bool debug = Debug::Instance().printPropertyEvaluation();
                const bool naive = Debug::Instance().naivePropertySearch();

                if(naive)
                {
                    if(debug)
                        return findBestKeyMatch_NaiveSearch<true>(key, transform);
                    else
                        return findBestKeyMatch_NaiveSearch<false>(key, transform);
                }
                else
                {
                    if(debug)
                        return findBestKeyMatch_BinSearch<true>(key, transform);
                    else
                        return findBestKeyMatch_BinSearch<false>(key, transform);
                }
            }

            std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const
            {
                ReturnValue              solution;
                std::vector<ReturnValue> solutions;
                double                   fitness;
            std:
                tie(solution, fitness) = findBestKeyMatch(key, transform);
                if(solution)
                    solutions.push_back(solution);
                return solutions;
            }

            template <bool T_Debug>
            std::tuple<ReturnValue, double> findBestKeyMatch_BinSearch(Key const& key,
                                                                       Transform  transform) const
            {
                if(this->table.empty())
                    return std::make_tuple(this->nullValue, std::numeric_limits<double>::max());

                auto comp = [](Entry const& e, Key const& key) { return e.key < key; };

                auto origIter = std::lower_bound(table.begin(), table.end(), key, comp);

                if(T_Debug)
                {
                    std::cout << "Key: ";
                    streamJoin(std::cout, key, ", ");
                    std::cout << std::endl;

                    std::cout << "Starting point: ";
                    streamJoin(std::cout, origIter->key, ", ");
                    std::cout << std::endl;

                    std::cout << "Rightward search..." << std::endl;
                }

                double bestDistance = std::numeric_limits<double>::max();
                auto   bestMatch    = this->nullValue;
                double bestSpeed    = 0.0;

                ptrdiff_t count = 0;

                for(auto iter = origIter; iter != table.end(); iter++)
                {
                    if(bestMatch && !distance.improvementPossible(key, iter->key, 0, bestDistance))
                    {
                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": Stopping rightward search early." << std::endl;
                        }

                        break;
                    }

                    count++;

                    auto myDistance = distance(key, iter->key);
                    bool thisMatch  = false;

                    if(myDistance < bestDistance
                       || (myDistance == bestDistance && iter->speed > bestSpeed))
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            bestSpeed    = iter->speed;
                            thisMatch    = true;
                        }
                    }

                    if(T_Debug)
                    {
                        if(myDistance <= bestDistance)
                            std::cout << std::endl;

                        std::cout << "speed: " << iter->speed << " | ";
                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;

                        if(myDistance < bestDistance)
                            std::cout << " < ";
                        else if(myDistance > bestDistance)
                            std::cout << " > ";
                        else
                            std::cout << " == ";

                        std::cout << bestDistance;

                        if(myDistance < bestDistance)
                        {
                            if(thisMatch)
                                std::cout << " <-- Best so far";
                            else
                                std::cout << " <-- Best distance, but no matching solution";
                        }

                        std::cout << std::endl;
                    }
                }

                auto iter = table.rbegin();

                if(origIter != table.end())
                    iter = std::make_reverse_iterator(origIter);

                if(T_Debug)
                    std::cout << "Leftward search..." << std::endl;

                for(; iter != table.rend(); iter++)
                {
                    if(bestMatch && !distance.improvementPossible(key, iter->key, 0, bestDistance))
                    {
                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": Stopping leftward search early." << std::endl;
                        }

                        break;
                    }

                    count++;

                    auto myDistance = distance(key, iter->key);
                    bool thisMatch  = false;

                    if(myDistance < bestDistance
                       || (myDistance == bestDistance && iter->speed > bestSpeed))
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            bestSpeed    = iter->speed;
                            thisMatch    = true;
                        }
                    }

                    if(T_Debug)
                    {
                        if(myDistance <= bestDistance)
                            std::cout << std::endl;

                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;

                        if(myDistance < bestDistance)
                            std::cout << " < ";
                        else if(myDistance > bestDistance)
                            std::cout << " > ";
                        else
                            std::cout << " == ";

                        std::cout << bestDistance;

                        if(myDistance < bestDistance)
                        {
                            if(thisMatch)
                                std::cout << " <-- Best so far";
                            else
                                std::cout << " <-- Best distance, but no matching solution";
                        }

                        std::cout << std::endl;
                    }
                }

                if((T_Debug || Debug::Instance().printLookupEfficiency()) && table.size() > 0)
                {
                    double considered = count;
                    considered /= table.size();
                    considered *= 100;
                    std::cout << "Considered " << considered << "% of entries." << std::endl;
                }

                if(T_Debug && bestMatch)
                    std::cout << "Solution index selected: " << bestMatch->index << std::endl;

                return std::make_tuple(bestMatch, bestDistance);
            }

            template <bool T_Debug>
            std::tuple<ReturnValue, double> findBestKeyMatch_NaiveSearch(Key const& key,
                                                                         Transform  transform) const
            {
                double bestDistance = std::numeric_limits<double>::max();

                auto iter = this->table.begin();
                if(iter == this->table.end())
                    return std::make_tuple(this->nullValue, bestDistance);

                ReturnValue bestMatch = transform(iter->value);

                if(bestMatch)
                    bestDistance = distance(key, iter->key);

                if(T_Debug)
                {
                    std::cout << "Key: ";
                    streamJoin(std::cout, key, ", ");
                    std::cout << std::endl;

                    streamJoin(std::cout, iter->key, ", ");

                    std::cout << ": " << bestDistance << " <-- First" << std::endl;
                }

                iter++;

                while(iter != this->table.end())
                {
                    auto myDistance = distance(key, iter->key);
                    bool thisMatch  = false;

                    if(myDistance < bestDistance)
                    {
                        auto myMatch = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            thisMatch    = true;
                        }
                    }

                    if(T_Debug)
                    {
                        streamJoin(std::cout, iter->key, ", ");
                        std::cout << ": " << myDistance;
                        if(myDistance < bestDistance)
                        {
                            std::cout << " <-- Best so far";

                            if(thisMatch)
                                std::cout << " (has a matching solution)";
                            else
                                std::cout << " (no match)";
                        }

                        std::cout << std::endl;
                    }

                    iter++;
                }

                if(T_Debug && bestMatch)
                    std::cout << "Solution index selected: " << bestMatch->index << std::endl;

                return std::make_tuple(bestMatch, bestDistance);
            }
        };

        /**
         * Specialization of DistanceMatchingTable for Equality Distance. This special case will
         * only select key in the table if it exactly matches the provided key
         */
        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct DistanceMatchingTable<Key, Object, Value, ReturnValue, Matching::Equality<Key>>
            : public DistanceMatchingCommon<Key,
                                            Object,
                                            Value,
                                            ReturnValue,
                                            Matching::Equality<Key>>
        {
            using Base       = MatchingTable<Object, Value, ReturnValue>;
            using Entry      = MatchingTableEntry<Key, Value>;
            using Transform  = typename Base::Transform;
            using Properties = typename Base::Properties;
            using Equality   = Matching::Equality<Key>;
            using Common
                = DistanceMatchingCommon<Key, Object, Value, ReturnValue, Matching::Equality<Key>>;
            using Common::distance;
            using Common::nullValue;
            using Common::table;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Common(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(properties, nullValue)
            {
            }

            DistanceMatchingTable(Equality const&   distance,
                                  Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(distance, properties, nullValue)
            {
            }

            std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                             Transform  transform) const
            {
                auto comp = [](Entry const& e, Key const& key) { return e.key < key; };
                auto iter = std::lower_bound(table.begin(), table.end(), key, comp);

                return (iter->key == key)
                           ? std::make_tuple(transform(iter->value), 0.0)
                           : std::make_tuple(this->nullValue, std::numeric_limits<double>::max());
            }

            std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const
            {
                ReturnValue              solution;
                std::vector<ReturnValue> solutions;
                double                   fitness;
            std:
                tie(solution, fitness) = findBestKeyMatch(key, transform);
                if(solution)
                    solutions.push_back(solution);
                return solutions;
            }
        };

        /**
         * Specialization of DistanceMatchingTable for GridBased Distance. This special case will
         * use a more efficient search algorithm that depends on the grid structure
         */
        template <typename Key, typename Object, typename Value, typename ReturnValue>
        struct DistanceMatchingTable<Key,
                                     Object,
                                     Value,
                                     ReturnValue,
                                     Matching::GridBasedDistance<Key>>
            : public DistanceMatchingCommon<Key,
                                            Object,
                                            Value,
                                            ReturnValue,
                                            Matching::GridBasedDistance<Key>>
        {
            using Base              = MatchingTable<Object, Value, ReturnValue>;
            using Entry             = MatchingTableEntry<Key, Value>;
            using Transform         = typename Base::Transform;
            using Properties        = typename Base::Properties;
            using GridBasedDistance = Matching::GridBasedDistance<Key>;
            using Common            = DistanceMatchingCommon<Key,
                                                  Object,
                                                  Value,
                                                  ReturnValue,
                                                  Matching::GridBasedDistance<Key>>;
            using Common::distance;
            using Common::nullValue;
            using Common::table;

            DistanceMatchingTable(ReturnValue nullValue = ReturnValue())
                : Common(nullValue)
            {
            }

            DistanceMatchingTable(Properties const& properties,
                                  ReturnValue       nullValue = ReturnValue())
                : Common(properties, nullValue)
            {
            }

            DistanceMatchingTable(GridBasedDistance const& distance,
                                  Properties const&        properties,
                                  ReturnValue              nullValue = ReturnValue())
                : Common(distance, properties, nullValue)
            {
            }

            std::tuple<ReturnValue, double> findBestKeyMatch(Key const& key,
                                                             Transform  transform) const
            {
                std::vector<ReturnValue> solutions = findTopKeyMatch(key, transform, 1);
                ReturnValue              solution  = this->nullValue;
                if(solutions.size() > 0)
                    solution = solutions[0];
                return std::make_tuple(solution, std::numeric_limits<double>::max());
            }

            std::vector<ReturnValue>
                findTopKeyMatch(Key const& key, Transform transform, int numSolutions) const
            {
                if(Debug::Instance().printPropertyEvaluation())
                    return findBestKeyMatch_GridBased<true>(key, transform, numSolutions);
                else
                    return findBestKeyMatch_GridBased<false>(key, transform, numSolutions);
            }

            template <bool T_Debug>
            std::vector<ReturnValue> findBestKeyMatch_GridBased(Key const& key,
                                                                Transform  transform,
                                                                int        numSolutions) const
            {
                std::vector<ReturnValue> bestmatches;
                if(this->table.empty())
                {
                    return bestmatches;
                }

                ptrdiff_t count = 0;
                bool      Debug = T_Debug;
                std::cout << std::setprecision(2) << std::fixed;

                if(Debug::Instance().gridBasedKDTree())
                {
                    // roctxRangePush("KDTree");
                    auto compK = [](KEntry<Value> const& e, int const N) {
                        return e.k < N;
                    };

                    auto k = key.size() > 3 ? key[3] : key[2];
                    std::vector<Point> points = this->kdTree.findKNearestPoints(key[0], key[1], numSolutions);
                    for(auto point : points)
                    {
                        if(Debug)
                            std::cout << "2D point: " << point.x << ", " << point.y << std::endl;
                        auto iter = this->kSolutionMap.find(std::make_tuple(point.x, point.y));
                        if(iter != this->kSolutionMap.end())
                        {
                            auto lower = std::lower_bound(iter->second.begin(), iter->second.end(), k, compK);
                            if (lower != iter->second.end()) {
                                auto prev = lower == iter->second.begin() ? lower : lower--;
                                if(prev != lower)
                                {
                                    if(std::abs(prev->k - k) < std::abs(lower->k - k))
                                    {
                                        lower = prev;
                                    }
                                }
                                auto thisMatch = transform(lower->value);
                                if(thisMatch)
                                {
                                    if(bestmatches.size())
                                    {
                                        if(std::find(bestmatches.begin(), bestmatches.end(), thisMatch) != bestmatches.end())
                                        {
                                            if(Debug)
                                                std::cout << "Duplicated solution" << std::endl;
                                            continue;
                                        }
                                    }
                                    if(Debug)
                                        std::cout << "Final selected points: " << point.x << ", " << point.y << " K: " << lower->k << std::endl;

                                    bestmatches.push_back(thisMatch);
                                }
                            }
                        }
                    }
                    // roctxRangePop();
                    return bestmatches;
                }

                // roctxRangePush("Binary");

                auto compM = [&count, Debug](Entry const& e, long const M) {
                    if(Debug)
                        printf("[%ld,%ld,%ld]\n", e.key[0], e.key[1], e.key[2]);
                    count++;
                    return e.key[0] < M;
                };

                auto compN = [&count, Debug](Entry const& e, long const N) {
                    if(Debug)
                        printf("[%ld,%ld,%ld]\n", e.key[0], e.key[1], e.key[2]);
                    count++;
                    return e.key[1] < N;
                };

                auto origIter_M_lower = table.begin();
                auto origIter_M_upper = table.begin();
                auto origIter_N_lower = table.begin();
                auto origIter_N_upper = table.begin();
                auto start            = table.begin();

                double bestDistance = std::numeric_limits<double>::max();
                auto   bestMatch    = this->nullValue;
                bool   thisMatch    = false;
                bool   uniqueSol    = false;
                int    baseN        = 0;
                int    stepN        = 0;
                for(int i = 1;; i++)
                {
                    if(i * (i + 1) / 2 >= numSolutions)
                    {
                        baseN = i;
                        break;
                    }
                }

                while(origIter_N_upper != table.end() and bestmatches.size() < numSolutions)
                {
                    bestDistance = std::numeric_limits<double>::max();
                    thisMatch    = false;
                    uniqueSol    = false;
                    if(T_Debug)
                    {
                        std::cout << "Searching next MN... ";
                        std::cout << std::endl << std::endl;
                    }

                    if(stepN == baseN && baseN != 1)
                    {
                        stepN = 1;
                        baseN--;
                        start = origIter_M_upper;
                    }
                    else
                    {
                        stepN++;
                        start = origIter_N_upper;
                    }

                    origIter_M_lower = std::lower_bound(
                        start, table.end(), std::min(key[0], (table.end() - 1)->key[0]), compM);

                    if(T_Debug)
                    {
                        std::cout << "M lower: ";
                        streamJoin(std::cout, origIter_M_lower->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_M_upper = std::lower_bound(
                        origIter_M_lower,
                        table.end(),
                        std::min(origIter_M_lower->key[0] + 1, (table.end() - 1)->key[0] + 1),
                        compM);

                    if(T_Debug)
                    {
                        std::cout << "M upper: ";
                        streamJoin(std::cout, (origIter_M_upper - 1)->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_N_lower
                        = std::lower_bound(origIter_M_lower,
                                           origIter_M_upper,
                                           std::min(key[1], (origIter_M_upper - 1)->key[1]),
                                           compN);

                    if(T_Debug)
                    {
                        std::cout << "N lower: ";
                        streamJoin(std::cout, origIter_N_lower->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    origIter_N_upper = std::lower_bound(
                        origIter_N_lower,
                        origIter_M_upper,
                        std::min(origIter_N_lower->key[1] + 1, (origIter_M_upper - 1)->key[1] + 1),
                        compN);

                    if(T_Debug)
                    {
                        std::cout << "N upper: ";
                        streamJoin(std::cout, (origIter_N_upper - 1)->key, ", ");
                        std::cout << std::endl << std::endl;

                        std::cout << "K start point: ";
                        streamJoin(std::cout, origIter_N_lower->key, ", ");
                        std::cout << std::endl << std::endl;

                        std::cout << "K End point: ";
                        streamJoin(std::cout, (origIter_N_upper - 1)->key, ", ");
                        std::cout << std::endl << std::endl;
                    }

                    for(auto iter = origIter_N_lower; iter != origIter_N_upper; iter++)
                    {

                        count++;

                        auto myDistance = distance(key, iter->key);

                        if(myDistance < bestDistance)
                        {
                            auto myMatch = transform(iter->value);

                            if(myMatch)
                            {
                                bestDistance = myDistance;
                                bestMatch    = myMatch;
                                thisMatch    = true;
                            }
                        }

                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": " << myDistance;

                            if(myDistance < bestDistance)
                                std::cout << " < ";
                            else if(myDistance > bestDistance)
                                std::cout << " > ";
                            else
                                std::cout << " == ";

                            std::cout << bestDistance;

                            if(myDistance < bestDistance)
                            {
                                if(thisMatch)
                                    std::cout << " <-- Best so far";
                                else
                                    std::cout << " <-- Best distance, but no matching solution";
                            }

                            std::cout << std::endl << std::endl;
                        }
                    }
                    if(thisMatch or bestmatches.size())
                    {
                        if(std::find(bestmatches.begin(), bestmatches.end(), bestMatch)
                           == bestmatches.end())
                        {
                            bestmatches.push_back(bestMatch);
                            uniqueSol = true;
                        }
                        if(T_Debug)
                        {
                            if(uniqueSol)
                                std::cout << "add Best to Top Solutions" << std::endl << std::endl;
                            else
                                std::cout << "Best solution is duplicated" << std::endl
                                          << std::endl;
                        }
                    }
                }

                if(bestmatches.size() < numSolutions)
                {
                    if(T_Debug)
                    {
                        std::cout << std::endl
                                  << "Foward Search end but solution not found" << std::endl;
                        std::cout << "Start to backward search..." << std::endl;
                    }

                    for(auto iter = std::make_reverse_iterator(origIter_N_lower);
                        iter != table.rend();
                        iter++)
                    {
                        thisMatch       = false;
                        uniqueSol       = false;
                        auto myDistance = distance(key, iter->key);
                        auto myMatch    = transform(iter->value);

                        if(myMatch)
                        {
                            bestDistance = myDistance;
                            bestMatch    = myMatch;
                            thisMatch    = true;
                            if(std::find(bestmatches.begin(), bestmatches.end(), bestMatch)
                               == bestmatches.end())
                            {
                                bestmatches.push_back(bestMatch);
                                uniqueSol = true;
                            }
                        }

                        if(T_Debug)
                        {
                            streamJoin(std::cout, iter->key, ", ");
                            std::cout << ": " << myDistance;
                            if(thisMatch)
                                if(uniqueSol)
                                    std::cout << " (has a matching solution)";
                                else
                                    std::cout << " (duplicated solution)";
                            else
                                std::cout << " (no match)";
                            std::cout << std::endl;
                        }

                        if(bestmatches.size() == numSolutions)
                            break;
                    }
                }

                if((T_Debug || Debug::Instance().printLookupEfficiency()) && table.size() > 0)
                {
                    double considered = count;
                    considered /= table.size();
                    considered *= 100;
                    std::cout << "Considered " << count << "(" << considered << "%) of entries."
                              << std::endl;
                }

                if(T_Debug && bestmatches.size())
                    std::cout << "Solution index selected: " << bestmatches[0]->index << std::endl;

                // roctxRangePop();
                return bestmatches;
            }
        };
    } // namespace Matching
} // namespace Tensile
