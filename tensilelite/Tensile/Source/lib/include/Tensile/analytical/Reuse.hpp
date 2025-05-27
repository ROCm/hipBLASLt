/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <vector>
#include <unordered_map>
#include <list>
#include <tuple>
#include <algorithm>
#include <Tensile/analytical/Hardware.hpp>
// #include "Hardware.hpp"

namespace TensileLite
{
    namespace analytical
    {

        // Function to reorder CU contiguous where a CU gets a contiguous set of work across the whole problem.
        size_t reorder_cu_contiguous_hash(size_t value, size_t total_elements, size_t num_cus_per_xcd, size_t num_xcds);

        // Reorder such that a CU gets a contiguous set of work.
        void reorder_cu_contiguous(std::vector<size_t> &remapped_vector, size_t num_cus_per_xcd, size_t num_xcds);

        // A hash to reorder the WG vector such that a single XCD gets contiguous WGs over WHOLE PROBLEM (not their # CUS)
        size_t reorder_xcd_contiguous_hash(size_t value, size_t total_elements, size_t num_cus_per_xcd, size_t num_xcds);

        // Reorder such that an XCD gets a contiguous set of functions.
        void reorder_xcd_contiguous(std::vector<size_t> &remapped_vector, size_t num_cus_per_xcd, size_t num_xcds);

        // Hash WGID Column-Major
        size_t hash_wgid_col(const std::vector<size_t> &dimensions, size_t input, bool debug, size_t debug_value);

        // Hilbert curve mapping function
        size_t hilbert_index_to_xy(size_t n, size_t d);

        /*
        The core hash_wgid function performs a mapping of the 1-D vector index, to the n-d vector index. So inputting 1 gives the projection of index 1 into a higher dimensional traverasal.
        radix : [4,4,2,2] for example can be viewed as a remapping of a vector originally structured as:

        1-D : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        Representing a 4x4 grid:
        2-D :
        [0  1   2  3
        4  5   6  7
        8  9  10 11
        12 13 14 15]

        to

        into a vector of 4x4 elements in 2x2 Z-order:
        3-D (Abstracting a group of 2x2 into 0,1,2,3):
        [0 1
        2 3]

        2-D:
        [0   1   4  5
        2   3   6  7
        8   9  12 13
        10 11  14 15]

        1-D:
        [0 1 4 5 2 3 6 7 8 9 12 13 10 11 14 15]

        It is profoundly important that this transformation is hash-like because this means if we know the dimensions of the grid and a 1-D index (such as WGID)
        we can STATICALLY and IN PARALLLEL map to the N-Dimensional output space from our 1-D idea. Meaning we can compute the transformation of all original members of the original vector
        in parallel.

        */
        size_t hash_wgid(const std::vector<size_t> &dimensions, size_t input, bool debug, size_t debug_value);

        // Function to calculate the product of elements in a vector
        size_t vector_product(const std::vector<size_t> &vec);

        // LRUCache class:
        // This class implements a Least Recently Used (LRU) cache to emulate the storage of tile objects in memory.
        // It is designed to simulate how tiles are managed in computations like matrix multiplication, with A, B, and C tiles having different sizes
        // The cache uses a CacheKey (comprising a type and indices) to uniquely identify each tile.
        // When the cache reaches its capacity, it evicts the least recently used tiles to make room for new ones.
        // This helps in modeling cache behavior and analyzing data reuse and cache efficiency in tile-based algorithms.
        class LRUCache
        {
        public:
            // Define a CacheKey struct
            struct CacheKey
            {
                char type; // 'A', 'B', 'C', etc.
                size_t idx1;
                size_t idx2;

                // Define equality operator
                bool operator==(const CacheKey &other) const
                {
                    return type == other.type && idx1 == other.idx1 && idx2 == other.idx2;
                }
            };

            // Hash function for CacheKey to be used in unordered_map
            struct CacheKeyHash
            {
                std::size_t operator()(const CacheKey &k) const
                {
                    return ((std::hash<char>()(k.type) ^ (std::hash<size_t>()(k.idx1) << 1)) >> 1) ^ (std::hash<size_t>()(k.idx2) << 1);
                }
            };

            LRUCache(size_t capacity)
            {
                max_capacity = capacity;
                current_size = 0;
            }

            bool contains(const CacheKey &key)
            {
                return cache_map.find(key) != cache_map.end();
            }

            void append(const CacheKey &key, size_t size)
            {
                // If key already exists, remove it to update its position
                if (cache_map.find(key) != cache_map.end())
                {
                    cache_list.erase(cache_map[key]);
                    current_size -= size; // Adjust current size
                }
                else
                {
                    // If cache is full, remove the least recently used item
                    while (current_size + size > max_capacity && !cache_list.empty())
                    {
                        CacheKey lru_key = cache_list.back();
                        cache_list.pop_back();
                        current_size -= entry_size_map[lru_key];
                        cache_map.erase(lru_key);
                        entry_size_map.erase(lru_key);
                    }
                }
                // Add the key to the front of the list
                cache_list.push_front(key);
                cache_map[key] = cache_list.begin();
                entry_size_map[key] = size;
                current_size += size;
            }

        private:
            size_t max_capacity;
            size_t current_size;
            std::list<CacheKey> cache_list;
            std::unordered_map<CacheKey, std::list<CacheKey>::iterator, CacheKeyHash> cache_map;
            std::unordered_map<CacheKey, size_t, CacheKeyHash> entry_size_map;
        };

        /*
        The `generalized_mix_radix_with_nonquantized_region` function reorders a given input vector to follow a **Z-order (Morton order) traversal**.
        his traversal is designed to preserve spatial locality, which can enhance cache performance and data processing efficiency.

        **Understanding Z-order Traversal:**

        Z-order traversal can be visualized by recursively splitting the grid and ordering the subgrids in a Z-shaped pattern. In a 2D grid, it interleaves the bits of the row and column indices to compute the Z-order index.


        [4,4,2,2] For example
        Remaps a vector of 4x4 elements:
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        Representing a 4x4 grid:
        [0  1   2  3
        4  5   6  7
        8  9  10 11
        12 13 14 15]
        into a vector of 4x4 elements in 2x2 Z-order:
        [0   1   4  5
        2   3   6  7
        8   9  12 13
        10 11  14 15]

        [4,4,1,4]
        Would map that same vector to:
        [0 4  8 12
        1 5  9 13
        2 6 10 14
        3 7 11 15]

        This can be viewed as factoring a n-dimensional set of loops out of the 2-D structure and representing it as a 1-D structure.

        The core hash_wgid function performs a mapping of the 1-D vector index, to the n-d vector index. So inputting 1 gives the projection of index 1 into a higher dimensional traverasal.
        */
        std::vector<size_t> generalized_mix_radix_with_nonquantized_region(
            const std::vector<size_t> &dimensions,
            const std::vector<size_t> &input_vector,
            bool debug,
            size_t debug_value,
            bool col_major,
            bool group_l2,
            bool hilbert,
            bool group_l1,
            size_t num_cus_per_xcd,
            size_t num_xcds);

        // Simulate LRU cache behavior of a blocked GEMM algorithm
        double compute_reuse_in_block_gemm(
            size_t grid_m,                      // Number of MT in the M dimension ceil(M/BLK_M)
            size_t grid_n,                      // Number of MT in the N dimension ceil(N/BLK_N)
            size_t grid_k,                      // Number of MT in the K dimension ceil(K/BLK_K)
            size_t A_size = 1,                  // Size of the A input mt (BLK_M*BLK_K)
            size_t B_size = 1,                  // Size of the B input mt (BLK_N*BLK_K)
            size_t C_size = 1,                  // Size of the C output mt (BLK_M*BLK_N)
            size_t nproc = 1,                   // Number of concurrent processors in cache scope
            size_t capacity = 1024,             // Capacity of cache in #elements
            std::vector<size_t> radix = {2, 2}, // Z order transform of processing order. Can be as long as you want for nested Z order.
            bool print_radix = true,            // Prsize_t traversal order
            bool print_output = false,          // Prsize_t reuse factor output
            size_t num_iterations = -1          // Early Exit option - Typically converges after 10-20 on most problems
        );

        // Compute the L2 hit rate of a blocked GEMM
        double compute_L2_hit_rate(const Hardware &hardware, size_t M, size_t N, size_t K, size_t MT_M, size_t MT_N, size_t MT_K, size_t WGM, size_t num_iterations = 25, bool debug = false);

    } // namespace analytical
} // namespace TensileLite
