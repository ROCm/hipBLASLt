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

#include <Tensile/analytical/Reuse.hpp>
// #include "Reuse.hpp"

#include <iostream>
#include <cmath>
#include <numeric>
#include <cassert>
#include <string>
#include <chrono>
#include <getopt.h> // For command-line argument parsing
#include <cstdlib>
#include <cstring>

namespace TensileLite
{
    namespace analytical
    {

        // Function to reorder CU contiguous where a CU gets a contiguous set of work across the whole problem.
        size_t reorder_cu_contiguous_hash(size_t value, size_t total_elements, size_t num_cus_per_xcd, size_t num_xcds)
        {
            size_t num_cu = num_cus_per_xcd * num_xcds;
            size_t full_timesteps = total_elements / num_cu;
            size_t num_even_wg_per_cu = full_timesteps; // Workgroups per CU
            size_t total_even_wg = num_even_wg_per_cu * num_cu;

            if (value < total_even_wg)
            {
                size_t cu_num = value % num_cu;   // Compute Unit number (0 to num_cu - 1)
                size_t timestep = value / num_cu; // Time step
                size_t new_index = cu_num * num_even_wg_per_cu + timestep;
                return new_index;
            }
            else
            {
                // Handle any remaining values beyond total_even_wg
                return value;
            }
        }

        // Reorder such that a CU gets a contiguous set of work.
        void reorder_cu_contiguous(std::vector<size_t> &remapped_vector, size_t num_cus_per_xcd, size_t num_xcds)
        {
            size_t total_elements = remapped_vector.size();
            std::vector<size_t> reordered_vector(total_elements);
            for (size_t value : remapped_vector)
            {
                size_t new_index = reorder_cu_contiguous_hash(value, total_elements, num_cus_per_xcd, num_xcds);
                reordered_vector[new_index] = value;
            }
            remapped_vector = reordered_vector;
        }

        // A hash to reorder the WG vector such that a single XCD gets contiguous WGs over WHOLE PROBLEM (not their # CUS)
        size_t reorder_xcd_contiguous_hash(size_t value, size_t total_elements, size_t num_cus_per_xcd, size_t num_xcds)
        {
            size_t num_cu = num_cus_per_xcd * num_xcds;
            size_t full_timesteps = total_elements / num_cu;
            size_t num_even_wg_per_xcd = num_cus_per_xcd * full_timesteps; // Number of workgroups each XCD will get

            if (value < num_even_wg_per_xcd * num_xcds)
            {
                size_t xcd_num = (value / num_cus_per_xcd) % num_xcds;
                size_t timestep = value / num_cu;
                size_t offset_in_timestep = value % num_cus_per_xcd;
                size_t timestep_start_in_xcd = timestep * num_cus_per_xcd;
                size_t xcd_start = xcd_num * num_even_wg_per_xcd;
                size_t new_index = xcd_start + timestep_start_in_xcd + offset_in_timestep;

                // Ensure new_index is within bounds
                new_index = new_index % total_elements;
                return new_index;
            }
            else
            {
                return value;
            }
        }

        // Reorder such that an XCD gets a contiguous set of functions.
        void reorder_xcd_contiguous(std::vector<size_t> &remapped_vector, size_t num_cus_per_xcd, size_t num_xcds)
        {
            size_t total_elements = remapped_vector.size();
            std::vector<size_t> reordered_vector(total_elements);
            for (size_t value : remapped_vector)
            {
                size_t new_index = reorder_xcd_contiguous_hash(value, total_elements, num_cus_per_xcd, num_xcds);
                reordered_vector[new_index] = value;
            }
            remapped_vector = reordered_vector;
        }

        // Hash WGID Column-Major
        size_t hash_wgid_col(const std::vector<size_t> &dimensions, size_t input, bool debug, size_t debug_value)
        {
            // Implemented similar to hash_wgid, adapted for column-major ordering
            // For brevity, not included in full detail
            // You can implement it based on your specific requirements
            return 0; // Placeholder
        }

        // Hilbert curve mapping function
        size_t hilbert_index_to_xy(size_t n, size_t d)
        {
            size_t x = 0;
            size_t y = 0;
            size_t t = d;
            size_t s = 1;
            size_t iters = 0;
            while (s < (1 << n))
            {
                size_t rx = 1 & (t / 2);
                size_t ry = 1 & (t ^ rx);
                if (ry == 0)
                {
                    if (rx == 1)
                    {
                        x = s - 1 - x;
                        y = s - 1 - y;
                    }
                    // Swap x and y
                    size_t temp = x;
                    x = y;
                    y = temp;
                }
                x += s * rx;
                y += s * ry;
                t = t / 4;
                s *= 2;
                iters += 1;
            }
            return (y * (1 << n)) + x;
        }

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
        size_t hash_wgid(const std::vector<size_t> &dimensions, size_t input, bool debug, size_t debug_value)
        {
            size_t radix_width = dimensions.size() - 1;
            size_t grid_x = dimensions[0];
            size_t grid_y = dimensions[1];

            size_t nonquantized_x = 0;
            size_t nonquantized_y = 0;
            size_t quantized_x = 0;
            size_t quantized_y = 0;
            size_t timestep_x_dim = 1;
            size_t timestep_y_dim = 1;
            for (size_t radix = 0; radix < radix_width / 2; ++radix)
            {
                timestep_x_dim *= dimensions[radix_width - (radix * 2) - 1];
                timestep_y_dim *= dimensions[radix_width - (radix * 2)];
            }
            quantized_x = (grid_x / timestep_x_dim) * timestep_x_dim;
            quantized_y = (grid_y / timestep_y_dim) * timestep_y_dim;
            nonquantized_x = grid_x - quantized_x;
            nonquantized_y = grid_y - quantized_y;
            size_t total_quantized_size = quantized_x * quantized_y;
            size_t y_region_start = (total_quantized_size - 1) + (nonquantized_x * grid_y);

            if (input > (total_quantized_size - 1))
            {
                // Handle nonquantized regions
                size_t new_grid_x, new_grid_y, new_index;
                if (input > ((total_quantized_size - 1) + (nonquantized_x * grid_y)))
                {
                    // Unquantized region Y
                    new_grid_x = ((input - total_quantized_size - (nonquantized_x * grid_y)) / nonquantized_y) % grid_x;
                    new_grid_y = quantized_y + (input % nonquantized_y);
                    new_index = (new_grid_y * grid_x) + new_grid_x;
                    return new_index;
                }
                else
                {
                    // Unquantized region X
                    new_grid_x = quantized_x + (input % nonquantized_x);
                    new_grid_y = ((input - total_quantized_size) / nonquantized_x) % grid_y;
                    new_index = (new_grid_y * grid_x) + new_grid_x;
                    return new_index;
                }
            }

            size_t cumulative_denominator = 1;
            std::vector<std::pair<int, int>> tile_indices;
            size_t temporal_tile_dim_x = 1;
            size_t temporal_tile_dim_y = 1;

            for (size_t radix = 0; radix < radix_width / 2; ++radix)
            {
                size_t level_x_radix = dimensions[radix_width - (radix * 2) - 1];
                size_t level_y_radix = dimensions[radix_width - (radix * 2)];

                temporal_tile_dim_x *= level_x_radix;
                temporal_tile_dim_y *= level_y_radix;

                size_t level_x_idx = (input / cumulative_denominator) % level_x_radix;
                cumulative_denominator *= level_x_radix;

                size_t level_y_idx = (input / cumulative_denominator) % level_y_radix;
                cumulative_denominator *= level_y_radix;

                tile_indices.push_back(std::make_pair(level_x_idx, level_y_idx));
            }

            size_t temporal_x = (input / cumulative_denominator) % (grid_x / temporal_tile_dim_x);
            cumulative_denominator *= (grid_x / temporal_tile_dim_x);

            size_t temporal_y = (input / cumulative_denominator) % (grid_y / temporal_tile_dim_y);

            size_t new_grid_x = 0;
            size_t new_grid_y = 0;
            size_t cumulative_x = 1;
            size_t cumulative_y = 1;

            for (size_t radix = 0; radix < tile_indices.size(); ++radix)
            {
                size_t level_x_radix = dimensions[radix_width - (radix * 2) - 1];
                size_t level_y_radix = dimensions[radix_width - (radix * 2)];

                new_grid_x += tile_indices[radix].first * cumulative_x;
                cumulative_x *= level_x_radix;

                new_grid_y += tile_indices[radix].second * cumulative_y;
                cumulative_y *= level_y_radix;
            }

            new_grid_x += temporal_x * cumulative_x;
            new_grid_y += temporal_y * cumulative_y;

            size_t new_index = (new_grid_y * grid_x) + new_grid_x;
            return new_index;
        }

        // Function to calculate the product of elements in a vector
        size_t vector_product(const std::vector<size_t> &vec)
        {
            return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<size_t>());
        }

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
            size_t num_xcds)
        {
            std::vector<size_t> remapped_vector(input_vector.size(), 0);
            std::vector<size_t> temp_vector = input_vector;
            if (group_l2)
            {
                reorder_xcd_contiguous(temp_vector, num_cus_per_xcd, num_xcds);
            }
            else if (group_l1)
            {
                reorder_cu_contiguous(temp_vector, num_cus_per_xcd, num_xcds);
            }

            for (size_t input = 0; input < temp_vector.size(); ++input)
            {
                size_t new_index = 0;
                if (col_major)
                {
                    new_index = hash_wgid_col(dimensions, input, debug, debug_value);
                }
                else if (hilbert)
                {
                    assert(dimensions[0] == dimensions[1]); // Hilbert only works with square grids
                    size_t n = std::log2(dimensions[0]);
                    new_index = hilbert_index_to_xy(n, input);
                }
                else
                {
                    new_index = hash_wgid(dimensions, input, debug, debug_value);
                }
                remapped_vector[new_index] = temp_vector[input];
            }
            return remapped_vector;
        }

        // Simulate LRU cache behavior of a blocked GEMM algorithm
        double compute_reuse_in_block_gemm(
            size_t grid_m,             // Number of MT in the M dimension ceil(M/BLK_M)
            size_t grid_n,             // Number of MT in the N dimension ceil(N/BLK_N)
            size_t grid_k,             // Number of MT in the K dimension ceil(K/BLK_K)
            size_t A_size,             // Size of the A input mt (BLK_M*BLK_K)
            size_t B_size,             // Size of the B input mt (BLK_N*BLK_K)
            size_t C_size,             // Size of the C output mt (BLK_M*BLK_N)
            size_t nproc,              // Number of concurrent processors in cache scope
            size_t capacity,           // Capacity of cache in #elements
            std::vector<size_t> radix, // Z order transform of processing order. Can be as long as you want for nested Z order.
            bool print_radix,          // Prsize_t traversal order
            bool print_output,         // Prsize_t reuse factor output
            size_t num_iterations      // Early Exit option - Typically converges after 10-20 on most problems
        )
        {
            // Check Hilbert
            bool hilbert = false;
            if (radix.size() == 1 && radix[0] == -1)
            {
                radix = {2, 2};
                hilbert = true;
            }

            // Insert grid dimensions at front (like your previous approach)
            radix.insert(radix.begin(), grid_m);
            radix.insert(radix.begin(), grid_n);

            size_t max_cached_elements = capacity;
            if (print_output)
            {
                std::cout << "Tile " << grid_m << "X" << grid_n << "X" << grid_k
                          << " with A_Size:" << A_size
                          << " B_Size:" << B_size
                          << " C_Size:" << C_size << "\n";
                std::cout << "Max cached elements : " << max_cached_elements << "\n";
            }

            // Initialize LRU cache
            LRUCache cache_fifo(max_cached_elements);

            // Reuse factors
            std::vector<std::vector<size_t>> a_reuse_factors(grid_k, std::vector<size_t>(grid_m, 0));
            std::vector<std::vector<size_t>> b_reuse_factors(grid_k, std::vector<size_t>(grid_n, 0));

            // pid_vector = [0..(grid_m*grid_n - 1)]
            size_t total_pids = grid_m * grid_n;
            std::vector<size_t> pid_vector(total_pids);
            for (size_t i = 0; i < total_pids; ++i)
            {
                pid_vector[i] = i;
            }

            // Rearrange pid_vector
            pid_vector = generalized_mix_radix_with_nonquantized_region(
                radix, pid_vector, /*debug=*/false, /*debug_value=*/5,
                /*col_major=*/false, /*group_l2=*/false, /*hilbert=*/hilbert,
                /*group_l1=*/false, /*num_cus_per_xcd=*/38, /*num_xcds=*/8);

            // Memory load counters
            size_t loads = 0;
            size_t cached_loads = 0;

            // We'll keep an iteration counter. Each time we process "k" for a given "pid" => 1 iteration
            size_t iteration_count = 0;
            bool early_exit = false;

            // Outer loop
            size_t pid = 0;
            while (pid < total_pids)
            {
                // Map pid to tile index
                // (Using a linear search; for big problems, you'd want an inverse index.)
                size_t tile_id = std::distance(
                    pid_vector.begin(),
                    std::find(pid_vector.begin(), pid_vector.end(), pid));

                // Mapping of each process to the tile it processes
                std::unordered_map<int, int> proc_tiles;
                for (size_t n = 0; n < nproc; ++n)
                {
                    if ((pid + n) < total_pids)
                    {
                        proc_tiles[n] = std::distance(
                            pid_vector.begin(),
                            std::find(pid_vector.begin(), pid_vector.end(), pid + n));
                    }
                }

                // Loop over k
                for (size_t kk = 0; kk < grid_k; ++kk)
                {

                    // For each process, load A/B
                    for (size_t proc = 0; proc < nproc; ++proc)
                    {
                        if (pid + proc < total_pids)
                        {
                            size_t tile_index = proc_tiles[proc];
                            size_t output_grid_m = tile_index / grid_n;
                            size_t output_grid_n = tile_index % grid_n;

                            // A
                            LRUCache::CacheKey cache_key_A = {'A', output_grid_m, kk};
                            if (cache_fifo.contains(cache_key_A))
                            {
                                a_reuse_factors[kk][output_grid_m] += 1;
                                cached_loads += 1;
                            }
                            else
                            {
                                cache_fifo.append(cache_key_A, A_size);
                                loads += 1;
                            }

                            // B
                            LRUCache::CacheKey cache_key_B = {'B', output_grid_n, kk};
                            if (cache_fifo.contains(cache_key_B))
                            {
                                b_reuse_factors[kk][output_grid_n] += 1;
                                cached_loads += 1;
                            }
                            else
                            {
                                cache_fifo.append(cache_key_B, B_size);
                                loads += 1;
                            }
                        }
                    }

                    // Each (pid, k) => one iteration
                    iteration_count++;
                    // If we exceed num_iterations, we exit early
                    if (num_iterations > 0 && iteration_count > num_iterations)
                    {
                        early_exit = true;
                        break;
                    }

                    if (early_exit)
                    {
                        break;
                    }
                }

                // If we broke out early, we do NOT store partial results for matrix C
                if (early_exit)
                {
                    break;
                }

                // Store results to matrix C
                for (size_t proc = 0; proc < nproc; ++proc)
                {
                    if (pid + proc < total_pids)
                    {
                        size_t tile_index = proc_tiles[proc];
                        size_t proc_m = tile_index / grid_n;
                        size_t proc_n = tile_index % grid_n;
                        LRUCache::CacheKey cache_key_C = {'C', proc_m, proc_n};
                        cache_fifo.append(cache_key_C, C_size);
                    }
                }

                pid += nproc;
            }

            // Compute final percentage
            size_t total_loads = loads + cached_loads;
            if (total_loads == 0)
            {
                total_loads = 1; // avoid divide-by-zero
            }
            double percent_cached_loads = (static_cast<double>(cached_loads) / total_loads) * 100.0;

            // Prsize_t if requested
            if (print_output)
            {
                std::cout << "A reuse factors (for each k and m):\n";
                for (const auto &row : a_reuse_factors)
                {
                    for (auto val : row)
                    {
                        std::cout << val << "\t";
                    }
                    std::cout << "\n";
                }
                std::cout << "B reuse factors (for each k and n):\n";
                for (const auto &row : b_reuse_factors)
                {
                    for (auto val : row)
                    {
                        std::cout << val << "\t";
                    }
                    std::cout << "\n";
                }
                std::cout << "Total Memory Loads (missed cache): " << loads << "\n";
                std::cout << "Total Cached Loads (hit cache): " << cached_loads << "\n";
                std::cout << percent_cached_loads << "% of loads were cached\n";

                if (early_exit)
                {
                    std::cout << "NOTE: Exited early after " << (iteration_count - 1)
                              << " full iterations (due to --num_iterations limit)\n";
                }
            }

            return percent_cached_loads;
        }

    } // namespace analytical
} // namespace TensileLite
