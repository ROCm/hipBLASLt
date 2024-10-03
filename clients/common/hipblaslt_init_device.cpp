/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_random.hpp"
#include <hipblaslt/hipblaslt.h>

template <typename T, typename F>
__global__ void fill_kernel(T* A, size_t size, F f)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        A[idx] = f(idx);
}

template <typename T, typename F>
void fill_batch(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, const F& f)
{
    size_t size       = std::max(lda * N, stride) * batch_count;
    size_t block_size = 256;
    size_t grid_size  = (size + block_size - 1) / block_size;
    fill_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(A, size, f);
}

__device__ uint32_t pseudo_random_device(size_t idx)
{
    // Numerical Recipes ranqd1, Chapter 7.1, ?An Even Quicker Generator, Eq. 7.1.6. parameters from Knuth and H. W. Lewis
    auto s = idx * 1664525 + 1013904223;
    // Run a few extra iterations to make the generators diverge
    // in case the seeds are still poor (consecutive ints)
    for(int i = 0; i < 3; i++)
    {
        // Marsaglia, G. (2003). "Xorshift RNGs". Journal of Statistical Software. 8 (14). doi:10.18637/jss.v008.i14
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
    }
    return s;
}

template <typename T>
void hipblaslt_init_device(ABC                      abc,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           T*                       A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           size_t                   stride,
                           size_t                   batch_count)
{
    if(is_nan)
    {
        std::array<T, 100> rand_nans;
        for(auto& r : rand_nans)
            r = T(hipblaslt_nan_rng());
        fill_batch(A, M, N, lda, stride, batch_count, [rand_nans](size_t idx) -> T {
            return rand_nans[pseudo_random_device(idx) % rand_nans.size()];
        });
    }
    else
    {
        switch(init)
        {
        case hipblaslt_initialization::rand_int:
            if(abc == ABC::A || abc == ABC::C)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(pseudo_random_device(idx) % 10 + 1.f);
                });
            else if(abc == ABC::B)
            {
                stride = std::max(lda * N, stride);
                fill_batch(A, M, N, lda, stride, batch_count, [stride, lda](size_t idx) -> T {
                    auto b     = idx / stride;
                    auto j     = (idx - b * stride) / lda;
                    auto i     = (idx - b * stride) - j * lda;
                    auto value = T(pseudo_random_device(idx) % 10 + 1.f);
                    return (i ^ j) & 1 ? value : negate(value);
                });
            }
            break;
        case hipblaslt_initialization::trig_float:
            if(abc == ABC::A || abc == ABC::C)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(sin(double(idx)));
                });
            else if(abc == ABC::B)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(cos(double(idx)));
                });
            break;
        case hipblaslt_initialization::hpl:
            fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                auto r = pseudo_random_device(idx);
                return T(double(r) / double(std::numeric_limits<decltype(r)>::max()) - 0.5);
            });
            break;
        case hipblaslt_initialization::special:
            if(abc == ABC::A)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(hipblasLtHalf(65280.0));
                });
            else if(abc == ABC::B)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(hipblasLtHalf(0.0000607967376708984375));
                });
            else if(abc == ABC::C)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(pseudo_random_device(idx) % 10 + 1.f);
                });
            break;
        case hipblaslt_initialization::zero:
            fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T { return T(0); });
            break;
        default:
            hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
            break;
        }
    }
}

void hipblaslt_init_device(ABC                      abc,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           void*                    A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           hipDataType              type,
                           size_t                   stride,
                           size_t                   batch_count)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_device<float>(
            abc, init, is_nan, static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_device<double>(
            abc, init, is_nan, static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_device<hipblasLtHalf>(
            abc, init, is_nan, static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_device<hip_bfloat16>(
            abc, init, is_nan, static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_device<hipblaslt_f8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_device<hipblaslt_bf8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        hipblaslt_init_device<hipblaslt_f8>(
            abc, init, is_nan, static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_device<hipblaslt_bf8>(
            abc, init, is_nan, static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_device<int32_t>(
            abc, init, is_nan, static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_device<hipblasLtInt8>(
            abc, init, is_nan, static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
        break;
    }
}
