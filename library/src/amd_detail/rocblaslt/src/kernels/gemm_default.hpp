/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <map>

#include "hipblaslt/hipblaslt-types.h"
#include "hipblaslt/hipblaslt_xfloat32.h"

template <typename TiA,
          typename TiB,
          typename To,
          typename Tc,
          bool     isColMajA,
          bool     isColMajB,
          uint32_t num_thread_x,
          uint32_t num_thread_y = num_thread_x>
__global__ void gemm_default_l1(To* __restrict__ D,
                                const TiA* __restrict__ A,
                                const TiB* __restrict__ B,
                                const To* __restrict__ C,
                                const Tc alpha,
                                const Tc beta,
                                uint32_t num_batch,
                                uint32_t I,
                                uint32_t J,
                                uint32_t K,
                                uint32_t lda,
                                uint32_t ldb,
                                uint32_t ldc,
                                uint32_t ldd,
                                uint32_t stride_a,
                                uint32_t stride_b,
                                uint32_t stride_c,
                                uint32_t stride_d)
{
    // problem: GEMM NN
    // size A: I x K
    // lds A: I
    // size B: K x J
    // lds B: K
    // thread_per_block_x = 32
    // thread_per_block_y = 32
    // no remainder, all elements could be covered by all blocks
    assert(num_thread_x == num_thread_y);
    uint32_t num_iter = K / num_thread_x;
    uint32_t remain   = K % num_thread_x;

    // allocate local memory
    __shared__ TiA A_local[num_thread_x * num_thread_y];
    __shared__ TiB B_local[num_thread_x * num_thread_y];

    // global read address
    uint32_t global_read_A_x;
    uint32_t global_read_A_y;
    uint32_t global_read_B_x;
    uint32_t global_read_B_y;
    if constexpr(isColMajA)
    {
        global_read_A_x = hipBlockIdx_x * num_thread_x + hipThreadIdx_x;
        global_read_A_y = hipThreadIdx_y;
    }
    else
    {
        global_read_A_x = hipThreadIdx_y;
        global_read_A_y = hipBlockIdx_x * num_thread_x + hipThreadIdx_x;
    }
    if constexpr(isColMajB)
    {
        global_read_B_x = hipThreadIdx_x;
        global_read_B_y = hipBlockIdx_y * num_thread_y + hipThreadIdx_y;
    }
    else
    {
        global_read_B_x = hipBlockIdx_y * num_thread_y + hipThreadIdx_y;
        global_read_B_y = hipThreadIdx_x;
    }
    uint32_t   global_read_C_x = hipBlockIdx_x * num_thread_x + hipThreadIdx_x;
    uint32_t   global_read_C_y = hipBlockIdx_y * num_thread_y + hipThreadIdx_y;
    const TiA* global_read_A   = A + global_read_A_x + global_read_A_y * lda;
    const TiB* global_read_B   = B + global_read_B_x + global_read_B_y * ldb;
    const To*  global_read_C   = C + global_read_C_x + global_read_C_y * ldc;

    // local write address
    uint32_t local_write_A_x = hipThreadIdx_x;
    uint32_t local_write_A_y = hipThreadIdx_y;
    uint32_t local_write_B_x = hipThreadIdx_x;
    uint32_t local_write_B_y = hipThreadIdx_y;
    TiA*     local_write_A   = A_local + local_write_A_x + local_write_A_y * num_thread_x;
    TiB*     local_write_B   = B_local + local_write_B_x + local_write_B_y * num_thread_x;

    // local read address
    uint32_t local_read_A_x = hipThreadIdx_x;
    uint32_t local_read_A_y = 0;
    uint32_t local_read_B_x = 0;
    uint32_t local_read_B_y = hipThreadIdx_y;
    TiA*     local_read_A   = A_local + local_read_A_x + local_read_A_y * num_thread_x;
    TiB*     local_read_B   = B_local + local_read_B_x + local_read_B_y * num_thread_x;

    // global write address
    uint32_t global_write_D_x = hipBlockIdx_x * num_thread_x + hipThreadIdx_x;
    uint32_t global_write_D_y = hipBlockIdx_y * num_thread_y + hipThreadIdx_y;
    To*      global_write_D   = D + global_write_D_x + global_write_D_y * ldd;

    // main
    for(int z = 0; z < num_batch; z++)
    { // for batch dimension
        TiA        a;
        TiB        b;
        Tc         acc               = 0;
        const TiA* global_read_A_bak = global_read_A;
        const TiB* global_read_B_bak = global_read_B;
        for(int i = 0; i < num_iter; i++)
        {
            // global read
            if(hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x < I) // for remainder of I
                a = *global_read_A_bak;
            if(hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y < J) // for remainder of J
                b = *global_read_B_bak;
            __syncthreads();

            // global read address increment
            if constexpr(isColMajA)
            {
                global_read_A_bak += num_thread_x * lda;
            }
            else
            {
                global_read_A_bak += num_thread_x;
            }
            if constexpr(isColMajB)
            {
                global_read_B_bak += num_thread_x;
            }
            else
            {
                global_read_B_bak += num_thread_x * ldb;
            }

            // local write
            *local_write_A = a;
            *local_write_B = b;
            __syncthreads();

            TiA* local_read_A_tmp = local_read_A;
            TiB* local_read_B_tmp = local_read_B;

            for(int j = 0; j < num_thread_x; j++)
            {
                // local read
                a = *local_read_A_tmp;
                b = *local_read_B_tmp;

                // local read address increment
                local_read_A_tmp += num_thread_x;
                local_read_B_tmp += 1;

                // matmul
                acc += static_cast<Tc>(a) * static_cast<Tc>(b) * alpha;
            }
        }

        // remainder loop
        if(remain)
        { // for remainder of K
            // global read
            if(hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x < I && hipThreadIdx_y < remain)
                a = *global_read_A_bak;
            if(hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y < J && hipThreadIdx_x < remain)
                b = *global_read_B_bak;
            __syncthreads();

            // local write
            *local_write_A = a;
            *local_write_B = b;
            __syncthreads();

            TiA* local_read_A_tmp = local_read_A;
            TiB* local_read_B_tmp = local_read_B;
            // #pragma unroll
            for(int j = 0; j < remain; j++)
            {
                // local read
                a = *local_read_A_tmp;
                b = *local_read_B_tmp;

                // local read address increment
                local_read_A_tmp += num_thread_x;
                local_read_B_tmp += 1;

                // matmul
                acc += static_cast<Tc>(a) * static_cast<Tc>(b) * alpha;
            }
        }

        // global write
        if(hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x < I
           && hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y < J)
        {
            To c = *global_read_C;
            acc += static_cast<Tc>(c) * beta;
            *global_write_D = static_cast<To>(acc);
        }
        __syncthreads();

        // next batch increment
        global_read_A += stride_a;
        global_read_B += stride_b;
        global_read_C += stride_c;
        global_write_D += stride_d;
    }
}

/* gemm default kernel starts */
template <typename TiA, typename TiB, typename To, typename Tc, bool isColMajA, bool isColMajB>
hipError_t launch_gemm_default_kernel(To*         D,
                                      const TiA*  A,
                                      const TiB*  B,
                                      const To*   C,
                                      const Tc    alpha,
                                      const Tc    beta,
                                      uint32_t    num_batch,
                                      uint32_t    m,
                                      uint32_t    n,
                                      uint32_t    k,
                                      uint32_t    lda,
                                      uint32_t    ldb,
                                      uint32_t    ldc,
                                      uint32_t    ldd,
                                      uint32_t    stride_a,
                                      uint32_t    stride_b,
                                      uint32_t    stride_c,
                                      uint32_t    stride_d,
                                      hipStream_t stream)
{
    // hardcode num_thread_x = num_thread_y = 32
    constexpr uint32_t tx = 32;
    constexpr uint32_t ty = 32;
    uint32_t           gx = (m + tx - 1) / tx;
    uint32_t           gy = (n + ty - 1) / ty;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemm_default_l1<TiA, TiB, To, Tc, isColMajA, isColMajB, tx, ty>),
        dim3(gx, gy),
        dim3(tx, ty),
        0,
        stream,
        D,
        A,
        B,
        C,
        alpha,
        beta,
        num_batch,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        ldd,
        stride_a,
        stride_b,
        stride_c,
        stride_d);
    return hipSuccess;
}

#define GEN_COMBINATION(TiA_key, TiB_key, To_key, Tc_key, TiA_val, TiB_val, To_val, Tc_val)     \
    {std::make_tuple(TiA_key, TiB_key, To_key, To_key, Tc_key, true, true),                     \
     [](void*       D,                                                                          \
        const void* A,                                                                          \
        const void* B,                                                                          \
        const void* C,                                                                          \
        const void* alpha,                                                                      \
        const void* beta,                                                                       \
        uint32_t    num_batch,                                                                  \
        uint32_t    m,                                                                          \
        uint32_t    n,                                                                          \
        uint32_t    k,                                                                          \
        uint32_t    lda,                                                                        \
        uint32_t    ldb,                                                                        \
        uint32_t    ldc,                                                                        \
        uint32_t    ldd,                                                                        \
        uint32_t    stride_a,                                                                   \
        uint32_t    stride_b,                                                                   \
        uint32_t    stride_c,                                                                   \
        uint32_t    stride_d,                                                                   \
        hipStream_t stream) {                                                                   \
         return launch_gemm_default_kernel<TiA_val, TiB_val, To_val, Tc_val, true, true>(       \
             static_cast<To_val*>(D),                                                           \
             static_cast<const TiA_val*>(A),                                                    \
             static_cast<const TiB_val*>(B),                                                    \
             static_cast<const To_val*>(C),                                                     \
             *reinterpret_cast<const Tc_val*>(alpha),                                           \
             *reinterpret_cast<const Tc_val*>(beta),                                            \
             num_batch,                                                                         \
             m,                                                                                 \
             n,                                                                                 \
             k,                                                                                 \
             lda,                                                                               \
             ldb,                                                                               \
             ldc,                                                                               \
             ldd,                                                                               \
             stride_a,                                                                          \
             stride_b,                                                                          \
             stride_c,                                                                          \
             stride_d,                                                                          \
             stream);                                                                           \
     }},                                                                                        \
        {std::make_tuple(TiA_key, TiB_key, To_key, To_key, Tc_key, true, false),                \
         [](void*       D,                                                                      \
            const void* A,                                                                      \
            const void* B,                                                                      \
            const void* C,                                                                      \
            const void* alpha,                                                                  \
            const void* beta,                                                                   \
            uint32_t    num_batch,                                                              \
            uint32_t    m,                                                                      \
            uint32_t    n,                                                                      \
            uint32_t    k,                                                                      \
            uint32_t    lda,                                                                    \
            uint32_t    ldb,                                                                    \
            uint32_t    ldc,                                                                    \
            uint32_t    ldd,                                                                    \
            uint32_t    stride_a,                                                               \
            uint32_t    stride_b,                                                               \
            uint32_t    stride_c,                                                               \
            uint32_t    stride_d,                                                               \
            hipStream_t stream) {                                                               \
             return launch_gemm_default_kernel<TiA_val, TiB_val, To_val, Tc_val, true, false>(  \
                 static_cast<To_val*>(D),                                                       \
                 static_cast<const TiA_val*>(A),                                                \
                 static_cast<const TiB_val*>(B),                                                \
                 static_cast<const To_val*>(C),                                                 \
                 *reinterpret_cast<const Tc_val*>(alpha),                                       \
                 *reinterpret_cast<const Tc_val*>(beta),                                        \
                 num_batch,                                                                     \
                 m,                                                                             \
                 n,                                                                             \
                 k,                                                                             \
                 lda,                                                                           \
                 ldb,                                                                           \
                 ldc,                                                                           \
                 ldd,                                                                           \
                 stride_a,                                                                      \
                 stride_b,                                                                      \
                 stride_c,                                                                      \
                 stride_d,                                                                      \
                 stream);                                                                       \
         }},                                                                                    \
        {std::make_tuple(TiA_key, TiB_key, To_key, To_key, Tc_key, false, true),                \
         [](void*       D,                                                                      \
            const void* A,                                                                      \
            const void* B,                                                                      \
            const void* C,                                                                      \
            const void* alpha,                                                                  \
            const void* beta,                                                                   \
            uint32_t    num_batch,                                                              \
            uint32_t    m,                                                                      \
            uint32_t    n,                                                                      \
            uint32_t    k,                                                                      \
            uint32_t    lda,                                                                    \
            uint32_t    ldb,                                                                    \
            uint32_t    ldc,                                                                    \
            uint32_t    ldd,                                                                    \
            uint32_t    stride_a,                                                               \
            uint32_t    stride_b,                                                               \
            uint32_t    stride_c,                                                               \
            uint32_t    stride_d,                                                               \
            hipStream_t stream) {                                                               \
             return launch_gemm_default_kernel<TiA_val, TiB_val, To_val, Tc_val, false, true>(  \
                 static_cast<To_val*>(D),                                                       \
                 static_cast<const TiA_val*>(A),                                                \
                 static_cast<const TiB_val*>(B),                                                \
                 static_cast<const To_val*>(C),                                                 \
                 *reinterpret_cast<const Tc_val*>(alpha),                                       \
                 *reinterpret_cast<const Tc_val*>(beta),                                        \
                 num_batch,                                                                     \
                 m,                                                                             \
                 n,                                                                             \
                 k,                                                                             \
                 lda,                                                                           \
                 ldb,                                                                           \
                 ldc,                                                                           \
                 ldd,                                                                           \
                 stride_a,                                                                      \
                 stride_b,                                                                      \
                 stride_c,                                                                      \
                 stride_d,                                                                      \
                 stream);                                                                       \
         }},                                                                                    \
        {std::make_tuple(TiA_key, TiB_key, To_key, To_key, Tc_key, false, false),               \
         [](void*       D,                                                                      \
            const void* A,                                                                      \
            const void* B,                                                                      \
            const void* C,                                                                      \
            const void* alpha,                                                                  \
            const void* beta,                                                                   \
            uint32_t    num_batch,                                                              \
            uint32_t    m,                                                                      \
            uint32_t    n,                                                                      \
            uint32_t    k,                                                                      \
            uint32_t    lda,                                                                    \
            uint32_t    ldb,                                                                    \
            uint32_t    ldc,                                                                    \
            uint32_t    ldd,                                                                    \
            uint32_t    stride_a,                                                               \
            uint32_t    stride_b,                                                               \
            uint32_t    stride_c,                                                               \
            uint32_t    stride_d,                                                               \
            hipStream_t stream) {                                                               \
             return launch_gemm_default_kernel<TiA_val, TiB_val, To_val, Tc_val, false, false>( \
                 static_cast<To_val*>(D),                                                       \
                 static_cast<const TiA_val*>(A),                                                \
                 static_cast<const TiB_val*>(B),                                                \
                 static_cast<const To_val*>(C),                                                 \
                 *reinterpret_cast<const Tc_val*>(alpha),                                       \
                 *reinterpret_cast<const Tc_val*>(beta),                                        \
                 num_batch,                                                                     \
                 m,                                                                             \
                 n,                                                                             \
                 k,                                                                             \
                 lda,                                                                           \
                 ldb,                                                                           \
                 ldc,                                                                           \
                 ldd,                                                                           \
                 stride_a,                                                                      \
                 stride_b,                                                                      \
                 stride_c,                                                                      \
                 stride_d,                                                                      \
                 stream);                                                                       \
         }},

using gemm_default_kernels_map_Key = std::
    tuple<hipDataType, hipDataType, hipDataType, hipDataType, rocblaslt_compute_type, bool, bool>;
using gemm_default_kernels_map_val = std::function<void(void*,
                                                        const void*,
                                                        const void*,
                                                        const void*,
                                                        const void*,
                                                        const void*,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        hipStream_t)>;
std::map<gemm_default_kernels_map_Key, gemm_default_kernels_map_val> gemm_default_kernels_map{
    // Real precisions
    GEN_COMBINATION(HIP_R_16F,
                    HIP_R_16F,
                    HIP_R_16F,
                    rocblaslt_compute_f32,
                    hipblasLtHalf,
                    hipblasLtHalf,
                    hipblasLtHalf,
                    hipblasLtFloat) GEN_COMBINATION(HIP_R_16BF,
                                                    HIP_R_16BF,
                                                    HIP_R_16BF,
                                                    rocblaslt_compute_f32,
                                                    hipblasLtBfloat16,
                                                    hipblasLtBfloat16,
                                                    hipblasLtBfloat16,
                                                    hipblasLtFloat)
        GEN_COMBINATION(HIP_R_32F,
                        HIP_R_32F,
                        HIP_R_32F,
                        rocblaslt_compute_f32,
                        hipblasLtFloat,
                        hipblasLtFloat,
                        hipblasLtFloat,
                        hipblasLtFloat)
    // Real precisions 1 bytes
    GEN_COMBINATION(HIP_R_8F_E4M3_FNUZ,
                    HIP_R_8F_E4M3_FNUZ,
                    HIP_R_32F,
                    rocblaslt_compute_f32,
                    hipblaslt_f8_fnuz,
                    hipblaslt_f8_fnuz,
                    hipblasLtFloat,
                    hipblasLtFloat) GEN_COMBINATION(HIP_R_8F_E5M2_FNUZ,
                                                    HIP_R_8F_E4M3_FNUZ,
                                                    HIP_R_32F,
                                                    rocblaslt_compute_f32,
                                                    hipblaslt_bf8_fnuz,
                                                    hipblaslt_f8_fnuz,
                                                    hipblasLtFloat,
                                                    hipblasLtFloat)
        GEN_COMBINATION(HIP_R_8F_E4M3_FNUZ,
                        HIP_R_8F_E5M2_FNUZ,
                        HIP_R_32F,
                        rocblaslt_compute_f32,
                        hipblaslt_f8_fnuz,
                        hipblaslt_bf8_fnuz,
                        hipblasLtFloat,
                        hipblasLtFloat) GEN_COMBINATION(HIP_R_8F_E4M3_FNUZ,
                                                        HIP_R_8F_E4M3_FNUZ,
                                                        HIP_R_32F,
                                                        rocblaslt_compute_f32,
                                                        hipblaslt_f8_fnuz,
                                                        hipblaslt_f8_fnuz,
                                                        hipblasLtHalf,
                                                        hipblasLtFloat)
            GEN_COMBINATION(HIP_R_8F_E5M2_FNUZ,
                            HIP_R_8F_E4M3_FNUZ,
                            HIP_R_32F,
                            rocblaslt_compute_f32,
                            hipblaslt_bf8_fnuz,
                            hipblaslt_f8_fnuz,
                            hipblasLtHalf,
                            hipblasLtFloat) GEN_COMBINATION(HIP_R_8F_E4M3_FNUZ,
                                                            HIP_R_8F_E5M2_FNUZ,
                                                            HIP_R_32F,
                                                            rocblaslt_compute_f32,
                                                            hipblaslt_f8_fnuz,
                                                            hipblaslt_bf8_fnuz,
                                                            hipblasLtHalf,
                                                            hipblasLtFloat)
    // Real precisions xf32
    // GEN_COMBINATION(HIP_R_32F, HIP_R_32F, HIP_R_32F, rocblaslt_compute_f32_fast_xf32,
    //                 hipblasLtFloat, hipblasLtFloat, hipblasLtFloat, hipblasLtXfloat32)
    // Real precisions i8
    GEN_COMBINATION(HIP_R_8I,
                    HIP_R_8I,
                    HIP_R_8I,
                    rocblaslt_compute_i32,
                    hipblasLtInt8,
                    hipblasLtInt8,
                    hipblasLtInt8,
                    hipblasLtInt32) GEN_COMBINATION(HIP_R_8I,
                                                    HIP_R_8I,
                                                    HIP_R_32I,
                                                    rocblaslt_compute_i32,
                                                    hipblasLtInt8,
                                                    hipblasLtInt8,
                                                    hipblasLtInt32,
                                                    hipblasLtInt32)
    // Real precisions dstf32
    GEN_COMBINATION(HIP_R_16F,
                    HIP_R_16F,
                    HIP_R_32F,
                    rocblaslt_compute_f32,
                    hipblasLtHalf,
                    hipblasLtHalf,
                    hipblasLtFloat,
                    hipblasLtFloat)
    // Real precisions gemm only
    GEN_COMBINATION(HIP_R_64F,
                    HIP_R_64F,
                    HIP_R_64F,
                    rocblaslt_compute_f64,
                    hipblasLtDouble,
                    hipblasLtDouble,
                    hipblasLtDouble,
                    hipblasLtDouble)
    // Real mix precisions
    GEN_COMBINATION(HIP_R_8F_E4M3_FNUZ,
                    HIP_R_16F,
                    HIP_R_16F,
                    rocblaslt_compute_f32_fast_f16,
                    hipblaslt_f8_fnuz,
                    hipblasLtHalf,
                    hipblasLtHalf,
                    hipblasLtHalf) GEN_COMBINATION(HIP_R_16F,
                                                   HIP_R_8F_E4M3_FNUZ,
                                                   HIP_R_16F,
                                                   rocblaslt_compute_f32_fast_f16,
                                                   hipblasLtHalf,
                                                   hipblaslt_f8_fnuz,
                                                   hipblasLtHalf,
                                                   hipblasLtHalf)
        GEN_COMBINATION(HIP_R_8F_E4M3_FNUZ,
                        HIP_R_16F,
                        HIP_R_32F,
                        rocblaslt_compute_f32_fast_f16,
                        hipblaslt_f8_fnuz,
                        hipblasLtHalf,
                        hipblasLtFloat,
                        hipblasLtFloat) GEN_COMBINATION(HIP_R_16F,
                                                        HIP_R_8F_E4M3_FNUZ,
                                                        HIP_R_32F,
                                                        rocblaslt_compute_f32_fast_f16,
                                                        hipblasLtHalf,
                                                        hipblaslt_f8_fnuz,
                                                        hipblasLtFloat,
                                                        hipblasLtFloat)
    // Real mix precisions fp8
    GEN_COMBINATION(HIP_R_8F_E4M3_FNUZ,
                    HIP_R_16F,
                    HIP_R_8F_E4M3_FNUZ,
                    rocblaslt_compute_f32_fast_f16,
                    hipblaslt_f8_fnuz,
                    hipblasLtHalf,
                    hipblaslt_f8_fnuz,
                    hipblasLtHalf) GEN_COMBINATION(HIP_R_16F,
                                                   HIP_R_8F_E4M3_FNUZ,
                                                   HIP_R_8F_E4M3_FNUZ,
                                                   rocblaslt_compute_f32_fast_f16,
                                                   hipblasLtHalf,
                                                   hipblaslt_f8_fnuz,
                                                   hipblaslt_f8_fnuz,
                                                   hipblasLtHalf)};
/* gemm default kernel ends */
