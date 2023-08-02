/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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
#include <functional>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

template <typename InType, typename OutType, typename AlphaType, typename BetaType>
struct Runner
{
    Runner(int64_t   m,
           int64_t   n,
           int64_t   k,
           int64_t   batch_count,
           AlphaType alpha,
           BetaType  beta,
           int64_t   max_workspace_size_in_bytes)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size_in_bytes)
    {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
        CHECK_HIP_ERROR(hipMalloc(&d_a, m * k * batch_count * sizeof(InType)));
        CHECK_HIP_ERROR(hipMalloc(&d_b, n * k * batch_count * sizeof(InType)));
        CHECK_HIP_ERROR(hipMalloc(&d_c, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipMalloc(&d_d, m * n * batch_count * sizeof(OutType)));

        CHECK_HIP_ERROR(hipHostMalloc(&a, m * k * batch_count * sizeof(InType)));
        CHECK_HIP_ERROR(hipHostMalloc(&b, n * k * batch_count * sizeof(InType)));
        CHECK_HIP_ERROR(hipHostMalloc(&c, m * n * batch_count * sizeof(OutType)));
        CHECK_HIP_ERROR(hipHostMalloc(&d, m * n * batch_count * sizeof(OutType)));

        if(max_workspace_size > 0)
            CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));

        for(int i = 0; i < m * k * batch_count; i++)
            ((InType*)a)[i] = static_cast<InType>((rand() % 7) - 3);
        for(int i = 0; i < n * k * batch_count; i++)
            ((InType*)b)[i] = static_cast<InType>((rand() % 7) - 3);
        for(int i = 0; i < m * n * batch_count; i++)
            ((InType*)c)[i] = static_cast<OutType>((rand() % 7) - 3);
    }

    ~Runner()
    {
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIP_ERROR(hipFree(a));
        CHECK_HIP_ERROR(hipFree(b));
        CHECK_HIP_ERROR(hipFree(c));
        CHECK_HIP_ERROR(hipFree(d));
        CHECK_HIP_ERROR(hipFree(d_a));
        CHECK_HIP_ERROR(hipFree(d_b));
        CHECK_HIP_ERROR(hipFree(d_c));
        CHECK_HIP_ERROR(hipFree(d_d));
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    void hostToDevice()
    {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_a, a, m * k * batch_count * sizeof(InType), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_b, b, n * k * batch_count * sizeof(InType), hipMemcpyHostToDevice, stream));
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d_c, c, m * n * batch_count * sizeof(OutType), hipMemcpyHostToDevice, stream));
    }

    void deviceToHost()
    {
        CHECK_HIP_ERROR(hipMemcpyAsync(
            d, d_d, m * n * batch_count * sizeof(OutType), hipMemcpyDeviceToHost, stream));
    }

    void run(const std::function<void()>& func)
    {
        hostToDevice();

        static_cast<void>(func());

        deviceToHost();
        hipStreamSynchronize(stream);
    }

    int64_t   m;
    int64_t   n;
    int64_t   k;
    int64_t   batch_count;
    AlphaType alpha;
    BetaType  beta;

    void *a, *b, *c, *d; // host
    void *d_a, *d_b, *d_c, *d_d; // device

    void*   d_workspace;
    int64_t max_workspace_size;

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};

template <typename InType, typename OutType, typename AlphaType, typename BetaType>
struct RunnerVec
{
    RunnerVec(const std::vector<int64_t>   m,
              const std::vector<int64_t>   n,
              const std::vector<int64_t>   k,
              const std::vector<int64_t>   batch_count,
              const std::vector<AlphaType> alpha,
              const std::vector<BetaType>  beta,
              const int64_t                max_workspace_size_in_bytes)
        : m(m)
        , n(n)
        , k(k)
        , batch_count(batch_count)
        , alpha(alpha)
        , beta(beta)
        , max_workspace_size(max_workspace_size_in_bytes)
    {
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
        d_a.resize(m.size(), nullptr);
        d_b.resize(m.size(), nullptr);
        d_c.resize(m.size(), nullptr);
        d_d.resize(m.size(), nullptr);
        a.resize(m.size(), nullptr);
        b.resize(m.size(), nullptr);
        c.resize(m.size(), nullptr);
        d.resize(m.size(), nullptr);
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipMalloc(&d_a[j], m[j] * k[j] * batch_count[j] * sizeof(InType)));
            CHECK_HIP_ERROR(hipMalloc(&d_b[j], n[j] * k[j] * batch_count[j] * sizeof(InType)));
            CHECK_HIP_ERROR(hipMalloc(&d_c[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));
            CHECK_HIP_ERROR(hipMalloc(&d_d[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));

            CHECK_HIP_ERROR(hipHostMalloc(&a[j], m[j] * k[j] * batch_count[j] * sizeof(InType)));
            CHECK_HIP_ERROR(hipHostMalloc(&b[j], n[j] * k[j] * batch_count[j] * sizeof(InType)));
            CHECK_HIP_ERROR(hipHostMalloc(&c[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));
            CHECK_HIP_ERROR(hipHostMalloc(&d[j], m[j] * n[j] * batch_count[j] * sizeof(OutType)));

            for(int i = 0; i < m[j] * k[j] * batch_count[j]; i++)
                ((InType*)a[j])[i] = static_cast<InType>((rand() % 7) - 3);
            for(int i = 0; i < n[j] * k[j] * batch_count[j]; i++)
                ((InType*)b[j])[i] = static_cast<InType>((rand() % 7) - 3);
            for(int i = 0; i < m[j] * n[j] * batch_count[j]; i++)
                ((InType*)c[j])[i] = static_cast<OutType>((rand() % 7) - 3);
        }
        if(max_workspace_size > 0)
            CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
    }

    ~RunnerVec()
    {
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipFree(a[j]));
            CHECK_HIP_ERROR(hipFree(b[j]));
            CHECK_HIP_ERROR(hipFree(c[j]));
            CHECK_HIP_ERROR(hipFree(d[j]));
            CHECK_HIP_ERROR(hipFree(d_a[j]));
            CHECK_HIP_ERROR(hipFree(d_b[j]));
            CHECK_HIP_ERROR(hipFree(d_c[j]));
            CHECK_HIP_ERROR(hipFree(d_d[j]));
        }
        CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    void hostToDevice()
    {
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipMemcpyAsync(d_a[j],
                                           a[j],
                                           m[j] * k[j] * batch_count[j] * sizeof(InType),
                                           hipMemcpyHostToDevice,
                                           stream));
            CHECK_HIP_ERROR(hipMemcpyAsync(d_b[j],
                                           b[j],
                                           n[j] * k[j] * batch_count[j] * sizeof(InType),
                                           hipMemcpyHostToDevice,
                                           stream));
            CHECK_HIP_ERROR(hipMemcpyAsync(d_c[j],
                                           c[j],
                                           m[j] * n[j] * batch_count[j] * sizeof(OutType),
                                           hipMemcpyHostToDevice,
                                           stream));
        }
    }

    void deviceToHost()
    {
        for(int j = 0; j < m.size(); j++)
        {
            CHECK_HIP_ERROR(hipMemcpyAsync(d[j],
                                           d_d[j],
                                           m[j] * n[j] * batch_count[j] * sizeof(OutType),
                                           hipMemcpyDeviceToHost,
                                           stream));
        }
    }

    void run(const std::function<void()>& func)
    {
        hostToDevice();

        static_cast<void>(func());

        deviceToHost();
        hipStreamSynchronize(stream);
    }

    std::vector<int64_t>   m;
    std::vector<int64_t>   n;
    std::vector<int64_t>   k;
    std::vector<int64_t>   batch_count;
    std::vector<AlphaType> alpha;
    std::vector<BetaType>  beta;

    std::vector<void*> a, b, c, d; // host
    std::vector<void*> d_a, d_b, d_c, d_d; // device

    void*   d_workspace;
    int64_t max_workspace_size;

    hipStream_t       stream;
    hipblasLtHandle_t handle;
};
