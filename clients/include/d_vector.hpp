/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#include "datatype_interface.hpp"
#include "hipblaslt_arguments.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_test.hpp"
#include "singletons.hpp"
#include <cinttypes>
#include <hipblaslt/hipblaslt.h>

#define MEM_MAX_GUARD_PAD 8192
#define MAX_DTYPE_SIZE sizeof(double)

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T>
class d_vector
{
private:
    size_t m_size;
    size_t m_pad, m_guard_len;
    size_t m_bytes;

    static bool m_init_guard;

protected:
    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

public:
    bool use_HMM = false;

public:
    static T m_guard[MEM_MAX_GUARD_PAD];

#ifdef GOOGLE_TEST
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(T))
        , m_bytes((s + m_pad * 2) * sizeof(T))
        , use_HMM(HMM)
    {
        // Initialize m_guard with random data
        if(!m_init_guard)
        {
            hipblaslt_init_nan(m_guard, MEM_MAX_GUARD_PAD);
            m_init_guard = true;
        }
    }
#else
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(0) // save current pad length
        , m_guard_len(0 * sizeof(T))
        , m_bytes(s ? s * sizeof(T) : sizeof(T))
        , use_HMM(HMM)
    {
    }
#endif

    T* device_vector_setup()
    {
        T* d = nullptr;
        if(use_HMM ? hipMallocManaged(&d, m_bytes) : (hipMalloc)(&d, m_bytes) != hipSuccess)
        {
            hipblaslt_cerr << "Error allocating " << m_bytes << " m_bytes (" << (m_bytes >> 30)
                           << " GB)" << std::endl;

            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            if(m_guard_len > 0)
            {
                // Copy m_guard to device memory before allocated memory
                EXPECT_EQ(hipMemcpy(d, m_guard, m_guard_len, hipMemcpyHostToDevice), hipSuccess);

                // Point to allocated block
                d += m_pad;

                // Copy m_guard to device memory after allocated memory
                EXPECT_EQ(hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyHostToDevice),
                          hipSuccess);
            }
        }
#endif
        return d;
    }

    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        if(m_guard_len > 0)
        {
            T* host = new T[m_pad];

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host, d + this->m_size, m_guard_len, hipMemcpyDeviceToHost),
                      hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

            // Point to m_guard before allocated memory
            d -= m_pad;

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

            delete[] host;
        }
#endif
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(m_pad > 0)
            {
                T* host = new T[m_pad];

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host, d + this->m_size, m_guard_len, hipMemcpyDeviceToHost),
                          hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

                // Point to m_guard before allocated memory
                d -= m_pad;

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

                delete[] host;
            }
#endif
            // Free device memory
            if((hipFree)(d) != hipSuccess)
            {
                hipblaslt_cerr << "free device memory failed" << std::endl;
            }
        }
    }
};

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
class d_vector_type
{
private:
    size_t      m_size;
    hipDataType m_dtype;
    size_t      m_pad, m_guard_len;
    size_t      m_bytes;

    inline static bool m_init_guard_type;

protected:
    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

public:
    bool use_HMM = false;

public:
    inline static char m_guard_type[MEM_MAX_GUARD_PAD * MAX_DTYPE_SIZE];

#ifdef GOOGLE_TEST
    d_vector_type(hipDataType dtype, size_t s, bool HMM = false)
        : m_size(s)
        , m_dtype(dtype)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * realDataTypeSize(dtype))
        , m_bytes((s + m_pad * 2) * realDataTypeSize(dtype))
        , use_HMM(HMM)
    {
        // Initialize m_guard with random data
        if(!m_init_guard_type)
        {
            hipblaslt_init_nan(m_guard_type, MEM_MAX_GUARD_PAD);
            m_init_guard_type = true;
        }
    }
#else
    d_vector_type(hipDataType dtype, size_t s, bool HMM = false)
        : m_size(s)
        , m_dtype(dtype)
        , m_pad(0) // save current pad length
        , m_guard_len(0 * realDataTypeSize(dtype))
        , m_bytes(s ? s * realDataTypeSize(dtype) : realDataTypeSize(dtype))
        , use_HMM(HMM)
    {
    }
#endif

    char* device_vector_setup()
    {
        char* d = nullptr;
        if(use_HMM ? hipMallocManaged(&d, m_bytes) : (hipMalloc)(&d, m_bytes) != hipSuccess)
        {
            hipblaslt_cerr << "Error allocating " << m_bytes << " m_bytes (" << (m_bytes >> 30)
                           << " GB)" << std::endl;

            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            if(m_guard_len > 0)
            {
                // Copy m_guard to device memory before allocated memory
                EXPECT_EQ(hipMemcpy(d, m_guard_type, m_guard_len, hipMemcpyHostToDevice),
                          hipSuccess);

                // Point to allocated block
                d += m_pad * realDataTypeSize(m_dtype);

                // Copy m_guard to device memory after allocated memory
                EXPECT_EQ(hipMemcpy(d + this->m_size * realDataTypeSize(m_dtype),
                                    m_guard_type,
                                    m_guard_len,
                                    hipMemcpyHostToDevice),
                          hipSuccess);
            }
        }
#endif
        return d;
    }

    void device_vector_check(char* d)
    {
#ifdef GOOGLE_TEST
        if(m_guard_len > 0)
        {
            char* host = new char[m_pad * realDataTypeSize(m_dtype)];

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host,
                                d + this->m_size * realDataTypeSize(m_dtype),
                                m_guard_len,
                                hipMemcpyDeviceToHost),
                      hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

            // Point to m_guard before allocated memory
            d -= m_pad * realDataTypeSize(m_dtype);

            // Copy device memory after allocated memory to host
            EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

            delete[] host;
        }
#endif
    }

    void device_vector_teardown(char* d)
    {
        if(d != nullptr)
        {
#ifdef GOOGLE_TEST
            if(m_pad > 0)
            {
                char* host = new char[m_pad * realDataTypeSize(m_dtype)];

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host,
                                    d + this->m_size * realDataTypeSize(m_dtype),
                                    m_guard_len,
                                    hipMemcpyDeviceToHost),
                          hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

                // Point to m_guard before allocated memory
                d -= m_pad * realDataTypeSize(m_dtype);

                // Copy device memory after allocated memory to host
                EXPECT_EQ(hipMemcpy(host, d, m_guard_len, hipMemcpyDeviceToHost), hipSuccess);

                // Make sure no corruption has occurred
                EXPECT_EQ(memcmp(host, m_guard_type, m_guard_len), 0);

                delete[] host;
            }
#endif
            // Free device memory
            if((hipFree)(d) != hipSuccess)
            {
                hipblaslt_cerr << "free device memory failed" << std::endl;
            }
        }
    }
};

template <typename T>
T d_vector<T>::m_guard[MEM_MAX_GUARD_PAD] = {};

template <typename T>
bool d_vector<T>::m_init_guard = false;

#undef MEM_MAX_GUARD_PAD
#undef MAX_DTYPE_SIZE
