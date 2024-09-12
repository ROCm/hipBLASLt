/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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

#include "hipblaslt_arguments.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_test.hpp"
#include "singletons.hpp"
#include <cinttypes>
#include <hipblaslt/hipblaslt.h>

#define MEM_MAX_GUARD_PAD 8192

/* ============================================================================================ */

//!
//! @brief enum to specify the type of allocated device memory.
//!
typedef enum
{
    // Normal
    HIPBLASLT_DEVICE_MEMORY_NORMAL,

    // Managed
    HIPBLASLT_DEVICE_MEMORY_MANAGED,

    // Un-cached
    HIPBLASLT_DEVICE_MEMORY_UNCACHED

} hipblaslt_device_memory_type;

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
    hipblaslt_device_memory_type m_type = HIPBLASLT_DEVICE_MEMORY_NORMAL;

public:
    static T m_guard[MEM_MAX_GUARD_PAD];

#ifdef GOOGLE_TEST
    d_vector(size_t s, hipblaslt_device_memory_type memType = HIPBLASLT_DEVICE_MEMORY_NORMAL)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(T))
        , m_bytes((s + m_pad * 2) * sizeof(T))
        , m_type(memType)
    {
        // Initialize m_guard with random data
        if(!m_init_guard)
        {
            hipblaslt_init_nan(m_guard, MEM_MAX_GUARD_PAD);
            m_init_guard = true;
        }
    }
#else
    d_vector(size_t s, hipblaslt_device_memory_type memType = HIPBLASLT_DEVICE_MEMORY_NORMAL)
        : m_size(s)
        , m_pad(0) // save current pad length
        , m_guard_len(0 * sizeof(T))
        , m_bytes(s ? s * sizeof(T) : sizeof(T))
        , m_type(memType)
    {
    }
#endif

    T* device_vector_setup()
    {
        T* d = nullptr;
        switch(m_type){
            case HIPBLASLT_DEVICE_MEMORY_NORMAL:
                if((hipMalloc)(&d, m_bytes) != hipSuccess)
                    d = nullptr;
                break;
            case HIPBLASLT_DEVICE_MEMORY_MANAGED:
                if(hipMallocManaged(&d, m_bytes) != hipSuccess)
                    d = nullptr;
                break;
            case HIPBLASLT_DEVICE_MEMORY_UNCACHED:
                if(hipExtMallocWithFlags((void**)&d, m_bytes, hipDeviceMallocUncached) != hipSuccess)
                    d = nullptr;
                break;
            default:
                hipblaslt_cerr << "Wrong device memory type: " << m_type << std::endl;
                return nullptr;
                break;
        }
        if(d == nullptr){
            hipblaslt_cerr << "Error allocating un-cached memory " << m_bytes << " m_bytes (" << (m_bytes >> 30)
                            << " GB)" << std::endl;
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
            T *host = new T[m_pad];

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
                T *host = new T[m_pad];

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

template <typename T>
T d_vector<T>::m_guard[MEM_MAX_GUARD_PAD] = {};

template <typename T>
bool d_vector<T>::m_init_guard = false;

#undef MEM_MAX_GUARD_PAD
