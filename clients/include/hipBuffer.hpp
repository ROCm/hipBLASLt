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

#include "d_vector.hpp"
#include "datatype_interface.hpp"
#include "hipblaslt_ostream.hpp"

class HipDeviceBuffer : public d_vector_type
{
public:
    HipDeviceBuffer(hipDataType dtype, std::size_t numElements, bool HMM = false)
        : d_vector_type(dtype, numElements, HMM)
        , numBytes(realDataTypeSize(dtype) * numElements)
        , buffer{this->device_vector_setup(), &hipFree}
    {
    }

    ~HipDeviceBuffer()
    {
        this->device_vector_teardown(static_cast<char*>(buffer.get()));
        buffer = nullptr;
    }

    HipDeviceBuffer(const HipDeviceBuffer&)            = delete;
    HipDeviceBuffer(HipDeviceBuffer&&)                 = default;
    HipDeviceBuffer& operator=(const HipDeviceBuffer&) = delete;
    HipDeviceBuffer& operator=(HipDeviceBuffer&&)      = default;

    void* buf()
    {
        return buffer.get();
    }

    const void* buf() const
    {
        return buffer.get();
    }

    std::size_t getNumBytes() const
    {
        return numBytes;
    }

    template <typename T>
    T* as()
    {
        return reinterpret_cast<T*>(buf());
    }

    template <typename T>
    const T* as() const
    {
        return reinterpret_cast<const T*>(buf());
    }

private:
    std::size_t                               numBytes;
    std::unique_ptr<void, decltype(&hipFree)> buffer;
};

class HipHostBuffer
{
public:
    HipHostBuffer(hipDataType dtype, std::size_t numElements)
        : numBytes(realDataTypeSize(dtype) * numElements ? realDataTypeSize(dtype) * numElements
                                                         : realDataTypeSize(dtype))
        , buffer(createBuffer(numBytes))
    {
    }

    ~HipHostBuffer()                               = default;
    HipHostBuffer(const HipHostBuffer&)            = delete;
    HipHostBuffer(HipHostBuffer&&)                 = default;
    HipHostBuffer& operator=(const HipHostBuffer&) = delete;
    HipHostBuffer& operator=(HipHostBuffer&&)      = default;

    void* end()
    {
        return (void*)((char*)buffer.get() + getNumBytes());
    }

    const void* end() const
    {
        return (void*)((const char*)buffer.get() + getNumBytes());
    }

    void* buf()
    {
        return buffer.get();
    }

    const void* buf() const
    {
        return buffer.get();
    }

    std::size_t getNumBytes() const
    {
        return numBytes;
    }

    template <typename T>
    T* as()
    {
        return reinterpret_cast<T*>(buf());
    }

    template <typename T>
    const T* as() const
    {
        return reinterpret_cast<const T*>(buf());
    }

private:
    std::size_t                                      numBytes;
    std::unique_ptr<void, decltype(&hipFree)>        buffer;
    static std::unique_ptr<void, decltype(&hipFree)> createBuffer(std::size_t numBytes)
    {
        void* rawBuf{};
        (void)hipHostMalloc(&rawBuf, numBytes);
        if(!rawBuf)
        {
            hipblaslt_cerr << "Error to allocate memory in HipHostBuffer" << std::endl;
        }
        return std::unique_ptr<void, decltype(&hipFree)>(rawBuf, &hipFree);
    }
};

inline hipError_t
    synchronize(HipDeviceBuffer& dBuf, const HipHostBuffer& hBuf, std::size_t repeats = 1)
{
    hipError_t hip_err;
    for(size_t i = 0; i < repeats; ++i)
    {
        hip_err = hipMemcpy(dBuf.as<char>() + i * dBuf.getNumBytes() / repeats,
                            hBuf.as<char>(),
                            dBuf.getNumBytes() / repeats,
                            dBuf.use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);

        if(hip_err != hipSuccess)
        {
            return hip_err;
        }
    }
    return hip_err;
}

inline hipError_t synchronize(HipHostBuffer& hBuf, const HipDeviceBuffer& dBuf)
{
    hipError_t hip_err;
    if(hipSuccess != (hip_err = hipDeviceSynchronize()))
        return hip_err;

    return hipMemcpy(hBuf.as<char>(), dBuf.as<char>(), hBuf.getNumBytes(), hipMemcpyDeviceToHost);
}

template <typename T1>
inline void copy_buf(HipHostBuffer& src, HipHostBuffer& dst)
{
    std::copy(
        static_cast<T1*>(src.buf()), static_cast<T1*>(src.end()), static_cast<T1*>(dst.buf()));
}

inline void copy_buf(HipHostBuffer& src, HipHostBuffer& dst, hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        copy_buf<float>(src, dst);
        break;
    case HIP_R_64F:
        copy_buf<double>(src, dst);
        break;
    case HIP_R_16F:
        copy_buf<hipblasLtHalf>(src, dst);
        break;
    case HIP_R_16BF:
        copy_buf<hip_bfloat16>(src, dst);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        copy_buf<hipblaslt_f8_fnuz>(src, dst);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        copy_buf<hipblaslt_bf8_fnuz>(src, dst);
        break;
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        copy_buf<hipblaslt_f8>(src, dst);
        break;
    case HIP_R_8F_E5M2:
        copy_buf<hipblaslt_bf8>(src, dst);
        break;
#endif
    case HIP_R_32I:
        copy_buf<int32_t>(src, dst);
        break;
    case HIP_R_8I:
        copy_buf<hipblasLtInt8>(src, dst);
        break;
    default:
        hipblaslt_cerr << "Error type in copy_buf" << std::endl;
        break;
    }
}

template <typename T1, typename Tc>
inline void transform_buf(HipHostBuffer& src, HipHostBuffer& dst)
{
    if constexpr(std::is_same<Tc, float>::value
                 || !(std::is_same<T1, hipblaslt_bf8_fnuz>::value
                      || std::is_same<T1, hipblaslt_f8_fnuz>::value))
    {
#ifdef ROCM_USE_FLOAT8
        if constexpr(std::is_same<Tc, float>::value
                     || !(std::is_same<T1, hipblaslt_bf8>::value
                          || std::is_same<T1, hipblaslt_f8>::value))
#endif
            std::transform(static_cast<T1*>(src.buf()),
                           static_cast<T1*>(src.end()),
                           static_cast<Tc*>(dst.buf()),
                           [](T1 c) -> Tc { return static_cast<Tc>(c); });
    }
}

template <typename T1>
inline void _transform_buf(HipHostBuffer& src, HipHostBuffer& dst, hipDataType typeTc)
{
    switch(typeTc)
    {
    case HIP_R_32F:
        transform_buf<T1, float>(src, dst);
        break;
    case HIP_R_64F:
        transform_buf<T1, double>(src, dst);
        break;
    case HIP_R_16F:
        transform_buf<T1, hipblasLtHalf>(src, dst);
        break;
    case HIP_R_32I:
        transform_buf<T1, int32_t>(src, dst);
        break;
    default:
        hipblaslt_cerr << "Error type in transform_buf" << std::endl;
        break;
    }
}

inline void
    transform_buf(HipHostBuffer& src, HipHostBuffer& dst, hipDataType type, hipDataType typeTc)
{
    switch(type)
    {
    case HIP_R_32F:
        _transform_buf<float>(src, dst, typeTc);
        break;
    case HIP_R_64F:
        _transform_buf<double>(src, dst, typeTc);
        break;
    case HIP_R_16F:
        _transform_buf<hipblasLtHalf>(src, dst, typeTc);
        break;
    case HIP_R_16BF:
        _transform_buf<hip_bfloat16>(src, dst, typeTc);
        break;
    case HIP_R_8F_E4M3_FNUZ:
        _transform_buf<hipblaslt_f8_fnuz>(src, dst, typeTc);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        _transform_buf<hipblaslt_bf8_fnuz>(src, dst, typeTc);
        break;
#ifdef ROCM_USE_FLOAT8
    case HIP_R_8F_E4M3:
        _transform_buf<hipblaslt_f8>(src, dst, typeTc);
        break;
    case HIP_R_8F_E5M2:
        _transform_buf<hipblaslt_bf8>(src, dst, typeTc);
        break;
#endif
    case HIP_R_32I:
        _transform_buf<int32_t>(src, dst, typeTc);
        break;
    case HIP_R_8I:
        _transform_buf<hipblasLtInt8>(src, dst, typeTc);
        break;
    default:
        hipblaslt_cerr << "Error type in transform_buf" << std::endl;
        break;
    }
}
