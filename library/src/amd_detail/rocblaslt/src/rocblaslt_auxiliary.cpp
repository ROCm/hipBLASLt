/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "definitions.h"
#include "handle.h"
#include "rocblaslt.h"
#include "rocblaslt_mat_utils.hpp"
#include "tensile_host.hpp"
#include "utility.hpp"

#ifndef WIN32
#include <link.h>
#endif

#include <hip/hip_runtime_api.h>
#include <unistd.h>
#include <utility>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

inline void assignAlphaBeta1(const rocblaslt_compute_type& compute_type, void* alpha, void* beta)
{
    if(compute_type == rocblaslt_compute_f64)
    {
        *((double*)alpha) = 1.f;
        *((double*)beta)  = 1.f;
    }
    else if(compute_type == rocblaslt_compute_i32)
    {
        *((int32_t*)alpha) = 1.f;
        *((int32_t*)beta)  = 1.f;
    }
    else
    {
        *((float*)alpha) = 1.f;
        *((float*)beta)  = 1.f;
    }
}

/******************************************************************************
 * construct_rocblaslt_problem creates RocblasltContractionProblem from mat    *
 * layout and descriptor for Tensile's findTopSolutions.                      *
 ******************************************************************************/
RocblasltContractionProblem construct_rocblaslt_problem(rocblaslt_handle            handle,
                                                        const rocblaslt_matmul_desc matmul_descr,
                                                        rocblaslt_matrix_layout     matA,
                                                        rocblaslt_matrix_layout     matB,
                                                        rocblaslt_matrix_layout     matC,
                                                        rocblaslt_matrix_layout     matD,
                                                        const void*                 alpha,
                                                        const void*                 beta,
                                                        size_t maxWorkSpaceBytes)
{
    int8_t      dummy;
    const void* dummy_ptr = &dummy;
    int64_t     m, n, k, lda, ldb, ldc, ldd, lde, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_stride_d, batch_stride_e;
    hipDataType            bias_type;
    hipDataType            a_type, b_type, c_type, d_type;
    rocblaslt_compute_type compute_type;
    void *                 bias = nullptr, *scaleAlphaVec = nullptr, *e = nullptr;
    bool                   gradient = false;
    rocblaslt_status       isValid  = rocblaslt_matmul_valid_args(matmul_descr,
                                                           dummy_ptr,
                                                           dummy_ptr,
                                                           dummy_ptr,
                                                           dummy_ptr,
                                                           matA,
                                                           matB,
                                                           matC,
                                                           matD,
                                                           alpha,
                                                           beta,
                                                           m,
                                                           n,
                                                           k,
                                                           a_type,
                                                           lda,
                                                           batch_stride_a,
                                                           b_type,
                                                           ldb,
                                                           batch_stride_b,
                                                           c_type,
                                                           ldc,
                                                           batch_stride_c,
                                                           d_type,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleAlphaVec,
                                                           e,
                                                           gradient,
                                                           compute_type);
    if(isValid != rocblaslt_status_continue)
    {
        m = 0;
        n = 0;
        k = 0;
    }

    // Internal assign
    hipblasOperation_t opA           = matmul_descr->op_A;
    hipblasOperation_t opB           = matmul_descr->op_B;
    int                num_batches_a = matA->batch_count;
    rocblaslt_epilogue epilogue      = matmul_descr->epilogue;
    void*              scaleA        = matmul_descr->scaleA;
    void*              scaleB        = matmul_descr->scaleB;
    void*              scaleC        = matmul_descr->scaleC;
    void*              scaleD        = matmul_descr->scaleD;
    void*              scaleE        = matmul_descr->scaleE;

    // Others
    constexpr bool strided_batch = true;
    constexpr bool grouped_gemm  = false;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        setTo1(matmul_descr->compute_type, (void*)alpha_1, &alpha);
    }

    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        a_type,
                                        nullptr,
                                        nullptr,
                                        lda,
                                        batch_stride_a,
                                        b_type,
                                        nullptr,
                                        nullptr,
                                        ldb,
                                        batch_stride_b,
                                        beta,
                                        c_type,
                                        nullptr,
                                        nullptr,
                                        ldc,
                                        batch_stride_c,
                                        d_type,
                                        nullptr,
                                        nullptr,
                                        ldd,
                                        batch_stride_d,
                                        e,
                                        nullptr,
                                        lde,
                                        batch_stride_e,
                                        num_batches_a,
                                        strided_batch,
                                        grouped_gemm,
                                        gradient,
                                        compute_type,
                                        bias,
                                        scaleA,
                                        scaleB,
                                        scaleC,
                                        scaleD,
                                        scaleE,
                                        scaleAlphaVec,
                                        bias_type,
                                        epilogue,
                                        nullptr,
                                        maxWorkSpaceBytes,
                                        nullptr,
                                        handle->Synchronizer};

    return problem;
}

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocblaslt_handle is a structure holding the rocblaslt library context.
 * It must be initialized using rocblaslt_create()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocblaslt_destroy().
 *******************************************************************************/
rocblaslt_status rocblaslt_create(rocblaslt_handle* handle)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        log_error(__func__, "invalid handle pointer", handle);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        *handle = nullptr;
        // Allocate
        try
        {
            *handle = new _rocblaslt_handle();
            log_api(__func__, "handle[out]", *handle);
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
rocblaslt_status rocblaslt_destroy(const rocblaslt_handle handle)
{
    if(handle == nullptr)
    {
        log_error(__func__, "handle", handle);
        return rocblaslt_status_invalid_value;
    }
    log_api(__func__, "handle", handle);
    // Destruct
    try
    {
        delete handle;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief rocblaslt_matrix_layout is a structure holding the rocblaslt matrix
 * content. It must be initialized using rocblaslt_matrix_layout_create()
 * and the retured handle must be passed
 * to all subsequent library function calls that involve the matrix.
 * It should be destroyed at the end using rocblaslt_matrix_layout_destory().
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_create(rocblaslt_matrix_layout* matDescr,
                                                hipDataType              valueType,
                                                uint64_t                 rows,
                                                uint64_t                 cols,
                                                int64_t                  ld)
{
    // Check if matDescr is valid
    if(matDescr == nullptr)
    {
        log_error(__func__, "invalid matDescr pointer", matDescr);
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        *matDescr = nullptr;
        // Allocate
        try
        {
            *matDescr         = new _rocblaslt_matrix_layout();
            (*matDescr)->m    = rows;
            (*matDescr)->n    = cols;
            (*matDescr)->ld   = ld;
            (*matDescr)->type = valueType;
            log_api(__func__,
                    "matLayout[out]",
                    matDescr,
                    "type",
                    hipDataType_to_string(valueType),
                    "rows",
                    rows,
                    "cols",
                    cols,
                    "ld",
                    ld);
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_destory(const rocblaslt_matrix_layout matDescr)
{
    if(matDescr == nullptr)
    {
        log_error(__func__, "matDescr", matDescr);
        return rocblaslt_status_invalid_pointer;
    }
    log_api(__func__, "matLayout", matDescr);
    // Destruct
    try
    {
        delete matDescr;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *descriptor.
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_set_attribute(rocblaslt_matrix_layout           matLayout,
                                                       rocblaslt_matrix_layout_attribute attr,
                                                       const void*                       buf,
                                                       size_t sizeInBytes)
{
    // Check if matLayout is valid
    if(matLayout == nullptr)
    {
        log_error(__func__, "invalid matLayout pointer", matLayout);
        return rocblaslt_status_invalid_handle;
    }
    else if(buf == nullptr)
    {
        log_error(__func__, "invalid buf pointer", buf);
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes <= 0)
    {
        log_error(__func__, "invalid buf size", sizeInBytes);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        // Allocate
        try
        {
            switch(attr)
            {
            case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matLayout->batch_count, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matLayout->batch_stride, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_TYPE:
                if(sizeof(uint32_t) <= sizeInBytes)
                    memcpy(&matLayout->type, buf, sizeof(uint32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_ORDER:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matLayout->order, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_ROWS:
                if(sizeof(uint64_t) <= sizeInBytes)
                    memcpy(&matLayout->m, buf, sizeof(uint64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_COLS:
                if(sizeof(uint64_t) <= sizeInBytes)
                    memcpy(&matLayout->n, buf, sizeof(uint64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATRIX_LAYOUT_LD:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matLayout->ld, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            default:
                log_error(__func__, "invalid attribute", attr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matLayout",
                    matLayout,
                    "attr",
                    rocblaslt_matrix_layout_attributes_to_string(attr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData",
                    (void*)(intptr_t)(*(int32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief Get the value of the specified attribute belonging to matrix
 *descriptor such as number of batches and their stride.
 *******************************************************************************/
rocblaslt_status rocblaslt_matrix_layout_get_attribute(rocblaslt_matrix_layout           matLayout,
                                                       rocblaslt_matrix_layout_attribute attr,
                                                       void*                             buf,
                                                       size_t  sizeInBytes,
                                                       size_t* sizeWritten)

{
    if(matLayout == nullptr)
    {
        log_error(__func__, "invalid matLayout pointer", matLayout);
        return rocblaslt_status_invalid_handle;
    }
    else if(sizeInBytes == 0 && sizeWritten == nullptr)
    {
        log_error(__func__, "invalid pointer: sizeWritten can't be nullptr if sizeInBytes is 0");
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes != 0 && buf == nullptr)
    {
        log_error(__func__, "invalid pointer: buf can't be nullptr if sizeInBytes isn't 0");
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        try
        {
            switch(attr)
            {
            case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matLayout->batch_count, sizeof(int32_t));
                break;
            case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
                if(sizeWritten)
                    *sizeWritten = sizeof(int64_t);
                if(sizeInBytes < sizeof(int64_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matLayout->batch_stride, sizeof(int64_t));
                break;
            default:
                log_error(__func__, "invalid attribute", attr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matLayout",
                    matLayout,
                    "attr",
                    rocblaslt_matrix_layout_attributes_to_string(attr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData[out]",
                    (void*)(intptr_t)(*(int32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_create(rocblaslt_matmul_desc* matmulDesc,
                                              rocblaslt_compute_type computeType,
                                              hipDataType            scaleType)
{
    // Check if matmulDesc is valid
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        *matmulDesc = nullptr;
        // Allocate
        try
        {
            switch(computeType)
            {
            case rocblaslt_compute_f32:
            case rocblaslt_compute_f32_fast_xf32:
            case rocblaslt_compute_f64:
            case rocblaslt_compute_i32:
            case rocblaslt_compute_f32_fast_f16:
            case rocblaslt_compute_f32_fast_bf16:
                break;
            default:
                log_error(__func__, "invalid compute type", computeType);
                throw rocblaslt_status_invalid_value;
            }

            if(scaleType != HIP_R_32F && scaleType != HIP_R_64F && scaleType != HIP_R_32I)
            {
                log_error(__func__, "invalid scale type", scaleType);
                throw rocblaslt_status_invalid_value;
            }

            *matmulDesc                 = new _rocblaslt_matmul_desc();
            (*matmulDesc)->compute_type = computeType;
            (*matmulDesc)->scale_type   = scaleType;
            auto computeTypeInit        = computeType == rocblaslt_compute_f32_fast_xf32
                                              ? rocblaslt_compute_f32
                                              : computeType;
            auto dataType               = HIP_R_32F;
            if(computeTypeInit == rocblaslt_compute_f64)
                dataType = HIP_R_64F;
            else if(computeType == rocblaslt_compute_i32)
                dataType = HIP_R_32I;

            initTensileGemmData(nullptr,
                                rocblaslt::RocGemmType::ROCBLASLT_GEMM,
                                HIPBLAS_OP_N,
                                HIPBLAS_OP_N,
                                dataType,
                                dataType,
                                dataType,
                                dataType,
                                computeTypeInit,
                                0,
                                (*matmulDesc)->m_data);

            log_api(__func__,
                    "matmulDesc[out]",
                    matmulDesc,
                    "computeType",
                    rocblaslt_compute_type_to_string(computeType),
                    "scaleType",
                    hipDataType_to_string(scaleType));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }

        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_destroy(const rocblaslt_matmul_desc matmulDesc)
{
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_pointer;
    }
    log_api(__func__, "matmulDesc", matmulDesc);

    // Destruct
    try
    {
        delete matmulDesc;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *multiplication descriptor.
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_set_attribute(rocblaslt_matmul_desc            matmulDesc,
                                                     rocblaslt_matmul_desc_attributes matmulAttr,
                                                     const void*                      buf,
                                                     size_t                           sizeInBytes)
{
    // Check if matmulDesc is valid
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_handle;
    }
    else if(buf == nullptr)
    {
        log_error(__func__, "invalid buf pointer", buf);
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes <= 0)
    {
        log_error(__func__, "invalid buf size", sizeInBytes);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        // Allocate
        try
        {
            switch(matmulAttr)
            {
            case ROCBLASLT_MATMUL_DESC_TRANSA:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->op_A, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_TRANSB:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->op_B, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->epilogue, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->bias, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleA, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleA buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleB, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleB buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_C_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleC, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleC buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_D_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleD, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleD buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->scaleE, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid scaleAux buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_POINTER_MODE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->pointermode, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid pointermode buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->bias_type, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
                if(sizeof(void*) <= sizeInBytes)
                    memcpy(&matmulDesc->e, buf, sizeof(void*));
                else
                {
                    log_error(__func__, "invalid e buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matmulDesc->lde, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid lde buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE:
                if(sizeof(int64_t) <= sizeInBytes)
                    memcpy(&matmulDesc->stride_e, buf, sizeof(int64_t));
                else
                {
                    log_error(__func__, "invalid stride_e buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                break;
            default:
                log_error(__func__, "invalid attribute", matmulAttr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matmulDesc",
                    matmulDesc,
                    "attr",
                    rocblaslt_matmul_desc_attributes_to_string(matmulAttr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData",
                    (void*)(uintptr_t)(*(uint32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *descriptor such as number of batches and their stride.
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_desc_get_attribute(rocblaslt_matmul_desc            matmulDesc,
                                                     rocblaslt_matmul_desc_attributes matmulAttr,
                                                     void*                            buf,
                                                     size_t                           sizeInBytes,
                                                     size_t*                          sizeWritten)

{
    // Check if matmulDesc is valid
    if(matmulDesc == nullptr)
    {
        log_error(__func__, "invalid matmulDescr pointer", matmulDesc);
        return rocblaslt_status_invalid_handle;
    }
    else if(sizeInBytes == 0 && sizeWritten == nullptr)
    {
        log_error(__func__, "invalid pointer: sizeWritten can't be nullptr if sizeInBytes is 0");
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes != 0 && buf == nullptr)
    {
        log_error(__func__, "invalid pointer: buf can't be nullptr if sizeInBytes isn't 0");
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        try
        {
            switch(matmulAttr)
            {
            case ROCBLASLT_MATMUL_DESC_TRANSA:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->op_A, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_TRANSB:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->op_B, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->epilogue, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
                if(sizeWritten)
                    *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->bias, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_A_SCALE_POINTER:
                if(sizeWritten)
                    *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->scaleA, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_B_SCALE_POINTER:
                if(sizeWritten)
                    *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->scaleB, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_POINTER_MODE:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->pointermode, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
                if(sizeWritten)
                    *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->bias_type, sizeof(int32_t));
                break;
            default:
                log_error(__func__, "invalid attribute", matmulAttr);
                return rocblaslt_status_invalid_value;
            }
            log_api(__func__,
                    "matmulDesc",
                    matmulDesc,
                    "attr",
                    rocblaslt_matmul_desc_attributes_to_string(matmulAttr),
                    "buf",
                    buf,
                    "sizeInBytes",
                    sizeInBytes,
                    "bufData[out]",
                    (void*)(uintptr_t)(*(uint32_t*)buf));
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_preference_create(rocblaslt_matmul_preference* pref)
{
    // Check if pref is valid
    if(pref == nullptr)
    {
        log_error(__func__, "invalid pointer", pref);
        return rocblaslt_status_invalid_handle;
    }
    *pref = nullptr;
    // Allocate
    try
    {
        *pref = new _rocblaslt_matmul_preference();
        log_api(__func__, "matmulPref[out]", *pref);
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_preference_destroy(const rocblaslt_matmul_preference pref)
{
    if(pref == nullptr)
    {
        log_error(__func__, "invalid pointer", pref);
        return rocblaslt_status_invalid_pointer;
    }

    log_api(__func__, "matmulPref", pref);
    // Destruct
    try
    {
        delete pref;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
    rocblaslt_matmul_preference_set_attribute(rocblaslt_matmul_preference            pref,
                                              rocblaslt_matmul_preference_attributes attribute,
                                              const void*                            data,
                                              size_t                                 dataSize)
{
    // Check if pref is valid
    if(data == nullptr || pref == nullptr)
    {
        log_error(__func__, "invalid pointer: data", data, "pref", pref);
        return rocblaslt_status_invalid_pointer;
    }
    else if(dataSize <= 0)
    {
        log_error(__func__, "invalid data size", dataSize);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        switch(attribute)
        {
        case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
            pref->search_mode = *(uint32_t*)data;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    dataSize,
                    "data",
                    pref->search_mode);
            break;
        case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            pref->max_workspace_bytes = *(uint64_t*)data;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    dataSize,
                    "data",
                    pref->max_workspace_bytes);
            break;
        default:
            log_error(__func__, "invalid attribute", attribute);
            return rocblaslt_status_invalid_value;
            break;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
    rocblaslt_matmul_preference_get_attribute(rocblaslt_matmul_preference            pref,
                                              rocblaslt_matmul_preference_attributes attribute,
                                              void*                                  data,
                                              size_t                                 sizeInBytes,
                                              size_t*                                sizeWritten)
{
    // Check if matmulDesc is valid
    if(data == nullptr || pref == nullptr)
    {
        log_error(__func__, "invalid pointer: data", data, "pref", pref);
        return rocblaslt_status_invalid_pointer;
    }
    else if(sizeInBytes <= 0)
    {
        log_error(__func__, "invalid data size", sizeInBytes);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        switch(attribute)
        {
        case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
            *sizeWritten     = sizeof(uint32_t);
            *(uint32_t*)data = pref->search_mode;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    sizeInBytes,
                    "data[out]",
                    pref->search_mode);
            break;
        case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
            *sizeWritten     = sizeof(uint64_t);
            *(uint64_t*)data = pref->max_workspace_bytes;
            log_api(__func__,
                    "matmulPref",
                    pref,
                    "attr",
                    attribute,
                    "buf",
                    data,
                    "sizeInBytes",
                    sizeInBytes,
                    "data[out]",
                    pref->max_workspace_bytes);
            break;
        default:
            return rocblaslt_status_invalid_value;
            break;
        }
        return rocblaslt_status_success;
    }
}

/********************************************************************************
 * \brief
 *******************************************************************************/

rocblaslt_status rocblaslt_matmul_is_algo_supported(rocblaslt_handle        handle,
                                                    rocblaslt_matmul_desc   matmul_descr,
                                                    const void*             alpha,
                                                    rocblaslt_matrix_layout matA,
                                                    rocblaslt_matrix_layout matB,
                                                    const void*             beta,
                                                    rocblaslt_matrix_layout matC,
                                                    rocblaslt_matrix_layout matD,
                                                    rocblaslt_matmul_algo*  algo,
                                                    size_t*                 workspaceSizeInBytes)
{
    // Check if handle is valid
    if(handle == nullptr || matmul_descr == nullptr || matA == nullptr || matB == nullptr
       || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return rocblaslt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr)
    {
        log_error(__func__, "invalid data pointer");
        return rocblaslt_status_invalid_pointer;
    }

    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        hipDataType            a_type       = matA->type;
        hipDataType            b_type       = matB->type;
        hipDataType            c_type       = matC->type;
        hipDataType            d_type       = matD->type;
        rocblaslt_compute_type compute_type = matmul_descr->compute_type;
        auto&                  gemmData     = matmul_descr->m_data;

        void* alphaf = (void*)alpha;
        void* betaf  = (void*)beta;
        auto  prob   = construct_rocblaslt_problem(
            handle, matmul_descr, matA, matB, matC, matD, alphaf, betaf, algo->max_workspace_bytes);
        status = isSolutionSupported(handle, prob, gemmData, algo, workspaceSizeInBytes);

        if(status != rocblaslt_status_success)
        {
            throw status;
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
    rocblaslt_matmul_algo_get_heuristic(rocblaslt_handle                  handle,
                                        rocblaslt_matmul_desc             matmul_desc,
                                        rocblaslt_matrix_layout           matA,
                                        rocblaslt_matrix_layout           matB,
                                        rocblaslt_matrix_layout           matC,
                                        rocblaslt_matrix_layout           matD,
                                        rocblaslt_matmul_preference       pref,
                                        int                               requestedAlgoCount,
                                        rocblaslt_matmul_heuristic_result heuristicResultsArray[],
                                        int*                              returnAlgoCount)
{
    // Check if handle is valid
    if(handle == nullptr || matmul_desc == nullptr || pref == nullptr || matA == nullptr
       || matB == nullptr || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid pointer");
        return rocblaslt_status_invalid_handle;
    }

    if(requestedAlgoCount < 1)
    {
        log_error(__func__, "invalid requested count", requestedAlgoCount);
        return rocblaslt_status_invalid_value;
    }
    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        hipDataType            a_type       = matA->type;
        hipDataType            b_type       = matB->type;
        hipDataType            c_type       = matC->type;
        hipDataType            d_type       = matD->type;
        rocblaslt_compute_type compute_type = matmul_desc->compute_type;
        auto&                  tensile_data = matmul_desc->m_data;

        int8_t alpha[16] = {0};
        int8_t beta[16]  = {0};
        assignAlphaBeta1(compute_type, (void*)alpha, (void*)beta);

        auto prob = construct_rocblaslt_problem(
            handle, matmul_desc, matA, matB, matC, matD, &alpha, &beta, pref->max_workspace_bytes);
        status = getBestSolutions(prob,
                                  handle,
                                  tensile_data,
                                  requestedAlgoCount,
                                  heuristicResultsArray,
                                  returnAlgoCount,
                                  pref->max_workspace_bytes);

        log_api(__func__, "returnAlogCount", *returnAlgoCount);

        //Try to get size independent solutions from getAllSolutions()
        if(requestedAlgoCount > *returnAlgoCount)
        {
            std::vector<rocblaslt_matmul_heuristic_result> allSolutionsResults;
            if(rocblaslt_status_success
               == getAllSolutions(prob, handle, allSolutionsResults, pref->max_workspace_bytes))
            {
                int oriReturnAlgoCount = *returnAlgoCount;
                for(int i = 0;
                    *returnAlgoCount < requestedAlgoCount && i < allSolutionsResults.size();
                    i++)
                {
                    bool duplicated_sol = false;
                    for(int j = 0; j < oriReturnAlgoCount; j++)
                        if(*(int*)(heuristicResultsArray[j].algo.data)
                           == *(int*)(allSolutionsResults[i].algo.data)) //solution index
                            duplicated_sol = true;

                    if(duplicated_sol == true
                       || rocblaslt_status_success
                              != isSolutionSupported(handle,
                                                     prob,
                                                     tensile_data,
                                                     &allSolutionsResults[i].algo,
                                                     &pref->max_workspace_bytes))
                        continue;
                    //append sol to heuristpicResultsArray
                    memcpy(heuristicResultsArray[*returnAlgoCount].algo.data,
                           allSolutionsResults[i].algo.data,
                           sizeof(heuristicResultsArray[i].algo.data));
                    heuristicResultsArray[*returnAlgoCount].algo.max_workspace_bytes
                        = pref->max_workspace_bytes;
                    heuristicResultsArray[*returnAlgoCount].algo.fallback = false;
                    heuristicResultsArray[*returnAlgoCount].state = rocblaslt_status_success;
                    heuristicResultsArray[*returnAlgoCount].workspaceSize
                        = allSolutionsResults[i].workspaceSize;
                    (*returnAlgoCount)++;
                }

                log_api(__func__, "final returnAlogCount", *returnAlgoCount);
            }
        }

        if(status != rocblaslt_status_success)
        {
            throw status;
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}
#ifdef __cplusplus
}
#endif

void rocblaslt_init_gemmData(rocblaslt_handle       handle,
                             rocblaslt::RocGemmType gemmType,
                             hipblasOperation_t     opA,
                             hipblasOperation_t     opB,
                             hipDataType            typeA,
                             hipDataType            typeB,
                             hipDataType            typeC,
                             hipDataType            typeD,
                             rocblaslt_compute_type typeCompute,
                             size_t                 maxWorkspaceBytes,
                             std::shared_ptr<void>& gemmData)
{
    initTensileGemmData(handle,
                        gemmType,
                        opA,
                        opB,
                        typeA,
                        typeB,
                        typeC,
                        typeD,
                        typeCompute,
                        maxWorkspaceBytes,
                        gemmData);
}

rocblaslt_status rocblaslt_matmul_get_all_algos_cpp(
    rocblaslt_handle                                handle,
    rocblaslt::RocGemmType                          typeGemm,
    hipblasOperation_t                              opA,
    hipblasOperation_t                              opB,
    hipDataType                                     typeA,
    hipDataType                                     typeB,
    hipDataType                                     typeC,
    hipDataType                                     typeD,
    rocblaslt_compute_type                          typeCompute,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        log_error(__func__, "invalid pointer");
        return rocblaslt_status_invalid_handle;
    }
    // Create dummy
    auto initMat = [](_rocblaslt_matrix_layout& mat, hipDataType type) {
        mat.m    = 1;
        mat.n    = 1;
        mat.ld   = 1;
        mat.type = type;
    };
    _rocblaslt_matmul_desc   matmul_desc;
    _rocblaslt_matrix_layout matA;
    _rocblaslt_matrix_layout matB;
    _rocblaslt_matrix_layout matC;
    _rocblaslt_matrix_layout matD;
    initMat(matA, typeA);
    initMat(matB, typeB);
    initMat(matC, typeC);
    initMat(matD, typeD);
    matmul_desc.op_A                  = opA;
    matmul_desc.op_B                  = opB;
    matmul_desc.compute_type          = typeCompute;
    matmul_desc.scale_type            = typeD;
    rocblaslt_status status           = rocblaslt_status_success;
    size_t           maxWorkspaceSize = std::numeric_limits<size_t>::max();
    try
    {
        int8_t alpha[16] = {0};
        int8_t beta[16]  = {0};
        assignAlphaBeta1(matmul_desc.compute_type, (void*)alpha, (void*)beta);

        auto prob = construct_rocblaslt_problem(
            handle, &matmul_desc, &matA, &matB, &matC, &matD, &alpha, &beta, maxWorkspaceSize);
        if(typeGemm == rocblaslt::RocGemmType::ROCBLASLT_GEMM)
        {
            status = getAllSolutions(prob, handle, heuristicResults, maxWorkspaceSize);
        }
        else if(typeGemm == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
        {
            std::vector<RocblasltContractionProblem> probs = {prob};
            status = getAllSolutions(probs, handle, heuristicResults, maxWorkspaceSize);
        }
        else
        {
            log_api(__func__, "Invalid gemm type", static_cast<int>(typeGemm));
            status = rocblaslt_status_not_implemented;
        }

        if(status != rocblaslt_status_success)
        {
            throw status;
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_matmul_get_algos_from_index_cpp(
    rocblaslt_handle                                handle,
    std::vector<int>&                               solutionIndex,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults)
{
    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        size_t maxWorkspaceSize = std::numeric_limits<size_t>::max();
        status = getSolutionsFromIndex(handle, solutionIndex, heuristicResults, maxWorkspaceSize);

        log_api(__func__, "returnAlogCount", heuristicResults.size());
        return status;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_is_algo_supported_cpp(rocblaslt_handle            handle,
                                                 rocblaslt::RocGemmType      gemmType,
                                                 std::shared_ptr<void>       gemmData,
                                                 rocblaslt_matmul_algo&      algo,
                                                 const rocblaslt::RocTuning* tuning,
                                                 size_t&                     workspaceSizeInBytes)
{
    return isSolutionSupported(handle, gemmType, gemmData, algo, tuning, workspaceSizeInBytes);
}

rocblaslt_status
    rocblaslt_algo_get_heuristic_cpp(rocblaslt_handle       handle,
                                     rocblaslt::RocGemmType gemmType,
                                     std::shared_ptr<void>  gemmData,
                                     const int              workspaceBytes,
                                     const int              requestedAlgoCount,
                                     std::vector<rocblaslt_matmul_heuristic_result>& results)
{
    if(requestedAlgoCount < 1)
    {
        log_error(__func__, "invalid requested count", requestedAlgoCount);
        return rocblaslt_status_invalid_value;
    }
    if(gemmType == rocblaslt::RocGemmType::ROCBLASLT_GROUPED_GEMM)
    {
        log_api(__func__, "not implement for groupedgemm, please use get_all_algos instead");
        return rocblaslt_status_success;
    }
    rocblaslt_status status = rocblaslt_status_success;
    try
    {
        status = getBestSolutions(
            handle, gemmType, gemmData, workspaceBytes, requestedAlgoCount, results);

        log_api(__func__, "returnAlogCount", results.size());
        if(status != rocblaslt_status_success)
        {
            throw status;
        }
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_copy_matmul(rocblaslt_matmul_desc src, rocblaslt_matmul_desc dst)
{
    if(src == nullptr)
    {
        log_error(__func__, "invalid src matmulDescr pointer", src);
        return rocblaslt_status_invalid_pointer;
    }
    if(dst == nullptr)
    {
        log_error(__func__, "invalid dst matmulDescr pointer", dst);
        return rocblaslt_status_invalid_pointer;
    }
    dst->copy(*src);
    return rocblaslt_status_success;
}

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

struct ArchName
{
    std::string operator()(const hipDeviceProp_t& prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop.gcnArchName);
        std::string gcnArch = gcnArchName.substr(0, gcnArchName.find(":"));
        return gcnArch;
    }
};

// exported. Get architecture name
std::string rocblaslt_internal_get_arch_name()
{
    int deviceId;
    static_cast<void>(hipGetDevice(&deviceId));
    hipDeviceProp_t deviceProperties;
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    return ArchName{}(deviceProperties);
}

bool rocblaslt_internal_test_path(const std::string& path)
{
#ifdef WIN32
    return ((_access(path.c_str(), 4) != -1) || (_access(path.c_str(), 6) != -1));
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

#ifndef WIN32
int hipblaslt_dl_iterate_phdr_callback(struct dl_phdr_info* hdr_info, size_t size, void* data)
{
    // uncomment to see all dependent .so files
    // fprintf(stderr, "hipblaslt so file: %s\n", hdr_info->dlpi_name);
    std::pair<std::string, std::string>* typedData
        = reinterpret_cast<std::pair<std::string, std::string>*>(data);
    if(hdr_info->dlpi_name && strstr(hdr_info->dlpi_name, typedData->second.c_str()))
    {
        typedData->first.assign(hdr_info->dlpi_name);
        return 1;
    }
    return 0;
}
#endif

std::string rocblaslt_internal_get_so_path(const std::string& keyword)
{
    std::pair<std::string, std::string> result{"", keyword};
    dl_iterate_phdr(hipblaslt_dl_iterate_phdr_callback, &result);
    return result.first;
}

void rocblaslt_log_error(const char* func, const char* var, const char* msg)
{
    log_error(func, var, msg);
}
