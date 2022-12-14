/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "definitions.h"
#include "handle.h"
#include "rocblaslt.h"
#include "tensile_host.hpp"
#include "utility.hpp"

#include <hip/hip_runtime_api.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

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
                                                hipblasDatatype_t        valueType,
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
                    hipblasDatatype_to_string(valueType),
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
                    (void*)(*(int32_t*)buf));
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
    else if(buf == nullptr or sizeWritten == nullptr)
    {
        log_error(__func__, "invalid pointer: buf", buf, "sizeWritten", sizeWritten);
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        try
        {
            switch(attr)
            {
            case ROCBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
                *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matLayout->batch_count, sizeof(int32_t));
                break;
            case ROCBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
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
                    (void*)(*(int32_t*)buf));
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
                                              hipblasDatatype_t      scaleType)
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
            if(computeType != rocblaslt_compute_f32)
            {
                log_error(__func__, "invalid compute type", computeType);
                throw rocblaslt_status_invalid_value;
            }

            if(scaleType != HIPBLAS_R_32F)
            {
                log_error(__func__, "invalid scale type", scaleType);
                throw rocblaslt_status_invalid_value;
            }

            *matmulDesc                 = new _rocblaslt_matmul_desc();
            (*matmulDesc)->compute_type = computeType;
            (*matmulDesc)->scale_type   = scaleType;
            log_api(__func__,
                    "matmulDesc[out]",
                    matmulDesc,
                    "computeType",
                    rocblaslt_compute_type_to_string(computeType),
                    "scaleType",
                    hipblasDatatype_to_string(scaleType));
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
            case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
                if(sizeof(int32_t) <= sizeInBytes)
                    memcpy(&matmulDesc->bias_type, buf, sizeof(int32_t));
                else
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
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
                    (void*)(*(uint32_t*)buf));
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
    else if(buf == nullptr or sizeWritten == nullptr)
    {
        log_error(__func__, "invalid pointer: buf", buf, "sizeWritten", sizeWritten);
        return rocblaslt_status_invalid_pointer;
    }
    else
    {
        try
        {
            switch(matmulAttr)
            {
            case ROCBLASLT_MATMUL_DESC_TRANSA:
                *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->op_A, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_TRANSB:
                *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->op_B, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_EPILOGUE:
                *sizeWritten = sizeof(int32_t);
                if(sizeInBytes < sizeof(int32_t))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->epilogue, sizeof(int32_t));
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
                *sizeWritten = sizeof(void*);
                if(sizeInBytes < sizeof(void*))
                {
                    log_error(__func__, "invalid buf size", sizeInBytes);
                    return rocblaslt_status_invalid_value;
                }
                memcpy(buf, &matmulDesc->bias, sizeof(void*));
                break;
            case ROCBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
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
                    (void*)(*(uint32_t*)buf));
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
        hipblasDatatype_t      a_type       = matA->type;
        hipblasDatatype_t      b_type       = matB->type;
        hipblasDatatype_t      c_type       = matC->type;
        hipblasDatatype_t      d_type       = matD->type;
        rocblaslt_compute_type compute_type = matmul_desc->compute_type;
        if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F)
        {
            if(c_type == HIPBLAS_R_32F && d_type == HIPBLAS_R_32F)
            {
                if(compute_type == rocblaslt_compute_f32)
                {
                    float alpha = 1.0;
                    float beta  = 1.0;
                    auto  prob
                        = ConstructRocblasltProblem<float, float, float>(handle,
                                                                         matmul_desc,
                                                                         matA,
                                                                         matB,
                                                                         matC,
                                                                         matD,
                                                                         &alpha,
                                                                         &beta,
                                                                         pref->max_workspace_bytes);
                    status = getBestSolutions<float, float, float>(prob,
                                                                   requestedAlgoCount,
                                                                   heuristicResultsArray,
                                                                   returnAlgoCount,
                                                                   pref->max_workspace_bytes);
                }
            }
        }
        else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F)
        {
            if(c_type == HIPBLAS_R_16F && d_type == HIPBLAS_R_16F)
            {
                if(compute_type == rocblaslt_compute_f32)
                {
                    float alpha = 1.0;
                    float beta  = 1.0;
                    auto  prob  = ConstructRocblasltProblem<rocblaslt_half, rocblaslt_half, float>(
                        handle,
                        matmul_desc,
                        matA,
                        matB,
                        matC,
                        matD,
                        &alpha,
                        &beta,
                        pref->max_workspace_bytes);
                    status = getBestSolutions<rocblaslt_half, rocblaslt_half, float>(
                        prob,
                        requestedAlgoCount,
                        heuristicResultsArray,
                        returnAlgoCount,
                        pref->max_workspace_bytes);
                }
            }
        }
        else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B)
        {
            if(c_type == HIPBLAS_R_16B && d_type == HIPBLAS_R_16B)
            {
                if(compute_type == rocblaslt_compute_f32)
                {
                    float alpha = 1.0;
                    float beta  = 1.0;
                    auto  prob
                        = ConstructRocblasltProblem<rocblaslt_bfloat16, rocblaslt_bfloat16, float>(
                            handle,
                            matmul_desc,
                            matA,
                            matB,
                            matC,
                            matD,
                            &alpha,
                            &beta,
                            pref->max_workspace_bytes);
                    status = getBestSolutions<rocblaslt_bfloat16, rocblaslt_bfloat16, float>(
                        prob,
                        requestedAlgoCount,
                        heuristicResultsArray,
                        returnAlgoCount,
                        pref->max_workspace_bytes);
                }
            }
        }
        else
        {
            status = rocblaslt_status_not_implemented;
        }

        log_api(__func__, "returnAlogCount", *returnAlgoCount);
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

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

// Emulate C++17 std::void_t
template <typename...>
using void_t = void;

// By default, use gcnArch converted to a string prepended by gfx
template <typename PROP, typename = void>
struct ArchName
{
    std::string operator()(const PROP& prop) const
    {
        return "gfx" + std::to_string(prop.gcnArch);
    }
};

// If gcnArchName exists as a member, use it instead
template <typename PROP>
struct ArchName<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP& prop) const
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
    hipGetDevice(&deviceId);
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties(&deviceProperties, deviceId);
    return ArchName<hipDeviceProp_t>{}(deviceProperties);
}
