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

#include "hipblaslt.h"
#include "exceptions.hpp"
#include "hipblaslt-ext-op.h"
#include "hipblaslt_internal.hpp"

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocblaslt.h>
#include <stdio.h>
#include <stdlib.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

hipblasStatus_t hipErrorToHIPBLASStatus(hipError_t status)
{
    switch(status)
    {
    case hipSuccess:
        return HIPBLAS_STATUS_SUCCESS;
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case hipErrorInvalidDevicePointer:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case hipErrorInvalidValue:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case hipErrorNoDevice:
    case hipErrorUnknown:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    default:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
}

hipblasStatus_t RocBlasLtStatusToHIPStatus(rocblaslt_status_ status)
{
    switch(status)
    {
    case rocblaslt_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblaslt_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblaslt_status_not_implemented:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_pointer:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_invalid_size:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblaslt_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_arch_mismatch:
        return HIPBLAS_STATUS_ARCH_MISMATCH;
    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIPBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)              \
    {                                                                  \
        hipblasStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPBLAS_STATUS_SUCCESS)             \
        {                                                              \
            return TMP_STATUS_FOR_CHECK;                               \
        }                                                              \
    }

#define RETURN_IF_ROCBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        rocblaslt_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblaslt_status_success)            \
        {                                                               \
            return RocBlasLtStatusToHIPStatus(TMP_STATUS_FOR_CHECK);    \
        }                                                               \
    }

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

hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle)
try
{
    // TODO: Synchronizer size pass into predicate SynchronizerSizeCheck
    // 1K just for small size now, need to cal corner case if support all situations
    void* d_Synchronizer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_Synchronizer, 16 * 1024 * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(d_Synchronizer, 0, sizeof(int) * 16 * 1024));

    // Check if handle is valid
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int             deviceId;
    hipError_t      err;
    hipblasStatus_t retval = HIPBLAS_STATUS_SUCCESS;

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = RocBlasLtStatusToHIPStatus(rocblaslt_create((rocblaslt_handle*)handle));
        (*(rocblaslt_handle*)handle)->Synchronizer = d_Synchronizer;
    }
    return retval;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle)
try
{
    if(handle != nullptr and (*(rocblaslt_handle)handle).Synchronizer != nullptr)
    {
        CHECK_HIP_ERROR(hipFree((*(rocblaslt_handle)handle).Synchronizer));
    }

    return RocBlasLtStatusToHIPStatus(rocblaslt_destroy((const rocblaslt_handle)handle));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* matDescr,
                                            hipDataType              valueType,
                                            uint64_t                 rows,
                                            uint64_t                 cols,
                                            int64_t                  ld)
try
{
    return RocBlasLtStatusToHIPStatus(rocblaslt_matrix_layout_create(
        (rocblaslt_matrix_layout*)matDescr, valueType, rows, cols, ld));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t descr)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_destory((const rocblaslt_matrix_layout)descr));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t* matmulDesc,
                                          hipblasComputeType_t   computeType,
                                          hipDataType            scaleType)
try
{
    return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_desc_create(
        (rocblaslt_matmul_desc*)matmulDesc, (rocblaslt_compute_type)computeType, scaleType));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t          matLayout,
                                                  hipblasLtMatrixLayoutAttribute_t attr,
                                                  const void*                      buf,
                                                  size_t                           sizeInBytes)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_set_attribute((rocblaslt_matrix_layout)matLayout,
                                              (rocblaslt_matrix_layout_attribute)attr,
                                              buf,
                                              sizeInBytes));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(hipblasLtMatrixLayout_t          matLayout,
                                                  hipblasLtMatrixLayoutAttribute_t attr,
                                                  void*                            buf,
                                                  size_t                           sizeInBytes,
                                                  size_t*                          sizeWritten)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_get_attribute((rocblaslt_matrix_layout)matLayout,
                                              (rocblaslt_matrix_layout_attribute)attr,
                                              buf,
                                              sizeInBytes,
                                              sizeWritten));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t descr)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_destroy((const rocblaslt_matmul_desc)descr));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t           matmulDesc,
                                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                                const void*                     buf,
                                                size_t                          sizeInBytes)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_set_attribute((rocblaslt_matmul_desc)matmulDesc,
                                            (rocblaslt_matmul_desc_attributes)matmulAttr,
                                            buf,
                                            sizeInBytes));
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t           matmulDesc,
                                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                                void*                           buf,
                                                size_t                          sizeInBytes,
                                                size_t*                         sizeWritten)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_get_attribute((rocblaslt_matmul_desc)matmulDesc,
                                            (rocblaslt_matmul_desc_attributes)matmulAttr,
                                            buf,
                                            sizeInBytes,
                                            sizeWritten));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* pref)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_create((rocblaslt_matmul_preference*)pref));
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_destroy((const rocblaslt_matmul_preference)pref));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t           pref,
                                          hipblasLtMatmulPreferenceAttributes_t attribute,
                                          const void*                           data,
                                          size_t                                dataSize)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_set_attribute((rocblaslt_matmul_preference)pref,
                                                  (rocblaslt_matmul_preference_attributes)attribute,
                                                  data,
                                                  dataSize));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulPreferenceGetAttribute(hipblasLtMatmulPreference_t           pref,
                                          hipblasLtMatmulPreferenceAttributes_t attribute,
                                          void*                                 data,
                                          size_t                                sizeInBytes,
                                          size_t*                               sizeWritten)
try
{
    return RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_get_attribute((rocblaslt_matmul_preference)pref,
                                                  (rocblaslt_matmul_preference_attributes)attribute,
                                                  data,
                                                  sizeInBytes,
                                                  sizeWritten));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t                handle,
                                    hipblasLtMatmulDesc_t            matmulDesc,
                                    hipblasLtMatrixLayout_t          Adesc,
                                    hipblasLtMatrixLayout_t          Bdesc,
                                    hipblasLtMatrixLayout_t          Cdesc,
                                    hipblasLtMatrixLayout_t          Ddesc,
                                    hipblasLtMatmulPreference_t      pref,
                                    int                              requestedAlgoCount,
                                    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                    int*                             returnAlgoCount)
try
{
    return RocBlasLtStatusToHIPStatus(rocblaslt_matmul_algo_get_heuristic(
        (rocblaslt_handle)handle,
        (rocblaslt_matmul_desc)matmulDesc,
        (rocblaslt_matrix_layout)Adesc,
        (rocblaslt_matrix_layout)Bdesc,
        (rocblaslt_matrix_layout)Cdesc,
        (rocblaslt_matrix_layout)Ddesc,
        (rocblaslt_matmul_preference)pref,
        requestedAlgoCount,
        (rocblaslt_matmul_heuristic_result*)heuristicResultsArray,
        returnAlgoCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t            handle,
                                hipblasLtMatmulDesc_t        matmul_descr,
                                const void*                  alpha,
                                const void*                  A,
                                hipblasLtMatrixLayout_t      matA,
                                const void*                  B,
                                hipblasLtMatrixLayout_t      matB,
                                const void*                  beta,
                                const void*                  C,
                                hipblasLtMatrixLayout_t      matC,
                                void*                        D,
                                hipblasLtMatrixLayout_t      matD,
                                const hipblasLtMatmulAlgo_t* algo,
                                void*                        workspace,
                                size_t                       workspaceSizeInBytes,
                                hipStream_t                  stream)
try
{
    hipblasStatus_t return_status          = HIPBLAS_STATUS_SUCCESS;
    rocblaslt_matmul_desc roc_matmul_desc  = (rocblaslt_matmul_desc)matmul_descr;
    rocblaslt_compute_type compute_type    = roc_matmul_desc->compute_type;
    bool amaxScaleB                        = roc_matmul_desc->amaxScaleB;
    bool isScaleAmaxDivisorB               = roc_matmul_desc->isScaleAmaxDivisorB;
    float amaxDividendB                    = roc_matmul_desc->amaxDividendB;
    void* scaleB                           = roc_matmul_desc->scaleB;
    const char* case2                      = getenv("CASE2");


    if (roc_matmul_desc->amaxScaleA)
    {
        rocblaslt_matrix_layout tmp_matA = (rocblaslt_matrix_layout)matA;
        if(roc_matmul_desc->isScaleAmaxDivisorA)
            hipblasltExtFastValueDevidedByAMax(tmp_matA->type, HIP_R_32F, roc_matmul_desc->scaleA, A, workspace, ((rocblaslt_handle)handle)->Synchronizer, tmp_matA->m, tmp_matA->n, roc_matmul_desc->amaxDividendA, stream);
        else
            hipblasltExtFastAMax(tmp_matA->type, HIP_R_32F, roc_matmul_desc->scaleA, A, workspace, ((rocblaslt_handle)handle)->Synchronizer, tmp_matA->m, tmp_matA->n, stream);
    }

    std::cout << "API amaxScaleB " << roc_matmul_desc->amaxScaleB << std::endl;
    std::cout << "API isScaleAmaxDivisorB " << roc_matmul_desc->isScaleAmaxDivisorB << std::endl;
    std::cout << "API amaxDividendB " << roc_matmul_desc->amaxDividendB << std::endl;
    std::cout << "API compute_type " << roc_matmul_desc->compute_type << std::endl;

//    if (case2 != nullptr)
//    {
//        if(roc_matmul_desc->scaleB != nullptr)
//            throw rocblaslt_status_internal_error;
//
//        roc_matmul_desc->compute_type        = rocblaslt_compute_f32_fast_f8_fnuz;
//        roc_matmul_desc->amaxScaleB          = true;
//        roc_matmul_desc->isScaleAmaxDivisorB = true;
//        roc_matmul_desc->amaxDividendB       = 240.0f;
//        roc_matmul_desc->scaleB              = workspace;
//        workspace                            = (void*)((char*)workspace + 4);
//        workspaceSizeInBytes                 = workspaceSizeInBytes - 4;
//    }

    if (roc_matmul_desc->amaxScaleB)
    {
        int* sync = ((int*)((rocblaslt_handle)handle)->Synchronizer);

        rocblaslt_matrix_layout tmp_matB = (rocblaslt_matrix_layout)matB;
        if(roc_matmul_desc->isScaleAmaxDivisorB)
            hipblasltExtFastValueDevidedByAMax(tmp_matB->type, HIP_R_32F, roc_matmul_desc->scaleB, B, workspace, sync, tmp_matB->m, tmp_matB->n, roc_matmul_desc->amaxDividendB, stream);
        else
            hipblasltExtFastAMax(tmp_matB->type, HIP_R_32F, roc_matmul_desc->scaleB, B, workspace, sync, tmp_matB->m, tmp_matB->n, stream);
    }

    if(roc_matmul_desc->amax_ptr == nullptr)
    {
        return_status
            = RocBlasLtStatusToHIPStatus(rocblaslt_matmul((rocblaslt_handle)handle,
                                                          (rocblaslt_matmul_desc)matmul_descr,
                                                          alpha,
                                                          A,
                                                          (rocblaslt_matrix_layout)matA,
                                                          B,
                                                          (rocblaslt_matrix_layout)matB,
                                                          beta,
                                                          C,
                                                          (rocblaslt_matrix_layout)matC,
                                                          D,
                                                          (rocblaslt_matrix_layout)matD,
                                                          (const rocblaslt_matmul_algo*)algo,
                                                          workspace,
                                                          workspaceSizeInBytes,
                                                          stream));
    }
    else //Amax path
    {
        rocblaslt_matrix_layout tmp_matC = (rocblaslt_matrix_layout)matC;
        rocblaslt_matrix_layout tmp_matD = (rocblaslt_matrix_layout)matD;
        size_t amax_workspace_size       = tmp_matD->m * tmp_matD->n * 4; //only support fp32 D temp
        void*  scaleD                    = ((rocblaslt_matmul_desc)matmul_descr)->scaleD;
        if(!(tmp_matD->type == HIP_R_8F_E4M3_FNUZ || tmp_matD->type == HIP_R_8F_E5M2_FNUZ)
           || tmp_matD->m != tmp_matD->ld || *(float*)beta != 0 || tmp_matD->batch_count > 1
           || workspaceSizeInBytes < amax_workspace_size || scaleD == nullptr)
            return HIPBLAS_STATUS_INTERNAL_ERROR;

        hipDataType c_type                            = tmp_matD->type;
        hipDataType d_type                            = tmp_matD->type;
        void*       D_TEMP                            = workspace;
        tmp_matC->type                                = HIP_R_32F;
        tmp_matD->type                                = HIP_R_32F;
        ((rocblaslt_matmul_desc)matmul_descr)->scaleD = nullptr;
        char* new_workspace                           = (char*)workspace;
        new_workspace += amax_workspace_size;
        workspaceSizeInBytes -= amax_workspace_size;
        return_status
            = RocBlasLtStatusToHIPStatus(rocblaslt_matmul((rocblaslt_handle)handle,
                                                          (rocblaslt_matmul_desc)matmul_descr,
                                                          alpha,
                                                          A,
                                                          (rocblaslt_matrix_layout)matA,
                                                          B,
                                                          (rocblaslt_matrix_layout)matB,
                                                          beta,
                                                          C,
                                                          (rocblaslt_matrix_layout)matC,
                                                          D_TEMP,
                                                          (rocblaslt_matrix_layout)matD,
                                                          (const rocblaslt_matmul_algo*)algo,
                                                          new_workspace,
                                                          workspaceSizeInBytes,
                                                          stream));
        //reset matD type
        tmp_matC->type                                = c_type;
        tmp_matD->type                                = d_type;
        ((rocblaslt_matmul_desc)matmul_descr)->scaleD = scaleD;
        if(return_status != HIPBLAS_STATUS_SUCCESS)
            return return_status;

        return_status = hipblasltExtAMaxWithScale(HIP_R_32F,
                                                  HIP_R_32F,
                                                  d_type,
                                                  ((rocblaslt_matmul_desc)matmul_descr)->amax_ptr,
                                                  D,
                                                  D_TEMP,
                                                  scaleD,
                                                  tmp_matD->m,
                                                  tmp_matD->n,
                                                  stream);
    }

//    if (case2 != nullptr)
//    {
//        roc_matmul_desc->compute_type        = compute_type;
//        roc_matmul_desc->amaxScaleB          = amaxScaleB;
//        roc_matmul_desc->isScaleAmaxDivisorB = isScaleAmaxDivisorB;
//        roc_matmul_desc->amaxDividendB       = amaxDividendB;
//        roc_matmul_desc->scaleB              = scaleB;
//    }

    return return_status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixTransformDescCreate(hipblasLtMatrixTransformDesc_t* transformDesc,
                                                   hipDataType                     scaleType)
{
    static_assert(sizeof(rocblaslt_matrix_transform_desc)
                      <= sizeof(hipblasLtMatrixTransformDescOpaque_t),
                  "hipblasLtMatrixTransformDescOpaque_t must have enough space");
    rocblaslt_matrix_transform_desc desc;
    desc.scaleType = scaleType;
    *transformDesc = new hipblasLtMatrixTransformDescOpaque_t;
    memcpy((*transformDesc)->data, &desc, sizeof(desc));
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasLtMatrixTransformDescDestroy(hipblasLtMatrixTransformDesc_t transformDesc)
{
    if(transformDesc)
        delete transformDesc;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t
    hipblasLtMatrixTransformDescSetAttribute(hipblasLtMatrixTransformDesc_t           transformDesc,
                                             hipblasLtMatrixTransformDescAttributes_t attr,
                                             const void*                              buf,
                                             size_t                                   sizeInBytes)
{
    if(!buf || sizeInBytes != sizeof(int32_t))
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    rocblaslt_matrix_transform_desc* desc
        = reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]);
    // all possible values should be int32_t
    assert(sizeInBytes == sizeof(int32_t));
    int32_t value{};
    memcpy(&value, buf, sizeInBytes);

    switch(attr)
    {
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
    {
        desc->scaleType = static_cast<hipDataType>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
    {
        desc->pointerMode = static_cast<hipblasLtPointerMode_t>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
    {
        desc->opA = static_cast<hipblasOperation_t>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
    {
        desc->opB = static_cast<hipblasOperation_t>(value);
        break;
    }
    default:
        assert(false && "Unknown attribute");
        return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    }
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t
    hipblasLtMatrixTransformDescGetAttribute(hipblasLtMatrixTransformDesc_t           transformDesc,
                                             hipblasLtMatrixTransformDescAttributes_t attr,
                                             void*                                    buf,
                                             size_t                                   sizeInBytes,
                                             size_t*                                  sizeWritten)
{
    if(!sizeInBytes && !sizeWritten)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    if(sizeInBytes && !sizeWritten)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    if(sizeInBytes != sizeof(int32_t))
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    rocblaslt_matrix_transform_desc* desc
        = reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]);
    int32_t value{};

    switch(attr)
    {
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
    {
        value = static_cast<int32_t>(desc->scaleType);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
    {
        value = static_cast<int32_t>(desc->pointerMode);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
    {
        value = static_cast<int32_t>(desc->opA);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
    {
        value = static_cast<int32_t>(desc->opB);
        break;
    }
    default:
        return HIPBLAS_STATUS_INVALID_VALUE;
        assert(false && "Unknown attribute");
        break;
    }

    memcpy(buf, &value, sizeInBytes);
    *sizeWritten = sizeof(int32_t);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasLtMatrixTransform(hipblasLtHandle_t              lightHandle,
                                         hipblasLtMatrixTransformDesc_t transformDesc,
                                         const void*             alpha, /* host or device pointer */
                                         const void*             A,
                                         hipblasLtMatrixLayout_t Adesc,
                                         const void*             beta, /* host or device pointer */
                                         const void*             B,
                                         hipblasLtMatrixLayout_t Bdesc,
                                         void*                   C,
                                         hipblasLtMatrixLayout_t Cdesc,
                                         hipStream_t             stream)
{
    return RocBlasLtStatusToHIPStatus(rocblaslt_matrix_transform(
        (rocblaslt_handle)lightHandle,
        reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]),
        alpha,
        A,
        (rocblaslt_matrix_layout)Adesc,
        beta,
        B,
        (rocblaslt_matrix_layout)Bdesc,
        C,
        (rocblaslt_matrix_layout)Cdesc,
        stream));
}

// Other Utilities
hipblasStatus_t hipblasLtGetVersion(hipblasLtHandle_t handle, int* version)
try
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    *version = HIPBLASLT_VERSION_MAJOR * 100000 + HIPBLASLT_VERSION_MINOR * 100
               + HIPBLASLT_VERSION_PATCH;

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtGetGitRevision(hipblasLtHandle_t handle, char* rev)
try
{
    // Get hipBLASLt revision
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    if(rev == nullptr)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(HIPBLASLT_VERSION_TWEAK);

    memcpy(rev, v, sizeof(v));

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtGetArchName(char** archName)
try
{
    *archName        = nullptr;
    std::string arch = rocblaslt_internal_get_arch_name();
    *archName        = (char*)malloc(arch.size() * sizeof(char));
    strncpy(*archName, arch.c_str(), arch.size());
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    if(archName != nullptr)
    {
        free(*archName);
        *archName = nullptr;
    }
    return exception_to_hipblas_status();
}

#ifdef __cplusplus
}
#endif
