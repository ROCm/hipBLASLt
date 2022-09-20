/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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
rocblaslt_status rocblaslt_create(rocblaslt_handle *handle) {
  // Check if handle is valid
  if (handle == nullptr) {
    return rocblaslt_status_invalid_handle;
  } else {
    *handle = nullptr;
    // Allocate
    try {
      *handle = new _rocblaslt_handle();
      log_trace(*handle, "rocblaslt_create");
    } catch (const rocblaslt_status &status) {
      return status;
    }
    return rocblaslt_status_success;
  }
}

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
rocblaslt_status rocblaslt_destroy(const rocblaslt_handle handle) {
  log_trace(handle, "rocblaslt_destroy");
  // Destruct
  try {
    delete handle;
  } catch (const rocblaslt_status &status) {
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
rocblaslt_status
rocblaslt_matrix_layout_create(rocblaslt_matrix_layout *matDescr,
                               hipDataType valueType, uint64_t rows,
                               uint64_t cols, int64_t ld) {
  // Check if matDescr is valid
  if (matDescr == nullptr) {
    return rocblaslt_status_invalid_pointer;
  } else {
    *matDescr = nullptr;
    // Allocate
    try {
      *matDescr = new _rocblaslt_matrix_layout();
      (*matDescr)->m = rows;
      (*matDescr)->n = cols;
      (*matDescr)->ld = ld;
      (*matDescr)->type = valueType;
    } catch (const rocblaslt_status &status) {
      return status;
    }
    return rocblaslt_status_success;
  }
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocblaslt_status
rocblaslt_matrix_layout_destory(const rocblaslt_matrix_layout matDescr) {
  if (matDescr == nullptr)
    return rocblaslt_status_invalid_pointer;
  // Destruct
  try {
    delete matDescr;
  } catch (const rocblaslt_status &status) {
    return status;
  }
  return rocblaslt_status_success;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
rocblaslt_matmul_desc_create(rocblaslt_matmul_desc *matmulDesc,
                             rocblaslt_compute_type computeType,
                             hipDataType scaleType) {
  // Check if matmulDesc is valid
  if (matmulDesc == nullptr) {
    return rocblaslt_status_invalid_pointer;
  } else {
    *matmulDesc = nullptr;
    // Allocate
    try {
      if (computeType != rocblaslt_compute_f32)
        throw rocblaslt_status_invalid_value;

      if (scaleType != HIP_R_32F)
        throw rocblaslt_status_invalid_value;

      *matmulDesc = new _rocblaslt_matmul_desc();
      (*matmulDesc)->compute_type = computeType;
      (*matmulDesc)->scale_type = scaleType;
    } catch (const rocblaslt_status &status) {
      return status;
    }

    return rocblaslt_status_success;
  }
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocblaslt_status
rocblaslt_matmul_desc_destroy(const rocblaslt_matmul_desc matmulDesc) {
  if (matmulDesc == nullptr) {
    return rocblaslt_status_invalid_pointer;
  }

  // Destruct
  try {
    delete matmulDesc;
  } catch (const rocblaslt_status &status) {
    return status;
  }
  return rocblaslt_status_success;
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *multiplication descriptor.
 *******************************************************************************/
rocblaslt_status
rocblaslt_matmul_desc_set_attribute(rocblaslt_matmul_desc matmulDesc,
                                    rocblaslt_matmul_desc_attributes matmulAttr,
                                    const void *buf, size_t sizeInBytes) {
  // Check if matmulDesc is valid
  if (matmulDesc == nullptr) {
    return rocblaslt_status_invalid_handle;
  } else if (buf == nullptr) {
    return rocblaslt_status_invalid_pointer;
  } else if (sizeInBytes <= 0) {
    return rocblaslt_status_invalid_value;
  } else {
    // Allocate
    try {
      switch (matmulAttr) {
      case ROCBLASLT_MATMUL_DESC_TRANSA:
        if (sizeof(int32_t) <= sizeInBytes)
          memcpy(&matmulDesc->op_A, buf, sizeof(int32_t));
        else
          return rocblaslt_status_invalid_value;
        break;
      case ROCBLASLT_MATMUL_DESC_TRANSB:
        if (sizeof(int32_t) <= sizeInBytes)
          memcpy(&matmulDesc->op_B, buf, sizeof(int32_t));
        else
          return rocblaslt_status_invalid_value;
        break;
      case ROCBLASLT_MATMUL_DESC_EPILOGUE:
        if (sizeof(int32_t) <= sizeInBytes)
          memcpy(&matmulDesc->epilogue, buf, sizeof(int32_t));
        else
          return rocblaslt_status_invalid_value;
        break;
      case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
        if (sizeof(void *) <= sizeInBytes)
          memcpy(&matmulDesc->bias, buf, sizeof(void *));
        else
          return rocblaslt_status_invalid_value;
        break;
      default:
        return rocblaslt_status_invalid_value;
      }
    } catch (const rocblaslt_status &status) {
      return status;
    }
    return rocblaslt_status_success;
  }
}

/********************************************************************************
 * \brief sets the value of the specified attribute belonging to matrix
 *descriptor such as number of batches and their stride.
 *******************************************************************************/
rocblaslt_status
rocblaslt_matmul_desc_get_attribute(rocblaslt_matmul_desc matmulDesc,
                                    rocblaslt_matmul_desc_attributes matmulAttr,
                                    void *buf, size_t sizeInBytes,
                                    size_t *sizeWritten)

{
  // Check if matmulDesc is valid
  if (matmulDesc == nullptr) {
    return rocblaslt_status_invalid_handle;
  } else if (buf == nullptr or sizeWritten == nullptr) {
    return rocblaslt_status_invalid_pointer;
  } else {
    try {
      switch (matmulAttr) {
      case ROCBLASLT_MATMUL_DESC_TRANSA:
        *sizeWritten = sizeof(int32_t);
        if (sizeInBytes < sizeof(int32_t))
          return rocblaslt_status_invalid_value;
        memcpy(buf, &matmulDesc->op_A, sizeof(int32_t));
        break;
      case ROCBLASLT_MATMUL_DESC_TRANSB:
        *sizeWritten = sizeof(int32_t);
        if (sizeInBytes < sizeof(int32_t))
          return rocblaslt_status_invalid_value;
        memcpy(buf, &matmulDesc->op_B, sizeof(int32_t));
        break;
      case ROCBLASLT_MATMUL_DESC_EPILOGUE:
        *sizeWritten = sizeof(int32_t);
        if (sizeInBytes < sizeof(int32_t))
          return rocblaslt_status_invalid_value;
        memcpy(buf, &matmulDesc->epilogue, sizeof(int32_t));
        break;
      case ROCBLASLT_MATMUL_DESC_BIAS_POINTER:
        *sizeWritten = sizeof(void *);
        if (sizeInBytes < sizeof(void *))
          return rocblaslt_status_invalid_value;
        memcpy(buf, &matmulDesc->bias, sizeof(void *));
        break;
      default:
        return rocblaslt_status_invalid_value;
      }
    } catch (const rocblaslt_status &status) {
      return status;
    }
    return rocblaslt_status_success;
  }
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status
rocblaslt_matmul_preference_create(rocblaslt_matmul_preference *pref) {
  // Check if pref is valid
  *pref = nullptr;
  // Allocate
  try {
    *pref = new _rocblaslt_matmul_preference();
  } catch (const rocblaslt_status &status) {
    return status;
  }
  return rocblaslt_status_success;
}

/********************************************************************************
 * \brief destroy matrix multiplication descriptor
 *******************************************************************************/
rocblaslt_status
rocblaslt_matmul_preference_destroy(const rocblaslt_matmul_preference pref) {
  if (pref == nullptr) {
    return rocblaslt_status_invalid_pointer;
  }

  // Destruct
  try {
    delete pref;
  } catch (const rocblaslt_status &status) {
    return status;
  }
  return rocblaslt_status_success;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul_preference_set_attribute(
    rocblaslt_matmul_preference pref,
    rocblaslt_matmul_preference_attributes attribute, const void *data,
    size_t dataSize) {
  // Check if pref is valid
  if (data == nullptr || pref == nullptr) {
    return rocblaslt_status_invalid_pointer;
  } else if (dataSize <= 0) {
    return rocblaslt_status_invalid_value;
  } else {
    switch (attribute) {
    case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
      pref->search_mode = *(uint32_t *)data;
      break;
    case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
      pref->max_workspace_bytes = *(uint64_t *)data;
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
rocblaslt_status rocblaslt_matmul_preference_get_attribute(
    rocblaslt_matmul_preference pref,
    rocblaslt_matmul_preference_attributes attribute, void *data,
    size_t dataSize)

{
  // Check if matmulDesc is valid
  if (data == nullptr || pref == nullptr) {
    return rocblaslt_status_invalid_pointer;
  } else if (dataSize <= 0) {
    return rocblaslt_status_invalid_value;
  } else {
    switch (attribute) {
    case ROCBLASLT_MATMUL_PREF_SEARCH_MODE:
      *(uint32_t *)data = pref->search_mode;
      break;
    case ROCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES:
      *(uint64_t *)data = pref->max_workspace_bytes;
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
rocblaslt_status rocblaslt_matmul_algo_get_heuristic(
    rocblaslt_handle handle, rocblaslt_matmul_desc matmul_desc,
    rocblaslt_matrix_layout matA, rocblaslt_matrix_layout matB,
    rocblaslt_matrix_layout matC, rocblaslt_matrix_layout matD,
    rocblaslt_matmul_preference pref, int requestedAlgoCount,
    rocblaslt_matmul_heuristic_result heuristicResultsArray[],
    int *returnAlgoCount) {
  // Check if handle is valid
  if (handle == nullptr || matmul_desc == nullptr || pref == nullptr ||
      matA == nullptr || matB == nullptr || matC == nullptr ||
      matD == nullptr) {
    return rocblaslt_status_invalid_handle;
  }

  if (requestedAlgoCount < 1)
    return rocblaslt_status_invalid_value;

  try {
    *returnAlgoCount = 1;
    heuristicResultsArray[0].algo.max_workspace_bytes =
        pref->max_workspace_bytes;
    heuristicResultsArray[0].state = rocblaslt_status_success;
  } catch (const rocblaslt_status &status) {
    return status;
  }
  return rocblaslt_status_success;
}
/********************************************************************************
 * \brief Get rocBLASLt version
 * version % 100        = patch level
 * version / 100 % 1000 = minor version
 * version / 100000     = major version
 *******************************************************************************/
rocblaslt_status rocblaslt_get_version(rocblaslt_handle handle, int *version) {
  // Check if handle is valid
  if (handle == nullptr) {
    return rocblaslt_status_invalid_handle;
  }
  *version = ROCBLASLT_VERSION_MAJOR * 100000 + ROCBLASLT_VERSION_MINOR * 100 +
             ROCBLASLT_VERSION_PATCH;

  log_trace(handle, "rocblaslt_get_version", *version);

  return rocblaslt_status_success;
}

/********************************************************************************
 * \brief Get rocBLASLt git revision
 *******************************************************************************/
rocblaslt_status rocblaslt_get_git_rev(rocblaslt_handle handle, char *rev) {
  // Check if handle is valid
  if (handle == nullptr) {
    return rocblaslt_status_invalid_handle;
  }

  if (rev == nullptr) {
    return rocblaslt_status_invalid_pointer;
  }

  static constexpr char v[] = TO_STR(ROCBLASLT_VERSION_TWEAK);

  memcpy(rev, v, sizeof(v));

  log_trace(handle, "rocblaslt_get_git_rev", rev);

  return rocblaslt_status_success;
}

#ifdef __cplusplus
}
#endif

/*******************************************************************************
 * GPU architecture-related functions
 ******************************************************************************/

// Emulate C++17 std::void_t
template <typename...> using void_t = void;

// By default, use gcnArch converted to a string prepended by gfx
template <typename PROP, typename = void> struct ArchName {
  std::string operator()(const PROP &prop) const {
    return "gfx" + std::to_string(prop.gcnArch);
  }
};

// If gcnArchName exists as a member, use it instead
template <typename PROP>
struct ArchName<PROP, void_t<decltype(PROP::gcnArchName)>> {
  std::string operator()(const PROP &prop) const {
    // strip out xnack/ecc from name
    std::string gcnArchName(prop.gcnArchName);
    std::string gcnArch = gcnArchName.substr(0, gcnArchName.find(":"));
    return gcnArch;
  }
};

// exported. Get architecture name
std::string rocblaslt_internal_get_arch_name() {
  int deviceId;
  hipGetDevice(&deviceId);
  hipDeviceProp_t deviceProperties;
  hipGetDeviceProperties(&deviceProperties, deviceId);
  return ArchName<hipDeviceProp_t>{}(deviceProperties);
}
