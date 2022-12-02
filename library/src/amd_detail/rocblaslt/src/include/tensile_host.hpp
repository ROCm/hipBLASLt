/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

/*********************************************************
 * Declaration of the rocBLAS<->Tensile interface layer. *
 *********************************************************/

#pragma once

/*****************************************************************************
 * WARNING: Tensile-specific data types, functions and macros should only be *
 * referenced from tensile_host.cpp. This header file defines the interface  *
 * that the rest of rocBLAS uses to access Tensile. If another Tensile       *
 * feature needs to be accessed, the API for accessing it should be defined  *
 * in this file, without referencing any Tensile-specific identifiers here.  *
 *****************************************************************************/

#include "handle.h"
//#include "tuple_helper.hpp"
#include "utility.hpp"
#include <atomic>

/********************************************************************
 * RocblasltContractionProblem captures the arguments for a GEMM-like *
 * contraction problem, to be passed to runContractionProblem.      *
 ********************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To>
struct RocblasltContractionProblem {
  rocblaslt_handle handle;
  hipblasOperation_t trans_a;
  hipblasOperation_t trans_b;

  // The RocblasltContractionProblem data members should exactly match
  // Tensile's parameter types, even if rocBLAS uses differently
  // sized or signed types. The constructors should convert rocBLAS
  // types into the corresponding Tensile types stored in this class.
  size_t m;
  size_t n;
  size_t k;

  const Tc *alpha;

  const Ti *A;
  const Ti *const *batch_A;
  size_t row_stride_a;
  size_t col_stride_a;
  size_t batch_stride_a;
  size_t buffer_offset_a;

  const Ti *B;
  const Ti *const *batch_B;
  size_t row_stride_b;
  size_t col_stride_b;
  size_t batch_stride_b;
  size_t buffer_offset_b;

  const Tc *beta;

  const To *C;
  const To *const *batch_C;
  size_t row_stride_c;
  size_t col_stride_c;
  size_t batch_stride_c;
  size_t buffer_offset_c;

  To *D;
  To *const *batch_D;
  size_t row_stride_d;
  size_t col_stride_d;
  size_t batch_stride_d;
  size_t buffer_offset_d;

  size_t batch_count;
  bool strided_batch;

  const To *bias;
  rocblaslt_epilogue epilogue;
  void *workspace;
  size_t workspaceSize;

  hipStream_t stream;

  // gemm_ex
  // gemm_strided_batched_ex
  RocblasltContractionProblem(
      rocblaslt_handle handle, hipblasOperation_t trans_a,
      hipblasOperation_t trans_b, int64_t m, int64_t n, int64_t k,
      const Tc *alpha, const Ti *A, const Ti *const *batch_A, int64_t ld_a,
      int64_t batch_stride_a, int64_t offset_a, const Ti *B,
      const Ti *const *batch_B, int64_t ld_b, int64_t batch_stride_b,
      int64_t offset_b, const Tc *beta, const To *C, const To *const *batch_C,
      int64_t ld_c, int64_t batch_stride_c, int64_t offset_c, To *D,
      To *const *batch_D, int64_t ld_d, int64_t batch_stride_d,
      int64_t offset_d, int64_t batch_count, bool strided_batch, const To *bias,
      rocblaslt_epilogue epilogue, void *workspace, size_t workspaceSize,
      hipStream_t stream)
      : handle(handle), trans_a(trans_a), trans_b(trans_b), m(m), n(n), k(k),
        alpha(alpha), A(A), batch_A(batch_A), row_stride_a(1),
        col_stride_a(ld_a), batch_stride_a(batch_stride_a),
        buffer_offset_a(offset_a), B(B), batch_B(batch_B), row_stride_b(1),
        col_stride_b(ld_b), batch_stride_b(batch_stride_b),
        buffer_offset_b(offset_b), beta(beta), C(C), batch_C(batch_C),
        row_stride_c(1), col_stride_c(ld_c), batch_stride_c(batch_stride_c),
        buffer_offset_c(offset_c), D(D), batch_D(batch_D), row_stride_d(1),
        col_stride_d(ld_d), batch_stride_d(batch_stride_d),
        buffer_offset_d(offset_d), batch_count(batch_count),
        strided_batch(strided_batch), bias(bias), epilogue(epilogue),
        workspace(workspace), workspaceSize(workspaceSize), stream(stream) {}
};

/*******************************************************************************
 * runContractionProblem() solves a RocblasltContractionProblem *
 *******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblaslt_status
runContractionProblem(const rocblaslt_matmul_algo *algo,
                      RocblasltContractionProblem<Ti, To, Tc> const &problem);

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for
 *testing) *
 ***********************************************************************************/
std::atomic_bool &rocblaslt_internal_tensile_is_initialized();

/**********************************************
 * Whether to suppress Tensile error messages *
 **********************************************/
inline bool &rocblaslt_suppress_tensile_error_messages() {
  thread_local bool t_suppress = false;
  return t_suppress;
}

/*******************************************************************************
 * ConstructRocblasltProblem() construct a RocblasltContractionProblem for     *
 * findTopSolutions                                                            *
 *******************************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To>
RocblasltContractionProblem<Ti, To, Tc> ConstructRocblasltProblem(
    const rocblaslt_handle handle, const rocblaslt_matmul_desc matmul_descr,
    rocblaslt_matrix_layout matA, rocblaslt_matrix_layout matB,
    rocblaslt_matrix_layout matC, rocblaslt_matrix_layout matD, const Tc *alpha,
    const Tc *beta, size_t maxWorkSpaceBytes);

/*******************************************************************************
 * getBestSolutions() calls finTopSolutions from Tensile and converts to       *
 * rocblaslt_matmul_heuristic_result                                           *
 *******************************************************************************/
template <typename Ti, typename To = Ti, typename Tc = To>
rocblaslt_status
getBestSolutions(RocblasltContractionProblem<Ti, To, Tc> prob,
                 int requestedAlgoCount,
                 rocblaslt_matmul_heuristic_result heuristicResultsArray[],
                 int *returnAlgoCount, size_t maxWorkSpaceBytes);
