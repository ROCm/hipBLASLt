/* ************************************************************************
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "definitions.h"
#include "handle.h"
#include "rocblaslt_mat_utils.hpp"
#include "tensile_host.hpp"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief
 *******************************************************************************/

rocblaslt_status rocblaslt_matmul_impl(const rocblaslt_handle       handle,
                                       const rocblaslt_matmul_desc  matmul_descr,
                                       const void*                  A,
                                       const void*                  B,
                                       const void*                  C,
                                       void*                        D,
                                       rocblaslt_matrix_layout      matA,
                                       rocblaslt_matrix_layout      matB,
                                       rocblaslt_matrix_layout      matC,
                                       rocblaslt_matrix_layout      matD,
                                       const void*                  alpha,
                                       const void*                  beta,
                                       const rocblaslt_matmul_algo* algo,
                                       void*                        workspace,
                                       size_t                       workspaceSizeInBytes,
                                       hipStream_t                  stream)
{
    int64_t m, n, k, lda, ldb, ldc, ldd, lde, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_stride_d, batch_stride_e;
    hipblasltDatatype_t    bias_type;
    hipblasltDatatype_t    type_a, type_b, type_c, type_d;
    rocblaslt_compute_type compute_type;
    void *                 bias = nullptr, *scaleAlphaVec = nullptr, *E = nullptr;
    bool                   gradient = false;
    rocblaslt_status       isValid  = rocblaslt_matmul_valid_args(matmul_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           D,
                                                           matA,
                                                           matB,
                                                           matC,
                                                           matD,
                                                           alpha,
                                                           beta,
                                                           m,
                                                           n,
                                                           k,
                                                           type_a,
                                                           lda,
                                                           batch_stride_a,
                                                           type_b,
                                                           ldb,
                                                           batch_stride_b,
                                                           type_c,
                                                           ldc,
                                                           batch_stride_c,
                                                           type_d,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleAlphaVec,
                                                           E,
                                                           gradient,
                                                           compute_type);
    if(isValid != rocblaslt_status_continue)
        return isValid;

    // Internal assign
    hipblasOperation_t  opA           = matmul_descr->op_A;
    hipblasOperation_t  opB           = matmul_descr->op_B;
    int                 num_batches_a = matA->batch_count;
    rocblaslt_epilogue  epilogue      = matmul_descr->epilogue;
    void*               scaleA        = matmul_descr->scaleA;
    void*               scaleB        = matmul_descr->scaleB;
    void*               scaleC        = matmul_descr->scaleC;
    void*               scaleD        = matmul_descr->scaleD;
    void*               scaleE        = matmul_descr->scaleE;
    hipblasltDatatype_t scale_type    = matmul_descr->scale_type;

    // Others
    bool strided_batch = true;
    bool grouped_gemm  = false;

    auto& gemmData = matmul_descr->m_data;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        if(matmul_descr->compute_type == rocblaslt_compute_f64)
        {
            *((double*)alpha_1) = 1.f;
            alpha               = alpha_1;
        }
        else if(matmul_descr->compute_type == rocblaslt_compute_i32)
        {
            *((int32_t*)alpha_1) = 1.f;
            alpha                = alpha_1;
        }
        else
        {
            *((float*)alpha_1) = 1.f;
            alpha              = alpha_1;
        }
    }

    // FIXME: Is this still needed?
    // // check alignment of pointers before casting
    // if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
    //    || !isAligned(d, sizeof(To)))
    // {
    //     std::cerr << "memmory is not aligned" << std::endl;
    //     return rocblaslt_status_invalid_size;
    // }

    if(algo)
        workspaceSizeInBytes = min(workspaceSizeInBytes, algo->max_workspace_bytes);
    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        type_a,
                                        A,
                                        nullptr,
                                        lda,
                                        batch_stride_a,
                                        type_b,
                                        B,
                                        nullptr,
                                        ldb,
                                        batch_stride_b,
                                        beta,
                                        type_c,
                                        C,
                                        nullptr,
                                        ldc,
                                        batch_stride_c,
                                        type_d,
                                        D,
                                        nullptr,
                                        ldd,
                                        batch_stride_d,
                                        E,
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
                                        workspace,
                                        workspaceSizeInBytes,
                                        stream,
                                        handle->Synchronizer};
    if (get_logger_layer_mode() & rocblaslt_layer_mode_log_bench) {

// To deal with some arguments may be invalid
#define GEN_BENCH_ARG(_fun, _type_str, _type) \
    strlen(_fun(_type)) && strcmp(_fun(_type), "invalid") ? _type_str : "", \
    _fun(_type)

        log_bench(__func__,
                  "-m",
                  m,
                  "-n",
                  n,
                  "-k",
                  k,
                  "--lda",
                  lda,
                  "--ldb",
                  ldb,
                  "--ldc",
                  ldc,
                  "--ldd",
                  ldd,
                  "--lde",
                  lde,
                  "--stride_a",
                  batch_stride_a,
                  "--stride_b",
                  batch_stride_b,
                  "--stride_c",
                  batch_stride_c,
                  "--stride_d",
                  batch_stride_d,
                  "--stride_e",
                  batch_stride_e,
                  "--alpha",
                  *(float*)alpha,
                  "--beta",
                  *(float*)beta,
                  "--transA",
                  hipblasOperation_to_bench_string(opA),
                  "--transB",
                  hipblasOperation_to_bench_string(opB),
                  "--batch_count",
                  num_batches_a,
                  grouped_gemm ? "--grouped_gemm" : "",
                  scaleA ? "--scaleA" : "",
                  scaleB ? "--scaleB" : "",
                  scaleC ? "--scaleC" : "",
                  scaleD ? "--scaleD" : "",
                  scaleAlphaVec ? "--scaleAlpha_vector" : "",
                  GEN_BENCH_ARG(hipblasltDatatype_to_bench_string, "--a_type", type_a),
                  GEN_BENCH_ARG(hipblasltDatatype_to_bench_string, "--b_type", type_b),
                  GEN_BENCH_ARG(hipblasltDatatype_to_bench_string, "--c_type", type_c),
                  GEN_BENCH_ARG(hipblasltDatatype_to_bench_string, "--d_type", type_d),
                  GEN_BENCH_ARG(rocblaslt_compute_type_to_bench_string, "--compute_type", compute_type),
                  GEN_BENCH_ARG(hipblasltDatatype_to_bench_string, "--scale_type", scale_type),
                  GEN_BENCH_ARG(hipblasltDatatype_to_bench_string, "--bias_type", bias_type),
                  rocblaslt_epilogue_to_bench_string(epilogue));

#undef GEN_BENCH_ARG

    }
    return runContractionProblem(handle, algo, problem, gemmData);
}

rocblaslt_status rocblaslt_gemm_create_cpp_impl(const rocblaslt_handle         handle,
                                                rocblaslt_matmul_desc          matmul_descr,
                                                const void*                    A,
                                                const void*                    B,
                                                const void*                    C,
                                                void*                          D,
                                                rocblaslt_matrix_layout        matA,
                                                rocblaslt_matrix_layout        matB,
                                                rocblaslt_matrix_layout        matC,
                                                rocblaslt_matrix_layout        matD,
                                                const void*                    alpha,
                                                const void*                    beta,
                                                rocblaslt::RocGemmProblemType& problemtype,
                                                std::shared_ptr<void>&         gemmData,
                                                size_t&                        gemmCount)
{
    int64_t m, n, k, lda, ldb, ldc, ldd, lde, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_stride_d, batch_stride_e;
    hipblasltDatatype_t    bias_type;
    hipblasltDatatype_t    type_a, type_b, type_c, type_d;
    rocblaslt_compute_type compute_type;
    void *                 bias = nullptr, *scaleAlphaVec = nullptr, *E = nullptr;
    bool                   gradient = false;
    rocblaslt_status       isValid  = rocblaslt_matmul_valid_args(matmul_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           D,
                                                           matA,
                                                           matB,
                                                           matC,
                                                           matD,
                                                           alpha,
                                                           beta,
                                                           m,
                                                           n,
                                                           k,
                                                           type_a,
                                                           lda,
                                                           batch_stride_a,
                                                           type_b,
                                                           ldb,
                                                           batch_stride_b,
                                                           type_c,
                                                           ldc,
                                                           batch_stride_c,
                                                           type_d,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleAlphaVec,
                                                           E,
                                                           gradient,
                                                           compute_type);
    if(isValid != rocblaslt_status_continue)
        return isValid;

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
    bool strided_batch = true;
    bool grouped_gemm  = false;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        setTo1(matmul_descr->compute_type, (void*)alpha_1, &alpha);
    }

    // // check alignment of pointers before casting
    // if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
    //    || !isAligned(d, sizeof(To)))
    // {
    //     std::cerr << "memmory is not aligned" << std::endl;
    //     return rocblaslt_status_invalid_size;
    // }

    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        type_a,
                                        A,
                                        nullptr,
                                        lda,
                                        batch_stride_a,
                                        type_b,
                                        B,
                                        nullptr,
                                        ldb,
                                        batch_stride_b,
                                        beta,
                                        type_c,
                                        C,
                                        nullptr,
                                        ldc,
                                        batch_stride_c,
                                        type_d,
                                        D,
                                        nullptr,
                                        ldd,
                                        batch_stride_d,
                                        E,
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
                                        0,
                                        0,
                                        handle->Synchronizer};
    return gemmCreate(problem, gemmData, gemmCount);
}

rocblaslt_status rocblaslt_gemm_create_cpp_impl_2(const rocblaslt_handle         handle,
                                                  int64_t                        m,
                                                  int64_t                        n,
                                                  int64_t                        b,
                                                  int64_t                        k,
                                                  int64_t                        lda,
                                                  int64_t                        ldb,
                                                  int64_t                        ldc,
                                                  int64_t                        ldd,
                                                  int64_t                        batch_stride_a,
                                                  int64_t                        batch_stride_b,
                                                  int64_t                        batch_stride_c,
                                                  int64_t                        batch_stride_d,
                                                  rocblaslt::RocGemmEpilogue&    rocEpilogue,
                                                  rocblaslt::RocGemmInputs&      inputs,
                                                  rocblaslt::RocGemmProblemType& problemtype,
                                                  std::shared_ptr<void>&         gemmData,
                                                  size_t&                        gemmCount)
{
    auto status = validateMatmulArgs(m,
                                     n,
                                     k,
                                     inputs.alpha,
                                     inputs.a,
                                     inputs.b,
                                     inputs.beta,
                                     inputs.c,
                                     inputs.d,
                                     b,
                                     b,
                                     b,
                                     b,
                                     batch_stride_a,
                                     batch_stride_b,
                                     batch_stride_c,
                                     batch_stride_d);

    void *              bias = nullptr, *scaleAlphaVec = nullptr, *E = nullptr;
    int64_t             lde = 0, batch_stride_e = 0;
    hipblasltDatatype_t bias_type = static_cast<hipblasltDatatype_t>(0);
    bool                gradient  = false;

    if(status == rocblaslt_status_continue)
        status = rocblaslt_epilogue_valid_args(rocEpilogue.mode,
                                               m,
                                               n,
                                               problemtype.type_d,
                                               rocEpilogue.bias_data_type,
                                               inputs.aux,
                                               rocEpilogue.aux_ld,
                                               rocEpilogue.aux_stride,
                                               inputs.bias,
                                               inputs.scaleAlphaVec,
                                               inputs.alpha,
                                               E,
                                               lde,
                                               batch_stride_e,
                                               bias,
                                               bias_type,
                                               scaleAlphaVec,
                                               gradient);
    if(status != rocblaslt_status_continue)
        return status;

    // Internal assign
    void*                   A             = inputs.a;
    void*                   B             = inputs.b;
    void*                   C             = inputs.c;
    void*                   D             = inputs.d;
    void*                   alpha         = inputs.alpha;
    void*                   beta          = inputs.beta;
    void*                   scaleA        = inputs.scaleA;
    void*                   scaleB        = inputs.scaleB;
    void*                   scaleC        = inputs.scaleC;
    void*                   scaleD        = inputs.scaleD;
    void*                   scaleE        = inputs.scaleE;
    hipblasOperation_t&     opA           = problemtype.op_a;
    hipblasOperation_t&     opB           = problemtype.op_b;
    hipblasltDatatype_t&    type_a        = problemtype.type_a;
    hipblasltDatatype_t&    type_b        = problemtype.type_b;
    hipblasltDatatype_t&    type_c        = problemtype.type_c;
    hipblasltDatatype_t&    type_d        = problemtype.type_d;
    int                     num_batches_a = b;
    rocblaslt_compute_type& compute_type  = problemtype.type_compute;
    rocblaslt_epilogue&     epilogue      = rocEpilogue.mode;

    // Others
    bool strided_batch = true;
    bool grouped_gemm  = false;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        setTo1(compute_type, (void*)alpha_1, (const void**)&alpha);
    }

    // // check alignment of pointers before casting
    // if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
    //    || !isAligned(d, sizeof(To)))
    // {
    //     std::cerr << "memmory is not aligned" << std::endl;
    //     return rocblaslt_status_invalid_size;
    // }

    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        type_a,
                                        A,
                                        nullptr,
                                        lda,
                                        batch_stride_a,
                                        type_b,
                                        B,
                                        nullptr,
                                        ldb,
                                        batch_stride_b,
                                        beta,
                                        type_c,
                                        C,
                                        nullptr,
                                        ldc,
                                        batch_stride_c,
                                        type_d,
                                        D,
                                        nullptr,
                                        ldd,
                                        batch_stride_d,
                                        E,
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
                                        0,
                                        0,
                                        handle->Synchronizer};
    return gemmCreate(problem, gemmData, gemmCount);
}

rocblaslt_status
    rocblaslt_groupedgemm_create_cpp_impl(const rocblaslt_handle                      handle,
                                          std::vector<rocblaslt_matmul_desc>&         matmul_descr,
                                          std::vector<const void*>&                   A,
                                          std::vector<const void*>&                   B,
                                          std::vector<const void*>&                   C,
                                          std::vector<void*>&                         D,
                                          std::vector<rocblaslt_matrix_layout>&       matA,
                                          std::vector<rocblaslt_matrix_layout>&       matB,
                                          std::vector<rocblaslt_matrix_layout>&       matC,
                                          std::vector<rocblaslt_matrix_layout>&       matD,
                                          std::vector<const void*>&                   alpha,
                                          std::vector<const void*>&                   beta,
                                          std::vector<rocblaslt::RocGemmProblemType>& problemtype,
                                          std::shared_ptr<void>&                      gemmData,
                                          size_t&                                     gemmCount)
{
    hipblasOperation_t     opA          = matmul_descr[0]->op_A;
    hipblasOperation_t     opB          = matmul_descr[0]->op_B;
    rocblaslt_compute_type compute_type = matmul_descr[0]->compute_type;
    hipblasltDatatype_t    type_a       = matA[0]->type;
    hipblasltDatatype_t    type_b       = matB[0]->type;
    hipblasltDatatype_t    type_c       = matC[0]->type;
    hipblasltDatatype_t    type_d       = matD[0]->type;

    std::vector<const void*>         A_vec, B_vec, C_vec, alpha_vec, beta_vec;
    std::vector<void*>               D_vec, E_vec;
    std::vector<const void*>         bias_vec;
    std::vector<const void*>         scaleA_vec;
    std::vector<const void*>         scaleB_vec;
    std::vector<const void*>         scaleC_vec;
    std::vector<const void*>         scaleD_vec;
    std::vector<const void*>         scaleE_vec;
    std::vector<const void*>         scaleAlpha_vec;
    std::vector<hipblasltDatatype_t> bias_type_vec;
    std::vector<rocblaslt_epilogue>  epilogue_vec;
    std::vector<int64_t>             m_vec, n_vec, k_vec;
    std::vector<int64_t>             lda_vec, batch_stride_a_vec, num_batches_a_vec;
    std::vector<int64_t>             ldb_vec, batch_stride_b_vec, num_batches_b_vec;
    std::vector<int64_t>             ldc_vec, batch_stride_c_vec, num_batches_c_vec;
    std::vector<int64_t>             ldd_vec, batch_stride_d_vec, num_batches_d_vec;
    std::vector<int64_t>             lde_vec, batch_stride_e_vec, num_batches_e_vec;
    std::vector<int8_t[16]>          alpha_1(matmul_descr.size());

    std::vector<bool> gradient_vec;

    std::vector<rocblaslt::RocGemmProblemType> tempprobemtype;
    for(int i = 0; i < matmul_descr.size(); i++)
    {
        // matrix A
        int64_t num_rows_a     = matA[i]->m;
        int64_t num_cols_a     = matA[i]->n;
        int64_t lda            = matA[i]->ld;
        int64_t batch_stride_a = matA[i]->batch_stride;
        int     num_batches_a  = matA[i]->batch_count;

        // matrix B
        // int64_t num_rows_b = matB[i]->m;
        // int64_t num_cols_b = matB[i]->n;
        int64_t ldb            = matB[i]->ld;
        int64_t batch_stride_b = matB[i]->batch_stride;
        int     num_batches_b  = matB[i]->batch_count;

        // matrix C
        // int64_t num_rows_c = matC[i]->m;
        // int64_t num_cols_c = matC[i]->n;
        int64_t ldc            = matC[i]->ld;
        int64_t batch_stride_c = matC[i]->batch_stride;
        int     num_batches_c  = matC[i]->batch_count;

        // matrix D
        int64_t num_rows_d     = matD[i]->m;
        int64_t num_cols_d     = matD[i]->n;
        int64_t ldd            = matD[i]->ld;
        int64_t batch_stride_d = matD[i]->batch_stride;
        int     num_batches_d  = matD[i]->batch_count;

        int64_t m = num_rows_d;
        int64_t n = num_cols_d;
        int64_t k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

        auto validArgs = validateMatmulArgs(m,
                                            n,
                                            k,
                                            alpha[i],
                                            A[i],
                                            B[i],
                                            beta[i],
                                            C[i],
                                            D[i],
                                            num_batches_a,
                                            num_batches_b,
                                            num_batches_c,
                                            num_batches_d,
                                            batch_stride_a,
                                            batch_stride_b,
                                            batch_stride_c,
                                            batch_stride_d);
        if(validArgs == rocblaslt_status_success)
            continue;

        void*               bias = nullptr;
        hipblasltDatatype_t bias_type;
        void*               scaleAlphaVec = nullptr;
        void*               E             = nullptr;
        int64_t             lde, batch_stride_e;
        bool                gradient;
        rocblaslt_epilogue  epilogue    = matmul_descr[i]->epilogue;
        const void*         alphaVecPtr = matmul_descr[i]->pointermode ? alpha[i] : nullptr;
        if(validArgs == rocblaslt_status_continue)
            validArgs = rocblaslt_epilogue_valid_args(epilogue, // add alpha
                                                      num_rows_d,
                                                      num_cols_d,
                                                      matD[i]->type,
                                                      matmul_descr[i]->bias_type,
                                                      matmul_descr[i]->e,
                                                      matmul_descr[i]->lde,
                                                      matmul_descr[i]->stride_e,
                                                      matmul_descr[i]->bias,
                                                      alphaVecPtr,
                                                      alpha[i],
                                                      E,
                                                      lde,
                                                      batch_stride_e,
                                                      bias,
                                                      bias_type,
                                                      scaleAlphaVec,
                                                      gradient);
        if(validArgs != rocblaslt_status_continue)
            return validArgs;

        const void* alphaTmp = nullptr;
        memset(alpha_1[i], 0, sizeof(int8_t) * 16);
        if(scaleAlphaVec)
        {
            setTo1(compute_type, (void*)alpha_1[i], &alphaTmp);
        }
        else
        {
            alphaTmp = alpha[i];
        }

        tempprobemtype.push_back({matmul_descr[i]->op_A,
                                  matmul_descr[i]->op_B,
                                  matA[i]->type,
                                  matB[i]->type,
                                  matC[i]->type,
                                  matD[i]->type,
                                  matmul_descr[i]->compute_type});

        bias_type_vec.push_back(bias_type);
        epilogue_vec.push_back(epilogue);
        bias_vec.push_back(bias);
        scaleA_vec.push_back(matmul_descr[i]->scaleA);
        scaleB_vec.push_back(matmul_descr[i]->scaleB);
        scaleC_vec.push_back(matmul_descr[i]->scaleC);
        scaleD_vec.push_back(matmul_descr[i]->scaleD);
        scaleE_vec.push_back(matmul_descr[i]->scaleE);
        scaleAlpha_vec.push_back(scaleAlphaVec);

        // matrix A
        // int64_t           num_rows_a     = matA[i]->m;
        // int64_t           num_cols_a     = matA[i]->n;
        lda_vec.push_back(lda);
        batch_stride_a_vec.push_back(batch_stride_a);
        num_batches_a_vec.push_back(num_batches_a);

        // matrix B
        ldb_vec.push_back(ldb);
        batch_stride_b_vec.push_back(batch_stride_b);
        num_batches_b_vec.push_back(num_batches_b);

        // matrix C
        ldc_vec.push_back(ldc);
        batch_stride_c_vec.push_back(batch_stride_c);
        num_batches_c_vec.push_back(num_batches_c);

        // matrix D
        // int64_t           num_rows_d     = matD[i]->m;
        // int64_t           num_cols_d     = matD[i]->n;
        ldd_vec.push_back(ldd);
        batch_stride_d_vec.push_back(batch_stride_d);
        num_batches_d_vec.push_back(num_batches_d);

        lde_vec.push_back(lde);
        batch_stride_e_vec.push_back(batch_stride_e);
        num_batches_e_vec.push_back(num_batches_d);

        m_vec.push_back(num_rows_d);
        n_vec.push_back(num_cols_d);
        k_vec.push_back((opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a);

        A_vec.push_back(A[i]);
        B_vec.push_back(B[i]);
        C_vec.push_back(C[i]);
        D_vec.push_back(D[i]);
        E_vec.push_back(E);
        alpha_vec.push_back(alphaTmp);
        beta_vec.push_back(beta[i]);

        gradient_vec.push_back(gradient);
    }

    problemtype = tempprobemtype;

    bool strided_batch = true;
    bool grouped_gemm  = true;

    std::vector<RocblasltContractionProblem> problems;
    for(int i = 0; i < m_vec.size(); i++)
    {
        problems.push_back(RocblasltContractionProblem{opA,
                                                       opB,
                                                       m_vec[i],
                                                       n_vec[i],
                                                       k_vec[i],
                                                       alpha_vec[i],
                                                       type_a,
                                                       A_vec[i],
                                                       nullptr,
                                                       lda_vec[i],
                                                       batch_stride_a_vec[i],
                                                       type_b,
                                                       B_vec[i],
                                                       nullptr,
                                                       ldb_vec[i],
                                                       batch_stride_b_vec[i],
                                                       beta_vec[i],
                                                       type_c,
                                                       C_vec[i],
                                                       nullptr,
                                                       ldc_vec[i],
                                                       batch_stride_c_vec[i],
                                                       type_d,
                                                       D_vec[i],
                                                       nullptr,
                                                       ldd_vec[i],
                                                       batch_stride_d_vec[i],
                                                       E_vec[i],
                                                       nullptr,
                                                       lde_vec[i],
                                                       batch_stride_e_vec[i],
                                                       num_batches_a_vec[i],
                                                       strided_batch,
                                                       grouped_gemm,
                                                       gradient_vec[i],
                                                       compute_type,
                                                       bias_vec[i],
                                                       scaleA_vec[i],
                                                       scaleB_vec[i],
                                                       scaleC_vec[i],
                                                       scaleD_vec[i],
                                                       scaleE_vec[i],
                                                       scaleAlpha_vec[i],
                                                       bias_type_vec[i],
                                                       epilogue_vec[i],
                                                       nullptr,
                                                       0,
                                                       0,
                                                       handle->Synchronizer});
    }
    return groupedGemmCreate(problems, gemmData, gemmCount);
}

rocblaslt_status
    rocblaslt_groupedgemm_create_cpp_impl_2(const rocblaslt_handle                      handle,
                                            std::vector<int64_t>&                       m,
                                            std::vector<int64_t>&                       n,
                                            std::vector<int64_t>&                       b,
                                            std::vector<int64_t>&                       k,
                                            std::vector<int64_t>&                       lda,
                                            std::vector<int64_t>&                       ldb,
                                            std::vector<int64_t>&                       ldc,
                                            std::vector<int64_t>&                       ldd,
                                            std::vector<int64_t>&                       strideA,
                                            std::vector<int64_t>&                       strideB,
                                            std::vector<int64_t>&                       strideC,
                                            std::vector<int64_t>&                       strideD,
                                            std::vector<rocblaslt::RocGemmEpilogue>&    rocEpilogue,
                                            std::vector<rocblaslt::RocGemmInputs>&      inputs,
                                            std::vector<rocblaslt::RocGemmProblemType>& problemtype,
                                            std::shared_ptr<void>&                      gemmData,
                                            size_t&                                     gemmCount)
{
    hipblasOperation_t     opA          = problemtype[0].op_a;
    hipblasOperation_t     opB          = problemtype[0].op_b;
    rocblaslt_compute_type compute_type = problemtype[0].type_compute;
    hipblasltDatatype_t    type_a       = problemtype[0].type_a;
    hipblasltDatatype_t    type_b       = problemtype[0].type_b;
    hipblasltDatatype_t    type_c       = problemtype[0].type_c;
    hipblasltDatatype_t    type_d       = problemtype[0].type_d;

    std::vector<const void*>         A_vec, B_vec, C_vec, alpha_vec, beta_vec;
    std::vector<void*>               D_vec, E_vec;
    std::vector<const void*>         bias_vec;
    std::vector<const void*>         scaleA_vec;
    std::vector<const void*>         scaleB_vec;
    std::vector<const void*>         scaleC_vec;
    std::vector<const void*>         scaleD_vec;
    std::vector<const void*>         scaleE_vec;
    std::vector<const void*>         scaleAlpha_vec;
    std::vector<hipblasltDatatype_t> bias_type_vec;
    std::vector<rocblaslt_epilogue>  epilogue_vec;

    std::vector<int64_t> lde_vec, batch_stride_e_vec, num_batches_e_vec;
    std::vector<bool>    gradient_vec;

    std::vector<int8_t[16]> alpha_1(m.size());

    for(int i = 0; i < m.size(); i++)
    {
        auto validArgs = validateMatmulArgs(m[i],
                                            n[i],
                                            k[i],
                                            inputs[i].alpha,
                                            inputs[i].a,
                                            inputs[i].b,
                                            inputs[i].beta,
                                            inputs[i].c,
                                            inputs[i].d,
                                            b[i],
                                            b[i],
                                            b[i],
                                            b[i],
                                            strideA[i],
                                            strideB[i],
                                            strideC[i],
                                            strideD[i]);
        if(validArgs == rocblaslt_status_success)
            continue;

        void*               bias = nullptr;
        hipblasltDatatype_t bias_type;
        void*               scaleAlphaVec = nullptr;
        void*               E             = nullptr;
        int64_t             lde, batch_stride_e;
        bool                gradient;

        int                iIdx     = (rocEpilogue.size() <= i) ? rocEpilogue.size() - 1 : i;
        int                iIdx2    = (problemtype.size() <= i) ? problemtype.size() - 1 : i;
        rocblaslt_epilogue epilogue = rocEpilogue[iIdx].mode;
        if(validArgs == rocblaslt_status_continue)
            validArgs = rocblaslt_epilogue_valid_args(epilogue,
                                                      m[i],
                                                      n[i],
                                                      problemtype[iIdx2].type_d,
                                                      rocEpilogue[iIdx].bias_data_type,
                                                      inputs[i].aux,
                                                      rocEpilogue[iIdx].aux_ld,
                                                      rocEpilogue[iIdx].aux_stride,
                                                      inputs[i].bias,
                                                      inputs[i].scaleAlphaVec,
                                                      inputs[i].alpha,
                                                      E,
                                                      lde,
                                                      batch_stride_e,
                                                      bias,
                                                      bias_type,
                                                      scaleAlphaVec,
                                                      gradient);
        if(validArgs != rocblaslt_status_continue)
            return validArgs;

        const void* alphaTmp = nullptr;
        memset(alpha_1[i], 0, sizeof(int8_t) * 16);
        if(scaleAlphaVec)
        {
            setTo1(compute_type, (void*)alpha_1[i], &alphaTmp);
        }
        else
        {
            alphaTmp = inputs[i].alpha;
        }

        bias_type_vec.push_back(bias_type);
        epilogue_vec.push_back(epilogue);
        bias_vec.push_back(bias);
        scaleA_vec.push_back(inputs[i].scaleA);
        scaleB_vec.push_back(inputs[i].scaleB);
        scaleC_vec.push_back(inputs[i].scaleC);
        scaleD_vec.push_back(inputs[i].scaleD);
        scaleE_vec.push_back(inputs[i].scaleE);
        scaleAlpha_vec.push_back(scaleAlphaVec);

        A_vec.push_back(inputs[i].a);
        B_vec.push_back(inputs[i].b);
        C_vec.push_back(inputs[i].c);
        D_vec.push_back(inputs[i].d);
        E_vec.push_back(E);
        alpha_vec.push_back(alphaTmp);
        beta_vec.push_back(inputs[i].beta);

        lde_vec.push_back(lde);
        batch_stride_e_vec.push_back(batch_stride_e);
        gradient_vec.push_back(gradient);
    }

    bool strided_batch = true;
    bool grouped_gemm  = true;

    std::vector<RocblasltContractionProblem> problems;
    for(int i = 0; i < m.size(); i++)
    {
        problems.push_back(RocblasltContractionProblem{opA,
                                                       opB,
                                                       m[i],
                                                       n[i],
                                                       k[i],
                                                       alpha_vec[i],
                                                       type_a,
                                                       A_vec[i],
                                                       nullptr,
                                                       lda[i],
                                                       strideA[i],
                                                       type_b,
                                                       B_vec[i],
                                                       nullptr,
                                                       ldb[i],
                                                       strideB[i],
                                                       beta_vec[i],
                                                       type_c,
                                                       C_vec[i],
                                                       nullptr,
                                                       ldc[i],
                                                       strideC[i],
                                                       type_d,
                                                       D_vec[i],
                                                       nullptr,
                                                       ldd[i],
                                                       strideD[i],
                                                       E_vec[i],
                                                       nullptr,
                                                       lde_vec[i],
                                                       batch_stride_e_vec[i],
                                                       b[i],
                                                       strided_batch,
                                                       grouped_gemm,
                                                       gradient_vec[i],
                                                       compute_type,
                                                       bias_vec[i],
                                                       scaleA_vec[i],
                                                       scaleB_vec[i],
                                                       scaleC_vec[i],
                                                       scaleD_vec[i],
                                                       scaleE_vec[i],
                                                       scaleAlpha_vec[i],
                                                       bias_type_vec[i],
                                                       epilogue_vec[i],
                                                       nullptr,
                                                       0,
                                                       0,
                                                       handle->Synchronizer});
    }
    return groupedGemmCreate(problems, gemmData, gemmCount);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul(rocblaslt_handle             handle,
                                  rocblaslt_matmul_desc        matmul_descr,
                                  const void*                  alpha,
                                  const void*                  A,
                                  rocblaslt_matrix_layout      matA,
                                  const void*                  B,
                                  rocblaslt_matrix_layout      matB,
                                  const void*                  beta,
                                  const void*                  C,
                                  rocblaslt_matrix_layout      matC,
                                  void*                        D,
                                  rocblaslt_matrix_layout      matD,
                                  const rocblaslt_matmul_algo* algo,
                                  void*                        workspace,
                                  size_t                       workspaceSizeInBytes,
                                  hipStream_t                  stream)

{
    // Check if handle is valid
    if(handle == nullptr || matmul_descr == nullptr || matA == nullptr || matB == nullptr
       || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return rocblaslt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr
       || D == nullptr)
    {
        log_error(__func__, "invalid data pointer");
        return rocblaslt_status_invalid_pointer;
    }
    if(workspace == nullptr && workspaceSizeInBytes > 0)
    {
        log_error(__func__, "invalid workspace pointer");
        return rocblaslt_status_invalid_pointer;
    }

    if(matC->type != matD->type)
    {
        log_error(__func__, "invalid  matrix datatype");
        return rocblaslt_status_type_mismatch;
    }

    if(get_logger_layer_mode() != rocblaslt_layer_mode_none)
    {
        log_api(__func__,
                "A",
                A,
                "Adesc",
                matA,
                "B",
                B,
                "Bdesc",
                matB,
                "C",
                C,
                "Cdesc",
                matC,
                "D",
                D,
                "Ddesc",
                matD,
                "computeDesc",
                matmul_descr,
                "algo",
                algo,
                "workSpace",
                workspace,
                "workSpaceSizeInBytes",
                workspaceSizeInBytes,
                "stream",
                stream);
    }

    if(get_logger_layer_mode() != rocblaslt_layer_mode_none)
    {
        log_trace(__func__,
                  "A",
                  A,
                  "Adesc",
                  rocblaslt_matrix_layout_to_string(matA),
                  "B",
                  B,
                  "Bdesc",
                  rocblaslt_matrix_layout_to_string(matB),
                  "C",
                  C,
                  "Cdesc",
                  rocblaslt_matrix_layout_to_string(matC),
                  "D",
                  D,
                  "Ddesc",
                  rocblaslt_matrix_layout_to_string(matD),
                  "computeDesc",
                  rocblaslt_matmul_desc_to_string(matmul_descr),
                  "workSpace",
                  workspace,
                  "workSpaceSizeInBytes",
                  workspaceSizeInBytes,
                  (matmul_descr->pointermode) ? "alphaVector" : "alpha",
                  *(reinterpret_cast<const float*>(alpha)),
                  "beta",
                  *(reinterpret_cast<const float*>(beta)),
                  "stream",
                  stream);
    }
    return rocblaslt_matmul_impl(handle,
                                 matmul_descr,
                                 A,
                                 B,
                                 C,
                                 D,
                                 matA,
                                 matB,
                                 matC,
                                 matD,
                                 alpha,
                                 beta,
                                 algo,
                                 workspace,
                                 workspaceSizeInBytes,
                                 stream);
}
#ifdef __cplusplus
}
#endif

rocblaslt_status rocblaslt_gemm_create_cpp(const rocblaslt_handle         handle,
                                           int64_t                        m,
                                           int64_t                        n,
                                           int64_t                        b,
                                           int64_t                        k,
                                           int64_t                        lda,
                                           int64_t                        ldb,
                                           int64_t                        ldc,
                                           int64_t                        ldd,
                                           int64_t                        strideA,
                                           int64_t                        strideB,
                                           int64_t                        strideC,
                                           int64_t                        strideD,
                                           rocblaslt::RocGemmEpilogue&    epilogue,
                                           rocblaslt::RocGemmInputs&      inputs,
                                           rocblaslt::RocGemmProblemType& problemtype,
                                           std::shared_ptr<void>&         gemmData,
                                           size_t&                        gemmCount)
{
    return rocblaslt_gemm_create_cpp_impl_2(handle,
                                            m,
                                            n,
                                            b,
                                            k,
                                            lda,
                                            ldb,
                                            ldc,
                                            ldd,
                                            strideA,
                                            strideB,
                                            strideC,
                                            strideD,
                                            epilogue,
                                            inputs,
                                            problemtype,
                                            gemmData,
                                            gemmCount);
}

rocblaslt_status rocblaslt_gemm_create_cpp(const rocblaslt_handle         handle,
                                           rocblaslt_matmul_desc          matmul_descr,
                                           const void*                    alpha,
                                           const void*                    A,
                                           rocblaslt_matrix_layout        matA,
                                           const void*                    B,
                                           rocblaslt_matrix_layout        matB,
                                           const void*                    beta,
                                           const void*                    C,
                                           rocblaslt_matrix_layout        matC,
                                           void*                          D,
                                           rocblaslt_matrix_layout        matD,
                                           rocblaslt::RocGemmProblemType& problemtype,
                                           std::shared_ptr<void>&         gemmData,
                                           size_t&                        gemmCount)
{
    // Check if handle is valid
    if(matmul_descr == nullptr || matA == nullptr || matB == nullptr || matC == nullptr
       || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return rocblaslt_status_invalid_handle;
    }

    if(matA->type != matB->type || matC->type != matD->type)
    {
        log_error(__func__, "invalid matrix datatype");
        return rocblaslt_status_type_mismatch;
    }

    return rocblaslt_gemm_create_cpp_impl(handle,
                                          matmul_descr,
                                          A,
                                          B,
                                          C,
                                          D,
                                          matA,
                                          matB,
                                          matC,
                                          matD,
                                          alpha,
                                          beta,
                                          problemtype,
                                          gemmData,
                                          gemmCount);
}

rocblaslt_status
    rocblaslt_groupedgemm_create_cpp(const rocblaslt_handle                      handle,
                                     std::vector<int64_t>&                       m,
                                     std::vector<int64_t>&                       n,
                                     std::vector<int64_t>&                       b,
                                     std::vector<int64_t>&                       k,
                                     std::vector<int64_t>&                       lda,
                                     std::vector<int64_t>&                       ldb,
                                     std::vector<int64_t>&                       ldc,
                                     std::vector<int64_t>&                       ldd,
                                     std::vector<int64_t>&                       strideA,
                                     std::vector<int64_t>&                       strideB,
                                     std::vector<int64_t>&                       strideC,
                                     std::vector<int64_t>&                       strideD,
                                     std::vector<rocblaslt::RocGemmEpilogue>&    epilogue,
                                     std::vector<rocblaslt::RocGemmInputs>&      inputs,
                                     std::vector<rocblaslt::RocGemmProblemType>& problemtype,
                                     std::shared_ptr<void>&                      gemmData,
                                     size_t&                                     gemmCount)
{
    if(problemtype.size() != 1)
    {
        log_error(__func__, "Currently only supports same problem type for grouped gemm.");
        return rocblaslt_status_invalid_value;
    }
    return rocblaslt_groupedgemm_create_cpp_impl_2(handle,
                                                   m,
                                                   n,
                                                   b,
                                                   k,
                                                   lda,
                                                   ldb,
                                                   ldc,
                                                   ldd,
                                                   strideA,
                                                   strideB,
                                                   strideC,
                                                   strideD,
                                                   epilogue,
                                                   inputs,
                                                   problemtype,
                                                   gemmData,
                                                   gemmCount);
}

rocblaslt_status
    rocblaslt_groupedgemm_create_cpp(const rocblaslt_handle                      handle,
                                     std::vector<rocblaslt_matmul_desc>&         matmul_descr,
                                     std::vector<const void*>&                   alpha,
                                     std::vector<const void*>&                   A,
                                     std::vector<rocblaslt_matrix_layout>&       matA,
                                     std::vector<const void*>&                   B,
                                     std::vector<rocblaslt_matrix_layout>&       matB,
                                     std::vector<const void*>&                   beta,
                                     std::vector<const void*>&                   C,
                                     std::vector<rocblaslt_matrix_layout>&       matC,
                                     std::vector<void*>&                         D,
                                     std::vector<rocblaslt_matrix_layout>&       matD,
                                     std::vector<rocblaslt::RocGemmProblemType>& problemtype,
                                     std::shared_ptr<void>&                      gemmData,
                                     size_t&                                     gemmCount)
{
    for(int i = 0; i < matmul_descr.size(); i++)
    {
        // Check if handle is valid
        if(matmul_descr[i] == nullptr || matA[i] == nullptr || matB[i] == nullptr
           || matC[i] == nullptr || matD[i] == nullptr)
        {
            log_error(__func__, "invalid handle pointer");
            return rocblaslt_status_invalid_handle;
        }

        if(matA[i]->type != matB[i]->type || matC[i]->type != matD[i]->type
           || matA[0]->type != matA[i]->type || matC[0]->type != matC[i]->type)
        {
            log_error(__func__, "invalid  matrix datatype");
            return rocblaslt_status_type_mismatch;
        }
    }

    return rocblaslt_groupedgemm_create_cpp_impl(handle,
                                                 matmul_descr,
                                                 A,
                                                 B,
                                                 C,
                                                 D,
                                                 matA,
                                                 matB,
                                                 matC,
                                                 matD,
                                                 alpha,
                                                 beta,
                                                 problemtype,
                                                 gemmData,
                                                 gemmCount);
}

rocblaslt_status rocblaslt_run_cpp(rocblaslt_handle       handle,
                                   rocblaslt::RocGemmType gemmType,
                                   std::shared_ptr<void>  gemmData,
                                   hipStream_t            stream)
{
    return runKernelFromInvocation(handle, gemmType, gemmData, stream);
}

rocblaslt_status rocblaslt_run_user_args_cpp(rocblaslt_handle       handle,
                                             rocblaslt::RocGemmType gemmType,
                                             std::shared_ptr<void>  gemmData,
                                             void*                  deviceUserArgs,
                                             hipStream_t            stream)
{
    return runKernelFromNewDeviceUserArguments(handle, gemmType, gemmData, deviceUserArgs, stream);
}

rocblaslt_status rocblaslt_run_user_args_cpp(rocblaslt_handle             handle,
                                             rocblaslt::RocGemmType       gemmType,
                                             size_t                       gemmCount,
                                             std::shared_ptr<void>        gemmData,
                                             const rocblaslt_matmul_algo& algo,
                                             void*                        deviceUserArgs,
                                             void*                        workspace,
                                             hipStream_t                  stream)
{
    return runKernelFromDeviceUserArguments(
        handle, gemmType, gemmCount, gemmData, algo, deviceUserArgs, workspace, stream);
}

rocblaslt_status rocblaslt_get_default_user_args(rocblaslt_handle       handle,
                                                 rocblaslt::RocGemmType gemmType,
                                                 std::shared_ptr<void>  gemmData,
                                                 void*                  hostDeviceUserArgs)
{
    return getDeviceUserArgumentsValuesFromContractionProblem(
        handle, gemmType, gemmData, hostDeviceUserArgs);
}

rocblaslt_status rocblaslt_makeArgument_cpp(rocblaslt_handle             handle,
                                            const rocblaslt::RocGemmType gemmType,
                                            const rocblaslt_matmul_algo& algo,
                                            const rocblaslt::RocTuning*  tuning,
                                            void*                        workspace,
                                            bool                         useUserArgs,
                                            hipStream_t                  stream,
                                            std::shared_ptr<void>        gemmData)
{
    return makeArgument(handle, gemmType, algo, tuning, workspace, useUserArgs, stream, gemmData);
}

std::string rocblaslt_get_kernel_name_from_algo(rocblaslt_handle             handle,
                                                const rocblaslt_matmul_algo& algo)
{
    return getKernelNameFromAlgoIndex(handle, algo);
}

std::string rocblaslt_get_solution_name_from_algo(rocblaslt_handle             handle,
                                                  const rocblaslt_matmul_algo& algo)
{
    return getSolutionNameFromAlgoIndex(handle, algo);
}
