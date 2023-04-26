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

#include "rocblaslt_mat.hpp"
#include "definitions.h"
#include "handle.h"
#include "rocblaslt_mat_utils.hpp"

#include <hip/hip_runtime_api.h>

inline rocblaslt_status rocblaslt_matmul_valid_args(const rocblaslt_matmul_desc matmul_descr,
                                                    const void*                 A,
                                                    const void*                 B,
                                                    const void*                 C,
                                                    const void*                 D,
                                                    rocblaslt_matrix_layout     matA,
                                                    rocblaslt_matrix_layout     matB,
                                                    rocblaslt_matrix_layout     matC,
                                                    rocblaslt_matrix_layout     matD,
                                                    const void*                 alpha,
                                                    const void*                 beta,
                                                    int64_t&                    m,
                                                    int64_t&                    n,
                                                    int64_t&                    k,
                                                    int64_t&                    lda,
                                                    int64_t&                    batch_stride_a,
                                                    int64_t&                    ldb,
                                                    int64_t&                    batch_stride_b,
                                                    int64_t&                    ldc,
                                                    int64_t&                    batch_stride_c,
                                                    int64_t&                    ldd,
                                                    int64_t&                    batch_stride_d,
                                                    int64_t&                    lde,
                                                    int64_t&                    batch_stride_e,
                                                    void*&                      bias,
                                                    hipblasDatatype_t&          bias_type,
                                                    void*&                      scaleD,
                                                    void*&                      E,
                                                    bool&                       gradient)
{
    // Set status
    rocblaslt_status status = rocblaslt_status_continue;
    // Internal assign
    hipblasOperation_t opA      = matmul_descr->op_A;
    rocblaslt_epilogue epilogue = matmul_descr->epilogue;

    // External update args
    bias_type = matmul_descr->bias_type == 0 ? matD->type : matmul_descr->bias_type;
    gradient  = is_grad_enabled(epilogue);

    if(is_bias_enabled(epilogue))
        bias = matmul_descr->bias;
    else
        bias = nullptr;

    if(matmul_descr->scaleD)
        scaleD = matmul_descr->scaleD;
    else
        scaleD = nullptr;

    // matrix A
    int64_t num_rows_a    = matA->m;
    int64_t num_cols_a    = matA->n;
    int     num_batches_a = matA->batch_count;
    lda                   = matA->ld;
    batch_stride_a        = matA->batch_stride;

    // matrix B
    int num_batches_b = matB->batch_count;
    ldb               = matB->ld;
    batch_stride_b    = matB->batch_stride;

    // matrix C
    int num_batches_c = matC->batch_count;
    ldc               = matC->ld;
    batch_stride_c    = matC->batch_stride;

    // matrix D
    int64_t num_rows_d    = matD->m;
    int64_t num_cols_d    = matD->n;
    int     num_batches_d = matD->batch_count;
    ldd                   = matD->ld;
    batch_stride_d        = matD->batch_stride;

    // matrix E
    E = nullptr;
    if(is_e_enabled(epilogue))
    {
        if(matmul_descr->e == nullptr)
            status = rocblaslt_status_invalid_pointer;
        E = (void*)matmul_descr->e;
    }
    int64_t num_rows_e = num_rows_d;
    int64_t num_cols_e = num_cols_d;
    lde                = matmul_descr->lde > 0 ? matmul_descr->lde : num_rows_e;
    batch_stride_e     = matmul_descr->stride_e > 0 ? matmul_descr->stride_e : lde * num_cols_e;
    if(E != nullptr
       && ((matmul_descr->lde < num_rows_e)
           || (matmul_descr->stride_e < (num_cols_e * num_rows_e))))
        status = rocblaslt_status_invalid_value;

    m = num_rows_d;
    n = num_cols_d;
    k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

    if(status == rocblaslt_status_continue)
        status = validateMatmulArgs(m,
                                    n,
                                    k,
                                    alpha,
                                    A,
                                    B,
                                    beta,
                                    C,
                                    D,
                                    num_batches_a,
                                    num_batches_b,
                                    num_batches_c,
                                    num_batches_d,
                                    batch_stride_a,
                                    batch_stride_b,
                                    batch_stride_c,
                                    batch_stride_d);
    return status;
}

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
    hipblasDatatype_t bias_type;
    void *            bias = nullptr, *scaleD = nullptr, *E = nullptr;
    bool              gradient = false;
    rocblaslt_status  isValid  = rocblaslt_matmul_valid_args(matmul_descr,
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
                                                           lda,
                                                           batch_stride_a,
                                                           ldb,
                                                           batch_stride_b,
                                                           ldc,
                                                           batch_stride_c,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleD,
                                                           E,
                                                           gradient);
    if(isValid != rocblaslt_status_continue)
        return isValid;

    // Internal assign
    hipblasOperation_t     opA           = matmul_descr->op_A;
    hipblasOperation_t     opB           = matmul_descr->op_B;
    hipblasDatatype_t      type_a        = matA->type;
    hipblasDatatype_t      type_b        = matB->type;
    hipblasDatatype_t      type_c        = matC->type;
    hipblasDatatype_t      type_d        = matD->type;
    int                    num_batches_a = matA->batch_count;
    rocblaslt_compute_type compute_type  = matmul_descr->compute_type;
    rocblaslt_epilogue     epilogue      = matmul_descr->epilogue;

    // Others
    bool strided_batch = true;
    bool grouped_gemm  = false;

#define EX_PARM                                                                                  \
    handle, opA, opB, m, n, k, alpha, A, type_a, lda, batch_stride_a, B, type_b, ldb,            \
        batch_stride_b, beta, C, type_c, ldc, batch_stride_c, D, type_d, ldd, batch_stride_d, E, \
        lde, batch_stride_e, num_batches_a, strided_batch, grouped_gemm, gradient, compute_type, \
        algo, workspace, workspaceSizeInBytes, bias, scaleD, bias_type, epilogue, stream

    return rocblaslt_matmul_template(EX_PARM);
}

rocblaslt_status rocblaslt_gemm_create_impl(rocblaslt_gemm          gemm,
                                            rocblaslt_matmul_desc   matmul_descr,
                                            const void*             A,
                                            const void*             B,
                                            const void*             C,
                                            void*                   D,
                                            rocblaslt_matrix_layout matA,
                                            rocblaslt_matrix_layout matB,
                                            rocblaslt_matrix_layout matC,
                                            rocblaslt_matrix_layout matD,
                                            const void*             alpha,
                                            const void*             beta)
{
    int64_t m, n, k, lda, ldb, ldc, ldd, lde, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_stride_d, batch_stride_e;
    hipblasDatatype_t bias_type;
    void *            bias = nullptr, *scaleD = nullptr, *E = nullptr;
    bool              gradient = false;
    rocblaslt_status  isValid  = rocblaslt_matmul_valid_args(matmul_descr,
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
                                                           lda,
                                                           batch_stride_a,
                                                           ldb,
                                                           batch_stride_b,
                                                           ldc,
                                                           batch_stride_c,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleD,
                                                           E,
                                                           gradient);
    if(isValid != rocblaslt_status_continue)
        return isValid;

    // Internal assign
    hipblasOperation_t     opA           = matmul_descr->op_A;
    hipblasOperation_t     opB           = matmul_descr->op_B;
    hipblasDatatype_t      type_a        = matA->type;
    hipblasDatatype_t      type_b        = matB->type;
    hipblasDatatype_t      type_c        = matC->type;
    hipblasDatatype_t      type_d        = matD->type;
    int                    num_batches_a = matA->batch_count;
    rocblaslt_compute_type compute_type  = matmul_descr->compute_type;
    rocblaslt_epilogue     epilogue      = matmul_descr->epilogue;

    // Others
    bool strided_batch = true;
    bool grouped_gemm  = false;

#define EX_PARM_GEMM                                                                             \
    gemm, opA, opB, m, n, k, alpha, A, type_a, lda, batch_stride_a, B, type_b, ldb,              \
        batch_stride_b, beta, C, type_c, ldc, batch_stride_c, D, type_d, ldd, batch_stride_d, E, \
        lde, batch_stride_e, num_batches_a, strided_batch, grouped_gemm, gradient, compute_type, \
        bias, scaleD, bias_type, epilogue

    return rocblaslt_gemm_create_template(EX_PARM_GEMM);
}

rocblaslt_status rocblaslt_groupedgemm_create_impl(rocblaslt_gemm                      groupedgemm,
                                                   std::vector<rocblaslt_matmul_desc>& matmul_descr,
                                                   std::vector<const void*>&           A,
                                                   std::vector<const void*>&           B,
                                                   std::vector<const void*>&           C,
                                                   std::vector<void*>&                 D,
                                                   std::vector<rocblaslt_matrix_layout>& matA,
                                                   std::vector<rocblaslt_matrix_layout>& matB,
                                                   std::vector<rocblaslt_matrix_layout>& matC,
                                                   std::vector<rocblaslt_matrix_layout>& matD,
                                                   std::vector<const void*>&             alpha,
                                                   std::vector<const void*>&             beta)
{
    hipblasOperation_t     opA          = matmul_descr[0]->op_A;
    hipblasOperation_t     opB          = matmul_descr[0]->op_B;
    rocblaslt_compute_type compute_type = matmul_descr[0]->compute_type;
    hipblasDatatype_t      type_a       = matA[0]->type;
    hipblasDatatype_t      type_b       = matB[0]->type;
    hipblasDatatype_t      type_c       = matC[0]->type;
    hipblasDatatype_t      type_d       = matD[0]->type;

    std::vector<const void*>        A_vec, B_vec, C_vec, alpha_vec, beta_vec;
    std::vector<void*>              D_vec;
    std::vector<const void*>        bias_vec;
    std::vector<const void*>        scaleD_vec;
    std::vector<hipblasDatatype_t>  bias_type_vec;
    std::vector<rocblaslt_epilogue> epilogue_vec;
    std::vector<int64_t>            m_vec, n_vec, k_vec;
    std::vector<int64_t>            lda_vec, batch_stride_a_vec, num_batches_a_vec;
    std::vector<int64_t>            ldb_vec, batch_stride_b_vec, num_batches_b_vec;
    std::vector<int64_t>            ldc_vec, batch_stride_c_vec, num_batches_c_vec;
    std::vector<int64_t>            ldd_vec, batch_stride_d_vec, num_batches_d_vec;

    for(int i = 0; i < matmul_descr.size(); i++)
    {
        const void*       bias = nullptr;
        hipblasDatatype_t bias_type
            = matmul_descr[i]->bias_type == 0 ? matD[i]->type : matmul_descr[i]->bias_type;
        rocblaslt_epilogue epilogue = matmul_descr[i]->epilogue;
        if(is_bias_enabled(epilogue))
            bias = matmul_descr[i]->bias;

        const void* scaleD = nullptr;
        if(matmul_descr[i]->scaleD)
            scaleD = matmul_descr[i]->scaleD;

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
        if(validArgs != rocblaslt_status_continue)
            return validArgs;

        bias_type_vec.push_back(bias_type);
        epilogue_vec.push_back(epilogue);
        bias_vec.push_back(bias);
        scaleD_vec.push_back(scaleD);

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

        m_vec.push_back(num_rows_d);
        n_vec.push_back(num_cols_d);
        k_vec.push_back((opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a);

        A_vec.push_back(A[i]);
        B_vec.push_back(B[i]);
        C_vec.push_back(C[i]);
        D_vec.push_back(D[i]);
        alpha_vec.push_back(alpha[i]);
        beta_vec.push_back(beta[i]);
    }

    bool strided_batch = true;
    bool grouped_gemm  = true;

#define EX_PARM_GroupedGemm                                                                      \
    groupedgemm, opA, opB, m_vec, n_vec, k_vec, alpha_vec, A_vec, type_a, lda_vec,               \
        batch_stride_a_vec, B_vec, type_b, ldb_vec, batch_stride_b_vec, beta_vec, C_vec, type_c, \
        ldc_vec, batch_stride_c_vec, D_vec, type_d, ldd_vec, batch_stride_d_vec,                 \
        num_batches_a_vec, strided_batch, grouped_gemm, compute_type, bias_vec, scaleD_vec,      \
        bias_type_vec, epilogue_vec

    return rocblaslt_groupedgemm_create_template(EX_PARM_GroupedGemm);
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

    if(matA->type != matB->type || matA->type != matC->type || matA->type != matD->type)
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
                  "alpha",
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

rocblaslt_status rocblaslt_gemm_create(rocblaslt_gemm*         gemm,
                                       rocblaslt_handle        handle,
                                       rocblaslt_matmul_desc   matmul_descr,
                                       const void*             alpha,
                                       const void*             A,
                                       rocblaslt_matrix_layout matA,
                                       const void*             B,
                                       rocblaslt_matrix_layout matB,
                                       const void*             beta,
                                       const void*             C,
                                       rocblaslt_matrix_layout matC,
                                       void*                   D,
                                       rocblaslt_matrix_layout matD)
{
    // Check if handle is valid
    if(gemm == nullptr)
    {
        log_error(__func__, "invalid gemm pointer", gemm);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        *gemm = nullptr;
        // Allocate
        try
        {
            *gemm                        = new _rocblaslt_gemm();
            (*gemm)->rocblaslt_gemm_type = _rocblaslt_gemm_type_enum::ROCBLASLT_GEMM;
            (*gemm)->handle              = handle;
            log_api(__func__, "gemm[out]", *gemm);
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }

        // Check if handle is valid
        if(matmul_descr == nullptr || matA == nullptr || matB == nullptr || matC == nullptr
           || matD == nullptr)
        {
            log_error(__func__, "invalid handle pointer");
            return rocblaslt_status_invalid_handle;
        }

        if(matA->type != matB->type || matA->type != matC->type || matA->type != matD->type)
        {
            log_error(__func__, "invalid matrix datatype");
            return rocblaslt_status_type_mismatch;
        }

        return rocblaslt_gemm_create_impl(
            *gemm, matmul_descr, A, B, C, D, matA, matB, matC, matD, alpha, beta);
    }
}

rocblaslt_status rocblaslt_groupedgemm_create(rocblaslt_gemm*                       groupedgemm,
                                              rocblaslt_handle                      handle,
                                              std::vector<rocblaslt_matmul_desc>&   matmul_descr,
                                              std::vector<const void*>&             alpha,
                                              std::vector<const void*>&             A,
                                              std::vector<rocblaslt_matrix_layout>& matA,
                                              std::vector<const void*>&             B,
                                              std::vector<rocblaslt_matrix_layout>& matB,
                                              std::vector<const void*>&             beta,
                                              std::vector<const void*>&             C,
                                              std::vector<rocblaslt_matrix_layout>& matC,
                                              std::vector<void*>&                   D,
                                              std::vector<rocblaslt_matrix_layout>& matD)

{
    // Check if handle is valid
    if(groupedgemm == nullptr)
    {
        log_error(__func__, "invalid groupedgemm pointer", groupedgemm);
        return rocblaslt_status_invalid_value;
    }
    else
    {
        *groupedgemm = nullptr;
        // Allocate
        try
        {
            *groupedgemm                        = new _rocblaslt_gemm();
            (*groupedgemm)->rocblaslt_gemm_type = _rocblaslt_gemm_type_enum::ROCBLASLT_GROUPED_GEMM;
            (*groupedgemm)->handle              = handle;
            log_api(__func__, "groupedgemm[out]", *groupedgemm);
        }
        catch(const rocblaslt_status& status)
        {
            return status;
        }

        for(int i = 0; i < matmul_descr.size(); i++)
        {
            // Check if handle is valid
            if(matmul_descr[i] == nullptr || matA[i] == nullptr || matB[i] == nullptr
               || matC[i] == nullptr || matD[i] == nullptr)
            {
                log_error(__func__, "invalid handle pointer");
                return rocblaslt_status_invalid_handle;
            }

            // Check if pointer is valid
            // if(alpha[i] == nullptr || beta[i] == nullptr
            // || (A[i] == nullptr && (matA[i]->m * matA[i]->n))
            // || (B[i] == nullptr && (matB[i]->m * matB[i]->n))
            // || (C[i] == nullptr && (matC[i]->m * matC[i]->n))
            // || (D[i] == nullptr && (matD[i]->m * matD[i]->n)))
            // {
            //     log_error(__func__, "invalid data pointer");
            //     return rocblaslt_status_invalid_pointer;
            // }
            // if(workspace == nullptr && workspaceSizeInBytes > 0)
            // {
            //     log_error(__func__, "invalid workspace pointer");
            //     return rocblaslt_status_invalid_pointer;
            // }

            if(matA[i]->type != matB[i]->type || matA[i]->type != matC[i]->type
               || matA[i]->type != matD[i]->type || matA[0]->type != matA[i]->type)
            {
                log_error(__func__, "invalid  matrix datatype");
                return rocblaslt_status_type_mismatch;
            }
        }

        return rocblaslt_groupedgemm_create_impl(
            *groupedgemm, matmul_descr, A, B, C, D, matA, matB, matC, matD, alpha, beta);
    }
}

rocblaslt_status rocblaslt_ext_destroy(const rocblaslt_gemm gemm)
{
    if(gemm == nullptr)
    {
        log_error(__func__, "invalid groupedgemm pointer", gemm);
        return rocblaslt_status_invalid_pointer;
    }
    log_api(__func__, "groupedgemm", gemm);

    // Destruct
    try
    {
        delete gemm;
    }
    catch(const rocblaslt_status& status)
    {
        return status;
    }
    return rocblaslt_status_success;
}

rocblaslt_status rocblaslt_run(rocblaslt_gemm gemm, hipStream_t stream)
{
    return rocblaslt_run_template(gemm, stream);
}

rocblaslt_status rocblaslt_makeArgument(rocblaslt_gemm               gemm,
                                        const rocblaslt_matmul_algo* algo,
                                        void*                        workspace,
                                        hipStream_t                  stream)
{
    return rocblaslt_makeArgument_template(gemm, algo, workspace, stream);
}

#ifdef __cplusplus
}
#endif
