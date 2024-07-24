#include <hip/hip_runtime.h>
#include <hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include "helper.h"

void simpleGemmScaleB(hipblasLtHandle_t handle,
                      hipblasOperation_t trans_a,
                      hipblasOperation_t trans_b,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      int64_t batch_count,
                      float& alpha,
                      float& beta,
                      void* d_a,
                      void* d_b,
                      void* d_c,
                      void* d_d,
                      void* d_workspace,
                      int64_t max_workspace_size,
                      hipStream_t stream);

int main()
{
    Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> runner(
        1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);

    runner.run([&runner] {
        simpleGemmScaleB(runner.handle,
                         HIPBLAS_OP_N,
                         HIPBLAS_OP_N,
                         runner.m,
                         runner.n,
                         runner.k,
                         runner.batch_count,
                         runner.alpha,
                         runner.beta,
                         runner.d_a,
                         runner.d_b,
                         runner.d_c,
                         runner.d_d,
                         runner.d_workspace,
                         runner.max_workspace_size,
                         runner.stream);
    });

    return 0;
}

void simpleGemmScaleB(hipblasLtHandle_t handle,
                      hipblasOperation_t trans_a,
                      hipblasOperation_t trans_b,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      int64_t batch_count,
                      float& alpha,
                      float& beta,
                      void* d_a,
                      void* d_b,
                      void* d_c,
                      void* d_d,
                      void* d_workspace,
                      int64_t max_workspace_size,
                      hipStream_t stream)
{
    void *in_scale; // host
    void *d_in_scale; // device

    CHECK_HIP_ERROR(hipMalloc(&d_in_scale, 1 * sizeof(float)));
    CHECK_HIP_ERROR(hipHostMalloc(&in_scale, 1 * sizeof(float)));

    *(float*)in_scale = (float)0.5;
    CHECK_HIP_ERROR(hipMemcpyAsync(d_in_scale, in_scale, 1 * sizeof(float), hipMemcpyHostToDevice, stream));

    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_16F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_16F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_16F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_16F, m, n, m));

    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_in_scale, sizeof(void*)));

    std::cout << "Matrix dimensions: m=" << m << ", n=" << n << ", k=" << k << std::endl;
    std::cout << "Alpha: " << alpha << ", Beta: " << beta << std::endl;

    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size)));

    const int request_solutions = 10;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul, matA, matB, matC, matD, pref, request_solutions, heuristicResult, &returnedAlgoCount));

    std::cout << "Returned Algorithm Count: " << returnedAlgoCount << std::endl;

    if (returnedAlgoCount == 0)
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    uint64_t workspace_size = max_workspace_size;
    for (int i = 0; i < returnedAlgoCount; i++)
        workspace_size = std::max(workspace_size, heuristicResult[i].workspaceSize);

    std::cout << "Using workspace size: " << workspace_size << std::endl;

    for (int i = 0; i < returnedAlgoCount; i++) {
        auto& res = heuristicResult[i];
        std::cout << "Solution " << i << ": state=" << (res.state == HIPBLAS_STATUS_SUCCESS ? "SUCCESS" : "FAILURE") << std::endl;
    }

    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(handle, matmul, &alpha, d_a, matA, d_b, matB, &beta, d_c, matC, d_d, matD, &heuristicResult[0].algo, d_workspace, workspace_size, stream));

    CHECK_HIP_ERROR(hipFree(d_in_scale));
    CHECK_HIP_ERROR(hipFree(in_scale));

    return;
}
