***********************
hipblasLtExt Usage
***********************

Introduction
====================

hipBLASLt has extension APIs with prefix hipblasLtExt. The extensions support:

1. Run API (Kernel direct launch)

2. Grouped gemm

3. Get all algorithms

Run API (Kernel direct launch)
====================

hipblasLt has its own instance (hipblasLtExtGemm_t). It stores the handle, gemm type, problem(s), number of problems, and workspace bytes when it is created.

.. code-block:: c++

    hipblasLtExtGemmCreate()
    hipblasLtExtGroupedGemmCreate()

The instance has to be destroy with the destroy API.

.. code-block:: c++

    hipblasLtExtDestroy()

THe user can get hueristic and make kernel arguments with the instance. If the properties of the gemm and the inputs don't change, the user can call the run API to launch the kernel directly.

.. code-block:: c++

    // Pseudo code
    hipblasLtExtGemm_t gemm;
    hipblasLtExtGemmCreate(gemm);
    hipblasLtExtAlgoGetHeuristic(gemm);
    hipblasLtExtMakeArgument(gemm);
    for(int i = 0; i < 10; i++)
    {
        hipblasLtExtRun(gemm);
    }
    hipblasLtExtDestroy(gemm);

Grouped gemm
====================

hipblasLtExt supports grouped gemm. The inputs are vectors of hipblasLtMatrixLayout_t and vectors of input pointers.

.. code-block:: c++

    HIPBLASLT_EXPORT
    hipblasStatus_t hipblasLtExtGroupedGemmCreate(hipblasLtExtGemm_t*                 groupedgemm,
                                                hipblasLtHandle_t                     handle,
                                                std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
                                                std::vector<float>&                   alpha,
                                                std::vector<void*>&                   A,
                                                std::vector<hipblasLtMatrixLayout_t>& matA,
                                                std::vector<void*>&                   B,
                                                std::vector<hipblasLtMatrixLayout_t>& matB,
                                                std::vector<float>&                   beta,
                                                std::vector<void*>&                   C,
                                                std::vector<hipblasLtMatrixLayout_t>& matC,
                                                std::vector<void*>&                   D,
                                                std::vector<hipblasLtMatrixLayout_t>& matD);

Get all algorithms
====================

Get all algorithms lets users to get all the algorithms of a specific problem type. It requires the transpose of A, B, the data type of the inputs, and the compute type.

.. code-block:: c++

    HIPBLASLT_EXPORT
    hipblasStatus_t hipblasLtExtGetAllAlgos(hipblasLtHandle_t                  handle,
                                            hipblasLtExtGemmTypeEnum_t         typeGemm,
                                            hipblasOperation_t                 opA,
                                            hipblasOperation_t                 opB,
                                            hipblasDatatype_t                  typeA,
                                            hipblasDatatype_t                  typeB,
                                            hipblasDatatype_t                  typeC,
                                            hipblasDatatype_t                  typeD,
                                            hipblasLtComputeType_t             typeCompute,
                                            hipblasLtMatmulHeuristicResult_t** heuristicResults,
                                            int*                               returnedAlgoCount);

This API does not require any problem size or epilogue as input, but will use another API "isAlgoSupported" to check if the algorithm supports a problem.
The returned hipblasLtMatmulHeuristicResult_t array has to be freed by API hipblasLtExtFreeAlgos.

The API "isAlgoSupported" supports both Matmul API and Run API.

.. code-block:: c++

    hipblasLtExtMatmulIsAlgoSupported()
    hipblasLtExtIsAlgoSupported()

The API will return the required workspace size in bytes if success.

.. code-block:: c++

    // Get all algorithms
    CHECK_HIPBLASLT_ERROR(hipblasLtExtGetAllAlgos(handle,
                                                  HIPBLASLT_GEMM,
                                                  trans_a,
                                                  trans_b,
                                                  in_out_datatype,
                                                  in_out_datatype,
                                                  in_out_datatype,
                                                  in_out_datatype,
                                                  HIPBLASLT_COMPUTE_F32,
                                                  &heuristicResult,
                                                  &returnedAlgoCount));

    validIdx.clear();
    for(int j = 0; j < returnedAlgoCount; j++)
    {
        size_t workspace_size = 0;
        if(hipblasLtExtMatmulIsAlgoSupported(handle,
                                             matmul,
                                             &(alpha),
                                             matA,
                                             matB,
                                             &(beta),
                                             matC,
                                             matD,
                                             &heuristicResult[j].algo,
                                             &workspace_size)
           == HIPBLAS_STATUS_SUCCESS)
        {
            validIdx.push_back(j);
            heuristicResult[j].workspaceSize = workspace_size;
        }
        else
        {
            heuristicResult[j].workspaceSize = 0;
        }
    }
