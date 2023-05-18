***********************
hipblasLtExt Usage
***********************

Introduction
====================

hipBLASLt has extension APIs with namespace hipblaslt_ext. It is C++ compatible only. The extensions support:

1. Run API (Kernel direct launch)

2. Grouped gemm

3. Get all algorithms

Run API (Kernel direct launch)
====================

hipblasLt has its own instance (hipblaslt_ext::Gemm). It stores the handle, and max workspace bytes when it is created.

.. code-block:: c++

    hipblaslt_ext::Gemm(handle, max_workspace_size);

Currently supports importing problems from hipblasLt APIs.

.. code-block:: c++

    gemm.setProblemFromhipBlasLt();

THe user can get hueristic and make kernel arguments with the instance. If the properties of the gemm and the inputs don't change, the user can call the run API to launch the kernel directly.

.. code-block:: c++

    // Pseudo code
    hipblaslt_ext::Gemm gemm;
    std::vector<hipblasLtMatmulHeuristicResult_t> hueristic;
    gemm.setProblemFromhipBlasLt();
    hipblaslt_ext::algoGetHeuristic(gemm, hueristic);
    gemm.makeArgument(hueristic[0].algo, stream);
    for(int i = 0; i < 10; i++)
    {
        gemm.run(stream);
    }

Grouped gemm
====================

hipblasLtExt supports grouped gemm. It shares the same class with normal gemm.

.. code-block:: c++

    hipblaslt_ext::Gemm(handle, max_workspace_size);

Currently supports importing problems from hipblasLt APIs. The inputs are vectors of hipblasLtMatrixLayout_t and vectors of input pointers.

.. code-block:: c++

    gemm.setGroupedProblemFromhipBlasLt();

After the problem is set, the user can check the problem type with function getGemmType().

.. code-block:: c++

    enum class GemmType
    {
        HIPBLASLT_GEMM             = 1,
        HIPBLASLT_GROUPED_GEMM     = 2,
        HIPBLASLT_GEMMTYPE_UNKNOWN = 3,
    };

Get all algorithms
====================

Get all algorithms lets users to get all the algorithms of a specific problem type. It requires the transpose of A, B, the data type of the inputs, and the compute type.

.. code-block:: c++

    HIPBLASLT_EXPORT
    hipblasStatus_t hipblaslt_ext::getAllAlgos(hipblasLtHandle_t                              handle,
                                               hipblasLtExtGemmTypeEnum_t                     typeGemm,
                                               hipblasOperation_t                             opA,
                                               hipblasOperation_t                             opB,
                                               hipblasDatatype_t                              typeA,
                                               hipblasDatatype_t                              typeB,
                                               hipblasDatatype_t                              typeC,
                                               hipblasDatatype_t                              typeD,
                                               hipblasLtComputeType_t                         typeCompute,
                                               std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

This API does not require any problem size or epilogue as input, but will use another API "isAlgoSupported" to check if the algorithm supports a problem.

The API "isAlgoSupported" supports both hipblasLt API and the extension API.

.. code-block:: c++

    hipblaslt_ext::matmulIsAlgoSupported()
    hipblaslt_ext::isAlgoSupported()

The API will return the required workspace size in bytes if success.

.. code-block:: c++

    // Get all algorithms
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     trans_a,
                                                     trans_b,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLASLT_COMPUTE_F32,
                                                     heuristicResult));

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
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
                                             heuristicResult[j].algo,
                                             workspace_size)
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
