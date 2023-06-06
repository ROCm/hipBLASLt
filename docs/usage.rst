********************************
hipBLASLtExt Reference
********************************

hipBLASLtExt Datatypes Reference
================================

GemmType
-------------------------------------
.. doxygenenum:: GemmType

GemmProblemType
-------------------------------------
.. doxygenstruct:: hipblaslt_ext::GemmProblemType
    :members:
    :protected-members:
    :private-members:

GemmEpilogue
-------------------------------------
.. doxygenstruct:: hipblaslt_ext::GemmEpilogue
    :members:
    :protected-members:
    :private-members:

GemmInputs
-------------------------------------
.. doxygenstruct:: hipblaslt_ext::GemmInputs
    :members:
    :protected-members:
    :private-members:

hipBLASLtExt Class Reference
================================

GemmPreference
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GemmPreference
    :members:
    :protected-members:
    :private-members:

GemmInstance
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GemmInstance
    :members:
    :protected-members:
    :private-members:

Gemm
-------------------------------------
.. doxygenclass:: hipblaslt_ext::Gemm
    :members:
    :protected-members:
    :private-members:

GroupedGemm
-------------------------------------
.. doxygenclass:: hipblaslt_ext::GroupedGemm
    :members:
    :protected-members:
    :private-members:

hipBLASLtExt API Reference
================================

getAllAlgos()
------------------------------------------
.. doxygenfunction:: getAllAlgos

getIndexFromAlgo()
------------------------------------------
.. doxygenfunction:: getIndexFromAlgo

getAlgosFromIndex()
------------------------------------------
.. doxygenfunction:: getAlgosFromIndex

matmulIsAlgoSupported()
------------------------------------------
.. doxygenfunction:: matmulIsAlgoSupported

hipblasLtExt Usage
================================

Introduction
--------------

hipBLASLt has extension APIs with namespace hipblaslt_ext. It is C++ compatible only. The extensions support:

1. Gemm

2. Grouped gemm

3. Get all algorithms

Gemm
--------------

hipblasLt has its own instance.

The user must assign the problem type when construct or import the problem from hipBLAS API.

.. code-block:: c++

    HIPBLASLT_EXPORT explicit Gemm(hipblasLtHandle_t      handle,
                                   hipblasOperation_t     opA,
                                   hipblasOperation_t     opB,
                                   hipblasDatatype_t      typeA,
                                   hipblasDatatype_t      typeB,
                                   hipblasDatatype_t      typeC,
                                   hipblasDatatype_t      typeD,
                                   hipblasLtComputeType_t typeCompute);

    HIPBLASLT_EXPORT explicit Gemm(hipblasLtHandle_t       handle,
                                   hipblasLtMatmulDesc_t   matmul_descr,
                                   const void*             alpha,
                                   const void*             A,
                                   hipblasLtMatrixLayout_t matA,
                                   const void*             B,
                                   hipblasLtMatrixLayout_t matB,
                                   const void*             beta,
                                   const void*             C,
                                   hipblasLtMatrixLayout_t matC,
                                   void*                   D,
                                   hipblasLtMatrixLayout_t matD);

After the instance is created, the user can set the problem with the API.
The API may requires the following structures:

GemmProblemType lets user able to change the problem type after the instance is initialized.

.. code-block:: c++

    struct GemmProblemType
    {
        hipblasOperation_t     op_a;
        hipblasOperation_t     op_b;
        hipblasDatatype_t      type_a;
        hipblasDatatype_t      type_b;
        hipblasDatatype_t      type_c;
        hipblasDatatype_t      type_d;
        hipblasLtComputeType_t type_compute;
    };

GemmEpilogue lets user to control the epilogue of the problem.

.. code-block:: c++

    struct GemmEpilogue
    {
        hipblasLtEpilogue_t mode = HIPBLASLT_EPILOGUE_DEFAULT;
        hipblasDatatype_t   bias_data_type;
        int                 aux_ld;
        int                 aux_stride;
    };

GemmInputs is the problem inputs.

.. code-block:: c++

    struct GemmInputs
    {
        void* a = nullptr;
        void* b = nullptr;
        void* c = nullptr;
        void* d = nullptr;
        void* alpha = nullptr;
        void* beta = nullptr;
        // Epilogue inputs
        void* bias = nullptr;
        void* scaleD = nullptr;
        void* aux = nullptr;
    };

And the setProblem APIs:

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(
        int64_t m, int64_t n, int64_t k, int64_t batch_count, GemmEpilogue& epilogue, GemmInputs& inputs);

The user can also set the leading dimensions, strides, and reassign the data type with the following API.

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(int64_t          m,
                                                int64_t          n,
                                                int64_t          k,
                                                int64_t          batch_count,
                                                int64_t          lda,
                                                int64_t          ldb,
                                                int64_t          ldc,
                                                int64_t          ldd,
                                                int64_t          strideA,
                                                int64_t          strideB,
                                                int64_t          strideC,
                                                int64_t          strideD,
                                                GemmEpilogue&    epilogue,
                                                GemmInputs&      inputs,
                                                GemmProblemType& problemtype);

The user can also importing problems from hipblasLt APIs after the instance is created, note that this may overwrite the problem type of the instance.

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(hipblasLtMatmulDesc_t   matmul_descr,
                                                const void*             alpha,
                                                const void*             A,
                                                hipblasLtMatrixLayout_t matA,
                                                const void*             B,
                                                hipblasLtMatrixLayout_t matB,
                                                const void*             beta,
                                                const void*             C,
                                                hipblasLtMatrixLayout_t matC,
                                                void*                   D,
                                                hipblasLtMatrixLayout_t matD);

The user can get hueristic and make kernel arguments with the instance. If the properties of the gemm and the inputs don't change, the user can call the run API to launch the kernel directly.

.. code-block:: c++

    // Pseudo code
    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);
    // Default epilogue mode is HIPBLASLT_EPILOGUE_DEFAULT
    hipblaslt_ext::GemmEpilogue epilogue;
    hipblaslt_ext::GemmInputs inputs;
    inputs.a = a;
    inputs.b = b;
    inputs.c = c;
    inputs.d = d;
    inputs.alpha = alpha;
    inputs.beta = beta;

    hipblaslt_ext::Gemm gemm(handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             HIPBLAS_R_16F,
                             HIPBLAS_R_16F,
                             HIPBLAS_R_16F,
                             HIPBLAS_R_16F,
                             HIPBLASLT_COMPUTE_F32);
    std::vector<hipblasLtMatmulHeuristicResult_t> hueristic;
    gemm.setProblem(1, 1, 1, 1, epilogue, inputs); // m, n, k, batch
    gemm.algoGetHeuristic(gemm, pref, hueristic);
    gemm.initialize(hueristic[0].algo, d_workspace, stream);
    for(int i = 0; i < 10; i++)
    {
        gemm.run(stream);
    }

Grouped Gemm
--------------

hipblasLtExt supports grouped gemm. It shares the same class with normal gemm.

After the problem is set, the user can check the problem type with function getGemmType().

.. code-block:: c++

    enum class GemmType
    {
        HIPBLASLT_GEMM             = 1,
        HIPBLASLT_GROUPED_GEMM     = 2
    };

The grouped gemm class also has the setProblem APIs.

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(
        int64_t m, int64_t n, int64_t k, int64_t batch_count, GemmEpilogue& epilogue, GemmInputs& inputs);

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&      m,
                                                std::vector<int64_t>&      n,
                                                std::vector<int64_t>&      k,
                                                std::vector<int64_t>&      batch_count,
                                                std::vector<GemmEpilogue>& epilogue,
                                                std::vector<GemmInputs>&   inputs);

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&      m,
                                                std::vector<int64_t>&      n,
                                                std::vector<int64_t>&      k,
                                                std::vector<int64_t>&      batch_count,
                                                std::vector<int64_t>&      lda,
                                                std::vector<int64_t>&      ldb,
                                                std::vector<int64_t>&      ldc,
                                                std::vector<int64_t>&      ldd,
                                                std::vector<int64_t>&      strideA,
                                                std::vector<int64_t>&      strideB,
                                                std::vector<int64_t>&      strideC,
                                                std::vector<int64_t>&      strideD,
                                                std::vector<GemmEpilogue>& epilogue,
                                                std::vector<GemmInputs>&   inputs,
                                                GemmProblemType&           problemtype);

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<hipblasLtMatmulDesc_t>&   matmul_descr,
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

For the following API, the argument "epilogue" supports broadcasting. They will be broadcasted to the length of the problem size by duplicating the last element.

.. code-block:: c++

    HIPBLASLT_EXPORT hipblasStatus_t setProblem(std::vector<int64_t>&      m,
                                                std::vector<int64_t>&      n,
                                                std::vector<int64_t>&      k,
                                                std::vector<int64_t>&      batch_count,
                                                std::vector<int64_t>&      lda,
                                                std::vector<int64_t>&      ldb,
                                                std::vector<int64_t>&      ldc,
                                                std::vector<int64_t>&      ldd,
                                                std::vector<int64_t>&      strideA,
                                                std::vector<int64_t>&      strideB,
                                                std::vector<int64_t>&      strideC,
                                                std::vector<int64_t>&      strideD,
                                                std::vector<GemmEpilogue>& epilogue,
                                                std::vector<GemmInputs>&   inputs,
                                                GemmProblemType&           problemtype);

Note that currently only supports problemtype size equals to 1 (Only one GemmProblemType for all problems).

.. code-block:: c++

    // Pseudo code
    std::vector<int64_t> m, n, k;
    // ...
    for(size_t i = 0; i < problem_size, i++)
    {
        // ...
    }
    std::vector<GemmProblemType> problemtypes;
    problemtypes.push_back(problemtype);
    groupedgemm.setProblem(m, n, k, batch_count, lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, epilogue, inputs, problemtypes);

The base class (GemmInstance)
--------------

This is the base class of class Gemm and GroupedGemm.

.. code-block:: c++

    // Gets huesristic from the instance.
    HIPBLASLT_EXPORT hipblasStatus_t algoGetHeuristic(const int                                      requestedAlgoCount,
                                                      const GemmPreference&                          pref,
                                                      std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

    // Returns SUCCESS if the algo is supported, also returns the required workspace size in bytes.
    HIPBLASLT_EXPORT hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo, size_t& workspaceSizeInBytes);

    // Initializes the instance before calling run. Requires every time the problem is set.
    HIPBLASLT_EXPORT hipblasStatus_t initialize(const hipblasLtMatmulAlgo_t& algo, void* workspace, hipStream_t stream);

    // Run the problem.
    HIPBLASLT_EXPORT hipblasStatus_t run(hipStream_t stream);

Get all algorithms
--------------

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

.. code-block:: c++

    hipblaslt_ext::matmulIsAlgoSupported()
    gemm.isAlgoSupported()

The API will return the required workspace size in bytes if success.

.. code-block:: c++

    // Get all algorithms
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
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
        if(hipblaslt_ext::matmulIsAlgoSupported(handle,
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

Using extension APIs.

Gemm
^^^^^^^^^^^^^^^^^

.. code-block:: c++

    // Pseudo code for gemm problem
    // Get all algorithms
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLASLT_COMPUTE_F32,
                                                     heuristicResult));

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);
    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.mode = HIPBLASLT_EPILOGUE_GELU;
    hipblaslt_ext::GemmInputs inputs;
    inputs.a = a;
    inputs.b = b;
    inputs.c = c;
    inputs.d = d;
    inputs.alpha = alpha;
    inputs.beta = beta;

    hipblaslt_ext::Gemm gemm(handle,
                             HIPBLAS_OP_N,
                             HIPBLAS_OP_N,
                             HIPBLAS_R_16F,
                             HIPBLAS_R_16F,
                             HIPBLAS_R_16F,
                             HIPBLAS_R_16F,
                             HIPBLASLT_COMPUTE_F32);

    gemm.setProblem(1, 1, 1, 1, epilogue, inputs); // m, n, k, batch

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
    {
        size_t workspace_size = 0;
        if(gemm.isAlgoSupported(heuristicResult[j].algo, workspace_size)
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

    if(validIdx.size() > 1)
    {
        gemm.initialize(heuristicResult[validIdx[0]].algo, d_workspace, stream);
        for(int i = 0; i < 10; i++)
        {
            gemm.run(stream);
        }
    }

Grouped gemm
^^^^^^^^^^^^^^^^^

.. code-block:: c++

    // Pseudo code for grouped gemm problem
    // Get all algorithms
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                     HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N,
                                                     HIPBLAS_OP_N,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     in_out_datatype,
                                                     HIPBLASLT_COMPUTE_F32,
                                                     heuristicResult));

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(1000000);

    std::vector<int64_t> m(gemm_count);
    std::vector<int64_t> n(gemm_count);
    std::vector<int64_t> k(gemm_count);
    std::vector<int64_t> batch_count(gemm_count);
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue(gemm_count);
    std::vector<hipblaslt_ext::GemmInputs> inputs(gemm_count);
    for(int i = 0; i < gemm_count; i++)
    {
        m[i] = 1;
        n[i] = 1;
        k[i] = 1;
        batch_count[i] = 1;
        epilogue[i].mode = HIPBLASLT_EPILOGUE_GELU;
        inputs[i].a = a[i];
        inputs[i].b = b[i];
        inputs[i].c = c[i];
        inputs[i].d = d[i];
        inputs[i].alpha = alpha[i];
        inputs[i].beta = beta[i];
    }


    hipblaslt_ext::GroupedGemm groupedGemm(handle,
                                           HIPBLAS_OP_N,
                                           HIPBLAS_OP_N,
                                           HIPBLAS_R_16F,
                                           HIPBLAS_R_16F,
                                           HIPBLAS_R_16F,
                                           HIPBLAS_R_16F,
                                           HIPBLASLT_COMPUTE_F32);

    groupedGemm.setProblem(m, n, k, batch_count, epilogue, inputs); // m, n, k, batch

    validIdx.clear();
    for(int j = 0; j < heuristicResult.size(); j++)
    {
        size_t workspace_size = 0;
        if(groupedGemm.isAlgoSupported(heuristicResult[j].algo, workspace_size)
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

    if(validIdx.size() > 1)
    {
        groupedGemm.initialize(heuristicResult[validIdx[0]].algo, d_workspace, stream);
        for(int i = 0; i < 10; i++)
        {
            groupedGemm.run(stream);
        }
    }

Algorithm Index
--------------

The extension API lets user to get the algorithm index from hipblasLtMatmulAlgo_t.

.. code-block:: c++

    HIPBLASLT_EXPORT int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo);


It also supports user to get the heuristic results by giving an index vector.

.. code-block:: c++

    HIPBLASLT_EXPORT
    hipblasStatus_t
        getAlgosFromIndex(hipblasLtHandle_t                              handle,
                          std::vector<int>&                              algoIndex,
                          std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

Example code
^^^^^^^^^^^^

.. code-block:: c++
    int index = hipblaslt_ext::getIndexFromAlgo(testResults[i].algo);
    // Save the index to disk or somewhere else for later use.

    // Get the index from previous state.
    std::vector<int> algoIndex(index);
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, heuristicResults));
