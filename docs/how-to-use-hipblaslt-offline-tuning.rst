.. meta::
   :description: A library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool, tuning, GEMM

.. _how-to-use-hipblaslt-offline-tuning:

********************************
User Offline Tuning
********************************

``hipblaslt-bench`` can be used to find the best-performing GEMM kernel for a given set of GEMM problems. Use the command line interface to access this functionality. (See :ref:`clients` for more details.)

``hipblaslt-bench`` provides the best solution index for a given problem size. This index can be used directly in future GEMM calls through the User Offline Tuning mechanism. However, these indices cannot be reused across library releases or across different device architectures.

Running
=================================

To find and use the best GEMM kernel for a problem, follow these steps:

1. Generate the tuning command line by setting the environment variable ``HIPBLASLT_LOG_MASK=32`` before calling any hipBLASLt APIs. For more details on how to use ``hipblaslt-bench``, see :ref:`Logging and heuristics <logging-heuristics>`.

   In the Bash shell, set the following environment variable:

.. code-block:: bash

    export HIPBLASLT_LOG_MASK=32

    In this case, `sample_hipblaslt_gemm.cpp <https://github.com/ROCm/hipBLASLt/blob/develop/clients/samples/01_basic_gemm/sample_hipblaslt_gemm.cpp>`_ is used as an example:

.. code-block:: bash

    ./sample_hipblaslt_gemm

   The tuning command displays the following log entry:

.. code-block:: bash

    hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 --lda 1024 --ldb 1024 --ldc 1024 --ldd 1024  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.000000 --beta 1.000000 --transA N --transB N --batch_count 1  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r 


2. Set the environment variable ``HIPBLASLT_TUNING_FILE=<file_name>`` to tune and store the tuning result of the best solution indices for those GEMM problems, where ``<file_name>`` points to the tuning file.

- Please note that the default settings will have the following changes in the tuning environment.

.. code-block:: bash

    --iters |-i <value>             Iterations to run inside timing loop                                                (Default value is: 1000)
    --cold_iters |-j <value>        Cold Iterations to run before entering the timing loop                              (Default value is: 1000)
    --algo_method <value>           Use different algorithm search API. Options: heuristic, all, index (still considered all).                  (Default value is: all)
    --requested_solution <value>    Requested solution num. Set to -1 to get all solutions. Only valid when algo_method is set to heuristic.    (Default value is: -1)
    --rotating <value>              Use rotating memory blocks for each iteration, size in MB.                          (Default value is: 512)

   After the tuning completes, the expected output is displayed as follows:

.. code-block:: bash

    ./hipblaslt-bench --api_method c -m 64 -n 64 -k 1024 --lda 64 --ldb 1024 --ldc 64 --ldd 64  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.100000 --beta 0.900000 --transA N --transB N --batch_count 1  --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r
    
    Winner: 
    transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us,soulution_index
    N,N,0,1,64,64,1024,1.1,64,65536,0.9,1024,65536,64,4096,64,4096,f32_r,f32_r,f32_r,f32_r,f32_r,0,0,0,0,0,none,0,f32_r,512,903.229,55.8608,9.28735,49257


3. Set the environment variable ``HIPBLASLT_TUNING_OVERRIDE_FILE=<file_name>`` to load the tuning file and override the default kernel selection with the optimal kernel choices, where ``<file_name>`` points to the tuning file.

- For example, user can use ``hisblaslt-bench`` (algo_method: heuristic) to obtain solutions for a problem, which will include the best tuning solution index.

.. code-block:: bash

    ./hipblaslt-bench --api_method c -m 64 -n 64 -k 1024 --lda 64 --ldb 1024 --ldc 64 --ldd 64  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.100000 --beta 0.900000 --transA N --transB N --batch_count 1  --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r --algo_method heuristic --requested_solution 3 --print_kernel_info 
    
    transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us
    [0]:
    N,N,0,1,64,64,1024,1.1,64,65536,0.9,1024,65536,64,4096,64,4096,f32_r,f32_r,f32_r,f32_r,f32_r,0,0,0,0,0,none,0,f32_r,512,906.538,56.0654,9.25345
    --Solution index: 49257
    [1]:
    N,N,0,1,64,64,1024,1.1,64,65536,0.9,1024,65536,64,4096,64,4096,f32_r,f32_r,f32_r,f32_r,f32_r,0,0,0,0,0,none,0,f32_r,512,411.904,25.4745,20.3655
    --Solution index: 48934
    [2]:
    N,N,0,1,64,64,1024,1.1,64,65536,0.9,1024,65536,64,4096,64,4096,f32_r,f32_r,f32_r,f32_r,f32_r,0,0,0,0,0,none,0,f32_r,512,394.991,24.4285,21.2375
    --Solution index: 49003