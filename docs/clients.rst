============================
Clients
============================

There are 4 client executables that can be used with hipBLASLt.

1. hipblaslt-test

2. hipblaslt-bench

3. example_hipblaslt_preference

4. example_hipblaslt_groupedgemm

5. example_hipblaslt_groupedgemm_get_all_algos

These clients can be built by following the instructions in the `Build and Install hipBLASLt github page <https://github.com/ROCmSoftwarePlatform/hipBLASLt>`_ . After building the hipBLASLt clients, they can be found in the directory ``hipBLASLt/build/release/clients/staging``.
The next section will cover a brief explanation and the usage of each hipBLASLt clients.

hipblaslt-test
============================
hipblaslt-test is the main regression gtest for hipBLASLt. All test items should pass.

Run full test items:

.. code-block:: bash

   ./hipblaslt-test

Run partial test items with filter:

.. code-block:: bash

   ./hipblaslt-test --gtest_filter=<test pattern>

Demo "quick" test:

.. code-block:: bash

   ./hipblaslt-test --gtest_filter=*quick*

hipblaslt-bench
============================
hipblaslt-bench is used to measure performance and to verify the correctness of hipBLASLt functions.

It has a command line interface.

For example, run fp32 GEMM with validation:

.. code-block:: bash

   ./hipblaslt-bench --precision f32_r -v 1
   transA,transB,M,N,K,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,d_type,compute_type,activation_type,bias_vector,hipblaslt-Gflops,us
   N,N,128,128,128,1,128,16384,0,128,16384,128,16384,128,16384,f32_r,f32_r,none,0, 415.278, 10.

For more information:

.. code-block:: bash

   ./hipblaslt-bench --help
   --sizem |-m <value>        Specific matrix size: the number of rows or columns in matrix.                      (Default value is: 128)
   --sizen |-n <value>        Specific matrix the number of rows or columns in matrix                             (Default value is: 128)
   --sizek |-k <value>        Specific matrix size: the number of columns in A and rows in B.                     (Default value is: 128)
   --lda <value>              Leading dimension of matrix A.
   --ldb <value>              Leading dimension of matrix B.
   --ldc <value>              Leading dimension of matrix C.
   --ldd <value>              Leading dimension of matrix D.
   --lde <value>              Leading dimension of matrix E.
   --any_stride               Do not modify input strides based on leading dimensions
   --stride_a <value>         Specific stride of strided_batched matrix A, second dimension * leading dimension.
   --stride_b <value>         Specific stride of strided_batched matrix B, second dimension * leading dimension.
   --stride_c <value>         Specific stride of strided_batched matrix C, second dimension * leading dimension.
   --stride_d <value>         Specific stride of strided_batched matrix D, second dimension * leading dimension.
   --stride_e <value>         Specific stride of strided_batched matrix E, second dimension * leading dimension.
   --alpha <value>            specifies the scalar alpha                                                          (Default value is: 1)
   --beta <value>             specifies the scalar beta                                                           (Default value is: 0)
   --function |-f <value>     BLASLt function to test. Options: matmul                                            (Default value is: matmul)
   --precision |-r <value>    Precision of matrix A,B,C,D  Options: f32_r,f16_r,bf16_r,f64_r,i32_r,i8_r           (Default value is: f16_r)
   --a_type <value>           Precision of matrix A. Options: f32_r,f16_r,bf16_r
   --b_type <value>           Precision of matrix B. Options: f32_r,f16_r,bf16_r
   --c_type <value>           Precision of matrix C. Options: f32_r,f16_r,bf16_r
   --d_type <value>           Precision of matrix D. Options: f32_r,f16_r,bf16_r
   --compute_type <value>     Precision of computation. Options: s,f32_r,x,xf32_r,f64_r,i32_r                     (Default value is: f32_r)
   --scale_type <value>       Precision of scalar. Options: f16_r,bf16_r
   --initialization <value>   Intialize matrix data.Options: rand_int, trig_float, hpl(floating)                  (Default value is: hpl)
   --transA <value>           N = no transpose, T = transpose, C = conjugate transpose                            (Default value is: N)
   --transB <value>           N = no transpose, T = transpose, C = conjugate transpose                            (Default value is: N)
   --batch_count <value>      Number of matrices. Only applicable to batched and strided_batched routines         (Default value is: 1)
   --HMM                      Parameter requesting the use of HipManagedMemory
   --verify |-v <value>       Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)                        (Default value is: )
   --iters |-i <value>        Iterations to run inside timing loop                                                (Default value is: 10)
   --cold_iters |-j <value>   Cold Iterations to run before entering the timing loop                              (Default value is: 2)
   --algo <value>             Reserved.                                                                           (Default value is: 0)
   --solution_index <value>   Reserved.                                                                           (Default value is: 0)
   --activation_type <value>  Options: None, gelu, relu                                                           (Default value is: none)
   --activation_arg1 <value>  Reserved.                                                                           (Default value is: 0)
   --activation_arg2 <value>  Reserved.                                                                           (Default value is: inf)
   --bias_type <value>        Precision of bias vector.Options: f16_r,bf16_r,f32_r,default(same with D type)
   --bias_source <value>      Choose bias source: a, b, d                                                         (Default value is: d)
   --bias_vector              Apply bias vector
   --scaleAlpha_vector        Apply scaleAlpha vector
   --use_e                    Apply AUX output/ gradient input
   --gradient                 Enable gradient
   --grouped_gemm <value>     Use grouped_gemm if non-zero. Number of gemms to run                                (Default value is: 0)
   --device <value>           Set default device to be used for subsequent program runs                           (Default value is: 0)
   --c_noalias_d              C and D are stored in separate memory
   --workspace <value>        Set fixed workspace memory size instead of using hipblaslt managed memory           (Default value is: 0)
   --log_function_name        Function name precedes other itmes.
   --function_filter <value>  Simple strstr filter on function name only without wildcards
   --api_method <value>       Use extension API. 0: C style API. 1: declaration with C hipblasLtMatmul Layout/Desc but set, initialize, and run the problem with C++ extension API. 2: Using C++ extension API only. Options: 0, 1, 2. (default: 0)  (Default value is: 0)
   --help |-h                 produces this help message
   --version <value>          Prints the version number

example_hipblaslt_preference
============================
example_hipblaslt_preference is a basic sample hipBLASLt app. Beginner can get start from its sample source code.

For more information:

.. code-block:: bash

   ./example_hipblaslt_preference --help
   Usage: ./example_hipblaslt_preference <options>
   options:
        -h, --help                              Show this help message
        -v, --verbose                           Verbose output
        -V, --validate                          Verify results
        -s, --request_solutions                 Number of solutions to run (default is 1)
        -m              m                       GEMM_STRIDED argument m
        -n              n                       GEMM_STRIDED argument n
        -k              k                       GEMM_STRIDED argument k
        --lda           lda                     GEMM_STRIDED argument lda
        --ldb           ldb                     GEMM_STRIDED argument ldb
        --ldc           ldc                     GEMM_STRIDED argument ldc
        --ldd           ldd                     GEMM_STRIDED argument ldd
        --trans_a       trans_a                 GEMM_STRIDED argument trans_a (N, T)
        --trans_b       trans_b                 GEMM_STRIDED argument trans_b (N, T)
        --datatype      datatype                GEMM_STRIDED argument in out datatype:fp32,fp16,bf16
        --stride_a      stride_a                GEMM_STRIDED argument stride_a
        --stride_b      stride_b                GEMM_STRIDED argument stride_b
        --stride_c      stride_c                GEMM_STRIDED argument stride_c
        --stride_d      stride_d                GEMM_STRIDED argument stride_d
        --alpha         alpha                   GEMM_STRIDED argument alpha
        --beta          beta                    GEMM_STRIDED argument beta
        --batch_count   batch                   GEMM_STRIDED argument batch count
        --act           act                     GEMM_STRIDED set activation type: relu or gelu
        --grad          grad                    GEMM_STRIDED enable grad: 0 or 1 (default is 0)
        --use_e         use_e                   GEMM_STRIDED enable use_e: 0 or 1 (default is 0)
        --bias          bias                    GEMM_STRIDED enable bias and choose bias src: A, B, D
        --header        header                  Print header for output (default is enabled)
        --timing        timing                  Bechmark GPU kernel performance:0 or 1 (default is 1)
        --bench_count   bench_count             Number of benchmark runs (default is 3)
        --sync_count    sync_count              Number of sync runs (default is 1)
        --cold_iters    cold_iters              Cold Iterations to run before entering the timing loop (default is 0)
        --ext           ext                     use Ext API
        --all           all                     Get all solutions

For example, to measure performance of fp32 gemm:

.. code-block:: bash

   ./example_hipblaslt_preference --datatype fp32 --trans_a N --trans_b N -m 4096 -n 4096 -k 4096 --alpha 1 --beta 1

On a mi210 machine the above command outputs a performance of 13509 Gflops below:

.. code-block:: bash

   transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count, alpha, beta, bias, activationType, ms, tflops
   NN, 4096, 4096, 4096, 4096, 4096, 4096, 16777216, 16777216, 16777216, 1, 1, 1, 0, none, 10.173825, 13.509074

The user can copy and change the above command. For example, to change the datatype to IEEE-16 bit and the size to 2048:

.. code-block:: bash

   ./example_hipblaslt_preference --datatype fp16 --trans_a N --trans_b N -m 2048 -n 2048 -k 2048 --alpha 1 --beta 1

Note that example_hipblaslt_preference also has the flag ``-V`` for correctness checks.

example_hipblaslt_groupedgemm
============================
example_hipblaslt_groupedgemm is a sample app for hipblaslt grouped gemm.

For more information:

.. code-block:: bash

   ./example_hipblaslt_groupedgemm --help
   Usage: ./example_hipblaslt_groupedgemm <options>
   options:
        -h, --help                              Show this help message
        -v, --verbose                           Verbose output
        -V, --validate                          Verify results
        --bench_count                           Number of benchmark runs (default is 1)
        --sync_count                            Number of sync runs (default is 1)
        --request_solutions                     Number of solutions to run (default is 1)
        --num_streams                           Run gemms by multi streams (default is 1)
        --grouped_gemm                          Run gemms by grouped gemm kernel (default is 0)
        --datatype              datatype        GEMM_STRIDED argument in out: fp32, fp16, bf16 (default is fp32)
        --trans_a               trans_a         GEMM_STRIDED argument trans_a: N or T (default is N)
        --trans_b               trans_b         GEMM_STRIDED argument trans_b: N or T (default is N)
        -m                      m               GEMM_STRIDED argument m
        -n                      n               GEMM_STRIDED argument n
        -k                      k               GEMM_STRIDED argument k
        --batch_count           batch_count     GEMM_STRIDED argument batch_count
        --lda                   lda             GEMM_STRIDED argument lda
        --ldb                   ldb             GEMM_STRIDED argument ldb
        --ldc                   ldc             GEMM_STRIDED argument ldc
        --ldd                   ldd             GEMM_STRIDED argument ldd
        --stride_a              stride_a        GEMM_STRIDED argument stride_a
        --stride_b              stride_b        GEMM_STRIDED argument stride_b
        --stride_c              stride_c        GEMM_STRIDED argument stride_c
        --stride_d              stride_d        GEMM_STRIDED argument stride_d
        --alpha                 alpha           GEMM_STRIDED argument alpha (default is 1)
        --beta                  beta            GEMM_STRIDED argument beta (default is 0)
        --act                   act             GEMM_STRIDED set activation type: relu, gelu, none (default is none)
        --bias                  bias            GEMM_STRIDED set bias: 0 or 1 (default is 0)
        --cpu_time              cpu_time        Bechmark timing using cpu time: 0 or 1 (default is 0)
        --all                   all             Get all solutions

For example, to solve 3 gemms by groupgemm sample app:

.. code-block:: bash

   ./example_hipblaslt_groupedgemm --datatype fp16 --trans_a T --trans_b N -m 1024 -n 1024 -k 1024 -m 512 -n 512 -k 512 -m 2048 -n 1024 -k 512 --sync_count 10 --grouped_gemm 1 --request_solutions 10

example_hipblaslt_groupedgemm_get_all_algos
============================
example_hipblaslt_groupedgemm_get_all_algos is a sample app for hipblaslt grouped gemm using hipblasLtExtGetAllAlgos API.

For more information:

.. code-block:: bash

   ./example_hipblaslt_groupedgemm_get_all_algos --help
   Usage: ./example_hipblaslt_groupedgemm_get_all_algos <options>
   options:
        -h, --help                              Show this help message
        -v, --verbose                           Verbose output
        -V, --validate                          Verify results
        --bench_count                           Number of benchmark runs (default is 1)
        --sync_count                            Number of sync runs (default is 1)
        --num_streams                           Run gemms by multi streams (default is 1)
        --grouped_gemm                          Run gemms by grouped gemm kernel (default is 0)
        --datatype              datatype        GEMM_STRIDED argument in out: fp32, fp16, bf16 (default is fp32)
        --trans_a               trans_a         GEMM_STRIDED argument trans_a: N or T (default is N)
        --trans_b               trans_b         GEMM_STRIDED argument trans_b: N or T (default is N)
        -m                      m               GEMM_STRIDED argument m
        -n                      n               GEMM_STRIDED argument n
        -k                      k               GEMM_STRIDED argument k
        --batch_count           batch_count     GEMM_STRIDED argument batch_count
        --lda                   lda             GEMM_STRIDED argument lda
        --ldb                   ldb             GEMM_STRIDED argument ldb
        --ldc                   ldc             GEMM_STRIDED argument ldc
        --ldd                   ldd             GEMM_STRIDED argument ldd
        --stride_a              stride_a        GEMM_STRIDED argument stride_a
        --stride_b              stride_b        GEMM_STRIDED argument stride_b
        --stride_c              stride_c        GEMM_STRIDED argument stride_c
        --stride_d              stride_d        GEMM_STRIDED argument stride_d
        --alpha                 alpha           GEMM_STRIDED argument alpha (default is 1)
        --beta                  beta            GEMM_STRIDED argument beta (default is 0)
        --act                   act             GEMM_STRIDED set activation type: relu, gelu, none (default is none)
        --bias                  bias            GEMM_STRIDED set bias: 0 or 1 (default is 0)
        --cpu_time              cpu_time        Bechmark timing using cpu time: 0 or 1 (default is 0)

For example, to solve 3 gemms by groupgemm sample app:

.. code-block:: bash

   ./example_hipblaslt_groupedgemm_get_all_algos --datatype fp16 --trans_a T --trans_b N -m 1024 -n 1024 -k 1024 -m 512 -n 512 -k 512 -m 2048 -n 1024 -k 512 --sync_count 10 --grouped_gemm 1
