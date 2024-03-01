# hipblaslt-bench test

```
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# run hipblaslt-bench
./clients/staging/hipblaslt-bench --help

./clients/staging/hipblaslt-bench [ --data <path> | --yaml <path> ] <options> ...

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
--initialization <value>   Initialize matrix data.Options: rand_int, trig_float, hpl(floating), special, zero  (Default value is: hpl)
--transA <value>           N = no transpose, T = transpose, C = conjugate transpose                            (Default value is: N)
--transB <value>           N = no transpose, T = transpose, C = conjugate transpose                            (Default value is: N)
--batch_count <value>      Number of matrices. Only applicable to batched and strided_batched routines         (Default value is: 1)
--HMM                      Parameter requesting the use of HipManagedMemory
--verify |-v               Validate GPU results with CPU?
--iters |-i <value>        Iterations to run inside timing loop                                                (Default value is: 10)
--cold_iters |-j <value>   Cold Iterations to run before entering the timing loop                              (Default value is: 2)
--algo_method <value>      Use different algorithm search API. Options: heuristic, all, index.                 (Default value is: heuristic)
--solution_index <value>   Used with --algo_method 2.  Specify solution index to use in benchmark.             (Default value is: -1)
--requested_solution <value> Requested solution num. Set to -1 to get all solutions. Only valid when algo_method is set to heuristic.  (Default value is: 1)
--activation_type <value>  Options: None, gelu, relu                                                           (Default value is: none)
--activation_arg1 <value>  Reserved.                                                                           (Default value is: 0)
--activation_arg2 <value>  Reserved.                                                                           (Default value is: inf)
--bias_type <value>        Precision of bias vector.Options: f16_r,bf16_r,f32_r,default(same with D type)
--bias_source <value>      Choose bias source: a, b, d                                                         (Default value is: d)
--bias_vector              Apply bias vector
--scaleA                   Apply scale for A buffer
--scaleB                   Apply scale for B buffer
--scaleAlpha_vector        Apply scaleAlpha vector
--amaxScaleA               Apple scale for A buffer by abs max of A buffer
--amaxScaleB               Apple scale for B buffer by abs max of B buffer
--use_e                    Apply AUX output/ gradient input
--gradient                 Enable gradient
--grouped_gemm             Use grouped_gemm.
--use_user_args            Use UserArguments located in device memory for grouped gemm.
--device <value>           Set default device to be used for subsequent program runs                           (Default value is: 0)
--c_noalias_d              C and D are stored in separate memory
--workspace <value>        Set fixed workspace memory size instead of using hipblaslt managed memory           (Default value is: 0)
--log_function_name        Function name precedes other items.
--function_filter <value>  Simple strstr filter on function name only without wildcards
--api_method <value>       Use extension API. c: C style API. mix: declaration with C hipblasLtMatmul Layout/Desc but set, initialize, and run the problem with C++ extension API. cpp: Using C++ extension API only. Options: c, mix, cpp.  (Default value is: c)
--print_kernel_info        Print solution, kernel name and solution index.
--rotating <value>         Use rotating memory blocks for each iteration, size in MB.                          (Default value is: 0)
--use_gpu_timer            Use hipEventElapsedTime to profile elapsed time.                                    (Default value is: false)
--splitk <value>           [Tuning parameter] Set split K for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)
--wgm <value>              [Tuning parameter] Set workgroup mapping for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)
--flush                    Flush icache
--help |-h                 produces this help message
--version <value>          Prints the version number
```

# demo
Run fp32 GEMM with validation
```
./clients/staging/hipblaslt-bench --precision f32_r -v
transA,transB,M,N,K,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,d_type,compute_type,activation_type,bias_vector,hipblaslt-Gflops,us
N,N,128,128,128,1,128,16384,0,128,16384,128,16384,128,16384,f32_r,f32_r,none,0, 415.278, 10.1
```
