# hipblaslt-bench test

```
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# run hipblaslt-bench
./clients/staging/hipblaslt-bench --help

./clients/staging/hipblaslt-bench [ --data <path> | --yaml <path> ] <options> ...

--sizem |-m <value>                Specific matrix size: the number of rows in matrix C. For yaml file input use M.    (Default value is: 128)
--sizen |-n <value>                Specific matrix size: the number of columns in matrix C. For yaml file input use N.  (Default value is: 128)
--sizek |-k <value>                Specific matrix size: if transA == N, the number of columns in matrix A. For yaml file input use K.  (Default value is: 128)
--lda <value>                      Leading dimension of matrix A.
--ldb <value>                      Leading dimension of matrix B.
--ldc <value>                      Leading dimension of matrix C.
--ldd <value>                      Leading dimension of matrix D.
--lde <value>                      Leading dimension of matrix E.
--any_stride                       Do not modify input strides based on leading dimensions. Not supported in yaml input.
--stride_a <value>                 Specific stride of strided_batched matrix A, second dimension * leading dimension.
--stride_b <value>                 Specific stride of strided_batched matrix B, second dimension * leading dimension.
--stride_c <value>                 Specific stride of strided_batched matrix C, second dimension * leading dimension.
--stride_d <value>                 Specific stride of strided_batched matrix D, second dimension * leading dimension.
--stride_e <value>                 Specific stride of strided_batched matrix E, second dimension * leading dimension.
--alpha <value>                    specifies the scalar alpha.                                                         (Default value is: 1)
--beta <value>                     specifies the scalar beta.                                                          (Default value is: 0)
--function |-f <value>             BLASLt function to test. Options: matmul. Short form 'f' is not supported in yaml file input.  (Default value is: matmul)
--precision |-r <value>            Precision of matrix A,B,C,D  Options: f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r, f8_r, bf8_r. Not supported in yaml input file.  (Default value is: f16_r)
--a_type <value>                   Precision of matrix A. Options: f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r, f8_r, bf8_r.
--b_type <value>                   Precision of matrix B. Options: f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r, f8_r, bf8_r.
--c_type <value>                   Precision of matrix C. Options: f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r, f8_r, bf8_r.
--d_type <value>                   Precision of matrix D. Options: f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r, f8_r, bf8_r.
--compute_type <value>             Precision of computation. Options: s,f32_r, x,xf32_r, d,f64_r, i,i32_r, f32_f16_r, f32_bf16_r.  (Default value is: f32_r)
--compute_input_typeA <value>      Precision of computation input A. Options: f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r, f8_r, bf8_r. The default value indicates that the compute_input_typeA has no effect.
--compute_input_typeB <value>      Precision of computation input B. Options: f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r, f8_r, bf8_r. The default value indicates that the compute_input_typeB has no effect.
--scale_type <value>               Precision of scalar. Options: f16_r,bf16_r.
--initialization <value>           Initialize matrix data. Options: rand_int, trig_float, hpl(floating), special, zero.  (Default value is: hpl)
--transA <value>                   N = no transpose, T = transpose.                                                    (Default value is: N)
--transB <value>                   N = no transpose, T = transpose.                                                    (Default value is: N)
--batch_count <value>              Number of matrices. Only applicable to batched and strided_batched routines.        (Default value is: 1)
--HMM                              Parameter requesting the use of HipManagedMemory.
--verify |-v                       Validate GPU results with CPU. For yaml file input, set norm_check and allclose_check.
--iters |-i <value>                Iterations to run inside timing loop. Short form 'i' is not supported in yaml input.  (Default value is: 10)
--cold_iters |-j <value>           Cold Iterations to run before entering the timing loop. Short form 'j' is not supported in yaml input.  (Default value is: 2)
--algo_method <value>              Use different algorithm search API. Options: heuristic, all, index. For yaml file input use: 0 (heuristic), 1 (all), 2 (algo index).  (Default value is: heuristic)
--solution_index <value>           Used with --algo_method 2 (index).  Specify solution index to use in benchmark.     (Default value is: -1)
--requested_solution_num <value>   Requested number of solutions. Set to -1 to get all solutions. Only valid when algo_method is set to 0 (heuristic).  (Default value is: 1)
--activation_type <value>          Options: none, relu, gelu.                                                          (Default value is: none)
--activation_arg1 <value>          Threshold when activation type is relu.                                             (Default value is: 0)
--activation_arg2 <value>          Upperbound when activation type is relu.                                            (Default value is: inf)
--bias_type <value>                Precision of bias vector. Options: f16_r,bf16_r,f32_r,default(same with D type).
--bias_source <value>              Choose bias source: a, b, d.                                                        (Default value is: d)
--bias_vector                      Apply bias vector.
--scaleA <value>                   Apply scale for A buffer. s = scalar, v = vector.                                   (Default value is: )
--scaleB <value>                   Apply scale for B buffer. s = scalar, v = vector.                                   (Default value is: )
--scaleAlpha_vector                Apply scaleAlpha vector.
--amaxScaleA                       Apply scale for A buffer by abs max of A buffer.
--amaxScaleB                       Apply scale for B buffer by abs max of B buffer.
--amaxD                            Output Amax of intermediate D matrix.
--use_e                            Apply AUX output/ gradient input.
--gradient                         Enable gradient.
--grouped_gemm                     Use grouped_gemm.
--use_user_args                    Use UserArguments located in device memory for grouped gemm.
--device <value>                   Set default device to be used for subsequent program runs. Not supported in yaml input.  (Default value is: 0)
--c_equal_d                        C and D are stored in same memory.
--user_allocated_workspace <value> Set fixed workspace memory size (bytes) instead of using hipblaslt managed memory.  (Default value is: 100)
--log_function_name                Function name precedes other items. Not supported in yaml input file.
--function_filter <value>          Simple strstr filter on function name only without wildcards. Not supported in yaml input file.
--api_method <value>               Use extension API. c: C style API. mix: declaration with C hipblasLtMatmul Layout/Desc but set, initialize, and run the problem with C++ extension API. cpp: Using C++ extension API only. Options: c, mix, cpp. For yaml file input use: 0 (c), 1 (mix), 2 (cpp).  (Default value is: c)
--print_kernel_info                Print solution, kernel name and solution index.
--rotating <value>                 Use rotating memory blocks for each iteration, size in MB.                          (Default value is: 0)
--use_gpu_timer                    Use hipEventElapsedTime to profile elapsed time.
--skip_slow_solution_ratio <value> Specifies a ratio to skip slow solution when warm up stage. Skip condition: (current solution's warm up time * ratio) > best solution's warm up time. Ratio range: 0 ~ 1. 0 means no skip.  (Default value is: 0)
--splitk <value>                   [Tuning parameter] Set split K for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp). For yaml input file use 'gsu_vector'.
--wgm <value>                      [Tuning parameter] Set workgroup mapping for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp). For yaml input file use 'wgm_vector'.
--flush                            Flush icache, only works for gemm.
--help |-h                         Produces this help message.
--version <value>                  Prints the version number.
```

# yaml input file
Accepted arguments specified through a yaml input file correspond to the fields in the C++ struct Arguments defined in hipblaslt_arguments.hpp.

# demo
Run fp32 GEMM with validation
```
./clients/staging/hipblaslt-bench --precision f32_r -v
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,hipblaslt-Gflops,hipblaslt-GB/s,us,CPU-Gflops,CPU-us,norm_error,atol,rtol
    N,N,0,1,128,128,128,1,128,16384,0,128,16384,128,16384,128,16384,f32_r,f32_r,f32_r,f32_r,f32_r,0,0,0,0,0,none,0,non-supported type,403.298,17.6063,10.4,0.0960059,43688,2.44879e-07,1e-05,1e-05
```
Show the frequency with environment variable
```
HIPBLASLT_BENCH_FREQ=1 ./clients/staging/hipblaslt-bench -m 16 -n 16 -k 4096 --transA T --transB N --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --activation_type none --compute_type f32_r
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,lowest-avg-freq,lowest-median-freq,avg-MCLK,median-MCLK,hipblaslt-Gflops,hipblaslt-GB/s,us
    T,N,0,1,16,16,4096,1,4096,65536,0,4096,65536,16,256,16,256,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,none,0,non-supported type,136,136,900,900,192.399,22.442,10.9
```
Show the multi-XCD frequencies with environment variable
```
HIPBLASLT_BENCH_FREQ_ALL=1 ./clients/staging/hipblaslt-bench -m 16 -n 16 -k 4096 --transA T --transB N --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --activation_type none --compute_type f32_r
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,avg-freq_0,avg-freq_1,avg-freq_2,avg-freq_3,avg-freq_4,avg-freq_5,avg-freq_6,avg-freq_7,median-freq_0,median-freq_1,median-freq_2,median-freq_3,median-freq_4,median-freq_5,median-freq_6,median-freq_7,avg-MCLK,median-MCLK,hipblaslt-Gflops,hipblaslt-GB/s,us
    T,N,0,1,16,16,4096,1,4096,65536,0,4096,65536,16,256,16,256,bf16_r,bf16_r,bf16_r,bf16_r,f32_r,0,0,0,0,0,none,0,non-supported type,143,141,143,143,142,143,141,141,143,141,143,143,142,143,141,141,900,900,148.734,17.3488,14.1
```
