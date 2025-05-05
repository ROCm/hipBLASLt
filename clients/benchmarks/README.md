# hipblaslt-bench test

```
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# run hipblaslt-bench
./clients/staging/hipblaslt-bench --help

./clients/staging/hipblaslt-bench [ --data <path> | --yaml <path> ] <options> ...

--sizem |-m <value>                Specific matrix size: the number of rows or columns in matrix.                      (Default value is: 128)
--sizen |-n <value>                Specific matrix the number of rows or columns in matrix                             (Default value is: 128)
--sizek |-k <value>                Specific matrix size: the number of columns in A and rows in B.                     (Default value is: 128)
--lda <value>                      Leading dimension of matrix A.
--ldb <value>                      Leading dimension of matrix B.
--ldc <value>                      Leading dimension of matrix C.
--ldd <value>                      Leading dimension of matrix D.
--lde <value>                      Leading dimension of matrix E.
--any_stride                       Do not modify input strides based on leading dimensions
--stride_a <value>                 Specific stride of strided_batched matrix A, second dimension * leading dimension.
--stride_b <value>                 Specific stride of strided_batched matrix B, second dimension * leading dimension.
--stride_c <value>                 Specific stride of strided_batched matrix C, second dimension * leading dimension.
--stride_d <value>                 Specific stride of strided_batched matrix D, second dimension * leading dimension.
--stride_e <value>                 Specific stride of strided_batched matrix E, second dimension * leading dimension.
--alpha <value>                    Specifies the scalar alpha                                                          (Default value is: 1)
--beta <value>                     Specifies the scalar beta                                                           (Default value is: 0)
--function |-f <value>             BLASLt function to test. Options: matmul                                            (Default value is: matmul)
--precision |-r <value>            Precision of matrix A,B,C,D  Options: f32_r,f16_r,bf16_r,f64_r,i32_r,i8_r           (Default value is: f16_r)
--a_type <value>                   Precision of matrix A. Options: f32_r,f16_r,bf16_r,i8_r
--b_type <value>                   Precision of matrix B. Options: f32_r,f16_r,bf16_r,i8_r
--c_type <value>                   Precision of matrix C. Options: f32_r,f16_r,bf16_r,i8_r
--d_type <value>                   Precision of matrix D. Options: f32_r,f16_r,bf16_r,i8_r
--compute_type <value>             Precision of computation. Options: s,f32_r,x,xf32_r,f64_r,i32_r                     (Default value is: f32_r)
--compute_input_typeA <value>      Precision of computation input A. Options: f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, The default value indicates that the compute_input_typeA has no effect.
--compute_input_typeB <value>      Precision of computation input B. Options: f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, The default value indicates that the compute_input_typeA has no effect.
--scale_type <value>               Precision of scalar. Options: f16_r,bf16_r
--initialization <value>           Initialize matrix data. Options: rand_int, trig_float, hpl(floating), special, zero, norm_dist  (Default value is: hpl)
--transA <value>                   N = no transpose, T = transpose                                                     (Default value is: N)
--transB <value>                   N = no transpose, T = transpose                                                     (Default value is: N)
--swizzleA                         Enable tensor swizzling for A
--batch_count <value>              Number of matrices. Only applicable to batched and strided_batched routines         (Default value is: 1)
--HMM                              Parameter requesting the use of HipManagedMemory
--verify |-v                       Validate GPU results with CPU?
--iters |-i <value>                Iterations to run inside timing loop                                                (Default value is: 10)
--cold_iters |-j <value>           Cold Iterations to run before entering the timing loop                              (Default value is: 2)
--bench_time <value>               Minimum time to run (with a minimum of iters iterations) inside timing loop (in seconds). Only supported with --use_gpu_timer.  (Default value is: 0)
--cold_bench_time <value>          Minimum cold time to run (with a minimum of cold_iters iterations) before entering the main timing loop (in seconds). Only supported with --use_gpu_timer.  (Default value is: 0)
--algo_method <value>              Use different algorithm search API. Options: heuristic, all, index.                 (Default value is: heuristic)
--solution_index <value>           Used with --algo_method 2.  Specify solution index to use in benchmark.             (Default value is: -1)
--requested_solution <value>       Requested solution num. Set to -1 to get all solutions. Only valid when algo_method is set to heuristic.  (Default value is: 1)
--activation_type <value>          Options: none, gelu, relu, swish                                                    (Default value is: none)
--activation_arg1 <value>          Reserved.                                                                           (Default value is: 0)
--activation_arg2 <value>          Reserved.                                                                           (Default value is: inf)
--bias_type <value>                Precision of bias vector. Options: f16_r,bf16_r,f32_r,default(same with D type)
--bias_source <value>              Choose bias source: a, b, d                                                         (Default value is: d)
--bias_vector                      Apply bias vector
--scaleA <value>                   Apply scale for A buffer. 0 = None, 1 = scalar, 2 = vector, 3 = block.              (Default value is: 0)
--scaleB <value>                   Apply scale for B buffer. 0 = None, 1 = scalar, 2 = vector, 3 = block.              (Default value is: 0)
--scaleC <value>                   Apply scale for C buffer. 0 = None, 1 = scalar                                      (Default value is: 0)
--scaleD <value>                   Apply scale for D buffer. 0 = None, 1 = scalar                                      (Default value is: 0)
--scaleAlpha_vector                Apply scaleAlpha vector
--scaleABlockRowSize <value>       Set the row size of scale block for A                                               (Default value is: 32)
--scaleABlockColSize <value>       Set the column size of scale block for A                                            (Default value is: 1)
--scaleBBlockRowSize <value>       Set the row size of scale block for B                                               (Default value is: 1)
--scaleBBlockColSize <value>       Set the column size of scale block for B                                            (Default value is: 32)
--amaxScaleA                       Apply scale for A buffer by abs max of A buffer
--amaxScaleB                       Apply scale for B buffer by abs max of B buffer
--amaxD                            Output Amax of intermediate D matrix
--use_e                            Apply AUX output/ gradient input
--aux_type <value>                 Used with --use_e. Precision of AUX output (matrix E).Options: f16_r, default (same with D type)
--gradient                         Enable gradient
--grouped_gemm                     Use grouped_gemm.
--use_user_args                    Use UserArguments located in device memory for grouped gemm.
--device <value>                   Set default device to be used for subsequent program runs                           (Default value is: 0)
--c_equal_d                        C and D are stored in same memory
--workspace <value>                Set fixed workspace memory size (bytes) instead of using hipblaslt managed memory   (Default value is: 134217728)
--log_function_name                Function name precedes other items.
--function_filter <value>          Simple strstr filter on function name only without wildcards
--api_method <value>               Use extension API. c: C style API. mix: declaration with C hipblasLtMatmul Layout/Desc but set, initialize, and run the problem with C++ extension API. cpp: Using C++ extension API only. Options: c, mix, cpp.  (Default value is: c)
--print_kernel_info                Print solution, kernel name and solution index.
--rotating <value>                 Use rotating memory blocks for each iteration, size in MB.                          (Default value is: 0)
--use_gpu_timer                    Use hipEventElapsedTime to profile elapsed time.
--skip_slow_solution_ratio <value> Specifies a ratio to skip slow solution when warm up stage. Skip condition: (current solution's warm up time * ratio) > best solution's warm up time. Ratio range: 0 ~ 1. 0 means no skip.  (Default value is: 0)
--splitk <value>                   [Tuning parameter] Set split K for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)
--wgm <value>                      [Tuning parameter] Set workgroup mapping for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)
--flush                            Flush icache, only works for gemm.
--help |-h                         Produces this help message
--version <value>                  Prints the version number
```

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
Show the efficiency and other performance related args with environment variable
```
HIPBLASLT_BENCH_EFF=1 ./hipblaslt-bench -m 4096 -n 4864 -k 32896 --alpha 1 --beta 0 --transA N --transB N --initialization trig_float --iters 466 --cold_iters 94 --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r --function matmul --rotating 512 --use_gpu_timer
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,total_gran,tiles_per_cu,num_cu's,tile_0_granularity,tile_1_granularity,cu_gran,wave_gran,mem_read_bytes,mem_write_bytes,lowest_avg_freq,lowest_median_freq,avg_MCLK,median_MCLK,hipblaslt-Gflops,efficiency,hipblaslt-GB/s,us
    N,N,0,1,4096,4864,32896,1,4096,134742016,0,32896,160006144,4096,19922944,4096,19922944,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,none,0,f16_r,512,0,0,0,0,0,0,0,0,0,1096,1067,908.602,900,555009,81.3366,248.176,2361.71
```
