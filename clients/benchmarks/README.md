# hipblaslt-bench test

```
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# run hipblaslt-bench
./clients/staging/hipblaslt-bench --help
--sizem |-m <value>        Specific matrix size: the number of rows or columns in matrix.                      (Default value is: 128)
--sizen |-n <value>        Specific matrix the number of rows or columns in matrix                             (Default value is: 128)
--sizek |-k <value>        Specific matrix size: the number of columns in A and rows in B.                     (Default value is: 128)
--lda <value>              Leading dimension of matrix A.
--ldb <value>              Leading dimension of matrix B.
--ldc <value>              Leading dimension of matrix C.
--ldd <value>              Leading dimension of matrix D.
--any_stride               Do not modify input strides based on leading dimensions
--stride_a <value>         Specific stride of strided_batched matrix A, second dimension * leading dimension.
--stride_b <value>         Specific stride of strided_batched matrix B, second dimension * leading dimension.
--stride_c <value>         Specific stride of strided_batched matrix C, second dimension * leading dimension.
--stride_d <value>         Specific stride of strided_batched matrix D, second dimension * leading dimension.
--alpha <value>            specifies the scalar alpha                                                          (Default value is: 1)
--beta <value>             specifies the scalar beta                                                           (Default value is: 0)
--function |-f <value>     BLASLt function to test. Options: matmul                                            (Default value is: matmul)
--precision |-r <value>    Precision of matrix A,B,C,D  Options: f32_r,f16_r,bf16_r                            (Default value is: f16_r)
--compute_type <value>     Precision of computation. Options: s,f32_r                                          (Default value is: f32_r)
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
--bias_vector              Apply bias vector
--scaleD_vector            Apply scaleD vector
--device <value>           Set default device to be used for subsequent program runs                           (Default value is: 0)
--c_noalias_d              C and D are stored in separate memory
--workspace <value>        Set fixed workspace memory size instead of using hipblaslt managed memory           (Default value is: 0)
--log_function_name        Function name precedes other itmes.
--function_filter <value>  Simple strstr filter on function name only without wildcards
--help |-h                 produces this help message
--version <value>          Prints the version number
```

# demo
Run fp32 GEMM with validation
```
./clients/staging/hipblaslt-bench --precision f32_r -v 1
transA,transB,M,N,K,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,d_type,compute_type,activation_type,bias_vector,hipblaslt-Gflops,us
N,N,128,128,128,1,128,16384,0,128,16384,128,16384,128,16384,f32_r,f32_r,none,0, 415.278, 10.1
```
