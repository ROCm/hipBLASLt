# Sample Unit test
Source code are in

clients/samples/example_hipblaslt_preference.cpp

clients/samples/example_hipblaslt_groupdgemm.cpp

# Go to hipBLASLt build directory
cd build/release

# Run gemm sample test
```
./clients/staging/example_hipblaslt_preference
Usage: ./example_hipblaslt_preference <options>
options:
        -h, --help                              Show this help message
        -v, --verbose                           Verbose output
        -V, --validate                          Verify results
        -m                      m               GEMM_STRIDED argument m
        -n                      n               GEMM_STRIDED argument n
        -k                      k               GEMM_STRIDED argument k
        --lda                   lda             GEMM_STRIDED argument lda
        --ldb                   ldb             GEMM_STRIDED argument ldb
        --ldc                   ldc             GEMM_STRIDED argument ldc
        --ldd                   ldd             GEMM_STRIDED argument ldd
        --trans_a               trans_a         GEMM_STRIDED argument trans_a
        --trans_b               trans_b         GEMM_STRIDED argument trans_b
        --datatype              datatype        GEMM_STRIDED argument in out datatype:fp32
        --stride_a              stride_a        GEMM_STRIDED argument stride_a
        --stride_b              stride_b        GEMM_STRIDED argument stride_b
        --stride_c              stride_c        GEMM_STRIDED argument stride_c
        --stride_d              stride_d        GEMM_STRIDED argument stride_d
        --alpha                 alpha           GEMM_STRIDED argument alpha
        --beta                  beta            GEMM_STRIDED argument beta
        --batch_count           batch           GEMM_STRIDED argument batch count
        --act                   act             GEMM_STRIDED set activation type: relu or gelu
        --bias                  bias            GEMM_STRIDED enable bias: 0 or 1 (default is 0)
        --scaleD                scaleD          GEMM_STRIDED enable scaleD: 0 or 1 (default is 0)
        --header                header          Print header for output (default is enabled)
        --timing                timing          Bechmark GPU kernel performance:0 or 1 (default is 1)
        --iters                 iters           Iterations to run inside timing loop (default is 3)
        --cold_iters            cold_iters      Cold Iterations to run before entering the timing loop (default is 0)

example:
./example_hipblaslt_preference --datatype fp16 --trans_a T --trans_b N -m 1024 -n 1024 -k 1024

result:
transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count, alpha, beta, bias, scaleD, activationType, us, tflops
TN, 1024, 1024, 1024, 1024, 1024, 1024, 1048576, 1048576, 1048576, 1, 2.00, 3.00, 0, 0, none, 92.333333, 23.257946
```

# Run grouped gemm sample test
```
./clients/staging/example_hipblaslt_groupedgemm
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
        --scaleD                scaleD          GEMM_STRIDED enable scaleD: 0 or 1 (default is 0)
        --cpu_time              cpu_time        Bechmark timing using cpu time: 0 or 1 (default is 0)

example:
./example_hipblaslt_groupedgemm --datatype fp16 --trans_a T --trans_b N -m 1024 -n 1024 -k 1024 -m 512 -n 512 -k 512 -m 2048 -n 1024 -k 512 --sync_count 10 --grouped_gemm 1 --request_solutions 10

result:
0, TN, 1024, 1024, 1024, 1024, 1024, 1024, 1048576, 1048576, 1048576, 1, 1, 0, 0, 0, none
1, TN, 512, 512, 512, 512, 512, 512, 262144, 262144, 262144, 1, 1, 0, 0, 0, none
2, TN, 2048, 1024, 512, 512, 512, 2048, 1048576, 524288, 2097152, 1, 1, 0, 0, 0, none
      Sol 0: Perf: 0.112191 ms, 40.675121 Tflops *
      Sol 1: Perf: 0.095744 ms, 47.662743 Tflops *
      Sol 2: Perf: 0.091152 ms, 50.063824 Tflops *
      Sol 3: Perf: 0.072816 ms, 62.670590 Tflops *
      Sol 4: Perf: 0.093104 ms, 49.014246 Tflops
      Sol 5: Perf: 0.082560 ms, 55.273975 Tflops
      Sol 6: Perf: 0.090880 ms, 50.213663 Tflops
      Sol 7: Perf: 0.071136 ms, 64.150669 Tflops *
      Sol 8: Perf: 0.093200 ms, 48.963706 Tflops
      Sol 9: Perf: 0.096768 ms, 47.158376 Tflops
```
