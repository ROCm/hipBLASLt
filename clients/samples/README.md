# Sample Unit test
Source code is in clients/samples/example_hipblaslt_preference.cpp
```
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# Run sample tests
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
        --act                   act             GEMM_STRIDED set activation type: relu or gelu
        --bias                  bias            GEMM_STRIDED enable bias: 0 or 1 (default is 0)
        --header                header          Print header for output (default is enabled)
        --timing                timing          Bechmark GPU kernel performance:0 or 1 (default is 1)
```
