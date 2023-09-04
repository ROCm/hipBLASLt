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
	-h, --help				Show this help message
	-v, --verbose				Verbose output
	-V, --validate				Verify results
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
	--in_datatype   datatype                GEMM_STRIDED argument in datatype:fp32,fp16,bf16
	--out_datatype  datatype                GEMM_STRIDED argument out datatype:fp32,fp16,bf16
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

Example 1:
./example_hipblaslt_preference --datatype fp16 --trans_a T --trans_b N -m 1024 -n 1024 -k 1024
result:
transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count, alpha, beta, bias, activationType, us, tflops
TN, 1024, 1024, 1024, 1024, 1024, 1024, 1048576, 1048576, 1048576, 1, 2.00, 3.00, 0, 0, none, 92.333333, 23.257946

Example 2: (run 10 solutions for single problem)
./example_hipblaslt_preference --datatype fp16 --trans_a T --trans_b N -m 1024 -n 1024 -k 1024 -s 10
result:
transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count, alpha, beta, bias, activationType, us, tflops, best solution
  Solution  0:    91.667 us,  23.427 Tflops *
  Solution  1:    46.000 us,  46.684 Tflops *
  Solution  2:    49.667 us,  43.238 Tflops
  Solution  3:    45.667 us,  47.025 Tflops *
  Solution  4:    52.667 us,  40.775 Tflops
  Solution  5:    40.000 us,  53.687 Tflops *
  Solution  6:    44.333 us,  48.439 Tflops
  Solution  7:    49.000 us,  43.826 Tflops
  Solution  8:    44.000 us,  48.806 Tflops
  Solution  9:    64.000 us,  33.554 Tflops
TN, 1024, 1024, 1024, 1024, 1024, 1024, 1048576, 1048576, 1048576, 1, 2.000, 3.000, 0, 0, none, 40.000, 53.687, solution 5
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
        --cpu_time              cpu_time        Bechmark timing using cpu time: 0 or 1 (default is 0)
        --all                   all             Get all solutions

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

# Run grouped gemm get all algos sample test
```
./clients/staging/example_hipblaslt_groupedgemm_get_all_algos
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

example:
./example_hipblaslt_groupedgemm_get_all_algos --datatype fp16 --trans_a T --trans_b N -m 1024 -n 1024 -k 1024 -m 512 -n 512 -k 512 -m 2048 -n 1024 -k 512 --sync_count 10 --grouped_gemm 1

result:
index, transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count, alpha, beta, bias, activationType
0, TN, 1024, 1024, 1024, 1024, 1024, 1024, 1048576, 1048576, 1048576, 1, 1, 0, 0, 0, none
1, TN, 512, 512, 512, 512, 512, 512, 262144, 262144, 262144, 1, 1, 0, 0, 0, none
2, TN, 2048, 1024, 512, 512, 512, 2048, 1048576, 524288, 2097152, 1, 1, 0, 0, 0, none
Is supported 236 / Total solutions: 238
      Sol 0: Perf: 0.168288 ms, 27.116706 Tflops *
      Sol 1: Perf: 0.203936 ms, 22.376696 Tflops
      Sol 2: Perf: 0.145472 ms, 31.369717 Tflops *
      Sol 3: Perf: 0.117936 ms, 38.693957 Tflops *
      Sol 4: Perf: 0.158896 ms, 28.719486 Tflops
      Sol 5: Perf: 0.113216 ms, 40.307156 Tflops *
      Sol 6: Perf: 0.142832 ms, 31.949532 Tflops
      Sol 7: Perf: 0.119968 ms, 38.038565 Tflops
      Sol 8: Perf: 0.134912 ms, 33.825106 Tflops
      Sol 9: Perf: 0.105520 ms, 43.246888 Tflops *
      Sol 10: Perf: 0.103568 ms, 44.062028 Tflops *
      Sol 11: Perf: 0.129776 ms, 35.163768 Tflops
      Sol 12: Perf: 0.105808 ms, 43.129213 Tflops
      Sol 13: Perf: 0.171440 ms, 26.618137 Tflops
      Sol 14: Perf: 0.118064 ms, 38.652039 Tflops
      Sol 15: Perf: 0.103840 ms, 43.946566 Tflops
      Sol 16: Perf: 0.107760 ms, 42.347953 Tflops
      Sol 17: Perf: 0.136448 ms, 33.444335 Tflops
      Sol 18: Perf: 0.100672 ms, 45.329502 Tflops *
      Sol 19: Perf: 0.127696 ms, 35.736541 Tflops
      Sol 20: Perf: 0.095040 ms, 48.015701 Tflops *
      Sol 21: Perf: 0.121872 ms, 37.444319 Tflops
      Sol 22: Perf: 0.101824 ms, 44.816664 Tflops
      Sol 23: Perf: 0.117952 ms, 38.688740 Tflops
      Sol 24: Perf: 0.091328 ms, 49.967292 Tflops *
      Sol 25: Perf: 0.110544 ms, 41.281400 Tflops
      Sol 26: Perf: 0.104672 ms, 43.597291 Tflops
      Sol 27: Perf: 0.113936 ms, 40.052441 Tflops
      Sol 28: Perf: 0.093296 ms, 48.913271 Tflops
      Sol 29: Perf: 0.138400 ms, 32.972636 Tflops
      Sol 30: Perf: 0.098224 ms, 46.459285 Tflops
      Sol 31: Perf: 0.104576 ms, 43.637273 Tflops
      Sol 32: Perf: 0.110592 ms, 41.263520 Tflops
      Sol 33: Perf: 0.172736 ms, 26.418427 Tflops
      Sol 34: Perf: 0.100416 ms, 45.445115 Tflops
      Sol 35: Perf: 0.116912 ms, 39.032866 Tflops
      Sol 36: Perf: 0.100352 ms, 45.474094 Tflops
      Sol 37: Perf: 0.106320 ms, 42.921477 Tflops
      Sol 38: Perf: 0.092864 ms, 49.140815 Tflops
      Sol 39: Perf: 0.120224 ms, 37.957597 Tflops
      Sol 40: Perf: 0.121424 ms, 37.582472 Tflops
      Sol 41: Perf: 0.131696 ms, 34.651114 Tflops
      Sol 42: Perf: 0.102640 ms, 44.460365 Tflops
      Sol 43: Perf: 0.097792 ms, 46.664519 Tflops
      Sol 44: Perf: 0.121984 ms, 37.409940 Tflops
      Sol 45: Perf: 0.169456 ms, 26.929784 Tflops
      Sol 46: Perf: 0.119504 ms, 38.186288 Tflops
      Sol 47: Perf: 0.109216 ms, 41.783358 Tflops
      Sol 48: Perf: 0.090064 ms, 50.668615 Tflops *
      Sol 49: Perf: 0.103168 ms, 44.232822 Tflops
      Sol 50: Perf: 0.094528 ms, 48.275775 Tflops
      Sol 51: Perf: 0.099648 ms, 45.795318 Tflops
      Sol 52: Perf: 0.081008 ms, 56.332879 Tflops *
      Sol 53: Perf: 0.090448 ms, 50.453440 Tflops
      Sol 54: Perf: 0.096000 ms, 47.535544 Tflops
      Sol 55: Perf: 0.102128 ms, 44.683303 Tflops
      Sol 56: Perf: 0.102432 ms, 44.550646 Tflops
      Sol 57: Perf: 0.113600 ms, 40.170903 Tflops
      Sol 58: Perf: 0.118144 ms, 38.625866 Tflops
      Sol 59: Perf: 0.128240 ms, 35.584945 Tflops
      Sol 60: Perf: 0.110592 ms, 41.263520 Tflops
      Sol 61: Perf: 0.137056 ms, 33.295973 Tflops
      Sol 62: Perf: 0.087520 ms, 52.141375 Tflops
      Sol 63: Perf: 0.116544 ms, 39.156152 Tflops
      Sol 64: Perf: 0.103088 ms, 44.267190 Tflops
      Sol 65: Perf: 0.153472 ms, 29.734491 Tflops
      Sol 66: Perf: 0.101472 ms, 44.972125 Tflops
      Sol 67: Perf: 0.108384 ms, 42.104141 Tflops
      Sol 68: Perf: 0.097392 ms, 46.856128 Tflops
      Sol 69: Perf: 0.114400 ms, 39.889990 Tflops
      Sol 70: Perf: 0.117776 ms, 38.746555 Tflops
      Sol 71: Perf: 0.115248 ms, 39.596475 Tflops
      Sol 72: Perf: 0.114208 ms, 39.957013 Tflops
      Sol 73: Perf: 0.112896 ms, 40.421369 Tflops
      Sol 74: Perf: 0.101440 ms, 44.986358 Tflops
      Sol 75: Perf: 0.112736 ms, 40.478772 Tflops
      Sol 76: Perf: 0.118352 ms, 38.557948 Tflops
      Sol 77: Perf: 0.223887 ms, 20.382571 Tflops
      Sol 78: Perf: 0.123120 ms, 37.064765 Tflops
      Sol 79: Perf: 0.139344 ms, 32.749257 Tflops
      Sol 80: Perf: 0.087360 ms, 52.236874 Tflops
      Sol 81: Perf: 0.108336 ms, 42.122757 Tflops
      Sol 82: Perf: 0.110192 ms, 41.413306 Tflops
      Sol 83: Perf: 0.095744 ms, 47.662645 Tflops
      Sol 84: Perf: 0.095920 ms, 47.575241 Tflops
      Sol 85: Perf: 0.119136 ms, 38.304242 Tflops
      Sol 86: Perf: 0.119408 ms, 38.216958 Tflops
      Sol 87: Perf: 0.112704 ms, 40.490230 Tflops
      Sol 88: Perf: 0.112928 ms, 40.409951 Tflops
      Sol 89: Perf: 0.111472 ms, 40.937769 Tflops
      Sol 90: Perf: 0.114064 ms, 40.007492 Tflops
      Sol 91: Perf: 0.120048 ms, 38.013246 Tflops
      Sol 92: Perf: 0.121408 ms, 37.587425 Tflops
      Sol 93: Perf: 0.105680 ms, 43.181411 Tflops
      Sol 94: Perf: 0.110816 ms, 41.180109 Tflops
      Sol 95: Perf: 0.107168 ms, 42.581887 Tflops
      Sol 96: Perf: 0.135200 ms, 33.753053 Tflops
      Sol 97: Perf: 0.102480 ms, 44.529777 Tflops
      Sol 98: Perf: 0.123504 ms, 36.949491 Tflops
      Sol 99: Perf: 0.115152 ms, 39.629453 Tflops
      Sol 100: Perf: 0.113392 ms, 40.244593 Tflops
      Sol 101: Perf: 0.112224 ms, 40.663451 Tflops
      Sol 102: Perf: 0.125296 ms, 36.421034 Tflops
      Sol 103: Perf: 0.141856 ms, 32.169328 Tflops
      Sol 104: Perf: 0.113408 ms, 40.238916 Tflops
      Sol 105: Perf: 0.119680 ms, 38.130132 Tflops
      Sol 106: Perf: 0.116928 ms, 39.027561 Tflops
      Sol 107: Perf: 0.147792 ms, 30.877261 Tflops
      Sol 108: Perf: 0.135008 ms, 33.801055 Tflops
      Sol 109: Perf: 0.107648 ms, 42.391974 Tflops
      Sol 110: Perf: 0.093040 ms, 49.047857 Tflops
      Sol 111: Perf: 0.106288 ms, 42.934398 Tflops
      Sol 112: Perf: 0.117424 ms, 38.862706 Tflops
      Sol 113: Perf: 0.116400 ms, 39.204590 Tflops
      Sol 114: Perf: 0.111312 ms, 40.996614 Tflops
      Sol 115: Perf: 0.104656 ms, 43.603915 Tflops
      Sol 116: Perf: 0.111728 ms, 40.843968 Tflops
      Sol 117: Perf: 0.118960 ms, 38.360913 Tflops
      Sol 118: Perf: 0.120368 ms, 37.912185 Tflops
      Sol 119: Perf: 0.133984 ms, 34.059389 Tflops
      Sol 120: Perf: 0.144384 ms, 31.606102 Tflops
      Sol 121: Perf: 0.146544 ms, 31.140217 Tflops
      Sol 122: Perf: 0.130736 ms, 34.905561 Tflops
      Sol 123: Perf: 0.123792 ms, 36.863530 Tflops
      Sol 124: Perf: 0.151584 ms, 30.104839 Tflops
      Sol 125: Perf: 0.143520 ms, 31.796372 Tflops
      Sol 126: Perf: 0.183136 ms, 24.918162 Tflops
      Sol 127: Perf: 0.098496 ms, 46.330985 Tflops
      Sol 128: Perf: 0.093136 ms, 48.997301 Tflops
      Sol 129: Perf: 0.098880 ms, 46.151010 Tflops
      Sol 130: Perf: 0.114016 ms, 40.024337 Tflops
      Sol 131: Perf: 0.101760 ms, 44.844894 Tflops
      Sol 132: Perf: 0.106336 ms, 42.915020 Tflops
      Sol 133: Perf: 0.099472 ms, 45.876392 Tflops
      Sol 134: Perf: 0.093792 ms, 48.654602 Tflops
      Sol 135: Perf: 0.086768 ms, 52.593276 Tflops
      Sol 136: Perf: 0.100688 ms, 45.322348 Tflops
      Sol 137: Perf: 0.101456 ms, 44.979222 Tflops
      Sol 138: Perf: 0.222847 ms, 20.477703 Tflops
      Sol 139: Perf: 0.100400 ms, 45.452307 Tflops
      Sol 140: Perf: 0.094544 ms, 48.267603 Tflops
      Sol 141: Perf: 0.093824 ms, 48.638007 Tflops
      Sol 142: Perf: 0.115680 ms, 39.448603 Tflops
      Sol 143: Perf: 0.083424 ms, 54.701450 Tflops
      Sol 144: Perf: 0.109616 ms, 41.630884 Tflops
      Sol 145: Perf: 0.078288 ms, 58.290014 Tflops *
      Sol 146: Perf: 0.103648 ms, 44.027973 Tflops
      Sol 147: Perf: 0.092480 ms, 49.344859 Tflops
      Sol 148: Perf: 0.095664 ms, 47.702552 Tflops
      Sol 149: Perf: 0.082208 ms, 55.510582 Tflops
      Sol 150: Perf: 0.103376 ms, 44.143819 Tflops
      Sol 151: Perf: 0.091952 ms, 49.628260 Tflops
      Sol 152: Perf: 0.106048 ms, 43.031566 Tflops
      Sol 153: Perf: 0.078192 ms, 58.361654 Tflops *
      Sol 154: Perf: 0.118800 ms, 38.412579 Tflops
      Sol 155: Perf: 0.094048 ms, 48.522162 Tflops
      Sol 156: Perf: 0.096448 ms, 47.314741 Tflops
      Sol 157: Perf: 0.097504 ms, 46.802307 Tflops
      Sol 158: Perf: 0.094144 ms, 48.472736 Tflops
      Sol 159: Perf: 0.080160 ms, 56.928818 Tflops
      Sol 160: Perf: 0.097696 ms, 46.710328 Tflops
      Sol 161: Perf: 0.083312 ms, 54.774985 Tflops
      Sol 162: Perf: 0.089936 ms, 50.740671 Tflops
      Sol 163: Perf: 0.085040 ms, 53.661965 Tflops
      Sol 164: Perf: 0.093184 ms, 48.972062 Tflops
      Sol 165: Perf: 0.099120 ms, 46.039265 Tflops
      Sol 166: Perf: 0.113968 ms, 40.041192 Tflops
      Sol 167: Perf: 0.094896 ms, 48.088565 Tflops
      Sol 168: Perf: 0.094064 ms, 48.513962 Tflops
      Sol 169: Perf: 0.100896 ms, 45.228868 Tflops
      Sol 170: Perf: 0.131088 ms, 34.811831 Tflops
      Sol 171: Perf: 0.102880 ms, 44.356687 Tflops
      Sol 172: Perf: 0.091088 ms, 50.098946 Tflops
      Sol 173: Perf: 0.077792 ms, 58.661747 Tflops *
      Sol 174: Perf: 0.098512 ms, 46.323414 Tflops
      Sol 175: Perf: 0.080704 ms, 56.545007 Tflops
      Sol 176: Perf: 0.099824 ms, 45.714575 Tflops
      Sol 177: Perf: 0.075120 ms, 60.748335 Tflops *
      Sol 178: Perf: 0.091888 ms, 49.662770 Tflops
      Sol 179: Perf: 0.091728 ms, 49.749398 Tflops
      Sol 180: Perf: 0.095440 ms, 47.814512 Tflops
      Sol 181: Perf: 0.096240 ms, 47.417002 Tflops
      Sol 182: Perf: 0.100480 ms, 45.416167 Tflops
      Sol 183: Perf: 0.101152 ms, 45.114400 Tflops
      Sol 184: Perf: 0.103856 ms, 43.939837 Tflops
      Sol 185: Perf: 0.081152 ms, 56.232920 Tflops
      Sol 186: Perf: 0.104880 ms, 43.510790 Tflops
      Sol 187: Perf: 0.087808 ms, 51.970358 Tflops
      Sol 188: Perf: 0.102752 ms, 44.411946 Tflops
      Sol 189: Perf: 0.098112 ms, 46.512271 Tflops
      Sol 190: Perf: 0.124064 ms, 36.782741 Tflops
      Sol 191: Perf: 0.090752 ms, 50.284487 Tflops
      Sol 192: Perf: 0.089648 ms, 50.903678 Tflops
      Sol 193: Perf: 0.090816 ms, 50.248996 Tflops
      Sol 194: Perf: 0.098176 ms, 46.481951 Tflops
      Sol 195: Perf: 0.100656 ms, 45.336712 Tflops
      Sol 196: Perf: 0.109504 ms, 41.673503 Tflops
      Sol 197: Perf: 0.109072 ms, 41.838519 Tflops
      Sol 198: Perf: 0.091888 ms, 49.662825 Tflops
      Sol 199: Perf: 0.095088 ms, 47.991463 Tflops
      Sol 200: Perf: 0.113248 ms, 40.295764 Tflops
      Sol 201: Perf: 0.168800 ms, 27.034440 Tflops
      Sol 202: Perf: 0.105104 ms, 43.418056 Tflops
      Sol 203: Perf: 0.106432 ms, 42.876350 Tflops
      Sol 204: Perf: 0.086320 ms, 52.866235 Tflops
      Sol 205: Perf: 0.096864 ms, 47.111539 Tflops
      Sol 206: Perf: 0.089808 ms, 50.812988 Tflops
      Sol 207: Perf: 0.096112 ms, 47.480152 Tflops
      Sol 208: Perf: 0.093280 ms, 48.921660 Tflops
      Sol 209: Perf: 0.105376 ms, 43.306027 Tflops
      Sol 210: Perf: 0.098896 ms, 46.143544 Tflops
      Sol 211: Perf: 0.110112 ms, 41.443395 Tflops
      Sol 212: Perf: 0.117040 ms, 38.990211 Tflops
      Sol 213: Perf: 0.106288 ms, 42.934398 Tflops
      Sol 214: Perf: 0.124096 ms, 36.773254 Tflops
      Sol 215: Perf: 0.098048 ms, 46.542633 Tflops
      Sol 216: Perf: 0.092208 ms, 49.490419 Tflops
      Sol 217: Perf: 0.104768 ms, 43.557303 Tflops
      Sol 218: Perf: 0.110160 ms, 41.425299 Tflops
      Sol 219: Perf: 0.087872 ms, 51.932505 Tflops
      Sol 220: Perf: 0.091920 ms, 49.645535 Tflops
      Sol 221: Perf: 0.093680 ms, 48.712773 Tflops
      Sol 222: Perf: 0.109088 ms, 41.832420 Tflops
      Sol 223: Perf: 0.104704 ms, 43.583928 Tflops
      Sol 224: Perf: 0.100400 ms, 45.452356 Tflops
      Sol 225: Perf: 0.151472 ms, 30.127098 Tflops
      Sol 226: Perf: 0.106688 ms, 42.773466 Tflops
      Sol 227: Perf: 0.103456 ms, 44.109728 Tflops
      Sol 228: Perf: 0.077552 ms, 58.843209 Tflops
      Sol 229: Perf: 0.094784 ms, 48.145387 Tflops
      Sol 230: Perf: 0.092864 ms, 49.140815 Tflops
      Sol 231: Perf: 0.104528 ms, 43.657354 Tflops
      Sol 232: Perf: 0.082592 ms, 55.252491 Tflops
      Sol 233: Perf: 0.103888 ms, 43.926264 Tflops
      Sol 234: Perf: 0.105072 ms, 43.431321 Tflops
      Sol 235: Perf: 0.122144 ms, 37.360935 Tflops
```

# Run gemm algo index sample test
```
./clients/staging/example_hipblaslt_algo_index
Usage: ./example_hipblaslt_algo_index <options>
options:
	-h, --help				Show this help message
	-v, --verbose				Verbose output
	-V, --validate				Verify results
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

Example:
./example_hipblaslt_algo_index --datatype fp16 -m 128 -n 128 -k 16 --beta 1 --act relu --trans_a T -V
result:
transAB, M, N, K, lda, ldb, ldc, stride_a, stride_b, stride_c, batch_count, alpha, beta, use_e, bias, activationType, us, tflops
gemm type gemm. problems: 1
Algo index found: 1
TN, 128, 128, 16, 16, 16, 128, 2048, 2048, 16384, 1, 2, 1, 0, none, 0, relu, 44.3333, 0.011826
PASS
```
