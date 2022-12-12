# hipBLASLt
hipBLASLt is a library that provides general matrix-matrix operations. hipBLASLt is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs. 
hipBLASLt provides flexible APIs to let users set attributes for solution selection.

## Documentation (TBD)
TBD

## Requirements
* Git
* CMake 3.16.8 or later
* python3.7 or later
* python3.7-venv or later
* AMD [ROCm] 5.5 or later

## Quickstart hipBLASLt build and install

#### Install script
You can build hipBLASLt using the *install.sh* script
```
# Clone hipBLASLt using git
TBD

# Go to hipBLASLt directory
cd hipBLASLt

# Run install.sh script
# Command line options:
#   -h|--help         - prints help message
#   -i|--install      - install after build
#   -d|--dependencies - install build dependencies
#   -c|--clients      - build library clients too (combines with -i & -d)
#   -g|--debug        - build with debug flag
./install.sh -dc
```

#### CMake
All compiler specifications are determined automatically. The compilation process can be performed by
```
# Clone hipBLASLt using git
git clone https://github.com/ROCmSoftwarePlatform/hipBLASLt

# Go to hipBLASLt directory, create and go to the build directory
cd hipBLASLt; mkdir -p build/release; cd build/release

# Configure hipBLASLt
# Build options:
#   BUILD_CLIENTS_TESTS      - build tests (OFF)
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (OFF)
#   BUILD_CLIENTS_SAMPLES    - build examples (ON)
#   BUILD_VERBOSE            - verbose output (OFF)
#   BUILD_SHARED_LIBS        - build hipBLASLt as a shared library (ON)
CXX=/opt/rocm/bin/hipcc cmake -DBUILD_CLIENTS_TESTS=ON ../..

# Build
make

# Install
[sudo] make install
```

## Unit tests
To run unit tests, hipBLASLt has to be built with option -DBUILD_CLIENTS_TESTS=ON.
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

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/
[GTest]: https://github.com/google/googletest
[the issue tracker]: TBD
[license file]: TBD
