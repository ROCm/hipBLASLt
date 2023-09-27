# hipBLASLt
hipBLASLt is a library that provides general matrix-matrix operations with a flexible API and extends funtionalities beyond traditional BLAS library.
hipBLASLt is exposed APIs in HIP programming language with an underlying optimized generator as a backend kernel provider.

This library adds flexibility in matrix data layouts, input types, compute types, and also in choosing the algorithmic implementations and heuristics through parameter programmability.
After a set of options for the intended GEMM operation are identified by the user, these options can be used repeatedly for different inputs.
The GEMM operation of hipBLASLt is performed by hipblasLtMatmul API.

The equation is as below:
```math
D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)
```
Where op(A)/op(B) refers to in-place operations such as transpose/non-transpose, and alpha, beta are scalars.
Acitivation function supports Gelu, Relu.
Bias vector match matrix D rows and broadcast to all D columns.

Here are data type supported list:
| A | B | C | D | Compute(Scale) |
| :---: | :---: | :---: | :---: | :---: |
| fp32  | fp32  | fp32  | fp32  | fp32  |
| fp16  | fp16  | fp16  | fp16  | fp32  |
| fp16  | fp16  | fp16  | fp32  | fp32  |
| bf16  | bf16  | bf16  | bf16  | fp32  |
| fp8   | fp8/bf8  | fp32   | fp32  | fp32  |
| fp8   | fp8/bf8  | fp16   | fp16  | fp32  |
| bf8   | fp8  | fp32  | fp32  | fp32  |
| bf8   | fp8  | fp16  | fp16  | fp32  |
(fp8 and bf8 are only supported by gfx94x platform)

## Documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```
## Hardware Requirements
* gfx90a card
* gfx94x card

## Software Requirements
* Git
* CMake 3.16.8 or later
* python3.7 or later
* python3.7-venv or later
* AMD [ROCm] 5.5 or later

## Required ROCM library
* hipBLAS (for the header file)

## Quickstart hipBLASLt build and install

#### Install script
You can build hipBLASLt using the *install.sh* script
```
# Clone hipBLASLt using git
git clone https://github.com/ROCmSoftwarePlatform/hipBLASLt

# Go to hipBLASLt directory
cd hipBLASLt

# Run install.sh script
# Command line options:
#   -h|--help         - prints help message
#   -i|--install      - install after build
#   -d|--dependencies - install build dependencies
#   -c|--clients      - build library clients too (combines with -i & -d)
#   -g|--debug        - build with debug flag
./install.sh -idc
```

## Unit tests
To build unit tests, hipBLASLt has to be built with --clients.\
All unit tests are in path build/release/clients/staging/.\
Please check these links for more information.\
[hipblaslt-test](clients/gtest/README.md)\
[hipblaslt-bench](clients/benchmarks/README.md)\
[example_XXX](clients/samples/README.md)

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/
[GTest]: https://github.com/google/googletest
[the issue tracker]: TBD
[license file]: TBD
