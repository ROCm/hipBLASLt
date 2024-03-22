# Changelog for hipBLASLt

Full documentation for hipBLASLt is available at [rocm.docs.amd.com/projects/hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html).

## hipBLASLt 0.7.0

### Additions

* Added `hipblasltExtSoftmax` extension API
* Added `hipblasltExtLayerNorm` extension API
* Added `hipblasltExtAMax` extension API
* Added `GemmTuning` extension parameter to set split-k by user
* Support for mix precision datatype: fp16/fp8 in with fp16 out

## hipBLASLt 0.6.0

### Additions

* New `UserArguments` variable for `GroupedGemm`
* Support for datatype: fp16 in with fp32 out
* Support for datatype: Int8 in Int32 out
* Support for gfx94x platform
* Support for fp8/bf8 datatype (only for gfx94x platform)
* Support scalar A,B,C,D for fp8/bf8 datatype
* Added samples

### Changes

* Replaced `hipblasDatatype_t` with `hipDataType`
* Replaced `hipblasLtComputeType_t` with `hipblasComputeType_t`

### Removals

* Deprecated `HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER`

## hipBLASLt 0.3.0

### Additions

* Added `getAllAlgos` extension APIs
* TensileLite support for new epilogues: gradient gelu, gradient D, gradient A/B, aux
* Added a sample package that includes three sample apps
* Added a new C++ GEMM class in the hipBLASLt extension

### Changes

* Refactored GroupGemm APIs as C++ class in the hipBLASLt extension
* Changed the scaleD vector enum to `HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER`

### Fixes

* Enabled norm check validation for CI

### Optimizations

* GSU kernel: wider memory, PGR N
* Updated logic yaml to improve some FP16 NN sizes
* GroupGemm support for GSU kernel
* Added grouped GEMM tuning for aldebaran

## hipBLASLt 0.2.0

### Additions

* Added CI tests for TensileLite
* Initialized extension group GEMM APIs (FP16 only)
* Added a group GEMM sample app: `example_hipblaslt_groupedgemm`

### Fixes

* Fixed incorrect results for the ScaleD kernel

### Optimizations

* Tuned equality sizes for the HHS data type
* Reduced host-side overhead for `hipblasLtMatmul()`
* Removed unused kernel arguments
* Schedule values setup before first `s_waitcnt`
* Refactored TensileLite host codes
* Optimized build time

## hipBLASLt 0.1.0

### Additions

* Enabled hipBLASLt APIs
* Support for gfx90a
* Support for problem type: fp32, fp16, bf16
* Support activation: relu, gelu
* Support for bias vectors
* Integrated with TensileLite kernel generator
* Added Gtest: `hipblaslt-test`
* Added the full function tool `hipblaslt-bench`
* Added the sample app `example_hipblaslt_preference`

### Optimizations

* gridBase solution search algorithm for untuned size
* Tuned 10k sizes for each problem type
