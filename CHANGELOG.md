# Change Log for hipBLASLt

## (Unreleased) hipBLASLt 0.6.0
### Added
- Add UserArguments for GroupedGemm
- Support datatype: fp16 in with fp32 out
- Add samples
- Support datatype: Int8 in Int32 out
- Support platform gfx94x
- Support fp8/bf8 datatype (only for gfx94x platform)
- Support Scalar A,B,C,D for fp8/bf8 datatype
### Changed
- Replace hipblasDatatype_t with hipDataType
- Replace hipblasLtComputeType_t with hipblasComputeType_t
- Deprecate HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER

## (Unreleased) hipBLASLt 0.3.0
### Added
- Add getAllAlgos extension APIs
- TensileLite support new epilogues: gradient gelu, gradient D, gradient A/B, aux
- Add sample package including three sample apps
- Add new C++ GEMM class in hipblaslt extension
### Changed
- refactor GroupGemm APIs as C++ class in hipblaslt extension
- change scaleD vector enum as HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER
### Fixed
- Enable norm check validation for CI
### Optimizations
- GSU kernel optimization: wider memory, PGR N
- update logic yaml to improve some FP16 NN sizes
- GroupGemm support GSU kernel
- Add grouped gemm tuning for aldebaran

## hipBLASLt 0.2.0
### Added
- Added CI tests for tensilelite
- Initilized extension group gemm APIs (FP16 only)
- Added group gemm sample app: example_hipblaslt_groupedgemm
### Fixed
- Fixed ScaleD kernel incorrect results
### Optimizations
- Tuned equality sizes for HHS data type
- Reduced host side overhead for hipblasLtMatmul()
- Removed unused kernel arguments
- Schedule valus setup before first s_waitcnt
- Refactored tensilelite host codes
- Optimized building time

## hipBLASLt 0.1.0
### Added
- Enable hipBLASLt APIs
- Support gfx90a
- Support problem type: fp32, fp16, bf16
- Support activation: relu, gelu
- Support bias vector
- Integreate with tensilelite kernel generator
- Add Gtest: hipblaslt-test
- Add full function tool: hipblaslt-bench
- Add sample app: example_hipblaslt_preference
### Optimizations
- Gridbase solution search algorithm for untuned size
- Tune 10k sizes for each problem type
