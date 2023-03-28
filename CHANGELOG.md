# Change Log for hipBLASLt

## (Unreleased) hipBLASLt 0.2.0
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

## (Unreleased) hipBLASLt 0.1.0
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
