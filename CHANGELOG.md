# Change Log for hipBLASLt

## (Unreleased) hipBLASLt 1.0.0
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
