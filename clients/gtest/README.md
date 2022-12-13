# Gtest
hipblaslt-test is the main regression gtest for hipBLASLt. All test items should pass.

```
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# Run full gtest tests
./clients/staging/hipblaslt-test

# Run gtest tests with filter
./clients/staging/hipblaslt-test --gtest_filter=<test pattern>

# Demo: gtest tests with filter
./clients/staging/hipblaslt-test --gtest_filter=*quick*

```
