*****************************
hipBLASLt Changelog
*****************************

ROCm 6.0
=============

General
-------------

1. ``hipblasDatatype_t`` has been deprecated, use ``hipblasltDatatype_t`` instead.

hipBLASLtExt
-------------

1. The alpha and beta arguments in API `setProblem` for grouped gemm have changed from `std::vector<float>` to `std::vector<void*>`.
