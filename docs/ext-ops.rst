********************************
hipBLASLtExt Operation Reference
********************************

hipBLASLtExt Operation API Reference
====================================

hipblasltExtSoftmax()
------------------------------------------
.. doxygenfunction:: hipblasltExtSoftmax


hipblasltExtLayerNorm()
------------------------------------------
.. doxygenfunction:: hipblasltExtLayerNorm


hipblasltExtAMax()
------------------------------------------
.. doxygenfunction:: hipblasltExtAMax

hipblasltExtAMaxWithScale()
------------------------------------------
.. doxygenfunction:: hipblasltExtAMaxWithScale


hipblasLtExt Operation Usage
================================

Introduction
--------------

hipBLASLt has extension operation APIs which is independent to gemm operation with. These extensions support:

1. hipblasltExtSoftmax
    | Softmax for 2D-tensor. Currently it performs softmax on second dimension of input tensor and it assumes input is contigious on second dimension.
    | For sample usage, please refer to clients/benchmarks/client_ext_op_softmax.cpp

2. hipblasltExtLayerNorm
    | Convert a 2D tensor by LayerNorm to generate a new 2D normalized tensor.
    | it is a independent function which can just call and get result.
    | sample code is in clients/samples/ext_op/sample_hipblaslt_ext_op_layernorm.cpp

3. hipblasltExtAMax
    | Abs Maximum value of a 2D tensor.
    | it is a independent function which can just call and get result.
    | sample code is in clients/samples/ext_op/sample_hipblaslt_ext_op_amax.cpp

4. hipblasltExtAMaxWithScale
    | Abs Maximum value and scaled output of a 2D tensor.
    | it is a independent function which can just call and get result.
    | sample code is in clients/samples/ext_op/sample_hipblaslt_ext_op_amax_with_scale.cpp
