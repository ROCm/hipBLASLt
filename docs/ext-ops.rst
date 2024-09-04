.. meta::
   :description: A library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _ext-ops:

hipBLASLtExt operation API reference
======================================

hipBLASLt has the following extension operation APIs that are independent to gemm operations.
These extensions support:

1. ``hipblasltExtSoftmax``
    Softmax for 2D-tensor. Currently, it performs softmax on the second dimension of input tensor and assumes the input to be contigious on the second dimension.
    For sample code, refer to `client_extop_softmax.cpp <https://github.com/ROCm/hipBLASLt/blob/docs/6.2.0/clients/benchmarks/client_extop_softmax.cpp>`_.

2. ``hipblasltExtLayerNorm``
    Converts a 2D tensor using LayerNorm to generate a new 2D normalized tensor.
    it is an independent function used to just call and get result.
    For sample code, refer to `sample_hipblaslt_ext_op_layernorm.cpp <https://github.com/ROCm/hipBLASLt/blob/docs/6.2.0/clients/samples/ext_op/sample_hipblaslt_ext_op_layernorm.cpp>`_.

3. ``hipblasltExtAMax``
    Abs maximum value of a 2D tensor.
    it is an independent function used to just call and get result.
    For sample code, refer to `sample_hipblaslt_ext_op_amax.cpp <https://github.com/ROCm/hipBLASLt/blob/docs/6.2.0/clients/samples/ext_op/sample_hipblaslt_ext_op_amax.cpp>`_.

4. ``hipblasltExtAMaxWithScale``
    Abs maximum value and scaled output of a 2D tensor.
    it is an independent function used to just call and get result.
    For sample code, refer to `sample_hipblaslt_ext_op_amax_with_scale.cpp <https://github.com/ROCm/hipBLASLt/blob/docs/6.2.0/clients/samples/ext_op/sample_hipblaslt_ext_op_amax_with_scale.cpp>`_.

These APIs are explained in detail below.

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
