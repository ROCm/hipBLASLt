********************************
hipBLASLtExt Operation Reference
********************************

hipBLASLtExt Opertion API Reference
================================

hipblasltExtSoftmax()
------------------------------------------
.. doxygenfunction:: hipblasltExtSoftmax


hipblasltExtLayerNorm()
------------------------------------------
.. doxygenfunction:: hipblasltExtLayerNorm


hipblasltExtAMax()
------------------------------------------
.. doxygenfunction:: hipblasltExtAMax


hipblasLtExt Operation Usage
================================

Introduction
--------------

hipBLASLt has extension opertion APIs which is independent to gemm operation with. These extensions support:

1. hipblasltExtSoftmax

2. hipblasltExtLayerNorm
    | covert a 2D tensor by LayerNorm to generate a new 2D normalized tensor .
    | it is a independ function which can just call and get result.
    | sample code is in clients/samples/ext_op/sample_hipblaslt_ext_op_layernorm.cpp

3. hipblasltExtAMax
    | Abs Maximum value of a 2D tensor.
    | it is a independ function which can just call and get result.
    | sample code is in clients/samples/ext_op/sample_hipblaslt_ext_op_amax.cpp

