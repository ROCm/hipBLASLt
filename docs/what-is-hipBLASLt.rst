.. meta::
   :description: A library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _what-is-hipblaslt:

What is hipBLASLt?
====================

hipBLASLt is a library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library.
hipBLASLt exposes APIs in the HIP programming language with an underlying optimized generator as a backend kernel provider.

This library adds flexibility in matrix data layouts, input types, compute types, and also in choosing the algorithmic implementations and heuristics through parameter programmability.
After you identify a set of options for the intended GEMM operations, you can use these options repeatedly for different inputs.

The GEMM operation of hipBLASLt is performed by :ref:`hipblasltmatmul`. Here is the equation:

.. math::

 D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)

where, :math:`op(A)/op(B)` refers to in-place operations such as transpose/non-transpose and :math:`alpha`, :math:`beta` are the scalars.
:math:`Activation` function supports Gelu and Relu. :math:`Bias` vector matches matrix :math:`D` rows and broadcasts to all :math:`D` columns.
