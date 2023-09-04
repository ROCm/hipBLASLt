.. hipBLASLt documentation master file, created by
   sphinx-quickstart on 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hipBLASLt's documentation!
=====================================

************
Introduction
************

General Description
===================

hipBLASLt is a library that provides general matrix-matrix operations with a flexible API and extends funtionalities beyond traditional BLAS library.
hipBLASLt is exposed APIs in HIP programming language with an underlying optimized generator as a backend kernel provider.

This library adds flexibility in matrix data layouts, input types, compute types, and also in choosing the algorithmic implementations and heuristics through parameter programmability.
After a set of options for the intended GEMM operation are identified by the user, these options can be used repeatedly for different inputs.

The GEMM operation of hipBLASLt is performed by :ref:`hipblasltmatmul`. The equation is listed here:

.. math::

 D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)

where :math:`op(A)/op(B)` refers to in-place operations such as transpose/non-transpose, and alpha, beta are scalars.
Acitivation function supports Gelu, Relu. Bias vector match matrix D rows and broadcast to all D columns.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/hipBLASLt

hipBLASLt Logging
=================
The hipBLASLt logging mechanism can be enabled by setting the following environment variables before launching the target application:

HIPBLASLT_LOG_LEVEL=<level> - while level is one of the following levels:

+------------------------------------------------------------------------------------------------------------------+
|"0" - Off - logging is disabled (default)                                                                         |
+------------------------------------------------------------------------------------------------------------------+
|"1" - Error - only errors will be logged                                                                          |
+------------------------------------------------------------------------------------------------------------------+
|"2" - Trace - API calls that launch HIP kernels will log their parameters and important information               |
+------------------------------------------------------------------------------------------------------------------+
|"3" - Hints - hints that can potentially improve the application's performance                                    |
+------------------------------------------------------------------------------------------------------------------+
|"4" - Info - provides general information about the library execution, may contain details about heuristic status |
+------------------------------------------------------------------------------------------------------------------+
|"5" - API Trace - API calls will log their parameter and important information                                    |
+------------------------------------------------------------------------------------------------------------------+

HIPLASLT_LOG_MASK=<mask> - while mask is a combination of the following masks:

+-----------------+
|"0" - Off        |
+-----------------+
|"1" - Error      |
+-----------------+
|"2" - Trace      |
+-----------------+
|"4" - Hints      |
+-----------------+
|"8" - Info       |
+-----------------+
|"16" - API Trace |
+-----------------+

HIPBLASLT_LOG_FILE=<file_name> - while file name is a path to a logging file. File name may contain %i, that will be replaced with the process ID. For example, "<file_name>_%i.log".
If HIPBLASLT_LOG_FILE is not defined, the log messages are printed to stdout.

Heuristics Cache
================
hipBLASLt uses heuristics to pick the most suitable matmul kernel for execution based on the problem sizes, GPU configuration, and other parameters. This requires performing some computations on the host CPU, which could take tens of microseconds.
To overcome this overhead, it is recommended to query the heuristics once using :ref:`hipblasltmatmulalgogetheuristic` and then reuse the result for subsequent computations using :ref:`hipblasltmatmul`.


hipBLASLt Extensions
================
See extension reference page for more information.
