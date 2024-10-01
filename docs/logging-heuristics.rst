.. meta::
   :description: A library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _logging-heuristics:

=======================
Logging and heuristics
=======================

This document contains information for debugging and improving the application performance when using hipBLASLt APIs.

Logging
==========

You can enable the hipBLASLt logging mechanism by setting the following environment variables before launching the target application:

``HIPBLASLT_LOG_LEVEL=<level>`` - where the ``level`` can be:

+------------------------------------------------------------------------------------------------------------------+
|"0" - Off - Logging is disabled (default)                                                                         |
+------------------------------------------------------------------------------------------------------------------+
|"1" - Error - Only errors are logged                                                                              |
+------------------------------------------------------------------------------------------------------------------+
|"2" - Trace - API calls that launch HIP kernels log their parameters and important information                    |
+------------------------------------------------------------------------------------------------------------------+
|"3" - Hints - Hints that can potentially improve the application's performance                                    |
+------------------------------------------------------------------------------------------------------------------+
|"4" - Info - Provides general information about the library execution, may contain details about heuristic status |
+------------------------------------------------------------------------------------------------------------------+
|"5" - API Trace - API calls log their parameters and important information                                        |
+------------------------------------------------------------------------------------------------------------------+

``HIPBLASLT_LOG_MASK=<mask>`` - where ``mask`` is a combination of the following:

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
|"32" - Bench     |
+-----------------+

``HIPBLASLT_LOG_FILE=<file_name>`` - where ``file_name`` is a path to a logging file. File name may contain ``%i``, that is replaced with the process ID. For example, ``<file_name>_%i.log``.
If ``HIPBLASLT_LOG_FILE`` is not defined, the log messages are printed to stdout.

``HIPBLASLT_ENABLE_MARKER=1``

Setting ``HIPBLASLT_ENABLE_MARKER`` to 1 will enable marker trace for rocprof profiling.

Heuristics cache
==================

hipBLASLt uses heuristics to pick the most suitable matmul kernel for execution based on the problem sizes, GPU configuration, and other parameters. This requires performing some computations on the host CPU, which could take tens of microseconds.
To overcome this overhead, it is recommended to query the heuristics once using :ref:`hipblasltmatmulalgogetheuristic` and then reuse the result for subsequent computations using :ref:`hipblasltmatmul`.
