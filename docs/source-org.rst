.. meta::
   :description: A library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _source-code-organization:

********************************
Library source code organization
********************************

The hipBLASLt source code is available in the following two directories:

- The ``library`` directory contains all source code for the library.
- The ``clients`` directory contains all test code and code to build clients.

``library`` directory
-----------------------

Here are the sub-directories in the ``library`` directory:

- ``library/include``

Contains C98 include files for the external API. These files also contain Doxygen
comments that document the API.

- ``library/src/amd_detail``

Contains implementation of hipBLASLt interface that is compatible with rocBLASLt APIs.

- ``library/src/include``

Contains internal include files for converting C++ exceptions to hipBLAS status.

``clients`` directory
-----------------------

Here are the sub-directories in ``clients`` directory:

- ``clients/samples``

Contains sample code for calling hipBLASLt functions

Infrastructure
--------------

- ``CMake`` is used to build and package hipBLASLt. There are ``CMakeLists.txt`` files throughout the code.
- ``Doxygen/Breathe/Sphinx/ReadTheDocs`` are used to produce documentation. The documentation is sourced from:

  - Doxygen comments in ``include`` files in the ``library/include`` directory
  - Files in the ``docs/source`` directory

- Jenkins is used to automate Continuous Integration testing.
- ``clang-format`` is used to format C++ code.
