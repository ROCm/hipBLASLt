.. meta::
   :description: A library that provides GEMM operations with flexible APIs and extends functionalities beyond the traditional BLAS library
   :keywords: hipBLASLt, ROCm, library, API, tool

.. _installation:

***********************
Installation
***********************

This document provides all the information required to build and install hipBLASLt on Linux systems.

Prerequisites
=============

* A ROCm enabled platform. For more information refer to `ROCm Documentation <https://rocm.docs.amd.com/>`_.
* A compatible version of hipBLAS

Installing prebuilt packages
=============================

Download prebuilt packages from `ROCm's native package manager <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#native-package-manager>`_ .

.. code-block:: bash
   sudo apt update && sudo apt install hipblaslt

Build using script
========================

Build library dependencies and library
---------------------------------------
The root of this repository has a helper bash script ``install.sh`` to build and install hipBLASLt with a single command.  It takes a lot of options and hard-coded configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.
A few commands in the script need sudo access so it may prompt you for a password.

Typical uses of ``install.sh`` to build (library dependencies and library) are listed below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+-----------------------------------+
|  Command                                  | Description                       |
+===========================================+===================================+
| ``./install.sh -h``                       | Help information.                 |
+-------------------------------------------+-----------------------------------+
| ``./install.sh -d``                       | Build library                     |
|                                           | dependencies and library          |
|                                           | in your local directory.          |
|                                           | Use ``-d`` flag only once.        |
|                                           | For subsequent invocations        |
|                                           | of ``install.sh``, it is not      |
|                                           | necessary to rebuild the          |
|                                           | dependencies.                     |
+-------------------------------------------+-----------------------------------+
| ``./install.sh``                          | Build library in your             |
|                                           | local directory. The dependencies |
|                                           | are assumed to be already built.  |
+-------------------------------------------+-----------------------------------+
| ``./install.sh -i``                       | Build library, then               |
|                                           | build and install                 |
|                                           | hipBLASLt package in              |
|                                           | ``/opt/rocm/hipblaslt``.          |
|                                           | This prompts for                  |
|                                           | sudo access and installs          |
|                                           | for all users.                    |
|                                           | If you want to keep               |
|                                           | hipBLASLt in your local           |
|                                           | directory, don't use ``-i`` flag. |
+-------------------------------------------+-----------------------------------+


Build library dependencies, client dependencies, library, and client
---------------------------------------------------------------------

The client contains executables as listed below:

============================= ========================================================
Executable Name                Description
============================= ========================================================
``hipblaslt-test``             Runs Google tests to test the library
``hipblaslt-bench``            Executable to benchmark or test individual functions
============================= ========================================================

Common uses of ``install.sh`` to build (dependencies, library, and client) are listed below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+------------------------------------+
| Command                                   | Description                        |
+===========================================+====================================+
| ``./install.sh -h``                       | Help information.                  |
+-------------------------------------------+------------------------------------+
| ``./install.sh -dc``                      | Build library                      |
|                                           | dependencies, client               |
|                                           | dependencies, library,             |
|                                           | and client in your local           |
|                                           | directory. Use ``-d`` flag         |
|                                           | only once. For subsequent          |
|                                           | invocations of                     |
|                                           | ``install.sh``, it is not          |
|                                           | necessary to rebuild the           |
|                                           | dependencies.                      |
+-------------------------------------------+------------------------------------+
| ``./install.sh -c``                       | Build library and client           |
|                                           | in your local directory.           |
|                                           | The dependencies are               |
|                                           | assumed to be already built.       |
+-------------------------------------------+------------------------------------+
| ``./install.sh -idc``                     | Build library                      |
|                                           | dependencies, client               |
|                                           | dependencies, library,             |
|                                           | client, then build and             |
|                                           | install the hipBLASLt              |
|                                           | package. This prompts for sudo     |
|                                           | access. To install for all users,  |
|                                           | use ``-i`` flag. To keep hipBLASLt |
|                                           | in your local directory, don't use |
|                                           | ``-i`` flag.                       |
+-------------------------------------------+------------------------------------+
| ``./install.sh -ic``                      | Build and install                  |
|                                           | hipBLASLt package, and             |
|                                           | build the client. This             |
|                                           | prompts for sudo access and        |
|                                           | installs for all users.            |
|                                           | To keep hipBLASLt in your local    |
|                                           | directory, don`t use ``-i`` flag.  |
+-------------------------------------------+------------------------------------+

Dependencies
--------------

Dependencies are listed in the ``install.sh`` script. Use ``install.sh`` with ``-d`` option to install dependencies.
CMake has a minimum version requirement which is listed in ``install.sh``. See ``--cmake_install`` flag in ``install.sh`` to upgrade automatically.

Manual build (all supported platforms)
=======================================

This section provides information on how to configure cmake and build manually using individual commands.

Build library manually
----------------------------------------

.. code-block:: bash
   mkdir -p [HIPBLASLT_BUILD_DIR]/release
   cd [HIPBLASLT_BUILD_DIR]/release
   # Default install location is in /opt/rocm, define -DCMAKE_INSTALL_PREFIX=<path> to specify other
   # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=<config> to specify other
   CXX=/opt/rocm/bin/hipcc ccmake [HIPBLASLT_SOURCE]
   make -j$(nproc)
   sudo make install # sudo required if installing into system directory such as /opt/rocm


Build library, tests, benchmarks, and samples manually
-----------------------------------------------------------------------

The repository contains source for clients that serve as samples, tests, and benchmarks. You can find the clients source in the clients sub-directory.

Dependencies for hipBLASLt clients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hipBLASLt samples have no external dependencies, but unit test and benchmarking applications do. These clients introduce the following dependencies:

- `Lapack <https://github.com/Reference-LAPACK/lapack-release>`_,  Lapack itself brings a dependency on a fortran compiler
- `googletest <https://github.com/google/googletest>`_

Googletest and Lapack are not easy to install. Many distros don't provide a googletest package with precompiled libraries and the Lapack packages don't have the necessary ``cmake`` config files for ``cmake`` to configure linking the ``cblas`` library. hipBLASLt provides a ``cmake`` script that builds the above dependencies from source. This is an optional step; you can provide your own builds of these dependencies and help ``cmake`` find them by setting the ``CMAKE_PREFIX_PATH`` definition. The following is a sequence of steps to build dependencies and install them to the ``cmake`` default ``/usr/local``.

One-time optional step
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mkdir -p [HIPBLASLT_BUILD_DIR]/release/deps
   cd [HIPBLASLT_BUILD_DIR]/release/deps
   ccmake -DBUILD_BOOST=OFF [HIPBLASLT_SOURCE]/deps   # assuming boost is installed through package manager as above
   make -j$(nproc) install

Once dependencies are available on the system, it is possible to configure the clients to build. This requires a few extra ``cmake`` flags to the library ``cmake`` configure script. If the dependencies are not installed into system defaults (like ``/usr/local`` ), pass the ``CMAKE_PREFIX_PATH`` to ``cmake`` to help find them.

.. code-block::bash

   -DCMAKE_PREFIX_PATH="<semicolon separated paths>"
   # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
   CXX=/opt/rocm/bin/hipcc ccmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [HIPBLASLT_SOURCE]
   make -j$(nproc)
   sudo make install   # sudo required if installing into system directory such as /opt/rocm
