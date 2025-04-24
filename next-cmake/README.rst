.. highlight:: rst

=========
hipBLASLt
=========

-----------------
Quick Start Guide
-----------------

This section describes how to configure and build the hipBLASLt project. We assume the user has a
ROCm installation, Python 3.8 or newer and CMake 3.25.0 or newer. 

The hipBLASLt project consists of three components:

1. host library
2. device libraries
3. client applications

Each component has a corresponding subdirectory. The host and device libraries are independently
configurable and buildable but the client applications require the host library build time and the
device libraries at runtime.

^^^^^^^^^^^^^^^
Python packages
^^^^^^^^^^^^^^^

hipBLASLt has two internal python packages:

- rocisa
- tensilelite

These must be installed into the local environment in order to use the device libraries. Assuming
the project is cloned into a directory named *hipBLASLt* one can install the python packages as
follows:

   .. code-block:: bash
      :linenos:
      :emphasize-lines: 1,2,4
   
      cd hipBLASLt
      pip install tensilelite/rocisa
      pip install tensilelite

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

hipBLASLt provides modern CMake support and relies on native CMake fnuctionality with exception of
some project specific options. As such, users are advised to refer to the CMake documentation for
general usage questions. Below are usage examples to get started. For details on all configuration
options see the options section.

Full build of hipBLASLt
-----------------------

   .. code-block:: cmake
      :linenos:
   
      cd hipBLSALt/next-cmake
      CC=/opt/rocm/bin/amdclang++          \
      CXX=/opt/rocm/bin/amdclang++         \
      cmake -D CMAKE_BUILD_TYPE=Release    \
            -D CMAKE_PREFIX_PATH=/opt/rocm \
            -D BUILD_SHARED_LIBS=ON        \
            -D GPU_TARGETS=gfx950          \
            -B build                       \
            -S .
      cmake --build build --parallel 32

Building device libraries
-------------------------
   .. code-block:: cmake
      :linenos:
      :emphasize-lines: 8,9
   
      cd hipBLSALt/next-cmake
      CC=/opt/rocm/bin/amdclang++          \
      CXX=/opt/rocm/bin/amdclang++         \
      cmake -D CMAKE_BUILD_TYPE=Release    \
            -D CMAKE_PREFIX_PATH=/opt/rocm \
            -D GPU_TARGETS=gfx950          \
            -D ENABLE_HOST=OFF             \
            -D ENABLE_CLIENT=OFF           \
            -B build                       \
            -S .
      cmake --build build --parallel 32

Options
-------

*Project wide options*:
- `ENABLE_HOST`: enables generation of host library (default: `ON`)
- `ENABLE_DEVICE`: enables generation of device libraries (default: `ON`)
- `ENABLE_CLIENT`: enables generation of client applications (default: `ON`)
- `ENABLE_OPENMP`: 
- `ENABLE_HIP`:
- `ENABLE_LLVM`:
- `ENABLE_BLIS`:

*Host library options:*
-

*Device libraries options:*
-

*Client options:*
-

CMake Targets
-------------

- `roc::hipblaslt`
- `rocisa::rocisa-cpp`

---------------
Physical Design
---------------

The hipBLASLt project consists of three components:

1. host library
2. device libraries
3. client applications

Each component has a corresponding directory. The host
and device libraries are independently configurable and
buildable but the client applications require the host
library to build and the device libraries to run.

^^^^^^^^^^^^
Host library
^^^^^^^^^^^^

The host library code is compiled and linked into a single
library - *libhipblaslt* - and composes three logical groups
of source into the `host-library` directory:

- hipblaslt
- rocblaslt
- tensilelite

^^^^^^^^^^^^^^^^
Device libraries
^^^^^^^^^^^^^^^^

The device libraries
