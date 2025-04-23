# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# TODO: Which cmake version?
cmake_minimum_required(VERSION 3.21)
# TODO: Which hipblaslt version?
project(hipblaslt VERSION 0.1.0)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# TODO: Handling OpenMP on windows may require a conditional
find_package(OpenMP REQUIRED)
find_package(rocm_smi REQUIRED)
find_package(hip REQUIRED)
find_package(hipblas-common REQUIRED)

add_library(hipblaslt-host)
add_library(roc::hipblaslt-host ALIAS hipblaslt-host)

target_compile_features(hipblaslt-host PUBLIC cxx_std_17)
target_compile_options(hipblaslt-host PRIVATE -Wno-switch)  # TODO(@bstefanuk): temp for easier stdout parsing
target_compile_options(hipblaslt-host PRIVATE -Wno-unused-command-line-argument)
target_compile_definitions(hipblaslt-host PRIVATE ROCM_USE_FLOAT16 __HIP_PLATFORM_AMD__)

target_link_libraries(hipblaslt-host
    PRIVATE
        hip::device # this should probably be public
        roc::hipblas-common
        rocisa::rocisa-cpp
        # TODO: Potentially need to guard against the openmp cxx lib
        OpenMP::OpenMP_CXX
        dl
)

################################################################################
# Configure clients
# -----
# We do not support Fortran clients in this build system
################################################################################
add_executable(hipblaslt-bench)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options(hipblaslt-bench PRIVATE -mf16c)
endif()
target_compile_options(hipblaslt-bench PRIVATE -Wno-switch)  # TODO(@bstefanuk): temp for easier stdout parsing
target_compile_features(hipblaslt-bench PRIVATE cxx_std_17)
target_compile_definitions(hipblaslt-bench
    PRIVATE
        HIPBLASLT_BENCH
        ROCM_USE_FLOAT16
        HIPBLASLT_INTERNAL_API
        CLIENTS_NO_FORTRAN
        __HIP_PLATFORM_AMD__
)

set_target_properties(hipblaslt-bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

target_link_libraries(hipblaslt-bench
    PRIVATE
        roc::hipblaslt-host
        hip::host hip::device
        rocisa::rocisa-cpp
        OpenMP::OpenMP_CXX
        rocm_smi64
        -L${HIP_CLANG_ROOT}/lib -Wl #-rpath=${HIP_CLANG_ROOT}/lib  # TODO: Not needed?
        lapack cblas
        -lm -lstdc++fs
        # -lflang -lflagrti  # TODO: Might be needed in some contexts?
)

add_subdirectory(rocisa/rocisa-cpp)
add_subdirectory(host-library)
# add_subdirectory(device-library)
add_subdirectory(clients)
