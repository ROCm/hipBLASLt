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
find_package(hip REQUIRED)
find_package(hipblas-common REQUIRED)

add_library(hipblaslt-host)
add_library(roc::hipblaslt-host ALIAS hipblaslt-host)

target_compile_features(hipblaslt-host PUBLIC cxx_std_17)
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
################################################################################
# add_executable(hipblaslt-bench)

# target_link_libraries(hipblaslt-bench
#     PRIVATE
#         roc::hipblaslt
#         hip::host hip::device
#         lapack cblas
#         -mf16c
#         libomp -lm -lstdc++fs
# )


# target_include_directories(hipblaslt-bench
#     PRIVATE
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../clients/include>
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/include>
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/src/include>
# )

# # TODO: Do we need this?
# target_include_directories(hipblaslt-bench
#     SYSTEM PRIVATE
#         $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
#         $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
#         $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # TODO: Which of these are needed?
# )

add_subdirectory(rocisa/rocisa-cpp)
add_subdirectory(host-library)
add_subdirectory(device-library)
# add_subdirectory(clients)
