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

cmake_minimum_required(VERSION 3.22...3.25.2)
project(hipblaslt VERSION 0.15.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(CMakeDependentOption)

set(ENABLE_DEVICE ON CACHE BOOL "Build hipBLASLt device libraries.")
set(ENABLE_CLIENT ON CACHE BOOL "Build hipBLASLt client apps.")
cmake_dependent_option(ENABLE_HOST "Build hipBLASLt host library." ON "ENABLE_CLIENT" OFF)
set(BUILD_TESTING ON CACHE BOOL "Build hipblaslt host tests.")
set(ENABLE_SAMPLES ON CACHE BOOL "Build client samples.")

set(ENABLE_HIP ON CACHE BOOL "Use the HIP runtime.")
set(ENABLE_MSGPACK ON CACHE BOOL "Use msgpack for parsing configuration files.")
set(ENABLE_OPENMP ON CACHE BOOL "Use OpenMP to improve performance.")
set(ENABLE_LLVM OFF CACHE BOOL "Use msgpack for parsing configuration files.")
set(ENABLE_ROCROLLER OFF CACHE BOOL "Use RocRoller library.")
set(ENABLE_BLIS ON CACHE BOOL "Enable BLIS support")
#set(TENSILE_STATIC_ONLY ON CACHE BOOL "Disable exposing Tensile symbols in a shared library.")


set(TEMP_TENSILE_HOST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/Tensile/Source/lib)
set(HIPBLASLT_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../library)
set(ROCBLASLT_LIB_DIR ${HIPBLASLT_LIB_DIR}/src/amd_detail/rocblaslt)

find_package(hip REQUIRED)
find_package(hipblas-common REQUIRED)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()
if(ENABLE_MSGPACK)
    find_package(msgpack REQUIRED NAMES msgpack msgpack-cxx msgpack-c)
endif()
if(ENABLE_LLVM)
    find_package(LLVM REQUIRED)
endif()
if(ENABLE_DEVICE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

add_library(hipblaslt)
add_library(roc::hipblaslt ALIAS hipblaslt)

target_link_libraries(hipblaslt
    PUBLIC
        roc::hipblas-common
        hip::device
    PRIVATE
        rocisa::rocisa-cpp
    PRIVATE
        roc::hipblas-common
        dl
)
if(ENABLE_OPENMP)
    target_link_libraries(hipblaslt PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_features(hipblaslt PUBLIC cxx_std_17)
target_compile_options(hipblaslt PRIVATE -Wno-unused-command-line-argument -Wno-switch)

target_compile_definitions(hipblaslt 
    PRIVATE 
        TENSILE_DEFAULT_SERIALIZATION
    PUBLIC
        ROCBLASLT_TENSILE_LAZY_LOAD
        ROCM_USE_FLOAT16
        __HIP_PLATFORM_AMD__
)

if(ENABLE_HIP)
    target_compile_definitions(hipblaslt PRIVATE TENSILE_USE_HIP)
endif()
if(ENABLE_MSGPACK)
    target_compile_definitions(hipblaslt PRIVATE TENSILE_MSGPACK)
endif()

target_include_directories(hipblaslt
    PUBLIC
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include>
)
target_include_directories(hipblaslt
    PRIVATE
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/include>
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/src/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    PUBLIC
        $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/src/include>
        $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/host-library/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(hipblaslt
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/host-library
)

if(ENABLE_HOST)
    add_subdirectory(rocisa/rocisa-cpp)
    add_subdirectory(host-library)
endif()
if(ENABLE_DEVICE)
    add_subdirectory(device-library)
endif()
if(ENABLE_CLIENT)
    add_subdirectory(clients)
endif()

install(TARGETS hipblaslt rocisa-cpp
    EXPORT hipblaslt-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES ${HIPBLASLT_LIB_INCLUDES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt
)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/host-library/include/hipblaslt/hipblaslt-export.h
        ${CMAKE_CURRENT_BINARY_DIR}/host-library/include/hipblaslt/hipblaslt-version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt
)
install(
    EXPORT hipblaslt-targets
    NAMESPACE roc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblaslt
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-config-version.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipblaslt-config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblaslt
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
if(ENABLE_DEVICE)
    install(FILES
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/hipblaslt/library
    )
endif()
