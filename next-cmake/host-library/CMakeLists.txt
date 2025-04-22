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

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(${PROJECT_NAME}_is_subproject TRUE)
endif()

cmake_minimum_required(VERSION 3.21)
project(hipBLASLt-host VERSION 0.10.0 LANGUAGES CXX)

set(TENSILE_USE_HIP ON CACHE BOOL "Use the HIP runtime.")
set(TENSILE_USE_MSGPACK ON CACHE BOOL "Use msgpack for parsing configuration files.")
set(TENSILE_USE_LLVM OFF CACHE BOOL "Use msgpack for parsing configuration files.")
set(TENSILE_USE_OPENMP ON CACHE BOOL "Use OpenMP to improve performance.")
set(TENSILE_STATIC_ONLY ON CACHE BOOL "Disable exposing Tensile symbols in a shared library.")

set(ENABLE_HIP ${TENSILE_USE_HIP})
set(ENABLE_LLVM ${TENSILE_USE_LLVM})
set(ENABLE_MSGPACK ${TENSILE_USE_MSGPACK})
set(ENABLE_OPENMP ${TENSILE_USE_OPENMP})

set(TEMP_TENSILE_HOST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/Tensile/Source/lib)

if(${PROJECT_NAME}_is_subproject)
    include(GNUInstallDirs)
endif()

# TODO: Deep dive into this in the future, is it really needed?
include(GenerateExportHeader)

if(ENABLE_MSGPACK)
    find_package(msgpack REQUIRED NAMES msgpack msgpack-cxx msgpack-c)
    target_compile_definitions(hipblaslt-host PRIVATE TENSILE_MSGPACK)
endif()

if(ENABLE_LLVM)
    find_package(LLVM REQUIRED CONFIG)
endif()

if(ENABLE_HIP)
    target_compile_definitions(hipblaslt-host PRIVATE TENSILE_USE_HIP)
endif()

target_link_libraries(hipblaslt-host PRIVATE rocisa::rocisa-cpp)

target_include_directories(hipblaslt-host
    PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include>
)

target_compile_definitions(hipblaslt-host PRIVATE TENSILE_DEFAULT_SERIALIZATION)

set(HIPBLASLT_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../library/)
set(ROCBLASLT_LIB_DIR ${HIPBLASLT_LIB_DIR}/src/amd_detail/rocblaslt)

configure_file(
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt-version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-version.h
)

# target_include_directories(hipblaslt-host
#     PRIVATE
#         $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/src/include>  # 1
#         $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/src/Tensile>  # 2: Tensile_INC
#         $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/include>
#         $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/src/include>
#     PUBLIC
#         $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/include>      # 3/4
#         $<BUILD_INTERFACE:${HIP_ROOT_DIR}/include>           # 5
#         $<BUILD_INTERFACE:${HIPBLAS-COMMON_INCLUDE_DIRS}>    # This may not be required because hipblaslt-common is an INTERFACE lib
#         $<INSTALL_INTERFACE:include>
target_include_directories(hipblaslt-host
    PRIVATE
        $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/src/include>
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/src/Tensile>
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/include>
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/src/include>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/include>
        $<BUILD_INTERFACE:${HIP_ROOT_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_subdirectory(src)
add_subdirectory(include)

generate_export_header(hipblaslt-host
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-export.h
    BASE_NAME hipblaslt
)

install(TARGETS hipblaslt-host rocisa-cpp
    EXPORT hipblaslt-host-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${HIPBLASLT_LIB_INCLUDES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-export.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt
)

install(EXPORT hipblaslt-host-targets
    FILE hipblaslt-host-targets.cmake
    NAMESPACE roc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblaslt
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-host-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

unset(${PROJECT_NAME}_is_subproject)
