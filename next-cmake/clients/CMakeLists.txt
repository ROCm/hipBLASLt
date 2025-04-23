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
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.21)
    project(clients LANGUAGES CXX)
endif()

set(TEMP_CLIENTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../clients)
set(TEMP_TENSILE_CLIENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/Tensile/Source/client)

set(ENABLE_BLIS ON CACHE BOOL "Enable BLIS support")

if(ENABLE_BLIS)
    find_package(BLIS REQUIRED)
endif()

find_package(Threads REQUIRED)  # TODO: Where is this used, should it be moved up a level?
find_package(cblas REQUIRED CONFIG)

# TODO: Improve handling of blis sources
if(BLIS_FOUND)
    set( BLAS_LIBRARY ${BLIS_LIB} )
    set( BLIS_CPP ${TEMP_CLIENTS_SOURCE_DIR}/common/blis_interface.cpp )
else()
    set( BLAS_LIBRARY "blas" )
endif()

target_link_libraries(hipblaslt-bench
    PRIVATE
        ${BLAS_LIBRARY}
)

target_sources(hipblaslt-bench
    PRIVATE
        # hipblaslt client sources
        ${TEMP_CLIENTS_SOURCE_DIR}/benchmarks/client.cpp
        # Common sources
        ${TEMP_CLIENTS_SOURCE_DIR}/common/singletons.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/utility.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/efficiency_monitor.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/cblas_interface.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/argument_model.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_parse_data.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_arguments.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_random.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_init_device.cpp
        ${BLIS_CPP}  # TODO: This impl could be improved
        # Includes
        ${TEMP_CLIENTS_SOURCE_DIR}/include/singletons.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/utility.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/efficiency_monitor.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/cblas_interface.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/argument_model.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_parse_data.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_arguments.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_random.hpp
        ${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_init.hpp
)

target_include_directories(hipblaslt-bench
    PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_CLIENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/host-library/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tensilelite>  # TODO: Maybe not needed
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>  # TODO: Maybe not needed
)

# TODO: Do we need this?
target_include_directories(hipblaslt-bench
    PRIVATE
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # TODO: May be empty?
        # TODO: Which of these are needed?

)
set_target_properties(hipblaslt-bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/staging
)

# TODO: Copy template files to build and install dir
