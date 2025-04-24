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

cmake_minimum_required(VERSION 3.21)
project(clients LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(TEMP_CLIENTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../clients)
set(TEMP_TENSILE_CLIENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/Tensile/Source/client)

set(ENABLE_BLIS ON CACHE BOOL "Enable BLIS support")

find_package(Threads REQUIRED)  # TODO: Where is this used, should it be moved up a level?
find_package(cblas REQUIRED CONFIG)

################################################################################
# hipblaslt-bench
################################################################################
if(ENABLE_BLIS)
    find_package(BLIS REQUIRED)
    target_link_libraries(hipblaslt-bench PRIVATE BLIS::BLIS)
    target_sources(hipblaslt-bench PRIVATE ${TEMP_CLIENTS_SOURCE_DIR}/common/blis_interface.cpp)
endif()

target_sources(hipblaslt-bench PRIVATE
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
        $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # TODO: May be empty?

)

set_target_properties(hipblaslt-bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/staging
)

################################################################################
# hipblaslt-test
################################################################################
# get_target_property(HIPBLASLT_BENCH_SOURCES hipblaslt-bench SOURCES)  # Other option for adding sources

target_compile_definitions(hipblaslt-test
    PRIVATE
        GOOGLE_TEST
)

target_sources(hipblaslt-test
    PRIVATE
        # Test sources
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_gtest_main.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_test.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/matmul_gtest.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/auxiliary_gtest.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/matrix_transform_gtest.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_gtest_ext_op.cpp
        # Test/bench common sources
        # ${HIPBLASLT_BENCH_SOURCES}  # Other option for adding these sources
        ${TEMP_CLIENTS_SOURCE_DIR}/common/singletons.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/utility.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/efficiency_monitor.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/cblas_interface.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/argument_model.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_parse_data.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_arguments.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_random.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_init_device.cpp
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

target_include_directories(hipblaslt-test
    PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_CLIENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # TODO: May be empty?
        $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
)

set_target_properties(hipblaslt-test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/staging
)

set(HIPBLASLT_COMMON "${PROJECT_BINARY_DIR}/staging/hipblaslt_common.yaml")
add_custom_command(OUTPUT "${HIPBLASLT_COMMON}"
                   COMMAND ${CMAKE_COMMAND} -E copy "${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_common.yaml" "${HIPBLASLT_COMMON}"
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}")

set(HIPBLASLT_TEMPLATE "${PROJECT_BINARY_DIR}/staging/hipblaslt_template.yaml")
add_custom_command(OUTPUT "${HIPBLASLT_TEMPLATE}"
                   COMMAND ${CMAKE_COMMAND} -E copy "${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_template.yaml" "${HIPBLASLT_TEMPLATE}"
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}")

set(HIPBLASLT_GENTEST "${PROJECT_BINARY_DIR}/staging/hipblaslt_gentest.py")
add_custom_command(OUTPUT "${HIPBLASLT_GENTEST}"
                   COMMAND ${CMAKE_COMMAND} -E copy "${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_gentest.py" "${HIPBLASLT_GENTEST}"
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}")

add_custom_target(hipblaslt-common ALL DEPENDS "${HIPBLASLT_COMMON}" "${HIPBLASLT_TEMPLATE}" "${HIPBLASLT_GENTEST}")

set(HIPBLASLT_TEST_DATA "${PROJECT_BINARY_DIR}/staging/hipblaslt_gtest.data")
add_custom_command(OUTPUT "${HIPBLASLT_TEST_DATA}"
                   COMMAND "${Python3_EXECUTABLE}" ${HIPBLASLT_GENTEST} -I "${TEMP_CLIENTS_SOURCE_DIR}/include" "${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_gtest.yaml" -o "${HIPBLASLT_TEST_DATA}"
                   DEPENDS ${HIPBLASLT_GENTEST} ${HIPBLASLT_COMMON}
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}/gtest")
add_custom_target(hipblaslt-test-data ALL
                  DEPENDS "${HIPBLASLT_TEST_DATA}")
add_dependencies(hipblaslt-test hipblaslt-test-data hipblaslt-common)