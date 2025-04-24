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

find_package(Threads REQUIRED) # TODO: Where is this used, should it be moved up a level?
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
# TODO: Still haven't added hipblaslt-bench-groupedgemm-fixed-mk and related targets

################################################################################
# hipblaslt-test
################################################################################
# get_target_property(HIPBLASLT_BENCH_SOURCES hipblaslt-bench SOURCES)  # Other option for adding sources
if(ENABLE_BLIS)
    find_package(BLIS REQUIRED)
    target_link_libraries(hipblaslt-test PRIVATE BLIS::BLIS)
    target_sources(hipblaslt-test PRIVATE ${TEMP_CLIENTS_SOURCE_DIR}/common/blis_interface.cpp)
endif()

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
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}> # TODO: Is this directory necessary?
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # TODO: May be empty?
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
)

set_target_properties(hipblaslt-test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/staging
)

################################################################################
# hipblaslt-sample-*
################################################################################
set(TEMP_HIPBLASLT_SAMPLE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/samples)

add_executable(sample_hipblaslt_gemm ${TEMP_HIPBLASLT_SAMPLE_DIR}/01_basic_gemm/sample_hipblaslt_gemm.cpp)
add_executable(sample_hipblaslt_gemm_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/01_basic_gemm_ext/sample_hipblaslt_gemm_ext.cpp)
add_executable(sample_hipblaslt_gemm_batched ${TEMP_HIPBLASLT_SAMPLE_DIR}/02_batched_gemm/sample_hipblaslt_gemm_batched.cpp)
add_executable(sample_hipblaslt_gemm_batched_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/02_batched_gemm_ext/sample_hipblaslt_gemm_batched_ext.cpp)
add_executable(sample_hipblaslt_gemm_tuning_splitk_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/03_splitk_gemm_ext/sample_hipblaslt_gemm_tuning_splitk_ext.cpp)
add_executable(sample_hipblaslt_gemm_bias ${TEMP_HIPBLASLT_SAMPLE_DIR}/04_gemm_bias/sample_hipblaslt_gemm_bias.cpp)
add_executable(sample_hipblaslt_gemm_bias_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/04_gemm_bias_ext/sample_hipblaslt_gemm_bias_ext.cpp)
add_executable(sample_hipblaslt_gemm_get_all_algos ${TEMP_HIPBLASLT_SAMPLE_DIR}/05_get_all_algos_gemm/sample_hipblaslt_gemm_get_all_algos.cpp)
add_executable(sample_hipblaslt_gemm_get_all_algos_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/05_get_all_algos_gemm_ext/sample_hipblaslt_gemm_get_all_algos_ext.cpp)
add_executable(sample_hipblaslt_gemm_get_algo_by_index_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/06_get_algo_by_index_gemm_ext/sample_hipblaslt_gemm_get_algo_by_index_ext.cpp)
add_executable(sample_hipblaslt_gemm_alphavec_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/07_gemm_alpha_vec_ext/sample_hipblaslt_gemm_alphavec_ext.cpp)
add_executable(sample_hipblaslt_gemm_gelu_aux_bias  ${TEMP_HIPBLASLT_SAMPLE_DIR}/08_gemm_gelu_aux_bias/sample_hipblaslt_gemm_gelu_aux_bias.cpp)
add_executable(sample_hipblaslt_gemm_gelu_aux_bias_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/08_gemm_gelu_aux_bias_ext/sample_hipblaslt_gemm_gelu_aux_bias_ext.cpp)
add_executable(sample_hipblaslt_gemm_amax ${TEMP_HIPBLASLT_SAMPLE_DIR}/09_gemm_amax/sample_hipblaslt_gemm_amax.cpp)
add_executable(sample_hipblaslt_gemm_amax_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/09_gemm_amax_ext/sample_hipblaslt_gemm_amax_ext.cpp)
add_executable(sample_hipblaslt_gemm_amax_with_scale ${TEMP_HIPBLASLT_SAMPLE_DIR}/10_gemm_amax_with_scale/sample_hipblaslt_gemm_amax_with_scale.cpp)
add_executable(sample_hipblaslt_gemm_amax_with_scale_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/10_gemm_amax_with_scale_ext/sample_hipblaslt_gemm_amax_with_scale_ext.cpp)
add_executable(sample_hipblaslt_gemm_bgradb ${TEMP_HIPBLASLT_SAMPLE_DIR}/11_gemm_bgradb/sample_hipblaslt_gemm_bgradb.cpp)
add_executable(sample_hipblaslt_gemm_ext_bgradb ${TEMP_HIPBLASLT_SAMPLE_DIR}/11_gemm_bgradb_ext/sample_hipblaslt_gemm_ext_bgradb.cpp)
add_executable(sample_hipblaslt_gemm_dgelu_bgrad ${TEMP_HIPBLASLT_SAMPLE_DIR}/12_gemm_dgelu_bgrad/sample_hipblaslt_gemm_dgelu_bgrad.cpp)
add_executable(sample_hipblaslt_gemm_dgelu_bgrad_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/12_gemm_dgelu_bgrad_ext/sample_hipblaslt_gemm_dgelu_bgrad_ext.cpp)
add_executable(sample_hipblaslt_gemm_is_tuned_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/13_is_tuned_gemm_ext/sample_hipblaslt_gemm_is_tuned_ext.cpp)
add_executable(sample_hipblaslt_gemm_tuning_wgm_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/14_tuning_wgm_gemm_ext/sample_hipblaslt_gemm_tuning_wgm_ext.cpp)
add_executable(sample_hipblaslt_gemm_with_scale_a_b  ${TEMP_HIPBLASLT_SAMPLE_DIR}/15_gemm_scale_a_b/sample_hipblaslt_gemm_with_scale_a_b.cpp)
add_executable(sample_hipblaslt_gemm_with_scale_a_b_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/15_gemm_scale_a_b_ext/sample_hipblaslt_gemm_with_scale_a_b_ext.cpp)
add_executable(sample_hipblaslt_groupedgemm_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/16_gemm_grouped_ext/sample_hipblaslt_groupedgemm_ext.cpp)
add_executable(sample_hipblaslt_groupedgemm_fixed_mk_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/17_fixed_mk_gemm_grouped_ext/sample_hipblaslt_groupedgemm_fixed_mk_ext.cpp)
add_executable(sample_hipblaslt_groupedgemm_get_all_algos_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/18_get_all_algos_gemm_grouped_ext/sample_hipblaslt_groupedgemm_get_all_algos_ext.cpp)
add_executable(sample_hipblaslt_gemm_mix_precision ${TEMP_HIPBLASLT_SAMPLE_DIR}/19_mix_precision_gemm/sample_hipblaslt_gemm_mix_precision.cpp)
add_executable(sample_hipblaslt_gemm_mix_precision_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/19_mix_precision_gemm_ext/sample_hipblaslt_gemm_mix_precision_ext.cpp)
add_executable(sample_hipblaslt_gemm_mix_precision_with_amax_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/20_mix_precision_gemm_amax_ext/sample_hipblaslt_gemm_mix_precision_with_amax_ext.cpp)
add_executable(sample_hipblaslt_gemm_attr_tciA_tciB ${TEMP_HIPBLASLT_SAMPLE_DIR}/21_gemm_cvtA_cvtB/sample_hipblaslt_gemm_attr_tciA_tciB.cpp)
add_executable(sample_hipblaslt_ext_op_layernorm ${TEMP_HIPBLASLT_SAMPLE_DIR}/22_layernorm_ext/sample_hipblaslt_ext_op_layernorm.cpp)
# add_executable(sample_hipblaslt_gemm_ext_deprecated ${TEMP_HIPBLASLT_SAMPLE_DIR}/deprecated/sample_hipblaslt_gemm_ext_deprecated.cpp)
add_executable(sample_hipblaslt_ext_op_amax ${TEMP_HIPBLASLT_SAMPLE_DIR}/23_amax_ext/sample_hipblaslt_ext_op_amax.cpp)
add_executable(sample_hipblaslt_ext_op_amax_with_scale ${TEMP_HIPBLASLT_SAMPLE_DIR}/24_amax_with_scale_ext/sample_hipblaslt_ext_op_amax_with_scale.cpp)
add_executable(sample_hipblaslt_gemm_with_TF32 ${TEMP_HIPBLASLT_SAMPLE_DIR}/25_gemm_with_TF32/sample_hipblaslt_gemm_with_TF32.cpp)
add_executable(sample_hipblaslt_gemm_swizzle_a ${TEMP_HIPBLASLT_SAMPLE_DIR}/26_gemm_swizzle_a/sample_hipblaslt_gemm_swizzle_a.cpp)
add_executable(sample_hipblaslt_gemm_bias_swizzle_a_ext ${TEMP_HIPBLASLT_SAMPLE_DIR}/26_gemm_swizzle_a/sample_hipblaslt_gemm_bias_swizzle_a_ext.cpp)
add_executable(sample_hipblaslt_weight_swizzle_padding ${TEMP_HIPBLASLT_SAMPLE_DIR}/26_gemm_swizzle_a/sample_hipblaslt_weight_swizzle_padding.cpp)
add_executable(sample_hipblaslt_gemm_swish_bias ${TEMP_HIPBLASLT_SAMPLE_DIR}/27_gemm_swish_bias/sample_hipblaslt_gemm_swish_bias.cpp)

set(hipblaslt_samples_all
    sample_hipblaslt_gemm
    sample_hipblaslt_gemm_ext
    sample_hipblaslt_gemm_batched
    sample_hipblaslt_gemm_batched_ext
    sample_hipblaslt_gemm_tuning_splitk_ext
    sample_hipblaslt_gemm_bias
    sample_hipblaslt_gemm_bias_ext
    sample_hipblaslt_gemm_get_all_algos
    sample_hipblaslt_gemm_get_all_algos_ext
    sample_hipblaslt_gemm_get_algo_by_index_ext
    sample_hipblaslt_gemm_alphavec_ext
    sample_hipblaslt_gemm_gelu_aux_bias
    sample_hipblaslt_gemm_gelu_aux_bias_ext
    sample_hipblaslt_gemm_amax
    sample_hipblaslt_gemm_amax_ext
    sample_hipblaslt_gemm_amax_with_scale
    sample_hipblaslt_gemm_amax_with_scale_ext
    sample_hipblaslt_gemm_bgradb
    sample_hipblaslt_gemm_ext_bgradb
    sample_hipblaslt_gemm_dgelu_bgrad
    sample_hipblaslt_gemm_dgelu_bgrad_ext
    sample_hipblaslt_gemm_is_tuned_ext
    sample_hipblaslt_gemm_tuning_wgm_ext
    sample_hipblaslt_gemm_with_scale_a_b
    sample_hipblaslt_gemm_with_scale_a_b_ext
    sample_hipblaslt_groupedgemm_ext
    sample_hipblaslt_groupedgemm_fixed_mk_ext
    sample_hipblaslt_groupedgemm_get_all_algos_ext
    sample_hipblaslt_gemm_mix_precision
    sample_hipblaslt_gemm_mix_precision_ext
    sample_hipblaslt_gemm_mix_precision_with_amax_ext
    sample_hipblaslt_gemm_attr_tciA_tciB
    sample_hipblaslt_ext_op_layernorm
    # sample_hipblaslt_gemm_ext_deprecated
    sample_hipblaslt_ext_op_amax
    sample_hipblaslt_ext_op_amax_with_scale
    sample_hipblaslt_gemm_with_TF32
    sample_hipblaslt_gemm_swizzle_a
    sample_hipblaslt_gemm_bias_swizzle_a_ext
    sample_hipblaslt_weight_swizzle_padding
    sample_hipblaslt_gemm_swish_bias
)

foreach(executable ${hipblaslt_samples_all})
    if(ENABLE_BLIS)
        find_package(BLIS REQUIRED)
        target_link_libraries(${executable} PRIVATE BLIS::BLIS)
        target_sources(${executable} PRIVATE ${TEMP_CLIENTS_SOURCE_DIR}/common/blis_interface.cpp)
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # GCC or hip-clang needs specific flags to turn on f16c intrinsics
        target_compile_options(${executable} PRIVATE -mf16c)
    endif()
    target_compile_options(${executable} PRIVATE -Wno-switch) # TODO(@bstefanuk): temp for easier stdout parsing
    target_compile_features(${executable} PRIVATE cxx_std_17)
    target_compile_definitions(${executable}
        PRIVATE
        ROCM_USE_FLOAT16
        HIPBLASLT_INTERNAL_API
        CLIENTS_NO_FORTRAN
        __HIP_PLATFORM_AMD__
    )
    target_sources(${executable}
        PRIVATE
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
    )
    target_include_directories(${executable}
        PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_CLIENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${TEMP_CLIENTS_SOURCE_DIR}/samples/common>
        # $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}> # TODO: Not needed?
        # $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # TODO: May be empty?
    )
    target_link_libraries(${executable}
        PRIVATE
        roc::hipblaslt
        # hip::host
        hip::device
        rocm_smi64
        rocisa::rocisa-cpp
        OpenMP::OpenMP_CXX
        rocm_smi64
        -L${HIP_CLANG_ROOT}/lib -Wl #-rpath=${HIP_CLANG_ROOT}/lib  # TODO: Not needed?
        # lapack
        cblas
        -lm -lstdc++fs
    )
    set_target_properties(${executable}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/staging
)

    install(TARGETS ${executable}
        EXPORT hipblaslt-samples-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endforeach()

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
