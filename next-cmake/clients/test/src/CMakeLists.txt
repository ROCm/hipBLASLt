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

target_sources(hipblaslt-test
    PRIVATE
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_gtest_main.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_test.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/matmul_gtest.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/auxiliary_gtest.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/matrix_transform_gtest.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_gtest_ext_op.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/singletons.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/utility.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/efficiency_monitor.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/cblas_interface.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/argument_model.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_parse_data.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_arguments.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_random.cpp
        ${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_init_device.cpp
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
