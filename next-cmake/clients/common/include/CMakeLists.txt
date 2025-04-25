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

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/include)

target_sources(hipblaslt-clients-common
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/allclose.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/utility.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_arguments.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/mxDataGen.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/argument_model.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_vector.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/d_vector.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/unit.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/norm.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/efficiency_monitor.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/TensorDataManipulation.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/test_cleanup.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/near.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/device_vector.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/host_vector.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_datatype2string.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipBuffer.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/singletons.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_math.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_init.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/flops.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/testing_matmul.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/datatype_interface.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_random.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/testing_auxiliary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_data.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_parse_data.hpp>
)

unset(CMAKE_CURRENT_SOURCE_DIR)
