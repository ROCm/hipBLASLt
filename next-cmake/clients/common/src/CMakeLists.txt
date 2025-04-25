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

set(_CMAKE_CURRENT_SOURCE_DIR "${TEMP_CLIENTS_SOURCE_DIR}/common")
target_sources(hipblaslt-clients-common
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/utility.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/efficiency_monitor.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_parse_data.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_arguments.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_random.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_init_device.cpp>
)

if(ENABLE_BLIS)
    target_sources(hipblaslt-clients-common PRIVATE ${_CMAKE_CURRENT_SOURCE_DIR}/blis_interface.cpp)
endif()

unset(_CMAKE_CURRENT_SOURCE_DIR)
