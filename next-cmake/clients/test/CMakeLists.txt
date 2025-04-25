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

add_subdirectory(src)

set(HIPBLASLT_COMMON "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_common.yaml")
add_custom_command(OUTPUT "${HIPBLASLT_COMMON}"
                   COMMAND ${CMAKE_COMMAND} -E copy "${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_common.yaml" "${HIPBLASLT_COMMON}"
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}"
)
set(HIPBLASLT_TEMPLATE "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_template.yaml")
add_custom_command(OUTPUT "${HIPBLASLT_TEMPLATE}"
                   COMMAND ${CMAKE_COMMAND} -E copy "${TEMP_CLIENTS_SOURCE_DIR}/include/hipblaslt_template.yaml" "${HIPBLASLT_TEMPLATE}"
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}"
)
set(HIPBLASLT_GENTEST "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_gentest.py")
add_custom_command(OUTPUT "${HIPBLASLT_GENTEST}"
                   COMMAND ${CMAKE_COMMAND} -E copy "${TEMP_CLIENTS_SOURCE_DIR}/common/hipblaslt_gentest.py" "${HIPBLASLT_GENTEST}"
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}"
)
add_custom_target(hipblaslt-common ALL DEPENDS "${HIPBLASLT_COMMON}" "${HIPBLASLT_TEMPLATE}" "${HIPBLASLT_GENTEST}")
set(HIPBLASLT_TEST_DATA "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_gtest.data")
add_custom_command(OUTPUT "${HIPBLASLT_TEST_DATA}"
                   COMMAND "${Python3_EXECUTABLE}" ${HIPBLASLT_GENTEST} -I "${TEMP_CLIENTS_SOURCE_DIR}/include" "${TEMP_CLIENTS_SOURCE_DIR}/gtest/hipblaslt_gtest.yaml" -o "${HIPBLASLT_TEST_DATA}"
                   DEPENDS ${HIPBLASLT_GENTEST} ${HIPBLASLT_COMMON}
                   WORKING_DIRECTORY "${TEMP_CLIENTS_SOURCE_DIR}/gtest")
add_custom_target(hipblaslt-test-data ALL
                  DEPENDS "${HIPBLASLT_TEST_DATA}")
add_dependencies(hipblaslt-test hipblaslt-test-data hipblaslt-common)
