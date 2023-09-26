# ########################################################################
# Copyright (C) 2023 Advanced Micro Devices, Inc.
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

file(GLOB BLIS_AOCL_LIBS /opt/AMD/aocl/aocl-linux-aocc-*/aocc/lib_ILP64/libblis-mt.a)  # Possible location 1
file(GLOB BLIS_LOCAL_LIB /usr/local/lib/libblis.a)  # Possible location 2
file(GLOB BLIS_DEPS_LIB ${CMAKE_SOURCE_DIR}/build/deps/blis/lib/libblis.a)  # wget location
set(BLIS_LIBS ${BLIS_LOCAL_LIB} ${BLIS_AOCL_LIBS} ${BLIS_DEPS_LIB})
list(REVERSE BLIS_LIBS)
list (GET BLIS_LIBS 0 BLIS_LIB)
if("${BLIS_LIB}" STREQUAL "NOTFOUND")
    message(FATAL_ERROR "BLIS lib not found.")
endif()

get_filename_component(BLIS_ROOT "${BLIS_LIB}/../.." ABSOLUTE)
if(BLIS_ROOT MATCHES "aocl-linux-aocc-")
    set(BLIS_INCLUDE_DIR ${BLIS_ROOT}/include_ILP64/)
else()
    set(BLIS_INCLUDE_DIR ${BLIS_ROOT}/include/blis)
endif()

set(BLIS_FOUND TRUE PARENT_SCOPE)
set(BLIS_INCLUDE_DIR ${BLIS_INCLUDE_DIR} PARENT_SCOPE)
set(BLIS_LIB ${BLIS_LIB} PARENT_SCOPE)
message("BLIS heeader directory found: ${BLIS_INCLUDE_DIR}")
message("BLIS lib found: ${BLIS_LIB}")

