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
project(device-library VERSION 0.0.1 LANGUAGES CXX ASM)

# Options and cache variables
set(Tensile_ARCHITECTURE "" CACHE STRING "Device (GFX) architecture(s) to build libraries for")
set(Tensile_COMPILER "" CACHE STRING "Compiler to use for building device libraries")
set(Tensile_CPU_THREADS "" CACHE STRING "Number of CPU threads to use for building device libraries")

set(TENSILELITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite)

# Find dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(Tensile_ARCHITECTURE)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--architecture=${Tensile_ARCHITECTURE}")
endif()
if(Tensile_COMPILER)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--cxx-compiler=${Tensile_COMPILER}")
endif()
if(Tensile_CPU_THREADS)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--jobs=${Tensile_CPU_THREADS}")
endif()

set(TENSILE_CREATE_LIBRARY_COMMAND
    ${TENSILELITE_DIR}/Tensile/bin/TensileCreateLibrary
    ${TENSILE_BUILD_OPTS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../library
    ${CMAKE_CURRENT_BINARY_DIR}/_BUILD_FROM_CMAKE
    HIP
)
string(REPLACE ";" " " TEMP_BUILD_STRING "${TENSILE_CREATE_LIBRARY_COMMAND}")
message(STATUS "Tensile build command: ${TEMP_BUILD_STRING}")

add_custom_command(
    COMMENT "Building device libraries..."
    COMMAND ${TENSILE_CREATE_LIBRARY_COMMAND}
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/library
)
add_custom_target(tensilelite-device-libraries ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/library)

add_subdirectory(extops) # can't add this given current architecture of Tensile
add_subdirectory(matrix-transform)