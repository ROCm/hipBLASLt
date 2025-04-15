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
    project(rocisa LANGUAGES HIP CXX)
endif()

set(TEMP_ROCISA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite/rocisa)

add_library(rocisa-cpp INTERFACE)
add_library(rocisa::rocisa-cpp ALIAS rocisa-cpp)

target_sources(rocisa-cpp INTERFACE
    $<BUILD_INTERFACE:${TEMP_ROCISA_SOURCE_DIR}/rocisa/include/enum.hpp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocisa/include/enum.hpp>
)
target_compile_features(rocisa-cpp INTERFACE cxx_std_20)
target_include_directories(rocisa-cpp INTERFACE ${TEMP_ROCISA_SOURCE_DIR})


# if(DEFINED Python_EXECUTABLE AND Python_EXECUTABLE)
#     message(STATUS "Manually set Python_EXECUTABLE to ${Python_EXECUTABLE}")
# endif()

# find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# if(Python_VERSION VERSION_GREATER_EQUAL "3.8")
#     message(STATUS "Found Python version: ${Python_VERSION}")
# else()
#     message(FATAL_ERROR "Python version 3.8 or higher is required, but found version ${Python_VERSION}")
# endif()

# include(FetchContent)
# FetchContent_Declare(
#   nanobind
#   GIT_REPOSITORY https://github.com/wjakob/nanobind.git
#   GIT_TAG        9b3afa9dbdc23641daf26fadef7743e7127ff92f # v2.6.1
# )
# FetchContent_MakeAvailable(nanobind)


# nanobind_add_module(rocisa NOMINSIZE NB_SUPPRESS_WARNINGS)

# target_sources(rocisa PRIVATE 
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/main.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/base.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/code.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/container.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/count.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/enum.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/label.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/macro.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/instruction.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/common.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/branch.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/cmp.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/cvt.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/mem.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/mfma.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/instruction/extension.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/pass/pass.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/pass/graph.cpp
#     ${TEMP_ROCISA_SOURCE_DIR}/rocisa/src/pass/composite.cpp
# )

# target_compile_features(rocisa PRIVATE cxx_std_20)
# target_include_directories(rocisa PUBLIC ${TEMP_ROCISA_SOURCE_DIR}/rocisa/include)
# set_target_properties(rocisa PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# add_library(rocisa::rocisa ALIAS rocisa)
# # export(TARGETS rocisa NAMESPACE rocisa:: FILE ${CMAKE_CURRENT_BINARY_DIR}/rocisa-targets.cmake)