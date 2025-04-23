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

set(_CMAKE_CURRENT_SOURCE_DIR "${TEMP_TENSILE_HOST_SOURCE_DIR}/source")

target_sources(hipblaslt
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Activation.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/AMDGPU.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/ArithmeticUnitTypes.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/ContractionProblem.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/ContractionSolution.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/EmbeddedData.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/EmbeddedLibrary.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/KernelArguments.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/KernelLanguageTypes.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/MLFeatures.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/MLPNet.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/PerformanceMetricTypes.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/ScalarValueTypes.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/TensorOps.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile.cpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp>
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-ffast-math" CXX_FAST_MATH_FLAG_SUPPORTED)
if(CXX_FAST_MATH_FLAG_SUPPORTED)
set_source_files_properties(MLPNet.cpp PROPERTIES COMPILE_FLAGS "-ffast-math")
endif()

#if(WIN32)
#target_compile_options( TensileHost PUBLIC -Wno-deprecated-declarations -Wno-ignored-attributes -Wdll-attribute-on-redeclaration -fdelayed-template-parsing )
#endif()

if(ENABLE_LLVM)
  add_subdirectory(llvm)
endif()

if(ENABLE_MSGPACK)
  add_subdirectory(msgpack)
endif()

if(ENABLE_HIP)
  add_subdirectory(hip)
endif()

unset(_CMAKE_CURRENT_SOURCE_DIR)