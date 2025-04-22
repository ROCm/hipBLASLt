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

target_sources(hipblaslt-host
    PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/Activation.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/AMDGPU.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ArithmeticUnitTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ContractionProblem.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ContractionSolution.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/DataTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/Debug.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/EmbeddedData.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/EmbeddedLibrary.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/KernelArguments.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/KernelLanguageTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/MLFeatures.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/MLPNet.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/PerformanceMetricTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ScalarValueTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/TensorDescriptor.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/TensorOps.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/Tensile.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/Utils.cpp>
)

if(ENABLE_LLVM)
  add_subdirectory(llvm)
endif()

if(ENABLE_MSGPACK)
  add_subdirectory(msgpack)
endif()

if(ENABLE_HIP)
  add_subdirectory(hip)
endif()
