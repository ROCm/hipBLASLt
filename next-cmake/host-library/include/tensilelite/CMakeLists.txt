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

if(ENABLE_LLVM)
  add_subdirectory(llvm)
endif()

if(ENABLE_MSGPACK)
  add_subdirectory(msgpack)
endif()

if(ENABLE_HIP)
  add_subdirectory(hip)
endif()

set(_CMAKE_CURRENT_SOURCE_DIR "${TEMP_TENSILE_HOST_SOURCE_DIR}/include")

target_sources(hipblaslt
    PRIVATE
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPU.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPUPredicates.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/AMDGPU_Detail.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Activation.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ArithmeticUnitTypes.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/CachingLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Comparison.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblemPredicates.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblemProperties.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem_Detail.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionProblem_fwd.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionSolution.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ContractionSolution_fwd.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Contractions.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_BFloat16.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Float8_BFloat8.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Float8_BFloat8_bc.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Half.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Int8.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_Int8x4.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DataTypes_XFloat32.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Debug.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Distance.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/DistinctType.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/EmbeddedData.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/EmbeddedLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ExactLogicLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/FreeSizeLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/GranularitySelectionLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelArguments.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/KernelLanguageTypes.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLFeatures.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassification.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MLPClassificationLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Macros.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MapLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MasterSolutionLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/MatchingLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/PerformanceMetricTypes.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/PlaceholderLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Predicates.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ProblemKey.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Properties.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/PropertyMatching.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/ScalarValueTypes.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Serialization.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SingleSolutionLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Singleton.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionLibrary_fwd.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/SolutionMapLibrary.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Tensile.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Tensile_fwd.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor_Detail.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorDescriptor_fwd.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorOps.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/TensorOps_fwd.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/Utils.hpp>
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/Tensile/geom.hpp>
)

unset(_CMAKE_CURRENT_SOURCE_DIR)
