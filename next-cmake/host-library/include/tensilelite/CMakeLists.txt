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

target_sources(hipblaslt-host
    PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/AMDGPU.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/AMDGPUPredicates.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/AMDGPU_Detail.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Activation.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ArithmeticUnitTypes.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/CachingLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Comparison.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionProblem.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionProblemPredicates.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionProblemProperties.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionProblem_Detail.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionProblem_fwd.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionSolution.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ContractionSolution_fwd.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Contractions.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes_BFloat16.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes_Float8_BFloat8.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes_Float8_BFloat8_bc.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes_Half.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes_Int8.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes_Int8x4.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DataTypes_XFloat32.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Debug.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Distance.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/DistinctType.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/EmbeddedData.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/EmbeddedLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ExactLogicLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/FreeSizeLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/GranularitySelectionLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/KernelArguments.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/KernelLanguageTypes.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/MLFeatures.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/MLPClassification.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/MLPClassificationLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Macros.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/MapLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/MasterSolutionLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/MatchingLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/PerformanceMetricTypes.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/PlaceholderLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Predicates.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ProblemKey.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Properties.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/PropertyMatching.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/ScalarValueTypes.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Serialization.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/SingleSolutionLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Singleton.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/SolutionLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/SolutionLibrary_fwd.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/SolutionMapLibrary.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Tensile.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Tensile_fwd.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/TensorDescriptor.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/TensorDescriptor_Detail.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/TensorDescriptor_fwd.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/TensorOps.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/TensorOps_fwd.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/Utils.hpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include/Tensile/geom.hpp>
)
