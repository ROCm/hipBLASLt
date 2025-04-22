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

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.21)
    project(hipBLASLt-host LANGUAGES CXX)

    include(GNUInstallDirs)

    set(${PROJECT_NAME}_is_subproject ON)
endif()

# TODO: Deep dive into this in the future, is it really needed?
include(GenerateExportHeader)

find_package(msgpack REQUIRED NAMES msgpack msgpack-cxx msgpack-c)

################################################################################
# Configure rocisa
################################################################################
# TODO: conditional use of LLVM if using YAML backend for serialization and parsing
#find_package(LLVM REQUIRED CONFIG)
# TODO: conditional use of msgpack if using msgpack backend for serialization and parsing
target_link_libraries(hipblaslt-host PRIVATE rocisa::rocisa-cpp)

################################################################################
# Configure Tensile host
################################################################################
set(TENSILE_USE_HIP ON CACHE BOOL "Use the HIP runtime.")
set(TENSILE_USE_MSGPACK ON CACHE BOOL "Use msgpack for parsing configuration files.")
set(TENSILE_USE_LLVM OFF CACHE BOOL "Use msgpack for parsing configuration files.")
set(TENSILE_USE_OPENMP ON CACHE BOOL "Use OpenMP to improve performance.")
set(TENSILE_STATIC_ONLY ON CACHE BOOL "Disable exposing Tensile symbols in a shared library.")

# TODO: Add these files in the directories where they actually live, through and add_subdirectories call
set(TEMP_TENSILE_HOST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite/Tensile/Source/lib)
# TODO(@bstefanuk): Remove the glob and itemize these headers explicitly
file(GLOB TENSILE_HOSTLIB_INCLUDES ${TEMP_TENSILE_HOST_SOURCE_DIR}/include/**/*.hpp)

target_sources(hipblaslt-host
    PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/AMDGPU.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ArithmeticUnitTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ContractionProblem.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ContractionSolution.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/DataTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/Debug.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/EmbeddedLibrary.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/KernelArguments.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/KernelLanguageTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/MLFeatures.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/PerformanceMetricTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/ScalarValueTypes.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/TensorDescriptor.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/TensorOps.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/Tensile.cpp>
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/source/Utils.cpp>
        # TODO(@bstefanuk): Make these header explicity (not a glob) and update them to use build and install interfaces for them
        ${TENSILE_HOSTLIB_INCLUDES}
)

# TODO(@bstefanuk): Move these calls into the appropriate CMakeLists in the subdirectory
if(TENSILE_USE_LLVM)
    target_sources(hipblaslt-host
        PRIVATE
            ${TEMP_TENSILE_HOST_SOURCE_DIR}/source/llvm/YAML.cpp
            ${TEMP_TENSILE_HOST_SOURCE_DIR}/source/llvm/Loading.cpp
    )
endif()

if(TENSILE_USE_MSGPACK)
    target_sources(hipblaslt-host
        PRIVATE
            ${TEMP_TENSILE_HOST_SOURCE_DIR}/source/msgpack/MessagePack.cpp
    )
endif()

if(TENSILE_USE_HIP)
    target_sources(hipblaslt-host
        PRIVATE
            ${TEMP_TENSILE_HOST_SOURCE_DIR}/source/hip/HipSolutionAdapter.cpp
            ${TEMP_TENSILE_HOST_SOURCE_DIR}/source/hip/HipHardware.cpp
    )
endif()


target_include_directories(hipblaslt-host
    PRIVATE
        $<BUILD_INTERFACE:${TEMP_TENSILE_HOST_SOURCE_DIR}/include>
)

# TODO: should these go higher up?
target_compile_definitions(hipblaslt-host PRIVATE TENSILE_USE_HIP)
target_compile_definitions(hipblaslt-host PUBLIC TENSILE_DEFAULT_SERIALIZATION)
target_compile_definitions(hipblaslt-host PUBLIC -DTENSILE_MSGPACK=1)
# TODO: clean this up, what's needed, add conditional configuration
# TODO: Below is copy and pasted from before, is it really needed?
if(TARGET msgpackc-cxx)
    get_target_property(msgpack_inc msgpackc-cxx INTERFACE_INCLUDE_DIRECTORIES)
elseif(TARGET msgpackc)
    get_target_property(msgpack_inc msgpackc INTERFACE_INCLUDE_DIRECTORIES)
elseif(TARGET msgpack-cxx)
    get_target_property(msgpack_inc msgpack-cxx INTERFACE_INCLUDE_DIRECTORIES)
endif()
if(DEFINED msgpack_inc)
    # include C++ headers manually
    # External header includes included as system files
    target_include_directories(hipblaslt-host
        SYSTEM PRIVATE $<BUILD_INTERFACE:${msgpack_inc}>
    )
endif()

################################################################################
# Configure hipBLASLt sources
################################################################################
set(HIPBLASLT_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../library/)
set(ROCBLASLT_LIB_DIR ${HIPBLASLT_LIB_DIR}/src/amd_detail/rocblaslt)

configure_file(
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt-version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-version.h
)

set(HIPBLASLT_LIB_INCLUDES
    ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-version.h
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt/hipblaslt.h
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt/hipblaslt-ext.hpp
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt/hipblaslt-ext-op.h
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt/hipblaslt-types.h
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt/hipblaslt_xfloat32.h
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt/hipblaslt_float8.h
    ${HIPBLASLT_LIB_DIR}/include/hipblaslt/hipblaslt_float8_bc.h
)

target_sources(hipblaslt-host
    PRIVATE
        # hipblaslt_source_common
        ${HIPBLASLT_LIB_DIR}/src/hipblaslt_ostream.cpp
        # hipblaslt_source
        ${HIPBLASLT_LIB_DIR}/src/amd_detail/hipblaslt.cpp
        ${HIPBLASLT_LIB_DIR}/src/amd_detail/hipblaslt-ext.cpp
        ${HIPBLASLT_LIB_DIR}/src/amd_detail/hipblaslt-ext-op.cpp
        ${HIPBLASLT_LIB_DIR}/src/amd_detail/hipblaslt.cpp
        # rocblaslt_source
        ${ROCBLASLT_LIB_DIR}/src/Debug.cpp
        ${ROCBLASLT_LIB_DIR}/src/handle.cpp
        ${ROCBLASLT_LIB_DIR}/src/status.cpp
        ${ROCBLASLT_LIB_DIR}/src/rocblaslt_auxiliary.cpp
        ${ROCBLASLT_LIB_DIR}/src/rocblaslt_mat.cpp
        ${ROCBLASLT_LIB_DIR}/src/utility.cpp
        ${ROCBLASLT_LIB_DIR}/src/rocblaslt_transform.cpp
        ${ROCBLASLT_LIB_DIR}/src/UserDrivenTuningParser.cpp  # TODO: This is also defined in Tensile host
        # Tensile_SRC
        ${ROCBLASLT_LIB_DIR}/src/tensile_host.cpp
        ${HIPBLASLT_LIB_INCLUDES}
)

# TODO: Move this up a level?
target_include_directories(hipblaslt-host
    PRIVATE
        $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/src/include>  # 1
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/src/Tensile>  # 2: Tensile_INC
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/include>
        $<BUILD_INTERFACE:${ROCBLASLT_LIB_DIR}/src/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    PUBLIC
        $<BUILD_INTERFACE:${HIPBLASLT_LIB_DIR}/include>      # 3/4
        $<BUILD_INTERFACE:${HIP_ROOT_DIR}/include>           # 5
        $<BUILD_INTERFACE:${HIPBLAS-COMMON_INCLUDE_DIRS}>    # This may not be required because hipblaslt-common is an INTERFACE lib
        $<INSTALL_INTERFACE:include>                         # 6
)

generate_export_header(hipblaslt-host
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-export.h
    BASE_NAME hipblaslt
)

################################################################################
# Install hipblaslt host library
################################################################################

# hipblaslt library
install(TARGETS hipblaslt-host rocisa-cpp
    EXPORT hipblaslt-host-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# hipblaslt public headers
install(FILES ${HIPBLASLT_LIB_INCLUDES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt
)

# hipblaslt generated public headers
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-export.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/hipblaslt/hipblaslt-version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt
)

install(EXPORT hipblaslt-host-targets
    FILE hipblaslt-host-targets.cmake
    NAMESPACE roc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblaslt
)

# hipblaslt cmake version
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-host-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

unset(${PROJECT_NAME}_is_subproject)
