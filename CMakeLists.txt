# ########################################################################
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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

# The ROCm platform requires at least Ubuntu 16.04 or Fedora 24, which have cmake 3.5
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# This will add compile option: -std=c++17
set(CMAKE_CXX_STANDARD 20 )
# Without this line, it will add -std=gnu++17 instead, which may have issues.
set(CMAKE_CXX_EXTENSIONS OFF )
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

if(NOT BUILD_CUDA)
# Adding CMAKE_PREFIX_PATH, needed for static builds
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )
endif()

if( NOT DEFINED ENV{HIP_PATH})
    set( HIP_PATH "/opt/rocm/hip" )
else( )
    set (HIP_PATH $ENV{HIP_PATH} )
endif( )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM 0.7.3 CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMClients )
include( ROCMHeaderWrapper )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip ${HIP_PATH}/cmake )

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# hipBLASLt project
project(hipblaslt LANGUAGES CXX)

# Force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)


# Build options
option(BUILD_SHARED_LIBS "Build hipBLASLt as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build hipBLASLt with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build hipBLASLt with address sanitizer enabled" OFF)
# Find CUDA if the user wants a CUDA version.
option(USE_CUDA "Look for CUDA and use that as a backend if found" OFF)

if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage -lgcov)
endif()

if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
    add_link_options(-fuse-ld=lld)
endif()

if(NOT BUILD_CUDA)
    # Determine if CXX Compiler is hip-clang
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Using hip-clang to build for amdgpu backend")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_HCC_COMPAT_MODE__=1" )
        if (CMAKE_BUILD_TYPE MATCHES "Debug" AND NOT WIN32)
            set (CMAKE_CXX_FLAGS_DEBUG "-O1 ${CMAKE_CXX_FLAGS_DEBUG} -gsplit-dwarf -ggdb" )
        endif()
    else()
        message(FATAL_ERROR "'hipcc' compiler required to compile for ROCm platform.")
    endif()
endif()
# Dependencies
include(cmake/Dependencies.cmake)
if (LINK_BLIS)
  include(cmake/findBLIS.cmake)
endif()

if (USE_CUDA)
	find_package( CUDA REQUIRED )
    # Hip headers required of all clients; clients use hip to allocate device memory
    find_package( HIP MODULE REQUIRED )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
else()


    #Set the AMDGPU_TARGETS with backward compatiblity
    rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
        TARGETS "gfx90a:xnack+;gfx90a:xnack-;gfx940;gfx941;gfx942"
    )

    if (AMDGPU_TARGETS)
        set(TMPAMDGPU_TARGETS "${AMDGPU_TARGETS}")
        if(TMPAMDGPU_TARGETS STREQUAL "all" )
            set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target" FORCE)
        else()
            set(AMDGPU_TARGETS "${TMPAMDGPU_TARGETS}" CACHE STRING "AMD GPU targets to compile for" FORCE)
        endif()
    else()
        set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
    endif()

    message(STATUS "AMDGPU_TARGETS: ${AMDGPU_TARGETS}")

    if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
        find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )
    endif( )

    option( BUILD_WITH_TENSILE "Build full functionality which requires tensile?" ON )

    if( BUILD_WITH_TENSILE )
    # we will have expanded "all" for tensile to ensure consistency as we have local rules
    set( Tensile_ARCHITECTURE "${AMDGPU_TARGETS}" CACHE STRING "Tensile to use which architecture?" FORCE)

    set( Tensile_LOGIC "asm_full" CACHE STRING "Tensile to use which logic?")
    set( Tensile_CODE_OBJECT_VERSION "V2" CACHE STRING "Tensile code_object_version")
    set( Tensile_COMPILER "hipcc" CACHE STRING "Tensile compiler")
    set( Tensile_LIBRARY_FORMAT "msgpack" CACHE STRING "Tensile library format")
    set( Tensile_CPU_THREADS "" CACHE STRING "Number of threads for Tensile parallel build")

    option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
    option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
    option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )

    set( Tensile_TEST_LOCAL_PATH "" CACHE PATH "Use local Tensile directory instead of fetching a GitHub branch" )

    set_property( CACHE Tensile_LOGIC PROPERTY STRINGS aldebaran asm_full asm_lite asm_miopen hip_lite other )
    set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS V2 V3 )
    set_property( CACHE Tensile_COMPILER PROPERTY STRINGS hcc hipcc)
    set_property( CACHE Tensile_LIBRARY_FORMAT PROPERTY STRINGS msgpack yaml)

    if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
      option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
      option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." OFF)
    else()
      option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." OFF)
      option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." ON)
    endif()

    if (WIN32)
      set( Tensile_ROOT "${CMAKE_BINARY_DIR}/virtualenv/Lib/site-packages/Tensile" )
    endif()

    include(virtualenv)
    if (Tensile_TEST_LOCAL_PATH)
      virtualenv_install(${Tensile_TEST_LOCAL_PATH})
      message (STATUS "using local Tensile from ${Tensile_TEST_LOCAL_PATH}, copied to ${Tensile_ROOT}")
    else()
      virtualenv_install(${CMAKE_SOURCE_DIR}/tensilelite)
    endif()
    message(STATUS "Adding ${VIRTUALENV_HOME_DIR} to CMAKE_PREFIX_PATH")
    list(APPEND CMAKE_PREFIX_PATH ${VIRTUALENV_HOME_DIR})
    if (TENSILE_VERSION)
      find_package(Tensile ${TENSILE_VERSION} EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
    else()
      find_package(Tensile 4.33.0 EXACT REQUIRED HIP LLVM OpenMP PATHS "${INSTALLED_TENSILE_PATH}")
    endif()
    endif()

    # setup hipblaslt defines used for both the library and clients
    if( BUILD_WITH_TENSILE )
        list(APPEND TENSILE_DEFINES BUILD_WITH_TENSILE=1)
    else()
        list(APPEND TENSILE_DEFINES BUILD_WITH_TENSILE=0)
    endif()
endif()

find_package( hipblas REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

# Setup version
set(VERSION_STRING "0.6.0")
rocm_setup_version(VERSION ${VERSION_STRING})
set(hipblaslt_SOVERSION 0.6)


if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  set( BUILD_CLIENTS ON )
endif()

# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include
  )
endif()

# hipBLASLt library
add_subdirectory( library )

# Trigger client builds if selected
# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")
  if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  endif()

  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "${GFORTRAN_DEB}"
        RPM "${GFORTRAN_RPM}")
  endif()
  if(BUILD_CLIENTS_SAMPLES)
    rocm_package_setup_client_component(samples)
  endif()
  add_subdirectory( clients )
endif( )

if( NOT BUILD_CUDA )
  rocm_package_add_dependencies(DEPENDS "hipblas >= 0.50.0")
endif( )

# Package specific CPACK vars
set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT")

if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
endif()

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" )

# Give hipblaslt compiled for CUDA backend a different name
if( NOT USE_CUDA )
    set( package_name hipblaslt )
else( )
    set( package_name hipblaslt-alt )
endif( )

set( HIPBLASLT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "Radeon Open Compute BLAS marshalling library"
    MAINTAINER "hipBLASLt Maintainer <hipblaslt-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLASLT_CONFIG_DIR}
)


#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > make coverage GTEST_FILTER=<>
  # will run:
  #  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
  #  > make coverage_output (generate html documentation)
  #

  #
  # Run coverage analysis
  #
  set(coverage_test ./clients/staging/hipblaslt-test)

  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ./clients/staging/hipblaslt-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis hipblaslt)

  #
  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  #
  add_custom_target(coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
    )

  #
  # Generate coverage output.
  #
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

  add_custom_target(coverage DEPENDS coverage_output)

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
