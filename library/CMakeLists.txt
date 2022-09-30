# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Configure a header file to pass the hipBLASLt version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipblaslt-version.h"
)

# Public hipBLAS headers
set(hipblaslt_headers_public
  include/hipblaslt.h
  ${PROJECT_BINARY_DIR}/include/hipblaslt-version.h
)

if(NOT BUILD_CUDA)
    set(hipblaslt_headers_public ${hipblaslt_headers_public} include/hipblaslt-types.h)
endif()
source_group("Header Files\\Public" FILES ${hipblaslt_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

# Create hipBLASLt library
add_library(hipblaslt ${hipblaslt_source} ${hipblaslt_headers_public})
add_library(roc::hipblaslt ALIAS hipblaslt)

# Target compile definitions
if(NOT BUILD_CUDA)
if( BUILD_WITH_TENSILE )

  if( BUILD_SHARED_LIBS )
    target_link_libraries( hipblaslt PRIVATE TensileHost )
  else()
    target_compile_definitions( hipblaslt PRIVATE HIPBLASLT_STATIC_LIB )

    # bypassing cmake dependencies chain for static link as it won't allow target from different directory

    # including tensile headers into hipblas tensileHost client so get compile properties
    get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
    target_include_directories( hipblaslt PRIVATE ${TensileHost_INCLUDES} )
    get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
    target_compile_definitions( hipblaslt PRIVATE ${TensileHost_DEFINES} )

    get_target_property( TensileHost_LIBDIR TensileHost BINARY_DIR )

    message (STATUS "TensileHost_INCLUDES == ${TensileHost_INCLUDES}")
    message (STATUS "TensileHost_DEFINES == ${TensileHost_DEFINES}")
    message (STATUS "TensileHost_LIBDIR == ${TensileHost_LIBDIR}")

    # recreate LLVM static dependencies
    if (${Tensile_LIBRARY_FORMAT} STREQUAL "yaml")
      find_package(LLVM 6.0 QUIET CONFIG)
      if(NOT LLVM_FOUND)
          find_package(LLVM 7.0 QUIET CONFIG)
          if(NOT LLVM_FOUND)
              find_package(LLVM 9.0 QUIET CONFIG)
              if(NOT LLVM_FOUND)
                  find_package(LLVM REQUIRED CONFIG)
              endif()
          endif()
      endif()
      find_library(LLVMObjectYAML_LIBRARY
        NAMES LLVMObjectYAML
        PATHS ${LLVM_LIBRARY_DIR})
      message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")

      target_link_libraries(hipblaslt PRIVATE LLVMObjectYAML )  # match tensile
    endif()

    # to get TensileHost built first, not to link target
    # as dependency chain can not be created
    add_dependencies(hipblaslt TensileHost)

  endif()

  target_compile_definitions(hipblaslt PRIVATE ${TENSILE_DEFINES} )
endif()

    target_compile_options(hipblaslt PRIVATE -Wno-unused-command-line-argument -Wall)
    target_compile_definitions(hipblaslt PRIVATE ROCM_USE_FLOAT16 __HIP_PLATFORM_HCC__)
else()
    target_compile_definitions(hipblaslt PRIVATE __HIP_PLATFORM_NVCC__)
endif()

# Target compile features
target_compile_features(hipblaslt PRIVATE cxx_nullptr)

# Target include directories
target_include_directories(hipblaslt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${Tensile_INC}>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                                   $<INSTALL_INTERFACE:include>
)

if(BUILD_CUDA)
target_include_directories(hipblaslt
                           PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
else()
target_include_directories(hipblaslt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/include>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/src/include>
)
endif()

# Target link libraries
if(NOT BUILD_CUDA)
# Target link libraries
  target_link_libraries(hipblaslt PRIVATE hip::device ${DL_LIB})
else()
  target_link_libraries(hipblaslt PRIVATE ${CUDA_cublaslt_LIBRARY})
endif()

# Target properties
rocm_set_soversion(hipblaslt ${hipblaslt_SOVERSION})
set_target_properties(hipblaslt PROPERTIES CXX_EXTENSIONS NO)
set_target_properties(hipblaslt PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipblaslt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_propertieS(hipblaslt PROPERTIES DEBUG_POSTFIX "-d")

# TODO ??
# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
  set_target_properties(hipblaslt PROPERTIES PREFIX "lib")
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(hipblaslt EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipblaslt-export.h)

# Install targets
rocm_install_targets(TARGETS hipblaslt
                     INCLUDE
                       ${CMAKE_SOURCE_DIR}/library/include
                       ${CMAKE_BINARY_DIR}/include
                     PREFIX hipblaslt
)

# Export targets
if(NOT BUILD_CUDA)
  rocm_export_targets(TARGETS roc::hipblaslt
                      PREFIX hipblaslt
                      DEPENDS PACKAGE hip
                      NAMESPACE roc::)
else()
  rocm_export_targets(TARGETS roc::hipblaslt
                      PREFIX hipblaslt
                      NAMESPACE roc::)
endif()

# Symbolic links
rocm_install_symlink_subdir(hipblaslt)

if(NOT BUILD_CUDA)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0")
endif()

# Package specific CPACK vars
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

if(NOT BUILD_CUDA)
  set(package_name hipblaslt)
else()
  set(package_name hipblaslt-alt)
endif()

set(HIPBLASLT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute BLASLt library"
  MAINTAINER "hipBLASLt Maintainer"
  LDCONFIG
  LDCONFIG_DIR ${HIPBLASLT_CONFIG_DIR}
)
