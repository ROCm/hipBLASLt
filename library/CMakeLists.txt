# ########################################################################
# Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
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

# Configure a header file to pass the rocBLASLt version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblaslt-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocblaslt-version.h"
)

# Public rocBLASLt headers
set(rocblaslt_headers_public
  include/rocblaslt-auxiliary.h
  include/rocblaslt-functions.h
  include/rocblaslt.h
  include/rocblaslt-types.h
  ${PROJECT_BINARY_DIR}/include/rocblaslt-version.h
)

source_group("Header Files\\Public" FILES ${rocblaslt_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

# Create rocBLASLt library
add_library(rocblaslt ${rocblaslt_source} ${rocblaslt_headers_public})
add_library(roc::rocblaslt ALIAS rocblaslt)


if( BUILD_WITH_TENSILE )

  if( BUILD_SHARED_LIBS )
    target_link_libraries( rocblaslt PRIVATE TensileHost )
  else()
    target_compile_definitions( rocblaslt PRIVATE ROCBLAS_STATIC_LIB )

    # bypassing cmake dependencies chain for static link as it won't allow target from different directory

    # including tensile headers into rocblas tensileHost client so get compile properties
    get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
    target_include_directories( rocblaslt PRIVATE ${TensileHost_INCLUDES} )
    get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
    target_compile_definitions( rocblaslt PRIVATE ${TensileHost_DEFINES} )

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

      target_link_libraries(rocblaslt PRIVATE LLVMObjectYAML )  # match tensile
    endif()

    # to get TensileHost built first, not to link target
    # as dependency chain can not be created
    add_dependencies(rocblaslt TensileHost)

  endif()

  target_compile_definitions(rocblaslt PRIVATE ${TENSILE_DEFINES} )
endif()

# Target compile options
target_compile_options(rocblaslt PRIVATE -Wno-unused-command-line-argument -Wall)
target_compile_definitions(rocblaslt PRIVATE ROCM_USE_FLOAT16)

# Target include directories
target_include_directories(rocblaslt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${Tensile_INC}>
                           PUBLIC  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<INSTALL_INTERFACE:include>
)

# Target link libraries
target_link_libraries(rocblaslt PRIVATE hip::device)
# Target properties
rocm_set_soversion(rocblaslt ${rocblaslt_SOVERSION})
set_target_properties(rocblaslt PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(rocblaslt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_properties(rocblaslt PROPERTIES DEBUG_POSTFIX "-d")

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocblaslt EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocblaslt-export.h)

# Install targets
rocm_install_targets(TARGETS rocblaslt
                     INCLUDE
                       ${CMAKE_SOURCE_DIR}/library/include
                       ${CMAKE_BINARY_DIR}/include
                     PREFIX rocblaslt
)

# Export targets
rocm_export_targets(TARGETS roc::rocblaslt
                    PREFIX rocblaslt
                    DEPENDS PACKAGE hip
                    NAMESPACE roc::
)

# Symbolic links
rocm_install_symlink_subdir(rocblaslt)

# Package specific CPACK vars
set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)")
set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name rocblaslt)

set(ROCBLASLT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute rocBLASLt library"
  MAINTAINER "rocBLASLt Maintainer <jichang@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCBLASLT_CONFIG_DIR}
)
