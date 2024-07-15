# ########################################################################
# Copyright (C) 2024 Advanced Micro Devices, Inc.
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

message( STATUS "Configuring hipblas-common external dependency" )
include( ExternalProject )

set( PREFIX_hipblascommon ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where hipblaslt should install" )
set( hipblascommon_cmake_args -DCMAKE_INSTALL_PREFIX=${PREFIX_hipblascommon} )
append_cmake_cli_arguments( hipblascommon_cmake_args hipblascommon_cmake_args )

set( hipblascommon_git_repository "git@github.com:ROCm/hipBLAS-common.git" CACHE STRING "URL to download hipblas-common from" )
set( hipblascommon_git_tag "develop" CACHE STRING "git branch" )

ExternalProject_Add(
  hipblascommon
  PREFIX ${CMAKE_BINARY_DIR}/hipblas-common
  GIT_REPOSITORY ${hipblascommon_git_repository}
  GIT_TAG ${hipblascommon_git_tag}
  CMAKE_ARGS ${hipblascommon_cmake_args}
  LOG_BUILD 1
  INSTALL_COMMAND ""
  LOG_INSTALL 1
)

ExternalProject_Get_Property( hipblascommon source_dir )

set_property( TARGET hipblascommon PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( hipblascommon install_dir )
ExternalProject_Get_Property( hipblascommon binary_dir )

set( hipblascommon_INSTALL_ROOT ${install_dir} )
set( hipblascommon_BINARY_ROOT ${binary_dir} )
