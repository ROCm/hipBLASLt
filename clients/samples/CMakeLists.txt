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

# Samples
add_executable( sample_hipblaslt_gemm gemm/sample_hipblaslt_gemm.cpp)
add_executable( sample_hipblaslt_gemm_ext gemm/sample_hipblaslt_gemm_ext.cpp)
add_executable( sample_hipblaslt_gemm_alphavec_ext gemm_alphavec/sample_hipblaslt_gemm_alphavec_ext.cpp)
add_executable( sample_hipblaslt_gemm_get_all_algos gemm_get_all_algos/sample_hipblaslt_gemm_get_all_algos.cpp)
add_executable( sample_hipblaslt_gemm_get_all_algos_ext gemm_get_all_algos/sample_hipblaslt_gemm_get_all_algos_ext.cpp)
add_executable( sample_hipblaslt_gemm_get_algo_by_index_ext gemm_get_algo_by_index/sample_hipblaslt_gemm_get_algo_by_index_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_ext groupedgemm/sample_hipblaslt_groupedgemm_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_fixed_mk_ext groupedgemm_fixed_mk/sample_hipblaslt_groupedgemm_fixed_mk_ext.cpp)
add_executable( sample_hipblaslt_groupedgemm_get_all_algos_ext groupedgemm_get_all_algos/sample_hipblaslt_groupedgemm_get_all_algos_ext.cpp)
set(samples sample_hipblaslt_gemm
            sample_hipblaslt_gemm_ext
            sample_hipblaslt_gemm_alphavec_ext
            sample_hipblaslt_gemm_get_all_algos
            sample_hipblaslt_gemm_get_all_algos_ext
            sample_hipblaslt_gemm_get_algo_by_index_ext
            sample_hipblaslt_groupedgemm_ext
            sample_hipblaslt_groupedgemm_fixed_mk_ext
            sample_hipblaslt_groupedgemm_get_all_algos_ext)

# Old examples
add_executable( example_hipblaslt_preference example_hipblaslt_preference.cpp)
add_executable( example_hipblaslt_fp8bf8 example_hipblaslt_fp8bf8.cpp)
add_executable( example_hipblaslt_groupedgemm example_hipblaslt_groupedgemm.cpp)
add_executable( example_hipblaslt_groupedgemm_fixed_mk example_hipblaslt_groupedgemm_fixed_mk.cpp)
add_executable( example_hipblaslt_groupedgemm_get_all_algos example_hipblaslt_groupedgemm_get_all_algos.cpp)
add_executable( example_hipblaslt_algo_index example_hipblaslt_algo_index.cpp)
add_executable( example_hipblaslt_matrixtransform example_hipblaslt_matrixtransform.cpp)
add_executable( example_hipblaslt_extop_softmax example_hipblaslt_extop_softmax.cpp)
add_executable( example_hipblaslt_extop_layernorm example_hipblaslt_extop_layernorm.cpp)
set( old_examples example_hipblaslt_preference  example_hipblaslt_fp8bf8 example_hipblaslt_groupedgemm example_hipblaslt_groupedgemm_fixed_mk example_hipblaslt_groupedgemm_get_all_algos example_hipblaslt_algo_index example_hipblaslt_matrixtransform example_hipblaslt_extop_softmax example_hipblaslt_extop_layernorm)
set( sample_list_all ${samples} ${old_examples})

set( sample_list_hip_device ${sample_list_all} )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::hipblaslt )

  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

  target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      )

  if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c -Wno-unused-result )
    target_compile_definitions( ${exe} PRIVATE ROCBLASLT_INTERNAL_API )
  endif( )
endforeach( )

foreach( exe ${sample_list_hip_device} )
  target_link_libraries( ${exe} PRIVATE hip::device )
endforeach( )

foreach( exe ${sample_list_all} )
  rocm_install(TARGETS ${exe} COMPONENT samples)
endforeach( )

# Add tests for special cases
add_test(NAME example_hipblaslt_groupedgemm_fixed_mk
         COMMAND example_hipblaslt_groupedgemm_fixed_mk --trans_a N --trans_b N --in_datatype fp16 --out_datatype fp16 -m 2048 -k 2048 -n 0 -n 1 -n 1 -n 0 -n 1 -n 0 -n 1 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -V)

# Add tests for samples
foreach( exe ${samples} )
  add_test(NAME ${exe} COMMAND ${exe})
endforeach()
