# ########################################################################
# Copyright (c) 2016-2022 Advanced Micro Devices, Inc.
# ########################################################################

add_executable( example_hipblaslt_preference example_hipblaslt_preference.cpp)

set( sample_list_tensile example_hipblaslt_preference)

set( sample_list_all ${sample_list_tensile})
set( sample_list_hip_device ${sample_list_tensile} )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::hipblaslt )

  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

  target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      )

  if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c )
    target_compile_definitions( ${exe} PRIVATE ROCBLASLT_INTERNAL_API )
  endif( )
endforeach( )

foreach( exe ${sample_list_hip_device} )
  target_link_libraries( ${exe} PRIVATE hip::device )
endforeach( )
