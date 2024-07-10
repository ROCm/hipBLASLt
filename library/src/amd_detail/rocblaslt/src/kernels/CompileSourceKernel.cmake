function(CompileSourceKernel source archs outputFolder)
    message("Setup source kernel targets")
    string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${archs}")
    list(REMOVE_DUPLICATES archs)
    list(JOIN archs "," archs)
    message("archs for source kernel compilation: ${archs}")
    add_custom_target(MatrixTransformKernels ALL
                      DEPENDS ${outputFolder}/hipblasltTransform.hsaco
                      VERBATIM)
    add_custom_command(OUTPUT ${outputFolder}/hipblasltTransform.hsaco
                       COMMAND bash  ${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/src/kernels/compile_code_object.sh ${source} ${archs} ${outputFolder}/hipblasltTransform.hsaco
                       COMMENT "Compiling source kernels")
endfunction()