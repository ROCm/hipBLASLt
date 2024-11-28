if("$ENV{Python_ROOT}" STREQUAL "" AND NOT Python_ROOT)
    message(STATUS "Python_ROOT is unset. Setting Python_ROOT to /usr.")
    message(STATUS "Configure Python_ROOT variable if a different installation is preferred.")
    set(Python_ROOT /usr)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter)

set(VIRTUALENV_PYTHON_EXE ${Python_EXECUTABLE})


set(VIRTUALENV_HOME_DIR ${CMAKE_BINARY_DIR}/virtualenv CACHE PATH "Path to virtual environment")

function(virtualenv_create)
    execute_process(
      RESULT_VARIABLE rc
      COMMAND ${VIRTUALENV_PYTHON_EXE} -m venv ${VIRTUALENV_HOME_DIR} --system-site-packages --clear
      COMMAND_ECHO STDOUT
    )
    if(rc)
        message(FATAL_ERROR ${rc})
    endif()

    if(WIN32)
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/Scripts CACHE PATH "Path to virtualenv bin directory")
    else()
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/bin CACHE PATH "Path to virtualenv bin directory")
    endif()

     # verify python executable name inside virtualenv as may be python3 or python (even if installed by python3)
     find_program(VIRTUALENV_INST_PYTHON_EXE python3 PATHS ${VIRTUALENV_BIN_DIR} NO_DEFAULT_PATH)
     if(NOT VIRTUALENV_INST_PYTHON_EXE)
         find_program(VIRTUALENV_INST_PYTHON_EXE python PATHS ${VIRTUALENV_BIN_DIR} NO_DEFAULT_PATH)
     endif()
 
     get_filename_component(VIRTUALENV_PYTHON_EXENAME ${VIRTUALENV_INST_PYTHON_EXE} NAME CACHE)
 
     # report the virtual env python version
     message("virtualenv python version: ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME}")
     execute_process(
         COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} --version
         )
endfunction()

function(virtualenv_install)
    virtualenv_create()
    execute_process(
      COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install --upgrade pip
      COMMAND_ECHO STDOUT
    )
    execute_process(
      COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install --upgrade setuptools
      COMMAND_ECHO STDOUT
    )
    execute_process(
      COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install ${ARGN}
      COMMAND_ECHO STDOUT
      RESULT_VARIABLE return_code
      ERROR_VARIABLE error_message
      OUTPUT_VARIABLE output_message      
    )

    if(return_code)
        message("Error Code: ${rc}")
        message("StdOut: ${output_message}")
        message(FATAL_ERROR "StdErr: ${error_message}" )
    endif()
endfunction()
