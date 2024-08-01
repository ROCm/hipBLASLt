#!/bin/sh
# usage: build.sh [--debug]

if [ ! -z "$1" ] && [ "$1" = "--debug" ]; then
    debug=true
    echo "DEBUG_BUILD"
    shift
else
    debug=false
    echo "RELEASE_BUILD"
fi

rm -rf build/
mkdir -p build
cd build

cmake_common_options="-DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"

# build type
if $debug == true; then
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
else
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
fi

cmake ${cmake_common_options} ./../.
ERR=$?
if [ $ERR -ne 0 ]
then
    echo one
    exit $ERR
fi

make
ERR=$?
if [ $ERR -ne 0 ]
then
    echo one
    exit $ERR
fi

cp -f runner ../
