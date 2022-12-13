#!/usr/bin/env bash

# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

#set -x #echo on

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "hipBLASLt build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
#  echo "    [--prefix] Specify an alternate CMAKE_INSTALL_PREFIX for cmake"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install build dependencies"
  echo "    [-t|--test_local_path] Specify a local path for Tensile instead of remote GIT repo"
  echo "    [-a|--architecture] Set GPU architecture target(s), e.g., all, gfx90a:xnack+;gfx90a:xnack-"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [-r]--relocatable] create a package to support relocatable ROCm"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [-k|--relwithdebinfo] -DCMAKE_BUILD_TYPE=RelWithDebInfo"
  echo "    [--hip-clang] build library for amdgpu backend using hip-clang"
  echo "    [--static] build static library"
  echo "    [--address-sanitizer] build with address sanitizer"
  echo "    [--codecoverage] build with code coverage profiling enabled"
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora|sles|opensuse-leap)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL, Fedora and SLES\n"
        exit 2
        ;;
  esac
}

# checks the exit code of the last call, requires exit code to be passed in to the function
check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code "$?"
  else
    $@
    check_exit_code "$?"
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root dnf install -y ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(rpm -q ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root zypper -n --no-gpg-checks install ${package}
    fi
  done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  if [ -z ${build_clients+foo} ]; then
    printf "install_packages(): \$build_clients must be set\n"
    exit 2
  fi

  # dependencies needed for library and clients to build
  local library_dependencies_ubuntu=( "gfortran" "make" "pkg-config" "libnuma1" "git")
  local library_dependencies_centos=( "epel-release" "make" "gcc-c++" "rpm-build" )
  local library_dependencies_centos8=( "epel-release" "make" "gcc-c++" "rpm-build" "numactl-libs" )
  local library_dependencies_fedora=( "gcc-gfortran" "make" "gcc-c++" "libcxx-devel" "rpm-build" "numactl-libs" )
  local library_dependencies_sles=( "gcc-fortran" "make" "gcc-c++" "libcxxtools9" "rpm-build" )

  local client_dependencies_ubuntu=( "python3" "python3-yaml")
  local client_dependencies_centos=( "python36" "python3-pip" )
  local client_dependencies_centos8=( "python39" "python3-virtualenv" )
  local client_dependencies_fedora=( "python36" "PyYAML" "python3-pip" )
  local client_dependencies_sles=( "pkg-config" "dpkg" "python3-pip" )

  if [[ "${tensile_msgpack_backend}" == true ]]; then
    library_dependencies_ubuntu+=("libmsgpack-dev")
    library_dependencies_fedora+=("msgpack-devel")
  fi

  # wget is needed for msgpack in this case
  if [[ ("${ID}" == "ubuntu") && ("${VERSION_ID}" == "16.04") && "${tensile_msgpack_backend}" == true ]]; then
    if ! $(dpkg -s "libmsgpackc2" &> /dev/null) || $(dpkg --compare-versions $(dpkg-query -f='${Version}' --show libmsgpackc2) lt 2.1.5-1); then
      library_dependencies_ubuntu+=("wget")
    fi
  fi

  # wget and openssl are needed for cmake
  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
    if $update_cmake == true; then
      library_dependencies_ubuntu+=("wget" "libssl-dev")
      library_dependencies_centos+=("wget" "openssl-devel")
      library_dependencies_centos8+=("wget" "openssl-devel")
      library_dependencies_fedora+=("wget")
      library_dependencies_sles+=("wget" "libopenssl-devel")
    fi
  fi

  if [[ ( "${ID}" == "centos" ) || ( "${ID}" == "rhel" ) ]]; then
    if [[ "${VERSION_ID}" == "6" ]]; then
      library_dependencies_centos+=( "numactl" )
    else
      library_dependencies_centos+=( "numactl-libs" )
    fi
    if (( "${VERSION_ID%%.*}" >= "8" )); then
      client_dependencies_centos8+=( "python3-pyyaml" )
    else
      client_dependencies_centos8+=( "PyYAML" )
    fi
  fi

  if [[ "${build_clients}" == true ]]; then
    library_dependencies_ubuntu+=( "gfortran" )
    library_dependencies_centos+=( "devtoolset-7-gcc-gfortran" )
    library_dependencies_centos8+=( "gcc-gfortran" )
    library_dependencies_fedora+=( "gcc-gfortran" )
    library_dependencies_sles+=( "gcc-fortran" "pkg-config" "dpkg" )
  fi

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_apt_packages "${client_dependencies_ubuntu[@]}"
      fi
      pip3 install wheel
      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      if (( "${VERSION_ID%%.*}" >= "8" )); then
        install_yum_packages "${library_dependencies_centos8[@]}"
        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos8[@]}"
          pip3 install pyyaml
        fi
      else
        install_yum_packages "${library_dependencies_centos[@]}"
        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos[@]}"
          pip3 install pyyaml
        fi
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_dnf_packages "${client_dependencies_fedora[@]}"
        pip3 install pyyaml
      fi
      ;;

    sles|opensuse-leap)
#     elevate_if_not_root zypper -y update
      install_zypper_packages "${library_dependencies_sles[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_zypper_packages "${client_dependencies_sles[@]}"
        pip3 install pyyaml
      fi
      ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac

  # wget is needed for msgpack in this case
  if [[ ("${ID}" == "ubuntu") && ("${VERSION_ID}" == "16.04") && "${tensile_msgpack_backend}" == true ]]; then
    if ! $(dpkg -s "libmsgpackc2" &> /dev/null) || $(dpkg --compare-versions $(dpkg-query -f='${Version}' --show libmsgpackc2) lt 2.1.5-1); then
      library_dependencies_ubuntu+=("wget")
    fi
  fi
}

install_msgpack_from_source( )
{
    if [[ ! -d "${build_dir}/deps/msgpack-c" ]]; then
      pushd .
      mkdir -p ${build_dir}/deps
      cd ${build_dir}/deps
      git clone -b cpp-3.0.1 https://github.com/msgpack/msgpack-c.git
      cd msgpack-c
      CXX=${cxx} CC=${cc} ${cmake_executable} -DMSGPACK_BUILD_TESTS=OFF -DMSGPACK_BUILD_EXAMPLES=OFF .
      make
      elevate_if_not_root make install
      popd
    fi
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
build_clients=false
build_release=true
build_hip_clang=true
build_static=false
build_release_debug=false
build_codecoverage=false
install_prefix=hipblaslt-install
build_relocatable=false
build_address_sanitizer=false
matrices_dir=
matrices_dir_install=
gpu_architecture=all

tensile_cov=
tensile_fork=
tensile_merge_files=
tensile_tag=
tensile_test_local_path=
tensile_version=
build_tensile=true
tensile_msgpack_backend=true
update_cmake=true

rocm_path=/opt/rocm
if ! [ -z ${ROCM_PATH+x} ]; then
    rocm_path=${ROCM_PATH}
fi

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,clients,dependencies,debug,hip-clang,static,relocatable,codecoverage,relwithdebinfo,address-sanitizer,merge-files,no-merge-files,no_tensile,no-tensile,msgpack,no-msgpack,logic:,cov:,fork:,branch:,test_local_path:,cpu_ref_lib:,use-custom-version:,architecture: --options hicdgrkalfbnu:t: -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
        -h|--help)
            display_help
            exit 0
            ;;
        -i|--install)
            install_package=true
            shift ;;
        -d|--dependencies)
            install_dependencies=true
            shift ;;
        -c|--clients)
            build_clients=true
            shift ;;
        -r|--relocatable)
            build_relocatable=true
            shift ;;
        -g|--debug)
            build_release=false
            shift ;;
        --hip-clang)
            build_hip_clang=true
            shift ;;
        --static)
            build_static=true
            shift ;;
        --address-sanitizer)
            build_address_sanitizer=true
            shift ;;
        -k|--relwithdebinfo)
            build_release=false
            build_release_debug=true
            shift ;;
        --codecoverage)
            build_codecoverage=true
            shift ;;
        -a|--architecture)
            gpu_architecture=${2}
            shift 2 ;;
        --prefix)
            install_prefix=${2}
            shift 2 ;;
        -l|--logic)
            tensile_logic=${2}
            shift 2 ;;
                -o|--cov)
            tensile_cov=${2}
            shift 2 ;;
        -f|--fork)
            tensile_fork=${2}
            shift 2 ;;
        -b|--branch)
            tensile_tag=${2}
            shift 2 ;;
        -t|--test_local_path)
            tensile_test_local_path=${2}
            shift 2 ;;
        -n|--no_tensile|--no-tensile)
            build_tensile=false
            shift ;;
        --merge-files)
            tensile_merge_files=true
            shift ;;
        -no-merge-files)
            tensile_merge_files=false
            shift 2;;
        -u|--use-custom-version)
            tensile_version=${2}
            shift 2;;
        --msgpack)
            tensile_msgpack_backend=true
            shift ;;
        --no-msgpack)
            tensile_msgpack_backend=false
            shift ;;
        --cmake_install)
            update_cmake=true
            shift ;;
        --) shift ; break ;;
        *)  echo "Unexpected command line parameter received: '${1}'; aborting";
            exit 1
            ;;
    esac
done

if [[ -z $tensile_cov ]]; then
    if [[ $build_hip_clang == true ]]; then
        tensile_cov=V3
    else
        tensile_cov=V2
    fi
fi

#
# If matrices_dir_install has been set up then install matrices dir and exit.
#
if ! [[ "${matrices_dir_install}" == "" ]];then
    cmake -DCMAKE_MATRICES_DIR=${matrices_dir_install} -P ./cmake/ClientMatrices.cmake
    exit 0
fi

#
# If matrices_dir has been set up then check if it exists and it contains expected files.
# If it doesn't contain expected file, it will create them.
#
if ! [[ "${matrices_dir}" == "" ]];then
    if ! [ -e ${matrices_dir} ];then
        echo "Invalid dir from command line parameter --matrices-dir: ${matrices_dir}; aborting";
        exit 1
    fi

    # Let's 'reinstall' to the specified location to check if all good
    # Will be fast if everything already exists as expected.
    # This is to prevent any empty directory.
    cmake -DCMAKE_MATRICES_DIR=${matrices_dir} -P ./cmake/ClientMatrices.cmake
fi

build_dir=$(readlink -m ./build)
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
elif [[ "${build_release_debug}" == true ]]; then
  rm -rf ${build_dir}/release-debug
else
  rm -rf ${build_dir}/debug
fi

# Default cmake executable is called cmake
cmake_executable=cmake

# If user provides custom ${rocm_path} path for hcc it has lesser priority,
# but with hip-clang existing path has lesser priority to avoid use of installed clang++
if [[ "${build_hip_clang}" == true ]]; then
  export PATH=${rocm_path}/bin:${rocm_path}/hip/bin:${rocm_path}/llvm/bin:${PATH}
fi

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then
  install_packages

  CMAKE_VERSION=$(cmake --version | grep -oP '(?<=version )[^ ]*' )
  if [ -z "$CMAKE_VERSION" ] || $(dpkg --compare-versions $CMAKE_VERSION lt 3.16.8); then
      if $update_cmake == true; then
        pushd
        printf "\033[32mBuilding \033[33mcmake\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
        CMAKE_REPO="https://github.com/Kitware/CMake/releases/download/v3.16.8/"
        mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
        wget -nv ${CMAKE_REPO}/cmake-3.16.8.tar.gz
        tar -xvf cmake-3.16.8.tar.gz
        rm cmake-3.16.8.tar.gz
        cd cmake-3.16.8
        ./bootstrap --no-system-curl --parallel=16
        make -j16
        sudo make install
        popd
      else
          echo "hipBLASLt requires CMake version >= 3.16.8 and CMake version ${CMAKE_VERSION} is installed. Run install.sh again with --cmake_install flag and CMake version 3.16.8 will be installed to /usr/local"
          exit 2
      fi
  fi

  # cmake is needed to install msgpack
  case "${ID}" in
    centos|rhel|sles|opensuse-leap)
      if [[ "${tensile_msgpack_backend}" == true ]]; then
        install_msgpack_from_source
      fi
      ;;
  esac

  # The following builds googletest & lapack from source, installs into cmake default /usr/local
  pushd .
    printf "\033[32mBuilding \033[33mgoogletest\033[32m from source; installing into \033[33m/usr/local\033[0m\n"
    mkdir -p ${build_dir}/deps && cd ${build_dir}/deps
    ${cmake_executable} ../../deps
    make -j$(nproc)
    elevate_if_not_root make install
  popd
fi

if [[ "${build_relocatable}" == true ]]; then
    if ! [ -z ${ROCM_PATH+x} ]; then
        rocm_path=${ROCM_PATH}
    fi

    rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
    if ! [ -z ${ROCM_RPATH+x} ]; then
        rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,${ROCM_RPATH}"
    fi
fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
if [[ "${build_relocatable}" == true ]]; then
    export PATH=${rocm_path}/bin:${PATH}
else
    export PATH=${PATH}:/opt/rocm/bin
fi

pushd .
  # #################################################
  # configure & build
  # #################################################
  cmake_common_options="-DAMDGPU_TARGETS=${gpu_architecture}"
  cmake_client_options=""

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release/clients && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  elif [[ "${build_release_debug}" == true ]]; then
    mkdir -p ${build_dir}/release-debug/clients && cd ${build_dir}/release-debug
    cmake_common_options="${cmake_common_options}  -DCMAKE_BUILD_TYPE=RelWithDebInfo"
  else
    mkdir -p ${build_dir}/debug/clients && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  # address sanitizer
  if [[ "${build_address_sanitizer}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_ADDRESS_SANITIZER=ON"
  fi

  # code coverage
  if [[ "${build_codecoverage}" == true ]]; then
      if [[ "${build_release}" == true ]]; then
          echo "Code coverage is disabled in Release mode, to enable code coverage select either Debug mode (-g | --debug) or RelWithDebInfo mode (-k | --relwithdebinfo); aborting";
          exit 1
      fi
      cmake_common_options="${cmake_common_options} -DBUILD_CODE_COVERAGE=ON"
  fi

  # library type
  if [[ "${build_static}" == true ]]; then
    cmake_common_options="{cmake_common_options} -DBUILD_SHARED_LIBS=OFF"
  fi

  # clients
  if [[ "${build_clients}" == true ]]; then
      cmake_client_options="${cmake_client_options} -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON"

      #
      # Add matrices_dir if exists.
      #
      if ! [[ "${matrices_dir}" == "" ]];then
          cmake_client_options="${cmake_client_options} -DCMAKE_MATRICES_DIR=${matrices_dir}"
      fi
  fi

 if [[ -n "${tensile_fork}" ]]; then
    cmake_common_options="${cmake_common_options} -Dtensile_fork=${tensile_fork}"
  fi

  if [[ -n "${tensile_tag}" ]]; then
    cmake_common_options="${cmake_common_options} -Dtensile_tag=${tensile_tag}"
  fi

  if [[ -n "${tensile_test_local_path}" ]]; then
    cmake_common_options="${cmake_common_options} -DTensile_TEST_LOCAL_PATH=${tensile_test_local_path}"
  fi

  if [[ -n "${tensile_version}" ]]; then
    cmake_common_options="${cmake_common_options} -DTENSILE_VERSION=${tensile_version}"
  fi

  tensile_opt=""
  if [[ "${build_tensile}" == false ]]; then
    tensile_opt="${tensile_opt} -DBUILD_WITH_TENSILE=OFF"
   else
    tensile_opt="${tensile_opt} -DTensile_LOGIC=${tensile_logic} -DTensile_CODE_OBJECT_VERSION=${tensile_cov}"
    if [[ ${build_jobs} != $(nproc) ]]; then
      tensile_opt="${tensile_opt} -DTensile_CPU_THREADS=${build_jobs}"
    fi
  fi

  if [[ "${tensile_merge_files}" == false ]]; then
    tensile_opt="${tensile_opt} -DTensile_MERGE_FILES=OFF"
  fi

  if [[ "${tensile_msgpack_backend}" == true ]]; then
    tensile_opt="${tensile_opt} -DTensile_LIBRARY_FORMAT=msgpack"
  else
    tensile_opt="${tensile_opt} -DTensile_LIBRARY_FORMAT=yaml"
  fi

  echo $cmake_common_options
  cmake_common_options="${cmake_common_options} ${tensile_opt}"

  compiler="hcc"
  if [[ "${build_hip_clang}" == true ]]; then
    compiler="${rocm_path}/bin/hipcc"
  fi

  if [[ "${build_clients}" == false ]]; then
    cmake_client_options=""
  fi

  # Build library with AMD toolchain because of existense of device kernels
  if [[ "${build_relocatable}" == true ]]; then
    FC=gfortran CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF \
      -DCMAKE_INSTALL_PREFIX=${install_prefix} \
      -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path} \
      -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" \
      -DCMAKE_PREFIX_PATH="${rocm_path} ${rocm_path}/hcc ${rocm_path}/hip" \
      -DCMAKE_MODULE_PATH="${rocm_path}/hip/cmake" \
      -DROCM_DISABLE_LDCONFIG=ON \
      -DROCM_PATH="${rocm_path}" ../..
  else
    FC=gfortran CXX=${compiler} ${cmake_executable} ${cmake_common_options} ${cmake_client_options} -DCPACK_SET_DESTDIR=OFF -DCMAKE_INSTALL_PREFIX=${install_prefix} -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path} -DROCM_PATH="${rocm_path}" ../..
  fi
  check_exit_code "$?"

  make -j$(nproc) install VERBOSE=1
  check_exit_code "$?"

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package
    check_exit_code "$?"

    case "${ID}" in
      ubuntu)
        elevate_if_not_root dpkg -i hipblaslt[-\_]*.deb
      ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall hipblaslt-*.rpm
      ;;
      fedora)
        elevate_if_not_root dnf install hipblaslt-*.rpm
      ;;
      sles|opensuse-leap)
        elevate_if_not_root zypper -n --no-gpg-checks install hipblaslt-*.rpm
      ;;
    esac

  fi
popd
