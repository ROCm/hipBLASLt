#!/usr/bin/python3
"""Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib

try:
  import psutil
  psutil_imported = True
except ImportError:
  psutil_imported = False

args = {}
OS_info = {}


# yapf: disable
def parse_args():
    """Parse command-line arguments"""
    global OS_info

    parser = argparse.ArgumentParser(description="""Checks build arguments""")

    parser.add_argument('-a', '--architecture', dest='gpu_architecture', required=False, default="all",
                        help='Set GPU architectures, e.g. all, auto, "gfx803;gfx906:xnack-", gfx1030, gfx1101 (optional, default: all)')

    parser.add_argument(       '--address-sanitizer', dest='address_sanitizer', required=False, default=False, action='store_true',
                        help='Build with address sanitizer enabled. (optional, default: False')

    parser.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Configure & build process output directory.(optional, default: ./build)')

    parser.add_argument('-b', '--branch', dest='tensile_tag', type=str, required=False, default="",
                        help='Specify the Tensile repository branch or tag to use. (eg. develop, mybranch or <commit hash> )')

    parser.add_argument('-c', '--clients', dest='build_clients', required=False, default=False, action='store_true',
                        help='Build the library clients benchmark and gtest (optional, default: False, Generated binaries will be located at <build_dir>/clients/staging)')

    parser.add_argument(      '--codecoverage', required=False, default=False, action='store_true',
                        help='Code coverage build. Requires Debug (-g|--debug) or RelWithDebInfo mode (-k|--relwithdebinfo), (optional, default: False)')

    parser.add_argument( '-d', '--dependencies', required=False, default=False, action='store_true',
                        help='Build and install external dependencies. (Handled by install.sh and on Windows rdeps.py')

    parser.add_argument('-f', '--fork', dest='tensile_fork', type=str, required=False, default="",
                        help='Specify the username to fork the Tensile GitHub repository (e.g., ROCmSoftwarePlatform or MyUserName)')

    parser.add_argument('-g', '--debug', required=False, default=False,  action='store_true',
                        help='Build in Debug mode (optional, default: False)')

    parser.add_argument('-i', '--install', required=False, default=False, dest='install', action='store_true',
                        help='Generate and install library package after build. Windows only. Linux use install.sh (optional, default: False)')

    parser.add_argument('-j', '--jobs', type=int, required=False, default=0,
                        help='Specify number of parallel jobs to launch, increases memory usage (default: heuristic around logical core count)')

    parser.add_argument('-k', '--relwithdebinfo', required=False, default=False, action='store_true',
                        help='Build in Release with Debug Info (optional, default: False)')

    parser.add_argument('-l', '--logic', dest='tensile_logic', type=str, required=False, default="asm_full",
                        help='Specify the Tensile logic target, e.g., asm_full, asm_lite, etc. (optional, default: asm_full)')

    parser.add_argument('-n', '--no_tensile', dest='build_tensile', required=False, default=True, action='store_false',
                        help='Build a subset of hipblaslt library which does not require Tensile.')

    parser.add_argument(     '--msgpack', dest='tensile_msgpack_backend', required=False, default=True, action='store_true',
                        help='Build Tensile backend to use MessagePack (optional, default: True)')

    parser.add_argument(     '--no-msgpack', dest='tensile_msgpack_backend', required=False, default=True, action='store_false',
                        help='Build Tensile backend not to use MessagePack and so use YAML (optional)')

    parser.add_argument('-s', '--static', required=False, default=False, dest='static_lib', action='store_true',
                        help='Build hipblaslt as a static library. (optional, default: False)')

    parser.add_argument('-t', '--test_local_path', dest='tensile_test_local_path', type=str, required=False, default="",
                        help='Use a local path for Tensile instead of remote GIT repo (optional)')

    parser.add_argument('-u', '--use-custom-version', dest='tensile_version', type=str, required=False, default="",
                        help='Ignore Tensile version and just use the Tensile tag (optional)')

    parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true',
                        help='Verbose build (optional, default: False)')

    return parser.parse_args()
# yapf: enable

def get_ram_GB():
    """
    Total amount of GB RAM available or zero if unknown
    """
    if psutil_imported:
        gb = round(psutil.virtual_memory().total / pow(1024, 3))
    else:
        gb = 0
    return gb

def strip_ECC(token):
    return token.replace(':sramecc+', '').replace(':sramecc-', '').strip()

def gpu_detect():
    global OS_info
    OS_info["GPU"] = ""
    cmd = "hipinfo.exe"

    process = subprocess.run([cmd], stdout=subprocess.PIPE)
    for line_in in process.stdout.decode().splitlines():
        if 'gcnArchName' in line_in:
            OS_info["GPU"] = strip_ECC( line_in.split(":")[1] )
            break

def os_detect():
    global OS_info
    OS_info["ID"] = platform.system()
    OS_info["NUM_PROC"] = os.cpu_count()
    OS_info["RAM_GB"] = get_ram_GB()

def jobs_heuristic():
    # auto jobs heuristics
    jobs = min(OS_info["NUM_PROC"], 128) # disk limiter
    ram = OS_info["RAM_GB"]
    if (ram >= 16): # don't apply if below minimum RAM
        jobs = min(round(ram/2), jobs) # RAM limiter
    hipcc_flags = os.getenv('HIPCC_COMPILE_FLAGS_APPEND', "")
    pjstr = hipcc_flags.split("parallel-jobs=")
    if (len(pjstr) > 1):
        pjobs = int(pjstr[1][0])
        if (pjobs > 1 and pjobs < jobs):
            jobs = round(jobs / pjobs)
    jobs = min(61, jobs) # multiprocessing limit (used by tensile)
    return int(jobs)

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join(os.getcwd(), dir_path)
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return


def delete_dir(dir_path):
    if (not os.path.exists(dir_path)):
        return
    run_cmd("RMDIR", f"/S /Q {dir_path}")

def cmake_path(os_path):
    return os_path.replace("\\", "/")

def fatal(msg, code=1):
    print(msg)
    exit(code)


def deps_cmd():
    exe = f"python rdeps.py"
    all_args = ""
    return exe, all_args


def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = "cmake"
    cmake_options = []
    src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    generator = f"-G Ninja"
    cmake_options.append(generator)

    # CMAKE_PREFIX_PATH set to rocm_path and HIP_PATH set BY SDK Installer
    raw_rocm_path = cmake_path(os.getenv('HIP_PATH', "C:/hip"))
    rocm_path = f'"{raw_rocm_path}"' # guard against spaces in path
    # CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
    #cmake_platform_opts.append(f"-DCPACK_PACKAGING_INSTALL_PREFIX=")
    #cmake_platform_opts.append(f'-DCMAKE_INSTALL_PREFIX="C:/hipSDK"')
    cmake_platform_opts.append( f"-DCPACK_PACKAGING_INSTALL_PREFIX=" )
    cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"C:/hipSDK\"" )
    toolchain = os.path.join(src_path, "toolchain-windows.cmake")

    print(f"Build source path: {src_path}")

    tools = f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    cmake_options.append(tools)

    cmake_options.extend(cmake_platform_opts)

    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}"
    cmake_options.append(cmake_base_options)

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF"
    cmake_options.append(cmake_pack_options)

    if os.getenv('CMAKE_CXX_COMPILER_LAUNCHER'):
        cmake_options.append(f'-DCMAKE_CXX_COMPILER_LAUNCHER={os.getenv("CMAKE_CXX_COMPILER_LAUNCHER")}')

    # build type
    cmake_config = ""
    build_dir = os.path.realpath(args.build_dir)
    if args.debug:
        build_path = os.path.join(build_dir, "debug")
        cmake_config = "Debug"
    elif args.relwithdebinfo:
        build_path = os.path.join(build_dir, "release-debug")
        cmake_config = "RelWithDebInfo"
    else:
        build_path = os.path.join(build_dir, "release")
        cmake_config = "Release"

    cmake_options.append(f"-DCMAKE_BUILD_TYPE={cmake_config}")

    if args.codecoverage:
        if args.debug or args.relwithdebinfo:
            cmake_options.append(f"-DBUILD_CODE_COVERAGE=ON")
        else:
            fatal("*** Code coverage is not supported for Release build! Aborting. ***")

    if args.address_sanitizer:
        cmake_options.append(f"-DBUILD_ADDRESS_SANITIZER=ON")

    # clean
    delete_dir(build_path)

    create_dir(os.path.join(build_path, "clients"))
    os.chdir(build_path)

    if args.static_lib:
        cmake_options.append(f"-DBUILD_SHARED_LIBS=OFF")

    if args.build_clients:
        cmake_build_dir = cmake_path(build_dir)
        cmake_options.append(
            f"-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_DIR={cmake_build_dir}"
        )

    if args.gpu_architecture == "auto":
        gpu_detect()
        if len(OS_info["GPU"]):
            args.gpu_architecture = OS_info["GPU"]
        else:
            fatal("Could not detect GPU as requested. Not continuing.")
    # not just for tensile
    cmake_options.append(f'-DAMDGPU_TARGETS=\"{args.gpu_architecture}\"')

    if not args.build_tensile:
        cmake_options.append(f"-DBUILD_WITH_TENSILE=OFF")
    else:
        cmake_options.append(f"-DTensile_CODE_OBJECT_VERSION=V3")
        if args.tensile_logic:
            cmake_options.append(f"-DTensile_LOGIC={args.tensile_logic}")
        if args.tensile_fork:
            cmake_options.append(f"-Dtensile_fork={args.tensile_fork}")
        if args.tensile_tag:
            cmake_options.append(f"-Dtensile_tag={args.tensile_tag}")
        if args.tensile_test_local_path:
            cmake_options.append(f"-DTensile_TEST_LOCAL_PATH={args.tensile_test_local_path}")
        if args.tensile_version:
            cmake_options.append(f"-DTENSILE_VERSION={args.tensile_version}")
        if args.tensile_msgpack_backend:
            cmake_options.append(f"-DTensile_LIBRARY_FORMAT=msgpack")
        else:
            cmake_options.append(f"-DTensile_LIBRARY_FORMAT=yaml")
        if args.jobs != OS_info["NUM_PROC"]:
            cmake_options.append(f"-DTensile_CPU_THREADS={str(args.jobs)}")

    cmake_options.append(f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts


def make_cmd():
    global args
    global OS_info

    make_options = []

    # the CMAKE_BUILD_PARALLEL_LEVEL currently doesn't work for windows build, so using -j
    # make_executable = f"cmake.exe -DCMAKE_BUILD_PARALLEL_LEVEL=4 --build . " # ninja
    make_executable = f"ninja.exe -j {args.jobs}"
    if args.verbose:
        make_options.append("--verbose")
    make_options.append("all")  # for cmake "--target all" )
    if args.install:
        make_options.append("package install")  # for cmake "--target package --target install" )
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts


def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode


def main():
    global args
    os_detect()
    args = parse_args()

    if args.jobs == 0:
        args.jobs = jobs_heuristic()
    if args.jobs > 61:
        print( f"WARNING: jobs > 61 may fail on windows python multiprocessing (jobs = {args.jobs}).")

    print(OS_info)

    root_dir = os.curdir

    # depdendency install
    if (args.dependencies):
        exe, opts = deps_cmd()
        if run_cmd(exe, opts):
            fatal("Dependency install failed. Not continuing.")

    # configure
    exe, opts = config_cmd()
    if run_cmd(exe, opts):
        fatal("Configuration failed. Not continuing.")

    # make
    exe, opts = make_cmd()
    if run_cmd(exe, opts):
        fatal("Build failed. Not continuing.")

    # Linux install and cleanup not supported from rmake yet


if __name__ == '__main__':
    main()
