################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import os
import re
import shlex
import shutil
import subprocess

from pathlib import Path
from timeit import default_timer as timer
from typing import List, Union

from ..Common import globalParameters, print1, print2, ensurePath, splitArchs

class SourceToolchain:
    def __init__(self, compiler: str, bundler: str, buildIdKind: str, asanBuild: bool=False, saveTemps: bool=False):
        self.compiler = compiler
        self.bundler = bundler
        self.buildIdKind = buildIdKind
        self.asanBuild = asanBuild
        self.saveTemps = saveTemps

    def invoke(self, args: List[str], desc: str=""):
      """Invokes a subprocess with the provided arguments.

      Args:
          args: A list of arguments to pass to the subprocess.
          desc: A description of the subprocess invocation.

      Raises:
          RuntimeError: If the subprocess invocation fails.
      """
      print2(f"{desc}: {' '.join(args)}")
      try:
          out = subprocess.check_output(args, stderr=subprocess.STDOUT)
      except subprocess.CalledProcessError as err:
          raise RuntimeError(
              f"Error with {desc}: {err.output}\n"
              f"Failed command: {' '.join(args)}"
          )
      print2(f"Output: {out}")
      return out

    def compile(self, srcPath: str, destPath: str, includePath: str, gfxs: List[str]):
        """Compiles a source file into an object file.

        Args:
            cmdlineArchs: List of architectures for offloading.
            kernelFile: The path to the kernel source file.
            buildPath: The build directory path.
            objectFilename: The name of the output object file.
            outputPath: The output directory path.
            globalParameters: A dictionary of global parameters.

        Raises:
            RuntimeError: If the compilation command fails.
        """
        launcher = shlex.split(os.environ.get("Tensile_CXX_COMPILER_LAUNCHER", ""))

        hipFlags = [
            "-D__HIP_HCC_COMPAT_MODE__=1",
            "--offload-device-only",
            "-x", "hip", "-O3",
            "-I", includePath,
            "-Xoffload-linker", f"--build-id={self.buildIdKind}",
            "-std=c++17",
        ]
        if self.asanBuild:
            hipFlags.extend(["-fsanitize=address", "-shared-libasan", "-fuse-ld=lld"])
        if self.saveTemps:
            hipFlags.append("--save-temps")
        if os.name == "nt":
            hipFlags.extend(["-fms-extensions", "-fms-compatibility", "-fPIC", "-Wno-deprecated-declarations"])

        archFlags = [f"--offload-arch={gfx}" for gfx in gfxs]

        args = [
            *launcher, self.compiler, *hipFlags, *archFlags, srcPath, "-c", "-o", destPath
        ]

        return self.invoke(args, f"Compiling HIP source kernels into objects (.cpp -> .o)")


    def targets(self, objFile: str):
        """Lists the target triples in an object file.

        Args:
            objFile: The object file path.

        Returns:
            List of target triples in the object file.
        """
        args = [self.bundler, "--type=o", f"--input={objFile}", "-list"]
        return self.invoke(args, f"Listing target triples in object file").decode().split("\n")

    def unbundle(self, target: str, srcPath: str, destPath: str):
        """Unbundles source code object files using the Clang Offload Bundler.

        Args:
            target: The target triple, see https://llvm.org/docs/AMDGPUUsage.html#target-triples.
            infile: The path to the input object file.
            outfileRaw: The path to the unbundled code object.

        Raises:
            RuntimeError: If unbundling the source code object file fails.
        """
        args = [
            self.bundler,
            "--type=o",
            f"--targets={target}",
            f"--input={srcPath}",
            f"--output={destPath}",
            "--unbundle",
        ]

        return self.invoke(args, f"Unbundling source code object file")


def _computeSourceCodeObjectFilename(target: str, base: str, buildPath: Union[Path, str], arch: str) -> Union[Path, None]:
    """Generates a code object file path using the target, base, and build path.

    Args:
        target: The target triple.
        base: The base name for the output file (name without extension).
        buildPath: The build directory path.

    Returns:
        Path to the code object file.
    """
    coPath = None
    buildPath = Path(buildPath)
    if "TensileLibrary" in base and "fallback" in base:
        coPath = buildPath / "{0}_{1}.hsaco.raw".format(base, arch)
    elif "TensileLibrary" in base:
        variant = [t for t in ["", "xnack-", "xnack+"] if t in target][-1]
        baseVariant = base + "-" + variant if variant else base
        if arch in baseVariant:
            coPath = buildPath / (baseVariant + ".hsaco.raw")
    else:
        coPath= buildPath / "{0}.so-000-{1}.hsaco.raw".format(base, arch)

    return coPath


def buildSourceCodeObjectFiles(
        toolchain: SourceToolchain,
        destDir: Union[Path, str],
        tmpObjDir: Union[Path, str],
        includeDir: Union[Path, str],
        kernelPath: Union[Path, str],
        fromTensile: bool
    ) -> List[str]:
    """Compiles a HIP source code file into a code object file.

    Args:
        toolchain: The source toolchain.
        destDir: The destination directory where HSA code object files are placed.
        tmpObjDir: The directory where HIP source object files are created.
        includeDir: The include directory path.
        kernelPath: The path to the kernel source file.

    Returns:
        List of paths to the created code objects.
    """
    start = timer()

    tmpObjDir = Path(ensurePath(tmpObjDir))
    destDir = Path(ensurePath(destDir))
    kernelPath = Path(kernelPath)

    if "CmakeCxxCompiler" in globalParameters and globalParameters["CmakeCxxCompiler"] is not None:
      os.environ["CMAKE_CXX_COMPILER"] = globalParameters["CmakeCxxCompiler"]

    objFilename = kernelPath.stem + '.o'
    coPathsRaw = []
    coPaths= []

    _, cmdlineArchs = splitArchs(globalParameters, fromTensile)

    objPath = str(tmpObjDir / objFilename)
    toolchain.compile(str(kernelPath), objPath, str(includeDir), cmdlineArchs)

    for target in toolchain.targets(objPath):
      match = re.search("gfx.*$", target)
      if match:
        arch = re.sub(":", "-", match.group())
        coPathRaw = _computeSourceCodeObjectFilename(target, kernelPath.stem, tmpObjDir, arch)
        if not coPathRaw: continue
        toolchain.unbundle(target, objPath, str(coPathRaw))

        coPath = str(destDir / coPathRaw.stem)
        coPathsRaw.append(coPathRaw)
        coPaths.append(coPath)

    for src, dst in zip(coPathsRaw, coPaths):
        shutil.move(src, dst)

    stop = timer()
    print1(f"buildSourceCodeObjectFile time (s): {(stop-start):3.2f}")

    return coPaths
