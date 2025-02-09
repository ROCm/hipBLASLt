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

from ..Common import globalParameters, print2, printExit

class SourceToolchain:
    def __init__(self, compiler: str, objDump: str, objLs: str, buildIdKind: str, asanBuild: bool=False, saveTemps: bool=False):
        self.compiler = compiler
        self.objLs = objLs
        self.objDump = objDump
        self.buildIdKind = buildIdKind
        self.asanBuild = asanBuild
        self.saveTemps = saveTemps

    def invoke(self, args: List[str], desc: str="", workingDir=None):
      """Invokes a subprocess with the provided arguments.

      Args:
          args: A list of arguments to pass to the subprocess.
          desc: A description of the subprocess invocation.

      Raises:
          RuntimeError: If the subprocess invocation fails.
      """
      print2(f"{desc}: {' '.join(args)}")
      try:
          if workingDir:
            out = subprocess.check_output(args, stderr=subprocess.STDOUT, cwd=workingDir)
          else:
             out = subprocess.check_output(args, stderr=subprocess.STDOUT)
      except subprocess.CalledProcessError as err:
          raise RuntimeError(
              f"Error with {desc}: {err.output}\n"
              f"Failed command: {' '.join(args)}"
          )
      print2(f"Output: {out}")
      return out

    def compile(self, srcPaths: List[str], destPath: str, includePath: str, gfxs: List[str]):
        launcher = shlex.split(os.environ.get("Tensile_CXX_COMPILER_LAUNCHER", ""))

        hipFlags = [
            "-shared",
            "-fPIC",
            "-fgpu-rdc",
            "-Xoffload-linker",
            "--lto-partitions=16",
            "-D__HIP_HCC_COMPAT_MODE__=1",
            "-x", "hip", "-O3",
            "-I", includePath,
            "-std=c++17",
            "-parallel-jobs=64"
        ]

        if self.asanBuild:
            hipFlags.extend(["-fsanitize=address", "-shared-libasan", "-fuse-ld=lld"])
        if self.saveTemps:
            hipFlags.append("--save-temps")
        if os.name == "nt":
            hipFlags.extend(["-fms-extensions", "-fms-compatibility", "-fPIC", "-Wno-deprecated-declarations"])

        archFlags = [f"--offload-arch={gfx}" for gfx in gfxs]

        args = [*launcher, self.compiler, *hipFlags, *archFlags]
        args += srcPaths
        args +=["-o", destPath]

        return self.invoke(args, f"Compiling HIP source kernels into objects (.cpp -> .o)")


    def list(self, sharedObjFile: str):
        """Lists the code objects in shared object.

        Args:
            sharedObjFile: Name of object file to list.

        Returns:
            List of objects embedded in shared object file.
        """
        args = [self.objLs, sharedObjFile]
        return [e.strip().split()[1:] for e in self.invoke(args, f"Listing code objects in shared object", Path(sharedObjFile).parent).decode().split("\n") if e.strip().split()[1:]]


    def extract(self, filename: str):
        """Extracts code objects from a shared object.

        Args:
            objFile: Name pf object file to extract.

        Returns:
            List of target triples in the object file.
        """
        args = [self.objDump, filename]
        return self.invoke(args, f"Extracting code object.", Path(filename.replace("file:","")).parent)


def buildSourceCodeObjectFile(toolchain: SourceToolchain, destPath: Union[Path, str], sharedObjPath: Union[Path, str], ) -> List[str]:
    """Compiles a HIP source code file into a code object file.

    Args:
        cxxCompiler: The C++ compiler to use.
        cxxCompiler: The offload bundler to use.
        outputPath: The output directory path where code objects will be placed.
        kernelPath: The path to the kernel source file.

    Returns:
        List of paths to the created code objects.
    """

    if "CmakeCxxCompiler" in globalParameters and globalParameters["CmakeCxxCompiler"] is not None:
      os.environ["CMAKE_CXX_COMPILER"] = globalParameters["CmakeCxxCompiler"]

    for target, filename in toolchain.list(sharedObjPath):
      match = re.search("gfx.*$", target)
      if match:
        arch = re.sub(":", "-", match.group())
        toolchain.extract(filename)
        src = str(Path(sharedObjPath).parent / (str(Path(filename).name).replace("#","-").replace("=","").replace("&","-") + ".co"))
        dst = str(destPath / f"Kernels.so-000-{arch}.hsaco")
        shutil.move(src, dst)
