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

import collections
import math
import os
import shlex
import shutil
import subprocess

from pathlib import Path
from typing import List, Union

from ..Common import globalParameters, print2, ensurePath, SemanticVersion, isaToGfx
from ..KernelWriterAssembly import KernelWriterAssembly
from ..Toolchain.Validators import getVersion
from ..SolutionStructs import Solution

class AssemblyToolchain:
    def __init__(self, assembler: str, bundler: str, buildIdKind: str, coVersion: str):
        self.assembler = assembler
        self.assemblerVersion = SemanticVersion(*[int(c) for c in getVersion(assembler).split(".")[:3]])
        self.bundler = bundler
        self.buildIdKind = buildIdKind
        self.coVersion = coVersion

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

    def assemble(self, srcPath: str, destPath: str, gfx: str, wavefrontSize: int, debug: bool=False):
      """Assemble an assembly source file into an object file.

      Args:
          srcPath: The path to the assembly source file.
          destPath: The destination path for the generated object file.
          coVersion: The code object version to use.
          isa: The target GPU architecture in ISA format.
          wavefrontSize: The wavefront size to use.
      """
      launcher = shlex.split(os.environ.get('Tensile_ASM_COMPILER_LAUNCHER', ''))
      args = [
          *launcher,
          self.assembler,
          "-x", "assembler",
          "--target=amdgcn-amd-amdhsa",
          f"-mcode-object-version={self.coVersion}",
          f"-mcpu={gfx}",
          "-mwavefrontsize64" if wavefrontSize == 64 else "-mno-wavefrontsize64"
          "-g" if debug else "",
          "-c",
          "-o", destPath, srcPath
      ]

      return self.invoke(args, "Assembling assembly source code into object file (.s -> .o)")

    def link(self, srcPaths: List[str], destPath: str):
        """Links object files into a code object file.

        Args:
            srcPaths: A list of paths to object files.
            destPath: A destination path for the generated code object file.

        Raises:
            RuntimeError: If linker invocation fails.
        """
        if os.name == "nt":
            # Use args file on Windows b/c the command may exceed the limit of 8191 characters
            with open(Path.cwd() / "clang_args.txt", "wt") as file:
                file.write(" ".join(objFiles))
                file.flush()
            args = [
                self.assembler,
                "--target=amdgcn-amd-amdhsa",
                "-o", destPath, "@clang_args.txt"]
        else:
            args = [
                self.assembler,
                "--target=amdgcn-amd-amdhsa",
                "-Xlinker", f"--build-id={self.buildIdKind}",
                "-o", destPath, *srcPaths
            ]

        return self.invoke(args, "Linking assembly object files into code object (*.o -> .co)")

    def compress(self, srcPath: str, destPath: str, gfx: str):
        """Compresses a code object file using the provided bundler.

        Args:
            srcPath: The source path of the code object file to be compressed.
            destPath: The destination path for the compressed code object file.
            gfx: The target GPU architecture.

        Raises:
            RuntimeError: If compressing the code object file fails.
        """
        args = [
            self.bundler,
            "--compress",
            "--type=o",
            "--bundle-align=4096",
            f"--targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa-unknown-{gfx}",
            "--input=/dev/null",
            f"--input={srcPath}",
            f"--output={destPath}",
        ]

        return self.invoke(args, "Bundling/compressing code object file (.co -> .co)")


def _batchObjectFiles(objFiles: List[str], coPathDest: Union[Path, str], maxObjFiles: int=10000) -> List[str]:
    numObjFiles = len(objFiles)

    if numObjFiles <= maxObjFiles:
      return objFiles

    batchedObjFiles = [objFiles[i:i+maxObjFiles] for i in range(0, numObjFiles, maxObjFiles)]
    numBatches = int(math.ceil(numObjFiles / maxObjFiles))

    newObjFiles = [str(coPathDest) + "." + str(i) for i in range(0, numBatches)]
    newObjFilesOutput = []

    for batch, filename in zip(batchedObjFiles, newObjFiles):
      if len(batch) > 1:
        args = [globalParameters["ROCmLdPath"], "-r"] + batch + [ "-o", filename]
        print2(f"Linking object files into fewer object files: {' '.join(args)}")
        subprocess.check_call(args)
        newObjFilesOutput.append(filename)
      else:
        newObjFilesOutput.append(batchedObjFiles[0])

    return newObjFilesOutput

def buildAssemblyCodeObjectFiles(
      toolchain: AssemblyToolchain,
      kernels: List[Solution],
      writer: KernelWriterAssembly,
      destDir: Union[Path, str],
      asmDir: Union[Path, str],
      compress: bool=True
    ):
    """Builds code object files from assembly files

    Args:
        toolchain: The assembly toolchain object to use for building.
        kernels: A list of the kernel objects to build.
        writer: The KernelWriterAssembly object to use.
        destDir: The destination directory for the code object files.
        asmDir: The directory containing the assembly files.
        compress: Whether to compress the code object files.
    """

    isAsm = lambda k: k["KernelLanguage"] == "Assembly"

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"

    destDir = Path(ensurePath(destDir))
    asmDir = Path(ensurePath(asmDir))

    archKernelMap = collections.defaultdict(list)
    for k in filter(isAsm, kernels):
      archKernelMap[tuple(k['ISA'])].append(k)

    coFiles = []
    for arch, archKernels in archKernelMap.items():
      if len(archKernels) == 0:
        continue

      gfx = isaToGfx(arch)

      objectFiles = [str(asmDir / (writer.getKernelFileBase(k) + extObj)) for k in archKernels if 'codeObjectFile' not in k]
      coFileMap = collections.defaultdict(list)
      if len(objectFiles):
        coFileMap[asmDir / ("TensileLibrary_"+ gfx + extCoRaw)] = objectFiles
      for kernel in archKernels:
        coName = kernel.get("codeObjectFile", None)
        if coName:
          coFileMap[asmDir / (coName + extCoRaw)].append(str(asmDir / (writer.getKernelFileBase(kernel) + extObj)))
      for coFileRaw, objFiles in coFileMap.items():
        objFiles = _batchObjectFiles(objFiles, coFileRaw)
        toolchain.link(objFiles, str(coFileRaw))
        coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
        if compress:
          toolchain.compress(str(coFileRaw), str(coFile), gfx)
        else:
          shutil.move(coFileRaw, coFile)
        coFiles.append(coFile)

    return coFiles
