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

from Tensile.Common import CHeader, printExit, state, DepthUConfig
from Tensile.LibraryIO import parseLibraryLogicFile, write
from Tensile.SolutionLibrary import MasterSolutionLibrary
from Tensile.Utilities.RequiredParameters import getRequiredParametersMin

from os import getpid
from pathlib import Path
from typing import Dict, Union


def generateSolutionsAndLibraries(assembler, isaInfoMap, lazy, logicFiles):
    solutions = []
    libraries = []
    for logicFileGroup in logicFiles:
        for logicFile in logicFileGroup[1]:
            libraryLogic = parseLibraryLogicFile(logicFile, assembler, False, False, False, DepthUConfig(), isaInfoMap, lazy)
            solutions.extend(libraryLogic.solutions)
            libraries.append((libraryLogic.architecture, libraryLogic.library))
    return solutions, libraries


def writeAssembly(asmPath: Path, result: tuple):
    if result[0]:
      printExit(f"Failed to generate kernel {result[3]} because it has error code {result[0]}")
    filepath = asmPath / f"{result.name}.s"
    isa =  result[5]
    wfsize = result[6]
    with open(filepath, "w", encoding="utf-8") as f:
      f.write(result[1])
      del result # result.src is very large so let gc know to clean up asap
 
    return filepath, isa, wfsize


def writeHelper(outputPath, kernelHelperObj) -> str:
    name = kernelHelperObj.getKernelName()
    KERNEL_HELPER_FILENAME_CPP = name + ".cpp"
    KERNEL_HELPER_FILENAME_H = name + ".h"
    kernelSourceFilename = str(Path(outputPath) / "Kernels" / KERNEL_HELPER_FILENAME_CPP)
    kernelHeaderFilename = str(Path(outputPath) / "Kernels" / KERNEL_HELPER_FILENAME_H)

    with open(kernelHeaderFilename, "w", encoding="utf-8") as kernelHeaderFile, \
          open(kernelSourceFilename, "w", encoding="utf-8") as kernelSourceFile:
        kernelSourceFile.write(CHeader)
        kernelHeaderFile.write(CHeader)
        kernelSourceFile.write("#include \"{}.h\"\n".format(name))
        kernelHeaderFile.write("#pragma once\n")
        kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
        kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
        kernelHeaderFile.write("#include \"KernelHeader.h\"\n\n")

        if "Enum" not in name:
            kernelHeaderFile.write("#include \"Kernels/TensileActivationEnum_S.h\"\n")
            kernelHeaderFile.write("#include \"Kernels/TensileActivationEnum_I.h\"\n")
            if hasattr(kernelHelperObj, "actGradientPrefix") and kernelHelperObj.actGradientPrefix == "Gradient":
                kernelHeaderFile.write("#include \"Kernels/TensileGradientActivationEnum_S.h\"\n")
        if "TensileActivation_S" not in name and "TensileActivation_I" not in name and "Enum" not in name:
            kernelHeaderFile.write("#include \"Kernels/TensileActivation_S_Hipblaslt_all.h\"\n")
            kernelHeaderFile.write("#include \"Kernels/TensileActivation_I_Hipblaslt_all.h\"\n")
        if "TensileGradientActivation_S" not in name and "Enum" not in name:
            if hasattr(kernelHelperObj, "actGradientPrefix") and kernelHelperObj.actGradientPrefix == "Gradient":
                kernelHeaderFile.write("#include \"Kernels/TensileGradientActivation_S_Hipblaslt_all.h\"\n")

        HeaderText = ""
        (err, src) = kernelHelperObj.getSourceFileString()
        kernelSourceFile.write(src)
        if err:
            print("*** warning: invalid kernel#%u" % name)
        HeaderText += kernelHelperObj.getHeaderFileString()
        kernelHeaderFile.write(HeaderText)

    return kernelSourceFilename


def genLazyMasterSolutionLibrary(libraryPath, libraryFormat, masterLib):
    # Can we do this asynchronously before the call to writeSolutionsAndKernels?
    for name, lib in list(masterLib.lazyLibraries.items()):
        catalogPath = libraryPath / name
        lib.applyNaming(False, getRequiredParametersMin())  # <-- This should be able to be replaced directly with `name`?
        write(str(catalogPath), state(lib), libraryFormat)


def generateParentLibrary(libraryFormat: str, libraryPath: Union[Path, str], masterLibs: Dict[str, MasterSolutionLibrary], lazyLibraryLoading: bool):
    base = "TensileLibrary_lazy_" if lazyLibraryLoading else "TensileLibrary_"
    for arch, masterLib in masterLibs.items():
        name = base + arch
        masterLib.applyNaming(False, getRequiredParametersMin())  # <-- This should be able to be replaced directly with `name`?
        write(str(libraryPath / name), state(masterLib), libraryFormat)
