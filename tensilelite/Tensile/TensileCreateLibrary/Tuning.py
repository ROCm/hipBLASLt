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
import rocisa

from Tensile.Common import CHeader, print1, print2, printExit, \
                           ParallelMap2, ensurePath, tqdm, ParallelMapConfig, getVerbosity
from Tensile.Common.GlobalParameters import globalParameters
from Tensile.KernelWriterBase import KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H
from Tensile.SolutionStructs import Solution, getKernelFileBase, getKeyNoInternalArgs
from Tensile.Toolchain.Assembly import buildAssemblyCodeObjectFiles
from Tensile.Toolchain.Source import buildSourceCodeObjectFile
from .IO import writeAssembly
from .Run import _processKernelSource

from functools import partial, reduce
from itertools import repeat
from pathlib import Path
from typing import List

def _writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H):
    kernelSourceFilename = str(Path(outputPath) / KERNEL_HELPER_FILENAME_CPP)
    kernelHeaderFilename = str(Path(outputPath) / KERNEL_HELPER_FILENAME_H)

    with open(kernelHeaderFilename, "w", encoding="utf-8") as kernelHeaderFile, \
          open(kernelSourceFilename, "w", encoding="utf-8") as kernelSourceFile:  
        kernelSourceFile.write(CHeader)
        kernelHeaderFile.write(CHeader)
        kernelSourceFile.write("#include \"Kernels.h\"\n")
        kernelHeaderFile.write("#pragma once\n")
        kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
        kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
        kernelHeaderFile.write("#include \"KernelHeader.h\"\n\n")  
        HeaderText = ""
        for ko in kernelHelperObjs:
            kernelName = ko.getKernelName()
            (err, src) = ko.getSourceFileString()
            kernelSourceFile.write(src)
            if err:
                print("*** warning: invalid kernel#%u" % kernelName)
            HeaderText += ko.getHeaderFileString()
        kernelHeaderFile.write(HeaderText)


def removeInvalidSolutionsAndKernels(results, kernels, solutions, errorTolerant, printLevel: bool, splitGSU: bool):
    removeKernels = []
    removeKernelNames = []
    removeSolutions = []
    removeResults = []

    for kernIdx, r in (
        tqdm(enumerate(results)) if printLevel > 1 else enumerate(results)
    ):
        if r.err != 0:
            if not errorTolerant:
                print(
                    "\nKernel generation failed for kernel: {}".format(
                        kernels[kernIdx]["SolutionIndex"]
                    )
                )
                print(kernels[kernIdx]["SolutionNameMin"])
            removeKernels.append(kernels[kernIdx])
            kName = getKeyNoInternalArgs(kernels[kernIdx], splitGSU)
            if kName not in removeKernelNames:
                removeKernelNames.append(kName)
            removeResults.append(results[kernIdx])

    if len(removeKernels) > 0 and not errorTolerant:
        printExit("** kernel generation failure **")

    for kern in removeKernels:
        kernels.remove(kern)

    for solution in (
        tqdm(solutions, "Finding invalid solutions")
        if printLevel > 1
        else solutions
    ):
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, splitGSU)
            if kName in removeKernelNames:
                removeSolutions.append(solution)
                break

    for solut in removeSolutions:
        solutions.remove(solut)

    for rel in removeResults:
        results.remove(rel)


def writeSolutionsAndKernels(
    outputPath,
    asmToolchain,
    srcToolchain,
    solutions,
    kernels,
    kernelHelperObjs,
    kernelWriterAssembly,
    splitGSU: bool,
    cmdlineArchs: List[str],
    kernelSerialNaming,
    kernelMinNaming,
    errorTolerant=False,
    generateSourcesAndExit=False,
    compress=True,
    useShortNames=False,
):
    codeObjectFiles = []

    outputPath = Path(outputPath)
    destLibPath = ensurePath(
        outputPath / "library"
    )  # Destination for code object library files (.co)
    buildTmpPath = ensurePath(outputPath / "build_tmp" / outputPath.stem.upper())  #
    assemblyTmpPath = ensurePath(
        buildTmpPath / "assembly"
    )  # Temp path for generated assembly files (.s)
    objectTmpPath = ensurePath(
        buildTmpPath / "code_object_tmp"
    )  # Temp path for HSA code object files (.hsaco)

    asmKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]

    visited = set()
    duplicates = 0
    for k in asmKernels:
        base = getKernelFileBase(useShortNames, splitGSU, kernelMinNaming, kernelSerialNaming, k)
        print1(base)
        k.duplicate = True if base in visited else False
        if not k.duplicate:
            k["BaseName"] = base
        duplicates += k.duplicate
        print2(f"Duplicate: {base}")
        visited.add(base)
    print1(f"Number of duplicate kernels: {duplicates}")

    numAsmKernels = len(asmKernels)
    numKernels = len(asmKernels)
    assert numKernels == numAsmKernels, "Only assembly kernels are supported in TensileLite"
    asmIter = zip(
        repeat(kernelWriterAssembly),
        repeat(rocisa.rocIsa.getInstance().getData()),
        repeat(useShortNames),
        repeat(splitGSU),
        repeat(kernelMinNaming),
        repeat(kernelSerialNaming),
        asmKernels
    )
    config = ParallelMapConfig(message="Generating assembly kernels", return_as="list", multiArg=True)
    asmResults = ParallelMap2(_processKernelSource, config, asmIter)
    removeInvalidSolutionsAndKernels(
        asmResults, asmKernels, solutions, errorTolerant, getVerbosity(), splitGSU
    )
    print1(f"After removal: {len(asmKernels)}")
    def assemble(ret):
        p, isa, wavefrontsize = ret
        asmToolchain.assembler(rocisa.isaToGfx(isa), wavefrontsize, str(p), str(p.with_suffix(".o")))

    unaryWriteAssembly = partial(writeAssembly, assemblyTmpPath)
    compose = lambda *F: reduce(lambda f, g: lambda x: f(g(x)), F)
    config = ParallelMapConfig(message="Writing assembly kernels", return_as="list", multiArg=False)
    ret = ParallelMap2(
        compose(assemble, unaryWriteAssembly),
        config, 
        asmResults,
    )

    _writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)

    if not generateSourcesAndExit:
        codeObjectFiles += buildAssemblyCodeObjectFiles(
            asmToolchain.linker,
            asmToolchain.bundler,
            globalParameters["ROCmLdPath"],
            destLibPath,
            assemblyTmpPath,
            compress,
            kernelMinNaming,
            asmKernels,
        )
        kernelsLib = str(objectTmpPath / "Kernels.so")
        kernelsSrc = [str(outputPath / "Kernels.cpp")]
        srcToolchain.compiler(kernelsSrc, str(kernelsLib), str(outputPath), cmdlineArchs)
        buildSourceCodeObjectFile(
            srcToolchain,
            destLibPath,
            kernelsLib,
        )

    return codeObjectFiles, numKernels
