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


import functools
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict

from Tensile.Common import (
    globalParameters,
    assignGlobalParameters,
    ParallelMap2,
    print1,
    printWarning,
    makeIsaInfoMap,
    SUPPORTED_ISA,
    gfxToIsa,
    IsaVersion,
    IsaInfo,
)
from Tensile.LibraryIO import readYAML
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.CustomKernels import isCustomKernelConfig, getCustomKernelConfig
from Tensile.SolutionStructs import Solution, matrixInstructionToMIParameters

from .ParseArguments import parseArguments
from .ValidMatrixInstruction import validateMatrixInstruction
from .ValidWorkGroup import validateWorkGroup


def getParams(isaInfoMap, cxxCompiler):
    gp = globalParameters
    assignGlobalParameters({"PrintSolutionRejectionReason": True}, isaInfoMap, cxxCompiler)
    return gp


def handleCustomKernel(sol: dict, isaInfoMap: dict) -> dict | None:
    if not isCustomKernelConfig(sol):
        return None

    name = sol["CustomKernelName"]
    print1(f">>     Custom kernel: {name}")

    custom = getCustomKernelConfig(name, {})
    sol.update(custom)

    mi = sol["MatrixInstruction"]
    if len(mi) != 9:
        printWarning(f"Custom kernel {name} has MI length {len(mi)}, expected 9.")

    isa = sol["ISA"]
    wavefrontSize = sol["WavefrontSize"]
    ptype = sol["ProblemType"]
    workgroup = sol["WorkGroup"]

    miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap)
    sol.update(miParams)
    return sol


def runChecks(logicPath: str, isaInfoMap: Dict[IsaVersion, IsaInfo], files: List[Path]):
    """
    Run checks on the given files.

    Args:
        logicPath: Path to the logic directory.
        gp: Global parameters.
        files: List of files to check.

    Returns:
        Tuple of (keep, total) where keep is the number of solutions to keep and
        total is the total number of solutions.
    """
    keep, total = 0, 0
    for file in files:
        if "Experimental" in file.parts:
            return keep, total

        solutions = readYAML(file)[5]  # Solutions are the 5th index
        print1(f">> {file.relative_to(logicPath)}")

        for s in solutions:
            s = handleCustomKernel(s, isaInfoMap)

            if s is None:
                continue

            if all(
                [
                    validateMatrixInstruction(s, isaInfoMap, file.relative_to(logicPath)),
                    validateWorkGroup(s, isaInfoMap, file.relative_to(logicPath)),
                ]
            ):
                keep += 1
            total += 1
    return keep, total


def main():
    args = parseArguments()
    if not any([args.Check]):
        print1("No checks specified. Exiting.")
        exit(0)

    jobs = int(args.Jobs)
    cxxCompiler = validateToolchain(args.CxxCompiler)

    isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)
    assignGlobalParameters({"PrintSolutionRejectionReason": True}, isaInfoMap)

    logicPath = Path(args.LogicPath)
    pattern = "**/*.yaml"
    files = list(logicPath.glob(pattern))

    batchSize = len(files) // jobs
    batches = (files[i : i + batchSize] for i in range(0, len(files), batchSize))

    fn = functools.partial(runChecks, logicPath, isaInfoMap)
    keep, total = 0, 0
    with Pool(processes=jobs) as pool:
        results = pool.map_async(fn, batches)

        # TIP: This is how to use joblib. Leave for reference.
        # for _keep, _total in ParallelMap2(
        #     fn, batches, multiArg=False, procs=jobs, return_as="generator_unordered"
        # ):

        for _keep, _total in results.get():
            keep += _keep
            total += _total

    rejects = total - keep
    print(f"Total  {total} solutions")
    print(f"Keep   {keep} solutions")
    print(f"Reject {rejects} solutions")

    if rejects > 0:
        exit(1)
