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


import yaml
import functools
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict

from Tensile.Common import (
    ParallelMap2,
    print1,
    IsaVersion,
    IsaInfo,
    verbosity
)

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters

from Tensile.LibraryIO import readYAML
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.CustomKernels import isCustomKernelConfig, getCustomKernelConfig
from Tensile import CUSTOM_KERNEL_PATH

from .ParseArguments import parseArguments
from .ValidMatrixInstruction import _validateMatrixInstruction
from .ValidWorkGroup import _validateWorkGroup


def handleCustomKernel(sol: dict, isaInfoMap: dict):
    if not isCustomKernelConfig(sol):
        return sol

    name = sol["CustomKernelName"]
    dir = CUSTOM_KERNEL_PATH
    config = getCustomKernelConfig(name, {}, dir)
    sol.update(config)

    mi = sol["MatrixInstruction"]
    print1(f">>     Found custom kernel: {name} with MI {mi}")

    if not (len(mi) == 4 or len(mi) == 0):
        raise ValueError(f">> Error: Custom kernels should have matrix instruction of length 4, or none at all, not length {len(mi)}\n{name}")

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

            if all(
                [
                    _validateMatrixInstruction(s, isaInfoMap, file.relative_to(logicPath)),
                    _validateWorkGroup(s, isaInfoMap, file.relative_to(logicPath)),
                ]
            ):
                keep += 1
            total += 1
    return keep, total


def main():
    args = parseArguments()

    global verbosity
    verbosity = args.Verbose

    jobs = int(args.Jobs)
    cxxCompiler = validateToolchain(args.CxxCompiler)

    isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)
    assignGlobalParameters({"PrintSolutionRejectionReason": True}, isaInfoMap)

    logicPath = Path(args.LogicPath)
    if logicPath.is_file() and logicPath.suffix == ".yaml":
        files = [logicPath]
    else:
        pattern = "**/*.yaml"
        files = list(logicPath.glob(pattern))

    if not any([args.Check]):
        print1("No checks specified. Exiting.")
        exit(0)
    if len(files) == 0:
        print1(f"No files found in {logicPath}")
        exit(1)

    print1(f"Found {len(files)} files")

    batchSize = len(files) // min(len(files), jobs)
    batches = (files[i : i + batchSize] for i in range(0, len(files), batchSize))

    fn = functools.partial(runChecks, logicPath, isaInfoMap)
    keep, total = 0, 0
    # with Pool(processes=jobs) as pool:
    #     results = pool.map_async(fn, batches)

    # # TIP: This is how to use joblib. Leave for reference.
    results = ParallelMap2(
        fn, batches, multiArg=False, procs=jobs, return_as="list"
    )

    for _keep, _total in results:
        # for _keep, _total in results.get():
        keep += _keep
        total += _total

    rejects = total - keep
    print(f"Total  {total} solutions")
    print(f"Keep   {keep} solutions")
    print(f"Reject {rejects} solutions")

    if rejects > 0:
        exit(1)
