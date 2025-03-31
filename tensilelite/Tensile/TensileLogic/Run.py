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

from copy import deepcopy
import functools

from pathlib import Path
from typing import List, Dict, NamedTuple, Callable, Optional

from Tensile.Common import ParallelMap2, print1, IsaVersion, IsaInfo, setVerbosity, elineno
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters
from Tensile.LibraryIO import readYAML, writeYAML
from Tensile.SolutionStructs.Validators.MatrixInstruction import validateMIParameters
from Tensile.SolutionStructs.Validators.WorkGroup import validateWorkGroup
from Tensile.SolutionStructs.Validators.KernelName import validateKernelName
from Tensile.Toolchain.Validators import validateToolchain

from .ParseArguments import parseArguments
from .HandleCustomKernel import handleCustomKernel, hasCustomKernel


class Action(NamedTuple):
    UpdateBuildKernels: bool
    CheckOnlyCustomKernels: bool
    CheckAll: bool


def _makeValidator(func: Callable) -> Callable:
    def validator(filepath: Path, sol: dict, *args):
        try:
            ret = func(sol, *args)
            assert sol["Valid"], f"Solution was rejected: {elineno()}"
            return ret
        except AssertionError as e:
            print(
                f"Error: Validation failed: {e} (file: {filepath}, index: {sol['SolutionIndex']})"
            )
            return False

    return validator


_validateMatrixInstruction = _makeValidator(validateMIParameters)
_validateWorkGroup = _makeValidator(validateWorkGroup)
_validateKernelName = _makeValidator(validateKernelName)


def _readFile(file: Path, logicPath: Path, action: Action) -> Optional[List[dict]]:
    """
    Get solutions from a logic file depending on the checks specified.

    Returns:
        List of solutions if the file is a custom kernel is found and CheckOnlyCustomKernel is
        enabled, or if all checks are enabled. Otherwise, an empty list is returned.
    """
    if action.CheckOnlyCustomKernels and hasCustomKernel(file):
        print1(f">> Checking: {file.relative_to(logicPath)}")
        return readYAML(file)
    elif action.CheckAll:
        print1(f">> Checking: {file.relative_to(logicPath)}")
        return readYAML(file)
    elif action.UpdateBuildKernels:
        print1(f">> Updating: {file.relative_to(logicPath)}")
        return readYAML(file)
    return None


def _runUpdates(logicPath: Path, action: Action, files: List[Path]):
    """
    Run updates to be conducted in-place on the given logic files.

    Args:
        logicPath: Path to a directory containing logic files or to an individual logic file.
        action: Object containing flags for checking.
        files: List of logic files to update.
    """
    kernelBuildSet = set()
    for file in files:
        if "Experimental" in file.parts:
            return 0

        yaml = _readFile(file, logicPath, action)
        if yaml:
            for s in yaml[5]:
                name = s["KernelNameMin"]
                if name in kernelBuildSet:
                    if "BuildKernel" in s:
                        del s["BuildKernel"]
                        print(f"  - removing `BuildKernel`: (file: {file.relative_to(logicPath)}, index: {s['SolutionIndex']})")
                else:
                    if "BuildKernel" not in s:
                        s["BuildKernel"] = True
                        print(f"  + adding `BuildKernel`: (file: {file.relative_to(logicPath)}, index: {s['SolutionIndex']})")
                    elif not s["BuildKernel"]:
                        raise ValueError("False values for `BuildKernel` are not permitted, remove `BuildKernel` to express False")
                    kernelBuildSet.add(name)
            writeYAML(file, yaml)


def _runChecks(
    logicPath: Path, isaInfoMap: Dict[IsaVersion, IsaInfo], action: Action, files: List[Path]
):
    """
    Run checks on the given logic files.

    Args:
        logicPath: Path to a directory containing logic files or to an individual logic file.
        isaInfoMap: Map of IsaVersion to IsaInfo.
        action: Object containing flags for checking.
        files: List of logic files to check.

    Returns:
        Tuple of (keep, total, numBuildKernels, names, numNames) where `keep` is the number of
        unrejected solutions, `total` is the total number of solutions parsed, `numBuildKernels`
        is the number of solutions with `BuildKernel` set to True, `names` is a list of unique kernel names,
        and `numNames` is the count of unique kernel names.
    """
    keep, total = 0, 0
    numBuildKernels, numNames, names = 0, 0, []
    for file in files:
        if "Experimental" in file.parts:
            return keep, total, numBuildKernels, names, numNames

        yaml = _readFile(file, logicPath, action)
        if yaml:
            for s in yaml[5]:  # Solutions are the 5th index
                s, isCustom = handleCustomKernel(s, isaInfoMap)
                if action.CheckOnlyCustomKernels and not isCustom:
                    continue

                # Rejection checks
                if all(
                    [
                        _validateMatrixInstruction(file.relative_to(logicPath), s, isaInfoMap),
                        _validateWorkGroup(file.relative_to(logicPath), s),
                    ]
                ):
                    keep += 1
                total += 1

                # Uniqueness checks
                numBuildKernels += int(s.get("BuildKernel", False))
                if _validateKernelName(file.relative_to(logicPath), s):
                    names.append(s["KernelNameMin"])
                    numNames += 1

    return keep, total, numBuildKernels, names, numNames


def _getLogicFiles(logicPath: Path) -> List[Path]:
    if logicPath.is_file() and logicPath.suffix == ".yaml":
        files = [logicPath]
    else:
        pattern = "**/*.yaml"
        files = list(logicPath.glob(pattern))
    if len(files) == 0:
        print1(f"No files found in {logicPath}")
        exit(1)
    print1(f"Found {len(files)} files")
    return files


def _setup():
    args = parseArguments()

    setVerbosity(args.verbose)
    jobs = int(args.jobs)
    cxxCompiler = validateToolchain(args.cxx_compiler)
    logicPath = Path(args.logic_path)

    # Setup checks and updates
    if not any([args.check_all, args.check_only_custom_kernels, args.update_build_kernels]):
        print1("No actions specified. Exiting.")
        exit(1)
    action = Action(
        CheckOnlyCustomKernels=args.check_only_custom_kernels,
        CheckAll=args.check_all,
        UpdateBuildKernels=args.update_build_kernels,
    )

    files = _getLogicFiles(logicPath)

    # Retrieve ISA info and globals
    isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, str(cxxCompiler))
    assignGlobalParameters({"PrintSolutionRejectionReason": True}, isaInfoMap)

    return jobs, isaInfoMap, logicPath, files, action


def main():
    jobs, isaInfoMap, logicPath, files, action = _setup()

    if action.UpdateBuildKernels:
        # Updates
        fn = functools.partial(_runUpdates, logicPath, action)
        fn(files)  # Must be syncronous for proper uniqueness evaluation of kernel names

    if any([action.CheckOnlyCustomKernels, action.CheckAll]):
        batchSize = len(files) // min(len(files), jobs)
        batches = (files[i : i + batchSize] for i in range(0, len(files), batchSize))

        fn = functools.partial(_runChecks, logicPath, isaInfoMap, action)

        keep, total = 0, 0
        numBuildKernels, numNames, names = 0, 0, []
        results = ParallelMap2(fn, batches, multiArg=False, procs=jobs, return_as="list")
        for _keep, _total, _buildk, _names, _numNames in results:
            keep += _keep
            total += _total
            numBuildKernels += _buildk
            names.extend(_names)
            numNames += _numNames

        # Post processing
        rejects = total - keep
        print(f"  Total    {total} solutions")
        print(f"  Keep     {keep} solutions")
        print(f">>  Reject {rejects} solution(s)")

        buildkDiff = abs(numBuildKernels - len(set(names)))
        print(f"  Num names (batched)   {numNames}")
        print(f"  Unique names          {len(set(names))}")
        print(f"  With BuildKernel      {numBuildKernels}")
        print(f">>  Difference          {buildkDiff}")

        if rejects > 0 or buildkDiff > 0:
            exit(1)
