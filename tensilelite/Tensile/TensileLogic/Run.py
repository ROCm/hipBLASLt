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
    assignGlobalParameters,
    ParallelMap2,
    print1,
    makeIsaInfoMap,
    SUPPORTED_ISA,
    IsaVersion,
    IsaInfo,
    verbosity
)
from Tensile.Common.GlobalParameters import globalParameters

from Tensile.LibraryIO import readYAML
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.CustomKernels import isCustomKernelConfig, getCustomKernelConfig
from Tensile.SolutionStructs import matrixInstructionToMIParameters
from Tensile import CUSTOM_KERNEL_PATH

from .ParseArguments import parseArguments
from .ValidMatrixInstruction import validateMatrixInstruction
from .ValidWorkGroup import validateWorkGroup


def getParams(isaInfoMap, cxxCompiler):
    gp = globalParameters
    assignGlobalParameters({"PrintSolutionRejectionReason": True}, isaInfoMap, cxxCompiler)
    return gp


def handleCustomKernel(sol: dict, isaInfoMap: dict):
    if not isCustomKernelConfig(sol):
        return None

    name = sol["CustomKernelName"]

    dir = CUSTOM_KERNEL_PATH
    # dir = str(Path(CUSTOM_KERNEL_PATH)/".."/"NEWCustomKernels")
    custom = getCustomKernelConfig(name, {}, dir)
    sol.update(custom)

    mi = sol["MatrixInstruction"]
    print1(f">> FOUND Custom kernel: {name} with MI {mi}")

    if len(mi) == 4:
        print1(f">> --DBG-- -> Success, no need to convert, has MI length {len(mi)}\n---")
        return None

    isa = IsaVersion(*sol["ISA"])
    wavefrontSize = sol["WavefrontSize"]
    ptype = sol["ProblemType"]
    workgroup = sol["WorkGroup"]

    miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap)

    ## Experimental custom kernel config checker code
    import pprint
    yamlstring = convert_pformat_to_condensed(pprint.pformat(miParams))
    print(yamlstring)

    ## end of experimetnal section

    # try:
    #     miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap)
    # except Exception as e:
    #     printWarning(f"Custom kernel {name} failed to convert MI to parameters: {e}")
    #     return None
    sol.update(miParams)
    return sol, yamlstring


def convert_pformat_to_condensed(pformat_str):
    # Remove the dictionary braces and split into lines
    lines = pformat_str.strip('{}').split('\n')

    # Initialize an empty list to store the formatted lines
    formatted_lines = []

    # Iterate over each line
    for line in lines:
        # Remove leading and trailing whitespace
        line = line.strip()

        # Remove the 'ISA' line as it is not needed in the output
        if line.startswith("'ISA'"):
            continue

        line = line.replace("'", "")
        line = line[:-1] if line.endswith(',') else line


        line = '   ' + line

        # Append the formatted line to the list
        formatted_lines.append(line)

    # Join the formatted lines into a single string
    result = '\n'.join(formatted_lines)

    return result

def replace_line_in_file(file_path, search_string, replacement_string):
    """
    Replaces a line in a file that matches the search_string with the replacement_string.

    Args:
        file_path (str): The path to the file.
        search_string (str): The string to search for in the file.
        replacement_string (str): The multi-line string to replace the matching line with.
    """
    # Read the file contents
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Open the file in write mode to overwrite the contents
    with open(file_path, 'w') as file:
        for line in lines:
            if search_string in line:
                # Replace the matching line with the replacement string
                file.write(replacement_string + '\n')
            else:
                # Write the original line
                file.write(line)

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

        # print1(f">> {file.relative_to(logicPath)}")
        for s in solutions:
            s = handleCustomKernel(s, isaInfoMap)

            if s is None:
                continue

            sol, replacement_string = s
            customfile = Path(CUSTOM_KERNEL_PATH) / (sol["CustomKernelName"] + ".s")
            print1(f"## UPDATING {customfile}")
            replace_line_in_file(customfile, "  MatrixInstruction:", replacement_string)


            print1(f"## Custom kernel {file.relative_to(logicPath)}")
            if all(
                [
                    validateMatrixInstruction(sol, isaInfoMap, file.relative_to(logicPath)),
                    validateWorkGroup(sol, isaInfoMap, file.relative_to(logicPath)),
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
