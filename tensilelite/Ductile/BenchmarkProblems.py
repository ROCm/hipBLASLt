################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import glob
import os
import shutil
import sys
import time

from copy import deepcopy
from pathlib import Path
from typing import Dict

from Tensile import ClientExecutable, LibraryIO
from Tensile.KernelWriter import DebugConfig
from Tensile.Toolchain.Component import Assembler
from Tensile.SolutionStructs.Problem import ProblemType, ProblemSizes
from Tensile.SolutionStructs.Solution import Solution
from Tensile.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters, validateMIParameters
from Tensile.SolutionStructs.Naming import getKeyNoInternalArgs, getSolutionNameMin, getKernelNameMin
from Tensile.ClientWriter import runClient
from Tensile.BenchmarkStructs import BenchmarkProcess
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Toolchain.Assembly import AssemblyToolchain
from Tensile.Toolchain.Source import SourceToolchain
from Tensile.Common import HR, print1, print2, IsaInfo, IsaVersion, \
        printExit, printWarning, ensurePath, state, \
        BENCHMARK_PROBLEMS_DIR, BENCHMARK_DATA_DIR
from Tensile.Common.GlobalParameters import globalParameters, startTime

from Tensile.BenchmarkProblems import writeBenchmarkFiles

import math
import numpy as np
import pandas as pd
from functools import partial

from .ductile import config as config
from .ductile.core import SearchSpace, Selection, Crossover, Mutation, Mating, Survival
from .ductile.algorithm import GeneticAlgorithm


def validate_solution(problemType, 
                      constantParams, 
                      assembler: Assembler,
                      debugConfig: DebugConfig, 
                      isaInfoMap: Dict[IsaVersion, IsaInfo],
                      perm):
    solution = {
            "ProblemType": deepcopy(problemType.state),
            "ISA": next(iter(isaInfoMap.keys()))
        }
    solution.update(constantParams)
    
    solution.update(perm)
    for p in perm:
        if p.startswith("group_"):
            solution.update(perm[p])

    mi = solution["MatrixInstruction"]
    wavefrontSize = solution["WavefrontSize"]
    workgroup = solution["WorkGroup"]
    ptype = solution["ProblemType"]
    isa = solution["ISA"]

    if len(mi) == 9:
        miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap)
        solution.update(miParams)
    elif len(mi) == 0:
        solution["EnableMatrixInstruction"] = False

    if not validateMIParameters(solution, isaInfoMap):
        return False
    
    
    solutionObject = Solution(
        solution,
        debugConfig.splitGSU,
        debugConfig.printSolutionRejectionReason,
        debugConfig.printIndexAssignmentInfo,
        assembler,
        isaInfoMap
    )
    if not solutionObject["Valid"]:
        return False
    
    kernelWriterAssembly = KernelWriterAssembly(assembler, debugConfig)
    try:
        kernelWriterAssembly._initKernel(solutionObject, {}, {})
        # kernelWriterAssembly._getKernelSource(solutionObject)
    except RuntimeError:
        return False
    
    return True


def generateGASolutions(problemType, constantParams, forkPermutations, assembler: Assembler, \
                            debugConfig: DebugConfig, isaInfoMap: Dict[IsaVersion, IsaInfo]):
    """Creates a list with a Solution object for each parameter combination in forkPermutations"""
    print1("# Enumerating Solutions")

    solutions = []
    solutionSet = set()
    for perm in forkPermutations:
        # Expect only a single ISA in the map for the Tensile context
        # because the GPU has to be physically present for benchmarking

        solution = {
            "ProblemType": deepcopy(problemType.state),
            "ISA": next(iter(isaInfoMap.keys()))
        }
        solution.update(constantParams)
        solution.update(perm)
        for p in perm:
            if p.startswith("group_"):
                solution.update(perm[p])


        mi = solution["MatrixInstruction"]
        wavefrontSize = solution["WavefrontSize"]
        workgroup = solution["WorkGroup"]
        ptype = solution["ProblemType"]
        isa = solution["ISA"]

        if len(mi) == 9:
            miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap)
            solution.update(miParams)
        elif len(mi) == 0:
            solution["EnableMatrixInstruction"] = False

        if validateMIParameters(solution, isaInfoMap):
            solutionObject = Solution(
                solution,
                debugConfig.splitGSU,
                debugConfig.printSolutionRejectionReason,
                debugConfig.printIndexAssignmentInfo,
                assembler,
                isaInfoMap
            )
            if solutionObject["Valid"]:
                if solutionObject not in solutionSet:
                    solutionSet.add(solutionObject)
                    solutions.append(solutionObject)
                else:
                    solutions.append(None)
            else:
                solutions.append(None)
                    
        elif debugConfig.printSolutionRejectionReason:
            print1("rejecting solution " + str(solution))
            solutions.append(None)
        else:
            solutions.append(None)

    return solutions



def _benchmarkProblemType(ductileConfig, problemTypeConfig, problemSizeGroupConfig, problemSizeGroupIdx, useCache,
                         asmToolchain: AssemblyToolchain, srcToolchain: SourceToolchain, cCompiler: str,
                         buildTmpPath: Path, benchmarkProblemsPath: Path,
                         debugConfig: DebugConfig, deviceId: int,
                         gfxName: str, isaInfoMap: Dict[str, IsaInfo]
    ):
    """Run the benchmarking for a single entry in the BenchmarkProblems of a Tensile config"""
    benchmarkTestFails = 0

    print1("")
    print1(HR)
    print1("# Converting Config to BenchmarkProcess Object")
    print1(HR)
    print1("")
    benchmarkProcess = BenchmarkProcess(problemTypeConfig, problemSizeGroupConfig, debugConfig.printIndexAssignmentInfo)

    enableTileSelection = benchmarkProcess.problemType["TileAwareSelection"]
    groupName = "{}_{:02d}".format(str(benchmarkProcess.problemType), problemSizeGroupIdx)
    groupNamePath = benchmarkProblemsPath / groupName
    ensurePath(groupNamePath / "Data")

    totalBenchmarkSteps = len(benchmarkProcess)
    resultsFileBaseFinal = None

    print1("# NumBenchmarkSteps: {}".format(totalBenchmarkSteps))
    print1("")
    print1(HR)
    print1("# Done Creating BenchmarkProcess Object")
    print1(HR)

    for benchmarkStepIdx in range(0, totalBenchmarkSteps):
        benchmarkStep = benchmarkProcess[benchmarkStepIdx]
        stepName = str(benchmarkStep)
        shortName = stepName

        print1("\n")
        print1(HR)
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        print1("# Benchmark Step: {} - {} {:.3f}s".format(groupName, stepName, elapsedTime))
        print1("# Num Sizes: {}".format(benchmarkStep.problemSizes.totalProblemSizes))
        print1("# Factor Dim steps: {}".format(benchmarkStep.factorDimArgs.totalProblemSizes))
        print1("# Bias Type steps: {}".format(benchmarkStep.biasTypeArgs.totalProblemSizes))
        print1("# Activation steps: {}".format(benchmarkStep.activationArgs.totalProblemSizes))
        print1("# ICacheFlush steps: {}".format(len(benchmarkStep.icacheFlushArgs)))
        print1("# Fork Parameters:")
        for k, v in benchmarkStep.forkParams.items():
            print1("#     {}: {}".format(k, v))
        if benchmarkStep.internalSupportParams:
            print("# InternalSupportParams: {}".format(benchmarkStep.internalSupportParams))

        shortNamePath = ensurePath(groupNamePath / shortName)
        stepBaseDir = shortNamePath
        resultsFileBase = os.path.normpath(shortNamePath / ".." / "Data" / shortName)

        if benchmarkStep.isFinal():
            resultsFileBaseFinal = resultsFileBase
        resultsFileName = resultsFileBase + ".csv"
        solutionsFileName = resultsFileBase + ".yaml"

        # check if a solution cache exists and if it matches our solution parameters
        cachePath = os.path.join(stepBaseDir, "cache.yaml")
        sourcePath = ensurePath(shortNamePath / "source")

        def _evaluate(X):
            benchmarkTestFails = 0
            regSolutions = generateGASolutions(benchmarkProcess.problemType, \
                    benchmarkStep.constantParams, X, asmToolchain.assembler, \
                        debugConfig, isaInfoMap)
            
            solutions = []
            for i, rs in enumerate(regSolutions):
                if rs is None:
                    continue
                rs.solIdx = i
                solutions.append(rs)        

            print1("# Actual Solutions: {} / {} after SolutionStructs\n" \
                .format(len(solutions), len(X)))

            # handle no valid solutions
            if len(solutions) == 0:
                return np.zeros((len(X),))

            #cleanup previous runs
            shutil.rmtree(sourcePath)

            # write benchmarkFiles
            prevCount = len(solutions)
            codeObjectFiles = writeBenchmarkFiles(stepBaseDir, solutions, \
                    benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs, \
                    benchmarkStep.factorDimArgs, benchmarkStep.activationArgs, \
                    benchmarkStep.icacheFlushArgs, shortName, [], asmToolchain, srcToolchain, \
                    sourcePath, debugConfig, deviceId, gfxName, isaInfoMap)
            # ^ this mutates solutions

            # write cache data
            cacheData = {
                "CodeObjectFiles": codeObjectFiles,
                "ConstantParams": benchmarkStep.constantParams,
                "ForkParams": benchmarkStep.forkParams,
                "ParamGroups": benchmarkStep.paramGroups,
                "CustomKernels": benchmarkStep.customKernels,
                "CustomKernelWildcard": benchmarkStep.customKernelWildcard
            }
            LibraryIO.writeYAML(cachePath, cacheData)

            print1("# Actual Solutions: {} / {} after KernelWriter\n" \
                    .format(len(solutions), prevCount ))

            # add SolutionIndex and SolutionNameMin into benchmark yaml
            for i in range(0, len(solutions)):
                solution = solutions[i]
                solution["SolutionIndex"] = i
                solution["SolutionNameMin"] = getSolutionNameMin(solution, debugConfig.splitGSU)
                solution["KernelNameMin"]   = getKernelNameMin(solution, debugConfig.splitGSU)
        

            # I think the size portion of this yaml could be removed,
            # but for now it's needed, so we update it even in the cache case
            LibraryIO.writeSolutions(solutionsFileName, benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs,
                benchmarkStep.activationArgs, solutions, False)

            libraryLogicPath = None
            forBenchmark = True
            returncode = runClient(libraryLogicPath, forBenchmark, enableTileSelection, srcToolchain.compiler, cCompiler, shortNamePath)

            if returncode:
                benchmarkTestFails += 1
                printWarning("BenchmarkProblems: Benchmark Process exited with code {}" \
                        .format(returncode))
            
            df = pd.read_csv(resultsFileBaseFinal + ".csv")
            
            cols = [c for c in df.columns.tolist() if c.lstrip().startswith("Cijk_")]
            assert len(cols) == len(solutions)
            
            n_sizes = df.shape[0]
            scores = df[cols].values.astype(np.float32)
            if n_sizes > 1:
                # weights = df[[" SizeI", " SizeJ", " SizeL"]].values.sum(axis=1).astype(np.float32)
                # weights = weights.min()/weights
                # scores *= weights[..., None]
                scores = scores.mean(axis=0)
            
            nGFlops = np.zeros((len(X),))
            idxs = [si.solIdx for si in solutions]
            nGFlops[idxs] = scores
            return nGFlops
        
        params = benchmarkStep.forkParams
        for i, group in enumerate(benchmarkStep.paramGroups):
            params[f"group_{i}"] = group
        
        cfg = config.update(ductileConfig)

        space = SearchSpace(params,
                            valid=partial(validate_solution, 
                                          benchmarkProcess.problemType, 
                                          benchmarkStep.constantParams, 
                                          asmToolchain.assembler,
                                          debugConfig, 
                                          isaInfoMap),
                            max_iters=cfg["max_iters"])
        selection = Selection.get(**config.populate(cfg, "selection"))
        crossover = Crossover.get(**config.populate(cfg, "crossover"))
        mutation = Mutation(space, **cfg["mutation"])
        mating = Mating(space=space,
                        selection=selection,
                        crossover=crossover,
                        mutation=mutation,
                        max_iters=cfg["max_iters"])
        survival = Survival.get(**config.populate(cfg, "survival"))
        algorithm = GeneticAlgorithm(space,
                                     mating,
                                     evaluate=_evaluate,
                                     survival=survival,
                                     pop_size=cfg["pop_size"],
                                     n_gen=cfg["n_gen"],
                                     tol=cfg["tol"],
                                     period=cfg["period"],
                                     seed=cfg["seed"],
                                     verbose=cfg["verbose"],
                                     log_file=cfg["log_file"])
        best, _ = algorithm.optimize()
        algorithm.evaluate([best])

    return (resultsFileBaseFinal, benchmarkTestFails)

def main(
    ductileConfig,
    config,
    useCache,
    asmToolchain: AssemblyToolchain,
    srcToolchain: SourceToolchain,
    cCompiler: str,
    outputPath: Path,
    buildTmpPath: Path,
    debugConfig: DebugConfig,
    deviceId: int,
    gfxName: str,
    isaInfoMap: Dict[str, IsaInfo]
):
    """Entry point for the "BenchmarkProblems" section of a Tensile config yaml"""
    ClientExecutable.getClientExecutable(str(srcToolchain.compiler.path), cCompiler, outputPath)

    if config is None:
        print(f'No config specified in {globalParameters["ConfigPath"]}, built client only')
        return

    benchmarkDataPath = ensurePath(outputPath / BENCHMARK_DATA_DIR)

    totalTestFails = 0
    for benchmarkProblemTypeConfig in config:
        problemTypeConfig = benchmarkProblemTypeConfig[0]
        if len(benchmarkProblemTypeConfig) < 2:
            problemSizeGroupConfigs = [{}]
        else:
            problemSizeGroupConfigs = benchmarkProblemTypeConfig[1:]

        for idx, sizeGroupConfig in enumerate(problemSizeGroupConfigs):
            print2("ProblemTypeConfig: {}".format(problemTypeConfig))
            problemTypeObj = ProblemType(problemTypeConfig, debugConfig.printIndexAssignmentInfo)

            # using a suffix to check the csv version (for later addFromCSV())
            csvSuffix = "_CSVWinner" if globalParameters["CSVExportWinner"] else ""
            # results files will be named
            newResultsFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.csv" \
                    .format(str(problemTypeObj), idx, csvSuffix) )
            newSolutionsFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.yaml" \
                    .format(str(problemTypeObj), idx, csvSuffix) )
            newGranularityFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.gsp" \
                    .format(str(problemTypeObj), idx, csvSuffix) )

            # skip if possible
            if globalParameters["ForceRedoBenchmarkProblems"] \
                    or not os.path.exists(newResultsFileName):

                # benchmark problem size group
                benchmarkProblemsPath = ensurePath(outputPath / BENCHMARK_PROBLEMS_DIR)
                (resultsFileBaseFinal, benchmarkErrors) = \
                        _benchmarkProblemType(
                            ductileConfig,
                            problemTypeConfig,
                            sizeGroupConfig,
                            idx,
                            useCache,
                            asmToolchain,
                            srcToolchain,
                            cCompiler,
                            buildTmpPath,
                            benchmarkProblemsPath,
                            debugConfig,
                            deviceId,
                            gfxName,
                            isaInfoMap
                        )
                totalTestFails += benchmarkErrors

                print("clientExit={} {} for {}" \
                        .format(totalTestFails, "(ERROR)" if totalTestFails else "(PASS)", \
                        globalParameters["ConfigPath"]) )

                # copy data
                resultsFileBase = resultsFileBaseFinal
                resultsFileName = resultsFileBase + ".csv"
                solutionsFileName = resultsFileBase + ".yaml"
                granularityFileName = resultsFileBase + "_Granularity.csv"
                shutil.copy(resultsFileName, newResultsFileName)
                shutil.copy(solutionsFileName, newSolutionsFileName)
                if os.path.isfile(granularityFileName):
                    shutil.copy(granularityFileName, newGranularityFileName)
            else:
                print1("# {}_{:02d} already benchmarked; skipping." \
                        .format(str(problemTypeObj), idx) )

    if globalParameters["ExitOnFails"] and totalTestFails:
        sys.exit(1)
