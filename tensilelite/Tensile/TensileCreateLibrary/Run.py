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

import rocisa

import functools
import os
import shutil
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, NamedTuple, Optional, Union

from Tensile import SOURCE_PATH
from Tensile.Common import (
    DebugConfig,
    DepthUConfig,
    ensurePath,
    HR,
    IsaVersion,
    ParallelMap2,
    ParallelMapConfig,
    print1,
    print2,
    printWarning,
    printWarning,
    setVerbosity,
)
from Tensile.Common.Architectures import gfxToIsa, isaToGfx, SUPPORTED_GFX
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.SolutionStructs.Naming import getKernelFileBase, getKeyNoInternalArgs, getMinNaming, getSerialNaming

from Tensile.CustomYamlLoader import load_logic_gfx_arch
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.SolutionLibrary import MasterSolutionLibrary
from Tensile.SolutionStructs import kernelObjectNames
from Tensile.SolutionStructs import Solution
from Tensile.Toolchain.Assembly import makeAssemblyToolchain, buildAssemblyCodeObjectFiles
from Tensile.Toolchain.Source import makeSourceToolchain, SourceToolchain, buildSourceCodeObjectFile
from Tensile.Toolchain.Validators import (
    ToolchainDefaults,
    validateToolchain,
)
from Tensile.Toolchain.Component import Assembler
from Tensile.Utilities.Decorators.Profile import profile
from Tensile.Utilities.Decorators.Timing import timing
from Tensile.Utilities.RequiredParameters import getRequiredParametersMin

from .IO import generateSolutionsAndLibraries, genLazyMasterSolutionLibrary, \
                generateParentLibrary, writeAssembly, writeHelper
from .Logic import logicFileList, schedule
from .ParseArguments import parseArguments


def copyStaticFiles(outputPath=None):
  if outputPath is None:
    outputPath = globalParameters["WorkingPath"]
  libraryStaticFiles = [
    "TensileTypes.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
    "tensile_float8_bfloat8_bc.h",
    "KernelHeader.h",
    "ReductionTemplate.h",
    "memory_gfx.h"
  ]

  for fileName in libraryStaticFiles:
    # copy file
    shutil.copy( os.path.join(SOURCE_PATH, fileName), outputPath)

  return libraryStaticFiles

class KernelCodeGenResult(NamedTuple):
    err: int
    src: str
    header: Optional[str]
    name: str
    targetObjFilename: str
    isa: IsaVersion
    wavefrontSize: int


def _processKernelSource(kernelWriterAssembly, 
                         data, 
                         useShortNames, 
                         splitGSU, 
                         kernelMinNaming, 
                         kernelSerialNaming, 
                         kernel) -> KernelCodeGenResult:
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    kernelWriter = kernelWriterAssembly
    kernelWriter.setTensileInstructions(data)
    asmFilename = getKernelFileBase(useShortNames, splitGSU, kernelMinNaming, kernelSerialNaming, kernel)
    err, src = kernelWriter.getSourceFileString(kernel, useShortNames)
    header = kernelWriter.getHeaderFileString(kernel)
    objFilename = kernel._state.get("codeObjectFile", None)

    return KernelCodeGenResult(err, src, header, asmFilename, objFilename, tuple(kernel["ISA"]), kernel["WavefrontSize"])

@timing
def buildAssemblyKernels(asmPath: Path,
                         assembler: Assembler,
                         kernelWriterAssembly: KernelWriterAssembly,
                         data,
                         kernelMinNaming,
                         removeTemporaries: bool,
                         solnLibs: List[tuple]):
  kernels = [s.getKernels()[0] for s in solnLibs[0]]
  uniqueAsmKernels = [k for k in kernels if "BuildKernel" in k]
  pksResults = [_processKernelSource(kernelWriterAssembly, data, False, False, kernelMinNaming, kernelMinNaming, k) for k in uniqueAsmKernels]
  for p, isa, wavefrontsize in set([writeAssembly(asmPath, k) for k in pksResults]):
    #Cijk_Ailk_Bljk_BSS_BH_UserArgs_MT128x256x64_MI16Vcp76y7W1sPIjpGVm_aV-u3wID5BM2LfzmbkyHs7rjQ=.s
    assembler(isaToGfx(isa), wavefrontsize, str(p), str(p.with_suffix(".o"))) # TODO: arguments
    #if removeTemporaries:
    #  p.unlink()
  return uniqueAsmKernels, solnLibs[1]


def generateKernelHelperObjects(solutions, isaInfoMap):
  khos = []
  visited = set()
  for solution in solutions:
      build = False
      names = kernelObjectNames(solution)
      for name in names:
         if name not in visited:
            visited.add(name)
            build = True
      if build:
        khos.extend(solution.initHelperKernelObjects(list(isaInfoMap.keys())))
  print(f"khos: {len(khos)}")
  return list(dict.fromkeys(khos))


#def generateMatchTable(masterSolutionLibraries):
#    def libraryIter(lib: MasterSolutionLibrary):
#        if len(lib.solutions):
#            for i, s in enumerate(lib.solutions.items()):
#                yield (i, *s)
#        else:
#            for _, lazyLib in lib.lazyLibraries.items():
#                yield from libraryIter(lazyLib)
#
#    matchTable = {}
#    for library in masterSolutionLibraries:
#      # Match yaml file solutions to solution index
#      for localIdx, _, s in libraryIter(library):
#        matchTable[s.index] = [srcFile, localIdx]
#      LibraryIO.write("MatchTable", matchTable)


def updateParentMasterLibrary(
    gfxName: str,
    masterLib: MasterSolutionLibrary, 
    masterLibraries: Dict[str, MasterSolutionLibrary], 
    nextIdx: Dict[str, int], 
) -> None:
    if gfxName in masterLibraries:
        nextIdx= masterLibraries[gfxName].merge(masterLib, nextIdx)
    else:
        masterLibraries[gfxName] = masterLib


def updateMasterLibrary(
    currMasterLib: MasterSolutionLibrary, 
    prevMasterLib: MasterSolutionLibrary, 
    nextIdx: int, 
) -> None:
    assert prevMasterLib is not None or currMasterLib is not None
    if prevMasterLib is not None:
        nextIdx = prevMasterLib.merge(currMasterLib, nextIdx)
    else:
        prevMasterLib = currMasterLib
        nextIdx = 0
    return prevMasterLib, nextIdx


def processMsl(libraryPath, libraryFormat, solnLibTup):
    solns, libraries = solnLibTup
    if len(libraries) > 0:
        masterLib = None
        nextSolutionIdx = 0
        for _, lib in libraries:
            masterLib, nextSolutionIdx = updateMasterLibrary(lib, masterLib, nextSolutionIdx)

        genLazyMasterSolutionLibrary(libraryPath, libraryFormat, masterLib) # can we make this async
    return solns, libraries


def makeDirectories(outputPath):
    outputPath = Path(outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)
    buildTmp = outputPath / "build_tmp" / str(outputPath.name).upper()
    srcCodeObjectPath = buildTmp / "code_object_tmp"
    srcCodeObjectPath.mkdir(parents=True, exist_ok=True)
    asmPath = buildTmp / "assembly"
    asmPath.mkdir(parents=True, exist_ok=True)
    kernelsIncludePath = outputPath / "Kernels"
    kernelsIncludePath.mkdir(parents=True, exist_ok=True)
    libraryPath = outputPath / "library"
    libraryPath.mkdir(exist_ok=True)
    return outputPath, buildTmp, asmPath, srcCodeObjectPath, libraryPath


def extractBuildResults(result):
    print1(type(result))
    masterLibs = {}
    nextIdx = 0
    flattenedList = []
    for l, library in result:
        if len(library) > 0:
          for gfxName, lib in library:
            updateParentMasterLibrary(gfxName, lib, masterLibs, nextIdx) # I think we can move this out and do this later
        flattenedList.extend(l)
    return flattenedList, masterLibs


################################################################################
# Tensile Create Library
################################################################################
@profile
def run():
  start = timer()
  print1("")
  print1(HR)
  print1("# Tensile Create Library")
  print2(HR)
  print2("")

  arguments = parseArguments()
  arguments["OutputPath"] = Path(arguments["OutputPath"]).resolve()
  outputPath, buildTmp, assemblyPath, srcCodeObjectPath, libraryPath = makeDirectories(arguments["OutputPath"])

  setVerbosity(arguments["PrintLevel"])
  outputPath = Path(ensurePath(os.path.abspath(arguments["OutputPath"])))
  cxxCompiler, offloadBundler, ls, extract = validateToolchain(
      arguments["CxxCompiler"],
      arguments["OffloadBundler"],
      arguments["RocObjLs"],
      arguments["RocObjExtract"]
  )

  if ";" in arguments["Architecture"]:
      archs = arguments["Architecture"].split(";")
  else:
      archs = arguments["Architecture"].split("_")
  archs = SUPPORTED_GFX if "all" in archs else archs

  targetIsas = [gfxToIsa(a) for a in archs]
  isaInfoMap = makeIsaInfoMap(targetIsas, cxxCompiler)
  assignGlobalParameters(arguments, isaInfoMap)

  asmToolchain = makeAssemblyToolchain(
      cxxCompiler,
      offloadBundler,
      arguments["CodeObjectVersion"],
      arguments["BuildIdKind"]
  )
  srcToolchain = makeSourceToolchain(
      cxxCompiler,
      offloadBundler,
      ls,
      extract,
      arguments["AsanBuild"],
      arguments["BuildIdKind"],
      arguments["CpuThreads"],
      save_temps=False
  )

  print1(asmToolchain.assembler)
  print1(asmToolchain.bundler)

  # List logic files
  unsortedLogic = logicFileList(archs, Path(arguments["LogicPath"]), arguments["LogicFilter"], arguments["Experimental"])
  logicFiles = list(filter(lambda x: x != [], schedule(unsortedLogic, 2*arguments["CpuThreads"], arguments["CpuThreads"])))

  # Phase1: Build assembly and master solution libraries
  writerAsm = KernelWriterAssembly(
      getRequiredParametersMin(), 
      getRequiredParametersMin(),
      asmToolchain.assembler,
      DebugConfig(),
  )

  unaryGenSolutions = functools.partial(generateSolutionsAndLibraries, 
                                        asmToolchain.assembler, 
                                        isaInfoMap, 
                                        arguments["LazyLibraryLoading"])

  #result = ParallelMap2(unaryGenSolutions, ParallelMapConfig(message="blah"), logicFiles)
  #d = {}
  #l_path = Path(arguments["LogicPath"])
  #for s, _, l in result:
  #  logic_file = Path(l).relative_to(l_path)
  #  d[str(logic_file)] = s[0]["codeObjectFile"]
  #import pprint
  #pprint.pprint(d, indent=2)
  #printExit("done")
  
  unaryProcessMsl = functools.partial(processMsl, libraryPath, arguments["LibraryFormat"])
  unaryBuildAsmKernels = functools.partial(buildAssemblyKernels, 
                                           assemblyPath, 
                                           asmToolchain.assembler, 
                                           writerAsm, 
                                           rocisa.rocIsa.getInstance().getData(), 
                                           getRequiredParametersMin(), 
                                           not arguments["KeepBuildTmp"])
  unaryBuildCOFile = functools.partial(buildAssemblyCodeObjectFiles, 
                                       asmToolchain.linker, 
                                       asmToolchain.bundler,
                                       globalParameters["ROCmLdPath"],
                                       libraryPath, 
                                       assemblyPath, 
                                       arguments["UseCompression"])
  def buildCoAndHelpers(input):
     uniqueAsmKernels, libraries = input
     unaryBuildCOFile(uniqueAsmKernels)
     return uniqueAsmKernels, libraries

  compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
  if arguments["LazyLibraryLoading"]:
    assembly = compose(buildCoAndHelpers, unaryBuildAsmKernels, unaryProcessMsl, unaryGenSolutions)
    parMap = functools.partial(ParallelMap2, assembly, ParallelMapConfig(message="Building Lazy Libraries"))
    phase1 = compose(extractBuildResults, parMap)
  else:
    assembly = compose(unaryBuildAsmKernels, unaryGenSolutions)
    parMap = functools.partial(ParallelMap2, assembly, ParallelMapConfig(message="Building Objects"))
    phase1 = compose(buildCoAndHelpers, extractBuildResults, parMap)

  kernels, masterLibs = phase1(logicFiles)

  # Phase2: Build Master Solution Library
  generateParentLibrary(arguments["LibraryFormat"], libraryPath, masterLibs, arguments["LazyLibraryLoading"])
  del masterLibs

  # Phase3: Write Kernel helpers and build Kernels.co
  copyStaticFiles(outputPath)
  unaryWriteHelpers = functools.partial(writeHelper, outputPath)
  srcFiles = ParallelMap2(unaryWriteHelpers, 
                          ParallelMapConfig(message="Generating Kernels code", return_as="list"), 
                          generateKernelHelperObjects(kernels, isaInfoMap))
  kernelsLib = str(srcCodeObjectPath / "Kernels.so")
  #srcToolchain.compiler(srcFiles, kernelsLib, str(outputPath), archs)
  #buildSourceCodeObjectFile(srcToolchain, libraryPath, kernelsLib)

  if not arguments["KeepBuildTmp"]:
    if buildTmp.exists() and buildTmp.is_dir():
      shutil.rmtree(buildTmp)
    else:
      printWarning(f"Cannot remove build_tmp")

  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

  stop = timer()
  print1(f"Total time (s): {(stop-start):3.2f}")
  numKernels = len(kernels)
  print1(f"Total kernels: {numKernels}")
  #print1(f"Total kernels processed: {numUniqueKernels}")
  #print1(f"Duplicate kernels removed: {numDuplicateKernels}")
  print1(f"Kernels processed per second: {(numKernels/(stop-start)):3.2f}")
  #print1(f"Total solutions processed: {numSoln}")
  #print1(f"Duplicate solutions: {numDuplicateSoln}")





    # start = timer()
    # print1("")
    # print1(HR)
    # print1("# Tensile Create Library")
    # print2(HR)
    # print2("")

    # arguments = parseArguments()
    # setVerbosity(arguments["PrintLevel"])
    # outputPath = Path(ensurePath(os.path.abspath(arguments["OutputPath"])))
    # cxxCompiler, _, offloadBundler, _, _ = validateToolchain(
    #     arguments["CxxCompiler"],
    #     arguments["CCompiler"],
    #     arguments["OffloadBundler"],
    #     arguments["Assembler"],
    #     ToolchainDefaults.HIP_CONFIG,
    # )

    # if ";" in arguments["Architecture"]:
    #     archs = arguments["Architecture"].split(";")
    # else:
    #     archs = arguments["Architecture"].split("_")
    # archs = SUPPORTED_GFX if "all" in archs else archs

    # targetIsas = [gfxToIsa(a) for a in archs]
    # isaInfoMap = makeIsaInfoMap(targetIsas, cxxCompiler)
    # assignGlobalParameters(arguments, isaInfoMap)

    # asmToolchain = makeAssemblyToolchain(
    #     cxxCompiler,
    #     offloadBundler,
    #     arguments["CodeObjectVersion"],
    #     arguments["BuildIdKind"]
    # )
    # srcToolchain = makeSourceToolchain(
    #     cxxCompiler,
    #     offloadBundler,
    #     arguments["AsanBuild"],
    #     arguments["BuildIdKind"],
    #     save_temps=False
    # )

    # print1(asmToolchain.assembler)
    # print1(asmToolchain.bundler)

    # if not os.path.exists(arguments["LogicPath"]):
    #     printExit(f"LogicPath {arguments['LogicPath']} doesn't exist")

    # logicExtFormat = ".yaml"
    # if arguments["LogicFormat"] == "yaml":
    #     pass
    # elif arguments["LogicFormat"] == "json":
    #     logicExtFormat = ".json"
    # else:
    #     printExit(f"Unrecognized LogicFormat: {arguments['LogicFormat']}")

    # def archMatch(arch: str, archs: List[str]):
    #     return (arch in archs) or any(a.startswith(arch) for a in archs)

    # def validLogicFile(p: Path):
    #     return p.suffix == logicExtFormat and (
    #         "all" in archs or archMatch(load_logic_gfx_arch(p), archs)
    #     )

    # globPattern = os.path.join(
    #     arguments["LogicPath"], f"**/{arguments['LogicFilter']}{logicExtFormat}"
    # )
    # print1(f"# LogicFilter:       {globPattern}")
    # logicFiles = (
    #     os.path.join(arguments["LogicPath"], file)
    #     for file in glob.iglob(globPattern, recursive=True)
    # )
    # logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]

    # print1(f"# Experimental:      {arguments['Experimental']}")
    # if not arguments["Experimental"]:
    #     logicFiles = [
    #         file for file in logicFiles if "experimental" not in map(str.lower, Path(file).parts)
    #     ]

    # print2(f"# LibraryLogicFiles: {len(logicFiles)}")

    # for logicFile in logicFiles:
    #     print2("#   %s" % logicFile)

    # solutions, masterLibraries = generateLogicDataAndSolutions(
    #     logicFiles, arguments, asmToolchain.assembler, isaInfoMap
    # )

    # kernels, kernelHelperObjs, _ = generateKernelObjectsFromSolutions(solutions)
    # kernelSerialNaming = getSerialNaming(kernels)
    # kernelMinNaming = getMinNaming(kernels)
    # kernelWriterAssembly = KernelWriterAssembly(
    #     kernelMinNaming,
    #     kernelSerialNaming,
    #     asmToolchain.assembler,
    #     DebugConfig(),
    # )

    # copyStaticFiles(outputPath)

    # numKernels = writeSolutionsAndKernelsTCL(
    #     outputPath,
    #     asmToolchain,
    #     srcToolchain,
    #     kernels,
    #     kernelHelperObjs,
    #     kernelWriterAssembly,
    #     archs,
    #     kernelSerialNaming,
    #     kernelMinNaming,
    #     useShortNames=arguments["ShortNames"],
    #     compress=arguments["UseCompression"],
    # )

    # archs = [ # is this really different than the other archs above?
    #     isaToGfx(arch)
    #     for arch in targetIsas
    #     if isaInfoMap[arch].asmCaps["SupportedISA"]
    # ]
    # newLibraryDir = ensurePath(os.path.join(outputPath, "library"))
    # splitGSU = False
    # for archName, newMasterLibrary in masterLibraries.items():
    #     if archName in archs:
    #         if arguments["LazyLibraryLoading"]:
    #             masterFile = os.path.join(newLibraryDir, "TensileLibrary_lazy_" + archName)
    #         else:
    #             masterFile = os.path.join(newLibraryDir, "TensileLibrary_" + archName)
    #         newMasterLibrary.applyNaming(splitGSU, kernelMinNaming)
    #         LibraryIO.write(masterFile, state(newMasterLibrary), arguments["LibraryFormat"])
    #         for name, lib in newMasterLibrary.lazyLibraries.items():
    #             filename = os.path.join(newLibraryDir, name)
    #             lib.applyNaming(splitGSU, kernelMinNaming)
    #             LibraryIO.write(filename, state(lib), arguments["LibraryFormat"])

    # if not arguments["KeepBuildTmp"]:
    #     buildTmp = Path(arguments["OutputPath"]).parent / "library" / "build_tmp"
    #     if buildTmp.exists() and buildTmp.is_dir():
    #         shutil.rmtree(buildTmp)
    #     buildTmp = Path(arguments["OutputPath"]) / "build_tmp"
    #     if buildTmp.exists() and buildTmp.is_dir():
    #         shutil.rmtree(buildTmp)
    #     else:
    #         printWarning(f"Cannot remove build_tmp")

    # print1("# Tensile Library Writer DONE")
    # print1(HR)
    # print1("")

    # stop = timer()

    # print1(f"Total time (s): {(stop-start):3.2f}")
    # print1(f"Total kernels processed: {numKernels}")
    # print1(f"Kernels processed per second: {(numKernels/(stop-start)):3.2f}")
