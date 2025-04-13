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
import shutil
from pathlib import Path
from os import path, getpid
from timeit import default_timer as timer
from typing import Dict, List, NamedTuple, Optional

from Tensile import SOURCE_PATH
from Tensile.Common import DebugConfig, HR, IsaVersion, ParallelMap2, ParallelMapConfig, setVerbosity
from Tensile.Common.Architectures import gfxToIsa, isaToGfx, SUPPORTED_GFX
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.SolutionStructs.Naming import getKernelFileBase, getKernelNameMin
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.SolutionLibrary import MasterSolutionLibrary
from Tensile.SolutionStructs import kernelObjectNames
from Tensile.Toolchain.Assembly import makeAssemblyToolchain, buildAssemblyCodeObjectFiles
from Tensile.Toolchain.Source import makeSourceToolchain, SourceToolchain, buildSourceCodeObjectFile
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.Toolchain.Component import Assembler
from Tensile.Utilities.Decorators.Profile import profile
from Tensile.Utilities.Decorators.Timing import timing

from .IO import generateSolutionsAndLibraries, genLazyMasterSolutionLibrary, \
                generateParentLibrary, writeAssembly, writeHelper
from .Logic import logicFileList, schedule, getCoFileNames
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
    shutil.copy( path.join(SOURCE_PATH, fileName), outputPath)

  return libraryStaticFiles

class KernelCodeGenResult(NamedTuple):
    err: int
    src: str
    header: Optional[str]
    name: str
    targetObjFilename: str
    isa: IsaVersion
    wavefrontSize: int
    cuoccupancy: int
    pgr: int
    mathclk: int


def _processKernelSource(kernelWriterAssembly, 
                         data, 
                         useShortNames, 
                         splitGSU, 
                         kernelSerialNaming, 
                         kernel) -> KernelCodeGenResult:
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    kernelWriter = kernelWriterAssembly
    kernelWriter.setTensileInstructions(data)
    asmFilename = getKernelFileBase(useShortNames, splitGSU, kernelSerialNaming, kernel)
    err, src = kernelWriter.getSourceFileString(kernel, useShortNames)
    header = kernelWriter.getHeaderFileString(kernel)
    objFilename = kernel._state.get("codeObjectFile", None)
    return KernelCodeGenResult(err,
                               src,
                               header,
                               asmFilename,
                               objFilename,
                               tuple(kernel["ISA"]),
                               kernel["WavefrontSize"],
                               kernel["CUOccupancy"],
                               int(kernel["PrefetchGlobalRead"]),
                               kernel["MathClocksUnrolledLoop"])

@timing
def buildAssemblyKernels(asmPath: Path,
                         assembler: Assembler,
                         kernelWriterAssembly: KernelWriterAssembly,
                         data,
                         removeTemporaries: bool,
                         solnLibs: List[tuple]):
  kernels = [s.getKernels()[0] for s in solnLibs[0]]

  visited = set()
  duplicates = 0
  for k in kernels:
    base = getKernelNameMin(k, False)
    k.duplicate = True if base in visited else False
    duplicates += k.duplicate
    visited.add(base)
  uniqueAsmKernels = [k for k in kernels if not k.duplicate]
  #uniqueAsmKernels = [k for k in kernels if "BuildKernels" in k]

  pksResults = [_processKernelSource(kernelWriterAssembly, data, False, False, None, k) for k in uniqueAsmKernels]
  asmPidPath = asmPath / str(getpid())
  asmPidPath.mkdir(exist_ok=True)
  for p, isa, wavefrontsize in set([writeAssembly(asmPidPath, k) for k in pksResults]):
    assembler(isaToGfx(isa), wavefrontsize, str(p), str(p.with_suffix(".o"))) # TODO: arguments
    if removeTemporaries:
      p.unlink()
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
  print("")
  print(HR)
  print("# Tensile Create Library")

  arguments = parseArguments()
  arguments["OutputPath"] = Path(arguments["OutputPath"]).resolve()
  outputPath, buildTmp, assemblyPath, srcCodeObjectPath, libraryPath = makeDirectories(arguments["OutputPath"])

  setVerbosity(arguments["PrintLevel"])
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
      arguments["CpuThreads"],
      arguments["AsanBuild"],
      arguments["BuildIdKind"],
      save_temps=False
  )

  print(asmToolchain.assembler)
  print(asmToolchain.bundler)

  # List logic files
  unsortedLogic = logicFileList(archs,
                                Path(arguments["LogicPath"]),
                                arguments["LogicFilter"],
                                arguments["Experimental"])
  cofiles = ParallelMap2(getCoFileNames,
                         ParallelMapConfig(message="Scheduling work. ",
                                           procs=arguments["CpuThreads"]),
                         unsortedLogic)
  logicFiles = schedule(cofiles, 2*arguments["CpuThreads"], arguments["CpuThreads"])

  # Phase1: Build assembly and master solution libraries
  writerAsm = KernelWriterAssembly(None, asmToolchain.assembler, DebugConfig())
  unaryGenSolutions = functools.partial(generateSolutionsAndLibraries, 
                                        asmToolchain.assembler, 
                                        isaInfoMap, 
                                        arguments["LazyLibraryLoading"])

  #result = ParallelMap2(unaryGenSolutions, ParallelMapConfig(message="blah"), logicFiles)
  #d = {}
  #for ss, _ in result:
  #  for s in ss:
  #    name = getKernelNameMin(s, False)
  #    if name in d:
  #       d[name].add(s["codeObjectFile"])
  #    else:
  #       d[name] = set()
  #       d[name].add(s["codeObjectFile"])
  #import pprint
  #pprint.pprint(d, indent=2)
  #count = 0
  #for k, v in d.items():
  #   if len(v) > 1:
  #      print(k,v)
  #      count += 1
  #print(count)
  #assert False

  unaryProcessMsl = functools.partial(processMsl, libraryPath, arguments["LibraryFormat"])
  unaryBuildAsmKernels = functools.partial(buildAssemblyKernels, 
                                           assemblyPath, 
                                           asmToolchain.assembler, 
                                           writerAsm, 
                                           rocisa.rocIsa.getInstance().getData(), 
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
  srcToolchain.compiler(srcFiles, kernelsLib, str(outputPath), archs)
  #buildSourceCodeObjectFile(srcToolchain, libraryPath, kernelsLib)

  if not arguments["KeepBuildTmp"]:
    if buildTmp.exists() and buildTmp.is_dir():
      shutil.rmtree(buildTmp)
    else:
      print(f"Warning: Cannot remove build_tmp")

  print("# Tensile Library Writer DONE")
  print(HR)
  print("")

  stop = timer()
  print(f"Total time (s): {(stop-start):3.2f}")
  numKernels = len(kernels)
  print(f"Total kernels: {numKernels}")
  #print(f"Total kernels processed: {numUniqueKernels}")
  #print(f"Duplicate kernels removed: {numDuplicateKernels}")
  print(f"Kernels processed per second: {(numKernels/(stop-start)):3.2f}")
  #print(f"Total solutions processed: {numSoln}")
  #print(f"Duplicate solutions: {numDuplicateSoln}")
