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

import functools
import os
import shutil

from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, NamedTuple, List, Optional

from Tensile.Toolchain.Assembly import AssemblyToolchain, buildAssemblyCodeObjectFiles
from Tensile.Toolchain.Source import SourceToolchain, buildSourceCodeObjectFile
from Tensile.Toolchain.Validators import validateToolchain, getVersion, ToolchainDefaults
from Tensile.TensileInstructions import getGfxName, TensileInstructions
from Tensile.Common import globalParameters, HR, print1, print2, printWarning, \
                    assignGlobalParameters, IsaVersion, ParallelMap2
from Tensile.Parallel import ParallelMapConfig
from Tensile.Utilities.RequiredParameters import getRequiredParametersMin
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.SolutionStructs import Solution
from Tensile.Utilities.Profile import profile
from Tensile.SolutionLibrary import MasterSolutionLibrary

from .IO import generateSolutionsAndLibraries, genLazyMasterSolutionLibrary, \
                generateParentLibrary, writeAssembly, writeHelper
from .Logic import logicFileList, schedule
from .ParseArguments import parseArguments

def timing(func):
  def wrapper(*args, **kwargs):
    start = timer()
    res = func(*args, **kwargs)
    end = timer()

    if globalParameters['PrintTiming']:
      print(f'{func.__name__} took {end - start} seconds')

    return res
  return wrapper


def copyStaticFiles(outputPath=None):
  if outputPath is None:
    outputPath = globalParameters["WorkingPath"]
  libraryStaticFiles = [
    "TensileTypes.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
    "hip_f8_impl.h",
    "KernelHeader.h",
    "ReductionTemplate.h",
    "memory_gfx.h" ]

  for fileName in libraryStaticFiles:
    # copy file
    shutil.copy( os.path.join(globalParameters["SourcePath"], fileName), \
        outputPath )

  return libraryStaticFiles

class KernelCodeGenResult(NamedTuple):
    err: int
    src: str
    header: Optional[str]
    name: str
    targetObjFilename: str
    isa: IsaVersion
    wavefrontSize: int


def _processKernelSource(kernelWriterAssembly, ti, kernel) -> KernelCodeGenResult:
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    kernelWriter = kernelWriterAssembly
    kernelWriter.setTensileInstructions(ti)
    asmFilename = kernelWriter.getKernelFileBase(kernel)
    err, src = kernelWriter.getSourceFileString(kernel)
    header = kernelWriter.getHeaderFileString(kernel)
    objFilename = kernel._state.get("codeObjectFile", None)

    return KernelCodeGenResult(err, src, header, asmFilename, objFilename, tuple(kernel["ISA"]), kernel["WavefrontSize"])


@timing
def buildAssemblyKernels(asmPath: Path, asmToolchain: AssemblyToolchain, kernelWriterAssembly: KernelWriterAssembly, ti: TensileInstructions, removeTemporaries: bool, solnLibs: List[tuple]):
  uniqueAsmKernels = [k.getKernels()[0] for k in solnLibs[0] if "BuildKernel" in k]
  pksResults = [_processKernelSource(kernelWriterAssembly, ti, k) for k in uniqueAsmKernels]
  for p, isa, wavefrontsize in set([writeAssembly(asmPath, k) for k in pksResults]):
    asmToolchain.assemble(str(p), str(p.with_suffix(".o")), getGfxName(isa), wavefrontsize)
    if removeTemporaries:
      p.unlink()
  
  return uniqueAsmKernels, solnLibs[1]


def generateKernelHelperObjects(solutions):
  khos = []
  for solution in solutions:
      khos.extend(solution.getHelperKernelObjects())
  return khos #should we deduplicate here?


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
    gfxName: str,
    currMasterLib: MasterSolutionLibrary, 
    prevMasterLib: MasterSolutionLibrary, 
    nextIdx: int, 
) -> None:
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
     for gfxName, lib in libraries:
         masterLib, nextSolutionIdx = updateMasterLibrary(gfxName, lib, masterLib, nextSolutionIdx)
     genLazyMasterSolutionLibrary(libraryPath, libraryFormat, masterLib) # can we make this async
     return solns, libraries


def makeDirectories(outputPath):
    outputPath = Path(outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)
    buildTmp = outputPath.parent / "build_tmp" / str(outputPath.name).upper()
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
  print1("")
  print1(HR)
  print1("# Tensile Create Library")
  print2(HR)
  print2("")

  arguments = parseArguments()
  arguments["OutputPath"] = Path(arguments["OutputPath"]).resolve()
  outputPath, buildTmp, assemblyPath, srcCodeObjectPath, libraryPath = makeDirectories(arguments["OutputPath"])

  cxxCompiler, cCompiler, offloadBundler, rocObjExtract, rocObjLs, assembler, hipconfig = validateToolchain(
      arguments["CxxCompiler"], arguments["CCompiler"], arguments["OffloadBundler"], arguments["RocObjExtract"], arguments["RocObjLs"], arguments["Assembler"], ToolchainDefaults.HIP_CONFIG
  )

  print1(f"# HIP Version:         {getVersion(hipconfig, regex=r'(.+)')}")
  print1(f"# Cxx Compiler:        {cxxCompiler} (version {getVersion(cxxCompiler)})")
  print1(f"# C Compiler:          {cCompiler} (version {getVersion(cCompiler)})")
  print1(f"# Assembler:           {assembler} (version {getVersion(assembler)})")
  print1(f"# Offload Bundler:     {offloadBundler} (version {getVersion(offloadBundler)})")
  print1(f"# Object Extractor:     {rocObjExtract} (version {getVersion(rocObjLs, '-v')})")
  print1(f"# Object Lister:     {rocObjLs} (version {getVersion(rocObjLs, '-v')})")
  print1(f"# Code Object Version: {arguments['CodeObjectVersion']}")
  print1(f"# Architecture(s):     {arguments['Architecture']}")
  print1(f"# Library Format:      {arguments['LibraryFormat']}")

  assignGlobalParameters(arguments, cxxCompiler)

  if ";" in arguments["Architecture"]:
    archs = arguments["Architecture"].split(";") # user arg list format
  else:
    archs = arguments["Architecture"].split("_") # workaround for cmake list in list issue

  # Construct Toolchain
  asmToolchain = AssemblyToolchain(assembler, offloadBundler, globalParameters["BuildIdKind"], arguments["CodeObjectVersion"])
  srcToolchain = SourceToolchain(cxxCompiler, rocObjExtract, rocObjLs, globalParameters["BuildIdKind"], globalParameters["AsanBuild"], globalParameters["SaveTemps"])

  # List logic files
  unsortedLogic = logicFileList(archs, Path(arguments["LogicPath"]), arguments["LogicFilter"], arguments["Experimental"])
  logicFiles = list(filter(lambda x: x != [], schedule(unsortedLogic, 2*arguments["CpuThreads"])))

  # Phase1: Build assembly and master solution libraries
  writerAsm = KernelWriterAssembly(getRequiredParametersMin(), asmToolchain.assembler)
  unaryGenSolutions = functools.partial(generateSolutionsAndLibraries, archs, cxxCompiler)
  unaryProcessMsl = functools.partial(processMsl, libraryPath, arguments["LibraryFormat"])
  unaryBuildAsmKernels = functools.partial(buildAssemblyKernels, assemblyPath, asmToolchain, writerAsm, TensileInstructions(), not arguments["KeepBuildTmp"])
  unaryBuildCOFile = functools.partial(buildAssemblyCodeObjectFiles, asmToolchain, assemblyPath, libraryPath, writerAsm, arguments["UseCompression"])

  def buildCoAndHelpers(input):
     uniqueAsmKernels, libraries = input
     unaryBuildCOFile(uniqueAsmKernels)
     return generateKernelHelperObjects(uniqueAsmKernels), libraries

  compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
  if arguments["LazyLibraryLoading"]:
    assembly = compose(buildCoAndHelpers, unaryBuildAsmKernels, unaryProcessMsl, unaryGenSolutions)
    parMap = functools.partial(ParallelMap2, assembly, ParallelMapConfig(message="Building Lazy Libraries"))
    phase1 = compose(extractBuildResults, parMap)
  else:
    assembly = compose(unaryBuildAsmKernels, unaryGenSolutions)
    parMap = functools.partial(ParallelMap2, assembly, ParallelMapConfig(message="Building Objects"))
    phase1 = compose(buildCoAndHelpers, extractBuildResults, parMap)

  kho, masterLibs = phase1(logicFiles)
  
  # Phase3: Build Master Solution Library
  generateParentLibrary(arguments["LibraryFormat"], libraryPath, masterLibs, arguments["LazyLibraryLoading"])
  del masterLibs

  # Phase2: Write Kernel helpers and build Kernels.co
  copyStaticFiles(outputPath)
  unaryWriteHelpers = functools.partial(writeHelper, outputPath)
  srcFiles = ParallelMap2(unaryWriteHelpers, ParallelMapConfig(message="Generating Kernels code", return_as="list"), list(dict.fromkeys(kho)))
  del kho

  kernelsLib = str(srcCodeObjectPath / "Kernels.so")
  srcToolchain.compile(srcFiles, kernelsLib, str(outputPath), archs)
  del srcFiles

  buildSourceCodeObjectFile(srcToolchain, libraryPath, kernelsLib)


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
  #print1(f"Total kernels: {numKernels}")
  #print1(f"Total kernels processed: {numUniqueKernels}")
  #print1(f"Duplicate kernels removed: {numDuplicateKernels}")
  #print1(f"Kernels processed per second: {(numUniqueKernels/(stop-start)):3.2f}")
  #print1(f"Total solutions processed: {numSoln}")
  #print1(f"Duplicate solutions: {numDuplicateSoln}")
