################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
import glob
import itertools
import os
import shutil
import subprocess

from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, NamedTuple, List, Optional, Union

from Tensile import Utils
from Tensile.Toolchain.Assembly import AssemblyToolchain, buildAssemblyCodeObjectFiles
from Tensile.Toolchain.Source import SourceToolchain, buildSourceCodeObjectFile
from Tensile.Toolchain.Validators import validateToolchain, getVersion, ToolchainDefaults
from Tensile.TensileInstructions import getGfxName, TensileInstructions
from Tensile.Common import globalParameters, HR, print1, print2, printExit, ensurePath, \
                    CHeader, assignGlobalParameters, architectureMap, IsaVersion, pushWorkingPath, \
                    popWorkingPath, ParallelMap2
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.KernelWriterBase import KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H
from Tensile import LibraryIO
from Tensile.SolutionStructs import Solution
from Tensile.CustomYamlLoader import load_logic_gfx_arch, load_yaml_sequence_item
from Tensile.Utilities.Profile import profile
from Tensile.Utilities.RequiredParameters import getRequiredParametersMin
from Tensile.SolutionLibrary import MasterSolutionLibrary

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


def processKernelSource(kernelWriterAssembly, ti, kernel) -> KernelCodeGenResult:
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


def removeInvalidSolutionsAndKernels(results, kernels, solutions, errorTolerant, globalParameters):
    removeKernels = []
    removeKernelNames = []
    removeSolutions = []
    removeResults = []

    for kernIdx, r in Utils.tqdm(enumerate(results)) if globalParameters["PrintLevel"] > 1 else enumerate(results):
        if r.err != 0:
            if not errorTolerant:
                print("\nKernel generation failed for kernel: {}".format(kernels[kernIdx]["SolutionIndex"]))
                print(kernels[kernIdx]["SolutionNameMin"])
            removeKernels.append(kernels[kernIdx])
            kName = Solution.getKeyNoInternalArgs(kernels[kernIdx])
            if kName not in removeKernelNames:
                removeKernelNames.append(kName)
            removeResults.append(results[kernIdx])

    if len(removeKernels) > 0 and not errorTolerant:
        printExit("** kernel generation failure **")

    for kern in removeKernels:
        kernels.remove(kern)

    for solution in Utils.tqdm(solutions, "Finding invalid solutions") if globalParameters["PrintLevel"] > 1 else solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = Solution.getKeyNoInternalArgs(kernel)
            if kName in removeKernelNames:
                removeSolutions.append(solution)
                break

    for solut in removeSolutions:
        solutions.remove(solut)

    for rel in removeResults:
        results.remove(rel)


def writeAssembly(asmPath: Union[Path, str], result: KernelCodeGenResult):
    if result.err:
      printExit(f"Failed to build kernel {result.name} because it has error code {result.err}")
    path = Path(asmPath) / str(os.getpid()) 
    path.mkdir(exist_ok=True)
    filepath = path / f"{result.name}.s"
    isa =  result.isa
    wfsize = result.wavefrontSize
    with open(filepath, "w", encoding="utf-8") as f:
      f.write(result.src)
      del result # result.src is very large so let gc know to clean up asap
 
    return filepath, isa, wfsize


def writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H):
    kernelSourceFilename = os.path.join(os.path.normcase(outputPath), KERNEL_HELPER_FILENAME_CPP)
    kernelHeaderFilename = os.path.join(os.path.normcase(outputPath), KERNEL_HELPER_FILENAME_H)

    with open(kernelHeaderFilename, "w", encoding="utf-8") as kernelHeaderFile, \
          open(kernelSourceFilename, "w", encoding="utf-8") as kernelSourceFile:  
        kernelSourceFile.write(CHeader)
        kernelHeaderFile.write(CHeader)
        kernelSourceFile.write("#include \"Kernels.h\"\n")
        kernelHeaderFile.write("#pragma once\n")
        if globalParameters["RuntimeLanguage"] == "HIP":
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


def writeHelper(outputPath, kernelHelperObj) -> str:
    name = kernelHelperObj.getKernelName()
    KERNEL_HELPER_FILENAME_CPP = name + ".cpp"
    KERNEL_HELPER_FILENAME_H = name + ".h"
    kernelSourceFilename = os.path.join(os.path.normcase(outputPath), "Kernels", KERNEL_HELPER_FILENAME_CPP)
    kernelHeaderFilename = os.path.join(os.path.normcase(outputPath), "Kernels", KERNEL_HELPER_FILENAME_H)

    with open(kernelHeaderFilename, "w", encoding="utf-8") as kernelHeaderFile, \
          open(kernelSourceFilename, "w", encoding="utf-8") as kernelSourceFile:
        kernelSourceFile.write(CHeader)
        kernelHeaderFile.write(CHeader)
        kernelSourceFile.write("#include \"{}.h\"\n".format(name))
        kernelHeaderFile.write("#pragma once\n")
        if globalParameters["RuntimeLanguage"] == "HIP":
          kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
          kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
        kernelHeaderFile.write("#include \"KernelHeader.h\"\n\n")
        if "Enum" not in name:
            kernelHeaderFile.write("#include \"Kernels/TensileActivationEnum_S.h\"\n")
            kernelHeaderFile.write("#include \"Kernels/TensileActivationEnum_I.h\"\n")
            kernelHeaderFile.write("#include \"Kernels/TensileGradientActivationEnum_S.h\"\n")
        if "TensileActivation_S" not in name and "TensileActivation_I" not in name and "Enum" not in name:
            kernelHeaderFile.write("#include \"Kernels/TensileActivation_S_Hipblaslt_all.h\"\n")
            kernelHeaderFile.write("#include \"Kernels/TensileActivation_I_Hipblaslt_all.h\"\n")
        if "TensileGradientActivation_S" not in name and "Enum" not in name:
            kernelHeaderFile.write("#include \"Kernels/TensileGradientActivation_S_Hipblaslt_all.h\"\n")

        HeaderText = ""
        (err, src) = kernelHelperObj.getSourceFileString()
        kernelSourceFile.write(src)
        if err:
            print("*** warning: invalid kernel#%u" % name)
        HeaderText += kernelHelperObj.getHeaderFileString()
        kernelHeaderFile.write(HeaderText)

    return kernelSourceFilename


def getKernelWriterAssembly(solutions, assembler):
  kernelSerialNaming   = Solution.getSerialNaming(solutions)
  kernelMinNaming      = Solution.getMinNaming(solutions)
  kernelWriterAssembly = KernelWriterAssembly(kernelMinNaming, kernelSerialNaming, assembler)
  return kernelWriterAssembly


def writeSolutionsAndKernels(outputPath, asmToolchain, srcToolchain, solutions, kernels, kernelHelperObjs, \
    kernelWriterAssembly, errorTolerant=False, generateSourcesAndExit=False, compress=True):
  codeObjectFiles = []

  pushWorkingPath('build_tmp')
  pushWorkingPath(os.path.basename(outputPath).upper())
  asmPath = ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))

  asmKernels = [k for k in kernels if k['KernelLanguage'] == 'Assembly']

  visited = set()
  duplicates = 0
  for k in asmKernels:
    base = kernelWriterAssembly.getKernelFileBase(k)
    k.duplicate = True if base in visited else False
    duplicates += k.duplicate
    print2(f"Duplicate: {base}")
    visited.add(base)
  print1(f"Number of duplicates: {duplicates}")

  numAsmKernels = len(asmKernels)
  numKernels = len(asmKernels)
  assert numKernels == numAsmKernels, "Only assembly kernels are supported in TensileLite"
  asmIter   = zip(itertools.repeat(kernelWriterAssembly), itertools.repeat(TensileInstructions()), asmKernels)
  asmResults = ParallelMap2(processKernelSource, asmIter, "Generating assembly kernels")
  removeInvalidSolutionsAndKernels(asmResults, asmKernels, solutions, errorTolerant, globalParameters)
  def assemble(ret):
    p, isa, wavefrontsize = ret
    asmToolchain.assemble(str(p), str(p.with_suffix(".o")), getGfxName(isa), wavefrontsize)
  unaryWriteAssembly = functools.partial(writeAssembly, asmPath)
  compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
  ret = ParallelMap2(compose(assemble, unaryWriteAssembly), asmResults, "Writing assembly kernels", return_as="list", multiArg=False)

  writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
  srcKernelFile = Path(outputPath) / "Kernels.cpp"
  
  if not generateSourcesAndExit:
      codeObjectFiles += buildAssemblyCodeObjectFiles(asmToolchain, asmKernels, kernelWriterAssembly, outputPath, compress)
      buildSourceCodeObjectFile(srcToolchain, outputPath, srcKernelFile)

  popWorkingPath() # build_tmp
  popWorkingPath() # workingDir

  return codeObjectFiles, numKernels

@timing
def writeSolutionsAndKernelsTCL(outputPath, asmToolchain, srcToolchain, solutions, assembler, compress=True):
  pushWorkingPath('build_tmp')
  pushWorkingPath(os.path.basename(outputPath).upper())
  asmPath = ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))

  asmKernels = [k.getKernels()[0] for k in solutions if k['KernelLanguage'] == 'Assembly']
  kernelHelperObjs = generateKernelHelperObjects(asmKernels)
  kernelWriterAssembly = getKernelWriterAssembly(asmKernels, assembler)

  uniqueAsmKernels = [k for k in asmKernels if "BuildKernel" in k]
  unique = set()
  for k in uniqueAsmKernels:
     name = kernelWriterAssembly.getKernelFileBase(k)
     if name not in unique:
        unique.add(name)
     else:
        print1(f"{k['SolutionIndex']} {k['LogicFileName']}")

  pksResults = [processKernelSource(kernelWriterAssembly, TensileInstructions(), k) for k in uniqueAsmKernels]
  for p, isa, wavefrontsize in [writeAssembly(asmPath, k) for k in pksResults]:
    asmToolchain.assemble(str(p), str(p.with_suffix(".o")), getGfxName(isa), wavefrontsize)
  
  buildAssemblyCodeObjectFiles(asmToolchain, uniqueAsmKernels, kernelWriterAssembly, outputPath, compress)

  popWorkingPath() # build_tmp
  popWorkingPath() # workingDir

  return kernelHelperObjs, len(uniqueAsmKernels), len(asmKernels)


def generateKernelHelperObjects(solutions):
  khos = []
  for solution in solutions:
      khos.extend(solution.getHelperKernelObjects())
  return list(dict.fromkeys(khos))


@timing
def generateSolutions(args, cxxCompiler, logicFiles):
    if ";" in args["Architecture"]:
        archs = args["Architecture"].split(";") # user arg list format
    else:
        archs = args["Architecture"].split("_") # workaround for cmake list in list issue
    solutions = []
    libraries = []
    for logicFileGroup in logicFiles:
        for logicFile in logicFileGroup[1]:
            libraryLogic = LibraryIO.parseLibraryLogicFile(logicFile, cxxCompiler, archs)
            solutions.extend(libraryLogic.solutions)
            libraries.append((libraryLogic.architecture, libraryLogic.library))

    numSoln = len(solutions)
    return solutions, libraries, numSoln, (numSoln-len(solutions))


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


def getArchitectures(arguments):
    if ";" in arguments["Architecture"]:
        archs = arguments["Architecture"].split(";")
    else:
        archs = arguments["Architecture"].split("_")
      
    logicArchs = set()
    for arch in archs:
        if arch in architectureMap:
            logicArchs.add(architectureMap[arch])
        else:
            printExit("Architecture %s not supported" % arch)
    return archs, logicArchs


def getLogicFileList(arguments):
    archs, _ = getArchitectures(arguments)
    
    def archMatch(arch: str, archs: List[str]):
        return (arch in archs) or any(a.startswith(arch) for a in archs)
    def validLogicFile(p: Path):
        return p.suffix == ".yaml" and ("all" in archs or archMatch(load_logic_gfx_arch(p), archs))
  
    if not os.path.exists(arguments["LogicPath"]):
        printExit(f"LogicPath {arguments['LogicPath']} doesn't exist")

    globPattern = os.path.join(arguments["LogicPath"], f"**/{arguments['LogicFilter']}.yaml")
    print1(f"# LogicFilter:         {globPattern}")
    logicFiles = (os.path.join(arguments["LogicPath"], file) for file in glob.iglob(globPattern, recursive=True))
    print1(f"# Experimental:        {arguments['Experimental']}")

    if not arguments["Experimental"]:
        logicFiles = [file for file in logicFiles if "experimental" not in map(str.lower, Path(file).parts)]
    
    logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]
    
    print2(f"# LibraryLogicFiles: {len(logicFiles)}")
    for logicFile in logicFiles:
        print2("#   %s" % logicFile)

    return logicFiles


def numberOfBuildKernerls(logicFile):
    from operator import itemgetter
    result = subprocess.run(['/bin/grep', "BuildKernel", logicFile], stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=False)
    return int(str(result.stdout).count("BuildKernel"))


def distribute(lst, n):
    import heapq
    list_of_lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    for value, f in lst:
        total, index = heapq.heappop(totals)
        list_of_lists[index].append((value, f))
        heapq.heappush(totals, (total + value, index))
    return sorted(list_of_lists, key=lambda x: sum(first for first, _ in x), reverse=True)


def schedule(logicFiles: list, numberOfTasks: int):
    from yaml import Loader
    problemMap = {}
    for logicFile in logicFiles:
        codeObjectFile = load_yaml_sequence_item(logicFile, Loader, 0)
        codeObjectFile = codeObjectFile["codeObjectFile"]
        if codeObjectFile in problemMap:
            problemMap[codeObjectFile].append(logicFile)
        else:
            problemMap[codeObjectFile] = [logicFile]

    result = []
    for codeObjectFile, logicFiles in problemMap.items():
        count = sum(numberOfBuildKernerls(logicFile) for logicFile in logicFiles)
        result.append((count, logicFiles))

    return distribute(result, numberOfTasks) # need to convert list of list of tuples to list of list of strings


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


def genLazyMasterSolutionLibrary(libraryPath, libraryFormat, libraries):
    _masterLib = None
    _nextSolutionIdx = 0
    if len(libraries) > 0:
        for gfxName, lib in libraries:
            _masterLib, _nextSolutionIdx = updateMasterLibrary(gfxName, lib, _masterLib, _nextSolutionIdx)
        # Can we do this asynchronously before the call to writeSolutionsAndKernels?
        newLibraryDir = libraryPath
        for name, lib in list(_masterLib.lazyLibraries.items()):
            catalogPath = newLibraryDir / name
            lib.applyNaming(getRequiredParametersMin())  # <-- This should be able to be replaced directly with `name`?
            LibraryIO.write(str(catalogPath), Utils.state(lib), libraryFormat)


@profile
def build(arguments, cxxCompiler, assembler, asmToolchain, srcToolchain, logicFiles):

    start = timer()
    libraryPath = Path(arguments["OutputPath"]) / "library"
    solutions, libraries, totalSoln, dupSoln = generateSolutions(arguments, cxxCompiler, logicFiles)
    genLazyMasterSolutionLibrary(libraryPath, arguments["LibraryFormat"], libraries)
    khos, numUniqueKernels, numKernels = writeSolutionsAndKernelsTCL(arguments["OutputPath"], asmToolchain, srcToolchain, solutions, 
                                                                     assembler, compress=arguments["UseCompression"])
    stop = timer()
    print2(f"Total time (s): {(stop-start):3.2f}")
    print2(f"Kernels per second: {(numUniqueKernels/(stop-start)):3.2f}")
    print2(f" {numUniqueKernels} {[l[1] for l in logicFiles]}")

    return libraries, khos, numUniqueKernels, numKernels, totalSoln, dupSoln


def generateParentLibrary(libraryFormat: str, libraryPath: Union[Path, str], masterLibs: Dict[str, MasterSolutionLibrary]):
    for arch, masterLib in masterLibs.items():
        name = "TensileLibrary_" + "lazy_" + arch
        masterLib.applyNaming(getRequiredParametersMin())  # <-- This should be able to be replaced directly with `name`?
        LibraryIO.write(str(libraryPath / name), Utils.state(masterLib), libraryFormat)


def createDirectories(outputPath):
    outputPath = Path(outputPath)
    buildTmp = outputPath.parent / "build_tmp" / str(outputPath.name).upper()
    srcCodeObjectPath = buildTmp / "code_object_tmp"
    srcCodeObjectPath.mkdir(parents=True, exist_ok=True)
    kernelsIncludePath = outputPath / "Kernels"
    kernelsIncludePath.mkdir(parents=True, exist_ok=True)
    libraryPath = outputPath / "library"
    libraryPath.mkdir(exist_ok=True)
    return outputPath, buildTmp, srcCodeObjectPath, libraryPath


def extracBuildResults(result):
    numKernels = 0
    numUniqueKernerls = 0
    numDuplicateKernels = 0
    numSoln = 0
    numDuplicateSoln = 0
    masterLibs = {}
    nextIdx = 0
    kho = []

    for library, khos, uniqueKernels, numKerns, soln, dupSoln in result:
      if len(library) > 0:
        for gfxName, lib in library:
          updateParentMasterLibrary(gfxName, lib, masterLibs, nextIdx)
      numUniqueKernerls += uniqueKernels
      numKernels += numKerns
      numDuplicateSoln += dupSoln
      numSoln += soln
      kho.extend(khos)

    return kho, masterLibs, numKernels, numUniqueKernerls, numDuplicateKernels, numSoln, numDuplicateSoln


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
  ensurePath(arguments["OutputPath"])
  arguments["OutputPath"] = os.path.abspath(arguments["OutputPath"])
  outputPath, buildTmpPath, srcCodeObjectPath, libraryPath = createDirectories(arguments["OutputPath"])

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

  asmToolchain = AssemblyToolchain(assembler, offloadBundler, globalParameters["BuildIdKind"], arguments["CodeObjectVersion"])
  srcToolchain = SourceToolchain(cxxCompiler, rocObjExtract, rocObjLs, globalParameters["BuildIdKind"], globalParameters["AsanBuild"], globalParameters["SaveTemps"])

  copyStaticFiles(outputPath)
  unsortedLogic = getLogicFileList(arguments)
  logicFiles = list(filter(lambda x: x != [], schedule(unsortedLogic, 2*arguments["CpuThreads"])))
  unaryBuild = functools.partial(build, arguments, cxxCompiler, assembler, asmToolchain, srcToolchain)
  result  = ParallelMap2(unaryBuild, logicFiles, "Building Library", multiArg=False, return_as="generator_unordered")

  kho, masterLibs, numKernels, numUniqueKernels, numDuplicateKernels, numSoln, numDuplicateSoln = extracBuildResults(result)
  unaryWriteHelpers = functools.partial(writeHelper, outputPath)
  srcFiles = ParallelMap2(unaryWriteHelpers, list(dict.fromkeys(kho)), "Generating Kernels code", multiArg=False, return_as="list")
  kernelsLib = str(srcCodeObjectPath / "Kernels.so")
  srcToolchain.compile(srcFiles, str(kernelsLib), str(outputPath), archs)
  buildSourceCodeObjectFile(srcToolchain, libraryPath, kernelsLib)

  generateParentLibrary(arguments["LibraryFormat"], libraryPath, masterLibs)

  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

  stop = timer()

  print1(f"Total time (s): {(stop-start):3.2f}")
  print1(f"Total kernels: {numKernels}")
  print1(f"Total kernels processed: {numUniqueKernels}")
  print1(f"Duplicate kernels removed: {numDuplicateKernels}")
  print1(f"Kernels processed per second: {(numKernels/(stop-start)):3.2f}")
  print1(f"Total solutions processed: {numSoln}")
  print1(f"Duplicate solutions: {numDuplicateSoln}")
  
