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

# This script only gets called by CMake

if __name__ == "__main__":
    print("This file can no longer be run as a script.  Run 'Tensile/bin/TensileCreateLibrary' instead.")
    exit(1)

import functools

from . import Common
from . import ClientExecutable
from . import EmbeddedData
from . import LibraryIO
from . import Utils
from .Toolchain.Assembly import AssemblyToolchain, buildAssemblyCodeObjectFiles
from .Toolchain.Source import SourceToolchain, buildSourceCodeObjectFile
from .Toolchain.Validators import validateToolchain, getVersion, ToolchainDefaults
from .TensileInstructions import getGfxName, TensileInstructions
from .Common import globalParameters, HR, print1, print2, printExit, ensurePath, \
                    CHeader, assignGlobalParameters, \
                    architectureMap, printWarning, \
                    IsaVersion
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterBase import KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H
from .SolutionLibrary import MasterSolutionLibrary
from .SolutionStructs import Solution
from .CustomYamlLoader import load_logic_gfx_arch
from .Utilities.Profile import profile
import argparse
import collections
import glob
import itertools
import os
import shutil
import sys
from timeit import default_timer as timer
from pathlib import Path
from typing import Sequence, List, Union, NamedTuple, Optional

def timing(func):
  def wrapper(*args, **kwargs):
    start = timer()
    res = func(*args, **kwargs)
    end = timer()

    if globalParameters['PrintTiming']:
      print(f'{func.__name__} took {end - start} seconds')

    return res
  return wrapper


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
    # get kernel name
    kernelWriter.setTensileInstructions(ti)
    asmFilename = kernelWriter.getKernelFileBase(kernel)
    err, src = kernelWriter.getSourceFileString(kernel)
    header = kernelWriter.getHeaderFileString(kernel)
    # will be put in Kernels.h/cpp if None
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
    path = Path(asmPath) / f"{result.name}.s"
    isa =  result.isa
    wfsize = result.wavefrontSize
    with open(path, "w", encoding="utf-8") as f:
      f.write(result.src)
      del result # result.src is very large so let gc know to clean up asap
 
    return path, isa, wfsize


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


################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeSolutionsAndKernels(outputPath, asmToolchain, srcToolchain, solutions, kernels, kernelHelperObjs, \
    kernelWriterAssembly, errorTolerant=False, compress=True):
  codeObjectFiles = []

  # Push working path into build_tmp folder because there may be more than
  # one process running this script. This is to avoid build directory clashing.
  # NOTE: file paths must not contain the lower case word 'kernel' or the
  # /opt/rocm/bin/extractkernel will fail.
  # See buildSourceCodeObjectFile:167 for the call to this binary.
  Common.pushWorkingPath('build_tmp')
  Common.pushWorkingPath(os.path.basename(outputPath).upper())
  asmPath = ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly"))

  asmKernels = [k for k in kernels if k['KernelLanguage'] == 'Assembly']

  # Kernels may be intended for different co files, but generate the same .o file
  # Mark duplicate kernels to avoid race condition
  # @TODO improve organization so this problem doesn't appear
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
  asmResults = Common.ParallelMap2(processKernelSource, asmIter, "Generating assembly kernels")
  removeInvalidSolutionsAndKernels(asmResults, asmKernels, solutions, errorTolerant, globalParameters)
  def assemble(ret):
    p, isa, wavefrontsize = ret
    asmToolchain.assemble(str(p), str(p.with_suffix(".o")), getGfxName(isa), wavefrontsize)
  unaryWriteAssembly = functools.partial(writeAssembly, asmPath)
  compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
  ret = Common.ParallelMap2(compose(assemble, unaryWriteAssembly), asmResults, "Writing assembly kernels", return_as="list", multiArg=False)
  codeObjectFiles += buildAssemblyCodeObjectFiles(asmToolchain, asmKernels, kernelWriterAssembly, outputPath, compress)

  srcKernels = [k for k in kernels if k['KernelLanguage'] != 'Assembly']
  if srcKernels:
    raise ValueError(f"Non-helper HIP source kernels are not supported Tensilelite, found {len(srcKernels)}")
  writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
  srcKernelFile = Path(outputPath) / "Kernels.cpp"
  buildSourceCodeObjectFile(srcToolchain, outputPath, srcKernelFile)

  Common.popWorkingPath() # build_tmp
  Common.popWorkingPath() # workingDir

  return codeObjectFiles, numKernels


def writeSolutionsAndKernelsTCL(outputPath, asmToolchain, srcToolchain, kernels, kernelHelperObjs, \
    kernelWriterAssembly, compress=True):

  Common.pushWorkingPath('build_tmp')
  Common.pushWorkingPath(os.path.basename(outputPath).upper())
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

  uniqueAsmKernels = [k for k in asmKernels if not k.duplicate]

  numAsmKernels = len(asmKernels)
  numKernels = len(kernels)
  assert numKernels == numAsmKernels, "Only assembly kernels are supported in TensileLite"

  def assemble(ret):
    p, isa, wavefrontsize = ret
    asmToolchain.assemble(str(p), str(p.with_suffix(".o")), getGfxName(isa), wavefrontsize)
  unaryProcessKernelSource = functools.partial(processKernelSource, kernelWriterAssembly, TensileInstructions())
  unaryWriteAssembly = functools.partial(writeAssembly, asmPath)
  compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
  ret = Common.ParallelMap2(compose(assemble, unaryWriteAssembly, unaryProcessKernelSource), uniqueAsmKernels, "Generating assembly kernels", multiArg=False)
  buildAssemblyCodeObjectFiles(asmToolchain, asmKernels, kernelWriterAssembly, outputPath, compress)

  writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
  srcKernelFile = Path(outputPath) / "Kernels.cpp"
  buildSourceCodeObjectFile(srcToolchain, outputPath, srcKernelFile)

  Common.popWorkingPath() # build_tmp
  Common.popWorkingPath() # workingDir

  return numKernels


##############################################################################
# Min Naming / Solution and Kernel Writers
##############################################################################
@timing
def getSolutionAndKernelWriters(solutions, kernels, assembler):

  # if any kernels are assembly, append every ISA supported
  kernelSerialNaming   = Solution.getSerialNaming(kernels)

  solutionMinNaming    = Solution.getMinNaming(solutions)
  kernelMinNaming      = Solution.getMinNaming(kernels)
  kernelWriterAssembly = KernelWriterAssembly(kernelMinNaming, kernelSerialNaming, assembler)

  return (kernelWriterAssembly, kernelMinNaming, solutionMinNaming)

################################################################################
# copy static cpp files and headers
################################################################################
@timing
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


################################################################################
# Generate Kernel Objects From Solutions
################################################################################
@timing
def generateKernelObjectsFromSolutions(solutions):
  # create solution writer and kernel writer
  kernels = []
  kernelHelperObjs = []
  kernelNames = set()
  kernelHelperNames = set()

  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
        kName = Solution.getKeyNoInternalArgs(kernel)
        if kName not in kernelNames:
            kernels.append(kernel)
            kernelNames.add(kName)
    solutionHelperKernels = solution.getHelperKernelObjects()
    kernelHelperObjs += solutionHelperKernels
    for ko in solutionHelperKernels:
      kernelHelperNames.add(ko.getKernelName())

  # remove duplicates while preserving order
  kernelHelperObjs = list(dict.fromkeys(kernelHelperObjs))
  return (kernels, kernelHelperObjs, kernelHelperNames)

################################################################################
# Generate Logic Data and Solutions
################################################################################
@timing
def generateLogicDataAndSolutions(logicFiles, args, cxxCompiler):

  # skip the logic which architectureName is not in the build target.
  if ";" in args.Architecture:
    archs = args.Architecture.split(";") # user arg list format
  else:
    archs = args.Architecture.split("_") # workaround for cmake list in list issue

  solutions = []
  masterLibraries = {}
  fullMasterLibrary = None
  nextSolIndex = 0
  matchTable = {}
  fIter = zip(logicFiles, itertools.repeat(cxxCompiler), itertools.repeat(archs))

  def libraryIter(lib: MasterSolutionLibrary):
    if len(lib.solutions):
      for i, s in enumerate(lib.solutions.items()):
        yield (i, *s)
    else:
      for _, lazyLib in lib.lazyLibraries.items():
        yield from libraryIter(lazyLib)

  for library in Common.ParallelMap2(LibraryIO.parseLibraryLogicFile, fIter, "Loading Logics...", return_as="generator_unordered"):
    _, architectureName, _, _, _, newLibrary, srcFile = library

    if architectureName == "":
      continue

    if globalParameters["SeparateArchitectures"] or globalParameters["LazyLibraryLoading"]:
      if architectureName in masterLibraries:
        nextSolIndex = masterLibraries[architectureName].merge(newLibrary, nextSolIndex)
      else:
        masterLibraries[architectureName] = newLibrary
        masterLibraries[architectureName].version = args.version
    else:
      if fullMasterLibrary is None:
        fullMasterLibrary = newLibrary
        fullMasterLibrary.version = args.version
      else:
        fullMasterLibrary.merge(newLibrary)

    if args.GenSolTable:
      # Match yaml file solutions to solution index
      for localIdx, _, s in libraryIter(newLibrary):
        matchTable[s.index] = [srcFile, localIdx]

  if globalParameters["SeparateArchitectures"] or globalParameters["LazyLibraryLoading"]:
    if "fallback" in masterLibraries.keys():
      for key, value in masterLibraries.items():
        if key != "fallback":
          value.merge(masterLibraries["fallback"])

      masterLibraries.pop("fallback")

    for _, masterLibrary in masterLibraries.items():
      for _, sol in masterLibrary.solutions.items():
        solutions.append(sol.originalSolution)
      for name, lib in masterLibrary.lazyLibraries.items():
        for _, sol in lib.solutions.items():
          sol.originalSolution._state["codeObjectFile"] = name
          solutions.append(sol.originalSolution)
  else:
    solutions = [sol.originalSolution for _, sol in fullMasterLibrary.solutions.items()]

  # remove duplicates while preserving order
  solutions = dict.fromkeys(solutions).keys()

  if args.GenSolTable:
    LibraryIO.write("MatchTable", matchTable)

  return solutions, masterLibraries, fullMasterLibrary


def validateLibrary(masterLibraries: MasterSolutionLibrary,
                    kernels: Sequence[Solution],
                    kernelWriterAssembly: KernelWriterAssembly):
  kernelsByCodeObjectFiles = {k: list(g) for k, g in itertools.groupby(kernels, lambda k: k["codeObjectFile"])}

  ok: bool = True

  for _, lib in masterLibraries.items():
    for name, llib in lib.lazyLibraries.items():
      uniqueKernelsInLib = {kernelWriterAssembly.getKernelName(s.originalSolution) for s in llib.solutions.values()}

      if len(uniqueKernelsInLib) != len(kernelsByCodeObjectFiles[name]):
        ok = False
        print(f"{name} library and co has inconsistent kernel size {len(uniqueKernelsInLib)} vs {len(kernelsByCodeObjectFiles[name])}")

  assert ok and "Inconsistent kernel sizes detected!"

################################################################################
# Tensile Create Library
################################################################################
@profile
def TensileCreateLibrary():
  start = timer()
  print1("")
  print1(HR)
  print1("# Tensile Create Library")
  print2(HR)
  print2("")

  ##############################################################################
  # Parse Command Line Arguments
  ##############################################################################
  def splitExtraParameters(par):
    """
    Allows the --global-parameters option to specify any parameters from the command line.
    """

    (key, value) = par.split("=")
    value = eval(value)
    return (key, value)

  print2("Arguments: %s" % sys.argv)
  argParser = argparse.ArgumentParser()
  argParser.add_argument("LogicPath",       help="Path to LibraryLogic.yaml files.")
  argParser.add_argument("OutputPath",      help="Where to write library files?")
  argParser.add_argument("RuntimeLanguage", help="Which runtime language?", choices=["OCL", "HIP", "HSA"])
  argParser.add_argument("--cxx-compiler",           dest="CxxCompiler",       action="store", default=ToolchainDefaults.CXX_COMPILER,
                         help=f"Default: {ToolchainDefaults.CXX_COMPILER}")
  argParser.add_argument("--c-compiler",             dest="CCompiler",         action="store", default=ToolchainDefaults.C_COMPILER)
  argParser.add_argument("--cmake-cxx-compiler",     dest="CmakeCxxCompiler",  action="store")
  argParser.add_argument("--offload-bundler",        dest="OffloadBundler",    action="store", default=ToolchainDefaults.OFFLOAD_BUNDLER)
  argParser.add_argument("--assembler",              dest="Assembler",         action="store", default=ToolchainDefaults.ASSEMBLER)
  argParser.add_argument("--code-object-version",    dest="CodeObjectVersion", choices=["4", "5"], action="store", default="4", type=str)
  argParser.add_argument("--architecture",           dest="Architecture",      type=str, action="store", default="all", help="Supported archs: " + " ".join(architectureMap.keys()))
  argParser.add_argument("--short-file-names",       dest="ShortNames",        action="store_true")
  argParser.add_argument("--no-short-file-names",    dest="ShortNames",        action="store_false")
  argParser.add_argument("--library-print-debug",    dest="LibraryPrintDebug", action="store_true")
  argParser.add_argument("--no-library-print-debug", dest="LibraryPrintDebug", action="store_false")
  argParser.add_argument("--no-compress",            dest="NoCompress",        action="store_true", help="Don't compress assembly code objects.")
  argParser.add_argument("--experimental",           dest="Experimental",      action="store_true",
                         help="Include logic files in directories named 'Experimental'.")
  argParser.add_argument("--no-enumerate",           action="store_true", help="Do not run rocm_agent_enumerator.")
  argParser.add_argument("--version", help="Version string to embed into library file.")
  argParser.add_argument("--logic-format", dest="LogicFormat", choices=["yaml", "json"], \
                         action="store", default="yaml", help="select which logic format to use")
  argParser.add_argument("--library-format", dest="LibraryFormat", choices=["yaml", "msgpack"],
                         action="store", default="msgpack", help="select which library format to use")
  argParser.add_argument("--generate-sources-and-exit",   dest="GenerateSourcesAndExit", action="store_true",
                          default=False, help="Output source files only and exit.")
  argParser.add_argument("--jobs", "-j", dest="CpuThreads", type=int,
                          default=-1, help="Number of parallel jobs to launch.")
  argParser.add_argument("--verbose", "-v", dest="PrintLevel", type=int,
                          default=1, help="Set printout verbosity level.")
  argParser.add_argument("--print-timing", dest="PrintTiming",
                          default=False, action="store_true", help="Print duration of each stage.")
  argParser.add_argument("--separate-architectures", dest="SeparateArchitectures", action="store_true",
                         default=False, help="Separates TensileLibrary file by architecture")
  argParser.add_argument("--lazy-library-loading", dest="LazyLibraryLoading", action="store_true",
                         default=False, help="Loads Tensile libraries when needed instead of upfront.")
  argParser.add_argument("--enable-marker", dest="EnableMarker", action="store_true",
                         default=False, help="Enable marker in Tensile.")
  argParser.add_argument("--global-parameters", nargs="+", type=splitExtraParameters, default=[])
  argParser.add_argument("--no-generate-solution-table", dest="GenSolTable", action="store_false", default=True,
                         help="Skip generating solution-yaml matching table")
  argParser.add_argument("--asm-debug", dest="AsmDebug", action="store_true", default=False,
                         help="Keep debug information for built code objects")
  argParser.add_argument("--build-id", dest="BuildIdKind", action="store", default="sha1")
  argParser.add_argument("--address-sanitizer", dest="AsanBuild", action="store_true",
                         default=False, help="Enable ASAN build.")
  argParser.add_argument("--keep-build-tmp", dest="KeepBuildTmp", action="store_true",
                          default=False, help="Do not remove the temporary build directory (may required hundreds of GBs of space)"),
  argParser.add_argument("--validate-library", dest="ValidateLibrary", action="store_true", default=False)
  argParser.add_argument("--logic-filter", dest="LogicFilter", action="store", default="*", type=str,
                        help="Cutomsized logic filter, default is *, i.e. all logics."
                        " Example: gfx942/Equality/* for building equality of gfx942 only")

  args = argParser.parse_args()
  args.CodeObjectVersion = "4" if args.CodeObjectVersion == "default" else args.CodeObjectVersion

  logicPath = args.LogicPath
  outputPath = args.OutputPath
  cxxCompiler = args.CxxCompiler
  offloadBundler   = args.OffloadBundler
  assembler = args.Assembler
  libraryFormat = args.LibraryFormat
  useCompression = not args.NoCompress
  coVersion = args.CodeObjectVersion

  print2("OutputPath: %s" % outputPath)
  ensurePath(outputPath)
  outputPath = os.path.abspath(outputPath)
  arguments = {}
  arguments["RuntimeLanguage"] = args.RuntimeLanguage
  arguments["CodeObjectVersion"] = args.CodeObjectVersion
  arguments["Architecture"] = args.Architecture
  arguments["SeparateArchitectures"] = args.SeparateArchitectures
  arguments["LazyLibraryLoading"] = args.LazyLibraryLoading
  arguments["EnableMarker"] = args.EnableMarker
  if args.CmakeCxxCompiler:
    os.environ["CMAKE_CXX_COMPILER"] = args.CmakeCxxCompiler
  arguments["ShortNames"] = args.ShortNames
  arguments["LibraryPrintDebug"] = args.LibraryPrintDebug
  arguments["CodeFromFiles"] = False
  arguments["LogicFormat"]  = args.LogicFormat
  arguments["LibraryFormat"] = args.LibraryFormat
  if args.no_enumerate:
    arguments["AMDGPUArchPath"] = False

  arguments["GenerateSourcesAndExit"] = args.GenerateSourcesAndExit
  if arguments["GenerateSourcesAndExit"]:
    # Generated sources are preserved and go into output dir
    arguments["WorkingPath"] = outputPath

  arguments["CpuThreads"] = args.CpuThreads
  arguments["PrintLevel"] = args.PrintLevel
  arguments["PrintTiming"] = args.PrintTiming
  arguments["AsmDebug"] = args.AsmDebug
  arguments["BuildIdKind"] = args.BuildIdKind
  arguments["KeepBuildTmp"] = args.KeepBuildTmp
  arguments["AsanBuild"] = args.AsanBuild
  arguments["ValidateLibrary"] = args.ValidateLibrary

  for key, value in args.global_parameters:
    arguments[key] = value

  cxxCompiler, cCompiler, offloadBundler, assembler, hipconfig = validateToolchain(
      args.CxxCompiler, args.CCompiler, args.OffloadBundler, args.Assembler, ToolchainDefaults.HIP_CONFIG
  )
  print1(f"# HIP Version:         {getVersion(hipconfig, regex=r'(.+)')}")
  print1(f"# Cxx Compiler:        {cxxCompiler} (version {getVersion(cxxCompiler)})")
  print1(f"# C Compiler:          {cCompiler} (version {getVersion(cCompiler)})")
  print1(f"# Assembler:           {assembler} (version {getVersion(assembler)})")
  print1(f"# Offload Bundler:     {offloadBundler} (version {getVersion(offloadBundler)})")
  print1(f"# Code Object Version: {coVersion}")
  print1(f"# Architecture(s):     {arguments['Architecture']}")
  print1(f"# Library Format:      {libraryFormat}")

  assignGlobalParameters(arguments, cxxCompiler)

  asmToolchain = AssemblyToolchain(assembler, offloadBundler, globalParameters["BuildIdKind"], coVersion)
  srcToolchain = SourceToolchain(cxxCompiler, offloadBundler, globalParameters["BuildIdKind"], globalParameters["AsanBuild"], globalParameters["SaveTemps"])

  if not os.path.exists(logicPath):
    printExit("LogicPath %s doesn't exist" % logicPath)

  if ";" in arguments["Architecture"]:
    archs = arguments["Architecture"].split(";") # user arg list format
  else:
    archs = arguments["Architecture"].split("_") # workaround for cmake list in list issue
  logicArchs = set()
  for arch in archs:
    if arch in architectureMap:
      logicArchs.add(architectureMap[arch])
    else:
      printExit("Architecture %s not supported" % arch)

  # Recursive directory search
  logicExtFormat = ".yaml"
  if args.LogicFormat == "yaml":
    pass
  elif args.LogicFormat == "json":
    logicExtFormat = ".json"
  else:
    printExit("Unrecognized LogicFormat", args.LogicFormat)

  def archMatch(arch: str, archs: List[str]):
    return (arch in archs) or any(a.startswith(arch) for a in archs)

  def validLogicFile(p: Path):
    return p.suffix == logicExtFormat and ("all" in archs or archMatch(load_logic_gfx_arch(p), archs))

  globPattern = os.path.join(logicPath, f"**/{args.LogicFilter}{logicExtFormat}")
  print1(f"# LogicFilter:       {globPattern}")
  logicFiles = (os.path.join(logicPath, file) for file in glob.iglob(globPattern, recursive=True))
  logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]

  print1(f"# Experimental:      {args.Experimental}")
  if not args.Experimental:
    logicFiles = [file for file in logicFiles if "experimental" not in map(str.lower, Path(file).parts)]

  print2(f"# LibraryLogicFiles: {len(logicFiles)}")
  for logicFile in logicFiles:
    print2("#   %s" % logicFile)


  ##############################################################################
  # Parse config files
  ##############################################################################

  # Parse logicData, solutions, and masterLibraries from logic files
  solutions, masterLibraries, fullMasterLibrary = generateLogicDataAndSolutions(logicFiles, args, cxxCompiler)
  kernels, kernelHelperObjs, _ = generateKernelObjectsFromSolutions(solutions)
  # if any kernels are assembly, append every ISA supported
  kernelWriterAssembly, kernelMinNaming, _ = getSolutionAndKernelWriters(solutions, kernels, assembler)

  if globalParameters["ValidateLibrary"]:
    validateLibrary(masterLibraries, kernels, kernelWriterAssembly)

  staticFiles = copyStaticFiles(outputPath)

  # Make sure to copy the library static files.
  for fileName in staticFiles:
    shutil.copy( os.path.join(globalParameters["SourcePath"], fileName), \
      outputPath )

  # write solutions and kernels
  numKernels = writeSolutionsAndKernelsTCL(outputPath, asmToolchain, srcToolchain, kernels, 
                                           kernelHelperObjs, kernelWriterAssembly, compress=useCompression)

  archs = [getGfxName(arch) for arch in globalParameters['SupportedISA'] \
             if globalParameters["AsmCaps"][arch]["SupportedISA"]]
  newLibraryDir = ensurePath(os.path.join(outputPath, 'library'))

  if globalParameters["SeparateArchitectures"] or globalParameters["LazyLibraryLoading"]:
    for archName, newMasterLibrary in masterLibraries.items():
      if archName in archs:
        if globalParameters["LazyLibraryLoading"]:
          masterFile = os.path.join(newLibraryDir, "TensileLibrary_lazy_"+archName)
        else:
          masterFile = os.path.join(newLibraryDir, "TensileLibrary_"+archName)
        newMasterLibrary.applyNaming(kernelMinNaming)
        LibraryIO.write(masterFile, Utils.state(newMasterLibrary), args.LibraryFormat)

        #Write placeholder libraries
        for name, lib in newMasterLibrary.lazyLibraries.items():
          filename = os.path.join(newLibraryDir, name)
          lib.applyNaming(kernelMinNaming) #@TODO Check to see if kernelMinNaming is correct
          LibraryIO.write(filename, Utils.state(lib), args.LibraryFormat)

  else:
    masterFile = os.path.join(newLibraryDir, "TensileLibrary")
    fullMasterLibrary.applyNaming = timing(fullMasterLibrary.applyNaming)
    fullMasterLibrary.applyNaming(kernelMinNaming)
    LibraryIO.write(masterFile, Utils.state(fullMasterLibrary), args.LibraryFormat)

  theMasterLibrary = fullMasterLibrary
  if globalParameters["SeparateArchitectures"] and len(masterLibraries) > 0:
    theMasterLibrary = list(masterLibraries.values())[0]

  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")

  stop = timer()

  print1(f"Total time (s): {(stop-start):3.2f}")
  print1(f"Total kernels processed: {numKernels}")
  print1(f"Kernels processed per second: {(numKernels/(stop-start)):3.2f}")
