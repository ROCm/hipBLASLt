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

from . import Common
from . import ClientExecutable
from . import EmbeddedData
from . import LibraryIO
from . import Utils
from .TensileInstructions import getGfxName, TensileInstructions
from .Common import globalParameters, HR, print1, print2, printExit, ensurePath, \
                    CHeader, CMakeHeader, assignGlobalParameters, \
                    architectureMap, supportedCompiler, printWarning
from .KernelWriterAssembly import KernelWriterAssembly
from .SolutionLibrary import MasterSolutionLibrary
from .SolutionStructs import Solution
from .CustomYamlLoader import load_logic_gfx_arch

import argparse
import collections
import glob
import itertools
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from timeit import default_timer as timer
from pathlib import Path
from typing import Sequence, List

def timing(func):
  def wrapper(*args, **kwargs):
    start = timer()
    res = func(*args, **kwargs)
    end = timer()

    if globalParameters['PrintTiming']:
      print(f'{func.__name__} took {end - start} seconds')

    return res
  return wrapper
################################################################################
def processKernelSource(kernel, kernelWriterAssembly, ti):
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    try:
        kernelWriter = kernelWriterAssembly
        # get kernel name
        kernelWriter.setTensileInstructions(ti)
        kernelName = kernelWriter.getKernelFileBase(kernel)
        (err, src) = kernelWriter.getSourceFileString(kernel)
        header = kernelWriter.getHeaderFileString(kernel)
        # will be put in Kernels.h/cpp if None
        filename = kernel._state.get("codeObjectFile", None)

    except RuntimeError:
        return (1, "", "", kernelName, None)

    return (err, src, header, kernelName, filename)

def getAssemblyCodeObjectFiles(kernels, kernelWriterAssembly, outputPath):
    destDir = ensurePath(os.path.join(outputPath, 'library'))
    asmDir = kernelWriterAssembly.getAssemblyDirectory()
    archs = collections.defaultdict(list)

    for k in filter(lambda k: k['KernelLanguage'] == 'Assembly', kernels):
      archs[tuple(k['ISA'])].append(k)

    coFiles = []

    for arch, archKernels in archs.items():
      if len(archKernels) == 0:
        continue

      archName = getGfxName(arch)

      if globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1 or globalParameters["LazyLibraryLoading"]:
        objectFiles = [kernelWriterAssembly.getKernelFileBase(k) + '.o' for k in archKernels if 'codeObjectFile' not in k]

        #Group kernels from placeholder libraries
        coFileMap = collections.defaultdict(list)
        if len(objectFiles):
          coFileMap[os.path.join(destDir, "TensileLibrary_"+archName+".co")] = objectFiles

        for kernel in archKernels:
          coName = kernel.get("codeObjectFile", None)
          if coName:
            coFileMap[os.path.join(destDir, coName+".co")] += [kernelWriterAssembly.getKernelFileBase(kernel) + '.o']

        for coFile, objectFiles in coFileMap.items():
          if os.name == "nt":
            # On Windows, the objectFiles list command line (including spaces)
            # exceeds the limit of 8191 characters, so using response file

            responseArgs = objectFiles
            responseFile = os.path.join(asmDir, 'clangArgs.txt')
            with open(responseFile, 'wt') as file:
              file.write( " ".join(responseArgs) )
              file.flush()

            args = [globalParameters['AssemblerPath'], '-target', 'amdgcn-amd-amdhsa', '-o', coFile, '@clangArgs.txt']
            subprocess.check_call(args, cwd=asmDir)
          else:
            numOfObjectFiles = len(objectFiles)
            splitFiles = 10000
            if numOfObjectFiles > splitFiles:
              slicedObjectFilesList = [objectFiles[x:x+splitFiles] for x in range(0, numOfObjectFiles, splitFiles)]
              objectFileBasename = os.path.split(coFile)[-1].split('.')[0]
              numOfOneSliceOfObjectFiles = int(math.ceil(numOfObjectFiles / splitFiles))
              newObjectFiles = [ objectFileBasename + "_" + str(i) + ".o" for i in range(0, numOfOneSliceOfObjectFiles)]
              newObjectFilesOutput = []
              for slicedObjectFiles, objectFile in zip(slicedObjectFilesList, newObjectFiles):
                if len(slicedObjectFiles) > 1:
                  args = [globalParameters["ROCmLdPath"], "-r"] + slicedObjectFiles + [ "-o", objectFile ]
                  if globalParameters["PrintCodeCommands"]:
                    print(asmDir)
                    print(' '.join(args))
                  subprocess.check_call(args, cwd=asmDir)
                  newObjectFilesOutput.append(objectFile)
                else:
                  newObjectFilesOutput.append(slicedObjectFiles[0])
              args = kernelWriterAssembly.getLinkCodeObjectArgs(newObjectFilesOutput, coFile)
              if globalParameters["PrintCodeCommands"]:
                print(asmDir)
                print(' '.join(args))
              subprocess.check_call(args, cwd=asmDir)
            else:
              args = kernelWriterAssembly.getLinkCodeObjectArgs(objectFiles, coFile)
              if globalParameters["PrintCodeCommands"]:
                print(asmDir)
                print(' '.join(args))
              subprocess.check_call(args, cwd=asmDir)

          coFiles.append(coFile)
      else:
        # no mergefiles
        def newCoFileName(kName):
          if globalParameters["PackageLibrary"]:
            return os.path.join(destDir, archName, kName + '.co')
          else:
            return os.path.join(destDir, kName + '_' + archName + '.co')

        def orgCoFileName(kName):
          return os.path.join(asmDir, kName + '.co')

        for src, dst in Utils.tqdm(((orgCoFileName(kName), newCoFileName(kName)) for kName in \
                                    map(lambda k: kernelWriterAssembly.getKernelFileBase(k), archKernels)), "Copying code objects"):
          shutil.copyfile(src, dst)
          coFiles.append(dst)
    return coFiles

def which(p):
    if supportedCompiler(p) and 'CMAKE_CXX_COMPILER' in os.environ and os.path.isfile(os.environ['CMAKE_CXX_COMPILER']):
        return os.environ['CMAKE_CXX_COMPILER']
    if os.name == "nt":
      exes = [p+x for x in ['.exe', '', '.bat']]  # bat may be front end for file with no extension
    else:
      exes = [p+x for x in ['', '.exe', '.bat']]
    system_path = os.environ['PATH'].split(os.pathsep)
    for dirname in system_path+[globalParameters["ROCmBinPath"]]:
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.isfile(candidate):
                return candidate
    return None

def splitArchs():
  # Helper for architecture
  def isSupported(arch):
    return globalParameters["AsmCaps"][arch]["SupportedISA"] and \
           globalParameters["AsmCaps"][arch]["SupportedSource"]

  if ";" in globalParameters["Architecture"]:
    wantedArchs = globalParameters["Architecture"].split(";")
  else:
    wantedArchs = globalParameters["Architecture"].split("_")
  archs = []
  cmdlineArchs = []
  if "all" in wantedArchs:
    for arch in globalParameters['SupportedISA']:
      if isSupported(arch):
        if (arch in [(9,0,6), (9,0,8), (9,0,10), (9,4,0), (9,4,1), (9,4,2)]):
          if (arch == (9,0,10)):
            archs += [getGfxName(arch) + '-xnack+']
            cmdlineArchs += [getGfxName(arch) + ':xnack+']
          archs += [getGfxName(arch) + '-xnack-']
          cmdlineArchs += [getGfxName(arch) + ':xnack-']
        else:
          archs += [getGfxName(arch)]
          cmdlineArchs += [getGfxName(arch)]
  else:
    for arch in wantedArchs:
      archs += [re.sub(":", "-", arch)]
      cmdlineArchs += [arch]
  return archs, cmdlineArchs

def buildSourceCodeObjectFile(CxxCompiler, outputPath, kernelFile):
    buildPath = ensurePath(os.path.join(globalParameters['WorkingPath'], 'code_object_tmp'))
    destDir = ensurePath(os.path.join(outputPath, 'library'))
    (_, filename) = os.path.split(kernelFile)
    (base, _) = os.path.splitext(filename)

    if "CmakeCxxCompiler" in globalParameters and globalParameters["CmakeCxxCompiler"] is not None:
      os.environ["CMAKE_CXX_COMPILER"] = globalParameters["CmakeCxxCompiler"]

    objectFilename = base + '.o'
    soFilename = base + '.so'

    coFilenames = []

    if supportedCompiler(CxxCompiler):
      archs, cmdlineArchs = splitArchs()

      archFlags = ['--offload-arch=' + arch for arch in cmdlineArchs]

      # needs to be fixed when Maneesh's change is made available
      hipFlags = ["-D__HIP_HCC_COMPAT_MODE__=1"]
      hipFlags += (
          ["--genco"] if CxxCompiler == "hipcc" else ["--cuda-device-only", "-x", "hip", "-O3"]
      )
      # if CxxCompiler == "amdclang++":
      # hipFlags += ["-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false"]

      hipFlags += ['-I', outputPath]
      hipFlags += ["-Xoffload-linker", "--build-id=%s"%globalParameters["BuildIdKind"]]
      hipFlags += ['-std=c++17']
      if globalParameters["SaveTemps"]:
        hipFlags += ['--save-temps']

      launcher = shlex.split(os.environ.get('Tensile_CXX_COMPILER_LAUNCHER', ''))

      if os.name == "nt":
        hipFlags += ['-fms-extensions', '-fms-compatibility', '-fPIC', '-Wno-deprecated-declarations']
        compileArgs = launcher + [which(CxxCompiler)] + hipFlags + archFlags + [kernelFile, '-c', '-o', os.path.join(buildPath, objectFilename)]
      else:
        compileArgs = launcher + [which(CxxCompiler)] + hipFlags + archFlags + [kernelFile, '-c', '-o', os.path.join(buildPath, objectFilename)]

      if globalParameters["PrintCodeCommands"]:
        print(CxxCompiler + ':' + ' '.join(compileArgs))
      subprocess.check_call(compileArgs)

      # If we aren't using hipcc what happens?
      # get hipcc version due to compatiblity reasons
      hipccver = globalParameters['HipClangVersion'].split(".")
      hipccMaj = int(hipccver[0])
      hipccMin = int(hipccver[1])
      # for hipclang 5.2 and above, clang offload bundler changes the way input/output files are specified
      inflag = "-inputs"
      outflag = "-outputs"
      if (hipccMaj == 5 and hipccMin >= 2) or hipccMaj >= 6:
        inflag = "-input"
        outflag = "-output"

      infile = os.path.join(buildPath, objectFilename)
      try:
        bundlerArgs = [globalParameters["ClangOffloadBundlerPath"], "-type=o", "%s=%s" % (inflag, infile), "-list"]
        listing = subprocess.check_output(bundlerArgs, stderr=subprocess.STDOUT).decode().split("\n")
        for target in listing:
          matched = re.search("gfx.*$", target)
          if matched:
            arch = re.sub(":", "-", matched.group())
            if "TensileLibrary" in base and "fallback" in base:
              outfile = os.path.join(buildPath, "{0}_{1}.hsaco".format(base, arch))
            elif "TensileLibrary" in base:
              variant = [t for t in ["", "xnack-", "xnack+"] if t in target][-1]
              baseVariant = base+"-"+variant if variant else base
              if arch in baseVariant:
                outfile = os.path.join(buildPath, baseVariant+".hsaco")
              else:
                outfile = None
            else:
              outfile = os.path.join(buildPath, "{0}-000-{1}.hsaco".format(soFilename, arch))

            #Compilation
            if outfile:
              coFilenames.append(os.path.split(outfile)[1])
              #bundlerArgs = [globalParameters["ClangOffloadBundlerPath"], "-type=o", "-targets=%s" % target, "-inputs=%s" % infile, "-outputs=%s" % outfile, "-unbundle"]
              bundlerArgs = [globalParameters["ClangOffloadBundlerPath"], "-type=o", "-targets=%s" % target,
                           "%s=%s" % (inflag, infile), "%s=%s" % (outflag, outfile), "-unbundle"]
              if globalParameters["PrintCodeCommands"]:
                print(' '.join(bundlerArgs))
              subprocess.check_call(bundlerArgs)

      except subprocess.CalledProcessError:
        for i in range(len(archs)):
          outfile = os.path.join(buildPath, "{0}-000-{1}.hsaco".format(soFilename, archs[i]))
          coFilenames.append(os.path.split(outfile)[1])
          #bundlerArgs = [globalParameters["ClangOffloadBundlerPath"], "-type=o", "-targets=hip-amdgcn-amd-amdhsa--%s" % cmdlineArchs[i], "-inputs=%s" % infile, "-outputs=%s" % outfile, "-unbundle"]
          bundlerArgs = [globalParameters["ClangOffloadBundlerPath"], "-type=o", "-targets=hip-amdgcn-amd-amdhsa--%s" % cmdlineArchs[i],
                         "%s=%s" % (inflag, infile), "%s=%s" % (outflag, outfile), "-unbundle"]
          if globalParameters["PrintCodeCommands"]:
            print(' '.join(bundlerArgs))
          subprocess.check_call(bundlerArgs)
    else:
      raise RuntimeError("Unknown compiler {}".format(CxxCompiler))

    destCosList = []
    if "PackageLibrary" in globalParameters and globalParameters["PackageLibrary"]:
      for arch in archs:
        ensurePath(os.path.join(destDir, arch))
        archCoFilenames = [name for name in coFilenames if arch in name]
        extractedCOs = [os.path.join(buildPath, name) for name in archCoFilenames]
        destCOs = [os.path.join(destDir, arch, name) for name in archCoFilenames]
        destCosList += destCOs
        if globalParameters["PrintCodeCommands"]:
          print ("# copy source code objects    : ", extractedCOs)
          print ("# to dest source code objects : ", destCOs)
        for (src, dst) in zip(extractedCOs, destCOs):
          shutil.copyfile(src, dst)
    else:
      coFilenames = [name for name in coFilenames]
      extractedCOs = [os.path.join(buildPath, name) for name in coFilenames]
      destCOs = [os.path.join(destDir, name) for name in coFilenames]
      destCosList += destCOs
      for (src, dst) in zip(extractedCOs, destCOs):
        shutil.copyfile(src, dst)

    return destCosList

def buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath):
    args    = zip(itertools.repeat(CxxCompiler), itertools.repeat(outputPath), kernelFiles)
    coFiles = Common.ParallelMap2(buildSourceCodeObjectFile, args, "Compiling source kernels")

    return itertools.chain.from_iterable(coFiles)

################################################################################
def prepAsm(kernelWriterAssembly):
  """
  Create and prepare the assembly directory  - called ONCE per output dir:
  """
  asmPath = ensurePath(os.path.join(globalParameters["WorkingPath"], "assembly") )
  assemblerFileName = os.path.join(asmPath, \
      "asm-new.%s"%("bat" if os.name=="nt" else "sh"))
  assemblerFile = open(assemblerFileName, "w")
  if os.name == "nt":
    assemblerFile.write("echo Windows: Copying instead of Assembling\n")
    assemblerFile.write("copy %1.s %1.o\n")
    assemblerFile.write("copy %1.o %1.co\n")
  else:
    assemblerFile.write("#!/bin/sh {log}\n".format(log = "-x" if globalParameters["PrintLevel"] >=2  else ""))
    assemblerFile.write("# usage: asm-new.sh kernelName(no extension) [--wave32]\n")

    assemblerFile.write("f=$1\n")
    assemblerFile.write("shift\n")
    assemblerFile.write('if [ ! -z "$1" ] && [ "$1" = "--wave32" ]; then\n')
    assemblerFile.write("    wave=32\n")
    assemblerFile.write("    shift\n")
    assemblerFile.write("else\n")
    assemblerFile.write("    wave=64\n")
    assemblerFile.write("fi\n")


    isa = globalParameters["CurrentISA"]
    assemblerFile.write("h={gfxName}\n".format(gfxName = getGfxName(isa)))

    debug = globalParameters.get("AsmDebug", False)
    cArgs32 = kernelWriterAssembly.getCompileArgs("$f.s", "$f.o", isa=isa, wavefrontSize=32, debug=debug)
    cArgs64 = kernelWriterAssembly.getCompileArgs("$f.s", "$f.o", isa=isa, wavefrontSize=64, debug=debug)
    lArgs = kernelWriterAssembly.getLinkCodeObjectArgs(["$f.o"], "$f.co")

    assemblerFile.write("if [ $wave -eq 32 ]; then\n")
    assemblerFile.write(" ".join(cArgs32) + "\n")
    assemblerFile.write("else\n")
    assemblerFile.write(" ".join(cArgs64) + "\n")
    assemblerFile.write("fi\n")


    assemblerFile.write(" ".join(lArgs) + "\n")

    assemblerFile.write("ERR=$?\n")
    assemblerFile.write("if [ $ERR -ne 0 ]\n")
    assemblerFile.write("then\n")
    assemblerFile.write("    echo one\n")
    assemblerFile.write("    exit $ERR\n")
    assemblerFile.write("fi\n")

    assemblerFile.write("cp $f.co ../../../library/${f}_$h.co\n")
    assemblerFile.write("mkdir -p ../../../asm_backup && ")
    assemblerFile.write("cp $f.s ../../../asm_backup/$f.s\n")

  assemblerFile.close()
  os.chmod(assemblerFileName, 0o777)

################################################################################
def buildKernelSourceAndHeaderFiles(results, outputPath, kernelsWithBuildErrs):
  """
  Logs errors and writes appropriate info to kernelSourceFile and kernelHeaderFile.

  Arguments:
    results:              list of (err, src, header, kernelName, filename)
    outputPath:           path to source directory
    kernelsWithBuildErrs: Dictionary to be updated with kernels that have errors
    kernelSourceFile:     File to write source data to
    kernelHeaderFile:     File to write header data to

  Returns:
    sourceFilenames:      Array containing source kernel filenames
  """

  # Find kernels to write
  kernelsToWrite = []
  filesToWrite = collections.defaultdict(list)
  validKernelCount = 0
  for (err,src,header,kernelName, filename) in results:

    # Keep track of kernels with errors
    if err:
      kernelsWithBuildErrs[kernelName] = err

    # Don't create a file for empty kernels
    if len(src.strip()) == 0:
      continue

    kernelsToWrite.append((err, src, header, kernelName))

    # Create list of files
    if filename:
      filesToWrite[os.path.join(os.path.normcase(outputPath),filename)].append((err, src, header, kernelName))
    elif globalParameters["MergeFiles"]:
      kernelSuffix = ""
      if globalParameters["NumMergedFiles"] > 1:
        kernelSuffix = validKernelCount % globalParameters["NumMergedFiles"]

      filesToWrite[os.path.join(os.path.normcase(outputPath), "Kernels"+kernelSuffix)]\
        .append((err, src, header, kernelName))
    else:
      filesToWrite[os.path.join(os.path.normcase(outputPath),kernelName)].append((err, src, header, kernelName))

    validKernelCount += 1

  #Ensure there's at least one kernel file for helper kernels
  if globalParameters["LazyLibraryLoading"] or (globalParameters["MergeFiles"] and not kernelsToWrite):
    kernelSuffix = ""
    if globalParameters["NumMergedFiles"] > 1:
      kernelSuffix = "0"

    filesToWrite[os.path.join(os.path.normcase(outputPath), "Kernels"+kernelSuffix)] = []


  # Write kernel data to files
  #Parse list of files and write kernels
  for filename, kernelList in filesToWrite.items():
    with open(filename+".h", "w", encoding="utf-8") as kernelHeaderFile, \
          open(filename+".cpp", "w", encoding="utf-8") as kernelSourceFile:

      kernelSourceFile.write(CHeader)
      kernelHeaderFile.write(CHeader)
      kernelSourceFile.write("#include \"{}.h\"\n".format(filename))
      kernelHeaderFile.write("#pragma once\n")
      if globalParameters["RuntimeLanguage"] == "HIP":
        kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
        kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
      kernelHeaderFile.write("#include \"KernelHeader.h\"\n\n")

      for err,src,header,kernelName in kernelList:
        kernelSourceFile.write(src)
        kernelHeaderFile.write(header)

  sourceFilenames = [filePrefix+".cpp" for filePrefix in filesToWrite]

  return sourceFilenames

################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
@timing
def writeSolutionsAndKernels(outputPath, CxxCompiler, problemTypes, solutions, kernels, kernelHelperObjs, \
    kernelWriterAssembly, errorTolerant=False):

  codeObjectFiles = []

  # Push working path into build_tmp folder because there may be more than
  # one process running this script. This is to avoid build directory clashing.
  # NOTE: file paths must not contain the lower case word 'kernel' or the
  # /opt/rocm/bin/extractkernel will fail.
  # See buildSourceCodeObjectFile:167 for the call to this binary.
  Common.pushWorkingPath('build_tmp')
  Common.pushWorkingPath(os.path.basename(outputPath).upper())

  print1("# Writing Kernels...")
  kernelFiles = []
  kernelSourceFile = None
  kernelHeaderFile = None

  if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1 or globalParameters["LazyLibraryLoading"]:
    ensurePath(os.path.join(outputPath, "Kernels"))

  ##############################################################################
  # Write Kernels
  ##############################################################################
  kernelsWithBuildErrs = {}

  prepAsm(kernelWriterAssembly)

  # Kernels may be intended for different co files, but generate the same .o file
  # Mark duplicate kernels to avoid race condition
  # @TODO improve organization so this problem doesn't appear
  objFilenames = set()
  for kernel in kernels:
    if kernel["KernelLanguage"] == "Assembly":
      base = kernelWriterAssembly.getKernelFileBase(kernel)
      if base in objFilenames:
        kernel.duplicate = True
      else:
        objFilenames.add(base)
        kernel.duplicate = False

  kIter   = zip(kernels, itertools.repeat(kernelWriterAssembly), itertools.repeat(TensileInstructions()))
  results = Common.ParallelMap2(processKernelSource, kIter, "Generating kernels")

  removeKernels = []
  removeKernelNames = []
  removeSolutions = []
  removeResults = []
  for kernIdx, res in Utils.tqdm(enumerate(results)):
    (err,src,header,kernelName, filename) = res
    if(err == -2):
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
  for solution in Utils.tqdm(solutions, "Finding invalid solutions"):
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

  kernelFiles += buildKernelSourceAndHeaderFiles(results, outputPath, kernelsWithBuildErrs)

  kernelsToBuild = kernels
  if errorTolerant:
      def success(kernel):
          writer = kernelWriterAssembly
          kernelName = writer.getKernelName(kernel)
          return kernelName not in kernelsWithBuildErrs
      kernelsToBuild = filter(success, kernelsToBuild)
  elif len(kernelsWithBuildErrs) > 0:
    print("\nKernel compilation failed in one or more subprocesses. May want to set CpuThreads=0 and re-run to make debug easier")
    printExit("** kernel compilation failure **")

  # Put all kernel helper objects into the first merged kernel file
  if globalParameters["NumMergedFiles"] > 1 and len(kernelFiles) > 0:
    kernelFilename = kernelFiles[0].replace(".cpp", "")
    kernelSourceFile = open(kernelFilename + ".cpp", 'a', encoding="utf-8")
    kernelHeaderFile = open(kernelFilename + ".h", 'a', encoding="utf-8")
  elif globalParameters["MergeFiles"] or globalParameters["LazyLibraryLoading"]:
    kernelSourceFilename = os.path.join(os.path.normcase(outputPath), "Kernels.cpp")
    kernelHeaderFilename = os.path.join(os.path.normcase(outputPath), "Kernels.h")
    kernelSourceFile = open(kernelSourceFilename, "a", encoding="utf-8")
    kernelHeaderFile = open(kernelHeaderFilename, "a", encoding="utf-8")

  HeaderText = ""
  # handle helper kernel function
  for ko in kernelHelperObjs:
    kernelName = ko.getKernelName()

    # write kernel.cpp
    if not globalParameters["MergeFiles"]:
      kernelSourceFilename = os.path.join(outputPath, "Kernels", kernelName+".cpp")
      kernelSourceFile = open(kernelSourceFilename, "w")
      kernelSourceFile.write(CHeader)
      kernelFiles.append(kernelSourceFilename)

    (err, src) = ko.getSourceFileString()
    kernelSourceFile.write(src)
    if err:
      print("*** warning: invalid kernel#%u"%kernelName)

    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()

    # write kernel.h
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile = open(os.path.join(os.path.normcase(outputPath), "Kernels", kernelName + ".h"), "w")
      kernelHeaderFile.write(CHeader)
      kernelHeaderFile.write(ko.getHeaderFileString())
    else:
      HeaderText += ko.getHeaderFileString()

    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()

  # write kernel.h in one shot
  if globalParameters["MergeFiles"]:
    kernelHeaderFile.write(HeaderText)

  # close merged
  if globalParameters["MergeFiles"]:
    if kernelSourceFile:
      kernelSourceFile.close()
    if kernelHeaderFile:
      kernelHeaderFile.close()

  if not globalParameters["GenerateSourcesAndExit"]:
    codeObjectFiles += buildSourceCodeObjectFiles(CxxCompiler, kernelFiles, outputPath)
    codeObjectFiles += getAssemblyCodeObjectFiles(kernelsToBuild, kernelWriterAssembly, outputPath)

  Common.popWorkingPath() # build_tmp
  Common.popWorkingPath() # workingDir

  return codeObjectFiles

def writeSolutionAndExactTable(scheduleName, deviceNames, schedProbName, problemType, \
                               solutionsForSchedule, solutionNames, exactLogic):
  s = ""
  s += "namespace { // Start schedule '%s'\n" % scheduleName

  s += "// solution table - function, name, assertion requirements\n"
  s += "static const SolutionInfo solutionTable_%s[] = {\n" % (schedProbName)
  for i in range(0, len(solutionsForSchedule)):
    solution = solutionsForSchedule[i]
    solutionName = solutionNames[i]
    s += "  {(void*)%s, \"%s\", {%d, %d, %d, %d, %d} }%s // %d" % \
      (solutionName, solutionName, \
        solution["AssertSummationElementMultiple"], \
        solution["AssertFree0ElementMultiple"], \
        solution["AssertFree1ElementMultiple"], \
        False, \
        "," if i < len(solutionsForSchedule)-1 else "", \
        i)
    s += "\n"

  s += "};\n\n"

  # Write the exact problems here
  s += "// table of exact problem dims and selected solutionIdx\n"
  s += "static const std::pair<const ProblemKey_%s, int> embeddedExactTable_%s[] = {\n" % (problemType,schedProbName)
  numSizes = problemType["TotalIndices"]
  for ruleIdx in range(0, len(exactLogic)):
    rule = exactLogic[ruleIdx]
    problemSize = rule[0][:numSizes]
    solutionIdx = rule[1][0]
    solutionGFlops = rule[1][1]
    s += " { {"
    for i in range(0, len(problemSize)):
      if i == 0:
        s += "%u" % problemSize[i]
      else:
        s += ", %u" % problemSize[i]
    s += "}, %u}" % (solutionIdx)
    s += "," if ruleIdx != len(exactLogic)-1 else " "
    s += " // %.0f GFlop/s" % (solutionGFlops)
    s += "\n"
  s += "};\n\n"

  # Create a solution mapper and init with the table above:
  s += "// The solution master constructor here adds device to the master solution mapper\n"
  s += "// The entrypoint to find a solution for this problem is through the master solution master\n"
  s += "static SolutionMapper_%s solutionMapper_%s(\n" % (problemType, schedProbName)
  s += "  \"%s\", // schedule+problem name\n" % (schedProbName)
  s += "  solutionTable_%s, %u,\n" % (schedProbName, len(solutionsForSchedule))
  s += "  embeddedExactTable_%s, %u,\n" % (schedProbName, len(exactLogic))
  s += "  &problemType_%s);\n" % (problemType)

  s += "} // end anonymous namespace\n"
  return s


################################################################################
# Write Range Logic Recursive
# ptr :
#   True : write logic to return the function pointer
#   False : write logic to return the function name
################################################################################
def writeExactLogic(problemType, indexOrder,
                    solutionsForSchedule, exactLogic, \
                    solutionNames, ptr):
  s = ""
  s += "  ProblemDims_%s pdims(" % problemType
  indexChars = globalParameters["IndexChars"]
  firstStrideAB = 0 if problemType["UseInitialStridesAB"] else 1
  firstStrideCD = 0 if problemType["UseInitialStridesCD"] else 1
  lastStrideD = problemType["NumIndicesC"]
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStrideCD,lastStrideD):
    if i != firstStrideCD: s += ", "
    s += "strideD%u%s" % (i, indexChars[i])
  for i in range(firstStrideCD,lastStrideC):
    s += ", strideC%u%s" % (i, indexChars[i])
  for i in range(firstStrideAB,lastStrideA):
    s += ", strideA%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsA"][i]])
  for i in range(firstStrideAB,lastStrideB):
    s += ", strideB%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsB"][i]])
  for i in range(0,len(indexOrder)):
    s += ", size%s" % indexChars[i]
  s += ");\n"

  s += "  auto solutionMapper = reinterpret_cast<SolutionMapper_%s *> (masterSolutionMapper_%s.mapper());\n"  \
      % (problemType, problemType)
  if ptr:
    s += "  return solutionMapper->getSolutionWithFallback(pdims,&masterSolutionMapper_%s);\n" % problemType
  else:
    s += "  return solutionMapper->getSolutionWithFallback(pdims,&masterSolutionMapper_%s)->_info->_name;\n" % problemType

  return s


################################################################################
# Write Solution Call
################################################################################
def writeSolutionCall(solutionName, problemType):
  indexChars = globalParameters["IndexChars"]
  s = ""
  s += "%s(" % solutionName
  # solution parameters
  s += " dataD, dataC, dataA, dataB, alpha"
  if problemType["UseBeta"]:
    s += ", beta"
  s += ", offsetC, offsetA, offsetB"
  firstStrideAB = firstStrideCD = 1
  if problemType["UseInitialStridesAB"]:
    firstStrideAB = 0
  if problemType["UseInitialStridesCD"]:
    firstStrideCD = 0
  lastStrideD = problemType["NumIndicesC"]
  lastStrideC = problemType["NumIndicesC"]
  lastStrideA = len(problemType["IndexAssignmentsA"])
  lastStrideB = len(problemType["IndexAssignmentsB"])
  for i in range(firstStrideCD,lastStrideD):
    s += ", strideD%u%s" % (i, indexChars[i])
  for i in range(firstStrideCD,lastStrideC):
    s += ", strideC%u%s" % (i, indexChars[i])
  for i in range(firstStrideAB,lastStrideA):
    s += ", strideA%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsA"][i]])
  for i in range(firstStrideAB,lastStrideB):
    s += ", strideB%u%s" % (i, \
        indexChars[problemType["IndexAssignmentsB"][i]])
  for i in range(0, problemType["TotalIndices"]):
    s += ", size%s" % indexChars[i]
  s += ", stream, numInputEvents, inputEvents, outputEvent )"
  return s

##############################################################################
# Min Naming / Solution and Kernel Writers
##############################################################################
@timing
def getSolutionAndKernelWriters(solutions, kernels):

  # if any kernels are assembly, append every ISA supported
  kernelSerialNaming   = Solution.getSerialNaming(kernels)

  solutionMinNaming    = Solution.getMinNaming(solutions)
  kernelMinNaming      = Solution.getMinNaming(kernels)
  kernelWriterAssembly = KernelWriterAssembly(kernelMinNaming, kernelSerialNaming)

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

@timing
def buildObjectFileNames(kernelWriterAssembly, kernels, kernelHelperObjs):

  # Build lists of output object names
  sourceKernelNames = []
  asmKernelNames = []
  kernelHelperObjNames = []

  solutionFiles = []
  sourceKernelFiles = []
  asmKernelFiles = []
  sourceLibFiles = []
  asmLibFiles = []

  asmKernels = (k for k in kernels if k['KernelLanguage'] == 'Assembly')

  # Build a list of kernel object names.
  # Asm based kernels target the configured ISA
  asmArchs = collections.defaultdict(list)
  for kernel in asmKernels:
    kernelName = kernelWriterAssembly.getKernelFileBase(kernel)
    asmKernelNames.append(kernelName)
    asmArchs[kernelName].append(getGfxName(kernel['ISA']))

  kernelHelperObjNames = [ko.getKernelName() for ko in kernelHelperObjs]

  CxxCompiler = globalParameters["CxxCompiler"]

  # Source based kernels are built for all supported architectures
  if supportedCompiler(CxxCompiler):
    sourceArchs, _ = splitArchs()
  else:
    raise RuntimeError("Unknown compiler %s" % CxxCompiler)

  # Build a list of source files
  if not globalParameters["MergeFiles"]:
    for kernelName in (sourceKernelNames + asmKernelNames + kernelHelperObjNames):
      sourceKernelFiles += [
        "%s.h"   % (kernelName),
        "%s.cpp" % (kernelName)]
  elif globalParameters["NumMergedFiles"] > 1:
    for kernelIndex in range(0, globalParameters["NumMergedFiles"]):
      sourceKernelFiles += [
        "Kernels%s.h"   % str(kernelIndex),
        "Kernels%s.cpp" % str(kernelIndex)]
    for kernelName in (kernelHelperObjNames):
      sourceKernelFiles += [
        "%s.h"   % (kernelName),
        "%s.cpp" % (kernelName)]
  else:
    sourceKernelFiles += ["Kernels.h", "Kernels.cpp"]

  # Build a list of assembly files
  for asmKernelName in asmKernelNames:
      asmKernelFiles += [
        "%s.s"  % (asmKernelName),
        "%s.o"  % (asmKernelName),
        "%s.co" % (asmKernelName)]

  # Build a list of lib names from source
  if not globalParameters["MergeFiles"]:

    allSources = sourceKernelNames + kernelHelperObjNames

    for kernelName in (allSources):
      if supportedCompiler(CxxCompiler):
        sourceLibFiles += ["%s.so-000-%s.hsaco" % (kernelName, arch) for arch in sourceArchs]
      else:
        raise RuntimeError("Unknown compiler {}".format(CxxCompiler))
  elif globalParameters["NumMergedFiles"] > 1:
    if supportedCompiler(CxxCompiler):
      for kernelIndex in range(0, globalParameters["NumMergedFiles"]):
        sourceLibFiles += ["Kernels%d.so-000-%s.hsaco" % (kernelIndex, arch) for arch in sourceArchs]
    else:
      raise RuntimeError("Unknown compiler {}".format(CxxCompiler))
  elif globalParameters["LazyLibraryLoading"]:
    fallbackLibs = list(set([kernel._state["codeObjectFile"] for kernel in kernels if "fallback" in kernel._state.get('codeObjectFile', "")]))
    sourceLibFiles += ["{0}_{1}.hsaco".format(name, arch) for name, arch in itertools.product(fallbackLibs, sourceArchs)]
    if supportedCompiler(CxxCompiler):
      sourceLibFiles += ["Kernels.so-000-%s.hsaco" % (arch) for arch in sourceArchs]
  else: # Merge
    if supportedCompiler(CxxCompiler):
      sourceLibFiles += ["Kernels.so-000-%s.hsaco" % (arch) for arch in sourceArchs]
    else:
      raise RuntimeError("Unknown compiler {}".format(CxxCompiler))

  # Returns names for all xnack versions
  def addxnack(name, ext):
    arch = re.search(r"gfx.*$", name).group()
    if arch in sourceArchs:
      return [name+ext]
    else:
      return [name+xnack[len(arch):]+ext for xnack in sourceArchs if arch in xnack]

  # Build a list of asm lib names
  if globalParameters["LazyLibraryLoading"]:

    # If assembly kernel with codeObjectFile specified
    cond = lambda k : "codeObjectFile" in k._state                      \
                       and "fallback" not in k._state["codeObjectFile"] \
                       and k._state['KernelLanguage'] == "Assembly"


    asmLibFiles += list(set([kernel._state["codeObjectFile"]+".co" for kernel in kernels if cond(kernel)]))

    # If architecture specific source kernel with codeObjectFile specified
    cond = lambda k : "codeObjectFile" in k._state                     \
                      and "fallback" not in k._state["codeObjectFile"] \
                      and k._state['KernelLanguage'] == "Source"

    sourceLibFiles += list(set(itertools.chain.from_iterable(
                          [addxnack(kernel._state["codeObjectFile"], ".hsaco") for kernel in kernels if cond(kernel)]
                      )))

  elif globalParameters["MergeFiles"]:
    # Find all unique arch values for current asm kernels
    uniqueArchs = set(itertools.chain(*asmArchs.values()))
    asmLibFiles += ["TensileLibrary_%s.co" % (arch) for arch in uniqueArchs]

  else:
    for asmKernelName, archs in asmArchs.items():
      asmLibFiles += ["%s_%s.co" % (asmKernelName, str(arch)) for arch in archs]

  return (solutionFiles, sourceKernelFiles, asmKernelFiles, sourceLibFiles, asmLibFiles)

@timing
def buildObjectFilePaths(prefixDir, solutionFiles, sourceKernelFiles, asmKernelFiles, sourceLibFiles, asmLibFiles, masterLibraries):
  solutionPaths = []
  sourceKernelPaths = []
  asmKernelPaths = []
  sourceLibPaths = []
  asmLibPaths = []
  libMetadataPaths = []

  # Build full paths for source kernel files
  sourceKernelDir = ""
  if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1:
    sourceKernelDir = os.path.join(prefixDir, "Kernels")
  else:
    sourceKernelDir = prefixDir

  for sourceKernelFile in sourceKernelFiles:
    sourceKernelPaths += [ os.path.join(sourceKernelDir, sourceKernelFile) ]

  # Build full paths for asm kernel files
  asmKernelDir = os.path.join(prefixDir, "assembly")

  for asmKernelFile in asmKernelFiles:
    asmKernelPaths += [ os.path.join(asmKernelDir, asmKernelFile) ]

  # Build full paths for source and asm library files
  libDir = os.path.join(prefixDir, "library")

  libraryExt = ".yaml" if globalParameters["LibraryFormat"] == "yaml" else ".dat"
  if not globalParameters["SeparateArchitectures"] and not globalParameters["LazyLibraryLoading"]:
    libMetadataPaths = [ os.path.join(libDir, "TensileLibrary"+libraryExt) ]

  for sourceLibFile in sourceLibFiles:
    sourceLibPaths += [ os.path.join(libDir, sourceLibFile) ]

  #Use set because of duplicate fallback libraries
  newMetadataPaths = set()
  for arch, lib in masterLibraries.items():
    if globalParameters["LazyLibraryLoading"]:
      newMetadataPaths.add(os.path.join(libDir, "TensileLibrary_lazy_"+arch+libraryExt))
    else:
      newMetadataPaths.add(os.path.join(libDir, "TensileLibrary_"+arch+libraryExt))
    for name, placeholder in lib.lazyLibraries.items():
      newMetadataPaths.add(os.path.join(libDir, name+libraryExt))

  libMetadataPaths += list(newMetadataPaths)

  for asmLibFile in asmLibFiles:
    # Asm lib files are enumerated in the form of
    # KernelName_gfxXXXXX.co
    # Strip the gfxXXXX portion and use that as a subdirectory
    asmLibFileNoExt = str(os.path.splitext(asmLibFile)[0])
    asmArch = asmLibFileNoExt[asmLibFileNoExt.find("_gfx"):]
    if globalParameters["PackageLibrary"]:

      # asmArch contains _gfxXXXX. Don't use the underscore in new path
      asmLibPaths += [ os.path.join(
        libDir, asmArch[1:], asmLibFile.replace(asmArch, ''))]
    else:
      asmLibPaths += [ os.path.join(libDir, asmLibFile) ]

  return (solutionPaths, sourceKernelPaths, asmKernelPaths, sourceLibPaths, asmLibPaths, libMetadataPaths)

################################################################################
# Write CMake
################################################################################
@timing
def writeCMake(outputPath, solutionFiles, kernelFiles, libraryStaticFiles, masterLibraries):
  print1("# Writing Custom CMake")

  # Build output file paths, using relative CMake symbol
  cmakeSrcDir = "${CMAKE_SOURCE_DIR}"
  (solutionPaths, sourceKernelPaths, asmKernelPaths, sourceLibPaths, asmLibPaths, _) = \
    buildObjectFilePaths(cmakeSrcDir, solutionFiles, kernelFiles, [], [], [], masterLibraries)

  # Build full paths the static library files
  staticFilePaths = []
  for staticFile in libraryStaticFiles:
    staticFilePaths += [ os.path.join(cmakeSrcDir, staticFile) ]

  # Proceed to generate cmake file
  generatedFile = open(os.path.join(os.path.normcase(outputPath), "Generated.cmake"), "w")
  generatedFile.write(CMakeHeader)

  # write TensileClient_KERNELS symbol
  generatedFile.write("set( TensileClient_KERNELS\n")
  for kernelFile in sourceKernelPaths:
    generatedFile.write("  %s\n" % (kernelFile))
  generatedFile.write("  )\n")

  # write TensileClient_SOURCE symbol
  generatedFile.write("set( TensileClient_SOURCE\n")
  for fileName in libraryStaticFiles:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/%s\n" % fileName)
  generatedFile.write("  )\n\n")

  generatedFile.close()

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
def generateLogicDataAndSolutions(logicFiles, args):

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
  fIter = zip(logicFiles, itertools.repeat(archs))

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

    if globalParameters["PackageLibrary"]:
      if architectureName in masterLibraries:
        masterLibraries[architectureName].merge(newLibrary)
      else:
        masterLibraries[architectureName] = newLibrary
        masterLibraries[architectureName].version = args.version
    elif globalParameters["SeparateArchitectures"] or globalParameters["LazyLibraryLoading"]:
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

################################################################################
# Write Benchmark Client Files
################################################################################
def writeBenchmarkClientFiles(libraryWorkingPath, tensileSourcePath, solutions, cxxCompiler):

  if not globalParameters["GenerateSourcesAndExit"]:
      copyStaticFiles(libraryWorkingPath)

  kernels, kernelsBetaOnly, _ = generateKernelObjectsFromSolutions(solutions)
  kernelWriterAssembly, \
    kernelMinNaming, _ = getSolutionAndKernelWriters(solutions, kernels)

  # write solution, kernels and CMake
  problemType = solutions[0]["ProblemType"]
  codeObjectFiles = writeSolutionsAndKernels( \
    libraryWorkingPath, cxxCompiler, [problemType], solutions, kernels, kernelsBetaOnly, \
    kernelWriterAssembly, errorTolerant=True )

  newLibraryDir = ensurePath(os.path.join(libraryWorkingPath, 'library'))
  newLibraryFile = os.path.join(newLibraryDir, "TensileLibrary.yaml")
  newLibrary = MasterSolutionLibrary.BenchmarkingLibrary(solutions)
  newLibrary.applyNaming(kernelMinNaming)

  LibraryIO.writeYAML(newLibraryFile, Utils.state(newLibrary))

  return (codeObjectFiles, newLibrary)

def WriteClientLibraryFromSolutions(solutionList, libraryWorkingPath, tensileSourcePath = None):

  if tensileSourcePath == None:
    tensileSourcePath = os.path.dirname(os.path.realpath(__file__))
  firstSolution = solutionList[0]
  problemType = firstSolution["ProblemType"].state
  problemType["DataType"] = problemType["DataType"].value
  problemType["DataTypeA"] = problemType["DataTypeA"].value
  problemType["DataTypeB"] = problemType["DataTypeB"].value
  problemType["DataTypeE"] = problemType["DataTypeE"].value
  problemType["DataTypeAmaxD"] = problemType["DataTypeAmaxD"].value
  problemType["DestDataType"] = problemType["DestDataType"].value
  problemType["ComputeDataType"] = problemType["ComputeDataType"].value
  problemType["F32XdlMathOp"] = problemType["F32XdlMathOp"].value
  if "DataTypeMetadata" in problemType:
    problemType["DataTypeMetadata"] = problemType["DataTypeMetadata"].value
  cxxCompiler = globalParameters["CxxCompiler"]

  effectiveWorkingPath = os.path.join(libraryWorkingPath, "library")
  ensurePath(effectiveWorkingPath)
  mataDataFilePath = os.path.join(effectiveWorkingPath, 'metadata.yaml')

  metaData = {"ProblemType":problemType}
  LibraryIO.writeYAML(mataDataFilePath, metaData)

  codeObjectFiles, newLibrary = writeBenchmarkClientFiles(libraryWorkingPath, tensileSourcePath, solutionList, cxxCompiler )

  return (codeObjectFiles, newLibrary)

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
@timing
def TensileCreateLibrary():
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
  argParser.add_argument("--cxx-compiler",           dest="CxxCompiler",       choices=["hipcc", "amdclang++"], action="store", default="amdclang++")
  argParser.add_argument("--cmake-cxx-compiler",     dest="CmakeCxxCompiler",  action="store")
  argParser.add_argument("--code-object-version",    dest="CodeObjectVersion", choices=["default", "V4", "V5"], action="store")
  argParser.add_argument("--architecture",           dest="Architecture",      type=str, action="store", default="all", help="Supported archs: " + " ".join(architectureMap.keys()))
  argParser.add_argument("--merge-files",            dest="MergeFiles",        action="store_true")
  argParser.add_argument("--no-merge-files",         dest="MergeFiles",        action="store_false")
  argParser.add_argument("--num-merged-files",       dest="NumMergedFiles",    type=int, default=1, help="Number of files the kernels should be written into.")
  argParser.add_argument("--short-file-names",       dest="ShortNames",        action="store_true")
  argParser.add_argument("--no-short-file-names",    dest="ShortNames",        action="store_false")
  argParser.add_argument("--library-print-debug",    dest="LibraryPrintDebug", action="store_true")
  argParser.add_argument("--no-library-print-debug", dest="LibraryPrintDebug", action="store_false")
  argParser.add_argument("--no-enumerate",           action="store_true", help="Do not run rocm_agent_enumerator.")
  argParser.add_argument("--package-library",        dest="PackageLibrary",    action="store_true", default=False)
  argParser.add_argument("--embed-library",          dest="EmbedLibrary",
                         help="Embed (new) library files into static variables.  Specify the name of the library.")

  argParser.add_argument("--embed-library-key",      dest="EmbedLibraryKey", default=None,
                         help="Access key for embedding library files.")
  argParser.add_argument("--version", help="Version string to embed into library file.")
  argParser.add_argument("--generate-manifest-and-exit",   dest="GenerateManifestAndExit", action="store_true",
                          default=False, help="Output manifest file with list of expected library objects and exit.")
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
  argParser.add_argument("--build-client", dest="BuildClient", action="store_true",
                         help="Build Tensile client")
  argParser.add_argument("--client-config", dest="ClientConfig", action="store_true",
                         help="Create client config for setting the library and code object files")
  argParser.add_argument("--global-parameters", nargs="+", type=splitExtraParameters, default=[])
  argParser.add_argument("--no-generate-solution-table", dest="GenSolTable", action="store_false", default=True,
                         help="Skip generating solution-yaml matching table")
  argParser.add_argument("--asm-debug", dest="AsmDebug", action="store_true", default=False,
                         help="Keep debug information for built code objects")
  argParser.add_argument("--build-id", dest="BuildIdKind", action="store", default="sha1")
  argParser.add_argument("--keep-build-tmp", dest="KeepBuildTmp", action="store_true",
                          default=False, help="Do not remove the temporary build directory (may required hundreds of GBs of space)"),
  argParser.add_argument("--validate-library", dest="ValidateLibrary", action="store_true", default=False)
  argParser.add_argument("--logic-filter", dest="LogicFilter", action="store", default="*", type=str,
                        help="Cutomsized logic filter, default is *, i.e. all logics."
                        " Example: gfx942/Equality/* for building equality of gfx942 only")

  args = argParser.parse_args()

  logicPath = args.LogicPath
  outputPath = args.OutputPath
  CxxCompiler = args.CxxCompiler
  libraryFormat = args.LibraryFormat
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
  arguments["CxxCompiler"] = args.CxxCompiler
  if args.CmakeCxxCompiler:
    os.environ["CMAKE_CXX_COMPILER"] = args.CmakeCxxCompiler
  arguments["MergeFiles"] = args.MergeFiles
  arguments["NumMergedFiles"] = args.NumMergedFiles
  arguments["ShortNames"] = args.ShortNames
  arguments["LibraryPrintDebug"] = args.LibraryPrintDebug
  arguments["CodeFromFiles"] = False
  arguments["EmbedLibrary"] = args.EmbedLibrary
  arguments["LogicFormat"]  = args.LogicFormat
  arguments["LibraryFormat"] = args.LibraryFormat
  if args.no_enumerate:
    arguments["ROCmAgentEnumeratorPath"] = False
  arguments["PackageLibrary"] = args.PackageLibrary

  arguments["GenerateManifestAndExit"] = args.GenerateManifestAndExit

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
  arguments["ValidateLibrary"] = args.ValidateLibrary

  for key, value in args.global_parameters:
    arguments[key] = value

  assignGlobalParameters(arguments)

  print1("# CodeObjectVersion from TensileCreateLibrary: %s" % arguments["CodeObjectVersion"])
  print1("# CxxCompiler       from TensileCreateLibrary: %s" % CxxCompiler)
  print1("# Architecture      from TensileCreateLibrary: %s" % arguments["Architecture"])
  print1("# LibraryFormat     from TensileCreateLibrary: %s" % libraryFormat)

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

  if globalParameters["LazyLibraryLoading"] and not (globalParameters["MergeFiles"] and globalParameters["SeparateArchitectures"]):
    printExit("--lazy-library-loading requires --merge-files and --separate-architectures enabled")

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
  print1(f"# LogicFilter: {globPattern}")
  logicFiles = (os.path.join(logicPath, file) for file in glob.iglob(globPattern, recursive=True))
  logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]

  print1(f"# LibraryLogicFiles({len(logicFiles)}):")
  for logicFile in logicFiles:
    print1("#   %s" % logicFile)

  ##############################################################################
  # Parse config files
  ##############################################################################

  # Parse logicData, solutions, and masterLibraries from logic files
  solutions, masterLibraries, fullMasterLibrary = generateLogicDataAndSolutions(logicFiles, args)

  kernels, kernelHelperObjs, _ = generateKernelObjectsFromSolutions(solutions)

  # if any kernels are assembly, append every ISA supported
  kernelWriterAssembly, kernelMinNaming, _ = getSolutionAndKernelWriters(solutions, kernels)

  if globalParameters["ValidateLibrary"]:
    validateLibrary(masterLibraries, kernels, kernelWriterAssembly)

  staticFiles = copyStaticFiles(outputPath)

  # Build a list of files to be expected
  (solutionFiles,
   sourceKernelFiles,
   asmKernelFiles,
   sourceLibFiles,
   asmLibFiles) = buildObjectFileNames(kernelWriterAssembly, kernels, kernelHelperObjs)

  (_,
   _,
   _,
   sourceLibPaths,
   asmLibPaths,
   libMetadataPaths) = buildObjectFilePaths(outputPath, solutionFiles, sourceKernelFiles, \
    asmKernelFiles, sourceLibFiles, asmLibFiles, masterLibraries)

  # Generate manifest file
  libraryPath = os.path.join(outputPath, "library")
  ensurePath(libraryPath)
  generatedFile = open(os.path.join(libraryPath, "TensileManifest.txt"), "w")

  # Manifest file contains YAML file, output library paths and cpp source for embedding.
  for filePath in libMetadataPaths + sourceLibPaths + asmLibPaths:
    generatedFile.write("%s\n" %(filePath) )
  generatedFile.close()

  if globalParameters["GenerateManifestAndExit"] == True:
    return

  # generate cmake for the source kernels,
  if not arguments["GenerateSourcesAndExit"]:
    writeCMake(outputPath, solutionFiles, sourceKernelFiles, staticFiles, masterLibraries)

  # Make sure to copy the library static files.
  for fileName in staticFiles:
    shutil.copy( os.path.join(globalParameters["SourcePath"], fileName), \
      outputPath )

  # write solutions and kernels
  codeObjectFiles = writeSolutionsAndKernels(outputPath, CxxCompiler, None, solutions,
                                             kernels, kernelHelperObjs, kernelWriterAssembly)

  bothLibSet = set(sourceLibPaths + asmLibPaths)
  setA = set( map( os.path.normcase, set(codeObjectFiles) ) )
  setB = set( map( os.path.normcase, bothLibSet ) )

  sanityCheck0 = setA - setB
  sanityCheck1 = setB - setA

  if globalParameters["PrintCodeCommands"]:
    print("codeObjectFiles:", codeObjectFiles)
    print("sourceLibPaths + asmLibPaths:", sourceLibPaths + asmLibPaths)

  assert len(sanityCheck0) == 0, "Unexpected code object files: {}".format(sanityCheck0)
  if not globalParameters["GenerateSourcesAndExit"]:
    assert len(sanityCheck1) == 0, "Missing expected code object files: {}".format(sanityCheck1)

  archs = [getGfxName(arch) for arch in globalParameters['SupportedISA'] \
             if globalParameters["AsmCaps"][arch]["SupportedISA"]]
  newLibraryDir = ensurePath(os.path.join(outputPath, 'library'))

  if globalParameters["PackageLibrary"]:
    for archName, newMasterLibrary in masterLibraries.items():
      if (archName in archs):
        archPath = ensurePath(os.path.join(newLibraryDir, archName))
        masterFile = os.path.join(archPath, "TensileLibrary")
        newMasterLibrary.applyNaming(kernelMinNaming)
        LibraryIO.write(masterFile, Utils.state(newMasterLibrary), args.LibraryFormat)
  elif globalParameters["SeparateArchitectures"] or globalParameters["LazyLibraryLoading"]:
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
  if globalParameters["PackageLibrary"] or globalParameters["SeparateArchitectures"]:
    theMasterLibrary = list(masterLibraries.values())[0]

  if args.EmbedLibrary is not None:
      embedFileName = os.path.join(outputPath, "library/{}.cpp".format(args.EmbedLibrary))
      with EmbeddedData.EmbeddedDataFile(embedFileName) as embedFile:

          ext = ".yaml" if globalParameters["LibraryFormat"] == "yaml" else ".dat"
          embedFile.embed_file(theMasterLibrary.cpp_base_class, masterFile + ext, nullTerminated=True,
                               key=args.EmbedLibraryKey)

          for co in Utils.tqdm(codeObjectFiles):
              embedFile.embed_file("SolutionAdapter", co, nullTerminated=False,
                                   key=args.EmbedLibraryKey)

  if args.BuildClient:
    print1("# Building Tensile Client")
    ClientExecutable.getClientExecutable(outputPath)

  if args.ClientConfig:
    # write simple ini for best solution mode linked to library we just made
    iniFile = os.path.join(outputPath, "best-solution.ini")
    with open(iniFile, "w") as f:
      def param(key, value):
        f.write("{}={}\n".format(key, value))

      libraryFile = masterFile + ".yaml" \
        if globalParameters["LibraryFormat"] == "yaml" else masterFile + ".dat"

      param("library-file", libraryFile)
      for coFile in codeObjectFiles:
        param("code-object", os.path.join(outputPath,coFile))

      param("best-solution", True)

  print1("# Check if generated files exists.")

  @timing
  def checkFileExistence(files):
    for filePath in files:
      if not os.path.exists(filePath):
        printExit("File %s is missing.", filePath)

  checkFileExistence(itertools.chain(libMetadataPaths, sourceLibPaths, asmLibPaths))

  if not globalParameters["KeepBuildTmp"]:
    buildTmp = Path(outputPath).parent / "library" / "build_tmp"
    if buildTmp.exists() and buildTmp.is_dir():
      shutil.rmtree(buildTmp)
    else:
      printWarning(f"Cannot remove {str(buildTmp)}")

  print1("# Tensile Library Writer DONE")
  print1(HR)
  print1("")
