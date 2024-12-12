import collections
import math
import os
import shutil
import subprocess

from pathlib import Path
from typing import List, Union

from .. import Utils
from ..TensileInstructions import getGfxName
from ..Common import globalParameters, print2, ensurePath, printWarning
from ..KernelWriterAssembly import KernelWriterAssembly
from .SharedCommands import compressCodeObject

def _linkIntoCodeObject(
    objFiles: List[str], coPathDest: Union[Path, str], kernelWriterAssembly: KernelWriterAssembly, assembler: str
):
    """Links object files into a code object file.

    Args:
        objectFiles: A list of object files to be linked.
        coPathDest: The destination path for the code object file.
        kernelWriterAssembly: An instance of KernelWriterAssembly to get link arguments.

    Raises:
        RuntimeError: If linker invocation fails.
    """
    if os.name == "nt":
      # Use args file on Windows b/c the command may exceed the limit of 8191 characters
      with open(Path.cwd() / "clangArgs.txt", 'wt') as file:
        file.write(" ".join(objFiles))
        file.flush()
      args = [assembler, '-target', 'amdgcn-amd-amdhsa', '-o', coFileRaw, '@clangArgs.txt']
      subprocess.check_call(args, cwd=asmDir)
    else:
      numObjFiles = len(objFiles)
      maxObjFiles = 10000
      
      if numObjFiles > maxObjFiles:
        batchedObjFiles = [objFiles[i:i+maxObjFiles] for i in range(0, numObjFiles, maxObjFiles)]
        numBatches = int(math.ceil(numObjFiles / maxObjFiles))

        newObjFiles = [str(coPathDest) + "." + str(i) for i in range(0, numBatches)]
        newObjFilesOutput = []

        for batch, filename in zip(batchedObjFiles, newObjFiles):
          if len(batch) > 1:
            args = [globalParameters["ROCmLdPath"], "-r"] + batch + [ "-o", filename]
            print2(f"Linking object files into fewer object files: {' '.join(args)}")
            subprocess.check_call(args)
            newObjFilesOutput.append(filename)
          else:
            newObjFilesOutput.append(batchedObjFiles[0])

        objFiles = newObjFilesOutput

      args = kernelWriterAssembly.getLinkCodeObjectArgs(objFiles, str(coPathDest))
      print2(f"Linking object files into code object: {' '.join(args)}")
      subprocess.check_call(args)



def buildAssemblyCodeObjectFiles(kernels, kernelWriterAssembly, outputPath, assembler: str, offloadBundler: str, compress: bool=True):
    
    isAsm = lambda k: k["KernelLanguage"] == "Assembly"

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"

    destDir = Path(ensurePath(os.path.join(outputPath, 'library')))
    asmDir = Path(kernelWriterAssembly.getAssemblyDirectory())

    archKernelMap = collections.defaultdict(list)
    for k in filter(isAsm, kernels):
      archKernelMap[tuple(k['ISA'])].append(k)

    coFiles = []
    for arch, archKernels in archKernelMap.items():
      if len(archKernels) == 0:
        continue

      gfx = getGfxName(arch)

      if globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1 or globalParameters["LazyLibraryLoading"]:
        objectFiles = [str(asmDir / (kernelWriterAssembly.getKernelFileBase(k) + extObj)) for k in archKernels if 'codeObjectFile' not in k]

        coFileMap = collections.defaultdict(list)

        if len(objectFiles):
          coFileMap[asmDir / ("TensileLibrary_"+ gfx + extCoRaw)] = objectFiles

        for kernel in archKernels:
          coName = kernel.get("codeObjectFile", None)
          if coName:
            coFileMap[asmDir / (coName + extCoRaw)].append(str(asmDir / (kernelWriterAssembly.getKernelFileBase(kernel) + extObj)))

        for coFileRaw, objFiles in coFileMap.items():

          _linkIntoCodeObject(objFiles, coFileRaw, kernelWriterAssembly, assembler)
          coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
          if compress:
            compressCodeObject(coFileRaw, coFile, gfx, offloadBundler)
          else:
            shutil.move(coFileRaw, coFile)

          coFiles.append(coFile)
      else:
        # no mergefiles
        def newCoFileName(kName):
          return os.path.join(destDir, kName + '_' + gfx + '.co')

        def orgCoFileName(kName):
          return os.path.join(asmDir, kName + '.co')

        for src, dst in Utils.tqdm(((orgCoFileName(kName), newCoFileName(kName)) for kName in \
                                    map(lambda k: kernelWriterAssembly.getKernelFileBase(k), archKernels)), "Copying code objects"):
          shutil.copyfile(src, dst)
          coFiles.append(dst)
        printWarning("Code object files are not compressed in `--no-merge-files` build mode.")

    return coFiles
