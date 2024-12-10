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

class AssemblyToolchain:
    def __init__(self, assembler: str, bundler: str, buildIdKind: str):
        self.assembler = assembler
        self.bundler = bundler
        self.buildIdKind = buildIdKind

    def link(self, srcPaths: List[str], destPath: str):
        """Links object files into a code object file.

        Args:
            srcPaths: A list of paths to object files.
            destPath: A destination path for the generated code object file.

        Raises:
            RuntimeError: If linker invocation fails.
        """
        if os.name == "nt":
            # Use args file on Windows b/c the command may exceed the limit of 8191 characters
            with open(Path.cwd() / "clang_args.txt", "wt") as file:
                file.write(" ".join(objFiles))
                file.flush()
            args = [
                self.assembler,
                "--target=amdgcn-amd-amdhsa",
                "-o", destPath, "@clang_args.txt"]
        else:
            args = [
                self.assembler,
                "--target=amdgcn-amd-amdhsa",
                "-Xlinker", f"--build-id={self.buildIdKind}",
                "-o", destPath, *srcPaths
            ]
        print2(f"Linking assembly object files into code object: {' '.join(args)}")
        subprocess.check_call(args)

    def compress(self, srcPath: str, destPath: str, gfx: str):
        """Compresses a code object file using the provided bundler.

        Args:
            srcPath: The source path of the code object file to be compressed.
            destPath: The destination path for the compressed code object file.
            gfx: The target GPU architecture.

        Raises:
            RuntimeError: If compressing the code object file fails.
        """
        args = [
            self.bundler,
            "--compress",
            "--type=o",
            "--bundle-align=4096",
            f"--targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--{gfx}",
            "--input=/dev/null",
            f"--input={srcPath}",
            f"--output={destPath}",
        ]

        print2(f"Bundling/compressing assembly code object: {' '.join(args)}")
        try:
            out = subprocess.check_output(args, stderr=subprocess.STDOUT)
            print2(f"Output: {out}")
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"Error compressing code object via bundling: {err.output}\nFailed command: {' '.join(args)}"
            )


def _batchObjectFiles(objFiles: List[str], coPathDest: Union[Path, str], maxObjFiles: int=10000) -> List[str]:
    numObjFiles = len(objFiles)
    
    if numObjFiles <= maxObjFiles:
      return objFiles

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

    return newObjFilesOutput

def buildAssemblyCodeObjectFiles(toolchain: AssemblyToolchain, kernels, kernelWriterAssembly, outputPath, compress: bool=True):
    
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

          objFiles = _batchObjectFiles(objFiles, coFileRaw)
          toolchain.link(objFiles, str(coFileRaw))

          coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
          if compress:
            toolchain.compress(str(coFileRaw), str(coFile), gfx)
          else:
            shutil.move(coFileRaw, coFile)

          coFiles.append(coFile)
      else:
        # no mergefiles
        def newCoFileName(kName):
          if globalParameters["PackageLibrary"]:
            return os.path.join(destDir, gfx, kName + '.co')
          else:
            return os.path.join(destDir, kName + '_' + gfx + '.co')

        def orgCoFileName(kName):
          return os.path.join(asmDir, kName + '.co')

        for src, dst in Utils.tqdm(((orgCoFileName(kName), newCoFileName(kName)) for kName in \
                                    map(lambda k: kernelWriterAssembly.getKernelFileBase(k), archKernels)), "Copying code objects"):
          shutil.copyfile(src, dst)
          coFiles.append(dst)
        printWarning("Code object files are not compressed in `--no-merge-files` build mode.")

    return coFiles
