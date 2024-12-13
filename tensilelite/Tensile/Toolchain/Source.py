import itertools
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Union

from ..Common import globalParameters, print2,  ensurePath, ParallelMap2, splitArchs

class SourceToolchain:
    def __init__(self, compiler: str, bundler: str, buildIdKind: str, asanBuild: bool=False, saveTemps: bool=False):
        self.compiler = compiler
        self.bundler = bundler
        self.buildIdKind = buildIdKind
        self.asanBuild = asanBuild
        self.saveTemps = saveTemps

    def compile(self, srcPath: str, destPath: str, includePath: str, gfxs: List[str]):
        """Compiles a source file into an object file.

        Args:
            cmdlineArchs: List of architectures for offloading.
            kernelFile: The path to the kernel source file.
            buildPath: The build directory path.
            objectFilename: The name of the output object file.
            outputPath: The output directory path.
            globalParameters: A dictionary of global parameters.

        Raises:
            RuntimeError: If the compilation command fails.
        """
        launcher = shlex.split(os.environ.get("Tensile_CXX_COMPILER_LAUNCHER", ""))

        hipFlags = [
            "-D__HIP_HCC_COMPAT_MODE__=1",
            "--cuda-device-only",
            "-x", "hip", "-O3",    
            "-I", includePath,
            "-Xoffload-linker", f"--build-id={self.buildIdKind}",
            "-std=c++17",
        ]
        if self.asanBuild:
            hipFlags.extend(["-fsanitize=address", "-shared-libasan", "-fuse-ld=lld"])
        if self.saveTemps:
            hipFlags.append("--save-temps")
        if os.name == "nt":
            hipFlags.extend(["-fms-extensions", "-fms-compatibility", "-fPIC", "-Wno-deprecated-declarations"])

        archFlags = [f"--offload-arch={gfx}" for gfx in gfxs]

        args = [
            *launcher, self.compiler, *hipFlags, *archFlags, srcPath, "-c", "-o", destPath
        ]
        try:
            out = subprocess.check_output(args, stderr=subprocess.STDOUT)
            print2(f"Output: {out}" if out else "")
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Error compiling source object file: {err.output}\nFailed command: {' '.join(args)}")

    def targets(self, objFile: str):
        """Lists the target triples in an object file.

        Args:
            objFile: The object file path.

        Returns:
            List of target triples in the object file.
        """
        args = [self.bundler, "--type=o", f"--input={objFile}", "-list"]
        try:
            listing = subprocess.check_output(args, stderr=subprocess.STDOUT).decode().split("\n")
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Error listing target triples in object files: {err.output}\nFailed command: {' '.join(args)}")
        return listing

    def unbundle(self, target: str, srcPath: str, destPath: str):
        """Unbundles source code object files using the Clang Offload Bundler.

        Args:
            target: The target triple, see https://llvm.org/docs/AMDGPUUsage.html#target-triples.
            infile: The path to the input object file.
            outfileRaw: The path to the unbundled code object.

        Raises:
            RuntimeError: If unbundling the source code object file fails.
        """
        args = [
            self.bundler,
            "--type=o",
            f"--targets={target}",
            f"--input={srcPath}",
            f"--output={destPath}",
            "--unbundle",
        ]

        print2("Unbundling source code object file: " + " ".join(args))
        try:
            out = subprocess.check_output(args, stderr=subprocess.STDOUT)
            print2(f"Output: {out}" if out else "")
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Error unbundling source code object file: {err.output}\nFailed command: {' '.join(args)}")
            

def _computeSourceCodeObjectFilename(target: str, base: str, buildPath: Union[Path, str], arch: str) -> Union[Path, None]:
    """Generates a code object file path using the target, base, and build path.

    Args:
        target: The target triple.
        base: The base name for the output file (name without extension).
        buildPath: The build directory path.

    Returns:
        Path to the code object file.
    """
    coPath = None
    buildPath = Path(buildPath)
    if "TensileLibrary" in base and "fallback" in base:
        coPath = buildPath / "{0}_{1}.hsaco.raw".format(base, arch)
    elif "TensileLibrary" in base:
        variant = [t for t in ["", "xnack-", "xnack+"] if t in target][-1]
        baseVariant = base + "-" + variant if variant else base
        if arch in baseVariant:
            coPath = buildPath / (baseVariant + ".hsaco.raw")
    else:
        coPath= buildPath / "{0}.so-000-{1}.hsaco.raw".format(base, arch)

    return coPath


def _buildSourceCodeObjectFile(toolchain: SourceToolchain, outputPath: Union[Path, str], kernelPath: Union[Path, str]) -> List[str]:
    """Compiles a HIP source code file into a code object file.

    Args:
        cxxCompiler: The C++ compiler to use.
        cxxCompiler: The offload bundler to use.
        outputPath: The output directory path where code objects will be placed.
        kernelPath: The path to the kernel source file.

    Returns:
        List of paths to the created code objects.
    """
    buildPath = Path(ensurePath(os.path.join(globalParameters['WorkingPath'], 'code_object_tmp')))
    destPath = Path(ensurePath(os.path.join(outputPath, 'library')))
    kernelPath = Path(kernelPath)

    if "CmakeCxxCompiler" in globalParameters and globalParameters["CmakeCxxCompiler"] is not None:
      os.environ["CMAKE_CXX_COMPILER"] = globalParameters["CmakeCxxCompiler"]

    objFilename = kernelPath.stem + '.o'
    coPathsRaw = []
    coPaths= []

    _, cmdlineArchs = splitArchs()

    objPath = str(buildPath / objFilename)
    toolchain.compile(str(kernelPath), objPath, str(outputPath), cmdlineArchs)

    for target in toolchain.targets(objPath):
      match = re.search("gfx.*$", target)
      if match:
        arch = re.sub(":", "-", match.group())
        coPathRaw = _computeSourceCodeObjectFilename(target, kernelPath.stem, buildPath, arch)
        if not coPathRaw: continue
        toolchain.unbundle(target, objPath, str(coPathRaw))

        coPath = str(destPath / coPathRaw.stem)
        coPathsRaw.append(coPathRaw)
        coPaths.append(coPath)

    for src, dst in zip(coPathsRaw, coPaths):
        shutil.move(src, dst)

    return coPaths

def buildSourceCodeObjectFiles(toolchain: SourceToolchain, kernelFiles: List[Path], outputPath: Path) -> Iterable[str]:
    """Compiles HIP source code files into code object files.

    Args:
        cxxCompiler: The C++ compiler to use.
        kernelFiles: List of paths to the kernel source files.
        outputPath: The output directory path where code objects will be placed.
        removeTemporaries: Whether to clean up temporary files.

    Returns:
        List of paths to the created code objects.
    """
    args    = zip(itertools.repeat(toolchain), itertools.repeat(outputPath), kernelFiles)
    coFiles = ParallelMap2(_buildSourceCodeObjectFile, args, "Compiling source kernels")
    return itertools.chain.from_iterable(coFiles)
