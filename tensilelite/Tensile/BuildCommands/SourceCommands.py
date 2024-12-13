import itertools
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Union

from ..Common import globalParameters, print2,  ensurePath, supportedCompiler, ParallelMap2, splitArchs, which

def _compileSourceObjectFile(cmdlineArchs: List[str], cxxCompiler: str, cxxSrcPath: str, objDestPath: str, outputPath: str):
    """Compiles a source file into an object file.

    Args:
        cmdlineArchs: List of architectures for offloading.
        cxxCompiler: The C++ compiler to use.
        kernelFile: The path to the kernel source file.
        buildPath: The build directory path.
        objectFilename: The name of the output object file.
        outputPath: The output directory path.
        globalParameters: A dictionary of global parameters.

    Raises:
        RuntimeError: If the compilation command fails.
    """
    archFlags = ['--offload-arch=' + arch for arch in cmdlineArchs]

    #TODO(@jichangjichang) Needs to be fixed when Maneesh's change is made available
    hipFlags = ["-D__HIP_HCC_COMPAT_MODE__=1"]
    hipFlags.extend(
        ["--genco"] if cxxCompiler == "hipcc" else ["--cuda-device-only", "-x", "hip", "-O3"]
    )

    hipFlags.extend(['-I', outputPath])
    hipFlags.extend(["-Xoffload-linker", "--build-id=%s"%globalParameters["BuildIdKind"]])
    hipFlags.append('-std=c++17')
    if globalParameters["AsanBuild"]:
      hipFlags.extend(["-fsanitize=address", "-shared-libasan", "-fuse-ld=lld"])
    if globalParameters["SaveTemps"]:
      hipFlags.append('--save-temps')

    launcher = shlex.split(os.environ.get('Tensile_CXX_COMPILER_LAUNCHER', ''))

    if os.name == "nt":
      hipFlags.extend(['-fms-extensions', '-fms-compatibility', '-fPIC', '-Wno-deprecated-declarations'])

    args = launcher + [which(cxxCompiler)] + hipFlags + archFlags + [cxxSrcPath, '-c', '-o', objDestPath]

    try:
      out = subprocess.check_output(args, stderr=subprocess.STDOUT)
      print2(f"Output: {out}" if out else "")
    except subprocess.CalledProcessError as err:
      raise RuntimeError(f"Error compiling source object file: {err.output}\nFailed command: {' '.join(args)}")


def _listTargetTriples(bundler: str, objFile: str) -> List[str]:
    """Lists the target triples in an object file.

    Args:
        bundler: The path to the bundler, typically ``clang-offload-bundler``.
        objFile: The object file path.

    Returns:
        List of target triples in the object file.
    """
    args = [bundler, "--type=o", f"--input={objFile}", "-list"]
    try:
        listing = subprocess.check_output(args, stderr=subprocess.STDOUT).decode().split("\n")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Error listing target triples in object files: {err.output}\nFailed command: {' '.join(args)}")
    return listing


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


def _unbundleSourceCodeObjects(bundler: str, target: str, infile: str, outfileRaw: str):
    """Unbundles source code object files using the Clang Offload Bundler.

    Args:
        bundler: The path to the bundler, typically ``clang-offload-bundler``.
        target: The target architecture string.
        infile: The input file path.
        outfileRaw: The output raw file path.

    Raises:
        RuntimeError: If unbundling the source code object file fails.
    """
    args = [
        bundler,
        "--type=o",
        f"--targets={target}",
        f"--input={infile}",
        f"--output={outfileRaw}",
        "--unbundle",
    ]

    print2("Unbundling source code object file: " + " ".join(args))
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        print2(f"Output: {out}" if out else "")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Error unbundling source code object file: {err.output}\nFailed command: {' '.join(args)}")


def _buildSourceCodeObjectFile(cxxCompiler: str, offloadBundler: str, outputPath: Union[Path, str], kernelPath: Union[Path, str]) -> List[str]:
    """Compiles a HIP source code file into a code object file.

    Args:
        cxxCompiler: The C++ compiler to use.
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

    if not supportedCompiler(cxxCompiler):
      raise RuntimeError("Unknown compiler {}".format(cxxCompiler))

    _, cmdlineArchs = splitArchs()

    objPath = str(buildPath / objFilename)
    _compileSourceObjectFile(cmdlineArchs, cxxCompiler, str(kernelPath), objPath, str(outputPath))

    if not offloadBundler:
      raise RuntimeError("No bundler found; set TENSILE_ROCM_OFFLOAD_BUNDLER_PATH to point to clang-offload-bundler")

    for target in _listTargetTriples(offloadBundler, objPath):
      match = re.search("gfx.*$", target)
      if match:
        arch = re.sub(":", "-", match.group())
        coPathRaw = _computeSourceCodeObjectFilename(target, kernelPath.stem, buildPath, arch)
        if not coPathRaw: continue
        _unbundleSourceCodeObjects(offloadBundler, target, objPath, str(coPathRaw))

        coPath = str(destPath / coPathRaw.stem)
        coPathsRaw.append(coPathRaw)
        coPaths.append(coPath)

    for src, dst in zip(coPathsRaw, coPaths):
        shutil.move(src, dst)

    return coPaths

def buildSourceCodeObjectFiles(cxxCompiler: str, offloadBundler: str, kernelFiles: List[Path], outputPath: Path) -> Iterable[str]:
    """Compiles HIP source code files into code object files.

    Args:
        cxxCompiler: The C++ compiler to use.
        kernelFiles: List of paths to the kernel source files.
        outputPath: The output directory path where code objects will be placed.
        removeTemporaries: Whether to clean up temporary files.

    Returns:
        List of paths to the created code objects.
    """
    args    = zip(itertools.repeat(cxxCompiler), itertools.repeat(offloadBundler), itertools.repeat(outputPath), kernelFiles)
    coFiles = ParallelMap2(_buildSourceCodeObjectFile, args, "Compiling source kernels")
    return itertools.chain.from_iterable(coFiles)
