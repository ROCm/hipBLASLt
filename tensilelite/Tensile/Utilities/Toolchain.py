import os
from pathlib import Path
from typing import List, NamedTuple
from warnings import warn

ROCM_BIN_PATH = Path("/opt/rocm/bin")
ROCM_LLVM_BIN_PATH = Path("/opt/rocm/lib/llvm/bin")

def osSelect(linux: str, windows: str) -> str:
    return windows if os.name == "nt" else linux

class ToolchainDefaults(NamedTuple):
    CXX_COMPILER= osSelect(linux="amdclang++", windows="clang++.exe") #+ "__deleteme__"
    C_COMPILER= osSelect(linux="amdclang", windows="clang.exe") #+ "__deleteme__"
    OFFLOAD_BUNDLER= osSelect(linux="clang-offload-bundler", windows="clang-offload-bundler.exe") #+ "__deleteme__"
    ASSEMBLER = osSelect(linux="amdclang++", windows="clang++.exe") #+ "__deleteme__"


def supportedCCompiler(compiler: str) -> bool:
    """Determine if a C compiler/assembler is supported by Tensile.

    Args:
        compiler: The name of a compiler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    target = ToolchainDefaults.C_COMPILER
    isSupported = compiler == target or Path(compiler).name == target
    return isSupported


def supportedCxxCompiler(compiler: str) -> bool:
    """Determine if a C++/HIP compiler/assembler is supported by Tensile.

    Args:
        compiler: The name of a compiler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    target = ToolchainDefaults.CXX_COMPILER
    isSupported = compiler == target or Path(compiler).name == target
    return isSupported


def supportedOffloadBundler(bundler: str) -> bool:
    """Determine if an offload bundler is supported by Tensile.

    Args:
        bundler: The name of an offload bundler to test for support.

    Return:
        If supported True; otherwise, False.
    """
    target = ToolchainDefaults.OFFLOAD_BUNDLER
    isSupported = bundler == target or Path(bundler).name == target
    return isSupported


def _exeExists(file: Path) -> bool:
    """Check if a file exists and is executable.

    Args:
        file: The file to check.

    Returns:
        If the file exists and is executable, True; otherwise, False
    """
    if os.access(file, os.X_OK):
        if "rocm" not in file.parts: warn(f"Found non-ROCm install of `{file.name}`: {file}")
        return True
    return False


def _validateExecutable(file: str, searchPaths: List[Path]) -> str:
    """Validate that the given toolchain component is in the PATH and executable.

    Args:
        file: The executable to validate.
        searchPaths: List of directories to search for the executable.

    Returns:
        The validated executable with an absolute path.
    """
    if not any((supportedCxxCompiler(file), supportedCCompiler(file), supportedOffloadBundler(file))):
        raise ValueError(f"{file} is not a supported toolchain component for OS: {os.name}")

    if _exeExists(Path(file)): return file
    for path in searchPaths:
        path /= file 
        if _exeExists(path): return str(path)
    raise FileNotFoundError(f"`{file}` either not found or not executable in any search path: {':'.join(map(str, searchPaths))}")


def validateToolchain(*args: str):
    """Validate that the given toolchain components are in the PATH and executable.

    Args:
        args: List of executable toolchain components to validate.
     
    Returns:
        List of validated executables with absolute paths.
    """
    if os.name == "nt":
      raise NotImplementedError("Toolchain verification is not support on Windows yet.")

    searchPaths = [
        ROCM_BIN_PATH,
        ROCM_LLVM_BIN_PATH,
    ] + [Path(p) for p in os.environ["PATH"].split(os.pathsep)]

    return (_validateExecutable(x, searchPaths) for x in args)