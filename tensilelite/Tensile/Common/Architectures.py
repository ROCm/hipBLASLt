################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import re
from subprocess import run, PIPE
from typing import List, Optional

from .Types import IsaVersion
from .Utilities import locateExe

# Translate GPU targets to filter filenames in Tensile_LOGIC directory
architectureMap = {
    "all": "_",
    "gfx000": "none",
    "gfx803": "r9nano",
    "gfx900": "vega10",
    "gfx906": "vega20",
    "gfx906:xnack+": "vega20",
    "gfx906:xnack-": "vega20",
    "gfx908": "arcturus",
    "gfx908:xnack+": "arcturus",
    "gfx908:xnack-": "arcturus",
    "gfx90a": "aldebaran",
    "gfx90a:xnack+": "aldebaran",
    "gfx90a:xnack-": "aldebaran",
    "gfx942": "aquavanjaram",
    "gfx942:xnack+": "aquavanjaram",
    "gfx942:xnack-": "aquavanjaram",
    "gfx1010": "navi10",
    "gfx1011": "navi12",
    "gfx1012": "navi14",
    "gfx1030": "navi21",
    "gfx1100": "navi31",
    "gfx1101": "navi32",
    "gfx1102": "navi33",
    "gfx1200": "gfx1200",
    "gfx1201": "gfx1201",
}

SUPPORTED_ISA = [
    IsaVersion(8, 0, 3),
    IsaVersion(9, 0, 0),
    IsaVersion(9, 0, 6),
    IsaVersion(9, 0, 8),
    IsaVersion(9, 0, 10),
    IsaVersion(9, 4, 0),
    IsaVersion(9, 4, 1),
    IsaVersion(9, 4, 2),
    IsaVersion(10, 1, 0),
    IsaVersion(10, 1, 1),
    IsaVersion(10, 1, 2),
    IsaVersion(10, 3, 0),
    IsaVersion(11, 0, 0),
    IsaVersion(11, 0, 1),
    IsaVersion(11, 0, 2),
    IsaVersion(12, 0, 0),
    IsaVersion(12, 0, 1),
]

SUPPORTED_GFX = [isaToGfx(isa) for isa in SUPPORTED_ISA]

def gfxToIsa(name: str) -> Optional[IsaVersion]:
    """Extracts the ISA version from a given gfx architecture name.

    Args:
        name: The gfx name of the GPU architecture (e.g., 'gfx906').

    Returns:
        An object representing the major, minor, and step version of the ISA.
            Returns None if the name does not match the expected pattern.
    """
    match = re.search(r"gfx([0-9a-fA-F]{3,})", name)
    if not match:
        return None
    ipart = match.group(1)
    step = int(ipart[-1], 16)

    ipart = ipart[:-1]
    minor = int(ipart[-1])

    ipart = ipart[:-1]
    major = int(ipart)
    return IsaVersion(major, minor, step)


def isaToGfx(arch: IsaVersion) -> str:
    """Converts an ISA version to a gfx architecture name.

    Args:
        arch: An object representing the major, minor, and step version of the ISA.

    Returns:
        The name of the GPU architecture (e.g., 'gfx906').
    """
    # Convert last digit to hex because reasons
    name = str(arch[0]) + str(arch[1]) + ("%x" % arch[2])
    return "gfx" + "".join(map(str, name))


def gfxToSwCodename(gfxName: str) -> Optional[str]:
    """Retrieves the common name for a given gfx architecture name.

    Args:
        gfxName: The name of the GPU architecture (e.g., gfx1100).

    Returns:
        The common name of the GPU architecture (e.g., navi31) if found in ``architectureMap``.
            Returns None if the name is not found.
    """
    if gfxName in architectureMap:
        return architectureMap[gfxName]
    else:
        for archKey in architectureMap:
            if gfxName in archKey:
                return architectureMap[archKey]
            return None


def cliArchsToIsa(cliArchs: str) -> List[IsaVersion]:
    """Maps the requested gfx architectures to ISA numbers.

    Args:
        archs: str of ";" or "_" separated gfx architectures (e.g., gfx1100 or gfx90a;gfx1101).

    Returns:
        List of tuples
    """
    archs = cliArchs.split(";") if ";" in cliArchs else cliArchs.split("_")
    return SUPPORTED_ISA if "all" in archs else [gfxToIsa(''.join(map(str, arch))) for arch in archs]


def _detectGlobalCurrentISA(detectionTool, deviceId: int):
    """
    Returns returncode if detection failure
    """
    process = run([detectionTool], stdout=PIPE)
    archList = []
    for line in process.stdout.decode().split("\n"):
        arch = gfxToIsa(line.strip())
        if arch is not None:
            if arch in SUPPORTED_ISA:
                print(f"# Detected GPU {deviceId} with ISA: " + isaToGfx(arch))
                archList.append(arch)
    if process.returncode:
        print(f"{detectionTool} exited with code {process.returncode}")
    return archList[deviceId] if (len(archList) > 0 and process.returncode == 0) else process.returncode


# locateExe silently fails which is not good
AMDGPUArchPath = locateExe(
    "/opt/rocm", "llvm/bin/amdgpu-arch"
)

ROCmAgentEnumeratorPath = locateExe(
    "/opt/rocm/bin","rocm_agent_enumerator"
)

def detectGlobalCurrentISA(deviceId: int):
    """
    Returns returncode if detection failure
    """
    result = _detectGlobalCurrentISA(AMDGPUArchPath, deviceId)
    if not isinstance(result, IsaVersion):
        print("Attempting to detect ISA with rocm_agent_enumerator")
        result = _detectGlobalCurrentISA(ROCmAgentEnumeratorPath, deviceId)
    if not isinstance(result, IsaVersion):
        raise Exception("Failed to detect currect ISA")
    return result
