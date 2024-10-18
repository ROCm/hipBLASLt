import re
from pathlib import Path
from typing import NamedTuple, Optional, Union, Tuple, Set, Dict

from ..Common import printWarning

class ArchVariant(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Optional[Set[str]]
    CUCount: Optional[str] = None


class LogicFileError(Exception):
    def __init__(self, message="Expected line is either not present or is malformed"):
        self.message = message
        super().__init__(self.message)


def extractArchVariant(file: Union[str, Path]) -> ArchVariant:
    """Extracts an architecture variant from a given file.

    The file is expected to have the following format:
    - Line 1: Minimum required version (e.g., "- {MinimumRequiredVersion: 4.33.0}")
    - Line 2: Name of the architecture variant (e.g., "- aquavanjaram")
    - Line 3: Architecture and CUCount (e.g., "- {Architecture: gfx900, CUCount: 64}")
    - Line 4: Device IDs (e.g., "- [Device 1234, Device 5678]")

    Args:
        file: Path to a logic file.

    Returns:
        ArchVariant: An object containing the extracted architecture variant.

    Raises:
        LogicFileError: If the file does not match the expected format.
    """

    def l0(line: str):
        if not re.match(r"- \{MinimumRequiredVersion", line):
            raise LogicFileError(f"Expected minimum required version: line: {line}")

    def l1(line: str):
        return line[2:].strip()

    def l2(line: str):
        if match := re.match(r"- \{Architecture: (\w+), CUCount: (\d+)\}", line):
            architecture, cu_count = match.groups()
            return architecture, f"cu={cu_count}"
        elif match := re.match(r"- gfx(\w+)", line):
            return line[2:].strip(), None
        else:
            raise LogicFileError(
                f"Expected architecture and CU count, or only an archiecture: line: {line}"
            )

    def l3(line: str):
        emulationIds = {"0049", "0050", "0051"}
        if re.match(r"- \[Device", line):
            devIds = re.findall(r"Device (\w+)", line)
            if any(id in emulationIds for id in devIds):
                printWarning("Emulation device ID found, interpreting as fallback device...")
                return None
            return set(f"id={id}" for id in devIds)
        if re.match(r"- \[all devices", line.lower()):
            return None
        else:
            raise LogicFileError(f"No device IDs found: line: {line}")

    with open(file, "r") as f:
        l0(f.readline())
        name = l1(f.readline())
        gfx, cu = l2(f.readline())
        deviceIds = l3(f.readline())

    return ArchVariant(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def matchArchVariant(
    variantMap: dict, targetLogicFile: Path
) -> bool:
    """Determines if the architecture variant specified in the target logic file matches the given criteria.

    Args:
        variantMap: A dictionary mapping GFX names to their corresponding variant files.
        targetLogicFile: Path to the target logic file.

    Returns:
        True if the architecture variant matches the given criteria, False otherwise.

    Raises:
        LineNotFoundError: If the target logic file does not contain the expected lines.
        FileNotFoundError: If the target logic file does not exist.
        ValueError: If the target logic file contains invalid entries.
    """
    # temporary, we shouldn't be including experimental files in the build anyway unless it's explicitly requested
    if "experimental/" in str(targetLogicFile).lower():
        return False

    variant = extractArchVariant(targetLogicFile)
    
    if variant.Gfx not in variantMap:
        return False

    variantMapFiles = variantMap[variant.Gfx]
    
    # If CUCount is None and device Ids is None, then this is a fallback file b/c no predicates are specified
    if variant.CUCount == None and variant.DeviceIds == None:
        print("Fallback: ", targetLogicFile.name)
        addAsFallbackList = []
        for key in variantMapFiles:
            if targetLogicFile.name not in variantMapFiles[key]:
                addAsFallbackList.append(True)
        return any(addAsFallbackList)
            
    
    addAsVariantList = []
    if variant.DeviceIds != None:
        print("Device Ids: ", variant.DeviceIds)
        for key in variantMapFiles:
            if "id" in key:
                for id in variant.DeviceIds:
                    if id == key and targetLogicFile.name not in variantMapFiles[key]:
                        variantMapFiles[key].add(targetLogicFile.name)
                        addAsVariantList.append(True)

    if variant.CUCount != None:
        print("CU Count: ", variant.CUCount)
        for key in variantMapFiles:
            if "cu" in key:
                if variant.CUCount == key and targetLogicFile.name not in variantMapFiles[key]:
                    variantMapFiles[key].add(targetLogicFile.name)
                    addAsVariantList.append(True)

    return any(addAsVariantList)
