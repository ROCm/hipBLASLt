import re
from pathlib import Path
from typing import NamedTuple, Optional, Union, Tuple, Set

class ArchVariant(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Set[str]
    CUCount: Optional[int] = None


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
            return architecture, int(cu_count)
        elif match := re.match(r"- gfx(\w+)", line):
            return line[2:].strip(), None
        else:
            raise LogicFileError(
                f"Expected architecture and CU count, or only an archiecture: line: {line}"
            )

    def l3(line: str):
        if re.match(r"- \[Device", line):
            devices = re.findall(r"Device (\w+)", line)
            return set(devices)
        else:
            raise LogicFileError(f"No device IDs found: line: {line}")

    with open(file, "r") as f:
        l0(f.readline())
        name = l1(f.readline())
        gfx, cu = l2(f.readline())
        deviceIds = l3(f.readline())

    return ArchVariant(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def parseArchVariantString(spec: Set[str]) -> Tuple[Set[str], Set[int]]:
    """Parses a set of architecture variant specs to extract device IDs and CU counts.

    An architecture variant specification must have the following form: "id=1234,cu=64".
    Note that all entries must be separated by a comma.

    Args:
        spec: The specification string in the format "id=1234,cu=64,id=5678,cu=32".

    Returns:
        A tuple containing a set of device IDs and a set of CU counts. If an empty string
        is provided, empty sets will be returned.

    Raises:
        ValueError: If the specification string is invalid for the following reasons:
            - The device ID is not 4 characters long.
            - The device ID contains a non-hexadecimal character.
            - The CU count is not a positive integer.
    """
    deviceIdLength = 4
    hexChars = "1234567890abcdef"

    deviceIds = set()
    cuCounts = set()

    idKey = "id"
    cuKey = "cu"
    split = "="

    if spec:
        for s in spec:
            key, _, value = s.strip().partition(split)
            value = value.strip()
            if key == idKey and all(v in hexChars for v in value.lower()) and len(value) == deviceIdLength:
                deviceIds.add(value)
            elif key == cuKey and value.isdigit():
                cuCounts.add(int(value))
            else:
                raise ValueError(f"Invalid architecture variant string: {spec}")

    return deviceIds, cuCounts


def matchArchVariant(
    gfxNames: Set[str], deviceIds: Set[str], cuCounts: Set[int], targetLogicFile: Union[str, Path]
):
    """Determines if the architecture variant specified in the target logic file matches the given criteria.

    Args:
        gfxNames: A set of valid GFX names to match against.
        deviceIds: A set of valid device IDs to match against.
        cuCounts: A set of valid CU counts to match against.
        targetLogicFile: Path to the target logic file.

    Returns:
        bool: True if the architecture variant matches the given criteria, False otherwise.

    Raises:
        LineNotFoundError: If the target logic file does not contain the expected lines.
        FileNotFoundError: If the target logic file does not exist.
        ValueError: If the target logic file contains invalid entries.
    """
    # temporary, we shouldn't be including experimental files in the build anyway unless it's explicitly requested
    if "experimental/" in str(targetLogicFile).lower():
        return False

    variant = extractArchVariant(targetLogicFile)
    conditions = [
        variant.Gfx in gfxNames,
        any(id in deviceIds for id in variant.DeviceIds) or deviceIds == set(),
        variant.CUCount in cuCounts or variant.CUCount == None or cuCounts == set(),
    ]
    if all(conditions):
        return True
    return False
