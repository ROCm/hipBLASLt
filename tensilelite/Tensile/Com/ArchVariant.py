import re
from pathlib import Path
from typing import NamedTuple, Optional, Union, Tuple, Set, Dict, List

class ArchVariant(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Optional[Set[str]]
    CUCount: Optional[str] = None


class LogicFileError(Exception):
    def __init__(self, message="Expected line is either not present or is malformed"):
        self.message = message
        super().__init__(self.message)


def _extractArchVariant(file: Union[str, Path]) -> ArchVariant:
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
        emulationIds = {"0049", "0050", "0051", "0052", "0054", "0062"}
        if re.match(r"- \[Device", line):
            devIds = re.findall(r"Device (\w+)", line)
            
            # Temporary, until we add the correct IDs
            if any(id in emulationIds for id in devIds):
                # printWarning("Emulation device ID found, interpreting as fallback device...")
                return None
            return set(f"id={id}" for id in devIds)
        if re.match(r"-\[alldevices", line.lower().replace(" ", "")):
            return None
        else:
            raise LogicFileError(f"No device IDs found: line: {line}")

    with open(file, "r") as f:
        l0(f.readline())
        name = l1(f.readline())
        gfx, cu = l2(f.readline())
        deviceIds = l3(f.readline())

    return ArchVariant(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def _addVariantMap(variantFiles: Dict[str, Set[Tuple[Path, str]]], spec: str, path: Path, fname: str) -> bool:
    if fname not in {x for _, x in variantFiles[spec]}:
        variantFiles[spec].add((path, fname))
        return True
    return False


def _populateVariantMap(variantMap: Dict[str, Dict[str, Set[Tuple[Path, str]]]], targetLogicFile: Path, fallbackKey: str):
        file = Path(targetLogicFile)
        path, fname = file.parent, file.name

        if "experimental/" in str(file).lower():
            return

        variant = _extractArchVariant(file)
        if variant.Gfx not in variantMap:
            return

        variantFiles = variantMap[variant.Gfx]

        # If CU and ID are both None, then this is a fallback file b/c no predicates are specified
        if variant.CUCount is None and variant.DeviceIds is None:
            if all(fname not in {nm for _, nm in variantFiles[spec]} for spec in variantFiles if spec != fallbackKey):
                variantFiles[fallbackKey].add((path, fname))
        else:
            removeFallbacks= []
            for spec in variantFiles:
                if "id" in spec and variant.DeviceIds:
                    removeFallbacks.extend(_addVariantMap(variantFiles, spec, path, fname) for id in variant.DeviceIds if id == spec)
                if "cu" in spec and variant.CUCount:
                    removeFallbacks.append(_addVariantMap(variantFiles, spec, path, fname) if variant.CUCount == spec else False)

            if removeFallbacks and all(removeFallbacks):
                variantFiles["fallback"] = set(filter(lambda x: x[1] != fname, variantFiles[fallbackKey]))


def filterVariants(logicFiles: List[str], variants: Dict[str, Dict[str, Set[Tuple[Path, str]]]]) -> List[str]:
    fallback = "fallback"
    # A `spec` here is a variant specification passed via the command line, e.g., "cu=64"
    # This is how the code differentiates variants of the same gfx, as well as "fallback" files
    variantMap = {gfx: {spec: set() for spec in specs} for gfx, specs in variants.items()}
    for file in variantMap.values():
        file[fallback] = set()

    for logicFile in logicFiles:
        _populateVariantMap(variantMap, Path(logicFile), fallback)

    return [str(p / file) for variantFiles in variantMap.values() for files in variantFiles.values() for p, file in files]