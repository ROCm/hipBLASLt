from dataclasses import dataclass
from typing import NamedTuple, Tuple

IsaVersion = Tuple[int, int, int]

@dataclass
class IsaInfo:
    assemblerPath: str
    asmCaps: dict
    archCaps: dict
    regCaps: dict
    asmBugs: dict

class SemanticVersion(NamedTuple):
    major: int
    minor: int
    patch: int
