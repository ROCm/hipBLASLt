from typing import Tuple, NamedTuple

IsaVersion = Tuple[int, int, int]

class SemanticVersion(NamedTuple):
    major: int
    minor: int
    patch: int
