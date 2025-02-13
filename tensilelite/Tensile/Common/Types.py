from typing import NamedTuple, Tuple

IsaVersion = Tuple[int, int, int]


class SemanticVersion(NamedTuple):
    major: int
    minor: int
    patch: int
