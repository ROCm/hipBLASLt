from typing import NamedTuple, Tuple

IsaVersion = Tuple[int, int, int]


class SemanticVersion(NamedTuple):
    major: int
    minor: int
    patch: int

class DebugConfig(NamedTuple):
  enableAsserts: bool=False
  enableDebugA: bool=False
  enableDebugB: bool=False
  enableDebugC: bool=False
  expectedValueC: float=16.0
  forceCExpectedValue: bool=False
  debugKernel: bool=False
  forceGenerateKernel: bool=False
  printSolutionRejectionReason: bool=False
  splitGSU: bool=False
