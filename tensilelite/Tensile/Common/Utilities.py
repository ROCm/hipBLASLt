import functools
import math
import os
import re
import sys
import time
from enum import Enum
from typing import List, Tuple

from Tensile import __version__

from .Architectures import isaToGfx


# get param values from structures.
def hasParam(name, structure):
    if isinstance(structure, list):
        for l in structure:
            if hasParam(name, l):
                return True
        return False
    elif isinstance(structure, dict):
        return name in structure
    else:
        return name == structure


def isExe(filePath):
    return os.path.isfile(filePath) and os.access(filePath, os.X_OK)


def locateExe(defaultPath, exeName):  # /opt/rocm/bin, hip-clang
    # look in defaultPath first
    exePath = os.path.join(defaultPath, exeName)
    if isExe(exePath):
        return exePath
    # look in PATH second
    for path in os.environ["PATH"].split(os.pathsep):
        exePath = os.path.join(path, exeName)
        if isExe(exePath):
            return exePath
    return None


def splitArchs(params: dict, fromTensile=False) -> Tuple[List[str], List[str]]:
    """
    Splits and processes the architecture strings based on the provided parameters.

    Args:
        params: A dictionary of global parameters.
        fromTensile: A flag indicating if the function is called from the context of Tensile.

    Returns:
        A tuple containing two lists:
            - archs: A list of architecture strings with ``-`` instead of ``:``
            - cmdlineArchs: A list of architecture strings that retain ``:`` characters.
    """

    def isSupported(arch):
        return (
            params["AsmCaps"][arch]["SupportedISA"] and params["AsmCaps"][arch]["SupportedSource"]
        )

    if ";" in params["Architecture"]:
        wantedArchs = params["Architecture"].split(";")
    else:
        wantedArchs = params["Architecture"].split("_")
    archs = []
    cmdlineArchs = []
    if "all" in wantedArchs:
        for arch in params["SupportedISA"]:
            if isSupported(arch):
                if arch in [(9, 0, 6), (9, 0, 8), (9, 0, 10), (9, 4, 2)]:
                    if arch == (9, 0, 10):
                        archs += [isaToGfx(arch) + "-xnack+"]
                        cmdlineArchs += [isaToGfx(arch) + ":xnack+"]
                    if params["AsanBuild"]:
                        archs += [isaToGfx(arch) + "-xnack+"]
                        cmdlineArchs += [isaToGfx(arch) + ":xnack+"]
                    else:
                        archs += [isaToGfx(arch) + "-xnack-"]
                        cmdlineArchs += [isaToGfx(arch) + ":xnack-"]
                else:
                    archs += [isaToGfx(arch)]
                    cmdlineArchs += [isaToGfx(arch)]
    else:
        for arch in wantedArchs:
            archs += [re.sub(":", "-", arch)]
            cmdlineArchs += [arch]

    # if calling from the context of Tensile we only want the arch associated with the current ISA
    if fromTensile:
        gfx = isaToGfx(params["CurrentISA"])
        archs = set(a for a in archs if gfx in a)
        cmdlineArchs = set(a for a in cmdlineArchs if gfx in a)

    return archs, cmdlineArchs


def ensurePath(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except OSError:
        printExit('Failed to create directory "%s" ' % (path))
    return path


def roundUp(f):
    return (int)(math.ceil(f))


################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def versionIsCompatible(queryVersionString):
    (qMajor, qMinor, qStep) = queryVersionString.split(".")
    (tMajor, tMinor, tStep) = __version__.split(".")

    # major version must match exactly
    if qMajor != tMajor:
        return False

    # minor.patch version must be >=
    if int(qMinor) > int(tMinor):
        return False
    if qMinor == tMinor:
        if int(qStep) > int(tStep):
            return False
    return True


################################################################################
# Progress Bar Printing
# prints "||||" up to width
################################################################################
class ProgressBar:
    def __init__(self, maxValue, width=80):
        self.char = "|"
        self.maxValue = maxValue
        self.width = width
        self.maxTicks = self.width - 7

        self.priorValue = 0
        self.fraction = 0
        self.numTicks = 0
        self.createTime = time.time()

    def increment(self, value=1):
        self.update(self.priorValue + value)

    def update(self, value):
        currentFraction = 1.0 * value / self.maxValue
        currentNumTicks = int(currentFraction * self.maxTicks)
        if currentNumTicks > self.numTicks:
            self.numTicks = currentNumTicks
            self.fraction = currentFraction
            self.printStatus()
        self.priorValue = value

    def printStatus(self):
        sys.stdout.write("\r")
        sys.stdout.write(
            "[%-*s] %3d%%" % (self.maxTicks, self.char * self.numTicks, self.fraction * 100)
        )
        if self.numTicks == self.maxTicks:
            stopTime = time.time()
            sys.stdout.write(" (%-.1f secs elapsed)\n" % (stopTime - self.createTime))
        sys.stdout.flush()

    def finish(self):
        pass


class DataDirection(Enum):
    NONE = (0,)
    READ = (1,)
    WRITE = 2


class SpinnyThing:
    def __init__(self):
        self.chars = ["|", "/", "-", "\\"]
        self.index = 0

    def increment(self, value=1):
        sys.stdout.write("\b" + self.chars[self.index])
        sys.stdout.flush()
        self.index = (self.index + 1) % len(self.chars)

    def finish(self):
        sys.stdout.write("\b*\n")
        sys.stdout.flush()


def iterate_progress(obj, *args, **kwargs):
    try:
        progress = ProgressBar(len(obj))
    except TypeError:
        progress = SpinnyThing()
    for o in obj:
        yield o
        progress.increment()
    progress.finish()


try:
    from tqdm import tqdm
except ImportError:
    tqdm = iterate_progress


def state(obj):
    if hasattr(obj, "state"):
        return obj.state()

    if hasattr(obj.__class__, "StateKeys"):
        rv = {}
        for key in obj.__class__.StateKeys:
            attr = key
            if isinstance(key, tuple):
                (key, attr) = key
            rv[key] = state(getattr(obj, attr))
        return rv

    if isinstance(obj, dict):
        return {k: state(v) for k, v in obj.items()}

    if isinstance(obj, (str, int, float)):
        return obj

    try:
        return [state(i) for i in obj]
    except TypeError:
        pass

    return obj


def state_key_ordering(cls):
    def tup(obj):
        return tuple([getattr(obj, k) for k in cls.StateKeys])

    def lt(a, b):
        return tup(a) < tup(b)

    def eq(a, b):
        return tup(a) == tup(b)

    cls.__lt__ = lt
    cls.__eq__ = eq

    return functools.total_ordering(cls)


def hash_combine(*objs, **kwargs):
    shift = 1
    if "shift" in kwargs:
        shift = kwargs["shift"]

    if len(objs) == 1:
        objs = objs[0]

    rv = 0
    try:
        it = iter(objs)
        rv = next(it)
        for value in it:
            rv = (rv << shift) ^ value
    except TypeError:
        return objs
    except StopIteration:
        pass
    return rv


def hash_objs(*objs, **kwargs):
    return hash(tuple(objs))


def ClientExecutionLock(lockPath: str):
    if not lockPath:
        return open(os.devnull)

    import filelock

    return filelock.FileLock(lockPath)
