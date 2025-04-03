################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import functools
import math
import os
import sys
import time
import re

from inspect import currentframe, getframeinfo
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List

from Tensile import __version__

_verbosity = 1

def setVerbosity(v: int):
    global _verbosity
    _verbosity = v

def getVerbosity():
    return _verbosity

_showProgressBar = True

def setProgressBar(v: bool):
    global _showProgress
    _showProgress = v

def getProgressBar():
    return _showProgress

################################################################################
# Printing
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints
################################################################################
def print1(message):
    if getVerbosity() >= 1:
        print(message)
        sys.stdout.flush()


def print2(message):
    if getVerbosity() >= 2:
        print(message)
        sys.stdout.flush()


def printWarning(message):
    print("Tensile::WARNING: %s" % message)
    sys.stdout.flush()


def printExit(message):
    print("Tensile::FATAL: %s" % message)
    sys.stdout.flush()
    sys.exit(-1)

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

    raise OSError(f"Failed to locate {exeName}")


def ensurePath(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except OSError:
        raise OSError('Failed to create directory "%s" ' % (path))
    return path


def roundUp(f):
    return (int)(math.ceil(f))


def elineno():
    """
    Return the file name and line number of the caller.
    """
    frame = getframeinfo(currentframe().f_back)
    return f"{Path(frame.filename).name}:{frame.lineno}"


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


class ProgressBar:
    """A class for displaying a progress bar in the console.

    This class provides a simple way to display and update a progress bar in the console
    to indicate the progress of a long-running operation. The progress bar can be updated
    incrementally and supports displaying a completion message and time upon finishing.

    Attributes:
        char: The character used to fill the progress bar.
        maxValue: The maximum value the progress bar can represent.
        width: The total width of the progress bar, including borders.
        maxTicks: The maximum number of ticks (fill characters) within the progress bar.
        priorValue: The value of the progress bar during the last update.
        fraction: The fraction of the progress bar that is filled.
        numTicks: The current number of ticks (fill characters) in the progress bar.
        createTime: The timestamp when the progress bar was created.
        message: The message displayed alongside the progress bar.
    """
    def __init__(self, maxValue: int, desc: str, width=40, showTime=False):
        self.char: str = '.'
        self.maxValue: int = maxValue
        self.width: int = width
        self.maxTicks: int = self.width - 10  # Adjusted for better alignment

        self.priorValue: int = 0
        self.fraction: float = 0
        self.numTicks: int = 0
        self.createTime: float = time.time()
        self.showTime: bool = showTime

        self.message: str = desc

    def increment(self, value=1):
        """Increments the progress bar by a given value and updates the display."""
        self.update(self.priorValue + value)

    def update(self, value):
        """Updates the progress bar to a specific value and refreshes the display."""
        currentFraction = 1.0 * value / self.maxValue
        currentNumTicks = int(currentFraction * self.maxTicks)
        if currentNumTicks > self.numTicks:
            self.numTicks = currentNumTicks
            self.fraction = currentFraction
            self.printStatus()
        self.priorValue = value

    def printStatus(self):
        """Prints the current status of the progress bar to the console."""
        progress_bar = self.char * self.numTicks + ' ' * (self.maxTicks - self.numTicks)
        status_msg = f"{self.message} {progress_bar} {self.fraction * 100:.1f}%"

        if self.numTicks == 0:
            sys.stdout.write(status_msg)
        else:
            sys.stdout.write('\r' + ' ' * len(status_msg) + '\r' + status_msg)
        sys.stdout.flush()

    def finish(self):
        """Marks the operation as done, cleans up the display, and prints the completion time."""
        stopTime = time.time()
        if self.showTime:
            sys.stdout.write(f" (took {stopTime - self.createTime:.1f} secs)")
        sys.stdout.write('\n')
        sys.stdout.flush()


class SpinnyThing:
    """A class to display a spinning indicator in the console for long-running operations.

    This class provides a simple way to visually indicate that a long-running operation
    is in progress by displaying a spinning character in the console. The spinner is
    updated at regular intervals, and a completion message with the elapsed time is
    displayed when the operation finishes.

    Attributes:
        msg: The message displayed alongside the spinner.
        chars: The sequence of characters used for the spinner animation.
        index: The current index in the `chars` list for the spinner.
        count: A counter to control the update frequency of the spinner.
        createTime: The timestamp when the spinner was created.
    """
    def __init__(self, desc: str):
        self.message: str = "# " + desc
        self.chars: List[str] = ['|', '/', '-', '\\']
        self.index: int = 0
        self.count: int = 0
        self.createTime: float = time.time()

    def increment(self):
        """Increments the spinner's position and updates the display if necessary."""
        self.count += 1
        if self.count % 3 != 0:
            return

        sys.stdout.write('\r' + ' ' * (len(self.message) + 10))
        sys.stdout.flush()

        sys.stdout.write('\r' + self.message + " " + self.chars[self.index])
        sys.stdout.flush()
        self.index = (self.index + 1) % len(self.chars)

    def finish(self):
        """Clears the spinner and displays a completion message with the elapsed time."""
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10))
        sys.stdout.flush()

        stopTime = time.time()
        elapsedTime = stopTime - self.createTime

        sys.stdout.write('\r' + self.message + f'... Done in {elapsedTime:.1f} secs\n')
        sys.stdout.flush()


def showProgress(objs, *args, **kwargs):
    if not getProgressBar():
        yield from objs
        return

    desc = kwargs.get('desc', '#')
    total = kwargs.get('total', None)
    progress = None

    try:
        progress = ProgressBar(total or len(objs), desc)
    except TypeError:
        progress = SpinnyThing(desc)

    for o in objs:
        yield o
        progress.increment()
    progress.finish()


class DataDirection(Enum):
    NONE = (0,)
    READ = (1,)
    WRITE = 2


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


def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, defaultDictionary):
    if key in sourceDictionary:
        destinationDictionary[key] = deepcopy(sourceDictionary[key])
    else:
        destinationDictionary[key] = deepcopy(defaultDictionary[key])


def isRhel8() -> bool:
    """
    Check if the current OS is Red Hat Enterprise Linux 8 by reading the /etc/os-release file.

    Returns:
        True if the current OS is RHEL 8, False otherwise
    """
    file = Path("/etc/os-release")
    pattern = r'NAME="Red Hat Enterprise Linux".*VERSION_ID="8\.\d+"'
    if not file.exists():
        return False
    with open(file, "r") as f:
        content = f.read()
    match = re.search(pattern, content, re.DOTALL)
    if match:
        printWarning("Rhel8 environments may not support all tools for system queries such as rocm-smi.")
        return True
    return False
