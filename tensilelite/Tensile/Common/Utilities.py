import functools
import math
import os
import re
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from enum import Enum
from typing import Optional

from .Architectures import gfxToIsa, isaToGfx
from .Capabilities import initArchCaps, initAsmBugs, initAsmCaps
from .GlobalParameters import (
    __version__,
    defaultGlobalParameters,
    globalParameters,
    validParameters,
)


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
        # printExit("structure %s is not list or dict" % structure)


################################################################################
# Print Debug
################################################################################
def print1(message):
    if globalParameters["PrintLevel"] >= 1:
        print(message)
        sys.stdout.flush()


def print2(message):
    if globalParameters["PrintLevel"] >= 2:
        print(message)
        sys.stdout.flush()


def printWarning(message):
    print("Tensile::WARNING: %s" % message)
    sys.stdout.flush()


def printExit(message):
    print("Tensile::FATAL: %s" % message)
    sys.stdout.flush()
    sys.exit(-1)


################################################################################
# Locate Executables
# rocm-smi, hip-clang, rocm_agent_enumerator, clang-offload-bundler
################################################################################
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


def which(p):
    if "CMAKE_CXX_COMPILER" in os.environ and os.path.isfile(os.environ["CMAKE_CXX_COMPILER"]):
        return os.environ["CMAKE_CXX_COMPILER"]
    if os.name == "nt":
        exes = [
            p + x for x in [".exe", "", ".bat"]
        ]  # bat may be front end for file with no extension
    else:
        exes = [p + x for x in ["", ".exe", ".bat"]]
    system_path = os.environ["PATH"].split(os.pathsep)
    for dirname in system_path + [globalParameters["ROCmBinPath"]]:
        for exe in exes:
            candidate = os.path.join(os.path.expanduser(dirname), exe)
            if os.path.isfile(candidate):
                return candidate
    return None


def splitArchs(fromTensile=False):
    # Helper for architecture
    def isSupported(arch):
        return (
            globalParameters["AsmCaps"][arch]["SupportedISA"]
            and globalParameters["AsmCaps"][arch]["SupportedSource"]
        )

    if ";" in globalParameters["Architecture"]:
        wantedArchs = globalParameters["Architecture"].split(";")
    else:
        wantedArchs = globalParameters["Architecture"].split("_")
    archs = []
    cmdlineArchs = []
    if "all" in wantedArchs:
        for arch in globalParameters["SupportedISA"]:
            if isSupported(arch):
                if arch in [(9, 0, 6), (9, 0, 8), (9, 0, 10), (9, 4, 0), (9, 4, 1), (9, 4, 2)]:
                    if arch == (9, 0, 10):
                        archs += [isaToGfx(arch) + "-xnack+"]
                        cmdlineArchs += [isaToGfx(arch) + ":xnack+"]
                    if globalParameters["AsanBuild"]:
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
        gfx = isaToGfx(globalParameters["CurrentISA"])
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


def checkParametersAreValid(param, validParams):
    """Ensures paramaters in params exist and have valid values as specified by validParames"""
    (name, values) = param
    if name == "ProblemSizes":
        return
    elif name == "InternalSupportParams":
        return

    if name not in validParams:
        printExit(
            "Invalid parameter name: {}\nValid parameters are {}.".format(
                name, sorted(validParameters.keys())
            )
        )

    for value in values:
        if validParams[name] != -1 and value not in validParams[name]:
            msgBase = "Invalid parameter value: {} = {}\nValid values for {} are {}{}."
            msgExt = (
                " (only first 32 combos printed)\nRefer to Common.py for more info"
                if len(validParams[name]) > 32
                else ""
            )
            printExit(msgBase.format(name, value, name, validParams[name][:32], msgExt))


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


def showwarning(message, category, filename, lineno, file=None, line=None):
    msg = f"> {category.__name__}: {message}"
    print(msg)


warnings.showwarning = showwarning


################################################################################
# Is query version compatible with current version
# a yaml file is compatible with tensile if
# tensile.major == yaml.major and tensile.minor.step > yaml.minor.step
################################################################################
def detectGlobalCurrentISA_(detectionTool):
    """
    Returns returncode if detection failure
    """
    global globalParameters

    if globalParameters["CurrentISA"] == (0, 0, 0) and detectionTool:
        process = subprocess.run([detectionTool], stdout=subprocess.PIPE)
        if os.name == "nt":
            line = ""
            for line_in in process.stdout.decode().splitlines():
                if "gcnArchName" in line_in:
                    line += line_in.split()[1]
                    break  # detemine if hipinfo will support multiple arch
            arch = gfxToIsa(line.strip())
            if arch is not None:
                if arch in globalParameters["SupportedISA"]:
                    print1("# Detected local GPU with ISA: " + isaToGfx(arch))
                    globalParameters["CurrentISA"] = arch
        else:
            archList = []
            for line in process.stdout.decode().split("\n"):
                arch = gfxToIsa(line.strip())
                if arch is not None:
                    if arch in globalParameters["SupportedISA"]:
                        print1("# Detected local GPU with ISA: " + isaToGfx(arch))
                        archList.append(arch)
            if len(archList) > 0:
                globalParameters["CurrentISA"] = archList[globalParameters["Device"]]
        if process.returncode:
            printWarning("%s exited with code %u" % (detectionTool, process.returncode))
        return process.returncode
    return 0


def detectGlobalCurrentISA():
    """
    Returns returncode if detection failure
    """
    errorCode = detectGlobalCurrentISA_(globalParameters["AMDGPUArchPath"])
    if errorCode:
        printWarning("Attempting to detect ISA with rocm_agent_enumerator")
        return detectGlobalCurrentISA_(globalParameters["ROCmAgentEnumeratorPath"])
    return errorCode


def restoreDefaultGlobalParameters():
    """
    Restores `globalParameters` back to defaults.
    """
    global globalParameters
    global defaultGlobalParameters
    # Can't just assign globalParameters = deepcopy(defaultGlobalParameters) because that would
    # result in dangling references, specifically in Tensile.Tensile().
    globalParameters.clear()
    for key, value in deepcopy(defaultGlobalParameters).items():
        globalParameters[key] = value


def printTable(rows):
    rows = list([[str(cell) for cell in row] for row in rows])
    colWidths = list([max([len(cell) for cell in col]) for col in zip(*rows)])

    for row in rows:
        for width, cell in zip(colWidths, row):
            pad = " " * (width - len(cell))
            print(pad, cell, sep="", end=" ")
        print()


def printCapTable(parameters):
    import itertools

    archs = [(0, 0, 0)] + parameters["SupportedISA"]
    gfxNames = list(map(isaToGfx, archs))

    headerRow = ["cap"] + gfxNames

    def capRow(caps, cap):
        return [cap] + [("1" if cap in caps[arch] and caps[arch][cap] else "0") for arch in archs]

    allAsmCaps = set(
        itertools.chain(*[caps.keys() for arch, caps in parameters["AsmCaps"].items()])
    )
    allAsmCaps = sorted(allAsmCaps, key=lambda k: (k.split("_")[-1], k))
    asmCapRows = [capRow(parameters["AsmCaps"], cap) for cap in allAsmCaps]

    allArchCaps = set(
        itertools.chain(*[caps.keys() for arch, caps in parameters["ArchCaps"].items()])
    )
    allArchCaps = sorted(allArchCaps)
    archCapRows = [capRow(parameters["ArchCaps"], cap) for cap in allArchCaps]

    printTable([headerRow] + asmCapRows + archCapRows)


def assignGlobalParameters(config, cxxCompiler=None):
    """
    Assign Global Parameters
    Each global parameter has a default parameter, and the user
    can override them, those overridings happen here
    """

    global globalParameters

    # Minimum Required Version
    if "MinimumRequiredVersion" in config:
        if not versionIsCompatible(config["MinimumRequiredVersion"]):
            printExit(
                "Config file requires version=%s is not compatible with current Tensile version=%s"
                % (config["MinimumRequiredVersion"], __version__)
            )

    # User-specified global parameters
    print2("GlobalParameters:")
    for key in globalParameters:
        defaultValue = globalParameters[key]
        if key in config:
            configValue = config[key]
            if configValue == defaultValue:
                print2(" %24s: %8s (same)" % (key, configValue))
            else:
                print2(" %24s: %8s (overriden)" % (key, configValue))
        else:
            print2(" %24s: %8s (unspecified)" % (key, defaultValue))

    globalParameters["ROCmPath"] = "/opt/rocm"
    if "ROCM_PATH" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("ROCM_PATH")
    if "TENSILE_ROCM_PATH" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("TENSILE_ROCM_PATH")
    if os.name == "nt" and "HIP_DIR" in os.environ:
        globalParameters["ROCmPath"] = os.environ.get("HIP_DIR")  # windows has no ROCM
    globalParameters["CmakeCxxCompiler"] = None
    if "CMAKE_CXX_COMPILER" in os.environ:
        globalParameters["CmakeCxxCompiler"] = os.environ.get("CMAKE_CXX_COMPILER")
    if "CMAKE_C_COMPILER" in os.environ:
        globalParameters["CmakeCCompiler"] = os.environ.get("CMAKE_C_COMPILER")

    globalParameters["ROCmBinPath"] = os.path.join(globalParameters["ROCmPath"], "bin")

    # ROCm AMD GPU Arch Path
    # ROCm Agent Enumerator Path
    if os.name == "nt":
        globalParameters["AMDGPUArchPath"] = locateExe(
            globalParameters["ROCmBinPath"], "hipinfo.exe"
        )
        globalParameters["ROCmAgentEnumeratorPath"] = locateExe(
            globalParameters["ROCmBinPath"], "hipinfo.exe"
        )
    else:
        globalParameters["AMDGPUArchPath"] = locateExe(
            globalParameters["ROCmPath"], "llvm/bin/amdgpu-arch"
        )
        globalParameters["ROCmAgentEnumeratorPath"] = locateExe(
            globalParameters["ROCmBinPath"], "rocm_agent_enumerator"
        )

    globalParameters["ROCmSMIPath"] = locateExe(globalParameters["ROCmBinPath"], "rocm-smi")
    globalParameters["ROCmLdPath"] = locateExe(
        os.path.join(globalParameters["ROCmPath"], "llvm/bin"), "ld.lld"
    )

    globalParameters["ExtractKernelPath"] = locateExe(
        os.path.join(globalParameters["ROCmPath"], "hip/bin"), "extractkernel"
    )

    if "AMDGPUArchPath" in config:
        globalParameters["AMDGPUArchPath"] = config["AMDGPUArchPath"]

    if "AsanBuild" in config:
        globalParameters["AsanBuild"] = config["AsanBuild"]

    if "KeepBuildTmp" in config:
        globalParameters["KeepBuildTmp"] = config["KeepBuildTmp"]

    if "CodeObjectVersion" in config:
        globalParameters["CodeObjectVersion"] = config["CodeObjectVersion"]

    # read current gfx version
    returncode = detectGlobalCurrentISA()
    if globalParameters["CurrentISA"] == (0, 0, 0):
        printWarning(
            "Did not detect SupportedISA: %s; cannot benchmark assembly kernels."
            % globalParameters["SupportedISA"]
        )
    if returncode:
        if os.name == "nt":
            globalParameters["CurrentISA"] = (9, 0, 6)
            printWarning("Failed to detect ISA so forcing (gfx906) on windows")

    globalParameters["AsmCaps"] = {}
    globalParameters["ArchCaps"] = {}
    globalParameters["AsmBugs"] = {}

    for v in globalParameters["SupportedISA"] + [(0, 0, 0)]:
        globalParameters["AsmCaps"][v] = initAsmCaps(v, cxxCompiler, False)
        globalParameters["ArchCaps"][v] = initArchCaps(v)
        globalParameters["AsmBugs"][v] = initAsmBugs(globalParameters["AsmCaps"][v])

    if globalParameters["PrintLevel"] >= 1:
        printCapTable(globalParameters)

    globalParameters["SupportedISA"] = list(
        [
            i
            for i in globalParameters["SupportedISA"]
            if globalParameters["AsmCaps"][i]["SupportedISA"]
        ]
    )

    validParameters["ISA"] = [(0, 0, 0), *globalParameters["SupportedISA"]]

    # For ubuntu platforms, call dpkg to grep the version of hip-clang.  This check is platform specific, and in the future
    # additional support for yum, dnf zypper may need to be added.  On these other platforms, the default version of
    # '0.0.0' will persist

    # Due to platform.linux_distribution() being deprecated, just try to run dpkg regardless.
    # The alternative would be to install the `distro` package.
    # See https://docs.python.org/3.7/library/platform.html#platform.linux_distribution

    # The following try except block computes the hipcc version
    try:
        if os.name == "nt":
            compileArgs = ["perl"] + [which("hipcc")] + ["--version"]
            output = subprocess.run(compileArgs, check=True, stdout=subprocess.PIPE).stdout.decode()
        else:
            compiler = "hipcc"
            output = subprocess.run(
                [compiler, "--version"], check=True, stdout=subprocess.PIPE
            ).stdout.decode()

        for line in output.split("\n"):
            if "HIP version" in line:
                globalParameters["HipClangVersion"] = line.split()[2]
                print1("# Found hipcc version " + globalParameters["HipClangVersion"])

    except (subprocess.CalledProcessError, OSError) as e:
        printWarning("Error: {} running {} {} ".format("hipcc", "--version", e))

    # The following keys may be present in the config, but are not (or no longer) global parameters.
    ignoreKeys = [
        "UseCompression",
        "CxxCompiler",
        "CCompiler",
        "OffloadBundler",
        "Assembler",
        "LogicPath",
        "LogicFilter",
        "OutputPath",
        "Experimental",
        "GenSolTable",
    ]
    for key in config:
        if key in ignoreKeys:
            continue
        value = config[key]
        if key not in globalParameters:
            printWarning("Global parameter %s = %s unrecognised." % (key, value))
        globalParameters[key] = value


def setupRestoreClocks():
    import atexit

    def restoreClocks():
        if globalParameters["PinClocks"]:
            rsmi = globalParameters["ROCmSMIPath"]
            subprocess.call([rsmi, "-d", "0", "--resetclocks"])
            subprocess.call([rsmi, "-d", "0", "--setfan", "50"])

    atexit.register(restoreClocks)


setupRestoreClocks()


################################################################################
# Assign Parameters
# populate dst with src[key] else give it the default/backup value
################################################################################
def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, defaultDictionary):
    if key in sourceDictionary:
        destinationDictionary[key] = deepcopy(sourceDictionary[key])
    else:
        destinationDictionary[key] = deepcopy(defaultDictionary[key])


def ClientExecutionLock():
    if not globalParameters["ClientExecutionLockPath"]:
        return open(os.devnull)

    import filelock

    return filelock.FileLock(globalParameters["ClientExecutionLockPath"])
