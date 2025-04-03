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

from Tensile.Common import print1, print2, ParallelMap2, ParallelMapConfig
from Tensile.LibraryIO import DataIndex
from Tensile.CustomYamlLoader import load_logic_gfx_arch, load_yaml_sequence_item
from Tensile.CodeObjectName import codeObjectFileBaseName

from glob import iglob
from pathlib import Path
from typing import List
from subprocess import PIPE, run

def logicFileList(archs, logicPath: Path, logicFilter: str, experimental: bool):
    def archMatch(arch: str, archs: List[str]):
        return (arch in archs) or any(a.startswith(arch) for a in archs)
    def validLogicFile(p: Path):
        return p.suffix == ".yaml" and ("all" in archs or archMatch(load_logic_gfx_arch(p), archs))

    assert logicPath.exists(), f"LogicPath {str(logicPath)} doesn't exist"

    globPattern = str(logicPath / f"**/{logicFilter}.yaml")
    logicFiles = (str(logicPath / file) for file in iglob(globPattern, recursive=True))
    if not experimental:
        logicFiles = [file for file in logicFiles if "experimental" not in map(str.lower, Path(file).parts)]

    logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]

    print1(f"# LogicFilter:         {globPattern}")
    print1(f"# Experimental:        {experimental}")
    print2(f"# LibraryLogicFiles: {len(logicFiles)}")
    for logicFile in logicFiles:
        print2("#   %s" % logicFile)

    return logicFiles


def distribute(lst, n):
    import heapq
    list_of_lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    for value, f in lst:
        total, index = heapq.heappop(totals)
        list_of_lists[index].append((value, f))
        heapq.heappush(totals, (total + value, index))
    return sorted(list_of_lists, key=lambda x: sum(first for first, _ in x), reverse=True)


def numberOfBuildKernerls(logicFile):
    result = run(['/bin/grep', "BuildKernel", logicFile], stderr=PIPE, stdout=PIPE, check=False)
    return int(str(result.stdout).count("BuildKernel"))


def getCoFileNames(logicFile):
    from yaml import Loader
    data = {}
    data["ProblemType"] = load_yaml_sequence_item(logicFile, Loader, DataIndex.PROBLEM_TYPE.value)
    properties = load_yaml_sequence_item(logicFile, Loader, DataIndex.DEVICE_PROPERTIES.value)
    if isinstance(properties, dict):
        data["ArchitectureName"] = properties["Architecture"]
        data["CUCount"] = properties["CUCount"]
    else:
        data["ArchitectureName"] = properties
        data["CUCount"] = None
    data["PerfMetric"] = load_yaml_sequence_item(logicFile, Loader, DataIndex.PERF_METRIC.value)
    return codeObjectFileBaseName(data), logicFile
    #coBasename = load_yaml_sequence_item(logicFile, Loader, DataIndex.CODE_OBJECT_NAME)
    #coBasename = load_yaml_sequence_item(logicFile, Loader, 0)
    #return coBasename["codeObjectFile"], logicFile


def schedule(logicFiles: list, numberOfTasks: int, procs: int):
    problemMap = {}
    cofiles = ParallelMap2(getCoFileNames, ParallelMapConfig(message="Scheduling work. ", procs=procs), logicFiles)
    for codeObjectFile, logicFile in cofiles:
        if codeObjectFile in problemMap:
            problemMap[codeObjectFile].append(logicFile)
        else:
            problemMap[codeObjectFile] = [logicFile]
    result = []
    for codeObjectFile, logicFiles in problemMap.items():
        count = sum(numberOfBuildKernerls(logicFile) for logicFile in logicFiles)
        result.append((count, logicFiles))

    return distribute(result, numberOfTasks) # need to convert list of list of tuples to list of list of strings
