################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from . import CUSTOM_KERNEL_PATH
from .Common import checkParametersAreValid, validParameters

import yaml

import os

def isCustomKernelConfig(config):
    return "CustomKernelName" in config and config["CustomKernelName"]

def getCustomKernelFilepath(name, directory=CUSTOM_KERNEL_PATH):
    return os.path.join(directory, (name + ".s"))

def getAllCustomKernelNames(directory=CUSTOM_KERNEL_PATH):
    return [fname[:-2] for fname in os.listdir(directory) if fname.endswith(".s")]

def getCustomKernelContents(name, directory=CUSTOM_KERNEL_PATH):
    try:
        with open(getCustomKernelFilepath(name, directory)) as f:
            return f.read()
    except:
        raise RuntimeError("Failed to find custom kernel: {}".format(os.path.join(directory, name)))

def getCustomKernelConfigAndAssembly(name, directory=CUSTOM_KERNEL_PATH):
    contents  = getCustomKernelContents(name, directory)
    config = "\n"    #Yaml configuration properties
    assembly = ""
    inConfig = False
    for line in contents.splitlines():
        if   line == "---": inConfig = True                          #Beginning of yaml section
        elif line == "...": inConfig = False                         #End of yaml section
        elif      inConfig: config   += line + "\n"
        else              : assembly += line + "\n"; config += "\n"  #Second statement to keep line numbers consistent for yaml errors

    return (config, assembly)

def readCustomKernelConfig(name, directory=CUSTOM_KERNEL_PATH):
    rawConfig, _ = getCustomKernelConfigAndAssembly(name, directory)
    try:
        return yaml.safe_load(rawConfig)["custom.config"]
    except yaml.scanner.ScannerError as e:
        raise RuntimeError("Failed to read configuration for custom kernel: {0}\nDetails:\n{1}".format(name, e))

def getCustomKernelConfig(kernelName, internalSupportParams, directory=CUSTOM_KERNEL_PATH):
    kernelConfig = readCustomKernelConfig(kernelName, directory)
    if "InternalSupportParams" not in kernelConfig:
        raise RuntimeError("Custom kernel %s config must have KernArgsVersion"%kernelName)
    else:
        # CustomKernelConfig must have signature version
        if "KernArgsVersion" not in kernelConfig["InternalSupportParams"]:
            raise RuntimeError("Custom kernel %s config must have KernArgsVersion"%kernelName)
        for key in internalSupportParams:
            if key not in kernelConfig["InternalSupportParams"]:
                kernelConfig["InternalSupportParams"][key] = internalSupportParams[key]
    for k, v in kernelConfig.items():
        if k != "ProblemType":
            checkParametersAreValid((k, [v]), validParameters)
    kernelConfig["KernelLanguage"] = "Assembly"
    kernelConfig["CustomKernelName"] = kernelName

    return kernelConfig
