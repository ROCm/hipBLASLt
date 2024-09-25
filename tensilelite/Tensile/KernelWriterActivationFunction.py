################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from copy import deepcopy

from .TensileInstructions import TensileInstructions
from .Common import globalParameters, CHeader, gfxArch, getGfxName
from .Activation import ActivationInline, ActivationType
from .KernelWriterBase import KernelWriterBase

class KernelWriterActivationFunction(KernelWriterBase):

  def __init__(self, state):
    super().__init__()
    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["Kernel"] = state["Kernel"]
    self._tf = TensileInstructions()

    self.actGradientPrefix = ""
    self.actExportType =  ActivationType.Export.NORMAL
    if self.state["ProblemType"]["Gradient"]:
      self.actGradientPrefix = "Gradient"
      self.actExportType = ActivationType.Export.GRADONLY
    self.gaurdStr = "NG" if self.state["ProblemType"]["ActivationNoGuard"] else ""

    self.enumName = "Tensile::%sActivationType_%s"%(self.actGradientPrefix, \
                                                    self.state["ProblemType"]["ActivationComputeDataType"])

    # Get supported archs
    if ";" in globalParameters["Architecture"]:
      self.supportedArchs = globalParameters["Architecture"].split(";")
    else:
      self.supportedArchs = globalParameters["Architecture"].split("_")
    if "all" in self.supportedArchs:
      self.supportedArchs = deepcopy(globalParameters['SupportedISA'])
    else:
      for idx, arch in enumerate(self.supportedArchs):
        self.supportedArchs[idx] = gfxArch(''.join(map(str, arch)))

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()

  def keys(self):
    return self.getKernelName()

  def getKernelName(self):
    return "Tensile%sActivation%s_%s_%s"%(self.actGradientPrefix, \
                                          self.gaurdStr, \
                                          self.state["ProblemType"]["ActivationComputeDataType"].toChar(), \
                                          self.state["ProblemType"]["ActivationType"])


  def getSourceFileString(self):
    fileString = "// This is a dummy file."
    return (0, fileString)

  def functionSignature(self):
    kStr = ""

    ptrStr = self.state["ProblemType"]["ActivationComputeDataType"].toDevice("HIP")
    names = ""
    if self.state["ProblemType"]["ActivationType"] == 'all':
      names += ",\n"
      names += "  %s const activationType"%self.enumName
    for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList(False):
      names += ",\n"
      names += "  %s const %s"%(ptrStr, name)
    changeLine = "\n  " if names else ""
    kStr += "__device__ inline %s activation%s(%s%s value%s)\n{\n"%(ptrStr, self.gaurdStr, changeLine, ptrStr, names)
    return kStr

  def getInlineAsm(self, activation: ActivationInline, spaces: int, activationType: str):
    activationStrList = []

    isa = tuple(self.state["Kernel"]["ISA"])
    if not self._tf.isInit():
      self._tf.init(isa, globalParameters["AssemblerPath"])
    self._tf.setKernelInfo(isa, self.state["Kernel"]["WavefrontSize"])

    for arch in self.supportedArchs:
      self._tf.init(arch, globalParameters["AssemblerPath"])
      self._tf.setKernelInfo(arch, self.state["Kernel"]["WavefrontSize"])
      activationStrList.append(activation.generateInlineAssemblyBody(spaces, activationType))

    activationStrSetList = list(set(activationStrList))
    # Return if all codes are the same.
    if len(activationStrSetList) == 1:
      return activationStrList[0]

    # Categorize them.
    cateArch = [[] for _ in range(len(activationStrSetList))]
    for idx, actStr in enumerate(activationStrList):
      for i, actSetStr in enumerate(activationStrSetList):
        if actStr == actSetStr:
          cateArch[i].append(tuple(self.supportedArchs[idx]))
          break

    # create marcos
    defineStr = []
    macroStr = "#if"
    for archList in cateArch:
      defStr = "%s defined(__%s__)"%(macroStr, getGfxName(archList[0]))
      for arch in archList:
        defStr += "|| defined(__%s__)"%getGfxName(arch)
      defStr += "\n"
      defineStr.append(defStr)
      macroStr = "#elif"

    kStr = ""
    # Insert activation inline codes to fileString
    for i, defStr in enumerate(defineStr):
      kStr += defStr
      kStr += activationStrSetList[i]
    kStr += "#endif\n"
    return kStr

  def getHeaderFileString(self):
    if self.state["ProblemType"]["ActivationType"] == 'none':
      return fileString

    activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
    isa = tuple(self.state["Kernel"]["ISA"])
    self._tf.init(isa, globalParameters["AssemblerPath"])
    self._tf.setKernelInfo(isa, self.state["Kernel"]["WavefrontSize"])
    activation = ActivationInline(activationCDataType, not self.state["ProblemType"]["ActivationNoGuard"])

    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += CHeader
      fileString += "#pragma once\n\n"
      fileString += "#include \"Tensile%sActivationEnum_%s.h\"\n"%(self.actGradientPrefix, activationCDataType.toChar())
      fileString += "\n"

    fileString += "#pragma clang diagnostic push\n"
    fileString += "#pragma clang diagnostic ignored \"-Winline-asm\"\n"
    fileString += self.functionSignature()
    if self.state["ProblemType"]["ActivationType"] == 'all':
      for index, enumStr in enumerate(ActivationType.getEnumStrList(activationCDataType, \
                                                                    includeNone=False, \
                                                                    exportType=self.actExportType)):
        if index == 0:
          fileString += "  if (activationType == %s::%s) {\n"%(self.enumName, ActivationType(enumStr).toEnum())
        else:
          fileString += "  else if (activationType == %s::%s) {\n"%(self.enumName, ActivationType(enumStr).toEnum())
        fileString += self.getInlineAsm(activation, 4, enumStr)
        fileString += "  }"
      fileString += "\n"
    else:
      fileString += self.getInlineAsm(activation, 2, self.state["ProblemType"]["ActivationType"])
    fileString += "  return value;\n"
    fileString += "}\n"
    fileString += "#pragma clang diagnostic pop\n"

    return fileString
