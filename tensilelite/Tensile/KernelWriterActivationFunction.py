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
from .Common import globalParameters, CHeader
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

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()

  def keys(self):
    return self.getKernelName()

  def getKernelName(self):
    return "Tensile%sActivation_%s_%s"%(self.actGradientPrefix, \
                                        self.state["ProblemType"]["ActivationComputeDataType"].toChar(), \
                                        self.state["ProblemType"]["ActivationType"])


  def getSourceFileString(self):
    fileString = "// This is a dummy file."
    return (0, fileString)

  def getHeaderFileString(self):
    activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += CHeader
      fileString += "#pragma once\n\n"
      fileString += "#include \"Tensile%sActivationEnum_%s.h\"\n"%(self.actGradientPrefix, activationCDataType.toChar())
      fileString += "\n"

    self._tf.setKernelInfo(tuple(self.state["Kernel"]["ISA"]), self.state["Kernel"]["WavefrontSize"])
    activation = ActivationInline(activationCDataType, self.state["Kernel"]["ActivationGuard"])
    fileString += activation.generateInlineAssemblyFunction(self.state["ProblemType"]["ActivationType"], exportType=self.actExportType)

    return fileString
