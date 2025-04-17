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

from copy import deepcopy
from enum import IntEnum

from Tensile.Common.GlobalParameters import internalParameters
from Tensile.KernelWriterBetaOnly import KernelWriterBetaOnly
from Tensile.KernelWriterConversion import KernelWriterConversion
from Tensile.KernelWriterActivationEnumHeader import KernelWriterActivationEnumHeader
from Tensile.KernelWriterActivationFunction import KernelWriterActivationFunction
from Tensile.KernelWriterActivationOnly import KernelWriterActivationOnly
from Tensile.KernelWriterReduction import KernelWriterReduction


class KernelHelperEnum(IntEnum):
    BetaOnly = 0
    Conversion = 1
    ActivationEnumHeader = 2
    ActivationFunction = 3
    ActivationOnly = 4
    Reduction = 5
    All = 6


def conversionKernelObjectsNames(solution):
  # need to check that the fields mutated in the the init function aren't used in the naming function
  # e.g. usebias got set to zero if gradient which changes behavior of name function.
  conversionKernelObjectsNames = []
  load_vector_width = [1, 2] if solution["ProblemType"]["DataType"].isDouble() else [1, 2, 4]
  gsuList = [internalParameters["GlobalSplitUPGR"]]
  if solution["GlobalSplitUAlgorithm"] == "SingleBuffer":
    gsuList = [1]
  elif solution["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
    return
  for vw in load_vector_width:
    # for _ in gsuList: I don't think this does anything
    if solution["ProblemType"]["UseBias"]:
      typeList = solution["ProblemType"]["BiasDataTypeList"]
      if solution["ProblemType"]["Gradient"]:
      #  # If gradient + bias D, generates a normal GSU kernel for bias D = nullptr case
        conversionKernelObjectsNames.append(KernelWriterConversion._getKernelName(solution, vw))
      for btype in typeList:
        conversionKernelObjectsNames.append(KernelWriterConversion._getKernelName(solution, vw, btype))
    else:
      conversionKernelObjectsNames.append(KernelWriterConversion._getKernelName(solution, vw))
  return conversionKernelObjectsNames if conversionKernelObjectsNames else []


def activationEnumHeaderObjectsNames(solution):
  activationEnumHeaderObjectsNames = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    activationEnumHeaderObjectsNames.append(KernelWriterActivationEnumHeader._getKernelName(solution))
  return activationEnumHeaderObjectsNames


def activationFunctionObjectsNames(solution):
  activationFunctionObjectsNames = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    activationFunctionObjectsNames.append(KernelWriterActivationFunction._getKernelName(solution))
  return activationFunctionObjectsNames


def activationOnlyKernelObjectsNames(solution):
  activationOnlyKernelObjectsNames = []
  if (solution["ActivationFused"] == False) and (solution["ProblemType"]["ActivationType"] != 'none'):
    activationOnlyKernelObjectsNames.append(KernelWriterActivationOnly._getKernelName(solution))
  return activationOnlyKernelObjectsNames


def reductionKernelObjectsNames(solution):
  reductionKernelObjectsNames = []
  if solution["ProblemType"]["Gradient"] and solution["ProblemType"]["UseBias"]:
    for btype in solution["ProblemType"]["BiasDataTypeList"]:
      reductionKernelObjectsNames.append(KernelWriterReduction._getKernelName(solution, btype))
  return reductionKernelObjectsNames


def betaOnlyKernelObjectsNames(solution):
  betaOnlyKernelObjectsNames = []
  if solution["GlobalSplitU"] > 1 or (solution["StreamK"] > 0 and solution["StreamKAtomic"] == 1):
    if solution["ProblemType"]["UseBias"]:
      for btype in solution["ProblemType"]["BiasDataTypeList"]:
        betaOnlyKernelObjectsNames.append(KernelWriterBetaOnly._getKernelName(solution, btype))
    else:
      betaOnlyKernelObjectsNames.append(KernelWriterBetaOnly._getKernelName(solution))
  return betaOnlyKernelObjectsNames


def kernelObjectNameCallables():
    return [(KernelHelperEnum.Conversion, conversionKernelObjectsNames), 
            (KernelHelperEnum.ActivationEnumHeader, activationEnumHeaderObjectsNames),
            (KernelHelperEnum.ActivationFunction, activationFunctionObjectsNames),
            (KernelHelperEnum.ActivationOnly, activationOnlyKernelObjectsNames),
            (KernelHelperEnum.Reduction, reductionKernelObjectsNames),
            (KernelHelperEnum.BetaOnly, betaOnlyKernelObjectsNames)]


def initHelperKernelObjects(solution, kernelHelperType, cxxCompiler, isaInfoMap):
    result = []
    if kernelHelperType == KernelHelperEnum.BetaOnly or kernelHelperType == KernelHelperEnum.All:
        result.extend(initBetaOnlyKernelObjects(solution))
    if kernelHelperType == KernelHelperEnum.Conversion or kernelHelperType == KernelHelperEnum.All:
        result.extend(initConversionKernelObjects(solution, isaInfoMap))
    if kernelHelperType == KernelHelperEnum.ActivationEnumHeader or kernelHelperType == KernelHelperEnum.All:
        result.extend(initActivationEnumHeaderObjects(solution))
    if kernelHelperType == KernelHelperEnum.ActivationFunction or kernelHelperType == KernelHelperEnum.All:
        result.extend(initActivationFunctionObjects(solution, cxxCompiler, isaInfoMap))
    if kernelHelperType == KernelHelperEnum.ActivationOnly or kernelHelperType == KernelHelperEnum.All:
        result.extend(initActivationOnlyKernelObjects(solution))
    if kernelHelperType == KernelHelperEnum.Reduction or kernelHelperType == KernelHelperEnum.All:
        result.extend(initReductionKernelObjects(solution))
    else:
        raise Exception("Failed to find kenerl helper type.")
    return result


def initBetaOnlyKernelObjects(solution):
  betaOnlyKernelObjects = []
  if solution["GlobalSplitU"] > 1 or (solution["StreamK"] > 0 and solution["StreamKAtomic"] == 1):
    if solution["ProblemType"]["UseBias"]:
      for btype in solution["ProblemType"]["BiasDataTypeList"]:
        state = {}
        state["ProblemType"] = deepcopy(solution["ProblemType"])
        state["ProblemType"]["GroupedGemm"] = False
        state["ProblemType"]["BiasDataTypeList"] = []
        state["ProblemType"]["BiasDataType"] = deepcopy(btype)
        state["KernelLanguage"] = "Source"
        state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
        betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))
    else:
      state = {}
      state["ProblemType"] = deepcopy(solution["ProblemType"])
      state["ProblemType"]["GroupedGemm"] = False
      state["KernelLanguage"] = "Source"
      state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
      betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))
  return betaOnlyKernelObjects


def initConversionKernelObjects(solution, isaInfoMap):
  conversionKernelObjects = []
  load_vector_width = [1, 2] if solution["ProblemType"]["DataType"].isDouble() else [1, 2, 4]
  genPGRPostKernels = True
  gsuList = [internalParameters["GlobalSplitUPGR"]]
  if solution["GlobalSplitUAlgorithm"] == "SingleBuffer":
    genPGRPostKernels = False
    gsuList = [1]
  elif solution["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
    return
  for vw in load_vector_width:
    for globalSplitU in gsuList:
      unrollOnly = False if globalSplitU == internalParameters["GlobalSplitUPGR"] else True
      if solution["ProblemType"]["UseBias"]:
        typeList = solution["ProblemType"]["BiasDataTypeList"]
        if solution["ProblemType"]["Gradient"]:
          # If gradient + bias D, generates a normal GSU kernel for bias D = nullptr case
          state = {}
          state["ProblemType"] = deepcopy(solution["ProblemType"])
          state["ProblemType"]["GroupedGemm"] = False
          state["ProblemType"]["UseBias"] = 0
          state["GenPGRPostKernels"] = genPGRPostKernels
          state["KernelLanguage"] = "Source"
          state["GlobalSplitU"] = globalSplitU
          state["UnrollOnly"] = unrollOnly
          state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
          state["ActivationFused"] = solution["ActivationFused"]
          conversionKernelObjects.append(KernelWriterConversion(state, vw, isaInfoMap))
        for btype in typeList:
          state = {}
          state["ProblemType"] = deepcopy(solution["ProblemType"])
          state["ProblemType"]["GroupedGemm"] = False
          state["ProblemType"]["BiasDataTypeList"] = []
          state["ProblemType"]["BiasDataType"] = deepcopy(btype)
          state["GenPGRPostKernels"] = genPGRPostKernels
          state["KernelLanguage"] = "Source"
          state["GlobalSplitU"] = globalSplitU
          state["UnrollOnly"] = unrollOnly
          state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
          state["ActivationFused"] = solution["ActivationFused"]
          conversionKernelObjects.append(KernelWriterConversion(state, vw, isaInfoMap))
      else:
        state = {}
        state["ProblemType"] = deepcopy(solution["ProblemType"])
        state["ProblemType"]["GroupedGemm"] = False
        state["GenPGRPostKernels"] = genPGRPostKernels
        state["KernelLanguage"] = "Source"
        state["GlobalSplitU"] = globalSplitU
        state["UnrollOnly"] = unrollOnly
        state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
        state["ActivationFused"] = solution["ActivationFused"]
        conversionKernelObjects.append(KernelWriterConversion(state, vw, isaInfoMap))
  return conversionKernelObjects


def initActivationEnumHeaderObjects(solution):
  activationEnumHeaderObjects = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    state = {}
    state["ProblemType"] = deepcopy(solution["ProblemType"])
    state["ProblemType"]["GroupedGemm"] = False
    state["KernelLanguage"] = "Source"
    activationEnumHeaderObjects.append(KernelWriterActivationEnumHeader(state))
  return activationEnumHeaderObjects


def initActivationFunctionObjects(solution, cxxCompiler, isaInfoMap):
  activationFunctionObjects = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    state = {}
    state["ProblemType"] = deepcopy(solution["ProblemType"])
    state["ProblemType"]["GroupedGemm"] = False
    state["KernelLanguage"] = "Source"
    state["Kernel"] = {"WavefrontSize": solution["WavefrontSize"], "ISA": tuple(solution["ISA"])}
    activationFunctionObjects.append(KernelWriterActivationFunction(state, cxxCompiler, list(isaInfoMap.keys())))
  return activationFunctionObjects


def initActivationOnlyKernelObjects(solution):
  activationOnlyKernelObjects = []
  if (solution["ActivationFused"] == False) and (solution["ProblemType"]["ActivationType"] != 'none') :
    state = {}
    state["ProblemType"] = deepcopy(solution["ProblemType"])
    state["ProblemType"]["GroupedGemm"] = False
    state["ProblemType"]["UseBias"] = 0
    state["ProblemType"]["BiasDataTypeList"] = []
    state["KernelLanguage"] = "Source"
    state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
    state["ActivationFused"] = solution["ActivationFused"]
    activationOnlyKernelObjects.append(KernelWriterActivationOnly(state))
  return activationOnlyKernelObjects


def initReductionKernelObjects(solution):
  reductionKernelObjects = []
  if solution["ProblemType"]["Gradient"] and solution["ProblemType"]["UseBias"]:
    for btype in solution["ProblemType"]["BiasDataTypeList"]:
      state = {}
      state["ProblemType"] = deepcopy(solution["ProblemType"])
      state["ProblemType"]["GroupedGemm"] = False
      state["ProblemType"]["BiasDataTypeList"] = []
      state["ProblemType"]["BiasDataType"] = deepcopy(btype)
      reductionKernelObjects.append(KernelWriterReduction(state))
  return reductionKernelObjects
