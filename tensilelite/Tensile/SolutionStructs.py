################################################################################
#
# Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import assignParameterWithDefault, \
                    defaultProblemType, defaultSolution, \
                    globalParameters, \
                    print2, printExit, printWarning, \
                    validMFMA, validSMFMA, validParameters, \
                    validGEMMTypes, HPATypes, roundUp
from .TensileInstructions import DataType, roundUpToNearestMultiple

from .KernelWriterBetaOnly import KernelWriterBetaOnly
from .KernelWriterConversion import KernelWriterConversion
from .KernelWriterActivationEnumHeader import KernelWriterActivationEnumHeader
from .KernelWriterActivationFunction import KernelWriterActivationFunction
from .KernelWriterActivationOnly import KernelWriterActivationOnly
from .KernelWriterReduction import KernelWriterReduction

from .Activation import ActivationType

from .CustomKernels import isCustomKernelConfig

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from typing import List

import collections
import math
import operator
import sys

########################################
# Print a reject message :
def reject(state, *args):
  if state and "NoReject" in state and state["NoReject"]:
    return

  if globalParameters["PrintSolutionRejectionReason"]:
    sys.stdout.write("\nreject: ")
    for a in args:
      print(a)
    #traceback.print_stack(None, 2)
    solutionIndex = state["SolutionIndex"] if (state != None and "SolutionIndex" in state) else -1
    if solutionIndex != -1:
      # If we have valid solutionIndex, this means we are during TensileCreateLibrary stage
      # In this stage, all solutions in the logic should be valid
      # So if any rejection happens, print the warning for further check
      # This will be done only when --global-parameters=PrintSolutionRejectionReason=True
      solutionNameMin = state["SolutionNameMin"] if ("SolutionNameMin" in state) else None
      # if we don't have SolutionNameMin, we simply use the problemTypeName
      solutionNameMin = str(state["ProblemType"]) if (solutionNameMin == None) else solutionNameMin
      print("!! Warning: Any rejection of a LibraryLogic is not expected, please check. \
        SolutionIndex: %d (or SolutionName/ProblemType: %s)"%(solutionIndex, solutionNameMin))
  if state != None:
    state["Valid"] = False

# print a labled variable
def pvar(state, field):
  return field + "=" + str(state[field])

def roundupRatio(dividend, divisor):
  return int(math.ceil(float(dividend) / float(divisor)))

class Fbs(Enum):
  Free=0     # Expect to be free dimension
  Batch=1    # Expect to be batch dimension
  Sum=2      # Expect to be summation dimension

################################################################################
# ProblemType
# name of solution should begin with name of problemType, and arguments can be listed out explicitly
class ProblemType(Mapping):
  ########################################
  def __init__(self, config):
    self.state = {}

    for key in defaultProblemType:
      assignParameterWithDefault(self.state, key, config, defaultProblemType)

    # adjusting all data types
    if "DataType" in config:
      self["DataType"] = DataType(config["DataType"])
    else:
      printExit("NO data type specified")
      self["DataType"] = DataType(0)

    if "DestDataType" in config:
      self["DestDataType"] = DataType(config["DestDataType"])
    else:
      if "DataType" in config:
        self["DestDataType"] = DataType(config["DataType"])
      else:
        printExit("NO dest data type or data type specified")
        self["DataType"] = DataType(0)

    if "ComputeDataType" in config:
      self["ComputeDataType"] = DataType(config["ComputeDataType"])
    else:
      if "DestDataType" in config:
        self["ComputeDataType"] = DataType(config["DestDataType"])
      else:
        if "DataType" in config:
          self["ComputeDataType"] = DataType(config["DataType"])
        else:
          printExit("NO compute data type, or dest data type, or data type specified")
          self["DataType"] = DataType(0)

    if "F32XdlMathOp" in config:
        self["F32XdlMathOp"] = DataType(config["F32XdlMathOp"])
    else:
        self["F32XdlMathOp"] = DataType(0)

    # Modifying ComputeDataType for HHH+HPA: if (HHH+HPA), convert it to HHS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isHalf() and self["DataType"].isHalf() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == f16, DestType == f16, ComputeDataType == f16, but HPA == True (HHH+HPA, no such a type); Converting HHH+HPA to HHS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    # Modifying ComputeDataType for BBB+HPA: if (BBB+HPA), convert it to BBS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isBFloat16() and self["DataType"].isBFloat16() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == bf16, DestType == bf16, ComputeDataType == bf16, but HPA == True (BBB+HPA, no such a type); Converting BBB+HPA to BBS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    # Modifying ComputeDataType for I8I8I_BH: if (I8I8I8+HPA), convert it to I8I8I_BH by setting ComputeDataType to i.
    if self["ComputeDataType"].isInt8() and DataType(config["DataType"]).isInt8() and self["HighPrecisionAccumulate"]:
      print2("DataType == i8 and HPA == True; setting compute data type to int32")
      self["ComputeDataType"] = DataType('i')

    if self["OperationType"] == "GEMM":
      self.checkIfSupportedGEMMType()
      self.initGEMM()
    else:
      printExit("Unsupported OperationType = %s" % self["OperationType"])

    self.state["AssignedDerivedParameters"] = False
    ProblemType.assignDerivedParameters(self.state)

    for tc in ('A', 'B'):
      for sc in self["SetConstStride%s"%tc] :
          (anchorDim, stride) = sc[:2]
          if anchorDim not in self.state["IndexAssignments%s"%tc]:
              printExit("SetConstStride%s=%s anchorDim=%u is not in IndexAssignments%s"%(tc, sc, anchorDim, tc))

    # Bias
    # If compute data type is not equal to dest data type, tensile will run conversion kernel.
    # In this case we don't need to apply bias in beta only kernel.
    if "UseBias" in config:
      if self["ComputeDataType"] != self["DestDataType"]:
        self["BetaOnlyUseBias"] = False
      else:
        self["BetaOnlyUseBias"] = self["UseBias"]
      if "BiasDataTypeList" in config:
        self["BiasDataTypeList"] = [DataType(btype) for btype in config["BiasDataTypeList"]]
        self["BiasDataTypeList"].sort() # Make name unique
      else:
        self["BiasDataTypeList"] = getBiasDataTypeListDefault(self)
    else:
      self["BetaOnlyUseBias"] = False
      self["BiasDataTypeList"] = []

    # Activation
    if "Activation" in config:
      typeStr = 'all' if config["Activation"] else 'none'
      self["ActivationType"] = ActivationType(typeStr)
    else:
      self["ActivationType"] = ActivationType('none')
    if "ActivationHPA" in config:
      self["ActivationHPA"] = config["ActivationHPA"]
    else:
      self["ActivationHPA"] = False

    if self["ActivationType"] != 'none':
      if ((not self["HighPrecisionAccumulate"]) and self["ActivationHPA"]):
          printExit("Must enable HighPrecisionAccumulate to use ActivationHPA.")
      if (not self["ActivationHPA"]) and \
        (self["DataType"].numRegisters() < self["DestDataType"].numRegisters()):
          printWarning("TensileLite only supports ActivationHPA = True if DestDataType > DataType. \
                        ActivationHPA will be set to True automatically.")
          self["ActivationHPA"] = True
      if self["ActivationHPA"] and (self["DataType"] == self["DestDataType"]) and \
        (self["DestDataType"].isSingle() or self["DestDataType"].isDouble()):
        printWarning("Single and Double does not support ActivationHPA. ActivationHPA will be set to False automatically.")
        self["ActivationHPA"] = False

    self["ActivationComputeDataType"] = self["ComputeDataType"] if self["ActivationHPA"] else \
                                        self["DestDataType"]

    if "UseE" in config:
      if config["UseE"]:
        if self["ActivationType"] == 'none':
          printWarning("Use E is disabled cause Activation is set to False.")
          self["UseE"] = False
        else:
          self["UseE"] = config["UseE"]
      else:
        self["UseE"] = config["UseE"]

    if "Gradient" in config:
      if config["Gradient"]:
        if (not self["UseBias"]) and self["ActivationType"] == 'none':
          printWarning("Gradient is disabled cause bias and activation are both disabled.")
          self["Gradient"] = False
        if self["ActivationType"] != 'none' and self["UseE"] == False:
          printWarning("Use E is enabled cause Activation is enabled.")
          self["UseE"] = True
        elif self["ActivationType"] != 'none' and self["UseE"] == False:
          printWarning("Use E is disabled cause Activation is disabled.")
          self["UseE"] = False
        # if self["UseScaleDVec"]:
        #   printWarning("Use scaleDVec is disabled cause Gradient is enabled.")
        #   self["UseScaleDVec"] = False
      self["Gradient"] = config["Gradient"]

    # Need gradient info
    biasSrcList = ["A", "B", "D"]
    if "BiasSrc" in config:
      if not self["Gradient"] and config["BiasSrc"] != "D":
        printWarning("BiasSrc is set to D cause Gradient is disabled.")
        self["BiasSrc"] = "D"
      elif self["Gradient"]:
        # # Currently only supports D :)
        # if config["BiasSrc"] != "D":
        #   printExit("BiasSrc currently only supports D.")
        if config["BiasSrc"] not in biasSrcList:
          printExit("BiasSrc only supports A, B, D.")

    if "ActivationNoGuard" in config:
      self["ActivationNoGuard"] = config["ActivationNoGuard"]
      if self["ActivationNoGuard"]:
        if self["ActivationType"] == 'none':
          printWarning("ActivationNoGuard is set to False cause Acivation is off.")
          self["ActivationNoGuard"] = False
        if (not self["Gradient"]):
          printWarning("ActivationNoGuard is set to False cause Gradient is off.")
          self["ActivationNoGuard"] = False

  ################################################################################
   # Function checkIfSupportedGEMMType:
  #   Assures 3 data-types are valid, supported and well-assigned
  #   See the discussion on Common.py for validGEMMTypes
  ################################################################################
  def checkIfSupportedGEMMType(self):
    inType = self["DataType"]
    outType = self["DestDataType"]
    computeType = self["ComputeDataType"]

    gemmType = ( inType.toChar(), outType.toChar(), computeType.toChar() )
    if gemmType not in validGEMMTypes:
      printExit("This typed-GEMM (Ti, To, Tc) = (%s, %s, %s) is not supported yet."%(gemmType[0],gemmType[1],gemmType[2]))

  ########################################
  def initGEMM(self):
    sumIdx = 3 if self["Batched"] else 2
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    self["IndexAssignmentsMetadata"] = [sumIdx, 0] # T
    #self["IndexAssignmentsMetadata"] = [0, sumIdx] # N
    if self["TransposeA"]:
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
      #self["IndexAssignmentsMetadata"] = [sumIdx, 0] # T
    if self["TransposeB"]:
      self["IndexAssignmentsB"] = [1, sumIdx] # T
    if self["Batched"]:
      self["IndexAssignmentsA"].append(2)
      self["IndexAssignmentsB"].append(2)
      self["IndexAssignmentsMetadata"].append(2)
      self["NumIndicesC"] = 3
    else:
      self["NumIndicesC"] = 2

    self["NumIndicesLD"] = 4
    self["IndexAssignmentsLD"][0] = self["NumIndicesC"] + 1
    for i in range(1, len(self["IndexAssignmentsLD"])):
      self["IndexAssignmentsLD"][i] = self["IndexAssignmentsLD"][i-1] + 1

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  # determine d0, d1, dU
  @staticmethod
  def assignDerivedParameters(state):
    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    state["TotalIndices"] = max(max(state["IndexAssignmentsA"])+1, \
        max(state["IndexAssignmentsB"])+1)

    # determine num free, batch
    state["IndicesFree"] = []
    state["IndicesBatch"] = []
    state["IndicesSummation"] = []

    for i in range(0, state["NumIndicesC"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesBatch"].append(i)

      elif inA or inB:
        state["IndicesFree"].append(i)
      else:
        printExit("invalid index %u (inC but not (inA or inB))" % i)

    # determine num summation
    for i in range(state["NumIndicesC"], state["TotalIndices"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesSummation"].append(i)
      else:
        printExit("invalid index %u (expected summation but not (inA and inB))" % i)
    # print index assignments
    if globalParameters["PrintIndexAssignments"]:
      print("IndicesFree:  %s" % state["IndicesFree"])
      print("IndicesBatch: %s" % state["IndicesBatch"])
      print("IndicesSum:   %s" % state["IndicesSummation"])
      print("IndexAssignmentsA:   %s" % state["IndexAssignmentsA"])
      print("IndexAssignmentsB:   %s" % state["IndexAssignmentsB"])
      print("NumIndicesC:  %s" % state["NumIndicesC"])

    for k in ('IndexAssignmentsA','IndexAssignmentsB'):
      if len(state[k]) != len(set(state[k])):
        printExit("duplicate index in %s=%s"% (k,state[k]))

    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])
    if not state["AllowNoFreeDims"] and state["NumIndicesFree"] < 2 :
      printExit("Tensile requires >= 2 free indices or set AllowNoFreeDims; FreeIndices=%s."% state["IndicesFree"])

    # by default, unroll index will be the last/inner summation index
    state["IndexUnroll"] = state["IndicesSummation"][len(state["IndicesSummation"])-1]
    for i in range(0, len(state["IndexAssignmentsA"])):
      if state["IndexAssignmentsA"][i] == state["IndexUnroll"]:
        state["IndexUnrollA"] = i
        break
    for i in range(0, len(state["IndexAssignmentsB"])):
      if state["IndexAssignmentsB"][i] == state["IndexUnroll"]:
        state["IndexUnrollB"] = i
        break
    for i in range(0, len(state["IndexAssignmentsMetadata"])):
      if state["IndexAssignmentsMetadata"][i] == state["IndexUnroll"]:
        state["IndexUnrollM"] = i
        break
    #print2("IndexUnrollA: %u" % state["IndexUnrollA"])
    #print2("IndexUnrollB: %u" % state["IndexUnrollB"])

    # assign d0, d1
    if state["AllowNoFreeDims"]:
      dimList = state["IndicesFree"] + state["IndicesBatch"]
    else:
      dimList = state["IndicesFree"]
    state["Index01A"] = [i for i in state["IndexAssignmentsA"] if i in dimList][0]
    state["Index01B"] = [i for i in state["IndexAssignmentsB"] if i in dimList][0]
    #print2("Index01A: %u" % state["Index01A"])
    #print2("Index01B: %u" % state["Index01B"])
    # Store code is optimized for 0 as the fastest-moving in memory
    # whichever has lower stride in C (lower value), is 0, other is 1
    if state["Index01A"] < state["Index01B"]:
      state["Index0"]  = state["Index01A"]
      state["Index1"]  = state["Index01B"]
      state["Tensor0"] = 0
      state["Tensor1"] = 1
      state["TileA"] = 0
      state["TileB"] = 1
    else:
      state["Index0"]  = state["Index01B"]
      state["Index1"]  = state["Index01A"]
      state["Tensor0"] = 1
      state["Tensor1"] = 0
      state["TileA"] = 1
      state["TileB"] = 0

    # generalize transpose
    strideIdxA = state["IndexAssignmentsA"].index(state["Index01A"])
    strideIdxB = state["IndexAssignmentsB"].index(state["Index01B"])
    unrollIdxA = state["IndexAssignmentsA"].index(state["IndexUnroll"])
    unrollIdxB = state["IndexAssignmentsB"].index(state["IndexUnroll"])
    state["TLUA"] = strideIdxA < unrollIdxA
    state["TLUB"] = strideIdxB < unrollIdxB
    #state["TLUB"] = True # hack

    if globalParameters["PrintIndexAssignments"]:
      print("TLUA:  %s (stridePosA(%d) <? unrollIdxA(%d)" % \
			(state["TLUA"], strideIdxA, unrollIdxA))
      print("TLUB:  %s (stridePosB(%d) <? unrollIdxB(%d)" % \
	  		(state["TLUB"], strideIdxB, unrollIdxB))
      print("Index01A:  %s" % state["Index01A"])
      print("Index01B:  %s" % state["Index01B"])
    #unrollDimStrideGreaterThanTileDimStrideA = TLUA = !transA = fast
    #!unrollDimStrideLessThanTileDimStrideB   = TLUB =  transB = fast
    state["AssignedDerivedParameters"] = True

    if state["SparseA"]:
      state["Index01Metadata"] = [i for i in state["IndexAssignmentsMetadata"] if i in dimList][0]
      strideIdxM = state["IndexAssignmentsMetadata"].index(state["Index01Metadata"])
      unrollIdxM = state["IndexAssignmentsMetadata"].index(state["IndexUnroll"])
      state["TLUMetadata"] = strideIdxM < unrollIdxM
      if globalParameters["PrintIndexAssignments"]:
        print("TLUMetadata:  %s (stridePosM(%d) <? unrollIdxM(%d)" % \
          (state["TLUMetadata"], strideIdxM, unrollIdxM))
        print("Index01Metadata:  %s" % state["Index01Metadata"])

  ########################################
  def __str__(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self["NumIndicesC"]):
      name += indexChars[i].lower()
    # A dimensions
    name += "_A"
    for i in self["IndexAssignmentsA"]:
      name += indexChars[i] if i in self["MirrorDimsA"] else indexChars[i].lower()
    if self["ComplexConjugateA"]:
      name += "C"
    # B dimensions
    name += "_B"
    for i in self["IndexAssignmentsB"]:
      name += indexChars[i] if i in self["MirrorDimsB"] else indexChars[i].lower()
    if self["ComplexConjugateB"]:
      name += "C"

    # DataTypes
    name += "_"
    name += self["DataType"].toChar() # Type of A/B

    # Special condition for some newly supported kernels:
    #   HHS, HSS, BSS and I8II kernels, use a clearer naming _TiToTc_
    # TODO: Distinguish all kernels by _TiToTc_ to be more consistent with rocblas
    gemmType = (self["DataType"].toChar(),self["DestDataType"].toChar(),self["ComputeDataType"].toChar() )
    if gemmType in HPATypes:
      name += self["DestDataType"].toChar()    # Type of C/D
      name += self["ComputeDataType"].toChar() # Type of Alpha/Beta
      name += "_"

    if not self["F32XdlMathOp"].isSingle() and self["DataType"].isSingle():
      name += "_M"
      name += self["F32XdlMathOp"].toChar()
      name += "_"

    # Other
    if self["UseBeta"]: name += "B"
    if self["HighPrecisionAccumulate"] and not self["SilentHighPrecisionAccumulate"]: name += "H"
    if self["Fp16AltImpl"]: name += "R"
    if self["UseInitialStridesAB"]: name += "I"
    if self["UseInitialStridesCD"]: name += "Ic"
    if self["UseBias"]:
      name += "_Bias" # Not showing bias types
      if self["BiasSrc"] and self["Gradient"]: # Show bias src if gradient = True
        name += "_BiasSrc%s"%self["BiasSrc"]
    if self["UseE"]:
      if self["Gradient"]:
        name += "_Grad"
      else:
        name += "_Aux" # Not showing aux types
    if self["SparseA"]: name += "_SA"

    # precision and other
    # name += "_SB" if self["StridedBatched"] else "_GB"
    if self["GroupedGemm"]:
      name += "_GG"
    else:
      name += "" if self["StridedBatched"] else "_GB" # legacy

    # Activation Naming
    if self["ActivationType"] != 'none':
      if self["ActivationType"] == 'all':
        name += "_A"
      else:
        name += "_%s"%str(self["ActivationType"]).upper()
    if self["ActivationHPA"]: name += "H"
    if self["ActivationNoGuard"]: name += "NG"

    if self["UseScaleDVec"]: name += "_SDV"

    return name

  def keys(self):
    return list(self.state.keys())
  def __len__(self):
    return len(self.state)
  def __iter__(self):
    return iter(self.state)
  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(str(self))
  def __eq__(self, other):
    return isinstance(other, ProblemType) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

  def get(self, key, default=None):
    try:
      return self.state[key]
    except:
      return default



################################################################################
# ProblemSizeRange
################################################################################
class ProblemSizeRange:

  ########################################
  def __init__(self, problemType, config):
    self.totalIndices = 1+max(problemType["IndexAssignmentsA"]) + problemType["NumIndicesLD"]
    if len(config) < self.totalIndices:
      for i in range(len(config), self.totalIndices):
        if i < self.totalIndices - problemType["NumIndicesLD"]:
          config.append(0)
        else:
          config.append([0])

    self.indexMax = []
    self.indexIsSized = []
    self.indicesSized = []
    self.indicesMapped = []
    for i in range(0, self.totalIndices):
      dim = deepcopy(config[i])
      if isinstance(dim, list):
        if len(dim) == 1:
          self.indicesSized.append([dim[0], 1, 0, dim[0]])
        elif len(dim) == 2:
          self.indicesSized.append([dim[0], dim[0], 0, dim[1]])
        elif len(dim) == 3:
          self.indicesSized.append([dim[0], dim[1], 0, dim[2]])
        elif len(dim) == 4:
          self.indicesSized.append([dim[0], dim[1], dim[2], dim[3]])
        else:
          printExit("dimension[%u] config (%s) has %u descriptors rather than 1-4."
              % ( i, dim, len(dim) ))
        self.indexIsSized.append(True)
        self.indexMax.append(self.indicesSized[len(self.indicesSized)-1][3])

      elif isinstance(dim, int):
        self.indicesMapped.append(dim)
        self.indexIsSized.append(False)
        self.indexMax.append(self.indicesSized[self.indicesMapped[ \
            len(self.indicesMapped)-1]][3])

    # max num elements in each tensor
    self.maxNumElements = [ 1, 1, 1 ]
    for i in range(0, problemType["NumIndicesC"]):
      self.maxNumElements[0] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsA"]:
      self.maxNumElements[1] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsB"]:
      self.maxNumElements[2] *= self.indexMax[i]

    self.totalProblemSizes = 1
    self.numProblemSizes = [] # per index
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    sizedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append({})
      if self.indexIsSized[i]:
        self.numProblemSizes.append(0)
        index = self.indicesSized[sizedIdx]
        sizedIdx += 1
        currentSize = index[0]
        currentIncrement = index[1]
        while currentSize <= index[3]:
          currentSize += currentIncrement
          currentIncrement += index[2]
          self.numProblemSizes[i] += 1
      else:
        self.numProblemSizes.append(1)
      self.totalProblemSizes *= self.numProblemSizes[i]

    ########################################
    # enumerate problem sizes
    currentSizedIndexSizes = []
    currentSizedIndexIncrements = []
    for i in range(0, len(self.indicesSized)):
      currentSizedIndexSizes.append(self.indicesSized[i][0])
      currentSizedIndexIncrements.append(self.indicesSized[i][1])

    # iterate over all problem sizes
    self.problemSizes = []
    moreProblemSizes = True
    problemIdx = 0
    problemSize = [0]*self.totalIndices
    while moreProblemSizes:
      #/ convert current sized and mapped indices to full sizes
      currentSizedIdx = 0
      currentMappedIdx = 0
      for i in range(0, self.totalIndices):
        if self.indexIsSized[i]:
          problemSize[i] = currentSizedIndexSizes[currentSizedIdx]
          currentSizedIdx+=1
        else:
          problemSize[i] = problemSize[self.indicesMapped[currentMappedIdx]]
          currentMappedIdx+=1
      self.problemSizes.append(tuple(problemSize))

      #/ increment sizes for next benchmark
      currentSizedIndexSizes[0] += currentSizedIndexIncrements[0]
      currentSizedIndexIncrements[0] += self.indicesSized[0][2]
      for i in range(1, len(self.indicesSized)+1):
        # if prior index past max, reset to min and increment next index
        if currentSizedIndexSizes[i-1] > self.indicesSized[i-1][3]:
          #/ reset prior index
          currentSizedIndexSizes[i-1] = self.indicesSized[i-1][0]
          currentSizedIndexIncrements[i-1] = self.indicesSized[i-1][1]
          # increment next index
          if i >= len(self.indicesSized):
            moreProblemSizes = False
          else:
            currentSizedIndexSizes[i] += currentSizedIndexIncrements[i]
            currentSizedIndexIncrements[i] += self.indicesSized[i][2]

      problemIdx+=1

  ########################################
  # YAML format
  def __str__(self):
    state = "[ "
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      if self.indexIsSized[i]:
        indices = self.indicesSized[sizedIdx]
        state += "[ %u, %u, %u, %u ]" \
            % (indices[0], indices[1], indices[2], indices[3])
        sizedIdx += 1
      else:
        indices = self.indicesSized[self.indicesMapped[mappedIdx]]
        state += str(self.indicesMapped[mappedIdx])
        mappedIdx += 1
      if i < len(self.indexIsSized)-1:
        state += ", "
    state += " ]"
    return state

class Problem:
  """ Problem sizes, strides, padding and other info"""
  def __init__(self, sizes=None, stridesA=None, stridesB=None, stridesC=None, stridesD=None, count=None):
    self.sizes = tuple(sizes) if sizes else None
    self.stridesA = tuple(stridesA) if stridesA else None
    self.stridesB = tuple(stridesB) if stridesB else None
    self.stridesC = tuple(stridesC) if stridesC else None
    self.stridesD = tuple(stridesD) if stridesD else None

    self.count = count

  def __str__(self):
    rv= "{ sizes:" + str(list(self.sizes))
    if self.stridesA:
      rv += ", stridesA:" + str(list(self.stridesA))
    if self.stridesB:
      rv += ", stridesB:" + str(list(self.stridesB))
    if self.stridesC:
      rv += ", stridesC:" + str(list(self.stridesC))
    if self.stridesD:
      rv += ", stridesD:" + str(list(self.stridesD))
    rv += " }"
    return rv

class ExactList(Problem):
  def __init__(self, e, problemType):
    if len(e) == problemType["TotalIndices"]:
      if -1 in e:
        printExit("ExactSize %s contains -1" % (e))
      if problemType["OperationType"] == "GEMM":
        e += [-1, -1, -1, -1]
        e = ExactList.convertLeadingDims(problemType, tuple(e))
      sizes=e

    elif len(e) == (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
      sizes = ExactList.convertLeadingDims(problemType, tuple(e))
    else:
      printExit("ExactSize %s doesn't match indices of ProblemType %s, totalIndices=%d" \
          % (e, problemType, problemType["TotalIndices"]) )

    # TODO- pass strides here, remove calls to convertLeadingDims
    Problem.__init__(self, sizes=sizes)

  def __str__(self):
    return str(list(self.sizes))

  @staticmethod
  def convertLeadingDims(problemType, problemSize, stridesA = None, stridesB = None, stridesC = None, stridesD = None):
    # FIXME-problem: refactor to eliminate max, pass strides in strideB parm rather than hacked
    # onto the end of the sizes list
    predStridesD = stridesD is not None and stridesD[1] != -1
    predStridesC = stridesC is not None and stridesC[1] != -1
    predStridesA = stridesA is not None and stridesA[1] != -1
    predStridesB = stridesB is not None and stridesB[1] != -1
    return problemSize[:problemType["NumIndicesC"]+1] + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][0]]) if not predStridesD else stridesD[1], ) + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][1]]) if not predStridesC else stridesC[1], ) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][2]],
                problemSize[problemType["IndexAssignmentsA"][0]]) if not predStridesA else stridesA[1], ) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][3]],
                problemSize[problemType["IndexAssignmentsB"][0]]) if not predStridesB else stridesB[1], )


class ExactDict(Problem):
  AllowedFields = [ 'count', 'sizes', 'stridesA', 'stridesB', 'stridesC', 'stridesD' ]

  def __init__(self, e, problemType):
    Problem.__init__(self)

    for f in e:
      if f in ExactDict.AllowedFields:
        setattr(self, f, e[f])
      else:
        raise RuntimeError ("specified field '%s' is not a valid Exact dict field"%f)

    if problemType:
      if "OperationType" in problemType and problemType["OperationType"] == "GEMM":
        sizesTuple = tuple(self.sizes + [-1, -1, -1, -1])
        self.sizes = ExactList.convertLeadingDims(problemType, sizesTuple, self.stridesA, self.stridesB, self.stridesC, self.stridesD)

    if problemType:
      if "OperationType" in problemType and problemType["OperationType"] == "GEMM":
        if len(self.sizes) != (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
        # FIXME-ExactDict size descriptor still (but preferrably not so) uses 8-tuple for GEMM problems
          raise RuntimeError ("specified size=%s does not have enough indices for problem (expected %d, got %d)" \
                % (self.sizes, problemType["TotalIndices"]+problemType["NumIndicesLD"], len(self.sizes)))
      elif len(self.sizes) != problemType["TotalIndices"]:
        raise RuntimeError ("specified size=%s does not have enough indices for problem (expected %d, got %d)" \
                % (self.sizes, problemType["TotalIndices"], len(self.sizes)))


################################################################################
# ProblemSizes
################################################################################
"""
Adapter class for class `ProblemSizes`. It satisfies the implicit usage requirement
of ClientWriter.writeClientConfig() by converting ExactLogic to list of `Problem` objects
"""
class ProblemSizesMock:
  def __init__(self, exactLogic):
    self.problems = [Problem(problem) for problem, solution in exactLogic]

class ProblemSizes:

  ########################################
  def __init__(self, problemType, config):
    self.problemType = problemType
    self.ranges = []
    self.exacts = []
    self.minStrides = None
    if config:
      for dictionary in config:
        for sizeTypeKey in dictionary:
          #print ("PROBLEM parsed:", sizeTypeKey, dictionary[sizeTypeKey])
          if sizeTypeKey == "Range":
            psr = ProblemSizeRange(problemType, dictionary[sizeTypeKey])
            self.ranges.append( psr )
          elif sizeTypeKey == "Exact":
            e= dictionary[sizeTypeKey]
            if isinstance(e,list):
              self.exacts.append(ExactList(e, problemType))
            elif isinstance(e,dict):
              self.exacts.append(ExactDict(e, problemType))
            else:
              printExit("Unsupported Exact type==%s"%type(e))
          elif sizeTypeKey == "MinStride":
            e = dictionary[sizeTypeKey]
            if len(e) != problemType["TotalIndices"]:
              printExit("MinStride %s doesn't match indices of ProblemType %s" \
                  % (e, problemType) )
            if self.minStrides:
              printExit("Only one MinStride command is allowed in a ProblemsSizes definition.  Previous minStrides:%s, New minstride:%s" \
                  % (self.minStrides, e) )

            self.minStrides=(tuple(e))
          else:
            printExit("ProblemSize Type %s not supported"%sizeTypeKey)

    if not self.minStrides:
      # set harmless default mins of 0
      self.minStrides = ([0]* problemType["TotalIndices"])

    # not the ideal spot, but convert leading dims that are below the minimum size
    if problemType["OperationType"] == "GEMM":
      for i in range(0, len(self.ranges)):
        self.ranges[i].problemSizes[:] = \
          [ExactList.convertLeadingDims(self.problemType, problemSize) for problemSize in self.ranges[i].problemSizes]

    self.problems = OrderedDict()
    for sizeRange in self.ranges:
      for rangeSize in sizeRange.problemSizes:
        self.problems.update({Problem(rangeSize) : 1})
    for e in self.exacts:
        self.problems.update({e : 1})
    if globalParameters["SortProblems"]:
      self.problems =  sorted(list( self.problems.keys()), key=operator.attrgetter("sizes"))
    else:
      self.problems =  list(self.problems.keys())
    self.totalProblemSizes = len(self.problems)

    # max sizes
    self.maxD = 0
    self.maxC = 0
    self.maxA = 0
    self.maxB = 0
    for problem in self.problems:
      problemSize = problem.sizes # FIXME-problem.   This should use problem.strides*

      sizeLdd = problemSize[self.problemType["IndexAssignmentsLD"][0]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeD = max(self.minStrides[0], sizeLdd)
      for i in range(1, problemType["NumIndicesC"]):
        sizeD *= max(self.minStrides[i], problemSize[i])

      sizeLdc = problemSize[self.problemType["IndexAssignmentsLD"][1]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeC = max(self.minStrides[0], sizeLdc)
      for i in range(1, problemType["NumIndicesC"]):
        sizeC *= max(self.minStrides[i], problemSize[i])

      sizeLda = problemSize[self.problemType["IndexAssignmentsLD"][2]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsA"][0]]
      sizeA = max(self.minStrides[self.problemType["IndexAssignmentsA"][0]], sizeLda)
      for i in self.problemType["IndexAssignmentsA"][1:]:
        sizeA *= max(self.minStrides[i], problemSize[i])

      sizeLdb = problemSize[self.problemType["IndexAssignmentsLD"][3]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsB"][0]]
      sizeB = max(self.minStrides[self.problemType["IndexAssignmentsB"][0]], sizeLdb)
      for i in self.problemType["IndexAssignmentsB"][1:]:
        sizeB *= max(self.minStrides[i], problemSize[i])

      self.maxD = max(self.maxD, sizeD)
      self.maxC = max(self.maxC, sizeC)
      self.maxA = max(self.maxA, sizeA)
      self.maxB = max(self.maxB, sizeB)

  def __str__(self):
    s = "ProblemSizes\n"
    for sizeRange in self.ranges:
      s += "  %s" % sizeRange
    return s

################################################################################
# Bias Type
################################################################################

def getBiasDataTypeListDefault(problem: ProblemType) -> List[DataType]:
  biasDataTypeList = list(set([problem["DataType"], problem["ComputeDataType"], problem["DestDataType"]]))
  biasDataTypeList.sort() # Make name unique
  return biasDataTypeList

class BiasTypeArgs:

  ########################################
  def __init__(self, problemType, config):
    self.biasTypes = []
    self.totalProblemSizes = 0
    if problemType["UseBias"]:
      for btype in config:
        datatype = DataType(btype)
        if datatype not in problemType["BiasDataTypeList"]:
          printWarning("Datatype: %s not support in this kernel (%s)"%(datatype, str(problemType["BiasDataTypeList"])))
        self.biasTypes.append(datatype)

      if not self.biasTypes:
        printExit("Must provide a bias type in benchmark parameters if UseBias is set to True.")

      self.totalProblemSizes = len(self.biasTypes)

  def __str__(self):
    s = "BiasTypesArgs\n"
    return s

################################################################################
# Activation
################################################################################

class activationSetting:
  def __init__(self):
    self.activationEnum = ""
class ActivationArgs:

  ########################################
  def __init__(self, problemType, config):
    self.settingList = []
    self.totalProblemSizes = 0
    if problemType["ActivationType"] == 'none':
      return
    if config:
      for settings in config:
        actSetting = activationSetting()
        for dictionary in settings:
          for sizeTypeKey in dictionary:
            if sizeTypeKey == "Enum":
              actSetting.activationEnum = ActivationType(dictionary[sizeTypeKey])
        if problemType["ActivationType"] == 'all':
          if (not actSetting.activationEnum):
            printExit("Must provide an activation enum if Activation is set to True.")
        else:
          actSetting.activationEnum = problemType["ActivationType"]
        self.settingList.append(actSetting)
        self.totalProblemSizes += 1
    if (problemType["ActivationType"] == 'all') and (not self.settingList):
        printExit("Must provide an activation enum in benchmark parameters if Activation is set to True.")
  def __str__(self):
    s = "ActivationArgs\n"
    return s

# kds is class Solution or class Kernel
# All free dims are packed
def isPackedIndex(ks, index):
  problemType = ks["ProblemType"]
  return index in problemType["IndicesFree"]

def isExtractableIndex(ks, index, tc='x'):
  xA = index in ks['PackedC0IndicesX'][:-1]
  xB = index in ks['PackedC1IndicesX'][:-1]
  if tc=='A':
    return xA
  elif tc=='B':
    return xB
  else:
    return xA or xB

################################################################################
# Solution
################################################################################
class Solution(collections.abc.Mapping):

  ########################################
  def __init__(self, config):
    self._name = None
    config = deepcopy(config)

    self._state = {}
    # problem type
    if "ProblemType" in config:
      self["ProblemType"] = ProblemType(config["ProblemType"])
    else:
      self["ProblemType"] = ProblemType(defaultProblemType)

    # assign parameters with defaults
    for key in defaultSolution:
      assignParameterWithDefault(self._state, key, config, defaultSolution)

    if 'ISA' not in self._state:
      if 'ISA' in config:
        if not globalParameters["AsmCaps"][tuple(config['ISA'])]["SupportedISA"]:
          defaultIsa = [9,0,0]
          print("warning: ISA:", config['ISA'], " is not supported; overriding with ", defaultIsa)
          self._state['ISA'] = defaultIsa
        else:
          self._state['ISA'] = config['ISA']
      else:
        # Assembly by default
        self._state['ISA'] = list(globalParameters["CurrentISA"])
        if 'KernelLanguage' in config:
          if config['KernelLanguage'] != 'Assembly':
            self._state['ISA'] = [0,0,0]

    if "CodeObjectVersion" not in self._state:
      if "CodeObjectVersion" in config:
        self._state["CodeObjectVersion"] = config["CodeObjectVersion"]
      else:
        self._state["CodeObjectVersion"] = globalParameters["CodeObjectVersion"]

    # assign parameters without defaults
    for key in config:
      if key != "ProblemType" and key not in self._state:
        self._state[key] = config[key]
    self["Valid"] = True
    # this could prevent OriginalSolution from re-assigning the parameters, save lots of time
    if "AssignedProblemIndependentDerivedParameters" not in self._state:
      self["AssignedProblemIndependentDerivedParameters"] = False
    if "AssignedDerivedParameters" not in self._state:
      self["AssignedDerivedParameters"] = False

    Solution.assignDerivedParameters(self._state)
    self._name = config["CustomKernelName"] if isCustomKernelConfig(config) else None
    self.initHelperKernelObjects()

  # these keys are copied from ProblemType to internal that may be overridden
  InternalKeys = ["UseSgprForGRO","VectorStore"]


  ########################################
  # get a list of kernel parameters for this solution
  def getKernels(self):
    kernel = deepcopy(self)
    kernel._state.update({"Kernel": True})
    kernels = []
    kernels.append(kernel)
    return kernels


  ########################################
  # create Helper Kernels
  def initHelperKernelObjects(self):
    self.initBetaOnlyKernelObjects()
    self.initConversionKernelObjects()
    self.initActivationEnumHeaderObjects()
    self.initActivationFunctionObjects()
    self.initActivationOnlyKernelObjects()
    self.initReductionKernelObjects()

  ########################################
  # create BetaONly Kernels
  def initBetaOnlyKernelObjects(self):
    self.betaOnlyKernelObjects = []
    if self["GlobalSplitU"] > 1:
      if self["ProblemType"]["UseBias"]:
        for btype in self["ProblemType"]["BiasDataTypeList"]:
          state = {}
          state["ProblemType"] = deepcopy(self["ProblemType"])
          state["ProblemType"]["BiasDataTypeList"] = []
          state["ProblemType"]["BiasDataType"] = deepcopy(btype)
          state["KernelLanguage"] = "Source"
          state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
          self.betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))
      else:
        state = {}
        state["ProblemType"] = deepcopy(self["ProblemType"])
        state["KernelLanguage"] = "Source"
        state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
        self.betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))


  ########################################
  # create Conversion Kernels
  def initConversionKernelObjects(self):
    self.conversionKernelObjects = []
    load_vector_width = [1, 2, 4]
    for vw in load_vector_width:
      if (self["GlobalSplitU"] > 1) and self["_GlobalAccumulation"]:
        if self["ProblemType"]["UseBias"]:
          typeList = self["ProblemType"]["BiasDataTypeList"]
          if self["ProblemType"]["Gradient"]:
            # If gradient + bias D, generates a normal GSU kernel for bias D = nullptr case
            state = {}
            state["ProblemType"] = deepcopy(self["ProblemType"])
            state["ProblemType"]["UseBias"] = False
            state["KernelLanguage"] = "Source"
            state["GlobalSplitU"] = self["GlobalSplitU"]
            state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
            state["ActivationFused"] = self["ActivationFused"]
            self.conversionKernelObjects.append(KernelWriterConversion(state, vw))
            # bias type list
            typeList = [self["ProblemType"]["ComputeDataType"]]
          for btype in typeList:
            state = {}
            state["ProblemType"] = deepcopy(self["ProblemType"])
            state["ProblemType"]["BiasDataTypeList"] = []
            state["ProblemType"]["BiasDataType"] = deepcopy(btype)
            state["KernelLanguage"] = "Source"
            state["GlobalSplitU"] = self["GlobalSplitU"]
            state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
            state["ActivationFused"] = self["ActivationFused"]
            self.conversionKernelObjects.append(KernelWriterConversion(state, vw))
        else:
          state = {}
          state["ProblemType"] = deepcopy(self["ProblemType"])
          state["KernelLanguage"] = "Source"
          state["GlobalSplitU"] = self["GlobalSplitU"]
          state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
          state["ActivationFused"] = self["ActivationFused"]
          self.conversionKernelObjects.append(KernelWriterConversion(state, vw))

  def initActivationEnumHeaderObjects(self):
    self.activationEnumHeaderObjects = []
    if ((self["ProblemType"]["ActivationType"] != 'none') and (self["ProblemType"]["ActivationType"] == 'all')) :
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["KernelLanguage"] = "Source"
      self.activationEnumHeaderObjects.append(KernelWriterActivationEnumHeader(state))

  def initActivationFunctionObjects(self):
    self.activationFunctionObjects = []
    if ((self["ProblemType"]["ActivationType"] != 'none') and (self["ProblemType"]["ActivationType"] == 'all') and \
        ((self["GlobalSplitU"] > 1) or (self["ActivationFused"] == False))) :
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["KernelLanguage"] = "Source"
      state["Kernel"] = {"WavefrontSize": self["WavefrontSize"], "ISA": tuple(self["ISA"])}
      self.activationFunctionObjects.append(KernelWriterActivationFunction(state))

  def initActivationOnlyKernelObjects(self):
    self.activationOnlyKernelObjects = []
    if (((self["GlobalSplitU"] > 1) and (not self["_GlobalAccumulation"])) or (self["ActivationFused"] == False)) \
      and (self["ProblemType"]["ActivationType"] != 'none') :
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["ProblemType"]["UseBias"] = False
      state["ProblemType"]["BiasDataTypeList"] = []
      state["KernelLanguage"] = "Source"
      state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
      state["ActivationFused"] = self["ActivationFused"]
      self.activationOnlyKernelObjects.append(KernelWriterActivationOnly(state))

  def initReductionKernelObjects(self):
    self.reductionKernelObjects = []
    if self["ProblemType"]["Gradient"] and self["ProblemType"]["UseBias"]:
      for btype in self["ProblemType"]["BiasDataTypeList"]:
        state = {}
        state["ProblemType"] = deepcopy(self["ProblemType"])
        state["ProblemType"]["BiasDataTypeList"] = []
        state["ProblemType"]["BiasDataType"] = deepcopy(btype)
        self.reductionKernelObjects.append(KernelWriterReduction(state))

  ########################################
  # get Helper Kernels
  def getHelperKernelObjects(self):
    return self.activationEnumHeaderObjects + self.activationFunctionObjects + \
           self.betaOnlyKernelObjects + self.conversionKernelObjects + \
           self.activationOnlyKernelObjects + self.reductionKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelBetaOlnyObjects(self):
    return self.betaOnlyKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelConversionObjects(self):
    return self.conversionKernelObjects


  ########################################
  # assign tile sizes
  @staticmethod
  def assignProblemIndependentDerivedParameters(state):

    if globalParameters["NewClient"] != 2:
      print("WARNING: Old client deprecated, NewClient parameter being set to 2.")
      globalParameters["NewClient"] = 2

    if "AssignedProblemIndependentDerivedParameters" in state:
      if state["AssignedProblemIndependentDerivedParameters"]:
        return
    state["AssignedProblemIndependentDerivedParameters"] = False
    if "Valid" not in state:
      state["Valid"] = True

    if (not state["ProblemType"]["StridedBatched"]) and (not state["ProblemType"]['Batched']):
      reject(state, "General Batched GEMM only support Batched Problem")

    if (not state["ProblemType"]["StridedBatched"]) and (state["ProblemType"]["OperationType"] != 'GEMM'):
      reject(state, "General Batched GEMM only support GEMM OperationType")

    Solution.MatrixInstructionToMIParameters(state)
    EnableMatrixInstruction = state["EnableMatrixInstruction"] if "EnableMatrixInstruction" in state else None
    if EnableMatrixInstruction == None:
      if  ("MIBlock" in state and len(state["MIBlock"]) == 6) \
          and ("MIWaveGroup" in state and len(state["MIWaveGroup"]) == 2) \
          and ("MIWaveTile" in state and len(state["MIWaveTile"]) == 2):
        EnableMatrixInstruction = True
      elif ("WorkGroup" in state and len(state["WorkGroup"]) == 3) \
          and ("ThreadTile" in state and len(state["ThreadTile"]) == 2) :
        EnableMatrixInstruction = False
      else:
        reject(state, "EnableMatrixInstruction undetermined")

    if EnableMatrixInstruction == True:
      state["MatrixInstM"]         = state["MIBlock"][0]
      state["MatrixInstN"]         = state["MIBlock"][1]
      state["MatrixInstK"]         = state["MIBlock"][2]
      state["MatrixInstB"]         = state["MIBlock"][3]
      state["MatrixInstBM"]        = state["MIBlock"][4]
      state["MatrixInstBN"]        = state["MIBlock"][5]

      state["LocalSplitU"]         = 1
      state["MIOutputVectorWidth"] = 1 if (state["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64') else 4
      state["MIRegPerOut"]         = 2 if (state["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64') else 1

      if state["MatrixInstM"] == 4:
        state["ThreadTile0"] = state["MIWaveTile"][0] * state["MIOutputVectorWidth"]
        state["ThreadTile1"] = state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * state["MatrixInstM"] * state["MatrixInstBM"] // state["MIOutputVectorWidth"]
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"] * state["MatrixInstBN"]
      else:
        state["ThreadTile0"] = state["MatrixInstBM"] * state["MIWaveTile"][0] * (state["MatrixInstM"] * state["MatrixInstN"] // state["WavefrontSize"])
        state["ThreadTile1"] = state["MatrixInstBN"] * state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * (state["WavefrontSize"] // state["MatrixInstN"])
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"]

      #for the old Logic yaml file which does not contain keys: MIInputPerThreadA/B
      if not "MIInputPerThreadA" in state:
        state["MIInputPerThreadA"] = state["MIInputPerThread"]
        state["MIInputPerThreadB"] = state["MIInputPerThread"]

    elif EnableMatrixInstruction == False:
      state["ThreadTile0"] = state["ThreadTile"][0]
      state["ThreadTile1"] = state["ThreadTile"][1]

      state["SubGroup0"]   = state["WorkGroup"][0]
      state["SubGroup1"]   = state["WorkGroup"][1]
      state["LocalSplitU"] = state["WorkGroup"][2]

    if "SubGroup0" in state and "SubGroup1" in state and "LocalSplitU" in state:
      state["NumThreads"]  = state["SubGroup0"] * state["SubGroup1"] * state["LocalSplitU"]

    # macro tile sizes
    if "SubGroup0" in state and "ThreadTile0" in state:
      state["MacroTile0"] = state["SubGroup0"]*state["ThreadTile0"]
    if "SubGroup1" in state and "ThreadTile1" in state:
      state["MacroTile1"] = state["SubGroup1"]*state["ThreadTile1"]
    if "MacroTile" in state:
      if state["MacroTile0"] != state["MacroTile"][0] \
          or state["MacroTile1"] != state["MacroTile"][1]:
        reject(state, "MacroTile mismatch")

    # done
    state["AssignedProblemIndependentDerivedParameters"] = True

  ########################################
  # This is the "classic" algorithm which requires that each threads load the same number of bytes
  # Called with tc=A and then with tc=B
  # totalVectors is totalElements/GRVW, this is #vectors loaded by the LoadTile
  # Reduces the GlobalLoadVectorWidth if necessary if each thread has a small amount of work to do.
  # Output from this function:
  #  state[GlobalLoadVectorWidth*]
  #  state[NumLoads*] # only used in SolutionStructs, with classic alg
  @staticmethod
  def setGlobalLoadVectorWidth(state, tc, totalVectors, grvw):
    validDepthU = True
    if totalVectors < state["NumThreads"]:
      # Try to reduce size of vector so every thread has a load to do
      pv = state["NumThreads"]//totalVectors
      if state["NumThreads"] % totalVectors != 0:
        reject(None, "NumThreads %u %% totalVectors %u != 0" \
            % (state["NumThreads"], totalVectors))
        validDepthU = False
      if pv * totalVectors != state["NumThreads"]:
        reject(None, "pv %u * totalVectors %u != NumThreads %u" \
            % (pv, totalVectors, state["NumThreads"]))
        validDepthU = False
      if grvw % pv != 0:
        reject(None, "GlobalReadVectorWidth %u %% pv %u != 0" \
            % (grvw, pv))
        validDepthU = False
    else:
      pv = 1 # no partial vector required
      if totalVectors % state["NumThreads"] != 0:
        reject(None, "totalVectors %u %% NumThreads %u != 0" \
            % (totalVectors, state["NumThreads"]))
        validDepthU = False

    state["GlobalLoadVectorWidth%s"%tc] = grvw//pv

    # metadata's glvw is equal to which global read's bpl.
    # metadata only accepted bpl is 1,2,4,8 or 16.
    if tc == "Metadata" and state["GlobalLoadVectorWidth%s"%tc] not in [1,2,4,8,16]:
      for i in reversed([1,2,4,8,16]):
        if state["GlobalLoadVectorWidth%s"%tc] % i == 0:
          totalElementsM = totalVectors * grvw
          grvw = i * pv
          totalVectors = totalElementsM // grvw
          return Solution.setGlobalLoadVectorWidth(state, tc, totalVectors, grvw)
      validDepthU = False

    # NumLoads is NOT used on the fractional path
    # NumLoads is number of vector loads per-thread
    state["NumLoads%s"%tc] = totalVectors * pv // state["NumThreads"]
    #print "result: ", pvar(state, "GlobalLoadVectorWidth%s"%tc), \
    #        pvar(state, "NumLoads%s"%tc)

    return validDepthU

  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalescedA]
  #   state[NumLoadsPerpendicularA]
  #   state[LSCA]
  #   state[LSPA]
  @staticmethod
  def setGlobalLoadTileDimClassic(state, tc, numLoads, totalVectorsCoalesced, totalElementsPerp, depthU):

    if state["WaveSeparateGlobalRead%s"%tc]:
      totalElementsPerp = roundupRatio(totalElementsPerp, state["NumThreads"] // state["WavefrontSize"])

    # nlc = 1
    if state["NumLoadsCoalesced%s"%tc] == 1 :
      foundValid = False
      nlcStart = 1
      if state["DirectToVgpr%s"%tc]:
        # adjust nlc for DirectToVgpr
        if state["ProblemType"]["TLU%s"%tc]:
          nlcStart = roundupRatio(state["MIWaveTile%s"%tc], state["GlobalLoadVectorWidth%s"%tc])
        else:
          nlcStart = roundupRatio(depthU, state["MatrixInstK"] * state["GlobalLoadVectorWidth%s"%tc])
      for nlc in range(nlcStart, int(state["NumLoads%s"%tc]+1)):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          #print("NumLoadsCoalesced",state["NumLoadsCoalesced%s"%tc])
          #print("NumLoadsPerpendicular",state["NumLoadsPerpendicular%s"%tc])
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=1 found"%tc)
        return False

    # nlc = -1
    elif state["NumLoadsCoalesced%s"%tc] == -1:
      foundValid = False
      for nlc in range(state["NumLoads%s"%tc], 0, -1):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=-1 found"%tc)
        return False

    # nlc = other
    else:
      if state["NumLoadsCoalesced%s"%tc] > state["NumLoads%s"%tc]:
        reject(state, "%s nlc > numLoads"%tc)
        return False

      state["NumLoadsPerpendicular%s"%tc] = state["NumLoads%s"%tc] \
          // state["NumLoadsCoalesced%s"%tc]

      if state["NumLoads%s"%tc] % state["NumLoadsCoalesced%s"%tc] != 0:
        reject(state, "%s: numLoads %u %% numLoadsCoalesced %u != 0" \
            % (tc, state["NumLoads%s"%tc], state["NumLoadsCoalesced%s"%tc]))
        return False

      if totalVectorsCoalesced % state["NumLoadsCoalesced%s"%tc] != 0 :
        reject(state, "%s: totalVectorsCoalesced %u %% numLoadsPara %u != 0" \
              % (tc, totalVectorsCoalesced, state["NumLoadsCoalesced%s"%tc]))
        return False
      if totalElementsPerp % state["NumLoadsPerpendicular%s"%tc] != 0:
        reject(state, "%s: totalElementsPerp %u %% numLoadsPerp %u != 0" \
              % (tc, totalElementsPerp, state["NumLoadsPerpendicular%s"%tc]))
        return False

    if state["ProblemType"]["TLU%s"%tc]:
      state["LSC%s"%tc] = state["MacroTile%s"%tc] // state["NumLoadsCoalesced%s"%tc]
      state["LSP%s"%tc] = int(math.ceil(float(depthU) / state["NumLoadsPerpendicular%s"%tc]))
    else:
      state["LSC%s"%tc] = int(math.ceil(float(depthU) / state["NumLoadsCoalesced%s"%tc]))
      state["LSP%s"%tc] = state["MacroTile%s"%tc] // state["NumLoadsPerpendicular%s"%tc]

    if state["WaveSeparateGlobalRead%s"%tc] == 1:
      state["LSP%s"%tc] = roundupRatio(state["LSP%s"%tc], state["NumThreads"] // state["WavefrontSize"])
    elif state["WaveSeparateGlobalRead%s"%tc] == 2:
      state["LSP%s"%tc] = state["NumThreads"] // state["WavefrontSize"]

    return True


  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalesced*]
  #   state[NumLoadsPerpendicular*]
  #   state[LSC*]
  #   state[LSP*]
  #   state[GlobalReadVectorWidth]
  #
  # LSC and LSP define the shape of the PerLoadTile, measured in elements.
  #   LSC*LSP is the elements loaded by a single instruction across all
  #   threads in the group.
  #   LSC is the number of elements loaded in the para(coalesced) dimension
  #   LSP is the number of elements loaded in the perp(noncoalesced) dimension
  #   PerLoadTile is always rectangular.
  #   When BufferLoad=1, the area (LSC*LSP) can be larger than NumThreads.
  #   In this case, some threads will generate a dummy OOB GRO.
  #   Related fields:
  #     LVC = LSC/GRVW  (LVCA = LSCA/GLVWA)
  #     LVP = LSP/GRVW  (LVPA = LSPA/GLVWA)
  #
  # NumLoadsCoalesced and NumLoadsPerpendicular define the number of times the
  #   PerLoadTile is loaded in each dimension to fetch the LoadTile
  # LoadTile = (LSC * NumLoadsCoalesced) * (LSP * NumLoadsPerpendicular).
  #   For Fractional, the LoadTile can be larger than the MacroTile. Buffer
  #   loads will clip any OOB references to 0 and will also avoid writing these
  #   into LDS.

  # Fractional load algorithm:
  #  - Each load instruction loads one or more (complete) rows of the load tile.
  #     - Each row is LSC elements wide
  #     - Rows are complete and do not wrap. This allows a single base GRO VGPR
  #       to be used for all loads in the tile.
  #     - Some work-items in the load may not perform useful work. These WI will
  #       set their GRO to a large OOB number so as to do no harm
  #     - Some G2L registers space may be unused as well.
  #     - The 'used' message at the bottom of this routine computes and prints the
  #       wasted register space.
  #     - The wasted space is removed when the data is written to LDS- the LWO
  #       for work-items beyond the valid ones are set to safely write to OOB locations.

  #     - In cases where each load is loading multiple rows (multiple lines of lsc
  #       elements), the last load is allowed to load fewer lines than the others.
  #       The KernelWriterAssembly will modify the LWO for the last load.  This allows
  #       flexibility in the unroll factors for example.
  @staticmethod
  def setGlobalLoadTileDimFractional(state, tc, depthU):

    assert(depthU > 0)
    dbFract = 0

    # parDim, perpDim define the LoadTile and are measured in elements
    if state["ProblemType"]["TLU%s"%tc]:
      parDim  = state["MacroTile%s"%tc]
      perpDim = depthU
    else:
      parDim  = depthU
      perpDim = state["MacroTile%s"%tc]

    if dbFract:
        print("\ninfo: %s Fractional MT%u_%u_%u Par=%u Perp=%u WG%02u_%02u_%02u NumThreads=%u GRWV=%u" \
          % (tc, state["MacroTile0"], state["MacroTile1"], depthU, \
            parDim, perpDim, \
            state["WorkGroup"][0], state["WorkGroup"][1], state["LocalSplitU"], \
            state["NumThreads"], state["GlobalReadVectorWidth"]))

    # Try to find a GRVW which is smaller than the LSC and also does not force
    # the LSC to wrap - both of these conditions can be tested with lsc % grvw ==0.
    # Each iteration divides GRWV by 2 which provides finer granularity
    # and a possible opportunity to handle the lsc
    grvw = state["GlobalReadVectorWidth"]
    minGrvw = 2 if state["ProblemType"]["DataType"].isHalf() and \
                globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"] else 1
    # TODO- check this for int8 and fractional load
    # minGrvw = 4 if state["ProblemType"]["DataType"].isInt8() and \
    #             globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"] else 1
    bestVw = -1
    while grvw >= minGrvw:
      # Per instruction across the entire group:
      elementsLoadedPerInst = state["NumThreads"]*grvw
      if (state["DirectToVgpr%s"%tc] and state["ProblemType"]["TLU%s"%tc]):
        elementsLoadedPerInst //= state["MatrixInstK"]
      # LSC, LSP - #elements loaded along specified dim with each load
      if parDim >= elementsLoadedPerInst:
        # entire work-group can work on (part) of the same row
        # DirectToVgpr case, LSC is limited to elementsLoadedPerInst // state["MatrixInstK"]
        state["LSC%s"%tc] = elementsLoadedPerInst
        state["LSP%s"%tc] = 1 if not (state["DirectToVgpr%s"%tc] and state["ProblemType"]["TLU%s"%tc]) else state["MatrixInstK"]
        state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim , state["LSC%s"%tc])
        state["NumLoadsPerpendicular%s"%tc] = 1
      else:
        # work-group exceeds read dimension so wraps to multiple rows
        state["LSC%s"%tc] = parDim
        state["LSP%s"%tc] = min(perpDim, elementsLoadedPerInst // parDim)
        state["NumLoadsCoalesced%s"%tc] = 1
        state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

      # Vector loads can't wrap to next P dim, so LSC must be divisible by vector elements;
      if dbFract:
        print("  lsc search : lsc(%u) %% grvw(%u) = %u (?0)" % (state["LSC%s"%tc], grvw, state["LSC%s"%tc] % grvw))
      if state["LSC%s"%tc] % grvw == 0:
        bestVw = grvw
        # Try to shrink GRVW if possible while keeping same LSC and LSP:
        # For example, avoid cases where we use a GRVW=4 with many empty addresses
        # when a GRVW=1 will do instead.
        validElementsLoadedPerInst = state["LSC%s"%tc] * state["LSP%s"%tc]
        grvw //= 2
        while grvw >= minGrvw:
          elementsLoadedPerInst = state["NumThreads"]*grvw
          if state["DirectToVgpr%s"%tc] and state["ProblemType"]["TLU%s"%tc]:
            elementsLoadedPerInst //= state["MatrixInstK"]
          if elementsLoadedPerInst < validElementsLoadedPerInst:
            break # Went too far, not enough load elements at this VW
          if state["LSC%s"%tc] % grvw == 0:
            if dbFract:
              print("  stepdown success (valid)elementsLoadedPerInst=", validElementsLoadedPerInst, "/", elementsLoadedPerInst, "grvw=", grvw, "lsc=", state["LSC%s"%tc])
            bestVw = grvw
          grvw //= 2
        break

      # TODO - could have this generate dwordx3 loads in addition, step down by 1 instead of div2
      # Would need to change asm code gen to generate x3
      grvw //= 2
      # end-- while loop

    if bestVw == -1:
      if dbFract:
        print ("reject fractional - no acceptable tile dim? GlobalReadVectorWidth", \
         state["GlobalReadVectorWidth"])
      return False  # could not find a solution, perhaps only possible for half ?

    state["GlobalLoadVectorWidth%s"%tc] = bestVw
    if bestVw != state["GlobalReadVectorWidth"]:
      if dbFract:
        print("  reducing GlobalLoadVectorWidth%s from %u to %u" \
            % (tc, state["GlobalReadVectorWidth"], bestVw))

    # How many loads per threads in each dimension.
    # threads which are outside the global read tile bounds will be clipped
    # in the assembly code generator.
    # Multiply the LSC*GRVW
    state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim, state["LSC%s"%tc])
    state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

    nlc = state["NumLoadsCoalesced%s"%tc]
    nlp = state["NumLoadsPerpendicular%s"%tc]

    # LoadTile must at least cover the MacroTile:
    assert(nlc*state["LSC%s"%tc] >= parDim)
    assert(nlp*state["LSP%s"%tc] >= perpDim)

    perpOverhang = perpDim % state["LSP%s"%tc]
    state["fractionalPerpOverhang%s"%tc] = perpOverhang
    if dbFract:
      # how many threads compute Global Read Offsets (GRO) that are not used
      print("  PerLoadTile=%ux%u elements Loads/WI=%ux%u LoadTile/WI=%ux%u (MT=%ux%u), %u/%u = %.1f%% WI GRO used %s" \
          % (state["LSC%s"%tc], state["LSP%s"%tc], \
             nlc, nlp, \
             nlc*state["LSC%s"%tc], nlp*state["LSP%s"%tc], \
             parDim, perpDim, \
             parDim*perpDim, \
             nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc], \
             float(parDim*perpDim), \
             float(nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc]) * 100.0) \
             )

      for p in range(0,nlp):
        elementWidth = 4
        if p != nlp-1:
          perp = state["LSP%s"%tc]
        else:
          perp = perpOverhang if perpOverhang else state["LSP%s"%tc]

        validElements = state["LSC%s"%tc] * perp
        print("  buffer_load_element_x%u %ux%ux%u bytes,  %u/%u valid GRO" %\
              (state["GlobalLoadVectorWidth%s"%tc], \
              state["LSC%s"%tc], perp, \
              elementWidth, \
              validElements//state["GlobalLoadVectorWidth%s"%tc],
              state["NumThreads"]))

    return True


  @staticmethod
  def MatrixInstructionToMIParameters(state):
    if len(state["MatrixInstruction"]) == 9:
      mi                          = state["MatrixInstruction"]
      state["MatrixInstruction"]  = [state["MatrixInstruction"][0],state["MatrixInstruction"][1],state["MatrixInstruction"][2],state["MatrixInstruction"][3]]

      waves                       = mi[7]* mi[8]
      miwg0                       = mi[4] * mi[0] * mi[7]
      state["WorkGroup"][0]       = miwg0
      state["WorkGroup"][1]       = waves*state["WavefrontSize"] // state["WorkGroup"][0]
      state["ThreadTile"][0]      = 1  # dummy
      state["ThreadTile"][1]      = 1  # dummy

      state["MFMA_BF16_1K"] = False
      if not state["ProblemType"]["SparseA"]:
        miDataType = state["ProblemType"]["DataType"] if (not state["EnableF32XdlMathOp"]) else state["ProblemType"]["F32XdlMathOp"]
        if not (miDataType.toChar() in validMFMA and \
          state["MatrixInstruction"] in validMFMA[miDataType.toChar()]):
          if miDataType.isBFloat16() and \
            state["MatrixInstruction"] in validMFMA["B1k"]:
            state["MFMA_BF16_1K"] = True
          else:
            reject(state, "MatrixInstruction %s not valid for DataType %s" % (state["MatrixInstruction"], miDataType))
      else:
        if not (state["ProblemType"]["DataType"].toChar() in validSMFMA and \
          state["MatrixInstruction"] in validSMFMA[state["ProblemType"]["DataType"].toChar()]):
          reject(state, "Sparse MatrixInstruction %s not valid for DataType %s" % (state["MatrixInstruction"], state["ProblemType"]["DataType"]))

      # set EnableMatrixInstruction
      state["EnableMatrixInstruction"] = True

      # set MIBlock
      MIBlock_BM = miwg0 // mi[0]
      MIBlock_BM = min(MIBlock_BM, mi[3])
      MIBlock_BN = mi[3] // MIBlock_BM

      state["MIBlock"]    = [32, 32, 2, 1, 1, 1]
      state["MIBlock"][0] = mi[0]
      state["MIBlock"][1] = mi[1]
      state["MIBlock"][2] = mi[2]
      state["MIBlock"][3] = mi[3]
      state["MIBlock"][4] = MIBlock_BM
      state["MIBlock"][5] = MIBlock_BN

      # set MIWaveGroup
      state['MIWaveGroup']     = [1, 1]
      state['MIWaveGroup'][0]  = min((miwg0 // mi[0]) // MIBlock_BM, waves)
      state['MIWaveGroup'][1]  = waves // state['MIWaveGroup'][0]

      # set MIWaveTile
      state['MIWaveTile']      = [1, 1]
      state['MIWaveTile'][0]   = mi[5]
      state['MIWaveTile'][1]   = mi[6]
      # set MIInputPerThread
      state['MIInputPerThread']  = mi[0] * mi[2] * mi[3] // 64
      state['MIInputPerThreadA'] = state['MIInputPerThread'] if not state["ProblemType"]["SparseA"] else state['MIInputPerThread']//2
      state['MIInputPerThreadB'] = state['MIInputPerThread']
      state['MIInputPerThreadMetadata'] = state['MIInputPerThread'] if not state["ProblemType"]["SparseA"] else state['MIInputPerThread']//8
    elif state["MatrixInstruction"] != [] and len(state["MatrixInstruction"]) == 4:
      state["EnableMatrixInstruction"] = True
    else:
      state["EnableMatrixInstruction"] = False


  ##############################################
  # check and calculate Wave Separate Global Read
  @staticmethod
  def checkAndAssignWaveSeparateGlobalRead(state, tc):
    # check can we use WaveSeparateGlobalRead
    numOfWaves = state["NumThreads"] // state["WavefrontSize"]
    if state["WaveSeparateGlobalRead%s"%tc]:
      if state["ProblemType"]["TLU%s"%tc] and (state["DepthU"] > 0) and (state["DepthU"] % numOfWaves != 0):
        reject(state, "didn't support WaveSeparateGlobalRead when DepthU is not multiple of wave %u in TLU%s" % (state["DepthU"], tc))
      if not state["ProblemType"]["TLU%s"%tc] and (state["MacroTile%s" % tc] % numOfWaves != 0):
        reject(state, "didn't support WaveSeparateGlobalRead when MacroTile is not multiple of wave %u in TLU%s" % (state["MacroTile%s"%tc], tc))


  ########################################
  # determine can we use DirectToVgpr
  @staticmethod
  def isDirectToVgprDoable(state, tc, depthU):
    tcOther = 'B' if tc == 'A' else 'B'
    MIindex = 0 if tc == 'A' else 1
    # With MatrixInstruction only (tentative)
    if not state["EnableMatrixInstruction"] :
      reject(state, "DirectToVgpr is for MatrixInstruction only")
      return False

    # Double/DoubleComplex only (tentative)
    if not (state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isDoubleComplex()):
      reject(state, "so far, DirectToVgpr is for double or double complex only")
      return False

    # Does not work with TLU = False and PrefetchLocalRead = 0
    if (not state["ProblemType"]["TLU%c"%tc]) and state["PrefetchLocalRead"] == 0:
      reject(state, "DirectToVgpr%c does not supports TLU%c = False and PrefetchLocalRead = 0"%(tc, tc))
      return False

    # MIWaveGroup check
    #  for A, MIWaveGroup should be [4, 1]
    #  for B, MIWaveGroup should be [1, 4]
    # This is to limit the number of Vgpr
    if tc == 'A' and not (state['MIWaveGroup'][0] == 4 and state['MIWaveGroup'][1] == 1):
      reject(state, "MIWaveGroup should be [4, 1] for DirectToVgprA. Current value is [%s]"%state['MIWaveGroup'])
      return False
    if tc == 'B' and not (state['MIWaveGroup'][0] == 1 and state['MIWaveGroup'][1] == 4):
      reject(state, "MIWaveGroup should be [1, 4] for DirectToVgprB. Current value is [%s]"%state['MIWaveGroup'])
      return False

    # Does not work with WaveSeparateGlobalRead
    if state["WaveSeparateGlobalRead%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports WaveSeparateGlobalRead%c"%(tc, tc))
      return False

    # Does not work with TLU and NumLoadsCoalesced != MIWaveTile / GlobalLoadVectorWidth
    # (only for FractionalLoad = False)
    if state["FractionalLoad"] == False:
      if state["ProblemType"]["TLU%s"%tc] and state["NumLoadsCoalesced%c"%tc] != state['MIWaveTile'][MIindex] / state["GlobalLoadVectorWidth%c"%tc]:
        reject(state, "DirectToVgpr%c does not supports NumLoadsCoalesced%c(=%u) != MIWaveTile[%u](=%u) / GlobalLoadVectorWidth%c(=%u)"\
                       %(tc, tc, state["NumLoadsCoalesced%c"%tc], MIindex, state['MIWaveTile'][MIindex], tc, state["GlobalLoadVectorWidth%c"%tc]))
        return False
    # Does not work with TLU and MIWaveTile < GlobalLoadVectorWidth
    if state["ProblemType"]["TLU%s"%tc] and state['MIWaveTile'][MIindex] < state["GlobalLoadVectorWidth%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports MIWaveTile[%u](=%u) < GlobalLoadVectorWidth%c(=%u)"\
                     %(tc, MIindex, state['MIWaveTile'][MIindex], tc, state["GlobalLoadVectorWidth%c"%tc]))
      return False

    # Does not work with ExpandPointerSwap = False
    if not state["ExpandPointerSwap"]:
      reject(state, "DirectToVgpr%c does not supports ExpandPointerSwap = False"%(tc))
      return False

    # Does not work with VectorWidth != GlobalReadVectorWidth (VW = 2 + GRVW = 1 or VW = 1 + GRVW = 2 does not work)
    if state["VectorWidth"] != state["GlobalLoadVectorWidth%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports VectorWidth(=%u) != GlobalReadVectorWidth%c(%u)"%(tc, state["VectorWidth"], tc, state["GlobalLoadVectorWidth%c"%tc]))
      return False

    # Does not work with AssertFree1ElementMultiple % VectorWidth != 0 (edge shift case) for B
    if tc == 'B' and state["AssertFree1ElementMultiple"] % state["VectorWidth"] != 0:
      reject(state, "DirectToVgpr%c does not supports AssertFree1ElementMultiple %% VectorWidth != 0"%(tc))
      return False

    # Does not work with FractionalLoad and (not TLU)
    if state["FractionalLoad"] and (not state["ProblemType"]["TLU%c"%tc]):
      reject(state, "DirectToVgpr%c does not supports FractionalLoad + TLU=False"%(tc))
      return False

    # Does not work with TLU=False and PGR=2 and DepthU<=MatrixInstK*VW
    if (not state["ProblemType"]["TLU%c"%tc]) and state["PrefetchGlobalRead"] == 2 and depthU <= state["MatrixInstK"] * state["VectorWidth"]:
      reject(state, "DirectToVgpr%c does not supports TLU=False and PrefetchGlobalRead==2 and DepthU<=MatrixInstK*VectorWidth"%(tc))
      return False

    # Does not work with TLU=False and NumLoadsCoalesced != DepthU//(MatrixInstK*VW)
    if (not state["ProblemType"]["TLU%c"%tc]) and state["NumLoadsCoalesced%c"%tc] != depthU // (state["MatrixInstK"] * state["VectorWidth"]):
      reject(state, "DirectToVgpr%c does not supports TLU=False and NumLoadsCoalesced%c != DepthU//(MatrixInstK*VectorWidth)"%(tc, tc))
      return False

    # Both TLU=False + TransposeLDS case, need GlobalLoadVectorWidth == LocalReadVectorWidth
    if (not state["ProblemType"]["TLU%c"%tc]) and (not state["ProblemType"]["TLU%c"%tcOther]) and state["TransposeLDS"] and \
       state["GlobalLoadVectorWidth%c"%tc] != state["LocalReadVectorWidth"]:
      reject(state, "DirectToVgpr%c does not supports TLUA=False and TLUB=False and GlobalLoadVectorWidth%c != LocalReadVectorWidth"%(tc, tc))
      return False

    # Does not work with TLU=False and PrefetchLocalRead=1 and VectorWidth>1
    if (not state["ProblemType"]["TLU%c"%tc]) and state["PrefetchLocalRead"] == 1 and state["VectorWidth"] > 1:
      reject(state, "DirectToVgpr%c does not supports TLU=False and PrefetchLocalRead=1 and VectorWidth>1)"%(tc))
      return False

    # Does not work with TLU=False and VectorWidth>1 and other side of TLU=True
    if (not state["ProblemType"]["TLU%c"%tc]) and state["VectorWidth"] > 1 and state["ProblemType"]["TLU%c"%tcOther]:
      reject(state, "DirectToVgpr%c does not supports TLU%c=False VectorWidth>1 and TLU%c=True"%(tc, tc, tcOther))
      return False

    # Does not work with DirectToLDS
    # -> this will be checked after DirectToLDS doable check is done

    return True

  ########################################
  # determine can we use DirectToLds
  @staticmethod
  def isDirectToLdsDoable(state, tc):
    numBytes = state["ProblemType"]["DataType"].numBytes()

    # x2/x4 support for directToLds

    # numelements_perlane = 4/numBytes
    # TN with transposeLDS feature should work as long as state["AssertSummationElementMultiple"] % (numelements_perlane*2) = 0
    #                                                     state["AssertSummationElementMultiple"] % (numelements_perlane*4) = 0

    #NT
    # use only for f32 & DGEMM and TLU = 1
    #TN
    # use for all precisions with TransposeLDS=1

    if state["ProblemType"]["DataType"].isHalf() and state["AssertSummationElementMultiple"] % (2 * state["GlobalLoadVectorWidth%c"%tc])  != 0:
      reject(state, "can't use DirectToLds for FP16 with AssertSummationElementMultiple %u" % state["AssertSummationElementMultiple"])
      return False

    if state["ProblemType"]["DataType"].isBFloat16() and state["AssertSummationElementMultiple"] % (2 * state["GlobalLoadVectorWidth%c"%tc]) != 0:
      reject(state, "can't use DirectToLds for BF16 with AssertSummationElementMultiple %u" % state["AssertSummationElementMultiple"])
      return False

    if state["NumThreads"] % state["WavefrontSize"] != 0:
      reject(state, "can't use DirectToLds for NumThreads % WavefrontSize != 0")
      return False

    # GLVW*BPe only for precision(s) < 4 (bpe)
    #if (state["ProblemType"]["TLU%c"%tc] == True and numBytes < 4):
    if (numBytes < 4):
      if state["GlobalLoadVectorWidth%c"%tc] * numBytes != 4:
        reject(state, "can't use DirectToLds for bpe < 4 and GlobalLoadVectorWidth * numBytes != 4"%tc)
        return False

    if state["ProblemType"]["TLU%c"%tc] == state["UnrollMajorLDS%c" % tc]:
      reject(state, "can't use DirectToLds for TLU%c == UnrollMajorLDS%c"%(tc, tc))
      return False

    # avoid picking x2&x4 for precisions < f32/f64 in [ProblemType][TLU] == TRUE
    if not state["EnableMatrixInstruction"]:
      if state["GlobalLoadVectorWidth%c"%tc] * numBytes * state["WavefrontSize"] > 256:
        reject(state, "can't use DirectToLds for not EnableMatrixInstruction and GlobalLoadVectorWidth%c * bpe * WavefrontSize > 256"%tc)
        return False

    # TODO revisit fp32 case for failure
    #if state["ProblemType"]["TLU%c"%tc] and numBytes < 8 and state["GlobalLoadVectorWidth%c"%tc] * numBytes > 4:
    if numBytes < 8 and state["GlobalLoadVectorWidth%c"%tc] * numBytes > 4:
      reject(state, "can't use DirectToLds for TLU%c and bpe < 8 and GlobalLoadVectorWidth%c * bpe > 4"%(tc, tc))
      return False


    if state["WaveSeparateGlobalRead%c" % tc]:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytes != state["WavefrontSize"] * state["GlobalLoadVectorWidth%c"%tc] * numBytes:
        reject(state, "can't use DirectToLds for LSC%c and LSP%c * bpe!= WavefrontSize * GlobalLoadVectorWidth%c * bpe > 4"%(tc, tc, tc))
        return False
    else:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytes != state["NumThreads"] * state["GlobalLoadVectorWidth%c"%tc] * numBytes:
        reject(state, "can't use DirectToLds for LSC%c and LSP%c * bpe != NumThreads * GlobalLoadVectorWidth%c * bpe > 4"%(tc, tc, tc))
        return False

    if (state["LdsBlockSizePerPad%c"%tc] == 0) \
        and (state["LdsPad%c"%tc] != 0):
#        and ((state["LSC%c"%tc] * numBytes) != (state["NumThreads"] * 4)): // TODO:
#        and ((state["LSC%c"%tc] * numBytes) % (state["WavefrontSize"] * 4) != 0):
      reject(state, "can't use DirectToLds for LdsBlockSizePerPad%c == 0 and LdsPad%c != 0"%(tc, tc))
      return False

    if (state["LdsBlockSizePerPad%c"%tc] != 0) \
        and (state["LdsPad%c"%tc] != 0) \
        and (state["LdsBlockSizePerPad%c"%tc] != state["WavefrontSize"] * state["GlobalLoadVectorWidth%c"%tc] * numBytes):
#        and (state["LdsBlockSizePerPad%tc"] % (state["WavefrontSize"] * 4) != 0): // TODO:
      reject(state, "can't use DirectToLds for LdsBlockSizePerPad%c != 0 and LdsPad%c != 0 and \
              LdsBlockSizePerPad%c != WavefrontSize * GlobalLoadVectorWidth%c * bpe"%(tc, tc, tc, tc))
      return False

    # so far, DirectToLds does not work well with PGR=2
    # performance is not good and a lot of ds_read for DTL can cause scheduling issue(need fix)
    if state["PrefetchGlobalRead"] == 2 and not (state["DirectToVgprA"] or state["DirectToVgprB"]):
      reject(state, "can't use DirectToLds for PrefetchGlobalRead == 2 without DirectToVgpr")
      return False

    # so far, DirectToLds does not work with LRVW=2
    if state["LocalReadVectorWidth"] == 2:
      reject(state, "can't use DirectToLds for LocalReadVectorWidth == 2")
      return False

    # Does not work with (NumLoadsCoalesced>1 and UseInstOffsetForGRO) + DGEMM
    if state["ProblemType"]["DataType"].isDouble() and \
      (state["NumLoadsCoalesced%c"%tc] > 1 and state["UseInstOffsetForGRO"]):
      reject(state, "DirectToLds%c does not supports NumLoadsCoalesced%c > 1 and UseInstOffsetForGRO for dgemm"%(tc, tc))
      return False

    # Does not work with NumLoadsCoalesced>1 + ZGEMM
    if state["ProblemType"]["DataType"].isDoubleComplex() and state["NumLoadsCoalesced%c"%tc] > 1:
      reject(state, "DirectToLds%c does not supports NumLoadsCoalesced%c > 1 for zgemm"%(tc, tc))
      return False

    # Does not work with PrefetchGlobalRead=2 and PrefetchLocalRead=1 (cannot schedule DTL global read after local read)
    if state["PrefetchGlobalRead"] == 2 and state["PrefetchLocalRead"] == 1:
      reject(state, "DirectToLds%c does not work with PrefetchGlobalRead=2 and PrefetchLocalRead=1"%(tc))
      return False

    # DirectToLds does not work if MacroTile is not power of 2
    # LDS offset swap/rotate logic works only when MacroTile is power of 2
    mt = state["MacroTile%c"%tc]
    if mt & (mt - 1) != 0:
      reject(state, "can't use DirectToLds if MacroTile%s is not power of 2"%tc)
      return False

    # DirectToLds does not work with TLU=False and bpe > bpr and DepthU//NumLoadsCoalesced < 8
    # bpe > bpr case, Lower and upper 4 bytes elements are stored separately.
    # if TLU=False and DepthU//NumLoadsCoalesced is smaller than lower block size (8 elements),
    # current offset swap logic does not work
    if (not state["ProblemType"]["TLU%c"%tc]) and state["ProblemType"]["DataType"].numRegisters() > 1 and \
       state["DepthU"] // state["NumLoadsCoalesced%c"%tc] < 8:
      reject(state, "DirectToLds%c does not work with TLU=False and bpe > bpr and DepthU//NumLoadsCoalesced%c < 8"%(tc, tc))
      return False

    return True

  @staticmethod
  def getDivisorName(state, tC):
    if state["GlobalReadCoalesceGroup{}".format(tC)]:
      if state["GlobalReadCoalesceVector{}".format(tC)]:
        divisorName = "LVC{}".format(tC)
      else:
        # Fractional load use the more accurate lsc, multiply by VW later
        divisorName = "LSC{}".format(tC)
    else:
      if state["GlobalReadCoalesceVector{}".format(tC)]:
        divisorName = "LSP{}".format(tC)
      else:
        divisorName = "LVP{}".format(tC)
    return divisorName

  ########################################
  # assign all derived parameters
  @staticmethod
  def assignDerivedParameters(state):

    state["EnableF32XdlMathOp"] = False #ignore the F32 xDL MathOp by default.
    #enable F32 xDL MathOp only when the input type is f32.
    if "F32XdlMathOp" in state["ProblemType"] \
       and (not state["ProblemType"]["F32XdlMathOp"].isSingle()) \
       and (state["ProblemType"]["DataType"].isSingle()):
      state["EnableF32XdlMathOp"] = True

    Solution.assignProblemIndependentDerivedParameters(state)

    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    for s in Solution.InternalKeys:
        state['_'+s] = state[s]
        #del state[s]

    if ("_GlobalAccumulation" not in state) or ("_WorkspaceSizePerElemC" not in state):
      state["_GlobalAccumulation"] = None
      state["_WorkspaceSizePerElemC"] = 0
      if state["GlobalSplitU"] > 1:
        computeName  = state["ProblemType"]["ComputeDataType"].toName()
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()

        if state["GlobalSplitUAlgorithm"] == 'SingleBuffer':
          if computeName != state["ProblemType"]["DestDataType"].toName():
            state["_GlobalAccumulation"] = 'SingleBuffer'
        elif state["GlobalSplitUAlgorithm"] == 'MultipleBuffer':
          state["_GlobalAccumulation"] = 'MultipleBuffer'

        if state["_GlobalAccumulation"] == 'SingleBuffer':
          state["_WorkspaceSizePerElemC"] = computeBytes
        elif state["_GlobalAccumulation"] == 'MultipleBuffer':
          state["_WorkspaceSizePerElemC"] = computeBytes * state["GlobalSplitU"]

    if("_WorkspaceSizePerElemBias" not in state):
      state["_WorkspaceSizePerElemBias"] = 0
      if state["ProblemType"]["UseBias"] and state["ProblemType"]["Gradient"]:
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
        if state["ProblemType"]["BiasSrc"] == "D" and (state["ProblemType"]["ComputeDataType"] != state["ProblemType"]["DestDataType"]):
          state["_WorkspaceSizePerElemBias"] = computeBytes
        elif state["GlobalSplitU"] > 1:
          if state["_GlobalAccumulation"] == 'SingleBuffer':
            state["_WorkspaceSizePerElemBias"] = computeBytes
          elif state["_GlobalAccumulation"] == 'MultipleBuffer':
            state["_WorkspaceSizePerElemBias"] = computeBytes * state["GlobalSplitU"]

    state["WorkspaceCheck"] = [state["_WorkspaceSizePerElemC"], state["_WorkspaceSizePerElemBias"]]

    if state["VectorStore"] == -1:
        state["_VectorStore"] = 1 # default, may be changed if needed to generate a valid kernel

    ProblemType.assignDerivedParameters(state["ProblemType"])
    if not state["Valid"]:
      print2("in assignDerivedParameters, state['Valid'] = False")
      return

    # Init LoopIters parameter in case of early exit
    # For backwards compatibility with older yaml files
    state["LoopIters"] = 0
    if "LoopUnroll" in state:
      state["LoopIters"] = state["LoopUnroll"]

    if state["ScheduleIterAlg"] == 2:
      state["InnerUnroll"] = state["DepthU"] // state["MatrixInstK"]
      state["PrefetchLocalRead"] = 1
      state["ExpandPointerSwap"] = 1
      state["1LDSBuffer"] = 1
      print2("\nSet SIA=2, force PrefetchLocalRead=1, ExpandPointerSwap=1, 1LDSBuffer=1")

    isa = tuple(state["ISA"])

    if state["WavefrontSize"] == 32 and not globalParameters["ArchCaps"][isa]["HasWave32"]:
      reject(state, "WavefrontSize=32 not supported for ISA {}".format(isa))

    if state["WavefrontSize"] == 32 and state["KernelLanguage"] == "Source":
      reject(state, "WavefrontSize=32 not yet supported for source kernels.")

    if state["EnableMatrixInstruction"]:
      if not (state["ProblemType"]["DataType"].isSingle() \
              or state["ProblemType"]["DataType"].isDouble() \
              or state["ProblemType"]["DataType"].isBFloat16() \
              or state["ProblemType"]["DataType"].isHalf() \
              or state["ProblemType"]["DataType"].isComplex() \
              or state["ProblemType"]["DataType"].isInt8()):
        reject(state, "didn't support Matrix Instruction with type %s" % str(state["ProblemType"]["DataType"]))
      if not state["MIBlock"] or len(state["MIBlock"]) != 6:
        reject(state, "invalid MIBlock")
      if not state["MIWaveGroup"] or len(state["MIWaveGroup"]) != 2:
        reject(state, "invalid MIWaveGroup")
      if not state["MIWaveTile"] or len(state["MIWaveTile"]) != 2:
        reject(state, "invalid MIWaveTile")
      if not state["ProblemType"]["HighPrecisionAccumulate"] \
         and state["ProblemType"]["DataType"].numRegisters() < 1 :
        reject(state, "Matrix instructions for half, bf16 (or i8) types are natively accumulated" + \
         " in fp32 (or i32) precision. Please add the following config:" + \
         "\n - HighPrecisionAccumulate: True")
    else:
      if not state["ProblemType"]["HighPrecisionAccumulate"] \
         and state["ProblemType"]["ComputeDataType"].numRegisters() > state["ProblemType"]["DataType"].numRegisters() :
        reject(state, "For non-MI Kernel, if sizeof(ComputeDataType) > sizeof(DataType), " + \
         "Please add the following config:" + \
         "\n - HighPrecisionAccumulate: True")
      if state["ProblemType"]["SparseA"]:
        reject(state, "Sparse A problem is only supported by SMFMA MI kernel.")

      if state["ThreadTile0"] > 16 or state["ThreadTile1"] > 16:
        reject(state, "Invalid value for ThreadTile")

      if state["ScheduleIterAlg"] == 2 or state["ScheduleIterAlg"] == 3:
        reject(state, "SIA2 and SIA3 only support MatrixInstruction")

    if state["ProblemType"]["Tensor0"]==0:
      state["ThreadTileA"] = state["ThreadTile0"]
      state["ThreadTileB"] = state["ThreadTile1"]
      state["SubGroupA"] = state["SubGroup0"]
      state["SubGroupB"] = state["SubGroup1"]
      state["MacroTileA"] = state["MacroTile0"]
      state["MacroTileB"] = state["MacroTile1"]
      if state["EnableMatrixInstruction"]:
        state["MIWaveTileA"] = state["MIWaveTile"][0]
        state["MIWaveTileB"] = state["MIWaveTile"][1]
    else:
      state["ThreadTileB"] = state["ThreadTile0"]
      state["ThreadTileA"] = state["ThreadTile1"]
      state["SubGroupB"] = state["SubGroup0"]
      state["SubGroupA"] = state["SubGroup1"]
      state["MacroTileB"] = state["MacroTile0"]
      state["MacroTileA"] = state["MacroTile1"]
      if state["EnableMatrixInstruction"]:
        state["MIWaveTileA"] = state["MIWaveTile"][1]
        state["MIWaveTileB"] = state["MIWaveTile"][0]

    Solution.checkAndAssignWaveSeparateGlobalRead(state, 'A')
    Solution.checkAndAssignWaveSeparateGlobalRead(state, 'B')

    if state["ProblemType"]["SparseA"]:
      if not state["DirectToVgprSparseMetadata"]:
        state["ThreadTileMetadata"] = state["ThreadTileA"]
        state["SubGroupMetadata"] = state["SubGroupA"]
        state["MacroTileMetadata"] = state["MacroTileA"]
        state["WaveSeparateGlobalReadMetadata"] = state["WaveSeparateGlobalReadA"]
        Solution.checkAndAssignWaveSeparateGlobalRead(state, 'Metadata')
        state["DirectToLdsMetadata"] = False
        state["DirectToVgprMetadata"] = False
        state["LocalWriteUseSgprMetadat"] = False
        state["ProblemType"]["MirrorDimsMetadata"]  = state["ProblemType"]["MirrorDimsA"]
      if state["EnableMatrixInstruction"]:
        state["MIWaveTileMetadata"] = state["MIWaveTileA"]
    elif not state["ProblemType"]["SparseA"]:
      state["DirectToVgprSparseMetadata"] = False
      state["MIWaveTileMetadata"] = 0

    # Init vars early since there are early-exit return statements below
    state["DirectToLdsA"] = False
    state["DirectToLdsB"] = False
    state["LocalWriteUseSgprA"] = False
    state["LocalWriteUseSgprB"] = False

    state["WorkGroupMapping" ] = abs(state["WorkGroupMapping"])

    problemType = state["ProblemType"]

    for (tc,batchMask) in (('A', 0x1), ('B', 0x2)):
      freeDims = [i for i in problemType["IndexAssignments%s"%tc] if i in problemType["IndicesFree"]]
      if not freeDims:
        reject(state, "tensor%s contains no free indices.")
        return False

    # Determine which indices will be packed together as this impacts several different parms (sizes, magic numbers, etc)
    # The order in PackedC*Indices also determines the order that dimensions are packed - the first elements in
    # the list are the fastest-moving elements.
    # The store code optimizes for C0 being the coalesced dimension and C1 the perp dimension.
    # C0/C1 indices can come from IndexAssignmentsA or IndexAssignmentsB
    # grid size [0,1]
    state["PackedC0IdxChars"] = []
    state["PackedC0IndicesX"] = []
    indexChars = globalParameters["IndexChars"]
    # Pack all the dimensions (free) of A into grid[0]

    if problemType["Index0"] in problemType["IndexAssignmentsA"]:
      tc0 = 'A'
      tc1 = 'B'
    else:
      tc0 = 'B'
      tc1 = 'A'
    assert(isPackedIndex(state, problemType["Index01A"]))
    assert(isPackedIndex(state, problemType["Index01B"]))

    # Pack all the dimensions (batch and free) of A into grid[0]
    for idx in problemType["IndexAssignments%s"%tc0]:
      if isPackedIndex(state, idx):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC0IdxChars"].append("%s" % indexChars[idx])
        state["PackedC0IndicesX"].append(idx)

    state["PackedC1IdxChars"] = []
    state["PackedC1IndicesX"] = []
    for idx in problemType["IndexAssignments%s"%tc1]:
      if isPackedIndex(state, idx):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC1IdxChars"].append("%s" % indexChars[idx])
        state["PackedC1IndicesX"].append(idx)

    # If dims are packed, then need to ensure a global vector load isn't split by a tensor dim
    # (since this could result in non-contiguous addresses)
    # Current implementation ensures that the vector load is not partial across the Free* boundary:
    # GlobalLoadVectorWidth=1 will always meet this requirement.
    # (TODO - could make this more sophisticated if dims use default strides and are thus contiguous)
    packedC0 = len(state["PackedC0IdxChars"])>1
    packedC1 = len(state["PackedC1IdxChars"])>1

    bufferLoad = state["BufferLoad"] and state["KernelLanguage"] == "Assembly"
    if not bufferLoad:
      state["DirectToLds"] = False
      state["_UseSgprForGRO"] = False
      if state["PrefetchGlobalRead"] == 2:
        reject(state, "BufferLoad=0 does not support PrefetchGlobalRead=2")
        return

      if problemType["UseBias"]:
        reject(state, "BufferLoad=0 does not support UseBias due to no suppress no load.")
        return

    #These modes only work under certain conditions, apply them here:
    #  - The "NoLoad" loop is only generated if PrefetchGlobalRead>0
    #  - And Suppress does not work if GSU>1 for some reason
    if state["SuppressNoLoadLoop"] == 1:
      if not (bufferLoad and state["PrefetchGlobalRead"] == 1 and (state["GlobalSplitU"]==1)):
        state["SuppressNoLoadLoop"] = 0

    if state["ExpandPointerSwap"] == 1:
      # Pointer swap only used if PGR==1 or (PGR>1 and double/double complex) - so set ExpandPointerSwap=0 here
      # So far, EPS=1 and PGR>1 works only with double/double complex.
      # DirectToVgpr case, bufferLoad=False can work with ExpandPointerSwap=1
      #if not (bufferLoad and state["PrefetchGlobalRead"] == 1):
      if not ((bufferLoad or state["DirectToVgprA"] or state["DirectToVgprB"]) and ( state["PrefetchGlobalRead"] == 1 \
              or (state["PrefetchGlobalRead"] > 1 and \
                  (state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isDoubleComplex()))
              or (state["ProblemType"]["SparseA"] and state["PrefetchGlobalRead"] > 0))):
        state["ExpandPointerSwap"] = 0
      # EPS not supported with SplitLDS yet
      if state["DepthULdsDivisor"] > 1:
        state["ExpandPointerSwap"] = 0

    #print("PackedC0IdxChars", state["PackedC0IdxChars"])
    #print("PackedC1IdxChars", state["PackedC1IdxChars"])

    # Set up stagger shift:
    bpeAB = int(4*state["ProblemType"]["DataType"].numRegisters())
    # (1<<staggerStrideShift) is number of loop iterations to traverse the stride
    if state["StaggerU"] == 0:
      state["StaggerUMapping"] = 0
      state["StaggerUStride"] = 0
    try:
        staggerStrideShift = (int)(math.ceil(math.log(state["StaggerUStride"] / \
                (state["DepthU"] * bpeAB), 2)))
    except ValueError: # i.e., StaggerUStride == 0
        staggerStrideShift = 0
    if staggerStrideShift < 0:
      reject(state, "StaggerUStride=%u is less than size of DepthU=%u * BytesPerElement=%u" \
        % (state["StaggerUStride"], state["DepthU"], bpeAB))
      return
    #print "staggerStrideShift=", staggerStrideShift, "depthu=", state["DepthU"]
    state["_staggerStrideShift"] = staggerStrideShift

    state["UnrollMajorLDSA"]     = state["TransposeLDS"] and (not state["ProblemType"]["TLUA"])
    state["UnrollMajorLDSB"]     = state["TransposeLDS"] and (not state["ProblemType"]["TLUB"])

    # VectorWidth default handling
    if state["VectorWidth"] < 1:
      if state["EnableMatrixInstruction"]:
        regPerElem = state["ProblemType"]["DataType"].numRegisters()
        if state["SourceSwap"] and not state["UnrollMajorLDSA"]:
          optVW = int(4 // regPerElem)
          while 1:
            if state["MIWaveTile"][0] % optVW == 0:
              state["VectorWidth"] = optVW
              break
            else:
              optVW //= 2
        else:
          state["VectorWidth"] = 1
      else:
        state["VectorWidth"] = int(4 / state["ProblemType"]["DataType"].numRegisters())
        while state["ThreadTile0"] % state["VectorWidth"] != 0 \
            or state["ThreadTile1"] % state["VectorWidth"] != 0:
          state["VectorWidth"] //= 2

    # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
    if state["EnableMatrixInstruction"]:
      if state["SourceSwap"] and ((state["MIWaveTile"][0] % state["VectorWidth"]) != 0):
        reject(state, "MIWaveTile0(%u) should be multiple of VectorWidth(%u)" % (state["MIWaveTile"][0], state["VectorWidth"]))
        return
    else:
      if state["ThreadTile0"] % state["VectorWidth"] != 0 \
          or state["ThreadTile1"] % state["VectorWidth"] != 0:
        reject(state, "ThreadTile0 %u or ThreadTile1 %u not a multiple of VectorWidth %u" \
            % (state["ThreadTile0"], state["ThreadTile1"], \
            state["VectorWidth"]))
        return

    if len(problemType["IndicesSummation"]) > 1:
      # not supported with multiple summations, bug is maybe something with
      # how stagger iteration is wrapped when unroll loop exits
      state["StaggerU"] = 0

    # Some restrictions for half:
    if state["KernelLanguage"] == "Assembly" \
      and state["ProblemType"]["DataType"].isHalf():

      # Vector-width must be at least 2 for Half (since unroll loop uses packed operations?)
      if (not state["EnableMatrixInstruction"]) and state["VectorWidth"] < 2:
        reject(state, "VectorWidth must be >= 2 for half")
      if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] and state["AssertFree0ElementMultiple"] % 2 != 0:
          # beta-on-edge has AF0EM requirement except for HPA kernels
          reject(state, "Archs with HasEccHalf require AF0EM%2==0 except for HPA kernels")

    # Some restrictions for int8:
    if state["KernelLanguage"] == "Assembly" and state["ProblemType"]["DataType"].isInt8():
      if (not state["EnableMatrixInstruction"]) and state["VectorWidth"] < 4:
        reject(state, "VectorWidth must be >= 4 for Int8")

    # Default GlobalReadVectorWidth
    if state["GlobalReadVectorWidth"] == -1:
      if state["EnableMatrixInstruction"]:
        curGRVW = 1/state["ProblemType"]["DataType"].numRegisters()
        state["GlobalReadVectorWidth"] = int(curGRVW)
        wlrMultiple = state["LocalReadVectorWidth"] // state["MIInputPerThread"]
        if state["DepthU"] // state["MatrixInstK"] == wlrMultiple:
          optGRVW = 2
          state["LocalReadVectorWidth"] //= 2
        else:
          optGRVW = 4
        while (curGRVW * state["ProblemType"]["DataType"].numRegisters() < optGRVW):
          curGRVW *= 2
          if (state["MacroTile0"]*state["DepthU"]//state["NumThreads"]) % curGRVW == 0 and (state["MacroTile1"]*state["DepthU"]//state["NumThreads"]) % curGRVW == 0:
            state["GlobalReadVectorWidth"] = int(curGRVW)
      else:
        state["GlobalReadVectorWidth"] = state["VectorWidth"]

    # Default GlobalStoreVectorWidth
    if state["StoreVectorWidth"] == -1:
      #TODO : re-enable later after running testlists
      #state["StoreVectorWidth"] = state["VectorWidth"]
      # use wider store for best store optimization
      if state["SourceSwap"]:
        state["StoreVectorWidth"] = state["VectorWidth"]
      elif state["ProblemType"]["DataType"].numRegisters() <= 1:
        state["StoreVectorWidth"] = 4
      else:
        state["StoreVectorWidth"] = 4//state["ProblemType"]["DataType"].numRegisters()

    if state["EnableMatrixInstruction"]:
      if state["SourceSwap"]:
        if ((state["VectorWidth"] % state["StoreVectorWidth"]) != 0):
          reject(state, "MFMA SourceSwap mode doesn't support vw(%u) with svw(%u)" % (state["VectorWidth"], state["StoreVectorWidth"]))
          return
      else:
        if ((state["MIOutputVectorWidth"] % state["StoreVectorWidth"]) != 0):
          reject(state, "MFMA non-SourceSwap mode doesn't support miovw(%u) with svw(%u)" % (state["MIOutputVectorWidth"], state["StoreVectorWidth"]))
          return

    # reject - VW too big
    if (state["VectorWidth"] * state["ProblemType"]["DataType"].numBytes()) > 16:
      reject(state, "VW * DataType.numBytes() > 16")
      return

    # reject - GRVW too big
    if (state["GlobalReadVectorWidth"] * state["ProblemType"]["DataType"].numBytes()) > 16:
      reject(state, "GRVW * DataType.numBytes() > 16")
      return

    # LocalSplitU too large?
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]

    if numElementsPerWorkGroup < state["NumThreads"]:
      reject(state, "NumElementsPerWorkGroup %u < NumThreads %u; reduce LocalSplitU" \
          % (numElementsPerWorkGroup, state["NumThreads"]))
      return

    state["NumElementsPerThread"] = numElementsPerWorkGroup // state["NumThreads"]
    state["GlobalWriteVectorWidth"] = min(state["VectorWidth"], state["NumElementsPerThread"] )
    if state["NumElementsPerThread"] % state["GlobalWriteVectorWidth"] != 0:
      reject(state, "LSU NumElementsPerThread %u not divisible into GWVW %u" \
          % (state["NumElementsPerThread"], state["GlobalWriteVectorWidth"]))
      return
    state["NumGlobalWriteVectorsPerThread"] = state["NumElementsPerThread"] \
        // state["GlobalWriteVectorWidth"]


    # LocalSplitU but can't NumThreads%MacroTile doesn't support sideways store
    if state["LocalSplitU"] > 1:
      if state["NumThreads"] % state["MacroTile0"] != 0:
        reject(state, "LocalSplitU but NumThreads=%u not divisible by MT0=%u for sideways store" \
            % (state["NumThreads"], state["MacroTile0"]))
        return
      if state["MacroTile0"]*state["MacroTile1"] % state["NumThreads"] != 0:
        reject(state, "LocalSplitU but MT0*MT1=%u elements doesn't divide into NumThreads=%u" \
            % (state["MacroTile0"]*state["MacroTile1"], state["NumThreads"]))
        return
      if state["ProblemType"]["DataType"].isInt8():
        reject(state, "int8 doesn't support LocalSplitU")
        return

    # GlobalSplitU doesn't work with some other things:
    if state["GlobalSplitU"] > 1:
      # added GSU support for DGEMM
      supported = \
        (state["ProblemType"]["DataType"].isSingle()) or \
        (state["ProblemType"]["DataType"].isDouble() and state["BufferStore"]) or \
        (state["ProblemType"]["DestDataType"].isInt32()) or \
        (state["KernelLanguage"] == "Assembly" and
            (state["ProblemType"]["DataType"].isHalf() and not state["ProblemType"]["HighPrecisionAccumulate"]) or
            (state["_GlobalAccumulation"])
        )
      if not supported:
        reject(state, "GlobalSplitU only compatible with single or asm and (half or mixed) precision")
        return

    if state["ProblemType"]["DataType"].isHalf() and state["KernelLanguage"] == "Assembly":

      if (not state["EnableMatrixInstruction"]) and state["VectorWidth"] < 2:
        reject(state, "Assembly half requires VectorWidth >= 2 for non-MFMA mode")

      if state["GlobalSplitU"] > 1 and (not state["_GlobalAccumulation"]):
        if state["AssertFree0ElementMultiple"] < 2:
          reject(state, "Assembly GSU half requires AF0EM>=2 (for atomics on edge tiles)")

    ########################################
    # Initial DepthU
    ########################################
    userDepthU = state["DepthU"]
    # DepthU == -1 means glvw=1
    if state["DepthU"] == -1:
      if state["MacroTile0"] != state["MacroTile1"]:
        reject(state, "DepthU=0 requires square MacroTile")
        return

    if userDepthU < 0:
      depthU     = 2
      depthULds  = 2
      maxDepthU  = globalParameters["MaxDepthU"]
      numOfWaves = state["NumThreads"] // state["WavefrontSize"]
      if state["ProblemType"]["TLUA"] and state["WaveSeparateGlobalReadA"]:
        depthU = max(depthU, numOfWaves)
      if state["ProblemType"]["TLUB"] and state["WaveSeparateGlobalReadB"]:
        depthU = max(depthU, numOfWaves)
    else:
      depthU = userDepthU
      depthULds = userDepthU//state["DepthULdsDivisor"]
      maxDepthU = userDepthU

    depthUA = depthU if not state["ProblemType"]["SparseA"] else depthU//2
    depthUB = depthU

    depthUM = depthUA // 4 if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"] else depthUA

    state["_DepthULds"] = state["DepthU"]//state["DepthULdsDivisor"] # internal
    state["_DepthULdsA"] = depthUA//state["DepthULdsDivisor"] # internal
    state["_DepthULdsB"] = depthUB//state["DepthULdsDivisor"] # internal
    state["_DepthULdsMetadata"] = depthUM//state["DepthULdsDivisor"] # internal

    ########################################
    # Search DepthU
    # Inputs:
    #  - depthU, userDepthU, state["LocalSplitU"], state["InnerUnroll"], state["MacroTile0/1"], state["GlobalReadVectorWidth"]
    #  - state["MatrixInstK"], ...
    # Outputs:
    #  - totalVectorsCoalescedA, totalVectorsCoalescedB, totalElementsPerpA, totalElementsPerpB, state["DepthU"]
    #######################################
    while True: # exit criteria at end
      validDepthU = True
      # peek LoopIters
      loopIters = (depthULds // state["LocalSplitU"]) // state["InnerUnroll"]
      if "MatrixInstK" in state:
        loopIters //= state["MatrixInstK"]
      if loopIters < 1:
        reject(state, "LoopIters need to greater than 0")
        return

      # Make sure the prefetch VGPR index plr[x] can be aligned for each loop
      # for example, if PLR3 result in 4 VGPR:
      #   PGR  - pre  : plr[0], plr[1], plr[2]
      #   loop - iter0: plr[3], iter1: plr[0], iter2: plr[1], iter3: plr[2] -> restart LOOP (from plr[3]...) -> OK
      #
      # but if PLR2 result in 3 VGPR:
      #   PGR  - pre  : plr[0], plr[1]
      #   loop - iter0: plr[2], iter1: plr[0], iter2: plr[1], iter3: plr[2] -> restart LOOP (from plr[2]...) -> !!
      if (depthULds % ((state["PrefetchLocalRead"]%loopIters)+1)) != 0:
        validDepthU = False

      # how many elements to load
      if state["ProblemType"]["TLUA"]:
        totalElementsCoalescedA = state["MacroTileA"]
        totalElementsPerpA = depthUA
      else:
        totalElementsCoalescedA = depthUA
        totalElementsPerpA = state["MacroTileA"]

      if state["ProblemType"]["TLUB"]:
        totalElementsCoalescedB = state["MacroTileB"]
        totalElementsPerpB = depthU
      else:
        totalElementsCoalescedB = depthU
        totalElementsPerpB = state["MacroTileB"]

      totalElementsA = totalElementsCoalescedA * totalElementsPerpA
      totalElementsB = totalElementsCoalescedB * totalElementsPerpB

      if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
        if state["ProblemType"]["TLUMetadata"]:
          totalElementsCoalescedM = state["MacroTileMetadata"]
          totalElementsPerpM = depthUM
        else:
          totalElementsCoalescedM = depthUM
          totalElementsPerpM = state["MacroTileMetadata"]
        totalElementsM = totalElementsCoalescedM * totalElementsPerpM

      GlobalReadVectorWidth = state["GlobalReadVectorWidth"]
      if state["DirectToVgprA"]:
        if not state["SourceSwap"]:
          GlobalReadVectorWidth = 1 # adjust GlobalReadVectorWidth to 1 in DirectToVgpr case (except for DirectToVgprA + SourceSwap)
      tva = totalElementsA // GlobalReadVectorWidth
      if not Solution.setGlobalLoadVectorWidth(state, "A", tva, GlobalReadVectorWidth):
        validDepthU = False
      GlobalReadVectorWidth = state["GlobalReadVectorWidth"]
      tvb = totalElementsB // GlobalReadVectorWidth
      if not Solution.setGlobalLoadVectorWidth(state, "B", tvb, GlobalReadVectorWidth):
        validDepthU = False

      if state["EnableMatrixInstruction"] and state["GlobalLoadVectorWidthA"]:
        partialA = state["ProblemType"]["TLUA"] and (state["AssertFree0ElementMultiple"] % state["GlobalLoadVectorWidthA"] != 0)
        if partialA:
          glvwAlimit = 16 // state["ProblemType"]["DataType"].numBytes()
          if state["SourceSwap"]:
            matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
            glvwAlimit = matrixInstM * state["VectorWidth"]
          else:
            matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
            glvwAlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstN)

          # reduce GLVA if GLVA larger than MIOVW
          if state["GlobalLoadVectorWidthA"] > glvwAlimit:
            tva = totalElementsA // glvwAlimit
            if not Solution.setGlobalLoadVectorWidth(state, "A", tva, glvwAlimit):
              validDepthU = False

      if state["EnableMatrixInstruction"] and state["GlobalLoadVectorWidthB"]:
        partialB = state["ProblemType"]["TLUB"] and (state["AssertFree1ElementMultiple"] % state["GlobalLoadVectorWidthB"] != 0)
        if partialB:
          glvwBlimit = 16 // state["ProblemType"]["DataType"].numBytes()
          if state["SourceSwap"]:
            matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
            glvwBlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstM)
          # else:  # use origin shiftptr for B
          #   matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
          #   glvwBlimit = matrixInstN # not support state["VectorWidth"] for B yet

          # reduce GLVB if GLVB larger than MIOVW
          if state["GlobalLoadVectorWidthB"] > glvwBlimit:
            tvb = totalElementsB // glvwBlimit
            if not Solution.setGlobalLoadVectorWidth(state, "B", tvb, glvwBlimit):
              validDepthU = False

      if validDepthU and state["KernelLanguage"] == "Assembly":
        if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"]:
          if state["ProblemType"]["DataType"].numRegisters() == 0.5 and (not state["ProblemType"]["HighPrecisionAccumulate"]):
              if state["GlobalLoadVectorWidthA"] == 1 or state["GlobalLoadVectorWidthB"] == 1:
                reject(state, "HalfEcc requires HPA if glvw = 1")
        # FIXME: a transpose, b non-transpose local write, c load not supported
        if state["ProblemType"]["DataType"].numRegisters() == 0.25:
          if state["GlobalLoadVectorWidthA"] < 4:
            reject(state, "Int8 requires GLVWA >= 4, current is %u"%state["GlobalLoadVectorWidthA"])
          if state["GlobalLoadVectorWidthB"] < 4:
            reject(state, "Int8 requires GLVWB >= 4, current is %u"%state["GlobalLoadVectorWidthB"])

      if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
        if state["GlobalReadVectorWidth"] % 4 != 0:
          reject(state, "Sparse requires GRVW % 4 == 0, current GRVW is %u"%state["GlobalReadVectorWidth"])
          return
        GlobalReadVectorWidth = state["GlobalReadVectorWidth"] // 4
        tvm = totalElementsM // GlobalReadVectorWidth

        if not Solution.setGlobalLoadVectorWidth(state, "Metadata", tvm, GlobalReadVectorWidth):
          validDepthU = False

        if state["EnableMatrixInstruction"] and state["GlobalLoadVectorWidthMetadata"]:
          partialM = state["ProblemType"]["TLUMetadata"] and (state["AssertFree0ElementMultiple"] % state["GlobalLoadVectorWidthA"] != 0)
          if partialM:
            glvwMlimit = 16
            if state["SourceSwap"]:
              matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
              glvwMlimit = matrixInstM * state["VectorWidth"]
            else:
              matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
              glvwMlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstN)

            # reduce GLVMetadata if GLVMetadata larger than MIOVW
            if state["GlobalLoadVectorWidthMetadata"] > glvwMlimit:
              tvm = totalElementsM // glvwMlimit
              if not Solution.setGlobalLoadVectorWidth(state, "Metadata", tvm, glvwMlimit):
                validDepthU = False

      # Now convert elements to vectors based on GlobalReadVectorWidth
      GlobalLoadVectorWidthA = state["GlobalLoadVectorWidthA"]
      GlobalLoadVectorWidthB = state["GlobalLoadVectorWidthB"]
      if GlobalLoadVectorWidthA == 0:
        GlobalLoadVectorWidthA = GlobalReadVectorWidth
      if GlobalLoadVectorWidthB == 0:
        GlobalLoadVectorWidthB = GlobalReadVectorWidth
      totalVectorsCoalescedA = totalElementsCoalescedA // GlobalLoadVectorWidthA
      totalVectorsCoalescedB = totalElementsCoalescedB // GlobalLoadVectorWidthB
      totalVectorsA = totalElementsA // GlobalLoadVectorWidthA
      totalVectorsB = totalElementsB // GlobalLoadVectorWidthB

      if 0:
        print("info:", pvar(state, "NumThreads"), pvar(state, "DepthU"), pvar(state, "DepthULdsDivisor"),
                      "TT=%ux%u" % (state["ThreadTile0"], state["ThreadTile1"]),
                      "WG=%ux%u" % (state["WorkGroup"][0], state["WorkGroup"][1]),
                      "MT=%ux%u" % (state["MacroTile0"], state["MacroTile1"]))
        print("info: totalElementsCoalescedA=", totalElementsCoalescedA,
              " totalVectorsCoalescedA=", totalVectorsCoalescedA, " totalVectorsA=", totalVectorsA)
        print("info: totalElementsCoalescedB=", totalElementsCoalescedB,
              " totalVectorsCoalescedB=", totalVectorsCoalescedB, " totalVectorsB=", totalVectorsB)
        print("info", pvar(state, "VectorWidth")
                , pvar(state, "GlobalLoadVectorWidthA"), pvar(state, "GlobalLoadVectorWidthB"))

      #if state["ProblemType"]["DataType"].isHalf() \
      #    and (state["GlobalLoadVectorWidthA"] == 1 \
      #    or state["GlobalLoadVectorWidthB"] == 1):
      #  validDepthU = False

      if userDepthU == -1: # no vectors
        if state["GlobalLoadVectorWidthA"] != 1 \
            or state["GlobalLoadVectorWidthB"] != 1:
          validDepthU = False
      elif userDepthU == -2:
        if max( state["GlobalLoadVectorWidthA"], \
            state["GlobalLoadVectorWidthB"]) \
            < state["GlobalReadVectorWidth"]:
          validDepthU = False
      elif userDepthU <= -3:
        if min( state["GlobalLoadVectorWidthA"], \
            state["GlobalLoadVectorWidthB"]) \
            < state["GlobalReadVectorWidth"]:
          validDepthU = False

      if validDepthU:
        if not state["ProblemType"]["TLUA"]:
          if depthUA < state["GlobalLoadVectorWidthA"]:
            validDepthU = False

        if not state["ProblemType"]["TLUB"]:
          if depthU < state["GlobalLoadVectorWidthB"]:
            validDepthU = False

        if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
          if not state["ProblemType"]["TLUMetadata"]:
            if depthUM < state["GlobalLoadVectorWidthMetadata"]:
              validDepthU = False
      # this depthU is valid, done unless user wants to double (for TN)
      if validDepthU:
        if userDepthU < -3: # for every int below -3, use next doubled value
          userDepthU += 1
          depthU *= 2
          depthULds = 2
          continue
        else: # use this found value
          state["DepthU"] = depthU
          state["_DepthULds"] = depthU//state["DepthULdsDivisor"]
          depthUA = depthU if not state["ProblemType"]["SparseA"] else depthU//2
          depthUB = depthU
          depthUM = depthUA // 4 if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"] else depthUA
          state["_DepthULdsA"] = depthUA//state["DepthULdsDivisor"]
          state["_DepthULdsB"] = depthUB//state["DepthULdsDivisor"]
          state["_DepthULdsMetadata"] = depthUM//state["DepthULdsDivisor"]
          break

      # this depthU not valid
      else:
        # keep looking
        if depthU < maxDepthU:
          depthU += 2
          depthULds = depthU//state["DepthULdsDivisor"]
          continue
        # give up
        else:
          reject(state, "No valid DepthU found")
          return
    ########################################
    # end DepthU loop
    ########################################

    assert(state["DepthU"]> 0)

    if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
      state["NumLoadsCoalescedMetadata"] = 1

    if not Solution.setGlobalLoadTileDimClassic(state, "A", state["NumLoadsA"], \
        totalVectorsCoalescedA, totalElementsPerpA, depthUA):
      return
    if not Solution.setGlobalLoadTileDimClassic(state, "B", state["NumLoadsB"], \
        totalVectorsCoalescedB, totalElementsPerpB, depthU):
      return

    # Try to enlarge GLVW for metadata
    if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
      if state["ProblemType"]["TLUMetadata"]:
        totalElementsCoalescedM = state["MacroTileMetadata"]
        totalElementsPerpM = depthUM
      else:
        totalElementsCoalescedM = depthUM
        totalElementsPerpM = state["MacroTileMetadata"]
      totalElementsM = totalElementsCoalescedM * totalElementsPerpM

      bGlobalLoadVectorWidthMetadata = state["GlobalLoadVectorWidthMetadata"]
      GlobalReadVectorWidth = state["GlobalLoadVectorWidthMetadata"] * state["NumLoadsPerpendicularA"] #sum all need read
      tvm = totalElementsM // GlobalReadVectorWidth
      if not Solution.setGlobalLoadVectorWidth(state, "Metadata", tvm, GlobalReadVectorWidth):
        #fallback
        tvm = totalElementsM // bGlobalLoadVectorWidthMetadata
        Solution.setGlobalLoadVectorWidth(state, "Metadata", tvm, bGlobalLoadVectorWidthMetadata)

      GlobalLoadVectorWidthMetadata = state["GlobalLoadVectorWidthMetadata"]
      if GlobalLoadVectorWidthMetadata == 0:
        GlobalLoadVectorWidthMetadata = 1
      totalVectorsCoalescedM = totalElementsCoalescedM // GlobalLoadVectorWidthMetadata
      totalVectorsM = totalElementsM // GlobalLoadVectorWidthMetadata

      if not Solution.setGlobalLoadTileDimClassic(state, "Metadata", state["NumLoadsMetadata"], \
          totalVectorsCoalescedM, totalElementsPerpM, depthUM):
        return

    # allow LocalReadVectorWidth for TLU + MatrixInstruction
    # so far, limited to double + (DTVB or (DTVA + no DTL)) only
    # some more limitations necessary to make this logic work
    # - SourceSwap
    # - VectorWidth >= LocalReadVectorWidth
    # - AssertFree1ElementMultiple % VectorWidth == 0 (no shift edge for B)
    # - the other side of MIWaveTile must be multiple of VectorWidth
    state["allowLRVWforTLUandMI"] = state["ProblemType"]["DataType"].isDouble() and \
                                (state["DirectToVgprB"] or state["DirectToVgprA"] and not state["DirectToLds"]) and \
                                state["EnableMatrixInstruction"] and state["ProblemType"]["TLUA"] and state["ProblemType"]["TLUB"] and \
                                state["VectorWidth"] >= state["LocalReadVectorWidth"] and \
                                state["AssertFree1ElementMultiple"] % state["VectorWidth"] == 0 and \
                                ((state["DirectToVgprA"] and (state["MIWaveTile"][1] % state["VectorWidth"] == 0)) or \
                                 (state["DirectToVgprB"] and (state["MIWaveTile"][0] % state["VectorWidth"] == 0)))

    # Determine if we can load directly-to-Vgpr
    if state["DirectToVgprA"]:
      if not Solution.isDirectToVgprDoable(state, 'A', depthUA):
        return  # rejected
    if state["DirectToVgprB"]:
      if not  Solution.isDirectToVgprDoable(state, 'B', depthU):
        return  # rejected

    # TODO
    if (0 and state["LSCA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lsca % grvw != 0")
      return
    if (0 and state["LSPA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lspa % grvw != 0")
      return
    if (0 and state["LSCB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lscb % grvw != 0")
      return
    if (0 and state["LSPB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lspb % grvw != 0")
      return

    state["LVCA"] = roundupRatio(state["LSCA"] , state["GlobalLoadVectorWidthA"])
    state["LVPA"] = roundupRatio(state["LSPA"] , state["GlobalLoadVectorWidthA"])
    state["LVCB"] = roundupRatio(state["LSCB"] , state["GlobalLoadVectorWidthB"])
    state["LVPB"] = roundupRatio(state["LSPB"] , state["GlobalLoadVectorWidthB"])

    if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
      state["LVCMetadata"] = roundupRatio(state["LSCMetadata"] , state["GlobalLoadVectorWidthMetadata"])
      state["LVPMetadata"] = roundupRatio(state["LSPMetadata"] , state["GlobalLoadVectorWidthMetadata"])

    for tc in ('A','B'):
      if problemType["TLU%s"%tc]:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["Index01%s"%tc])
      else:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["IndexUnroll"])

    # Some of these might become 0?
    if 0:
      print("info: ", pvar(state, "LVCA"), pvar(state, "LVPA"), \
            pvar(state, "LVCB"), pvar(state, "LVPB"))

    # lds buffer size for A, B
    if state["KernelLanguage"] == "Source" and \
       state["LdsPadA"] != state["LdsPadB"]:
      reject(state, "Source KernelLanguage only supports LdsPadA == LdsPadB")
      return

    # NoTailLoop condition check
    # So far, NoTailLoop option is not exposed.
    validNoTailLoop = True
    invalidComment = ""
    if not bufferLoad:
      validNoTailLoop = False
      invalidComment = "does not support BufferLoad=0"
    if not (state["ProblemType"]["TLUA"] or state["ProblemType"]["TLUB"]):
      validNoTailLoop = False
      invalidComment = "does not support TLUA=False and TLUB=False"
    # NoTailLoop parameter initialization. Set True for the following cases
    #  1. ASEM is multiple of DepthU. TailLoop code will not be used in this case.
    #  2. DirectToVgpr is enabled
    # Except for case 1, validNoTailLoop should be True to enable NoTailLoop.
    # Otherwise, it will be rejected.
    state["NoTailLoop"] = False
    if state["AssertSummationElementMultiple"] % state["DepthU"] == 0:
      state["NoTailLoop"] = True
    elif state["DirectToVgprA"] or state["DirectToVgprB"]:
      if not validNoTailLoop:
        reject(state, "DirectToVgpr + AssertSummationElementMultiple%%DepthU!=0 %s to enable NoTailLoop"%invalidComment)
        return
      else:
        state["NoTailLoop"] = True

    ########################################
    # LDS
    ########################################

    state["TransposeLDSMetadata"] = True
    state["UnrollMajorLDSMetadata"] = True
    if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
      state["UnrollMajorLDSMetadata"] = state["TransposeLDSMetadata"] and (not state["ProblemType"]["TLUMetadata"])

    if state["LdsBlockSizePerPadA"] == -1:
      if state["UnrollMajorLDSA"]:
        state["LdsBlockSizePerPadA"] = roundUpToNearestMultiple(state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes(), 128)
        if state["SourceSwap"] and state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes() * state["VectorWidth"] > 128:
          state["LdsBlockSizePerPadA"] = roundUpToNearestMultiple(state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes() * state["VectorWidth"], 128)
      else:
        if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
          state["LdsBlockSizePerPadA"] = (state["MatrixInstK"] // 4) * state["MacroTile0"] * state["ProblemType"]["DataType"].numBytes()
        else:
          state["LdsBlockSizePerPadA"] = 0

    if state["LdsBlockSizePerPadB"] == -1:
      if state["UnrollMajorLDSB"]:
        state["LdsBlockSizePerPadB"] = roundUpToNearestMultiple(state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes(), 128)
      else:
        if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
          state["LdsBlockSizePerPadB"] = (state["MatrixInstK"] // 4) * state["MacroTile1"] * state["ProblemType"]["DataType"].numBytes()
        else:
          state["LdsBlockSizePerPadB"] = 0

    if state["EnableMatrixInstruction"]:
      if state["LdsBlockSizePerPadA"]:
        if state["UnrollMajorLDSA"]:
          if state["LdsBlockSizePerPadA"] % (state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes()) != 0:
            reject(state, "reject: LdsBlockSizePerPadA %u mod DepthULds %u x bpe != 0" % (state["LdsBlockSizePerPadA"],state["_DepthULds"]))
          if (state["LdsBlockSizePerPadA"] // (state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes())) % state["LSPA"] != 0 and \
              state["LSPA"] % (state["LdsBlockSizePerPadA"] // (state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes())) != 0:
            reject(state, "can't pad by addrVgpr or instOffset")

      if state["LdsBlockSizePerPadB"]:
        if state["UnrollMajorLDSB"]:
          if state["LdsBlockSizePerPadB"] % state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes() != 0:
            reject(state, "reject: LdsBlockSizePerPadB %u mod DepthULds %u x bpe != 0" % (state["LdsBlockSizePerPadB"],state["_DepthULds"]))
          if (state["LdsBlockSizePerPadB"] // (state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes())) % state["LSPB"] != 0 and \
              state["LSPB"] % (state["LdsBlockSizePerPadB"] // (state["_DepthULds"] * state["ProblemType"]["DataType"].numBytes())) != 0:
            reject(state, "can't pad by addrVgpr or instOffset")
    else:
      if state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]:
        reject(state, "didn't support UnrollMajorLDS in VALU mode yet")
      if state["LdsBlockSizePerPadA"] != 0 or state["LdsBlockSizePerPadB"] != 0:
        reject(state, "didn't support LdsBlockSizePerPad in VALU mode yet")

    # Determine if we can load directly-to-LDS.
    # Transpose requires a trip through registers to perform the transpose so can't use DirectToLdsA
    # LDS loads always write 4 bytes apart so can use only 4-byte operations
    #   TODO - for doubles we need to add something special here?
    # The matrix must not require transposing since that is done by reading to VGPR and writing in different order
    # The LSC (load size coalesced) must load some multiple of 256 bytes since that is what each DirectToLds load provides
    # Note for these matrices LSC is same as MacroTile dim
    # MatrixInstruction rules:
    # DirectToLDS is supported for TLU=0  (make sure transposeLDS=1)
    # LDS (load size coalesced) * LSPA must load some multiple of 256 bytes.
    # added support for loadX2/loadx4 .
    # x2/x4 (use x4 for better load efficiency)
    # x2/x4 support for NT layout for f64/f32   TN layout (f16/bf16/f32/f64) with transposeLDS=1
    # ignore x2/x4 precision < 4 bytes in NT layout
    if state["DirectToLds"]:
      # DirectToLdsA + DirectToVgprA does not work. Not enable DirectToLdsA if DirectToVgprA is true
      if (not state["DirectToVgprA"]) and Solution.isDirectToLdsDoable(state, 'A'):
        state["DirectToLdsA"] = True
        state["LocalWriteUseSgprA"] = True
        #print("DirectToLdsA", state["DirectToLdsA"])

      # DirectToLdsB + DirectToVgprB does not work. Not enable DirectToLdsB if DirectToVgprB is true
      if (not state["DirectToVgprB"]) and Solution.isDirectToLdsDoable(state, 'B'):
        state["DirectToLdsB"] = True
        state["LocalWriteUseSgprB"] = True
        #print("DirectToLdsB", state["DirectToLdsB"])

      # Update parent variable so kernel display is accurate
      state["DirectToLds"] = state["DirectToLdsA"] or state["DirectToLdsB"]
      if state["1LDSBuffer"] == -1 and state["DirectToLds"]:
        #1LDS buffer must be 0 for DirectToLdsA
        state["1LDSBuffer"] = 0

    # set NoLdsWriteCode if DirectToLds + DirectToVgpr or DirectToLdsA+B is enabled
    state["NoLdsWriteCode"] = False
    if (state["DirectToVgprA"] and state["DirectToLdsB"]) or (state["DirectToVgprB"] and state["DirectToLdsA"]) or \
        (state["DirectToLdsA"] and state["DirectToLdsB"]):
      state["NoLdsWriteCode"] = True

    # Default LocalReadVectorWidth
    if state["LocalReadVectorWidth"] == -1:
      if state["EnableMatrixInstruction"] and not state["allowLRVWforTLUandMI"]:
        state["LocalReadVectorWidth"] = state["MIInputPerThread"]
        # enable less than state["MIInputPerThread"]
        # for fp64 this means ds_read_b32
        if ((state["DirectToLdsA"] and state["ProblemType"]["TLUA"]) or \
            (state["DirectToLdsB"] and state["ProblemType"]["TLUB"])):
             state["LocalReadVectorWidth"] = 1 if (state["ProblemType"]["DataType"].numBytes() >= 4) else state["LocalReadVectorWidth"]
      else:
        state["LocalReadVectorWidth"] = state["VectorWidth"]
    else:
      if state["EnableMatrixInstruction"]:
        # support LocalReadVectorWidth < miInputPerThread for directToLdsX2/X4
        if state["LocalReadVectorWidth"] < state["MIInputPerThread"] and not (state["DirectToLdsA"] or state["DirectToLdsB"]):
          reject(state, "LocalReadVectorWidth < %u" %(state["MIInputPerThread"]))
        if state["LocalReadVectorWidth"] > state["MIInputPerThread"] and not state["TransposeLDS"] \
           and not state["allowLRVWforTLUandMI"]:
          reject(state, "LocalReadVectorWidth require Transpose LDS")
      else:
        if state["LocalReadVectorWidth"] != state["VectorWidth"]:
          reject(state, "LocalReadVectorWidth must equal VectorWidth for non MI kernels")

    # set pad as readRegs to avoid unaligned read
    optPadA = optPadB = state["LocalReadVectorWidth"]
    readRegsA = readRegsB = state["LocalReadVectorWidth"]*state["ProblemType"]["DataType"].numBytes()//4
    if state["ProblemType"]["SparseA"]:
      optPadA //= 2
      readRegsA //= 2
    if readRegsA > 4 or readRegsB > 4:
      reject(state, "LocalReadVectorWidth results in attemping to read LDS larger than b128, reject")
    if state["EnableMatrixInstruction"]:
      # for readRegs = 1 or 4, we need to double pad for MI16x16xNx1 to avoid bank conflict.
      if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
        if readRegsA == 4 or readRegsA == 1:
          optPadA *= 2
        if readRegsB == 4 or readRegsB == 1:
          optPadB *= 2
    if state["LdsPadA"] == -1:
      if state["ProblemType"]["TLUA"]:
        if state["EnableMatrixInstruction"]:
          state["LdsPadA"] = 0
          if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
            state["LdsPadA"] = ((16 * state["ProblemType"]["DataType"].numBytes() + (state["MatrixInstK"] // 4) * state["MacroTile0"] * state["ProblemType"]["DataType"].numBytes()) % 128) // state["ProblemType"]["DataType"].numBytes()
          if state["SourceSwap"] and state["VectorWidth"] > 1:
            pass
        else:
          state["LdsPadA"] = 0
      else:
        if state["EnableMatrixInstruction"] and state["TransposeLDS"]:
          state["LdsPadA"] = max(state["GlobalReadVectorWidth"],optPadA)
        else:
          state["LdsPadA"] = state["VectorWidth"]
        ## turn-off padding for directToLds
        if state["EnableMatrixInstruction"] and state["TransposeLDS"] and state["DirectToLdsA"]:
          state["LdsPadA"] = 0
      assert(state["LdsPadA"] >= 0)

    if state["LdsPadB"] == -1:
      if state["ProblemType"]["TLUB"]:
        if state["EnableMatrixInstruction"]:
          state["LdsPadB"] = 0
          if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
            state["LdsPadB"] = ((16 * state["ProblemType"]["DataType"].numBytes() + (state["MatrixInstK"] // 4) * state["MacroTile1"] * state["ProblemType"]["DataType"].numBytes()) % 128) // state["ProblemType"]["DataType"].numBytes()
        else:
          state["LdsPadB"] = 0
      else:
        if state["EnableMatrixInstruction"] and state["TransposeLDS"]:
          state["LdsPadB"] = max(state["GlobalReadVectorWidth"],optPadB)
        else:
          state["LdsPadB"] = state["VectorWidth"]
        if state["EnableMatrixInstruction"] and state["TransposeLDS"] and state["DirectToLdsB"]:
          state["LdsPadB"] = 0
      assert(state["LdsPadB"] >= 0)

    if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
      optPadM = optPadA // 4
      if state["LdsPadMetadata"] == -1:
        if state["ProblemType"]["TLUMetadata"]:
          state["LdsPadMetadata"] = 0
        else:
          if state["EnableMatrixInstruction"] and state["TransposeLDSMetadata"]:
            state["LdsPadMetadata"] = max(state["GlobalReadVectorWidth"] // 4, optPadM)
          else:
            state["LdsPadMetadata"] = state["VectorWidth"] // 4
          ## turn-off padding for directToLds
          if state["EnableMatrixInstruction"] and state["TransposeLDSMetadata"] and state["DirectToLdsMetadata"]:
            state["LdsPadMetadata"] = 0
        assert(state["LdsPadMetadata"] >= 0)

    if (state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]) and (not state["EnableMatrixInstruction"]):
        reject(state, "UnrollMajorLDS Supports only in EnableMatrixInstruction=1")

    ldsAlign = int(64 / state["ProblemType"]["DataType"].numRegisters())

    if state["UnrollMajorLDSA"]:
      ldsNumElementsA = (state["_DepthULdsA"] + state["LdsPadA"]) * state["MacroTileA"]
    else:
      ldsNumElementsA = state["_DepthULdsA"] * (state["MacroTileA"] + state["LdsPadA"])
    padInterval = state["LdsBlockSizePerPadA"] // bpeAB
    if padInterval != 0:
      ldsNumElementsA = int((state["_DepthULdsA"] * state["MacroTileA"]) / padInterval * (padInterval + state["LdsPadA"]))
    ldsNumElementsAlignedA = roundUpToNearestMultiple(ldsNumElementsA, ldsAlign)
    if state["DirectToVgprA"]:
      # DirectToVgpr does not use LDS. Set to 0.
      ldsNumElementsA = 0
      ldsNumElementsAlignedA = 0

    if state["UnrollMajorLDSB"]:
      ldsNumElementsB = (state["_DepthULdsB"] + state["LdsPadB"]) * state["MacroTileB"]
    else:
      ldsNumElementsB = state["_DepthULdsB"] * (state["MacroTileB"] + state["LdsPadB"])
    padInterval = state["LdsBlockSizePerPadB"] // bpeAB
    if padInterval != 0:
      ldsNumElementsB = int((state["_DepthULdsB"] * state["MacroTileB"]) / padInterval * (padInterval + state["LdsPadB"]))
    ldsNumElementsAlignedB = roundUpToNearestMultiple(ldsNumElementsB, ldsAlign)
    if state["DirectToVgprB"]:
      # DirectToVgpr does not use LDS. Set to 0.
      ldsNumElementsB = 0
      ldsNumElementsAlignedB = 0

    if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
      if state["UnrollMajorLDSMetadata"]:
        ldsNumElementsMetadata = (state["_DepthULdsMetadata"] + state["LdsPadMetadata"]) * state["MacroTileMetadata"]
      else:
        ldsNumElementsMetadata = state["_DepthULdsMetadata"] * (state["MacroTileMetadata"] + state["LdsPadMetadata"])
      ldsNumElementsMetadata = roundUp(ldsNumElementsMetadata / bpeAB) # metadata is in byte type. so divide ldsNumElementsMetadata by A,B's bpe
      padInterval = state["LdsBlockSizePerPadMetadata"]
      if padInterval != 0:
        ldsNumElementsMetadata = int(roundUp(state["_DepthULdsMetadata"] * state["MacroTileMetadata"]/bpeAB) / padInterval * (padInterval + state["LdsPadMetadata"]))
      ldsNumElementsAlignedMetadata = roundUpToNearestMultiple(ldsNumElementsMetadata, ldsAlign)
    else:
      ldsNumElementsMetadata = 0
      ldsNumElementsAlignedMetadata = 0

    # todo, can the alignment be a power of 2?
    state["LdsOffsetA"] = 0
    if state["PrefetchGlobalRead"]:
      state["LdsNumElementsAlignedA"] = ldsNumElementsAlignedA
      state["LdsNumElementsAlignedB"] = ldsNumElementsAlignedB
      state["LdsNumElementsAlignedMetadata"] = ldsNumElementsAlignedMetadata
      state["LdsOffsetMetadata"] = state["LdsOffsetA"] + state["LdsNumElementsAlignedA"]
      state["LdsOffsetB"] = state["LdsOffsetMetadata"] + state["LdsNumElementsAlignedMetadata"]

      offsetBlk = state["LdsOffsetB"] +  ldsNumElementsAlignedB
      offsetBlk = int(2**(math.ceil(math.log(offsetBlk, 2))))

      state["LdsOffsetA_Blk"] = offsetBlk
      state["LdsOffsetMetadata_Blk"] = state["LdsOffsetA_Blk"] + state["LdsNumElementsAlignedA"]
      state["LdsOffsetB_Blk"] = state["LdsOffsetMetadata_Blk"] + state["LdsNumElementsAlignedMetadata"]
      ldsNumElementsAB = state["LdsOffsetB_Blk"]+ ldsNumElementsB
    else:
      state["LdsOffsetMetadata"] = ldsNumElementsAlignedA
      state["LdsOffsetB"] = state["LdsOffsetMetadata"] + ldsNumElementsAlignedMetadata
      ldsNumElementsAB = state["LdsOffsetB"] + ldsNumElementsB

    # lds buffer size for reduction
    ldsNumElementsReduction = state["LocalSplitU"]*state["MacroTile0"]*state["MacroTile1"] if state["LocalSplitU"] > 1 else 0

    # lds max occupancy
    ldsSizeOccupancy = globalParameters["DeviceLDS"] // state["MaxOccupancy"]
    ldsNumElementsOccupancy = ldsSizeOccupancy // state["ProblemType"]["DestDataType"].numBytes()

    #print("ldsNumElementsA", ldsNumElementsA)
    #print("ldsNumElementsB", ldsNumElementsB)
    #print("ldsNumElementsMetadata", ldsNumElementsMetadata)
    #print("ldsNumElementsAlignedA", ldsNumElementsAlignedA)
    #print("ldsNumElementsAlignedB", ldsNumElementsAlignedB)
    #print("ldsNumElementsAlignedMetadata", ldsNumElementsAlignedMetadata)
    #print("ldsNumElementsAB", ldsNumElementsAB)
#
    #print("LdsOffsetB", state["LdsOffsetB"])
    #print("LdsOffsetMetadata", state["LdsOffsetMetadata"])
    #if state["PrefetchGlobalRead"]:
    #  print("LdsOffsetA_BLK", state["LdsOffsetA_Blk"])
    #  print("LdsOffsetB_BLK", state["LdsOffsetB_Blk"])
    #  print("LdsOffsetMetadata_BLK", state["LdsOffsetMetadata_Blk"])

    if state["EnableMatrixInstruction"]:
      if state["DirectToLds"] and state["1LDSBuffer"]:
        reject(state, "1LDSBuffer must be 0 for directToLds")

    if state["1LDSBuffer"] == -1:
      if state["MIWaveTile"][0] == 1 and state["MIWaveTile"][1] == 1 or \
          ldsNumElementsAB * state["ProblemType"]["DataType"].numBytes() <= max(ldsSizeOccupancy,32768) or \
          (state["ProblemType"]["ComputeDataType"].numBytes() * state["MacroTile0"] * state["MacroTile1"] > 32768*4 and \
            not (ldsNumElementsAB * state["ProblemType"]["DataType"].numBytes() > globalParameters["DeviceLDS"])):
        state["1LDSBuffer"] = 0
      else:
        state["1LDSBuffer"] = 1

    if state["1LDSBuffer"]:
      if not state["PrefetchGlobalRead"]:
        reject(state, "PGR=0 already use 1 LDS buffer only")
      # Should be able to support as long as NO scheduleLocalWrite
      if (not state["ScheduleIterAlg"] == 2) and (not state["ScheduleIterAlg"] == 3) and (state["ScheduleLocalWrite"]):
        reject(state, "1LDSBuffer only support SIA2 or SIA3, or SIA1 without SLW")
      state["LdsOffsetB"] = ldsNumElementsAlignedA
      state["LdsOffsetMetadata"] = state["LdsOffsetB"] + ldsNumElementsAlignedB
      ldsNumElementsAB = ldsNumElementsAlignedA + ldsNumElementsAlignedB + ldsNumElementsMetadata

    # lds size is the greater of the two
    ldsNumElements = max(ldsNumElementsAB, ldsNumElementsReduction, ldsNumElementsOccupancy)

    if state["NumElementsPerBatchStore"] == -1:
      if ldsNumElements * state["ProblemType"]["DataType"].numBytes() > 32768 or \
          state["ProblemType"]["ComputeDataType"].numBytes() * state["MacroTile0"] * state["MacroTile1"] > 32768*4:
        state["NumElementsPerBatchStore"] = 0
        state["StorePriorityOpt"] = 0
        state["StoreSyncOpt"] = 0
        state["GroupLoadStore"] = 0
      else:
        state["NumElementsPerBatchStore"] = 2 if not state["ProblemType"]["DataType"].numBytes() == 8 else 1

    if state["StoreRemapVectorWidth"] == -1:
      # use de_read_b64 as default in storeRemap to avoid bank conflict
      defaultRemap = 8 // state["ProblemType"]["DestDataType"].numBytes()
      defaultRemap = max(defaultRemap, state["MacroTile0"]//state["WavefrontSize"])
      ldsRemapPad = max(defaultRemap, state["MIOutputVectorWidth"])
      ldsNumElementsRemapC = (state["MacroTile0"]+ldsRemapPad)* state["MatrixInstN"] * state["MIWaveGroup"][1]
      if state["_GlobalAccumulation"]:
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
        ldsNumElementsRemapC *= (computeBytes / state["ProblemType"]["DestDataType"].numBytes())
      ldsSize = ldsNumElementsRemapC * state["ProblemType"]["DestDataType"].numBytes()
      if not math.log(state["MacroTile0"],2).is_integer() or \
          ldsSize > globalParameters["MaxLDS"] or \
          state["SourceSwap"] or \
          (state["GlobalSplitU"] > 1) and (state["_GlobalAccumulation"] != 'MultipleBuffer') or \
          state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4:
        state["StoreRemapVectorWidth"] = 0
      else:
        state["StoreRemapVectorWidth"] = defaultRemap

    # GuaranteeNoPartial
    if state["ProblemType"]["TLUA"]:
      state["GuaranteeNoPartialA"] = state["AssertFree0ElementMultiple"]%state["GlobalLoadVectorWidthA"]==0
    else:
      state["GuaranteeNoPartialA"] = True

    if state["ProblemType"]["TLUB"]:
      state["GuaranteeNoPartialB"] = state["AssertFree1ElementMultiple"]%state["GlobalLoadVectorWidthB"]==0
    else:
      state["GuaranteeNoPartialB"] = True

    # SourceSwap
    if state["SourceSwap"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, "SourceSwap only applies to MatrixInstruction kernels")
        return
      if state["StoreRemapVectorWidth"]:
        reject(state, "SourceSwap not compatible with StoreRemap")
        return

    # SparseA problem
    if state["ProblemType"]["SparseA"]:
      if state["PrefetchGlobalRead"] and not state["ExpandPointerSwap"]:
        reject(state, "Sparse A kernel only support PGR with EPS=1.")
        return
      if state["MIArchVgpr"]:
        reject(state, "Sparse A kernel does not support MIArchVgpr yet.")
        return
      if state["DepthULdsDivisor"] > 1:
        reject(state, "Sparse A kernel does not support SplitLDS yet.")
        return
      # Not Support Feature
      if state["SourceSwap"]:
        reject(state, "Sparse A kernel cannot support SourceSwap.")
        return
      state["AssertSummationElementMultiple"] = 8

    # check if need to use lds init Acc vgprs
    state["LdsInitCVgprs"] = False
    if globalParameters["ArchCaps"][isa]["HasAccCD"] and \
         state["EnableMatrixInstruction"] and state["StorePriorityOpt"] and \
         state["ProblemType"]["DataType"].isDouble():
      state["LdsInitCVgprs"] = True

    if state["MIArchVgpr"]:
      if not globalParameters["ArchCaps"][isa]["HasAccCD"] or \
         not state["EnableMatrixInstruction"]:
        reject(state, "MIArchVgpr requires gcn support ACC_CD bit for MatrixInstruction")
        return

      if not (state["ProblemType"]["ComputeDataType"].isDouble() or \
              state["ProblemType"]["ComputeDataType"].isSingle() or \
              (state["ProblemType"]["ComputeDataType"].isHalf() and state["ProblemType"]["HighPrecisionAccumulate"]) or \
              state["ProblemType"]["ComputeDataType"].isInt32() or \
              state["ProblemType"]["ComputeDataType"].isDoubleComplex()):
        reject(state, "MIArchVgpr now only support fp64, fp32, fp16, int8 MatrixInstruction.")
        return

    if state["ProblemType"]["Fp16AltImpl"]:
      if not (state["ProblemType"]["DataType"].isHalf() and \
              state["ProblemType"]["HighPrecisionAccumulate"] and \
              state["EnableMatrixInstruction"]):
        reject(state, "Fp16AltImpl requires FP16 HPA MFMA")
        return

    #check not support cases and calculate lds resources
    if state["StoreRemapVectorWidth"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, "storeRemap only support MatrixInstruction kernel")
        return
      if (state["GlobalSplitU"] > 1) and (state["_GlobalAccumulation"] != 'MultipleBuffer'):
        reject(state, "storeRemap doesn't support GlobalSplitU yet, except GSU algorithm 2")
        return
      if packedC0 or packedC1:
        reject(state, "storeRemap doesn't support packedC0 and packedC1 yet")
        return
      if state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4:
        reject(state, "storeRemap doesn't support MI4x4 multi blocks in N direction yet")
        return
      if not math.log(state["MacroTile0"],2).is_integer():
        reject(state, "storeRemap only supports power-of-2 MT0")
        # TODO - this return should be here, but this is a hotfix,
        # Somehow we have a "Validation Failed" kernel in rocBLAS now (SRVW=4 and MT0=96) and this will stop the whole building process
        # Actions: 1. Hotfix, comment out this "return" temporarily for that invalidated kernel
        #          2. Remove / replace that invalidated kernel
        #          3. Put back this return
        #          4. How to design a better way to prevent from invalid kernel in rocBLAS?
        # return

      storeInstMinWidth = 1 # minimum dwordx1
      storeInstMaxWidth = 4 # maximum dwordx4
      srMinVw = max(storeInstMinWidth, int(storeInstMinWidth/state["ProblemType"]["DestDataType"].numRegisters()))
      numReg  = state["ProblemType"]["DestDataType"].numRegisters()
      if state["_GlobalAccumulation"]:
        numReg = state["ProblemType"]["ComputeDataType"].numRegisters()

      srMaxVw = int(storeInstMaxWidth/numReg)
      if srMinVw > state["StoreRemapVectorWidth"] or srMaxVw < state["StoreRemapVectorWidth"]:
        reject(state, "StoreRemapVectorWidth %u is not allowed for this data type" % state["StoreRemapVectorWidth"])
        return

      if state["StoreRemapVectorWidth"] * state["WavefrontSize"] < state["MacroTile0"]:
        reject(state, "storeRemap: Per wave single global write instruction doesn't enough to write one M column." + \
               " Please use larger StoreRemapVectorWidth.")
        return
      if (state["MacroTile0"]*state["MatrixInstN"])//state["MIWaveGroup"][0] < state["StoreRemapVectorWidth"]*state["WavefrontSize"]:
        reject(state, "storeRemap: number elements of lds less than per wave per local read elements." + \
               " Please use smaller StoreRemapVectorWidth.")
        return
      ldsRemapPad = max(state["StoreRemapVectorWidth"],state["MIOutputVectorWidth"])
      ldsNumElementsRemapC = (state["MacroTile0"]+ldsRemapPad)* state["MatrixInstN"] * state["MIWaveGroup"][1]

      if state["_GlobalAccumulation"]:
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
        multiplier = computeBytes // state["ProblemType"]["DataType"].numBytes()
      elif state["ProblemType"]["DestDataType"].numBytes() > state["ProblemType"]["DataType"].numBytes():
        # Determine ratio of output to input element size.
        # SRVW remaps output so we need to scale up resources.
        multiplier = state["ProblemType"]["DestDataType"].numBytes() // state["ProblemType"]["DataType"].numBytes()
      else:
        multiplier = 1

      ldsNumElementsRemapC *= multiplier

      #print("ldsNumElementsRemapC=%u" % ldsNumElementsRemapC)

      # if LDS is bound by RemapC (SRVW), then 1LDSBuffer actually doesn't help in SIA3
      # since LDS usage couldn't be reduced
      if state["1LDSBuffer"] and (state["ScheduleIterAlg"] == 3) and (ldsNumElements < ldsNumElementsRemapC):
        # TODO- Remove this DataType test condition,
        # Currently we do this test is just because we don't want to affect existing logic in rocBLAS
        if state["ProblemType"]["DataType"].isInt8():
          reject(state, "LDS usage is bound be StoreRemap, thus 1LDSBuffer wouldn't have any help. Skip.")
          return

      ldsNumElements = max(ldsNumElements, ldsNumElementsRemapC)

    if state["ProblemType"]["UseBias"]:
      # Currently all offsets starts from 0
      state["LdsOffsetBias"] = 0 # TODO: ldsBiasOffset = ldsNumElementsAB
      ldsBiasMaxElements = 0
      if state["ProblemType"]["Gradient"]:
        if state["ProblemType"]["BiasSrc"] == "A":
          tile01 = state["ProblemType"]["Index01A"]
        elif state["ProblemType"]["BiasSrc"] == "B":
          tile01 = state["ProblemType"]["Index01B"]
        elif state["ProblemType"]["BiasSrc"] == "D":
          tile01 = -1
        else:
          assert 0 and "Unsupported tile01 for bias lds calculation."
        # Don't need to calculate lds bias
        if tile01 > -1:
          maxKId = state["WavefrontSize"] // ((state["MatrixInstM"] if (tile01 == 0) else state["MatrixInstN"]) * state["MatrixInstB"])
          for dataType in state["ProblemType"]["BiasDataTypeList"]:
            ldsBiasMaxElements = max(ldsBiasMaxElements, state["MacroTile%d"%tile01] * maxKId * dataType.numBytes())
      else:
        for dataType in state["ProblemType"]["BiasDataTypeList"]:
          ldsBiasMaxElements = max(ldsBiasMaxElements, state["MacroTile0"] * dataType.numBytes())
      ldsNumElements = max(ldsNumElements, state["LdsOffsetBias"] + ldsBiasMaxElements)

    state["LdsNumElements"] = ldsNumElements
    ldsSize = ldsNumElements * state["ProblemType"]["DataType"].numBytes()
    if ldsSize > globalParameters["MaxLDS"]:
      reject(state, "Kernel Uses %u > %u bytes of LDS" % ( ldsSize, globalParameters["MaxLDS"]))
      return

    # LoopUnroll  = DepthU / LocalSplitU
    if "LocalSplitU" in state and "_DepthULds" in state:
      state["LoopUnroll"] = state["_DepthULds"] // state["LocalSplitU"]
    if state["LoopUnroll"] * state["LocalSplitU"] != state["_DepthULds"]:
      state["Valid"] = False
    if state["KernelLanguage"] != "Assembly" and state["InnerUnroll"] != 1:
      reject(state, "InnerUnroll only supported on assembly")
    state["LoopUnroll"] //= state["InnerUnroll"]

    if 0:
      print("info: ", pvar(state, "LoopUnroll"), " LDS Stats:", pvar(state, "LdsOffsetA"), pvar(state, "LdsOffsetB"))
      print("info: ", pvar(state["ProblemType"], "TLUA"), \
          pvar(state, "NumLoadsCoalescedA"), pvar(state, "NumLoadsPerpendicularA"), \
          pvar(state, "LSCA"), pvar(state, "LSPA"))
      print("info:", pvar(state["ProblemType"], "TLUB"), \
          pvar(state, "NumLoadsCoalescedB"), pvar(state, "NumLoadsPerpendicularB"), \
          pvar(state, "LSCB"), pvar(state, "LSPB"))

    state["LoopIters"] = state["LoopUnroll"]
    if "MatrixInstK" in state:
      state["LoopIters"] //= state["MatrixInstK"]

    if state["LoopIters"] < 1:
      reject(state, "LoopIters need to greater than 0")
      return

    # Since we use PLR >= LoopIters for allocating numberOfIters vgprBuffer for a while
    # we need to support both PLR >= LoopIters and CLR parameter for solutions in rocBLAS
    if state["ClusterLocalRead"] and state["PrefetchLocalRead"] >= state["LoopIters"] and not state["ScheduleIterAlg"] == 2:
      reject(state, "\"PLR >= LoopIters\" expression has been deprecated, please use ClusterLocalRead with PLR < LoopIters")

    if state["ClusterLocalReadPack"]:
      if not state["EnableMatrixInstruction"] or not state["ProblemType"]["DataType"].numRegisters() < 1:
        reject(state, "CLRP only support Matrixinstruction")
      if not state["ClusterLocalRead"]:
        reject(state, "no meaning if set CLRP without CLR")

    # reject iterations are not enough to use wider local read
    if state["EnableMatrixInstruction"] and state["PrefetchLocalRead"] > 0:
      # Multiple = WLR-size / input-size = how many iters could be covered by one WLR ?
      wlrMultiple = state["LocalReadVectorWidth"]//state["MIInputPerThread"]
      if wlrMultiple == 0:
        reject(state, "LocalReadVectorWidth %u is less than MIInput" % (state["LocalReadVectorWidth"]))
        return
      # for example, if the original ds_read is b32...
      #   1. if LoopIters = 5 (b32 x 5 times), WLR-Multiple = 2 (b64), then we can fit the WLR
      #   2. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 4 (b128), this is not allowed
      #   3. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 2 (b64), this is allowed
      if state["LoopIters"] % wlrMultiple != 0:
        reject(state, "LocalReadVectorWidth %u cannot be distributed evenly, LoopIters %u should be divisible by WLR-Multiple %u" \
          % (state["LocalReadVectorWidth"], state["LoopIters"], wlrMultiple))

      if state["LoopIters"] - (state["PrefetchLocalRead"] * wlrMultiple) < 0 :
        reject(state, "with PrefetchLocalRead %u LoopIters %u LocalReadVectorWidth %u, not enough LoopIters to prefetch %ux%u iterations, " \
          % (state["PrefetchLocalRead"],state["LoopIters"],state["LocalReadVectorWidth"], state["PrefetchLocalRead"] , wlrMultiple) )

    # # reject conditions with lower performance
    # if state["ScheduleIterAlg"] == 2 and \
    # (state["ExpandPointerSwap"] != 1 or state["LoopIters"] != 1 or state["ScheduleGlobalRead"] != 1):
    #   reject(state, "ScheduleIterAlg 2 only work with EPS1_SGR1, LoopIter=1")

    if state["TransposeLDS"] == 1:
      if not state["EnableMatrixInstruction"]:
        reject(state, "TransposeLds Supports only in MatrixInstruction=1")
      if state["ProblemType"]["TLUA"] and state["ProblemType"]["TLUB"]:
          # TODO: Now in rocBLAS, lot of logic yamls are Type=NT and TLDS=1? Why aren't they rejected and how to get rid of them?
          reject(state, "TransposeLds requires TLUA=0 or TLUB=0")
    if state["EnableMatrixInstruction"]:
      # enable widerLocalRead
      if state["LocalReadVectorWidth"] > state["MIInputPerThread"]:
        # wider localRead support 2 types
        # 1. prefetch all lds to register
        # 2. using larger InnerUnroll
        if not (state["PrefetchLocalRead"] >= state["LoopIters"] and state["InnerUnroll"] == 1) and \
            not state["ClusterLocalRead"] and \
            not state["InnerUnroll"] >= state["LocalReadVectorWidth"] // state["MIInputPerThread"]:
          reject(state, "wider localRead only support ClusterLocalRead or (InnerUnroll > WiderLocalReadxN)")

    if state["DepthULdsDivisor"] > 1:
      if state["PrefetchGlobalRead"] == 2:
        reject(state, "DepthULdsDivisor > 1 does not support PrefetchGlobalRead=2")
      if state["ScheduleIterAlg"] != 3:
        reject(state, "DepthULdsDivisor > 1 does not support SchedulIterAlg other than 3")
      if state["DirectToLds"] == True:
        reject(state, "DepthULdsDivisor > 1 does not support DirectToLds")
      if state["ProblemType"]["TLUA"] or state["ProblemType"]["TLUA"] or not state["TransposeLDS"]:
        reject(state, "DepthULdsDivisor > 1: Only works with TN problem layout and TransposeLDS")
      if state["PrefetchGlobalRead"]==1 and state["PrefetchLocalRead"]==0:
        reject(state, "PGR1 + PLR0 in SplitLDS requires double G2L buffer which is yet to be implemented")
      if state["ProblemType"]["DataType"].numRegisters()*state["GlobalReadVectorWidth"] < state["DepthULdsDivisor"]:
        reject(state, "SplitLDS requires wider GlobalReadVectorWidth; needs RegisterPerElem (%f) * GRVW (%u) >= DepthULdsDivisor (%u)"%
          (state["ProblemType"]["DataType"].numRegisters(),state["GlobalReadVectorWidth"],state["DepthULdsDivisor"]))

    if state["GlobalReadPerMfma"] > 1 and state["PrefetchGlobalRead"] == 2:
      reject(state, "GlobalReadPerMfma need to be 1 if PGR2")

    if state["UseInstOffsetForGRO"] == -1:
      state["UseInstOffsetForGRO"] = 1 if state["DirectToLds"] else 0

    # guard against out of bounds reads
    # None: don't guard against ou
    # ShiftPtr: shift read pointers to be in bounds, then unshift registers (source & assembly),
    # ShiftPtr does not support very small problem dims < global load vector width since the shift
    # would move outside the array bounds.
    # If GLVW==1 or Assert*ElementMultiple for the coalesced dim is > GRVW, then shifting is not
    # necessary and the shift/unshift code will not be generated
    state["EdgeType"] = "ShiftPtr" # Use ShiftPtr by default

    # Precise bounds check uses the "num_records" field in the buffer to
    # precisely detect when we are inbounds or not.  Only a one-dimensional
    # check is used since this is faster and also for computation we only
    # need to ensure that none of the loads fault.  threads which are
    # computing bogus sections of the C tile will later be ignored.
    # precise checking only works when all elements of the load are in-bounds
    # since if the vload crosses boundary we ignore all components not just the
    # ones that are OOB. See comments for groOffsetInMacroTile in KernelWriterAssembly.py
    #
    # So check for the cases where the unroll loop can
    # generate partial loads here and reject PBC solutions:
    # For non-TLU the free dim is in perp dim - should always be TRUE?  TODO

    #--
    # ShiftPtr can't use UseSgprForGRO since it needs to modify the VGPR pointers
    if bufferLoad and state["_UseSgprForGRO"] and state["EdgeType"]=="ShiftPtr":
      if not state["GuaranteeNoPartialA"] or not state["GuaranteeNoPartialB"]:
        state["_UseSgprForGRO"] = False
        #reject(state, "PBC with wide load has insufficient overlap guarantees- try GRVW=1 or adding appropriate Assert*ElementMultiple")




    if state["EnableMatrixInstruction"]:
      cont1 = not state["GuaranteeNoPartialB"]
      cont2 = ((state["MatrixInstN"] % state["GlobalLoadVectorWidthB"]) != 0)
      if cont1 and cont2:
        reject(state, "MatrixInstN %u %% GlobalLoadVectorWidthB %u must be 0" % \
          (state["MatrixInstN"], state["GlobalLoadVectorWidthB"]))
    else:
      if not bufferLoad or not state["GuaranteeNoPartialA"]:
        # Restrict GRVW/VW combos so shift-ptr logic will work
        if state["GlobalLoadVectorWidthA"] > 1 \
            and state["GlobalLoadVectorWidthA"] != state["VectorWidth"]:
            reject(state, "GlobalLoadVectorWidthA %u must be == VectorWidth %u or == 1" % \
                    (state["GlobalLoadVectorWidthA"], state["VectorWidth"]))

      if not bufferLoad or not state["GuaranteeNoPartialB"]:
        # Restrict GRVW/VW combos so shift-ptr logic will work
        if state["GlobalLoadVectorWidthB"] > 1 \
            and state["GlobalLoadVectorWidthB"] != state["VectorWidth"]:
            reject(state, "GlobalLoadVectorWidthB %u must be == VectorWidth %u or == 1" % \
                    (state["GlobalLoadVectorWidthB"], state["VectorWidth"]))

    # Use SGPR to store an offset from GlobalReadOffsetA+0.
    # (as opposed to using dedicated VGPR for each GRO
    # Requires preciseBounds check since we rely on the buffer bounds check, not
    # individual vector registers doing bounds compares.

    if state["_UseSgprForGRO"] == -1:
      # Don't use SGPR if it looks like we might not have enough - better to leave PBC enabled even if we have to use VGPR
      # 40 is based on current SGPR usage, this may need to be tuned in the future:
      numLoadsA = state["NumLoadsCoalescedA"]*state["NumLoadsPerpendicularA"]
      numLoadsB = state["NumLoadsCoalescedB"]*state["NumLoadsPerpendicularB"]
      numLoadsM = 0
      if state["ProblemType"]["SparseA"] and not state["DirectToVgprSparseMetadata"]:
        numLoadsM = state["NumLoadsCoalescedMetadata"]*state["NumLoadsPerpendicularMetadata"]
      if numLoadsA + numLoadsB + numLoadsM > 35 or state["DirectToVgprA"] or state["DirectToVgprB"]: # force _UseSgprForGRO = 0 if DirectToVgpr is enabled
        #print "info: Disabling UseSgprForGRO since predicting too many SGPR will be used"
        state["_UseSgprForGRO"] = 0
      else:
        state["_UseSgprForGRO"] = 1


    if packedC0 and not state["GuaranteeNoPartialA"]:
      reject(state, "packedC0 requires GuaranteeNoPartialA")
    if packedC1 and not state["GuaranteeNoPartialB"]:
      reject(state, "packedC1 requires GuaranteeNoPartialB")

    if packedC0 or packedC1:
      state["_UseSgprForGRO"] = 0

      if state["EdgeType"] != "ShiftPtr":
        reject(state, "Packed dims requires EdgeType==ShiftPtr")
      if state["KernelLanguage"] == "Assembly":
        if not bufferLoad:
          reject(state, "Packed dims for Assembly requires BufferLoad")

    if packedC0: # VectorWidth must not span tensor dim
      if state["KernelLanguage"] == "Source":
        if state["AssertFree0ElementMultiple"]<state["VectorWidth"]:
          reject(state, "packedC0 Source requires AF0EM>=VectorWidth (for loads and stores)")
      else:
        if state["AssertFree0ElementMultiple"]<state["VectorWidth"]\
          or state["AssertFree0ElementMultiple"] == 1:
            if state["VectorStore"] <= 0:
              state["_VectorStore"] = 0
            else:
              reject(state, "packedC0 Assembly requires AF0EM>=VectorWidth or not VectorStore (for stores)")

    state["AssignedDerivedParameters"] = True

    # UnrollLoopEfficiencyEnable does not work with f16/bf16/int8x4
    if globalParameters["UnrollLoopEfficiencyEnable"] and (state["ProblemType"]["DataType"].isHalf() or \
       state["ProblemType"]["DataType"].isBFloat16() or state["ProblemType"]["DataType"].isInt8x4()):
      reject(state, "UnrollLoopEfficiencyEnable does not support f16/bf16/int8x4")

    # UnrollLoopEfficiencyEnable supports only ThreadTile0,1=[6,4] or [4,6] or [4,4] or [6.6] or [8,4] or [4,8]
    if globalParameters["UnrollLoopEfficiencyEnable"] and \
      not ((state["ThreadTile0"] == 6 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 6) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 6 and state["ThreadTile1"] == 6) or \
           (state["ThreadTile0"] == 8 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 8)):
      reject(state, "UnrollLoopEfficiencyEnable does not support ThreadTile0,1 = [%u,%u]"%(state["ThreadTile0"], state["ThreadTile1"]))

    # Set E
    if state["ProblemType"]["UseE"]:
      if (state["_GlobalAccumulation"] == 'SingleBuffer') and state["GlobalSplitU"] > 1:
        reject(state, "GlobalSplitU > 1 only compatible with MultipleBuffer")
      if len(state["PackedC1IndicesX"]) > 1:
        reject(state, "Use E does not support len(PackedC1IndicesX) > 1.")
      if not state["BufferStore"]:
        reject(state, "Use E only supports BufferStore due to no suppress no store.")
      if state["StoreRemapVectorWidth"] and (state["GlobalSplitU"] == 1):
        reject(state, "Use E does not support StoreRemapVectorWidth if GSU == 1.")
      if state["GroupLoadStore"]:
        reject(state, "Use E does not support GroupLoadStore.")

    # Activation
    # Function call is set to false if GSU != 1 or Activation is not fused or ActivationType is not All.
    if not ((state["GlobalSplitU"] == 1) and state["ActivationFused"] and state["ProblemType"]["ActivationType"] == 'all') \
      and state["ActivationFuncCall"]:
      state["ActivationFuncCall"] = False

    if state["ActivationAlt"]:
      if state["GlobalSplitU"] > 1:
        # Turn off ActivationAlt if GSU > 1
        state["ActivationAlt"] = False
      if not state["ProblemType"]["Gradient"]:
        reject(state, "ActivationAlt does not support gradient.")

    # Bias reduction
    if state["ProblemType"]["UseBias"] and state["ProblemType"]["Gradient"]:
      if (state["ProblemType"]["BiasSrc"] == "A" or state["ProblemType"]["BiasSrc"] == "B"):
        if state["allowLRVWforTLUandMI"]:
          # Block for not verified.
          reject(state, "Bias reduction on A, B does not support allowLRVWforTLUandMI")
      if (state["_GlobalAccumulation"] == 'SingleBuffer') and state["GlobalSplitU"] > 1:
        reject(state, "GlobalSplitU > 1 only compatible with MultipleBuffer for bias reduction")
      if len(state["PackedC1IndicesX"]) > 1:
        reject(state, "Bias reduction does not support len(PackedC1IndicesX) > 1.")
      if not state["BufferStore"]:
        reject(state, "Bias reduction only supports BufferStore due to no suppress no store.")
      if state["StoreRemapVectorWidth"] and (state["GlobalSplitU"] == 1):
        reject(state, "Bias reduction does not support StoreRemapVectorWidth if GSU == 1.")
      if state["GroupLoadStore"]:
        reject(state, "Bias reduction does not support GroupLoadStore.")

  ########################################
  # create a dictionary with booleans on whether to include parameter in name
  @staticmethod
  def getMinNaming(objs):
    nonCKObjs = [obj for obj in objs if not isCustomKernelConfig(obj)]

    # early return
    if len(nonCKObjs) == 0:
      return {}

    # determine keys
    requiredParameters = {}
    if isinstance(nonCKObjs[0], Solution):
      keys = list(nonCKObjs[0]._state.keys())
    else:
      keys = list(nonCKObjs[0].keys())
    # only 1, rather than name being nothing, it'll be everything
    if len(nonCKObjs) == 1:
      for key in keys:
        if key in list(validParameters.keys()):
          requiredParameters[key] = False
    else:
      for key in keys:
        required = False
        if key in list(validParameters.keys()):
          for i in range(1, len(nonCKObjs)):
            if nonCKObjs[0][key] != nonCKObjs[i][key]:
              required = True
              break
        if required:
          requiredParameters[key] = True
        else:
          requiredParameters[key] = False

    if "MatrixInstM" in nonCKObjs[0]._state:
      # Use MIWaveGroup and MIWaveTile instead of WG and MT
      requiredParameters["MIWaveTile"]  = True
      requiredParameters["ThreadTile"]  = False

    requiredParameters["ProblemType"]       = False # always prepended
    requiredParameters["MacroTile0"]        = False # always prepended
    requiredParameters["MacroTile1"]        = False # always prepended
    requiredParameters["DepthU"]            = False # always prepended
    requiredParameters["MatrixInstruction"] = False # always prepended
    requiredParameters["MatrixInstM"]       = False # always prepended
    requiredParameters["MatrixInstN"]       = False # always prepended
    requiredParameters["MatrixInstK"]       = False # always prepended
    requiredParameters["MatrixInstB"]       = False # always prepended
    requiredParameters["MatrixInstBM"]      = False # always prepended
    requiredParameters["MatrixInstBN"]      = False # always prepended
    requiredParameters["CustomKernelName"]  = False # Will not affect naming
    requiredParameters["Fp16AltImpl"]       = False # Will show up as a different type

    requiredParameters["Kernel"]            = True  # distinguish kernels from solutions
                                                    # for single-source compilation
    return requiredParameters

  ########################################
  @ staticmethod
  def getNameFull(state):
    requiredParameters = {}
    for key in state:
      if key in list(validParameters.keys()):
        requiredParameters[key] = True
    if "MatrixInstM" in state:
      # Use MIWaveGroup and MIWaveTile instead of WG and MT
      requiredParameters["MIWaveTile"]  = True
      requiredParameters["ThreadTile"]  = False
    return Solution.getNameMin(state, requiredParameters)

  ########################################
  # Get Name Min
  @ staticmethod
  def getNameMin(state, requiredParameters):
    if isCustomKernelConfig(state):
      return state["CustomKernelName"]

    name = ""
    first = True
    # put problem first
    if "ProblemType" in state:
      name += str(state["ProblemType"]) + "_"
    if "MacroTile0" in state \
        and "MacroTile1" in state \
        and "DepthU" in state:
      name += "%s%ux%ux%u_" \
          % ( Solution.getParameterNameAbbreviation("MacroTile"), \
          state["MacroTile0"], state["MacroTile1"], state["DepthU"] )
    if "MatrixInstM" in state:
      name += "%s%ux%ux%ux%u_" \
          % ( Solution.getParameterNameAbbreviation("MatrixInstruction"), \
          state["MatrixInstM"], state["MatrixInstN"], state["MatrixInstK"], state["MatrixInstB"])
    name += "SN_" # LdcEqualsLdd Removed
    for key in sorted(state.keys()):
      if key in requiredParameters and key[0] != '_':
        if requiredParameters[key] and key != "CustomKernelName":
          if not first:
            name += "_"
          else:
            first = False
          name += "%s%s" % ( Solution.getParameterNameAbbreviation(key), \
              Solution.getParameterValueAbbreviation(key, state[key]) )
    return name

  ########################################
  # create a dictionary of lists of parameter values
  @staticmethod
  def getSerialNaming(objs):
    data = {}
    for objIdx in range(0, len(objs)):
      obj = objs[objIdx]
      for paramName in sorted(obj.keys()):
        if paramName in list(validParameters.keys()):
          paramValue = obj[paramName]
          if paramName in data:
            if paramValue not in data[paramName]:
              data[paramName].append(paramValue)
          else:
            data[paramName] = [ paramValue ]
    maxObjs = 1
    for paramName in data:
      if not isinstance(data[paramName][0],dict):
        data[paramName] = sorted(data[paramName])
      maxObjs *= len(data[paramName])
    numDigits = len(str(maxObjs))
    return [ data, numDigits ]

  ########################################
  # Get Name Serial
  @ staticmethod
  def getNameSerial(state, serialNaming):
    data = serialNaming[0]
    numDigits = serialNaming[1]

    serial = 0
    multiplier = 1
    for paramName in sorted(state.keys()):
      if paramName in list(validParameters.keys()):
        paramValue = state[paramName]
        paramData = data[paramName]
        paramNameMultiplier = len(paramData)
        if paramValue in paramData:
          paramValueIdx = paramData.index(paramValue)
        serial += paramValueIdx * multiplier
        multiplier *= paramNameMultiplier
    name = "%s%0*u" % ("S" if isinstance(state, Solution) else "K", \
        numDigits, serial)
    return name


  ########################################
  @ staticmethod
  def getParametersIndented(state, indent):
    s = ""
    s += "%sProblemType: %s\n" % (indent, str(state["ProblemType"]))
    for key in sorted(state):
      s += "%s%s: %s\n" % (indent, str(key), str(state[key]))
    return s

  ########################################
  @ staticmethod
  def getParameterNameAbbreviation( name ):
    return ''.join([c for c in name if not c.islower()])

  ########################################
  @ staticmethod
  def getParameterValueAbbreviation( key, value ):
    if key == 'ISA':
      return str(value[0]) + str(value[1]) + ('%x' % value[2])
    elif isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      if value >= 0:
        return "%u" % value
      else: # -1 -> n1
        return "n%01u" % abs(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, tuple):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += str(value[i])
      return abbrev
    elif isinstance(value, list):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += Solution.getParameterValueAbbreviation(key, value[i])
        if i < len(value)-1:
          abbrev += "_"
      return abbrev
    elif isinstance(value, dict):
      s =  "_".join(["%d%d"%(pos,k) for pos,k in value.items()])
      return s
    elif isinstance(value, float):
      val1 = int(value)
      val2 = int(round(value*100)) - int(value)*100
      if val2 > 0:
        s =  "%dp%s" % (val1,str(val2).zfill(2))
      else:
        s = "%d" % (val1)
      return s
    else:
      printExit('Parameter {key}={value} is new object type ({t})'.format(key=key, value=value, t=type(value)))
      return str(value)


  ##########################
  # make class look like dict
  def keys(self):
    return list(self._state.keys())

  def __len__(self):
    return len(self._state)

  def __iter__(self):
    return iter(self._state)

  def __getitem__(self, key):
    return self._state[key]

  def __setitem__(self, key, value):
    self._name = None
    self._state[key] = value

  def __str__(self):
    if self._name is None:
      self._name = Solution.getNameFull(self._state)
    return self._name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return deepcopy(self._state)

  def __hash__(self):
    return hash(str(self) + self._state.get("codeObjectFile", ""))
    #return hash(self.getAttributes())

  def __eq__(self, other):
    #return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
    return isinstance(other, Solution) and str(self) == str(other)

  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

  @property
  def enabledSplitLDS(self):
    return self["DepthULdsDivisor"] > 1

  @property
  def enabledSetPrioSplitLDS(self):
    # The interaction between SplitLDS's priority policy and StorePriorityOpt's is yet to be
    # investigated. For now, disable SplitLDS's priority policy when StorePriorityOpt is present
    # TODO: determine suitable priority policy when both are present
    return self.enabledSplitLDS and not self["StorePriorityOpt"]
