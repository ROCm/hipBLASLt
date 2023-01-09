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

from .Common import globalParameters, CHeader
from .KernelWriterBase import KernelWriterBase
from .TensileInstructions import DataType

class KernelWriterConversion(KernelWriterBase):

  def __init__(self, state):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]
    self.state["ActivationFused"] = state["ActivationFused"]

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()
    self.datatype = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
    if self.state["ProblemType"]["DataType"].isInt8() and self.state["ProblemType"]["ComputeDataType"].isSingle() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
      self.datatype = DataType('int32').toDevice(self.language)

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[self.state["ProblemType"]["Index0"]] = "0" + self.indexChars[self.state["ProblemType"]["Index0"]]
    self.indexChars[self.state["ProblemType"]["Index1"]] = "1" + self.indexChars[self.state["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[self.state["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[self.state["ProblemType"]["Index1"]]


  def functionSignature(self):
    kStr = ""

    # kernel name
    kStr += self.endLine
    kStr += "extern \"C\"\n"
    kStr += "__global__ "
    kStr += "void %s" % ( self.kernelName )
    kStr += "(" + self.endLine

    # pointers
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    ptrStr += '' if self.state["ProblemType"]["StridedBatched"] else '*'
    bStr = '' if self.state["ProblemType"]["StridedBatched"] else 'Batch'

    kStr += "  " + ptrStr + " * " + bStr + "D," + self.endLine
    kStr += "  " + self.datatype + " * W," + self.endLine
    kStr += "  " + ptrStr + " const * " + bStr + "C," + self.endLine

    # bias
    if self.state["ProblemType"]["UseBias"]:
      biasPtrStr = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
      kStr += "  " + biasPtrStr + " const * " + "Bias," + self.endLine

    # interface: ScaleD GSU>1 GSUA "MUL"
    if self.state["ProblemType"]["UseScaleD"]:
      scaleDPtrStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
      kStr += "  " + scaleDPtrStr + " const * " + "ScaleD," + self.endLine

    # alpha & beta
    kStr += "  %s const alpha,%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    kStr += "  %s const beta,%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)

    # activation
    activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
    enumName = "Tensile::ActivationType_%s"%activationCDataType.toChar()
    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      activationCDataType = self.state["ProblemType"]["ComputeDataType"] if self.state["ProblemType"]["ActivationHPA"] else \
                            self.state["ProblemType"]["DestDataType"]
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        kStr += "  %s const %s,%s" % (activationCDataType.toDevice(self.language), name, self.endLine)
      if self.state["ProblemType"]["ActivationType"] == 'all':
        kStr += "  %s const activationType,%s" % (enumName, self.endLine)

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideW%s,%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideC%s,%s" % (self.indexChars[i], self.endLine)

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int const size%s,%s" % (self.indexChars[i], self.endLine)

    # offset
    kStr += "  unsigned int offsetD,%s" % self.endLine
    kStr += "  unsigned int offsetC,%s" % self.endLine

    # gsu
    kStr += "  unsigned int const gsu)%s" % self.endLine

    return kStr


  def kernelBody(self):
    kStr = ""
    kStr += "{%s" % self.endLine
    problemType = self.state["ProblemType"]

    ########################################
    # defined initial strides
    firstStride = 0
    if problemType["UseInitialStridesCD"]:
      # no strides #defined
      lastStrideC = 0
      assert 0  # need to fix beta-clear routine to pass initial stride parms
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides */%s" % self.endLine
      lastStrideC = 1
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideW" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_D()
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_W()
    kStr += "#define GLOBAL_W(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideW%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideW%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_C()
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)
    kStr += "  if (id >= (size%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += "*size%s" % self.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += self.endLine
    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    for i in range(0, problemType["NumIndicesC"]):
      kStr += "  id%d = id %% size%s;%s" % (i, self.indexChars[i], self.endLine)
      kStr += "  id  = id / size%s;%s" % (self.indexChars[i], self.endLine)

    nonTileFreeIndices = []

    ########################################
    # apply batch
    if not self.state["ProblemType"]["StridedBatched"]:
      nonTileFreeIndices = list(range(0, self.state["ProblemType"]["NumIndicesC"]))
      nonTileFreeIndices.remove(self.state["ProblemType"]["Index0"])
      nonTileFreeIndices.remove(self.state["ProblemType"]["Index1"])

      kStr += self.endLine
      kStr += "  uint64_t wg = 0"
      batchStride = "1"
      for i in nonTileFreeIndices:
        kStr += " + id%d * %s " % (i, batchStride)
        batchStride += " * size%s" % self.indexChars[i]
      kStr += ";" + self.endLine

      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      kStr += "  " + ptrStr + " * D = BatchD[wg];" + self.endLine
      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      zeroStr = self.state["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)
      kStr += "  " + ptrStr + f" const* C = (beta == {zeroStr}) ? nullptr : BatchC[wg];" + self.endLine

    ########################################
    # apply offset
    kStr += self.endLine
    kStr += "  D = D + offsetD;" + self.endLine
    kStr += "  C = C + offsetC;" + self.endLine

    ########################################
    # D index
    kStr += self.endLine
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
    kStr += ");%s" % (self.endLine)

    # W index
    kStr += "  %s idxW = GLOBAL_W( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += 'id%d' % i
    kStr += ");%s" % (self.endLine)

    # D index
    kStr += "  %s idxC = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      kStr += ', ' if i else ''
      kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
    kStr += ");%s" % (self.endLine)

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    intermediateDataType = self.datatype
    if self.state["ProblemType"]["DataType"].isInt8() and self.state["ProblemType"]["ComputeDataType"].isSingle() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
      intermediateDataType = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)

    indexChar = self.indexChars[0]
    kStr += "  %s strideW = 1 + (size%s - 1) * strideW%s" % (self.uint64Str, indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (size%s - 1) * strideW%s" % (indexChar, indexChar)
    kStr += ";" + self.endLine

    kStr += "  " + intermediateDataType + " accum = 0;%s" % self.endLine
    kStr += "  " + self.datatype + " result = 0;%s" % self.endLine
    kStr += "  for (int i=0; i<gsu; i++) {%s" % self.endLine
    kStr += "    accum += W[idxW];%s" % self.endLine
    kStr += "    idxW  += strideW;%s" % self.endLine
    kStr += "  }%s" % self.endLine

    biasStr = ""
    if self.state["ProblemType"]["UseBias"]:
      biasStr = " + ((" + intermediateDataType + ")(Bias == 0 ? 0 : Bias[id0]))"

    scaleDStr = ""
    if self.state["ProblemType"]["UseScaleD"]:
      scaleDStr = " * ((" + intermediateDataType + ")(ScaleD == 0 ? 1 : ScaleD[id0]))"

    kStr += "  if( beta == (%s)0)%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    kStr += "    accum = ((" + intermediateDataType + ")alpha) * accum%s;%s" % (biasStr, self.endLine)
    kStr += "  else%s" % self.endLine
    kStr += "    accum = (((" + intermediateDataType + ")alpha) * accum + ((" + intermediateDataType + ")beta) * ((" + intermediateDataType + ")C[idxC])" + biasStr + ");" + self.endLine

    typeStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      typeActivationStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language) if self.state["ProblemType"]["ActivationHPA"] else \
                          self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      names = ""
      if self.state["ProblemType"]["ActivationType"] == 'all':
        names += ", activationType"
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        names += (", " + name)
      rvalueStr = "activation((%s)accum%s)" % (typeActivationStr, names)
    else:
      rvalueStr = "accum"
    if self.state["ProblemType"]["DestDataType"].isInt8() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
      rvalueStr = "min(127, max(-128, (int32_t)std::nearbyint(%s)))" % rvalueStr

    kStr += "  result = (%s)%s;%s" % (typeStr, rvalueStr, self.endLine)
    kStr += "  D[idxD] = (%s)(result%s);%s" % (typeStr, scaleDStr, self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideW" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideC" + self.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_W%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    name += "" if self.state["ProblemType"]["StridedBatched"] else "_GB"
    name += "_Bias%s"%self.state["ProblemType"]["BiasDataType"].toChar() if self.state["ProblemType"]["UseBias"] else ""
    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      if self.state["ProblemType"]["ActivationType"] == 'all':
        name += "_A"
      else:
        name += "_%s"%str(self.state["ProblemType"]["ActivationType"]).upper()
      name += ("h" if self.state["ProblemType"]["ActivationHPA"] else "")
    name += "_ScaleD" if self.state["ProblemType"]["UseScaleD"] else ""
    name += "_PostGSU"
    return name


  def getHeaderFileString(self):
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += CHeader
      fileString += "#pragma once\n\n"
      fileString += "\n"
      fileString += "#include <KernelHeader.h>\n\n"
      fileString += "#include <hip/hip_runtime.h>\n"
      fileString += "#include <hip/hip_fp16.h>\n"
      fileString += "\n"
      activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
      if self.state["ProblemType"]["ActivationType"] == 'all':
        fileString += "#include \"TensileActivation_%s_%s.h\"\n"%(activationCDataType.toChar(), \
                                                                  self.state["ProblemType"]["ActivationType"])
      fileString += "\n"

    fileString += self.functionSignature()
    fileString += ";\n"

    return fileString


  def getSourceFileString(self):
    fileString = ""
    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"%s.h\"\n" % self.kernelName
      fileString += "\n"

    fileString += self.functionSignature()
    fileString += self.kernelBody()

    return (0, fileString)
