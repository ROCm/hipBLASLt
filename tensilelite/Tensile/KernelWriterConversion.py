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

from copy import deepcopy

from .Common import globalParameters, CHeader
from .KernelWriterBase import KernelWriterBase
from .TensileInstructions import DataType

class KernelWriterConversion(KernelWriterBase):

  def __init__(self, state, load_vw):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]
    self.state["ActivationFused"] = state["ActivationFused"]
    self.state["GlobalSplitU"] = state["GlobalSplitU"]

    self.actGradientPrefix = ""
    if self.state["ProblemType"]["Gradient"]:
      self.actGradientPrefix = "Gradient"
    self.gaurdStr = "NG" if self.state["ProblemType"]["ActivationNoGuard"] else ""

    # setup load vector width
    self.num_elements_load = load_vw

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

  def functionArgument(self):
    kStr = ""

    # argument structure start
    kStr += self.endLine
    kStr += "struct __attribute__((__packed__)) argument_%s" % ( self.kernelName )
    kStr += "{" + self.endLine

    # pointers
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    ptrStr += '' if self.state["ProblemType"]["StridedBatched"] else '*'
    bStr = '' if self.state["ProblemType"]["StridedBatched"] else 'Batch'

    if self.state["ProblemType"]["UseE"]:
      ptrCStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
      ptrCStr += '' if self.state["ProblemType"]["StridedBatched"] else '*'
      kStr += "  " + ptrCStr + " * " + bStr + "E;" + self.endLine
    kStr += "  " + ptrStr + " * " + bStr + "D;" + self.endLine
    kStr += "  " + self.datatype + " * W;" + self.endLine
    kStr += "  " + ptrStr + " const * " + bStr + "C;" + self.endLine

    # bias
    if self.state["ProblemType"]["UseBias"]:
      if (not self.state["ProblemType"]["Gradient"]):
        biasPtrStr = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
        kStr += "  " + biasPtrStr + " const * " + "Bias;" + self.endLine
      elif self.state["ProblemType"]["Gradient"] and (self.state["ProblemType"]["BiasSrc"] == "A" or self.state["ProblemType"]["BiasSrc"] == "B"):
        biasPtrStr = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
        kStr += "  " + biasPtrStr + "* " + "Bias;" + self.endLine

    # interface: ScaleD GSU>1 GSUA "MUL"
    if self.state["ProblemType"]["UseScaleD"]:
      scaleDPtrStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
      kStr += "  " + scaleDPtrStr + " const * " + "ScaleD;" + self.endLine

    # alpha & beta
    kStr += "  %s const alpha;%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    kStr += "  %s const beta;%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)

    # activation
    activationCDataType = self.state["ProblemType"]["ActivationComputeDataType"]
    enumName = "Tensile::%sActivationType_%s"%(self.actGradientPrefix, activationCDataType.toChar())
    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      activationCDataType = self.state["ProblemType"]["ComputeDataType"] if self.state["ProblemType"]["ActivationHPA"] else \
                            self.state["ProblemType"]["DestDataType"]
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        kStr += "  %s const %s;%s" % (activationCDataType.toDevice(self.language), name, self.endLine)
      if self.state["ProblemType"]["ActivationType"] == 'all':
        kStr += "  %s const activationType;%s" % (enumName, self.endLine)

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    if self.state["ProblemType"]["UseE"]:
      for i in range(firstStrideCD, lastStrideC):
        kStr += "  unsigned int const strideE%s;%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s;%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideW%s;%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideC%s;%s" % (self.indexChars[i], self.endLine)

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      if i < (self.state["ProblemType"]["NumIndicesC"] - 1):
        kStr += "  unsigned int const size%s;%s" % (self.indexChars[i], self.endLine)
      else:
        kStr += "  unsigned int const size%s;%s" % (self.indexChars[i], self.endLine)

    # argument structure end
    kStr += "};" + self.endLine

    return kStr

  def functionSignature(self):
    kStr = ""

    # kernel name
    kStr += self.endLine
    kStr += "extern \"C\"\n"
    kStr += "__global__ "
    kStr += "void %s" % ( self.kernelName )
    kStr += "(" + self.endLine

    # kernel argument
    if self.state["ProblemType"]["GroupedGemm"]:
      kStr += "  uint32_t* wiTablePtr, argument_%s* argsPtr, uint32_t gemm_count)" % ( self.kernelName ) + self.endLine
    else:
      kStr += "  argument_%s arg)" % ( self.kernelName ) + self.endLine

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
    if self.state["ProblemType"]["UseE"]:
      for i in range(firstStride, lastStrideC):
        kStr += "#define strideE" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideW" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_E()
    if self.state["ProblemType"]["UseE"]:
      kStr += "#define GLOBAL_E(IDX%s" % self.indexChars[0]
      for i in range(1, problemType["NumIndicesC"]):
        kStr += ", IDX%s" % self.indexChars[i]
      indexChar = self.indexChars[0]
      kStr += ") (( (IDX%s)*strideE%s" % (indexChar, indexChar)
      for i in range(1, problemType["NumIndicesC"]):
        indexChar = self.indexChars[i]
        kStr += " + (IDX%s)*arg.strideE%s" % (indexChar, indexChar)
      kStr += " ))" + self.endLine

    # GLOBAL_D()
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*arg.strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_W()
    kStr += "#define GLOBAL_W(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideW%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*arg.strideW%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_C()
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*arg.strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    self.num_dword_load = int(self.num_elements_load * self.state["ProblemType"]["ComputeDataType"].numBytes() / 4)
    self.num_dword_store = int(self.num_elements_load * self.state["ProblemType"]["DestDataType"].numBytes() / 4)
    kStr += "#define NUM_ELEMENT_LOAD %d%s" % ( self.num_elements_load, self.endLine)
    kStr += "#define NUM_GSU %d%s" % (self.state["GlobalSplitU"], self.endLine)

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)

    ########################################
    # Grouped gemm: find index of gemm
    if self.state["ProblemType"]["GroupedGemm"]:
      kStr += self.endLine
      kStr += "  uint32_t left = 0;" + self.endLine
      kStr += "  uint32_t middle;" + self.endLine
      kStr += "  uint32_t right = gemm_count;" + self.endLine
      kStr += "  uint32_t wiMiddle, wiLeft;" + self.endLine
      kStr += "  uint32_t targetP1 = id + 1;" + self.endLine
      kStr += "  while(left < right)" + self.endLine
      kStr += "  {" + self.endLine
      kStr += "    middle = (left + right) / 2;" + self.endLine
      kStr += "    wiMiddle = wiTablePtr[middle];" + self.endLine
      kStr += "    if(wiMiddle < targetP1)" + self.endLine
      kStr += "    {" + self.endLine
      kStr += "      left = middle + 1;" + self.endLine
      kStr += "      wiLeft = wiMiddle;" + self.endLine
      kStr += "    }" + self.endLine
      kStr += "    else" + self.endLine
      kStr += "      right = middle;" + self.endLine
      kStr += "  }" + self.endLine
      kStr += "  id = id - wiLeft;"  + self.endLine
      kStr += "  argument_%s arg = argsPtr[left-1];" % ( self.kernelName ) + self.endLine

    ########################################
    # kernel start
    kStr += self.endLine
    kStr += "  if (id >= (arg.size%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += " * arg.size%s" % self.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += self.endLine
    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    for i in range(0, problemType["NumIndicesC"]):
      if i == 0:
        kStr += "  id%d = (id %% (arg.size%s/NUM_ELEMENT_LOAD)) * NUM_ELEMENT_LOAD;%s" % (i, self.indexChars[i], self.endLine)
        kStr += "  id  = id / (arg.size%s/NUM_ELEMENT_LOAD);%s" % (self.indexChars[i], self.endLine)
      else:
        kStr += "  id%d = id %% arg.size%s;%s" % (i, self.indexChars[i], self.endLine)
        kStr += "  id  = id / arg.size%s;%s" % (self.indexChars[i], self.endLine)

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
        batchStride += " * arg.size%s" % self.indexChars[i]
      kStr += ";" + self.endLine

      if self.state["ProblemType"]["UseE"]:
        ptrStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language)
        kStr += "  " + ptrStr + " * arg.E = arg.BatchE[wg];" + self.endLine
      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      kStr += "  " + ptrStr + " * arg.D = arg.BatchD[wg];" + self.endLine
      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      zeroStr = self.state["ProblemType"]["ComputeDataType"].zeroString(self.language, 1)
      kStr += "  " + ptrStr + f" const* arg.C = (arg.beta == {zeroStr}) ? nullptr : arg.BatchC[wg];" + self.endLine

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

    # C index
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

    destTypeStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)

    indexChar = self.indexChars[0]
    kStr += "  %s strideW = 1 + (arg.size%s - 1) * strideW%s" % (self.uint64Str, indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (arg.size%s - 1) * arg.strideW%s" % (indexChar, indexChar)
    kStr += ";" + self.endLine

    kStr += "  " + intermediateDataType + " accum[NUM_ELEMENT_LOAD] = {0};" + self.endLine
    kStr += "  " + destTypeStr + " result[NUM_ELEMENT_LOAD];" + self.endLine

    #TODO: workspace type is half precision
    if self.state["ProblemType"]["UseBias"] and self.state["ProblemType"]["Gradient"] and self.state["ProblemType"]["BiasSrc"] == "D":
      kStr += "  auto idxW_ori = idxW;%s"%self.endLine
    loadTypeStr = "float%s" % ( "" if self.num_dword_load == 1 else self.num_dword_load)
    storeTypeStr = "float%s" % ( "" if self.num_dword_store == 1 else self.num_dword_store)

    #Bias A/B
    if self.state["ProblemType"]["UseBias"] and self.state["ProblemType"]["Gradient"] and (self.state["ProblemType"]["BiasSrc"] == "A" or self.state["ProblemType"]["BiasSrc"] == "B"):
      size          = "arg.size0I" if self.state["ProblemType"]["BiasSrc"] == "A" else "arg.size1J"
      barrier       = "id1" if self.state["ProblemType"]["BiasSrc"] == "A" else "id0"
      biasIdxStr    = "id0" if self.state["ProblemType"]["BiasSrc"] == "A" else "id1"
      biasIdxGsuStr = biasIdxStr + "Gsu"
      biasPtrStr    = self.state["ProblemType"]["BiasDataType"].toDevice(self.language)
      kStr += "  if(%s == 0 && id2 == 0)%s"%(barrier, self.endLine)
      kStr += "  {%s" % self.endLine
      kStr += "    auto offset = strideW * NUM_GSU;%s"% self.endLine
      kStr += "    auto strideBias = %s;%s"%(size, self.endLine)
      kStr += "    auto %s = %s + offset;%s"%(biasIdxGsuStr, biasIdxStr, self.endLine)
      biasLoadCount = 1
      if self.num_dword_load != 1 and self.state["ProblemType"]["BiasSrc"] == "A":
        biasLoadCount = self.num_dword_load
      kStr += "    " + intermediateDataType + " biasAccum[%d] = {0};%s" % (biasLoadCount ,self.endLine)
      kStr += "    for (int i = 0; i < NUM_GSU; i++) {%s" % self.endLine
      for vIdx in range(biasLoadCount):
        kStr += "      biasAccum[%d] += arg.W[%s+%d];%s" % (vIdx, biasIdxGsuStr, vIdx, self.endLine)
      kStr += "      %s  += strideBias;%s" % (biasIdxGsuStr, self.endLine)
      kStr += "    }%s" % self.endLine
      for vIdx in range(biasLoadCount):
        kStr += "    arg.Bias[%s+%d] = (%s)biasAccum[%d];%s"%(biasIdxStr, vIdx, biasPtrStr, vIdx, self.endLine)
      kStr += "  }%s" % self.endLine
    kStr += self.endLine

    #Load GSU D buffer
    kStr += "  %s temp[NUM_GSU];" % loadTypeStr + self.endLine
    for gsuIdx in range(self.state["GlobalSplitU"]):
      kStr += "  temp[%d] = *((%s*)(arg.W+idxW));%s" % (gsuIdx, loadTypeStr, self.endLine)
      kStr += "  idxW  += strideW;" + self.endLine
    kStr += self.endLine

    #Accumlate all D buffer
    for gsuIdx in range(self.state["GlobalSplitU"]):
      if self.num_dword_load == 1:
        kStr += "  accum[0] += temp[%d];" % gsuIdx + self.endLine
      elif self.num_dword_load >= 2:
        kStr += "  accum[0] += temp[%d].x;" % gsuIdx + self.endLine
        kStr += "  accum[1] += temp[%d].y;" % gsuIdx + self.endLine
      if self.num_dword_load == 4:
        kStr += "  accum[2] += temp[%d].z;" % gsuIdx + self.endLine
        kStr += "  accum[3] += temp[%d].w;" % gsuIdx + self.endLine
    kStr += self.endLine

    accumStr = "accum"
    resultStr = "result"

    #alpha
    for vIdx in range(self.num_dword_load):
      kStr += "  %s[%d] *= (%s)arg.alpha;%s" % (accumStr, vIdx, intermediateDataType, self.endLine)
    kStr += self.endLine

    #Beta
    kStr += "  if(arg.beta != (%s)0){%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    for vIdx in range(self.num_dword_load):
      kStr += "    %s[%d] += arg.beta * (%s)arg.C[idxC+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
    kStr += "  }" + self.endLine
    kStr += self.endLine

    #Bias
    if self.state["ProblemType"]["UseBias"] and (not self.state["ProblemType"]["Gradient"]):
      kStr += "  if(arg.Bias != 0){" + self.endLine
      for vIdx in range(self.num_dword_load):
        kStr += "    %s[%d] += (%s)arg.Bias[id0+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
      kStr += "  }" + self.endLine
      kStr += self.endLine

    #Handle E
    if self.state["ProblemType"]["UseE"]:
      if self.state["ProblemType"]["Gradient"]:
        kStr += "  %s idxE = GLOBAL_E( (%s)" % (self.uint64Str, self.uint64Str)
        for i in range(problemType["NumIndicesC"]):
          kStr += ', ' if i else ''
          kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
        kStr += ");%s" % (self.endLine)
        kStr += "  %s dataE[%d];%s" % (intermediateDataType, self.num_dword_load, self.endLine)
        for vIdx in range(self.num_dword_load):
          kStr += "  dataE[%d] = (%s)arg.E[idxE+%d];%s" % ( vIdx, intermediateDataType, vIdx, self.endLine)
      else:
        # E index
        kStr += "  if( arg.E != nullptr)%s" % (self.endLine)
        kStr += "  {%s" % (self.endLine)
        kStr += "    %s idxE = GLOBAL_E( (%s)" % (self.uint64Str, self.uint64Str)
        for i in range(problemType["NumIndicesC"]):
          kStr += ', ' if i else ''
          kStr += '0'  if i in nonTileFreeIndices else ('id%d' % i)
        kStr += ");%s" % (self.endLine)
        for vIdx in range(self.num_dword_load):
          kStr += "    arg.E[idxE+%d] = (%s)(accum[%d]);%s" % (vIdx, intermediateDataType, vIdx, self.endLine)
        kStr += "  }%s" % (self.endLine)

    #Activation
    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      typeActivationStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language) if self.state["ProblemType"]["ActivationHPA"] else \
                          self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      actArgs = ""
      if self.state["ProblemType"]["ActivationType"] == 'all':
        actArgs += ", arg.activationType"
      for args in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        actArgs += (", " + "arg." + args)
      for vIdx in range(self.num_dword_load):
        if self.state["ProblemType"]["Gradient"]:
          kStr += "  %s[%d] *= activation%s((%s)dataE[%d]%s);%s" % (accumStr, vIdx, self.gaurdStr, typeActivationStr, vIdx, actArgs, self.endLine)
        else:
          kStr += "  %s[%d] = activation%s((%s)%s[%d]%s);%s" % (accumStr, vIdx, self.gaurdStr, typeActivationStr, accumStr, vIdx, actArgs, self.endLine)
      kStr += self.endLine

    #ScaleD
    if self.state["ProblemType"]["UseScaleD"]:
      kStr += "  if(arg.ScaleD != nullptr){" + self.endLine
      for vIdx in range(self.num_dword_load):
        kStr += "    %s[%d] *= (%s)arg.ScaleD[id0+%d];%s" % (accumStr, vIdx, intermediateDataType, vIdx, self.endLine)
      kStr += "  }" + self.endLine
      kStr += self.endLine

    #Output high precision D to WS
    if self.state["ProblemType"]["UseBias"] and self.state["ProblemType"]["Gradient"] and self.state["ProblemType"]["BiasSrc"] == "D":
      for vIdx in range(self.num_dword_load):
        kStr += "  arg.W[idxW_ori+%d] = accum[%d];%s" % (vIdx, vIdx, self.endLine)

    #Saturation
    if self.state["ProblemType"]["DestDataType"].isInt8() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
      for vIdx in range(self.num_dword_load):
        kStr += "  %s[%d] *= min(127, max(-128, (int32_t)std::nearbyint(%s[%d])));%s" % (accumStr, vIdx, accumStr, vIdx, self.endLine)
      kStr += self.endLine

    #covert to output
    for vIdx in range(self.num_dword_load):
      kStr += "  %s[%d] = (%s)%s[%d];%s" % (resultStr, vIdx, destTypeStr, accumStr, vIdx, self.endLine)

    if self.num_elements_load > 1:
      kStr += "  *(%s *)(arg.D+idxD) = *(%s *)%s;%s" % (storeTypeStr, storeTypeStr, resultStr, self.endLine)
    else:
      kStr += "  arg.D[idxD] = (%s)(%s[0]);%s" % (destTypeStr, resultStr, self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    kStr += "#undef NUM_GSU" + self.endLine
    kStr += "#undef NUM_ELEMENT_LOAD" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideW" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideC" + self.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_W%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)
    if self.state["ProblemType"]["UseE"]:
      kStr += "#undef GLOBAL_E%s" % (self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    if self.state["ProblemType"]["GroupedGemm"]:
      name += "_GG"
    else:
      name += "" if self.state["ProblemType"]["StridedBatched"] else "_GB"
    if self.state["ProblemType"]["UseBias"]:
      if self.state["ProblemType"]["Gradient"]:
        name += "_DBias%s%s"%(self.state["ProblemType"]["BiasSrc"], self.state["ProblemType"]["BiasDataType"].toChar())
      else:
        name += "_Bias%s"%self.state["ProblemType"]["BiasDataType"].toChar()
    if self.state["ProblemType"]["UseE"]:
      if self.state["ProblemType"]["Gradient"]:
        name += "_Grad%s"%self.state["ProblemType"]["ComputeDataType"].toChar()
      else:
        name += "_Aux%s"%self.state["ProblemType"]["ComputeDataType"].toChar()

    if ((self.state["ProblemType"]["ActivationType"] != 'none') and self.state["ActivationFused"]):
      if self.state["ProblemType"]["ActivationType"] == 'all':
        name += "_A"
      else:
        name += "_%s"%str(self.state["ProblemType"]["ActivationType"]).upper()
      name += ("h" if self.state["ProblemType"]["ActivationHPA"] else "")
      name += ("ng" if self.state["ProblemType"]["ActivationNoGuard"] else "")
    name += "_ScaleD" if self.state["ProblemType"]["UseScaleD"] else ""
    name += "_PostGSU" + str(self.state["GlobalSplitU"])
    if self.num_elements_load != None:
      name += "_VW" + str(self.num_elements_load)
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
        fileString += "#include \"Tensile%sActivation%s_%s_%s.h\"\n"%(self.actGradientPrefix, \
                                                                      self.gaurdStr, \
                                                                      activationCDataType.toChar(), \
                                                                      self.state["ProblemType"]["ActivationType"])
      fileString += "\n"

    fileString += self.functionArgument()
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
