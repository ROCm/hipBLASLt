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

from .TensileInstructions import KernelBody, Label, Macro, Module, RegSet, SrdUpperValue, \
                          StructuredModule, TextBlock, ValueEndif, ValueIf, ValueSet, SignatureBase, \
                          MUBUFModifiers, RegisterContainer, InstType, SelectBit, SGetPositivePCOffset, \
                          SLongBranchPositive, SCLongBranchScc0, SCLongBranchScc1, \
                          SBranchIfZero, SBranchIfNotZero, SMulInt64to32, DSInit, VCvtBF16toFP32, \
                          ArgumentLoader, bomb, vectorStaticDivideAndRemainder, \
                          vectorStaticDivide, vectorStaticRemainder, \
                          scalarStaticDivideAndRemainder, sMagicDiv, staticMultiply, \
                          scalarStaticMultiply, MacroVMagicDiv, MacroVDynamicScalarDiv, \
                          RegisterPool, allocTmpGpr, RegisterPoolResource, Holder, \
                          vgpr, sgpr, accvgpr, mgpr, log2, ceilDivide, DataType, \
                          dataTypeToMfmaInstTypePair
from .TensileInstructions.Instructions import *
from .TensilePass import getActivationFunctionModuleName, getActivationBranchModuleName
from .Common import globalParameters, print2, printExit, printWarning, roundUp
from .Component import Component
from .KernelWriter import KernelWriter
from .KernelWriterModules import *
from .SolutionStructs import isPackedIndex
from .AsmStoreState import StoreState
from .Activation import ActivationType, ActivationModule

from math import ceil, log
from copy import deepcopy
from typing import NamedTuple

################################################################################
# Assembly Kernel
################################################################################

class KernelWriterAssembly(KernelWriter):

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    super(KernelWriterAssembly, self).__init__( \
        kernelMinNaming, kernelSerialNaming)

  def getVgprOccupancy(self, numThreads, vgprs, unifiedVgprRegs=False):
    multiplier = int(ceil(max(numThreads, 256) / 256.0)) # example: wg=512 multiplier=2, 1024=4
    maxOccupancy = self.consts.maxOccupancy//multiplier

    vgprAllocateAligned = 4    if not unifiedVgprRegs else 8
    totalVgprs = self.consts.maxVgprs if not unifiedVgprRegs else self.consts.maxVgprs*2
    vgprsAligned = int(ceil(vgprs/vgprAllocateAligned))*vgprAllocateAligned
    vgprsAligned *= multiplier

    if   vgprsAligned > totalVgprs:  return 0
    elif vgprsAligned < 1:           return maxOccupancy
    occupancy = min(totalVgprs//vgprsAligned, maxOccupancy)

    #print("vgprs = ", vgprs, "vgprsAligned = ", vgprsAligned, "unifiedVgprRegs = " ,unifiedVgprRegs, "Occupancy = ", occupancy)

    return occupancy

  ########################################
  def getOccupancy(self, numThreads, vgprs, ldsSize, accvgprs=0, unifiedVgprRegs=False):

    ldsLimitedOccupancy = self.getLdsLimitedOccupancy(ldsSize)

    if not unifiedVgprRegs:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, vgprs,          unifiedVgprRegs)
      accvgprLimitedOccupancy = self.getVgprOccupancy(numThreads, accvgprs,       unifiedVgprRegs)
    else:
      vgprLimitedOccupancy    = self.getVgprOccupancy(numThreads, vgprs+accvgprs, unifiedVgprRegs)
      accvgprLimitedOccupancy = vgprLimitedOccupancy

    return min(ldsLimitedOccupancy, vgprLimitedOccupancy, accvgprLimitedOccupancy)

  # TODO: also consider sgpr
  def getMaxRegsForOccupancy(self, numThreads, vgprs, ldsSize, accvgprs=0, unifiedVgprRegs=False):
    lastVgprs = vgprs
    considerAccVgprs = 0       if not unifiedVgprRegs else accvgprs
    totalVgprs = self.consts.maxVgprs if not unifiedVgprRegs else self.consts.maxVgprs*2

    initOccupancy = self.getOccupancy(numThreads, vgprs, ldsSize, accvgprs, unifiedVgprRegs)
    if initOccupancy == 0: return lastVgprs

    while (vgprs + considerAccVgprs) < totalVgprs and vgprs < self.consts.maxVgprs:
      vgprs += 1
      if self.getVgprOccupancy(numThreads, vgprs + considerAccVgprs, unifiedVgprRegs) >= initOccupancy:
        lastVgprs = vgprs
        next
      else:
        break

    return lastVgprs

  @staticmethod
  def getLdsLimitedOccupancy(ldsSize):
    maxLds = 65536
    # As ldsSize gets large, rounding might push us slightly higher than maxLds.
    # Clamp at maxLds
    ldsSize = min(ldsSize + 255, maxLds) & 0x1ff00 # 256-byte granularity

    ldsLimitedOccupancy = maxLds//ldsSize
    return ldsLimitedOccupancy

  @staticmethod
  def getLdsSize(kernel):
    ldsSize = kernel["LdsNumElements"] * kernel["ProblemType"]["DataType"].numBytes()
    return ldsSize

  ########################################
  def sizeRef(self, idx):
    """
    Return sgpr() or const with the specified size
    See above definitions for how these are mapped to Free or Sum sizes
    based on the problem definition.
    """
    idxChar= globalParameters["IndexChars"][idx]
    return sgpr("Size%s"%idxChar)

  def loopChar(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return globalParameters["IndexChars"][loopDim]

  def loopSizeRef(self, kernel, loopIdx):
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    return self.sizeRef(loopDim)

  def loopCounterName(self, kernel, loopIdx):
    return "LoopCounter%s"%(self.loopChar(kernel, loopIdx))

  def loopCounter(self, kernel, loopIdx):
    """
    Return loopCounter for loopIdx wrapped in "SGPR" syntax
    loop idx is 0...unrollIdx
    """
    return sgpr(self.loopCounterName(kernel,loopIdx))

  def checkLastIter(self, kernel, comment="at last iteration?"):
    """ Return last iteration of unroll loop. """
    return SCmpEQU32(src0=self.loopCounter(kernel, self.states.unrollIdx), src1=0, comment=comment)

  def isConstUnitStride(self, stride):
      if isinstance(stride, RegisterContainer):
        return False
      return stride.startswith("const")

  ########################################
  def strideRef(self, tc, dim):
    """
    Return sgpr with specified stride or define starting with const if constant.
    dim is index 0...max indices and is in global index space.
    """
    problemType = self.states.kernel["ProblemType"]
    if tc in ['A','B']:
      if not problemType["UseInitialStridesAB"] and \
          dim == problemType["IndexAssignments%s"%tc][0]:
        return ("constStride%s%s"%(tc,self.states.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.states.indexChars[dim]))
    elif tc in ['D','C']:
      if not problemType["UseInitialStridesCD"] and dim == 0:
        return ("constStride%s%s"%(tc,self.states.indexChars[dim]))
      else:
        return sgpr("Stride%s%s"%(tc,self.states.indexChars[dim]))
    else:
      raise ValueError("unexpected tensorChar='%s' in stride function"%tc)

  ##############################################################################
  # Find Memory Instruction For Width and Stride
  ##############################################################################
  def findMemoryInstructionForWidthStride(self, width, strides, combine, \
      instructions):
    for i in range(0, len(instructions)):
      instruction = instructions[i]
      numAddresses = instruction.numAddresses
      numOffsets = instruction.numOffsets
      offsetMultiplier = instruction.offsetMultiplier
      blockWidth = instruction.blockWidth
      valid = True
      if width < blockWidth:
        valid = False
      if combine: # try to combine ops
        if numOffsets > 0: # if inst combines using offsets
          for stride in strides:
            if stride % offsetMultiplier != 0:
              valid = False
      else: # don't try to combine ops
        if numOffsets > 1 or numAddresses > 1:
          valid = False
      if valid:
        return i
      else:
        continue

    printWarning("Could not find valid memory instruction for width=%f" % width)
    return len(instructions)

  ##############################################################################
  # Select Memory Instruction
  # when selecting instruction, need to support stride in both dims
  ##############################################################################
  def selectMemoryInstruction(self,
      operation, # ReadGlobal, WriteLocal, ReadLocal
      width, # num registers 1 chunk
      write2, # Para, Perp, None
      para2, # NumLoadsPara >= 2
      perp2, # NumLoadsPerp >= 2
      strides ):

    #instructions = self.memoryArchitecture[operation]
    instructions = self.memoryInstructions[operation]
    # try to combine
    if (write2 == "Coalesced" and para2) \
        or (write2 == "Perpendicular" and perp2):
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, True, instructions)
    # don't or can't combine
    else:
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, False, instructions)

    if instructionIdx < len(instructions): # found
      return instructionIdx
    else:
      raise RuntimeError("Could not find valid memory instruction for operation=%s, width=%f, kernel=%s" %(operation, width, self.states.kernelName))

  def initGlobalReadMemoryInstruction(self, instructions, tP, bpr):
    # globalRead instruction; no flat_load2_*
    globalReadWidth = float(tP["nrcv"]*tP["bpe"])/bpr
    globalRead2Coalesced = tP["nrc"] > 1
    globalRead2Perpendicular = tP["nrp"] > 1
    globalReadInstructionIdx = self.selectMemoryInstruction("GlobalRead", globalReadWidth, \
                                False, \
                                globalRead2Coalesced, globalRead2Perpendicular, [] )

    tP["globalReadInstruction"] = instructions["GlobalRead"][globalReadInstructionIdx]

  def initLocalWriteMemoryInstruction(self, instructions, kernel, tP, bpr):
    ########################################
    # localWrite instruction
    # for local, tile->para, unroll->perp
    # self.localWriteWidth = 1 if tP["wtc"] else kernel["VectorWidth"]
    # wtc = writeTileDimComponents
    localWriteWidth = tP["nwcv"]*tP["bpe"]//bpr
    if localWriteWidth < 1:
      localWriteWidth = (1.0*tP["nwcv"]*tP["bpe"])/bpr
    localWrite2Coalesced = tP["nrc"]>1 or tP["wtc"]
    localWrite2Perpendicular = tP["nrp"]>1
    # localWrite stride tile
    if tP["tlu"]:
      if tP["wtc"]:
        localWriteStrideTile = 1
      else:
        localWriteStrideTile = kernel[tP["lsc"]]
    else:
      localWriteStrideTile = kernel[tP["lsp"]]
    localWriteStrideTile = localWriteStrideTile*tP["bpe"]//bpr
    # localWrite stride unroll
    if tP["tlu"]:
      localWriteStrideUnroll = kernel[tP["lsc"]]*kernel[tP["mt"]]
    else:
      if tP["wtc"]:
        localWriteStrideUnroll = 1*kernel[tP["mt"]]
      else:
        localWriteStrideUnroll = kernel[tP["lsc"]]*kernel[tP["mt"]]
    localWriteStrideUnroll = \
        (localWriteStrideUnroll*tP["bpe"])//bpr
    localWriteInstructionIdx = self.selectMemoryInstruction("LocalWrite", localWriteWidth, \
                                False, \
                                localWrite2Coalesced, localWrite2Perpendicular,
                                [localWriteStrideTile, localWriteStrideUnroll] )

    tP["localWrite2Coalesced"]     = localWrite2Coalesced
    tP["localWrite2Perpendicular"] = localWrite2Perpendicular
    tP["localWriteStrideTile"]     = localWriteStrideTile
    tP["localWriteStrideUnroll"]   = localWriteStrideUnroll
    tP["localWriteInstruction"]    = instructions["LocalWrite"][localWriteInstructionIdx]

  def initLocalReadMemoryInstruction(self, instructions, kernel, tP, bpr, lrvw):

    tChar = "A" if tP["isA"] else "B"
    localReadWidth = (kernel["VectorWidth"] * tP["bpe"]) // bpr
    if kernel["EnableMatrixInstruction"]:
      if tP["tlu"] and kernel["allowLRVWforTLUandMI"]:
        localReadWidth = (lrvw * tP["bpe"]) // bpr
      else:
        localReadWidth = tP["bpe"] / bpr
    if kernel["UnrollMajorLDS%s"%tChar]:
      localReadWidth = (lrvw * tP["bpe"]) // bpr
    # for directToLds x2/x4 support
    if kernel["DirectToLds%s"%tChar]:
      localReadWidth  = 1    # for fp64 its f32

    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    localReadStrideCoalesced = \
        kernel[tP["tt"]] * tP["bpe"]//bpr
    localRead2Coalesced = kernel[tP["tt"]]//kernel["VectorWidth"] > 1
    localReadInstructionIdx = self.selectMemoryInstruction("LocalRead", localReadWidth, \
                              False, \
                              localRead2Coalesced, localRead2Perpendicular,
                              [localReadStrideCoalesced] )
    tP["localRead2Coalesced"]      = localRead2Coalesced
    tP["localRead2Perpendicular"]  = localRead2Perpendicular
    tP["localReadStrideCoalesced"] = localReadStrideCoalesced
    tP["localReadInstruction"]     = instructions["LocalRead"][localReadInstructionIdx]

  def allocTmpSgpr(self, num: int, alignment=None, tag=None):
    def overflowListener(e):
      self.states.overflowedResources = 2
      if self.db["AssertOnSgprOverflow"]:
        raise e

    tmpSgpr = allocTmpGpr(self.sgprPool, num, self.consts.maxSgprs, alignment, tag, overflowListener)
    return tmpSgpr

  def defineSgpr(self, name, numSgprs, align=1):
    if numSgprs == 0: return

    sgprIdx = self.sgprPool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=0)
    #self.sgprIdx = roundUpToNearestMultiple(self.sgprIdx,align)
    #print (name, "->", self.sgprIdx, "+", numSgprs)
    self.sgprs[name] = sgprIdx

    return sgprIdx

  def undefineSgpr(self, name):
    self.sgprPool.checkIn(self.sgprs[name])
    # later references will result in compile-time error (with odd 'error: expected relocatable expression')
    # and 'Kernel ... not found in any loaded module'
    # TODO: temporarily disable undef as it seems to have issues
    return ValueSet(name=name, value="UNDEF", format = -1)

  def defineVariableSgprs(self, kernel):
    #------------------------
    # Registers defined below this point are not available in the post-loop
    # Post-loop is after tail loop exits, ie the store code.
    # (we reclaim them to use as temps, typically for execmasks)
    # Mostly impacts flat kernels and GSU edge since these need SGPR
    # for conditionals
    # self.states.lastPostLoopSgpr = self.sgprPool.size()
    if kernel["BufferLoad"]:
       # resource descriptor (SRD) A and B, must be aligned on 4-SGPR boundary
      self.defineSgpr("SrdA", 4, 4)
      self.defineSgpr("SrdB", 4, 4)

    if self.states.use64bShadowLimit:
      # If need more SGPR could overlap this with the Tensor2dSize regs
      self.defineSgpr("ShadowLimitA", 2, 2)
      self.defineSgpr("ShadowLimitB", 2, 2)

    if self.states.staggerU:
      self.defineSgpr("StaggerUIter", 1)  # stagger loop iterations, used for various iter counts in the code
      self.defineSgpr("WrapUA", 2)  # Bytes to add to SrdA to reset address from N-1 iter to AddressA
      self.defineSgpr("WrapUB", 2)  # Bytes to add to SrdB to reset address from N-1 iter to AddressB

    self.defineSgpr("GlobalReadIncsA", self.states.a.numSgprGlobalReadIncs)
    self.defineSgpr("GlobalReadIncsB", self.states.b.numSgprGlobalReadIncs)

    if kernel["LocalWriteUseSgprA"]:
        self.defineSgpr("LocalWriteAddrA", 1)
    if kernel["LocalWriteUseSgprB"]:
        self.defineSgpr("LocalWriteAddrB", 1)

    if kernel["_UseSgprForGRO"]:
      needFirstSgprOffset = kernel["DirectToLdsA"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.states.a.numVgprGlobalReadOffsets if needFirstSgprOffset else (self.states.a.numVgprGlobalReadOffsets-1)
      self.defineSgpr("ScalarGlobalReadOffsetA", numberOfSgpr)

      needFirstSgprOffset = kernel["DirectToLdsB"] and kernel["UseInstOffsetForGRO"]
      numberOfSgpr = self.states.b.numVgprGlobalReadOffsets if needFirstSgprOffset else (self.states.b.numVgprGlobalReadOffsets-1)
      self.defineSgpr("ScalarGlobalReadOffsetB", numberOfSgpr)

    # debug flag to allocate dummy / unused sgpr
    # useful when comparing code that adds new kernel arguments to see what
    # was actually changed
    numDummySgpr= 0
    for i in range(numDummySgpr):
      self.defineSgpr("DummySgpr%d"%i, 1)

    if self.sgprPool.size() > self.consts.maxSgprs:
      print ("warning: Number of defined SGPRS (%d) overflowed max SGPRS (%d)." \
               % (self.sgprPool.size(), self.consts.maxSgprs))

  ##############################################################################
  def functionSignature(self) -> SignatureBase:
    """
    Function Signature
    called after rest of code
    """
    signatureClass = Component.Signature.find(self)
    signature = signatureClass(self)
    return signature

  def macroAndSet(self, kernel, tPA, tPB) -> Module:
    module = Module("MacroNSet")
    module.add(MacroVMagicDiv(kernel["MagicDivAlg"]))

    ########################################
    # VGPR Macros
    ########################################
    module.addComment2("VGPR Assignments")
    module.addComment0("ValuC range: [%u-%u), %s"%(self.states.c.startVgprValu, self.states.c.startVgprValu+self.states.c.numVgprValu, \
                           "serializedStore enabled" if self.states.serializedStore else ""))
    module.add(RegSet("v", "vgprValuC", self.states.c.startVgprValu))

    module.addComment0("ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx")
    # PLR index: from X0 to X<LoopIters-1> (at most) -> VGPRs will be duplicated LoopIters times (at most)
    # eg, if LoopIters = 4, there would be at most 4*VGPRs
    # PLR = kernel["PrefetchLocalRead"] if kernel["PrefetchLocalRead"] < kernel["LoopIters"] else kernel["LoopIters"] - 1
    PLR = min(kernel["PrefetchLocalRead"], kernel["LoopIters"]-1)
    numBi = PLR+1
    # double the number of VgprValue if self.states.vgprValuDouble is true
    if self.states.vgprValuDouble:
      numBi *= 2
    ri = 0
    if self.states.a.numVgprValu > 0: # Do not generate vgprValuA if numVgprValuA is 0
      for bi in range(0,numBi): # buffer indices
        for iui in range(0, kernel["InnerUnroll"]):
          module.add(RegSet("v", "vgprValuA_X%u_I%u"%(bi,iui), self.states.a.startVgprValu +ri))
          ri += self.states.a.numVgprValuPerBlock
    if not kernel["DirectToLdsA"] or self.do["KeepDirectToLdsAlloc"]:
        module.add(RegSet("v", "vgprG2LA", self.states.a.startVgprG2L))
        if kernel["DirectToVgprA"]:
          # additional definition G2LA0, G2LA1 for swapping register sets
          module.add(RegSet("v", "vgprG2LA0", self.states.a.startVgprG2L))
          module.add(RegSet("v", "vgprG2LA1", self.states.a.startVgprG2L + self.states.a.numVgprG2L//2))

    ri = 0
    if self.states.b.numVgprValu > 0: # Do not generate vgprValuB if numVgprValuB is 0
      for bi in range(0,numBi): # buffer indices
        for iui in range(0, kernel["InnerUnroll"]):
          module.add(RegSet("v", "vgprValuB_X%u_I%u"%(bi,iui), self.states.b.startVgprValu+ri))
          ri += self.states.b.numVgprValuPerBlock
    if not kernel["DirectToLdsB"] or self.do["KeepDirectToLdsAlloc"]:
        module.add(RegSet("v", "vgprG2LB", self.states.b.startVgprG2L))
        if kernel["DirectToVgprB"]:
          # additional definition G2LB0, G2LB1 for swapping register sets
          module.add(RegSet("v", "vgprG2LB0", self.states.b.startVgprG2L))
          module.add(RegSet("v", "vgprG2LB1", self.states.b.startVgprG2L + self.states.b.numVgprG2L//2))
    if not kernel["LocalWriteUseSgprA"] and self.states.a.numVgprLocalWriteAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteAddrA", \
          self.states.a.startVgprLocalWriteAddr))
      if self.states.a.numVgprLocalWriteAddr > 1:
        module.add(RegSet("v", "vgprLocalWriteAddrOverhangA", \
            self.states.a.startVgprLocalWriteAddr+1))
    if not kernel["LocalWriteUseSgprB"] and self.states.b.numVgprLocalWriteAddr > 0:
      module.add(RegSet("v", "vgprLocalWriteAddrB", \
          self.states.b.startVgprLocalWriteAddr))
      if self.states.b.numVgprLocalWriteAddr > 1:
        module.add(RegSet("v", "vgprLocalWriteAddrOverhangB", \
            self.states.b.startVgprLocalWriteAddr+1))
    if kernel["BufferLoad"]:
      module.add(RegSet("v", "vgprGlobalReadOffsetA", \
          self.startVgprGlobalReadOffsetA))
      module.add(RegSet("v", "vgprGlobalReadOffsetB", \
          self.startVgprGlobalReadOffsetB))
    else:
      module.add(RegSet("v", "vgprGlobalReadAddrA", \
          self.startVgprGlobalReadAddressesA))
      module.add(RegSet("v", "vgprGlobalReadAddrB", \
          self.startVgprGlobalReadAddressesB))

    if self.states.globalReadIncsUseVgpr:
      module.add(RegSet("v", "vgprGlobalReadIncsA", \
          self.startVgprGlobalReadIncsA))
      module.add(RegSet("vgprGlobalReadIncsB", "v", \
          self.startVgprGlobalReadIncsB))
    if self.states.a.numVgprLocalReadAddr > 0:
      module.add(RegSet("v", "vgprLocalReadAddrA", \
          self.states.a.startVgprLocalReadAddr))
    if self.states.b.numVgprLocalReadAddr > 0:
      module.add(RegSet("v", "vgprLocalReadAddrB", \
          self.states.b.startVgprLocalReadAddr))

    if kernel["ProblemType"]["DataType"].isDoubleComplex() and kernel["MIArchVgpr"]:
      module.add(RegSet("v", "vgprAlphaTmp", \
          self.states.startVgprAlphaTmp))

    module.add(RegSet("v", "vgprSerial", self.states.startVgprSerial))

    if globalParameters["DebugKernel"]:
      module.add(RegSet("v", "vgprAddressDbg", \
          self.states.startVgprAddressDbg))
    #module.addComment0("Occu: %u waves/simd" % self.numWavesPerSimd )
    # module.addComment0("Num VGPR=%u"%self.vgprPool.size())
    # module.addComment0("Num AccVGPR=%u"%self.agprPool.size())

    ########################################
    # SGPR Macros
    ########################################
    module.addComment2("SGPR Assignments")

    # Emit declarations for all sgprs allocated with defineSgpr
    # in the order they were declared
    for skey in self.sgprs:
      module.add(RegSet("s", "sgpr"+skey, self.sgprs[skey]))
    # module.addComment0("max SGPR=%u"%self.sgprPool.size())

    module.addSpaceLine()
    module.addComment0("Size Assignments")
    problemType = kernel["ProblemType"]
    for idx in range(max(problemType["IndexAssignmentsA"] + problemType["IndexAssignmentsB"])+1):
      idxChar= globalParameters["IndexChars"][idx]
      if idx in problemType["IndicesFree"] or idx in problemType["IndicesBatch"]:
        idxType="Free"
      elif idx in problemType["IndicesSummation"]:
        idxType="Sum"
        idx = idx - problemType["NumIndicesC"]
      else:
        raise ValueError("unexpected index type in size assignments")

      module.add(RegSet("s", "sgprSize%s"%(idxChar), \
                  "sgprSizes%s"%idxType, idx))

    module.addSpaceLine()
    module.addComment0("Stride Assignments")
    for tc in ('D','C'):
      for idx in range(0, problemType["NumIndicesC"]):
        i = idx
        idxChar= self.states.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesCD"]:
          module.add(ValueSet("constStride%s%s"%(tc,idxChar), 1))
        else:
          if not kernel["ProblemType"]["UseInitialStridesCD"]:
            i = i-1
          module.add(RegSet("s", "sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s"%tc, i))

    for tc in ('A','B'):
      for i, idx in enumerate(problemType["IndexAssignments%s"%tc]):
        idxChar= self.states.indexChars[idx]
        if i == 0 and not kernel["ProblemType"]["UseInitialStridesAB"]:
          module.add(ValueSet("constStride%s%s"%(tc,idxChar), 1))
        else:
          if not kernel["ProblemType"]["UseInitialStridesAB"]:
            i = i-1
          module.add(RegSet("s", "sgprStride%s%s"%(tc,idxChar), \
                    "sgprStrides%s"%tc, i))

    module.addSpaceLine()
    module.add(ValueSet("MT0", kernel["MacroTile0"]))
    module.add(ValueSet("MT1", kernel["MacroTile1"]))
    module.add(ValueSet("DepthU", kernel["DepthU"]))
    module.add(ValueSet("GSU", kernel["GlobalSplitU"]))
    module.add(ValueSet("BpeA", tPA["bpe"]))
    module.add(ValueSet("BpeALog2", log2(tPA["bpe"])))
    module.add(ValueSet("BpeB", tPB["bpe"]))
    module.add(ValueSet("BpeBLog2", log2(tPB["bpe"])))
    module.addComment0("Number of elements to shift-left SRD")
    module.add(ValueSet("SrdShiftLeftA", self.states.srdShiftLeft['A']))
    module.add(ValueSet("SrdShiftLeftB", self.states.srdShiftLeft['B']))

    if kernel["BufferLoad"] or kernel["BufferStore"]:
      module.addComment0("2GB limit - set offsets to -1 to exceed this and clamp")
      module.add(ValueSet("BufferLimit", 0xffffffff, format=1))
      #TODO-64 : This is max 32-bit negative value, the tail loop
      # does incrementally step through the GRO and increment GRO
      # which are initialized with this value
      module.add(ValueSet("BufferOOB", 0x80000000, format=1))

      srdUpperValue = SrdUpperValue(self.states.version)
      module.addComment2("Bits 127:96 of SRD.\n" + srdUpperValue.desc())
      module.add(ValueSet("Srd127_96", srdUpperValue.getValue(), format=1))

    ########################################
    # Global Offsets
    ########################################
    # justOffset32 means we should only write the 32-bit offset
    # This is used in Buffer addressing modes.
    # Flat addressing modes expect the GLOBAL_OFFSET to initialize a full 64-bit address
    for (tc, indices, justOffset32, tP) in [ \
        ("C", list(range(0, kernel["ProblemType"]["NumIndicesC"])), kernel["BufferStore"], None), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"], kernel["BufferLoad"], tPA), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"], kernel["BufferLoad"], tPB) ]:

      # BufferStore does not use this macro so don't generate it:
      if tc == "C" and kernel["BufferStore"]:
        continue

      module.addComment1("Global Offset %s"%tc)
      numDim = len(indices)
      idxChars = []
      for i in indices:
        idxChars.append(self.states.indexChars[i])

      # macro declaration
      calcDims = [] # dimensions which are participating in the address calc (ignores other summation)
      mirrorSumDims = []
      macroArgs = []
      for i in range(0, numDim):
        if tc == 'C':
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesCD"]
          idxChar = self.states.indexChars[i]
        else:
          useInitialStrides = kernel["ProblemType"]["UseInitialStridesAB"]
          idxChar = self.states.indexChars[tP['ia'][i]]

        # tile index or unroll vgpr or summation
        # other summation (other than unroll) are included in the GLOBAL_OFFSET macro but not used in address calc
        if     tc in ('A','C') and indices[i] == kernel["ProblemType"]["Index0"] \
            or tc in ('B','C') and indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          macroArgs.append("vgprOffset%s:req" % idxChars[i])
          calcDims.append(i)
        elif indices[i] in kernel["ProblemType"]["IndicesSummation"]:
          # other summation index (not unroll)
          if tc in ('A', 'B') and indices[i] in kernel["ProblemType"]["MirrorDims%s" % tc]:
            mirrorSumDims.append(i)
          continue
        else:
          # other batch or free index
          if isPackedIndex(kernel, indices[i]):
            calcDims.append(i)
            macroArgs.append("vgprOffset%s:req" % idxChars[i])
          elif not justOffset32: # buffer/justOffset32 scalars are included in SRD not the offset, so skip here
            calcDims.append(i)
            macroArgs.append("sgprOffset%s:req" % idxChars[i])
      macro = Macro("GLOBAL_OFFSET_%s" % tc, "vgprAddr:req", *macroArgs, "vgprTmp:req")

      # Each index may be skipped, scaled by stride, or unscaled
      # If destLo is unset, no accumulation is necessary.

      # if the first index (i==0) is unscaled (UseInitialStrides),
      # it can be combined at the next update or moved at end
      # (if there is no next update)

      pendingOffset = None # offset pending for accumulation
      offsetIsVgpr = False # True if the source is VGPR ; False if SGPR
      destLo = None

      # true for first addr calc. In this case, we can directly write addr
      # rather than accumulating through a tmp
      writeDirectToAddr = 1

      # mirror other summation indices
      for i in mirrorSumDims:
        if writeDirectToAddr:
          dest = "v[\\vgprAddr+0]"
          needAdd = 0 # don't need add since writing address directly.
          writeDirectToAddr = 0
        else:
          dest = "v[\\vgprTmp+0]"
          needAdd = 1
        macro.add(VSubU32(dst=dest, \
                src0=sgpr("Size%s"%globalParameters["IndexChars"][indices[i]]), \
                src1=1, \
                comment="mirror %s%s 1"%(tc, globalParameters["IndexChars"][indices[i]])))
        macro.add(VMulLOU32(dst=dest, \
                src0=dest, \
                src1=self.strideRef(tc, indices[i]), \
                comment="mirror %s%s 2"%(tc, globalParameters["IndexChars"][indices[i]])))

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = "v[\\vgprAddr+0]"
          destHi = "v[\\vgprAddr+1]"

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          macro.add(VAddCOU32(dst=destLo, \
            dst1=VCC(), \
            src0=srcLo, \
            src1="v[\\vgprTmp+0]", \
            comment="accumulate %s lower"%idxChar))

      for i in calcDims:
        # should have eliminated these above
        idx = indices[i]
        isMirrorIdx = tc in ('A', 'B') and idx in kernel["ProblemType"]["MirrorDims%s" % tc]
        assert not (idx in kernel["ProblemType"]["IndicesSummation"] and idx != kernel["ProblemType"]["IndexUnroll"])

        if indices[i] == kernel["ProblemType"]["Index0"] \
            or indices[i] == kernel["ProblemType"]["Index1"] \
            or indices[i] == kernel["ProblemType"]["IndexUnroll"]:
          offsetIsVgpr = True
        # other c index sgpr (free or batch)
        elif indices[i] < kernel["ProblemType"]["NumIndicesC"]:
          if isPackedIndex(kernel, indices[i]):
            offsetIsVgpr = True
          else:
            offsetIsVgpr = False
        else:
          assert(0) # no other type allowed

        # offset is VGPR or SGPR string to use for the offset
        if offsetIsVgpr:
          offset = "v[\\vgprOffset%s]" % idxChars[i]
        else:
          offset = "s[\\sgprOffset%s]" % idxChars[i]

        # macro.addComment0("dim%s pendingOffset=%s offset=%s offsetIsVgpr=%s" \
        #    % (self.states.indexChars[indices[i]], pendingOffset, offset, offsetIsVgpr))

        needAdd = 0
        # should be indices[i]??
        if i==0 and not useInitialStrides:
          # slide into next address calc - can do addr = pendingOffset + nextAddrCalc
          pendingOffset = offset
          writeDirectToAddr = 0
        else:
          # tile index or unroll vgpr
          if offsetIsVgpr:
            if writeDirectToAddr:
              destLo = "v[\\vgprAddr+0]"
              destHi = "v[\\vgprAddr+1]"
              needAdd = 0 # don't need add since writing address directly.
              writeDirectToAddr = 0
            else:
              destLo = "v[\\vgprTmp+0]"
              destHi = "v[\\vgprTmp+1]"
              needAdd = 1
            if isMirrorIdx:
              macro.add(VSubI32(
                dst="v[\\vgprTmp+0]",
                src0=sgpr("Size%s"%globalParameters["IndexChars"][idx]), \
                src1=offset, \
                comment="mirror %s%s 1"%(tc, globalParameters["IndexChars"][indices[i]])))
              macro.add(VSubI32(\
                dst="v[\\vgprTmp+0]",
                src0="v[\\vgprTmp+0]", \
                src1=1, \
                comment="mirror %s%s 2"%(tc, globalParameters["IndexChars"][indices[i]])))
              offset = "v[\\vgprTmp+0]"

            # offset * stride
            macro.add(VMulLOU32(dst=destLo,
                src0=self.strideRef(tc, indices[i]), \
                src1=offset, \
                comment="mul d%u lower"%i))
            if not justOffset32:
              macro.add(VMulHIU32(dst=destHi,
                  src0=self.strideRef(tc, indices[i]), \
                  src1=offset, \
                  comment="mul d%u upper"%i))
          else: # offset is SGPR:
            assert not isMirrorIdx
            if not justOffset32:
              # buffer mode (aka justOffset32) does scalars into SRD not offset
              macro.add(VMovB32(dst="v[\\vgprTmp+2]", src="s[\\sgprOffset%s]"%idxChars[i], \
                  comment="sgprOffset -> vgprTmp+2"))
              # offset * stride
              macro.add(VMulLOU32(dst="v[\\vgprTmp+0]", \
                  src0=self.strideRef(tc, indices[i]), src1="v[\\vgprTmp+2]",  \
                  comment="other stride mul d%u lower"%i))
              macro.add(VMulHIU32(dst="v[\\vgprTmp+1]", \
                  src0=self.strideRef(tc, indices[i]), src1="v[\\vgprTmp+2]",  \
                  comment="mul d%u upper"%i))
              needAdd = 1

        if needAdd:
          writeDirectToAddr = 0 # safety net, once we write address can't directly overwrite it later
          destLo = "v[\\vgprAddr+0]"
          destHi = "v[\\vgprAddr+1]"
          # addr += offset * stride (lo) : accumulate just-computed address term into addr

          srcLo = pendingOffset if pendingOffset else destLo
          srcHi = 0 if pendingOffset else destHi
          macro.add(VAddCOU32(dst=destLo, dst1=VCC(), \
            src0=srcLo, src1="v[\\vgprTmp+0]", \
            comment="accumulate %s lower"%idxChar))

          # addr += offset * stride (hi)
          if not justOffset32:
            macro.add(VAddCCOU32(dst="v[\\vgprAddr+1]", dst1=VCC(), \
                src0="v[\\vgprTmp+1]", src1=srcHi, src2=VCC(), \
                comment="accumulate %s upper"%idxChar))
          pendingOffset = None

      # pendingOffset but never got a chance to apply it,
      # need to just add an explicit move or add:
      # this can happen for small-order tensors
      if pendingOffset != None:
        destLo = "v[\\vgprAddr+0]"
        if writeDirectToAddr:
          macro.add(VMovB32(dst=destLo, src=offset, comment="setup d0 lower"))
          if not justOffset32:
            macro.add(VMovB32(dst="v[\\vgprAddr+1]", src=hex(0), comment="d0 upper"))
        else:
          macro.add(VAddCOU32(dst=destLo, dst1=VCC(), \
            src0=destLo, src1=pendingOffset, \
            comment="accumulate final pendingOffset"))


      if tP != None and kernel["BufferLoad"] and self.states.srdShiftLeft[tc]:
        macro.add(VAddU32(dst="v[\\vgprAddr+0]", \
            src0=hex(self.states.srdShiftLeft[tc]), \
            src1="v[\\vgprAddr+0]", \
            comment="add prepad for pointer shift"))

      # addr *= bytes/element
      if justOffset32:
        macro.add(staticMultiply("v[\\vgprAddr+0]", "v[\\vgprAddr+0]", self.states.bpeAB, None, "offset *= bytes/element"))
      else:
        macro.add(VLShiftLeftB64(dst="v[\\vgprAddr+0:\\vgprAddr+1]", \
            shiftHex=hex(log2(self.states.bpeAB)), \
            src="v[\\vgprAddr+0:\\vgprAddr+1]", \
            comment="offset *= bytes/element"))
      module.add(macro)

    module.add(MacroVDynamicScalarDiv(kernel["WavefrontSize"]))

    module.setNoOpt(True)
    return module

  def checkResources(self, mkb: KernelBody):
    # register allocation
    totalVgprs = self.vgprPool.size()
    totalAgprs = self.agprPool.size()
    totalSgprs = self.sgprPool.size()

    mkb.setGprs(totalVgprs=totalVgprs, totalAgprs=totalAgprs, totalSgprs=totalSgprs)
    module = mkb.body

    if self.states.overflowedResources:
      if self.states.overflowedResources == 1:
        msg = "too many vgprs"
      elif self.states.overflowedResources == 2:
        msg = "too many sgprs"
      elif self.states.overflowedResources == 3:
        msg = "half store requires at least two elements per batch"
      elif self.states.overflowedResources == 4:
        msg = "Occupancy limit"
      elif self.states.overflowedResources == 5:
        msg = "reading and writing LDS at same time require 2 LDS buffer"
      elif self.states.overflowedResources == 6:
        msg = "SIA2 better with occupancy 2"
      else:
        msg = "unknown"

      if globalParameters["PrintSolutionRejectionReason"]:
        printWarning("%s overflowed resources.  errorCode=%d, msg=\"%s\", vgprs=%u, sgprs=%u" \
          % (self.states.kernelName, self.states.overflowedResources, msg, \
          self.vgprPool.size(), self.sgprPool.size()))
      module.add(SEndpgm(comment="overflowed resources"), 0)
      module.add(ValueIf(value=0), 1)

  ##############################################################################
  # code phrase for load batched address from array of buffer pointer
  ##############################################################################
  def loadBatchedAddress(self, kernel, Batch, tmpSgprResource: RegisterPoolResource):
    tmpSgpr = tmpSgprResource.idx
    laneSC = tmpSgprResource.size
    module = Module("loadBatchedAddress %s" % Batch)
    module.addSpaceLine()

    # handle Batch C/D
    if not kernel["_GlobalAccumulation"]:
      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(Batch), src1=0x8, comment="offset of global buffer address"))
          module.add(SLoadB64(dst=sgpr("AddressD", 2), base=sgpr("AddressD",2), soffset=sgpr(tmpSgpr), comment="load global buffer D address"))

      endCheckLabel = Label(self.labels.getName(f"label_skip_c_buffer_deref_{Batch}"), "")
      module.add(SBranchIfZero("Beta", kernel["ProblemType"]["ComputeDataType"], tmpSgpr, laneSC, endCheckLabel, \
                     kernel['WavefrontSize']))

      for idx in kernel["ProblemType"]["IndicesBatch"]:
        if not isPackedIndex(kernel,idx):
          module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(Batch), src1=0x8, comment="offset of global buffer address"))
          module.add(SLoadB64(dst=sgpr("AddressC", 2), base=sgpr("AddressC",2), soffset=sgpr(tmpSgpr), comment="load global buffer C address"))

      module.add(endCheckLabel)

    #handle Batch A/B
    endCheckLabel = Label(self.labels.getName(f"label_skip_ab_buffer_deref_{Batch}"), "")
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(1), comment="check summation size"))
    for i in range(0, self.states.numSgprSizesSum):
      module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr("SizesSum+%u"%(i)), src1=sgpr(tmpSgpr), comment="check summation size"))
    module.add(SCmpEQU32(src0=sgpr(tmpSgpr), src1=hex(0), comment="skip buffer deref is size of summation is 0"))
    module.add(SCBranchSCC1(labelName=endCheckLabel.getLabelName(), comment="skip buffer deref is size of summation is 0"))
    module.add(SBranchIfZero("Alpha", kernel["ProblemType"]["ComputeDataType"], tmpSgpr, laneSC, endCheckLabel, \
                                 kernel['WavefrontSize']))

    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(Batch), src1=0x8, comment="offset of global buffer address"))
    for idx in kernel["ProblemType"]["IndicesBatch"]:
      if not isPackedIndex(kernel,idx):
        module.add(SLoadB64(dst=sgpr("AddressA", 2), base=sgpr("AddressA",2), soffset=sgpr(tmpSgpr), comment="load global buffer A address"))
        module.add(SLoadB64(dst=sgpr("AddressB", 2), base=sgpr("AddressB",2), soffset=sgpr(tmpSgpr), comment="load global buffer B address"))

    module.add(endCheckLabel)

    return module

  ##############################################################################
  def defineAndResources(self, kernel, tPA, tPB):
    module = Module("allocateResources")
    unsetD = self.undefineSgpr("OffsetD")
    unsetC = self.undefineSgpr("OffsetC")
    unsetA = self.undefineSgpr("OffsetA")
    unsetB = self.undefineSgpr("OffsetB")
    self.defineVariableSgprs(kernel)
    module.add(self.macroAndSet(kernel, tPA, tPB))

    runActivation = True if ((kernel["ProblemType"]["ActivationType"] != 'none') and (kernel["GlobalSplitU"] == 1) \
        and kernel["ActivationFused"]) else False
    storeSgprLoad = 0
    if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
      self.states.numSgprAddressBias = 2 # 64-bit
      if runActivation:
        self.states.BiasType = max(1, kernel["ProblemType"]["DestDataType"].numRegisters())
      else:
        self.states.BiasType = 1
      storeSgprLoad += self.states.numSgprAddressBias + self.states.BiasType
    if runActivation:
      if kernel["ProblemType"]["ActivationType"] == 'all':
        self.states.numActivationTypeArgSize = 1
      storeSgprLoad += self.states.numActivationTypeArgSize + self.states.numactivationArgTotalSize
    self.states.numStoreSgprToLoad = storeSgprLoad

    module.addComment2("Allocate Resources")
    if kernel["StorePriorityOpt"]:
      module.add(SSetPrior(prior=3, comment="optimization store"))

    if self.do["PreLoop"]:
      if self.db["InitSgpr"] & 0x1:
        module.addComment1("Init SGPRs")
        for i in range(self.states.firstInitSgpr, self.sgprPool.size()):
          module.add(SMovB32(dst=sgpr(i), src=hex(self.consts.initSgprValue), comment="InitSgpr&0x1"))
        module.addSpaceLine()

      if self.db["InitVgpr"] & 0x1:
        module.addComment1("Init VGPRs")
        for i in range(1, self.states.totalVgprs):
          module.add(VMovB32(dst=vgpr(i), src=hex(self.consts.initVgprValue), comment="InitVgpr&0x1"))
        module.addSpaceLine()

      # set m0
      module.add(SMovB32(dst=mgpr(0), src=hex(kernel["LdsNumElements"] * self.states.bpeAB),
          comment="LDS clamp at %u bytes"%(kernel["LdsNumElements"] * self.states.bpeAB)))

      # set Serial id vgpr
      module.add(VMovB32(dst=vgpr("Serial"), src=vgpr(0), comment="thread serial id"))

      if self.states.kernel["WavefrontSize"] == 32:
        module.add(SMovB32(dst=VCC(setHi=True), src=0, comment="Ensure hi bits are zero"))

      ########################################
      # load kernel args
      moduleArgs = Module("load arguments")
      self.kernArgOffset = 0
      self.argLoader = ArgumentLoader()
      moduleArgs.addComment1("Load Kernel Args")
      if globalParameters["DebugKernel"]:
        moduleArgs.add(self.argLoader.loadKernArg("AddressDbg", "KernArgAddress", dword=2))

      self.argLoader.loadKernArg("Tensor2dSizeC", "KernArgAddress", dword=2, writeSgpr=False)

      load = self.states.numSgprToLoad
      sgprStart = self.sgprs["Tensor2dSizeA"]
      moduleArgs.addModuleAsFlatItems(self.argLoader.loadAllKernArg(sgprStart, "KernArgAddress", load))
      if kernel.enabledSetPrioSplitLDS:
        moduleArgs.add(SSetPrior(prior=1, comment="prioritize init code so as to issue load sooner"))
      moduleArgs.add(SWaitCnt(lgkmcnt=0, comment="wait for %u bytes of kern args" % self.argLoader.getOffset()))

      if not kernel["ProblemType"]["StridedBatched"]:
        with self.allocTmpSgpr(self.states.laneSGPRCount) as tmpSgpr:
          moduleArgs.add(self.loadBatchedAddress(kernel, "WorkGroup2", tmpSgpr))
        moduleArgs.add(SWaitCnt(lgkmcnt=0, comment="wait global buffer address ready"))

      if kernel["ProblemType"]["GroupedGemm"]:
        tmpSgprGemmIdxMiddle = 7
        tmpSgprGemmIdxLeft = 8
        tmpSgprGemmIdxRight = 9

        tmpSgpr0 = 10
        tmpSgpr1 = 11

        tmpSgprTargetPlus1 = 10
        tmpSgprWgMiddle = 12
        tmpSgprWgLeft = 13

        tmpSgprNumGemm = 14

        module.addComment1("Grouped Gemm: Load num of Gemms")
        module.add(self.argLoader.loadKernArg(tmpSgprNumGemm, "KernArgAddress", hex(0), dword=1))
        module.addComment1("Grouped Gemm: Load address of kernel arguments")
        module.add(self.argLoader.loadKernArg("KernArgAddress", "KernArgAddress", hex(4), dword=2))
        module.add(SWaitCnt(lgkmcnt=0))

        module.addComment1("Grouped Gemm: binary search gemmIdx by workgroup table")
        module.add(SMovB32(dst=sgpr(tmpSgprGemmIdxLeft), src=0))
        module.add(SMovB32(dst=sgpr(tmpSgprGemmIdxRight), src=sgpr(tmpSgprNumGemm)))
        module.add(SAddU32(dst=sgpr(tmpSgprTargetPlus1), src0=sgpr("WorkGroup0"), src1=hex(1)))
        label_findGemm = Label("FIND_GEMM", "")
        module.add(label_findGemm)
        module.add(SAddU32(dst=sgpr(tmpSgprGemmIdxMiddle), src0=sgpr(tmpSgprGemmIdxLeft), src1=sgpr(tmpSgprGemmIdxRight)))
        module.add(SLShiftRightB32(dst=sgpr(tmpSgprGemmIdxMiddle), src=sgpr(tmpSgprGemmIdxMiddle), shiftHex=log2(2)))
        module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr1), src=sgpr(tmpSgprGemmIdxMiddle), shiftHex=log2(4)))
        module.add(self.argLoader.loadKernArg(tmpSgprWgMiddle, "KernArgAddress", sgpr(tmpSgpr1), dword=1))
        module.add(SWaitCnt(lgkmcnt=0))
        module.add(SCmpLtI32(src0=sgpr(tmpSgprWgMiddle), src1=sgpr(tmpSgprTargetPlus1)))
        module.add(SCSelectB32(dst=sgpr(tmpSgprWgLeft),       src0=sgpr(tmpSgprWgMiddle),      src1=sgpr(tmpSgprWgLeft)))
        module.add(SCSelectB32(dst=sgpr(tmpSgprGemmIdxRight), src0=sgpr(tmpSgprGemmIdxRight),  src1=sgpr(tmpSgprGemmIdxMiddle)))
        module.add(SCSelectB32(dst=sgpr(tmpSgpr1),            src0=sgpr(tmpSgprGemmIdxMiddle), src1=sgpr(tmpSgprGemmIdxLeft)))
        module.add(SAddCU32(dst=sgpr(tmpSgprGemmIdxLeft), src0=sgpr(tmpSgpr1), src1=hex(0)))
        module.add(SCmpLtU32(src0=sgpr(tmpSgprGemmIdxLeft), src1=sgpr(tmpSgprGemmIdxRight)))
        module.add(SCBranchSCC1(labelName=label_findGemm.getLabelName()))
        module.add(SSubU32(dst=sgpr(tmpSgprGemmIdxLeft), src0=sgpr(tmpSgprGemmIdxLeft), src1=hex(1)))

        module.addComment1("Grouped Gemm: offset argument address to gemm")
        module.addComment0("Grouped Gemm: offset address from wg_table_start to args_start")
        module.add(SLShiftLeft2AddU32(dst=sgpr("KernArgAddress"), src0=sgpr(tmpSgprNumGemm), src1=sgpr("KernArgAddress")))
        module.add(SAddCU32(dst=sgpr("KernArgAddress+1"), src0=sgpr("KernArgAddress+1"), src1=hex(0)))
        module.addComment0("Grouped Gemm: offset address from args_start to gemm_start")
        module.add(SMulI32(dst=sgpr(tmpSgprGemmIdxLeft), src0=sgpr(tmpSgprGemmIdxLeft),\
                           src1=(self.argLoader.getOffset() + (storeSgprLoad * 4))))
        module.add(SAddU32(dst=sgpr("KernArgAddress"), src0=sgpr("KernArgAddress"), src1=sgpr(tmpSgprGemmIdxLeft)))
        module.add(SAddCU32(dst=sgpr("KernArgAddress+1"), src0=sgpr("KernArgAddress+1"), src1=hex(0)))

        module.add(moduleArgs)

        module.addComment1("Grouped Gemm: remap wg from 1D(numWG012) to 3D(wg2,wg1,wg0)")
        module.addComment0("numWG012 = hw_wg - accu_wg")
        module.add(SSubU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgprWgLeft)))
        module.addComment0("wg2 = numWG012 * smallMagicNumber(1/(numWG0*numWG1))")
        module.add(self.sMagicDivWrapper(dest=tmpSgpr0, dividend=sgpr("WorkGroup0"), \
                   magicNumber=sgpr("SmallMagicNumberDivWg01"), magicShift=31))
        module.add(SMovB32(dst=sgpr("WorkGroup2"), src=sgpr(tmpSgpr0)))
        module.addComment0("numWG01 = numWG012 - wg2 * numWG0 * numWG1")
        module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr("WorkGroup2"), src1=sgpr("NumWorkGroups0")))
        module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr("NumWorkGroups1")))
        module.add(SSubU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr0)))
        module.addComment0("wg1 = numWG01 * smallMagicNumber(1/numWG0)")
        module.add(self.sMagicDivWrapper(dest=tmpSgpr0, dividend=sgpr("WorkGroup0"), \
                   magicNumber=sgpr("SmallMagicNumberDivWg0"), magicShift=31))
        module.add(SMovB32(dst=sgpr("WorkGroup1"), src=sgpr(tmpSgpr0)))
        module.addComment0("wg0 = numWG01 - wg1 * numWG0")
        module.add(SMulI32(dst=sgpr(tmpSgpr0), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0")))
        module.add(SSubU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(tmpSgpr0)))

        module.addSpaceLine()
        module.add(self.undefineSgpr("SmallMagicNumberDivWg0"))
        module.add(self.undefineSgpr("SmallMagicNumberDivWg01"))
      else:
        module.add(moduleArgs)
    else:
      module.add(ValueIf(0))

    # add offset to buffer
    module.addComment1("add offset to buffer")
    def addOffset2Buffer(imod, mat, value):
      imod.add(SLShiftLeftB32(dst=sgpr("Offset%s"%mat), src=sgpr("Offset%s"%mat), shiftHex=hex(value), comment="elements offset to bytes offset"))
      imod.add(SAddU32(dst=sgpr("Address%s+0"%mat), src0=sgpr("Address%s+0"%mat), src1=sgpr("Offset%s"%mat), comment="add offset to buffer address"))
      imod.add(SAddCU32(dst=sgpr("Address%s+1"%mat), src0=sgpr("Address%s+1"%mat), src1=0, comment="add offset to buffer address"))

    if not kernel["_GlobalAccumulation"]:
      addOffset2Buffer(module, "D", log2(self.states.bpeCexternal))
      addOffset2Buffer(module, "C", log2(self.states.bpeCexternal))
    addOffset2Buffer(module, "A", log2(self.states.bpeAB))
    addOffset2Buffer(module, "B", log2(self.states.bpeAB))

    # self.states.groOffsetInMacroTile == 1 case, subtract pre-pad here
    if self.states.groOffsetInMacroTile:
      prePad = self.states.srdShiftLeft["A"] * tPA["bpe"] # leave room in case we have to pointer shift
      module.add(SSubU32(dst=sgpr("AddressA+0"), src0=sgpr("AddressA+0"), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
      module.add(SSubBU32(dst=sgpr("AddressA+1"), src0=sgpr("AddressA+1"), src1=0, comment="pre-pad to make room for possible pointer shift"))
      prePad = self.states.srdShiftLeft["B"] * tPB["bpe"] # leave room in case we have to pointer shift
      module.add(SSubU32(dst=sgpr("AddressB+0"), src0=sgpr("AddressB+0"), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
      module.add(SSubBU32(dst=sgpr("AddressB+1"), src0=sgpr("AddressB+1"), src1=0, comment="pre-pad to make room for possible pointer shift"))

    # undefine Offset sgpr
    module.addSpaceLine()
    module.add(unsetD)
    module.add(unsetC)
    module.add(unsetA)
    module.add(unsetB)

    # Check alpha == 0, is done before kernel body
    # so if alpha/beta=Half, they haven't been converted to f32
    # This means we can use ComputeDataType as AlphaType (even <h,h,h,h,"h,h"> +"HPA")
    if self.do["ApplyAlpha"]:
      module.addComment1("Short circuit condition if Alpha == 0, then sumDims=0")
      endCheckLabel = Label("AlphaNonZero", "")
      module.add(SBranchIfNotZero("Alpha", kernel["ProblemType"]["ComputeDataType"], endCheckLabel))

      # Conditional set summation dimensions to 0 on SCC==1
      for i in range(0, self.states.numSgprSizesSum):
        module.add(SMovB32(dst=sgpr("SizesSum+%u"%(i)), src=hex(0), comment="Set summation dim=0 if Alpha == 0"))

      # Jump here if alpha is non-zero
      module.add(endCheckLabel)

    if kernel["MagicDivAlg"]==2:
      for idxChar in sorted(set(kernel["PackedC0IdxChars"][:-1] + kernel["PackedC1IdxChars"][:-1])):
          module.add(SLShiftRightB32(dst=sgpr("MagicAbitSize%s"%idxChar), src=sgpr("MagicShiftSize%s"%idxChar), shiftHex=31, comment="extract abit"))
          module.add(SAndB32(dst=sgpr("MagicShiftSize%s"%idxChar), src0=sgpr("MagicShiftSize%s"%idxChar), src1=hex(0x7fffffff), comment="remove abit"))

    ########################################
    # Debug Buffer
    if globalParameters["DebugKernel"]:
      module.addComment1("Debug Buffer")

      # nwg0 FIXME use NumWorkGroups0
      nwg0 = self.vgprPool.checkOut(1)
      tmpVgpr = self.vgprPool.checkOutAligned(2, 2)
      tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
      module.addComment("nwg0 = (size%s + MT%s - 1) / MT%s;" \
          % (self.states.tileChar0, self.states.tileChar0, self.states.tileChar0))
      module.add(VMovB32(dst=vgpr(tmpVgpr), src=hex(kernel["MacroTile0"]-1), comment="MT0-1"))
      module.add(VAddCOU32(dst=vgpr(nwg0), dst1=VCC(), src0=sgpr("SizesFree+0"), \
          src1=vgpr(tmpVgpr), comment="%s = size0+MT0-1"%vgpr(nwg0)))
      module.add(vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgprRes))
      self.vgprPool.checkIn(tmpVgpr)
      self.nipt = 16 # num integers per thread
      v = self.vgprPool.checkOut(3)
      module.add(VMovB32(dst=vgpr(v), src=sgpr("WorkGroup0"), comment="%s=wg0"%vgpr(v) ))
      module.add(VMovB32(dst=vgpr(v+1), src=sgpr("WorkGroup1"), comment="%s=wg1"%vgpr(v+1) ))
      module.add(VMulLOU32(dst=vgpr(v+1), src0=vgpr(v+1), src1=vgpr(nwg0), \
          comment="%s=wg1*nwg0"%vgpr(v+1) ))
      module.add(VAddCOU32(dst=vgpr(v), dst1=VCC(), src0=vgpr(v), src1=vgpr(v+1), \
          comment="%s=wg1*nwg0+wg0"%vgpr(v) ))
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(staticMultiply(vgpr(v), vgpr(v), kernel["NumThreads"], tmpSgprInfo))
      module.add(VAddCOU32(dst=vgpr(v), dst1=VCC(), src0=vgpr(v), src1=vgpr("Serial"), \
          comment="%s=tid+NT*(wg1*nwg0+wg0)=serial"%vgpr(v) ))
      module.add(VMulLOU32(dst=vgpr(v), src0=hex(self.nipt*4), src1=vgpr(v), \
          comment="%s=serial*nipt*4"%vgpr(v) ))
      module.add(VMovB32(dst=vgpr(v+1), src=0))
      module.add(VAddCOU32(dst=vgpr("AddressDbg"), dst1=VCC(), src0=sgpr("AddressDbg"), src1=vgpr(v), \
          comment="%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") ))
      module.add(VMovB32(dst=vgpr(v+2), src=sgpr("AddressDbg+1"), comment="%s=AddressD1"%vgpr(v+2) ))
      module.add(VAddCCOU32(dst=vgpr("AddressDbg+1"), dst1=VCC(), src0=vgpr(v+2), \
          src1=vgpr(v+1), src2=VCC(), comment="%s=AddrD* + serial*nipt*4"%vgpr("AddressDbg") ))
      module.add(SMovB32(dst=sgpr("DebugKernelItems"), src=0))
      self.vgprPool.checkIn(v)
      self.vgprPool.checkIn(nwg0)

    if self.db["InitLds"]:
      tmp = RegisterPoolResource(idx=self.vgprPool.checkOut(2), size=2)
      numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
      module.add(DSInit(tmp, kernel["NumThreads"], kernel["LdsNumElements"], \
        numBytesPerElement, self.consts.initLdsValue))
      self.vgprPool.checkIn(tmp.idx)

    return module

  def extractPackedCoord1ToRowStart(self, kernel, packedC1, packedCoordVgpr, storeChar):
    # calculate packed rowStart vgpr
    # vgprTmp assignments:
    #   - tmp+0 is the incoming packed coordinate 1, used on replay too
    #   - tmp+1 is DIV output
    #   - tmp+2 is scratch
    #   - tmp+3 holds thread rowStart free1 offset
    module = Module("extractPackedCoord1ToRowStart")
    tmpV0 = self.vgprPool.checkOut(4)
    tmpV1 = tmpV0 + 1
    tmpV2 = tmpV0 + 2
    tmpV3 = tmpV0 + 3

    module.add(VMovB32(dst=vgpr(tmpV0), src=vgpr(packedCoordVgpr),  comment="copy coord1 then unpack"))
    for i,idx in enumerate(packedC1[:-1]):
      idxChar= globalParameters["IndexChars"][idx]
      module.addComment0("extract %s"%self.sizeRef(idx))
      module.add(MacroInstruction(name="V_MAGIC_DIV", \
                args=[tmpV1, vgpr(tmpV0), sgpr("MagicNumberSize%s"%idxChar), \
                sgpr("MagicShiftSize%s"%idxChar), (sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0")]))
      module.add(VMulLOU32(dst=vgpr(tmpV2), src0=vgpr(tmpV1), src1=self.sizeRef(idx), comment="remainder part 1"))
      module.add(VSubU32(dst=vgpr(tmpV2), src0=vgpr(tmpV0), src1=vgpr(tmpV2), comment="remainder part 2"))
      if i==0:
        module.add(VMulLOU32(dst=vgpr(tmpV3), src0=vgpr(tmpV2), \
                  src1=self.strideRef(storeChar, idx), comment="addrCalc <- scaled extracted dim"))
      else:
        module.add(VMulLOU32(dst=vgpr(tmpV2), src0=vgpr(tmpV2), \
                  src1=self.strideRef(storeChar, idx), comment="scale extracted dim"))
        module.add(VAddU32(dst=vgpr(tmpV3), src0=vgpr(tmpV3), \
                  src1=vgpr(tmpV2), comment="addrCalc += scaled extracted dim "))

      if i < len(packedC1)-2:
        module.add(VMovB32(dst=vgpr(tmpV0), src=vgpr(tmpV1), \
                  comment="Copy remaining bits for next divide"))

    module.addComment0("extract final %s"%self.sizeRef(packedC1[-1]))
    module.add(VMulLOU32(dst=vgpr(tmpV2), src0=vgpr(tmpV1), \
              src1=self.strideRef(storeChar, packedC1[-1]), comment="scale final extracted dim"))
    module.add(VAddU32(dst=vgpr(self.vgprs.coutRowPtr), src0=vgpr(tmpV3), \
              src1=vgpr(tmpV2), comment="rowStart += scaled extracted dim "))

    self.vgprPool.checkIn(tmpV0)
    return module

  ##############################################################################
  # Global Read Addresses: WorkGroup
  ##############################################################################
  def graWorkGroup(self, kernel):
    module = Module("graWorkGroup")
    module.addComment0("graWorkGroup mapping")
    if kernel["GlobalSplitU"] > 1:
      module.addComment("GSU-not-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;" \
          % (self.states.tileChar1, self.states.tileChar1, self.states.tileChar1))

      # gsuSumIdx = wg1 % GSU
      # wg1       = wg1 / GSU
      with self.allocTmpSgpr(3) as tmpSgprInfo:# needs 3
        tmpSgpr = tmpSgprInfo.idx
        divisor = tmpSgpr+2
        module.add(SMovB32(dst=sgpr(divisor), src=sgpr("WorkGroup1"), \
            comment="copying for divisor"))

        module.add(scalarStaticDivideAndRemainder("WorkGroup1", "GSUSumIdx", \
            divisor, kernel["GlobalSplitU"], tmpSgprInfo, 1))

    ########################################
    # Blocked rows or columns
    absWgm = abs(kernel["WorkGroupMapping"])
    if abs(kernel["WorkGroupMapping"]) > 1:
      smallNumMagicShift = 31
      magicNumberWgm = ((1<<smallNumMagicShift) // absWgm + 1)

      with self.allocTmpSgpr(4) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        blockId2  = tmpSgpr+0
        wgSerial2 = tmpSgpr+1
        wgmDivisor = tmpSgpr+2
        wgmDivisorMagicNumber = tmpSgpr+3

        module.add(SMovB32(dst=sgpr(wgmDivisorMagicNumber), src=hex(magicNumberWgm)+'L', \
            comment="magic number for WGM==%u"%absWgm))
        # blockId and serial within block

        # note this overwrites blockId2+1
        module.add(self.sMagicDivWrapper(dest=blockId2, dividend=sgpr("WorkGroup1"), \
            magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift))
        module.add(SMulI32(dst=sgpr(wgSerial2), src0=sgpr(blockId2), src1=absWgm, comment="quotient * non-magic divisor"))
        module.add(SSubU32(dst=sgpr(wgSerial2), src0=sgpr("WorkGroup1"), src1=sgpr(wgSerial2), comment="WorkGroup1=remainder"))
        module.add(SMulI32(dst=sgpr(wgSerial2), src0=sgpr(wgSerial2), src1=sgpr("NumWorkGroups0"), comment="(wg1 % WGM)*nwg0"))
        module.add(SAddU32(dst=sgpr(wgSerial2), src0=sgpr(wgSerial2), src1=sgpr("WorkGroup0"), comment="wgSerial = wg0 + (wg1 % WGM)*nwg0"))

        module.add(SCmpGeU32(src0=sgpr(blockId2), src1=sgpr("NumFullBlocks"), comment="blockId >= numFullBlocks ?"))
        # reuse wgmDivisorMagicNumber - may override with remainder here:
        module.add(SCMovB32(dst=sgpr(wgmDivisorMagicNumber), src=sgpr("MagicNumberWgmRemainder1")))
        module.add(SCSelectB32(dst=sgpr(wgmDivisor), src0=sgpr("WgmRemainder1"), src1=absWgm))

        # No longer supported for kernel["WorkGroupMapping"] < 0
        assert(self.sgprs["WorkGroup0"] & 0x1 == 0) # must be even and ...
        assert(self.sgprs["WorkGroup0"]+1 == self.sgprs["WorkGroup1"] ) # must be consecutive (for magic div below)
        module.add(self.sMagicDivWrapper(dest=self.sgprs["WorkGroup0"], dividend=sgpr(wgSerial2), \
            magicNumber=sgpr(wgmDivisorMagicNumber), magicShift=smallNumMagicShift))
        module.add(SMulI32(dst=sgpr("WorkGroup1"), src0=sgpr("WorkGroup0"), src1=sgpr(wgmDivisor), comment="quotient * non-magic divisor"))
        module.add(SSubU32(dst=sgpr("WorkGroup1"), src0=sgpr(wgSerial2), src1=sgpr("WorkGroup1"), comment="WorkGroup1=remainder"))
        module.add(SMulI32(dst=sgpr(blockId2), src0=sgpr(blockId2), src1=abs(kernel["WorkGroupMapping"]), comment="blockId * WGM"))
        module.add(SAddU32(dst=sgpr("WorkGroup1"), src0=sgpr("WorkGroup1"), src1=sgpr(blockId2), comment="wg1 += blockId * WGM"))

    return module

  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  # global read addresses: tile offset assignment (message from .s)
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    module = Module("graTileAssignment")
    tc = tP["tensorChar"]

    divisorName = tP["lvc"]
    divisor = kernel[divisorName]

    # force to swap gro-tile and gro-unroll for DirectToVgpr + TLU=False
    forceSwap = (kernel["DirectToVgpr%s"%tc] and not tP["tlu"])
    if tP["tlu"] or forceSwap:
      rReg = self.vgprPool.checkOut(1, "graTA rReg0", self.states.preventVgprOverflowDuringNewTile) # gro-tile = serial%divisor
      qReg = self.vgprPool.checkOut(1, "graTA qReg0", self.states.preventVgprOverflowDuringNewTile) # gro-unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprPool.checkOut(1, 'graTA qReg1', self.states.preventVgprOverflowDuringNewTile) # gro-tile = serial/divisor
      rReg = self.vgprPool.checkOut(1, 'graTA rReg1', self.states.preventVgprOverflowDuringNewTile) # gro-unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"

    module.addComment0("%s = %u" % (divisorName, kernel[divisorName]))
    if self.states.groOffsetInMacroTile:
      tReg2 = tReg
      # treg2 and treg same register and value - we store the 'static'
      # part of the address calculation in the SRD to maximize the
      # range of the 32-bit GRO
      module.addComment0("%s = (local)gro%s-tile = serial%s%s (note (wg%s*MT%s) will be added to SRD)" \
          % (vgpr(tReg2), tc, tOpStr, divisorName, tc, tc) )
    else:
      tReg2 = self.vgprPool.checkOut(1, 'treg2', self.states.preventVgprOverflowDuringNewTile)
      module.addComment0("%s = gro%s-tile = serial%s%s + (wg%s*MT%s)" \
          % (vgpr(tReg2), tc, tOpStr, divisorName, tc, tc) )

    module.addComment0("%s = gro%s-unroll = serial%s%s" \
        % (vgpr(uReg), tc, uOpStr, divisorName) )

    tmpVgpr = self.vgprPool.checkOutAligned(2, 2, 'graTA vgpr', self.states.preventVgprOverflowDuringNewTile)
    tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)

    dividendReg = "Serial" # local serial

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      dividendReg = self.vgprPool.checkOut(1, "idInWave", self.states.preventVgprOverflowDuringNewTile)
      dummy       = self.vgprPool.checkOut(1, "dummy", self.states.preventVgprOverflowDuringNewTile)
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(vectorStaticRemainder(dummy, dividendReg, "Serial", kernel["WavefrontSize"], tmpVgprRes, tmpSgprInfo))

    if kernel["DirectToVgpr%s"%tc]:
      # offset calculation for DirectToVgpr
      # ported code from local read for DirectToVgpr
      # alloc vgpr
      wReg       = self.vgprPool.checkOut(1,"wReg") # quotient
      # parameters
      tile01      = tP["tile01Idx"]
      waveWidth   = kernel["WavefrontSize"]
      num1DBlocks = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
      num1DWaves  = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
      vectorWidth = 1 # kernel["VectorWidth"] if ((tile01 == 0) and kernel["SourceSwap"]) else 1 # TODO: nonSwap VectorWidth
      strideTile  = 1 # tentative
      strideWave  = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth
      # tile offset
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(vectorStaticRemainder(wReg, qReg, dividendReg, waveWidth, tmpVgprRes, tmpSgprInfo))
        module.add(vectorStaticRemainder(wReg, rReg, qReg, kernel["MatrixInstN"], tmpVgprRes, tmpSgprInfo))
        # block offset (no code. assuming num1DBlocks == 1)
        # unroll offset (no code here. This will be handled in GlobalOffset)
        # wave offset
        if num1DWaves > 1:
          module.add(vectorStaticDivide(wReg, dividendReg, waveWidth, tmpVgprRes))
          module.add(vectorStaticRemainder(tmpVgpr, wReg, wReg, num1DWaves, tmpVgprRes, tmpSgprInfo))
          module.add(staticMultiply(vgpr(wReg), vgpr(wReg), strideWave, tmpSgprInfo))
          module.add(VAddU32(dst=vgpr(rReg), src0=vgpr(wReg), src1=vgpr(rReg)))
          # need division for qReg
          module.add(vectorStaticDivide(qReg, qReg, kernel["MatrixInstN"], tmpVgprRes))
          lrvwOther = self.states.lrvwB if tP["isA"] else self.states.lrvwA # The other side of lrvw
          if lrvwOther == 2 and not kernel["allowLRVWforTLUandMI"] and tP["tlu"]:
            # DirectToVgpr + LocalReadVectorWidth=2 case, multiply qReg by 2
            module.add(staticMultiply(vgpr(qReg), vgpr(qReg), lrvwOther, tmpSgprInfo))
      # release register
      self.vgprPool.checkIn(wReg)
    else:
      module.add(vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgprRes))


    if kernel["WaveSeparateGlobalRead%s"%tc]:
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr), src=vgpr("Serial"), comment="WaveIdxWavefrontWidth"))
        module.add(SLShiftRightB32(dst=sgpr(tmpSgpr), src=sgpr(tmpSgpr), shiftHex=hex(log2(kernel["WavefrontSize"])), comment="WaveId"))
        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=(kernel[tP["lsp"]] * tP["nrp"]), \
            comment="Global Read Wave: each wave loads continuous lsp(%u)*nrp(%u) columns" % (kernel[tP["lsp"]], tP["nrp"])))
        module.add(VAddU32(dst=vgpr(qReg), src0=sgpr(tmpSgpr), src1=vgpr(qReg), \
            comment="Global Read Wave: add back to column index"))
      self.vgprPool.checkIn(dividendReg)
      self.vgprPool.checkIn(dummy)

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      if tP["glvw"] > 1:
        if tP["tlu"]:
          module.addComment0("gro-tile *= glvw")
          module.add(staticMultiply(vgpr(tReg), vgpr(tReg), tP["glvw"], tmpSgprInfo))
        else:
          module.addComment0("gro-unroll *= glvw")
          module.add(staticMultiply(vgpr(uReg), vgpr(uReg), tP["glvw"], tmpSgprInfo))
      if forceSwap:
        # in this case, need to multiply vw to gro-tile
        module.addComment0("gro-tile *= vw")
        module.add(staticMultiply(vgpr(tReg), vgpr(tReg), kernel["VectorWidth"], tmpSgprInfo))

      if not self.states.groOffsetInMacroTile:
        # Buffer Load will set the SRD to start of the MacroTile
        # So don't add the static wg-related component here - save for later.
        module.add(staticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]], tmpSgprInfo))  # workgroup
        module.add(VAddCOU32(dst=vgpr(tReg2), dst1=VCC(), src0=vgpr(tmpVgpr), \
            src1=vgpr(tReg), comment="gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
            % (tc, tOpStr, divisorName, tc, tc) ))

    if kernel["GlobalSplitU"] > 1:
      uReg2 = self.vgprPool.checkOut(1, "uReg2", self.states.preventVgprOverflowDuringNewTile)
      module.add(VMovB32(dst=vgpr(uReg2), src=vgpr(uReg), comment="copy for GlobalSplitU"))
      tP["gpr"]["uReg2"] = uReg2
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["tReg"] = tReg2
    tP["gpr"]["uReg"] = uReg
    self.vgprPool.checkIn(tmpVgpr)

    return Module("graTileAssignment (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Unroll Assignment
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    module = Module("graUnrollAssignment")
    # note groOffsetInMacroTile rolls these into SRD so don't change here:
    if not self.states.groOffsetInMacroTile and kernel["GlobalSplitU"] > 1:
      gsuOffset = self.vgprPool.checkOut(1, "gsuOffset", self.states.preventVgprOverflowDuringNewTile)
      module.add(VMovB32(dst=vgpr(gsuOffset), src=sgpr("GSUSumIdx"), comment="=gsuSumIdx"))

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        # graUnrollAssignment += gsuSumIdx*DepthU
        module.add(staticMultiply(vgpr(gsuOffset), vgpr(gsuOffset), kernel["DepthU"], tmpSgprInfo))

      module.add(VAddCOU32(dst=vgpr(tP["gpr"]["uReg"]), dst1=VCC(), \
          src0=vgpr(gsuOffset), src1=vgpr(tP["gpr"]["uReg"]), \
          comment="graUnrollAssignment += gsuOffset"))
      self.vgprPool.checkIn(gsuOffset)
    else:
      module.addComment0(vgpr(tP["gpr"]["uReg"]))

    return Module("graUnrollAssignment (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self):
    module = Module("graOtherFreeAssignments")
    module.addComment0(sgpr("WorkGroup2"))
    return module

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    module = Module("graOtherSummationAssignments")
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      module.add(ValueSet(name="globalReadOffsetA%s" % self.states.indexChars[index], value=0))
      module.add(ValueSet(name="globalReadOffsetB%s" % self.states.indexChars[index], value=0))
    return module

  ##############################################################################
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    module = Module("graTileOffsets")
    tc = tP["tensorChar"]
    tP["vgprPackedOffsets"] = None
    tP["vgprTileOffsetsCheckOut"] = False
    tP["numVgprTileOffsets"] = 0
    if kernel["_UseSgprForGRO"]:
      # Let the vgprTileOffsets checkin handle tReg later since these are same vgpr
      tP["vgprTileOffsets"] = tP["gpr"]["tReg"]
    else:
      numTileOffsets = tP["nrt"]
      tP["vgprTileOffsets"] = self.vgprPool.checkOut(numTileOffsets, "vgprTileOffsets", self.states.preventVgprOverflowDuringNewTile)
      tP["vgprTileOffsetsCheckOut"] = True
      v = tP["vgprTileOffsets"]
      numExtraPackedOffsetsPerTile = len(tP["PackedIndices"])-1
      if numExtraPackedOffsetsPerTile:
        tP["vgprPackedOffsets"] = self.vgprPool.checkOut(numExtraPackedOffsetsPerTile * numTileOffsets, "vgprPackedOffsets", self.states.preventVgprOverflowDuringNewTile)
      strideIdx = tP["lsc"] if tP["tlu"] else tP["lsp"]
      stride = kernel[strideIdx]
      # adjustment for DirectToVgpr + tlu=False + VW > 1 case
      strideInterleave = False
      if kernel["DirectToVgpr%c"%tc] and (not tP["tlu"]) and kernel["VectorWidth"] > 1:
        strideInterleave = True
        stride = stride * kernel["VectorWidth"] - (kernel["VectorWidth"] - 1)


      module.add(VMovB32(dst=vgpr(v), src=vgpr(tP["gpr"]["tReg"]), comment="gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], 0) ))
      for l in range(1, tP["nrt"]):
        strideValue = stride
        if strideInterleave and (l & 1) != 0:
          strideValue = 1
        module.add(VAddCOU32(dst=vgpr(v+l), dst1=VCC(), src0=strideValue, \
            src1=vgpr(v+l-1), comment="gro%s%s_%u += %s"%(tP["tensorChar"], tP["tileChar"], l, strideIdx) ))
      if numExtraPackedOffsetsPerTile:
        tmpV = self.vgprPool.checkOutAligned(2,2,"packTmp", self.states.preventVgprOverflowDuringNewTile)

        for l in range(0, tP["nrt"]):
          lastGroVgpr = vgpr(v+l)
          lastGroIdx = tP["PackedIndices"][0]
          module.addSpaceLine()
          for p in range(0, numExtraPackedOffsetsPerTile):
            groIdx  = tP["PackedIndices"][p+1]
            groChar = globalParameters["IndexChars"][tP["PackedIndices"][p+1]]
            groVgpr = vgpr(tP["vgprPackedOffsets"] + l*numExtraPackedOffsetsPerTile + p)
            pChar = globalParameters["IndexChars"][tP["PackedIndices"][p]]
            module.add(MacroInstruction(name="V_MAGIC_DIV", \
                args=[tmpV, lastGroVgpr, sgpr("MagicNumberSize%s"%pChar), \
                sgpr("MagicShiftSize%s"%pChar), (sgpr("MagicAbitSize%s"%pChar) if kernel["MagicDivAlg"]==2 else "0")] ))
            module.add(VMovB32(dst=groVgpr, src=vgpr(tmpV), comment="extract gro%s%s_%u (%s)"%(tc,groChar,l,groVgpr)))
            module.add(VMulLOU32(dst=vgpr(tmpV), src0=groVgpr, src1=sgpr("SizesFree+%u"%lastGroIdx), comment="remainder part 1"))
            module.add(VSubU32(dst=lastGroVgpr, src0=lastGroVgpr, src1=vgpr(tmpV), \
                comment="remove extracted bits from gro%s%s_%u (%s)"%(tc, globalParameters["IndexChars"][lastGroIdx], l, lastGroVgpr)))
            lastGroVgpr = groVgpr
            lastGroIdx = groIdx
        self.vgprPool.checkIn(tmpV)

      # groOffsetInMacroTile uses same register for both of these, don't free it here:
      if tP["gpr"]["lwoT"] != tP["gpr"]["tReg"] :
        self.vgprPool.checkIn(tP["gpr"]["tReg"])
        tP["gpr"]["tReg"] = None
    return Module("graTileOffsets (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    module = Module("graUnrollOffsets")
    tc = tP["tensorChar"]
    if kernel["_UseSgprForGRO"]:
      tP["gpr"]["unrollOffsets"] = tP["gpr"]["uReg"]
    else:
      numUnrollOffsets = tP["nru"]
      tP["gpr"]["unrollOffsets"] = self.vgprPool.checkOut(numUnrollOffsets, "unrollOffsets", self.states.preventVgprOverflowDuringNewTile)
      v = tP["gpr"]["unrollOffsets"]
      strideIdx = (tP["lsp"] if tP["tlu"] else tP["lsc"])
      stride = kernel[strideIdx]
      prevStride = 0
      totalStride = 0
      lrvwOther = self.states.lrvwB if tP["isA"] else self.states.lrvwA # The other side of lrvw
      tluOther = kernel["ProblemType"]["TLUB"] if tP["isA"] else kernel["ProblemType"]["TLUA"] # The other side of tlu
      module.add(VMovB32(dst=vgpr(v), src=vgpr(tP["gpr"]["uReg"]), comment="gro%s%s_%u"%(tP["tensorChar"], self.states.unrollChar, 0)))
      for l in range(1, tP["nru"]):
        totalStride += stride
        if tP["tlu"] and kernel["DirectToVgpr%s"%tc] and lrvwOther == 2 and not tluOther:
          # DirectToVgpr + LocalReadVectorWidth=2 case, stride * 2 is added every 2. Add 1 in odd l case
          totalStride = stride * (l - (l % 2)) + (l % 2)
        currStride = totalStride - prevStride
        prevStride = totalStride
        module.add(VAddCOU32(dst=vgpr(v+l), dst1=VCC(), src0=currStride, \
            src1=vgpr(v+l-1), comment="gro%s%s_%u + %s"%(tP["tensorChar"], self.states.unrollChar, l, strideIdx)))
      #self.vgprPool.checkIn(tP["gpr"]["uReg"])
    return Module("graUnrollOffsets (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Shift A/B
  # See if the load (including vw) will extend past the 'free' dim of the
  # tensor.  If so clip to the last legal value which is inside the array
  ##############################################################################
  def graShift(self, kernel, tP):
    # graShift requires a vgpr for each address component (so each component
    # can be examined and shifted if necessary) - therefore does not work
    # with UseSgprForGRO.
    assert(not kernel["_UseSgprForGRO"])

    module = Module("graShift")
    #tc = tP["tensorChar"]
    # edge value
    margin = tP["glvw"] if tP["rtv"] else 1
    edge = self.vgprPool.checkOut(1, "edge", self.states.preventVgprOverflowDuringNewTile)

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      if self.states.groOffsetInMacroTile:
        # Subtract the static component from SizesFree:
        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tP["wg"]), src1=kernel[tP["mt"]], comment="WorkGroup[01] * MT"))
        module.add(SSubU32(dst=sgpr(tmpSgpr), src0=self.sizeRef(tP["idx"]), src1=sgpr(tmpSgpr), \
                  comment="edge = Size%s - WG*MT"%(tP["tileChar"])))
        # use math here to use unsigned (to increase range)
        #  - add srdShiftLeft to tmpSgpr - ensure it is always positive
        #  - below add srdShiftLeft to a tmp copy of the offset used for the compare
        # edge = (Size - WG*MT) - margin = the last valid load position that won't cause OOB
        # offset = the current load position for this thread
        # so if offset is larger than edge, we go back to the edge position
        module.add(SSubU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=margin, comment="edge -= margin(%u)"%(margin)))
        module.add(VMovB32(dst=vgpr(edge), src=sgpr(tmpSgpr), comment="edge vgpr = Size%s- WG*MT - margin(%u)"%(tP["tileChar"], margin) ))
        #shiftedEdge = self.vgprPool.checkOut(1, "shiftedEdge", self.states.preventVgprOverflowDuringNewTile)
        #module.add(VAddCOU32(dst=vgpr(shiftedEdge), dst1=VCC(), src0=vgpr(edge), src1=self.states.srdShiftLeft[tc],
        #             comment="shiftedEdge = edge + srdShiftLeft({})".format(self.states.srdShiftLeft[tc])))
      else:
        module.add(SSubU32(dst=sgpr(tmpSgpr), src0=self.sizeRef(tP["idx"]), src1=margin, \
            comment="edge = Size%s-%u"%(tP["tileChar"], margin) ))
        module.add(VMovB32(dst=vgpr(edge), src=sgpr(tmpSgpr), \
            comment="edge vgpr = Size%s-%u"%(tP["tileChar"], margin) ))

    # shift offsets
    vSrc = tP["vgprTileOffsets"]
    vDst = tP["vgprTileOffsets"]
    with self.allocTmpSgpr(self.states.laneSGPRCount) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      for l in range(0, tP["nrt"]):
        # compare
        cmpCommentText = "offset < edge"
        if self.states.groOffsetInMacroTile:
          module.add(VMinI32(dst=vgpr(vDst+l), src0=vgpr(edge), src1=vgpr(vSrc+l),
                      comment="offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge)))
        else:
          module.add(VCmpLtU32(dst=sgpr(tmpSgpr,self.states.laneSGPRCount), src0=vgpr(vSrc+l), src1=vgpr(edge),
                      comment="shiftedOffset < shiftedEdge"))
          # shift
          module.add(VCndMaskB32(dst=vgpr(vDst+l), src0=vgpr(edge), src1=vgpr(vSrc+l), src2=sgpr(tmpSgpr,self.states.laneSGPRCount),
                      comment="offset = (%s) ? offset(v%u) : edge(v%u)"%(cmpCommentText, vSrc+l, edge)))
    self.vgprPool.checkIn(edge)

    return module

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    module = Module("graFinalOffsets")
    tc = tP["tensorChar"]
    tmp = self.vgprPool.checkOut(3, "tmp", self.states.preventVgprOverflowDuringNewTile)
    graIdx = 0
    swapPerpPara = (((tc=="A" and kernel["DirectToVgprA"]) or (tc=="B" and kernel["DirectToVgprB"])) \
                    and (not tP["tlu"]) and tP["nrp"] > 1)

    if not swapPerpPara:
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
              # single loop
              singleModule, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              module.add(singleModule)
    else:
      # swap para and perp
      for para in range(0, tP["nrc"]):
        for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
          for perp in range(0, tP["nrp"]):
            for sPerp in range(0, tP["nrpv"]):
              # single loop
              singleStr, graIdx = self.graFinalOffsetsSingleLoop(kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara)
              module.add(singleModule)

    if tP["vgprTileOffsetsCheckOut"]:
      self.vgprPool.checkIn(tP["vgprTileOffsets"])
      tP["vgprTileOffsets"] = None
      tP["vgprTileOffsetsCheckOut"] = False
      # _UseSgprForGRO uses same vgpr for ureg and tP["gpr"]["unrollOffsets"] so
      # let checkin(ureg) do the checkin
      # vgprTileOffsets is renamed version of treg/lwo so checkin here

    if not kernel["_UseSgprForGRO"]:
      self.vgprPool.checkIn(tP["gpr"]["unrollOffsets"])
      tP["gpr"]["unrollOffsets"] = None

    if tP["vgprPackedOffsets"] != None:
      self.vgprPool.checkIn(tP["vgprPackedOffsets"])
      tP["vgprPackedOffsets"] = None

    self.vgprPool.checkIn(tmp)
    #if tP["isB"]:
    #  module.add(self.getBomb(0x100))

    return Module("Global Read Addresses: Final Offsets A/B (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B (single loop)
  ##############################################################################
  def graFinalOffsetsSingleLoop(self, kernel, tP, tc, tmp, graIdx, perp, sPerp, para, sPara):
    module = Module("graFinalOffsetsSingleLoop")
    problemType = kernel["ProblemType"]
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    # single loop start

    # vgpr assignments
    if tP["tlu"]:
      vgprTile   = tP["vgprTileOffsets"]   + para*tVW + sPara*tVS
      vgprUnroll = tP["gpr"]["unrollOffsets"] + perp*uVW + sPerp*uVS
    else:
      vgprTile   = tP["vgprTileOffsets"]   + perp*tVW + sPara*tVS
      vgprUnroll = tP["gpr"]["unrollOffsets"] + para*uVW + sPerp*uVS

    if graIdx==0 or not kernel["_UseSgprForGRO"]:
      # emit global offset macro
      # TODO -refactor this and macro def to pass all indices, use the ones we need
      if kernel["BufferLoad"]:
        bfName = "GLOBAL_OFFSET_%s" % tP["tensorChar"]
        bfArgs = ["vgprGlobalReadOffset%s+%u"%(tP["tensorChar"], graIdx)]
      else:
        bfName = "GLOBAL_OFFSET_%s" % tP["tensorChar"]
        bfArgs = ["vgprGlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx)]
      packedIter = 0 #iterator through ia
      iaToGpr = [None] * problemType["TotalIndices"]
      for i in tP["ia"]:
        if i < problemType["NumIndicesC"]:
          if i == tP["tileIdx"]:
            iaToGpr[i] = vgprTile
            bfArgs.append( "%2u" % iaToGpr[i] )
          else:
            if isPackedIndex(kernel,i):
              iaToGpr[i] = tP["vgprPackedOffsets"] + \
                            (vgprTile-tP["vgprTileOffsets"])*(len(tP["PackedIndices"])-1) + \
                            packedIter
              bfArgs.append( "%2u" % (iaToGpr[i]) )
              packedIter += 1
            else:
              # just a group index
              if not kernel["BufferLoad"]:  # buffer load adds these to SRD not the GLOBAL_OFFSET here
                bfArgs.append( "sgprWorkGroup%u"%i )
        else: # summation index
          if i == problemType["IndexUnroll"]:
            iaToGpr[i] = vgprUnroll
            bfArgs.append( "%2u" % iaToGpr[i] )
          # other summation indices are ignored

      bfArgs.append( "%u" % tmp )
      bfComment = "gRO%s_%u_%u_%u_%u" % (tP["tensorChar"], para, sPara, perp, sPerp)
      module.add(MacroInstruction(name=bfName, args=bfArgs, comment=bfComment))

      with self.allocTmpSgpr(2) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx

        # modify start
        if (not kernel["_UseSgprForGRO"]) and kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
          # add room for instruction offset
          groVgpr = "GlobalReadOffset%s+%u" % (tP["tensorChar"], graIdx)
          module.add(SMovB32(dst=sgpr(tmpSgpr), src=self.buff_load_inst_offset_max))
          module.add(VAddU32(dst=vgpr(groVgpr), src0=vgpr(groVgpr), src1=sgpr(tmpSgpr), comment="shift for UseInstOffsetForGRO"))

          ldsInc = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]
          if kernel["LdsBlockSizePerPad%s"%tc] != 0:
            ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
          else:
            padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
            ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]

          # buffer_load only support 12 bit instruction offset
          # we have to increase m0 if offset is larger thant 12 bits
          # so only keep 12 bit offset and subtract it on global address
          # global address will add back by buffer_load instruction offset
          ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
          if (ldsInc != 0):
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=ldsInc))
            module.add(VSubU32(dst=vgpr(groVgpr), src0=vgpr(groVgpr), src1=sgpr(tmpSgpr), comment="sub offset for buffer_load instoffset"))

    def computeScalarGroImpl(scalarGro):
      # this needs unroll stride in some cases and free stride in others
      # if we have multiple free strides - what is expected behavior?
      # could just extract the first free dimension from A?
      stride1 = "Stride%s%s"%(tc,self.states.indexChars[tP["idx"]])
      if tP["tlu"]:
        tileStride   = kernel[tP["lsc"]] * (para*tVW + sPara*tVS)
        unrollStride = kernel[tP["lsp"]] * (perp*uVW + sPerp*uVS)
        unrollSummation = [ i for i in tP["ia"] if i in problemType["IndicesSummation"] ]
        strideU = "Stride%s%s"%(tc,self.states.indexChars[unrollSummation[-1]])
        module.add(SMulI32(dst=sgpr(scalarGro), src0=sgpr(strideU), src1=unrollStride, \
                    comment="compute offset diff (scaled unrollDim)"))
        if tileStride:
          module.add(SAddU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=tileStride, \
                    comment="compute offset diff (tileDim)"))
      else:
        tileStride   = kernel[tP["lsp"]] * (perp*tVW + sPara*tVS)
        unrollStride = kernel[tP["lsc"]] * (para*uVW + sPerp*uVS)
        strideF = "Stride%s%s"%(tc,self.states.indexChars[tP['tileIdx']])
        module.add(SMulI32(dst=sgpr(scalarGro), src0=sgpr(strideF), src1=tileStride, \
                    comment="compute offset diff (scaled tileDim)"))
        if unrollStride:
          module.add(SAddU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=unrollStride, \
                    comment="compute offset diff (unrollDim)"))

      # Using offsets so GRO holds a byte offset not an element offset
      # So scale here before comparison:
      module.add(SLShiftLeftB32(
          dst=sgpr(scalarGro), \
          src=sgpr(scalarGro), \
          shiftHex=hex(log2(tP["bpe"])), \
          comment="scalar offset *= bytes/element"))

      if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
        # add room for instruction offset
        module.add(SAddU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=self.buff_load_inst_offset_max, comment="shift for UseInstOffsetForGRO"))

        ldsInc = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]
        if kernel["LdsBlockSizePerPad%s"%tc] != 0:
          ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
        else:
          padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
          ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]

        # buffer_load only support 12 bit instruction offset
        # we have to increase m0 if offset is larger thant 12 bits
        # so only keep 12 bit offset and subtract it on global address
        # global address will add back by buffer_load instruction offset
        ldsInc = (ldsInc * graIdx) % self.buff_load_inst_offset_max
        if (ldsInc != 0):
          module.add(SSubU32(dst=sgpr(scalarGro), src0=sgpr(scalarGro), src1=ldsInc, comment="sub offset for buffer_load instoffset"))

      if self.states.checkGRO:
        # Debug mode to verify that the computed offsets are offset by the expected scalar
        print(tc, "tileStride=", tileStride, "unrollStride=", unrollStride, \
              "stride=%s"%(stride1))

        module.add(self.getVectorDiffAssert(vgpr("GlobalReadOffset%s+%u"%(tc,0)), \
                                            vgpr("GlobalReadOffset%s+%u"%(tc,graIdx)), \
                                            sgpr(scalarGro)))

    needFirstSgprOffset = kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]

    if (kernel["_UseSgprForGRO"] or self.states.checkGRO) and (needFirstSgprOffset or graIdx > 0):
      # compute offsets for scalar global read offsets:
      if kernel["_UseSgprForGRO"]:
        tmpIdx = graIdx if needFirstSgprOffset else graIdx-1
        scalarGro = "ScalarGlobalReadOffset%s+%u"%(tc, tmpIdx)
        computeScalarGroImpl(scalarGro)
      else:
        # TODO: need for investagation for replacing by allocTmpSgpr
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          scalarGro = tmpSgprInfo.idx
          computeScalarGroImpl(scalarGro)

    # dump final offsets
    # BufferLoad flavor:
    #if tP["isA"]:
    #  module.add(self.dump(vgpr("GlobalReadOffset%s+%u+0"%(tP["tensorChar"], graIdx))))
    # Flat load flavor:
    #module.add(dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx))))
    #module.add(dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx))))
    graIdx += self.states.rpgo if kernel["BufferLoad"] else self.states.rpga

    return module, graIdx

  ##############################################################################
  # Add the constant offsets to the specified srd.
  # Srd is set to point to the base of the tile. All offsets except lowest-order
  # 2d dims are computed into the SRD.
  # GRO are offset from the tile SRD and the first GRO will be 0
  # Only called for BufferLoad=1 (or eventually BufferStore=1)
  ##############################################################################
  def computeLoadSrd(self, kernel, tP, tc, indices, bpe):
    module = Module("computeLoadSrd")

    with self.allocTmpSgpr(2 + 2) as tmpSgprInfo:
      stmp = tmpSgprInfo.idx
      tileStart = stmp+2
      wroteTileStart = False
      #---
      # Compute tileStart #elements from the 2D array start
      # Add tile (and unroll if GSU) component into SRD - SRD will point to beginning of the macro-tile:
      if self.states.groOffsetInMacroTile:
        # packed modes can't use this mode, and code here assumes 1 index.
        assert(len(kernel["PackedC0IndicesX"])==1)
        assert(len(kernel["PackedC1IndicesX"])==1)

        wroteTileStart = True
        #tP['ia'][1]

        # This is guaranteed to fit in 32-bit since the WG*MT is a number of elements in some unsigned direction:
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(tP["wg"]), kernel[tP["mt"]], "WorkGroup[01] * MT"))
        strideF = self.strideRef(tc, tP['tileIdx'])
        if not self.isConstUnitStride(strideF):
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart), sgpr(tileStart+1), sgpr(tileStart+0), \
                    strideF, "tlu=0, scaled tile-offset by stride"))

        if kernel["GlobalSplitU"] > 1:
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), kernel["DepthU"], sgpr("GSUSumIdx"), "gsuOffset = DepthU*bpe*GSUSumIdx"))
          unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]
          stride = self.strideRef(tc,unrollSummation[-1])
          if tP["tlu"] and not self.isConstUnitStride(stride):
            # non-transpose case, unroll is in perp dim and should be scaled by unroll Stride
            module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), sgpr(stmp+0), \
                      stride, "tlu=1, scaled unroll-offset by stride"))

          module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(stmp+0), comment="accum GsuOffset term to tilestart"))
          module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(stmp+1), comment="accum GsuOffset term to tilestart"))

      # Output : tileStart[0:1] have offset in elements from the 2D start of the tile.
      # if groOffsetInMacroTile=1, 2DStart + tileStart gives the the start of the macro-tile;
      # This is used to compute the limit.
      # Later we modify tileStart to include batch and higher-order dims and add this to SRD.

      #---
      # Compute BUFFER Limit:
      prePad = self.states.srdShiftLeft[tc] * tP["bpe"] # leave room in case we have to pointer shift

      if not wroteTileStart:
        module.add(SMovB64(dst=sgpr(tileStart, 2), src=0, comment="set default tileStart"))

      if self.states.use64bShadowLimit:
        limitTmp0 = "ShadowLimit%s+0"%tc
        limitTmp1 = "ShadowLimit%s+1"%tc
      else:
        limitTmp0 = stmp+0
        limitTmp1 = stmp+1

      module.add(SSubU32(dst=sgpr(limitTmp0), src0=sgpr("Tensor2dSize%s"%tc), src1=sgpr(tileStart+0), comment="sub tileStart"))
      module.add(SSubBU32(dst=sgpr(limitTmp1), src0=sgpr("Tensor2dSize%s+1"%tc), src1=sgpr(tileStart+1), comment="sub tileStart"))

      if self.states.use64bShadowLimit:
        # Set initial buffer limit
        # if the limit is >64bit, incrementSrd decrements the shadow as the SRD increments,
        # and when we get within 32-bit we start to step down the SRD
        # if the limit is <32bits, set it accurately here:
        # Note lshl_b64 the higher-numbered SGPR has the upper 32-bits
        module.add(SLShiftLeftB64(dst=sgpr("ShadowLimit%s"%tc,2),  src=sgpr("ShadowLimit%s"%tc,2), \
            shiftHex=hex(log2(tP["bpe"])), comment="Set limit to use bytes"))
        if prePad:
          module.add(SAddU32(dst=sgpr("ShadowLimit%s+0"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1=prePad, comment="extend limit for pre-pad"))
          module.add(SAddCU32(dst=sgpr("ShadowLimit%s+1"%tc), src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="extend limit for pre-pad"))

        if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
          module.add(SAddU32(dst=sgpr("ShadowLimit%s+0"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1=self.buff_load_inst_offset_max, comment="extend limit for directToLDS instruction offset"))
          module.add(SAddCU32(dst=sgpr("ShadowLimit%s+1"%tc), src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="extend limit for directToLDS instruction offset"))

        module.add(SCmpEQU32(src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="are we within 2^32?"))
        module.add(SCSelectB32(dst=sgpr("Srd%s+2"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1="BufferLimit", comment="Move shadow to real if we are within 2^32"))
      else:
        # put limit directly into SRD:
        module.add(SLShiftLeftB32(dst=sgpr("Srd%s+2"%tc), src=sgpr(stmp+0), shiftHex=hex(log2(tP["bpe"])), comment="Set limit to use bytes"))
        module.add(SAddU32(dst=sgpr("Srd%s+2"%tc), src0=sgpr("Srd%s+2"%tc), src1=prePad, comment="extend limit for pre-pad"))

      # Apply any high-order address components to the tileStart and eventually the SRD - batch idx for batched gemm
      if kernel["ProblemType"]["StridedBatched"]:
        numDim = len(indices)
        wg=2 # TODO - refactor since only WG2 is supported and this is always batch
        for i in range(1, numDim):
          idx = indices[i]
          if idx == kernel["ProblemType"]["Index0"] \
              or idx == kernel["ProblemType"]["Index1"] \
              or idx in kernel["ProblemType"]["IndicesSummation"] \
              or isPackedIndex(kernel, idx):
                continue # these will be captured in GRO not the SRD (or other summations are always 0)
          else:
            assert(wg==2) # can only have one wg2 with a batch. Other dimensions should be packed into wg0/wg1
            stride = "Stride%s%s"%(tc,self.states.indexChars[tP['ia'][i]])
            if not wroteTileStart:
              module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tileStart+0), sgpr(tileStart+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG"))
              wroteTileStart = True
            else:
              module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), sgpr(stride), sgpr("WorkGroup2"), "Stride*WG"))
              module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(stmp+0), comment="accum wg term to tilestart"))
              module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(stmp+1), comment="accum wg term to tilestart"))
            wg+=1

    # Add the tile start to the SRD
    if wroteTileStart:
      module.add(scalarStaticMultiply(sgpr(tileStart,2), sgpr(tileStart,2), bpe, None, "tileStart *= BPE"))
      module.add(SAddU32(dst=sgpr("Srd%s+0"%tc), src0=sgpr("Address%s+0"%tc), src1=sgpr(tileStart+0), comment="SRD base = Address+ tileStart0"))
      module.add(SAddCU32(dst=sgpr("Srd%s+1"%tc), src0=sgpr("Address%s+1"%tc), src1=sgpr(tileStart+1), comment="SRD base = Address+ tileStart1"))
    else:
      module.add(SMovB64(dst=sgpr("Srd%s"%tc, 2), src=sgpr("Address%s"%tc, 2), comment="init SRD base address"))

    # self.states.groOffsetInMacroTile == 1 case,  pre-pad is already subtracted from AddressA/B
    if prePad and self.states.groOffsetInMacroTile == 0:
      module.add(SSubU32(dst=sgpr("Srd%s+0"%tc), src0=sgpr("Srd%s+0"%tc), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
      module.add(SSubBU32(dst=sgpr("Srd%s+1"%tc), src0=sgpr("Srd%s+1"%tc), src1=0, comment="pre-pad to make room for possible pointer shift"))

    if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
      module.add(SSubU32(dst=sgpr("Srd%s+0"%tc), src0=sgpr("Srd%s+0"%tc), src1=self.buff_load_inst_offset_max, comment="make room for directToLDS instruction offset"))
      module.add(SSubBU32(dst=sgpr("Srd%s+1"%tc), src0=sgpr("Srd%s+1"%tc), src1=0, comment="make room for directToLDS instruction offset"))

    module.add(SMovB32(dst=sgpr("Srd%s+3"%tc), src="Srd127_96", comment="Set bits 127_96 in SRD"))

    #if tP["isB"]:
    #  module.add(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup1"), 0xA))

    return module

  ##############################################################################
  # Global Read Addresses: Addresses A/B
  ##############################################################################
  def graAddresses(self, kernel, tP):
    module = Module("graAddresses")
    tc = tP["tensorChar"]
    graIdx = 0

    if kernel["BufferLoad"]:
      # maxAddrSgpr = size[n] * stride[n-1]
      module.addComment0("max read offset = size[n] * stride[n-1]")

      module.add(self.computeLoadSrd(kernel, tP, tc, kernel["ProblemType"]["IndexAssignments%s"%tc], tP["bpe"]))

      #module.add(self.getBomb(0x13)) # after addresses and SRD set
    else:
      tmp = self.vgprPool.checkOut(2, "tmp", self.states.preventVgprOverflowDuringNewTile)
      module.add(VMovB32(dst=vgpr(tmp+0), src=sgpr("Address%s+0"%tP["tensorChar"])))
      module.add(VMovB32(dst=vgpr(tmp+1), src=sgpr("Address%s+1"%tP["tensorChar"])))
      for perp in range(0, tP["nrp"]):
        for sPerp in range(0, tP["nrpv"]):
          for para in range(0, tP["nrc"]):
            for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):

              comment = "gRA%s_%u_%u_%u_%u = addr%s+grO%s_%u_%u_%u_%u" \
                  % (tP["tensorChar"], para, sPara, perp, sPerp, \
                  tP["tensorChar"], tP["tensorChar"], \
                  para, sPara, perp, sPerp )
              module.add(VAddCOU32(
                  dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  dst1=VCC(), \
                  src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  src1=vgpr(tmp+0), \
                  comment=comment+" (lower)"))
              module.add(VAddCCOU32(
                  dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  dst1=VCC(), \
                  src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  src1=vgpr(tmp+1), \
                  src2=VCC(), \
                  comment=comment+" (upper)"))
              #module.add(dump(vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx))))
              #module.add(dump(vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx))))
              graIdx += self.states.rpga
      #module.add(SEndpgm())
      self.vgprPool.checkIn(tmp)

    return module

  ##############################################################################
  # Global Read Addresses: Increments
  # Define graIncrements, called once for each summation
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    module = Module("graIncrements")
    tc = tP["tensorChar"]

    dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
    loopChar = self.states.indexChars[dimIdx]

    stride = self.strideRef(tc, dimIdx)
    isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

    #print (tc, ": loopIdx=", loopIdx, "dimIdx=", dimIdx, "strideIdx=", strideIdx)

    gsu = 1
    if kernel["GlobalSplitU"] > 1:
      gsu = kernel["GlobalSplitU"]

    assert(self.states.unrollIdx == kernel["ProblemType"]["NumIndicesSummation"]-1)
    if loopIdx==self.states.unrollIdx:
      if self.states.globalReadIncsUseVgpr:
        with self.allocTmpSgpr(2) as tmpSgprInfo:
          tmpSgpr = tmpSgprInfo.idx
          module.add(SMulI32(dst=sgpr(tmpSgpr+0), \
              src0="DepthU*%d"%(gsu*tP["bpe"]), src1=stride, \
              comment="incr%s%s = %s*DepthU*bpe (unrollIdx)"%(tc, loopChar, stride) ))
          # TODO - this should be mul-H??
          module.add(SMovB32(
              dst=sgpr(tmpSgpr+1), \
              src=hex(0), \
              comment="(carry)"))
          module.add(VMovB32(
              dst=vgpr("GlobalReadIncs%s+%u+0"%(tc, 2*loopIdx)), \
              src=sgpr(tmpSgpr+0)))
          module.add(VMovB32(
              dst=vgpr("GlobalReadIncs%s+%u+1"%(tc, 2*loopIdx)), \
              src=sgpr(tmpSgpr+1)))
      else: # not globalReadIncsUseVgpr, ie use SGPR

        m = "DepthU*Bpe%s"%(tc)
        if gsu>1:
          m += "*%d"%gsu

        if isMirrorIdx:
          m = "-%s"%(m)

        # multiply by stride, optimizing if unit stride
        if self.isConstUnitStride(stride):
          module.add(SMovB32(dst=sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), src=m, \
              comment="incr%s (unrollIdx)"%(tc) ))
        else:
          module.add(SMulI32(dst=sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), \
              src0=m, src1=stride, \
              comment="incr%s unrollIdx)"%(tc) ))
    else:
      # other summation
      if self.states.globalReadIncsUseVgpr:
        printExit("NumIndicesSummation=%u not yet supported in assembly unless globalReadIncsUseVgpr==0" \
            % kernel["ProblemType"]["NumIndicesSummation"] )
      else:
        graInc = "GlobalReadIncs%s+%u"%(tc, loopIdx)
        # subtract increments done by the inner iterations
        # may be negative:
        loopIdxPrev = loopIdx + 1
        dimIdxPrev    = kernel["ProblemType"]["IndicesSummation"][loopIdxPrev] # dimension index
        loopCharPrev  = self.states.indexChars[dimIdxPrev]
        stridePrev = self.strideRef(tc, dimIdxPrev)
        isMirrorIdxPrev = dimIdxPrev in kernel["ProblemType"]["MirrorDims%s"%tc]

        module.addComment1("compute globalReadInc for higher-level loop")

        with self.allocTmpSgpr(3) as tmpSgprInfo:
          tmpSgpr = tmpSgprInfo.idx
          # Summations always appear in both A and B, can compute number of iterations just once:
          if loopIdxPrev==self.states.unrollIdx:
            loopCounterName= self.loopCounterName(kernel, self.states.unrollIdx)
            if tP["isA"]:
              quotient = loopCounterName
              dividend = "SizesSum+%u"%self.states.unrollIdx
              divisor = kernel["DepthU"]
              if kernel["NoTailLoop"] and kernel["AssertSummationElementMultiple"] % kernel["DepthU"] != 0:
                # round up SizesSum/DepthU for noTailLoop case
                module.add(SAddI32(dst=sgpr(quotient), src0=(divisor - 1), src1=sgpr(dividend), \
                    comment="round up SizeSum / DepthU" ))
                module.add(scalarStaticDivideAndRemainder(quotient, None, quotient, \
                            divisor, tmpSgprInfo, 0))
              else:
                module.add(scalarStaticDivideAndRemainder(quotient, None, dividend, \
                            divisor, tmpSgprInfo, 0))

              if kernel["GlobalSplitU"] > 1:
                module.add(self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgprInfo))

              module.add(SMulI32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), \
                        src1=kernel["GlobalSplitU"]*kernel["DepthU"], \
                        comment="=loopCounterName*DepthU"))
            module.add(SMulI32(dst=sgpr(graInc), src0=stridePrev, src1=sgpr(loopCounterName), \
                  comment="tmp <- stride%s%s * myWgUnrollIters" %(tc, loopCharPrev)))
          else:
            module.add(SMulI32(dst=sgpr(graInc), src0=stridePrev, src1=self.sizeRef(dimIdxPrev), \
                  comment="tmp <- stride%s%s * size%s%s" %(tc, loopCharPrev, tc, loopCharPrev)))

        # subtract amount that previous inner loop will have already incremented:
        # graInc is used as temp for the prev loop calc
        if isMirrorIdx and isMirrorIdxPrev:
          module.add(SSubI32(dst=sgpr(graInc), \
              src0=sgpr(graInc), \
              src1=stride, \
              comment="incr%s%s = <prev-incs> - stride%s%s"%(tc, loopChar, tc, loopChar) ))
        elif isMirrorIdx:
          module.add(SAddI32(dst=sgpr(graInc), \
              src0=stride, \
              src1=sgpr(graInc), \
              comment="incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) ))
          module.add(SSubI32(dst=sgpr(graInc), \
              src0=0, \
              src1=sgpr(graInc), \
              comment="incr%s%s = - (stride%s%s + <prev-incs>)"%(tc, loopChar, tc, loopChar) ))
        elif isMirrorIdxPrev:
          module.add(SAddI32(dst=sgpr(graInc), \
              src0=stride, \
              src1=sgpr(graInc), \
              comment="incr%s%s = stride%s%s + <prev-incs>"%(tc, loopChar, tc, loopChar) ))
        else:
          module.add(SSubI32(dst=sgpr(graInc), \
              src0=stride, \
              src1=sgpr(graInc), \
              comment="incr%s%s = stride%s%s - <prev-incs>"%(tc, loopChar, tc, loopChar) ))

        module.add(SLShiftLeftB32(
            dst=sgpr(graInc), \
            src=sgpr(graInc), \
            shiftHex="Bpe%sLog2"%tc,
            comment="<- scale by bpe"))

        if 0 and tP["isB"] and loopIdx==0:
          module.add(self.getBomb())
          #module.add(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup1"),0))

    #module.add(dump(vgpr("GlobalReadIncs%s"%tP["tensorChar"])))
    #module.add(SEndpgm())
    #if tP["isB"]:
    #  module.add(self.getBomb(0x100))
    return Module("graIncrements (Empty)") if self.dontAppendCode else module

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B
  ##############################################################################
  def lwaTileAssignment(self, tP):
    module = Module("lwaTileAssignment")
    module.addComment0("lwaTileAssignment%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["lwoT"])))
    return module

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    module = Module("lwaUnrollAssignment")
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    module.addComment0("lwaUnrollAssignment%s = %s" % (tP["tensorChar"], vgpr(uReg)))
    if kernel.enabledSplitLDS and kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      if self.states.inTailLoop:
        subIterReg = self.vgprPool.checkOut(1, "subIterReg")
        module.addComment0("Each wg writes 1/%u of G2L data to LDS"%kernel["DepthULdsDivisor"])
        module.add(VLShiftRightB32(dst=vgpr(subIterReg), shiftHex=log2(kernel["_DepthULds"]), src=vgpr(uReg), comment="sub_G2L_idx = uIdx / DepthU_Compute"))
        module.add(VAndB32(dst=vgpr(uReg), src0=vgpr(uReg), src1=(kernel["_DepthULds"]-1), comment="unrollIdx = unrollIdx % DepthU_Compute"))
        tP["gpr"]["subIterReg"] = subIterReg
      else:
        module.addComment0("Each thd writes 1/%u of G2L data to LDS"%kernel["DepthULdsDivisor"])
        module.add(VLShiftRightB32(dst=vgpr(uReg), shiftHex=log2(kernel["DepthULdsDivisor"]), src=vgpr(uReg), comment="sub_G2L_idx = uIdx / DepthULdsDivisor"))
    return module

  ##############################################################################
  # Local Write Addresses: First Offset A/B
  # uDu: which part of G2L buffer to write to LDS
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP, uDu=0):
    module = Module("lwaFirstOffset")
    tc = tP["tensorChar"]
    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.states.unrollChar, tP["tileChar"])
    uReg = tP["gpr"]["uReg2" if kernel["GlobalSplitU"] > 1 else "uReg"]
    if kernel["LocalWriteUseSgpr%s"%tc]:
      destVgpr = self.vgprPool.checkOut(1, "destVgpr", self.states.preventVgprOverflowDuringNewTile)
    else:
      destVgpr = "LocalWriteAddr%s"%tc

    if kernel["UnrollMajorLDS%s" % tc]:
      lds_stride = kernel["_DepthULds"] + LdsPad
      module.add(VMulU32U24(dst=vgpr(destVgpr), src0=hex(lds_stride), src1=vgpr(tP["gpr"]["lwoT"]), \
          comment="lw%s%s**(DepthU_Compute + PAD)"%(tP["tensorChar"], self.states.unrollChar)))
      module.add(VAddLShiftLeftU32(dst=vgpr(destVgpr), src0=vgpr(uReg), src1=vgpr(destVgpr), shiftHex=hex(log2(tP["bpe"])), \
          comment="lwFO%s = (lw%s%s + lw%s%s*(DepthU+PAD))*bpe" % (tc, tc, tc, tc, self.states.unrollChar) ))
    else:
      lds_stride = kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad
      module.add(VMulU32U24(dst=vgpr(destVgpr), src0=hex(lds_stride), src1=vgpr(uReg), \
          comment="lw%s%s**(MT%s + PAD)"%(tP["tensorChar"], self.states.unrollChar, tP["tensorChar"])))
      module.add(VAddLShiftLeftU32(dst=vgpr(destVgpr), src0=vgpr(tP["gpr"]["lwoT"]), src1=vgpr(destVgpr), shiftHex=hex(log2(tP["bpe"])), \
          comment="lwFO%s = (lw%s%s + lw%s%s*(MT%s+PAD))*bpe" % (tc, tc, tc, tc, self.states.unrollChar, tP["tileChar"]) ))

    # LdsBlockSizePerPad: add padding
    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      tmpVgpr = self.vgprPool.checkOut(2)
      tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
      module.add(vectorStaticDivide(uReg, destVgpr, kernel["LdsBlockSizePerPad%s"%tc], tmpVgprRes, \
        "padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(staticMultiply(vgpr(uReg), vgpr(uReg), kernel["LdsPad%s"%tc] * tP["bpe"], tmpSgprInfo, \
          "padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      module.add(VAddU32(dst=vgpr(destVgpr), src0=vgpr(uReg), src1=vgpr(destVgpr), \
        comment="add padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      self.vgprPool.checkIn(tmpVgpr)

    if tP["isB"]:
      if kernel["LdsOffsetB"] != 0: # LdsOffsetB can be 0 if DirectToVgprA is enabled
        module.add(VAddCOU32(
            dst=vgpr(destVgpr), \
            dst1=VCC(), \
            src0=hex(kernel["LdsOffsetB"]*tP["bpe"]), \
            src1=vgpr(destVgpr), \
            comment="lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
            self.states.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.states.bpeAB) ))

    self.vgprPool.checkIn(tP["gpr"]["lwoT"])
    tP["gpr"]["lwoT"] = None
    if kernel["GlobalSplitU"] > 1:
      self.vgprPool.checkIn(tP["gpr"]["uReg2"])
      tP["gpr"]["uReg2"] = None
    #LSC_ * LSP_
    numBytesPerElement = kernel["ProblemType"]["DataType"].numBytes()
    validWIPerLoad     = kernel[tP["lsc"]] * kernel[tP["lsp"]]//tP["glvw"]
    validBytesPerLoad  = kernel[tP["lsc"]] * kernel[tP["lsp"]] * numBytesPerElement
    maxBytesPerLoad    = kernel["NumThreads"] * tP["glvw"] * numBytesPerElement

    if kernel["WaveSeparateGlobalRead%s"%tc]:
      validBytesPerLoad *= (kernel["NumThreads"] // self.states.kernel["WavefrontSize"])

    assert (validBytesPerLoad <= maxBytesPerLoad)
    assert (kernel[tP["lsc"]] * kernel[tP["lsp"]] % tP["glvw"] == 0)

    if validBytesPerLoad != maxBytesPerLoad:
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.add(SMovB32(dst=sgpr(tmpSgpr), src=validWIPerLoad, \
            comment="lsc*lsp=%u*%u"%(kernel[tP["lsc"]],kernel[tP["lsp"]] )))
        module.add(VCmpLtU32(
            dst=VCC(), \
            src0=vgpr("Serial"), \
            src1=sgpr(tmpSgpr), \
            comment="fractional: ensure tid < global read tile elements"))
        tmpVgpr = self.vgprPool.checkOut(1, "tmpVgpr", self.states.preventVgprOverflowDuringNewTile)
        module.add(VMovB32(dst=vgpr(tmpVgpr), src=hex(self.consts.ldsOOB)))
        module.add(VCndMaskB32(
                    dst=vgpr(destVgpr), \
                    src0=vgpr(tmpVgpr), \
                    src1=vgpr(destVgpr), \
                    comment="Mask load so out-of-gr-tile bounds returns 0"))
        self.vgprPool.checkIn(tmpVgpr)

    elif self.states.inTailLoop and kernel.enabledSplitLDS: # where (DepthU for global read) != (DepthU for compute)
      # only for TN tensor + TN lds layout
      assert tP["tlu"] == 0
      module.add(VCmpEQU32(dst=VCC(), src0=vgpr(tP["gpr"]["subIterReg"]), src1=uDu, comment="if sub_g2l_idx == %u ?"%uDu))

      ldsOOB = self.vgprPool.checkOut(1, "lds OOB addr", self.states.preventVgprOverflowDuringNewTile)
      module.add(VMovB32(dst=vgpr(ldsOOB), src=hex(self.consts.ldsOOB), comment="lds OOB address"))
      module.add(VCndMaskB32( \
                  dst=vgpr(destVgpr), \
                  src0=vgpr(ldsOOB), \
                  src1=vgpr(destVgpr), \
                  comment="Mask threads not belonging to current sub_g2l_idx by assigning OOB"))
      self.vgprPool.checkIn(ldsOOB)

    if kernel["LocalWriteUseSgpr%s"%tc]:
      # TODO: Can refactor code above to Compute this directly:
      module.add(VReadfirstlaneB32(
          dst=sgpr("LocalWriteAddr%s"%tc), \
          src=vgpr(destVgpr), \
          comment="Copy lds write address VGPR to SGPR"))
      self.vgprPool.checkIn(destVgpr)

    self.vgprPool.checkIn(tP["gpr"]["uReg"])
    tP["gpr"]["uReg"] = None
    if "subIterReg" in tP["gpr"]:
      if tP["gpr"]["subIterReg"] is not None:
        self.vgprPool.checkIn(tP["gpr"]["subIterReg"])
      tP["gpr"]["subIterReg"] = None
    # dump lds write offsets
    #if tP["isA"]:
      #module.add(self.dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"])))
      #module.add(self.getBomb(-40))
    # do not generate local write address code if DirectToVgpr is enabled
    return Module("lwaUnrollAssignment (Empty)") if self.dontAppendCode or kernel["DirectToVgpr%s"%tc] else module

  ##############################################################################
  # Local Read Addresses: Tile Assignment
  ##############################################################################
  def lraTileAssignment(self, kernel, tPA, tPB):
    module = Module("lraTileAssignment")

    component = Component.LraTileAssignment.find(self)

    tP0 = tPA if tPB["tile01Idx"] else tPB
    tP1 = tPB if tPB["tile01Idx"] else tPA

    if component:
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP0["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        module.add(component(self, kernel, tP0))
      # do not generate local read code if DirectToVgpr is enabled
      tc = tP1["tensorChar"]
      if not kernel["DirectToVgpr%s"%tc]:
        module.add(component(self, kernel, tP1))

    return module

  ##############################################################################
  # Local Read Addresses: Final Offset A/B
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    module = Module("lraFinalOffset")

    # do not generate local read code if DirectToVgpr is enabled
    tc = tP["tensorChar"]
    if kernel["DirectToVgpr%s"%tc]:
      return module

    # allocate resources
    sgid    = self.vgprPool.checkOut(1) # quotient
    rReg    = self.vgprPool.checkOut(1) # remainder, unused here
    tmpVgpr = self.vgprPool.checkOutAligned(2, 2,"tmpVgpr")
    tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)

    # constant
    tc          = tP["tensorChar"]
    tile01      = tP["tile01Idx"]
    LdsPad      = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
    divisor     = kernel["SubGroup0"] * kernel["SubGroup1"]
    mtAddPad    = kernel["MacroTile%u" % tile01] + LdsPad

    # generate instruction
    module.add(vectorStaticDivide(sgid, "Serial", divisor, tmpVgprRes, \
      "LSU offset: sgid = Serial / subGroup(%u)" % divisor))
    with self.allocTmpSgpr(1) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      module.add(SMovB32(dst=sgpr(tmpSgpr), src=mtAddPad, \
        comment="LSU offset: stride = MT%u(%u) + PAD%u(%u)" % (tile01, kernel["MacroTile%u" % tile01], tile01, LdsPad)))
      module.add(VMulLOU32(dst=vgpr(sgid), src0=sgpr(tmpSgpr), src1=vgpr(sgid), \
        comment="LSU offset: lsuoffset = sgid*(MT%u+PAD)"%tile01))
      if not kernel["EnableMatrixInstruction"] and kernel["VectorWidth"] > 1:
        module.add(staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), kernel["VectorWidth"], tmpSgprInfo, \
        "Final Offset: lr%sOffset * VW" % tc))

    # final offset
    finalVgpr = vgpr("LocalReadAddr%s"%tc)
    if (kernel["DirectToLds%s" % tc] and \
        kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"] > 4):
      # DirectToLds + DGEMM case
      # use bpr for LSU offset instead of bpe (DirectToLds needs _ds_load_b32)
      module.add(VLShiftLeftB32(dst=vgpr(sgid), shiftHex=hex(log2(self.states.bpr)), src=vgpr(sgid),  \
              comment="LSU offset: lsuoffset = lsuoffset * bpr"))
      module.add(VLShiftLeftB32(dst=vgpr(tP["gpr"]["lro"]), shiftHex=hex(log2(tP["bpe"])), src=vgpr(tP["gpr"]["lro"]),  \
              comment="Final Offset: offset = (lro%s*VW)*bpe+lsuoffset*bpr" % tile01))
      module.add(VAddU32(dst=finalVgpr, src0=vgpr(sgid), src1=vgpr(tP["gpr"]["lro"])))
      # need magic offset calc here (after final offset)
      # offset calculation for TLU=1 when glvw * bpe * wavefrontsize > 256
      # x2/x4 directToLds stores 8/16 bytes into LDS like below
      # address offset in LDS in bytes
      # DWORD# written by LDS_DMA
      #  address offset in LDS (byte offset)
      #  0    4    8    12    16   20   24   28   32   36   40   44    48    52   56   60
      #  data dword#:
      #  0    4    8    12    2    6    10   14    1   5    9    13     3    7    11   15
      #  Noffset calculation for VW =1 (BPe=8) / VW =2 (BPE=4)
      #  use direcToLds for best VW and GRVW case; other cases requires bit more lane manipulation.
      #  offset calculation  for B might benefit from some optimization.
      #  offset calculation for x2/x4  is basically manipulation lane offset based on layout
      tmp1    = self.vgprPool.checkOut(1,"tmp1")
      tmp2    = self.vgprPool.checkOut(1,"tmp2")
      if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 8):
        # (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
        module.add(VAndB32(dst=vgpr(tmp1), src0=hex(0x4), src1=finalVgpr, comment="magic offset calc"))
        module.add(VLShiftLeftB32(dst=vgpr(tmp1), shiftHex=hex(3), src=vgpr(tmp1)))
        module.add(VAndB32(dst=vgpr(tmp2), src0=hex(0x38), src1=finalVgpr))
        module.add(VLShiftRightB32(dst=vgpr(tmp2), shiftHex=hex(1), src=vgpr(tmp2)))
        module.add(VOrB32(dst=vgpr(tmp1), src0=vgpr(tmp1), src1=vgpr(tmp2)))
        module.add(VAndB32(dst=finalVgpr, src0=hex(0xffffffc3), src1=finalVgpr))
        module.add(VOrB32(dst=finalVgpr, src0=finalVgpr, src1=vgpr(tmp1)))
      else:  #if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 16):  # most preferred case
        # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
        module.add(VAndB32(dst=vgpr(tmp1), src0=hex(0x4), src1=finalVgpr, comment="magic offset calc"))
        module.add(VLShiftLeftB32(dst=vgpr(tmp1), shiftHex=hex(3), src=vgpr(tmp1)))
        module.add(VAndB32(dst=vgpr(tmp2), src0=hex(0x8), src1=finalVgpr))
        module.add(VLShiftLeftB32(dst=vgpr(tmp2), shiftHex=hex(1), src=vgpr(tmp2)))
        module.add(VOrB32(dst=vgpr(tmp1), src0=vgpr(tmp1), src1=vgpr(tmp2)))
        module.add(VAndB32(dst=vgpr(tmp2), src0=hex(0x30), src1=finalVgpr))
        module.add(VLShiftRightB32(dst=vgpr(tmp2), shiftHex=hex(2), src=vgpr(tmp2)))
        module.add(VOrB32(dst=vgpr(tmp1), src0=vgpr(tmp1), src1=vgpr(tmp2)))
        module.add(VAndB32(dst=finalVgpr, src0=hex(0xffffffc3), src1=finalVgpr))
        module.add(VOrB32(dst=finalVgpr, src0=finalVgpr, src1=vgpr(tmp1)))
      # TODO: cover other cases

      # another address conversion for DirectToLds + NumLoadsCoalesced > 1
      newModule, dummy = self.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val=0, generateAsm=True, \
                                                              finalVgpr=finalVgpr, tmp1=tmp1, tmp2=tmp2)
      module.add(newModule)

      self.vgprPool.checkIn(tmp1)
      self.vgprPool.checkIn(tmp2)
    else:
      module.add(VAddLShiftLeftU32(dst=finalVgpr, src0=vgpr(sgid), src1=vgpr(tP["gpr"]["lro"]), shiftHex=hex(log2(tP["bpe"])), \
        comment="Final Offset: offset = (lro%s*VW+lsuoffset)*bpe" % tile01 ))

    # LdsBlockSizePerPad: add padding
    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] !=0:
      module.add(vectorStaticDivide(rReg, "LocalReadAddr%s"%tc, kernel["LdsBlockSizePerPad%s"%tc], tmpVgprRes, \
        "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.add(staticMultiply(vgpr(rReg), vgpr(rReg), kernel["LdsPad%s"%tc] * tP["bpe"], tmpSgprInfo, \
          "Final Offset: padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))
      module.add(VAddU32(dst=vgpr("LocalReadAddr%s"%tc), src0=vgpr(rReg), src1=vgpr("LocalReadAddr%s"%tc), \
        comment="Final Offset: add padding %u per block %u" % (kernel["LdsPad%s"%tc], kernel["LdsBlockSizePerPad%s"%tc])))

    # release resources
    self.vgprPool.checkIn(tmpVgpr)
    self.vgprPool.checkIn(sgid)
    self.vgprPool.checkIn(rReg)
    self.vgprPool.checkIn(tP["gpr"]["lro"])

    return module

  ##############################################################################
  # Local Read Addresses offset conversion for DTL + NLC > 1
  ##############################################################################
  def lraOffsetConversionForDTLandNLC(self, kernel, tP, offset_val, generateAsm=False, \
                                      finalVgpr=None, tmp1=None, tmp2=None):
    module = Module("lraOffsetConversionForDTLandNLC")
    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
    divisorName = tP["lvc"]
    divisor = kernel[divisorName]
    width = kernel["WavefrontSize"] if tP["tlu"] else kernel["DepthU"]
    if divisor < width:
      # DirectToLds + above conditions, rotate offset_val bits to adjust LDS offset
      lowerScale = tP["nrc"]
      upperScale = (kernel["WavefrontSize"] // divisor)
      # bit rotation necessary only when nrc > 1
      if lowerScale > 1:
        tile01 = tP["tile01Idx"]
        rightShift = int(log2(lowerScale)) # assuming power of 2
        leftShift = int(log2(upperScale)) # assuming power of 2
        line = kernel["MacroTile%u" % tile01] if tP["tlu"] else kernel["DepthU"]
        ldsLineSize = line * tP["bpe"] // lowerScale
        maskBitsLow = (lowerScale - 1) * ldsLineSize
        maskBitsHigh = (upperScale - 1) * lowerScale * ldsLineSize
        maskBitsAll = (maskBitsLow | maskBitsHigh)

        # offset_val conversion
        low = offset_val & maskBitsLow
        high = offset_val & maskBitsHigh
        low <<= leftShift
        high >>= rightShift
        val = low | high
        offset_val = (offset_val & (~maskBitsAll)) | val

        # generate asm code
        if generateAsm:
          with self.allocTmpSgpr(1) as tmpSgprInfo:
            tmpSgpr2 = tmpSgprInfo.idx
            module.add(VAndB32(dst=vgpr(tmp1), src0=hex(maskBitsLow), src1=finalVgpr, \
              comment="Offset rotation for DirectToLds + %s > 1"%tP["lsc"]))
            module.add(VAndB32(dst=vgpr(tmp2), src0=hex(maskBitsHigh), src1=finalVgpr))
            module.add(VLShiftLeftB32(dst=vgpr(tmp1), shiftHex=hex(leftShift), src=vgpr(tmp1)))
            module.add(VLShiftRightB32(dst=vgpr(tmp2), shiftHex=hex(rightShift), src=vgpr(tmp2)))
            module.add(VOrB32(dst=vgpr(tmp1), src0=vgpr(tmp1), src1=vgpr(tmp2)))
            module.add(SMovB32(dst=sgpr(tmpSgpr2), src=hex(maskBitsAll)))
            module.add(VNotB32(dst=vgpr(tmp2), src=sgpr(tmpSgpr2)))
            module.add(VAndB32(dst=finalVgpr, src0=vgpr(tmp2), src1=finalVgpr))
            module.add(VOrB32(dst=finalVgpr, src0=vgpr(tmp1), src1=finalVgpr))

    return module, offset_val

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    module = Module("lraDeclareAddresses")
    if tP["isA"]:
      module.addComment0("N/A")

    else:
      # no local read code if DirectToVgpr is enabled
      # no need to generate add code if LdsOffset is 0
      if kernel["DirectToVgprB"] or kernel["LdsOffset%s"%tP["tensorChar"]] == 0:
        module = Module("lraDeclareAddresses (Empty)")
      else:
        module.add(VAddCOU32(
            dst=vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
            dst1=VCC(), \
            src0=hex(kernel["LdsOffset%s"%tP["tensorChar"]]*tP["bpe"]), \
            src1=vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
            comment=" += LdsOffset%s (lower)"%tP["tensorChar"]))
    return module

  ##############################################################################
  # openShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def openShadowInit(self):
    module = Module("openShadowInit")
    module.add(Label("ShadowInitStart", ""))
    return module

  ##############################################################################
  # closeShadowInit
  # Label after prefetches are launched.  This is present even if ShadowInit not
  # used.
  ##############################################################################
  def closeShadowInit(self, kernel):
    module = Module("closeShadowInit")
    assert(self.states.doShadowInit and kernel["PrefetchGlobalRead"])

    module.add(self.checkLastIter(kernel))
    if kernel["SuppressNoLoadLoop"]:
      loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx]]
      lastIterEnd = Label("LoopEnd%s"%loopChar, "")
    else:
      lastIterEnd = Label("PrefetchGlobalLastIterEnd", "")

    # This branch could potentially be very far e.g. > SIMM16
    module.addComment1("after InitC, skip to end of prefetch last iter if numIter==0")
    # use positive offset only long jump
    module.add(self.longBranchScc1(lastIterEnd, posNeg=1))

    return module

  ##############################################################################
  # Initialize C
  ##############################################################################
  def initC(self, kernel):
    module = Module("initC")
    module.addComment1("initC: remove C-tile %u-%u from pool"%(self.states.c.startVgprValu, self.states.c.startVgprValu+self.states.c.numVgprValu))
    self.vgprPool.remove(self.states.c.startVgprValu, self.states.c.numVgprValu, "ValuC")
    numAccvgprs = self.states.totalAgprs
    self.agprPool.remove(0, numAccvgprs, "ValuC")
    module.addComment1("initC: remove AB-tile %u-%u from pool"%(self.states.a.startVgprValu , self.states.lastValuAB))
    self.vgprPool.remove(self.states.a.startVgprValu , self.states.lastValuAB - self.states.a.startVgprValu , "ValuAB")
    numCVgpr = max(self.states.c.numVgprValu, numAccvgprs)

    startNumCVgpr = 0
    if self.states.useInitAccVgprOpt:
      # init accvgpr opt. initialize only the last set of accvgpr instead of whole accvgpr
      numRegistersOut  = kernel["MIRegPerOut"]
      accs_per_wave    = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] \
                         // self.states.kernel["WavefrontSize"] * numRegistersOut
      startNumCVgpr = numCVgpr - accs_per_wave

    if kernel["LdsInitCVgprs"]:
      tmpAddr = self.vgprPool.checkOut(1,"tmp vgpr for lds init C registers")
      module.add(VMovB32(dst=vgpr(tmpAddr), src=self.consts.ldsOOB, comment="set out-of-bound addr"))

    for i in range(startNumCVgpr, numCVgpr):
      copyInst = VMovB32 if self.states.c.numVgprValu else VAccvgprWrite
      regStr = vgpr("ValuC+%u"%i) if self.states.c.numVgprValu else accvgpr(i)
      if not kernel["LdsInitCVgprs"]:
        module.add(copyInst(dst=regStr, src=hex(0), comment="initC"))
      else:
        module.add(DSLoadB32(dst=regStr, src=vgpr(tmpAddr), ds=DSModifiers(offset=0), comment="initC"))

    if kernel["LdsInitCVgprs"]:
      self.vgprPool.checkIn(tmpAddr)

    return module

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  # Output: Sets sgpr(StaggerRowMask)
  ##############################################################################
  def declareStaggerParms(self, kernel):
    module = Module("declareStaggerParms")
    if self.states.staggerU:
      # this could be dynamic?
      if kernel["StaggerUMapping"] == 0:
        staggerInput = sgpr("WorkGroup0")
      elif kernel["StaggerUMapping"] == 1:
        staggerInput = sgpr("WorkGroup1")
      elif kernel["StaggerUMapping"] == 2:
        staggerInput = sgpr("WorkGroup2")
      elif kernel["StaggerUMapping"] == 3:
        with self.allocTmpSgpr(2) as tmpSgprInfo:
          tmpSgpr = tmpSgprInfo.idx
          # wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0
          wgSerial = tmpSgpr
          tmp = tmpSgpr+1
          module.add(SMulI32(dst=sgpr(wgSerial), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), \
            comment="wgSerial = (nwg0*ngw1)*wg2 + (nwg0)*wg1 + wg0"))
          module.add(SMulI32(dst=sgpr(wgSerial), src0=sgpr(wgSerial), src1=sgpr("WorkGroup2")))
          module.add(SMulI32(dst=sgpr(tmp), src0=sgpr("NumWorkGroups0"), src1=sgpr("WorkGroup1")))
          module.add(SAddU32(dst=sgpr(wgSerial), src0=sgpr(wgSerial), src1=sgpr(tmp)))
          module.add(SAddU32(dst=sgpr(wgSerial), src0=sgpr(wgSerial), src1=sgpr("WorkGroup0")))
          staggerInput = sgpr(wgSerial)
      elif kernel["StaggerUMapping"] == 4:
        staggerInput = -1

      module.add(SAndB32(dst=sgpr("StaggerUIter"), src0=sgpr("OrigStaggerUIter"), \
                src1=staggerInput, \
                comment="Compute actual stagger start for this tile"))
      module.add(SLShiftLeftB32(dst=sgpr("StaggerUIter"), src=sgpr("StaggerUIter"), \
                shiftHex=kernel["_staggerStrideShift"], comment="shift by StaggerUStride"))
    return module

  ##############################################################################
  # Calculate and apply stagger offsets and edge
  ##############################################################################
  def calculateStagger(self, kernel, tP):
    imod = Module("calculateStagger")
    tc = tP["tensorChar"]

    if self.states.staggerU:
      assert (kernel["BufferLoad"])

      with self.allocTmpSgpr(2) as tmpSgprInfo:
        staggerTmp = tmpSgprInfo.idx

        #---
        imod.addComment1("SRDs += (StaggerUIter) * GlobalReadIncs%s+%u"% (tc, self.states.unrollIdx))

        # Calculate the stagger byte offset
        imod.addModuleAsFlatItems(self.s_mul_i64_i32(
                  sgpr(staggerTmp), sgpr(staggerTmp+1), \
                  sgpr("StaggerUIter"), sgpr("GlobalReadIncs%s+%u"%(tc, self.states.unrollIdx)), \
                  " stagger byte offset"))

        # Amount of bytes to add to get back to start.
        # on the llop iteration which matches StaggerUIter, this offset added instead of GlobalReadInc
        imod.addModuleAsFlatItems(self.s_mul_i64_i32(sgpr("WrapU%s+0"%tc), sgpr("WrapU%s+1"%tc), \
                  self.loopCounter(kernel, self.states.unrollIdx), sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                  "Number of bytes accessed by the unroll loop"))

        imod.add(SSubU32(dst=sgpr("WrapU%s+0"%tc),  \
                  src0=sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                  src1=sgpr("WrapU%s+0"%tc), \
                  comment="remove one iteration"))
        imod.add(SSubBU32(dst=sgpr("WrapU%s+1"%tc), \
                  src0=0, \
                  src1=sgpr("WrapU%s+1"%tc), \
                  comment="remove one iteration"))

        imod.add(self.incrementSrd(tP, sgpr(staggerTmp), sgpr(staggerTmp+1)))

      if tP["isB"]:
        # Convert passed in S' to S for easy loop comparison.  S=S-(PGR-1)'
        imod.add(SAddU32(dst=sgpr("StaggerUIter"), src0=sgpr("StaggerUIter"), \
                src1=(2 if kernel["PrefetchGlobalRead"] else 1), \
                comment="Subtract (PGR-1); StaggerUIter now contains target iteration to wrap"))
    return imod

  ##############################################################################
  # Remove stagger offset (before tail loop)
  # |          |           |   |
  # |-- S'*I --|
  # |---------- W' --------|-I-|
  #           ^ current SRD pos
  # ^unrollLoopStart           ^tailLoopStart   (in summation0 dimension)

  #
  # S = sgprStaggerUIter = S+(PGR+1)'
  # W = sgprWrapU
  # PGR = kernel["PrefetchGlobalRead"]
  #
  # S' = StaggUIter that is passed into the kernel = -PGR+1+S
  # S'*I is also the global read offset (from unrollLoopStart) at unroll loop exit ?
  # I = GlobalReadIncs
  # W' = W

  # Need to move it to tailLoopStart

  # To compute position where tail loop should start:
  #  = W' - S'*I + I
  #  = W - (S+PGR+1)*I) + I
  #  = W - (S+PGR+1)*I + I
  #  = W - (S+2+PGR)*I
  ##############################################################################
  def removeStagger(self, kernel, tP):
    imod = Module("removeStagger")
    if self.states.staggerU:
      tc = tP["tensorChar"]
      with self.allocTmpSgpr(2) as tmpSgprInfo:
        tmp = tmpSgprInfo.idx
        # might be able to refactor this to eliminate signed math
        imod.add(SSubI32(dst=sgpr(tmp), src0=3 if kernel["PrefetchGlobalRead"] else 2, \
                src1=sgpr("StaggerUIter")))
        imod.addModuleAsFlatItems(self.s_mul_i64_i32(sgpr(tmp), sgpr(tmp+1), \
                    sgpr(tmp), sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                    "start offset S in bytes"))
        imod.add(SSubU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr("WrapU%s"%tc), comment="S - WrapU"))
        imod.add(SSubBU32(dst=sgpr(tmp+1), src0=sgpr(tmp+1), src1=sgpr("WrapU%s+1"%(tc)), comment="S - WrapU"))

        imod.add(self.incrementSrd(tP, sgpr(tmp), sgpr(tmp+1)))

    return imod

  ##############################################################################
  # Emit code to compute loop iterations for GSU.
  # See same function in KernelWriterSource.py for background explanation
  # This function is used to compute number of loop iters and also
  # for computing the global read increment for GSU case.
  # For multiple summation, the number of loop iterations needs to be reset
  # for each iteration so replicate the code in addr inc and at open of unroll loop

  # tmpSgpr is allocation of at least 3 tmpSgpr

  # Output: SGPR(destName) contains the number of unroll iterations for
  # this workgroup.
  ##############################################################################
  def calculateLoopNumIterGsu(self, kernel, destName, tmpSgprRes: RegisterPoolResource):
    module = Module("calculateLoopNumIterGsu")

    loopCounter = sgpr(destName)
    quotient = destName
    remainder = "GSUSumIdx+1" # numIterPerWgRemainder
    dividend = tmpSgprRes.idx + 2 # numIterMyWg
    divisor = kernel["GlobalSplitU"]
    if log(divisor,2).is_integer():
      module.add(SMovB32(dst=sgpr(dividend), src=loopCounter, comment="copy for divide IterGsu"))
      module.add(scalarStaticDivideAndRemainder(quotient, remainder, dividend, divisor, tmpSgprRes , 1))
    else:
      qReg = self.vgprPool.checkOut(1,"qReg")
      rReg = self.vgprPool.checkOut(1,"rReg")
      dReg = self.vgprPool.checkOut(1,"dReg")
      tmpVgpr = self.vgprPool.checkOutAligned(2,2,"tmpReg")
      tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
      module.add(VMovB32(dst=vgpr(dReg), src=loopCounter, comment="copy for divide IterGsu"))
      module.add(vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgprRes))
      module.add(VReadfirstlaneB32(dst=sgpr(quotient), src=vgpr(qReg)))
      module.add(VReadfirstlaneB32(dst=sgpr(remainder), src=vgpr(rReg)))
      self.vgprPool.checkIn(tmpVgpr)
      self.vgprPool.checkIn(dReg)
      self.vgprPool.checkIn(rReg)
      self.vgprPool.checkIn(qReg)

    # if gsuSumIdx < numIterPerWgRemainder
    module.add(SAddU32(dst=sgpr(tmpSgprRes.idx), src0=1, \
                  src1=loopCounter, comment="tmp<-numIterMyWg+" ))
    module.add(SCmpLtU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), \
        comment="gsuSumIdx < numIterPerWgRemainder" ))
    module.add(SCMovB32(dst=loopCounter, src=sgpr(tmpSgprRes.idx), comment="numIterMyWg++ if needed"))

    return module

  ##############################################################################
  # Calculate Loop Num Iter
  # loopIdx is the index of the loop (used for contractions with multiple summations)
  # 0 is outermost; self.states.unrollIdx is the unroll index.
  # -1 is tail loop (used only for the unroll loop)
  ##############################################################################
  def calculateLoopNumIter(self, kernel, tPA, tPB, loopIdx):
    module = Module("calculateLoopNumIter")

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.states.unrollIdx
    loopDim = kernel["ProblemType"]["IndicesSummation"][loopIdx]
    loopChar = self.states.indexChars[loopDim]

    ########################################
    # Tail Loop
    if tailLoop:
      with self.allocTmpSgpr(4) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        loopCounterName = self.loopCounterName(kernel, loopIdx)
        module.addSpaceLine()
        if kernel["SuppressNoLoadLoop"]:
          # If the tail loop is suppressed, then final iterations will have moved the Srd base forward
          # (and also moved back the srd shadow limit) and slammed Limit to 0, so need to 'undo'
          # those increments - see setTailSrd
          assert(kernel["PrefetchGlobalRead"] == 1) #if >1 would need a multiply here
          module.add(SCmpEQU32(src0=sgpr("OrigLoopCounter"), src1=0, comment="completely skipped unroll loop?"))
          module.add(SCSelectB32(dst=sgpr(tmpSgpr+0), src0=0, src1=sgpr("GlobalReadIncsA"), comment="force to 0?"))
          module.add(SCSelectB32(dst=sgpr(tmpSgpr+1), src0=0, src1=sgpr("GlobalReadIncsB"), comment="force to 0?"))
          module.add(self.setTailSrd(tPA, sgpr(tmpSgpr+0)))
          module.addSpaceLine()
          module.add(self.setTailSrd(tPB, sgpr(tmpSgpr+1)))
          module.addSpaceLine()
          #module.add(self.getBomb())

        module.addComment("numIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)" \
            % (self.states.unrollChar, self.states.unrollChar))
        # size % DepthU
        module.add(scalarStaticDivideAndRemainder(tmpSgpr, loopCounterName, \
          "SizesSum+%u"%loopIdx, kernel["DepthU"], RegisterPoolResource(tmpSgpr+2, 2), 2))
        loopCounter = sgpr(loopCounterName)

        if kernel["LocalSplitU"] > 1:
          # (size % DepthU) + LSU - 1
          module.add(SAddU32(dst=loopCounter, src0=hex(kernel["LocalSplitU"]-1), src1=loopCounter, comment="(size % DepthU) + LSU - 1" ))
          dividend = tmpSgpr+2
          module.add(SMovB32(dst=sgpr(dividend), src=loopCounter, comment="copy for divide LSU" ))
          module.add(scalarStaticDivideAndRemainder( loopCounterName, None, dividend, kernel["LocalSplitU"], tmpSgprInfo, 0))

      # if GSU numIter=0 if gsuSumIdx != remainder
      if kernel["GlobalSplitU"] > 1:
        module.add(SCmpLgU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), \
            comment="gsuSumIdx == numIterPerWgRemainder"))
        module.add(SCMovB32(dst=loopCounter, src=hex(0), comment="numIter=0 if gsuSimIdx!=remainder"))

      # if tail numIter == 0 skip altogether
      skipTailLoopLabel = Label.getFormatting("SkipTailLoop%s"%(loopChar) )
      module.add(SCmpEQU32(src0=loopCounter, src1=hex(0), comment="numIter%s == 0"%loopChar ))
      module.add(SMovB32(dst=sgpr("OrigLoopCounter"), src=0, comment="repurpose to count each localRead increment"))
      module.add(SCBranchSCC1(labelName=skipTailLoopLabel, \
                comment="skip to end of tail loop b/c numIter==0"))

    ########################################
    # Unrolled Loop
    elif loopIdx == self.states.unrollIdx:
      loopCounterName = self.loopCounterName(kernel, loopIdx)
      loopCounter = sgpr(loopCounterName)
      if not self.do["PreLoop"]: module.add(ValueEndif())

      sumSize = "SizesSum+%u"%loopIdx
      #sumSize = self.sumSize(kernel, loopIdx)

      # TODO - use named arguments
      with self.allocTmpSgpr(3) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        quotient = loopCounterName
        dividend = sumSize
        divisor = kernel["DepthU"]
        if kernel["NoTailLoop"] and kernel["AssertSummationElementMultiple"] % kernel["DepthU"] != 0:
          # round up SizesSum/DepthU for noTailLoop case
          module.add(SAddI32(dst=sgpr(quotient), src0=(divisor - 1), src1=sgpr(dividend), \
              comment="round up SizeSum / DepthU" ))
          module.add(scalarStaticDivideAndRemainder(quotient, None, quotient, divisor, tmpSgprInfo, 0))
        else:
          module.add(scalarStaticDivideAndRemainder(quotient, None, dividend, divisor, tmpSgprInfo, 0))
        # if GSU numIter++ if gsuSumIdx < remainder
        if kernel["GlobalSplitU"] > 1:
          module.add(self.calculateLoopNumIterGsu(kernel, loopCounterName, tmpSgprInfo))

        module.add(SMovB32(dst=sgpr("OrigLoopCounter"), \
                  src=loopCounter, \
                  comment="copy loop counter"))
    else:
      # other summation, not unroll loop
      #printExit("no assembly support for 2+ dimensional summation")
      module.addComment1("%sother summation, numIter%s = size%s" \
          % (self.indent, loopChar, loopChar))
      loopCounter = self.loopCounter(kernel, loopIdx)
      module.add(SMovB32(dst=loopCounter, \
                src=sgpr("SizesSum+%u"%loopIdx), \
                comment="init loop counter"))

    return module

  ##############################################################################
  # Open Loop
  # uDu: 'None' means not generating branching label which decides which part of G2L
  #      buffer to write to LDS
  ##############################################################################
  def openLoop(self, kernel, tPA, tPB, loopIdx, uDu=None, noLabelGen=False, beginLabelOnly=False):
    module = Module("openLoop")
    # TODO - rewrite this function to simplify control-flow between tail-loop / unroll loop
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.states.unrollIdx
      self.states.inTailLoop = True
    loopChar = self.states.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if not tailLoop and not noLabelGen:
      module.add(Label("openLoop%s"%loopChar, ""))
    loopLabelBegin = Label("%sLoopBegin%s%s"%("Tail" if tailLoop else "", loopChar, "_G2L%s"%uDu if uDu is not None else "" ), "" )
    loopLabelEnd = Label("%sLoopEnd%s%s"%("Tail" if tailLoop else "", loopChar, "_G2L%s"%uDu if uDu is not None else ""), "" )

    if beginLabelOnly:
      # generate only beginLabel, then, return
      module.add(loopLabelBegin)
      return module

    # is numIter at least 1? otherwise skip to end
    # PGL needs a skip-check here if not bufferload
    # If kernel["SuppressNoLoadLoop"] we don't have a special loop for the 'last iter'
    loopCounter = self.loopCounter(kernel, loopIdx)
    if tailLoop:
      endCounter = 0
    elif kernel["PrefetchGlobalRead"] == 1:
      if kernel["SuppressNoLoadLoop"]:
        endCounter =  0
      else:
        endCounter = 1
    elif kernel["PrefetchGlobalRead"] == 2:
      if kernel["SuppressNoLoadLoop"]:
        endCounter =  1
      else:
        endCounter = 2
    else:
      endCounter =  0

    if tailLoop:
      # LSU not all threads will do summation
      if kernel["LocalSplitU"] > 1:
        module.addComment1("apply exec mask for LSU")
        tmpVgpr = self.vgprPool.checkOutAligned(2, 2, "tmpVgpr")
        tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
        dummy = self.vgprPool.checkOut(1,"dummy")
        sgId = self.vgprPool.checkOut(1,"sgId")
        divisor = kernel["SubGroup0"]*kernel["SubGroup1"]
        module.add(vectorStaticDivide(sgId, "Serial", divisor, tmpVgprRes))
        numIter = self.vgprPool.checkOut(1,"numIter")
        module.add(VMovB32(dst=vgpr(numIter), src=sgpr("SizesSum+0"), comment="sizeU to vgpr"))
        divisor = kernel["DepthU"]
        module.add(vectorStaticDivideAndRemainder(dummy, numIter, numIter, divisor, tmpVgprRes))
        self.vgprPool.checkIn(dummy)
        #module.add() dump(vgpr(sgId)) )
        #module.add() dump(vgpr(numIter)) )
        module.add(VCmpXLtU32(dst=VCC(), \
            src0=vgpr(sgId), src1=vgpr(numIter), comment="sgId < numIter"))
        self.vgprPool.checkIn(tmpVgpr)
        #self.tailNumIter = numIter
        #self.vgprPool.checkIn(numIter)
        # thread is active is sgId < numIter % LocalSplitU

      # begin loop
      if not noLabelGen:
        module.add(loopLabelBegin)

      # LSU mask for this iteration
      if kernel["LocalSplitU"] > 1:
        module.add(VCmpXLtU32(dst=VCC(), \
            src0=vgpr(sgId), src1=vgpr(numIter), comment="sgId < numIter"))
        module.add(VAddCOU32(dst=vgpr(sgId), dst1=VCC(), src0=hex(kernel["LocalSplitU"]), \
            src1=vgpr(sgId), comment="sgId+=LSU"))
        self.vgprPool.checkIn(sgId)
        self.vgprPool.checkIn(numIter)
        #module.add() dump(vgpr(sgId)) )

    else: # not tailloop:

      if loopIdx == self.states.unrollIdx:
        # 1 loop check is necessary only when AssertSummationElementMultiple % (DepthU * 2) != 0
        if kernel["PrefetchGlobalRead"] == 2 and kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) != 0:
          module.add(SCmpEQU32(
              src0=loopCounter, \
              src1=hex(endCounter-1), \
              comment="LoopCounter%s < EndCounter"%(loopChar) ))
          toPGR1 = Label.getFormatting(self.labels.getName("toPGR1"))
          module.add(SCBranchSCC1(labelName=toPGR1, comment="PGR=2 but only 1 loop, toPGR1"))

        module.add(SCmpLeU32(
            src0=loopCounter, \
            src1=hex(endCounter), \
            comment="LoopCounter%s < EndCounter"%(loopChar) ))
        jumpLabel = loopLabelEnd
        if kernel["PrefetchGlobalRead"]==2 and (not kernel["SuppressNoLoadLoop"]) and kernel["ExpandPointerSwap"]:
          # PGR=2 and EPS and no SuppressNoLoadLoop case, need to jump to EvenExit
          jumpLabel = Label("LoopEnd%s_evenexit"%(loopChar), "" )
        module.add(SCBranchSCC1(labelName=jumpLabel.getLabelName(), \
                  comment="do not enter Loop%s"%loopChar ))

      if not noLabelGen:
        module.add(loopLabelBegin)

      if loopIdx != self.states.unrollIdx:
        # reset LRO since these may have changed due to odd-iter exit ?
        if kernel["PrefetchGlobalRead"]:
          module.addComment0("openLoop - reset LRO for possible odd-iter exit")
          module.add(self.localReadResetOffsets(kernel, tPA))
          module.add(self.localReadResetOffsets(kernel, tPB))

    return module

  ##############################################################################
  # Close Loop
  # finalLoop : final unroll loop
  # uDu: 'None' means not generating branching label which decides which part of G2L
  #      buffer to write to LDS
  ##############################################################################
  def closeLoop(self, kernel, tPA, tPB, loopIdx, finalLoop, uDu=None, emitEndLabelOnly=False, oddLabel=False):
    module = Module("closeLoop")
    if emitEndLabelOnly:
      loopIdx = self.states.unrollIdx
      loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      module.add(Label("SkipTailLoop%s"%(loopChar), ""))
      return module

    finalJump = SCBranchSCC0
    nonFinalJumpNeeded = True

    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.states.unrollIdx
      loopChar = self.states.indexChars[kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = Label("TailLoopBegin%s%s"%(loopChar, "_G2L%s"%uDu if uDu is not None else ""), "" )
      loopLabelEnd = Label("TailLoopEnd%s%s"%(loopChar, "_G2L%s"%uDu if uDu is not None else ""), "" )
      loopLabelEndOddExit = Label("TailLoopEnd%s_oddexit"%(loopChar), "unroll loop odditer exit" )
      loopCounter = self.loopCounter(kernel, loopIdx)

      unrollInc      = 1
      KinInnerUnroll = kernel["InnerUnroll"]
      if kernel["EnableMatrixInstruction"]:
        unrollInc      *= kernel["MatrixInstK"]
        KinInnerUnroll *= kernel["MatrixInstK"]
      if kernel["AssertSummationElementMultiple"] % KinInnerUnroll == 0:
        unrollInc *= kernel["InnerUnroll"]

      module.addComment1("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

      module.add(SSubI32(
          dst=loopCounter, \
          src0=loopCounter, \
          src1=hex(unrollInc), \
          comment="dec counter%s (tailLoop)"%(loopChar) ))

      # Track # LDS reads?
      module.add(SAddU32(
        dst=sgpr("OrigLoopCounter"), \
        src0=sgpr("OrigLoopCounter"), \
        src1=hex(unrollInc),
        comment="inc counter%s"%(loopChar) ))

      endCounter = 0
      module.add(SCmpLeI32(
          src0=loopCounter, \
          src1=hex(endCounter), \
        comment="counter%s<=%d"%(loopChar,endCounter) ))
    else: # not tailloop
      loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]
      loopLabelBegin = Label("LoopBegin%s"%(loopChar), "" )
      loopLabelEnd = Label("LoopEnd%s"%(loopChar), "" )
      loopLabelEndOddExit = Label("LoopEnd%s_oddexit"%(loopChar), "unroll loop odditer exit" )
      loopLabelEndEvenExit = Label("LoopEnd%s_evenexit"%(loopChar), "unroll loop eveniter exit" )
      loopCounter = self.loopCounter(kernel, loopIdx)
      module.addComment1("closeLoop loop%s finalLoop=%d tailLoop=%d" % (loopChar, finalLoop, tailLoop))

      # If PrefetchGlobalRead=1 the loads in the loop prefetch next macro-tile
      # For the final trip through the unroll loop we need to ensure those loads stay in bounds.

      # One technique is to create a copy of the unroll loop with all loads removed.
      # However buffer load doesn't need this loop copy since we OOB loads can be suppressed by buffer limit hardware
      # So can do one more iteration (endCounter==0) in the main unroll loop, and adjust the pointer
      # increments appropriately.
      # Also sum idx other than unroll always compare against 0 (there is no PGR to account for)
      if kernel["PrefetchGlobalRead"] == 1 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.states.unrollIdx:
        endCounter = 1
      elif kernel["PrefetchGlobalRead"] == 2 and not kernel["SuppressNoLoadLoop"] and loopIdx == self.states.unrollIdx:
        endCounter = 2
      else:
        endCounter = 0

      if kernel["AssertSummationElementMultiple"] % (kernel["DepthU"] * 2) == 0 and endCounter > 0:
        # if AssertSummationElementMultiple is multiple of DepthU*2, loop exit is necessary only once in 2 Loop iterations
        #  In endCounter % 2 == 1 case, exit at lc % 2 == 0 (= oddLabel). It means no exit if not oddLabel
        #  In endCounter % 2 == 0 case, exit at lc % 2 == 1 (= not oddLabel). It means no exit if oddLabel
        # No exit case, no code is necessary except for final Loop

        # decrement by 2 if PGR=2 and StaggerU is 0, else 1
        decValue = 2 if kernel["PrefetchGlobalRead"]==2 and kernel["StaggerU"] == 0 else 1
        decCode = SSubU32(dst=loopCounter, src0=loopCounter, \
            src1=decValue, \
            comment="dec counter%s"%(loopChar) )
        condCode = SCmpEQI32(src0=loopCounter, \
            src1=hex(endCounter), \
            comment="counter%s==%d"%(loopChar,endCounter) )

        noExit = False

        if endCounter%2 != 0:
          if not oddLabel:
            noExit = True
        else:
          if oddLabel:
            noExit = True

        if noExit:
          # No exit. No dec code if decValue is 2
          if decValue == 2:
            decCode = ""
          condCode = ""
          nonFinalJumpNeeded = False
          if finalLoop:
            # No exit and finalLoop case, use s_branch (no condition)
            finalJump = SBranch

        if decCode: module.add(decCode)
        if condCode: module.add(condCode)
      else:
        module.add(SSubU32(
            dst=loopCounter, src0=loopCounter, \
            src1=1, \
            comment="dec counter%s"%(loopChar) ))

        module.add(SCmpEQI32(
            src0=loopCounter, \
            src1=hex(endCounter), \
            comment="counter%s==%d"%(loopChar,endCounter) ))

    jumpLabel = loopLabelEnd
    if not tailLoop and not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
      # in this case, odd or/and even code is generated and use odd/even exit to avoid skipping odd/even code
      # (end label is generated after odd/even code)
      jumpLabel = loopLabelEndOddExit if oddLabel else loopLabelEndEvenExit
    if not finalLoop:
      if nonFinalJumpNeeded:
        # just an exit check, else fall through to the next loop copy
        module.add(SCBranchSCC1(labelName=jumpLabel.getLabelName(), comment="exit Loop%s"%loopChar ))
    else: #finalLoop:

      if tailLoop and kernel.enabledSplitLDS:
        tailLoopLabelEnd = Label.getFormatting(
          "TailLoopEnd%s%s"%(loopChar, "_G2L%s"%(kernel["DepthULdsDivisor"]-1) if kernel.enabledSplitLDS else "") )
        module.add(SCBranchSCC1(labelName=tailLoopLabelEnd, comment="break Loop%s"%loopChar))
        thresForNextSubLoop = (uDu+1)*(kernel["_DepthULds"])
        module.add(SCmpGeU32(src0=sgpr("OrigLoopCounter"), src1=thresForNextSubLoop,
          comment="OrigLoopCounter >= %u (G2L buffer %u/%u)"%(thresForNextSubLoop, uDu, kernel["DepthULdsDivisor"]) ))

      module.add(finalJump(labelName=loopLabelBegin.getLabelName(), comment="restart Loop%s"%(loopChar)))

      if not tailLoop and loopIdx == self.states.unrollIdx:
        oddIterPreCode = Module()
        oddIterCode = Module()
        evenIterPreCode = Module()
        evenIterCode = Module()
        if not kernel["SuppressNoLoadLoop"] and kernel["ExpandPointerSwap"]:
          oddIterPreCode.add(loopLabelEndOddExit)
          # In this case we kept the 'no-load' loop which has LDS offsets assuming first bank of LDS
          # if we exit the main loop at an odd iter - need to swap LDS read pointers
          # so the ds_reads read from the 'high' buffer of LDS
          oddIterPreCode.addComment1("Select high bank of LDS")
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprA"]:
            oddIterCode.add(self.localReadSwapOffsets(kernel, False, tPA))
          # Generate local read address code only if DirectToVgpr is not enabled
          if not kernel["DirectToVgprB"]:
            oddIterCode.add(self.localReadSwapOffsets(kernel, False, tPB))

          evenIterPreCode.add(loopLabelEndEvenExit)
          # generate even code here (so far, for PrefetchGlobalRead=2 only)
          if kernel["PrefetchGlobalRead"]==2:
            # Generate local write address code only for PrefetchGlobalRead==2 (localWriteSwapOffsets does nothing if DirectToVgpr is enabled)
            # Code is unnecessary if DirectToLds is enabled, but internal SwapOffset is necessary if states.useInitAccVgprOpt is True
            if kernel["DirectToLdsA"]:
              if self.states.useInitAccVgprOpt:
                self.localWriteSwapOffsets(kernel, True, tPA)
            else:
              evenIterCode.add(self.localWriteSwapOffsets(kernel, False, tPA))
            if kernel["DirectToLdsB"]:
              if self.states.useInitAccVgprOpt:
                self.localWriteSwapOffsets(kernel, True, tPB)
            else:
              evenIterCode.add(self.localWriteSwapOffsets(kernel, False, tPB))
            # swap internal write pointer as well (except for states.useInitAccVgprOpt case)
            if not self.states.useInitAccVgprOpt:
              evenIterCode.add(self.localWriteSwapOffsets(kernel, True, tPA))
              evenIterCode.add(self.localWriteSwapOffsets(kernel, True, tPB))

        # generate even, odd exit code
        # not oddLabel case, order is even -> odd
        firstPreCode = evenIterPreCode
        firstCode = evenIterCode
        secondPreCode = oddIterPreCode
        secondCode = oddIterCode
        if oddLabel:
          # oddLabel case, swap the order (odd -> even)
          firstPreCode, secondPreCode = secondPreCode, firstPreCode
          firstCode, secondCode = secondCode, firstCode

        module.add(firstPreCode)
        module.add(firstCode)

        # if secondCode exist, add jump to skip secondCode
        if secondCode.count():
          module.add(SBranch(labelName=loopLabelEnd.getLabelName(), \
                    comment="exit unroll loop%s (and skip second exit code)"%(loopChar)))
        module.add(secondPreCode)
        module.add(secondCode)

      module.add(loopLabelEnd)

      if tailLoop:
        if len(kernel["ProblemType"]["IndicesSummation"]) > 1:
          # recover the 'damage' done to LRO:

          # if LRA is backed-up before (wlr case), we simply restore the addr (sub inc*loop doesn't work)
          tPList = []
          if self.oriLraA != None:
            if not kernel["DirectToVgprA"]: # no local read code if DirectToVgpr is enabled
              module.add(VMovB32(dst=vgpr("LocalReadAddrA"), src=vgpr(self.oriLraA), comment="restore LRA"))
            self.vgprPool.checkIn(self.oriLraA)
            self.oriLraA = None
          else:
            tPList.append(tPA)
          if self.oriLraB != None:
            if not kernel["DirectToVgprB"]: # no local read code if DirectToVgpr is enabled
              module.add(VMovB32(dst=vgpr("LocalReadAddrB"), src=vgpr(self.oriLraB), comment="restore LRA"))
            self.vgprPool.checkIn(self.oriLraB)
            self.oriLraB = None
          else:
            tPList.append(tPB)
          for tP in tPList:
            tc     = tP["tensorChar"]
            LdsPad = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
            inc    = kernel["LocalSplitU"]*(kernel["MacroTile%s"%tc]+LdsPad)*tP["bpe"]

            # aligned with localReadInc
            if kernel["EnableMatrixInstruction"]:
              if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
                inc = kernel["LocalSplitU"] * tP["bpe"]
              # No need to *= K, because LoopCounter is increased by K each time
              # inc *= kernel["MatrixInstK"]

            if not kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
              with self.allocTmpSgpr(1) as tmpSgprInfo:
                stmp = tmpSgprInfo.idx
                module.add(SMovB32(dst=sgpr(stmp), src=inc, comment="tailloop lds offset"))
                module.add(SMulI32(dst=sgpr(stmp), src0=sgpr("OrigLoopCounter"), src1=sgpr(stmp), comment="scale by mul"))
                module.add(VSubU32(dst=vgpr("LocalReadAddr%s"%tc), src0=vgpr("LocalReadAddr%s"%tc), src1=sgpr(stmp), comment="remove lro damage"))
          # if LWA is backed-up before, we simply restore the addr
          if self.oriLwaA != None:
            if not kernel["DirectToVgprA"]: # no local write code if DirectToVgpr is enabled
              module.add(VMovB32(dst=vgpr("LocalWriteAddrA"), src=vgpr(self.oriLwaA), comment="restore LWA"))
            if not kernel["DirectToVgprB"]: # no local write code if DirectToVgpr is enabled
              module.add(VMovB32(dst=vgpr("LocalWriteAddrB"), src=vgpr(self.oriLwaB), comment="restore LWA"))
            self.vgprPool.checkIn(self.oriLwaA)
            self.vgprPool.checkIn(self.oriLwaB)
            self.oriLwaA = None
            self.oriLwaB = None

    # restore all threads
    if tailLoop and kernel["LocalSplitU"] > 1:
      sgprCnt = self.states.laneSGPRCount
      waveSize = kernel["WavefrontSize"]
      module.addComment1("restore full exec mask")

      with self.allocTmpSgpr(sgprCnt) as tmpSgprInfo:
        fullExec = tmpSgprInfo.idx
        activeMask    = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
        SMovBX        = SMovB32 if (waveSize == 32) else SMovB64
        SOrSaveExecBX = SOrSaveExecB32 if (waveSize == 32) else SOrSaveExecB64
        module.add(SMovBX(dst=sgpr(fullExec,sgprCnt), src=activeMask, comment="restore all threads active"))
        module.add(SOrSaveExecBX (dst=sgpr(fullExec,sgprCnt), src=sgpr(fullExec,sgprCnt), comment="full mask -> exec" ))
    return module

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self, kernel, label = None):
    module = Module("endSummation")

    module.add(Label((self.labels.getUniqueNamePrefix("Summation_End") if label is None else label), ""))

    if kernel["StorePriorityOpt"]:
      module.add(SSetPrior(prior=0, comment="optimization store"))

    vbegin = self.states.a.startVgprValu
    vsize = self.states.lastVgprForReads - self.states.a.startVgprValu

    self.vgprPool.add(vbegin, vsize, "endSummation")
    module.addComment0("endSummation: add vgpr [%u...%u) to pool" % \
            (vbegin, vbegin+vsize))

    lastRegTag=None
    for i in range(self.states.lastPostLoopSgpr, self.sgprPool.size()):
      regTag = self.sgprPool.pool[i].tag
      if regTag != lastRegTag:
        lastRegTag = regTag
        if self.sgprPool.pool[i].status == RegisterPool.Status.InUse:
          module.add(self.undefineSgpr(regTag))

    if self.db["InitVgpr"] & 0x2:
      module.add(self.vgprPool.initTmps(self.consts.initVgprValue,start=0, stop=100))
    if 0: # FIXME: Can remove?
      for i in range(0,16+1):
         module.add(VMovB32(dst=vgpr(21), src=vgpr(21), comment="hack tmp in pool"))

    # this doesn't seem to do anything - not being aggressive with lastPostLoopSgpr
    if self.db["InitSgpr"] & 0x2:
      module.add(self.sgprPool.initTmps(self.consts.initSgprValue))

    if self.db["ConservativeWaitCnt"] & 0x10:
      module.add(SBarrier(comment="debug"))
      module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment=""))

    if kernel["SuppressNoLoadLoop"]:
      module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment="wait for all summation activity"))

    ########################################
    # Load kernel args needed by global write batch
    # Calculate storeSgprLoad
    module.addComment0("load store sgprs")
    storeSgprLoad = self.states.numStoreSgprToLoad

    # Define sgprs for kernel args
    runActivation = True if ((kernel["ProblemType"]["ActivationType"] != 'none') and (kernel["GlobalSplitU"] == 1) \
        and kernel["ActivationFused"]) else False
    self.defineSgpr("LoadStoreSgprs", storeSgprLoad, align=4)
    if storeSgprLoad:
      soffset = self.sgprs["LoadStoreSgprs"]
      if self.states.numSgprAddressBias:
        module.add(RegSet("s", "sgprAddressBias", soffset))
        soffset += self.states.numSgprAddressBias
        module.add(RegSet("s", "sgprBiasType", soffset))
        soffset += self.states.BiasType
      if runActivation:
        for name in kernel["ProblemType"]["ActivationType"].getAdditionalArgStringList():
          module.add(RegSet("s", "sgpr"+name, soffset))
          soffset += self.states.numActivationArgSize
      if self.states.numActivationTypeArgSize:
        module.add(RegSet("s", "sgprActivationType", soffset))
      argOffset = self.argLoader.getOffset() # Backup offset
      module.addModuleAsFlatItems(self.argLoader.loadAllKernArg(self.sgprs["LoadStoreSgprs"], "KernArgAddress", storeSgprLoad))
      self.argLoader.setOffset(argOffset) # Restore offset

    # define the rest sgprs
    if (not self.states.doShadowInit) and kernel["BufferStore"]:
      self.defineSgpr("SrdD", 4, 4)
      self.defineSgpr("SrdC", 4, 4)
      module.add(RegSet("s", "sgprSrdC", self.sgprs["SrdC"]))
      module.add(RegSet("s", "sgprSrdD", self.sgprs["SrdD"]))
    if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
      self.defineSgpr("SrdBias", 4, 4)
      module.add(RegSet("s", "sgprSrdBias", self.sgprs["SrdBias"]))
    # Load kernel args end
    ########################################

    # copy accumulated C from agpr to vgpr
    if kernel["EnableMatrixInstruction"]:
      #TODO avoid s_nop if its possible
      #instCycles = kernel["MatrixInstM"] // 2 # 32x32 is 64 cycles, 16x16 is 32 cycles, 4x4 is 8 cycles
      #module.add(SNop(waitState=instCycles))
      module.addComment1("Mapping of Acc register -> C Vgpr register")
      self.codes.accVgprRead = mapAcctoArchRegs(kernel, self.states.lrvwB)
      if kernel["MIArchVgpr"]:
        module.addComment1("Multiply MI out register with Alpha -> C Vgpr register")
        self.codes.mulAlpha = mulMIoutAlphaToArch(kernel, self.states.lrvwB, self.states.startVgprAlphaTmp)

    return module

  ##############################################################################
  # MFMA Iteration
  ##############################################################################
  def mfmaIter(self, kernel, tPA, tPB, u, innerUnroll, vregSetIdx, tail=False, firstIter=False):
    imod = Module("mi")
    shiftK = Module("shiftK")
    m = (u) % (self.states.numVgprBuffer+1) # local to use for MACs

    # calculate constant
    numRegistersIn   = kernel["ProblemType"]["DataType"].numRegisters()
    numRegistersOut  = kernel["MIRegPerOut"]
    loopCounterName  = self.loopCounterName(kernel, self.states.unrollIdx)
    accs_per_wave    = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] \
                       // self.states.kernel["WavefrontSize"] * numRegistersOut
    dividerFortidInK = kernel["MatrixInstN"] * kernel["MatrixInstB"]
    numMIInput       = kernel["MIInputPerThread"]
    miInInstType, miOutInstType = dataTypeToMfmaInstTypePair(kernel["ProblemType"]["DataType"], \
      kernel["ProblemType"]["Fp16AltImpl"])
    vgprPerInput     = int(numMIInput * numRegistersIn)
    shiftPerElement  = int(numRegistersIn * 32)
    s_nop            = 0
    gprfunc          = accvgpr if not kernel["MIArchVgpr"] else vgpr
    accumRegType     = "acc" if not kernel["MIArchVgpr"] else "v"
    mfma_1k          = True if (kernel["MFMA_BF16_1K"] or kernel["ProblemType"]["Fp16AltImpl"]) else False
    accStoreCIdx     = 0

    # alloc vgpr
    kReg    = None
    abReg   = None
    tmpVgpr = None
    dummy   = None

    if (numRegistersIn < 1) and ((kernel["UnrollMajorLDSA"] == False) or (kernel["UnrollMajorLDSB"] == False)):
      s_nop = 2

    # here we remap index to where it read for wider local read
    # ex. if we read 2 iteration at a time,
    #   original   : _ds_load_b64  valuA_X0_I0
    #   read 2 iter: _ds_load_b128 valuA_X0_I0 (we read valuA_X0_I0 and valuA_X1_I0)
    # instead of using valuA_X1_I0, we use valuA_X0_I0+2 as mfma input

    vgprBufferA_new = (m//self.states.numIterPerCoalescedReadA)*self.states.numIterPerCoalescedReadA
    vgprBufferA_new_offset = m%self.states.numIterPerCoalescedReadA*kernel["InnerUnroll"]*vgprPerInput

    vgprBufferB_new = (m//self.states.numIterPerCoalescedReadB)*self.states.numIterPerCoalescedReadB
    vgprBufferB_new_offset = m%self.states.numIterPerCoalescedReadB*kernel["InnerUnroll"]*vgprPerInput

    numVgprPerBlockA = self.states.a.numVgprG2L // 2
    numVgprPerBlockB = self.states.b.numVgprG2L // 2

    # handle multiple K element in MFMA instruction
    if tail and kernel["MatrixInstK"] > 1:
      kReg    = self.vgprPool.checkOut(1,"kReg") # remainder
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        shiftK.add(vectorStaticRemainder(dummy, kReg, "Serial", self.states.kernel["WavefrontSize"], tmpVgpr, tmpSgprInfo))
        shiftK.add(vectorStaticDivide(kReg, kReg, dividerFortidInK, tmpVgpr))
        shiftK.add(staticMultiply(vgpr(kReg), vgpr(kReg), numMIInput, tmpSgprInfo))

      with self.allocTmpSgpr(3) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        # replace 0 for differnet thread
        shiftK.add(VCmpGEI32(dst=sgpr(tmpSgpr, 2), src0=vgpr(kReg), src1=sgpr(loopCounterName), comment="check K index >= Size L"))
        for bk in range(0, vgprPerInput):
          for a in range(0, kernel["MIWaveTileA"]):
            for iui in range(0, innerUnroll):
              aStr = vgpr("ValuA_X%u_I%u+%u+%u" % (m, iui, a*vgprPerInput, bk), 1)
              shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=hex(0), src2=sgpr(tmpSgpr, 2), comment="set 0 if K_idx >= sizeL"))
          for b in range(0, kernel["MIWaveTileB"]):
            for iui in range(0, innerUnroll):
              bStr = vgpr("ValuB_X%u_I%u+%u+%u" % (m, iui, b*vgprPerInput, bk), 1)
              shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=hex(0), src2=sgpr(tmpSgpr, 2), comment="set 0 if K_idx >= sizeL"))

        # replace 0 for same thread
        if numMIInput > 1:
          abReg   = self.vgprPool.checkOutAligned(vgprPerInput, 2 if vgprPerInput>1 else 1, "abReg")
          tmpVgpr = self.vgprPool.checkOutAligned(2,2,"tmpVgpr")
          dummy   = self.vgprPool.checkOut(1,"dummy")
          shiftK.add(VSubU32(dst=vgpr(kReg), src0=sgpr(loopCounterName), src1=vgpr(kReg), comment="get distance between size and k index"))
          shiftK.add(VCmpLtI32(dst=sgpr(tmpSgpr,2), src0=vgpr(kReg), src1=numMIInput, comment="set partial 0 if distance less than input per thread"))
          shiftK.add(SAndB32(dst=sgpr(tmpSgpr+2), src0=sgpr(loopCounterName), src1=numMIInput-1, comment="get inputs for edge thread"))
          shiftK.add(SSubU32(dst=sgpr(tmpSgpr+2), src0=numMIInput, src1=sgpr(tmpSgpr+2), comment="use shift to fill 0 for outside element"))
          shiftK.add(SLShiftLeftB32(dst=sgpr(tmpSgpr+2), shiftHex=log2(shiftPerElement), src=sgpr(tmpSgpr+2), comment="use shift to fill 0 for outside element"))
          if vgprPerInput == 1:
            VShiftLeft = VLShiftLeftB32
          elif vgprPerInput == 2:
            VShiftLeft = VLShiftLeftB64
          for a in range(0, kernel["MIWaveTileA"]):
            for iui in range(0, innerUnroll):
              iuiA_new = (iui//self.states.numReadsIterCoalescedA)*self.states.numReadsIterCoalescedA
              iuiA_new_offset = iui%self.states.numReadsIterCoalescedA*vgprPerInput
              a_new = a*vgprPerInput*self.states.numReadsIterCoalescedA
              aStr = vgpr("ValuA_X%u_I%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset), vgprPerInput)
              tmpVregIdx = 0
              shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgpr+2), src=aStr, comment=""))
              for bk in range(0, vgprPerInput):
                aStr  = vgpr("ValuA_X%u_I%u+%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset, bk), 1)
                if kernel["DirectToVgprA"]:
                  # overwrite aStr for DirectToVgprA
                  tmp   = tmpVregIdx + bk
                  aStr  = vgpr("G2LA+%u" % (tmp), vgprPerInput)
                shiftK.add(VCndMaskB32(dst=aStr, src0=aStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgpr, 2), comment=""))
          for b in range(0, kernel["MIWaveTileB"]):
            for iui in range(0, innerUnroll):
              iuiB_new = (iui//self.states.numReadsIterCoalescedB)*self.states.numReadsIterCoalescedB
              iuiB_new_offset = iui%self.states.numReadsIterCoalescedB*vgprPerInput
              b_new = b*vgprPerInput*self.states.numReadsIterCoalescedB
              bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset), vgprPerInput)
              tmpVregIdx = 0
              shiftK.add(VShiftLeft(dst=vgpr(abReg, vgprPerInput), shiftHex=sgpr(tmpSgpr+2), src=bStr, comment=""))
              for bk in range(0, vgprPerInput):
                bStr = vgpr("ValuB_X%u_I%u+%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset, bk), 1)
                if kernel["DirectToVgprB"]:
                  # overwrite bStr for DirectToVgprB
                  tmp   = tmpVregIdx + bk
                  bStr  = vgpr("G2LB+%u" % (tmp), 1)
                shiftK.add(VCndMaskB32(dst=bStr, src0=bStr, src1=vgpr(abReg+bk), src2=sgpr(tmpSgpr, 2), comment=""))

      s_nop = 2

    if s_nop != 0:
      imod.add(SNop(waitState=(s_nop - 1), comment=""))

    for iui in range(0, innerUnroll):
      iuiA_new = (iui//self.states.numReadsIterCoalescedA)*self.states.numReadsIterCoalescedA
      iuiA_new_offset = iui%self.states.numReadsIterCoalescedA*vgprPerInput
      iuiB_new = (iui//self.states.numReadsIterCoalescedB)*self.states.numReadsIterCoalescedB
      iuiB_new_offset = iui%self.states.numReadsIterCoalescedB*vgprPerInput
      zgemmVaddSrcCheck = [[], [], []] # to avoid generating redundant v_add
      outer = 1
      loopSwap = False
      # complex case, swap inner loop and outer loop so that idxA comes outer
      # this is to re-use same tmp vgpr to nagate ai or ar
      if kernel["ProblemType"]["DataType"].isComplex() and tPB["tile01Idx"]:
        outer = 0
        loopSwap = True
      inner = 1 - outer # inner is the opposite of outer
      for idxOuter in range(0, kernel["MIWaveTile"][outer]):
        for idxInner in range(0, kernel["MIWaveTile"][inner]):
          idx0 = idxInner
          idx1 = idxOuter
          if loopSwap:
            idx0, idx1 = idx1, idx0
          accIdx   = idx1 * kernel["MIWaveTile"][0] + idx0
          accStart = accIdx * accs_per_wave
          accEnd   = accStart + accs_per_wave - 1
          accStartSrc1 = accStart
          accEndSrc1   = accEnd
          accStartSrc2 = accStart
          accEndSrc2   = accEnd
          if firstIter:
            # use the last accs_per_wave as src (assuming only these are initialized to 0)
            numAccvgprs = self.states.c.numVgprValu if kernel["MIArchVgpr"] else self.states.totalAgprs
            accStartSrc1 = numAccvgprs - accs_per_wave
            accEndSrc1   = accStartSrc1 + accs_per_wave - 1
          idxA     = idx0 if tPB["tile01Idx"] else idx1
          idxB     = idx1 if tPB["tile01Idx"] else idx0
          a_new    = idxA*vgprPerInput*self.states.numReadsIterCoalescedA
          b_new    = idxB*vgprPerInput*self.states.numReadsIterCoalescedB
          aStr     = "ValuA_X%u_I%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset)
          bStr     = "ValuB_X%u_I%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset)
          if kernel["DirectToVgprA"]:
              # overwrite aStr for DirectToVgprA
              numVgprValuAPerBlock = kernel["MIWaveTileA"] * kernel["MIInputPerThread"] * tPA["bpe"] // self.states.bpr
              # re-calculate vgprBufferA_new and offset for DirectToVgpr. Use u instead of m (number of local prefetch buffer does not matter)
              vgprBufferA_new = (u//self.states.numIterPerCoalescedReadA)*self.states.numIterPerCoalescedReadA
              vgprBufferA_new_offset = u%self.states.numIterPerCoalescedReadA*kernel["InnerUnroll"]*vgprPerInput
              a_new += vregSetIdx * numVgprPerBlockA + (iuiA_new + vgprBufferA_new * kernel["InnerUnroll"]) * numVgprValuAPerBlock
              aStr  = "G2LA+%u+%u+%u" % (a_new, vgprBufferA_new_offset, iuiA_new_offset)
              # self.states.vgprValuDouble case, need to change valuB to toggle double buffer
              if self.states.vgprValuDouble and vregSetIdx > 0:
                numOneSet = self.states.b.numVgprValu//2
                bStr += "+%u"%(vregSetIdx * numOneSet)
          if kernel["DirectToVgprB"]:
              # overwrite bStr for DirectToVgprB
              numVgprValuBPerBlock = kernel["MIWaveTileB"] * kernel["MIInputPerThread"] * tPB["bpe"] // self.states.bpr
              # re-calculate vgprBufferB_new and offset for DirectToVgpr. Use u instead of m (number of local prefetch buffer does not matter)
              vgprBufferB_new = (u//self.states.numIterPerCoalescedReadB)*self.states.numIterPerCoalescedReadB
              vgprBufferB_new_offset = u%self.states.numIterPerCoalescedReadB*kernel["InnerUnroll"]*vgprPerInput
              b_new += vregSetIdx * numVgprPerBlockB + (iuiB_new + vgprBufferB_new * kernel["InnerUnroll"]) * numVgprValuBPerBlock
              bStr  = "G2LB+%u+%u+%u" % (b_new, vgprBufferB_new_offset, iuiB_new_offset)
              # self.states.vgprValuDouble case, need to change valuA to toggle double buffer
              if self.states.vgprValuDouble and vregSetIdx > 0:
                numOneSet = self.states.a.numVgprValu//2
                aStr += "+%u"%(vregSetIdx * numOneSet)
          aStr     = vgpr(aStr, vgprPerInput)
          bStr     = vgpr(bStr, vgprPerInput)
          Str0     = aStr if tPB["tile01Idx"] else bStr
          Str1     = bStr if tPB["tile01Idx"] else aStr

          if kernel["ProblemType"]["DataType"].isComplex():
            # override because complex mul is emulated by 4 mfma insts
            # TODO: adopt component system
            miInInstType = miOutInstType #"f32" for SingleComplex, "f64" for DoubleComplex
            ccA = kernel["ProblemType"]["ComplexConjugateA"]
            ccB = kernel["ProblemType"]["ComplexConjugateB"]
            ccVgprs = [None]*3 # three terms that can be negated: [real1, imag0, imag1]
            ccInsts = [None]*3
            accImOffset = accVgprImagNumOffset(kernel, self.states.lrvwB)
            # for firstIter, need to use accStartSrc for img instead of adding accImOffset
            accStartSrcImg1 = accStartSrc1 if firstIter else accStartSrc1+accImOffset
            accEndSrcImg1 = accStartSrcImg1 + accs_per_wave - 1
            accStartSrcImg2 = accStartSrc2+accImOffset
            accEndSrcImg2 = accStartSrcImg2 + accs_per_wave - 1

            # vgpr A,B setting. In complex case, numRegistersIn does not match. Use numRegistersOut instead
            ar = vgpr("ValuA_X%u_I%u+%u+%u+%u"   % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset), numRegistersOut)
            ai = vgpr("ValuA_X%u_I%u+%u+%u+%u+%u" % (vgprBufferA_new, iuiA_new, a_new, vgprBufferA_new_offset, iuiA_new_offset, numRegistersOut), numRegistersOut)
            br = vgpr("ValuB_X%u_I%u+%u+%u+%u"   % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset), numRegistersOut)
            bi = vgpr("ValuB_X%u_I%u+%u+%u+%u+%u" % (vgprBufferB_new, iuiB_new, b_new, vgprBufferB_new_offset, iuiB_new_offset, numRegistersOut), numRegistersOut)
            if kernel["DirectToVgprA"]:
              ## overwrite aStr for DirectToVgprA
              ar  = vgpr("G2LA+%u+%u+%u" % (a_new, vgprBufferA_new_offset, iuiA_new_offset), numRegistersOut)
              ai  = vgpr("G2LA+%u+%u+%u+%u" % (a_new, vgprBufferA_new_offset, iuiA_new_offset, numRegistersOut), numRegistersOut)
            if kernel["DirectToVgprB"]:
              # overwrite bStr for DirectToVgprB
              br  = vgpr("G2LB+%u+%u+%u" % (b_new, vgprBufferB_new_offset, iuiB_new_offset), numRegistersOut)
              bi  = vgpr("G2LB+%u+%u+%u+%u" % (b_new, vgprBufferB_new_offset, iuiB_new_offset, numRegistersOut), numRegistersOut)
            minus_ar = ar.getMinus()
            minus_ai = ai.getMinus()
            if miOutInstType == InstType.INST_F32:
              VAddX = VAddF32
            elif miOutInstType == InstType.INST_F64:
              VAddX = VAddF64
            else:
              printExit("Unsupported v_add type %s"%miOutInstType)
            offsetVgpr = [0,0,0]
            forceGenerate = ccA and ccB # so far, v_add is always necessary for ccA and ccB case
            if ccA == ccB:
              arrayIndex = 0
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate r1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = VAddX(dst=vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), src0=minus_ai, src1=0, comment="Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
            if ccA:
              arrayIndex = 1
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i0")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ai not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = VAddX(dst=vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), src0=minus_ai, src1=0, comment="Ai=-Ai")
                zgemmVaddSrcCheck[arrayIndex].append(ai)
            if ccB:
              arrayIndex = 2
              ccVgprs[arrayIndex] = self.vgprPool.checkOutAligned(numRegistersOut, numRegistersOut, "negate i1")
              # generate negate code only when same code is not generated (avoid generating same (redundant) code again
              if forceGenerate or (ar not in zgemmVaddSrcCheck[arrayIndex]):
                ccInsts[arrayIndex] = VAddX(dst=vgpr(ccVgprs[arrayIndex] + offsetVgpr[arrayIndex], numRegistersOut), src0=minus_ar, src1=0, comment="Ar=-Ar")
                zgemmVaddSrcCheck[arrayIndex].append(ar)
            (src0, src1) = (br, ar) if kernel["SourceSwap"] else (ar, br)
            for inst in ccInsts:
              if inst is not None:
                imod.add(inst)
            variant = [kernel["MatrixInstM"], kernel["MatrixInstN"], kernel["MatrixInstK"]]
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=gprfunc(accStart, (accEnd-accStart+1)), a=src0, b=src1, acc2=gprfunc(accStartSrc1, (accEndSrc1-accStartSrc1+1)), \
                     comment="Cr += Ar*Br"))
            (src0, src1) = (bi, (vgpr(ccVgprs[0] + offsetVgpr[0], numRegistersOut) if ccVgprs[0] else ai)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[0] + offsetVgpr[0], numRegistersOut) if ccVgprs[0] else ai), bi)
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=gprfunc((accStart+accStoreCIdx), (accEnd-accStart+1)), a=src0, b=src1, acc2=gprfunc(accStartSrc2, (accEndSrc2-accStartSrc2+1)), \
                     comment="Cr += %sAi*Bi"%("-" if ccVgprs[0] else "")))
            (src0, src1) = (br, (vgpr(ccVgprs[1] + offsetVgpr[1], numRegistersOut) if ccVgprs[1] else ai)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[1] + offsetVgpr[1], numRegistersOut) if ccVgprs[1] else ai), br)
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=gprfunc((accStart+accImOffset), (accEnd-accStart+1)), a=src0, b=src1, acc2=gprfunc(accStartSrcImg1, (accEndSrcImg1-accStartSrcImg1+1)), \
                     comment="Ci += %sAi*Br"%("-" if ccVgprs[1] else "")))
            (src0, src1) = (bi, (vgpr(ccVgprs[2] + offsetVgpr[2], numRegistersOut) if ccVgprs[2] else ar)) if kernel["SourceSwap"] else ((vgpr(ccVgprs[2] + offsetVgpr[2], numRegistersOut) if ccVgprs[2] else ar), bi)
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=False, \
                     acc=gprfunc((accStart+accImOffset+accStoreCIdx), (accEnd-accStart+1)), a=src0, b=src1, acc2=gprfunc(accStartSrcImg2, (accEndSrcImg2-accStartSrcImg2+1)), \
                     comment="Ci += %sAr*Bi"%("-" if ccVgprs[2] else "")))

            for v in ccVgprs:
              if v is not None: self.vgprPool.checkIn(v)
          else:
            if kernel["SourceSwap"]:
              src0 = Str1
              src1 = Str0
            else:
              src0 = Str0
              src1 = Str1
            variant = [kernel["MatrixInstM"], kernel["MatrixInstN"], kernel["MatrixInstK"]]
            imod.add(MFMAInstruction(instType=miInInstType, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                     acc=gprfunc((accStart+accStoreCIdx), (accEnd-accStart+1)), \
                                     a=src0, b=src1, acc2=gprfunc(accStartSrc1, (accEndSrc1-accStartSrc1+1)), \
                                     comment="left value = %s[%u+%u:%u+%u]" % (accumRegType, accStart, accStoreCIdx, accEnd, accStoreCIdx)))

    # release register
    if kReg is not None: self.vgprPool.checkIn(kReg)
    if abReg is not None: self.vgprPool.checkIn(abReg)
    if tmpVgpr is not None: self.vgprPool.checkIn(tmpVgpr)
    if dummy is not None: self.vgprPool.checkIn(dummy)

    mfmaMod = Module("mfmaCode")
    mfmaMod.add(shiftK)
    mfmaMod.add(imod)

    return mfmaMod

  ##############################################################################
  # At Least 1 Unroll
  # prefetch means this is in the prefetch code, either before unroll loop
  # or in the PAP code.
  # isOptNLL : this is for the store-interleaved NLL optimization
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel, prefetch, isOptNLL):
    isLongBranch = False
    if kernel["ProblemType"]["ActivationType"] == 'all':
      acclen = getAccToArchLen(kernel, self.states.lrvwB)
      # Just a rough calculation
      if acclen > 100:
        isLongBranch = True

    module = Module("openSumAtLeastUnroll")
    if prefetch:
      if not isOptNLL:
        module.add(self.checkLastIter(kernel))
        if kernel["StorePriorityOpt"]:
          module.add(SSetPrior(prior=0, comment="optimization store"))
        if self.states.doShadowInit:
          shadowName = Label.getFormatting("ShadowInitStart")
          module.add(SCBranchSCC1(labelName=shadowName, \
              comment="skip to ShadowInitStart iter b/c numIter==0"))
        else:
          loopChar = self.states.indexChars[ \
              kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx]]
          labelName = Label.getFormatting("LoopEnd%s"%loopChar)
          module.add(SCBranchSCC1(labelName=labelName, \
                    comment="skip to unrollLoop end loop%s iter b/c numIter==0" % loopChar))
    elif isOptNLL:
      skipOptNLL = Label("OptNLL_End", "")
      with self.allocTmpSgpr(2) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.add(self.checkIsBetaZero(kernel, tmpSgpr, skipOptNLL, isLongBranch=isLongBranch, posNeg=1))

        # check alpha
        if self.do["ApplyAlpha"]:
          # (The new hgemm (h,h,h,h,s,s) is included in ComputeType=Single)
          if kernel["ProblemType"]["ComputeDataType"].isHalf():
            # for (h,h,h,h,h,h) no HPA,
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x3c003c00), comment="Packed alpha==1.0"))
            module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=sgpr(tmpSgpr), comment="alpha == 1.0?"))

          # Shouldn't go here. Currently, DataType=B->ComputeDataType=S
          # (bf-gemm is included in ComputeType=Single)
          elif kernel["ProblemType"]["ComputeDataType"].isBFloat16():
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x3f803f80), comment="Packed alpha==1.0"))
            module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=sgpr(tmpSgpr), comment="alpha == 1.0?"))

          elif kernel["ProblemType"]["ComputeDataType"].isInt32():
            module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=1, comment="Alpha == 1.0 ?"))

          # This covers sgemm, bfgemm + HPA (b,b,b,b,s,s), and also hgemm (h,h,h,h,s,s)
          elif kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(SCmpEQU32(src0=sgpr("Alpha"), src1=1.0, comment="Alpha == 1.0 ?"))

          elif kernel["ProblemType"]["ComputeDataType"].isDouble():
            module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(0x00000000), comment="Low part of double 1.0"))
            module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0x3ff00000), comment="High part of double 1.0"))
            module.add(SCmpEQU64(dst=sgpr("Alpha",2), src=sgpr(tmpSgpr,2), comment="Alpha == 1.0 ?"))

          elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
            module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(1.0), comment="Real part of 1.0"))
            module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0.0), comment="Imaginary part of 1.0"))
            module.add(SCmpEQU64(src0=sgpr("Alpha",2), src1=sgpr(tmpSgpr,2), comment="Alpha == 1.0 ?"))

          elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
            module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(0x00000000), comment="lsb of real part of 1.0"))
            module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0x3ff00000), comment="msb of real part of 1.0"))
            module.add(SCmpEQU64(src0=sgpr("Alpha",2), src1=sgpr(tmpSgpr,2), comment="Alpha.real == 1.0 ?"))
            module.add(SCBranchSCC0(labelName=skipOptNLL.getLabelName(), comment="branch if alpha.real != 1"))
            module.add(SMovB32(dst=sgpr(tmpSgpr+0), src=hex(0x00000000), comment="lsb of imag part of 0.0"))
            module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0x00000000), comment="msb of imag part of 0.0"))
            module.add(SCmpEQU64(src0=sgpr("Alpha+2",2), src1=sgpr(tmpSgpr,2), comment="Alpha.imag == 0.0 ?"))

          if isLongBranch:
            module.add(self.longBranchScc0(skipOptNLL, posNeg=1, comment="branch if alpha != 1"))
          else:
            module.add(SCBranchSCC0(labelName=skipOptNLL.getLabelName(), comment="branch if alpha != 1"))
          module.addSpaceLine()

        module.add(self.checkIsEdge(kernel, tmpSgpr, skipOptNLL, isLongBranch=isLongBranch))
        module.addSpaceLine()

        # Check tail loop required:
        # Skip tail loop check if noTailLoop is true
        if not kernel["NoTailLoop"]:
          loopChar = self.states.indexChars[ \
              kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx]]
          module.add(scalarStaticDivideAndRemainder(tmpSgpr, tmpSgpr+1, "SizesSum+%u"%self.states.unrollIdx, \
                    kernel["DepthU"], RegisterPoolResource(tmpSgpr+2, 2), 2))
          module.add(SCmpEQU32(src0=sgpr(tmpSgpr+1), src1=hex(0), comment="numIter%s == 0"%loopChar ))
          if isLongBranch:
            module.add(self.longBranchScc0(skipOptNLL, posNeg=1, comment="skip if tail loop required"))
          else:
            module.add(SCBranchSCC0(labelName=skipOptNLL.getLabelName(), comment="skip if tail loop required"))

      # save the vgprPool for generating the normal path.
      # dump the 'dirty' pool upon s_endpgm and swap back the 'clean' pool
      # so we can avoid explicit vgpr check-in/out
      self.savedVgprPool = deepcopy(self.vgprPool)
      self.savedSgprPool = deepcopy(self.sgprPool)

      # comment out the following codes that attempt to reduce vgpr consumption
      # however, the kernel vgpr count is governed by peak vgpr consumption so saving
      # a few here shouldn't affect kernel's overall vgpr consumption.
      # the following code is for reference and will be removed in the future
      """
      added = [] # track registers added to pool
      if kernel["PrefetchGlobalRead"]:
        if not kernel["DirectToLdsA"]:
          added.append(self.vgprPool.addRange(self.states.a.startVgprG2L, \
              self.states.a.startVgprG2L+self.states.a.numVgprG2L-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.states.a.startVgprLocalWriteAddr, \
                       self.states.a.startVgprLocalWriteAddr, "startOptNLL"))
        if not kernel["DirectToLdsB"]:
          added.append(self.vgprPool.addRange(self.states.b.startVgprG2L, \
              self.states.b.startVgprG2L+self.states.b.numVgprG2L-1, "startOptNLL"))
          added.append(self.vgprPool.addRange(self.states.b.startVgprLocalWriteAddr, \
                       self.states.b.startVgprLocalWriteAddr, "startOptNLL"))

      if kernel["BufferLoad"]:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadOffsetA, \
            self.startVgprGlobalReadOffsetB, "startOptNLL"))
      else:
        added.append(self.vgprPool.addRange(self.startVgprGlobalReadAddressesA, \
            self.startVgprGlobalReadAddressesB, "startOptNLL"))
      module.addComment1("reclaim VGPRS: " + ", ".join(added))
      """

    return module

  ##############################################################################
  def closeSumAtLeastUnroll(self, kernel, tPA, tPB, prefetch, isOptNLL, isNGLL):
    module = Module("closeSumAtLeastUnroll")
    if not prefetch:
      if isNGLL:
        toPGR1 = Label(self.labels.getName("toPGR1"), "")
        module.add(toPGR1)
      else:
        if isOptNLL:
            endSumLabel = "Summation_End_OptNLL"

            module.addComment0("Stores for OptNLL")
            module.add(self.endSummation(kernel, endSumLabel))

            # perhaps could work with LSU>1 by adding other indices here, but not tested
            assert (kernel["LocalSplitU"] == 1)
            module.add(self.notLocalSplitUGlobalWriteIndices(kernel))

            # add stores for opt NLL
            (fullVw, elements) = self.notLocalFullTileElements(kernel, False)
            alpha = False
            beta = False
            module.add(self.globalWriteElements(kernel, tPA, tPB, [fullVw], [elements], applyAlpha=alpha, betas=[beta], edges=[False]))

            self.cleanupGlobalWrite(kernel)
            module.addSpaceLine()
            module.add(self.functionEnd(False))
            module.add(Label("OptNLL_End", ""))

        else:
          module.add(Label("PrefetchGlobalLastIterEnd", ""))

    # swap back vgpr pool if any
    if self.savedVgprPool != None:
      # in case pool size in current path is larger than pool size in main path
      # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
      oldSize = self.savedVgprPool.size()
      newSize = self.vgprPool.size()
      if newSize > self.savedVgprPool.size():
        for i in range(oldSize,newSize):
          self.savedVgprPool.pool.append(self.savedVgprPool.Register(RegisterPool.Status.Available,"restore vgprPool"))
      self.vgprPool = self.savedVgprPool # restore vgprPool before alternate path
      self.savedVgprPool = None
    # swap back sgpr pool if any
    if self.savedSgprPool != None:
      # in case pool size in current path is larger than pool size in main path
      # and it will miss allocate vgpr since allocating vgpr is based on pool size in main path
      oldSize = self.savedSgprPool.size()
      newSize = self.sgprPool.size()
      if newSize > self.savedSgprPool.size():
        for i in range(oldSize-1,newSize):
          self.savedSgprPool.pool.append(self.savedSgprPool.Register(RegisterPool.Status.Available,"restore sgprPool"))
      self.sgprPool = self.savedSgprPool # restore vgprPool before alternate path
      self.savedSgprPool = None
    return module

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def incrementSrd(self, tP, incLower, incUpper, checkShadowLimitCopy=True):
    imod = Module("incrementSrd")
    tc = tP["tensorChar"]

    imod.add(SAddU32(dst=sgpr("Srd%s+0"%(tc)), \
                     src0=sgpr("Srd%s+0"%(tc)), \
                     src1=incLower, \
                     comment="gra SRD += inc(lower)" ))
    imod.add(SAddCU32(dst=sgpr("Srd%s+1"%(tc)), \
                      src0=sgpr("Srd%s+1"%(tc)), \
                      src1=incUpper, \
                      comment="gra SRD += inc(upper)" ))

    # also have to move the boundary since we change the base
    # so less buffers to the edge:
    if self.states.use64bShadowLimit:
      imod.add(SSubU32(dst=sgpr("ShadowLimit%s+0"%tc), \
                       src0=sgpr("ShadowLimit%s+0"%tc), \
                       src1=incLower, \
                       comment="limit -= inc)"))
      imod.add(SSubBU32(dst=sgpr("ShadowLimit%s+1"%tc), \
                        src0=sgpr("ShadowLimit%s+1"%tc), \
                        src1=incUpper, \
                        comment="limit -= inc)" ))
      if checkShadowLimitCopy:
        imod.add(SCmpEQU32(src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="are we within 2^32?"))
        if self.states.staggerU:
          # staggerU case, need to restore BufferLimit when ShadowLimit goes to negative value
          imod.add(SCSelectB32(dst=sgpr("Srd%s+2"%tc), src0=sgpr("ShadowLimit%s+0"%tc), src1="BufferLimit", comment="Move shadow to real if we are within 2^32"))
        else:
          imod.add(SCMovB32(dst=sgpr("Srd%s+2"%tc), src=sgpr("ShadowLimit%s+0"%tc), comment="Move shadow to real if we are within 2^32"))
    else:
      imod.add(SSubU32(dst=sgpr("Srd%s+2"%(tc)), \
                       src0=sgpr("Srd%s+2"%(tc)), \
                       src1=incLower, \
                       comment="limit -= inc)" ))
    return imod

  ##############################################################################
  # incLower must be constant or SGPR unsigned value
  def setTailSrd(self, tP, incLower):
    # In SuppressNoLoadLoop, the final loop iteration moves the SRD base forward
    # and the ShadowLimit backwards by one extra 'click' of GlobalReadIncs[AB].
    # Note the ShadowLimit may become negative - for example edge tiles where the
    # increment is > tile width.
    # The SuppressNoLoadLoop mode also forces the SRD limit to 0 on the final iteration.
    # The code here undoes the final click step by moving the base backwards and the
    # limit forwards (reading from the ShadowLimit).
    # It only works if use64bShadowLimit is enabled (since this enables use of the ShadowLimit)

    tc = tP["tensorChar"]
    module = Module("setTailSrd")
    incUpper = 0

    module.add(SSubU32(dst=sgpr("Srd%s+0"%(tc)), \
          src0=sgpr("Srd%s+0"%(tc)), src1=incLower, \
          comment="gra SRD -= inc(lower)" ))
    module.add(SSubBU32(dst=sgpr("Srd%s+1"%(tc)), \
          src0=sgpr("Srd%s+1"%(tc)), src1=incUpper, \
          comment="gra SRD -= inc(upper)" ))

    # using Shadow limit here which only works with 64-bit PBC:
    assert(self.states.use64bShadowLimit)

    module.add(SAddU32(dst=sgpr("ShadowLimit%s+0"%tc), \
          src0=sgpr("ShadowLimit%s+0"%tc), src1=incLower, \
          comment="limit -= inc)"))
    module.add(SAddCU32(dst=sgpr("ShadowLimit%s+1"%tc), \
          src0=sgpr("ShadowLimit%s+1"%tc), src1=incUpper, \
          comment="limit -= inc)" ))
    module.add(SCmpEQU32(src0=sgpr("ShadowLimit%s+1"%tc), src1=0, comment="are we within 2^32?"))
    module.add(SCMovB32(dst=sgpr("Srd%s+2"%tc), src=sgpr("ShadowLimit%s+0"%tc), comment="Move shadow to real if we are within 2^32"))

    return module

  ##############################################################################
  # Global Read: Increment A/B
  # loopIdx is summation idx:
  #   self.states.unrollIdx, or an idx from 0..NumIndicesSummation
  # prefetchIndex is >0 (1...PrefetchGlobalRead) if this increment follows a
  #   global prefetch or 0 otherwise
  ##############################################################################
  def globalReadIncrement(self, kernel, imod, loopIdx, tP, prefetchIndex):
    if not self.do["GlobalInc"]: return ""
    tc = tP["tensorChar"]
    loopChar = self.states.indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][loopIdx]]

    imod.addComment1("global read inc %s loop%s"%(tc,loopChar))

    if kernel["BufferLoad"]:
      # TODO - does this handle N-dim tensors correctly?
      #if tP["isB"]:
      #  module.add(SMovB32(dst=sgpr("OffsetB"), src=sgpr("SrdB+0"), comment="hack to save"))
      if self.states.staggerU and loopIdx == self.states.unrollIdx:
        # add a wrap increment, if needed:
        with self.allocTmpSgpr(3) as tmpSgprInfo:
          incLower = tmpSgprInfo.idx
          incUpper = incLower + 1
          tmpS =    incLower + 2
          if prefetchIndex:
            imod.add(SAddU32(dst=sgpr(tmpS), src0=self.loopCounter(kernel, self.states.unrollIdx), src1=prefetchIndex, comment="remove pf(%u)"%prefetchIndex))
            imod.add(SCmpEQU32(src0=sgpr("StaggerUIter"), src1=sgpr(tmpS), comment="Is this wrapIter? (pf)"))
          else:
            imod.add(SCmpEQU32(src0=self.loopCounter(kernel, self.states.unrollIdx), \
                      src1=sgpr("StaggerUIter"), comment="Is this the wrapIter?"))
          imod.add(SCSelectB32(dst=sgpr(incLower), src0=sgpr("WrapU%s+0"%tc), src1=sgpr("GlobalReadIncs%s+%u"%(tc,self.states.unrollIdx)), \
                      comment="incLower <- ?"))
          imod.add(SCSelectB32(dst=sgpr(incUpper), src0=sgpr("WrapU%s+1"%tc), src1=0,
                      comment="incUpper <- ?"))
          imod.add(self.incrementSrd(tP, sgpr(incLower), sgpr(incUpper), checkShadowLimitCopy=True))
      else:
        if loopIdx != self.states.unrollIdx or (tc in ('A', 'B') and kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx] in kernel["ProblemType"]["MirrorDims%s"%tc]):
          with self.allocTmpSgpr(1) as tmpSgprInfo:
            incUpper = tmpSgprInfo.idx
            # GRO may be negative for other summation if stride-other < stride-unroll or if mirror dim.
            imod.add(SAShiftRightI32(dst=sgpr(incUpper), shiftHex=31, src=sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), comment="sign-extend"))
        else:
          incUpper = 0 # GRO is positive for loop unroll
        imod.add( self.incrementSrd(tP, sgpr("GlobalReadIncs%s+%u"%(tc,loopIdx)), incUpper))
    else:
      graIdx = 0
      for _ in range(0, tP["nrp"]):
        for _ in range(0, tP["nrpv"]):
          for _ in range(0, tP["nrc"]):
            for _ in range(0, tP["nrcv"]//tP["nrcvpi"]):
              if self.states.globalReadIncsUseVgpr:
                imod.add(VAddCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    src1=vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], 2*loopIdx)), \
                    comment="gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar)))
                imod.add(VAddCCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    src1=vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], 2*loopIdx)), \
                    src2=VCC(), \
                    comment="gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar)))
              else:
                imod.add(VAddCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    src1=sgpr("GlobalReadIncs%s+%u"%(tP["tensorChar"], loopIdx)), \
                    comment="gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar)))
                imod.add(VAddCCOU32( \
                    dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    src1=0,
                    src2=VCC(), \
                    comment="gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar)))
              graIdx += self.states.rpga
      #module.add(dump(vgpr("GlobalReadAddrA+0")))
      #module.add(dump(vgpr("GlobalReadAddrA+1")))
      #module.add(SEndpgm())

  def globalReadIncrementAB(self, kernel, tPA, tPB, loopIdx, prefetchIndex):
    imod = Module("globalReadIncrementAB%s")

    incCodeA = imod.add(Module("globalReadIncrementA"))
    incCodeB = imod.add(Module("globalReadIncrementB"))

    self.globalReadIncrement(kernel, incCodeA, loopIdx, tPA, prefetchIndex)
    self.globalReadIncrement(kernel, incCodeB, loopIdx, tPB, prefetchIndex)
    return imod

  ##############################################################################
  # Global Read:
  # globalReadGuardK is called for loads in the tail loop
  # Must ensure each load is in bounds - either using buffer bounds
  # or exec-mask checks.
  ##############################################################################
  def globalReadGuardK(self, kernel, tP, vregSetIdx):
    module = Module("globalReadGuardK")
    tc = tP["tensorChar"]
    problemType = self.states.kernel["ProblemType"]
    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth

    ########################################
    # Calculate Max Addr
    ########################################

    if not kernel["BufferLoad"]:
      with self.allocTmpSgpr(2) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        maxAddrSgpr = tmpSgpr
        module.addComment0("flat addressing - max read address = size[n] * stride[n-1]")
        dim = len(tP["ia"])-1 # dim
        sizeIdx = tP["ia"][dim]
        sizeIdxIsSum = sizeIdx in kernel["ProblemType"]["IndicesSummation"]
        if sizeIdxIsSum:
          sizeIdx -= kernel["ProblemType"]["NumIndicesC"]
        # TODO-multiply by largest stride
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(maxAddrSgpr+0), sgpr(maxAddrSgpr+1),  \
                    sgpr("Sizes%s+%u"%("Sum" if sizeIdxIsSum else "Free", sizeIdx)),  \
                    sgpr("Stride%s%s"%(tc, self.states.indexChars[tP['ia'][-1]])), \
                    "64b tensor%s size in elements"%tc))
        module.add(SLShiftLeftB64(dst=sgpr(maxAddrSgpr,2), src=sgpr(maxAddrSgpr,2), \
          shiftHex=hex(log2(tP["bpe"])), comment="<- tensor%s size in bytes"%tc))

        module.add(SAddU32(
            dst=sgpr(maxAddrSgpr+0), \
            src0=sgpr(self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"]), \
            src1=sgpr(maxAddrSgpr+0), \
            comment="prepend address lower"))
        module.add(SAddCU32(
            dst=sgpr(maxAddrSgpr+1), \
            src0=sgpr((self.sgprs["AddressA"] if tP["isA"] else self.sgprs["AddressB"])+1), \
            src1=sgpr(maxAddrSgpr+1), \
            comment="prepend address upper"))
        # sgpr->vgpr
        maxAddrVgpr = self.vgprPool.checkOutAligned(2, 2, "maxAddrVgpr")
        module.add(VMovB32(dst=vgpr(maxAddrVgpr+0), src=sgpr(maxAddrSgpr+0), comment="sgpr->vgpr"))
        module.add(VMovB32(dst=vgpr(maxAddrVgpr+1), src=sgpr(maxAddrSgpr+1), comment="sgpr->vgpr"))

        # full exec mask
        fullExec = tmpSgpr
        sgprCnt = self.states.laneSGPRCount
        waveSize = kernel["WavefrontSize"]
        activeMask = "0xFFFFFFFF" if (waveSize == 32) else "0xFFFFFFFFFFFFFFFF"
        SMovBX     = SMovB32 if (waveSize == 32) else SMovB64
        module.add(SMovBX(dst=sgpr(fullExec,sgprCnt), src=activeMask, comment="to restore all threads active"))
        bpeVgpr = self.vgprPool.checkOut(1, "bpeVgpr")
        module.add(VMovB32(dst=vgpr(bpeVgpr), src=hex(tP["bpe"]), comment="bpe"))

        # can remove this?
        zeroVgpr = self.vgprPool.checkOut(1,"zeroVgpr")
        module.add(VMovB32(dst=vgpr(zeroVgpr), src=hex(0), comment="zero"))


    isGlc = True if tP["NonTemporal"]%2==1 else False
    isSlc = True if tP["NonTemporal"]//2==1 else False
    isLds = True if kernel["DirectToLds%s"%tc] else False

    directToLdsLoads = 0
    prevLdsOffset    = 0

    instOffset = 0
    loopCnt = -1

    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
            i = sPara + (tP["nrcv"] // tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            loopCnt += 1
            graIdx = i * self.states.rpgo if kernel["BufferLoad"] else i * self.states.rpga
            g2lIdx = i * loadWidth

            destVgprHi = None
            dataIsI8 = False
            packInt8Code = None
            eccOffset = 0

            instOffsetInc = 0 # increment value for instOffset. Need to apply after r loop

            r = 0
            numLoadVectorComp = loadWidth*self.states.bpr//tP["bpe"]
            if kernel["ProblemType"]["DataType"].isDouble() and kernel["BufferLoad"]:
              # adjustment for dgemm + BufferLoad
              # use same buffer_load instruction for tail loop as out of tail loop
              # this is mandatory for DirectToLds case. Also, it improves tail loop performance.
              # so far, limit to double only
              numLoadVectorComp = numLoadVectorComp // kernel["GlobalLoadVectorWidth%c"%tc]

            int8TempVgpr = numLoadVectorComp - 1
            # for each component in vector
            while r < numLoadVectorComp:
              numElementsPerLoad = 1
              if kernel["ProblemType"]["DataType"].isInt8():
                # TODO-Int8, Check this:
                # if tP["glvw"]>1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
                # # Pack two FP16 values into a single load dword x2
                #   numElementsPerLoad = 2
                # elif self.states.archCaps["HasEccHalf"]:
                #   destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi')

                # Check out 3 regs once , for component 1,2,3 (r = 1,2,3)
                if r == 1:
                  packInt8Code = Module()
                  destVgprHi = self.vgprPool.checkOut( int8TempVgpr , 'destVgprHi')
                dataIsI8 = True
                regIdx = r // 4
              elif kernel["ProblemType"]["DataType"].isHalf() or \
                 kernel["ProblemType"]["DataType"].isBFloat16():
                if tP["glvw"]>1 and kernel["AssertSummationElementMultiple"] % 2 == 0:
                # Pack two FP16 values into a single load dword x2
                  numElementsPerLoad = 2
                elif self.states.archCaps["HasEccHalf"]:
                  # In some cards, loading half types into register will zero out
                  # the other half. Therefore we need to load into a separate register
                  # then pack 2 registers into one
                  if (tP["localWriteInstruction"].blockWidth == 0.5) and (r%2 == 0):
                    numVgprG2L = self.states.a.numVgprG2L if tc == 'A' else self.states.b.numVgprG2L
                    eccOffset = _getEccOffset(tP["globalReadInstruction"].totalWidth, bpr=self.states.bpr, bpe=tP["bpe"], \
                      glvw=tP["glvw"], idx=loopCnt, numVgprG2L=numVgprG2L)
                  else:
                    destVgprHi = self.vgprPool.checkOut(1, 'destVgprHi')

                regIdx = r // 2
              elif kernel["ProblemType"]["DataType"].isInt8x4() or \
                   kernel["ProblemType"]["DataType"].isSingle():
                regIdx = r
              elif kernel["ProblemType"]["DataType"].isDouble():
                numElementsPerLoad = kernel["GlobalLoadVectorWidth%c"%tc] # adjust numElementsPerLoad for DGEMM
                regIdx = r*2
              elif kernel["ProblemType"]["DataType"].isSingleComplex():
                regIdx = r*2
              elif kernel["ProblemType"]["DataType"].isDoubleComplex() :
                regIdx = r*4
              else:
                printWarning("DataType unsupported")
              module.addComment0("g2l=%u, load component %u"%(g2lIdx, r))

              offset = 0

              if kernel["BufferLoad"]:
                # Use buffer limit to stay in-bounds - the limit was set to edge when SRD initialized
                # and each increment of SRD base in the unroll loop does a corresponding decrement
                # of the srd limit - so base+limit stays constant and also points at maximum
                # element that should be accessed.
                if kernel["_UseSgprForGRO"]:
                  offsetVgpr = "GlobalReadOffset%s+0"%(tc)
                else:
                  offsetVgpr = "GlobalReadOffset%s+%u"%(tc, graIdx)

                # Vgpr for GRO
                if not kernel["_UseSgprForGRO"]:
                  soffset = "0"
                # instruction offset with Sgpr for GRO
                elif kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx))
                # Sgpr for GRO
                else:
                  soffset = "0" if graIdx == 0 else sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

                unrollMirrorWithSoffset = kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx] in problemType["MirrorDims%s"%tc] and soffset != "0"
                # ScalarGlobalReadOffset should be negative value with unroll mirroring.
                # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
                if unrollMirrorWithSoffset:
                  codeMod = Module("mirrorIdx%u"%loopCnt)
                  codeMod.add(VSubU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset, comment="mirror unroll: GRO=GRO-SGRO, soffset=0"))
                  module.add(codeMod)
                  soffset_prev = soffset
                  soffset = "0"

                if kernel["DirectToLds%s"%tc]:
                  # need to increment ldsInc only once per each loopCnt
                  # this is pre count up, so increment it at r == 0
                  if r == 0:
                    ldsInc = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]
                  else:
                    ldsInc = 0
                  if kernel["LdsBlockSizePerPad%s"%tc] != 0:
                    ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
                  else:
                    padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
                    ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]
                  #print("ldsInc", ldsInc)
                  #print("GlobalLoadVectorWidth", kernel["GlobalLoadVectorWidth%c"%tc])
                  #print("bpr", self.states.bpr)
                  if kernel["UseInstOffsetForGRO"]:
                    # buffer_load only support 12 bit instruction offset
                    # we have to increase m0 if offset is larger thant 12 bits
                    if instOffset >= self.buff_load_inst_offset_max:
                      inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                      module.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=inc, comment="Move LDS write address to next base" ))
                      instOffset -= inc
                  elif directToLdsLoads != 0 and ldsInc > 0:
                      if tP["nrc"] > 1:
                        # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                        divisorName = tP["lvc"]
                        divisor = kernel[divisorName]
                        # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                        wSize = max(self.states.kernel["WavefrontSize"], divisor)
                        lscaOffset = para * wSize * tP["bpe"] * tP["glvw"]
                        ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                        ldsInc = ldsOffset - prevLdsOffset
                        prevLdsOffset = ldsOffset
                      module.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=ldsInc, comment="Move LDS write address to next line" ))

                  destVgpr=0
                elif kernel["DirectToVgpr%s"%tc]:
                  numVgprG2L = self.states.a.numVgprG2L if tP["isA"] else self.states.b.numVgprG2L
                  numVgprPerBlock = numVgprG2L // 2 # numVgprG2L is doubled for DirectToVgpr
                  idx = g2lIdx + vregSetIdx * numVgprPerBlock
                  destVgpr="G2L%s+%u+%u"%(tc, idx, regIdx)
                else:
                  destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx+eccOffset)

                offset = r * tP["bpe"] + instOffset
                hi8 = 0
                hi16 = 0
                comment = "load one buffer value"
                if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
                  if numElementsPerLoad==2:
                    # Pack two FP16 values into a single load dword x2
                    r += 1 # skip next element since we loaded 2X here
                    comment = "load packed 2X half buffer value"
                  elif not kernel["DirectToLds%s"%tc]:
                    hi16=loopCnt%2 if tP["glvw"]==1 else r%2
                    comment="load one buffer value"

                if kernel["ProblemType"]["DataType"].isInt8():
                  # TODO-Int8, Check this:
                  # if numElementsPerLoad==2:
                  #   # Pack two FP16 values into a single load dword x2
                  #   r += 1 # skip next element since we loaded 2X here
                  #   comment = "load packed 2X half buffer value"
                  if not kernel["DirectToLds%s"%tc]:
                    hi8  = (loopCnt%4) %2 if tP["glvw"]==1 else (r%4) %2
                    hi16 = (loopCnt%4)//2 if tP["glvw"]==1 else (r%4)//2
                    comment="load one buffer value"

                bpl = numElementsPerLoad*self.states.bpeAB # bytesPerLoad

                # if hi8=1 or hi16=1 (component 1,2,3 for int8) or (component 1 for half), use the temp destVgprHi
                # but only when hi16=1 we use the _d16_hi version instruction, see the below visualized int8 comment
                loadVgpr = destVgprHi if ((hi16 or hi8) and destVgprHi != None) else destVgpr
                if kernel["ProblemType"]["DataType"].isInt8() and (not self.states.archCaps["HasEccHalf"]):
                  module.add(VMovB32(dst=vgpr(loadVgpr), src=0, comment="set to zero to avoid unexpected value"))
                module.add(self.chooseGlobalRead(True, \
                          bpl, destVgpr=loadVgpr, \
                          addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                          soffset=soffset, offset=offset, \
                          glc=isGlc, slc=isSlc, lds=isLds, \
                          hi16=hi16, \
                          comment=comment))

                if unrollMirrorWithSoffset:
                  codeMod = Module("mirrorIdx%u"%loopCnt)
                  codeMod.add(VAddU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset_prev, comment="mirror unroll: restore GRO=GRO+SGRO"))
                  module.add(codeMod)

                if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  instOffsetInc += ldsInc
                # print("  bpl={}, destVgpr={}, soffset={}, offset={}, hi16={}".format(bpl, destVgpr, soffset, offset, hi16))

              else: # Not buffer load, ie 'flat' load
                # mask if current address if in bounds
                module.add(VCmpXLtU64(dst=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2), \
                    src1=vgpr(maxAddrVgpr,2), \
                    comment="addr < maxAddr"))
                hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and r%2==1
                destVgpr="G2L%s+%u+%u"%(tc, g2lIdx, regIdx)
                # load one element from address
                module.add(self.chooseGlobalRead(False, \
                          self.states.bpeAB, destVgpr=destVgprHi if (hi16 and destVgprHi != None) else destVgpr, \
                          addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                          soffset=0, offset=0, \
                          glc=isGlc, slc=isSlc, lds=isLds, \
                          hi16=hi16, \
                          comment="load one flat value"))

                # restore full exec mask
                SOrSaveExecBX = SOrSaveExecB32 if self.states.kernel["WavefrontSize"] == 32 else SOrSaveExecB64
                module.add(SOrSaveExecBX(dst=VCC(), src=sgpr(fullExec,self.states.laneSGPRCount), comment="all threads active"))

                # increment address by 1 element (BPE)
                module.add(VAddCOU32(
                    dst=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                    src1=vgpr(bpeVgpr), comment="gra += 1 (lower)"))
                module.add(VAddCCOU32(
                    dst=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    dst1=VCC(), \
                    src0=vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                    src1=vgpr(zeroVgpr), \
                    src2=VCC(), comment="gra += 1 (upper)"))

              # int8 byte:
              # |--------|--------|--------|---V0---|, r = 0, hi8=0, hi16=0, load d16
              # |--------|--------|--------|---V1---|, r = 1, hi8=1, hi16=0, load d16
              # |--------|---V2---|--------|--------|, r = 2, hi8=0, hi16=1, load d16_hi
              # |--------|---V3---|--------|--------|, r = 3, hi8=1, hi16=1, load d16_hi
              # V1, V3 -> shift left 8 bits, or 4 regs (pack)
              # DestV0|=(V1 << 8), DestV0|= V2, DestV0|=(V3 << 8)
              # Int8 (byte)
              if dataIsI8 and (destVgprHi != None):
                # hi8  -> r = 1,3
                # hi16 -> r = 2,3
                if hi8 or hi16:
                  # r = 1,2,3, vmcnt needed for one packing
                  packInt8Code.add(SWaitCnt(vmcnt=(int8TempVgpr-r), comment=""))
                if hi8:
                  # r = 1,3,   shift needed
                  packInt8Code.add(VLShiftLeftB32(dst=vgpr(destVgprHi), shiftHex=hex(0x8), src=vgpr(destVgprHi), comment="shift left to higher 8 bits"))
                if hi8 or hi16:
                  # r = 1,2,3, packing
                  packInt8Code.add(VOrB32(dst=vgpr(destVgpr), src0=vgpr(destVgpr), src1=vgpr(destVgprHi), comment="pack a sub 8-bit with dest"))
                destVgprHi += 1

              # Half
              elif destVgprHi != None and r % 2 == 1:
                module.add(SWaitCnt(vmcnt=0, comment=""))
                module.add(VOrB32(dst=vgpr(destVgpr), src0=vgpr(destVgpr), src1=vgpr(destVgprHi), comment="HasEccHalf: pack"))

              # For half (bf16). Note: for int8, we will checkin after loading all components
              if (destVgprHi != None) and (not dataIsI8):
                self.vgprPool.checkIn(destVgprHi)
                destVgprHi = None

              r += 1 # next component (for half, byte)

            # end R loop

            instOffset += instOffsetInc # add increment value for instOffset. Need to apply after r loop
            # increment once per r loop (at the end)
            directToLdsLoads+=1

            # for int8:
            # we do the 3 packs, and checking the 3 extra vgprs after loading all components
            if dataIsI8:
              assert packInt8Code != None and destVgprHi != None
              module.add(packInt8Code)
              self.vgprPool.checkIn(destVgprHi - int8TempVgpr)
              destVgprHi = None

    if self.db["ConservativeWaitCnt"] & 0x1:
        module.add(SBarrier(comment="debug"))
        module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment=""))
        module.add(SBarrier(comment="debug"))
        #module.add(self.getCmpAssert(self.asmAssert.lt, vgpr("Serial"), 64)) # examine second wavefront

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]]:
      module.add(SMovB32(dst=mgpr(0), src=hex(kernel["LdsNumElements"] * tP["bpe"]), \
          comment="Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"])))

    if not kernel["BufferLoad"]:
      self.vgprPool.checkIn(maxAddrVgpr)
      self.vgprPool.checkIn(bpeVgpr)
      self.vgprPool.checkIn(zeroVgpr)

    return module

  ##############################################################################
  # DirectToLds M0 update: Do It A/B
  ##############################################################################
  def directToLdsM0Update(self, kernel, mode, tP, usePlaceHolder=False):
    tc = tP["tensorChar"]
    imod = Module("directToLdsM0Update%s_%u"%(tc,mode))
    DtldsModule = imod.add(Module("dtls_offset%s"%tP["tensorChar"]))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod
    if kernel["DirectToLds%s"%tc]:
      # DirectToLds only enabled for TLU=1 cases, where the registers are directly copied into LDS
      # for cases both A&B are DTLS, updating m0 for each GlobalRead requires instruction schedule
      # along with global reads
      assert (kernel["LocalWriteUseSgpr%s"%tc])
      if kernel["ExpandPointerSwap"]:
        DtldsModule.add(SAddU32(dst=mgpr(0), src0=sgpr("LocalWriteAddr%s"%tc), \
                      src1=tP["localWriteSwapByteOffset"], comment="m0 <- LDS write address"))
      else:
        DtldsModule.add(SMovB32(dst=mgpr(0), src=sgpr("LocalWriteAddr%s"%tc), comment="m0 <- LDS write address"))

      # PrefetchGlobalRead=2 case, generate local read wait for DirectToLds
      if kernel["PrefetchGlobalRead"]==2:
        # do not generate local read wait for PGR=2
        DtldsModule.addComment0("before DirectToLds load, ensure prior ds_reads have finished")
        DtldsModule.add(SWaitCnt(lgkmcnt=0, comment=""))
        if not kernel["NoLdsWriteCode"]:
          if usePlaceHolder:
            waitStr = Holder(idx=0)
          else:
            waitStr = 0
          DtldsModule.add(SWaitCnt(vmcnt=waitStr, comment=""))
        DtldsModule.add(SBarrier())

    return imod

  ##############################################################################
  # Global Read: Do It A/B
  ##############################################################################
  def globalReadDo(self, kernel, mode, tP, vregSetIdx=0):
    tc = tP["tensorChar"]
    problemType = self.states.kernel["ProblemType"]
    imod = StructuredModule("globalReadDo%s_%u"%(tc,mode))
    if not self.do["GlobalRead%s"%tP["tensorChar"]]: return imod

    # sizeK % LOCAL_DEPTHU
    guardK = (mode==2)

    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth # load width in elements?
    bpl = self.states.bpeAB * tP["glvw"] # bytes per load
    instOffset = 0

    loopIdx = self.states.unrollIdx # TODO - does this handle multiple summation indices?
    if kernel["SuppressNoLoadLoop"]:
      if mode==1 and tP["isA"]:
        imod.header.add(SCmpEQI32(
              src0=self.loopCounter(kernel, loopIdx), \
              src1=1, \
              comment="%s"%"is this the last iteration"))
        imod.header.add(SCMovB32(
              dst=sgpr("SrdA+2"), src=0,
              comment="Set limit to 0 for last iteration"))
        imod.header.add(SCMovB32(
              dst=sgpr("SrdB+2"), src=0,
              comment="Set limit to 0 for last iteration"))

    # set the first tc for below wait code for DirectToLds
    # if DirectToVgprA is enabled, change the first to B
    tc1st = 'A'
    if kernel["DirectToVgprA"]:
      tc1st = 'B'

    if tc == tc1st and (kernel["DirectToLdsA"] or kernel["DirectToLdsB"]) and not kernel["PrefetchGlobalRead"]==2:
      # generate local read wait for DirectToLds except for PrefetchGlobalRead=2 (for PGR=2, generate wait after m0 value setting)
      imod.header.addComment0("before DirectToLds load, ensure prior ds_reads have finished")
      if (kernel["DirectToVgprA"] or kernel["DirectToVgprB"]): # do not generate sync here if DirectToVgpr is enabled
        imod.header.add(SWaitCnt(lgkmcnt=0, comment=""))
      else:
        imod.header.add(self._syncThreads(kernel))


    if guardK:
      imod.middle.add(self.globalReadGuardK(kernel, tP, vregSetIdx))
      return imod

    # else not-guardK below:

    isGlc = True if tP["NonTemporal"]%2==1 else False
    isSlc = True if tP["NonTemporal"]//2==1 else False
    isLds = True if kernel["DirectToLds%s"%tc] else False

    directToLdsLoads = 0
    instOffset       = 0
    prevLdsOffset    = 0

    loopCnt = -1
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]//tP["nrcvpi"]):
            i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp))
            loopCnt += 1
            graIdx = i * self.states.rpgo if kernel["BufferLoad"] else i * self.states.rpga
            g2lIdx = i * loadWidth
            # Each load may contains a small bundle of instructions, package them together in loadModule:
            loadModule = Module("load%u"%loopCnt)
            imod.middle.add(loadModule)

            if self.states.archCaps["HasEccHalf"]:
              numVgprG2L = self.states.a.numVgprG2L if tc == 'A' else self.states.b.numVgprG2L
              eccOffset = _getEccOffset(loadWidth, bpr=self.states.bpr, bpe=tP["bpe"], \
                glvw=tP["glvw"], idx=i, numVgprG2L=numVgprG2L)
            else:
              eccOffset = 0

            if kernel["BufferLoad"]:
              if kernel["_UseSgprForGRO"]:
                offsetVgpr= "GlobalReadOffset%s+0"%(tc)
              else:
                offsetVgpr= "GlobalReadOffset%s+%u"%(tc, graIdx)

              # vgpr for GRO
              if not kernel["_UseSgprForGRO"]:
                soffset = "0"
              # instruction offset with Sgpr for GRO
              elif kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                soffset = sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx))
              # Sgpr for GRO
              else:
                soffset = "0" if graIdx == 0 else sgpr("ScalarGlobalReadOffset%s+%u"%(tc, graIdx-1))

              unrollMirrorWithSoffset = kernel["ProblemType"]["IndicesSummation"][self.states.unrollIdx] in problemType["MirrorDims%s"%tc] and soffset != "0"
              # ScalarGlobalReadOffset should be negative value with unroll mirroring.
              # However, buffer_load uses soffset as uint value, so GRO - SGRO, SGRO = 0
              if unrollMirrorWithSoffset:
                codeMod = Module("mirrorIdx%u"%loopCnt)
                codeMod.add(VSubU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset, comment="mirror unroll: GRO=GRO-SGRO, soffset=0"))
                loadModule.add(codeMod)
                soffset_prev = soffset
                soffset = "0"

              if kernel["DirectToLds%s"%tc]:
                # use bpe with GlobalLoadVectorWidth
                ldsInc = (self.states.kernel["WavefrontSize"] * kernel["GlobalLoadVectorWidth%c"%tc] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"] * kernel["GlobalLoadVectorWidth%c"%tc]) * tP["bpe"]
                if kernel["LdsBlockSizePerPad%s"%tc] != 0:
                  ldsInc += (ldsInc // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
                else:
                  padInterval = (self.states.kernel["WavefrontSize"] if kernel["WaveSeparateGlobalRead%c"%tc] else kernel["NumThreads"]) * self.states.bpr
                  ldsInc += (ldsInc // padInterval) * kernel["LdsPad%s"%tc] * tP["bpe"]

                if kernel["UseInstOffsetForGRO"]:
                  # buffer_load only support 12 bit instruction offset
                  # we have to increase m0 if offset is larger thant 12 bits
                  if instOffset >= self.buff_load_inst_offset_max:
                    inc = (instOffset // self.buff_load_inst_offset_max) * self.buff_load_inst_offset_max
                    loadModule.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=inc, comment="Move LDS write address to next base" ))
                    instOffset -= inc
                elif directToLdsLoads != 0:
                  # m0 offset conversion (only for UseInstOffsetForGRO == 0)
                  # in tP["glvw"] == 1 and tP["nrc"] > 1 case, only m0 offset conversion is necessary. row and column index conversion is not necessary.
                  if tP["nrc"] > 1:
                    # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                    divisorName = tP["lvc"]
                    divisor = kernel[divisorName]
                    # DirectToLds + NumLoadsCoalesced>1 case, need to adjust m0 increment value to store values to correct location in LDS
                    wSize = max(self.states.kernel["WavefrontSize"], divisor)
                    lscaOffset = para * wSize * tP["bpe"] * tP["glvw"]
                    ldsOffset = ldsInc * tP["nrc"] * (sPerp + tP["nrpv"] * perp) + lscaOffset
                    ldsInc = ldsOffset - prevLdsOffset
                    prevLdsOffset = ldsOffset
                  loadModule.add(SAddU32(dst=mgpr(0), src0=mgpr(0), src1=ldsInc, comment="Move LDS write address to next line" ))
                directToLdsLoads+=1
                destVgpr=0
              elif kernel["DirectToVgpr%s"%tc]:
                # DirectToVgpr case. Need to toggle destination vreg set and adjust instOffset
                destVgpr="G2L%s%u+%u"%(tc, vregSetIdx, g2lIdx)
              else:
                destVgpr="G2L%s+%u"%(tc, (g2lIdx+eccOffset))

              # TODO: is it possible to load only hi16 when no in tail? (need to check INT8 too)
              loadModule.add( self.chooseGlobalRead(kernel["BufferLoad"], \
                        bpl, destVgpr=destVgpr, \
                        addr0=vgpr(offsetVgpr), addr1=sgpr("Srd%s"%tc, 4), \
                        soffset=soffset, offset=instOffset, \
                        glc=isGlc, slc=isSlc, lds=isLds, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp)))

              if unrollMirrorWithSoffset:
                codeMod = Module("mirrorIdx%u"%loopCnt)
                codeMod.add(VAddU32(dst=vgpr(offsetVgpr), src0=vgpr(offsetVgpr), src1=soffset_prev, comment="mirror unroll: restore GRO=GRO+SGRO"))
                loadModule.add(codeMod)

              if kernel["DirectToLds%s"%tc] and kernel["UseInstOffsetForGRO"]:
                  instOffset += ldsInc

              #print "IM=", type(imod.instList[-1]), imod.instList[-1],
            else: # not buffer load
              # load one element from address
              if kernel["DirectToVgpr%s"%tc]:
                # DirectToVgpr case. Need to toggle destination vreg set and adjust instOffset
                destVgpr="G2L%s%u+%u"%(tc, vregSetIdx, g2lIdx)
              else:
                destVgpr="G2L%s+%u"%(tc, g2lIdx)
              loadModule.add( self.chooseGlobalRead(False, \
                        bpl, \
                        destVgpr=destVgpr, \
                        addr0=vgpr("GlobalReadAddr%s+%u"%(tc,graIdx),2), addr1="", \
                        soffset=0, offset=0, \
                        glc=isGlc, slc=isSlc, lds=isLds, \
                        hi16=(kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) and loopCnt%2==1, \
                        comment="G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp )))

    if self.db["ConservativeWaitCnt"] & 0x1:
        imod.footer.add(SBarrier(comment="debug"))
        imod.footer.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment="conservative wait"))
        imod.footer.add(SBarrier(comment="debug"))
        #module.add(self.getCmpAssert(self.asmAssert.lt, vgpr("Serial"), 64)) # examine second wavefront

    # TODO - can remove one of these m0 restores if A and B both TLU
    if kernel["DirectToLds%s"%tP["tensorChar"]] and not (mode == 1 and kernel["PrefetchGlobalRead"]==2):
      dst = mgpr(0)
      src = hex(kernel["LdsNumElements"] * tP["bpe"])
      comment = "Restore LDS clamp at %u bytes"%(kernel["LdsNumElements"] * tP["bpe"])
      # PGR=2 case, footer is located before global read. To avoid setting clamp before global read, store lds clamp code in middle
      if kernel["PrefetchGlobalRead"] == 2:
        imod.middle.add(SMovB32(dst=dst, src=src, comment=comment))
      else:
        imod.footer.add(SMovB32(dst=dst, src=src, comment=comment))

    return imod

  ##############################################################################
  # Local Write: Swap Offsets A/B
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, internalPointerSwap, tP):
    if not self.do["LocalWrite"]: return Module("localWriteSwapOffsets (No local write)")
    if kernel["1LDSBuffer"]: return Module("localWriteSwapOffsets (Empty)")
    module = Module("localWriteSwapOffsets")
    tc = tP["tensorChar"]
    #fixme-iui  need to use wrapping increment for double or triple buffering:
    if internalPointerSwap:
      tP["localWriteSwapByteOffset"] = 0 if tP["localWriteSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
      module.addComment1("(EPS=1) local write swap internal offset -> %u" % tP["localWriteSwapByteOffset"])
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        module.add(SXorB32(
            dst=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            src0=hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
            src1=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            comment="swap Red Blk SGPR"))
      elif not kernel["DirectToVgpr%s"%tc]: # no local write code if DirectToVgpr is enabled
        numLwa = self.states.a.numVgprLocalWriteAddr if tP["isA"] else self.states.b.numVgprLocalWriteAddr
        for i in range(0,numLwa):
          module.add(VXorB32(
              dst=vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              src0=hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
              src1=vgpr("LocalWriteAddr%s+%u"%(tc,i)), \
              comment="swap Red Blk"))
    return module

  ##############################################################################
  # Local Write: Reset Offsets A/B
  # used for global-read + tail-loop to reset to writing in red
  ##############################################################################
  def localWriteResetOffsets(self, kernel, internalPointerSwap, tP):
    tc = tP["tensorChar"]
    if not self.do["LocalWrite"]: return Module("localWriteResetOffsets (no local write)")
    if kernel["1LDSBuffer"] or kernel["DirectToVgpr%s"%tc]: # no local write code if DirectToVgpr is enabled
      return Module("localWriteResetOffsets (Empty)")
    module = Module("localWriteResetOffsets")
    resetMask = hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1 | self.consts.ldsOOB)
    if internalPointerSwap:
      tP["localWriteSwapByteOffset"] = 0
    else:
      if kernel["LocalWriteUseSgpr%s"%tc]:
        module.add(SAndB32(
            dst=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            src0=resetMask, \
            src1=sgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            comment="reset to Red"))
      else:
        module.add(VAndB32(
            dst=vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            src0=resetMask, \
            src1=vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
            comment="reset to Red"))
    return module

  ##############################################################################
  # Calculate offset to use for LDS write
  # Intro:
  #   Each WI has a 2D tile index (coal, perp).
  #     - Code above computes global mem address by scaling one dim by the
  #       lda and adding the other.
  #     - Here we compute a linear LDS offset by scaling one dim by the MT
  #       dim and adding the other.
  #   Result is we map a tile from global memory into LDS.  Consecutive LDS
  #   locations contain elements from different summation 'rows' - therefore
  #   loading a row of LDS will feed computations for different C tile indices.
  # Notes:
  #   Total load insts is nrc * nrp which load the macro-tile.
  #   Par and coalesced are ~synonyms referring to same dimension
  #   Either nrpv or nrvc must be 1 - can't have vectors in both dimensions.
  #     Thus either sPerp or sPara is 0.
  # Inputs:
  #   perp : index of the load in perp dimension (0...nrp)
  #   par  : index of the load in the para dim (0...nrc)
  #   sPerp : component index of the perp vector (0...nrpv)
  #   sPara : component index of the par vector (0...nrcv)
  # Outputs:
  #   offsetBytes : Offset in bytes for the _ds_store instruction
  #   i : i-th instruction
  #   comment : Comment with the text version of the formula
  #############################################################################
  def calculateLdsWriteOffset(self, perp, para, sPerp, sPara, kernel, tP):
    tc = tP["tensorChar"]
    mask = 0
    #print "tc ", tc, " perp ", perp, " para ", para, " sPerp ", sPerp, " sPara ", sPara
    lscaOffset = para * kernel[tP["lsc"]]
    perp_masked = perp
    perp_rem = 0
    lspaOffset = perp_masked * kernel[tP["lsp"]]
    rem = 0

    # Add component offset to interleave from different regs
    # and compute mysterious "i"
    assert(sPerp==0 or sPara==0)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset += sPerp & mask
      lscaOffset += sPara
      rem = (sPerp & ~mask)
      i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para + tP["nrc"] * (sPerp + tP["nrpv"] * perp_masked))
      #print "nrcv ", tP["nrcv"], " nrcvpi ", tP["nrcvpi"], " nrc ", tP["nrc"], " nrpv ", tP["nrpv"]
    else:
      lscaOffset += sPara
      lspaOffset += sPerp
      rem = 0
      i = sPara + (tP["nrcv"]//tP["nrcvpi"]) * (para * tP["glvw"] + tP["nrc"] * (sPerp + tP["glvw"] * tP["nrpv"] * perp ))

    #if not tP["tlu"]:
    #  tmp = sPara
    #  sPara = sPerp
    #  sPerp = tmp
    # print("0lspaOffset", lspaOffset)
    # print("0lscaOffset", lscaOffset)

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
    lds_stride = (kernel["_DepthULds"] + LdsPad) if kernel["UnrollMajorLDS%s" % tP["tensorChar"]] \
            else (kernel[tP["mt"]] + LdsPad)

    if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
      lspaOffset *= lds_stride
      lspaOffset += rem + perp_rem
    else:
      lscaOffset *= lds_stride
      lscaOffset += rem

    # print("1lspaOffset", lspaOffset)
    # print("1lscaOffset", lscaOffset)
    #if tP["tlu"]:
    #  lspaOffset *= tP["glvw"]
    #  lscaOffset *= tP["glvw"]

    # print("2lspaOffset", lspaOffset)
    # print("2lscaOffset", lscaOffset)
    offsetElements = (lspaOffset + lscaOffset)
    # print("offsetElements", offsetElements)
    offsetBytes = offsetElements*tP["bpe"]

    if kernel["LdsBlockSizePerPad%s"%tc] != 0 and kernel["LdsPad%s"%tc] != 0:
      offsetBytes = offsetBytes + (offsetBytes // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]

    offsetBytes += tP["localWriteSwapByteOffset"]

    #print("offsetBytes", offsetBytes)
    #print "offset", offset

    comment = "lwo%s_%u_%u_%u_%u = (%s%d*%s)" \
        % (tP["tensorChar"], \
        para, sPara, perp, sPerp, \
        (("%u + "%sPara) if tP["wtc"] else ""), \
        para, tP["lsc"] )
    if not tP["tlu"]:
      comment += "*(MT%s+PAD)" % (tP["tileChar"])
    comment += " + (%d*%s)" % (perp, tP["lsp"])
    if tP["tlu"]:
      comment += "(*MT%s+PAD)" % (tP["tileChar"])
    comment += " = %u" % (offsetBytes)

    return (offsetBytes, i, comment)

  def recalcLocalWriteAddresses(self, kernel, tP, uDu):

    tc = tP["tensorChar"]

    module = Module("recalcLocalWriteAddresses")
    module.addComment1("recalculate LocalWriteAddr{}".format(tc))

    lwvw = getattr(self, "localWriteWidth{}".format(tc))
    newInstIdx = self.selectMemoryInstruction("LocalWrite", lwvw*kernel["DepthULdsDivisor"], \
        False, \
        tP["localWrite2Coalesced"], tP["localWrite2Perpendicular"],
        [tP["localWriteStrideTile"], tP["localWriteStrideUnroll"]] )
    tP["localWriteInstruction"] = self.memoryInstructions["LocalWrite"][newInstIdx]

    # global read tile assignment
    module.add(self.graTileAssignment(kernel, tP))
    # global read tile offsets
    module.add(self.graTileOffsets(kernel, tP))
    # global read unroll offsets
    module.add(self.graUnrollOffsets(kernel, tP))
    # still needed for vgpr resource management
    # intentionally not emitting code
    self.graFinalOffsets(kernel, tP)

    # local write tile assignments
    module.add(self.lwaTileAssignment(tP))
    # local write unroll assignments
    module.add(self.lwaUnrollAssignment(kernel, tP))
    # local write local write first offsets
    module.add(self.lwaFirstOffset(kernel, tP, uDu))

    return module

  def recalcLocalReadAddressesAB(self, kernel, tPA, tPB):
    imod = Module()

    if self.states.inTailLoop:
      # it do 1 iteration each loop in tail loop, and is no use to wider local read next iteration.
      # In 1 block MI, it remap localReadAddr in order to let each thread wider local read continuous k
      # this decrease performance since it require more loop to handle continuous k in each thread.
      # recalculate localReadAddr to cancel wider local read in tail loop
      # TODO: If DepthULdsDivisor>1, local read addr is incremented for each K the loop iterates, which
      # upon second sub-loop needs to be reset to its original value. Backing up local read address would
      # be nicer than recomputing them
      if kernel.enabledSplitLDS or ((self.states.numReadsIterCoalescedA > 1 or self.states.numReadsIterCoalescedB > 1) and kernel["MatrixInstB"] == 1): #and tP["isB"]:
        self.states.numReadsIterCoalescedA = 1
        self.states.numReadsIterCoalescedB = 1
        self.states.lrvwA = kernel["MIInputPerThread"]
        self.states.lrvwB = kernel["MIInputPerThread"]

        imod.add(self.lraTileAssignment(kernel, tPA, tPB))
        imod.add(self.lraFinalOffset(kernel, tPA))
        imod.add(self.lraDeclareAddresses(kernel, tPA))
        imod.add(self.lraFinalOffset(kernel, tPB))
        imod.add(self.lraDeclareAddresses(kernel, tPB))
        localRead2Perpendicular = False
        instructions = self.memoryInstructions

        localReadWidth = tPA["bpe"] / self.states.bpr
        if kernel["UnrollMajorLDSA"]:
          localReadWidth = (kernel["MIInputPerThread"] * tPA["bpe"]) // self.states.bpr
        localReadInstructionIdxA = \
          self.selectMemoryInstruction("LocalRead", localReadWidth, \
          False, \
          tPA["localRead2Coalesced"], localRead2Perpendicular,
          [tPB["localReadStrideCoalesced"]] )
        tPA["localReadInstruction"] = instructions["LocalRead"][localReadInstructionIdxA]

        localReadWidth = tPB["bpe"] / self.states.bpr
        if kernel["UnrollMajorLDSB"]:
          localReadWidth = (kernel["MIInputPerThread"] * tPB["bpe"]) // self.states.bpr
        localReadInstructionIdxB = \
          self.selectMemoryInstruction("LocalRead", localReadWidth, \
          False, \
          tPB["localRead2Coalesced"], localRead2Perpendicular,
          [tPB["localReadStrideCoalesced"]] )
        tPB["localReadInstruction"] = instructions["LocalRead"][ \
          localReadInstructionIdxB]

    return imod

  ##############################################################################
  # Local Write in Prefetch Pass (PreLoop): Do It A/B
  ##############################################################################
  def preLoopLocalWriteDo(self, kernel, tPA, tPB):
    imod = Module()

    LWDoMod = imod.add(Module())
    LWDoA = self.localWriteDo(kernel, tPA)
    LWDoB = self.localWriteDo(kernel, tPB)
    LWDoMod.addComment1("local write a")
    LWDoMod.add(LWDoA)
    LWDoMod.addComment1("local write b")
    LWDoMod.add(LWDoB)
    return imod

  ##############################################################################
  # Local Write: Do It A/B
  # uDu: 'None' means to use fractional local write (where not all threads are active)
  #      when DepthULdsDivisor > 1
  ##############################################################################
  def localWriteDo(self, kernel, tP, uDu=0):
    if not self.do["LocalWrite"]: return "", -1

    tc = tP["tensorChar"]
    imod = Module()

    if (not kernel["DirectToLds%s"%tc]) and (not kernel["DirectToVgpr%s"%tc]):
      instruction = tP["localWriteInstruction"]
      numBlocks = instruction.numBlocks
      numOffsets = instruction.numOffsets
      blockWidth = instruction.blockWidth
      #offsetMultiplier = instruction.offsetMultiplier
      g2lIdx = 0
      #module.add(dump(vgpr("LocalWriteAddr%s"%tP["tensorChar"])))
      if 0:
        print("\nLocalWrite", tP["tensorChar"])
        print("tlu", tP["tlu"])
        print("lsc", kernel[tP["lsc"]])
        print("lsp", kernel[tP["lsp"]])
        print("wtc", tP["wtc"])
        print("nrc", tP["nrc"])
        print("nrp", tP["nrp"])
        print("nwcv", tP["nwcv"])
        print("nwpv", tP["nwpv"])
        print("nrcvpi", tP["nrcvpi"])
        print("nwcvpi", tP["nwcvpi"])

      tmpLocalWriteAddr = -1

      # using _ds_store_b8: need one more vgpr space to do lshr
      tmpVgprOffset = ((self.states.a.numVgprG2L if (tP['tensorChar'] == 'A') else self.states.b.numVgprG2L) / 2) if (blockWidth == 0.25) else 0

      # if transposing, positions of sPerp and sPara are transposed
      instructionCnt = -1
      fp16AltMap = {}
      g2lIdxDict = {}
      for perp in range(0, tP["nrp"]):
        instructionCnt += 1
        localWriteCode = imod.add(Module("LocalWrite%u perp=%d"%(instructionCnt,perp)))
        lwa = "LocalWriteAddr%s"%tc  # default

        for para in range(0, tP["nrc"]):
          if para>=1:
            localWriteCode = imod.add(Module("LocalWrite%u perp=%d para=%d"%(instructionCnt,perp,para)))

          for s in range(0, max(tP["nwcv"],tP["nwpv"])//tP["nwcvpi"]):
            sPerp = 0
            sPara = 0
            if tP["tlu"] != kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
              if tP["wtc"]:
                sPerp = s
            else:
              if tP["wtc"]:
                sPara = s

            #print("perp:{}/{} para:{}/{} sPerp:{} sPara:{}".format(perp,tP["nrp"],para,tP["nrc"],sPerp,sPara))
            (offset, i, comment) = self.calculateLdsWriteOffset(perp, para, sPerp, sPara, kernel, tP)

            if uDu is None:
              g2lIdx = int(i * blockWidth)
            else:
              # Example: DepthULdsDivisor=2
              # v0, v1, v2, v3 | v0, v1, v2, v3 | ... ----> unroll dim
              # -----Thd 0----- -----Thd 1-----   ...
              # 1st subloop writes v0,v1 to LDS
              # 2nd subloop writes v2,v3 to LDS
              g2lIdx = int((i * kernel["DepthULdsDivisor"] + uDu) * blockWidth)
              #print("uDu=%u, g2lIdx = %u, offset: %u"%(uDu, g2lIdx, offset))

            # If g2lIdx is already in the dict and blockWidth < 1, the data may
            # be packed into one register.
            instHi = 0
            if g2lIdx in g2lIdxDict:
              g2lIdxDict[g2lIdx] += 1
            else:
              g2lIdxDict[g2lIdx] = 0
            instHi = g2lIdxDict[g2lIdx]

            # TODO- INT8: check uDu
            if (blockWidth == 0.25) and ((s % 4) == 0):
                src = "G2L%s+%u" % (tc, g2lIdx)
                dst = "G2L%s+%u+%u" % (tc, tmpVgprOffset, g2lIdx)
                localWriteCode.add(VMovB32(dst=vgpr(dst), src=vgpr(src), comment="another VGPR storing lshr 8-bit value"))
                localWriteCode.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(0x8), src=vgpr(dst), comment="G2L Vpgr >> 8"))

            if self.states.archCaps["HasEccHalf"]:
              numVgprG2L = self.states.a.numVgprG2L if tc == 'A' else self.states.b.numVgprG2L
              eccOffset = _getEccOffset(tP["globalReadInstruction"].totalWidth, bpr=self.states.bpr, bpe=tP["bpe"], \
                glvw=tP["glvw"], idx=instHi, numVgprG2L=numVgprG2L)
            else:
              eccOffset = 0

            paramList = []
            for _ in range(0, numBlocks):
              if blockWidth == 1:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
              elif blockWidth == 0.25 and ((s % 2) == 1): # Int8, s = 1 or 3 (high8Bits)
                paramList.append(vgpr("G2L%s+%u+%u"%(tc, tmpVgprOffset, g2lIdx)))
              else:
                paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx + eccOffset), blockWidth))
              if self.db["ForceInputValue%s"%tc]:
                localWriteCode.add(VMovB32(dst=vgpr("G2L%s+%u"%(tc, g2lIdx)), src=self.db["ForceValue%s"%tc], comment="ForceInputValue"))
              if kernel["ProblemType"]["Fp16AltImpl"]:
                numIters = 1 if blockWidth <= 1 else blockWidth
                vgprTmp = self.vgprPool.checkOut(2)
                for iter in range(0, numIters):
                  f16Tobf16Idx = g2lIdx + iter
                  if f16Tobf16Idx in fp16AltMap:
                    fp16AltMap[f16Tobf16Idx] += 1
                    continue
                  fp16AltMap[f16Tobf16Idx] = 1
                  sdwa = SDWAModifiers(src0_sel=SelectBit.WORD_1)
                  localWriteCode.add(VCvtF16toF32(dst=vgpr(vgprTmp), src=vgpr("G2L%s+%u"%(tc, f16Tobf16Idx))))
                  localWriteCode.add(VCvtF16toF32(dst=vgpr(vgprTmp+1), src=vgpr("G2L%s+%u"%(tc, f16Tobf16Idx)),sdwa=sdwa))
                  localWriteCode.add(VPackF16toB32(dst=vgpr("G2L%s+%u"%(tc, f16Tobf16Idx)), src0=vgpr(vgprTmp), src1=vgpr(vgprTmp+1),
                                     vop3=VOP3PModifiers(op_sel=[1,1,0])))
                self.vgprPool.checkIn(vgprTmp)

            for oIdx in range(0, numOffsets):
              paramList.append(offset)

            #print "offset", offset

            #comment = "Reg -> L %u_%u_%u_%u"%(para, sPara, perp, sPerp)
            isHigh16Bits = False
            if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
              if s%2==1:
                isHigh16Bits = True
              if tP["glvw"]==1 and instHi%2==1:
                isHigh16Bits = True

            #       |  hi16  |  hi16  |        |        |
            #       |  hi8   |        |   hi8  |        |
            #############################################
            # VGPR: |---w4---|---w3---|---w2---|---w1---| -> b8_d16: get w1 / _b8_d16_hi: get w3
            # LSHR: |--------|---w4---|--------|---w2---| -> b8_d16: get w2 / _b8_d16_hi: get w4
            elif kernel["ProblemType"]["DataType"].isInt8():
              isHigh16Bits = (s % 4) > 1 # 2,3
              # TODO
              # if tP["glvw"]==1 and instructionCnt%2==1:
              #   isHigh16Bits = True
            LocalWriteX = tP["localWriteInstruction"].getInst(isHigh16Bits)
            if numBlocks == 1:
              ds        = DSModifiers(na=1, offset=paramList[1])
              writeInst = LocalWriteX(dstAddr=vgpr(lwa), src=paramList[0], ds=ds, comment=comment)
            else:
              ds        = DSModifiers(na=2, offset0=paramList[2], offset1=paramList[3])
              writeInst = LocalWriteX(dstAddr=vgpr(lwa), src0=paramList[0], src1=paramList[1], ds=ds, comment=comment)
            localWriteCode.add(writeInst)

      if tmpLocalWriteAddr != -1:
        self.vgprPool.checkIn(tmpLocalWriteAddr)

    if 0 and tP["isB"]: # post-lds-write
      localWriteCode.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment=""))
      localWriteCode.add(SBarrier(comment="dump LDS"))
      localWriteCode.add(self.getCmpAssert(self.asmAssert.ne, sgpr("WorkGroup0"),1))
      #localWriteCode.add(self.getBomb())

    return imod

  ##############################################################################
  # Local Read: Swap Offsets A/B
  # internalPointerSwap: swap internally tracked offsets - rather than
  #    emit specific instructions to do the pointer swap
  ##############################################################################
  def localReadSwapOffsets(self, kernel, internalPointerSwap, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]) or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return Module("localReadSwapOffsets (no local read)")
    if kernel["1LDSBuffer"]:
      return Module("localReadSwapOffsets (Empty)")
    module = Module("localReadSwapOffsets")
    if internalPointerSwap:
      tP["localReadSwapByteOffset"] = 0 if tP["localReadSwapByteOffset"] else kernel["LdsOffsetA_Blk"]*tP["bpe"]
      module.addComment1("local read swap internal offset -> %u" % tP["localReadSwapByteOffset"])
    else:
      module.add(VXorB32(
          dst=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          src0=hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]), \
          src1=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          comment="swap Red Blk"))
    return module

  ##############################################################################
  # Local Read: Reset Offsets A/B
  # x % n == n & (n-1) for n power of 2
  # tP[localReadOffset] maintains running count of offsets
  # This is called from the tail loop to reset read offsets?
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    tc=tP["tensorChar"]
    if not self.do["LocalRead%s"%tc]: return Module("localReadResetOffsets (no local read)")
    if kernel["1LDSBuffer"] or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return Module("localReadResetOffsets (Empty)")
    module = Module("localReadResetOffsets")
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadSwapByteOffset"] = 0
      module.addComment1("localReadResetOffsets")
      tP["localReadOffset"] = 0
      module.addComment0("handled internally")
    module.add(VAndB32(
        dst=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        src0=hex(kernel["LdsOffsetA_Blk"]*tP["bpe"]-1), \
        src1=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        comment="reset Red,Blk -> Red"))
    return module

  ##############################################################################
  # Local Read: Init Pointers A/B
  ##############################################################################
  def localReadInitPointers(self, kernel, tPA, tP):
    tc=tP["tensorChar"]
    if (not self.do["LocalRead%s"%tc]) or kernel["DirectToVgpr%s"%tc]:# no local read code if DirectToVgpr is enabled
      return Module("localReadInitPointers (Empty)")
    module = Module("localReadInitPointers")
    if tPA["localReadInstruction"].numOffsets == 1:
      module.addComment1("localReadInitPointers")
      tP["localReadOffset"] = 0
    else:
      module.add(VAndB32(
          dst=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          src0=hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*tP["bpe"]-1), \
          src1=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          comment="init Red,Blk -> Red"))
    return module

  ##############################################################################
  # Local Read offset conversion for DirectToLds
  ##############################################################################
  def localReadOffsetConvForDTL(self, kernel, tP, offset_val):
    tc = tP["tensorChar"]
    bit2 = offset_val & 4
    bit3 = offset_val & 8
    bit4 = offset_val & 16
    bit5 = offset_val & 32
    if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 8):
      # dword_x2 case
      # (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
      newVal = (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
    else:  #if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 16):  # most preferred case
      # dword_x4 case
      # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
      newVal = (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
    offset_val = offset_val & (~0x3c)
    offset_val = offset_val | newVal
    return offset_val

  ##############################################################################
  # Local Read: Increment A/B
  ##############################################################################
  def localReadInc(self, kernel, iui, tP):
    tc = tP["tensorChar"]
    if not self.do["LocalRead%s" % tc] or kernel["DirectToVgpr%s"%tc]: # no local read code if DirectToVgpr is enabled
      return Module("localReadInc (Empty)")

    module = Module("localReadInc")

    LdsPad = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0

    if self.states.inTailLoop:
      inc = kernel["LocalSplitU"] * (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad) * tP["bpe"]
      comment = " (LSU*(MT+PAD)*bpe)"
      if kernel["EnableMatrixInstruction"]:
        matrixInstK = kernel["MatrixInstK"]
        if kernel["UnrollMajorLDS%s" % tc]:
          if kernel["DirectToLds%s" % tc] and kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"] > 4:
            # DirectToLds special case. Need special address coonversion
            localReadOffset = kernel["LocalSplitU"] * kernel["MatrixInstK"] * max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB)
            localReadOffset *= tP["bpe"]
            prev_offset_val = 0 if iui == 0 else localReadOffset * iui
            offset_val = localReadOffset * (iui + 1)
            # offset conversion or DirectToLds
            prev_offset_val= self.localReadOffsetConvForDTL(kernel, tP, prev_offset_val)
            offset_val= self.localReadOffsetConvForDTL(kernel, tP, offset_val)
            inc = offset_val - prev_offset_val
            matrixInstK = 1 # multiplying matrixInstK is not necessary
            comment = ""
          else:
            inc = kernel["LocalSplitU"] * tP["bpe"]
            comment = " (LSU*bpe)"
        inc *= matrixInstK

      with self.allocTmpSgpr(1) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(inc), comment="inc"))
        module.add(VAddCOU32(
            dst=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            dst1=VCC(), \
            src0=sgpr(tmpSgpr), \
            src1=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            comment="lr%s += %u%s"%(tP["tensorChar"], inc, comment) ))
    else:
      if tP["localReadInstruction"].numOffsets == 1:
        if kernel["EnableMatrixInstruction"]:
          if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            tP["localReadOffset"] += kernel["LocalSplitU"] * kernel["MatrixInstK"] * max(self.states.numReadsIterCoalescedA,self.states.numReadsIterCoalescedB)
          else:
            if tc == "A":
              if kernel["MatrixInstB"] != 1 or self.states.lrvwA == self.states.lrvwB:
                tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * kernel["MatrixInstK"] * self.states.numReadsIterCoalescedA
              else:
                if (self.states.localReadDoCntA)%(kernel["LocalReadVectorWidth"]//self.states.lrvwA):
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * self.states.lrvwA
                else:
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*kernel["LocalReadVectorWidth"]//self.states.lrvwA-self.states.lrvwA*(kernel["LocalReadVectorWidth"]//self.states.lrvwA-1))
            else:
              if kernel["MatrixInstB"] != 1 or self.states.lrvwA == self.states.lrvwB:
                tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * kernel["MatrixInstK"] * self.states.numReadsIterCoalescedB
              else:
                if (self.states.localReadDoCntB)%(kernel["LocalReadVectorWidth"]//self.states.lrvwB):
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * self.states.lrvwB
                else:
                  tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad) * (kernel["MatrixInstK"]*kernel["LocalReadVectorWidth"]//self.states.lrvwB-self.states.lrvwB*(kernel["LocalReadVectorWidth"]//self.states.lrvwB-1))
        else:
          tP["localReadOffset"] += kernel["LocalSplitU"] * (kernel["MacroTile%s"%tP["tensorChar"]] + LdsPad)
        module.addComment0("N/A, lro->%d" % tP["localReadOffset"])
        module.addComment0("self.localReadDoCntA %d self.localReadDoCntB %d" % (self.states.localReadDoCntA,self.states.localReadDoCntB))
      else:
        inc = kernel["LocalSplitU"] * (kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad)
        module.add(VAddCOU32(
            dst=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            dst1=VCC(), \
            src0=hex(inc), \
            src1=vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
            comment="lr%s += %u (LSU+(MT+Pad)*bpe"%(tP["tensorChar"], inc) ))

    return module

  ##############################################################################
  # Local Read: Do It A/B
  # iui = Inner Unroll Idx
  # uIdx - Unroll Idx
  # epsi = expand pointer swap index. Only used for PAP
  ##############################################################################
  def localReadDo(self, kernel, bufferIdx, iui, epsi, tP):

    if not self.do["LocalRead%s" % tP["tensorChar"]]:
      imod = Module("LocalReadDo%s_I%s" % (tP["tensorChar"], iui))
      pack = Module("pack%s_I%s" % (tP["tensorChar"], iui))
      return imod, pack

    component = Component.LocalRead.find(self)
    if component:
      return component(self, kernel, bufferIdx, iui, epsi, tP)

  ##############################################################################
  # Save the local read pointers, for example when creating a duplicated
  # optimized path (like optNLL)
  ##############################################################################
  def saveLocalPointers(self, kernel, tPA, tPB):
    tPA["savedLocalReadOffset"] = tPA["localReadOffset"]
    tPB["savedLocalReadOffset"] = tPB["localReadOffset"]
    self.states.savedLocalReadDoCntA = self.states.localReadDoCntA
    self.states.savedLocalReadDoCntB = self.states.localReadDoCntB
    if kernel["ExpandPointerSwap"]:
      tPA["savedLocalWriteSwapByteOffset"] = tPA["localWriteSwapByteOffset"]
      tPB["savedLocalWriteSwapByteOffset"] = tPB["localWriteSwapByteOffset"]

  ##############################################################################
  # Restore the saved local read pointers
  # Must be paired with an earlier call to savePointers
  ##############################################################################
  def restoreLocalPointers(self, kernel, tPA, tPB):
    tPA["localReadOffset"] = tPA["savedLocalReadOffset"]
    tPB["localReadOffset"] = tPB["savedLocalReadOffset"]
    self.states.localReadDoCntA = self.states.savedLocalReadDoCntA
    self.states.localReadDoCntB = self.states.savedLocalReadDoCntB
    if kernel["ExpandPointerSwap"]:
      tPA["localWriteSwapByteOffset"] = tPA["savedLocalWriteSwapByteOffset"]
      tPB["localWriteSwapByteOffset"] = tPB["savedLocalWriteSwapByteOffset"]

  ##############################################################################
  # Shift Vector Components d0,1
  ##############################################################################
  def shiftVectorComponents(self, kernel, tP):
    component = Component.ShiftVectorComponents.find(self)
    if component:
      return component(self, kernel, tP)

  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    module = Module("localSplitULocalWrite")
    # wait for summation to be done with lds before writing reduction values
    module.add(self._syncThreads(kernel, "pre-lsu local write"))

    tmpVgpr = self.vgprPool.checkOutAligned(2, 2, "tmpVgpr")
    tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
    lr0 = self.vgprPool.checkOut(1,"lr0")
    lr1 = self.vgprPool.checkOut(1,"lr1")
    sg = self.vgprPool.checkOut(1,"sg")
    copy = self.vgprPool.checkOut(1,"copy")

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx

      # lr0 = serial % SG0
      module.add(vectorStaticDivideAndRemainder(lr1, lr0, "Serial", \
          kernel["SubGroup0"], tmpVgprRes))

      # lr1 = (serial / SG0) % SG1
      # sg  = (serial / SG0) / SG1
      module.add(VMovB32(dst=vgpr(copy), src=vgpr(lr1), comment="copy for divide"))
      module.add(vectorStaticDivideAndRemainder(sg, lr1, copy, \
          kernel["SubGroup1"], tmpVgprRes))

      # lr0 *= VW
      module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(kernel["VectorWidth"]*self.states.bpeCinternal), comment="VW"))
      module.add(VMulLOU32(dst=vgpr(lr0), src0=sgpr(tmpSgpr), src1=vgpr(lr0), comment="lr0 *= VW"))
      # lr1 *= VW*MT0
      module.add(SMovB32(dst=sgpr(tmpSgpr), \
          src=hex(kernel["VectorWidth"]*kernel["MacroTile0"]*self.states.bpeCinternal), comment="VW*MT0"))
      module.add(VMulLOU32(dst=vgpr(lr1), src0=sgpr(tmpSgpr), src1=vgpr(lr1), comment="lr1 *= VW*MT0"))
      # sg  *= MT0*MT1
      module.add(SMovB32(dst=sgpr(tmpSgpr), \
          src=hex(kernel["MacroTile0"]*kernel["MacroTile1"]*self.states.bpeCinternal), comment="MT0*MT1"))
      module.add(VMulLOU32(dst=vgpr(sg), src0=sgpr(tmpSgpr), src1=vgpr(sg), comment="sg *= MT0*MT1"))

    # thread offset
    addr = lr0
    module.add(VAddCOU32(dst=vgpr(addr), dst1=VCC(), src0=vgpr(lr1), src1=vgpr(addr)))
    module.add(VAddCOU32(dst=vgpr(addr), dst1=VCC(), src0=vgpr(sg), src1=vgpr(addr), comment="threadOffset"))
    self.vgprPool.checkIn(lr0)
    self.vgprPool.checkIn(lr1)
    self.vgprPool.checkIn(sg)
    self.vgprPool.checkIn(copy)
    self.vgprPool.checkIn(tmpVgpr)

    # dump addr
    # module.add(dump(vgpr(addr)))

    # do writes
    # LDS Layout example (for Sgemm, LSU=4, TT=8x8, WG=[8,4,4]), 128 WI/WG
    # VectorWidth = GlobalWriteVectorWidth = 4
    # SubGroup0 (WI:00-32)  : LDS 0x0000-
    # SubGroup1 (WI:33-64)  : LDS 0x2000-
    # SubGroup2 (WI:65-95)  : LDS 0x4000-
    # SubGroup3 (WI:96-127) : LDS 0x6000-

    # Interleave within a subgroup is interesting...
    #       Start LDS Addr
    # WI00 - 0x000
    # WI01 - 0x010
    # ...
    # WI07 - 0x070
    # WI08 - 0x400
    # WI09 - 0x410
    # ...
    # WI0F - 0x470
    # WI10 - 0x800
    # ...
    # ...
    # WI1f - 0xc70
    # WI20 - 0x1000  (start SubGroup1)

    # so a zoom-in on the pattern at beginning of LDS, for the case above:
    #   WI (hex) |x00-|x01-|...   |x07-|0x0-|0x1-|...|0x7-|0x0-| ... ... ||0x8-|
    # ValuC      |0123|0123|...   |0123|4567|4567|...|4567|89AB| ... ... ||0123
    #            |                     |                  |               |
    # LDS Addr  0x0                  0x80               0x100           0x400

    bytesPerElem = kernel["ProblemType"]["ComputeDataType"].numBytes()
    regsPerElem  = kernel["ProblemType"]["ComputeDataType"].numRegisters()
    bytesPerVector = kernel["VectorWidth"] * bytesPerElem
    bytesPerStep = min(bytesPerVector, 16) # max length of ds inst is 16 bytes(128bits)
    regsPerStep  = int((bytesPerStep+3)//4)
    elementStep = bytesPerStep // bytesPerElem

    for j in range(0, kernel["ThreadTile1"]//kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]//kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          for vc in range(0, kernel["VectorWidth"], elementStep):
            # for half, write 2 elements (4 bytes)
            # for single, write 1 element (4 bytes)
            # double doesn't work yet
            writeOffset = vc \
                + i*kernel["SubGroup0"]*kernel["VectorWidth"] \
                + s*kernel["MacroTile0"] \
                + j*kernel["MacroTile0"]*kernel["SubGroup1"]*kernel["VectorWidth"]
            regIdx = vc \
                + i*kernel["VectorWidth"] \
                + s*kernel["ThreadTile0"] \
                + j*kernel["ThreadTile0"]*kernel["VectorWidth"]
            regIdx = int(regIdx * regsPerElem)

            DSStoreBX = {128: DSStoreB128,
                         64:  DSStoreB64,
                         32:  DSStoreB32,
                         16:  DSStoreB16,
                         8:   DSStoreB8}[bytesPerStep*8]
            module.add(DSStoreBX(dstAddr=vgpr(addr), src=vgpr("ValuC+%u"%regIdx, regsPerStep), \
                ds=DSModifiers(offset=(writeOffset*self.states.bpeCinternal)),
                comment="j=%u i=%u s=%u vc=%u"%(j,i,s,vc)))

    module.add(SWaitCnt(lgkmcnt=0, vscnt=0, comment="wait for all writes"))
    module.add(self._syncThreads(kernel, "post-lsu local write"))
    # module.add(self.dumpLDS(kernel, 0, 16))
    #module.add(self.getBomb(5))
    return module

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    # calculate parameters
    bytesPerElem = kernel["ProblemType"]["ComputeDataType"].numBytes()
    regsPerElem  = kernel["ProblemType"]["ComputeDataType"].numRegisters()
    bytesPerVector = kernel["GlobalWriteVectorWidth"] * bytesPerElem
    bytesPerStep = 16
    while (bytesPerVector % bytesPerStep) != 0:
      bytesPerStep //= 2
    regsPerStep  = int((bytesPerStep+3)//4)
    elementStep = bytesPerStep // bytesPerElem

    # alloc resource
    baseAddr = self.vgprPool.checkOut(1,"baseAddr")

    with self.allocTmpSgpr(1) as tmpSgprInfo:
      # generate source
      module = Module("localSplitULocalRead")
      module.add(staticMultiply(vgpr(baseAddr), vgpr("Serial"), kernel["GlobalWriteVectorWidth"]*self.states.bpeAB, tmpSgprInfo))
      DSLoadBX = {128: DSLoadB128,
                  64:  DSLoadB64,
                  32:  DSLoadB32}[bytesPerStep*8]
      # Load values for each subgroup
      for r in range(0, kernel["LocalSplitU"]):
        for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
          for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
            offset = s + i*kernel["NumThreads"]*kernel["GlobalWriteVectorWidth"] + r * kernel["MacroTile0"]*kernel["MacroTile1"]
            regIdx = int((s + i*kernel["GlobalWriteVectorWidth"] + r*kernel["GlobalWriteVectorWidth"]*kernel["NumGlobalWriteVectorsPerThread"]) * regsPerElem)
            module.add(DSLoadBX(dst=vgpr("ValuC+%u"%regIdx,regsPerStep), src=vgpr(baseAddr), \
                ds=DSModifiers(offset=(offset*self.states.bpeCinternal)), comment="r=%u i=%u s=%u"%(r,i,s)))
      module.add(SWaitCnt(lgkmcnt=0, comment="wait for all reads"))

    if self.states.archCaps["SeparateVscnt"]:
      module.add(SWaitCnt(vscnt=0))

    # free resources
    self.vgprPool.checkIn(baseAddr)

    return module

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    module = Module("localSplitUReduction")

    is_non_hpa_fp16 = kernel["ProblemType"]["DataType"].isHalf() and (not kernel["ProblemType"]["HighPrecisionAccumulate"])
    elementStep = 2 if is_non_hpa_fp16 else 1
    regsPerElem = kernel["ProblemType"]["DataType"].numRegisters()

    for r in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
        for s in range(0, kernel["GlobalWriteVectorWidth"], elementStep):
          cIdx = int((s + i * kernel["GlobalWriteVectorWidth"]) * regsPerElem)
          regIdx = int((s + i * kernel["GlobalWriteVectorWidth"] + r * kernel["GlobalWriteVectorWidth"] * kernel["NumGlobalWriteVectorsPerThread"]) * regsPerElem)

          if is_non_hpa_fp16:
            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%cIdx), src0=vgpr("ValuC+%u" % regIdx), src1=vgpr("ValuC+%u"%cIdx), \
                        comment="c[%u] += c[%u]"%(cIdx, regIdx) ))
          elif kernel["ProblemType"]["DataType"].isInt8x4():
            module.add(VAddI32(dst=vgpr("ValuC+%u"%cIdx), src0=vgpr("ValuC+%u" % regIdx), src1=vgpr("ValuC+%u"%cIdx), \
                        comment="c[%u] += c[%u]"%(cIdx, regIdx)))

          elif kernel["ProblemType"]["DataType"].isSingle():
            module.add(VAddF32(dst=vgpr("ValuC+%u"%cIdx), src0=vgpr("ValuC+%u" % regIdx), src1=vgpr("ValuC+%u"%cIdx), \
                        comment="c[%u] += c[%u]"%(cIdx, regIdx)))
          elif kernel["ProblemType"]["DataType"].isDouble():
            module.add(VAddF64(dst=vgpr("ValuC+%u"%cIdx,2), src0=vgpr("ValuC+%u" % regIdx,2), src1=vgpr("ValuC+%u"%cIdx,2), \
                        comment="c[%u] += c[%u]"%(cIdx, regIdx)))
          elif kernel["ProblemType"]["DataType"].isSingleComplex():
            module.add(VAddF32(dst=vgpr("ValuC+%u"%(cIdx+0)), src0=vgpr("ValuC+%u" % (regIdx+0)), src1=vgpr("ValuC+%u"%(cIdx+0)), \
                        comment="c[%u] += c[%u], real part"%(cIdx, regIdx) ))
            module.add(VAddF32(dst=vgpr("ValuC+%u"%(cIdx+1)), src0=vgpr("ValuC+%u" % (regIdx+1)), src1=vgpr("ValuC+%u"%(cIdx+1)), \
                        comment="c[%u] += c[%u], imaginary part"%(cIdx+1, regIdx+1) ))
          elif kernel["ProblemType"]["DataType"].isDoubleComplex():
            module.add(VAddF64(dst=vgpr("ValuC+%u"%(cIdx+0),2), src0=vgpr("ValuC+%u" % (regIdx+0),2), src1=vgpr("ValuC+%u"%(cIdx+0),2), \
                        comment="c[%u] += c[%u], real part"%(cIdx, regIdx) ))
            module.add(VAddF64(dst=vgpr("ValuC+%u"%(cIdx+2),2), src0=vgpr("ValuC+%u" % (regIdx+2),2), src1=vgpr("ValuC+%u"%(cIdx+2),2), \
                        comment="c[%u] += c[%u], imaginary part"%(cIdx+2, regIdx+2) ))
          else:
            # TODO: hpa_half, int8
            assert(0) # unsupported data type, need to modify here and LSU write/read code
    return module

  ##############################################################################
  # computeStoreSrd
  # Add tile assignment fields to store srd
  # This is based on WG not the WI/TT assignment
  ##############################################################################
  def computeStoreSrdStart(self, kernel):
    module = Module("computeStoreSrdStart")

    with self.allocTmpSgpr(3) as tmpSgprInfo:
      tmpS0 = tmpSgprInfo.idx
      tmpS1 = tmpS0+1
      wgMT1 = tmpS0+2

      # Compute and save wg1*MT1 - the element offset that is top of the macro-tile in output space
      assert kernel["BufferStore"]
      module.addSpaceLine()
      module.add(SMulI32(
          dst=sgpr(wgMT1), \
          src0="MT1", \
          src1=sgpr("WorkGroup1"), \
          comment="<- wg1*MT1"))

      # Overall strategy is to set the SRD to the top-left of the macro-tile.
      # TT offsets are from this base (and include the column)

      # In non-packed mode:
      # higher-order tensor dims are static since this kernel operates within
      # the 2D Tensor formed by Index0 and Indexa.
      # Index0 and Index1 vary for each work-item (aka 'dynamic') so roll these into the VGPR

      # In packed mode:
      # Higher-order dimensions may be packed into coord0 / coord1 - see rowstart calculation below

      # Walk through addressing components (each tensor index) in C
      # For static dims add to SrdC / SrdD to compute a new base.
      # For dynamic (based on TT assignment) - save in coutRowPtr in computeStoreVgprs,
      # which saves the TT assignment for each WI scaled by StrideC0
      # TODO - future opportunities for store vgpr and other optimization
      #  - coutRowPtr and tid1 are strongly related - can we merge or remove one of these?
      # Packed follows same philosophy but may have more vector components
      indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
      numDim = len(indices)
      addrSrcSgpr = "Address" # use "Address" only for the first iteration
      for i in range(1, numDim):
        if i == kernel["ProblemType"]["Index0"]:
          # Used if the output is transposed?
          addToSrd = False
        elif i == kernel["ProblemType"]["Index1"] and len(kernel["PackedC1IndicesX"]) == 1:
          coord = sgpr(wgMT1)
          addToSrd = True
        elif i != kernel["ProblemType"]["Index0"] and i != kernel["ProblemType"]["Index1"] and not isPackedIndex(kernel, i):
          # group index, this is higher-order Tensor dimension, just add to SRD base:
          isStridedBuffer = kernel["ProblemType"]["StridedBatched"] or kernel["_GlobalAccumulation"]
          coord = sgpr("WorkGroup2") if isStridedBuffer else None
          addToSrd = True if isStridedBuffer else False
        else:
          # could be packed higher-order index, just ignore
          coord = None
          addToSrd = False

        if addToSrd:
          # These are constant across all workitems, just add to the SRD:
          strideC = "StrideC%s"%self.states.indexChars[i]
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(strideC), "CScale %s by Stride"%coord))
          module.add(SLShiftLeftB64(dst=sgpr(tmpS0,2), src=sgpr(tmpS0,2), shiftHex=log2(self.states.bpeCexternal), comment="scale by bpe"))

          module.add(SAddU32(dst=sgpr("SrdC+0"), src0=sgpr("%sC+0"%addrSrcSgpr), src1=sgpr(tmpS0), comment="add lo to SRD"))
          module.add(SAddCU32(dst=sgpr("SrdC+1"), src0=sgpr("%sC+1"%addrSrcSgpr), src1=sgpr(tmpS1), comment="add hi to SRD"))

          # These are constant across all workitems, just add to the SRD:
          stride = "StrideD%s" % (self.states.indexChars[i])
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpS0), sgpr(tmpS1), coord, sgpr(stride), "Scale %s by Stride"%coord))
          module.add(SLShiftLeftB64(dst=sgpr(tmpS0,2), src=sgpr(tmpS0,2), shiftHex=log2(self.states.bpeCexternal), comment="scale by bpe"))

          module.add(SAddU32(dst=sgpr("SrdD+0"), src0=sgpr("%sD+0"%addrSrcSgpr), src1=sgpr(tmpS0), comment="add lo to SRD"))
          module.add(SAddCU32(dst=sgpr("SrdD+1"), src0=sgpr("%sD+1"%addrSrcSgpr), src1=sgpr(tmpS1), comment="add hi to SRD"))

          module.addSpaceLine()

          addrSrcSgpr = "Srd" # update src Sgpr for the second or later iterations

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      # GSU algorithm 2: adjust output buffer address to per GSU buffer
      with self.allocTmpSgpr(5) as tmpSgprInfo:
        tmpSgpr = tmpSgprInfo.idx
        module.addComment("GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s")
        module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpSgpr+0), sgpr(tmpSgpr+1), sgpr("SizesFree+0"), sgpr("GSUSumIdx"), "Free0"))
        for i in range(1, numDim):
          module.add(SSubU32(dst=sgpr(tmpSgpr+4), src0=sgpr("SizesFree+%u"%i), src1=1, comment="Free%u" % i))
          module.add(SMulI32(dst=sgpr(tmpSgpr+4), src0=sgpr(tmpSgpr+4), src1=sgpr("GSUSumIdx"), comment="Free%u" % i))
          module.addModuleAsFlatItems(self.s_mul_u64_u32(sgpr(tmpSgpr+2), sgpr(tmpSgpr+3), sgpr(tmpSgpr+4), sgpr("StrideC%s"%self.states.indexChars[i]), "Free%u" % i))
          module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=sgpr(tmpSgpr+2), comment="Free%u" % i))
          module.add(SAddCU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+1), src1=sgpr(tmpSgpr+3), comment="Free%u" % i))
        module.add(SLShiftLeftB64(dst=sgpr(tmpSgpr+0,2), src=sgpr(tmpSgpr+0,2), shiftHex=log2(self.states.bpeCexternal), comment="scale by bpe"))
        module.add(SAddU32(dst=sgpr("SrdD+0"), src0=sgpr("SrdD+0"), src1=sgpr(tmpSgpr+0), comment="add lo GSU offset to SRD"))
        module.add(SAddCU32(dst=sgpr("SrdD+1"), src0=sgpr("SrdD+1"), src1=sgpr(tmpSgpr+1), comment="add hi GSU offset to SRD"))

    for cdir in (0,1):
      indices = kernel["PackedC%uIndicesX"%cdir]
      packedSizes = "PackedSize%u"%cdir
      if len(indices) > 1:
        for i,idx in enumerate(indices[1:]):
          if i==0:
            module.add(SMulI32(dst=sgpr(packedSizes), src0=self.sizeRef(indices[0]), \
                      src1=self.sizeRef(idx), comment="first packed size"))
          else:
            module.add(SMulI32(dst=sgpr(packedSizes), src0=sgpr(packedSizes), \
                      src1=self.sizeRef(idx), comment="first packed size"))

    return module

  ##############################################################################
  # computeStoreVgprs
  # Compute workitem/TT offsets in VGPRS
  # and coord0/coord1
  # tid0Scale specifies the number of output elements in 0/coalesced dim
  # that should be written by each work-item in each batch element.
  ##############################################################################
  def computeStoreVgprs(self, kernel, divisor, tid0Scale, tid1Scale):
    module = Module("computeStoreVgprs")
    module.addComment0("computeStoreVgprs")
    component = Component.ComputeStoreVgprs.find(self)
    if component:
      module.add(component(self, kernel, divisor, tid0Scale, tid1Scale)) #FIXME
    return module

  ##############################################################################
  # globalWriteWorkGroupInit:
  ##############################################################################
  def globalWriteWorkGroupInit(self, kernel):
    module = Module("globalWriteWorkGroupInit")
    if kernel["BufferStore"]:
      module.add(self.allocPostLoopSrd("D"))
      module.add(self.allocPostLoopSrd("C"))
      if kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
        labelStr = self.labels.getNameInc("ScaleD")
        module.add(allocPostLoopSrdSuppress("ScaleD", labelStr))
        module.add(SMulI32(dst=sgpr("SrdScaleD+2"), src0=hex(self.states.bpeCinternal), src1=sgpr("SrdScaleD+2"), comment="scaled by BPE"))# scaled by BPE
      module.add(self.computeStoreSrdStart(kernel))
    return module

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    module = Module("localSplitUGlobalWriteIndices")

    # lr0 = serial % SG0
    module.add(self.computeStoreVgprs(kernel, \
              divisor = kernel["MacroTile0"] // kernel["GlobalWriteVectorWidth"], \
              tid0Scale=kernel["GlobalWriteVectorWidth"], \
              tid1Scale=1))

    if kernel["BufferStore"]:
      #print "----AddressC-LocalSplitU"
      #print self.vgprPool.state()
      self.vgprs.addrD    = -1
      self.vgprs.addrC    = -1
      self.vgprs.addrBias = -1
      self.vgprs.addrScaleD = -1
    else:
      self.vgprs.addrD = self.vgprPool.checkOut(2)
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+0), \
          src=sgpr("AddressD+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+1), \
          src=sgpr("AddressD+1"), \
          comment="sgpr -> vgpr"))
      self.vgprs.addrC = self.vgprPool.checkOut(2)
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+0), \
          src=sgpr("AddressC+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+1), \
          src=sgpr("AddressC+1"), \
          comment="sgpr -> vgpr"))
      if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
        self.vgprs.addrBias = self.vgprPool.checkOut(2, 'addrBias')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+0), \
            src=sgpr("AddressBias+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+1), \
            src=sgpr("AddressBias+1"), \
            comment="sgpr -> vgpr"))
      if kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
        self.vgprs.addrScaleD = self.vgprPool.checkOut(2, 'addrScaleD')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleD+0), \
            src=sgpr("AddressScaleD+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleD+1), \
            src=sgpr("AddressScaleD+1"), \
            comment="sgpr -> vgpr"))

    return module

  ##############################################################################
  def allocPostLoopSrd(self, ch: str):
    module = Module("allocPostLoopSrd")
    # Buffer-load uses one base read pointer stored in the SRD - set it here:
    module.add(SMovB32(dst=sgpr("Srd%s+0"%ch), src=sgpr("Address%s+0"%ch), comment="init SRD base address (lower)" ))
    module.add(SMovB32(dst=sgpr("Srd%s+1"%ch), src=sgpr("Address%s+1"%ch), comment="init SRD base address (upper) + other fields" ))
    module.add(SMovB32(dst=sgpr("Srd%s+2"%ch), src=hex(0x80000000)))
    module.add(SMovB32(dst=sgpr("Srd%s+3"%ch), src="Srd127_96", comment="Set bits 127_96 in post-loop SRD"))
    module.addSpaceLine()
    return module

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    module = Module("notLocalSplitUGlobalWriteIndices")

    module.add(self.computeStoreVgprs(kernel,
              divisor = kernel["SubGroup0"],\
              tid0Scale=kernel["VectorWidth"], \
              tid1Scale=kernel["VectorWidth"]))

    if kernel["BufferStore"]:
      #print "----AddressC-nonLSU-----"
      #print self.vgprPool.state()
      self.vgprs.addrD    = -1
      self.vgprs.addrC    = -1
      self.vgprs.addrBias = -1
      self.vgprs.addrScaleD = -1
    else:
      self.vgprs.addrD = self.vgprPool.checkOut(2, 'addrD')
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+0), \
          src=sgpr("AddressD+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrD+1), \
          src=sgpr("AddressD+1"), \
          comment="sgpr -> vgpr"))
      self.vgprs.addrC = self.vgprPool.checkOut(2, 'addrC')
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+0), \
          src=sgpr("AddressC+0"), \
          comment="sgpr -> vgpr"))
      module.add(VMovB32(
          dst=vgpr(self.vgprs.addrC+1), \
          src=sgpr("AddressC+1"), \
          comment="sgpr -> vgpr"))
      if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
        self.vgprs.addrBias = self.vgprPool.checkOut(2, 'addrBias')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+0), \
            src=sgpr("AddressBias+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrBias+1), \
            src=sgpr("AddressBias+1"), \
            comment="sgpr -> vgpr"))
      if kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
        self.vgprs.addrScaleD = self.vgprPool.checkOut(2, 'addrScaleD')
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleD+0), \
            src=sgpr("AddressScaleD+0"), \
            comment="sgpr -> vgpr"))
        module.add(VMovB32( \
            dst=vgpr(self.vgprs.addrScaleD+1), \
            src=sgpr("AddressScaleD+1"), \
            comment="sgpr -> vgpr"))
    return module

  ##############################################################################
  # Release any resources used by the global write
  def cleanupGlobalWrite(self, kernel):
    self.vgprPool.checkIn(self.vgprs.coord0)
    self.vgprPool.checkIn(self.vgprs.coord1)

    if kernel["StoreRemapVectorWidth"]:
      self.vgprPool.checkIn(self.vgprs.storeRemapLW)
      self.vgprPool.checkIn(self.vgprs.storeRemapLR)
      self.vgprPool.checkIn(self.vgprs.storeRemapCoord0)
      self.vgprPool.checkIn(self.vgprs.storeRemapCoord1)
      self.vgprPool.checkIn(self.vgprs.storeRemapOffsetCoord1)
    if kernel["BufferStore"]:
      self.vgprPool.checkIn(self.vgprs.cinRowPtr)
      self.vgprPool.checkIn(self.vgprs.coutRowPtr)
    if not kernel["BufferStore"]:
      self.vgprPool.checkIn(self.vgprs.addrD)
      self.vgprPool.checkIn(self.vgprs.addrC)
      if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
        self.vgprPool.checkIn(self.vgprs.addrBias)
      if kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
        self.vgprPool.checkIn(self.vgprs.addrScaleD)

  ##############################################################################
  # Return max global write vector width, in elements
  def maxGwvw(self, kernel):
    atomic = (kernel["GlobalSplitU"] > 1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')

    if kernel["BufferStore"]:
      if atomic:
        return self.getVectorAtomicWidth(kernel)
      else:
        return 1000  # no limit
    else:
      if atomic:
        return 1  # flat vector atomic is not tested
      else:
        return 1000  # no limit

  def getVectorAtomicWidth(self, kernel):
    if kernel["ProblemType"]["DataType"].isHalf() and (not kernel["_GlobalAccumulation"]):
      return 2
    return 1

  ##############################################################################
  # Partition thread-tile into writeElements for store code
  # This function creates the writeElement mapping for full tiles
  # (ie non-edge cases)
  ##############################################################################
  def notLocalFullTileElements(self, kernel, edge):
    component = Component.NotLocalFullTileElements.find(self)
    if component:
      return component(self, kernel, edge)

  ##############################################################################
  # Store Remap: Local Write
  ##############################################################################
  def storeRemapAddLocalWrite(self, ss, addrCalc, srcVgpr):
    """
    Add localWrite for the element with addrCalc and srcVgpr.
    """

    module = Module("storeRemapAddLocalWrite srcVgpr %s"%str(srcVgpr))

    bps = self.states.bpeCexternal * ss.cfg.gwvw
    rpv = self.states.bpeCexternal * ss.cfg.gwvw / self.states.bpr

    addr0  = vgpr(self.vgprs.storeRemapLW)
    offset = addrCalc.coordOffset0 * self.states.bpeCexternal
    ds     = DSModifiers(offset=offset)

    if bps==1:
      module.add(DSStoreB8(dstAddr=addr0, src=vgpr(srcVgpr, rpv*4), \
                ds=ds, comment="storeRemap lw"))
    elif bps==2:
      module.add(DSStoreB16(dstAddr=addr0, src=vgpr(srcVgpr, rpv*2), \
                ds=ds, comment="storeRemap lw"))
    elif bps==4:
      module.add(DSStoreB32(dstAddr=addr0, src=vgpr(srcVgpr, rpv), \
                ds=ds, comment="storeRemap lw"))
    elif bps==8:
      module.add(DSStoreB64(dstAddr=addr0, src=vgpr(srcVgpr, rpv), \
                ds=ds, comment="storeRemap lw"))
    elif bps==16:
      module.add(DSStoreB128(dstAddr=addr0, src=vgpr(srcVgpr, rpv), \
                ds=ds, comment="storeRemap lw"))
    else:
      assert 0, "StoreRemap: bad bps!"

    return module

  ##############################################################################
  # Store Remap: Local Read and Global Write
  ##############################################################################
  def storeRemapAddStore(self, kernel, tmpVgpr, tmpS01, edge):
    module = Module("storeRemapAddStore")

    module.add(SWaitCnt(lgkmcnt=0, comment="wait for LDS write"))

    numStoreInst = 0

    #Data exchange between different waves
    #Make sure LDS writes are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      # FIXME: Indent?
      module.add(SBarrier(comment="wait all lds write finished"))
    module.addSpaceLine()

    gwvw = kernel["StoreRemapVectorWidth"]
    nElements = kernel["MacroTile0"]*kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]//self.states.kernel["WavefrontSize"]

    bpe = self.states.bpeCexternal
    bps = bpe * gwvw
    rpe = self.states.bpeCexternal / self.states.bpr
    rpv = rpe * gwvw

    # num registers to check out
    storeRegs = []
    for i in range(0, nElements, gwvw):
      storeRegs.append(self.vgprPool.checkOutAligned(int(rpv), int(rpv), "store element d"))
    src = vgpr(self.vgprs.storeRemapLR)
    for rIdx, i in enumerate(range(0, nElements, gwvw)):
      offset = self.storeRemapLrOffset * bpe * (i//gwvw)
      ds  = DSModifiers(offset=offset)
      dst = vgpr(storeRegs[rIdx], rpv)
      if bps==4:
        module.add(DSLoadB32(dst=dst, src=src, ds=ds, readToTempVgpr=False, comment="storeRemap lr"))
      elif bps==8:
        module.add(DSLoadB64(dst=dst, src=src, ds=ds, readToTempVgpr=False, comment="storeRemap lr"))
      elif bps==16:
        module.add(DSLoadB128(dst=dst, src=src, ds=ds, readToTempVgpr=False, comment="storeRemap lr"))
      else:
        assert 0, "StoreRemap: bad bps!"

    module.addSpaceLine()

    # Global Write
    ntStr = ""
    if kernel.enabledSetPrioSplitLDS:
      module.add(SSetPrior(prior=1))
    if kernel["NonTemporalD"]%2==1:
      ntStr += " glc"
    if kernel["NonTemporalD"]//2==1:
      ntStr += " slc"

    addr1 = sgpr("SrdD", 4)
    packedD1 = kernel["PackedC1IndicesX"]
    strideD1 = "StrideD%s" % (self.states.indexChars[packedD1[0]])

    vTmp = self.vgprPool.checkOut(1, "SR Store temp addr0")
    addr0 = vgpr(vTmp)

    if not edge:
      for rIdx, i in enumerate(range(0, nElements, gwvw)):
        if i == 0:
          module.add(VMovB32(dst=addr0, src=vgpr(self.vgprs.storeRemapOffsetCoord1), comment="coord1"))
        else:
          currentStep = i//gwvw
          module.add(VAddU32(dst=addr0, src0=vgpr(self.vgprs.storeRemapOffsetCoord1), src1=(self.storeRemapNCPL * currentStep), comment="coord1 += nColPerLoad"))

        module.add(VMulLOU32(dst=addr0, src0=addr0, src1=sgpr(strideD1), comment="coord1 offset =  coord1 * StrideD"))
        module.add(VAddLShiftLeftU32(dst=addr0, src0=addr0, src1=vgpr(self.vgprs.storeRemapCoord0), shiftHex=hex(log2(bpe)), comment="global write D address"))

        lgkmcnt = min((nElements-i)//gwvw - 1, 15)
        module.add(SWaitCnt(lgkmcnt=lgkmcnt, comment="wait for LDS read"))

        numStoreInst += 1
        module.add(self.chooseGlobalWrite(True, bps, storeRegs[rIdx], rpv, addr0, addr1, 0, ntStr))
    else:
      tmpS23 = tmpS01+self.states.laneSGPRCount
      coord0 = tmpVgpr
      coord1 = coord0+1
      lrVw = kernel["StoreRemapVectorWidth"]
      edgeVw = min(kernel["AssertFree0ElementMultiple"],kernel["StoreRemapVectorWidth"])
      bps = self.states.bpeCexternal * edgeVw
      rpv = self.states.bpeCexternal / self.states.bpr * edgeVw
      for rIdx, i in enumerate(range(0, nElements, lrVw)):
        for vi in range (0, lrVw, edgeVw):

          if vi == 0:
            lgkmcnt = min((nElements-i)//lrVw - 1, 15)
            module.add(SWaitCnt(lgkmcnt=lgkmcnt, comment="wait for LDS read"))

          sizeBoundary = [0,0]
          sizeBoundary[0] = \
              sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index0"])
          sizeBoundary[1] = \
              sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
              else self.sizeRef(kernel["ProblemType"]["Index1"])

          currentStep = i//lrVw

          # calculate global coordination
          module.add(VAddU32(dst=vgpr(coord1), src0=vgpr(self.vgprs.storeRemapCoord1), src1=(self.storeRemapNCPL * currentStep), comment="coord1 += nColPerLoad"))
          module.add(VAddU32(dst=vgpr(coord0), src0=vgpr(self.vgprs.storeRemapCoord0), src1=vi, comment="coord0 += element index of load vector"))
          module.add(VAddU32(dst=addr0, src0=vgpr(self.vgprs.storeRemapOffsetCoord1), src1=(self.storeRemapNCPL * currentStep), \
                      comment="offset coord1 += nColPerLoad"))

          module.add(VCmpLtU32(dst=sgpr(tmpS01,self.states.laneSGPRCount), src0=vgpr(coord0), src1=sizeBoundary[0], comment="coord0 < size0" ))
          module.add(VCmpLtU32(dst=sgpr(tmpS23,self.states.laneSGPRCount), src0=vgpr(coord1), src1=sizeBoundary[1], comment="coord1 < size1" ))
          SAndBX = SAndB32 if self.states.kernel["WavefrontSize"] == 32 else SAndB64
          module.add(SAndBX(
                      dst=sgpr(tmpS23,self.states.laneSGPRCount),
                      src0=sgpr(tmpS01,self.states.laneSGPRCount),
                      src1=sgpr(tmpS23,self.states.laneSGPRCount), comment="in0 && in1" ))

          module.add(VMulLOU32(dst=addr0, src0=addr0, src1=sgpr(strideD1), comment="coord1 element offset =  coord1 * StrideD"))
          module.add(VAddLShiftLeftU32(dst=addr0, src0=addr0, src1=vgpr(coord0), shiftHex=hex(log2(bpe)), comment="scale to BPE"))
          module.add(VCndMaskB32(dst=addr0, src0=-1, src1=addr0, src2=sgpr(tmpS23,self.states.laneSGPRCount), comment="clip if OOB. offset" ))

          sumIdx = storeRegs[rIdx] + int(vi*rpe)
          numStoreInst += 1
          if bps == 2:
            module.add(self.chooseGlobalWrite(True, bpe, sumIdx, rpe, addr0, addr1, 0, ntStr, hi16=vi%2))
          else:
            module.add(self.chooseGlobalWrite(True, bps, sumIdx, rpv, addr0, addr1, 0, ntStr))

          if bps == 1:
            module.add(VAShiftRightI32(dst=vgpr("ValuC+%u"%sumIdx), shiftHex=8, src=vgpr("ValuC+%u"%sumIdx), comment=" shift 1 byte" ))

    module.addSpaceLine()
    self.vgprPool.checkIn(vTmp)
    for v in storeRegs:
      self.vgprPool.checkIn(v)

    #Data exchange between different waves
    #Make sure LDS reads are finished of all waves
    if kernel["MIWaveGroup"][0] > 1:
      module.add(SBarrier(comment="wait all lds read finished"))

    return module, numStoreInst

  ##############################################################################
  # Store remap compute vgprs:
  ##############################################################################
  def storeRemapComputeStoreVgprs(self, kernel):
    module = Module("storeRemapComputeStoreVgprs")
    module.addComment0("Store Remap Local Write address")

    with self.allocTmpSgpr(2) as tmpSgprInfo:
      tmpS0 = tmpSgprInfo.idx
      wgMT1 = tmpS0 + 1

      wg0="WorkGroup0"
      wg1="WorkGroup1"

      tid0 = self.vgprPool.checkOut(1, "SR coord0")
      tid1 = self.vgprPool.checkOut(1, "SR coord1")
      coord1Offset = self.vgprPool.checkOut(1, "SR coord1 offset")
      storeRemapLW = self.vgprPool.checkOut(1, "SR local write")
      storeRemapLR = self.vgprPool.checkOut(1, "SR local read")

      tmpV0 = self.vgprPool.checkOut(5, "tmpV0")
      waveCoord0 = tmpV1 = tmpV0+1
      ldsStride = tmpV0+2
      coord0 = tmpV0+3
      waveCoord1 = tmpV0+4

      gwvw = kernel["StoreRemapVectorWidth"]
      ldsPad = max(kernel["StoreRemapVectorWidth"],kernel["MIOutputVectorWidth"])

      #calculate local write Address: v[vgprLocalWriteAddrC]
      module.add(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.states.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], \
        RegisterPoolResource(tmpV0, 2)))

      module.add(VMulLOU32(dst=vgpr(waveCoord1),
                src0=hex(kernel["MatrixInstN"]), src1=vgpr(tid1), comment="coord1 offset of LDS for each Wave"))
      module.add(VAndB32(dst=vgpr(tid1),
                src0=hex(kernel["MatrixInstN"]-1), src1=vgpr("Serial"), comment="coord1 offset of LDS for each thread"))
      module.add(VAddU32(dst=vgpr(tid1), src0=vgpr(waveCoord1), src1=vgpr(tid1), comment="coord1 offset in MacroTile"))
      module.add(VMovB32(dst=vgpr(ldsStride), src=hex(kernel["MacroTile0"]+ldsPad), \
                comment="lds stride = MT0 + PAD"))
      module.add(VMulLOU32(dst=vgpr(tmpV0), src0=vgpr(tid1), src1=vgpr(ldsStride), \
                comment="lds coord1 offset = Col-id* lds stride"))

      module.add(vectorStaticDivideAndRemainder(waveCoord0, tid0, tid0, self.states.kernel["WavefrontSize"], RegisterPoolResource(tmpV0, 2)))
      module.add(VLShiftRightB32(dst=vgpr(coord0),
                shiftHex=hex(log2(kernel["MatrixInstN"])), src=vgpr(tid0), \
                comment="tid / matrixInstN"))

      if kernel["MIOutputVectorWidth"] > 1:
        module.add(VLShiftLeftB32(dst=vgpr(coord0), shiftHex=hex(log2(kernel["MIOutputVectorWidth"])), src=vgpr(coord0), \
                comment="lds coord0 offset *= 4 (each thread hold 4 element)"))

      module.add(VMadU32U24(dst=vgpr(coord0), src0=(kernel["MatrixInstM"]*kernel["MatrixInstBM"]), src1=vgpr(waveCoord0), src2=vgpr(coord0), \
                comment="coord0 += waveCoord0 * wave M shape(blockM*MiM)"))

      module.add(VAddLShiftLeftU32(
        dst=vgpr(storeRemapLW), \
        src0=vgpr(tmpV0), \
        src1=vgpr(coord0), \
        shiftHex=hex(log2(self.states.bpeCexternal)), \
        comment="local write C address"))

      module.addSpaceLine()
      # calculate local read address : v[vgprLocalReadAddrC]

      module.addComment0("Store Remap Local Read address")

      module.add(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.states.kernel["WavefrontSize"], \
        RegisterPoolResource(tmpV0, 2)))
      module.add(VMulLOU32(dst=vgpr(waveCoord1),
                src0=hex(kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]), src1=vgpr(tid1), comment="coord1 offset of LDS for each Wave"))

      nThreadPerCol = kernel["MacroTile0"] // gwvw
      nColPerLoad = self.states.kernel["WavefrontSize"] // nThreadPerCol
      self.storeRemapLrOffset = (kernel["MacroTile0"]+ldsPad) * nColPerLoad
      self.storeRemapNCPL = nColPerLoad

      module.add(VLShiftRightB32(dst=vgpr(tmpV1),\
                shiftHex=hex(log2(nThreadPerCol)), src=vgpr(tid0), \
                comment="tid / nThreadPerCol"))
      module.add(VAddU32(dst=vgpr(coord1Offset), src0=vgpr(waveCoord1), src1=vgpr(tmpV1), comment="coord1 offset in MacroTile"))
      module.add(VMulLOU32(dst=vgpr(tmpV0), src0=vgpr(coord1Offset), src1=vgpr(ldsStride), \
                comment="lds coord1 offset = Col-id* lds stride"))

      module.add(VAndB32(dst=vgpr(coord0), src0=hex(nThreadPerCol-1), src1=vgpr(tid0),
                comment="coord0 offset of LDS for each thread"))
      module.add(VLShiftLeftB32(dst=vgpr(coord0), shiftHex=hex(log2(gwvw)), src=vgpr(coord0), \
                comment="lds coord0 offset *= gwvw (each thread hold gwvw element)"))

      module.add(VAddLShiftLeftU32(
                dst=vgpr(storeRemapLR), \
                src0=vgpr(tmpV0), \
                src1=vgpr(coord0), \
                shiftHex=hex(log2(self.states.bpeCexternal)), \
                comment="local read C address"))
      module.addSpaceLine()

      # calculate global write coord0 and coord1
      module.addComment0("Store Remap global write coord0 and coord1")
      module.add(vectorStaticDivideAndRemainder(tid1, tid0, "Serial", self.states.kernel["WavefrontSize"]*kernel["MIWaveGroup"][0], \
        RegisterPoolResource(tmpV0, 2)))

      ColsPerBlockShape = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

      module.add(VMulLOU32(dst=vgpr(waveCoord1), src0=hex(ColsPerBlockShape), src1=vgpr(tid1),
                comment="coord1 offset of global memory for each Wave"))

      module.add(vectorStaticDivideAndRemainder(tid1, tid0, tid0, self.states.kernel["WavefrontSize"], \
        RegisterPoolResource(tmpV0, 2)))
      module.add(VMadU32U24(dst=vgpr(waveCoord1), src0=(kernel["MatrixInstN"]//kernel["MIWaveGroup"][0]), src1=vgpr(tid1), src2=vgpr(waveCoord1), \
                comment="waveCoord1 += waveCoord0 * MiN / WaveGroupM"))

      module.add(VLShiftRightB32(dst=vgpr(tmpV1), shiftHex=hex(log2(nThreadPerCol)), src=vgpr(tid0), \
                comment="tid / nThreadPerCol"))

      module.add(VAddU32(dst=vgpr(coord1Offset), src0=vgpr(waveCoord1), src1=vgpr(tmpV1), comment="coord1 offset in MacroTile"))

      module.add(SMulI32(dst=sgpr(tmpS0), src0=hex(kernel["MacroTile0"]), src1=sgpr(wg0), comment="%s = wg0*MT0"%sgpr(tmpS0)))

      module.add(VAddCOU32(dst=vgpr(tid0), dst1=VCC(), src0=sgpr(tmpS0), src1=vgpr(coord0), comment="coord0 = coord0 + wg0 * MT0"))

      module.add(SMulI32(dst=sgpr(wgMT1), src0="MT1", src1=sgpr(wg1), comment="<- wg1*MT1"))
      module.add(VAddCOU32(dst=vgpr(tid1), dst1=VCC(), src0=sgpr(wgMT1), src1=vgpr(coord1Offset), comment="coord1 = tid1*VW + wg1*MT1"))

      module.addSpaceLine()

      module.add(self._syncThreads(kernel, "StoreRemap Start"))

      self.vgprs.storeRemapLW = storeRemapLW  #local write
      self.vgprs.storeRemapLR = storeRemapLR  #local read
      self.vgprs.storeRemapCoord0 = tid0      #global coord0
      self.vgprs.storeRemapCoord1 = tid1      #global coord1
      self.vgprs.storeRemapOffsetCoord1 = coord1Offset #offset coord1

      self.vgprPool.checkIn(tmpV0)

    return module

  ##############################################################################
  # Not LocalSplitU: Global Write
  # Determine write batching pattern
  # element() specifies TT 'coordinate' to write
  # vectorWidths specifies width of vector to store
  # TODO - why does this use VectorWidth to control store width?  Could be GlobalWriteVectorWidth?
  #
  # Function creates one mapping for full tiles and one for edge tiles,
  # then calls globalWriteElements to generate the code for the new tiles.
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel, tPA, tPB):
    if not self.do["PostLoop"]: return ""
    elements = [[] for y in range(2)] # 2D array for Full, Edge

    (fullVw, elements[False]) = self.notLocalFullTileElements(kernel, False)
    (edgeVw, elements[True])  = self.notLocalFullTileElements(kernel, True)

    # print("len(elements[False])= ", len(elements[False]))
    # print("len(elements[True])= ", len(elements[True]))
    vectorWidths = [fullVw, edgeVw]

    module = Module("notLocalSplitUGlobalWrite")
    module.add(self.globalWriteElements(kernel, tPA, tPB, vectorWidths, elements))

    self.cleanupGlobalWrite(kernel)

    return module

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel, tPA, tPB):
    if not self.do["PostLoop"]: return ""

    fullVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    fullVw = min(fullVw, self.maxGwvw(kernel))
    elements = [[] for y in range(2)] # 2D array for Full, Edge
    # Full tile loop:
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], fullVw): # note step by fullVw
            element = (tt1, tt0, vc1, vc0)
            elements[False].append(element)

    # Edge tile loop - note if we know AF0EM we can can use a larger vector
    # and reduce the boundary checks accordingly.  But if no AF0EM guarantee
    # then use a conservative 1
    edgeVw = kernel["GlobalWriteVectorWidth"] if kernel["_VectorStore"] else 1
    edgeVw = min(edgeVw, self.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
    assert(kernel["GlobalWriteVectorWidth"]%edgeVw == 0)
    for tt1 in range(0, kernel["NumGlobalWriteVectorsPerThread"]):
      for vc1 in range(0, 1):
        for tt0 in range(0, 1):
          for vc0 in range(0, kernel["GlobalWriteVectorWidth"], edgeVw):
            element = (tt1, tt0, vc1, vc0)
            elements[True].append(element)

    vectorWidths = [fullVw, edgeVw]
    module = Module("localSplitUGlobalWrite")
    module.add(self.globalWriteElements(kernel, tPA, tPB, vectorWidths, elements))
    self.cleanupGlobalWrite(kernel)
    return module

  ##############################################################################
  # checkIsBetaZero
  # tmpSgpr is one temp sgpr
  # betaLabel is label to branch to if beta != 0
  ##############################################################################
  def checkIsBetaZero(self, kernel, tmpSgpr, betaLabel, isLongBranch=False, posNeg: int=0):
    module = Module("checkIsBetaZero label %s"%betaLabel)
    assert(isinstance(betaLabel, Label))
    betaLabelName = betaLabel.getLabelName()
    if kernel["ProblemType"]["UseBeta"]:
      if self.states.bpeCinternal <= self.states.bpr: # 1 register to check for Beta==0
        module.add(SCmpKEQU32(src=sgpr("Beta"), simm16=hex(0), comment="Beta == 0"))
      else: # multiple registers to check for Beta==0
        module.add(SMovB32(dst=sgpr(tmpSgpr), src=sgpr("Beta+0"), comment="tmp = Beta[0]"))
        for i in range(1, self.states.bpeCinternal//self.states.bpr):
          module.add(SOrB32(dst=sgpr(tmpSgpr), src0=sgpr("Beta+%u"%i), src1=sgpr(tmpSgpr), comment="tmp |= Beta[%u] " % i))
        module.add(SCmpKEQU32(src=sgpr(tmpSgpr), simm16=hex(0), comment="Beta == 0"))
      if isLongBranch:
        module.add(self.longBranchScc0(betaLabel, posNeg))
      else:
        module.add(SCBranchSCC0(labelName=betaLabelName, comment="Branch if Beta is not zero"))
    module.addSpaceLine()
    return module

  ##############################################################################
  # checkIsEdge
  # tmpSgpr must have at least 6 free SGPR
  # isEdgeTarget is the branch target if edges are required
  ##############################################################################
  def checkIsEdge(self, kernel, tmpSgpr, isEdgeTarget, isLongBranch=False):
    assert(isinstance(isEdgeTarget, Label))
    isEdgeTargetLabel = isEdgeTarget.getLabelName()
    module = Module("checkIsEdge")
    tmpS0  = tmpSgpr
    tmpS1  = tmpS0 + 1
    tmpS23 = tmpS1 + 1

    wg0="WorkGroup0"
    wg1="WorkGroup1"

    # check edge0 ###
    # s23 = rMT0 = Size0 % MT0
    #--
    sizeBoundary = [0,0]
    sizeBoundary[0] = \
        sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index0"])
    sizeBoundary[1] = \
        sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
        else self.sizeRef(kernel["ProblemType"]["Index1"])

    module.add(scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[0], kernel["MacroTile0"], \
      RegisterPoolResource(tmpS23, 2), 2))
    # s23 = nwg0-1
    module.add(SAddU32(dst=sgpr(tmpS1), src0=hex(-1), src1=sgpr("NumWorkGroups0")))
    module.add(SCmpGeU32(src0=sgpr(wg0), src1=sgpr(tmpS1), comment="wg0 >= nwg0-1 ?"))
    module.add(SCSelectB32(dst=sgpr(tmpS0), src0=sgpr(tmpS0), src1=0, comment="set rMT0"))
    # s01 now = myMT0 = wg0 < nwg0-1 ? MT0 : rMT0

    # if rMT0 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      module.add(SCmpKGtU32(src=sgpr(tmpS0), simm16=hex(0), comment="rMT0 > 0"))
      if self.db["ForceEdgeStores"]:
        module.add(SCmpEQU32(src0=sgpr(tmpS0), src1=sgpr(tmpS0), comment="ForceEdgeStores!"))
      if isLongBranch:
        module.add(self.longBranchScc1(isEdgeTarget, posNeg=1, comment="jump if edges required"))
      else:
        module.add(SCBranchSCC1(labelName=isEdgeTargetLabel, comment="jump if edges required"))

    # check edge1 ###
    # TODO-packed - this only needs to change to handle packing into C1 index
    # change would be similar to above - multiply by product of packed sizes in C1
    # --

    # s23 = rMT1 = Size1 % MT1
    module.add(scalarStaticDivideAndRemainder(tmpS1, tmpS0, sizeBoundary[1], kernel["MacroTile1"], \
      RegisterPoolResource(tmpS23, 2), 2))
    # s01 now = myMT1 = wg1 < nwg1-1 ? MT1 : rMT1

    # s23 = nwg1-1
    module.add(SAddU32(dst=sgpr(tmpS1), src0=hex(-1), src1=sgpr("NumWorkGroups1")))
    module.add(SCmpGeU32(src0=sgpr(wg1), src1=sgpr(tmpS1), comment="wg1 >= nwg1-1"))
    module.add(SCSelectB32(dst=sgpr(tmpS0), src0=sgpr(tmpS0), src1=0, comment="set rMT1"))

    # if rMT1 > 0 goto label_B?_E1
    if self.do["EdgeWrite"]:
      module.add(SCmpKGtU32(src=sgpr(tmpS0), simm16=hex(0), comment="rMT1 > 0"))
      if isLongBranch:
        module.add(self.longBranchScc1(isEdgeTarget, posNeg=1, comment="jump if edges required"))
      else:
        module.add(SCBranchSCC1(labelName=isEdgeTargetLabel, comment="jump if edges required"))

    return module

  ##############################################################################
  # Global Write Elements
  ##############################################################################

  class BF16CVTVgprStruct(NamedTuple): # class for bf16 vgprs
    vgprBf16Temp: int = -1
    vgprBf16Mask: int = -1
    vgprFp32Nan: int = -1
    vgprBf16Inc: int = -1

  class ActivationSetPCStruct(NamedTuple):
    sgprOffsetActivation: int = -1
    sgprOffsetBack: int = -1
    vgprActCopy: int = -1

  def globalWriteElements(self, kernel, tPA, tPB, vectorWidths, elements,
                          applyAlpha=True, # defaults to generating *=alpha codes
                          betas=None, # if left unspecified, then let global parameter decide
                          edges=None):
    if not self.do["PostLoop"]: return Module("GlobalWriteElements (Empty)")
    module = Module("GlobalWriteElements")
    module.addComment2("Global Write Elements")
    if self.states.numStoreSgprToLoad: # Wait for kernel args
      module.add(SWaitCnt(lgkmcnt=0, comment="wait for %u bytes of kern args."%(self.states.numStoreSgprToLoad * 4)))

    atomic = (kernel["GlobalSplitU"] > 1) and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')
    activation = ActivationModule()

    # write possibilities and labels
    # if beta/edge combo not specified fall back to global param definition
    if betas is None:
      hasBeta = kernel["ProblemType"]["UseBeta"] and (kernel["_GlobalAccumulation"] != 'MultipleBuffer')
      betas = [False, True] if hasBeta else [False]
    if edges is None:
      edges = [False, True] if self.do["EdgeWrite"] else [False]
    writeLabels = {}
    for beta in betas:
      writeLabels[beta] = {}
      for edge in edges:
        writeLabels[beta]["EdgeCheck0"] = Label(self.labels.getNameInc("GW_B%u_E%u_EdgeCheck0" % ( 1 if beta else 0, 1 if edge else 0) ), "")
        writeLabels[beta]["EdgeCheck1"] = Label(self.labels.getNameInc("GW_B%u_E%u_EdgeCheck1" % ( 1 if beta else 0, 1 if edge else 0) ), "")
        writeLabels[beta][edge] = Label(self.labels.getNameInc("GW_B%u_E%u" % ( 1 if beta else 0, 1 if edge else 0) ), "")
      if not beta:
        betaLabel = Label(self.labels.getNameInc("GW_Beta"), "")
    endLabel = Label(self.labels.getNameInc("GW_End"), "")

    # Layout
    """
    if B1 goto label_B1
    if E1 goto label_B0_E1
    label_B0_E0:
    writes
    goto label_End
    label_B0_E1:
    writes
    goto label_End
    label_B1:
    if E1 goto label_B1_E1
    label_B1_E0:
    writes
    goto label_End
    label_B1_E1:
    writes
    goto label_End
    label_End
    """

    ########################################
    # Vgprs
    if kernel["BufferStore"]:
      numTmpVgpr = 2
      if len(kernel["PackedC0IndicesX"]) > 1:
        numTmpVgpr += 1
    else:
      numTmpVgpr = 2 + 3 # GLOBAL_OFFSET_C needs 3, plus 2 tmps?
    # Get max vgpr and sgpr for activation
    actPCGwvwVgpr = 0
    actPCMaxTempSgpr = 0
    actTempSgpr = 0
    if kernel["ActivationFuncCall"]:
      maxVw = max(vectorWidths)
      usage = activation.getAllGprUsage(kernel["ProblemType"]["ActivationComputeDataType"])
      actPCMaxTempVgpr = 0
      for _, gprs in usage.items():
        actPCMaxTempVgpr = max(actPCMaxTempVgpr, gprs["vgpr"])
        actPCMaxTempSgpr = max(actPCMaxTempSgpr, gprs["sgpr"])
      actPCGwvwVgpr = int(ceil(maxVw * kernel["ProblemType"]["ActivationComputeDataType"].numRegisters()))
      numTmpVgpr = max(numTmpVgpr, actPCMaxTempVgpr + actPCGwvwVgpr)
    tmpVgpr = self.vgprPool.checkOutAligned(numTmpVgpr, 2, "store tmps")

    bf16CVTVgprStruct = None
    bf16CVTVgpr       = None
    if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
      bf16CVTVgpr = self.vgprPool.checkOut(4)
      bf16CVTVgprStruct = self.BF16CVTVgprStruct(vgprBf16Temp=bf16CVTVgpr, vgprBf16Mask=(bf16CVTVgpr+1), \
                                                 vgprFp32Nan=(bf16CVTVgpr+2), vgprBf16Inc=(bf16CVTVgpr+3))

    # Add bias lds
    if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
      # Init bias Srd
      labelStr = self.labels.getNameInc("Bias")
      module.add(allocPostLoopSrdSuppress("Bias", labelStr))
      multiBiasTypeLabel = []
      for i in kernel["ProblemType"]["BiasDataTypeList"]:
        name = self.labels.getNameInc("Load_Bias%s"%i.toNameAbbrev())
        multiBiasTypeLabel.append(Label(name, ""))
      loadBiasEndLabel = Label(self.labels.getNameInc("Load_Bias_End"), "")
      multiBiasTypeLabel.append(loadBiasEndLabel)
      offsetVgpr  = self.vgprPool.checkOut(1, 1)
      tmpVgprRes = RegisterPoolResource(idx=tmpVgpr, size=4)
      with self.allocTmpSgpr(1, 1) as tmpSgprRes:
        if len(kernel["ProblemType"]["BiasDataTypeList"]) == 1:
          module.add(self.readBiasToLDS(kernel["ProblemType"]["BiasDataTypeList"][0], kernel, 1, offsetVgpr, tmpSgprRes.idx, tmpVgprRes))
        else:
          for i, label in enumerate(multiBiasTypeLabel[1:]):
            module.add(multiBiasTypeLabel[i])
            module.add(SCmpKLGU32(sgpr("BiasType"), kernel["ProblemType"]["BiasDataTypeList"][i].value, "BiasType != %u"%i))
            module.add(SCBranchSCC1(label.getLabelName(), "Branch if true"))
            module.add(self.readBiasToLDS(kernel["ProblemType"]["BiasDataTypeList"][i], kernel, 1, offsetVgpr, tmpSgprRes.idx, tmpVgprRes))
            module.add(SBranch(labelName=loadBiasEndLabel.getLabelName(), comment="Branch to load bias end"))
          module.add(loadBiasEndLabel)
      self.vgprPool.checkIn(offsetVgpr)

    activationSetPCStruct = None
    activationLabelList = None
    activationEnumStrList = None
    toActModuleList = None
    isInsertActFunctionCallAddrCalc = True
    if kernel["ActivationFuncCall"]:
      sgprOffsetActivation = self.sgprPool.checkOutAligned(2, 2)
      sgprOffsetBack = self.sgprPool.checkOutAligned(2, 2)
      activationSetPCStruct = self.ActivationSetPCStruct(sgprOffsetActivation=sgprOffsetActivation, \
        sgprOffsetBack=sgprOffsetBack, vgprActCopy=tmpVgpr)
      activationCDataType = kernel["ProblemType"]["ActivationComputeDataType"]
      activationLabelList = {}
      toActModuleList = {}
      activationEnumStrList = ActivationType.getEnumStrList(activationCDataType)
      for gwvw in vectorWidths:
        if gwvw in activationLabelList:
          continue
        activationLabelList[gwvw] = []
        toActModuleList[gwvw] = []
        for enumStr in activationEnumStrList:
          name = self.labels.getNameInc("Activation_%s_VW%u"% (enumStr.capitalize(), gwvw))
          activationLabelList[gwvw].append(Label(name, ""))
          toActModuleList[gwvw].append(Label("To_%s"% (name), ""))
      # Add branch here if all elements are identical
      if vectorWidths.count(vectorWidths[0]) == len(vectorWidths):
        isInsertActFunctionCallAddrCalc = False
        module.add(self.insertActFunctionCallAddrCalc(activationSetPCStruct.sgprOffsetActivation, \
          vectorWidths[0], toActModuleList, activationEnumStrList, activationLabelList))

    ########################################
    # Sgprs

    # allocate tmps for the store header (before the batch implementations)
    # branch B1 or B0
    betaLabel = Label(self.labels.getNameInc("GW_Beta"), "")

    betaModules = Module("Betas")
    currentInstLength = 0
    for idx0 in reversed(range(len(betas))):
      beta = betas[idx0]
      betaModule = Module("Beta_%u"%idx0)
      # start B1
      if beta:
        betaModule.add(betaLabel)

      mod_pos = len(betaModule.items())
      # by now we either jumped to E1 or stayed at E0
      for idx1 in reversed(range(len(edges))):
        edge = edges[idx1]
        edgeModule = Module("edge_%u"%idx1)
        edgeModule.add(writeLabels[beta][edge])

        # for storeRemap edge case, non-beta still can enable vector stores
        if kernel["StoreRemapVectorWidth"] and not beta:
          edgeI = False
        else:
          edgeI = edge
        #edgeI = True  # set to True to disable vector stores
        gwvw = vectorWidths[edgeI]

        #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

        ########################################
        # Calculate Vgprs for Write Batching
        ########################################

        ss = StoreState(self, kernel, gwvw, edge, beta, atomic, elements[edgeI])

        # how many vgprs are needed for zero elements
        # 2 for addressC in vgpr for addition - already checked out
        # 2 for coord0,1 of thread - already checked out
        # 2 for tmp - already checked out

        # 5 = how many vgprs are needed per element (flat)
        #  - 2 for addr
        #  - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
        #  - if beta gwvw*rpe for new value
        #  - if atomic 2*rpe for old and cmp values

        #print self.vgprPool.state()
        # Use VGPR up to next occupancy threshold:
        maxVgprs = self.getMaxRegsForOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
                                              self.getLdsSize(kernel), self.agprPool.size(), self.states.unifiedVgprRegs)
        if self.states.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
          _growPool(self.vgprPool, self.vgprPool.size()-self.vgprPool.available(), maxVgprs, 1, \
            "grow-pool up to next occupancy for GlobalWrite")
        # Get numVgprAvailable
        numVgprAvailable = self.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)

        # Grow the register pool if needed - we need enough regs for at least one element
        # Unfortunate since this means the write logic is setting the VGPR requirement
        # for the entire kernel but at least we have a functional kernel.
        # Before growing the pool, see if we can shrink the write vector width instead?
        # TODO : the vgprSerial is needed for-ever and if we grow here will split the
        # range of the tmps.  Maybe want to move vgprSerial to first vgpr?

        # TODO: Minimum elems for StoreRemap
        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        minElements = 2 if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()) else 1
        minNeeded = minElements * ss.numVgprsPerElement
        shrinkDb = 0
        if shrinkDb:
          print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)
        if numVgprAvailable < minNeeded:
          gwvwOrig = gwvw
          currentOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
              self.vgprPool.size(), self.agprPool.size(), self.states.unifiedVgprRegs)
          futureOccupancy = self.getOccupancy(kernel["NumThreads"], self.getLdsSize(kernel), \
              self.vgprPool.size() - numVgprAvailable + minNeeded, self.agprPool.size(), self.states.unifiedVgprRegs)

          if shrinkDb:
            print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                % (currentOccupancy, futureOccupancy, self.vgprPool.size(), \
                  numVgprAvailable, minElements*ss.numVgprsPerElement))
          if futureOccupancy > currentOccupancy:
            if shrinkDb:
              print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                    (self.states.kernelName))
              print("   numVgprAvailable=", numVgprAvailable, \
                    "numVgprsPerElement=", ss.numVgprsPerElement, "atomic=", atomic, \
                    "beta=", beta, "gwvw=", gwvw)
          elif gwvw != gwvwOrig:
            ss.cfg.gwvw = gwvw # make both representations consistent
            if shrinkDb:
              print2("info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                  % (self.states.kernelName, gwvwOrig, gwvw, currentOccupancy))

          if numVgprAvailable < minElements*ss.numVgprsPerElement:
            print2("info: growing pool += %d * %d for GlobalWrite\n" \
                % (minElements,ss.numVgprsPerElement))
            print2(self.vgprPool.state())
            _growPool(self.vgprPool, 0, minElements, ss.numVgprsPerElement, \
              "grow-pool for GlobalWrite")
            numVgprAvailable = self.vgprPool.available()
            print2(self.vgprPool.state())

        # set atomicW after we potentially resize GWVW
        atomicW = min(gwvw, self.getVectorAtomicWidth(kernel))

        # print("NumVgprAvailable", numVgprAvailable)
        if ss.numVgprsPerElement:
          numElementsPerBatch = numVgprAvailable // ss.numVgprsPerElement
        else:
          numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

        assert(self.states.c.numVgprValu % gwvw == 0) # sanity check

        numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

        if shrinkDb:
          print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
              "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
        if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
          numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
          # only do an even number of halves - since these share hi/lo pieces of some registers?
          if numElementsPerBatch > 1:
            numElementsPerBatch = int(numElementsPerBatch/2)*2
          elif not kernel["EnableMatrixInstruction"]:
            # The globalWriteBatch routine below can't handle odd elements per batch
            # and 0 elements per batch is illegal.
            # so if we don't have *GPR resources to handle a larger batch then need
            # to mark overflowedResources rather than generate a kernel that won't work.
            # It might be possible to fix globalWriteBatch to handle this case but these
            # are likely to be low-performing so likely not worth optimizing.
            if shrinkDb:
              print("WARNING: half requires at least two elements per batch")
            self.states.overflowedResources = 3

        assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%self.states.kernelName

        #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
        if atomic and (ss.optSingleColVgpr or ss.optSharedColVgpr):
          # hack to avoid re-using address vgpr across rows
          # atomics need to perform several memory operations
          # if the batch spans multiple rows, need multiple address vgpr
          # which is not currently supported in the two opt*ColVgpr modes
          firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
          numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

        # check best numElementsPerBatch to handle a column block
        # elements of column block must be multiple size of numElementsPerBatch
        if kernel["StoreRemapVectorWidth"]:
          firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0] # format for element = (tt1, tt0, vc1, vc0)
          # find the largest factor and smaller than numElementPerBatch
          nBatchesPerRow = 1
          for d in range(1, len(firstRow)+1):
            largestFactor = len(firstRow)//d
            if len(firstRow)%d == 0 and largestFactor <= numElementsPerBatch:
              numElementsPerBatch = largestFactor
              nBatchesPerRow = d
              break

        # if no atomics and no edge, then write whole vectors
        #if not atomic and not edge:
        #  numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
        #  #print "  NumVectorsPerBatch", numVectorsPerBatch
        #  numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
        numBatches = max(1, ceilDivide(len(elements[edgeI]),numElementsPerBatch))

        numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

        if activationLabelList and isInsertActFunctionCallAddrCalc:
          assert activationSetPCStruct, activationEnumStrList and activationLabelList and toActModuleList
          numSgprs = max(actPCMaxTempSgpr, numSgprs)
          edgeModule.add(self.insertActFunctionCallAddrCalc(activationSetPCStruct.sgprOffsetActivation, \
            gwvw, toActModuleList, activationEnumStrList, activationLabelList, \
            idx0, idx1))

        if self.db["PrintStoreRegisterDb"]:
          print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", ss.numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
          print ("numSgprs=", numSgprs, "sgprPool.size()=", self.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
                "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
          print(self.sgprPool.state())
        edgeModule.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
            (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
        #edgeModule.addComment("storeStats, %d, %d, %d"% (edgeI, numSgprs, numElementsPerBatch))
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        # Activation
        actLoopEndLabel, actLoopLabelModules, actLoopEnumStrList = self.initActivationLoop(kernel, beta, edge)
        actLoopModuleList = []
        actLoopModuleCodeLength = []
        with self.allocTmpSgpr(numSgprs, 2) as tmpSgprRes:
          for index, activationLabelModule in enumerate(actLoopLabelModules):
            actLoopModule = Module("Activation Loop %s"%index)
            activationTypeStr = actLoopEnumStrList[index]
            if activationLabelModule:
              actLoopModule.add(activationLabelModule)

            tmpSgpr = tmpSgprRes.idx
            actTempSgpr = tmpSgpr # Get sgpr start address, should always be the same
            elementSgprs = tmpSgpr + ss.cfg.numTempSgprPerBatch
            codeAccVgprRead = deepcopy(self.codes.accVgprRead) if self.states.serializedStore else None
            codeMulAlpha    = deepcopy(self.codes.mulAlpha) if self.states.serializedStore else None

            self.alphaBeforeLoadC = False
            if kernel["MIArchVgpr"] and applyAlpha:
              codeAccVgprRead = None

              #Only apply when 2 wave optimization features are enabled
              if (kernel["StorePriorityOpt"] or kernel["StoreSyncOpt"]) and beta:
                self.alphaBeforeLoadC = True
            else:
              codeMulAlpha = None

            for batchIdx in range(0, numBatches):
              elementStartIdx = batchIdx * numElementsPerBatch
              elementStopIdx = min( elementStartIdx + numElementsPerBatch, len(elements[edgeI]) )
              elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
              #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,ss.numVgprsPerElement ))
              # elementVgprs can be large and should be perfectly tuned to the number of available
              # VGPRS.  We do not want to accidentally overflow and grow the pool here:

              if kernel["StoreRemapVectorWidth"]:
                #Indication if this batch is last batch for this column block shape
                self.StoreRemapLastBatch = 1 if (batchIdx+1) % nBatchesPerRow == 0 else 0

              actLoopModule.add(self.globalWriteBatch(kernel, tPA, tPB, activation, ss, batchIdx, \
                  applyAlpha, beta, edge, atomic, gwvw, atomicW, \
                  elementsThisBatch, self.vgprs.addrD, self.vgprs.addrC, self.vgprs.addrBias, self.vgprs.addrScaleD, \
                  tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, \
                  elementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha))

            ss.resetState()
            actLoopModuleList.append(actLoopModule)
            actLoopModuleCodeLength.append(actLoopModule.countType(Instruction))

        if len(actLoopLabelModules) > 1:
          actInstCounter = 0
          # Add activation branch
          for index, actLoopLabelModule in enumerate(actLoopLabelModules):
            enumIndex = ActivationType.getEnumIndex(actLoopEnumStrList[index])
            edgeModule.add(SCmpKEQU32(sgpr("ActivationType"), enumIndex, "activationType == %u"%enumIndex))
            if actInstCounter >= 16384:
              edgeModule.add(self.longBranchScc1(actLoopLabelModule, posNeg=1, comment="Branch if true"))
            else:
              edgeModule.add(SCBranchSCC1(actLoopLabelModule.getLabelName(), "Branch if true"))
            actInstCounter += actLoopModuleCodeLength[index]
          # Add jump to activation end
          for index, _ in enumerate(actLoopLabelModules):
            actLoopModule = actLoopModuleList[index]
            if (index < (len(actLoopLabelModules) - 1)):
              if actInstCounter >= 16384:
                with self.allocTmpSgpr(3) as tmpSgprInfo:
                  SLongBranchPositive(actLoopEndLabel, tmpSgprInfo)
              else:
                actLoopModule.add(SBranch(labelName=actLoopEndLabel.getLabelName()))
            actInstCounter -= actLoopModuleCodeLength[index]

        # Append to edgeModule
        for actLoopModule in actLoopModuleList:
          edgeModule.appendModule(actLoopModule)
        # Add actLoopEndLabel if needed
        if len(actLoopLabelModules) > 1:
          edgeModule.add(actLoopEndLabel)

        if currentInstLength >= 16384:
          with self.allocTmpSgpr(3) as tmpSgprInfo:
            edgeModule.add(SLongBranchPositive(endLabel, tmpSgprInfo, comment="jump to end"))
        else:
          edgeModule.add(SBranch(labelName=endLabel.getLabelName(), comment="jump to end"))
        currentInstLength += edgeModule.countType(Instruction)
        betaModule.add(edgeModule, pos=mod_pos)
        del ss
      ########################################
      # branch if Edge0 or Edge1
      if False in edges and True in edges:
        isLongBranch = True if currentInstLength >= 16384 else False
        with self.allocTmpSgpr(4) as tmpSgprInfo:
          checkIsEdge = betaModule.add(self.checkIsEdge(kernel, tmpSgprInfo.idx, \
            writeLabels[beta][True], isLongBranch=isLongBranch), pos=mod_pos)
          currentInstLength += checkIsEdge.countType(Instruction)
      betaModules.add(betaModule, pos=0)

    # Check if branch exceeds
    if False in betas and True in betas:
      isBetaLongBranch = False
      def findInstCount(module, targetItem, count):
        for inst in module.items():
          if isinstance(inst, Module):
            count, found = findInstCount(inst, targetItem, count)
            if found:
              return count, True
          elif (inst is targetItem):
            return count, True
          elif (not isinstance(inst, TextBlock)):
            count += 1
        return count, False
      count = 0
      count, found = findInstCount(betaModules, betaLabel, count)
      if found:
        if count >= 16384:
          isBetaLongBranch = True
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          module.add(self.checkIsBetaZero(kernel, tmpSgprInfo.idx, betaLabel, isBetaLongBranch, posNeg=1))
    module.appendModule(betaModules)

    if activationLabelList:
      assert activationEnumStrList and activationSetPCStruct
      for key, activationLabelModules in activationLabelList.items():
        gwvw = key
        actModules = Module(getActivationFunctionModuleName(gwvw, \
          activationSetPCStruct.vgprActCopy, tmpVgpr, actTempSgpr))
        for index, activationLabelModule in enumerate(activationLabelModules):
          actModule = Module(activationLabelModule.getLabelName())
          actModule.add(activationLabelModule)
          activationTypeStr = activationEnumStrList[index]
          if self.insertActivationAfterPacked(kernel, activationTypeStr):
            actModule.appendModule(self.getActivationDestDataType(kernel, activation, \
              activationTypeStr, gwvw, activationSetPCStruct.vgprActCopy, (tmpVgpr + actPCGwvwVgpr), \
              actTempSgpr))
          else:
            actModule.appendModule(self.getActivationActivationComputeType(kernel, activation, \
              activationTypeStr, gwvw, activationSetPCStruct.vgprActCopy, (tmpVgpr + actPCGwvwVgpr), \
              actTempSgpr))
          actModule.add(SSetPCB64(src=sgpr(activationSetPCStruct.sgprOffsetBack,2)))
          actModules.add(actModule)
        module.add(actModules)
      self.sgprPool.checkIn(activationSetPCStruct.sgprOffsetActivation)
      self.sgprPool.checkIn(activationSetPCStruct.sgprOffsetBack)

    # End label
    module.add(endLabel)
    self.vgprPool.checkIn(tmpVgpr)
    if bf16CVTVgpr is not None:
      self.vgprPool.checkIn(bf16CVTVgpr)
    return module

  ##############################################################################
  # chooseGlobalRead :
  # create the load instruction for requested vector width and other parms
  # return an Inst class
  #
  # bpl = bytes per load op
  ##############################################################################
  def chooseGlobalRead(self, useBuffer, bpl, destVgpr, \
                       addr0, addr1, soffset, offset, \
                       glc=False, slc=False, lds=False, hi16=0, comment="load C"):
  # rpv = regs per vector
    rpv = bpl/4.0

    if useBuffer:
      rv = Module("Global Read")
      mubuf = MUBUFModifiers(offen=True, offset12=offset, glc=glc, slc=slc, lds=lds)

      # Nested buffer load implementation function for easy branching for soffset
      def bufferLoadImpl(soffset):
        nonlocal rv
        if bpl==1 and hi16:
          rv.add(BufferLoadD16HIU8(dst=vgpr(destVgpr, rpv*4), vaddr=addr0, saddr=addr1, \
                                  soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==1 and not hi16:
          rv.add(BufferLoadD16U8(dst=vgpr(destVgpr, rpv*4), vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==2 and hi16:
          rv.add(BufferLoadD16HIB16(dst=vgpr(destVgpr, rpv*2), vaddr=addr0, saddr=addr1, \
                                    soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==2 and not hi16:
          rv.add(BufferLoadD16B16(dst=vgpr(destVgpr, rpv*2), vaddr=addr0, saddr=addr1, \
                                  soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==4:
          rv.add(BufferLoadB32(dst=vgpr(destVgpr, rpv), vaddr=addr0, saddr=addr1, \
                              soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==8:
          rv.add(BufferLoadB64(dst=vgpr(destVgpr, rpv), vaddr=addr0, saddr=addr1, \
                              soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==16:
          rv.add(BufferLoadB128(dst=vgpr(destVgpr, rpv), vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        elif bpl==32:
          # split into two dwordx4 loads. Second load offset is +0.5 bpl
          mubuf.offset12 = (offset + bpl/2)

          rv = Module("emulated _buffer_load_b256")
          rv.add(BufferLoadB128(dst=vgpr(destVgpr, rpv/2), vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))
          rv.add(BufferLoadB128(dst=vgpr(int(destVgpr + rpv/2), rpv/2), vaddr=addr0, saddr=addr1, \
                                soffset=soffset, mubuf=mubuf, comment=comment))
          return rv
        else:
          assert 0, "chooseGlobalRead: bad bpl"

      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      if offset >= 4096:
        if soffset in (0, "0"):
          mubuf.offset12 = 0
          with self.allocTmpSgpr(1) as tmpSgprInfo:
            soffset = sgpr(tmpSgprInfo.idx)
            rv.add(SMovB32(dst=soffset, src=offset, comment="large offset"))
            bufferLoadImpl(soffset)
        else:
          assert 0, "offset too large and soffset set"
      else:
        bufferLoadImpl(soffset)
      return rv
    else:
      flat = FLATModifiers(glc=glc, slc=slc, lds=lds)
      if bpl==2 and hi16:
        return FlatLoadD16HIB16(dst=vgpr(destVgpr, rpv*2), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==2 and not hi16:
        return FlatLoadD16B16(dst=vgpr(destVgpr, rpv*2), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==4:
        return FlatLoadB32(dst=vgpr(destVgpr, rpv), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==8:
        return FlatLoadB64(dst=vgpr(destVgpr, rpv), vaddr=addr0, flat=flat, comment=comment)
      elif bpl==16:
        return FlatLoadB128(dst=vgpr(destVgpr, rpv), vaddr=addr0, flat=flat, comment=comment)
      else:
        assert 0, "chooseGlobalRead: bad bpl"

  ##############################################################################
  def chooseGlobalWrite(self, useBuffer, bps, srcVgpr, rpv, \
                        addr0, addr1, offset, glc=False, slc=False, hi16=0):
    """
    create the store instruction for requested vector width and other parms
    rpv = regs per vector
    """

    module = Module("chooseGlobalWrite %s -> %s (%s)"%(srcVgpr, addr0, addr1))

    def bufferStoreImpl(tmpSgpr, mubuf):
      if bps==1 and hi16:
        module.add(BufferStoreD16HIB16(src=vgpr(srcVgpr, rpv*4), vaddr=addr0, \
                                       saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      elif bps==1 and not hi16:
        module.add(BufferStoreB8(src=vgpr(srcVgpr, rpv*4), vaddr=addr0, \
                                 saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      elif bps==2 and hi16:
        module.add(BufferStoreD16HIB16(src=vgpr(srcVgpr, rpv*2), vaddr=addr0, \
                                       saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      elif bps==2 and not hi16:
        module.add(BufferStoreB16(src=vgpr(srcVgpr, rpv*2), vaddr=addr0, \
                                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      elif bps==4:
        module.add(BufferStoreB32(src=vgpr(srcVgpr, rpv), vaddr=addr0, \
                                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      elif bps==8:
        module.add(BufferStoreB64(src=vgpr(srcVgpr, rpv), vaddr=addr0, \
                                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      elif bps==16:
        module.add(BufferStoreB128(src=vgpr(srcVgpr, rpv), vaddr=addr0, \
                                   saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      elif bps == 32:
        # split into two dwordx4 loads. Offset the second by +0.5 bps
        module.add(BufferStoreB128(src=vgpr(srcVgpr, rpv/2), vaddr=addr0, \
                                   saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))

        module.add(BufferStoreB128(src=vgpr(int(srcVgpr +rpv/2), rpv/2), vaddr=addr0, \
                  saddr=addr1, soffset=tmpSgpr, mubuf=mubuf, comment="store D"))
      else:
        assert 0, "bad bps"

    if useBuffer:
      mubuf = MUBUFModifiers(offen=True, offset12=offset, glc=glc, slc=slc)
      # buffer_load offset field is 12-bit.
      # if offset >= 4096, use soffset instead
      if offset >= 4096:
        with self.allocTmpSgpr(1) as tmpSgprInfo:
          tmpSgpr = sgpr(tmpSgprInfo.idx)
          module.add(SMovB32(dst=tmpSgpr, src=offset, comment="large offset"))
          mubuf.offen = False
          mubuf.offset12 = 0
          bufferStoreImpl(tmpSgpr, mubuf)
      else:
        bufferStoreImpl(0, mubuf)

    else:
      flat = FLATModifiers(glc=glc, slc=slc)
      if bps==2 and hi16:
        module.add(FlatStoreD16HIB16(vaddr=addr0, src=vgpr(srcVgpr*2), flat=flat, comment="store D"))
      elif bps==2 and not hi16:
        module.add(FlatStoreD16B16(vaddr=addr0, src=vgpr(srcVgpr, rpv*2), flat=flat, comment="store D"))
      elif bps==4:
        module.add(FlatStoreB32(vaddr=addr0, src=vgpr(srcVgpr, rpv), flat=flat, comment="store D"))
      elif bps==8:
        module.add(FlatStoreB64(vaddr=addr0, src=vgpr(srcVgpr, rpv), flat=flat, comment="store D"))
      elif bps==16:
        module.add(FlatStoreB128(vaddr=addr0, src=vgpr(srcVgpr, rpv), flat=flat, comment="store D"))
      else:
         assert 0, "bad bps"

    return module

  def addBiasGlobalLoad(self, dataType, kernel, biasVgpr, addr0, addr1, offset, gwvw):
    """
    Add bias for the element with addrCalc, elementIdx, and biasVgpr.
    biasVgpr is one or more vgpr :temp vGPR ( = gwvw * numbytes // 4 + 1 if cvt is needed)
    """
    # Add bias here
    module = Module("addBias")
    if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
      bps = dataType.numBytes() * gwvw

      useBuffer = kernel["BufferLoad"]
      if dataType.isHalf() or dataType.isBFloat16():
        module.add(self.chooseGlobalRead(useBuffer, bps, biasVgpr, \
                          addr0, addr1, soffset=0, offset=offset, hi16=0, comment="load bias"))
      elif dataType.isInt32() or dataType.isSingle():
        module.add(self.chooseGlobalRead(useBuffer, bps, biasVgpr, \
                          addr0, addr1, soffset=0, offset=offset, comment="load bias"))
      elif dataType.isDouble() or dataType.isSingleComplex() :
        module.add(self.chooseGlobalRead(useBuffer, bps, biasVgpr, \
                          addr0, addr1, soffset=0, offset=offset, comment="load bias"))
      else:
        printExit("Unsupported bias type %s."%(str(dataType)))
    return module

  def addScaleDLoad(self, kernel, ss, addrCalc, scaleDVgpr):
    """
    Add scaleD for the element with addrCalc, elementIdx, and scaleDVgpr.
    scaleDVgpr is one or more vgpr :temp vGPR ( = gwvw * numbytes // 4 + 1 if cvt is needed)
    """
    module = Module("addScaleD")
    if kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
      bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * ss.cfg.gwvw
      if kernel["BufferLoad"]:
        addr0 = vgpr(addrCalc.addrScaleDVgpr)
        addr1 = sgpr("SrdScaleD", 4)
      else:
        addr0 = vgpr(addrCalc.addrScaleDVgpr,2)
        addr1 = ""

      useBuffer = kernel["BufferLoad"]

      if kernel["ProblemType"]["ComputeDataType"].isHalf() or kernel["ProblemType"]["ComputeDataType"].isBFloat16():
        module.add(self.chooseGlobalRead(useBuffer, bps, scaleDVgpr, \
                          addr0, addr1, soffset=0, offset=addrCalc.scaleDOffset, hi16=0, comment="load scaleDH"))
      elif kernel["ProblemType"]["ComputeDataType"].isInt32() or kernel["ProblemType"]["ComputeDataType"].isSingle():
        module.add(self.chooseGlobalRead(useBuffer, bps, scaleDVgpr, \
                          addr0, addr1, soffset=0, offset=addrCalc.scaleDOffset, comment="load scaleDI"))
      elif kernel["ProblemType"]["ComputeDataType"].isDouble() or kernel["ProblemType"]["ComputeDataType"].isSingleComplex() :
        module.add(self.chooseGlobalRead(useBuffer, bps, scaleDVgpr, \
                          addr0, addr1, soffset=0, offset=addrCalc.scaleDOffset, comment="load scaleD"))
      else:
        printExit("Unsupported scaleD type %s."%(str(kernel["ProblemType"]["ComputeDataType"])))

    return module

  def addBiasLoad(self, dataType, kernel, ss, addrCalc, biasVgpr, isLocal=False):
    if isLocal and kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
      module = Module("addBias")
      dst = vgpr(biasVgpr)
      src = vgpr(addrCalc.addrBiasVgpr)
      ds = DSModifiers(offset=addrCalc.biasOffset)
      bps = dataType.numBytes() * ss.cfg.gwvw
      if bps==2:
        module.add(DSLoadU16(dst=dst, src=src, readToTempVgpr=False, ds=ds, comment="load bias"))
      elif bps==4:
        module.add(DSLoadB32(dst=dst, src=src, readToTempVgpr=False, ds=ds, comment="load bias"))
      elif bps==8:
        module.add(DSLoadB64(dst=vgpr(biasVgpr, 2), src=src, readToTempVgpr=False, ds=ds, comment="load bias"))
      elif bps==16:
        module.add(DSLoadB128(dst=vgpr(biasVgpr, 4), src=src, readToTempVgpr=False, ds=ds, comment="load bias"))
      return module

    if kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
      if kernel["BufferLoad"]:
        addr0 = vgpr(addrCalc.addrBiasVgpr)
        addr1 = sgpr("SrdBias", 4)
      else:
        addr0 = vgpr(addrCalc.addrBiasVgpr,2)
        addr1 = ""
    else:
      addr0 = ""
      addr1 = ""
    return self.addBiasGlobalLoad(dataType, kernel, biasVgpr, addr0, addr1, addrCalc.biasOffset, ss.cfg.gwvw)

  ##############################################################################
  def addStore(self, kernel, ss, addrCalc, sumIdx, tmpS01, edge):
    """
    Add stores for the element with addrCalc and sumIdx.
    tmpS01 is a single :temp sGPR
    """
    module = Module("addStore sumIdx %s"%(str(sumIdx)))
    if self.do["GlobalWrite"]:
      # perform vector stores here, so no VI indexing.
      # if GWVW > Vw, might need to support loops to
      # implement wider stores
      isGlc = False
      isSlc = False
      if kernel["NonTemporalD"]%2==1:
        isGlc = True
      if kernel["NonTemporalD"]//2==1:
        isSlc = True

      bps = self.states.bpeCexternal * ss.cfg.gwvw
      rpv = self.states.bpeCexternal * ss.cfg.gwvw / self.states.bpr

      if kernel["BufferStore"]:
        addr0 = vgpr(addrCalc.addrDVgpr)
        addr1 = sgpr("SrdD", 4)
      else:
        addr0 = vgpr(addrCalc.addrDVgpr,2)
        addr1 = ""

      useBuffer = kernel["BufferStore"]
      if ss.optSrdIncForRow and addrCalc.rowInc:
        module.add(addrCalc.incrementToNextRow(kernel, "D", ss, tmpS01))
      if kernel["ProblemType"]["DestDataType"].isHalf() or kernel["ProblemType"]["DestDataType"].isBFloat16():

        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (H,H,H,H,H,H), internal H
          module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx//2, rpv, \
                           addr0, addr1, addrCalc.globalOffset, isGlc, isSlc, hi16=sumIdx%2))
        else:
          # (B,B,B,B,S,S), internal S
          # (H,H,H,H,H,H), internal S
          # (H,H,H,H,S,S), internal S
          module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                         addr0, addr1, addrCalc.globalOffset, isGlc, isSlc, hi16=0))
      elif kernel["ProblemType"]["DestDataType"].isInt32() or kernel["ProblemType"]["DestDataType"].isSingle():
        module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                       addr0, addr1, addrCalc.globalOffset, isGlc, isSlc))
      elif kernel["ProblemType"]["DestDataType"].isDouble() or kernel["ProblemType"]["DestDataType"].isSingleComplex():
        module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx*2, rpv, \
                       addr0, addr1, addrCalc.globalOffset, isGlc, isSlc))
      elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
        rps = kernel["ProblemType"]["DestDataType"].numRegisters()
        module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx*rps, rpv, \
                       addr0, addr1, addrCalc.globalOffset, isGlc, isSlc))
      elif kernel["ProblemType"]["DestDataType"].isInt8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          module.add(self.chooseGlobalWrite(useBuffer, bps, sumIdx, rpv, \
                         addr0, addr1, addrCalc.globalOffset, isGlc, isSlc))
    return module

  ##############################################################################
  # Global Read C Input
  ##############################################################################
  def readCInput(self, kernel, ss, addrCalc, vc0, data, gwvw, addr, tmpS01):
    module = Module("readCInput")
    bps = kernel["ProblemType"]["DestDataType"].numBytes() * gwvw
    useBuffer = kernel["BufferStore"]

    if kernel["BufferStore"]:
      addr0 = vgpr(addr)
      addr1 = sgpr("SrdC", 4)
    else:
      addr0 = vgpr(addr,2)
      addr1 = ""

    isGlc = True if kernel["NonTemporalC"]%2==1 else False
    isSlc = True if kernel["NonTemporalC"]//2==1 else False

    if ss.optSrdIncForRow and addrCalc.rowInc:
      module.add(addrCalc.incrementToNextRow(kernel, "C", ss, tmpS01))

    if kernel["ProblemType"]["DestDataType"].isHalf():
      module.add(self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                glc=isGlc, slc=isSlc, lds=False, hi16=vc0 % 2,
                comment="load C for beta calc"))
    elif kernel["ProblemType"]["DestDataType"].isInt8():
     module.add(self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                glc=isGlc, slc=isSlc, lds=False, \
                #hi16=vc0 % 4,
                comment="load C for beta calc"))
    elif kernel["ProblemType"]["DestDataType"].isBFloat16() or \
         kernel["ProblemType"]["DestDataType"].isInt32() or \
         kernel["ProblemType"]["DestDataType"].isSingle() or \
         kernel["ProblemType"]["DestDataType"].isDouble() or \
         kernel["ProblemType"]["DestDataType"].isSingleComplex() or \
         kernel["ProblemType"]["DestDataType"].isDoubleComplex():
      module.add(self.chooseGlobalRead(useBuffer, bps, data, \
                addr0, addr1, soffset=0, offset=addrCalc.globalOffset, \
                glc=isGlc, slc=isSlc, lds=False, \
                comment="load C for beta calc"))

    return module

  ##############################################################################
  # Global Write Batch
  ##############################################################################
  def globalWriteBatch(self, kernel, tPA, tPB, activation, ss: StoreState, batchIdx, \
      applyAlpha, beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrD, addrC, addrBias, addrScaleD, \
      tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, \
      batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha) -> Module:
      packdata = Component.PackData.find(self)
      gwriter  = Component.GlobalWriteComponents.find(self)
      return gwriter(kernel, tPA, tPB, activation, ss, \
        batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
        batchElements, addrD, addrC, addrBias, addrScaleD, tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, \
        batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, packdata, self)

  ##############################################################################
  def openPrefetchGlobalRead2(self, kernel):
    imod = Module()
    loopCounter = self.loopCounter(kernel, self.states.unrollIdx)
    imod.add(SCmpEQU32(src0=loopCounter, src1=hex(1), comment="PGR=2 but only 1 loop"))
    skipPGR2 = Label(self.labels.getName("skipPGR2"), "")
    imod.add(SCBranchSCC1(labelName=skipPGR2.getLabelName(), comment="PGR=2 but only 1 loop"))
    return imod

  def closePrefetchGlobalRead2(self):
    imod = Module()
    skipPGR2 = Label(self.labels.getName("skipPGR2"), "")
    imod.add(skipPGR2)
    return imod

  ########################################
  # Bias related
  ########################################
  def readBiasToLDS(self, biasDataType, kernel, gwvw, offsetVgpr, tmpSgpr, tmpVgpr1Res: RegisterPoolResource):
    assert gwvw == 1
    assert tmpVgpr1Res.size >= gwvw * kernel["ProblemType"]["ComputeDataType"].numRegisters()
    # Params
    biasBpe = int(self.states.bpr * biasDataType.numRegisters())
    module = Module("ReadBiasToLds")
    module.addComment2("Read Bias to LDS")
    # Recalculate bias length
    module.add(SMulI32(dst=sgpr("SrdBias+2"), src0=hex(biasBpe), src1=sgpr("SrdBias+2"), comment="scaled by BPE"))
    # Calculate global offset- macro tile 0 part
    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
    module.add(VAddU32(dst=vgpr(offsetVgpr), src0=sgpr(tmpSgpr), src1=vgpr("Serial"), comment="coord 0 = wgp0 * MT0 + thread offset"))
    module.add(VLShiftLeftB32(dst=vgpr(offsetVgpr), \
                              shiftHex=hex(log2(biasBpe)), \
                              src=vgpr(offsetVgpr), \
                              comment="Global bias address scaled by BPE"))
    # Offset
    numVgprs  = int(ceil(biasDataType.numRegisters() * gwvw))
    reg = biasDataType.numRegisters() if biasDataType.numRegisters() >= kernel["ProblemType"]["ComputeDataType"].numRegisters() \
      else kernel["ProblemType"]["ComputeDataType"].numRegisters()
    shiftOffset  = (gwvw * reg - numVgprs)
    # global load
    tmpVgpr1 = tmpVgpr1Res.idx
    addr0 = vgpr(offsetVgpr)
    addr1 = sgpr("SrdBias", 4)
    divisor = kernel["SubGroup0"] * kernel["SubGroup1"]
    turn    = ceil(kernel["MacroTile0"] / (divisor * gwvw))
    offset  = (divisor * gwvw) * biasBpe
    tmpVgprN = tmpVgpr1
    for i in range(turn):
      if i != (turn - 1):
        module.add(VAddU32(dst=vgpr(offsetVgpr), src0=offset, src1=vgpr(offsetVgpr), comment="add subgroup offset"))
      module.add(self.addBiasGlobalLoad(biasDataType, kernel, tmpVgprN + shiftOffset, addr0, addr1, 0, gwvw))
      tmpVgprN += 1
    # Local write
    module.add(VLShiftLeftB32(dst=vgpr(offsetVgpr), \
                              shiftHex=hex(log2(self.states.bpeCinternal)), \
                              src=vgpr("Serial"), \
                              comment="Local bias address scaled by BPE"))
    offset  = (divisor * gwvw) * self.states.bpeCinternal
    tmpVgprN = tmpVgpr1
    for i in reversed(range(turn)):
      if i != (turn - 1):
        module.add(VAddU32(dst=vgpr(offsetVgpr), src0=offset, src1=vgpr(offsetVgpr), comment="add subgroup offset"))
      module.add(SWaitCnt(vmcnt=i, comment="wait for bias load"))
      bps = kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw
      ds  = DSModifiers(offset=0)
      dst = vgpr(offsetVgpr)
      for vi in range(gwvw):
        # Does not support hi/lo yet
        shiftOffset2 = shiftOffset + int(vi * biasDataType.numRegisters())
        if kernel["ProblemType"]["ComputeDataType"].isSingle():
          if biasDataType.isHalf():
            module.add(VCvtF16toF32(dst=vgpr(tmpVgprN + vi), src=vgpr(tmpVgprN + shiftOffset2), comment="convert to FP32"))
          elif biasDataType.isBFloat16():
            module.add(VCvtBF16toFP32(dst=(tmpVgprN + vi), src=(tmpVgprN + shiftOffset2), vgprMask=None, vi=0))
          elif biasDataType == kernel["ProblemType"]["ComputeDataType"]:
            pass # Same, no need to convert
          else:
            printExit("Unrecognized bias type %s."%str(biasDataType))
        else:
          printExit("Does not support ComputeDataType != float")
      if bps==2:
        module.add(DSStoreB16(dstAddr=dst, src=vgpr(tmpVgprN), ds=ds, comment="store bias"))
      elif bps==4:
        module.add(DSStoreB32(dstAddr=dst, src=vgpr(tmpVgprN), ds=ds, comment="store bias"))
      elif bps==8:
        module.add(DSStoreB64(dstAddr=dst, src=vgpr(tmpVgprN, 2), ds=ds, comment="store bias"))
      else:
        assert 0
    # We move lgkmcnt and s_barrier before local load
    return module

  ########################################
  # Activation related
  ########################################
  def initActivationLoop(self, kernel, beta, edge):
    # Create a suffix and check if the string exists
    activationLabelSuffix = self.labels.getNameInc( \
      "%s%s"%("_Beta" if beta else "", "_Edge" if edge else ""))
    activationCDataType = kernel["ProblemType"]["ActivationComputeDataType"]
    activationEndLabel = Label("Activation_End%s"%activationLabelSuffix, "")
    activationLabelModules = []
    activationEnumStrList = []
    if kernel["ActivationFuncCall"]:
      activationLabelModules.append("")
      activationEnumStrList.append("none")
    elif ((kernel["GlobalSplitU"] == 1) and kernel["ActivationFused"]) and \
      (kernel["ProblemType"]["ActivationType"] != 'none'):
      if kernel["ProblemType"]["ActivationType"] == 'all':
        activationEnumStrList = ActivationType.getEnumStrList(activationCDataType)
        for _, enumStr in enumerate(activationEnumStrList):
          activationLabelModule = Label("Activation_%s%s"% (enumStr.capitalize(), activationLabelSuffix), "")
          activationLabelModules.append(activationLabelModule)
      else:
        activationEnumStrList.append(str(kernel["ProblemType"]["ActivationType"]).lower())
    else:
      activationLabelModules.append("")
      activationEnumStrList.append("none")
    return activationEndLabel, activationLabelModules, activationEnumStrList

  def insertActFunctionCallAddrCalc(self, sgprOffset, gwvw, \
    toActModuleList, activationEnumStrList, activationLabelList, \
    betaIdx = -1, edgeIdx = -1):
    activationLabelModules = activationLabelList[gwvw]
    module = Module(getActivationBranchModuleName())
    setAddrEndLabel = Label(self.labels.getNameInc("ActivationSetPCAddrEnd"), "")
    toActModules = deepcopy(toActModuleList[gwvw])
    for index, toActModule in enumerate(toActModules):
      if betaIdx >= 0 and edgeIdx >= 0:
        toActModule.label = self.labels.getNameInc(toActModule.label + "_beta_%u_edge_%u"%(betaIdx, edgeIdx))
      if index != 0:
        enumIndex = ActivationType.getEnumIndex(activationEnumStrList[index])
        module.add(SCmpKEQU32(sgpr("ActivationType"), enumIndex, "activationType == %u"%enumIndex))
        module.add(SCBranchSCC1(toActModule.getLabelName(), "Branch if true"))
    for index, activationLabelModule in enumerate(activationLabelModules):
      toActModule = toActModules[index]
      module.add(toActModule)
      with self.allocTmpSgpr(1) as tmpSgprInfo:
        module.addModuleAsFlatItems(SGetPositivePCOffset(sgprOffset, activationLabelModule, tmpSgprInfo))
      module.add(SBranch(setAddrEndLabel.getLabelName()))
    module.add(setAddrEndLabel)
    return module

  def insertActivationAfterPacked(self, kernel, activationTypeStr):
    result = False
    if ((kernel["ProblemType"]["ActivationType"] != 'none') and \
      (kernel["GlobalSplitU"] == 1) and kernel["ActivationFused"]):
      if kernel["ActivationFuncCall"]:
        return (not kernel["ProblemType"]["ActivationHPA"])
      elif kernel["ProblemType"]["ActivationHPA"]:
        # Still use BFloat16 for abs.
        if kernel["ProblemType"]["DestDataType"].isBFloat16() and (activationTypeStr == 'abs'):
          result = True
        elif kernel["ProblemType"]["DestDataType"].isHalf() and \
           ((activationTypeStr == 'abs') or (activationTypeStr == 'relu')):
          result = True
      else:
        result = True
    return result

  def getActivationDestDataType(self, kernel, activation, activationTypeStr: str, gwvw, \
  elementSumIdx, tmpVgpr, tmpSgpr):
    module = Module("ActivationAfterPack")
    for vi in range(0, gwvw):
      sumIdxV = elementSumIdx + vi
      if kernel["ProblemType"]["DestDataType"].isHalf() or \
          kernel["ProblemType"]["DestDataType"].isBFloat16():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # Generate single f16 code if edge is detected.
          if ((vi + 1) == gwvw) and ((gwvw % 2) == 1):
            activation.setUsePK(False)
          # Original packed route
          elif vi%2 == 1:
            assert (gwvw % 2 == 0)
          else:
            continue
          vgprIdx = elementSumIdx + vi//2
        else:
          if (sumIdxV % 2 != 0):
            continue
          vgprIdx = sumIdxV // 2
      elif kernel["ProblemType"]["DestDataType"].isSingle():
        vgprIdx = sumIdxV
      elif kernel["ProblemType"]["DestDataType"].isDouble():
        vgprIdx = sumIdxV * 2
      elif kernel["ProblemType"]["DestDataType"].isInt32():
        vgprIdx = sumIdxV
      else:
        raise RuntimeError("Unsupported data type %s for activation vgpr index."%str(self.states.kernel["ProblemType"]["DestDataType"]))
      # Here we still use DestDataType cause the data is ready to be written to global
      actModule = activation.getModule(self.states.kernel["ProblemType"]["DestDataType"], activationTypeStr, vgprIdx)
      module.add(activation.assignGpr(actModule, tmpVgpr, tmpSgpr))
      activation.setUsePK(True)
    return module

  def getActivationActivationComputeType(self, kernel, activation, activationTypeStr: str, gwvw, \
    elementSumIdx, tmpVgpr, tmpSgpr, satInt8=False):
    module = Module("ActivationBeforePack")
    if satInt8:
      activation.setSaturationForInt8(True)
    activation.setVgprPrefixFormat("ValuC+%u")
    for vi in range(0, gwvw):
      vgprIdx = elementSumIdx + vi - self.states.c.startVgprValu
      actModule = activation.getModule(kernel["ProblemType"]["ActivationComputeDataType"], activationTypeStr, vgprIdx)
      module.add(activation.assignGpr(actModule, tmpVgpr, tmpSgpr))
    activation.setSaturationForInt8(False)
    activation.setVgprPrefixFormat("")
    return module

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, addLabel=True):
    imod = Module()
    if addLabel:
      imod.add(Label("KernelEnd", ""))
    imod.add(SEndpgm(comment="Kernel End"))
    return imod

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    if self.vgprPool.size() > self.consts.maxVgprs:
      self.states.overflowedResources = 1
    elif self.sgprPool.size() > self.consts.maxSgprs:
      self.states.overflowedResources = 2

    if kernel["ScheduleIterAlg"] == 2 and \
        self.getOccupancy(kernel["NumThreads"], self.vgprPool.size(), \
        self.getLdsSize(kernel), self.agprPool.size(), self.states.unifiedVgprRegs) < 2:
      self.states.overflowedResources = 6

    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU // kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy // max(self.vgprPool.size(), self.agprPool.size())
    if numWorkGroupsPerCU < 1:
      self.states.overflowedResources = 4

    module = Module("functionSuffix")
    if self.states.overflowedResources:
      module.add(ValueEndif("overflowed resources"))

    self.vgprPool.checkFinalState()
    return module

  ##############################################################################
  # openOddNoLoadLoopForDTV
  # generate open code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def openOddNoLoadLoopForDTV(self, name):
    module = Module("openOddNoLoadLoopForDTV")
    evenStartLabelName = Label.getFormatting("EvenStart" + name)
    # odd exit check code
    # use OrigLoopCounter & 1
    with self.allocTmpSgpr(1) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      #scc0or1 = 0 if isNGLL else 1
      #oddOrEven = "Even" if isNGLL else "Odd"
      module.add(SAndB32(dst=sgpr(tmpSgpr), src0=sgpr("OrigLoopCounter"), src1=1, comment="test if OrigLoopCounter is Odd ?"))
      module.add(SCBranchSCC0(labelName=evenStartLabelName, comment="Skip odd code if OrigLoopCounter is Even"))

    return module

  ##############################################################################
  # closeOddNoLoadLoopForDTV
  # generate close code for DirectToVgpr + odd exit case in noLoadLoop code
  ##############################################################################
  def closeOddNoLoadLoopForDTV(self, name):
    module = Module("closeOddNoLoadLoopForDTV")
    evenStartLabelName = Label.getFormatting("EvenStart" + name)
    evenEndLabelName = Label.getFormatting("EvenEnd" + name)
    # odd exit code
    module.add(SBranch(labelName=evenEndLabelName, comment="Skip even code"))
    # generate even start label
    module.add(Label(evenStartLabelName, ""))
    return module

  ##############################################################################
  # generateEvenEndLabeNoLoadLoopForDTV
  # generate even end label for DirectToVgpr
  ##############################################################################
  def generateEvenEndLabeNoLoadLoopForDTV(self, name):
    module = Module("generateEvenEndLabeNoLoadLoopForDTV")
    # generate even end label
    module.add(Label("EvenEnd" + name, ""))
    return module

  ##############################################################################
  # generateOddEndVgprCopyForDTV
  # generate odd end vgpr copy for DirectToVgpr
  ##############################################################################
  def generateOddEndVgprCopyForDTV(self, kernel):
    module = Module("generateOddEndVgprCopyForDTV")
    vregNameBase = "G2LA" if kernel["DirectToVgprA"] else "G2LB"
    numVreg = self.states.a.numVgprG2L//2 if kernel["DirectToVgprA"] else self.states.b.numVgprG2L//2
    vregSet0 = vregNameBase + "0+"
    vregSet1 = vregNameBase + "1+"
    self.comment("copy Vreg set1 to Vreg set0 for DirectToVgpr + PrefetchAcrossPersistent")
    for index in range(numVreg):
      module.add(VMovB32(dst=vgpr(vregSet0+str(index)), src=vgpr(vregSet1+str(index))))
    return module

  ##############################################################################
  # Wrappers
  ##############################################################################

  ##############################################################################
  # longBranchScc0 - 32 bit offset
  # Conditional branch to label when SCC == 0
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc0(self, label: Label, posNeg: int=0, comment=""):
    with self.allocTmpSgpr(3) as tmpSgprInfo:
      return SCLongBranchScc0(label, tmpSgprInfo, \
        self.labels.getUniqueNamePrefix("NoBranch"), \
        self.labels.getUniqueNamePrefix("Positive"),
        posNeg, comment)

  ##############################################################################
  # longBranchScc1 - 32 bit offset
  # Conditional branch to label when SCC == 1
  # Use when erroring out "invalid operand due to label > SIMM16"
  ##############################################################################
  def longBranchScc1(self, label: Label, posNeg: int=0, comment=""):
    with self.allocTmpSgpr(3) as tmpSgprInfo:
      return SCLongBranchScc1(label, tmpSgprInfo, \
        self.labels.getUniqueNamePrefix("NoBranch"), \
        self.labels.getUniqueNamePrefix("Positive"),
        posNeg, comment)

  def sMagicDivWrapper(self, dest, dividend, magicNumber, magicShift):
    tmpVgpr = self.vgprPool.checkOut(2)
    module = sMagicDiv(dest, self.states.asmCaps["HasSMulHi"], dividend, magicNumber, magicShift, tmpVgpr)
    self.vgprPool.checkIn(tmpVgpr)
    return module

  def s_mul_u64_u32 (self, dst0, dst1,  src0, src1, comment):
    vtmp0 = self.vgprPool.checkOut(2)
    module = SMulInt64to32(self.states.asmCaps["HasSMulHi"], \
                           dst0, dst1, src0, src1, False, vtmp0, comment)
    self.vgprPool.checkIn(vtmp0)
    return module

  def s_mul_i64_i32 (self, dst0, dst1,  src0, src1, comment):
    vtmp0 = self.vgprPool.checkOut(2)
    module = SMulInt64to32(self.states.asmCaps["HasSMulHi"], \
                           dst0, dst1, src0, src1, True, vtmp0, comment)
    self.vgprPool.checkIn(vtmp0)
    return module

  def getBomb(self, cookie=None) -> Module:
    scratchVgpr = self.vgprPool.checkOut(2)
    bombCode = bomb(scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)
    return bombCode

  def getCmpAssert(self, function, val0, val1, cookie=-1):
    scratchVgpr = self.vgprPool.checkOut(2)
    function(val0, val1, scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)

  def getMultipleB32Assert(self, sval, multiple2, cookie=-1):
    scratchVgpr = self.vgprPool.checkOut(2)
    self.asmAssert.multiple_b32(sval, multiple2, scratchVgpr, cookie)
    self.vgprPool.checkIn(scratchVgpr)

  def getVectorDiffAssert(self, v0, v1, expectedScalarDiff, cookie=-1):
    cmpvtmp = self.vgprPool.checkOut(1)
    vtmp = self.vgprPool.checkOut(2)
    self.asmAssert.assert_vector_diff(v0, v1, expectedScalarDiff, cmpvtmp, vtmp, cookie)
    self.vgprPool.checkIn(vtmp)
    self.vgprPool.checkIn(cmpvtmp)

  def dump(self, vgprStore):
    return self.dumpData.dumpVgpr(vgprStore, self.labels.getUniqueName())

  def dumpSgpr(self, sgprStore):
    tmp = RegisterPoolResource(idx=self.vgprPool.checkOut(1,"tmp"), size=1)
    module = self.dumpData.dumpSgpr(sgprStore, tmp, self.labels.getUniqueName())
    self.vgprPool.checkIn(tmp.idx)
    return module

  def dumpLDS(self, kernel, startU, numU):
    tmp = RegisterPoolResource(idx=self.vgprPool.checkOut(2), size=2)
    module = self.dumpData.dumpLds(startU, numU, tmp, self.states.bpeAB, kernel["NumThreads"], \
      self.labels.getUniqueName())
    self.vgprPool.checkIn(tmp.idx)
    return module

def _getEccOffset(totalWidth, bpr, bpe, glvw, idx, numVgprG2L):
  if totalWidth < 1: # Need extra offset if global read < 1
    modVal = int(bpr / (bpe * glvw))
    left = idx % modVal
    return numVgprG2L * left
  else:
    return 0

def _growPool(pool: RegisterPool, rangeStart: int, rangeEnd: int, checkOutSize: int, comment: str=""):
  tl = []
  for _ in range(rangeStart, rangeEnd):
    tl.append(pool.checkOut(checkOutSize, comment))
  for t in tl:
    pool.checkIn(t)
