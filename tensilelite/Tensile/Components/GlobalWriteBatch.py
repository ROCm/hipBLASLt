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

from ..Common import globalParameters
from ..Component import GlobalWriteComponents
from ..SolutionStructs import Solution
from ..Activation import ActivationModule, ActivationType
from ..AsmStoreState import StoreState
from ..TensileInstructions import Label, Module, EXEC, SDWAModifiers, VCC, SelectBit, \
                            vgpr, sgpr, replaceHolder, SaturateCastType, VCvtBF16toFP32, \
                            DataType, CvtType, RoundType
from ..TensileInstructions.Instructions import *
from ..AsmAddressCalculation import AddrCalculation
from ..Components.PackData import formatting

from math import ceil

class GlobalWriteBatchComponent(GlobalWriteComponents):
  kernel = {"ProblemType": {"OperationType": "GEMM" }}
  def __call__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrD, addrC, addrBias, addrScaleD, \
    tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
    packdata, parentWriter) -> Module:
    return GlobalWriteBatchWriter(kernel, tPA, tPB, activation, ss, batchIdx, applyAlpha, \
      beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrD, addrC, addrBias, addrScaleD, \
      tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      packdata, parentWriter).emit()

class GlobalWriteBatchWriter:
  def __init__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrD, addrC, addrBias, addrScaleD, \
    tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      packdata, parentWriter):
    self.kernel = kernel
    self.tPA    = tPA
    self.tPB    = tPB
    self.activation = activation
    self.ss = ss
    self.batchIdx = batchIdx
    self.applyAlpha = applyAlpha
    self.beta = beta
    self.edge = edge
    self.atomic = atomic
    self.gwvw = gwvw
    self.atomicW = atomicW
    self.batchElements = batchElements
    self.addrD    = addrD
    self.addrC    = addrC
    self.addrBias = addrBias
    self.addrScaleD = addrScaleD
    self.activationSetPCStruct = activationSetPCStruct
    self.activationTypeStr     = activationTypeStr
    self.tmpVgpr = tmpVgpr
    self.bf16CVTVgprStruct = bf16CVTVgprStruct
    self.batchElementSgprs = batchElementSgprs
    self.tmpSgpr = tmpSgpr
    self.codeAccVgprRead = codeAccVgprRead
    self.codeMulAlpha = codeMulAlpha
    self.packdata     = packdata
    self.parentWriter = parentWriter
    self.loadsIssued = 0
    self.storesIssued = 0
    self.biasLocalBarrierInit = False

  @property
  def wavelen(self) -> int:
    return self.kernel["WavefrontSize"]

  @property
  def laneSGPRC(self) -> int:
    return self.parentWriter.states.laneSGPRCount

  @property
  def tmpS01(self):
    return self.tmpSgpr

  @property
  def tmpS23(self):
    return self.tmpS01 + self.laneSGPRC

  @property
  def debugConfig(self):
    return self.parentWriter.db

  @property
  def computeDataType(self) -> DataType:
    return self.kernel["ProblemType"]["ComputeDataType"]

  @property
  def destDataType(self) -> DataType:
    return self.kernel["ProblemType"]["DestDataType"]

  @property
  def moduleName(self):
    return "globalWriteBatch (Atomic)" if self.atomic else "globalWriteBatch (Non atomic)"

  def getEdgeMovInstType(self):
    return SMovB32 if self.wavelen == 32 else SMovB64

  def getEdgeOrInstType(self):
    return SOrB32 if self.wavelen == 32 else SOrB64

  def getEdgeAndInstType(self):
    return SAndB32 if self.wavelen == 32 else SAndB64

  def getSOrSaveExecType(self):
    return SOrSaveExecB32 if self.wavelen == 32 else SOrSaveExecB64

  def emit(self) -> Module:
    assert self._checkAtomicPreconditions()
    module = Module(self.moduleName)
    self._prolog(module)
    self._emitAdd(module)
    self._epilog(module)
    return module

  ##############################################################################
  # choose the ADD instruction for combining external C with internal C
  # used in atomic=1 case to compute expected external data
  ##############################################################################
  def _chooseAddForAtomic(self, kernel, dst, src0, src1, comment):
    module = Module("chooseAddForAtomic")
    if kernel["ProblemType"]["DataType"].isBFloat16():
      if kernel["_GlobalAccumulation"]:
        module.add(VAddF32(dst, src0, src1, comment=comment))
    elif kernel["ProblemType"]["DataType"].isHalf():
      if kernel["_GlobalAccumulation"]:
        module.add(VAddF32(dst, src0, src1, comment=comment))
      elif kernel["ProblemType"]["HighPrecisionAccumulate"]:
        if self.parentWriter.states.asmCaps["v_fma_mix_f32"]:
          module.add(VFmaMixF32(dst, src0, 1, src1, comment=comment))
        elif self.parentWriter.states.asmCaps["v_mad_mix_f32"]:
          module.add(VMadMixF32(dst, src0, 1, src1, comment=comment))
        else:
          assert False, "No valid v_mad_mix_f32 equivalent"
      else:
        module.add(VAddPKF16(dst, src0, src1, comment))
    elif kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isInt8():
      # assume v_add_i32 can be used in place of v_add_f32
      # need to add saturation directive to v_add_i32 instruction to clamp integer arithmetic
      module.add(VAddI32(dst, src0, src1, comment=comment))
    elif kernel["ProblemType"]["DataType"].isSingle():
      module.add(VAddF32(dst, src0, src1, comment=comment))
    else:
       #support for double
      module.add(VAddF64(dst, src0, src1, comment=comment))

    return module

  def _prolog(self, module: Module):
    module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
              (self.ss.optSingleColVgpr, self.ss.optSharedColVgpr, self.ss.optSGPRUsage, self.ss.optSrdIncForRow))

    if self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt(module)

    # comment tt1, tt0, vc1, vc0
    # tt = thread tile, vc=vector component
    commentStr = "Global Write%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Beta" if self.beta else "", " Edge" if self.edge else "", self.batchIdx)

    commentStr = ''.join([commentStr] \
                            + ["(%u,%u,%u,%u:vw%u%s)%s" % \
                               (element[0], element[1], element[2], element[3], self.gwvw,
                               ":vaw:%u"%self.atomicW if self.atomic else "",
                               "" if idx == len(self.batchElements) -1 else "; ")
                               for idx, element in enumerate(self.batchElements)])
    module.addComment2(commentStr)

    self.ss.setupStoreElementsForBatch(self.kernel, self.gwvw, self.batchElements, self.batchElementSgprs, isOptNLL=False, \
                                  allowLRVWforTLUandMI=self.kernel["allowLRVWforTLUandMI"], lrvwB=self.parentWriter.states.lrvwB)

    self.loadsIssued     = 0
    self.localLoadIssued = 0
    self.storesIssued    = 0
    self.loadsScaleDIssued     = 0

    ########################################
    # calculate addr and masks
    module.addComment1("calc coords, apply mask, and issue loads (if necessary)")
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.debugConfig["ConservativeWaitCnt"] & 0x10:
      module.add(SBarrier("debug"))
      module.add(SWaitCnt(vmcnt=0, comment="ConservativeWaitCnt"))
      if self.parentWriter.states.archCaps["SeparateVscnt"]:
        module.add(SWaitCnt(vscnt=0, comment="writes"))
      module.add(SBarrier("debug"))
    if not self.edge and self.debugConfig["ForceEdgeStores"] >= 2:
      module.add(self.parentWriter.getBomb()) # should not get here
    if self.edge and self.debugConfig["AssertNoEdge"]:
      module.add(self.parentWriter.getBomb()) # should not get here

    ########################################
    # rC *= alpha
    if not self.kernel["InterleaveAlpha"] and self.applyAlpha and self.parentWriter.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%self.batchElements)
      if self.codeMulAlpha is None:
        for elementIdx in range(len(self.batchElements)):
          module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))
      else:
          regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
          for elementIdx in range(len(self.batchElements)):
            for vi in range(self.gwvw):
              module.add(replaceHolder(self.codeMulAlpha.items().pop(0), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi))

    loadInputCode    = Module("loadInputCode")

    self.biasLoadIssued = []
    self.scaleDLoadIssued = []
    loadedDataBias = {}
    loadedDataScaleD = {}
    for elementIdx, element in enumerate(self.batchElements):
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addrCVgpr    = addrCalc.addrCVgpr
      addrDVgpr    = addrCalc.addrDVgpr
      addrBiasVgpr = addrCalc.addrBiasVgpr
      addrScaleDVgpr = addrCalc.addrScaleDVgpr
      data     = self.ss.elementData[elementIdx]
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleD = self.ss.elementDataScaleD[elementIdx]
      mask     = self.ss.elementMask[elementIdx]
      vc0 = element[3]

      module.add(addrCalc.emitAddressSetupCode(self.kernel, self.tPB, self.ss, self.tmpVgpr, self.tmpS01, self.edge, self.beta, self.atomic, elementIdx, addrDVgpr))

      if self.edge:
        module.add(addrCalc.edgeProtectCode(self.kernel, self.edge, self.beta, self.atomic, mask, self.tmpSgpr))

      # create code Module to push mov vgpr,acc instructions
      if self.beta:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'C', self.edge, self.beta, mask, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrCVgpr, self.addrC))
        if self.kernel["GroupLoadStore"]:
          loadInputCode.add(self.parentWriter.readCInput(self.kernel, self.ss, addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
        else:
          module.add(self.parentWriter.readCInput(self.kernel, self.ss, addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
        self.loadsIssued += 1
      if self.kernel["ProblemType"]["UseBias"]:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'Bias', self.edge, self.beta, mask, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrBiasVgpr, self.addrBias))
        if dataBias not in loadedDataBias:
          if self.kernel["GroupLoadStore"]:
            # Group bias load with C input to
            if (self.kernel["GlobalSplitU"] == 1) and (not self.biasLocalBarrierInit):
              loadInputCode.add(SWaitCnt(lgkmcnt=0, comment="Wait for Bias LDS write"))
              loadInputCode.add(SBarrier("Bias LDS write barrier"))
              self.biasLocalBarrierInit = True
            loadInputCode.add(self.parentWriter.addBiasLoad(self.kernel["ProblemType"]["ComputeDataType"], self.kernel, self.ss, addrCalc, dataBias, True))
          else:
            if (self.kernel["GlobalSplitU"] == 1) and (not self.biasLocalBarrierInit):
              module.add(SWaitCnt(lgkmcnt=0, comment="Wait for Bias LDS write"))
              module.add(SBarrier("Bias LDS write barrier"))
              self.biasLocalBarrierInit = True
            module.add(self.parentWriter.addBiasLoad(self.kernel["ProblemType"]["ComputeDataType"], self.kernel, self.ss, addrCalc, dataBias, True))
          loadedDataBias[dataBias] = 1
          self.localLoadIssued += 1
      self.biasLoadIssued.append(len(loadedDataBias))

      if self.kernel["ProblemType"]["UseScaleD"]:# and (self.kernel["GlobalSplitU"] == 1):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'ScaleD', self.edge, self.beta, mask, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrScaleDVgpr, self.addrScaleD))
        if dataScaleD not in loadedDataScaleD:
          # Shift right several vgprs for cvt ops if needed
          numVgprs = int(ceil(self.kernel["ProblemType"]["ComputeDataType"].numRegisters() * self.ss.cfg.gwvw))
          reg = self.kernel["ProblemType"]["ComputeDataType"].numRegisters() if self.kernel["ProblemType"]["ComputeDataType"].numRegisters() >= 1 else 1
          gprShiftScaleD = dataScaleD + (self.ss.cfg.gwvw * reg - numVgprs)
          if self.kernel["GroupLoadStore"]:
            # Group scaleD load with C input to
            loadInputCode.add(self.parentWriter.addScaleDLoad(self.kernel, self.ss, addrCalc, gprShiftScaleD))
          else:
            module.add(self.parentWriter.addScaleDLoad(self.kernel, self.ss, addrCalc, gprShiftScaleD))
          loadedDataScaleD[dataScaleD] = 1
          self.loadsScaleDIssued += 1
      self.scaleDLoadIssued.append(len(loadedDataScaleD))

      module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'D', self.edge, self.beta, mask, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrDVgpr, self.addrD))

      if self.atomic and (not self.parentWriter.states.useAtomicAdd):
        # load c into data+1 because of CAS structure
        # TODO - Fix for double here, would need bigger load
        # FIXME
        # gwvw is the number of elements in the batch
        # iterate over number of atomic operations to perform, each of width atomicW
        for avi in range(self.gwvw // self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
          bpm = self.parentWriter.states.bpeCexternal * self.atomicW
          useBuffer = self.kernel["BufferStore"]
          if self.kernel["BufferStore"]: # yes, BufferStore here - use same addressing regs for this load
            addr0 = vgpr(addrDVgpr)
            addr1 = sgpr("SrdD", 4)
          else:
            addr0 = vgpr(addrDVgpr, 2)
            addr1 = ""
          # Calculate vgpr Index for 32-bit/64-bit instruction
          # DGEMM use SRCS[2] register
          vgprIdx = bpm // 4
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bpm, dataV + vgprIdx, \
                    addr0, addr1, soffset=0, offset=addrCalc.globalOffset,
                    comment="load D (atomic) bpm=%u vaw=%u"%(bpm,self.atomicW)))

      if self.kernel["InterleaveAlpha"] and self.applyAlpha:
        module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))

      if not self.kernel["BufferStore"]:
        offsetSrc = (self.tmpVgpr + 2) if self.beta else addrDVgpr

        module.add(VAddCOU32(vgpr(addrDVgpr+0), VCC(), vgpr(self.addrD+0), \
            vgpr(offsetSrc+0), "addrDVgpr = D + index*bytes (lo)"))
        module.add(VAddCCOU32(vgpr(addrDVgpr+1), VCC(), vgpr(self.addrD+1), \
            vgpr(offsetSrc+1), VCC(), "addrDVgpr = D + index*bytes (hi)"))

        # restore full exec mask for calculating addr of next element
        if self.edge and (self.beta or self.atomic):
          module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -1 -> exec"))

    module.add(loadInputCode)

    if self.beta and self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt(module)

    ########################################
    # AccVgpr read
    if self.kernel.enabledSetPrioSplitLDS:
      module.add(SSetPrior(0))
    if self.codeAccVgprRead is not None:
      regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(len(self.batchElements)):
        # loop over scalars within one store instruction
        for vi in range(self.gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            module.add(replaceHolder(self.codeAccVgprRead.items().pop(0), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - self.parentWriter.states.c.startVgprValu))

      if not self.kernel["MIArchVgpr"]:
        module.add(SNop(1, "2 wait states required before reading vgpr"))

    ########################################
    # rC *= alpha
    if not self.kernel["InterleaveAlpha"] and self.applyAlpha and not self.parentWriter.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%self.batchElements)
      if self.codeMulAlpha is None:
        for elementIdx in range(len(self.batchElements)):
          module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))
      else:
          regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
          for elementIdx in range(len(self.batchElements)):
            for vi in range(self.gwvw):
              module.add(replaceHolder(self.codeMulAlpha.items().pop(0), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi - self.parentWriter.states.c.startVgprValu ))

  def _epilog(self, module: Module):
    # return registers to pool:
    lastData        = -1
    checkedDataBias = {}
    checkedDataScaleD = {}
    for elementIdx in range(len(self.batchElements)):
      if not self.ss.sharedColDVgprs:
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        addrDVgpr    = addrCalc.addrDVgpr
        addrCVgpr    = addrCalc.addrCVgpr
        addrBiasVgpr = addrCalc.addrBiasVgpr
        addrScaleDVgpr = addrCalc.addrScaleDVgpr
        self.parentWriter.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.parentWriter.vgprPool.checkIn(addrCVgpr)
        if addrBiasVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrBiasVgpr)
        if addrScaleDVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleDVgpr)

      data = self.ss.elementData[elementIdx]
      if data != 0:
        if data != lastData:
          self.parentWriter.vgprPool.checkIn(data)
        lastData = data

      dataBias = self.ss.elementDataBias[elementIdx]
      if dataBias != 0:
        if dataBias not in checkedDataBias:
          self.parentWriter.vgprPool.checkIn(dataBias)
        checkedDataBias[dataBias] = 1

      dataScaleD = self.ss.elementDataScaleD[elementIdx]
      if dataScaleD != 0:
        if dataScaleD not in checkedDataScaleD:
          self.parentWriter.vgprPool.checkIn(dataScaleD)
        checkedDataScaleD[dataScaleD] = 1

    self.ss.firstBatch = False
    self.ss.checkInTempVgprC()
    if self.kernel["StoreRemapVectorWidth"]:
      if self.parentWriter.StoreRemapLastBatch == 1:
        module.addComment1("Handle local read and global write")
        # this seems buggy? it's possible to issue more than one stores for SR
        # module.add(self.storeRemapAddStore(kernel, tmpVgpr, tmpS01, edge))
        # storesIssued += 1
        storeModule, numNewStores = self.parentWriter.storeRemapAddStore(self.kernel, self.tmpVgpr, self.tmpS01, self.edge)
        module.add(storeModule)
        self.storesIssued += numNewStores

    if self.parentWriter.states.serializedStore:
      module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

  def _emitAdd(self, module: Module):
    if self.atomic:
      del self.tmpVgpr # catch bugs
      if self.parentWriter.states.useAtomicAdd:
        self._emitAtomicAdd(module)
      else:
        self._emitCasAdd(module)
    else:
      self._emitNonatomicAdd(module)

  def _emitNonatomicAdd(self, module: Module):
    ########################################
    # Not Atomic
    ########################################
    # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
    interleaveStoreVmcnt = self.parentWriter.states.interleaveStoreVmcnt and not self.edge

    for elementIdx in range(len(self.batchElements)):
      for vi in range(self.gwvw):
        sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
        if self.kernel["ProblemType"]["ComputeDataType"].isInt32() or \
            self.kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
            if self.debugConfig["ForceExpectedValue"]:
              module.add(VMovB32(vgpr("ValuC+%u"%newSumIdxV), self.debugConfig["ValueCExpectedValue"], "force expected value" ))
            if self.parentWriter.db["ForceVSerial"]:
              module.add(VMovB32(vgpr("ValuC+%u"%newSumIdxV), vgpr("Serial"), "force expected value to serial" ))
            if self.parentWriter.db["CheckValueC"]:
              module.add(SMovB32(sgpr(self.tmpS01), self.debugConfig["ValueCExpectedValue"], "Move expected value"))
              module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%newSumIdxV), sgpr(self.tmpS01)))

    ########################################
    # wait for batched load
    if not interleaveStoreVmcnt:
      biasWait = self.kernel["ProblemType"]["UseBias"] and (self.kernel["GlobalSplitU"] == 1)
      scaleDWait = self.kernel["ProblemType"]["UseScaleD"] and (self.kernel["GlobalSplitU"] == 1)
      if self.beta:
        extraStr = " / Bias " if biasWait else ""
        ScaleDStr = " / ScaleD " if scaleDWait else ""
        extraStr += ScaleDStr
        if biasWait:
          module.add(SWaitCnt(vmcnt=0, vscnt=0, lgkmcnt=0, comment="writes & wait C" + extraStr))
        else:
          module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="writes & wait C" + extraStr))
      else:
        if scaleDWait:
          module.add(SWaitCnt(vmcnt=0, comment="writes & wait scaleD"))
        if biasWait:
          module.add(SWaitCnt(lgkmcnt=0, comment="writes & wait bias LDS"))

    module.addComment1("apply mask, calc new C and issue writes")
    # module.add(self.getBomb()) # can see store addresses just before the store inst

    activationCDataType = self.kernel["ProblemType"]["ActivationComputeDataType"]

    if self.kernel["ProblemType"]["DestDataType"].isBFloat16() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
      module.add(VMovB32(vgpr(self.bf16CVTVgprStruct.vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" ))
      module.add(VMovB32(vgpr(self.bf16CVTVgprStruct.vgprFp32Nan), "0x7fff0000", "fp32 Nan" ))
      module.add(VMovB32(vgpr(self.bf16CVTVgprStruct.vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" ))

    storeCode = Module("GroupLoadStore")
    biasWaitDict = {}
    scaleDWaitDict = {}
    for elementIdx in range(0, len(self.batchElements)):
      element = self.batchElements[elementIdx]
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleD = self.ss.elementDataScaleD[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      vc0 = element[3]
      sumIdx = self.ss.elementSumIdx[elementIdx]

      # print(str(element)+" rowInc="+str(addrCalc.rowInc))
      # Already write wave column block into LDS
      # Now read lds data back to registers and write to global memroy
      if self.ss.optSrdIncForRow and addrCalc.rowInc and self.kernel["StoreRemapVectorWidth"] > 0:
        module.addComment1("StoreRemap: shift coord1 address")
        module.add(addrCalc.incrementToNextRow(self.kernel, "D", self.ss, self.tmpS01))
        module.add(VMovB32(vgpr(self.tmpVgpr), addrCalc.rowInc, "set shift rows"))
        module.add(VAddU32(vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.tmpVgpr), "shift storeRemap coord1"))

      # apply in-bounds exec mask
      if self.edge and not self.kernel["BufferStore"]:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.laneSGPRC), "sgprs -> exec"))

      # if GWVW=1 the half path still assumes we have
      # at least two stores so does some combining across VI -
      # for example assuming we can have two elements and can use pk_mul
      # here:
      if (self.beta or ((self.kernel["ProblemType"]["UseBias"] or self.kernel["ProblemType"]["UseScaleD"]) and (self.kernel["GlobalSplitU"] == 1))) and interleaveStoreVmcnt:
        if (not self.beta): # If no beta

          localLoadCnt = -1
          if self.kernel["ProblemType"]["UseBias"] and (dataBias not in biasWaitDict):
            localLoadCnt = self.localLoadIssued - self.biasLoadIssued[elementIdx]

          waitLoadCnt = 0
          waitLoadCntScaleDVskip = waitLoadCnt
          if self.kernel["ProblemType"]["UseScaleD"] and (dataScaleD not in scaleDWaitDict):
            waitLoadCnt = waitLoadCnt + self.scaleDLoadIssued[elementIdx]

          vmcnt = self.loadsIssued + self.loadsScaleDIssued - waitLoadCnt

          if self.parentWriter.states.archCaps["SeparateVscnt"]:
            waitStoreCnt = 0
            vmComment = "{} = {} - {} - 1".format(vmcnt, self.loadsIssued, waitLoadCnt)
          else:
            waitStoreCnt = self.storesIssued if not self.kernel["GroupLoadStore"] else 0
            vmComment = " {} = {},{} - {},{} + {} - 1".format(vmcnt + waitStoreCnt, self.loadsIssued + self.loadsScaleDIssued, self.loadsScaleDIssued, waitLoadCnt, self.scaleDLoadIssued[elementIdx], waitStoreCnt)

          if (dataBias not in biasWaitDict) or (dataScaleD not in scaleDWaitDict):
            if self.kernel["ProblemType"]["UseScaleD"]:
              module.addSpaceLine()
              module.add(SWaitCnt(vmcnt=vmcnt, vscnt=waitStoreCnt, comment="wait scaleD (interleaved)" + vmComment))
            if self.kernel["ProblemType"]["UseBias"]:
              module.addSpaceLine()
              module.add(SWaitCnt(lgkmcnt=localLoadCnt, comment="wait bias lds load (interleaved)"))
        else:
          waitLoadCnt = 0
          waitLocalLoadCnt = -1

          if self.beta: waitLoadCnt = elementIdx + 1
          if self.kernel["ProblemType"]["UseBias"] and (dataBias not in biasWaitDict):
            waitLocalLoadCnt = self.localLoadIssued - self.biasLoadIssued[elementIdx]

          waitLoadCntScaleDVskip = waitLoadCnt
          if self.kernel["ProblemType"]["UseScaleD"] and (dataScaleD not in scaleDWaitDict):
            waitLoadCnt = waitLoadCnt + self.scaleDLoadIssued[elementIdx]

          vmcnt = self.loadsIssued + self.loadsScaleDIssued - waitLoadCnt - 1

          if self.parentWriter.states.archCaps["SeparateVscnt"]:
            waitStoreCnt = 0
            vmComment = "{} = {} - {} - 1".format(vmcnt, self.loadsIssued, waitLoadCnt)
          else:
            waitStoreCnt = self.storesIssued if not self.kernel["GroupLoadStore"] else 0
            vmComment = " {} = {}{},{} - {},{},{} + {} - 1".format(vmcnt + waitStoreCnt,
                                                            self.loadsIssued+self.loadsScaleDIssued, self.loadsIssued, self.loadsScaleDIssued,
                                                            waitLoadCnt, elementIdx, self.scaleDLoadIssued[elementIdx],
                                                            waitStoreCnt)

          module.addSpaceLine()
          module.add(SWaitCnt(lgkmcnt=waitLocalLoadCnt, vmcnt=vmcnt, vscnt=waitStoreCnt, comment="wait C (interleaved)" + vmComment))

        biasWaitDict[dataBias] = 1 # No need to wait again if the bias data is already loaded.
        scaleDWaitDict[dataScaleD] = 1 # No need to wait again if the scaleD data is already loaded.

      if self.beta:
        module.add(self._addSumAlphaWithCBeta(self.kernel, self.ss, self.gwvw, elementIdx, vc0, self.tmpVgpr, self.bf16CVTVgprStruct))
      elif ((self.kernel["ProblemType"]["UseBias"] and (self.kernel["GlobalSplitU"] == 1)) or self.kernel["ActivationFuncCall"]) and not self.applyAlpha: # case of alpha=1 and beta=0
        if (self.kernel["ProblemType"]["DestDataType"].isInt8() or self.kernel["ProblemType"]["DestDataType"].isInt32()) and self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          module.add(convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_I32_to_F32, \
                                      inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu))

      # Add bias
      mergeActFuncCall = False
      if self.kernel["ProblemType"]["UseBias"] and (self.kernel["GlobalSplitU"] == 1):
        if activationCDataType == self.kernel["ProblemType"]["ComputeDataType"] and self.kernel["ActivationFuncCall"]:
          mergeActFuncCall = True
        for vi in range(0, self.gwvw):
          inputVgpr = dataBias + vi
          sumIdxV   = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            vgprDst = (self.activationSetPCStruct.vgprActCopy + vi) if mergeActFuncCall else "ValuC+%d"%vgprIdx
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              module.add(VAddF32(dst=vgpr(vgprDst), src0=vgpr(inputVgpr), src1=vgpr("ValuC+%d"%vgprIdx), \
                                 comment="C += bias"))

            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              module.add(VAddPKF32(dst=vgpr(vgprDst, 2), src0=vgpr(inputVgpr, 2), \
                                   src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="C += bias"))
          else:
            raise RuntimeError("Unsupported bias compute data type %s."%str(self.kernel["ProblemType"]["ComputeDataType"]))

      SaturateTypeInt8 = SaturateCastType.NORMAL
      # Activation
      activationModule = None
      isActivationInsertAfter = False
      if self.kernel["ActivationFuncCall"]:
        if (activationCDataType == self.kernel["ProblemType"]["DestDataType"]) and \
          (activationCDataType != self.kernel["ProblemType"]["ComputeDataType"]):
          isActivationInsertAfter = True
        activationModule = Module("ActivationFuncCall")
        if (not mergeActFuncCall) and (not isActivationInsertAfter):
          activationModule.appendModule (copyData(activationCDataType, self.ss.elementSumIdx[elementIdx], self.gwvw, \
            self.activationSetPCStruct.vgprActCopy))
        activationModule.add(SSwapPCB64(dst=sgpr(self.activationSetPCStruct.sgprOffsetBack, 2), \
          src=sgpr(self.activationSetPCStruct.sgprOffsetActivation, 2)))
        activationModule.appendModule (copyData(activationCDataType, self.ss.elementSumIdx[elementIdx], self.gwvw, \
          self.activationSetPCStruct.vgprActCopy, 1))
      elif self.parentWriter.insertActivationAfterPacked(self.kernel, self.activationTypeStr):
        isActivationInsertAfter = True
        activationModule = self.parentWriter.getActivationDestDataType(self.kernel, self.activation, \
          self.activationTypeStr, self.gwvw, self.ss.elementSumIdx[elementIdx], self.tmpVgpr, self.tmpSgpr)
      else:
        satInt8 = False
        if self.kernel["ProblemType"]["DestDataType"].isInt8():
          if (self.activationTypeStr == 'abs') or (self.activationTypeStr == 'relu'):
            SaturateTypeInt8 = SaturateCastType.DO_NOTHING
            satInt8 = True
        activationModule = self.parentWriter.getActivationActivationComputeType(self.kernel, self.activation, \
          self.activationTypeStr, self.gwvw, self.ss.elementSumIdx[elementIdx], self.tmpVgpr, self.tmpSgpr, satInt8)

      # pack stores, beta and non-beta reach here:
      packModule = Module("Empty pack module")
      convertModule = Module("Empty convert module")
      if self.kernel["ProblemType"]["HighPrecisionAccumulate"] and (self.kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
        if self.kernel["ActivationFuncCall"] and activationCDataType == self.kernel["ProblemType"]["DestDataType"]:
          destIdx = self.activationSetPCStruct.vgprActCopy
        else:
          destIdx = self.ss.elementSumIdx[elementIdx]
        if self.kernel["ProblemType"]["DestDataType"].isHalf():
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isBFloat16():
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf16CVTVgprStruct=self.bf16CVTVgprStruct,
                                     tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt32():
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.kernel["ProblemType"]["UseBias"] and (self.kernel["GlobalSplitU"] == 1)) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt8():
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.kernel["ProblemType"]["UseBias"] and (self.kernel["GlobalSplitU"] == 1)) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, roundType=RoundType.ROUND_TO_NEAREST_EVEN, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], self.tmpVgpr, self.tmpS01,
                                     SaturateTypeInt8=SaturateTypeInt8, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)

      scaleDModule = Module("scaleDModule")
      if self.kernel["ProblemType"]["UseScaleD"] and (self.kernel["GlobalSplitU"] == 1):
        for vi in range(0, self.gwvw):
          inputScaleDVgpr = dataScaleD + vi
          sumIdxV   = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu

            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):

              scaleDModule.add(VCmpGtU32(dst=sgpr("AddressScaleD",2), src0=sgpr("SrdScaleD+2"), src1=0, comment=" == 0 ?"))
              scaleDModule.add(VCndMaskB32(
                dst=vgpr(inputScaleDVgpr), \
                src1=vgpr(inputScaleDVgpr), \
                src0=1.0, \
                src2=sgpr("AddressScaleD",2), \
                comment="1. mul 1 if 0"))

              if isActivationInsertAfter:
                if (self.kernel["ProblemType"]["DestDataType"].isHalf()):
                  scaleDModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                if self.kernel["ProblemType"]["DestDataType"].isBFloat16():
                  scaleDModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%vgprIdx), src=("ValuC+%d"%vgprIdx), vgprMask=None, vi=0))
                if self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isInt8():
                  scaleDModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                scaleDModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleDVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= scaleDVMul" ))
              else:
                scaleDModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleDVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= scaleDVMul" ))

            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              scaleDModule.add(VCmpGtU32(dst=sgpr("AddressScaleD",2), src0=sgpr("SrdScaleD+2"), src1=0, comment=" == 0 ?"))
              scaleDModule.add(VCndMaskB32(
                dst=vgpr(inputScaleDVgpr), \
                src1=vgpr(inputScaleDVgpr), \
                src0=1.0, \
                src2=sgpr("AddressScaleD",2), \
                comment="1. mul 1 if 0"))

              scaleDModule.add(VCndMaskB32(
                dst=vgpr(inputScaleDVgpr+1), \
                src1=vgpr(inputScaleDVgpr+1), \
                src0=1.0, \
                src2=sgpr("AddressScaleD",2), \
                comment="1. mul 1 if 0"))

              if isActivationInsertAfter:
                if self.kernel["ProblemType"]["DestDataType"].isHalf():
                  scaleDModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                  scaleDModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src=vgpr("ValuC+%d"%(vgprIdx+1))))
                if self.kernel["ProblemType"]["DestDataType"].isBFloat16():
                  scaleDModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%vgprIdx), src=("ValuC+%d"%vgprIdx), vgprMask=None, vi=0))
                  scaleDModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%(vgprIdx+1)), src=("ValuC+%d"%(vgprIdx+1)), vgprMask=None, vi=0))
                if self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isInt8():
                  scaleDModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                  scaleDModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src=vgpr("ValuC+%d"%(vgprIdx+1))))
                scaleDModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleDVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= scaleDVMulPK(%d)(%d)"%(dataScaleD,vi)))
              else:
                scaleDModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleDVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= scaleDVMulPK(%d)(%d)"%(dataScaleD,vi)))
          else:
            raise RuntimeError("Unsupported scaleD compute data type %s."%str(self.kernel["ProblemType"]["ComputeDataType"]))

      if isActivationInsertAfter:
        module.add(convertModule)
        module.add(packModule)
        module.add(activationModule)
        module.add(scaleDModule)
        if self.kernel["ProblemType"]["UseScaleD"] and (self.kernel["GlobalSplitU"] == 1):
          module.add(convertModule)
          module.add(packModule)
      else:
        module.add(activationModule)
        module.add(scaleDModule)
        module.add(convertModule)
        module.add(packModule)

      if not self.kernel["StoreRemapVectorWidth"]:
        tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, addrCalc, sumIdx, self.tmpS01, self.edge)
        if self.kernel["GroupLoadStore"]:
          storeCode.add(tmpStoreCode)
        else:
          module.add(tmpStoreCode)
        self.storesIssued += 1

      else:
        rpe = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr
        module.add(self.parentWriter.storeRemapAddLocalWrite(self.ss, addrCalc, sumIdx*rpe))
        # Column Block Shape has been written to LDS
        # Now read back and write out to global memory

    module.add(storeCode)

    if self.parentWriter.db["CheckStoreC"]>=0:
      useBuffer = self.kernel["BufferStore"]
      # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
      # Note - TODO- CheckStoreC also won't work for StoreRemap
      module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="CheckStoreC, wait for stores to complete"))
      for elementIdx in range(0, len(self.batchElements)):
        addr = self.ss.elementAddr[elementIdx].addrDVgpr
        sumIdx = self.ss.elementSumIdx[elementIdx]

        bps = self.kernel["ProblemType"]["DestDataType"].numBytes() * self.gwvw
        if self.kernel["BufferStore"]:
          addr0 = vgpr(addr)
          addr1 = sgpr("SrdC", 4)
        else:
          addr0 = vgpr(addr,2)
          addr1 = ""

        if self.kernel["ProblemType"]["DestDataType"].isHalf() or self.kernel["ProblemType"]["DestDataType"].isBFloat16():
          if not self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
                                  addr0, addr1, soffset=0, offset=0, hi16=sumIdx%2))
          else:
            module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                  addr0, addr1, soffset=0, offset=0, hi16=0))
        elif self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isSingle():
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                addr0, addr1, soffset=0, offset=0))
        elif self.kernel["ProblemType"]["DestDataType"].isDouble() or self.kernel["ProblemType"]["DestDataType"].isSingleComplex() :
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
                                addr0, addr1, soffset=0, offset=0))
        elif self.kernel["ProblemType"]["DestDataType"].isDoubleComplex():
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
                                addr0, addr1, soffset=0, offset=0))
      module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="CheckStoreC, wait for stores to complete"))
      # Add checks for expected values:
      module.add(SMovB32(sgpr(self.tmpS01), self.parentWriter.db["CheckStoreC"], "expected value"))
      for elementIdx in range(0, len(self.batchElements)):
        sumIdx = self.ss.elementSumIdx[elementIdx]
        # Need to fix for other types:
        assert (self.kernel["ProblemType"]["DestDataType"].isSingle() or self.kernel["ProblemType"]["DestDataType"].isInt32())
        module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr(sumIdx), sgpr(self.tmpS01)))


    if self.edge and (self.atomic or not self.kernel["BufferStore"]):
      # subsequent batch must start with full exec mask
      # BufferStore doesn't need exec since it used buffer range checking when
      # possible
      module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

    if self.parentWriter.db["ConservativeWaitCnt"] & 0x40:
      module.add(SBarrier("debug"))
      module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
      module.add(SBarrier("debug"))

  def _emitAtomicAdd(self, module: Module):
    ########################################
    # first attempt write
    module.addComment1("issue first atomic writes")
    for elementIdx in range(len(self.batchElements)):
      addrCalc = self.ss.elementAddr[elementIdx]
      mask     = self.ss.elementMask[elementIdx]

      # apply in-bounds exec mask
      if self.edge:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.laneSGPRC), "sgprs -> exec (before atomic)"))

      for avi in range(0, self.gwvw // self.atomicW):
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            module.add(BufferAtomicAddF32(vgpr("ValuC+%u"%newSumIdxV), \
                         vgpr(addrCalc.addrDVgpr,1), \
                         sgpr("SrdD", 4), \
                         0,
                         MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset),
                         "attempt write avi=%u" % (avi)))
          else:
            pass # TODO:

    if self.edge:
      module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

  def _emitCasAdd(self, module: Module):
    # TODO for atomic GWVW:
    #  - Use vi to compute addresses, sumIdx.
    #  - Need a solution for the mask.  Can move to all buffer or can fix?
    element = self.batchElements[0]
    d1 = element[0]
    d0 = element[1]
    vc1 = element[2]
    vc0 = element[3]
    labels = self.parentWriter.labels
    labelString = "Global_Write%s%s_%u_%u_%u_%u" % ("_Beta" if self.beta else "", "_Edge" if self.edge else "", vc0, vc1, d0, d1 )
    labelComment = "Global_Write (Beta) (Edge) vc0 vc1 d0 d1"
    label = Label(labels.getName(labelString), labelComment)
    labelString += "_EarlyExit"
    labelAfterAtomicLoop = Label(labels.getName(labelString), labelComment)

    ########################################
    # wait for batched load
    # TODO - we are always atomic here?
    module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="wait C (atomic)"))
    ########################################
    # first attempt write
    module.addComment1("issue first atomic writes")
    for elementIdx, element in enumerate(self.batchElements):
      addrCalc = self.ss.elementAddr[elementIdx]
      mask = self.ss.elementMask[elementIdx]

      # apply in-bounds exec mask
      if self.edge:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.parentWriter.states.laneSGPRCount), "sgprs -> exec (before atomic)"))

      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        ## number of src[s]/dst[s] register for DGEMM / SGEMM HGEMM
        vgprCnt = 2 if self.kernel["ProblemType"]["DestDataType"].isDouble() else 1
        if self.kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not self.kernel["_GlobalAccumulation"]:
          sumIdxV //= 2
        if self.kernel["ProblemType"]["DestDataType"].isDouble(): sumIdxV = sumIdxV * 2
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        bpm = self.parentWriter.states.bpeCexternal * self.atomicW
        # Calculate vgpr Index for 32-bit/64-bit instruction
        # DGEMM use SRCS[2] register
        vgprIdx = 1*(bpm//4)
        # for atomic, data[1] = original c, data[0] = new c
        module.add(self._chooseAddForAtomic(self.kernel, \
                  vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%newSumIdxV,vgprCnt), \
                  "desired value avi=%u"%avi))

        # attempt write
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            # use cmpswap_x2 for DGEMM in CAS loop
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(BufferAtomicCmpswapB64(vgpr(dataV,4), \
                              vgpr(addrCalc.addrDVgpr,1), \
                              sgpr("SrdD", 4),  \
                              0,
                              MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True),
                              "attempt write avi=%u"%(avi)))
            else:
            # use cmpswap for SGEMM in CAS loop
              module.add(BufferAtomicCmpswapB32(vgpr(dataV,2), \
                           vgpr(addrCalc.addrDVgpr,1), \
                           sgpr("SrdD", 4), \
                           0, \
                           MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True), \
                           "attempt write avi=%u"%(avi)))
          else:
            module.add(FlatAtomicCmpswapB32(vgpr(atomicDestVgpr), \
                                            vgpr(addrCalc.addrDVgpr,2), \
                                            vgpr(dataV,2),
                                            FLATModifiers(glc=True),
                                            "attempt write"))
        else:
            # Fake successful CAS swap
            module.add(VMovB32(vgpr(atomicDestVgpr), vgpr(dataV+1), "Fake successful CAS" ))

    ########################################
    # wait for first attempt write
    module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="wait for atomic writes"))
    ########################################
    # check first attempt
    module.addComment1("check success of writes, update masks")
    for elementIdx, element in enumerate(self.batchElements):
      mask = self.ss.elementMask[elementIdx]

      # calculate new masks
      if self.edge:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.laneSGPRC), "sgprs -> exec"))
        for avi in range(0, self.gwvw // self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi * self.ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
          # need to apply element mask before comparison
          # so that all valid lanes are doing the cmp
          if avi == 0:
            # use u64 for DGEMM
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(VCmpNeU64(sgpr(self.tmpS01, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                  vgpr(dataV+2,2), comment="c read during atomic == c read during prior load (avi=%u, first)"%avi))
            else:
              module.add(VCmpNeU32(sgpr(self.tmpS01, self.laneSGPRC), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), comment="c read during atomic == c read during prior load (avi=%u, first)"%avi))
          else:
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(VCmpNeU64(sgpr(self.tmpS23, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                  vgpr(dataV+2,2), comment="c read during atomic != c read during prior load"))
            else:
              module.add(VCmpNeU32(sgpr(self.tmpS23, self.laneSGPRC), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), comment="c read during atomic == c read during prior load (avi=%u)"%avi))
            module.add(self.getEdgeOrInstType()(sgpr(self.tmpS01, self.laneSGPRC), \
                  sgpr(self.tmpS01, self.laneSGPRC), sgpr(self.tmpS23, self.laneSGPRC), "combine with tmp mask"))

        module.add(self.getEdgeAndInstType()(sgpr(mask, self.laneSGPRC), sgpr(self.tmpS01, self.laneSGPRC), sgpr(mask,self.laneSGPRC), "inBounds & must try again"))

      else:
        for avi in range(0, self.gwvw//self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
          if self.kernel["ProblemType"]["DestDataType"].isDouble():
            module.add(VCmpNeU64(sgpr(mask, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                vgpr(dataV+2,2), comment="c read during atomic != c read during prior load"))
          else:
            module.add(VCmpNeU32(sgpr(mask, self.laneSGPRC), vgpr(atomicDestVgpr), \
                vgpr(dataV+1), comment="c read during atomic != c read during prior load"))

    # or masks together to check early exit
    module.addComment1("or masks to check for exit")
    module.add(self.getEdgeMovInstType()(sgpr(self.tmpS01, self.laneSGPRC), hex(0), "empty mask"))
    for elementIdx in range(0, len(self.batchElements)):
      mask = self.ss.elementMask[elementIdx]
      module.add(self.getEdgeOrInstType()(sgpr(self.tmpS01, self.laneSGPRC), sgpr(mask, self.laneSGPRC), sgpr(self.tmpS01, self.laneSGPRC), "or to add threads"))
    module.add(self.getSOrSaveExecType()(sgpr(self.tmpS23,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "apply combined mask"))
    module.add(SCBranchExecZ(labelAfterAtomicLoop.getLabelName(), "if exec is zero skip loop"))

    # begin atomic loop
    module.addComment1("atomic CAS loop")
    module.add(label)

    module.addComment1("apply updated masks and issue writes again")
    for elementIdx in range(0, len(self.batchElements)):
      addrCalc = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      mask = self.ss.elementMask[elementIdx]
      vgprCnt = 2 if self.kernel["ProblemType"]["DestDataType"].isDouble() else 1   # number of registers for f32/f64
      bpm = self.parentWriter.states.bpeCexternal * self.atomicW
      vgprIdx = 1*(bpm//4)   # index register

      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        if self.kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not self.kernel["_GlobalAccumulation"]:
          sumIdxV //= 2
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          sumIdxV =  sumIdxV * 2
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu

        # apply mask for element
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask,self.laneSGPRC), "must try again"))
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          #64-bit C val move by 2 32-bit instructions
          module.add(VMovB32(vgpr(dataV+2), vgpr(atomicDestVgpr), "dataV+2 = tmp (new original C)" ))
          module.add(VMovB32(vgpr(dataV+3), vgpr(atomicDestVgpr+1), "dataV+3 = tmp (new original C)" ))
        else:
          module.add(VMovB32(vgpr(dataV+1), vgpr(atomicDestVgpr), "dataV+1 = tmp (new original C)" ))
        module.add(self._chooseAddForAtomic(self.kernel, \
                        vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%newSumIdxV,vgprCnt), \
                        "newC = rC + originalC"))
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            # Using no-ret version here?
            # cmpswap_x2 for DGEMM
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(BufferAtomicCmpswapB64(vgpr(dataV,4), \
                          vgpr(addr,1), \
                          sgpr("SrdD", 4), \
                          0,
                          MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True),
                          "try again"))
            else:
              module.add(BufferAtomicCmpswapB32(
                          vgpr(dataV,2), \
                          vgpr(addr,1), \
                          sgpr("SrdD", 4), \
                          0,
                          MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True),
                          "try again"))
          else:
            module.add(FlatAtomicCmpswapB32(vgpr(atomicDestVgpr), \
                                            vgpr(addr,2), \
                                            vgpr(dataV,2), \
                                            FLATModifiers(glc=True), \
                                            "try again"))

    # wait for batched write
    module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="wait for atomic writes"))
    # check batched write success
    module.addComment1("apply masks and check for success")
    for elementIdx in range(0, len(self.batchElements)):
      data = self.ss.elementData[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2

        # apply mask for element
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask,self.laneSGPRC), "must try again"))

        # compare success
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          module.add(VCmpNeU64(sgpr(self.tmpS01,self.laneSGPRC), vgpr(data+2,2), vgpr(atomicDestVgpr,2), \
              comment="c read during atomic != c read during prior load"))
        else:
          module.add(VCmpNeU32(sgpr(self.tmpS01,self.laneSGPRC), vgpr(data+1), vgpr(atomicDestVgpr), \
              comment="c read during atomic == c read during prior load"))
        # update element mask
        module.add(self.getEdgeAndInstType()(sgpr(mask,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), sgpr(mask,self.laneSGPRC), "inBounds & must try again"))

    # or masks together
    module.addComment1("or masks to check for exit")
    module.add(self.getEdgeMovInstType()(sgpr(self.tmpS01,self.laneSGPRC), hex(0), "empty mask"))
    for elementIdx in range(0, len(self.batchElements)):
      mask = self.ss.elementMask[elementIdx]
      module.add(self.getEdgeOrInstType()(sgpr(self.tmpS01,self.laneSGPRC), sgpr(mask,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "or to add threads"))

    # apply combined masks and exit
    module.add(self.getSOrSaveExecType()(sgpr(self.tmpS23, self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "apply combined mask"))
    module.add(SCBranchExecNZ(label.getLabelName(), "try again if not complete"))
    module.add(labelAfterAtomicLoop)
    module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

  def _checkAtomicPreconditions(self) -> bool:
    if self.atomic:
      # all kinds of code relies on this assumption:
      if self.atomicW > self.gwvw:
        return False

      if (self.kernel["ProblemType"]["DataType"].isHalf() or self.kernel["ProblemType"]["DataType"].isBFloat16()) \
        and not self.kernel["_GlobalAccumulation"]:
        return self.atomicW >= 2
    return True

  def _storeSyncOpt(self, module: Module):
    module.add(SSleep(self.kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
    module.add(SBarrier())

  def _applyAlpha(self, kernel, gwvw, elementSumIdx, elementIdx, tmpS01):
    module = Module("applyAlpha")

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      return module

    if self.parentWriter.do["ApplyAlpha"]:
      for vi in range(0, gwvw):
        sumIdxV = elementSumIdx[elementIdx] + vi

        if kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (h,h,h,h,h,h), internal alpha is f16 (2-16bits)
          if sumIdxV%2:
            newSumIdx = sumIdxV // 2 - self.parentWriter.states.c.startVgprValu
            module.add(VMulPKF16(dst=vgpr("ValuC+%u"%(newSumIdx)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx)), comment="*= alpha sumIdx=%u vi=%u"%(elementSumIdx[elementIdx], vi)))

        # Int8 (TODO- Int8x4 not checked, but should be OK)
        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          newSumIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
          # below assume we use v_mul_lo_u32. Could also use v_mul_i32_i24.
          # module.add(VMulI32I24(dst=vgpr("ValuC+%u"%newSumIdx), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%newSumIdx), comment="*= alpha" )_
          module.add(VMulLOU32(dst=vgpr("ValuC+%u"%newSumIdx), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%newSumIdx), comment="*= alpha" ))
          if self.parentWriter.db["ForceExpectedValue"]:
            module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdx), src=self.parentWriter.db["ValueCExpectedValue"], comment="force expected value" ))
          if self.parentWriter.db["CheckValueC"]:
            module.add(SMovB32(dst=sgpr(tmpS01), src=self.parentWriter.db["ValueCExpectedValue"], comment="Move expected value"))
            module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%newSumIdx), sgpr(tmpS01)))

        # sgemm, HPA-bfgemm(b,b,b,b,s,s), and HPA-hgemm(h,h,h,h,s,s)
        # (h,h,h,h,h,h) + HPA (will be converted to (h,h,h,h,s,s)), internal alpha is single
        elif kernel["ProblemType"]["ComputeDataType"].isSingle() or (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):

          if kernel["ProblemType"]["DataType"].isInt8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VCvtI32toF32(dst=vgpr("ValuC+%u"%sumIdxV), src=vgpr("ValuC+%u"%sumIdxV), comment="convert to fp32" ))

          newSumIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
          module.add(VMulF32(dst=vgpr("ValuC+%u"%newSumIdx), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%newSumIdx), comment="*= alpha" ))
          if self.parentWriter.db["ForceExpectedValue"]:
            module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdx), src=self.parentWriter.db["ValueCExpectedValue"], comment="force expected value" ))
          if self.parentWriter.db["ForceVSerial"]:
            module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdx), src=vgpr("Serial"), comment="force expected value to serial" ))
          if self.parentWriter.db["CheckValueC"]:
            module.add(SMovB32(dst=sgpr(tmpS01), src=self.parentWriter.db["ValueCExpectedValue"], comment="Move expected value"))
            module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%newSumIdx), sgpr(tmpS01)))

        # dgemm
        elif kernel["ProblemType"]["ComputeDataType"].isDouble():
          newSumIdx = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
          module.add(VMulF64(dst=vgpr("ValuC+%u"%(newSumIdx),2), src0=sgpr("Alpha",2), src1=vgpr("ValuC+%u"%(newSumIdx),2), comment="*= alpha"))

        # single precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
          newSumIdx = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
          tmpVgpr = self.parentWriter.vgprPool.checkOut(1)
          module.add(VMovB32(dst=vgpr(tmpVgpr), src=vgpr("ValuC+%u"%(newSumIdx)), comment="store Cr"))
          module.add(VMulF32(dst=vgpr("ValuC+%u"%(newSumIdx)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx)), comment="*= alpha ( Cr = Ar * Cr)"))
          module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdx)), src0=(sgpr("Alpha+1").getMinus()), src1=vgpr("ValuC+%u"%(newSumIdx+1)), comment="*= alpha ( Cr += -Ai * Ci )"))
          module.add(VMulF32(dst=vgpr("ValuC+%u"%(newSumIdx+1)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx+1)), comment="*= alpha ( Ci = Ar * Ci)"))
          module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdx+1)), src0=sgpr("Alpha+1"), src1=vgpr(tmpVgpr), comment="*= alpha ( Ci += Ai * Cr_backup )"))
          self.parentWriter.vgprPool.checkIn(tmpVgpr)

        # double precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
          newSumIdx = sumIdxV * 4 - self.parentWriter.states.c.startVgprValu
          vtmp1 = self.parentWriter.vgprPool.checkOutAligned(2, 2)
          vtmp2 = self.parentWriter.vgprPool.checkOutAligned(2, 2)
          # tmp1 = a.real * b.real
          module.add(VMulF64(dst=vgpr(vtmp1,2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(newSumIdx+0),2)))
          # tmp2 = a.imag * b.real
          module.add(VMulF64(dst=vgpr(vtmp2,2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(newSumIdx+0),2)))
          # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
          module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdx+0),2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(newSumIdx+2),2), src2=vgpr(vtmp1,2)))
          # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
          module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdx+2),2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(newSumIdx+2),2), src2=vgpr(vtmp2,2)))
          self.parentWriter.vgprPool.checkIn(vtmp1)
          self.parentWriter.vgprPool.checkIn(vtmp2)
    return module

  def _addSumAlphaWithCBeta(self, kernel, ss, gwvw, elementIdx, vc0, tmpVgpr, bf16CVTVgprStruct):
    module = Module("addSumAlphaWithCBeta #elementIdx%u, vc0 %u"%(elementIdx, vc0))
    for vi in range(0, gwvw):
      dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
      sumIdxV = ss.elementSumIdx[elementIdx] + vi
      if kernel["ProblemType"]["DestDataType"].isHalf():
        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          if sumIdxV%2==0 or (not ss.cfg.halfDataRegPerVI and gwvw==1):
            newSumIdxV = sumIdxV // 2 - self.parentWriter.states.c.startVgprValu
            # dataV+0 = new c = old c*beta
            module.add(VMulPKF16(dst=vgpr(dataV), src0=sgpr("Beta"), src1=vgpr(dataV+0), \
                comment="%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi)))
            # dataV+0 = new c = old c*beta + rC
            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(newSumIdxV)), \
                comment="sum*alpha + C*beta"))
          else:
            pass # add will have been done previously
        else: # HPA
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          # dataV+0 = new c = old c*beta + rC
          # src0 = beta = f32 = opsel 00
          # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
          # src2 = sumIdxV = f32 = opsel 00
          dataCExternal = ss.elementData[elementIdx] + vi//2
          hi16 = (vi + gwvw*vc0) % 2
          module.add(self.parentWriter.states.mixinst(dst=vgpr("ValuC+%u"%newSumIdxV), src0=sgpr("Beta"), \
              src1=vgpr(dataCExternal), src2=vgpr("ValuC+%u"%newSumIdxV), \
              vop3=VOP3PModifiers(op_sel=[0,hi16,0], op_sel_hi=[0,1,0]),
              comment="//C*=beta"))

      elif kernel["ProblemType"]["DestDataType"].isBFloat16():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # dataV+0 = new c = old c*beta + rC
          # src0 = beta = f32 = opsel 00
          # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
          # src2 = sumIdxV = f32 = opsel 00
          dataCExternal = ss.elementData[elementIdx] + vi//2
          if (vi%2) == 1:
            module.add(VAndB32(dst=vgpr(tmpVgpr), src0=vgpr(dataCExternal), src1=vgpr(bf16CVTVgprStruct.vgprBf16Mask), comment="convert bf16 to fp32"))
          else:
            module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=16, src=vgpr(dataCExternal), comment="convert bf16 to fp32" ))
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), \
              comment="finalSum = sum*alpha + C*beta"))
      elif kernel["ProblemType"]["DestDataType"].isSingle():
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=sgpr("Beta"), \
            comment="finalSum = sum*alpha + C*beta"))

      elif kernel["ProblemType"]["DestDataType"].isInt8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          if (vi%4) != 3:
            module.add(VBfeI32(dst=vgpr(tmpVgpr), src0=vgpr(dataV+0), src1=(vi * 8), src2=8, comment="int8 to int32"))
          else:
            module.add(VAShiftRightI32(dst=vgpr(tmpVgpr), shiftHex=24, src=vgpr(dataV+0), comment="int8 to int32"))

          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          if kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(VCvtI32toF32(dst=vgpr(tmpVgpr), src=vgpr(tmpVgpr), comment="convert to fp32" ))
            module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), \
                               comment="finalSum = sum*alpha + C*beta"))
          else:
            module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr("Beta"), src1=vgpr(tmpVgpr), comment="C = C*beta"))
            module.add(VAddU32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=vgpr("ValuC+%u"%newSumIdxV), comment="finalSum = sum*alpha + C*beta"))

      elif kernel["ProblemType"]["DestDataType"].isInt32():
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        if kernel["ProblemType"]["ComputeDataType"].isSingle():
          module.add(VCvtI32toF32(dst=vgpr(dataV+0), src=vgpr(dataV+0), comment="convert to fp32" ))
          module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta"))
        else:
          # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
          # v_mad_i32_i24
          # module.add(VMadI32I24(dst=vgpr("ValuC+%u"%sumIdxV), src0=vgpr(dataV+0), src1=sgpr("Beta"), src2=vgpr("ValuC+%u"%sumIdxV), \
          #     comment="finalSum = sum*alpha + C*beta"))
          module.add(VMulLOU32(dst=vgpr(dataV+0), src0=sgpr("Beta"), src1=vgpr(dataV+0), comment="C = C*beta"))
          module.add(VAddU32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=vgpr("ValuC+%u"%newSumIdxV), comment="finalSum = sum*alpha + C*beta"))

      elif kernel["ProblemType"]["DestDataType"].isDouble():
        newSumIdxV = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
        # dataV+0 = new c = old c*beta
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV),2), src0=vgpr(dataV+0,2), src1=sgpr("Beta",2), src2=vgpr("ValuC+%u"%(newSumIdxV),2), \
            comment="finalSum = sum*alpha + C*beta"))

      # single precision complex
      elif kernel["ProblemType"]["DestDataType"].isSingleComplex():
        newSumIdxV = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV+0), src1=sgpr("Beta"), comment="finalSum Cr += old Cr * Br"))
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV+1), src1=sgpr("Beta+1").getMinus(), comment="finalSum Cr += old Ci * -Bi"))
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(dataV+1), src1=sgpr("Beta"), comment="finalSum Ci += old Ci * Br"))
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(dataV+0), src1=sgpr("Beta+1"), comment="finalSum Ci += old Cr * Bi"))

      # double precision complex
      elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
        newSumIdxV = sumIdxV * 4 - self.parentWriter.states.c.startVgprValu
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+0),2), src0=vgpr(dataV+0,2), src1=sgpr("Beta+0",2), src2=vgpr("ValuC+%u"%(newSumIdxV+0),2), comment="c.real += a.real * b.real"))
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+0),2), src0=vgpr(dataV+2,2), src1=sgpr("Beta+2",2), src2=vgpr("ValuC+%u"%(newSumIdxV+0),2), comment="c.real -= a.imag * b.imag"))
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+2),2), src0=vgpr(dataV+0,2), src1=sgpr("Beta+2",2), src2=vgpr("ValuC+%u"%(newSumIdxV+2),2), comment="c.imag += a.real * b.imag"))
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+2),2), src0=vgpr(dataV+2,2), src1=sgpr("Beta+0",2), src2=vgpr("ValuC+%u"%(newSumIdxV+2),2), comment="c.imag += a.imag * b.real"))
    return module

def copyData(computeDataType, elementSumIdx, gwvw, vgprStart, direction=0):
  module = Module("Copy Data")
  for vi in range(0, gwvw):
    sumIdxV = elementSumIdx + vi
    if computeDataType.isHalf() or computeDataType.isBFloat16():
      if (sumIdxV % 2 != 0):
        continue
      vgprIdx = elementSumIdx + vi // 2
      module.add(VMovB32(dst=vgpr(vgprStart + (vi // 2)), src=vgpr(vgprIdx)))
    elif computeDataType.isSingle():
      vgprIdx = sumIdxV
      module.add(VMovB32(dst=vgpr(vgprStart + vi), src=vgpr(vgprIdx)))
    elif computeDataType.isDouble():
      vgprIdx = elementSumIdx + vi * 2
      module.add(VMovB32(dst=vgpr(vgprStart + vi * 2), src=vgpr(vgprIdx)))
      module.add(VMovB32(dst=vgpr(vgprStart + vi * 2 + 1), src=vgpr(vgprIdx+1)))
    elif computeDataType.isInt32():
      vgprIdx = sumIdxV
      module.add(VMovB32(dst=vgpr(vgprStart + vi), src=vgpr(vgprIdx)))
    else:
      assert 0

  if direction == 1:
    for i in module.items():
      tmp = i.src[0]
      i.src[0] = i.dst
      i.dst = tmp
  return module

def convertData(gwvw, elementSumIdx, cvtType: CvtType, roundType: RoundType = RoundType.ROUND_UP, inputPrefix="", prefixOffset=0):
  module = Module("ConvertData")
  for vi in range(0, gwvw):
    sumIdxV = elementSumIdx + vi
    formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
    if cvtType == CvtType.CVT_F32_to_I32:
        if roundType == RoundType.ROUND_TO_NEAREST_EVEN:
          module.add(VRndneF32(dst=vgpr(formatVgpr), src=vgpr(formatVgpr), comment=" round to even"))
        module.add(VCvtF32toI32(dst=vgpr(formatVgpr), src=vgpr(formatVgpr), comment=" convert fp32 to i32"))
    elif cvtType == CvtType.CVT_I32_to_F32:
        module.add(VCvtI32toF32(dst=vgpr(formatVgpr), src=vgpr(formatVgpr), comment=" convert to fp32"))
    else:
      #TODO add other convert types here.
      assert 0
  return module
