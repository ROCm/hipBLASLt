################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
from ..Utils import DataDirection
from ..TensileInstructions import Label, Module, EXEC, SDWAModifiers, VCC, SelectBit, \
                            vgpr, sgpr, replaceHolder, SaturateCastType, VCvtBF16toFP32, \
                            DataType, CvtType, RoundType
from ..TensileInstructions.Instructions import *
from ..AsmAddressCalculation import AddrCalculation
from ..Components.PackData import formatting, PackData_F16, PackData_BF16

from math import ceil
from ..TensileInstructions import log2

class GlobalWriteBatchComponent(GlobalWriteComponents):
  kernel = {"ProblemType": {"OperationType": "GEMM" }}
  def __call__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrE, addrD, addrC, addrBias, addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, isLocalBarrierInit: bool, \
    tmpVgpr, cvtVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
    packdata, parentWriter, factorDim) -> Module:
    return GlobalWriteBatchWriter(kernel, tPA, tPB, activation, ss, batchIdx, applyAlpha, \
      beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrE, addrD, addrC, addrBias, addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, isLocalBarrierInit, \
      tmpVgpr, cvtVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      packdata, parentWriter, factorDim).emit()

class GlobalWriteBatchWriter:
  def __init__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrE, addrD, addrC, addrBias, addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, isLocalBarrierInit: bool, \
    tmpVgpr, cvtVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      packdata, parentWriter, factorDim):
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
    self.addrE    = addrE
    self.addrD    = addrD
    self.addrC    = addrC
    self.addrBias = addrBias
    self.addrScaleAVec = addrScaleAVec
    self.addrScaleBVec = addrScaleBVec
    self.addrScaleAlphaVec = addrScaleAlphaVec
    self.isLocalBarrierInit  = isLocalBarrierInit
    self.activationSetPCStruct = activationSetPCStruct
    self.activationTypeStr     = activationTypeStr
    self.tmpVgpr = tmpVgpr
    self.cvtVgprStruct = cvtVgprStruct
    self.batchElementSgprs = batchElementSgprs
    self.tmpSgpr = tmpSgpr
    self.codeAccVgprRead = codeAccVgprRead
    self.codeMulAlpha = codeMulAlpha
    self.packdata     = packdata
    self.parentWriter = parentWriter
    self.storesIssued = 0
    self.factorDim = factorDim

    # Internal state for GlobalWriteBatch
    # 0 for None, 1 for WorkGroupReduction = False, 2 for WorkGroupReduction = True
    self.storeBiasD = 0
    if self.parentWriter.states.useBias == DataDirection.WRITE and \
      (not self.kernel["WorkGroupReduction"]) and \
      self.kernel["ProblemType"]["BiasSrc"] == "D":
      self.storeBiasD = 1



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

  def GSUSynccodegen(self, labelend, vgprstart, globalOffset, vgproffset):
    module = Module("GSUSYNC")

    WaveNum = str(self.kernel["MIWaveGroup"][0]*self.kernel["MIWaveGroup"][1])

    module.addComment("check done start")

    #####################################synchronizer offset cal and set synchronizer#####################################
    #####################################WaveId+WgId*WaveNum+WgNum*WaveNum*Batch
    #####################################WgId+WaveId*WgNum+WgNum*WaveNum*Batch
    module.addComment("synchronizer offset cal")

    tmpS02 = self.parentWriter.sgprPool.checkOut(1, preventOverflow=False) #
    tmpS01 = self.parentWriter.sgprPool.checkOut(1, preventOverflow=False) #
    tmpS03 = self.parentWriter.sgprPool.checkOut(1, preventOverflow=False) #

    module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr("NumWorkGroups1"), src1=sgpr("NumWorkGroups0"), comment=""))
    module.add(SMulI32(dst=sgpr(tmpS02), src0=sgpr(tmpS03), src1=sgpr("WorkGroup2"), comment=""))
    module.add(SMulI32(dst=sgpr(tmpS01), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0"), comment=""))
    module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr("WorkGroup0")))
    module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr(tmpS02)))

    module.add(VReadfirstlaneB32(dst=sgpr(tmpS02), src=vgpr("Serial")))

    module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr(tmpS03), src1=sgpr("SizeK"), comment="cal a wave offset"))
    module.add(SLShiftRightB32(dst=sgpr(tmpS02), shiftHex=hex(log2(self.kernel["WavefrontSize"])), src=sgpr(tmpS02)))
    module.add(SMulI32(dst=sgpr(tmpS02), src0=sgpr(tmpS03), src1=sgpr(tmpS02), comment="wave offset at batch")) # WaveId*WgNum
    module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS02), src1=sgpr(tmpS01))) # WaveId*WgNum+WgId
    if self.batchIdx > 0:
      module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr(tmpS03), src1=int(WaveNum), comment="cal a batch offset")) # WgNum*WaveNum
      module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr(tmpS03), src1=self.batchIdx, comment="this batch offset")) # WgNum*WaveNum*Batch
      module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr(tmpS03))) # WaveId*WgNum+WgId + WgNum*WaveNum*Batch
    module.add(SLShiftLeftB32(dst=sgpr(tmpS01), src=sgpr(tmpS01), shiftHex=hex(2), comment="")) # atomic 32bits
    #####################################set synchronizer
    module.add(SAddU32(dst=sgpr("SrdSync+0"), \
                                    src0=sgpr("Synchronizer+0"), \
                                    src1=sgpr(tmpS01), \
                                    comment="" ))
    module.add(SAddCU32(dst=sgpr("SrdSync+1"), \
                        src0=sgpr("Synchronizer+1"), \
                        src1=hex(0), \
                        comment="" ))

    module.add(SWaitCnt(waitAll=True, comment="wait store done before synchronizer start load and add"))
    module.add(SAndB32(dst=sgpr(tmpS02), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))
    module.add(SSubU32(dst=sgpr(tmpS02), src0=sgpr(tmpS02), src1=hex(1), comment=""))
    module.add(SAtomicDec(dst=sgpr(tmpS02), base=sgpr("SrdSync", 2), smem=SMEMModifiers(glc=1)))
    module.addSpaceLine()
    #####################################cal synchronizer sum offset#####################################
    module.addComment("synchronizer sum offset cal")

    tmpS04 = self.parentWriter.sgprPool.checkOutAligned(2,2, preventOverflow=False) #
    tmpS05 = self.parentWriter.sgprPool.checkOutAligned(2,2, preventOverflow=False) #

    indices = list(range(0, self.kernel["ProblemType"]["NumIndicesC"]))
    numDim = len(indices)
    with self.parentWriter.allocTmpSgpr(5) as tmpSgprInfo:
      tmpSgpr = tmpSgprInfo.idx
      module.addModuleAsFlatItems(self.parentWriter.s_mul_u64_u32(sgpr(tmpSgpr+0), sgpr(tmpSgpr+1), sgpr("SizesFree+0"), 1, "Free0"))
      for i in range(1, numDim):
        module.add(SSubU32(dst=sgpr(tmpSgpr+4), src0=sgpr("SizesFree+%u"%i), src1=1, comment="Free%u" % i))
        module.add(SMulI32(dst=sgpr(tmpSgpr+4), src0=sgpr(tmpSgpr+4), src1=1, comment="Free%u" % i))
        module.addModuleAsFlatItems(self.parentWriter.s_mul_u64_u32(sgpr(tmpSgpr+2), sgpr(tmpSgpr+3), sgpr(tmpSgpr+4), sgpr("StrideC%s"%self.parentWriter.states.indexChars[i]), "Free%u" % i))
        module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=sgpr(tmpSgpr+0), src1=sgpr(tmpSgpr+2), comment="Free%u" % i))
        module.add(SAddCU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+1), src1=sgpr(tmpSgpr+3), comment="Free%u" % i))

      bpetmp = int(self.parentWriter.states.bpr * self.kernel["ProblemType"]["DestDataType"].numRegisters()) # self.states.bpeCinternal
      module.add(SLShiftLeftB64(dst=sgpr(tmpS04,2), src=sgpr(tmpSgpr+0,2), shiftHex=log2(self.parentWriter.states.bpeCexternal), comment="scale by bpe"))

    module.addSpaceLine()
    #####################################cal synchronizer sum start#####################################
    # no need to do because we have workspace start sgpr
    #####################################check synchronizer done#####################################
    checkSyncCode = Module("check synchronizer done")
    checkSyncCode.addComment("check synchronizer done")

    checkSyncCode.add(SWaitCnt(lgkmcnt=0, comment="Wait for synchronizer"))
    checkSyncCode.add(SCmpEQU32(
        src0=sgpr(tmpS02), \
        src1=hex(1), \
        comment=""))

    # checkSyncCode.add(SCBranchSCC0(labelName=labelendname, comment=""))
    checkSyncCode.add(self.parentWriter.longBranchScc0(label=labelend, posNeg=1, comment="long branch sync"))

    checkSyncCode.addComment("check done end")
    checkSyncCode.addSpaceLine()
    #####################################load buffer#####################################
    checkSyncCode.addComment("buffer load start")
    SyncloadedData = 0

    tmpS06 = self.parentWriter.sgprPool.checkOutAligned(4,4, preventOverflow=False) #overflow?
    addr1 = sgpr(tmpS06, 4)
    addr0 = vgpr(vgproffset)
    bps = self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.gwvw
    for elementIdx in range(0, len(self.batchElements)):
      mask     = self.ss.elementMask[elementIdx]
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      SyncloadedData = 0

      SynchronizerAddSkiplabelString = "Synchronizer_read_add_skip"
      SynchronizerAddSkipComment = "Synchronizer read add skip"
      SynchronizerAddSkiplabel = Label(self.parentWriter.labels.getNameInc(SynchronizerAddSkiplabelString), SynchronizerAddSkipComment)

      addr0 = vgpr(addrCalc.addrDVgpr)

      GSUtotal = 16
      if (self.kernel["MIWaveTile"][0]*self.kernel["MIWaveTile"][1])*(self.kernel["MIWaveGroup"][0]*self.kernel["MIWaveGroup"][1]) > 8:
        GSUtotal = int(GSUtotal/int((self.kernel["MIWaveTile"][0]*self.kernel["MIWaveTile"][1])*(self.kernel["MIWaveGroup"][0]*self.kernel["MIWaveGroup"][1])/8))
      GSUtotal = max(2,GSUtotal)
      SynchronizerAddEndlabel = [""] * GSUtotal

      for idx in range(0, GSUtotal):
        SynchronizerAddEndlabelString = "Synchronizer_read_add_end_"+str(idx+1)
        SynchronizerAddEndComment = "Synchronizer read add end_"+str(idx+1)
        SynchronizerAddEndlabel[idx] = Label(self.parentWriter.labels.getNameInc(SynchronizerAddEndlabelString), SynchronizerAddEndComment)

      bufferOOB = self.parentWriter.vgprPool.checkOut(1, "BufferOOB")
      module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))

      module.add(SMovB32(sgpr(tmpS06+0), sgpr("WSDstart+0"), "Move workspace start"))
      module.add(SMovB32(sgpr(tmpS06+1), sgpr("WSDstart+1"), "Move workspace start"))
      module.add(SMovB32(sgpr(tmpS06+2), sgpr("SrdD+2"), ""))
      module.add(SMovB32(sgpr(tmpS06+3), sgpr("SrdD+3"), ""))

      if elementIdx == 0:
        # Insert check synchronizer done code here for better scheduling
        module.add(checkSyncCode)

      for times in range(elementIdx, elementIdx+1):
        addrCalctmp: AddrCalculation = self.ss.elementAddr[times]
        if self.ss.optSrdIncForRow and addrCalctmp.rowInc:
          module.add(addrCalctmp.incrementToNextRow(self.kernel, "D", self.ss, tmpS05, dst=tmpS06))
          module.add(SAddU32(dst=sgpr("WSDstart+0"), \
                                            src0=sgpr("WSDstart+0"), \
                                            src1=sgpr(tmpS05), \
                                            comment="" ))
          module.add(SAddCU32(dst=sgpr("WSDstart+1"), \
                              src0=sgpr("WSDstart+1"), \
                              src1=hex(0), \
                              comment="" ))

      vgprstart = self.ss.elementSumIdx[elementIdx] #here
      dataType     = self.kernel["ProblemType"]["DestDataType"]
      if dataType.isDouble() or dataType.isSingleComplex():
        vgprstart = vgprstart*2
      module.add(self.parentWriter.chooseGlobalRead(True, bps, vgprstart, \
                      addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=1, slc=1,\
                      comment="load GSU D 0 "+str(vgprstart)))
      SyncloadedData += 1

      GSUMvgpr = self.parentWriter.vgprPool.checkOut(1, "GSUMvgpr")
      module.add(SAndB32(dst=sgpr("GSUSync"), src0=sgpr("GSU"), src1=hex(0x3FFF), comment="Restore GSU"))

      SynchronizerlabelString = "Synchronizer_read_add"
      SynchronizerComment = "Synchronizer read add"
      Synchronizerlabel = Label(self.parentWriter.labels.getNameInc(SynchronizerlabelString), SynchronizerComment)

      if(self.kernel["ProblemType"]["DestDataType"].numRegisters() > 1):
        tmpVAdd = self.parentWriter.vgprPool.checkOutAligned((GSUtotal-1)*self.gwvw*self.kernel["ProblemType"]["DestDataType"].numRegisters(), 4)
      else:
        tmpVAdd = self.parentWriter.vgprPool.checkOutAligned((GSUtotal-1)*self.gwvw, 4)

      GSUP1 = GSUtotal-1

      for i in range(0,GSUP1):
        module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1, comment="%u" % i))

        module.add(SAddU32(dst=sgpr(tmpS06+0), \
                                        src0=sgpr(tmpS06+0), \
                                        src1=sgpr(tmpS04+0), \
                                        comment="" ))
        module.add(SAddCU32(dst=sgpr(tmpS06+1), \
                            src0=sgpr(tmpS06+1), \
                            src1=sgpr(tmpS04+1), \
                            comment="" ))

        module.add(SCmpEQI32(src0=sgpr("GSUSync"), src1=0, comment=""))#GSUSync+GSUP1==GSU
        module.add(SCBranchSCC1(labelName=SynchronizerAddEndlabel[i].getLabelName(), comment="SyncAddbranchhere"))

        if(self.kernel["ProblemType"]["DestDataType"].numRegisters() > 1):
          module.add(self.parentWriter.chooseGlobalRead(True, bps, tmpVAdd+self.gwvw*self.kernel["ProblemType"]["DestDataType"].numRegisters()*i, \
                        addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=1, slc=1, \
                        comment="load GSU DD %u %u %u" % (bps, self.gwvw, self.kernel["ProblemType"]["DestDataType"].numRegisters())))
        else:
          module.add(self.parentWriter.chooseGlobalRead(True, bps, tmpVAdd+self.gwvw*i, \
                        addr0, addr1, soffset=0, offset=addrCalc.globalOffset, glc=1, slc=1, \
                        comment="load GSU DD %u %u %u" % (bps, self.gwvw, self.kernel["ProblemType"]["DestDataType"].numRegisters())))

        SyncloadedData += 1
      module.addComment("buffer load end\n")

      #####################################> GSUtotal reduction start#####################################
      module.addComment("buffer add start")
      vscnt = 0
      lgkmcnt = -1
      vmcnt = SyncloadedData = SyncloadedData -1

      module.add(Synchronizerlabel)

      for i in range(0, GSUP1):
        module.addSpaceLine()
        vmcnt = SyncloadedData = SyncloadedData -1
        module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))

        if ((self.gwvw % 2) == 1):
          for j in range(0, int(self.gwvw)):
            module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+self.gwvw*i+j), \
                          comment="buffer add"))
        else:
          for j in range(0, int(self.gwvw/2)):
            module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                              src1=vgpr(tmpVAdd+0+self.gwvw*i+j*2, 2), comment="buffer pk"))

        module.add(SSubI32(dst=sgpr("GSUSync"), src0=sgpr("GSUSync"), src1=1, comment="%u" % i))
        module.add(SCmpLeI32(src0=sgpr("GSUSync"), src1=0-(GSUP1-1), comment=""))#GSUSync+GSUP1==GSU
        module.add(SCBranchSCC1(labelName=SynchronizerAddSkiplabel.getLabelName(), comment="SyncAddbranch"))

        module.add(SAddU32(dst=sgpr(tmpS06+0), \
                                        src0=sgpr(tmpS06+0), \
                                        src1=sgpr(tmpS04+0), \
                                        comment="" ))
        module.add(SAddCU32(dst=sgpr(tmpS06+1), \
                            src0=sgpr(tmpS06+1), \
                            src1=sgpr(tmpS04+1), \
                            comment="" ))

        module.add(VCmpGEI32(dst=sgpr(tmpS05,2), src0=0, src1=sgpr("GSUSync"), comment=""))
        module.add(VCndMaskB32(
              dst=vgpr(GSUMvgpr), \
              src1=vgpr(bufferOOB), \
              src0=addr0, \
              src2=sgpr(tmpS05,2), \
              comment="protect if OOB"))

        if(self.kernel["ProblemType"]["DestDataType"].numRegisters() > 1):
            module.add(self.parentWriter.chooseGlobalRead(True, bps, tmpVAdd+self.gwvw*self.kernel["ProblemType"]["DestDataType"].numRegisters()*i, \
                        vgpr(GSUMvgpr), addr1, soffset=0, offset=addrCalc.globalOffset, glc=1, slc=1, \
                        comment="load GSU DD %u" % bps))
        else:
          module.add(self.parentWriter.chooseGlobalRead(True, bps, tmpVAdd+self.gwvw*i, \
                        vgpr(GSUMvgpr), addr1, soffset=0, offset=addrCalc.globalOffset, glc=1, slc=1, \
                        comment="load GSU DD %u" % bps))

        SyncloadedData += 1

      module.addComment("buffer add end\n")

      module.add(SCmpGtI32(
        src0=sgpr("GSUSync"), \
        src1=hex(1-(GSUP1)), \
        comment=""))
      module.add(SCBranchSCC1(labelName=Synchronizerlabel.getLabelName(), comment="Syncbranchhere"))

      #####################################< GSUtotal reduction start#####################################
      for k in range(GSUtotal-2, -1, -1):
        module.addSpaceLine()
        module.add(SynchronizerAddEndlabel[k])

        vmcnt = k
        for i in range(0, k):
          module.addSpaceLine()
          vmcnt = vmcnt -1 if vmcnt > 0 else 0
          module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(wait for buffer ready)"))

          if ((self.gwvw % 2) == 1):
            for j in range(0, int(self.gwvw)):
              module.add(VAddF32(dst=vgpr(vgprstart+j), src0=vgpr(vgprstart+j), src1=vgpr(tmpVAdd+0+self.gwvw*i+j), \
                            comment="buffer add"))
          else:
            for j in range(0, int(self.gwvw/2)):
              module.add(VAddPKF32(dst=vgpr(vgprstart+j*2, 2), src0=vgpr(vgprstart+j*2, 2), \
                                src1=vgpr(tmpVAdd+0+self.gwvw*i+j*2, 2), comment="buffer pk"))

          if i == k-1:
            module.add(SBranch(labelName=SynchronizerAddSkiplabel.getLabelName(), comment="SyncAddbranch"))

      module.add(SynchronizerAddSkiplabel)

      self.parentWriter.vgprPool.checkIn(bufferOOB)
      self.parentWriter.vgprPool.checkIn(GSUMvgpr)
      module.addComment("buffer add end2\n")

      self.parentWriter.vgprPool.checkIn(tmpVAdd)
    self.parentWriter.sgprPool.checkIn(tmpS06)
    self.parentWriter.sgprPool.checkIn(tmpS05)
    self.parentWriter.sgprPool.checkIn(tmpS04)
    self.parentWriter.sgprPool.checkIn(tmpS03)
    self.parentWriter.sgprPool.checkIn(tmpS02)
    self.parentWriter.sgprPool.checkIn(tmpS01)

    return module

  def _prolog(self, module: Module):
    module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u factorDim=%u" % \
              (self.ss.optSingleColVgpr, self.ss.optSharedColVgpr, self.ss.optSGPRUsage, self.ss.optSrdIncForRow, self.factorDim))

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

    if self.kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel":
      self.ss.setupStoreElementsForBatch(self.kernel, self.gwvw, self.batchElements, self.batchElementSgprs, isOptNLL=False, factorDim=self.factorDim)
    else:
      self.ss.setupStoreElementsForBatch(self.kernel, self.gwvw, self.batchElements, self.batchElementSgprs, isOptNLL=True, factorDim=self.factorDim)

    self.localLoadsBiasIssued = 0
    self.storesIssued    = 0
    self.loadsBetaIssued   = 0
    self.loadsEIssued      = 0
    self.loadsScaleAVecIssued = 0
    self.loadsScaleBVecIssued = 0
    self.loadsScaleAlphaVecIssued     = 0

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

    self.betaLoadIssued = []
    self.eLoadIssued = []
    self.biasLoadIssued = []
    self.scaleAVecLoadIssued = []
    self.scaleBVecLoadIssued = []
    self.scaleAlphaVecLoadIssued = []
    loadedDataBeta = {}
    loadedDataE = {}
    loadedDataBias = {}
    loadedDataScaleAVec = {}
    loadedDataScaleBVec = {}
    loadedDataScaleAlphaVec = {}

    if self.kernel["BufferStore"] and self.edge:
      bufferOOB = self.parentWriter.vgprPool.checkOut(1, "BufferOOB")
      module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))
    else:
      bufferOOB = None
    #when factorDim = 1 the bias's gwvw is alwasy be 1.
    factor_gwvw = 1 if self.factorDim else self.ss.cfg.gwvw
    for elementIdx, element in enumerate(self.batchElements):
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addrCVgpr    = addrCalc.addrCVgpr
      addrDVgpr    = addrCalc.addrDVgpr
      addrEVgpr    = addrCalc.addrEVgpr
      addrBiasVgpr = addrCalc.addrBiasVgpr
      addrScaleAVecVgpr = addrCalc.addrScaleAVecVgpr
      addrScaleBVecVgpr = addrCalc.addrScaleBVecVgpr
      addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
      data     = self.ss.elementData[elementIdx]
      dataBeta = self.ss.elementData[elementIdx]
      dataE    = self.ss.elementDataE[elementIdx]
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleAVec = self.ss.elementDataScaleAVec[elementIdx]
      dataScaleBVec = self.ss.elementDataScaleBVec[elementIdx]
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      mask     = self.ss.elementMask[elementIdx]
      vc0 = element[3]
      sumIdxGSUSYNC = self.ss.elementSumIdx[elementIdx]

      module.add(addrCalc.emitAddressSetupCode(self.kernel, self.tPB, self.ss, self.tmpVgpr, self.tmpS01, self.edge, self.beta, self.atomic, elementIdx, addrDVgpr))

      if self.edge:
        module.add(addrCalc.edgeProtectCode(self.kernel, self.edge, self.beta, self.atomic, mask, self.tmpSgpr))
        if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
          module.addComment1("edge Protect")
      # create code Module to push mov vgpr,acc instructions
      if self.beta:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'C', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrCVgpr, self.addrC, 0))
        if dataBeta not in loadedDataBeta:
          if self.kernel["GroupLoadStore"]:
            loadInputCode.add(self.parentWriter.readInput(self.kernel, self.ss, 'C', self.kernel["ProblemType"]["DestDataType"], addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
          else:
            module.add(self.parentWriter.readInput(self.kernel, self.ss, 'C', self.kernel["ProblemType"]["DestDataType"], addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
          loadedDataBeta[dataBeta] = ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsBetaIssued += ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.gwvw / 16)
      self.betaLoadIssued.append(len(loadedDataBeta) * ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.ss.cfg.gwvw / 16))

      if (self.kernel["ProblemType"]["UseE"] and self.kernel["ProblemType"]["Gradient"] and self.kernel["ProblemType"]["ActivationType"] != 'none') and (self.kernel["GlobalSplitU"] == 1):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'E', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrEVgpr, self.addrE, 0))
        if dataE not in loadedDataE:
          loadOffset = int((self.kernel["ProblemType"]["ComputeDataType"].numRegisters() - self.kernel["ProblemType"]["DataTypeE"].numRegisters()) * self.ss.cfg.gwvw)
          if self.kernel["GroupLoadStore"]:
            loadInputCode.add(self.parentWriter.readInput(self.kernel, self.ss, 'E', self.kernel["ProblemType"]["DataTypeE"], addrCalc, vc0, dataE + loadOffset, self.gwvw, addrEVgpr, self.tmpS01))
          else:
            module.add(self.parentWriter.readInput(self.kernel, self.ss, 'E', self.kernel["ProblemType"]["DataTypeE"], addrCalc, vc0, dataE + loadOffset, self.gwvw, addrEVgpr, self.tmpS01))
          loadedDataE[dataE] = ceil(self.kernel["ProblemType"]["DataTypeE"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsEIssued += ceil(self.kernel["ProblemType"]["DataTypeE"].numBytes() * self.gwvw / 16)
        self.loadE = True
      else:
        self.loadE = False
      self.eLoadIssued.append(len(loadedDataE) * ceil(self.kernel["ProblemType"]["DataTypeE"].numBytes() * self.ss.cfg.gwvw / 16))

      def addEpilogueLoad(modGwvw, ldName: str, addrVecVgpr, addrVec, dataVec, loadedDataVec, vecOffset, gwvw, referenceVgpr, referenceDim, skipLoad=False, comment=""):
        loadsIssued = 0
        module.add(addrCalc.emitLdChangeReference(self.kernel, self.ss, ldName, self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrVecVgpr, addrVec, self.factorDim, referenceVgpr, referenceDim))
        if dataVec not in loadedDataVec:
          if self.kernel["GroupLoadStore"]:
            # Group bias load with C input to
            if ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")) and (not self.isLocalBarrierInit):
              loadInputCode.add(SWaitCnt(lgkmcnt=0, comment="Wait for LDS write"))
              loadInputCode.add(SBarrier("LDS write barrier"))
              self.isLocalBarrierInit = True
            loadInputCode.add(self.parentWriter.addLdsLoad(self.kernel["ProblemType"]["ComputeDataType"], dataVec, addrVecVgpr, vecOffset, gwvw, comment=comment))
          else:
            if ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")) and (not self.isLocalBarrierInit):
              module.add(SWaitCnt(lgkmcnt=0, comment="Wait for LDS write"))
              module.add(SBarrier("LDS write barrier"))
              self.isLocalBarrierInit = True
            module.add(self.parentWriter.addLdsLoad(self.kernel["ProblemType"]["ComputeDataType"], dataVec, addrVecVgpr, vecOffset, gwvw, comment=comment))
          loadedDataVec[dataVec] = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw / 16)
          loadsIssued = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw / 16)
          if (self.ss.cfg.gwvw != gwvw) and (not skipLoad):
            remain_load = self.ss.cfg.gwvw - 1
            bpl = self.kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw
            bpr = ceil(bpl / self.parentWriter.states.bpr)
            #For below ds_read instruction do not add bias issued , because of all ds_load instructions need to be completed at the same time in this batch.
            for r in range(remain_load):
              modGwvw.add(self.parentWriter.addLdsLoad(self.kernel["ProblemType"]["ComputeDataType"], dataVec, addrVecVgpr, vecOffset, factor_gwvw, comment=comment))
        return loadsIssued

      skipLoad = True if self.factorDim else False

      modGwvwScale = []
      biasReferenceVgpr = None
      if self.parentWriter.states.useBias == DataDirection.READ:
        biasReferenceVgpr = addrBiasVgpr
        modGwvwBias = Module("GwvwBias")
        self.localLoadsBiasIssued += addEpilogueLoad(modGwvwBias, 'Bias', addrBiasVgpr, self.addrBias, dataBias, loadedDataBias, addrCalc.biasOffset[self.factorDim], factor_gwvw, None, self.factorDim, skipLoad=skipLoad, comment="load Bias")
        modGwvwScale.append(modGwvwBias)
      self.biasLoadIssued.append(len(loadedDataBias) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw / 16))

      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        modGwvwScaleAlpha = Module("GwvwScaleAlpha")
        self.loadsScaleAlphaVecIssued += addEpilogueLoad(modGwvwScaleAlpha, "ScaleAlphaVec", addrScaleAlphaVecVgpr, self.addrScaleAlphaVec, dataScaleAlphaVec, loadedDataScaleAlphaVec, addrCalc.scaleAlphaVecOffset[self.factorDim], factor_gwvw, biasReferenceVgpr, self.factorDim, skipLoad=skipLoad, comment="load scaleAlpha")
        modGwvwScale.append(modGwvwScaleAlpha)
      self.scaleAlphaVecLoadIssued.append(len(loadedDataScaleAlphaVec) if self.factorDim else len(loadedDataScaleAlphaVec) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw / 16))

      if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        modGwvwScaleA = Module("GwvwScaleA")
        modGwvwScaleB = Module("GwvwScaleB")
        self.loadsScaleAVecIssued += addEpilogueLoad(modGwvwScaleA, "ScaleAVec", addrScaleAVecVgpr, self.addrScaleAVec, dataScaleAVec, loadedDataScaleAVec, addrCalc.scaleAVecOffset, self.ss.cfg.gwvw, biasReferenceVgpr, self.factorDim, comment="load scaleA")
        self.loadsScaleBVecIssued += addEpilogueLoad(modGwvwScaleB, "ScaleBVec", addrScaleBVecVgpr, self.addrScaleBVec, dataScaleBVec, loadedDataScaleBVec, addrCalc.scaleBVecOffset, 1, biasReferenceVgpr, self.factorDim, skipLoad=True, comment="load scaleB")
        modGwvwScale.append(modGwvwScaleA)
        modGwvwScale.append(modGwvwScaleB)
      self.scaleAVecLoadIssued.append(len(loadedDataScaleAVec) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16))
      self.scaleBVecLoadIssued.append(len(loadedDataScaleBVec))

      # Reorder scale
      length = 0
      for mod in modGwvwScale:
        length = max(length, len(mod.items()))

      for index in range(0, length):
        for mod in modGwvwScale:
          if len(mod.items()) > index:
            module.add(mod.items()[index])

      # This is a helper function that generates vector global read
      # The following is an example of how to use scaleVecPattern
      # We are changing scaleAlphaVector to local read
      # if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
      #   modGwvwScaleAlpha = Module("GwvwScaleAlpha")
      #   self.loadsScaleAlphaVecIssued += scaleVecPattern(modGwvwScaleAlpha, "AlphaVec", "Alpha", dataScaleAlphaVec, self.addrScaleAlphaVec, loadedDataScaleAlphaVec, addrScaleAlphaVecVgpr, addrCalc.scaleAlphaVecOffset[self.factorDim], factor_gwvw, True, skipLoad=skipLoad)
      #   modGwvwScale.append(modGwvwScaleAlpha)
      # self.scaleAlphaVecLoadIssued.append(len(loadedDataScaleAlphaVec) if self.factorDim else len(loadedDataScaleAlphaVec) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw / 16))

      def scaleVecPattern(modGwvw, name: str, srdName: str, dataScaleVec, addrScaleVec, loadedDataScaleVec, addrScaleVecVgpr, scaleVecOffset, factor_gwvw, addVecPostFix, skipLoad=False):
        loadsScaleVecIssued = 0
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'Scale%sVec'%srdName, self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrScaleVecVgpr, addrScaleVec, self.factorDim))
        if dataScaleVec not in loadedDataScaleVec:
          # Shift right several vgprs for cvt ops if needed
          numVgprs = int(ceil(self.kernel["ProblemType"]["ComputeDataType"].numRegisters() * self.ss.cfg.gwvw))
          reg = self.kernel["ProblemType"]["ComputeDataType"].numRegisters() if self.kernel["ProblemType"]["ComputeDataType"].numRegisters() >= 1 else 1
          gprShiftScaleVec = dataScaleVec + (self.ss.cfg.gwvw * reg - numVgprs)
          if self.kernel["GroupLoadStore"]:
            # Group scaleVec load with C input to
            loadInputCode.add(self.parentWriter.addScaleVecLoad(self.kernel, self.ss, name, srdName, addrScaleVecVgpr, gprShiftScaleVec, factor_gwvw, scaleVecOffset, addVecPostFix))
          else:
            module.add(self.parentWriter.addScaleVecLoad(self.kernel, self.ss, name, srdName, addrScaleVecVgpr, gprShiftScaleVec, factor_gwvw, scaleVecOffset, addVecPostFix))
          loadedDataScaleVec[dataScaleVec] = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw / 16)
          loadsScaleVecIssued = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw / 16)
          if (self.ss.cfg.gwvw != factor_gwvw) and (not skipLoad):
            bpl = self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw
            bpr = ceil(bpl / self.parentWriter.states.bpr)
            #For below ds_read instruction do not add bias issued , because of all ds_load instructions need to be completed at the same time in this batch.
            for r in range(self.ss.cfg.gwvw - 1):
              modGwvw.add(self.parentWriter.addScaleVecLoad(self.kernel, self.ss, name, srdName, addrScaleVecVgpr, gprShiftScaleVec  + (r + 1) * bpr, factor_gwvw, scaleVecOffset, addVecPostFix))
        return loadsScaleVecIssued

      if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'E', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrEVgpr, self.addrE, 0))
      if self.storeBiasD == 1:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'Bias', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrBiasVgpr, self.addrBias, self.factorDim))
      module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'D', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrDVgpr, self.addrD, 0))
      if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'TD', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrCalc.addrGSUSyncVgprs, self.addrD, 0))

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
        if self.edge and (self.beta or self.loadE or self.atomic):
          module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -1 -> exec"))

      if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
        if self.ss.optSrdIncForRow and addrCalc.rowInc and self.kernel["StoreRemapVectorWidth"] > 0:
          module.addComment1("StoreRemap: shift coord1 address MultipleBufferSingleKernel")
          if self.kernel["ProblemType"]["UseE"] and (self.kernel["GlobalSplitU"] == 1):
            printExit("Use E does not support StoreRemapVectorWidth if GSU == 1.")
            # module.add(addrCalc.incrementToNextRow(self.kernel, "E", self.ss, self.tmpS01, isCompute=True))
          module.add(addrCalc.incrementToNextRow(self.kernel, "D", self.ss, self.tmpS01))
          module.add(VMovB32(vgpr(self.tmpVgpr), addrCalc.rowInc, "set shift rows"))
          module.add(VAddU32(vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.tmpVgpr), "shift storeRemap coord1"))

    if self.kernel["BufferStore"] and self.edge:
      self.parentWriter.vgprPool.checkIn(bufferOOB)

    module.add(loadInputCode)

    if self.beta and self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt(module)

    ########################################
    # AccVgpr read
    if self.codeAccVgprRead is not None and self.kernel["LocalSplitU"] == 1:
      regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
      # loop over store instructions within one batch
      for elementIdx in range(len(self.batchElements)):
        # loop over scalars within one store instruction
        for vi in range(self.gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            module.add(replaceHolder(self.codeAccVgprRead.items().pop(0), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - self.parentWriter.states.c.startVgprValu))
    elif self.kernel["LocalSplitU"] > 1:
      # read from LSU VGPRs
      regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
      if self.ss.lsuStartVgprOffset > 0:
        for elementIdx in range(len(self.batchElements)):
          for vi in range(self.gwvw):
            for rIdx in range(0, regsPerScalar):
              idx = self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - self.parentWriter.states.c.startVgprValu
              module.add(VMovB32(vgpr("ValuC+%u"%(idx)), vgpr("ValuC+%u"%(idx + self.ss.lsuStartVgprOffset)), "load from "+str(idx + self.ss.lsuStartVgprOffset)+" to "+str(idx) ))
      self.ss.lsuStartVgprOffset += len(self.batchElements) * self.gwvw * regsPerScalar

      if not self.kernel["MIArchVgpr"]:
        module.add(SNop(1, "2 wait states required before reading vgpr"))

    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
      module.addComment1("store after Acc, "+"GSU: "+str(self.kernel["GlobalSplitU"]))

    storeCodeGSUSK = Module("GroupLoadStore")
    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":#GSUGSU
      module.addComment1("store after Acc, "+"GSU: "+str(self.kernel["GlobalSplitU"]))
      for elementIdx in range(0, len(self.batchElements)):
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
          vgprIdx = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
          vgprDst = self.activationSetPCStruct.vgprActCopy if mergeActFuncCall else "ValuC+%d"%vgprIdx
          module.add(self.parentWriter.addStore(self.kernel, self.ss, 'E', addrCalc, vgprDst, self.tmpS01, self.edge, comment="store E"))

        sumIdx = self.ss.elementSumIdx[elementIdx]
        if not self.kernel["StoreRemapVectorWidth"]:
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'D', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store D %u" %sumIdx) #here
          if self.kernel["GroupLoadStore"]:
            storeCodeGSUSK.add(tmpStoreCode)
          else:
            module.addSpaceLine()
            module.add(tmpStoreCode)
        else:
          rpe = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr
          module.add(self.parentWriter.storeRemapAddLocalWrite(self.kernel, self.ss, addrCalc, sumIdx*rpe))
          # Column Block Shape has been written to LDS
          # Now read back and write out to global memory
      module.add(storeCodeGSUSK)

    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel" and self.kernel["StoreRemapVectorWidth"]:
      if self.parentWriter.StoreRemapLastBatch == 1:
        module.addComment1("Handle local read and global write")
        storeModule, numNewStores = self.parentWriter.storeRemapAddStore(self.kernel, self.tmpVgpr, self.tmpS01, self.edge, self.parentWriter.StoreRemapLastBatch)
        module.add(storeModule)
        self.storesIssued += numNewStores

    if (self.kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel'):
      if self.parentWriter.states.serializedStore:
        module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

      ########################################################

      module.addSpaceLine()
      SynchronizerEndlabelString = "Sync_EDN%s%s" % ("_Beta" if self.beta else "", "_Edge" if self.edge else "" )
      SynchronizerEndlabelComment = "Sync_EDN"
      SynchronizerEndlabel = Label(self.parentWriter.labels.getNameInc(SynchronizerEndlabelString), SynchronizerEndlabelComment)
      SynchronizerEndlabel = Label(self.parentWriter.labels.getName(SynchronizerEndlabelString), SynchronizerEndlabelComment)

      module.addselfAsm("//sourece store done, GSU:"+str(self.kernel["GlobalSplitU"])+"\n") #GSUSYNC
      module.addSpaceLine()

      module.add(self.GSUSynccodegen(SynchronizerEndlabel, sumIdxGSUSYNC, addrCalc.globalOffset, addrCalc.addrDVgpr))

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
    lastDataD       = -1
    lastDataE       = -1
    checkedDataBias = {}
    checkedDataScaleAVec = {}
    checkedDataScaleBVec = {}
    checkedDataScaleAlphaVec = {}
    for elementIdx in range(len(self.batchElements)):
      sumIdxGSUSYNC = self.ss.elementSumIdx[elementIdx]
      if not self.ss.sharedColDVgprs:
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        addrEVgpr    = addrCalc.addrEVgpr
        addrDVgpr    = addrCalc.addrDVgpr
        addrGSUSyncVgprs    = addrCalc.addrGSUSyncVgprs
        addrCVgpr    = addrCalc.addrCVgpr
        addrBiasVgpr = addrCalc.addrBiasVgpr
        addrScaleAVecVgpr = addrCalc.addrScaleAVecVgpr
        addrScaleBVecVgpr = addrCalc.addrScaleBVecVgpr
        addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
        if addrEVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrEVgpr)
        self.parentWriter.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.parentWriter.vgprPool.checkIn(addrCVgpr)
        if addrGSUSyncVgprs != None:
          self.parentWriter.vgprPool.checkIn(addrGSUSyncVgprs)
        if addrBiasVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrBiasVgpr)
        if addrScaleAVecVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleAVecVgpr)
        if addrScaleBVecVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleBVecVgpr)
        if addrScaleAlphaVecVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleAlphaVecVgpr)

      data = self.ss.elementData[elementIdx]
      if data != 0:
        if data != lastDataD:
          self.parentWriter.vgprPool.checkIn(data)
        lastDataD = data

      dataBias = self.ss.elementDataBias[elementIdx]
      if dataBias != 0:
        if dataBias not in checkedDataBias:
          self.parentWriter.vgprPool.checkIn(dataBias)
        checkedDataBias[dataBias] = 1

      dataE = self.ss.elementDataE[elementIdx]
      if dataE != 0:
        if dataE != lastDataE:
          self.parentWriter.vgprPool.checkIn(dataE)
        lastDataE = dataE

      def checkScaleVec(dataScaleVec, checkedDataScaleVec):
        if dataScaleVec != 0:
          if dataScaleVec not in checkedDataScaleVec:
            self.parentWriter.vgprPool.checkIn(dataScaleVec)
          checkedDataScaleVec[dataScaleVec] = 1

      checkScaleVec(self.ss.elementDataScaleAVec[elementIdx], checkedDataScaleAVec)
      checkScaleVec(self.ss.elementDataScaleBVec[elementIdx], checkedDataScaleBVec)
      checkScaleVec(self.ss.elementDataScaleAlphaVec[elementIdx], checkedDataScaleAlphaVec)

    self.ss.firstBatch = False
    self.ss.checkInTempVgprC()
    if self.kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel" and self.kernel["StoreRemapVectorWidth"]:
      if self.parentWriter.StoreRemapLastBatch == 1:
        module.addComment1("Handle local read and global write")
        # this seems buggy? it's possible to issue more than one stores for SR
        # module.add(self.storeRemapAddStore(kernel, tmpVgpr, tmpS01, edge))
        # storesIssued += 1
        storeModule, numNewStores = self.parentWriter.storeRemapAddStore(self.kernel, self.tmpVgpr, self.tmpS01, self.edge, self.parentWriter.StoreRemapLastBatch)
        module.add(storeModule)
        self.storesIssued += numNewStores

    if self.parentWriter.states.serializedStore:
      module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
      module.addselfAsm("//GW end\n") #GSUSYNC

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
    # Here we wait all
    if not interleaveStoreVmcnt:
      vmcnt = -1
      lgkmcnt = -1
      commentList = []
      # Global read wait
      if self.beta:
        vmcnt = 0
        commentList.append("Beta")
      if self.loadE:
        vmcnt = 0
        commentList.append("E")
      # Local read wait
      if self.parentWriter.states.useBias == DataDirection.READ:
        lgkmcnt = 0
        commentList.append("Bias LDS")
      if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        lgkmcnt = 0
        commentList.append("ScaleABVec")
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        lgkmcnt = 0
        commentList.append("ScaleAlphaVec")
      if (vmcnt != -1) or (lgkmcnt != -1):
        # Get comment
        comment = "wait for " + commentList[0]
        for c in commentList[1:]:
          comment += ", %s"%c
        module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=-1, comment=comment))

    module.addComment1("apply mask, calc new C and issue writes")
    # module.add(self.getBomb()) # can see store addresses just before the store inst

    activationCDataType = self.kernel["ProblemType"]["ActivationComputeDataType"]

    if self.kernel["ProblemType"]["DestDataType"].isBFloat16() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
      module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" ))
      module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp32Nan), "0x7fff0000", "fp32 Nan" ))
      module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" ))
    elif self.kernel["ProblemType"]["DestDataType"].isFloat8() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
      module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8NanInf), "0x207", "Nan and +/- inf" ))
      if self.parentWriter.states.archCaps["HasFP8_OCP"]:
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Max), "0x43E00000", "Fp8 Max value 448 as float32" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Min), "0xc3E00000", "Fp8 Min value -448 as float32" ))
      else:
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Max), "0x43700000", "Fp8 Max value 240 as float32" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Min), "0xc3700000", "Fp8 Min value -240 as float32" ))
    elif self.kernel["ProblemType"]["DestDataType"].isBFloat8() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
      module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBF8NanInf), "0x207", "Nan and +/- inf" ))
      module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBF8Max), "0x47600000", "BF8 Max value 57344 as float32" ))
      module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBF8Min), "0xc7600000", "BF8 Min value -57344 as float32" ))

    storeCode = Module("GroupLoadStore")
    vmcntTotalIssued = self.loadsBetaIssued + self.loadsEIssued
    lgkmcntTotalIssued = self.localLoadsBiasIssued + self.loadsScaleAVecIssued + self.loadsScaleBVecIssued + self.loadsScaleAlphaVecIssued
    waitCnter = [vmcntTotalIssued, lgkmcntTotalIssued]
    for elementIdx in range(0, len(self.batchElements)):
      element = self.batchElements[elementIdx]
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      dataE = self.ss.elementDataE[elementIdx]
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleAVec = self.ss.elementDataScaleAVec[elementIdx]
      dataScaleBVec = self.ss.elementDataScaleBVec[elementIdx]
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      vc0 = element[3]
      sumIdx = self.ss.elementSumIdx[elementIdx]

      # print(str(element)+" rowInc="+str(addrCalc.rowInc))
      # Already write wave column block into LDS
      # Now read lds data back to registers and write to global memroy
      if self.kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel":
        if self.ss.optSrdIncForRow and addrCalc.rowInc and self.kernel["StoreRemapVectorWidth"] > 0:
          module.addComment1("StoreRemap: shift coord1 address")
          if self.kernel["ProblemType"]["UseE"] and (self.kernel["GlobalSplitU"] == 1):
            printExit("Use E does not support StoreRemapVectorWidth if GSU == 1.")
            # module.add(addrCalc.incrementToNextRow(self.kernel, "E", self.ss, self.tmpS01, isCompute=True))
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
      if interleaveStoreVmcnt:
        waitLocalLoadCnt = 0
        waitLocalLoadCntStrList = []
        waitLoadCnt = 0
        waitLoadCntStrList = []
        # Calculate global loads
        if self.beta:
          waitLoadCnt += self.betaLoadIssued[elementIdx]
          waitLoadCntStrList.append("%d (beta)"%self.betaLoadIssued[elementIdx])
        if self.loadE:
          waitLoadCnt += self.eLoadIssued[elementIdx]
          waitLoadCntStrList.append("%d (load E)"%self.eLoadIssued[elementIdx])
        # Calculate local loads
        if self.parentWriter.states.useBias == DataDirection.READ:
          waitLocalLoadCnt += self.biasLoadIssued[elementIdx]
          waitLocalLoadCntStrList.append("%d (bias)"%self.biasLoadIssued[elementIdx])
        if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
          waitLocalLoadCnt += self.scaleAVecLoadIssued[elementIdx]
          waitLocalLoadCntStrList.append("%d (scaleAVec)"%self.scaleAVecLoadIssued[elementIdx])
          waitLocalLoadCnt += self.scaleBVecLoadIssued[elementIdx]
          waitLocalLoadCntStrList.append("%d (scaleBVec)"%self.scaleBVecLoadIssued[elementIdx])
        if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
          waitLocalLoadCnt += self.scaleAlphaVecLoadIssued[elementIdx]
          waitLocalLoadCntStrList.append("%d (scaleAlphaVec)"%self.scaleAlphaVecLoadIssued[elementIdx])
        # Get vmcnt and lgkmcnt
        vmcnt = vmcntTotalIssued - waitLoadCnt
        if waitCnter[0] > 0  or vmcnt != waitCnter[0] : # Check if global load issued > 0
          if waitCnter[0] == vmcnt: # No need to wait if the global load cnt doesn't change
            vmcnt = -1
          else:
            waitCnter[0] = vmcnt
        else:
          vmcnt = -1

        lgkmcnt = lgkmcntTotalIssued - waitLocalLoadCnt
        if waitCnter[1] > 0 or lgkmcnt != waitCnter[1]: # Check if local load issued > 0
          if waitCnter[1] == lgkmcnt: # No need to wait if the local load cnt doesn't change
            lgkmcnt = -1
          else:
            waitCnter[1] = lgkmcnt
        else:
          lgkmcnt = -1
        # Get vscnt
        if vmcnt != -1:
          if self.parentWriter.states.archCaps["SeparateVscnt"]:
            vscnt = 0
          else:
            vscnt = self.storesIssued if not self.kernel["GroupLoadStore"] else 0
        else:
          vscnt = -1
        if (vmcnt != -1) or (vscnt != -1) or (lgkmcnt != -1):
          # Get comment
          comment = ""
          if vmcnt != -1:
            tmp = ""
            for cntStr in waitLoadCntStrList:
              tmp += " - %s"%cntStr
            comment = "vmcnt(%s) = %d%s"%(vmcnt, vmcntTotalIssued, tmp)
          if lgkmcnt != -1:
            tmp = ""
            for cntStr in waitLocalLoadCntStrList:
              tmp += " - %s"%cntStr
            comment = comment + (" " if comment else "") + "lgkmcnt(%d) = %d%s"%(lgkmcnt, lgkmcntTotalIssued, tmp)
          module.addSpaceLine()
          if not self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
            module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="%s (interleaved)"%comment))

      def applyScaleVec(vecModule, addressStr, dataScaleVec, factorDim, isGlobal=True):
        if not self.beta and not self.applyAlpha: # case for beta-0 and alpha == 1,(OptNLL)
          if (self.kernel["ProblemType"]["DestDataType"].isInt8() or self.kernel["ProblemType"]["DestDataType"].isInt32() or \
              (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isHalf()) or \
              (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isBFloat16())) and \
            self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_I32_to_F32, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu))

        if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          maskConst = 1.0
        elif self.kernel["ProblemType"]["ComputeDataType"].isInt32():
          maskConst = 1

        gwvw = 1 if factorDim else self.gwvw
        if isGlobal:
          vecModule.add(VCmpGtU32(dst=sgpr("Address%s"%addressStr, self.parentWriter.states.laneSGPRCount), src0=sgpr("Srd%s+2"%addressStr), src1=0, comment=" == 0 ?"))
          for vi2 in range(0, gwvw):
            vecModule.add(VCndMaskB32(
              dst=vgpr(dataScaleVec + vi2), \
              src1=vgpr(dataScaleVec + vi2), \
              src0=maskConst, \
              src2=sgpr("Address%s"%addressStr, self.parentWriter.states.laneSGPRCount), \
              comment="1. mul 1 if 0"))
        if factorDim and self.gwvw > 1:
          vecModule.add(VMovB32(dst=vgpr(dataScaleVec+1), src=vgpr(dataScaleVec), comment="copy data%s to data%s+1"%(addressStr, addressStr)))

        for vi in range(0, self.gwvw):
          inputScaleVecVgpr = dataScaleVec + (0 if factorDim else vi)
          sumIdxV   = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              vecModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= %sVMul"%addressStr ))
            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              vecModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleVecVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= %sVMulPK(%d)(%d)"%(addressStr, dataScaleVec,vi)))
          elif self.kernel["ProblemType"]["ComputeDataType"].isInt32():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single i32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              vecModule.add(VMulLOU32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= %sVMul"%addressStr ))
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              vecModule.add(VMulLOU32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= %sVMulPK(%d)(%d)"%(addressStr, dataScaleAlphaVec,vi)))
              vecModule.add(VMulLOU32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src0=vgpr(inputScaleVecVgpr+1), src1=vgpr("ValuC+%d"%(vgprIdx+1)), comment="*= %sVMulPK(%d)(%d)"%(addressStr, dataScaleAlphaVec,vi)))
          else:
            raise RuntimeError("Unsupported %s compute data type %s."%(addressStr, str(self.kernel["ProblemType"]["ComputeDataType"])))

      scaleAVecModule = Module("ScaleAVecModule")
      scaleBVecModule = Module("ScaleBVecModule")
      if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        applyScaleVec(scaleAVecModule, "ScaleA", dataScaleAVec, 0, isGlobal=False)
        applyScaleVec(scaleBVecModule, "ScaleB", dataScaleBVec, 1, isGlobal=False)
      module.add(scaleAVecModule)
      module.add(scaleBVecModule)

      scaleAlphaVecModule = Module("scaleAlphaVecModule")
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        applyScaleVec(scaleAlphaVecModule, "ScaleAlphaVec", dataScaleAlphaVec, self.factorDim, isGlobal=False)
      module.add(scaleAlphaVecModule)

      if self.beta:
        module.add(self._addSumAlphaWithCBeta(self.kernel, self.ss, self.gwvw, elementIdx, vc0, self.tmpVgpr, self.cvtVgprStruct))
      elif ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"]) and not self.applyAlpha \
        and not ( self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel"))): # case of alpha=1 and beta=0
        if (self.kernel["ProblemType"]["DestDataType"].isInt8() or self.kernel["ProblemType"]["DestDataType"].isInt32() or \
            (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isHalf()) or \
            (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isBFloat16())) and \
           self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          module.add(convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_I32_to_F32, \
                                      inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu))

      # Add bias
      mergeActFuncCall = False
      if self.parentWriter.states.useBias == DataDirection.READ:
        if activationCDataType == self.kernel["ProblemType"]["ComputeDataType"] and self.kernel["ActivationFuncCall"]:
          mergeActFuncCall = True
        if (self.kernel["ProblemType"]["Gradient"] and self.kernel["ProblemType"]["ActivationType"] != 'none' and self.kernel["ProblemType"]["UseE"]) and (self.kernel["GlobalSplitU"] == 1):
          mergeActFuncCall = False

        if self.factorDim and self.gwvw > 1:
          module.add(VMovB32(dst=vgpr(dataBias+1), src=vgpr(dataBias), comment="copy dataBias to dataBIas+1"))

        for vi in range(0, self.gwvw):
          inputVgpr = dataBias + + (0 if self.factorDim else vi)
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

      if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
        vgprIdx   = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
        vgprDst   = self.activationSetPCStruct.vgprActCopy if mergeActFuncCall else vgprIdx
        prefixStr = "" if mergeActFuncCall else "ValuC+"
        prefixOffset = 0 if mergeActFuncCall else self.parentWriter.states.c.startVgprValu
        # Packdata if needed
        tmpVgpr = self.tmpVgpr
        if mergeActFuncCall:
          tmpVgpr += self.gwvw * self.kernel["ProblemType"]["ComputeDataType"].numRegisters()
        if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          if self.kernel["ProblemType"]["DataTypeE"].isHalf():
            packdata = PackData_F16()
            module.add(packdata(self.gwvw, tmpVgpr, vgprDst, tmpVgpr=tmpVgpr, inputPrefix=prefixStr, prefixOffset=prefixOffset))
            vgprDst = tmpVgpr
          elif self.kernel["ProblemType"]["DataTypeE"].isBFloat16():
            packdata = PackData_BF16()
            module.add(packdata(self.gwvw, tmpVgpr, vgprDst, self.cvtVgprStruct, self.tmpS01, self.laneSGPRC,
                                tmpVgpr=tmpVgpr, inputPrefix=prefixStr, prefixOffset=prefixOffset))
            vgprDst = tmpVgpr
          elif self.kernel["ProblemType"]["DataTypeE"].isSingle():
            if not mergeActFuncCall:
              vgprDst = "ValuC+%d" % vgprDst
          else:
            printExit("Unsupport type for E output. (%s)"%self.kernel["ProblemType"]["DataTypeE"].toEnum())
        else:
          printExit("Unsupport compute type for E output. (%s)"%self.kernel["ProblemType"]["ComputeDataType"].toEnum())

        module.add(self.parentWriter.addStore(self.kernel, self.ss, 'E', addrCalc, vgprDst, self.tmpS01, self.edge, comment="store E"))

      SaturateTypeInt8 = SaturateCastType.NORMAL

      gradientCvtModule = Module("gradientCvtModule")
      if (self.kernel["ProblemType"]["UseE"] and self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
        loadOffset = int((self.kernel["ProblemType"]["ComputeDataType"].numRegisters() - self.kernel["ProblemType"]["DataTypeE"].numRegisters()) * self.ss.cfg.gwvw)
        if activationCDataType != self.kernel["ProblemType"]["DataTypeE"]:
          if activationCDataType.isSingle() and self.kernel["ProblemType"]["DataTypeE"].isHalf():
            for vi in range(0, self.gwvw):
              dataEV  = dataE + vi
              dataEV2 = dataE + vi // 2
              selectbit = SelectBit.WORD_0 if (self.gwvw != 1 and vi % 2 == 0) or (self.gwvw == 1 and elementIdx % 2 == 0) else SelectBit.WORD_1
              gradientCvtModule.add(VCvtF16toF32(dst=vgpr(dataEV), src=vgpr(dataEV2+loadOffset), sdwa=SDWAModifiers(src0_sel=selectbit), comment="gwvw %d, elementIdx %d"%(self.gwvw, elementIdx)))
          elif activationCDataType.isSingle() and self.kernel["ProblemType"]["DataTypeE"].isBFloat16():
            for vi in range(0, self.gwvw):
              dataEV  = dataE + vi
              dataEV2 = dataE + vi // 2
              selectWord = 0 if (self.gwvw != 1 and vi % 2 == 0) or (self.gwvw == 1 and elementIdx % 2 == 0) else 1
              module.add(VCvtBF16toFP32(dst=(dataEV), src=(dataEV2+loadOffset), vgprMask=(self.cvtVgprStruct.vgprBf16Mask), vi=(selectWord), additionalCmts="gwvw %d, elementIdx %d"%(self.gwvw, elementIdx)))
          else:
            printExit("[Gradient input] Unsupported conversion.")

      # Activation
      activationModule = None
      isActivationInsertAfter = False
      if self.kernel["ProblemType"]["Gradient"] and (self.kernel["GlobalSplitU"] == 1):
        gradientInput = dataE
        enableValuC   = False
      else:
        gradientInput = self.ss.elementSumIdx[elementIdx]
        enableValuC   = True
        if self.kernel["LocalSplitU"] > 1:
          # When LSU > 1, the VGPRs are from LSU output.
          # the elementSumIdx has indicated the VGPRs from LSU.
          # Don't use the ValuC prefix here.
          enableValuC = False
      if self.kernel["ActivationFuncCall"]:
        if (activationCDataType == self.kernel["ProblemType"]["DestDataType"]) and \
          (activationCDataType != self.kernel["ProblemType"]["ComputeDataType"]) and ((self.kernel["ProblemType"]["UseScaleCD"] == False) or (self.kernel["ProblemType"]["UseScaleAlphaVec"] == False)):
          isActivationInsertAfter = True
        activationModule = Module("ActivationFuncCall")
        if (not mergeActFuncCall) and (not isActivationInsertAfter):
          activationModule.appendModule (copyData(activationCDataType, gradientInput, self.gwvw, \
            self.activationSetPCStruct.vgprActCopy))
        activationModule.add(SSwapPCB64(dst=sgpr(self.activationSetPCStruct.sgprOffsetBack, 2), \
          src=sgpr(self.activationSetPCStruct.sgprOffsetActivation, 2)))
        activationModule.appendModule (copyData(activationCDataType, gradientInput, self.gwvw, \
          self.activationSetPCStruct.vgprActCopy, 1))
      elif self.parentWriter.insertActivationAfterPacked(self.kernel, self.activationTypeStr) and (self.kernel["ProblemType"]["UseScaleAlphaVec"] == False):
        isActivationInsertAfter = True
        activationModule = self.parentWriter.getActivationDestDataType(self.kernel, self.activation, \
          self.activationTypeStr, self.gwvw, gradientInput , gradientInput, self.tmpVgpr, self.tmpSgpr)
      else:
        satInt8 = False
        if self.kernel["ProblemType"]["DestDataType"].isInt8():
          if (self.activationTypeStr == 'abs') or (self.activationTypeStr == 'relu'):
            SaturateTypeInt8 = SaturateCastType.DO_NOTHING
            satInt8 = True
        activationModule = self.parentWriter.getActivationActivationComputeType(self.kernel, self.activation, \
          self.activationTypeStr, self.gwvw, gradientInput, gradientInput, self.tmpVgpr, self.tmpSgpr, satInt8, enableValuC)
      # Add C *= GradientAct
      if self.kernel["ProblemType"]["ActivationType"] != 'none' and self.kernel["ProblemType"]["Gradient"] and (self.kernel["GlobalSplitU"] == 1):
        if isActivationInsertAfter:
          assert 0, "Gradient does not support isActivationInsertAfter."
        for vi in range(0, self.gwvw):
          sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
          dataEV  = dataE + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              activationModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr("ValuC+%d"%vgprIdx), src1=vgpr(dataEV), comment="C *= GradAct"))
            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              activationModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr("ValuC+%d"%vgprIdx, 2), src1=vgpr(dataEV, 2), comment="C *= GradAct"))
          else:
            assert 0, "Unsupported gradient type"

      scaleDModule = Module("Empty scaleDModule")
      if self.kernel["ProblemType"]["UseScaleCD"] and (self.kernel["GlobalSplitU"] == 1):
        for vi in range(0, self.gwvw):
          sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              if self.kernel["ProblemType"]["OutputAmaxD"]:
                if self.edge:
                  activationModule.add(VCmpEQU32(dst="vcc", src0="BufferOOB", src1=(vgpr(addrCalc.addrDVgpr)), comment =""))
                  activationModule.add(VCndMaskB32(dst=vgpr("AmaxOutB"), src0=vgpr("ValuC+%d"%vgprIdx), src1=hex(0), src2="vcc", comment="Check If OOB, put zero if OOB"))
                  activationModule.add(VMaxF32(dst=vgpr("AmaxOut"), src0=vgpr("AmaxOut"), src1=SrcAbs(vgpr("AmaxOutB")), comment="absmax"))
                else:
                  activationModule.add(VMaxF32(dst=vgpr("AmaxOut"), src0=vgpr("AmaxOut"), src1=SrcAbs(vgpr("ValuC+%d"%vgprIdx)), comment="absmax"))
              activationModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr("ValuC+%d"%vgprIdx), src1=sgpr("ScaleD"), comment="result *= ScaleD"))
            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              activationModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr("ValuC+%d"%vgprIdx, 2), src1=sgpr("ScaleD", 2), comment="result *= ScaleD"))
          else:
            assert 0, "Unsupported scaleD type"


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
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf16CVTVgprStruct=self.cvtVgprStruct,
                                     tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isFloat8():
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], fp8CVTVgprStruct=self.cvtVgprStruct, \
                                     tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isBFloat8():
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf8CVTVgprStruct=self.cvtVgprStruct, \
                                     tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt32():
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt8():
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, roundType=RoundType.ROUND_TO_NEAREST_EVEN, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], self.cvtVgprStruct, self.tmpS01,
                                     SaturateTypeInt8=SaturateTypeInt8, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)

      if self.parentWriter.states.asmCaps["HasWMMA_V1"] and self.kernel["EnableMatrixInstruction"] and self.kernel["ProblemType"]["DestDataType"].isHalf() and (not self.kernel["ProblemType"]["HighPrecisionAccumulate"]):
        for vi in range(0, self.gwvw):
          sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
          if vi%2 == 1:
            formatVgpr = formatting(sumIdxV, "ValuC+", self.parentWriter.states.c.startVgprValu)
            d = self.ss.elementSumIdx[elementIdx] + vi//2
            dVgpr = formatting(d, "ValuC+", self.parentWriter.states.c.startVgprValu)
            packModule.add(VPackF16toB32(dst=vgpr(dVgpr), src0=vgpr(formatting(sumIdxV-1, "ValuC+", self.parentWriter.states.c.startVgprValu)), src1=vgpr(formatVgpr), \
                          comment="Pack with neighbor"))

      biasReductionModule = Module("biasReductionModule")
      if self.storeBiasD == 1:
        vgprIdx = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
        biasReductionModule.add(self.parentWriter.addStore(self.kernel, self.ss, 'Bias', addrCalc, "ValuC+%d"%vgprIdx, self.tmpS01, self.edge, comment="store Bias"))

      if isActivationInsertAfter:
        module.add(convertModule)
        module.add(packModule)
        module.add(gradientCvtModule)
        module.add(activationModule)
      else:
        module.add(gradientCvtModule)
        module.add(activationModule)
        module.add(scaleDModule)
        module.add(biasReductionModule)
        module.add(convertModule)
        module.add(packModule)

      if not self.kernel["StoreRemapVectorWidth"]:
        if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":#GSUGSU
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'TD', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store TD not StoreRemapVectorWidth")
        else:
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'D', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store D")
        if self.kernel["GroupLoadStore"]:
          storeCode.add(tmpStoreCode)
        else:
          module.add(tmpStoreCode)

        self.storesIssued += 1
        if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
          self.storesIssued += 1
        if self.storeBiasD == 1:
          self.storesIssued += 1

      else:
        if not self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":#GSUGSU
          rpe = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr
          module.add(self.parentWriter.storeRemapAddLocalWrite(self.kernel, self.ss, addrCalc, sumIdx*rpe))
          # Column Block Shape has been written to LDS
          # Now read back and write out to global memory
        else:
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'TD', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store TD StoreRemapVectorWidth")

          if self.kernel["GroupLoadStore"]:
            storeCode.add(tmpStoreCode)

          module.add(tmpStoreCode)

          self.storesIssued += 1
          if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
            self.storesIssued += 1
          if self.storeBiasD == 1:
            self.storesIssued += 1

    module.add(storeCode)

    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":#GSUGSU
      SynchronizerEndlabelString = "Sync_EDN%s%s" % ("_Beta" if self.beta else "", "_Edge" if self.edge else "" )
      SynchronizerEndlabelComment = "Sync_EDN"
      SynchronizerEndlabel = Label(self.parentWriter.labels.getName(SynchronizerEndlabelString), SynchronizerEndlabelComment)
      module.add(SynchronizerEndlabel)

      module.addSpaceLine()
      module.addselfAsm("//synchronizer store end\n")

      module.addSpaceLine()

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
                          MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True,),
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

  def _addSumAlphaWithCBeta(self, kernel, ss, gwvw, elementIdx, vc0, tmpVgpr, cvtVgprStruct):
    module = Module("addSumAlphaWithCBeta #elementIdx%u, vc0 %u"%(elementIdx, vc0))
    for vi in range(0, gwvw):
      dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
      sumIdxV = ss.elementSumIdx[elementIdx] + vi
      if kernel["ProblemType"]["DestDataType"].isHalf():
        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          if self.parentWriter.states.asmCaps["HasWMMA_V1"] and kernel["EnableMatrixInstruction"]:
            dataV = ss.elementData[elementIdx] + int(vi / 2 * ss.cfg.numVgprsPerDataPerVI)
            if (vi % 2) == 0:
              module.add(VMulPKF16(dst=vgpr(dataV), src0=sgpr("Beta"), src1=vgpr(dataV+0), \
                    comment="%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi)))
            else:
              module.add(VLShiftRightB32(dst=vgpr(dataV), shiftHex=16, src=vgpr(dataV), \
                    comment="shift 16bit to get next half of packed ValueC"))
            # dataV+0 = new c = old c*beta + rC
            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(sumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(sumIdxV)), \
                comment="sum*alpha + C*beta"))
          elif sumIdxV%2==0 or (not ss.cfg.halfDataRegPerVI and gwvw==1):
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
          module.add(VCvtBF16toFP32(dst=(tmpVgpr), src=(dataCExternal), vgprMask=(cvtVgprStruct.vgprBf16Mask), vi=(vi)))
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
            module.add(VMovB32(dst=vgpr(tmpVgpr+1), src=hex(vi * 8), comment="value = %u"%(vi * 8)))
            module.add(VBfeI32(dst=vgpr(tmpVgpr), src0=vgpr(dataV+0), src1=vgpr(tmpVgpr+1), src2=8, comment="int8 to int32"))
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

      # float8 precision
      elif kernel["ProblemType"]["DestDataType"].isFloat8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          # Generate single f32 code if edge is detected.
          isPK = False
          if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
            if self.parentWriter.states.archCaps["NoSDWA"]:
              sb = 0 if self.gwvw == 1 else 1
              module.add(VCvtFP8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataV), vop3=VOP3PModifiers(op_sel=[0,sb])))
            else:
              sb = SelectBit.BYTE_0 if self.gwvw == 1 else SelectBit.BYTE_2
              module.add(VCvtFP8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
          # Original packed route
          elif vi%2 == 1:
            continue
          else:
            isPK = True
            if self.parentWriter.states.archCaps["NoSDWA"]:
              sb = 0 if vi ==0 else 1
              module.add(VCvtPkFP8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataV), vop3=VOP3PModifiers(op_sel=[sb])))
            else:
              sb = SelectBit.WORD_0 if vi == 0 else SelectBit.WORD_1
              module.add(VCvtPkFP8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
          module.add(SNop(waitState=0))
          if kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta"))
            if isPK:
              module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(tmpVgpr+1), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta (PK)"))
      # bfloat8 precision
      elif kernel["ProblemType"]["DestDataType"].isBFloat8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          # Generate single f32 code if edge is detected.
          isPK = False
          if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
            if self.parentWriter.states.archCaps["NoSDWA"]:
              sb = 0 if self.gwvw == 1 else 1
              module.add(VCvtBF8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataV), vop3=VOP3PModifiers(op_sel=[0,sb])))
            else:
              sb = SelectBit.BYTE_0 if self.gwvw == 1 else SelectBit.BYTE_2
              module.add(VCvtBF8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
          # Original packed route
          elif vi%2 == 1:
            continue
          else:
            isPK = True
            if self.parentWriter.states.archCaps["NoSDWA"]:
              sb = 0 if vi ==0 else 1
              module.add(VCvtPkBF8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataV), vop3=VOP3PModifiers(op_sel=[sb])))
            else:
              sb = SelectBit.WORD_0 if vi == 0 else SelectBit.WORD_1
              module.add(VCvtPkBF8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
          module.add(SNop(waitState=0))
          if kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta"))
            if isPK:
              module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(tmpVgpr+1), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta (PK)"))
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
      tmp = i.srcs[0]
      i.srcs[0] = i.dst
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
