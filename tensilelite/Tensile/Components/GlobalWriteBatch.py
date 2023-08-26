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
from ..Utils import DataDirection
from ..TensileInstructions import Label, Module, EXEC, SDWAModifiers, VCC, SelectBit, \
                            vgpr, sgpr, replaceHolder, SaturateCastType, VCvtBF16toFP32, \
                            DataType, CvtType, RoundType
from ..TensileInstructions.Instructions import *
from ..AsmAddressCalculation import AddrCalculation
from ..Components.PackData import formatting

from math import ceil
from ..TensileInstructions import log2

class GlobalWriteBatchComponent(GlobalWriteComponents):
  kernel = {"ProblemType": {"OperationType": "GEMM" }}
  def __call__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrE, addrD, addrC, addrBias, addrScaleDVec, addrScaleAlphaVec, biasLocalBarrierInit: bool, \
    tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
    packdata, label2, parentWriter) -> Module:
    return GlobalWriteBatchWriter(kernel, tPA, tPB, activation, ss, batchIdx, applyAlpha, \
      beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrE, addrD, addrC, addrBias, addrScaleDVec, addrScaleAlphaVec, biasLocalBarrierInit, \
      tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      packdata, label2, parentWriter).emit()

class GlobalWriteBatchWriter:
  def __init__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrE, addrD, addrC, addrBias, addrScaleDVec, addrScaleAlphaVec, biasLocalBarrierInit: bool, \
    tmpVgpr, bf16CVTVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeMulAlpha, \
      packdata, label2, parentWriter):
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
    self.addrScaleDVec = addrScaleDVec
    self.addrScaleAlphaVec = addrScaleAlphaVec
    self.biasLocalBarrierInit  = biasLocalBarrierInit
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
    self.storesIssued = 0
    self.label2 = label2

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

  def GSUSYNC0(self, GSU, MT0, MT1, labelname, labelendname):
    module = Module("GSUSYNC0")
    #module.addComment1("Magic div and 2mod functions")

    # macro = Macro("GSUSYNC0", "labelname", "labelendname")
    if MT1>MT0:
        WaveNum = "MT1/MT0"
    else:
        WaveNum = "MT0/MT1"

    if self.kernel["ProblemType"]["UseScaleDVec"]:
      offset = "0x8C"
    else:
      offset = "0x60"
    contents = \
    "\n\
s_waitcnt vmcnt(0)\n\
//check done\n\
"+str(labelname)+":\n\
s_mov_b32 s[sgprGSUSync], 0\n\
s_atomic_add s[sgprGSUSync], s[sgprKernArgAddress:sgprKernArgAddress+1], "+offset+" glc\n\
\n"
    module.addGSUSYNC(contents)

    return module

  def GSUSYNCcodegen(self, GSU, MT0, MT1, StoreVectorWidth, labelname, labelendname, vgprstart, vgprstart2, vgproffset):
    module = Module("GSUSYNC")

    WaveNum = str(MT1*MT0)

    GSUxWaveNum = str(hex(self.kernel["GlobalSplitU"]*MT1*MT0-1))

    # module.addGSUSYNC("/*\n")

    module.add(SWaitCnt(waitAll=True, comment=""))
    module.add(SCmpEQU32(
        src0=sgpr("GSUSync"), \
        src1=hex(int(WaveNum)), \
        comment=""))
    module.add(SCBranchSCC0(labelName=labelname, comment=""))
    module.addComment("check done start")
    # print("GSUSYNCcodegen checkout")
    module.addComment("synchronizer check")
    module.add(SMovB32(dst=sgpr("GSUSync"), src=hex(GSU-1), comment=""))
    # module.add(SNop(8))
    tmpS01 = self.parentWriter.sgprPool.checkOut(1, preventOverflow=False) #overflow?
    module.add(SMulI32(dst=sgpr(tmpS01), src0=sgpr("WorkGroup1"), src1=sgpr("NumWorkGroups0"), comment=""))
    # module.add(SNop(8))
    module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr("WorkGroup0")))
    # module.add(SNop(8))
    tmpV01 = self.parentWriter.vgprPool.checkOut(1)
    module.add(VLShiftRightB32(dst=vgpr(tmpV01), shiftHex=hex(log2(self.kernel["WavefrontSize"])), src=vgpr("Serial")))
    module.add(SNop(0))
    tmpS02 = self.parentWriter.sgprPool.checkOut(1, preventOverflow=False) #overflow?
    module.add(VReadfirstlaneB32(dst=sgpr(tmpS02), src=vgpr(tmpV01)))
    # module.add(SNop(8))
    tmpS03 = self.parentWriter.sgprPool.checkOut(1, preventOverflow=False) #overflow?
    module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr("NumWorkGroups0"), src1=sgpr("NumWorkGroups1"), comment=""))
    # module.add(SNop(8))
    module.add(SMulI32(dst=sgpr(tmpS03), src0=sgpr(tmpS03), src1=sgpr(tmpS02), comment=""))
    # module.add(SNop(8))
    module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=sgpr(tmpS03)))
    beptmp = 2
    # module.add(SNop(8))
    module.add(SLShiftLeftB32(dst=sgpr(tmpS01), src=sgpr(tmpS01), shiftHex=hex(beptmp), comment=""))
    # module.add(SNop(8))
    module.add(SAddU32(dst=sgpr("SrdSync+0"), \
                                    src0=sgpr("SrdSync+0"), \
                                    src1=sgpr(tmpS01), \
                                    comment="" ))
    # module.add(SNop(8))
    module.add(SAddCU32(dst=sgpr("SrdSync+1"), \
                        src0=sgpr("SrdSync+1"), \
                        src1=hex(0), \
                        comment="" ))
    # module.add(SNop(8))  
    module.addGSUSYNC("s_buffer_atomic_dec s[sgprGSUSync], s[sgprSrdSync:sgprSrdSync+3], glc\n")
    
    module.add(SSubU32(dst=sgpr(tmpS01), src0=sgpr("SizesFree+1"), src1=hex(1)))
    tmpS04 = self.parentWriter.sgprPool.checkOutAligned(2,2, preventOverflow=False) #overflow?
    module.add(SMulHIU32(dst=sgpr(tmpS04+1), src0=sgpr(tmpS01), src1=sgpr("StrideC1J"), comment=""))
    module.add(SMulI32(dst=sgpr(tmpS04+0), src0=sgpr(tmpS01), src1=sgpr("StrideC1J"), comment=""))

    tmpS05 = self.parentWriter.sgprPool.checkOutAligned(2,2, preventOverflow=False) #overflow?
    module.add(SAddU32(dst=sgpr(tmpS05+0), \
                                    src0=sgpr("SizesFree+0"), \
                                    src1=sgpr(tmpS04+0), \
                                    comment="" ))
    module.add(SAddCU32(dst=sgpr(tmpS05+1), \
                        src0=hex(0), \
                        src1=sgpr(tmpS04+1), \
                        comment="" ))

    module.add(SSubU32(dst=sgpr(tmpS01), src0=sgpr("SizesFree+2"), src1=hex(1)))
    module.add(SMulHIU32(dst=sgpr(tmpS04+1), src0=sgpr(tmpS01), src1=sgpr("StrideCK"), comment=""))
    module.add(SMulI32(dst=sgpr(tmpS04+0), src0=sgpr(tmpS01), src1=sgpr("StrideCK"), comment=""))

    module.add(SAddU32(dst=sgpr(tmpS05+0), \
                                    src0=sgpr(tmpS05+0), \
                                    src1=sgpr(tmpS04+0), \
                                    comment="" ))
    module.add(SAddCU32(dst=sgpr(tmpS05+1), \
                        src0=sgpr(tmpS05+1), \
                        src1=sgpr(tmpS04+1), \
                        comment="" ))

    bpe = self.parentWriter.states.bpeCinternal
    module.add(SLShiftLeftB64(dst=sgpr("tmp2E",2), src=sgpr(tmpS05,2), shiftHex=log2(bpe), comment="scale by bpe"))

    module.add(SSubU32(dst=sgpr(tmpS02), src0=sgpr("GSUSumIdx"), src1=hex(0), comment=""))
    module.add(SMulI32(dst=sgpr(tmpS05+1), src0=sgpr(tmpS02), src1=sgpr("tmp3E"), comment=""))
    module.add(SMulI32(dst=sgpr(tmpS05+0), src0=sgpr(tmpS02), src1=sgpr("tmp2E"), comment=""))
    module.add(SMulHIU32(dst=sgpr(tmpS01), src0=sgpr(tmpS02), src1=sgpr("tmp2E"), comment=""))
    module.add(SAddU32(dst=sgpr(tmpS05+1), \
                                    src0=sgpr(tmpS05+1), \
                                    src1=sgpr(tmpS01), \
                                    comment="" ))

    module.add(SSubU32(dst=sgpr("SrdD+0"), \
                                    src0=sgpr("SrdD+0"), \
                                    src1=sgpr(tmpS05+0), \
                                    comment="" ))
    module.add(SSubBU32(dst=sgpr("SrdD+1"), \
                        src0=sgpr("SrdD+1"), \
                        src1=sgpr(tmpS05+1), \
                        comment="" ))

    module.addSpaceLine()


    module.add(SWaitCnt(waitAll=True, comment=""))
    module.add(SCmpEQU32(
        src0=sgpr("GSUSync"), \
        src1=hex(1), \
        comment=""))
    module.add(SCBranchSCC0(labelName=labelendname, comment=""))
    module.addComment("check done end")

    module.addSpaceLine()

    SyncloadedData = 0
    module.addGSUSYNC("buffer_load_dwordx4 v["+str(vgprstart)+"+4*0:"+str(vgprstart)+"+3+4*0], "+str(vgproffset)+", s[sgprSrdD:sgprSrdD+3], 0 offen offset:0, sc0 sc1 // load GSU D\n")
    SyncloadedData += 1

    for i in range(1,GSU):
      module.add(SAddU32(dst=sgpr("SrdD+0"), \
                                      src0=sgpr("SrdD+0"), \
                                      src1=sgpr("tmp2E+0"), \
                                      comment="" ))
      module.add(SAddCU32(dst=sgpr("SrdD+1"), \
                          src0=sgpr("SrdD+1"), \
                          src1=sgpr("tmp2E+1"), \
                          comment="" ))
      module.addGSUSYNC("buffer_load_dwordx4 v["+str(vgprstart)+"+4*"+str(i)+":"+str(vgprstart)+"+3+4*"+str(i)+"], "+str(vgproffset)+", s[sgprSrdD:sgprSrdD+3], 0 offen offset:0, sc0 sc1 // load GSU D\n")
      SyncloadedData += 1
      # # if GWVW=1 the half path still assumes we have
      # # at least two stores so does some combining across VI -
      # # for example assuming we can have two elements and can use pk_mul
      # # here:
    vscnt = 0
    lgkmcnt = -1
    
    # module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(Victor yes)"))
    vmcnt = SyncloadedData = SyncloadedData -1

    # module.addGSUSYNC("/*\n")

    for i in range(1, self.kernel["GlobalSplitU"]):
      module.addSpaceLine()
      vmcnt = SyncloadedData = SyncloadedData -1
      module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="(Victor yes)"))
      module.add(VAddPKF32(dst=vgpr(vgprstart, 2), src0=vgpr(vgprstart, 2), \
                                     src1=vgpr(vgprstart+0+4*i, 2), comment="C += bias"))
      module.add(VAddPKF32(dst=vgpr(vgprstart+2, 2), src0=vgpr(vgprstart+2, 2), \
                                     src1=vgpr(vgprstart+2+4*i, 2), comment="C += bias"))

      # module.addGSUSYNC("*/\n")

    # module.add(SSubU32(dst=sgpr("SrdD+0"), src0=sgpr("SrdD+0"), src1=prePad, comment="pre-pad to make room for possible pointer shift"))
    # module.add(SSubBU32(dst=sgpr("SrdD+1"), src0=sgpr("SrdD+1"), src1=0, comment="pre-pad to make room for possible pointer shift"))
    # s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 4  // pre-pad to make room for possible pointer shift
    # s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
    # print("GSUSYNCcodegen checkin")
    self.parentWriter.sgprPool.checkIn(tmpS05)
    self.parentWriter.sgprPool.checkIn(tmpS04)
    self.parentWriter.sgprPool.checkIn(tmpS03)
    self.parentWriter.sgprPool.checkIn(tmpS02)
    self.parentWriter.vgprPool.checkIn(tmpV01)
    self.parentWriter.sgprPool.checkIn(tmpS01)

    if StoreVectorWidth==2:
        contents = \
    "\n\
v_mov_b32 v["+str(vgprstart)+"2+0], v["+str(vgprstart)+"+0]\n\
v_mov_b32 v["+str(vgprstart)+"2+1], v["+str(vgprstart)+"+1]\n\
v_mov_b32 v["+str(vgprstart)+"+0], v["+str(vgprstart)+"+2]\n\
v_mov_b32 v["+str(vgprstart)+"+1], v["+str(vgprstart)+"+3]\n"
        # module.addGSUSYNC(contents)

    # module.addGSUSYNC("*/\n")

    return module

  def GSUSYNC(self, GSU, MT0, MT1, StoreVectorWidth, labelname, labelendname, vgprstart, vgprstart2, vgproffset):
    module = Module("GSUSYNC")
    #module.addComment1("Magic div and 2mod functions")
    # if StoreVectorWidth==2:
    #     macro = Macro("GSUSYNC", "labelname", "labelendname", "vgprstart", "vgprstart2", "vgproffset")
    # else:
    #     macro = Macro("GSUSYNC", "labelname", "labelendname", "vgprstart", "vgproffset")
    # macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))

    # if MT1>MT0:
    #     WaveNum = "MT1/MT0"
    # else:
    #     WaveNum = "MT0/MT1"

    WaveNum = str(MT1*MT0)

    GSUxWaveNum = str(hex(self.kernel["GlobalSplitU"]*MT1*MT0-1))

    contents = \
    "\n\
s_waitcnt lgkmcnt(0)\n\
s_cmp_eq_u32 s[sgprGSUSync], "+WaveNum+"    //\n\
s_cbranch_scc0 "+str(labelname)+"           // jump if XX required\n\
//check done\n\
\n\
//synchronizer check\n\
s_mov_b32 s[sgprGSUSync] "+hex(GSU-1)+"\n\
\n\
//s_mov_b32 s[sgprtmp0E], s[sgprGSUSumIdx]                          //cal synchronizer position\n\
s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup1], s[sgprNumWorkGroups0]\n\
s_add_u32 s[sgprtmp0E], s[sgprtmp0E], s[sgprWorkGroup0]\n\
\n\
v_lshrrev_b32 v0, 6, v[vgprSerial]\n\
v_readfirstlane_b32 s[sgprtmp1E], v0      // set back to numWorkGroup0\n\
s_mul_i32 s[sgprtmp2E], s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]\n\
s_mul_i32 s[sgprtmp2E], s[sgprtmp2E], s[sgprtmp1E]\n\
s_add_u32 s[sgprtmp0E], s[sgprtmp0E], s[sgprtmp2E]\n\
s_lshl_b32 s[sgprtmp0E], s[sgprtmp0E], 2\n\
\n\
s_mul_hi_u32 s[sgprtmp3E], s[sgprStrideDK], "+str(GSU)+"                   // Scale by Stride\n\
s_mul_i32 s[sgprtmp2E], s[sgprStrideDK], "+str(GSU)+"                      // Scale by Stride\n\
s_lshl_b64 s[sgprtmp2E:sgprtmp2E+1], s[sgprtmp2E:sgprtmp2E+1], 2  // scale by bpe\n\
\n\
s_mov_b32 s[sgprSrdDd+2], 0x80000000\n\
s_mov_b32 s[sgprSrdDd+3], Srd127_96                               //\n\
\n\
s_add_u32 s[sgprSrdDd+0], s[sgprAddressD+0], s[sgprtmp2E]         // add lo to SRD\n\
s_addc_u32 s[sgprSrdDd+1], s[sgprAddressD+1], s[sgprtmp3E]        // add hi to SRD\n\
\n\
s_add_u32 s[sgprSrdDd+0], s[sgprSrdDd+0], s[sgprtmp0E]            // add lo to SRD\n\
s_addc_u32 s[sgprSrdDd+1], s[sgprSrdDd+1], 0                      // add hi to SRD\n\
s_buffer_atomic_dec s[sgprGSUSync], s[sgprSrdDd:sgprSrdDd+3], glc\n\
\n\
\n\
//s_mov_b32 s[sgprGSUSumIdx] 1\n\
s_mul_i32 s[sgprtmp4E], MT1, s[sgprWorkGroup1]                        //\n\
s_mul_hi_u32 s[sgprtmp1E], s[sgprtmp4E], s[sgprStrideD1J]             // cal GSU D position\n\
s_mul_i32 s[sgprtmp0E], s[sgprtmp4E], s[sgprStrideD1J]                //\n\
s_lshl_b64 s[sgprtmp0E:sgprtmp1E], s[sgprtmp0E:sgprtmp1E], 2          // scale by bpe\n\
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s[sgprtmp0E]              // add lo to SRD\n\
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s[sgprtmp1E]             // add hi to SRD\n\
\n\
s_mul_hi_u32 s[sgprtmp1E], s[sgprWorkGroup2], s[sgprStrideDK]         //\n\
s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup2], s[sgprStrideDK]            //\n\
s_lshl_b64 s[sgprtmp0E:sgprtmp1E], s[sgprtmp0E:sgprtmp1E], 2          // scale by bpe\n\
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp0E]                  // add lo to SRD\n\
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp1E]                 // add hi to SRD\n\
\n\
s_waitcnt lgkmcnt(0)\n\
s_cmp_eq_u32 s[sgprGSUSync], 0x1                // s[sgprGSUSync] == GSU*WaveNum-1 ?\n\
s_cbranch_scc0 "+str(labelendname)+" //label_GW_End_1 //label_AFTERsummary_Edge\n\
//synchronizer check\n\
\n\
//synchronizer\n\
\n\
buffer_load_dwordx4 v["+str(vgprstart)+"+4*0:"+str(vgprstart)+"+3+4*0], "+str(vgproffset)+", s[sgprSrdD:sgprSrdD+3], 0 offen offset:0, sc0 sc1 // load GSU D\n\
\n\
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s\n\
//s_mul_hi_u32 s[sgprtmp1E], s[sgprSizesFree+0], s[sgprGSUSumIdx]   //\n\
//s_mul_i32 s[sgprtmp0E], s[sgprSizesFree+0], s[sgprGSUSumIdx]      //\n\
s_sub_u32 s[sgprtmp5E], s[sgprSizesFree+1], 1                       // cal GSU D offset\n\
//s_mul_i32 s[sgprtmp5E], s[sgprtmp5E], s[sgprGSUSumIdx]            //\n\
s_mul_hi_u32 s[sgprtmp3E], s[sgprtmp5E], s[sgprStrideC1J]           //\n\
s_mul_i32 s[sgprtmp2E], s[sgprtmp5E], s[sgprStrideC1J]              //\n\
s_add_u32 s[sgprtmp0E], s[sgprSizesFree+0], s[sgprtmp2E]            //\n\
s_addc_u32 s[sgprtmp1E], 0, s[sgprtmp3E]                            //\n\
s_sub_u32 s[sgprtmp5E], s[sgprSizesFree+2], 1                       //\n\
//s_mul_i32 s[sgprtmp5E], s[sgprtmp5E], s[sgprGSUSumIdx]            //\n\
s_mul_hi_u32 s[sgprtmp3E], s[sgprtmp5E], s[sgprStrideCK]            //\n\
s_mul_i32 s[sgprtmp2E], s[sgprtmp5E], s[sgprStrideCK]               //\n\
s_add_u32 s[sgprtmp0E], s[sgprtmp0E], s[sgprtmp2E]                  //\n\
s_addc_u32 s[sgprtmp1E], s[sgprtmp1E], s[sgprtmp3E]                 //\n\
s_lshl_b64 s[sgprtmp2E:sgprtmp3E], s[sgprtmp0E:sgprtmp1E], 2        // scale by bpe\n\
\n"
    module.addGSUSYNC(contents)

    for i in range(1,GSU):
        # print(i)
        contents = \
        "\n\
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp2E]        // add lo synchronizer offset to SRD\n\
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp3E]       // add hi synchronizer offset to SRD\n\
buffer_load_dwordx4 v["+str(vgprstart)+"+4*"+str(i)+":"+str(vgprstart)+"+3+4*"+str(i)+"], "+str(vgproffset)+", s[sgprSrdD:sgprSrdD+3], 0 offen offset:0, sc0 sc1 // load GSU D\n"
        module.addGSUSYNC(contents)

    contents = \
"\n\
s_waitcnt lgkmcnt(0)\n\
\n\
s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
s_cbranch_scc0 "+str(labelendname)+" //label_GW_End_1 //label_AFTERsummary_Edge\n"
    # module.addGSUSYNC(contents)

    for i in range(1,GSU):
        # print(i)
        contents = \
        "\n\
s_waitcnt vmcnt("+str(GSU-1-i)+")\n\
V_PK_ADD_F32 v["+str(vgprstart)+"+0:"+str(vgprstart)+"+1], v["+str(vgprstart)+"+0:"+str(vgprstart)+"+1], v["+str(vgprstart)+"+4*"+str(i)+"+0:"+str(vgprstart)+"+4*"+str(i)+"+1]\n\
V_PK_ADD_F32 v["+str(vgprstart)+"+2:"+str(vgprstart)+"+3], v["+str(vgprstart)+"+2:"+str(vgprstart)+"+3], v["+str(vgprstart)+"+4*"+str(i)+"+2:"+str(vgprstart)+"+4*"+str(i)+"+3]\n"
        module.addGSUSYNC(contents)

    if StoreVectorWidth==2:
        contents = \
    "\n\
v_mov_b32 v["+str(vgprstart)+"2+0], v["+str(vgprstart)+"+0]\n\
v_mov_b32 v["+str(vgprstart)+"2+1], v["+str(vgprstart)+"+1]\n\
v_mov_b32 v["+str(vgprstart)+"+0], v["+str(vgprstart)+"+2]\n\
v_mov_b32 v["+str(vgprstart)+"+1], v["+str(vgprstart)+"+3]\n"
        module.addGSUSYNC(contents)

    contents = \
"\n\
s_waitcnt lgkmcnt(0)\n\
\n\
s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
s_cbranch_scc0 "+str(labelendname)+" //label_GW_End_1 //label_AFTERsummary_Edge\n"
    # macro.addGSUSYNC(contents)

    contents = "//synchronizer\n"
    module.addGSUSYNC(contents)

    # module.add(macro)
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

    self.ss.setupStoreElementsForBatch(self.kernel, self.gwvw, self.batchElements, self.batchElementSgprs, isOptNLL=False)

    self.localLoadsBiasIssued = 0
    self.storesIssued    = 0
    self.loadsBetaIssued   = 0
    self.loadsEIssued      = 0
    self.loadsScaleDVecIssued = 0
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
    self.scaleDVecLoadIssued = []
    self.scaleAlphaVecLoadIssued = []
    loadedDataBeta = {}
    loadedDataE = {}
    loadedDataBias = {}
    loadedDataScaleDVec = {}
    loadedDataScaleAlphaVec = {}

    if self.kernel["BufferStore"] and self.edge:
      bufferOOB = self.parentWriter.vgprPool.checkOut(1, "BufferOOB")
      module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))
    else:
      bufferOOB = None

    for elementIdx, element in enumerate(self.batchElements):
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addrCVgpr    = addrCalc.addrCVgpr
      addrDVgpr    = addrCalc.addrDVgpr
      addrEVgpr    = addrCalc.addrEVgpr
      addrBiasVgpr = addrCalc.addrBiasVgpr
      addrScaleDVecVgpr = addrCalc.addrScaleDVecVgpr
      addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
      data     = self.ss.elementData[elementIdx]
      dataBeta = self.ss.elementData[elementIdx]
      dataE    = self.ss.elementDataE[elementIdx]
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleDVec = self.ss.elementDataScaleDVec[elementIdx]
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      mask     = self.ss.elementMask[elementIdx]
      vc0 = element[3]
      sumIdxGSUSYNC = self.ss.elementSumIdx[elementIdx]

      module.add(addrCalc.emitAddressSetupCode(self.kernel, self.tPB, self.ss, self.tmpVgpr, self.tmpS01, self.edge, self.beta, self.atomic, elementIdx, addrDVgpr))

      if self.edge:
        module.add(addrCalc.edgeProtectCode(self.kernel, self.edge, self.beta, self.atomic, mask, self.tmpSgpr))

      # create code Module to push mov vgpr,acc instructions
      if self.beta:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'C', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrCVgpr, self.addrC))
        if dataBeta not in loadedDataBeta:
          if self.kernel["GroupLoadStore"]:
            loadInputCode.add(self.parentWriter.readInput(self.kernel, self.ss, 'C', self.kernel["ProblemType"]["DestDataType"], addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
          else:
            module.add(self.parentWriter.readInput(self.kernel, self.ss, 'C', self.kernel["ProblemType"]["DestDataType"], addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01))
          loadedDataBeta[dataBeta] = ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsBetaIssued += ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.gwvw / 16)
      self.betaLoadIssued.append(len(loadedDataBeta) * ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.ss.cfg.gwvw / 16))

      if (self.kernel["ProblemType"]["UseE"] and self.kernel["ProblemType"]["Gradient"] and self.kernel["ProblemType"]["ActivationType"] != 'none') and (self.kernel["GlobalSplitU"] == 1):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'E', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrEVgpr, self.addrE))
        if dataE not in loadedDataE:
          if self.kernel["GroupLoadStore"]:
            loadInputCode.add(self.parentWriter.readInput(self.kernel, self.ss, 'E', self.kernel["ProblemType"]["ComputeDataType"], addrCalc, vc0, dataE, self.gwvw, addrEVgpr, self.tmpS01))
          else:
            module.add(self.parentWriter.readInput(self.kernel, self.ss, 'E', self.kernel["ProblemType"]["ComputeDataType"], addrCalc, vc0, dataE, self.gwvw, addrEVgpr, self.tmpS01))
          loadedDataE[dataE] = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsEIssued += ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.gwvw / 16)
        self.loadE = True
      else:
        self.loadE = False
      self.eLoadIssued.append(len(loadedDataE) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16))

      if self.parentWriter.states.useBias == DataDirection.READ:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'Bias', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrBiasVgpr, self.addrBias))
        if dataBias not in loadedDataBias:
          if self.kernel["GroupLoadStore"]:
            # Group bias load with C input to
            if ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")) and (not self.biasLocalBarrierInit):
              loadInputCode.add(SWaitCnt(lgkmcnt=0, comment="Wait for Bias LDS write"))
              loadInputCode.add(SBarrier("Bias LDS write barrier"))
              self.biasLocalBarrierInit = True
            loadInputCode.add(self.parentWriter.addBiasLoad(self.kernel["ProblemType"]["ComputeDataType"], self.kernel, self.ss, addrCalc, dataBias, True))
          else:
            if ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")) and (not self.biasLocalBarrierInit):
              module.add(SWaitCnt(lgkmcnt=0, comment="Wait for Bias LDS write"))
              module.add(SBarrier("Bias LDS write barrier"))
              self.biasLocalBarrierInit = True
            module.add(self.parentWriter.addBiasLoad(self.kernel["ProblemType"]["ComputeDataType"], self.kernel, self.ss, addrCalc, dataBias, True))
          loadedDataBias[dataBias] = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16)
          self.localLoadsBiasIssued += ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16)
      self.biasLoadIssued.append(len(loadedDataBias) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16))

      if self.kernel["ProblemType"]["UseScaleDVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'ScaleDVec', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrScaleDVecVgpr, self.addrScaleDVec))
        if dataScaleDVec not in loadedDataScaleDVec:
          # Shift right several vgprs for cvt ops if needed
          numVgprs = int(ceil(self.kernel["ProblemType"]["ComputeDataType"].numRegisters() * self.ss.cfg.gwvw))
          reg = self.kernel["ProblemType"]["ComputeDataType"].numRegisters() if self.kernel["ProblemType"]["ComputeDataType"].numRegisters() >= 1 else 1
          gprShiftScaleDVec = dataScaleDVec + (self.ss.cfg.gwvw * reg - numVgprs)
          if self.kernel["GroupLoadStore"]:
            # Group scaleDVec load with C input to
            loadInputCode.add(self.parentWriter.addScaleDVecLoad(self.kernel, self.ss, addrCalc, gprShiftScaleDVec))
          else:
            module.add(self.parentWriter.addScaleDVecLoad(self.kernel, self.ss, addrCalc, gprShiftScaleDVec))
          loadedDataScaleDVec[dataScaleDVec] = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsScaleDVecIssued += ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16)
      self.scaleDVecLoadIssued.append(len(loadedDataScaleDVec) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16))
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'ScaleAlphaVec', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, self.tmpSgpr, addrScaleAlphaVecVgpr, self.addrScaleAlphaVec))
        if dataScaleAlphaVec not in loadedDataScaleAlphaVec:
          # Shift right several vgprs for cvt ops if needed
          numVgprs = int(ceil(self.kernel["ProblemType"]["ComputeDataType"].numRegisters() * self.ss.cfg.gwvw))
          reg = self.kernel["ProblemType"]["ComputeDataType"].numRegisters() if self.kernel["ProblemType"]["ComputeDataType"].numRegisters() >= 1 else 1
          gprShiftScaleAlphaVec = dataScaleAlphaVec + (self.ss.cfg.gwvw * reg - numVgprs)
          if self.kernel["GroupLoadStore"]:
            # Group scaleAlphaVec load with C input to
            loadInputCode.add(self.parentWriter.addScaleAlphaVecLoad(self.kernel, self.ss, addrCalc, gprShiftScaleAlphaVec))
          else:
            module.add(self.parentWriter.addScaleAlphaVecLoad(self.kernel, self.ss, addrCalc, gprShiftScaleAlphaVec))
          loadedDataScaleAlphaVec[dataScaleAlphaVec] = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsScaleAlphaVecIssued += ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16)
      self.scaleAlphaVecLoadIssued.append(len(loadedDataScaleAlphaVec) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16))

      if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'E', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrEVgpr, self.addrE))
      if self.storeBiasD == 1:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'Bias', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrBiasVgpr, self.addrBias))
      module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'D', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, self.tmpSgpr, addrDVgpr, self.addrD))

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

    if self.kernel["BufferStore"] and self.edge:
      self.parentWriter.vgprPool.checkIn(bufferOOB)

    module.add(loadInputCode)

    if self.beta and self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt(module)

    ########################################
    # AccVgpr read
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

    ########################################################
    # interleaveStoreVmcnt = self.parentWriter.states.interleaveStoreVmcnt and not self.edge

    # for elementIdx in range(len(self.batchElements)):
    #   for vi in range(self.gwvw):
    #     sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
    #     newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
    #     # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
    #     if self.kernel["ProblemType"]["ComputeDataType"].isInt32() or \
    #         self.kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
    #         if self.debugConfig["ForceExpectedValue"]:
    #           module.add(VMovB32(vgpr("ValuC+%u"%newSumIdxV), self.debugConfig["ValueCExpectedValue"], "force expected value" ))
    #         if self.parentWriter.db["ForceVSerial"]:
    #           module.add(VMovB32(vgpr("ValuC+%u"%newSumIdxV), vgpr("Serial"), "force expected value to serial" ))
    #         if self.parentWriter.db["CheckValueC"]:
    #           module.add(SMovB32(sgpr(self.tmpS01), self.debugConfig["ValueCExpectedValue"], "Move expected value"))
    #           module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%newSumIdxV), sgpr(self.tmpS01)))

    # ########################################
    # # wait for batched load
    # # Here we wait all
    # if not interleaveStoreVmcnt:
    #   vmcnt = -1
    #   lgkmcnt = -1
    #   commentList = []
    #   # Global read wait
    #   if self.beta:
    #     vmcnt = 0
    #     commentList.append("Beta")
    #   if self.loadE:
    #     vmcnt = 0
    #     commentList.append("E")
    #   # if self.kernel["ProblemType"]["UseScaleDVec"] and (self.kernel["GlobalSplitU"] == 1):
    #   if self.kernel["ProblemType"]["UseScaleDVec"]:
    #     vmcnt = 0
    #     commentList.append("ScaleDVec")
    #   # Local read wait
    #   if self.parentWriter.states.useBias == DataDirection.READ:
    #     lgkmcnt = 0
    #     commentList.append("Bias LDS")
    #   if (vmcnt != -1) or (lgkmcnt != -1):
    #     # Get comment
    #     comment = "wait for " + commentList[0]
    #     for c in commentList[1:]:
    #       comment += ", %s"%c
    #     module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=-1, comment=comment))

      storeCodeGSUGSU = Module("GroupLoadStore")
      waitCnterGSUGSU = [self.loadsBetaIssued + self.loadsEIssued + self.loadsScaleDVecIssued + self.loadsScaleAlphaVecIssued, self.localLoadsBiasIssued]
      for elementIdx in range(0, len(self.batchElements)):
        element = self.batchElements[elementIdx]
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        addr = addrCalc.addrDVgpr
        dataE = self.ss.elementDataE[elementIdx]
        dataBias = self.ss.elementDataBias[elementIdx]
        dataScaleDVec = self.ss.elementDataScaleDVec[elementIdx]
        dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
        mask = self.ss.elementMask[elementIdx]
        vc0 = element[3]
        sumIdx = self.ss.elementSumIdx[elementIdx]

        # print(str(element)+" rowInc="+str(addrCalc.rowInc))
        # Already write wave column block into LDS
        # Now read lds data back to registers and write to global memroy
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

      # # if GWVW=1 the half path still assumes we have
      # # at least two stores so does some combining across VI -
      # # for example assuming we can have two elements and can use pk_mul
      # # here:
      # if interleaveStoreVmcnt:
      #   waitLocalLoadCnt = 0
      #   waitLocalLoadCntStrList = []
      #   waitLoadCnt = 0
      #   waitLoadCntStrList = []
      #   # Calculate global loads
      #   if self.beta:
      #     betaCnt = elementIdx + 1
      #     waitLoadCnt += betaCnt
      #     waitLoadCntStrList.append("%d (beta)"%betaCnt)
      #   if self.loadE:
      #     loadECnt = elementIdx + 1
      #     waitLoadCnt += loadECnt
      #     waitLoadCntStrList.append("%d (load E)"%loadECnt)
      #   # if self.kernel["ProblemType"]["UseScaleDVec"] and (self.kernel["GlobalSplitU"] == 1):
      #   if self.kernel["ProblemType"]["UseScaleDVec"]:
      #     waitLoadCnt += self.scaleDVecLoadIssued[elementIdx]
      #     waitLoadCntStrList.append("%d (scaleDVec)"%self.scaleDVecLoadIssued[elementIdx])
      #   # Calculate local loads
      #   if self.parentWriter.states.useBias == DataDirection.READ:
      #     waitLocalLoadCnt += self.biasLoadIssued[elementIdx]
      #     waitLocalLoadCntStrList.append("%d (bias)"%self.biasLoadIssued[elementIdx])
      #   # Get vmcnt and lgkmcnt
      #   if waitCnterGSUGSU[0] > 0: # Check if global load issued > 0
      #     vmcnt = self.loadsIssued + self.loadsScaleDVecIssued - waitLoadCnt
      #     if waitCnterGSUGSU[0] == vmcnt: # No need to wait if the global load cnt doesn't change
      #       vmcnt = -1
      #     waitCnterGSUGSU[0] = vmcnt
      #   else:
      #     vmcnt = -1
      #   if waitCnterGSUGSU[1] > 0: # Check if local load issued > 0
      #     lgkmcnt = self.localLoadIssued - waitLocalLoadCnt
      #     if waitCnterGSUGSU[1] == lgkmcnt: # No need to wait if the local load cnt doesn't change
      #       lgkmcnt = -1
      #     waitCnterGSUGSU[1] = lgkmcnt
      #   else:
      #     lgkmcnt = -1
      #   # Get vscnt
      #   if vmcnt != -1:
      #     if self.parentWriter.states.archCaps["SeparateVscnt"]:
      #       vscnt = 0
      #     else:
      #       vscnt = self.storesIssued if not self.kernel["GroupLoadStore"] else 0
      #   else:
      #     vscnt = -1
      #   if (vmcnt != -1) or (vscnt != -1) or (lgkmcnt != -1):
      #     # Get comment
      #     comment = ""
      #     if vmcnt != -1:
      #       tmp = ""
      #       for cntStr in waitLoadCntStrList:
      #         tmp += " - %s"%cntStr
      #       comment = "vmcnt(%s) = %d%s"%(vmcnt, self.loadsIssued + self.loadsScaleDVecIssued, tmp)
      #     if lgkmcnt != -1:
      #       tmp = ""
      #       for cntStr in waitLocalLoadCntStrList:
      #         tmp += " - %s"%cntStr
      #       comment = comment + (" " if comment else "") + "lgkmcnt(%d) = %d%s"%(lgkmcnt, self.localLoadIssued, tmp)
      #     module.addSpaceLine()
      #     if 0:
      #       module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="%s (interleaved)"%comment))

      if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
        vgprIdx = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
        vgprDst = self.activationSetPCStruct.vgprActCopy if mergeActFuncCall else "ValuC+%d"%vgprIdx
        module.add(self.parentWriter.addStore(self.kernel, self.ss, 'E', addrCalc, vgprDst, self.tmpS01, self.edge, comment="store E"))

      if not self.kernel["StoreRemapVectorWidth"]:
        tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'D', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store D")
        if self.kernel["GroupLoadStore"]:
          storeCodeGSUGSU.add(tmpStoreCode)
        else:
          module.addGSUSYNC("\n") #GSUSYNC
          module.add(tmpStoreCode)
        self.storesIssued += 1
        if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
          self.storesIssued += 1
        if self.storeBiasD == 1:
          self.storesIssued += 1

      else:
        rpe = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr
        module.add(self.parentWriter.storeRemapAddLocalWrite(self.ss, addrCalc, sumIdx*rpe))
        # Column Block Shape has been written to LDS
        # Now read back and write out to global memory

    module.add(storeCodeGSUGSU)

    if self.parentWriter.states.serializedStore and 1:
      module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))
    ########################################################

    # module.addComment("GSUSYNC label_BUSYWAIT1 label_Activation_End 24 v9") #GSUSYNC

    # labelname = "label_BUSYWAIT%s%s_" \
    #     % ("_Beta" if self.beta else "", "_Edge" if self.edge else "")

    # labelname += self.activationTypeStr
    # labelname = self.parentWriter.labels.getNameInc(labelname)
    module.addGSUSYNC("\n") #GSUSYNC
    labelname = self.parentWriter.labels.getNameInc(self.label2)
    # labelname = self.label2
    if 0: #self.kernel["StoreVectorWidth"]==2:
      module.add(self.GSUSYNC0(self.kernel["GlobalSplitU"], self.kernel["MIWaveGroup"][0], self.kernel["MIWaveGroup"][1], self.kernel["StoreVectorWidth"], labelname, "label_KernelEnd"))
      # module.add(MacroInstruction("GSUSYNC0", args=[labelname, "label_KernelEnd"]))
      # module.add(MacroInstruction("GSUSYNC1", args=[labelname, "label_KernelEnd"]))
      module.add(self.GSUSYNCcodegen(self.kernel["GlobalSplitU"], self.kernel["MacroTile0"], self.kernel["MacroTile1"], labelname, "label_KernelEnd", sumIdxGSUSYNC, self.ss.elementSumIdx[elementIdx-1], vgpr(self.ss.elementAddr[elementIdx-1].addrDVgpr)))
      # module.add(self.GSUSYNC(self.kernel["GlobalSplitU"], self.kernel["MacroTile0"], self.kernel["MacroTile1"], labelname, "label_KernelEnd", sumIdxGSUSYNC, self.ss.elementSumIdx[elementIdx-1], vgpr(self.ss.elementAddr[elementIdx-1].addrDVgpr)))
      # module.add(MacroInstruction("GSUSYNC", args=[labelname, "label_KernelEnd", sumIdxGSUSYNC, self.ss.elementSumIdx[elementIdx-1], vgpr(self.ss.elementAddr[elementIdx-1].addrDVgpr)]))
    else:
      if (self.kernel["GlobalSplitU"] != 1 and self.kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel'):
        module.add(self.GSUSYNC0(self.kernel["GlobalSplitU"], self.kernel["MacroTile0"], self.kernel["MacroTile1"], labelname, "label_KernelEnd"))
        # module.add(MacroInstruction("GSUSYNC0", args=[labelname, "label_KernelEnd"]))
        # module.add(MacroInstruction("GSUSYNC1", args=[labelname, "label_KernelEnd"]))
        module.add(self.GSUSYNCcodegen(self.kernel["GlobalSplitU"], self.kernel["MIWaveGroup"][0], self.kernel["MIWaveGroup"][1], self.kernel["StoreVectorWidth"], labelname, "label_KernelEnd", sumIdxGSUSYNC, 0, vgpr(addrCalc.addrDVgpr)))
        # module.add(self.GSUSYNC(self.kernel["GlobalSplitU"], self.kernel["MIWaveGroup"][0], self.kernel["MIWaveGroup"][1], self.kernel["StoreVectorWidth"], labelname, "label_KernelEnd", sumIdxGSUSYNC, 0, vgpr(addrCalc.addrDVgpr)))
        # module.add(MacroInstruction("GSUSYNC", args=[labelname, "label_KernelEnd", sumIdxGSUSYNC, vgpr(addrCalc.addrDVgpr)]))
    # module.addGSUSYNC("\n") #GSUSYNC

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
    checkedDataScaleDVec = {}
    checkedDataScaleAlphaVec = {}
    for elementIdx in range(len(self.batchElements)):
      if not self.ss.sharedColDVgprs:
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        addrEVgpr    = addrCalc.addrEVgpr
        addrDVgpr    = addrCalc.addrDVgpr
        addrCVgpr    = addrCalc.addrCVgpr
        addrBiasVgpr = addrCalc.addrBiasVgpr
        addrScaleDVecVgpr = addrCalc.addrScaleDVecVgpr
        addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
        if addrEVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrEVgpr)
        self.parentWriter.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.parentWriter.vgprPool.checkIn(addrCVgpr)
        if addrBiasVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrBiasVgpr)
        if addrScaleDVecVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleDVecVgpr)
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

      dataScaleDVec = self.ss.elementDataScaleDVec[elementIdx]
      if dataScaleDVec != 0:
        if dataScaleDVec not in checkedDataScaleDVec:
          self.parentWriter.vgprPool.checkIn(dataScaleDVec)
        checkedDataScaleDVec[dataScaleDVec] = 1
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      if dataScaleAlphaVec != 0:
        if dataScaleAlphaVec not in checkedDataScaleAlphaVec:
          self.parentWriter.vgprPool.checkIn(dataScaleAlphaVec)
        checkedDataScaleAlphaVec[dataScaleAlphaVec] = 1

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
      if 0:#GSUGSU
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

  def GSUSYNC2(self, StoreVectorWidth, issingle, vgprstart, vgprstart2, vgproffset):
    module = Module("GSUSYNC2")
    #module.addComment1("Magic div and 2mod functions")
    # if StoreVectorWidth==2:
    #     macro = Macro("GSUSYNC2", "vgprstart", "vgprstart2", "vgproffset")
    # else:
    #     macro = Macro("GSUSYNC2", "vgprstart", "vgproffset")
    # macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))

    contents = "//synchronizer store\n"
    module.addGSUSYNC(contents)

    if StoreVectorWidth==2:
        contents = \
    "\n\
v_mov_b32 v["+str(vgprstart)+"+2], v["+str(vgprstart)+"+0]\n\
v_mov_b32 v["+str(vgprstart)+"+3], v["+str(vgprstart)+"+1]\n\
v_mov_b32 v["+str(vgprstart)+"+0], v["+str(vgprstart)+"2+0]\n\
v_mov_b32 v["+str(vgprstart)+"+1], v["+str(vgprstart)+"2+1]\n"
        module.addGSUSYNC(contents)

    contents = ""
    if not issingle:
        contents = \
    "\n\
V_LSHRREV_B32 "+str(vgproffset)+", 0x1, "+str(vgproffset)+"\n\
    \n"
    module.addGSUSYNC(contents)

    if issingle:
        contents = \
    "\n\
s_mov_b32 s[sgprSrdD+2], 0x80000000\n\
s_mov_b32 s[sgprSrdD+3], Srd127_96                 //\n\
\n\
s_mul_i32 s[sgprtmp2E], MT1, s[sgprWorkGroup1]                    // cal store position\n\
s_mul_hi_u32 s[sgprtmp1E], s[sgprtmp2E], s[sgprStrideC1J]         //\n\
s_mul_i32 s[sgprtmp0E], s[sgprtmp2E], s[sgprStrideC1J]            //\n\
s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 2  // scale by bpe\n\
s_add_u32 s[sgprSrdD+0], s[sgprAddressTC+0], s[sgprtmp0E]         // add lo to SRD\n\
s_addc_u32 s[sgprSrdD+1], s[sgprAddressTC+1], s[sgprtmp1E]        // add hi to SRD\n\
\n\
s_mul_hi_u32 s[sgprtmp1E], s[sgprWorkGroup2], s[sgprStrideCK]     //\n\
s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup2], s[sgprStrideCK]        //\n\
s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 2  // scale by bpe\n\
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp0E]              // add lo to SRD\n\
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp1E]             // add hi to SRD\n"
    else:
        contents = \
    "\n\
s_mov_b32 s[sgprSrdD+2], 0x80000000\n\
s_mov_b32 s[sgprSrdD+3], Srd127_96                 //\n\
\n\
s_mul_i32 s[sgprtmp2E], MT1, s[sgprWorkGroup1]                    // cal store position\n\
s_mul_hi_u32 s[sgprtmp1E], s[sgprtmp2E], s[sgprStrideC1J]         //\n\
s_mul_i32 s[sgprtmp0E], s[sgprtmp2E], s[sgprStrideC1J]            //\n\
s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 1  // scale by bpe\n\
s_add_u32 s[sgprSrdD+0], s[sgprAddressTC+0], s[sgprtmp0E]         // add lo to SRD\n\
s_addc_u32 s[sgprSrdD+1], s[sgprAddressTC+1], s[sgprtmp1E]        // add hi to SRD\n\
\n\
s_mul_hi_u32 s[sgprtmp1E], s[sgprWorkGroup2], s[sgprStrideCK]     //\n\
s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup2], s[sgprStrideCK]        //\n\
s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 1  // scale by bpe\n\
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp0E]              // add lo to SRD\n\
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp1E]             // add hi to SRD\n"

    module.addGSUSYNC(contents)

    if issingle:
        contents = \
        "\n\
\n\
buffer_store_dwordx4 v["+str(vgprstart)+":"+str(vgprstart)+"+3], "+str(vgproffset)+", s[sgprSrdD:sgprSrdD+3], 0 offen offset:0, sc0 sc1// store D\n\
\n\
//synchronizer store\n"
    else:
        contents = \
        "\n\
# v_mov_b32 v["+str(vgprstart)+"+0], 1.0\n\
# v_mov_b32 v["+str(vgprstart)+"+1], 2.0\n\
# v_mov_b32 v["+str(vgprstart)+"+2], 3.0\n\
# v_mov_b32 v["+str(vgprstart)+"+3], 4.0\n\
v_cvt_f16_f32 v["+str(vgprstart)+"+0], v["+str(vgprstart)+"+0]\n\
v_cvt_f16_f32 v["+str(vgprstart)+"+1], v["+str(vgprstart)+"+1]\n\
v_cvt_f16_f32 v["+str(vgprstart)+"+2], v["+str(vgprstart)+"+2]\n\
v_cvt_f16_f32 v["+str(vgprstart)+"+3], v["+str(vgprstart)+"+3]\n\
\n\
v_pack_b32_f16 v["+str(vgprstart)+"+0], v["+str(vgprstart)+"+0], v["+str(vgprstart)+"+1]\n\
v_pack_b32_f16 v["+str(vgprstart)+"+1], v["+str(vgprstart)+"+2], v["+str(vgprstart)+"+3]\n\
\n\
buffer_store_dwordx2 v["+str(vgprstart)+":"+str(vgprstart)+"+1], "+str(vgproffset)+", s[sgprSrdD:sgprSrdD+3], 0 offen offset:0, sc0 sc1// store D\n\
\n\
//synchronizer store\n"
    module.addGSUSYNC(contents)

    return module

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
      if self.kernel["ProblemType"]["UseScaleDVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        vmcnt = 0
        commentList.append("ScaleDVec")
        # print("ScaleDVec vmcnt")
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        vmcnt = 0
        commentList.append("ScaleAlphaVec")
        # print("ScaleAlphaVec vmcnt")
      # Local read wait
      if self.parentWriter.states.useBias == DataDirection.READ:
        lgkmcnt = 0
        commentList.append("Bias LDS")
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
      module.add(VMovB32(vgpr(self.bf16CVTVgprStruct.vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" ))
      module.add(VMovB32(vgpr(self.bf16CVTVgprStruct.vgprFp32Nan), "0x7fff0000", "fp32 Nan" ))
      module.add(VMovB32(vgpr(self.bf16CVTVgprStruct.vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" ))

    storeCode = Module("GroupLoadStore")
    waitCnter = [self.loadsBetaIssued + self.loadsEIssued + self.loadsScaleDVecIssued + self.loadsScaleAlphaVecIssued, self.localLoadsBiasIssued]
    for elementIdx in range(0, len(self.batchElements)):
      element = self.batchElements[elementIdx]
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      dataE = self.ss.elementDataE[elementIdx]
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleDVec = self.ss.elementDataScaleDVec[elementIdx]
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      vc0 = element[3]
      sumIdx = self.ss.elementSumIdx[elementIdx]

      # print(str(element)+" rowInc="+str(addrCalc.rowInc))
      # Already write wave column block into LDS
      # Now read lds data back to registers and write to global memroy
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
        if self.kernel["ProblemType"]["UseScaleDVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
          waitLoadCnt += self.scaleDVecLoadIssued[elementIdx]
          waitLoadCntStrList.append("%d (scaleDVec)"%self.scaleDVecLoadIssued[elementIdx])
        if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
          waitLoadCnt += self.scaleAlphaVecLoadIssued[elementIdx]
          waitLoadCntStrList.append("%d (scaleAlphaVec)"%self.scaleAlphaVecLoadIssued[elementIdx])
        # Calculate local loads
        if self.parentWriter.states.useBias == DataDirection.READ:
          waitLocalLoadCnt += self.biasLoadIssued[elementIdx]
          waitLocalLoadCntStrList.append("%d (bias)"%self.biasLoadIssued[elementIdx])
        # Get vmcnt and lgkmcnt
        if waitCnter[0] > 0: # Check if global load issued > 0
          vmcnt = self.loadsBetaIssued + self.loadsEIssued + self.loadsScaleDVecIssued + self.loadsScaleAlphaVecIssued - waitLoadCnt
          if waitCnter[0] == vmcnt: # No need to wait if the global load cnt doesn't change
            vmcnt = -1
          waitCnter[0] = vmcnt
        else:
          vmcnt = -1
        if waitCnter[1] > 0: # Check if local load issued > 0
          lgkmcnt = self.localLoadsBiasIssued - waitLocalLoadCnt
          if waitCnter[1] == lgkmcnt: # No need to wait if the local load cnt doesn't change
            lgkmcnt = -1
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
            comment = "vmcnt(%s) = %d%s"%(vmcnt, self.loadsBetaIssued + self.loadsEIssued + self.loadsScaleDVecIssued + self.loadsScaleAlphaVecIssued, tmp)
          if lgkmcnt != -1:
            tmp = ""
            for cntStr in waitLocalLoadCntStrList:
              tmp += " - %s"%cntStr
            comment = comment + (" " if comment else "") + "lgkmcnt(%d) = %d%s"%(lgkmcnt, self.localLoadsBiasIssued, tmp)
          module.addSpaceLine()
          if 0:
            module.add(SWaitCnt(lgkmcnt=lgkmcnt, vmcnt=vmcnt, vscnt=vscnt, comment="%s (interleaved)"%comment))
        else:
          module.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment="%d(interleaved%dVictor)%d"%(lgkmcnt, vmcnt, vscnt)))

      scaleAlphaVecModule = Module("scaleAlphaVecModule")
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        for vi in range(0, self.gwvw):
          inputScaleAlphaVecVgpr = dataScaleAlphaVec + vi
          sumIdxV   = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu

            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):

              scaleAlphaVecModule.add(VCmpGtU32(dst=sgpr("AddressScaleAlphaVec",2), src0=sgpr("SrdScaleAlphaVec+2"), src1=0, comment=" == 0 ?"))
              scaleAlphaVecModule.add(VCndMaskB32(
                dst=vgpr(inputScaleAlphaVecVgpr), \
                src1=vgpr(inputScaleAlphaVecVgpr), \
                src0=1.0, \
                src2=sgpr("AddressScaleAlphaVec",2), \
                comment="1. mul 1 if 0"))

              if 0: #isActivationInsertAfter:
                if (self.kernel["ProblemType"]["DestDataType"].isHalf()):
                  scaleAlphaVecModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                if self.kernel["ProblemType"]["DestDataType"].isBFloat16():
                  scaleAlphaVecModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%vgprIdx), src=("ValuC+%d"%vgprIdx), vgprMask=None, vi=0))
                if self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isInt8():
                  scaleAlphaVecModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                scaleAlphaVecModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleAlphaVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= scaleAlphaVecVMul" ))
              else:
                scaleAlphaVecModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleAlphaVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= scaleAlphaVecVMul" ))

            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              scaleAlphaVecModule.add(VCmpGtU32(dst=sgpr("AddressScaleAlphaVec",2), src0=sgpr("SrdScaleAlphaVec+2"), src1=0, comment=" == 0 ?"))
              scaleAlphaVecModule.add(VCndMaskB32(
                dst=vgpr(inputScaleAlphaVecVgpr), \
                src1=vgpr(inputScaleAlphaVecVgpr), \
                src0=1.0, \
                src2=sgpr("AddressScaleAlphaVec",2), \
                comment="1. mul 1 if 0"))

              scaleAlphaVecModule.add(VCndMaskB32(
                dst=vgpr(inputScaleAlphaVecVgpr+1), \
                src1=vgpr(inputScaleAlphaVecVgpr+1), \
                src0=1.0, \
                src2=sgpr("AddressScaleAlphaVec",2), \
                comment="1. mul 1 if 0"))

              if 0: #isActivationInsertAfter:
                if self.kernel["ProblemType"]["DestDataType"].isHalf():
                  scaleAlphaVecModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                  scaleAlphaVecModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src=vgpr("ValuC+%d"%(vgprIdx+1))))
                if self.kernel["ProblemType"]["DestDataType"].isBFloat16():
                  scaleAlphaVecModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%vgprIdx), src=("ValuC+%d"%vgprIdx), vgprMask=None, vi=0))
                  scaleAlphaVecModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%(vgprIdx+1)), src=("ValuC+%d"%(vgprIdx+1)), vgprMask=None, vi=0))
                if self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isInt8():
                  scaleAlphaVecModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                  scaleAlphaVecModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src=vgpr("ValuC+%d"%(vgprIdx+1))))
                scaleAlphaVecModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleAlphaVecVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= scaleAlphaVecVMulPK(%d)(%d)"%(dataScaleAlphaVec,vi)))
              else:
                scaleAlphaVecModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleAlphaVecVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= scaleAlphaVecVMulPK(%d)(%d)"%(dataScaleAlphaVec,vi)))
          else:
            raise RuntimeError("Unsupported scaleAlphaVec compute data type %s."%str(self.kernel["ProblemType"]["ComputeDataType"]))

      module.add(scaleAlphaVecModule)

      if self.beta:
        module.add(self._addSumAlphaWithCBeta(self.kernel, self.ss, self.gwvw, elementIdx, vc0, self.tmpVgpr, self.bf16CVTVgprStruct))
      elif ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"]) and not self.applyAlpha: # case of alpha=1 and beta=0
        if (self.kernel["ProblemType"]["DestDataType"].isInt8() or self.kernel["ProblemType"]["DestDataType"].isInt32()) and self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          module.add(convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_I32_to_F32, \
                                      inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu))

      # Add bias
      mergeActFuncCall = False
      if self.parentWriter.states.useBias == DataDirection.READ:
        if activationCDataType == self.kernel["ProblemType"]["ComputeDataType"] and self.kernel["ActivationFuncCall"]:
          mergeActFuncCall = True
        if (self.kernel["ProblemType"]["Gradient"] and self.kernel["ProblemType"]["ActivationType"] != 'none' and self.kernel["ProblemType"]["UseE"]) and (self.kernel["GlobalSplitU"] == 1):
          mergeActFuncCall = False
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

      if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
        vgprIdx = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
        vgprDst = self.activationSetPCStruct.vgprActCopy if mergeActFuncCall else "ValuC+%d"%vgprIdx
        module.add(self.parentWriter.addStore(self.kernel, self.ss, 'E', addrCalc, vgprDst, self.tmpS01, self.edge, comment="store E"))

      SaturateTypeInt8 = SaturateCastType.NORMAL
      # Activation
      activationModule = None
      isActivationInsertAfter = False
      gradientInput = dataE if self.kernel["ProblemType"]["Gradient"] and (self.kernel["GlobalSplitU"] == 1) else self.ss.elementSumIdx[elementIdx]
      if self.kernel["ActivationFuncCall"]:
        if (activationCDataType == self.kernel["ProblemType"]["DestDataType"]) and \
          (activationCDataType != self.kernel["ProblemType"]["ComputeDataType"]) and ((self.kernel["ProblemType"]["UseScaleDVec"] == False) or (self.kernel["ProblemType"]["UseScaleAlphaVec"] == False)):
          isActivationInsertAfter = True
        activationModule = Module("ActivationFuncCall")
        if (not mergeActFuncCall) and (not isActivationInsertAfter):
          activationModule.appendModule (copyData(activationCDataType, gradientInput, self.gwvw, \
            self.activationSetPCStruct.vgprActCopy))
        activationModule.add(SSwapPCB64(dst=sgpr(self.activationSetPCStruct.sgprOffsetBack, 2), \
          src=sgpr(self.activationSetPCStruct.sgprOffsetActivation, 2)))
        activationModule.appendModule (copyData(activationCDataType, gradientInput, self.gwvw, \
          self.activationSetPCStruct.vgprActCopy, 1))
      elif self.parentWriter.insertActivationAfterPacked(self.kernel, self.activationTypeStr) and ((self.kernel["ProblemType"]["UseScaleDVec"] == False) or (self.kernel["ProblemType"]["UseScaleAlphaVec"] == False)):
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
          self.activationTypeStr, self.gwvw, gradientInput, gradientInput, self.tmpVgpr, self.tmpSgpr, satInt8)
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

      # pack stores, beta and non-beta reach here:
      packModule = Module("Empty pack module")
      convertModule = Module("Empty convert module")
      print("self.kernel[_GlobalAccumulation] != 'MultipleBuffer'", self.kernel["_GlobalAccumulation"] != 'MultipleBuffer')
      # if self.kernel["ProblemType"]["HighPrecisionAccumulate"] and (self.kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
      if self.kernel["ProblemType"]["HighPrecisionAccumulate"] and (self.kernel["_GlobalAccumulation"] != 'MultipleBuffer') and (self.kernel["_GlobalAccumulation"] != 'MultipleBufferSingleKernel'):
        if self.kernel["ActivationFuncCall"] and activationCDataType == self.kernel["ProblemType"]["DestDataType"]:
          destIdx = self.activationSetPCStruct.vgprActCopy
        else:
          destIdx = self.ss.elementSumIdx[elementIdx]
        if self.kernel["ProblemType"]["DestDataType"].isHalf():
          print("DestDataType isHalf", self.kernel["ProblemType"]["DestDataType"].isHalf())
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isBFloat16():
          print("DestDataType isBFloat16", self.kernel["ProblemType"]["DestDataType"].isBFloat16())
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf16CVTVgprStruct=self.bf16CVTVgprStruct,
                                     tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt32():
          print("DestDataType isInt32", self.kernel["ProblemType"]["DestDataType"].isInt32())
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt8():
          print("DestDataType isInt8", self.kernel["ProblemType"]["DestDataType"].isInt8())
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, roundType=RoundType.ROUND_TO_NEAREST_EVEN, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], self.tmpVgpr, self.tmpS01,
                                     SaturateTypeInt8=SaturateTypeInt8, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)

      if self.parentWriter.states.asmCaps["HasWMMA"] and self.kernel["EnableMatrixInstruction"] and self.kernel["ProblemType"]["DestDataType"].isHalf() and (not self.kernel["ProblemType"]["HighPrecisionAccumulate"]):
        for vi in range(0, self.gwvw):
          sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
          if vi%2 == 1:
            formatVgpr = formatting(sumIdxV, "ValuC+", self.parentWriter.states.c.startVgprValu)
            d = self.ss.elementSumIdx[elementIdx] + vi//2
            packModule.add(VPackF16toB32(dst=vgpr(d), src0=vgpr(formatting(sumIdxV-1, "ValuC+", self.parentWriter.states.c.startVgprValu)), src1=vgpr(formatVgpr), \
                          comment="Pack with neighbor"))

      scaleDVecModule = Module("scaleDVecModule")
      if self.kernel["ProblemType"]["UseScaleDVec"] and ((self.kernel["GlobalSplitU"] == 1) or (self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel")):
        for vi in range(0, self.gwvw):
          inputScaleDVecVgpr = dataScaleDVec + vi
          sumIdxV   = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu

            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):

              scaleDVecModule.add(VCmpGtU32(dst=sgpr("AddressScaleDVec",2), src0=sgpr("SrdScaleDVec+2"), src1=0, comment=" == 0 ?"))
              scaleDVecModule.add(VCndMaskB32(
                dst=vgpr(inputScaleDVecVgpr), \
                src1=vgpr(inputScaleDVecVgpr), \
                src0=1.0, \
                src2=sgpr("AddressScaleDVec",2), \
                comment="1. mul 1 if 0"))

              if isActivationInsertAfter:
                if (self.kernel["ProblemType"]["DestDataType"].isHalf()):
                  scaleDVecModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                if self.kernel["ProblemType"]["DestDataType"].isBFloat16():
                  scaleDVecModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%vgprIdx), src=("ValuC+%d"%vgprIdx), vgprMask=None, vi=0))
                if self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isInt8():
                  scaleDVecModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                scaleDVecModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleDVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= scaleDVecVMul" ))
              else:
                scaleDVecModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleDVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= scaleDVecVMul" ))

            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              scaleDVecModule.add(VCmpGtU32(dst=sgpr("AddressScaleDVec",2), src0=sgpr("SrdScaleDVec+2"), src1=0, comment=" == 0 ?"))
              scaleDVecModule.add(VCndMaskB32(
                dst=vgpr(inputScaleDVecVgpr), \
                src1=vgpr(inputScaleDVecVgpr), \
                src0=1.0, \
                src2=sgpr("AddressScaleDVec",2), \
                comment="1. mul 1 if 0"))

              scaleDVecModule.add(VCndMaskB32(
                dst=vgpr(inputScaleDVecVgpr+1), \
                src1=vgpr(inputScaleDVecVgpr+1), \
                src0=1.0, \
                src2=sgpr("AddressScaleDVec",2), \
                comment="1. mul 1 if 0"))

              if isActivationInsertAfter:
                if self.kernel["ProblemType"]["DestDataType"].isHalf():
                  scaleDVecModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                  scaleDVecModule.add(VCvtF16toF32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src=vgpr("ValuC+%d"%(vgprIdx+1))))
                if self.kernel["ProblemType"]["DestDataType"].isBFloat16():
                  scaleDVecModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%vgprIdx), src=("ValuC+%d"%vgprIdx), vgprMask=None, vi=0))
                  scaleDVecModule.add(VCvtBF16toFP32(dst=("ValuC+%d"%(vgprIdx+1)), src=("ValuC+%d"%(vgprIdx+1)), vgprMask=None, vi=0))
                if self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isInt8():
                  scaleDVecModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%vgprIdx), src=vgpr("ValuC+%d"%vgprIdx)))
                  scaleDVecModule.add(VCvtI32toF32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src=vgpr("ValuC+%d"%(vgprIdx+1))))
                scaleDVecModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleDVecVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= scaleDVecVMulPK(%d)(%d)"%(dataScaleDVec,vi)))
              else:
                scaleDVecModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleDVecVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= scaleDVecVMulPK(%d)(%d)"%(dataScaleDVec,vi)))
          else:
            raise RuntimeError("Unsupported scaleDVec compute data type %s."%str(self.kernel["ProblemType"]["ComputeDataType"]))

      biasReductionModule = Module("biasReductionModule")
      if self.storeBiasD == 1:
        vgprIdx = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
        biasReductionModule.add(self.parentWriter.addStore(self.kernel, self.ss, 'Bias', addrCalc, "ValuC+%d"%vgprIdx, self.tmpS01, self.edge, comment="store Bias"))

      if isActivationInsertAfter:
        module.add(convertModule)
        module.add(packModule)
        module.add(activationModule)
      else:
        module.add(activationModule)
        module.add(scaleDVecModule)
        module.add(biasReductionModule)
        module.add(convertModule)
        module.add(packModule)

      if not self.kernel["StoreRemapVectorWidth"]:
        tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'D', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store D")
        if self.kernel["GroupLoadStore"]:
          storeCode.add(tmpStoreCode)
        else:
          if 1:#GSUGSU
            module.addGSUSYNC("//")
          module.add(tmpStoreCode)
          if 1:#GSUGSU
            module.addGSUSYNC("//")
            module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))
        self.storesIssued += 1
        if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and (self.kernel["GlobalSplitU"] == 1):
          self.storesIssued += 1
        if self.storeBiasD == 1:
          self.storesIssued += 1

      else:
        rpe = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr
        module.add(self.parentWriter.storeRemapAddLocalWrite(self.ss, addrCalc, sumIdx*rpe))
        # Column Block Shape has been written to LDS
        # Now read back and write out to global memory

    module.add(storeCode)

    # module.addComment("GSUSYNC2 24 v9") #GSUSYNC
    # print("GSUSYNC2 ", "addrVgpr: ", addrCalc.addrDVgpr, " ", self.tmpS01, " ", sumIdx)
    module.addGSUSYNC("\n") #GSUSYNC

    if 0: #self.kernel["StoreVectorWidth"]==2:
      module.add(self.GSUSYNC2(self.kernel["StoreVectorWidth"], self.kernel["ProblemType"]["DestDataType"].isSingle(), sumIdx , self.ss.elementSumIdx[elementIdx-1], vgpr(self.ss.elementAddr[elementIdx-1].addrDVgpr)))
      # module.add(MacroInstruction("GSUSYNC2", args=[sumIdx , self.ss.elementSumIdx[elementIdx-1], vgpr(self.ss.elementAddr[elementIdx-1].addrDVgpr)]))
    else:
      if (self.kernel["GlobalSplitU"] != 1 and self.kernel["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel'):
        module.add(self.GSUSYNC2(self.kernel["StoreVectorWidth"], self.kernel["ProblemType"]["DestDataType"].isSingle(), sumIdx ,0 ,vgpr(addrCalc.addrDVgpr)))
      # module.add(MacroInstruction("GSUSYNC2", args=[sumIdx ,vgpr(addrCalc.addrDVgpr)]))

    module.addGSUSYNC("\n") #GSUSYNC

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

  def _addSumAlphaWithCBeta(self, kernel, ss, gwvw, elementIdx, vc0, tmpVgpr, bf16CVTVgprStruct):
    module = Module("addSumAlphaWithCBeta #elementIdx%u, vc0 %u"%(elementIdx, vc0))
    for vi in range(0, gwvw):
      dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
      sumIdxV = ss.elementSumIdx[elementIdx] + vi
      if kernel["ProblemType"]["DestDataType"].isHalf():
        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          if self.parentWriter.states.asmCaps["HasWMMA"] and kernel["EnableMatrixInstruction"]:
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
