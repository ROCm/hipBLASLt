################################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .TensileInstructions import DataType, Label, Module, vgpr, sgpr, accvgpr, \
                                 Holder, SBranchIfNotZero
from .TensileInstructions.Instructions import *

def allocPostLoopSrdSuppress(ch: str, labelStr: str) -> Module:
    module = Module("allocPostLoopSrdSuppress")
    label  = Label("%sAddrValid"%labelStr, "")
    label2 = Label("%sAddrValid_End"%labelStr, "")
    # Buffer-load uses one base read pointer stored in the SRD - set it here:
    module.add(SMovB32(dst=sgpr("Srd%s+0"%ch), src=sgpr("Address%s+0"%ch), comment="init SRD base address (lower)" ))
    module.add(SMovB32(dst=sgpr("Srd%s+1"%ch), src=sgpr("Address%s+1"%ch), comment="init SRD base address (upper) + other fields" ))
    module.add(SMovB32(dst=sgpr("Srd%s+3"%ch), src="Srd127_96", comment="Set bits 127_96 in post-loop SRD"))
    module.add(SBranchIfNotZero("Address%s"%ch, DataType('int64'), label))
    module.add(SMovB32(dst=sgpr("Srd%s+2"%ch), src=0))
    module.add(SBranch(label2.getLabelName()))
    module.add(label)
    module.add(SMovB32(dst=sgpr("Srd%s+2"%ch), src=sgpr("SizeI")))
    module.add(label2)
    module.addSpaceLine()
    return module

##############################################################################
# WaitCnt
# 3 components can contribute to the waitcnt:
#   - Pending global reads.  (skipGlobalRead)
#   - Pending local write.  (skipLocalWrite)
#   - Pending local reads (skipLocalRead)
# If a skip* arg is -1, the associated component does not contribute to
# the expected lgkmcnt or vmcnt
##############################################################################
def wait(states, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, \
    skipLocalRead, conservativeWaitCnt: int, comment):
    # skip = -1 -> ignore
    # skip =  n -> waitcnt(n*num)

    lgkmcnt = 0 if skipLocalWrite > -1 or skipLocalRead > -1 else -1

    if skipLocalWrite > -1 or skipLocalRead > -1:
        if skipLocalWrite > -1:
            numA = 0 if kernel["DirectToLdsA"] \
                   else tPA["nrp"]*tPA["nrc"]*max(tPA["nwcv"],tPA["nwpv"])//tPA["nwcvpi"]
            numB = 0 if kernel["DirectToLdsB"] \
                   else tPB["nrp"]*tPB["nrc"]*max(tPB["nwcv"],tPB["nwpv"])//tPB["nwcvpi"]
            lgkmcnt += skipLocalWrite * (numA + numB)
        if skipLocalRead > -1:
            readsPerIter = states.numReadsPerIterA + states.numReadsPerIterB
            lgkmcnt += skipLocalRead * readsPerIter

    vmcnt = 0 if skipGlobalRead > -1 else -1
    if skipGlobalRead > -1:
        numA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"]
        numB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"]
        vmcnt += skipGlobalRead * (numA + numB)

        # Unlike flat loads, BufferLoad do not increment the outstanding
        # lgkmcnt
        if lgkmcnt > -1 and not kernel["BufferLoad"]:
            lgkmcnt += skipGlobalRead * (numA + numB)

    if (conservativeWaitCnt & 0x2) and skipGlobalRead != -1 or \
       (conservativeWaitCnt & 0x4) and skipLocalWrite != -1 or \
       (conservativeWaitCnt & 0x8) and skipLocalRead  != -1:
        imod = Module("ConservativeWaitCnt")
        imod.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=0, comment="debug %s"%comment))
        imod.add(SBarrier(comment="debug"))
        return imod

    maxLgkmcnt = states.asmCaps["MaxLgkmcnt"]
    lgkmcnt = min(lgkmcnt, maxLgkmcnt)
    if lgkmcnt >= 0 and vmcnt >= 0:
        vmcnt = -1 # preserve prior behavior of removing vmcnt here?
    maxVmcnt = states.asmCaps["MaxVmcnt"]
    vmcnt = min(vmcnt, maxVmcnt)
    # This line is added for backward compatibility
    vscnt = vmcnt if lgkmcnt != -1 and vmcnt != -1 and states.archCaps["SeparateVscnt"] else -1

    waitcnt = SWaitCnt(lgkmcnt,vmcnt, vscnt, comment)
    return waitcnt

##############################################################################
# SyncThreads
##############################################################################
def syncThreads(kernel, archCaps, comment=""):
    imod = Module("syncThreads")
    if kernel["NumThreads"] > kernel["WavefrontSize"]:
        if archCaps["SeparateVscnt"]:
            imod.add(SWaitCnt(vscnt=0))
        elif kernel.enabledSplitLDS or kernel["ScheduleIterAlg"] == 2 \
          or kernel["PrefetchGlobalRead"] == 2:
            imod.addComment("Skip force waitcnt0")
        elif archCaps["Waitcnt0Disabled"]:
            # FIXME: should we add s_waitcnt_vscnt?
            imod.add(SWaitCnt(lgkmcnt=0, vmcnt=0, vscnt=-1, comment="force waitcnt0"))

        imod.add(SBarrier(comment=comment))
    else:
        imod.addComment("Skip barrier: NumThreads=%s"%(kernel["NumThreads"]) + \
                comment)
    return imod

def _getAccToArchInfo(kernel, lrvwB):
  matrixInstM  = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
  matrixInstN  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
  matrixInstBM = 1                                                if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
  matrixInstBN = 1                                                if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

  OutputsPerMFMA1B = matrixInstM * matrixInstN // kernel["WavefrontSize"]
  VectorWidth0     = kernel["VectorWidth"] if kernel["SourceSwap"] else 1
  outerTT0         = kernel["MIWaveTile"][0] // VectorWidth0
  lrvwB            = lrvwB if kernel["allowLRVWforTLUandMI"] else 1
  VectorWidth1     = lrvwB
  outerTT1         = kernel["MIWaveTile"][1] // VectorWidth1
  return matrixInstBM, matrixInstBN, OutputsPerMFMA1B, VectorWidth0, outerTT0, lrvwB, outerTT1

def getAccToArchLen(kernel, lrvwB):
  matrixInstBM, matrixInstBN, OutputsPerMFMA1B, VectorWidth0, outerTT0, lrvwB, outerTT1 = _getAccToArchInfo(kernel, lrvwB)
  return (outerTT1 * lrvwB *  outerTT0 * matrixInstBN * matrixInstBM * OutputsPerMFMA1B * VectorWidth0)

##############################################################################
# accToArchMapper
# Provides forward (acc2arch) and backward (arch2acc) index transformation
#  - Forward transformation is currently used for acc->vgpr copying
#  - Backward transformation is used in ShiftVectorComponent() to map logical
#    C-tile index back to original acc index
##############################################################################
def accToArchMapper(kernel, lrvwB):
  acc2arch = dict()
  arch2acc = dict()

  matrixInstBM, matrixInstBN, OutputsPerMFMA1B, VectorWidth0, outerTT0, lrvwB, outerTT1 = _getAccToArchInfo(kernel, lrvwB)

  for wgIdx1 in range(0, outerTT1):
    for lb in range(0, lrvwB):
      for wgIdx0 in range(0, outerTT0):
        for bIdx1 in range(0, matrixInstBN):
          for bIdx0 in range(0, matrixInstBM):
            for tIdx in range(0, OutputsPerMFMA1B):
              for vw0 in range(0, VectorWidth0):
                src, dst = 0, 0
                if kernel["SourceSwap"]:
                  src = tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (bIdx1 + matrixInstBN * (vw0 + VectorWidth0 * (wgIdx0 + outerTT0 * (wgIdx1 * lrvwB + lb)))))
                  dst = vw0 + VectorWidth0 * ((bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * ((tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * wgIdx1)) * lrvwB + lb))))
                else:
                  src = tIdx + OutputsPerMFMA1B * (bIdx1 + matrixInstBN * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * wgIdx1)))
                  dst = tIdx + OutputsPerMFMA1B * (bIdx0 + matrixInstBM * (wgIdx0 + outerTT0 * (bIdx1 + matrixInstBN * wgIdx1)))
                acc2arch[src] = dst
                arch2acc[dst] = src
  return acc2arch, arch2acc

def accVgprImagNumOffset(kernel, lrvwB):
  acc2arch, _ = accToArchMapper(kernel, lrvwB)
  return len(acc2arch) * kernel["MIRegPerOut"]

##############################################################################
# MapAcctoArch
# function to map MFMA Acc  Registers to Arch VGPR register
##############################################################################
def mapAcctoArchRegs(kernel, lrvwB):
  acc2arch, _ = accToArchMapper(kernel, lrvwB)

  complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1
  imod = Module("AccVgprRead")
  imod.itemList = [None] * kernel["MIRegPerOut"] * complexMultiplier * len(acc2arch)
  accImOffset = accVgprImagNumOffset(kernel, lrvwB)
  for i in range(len(acc2arch)):
    for cm in range(complexMultiplier):
      for r in range(kernel["MIRegPerOut"]):
        destIdx = (acc2arch[i]*complexMultiplier + cm) * kernel["MIRegPerOut"] + r
        srcIdx = ((i * kernel["MIRegPerOut"] + r) + (cm*accImOffset))
        if not kernel["MIArchVgpr"]:
          accStr = accvgpr(srcIdx)
          imod.itemList[destIdx] =  VAccvgprReadB32(dst=vgpr(Holder(name="ValuC")),
                                                            src=accStr,
                                                            comment="copy acc to vreg[%u]" % destIdx)
        else:
          imod.itemList[destIdx] =  VMovB32(dst=vgpr(Holder(name="ValuC")),
                                                            src=vgpr("ValuC+%u"%srcIdx),
                                                            comment="copy MI out reg to vreg[%u]" % destIdx)
  return imod

##############################################################################
# MulMIoutAlphaToArch
# function to handle MFMA alpha*MIout to Arch VGPR register
##############################################################################
def mulMIoutAlphaToArch(kernel, lrvwB, startVgprAlphaTmp):
  acc2arch, _ = accToArchMapper(kernel, lrvwB)

  imod = Module("MulAlpha")
  imod.itemList = [None] * len(acc2arch)
  for i in range(len(acc2arch)):
    destIdx = acc2arch[i]
    srcIdx  = i * kernel["MIRegPerOut"]
    if kernel["ProblemType"]["ComputeDataType"].isDouble():
      imod.itemList[destIdx] = VMulF64(dst=vgpr(Holder(name="ValuC"),2),
                                                    src0=sgpr("Alpha",2), src1=vgpr("ValuC+%u"%srcIdx,2),
                                                    comment="Multiply MI out reg with alpha")
    elif kernel["ProblemType"]["ComputeDataType"].isSingle() or \
        (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):
      imod.itemList[destIdx] = VMulF32(dst=vgpr(Holder(name="ValuC")),
                                                    src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%srcIdx),
                                                    comment="Multiply MI out reg with alpha")
    elif kernel["ProblemType"]["ComputeDataType"].isInt32():
      imod.itemList[destIdx] = VMulLOU32(dst=vgpr(Holder(name="ValuC")),
                                                      src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%srcIdx),
                                                       comment="Multiply MI out reg with alpha")
    elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
      accImOffset = accVgprImagNumOffset(kernel, lrvwB)
      cimod = Module()
      # cannot use tmp vgpr for write batch, use allocated vgpr instead
      vtmp1 = startVgprAlphaTmp
      vtmp2 = vtmp1 + 2
      # tmp1 = a.real * b.real
      cimod.add(VMulF64(dst=vgpr(vtmp1,2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%srcIdx,2)))
      # tmp2 = a.imag * b.real
      cimod.add(VMulF64(dst=vgpr(vtmp2,2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%srcIdx,2)))
      # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
      cimod.add(VFmaF64(dst=vgpr(Holder(name="ValuC"),2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(srcIdx+accImOffset),2), src2=vgpr(vtmp1,2)))
      # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
      cimod.add(VFmaF64(dst=vgpr(Holder(name="ValuC+2"),2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(srcIdx+accImOffset),2), src2=vgpr(vtmp2,2)))
      imod.itemList[destIdx] = cimod
  return imod
