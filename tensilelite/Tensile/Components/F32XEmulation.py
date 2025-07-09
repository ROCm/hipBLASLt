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

from rocisa import rocIsa
from rocisa.code import Module, TextBlock
from rocisa.container import vgpr, sgpr,SDWAModifiers, VOP3PModifiers
from rocisa.enum import SelectBit, UnusedBit
from rocisa.instruction import VAdd3U32, VCvtF32toF16, VLShiftRightB32, \
                            VCmpUF32, VCndMaskB32, VCvtPkF32toFP8, VCvtPkF32toBF8, \
                            VCmpClassF32, VOrB32, VPackF16toB32, \
                            VAndOrB32, VBfeU32, VLShiftLeftB16, SNop, VMed3F32, \
                            VCvtPkF32toBF16, VAndB32, VCvtBF16toFP32, SWaitCnt, \
                            VMovB32, VLShiftLeftB32, VSubF32, MFMAInstruction
from rocisa.enum import InstType
# from ..TensileInstructions import *
#from ..TensileInstructions.Instructions import *
# from ..TensileInstructions import DataType, \
                            # SaturateCastType, VSaturateCastInt

from ..Component import F32XEmulation

import re, types

class F32XEmulationCvtLocalWrite(F32XEmulation):
    asmCaps = {"HasMFMA_xf32": True}
    dbgCounter = 0
    def __call__(self, srcStart):
        tf32mod = Module()
        tf32mod.add(TextBlock("/*TF32 Emulation write lds*/\n"))
        if (F32XEmulationCvtLocalWrite.dbgCounter == 0):
            tf32mod.add(TextBlock(str("label_tf32lds_begin_") + str(F32XEmulationCvtLocalWrite.dbgCounter) + ":\n"))
        # From:
        #
        # 0: G2LA+0 = 0, 4, 8, 12 <repeat>
        # 1: G2LA+1 = 1, 5, 9, 13 <repeat>
        # 2: G2LA+2 = 2, 6, 10, 14 <repeat>
        # 3: G2LA+3 = 3, 7, 11, 15
        #
        # To:
        #
        # 0: [0high, 0low]
        # 1: [1high, 1low]
        # 2: [2high, 2low]
        # 3: [3high, 3low]
        #
        # Carson: cannot do this, as it will break the 4 stride reassembly
        # 0: [0high, 1high]
        # 1: [2high, 3high]
        # 2: [0low, 1low]
        # 3: [2low, 3low]
        #


        # # high bits
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+0"), src0=vgpr(srcStart), src1=vgpr(srcStart + "+1")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+1"), src0=vgpr(srcStart + "+2"), src1=vgpr(srcStart + "+3")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        # low bits
        tf32mod.add(VCvtBF16toFP32(dst=vgpr("Cvt+8"), src=vgpr("Cvt+0"), vgprMask=None, vi=0))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+2"), src0=vgpr(srcStart), src1=vgpr("Cvt+8")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtBF16toFP32(dst=vgpr("Cvt+9"), src=vgpr("Cvt+0"), vgprMask=None, vi=1))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+3"), src0=vgpr(srcStart + "+1"), src1=vgpr("Cvt+9")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtBF16toFP32(dst=vgpr("Cvt+8"), src=vgpr("Cvt+1"), vgprMask=None, vi=0))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+4"), src0=vgpr(srcStart + "+2"), src1=vgpr("Cvt+8")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtBF16toFP32(dst=vgpr("Cvt+9"), src=vgpr("Cvt+1"), vgprMask=None, vi=1))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+5"), src0=vgpr(srcStart + "+3"), src1=vgpr("Cvt+9")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        #tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr(srcStart + "+0"), src0=vgpr("Cvt+2"), src1=vgpr(srcStart + "+0")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr(srcStart + "+1"), src0=vgpr("Cvt+3"), src1=vgpr(srcStart + "+1")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr(srcStart + "+2"), src0=vgpr("Cvt+4"), src1=vgpr(srcStart + "+2")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr(srcStart + "+3"), src0=vgpr("Cvt+5"), src1=vgpr(srcStart + "+3")))
        # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))

        # tf32mod.add(VMovB32(dst=vgpr("G2LA+0"), src=vgpr("Cvt+0")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        # tf32mod.add(VMovB32(dst=vgpr("G2LA+1"), src=vgpr("Cvt+1")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        # tf32mod.add(VCvtPkF32toBF16(dst=vgpr("G2LA+2"), src0=vgpr("Cvt+2"), src1=vgpr("G2LA+3")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        # tf32mod.add(VCvtPkF32toBF16(dst=vgpr("G2LA+3"), src0=vgpr("Cvt+4"), src1=vgpr("G2LA+5")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        if (F32XEmulationCvtLocalWrite.dbgCounter == 0):
            tf32mod.add(TextBlock(str("label_tf32lds_end_") + str(F32XEmulationCvtLocalWrite.dbgCounter) + ":\n"))
        F32XEmulationCvtLocalWrite.dbgCounter += 1

        return tf32mod

def issueLatencyOp(self):
    return

class F32XEmulationCvtLocalRead(F32XEmulation):
    asmCaps = {"HasMFMA_xf32": True}
    dbgCounter = 0
    def __call__(self, dstStart, width):
        numElementsPerIter = 4
        vgprSize = width / 2
        numIters = int(width / numElementsPerIter)
        tf32mod = Module()
        # Carson: textblock here (or in localread) is causing python issues. rocisa ambiguity issue?
        tf32mod.add(TextBlock("/*TF32 Emulation read lds*/\n"))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        #tf32mod.add(SNop(waitState=27, comment="wait for ds_read"))
        tf32mod.add(TextBlock(str("label_tf32Read_") + str(F32XEmulationCvtLocalRead.dbgCounter) + ":\n"))
        F32XEmulationCvtLocalRead.dbgCounter += 1
        #tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        for itr in range(numIters):
            offset = itr * 4
            offsetStr = "+" + str(offset)
            tf32mod.add(VMovB32(dst=vgpr("Cvt+0" + offsetStr), src=vgpr(dstStart + "+0" + offsetStr)))
            tf32mod.add(VMovB32(dst=vgpr("Cvt+1" + offsetStr), src=vgpr(dstStart + "+1" + offsetStr)))
            tf32mod.add(VMovB32(dst=vgpr("Cvt+2" + offsetStr), src=vgpr(dstStart + "+2" + offsetStr)))
            tf32mod.add(VMovB32(dst=vgpr("Cvt+3" + offsetStr), src=vgpr(dstStart + "+3" + offsetStr)))
        #tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))

        cvtOffset = 0
        cvtOffsetStr = "+" + str(cvtOffset)
        ldsOffset = 0
        ldsOffsetStr = "+" + str(ldsOffset)
        #pack high bits
        for itr in range(numIters):
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+0" + ldsOffsetStr), src=vgpr("Cvt+0" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_1)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+0" + ldsOffsetStr), src=vgpr("Cvt+1" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_1)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+1" + ldsOffsetStr), src=vgpr("Cvt+2" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_1)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+1" + ldsOffsetStr), src=vgpr("Cvt+3" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_1)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            cvtOffset += 4
            cvtOffsetStr = "+" + str(cvtOffset)
            ldsOffset += 2
            ldsOffsetStr = "+" + str(ldsOffset)
        cvtOffset = 0
        cvtOffsetStr = "+" + str(offset)
        #pack low bits
        for itr in range(numIters):
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+0" + ldsOffsetStr), src=vgpr("Cvt+0" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_0)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+0" + ldsOffsetStr), src=vgpr("Cvt+1" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_0)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+1" + ldsOffsetStr), src=vgpr("Cvt+2" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_0)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(VMovB32(dst=vgpr(dstStart + "+1" + ldsOffsetStr), src=vgpr("Cvt+3" + cvtOffsetStr), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_0)))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            cvtOffset += 4
            cvtOffsetStr = "+" + str(cvtOffset)
            ldsOffset += 2
            ldsOffsetStr = "+" + str(ldsOffset)

        # read format:
        # 0: [0high, 1high]
        # 1: [2high, 3high]
        # 2: [0low, 1low]
        # 3: [2low, 3low]
        #

        tf32mod.add(TextBlock(str("label_tf32Read_end_") + str(F32XEmulationCvtLocalRead.dbgCounter) + ":\n"))
        F32XEmulationCvtLocalRead.dbgCounter += 1
        return tf32mod


class F32XEmulationMFMA(F32XEmulation):
    asmCaps = {"HasMFMA_xf32": True}
    #kernel = {"ProblemType": {"DataType": "F32XdlMathOp"}, "UseF32XEmulation"}
    dbgCounter = 0
    # width = vgprPerInput (4 or 8)
    def __call__(self, kernel, acc, acc2, src0, src1, miInInstType, miOutInstType, variant, mfma_1k, width, neg_flag):
        tf32mod = Module()
        tf32mod.add(TextBlock("/*tf32 emulation*/\n"))
        aStart = re.search(r'v\[vgpr(.*?):', str(src1)).group(1)
        bStart = re.search(r'v\[vgpr(.*?):', str(src0)).group(1)
        # print("src0: {0} aStart: {1}".format(src0, aStart))
        # print("src1: {0} bStart: {1}".format(src1, bStart))
        tf32mod.add(TextBlock("/*f32 to 2 bfloat16 per input*/\n"))
        tf32mod.add(TextBlock("/*bf16AHigh*/\n"))
        numElementsPerIter = 4
        vgprSize = width / 2

        # Cvt register layout:
        #cvt0-1: Tmp
        #cvt2-5: AHigh
        #cvt6-9: Bhigh
        #cvt12-15: ALow
        #cvt16-19: BLow

        aHigh = "Cvt+2"
        bHigh = "Cvt+6"
        aLow = "Cvt+12"
        bLow = "Cvt+16"

        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(SNop(waitState=64, comment="1 wait states for ds_read"))
        tf32mod.add(TextBlock(str("label_tf32_") + str(F32XEmulationMFMA.dbgCounter) + "_0:\n"))

        for itr in range(int(width / numElementsPerIter)):
            #tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            #tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
            offset = "+" + str(itr * 2)
            aHigh1 = aHigh + offset
            aHigh2 = aHigh + "+1" + offset
            aLow1 = aLow + offset
            aLow2 = aLow + "+1" + offset
            bHigh1 = bHigh + offset
            bHigh2 = bHigh + "+1" + offset
            bLow1 = bLow + offset
            bLow2 = bLow + "+1" + offset
            tmp1 = "Cvt+0"
            tmp2 = "Cvt+1"
            aIndex = aStart + str(itr * 4)
            bIndex = bStart + str(itr * 4)
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
            if not kernel["EnableF32XEmulationLds"]:
                tf32mod.add(TextBlock("/*bf16AHigh*/\n"))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(aHigh1), src0=vgpr(aIndex), src1=vgpr(aIndex + "+1")))
                tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(aHigh2), src0=vgpr(aIndex + "+2"), src1=vgpr(aIndex + "+3")))
                tf32mod.add(TextBlock("/*bf16BHigh*/\n"))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(bHigh1), src0=vgpr(bIndex), src1=vgpr(bIndex + "+1")))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(bHigh2), src0=vgpr(bIndex + "+2"), src1=vgpr(bIndex + "+3")))
        tf32mod.add(TextBlock(str("label_tf32_") + str(F32XEmulationMFMA.dbgCounter) + "_1:\n"))

        tf32mod.add(SNop(waitState=64, comment="1 wait states for ds_read"))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BHigh*/\n"))
        if kernel["EnableF32XEmulationLds"]:
            (src0, src1) = (vgpr(bStart,vgprSize), vgpr(aStart,vgprSize))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
        else:
            (src0, src1) = (vgpr(bHigh,vgprSize), vgpr(aHigh,vgprSize))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2))

        for itr in range(int(width / numElementsPerIter)):
            offset = "+" + str(itr * 2)
            aHigh1 = aHigh + offset
            aHigh2 = aHigh + "+1" + offset
            aLow1 = aLow + offset
            aLow2 = aLow + "+1" + offset
            bHigh1 = bHigh + offset
            bHigh2 = bHigh + "+1" + offset
            bLow1 = bLow + offset
            bLow2 = bLow + "+1" + offset
            tmp1 = "Cvt+0"
            tmp2 = "Cvt+1"
            aIndex = aStart + str(itr * 4)
            bIndex = bStart + str(itr * 4)
            if kernel["EnableF32XEmulationLds"]:
                None
                #tf32mod.add(SNop(waitState=1020, comment="1 wait states for ds_read"))
                #tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
                #tf32mod.add(TextBlock("/*acc = bf16ALow * bf16BHigh*/\n"))
                #(src0, src1) = (vgpr("Cvt+2",2), vgpr(aStart + "+2",2))
                #tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                #                    acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
                #tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            if not kernel["EnableF32XEmulationLds"]:
                tf32mod.add(TextBlock("/*bf16ALow = A - float32(bf16AHigh)*/\n"))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp1), src=vgpr(aHigh1), vgprMask=None, vi=0))
                tf32mod.add(VSubF32(dst=vgpr(tmp1), src0=vgpr(aIndex+"+0"), src1=vgpr(tmp1)))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp2), src=vgpr(aHigh1), vgprMask=None, vi=1))
                tf32mod.add(VSubF32(dst=vgpr(tmp2), src0=vgpr(aIndex+"+1"), src1=vgpr(tmp2)))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(aLow1), src0=vgpr(tmp1), src1=vgpr(tmp2)))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp1), src=vgpr(aHigh2), vgprMask=None, vi=0))
                tf32mod.add(VSubF32(dst=vgpr(tmp1), src0=vgpr(aIndex+"+2"), src1=vgpr(tmp1)))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp2), src=vgpr(aHigh2), vgprMask=None, vi=1))
                tf32mod.add(VSubF32(dst=vgpr(tmp2), src0=vgpr(aIndex+"+3"), src1=vgpr(tmp2)))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(aLow2), src0=vgpr(tmp1), src1=vgpr(tmp2)))

        tf32mod.add(SNop(waitState=64, comment="1 wait states for ds_read"))
        tf32mod.add(TextBlock("/*acc = bf16ALow * bf16BHigh*/\n"))
        if kernel["EnableF32XEmulationLds"]:
            (src0, src1) = (vgpr(bStart,vgprSize), vgpr(aStart + "+" + str(vgprSize),vgprSize))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
        else:
            (src0, src1) = (vgpr(bHigh,vgprSize), vgpr(aLow,vgprSize))
            # # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            # tf32mod.add(SNop(waitState=64, comment="1 wait s00tates for ds_read"))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))

        for itr in range(int(width / numElementsPerIter)):
            offset = "+" + str(itr * 2)
            aHigh1 = aHigh + offset
            aHigh2 = aHigh + "+1" + offset
            aLow1 = aLow + offset
            aLow2 = aLow + "+1" + offset
            bHigh1 = bHigh + offset
            bHigh2 = bHigh + "+1" + offset
            bLow1 = bLow + offset
            bLow2 = bLow + "+1" + offset
            tmp1 = "Cvt+0"
            tmp2 = "Cvt+1"
            aIndex = aStart + str(itr * 4)
            bIndex = bStart + str(itr * 4)

            if not kernel["EnableF32XEmulationLds"]:
                tf32mod.add(TextBlock("/*bf16BLow = B - float32(bf16BHigh)*/\n"))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp1), src=vgpr(bHigh1), vgprMask=None, vi=0))
                tf32mod.add(VSubF32(dst=vgpr(tmp1), src0=vgpr(bIndex+"+0"), src1=vgpr(tmp1)))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp2), src=vgpr(bHigh1), vgprMask=None, vi=1))
                tf32mod.add(VSubF32(dst=vgpr(tmp2), src0=vgpr(bIndex+"+1"), src1=vgpr(tmp2)))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(bLow1), src0=vgpr(tmp1), src1=vgpr(tmp2)))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp1), src=vgpr(bHigh2), vgprMask=None, vi=0))
                tf32mod.add(VSubF32(dst=vgpr(tmp1), src0=vgpr(bIndex+"+2"), src1=vgpr(tmp1)))
                tf32mod.add(VCvtBF16toFP32(dst=vgpr(tmp2), src=vgpr(bHigh2), vgprMask=None, vi=1))
                tf32mod.add(VSubF32(dst=vgpr(tmp2), src0=vgpr(bIndex+"+3"), src1=vgpr(tmp2)))
                tf32mod.add(VCvtPkF32toBF16(dst=vgpr(bLow2), src0=vgpr(tmp1), src1=vgpr(tmp2)))

            # aStart = aStart + "+4"
            # bStart = bStart + "+4"

        #todo: working impl using in situ cvt cmd. lds currently some kernels are failing
        tf32mod.add(SNop(waitState=64, comment="1 wait states for ds_read"))
        tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BLow*/\n"))
        if kernel["EnableF32XEmulationLds"]:
            # tf32mod.add(TextBlock("/*acc = bf16ALow * bf16BHigh*/\n"))
            # (src0, src1) = (vgpr("Cvt+2",2), vgpr(aStart + "+2",2))
            # tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
            #                     acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            (src0, src1) = (vgpr(bStart + "+" + str(vgprSize),vgprSize), vgpr(aStart,vgprSize))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            # tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BHigh*/\n"))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            # (src0, src1) = (vgpr("Cvt+2",2), vgpr(aStart,2))
            # tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
            #                     acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        else:
            (src0, src1) = (vgpr(bLow,vgprSize), vgpr(aHigh,vgprSize))
            # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                               acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            # tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BHigh*/\n"))
            # (src0, src1) = (vgpr(bHigh,vgprSize), vgpr(aHigh,vgprSize))
            # # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            # tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
            #                     acc=acc, a=src0, b=src1, acc2=acc2))            # tf32mod.add(TextBlock("/*acc = bf16ALow * bf16BHigh*/\n"))
            # tf32mod.add(TextBlock("/*acc = bf16ALow * bf16BHigh*/\n"))
            # (src0, src1) = (vgpr(bHigh,vgprSize), vgpr(aLow,vgprSize))
            # # tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
            # tf32mod.add(SNop(waitState=64, comment="1 wait s00tates for ds_read"))
            # tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
            #                    acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))

        tf32mod.add(TextBlock(str("label_tf32_") + str(F32XEmulationMFMA.dbgCounter) + ":\n"))
        F32XEmulationMFMA.dbgCounter += 1
        return tf32mod
