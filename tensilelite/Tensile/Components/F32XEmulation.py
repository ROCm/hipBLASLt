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
                            VCvtPkF32toBF16, VAndB32, \
                            VMovB32, VLShiftLeftB32
from ..TensileInstructions import *
from ..TensileInstructions import VCvtBF16toFP32
from ..TensileInstructions.Instructions import *
from ..TensileInstructions import DataType, \
                            SaturateCastType, VSaturateCastInt

from ..Component import F32XEmulation

import re, types

class F32XEmulationCvtLocalWrite(F32XEmulation):
    asmCaps = {"HasMFMA_xf32": True}
    dbgCounter = 0
    def __call__(self):
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
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+0"), src0=vgpr("G2LA+0"), src1=vgpr("G2LA+1")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+1"), src0=vgpr("G2LA+2"), src1=vgpr("G2LA+3")))
        # tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        # low bits
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+8", src="Cvt+0", vgprMask="", vi=0))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+2"), src0=vgpr("G2LA+0"), src1=vgpr("Cvt+8")))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+9", src="Cvt+0", vgprMask="", vi=1))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+3"), src0=vgpr("G2LA+1"), src1=vgpr("Cvt+9")))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+8", src="Cvt+1", vgprMask="", vi=0))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+4"), src0=vgpr("G2LA+2"), src1=vgpr("Cvt+8")))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+9", src="Cvt+1", vgprMask="", vi=1))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+5"), src0=vgpr("G2LA+3"), src1=vgpr("Cvt+9")))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("G2LA+0"), src0=vgpr("Cvt+2"), src1=vgpr("G2LA+0")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("G2LA+1"), src0=vgpr("Cvt+3"), src1=vgpr("G2LA+1")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("G2LA+2"), src0=vgpr("Cvt+4"), src1=vgpr("G2LA+2")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("G2LA+3"), src0=vgpr("Cvt+5"), src1=vgpr("G2LA+3")))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))

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

class A:
    FOO = "foo of A"

def my_method(cls):
    return (cls, cls.FOO)


def issueLatencyOp(self):
    return

class F32XEmulationCvtLocalRead(F32XEmulation):
    asmCaps = {"HasMFMA_xf32": True}
    dbgCounter = 0
    def __call__(self, LocalReadX):
        tf32mod = Module()
        # Carson: textblock here (or in localread) is causing python issues. rocisa ambiguity issue?
        tf32mod.add(TextBlock("/*TF32 Emulation read lds*/\n"))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(TextBlock(str("label_tf32Read_") + str(F32XEmulationCvtLocalRead.dbgCounter) + ":\n"))
        F32XEmulationCvtLocalRead.dbgCounter += 1
        tf32mod.add(LocalReadX(dst=vgpr("Cvt+0"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=0)))
        tf32mod.add(LocalReadX(dst=vgpr("Cvt+1"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=256)))
        tf32mod.add(LocalReadX(dst=vgpr("Cvt+2"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=512)))
        tf32mod.add(LocalReadX(dst=vgpr("Cvt+3"), src=vgpr("LocalReadAddrA"), ds=DSModifiers(na=1, offset=768)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))

        #pack high bits
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+0"), src=vgpr("Cvt+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_1)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+0"), src=vgpr("Cvt+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_1)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+1"), src=vgpr("Cvt+2"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_1)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+1"), src=vgpr("Cvt+3"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_1)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        #pack low bits
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+2"), src=vgpr("Cvt+0"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_0)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+2"), src=vgpr("Cvt+1"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_0)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+3"), src=vgpr("Cvt+2"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_0, src0_sel=SelectBit.WORD_0)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))
        tf32mod.add(VMovB32(dst=vgpr("ValuA_X0_I0+3"), src=vgpr("Cvt+3"), sdwa=SDWAModifiers(dst_sel=SelectBit.WORD_1, src0_sel=SelectBit.WORD_0)))
        tf32mod.add(SWaitCnt(lgkmcnt=0, comment="wait for lds read"))

        # read format:
        # 0: [0high, 1high]
        # 1: [2high, 3high]
        # 2: [0low, 1low]
        # 3: [2low, 3low]
        #

        tf32mod.add(TextBlock(str("label_tf32Read_") + str(F32XEmulationCvtLocalRead.dbgCounter) + ":\n"))
        F32XEmulationCvtLocalRead.dbgCounter += 1
        return tf32mod


class F32XEmulationMFMA(F32XEmulation):
    asmCaps = {"HasMFMA_xf32": True}
    #kernel = {"ProblemType": {"DataType": "F32XdlMathOp"}, "UseF32XEmulation"}
    dbgCounter = 0
    def __call__(self, kernel, acc, acc2, src0, src1, miInInstType, miOutInstType, variant, mfma_1k, neg_flag):
        tf32mod = Module()
        tf32mod.add(TextBlock("/*tf32 emulation*/\n"))
        aStart = re.search(r'v\[vgpr(.*?):', str(src1)).group(1)
        bStart = re.search(r'v\[vgpr(.*?):', str(src0)).group(1)
        tf32mod.add(TextBlock("/*f32 to 2 bfloat16 per input*/\n"))
        tf32mod.add(TextBlock("/*bf16AHigh*/\n"))
        tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
        if not kernel["EnableF32XEmulationLds"]:
            tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+0"), src0=vgpr(aStart), src1=vgpr(aStart + "+1")))
            tf32mod.add(SNop(waitState=1, comment="1 wait states for ds_read"))
            tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+1"), src0=vgpr(aStart + "+2"), src1=vgpr(aStart + "+3")))
        tf32mod.add(TextBlock("/*bf16BHigh*/\n"))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+2"), src0=vgpr(bStart), src1=vgpr(bStart + "+1")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+3"), src0=vgpr(bStart + "+2"), src1=vgpr(bStart + "+3")))
        tf32mod.add(TextBlock("/*bf16ALow = A - float32(bf16AHigh)*/\n"))
        if not kernel["EnableF32XEmulationLds"]:
            tf32mod.add(VCvtBF16toFP32(dst="Cvt+8", src="Cvt+0", vgprMask="", vi=0))
            tf32mod.add(VSubF32(dst=vgpr("Cvt+8"), src0=vgpr(aStart+"+0"), src1=vgpr("Cvt+8")))
            tf32mod.add(VCvtBF16toFP32(dst="Cvt+9", src="Cvt+0", vgprMask="", vi=1))
            tf32mod.add(VSubF32(dst=vgpr("Cvt+9"), src0=vgpr(aStart+"+1"), src1=vgpr("Cvt+9")))
            tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+4"), src0=vgpr("Cvt+8"), src1=vgpr("Cvt+9")))
            tf32mod.add(VCvtBF16toFP32(dst="Cvt+8", src="Cvt+1", vgprMask="", vi=0))
            tf32mod.add(VSubF32(dst=vgpr("Cvt+8"), src0=vgpr(aStart+"+2"), src1=vgpr("Cvt+8")))
            tf32mod.add(VCvtBF16toFP32(dst="Cvt+9", src="Cvt+1", vgprMask="", vi=1))
            tf32mod.add(VSubF32(dst=vgpr("Cvt+9"), src0=vgpr(aStart+"+3"), src1=vgpr("Cvt+9")))
            tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+5"), src0=vgpr("Cvt+8"), src1=vgpr("Cvt+9")))
        tf32mod.add(TextBlock("/*bf16BLow = B - float32(bf16BHigh)*/\n"))
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+8", src="Cvt+2", vgprMask="", vi=0))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+8"), src0=vgpr(bStart+"+0"), src1=vgpr("Cvt+8")))
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+9", src="Cvt+2", vgprMask="", vi=1))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+9"), src0=vgpr(bStart+"+1"), src1=vgpr("Cvt+9")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+6"), src0=vgpr("Cvt+8"), src1=vgpr("Cvt+9")))
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+8", src="Cvt+3", vgprMask="", vi=0))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+8"), src0=vgpr(bStart+"+2"), src1=vgpr("Cvt+8")))
        tf32mod.add(VCvtBF16toFP32(dst="Cvt+9", src="Cvt+3", vgprMask="", vi=1))
        tf32mod.add(VSubF32(dst=vgpr("Cvt+9"), src0=vgpr(bStart+"+3"), src1=vgpr("Cvt+9")))
        tf32mod.add(VCvtPkF32toBF16(dst=vgpr("Cvt+7"), src0=vgpr("Cvt+8"), src1=vgpr("Cvt+9")))

        #todo: working impl using in situ cvt cmd. lds currently some kernels are failing
        if kernel["EnableF32XEmulationLds"]:
            tf32mod.add(TextBlock("/*acc = bf16ALow * bf16BHigh*/\n"))
            (src0, src1) = (vgpr("Cvt+2",2), vgpr(aStart + "+2",2))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BLow*/\n"))
            (src0, src1) = (vgpr("Cvt+6",2), vgpr(aStart,2))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BHigh*/\n"))
            (src0, src1) = (vgpr("Cvt+2",2), vgpr(aStart,2))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
        else:
            tf32mod.add(TextBlock("/*acc = bf16ALow * bf16BHigh*/\n"))
            (src0, src1) = (vgpr("Cvt+2",2), vgpr("Cvt+4",2))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BLow*/\n"))
            (src0, src1) = (vgpr("Cvt+6",2), vgpr("Cvt+0",2))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2, neg=neg_flag))
            tf32mod.add(TextBlock("/*acc += bf16AHigh * bf16BHigh*/\n"))
            (src0, src1) = (vgpr("Cvt+2",2), vgpr("Cvt+0",2))
            tf32mod.add(MFMAInstruction(instType=InstType.INST_BF16, accType=miOutInstType, variant=variant, mfma1k=mfma_1k, \
                                acc=acc, a=src0, b=src1, acc2=acc2))

        tf32mod.add(TextBlock(str("label_tf32_") + str(F32XEmulationMFMA.dbgCounter) + ":\n"))
        F32XEmulationMFMA.dbgCounter += 1
        return tf32mod