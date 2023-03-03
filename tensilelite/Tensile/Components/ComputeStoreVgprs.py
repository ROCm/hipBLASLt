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

from ..TensileInstructions import Module, SMulI32, VAddLShiftLeftU32, VAddU32, VMulLOU32, \
                            SWaitCnt, staticMultiply, vectorStaticDivide, \
                            vectorStaticRemainder, RegisterPoolResource, vgpr, sgpr, log2
from ..Component import ComputeStoreVgprs

class ComputeStoreVgprsMFMA(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.vgprs.cinRowPtr  : C buffer coulmn offset
        # writer.vgprs.coutRowPtrD : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.vgprs.cinRowPtr   = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.vgprs.coutRowPtrD  = writer.vgprPool.checkOut(1, "coutRowPtrD")
            if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1):
                writer.vgprs.coutRowPtrE = writer.vgprPool.checkOut(1, "coutRowPtrE")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        tmpVgpr0Res = RegisterPoolResource(tmpVgpr0, 1)
        tmpVgpr1Res = RegisterPoolResource(tmpVgpr1, 2)
        dummy    = writer.vgprPool.checkOut(1,"dummy")

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx

            # constant
            MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
            MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

            # matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
            matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

            module = Module("ComputeStoreVgprsMFMA")

            # coord 1 : wave part
            module.add(vectorStaticDivide(wave_id, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res))
            module.add(vectorStaticDivide(tid1, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res))
            module.add(VMulLOU32(dst=vgpr(tid1), src0=hex(MIBShape1), src1=vgpr(tid1), comment="wave coordination offset 1"))

            # coord 1 : thread part
            module.add(vectorStaticRemainder(dummy, tmpVgpr0, "Serial", matrixInstN, tmpVgpr1Res, tmpSgprInfo))
            module.add(VAddU32(dst=vgpr(tid1), src0=vgpr(tmpVgpr0), src1=vgpr(tid1), comment="coordination 1 = wave_id1 + tid1"))

            # coord 1 : offset part
            packedC1 = kernel["PackedC1IndicesX"]
            strideC1 = "StrideC%s" % (writer.states.indexChars[packedC1[0]])
            strideD1 = "StrideD%s" % (writer.states.indexChars[packedC1[0]])
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.cinRowPtr), src0=vgpr(tid1), src1=sgpr(strideC1), comment=" offset 1"))
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrD), src0=vgpr(tid1), src1=sgpr(strideD1), comment=" offset 1"))
            if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1):
                if writer.states.numStoreSgprToLoad: # Wait for kernel args
                    lgkwait = writer.states.numStoreSgprInst - 1
                    module.add(SWaitCnt(lgkmcnt=lgkwait, comment="wait for 1 s_load."))
                strideE1 = "StrideE%s" % (writer.states.indexChars[packedC1[0]])
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrE), src0=vgpr(tid1), src1=sgpr(strideE1), comment=" offset 1"))

            # coord 0 : wave part
            module.add(vectorStaticRemainder(dummy, tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res, tmpSgprInfo))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr0), src0=hex(MIBShape0), src1=vgpr(tmpVgpr0), comment="wave coordination offset 0"))

            # coord 0 : thread part
            module.add(vectorStaticRemainder(dummy, tid0, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res, tmpSgprInfo))
            module.add(vectorStaticDivide(tid0, tid0, matrixInstN, tmpVgpr1Res))
            module.add(staticMultiply(vgpr(tid0), vgpr(tid0), kernel["MIOutputVectorWidth"], tmpSgprInfo, "thread0 * continuous_output"))
            module.add(VAddU32(dst=vgpr(tid0), src0=vgpr(tmpVgpr0), src1=vgpr(tid0), comment="coordination 0 = wave_id0 + tid0"))

            wg0="WorkGroup0"
            wg1="WorkGroup1"

            # macro tile 0 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr(wg0), comment="wgp0 * MT0"))
            module.add(VAddU32(dst=vgpr(tid0), src0=sgpr(tmpSgpr), src1=vgpr(tid0), comment="coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0"))

            # macro tile 1 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr(wg1), comment="wgp1 * MT1"))
            module.add(VAddU32(dst=vgpr(tid1), src0=sgpr(tmpSgpr), src1=vgpr(tid1), comment="coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1"))

            # extract packed rowStart vgpr
            if len(packedC1) > 1:
                module.add(writer.extractPackedCoord1ToRowStart(kernel, packedC1, tid1, 'D'))

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            module.add(writer.storeRemapComputeStoreVgprs(kernel))

        writer.vgprs.coord0 = tid0
        writer.vgprs.coord1 = tid1

        return module

class ComputeStoreVgprsMFMASwap(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": True}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.vgprs.cinRowPtr  : C buffer coulmn offset
        # writer.vgprs.coutRowPtrD : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.vgprs.cinRowPtr  = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.vgprs.coutRowPtrD = writer.vgprPool.checkOut(1, "coutRowPtrD")
            if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1):
                writer.vgprs.coutRowPtrE = writer.vgprPool.checkOut(1, "coutRowPtrE")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        tmpVgpr1Res = RegisterPoolResource(tmpVgpr1, 2)
        dummy    = writer.vgprPool.checkOut(1,"dummy")

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            tmpSgpr  = tmpSgprInfo.idx

            # constant
            MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
            MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

            matrixInstM = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
            # matrixInstN = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]

            module = Module("ComputeStoreVgprsMFMASwap")

            module.add(vectorStaticDivide(wave_id, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res))

            # coord 1 : wave part
            module.add(vectorStaticDivide(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr0), src0=hex(MIBShape1), src1=vgpr(tmpVgpr0), comment="wave coordination offset 1"))

            # coord 1 : thread part
            module.add(vectorStaticRemainder(dummy, tid1, "Serial", writer.states.kernel["WavefrontSize"], tmpVgpr1Res, tmpSgprInfo))
            module.add(vectorStaticDivide(tid1, tid1, matrixInstM, tmpVgpr1Res))
            module.add(staticMultiply(vgpr(tid1), vgpr(tid1), kernel["MIOutputVectorWidth"], tmpSgprInfo, "thread0 * continuous_output"))
            module.add(VAddU32(dst=vgpr(tid1), src0=vgpr(tmpVgpr0), src1=vgpr(tid1), comment="coordination 1 = wave_id1 + tid1"))
            if kernel["allowLRVWforTLUandMI"] and writer.states.lrvwB > 1:
                module.add(staticMultiply(vgpr(tid1), vgpr(tid1), writer.states.lrvwB, tmpSgprInfo, "coordination 1 *= lrvwB"))

            # coord 1 : offset part
            packedC1 = kernel["PackedC1IndicesX"]
            strideC1 = "StrideC%s" % (writer.states.indexChars[packedC1[0]])
            strideD1 = "StrideD%s" % (writer.states.indexChars[packedC1[0]])
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.cinRowPtr), src0=vgpr(tid1), src1=sgpr(strideC1), comment=" offset 1"))
            module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrD), src0=vgpr(tid1), src1=sgpr(strideD1), comment=" offset 1"))
            if kernel["ProblemType"]["UseE"] and (kernel["GlobalSplitU"] == 1):
                if writer.states.numStoreSgprToLoad: # Wait for kernel args
                    lgkwait = writer.states.numStoreSgprInst - 1
                    module.add(SWaitCnt(lgkmcnt=lgkwait, comment="wait for 1 s_load."))
                strideE1 = "StrideE%s" % (writer.states.indexChars[packedC1[0]])
                module.add(VMulLOU32(dst=vgpr(writer.vgprs.coutRowPtrE), src0=vgpr(tid1), src1=sgpr(strideE1), comment=" offset 1"))

            # coord 0 : wave part
            module.add(vectorStaticRemainder(dummy, tid0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1Res, tmpSgprInfo))
            module.add(VMulLOU32(dst=vgpr(tid0), src0=hex(MIBShape0), src1=vgpr(tid0), comment="wave coordination offset 0"))

            # coord 0 : thread part
            module.add(vectorStaticRemainder(dummy, tmpVgpr0, "Serial", matrixInstM, tmpVgpr1Res, tmpSgprInfo))
            module.add(VAddLShiftLeftU32(dst=vgpr(tid0), src0=vgpr(tmpVgpr0), src1=vgpr(tid0), shiftHex=log2(kernel["VectorWidth"]), comment="coordination 0 = wave_id0 + tid0"))

            wg0="WorkGroup0"
            wg1="WorkGroup1"

            # macro tile 0 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr(wg0), comment="wgp0 * MT0"))
            module.add(VAddU32(dst=vgpr(tid0), src0=sgpr(tmpSgpr), src1=vgpr(tid0), comment="coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0"))

            # macro tile 1 part
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile1"], src1=sgpr(wg1), comment="wgp1 * MT1"))
            module.add(VAddU32(dst=vgpr(tid1), src0=sgpr(tmpSgpr), src1=vgpr(tid1), comment="coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1"))

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            module.add(writer.storeRemapComputeStoreVgprs(kernel))

        writer.vgprs.coord0 = tid0
        writer.vgprs.coord1 = tid1

        return module
