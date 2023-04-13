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

from ..Component import SumUnroll
from ..Common import printExit
from ..TensileInstructions import Module, VDot2F32F16, SMovB32, VAddU32, VCmpXEqU32, \
    VLShiftLeftB32, \
    staticMultiply, vectorStaticDivide, vectorStaticRemainder, \
    DSModifiers, SSetMask, DSStoreB16, DSStoreB32, DSStoreB64, \
    EXEC, vgpr, sgpr, RegisterPoolResource, log2

class SumUnrollMfma(SumUnroll):
    kernel = {"EnableMatrixInstruction": True}

    """
    Sum unroll for reduction
    Use the same pattern as mfma
    """
    def __call__(self):
        assert(0)

    def loopSum(self, writer, kernel, tc, u, innerUnroll):
        imod = Module("SumUnroll%s_I%s" % (tc, innerUnroll))
        assert (not kernel["DirectToVgpr%s"%tc])

        m = (u) % (writer.states.numVgprBuffer+1) # local to use for MACs

        # calculate constant
        numRegistersIn   = kernel["ProblemType"]["DataType"].numRegisters()
        numMIInput       = kernel["MIInputPerThread"]
        vgprPerInput     = int(numMIInput * numRegistersIn)

        if tc == "A":
            waveTile = kernel["MIWaveTile"][0]
            numIterPerCoalescedRead = writer.states.numIterPerCoalescedReadA
            numReadsIterCoalesced = writer.states.numReadsIterCoalescedA
        elif tc == "B":
            waveTile = kernel["MIWaveTile"][1]
            numIterPerCoalescedRead = writer.states.numIterPerCoalescedReadB
            numReadsIterCoalesced = writer.states.numReadsIterCoalescedB
        else:
            printExit("Unsupported tc %s"%tc)
        # here we remap index to where it read for wider local read
        # ex. if we read 2 iteration at a time,
        #   original   : _ds_load_b64  valuA_X0_I0
        #   read 2 iter: _ds_load_b128 valuA_X0_I0 (we read valuA_X0_I0 and valuA_X1_I0)
        # instead of using valuA_X1_I0, we use valuA_X0_I0+2 as mfma input

        vgprBuffer_new = (m//numIterPerCoalescedRead)*numIterPerCoalescedRead
        vgprBuffer_new_offset = m%numIterPerCoalescedRead*kernel["InnerUnroll"]*vgprPerInput

        tmpSgpr = -1
        if numRegistersIn < 1:
            assert kernel["ProblemType"]["DataType"].isHalf()
            tmpSgpr = writer.sgprPool.checkOut(1)
            imod.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x3c003c00), comment="packed 1.0"))

        for iui in range(0, innerUnroll):
            iui_new = (iui//numReadsIterCoalesced)*numReadsIterCoalesced
            iui_new_offset = iui%numReadsIterCoalesced*vgprPerInput
            for idx in range(0, waveTile):
                new     = idx*vgprPerInput*numReadsIterCoalesced
                # valuStr = "Valu%s_X%u_I%u+%u+%u+%u" % (tc, vgprBuffer_new, iui_new, new, vgprBuffer_new_offset, iui_new_offset)
                valuStr    = "Valu%s_X%u_I%u+%u+%u" % (tc, vgprBuffer_new, iui_new, new, vgprBuffer_new_offset)
                valuSumStr = "ValuSum+%u"%idx
                # If direct ot vgpr, use "G2LA+%u+%u+%u", currently not supported
                if kernel["ProblemType"]["DataType"].isHalf():
                    # First version only supports mfma with K > 1
                    if vgprPerInput > 1 and (vgprPerInput % 2 == 0):
                        for inputIdx in range(0, vgprPerInput):
                            imod.add(VDot2F32F16(dst=vgpr(valuSumStr), src0=vgpr("%s+%s"%(valuStr, iui_new_offset + inputIdx)), src1=sgpr(tmpSgpr), src2=vgpr(valuSumStr), comment="sum K"))
                    else:
                        printExit("Currently unsupported vgprPerInput %u"%vgprPerInput)
                else:
                    printExit("Currently unsupported data type")

        if tmpSgpr != -1:
            writer.sgprPool.checkIn(tmpSgpr)

        return imod

    """
    Store sum to LDS
    Use the same pattern as local read, except the leading dimension is the sum index instead of free indices
    totalVgprToBeStoredInK is calculated to find out the length of the sum index.
    """
    def storeSumLDS(self, writer, kernel, tPA, tPB):
        # bias data type
        diasBpe        = kernel["ProblemType"]["ComputeDataType"].numBytes()
        # get constant parameter
        tP             = tPA if kernel["ProblemType"]["BiasSrc"] == "A" else tPB
        tc             = tP["tensorChar"]
        tile01         = tP["tile01Idx"]
        waveWidth      = writer.states.kernel["WavefrontSize"]
        mt             = kernel["MacroTile%u" % tile01]
        flatWWorkGroup = kernel["SubGroup0"] * kernel["SubGroup1"] * kernel["LocalSplitU"]

        wReg    = writer.vgprPool.checkOut(1,"wReg") # quotient
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
        ldsVgpr = writer.vgprPool.checkOut(1,"ldsVgpr")
        ldsVgpr1 = writer.vgprPool.checkOut(1,"ldsVgpr1")
        dummy   = writer.vgprPool.checkOut(1,"dummy")

        # parameter for get each type index
        dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DBlocks1     = kernel["MatrixInstBN"] if (tile01 == 0) else kernel["MatrixInstBM"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = kernel["MatrixInstM"] if (tile01 == 0) else (kernel["MatrixInstM"] * kernel["MatrixInstBM"])
            dividedForBlkId1 = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (tile01 == 0) else kernel["MatrixInstM"]
        else:
            dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]
            dividedForBlkId1 = kernel["MatrixInstN"] if (tile01 == 0) else (kernel["MatrixInstN"] * kernel["MatrixInstBN"])
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth"] if ((tile01 == 0) and kernel["SourceSwap"]) else 1 # TODO: nonSwap VectorWidth

        # strider for each type of index
        # Calculate K
        totalVgprToBeStoredInK = flatWWorkGroup * num1DBlocks * kernel["MIWaveGroup"][tile01] * kernel["MIWaveTile"][tile01] \
                    // kernel["MatrixInstB"] // (kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]) // mt
        strideTile       = totalVgprToBeStoredInK
        strideBlock      = kernel["MatrixInstM"] * strideTile
        strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        module = Module("StoreSumLDS")
        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            # tile offset
            module.add(vectorStaticRemainder(dummy, kReg, "Serial", waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            module.add(vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgprRes, tmpSgprInfo, \
                "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"]))
            module.add(staticMultiply(vgpr(tReg), vgpr(tReg), strideTile, tmpSgprInfo, \
                "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile))
            # block offset
            # Here we calculate the coordinate of the block offset, and remove the duplicated blocks.
            # For example we have block = 2x2, source swap = False:
            # 0 1
            # 2 3
            # For tile01 = 0 we should only use 0, 2. For tile01 = 1 we should only use 0, 1.
            module.add(vectorStaticDivide(wReg, kReg, dividedForBlkId1, tmpVgprRes, \
                "2-1. block offset: bnIdx = wtid / dividedForBlkId1(%u)" % dividedForBlkId1))
            module.add(vectorStaticRemainder(dummy, wReg, wReg, num1DBlocks1, tmpVgprRes, tmpSgprInfo, \
                "2-1. block offset: bnIdx = bnIdx %% num1DBlocks1(%u)" % num1DBlocks1))
            module.add(VCmpXEqU32(dst=EXEC(), src0=vgpr(wReg), src1=0, comment="2-1. True if ans = 0"))
            module.add(vectorStaticDivide(wReg, kReg, dividedForBlkId, tmpVgprRes, \
                "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId))
            module.add(vectorStaticRemainder(dummy, wReg, wReg, num1DBlocks, tmpVgprRes, tmpSgprInfo, \
                "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks))
            module.add(staticMultiply(vgpr(wReg), vgpr(wReg), strideBlock, tmpSgprInfo, \
                "2. block offset: bnOffset = bnIdx * strideBlock(%u)" % strideBlock))
            module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(wReg), src1=vgpr(tReg), \
                comment="3. add N and block offset: bnOffset = block and N offset"))
            module.add(staticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                "3. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))
            # unroll offset
            module.add(vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgprRes, \
                "4. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"])))
            module.add(staticMultiply(vgpr(kReg), vgpr(kReg), 1, tmpSgprInfo, \
                "4. K offset: lrKOffset = kIdx * mStride(1)"))

            module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(kReg), src1=vgpr(tReg), \
                comment="5. offset in wave: lrOffset = bnOffset + lrKOffset"))
            # wave offset
            if num1DWaves > 1:
            # FIXME: Should be two cases tile01 == 0 and tile01 == 1
                module.add(vectorStaticDivide(wReg, "Serial", dividedForWaveId, tmpVgprRes, \
                    "6. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                # Here we calculate the coordinate of the block offset, and remove the duplicated wave.
                if tile01 == 0:
                    module.add(vectorStaticDivide(dummy, wReg, num1DWaves, tmpVgprRes, \
                    "6-1.mask duplicated: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                else:
                    module.add(vectorStaticDivide(dummy, "Serial", waveWidth, tmpVgprRes, \
                        "6-1. wave offset in N dimen (waveWidth): wtid = tid / waveWidth(%u)" % waveWidth))
                    module.add(vectorStaticRemainder(dummy, dummy, wReg, kernel["MIWaveGroup"][0], tmpVgprRes, tmpSgprInfo, \
                        "6-1. wave offset in M dimen (MIWaveGroup0): wtid0 = wtid % MIWaveGroup0(%u)" % kernel["MIWaveGroup"][0]))
                module.add(VCmpXEqU32(dst=EXEC(), src0=vgpr(dummy), src1=0, comment="6-1. True if ans = 0"))
                module.add(vectorStaticRemainder(dummy, wReg, wReg, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "6. wave offset in M dimen: wtid0 = wtid %% num1DWaves(%u)" % num1DWaves))
                module.add(staticMultiply(vgpr(wReg), vgpr(wReg), strideWave, tmpSgprInfo, \
                    "6. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u)" % strideWave))
                module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(wReg), src1=vgpr(tReg), \
                    comment="7. final local read offset: flrOffset = lrOffset + WOffset"))
            module.add(VLShiftLeftB32(dst=vgpr(tReg), src=vgpr(tReg), shiftHex=hex(log2(diasBpe)), \
                comment="offset = offset*bpe" ))
        # release register
        writer.vgprPool.checkIn(wReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(ldsVgpr)
        writer.vgprPool.checkIn(ldsVgpr1)
        writer.vgprPool.checkIn(dummy)
        MIWaveGroupShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * vectorWidth, \
                            kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * 1]
        numReadPerTileVector = vectorWidth if (tile01 == 0) else 1
        numVectorsPerTile    = kernel["MIWaveTile"][tile01] // numReadPerTileVector
        idx = 0
        for vIdx in range(0, numVectorsPerTile):
            for eIdx in range(0, numReadPerTileVector):
                # normal case
                offset_val = (eIdx + vIdx * MIWaveGroupShape[tile01]) * totalVgprToBeStoredInK * kernel["ProblemType"]["ComputeDataType"].numBytes()
                bps = kernel["ProblemType"]["ComputeDataType"].numBytes()
                ds  = DSModifiers(offset=offset_val)
                dst = vgpr(tReg)
                vgprStr = "ValuSum+%u"%idx
                if bps==2:
                    module.add(DSStoreB16(dstAddr=dst, src=vgpr(vgprStr), ds=ds, comment="local store bias"))
                elif bps==4:
                    module.add(DSStoreB32(dstAddr=dst, src=vgpr(vgprStr), ds=ds, comment="local store bias"))
                elif bps==8:
                    module.add(DSStoreB64(dstAddr=dst, src=vgpr(vgprStr, 2), ds=ds, comment="local store bias"))
                else:
                    assert 0
                idx += 1
        # tReg
        writer.vgprPool.checkIn(tReg)
        module.add(SSetMask(dst=EXEC(), src=-1, comment="reset mask" ))
        # For later write in Global Write Batch
        return module
