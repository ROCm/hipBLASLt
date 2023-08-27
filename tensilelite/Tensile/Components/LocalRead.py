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

from ..Component import LocalRead
from ..TensileInstructions import Module, DSModifiers, vgpr, sgpr, \
                            SMovB32, SWaitCnt, VOrB32, VPermB32, VLShiftLeftOrB32
from math import ceil

class LocalReadMFMA(LocalRead):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read: Do It A/B
    iui = Inner Unroll Idx
    epsi = expand pointer swap index. Only used for PAP
    """
    def __call__(self, writer, kernel, bufferIdx, iui, epsi, tP):
        imod = Module("LocalReadDo%s_I%s" % (tP["tensorChar"],iui))

        tc               = tP["tensorChar"]
        if tc == "A":
            writer.states.localReadDoCntA += 1
        elif tc == "Metadata":
            writer.states.localReadDoCntMetadata += 1
        else:
            writer.states.localReadDoCntB += 1
        tile01           = tP["tile01Idx"]
        instruction      = tP["localReadInstruction"]

        numOffsets       = instruction.numOffsets
        blockWidth       = instruction.blockWidth
        unrollBlockWidth = instruction.blockWidth if kernel["UnrollMajorLDS%s"%tc] else tP["bpe"]/4
        tileBlockWidth   = tP["bpe"]/4 if kernel["UnrollMajorLDS%s"%tc] else instruction.blockWidth

        vectorWidth  = kernel["VectorWidth%s"%tc]

        MIWaveGroupShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * kernel["VectorWidthA"], \
                            kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * kernel["VectorWidthB"]]

        LdsPad           = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
        tileStride       = 1
        UnrollStride     = kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad
        if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            tileStride   = kernel["_DepthU%s"%tc] + LdsPad
            UnrollStride = 1

        numVectorsPerTile = kernel["MIWaveTile"][tile01] // vectorWidth
        numReadsPerVector = (vectorWidth * tP["bpe"]) // int(tileBlockWidth * 4) # bytes/register
        # overloading numReadsPerUnroll for DirectToLds x2/x4 case when blockWidth of instruction < LocalReadVectorWidth
        # fp64 TLU=1 reading 0.5element/lane/read..
        # for TLU=0 case, blockWidth and LRVW should match
        numReadsPerUnroll = ceil(tP["bpe"] * kernel["MIInputPerThread%s"%tc] / int(unrollBlockWidth * 4)) # bytes/register
        numVgpr  = int(ceil(blockWidth))

        # pack register
        if writer.states.archCaps["HasEccHalf"]:
            needPack = tP["bpe"] < 4 and not kernel["UnrollMajorLDS%s"%tc] and not tP["isM"]
        else:
            needPack = blockWidth == 0.25
        pack     = Module("pack%s_I%s"%(tc,iui))

        valufIdx = 0
        for vIdx in range(0, numVectorsPerTile):
            for eIdx in range(0, numReadsPerVector):
                valuiIdx = int(valufIdx)
                localReadCode = imod.add(Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
                if needPack:
                    packCode = pack.add(Module("packCode"))
                for rIdx in range(0, numReadsPerUnroll):
                    valuiIdx = int(valufIdx)
                    baseLRVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx), numVgpr)
                    destVgpr = baseLRVgpr

                    if (writer.states.lrvwTileA > 1 and tc == 'A') or (writer.states.lrvwTileB > 1 and tc == 'B'):
                        highBitsForHalf = 0
                        isHigh8Bits = 0
                        isHigh16Bits = 0
                        if needPack:
                            destVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%(kernel["MIInputPerThread%s"%tc]), vIdx*numVgpr), numVgpr)
                        if rIdx == numReadsPerUnroll-1:
                            for i in range(0, numVgpr):
                                # convert from [tile][MiInputPerThread][vector] to [tile][vector][MiInputPerThread]
                                vgprIdx = (vIdx*numVgpr+i)*tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr*min(writer.states.bpr//tP["bpe"],vectorWidth)
                                if kernel["ProblemType"]["DataType"].isHalf() or kernel["MFMA_BF16_1K"] or kernel["ProblemType"]["DataType"].isBFloat16():
                                    vgprOffset = 0
                                    for vectorIdx in range(0, 2):
                                        for elementIdx in range(0, tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr):
                                            packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2+1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*2, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                  comment="select K=%u%u for vector=%u"%(elementIdx*2,  elementIdx*2+1, vectorIdx)))
                                            vgprOffset += 1
                                elif kernel["ProblemType"]["DataType"].isInt8() or kernel["ProblemType"]["DataType"].is8bitFloat():
                                    vgprOffset = 0
                                    # vertorIdx 2,3 is for the case vectorWidth > 2
                                    for vectorIdx in range(0, 4):
                                        if vectorWidth <= 2 and vectorIdx > 1:
                                            break
                                        for elementIdx in range(0, tP["bpe"]*kernel["MIInputPerThread%s"%tc]//writer.states.bpr):
                                            packCode.add(VPermB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx+vgprOffset)), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*4+1, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*4, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                  comment="select K=%u%u for vector=%u"%(elementIdx*4,  elementIdx*4+1, vectorIdx)))
                                            packCode.add(VPermB32(dst=vgpr("PackTemp"), src0=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*4+3, i+vIdx*numVgpr)), src1=vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, elementIdx*4+2, i+vIdx*numVgpr)), src2=sgpr("PackKForV%u"%vectorIdx), \
                                                                  comment="select K=%u%u for vector=%u"%(elementIdx*4+2,  elementIdx*4+3, vectorIdx)))
                                            packCode.add(VLShiftLeftOrB32(dst=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + vgprOffset)), src0=vgpr("PackTemp"), shiftHex=16, src1=vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, vgprIdx + vgprOffset)), comment="pack two half Vgpr to one Vgpr"))
                                            vgprOffset += 1

                    else:
                        # pack for blockWidth 0.5 type
                        highBitsForHalf = (blockWidth == 0.5) and ((rIdx % 2) == 1) # rIdx = 1,3
                        if writer.states.archCaps["HasEccHalf"]: # ECC pack
                            if needPack and highBitsForHalf:
                                highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%2, valuiIdx), numVgpr)
                                packCode.add(VOrB32(dst=baseLRVgpr, src0=baseLRVgpr, src1=highVgpr, comment="pack two half Vgpr to one Vgpr"))
                                destVgpr = highVgpr
                            # pack for blockWidth 0.25 type
                            isHigh8Bits  = (blockWidth == 0.25) and ( ((rIdx % 4) % 2) == 1) # rIdx = 1,3
                            isHigh16Bits = (blockWidth == 0.25) and ( ((rIdx % 4) //2) == 1) # rIdx = 2,3
                            if needPack and rIdx != 0:
                                if isHigh8Bits or isHigh16Bits:
                                    highVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, rIdx%4, valuiIdx), numVgpr)
                                    destVgpr = highVgpr
                                if isHigh8Bits:
                                    lowVgpr = vgpr("Valu%s_X%u_I%u_D%u+%u"%(tc, bufferIdx, iui, (rIdx%4)-1, valuiIdx), numVgpr) if isHigh16Bits else baseLRVgpr
                                    packCode.add(VLShiftLeftOrB32(dst=lowVgpr, src0=highVgpr, shiftHex=8, src1=lowVgpr, comment="pack two int8 Vgpr to one half Vgpr"))
                                    if isHigh16Bits:
                                        packCode.add(VOrB32(dst=baseLRVgpr, src0=baseLRVgpr, src1=lowVgpr, comment="pack two half Vgpr to one Vgpr"))
                        else: # no ECC pack
                        # pack for No ECC blockwidth 0.25 type
                            isHigh8Bits  = (blockWidth == 0.25) and ( ((rIdx % 4) % 2) == 1) # 1,3
                            isHigh16Bits = (blockWidth == 0.25) and ( ((rIdx % 4) //2) == 1) # 2,3
                            if needPack and rIdx != 0:
                                if isHigh8Bits:
                                    highVgpr = vgpr(tmpVgprIdx)
                                    destVgpr = highVgpr
                                if isHigh8Bits and isHigh16Bits:
                                    packCode.add(VLShiftLeftOrB32(dst=baseLRVgpr, src0=highVgpr, shiftHex=hex(0x8), src1=baseLRVgpr, comment="pack two int8x2 Vgpr to one Vgpr"))
                                    tmpVgprIdx += 1

                    valufIdx += blockWidth if not tP["isM"] else 1

                    # load read instrution
                    paramList = []

                    for oIdx in range(0, numOffsets):
                        if (kernel["DirectToLds%s" % tP["tensorChar"]] and  \
                            kernel["GlobalReadVectorWidth%c"%tc] * tP["bpe"] > 4):
                          # directToLds special case
                          divVal = 4 if kernel["ProblemType"]["DataType"].isDoubleComplex() else 2
                          rIdxMod = rIdx % divVal
                          rIdxDiv = rIdx // divVal
                          offset_val = (eIdx + (vIdx * numOffsets+oIdx) * MIWaveGroupShape[tile01]) * tileStride
                          offset_val = (rIdxDiv * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpe"]  + rIdxMod * writer.states.bpr
                        else:
                          # normal case
                          offset_val = (eIdx + (vIdx * numOffsets+oIdx) * MIWaveGroupShape[tile01]) * tileStride
                          offset_val = (rIdx * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpe"]
                        if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                            offset_val = offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
                        offset_val = offset_val + tP["localReadSwapByteOffset"]
                        if (kernel["DirectToLds%s" % tc] and  \
                            kernel["GlobalReadVectorWidth%c"%tc] * tP["bpe"] > 4):

                          # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                          dummy, offset_val = writer.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val)

                          # offset conversion for DirectToLds
                          # TLU=0 case, modify bit3-6 of offset_val as follows
                          # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
                          bit2 = offset_val & 4
                          bit3 = offset_val & 8
                          bit4 = offset_val & 16
                          bit5 = offset_val & 32
                          if (kernel["GlobalReadVectorWidth%s"%tc] * tP["bpe"] == 8):
                            # dword_x2 case
                            # (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
                            newVal = (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
                          else:  #if (kernel["GlobalReadVectorWidth%s"%tc] * tP["bpe"] == 16):  # most preferred case
                            # dword_x4 case
                            # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
                            newVal = (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
                          offset_val = offset_val & (~0x3c)
                          offset_val = offset_val | newVal

                        paramList.append(int(offset_val))

                    comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u eIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u" \
                            % (tP["localReadOffset"], tP["localReadSwapByteOffset"], MIWaveGroupShape[tile01], vIdx, eIdx, rIdx, oIdx, bufferIdx, iui)

                    highBits = highBitsForHalf or isHigh16Bits

                    if numOffsets == 1:
                        ds = DSModifiers(na=1, offset=paramList[0])
                    else:
                        ds = DSModifiers(na=2, offset0=paramList[0], offset1=paramList[1])
                    LocalReadX = instruction.getInst(highBits)
                    localReadCode.add(LocalReadX(dst=destVgpr, src=vgpr("LocalReadAddr%s"%tc), ds=ds, comment=comment))

                    # TODO - handle vector-load
                    with writer.allocTmpSgpr(1) as tmpSgprInfo:
                        tmpSgpr = tmpSgprInfo.idx
                        if writer.db["CheckValue1%s"%tc] and not writer.inTailLoop:

                            dbgVgpr = destVgpr
                            dbgVgprList = destVgpr.split("v[")
                            if len(dbgVgprList) == 1: # vIdx, no []
                                dbgVgpr = dbgVgprList[0]
                            else:
                                # We only check the first one now
                                # TODO: Handle vector, but need to take care the last one
                                dbgVgprList = (dbgVgprList[1].split("]")[0]).split(':')
                                dbgVgpr = "v[%s]"%dbgVgprList[0]

                            localReadCode.add(SWaitCnt(lgkmcnt=0, vscnt=0, comment="CheckValue1 wait for LDS read"))

                            if kernel["ProblemType"]["DataType"].isHalf():
                                hexValue = hex(0x3c003c00)     # packed 1s
                                if needPack:
                                    hexValue = hex(0x3c000000) if highBitsForHalf else hex(0x00003c00)
                                localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: FP16"))
                                localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                            elif kernel["ProblemType"]["DataType"].isBFloat16():
                                hexValue = hex(0x3f803f80)     # packed 1s
                                if needPack:
                                    hexValue = hex(0x3f800000) if highBitsForHalf else hex(0x00003f80)
                                localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: BF16"))
                                localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                            if kernel["ProblemType"]["DataType"].isInt8():
                                if needPack:
                                    hexValue = hex(0x00010000) if isHigh16Bits else hex(0x00000001)
                                    localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hexValue, comment="CheckValue1: INT8"))
                                    localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                            # TODO - Check if this works. But need this? MFMA would use INT8
                            elif kernel["ProblemType"]["DataType"].isInt8x4():
                                localReadCode.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(0x01010101), comment="CheckValue1: INT8x4"))
                                localReadCode.add(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                            elif kernel["ProblemType"]["DataType"].isSingle():
                                localReadCode.add(writer.assert_eq( dbgVgpr, 1.0) )

        return imod, pack
