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

from ..TensileInstructions import Module, VAddU32, staticMultiply, vectorStaticDivide, \
                                vectorStaticRemainder, RegisterPoolResource, vgpr
from ..Component import LraTileAssignment, LraTileProperties
from dataclasses import dataclass

@dataclass
class LraTilePropertiesMFMA(LraTileProperties):
   dividendForKId: int
   num1DBlocks: int
   num1DWaves: int
   dividedForBlkId: int
   dividedForWaveId: int
   vectorWidth: int
   maxKId: int

class LraTileAssignmentMFMA(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read Addresses: Tile Assignment A/B
    """
    def __call__(self, writer, kernel, tP):
        module = Module("LraTileAssignmentMFMA")
        module.addComment0("lr%s" % tP["tileChar"])

        # alloc vgpr
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
        ldsVgpr = writer.vgprPool.checkOut(1,"ldsVgpr")
        ldsVgpr1 = writer.vgprPool.checkOut(1,"ldsVgpr1")
        dummy   = writer.vgprPool.checkOut(1,"dummy")

        isMfma = writer.states.asmCaps["HasMFMA"]

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.states.kernel["WavefrontSize"]
        inputPerThread   = kernel["LocalReadVectorWidth"] if not writer.states.inTailLoop else kernel["MIInputPerThread%s"%tc]
        if kernel["ProblemType"]["SparseA"]:
          inputPerThread = inputPerThread // (2 if tP["isA"] else 8 if tP["isM"] else 1)
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = kernel["MatrixInstM"] if (tile01 == 0) else (kernel["MatrixInstM"] * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth%s"%tc]
        maxKId = waveWidth // ((kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]) * kernel["MatrixInstB"])
        writer.states.lraTileProperties[tile01] = LraTilePropertiesMFMA(dividendForKId=dividendForKId, \
                                                                        num1DBlocks=num1DBlocks, \
                                                                        num1DWaves=num1DWaves, \
                                                                        dividedForBlkId=dividedForBlkId, \
                                                                        dividedForWaveId = dividedForWaveId, \
                                                                        vectorWidth=vectorWidth, \
                                                                        maxKId=maxKId)

        # strider for each type of index
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        mt               = kernel["MacroTile%u" % tile01]
        strideTile       = kernel["_DepthU%s"%tc] + LdsPad if umlds else 1
        strideK          = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        strideBlock      = kernel["MatrixInstM"] * strideTile
        strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            # tile offset
            module.add(vectorStaticRemainder(dummy, kReg, "Serial", waveWidth, tmpVgprRes, tmpSgprInfo, \
                "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
            module.add(vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgprRes, tmpSgprInfo, \
                "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"]))
            module.add(staticMultiply(vgpr(tReg), vgpr(tReg), strideTile, tmpSgprInfo, \
                "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile))
            # block offset
            module.add(vectorStaticDivide(kReg, kReg, dividedForBlkId, tmpVgprRes, \
                "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId))
            module.add(vectorStaticRemainder(dummy, kReg, kReg, num1DBlocks, tmpVgprRes, tmpSgprInfo, \
                "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks))
            module.add(staticMultiply(vgpr(kReg), vgpr(kReg), strideBlock, tmpSgprInfo, \
                "2. block offset: bnOffset = bnIdx * strideBlock(%u)" % strideBlock))
            module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(kReg), src1=vgpr(tReg), \
                comment="3. add N and block offset: bnOffset = block and N offset"))
            module.add(staticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, tmpSgprInfo, \
                "4. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth))

            # unroll offset
            if isMfma and (dividendForKId != waveWidth):
                module.add(vectorStaticRemainder(dummy, kReg, "Serial", waveWidth, tmpVgprRes, tmpSgprInfo, \
                "5. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth))
                module.add(vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgprRes, \
                "5. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"])))
                module.add(staticMultiply(vgpr(kReg), vgpr(kReg), strideK, tmpSgprInfo, \
                "5. K offset: lrKOffset = kIdx * mStride(%u)" % strideK))
                module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(kReg), src1=vgpr(tReg), \
                comment="6. offset in wave: lrOffset = bnOffset + lrKOffset"))

            # wave offset
            if num1DWaves > 1:
                module.add(vectorStaticDivide(kReg, "Serial", dividedForWaveId, tmpVgprRes, \
                    "7. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId))
                module.add(vectorStaticRemainder(dummy, kReg, kReg, num1DWaves, tmpVgprRes, tmpSgprInfo, \
                    "7. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves))
                module.add(staticMultiply(vgpr(kReg), vgpr(kReg), strideWave, tmpSgprInfo, \
                    "7. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u)" % strideWave))
                module.add(VAddU32(dst=vgpr(tReg), src0=vgpr(kReg), src1=vgpr(tReg), \
                    comment="7. final local read offset: flrOffset = lrOffset + WOffset"))

        # release register
        tP["gpr"]["lro"] = tReg
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(ldsVgpr)
        writer.vgprPool.checkIn(ldsVgpr1)
        writer.vgprPool.checkIn(dummy)

        return module
