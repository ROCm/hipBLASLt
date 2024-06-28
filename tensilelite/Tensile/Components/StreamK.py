################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from ..TensileInstructions import Module, Label, SAddU32, RegisterPoolResource, sgpr, scalarStaticDivideAndRemainder, \
    SCmpLtU32, SCSelectB32, sMagicDivAlg2, SMulI32, SSubU32, SMinU32, SMovB32, SCBranchSCC1, SCmpLeU32, VMovB32, vgpr, \
    SAddCU32, SCmpGtU32, SCMovB32, SAddI32, SCmpEQU32, SCBranchSCC0, SLShiftLeftB32, SLoadB32, SWaitCnt, SMEMModifiers, \
    log2
# from ..TensileInstructions.Containers import SMEMModifiers
from ..Component import Component
import abc

class XCCMapping(Component):
    """
    XCC mapping code.
    """
    pass

class XCCMappingOff(XCCMapping):
    kernel = {"StreamKXCCMapping": 0}

    def __call__(self, writer, kernel):
        module = Module("XCCMapping Off")
        return module

class XCCMappingOn(XCCMapping):

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["StreamKXCCMapping"] > 0
    
    def __call__(self, writer, kernel):
        module = Module("XCCMapping On")

        with writer.allocTmpSgpr(4) as tmpSgprRes:
            sXCC   = tmpSgprRes.idx
            sGridC = tmpSgprRes.idx + 1
            sGridF = tmpSgprRes.idx + 2
            sGridM = tmpSgprRes.idx + 3
            sTmp = None
            sTmpRes = None
            sqTmp = None
            divisor = kernel["StreamKXCCMapping"]
            if ((divisor & (divisor - 1)) != 0): # Need temp registers if not power of 2
                sTmp = writer.sgprPool.checkOut(2, "sTmp", preventOverflow=False)
                sTmpRes  = RegisterPoolResource(idx=sTmp, size=2)
                sqTmp = writer.sgprPool.checkOut(1, "sqTmp", preventOverflow=False)

            # sGridC = ceil(grid / xccm)
            module.add(SAddU32(dst=sgpr(sXCC), src0=sgpr("skGrid"), src1=hex(kernel["StreamKXCCMapping"] - 1), comment="ceil(grid/xccm)"))
            module.add(scalarStaticDivideAndRemainder(qReg=sGridC, rReg=None, dReg=sGridC, divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=0))
            # sGridF = floor(grid / xccm)
            # sGridM = grid % xccm
            module.add(scalarStaticDivideAndRemainder(qReg=sGridF, rReg=sGridM, dReg="skGrid", divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes))
            # sXCC = wg0 % xccm
            # sqtmp is temp register for quotient for non-power-of-2 case
            # sqtmp overlaps temp registers, works in this case and output is discarded
            module.add(scalarStaticDivideAndRemainder(qReg=sqTmp, rReg=sXCC, dReg="WorkGroup0", divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=2))
            # Check if current XCC requires a remainder WG or not
            module.add(SCmpLtU32(src0=sgpr(sXCC), src1=sgpr(sGridM), comment="XCCM < Remainder"))
            module.add(SCSelectB32(dst=sgpr(sGridC), src0=sgpr(sGridC), src1=sgpr(sGridF), comment="Select multiplier"))
            module.add(SCSelectB32(dst=sgpr(sGridM), src0=0, src1=sgpr(sGridM), comment="Select remainder"))
            # WG = floor(wg0 / xccm) * xccm + XCCoffset + optional remainder
            module.add(scalarStaticDivideAndRemainder(qReg="WorkGroup0", rReg=None, dReg="WorkGroup0", divisor=kernel["StreamKXCCMapping"], tmpSgprRes=sTmpRes, doRemainder=0))
            module.add(SMulI32(dst=sgpr(sXCC), src0=sgpr(sXCC), src1=sgpr(sGridC), comment="XCC group id"))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sXCC), comment="Add XCC group offset"))
            module.add(SAddU32(dst=sgpr("WorkGroup0"), src0=sgpr("WorkGroup0"), src1=sgpr(sGridM), comment="Add remainder offset"))
            if sTmp is not None:
                writer.sgprPool.checkIn(sTmp)
                writer.sgprPool.checkIn(sqTmp)

        return module


class StreamK(Component):
    """
    StreamK code.
    """
    def __call__(self):
        assert(0)

    @abc.abstractmethod
    def preLoop(self, writer, kernel):
        pass
        
    @abc.abstractmethod
    def graWorkGroup(self, writer, kernel, tPA, tPB):
        pass
        
    def skTileIndex(self, writer, kernel, sTmp, tPA, tPB):
        module = Module("StreamK skTileIndex")
    
        # Always reset pointers to handle odd-exit case which moves LRO to the upper bank
        if kernel["PrefetchGlobalRead"]: # not self.prefetchAcrossPersistent
            module.add(writer.localReadResetOffsets(kernel, tPA))
            module.add(writer.localReadResetOffsets(kernel, tPB))

        module.addComment0("StreamK calculate tile idx and map to WG")

        # sTmp = tile index
        module.add(sMagicDivAlg2(sTmp, sgpr("StreamKIter"), sgpr("MagicNumberItersPerTile"), sgpr("MagicShiftItersPerTile")))
        # sTmp+1 = tile start
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sTmp), src1=sgpr("ItersPerTile"), comment="Tile start iteration"))
        # sTmp+2 = tile end
        module.add(SAddU32(dst=sgpr(sTmp+2), src0=sgpr(sTmp+1), src1=sgpr("ItersPerTile"), comment="Tile end iteration"))
        # local start
        module.add(SSubU32(dst=sgpr("StreamKLocalStart"), src0=sgpr("StreamKIter"), src1=sgpr(sTmp+1), comment="Local iteration start"))
        # local end (SK tile)
        module.add(SMinU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp+2), comment="1. (Local) iteration end (SK tile)"))
        module.add(SSubU32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalEnd"), src1=sgpr(sTmp+1), comment="2. Local iteration end (SK tile)"))

        return module
        
    def skIndexToWG(self, writer, kernel, sTmp):
        module = Module("StreamK skIndexToWG")

        # Map StreamK tile index to wg0/1
        module.addComment0("Map StreamK tile index to wg0/1/2")
        module.add(sMagicDivAlg2(sTmp+1, sgpr(sTmp), sgpr("MagicNumProblemNumGroupTiles0By1"), sgpr("MagicShiftProblemNumGroupTiles0By1")))
        module.add(SMovB32(dst=sgpr("WorkGroup2"), src=sgpr(sTmp+1), comment="wg2 = Tile Idx / problemNumGroupTiles0By1"))
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("NumWorkGroups0"), comment="remainder part 1 : quotient * divisor"))
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("NumWorkGroups1"), comment="remainder part 1 : quotient * divisor"))
        module.add(SSubU32(dst=sgpr(sTmp), src0=sgpr(sTmp), src1=sgpr(sTmp+1), comment="remainder"))
        module.add(sMagicDivAlg2(sTmp+1, sgpr(sTmp), sgpr("MagicNumberProblemNumGroupTiles0"), sgpr("MagicShiftProblemNumGroupTiles0")))
        module.add(SMovB32(dst=sgpr("WorkGroup1"), src=sgpr(sTmp+1), comment="wg1 = Tile Idx / problemNumGroupTiles0"))
        module.add(SMulI32(dst=sgpr("WorkGroup0"), src0=sgpr(sTmp+1), src1=sgpr("NumWorkGroups0"), comment="remainder part 1 : quotient * divisor"))
        module.add(SSubU32(dst=sgpr("WorkGroup0"), src0=sgpr(sTmp), src1=sgpr("WorkGroup0"), comment="wg0 = Tile Idx % problemNumGroupTiles0"))
        module.addSpaceLine()

        return module
    
    @abc.abstractmethod
    def computeLoadSrd(self, writer, kernel, tc, sTmp):
        pass

    def computeLoadSrdCommon(self, writer, kernel, tc, sTmp):
        module = Module("StreamK Common computeLoadSrd")

        tileStart = sTmp + 2
        # StreamK partial tile - offset to tile start index
        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr("StreamKLocalStart"), src1="DepthU", comment="StreamK tile start offset"))
        strideL = writer.strideRef(tc, kernel["ProblemType"]["IndicesSummation"][0])
        module.add(writer.s_mul_u64_u32(sgpr(sTmp), sgpr(sTmp+1), sgpr(sTmp), strideL, "StreamK tile start offset"))
        # Overflow check removed
        # if kernel["CheckDimOverflow"] >=2:
        #     kStr += self.assert_eq(sgpr(sTmp+1),0)
        module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(sTmp+0), comment="accum GsuOffset term to tilestart"))
        module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(sTmp+1), comment="accum GsuOffset term to tilestart"))

        return module

    @abc.abstractmethod
    def graAddresses(self, writer, kernel, tP, vTmp):
        pass

    def graAddressesCommon(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Common graAddresses")
        
        tc = tP["tensorChar"]
        # StreamK partial tile - offset to tile start index
        tmpOffset = writer.sgprPool.checkOut(2, "skStartOffset", preventOverflow=0)
        module.add(SMulI32(dst=sgpr(tmpOffset), src0=sgpr("StreamKLocalStart"), src1="DepthU*%d" % (tP["bpe"]), comment="StreamK tile start offset"))
        strideL = writer.strideRef(tc, kernel["ProblemType"]["IndicesSummation"][0])
        module.add(writer.s_mul_u64_u32(sgpr(tmpOffset), sgpr(tmpOffset+1), sgpr(tmpOffset), strideL, "StreamK tile start offset"))
        # Overflow check removed
        # if kernel["CheckDimOverflow"] >=2:
        #     kStr += self.assert_eq(sgpr(tmpOffset+1),0)
        module.add(SAddU32(dst=sgpr(tmpOffset+0), src0=sgpr(tmpOffset+0), src1=sgpr("Address%s+0" % tc), comment="accum skOffset term to tilestart"))
        module.add(SAddCU32(dst=sgpr(tmpOffset+1), src0=sgpr(tmpOffset+1), src1=sgpr("Address%s+1" % tc), comment="accum skOffset term to tilestart"))
        module.add(VMovB32(dst=vgpr(vTmp+0), src=sgpr(tmpOffset+0)))
        module.add(VMovB32(dst=vgpr(vTmp+1), src=sgpr(tmpOffset+1)))
        writer.sgprPool.checkIn(tmpOffset)

        return module

    @abc.abstractmethod
    def declareStaggerParms(self, writer, kernel):
        pass

    def declareStaggerParmsCommon(self, writer, kernel):
        module = Module("StreamK Common declareStaggerParms")

        # Set stagger=0 for partial tiles to avoid using stagger larger than workload
        module.add(SCmpGtU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
        module.add(SCMovB32(dst=sgpr("StaggerUIter"), src=0, comment="set stagger=0 for partial tiles"))
        module.add(SCmpLtU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr("ItersPerTile"), comment="does wg finish tile?"))
        module.add(SCMovB32(dst=sgpr("StaggerUIter"), src=0, comment="set stagger=0 for partial tiles"))

        return module
    
    @abc.abstractmethod
    def tailLoopNumIter(self, writer, kernel, loopCounter):
        pass
    
    def tailLoopNumIterCommon(self, writer, kernel, loopCounter):
        module = Module("StreamK Common tailLoopNumIter")

        # skip tail loop if StreamK WG not processing final iteration
        # Check if tile finished
        module.add(SCmpLtU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr("ItersPerTile"), comment="Check if WG processes final iteration of tile"))
        module.add(SCMovB32(dst=loopCounter, src=hex(0), comment="This WG not completing tile"))

        return module
    
    @abc.abstractmethod
    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        pass

    def calculateLoopNumIterCommon(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Common calculateLoopNumIter")

        # Use StreamK params for loop count
        module.add(SSubU32(dst=sgpr(loopCounterName), src0=sgpr("StreamKLocalEnd"), src1=sgpr("StreamKLocalStart"), comment="StreamK loop counter = localEnd - localStart"))
        # Adjust loop count for tail loop
        if not kernel["NoTailLoop"]:
            tmpSgpr = tmpSgprInfo.idx
            unrollIdx = writer.states.unrollIdx
            loopChar = writer.states.indexChars[kernel["ProblemType"]["IndicesSummation"][unrollIdx]]

            assert kernel["DepthU"] % 2 == 0 # Assuming DepthU is power of 2, if odd DepthU were supported this divide would need 2 more temp registers for divide
            module.add(scalarStaticDivideAndRemainder(qReg=tmpSgpr, rReg=tmpSgpr+1, dReg=("SizesSum+%u" % unrollIdx), divisor=kernel["DepthU"], tmpSgprRes=None, doRemainder=2))
            module.add(SCmpEQU32(src0=sgpr(tmpSgpr+1), src1=hex(0), comment="numIter%s == 0"%loopChar ))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=0, src1=1, comment="check if size uses tail loop"))
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr("ItersPerTile"), comment="Check if WG processes final iteration of tile"))
            module.add(SCSelectB32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=0, comment="this WG runs tail loop"))
            module.add(SSubU32(dst=sgpr(loopCounterName), src0=sgpr(loopCounterName), src1=sgpr(tmpSgpr), comment="Adjust loop counter for tail loop"))

        return module
    
    @abc.abstractmethod
    def storeBranches(self, writer, kernel, skPartialsLabel):
        pass

    def storeBranchesCommon(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK Common storeBranches")

        # No branches for atomic mode
        if kernel["StreamKAtomic"]:
            return module
        
        skFixupLabel = Label(label=writer.labels.getNameInc("SK_Fixup"), comment="")
        skStoreLabel = Label(label=writer.labels.getNameInc("SK_Store"), comment="")

        # StreamK store branches
        tmpSgpr = self.sgprPool.checkOut(4, "globalWriteElements", preventOverflow=0)
        # if we did not start the tile, store partials
        # branch to beta == 0 store path
        module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
        module.add(SCBranchSCC0(labelName=skPartialsLabel.getLabelName(), comment="Branch if not start tile, store partials"))

        if kernel["DebugStreamK"] & 1 == 0:
            # if we started and finished the tile, regular store code
            # branch to regular store code, skip fixup step
            module.add(SCmpEQU32(src0=sgpr("StreamKLocalEnd"), src1=sgpr("ItersPerTile"), comment="does wg finish tile?"))
            module.add(SCBranchSCC1(labelName=skStoreLabel.getLabelName(), comment="Branch if started and finished tile, go to regular store code"))

            # if we started the tile but did not finish it, fix up step
            # run fixup code before regular store code
            sCtaIdx = writer.sgprPool.checkOut(1, "CtaIdx", preventOverflow=0) # self.defineSgpr("CtaIdx", 1)
            module.add(SAddU32(dst=sgpr(sCtaIdx), src0=sgpr("StreamKIdx"), src1=1, comment="input partial tile index"))

            sFixupEnd = writer.sgprPool.checkOut(1, "FixupEnd", preventOverflow=0) # self.defineSgpr("CtaEnd", 1)
            module.add(sMagicDivAlg2(tmpSgpr, sgpr("StreamKIterEnd"), sgpr("MagicNumberItersPerTile"), sgpr("MagicShiftItersPerTile")))
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=sgpr("ItersPerTile"), comment="start iteration of partial tile"))
            module.add(SSubU32(dst=sgpr(sFixupEnd), src0=sgpr("StreamKIterEnd"), src1=sgpr(tmpSgpr), comment="calc iterations completed by this WG"))

            module.add(skFixupLabel)

            # Check flag
            module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr(sCtaIdx), shiftHex=log2(4), comment="flag offset based on CTA index"))
            module.add(SLoadB32(dst=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=1), comment="get flag"))

            module.add(SWaitCnt(lgkmcnt=0, comment="wait for flag load"))
            if kernel["DebugStreamK"] & 2 == 0:
                module.add(SCmpEQU32(src0=sgpr(tmpSgpr+2), src1=1, comment="check if ready"))
                module.add(SCBranchSCC0(labelName=skFixupLabel.getLabelName(), comment="if flag not set, wait and check again"))

            writer.sgprPool.checkIn(tmpSgpr)

            # TODO FIXUP STEP!!!!!!!!!!!!!!!
            fixupEdge = [False] # Temporary hack to test no edge variant
            kStr += self.fixupStep(kernel, vectorWidths, elements, fixupEdge, tmpVgpr, tmpCVTVgpr, sCtaIdx, skStoreLabel)
            
            if kernel["StreamK"] >= 2:
                sIterCount = writer.sgprPool.checkOut(1, "iterCount", preventOverflow=0)
                module.add(SAddU32(dst=sgpr(sIterCount), src0=sgpr("SKItersPerWG"), src1=1, comment="Add extra iter"))
                module.add(SCmpLtU32(src0=sgpr(sCtaIdx), src1=sgpr("skExtraIters"), comment="Check if next WG had an extra iteration"))
                module.add(SCSelectB32(dst=sgpr(sIterCount), src0=sgpr(sIterCount), src1=sgpr("SKItersPerWG"), comment="Select correct number of iterations for next WG"))
                module.add(SAddU32(dst=sgpr(sFixupEnd), src0=sgpr(sFixupEnd), src1=sgpr(sIterCount), comment="next partial tile iteration"))
                writer.sgprPool.checkIn(sIterCount)
            module.add(SAddU32(dst=sgpr(sCtaIdx), src0=sgpr(sCtaIdx), src1=1, comment="next partial tile index"))
            if kernel["StreamK"] == 1:
                module.add(SAddU32(dst=sgpr(sFixupEnd), src0=sgpr(sFixupEnd), src1=sgpr("SKItersPerWG"), comment="next partial tile iteration"))
            module.add(SCmpLtU32(src0=sgpr(sFixupEnd), src1=sgpr("ItersPerTile"), comment="done loading partial tiles?"))
            module.add(SCBranchSCC1(labelName=skFixupLabel.getLabelName(), comment="Branch to continue fixup loop"))
            
            writer.sgprPool.checkIn(sFixupEnd)
            writer.sgprPool.checkIn(sCtaIdx)

        module.add(skStoreLabel)

        return module
    
    @abc.abstractmethod
    def writePartials(self, writer, kernel, skPartialsLabel):
        pass

    def writePartialsCommon(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK Common writePartials")

        # No partials for atomic mode
        if kernel["StreamKAtomic"]:
            return module
        
        module.add(skPartialsLabel.getLabelName())
        if kernel["DebugStreamK"] & 2 == 0:
            fixupEdge = [False] # Temporary hack to test no edge variant
            kStr += self.writePartials(kernel, vectorWidths, elements, fixupEdge, atomic, tmpVgpr, tmpCVTVgpr, isOptNLL, endLabel)
            
        return module

class StreamKOff(StreamK):
    kernel = {"StreamK": 0}

    def preLoop(self, writer, kernel):
        module = Module("StreamK Off openLoop")
        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK Off graWorkGroup")
        return module

    def computeLoadSrd(self, writer, kernel, tc, sTmp):
        module = Module("StreamK Off computeLoadSrd")
        return module
    
    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Off graAddresses")

        tc = tP["tensorChar"]
        module.add(VMovB32(dst=vgpr(vTmp+0), src=sgpr("Address%s+0" % tc)))
        module.add(VMovB32(dst=vgpr(vTmp+1), src=sgpr("Address%s+1" % tc)))
        
        return module
    
    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK Off declareStaggerParms")
        return module
    
    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK Off tailLoopNumIter")
        return module
    
    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Off calculateLoopNumIter")

        quotient = loopCounterName
        dividend = "SizesSum+%u" % loopIdx #sumSize = self.sumSize(kernel, loopIdx)
        divisor = kernel["DepthU"]

        if kernel["NoTailLoop"] and kernel["AssertSummationElementMultiple"] % kernel["DepthU"] != 0:
            # round up SizesSum/DepthU for noTailLoop case
            module.add(SAddI32(dst=sgpr(quotient), src0=(divisor - 1), src1=sgpr(dividend), comment="round up SizeSum / DepthU" ))
            module.add(scalarStaticDivideAndRemainder(qReg=quotient, rReg=None, dReg=quotient, divisor=divisor, tmpSgprRes=tmpSgprInfo, doRemainder=0))
        else:
            module.add(scalarStaticDivideAndRemainder(qReg=quotient, rReg=None, dReg=dividend, divisor=divisor, tmpSgprRes=tmpSgprInfo, doRemainder=0))

        return module
    
    def storeBranches(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK Off storeBranches")
        return module
    
    def writePartials(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK Off writePartials")
        return module

class StreamKBasic(StreamK):
    kernel = {"StreamK": 1}

    def preLoop(self, writer, kernel):
        module = Module("StreamK Basic openLoop")

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"), comment="Save original StreamK index"))
        # Basic SK
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"), src1=sgpr("SKItersPerWG"), comment="StreamK starting iteration"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr("SKItersPerWG"), comment="StreamK ending iteration"))
        module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr("TotalIters"), comment="Cap ending iter at total iters"))
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("StreamKIterEnd"), comment="Make sure there's work to do"))
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))
        module.add(writer.undefineSgpr("TotalIters"))

        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK Basic graWorkGroup")

        # StreamK workgroup mapping
        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=0)

        module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        # Increment StreamK iteration
        module.add(SMovB32(dst=sgpr("StreamKIter"), src=sgpr(sTmp+2), comment="Increment StreamK Iteration"))

        module.add(self.skIndexToWG(writer, kernel, sTmp))

        writer.sgprPool.checkIn(sTmp)

        return module

    def computeLoadSrd(self, writer, kernel, tc, sTmp):
        module = Module("StreamK Basic computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tc, sTmp))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK Basic graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module
    
    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK Basic declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK Basic tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK Basic calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module

    def storeBranches(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK Basic storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel))
        return module
    
    def writePartials(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK Basic writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel))
        return module

class StreamKTwoTileOriginal(StreamK):
    kernel = {"StreamK": 2}

    def preLoop(self, writer, kernel):
        module = Module("StreamK TwoTileOriginal openLoop")

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"), comment="Save original StreamK index"))
        # Two-tile SK (SK first)
        # iter count after all extra iters have been distributed
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"), src1=sgpr("SKItersPerWG"), comment="StreamK starting iteration (case: after extra iters)"))
        module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr("skExtraIters"), comment="Add extra iters"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr("SKItersPerWG"), comment="StreamK ending iteration (case: after extra iters)"))
        # iter count before all extra iters have been distributed
        # sTmp+1 = SKItersPerWG + 1 extra iteration
        sIter = writer.sgprPool.checkOut(2, "SKIter", preventOverflow=False)
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr("SKItersPerWG"), src1=1, comment="Spread out extra iterations"))
        module.add(SMulI32(dst=sgpr(sIter), src0=sgpr("StreamKIdx"), src1=sgpr(sIter+1), comment="StreamK starting iteration (case: before extra iters)"))
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1), comment="StreamK ending iteration (case: before extra iters)"))
        # select correct start/end iteration index
        module.add(SCmpLtU32(src0=sgpr("StreamKIdx"), src1=sgpr("skExtraIters"), comment="Check if lane gets an extra iteration"))
        module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter), src1=sgpr("StreamKIter"), comment="Set start iter"))
        module.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1), src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
        writer.sgprPool.checkIn(sIter)
        # clamp to end of sk iterations
        # TODO maybe remove clamp, since extra iters code should guarantee total iterations match
        sTmp = writer.sgprPool.checkOut(1, "TotalSKIters", preventOverflow=False)
        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr("skTiles"), src1=sgpr("ItersPerTile"), comment="Total SK iters"))
        module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp), comment="Cap ending iter at total SK iters"))
        writer.sgprPool.checkIn(sTmp)
        # check if this WG has no work to do
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("TotalIters"), comment="Make sure there's work to do"))
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK TwoTileOriginal graWorkGroup")

        # StreamK workgroup mapping
        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=0)

        module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        # local end (DP tile)
        # TODO This line isnt needed?
        module.add(SSubU32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+2), src1=sgpr(sTmp+1), comment="Local iteration end (DP tile)"))
        # select correct local end
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("StreamKIterEnd"), comment="Check if in SK or DP section"))
        module.add(SCSelectB32(dst=sgpr("StreamKLocalEnd"), src0=sgpr("StreamKLocalEnd"), src1=sgpr(sTmp+3), comment="Apply SK or DP end iteration"))

        # Increment StreamK iteration
        # If moving from SK to DP, next iteration is first DP
        # sTmp = offset to first DP tile
        module.add(SMulI32(dst=sgpr(sTmp+3), src0=sgpr("skTiles"), src1=sgpr("ItersPerTile"), comment="Offset to first DP tile"))
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr("StreamKIdx"), src1=sgpr("ItersPerTile"), comment="WG tile offset"))
        module.add(SAddU32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr(sTmp+1), comment="DP start offset + WG offset"))
        # If already in DP, add dpShift
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr("skGrid"), src1=sgpr("ItersPerTile"), comment="DP iterations shift"))
        module.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("StreamKIter"), comment="Add DP shift"))
        # Save DP iter in sTmp
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("StreamKIterEnd"), comment="Check if in SK or DP section"))
        module.add(SCSelectB32(dst=sgpr(sTmp+3), src0=sgpr(sTmp+3), src1=sgpr(sTmp+1), comment="Select first DP tile, or add DP shift"))
        # If staying in SK portion, next iteration is sTmp+2
        module.add(SCmpLtU32(src0=sgpr(sTmp+2), src1=sgpr("StreamKIterEnd"), comment="Check if there are more SK tiles"))
        module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sTmp+2), src1=sgpr(sTmp+3), comment="Select next SK or DP tile"))

        module.add(self.skIndexToWG(writer, kernel, sTmp))

        writer.sgprPool.checkIn(sTmp)

        return module

    def computeLoadSrd(self, writer, kernel, tc, sTmp):
        module = Module("StreamK TwoTileOriginal computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tc, sTmp))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK TwoTileOriginal graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module
    
    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK TwoTileOriginal declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK TwoTileOriginal tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK TwoTileOriginal calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module
        
    def storeBranches(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK TwoTileOriginal storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel))
        return module

    def writePartials(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK TwoTileOriginal writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel))
        return module


class StreamKTwoTileDPFirst(StreamK):
    kernel = {"StreamK": 3}

    def preLoop(self, writer, kernel):
        module = Module("StreamK TwoTileDPFirst openLoop")

        xccMapping = Component.XCCMapping.find(writer)
        module.add(xccMapping(writer, kernel))

        module.add(SMovB32(dst=sgpr("StreamKIdx"), src=sgpr("WorkGroup0"), comment="Save original StreamK index"))
        # Two-tile SK (DP first)
        # Do DP tiles before SK
        skInitDone = Label("SK_InitDone", "")
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"), src1=sgpr("ItersPerTile"), comment="DP starting iteration (case: DP work to do)"))
        module.add(SMovB32(dst=sgpr("StreamKIterEnd"), src=sgpr("TotalIters"), comment="DP ending iteration (case: only DP work to do)"))
        sTmp = writer.sgprPool.checkOut(1, "TotalSKIters", preventOverflow=False)
        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr("skTiles"), src1=sgpr("ItersPerTile"), comment="Total SK iters"))
        module.add(SCmpLtU32(src0=sgpr(sTmp), src1=sgpr("TotalIters"), comment="Check if there are DP tiles to do"))
        module.add(SCBranchSCC1(labelName=skInitDone.getLabelName(), comment="Done init"))
        writer.sgprPool.checkIn(sTmp)

        # If there are no DP tiles to do, regular SK init
        # iter count after all extra iters have been distributed
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"), src1=sgpr("SKItersPerWG"), comment="StreamK starting iteration (case: after extra iters)"))
        module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr("skExtraIters"), comment="Add extra iters"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr("SKItersPerWG"), comment="StreamK ending iteration (case: after extra iters)"))
        # iter count before all extra iters have been distributed
        # sTmp+1 = SKItersPerWG + 1 extra iteration
        sIter = writer.sgprPool.checkOut(2, "SKIter", preventOverflow=False)
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr("SKItersPerWG"), src1=1, comment="Spread out extra iterations"))
        module.add(SMulI32(dst=sgpr(sIter), src0=sgpr("StreamKIdx"), src1=sgpr(sIter+1), comment="StreamK starting iteration (case: before extra iters)"))
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1), comment="StreamK ending iteration (case: before extra iters)"))
        # select correct start/end iteration index
        module.add(SCmpLtU32(src0=sgpr("StreamKIdx"), src1=sgpr("skExtraIters"), comment="Check if lane gets an extra iteration"))
        module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter), src1=sgpr("StreamKIter"), comment="Set start iter"))
        module.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1), src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
        writer.sgprPool.checkIn(sIter)
        # clamp to end of sk iterations
        # TODO maybe remove clamp, since extra iters code should guarantee total iterations match
        sTmp = writer.sgprPool.checkOut(1, "TotalSKIters", preventOverflow=False)
        module.add(SMulI32(dst=sgpr(sTmp), src0=sgpr("skTiles"), src1=sgpr("ItersPerTile"), comment="Total SK iters"))
        module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp), comment="Cap ending iter at total SK iters"))
        writer.sgprPool.checkIn(sTmp)

        module.add(skInitDone)
        # check if this WG has no work to do
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("TotalIters"), comment="Make sure there's work to do"))
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1))

        return module

    def graWorkGroup(self, writer, kernel, tPA, tPB):
        module = Module("StreamK TwoTileDPFirst graWorkGroup")

        # StreamK workgroup mapping
        sTmp = writer.sgprPool.checkOutAligned(4, 2, "SKMappingTemp", preventOverflow=0)

        module.add(self.skTileIndex(writer, kernel, sTmp, tPA, tPB))

        skUpdateDone = Label("SK_UpdateDone", "")
        # sTmp+3 = Offset to first SK tile
        module.add(SMulI32(dst=sgpr(sTmp+3), src0=sgpr("skTiles"), src1=sgpr("ItersPerTile"), comment="Total SK iters"))
        module.add(SSubU32(dst=sgpr(sTmp+3), src0=sgpr("TotalIters"), src1=sgpr(sTmp+3), comment="Offset to first SK tile"))
        # If in DP, add dpShift
        module.add(SMulI32(dst=sgpr(sTmp+1), src0=sgpr("skGrid"), src1=sgpr("ItersPerTile"), comment="DP iterations shift"))
        module.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr(sTmp+1), src1=sgpr("StreamKIter"), comment="Add DP shift"))
        # if sTmp+1 < sTmp+3, continue DP (add dpShift)
        module.add(SCmpLtU32(src0=sgpr(sTmp+1), src1=sgpr(sTmp+3), comment="Check if still in DP section"))
        module.add(SCBranchSCC1(labelName=skUpdateDone.getLabelName(), comment="Done update"))
        # if StreamKIter >= sTmp+3, continue SK (add skShift?)
        module.add(SMovB32(dst=sgpr(sTmp+1), src=sgpr(sTmp+2), comment="SK iterations shift"))
        module.add(SCmpLeU32(src0=sgpr(sTmp+3), src1=sgpr("StreamKIter"), comment="Check if continuing in SK section"))
        module.add(SCBranchSCC1(labelName=skUpdateDone.getLabelName(), comment="Done update"))
        # if sTmp+1 > sTmp+3 and StreamKIter < sTmp+3, switch from DP to SK (add dpShift)
        # iter count after all extra iters have been distributed
        module.add(SMulI32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIdx"), src1=sgpr("SKItersPerWG"), comment="StreamK starting iteration (case: after extra iters)"))
        module.add(SAddU32(dst=sgpr("StreamKIter"), src0=sgpr("StreamKIter"), src1=sgpr("skExtraIters"), comment="Add extra iters"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIter"), src1=sgpr("SKItersPerWG"), comment="StreamK ending iteration (case: after extra iters)"))
        # iter count before all extra iters have been distributed
        # sTmp+1 = SKItersPerWG + 1 extra iteration
        sIter = writer.sgprPool.checkOut(2, "SKIter", preventOverflow=0)
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr("SKItersPerWG"), src1=1, comment="Spread out extra iterations"))
        module.add(SMulI32(dst=sgpr(sIter), src0=sgpr("StreamKIdx"), src1=sgpr(sIter+1), comment="StreamK starting iteration (case: before extra iters)"))
        module.add(SAddU32(dst=sgpr(sIter+1), src0=sgpr(sIter), src1=sgpr(sIter+1), comment="StreamK ending iteration (case: before extra iters)"))
        # select correct start/end iteration index
        module.add(SCmpLtU32(src0=sgpr("StreamKIdx"), src1=sgpr("skExtraIters"), comment="Check if lane gets an extra iteration"))
        module.add(SCSelectB32(dst=sgpr("StreamKIter"), src0=sgpr(sIter), src1=sgpr("StreamKIter"), comment="Set start iter"))
        module.add(SCSelectB32(dst=sgpr("StreamKIterEnd"), src0=sgpr(sIter+1), src1=sgpr("StreamKIterEnd"), comment="Set end iter"))
        writer.sgprPool.checkIn(sIter)
        module.add(SAddU32(dst=sgpr(sTmp+1), src0=sgpr("StreamKIter"), src1=sgpr(sTmp+3), comment="Offset to start of SK section"))
        module.add(SAddU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr(sTmp+3), comment="Offset to start of SK section"))
        # clamp to end of iterations
        # TODO maybe remove clamp, since extra iters code should guarantee total iterations match
        module.add(SMinU32(dst=sgpr("StreamKIterEnd"), src0=sgpr("StreamKIterEnd"), src1=sgpr("TotalIters"), comment="Cap ending iter at total SK iters"))
        # check if this WG has no work to do
        # TODO Shouldn't need this check!
        module.add(SCmpLtU32(src0=sgpr("StreamKIter"), src1=sgpr("TotalIters"), comment="Make sure there's work to do"))
        module.add(writer.longBranchScc0(Label("KernelEnd", ""), posNeg=1)) # reuse tmp
        
        # If in SK, next iteration is sTmp+2
        # Increment StreamK iteration
        module.add(skUpdateDone)
        module.add(SMovB32(dst=sgpr("StreamKIter"), src=sgpr(sTmp+1), comment="Store current iteration"))

        module.add(self.skIndexToWG(writer, kernel, sTmp))

        writer.sgprPool.checkIn(sTmp)

        return module

    def computeLoadSrd(self, writer, kernel, tc, sTmp):
        module = Module("StreamK TwoTileDPFirst computeLoadSrd")
        module.add(self.computeLoadSrdCommon(writer, kernel, tc, sTmp))
        return module

    def graAddresses(self, writer, kernel, tP, vTmp):
        module = Module("StreamK TwoTileDPFirst graAddresses")
        module.add(self.graAddressesCommon(writer, kernel, tP, vTmp))
        return module

    def declareStaggerParms(self, writer, kernel):
        module = Module("StreamK TwoTileDPFirst declareStaggerParms")
        module.add(self.declareStaggerParmsCommon(writer, kernel))
        return module

    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("StreamK TwoTileDPFirst tailLoopNumIter")
        module.add(self.tailLoopNumIterCommon(writer, kernel, loopCounter))
        return module

    def calculateLoopNumIter(self, writer, kernel, loopCounterName, loopIdx, tmpSgprInfo):
        module = Module("StreamK TwoTileDPFirst calculateLoopNumIter")
        module.add(self.calculateLoopNumIterCommon(writer, kernel, loopCounterName, loopIdx, tmpSgprInfo))
        return module

    def storeBranches(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK TwoTileDPFirst storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel))
        return module
    
    def writePartials(self, writer, kernel, skPartialsLabel):
        module = Module("StreamK TwoTileDPFirst writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel))
        return module
