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
    log2, SBarrier, SStoreB32, SLongBranchPositive, SBranch, ceilDivide, replaceHolder, SNop, staticMultiply, SSleep, \
    VAddF32, VAddF64
from ..Common import print2
# from ..TensileInstructions.Containers import SMEMModifiers
from ..Component import Component
from ..AsmStoreState import StoreState
import abc
from copy import deepcopy

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
    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        pass

    def storeBranchesCommon(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Common storeBranches")

        # No branches for atomic mode
        if kernel["StreamKAtomic"]:
            return module
        
        skFixupLabel = Label(label=writer.labels.getNameInc("SK_Fixup"), comment="")
        skStoreLabel = Label(label=writer.labels.getNameInc("SK_Store"), comment="")

        # StreamK store branches
        tmpSgpr = writer.sgprPool.checkOut(4, "globalWriteElements", preventOverflow=0)
        # if we did not start the tile, store partials
        # branch to beta == 0 store path
        module.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0, comment="does wg start tile?"))
        module.add(writer.longBranchScc0(skPartialsLabel, posNeg=1))
        # module.add(SCBranchSCC0(labelName=skPartialsLabel.getLabelName(), comment="Branch if not start tile, store partials"))

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

            # TODO Barrier here to sync all threads in workgroup, but maybe better to have separate flag for each wavefront (to be tested)
            module.add(SBarrier(comment="wait for all workgroups before resetting flag"))
            module.add(SMovB32(dst=sgpr(tmpSgpr+2), src=0, comment="flag data"))
            module.add(SStoreB32(src=sgpr(tmpSgpr+2), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=1), comment="reset flag"))
            writer.sgprPool.checkIn(tmpSgpr)

            fixupEdge = [False] # Test no edge variant
            module.add(self.fixupStep(writer, kernel, vectorWidths, elements, fixupEdge, tmpVgpr, cvtVgprStruct, sCtaIdx))
            
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
    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        pass

    def writePartialsCommon(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Common writePartials")

        # No partials for atomic mode
        if kernel["StreamKAtomic"]:
            return module
        
        module.add(skPartialsLabel)
        if kernel["DebugStreamK"] & 2 != 0:
            return module
        
        # fixupEdge = [False] # Temporary hack to test no edge variant
        edges = [False]

        partialsLabels = {}
        for edge in edges:
            partialsLabels[edge] = Label(writer.labels.getNameInc("GW_Partials_E%u" % ( 1 if edge else 0)), comment="")

        if False in edges and True in edges:
            with self.allocTmpSgpr(4) as tmpSgprInfo:
                module.add(writer.checkIsEdge(kernel, tmpSgprInfo, partialsLabels[True], partialsLabels[True]))

        for edge in edges:
            module.add(partialsLabels[edge])
            module.add(self.computeWorkspaceSrd(writer, kernel, sgpr("StreamKIdx")))
            module.add(self.partialsWriteProcedure(writer, kernel, vectorWidths, elements, False, False, edge, tmpVgpr, cvtVgprStruct, endLabel))
        
        return module

    def computeWorkspaceSrd(self, writer, kernel, sCtaIdx, tmpSgpr = None):
        module = Module("StreamK Common computeWorkspaceSrd")

        # Base Address
        module.add(SMovB32(dst=sgpr("SrdWS+0"), src=sgpr("AddressWS+0"), comment="init SRD base address (lower)"))
        module.add(SMovB32(dst=sgpr("SrdWS+1"), src=sgpr("AddressWS+1"), comment="init SRD base address (upper) + other fields"))
        module.add(SMovB32(dst=sgpr("SrdWS+2"), src="BufferOOB", comment=""))
        module.add(SMovB32(dst=sgpr("SrdWS+3"), src="Srd127_96", comment="Set bits 127_96 in post-loop SRD"))

        tmpLocal = None
        if tmpSgpr == None:
            tmpLocal = writer.sgprPool.checkOut(1, "SKMappingTemp", preventOverflow=0)
            tmpSgpr = tmpLocal

        assert kernel["BufferStore"]
        module.addSpaceLine()
        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=hex(kernel["MacroTile0"]*kernel["MacroTile1"]*writer.states.bpeCinternal), src1=sCtaIdx, comment="Offset to correct partials tile"))
        module.add(SAddU32(dst=sgpr("SrdWS+0"), src0=sgpr("SrdWS+0"), src1=sgpr(tmpSgpr), comment="add lo to SRD"))
        module.add(SAddCU32(dst=sgpr("SrdWS+1"), src0=sgpr("SrdWS+1"), src1=0, comment="add hi to SRD"))

        if tmpLocal is not None:
            writer.sgprPool.checkIn(tmpLocal)

        return module
    
    def partialsWriteProcedure(self, writer, kernel, vectorWidths, elements, alpha, beta, edge, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Common partialsWriteProcedure")

        # PreLoopVmcntCaseStr = ""
        # # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
        # if self.canOptimizePreLoopLWVmcnt:
        #     if beta:
        #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_B1_Store
        #     elif edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
        #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
        #     else:
        #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
        #     PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
        #         "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
        #     # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
        #     # OrdNLL_B1_Store is excluded
        #     if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

        edgeI = edge
        #edgeI = True    # set to True to disable vector stores
        gwvw = vectorWidths[edgeI]
        #print "globalWriteElements: edge=", edge, "beta=", beta, "atomic=", atomic

        ########################################
        # Calculate Vgprs for Write Batching
        ########################################

        ss = StoreState(writer, kernel, gwvw, edge, beta, False, elements[edgeI], isWorkspace=True)

        #print self.vgprPool.state()
        # Use VGPR up to next occupancy threshold:
        maxVgprs = writer.getMaxRegsForOccupancy(kernel["NumThreads"], writer.vgprPool.size(), \
            writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
        if writer.states.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
            # len(elements[edgeI])
            # tl = []
            # for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
            #     tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
            # for t in tl:
            #     self.vgprPool.checkIn(t)
            writer.vgprPool.growPool(writer.vgprPool.size()-writer.vgprPool.available(), maxVgprs, 1, \
                "grow-pool up to next occupancy for GlobalWrite")
        # align = 1
        # # align adjustment
        # if self.ss.cfg.numVgprsPerAddr > 1:
        #     align = max(align, self.ss.cfg.numVgprsPerAddr)
        # if self.ss.cfg.numVgprPerValuC*gwvw > 1:
        #     align = max(align, self.ss.cfg.numVgprPerValuC*gwvw)
        # if int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
        #     align = max(align, int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)))
        numVgprAvailable = writer.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)

        # Grow the register pool if needed - we need enough regs for at least one element
        # Unfortunate since this means the write logic is setting the VGPR requirement
        # for the entire kernel but at least we have a functional kernel.
        # Before growing the pool, see if we can shrink the write vector width instead?
        # TODO : the vgprSerial is needed for-ever and if we grow here will split the
        # range of the tmps.    Maybe want to move vgprSerial to first vgpr?

        # TODO: Minimum elems for StoreRemap
        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        minElements = 1 
        if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
            minElements = 2
        elif kernel["ProblemType"]["DataType"].is8bitFloat():
            # TODO STREAM-K check if needed
            minElements = 4
        minNeeded = minElements * ss.numVgprsPerElement

        shrinkDb = 0
        if shrinkDb:
            print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)

        if numVgprAvailable < minNeeded:
            gwvwOrig = gwvw
            currentOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.getLdsSize(kernel), \
                writer.vgprPool.size(), writer.agprPool.size(), writer.states.doubleVgpr)
            futureOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.getLdsSize(kernel), \
                writer.vgprPool.size() - numVgprAvailable + minNeeded, writer.agprPool.size(), writer.states.doubleVgpr)

            if shrinkDb:
                print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                    % (currentOccupancy, futureOccupancy, writer.vgprPool.size(), \
                    numVgprAvailable, minElements*ss.numVgprsPerElement))
            if futureOccupancy > currentOccupancy:
                if shrinkDb:
                    print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                        (writer.states.kernelName))
                    print("     numVgprAvailable=", numVgprAvailable, \
                        "numVgprsPerElement=", ss.numVgprsPerElement, \
                        "beta=", beta, "gwvw=", gwvw)
            elif gwvw != gwvwOrig:
                ss.gwvw = gwvw # make both representations consistent
                if shrinkDb:
                    print2("info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                        % (writer.states.kernelName, gwvwOrig, gwvw, currentOccupancy))

            if numVgprAvailable < minElements*ss.numVgprsPerElement:
                print2("info: growing pool += %d * %d for GlobalWrite\n" \
                    % (minElements,ss.numVgprsPerElement))
                print2(writer.vgprPool.state())
                # tl = []
                # for i in range(0,minElements):
                #     tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
                # for t in tl:
                #     self.vgprPool.checkIn(t)
                writer.vgprPool.growPool(0, minElements, ss.numVgprsPerElement, \
                    "grow-pool for GlobalWrite")
                numVgprAvailable = writer.vgprPool.available()
                print2(writer.vgprPool.state())

        # set atomicW after we potentially resize GWVW
        # atomicW = min(gwvw, kernel["VectorAtomicWidth"])
        atomicW = min(gwvw, writer.getVectorAtomicWidth(kernel))

        # print("NumVgprAvailable", numVgprAvailable)
        if ss.numVgprsPerElement:
            numElementsPerBatch = numVgprAvailable // ss.numVgprsPerElement
        else:
            numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

        # assert(writer.states.numVgprValuC % gwvw == 0) # sanity check

        numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

        if shrinkDb:
            print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
                "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
        if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
            numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

        # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
        if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
            # only do an even number of halves - since these share hi/lo pieces of some registers?
            if numElementsPerBatch > 1:
                numElementsPerBatch = int(numElementsPerBatch/2)*2
            elif not kernel["EnableMatrixInstruction"]:
                # (excluding MFMA+LSU case. It can work without an issue)
                # The globalWriteBatch routine below can't handle odd elements per batch
                # and 0 elements per batch is illegal.
                # so if we don't have *GPR resources to handle a larger batch then need
                # to mark overflowedResources rather than generate a kernel that won't work.
                # It might be possible to fix globalWriteBatch to handle this case but these
                # are likely to be low-performing so likely not worth optimizing.
                if shrinkDb:
                    print("WARNING: half requires at least two elements per batch")
                self.overflowedResources = 3
        #elif kernel["ProblemType"]["DataType"].is8bitFloat():
        #    if numElementsPerBatch > 1:
        #        numElementsPerBatch = int(numElementsPerBatch/4)*4

        assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%writer.states.kernelName

        #numElementsPerBatch=min(2,numElementsPerBatch) # hack to control number of batches
        # if atomic and (ss.optSingleColVgpr or ss.optSharedColVgpr):
        #     # hack to avoid re-using address vgpr across rows
        #     # atomics need to perform several memory operations
        #     # if the batch spans multiple rows, need multiple address vgpr
        #     # which is not currently supported in the two opt*ColVgpr modes
        #     firstRow = [e for e in elements[edgeI] if e[0]==0 and e[2]==0]
        #     numElementsPerBatch=min(len(firstRow),numElementsPerBatch)

        numBatches = max(1, ceilDivide(len(elements[edgeI]),numElementsPerBatch))

        numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

        # TODO STREAM-K activation code

        if writer.db["PrintStoreRegisterDb"]:
            print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", ss.numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
            print("numSgprs=", numSgprs, "sgprPool.size()=", writer.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
                "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
            print(writer.sgprPool.state())
        module.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
            (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
        #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
        # so if we don't have *GPR resources to handle a larger batch then need
        # to mark overflowedResources rather than generate a kernel that won't work.
        with writer.allocTmpSgpr(numSgprs, 2) as tmpSgprRes:
            tmpSgpr = tmpSgprRes.idx
            elementSgprs = tmpSgpr + ss.cfg.numTempSgprPerBatch

            codeAccVgprRead = deepcopy(writer.codes.accVgprRead) if writer.states.serializedStore else None
            # TODO STREAM-K remove this?
            useCodeMulAlpha = kernel["MIArchVgpr"] and alpha and not (kernel["GlobalSplitU"] > 1)
            if useCodeMulAlpha: # do not set codeAccVgprRead=None if GSU>1
                codeAccVgprRead = None

            for batchIdx in range(0, numBatches):
                elementStartIdx = batchIdx * numElementsPerBatch
                elementStopIdx = min(elementStartIdx + numElementsPerBatch, len(elements[edgeI]))
                elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
                #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
                # elementVgprs can be large and should be perfectly tuned to the number of available
                # VGPRS.    We do not want to accidentally overflow and grow the pool here:

                module.add(self.partialsWriteBatch(writer, kernel, ss, batchIdx, alpha, beta, edge, gwvw, atomicW, \
                        elementsThisBatch, writer.vgprs.addrD, writer.vgprs.addrC, \
                        tmpVgpr, cvtVgprStruct, \
                        elementSgprs, tmpSgpr, codeAccVgprRead))
            # delay PreLoopVmcntCase code after globalWrite
            # if self.canOptimizePreLoopLWVmcnt:
            #     kStr += PreLoopVmcntCaseStr

            # Set flag
            module.add(SWaitCnt(vmcnt=0, comment="wait for data store"))
            module.add(SBarrier(comment="store all data before setting flag"))
            module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr), src=sgpr("StreamKIdx"), shiftHex=log2(4), comment="flag offset based on CTA index"))
            with writer.allocTmpSgpr(1) as flagSgprRes:
                flagSgpr = flagSgprRes.idx
                module.add(SMovB32(dst=sgpr(flagSgpr), src=1, comment="flag data"))
                module.add(SStoreB32(src=sgpr(flagSgpr), base=sgpr("AddressFlags", 2), soffset=sgpr(tmpSgpr), smem=SMEMModifiers(glc=1), comment="set flag"))
            module.add(SWaitCnt(lgkmcnt=0, comment="wait for flag")) # TODO just for testing
        
        # TODO - if this is the last tile, don't need to jump to next instruction
        # NOTE: in SR kernel, we need long branch since PRNG explodes the line of codes 
        if kernel["ProblemType"]["StochasticRounding"]: # in-device RND
            with self.allocTmpSgpr(3) as tmpSgprInfo:
                module.add(SLongBranchPositive(endLabel, tmpSgprInfo))
        else:
            module.add(SBranch(labelName=endLabel.getLabelName(), comment="jump to end"))

        # Finish one write path, reset currPreLoopVmcntCase to Undefined
        # self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined

        return module

    def partialsWriteBatch(self, writer, kernel, ss, batchIdx, applyAlpha, beta, edge, gwvw, atomicW, \
            batchElements, addrD, addrC, \
            tmpVgpr, cvtVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead):
        module = Module("StreamK Common partialsWriteBatch")

        module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
            (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

        if kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        # comment tt1, tt0, vc1, vc0
        # tt = thread tile, vc=vector component
        commentStr = "Partials Write%s%s%s Batch #%u (d1,d0,vc1,vc0) =\n     " \
            % (" Alpha" if applyAlpha else "", " Beta" if beta else "", " Edge" if edge else "", batchIdx)
        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            commentStr += "(%u,%u,%u,%u:vw%u)" % (element[0], element[1], element[2], element[3], gwvw)
            if elementIdx < len(batchElements)-1:
                commentStr += "; "
        module.addComment2(commentStr)

        # allow expanding vgpr pool for OptNLL
        # preventOverflow = (not isOptNLL)
        # ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=isOptNLL, isWorkspace=True)
        ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=False, biasDim=0, isWorkspace=True)

        storesIssued = 0
        tmpS01 = tmpSgpr # scratch sgprs

        ########################################
        # calculate addr and masks
        module.addComment1("calc coords, apply mask, and issue loads (if necessary)")
        # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
        # on the thread and tid number.    These are ELEMENT offsets from start of tensor C
        # for the top-left corner this thread will write.    These are not changed
        # across all the store loop iters.
        if writer.db["ConservativeWaitCnt"] & 0x10:
            module.add(SBarrier("debug"))
            module.add(SWaitCnt(vmcnt=0, comment="ConservativeWaitCnt"))
            if writer.states.archCaps["SeparateVscnt"]:
                module.add(SWaitCnt(vscnt=0, comment="writes"))
            module.add(SBarrier("debug"))
        if not edge and writer.db["ForceEdgeStores"]>=2:
            module.add(self.parentWriter.getBomb()) # should not get here
        if edge and writer.db["AssertNoEdge"]:
            module.add(self.parentWriter.getBomb()) # should not get here

        ## create code Module to push mov vgpr,acc instructions
        # if kernel["StoreCInUnroll"] and not edge:
        #     accVgprRead = Code.Module("movaccVgpr")
        #     self.StoreCUnrollLoadCWaitComment = "waitcnt for LoadC" # this will be used later to identify waitcnt for loadC

        ########################################
        # AccVgpr read
        # if kernel.enabledSetPrioSplitLDS:
        #     kStr += inst("s_setprio", "0", "")
        if codeAccVgprRead is not None: # and writer.kernel["LocalSplitU"] == 1
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        module.add(replaceHolder(codeAccVgprRead.items().pop(0), ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
                        # module.add(replaceHolder(self.codeAccVgprRead.items().pop(0), ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - self.parentWriter.states.c.startVgprValu))
                        # if kernel["StoreCInUnroll"] and not edge:
                        #     tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
                        #     accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

            if not kernel["MIArchVgpr"]:
                module.add(SNop(1, "2 wait states required before reading vgpr"))

        ########################################
        # Not Atomic
        ########################################
        # else:
        # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
        for elementIdx in range(len(batchElements)):
            for vi in range(gwvw):
                sumIdxV = ss.elementSumIdx[elementIdx] + vi
                # TODO STREAM-K is start value needed now?
                # TODO KUPO!!!!!!!!!!!!!!!!
                # newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
                if kernel["ProblemType"]["ComputeDataType"].isInt32() or kernel["ProblemType"]["ComputeDataType"].isSingle():
                    if writer.db["ForceExpectedValue"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=writer.db["ValueCExpectedValue"], comment="force expected value"))
                        # module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdxV), src=self.debugConfig["ValueCExpectedValue"], comment="force expected value" ))
                    if writer.db["ForceVSerial"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=vgpr("Serial"), comment="force expected value to serial"))
                        # module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdxV), src=vgpr("Serial"), comment="force expected value to serial" ))
                    if writer.db["CheckValueC"]:
                        module.add(SMovB32(dst=sgpr(tmpS01), src=writer.db["ValueCExpectedValue"], comment="Move expected value"))
                        module.add(writer.getCmpAssert(writer.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01)))

        module.addComment1("apply mask, calc new C and issue writes")
        #kStr += self.bomb() # can see store addresses just before the store inst

        # if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        #     vgprBf16Temp = tmpCVTVgpr
        #     vgprBf16Mask = vgprBf16Temp + 1
        #     vgprFp32Nan = vgprBf16Temp + 2
        #     vgprBf16Inc = vgprBf16Temp + 3
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
        #     kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )
        if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp32Nan), "0x7fff0000", "fp32 Nan" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" ))
        elif kernel["ProblemType"]["DestDataType"].isFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8NanInf), "0x207", "Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Max), "0x43700000", "Fp8 Max value 240 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Min), "0xc3700000", "Fp8 Min value -240 as float32" ))
        elif kernel["ProblemType"]["DestDataType"].isBFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8NanInf), "0x207", "Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Max), "0x47600000", "BF8 Max value 57344 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Min), "0xc7600000", "BF8 Min value -57344 as float32" ))
        
        # DestDataType for 8bit Float can only be F8 or B8
        # if kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8(): # F8 is always HPA
        #     # make vgprF8Temp0 always even to use pk instruction later
        #     if tmpCVTVgpr % 2 == 0:
        #         vgprF8Temp0 = tmpCVTVgpr
        #         vgprF8Max = vgprF8Temp0 + 2
        #         vgprF8Min = vgprF8Temp0 + 3
        #     else: 
        #         vgprF8Max = tmpCVTVgpr
        #         vgprF8Temp0 = vgprF8Max + 1
        #         vgprF8Min = vgprF8Max + 3
            
        #     if kernel["ProblemType"]["Fp32toFp8SWClip"]:
        #         # set flag of f32 NaN and +/- INF for v_cmp_class
        #         vgprFp32NanInfFlag = vgprF8Min + 1
        #         kStr += inst("v_mov_b32", vgpr(vgprFp32NanInfFlag), "0x207", "flag for Nan and +/- inf" )
        #         # set max/min values for clipping
        #         if kernel["ProblemType"]["DestDataType"].isFloat8():
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x43700000", "save 240.0f as max for clipping" )
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC3700000", "save -240.0f as min for clipping" )
        #         else: #BFloat8 
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x47600000", "save 57344.0f as max for clipping" )
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC7600000", "save -57344`.0f as min for clipping" )

        storeCode = Module("Partials GroupLoadStore")
        for elementIdx in range(len(batchElements)):
            element = batchElements[elementIdx]
            addrCalc: AddrCalculation = ss.elementAddr[elementIdx]
            addr = addrCalc.addrDVgpr
            sumIdx = ss.elementSumIdx[elementIdx]

            storeWidth = kernel["StoreVectorWidth"]
            # storeWidth = 2
            if batchIdx == 0 and elementIdx == 0:
                tmpSgprRes = RegisterPoolResource(idx=tmpS01, size=1)
                module.add(staticMultiply(vgpr(addr), vgpr("Serial"), storeWidth * writer.states.bpeCinternal, tmpSgprRes))
                # kStr += inst("v_mul_lo_u32", , "Partials buffer address")
                module.add(SMovB32(dst=sgpr(tmpS01), src=0, comment="Init sgpr offset"))
            else:
                increment = (kernel["WavefrontSize"] * 4) * storeWidth * writer.states.bpeCinternal
                module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=increment, comment="Inc sgpr offset"))
            
            # TODO StreamK need this packing code???
            # if self.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"] and kernel["ProblemType"]["DestDataType"].isHalf() and (not kernel["ProblemType"]["HighPrecisionAccumulate"]):
            #     for vi in range(0, gwvw):
            #         sumIdxV = ss.elementSumIdx[elementIdx] + vi
            #         if vi%2 == 1:
            #             d = ss.elementSumIdx[elementIdx] + vi//2
            #             kStr += inst("v_pack_b32_f16", vgpr(d), vgpr("ValuC+%u"%(sumIdxV-1)), vgpr("ValuC+%u"%sumIdxV), "Pack with neighbor" )

            # if not kernel["StoreRemapVectorWidth"]:
            tmpStoreCode = writer.addStore(kernel, ss, 'WS', addrCalc, sumIdx, tmpS01, edge, wsOffset=sgpr(tmpS01))
            if kernel["GroupLoadStore"]:
                storeCode.add(tmpStoreCode)
            else:
                module.add(tmpStoreCode)
            storesIssued += 1

        module.add(storeCode)

        # return registers to pool:
        lastData = -1
        for elementIdx in range(0, len(batchElements)):
            if not ss.sharedColDVgprs:
                addrCalc: AddrCalculation = ss.elementAddres[elementIdx]
                addrDVgpr = addrCalc.addrDVgpr
                addrCVgpr = addrCalc.addrCVgpr
                writer.vgprPool.checkIn(addrDVgpr)
                if addrCVgpr != addrDVgpr:
                    writer.vgprPool.checkIn(addrCVgpr)

            data = ss.elementData[elementIdx]
            if data != 0:
                if data != lastData:
                    writer.vgprPool.checkIn(data)
                lastData = data

        ss.firstBatch = False
        ss.checkInTempVgprC()

        if writer.states.serializedStore:
            module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

        # Update the store cnt to preLoopVmcntDict for Case2/3
        # (No need to update for Case0:'Undefined' or Case4:'OrdNLL_B1_Store')
        # TODO STREAM-K Need this?
        # if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        #     if not self.archCaps["SeparateVscnt"]:
        #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] += storesIssued

        return module

    def fixupStep(self, writer, kernel, vectorWidths, elements, edges, tmpVgpr, cvtVgprStruct, sCtaIdx):
        module = Module("StreamK Common fixupStep")

        fixupLabels = {}
        for edge in edges:
            fixupLabels[edge] = Label(writer.labels.getNameInc("Fixup_E%u" % ( 1 if edge else 0)), comment="")

        # branch if Edge0 or Edge1
        if False in edges and True in edges:
            module.add(writer.checkIsEdge(kernel, tmpSgprInfo, fixupLabels[True], fixupLabels[True]))

        # by now we either jumped to E1 or stayed at E0
        for edge in edges:
            # write label for batch case
            module.add(fixupLabels[edge])

            # PreLoopVmcntCaseStr = ""
            # # not generate Case 2 if StoreCInUnroll with StoreVectorWidth==1 (Case 2 will be same as Case 3)
            # if self.canOptimizePreLoopLWVmcnt:
            #     if edge or (kernel["StoreCInUnroll"] and kernel["StoreVectorWidth"]==1):
            #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OrdNLL_E1_Store
            #     else:
            #         self.currPreLoopVmcntCase = PreLoopVmcntCase.OptNLL_Store
            #     PreLoopVmcntCaseStr = inst("s_mov_b32", sgpr("PreLoopLWVmcntCase"), hex(self.currPreLoopVmcntCase.value), \
            #         "for optimizing next PreLoop LW vmcnt, set to Case%u"%self.currPreLoopVmcntCase.value)
            #     # reset vmcnt if the dict has this key (OptNLL_Store, OrdNLL_E1_Store),
            #     # OrdNLL_B1_Store is excluded
            #     if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
            #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] = 0

            edgeI = edge
            #edgeI = True    # set to True to disable vector stores
            gwvw = vectorWidths[edgeI]

            ########################################
            # Calculate Vgprs for Write Batching
            ########################################

            # ss = StoreState(writer, kernel, gwvw, edge, beta, False, elements[edgeI], isWorkspace=True)
            ss = StoreState(writer, kernel, gwvw, edge, True, False, elements[edgeI], isWorkspace=True)
            # ss = StoreState(writer, kernel, gwvw, edge, False, False, elements[edgeI], isWorkspace=True)

            # how many vgprs are needed for zero elements
            # 2 for addressC in vgpr for addition - already checked out
            # 2 for coord0,1 of thread - already checked out
            # 2 for tmp - already checked out

            # 5 = how many vgprs are needed per element (flat)
            #    - 2 for addr
            #    - 3 for GLOBAL_OFFSET_C calculation (can overlap below, therefore max)
            #    - if beta gwvw*rpe for new value
            #    - if atomic 2*rpe for old and cmp values

            # print("numVgprsPerAddr=%u, numVgprsPerDataPerVI=%u, numVgprPerValuC=%u"%(self.ss.cfg.numVgprsPerAddr, self.ss.cfg.numVgprsPerDataPerVI, self.ss.cfg.numVgprPerValuC))
            # numVgprsPerElement = self.ss.cfg.numVgprPerValuC*gwvw + self.ss.cfg.numVgprsPerAddr + int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw))

            # if kernel["GroupLoadStore"] and kernel["ProblemType"]["UseBeta"]:
            #     numVgprsPerElement += self.ss.cfg.numVgprsPerAddr

            #print self.vgprPool.state()
            # Use VGPR up to next occupancy threshold:
            maxVgprs = writer.getMaxRegsForOccupancy(kernel["NumThreads"], writer.vgprPool.size(), \
                writer.getLdsSize(kernel), writer.agprPool.size(), writer.states.doubleVgpr)
            if writer.states.serializedStore: # get aggressive when serializedStore is on; not necessarily exclusive to this parameter
                # len(elements[edgeI])
                # tl = []
                # for i in range(self.vgprPool.size()-self.vgprPool.available(), maxVgprs):
                #     tl.append(self.vgprPool.checkOut(1, "grow-pool up to next occupancy for GlobalWrite"))
                # for t in tl:
                #     self.vgprPool.checkIn(t)
                writer.vgprPool.growPool(writer.vgprPool.size()-writer.vgprPool.available(), maxVgprs, 1, \
                    "grow-pool up to next occupancy for GlobalWrite")
            # align = 1
            # # align adjustment
            # if self.ss.cfg.numVgprsPerAddr > 1:
            #     align = max(align, self.ss.cfg.numVgprsPerAddr)
            # if self.ss.cfg.numVgprPerValuC*gwvw > 1:
            #     align = max(align, self.ss.cfg.numVgprPerValuC*gwvw)
            # if int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)) > 1:
            #     align = max(align, int(ceil(self.ss.cfg.numVgprsPerDataPerVI * gwvw)))
            numVgprAvailable = writer.vgprPool.availableBlock(ss.numVgprsPerElement, ss.align)

            # Grow the register pool if needed - we need enough regs for at least one element
            # Unfortunate since this means the write logic is setting the VGPR requirement
            # for the entire kernel but at least we have a functional kernel.
            # Before growing the pool, see if we can shrink the write vector width instead?
            # TODO : the vgprSerial is needed for-ever and if we grow here will split the
            # range of the tmps.    Maybe want to move vgprSerial to first vgpr?

            # TODO: Minimum elems for StoreRemap
            # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
            minElements = 1 
            if kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16():
                minElements = 2
            elif kernel["ProblemType"]["DataType"].is8bitFloat():
                minElements = 4
            minNeeded = minElements * ss.numVgprsPerElement

            shrinkDb = 0
            if shrinkDb:
                print("numVgprAvailable=", numVgprAvailable, "minElements=", minElements, "minNeeded=", minNeeded)

            if numVgprAvailable < minNeeded:
                gwvwOrig = gwvw
                currentOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.getLdsSize(kernel), \
                        writer.vgprPool.size(), writer.agprPool.size(), writer.states.doubleVgpr)
                futureOccupancy = writer.getOccupancy(kernel["NumThreads"], writer.getLdsSize(kernel), \
                        writer.vgprPool.size() - numVgprAvailable + minNeeded, writer.agprPool.size(), writer.states.doubleVgpr)

                if shrinkDb:
                    print("currentOccupancy=%u futureOccupancy=%u VGPRs=%u numVgprAvail=%u vgprPerElem=%u" \
                        % (currentOccupancy, futureOccupancy, writer.vgprPool.size(), \
                        numVgprAvailable, minElements*ss.numVgprsPerElement))
                if futureOccupancy > currentOccupancy:
                    if shrinkDb:
                        print("warning: %s growing VGPR for GlobalWrite batching - this may bloat VGPR usage" % \
                            (writer.states.kernelName))
                        print("     numVgprAvailable=", numVgprAvailable, \
                            "numVgprsPerElement=", ss.numVgprsPerElement, \
                            "gwvw=", gwvw)
                elif gwvw != gwvwOrig:
                    ss.gwvw = gwvw # make both representations consistent
                    if shrinkDb:
                        print2(3, "info: %s shrank gwvw from %u to %u but kept occupancy same=%u." \
                            % (writer.states.kernelName, gwvwOrig, gwvw, currentOccupancy))

                if numVgprAvailable < minElements*ss.numVgprsPerElement:
                    print2(3, "info: growing pool += %d * %d for GlobalWrite\n" \
                        % (minElements,ss.numVgprsPerElement))
                    print2(3, writer.vgprPool.state())
                    # tl = []
                    # for i in range(0,minElements):
                    #     tl.append(self.vgprPool.checkOut(numVgprsPerElement, "grow-pool for GlobalWrite"))
                    # for t in tl:
                    #     self.vgprPool.checkIn(t)
                    writer.vgprPool.growPool(0, minElements, ss.numVgprsPerElement, \
                        "grow-pool for GlobalWrite")
                    numVgprAvailable = writer.vgprPool.available()
                    print2(3, writer.vgprPool.state())

            # print("NumVgprAvailable", numVgprAvailable)
            if ss.numVgprsPerElement:
                numElementsPerBatch = numVgprAvailable // ss.numVgprsPerElement
            else:
                numElementsPerBatch = len(elements[edgeI]) # max, do 'em all

            # assert(self.numVgprValuC % gwvw == 0) # sanity check

            numElementsPerBatch = numElementsPerBatch if not kernel["NumElementsPerBatchStore"] else min(kernel["NumElementsPerBatchStore"],numElementsPerBatch)

            if shrinkDb:
                print("NumElementsPerBatch=", numElementsPerBatch, "LimitedBySgprs=", ss.cfg.numElementsPerBatchLimitedBySgprs, \
                        "WARNING" if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch else "okay")
            if ss.cfg.numElementsPerBatchLimitedBySgprs < numElementsPerBatch:
                numElementsPerBatch = ss.cfg.numElementsPerBatchLimitedBySgprs

            # TODO: Which of DataType or DestDataType is in a better sense? 0114: Check Using DestDataType + HSS
            if (kernel["ProblemType"]["DataType"].isHalf() or kernel["ProblemType"]["DataType"].isBFloat16()):
                # only do an even number of halves - since these share hi/lo pieces of some registers?
                if numElementsPerBatch > 1:
                    numElementsPerBatch = int(numElementsPerBatch/2)*2
                elif not kernel["EnableMatrixInstruction"]:
                    # (excluding MFMA+LSU case. It can work without an issue)
                    # The globalWriteBatch routine below can't handle odd elements per batch
                    # and 0 elements per batch is illegal.
                    # so if we don't have *GPR resources to handle a larger batch then need
                    # to mark overflowedResources rather than generate a kernel that won't work.
                    # It might be possible to fix globalWriteBatch to handle this case but these
                    # are likely to be low-performing so likely not worth optimizing.
                    if shrinkDb:
                        print("WARNING: half requires at least two elements per batch")
                    self.overflowedResources = 3
            #elif kernel["ProblemType"]["DataType"].is8bitFloat():
            #    if numElementsPerBatch > 1:
            #        numElementsPerBatch = int(numElementsPerBatch/4)*4

            assert numElementsPerBatch > 0, "numElementsPerBatch=0 for %s"%writer.states.kernelName

            # if no atomics and no edge, then write whole vectors
            # ERROR commented out in globalWriteELements, causes numVectorsPerBatch to not be int
            # if not edge: # not atomic and 
            #    numVectorsPerBatch = numElementsPerBatch / kernel["GlobalWriteVectorWidth"]
            #    #print "    NumVectorsPerBatch", numVectorsPerBatch
            #    numElementsPerBatch = numVectorsPerBatch * kernel["GlobalWriteVectorWidth"]
            numBatches = max(1, ceilDivide(len(elements[edgeI]),numElementsPerBatch))

            numSgprs = ss.cfg.numTempSgprPerBatch + ss.cfg.numMaskSgprPerBatch + ss.cfg.numMaskSgprPerElement * numElementsPerBatch

            if writer.db["PrintStoreRegisterDb"]:
                print("edgeI", edgeI, "NumBatches", numBatches, "NumElementsPerBatch", numElementsPerBatch, "numVgprsPerElement", ss.numVgprsPerElement, "len(elements[edgeI])", len(elements[edgeI]))
                print ("numSgprs=", numSgprs, "sgprPool.size()=", writer.sgprPool.size(), "numTempSgprPerBatch=", ss.cfg.numTempSgprPerBatch,
                    "numMaskSgprPerBatch=", ss.cfg.numMaskSgprPerBatch, "numMaskSgprPerElement=", ss.cfg.numMaskSgprPerElement)
                print(writer.sgprPool.state())
            module.addComment1("edge=%d, allocate %u sgpr. perBatchTmpS=%u perBatchMaskS=%u perElementMaskS=%u elementsPerBatch=%u" %
                    (edgeI, numSgprs, ss.cfg.numTempSgprPerBatch, ss.cfg.numMaskSgprPerBatch, ss.cfg.numMaskSgprPerElement, numElementsPerBatch))
            #kStr += "// storeStats, %d, %d, %d\n"% (edgeI, numSgprs, numElementsPerBatch)
            # so if we don't have *GPR resources to handle a larger batch then need
            # to mark overflowedResources rather than generate a kernel that won't work.

            with writer.allocTmpSgpr(numSgprs, 2) as tmpSgprRes:
                tmpSgpr = tmpSgprRes.idx
                elementSgprs = tmpSgpr + ss.cfg.numTempSgprPerBatch

                codeAccVgprRead = deepcopy(writer.codes.accVgprRead) if writer.states.serializedStore else None
                # codeAccVgprRead = deepcopy(writer.codes.codeAccVgprRead) if writer.states.serializedStore else None
                codeAccVgprWrite = deepcopy(writer.codes.accVgprWrite) if writer.states.serializedStore else None

                module.add(self.computeWorkspaceSrd(writer, kernel, sgpr(sCtaIdx), tmpSgpr))

                for batchIdx in range(0, numBatches):
                    elementStartIdx = batchIdx * numElementsPerBatch
                    elementStopIdx = min(elementStartIdx + numElementsPerBatch, len(elements[edgeI]))
                    elementsThisBatch = elements[edgeI][elementStartIdx:elementStopIdx]
                    #print("BATCH[%u/%u]: elements[edgeI][%u:%u] VGPRs=%u" % (batchIdx, numBatches, elementStartIdx, elementStopIdx,numVgprsPerElement ))
                    # elementVgprs can be large and should be perfectly tuned to the number of available
                    # VGPRS.    We do not want to accidentally overflow and grow the pool here:

                    module.add(self.fixupBatch(writer, kernel, ss, batchIdx, edge, gwvw, \
                            elementsThisBatch, writer.vgprs.addrD, writer.vgprs.addrC, \
                            tmpVgpr, cvtVgprStruct, \
                            elementSgprs, tmpSgpr, codeAccVgprRead, codeAccVgprWrite))
                # delay PreLoopVmcntCase code after globalWrite
                # if self.canOptimizePreLoopLWVmcnt:
                #     kStr += PreLoopVmcntCaseStr

            # Finish one write path, reset currPreLoopVmcntCase to Undefined
            # self.currPreLoopVmcntCase = PreLoopVmcntCase.Undefined

            # kStr += inst("s_branch", skStoreLabel, "jump to store")

        return module

    def fixupBatch(self, writer, kernel, ss, batchIdx, edge, gwvw, \
            batchElements, addrD, addrC, \
            tmpVgpr, cvtVgprStruct, batchElementSgprs, tmpSgpr, codeAccVgprRead, codeAccVgprWrite):
        module = Module("StreamK Common fixupBatch")

        module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u" % \
            (ss.optSingleColVgpr, ss.optSharedColVgpr, ss.optSGPRUsage, ss.optSrdIncForRow))

        if kernel["StoreSyncOpt"]:
            module.add(SSleep(kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
            module.add(SBarrier())

        # comment tt1, tt0, vc1, vc0
        # tt = thread tile, vc=vector component
        commentStr = "Fixup%s Batch #%u (d1,d0,vc1,vc0) =\n     " \
            % (" Edge" if edge else "", batchIdx)
        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            commentStr += "(%u,%u,%u,%u:vw%u)" % (element[0], element[1], element[2], element[3], gwvw)
            if elementIdx < len(batchElements)-1:
                commentStr += "; "
        module.addComment2(commentStr)
        # print(self.kernelName)
        # print(commentStr)

        # allow expanding vgpr pool for OptNLL
        # preventOverflow = True #(not isOptNLL)
        # ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, preventOverflow=preventOverflow, isWorkspace=True)
        ss.setupStoreElementsForBatch(kernel, gwvw, batchElements, batchElementSgprs, isOptNLL=False, biasDim=0, isWorkspace=True)

        loadsIssued = 0
        storesIssued = 0
        tmpS01 = tmpSgpr # scratch sgprs

        wavelen = kernel["WavefrontSize"]
        # laneSGPRC = writer.states.laneSGPRCount
        # always use gwvw for buffer load C for atomic_cmpswap
        # bpm = self.bpeCexternal * atomicW
        # bpm = self.bpeCexternal * gwvw
        # vgprLoadDW = 1*(bpm//4)
        # atomic oparation width. 1 for b32, 2 for b64
        # atomicOpW = (atomicW * self.bpeCexternal) // 4
        # if atomicOpW > 2:
        #     # should not exceeding 2.
        #     atomicOpW = 2

        ########################################
        # calculate addr and masks
        module.addComment1("calc coords, apply mask, and issue loads (if necessary)")
        # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
        # on the thread and tid number.    These are ELEMENT offsets from start of tensor C
        # for the top-left corner this thread will write.    These are not changed
        # across all the store loop iters.
        if writer.db["ConservativeWaitCnt"] & 0x10:
            module.add(SBarrier("debug"))
            module.add(SWaitCnt(vmcnt=0, comment="ConservativeWaitCnt"))
            if writer.states.archCaps["SeparateVscnt"]:
                module.add(SWaitCnt(vscnt=0, comment="writes"))
            module.add(SBarrier("debug"))
        if not edge and writer.db["ForceEdgeStores"]>=2:
            module.add(self.parentWriter.getBomb()) # should not get here
        if edge and writer.db["AssertNoEdge"]:
            module.add(self.parentWriter.getBomb()) # should not get here

        # atomicAddC = kernel["AtomicAddC"] and not edge

        ## create code Module to push mov vgpr,acc instructions
        # if kernel["StoreCInUnroll"] and not edge:
        #     accVgprRead = Code.Module("movaccVgpr")
        #     self.StoreCUnrollLoadCWaitComment = "waitcnt for LoadC" # this will be used later to identify waitcnt for loadC

        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            addrCVgpr = ss.elementAddr[elementIdx].addrCVgpr
            # addrDVgpr = ss.elementAddr[elementIdx].addrDVgpr
            addrCalc = ss.elementAddr[elementIdx]
            data = ss.elementData[elementIdx]
            # mask = ss.elementMask[elementIdx]
            # sumIdx = ss.elementSumIdx[elementIdx]
            # d1 = element[0]
            # d0 = element[1]
            # vc1 = element[2]
            vc0 = element[3]

            storeWidth = kernel["StoreVectorWidth"]
            # storeWidth = 2
            if batchIdx == 0 and elementIdx == 0:
                module.add(staticMultiply(vgpr(addrCVgpr), vgpr("Serial"), storeWidth * writer.states.bpeCinternal, sgpr(tmpS01)))
                # kStr += inst("v_mul_lo_u32", , "Partials buffer address")
                module.add(SMovB32(dst=sgpr(tmpS01), src=0, comment="Init sgpr offset"))
            else:
                increment = (kernel["WavefrontSize"] * 4) * storeWidth * writer.states.bpeCinternal
                module.add(SAddU32(dst=sgpr(tmpS01), src0=sgpr(tmpS01), src1=increment, comment="Inc sgpr offset"))

            module.add(writer.readInput(kernel, ss, 'WS', kernel["ProblemType"]["ComputeDataType"], addrCalc, vc0, data, gwvw, addrCVgpr, sgpr(tmpS01)))
            loadsIssued += 1

        ########################################
        # AccVgpr read
        # if kernel.enabledSetPrioSplitLDS:
        #     kStr += inst("s_setprio", "0", "")
        if codeAccVgprRead is not None:
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        module.add(replaceHolder(codeAccVgprRead.items().pop(0), ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - writer.states.c.startVgprValu))
                        # tempStr = str(codeAccVgprRead.items().pop(0))
                        # kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
                        # if kernel["StoreCInUnroll"] and not edge:
                        #     tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
                        #     accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

            if not kernel["MIArchVgpr"]:
                module.add(SNop(1, "2 wait states required before reading vgpr"))

        ########################################
        # Not Atomic
        ########################################
        # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
        interleaveStoreVmcnt = writer.states.interleaveStoreVmcnt and not edge
        for elementIdx in range(0, len(batchElements)):
            for vi in range(0, gwvw):
                sumIdxV = ss.elementSumIdx[elementIdx] + vi
                # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
                if kernel["ProblemType"]["ComputeDataType"].isInt32() or kernel["ProblemType"]["ComputeDataType"].isSingle():
                    if writer.db["ForceExpectedValue"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=writer.db["ValueCExpectedValue"], comment="force expected value"))
                    if writer.db["ForceVSerial"]:
                        module.add(VMovB32(dst=vgpr("ValuC+%u"%sumIdxV), src=vgpr("Serial"), comment="force expected value to serial"))
                    if writer.db["CheckValueC"]:
                        module.add(SMovB32(dst=sgpr(tmpS01), src=writer.db["ValueCExpectedValue"], comment="Move expected value"))
                        module.add(writer.getCmpAssert(writer.asmAssert.eq, vgpr("ValuC+%u"%sumIdxV), sgpr(tmpS01)))

        ########################################
        # wait for batched load
        if not interleaveStoreVmcnt: # beta and 
            module.add(SWaitCnt(vmcnt=0, comment="wait C"))
            if writer.states.archCaps["SeparateVscnt"]:
                module.add(SWaitCnt(vscnt=0, comment="writes"))

            # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
            # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
            # And this only occurs when beta=true, so case must not be 2 or 3
            # assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, \
            #     "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

        module.addComment1("apply mask, calc new C and issue writes")
        #kStr += self.bomb() # can see store addresses just before the store inst

        # if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
        #     vgprBf16Temp = tmpCVTVgpr
        #     vgprBf16Mask = vgprBf16Temp + 1
        #     vgprFp32Nan = vgprBf16Temp + 2
        #     vgprBf16Inc = vgprBf16Temp + 3
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" )
        #     kStr += inst("v_mov_b32", vgpr(vgprFp32Nan), "0x7fff0000", "fp32 Nan" )
        #     kStr += inst("v_mov_b32", vgpr(vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" )
        if kernel["ProblemType"]["DestDataType"].isBFloat16() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Mask), "0xffff0000", "mask for pack two bfloat16 element to 32bit" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp32Nan), "0x7fff0000", "fp32 Nan" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBf16Inc), "0x7fff", "rounding bias for bfloat16" ))
        elif kernel["ProblemType"]["DestDataType"].isFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8NanInf), "0x207", "Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Max), "0x43700000", "Fp8 Max value 240 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprFp8Min), "0xc3700000", "Fp8 Min value -240 as float32" ))
        elif kernel["ProblemType"]["DestDataType"].isBFloat8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8NanInf), "0x207", "Nan and +/- inf" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Max), "0x47600000", "BF8 Max value 57344 as float32" ))
            module.add(VMovB32(vgpr(cvtVgprStruct.vgprBF8Min), "0xc7600000", "BF8 Min value -57344 as float32" ))

        # DestDataType for 8bit Float can only be F8 or B8
        # if kernel["ProblemType"]["DestDataType"].isFloat8() or kernel["ProblemType"]["DestDataType"].isBFloat8(): # F8 is always HPA
        #     # make vgprF8Temp0 always even to use pk instruction later
        #     if tmpCVTVgpr % 2 == 0:
        #         vgprF8Temp0 = tmpCVTVgpr
        #         vgprF8Temp1 = vgprF8Temp0 + 1
        #         vgprF8Max = vgprF8Temp0 + 2
        #         vgprF8Min = vgprF8Temp0 + 3
        #     else: 
        #         vgprF8Max = tmpCVTVgpr
        #         vgprF8Temp0 = vgprF8Max + 1
        #         vgprF8Temp1 = vgprF8Max + 2
        #         vgprF8Min = vgprF8Max + 3
            
        #     if kernel["ProblemType"]["Fp32toFp8SWClip"]:
        #         # set flag of f32 NaN and +/- INF for v_cmp_class
        #         vgprFp32NanInfFlag = vgprF8Min + 1
        #         kStr += inst("v_mov_b32", vgpr(vgprFp32NanInfFlag), "0x207", "flag for Nan and +/- inf" )
        #         # set max/min values for clipping
        #         if kernel["ProblemType"]["DestDataType"].isFloat8():
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x43700000", "save 240.0f as max for clipping" )
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC3700000", "save -240.0f as min for clipping" )
        #         else: #BFloat8 
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Max), "0x47600000", "save 57344.0f as max for clipping" )
        #             kStr += inst("v_mov_b32", vgpr(vgprF8Min), "0xC7600000", "save -57344`.0f as min for clipping" )

        for elementIdx in range(0, len(batchElements)):
            element = batchElements[elementIdx]
            addr = ss.elementAddr[elementIdx].addrDVgpr
            mask = ss.elementMask[elementIdx]
            addrCalc = ss.elementAddr[elementIdx]
            # d1 = element[0]
            # d0 = element[1]
            # vc1 = element[2]
            vc0 = element[3]
            sumIdx = ss.elementSumIdx[elementIdx]

            # apply in-bounds exec mask
            if edge and not kernel["BufferStore"]:
                module.add(writer.getEdgeMovInstType()(EXEC(), sgpr(mask, writer.states.laneSGPRC), "sgprs -> exec"))
                # kStr += inst("s_mov_b{}".format(wavelen), self.exec, sgpr(mask,laneSGPRC), "sgprs -> exec" )

            # if beta:
            # if GWVW=1 the half path still assumes we have
            # at least two stores so does some combining across VI -
            # for example assuming we can have two elements and can use pk_mul
            # here:
            if interleaveStoreVmcnt: # beta and
                if writer.states.archCaps["SeparateVscnt"]:
                    vmcnt = loadsIssued - elementIdx - 1
                    vmComment = "{} = {} - {} - 1".format(vmcnt, loadsIssued, elementIdx)
                else:
                    waitStoreCnt = storesIssued if not kernel["GroupLoadStore"] else 0
                    vmcnt = loadsIssued - elementIdx + waitStoreCnt - 1
                    vmComment = "{} = {} - {} + {} - 1".format(vmcnt, loadsIssued, elementIdx, waitStoreCnt)

                maxVmcnt = writer.states.asmCaps["MaxVmcnt"]
                vmcnt = min(vmcnt, maxVmcnt)
                #print "wmvcnt=", vmcnt
                module.addSpaceLine()
                # if not atomicAddC:
                module.add(SWaitCnt(vmcnt=vmcnt, comment="wait C (interleaved) {}".format(vmComment)))

                # PreLoop LWVmcnt: When a vmcnt(cnt) is inserted here, means the GlobalLoad for PAP is finished
                # So the preLoopVmcntDict value is meaningless since we no longer need to wait in next PreLoop
                # And this only occurs when beta=true, so case must not be 2 or 3
                # assert self.currPreLoopVmcntCase not in self.preLoopVmcntDict, "PreLoopVmcntCase 2 or 3 shouldn't enter the beta true case"

            for vi in range(0, gwvw):
                dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
                sumIdxV = ss.elementSumIdx[elementIdx] + vi
                if kernel["ProblemType"]["ComputeDataType"].isHalf():
                    if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
                        if writer.states.asmCaps["HasWMMA"] and kernel["EnableMatrixInstructionStore"]:
                            dataV = ss.elementData[elementIdx] + int(vi / 2 * ss.cfg.numVgprsPerDataPerVI)
                            # if (vi % 2) == 0:
                            #         kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                            #                 "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                            # else:
                            if (vi % 2) != 0:
                                module.add(VLShiftRightB32(dst=vgpr(dataV), shiftHex=16, src=vgpr(dataV), \
                                    comment="shift 16bit to get next half of packed ValueC"))
                            # dataV+0 = new c = old c*beta + rC
                            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(sumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(sumIdxV)), \
                                comment="sum*alpha + C*beta"))
                        elif sumIdxV%2==0 or (not ss.cfg.halfDataRegPerVI and gwvw==1):
                            newSumIdxV = sumIdxV // 2 - writer.states.c.startVgprValu
                            # dataV+0 = new c = old c*beta
                            # kStr += inst("v_pk_mul_f16", vgpr(dataV), sgpr("Beta"), vgpr(dataV+0), \
                            #         "%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi))
                            # dataV+0 = new c = old c*beta + rC
                            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(newSumIdxV)), \
                                comment="sum*alpha + C*beta"))
                        else:
                            pass # add will have been done previously
                    else: # HPA
                        newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                        # dataV+0 = new c = old c*beta + rC
                        # src0 = beta = f32 = opsel 00
                        # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                        # src2 = sumIdxV = f32 = opsel 00
                        dataCExternal = ss.elementData[elementIdx] + vi//2
                        hi16 = (vi + gwvw*vc0) % 2
                        # TODO try to replace with add? need opsel for f16 src
                        # kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), sgpr("Beta"), \
                        # module.add(writer.states.mixinst(dst=vgpr("ValuC+%u"%newSumIdxV), src0=sgpr("Beta"), \
                        #     src1=vgpr(dataCExternal), src2=vgpr("ValuC+%u"%newSumIdxV), \
                        #     vop3=VOP3PModifiers(op_sel=[0,hi16,0], op_sel_hi=[0,1,0]),
                        #     comment="//C*=beta"))
                        module.add(writer.states.mixinst(dst=vgpr("ValuC+%u"%newSumIdxV), src0=1, \
                            src1=vgpr(dataCExternal), src2=vgpr("ValuC+%u"%newSumIdxV), \
                            vop3=VOP3PModifiers(op_sel=[0,hi16,0], op_sel_hi=[0,1,0]),
                            comment="//C*=beta"))
                        # kStr += inst(self.mixinst, vgpr("ValuC+%u"%sumIdxV), 1, \
                        #         vgpr(dataCExternal), vgpr("ValuC+%u"%sumIdxV), \
                        #         "op_sel:[0,%u,0] op_sel_hi:[0,1,0]" % (hi16), \
                        #         "//C*=beta")
                
                elif kernel["ProblemType"]["ComputeDataType"].isBFloat16():
                    if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                        # dataV+0 = new c = old c*beta + rC
                        # src0 = beta = f32 = opsel 00
                        # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
                        # src2 = sumIdxV = f32 = opsel 00
                        dataCExternal = ss.elementData[elementIdx] + vi//2
                        # if (vi%2) == 1:
                        #     kStr += inst("v_and_b32", vgpr(tmpVgpr), vgpr(dataCExternal), vgpr(vgprBf16Mask), "convert bf16 to fp32")
                        # else:
                        #     kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), "16", vgpr(dataCExternal), "convert bf16 to fp32" )
                        module.add(VCvtBF16toFP32(dst=(tmpVgpr), src=(dataCExternal), vgprMask=(cvtVgprStruct.vgprBf16Mask), vi=(vi)))
                        newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                        module.add(VAddF32(dst=vgpr("ValuC+%u"%sumIdxV), src0=vgpr("ValuC+%u"%sumIdxV), src1=vgpr(tmpVgpr), comment="accum partials"))

                # float8 precision
                elif kernel["ProblemType"]["DestDataType"].isFloat8():
                    if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
                        # Generate single f32 code if edge is detected.
                        isPK = False
                        if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
                            sb = SelectBit.BYTE_0 if self.gwvw == 1 else SelectBit.BYTE_2
                            module.add(VCvtFP8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
                        # Original packed route
                        elif vi%2 == 1:
                            continue
                        else:
                            isPK = True
                            sb = SelectBit.WORD_0 if vi == 0 else SelectBit.WORD_1
                            module.add(VCvtPkFP8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
                        module.add(SNop(waitState=0))
                        if kernel["ProblemType"]["ComputeDataType"].isSingle():
                            module.add(VAddF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr("ValuC+%u"%newSumIdxV), src1=vgpr(tmpVgpr), comment="accum partials"))
                            if isPK:
                                module.add(VAddF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr("ValuC+%u"%(newSumIdxV+1)), src1=vgpr(tmpVgpr+1), comment="accum partials"))

                # bfloat8 precision
                elif kernel["ProblemType"]["DestDataType"].isBFloat8():
                    if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
                        # Generate single f32 code if edge is detected.
                        isPK = False
                        if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
                            sb = SelectBit.BYTE_0 if self.gwvw == 1 else SelectBit.BYTE_2
                            module.add(VCvtBF8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
                        # Original packed route
                        elif vi%2 == 1:
                            continue
                        else:
                            isPK = True
                            sb = SelectBit.WORD_0 if vi == 0 else SelectBit.WORD_1
                            module.add(VCvtPkBF8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataV), sdwa=SDWAModifiers(src0_sel=sb)))
                        module.add(SNop(waitState=0))
                        if kernel["ProblemType"]["ComputeDataType"].isSingle():
                            module.add(VAddF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr("ValuC+%u"%newSumIdxV), src1=vgpr(tmpVgpr), comment="accum partials"))
                            if isPK:
                                module.add(VAddF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr("ValuC+%u"%(newSumIdxV+1)), src1=vgpr(tmpVgpr+1), comment="accum partials"))

                elif kernel["ProblemType"]["ComputeDataType"].isSingle():
                    newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                    module.add(VAddF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr("ValuC+%u"%newSumIdxV), src1=vgpr(dataV+0), comment="accum partials"))
                
                elif kernel["ProblemType"]["ComputeDataType"].isInt32():
                    newSumIdxV = sumIdxV - writer.states.c.startVgprValu
                    # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
                    # v_mad_i32_i24
                    module.add(VAddU32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=vgpr("ValuC+%u"%newSumIdxV), comment="accum partials"))

                elif kernel["ProblemType"]["ComputeDataType"].isDouble():
                    newSumIdxV = sumIdxV * 2 - writer.states.c.startVgprValu
                    # dataV+0 = new c = old c*beta
                    module.add(VAddF64(dst=vgpr("ValuC+%u"%(newSumIdxV*2),2), src0=vgpr("ValuC+%u"%(newSumIdxV*2),2), src1=vgpr(dataV+0,2), comment="accum partials"))

                # single precision complex
                elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
                    newSumIdxV = sumIdxV * 2 - writer.states.c.startVgprValu
                    module.add(VAddF32(dst=vgpr("ValuC+%u"%(newSumIdxV*2)), src0=vgpr("ValuC+%u"%(newSumIdxV*2)), src1=vgpr(dataV+0), comment="accum partials real"))
                    module.add(VAddF32(dst=vgpr("ValuC+%u"%(newSumIdxV*2+1)), src0=vgpr("ValuC+%u"%(newSumIdxV*2+1)), src1=vgpr(dataV+1), comment="accum partials imag"))

                # double precision complex
                elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
                    newSumIdxV = sumIdxV * 4 - self.parentWriter.states.c.startVgprValu
                    module.add(VAddF64(dst=vgpr("ValuC+%u"%(newSumIdxV*4+0),2), src0=vgpr("ValuC+%u"%(newSumIdxV*4+0),2), src1=vgpr(dataV+0,2), comment="accum partials real"))
                    module.add(VAddF64(dst=vgpr("ValuC+%u"%(newSumIdxV*4+2),2), src0=vgpr("ValuC+%u"%(newSumIdxV*4+2),2), src1=vgpr(dataV+2,2), comment="accum partials imag"))

        ########################################
        # AccVgpr write
        # if kernel.enabledSetPrioSplitLDS:
        #     kStr += inst("s_setprio", "0", "")
        if codeAccVgprWrite is not None:
            regsPerScalar = writer.states.bpeCinternal // writer.states.bpr # register per scalar
            # loop over store instructions within one batch
            for elementIdx in range(0, len(batchElements)):
                # loop over scalars within one store instruction
                for vi in range(0, gwvw):
                    # loop over registers within one scalar
                    for rIdx in range(0, regsPerScalar):
                        module.add(replaceHolder(codeAccVgprWrite.items().pop(0), ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - writer.states.c.startVgprValu))
                        # tempStr = str(codeAccVgprWrite.items().pop(0))
                        # kStr += tempStr.replace("__placeholder__", str(ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx))
                        # if kernel["StoreCInUnroll"] and not edge:
                        #     tempStr = tempStr.replace("__placeholder__",str(elementIdx*gwvw*regsPerScalar + regsPerScalar*vi + rIdx))
                        #     accVgprRead.addCode(tempStr.replace("ValuC","L2GC"))

            if not kernel["MIArchVgpr"]:
                module.add(SNop(1, "2 wait states required before reading vgpr"))

        #kStr += self.bomb(5)
        # if self.db["CheckStoreC"]>=0:
        #     useBuffer = kernel["BufferStore"]
        #     # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
        #     # Note - TODO- CheckStoreC also won't work for StoreRemap
        #     kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        #     if self.archCaps["SeparateVscnt"]:
        #         kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")
        #     for elementIdx in range(0, len(batchElements)):
        #         addr = ss.elementAddr[elementIdx].addrDVgpr
        #         sumIdx = ss.elementSumIdx[elementIdx]

        #         bps = kernel["ProblemType"]["DestDataType"].numBytes() * gwvw
        #         if kernel["BufferStore"]:
        #             addr0 = vgpr(addr)
        #             addr1 = sgpr("SrdC", 4)
        #         else:
        #             addr0 = vgpr(addr,2)
        #             addr1 = ""

        #         if kernel["ProblemType"]["DestDataType"].isHalf() or kernel["ProblemType"]["DestDataType"].isBFloat16():
        #             if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
        #                 kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
        #                                     addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=sumIdx%2).toStr()
        #             else:
        #                 kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
        #                                     addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False, hi16=0).toStr()
        #         elif kernel["ProblemType"]["DestDataType"].isInt32() or kernel["ProblemType"]["DestDataType"].isSingle():
        #             kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx, \
        #                                 addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        #         elif kernel["ProblemType"]["DestDataType"].isDouble() or kernel["ProblemType"]["DestDataType"].isSingleComplex() :
        #             kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
        #                                 addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        #         elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
        #             kStr += self.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
        #                                 addr0, addr1, soffset=0, offset=0, extraFields="", dtlNoDestVgpr=False).toStr()
        #     kStr += inst("s_waitcnt", "vmcnt(0)", "CheckStoreC, wait for stores to complete" )
        #     if self.archCaps["SeparateVscnt"]:
        #         kStr += inst("s_waitcnt_vscnt", "null", "0", "writes")

        #     # Add checks for expected values:
        #     kStr += inst("s_mov_b32", sgpr(tmpS01), self.db["CheckStoreC"], "expected value")
        #     for elementIdx in range(0, len(batchElements)):
        #         sumIdx = ss.elementSumIdx[elementIdx]
        #         # Need to fix for other types:
        #         assert (kernel["ProblemType"]["DestDataType"].isSingle() or kernel["ProblemType"]["DestDataType"].isInt32())
        #         kStr += self.assert_eq(vgpr(sumIdx), sgpr(tmpS01))


        if edge and (not kernel["BufferStore"]): # atomic or 
            # subsequent batch must start with full exec mask
            # BufferStore doesn't need exec since it used buffer range checking when
            # possible
            module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

        if writer.db["ConservativeWaitCnt"] & 0x40:
            module.add(SBarrier("debug"))
            module.add(SWaitCnt(vmcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
            module.add(SBarrier("debug"))
        ########################################
        # End Not Atomic
        ########################################

        # return registers to pool:
        lastData = -1
        for elementIdx in range(0, len(batchElements)):
            if not ss.sharedColDVgprs:
                addrCalc: AddrCalculation = ss.elementAddres[elementIdx]
                addrDVgpr = addrCalc.addrDVgpr
                addrCVgpr = addrCalc.addrCVgpr
                writer.vgprPool.checkIn(addrDVgpr)
                if addrCVgpr != addrDVgpr:
                    writer.vgprPool.checkIn(addrCVgpr)

            data = ss.elementData[elementIdx]
            if data != 0:
                if data != lastData:
                    writer.vgprPool.checkIn(data)
                lastData = data

        ss.firstBatch = False
        ss.checkInTempVgprC()

        if writer.states.serializedStore:
            module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

        # Update the store cnt to preLoopVmcntDict for Case2/3
        # (No need to update for Case0:'Undefined' or Case4:'OrdNLL_B1_Store')
        # if self.currPreLoopVmcntCase in self.preLoopVmcntDict:
        #     if not self.archCaps["SeparateVscnt"]:
        #         self.preLoopVmcntDict[self.currPreLoopVmcntCase] += storesIssued

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
    
    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Off storeBranches")
        return module
    
    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
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

    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK Basic storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct))
        return module
    
    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK Basic writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel))
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
        
    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK TwoTileOriginal storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel, vectorWidths, element, tmpVgpr, cvtVgprStruct))
        return module

    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK TwoTileOriginal writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel))
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

    def storeBranches(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct):
        module = Module("StreamK TwoTileDPFirst storeBranches")
        module.add(self.storeBranchesCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct))
        return module
    
    def writePartials(self, writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel):
        module = Module("StreamK TwoTileDPFirst writePartials")
        module.add(self.writePartialsCommon(writer, kernel, skPartialsLabel, vectorWidths, elements, tmpVgpr, cvtVgprStruct, endLabel))
        return module
