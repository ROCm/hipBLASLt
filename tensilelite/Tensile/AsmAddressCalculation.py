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

from .TensileInstructions import Module, EXEC, vgpr, sgpr, log2
from .TensileInstructions.Instructions import *
from .Common import globalParameters

##############################################################################
# Fields associated with computing address
##############################################################################
class AddrCalculation:
    # rowInc is number of rows to add to the base address
    # coord0Vgpr : This is VGPR that holds coord0.  Coord0 is element-space
    #    packed index for the 0 coordinate of the C/D matrix.
    # coord1Vgpr : VGPR which tracks the last coord1 calculation.
    #          If this is new coord1, just overwrite it with latest calc.
    def __init__(self, kernelWriter, ss, addrCVgpr, addrDVgpr, addrBiasVgpr, addrScaleDVgpr, element, \
        coordOffset0, coord1Vgpr, coordOffset1, rowInc, newCoord1):
        self.kernelWriter = kernelWriter

        # vgprs for address, could be more than one (for flat)
        self.addrDVgpr    = addrDVgpr
        self.addrCVgpr    = addrCVgpr
        self.addrBiasVgpr = addrBiasVgpr
        self.addrScaleDVgpr = addrScaleDVgpr
        self.coord1Vgpr = coord1Vgpr # vgpr that stores coord1Vgpr

        self.element = element
        self.coordOffset0 = coordOffset0
        self.coordOffset1 = coordOffset1
        self.rowInc = rowInc
        self.rowIncDirtyRowPtr = 0 # rowInc was used to modify rowPtr, need to recompute addr
        self.newCoord1 = newCoord1 # vgpr that stores newCoord1

        if ss.optSingleColVgpr:
            # optimized stores use the load offset for coordOffset0 calculations.
            self.biasOffset   = coordOffset0 * kernelWriter.states.bpeCinternal
            self.scaleDOffset   = coordOffset0 * kernelWriter.states.bpeCinternal
            self.globalOffset = coordOffset0 * kernelWriter.states.bpeCexternal
        else:
            # else non-opt stores include the coord0 offset into VGPR address calcs
            self.biasOffset   = 0
            self.scaleDOffset   = 0
            self.globalOffset = 0

    def addScaled(self, destV, src0, src1, scale1, tmpS01, comment=""):
        """
        Use minimally efficient instructions to add stride*scale
        """

        module = Module("addScaled")
        if scale1 == 1:
            module.add(VAddU32(dst=destV, src0=src0, src1=src1, comment=comment))
        else:
            module.add(SMulI32(dst=sgpr(tmpS01), src0=src1, src1=scale1, comment="scale stride"))
            module.add(VAddU32(dst=destV, src0=src0, src1=sgpr(tmpS01), comment=comment))
        return module


    def emitAddressCoordIncrement(self, kernel, ss, tmpVgpr, tmpS01, updateCoord1):
        """
        Emit code that computes the coord0 and coord1 for this element
        sets self.coord0Vgpr with the address that holds the coord0 value for this element.
        Input:
          - tmpVgpr is a 1 temporary VGPR used for coord0 calculation on edges
        """

        module = Module("emitAddressCoordIncrement")
        kw = self.kernelWriter
        (d1,d0,vc1,vc0) = self.element
        self.coord0Vgpr = None # will set below

        # module.addComment0("store addr=v%u coordOffset0=%u"% \
        #    (self.addr, self.coordOffset0))
        module.addComment0("(d1,vc1,d0,vc0)=(%u,%u,%u,%u)"\
            % (d1,vc1,d0,vc0))
        if ss.optSingleColVgpr:
            self.coord0Vgpr = kw.vgprs.coord0
        elif not ss.optSharedColVgpr or (d1 == vc1 == 0):
            # not share mode or first row always does the address calc math:

            if self.coordOffset0 == 0:
                self.coord0Vgpr = kw.vgprs.coord0
            elif self.coordOffset0 <= 64:
                self.coord0Vgpr = tmpVgpr
                module.add(VAddCOU32(dst=vgpr(self.coord0Vgpr), dst1=VCC(), src0=vgpr(kw.vgprs.coord0), src1=self.coordOffset0, \
                          comment="coord0.1: coord0 += d0*sg0*VW + vc0"))
            else:
                self.coord0Vgpr = tmpVgpr
                module.add(SMovB32(dst=sgpr(tmpS01), src=self.coordOffset0, comment="coordOffset0 d0=%u vc0=%u"%(d0, vc0)))
                module.add(VAddCOU32(dst=vgpr(self.coord0Vgpr), dst1=VCC(), src0=vgpr(kw.vgprs.coord0), src1=sgpr(tmpS01), \
                          comment="coord0.2: coord0 += d0*sg0*VW + vc0"))

            if self.newCoord1:
                if not kernel["BufferStore"] or updateCoord1:
                    if self.rowInc== 0:
                        None
                    elif self.rowInc <= 64:
                        # rowInc fits in instruction:
                        module.add(VAddCOU32(dst=vgpr(self.coord1Vgpr), dst1=VCC(), \
                                  src0=vgpr(self.kernelWriter.vgprs.coord1), src1=self.rowInc, \
                                  comment="coord1.1: coord1Vgpr += d1*sg1*VW + vc1"))
                    else:
                        module.add(SMovB32(dst=sgpr(tmpS01), src=self.rowInc, comment="rowInc d1=%u vc1=%u"%(d0, vc0)))
                        module.add(VAddCOU32(dst=vgpr(self.coord1Vgpr), dst1=VCC(), \
                                  src0=vgpr(self.kernelWriter.vgprs.coord1), src1=sgpr(tmpS01), \
                                  comment="coord1.2: coord1 += d1*sg1*VW + vc1"))
        return module

    # storeChar is 'C' or 'D'
    # elementVgpr is coord0Vgpr*strideCD0, or optimized to just coord0Vgpr if strideCD0 is unit const
    def emitExtractAndScalePackedDims(self, kernel, ss, tmpVgpr, storeChar):
        module = Module("emitExtractAndScalePackedDims")
        kw = self.kernelWriter
        packedIndices = kernel["PackedC0IndicesX"]
        packedBits = self.coord0Vgpr # start with coord0, will move to temp below
        rowPtr = kw.vgprs.cinRowPtr if (storeChar == 'C') else kw.vgprs.coutRowPtr
        addrVgpr = self.addrCVgpr if (storeChar == 'C') else self.addrDVgpr

        for i,idx in enumerate(packedIndices[:-1]):
            # vgprTmp assignments:
            #   - tmp+0 may be the incoming packed coordinate 0, used on replay too
            #   - tmp+1 is DIV output
            #   - tmp+2 is scratch
            idxChar= globalParameters["IndexChars"][idx]
            module.addComment0("extract %s"%kw.sizeRef(idx))
            assert(tmpVgpr+1 != packedBits) # bad since we still need packedBits below for remainder (can't overwrite here)
            module.add(MacroInstruction("V_MAGIC_DIV", \
                           args=[tmpVgpr+1, vgpr(packedBits), sgpr("MagicNumberSize%s"%idxChar), \
                           sgpr("MagicShiftSize%s"%idxChar), sgpr("MagicAbitSize%s"%idxChar) if kernel["MagicDivAlg"]==2 else "0"]))
            # tmpVgpr+1 returns the quotient, tmpVgpr+2 is overwritten

            # compute remainder, packedBits % sizeIdx - this is the 'extracted' index that must be scaled
            # remainder is mul and sub
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+2), src0=vgpr(tmpVgpr+1), src1=kw.sizeRef(idx), \
                           comment="remainder part 1"))
            module.add(VSubU32(dst=vgpr(tmpVgpr+2), src0=vgpr(packedBits), src1=vgpr(tmpVgpr+2),
                           comment="remainder part 2"))

            if i==0:
                module.add(VMulLOU32(dst=vgpr(addrVgpr), src0=vgpr(tmpVgpr+2), \
                          src1=kw.strideRef(storeChar, idx), comment="addrCalc <- scaled extracted dim"))
            else:
                module.add(VMulLOU32(dst=vgpr(tmpVgpr+2), src0=vgpr(tmpVgpr+2), \
                          src1=kw.strideRef(storeChar, idx), comment="scale extracted dim"))
                module.add(VAddU32(dst=vgpr(addrVgpr), src0=vgpr(addrVgpr), \
                          src1=vgpr(tmpVgpr+2), comment="addrCalc += scaled extracted dim "))

            if i < len(packedIndices)-2:
                # TODO - might be able to eliminate this
                module.add(VMovB32(dst=vgpr(tmpVgpr+0), src=vgpr(tmpVgpr+1), \
                          comment="Copy remaining bits for next divide"))
                packedBits = tmpVgpr+0

        if len(packedIndices)>1:
            # if we unpacked something, then scale it to BPE
            module.addComment0("extract final %s"%kw.sizeRef(packedIndices[-1]))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+2), src0=vgpr(tmpVgpr+1), \
                      src1=kw.strideRef(storeChar, packedIndices[-1]), comment="scale final extracted dim"))
            module.add(VAddU32(dst=vgpr(addrVgpr), src0=vgpr(addrVgpr), \
                      src1=vgpr(tmpVgpr+2), comment="addrCalc += scaled extracted dim "))

            module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                      src0=vgpr(rowPtr), \
                      src1=vgpr(addrVgpr), \
                      shiftHex=hex(log2(kw.states.bpeCexternal)), \
                      comment="packed: add rowPtr and scaleToBpe"))

        return module

    def emitScaleToBpe(self, kernel, ss, tmpVgpr, tmpSgpr, singleUpdate, tc):
        """
        Needs 3 temporary VGPRs
        """

        module = Module("emitScaleToBpe")
        kw = self.kernelWriter
        (d1,d0,vc1,vc0) = self.element
        rowPtr = kw.vgprs.cinRowPtr if (tc == 'C') else kw.vgprs.coutRowPtr
        addrVgpr = self.addrCVgpr if (tc == 'C') else self.addrDVgpr
        # set when we generate code that updates the address
        # optSingleColVgpr and optSharedColVgpr attempt to minimize these updates
        updatedAddr = False

        # scale and set final address:
        stride0 = kw.strideRef('D', 0) if ((tc == 'Bias') or (tc == 'ScaleD')) else kw.strideRef(tc, 0)
        if kw.isConstUnitStride(stride0):
            elementVgpr = self.coord0Vgpr
        else:
            module.add(VMulLOU32(dst=vgpr(addrVgpr), \
                src0=vgpr(self.coord0Vgpr), \
                src1=stride0, \
                comment="scale element by non-unit stride"))
            elementVgpr = addrVgpr

        if ss.optSingleColVgpr:
            # This is first element in the first batch, create a byte address that will
            # be re-used by subsequent elements:
            # if this element is firstInBatch - may need to set up a bpe-scaled row pointer for the batch:
            #  - need row-ptr start of each batch
            assert (kw.vgprs.coord0 == self.coord0Vgpr) # elementAddr assignment above assumes these are the same
            if singleUpdate:
                updatedAddr = True
                singleColAddrUpdated = ss.singleColCAddrUpdated if (tc == 'C') else ss.singleColDAddrUpdated
                if not singleColAddrUpdated or not ss.optSrdIncForRow:
                    if tc == 'Bias' and kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
                        module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                        module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                        module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                                shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                                src=vgpr(self.addrBiasVgpr), \
                                                comment="Bias address scaled by BPE"))
                        return module
                    if tc == 'ScaleD' and kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
                        module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleDVgpr), \
                                                 shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                                 src=vgpr(self.coord0Vgpr), \
                                                comment="ScaleD address scaled by BPE"))
                        return module
                    if tc == 'C':
                        ss.singleColCAddrUpdated = True
                    else:
                        ss.singleColDAddrUpdated = True
                    module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                      src0=vgpr(rowPtr), \
                      src1=vgpr(elementVgpr), \
                      shiftHex=hex(log2(kw.states.bpeCexternal)), \
                      comment="optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=%d, coord0Vgpr=%d"%(kw.vgprs.coord0, self.coord0Vgpr)))
        elif ss.optSharedColVgpr:
            # Need an address calculation for the first address in each row:
            if d1==0 and vc1==0:
                if tc == 'Bias' and kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
                    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                    module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                            shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                            src=vgpr(self.addrBiasVgpr), \
                                            comment="Bias address scaled by BPE"))
                    return module
                if tc == 'ScaleD' and kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
                    module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleDVgpr), \
                                             shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                             src=vgpr(self.coord0Vgpr), \
                                            comment="ScaleD address scaled by BPE"))
                    return module
                packedIndices = kernel["PackedC0IndicesX"]
                if len(packedIndices) > 1:
                    updatedAddr = True
                    module.add(self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc))
                else:
                    updatedAddr = True
                    module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                      src0=vgpr(rowPtr), \
                      src1=vgpr(elementVgpr), \
                      shiftHex=hex(log2(kw.states.bpeCexternal)), \
                      comment="optSharedColVgpr scaleToBpe for first row: col addr <- cinRowPtr + coord0, scaled by BPE"))
        else:
            # Generate final address calculation (to bytes) for each element
            # The unpacking takes 8-10 instructions so could be worth optimizing someday :
            # each col has same offset so could create a class to hold column-specific state including
            # the byte address offset for that col and the mask in/out.
            if tc == 'Bias' and kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
                module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                        shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                        src=vgpr(self.addrBiasVgpr), \
                                        comment="Bias address scaled by BPE"))
                return module
            if tc == 'ScaleD' and kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
                module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleDVgpr), \
                                         shiftHex=hex(log2(kw.states.bpeCinternal)), \
                                         src=vgpr(self.coord0Vgpr), \
                                        comment="ScaleD address scaled by BPE"))
                return module
            packedIndices = kernel["PackedC0IndicesX"]
            if len(packedIndices) > 1:
                updatedAddr = True
                module.add(self.emitExtractAndScalePackedDims(kernel, ss, tmpVgpr, tc))
            else:
                updatedAddr = True
                module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
                    src0=vgpr(rowPtr), \
                    src1=vgpr(elementVgpr), \
                    shiftHex=hex(log2(kw.states.bpeCexternal)), \
                    comment="scaleToBpe: accumulate d0 lower and *= bpe into Cin addr"))

        # if not optSrdIncForRow then we may have moved the row pointer
        # and depending on paths above may not have refreshed addrVgpr already.
        # if so - do it here:
        if self.rowIncDirtyRowPtr and not updatedAddr:
            module.add(VAddLShiftLeftU32(dst=vgpr(addrVgpr), \
              src0=vgpr(rowPtr), \
              src1=vgpr(kw.vgprs.coord0), \
              shiftHex=hex(log2(kw.states.bpeCexternal)), \
              comment="scaleToBpe: Update address with new rowPtr"))

        return module

    def edgeProtectCode(self, kernel, edge, beta, atomic, mask, tmpSgpr):
        """
        Generate code to protect address offset in edge case
        """

        module = Module("edgeProtectCode")
        kw = self.kernelWriter
        tmpS01 = tmpSgpr
        tmpS23 = tmpSgpr+self.kernelWriter.states.laneSGPRCount

        laneSGPRCount = self.kernelWriter.states.laneSGPRCount
        wavefrontSize = kernel["WavefrontSize"]

        # Now do the edge check and compute the address in bytes:
        if kernel["BufferStore"]:
            if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and beta)):
                # Set address to -1 if OOB on either dimension
                # and only check the x/coord0 index here, save a couple inst
                sizeBoundary = [0,0]
                sizeBoundary[0] = \
                    sgpr("PackedSize0") if len(kernel["PackedC0IndicesX"]) > 1 \
                    else kw.sizeRef(kernel["ProblemType"]["Index0"])
                sizeBoundary[1] = \
                    sgpr("PackedSize1") if len(kernel["PackedC1IndicesX"]) > 1 \
                    else kw.sizeRef(kernel["ProblemType"]["Index1"])

                module.add(VCmpLtU32(dst=sgpr(tmpS01,laneSGPRCount), src0=vgpr(self.coord0Vgpr), src1=sizeBoundary[0], comment="coord0 < size0" ))
                module.add(VCmpLtU32(dst=sgpr(mask,laneSGPRCount), src0=vgpr(self.coord1Vgpr), src1=sizeBoundary[1], comment="coord1 < size1" ))
                SAndX = SAndB64 if wavefrontSize == 64 else SAndB32
                module.add(SAndX(dst=sgpr(mask,laneSGPRCount), src0=sgpr(tmpS01,laneSGPRCount), src1=sgpr(mask,laneSGPRCount), comment="in0 && in1" ))
        else:
            module.add(VCmpLtU32(dst=sgpr(tmpS01,laneSGPRCount), src0=vgpr(self.coord0Vgpr), src1=sgpr("SizesFree+0"), comment="coord0 < size0" ))
            module.add(VCmpLtU32(dst=sgpr(tmpS23,laneSGPRCount), src0=vgpr(self.coord1Vgpr), src1=sgpr("SizesFree+1"), comment="coord1 < size1" ))
            SAndX = SAndB64 if wavefrontSize == 64 else SAndB32
            module.add(SAndX(dst=sgpr(mask,laneSGPRCount), src0=sgpr(tmpS01,laneSGPRCount), src1=sgpr(tmpS23,laneSGPRCount), comment="in0 && in1" ))

            if (beta or atomic):
                SMovX = SMovB64 if wavefrontSize == 64 else SMovB32
                module.add(SMovX(dst=EXEC(), src=sgpr(mask,laneSGPRCount), comment="sgprs -> exec" ))

        return module

    # TODO - mask should be part of AddrCalc state not passed as parm
    def emitAddressSetupCode(self, kernel, tPB, ss, tmpVgpr, tmpS01, edge, beta, atomic, elementIdx, addrVgpr):
        """
        Generate code to set up the address vgpr
        Input:
          tmpVgpr : two temp vgprs
        Output:
          Returns kStr with appropriate setup code
          Sets self.coord0Vgpr with vgpr that contains the coord0 for this element.  This enables
            optimization - if no setup code is required the coord0 can be the input.
        """

        module = Module("emitAddressSetupCode")
        kw = self.kernelWriter

        updateCoord1 = (edge or len(kernel["PackedC1IndicesX"]) > 1)
        module.add(self.emitAddressCoordIncrement(kernel, ss, tmpVgpr, tmpS01, updateCoord1))

        # calculate flat load offset
        if not kernel["BufferStore"]:
            # flat: in-bounds exec mask
            # global offset macro (requires 3 tmpVgpr)
            # final address = C + index*bytes
            params = ["%u" % addrVgpr]
            for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
                if i == kernel["ProblemType"]["Index0"]:
                    params.append("%s" % (self.coord0Vgpr))
                elif i == kernel["ProblemType"]["Index1"]:
                    params.append("%s" % (self.coord1Vgpr))
                else: # just a group index
                    params.append("sgprWorkGroup%u"%i)
            params.append("%s" % (tmpVgpr+2))
            module.add(MacroInstruction(name="GLOBAL_OFFSET_C", args=params))
            module.add(VMovB32(dst=vgpr(tmpVgpr+2), src=vgpr(addrVgpr+0), comment="temp store offset 0"))
            module.add(VMovB32(dst=vgpr(tmpVgpr+3), src=vgpr(addrVgpr+1), comment="temp store offset 1"))

        # Move the row ptr VGPR
        # optSrdIncForRow moves the SRD so don't move here
        if not ss.optSrdIncForRow and kernel["BufferStore"]:
            if self.rowInc > 0:
                self.rowIncDirtyRowPtr = 1
                #assert (not kernel["ProblemType"]["UseInitialStridesCD"])
                module.addComment1("Fix for UseInitialStridesCD, emitAddressSetupCode")

                if len(kernel["PackedC1IndicesX"]) == 1:
                    strideChar = self.kernelWriter.states.indexChars[kernel["PackedC1IndicesX"][0]]
                    module.add(self.addScaled(vgpr(kw.vgprs.cinRowPtr),  vgpr(kw.vgprs.cinRowPtr),  \
                              sgpr("StrideC%s"%strideChar), self.rowInc, tmpS01, "ROWINC- Move cinRowPtr to next row"))
                    module.add(self.addScaled(vgpr(kw.vgprs.coutRowPtr), vgpr(kw.vgprs.coutRowPtr), \
                              sgpr("StrideD%s"%strideChar), self.rowInc, tmpS01, "Move coutRowPtr to next row"))
                elif len(kernel["PackedC1IndicesX"]) > 1:
                    module.add(kw.extractPackedCoord1ToRowStart(kernel, kernel["PackedC1IndicesX"] , self.coord1Vgpr, 'D'))

        # Shift Pointer for MFMA:
        #   For MFMA shift pointer, correct data is stored in another thread.
        #   Therefore, MFMA cannot use v_mov to amend store data
        #   It needs to modify the coord1 of thread directly.
        if (not kernel["SourceSwap"]) and (not kernel["GuaranteeNoPartialB"]) and tPB["rtv"] and kernel["EnableMatrixInstruction"] and edge:
            (d1,d0,vc1,vc0) = self.element
            if (d1 == vc1 == d0 == vc0 == 0) or self.newCoord1:
                sgprCnt = self.kernelWriter.states.laneSGPRCount
                waveSize = kernel["WavefrontSize"]
                packedC1 = kernel["PackedC1IndicesX"]
                strideC1 = "StrideC%s" % (kw.states.indexChars[packedC1[0]])
                strideD1 = "StrideD%s" % (kw.states.indexChars[packedC1[0]])

                module.addComment1("shift vector components d1")
                vw = kernel["GlobalLoadVectorWidthB"]
                vTmp1 = tmpVgpr
                vTmp2 = tmpVgpr+1
                sTmp1 = tmpS01
                sTmp2 = tmpS01+sgprCnt
                # check conditions
                module.add(VBfiB32(dst=vgpr(vTmp1), src0=vw-1, src1=0, src2=vgpr(self.coord1Vgpr), comment="coord1 & ~(vw-1)"))
                module.add(VBfiB32(dst=vgpr(vTmp2), src0=vw-1, src1=0, src2=sgpr("SizesFree+%u"%tPB["idx"]), comment="sizeFree & ~(vw-1)"))
                module.add(VCmpEQU32(dst=sgpr(sTmp1,sgprCnt), src0=vgpr(vTmp1), src1=vgpr(vTmp2), comment="if coord1 is in edge glvw"))
                module.add(VAndB32(dst=vgpr(vTmp2), src0=sgpr("SizesFree+%u"%tPB["idx"]), src1=vw-1, comment="sizeFree mod VW"))
                module.add(VCmpGtU32(dst=sgpr(sTmp2,sgprCnt), src0=vgpr(vTmp2), src1=0, comment="this problem is not multiple size of glvw"))
                SAndBX = SAndB64 if waveSize == 64 else SAndB32
                module.add(SAndBX(dst=sgpr(sTmp1,sgprCnt), src0=sgpr(sTmp1,sgprCnt), src1=sgpr(sTmp2,sgprCnt), comment="AND both conditions"))
                # calculate new coord
                module.add(VAddU32(dst=vgpr(vTmp1), src0=vgpr(self.coord1Vgpr), src1=vgpr(vTmp2), comment="shift coord1"))
                module.add(VBfiB32(dst=vgpr(vTmp1), src0=vw-1, src1=vgpr(vTmp1), src2=sgpr("SizesFree+%u"%tPB["idx"]), comment="new coord1 = (shift coord1 & (vw-1)) |  (sizeFree & ~(vw-1))"))
                module.add(VSubI32(dst=vgpr(vTmp2), src0=vgpr(vTmp1), src1=vgpr(self.coord1Vgpr), comment="shift how many column"))
                module.add(VCndMaskB32(dst=vgpr(self.coord1Vgpr), src0=vgpr(self.coord1Vgpr), src1=vgpr(vTmp1), \
                              src2=sgpr(sTmp1,sgprCnt), comment="set new coord1 if meet conditions" ))

                module.add(VMadI32I24(dst=vgpr(vTmp1), src0=sgpr(strideC1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.cinRowPtr), \
                             comment="new rowStart address += shift column * StridesC"))
                module.add(VCndMaskB32(dst=vgpr(kw.vgprs.cinRowPtr), src0=vgpr(kw.vgprs.cinRowPtr), src1=vgpr(vTmp1), src2=sgpr(sTmp1,sgprCnt), \
                             comment="set new rowStart if meet conditions" ))
                module.add(VMadI32I24(dst=vgpr(vTmp1), src0=sgpr(strideD1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.coutRowPtr), \
                             comment="new rowStart address += shift column * StridesD"))
                module.add(VCndMaskB32(dst=vgpr(kw.vgprs.coutRowPtr), src0=vgpr(kw.vgprs.coutRowPtr), src1=vgpr(vTmp1), src2=sgpr(sTmp1,sgprCnt), \
                             comment="set new rowStart if meet conditions" ))

                if kernel["StoreRemapVectorWidth"]:
                    ldsPad = max(kernel["StoreRemapVectorWidth"],kernel["MIOutputVectorWidth"])
                    module.add(VMovB32(dst=vgpr(vTmp1), src=hex((kernel["MacroTile0"]+ldsPad)*kw.states.bpeCexternal), \
                                comment="lds byte stride = (MT0 + PAD) * bpe"))
                    module.add(VMadI32I24(dst=vgpr(vTmp1), src0=vgpr(vTmp1), src1=vgpr(vTmp2), src2=vgpr(kw.vgprs.storeRemapLW), \
                                comment="new lds write address += shift column * Lds byte Stride"))
                    module.add(VCndMaskB32(dst=vgpr(kw.vgprs.storeRemapLW), src0=vgpr(kw.vgprs.storeRemapLW), src1=vgpr(vTmp1), \
                                  src2=sgpr(sTmp1,sgprCnt), comment="set new rowStart if meet conditions" ))

                module.addSpaceLine()

        return module

    def emitLdChange(self, kernel, ss, tc, edge, beta, mask, singleUpdate, tmpVgpr, tmpSgpr, addrVgpr, BufAddr):
        """
        Generate code for final C read/D write address
        """

        laneSGPRCount = self.kernelWriter.states.laneSGPRCount

        module = Module("emitLdChange")
        if kernel["BufferStore"]:
            module.add(self.emitScaleToBpe(kernel, ss, tmpVgpr, tmpSgpr, singleUpdate, tc))
            if edge and (not kernel["StoreRemapVectorWidth"] or (kernel["StoreRemapVectorWidth"] and beta)) and \
                ((tc != 'Bias') and (tc != 'ScaleD')):
                module.add(VCndMaskB32(dst=vgpr(addrVgpr), src0=-1, src1=vgpr(addrVgpr), \
                               src2=sgpr(mask,laneSGPRCount), comment="LD%s clip if OOB. offset" % tc ))
        else:
            if tc == 'Bias' and kernel["ProblemType"]["UseBias"] and (kernel["GlobalSplitU"] == 1):
                module.add(SMulI32(dst=sgpr(tmpSgpr), src0=kernel["MacroTile0"], src1=sgpr("WorkGroup0"), comment="wgp0 * MT0"))
                module.add(VSubU32(dst=vgpr(self.addrBiasVgpr), src0=vgpr(self.coord0Vgpr), src1=sgpr(tmpSgpr)))
                module.add(VLShiftLeftB32(dst=vgpr(self.addrBiasVgpr), \
                                        shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                        src=vgpr(self.addrBiasVgpr), \
                                        comment="Bias address scaled by BPE"))
            if tc == 'ScaleD' and kernel["ProblemType"]["UseScaleD"] and (kernel["GlobalSplitU"] == 1):
                module.add(VLShiftLeftB32(dst=vgpr(self.addrScaleDVgpr), \
                                        shiftHex=hex(log2(self.kernelWriter.states.bpeCinternal)), \
                                        src=vgpr(self.coord0Vgpr), \
                                        comment="ScaleD address scaled by BPE"))

            else:
                # store a copy of the offset in 2 of the tmpVgpr for D
                module.add(VAddCOU32(dst=vgpr(addrVgpr+0), dst1=VCC(), src0=vgpr(BufAddr+0), src1=vgpr(tmpVgpr+2), \
                            comment="addrVgpr = C(D) + index*bytes (lo)" ))
                module.add(VAddCCOU32(dst=vgpr(addrVgpr+1), dst1=VCC(), src0=vgpr(BufAddr+1), src1=vgpr(tmpVgpr+3), \
                            src2=VCC(), comment="addrVgpr = C(D) + index*bytes (hi)"))
        return module

    def incrementToNextRow(self, kernel, tc, ss, stmp):
        """
        Generate code to move to the next row(s)
        If optSrdIncForRow, this will move the SRD forward
        If not, this could generate some other instructions
        """

        module = Module("incrementToNextRow")
        numRows = self.rowInc
        tmpBpe = self.kernelWriter.states.bpeCexternal
        if ss.optSrdIncForRow:
            if numRows:
                packedC1 = kernel["PackedC1IndicesX"]
                assert(len(packedC1) == 1)  # would need to extract each dim and scale
                strideCD1 = "Stride%s%s"%(tc,self.kernelWriter.states.indexChars[packedC1[0]])
                if numRows > 1:
                    module.add(SMulI32(dst=sgpr(stmp), \
                                src0=sgpr(strideCD1), \
                                src1=numRows*tmpBpe, \
                                comment="scale Stride%s *= numRows(%u) * bpe"%(tc,numRows)))
                else:
                    module.add(SLShiftLeftB32(dst=sgpr(stmp), \
                                src=sgpr(strideCD1), \
                                shiftHex=log2(tmpBpe), \
                                comment="incToNextRow: Scale by BPE"))

                module.add(SAddU32(dst=sgpr("Srd%s+0"%(tc)), \
                                    src0=sgpr("Srd%s+0"%(tc)), \
                                    src1=sgpr(stmp), \
                                    comment="incToNextRow: gra SRD += inc(lower)" ))
                module.add(SAddCU32(dst=sgpr("Srd%s+1"%(tc)), \
                                    src0=sgpr("Srd%s+1"%(tc)), \
                                    src1=0, \
                                    comment="incToNextRow: gra SRD += inc(upper)" ))

            None

        return module
