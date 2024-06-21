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

from ..TensileInstructions import Module, Label, RegisterPoolResource, SAddU32, SAddCU32, SCmpEQU32, SCBranchSCC1, \
    scalarUInt32DivideAndRemainder, SMovB32, SMulI32, SBranch, SMovB64, SLShiftRightB32, sgpr, log2, \
    SCmpLtU32, SCMovB32, SSubU32, SLShiftLeftB64, SCBranchSCC0, fastdeepcopy, Instruction, SCmpLgU32
from ..Component import Component
import abc

class GSU(Component):
    """
    GSU block.
    """
    @abc.abstractmethod
    def graWorkGroup(self, writer, kernel):
        pass

    @abc.abstractmethod
    def computeLoadSrd(self, writer, kernel, tP, stmp, tileStart):
        pass

    @abc.abstractmethod
    def graIncrements(self, writer, kernel, loopIdx, tP):
        pass

    def graIncrementsCommon(self, writer, loopIdx, tc, stride, m):
        module = Module("GSU Common graIncrements")

        # multiply by stride, optimizing if unit stride
        if writer.isConstUnitStride(stride):
            module.add(SMovB32(dst=sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), src=m, \
                comment="incr%s (unrollIdx)"%(tc) ))
        else:
            module.add(SMulI32(dst=sgpr("GlobalReadIncs%s+%u"%(tc, loopIdx)), \
                src0=m, src1=stride, \
                comment="incr%s unrollIdx)"%(tc) ))
        
        return module
    
    @abc.abstractmethod
    def calculateLoopNumIter(self, writer, kernel, loopCounterName, tmpSgprInfo):
        pass

    @abc.abstractmethod
    def computeStoreSrdStart(self, writer, kernel):
        pass

    @abc.abstractmethod
    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack):
        pass

    @abc.abstractmethod
    def tailLoopNumIter(self, writer, kernel, loopCounter):
        pass

    @abc.abstractmethod
    def setupNewTile(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        pass

    def graIncrementsAB(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        module = Module("GSU Common graIncrementsAB")

        for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
            module.add(writer.graIncrements(kernel, i, tensorParametersA))
        if kernel["ProblemType"]["Sparse"] and not kernel["DirectToVgprSparseMetadata"]:
            module.addComment1("global read addresses: increments metadata")
            for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
                module.add(writer.graIncrements(kernel, i, tPM))
        module.addComment1("global read addresses: increments b")
        for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
            module.add(writer.graIncrements(kernel, i, tensorParametersB))
        
        return module

class GSUOff(GSU):
    kernel = {"GlobalSplitU": 0}

    def __call__(self):
        assert(0)

    def graWorkGroup(self, writer, kernel):
        module = Module("GSU Off graWorkGroup")
        return module
    
    def computeLoadSrd(self, writer, kernel, tP, stmp, tileStart):
        module = Module("GSU Off computeLoadSrd")
        return module
    
    def graIncrements(self, writer, kernel, loopIdx, tP):
        module = Module("GSU Off graIncrements")

        tc = tP["tensorChar"]
        dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
        stride = writer.strideRef(tc, dimIdx)
        isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

        m = "DepthU*Bpe%s"%(tc)
        if isMirrorIdx:
          m = "-%s"%(m)

        module.add(self.graIncrementsCommon(writer, loopIdx, tc, stride, m))

        return module
    
    def calculateLoopNumIter(self, writer, kernel, loopCounterName, tmpSgprInfo):
        module = Module("GSU Off calculateLoopNumIter")
        return module
    
    def computeStoreSrdStart(self, writer, kernel):
        module = Module("GSU Off computeStoreSrdStart")
        return module

    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack):
        module = Module("GSU Off noLoadLoop")
        return module
    
    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("GSU Off tailLoopNumIter")
        return module

    def setupNewTile(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        module = Module("GSU Off setupNewTile")

        module.add(self.graIncrementsAB(writer, kernel, tensorParametersA, tensorParametersB, tPM))
        
        return module

class GSUOn(GSU):

    @classmethod
    def matches(cls, writer, debug=False):
        return writer.states.kernel["GlobalSplitU"] > 0
    
    def __call__(self):
        assert(0)

    def graWorkGroup(self, writer, kernel):
        module = Module("GSU On graWorkGroup")

        gsuLabel    = Label(label=writer.labels.getNameInc("GSU"), comment="")
        gsuLabelEnd = Label(label=writer.labels.getNameInc("GSU_End"), comment="")
        module.add(SCmpEQU32(src0=sgpr("GSU"), src1=1, comment="GSU == 1 ?"))
        module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
        module.addComment("GSU-not-WGMapRR :nwg1 = (size%s + MT%s - 1) / MT%s;" \
            % (writer.states.tileChar1, writer.states.tileChar1, writer.states.tileChar1))

        # gsuSumIdx = wg1 % GSU
        # wg1       = wg1 / GSU
        divisor = "WorkGroup1"
        tmpVgpr = writer.vgprPool.checkOut(2, "tmp")
        tmpVgprRes = RegisterPoolResource(idx=tmpVgpr, size=2)
        module.add(scalarUInt32DivideAndRemainder("WorkGroup1", divisor, "GSU", "GSUSumIdx", tmpVgprRes, wavewidth=kernel["WavefrontSize"]))
        writer.vgprPool.checkIn(tmpVgpr)
        module.add(SMovB32(dst=sgpr("GSULog2BpeC"), src=log2(int(writer.states.bpr * kernel["ProblemType"]["DestDataType"].numRegisters()))))
        module.add(SMovB32(dst=sgpr("GSULog2BpeD"), src=log2(writer.states.bpeCinternal)))

        module.add(SBranch(gsuLabelEnd.getLabelName()))
        module.add(gsuLabel)
        module.add(SMovB64(dst=sgpr("GSUSumIdx", 2), src=0, comment="Set GSUSumIdx to 0"))
        module.add(SMovB32(dst=sgpr("GSULog2BpeC"), src=log2(writer.states.bpeCexternalGSU1)))
        module.add(SMovB32(dst=sgpr("GSULog2BpeD"), src=log2(writer.states.bpeCexternalGSU1)))
        module.add(gsuLabelEnd)

        return module
    
    def computeLoadSrd(self, writer, kernel, tP, stmp, tileStart):
        module = Module("GSU On computeLoadSrd")

        tc = tP["tensorChar"]
        depthU = kernel["DepthU"]
        gsuOffset_str = "gsuOffset = DepthU*bpeGR*GSUSumIdx"
        divider = 1
        if kernel["ProblemType"]["Sparse"]:
            if (kernel["ProblemType"]["Sparse"] == 2 and tP["isB"]) or \
              (kernel["ProblemType"]["Sparse"] == 1 and tP["isA"]) :
                divider = 2
            elif tP["isM"]:
                divider = 8
            if divider != 1:
                depthU = depthU // divider
                gsuOffset_str = "gsuOffset = DepthU/%s*bpeGR*GSUSumIdx"%(divider)
        module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(stmp+0), sgpr(stmp+1), depthU, sgpr("GSUSumIdx"), gsuOffset_str))

        unrollSummation = [ i for i in tP["ia"] if i in kernel["ProblemType"]["IndicesSummation"] ]
        stride = writer.strideRef(tc,unrollSummation[-1])
        if tP["tlu"] and not writer.isConstUnitStride(stride):
            # non-transpose case, unroll is in perp dim and should be scaled by unroll Stride
            module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(stmp), sgpr(stmp+1), sgpr(stmp+0), \
                stride, "tlu=1, scaled unroll-offset by stride"))

        module.add(SAddU32(dst=sgpr(tileStart+0), src0=sgpr(tileStart+0), src1=sgpr(stmp+0), comment="accum GsuOffset term to tilestart"))
        module.add(SAddCU32(dst=sgpr(tileStart+1), src0=sgpr(tileStart+1), src1=sgpr(stmp+1), comment="accum GsuOffset term to tilestart"))

        return module
    
    def graIncrements(self, writer, kernel, loopIdx, tP):
        module = Module("GSU On graIncrements")

        tc = tP["tensorChar"]
        dimIdx = kernel["ProblemType"]["IndicesSummation"][loopIdx] # dimension index
        stride = writer.strideRef(tc, dimIdx)
        isMirrorIdx = dimIdx in kernel["ProblemType"]["MirrorDims%s"%tc]

        with writer.allocTmpSgpr(1) as tmpSgprInfo:
            gsuSgpr = tmpSgprInfo.idx

            tcGR = tc if tc == "Metadata" else (tc + "GR")
            module.add(SMulI32(dst=sgpr(gsuSgpr), src0=sgpr("GSU"), src1="DepthU*Bpe%s"%(tcGR)))

            if kernel["ProblemType"]["Sparse"]:
                if tP["is_sparse"]:
                    module.add(SLShiftRightB32(dst=sgpr(gsuSgpr), shiftHex=hex(log2(2)), src=sgpr(gsuSgpr)))
                elif tP["isM"]:
                    module.add(SLShiftRightB32(dst=sgpr(gsuSgpr), shiftHex=hex(log2(8)), src=sgpr(gsuSgpr)))

            m = sgpr(gsuSgpr)

            if isMirrorIdx:
                m.setMinus(True)

            module.add(self.graIncrementsCommon(writer, loopIdx, tc, stride, m))

        return module
    
    def calculateLoopNumIter(self, writer, kernel, loopCounterName, tmpSgprInfo):
        module = Module("GSU On calculateLoopNumIter")

        # if GSU numIter++ if gsuSumIdx < remainder
        gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
        module.add(SCmpEQU32(src0=sgpr("GSU"), src1=1, comment="GSU == 1 ?"))
        module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
        module.add(self.calculateLoopNumIterGsu(writer, kernel, loopCounterName, tmpSgprInfo))
        module.add(gsuLabel)

        return module

    ##############################################################################
    # Emit code to compute loop iterations for GSU.
    # See same function in KernelWriterSource.py for background explanation
    # This function is used to compute number of loop iters and also
    # for computing the global read increment for GSU case.
    # For multiple summation, the number of loop iterations needs to be reset
    # for each iteration so replicate the code in addr inc and at open of unroll loop

    # tmpSgpr is allocation of at least 3 tmpSgpr

    # Output: SGPR(destName) contains the number of unroll iterations for
    # this workgroup.
    ##############################################################################
    def calculateLoopNumIterGsu(self, writer, kernel, destName, tmpSgprRes: RegisterPoolResource):
        module = Module("calculateLoopNumIterGsu")

        loopCounter = sgpr(destName)
        quotient = destName
        remainder = "GSUSumIdx+1" # numIterPerWgRemainder
        dividend = destName

        tmpVgpr = writer.vgprPool.checkOut(2,"tmp")
        tmpVgprRes = RegisterPoolResource(idx=tmpVgpr, size=2)
        module.add(scalarUInt32DivideAndRemainder(quotient, dividend, "GSU", remainder, tmpVgprRes, wavewidth=kernel["WavefrontSize"]))
        writer.vgprPool.checkIn(tmpVgpr)

        # if gsuSumIdx < numIterPerWgRemainder
        module.add(SAddU32(dst=sgpr(tmpSgprRes.idx), src0=1, \
            src1=loopCounter, comment="tmp<-numIterMyWg+" ))
        module.add(SCmpLtU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), \
            comment="gsuSumIdx < numIterPerWgRemainder" ))
        module.add(SCMovB32(dst=loopCounter, src=sgpr(tmpSgprRes.idx), comment="numIterMyWg++ if needed"))

        return module
    
    def computeStoreSrdStart(self, writer, kernel):
        module = Module("GSU On computeStoreSrdStart")

        indices = list(range(0, kernel["ProblemType"]["NumIndicesC"]))
        numDim = len(indices)

        if kernel["GlobalSplitUAlgorithm"] == 'MultipleBuffer' or kernel["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
            gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
            module.add(SCmpEQU32(src0=sgpr("GSU"), src1=1, comment="GSU == 1 ?"))
            module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
            # GSU algorithm 2: adjust output buffer address to per GSU buffer
            with writer.allocTmpSgpr(4, alignment=1) as tmpSgprInfo:
                if tmpSgprInfo.idx % 2 == 0:
                    tmpSgprX2 = tmpSgprInfo.idx+0
                    tmpSgpr0 = tmpSgprInfo.idx+0
                    tmpSgpr1 = tmpSgprInfo.idx+1
                    tmpSgpr2 = tmpSgprInfo.idx+2
                    tmpSgpr3 = tmpSgprInfo.idx+3
                else:
                    tmpSgprX2 = tmpSgprInfo.idx+1
                    tmpSgpr0 = tmpSgprInfo.idx+1
                    tmpSgpr1 = tmpSgprInfo.idx+2
                    tmpSgpr2 = tmpSgprInfo.idx+0
                    tmpSgpr3 = tmpSgprInfo.idx+3
                module.addComment("GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s")
                module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr0), sgpr(tmpSgpr1), sgpr("SizesFree+0"), sgpr("GSUSumIdx"), "Free0"))
                for i in range(1, numDim):
                    module.add(SSubU32(dst=sgpr(tmpSgpr2), src0=sgpr("SizesFree+%u"%i), src1=1, comment="Free%u" % i))
                    module.add(SMulI32(dst=sgpr(tmpSgpr2), src0=sgpr(tmpSgpr2), src1=sgpr("GSUSumIdx"), comment="Free%u" % i))
                    module.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmpSgpr2), sgpr(tmpSgpr3), sgpr(tmpSgpr2), sgpr("StrideC%s"%writer.states.indexChars[i]), "Free%u" % i))
                    module.add(SAddU32(dst=sgpr(tmpSgpr0), src0=sgpr(tmpSgpr0), src1=sgpr(tmpSgpr2), comment="Free%u" % i))
                    module.add(SAddCU32(dst=sgpr(tmpSgpr1), src0=sgpr(tmpSgpr1), src1=sgpr(tmpSgpr3), comment="Free%u" % i))
                module.add(SLShiftLeftB64(dst=sgpr(tmpSgprX2,2), src=sgpr(tmpSgprX2,2), shiftHex=log2(writer.states.bpeCinternal), comment="scale by bpe"))
                module.add(SAddU32(dst=sgpr("SrdD+0"), src0=sgpr("SrdD+0"), src1=sgpr(tmpSgprX2), comment="add lo GSU offset to SRD"))
                module.add(SAddCU32(dst=sgpr("SrdD+1"), src0=sgpr("SrdD+1"), src1=sgpr(tmpSgpr1), comment="add hi GSU offset to SRD"))
            module.add(gsuLabel)
    
        return module

    def noLoadLoop(self, writer, kernel, tensorParametersA, tensorParametersB, pack):
        module = Module("GSU On noLoadLoop")

        gsuLabel = Label(label=writer.labels.getNameInc("GSU"), comment="")
        module.add(SCmpEQU32(src0=sgpr("GSU"), src1=1, comment="GSU == 1 ?"))
        noLoadLoopModules = None
        acclen = 0
        gsuBackup          = kernel["GlobalSplitU"]
        gsuAccumBackup     = kernel["_GlobalAccumulation"]
        bpeCexternalBackup = writer.states.bpeCexternal
        kernel["GlobalSplitU"] = 1
        kernel["_GlobalAccumulation"] = None
        writer.states.bpeCexternal = writer.states.bpeCexternalGSU1
        if kernel["KernelLanguage"] == "Assembly" and kernel["OptNoLoadLoop"] and \
            kernel["BufferLoad"] and kernel["BufferStore"] and writer.states.doShadowInit and \
            kernel["LocalSplitU"]==1 and \
            writer.states.actualSummationLoops==1:

            # two different noLoadLoops:
            # 1. OptNLL & PAP global-read interleaved (only for PAP=ON)
            # (2. OptNLL : No PAP global-read (For PAP=OFF, or PAP=ON but the last tile))
            #  -> this is unified with 1. global-read is invalidated at the last tile.
            # 3. OrdinaryNLL (Not Opt.)
            writer.saveLocalPointers(kernel, tensorParametersA, tensorParametersB)
            # deepCopy packCode for OptNLL noLoadLoop
            deepCopyPack = fastdeepcopy(pack)
            noLoadLoopModules = writer.noLoadLoop(kernel, tensorParametersA, tensorParametersB, isOptNLL=True, isNGLL=False, pack=deepCopyPack)
            acclen = noLoadLoopModules.countType(Instruction)
            writer.restoreLocalPointers(kernel, tensorParametersA, tensorParametersB)
        kernel["GlobalSplitU"] = gsuBackup
        kernel["_GlobalAccumulation"] = gsuAccumBackup
        writer.states.bpeCexternal = bpeCexternalBackup

        if acclen > 16384:
            with writer.allocTmpSgpr(3) as tmpSgprInfo:
                module.add(writer.longBranchScc0(gsuLabel, posNeg=1, tmpSgprInfo=tmpSgprInfo, comment="branch if GSU != 1"))
        else:
            module.add(SCBranchSCC0(labelName=gsuLabel.getLabelName(), comment="branch if GSU != 1"))

        if noLoadLoopModules != None:
            module.add(noLoadLoopModules)
        module.add(gsuLabel)

        return module
    
    def tailLoopNumIter(self, writer, kernel, loopCounter):
        module = Module("GSU On tailLoopNumIter")

        # if GSU numIter=0 if gsuSumIdx != remainder
        module.add(SCmpLgU32(src0=sgpr("GSUSumIdx"), src1=sgpr("GSUSumIdx+1"), \
            comment="gsuSumIdx == numIterPerWgRemainder"))
        module.add(SCMovB32(dst=loopCounter, src=hex(0), comment="numIter=0 if gsuSimIdx!=remainder"))

        return module
    
    def setupNewTile(self, writer, kernel, tensorParametersA, tensorParametersB, tPM):
        module = Module("GSU On setupNewTile")

        addBranch = False
        for i in reversed(range(kernel["ProblemType"]["NumIndicesSummation"])):
            if i != writer.states.unrollIdx:
                addBranch = True
                break
        if addBranch:
            gsuBackup   = kernel["GlobalSplitU"]
            gsuLabel    = Label(label=writer.labels.getNameInc("GSU"), comment="")
            gsuLabelEnd = Label(label=writer.labels.getNameInc("GSU_End"), comment="")
            module.add(SCmpEQU32(src0=sgpr("GSU"), src1=1, comment="GSU == 1 ?"))
            module.add(SCBranchSCC1(labelName=gsuLabel.getLabelName(), comment="branch if GSU == 1"))
            module.addComment1("global read addresses: increments a")
            kernel["GlobalSplitU"] = 2
        module.add(self.graIncrementsAB(writer, kernel, tensorParametersA, tensorParametersB, tPM))
        if addBranch:
            module.add(SBranch(gsuLabelEnd.getLabelName()))
            module.add(gsuLabel)
            kernel["GlobalSplitU"] = 1
            module.add(self.graIncrementsAB(writer, kernel, tensorParametersA, tensorParametersB, tPM))
            kernel["GlobalSplitU"] = gsuBackup
            module.add(gsuLabelEnd)

        return module