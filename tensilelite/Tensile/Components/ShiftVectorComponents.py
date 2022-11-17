################################################################################
# Copyright 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..TensileInstructions import Label, Module, VCC, DSModifiers, \
                                DSBPermuteB32, SBranch, SCBranchVCCNZ, \
                                SMovB32, SMovB64, SNop, \
                                SOrSaveExecB32, SOrSaveExecB64, SWaitCnt, \
                                VAccvgprReadB32, VAccvgprWriteB32, VAddCOU32, \
                                VAndB32, VCmpEQU32, VCmpLtU32, VCmpXEqU32, \
                                VCndMaskB32, VMovB32, VMulI32I24, VLShiftLeftB32, \
                                VLShiftRightB32, VSubU32, \
                                RegisterPoolResource, staticMultiply, vectorStaticDivide, \
                                vectorStaticRemainder, vgpr, sgpr, accvgpr, log2
from ..Component import ShiftVectorComponents

class ShiftVectorComponentsMFMA(ShiftVectorComponents):
    kernel = {"EnableMatrixInstruction": True}

    """
    Shift Vector Components d0,1
    """
    def __call__(self, writer, kernel, tP):
        """ when we enable shift ptr with vectorwidth(2), we shift global read on edge block when size % vectorwidth != 0.
            For example if M size == 3 vector width == 2, we want to do global read for [0-1] and [2-3].
            But 3 is not in memory object, so we shift to do global read [0-1] and [1-2].
            So buffer become [0, 1, 1, 2], assume result in register is same as input [0, 1, 1, 2]
            We need to shift it back to [0, 1, 2].

            In MFMA outputs, We have numContinuousOutput(4) for each thread.
            We have numThreadInWave(64) threads.
            number of thread in N is sames as kernel["MatrixInstN"] (32)
            number of thread in M is numThreadInWave/numOutputThreads1 = 2
            stride of continuous output for each thread (numSubOutputPerWave0) is numOutputThreads0 * numContinuousOutput, (8).
            we have numSubOutputGroupsPerWave0 which is 4 (kernel[tP["mt"]](64) // numSubOutputPerWave0(8))

            So we do shift back by below algorithm.
            1. check if M_size % GlobalLoadVectorWidth != 0, return if == 0
            2. decide which subgroup we need to shift, M_size(3) means 3/8 = group 0
            3. decide which thread we need to shift, we have different groups of thread, (0-31) for first group, (32-63) for second group.
            4. decide which shift block (subTile1) we want to shift. for ex [0-1], [1-2], we want to shift second subtile
        """

        # TODO: use this for non SourceSwap for B?
        # this part can  support non SourceSwap for B
        # But let non SourceSwap for B go original shiftptr path
        if (not kernel["SourceSwap"]) and tP["isB"]:
            return Module("ShiftVectorComponentsMFMA (Empty)")

        # common parameter
        regPerElem      = kernel["MIRegPerOut"]
        glvw            = tP["glvw"]
        numThreadInWave = writer.kernel["WavefrontSize"]
        accImOffset     = writer.AccVgprImagNumOffset(kernel)
        vectorWidth     = kernel["VectorWidth"] if (kernel["SourceSwap"] and tP["isA"]) else 1

        # use to handle MatrixInst 4x4
        matrixInstM     = kernel["MatrixInstM"] * kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstM"]
        matrixInstN     = kernel["MatrixInstN"] * kernel["MatrixInstBN"] if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstN"]
        matrixInstBM    = 1 if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
        matrixInstBN    = 1 if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]

        # unify process for dimension M/N
        matrixInstCoal  = matrixInstM              if tP["isA"] else matrixInstN
        matrixInstPrep  = matrixInstN              if tP["isA"] else matrixInstM
        matrixInstBCoal = matrixInstBM             if tP["isA"] else matrixInstBN
        matrixInstBPrep = matrixInstBN             if tP["isA"] else matrixInstBM
        miWaveGroupCoal = kernel["MIWaveGroup"][0] if tP["isA"] else kernel["MIWaveGroup"][1]
        miWGIdStride    = numThreadInWave          if tP["isA"] else (numThreadInWave * kernel["MIWaveGroup"][0])
        miWaveTitleCoal = kernel["MIWaveTile"][0]  if tP["isA"] else kernel["MIWaveTile"][1]
        miWaveTitlePrep = kernel["MIWaveTile"][1]  if tP["isA"] else kernel["MIWaveTile"][0]

        # unify process for SourceSwap and non-SourceSwap
        conThInProcDim  = kernel["SourceSwap"] ^ tP["isB"] # continuous threads in processed dimension(Coalesced dimension)

        threadInterval  = 1 if conThInProcDim else matrixInstPrep
        numThreadInCoal = matrixInstCoal if conThInProcDim else (numThreadInWave // matrixInstPrep)

        numContOutCoal  = 1 if conThInProcDim else kernel["MIOutputVectorWidth"]
        allContOutCoal  = numContOutCoal * vectorWidth

        OutBlocksInMI   = (matrixInstCoal * matrixInstPrep) // numThreadInWave // numContOutCoal
        OutBlocksInMI   = 1 if conThInProcDim else OutBlocksInMI

        subMBShapeCoal  = (matrixInstCoal * vectorWidth) if conThInProcDim else ((numThreadInWave // matrixInstPrep) * numContOutCoal)
        MBShapeCoal     = subMBShapeCoal * OutBlocksInMI
        MIBShapeCoal    = MBShapeCoal * matrixInstBCoal
        WGShapeCoal     = MIBShapeCoal * miWaveGroupCoal
        miOuterTTCoal   = miWaveTitleCoal // vectorWidth

        numOutputsPrep  = (matrixInstCoal * matrixInstPrep // numThreadInWave) if conThInProcDim else 1
        numOutputsPrep  = numOutputsPrep * matrixInstBPrep * miWaveTitlePrep
        complexMultiplier = 2 if kernel["ProblemType"]["DataType"].isComplex() else 1

        # unify process for dimension M/N
        regStrideCoal = 1                                                                if tP["isA"] else numOutputsPrep
        regStridePrep = miOuterTTCoal * matrixInstBCoal * OutBlocksInMI * allContOutCoal if tP["isA"] else 1


        # labels for shiftptr
        glvwLabels = []
        MBblockLabels = []
        VWBlockLabels = []
        for i in range(0, glvw): # grvw block
            r = (i+1) % glvw    # r = [1,2,3,...,glvw-1, 0], the last one glvwLabels[glvw-1] stores for r=0 -> no shift
            comment = "end shift0" if i == glvw-1 else ""
            label = Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u" % (tP["idx"], r) ), comment)
            glvwLabels.append(label)
            subMBLabels = []
            subVWBlockLabels = []
            for mb in range(0, OutBlocksInMI * matrixInstBCoal * miOuterTTCoal): # unit block of each thread
                label = Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u_BM%u" % (tP["idx"], r, mb)), "")
                subMBLabels.append(label)
                sub2VWBlockLabels = []
                for vw in range(0, max(1, allContOutCoal//glvw)): # vw block of glvw
                    label = Label(writer.labels.getName("ShiftVectorComponents%u_GLVW%u_BM%u_VW%u" % (tP["idx"], r, mb, vw)), "")
                    sub2VWBlockLabels.append(label)
                subVWBlockLabels.append(sub2VWBlockLabels)
            MBblockLabels.append(subMBLabels)
            VWBlockLabels.append(subVWBlockLabels)

        with writer.allocTmpSgpr(writer.laneSGPRCount) as tmpSgprInfo:
            # wgMT value
            tmpSgpr = tmpSgprInfo.idx
            tmpVgpr = writer.vgprPool.checkOutAligned(2,2)
            tmpVgprRes = RegisterPoolResource(tmpVgpr, 2)
            dummy   = writer.vgprPool.checkOut(1)
            wgMT    = writer.vgprPool.checkOut(1)
            wg      = tP["wg"]

            module = Module("ShiftVectorComponentsMFMA")

            # get M size of edge block
            mtReg = writer.vgprPool.checkOut(1)
            module.add(VMovB32(dst=vgpr(wgMT), src=sgpr(wg)))
            module.add(VMulI32I24(dst=vgpr(wgMT), src0=hex(-kernel[tP["mt"]]), src1=vgpr(wgMT), comment="wg*MT"))
            module.add(VAddCOU32(dst=vgpr(wgMT), dst1=VCC(), src0=sgpr("SizesFree+%u"%tP["idx"]), src1=vgpr(wgMT), comment="wgMT = Size - wg*MT"))
            module.add(VMovB32(dst=vgpr(mtReg), src=hex(kernel[tP["mt"]]), comment="MT"))
            module.add(VCmpLtU32(dst=sgpr(tmpSgpr,writer.laneSGPRCount), src0=vgpr(wgMT), src1=vgpr(mtReg), comment="wgMT < MT"))
            module.add(VCndMaskB32(dst=vgpr(wgMT), src0=vgpr(mtReg), src1=vgpr(wgMT), src2=sgpr(tmpSgpr,writer.laneSGPRCount), comment="wgMT = (wgMT < MT) ? wgMT : MT" ))

            # identify which wave have to process
            wReg = writer.vgprPool.checkOut(1)
            sReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(wReg, "Serial", miWGIdStride, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, wReg, wReg, miWaveGroupCoal, tmpVgprRes, tmpSgprInfo))
            module.add(vectorStaticDivide(sReg, wgMT, MIBShapeCoal, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, sReg, sReg, miWaveGroupCoal, tmpVgprRes, tmpSgprInfo))
            module.add(VCmpEQU32(dst=sgpr(tmpSgpr,writer.laneSGPRCount), src0=vgpr(sReg), src1=vgpr(wReg), comment="wave_id == block_belong_to_wave?" ))
            module.add(VCndMaskB32(dst=vgpr(wgMT), src0=vgpr(mtReg), src1=vgpr(wgMT), src2=sgpr(tmpSgpr,writer.laneSGPRCount), comment="wgMT = (wgMT < MT) ? wgMT : MT" ))
            writer.vgprPool.checkIn(mtReg)
            writer.vgprPool.checkIn(sReg)

            # mbReg: which mb block meed to shift, mb(matrixInstM*VectorWidth)
            module.addComment1("mbReg: which mb block need to shift, mb(matrixInstCoal(%u) * VectorWidth(%u))" % (matrixInstCoal, vectorWidth))
            mbReg = writer.vgprPool.checkOut(1)
            tReg  = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(mbReg, wgMT, subMBShapeCoal, tmpVgprRes))
            module.add(staticMultiply(vgpr(tReg), vgpr(wReg), (matrixInstBCoal * OutBlocksInMI), tmpSgprInfo))
            module.add(VSubU32(dst=vgpr(mbReg), src0=vgpr(mbReg), src1=vgpr(tReg)))
            writer.vgprPool.checkIn(tReg)

            # gbReg: glvw block id
            module.addComment1("gbReg: glvw block id")
            gbReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(gbReg, wgMT, glvw, tmpVgprRes))

            # tgbReg: thread in glvw block
            module.addComment1("tgbReg: glvw block id")
            tgbReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticDivide(tgbReg, "Serial", threadInterval, tmpVgprRes))
            module.add(vectorStaticRemainder(dummy, tgbReg, tgbReg, numThreadInCoal, tmpVgprRes, tmpSgprInfo))
            module.add(staticMultiply(vgpr(tgbReg), vgpr(tgbReg), allContOutCoal, tmpSgprInfo))
            module.add(vectorStaticDivide(tgbReg, tgbReg, glvw, tmpVgprRes))
            module.add(staticMultiply(vgpr(wReg), vgpr(wReg), MIBShapeCoal//glvw, tmpSgprInfo))
            module.add(VAddCOU32(dst=vgpr(tgbReg), dst1=VCC(), src0=vgpr(wReg), src1=vgpr(tgbReg), comment="tgbReg = (tid_coal * continOut) / GLVW"))
            module.add(VSubU32(dst=vgpr(gbReg), src0=vgpr(gbReg), src1=vgpr(tgbReg)))
            writer.vgprPool.checkIn(wReg)
            writer.vgprPool.checkIn(tgbReg)

            # vw block of glvw
            module.addComment1("vwReg: glvw in which vw block?")
            vwReg = writer.vgprPool.checkOut(1)
            module.add(VAndB32(dst=vgpr(vwReg), src0=allContOutCoal-1, src1=vgpr(wgMT), comment="permute register between threads"))
            module.add(VLShiftRightB32(dst=vgpr(vwReg), shiftHex=log2(glvw), src=vgpr(vwReg), comment="permute register between threads"))

            # rReg : reminder of M_size % vectorwidth
            # decide to jump to block which handle this case, M_size % vector width
            module.addComment1("rReg : reminder of M_size % GlobalLoadVectorWidth")
            rReg = writer.vgprPool.checkOut(1)
            module.add(vectorStaticRemainder(dummy, rReg, wgMT, glvw, tmpVgprRes, tmpSgprInfo))
            for r in range(1, glvw):
                module.add(VCmpEQU32(dst=VCC(), src0=vgpr(rReg), src1=hex(r), comment="wgMT%%VW == %u"%r ))
                module.add(SCBranchVCCNZ(labelName=glvwLabels[(r-1)].getLabelName(), comment="branch to shift d%u r=%u"%(tP["idx"], r)))
            module.add(SBranch(labelName=glvwLabels[glvw-1].getLabelName(), comment="no shifting" ))
            writer.vgprPool.checkIn(rReg)

            _, arch2acc = writer.AccToArchMapper(kernel)

            # blocks for handle M_size % vector width
            for r in range(1, glvw):
                module.addComment2("shift d%u r=%u"%(tP["idx"], r))
                module.add(glvwLabels[r-1])
                for tt in range(0, miOuterTTCoal):
                    for bm in range(0, matrixInstBCoal):
                        for ob in range(0, OutBlocksInMI):
                            label  = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                            target = ob + OutBlocksInMI * (bm + matrixInstBCoal * miWaveGroupCoal * tt)
                            module.add(VCmpEQU32(dst=VCC(), src0=vgpr(mbReg), src1=hex(target)))
                            module.add(SCBranchVCCNZ(labelName=MBblockLabels[r-1][label].getLabelName(), comment="branch to shift d%u r%u mb%u" % (tP["idx"], r, label)))

            for r in range(1, glvw):
                for mb in range(0, miOuterTTCoal * matrixInstBCoal * OutBlocksInMI):
                    module.addComment2("shift d%u r=%u mb=%u"%(tP["idx"], r, mb))
                    MBblockLabels[r-1][mb].comment = "r%u mb%u"%(r, mb)
                    module.add(MBblockLabels[r-1][mb])
                    for vw in range(0, max(1, allContOutCoal//glvw)):
                        module.add(VCmpEQU32(dst=VCC(), src0=vgpr(vwReg), src1=hex(vw)))
                        module.add(SCBranchVCCNZ(labelName=VWBlockLabels[r-1][mb][vw].getLabelName(), comment="branch to shift d%u r%u mb%u vw%u" % (tP["idx"], r, mb, vw)))

            # blocks for handle M_size % vector width
            tReg  = writer.vgprPool.checkOut(min(glvw, allContOutCoal))
            for r in range(1, glvw):
                for tt in range(0, miOuterTTCoal):
                    for bm in range(0, matrixInstBCoal):
                        for ob in range(0, OutBlocksInMI):
                            mb = ob + OutBlocksInMI * (bm + matrixInstBCoal * tt)
                            for vw in range(0, max(1, allContOutCoal//glvw)):
                                module.addComment2("shift d%u r=%u mb=%u vw%d"%(tP["idx"], r, mb, vw))
                                VWBlockLabels[r-1][mb][vw].comment = "r%u mb%u vw%u"%(r, mb, vw)
                                module.add(VWBlockLabels[r-1][mb][vw])
                                module.add(SMovB32(dst=sgpr(tmpSgpr), src=(((ob*subMBShapeCoal + bm*MBShapeCoal + tt*WGShapeCoal) // glvw) + vw)))
                                module.add(VCmpXEqU32(dst=sgpr(tmpSgpr, writer.laneSGPRCount), src0=vgpr(gbReg), src1=sgpr(tmpSgpr), comment="is thread in edge glvw region" ))
                                module.add(VAndB32(dst=vgpr(tmpVgpr), src0=kernel["WavefrontSize"]-1, src1=vgpr("Serial"), comment="permute register between threads"))
                                module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=log2(writer.bpr), src=vgpr(tmpVgpr), comment="permute register between threads"))

                                for ot in range(numOutputsPrep):
                                    for c  in range(complexMultiplier):
                                        for nr in range(regPerElem):
                                            vgprOffsetForSCIU = 0
                                            copyInst = VAccvgprReadB32 if not kernel["MIArchVgpr"] else VMovB32
                                            for e in range(min(r, allContOutCoal)):
                                                src = (e+(glvw-r)) % allContOutCoal
                                                srcVgpr = (src + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                                srcVgpr = srcVgpr + ot * regStridePrep
                                                srcVgpr = arch2acc[srcVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                                srcVal  = accvgpr(srcVgpr) if not kernel["MIArchVgpr"] else vgpr(srcVgpr)
                                                module.add(copyInst(dst=vgpr(tReg+e), src=srcVal, comment="glvw %u mb %u tt1 %u r %u" % (r, mb, ot, nr)))

                                            if not kernel["MIArchVgpr"]:
                                                module.add(SNop(waitState=1, comment="v_accvgpr read vgpr after write vgpr: 2 wait states"))

                                            needWait = False
                                            for e in range(min(r, allContOutCoal)):
                                                crossThread = (e+(glvw-r)) // allContOutCoal
                                                if crossThread != 0:
                                                    ds = DSModifiers(na=1, offset=crossThread*threadInterval*4)
                                                    module.add(DSBPermuteB32(dst=vgpr(tReg+e), src0=vgpr(tmpVgpr), src1=vgpr(tReg+e), ds=ds, comment="permute edge values"))
                                                    needWait = True

                                            if needWait:
                                                module.add(SWaitCnt(waitAll=True, comment="wait for swizzle operation"))

                                            copyInst = VAccvgprWriteB32 if not kernel["MIArchVgpr"] else VMovB32
                                            for e in range(min(r, allContOutCoal)):
                                                dstVgpr = (e + (vw * glvw) + allContOutCoal * mb) * regStrideCoal
                                                dstVgpr = dstVgpr + ot * regStridePrep
                                                dstVgpr = arch2acc[dstVgpr] * regPerElem + nr + c * accImOffset + vgprOffsetForSCIU
                                                dstStr = accvgpr(dstVgpr) if not kernel["MIArchVgpr"] else vgpr(dstVgpr)
                                                module.add(copyInst(dst=dstStr, src=vgpr(tReg+e)))

                                # end shift reset mask and jump out
                                all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                                SMovBX = SMovB64 if kernel["WavefrontSize"] == 64 else SMovB32
                                SOrSaveExecBX = SOrSaveExecB64 if kernel["WavefrontSize"] == 64 else SOrSaveExecB32
                                module.add(SMovBX(dst=sgpr(tmpSgpr, writer.laneSGPRCount), src=all1mask, comment="to restore all threads active"))
                                module.add(SOrSaveExecBX(dst=VCC(), src=sgpr(tmpSgpr,writer.laneSGPRCount), comment="all threads active"))
                                module.add(SBranch(labelName=glvwLabels[glvw-1].getLabelName(), comment="done shifting" ))
                                module.addSpaceLine()

            module.add(glvwLabels[glvw-1])
            writer.vgprPool.checkIn(tReg)

            # checkin scratch vgprs
            writer.vgprPool.checkIn(tmpVgpr)
            writer.vgprPool.checkIn(wgMT)
            writer.vgprPool.checkIn(dummy)
            writer.vgprPool.checkIn(gbReg)
            writer.vgprPool.checkIn(vwReg)
            writer.vgprPool.checkIn(mbReg)

        return module
