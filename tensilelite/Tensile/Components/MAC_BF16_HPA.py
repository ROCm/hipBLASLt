################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..TensileInstructions import DataType, Module, vgpr, SSetPrior, VFmaF32, VAndB32, VLShiftLeftB32
from ..Component import Component, MAC

class FMA_BF16_HPA(MAC):
    asmCaps = {"v_fma_f32": True}
    kernel = {"ProblemType": {"DataType": DataType(DataType.bfloat16),
                              "HighPrecisionAccumulate": True}}

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel
        module = Module("FMA_BF16_HPA")
        module.addComment(self.commentHeader())

        vars = {}
        vars["m"] = m
        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile0Half"] = kernel["ThreadTile0"] // 2
        vars["ThreadTile0Halfx2"] = vars["ThreadTile0Half"] * 2

        for iui in range(0, innerUnroll):
            vars["iui"] = iui
            for blockA in range(kernel["ThreadTileA"]//2-1, -1, -1):
                vars["blockA"] = blockA
                vars["blockAx2"] = blockA * 2
                dst = vgpr("ValuA_X{m}_I{iui}+{blockAx2}+1".format_map(vars))
                src = vgpr("ValuA_X{m}_I{iui}+{blockA}".format_map(vars))
                module.add(VAndB32(dst=dst, src0="0xffff0000", src1=src))
                dst = vgpr("ValuA_X{m}_I{iui}+{blockAx2}".format_map(vars))
                module.add(VLShiftLeftB32(dst=dst, shiftHex=16, src=src))
            for blockB in range(kernel["ThreadTileB"]//2-1, -1, -1):
                vars["blockB"] = blockB
                vars["blockBx2"] = blockB * 2
                dst = vgpr("ValuB_X{m}_I{iui}+{blockBx2}+1".format_map(vars))
                src = vgpr("ValuB_X{m}_I{iui}+{blockB}".format_map(vars))
                module.add(VAndB32(dst=dst, src0="0xffff0000", src1=src))
                dst = vgpr("ValuB_X{m}_I{iui}+{blockBx2}".format_map(vars))
                module.add(VLShiftLeftB32(dst=dst, shiftHex=16, src=src))

        for block1 in range(0, kernel["ThreadTile1"]//2):
            vars["block1"] = block1
            vars["block1x2"] = block1 * 2
            for block0 in range(0, kernel["ThreadTile0"]//2):
                vars["block0"] = block0
                vars["block0x2"] = block0 * 2
                if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                    # we treat HighPrecisionAccumulate as expanded packed math
                    for iui in range(0, innerUnroll):
                        vars["iui"] = iui

                        vars["blockA"] = block0 if tPB["tile01Idx"] else block1
                        vars["blockB"] = block1 if tPB["tile01Idx"] else block0
                        vars["blockAx2"] = vars["blockA"] * 2
                        vars["blockBx2"] = vars["blockB"] * 2

                        aStr0 = vgpr("ValuA_X{m}_I{iui}+{blockAx2}+0".format_map(vars))
                        aStr1 = vgpr("ValuA_X{m}_I{iui}+{blockAx2}+1".format_map(vars))
                        bStr0 = vgpr("ValuB_X{m}_I{iui}+{blockBx2}+0".format_map(vars))
                        bStr1 = vgpr("ValuB_X{m}_I{iui}+{blockBx2}+1".format_map(vars))

                        vars["block1xThreadTile0x2"] = vars["block1"] * vars["ThreadTile0"] * 2
                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + 0
                        cStr = vgpr("ValuC+{block0x2}+{block1xThreadTile0x2}+0+0".format_map(vars))
                        module.add(VFmaF32(dst=cStr, src0=aStr0, src1=bStr0, src2=cStr, comment="ValuC[%u]" % cidx))

                        if (block1 == 0) and (block0 == 0) and (iui == 0):
                            module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

                        aStr = aStr1 if tPB["tile01Idx"] else aStr0
                        bStr = bStr0 if tPB["tile01Idx"] else bStr1
                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + 1
                        cStr = vgpr("ValuC+{block0x2}+{block1xThreadTile0x2}+0+1".format_map(vars)) # *2 b/c of fp32
                        module.add(VFmaF32(dst=cStr, src0=aStr, src1=bStr, src2=cStr, comment="ValuC[%u]" % cidx))

                        aStr = aStr0 if tPB["tile01Idx"] else aStr1
                        bStr = bStr1 if tPB["tile01Idx"] else bStr0
                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                        cStr = vgpr("ValuC+{block0x2}+{block1xThreadTile0x2}+{ThreadTile0Halfx2}+0".format_map(vars))
                        module.add(VFmaF32(dst=cStr, src0=aStr, src1=bStr, src2=cStr, comment="ValuC[%u]" % cidx))

                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                        cStr = vgpr("ValuC+{block0x2}+{block1xThreadTile0x2}+{ThreadTile0Halfx2}+1".format_map(vars))
                        module.add(VFmaF32(dst=cStr, src0=aStr1, src1=bStr1, src2=cStr, comment="ValuC[%u]" % cidx))
                        """
                        ignore this, not quite correct for mixed precision
                        D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                        D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                        C[0] = A[0]*B[0]+D[0]
                        C[1] = A[1]*B[1]+D[1]
                        """

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))
        return module
