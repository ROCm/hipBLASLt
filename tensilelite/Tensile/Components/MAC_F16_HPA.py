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

from ..TensileInstructions import DataType, Module, vgpr, SSetPrior, VFmaMixF32, VMadMixF32, VOP3PModifiers, VDot2F32F16, VDot2CF32F16
from ..Component import Component, MAC

# huang
class FMA_F16_HPA_DOT2(MAC):
    asmCaps = lambda caps: caps['v_dot2_f32_f16'] or caps['v_dot2c_f32_f16']
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": True},
             }

    def __call__(self, writer, tPA, tPB, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("FMA_F16_HPA_DOT2")
        module.addComment(self.commentHeader())

        vars = {}

        if writer.states.asmCaps["v_dot2_f32_f16"]:
            instruction = VDot2F32F16
        else:
            instruction = VDot2CF32F16

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        # vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        # vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

        for block1 in range(0, kernel["ThreadTile1"]):
            for block0 in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    vars["block0"] = block0
                    vars["block1"] = block1
                    vars["blockA"] = block0 if tPA["tileIdx"] == 0 else block1
                    vars["blockB"] = block1 if tPB["tileIdx"] != 0 else block0
                    vars["iui"] = iui

                    vars["cIdxExpr"] = "%d+%d" % (vars["block0"], vars["block1"]*vars["ThreadTile0"])
                    cidx = eval(vars["cIdxExpr"])
                    cStr = "ValuC+{cIdxExpr}".format_map(vars)
                    aStr = "ValuA_X{m}_I{iui}+{blockA}".format_map(vars)
                    bStr = "ValuB_X{m}_I{iui}+{blockB}".format_map(vars)
                    module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr), \
                                           comment="ValuC[%u] iui=%u" % (cidx, vars["iui"])))

                    if (block1 == 0) and (block0 == 0) and (iui == 0):
                        module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

        module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

        return module

# class FMA_F16_HPA_MAD_MIX(MAC):
#     asmCaps = lambda caps: caps['v_mad_mix_f32'] or caps['v_fma_mix_f32']
#     #archCaps = {}
#     kernel = {"ProblemType": {"DataType": DataType(DataType.half),
#                               "HighPrecisionAccumulate": True},
#              }

#     def __call__(self, writer, tPA, tPB, m, innerUnroll):
#         kernel = writer.states.kernel

#         module = Module("FMA_F16_HPA_MAD_MIX")
#         module.addComment(self.commentHeader())

#         vars = {}

#         if writer.states.asmCaps["v_fma_mix_f32"]:
#             instruction = VFmaMixF32
#         else:
#             instruction = VMadMixF32

#         vars["m"] = m
#         vars["kernel"] = kernel

#         vars["ThreadTile0"] = kernel["ThreadTile0"]
#         vars["ThreadTile1"] = kernel["ThreadTile1"]

#         vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
#         vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

#         for block1 in range(0, kernel["ThreadTile1"]//2):
#             for block0 in range(0, kernel["ThreadTile0"]//2):
#                 for iui in range(0, innerUnroll):
#                     vars["block0"] = block0
#                     vars["block1"] = block1
#                     vars["blockA"] = block0 if tPA["tileIdx"] == 0 else block1
#                     vars["blockB"] = block1 if tPB["tileIdx"] != 0 else block0
#                     vars["iui"] = iui

#                     vars["cIdxExpr"] = "%d+%d" % (vars["block0"]*2, vars["block1"]*vars["ThreadTile0"]*2)
#                     cidx = eval(vars["cIdxExpr"])
#                     cStr = "ValuC+{cIdxExpr}".format_map(vars) # *2 b/c of fp32
#                     aStr = "ValuA_X{m}_I{iui}+{blockA}".format_map(vars)
#                     bStr = "ValuB_X{m}_I{iui}+{blockB}".format_map(vars)
#                     module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr), \
#                                            vop3=VOP3PModifiers(op_sel=[0,0,0], op_sel_hi=[1,1,0]), comment="ValuC[%u] iui=%u" % (cidx, vars["iui"])))

#                     if (block1 == 0) and (block0 == 0) and (iui == 0):
#                         module.add(SSetPrior(prior=1, comment="Raise priority while processing macs"))

#                     vars["cIdxExpr"] = "%d+%d+1" % (vars["block0"]*2, vars["block1"]*vars["ThreadTile0"]*2)
#                     cidx  = eval(vars["cIdxExpr"])
#                     cStr  = "ValuC+{cIdxExpr}".format_map(vars) # *2 b/c of fp32
#                     opSel = [1,0,0] if tPA["tileIdx"] == 0 else [0,1,0]
#                     module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr), \
#                                            vop3=VOP3PModifiers(op_sel=opSel, op_sel_hi=[1,1,0]), comment="ValuC[%u]" % cidx))

#                     vars["cIdxExpr"] = "%d+%d+%d" % (vars["block0"]*2, vars["block1"]*vars["ThreadTile0"]*2, vars["Half_ThreadTile0"]*2)
#                     cidx  = eval(vars["cIdxExpr"])
#                     cStr  = "ValuC+{cIdxExpr}".format_map(vars)
#                     opSel = [0,1,0] if tPA["tileIdx"] == 0 else [1,0,0]
#                     module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr), \
#                                            vop3=VOP3PModifiers(op_sel=opSel, op_sel_hi=[1,1,0]), comment="ValuC[%u]" % cidx))

#                     vars["cIdxExpr"] = "%d+%d+%d+1" % (vars["block0"]*2, vars["block1"]*vars["ThreadTile0"]*2, vars["Half_ThreadTile0"]*2)
#                     cidx = eval(vars["cIdxExpr"])
#                     cStr = "ValuC+{cIdxExpr}".format_map(vars)
#                     module.add(instruction(dst=vgpr(cStr), src0=vgpr(aStr), src1=vgpr(bStr), src2=vgpr(cStr), \
#                                            vop3=VOP3PModifiers(op_sel=[1,1,0], op_sel_hi=[1,1,0]), comment="ValuC[%u]" % cidx))

#         module.add(SSetPrior(prior=0, comment="Reset priority after macs"))

#         return module
