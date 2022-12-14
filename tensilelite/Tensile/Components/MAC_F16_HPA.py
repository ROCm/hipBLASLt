################################################################################
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..TensileInstructions import DataType, Module
from ..Component import Component, MAC

class FMA_F16_HPA_MAD_MIX(MAC):
    asmCaps = lambda caps: caps['v_mad_mix_f32'] or caps['v_fma_mix_f32']
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": True},
             }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel

        module = Module("FMA_F16_HPA_MAD_MIX")
        module.addComment(self.commentHeader())
        priority = Component.Priority.find(writer)

        vars = {}

        if writer.states.asmCaps["v_fma_mix_f32"]:
            instruction = "v_fma_mix_f32"
        else:
            instruction = "v_mad_mix_f32"

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["Half_ThreadTile0"] = kernel["ThreadTile0"] // 2
        vars["Half_ThreadTile1"] = kernel["ThreadTile1"] // 2

        for block1 in range(0, kernel["ThreadTile1"]//2):
            for block0 in range(0, kernel["ThreadTile0"]//2):
                for iui in range(0, innerUnroll):
                    vars["block0"] = block0
                    vars["block1"] = block1
                    vars["blockA"] = block0 if writer.tPA["tileIdx"] == 0 else block1
                    vars["blockB"] = block1 if writer.tPB["tileIdx"] != 0 else block0
                    vars["iui"] = iui

                    vars["aBase"] = "vgprValuA_X{m}_I{iui}".format_map(vars)
                    vars["bBase"] = "vgprValuB_X{m}_I{iui}".format_map(vars)

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + 0*2 + 0".format_map(vars)
                    cidx = eval(vars["cIdxExpr"])
                    cStr = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                    aStr = "v[{aBase}+{blockA}]".format_map(vars)
                    bStr = "v[{bBase}+{blockB}]".format_map(vars)
                    module.addInst(instruction, cStr, aStr, bStr, cStr, "op_sel:[0,0,0]", "op_sel_hi:[1,1,0]", "ValuC[%u] iui=%u" % (cidx, vars["iui"]))

                    module.add(priority(writer, 1, "Raise priority while processing macs"))

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + 0*2 + 1".format_map(vars)
                    cidx  = eval(vars["cIdxExpr"])
                    cStr  = "v[vgprValuC + {cIdxExpr}]".format_map(vars) # *2 b/c of fp32
                    opSel = "op_sel:[1,0,0]" if writer.tPA["tileIdx"] == 0 else "op_sel:[0,1,0]"
                    module.addInst(instruction, cStr, aStr, bStr, cStr, opSel, "op_sel_hi:[1,1,0]", "ValuC[%u]" % cidx)

                    vars["cIdxExpr"] = "{block0}*2 + {block1}*{ThreadTile0}*2 + {Half_ThreadTile0}*2 + 0".format_map(vars)
                    cidx  = eval(vars["cIdxExpr"])
                    cStr  = "v[vgprValuC+{cIdxExpr}]".format_map(vars)
                    opSel = "op_sel:[0,1,0]" if writer.tPA["tileIdx"] == 0 else "op_sel:[1,0,0]"
                    module.addInst(instruction, cStr, aStr, bStr, cStr, opSel, "op_sel_hi:[1,1,0]", "ValuC[%u]" % cidx)

                    vars["cIdxExpr"] = "{block0}*2+{block1}*{ThreadTile0}*2+{Half_ThreadTile0}*2+1".format_map(vars)
                    cidx = eval(vars["cIdxExpr"])
                    cStr = "v[vgprValuC+{cIdxExpr}]".format_map(vars)
                    module.addInst(instruction, cStr, aStr, bStr, cStr, "op_sel:[1,1,0]", "op_sel_hi:[1,1,0]", "ValuC[%u]" % cidx)

        module.add(priority(writer, 0, "Reset priority after macs"))

        return module
