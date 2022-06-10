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

from ..TensileInstructions import DataType, Module
from ..Component import Component, MAC

class MAC_F32_Plain(MAC):
    """
    Plain MAC instruction implementation
    """
    @staticmethod
    def asmCaps(caps):
        return caps["v_mac_f32"] or caps["v_fma_f32"]
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.single)}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.states.kernel

        if writer.states.asmCaps["v_fmac_f32"]:
            instruction = "v_fmac_f32"
        elif writer.states.asmCaps["v_fma_f32"]:
            instruction = "v_fma_f32"
        elif writer.states.asmCaps["v_mac_f32"]:
            instruction = "v_mac_f32"
        else:
            raise RuntimeError("FMA and MAC instructions are not supported on {}".format(kernel["ISA"]))

        if not writer.states.asmCaps[instruction]:
            raise RuntimeError("{} instruction specified but not supported on {}".format(instruction, kernel["ISA"]))

        module = Module("MAC_F32_Plain")
        module.addComment(self.commentHeader())

        vars = {}

        vars["m"] = m
        vars["kernel"] = kernel

        vars["ThreadTile0"] = kernel["ThreadTile0"]
        vars["ThreadTile1"] = kernel["ThreadTile1"]

        vars["instruction"] = instruction

        priority = Component.Priority.find(writer)

        for idx1 in range(0, kernel["ThreadTile1"]):
            for idx0 in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    vars["idx0"] = idx0
                    vars["idx1"] = idx1
                    vars["a"] = idx0 if writer.tPB["tile01Idx"] else idx1
                    vars["b"] = idx1 if writer.tPB["tile01Idx"] else idx0
                    vars["iui"] = iui

                    cStr = "v[vgprValuC + {idx0} + {idx1}*{ThreadTile0}]".format_map(vars)
                    aStr = "v[vgprValuA_X{m}_I{iui} + {a}]".format_map(vars)
                    bStr = "v[vgprValuB_X{m}_I{iui} + {b}]".format_map(vars)

                    if instruction == "v_fma_f32":
                        module.addInst("v_fma_f32", cStr, aStr, bStr, cStr, "")
                    else:
                        module.addInst(instruction, cStr, aStr, bStr, "")

                    module.add(priority(writer, 1, "Raise priority while processing macs"))

        module.add(priority(writer, 0, "Reset priority after macs"))

        return module
