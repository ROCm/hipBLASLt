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

from .TensileInstructions import Module, SAddI32, SEndpgm, fastdeepcopy, BranchInstruction, CommonInstruction

from dataclasses import dataclass, field

#######################################
# Public functions
#######################################
@dataclass
class TensilePassOptions:
    removeDupActFunc: bool = field(init=False)

def TensilePass(module, options: TensilePassOptions):
    if options.removeDupActFunc:
        _removeDuplicatedActivationFunctions(module)


def getActivationFunctionModuleName(gwvw, sgpr, tmpVgpr, tmpSgpr):
    return "ActFunc_VW%d_Sgpr%d_Tmp%s_%s"%(gwvw, sgpr, tmpVgpr, tmpSgpr)

def getActivationBranchModuleName():
    return "InsertActFunctionCallAddrCalc"

#######################################
# Internal functions
#######################################
def _findActFunc(module) -> dict:
    modFunc = {}
    for item in module.items():
        if isinstance(item, Module):
            if "ActFunc_VW" in item.name:
                if item.name in modFunc:
                    modFunc[item.name].append(item)
                else:
                    modFunc[item.name] = [item]
            else:
                tmp = _findActFunc(item)
                for key, t in tmp.items():
                    if key in modFunc:
                        modFunc[key].extend(t)
                    else:
                        modFunc[key] = t
    return modFunc

def _replaceActBranchLabel(module, labels):
    for item in module.items():
        if isinstance(item, Module):
            if "InsertActFunctionCallAddrCalc" in item.name:
                labelLeft = labels[1:]
                replaceLabel = False
                for inst in item.items():
                    if isinstance(inst, SAddI32) and inst.comment == "target branch offset":
                        if inst.srcs[0] in labelLeft:
                            replaceLabel = True
                            break
                if replaceLabel:
                    for inst in item.items():
                        if isinstance(inst, SAddI32) and inst.comment == "target branch offset":
                            # The label is generated in the format of XXXX_1, XXXX_2
                            # and string.rpartition returns ('XXXX', '_', '1').
                            # We only need the first string.
                            part = inst.srcs[0].rpartition("_")
                            inst.srcs[0] = part[0]
            else:
                _replaceActBranchLabel(item, labels)

def _replaceActBranchLabelToLabel(module, labelToLabels):
    for item in module.items():
        if isinstance(item, Module):
            if "InsertActFunctionCallAddrCalc" in item.name:
                for inst in item.items():
                    if isinstance(inst, SAddI32) and inst.comment == "target branch offset":
                        if str(inst.srcs[0]) in labelToLabels:
                            inst.srcs[0] = labelToLabels[str(inst.srcs[0])]
            else:
                _replaceActBranchLabelToLabel(item, labelToLabels)

def _getModuleInstArgStr(module):
    argStr = ""
    for item in module.items():
        if isinstance(item, CommonInstruction):
            argStr += item.getArgStr()
        elif isinstance(item, Module):
            argStr += _getModuleInstArgStr(item)
    return argStr

def _removeDuplicatedActivationFunctions(module):
    modFunc = _findActFunc(module)
    moduleLast = Module("AddToLast")
    for _, mlist in modFunc.items():
        if len(mlist) > 1:
            labels = []
            sgprMapToLabel = dict()
            labelToLabel = dict()
            for ml in mlist:
                for i in range(len(ml.items())):
                    labelName = ml.items()[i].items()[0].getLabelName()
                    resourceName = ""
                    for inst in (ml.items()[i].items()):
                        if isinstance(inst, BranchInstruction):
                            resourceName += str(inst)
                        elif isinstance(inst, Module):
                            resourceName += _getModuleInstArgStr(inst)
                    prefix, _, n = labelName.rpartition("_")
                    if n.isnumeric():
                        resourceName = str(prefix) + resourceName
                    else:
                        resourceName = str(labelName) + resourceName
                    if resourceName in sgprMapToLabel:
                        labelToLabel[labelName] = sgprMapToLabel[resourceName]
                    else:
                        sgprMapToLabel[resourceName] = labelName
                        moduleLast.add(ml.items()[i])
                ml.parent.removeItem(ml)
            # Avoid using deepcopy
            _replaceActBranchLabelToLabel(module, labelToLabel)
    if moduleLast.items():
        module.add(moduleLast)
        module.add(SEndpgm())
