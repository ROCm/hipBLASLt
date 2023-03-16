################################################################################
#
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

from .Base import getGfxName
from .Code import Module
from .Containers import HolderContainer, RegisterContainer, RegName
from .DataType import DataType
from .Enums import InstType
from .Formatting import printExit
from .Instructions import Instruction, SWaitCnt

from math import log
from typing import List, Tuple
import os
import random
import shlex
import string

########################################
# Format GPRs
########################################

def _gpr(*args):
    gprType = args[0]
    args = args[1]
    if isinstance(args[0], Holder):
        idx  = args[0].idx
        name = args[0].name
        if len(args) == 1:
            return HolderContainer(gprType, name, idx, 1)
        elif len(args) == 2:
            return HolderContainer(gprType, name, idx, args[1])
    elif isinstance(args[0], int):
        if len(args) == 1:
            return RegisterContainer(gprType, None, args[0], 1)
        elif len(args) == 2:
            return RegisterContainer(gprType, None, args[0], args[1])
    elif isinstance(args[0], str):
        name = _generateRegName(args[0])
        if len(args) == 1:
            return RegisterContainer(gprType, name, None, 1)
        elif len(args) == 2:
            return RegisterContainer(gprType, name, None, args[1])
    else:
        printExit("Unknown %sgpr name or index"%gprType)

def vgpr(*args):
    return _gpr("v", args)

def sgpr(*args):
    return _gpr("s", args)

def accvgpr(*args):
    return _gpr("acc", args)

def mgpr(*args):
    return _gpr("m", args)

def _generateRegName(rawText):
    splitTxt = rawText.split("+")
    offsets = []
    if len(splitTxt) > 1:
        for arg in splitTxt[1:]:
            offsets.append(int(arg))
    return RegName(splitTxt[0], *offsets) if offsets else RegName(splitTxt[0])

class Holder:
    def __init__(self, idx=None, name=None):
        if name:
            self.name = _generateRegName(name)
            assert(idx == None)
        else:
            self.name = name
            assert(name == None)
        self.idx    = idx

########################################
# mfma
########################################

def _dataTypeNameAbbrevToInstType(abbrev: str) -> InstType:
    if abbrev == 'f64':
        return InstType.INST_F64
    elif abbrev == 'f32':
        return InstType.INST_F32
    elif abbrev == 'f16':
        return InstType.INST_F16
    elif abbrev == 'i32':
        return InstType.INST_I32
    elif abbrev == 'i8':
        return InstType.INST_I8
    elif abbrev == 'bf16':
        return InstType.INST_BF16
    else:
        assert("Unsupported data type.")
    return InstType.INST_NOTYPE

def dataTypeToMfmaInstTypePair(dataType: DataType, Fp16AltImpl: bool) -> Tuple[InstType, InstType]:
    miInTypeStr      = "bf16" if Fp16AltImpl else dataType.toNameAbbrev()
    miInInstType = _dataTypeNameAbbrevToInstType(miInTypeStr) # v_mfma_[...xK]<InType>
    miOutInstType = _dataTypeNameAbbrevToInstType(dataType.MIOutputTypeNameAbbrev()) # v_mfma_<OutType>..
    return miInInstType, miOutInstType

########################################
# Label Manager
########################################

def magicGenerator(chars=(string.ascii_uppercase + string.digits)):
    return ''.join(random.choice(chars) for _ in range(16))

class LabelManager():
    def __init__(self):
        self.labelDict = dict()

    def addName(self, name):
        if name not in self.labelDict:
            self.labelDict[name] = 0
        else:
            self.labelDict[name] += 1

    def getUniqueName(self):
        name = magicGenerator()
        while 1:
            if name not in self.labelDict:
                break
            name = magicGenerator()
        return self.getName(name)

    def getUniqueNamePrefix(self, prefix):
        name = prefix + "_" + magicGenerator()
        while 1:
            if name not in self.labelDict:
                break
            name = prefix + "_" + magicGenerator()
        return self.getName(name)

    def getName(self, name):
        if name not in self.labelDict:
            self.labelDict[name] = 0
        return name + "_" + str(self.labelDict[name])

    def getNameInc(self, name):
        self.addName(name)
        if self.labelDict[name] == 0:
            return name
        return name + "_" + str(self.labelDict[name])

    def getNameIndex(self, name, index):
        if name not in self.labelDict:
            printExit("You have to add a label first to get a label name with specific index.")
        if index > self.labelDict[name]:
            printExit("The index %u exceeded. (> %u)"%(index, self.labelDict[name]))
        return name + "_" + str(index)

########################################
# Math
########################################

def log2(x):
    return int(log(x, 2) + 0.5)

def ceilDivide(numerator, denominator):
    # import pdb
    # pdb.set_trace()
    try:
        if numerator < 0 or denominator < 0:
            raise ValueError
    except ValueError:
        print("ERROR: Can't have a negative register value")
        return 0
    try:
        div = int((numerator+denominator-1) // denominator)
    except ZeroDivisionError:
        print("ERROR: Divide by 0")
        return 0
    return div

def roundUpToNearestMultiple(numerator, denominator):
    return ceilDivide(numerator,denominator)*int(denominator)

########################################
# Others
########################################

def replaceHolder(module, dst):
    if isinstance(module, Module):
        for item in module.items():
            replaceHolder(item, dst)
    elif isinstance(module, Instruction):
        for param in module.getParams():
            if isinstance(param, HolderContainer):
                param.setRegNum(dst)
                param = param.getCopiedRC()
    elif isinstance(module, SWaitCnt):
        assert(isinstance(dst, int))
        if isinstance(module.vmcnt, HolderContainer):
            module.vmcnt = module.vmcnt.holderIdx + dst
        if isinstance(module.lgkmcnt, HolderContainer):
            module.lgkmcnt = module.lgkmcnt.holderIdx + dst
        if isinstance(module.vscnt, HolderContainer):
            module.vscnt = module.vscnt.holderIdx + dst

    return module

def getAsmCompileArgs(assemblerPath: str, codeObjectVersion: str, archHasV3: bool, \
    isa: Tuple[int, int, int], wavefrontSize: int, \
    sourceFileName: str, objectFileName: str, *moreArgs):
    launcher = shlex.split(os.environ.get('Tensile_ASM_COMPILER_LAUNCHER', ''))
    rv = launcher + [assemblerPath, '-x', 'assembler', '-target', 'amdgcn-amd-amdhsa']

    if archHasV3:
        rv += ['-mcode-object-version=2' if codeObjectVersion == "V2" else '-mcode-object-version=4']

    rv += ['-mcpu=' + getGfxName(isa)]

    if wavefrontSize == 64:
        rv += ['-mwavefrontsize64']
    else:
        rv += ['-mno-wavefrontsize64']

    rv += moreArgs
    rv += ['-c', '-o', objectFileName, sourceFileName]
    return rv

def getAsmLinkCodeObjectArgs(assemblerPath: str, objectFileNames: List[str], \
    coFileName: str, *moreArgs):
    rv = [assemblerPath, '-target', 'amdgcn-amd-amdhsa']
    rv += moreArgs
    rv += ['-o', coFileName] + objectFileNames
    return rv
