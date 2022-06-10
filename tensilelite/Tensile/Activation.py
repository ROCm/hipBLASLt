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

import ctypes
import math
import struct
from collections import OrderedDict

from .TensileInstructions import Module, TextBlock, HolderContainer, RegisterContainer, \
                          VCC, vgpr, sgpr, Holder, fastdeepcopy
from .TensileInstructions.Enums import *
from .TensileInstructions.Instructions import *
from .Common import printExit, printWarning

################################################################################
# How to add an activation
# 1. Add a new type in ActivationType
# 2. Create a new getXXXAssembly function in class Activation
# 3. Add if-else condition in generateAssembly in class Activation
# 4. Add if-else condition in generateInlineAssemblyBody in class
#    ActivationInline
#
# Helper function(s)
# 1. getRegAndInitAssembly
#    ```
#    getRegAndInitAssembly(<v for vgpr/ s for sgpr>,
#                          <False for reg pool/ True for tmp reg pool>,
#                          <size of reg>,
#                          <init value>,
#                          <key>,
#                          <comment>)
#    ```
#    Returns
#    1. sgprinf: The original checkOut return value
#    2. regInitStr: The init instruction string
#
#    Example,
#    ```
#    sgprinf, regInitStr = self.getRegAndInitAssembly('s', False, 1, \
#        "0x3f4c422a", "FloatGeluK0", "float gelu k0")
#    ```
#    this will generate ``regInitStr`` as
#    ```
#    s_mov_b32 sXX, "0x3f4c422a" // float16 max
#    ```
#    if the key "FloatGeluK0" is not found in sgprDict
# 2. class ActivationRegisterPool
#    A wrapper of RegisterPool. All the checkOut-ed registers will be checkIn-ed
#    at the end of the numBatches for loop.
#    When ActivationType is set to 'all', the registers will be checkIn-ed after
#    activation's gwvw for loop.
################################################################################

################################################################################
# This is the ActivationType class
# stringList:
#   This list stores the names of extra arguments, e.g.
#   y = (x > 0) ? x : x * alpha
# lookup:
#   This dict stores the supported activation types as keys and number of
#   arguments as values. Insert any new type before 'none' and 'all'. The
#   sequence of the table should match the enum in Activation.hpp.
#
# To add an activation type, see the instruction in Activation.py.
################################################################################

class ActivationAvailable:
    def __init__(self, canHalf=False, canSingle=False, canDouble=False, canBFloat16=False, canInt8=False, canInt16=False, canInt32=False):
        self.half = canHalf
        self.single = canSingle
        self.double = canDouble
        self.bfloat16 = canBFloat16
        self.int8 = canInt8
        self.int16 = canInt16
        self.int32 = canInt32

class ActivationTypeRegister:
    def __init__(self, name, extraArgs, canHalf=False, canSingle=False, canDouble=False, canBFloat16=False, canInt8=False, canInt16=False, canInt32=False):
        self.name = name
        self.extraArgs = extraArgs
        self.can = ActivationAvailable(canHalf, canSingle, canDouble, canBFloat16, canInt8, canInt16, canInt32)
    def typeAvailable(self, dataType):
        if dataType.isHalf() and self.can.half:
            return True
        elif dataType.isSingle() and self.can.single:
            return True
        elif dataType.isDouble() and self.can.double:
            return True
        elif dataType.isBFloat16() and self.can.bfloat16:
            return True
        elif dataType.isInt8() and self.can.int8:
            return True
        elif dataType.isInt32() and self.can.int32:
            return True
        return False

class ActivationType:
    stringList = ['alpha', 'beta', 'gamma', 'delta' ]
    # Exp is only for verification. So we will not return exp in the supported list.
                                                                             # Half,Single,Double,BFloat16,  Int8, Int16, Int32
    lookupVeri = OrderedDict([('exp',       ActivationTypeRegister('exp', 0,       True,  True, False,   False, False, False, False)) ])

    # Note: The BFloat16 gemm uses Single type activations. The int8 gemm uses int32 type activations.
                                                                                 # Half,Single,Double,BFloat16,  Int8, Int16, Int32
    lookup = OrderedDict([('none',        ActivationTypeRegister('none', 0,        True,  True,  True,    True,  True,  True,  True)), \
                          ('abs',         ActivationTypeRegister('abs', 0,         True,  True,  True,    True, False, False,  True)), \
                          ('clippedrelu', ActivationTypeRegister('clippedrelu', 2, True,  True,  True,   False, False, False,  True)), \
                          ('gelu',        ActivationTypeRegister('gelu', 0,        True,  True, False,   False, False, False, False)), \
                          ('leakyrelu',   ActivationTypeRegister('leakyrelu', 1,   True,  True,  True,   False, False, False,  True)), \
                          ('relu',        ActivationTypeRegister('relu', 0,        True,  True,  True,   False, False, False,  True)), \
                          ('sigmoid',     ActivationTypeRegister('sigmoid', 0,     True,  True, False,   False, False, False, False)), \
                          ('tanh',        ActivationTypeRegister('tanh', 2,        True,  True, False,   False, False, False, False)), \
                          ('all',         ActivationTypeRegister('all', 0)) ])
    def __init__(self, value):
        if isinstance(value, str):
            strValue = value.lower()
            if strValue in self.lookup:
                self.value = strValue
            elif strValue in self.lookupVeri:
                self.value = strValue
            else:
                raise RuntimeError("Unrecognized activation type %s"%value)
        elif isinstance(value, ActivationType):
            self.value = value.value
        else:
            raise RuntimeError("Unrecognized input type %s, should be string or ActivationType"%str(value))
    def getAdditionalArgNum(self):
        if self.value == 'all':
            maxArgNum = 0
            for key, activationInst in self.lookup.items():
                maxArgNum = max(maxArgNum, activationInst.extraArgs)
            return maxArgNum
        elif self.value in self.lookup:
            return self.lookup[self.value].extraArgs
        return 0
    def getAdditionalArgStringList(self, addPrefix=True):
        list = []
        for i in range(0, self.getAdditionalArgNum()):
            if addPrefix:
                list.append("activation" + self.stringList[i].capitalize())
            else:
                list.append(self.stringList[i])
        return list
    @classmethod
    def getEnumIndex(cls, enumStr):
        return list(cls.lookup.keys()).index(enumStr)
    @classmethod
    def getEnumStrList(cls, dataType, includeNone = True):
        enumList = []
        for key, activationInst in cls.lookup.items():
            if (((key != 'none') or includeNone) and (key != 'all')):
                if activationInst.typeAvailable(dataType):
                    enumList.append(key)
        if not enumList:
            printWarning("No available activation for this data type %s.\n"%str(dataType))
        return enumList
    def state(self): return self.value.capitalize()
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return self.value.capitalize()
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other.lower()
        elif isinstance(other, ActivationType):
            return self.value == other.value
        else:
            raise RuntimeError("Unrecognized type in rhs, should be string or ActivationType")
    def toEnum(self):
        return self.value.capitalize()

ActivationMagicNumbers = {"FloatGeluK0": 0x3f4c422a, \
                          "FloatGeluK1": 0x3d372713, \
                          "Float16GeluK1": 0x29b9 }

# float32 union
class floatUnion(ctypes.Union):
    _fields_ = [('u', ctypes.c_uint), ('f', ctypes.c_float)]

class ActivationModule:
    ################################################################################
    ################################################################################
    ###
    ###   Public Functions
    ###
    ################################################################################
    ################################################################################

    def __init__(self) -> None:
        self.usePK = True
        self.vgprCounter = 0
        self.sgprCounter = 0
        self.saturateI8 = False
        self.vgprPrefixFormat = ""

    # Public function
    def getModule(self, cDataType, activationType, vgprIdx):
        module = ""
        self.resetGprCounter()
        if (activationType == 'abs'):
            module = self.getAbsModule(cDataType, vgprIdx)
        elif (activationType == 'clippedrelu'):
            module = self.getClippedReluModule(cDataType, vgprIdx, "activationAlpha", "activationBeta")
        elif (activationType == 'exp'):
            module = self.getExpModule(cDataType, vgprIdx)
        elif (activationType == 'gelu'):
            module = self.getGeluModule(cDataType, vgprIdx)
        elif (activationType == 'leakyrelu'):
            module = self.getLeakyReluModule(cDataType, vgprIdx, "activationAlpha")
        elif (activationType == 'relu'):
            module = self.getReluModule(cDataType, vgprIdx)
        elif (activationType == 'sigmoid'):
            module = self.getSigmoidModule(cDataType, vgprIdx)
        elif (activationType == 'tanh'):
            module = self.getTanhModule(cDataType, vgprIdx, "activationAlpha", "activationBeta")
        elif (activationType == 'none'):
            module = Module("No activation")
        else:
            module = Module("%s not implemented"%activationType)

        return self.postProcess(cDataType, module)

    def getAllGprUsage(self, cDataType) -> dict:
        usage = {}
        enumList = ActivationType.getEnumStrList(cDataType)
        for enumStr in enumList:
            _ = self.getModule(cDataType, enumStr, 0) # dummy vgpr
            usage[enumStr] = {"vgpr": self.vgprCounter, "sgpr": self.sgprCounter}
        return usage

    def postProcess(self, cDataType, module):
        CombineInstructions(module)
        module = ConvertCoeffToHex(module, cDataType, self.usePK)
        return module

    def assignGpr(self, module, vgprIdx, sgprIdx):
        patternPrefix = ["v", "s"]
        gprIdx = [vgprIdx, sgprIdx]
        for idx, pf in enumerate(patternPrefix):
            module = HolderToGpr(module, gprIdx[idx], pf)
        return module

    def setUsePK(self, usePK):
        self.usePK = usePK

    def setSaturationForInt8(self, sat):
        self.saturateI8 = sat

    def setVgprPrefixFormat(self, formatting):
        self.vgprPrefixFormat = formatting

    ################################################################################
    ################################################################################
    ###
    ###   Internal Helper Functions
    ###
    ################################################################################
    ################################################################################

    def resetGprCounter(self):
        self.vgprCounter = 0
        self.sgprCounter = 0

    def getVgpr(self, num):
        value = self.vgprCounter
        self.vgprCounter += num
        return value

    def getSgpr(self, num):
        value = self.sgprCounter
        self.sgprCounter += num
        return value

    def vgprPrefix(self, *args):
        if isinstance(args[0], int) and self.vgprPrefixFormat:
            vgprStr = self.vgprPrefixFormat%args[0]
        else:
            vgprStr = args[0]

        if len(args) == 1:
            return vgpr(vgprStr)
        else:
            args = args[1]
            return vgpr(vgprStr, args)

    ################################################################################
    ################################################################################
    ###
    ###   Activation Functions
    ###
    ################################################################################
    ################################################################################

    def getAbsModule(self, cDataType, vgprIdx):
        module = Module("Abs")
        if cDataType.isHalf() or cDataType.isBFloat16():
            absMagic = "0x7fff7fff" if self.usePK else "0x7fff"
            module.add(VAndB32(dst=self.vgprPrefix(vgprIdx), src0=absMagic, src1=self.vgprPrefix(vgprIdx), comment="Remove sign bit"))
        elif cDataType.isSingle():
            module.add(VAndB32(dst=self.vgprPrefix(vgprIdx), src0="0x7fffffff", src1=self.vgprPrefix(vgprIdx), comment="Remove sign bit"))
        elif cDataType.isDouble():
            module.add(VAndB32(dst=self.vgprPrefix(vgprIdx+1), src0="0x7fffffff", src1=self.vgprPrefix(vgprIdx+1), comment="Remove sign bit"))
        elif cDataType.isInt32():
            vgprTemp = self.getVgpr(1)
            module.add(VSubI32(dst=vgpr(Holder(idx=vgprTemp)), src0=0, src1=self.vgprPrefix(vgprIdx), comment="x2 = -x"))
            if self.saturateI8:
                vgprTemp2 = self.getVgpr(1)
                module.add(VMovB32(dst=vgpr(Holder(idx=vgprTemp2)), src=hex(127), comment="value = 127"))
                module.add(VMed3I32(dst=self.vgprPrefix(vgprIdx), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), src2=vgpr(Holder(idx=vgprTemp2)), comment="y = min(127, max(x, x2))"))
            else:
                module.add(VMaxI32(dst=self.vgprPrefix(vgprIdx), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIdx), comment="y = max(x, x2)"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getClippedReluModule(self, cDataType, vgprIdx, activationAlpha, activationBeta):
        module = Module("ClippedRelu")
        if cDataType.isHalf():
            for i in range(0, 2):
                select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                module.add(VCmpGEF16(dst=VCC(), src0=self.vgprPrefix(vgprIdx), src1=sgpr(activationAlpha), \
                           sdwa=SDWAModifiers(src0_sel=select_bit, src1_sel=SelectBit.WORD_0), comment="x > alpha?"))
                module.add(VMinF16(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprIdx), \
                           sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, \
                                              src0_sel=select_bit, src1_sel=select_bit), \
                           comment="min(x, beta)"))
                module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=0.0, src1=self.vgprPrefix(vgprIdx), \
                           sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, \
                                              src0_sel=select_bit, src1_sel=select_bit), \
                           comment="set x to 0 if < alpha"))
            module.add(SNop(waitState=0, comment="1 wait states")) # workaround for emulator
        elif cDataType.isSingle():
            module.add(VCmpGEF32(dst=VCC(), src0=self.vgprPrefix(vgprIdx), src1=sgpr(activationAlpha), comment="x >= alpha ?"))
            module.add(VMinF32(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprIdx), comment="min(x, beta)"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=0.0, src1=self.vgprPrefix(vgprIdx), comment="set x to 0 if < alpha"))
        elif cDataType.isDouble():
            module.add(VCmpGEF64(dst=VCC(), src0=self.vgprPrefix(vgprIdx, 2), src1=sgpr(activationAlpha, 2), comment="x >= alpha ?"))
            module.add(VMinF64(dst=self.vgprPrefix(vgprIdx, 2), src0=sgpr(activationBeta, 2), src1=self.vgprPrefix(vgprIdx, 2), comment="min(x, beta)"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=0, src1=self.vgprPrefix(vgprIdx), comment="set x to 0 if < 0"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx+1), src0=0, src1=self.vgprPrefix(vgprIdx+1), comment="set x to 0 if < 0"))
        elif cDataType.isInt32():
            module.add(VCmpGEI32(dst=VCC(), src0=self.vgprPrefix(vgprIdx), src1=sgpr(activationAlpha), comment="x >= alpha ?"))
            module.add(VMinI32(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprIdx), comment="min(x, beta)"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=0.0, src1=self.vgprPrefix(vgprIdx), comment="set x to 0 if < alpha"))
        return module

    def getExpModule(self, cDataType, vgprIdx):
        module = Module("Exp")
        if cDataType.isHalf():
            sgprMagic = self.getSgpr(1)
            module.add(SMovB32(dst=sgpr(Holder(idx=sgprMagic)), src=math.log(math.e,2), comment="exp magic"))
            if self.usePK:
                module.add(VMulPKF16(dst=self.vgprPrefix(vgprIdx), src0=sgpr(Holder(idx=sgprMagic)), src1=self.vgprPrefix(vgprIdx), comment="exp step 1"))
                for i in range(0, 2):
                    select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                    module.add(VExpF16(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), \
                                       sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, \
                                                          src0_sel=select_bit), \
                                       comment="exp step 2"))
                    module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
            else:
                module.add(VMulF16(dst=self.vgprPrefix(vgprIdx), src0=sgpr(Holder(idx=sgprMagic)), src1=self.vgprPrefix(vgprIdx), comment="exp step 1"))
                module.add(VExpF16(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), comment="exp step 2"))
                module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
        elif cDataType.isSingle():
            module.add(VMulF32(dst=self.vgprPrefix(vgprIdx), src0=math.log(math.e,2), src1=self.vgprPrefix(vgprIdx), comment="exp step 1"))
            module.add(VExpF32(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), comment="exp step 2" ))
            module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getGeluModule(self, cDataType, vgprIdx):
        module = Module("Gelu")
        # Gelu(x) = 0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x)))
        if cDataType.isHalf():
            flt16GeluK1Str = HexToStr(cDataType, self.usePK, ActivationMagicNumbers["Float16GeluK1"])
            sgprMagicK1 = self.getSgpr(1)
            module.add(SMovB32(dst=sgpr(Holder(idx=sgprMagicK1)), src=flt16GeluK1Str, comment="Float16GeluK1" ))
            vgprTemp = self.getVgpr(1)
            if self.usePK:
                module.add(VMulPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=self.vgprPrefix(vgprIdx), comment="x * x" ))
                module.add(VFmaPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=vgpr(Holder(idx=vgprTemp)), src1=sgpr(Holder(idx=sgprMagicK1)), src2=1.0, \
                                     vop3=VOP3PModifiers(op_sel_hi=[1,1,0,1]), comment="x^2 * k1 + 1"))
                module.add(VMulPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), comment="x * (x^2 * k1 + 1)"))
                coef = floatUnion(u=ActivationMagicNumbers["FloatGeluK0"])
                module.add(VMulPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=coef.f, src1=vgpr(Holder(idx=vgprTemp)), comment="k0 * x * (x^2 * k1 + 1)"))
                module.add(self.getTanhModule(cDataType, Holder(idx=vgprTemp), "", ""))
                module.add(VAddPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=1.0, src1=vgpr(Holder(idx=vgprTemp)), \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,1]), comment="1 + tanh(...)" ))
                module.add(VMulPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), comment="x * (1 + tanh(...))"))
                module.add(VMulPKF16(dst=self.vgprPrefix(vgprIdx), src0=0.5, src1=vgpr(Holder(idx=vgprTemp)), \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,1]), comment="0.5 * x * (1 + tanh(...))"))
            else:
                module.add(VMulF16(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=self.vgprPrefix(vgprIdx), comment="x * x" ))
                module.add(VFmaF16(dst=vgpr(Holder(idx=vgprTemp)), src0=vgpr(Holder(idx=vgprTemp)), src1=sgpr(Holder(idx=sgprMagicK1)), src2=1.0, comment="x^2 * k1 + 1"))
                module.add(VMulF16(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), comment="x * (x^2 * k1 + 1)"))
                coef = floatUnion(u=ActivationMagicNumbers["FloatGeluK0"])
                module.add(VMulF16(dst=vgpr(Holder(idx=vgprTemp)), src0=coef.f, src1=vgpr(Holder(idx=vgprTemp)), comment="k0 * x * (x^2 * k1 + 1)"))
                module.add(self.getTanhModule(cDataType, Holder(idx=vgprTemp), "", ""))
                module.add(VAddF16(dst=vgpr(Holder(idx=vgprTemp)), src0=1.0, src1=vgpr(Holder(idx=vgprTemp)), comment="1 + tanh(...)" ))
                module.add(VMulF16(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), comment="x * (1 + tanh(...))"))
                module.add(VMulF16(dst=self.vgprPrefix(vgprIdx), src0=0.5, src1=vgpr(Holder(idx=vgprTemp)), comment="0.5 * x * (1 + tanh(...))"))
        elif cDataType.isSingle():
            vgprTemp = self.getVgpr(1)
            flt16GeluK1Str = HexToStr(cDataType, self.usePK, ActivationMagicNumbers["FloatGeluK1"])
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp)), src0=flt16GeluK1Str, src1=self.vgprPrefix(vgprIdx), comment="k1 * x"))
            module.add(VFmaF32(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), src2=1.0, comment="1 + (k1 * x * x)"))
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), comment="x * (1 + k1 * x * x)"))
            coef = floatUnion(u=ActivationMagicNumbers["FloatGeluK0"])
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp)), src0=coef.f, src1=vgpr(Holder(idx=vgprTemp)), comment="k0 * x * (x^2 * k1 + 1)"))
            module.add(self.getTanhModule(cDataType, Holder(idx=vgprTemp), "", ""))
            module.add(VAddF32(dst=vgpr(Holder(idx=vgprTemp)), src0=1.0, src1=vgpr(Holder(idx=vgprTemp)), comment="1 + tanh(...)" ))
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp)), src0=self.vgprPrefix(vgprIdx), src1=vgpr(Holder(idx=vgprTemp)), comment="x * (1 + tanh(...))"))
            module.add(VMulF32(dst=self.vgprPrefix(vgprIdx), src0=0.5, src1=vgpr(Holder(idx=vgprTemp)), comment="0.5 * x * (1 + tanh(...))"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getLeakyReluModule(self, cDataType, vgprIdx, activationAlpha):
        module = Module("LeakyRelu")
        if cDataType.isHalf():
            vgprTemp = self.getVgpr(1)
            module.add(VMulPKF16(dst=vgpr(Holder(idx=vgprTemp)), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIdx), comment="tmp = x * alpha"))
            for i in range(0, 2):
                select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                module.add(VCmpGEF16(dst=VCC(), src0=self.vgprPrefix(vgprIdx), src1=0.0, \
                                     sdwa=SDWAModifiers(src0_sel=select_bit, src1_sel=SelectBit.WORD_0), \
                                     comment="x > 0 ?"))
                module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIdx), \
                                       sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, \
                                                          src0_sel=select_bit, src1_sel=select_bit), \
                                       comment="set x to tmp if < 0"))
        elif cDataType.isSingle():
            vgprTemp = self.getVgpr(1)
            module.add(VMulF32(dst=vgpr(Holder(idx=vgprTemp)), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIdx), comment="tmp = x * alpha"))
            module.add(VCmpGEF32(dst=VCC(), src0=self.vgprPrefix(vgprIdx), src1=0.0, comment="x >= 0 ?"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIdx), comment="set x to tmp if < 0"))
        elif cDataType.isDouble():
            vgprTemp = self.getVgpr(2)
            module.add(VMulF64(dst=vgpr(Holder(idx=vgprTemp), 2), src0=sgpr(activationAlpha, 2), src1=self.vgprPrefix(vgprIdx, 2), comment="tmp = x * alpha"))
            module.add(VCmpGEF64(dst=VCC(), src0=self.vgprPrefix(vgprIdx, 2), src1=0.0, comment="x >= 0 ?"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIdx), comment="set x to tmp if < 0"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx+1), src0=vgpr(Holder(idx=vgprTemp+1)), src1=self.vgprPrefix(vgprIdx+1), comment="set x to tmp if < 0"))
        elif cDataType.isInt32():
            vgprTemp = self.getVgpr(1)
            module.add(VMulLOU32(dst=vgpr(Holder(idx=vgprTemp)), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIdx), comment="tmp = x * alpha"))
            module.add(VCmpGEI32(dst=VCC(), src0=self.vgprPrefix(vgprIdx), src1=0, comment="x >= 0 ?"))
            module.add(VCndMaskB32(dst=self.vgprPrefix(vgprIdx), src0=vgpr(Holder(idx=vgprTemp)), src1=self.vgprPrefix(vgprIdx), comment="set x to tmp if < 0"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getReluModule(self, cDataType, vgprIdx):
        module = Module("LeakyRelu")
        if cDataType.isHalf():
            module.add(VMaxPKF16(dst=self.vgprPrefix(vgprIdx), src0=self.vgprPrefix(vgprIdx), src1=0, comment="x = max(0, x)" ))
        elif cDataType.isSingle():
            module.add(VMaxF32(dst=self.vgprPrefix(vgprIdx), src0=self.vgprPrefix(vgprIdx), src1=0, comment="x = max(0, x)" ))
        elif cDataType.isDouble():
            module.add(VMaxF64(dst=self.vgprPrefix(vgprIdx, 2), src0=self.vgprPrefix(vgprIdx, 2), src1=0, comment="x = max(0, x)" ))
        elif cDataType.isInt32():
            if self.saturateI8:
                vgprTemp = self.getVgpr(1)
                module.add(VMovB32(dst=vgpr(Holder(idx=vgprTemp)), src=hex(127), comment="value = 127"))
                module.add(VMed3I32(dst=self.vgprPrefix(vgprIdx), src0=self.vgprPrefix(vgprIdx), src1=0, src2=vgpr(Holder(idx=vgprTemp)), comment="x = min(127, max(0, x))" ))
            else:
                module.add(VMaxI32(dst=self.vgprPrefix(vgprIdx), src0=self.vgprPrefix(vgprIdx), src1=0, comment="x = max(0, x)" ))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getSigmoidModule(self, cDataType, vgprIdx):
        module = Module("Sigmoid")
        if cDataType.isHalf():
            if self.usePK:
                module.add(VMulPKF16(dst=self.vgprPrefix(vgprIdx), src0=-1.0, src1=self.vgprPrefix(vgprIdx), comment=" x = -x"))
                module.add(self.getExpModule(cDataType, vgprIdx))
                module.add(VAddPKF16(dst=self.vgprPrefix(vgprIdx), src0=1.0, src1=self.vgprPrefix(vgprIdx), \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,1]), comment="1 + exp(-x)"))
                for i in range(0, 2):
                    select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                    module.add(VRcpF16(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), \
                                       sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, src0_sel=select_bit), \
                                       comment="1 / (1 + exp(-x))"))
                module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
            else:
                module.add(VMulF16(dst=self.vgprPrefix(vgprIdx), src0=-1.0, src1=self.vgprPrefix(vgprIdx), comment=" x = -x"))
                module.add(self.getExpModule(cDataType, vgprIdx))
                module.add(VAddF16(dst=self.vgprPrefix(vgprIdx), src0=1.0, src1=self.vgprPrefix(vgprIdx), comment="1 + exp(-x)"))
                module.add(VRcpF16(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), comment="1 / (1 + exp(-x))"))
                module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
        elif cDataType.isSingle():
            module.add(VMulF32(dst=self.vgprPrefix(vgprIdx), src0=-1.0, src1=self.vgprPrefix(vgprIdx), comment=" x = -x"))
            module.add(self.getExpModule(cDataType, vgprIdx))
            module.add(VAddF32(dst=self.vgprPrefix(vgprIdx), src0=1.0, src1=self.vgprPrefix(vgprIdx), comment="1 + exp(-x)" ))
            module.add(VRcpF32(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), comment="1 / (1 + exp(-x))" ))
            module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

    def getTanhModule(self, cDataType, vgprIdx, activationAlpha, activationBeta):
        module = Module("Tanh")
        if cDataType.isHalf():
            # We don't need s_pack_ll_b32_b16 cause the input is already duplicated
            if self.usePK:
                if activationAlpha:
                    module.add(VMulPKF16(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIdx), comment="x * alpha"))
                module.add(VMulPKF16(dst=self.vgprPrefix(vgprIdx), src0=2, src1=self.vgprPrefix(vgprIdx), comment=" x = 2 * x"))
                module.add(self.getExpModule(cDataType, vgprIdx))
                module.add(VAddPKF16(dst=self.vgprPrefix(vgprIdx), src0=1.0, src1=self.vgprPrefix(vgprIdx), \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,1]), comment="e^2x + 1"))
                for i in range(0, 2):
                    select_bit = SelectBit.WORD_0 if i == 0 else SelectBit.WORD_1
                    vgprCtrl = "dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE src0_sel:WORD_%d"%(i, i)
                    module.add(VRcpF16(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), \
                                       sdwa=SDWAModifiers(dst_sel=select_bit, dst_unused=UnusedBit.UNUSED_PRESERVE, \
                                                          src0_sel=select_bit), \
                                       comment="1 / (1 + exp(-x))"))
                    module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
                module.add(VFmaPKF16(dst=self.vgprPrefix(vgprIdx), src0=-2.0, src1=self.vgprPrefix(vgprIdx), src2=1.0, \
                                     vop3=VOP3PModifiers(op_sel_hi=[0,1,0,1]), comment="tanh(x) = (1 / (e^2x + 1)) * (-2) + 1"))
                if activationBeta:
                    module.add(VMulPKF16(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprIdx), comment="beta * tanh(x)"))
            else:
                if activationAlpha:
                    module.add(VMulF16(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIdx), comment="x * alpha"))
                module.add(VMulF16(dst=self.vgprPrefix(vgprIdx), src0=2, src1=self.vgprPrefix(vgprIdx), comment=" x = 2 * x"))
                module.add(self.getExpModule(cDataType, vgprIdx))
                module.add(VAddF16(dst=self.vgprPrefix(vgprIdx), src0=1.0, src1=self.vgprPrefix(vgprIdx), comment="e^2x + 1"))
                module.add(VRcpF16(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), comment="1 / (1 + exp(-x))"))
                module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
                module.add(VFmaF16(dst=self.vgprPrefix(vgprIdx), src0=-2.0, src1=self.vgprPrefix(vgprIdx), src2=1.0, comment="tanh(x) = (1 / (e^2x + 1)) * (-2) + 1"))
                if activationBeta:
                    module.add(VMulF16(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprIdx), comment="beta * tanh(x)"))
        elif cDataType.isSingle():
            if activationAlpha:
                module.add(VMulF32(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationAlpha), src1=self.vgprPrefix(vgprIdx), comment="x * alpha"))
            module.add(VMulF32(dst=self.vgprPrefix(vgprIdx), src0=2, src1=self.vgprPrefix(vgprIdx), comment=" x = 2 * x"))
            module.add(self.getExpModule(cDataType, vgprIdx))
            module.add(VAddF32(dst=self.vgprPrefix(vgprIdx), src0=1.0, src1=self.vgprPrefix(vgprIdx), comment="e^2x + 1"))
            module.add(VRcpF32(dst=self.vgprPrefix(vgprIdx), src=self.vgprPrefix(vgprIdx), comment="1 / (e^2x + 1)"))
            module.add(SNop(waitState=0, comment="1 wait states")) #workaround for emulator
            module.add(VFmaF32(dst=self.vgprPrefix(vgprIdx), src0=-2.0, src1=self.vgprPrefix(vgprIdx), src2=1.0, comment="(-2) * (1 / (e^2x + 1)) + 1"))
            if activationBeta:
                module.add(VMulF32(dst=self.vgprPrefix(vgprIdx), src0=sgpr(activationBeta), src1=self.vgprPrefix(vgprIdx), comment="beta * tanh(x)"))
        else:
            raise RuntimeError("Unsupported data type %s."%cDataType.toDevice("HIP"))
        return module

################################################################################
################################################################################
###
###   Post Functions
###
################################################################################
################################################################################

__FUSE_MAGIC_NAME__ = "dbiw@I$HONIhnjf4_fused"

# Public
def CombineInstructions(module, fuseDebug = False):
    moduleAndIndex = dict()
    CombineInstructionsBetweenModules(module, moduleAndIndex, fuseDebug)
    # Remove Empty Blocks
    module = RemoveEmptyBlocks(module)
    return module

# Does not support modules with branches
def CombineInstructionsBetweenModules(module, moduleAndIndex, fuseDebug):
    index = 0
    while index < len(module.items()):
        item = module.items()[index]
        if isinstance(item, Module):
            CombineInstructionsBetweenModules(item, moduleAndIndex, fuseDebug)
            index = module.items().index(item)
        elif isinstance(item, SNop):
            pass
        elif isinstance(item, Instruction):
            newItem = item
            if moduleAndIndex:
                newItem = FuseInstruction(item, moduleAndIndex, fuseDebug)
                index = module.items().index(newItem)
            if isinstance(newItem.dst, RegisterContainer):
                # Update the dict
                moduleAndIndex[newItem.dst] = newItem
        index += 1

def RemoveEmptyBlocks(module):
    for idx, item in enumerate(module.items()):
        if isinstance(item, Module):
            newItem = RemoveEmptyBlocks(item)
            module.items()[idx] = newItem
    if len(module.items()) == 1 and isinstance(module.items()[0], Module):
        return module.items()[0]
    return module

def FuseInstruction(currentInst, moduleAndIndex, fuseDebug):
    assert(isinstance(currentInst, Instruction))
    newInst = None
    # Fuses if v_add_f16 to v_fma_f16 if v_add_f16 is a self adding instruction.
    # Currently, we only fuse when the vgpr is add by 1 in both instructions.
    # ex. v_add_f16 v0, 1.0, v0
    #     +  v_fma_f16 v0, -2.0, v0, 1.0
    #     => v_fma_f16 v0, -2.0, v0, 2.0
    if isinstance(currentInst, VAddF16) or isinstance(currentInst, VAddPKF16) or \
       isinstance(currentInst, VAddF32):
        isPK = isinstance(currentInst, VAddPKF16)
        outVgpr = currentInst.dst
        addConst = ""
        isSelfAddConst = False
        for param in currentInst.src:
            if param == outVgpr:
                isSelfAddConst = True
            if (isinstance(param, float) or isinstance(param, int)):
                if param == 1:
                    addConst = param

        if isSelfAddConst and addConst:
            oldInst = moduleAndIndex.get(outVgpr)
            if isinstance(oldInst, Instruction):
                if currentInst.instType == InstType.INST_F16:
                    func = VFmaPKF16 if isPK else VFmaF16
                elif currentInst.instType == InstType.INST_F32:
                    func = VFmaF32
                else:
                    assert("You should not reach here.")
                if isinstance(oldInst, func) and oldInst.src[2] == 1.0:
                    # Cannot fuse if the target instruction has any rvalue reassigned or its lvalue
                    # used before the current instruction
                    if not FindAssignAndUse(oldInst, currentInst, outVgpr, outVgpr):
                        newInst = fastdeepcopy(oldInst)
                        newInst.src[2] = addConst + newInst.src[2]
                        newInst.comment += " ( + 1 (fused))"
                        replaceInst(currentInst, newInst, fuseDebug)
                        removeOldInst(oldInst, currentInst, newInst, fuseDebug)
    # Fuses if v_mul_f16 to v_mul_f16 if the later one is a self multiplying instruction.
    # Only fuses when both instructions multiply constant
    elif isinstance(currentInst, VMulF16) or isinstance(currentInst, VMulPKF16) or \
         isinstance(currentInst, VMulF32) or isinstance(currentInst, VMulF64):
        isPK = isinstance(currentInst, VMulPKF16)
        outVgpr = currentInst.dst
        mulConst = ""
        newFuseInst = ""
        isSelfMulConst = False
        for param in currentInst.src:
            if param == outVgpr:
                isSelfMulConst = True
            # The constant may be an sgpr
            if isinstance(param, RegisterContainer) and param.regType == 's':
                oldInst = moduleAndIndex.get(param)
                if isinstance(oldInst, SMovB32):
                    oldparam = oldInst.src[0]
                    if oldInst.dst == param and (isinstance(oldparam, float) or isinstance(oldparam, int)):
                        # Cannot fuse if another instruction is using the same sgpr before a new assignment occurs
                        if not FindUse(oldInst, currentInst, param):
                            mulConst = oldparam
                            newFuseInst = oldInst
            if (isinstance(param, float) or isinstance(param, int)):
                mulConst = param

        if isSelfMulConst and mulConst:
            oldInst = moduleAndIndex.get(outVgpr)
            if isinstance(oldInst, Instruction):
                if currentInst.instType == InstType.INST_F16:
                    func = VMulPKF16 if isPK else VMulF16
                elif currentInst.instType == InstType.INST_F32:
                    func = VMulF32
                elif currentInst.instType == InstType.INST_F64:
                    func = VMulF64
                else:
                    assert("You should not reach here.")

                if isinstance(oldInst, func):
                    # Cannot fuse if the target instruction has any rvalue reassigned or its lvalue
                    # used before the current instruction
                    if not FindAssignAndUse(oldInst, currentInst, outVgpr, outVgpr):
                        for paramIdx, param in enumerate(oldInst.src):
                            if (isinstance(param, float) or isinstance(param, int)):
                                newInst = fastdeepcopy(oldInst)
                                newValue = param * mulConst
                                formatting = " (fused %f)" if isinstance(param, float) else " (fused %d)"
                                if newFuseInst:
                                    newFuseInst.src[0] = newValue
                                    newInst.src[paramIdx] = newFuseInst.dst
                                    newFuseInst.comment += formatting%newValue
                                else:
                                    newInst.src[paramIdx] = newValue
                                newInst.comment += formatting%newValue
                                replaceInst(currentInst, newInst, fuseDebug)
                                removeOldInst(oldInst, currentInst, newInst, fuseDebug)
                                break
    return newInst if newInst else currentInst

# This only works for Activation.py
def FindUse(startInst, targetInst, varTarget):
    _, isUse = FindUseIter(startInst, targetInst, varTarget)
    return isUse

# This only works for Activation.py
def FindUseIter(startItem, targetInst, varTarget):
    module = startItem
    idx = -1
    isEnd = False
    isUse = False
    if isinstance(startItem, Instruction):
        module = startItem.parent
        idx = module.items().index(startItem)
    assert(isinstance(module, Module))
    if idx + 1 < len(module.items()[idx + 1:]):
        for item in module.items()[idx + 1:]:
            if item is targetInst:
                pass
            elif isinstance(item, SNop):
                pass
            elif isinstance(item, Instruction):
                if item.src:
                    for param in item.src:
                        if param == varTarget:
                            isEnd = True
                            isUse = True
                            break
                elif item.dst == varTarget:
                    isEnd = True
                    isUse = False
            elif isinstance(item, Module):
                isEnd, isUse = FindUseIter(item, targetInst, varTarget)
            if isEnd:
                return isEnd, isUse
    return False, isUse

# This only works for Activation.py
def FindAssignAndUse(startInst, endInst, assignVar, useVar):
    _, isUse = FindAssignAndUseIter(startInst, endInst, assignVar, useVar)
    return isUse

# This only works for Activation.py
def FindAssignAndUseIter(startItem, endInst, assignVar, useVar):
    module = startItem
    idx = -1
    isEnd = False
    isUse = False
    if isinstance(startItem, Instruction):
        module = startItem.parent
        idx = module.items().index(startItem)
    assert(isinstance(module, Module))
    if idx + 1 < len(module.items()[idx + 1:]):
        for item in module.items()[idx + 1:]:
            # Use
            if item is endInst:
                isEnd = True
            elif isinstance(item, SNop):
                pass
            elif isinstance(item, Instruction):
                if item.dst == assignVar:
                    isEnd = True
                    isUse = True
                # Check use
                if item.src:
                    for param in item.src:
                        if param == useVar:
                            isEnd = True
                            isUse = True
                            break
            elif isinstance(item, Module):
                isEnd, isUse = FindAssignAndUseIter(item, endInst, assignVar, useVar)
            if isEnd:
                return isEnd, isUse
    return isEnd, isUse

def removeOldInst(removeInst, dstInst, fusedInst, debug):
    module = removeInst.parent
    targetIdx = -1
    for idx, item in enumerate(module.items()):
        if item == removeInst:
            if debug:
                tb = TextBlock("\n/* Fused to block %s + %s -> %s */\n"%(str(removeInst), str(dstInst), str(fusedInst)))
                tb.name = __FUSE_MAGIC_NAME__
                module.items()[idx] = tb
            else:
                targetIdx = idx
            break

    if targetIdx > -1:
        module.removeItemByIndex(targetIdx)

def replaceInst(srcInst, dstInst, debug):
    module = srcInst.parent
    targetIdx = -1
    for idx, item in enumerate(module.items()):
        if item == srcInst:
            if debug:
                dstInst.comment += " (Block replaced %s)"%(str(srcInst))
            targetIdx = idx
            break
    if targetIdx > -1:
        module.replaceItemByIndex(targetIdx, dstInst)

################################################################################
################################################################################
###
###   Helper Functions
###
################################################################################
################################################################################

def getMagic(cDataType, value, isPack=False):
    if cDataType.isDouble():
        printExit("Currently magic does not support double.")
    elif cDataType.isHalf():
        fu = value if isinstance(value, floatUnion) else floatUnion(f=value)
        magicNum = struct.unpack('<H', struct.pack('<e', fu.f))[0]
        if isPack:
            magicNum = ctypes.c_uint(magicNum).value
            magicNum = ((magicNum << 16) | magicNum)
    elif cDataType.isSingle():
        fu = value if isinstance(value, floatUnion) else floatUnion(f=value)
        magicNum = fu.u
    return hex(magicNum)

def getMagicStr(cDataType, value, isPack=False):
    return str(getMagic(cDataType, value, isPack))

def HexToStr(cDataType, isPack, *args):
    if len(args) == 1:
        magicNum = args[0]
        uint32 = ctypes.c_uint(magicNum).value
        if isPack and cDataType.isHalf():
            uint32 = ((uint32 << 16) | uint32)
        hexStr = str(hex(uint32))
    else:
        raise RuntimeError("Currently does not support multiple args.")
    return hexStr

def ConvertCoeffToHex(module, cDataType, isPack):
    if (module.name == "Exp"):
        param = module.items()[0].src[0]
        module.items()[0].src[0] = getMagic(cDataType, param, isPack)
        return module
    for itemIdx, item in enumerate(module.items()):
        if isinstance(item, Module):
            newItem = ConvertCoeffToHex(item, cDataType, isPack)
            module.items()[itemIdx] = newItem
    return module

def HolderToGpr(module, idx, pf):
    for itemIdx, item in enumerate(module.items()):
        if isinstance(item, Module):
            newItem = HolderToGpr(item, idx, pf)
            module.items()[itemIdx] = newItem
        elif isinstance(item, SNop):
            pass
        elif isinstance(item, Instruction):
            if isinstance(item.dst, HolderContainer) and item.dst.regType == pf:
                item.dst.setRegNum(idx)
                item.dst = item.dst.getCopiedRC()
            if item.src:
                for itemIdx, param in enumerate(item.src):
                    if isinstance(param, HolderContainer) and param.regType == pf:
                        param.setRegNum(idx)
                        item.src[itemIdx] = param.getCopiedRC()
    return module

def addSpace(alignStr, str):
  totalLength = len(alignStr) + len(str)
  return '{message: >{width}}'.format(message=str, width=totalLength)

class ActivationInline:
  def __init__(self, dataType) -> None:
    self.dataType = dataType
    self.asmStr = "asm("

  # Public Function
  def generateInlineAssemblyFunction(self, activationType):
    kStr = ""
    if activationType == 'none':
      return kStr

    enumName = "Tensile::ActivationType_%s"%self.dataType.toChar()

    ptrStr = self.dataType.toDevice("HIP")
    names = ""
    if activationType == 'all':
      names += ",\n"
      names += "  %s const activationType"%enumName
    for name in activationType.getAdditionalArgStringList(False):
      names += ",\n"
      names += "  %s const %s"%(ptrStr, name)
    changeLine = "\n  " if names else ""
    kStr += "__device__ inline %s activation(%s%s value%s)\n{\n"%(ptrStr, changeLine, ptrStr, names)
    # function body
    if activationType == 'all':
      for index, enumStr in enumerate(ActivationType.getEnumStrList(self.dataType, includeNone=False)):
        if index == 0:
          kStr += "  if (activationType == %s::%s) {\n"%(enumName, ActivationType(enumStr).toEnum())
        else:
          kStr += " else if (activationType == %s::%s) {\n"%(enumName, ActivationType(enumStr).toEnum())
        kStr += self.generateInlineAssemblyBody(4, enumStr)
        kStr += "  }"
      kStr += "\n"
    else:
      kStr += self.generateInlineAssemblyBody(2, activationType)
    # function body end
    kStr += "  return value;\n"
    kStr += "}\n"
    return kStr

  def replaceGpr(self, module):
    for item in module.items():
        if isinstance(item, SNop):
            pass
        elif isinstance(item, Instruction):
            if isinstance(item.dst, RegisterContainer):
                if not item.dst.regName:
                    item.dst.setInlineAsm(True)
            if item.src:
                for param in item.src:
                    if isinstance(param, RegisterContainer):
                        if not param.regName:
                            param.setInlineAsm(True)
        elif isinstance(item, Module):
            self.replaceGpr(item)

  def getActivationAsmStr(self, activation, module, spaces):
    module = activation.postProcess(self.dataType, module)
    self.replaceGpr(module)
    activation.assignGpr(module, 0, 0)
    module.setInlineAsmPrintMode(True)
    kStr = str(module)
    newStr = ""
    for instStr in kStr.split("\n"):
        if instStr:
            newStr += spaces + instStr + "\n"
    return newStr

  # Internal Function
  def generateInlineAssemblyBody(self, spaces, activationType):
    ptrStr = self.dataType.toDevice("HIP")
    activation = ActivationModule()
    activation.setUsePK(False)
    activation.resetGprCounter()
    kStr = ""
    padSpacesStr = ' ' * spaces
    asm = padSpacesStr + self.asmStr
    if (activationType == 'abs'):
      if self.dataType.isHalf() or self.dataType.isBFloat16():
        unionDataTypeStr = "_Float16" if self.dataType.isHalf() else "BFloat16"
        unionName = "f16_union" if self.dataType.isHalf() else "bf16_union"
        kStr += (padSpacesStr + "union {\n")
        kStr += (padSpacesStr + "  %s f;\n"%unionDataTypeStr)
        kStr += (padSpacesStr + "  short s;\n")
        kStr += (padSpacesStr + "} %s;\n"%unionName)
        kStr += (padSpacesStr + "%s.f = value;\n"%unionName)
        kStr += (padSpacesStr + "%s.s = %s.s & 0x7fff;\n"%(unionName, unionName))
        kStr += (padSpacesStr + "value = %s.f;\n"%unionName)
      elif (self.dataType.isSingle() or self.dataType.isDouble() or self.dataType.isInt32()):
        kStr += (padSpacesStr + "value = abs(value);\n")
      else:
        raise RuntimeError("Unrecognized data type %s."%self.dataType)
    elif (activationType == 'clippedrelu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = (value >= alpha) ? min(value, beta) : 0.0;\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = (value >= alpha) ? min(value, beta) : 0;\n")
    elif (activationType == 'exp'):
      kStr += (asm + " // Exp\n")
      module = activation.getExpModule(self.dataType, 0)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    elif (activationType == 'gelu'):
      kStr += (asm + " // gelu\n")
      module = activation.getGeluModule(self.dataType, 0)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    elif (activationType == 'leakyrelu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = (value >= 0.0) ? value : (value * alpha);\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = (value >= 0) ? value : (value * alpha);\n")
      else:
        raise RuntimeError("Unsupported data type %s."%ptrStr)
    elif (activationType == 'relu'):
      if (self.dataType.isSingle() or self.dataType.isHalf() or self.dataType.isDouble()):
        kStr += (padSpacesStr + "value = max(0.0, value);\n")
      elif self.dataType.isInt32():
        kStr += (padSpacesStr + "value = max(0, value);\n")
      else:
        raise RuntimeError("Unsupported data type %s."%ptrStr)
    elif (activationType == 'sigmoid'):
      kStr += (asm + " // Sigmoid\n")
      module = activation.getSigmoidModule(self.dataType, 0)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    elif (activationType == 'tanh'):
      kStr += (asm + " // tanh\n")
      module = activation.getTanhModule(self.dataType, 0, 1, 2)
      kStr += self.getActivationAsmStr(activation, module, (len(asm) * " "))
      kStr += addSpace(asm, ": \"+v\"(value) : \"s\"(alpha), \"s\"(beta)\n")
      kStr += self.getRequiredRegStr(asm, activation.vgprCounter, activation.sgprCounter)
    else:
      if (activationType != 'none'):
        raise RuntimeError("Unrecognized type %s."%activationType)
    return kStr
  # Internal use. Automatically gets the required vgprs and sgprs for inline assembly
  def getRequiredRegStr(self, spaceAlignStr, numOfVgpr, numOfSgpr):
    requiredReg = []
    for i in range(0, numOfVgpr):
      requiredReg.append("\"v%d\""%i)
    for i in range(0, numOfSgpr):
      requiredReg.append("\"s%d\""%i)
    requiredStr = ""
    if (len(requiredReg) > 0):
      requiredStr = requiredReg[0]
      for i in range(1, len(requiredReg)):
        requiredStr += ", %s"%requiredReg[i]
    kStr = ""
    kStr += addSpace(spaceAlignStr,":%s);\n"%requiredStr)
    return kStr
