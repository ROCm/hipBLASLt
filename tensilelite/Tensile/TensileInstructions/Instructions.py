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

from .Base import Item
from .Enums import InstType, CvtType
from .Containers import DSModifiers, FLATModifiers, MUBUFModifiers, SMEMModifiers, SDWAModifiers, VOP3PModifiers, VCC, \
                        RegisterContainer, HolderContainer
from .Formatting import formatStr, printExit
import abc
from enum import Enum
from typing import List, Optional, Union

################################################################################
################################################################################
###
###   Instruction classes
###
################################################################################
################################################################################

class Instruction(Item, abc.ABC):
    def __init__(self, instType: InstType, comment="") -> None:
        Item.__init__(self, "instruction")
        self.instType = instType
        self.comment  = comment

        # Inst string
        self.instStr  = ""

        # Settings
        self.outputInlineAsm = False

    def setInlineAsm(self, isTrue: bool) -> None:
        self.outputInlineAsm = isTrue

    def formatOnly(self, instStr, comment) -> str:
        return formatStr(self.outputInlineAsm, instStr, comment)

    def formatWithComment(self, instStr) -> str:
        return formatStr(self.outputInlineAsm, instStr, self.comment)

    def formatWithExtraComment(self, instStr, comment) -> str:
        comment = self.comment + comment
        return formatStr(self.outputInlineAsm, instStr, comment)

    def setInst(self, instStr: str):
        self.instStr = instStr

    def preStr(self):
        # Overwrite this if needed
        pass

    @abc.abstractmethod
    def getParams(self) -> list:
        pass

    @abc.abstractmethod
    def toList(self) -> list:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

class CompositeInstruction(Instruction):
    def __init__(self, instType: InstType, dst, srcs: Optional[List], comment=""):
        super().__init__(instType, comment)
        self.instructions: List = []
        self.dst = dst
        self.srcs = srcs

    def getInstructions(self) -> List[Instruction]:
        self.setupInstructions()
        return self.instructions

    def toList(self) -> List[List]:
        self.setupInstructions()
        return [inst.toList() for inst in self.instructions]

    @abc.abstractmethod
    def setupInstructions(self):
        pass

    def getParams(self) -> list:
        assert 0
        return []

    def preStr(self):
        self.setupInstructions()

    def __str__(self):
        self.preStr()
        return '\n'.join([str(s) for s in self.instructions])

class CommonInstruction(Instruction):
    def __init__(self, instType: InstType, dst, src: list, \
                 sdwa: Optional[SDWAModifiers]=None, vop3: Optional[VOP3PModifiers]=None, \
                 comment="") -> None:
        super().__init__(instType, comment)
        self.dst      = dst
        self.dst1     = None # Usually we don't need this
        self.src      = src
        self.sdwa     = sdwa
        self.vop3     = vop3

    def getArgStr(self) -> str:
        kStr = ""
        if self.dst:
            kStr += str(self.dst)
        if self.dst1:
            if kStr:
                kStr += ", "
            kStr += str(self.dst1)
        if self.src:
            if kStr:
                kStr += ", "
            kStr += str(self.src[0])
        for i in self.src[1:]:
            kStr += ", " + str(i)
        return kStr

    def getParams(self) -> list:
        l = []
        if self.dst:
            l.append(self.dst)
        if self.dst1:
            l.append(self.dst1)
        if self.src:
            l.extend(self.src)
        return l

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr]
        if self.dst:
            l.append(self.dst)
        if self.dst1:
            l.append(self.dst1)
        if self.src:
            l.extend(self.src)
        l.extend(self.sdwa.toList()) if self.sdwa else ""
        l.extend(self.vop3.toList()) if self.vop3 else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.sdwa) if self.sdwa else ""
        kStr += str(self.vop3) if self.vop3 else ""
        return self.formatWithComment(kStr)

class BranchInstruction(Instruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(InstType.INST_NOTYPE, comment)
        self.labelName = labelName

    def getParams(self) -> list:
        return [self.labelName]

    def toList(self) -> list:
        return [self.instStr, self.labelName, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(self.instStr + " " + str(self.labelName))

class VCmpInstruction(CommonInstruction):
    def __init__(self, instType: InstType, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(instType, dst, [src0, src1], sdwa, None, comment)

class VCmpXInstruction(CommonInstruction):
    def __init__(self, instType: InstType, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(instType, dst, [src0, src1], sdwa, None, comment)

    def toList(self) -> list:
        self.preStr()
        if self.archCaps["CMPXWritesSGPR"]:
            l = [self.instStr, self.dst]
            if self.src:
                l.extend(self.src)
            l.extend(self.sdwa.toList()) if self.sdwa else ""
            l.append(self.comment)
        else:
            instStr = self.instStr.replace("_cmpx_", "_cmp_")
            l1 = [instStr, self.dst]
            if self.src:
                l1.extend(self.src)
            l1.extend(self.sdwa.toList()) if self.sdwa else ""
            l1.append(self.comment)
            if self.kernel.wavefrontSize == 64:
                l2 = ["s_mov_b64", "exec", self.dst, ""]
            else:
                l2 = ["s_mov_b32", "exec_lo", self.dst, ""]
            l = [l1, l2]
        return l

    def __str__(self) -> str:
        self.preStr()
        if self.archCaps["CMPXWritesSGPR"]:
            kStr = self.instStr + " " + self.getArgStr()
            kStr += str(self.sdwa) if self.sdwa else ""
            kStr += str(self.vop3) if self.vop3 else ""
            kStr = self.formatWithComment(kStr)
        else:
            kStr = self.instStr.replace("_cmpx_", "_cmp_")
            kStr += " " + self.getArgStr()
            kStr += str(self.sdwa) if self.sdwa else ""
            kStr = self.formatWithComment(kStr)
            if self.kernel.wavefrontSize == 64:
                kStr2 = "s_mov_b64 exec " + str(self.dst)
            else:
                kStr2 = "s_mov_b32 exec_lo " + str(self.dst)
            kStr2 = self.formatWithComment(kStr2)
            kStr += kStr2
        return kStr

class VCvtInstruction(CommonInstruction):
    def __init__(self, cvtType: CvtType, dst, src, sdwa: Optional[SDWAModifiers] = None, \
                 comment="") -> None:
        super().__init__(InstType.INST_CVT, dst, [src], sdwa, None, comment)
        self.cvtType = cvtType

class MFMAInstruction(Instruction):
    def __init__(self, instType: InstType, accType: InstType, variant: List[int], mfma1k, \
                 acc, a, b, acc2=None, comment="") -> None:
        super().__init__(instType, comment)
        self.accType = accType
        self.variant = variant
        self.mfma1k  = mfma1k
        self.acc     = acc
        self.a       = a
        self.b       = b
        self.acc2    = acc if acc2 == None else acc2

    def typeConvert(self, iType) -> str:
        if iType == InstType.INST_F16:
            kStr = "f16"
        elif iType == InstType.INST_F32:
            kStr = "f32"
        elif iType == InstType.INST_BF16:
            kStr = "bf16"
        elif iType == InstType.INST_I8:
            kStr = "i8"
        elif iType == InstType.INST_I32:
            kStr = "i32"
        else:
            printExit("Type %s not found"%str(iType))
        return kStr

    def getParams(self) -> list:
        return [self.acc, self.a, self.b, self.acc2]

    def preStr(self) -> None:
        if len(self.variant) == 3:
            variantStr = "{}x{}x{}".format(*self.variant)
            mfma_1k = "_1k" if self.mfma1k else ""
            self.setInst("v_mfma_%s_%s%s%s"%(self.typeConvert(self.accType), variantStr, \
                         self.typeConvert(self.instType), mfma_1k))
        else:
            assert("Currently does not support mfma variant != 3" and 0)

    def getArgStr(self) -> str:
        return str(self.acc) + ", " + str(self.a) + ", " + str(self.b) + ", " + str(self.acc2)

    def toList(self) -> list:
        self.preStr()
        return [self.instStr, self.acc, self.a, self.b, self.acc2, self.comment]

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        return self.formatWithComment(kStr)

class MacroInstruction(Instruction):
    def __init__(self, name: str, args: list, comment="") -> None:
        super().__init__(InstType.INST_MACRO, comment)
        self.name = name
        self.args = args

    def getParams(self) -> list:
        return self.args

    def getArgStr(self) -> str:
        kStr = ""
        if self.args:
            kStr += " " + str(self.args[0])
            for arg in self.args[1:]:
                kStr += ", " + str(arg)
        return kStr

    def toList(self) -> list:
        l = [self.name]
        l.extend(self.args)
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        return self.formatWithComment(self.name + self.getArgStr())

class ReadWriteInstruction(Instruction):
    class RWType(Enum):
        RW_TYPE0 = 1
        RW_TYPE1 = 2

    def __init__(self, instType: InstType, rwType: RWType, comment="") -> None:
        super().__init__(instType, comment)
        self.rwType = rwType

    def typeConvert(self) -> str:
        kStr = ""
        if self.rwType == ReadWriteInstruction.RWType.RW_TYPE0:
            if self.instType == InstType.INST_U16:
                kStr = "ushort" if self.kernel.isa[0] < 11 else "u16"
            elif self.instType == InstType.INST_B8:
                kStr = "byte" if self.kernel.isa[0] < 11 else "b8"
            elif self.instType == InstType.INST_B16:
                kStr = "short" if self.kernel.isa[0] < 11 else "b16"
            elif self.instType == InstType.INST_B32:
                kStr = "dword" if self.kernel.isa[0] < 11 else "b32"
            elif self.instType == InstType.INST_B64:
                kStr = "dwordx2" if self.kernel.isa[0] < 11 else "b64"
            elif self.instType == InstType.INST_B128:
                kStr = "dwordx4" if self.kernel.isa[0] < 11 else "b128"
            elif self.instType == InstType.INST_B256:
                kStr = "dwordx8" if self.kernel.isa[0] < 11 else "b256"
            elif self.instType == InstType.INST_B512:
                kStr = "dwordx16" if self.kernel.isa[0] < 11 else "b512"
            elif self.instType == InstType.INST_D16_U8:
                kStr = "ubyte_d16" if self.kernel.isa[0] < 11 else "d16_u8"
            elif self.instType == InstType.INST_D16_HI_U8:
                kStr = "ubyte_d16_hi" if self.kernel.isa[0] < 11 else "d16_hi_u8"
            elif self.instType == InstType.INST_D16_HI_B8:
                kStr = "byte_d16_hi" if self.kernel.isa[0] < 11 else "d16_hi_b8"
            elif self.instType == InstType.INST_D16_B16:
                kStr = "short_d16" if self.kernel.isa[0] < 11 else "d16_b16"
            elif self.instType == InstType.INST_D16_HI_B16:
                kStr = "short_d16_hi" if self.kernel.isa[0] < 11 else "d16_hi_b16"
        return kStr

    def preStr(self):
        # Local read is set in DSLoad and DSStore
        self.instStr += self.typeConvert()

    @abc.abstractmethod
    def toList(self) -> list:
        pass
    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @staticmethod
    def issueLatency():
        # In Quad-Cycle
        return 1

class GlobalReadInstruction(ReadWriteInstruction):
    def __init__(self, instType: InstType, dst, comment="") -> None:
        super().__init__(instType, ReadWriteInstruction.RWType.RW_TYPE0, comment)
        self.dst = dst

class FLATReadInstruction(GlobalReadInstruction):
    def __init__(self, instType: InstType, dst, vaddr, \
                 flat: Optional[FLATModifiers] = None, \
                 comment="") -> None:
        super().__init__(instType, dst, comment)
        self.instStr = "flat_load_"
        self.vaddr = vaddr
        self.flat   = flat

    def getParams(self) -> list:
        return [self.dst, self.vaddr]

    def getArgStr(self) -> str:
        return str(self.dst) + ", " + str(self.vaddr)

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr, self.dst, self.vaddr]
        l.extend(self.flat.toList()) if self.flat else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.flat) if self.flat else ""
        return self.formatWithComment(kStr)

class MUBUFReadInstruction(GlobalReadInstruction):
    def __init__(self, instType: InstType, dst, vaddr, saddr, soffset, \
                 mubuf: Optional[MUBUFModifiers] = None, \
                 comment="") -> None:
        super().__init__(instType, dst, comment)
        self.instStr = "buffer_load_"
        self.vaddr = vaddr
        self.saddr   = saddr
        self.soffset = soffset
        self.mubuf   = mubuf

    def getParams(self) -> list:
        return [self.dst, self.vaddr, self.saddr, self.soffset]

    def getArgStr(self) -> str:
        return str(self.dst) + ", " + str(self.vaddr) + ", " + str(self.saddr) + ", " + str(self.soffset)

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr, self.dst, self.vaddr, self.saddr, self.soffset]
        l.extend(self.mubuf.toList()) if self.mubuf else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.mubuf) if self.mubuf else ""
        return self.formatWithComment(kStr)

class SMemLoadInstruction(GlobalReadInstruction):
    def __init__(self, instType: InstType, dst, base, soffset,
                 smem: Optional[SMEMModifiers]=None, comment="") -> None:
        super().__init__(instType, dst, comment)
        self.instStr = "s_load_"
        self.base    = base
        self.soffset = soffset
        self.smem    = smem

    def getParams(self) -> list:
        return [self.dst, self.base, self.soffset]

    def getArgStr(self) -> str:
        return str(self.dst) + ", " + str(self.base) + ", " + str(self.soffset)

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr, self.dst, self.base, self.soffset]
        l.extend(self.smem.toList()) if self.smem else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.smem) if self.smem else ""
        return self.formatWithComment(kStr)

class GlobalWriteInstruction(ReadWriteInstruction):
    def __init__(self, instType: InstType, srcData, comment="") -> None:
        super().__init__(instType, ReadWriteInstruction.RWType.RW_TYPE0, comment)
        self.srcData = srcData

class FLATStoreInstruction(GlobalWriteInstruction):
    def __init__(self, instType: InstType, vaddr, srcData, \
                 flat: Optional[FLATModifiers] = None, \
                 comment="") -> None:
        super().__init__(instType, srcData, comment)
        self.instStr = "flat_store_"
        self.vaddr = vaddr
        self.flat   = flat

    def getParams(self) -> list:
        return [self.vaddr, self.srcData]

    def getArgStr(self) -> str:
        return str(self.vaddr) + ", " + str(self.srcData)

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr, self.vaddr, self.srcData]
        l.extend(self.flat.toList()) if self.flat else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.flat) if self.flat else ""
        return self.formatWithComment(kStr)

class MUBUFStoreInstruction(GlobalWriteInstruction):
    def __init__(self, instType: InstType, srcData, vaddr, saddr, soffset, \
                 mubuf: Optional[MUBUFModifiers] = None, \
                 comment="") -> None:
        super().__init__(instType, srcData, comment)
        self.instStr = "buffer_store_"
        self.vaddr = vaddr
        self.saddr   = saddr
        self.soffset = soffset
        self.mubuf   = mubuf

    def getParams(self) -> list:
        return [self.srcData, self.vaddr, self.saddr, self.soffset]

    def getArgStr(self) -> str:
        return str(self.srcData) + ", " + str(self.vaddr) + ", " + str(self.saddr) + ", " + str(self.soffset)

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr, self.srcData, self.vaddr, self.saddr, self.soffset]
        l.extend(self.mubuf.toList()) if self.mubuf else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.mubuf) if self.mubuf else ""
        return self.formatWithComment(kStr)

class LocalReadInstruction(ReadWriteInstruction):
    def __init__(self, instType: InstType, dst, src, \
                 readToTempVgpr: bool, comment="") -> None:
        super().__init__(instType, ReadWriteInstruction.RWType.RW_TYPE1, comment)
        self.dst            = dst
        self.src            = src

        self.readToTempVgpr = readToTempVgpr

class DSLoadInstruction(LocalReadInstruction):
    def __init__(self, instType: InstType, dst, src, \
                 readToTempVgpr: bool, \
                 ds: Optional[DSModifiers] = None, \
                 comment="") -> None:
        super().__init__(instType, dst, src, readToTempVgpr, comment)
        self.ds             = ds

    def getParams(self) -> list:
        return [self.dst, self.src]

    def preStr(self):
        if self.kernel.isa[0] < 11:
            self.instStr = self.instStr.replace("load", "read")

    def getArgStr(self) -> str:
        return str(self.dst) + ", " + str(self.src)

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr, self.dst, self.src]
        l.extend(self.ds.toList()) if self.ds else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.ds) if self.ds else ""
        return self.formatWithComment(kStr)

class LocalWriteInstruction(ReadWriteInstruction):
    def __init__(self, instType: InstType, dstAddr, src0, src1, \
                 comment="") -> None:
        super().__init__(instType, ReadWriteInstruction.RWType.RW_TYPE1, comment)
        self.dstAddr      = dstAddr
        self.src0         = src0
        self.src1         = src1

class DSStoreInstruction(LocalWriteInstruction):
    def __init__(self, instType: InstType, dstAddr, src0, src1, \
                 ds: Optional[DSModifiers] = None, \
                 comment="") -> None:
        super().__init__(instType, dstAddr, src0, src1, comment)
        self.ds             = ds

    def getParams(self) -> list:
        return [self.dstAddr, self.src0, self.src1]

    def preStr(self):
        if self.kernel.isa[0] < 11:
            self.instStr = self.instStr.replace("store", "write")

    def getArgStr(self) -> str:
        kStr = str(self.dstAddr) + ", " + str(self.src0)
        if self.src1:
            kStr += ", " + str(self.src1)
        return kStr

    def toList(self) -> list:
        self.preStr()
        l = [self.instStr, self.dstAddr, self.src0]
        l.extend(self.src1) if self.src1 else ""
        l.extend(self.ds.toList()) if self.ds else ""
        l.append(self.comment)
        return l

    def __str__(self) -> str:
        self.preStr()
        kStr = self.instStr + " " + self.getArgStr()
        kStr += str(self.ds) if self.ds else ""
        return self.formatWithComment(kStr)

################################################################################
################################################################################
###
###   Instructions
###
################################################################################
################################################################################

################################################################################
###   Read/ Write instructions
################################################################################

## Buffer load
class BufferLoadD16HIU8(MUBUFReadInstruction):
    def __init__(self, dst, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_U8, dst, vaddr, saddr, soffset, mubuf, comment)

class BufferLoadD16U8(MUBUFReadInstruction):
    def __init__(self, dst, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_U8, dst, vaddr, saddr, soffset, mubuf, comment)

class BufferLoadD16HIB16(MUBUFReadInstruction):
    def __init__(self, dst, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_B16, dst, vaddr, saddr, soffset, mubuf, comment)

class BufferLoadD16B16(MUBUFReadInstruction):
    def __init__(self, dst, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_B16, dst, vaddr, saddr, soffset, mubuf, comment)

class BufferLoadB32(MUBUFReadInstruction):
    def __init__(self, dst, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, vaddr, saddr, soffset, mubuf, comment)

class BufferLoadB64(MUBUFReadInstruction):
    def __init__(self, dst, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, vaddr, saddr, soffset, mubuf, comment)

class BufferLoadB128(MUBUFReadInstruction):
    def __init__(self, dst, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B128, dst, vaddr, saddr, soffset, mubuf, comment)

## Flat load
class FlatLoadD16HIB16(FLATReadInstruction):
    def __init__(self, dst, vaddr, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_B16, dst, vaddr, flat, comment)

class FlatLoadD16B16(FLATReadInstruction):
    def __init__(self, dst, vaddr, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_B16, dst, vaddr, flat, comment)

class FlatLoadB32(FLATReadInstruction):
    def __init__(self, dst, vaddr, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, vaddr, flat, comment)

class FlatLoadB64(FLATReadInstruction):
    def __init__(self, dst, vaddr, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, vaddr, flat, comment)

class FlatLoadB128(FLATReadInstruction):
    def __init__(self, dst, vaddr, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B128, dst, vaddr, flat, comment)

## Buffer store
class BufferStoreB8(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B8, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreD16HIU8(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_U8, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreD16U8(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_U8, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreD16HIB16(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_B16, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreD16B16(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_B16, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreB16(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B16, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreB32(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreB64(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, src, vaddr, saddr, soffset, mubuf, comment)

class BufferStoreB128(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B128, src, vaddr, saddr, soffset, mubuf, comment)

class BufferAtomicAddF32(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, src, vaddr, saddr, soffset, mubuf, comment)
        self.setInst("buffer_atomic_add_f32")

    def typeConvert(self) -> str:
        return ""

class BufferAtomicCmpswapB32(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, src, vaddr, saddr, soffset, mubuf, comment)
        self.setInst("buffer_atomic_cmpswap")

    def typeConvert(self) -> str:
        return ""

class BufferAtomicCmpswapB64(MUBUFStoreInstruction):
    def __init__(self, src, vaddr, saddr, soffset, mubuf: Optional[MUBUFModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, src, vaddr, saddr, soffset, mubuf, comment)
        self.setInst("buffer_atomic_cmpswap_x2")

    def typeConvert(self) -> str:
        return ""

## Flat store
class FlatStoreD16HIB16(FLATStoreInstruction):
    def __init__(self, vaddr, src, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_B16, vaddr, src, flat, comment)

class FlatStoreD16B16(FLATStoreInstruction):
    def __init__(self, vaddr, src, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_B16, vaddr, src, flat, comment)

class FlatStoreB32(FLATStoreInstruction):
    def __init__(self, vaddr, src, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, vaddr, src, flat, comment)

class FlatStoreB64(FLATStoreInstruction):
    def __init__(self, vaddr, src, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, vaddr, src, flat, comment)

class FlatStoreB128(FLATStoreInstruction):
    def __init__(self, vaddr, src, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B128, vaddr, src, flat, comment)

class FlatAtomicCmpswapB32(FLATStoreInstruction):
    def __init__(self, vaddr, tmp, src, flat: Optional[FLATModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, vaddr, src, flat, comment)
        self.tmp = tmp
        self.setInst("flat_atomic_cmpswap")

    def getArgStr(self) -> str:
        return ", ".join(map(str, [self.vaddr, self.tmp, self.srcData]))

    def toList(self) -> List:
        self.preStr()
        l = [self.instStr, self.vaddr, self.tmp, self.srcData]
        if self.flat: l.extend(self.flat.toList())
        l.append(self.comment)
        return l

    def typeConvert(self) -> str:
        return ""

    def __str__(self) -> str:
        self.preStr()
        kStr = " ".join([self.instStr, self.getArgStr()])
        kStr += str(self.flat) if self.flat else ""
        return self.formatWithComment(kStr)

# DS Load
class DSLoadU8(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U8, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_load_u8")

class DSLoadD16HIU8(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_U8, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_load_u8_d16_hi")

class DSLoadU16(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U16, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_load_u16")

class DSLoadD16HIU16(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_U16, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_load_u16_d16_hi")

class DSLoadB32(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_load_b32")

class DSLoadB64(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_load_b64")

class DSLoadB128(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B128, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_load_b128")

    @staticmethod
    def issueLatency():
        return 2

class DSLoad2B32(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 2
        self.setInst("ds_load2_b32")

class DSLoad2B64(DSLoadInstruction):
    def __init__(self, dst, src, readToTempVgpr: bool, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, src, readToTempVgpr, ds, comment)
        if ds: ds.na = 2
        self.setInst("ds_load2_b64")

# DS store
class DSStoreU16(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U16, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_u16")

    @staticmethod
    def issueLatency():
        return 2

class DSStoreB8(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B8, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_b8")

class DSStoreB16(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B16, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_b16")

class DSStoreB8HID16(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B8_HI_D16, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_b8_d16_hi")

class DSStoreD16HIB16(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_D16_HI_B16, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_b16_d16_hi")

class DSStoreB32(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_b32")

    @staticmethod
    def issueLatency():
        return 2

class DSStoreB64(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_b64")

    @staticmethod
    def issueLatency():
        return 3

class DSStoreB128(DSStoreInstruction):
    def __init__(self, dstAddr, src, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B128, dstAddr, src, None, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_store_b128")

    @staticmethod
    def issueLatency():
        return 5

class DSStore2B32(DSStoreInstruction):
    def __init__(self, dstAddr, src0, src1, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, dstAddr, src0, src1, ds, comment)
        if ds: ds.na = 2
        self.setInst("ds_store2_b32")

    @staticmethod
    def issueLatency():
        return 3

class DSStore2B64(DSStoreInstruction):
    def __init__(self, dstAddr, src0, src1, ds: Optional[DSModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B64, dstAddr, src0, src1, ds, comment)
        if ds: ds.na = 2
        self.setInst("ds_store2_b64")

    @staticmethod
    def issueLatency():
        return 3

# DS instructions
class DSBPermuteB32(DSStoreInstruction):
    def __init__(self, dst, src0, src1, ds: Optional[DSModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, src0, src1, ds, comment)
        if ds: ds.na = 1
        self.setInst("ds_bpermute_b32")

################################################################################
###   SGPR instructions
################################################################################

# Abs
class SAbsI32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src], None, None, comment)
        self.setInst("s_abs_i32")

# Arithmetic
class SAddI32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], None, None, comment)
        self.setInst("s_add_i32")
class SAddU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)
        self.setInst("s_add_u32")

class SAddCU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)
        self.setInst("s_addc_u32")

class SMulI32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], None, None, comment)
        self.setInst("s_mul_i32")

class SMulHII32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_HI_I32, dst, [src0, src1], None, None, comment)
        self.setInst("s_mul_hi_i32")

class SMulHIU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_HI_U32, dst, [src0, src1], None, None, comment)
        self.setInst("s_mul_hi_u32")

class SSubI32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], None, None, comment)
        self.setInst("s_sub_i32")

class SSubU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)
        self.setInst("s_sub_u32")

class SSubBU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)
        self.setInst("s_subb_u32")

# Cmp
class SCmpEQI32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_eq_i32")

class SCmpEQU32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_eq_u32")

class SCmpEQU64(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U64, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_eq_u64")

class SCmpGeI32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_ge_i32")

class SCmpGeU32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_ge_u32")

class SCmpLeI32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_le_i32")

class SCmpLeU32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_le_u32")

# SCC = (S0 != S1)
class SCmpLgU32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_lg_u32")

class SCmpLtI32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_lt_i32")

class SCmpLtU32(CommonInstruction):
    def __init__(self, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src0, src1], None, None, comment)
        self.setInst("s_cmp_lt_u32")

# S Cmp K
# SCC = (S0.u == SIMM16)
class SCmpKEQU32(CommonInstruction):
    def __init__(self, src, simm16: Union[int, str], comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src, simm16], None, None, comment)
        self.setInst("s_cmpk_eq_u32")
class SCmpKGtU32(CommonInstruction):
    def __init__(self, src, simm16: str, comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src, simm16], None, None, comment)
        self.setInst("s_cmpk_gt_u32")

class SCmpKLGU32(CommonInstruction):
    def __init__(self, src, simm16: Union[int, str], comment="") -> None:
        super().__init__(InstType.INST_U32, None, [src, simm16], None, None, comment)
        self.setInst("s_cmpk_lg_u32")

# S Select
# D.u = SCC ? S0.u : S1.u
class SCSelectB32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_cselect_b32")

# Logic
class SAndB32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_and_b32")

class SAndB64(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src0, src1], None, None, comment)
        self.setInst("s_and_b64")

class SOrB32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_or_b32")

class SXorB32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_xor_b32")
class SOrB64(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src0, src1], None, None, comment)
        self.setInst("s_or_b64")

# Branch
class SBranch(BranchInstruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(labelName, comment)
        self.setInst("s_branch")
class SCBranchSCC0(BranchInstruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(labelName, comment)
        self.setInst("s_cbranch_scc0")

class SCBranchSCC1(BranchInstruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(labelName, comment)
        self.setInst("s_cbranch_scc1")

class SCBranchVCCNZ(BranchInstruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(labelName, comment)
        self.setInst("s_cbranch_vccnz")

class SCBranchVCCZ(BranchInstruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(labelName, comment)
        self.setInst("s_cbranch_vccz")

# S PC
class SGetPCB64(CommonInstruction):
    def __init__(self, dst, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [], None, None, comment)
        self.setInst("s_getpc_b64")

class SSetPCB64(BranchInstruction):
    def __init__(self, src, comment="") -> None:
        super().__init__("", comment)
        self.src = src
        self.setInst("s_setpc_b64")

    def toList(self) -> list:
        return [self.instStr, self.src, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(self.instStr + " " + str(self.src))

class SSwapPCB64(BranchInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__("", comment)
        self.dst = dst
        self.src = src
        self.setInst("s_swappc_b64")

    def toList(self) -> list:
        return [self.instStr, self.dst, self.src, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(self.instStr + " " + str(self.dst) + ", " + str(self.src))

class SCBranchExecZ(BranchInstruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(labelName, comment)
        self.setInst("s_cbranch_execz")

class SCBranchExecNZ(BranchInstruction):
    def __init__(self, labelName: str, comment="") -> None:
        super().__init__(labelName, comment)
        self.setInst("s_cbranch_execnz")

# S Shift
class SLShiftLeftB32(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src, shiftHex], None, None, comment)
        self.setInst("s_lshl_b32")

class SLShiftRightB32(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src, shiftHex], None, None, comment)
        self.setInst("s_lshr_b32")

class SLShiftLeftB64(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src, shiftHex], None, None, comment)
        self.setInst("s_lshl_b64")

class SLShiftRightB64(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src, shiftHex], None, None, comment)
        self.setInst("s_lshr_b64")

# Arithmetic shift right (preserve sign bit)
class SAShiftRightI32(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src, shiftHex], None, None, comment)
        self.setInst("s_ashr_i32")

class SLShiftLeft1AddU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_lshl1_add_u32")

class SLShiftLeft2AddU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_lshl2_add_u32")

class SLShiftLeft3AddU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_lshl3_add_u32")

class SLShiftLeft4AddU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("s_lshl4_add_u32")

# S mov
class SMovB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("s_mov_b32")

class SMovB64(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src], None, None, comment)
        self.setInst("s_mov_b64")

class SCMovB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("s_cmov_b32")

class SCMovB64(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src], None, None, comment)
        self.setInst("s_cmov_b64")

# Sign ext
class SMovkI32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src], None, None, comment)
        self.setInst("s_movk_i32")

# S exec
class SAndSaveExecB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("s_and_saveexec_b32")

class SAndSaveExecB64(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src], None, None, comment)
        self.setInst("s_and_saveexec_b64")

class SOrSaveExecB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("s_or_saveexec_b32")

class SOrSaveExecB64(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [src], None, None, comment)
        self.setInst("s_or_saveexec_b64")

class SSetPrior(Instruction):
    def __init__(self, prior, comment="") -> None:
        super().__init__(InstType.INST_NOTYPE, comment)
        self.prior = prior
        self.setInst("s_setprio")

    def getParams(self) -> list:
        return [self.prior]

    def toList(self) -> list:
        return [self.instStr, self.prior, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(self.instStr + " " + str(self.prior))

class SBarrier(Instruction):
    def __init__(self, comment="") -> None:
        super().__init__(InstType.INST_NOTYPE, comment)
        self.setInst("s_barrier")

    def getParams(self) -> list:
        return []

    def toList(self) -> list:
        return [self.instStr, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(self.instStr)
class SNop(Instruction):
    def __init__(self, waitState: int, comment="") -> None:
        super().__init__(InstType.INST_NOTYPE, comment)
        self.waitState = waitState
        self.setInst("s_nop")

    def getParams(self) -> list:
        return [self.waitState]

    def toList(self) -> list:
        return [self.instStr, self.waitState, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(self.instStr + " " + str(self.waitState))

class SEndpgm(Instruction):
    def __init__(self, comment="") -> None:
        super().__init__(InstType.INST_NOTYPE, comment)
        self.setInst("s_endpgm")

    def getParams(self) -> list:
        return []

    def toList(self) -> list:
        return [self.instStr, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(self.instStr)
class SSleep(Instruction):
    def __init__(self, simm16, comment=""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.simm16 = simm16
        self.setInst("s_sleep")

    def getParams(self) -> list:
        return [self.simm16]

    def toList(self) -> List:
        return [self.instStr, self.simm16, self.comment]

    def __str__(self) -> str:
        return self.formatWithComment(f"{self.instStr} {self.simm16}")

# S WaitCnt
class _SWaitCnt(Instruction):
    def __init__(self, lgkmcnt: int=-1, vmcnt: int=-1, comment=""):
        super().__init__(InstType.INST_NOTYPE, comment)
        self.lgkmcnt = lgkmcnt
        self.vmcnt   = vmcnt

    def getParams(self) -> list:
        return [self.lgkmcnt, self.vmcnt]

    def toList(self) -> list:
        assert 0 and "Not supported."
        return []

    def __str__(self) -> str:
        if self.lgkmcnt == 0 and self.vmcnt == 0:
            waitStr = "0"
        else:
            waitStr = ""
            if self.lgkmcnt != -1:
                maxLgkmcnt = self.asmCaps["MaxLgkmcnt"]
                waitStr = "lgkmcnt(%u)" % (min(self.lgkmcnt,maxLgkmcnt))
            if self.vmcnt != -1:
                waitStr += (", " if waitStr != "" else "") + "vmcnt(%u)"%self.vmcnt
        return self.formatWithComment("s_waitcnt %s"%(waitStr))

class _SWaitCntVscnt(Instruction):
    def __init__(self, vscnt: int=-1, comment="") -> None:
        super().__init__(InstType.INST_NOTYPE, comment)
        self.vscnt = vscnt

    def getParams(self) -> list:
        return [self.vscnt]

    def toList(self) -> list:
        assert 0 and "Not supported."
        return []

    def __str__(self) -> str:
        return self.formatWithComment("s_waitcnt_vscnt %u"%(self.vscnt))

class SWaitCnt(CompositeInstruction):
    """
    Construct a waitcnt from specified lgkmcnt and vmcnt:
    lgkmcnt, vmcnt:
      if -1 then will not be added to the wait term.

    If lgkmcnt=vmcnt=vscnt=-1 then the waitcnt is a nop and
    an instruction with a comment is returned.
    """
    def __init__(self, lgkmcnt: int=-1, vmcnt: int=-1, vscnt: int=-1, comment="", waitAll=False):
        super().__init__(InstType.INST_NOTYPE, None, None, comment=comment)
        self.lgkmcnt = lgkmcnt
        self.vmcnt   = vmcnt
        self.vscnt   = vscnt
        self.waitAll = waitAll

    def setupInstructions(self):
        super().setupInstructions()
        if self.waitAll:
            lgkmcnt = 0
            vmcnt   = 0
            vscnt   = 0
            comment = "(Wait all)"
        else:
            lgkmcnt = self.lgkmcnt
            vmcnt   = self.vmcnt
            vscnt   = self.vscnt
            comment = self.comment

        maxVmcnt = self.asmCaps["MaxVmcnt"]
        if self.archCaps["SeparateVscnt"]:
            vmcnt = min(vmcnt, maxVmcnt)
            self.instructions = [_SWaitCnt(lgkmcnt, vmcnt, comment),
                                 _SWaitCntVscnt(vscnt, comment)]
        else:
            vmvscnt = -1
            if vscnt != -1:
                vmvscnt = vscnt
            if vmcnt != -1:
                vmvscnt = vmcnt + (vmvscnt if vmvscnt != -1 else 0)
            vmvscnt = min(vmvscnt, maxVmcnt)
            self.instructions = [_SWaitCnt(lgkmcnt, vmvscnt, comment)]

# S Load
class SLoadB32(SMemLoadInstruction):
    def __init__(self, dst, base, soffset, smem: Optional[SMEMModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, base, soffset, smem, comment)

class SLoadB64(SMemLoadInstruction):
    def __init__(self, dst, base, soffset, smem: Optional[SMEMModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, base, soffset, smem, comment)

class SLoadB128(SMemLoadInstruction):
    def __init__(self, dst, base, soffset, smem: Optional[SMEMModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B128, dst, base, soffset, smem, comment)

class SLoadB256(SMemLoadInstruction):
    def __init__(self, dst, base, soffset, smem: Optional[SMEMModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B256, dst, base, soffset, smem, comment)

class SLoadB512(SMemLoadInstruction):
    def __init__(self, dst, base, soffset, smem: Optional[SMEMModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B512, dst, base, soffset, smem, comment)

################################################################################
###   VGPR instructions
################################################################################

# Arithmetic
class VAddF16(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_add_f16")

class VAddF32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_add_f32")

class VAddF64(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F64, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_add_f64")

class VAddI32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_add_i32")

    def preStr(self):
        if self.asmBugs["ExplicitNC"]:
            self.setInst("v_add_nc_i32")
        elif self.asmBugs["ExplicitCO"]:
            self.setInst("v_add_i32")
        else:
            self.setInst("v_add_i32")

class VAddU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)

    def preStr(self):
        if self.asmBugs["ExplicitNC"]:
            self.setInst("v_add_nc_u32")
            self.dst1 = None
        elif self.asmBugs["ExplicitCO"]:
            self.setInst("v_add_u32")
            self.dst1 = None
        else:
            self.setInst("v_add_u32")
            self.dst1 = VCC()

class VAddCOU32(CommonInstruction):
    def __init__(self, dst, dst1, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)
        self.dst1 = dst1

    def preStr(self):
        if self.asmBugs["ExplicitCO"]:
            self.setInst("v_add_co_u32")
        else:
            self.setInst("v_add_u32")

class VAddCCOU32(CommonInstruction):
    def __init__(self, dst, dst1, src0, src1, src2, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1, src2], None, None, comment)
        self.dst1 = dst1
        self.setInst("_v_addc_co_u32")

    def preStr(self):
        if self.asmBugs["ExplicitNC"]:
            self.setInst("v_add_co_ci_u32")
        elif self.asmBugs["ExplicitCO"]:
            self.setInst("v_addc_co_u32")
        else:
            self.setInst("v_addc_u32")

class VAddPKF16(CommonInstruction):
    def __init__(self, dst, src0, src1, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1], None, vop3, comment)
        self.setInst("v_pk_add_f16")

class _VAddPKF32(CommonInstruction):
    def __init__(self, dst, src0, src1, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], None, vop3, comment)
        self.setInst("v_pk_add_f32")

class VAddPKF32(CompositeInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], comment)
        self.setInst("v_pk_add_f32")

    def getParams(self) -> list:
        assert 0 and "Not supported."
        return []

    def toList(self) -> list:
        assert 0 and "Not supported."
        return []

    def setupInstructions(self):
        super().setupInstructions()
        assert isinstance(self.srcs, List)
        if self.asmCaps["v_pk_add_f32"]:
            self.instructions = [_VAddPKF32(self.dst, self.srcs[0], self.srcs[1], None, self.comment)]
        else:
            dst1, dst2 = self.dst.splitRegContainer()
            srcs1 = []
            srcs2 = []
            for s in self.srcs:
                if isinstance(s, RegisterContainer) or isinstance(s, HolderContainer):
                    r1, r2 = s.splitRegContainer()
                    srcs1.append(r1)
                    srcs2.append(r2)
                else:
                    srcs1.append(s)
                    srcs2.append(s)
            self.instructions = [VAddF32(dst1, srcs1[0], srcs1[1], None, self.comment),
                                VAddF32(dst2, srcs2[0], srcs2[1], None, self.comment)]

        assert all(inst.vop3 is None for inst in self.instructions), "Currently does not support with vop3 enabled"

class VAdd3U32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_add3_u32")

class VMulF16(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_mul_f16")

class VMulF32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_mul_f32")

class VMulF64(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F64, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_mul_f64")

class VMulPKF16(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1], sdwa, vop3, comment)
        self.setInst("v_pk_mul_f16")

class VMulPKF32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], sdwa, vop3, comment)
        self.setInst("v_pk_mul_f32")

class VMulLOU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_LO_U32, dst, [src0, src1], None, None, comment)
        self.setInst("v_mul_lo_u32")

class VMulHII32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_HI_I32, dst, [src0, src1], None, None, comment)
        self.setInst("v_mul_hi_i32")

class VMulHIU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_HI_U32, dst, [src0, src1], None, None, comment)
        self.setInst("v_mul_hi_u32")

class VMulI32I24(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], None, None, comment)
        self.setInst("v_mul_i32_i24")

class VMulU32U24(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)
        self.setInst("v_mul_u32_u24")

class VSubI32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], None, None, comment)

    def preStr(self):
        if self.asmBugs["ExplicitNC"]:
            self.setInst("v_sub_nc_i32")
        elif self.asmBugs["ExplicitCO"]:
            self.setInst("v_sub_i32")
        else:
            self.setInst("v_sub_i32")

class VSubU32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)

    def preStr(self):
        if self.asmBugs["ExplicitNC"]:
            self.setInst("v_sub_nc_u32")
        elif self.asmBugs["ExplicitCO"]:
            self.setInst("v_sub_u32")
        else:
            self.setInst("v_sub_u32")

class VSubCoU32(CommonInstruction):
    def __init__(self, dst, dst1, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1], None, None, comment)
        self.dst1 = dst1

    def preStr(self):
        if self.asmBugs["ExplicitCO"]:
            self.setInst("v_sub_co_u32")
        else:
            self.setInst("v_sub_u32")

# MAC
class VMacF32(CommonInstruction):
    def __init__(self, dst, src0, src1, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], None, vop3, comment)
        self.setInst("v_mac_f32")
        self.addDstToSrc = False

    def preStr(self):
        if self.asmCaps["v_fmac_f32"]:
            self.setInst("v_fmac_f32")
        elif self.asmCaps["v_fma_f32"]:
            self.addDstToSrc = True
            self.setInst("v_fmac_f32")
        elif self.asmCaps["v_mac_f32"]:
            self.setInst("v_mac_f32")
        else:
            raise RuntimeError("FMA and MAC instructions are not supported.")

    def getArgStr(self) -> str:
        kStr = super().getArgStr()
        if self.addDstToSrc:
            kStr += ", " + str(self.dst)
        return kStr

# Dot
class VDot2CF32F16(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_dot2c_f32_f16")

    def preStr(self):
        if self.kernel.isa[0] >= 11:
            self.setInst("v_dot2acc_f32_f16")

# Fma
class VFmaF16(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_fma_f16")

class VFmaF32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_fma_f32")

class VFmaF64(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F64, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_fma_f64")

class VFmaPKF16(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_pk_fma_f16")

class VFmaMixF32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_fma_mix_f32")

# V Mad
class VMadI32I24(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_mad_i32_i24")

class VMadU32U24(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_mad_u32_u24")

class VMadMixF32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1, src2], None, vop3, comment)
        self.setInst("v_mad_mix_f32")

# Exp, rcp
class VExpF16(CommonInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src], sdwa, None, comment)
        self.setInst("v_exp_f16")

class VExpF32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src], None, None, comment)
        self.setInst("v_exp_f32")

class VRcpF16(CommonInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src], sdwa, None, comment)
        self.setInst("v_rcp_f16")

class VRcpF32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src], None, None, comment)
        self.setInst("v_rcp_f32")

# Cmp
class VCmpEQF32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_eq_f32")

class VCmpEQF64(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F64, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_eq_f64")

class VCmpEQU32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_eq_u32")

class VCmpGEF16(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ge_f16")

class VCmpGEF32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ge_f32")

class VCmpGEF64(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F64, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ge_f64")

class VCmpGEI32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ge_i32")

class VCmpGEU32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ge_u32")

class VCmpGtU32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_gt_u32")

class VCmpLeU32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_le_u32")

class VCmpLtI32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_lt_i32")

class VCmpLtU32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_lt_u32")

# D.u64[threadId] = (isNan(S0) || isNan(S1))
class VCmpUF32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_u_f32")

class VCmpNeI32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ne_i32")

class VCmpNeU32(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ne_u32")

class VCmpNeU64(VCmpInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U64, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmp_ne_u64")

# CmpX
class VCmpXEqU32(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_eq_u32")

class VCmpXGeU32(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_ge_u32")

class VCmpXGtU32(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_gt_u32")

class VCmpXLeU32(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_le_u32")

class VCmpXLtI32(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_lt_i32")

class VCmpXLtU32(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_lt_u32")

class VCmpXLtU64(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U64, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_lt_u64")

class VCmpXNeU16(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U16, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_ne_u16")

class VCmpXNeU32(VCmpXInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, src0, src1, sdwa, comment)
        self.setInst("v_cmpx_ne_u32")

# Min Max
class VMaxF16(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_max_f16")

class VMaxF32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_max_f32")

class VMaxF64(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F64, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_max_f64")

class VMaxI32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_max_i32")

class VMaxPKF16(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1], sdwa, vop3, comment)
        self.setInst("v_pk_max_f16")

class VMed3I32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1, src2], None, None, comment)
        self.setInst("v_med3_i32")

class VMinF16(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F16, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_min_f16")

class VMinF32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_min_f32")

class VMinF64(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_F64, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_min_f64")

class VMinI32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_min_i32")

# V Logic
class VAndB32(CommonInstruction):
    def __init__(self, dst, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, None, comment)
        self.setInst("v_and_b32")

class VAndOrB32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1, src2], None, None, comment)
        self.setInst("v_and_or_b32")

class VNotB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("v_not_b32")

class VOrB32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_or_b32")

class VXorB32(CommonInstruction):
    def __init__(self, dst, src0, src1, sdwa: Optional[SDWAModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], sdwa, None, comment)
        self.setInst("v_xor_b32")

# V Convert
class VCvtF16toF32(VCvtInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(CvtType.CVT_F16_to_F32, dst, src, sdwa, comment)
        self.setInst("v_cvt_f32_f16")

class VCvtF32toF16(VCvtInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(CvtType.CVT_F32_to_F16, dst, src, sdwa, comment)
        self.setInst("v_cvt_f16_f32")

class VCvtF32toU32(VCvtInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(CvtType.CVT_F32_to_U32, dst, src, sdwa, comment)
        self.setInst("v_cvt_u32_f32")

class VCvtU32toF32(VCvtInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(CvtType.CVT_U32_to_F32, dst, src, sdwa, comment)
        self.setInst("v_cvt_f32_u32")

class VCvtI32toF32(VCvtInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(CvtType.CVT_I32_to_F32, dst, src, sdwa, comment)
        self.setInst("v_cvt_f32_i32")

class VCvtF32toI32(VCvtInstruction):
    def __init__(self, dst, src, sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(CvtType.CVT_F32_to_I32, dst, src, sdwa, comment)
        self.setInst("v_cvt_i32_f32")

# V Mask
class VCndMaskB32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2 = VCC(), sdwa: Optional[SDWAModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1, src2], sdwa, None, comment)
        self.setInst("v_cndmask_b32")

# V Shift
class VLShiftLeftB16(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B16, dst, [shiftHex, src], None, None, comment)
        self.setInst("v_lshlrev_b16")

class VLShiftLeftB32(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [shiftHex, src], None, None, comment)
        self.setInst("v_lshlrev_b32")

class VLShiftRightB32(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [shiftHex, src], None, None, comment)
        self.setInst("v_lshrrev_b32")

class VLShiftLeftB64(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [shiftHex, src], None, None, comment)
        self.setInst("v_lshlrev_b64")

class VLShiftRightB64(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment="") -> None:
        super().__init__(InstType.INST_B64, dst, [shiftHex, src], None, None, comment)
        self.setInst("v_lshrrev_b64")

class _VLShiftLeftOrB32(CommonInstruction):
    def __init__(self, dst, shiftHex, src0, src1, comment):
        super().__init__(InstType.INST_B32, dst, [src0, shiftHex, src1], comment=comment)
        self.setInst("v_lshl_or_b32")

# V Arith Shift
# D.i = signext(S1.i) >> S0.i[4:0]
class VAShiftRightI32(CommonInstruction):
    def __init__(self, dst, shiftHex, src, comment):
        super().__init__(InstType.INST_I32, dst, [shiftHex, src], comment=comment)
        self.setInst("v_ashrrev_i32")

# V Shift + Logic
class VLShiftLeftOrB32(CompositeInstruction):
    def __init__(self, dst, shiftHex, src0, src1, comment: str=""):
        super().__init__(InstType.INST_B32, dst, [shiftHex, src0, src1], comment)

    @property
    def shift(self):
        assert isinstance(self.srcs, List)
        return self.srcs[0]

    def setupInstructions(self):
        super().setupInstructions()
        assert isinstance(self.srcs, List)
        if self.asmCaps["HasLshlOr"]:
            self.instructions = [_VLShiftLeftOrB32(self.dst, self.shift, self.srcs[1], self.srcs[2], self.comment)]
        else:
            self.instructions = [VLShiftLeftB32(self.dst, self.shift, self.srcs[1], self.comment), \
                                 VOrB32(self.dst, self.dst, self.srcs[2])]

        assert all(inst.vop3 is None for inst in self.instructions), "Currently does not support with vop3 enabled"

# V Add + Shift
class _VAddLShiftLeftU32(CommonInstruction):
    def __init__(self, dst, shiftHex, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1, shiftHex], None, None, comment)
        self.setInst("v_add_lshl_u32")

class VAddLShiftLeftU32(CompositeInstruction):
    def __init__(self, dst, shiftHex, src0, src1, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1, shiftHex], comment)
        self.setInst("v_add_lshl_u32")

    @property
    def shift(self):
        assert isinstance(self.srcs, List)
        return self.srcs[2]

    def setupInstructions(self):
        super().setupInstructions()
        assert isinstance(self.srcs, List)
        if self.asmCaps["HasAddLshl"]:
            self.instructions = [_VAddLShiftLeftU32(self.dst, self.shift, self.srcs[0], self.srcs[1], self.comment)]
        else:
            if self.asmBugs["ExplicitCO"]:
                vadd = VAddCCOU32(self.dst, VCC(), self.srcs[0], self.srcs[1], self.comment)
            else:
                vadd = VAddU32(self.dst, self.srcs[0], self.srcs[1], self.comment)
            self.instructions = [vadd, VLShiftLeftB32(self.dst, self.shift, self.dst, self.comment)]

        assert all(inst.vop3 is None for inst in self.instructions), "Currently does not support with vop3 enabled"

class _VLShiftLeftAddU32(CommonInstruction):
    def __init__(self, dst, shiftHex, src0, src1, vop3: Optional[VOP3PModifiers] = None, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1, shiftHex], None, vop3, comment)
        self.setInst("v_lshl_add_u32")

class VLShiftLeftAddU32(CompositeInstruction):
    def __init__(self, dst, shiftHex, src0, src1, comment: str=""):
        super().__init__(InstType.INST_U32, dst, [src0, src1, shiftHex], comment)

    @property
    def shift(self):
        assert isinstance(self.srcs, List)
        return self.srcs[2]

    def setupInstructions(self):
        super().setupInstructions()
        assert isinstance(self.srcs, List)
        if self.asmCaps["HasAddLshl"]:
            self.instructions = [_VLShiftLeftAddU32(self.dst, self.shift, self.srcs[0], self.srcs[1], comment=self.comment)]
        else:
            if self.asmBugs["ExplicitCO"]:
                inst = VAddCOU32(self.dst, VCC(), self.srcs[0], self.srcs[1])
            else:
                inst = VAddU32(self.dst, self.srcs[0], self.srcs[1])
            self.instructions = [VLShiftLeftB32(self.dst, self.shift, self.dst, self.comment), inst]

        assert all(inst.vop3 is None for inst in self.instructions), "Currently does not support with vop3 enabled"

# V Mov
class VMovB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("v_mov_b32")

# V Bfe
class VBfeI32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, comment="") -> None:
        super().__init__(InstType.INST_I32, dst, [src0, src1, src2], None, None, comment)
        self.setInst("v_bfe_i32")

# D.u = (S0.u >> S1.u[4:0]) & ((1 << S2.u[4:0]) - 1)
class VBfeU32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, comment="") -> None:
        super().__init__(InstType.INST_U32, dst, [src0, src1, src2], None, None, comment)
        self.setInst("v_bfe_u32")

# V Bfi
# D.u = (S0.u & S1.u) | (~S0.u & S2.u)
class VBfiB32(CommonInstruction):
    def __init__(self, dst, src0, src1, src2, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1, src2], None, None, comment)
        self.setInst("v_bfi_b32")

# V Pack
class VPackF16toB32(CommonInstruction):
    def __init__(self, dst, src0, src1, vop3: Optional[VOP3PModifiers]=None, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src0, src1], None, vop3, comment)
        self.setInst("v_pack_b32_f16")

# Read/ Write
class VAccvgprReadB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("v_accvgpr_read_b32")

class VAccvgprWrite(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_NOTYPE, dst, [src], None, None, comment)
        self.setInst("v_accvgpr_write")

class VAccvgprWriteB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("v_accvgpr_write_b32")

class VReadfirstlaneB32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_B32, dst, [src], None, None, comment)
        self.setInst("v_readfirstlane_b32")

# Rounding
class VRndneF32(CommonInstruction):
    def __init__(self, dst, src, comment="") -> None:
        super().__init__(InstType.INST_F32, dst, [src], None, None, comment)
        self.setInst("v_rndne_f32")
