################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.enum import SelectBit, UnusedBit
from rocisa.container import RegName, RegisterContainer, HolderContainer

from .Base import Item
from . import getGlcBitName, getSlcBitName

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
import math

class Container(Item):
    def __init__(self) -> None:
        super().__init__("Container")

@dataclass
class DSModifiers(Container):
    # Nums of addresses needed
    na: int = 1
    # Used with DS instructions that expect a single address
    offset: int = 0
    # Used with DS instructions that expect two addresses
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self):
        super().__init__()

    def toList(self) -> List[str]:
        l = [str(self).lstrip()]
        return l

    def __str__(self) -> str:
        kStr = ""
        if self.na == 1:
            kStr += " offset:%u"%self.offset
        elif self.na == 2:
            kStr += " offset0:%u offset1:%u"%(self.offset0, self.offset1)
        if self.gds:
            kStr += " gds"
        return kStr

@dataclass
class FLATModifiers(Container):
    offset12: int  = 0
    glc:      bool = False
    slc:      bool = False
    lds:      bool = False
    isStore:  bool = False

    def __post_init__(self):
        super().__init__()

    def toList(self) -> List[str]:
        l = [str(self).lstrip()]
        return l

    def __str__(self) -> str:
        hasGLCModifier = self.asmCaps["HasGLCModifier"]
        kStr = ""
        if self.offset12 != 0:
            kStr += " offset:%u"%self.offset12
        if self.glc:
            kStr += " " + getGlcBitName(hasGLCModifier)
        if self.slc:
            kStr += " " + getSlcBitName(hasGLCModifier)
        if self.lds:
            kStr += " lds"
        return kStr

@dataclass
class MUBUFModifiers(Container):
    offen:    bool = False
    offset12: int  = 0
    glc:      bool = False
    slc:      bool = False
    nt:       bool = False
    lds:      bool = False
    isStore:  bool = False

    def __post_init__(self):
        super().__init__()

    def toList(self) -> List[str]:
        l = [str(self).lstrip()]
        return l

    def __str__(self) -> str:
        hasGLCModifier = self.asmCaps["HasGLCModifier"]
        hasNTModifier = self.asmCaps["HasNTModifier"]
        kStr = ""
        if self.offen:
            kStr += " offen offset:%u"%self.offset12
        if (self.glc or self.slc or self.lds):
            kStr += ","
        if self.glc:
            kStr += " " + getGlcBitName(hasGLCModifier)
        if self.slc:
            kStr += " " + getSlcBitName(hasGLCModifier)
        if hasNTModifier and self.nt:
            kStr += " nt"
        if self.lds:
            kStr += " lds"
        return kStr

@dataclass
class SMEMModifiers(Container):
    glc:      bool = False
    nv:       bool = False
    offset: int    = 0 # 20u 21s shaes the same

    def __post_init__(self):
        super().__init__()

    def toList(self) -> List[str]:
        l = [str(self).lstrip()]
        return l

    def __str__(self) -> str:
        kStr = ""
        if self.offset != 0:
            kStr += " offset:%d"%self.offset
        if self.glc:
            kStr += " glc"
        if self.nv:
            kStr += " nv"
        return kStr

@dataclass
class SDWAModifiers(Container):
    dst_sel:    Optional[SelectBit] = None
    dst_unused: Optional[UnusedBit] = None
    src0_sel:   Optional[SelectBit] = None
    src1_sel:   Optional[SelectBit] = None

    def __post_init__(self):
        super().__init__()

    def toList(self) -> List[str]:
        l = []
        if self.dst_sel != None:
            l.append("dst_sel:" + self.dst_sel.name)
        if self.dst_unused != None:
            l.append("dst_unused:" + self.dst_unused.name)
        if self.src0_sel != None:
            l.append("src0_sel:" + self.src0_sel.name)
        if self.src1_sel != None:
            l.append("src1_sel:" + self.src1_sel.name)
        return l

    def __str__(self) -> str:
        kStr = ""
        if self.dst_sel != None:
            kStr += " dst_sel:" + self.dst_sel.name
        if self.dst_unused != None:
            kStr += " dst_unused:" + self.dst_unused.name
        if self.src0_sel != None:
            kStr += " src0_sel:" + self.src0_sel.name
        if self.src1_sel != None:
            kStr += " src1_sel:" + self.src1_sel.name
        return kStr

# dot2: for WaveSplitK reduction. Only a subset of DPP modifiers are used here
@dataclass
class DPPModifiers(Container):
    row_shr:    Optional[int] = None
    row_bcast:  Optional[int] = None
    bound_ctrl: Optional[int] = None

    def __post_init__(self):
        super().__init__()

    def toList(self) -> List[str]:
        l = []
        if self.row_shr != None:
            l.append("row_shr:" + str(self.row_shr))
        if self.row_bcast != None:
            l.append("row_bcast:" + str(self.row_bcast))
        if self.bound_ctrl != None:
            l.append("bound_ctrl:" + str(self.bound_ctrl))
        return l

    def __str__(self) -> str:
        kStr = ""
        if self.row_shr != None:
            kStr += " row_shr:" + str(self.row_shr)
        if self.row_bcast != None:
            kStr += " row_bcast:" + str(self.row_bcast)
        if self.bound_ctrl != None:
            kStr += " bound_ctrl:" + str(self.bound_ctrl)
        return kStr

@dataclass
class VOP3PModifiers(Container):
    op_sel:     Optional[List[int]] = None
    op_sel_hi:  Optional[List[int]] = None
    byte_sel:   Optional[List[int]] = None

    def __post_init__(self):
        super().__init__()

    def toList(self) -> List[str]:
        l = []
        if self.op_sel != None:
            l.append("op_sel:" + str(self.op_sel).replace(" ", ""))
        if self.op_sel_hi != None:
            l.append("op_sel_hi:" + str(self.op_sel_hi).replace(" ", ""))
        if self.byte_sel != None:
            l.append("byte_sel:" + str(self.byte_sel).replace(" ", ""))
        return l

    def __str__(self) -> str:
        kStr = ""
        if self.op_sel != None:
            kStr += " op_sel:" + str(self.op_sel).replace(" ", "")
        if self.op_sel_hi != None:
            kStr += " op_sel_hi:" + str(self.op_sel_hi).replace(" ", "")
        if self.byte_sel != None:
            kStr += " byte_sel:" + str(self.byte_sel).replace(" ", "")
        return kStr

class EXEC(Container):
    def __init__(self, setHi=False) -> None:
        super().__init__()
        self.setHi = setHi

    def __str__(self) -> str:
        if self.kernel.wavefrontSize == 64:
            return "exec"
        else:
            return "exec_lo"

class VCC(Container):
    def __init__(self, setHi=False) -> None:
        super().__init__()
        self.setHi = setHi

    def __str__(self) -> str:
        if self.kernel.wavefrontSize == 64:
            return "vcc"
        else:
            return "vcc_hi" if self.setHi else "vcc_lo"

class HWRegContainer(Container):
    def __init__(self, reg: str, value: List[int]) -> None:
        super().__init__()
        self.reg = reg
        self.value = value

    def __str__(self) -> str:
        s = "hwreg("
        s += self.reg
        for v in self.value:
            s += ("," + str(v))
        s += ")"
        return s
