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

from .Code import Macro, Module
from .Containers import VCC
from .Instructions import SAndB32, SAndB64, VAddCOU32, VAddU32, \
                        VCmpGEU32, VCmpLeU32, VCmpNeI32, VCndMaskB32, \
                        VCvtF32toU32, VCvtU32toF32, VMulF32, \
                        VMulHIU32, VMulLOU32, VRcpF32, VLShiftRightB32, \
                        VLShiftRightB64, VSubCoU32

# Performs a division using 'magic number' computed on host
# Argument requirements:
#   - dstIdx must be two consecutive registers ; on exit the lower one will contain the quotient.  The upper is used as a temp.
#   - First parm is passed as an integer vgpr index ; remaining are vgpr or sgpr symbolic names
#   - dstIdx+1 cannot be same as dividend.  dividend+0 can be same as dividend and this may be useful for chaining divides.
def MacroVMagicDiv(magicDivAlg) -> Module:
    module = Module("defineMagicDivMacros")
    module.addComment1("Magic div and mod functions")
    macro = Macro("V_MAGIC_DIV", "dstIdx:req", "dividend:req", "magicNumber:req", "magicShift:req", "magicA:req")
    if magicDivAlg==1: # TODO: remove me
        macro.add(VMulHIU32(dst="v[\\dstIdx+1]", src0="\\dividend", src1="\\magicNumber"))
        macro.add(VMulLOU32(dst="v[\\dstIdx+0]", src0="\\dividend", src1="\\magicNumber"))
        macro.add(VLShiftRightB64(dst="v[\\dstIdx:\\dstIdx+1]", shiftHex="\\magicShift", src="v[\\dstIdx:\\dstIdx+1]"))
    elif magicDivAlg==2:
        macro.add(VMulHIU32(dst="v[\\dstIdx+1]", src0="\\dividend", src1="\\magicNumber"))
        macro.add(VMulLOU32(dst="v[\\dstIdx+0]", src0="\\dividend", src1="\\magicA"))
        macro.add(VAddU32(dst="v[\\dstIdx+0]", src0="v[\\dstIdx+0]", src1="v[\\dstIdx+1]"))
        macro.add(VLShiftRightB32(dst="v[\\dstIdx+0]", shiftHex="\\magicShift", src="v[\\dstIdx+0]"))
    module.add(macro)
    return module

def MacroVDynamicScalarDiv(wavefrontSize) -> Module:
    module = Module("Dynamic scalar divide macros")
    module.addComment1("Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor;")
    macro = Macro("DYNAMIC_VECTOR_DIVIDE", "vQuotient", "vRemainder", "vDividend", "vDivisor", "vTmp0", "vTmp1", "sTmp")
    sTmpStr = "s[\\sTmp]" if (wavefrontSize == 32) else "s[\\sTmp:\\sTmp+1]"
    macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))
    macro.add(VRcpF32(dst="v[\\vQuotient]", src="v[\\vQuotient]"))
    macro.add(VMulF32(dst="v[\\vQuotient]", src0=hex(0x4f800000), src1="v[\\vQuotient]"))
    macro.add(VCvtF32toU32(dst="v[\\vQuotient]", src="v[\\vQuotient]"))
    macro.add(VMulLOU32(dst="v[\\vRemainder]", src0="v[\\vDivisor]", src1="v[\\vQuotient]"))
    macro.add(VMulHIU32(dst="v[\\vTmp0]", src0="v[\\vDivisor]", src1="v[\\vQuotient]"))
    macro.add(VSubCoU32(dst="v[\\vTmp1]", dst1=VCC(), src0=hex(0), src1="v[\\vRemainder]"))
    macro.add(VCmpNeI32(dst=sTmpStr, src0=hex(0), src1="v[\\vTmp0]"))
    macro.add(VCndMaskB32(dst="v[\\vRemainder]", src0="v[\\vTmp1]", src1="v[\\vRemainder]", src2=sTmpStr)) # type: ignore
    macro.add(VMulHIU32(dst="v[\\vRemainder]", src0="v[\\vRemainder]", src1="v[\\vQuotient]"))
    macro.add(VSubCoU32(dst="v[\\vTmp0]", dst1=VCC(), src0="v[\\vQuotient]", src1="v[\\vRemainder]"))
    macro.add(VAddCOU32(dst="v[\\vQuotient]", dst1=VCC(), src0="v[\\vQuotient]", src1="v[\\vRemainder]"))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0="v[\\vQuotient]", src1="v[\\vTmp0]", src2=sTmpStr)) # type: ignore
    macro.add(VMulHIU32(dst="v[\\vQuotient]", src0="v[\\vQuotient]", src1="v[\\vDividend]"))
    macro.add(VMulLOU32(dst="v[\\vRemainder]", src0="v[\\vQuotient]", src1="v[\\vDivisor]"))
    macro.add(VSubCoU32(dst="v[\\vTmp0]", dst1=VCC(), src0="v[\\vDividend]", src1="v[\\vRemainder]"))
    macro.add(VCmpGEU32(dst=sTmpStr, src0="v[\\vDividend]", src1="v[\\vRemainder]"))
    macro.add(VAddCOU32(dst="v[\\vRemainder]", dst1=VCC(), src0=hex(1), src1="v[\\vQuotient]"))
    macro.add(VAddCOU32(dst="v[\\vTmp1]", dst1=VCC(), src0=-1, src1="v[\\vQuotient]"))
    macro.add(VCmpLeU32(dst=VCC(), src0="v[\\vDivisor]", src1="v[\\vTmp0]"))
    SAndBX = SAndB32 if wavefrontSize == 32 else SAndB64
    macro.add(SAndBX(dst=VCC(), src0=sTmpStr, src1=VCC()))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0="v[\\vQuotient]", src1="v[\\vRemainder]", src2=VCC()))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0="v[\\vTmp1]",     src1="v[\\vQuotient]", src2=sTmpStr)) # type: ignore
    macro.add(VCmpNeI32(dst=VCC(), src0=hex(0), src1="v[\\vDivisor]"))
    macro.add(VCndMaskB32(dst="v[\\vQuotient]", src0=-1, src1="v[\\vQuotient]", src2=VCC(), comment="final result" ))
    macro.add(VMulLOU32(dst="v[\\vRemainder]", src0="v[\\vQuotient]", src1="v[\\vDivisor]"))
    macro.add(VSubCoU32(dst="v[\\vRemainder]", dst1=VCC(), src0="v[\\vDividend]", src1="v[\\vRemainder]", comment="final result" ))
    module.add(macro)
    return module
