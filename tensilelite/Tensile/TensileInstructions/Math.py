################################################################################
# Copyright 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from typing import Optional
from .Code import Module
from .Containers import RegisterContainer
from .ExtInstructions import SMulInt64to32
from .Instructions import *
from .RegisterPool import RegisterPoolResource
from .Utils import vgpr, sgpr, log2

########################################
# Divide & Remainder
# quotient register, remainder register, dividend register, divisor, tmpVgprx2
########################################

def vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgprRes: Optional[RegisterPoolResource], doRemainder=True, comment=""):
    dComment = "%s = %s / %s"    % (vgpr(qReg), vgpr(dReg), divisor) if (comment=="") else comment
    rComment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor) if (comment=="") else comment

    module = Module("vectorStaticDivideAndRemainder")
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        divisor_log2 = log2(divisor)
        module.add(VLShiftRightB32(dst=vgpr(qReg), shiftHex=divisor_log2, src=vgpr(dReg), comment=dComment))
        if doRemainder:
            module.add(VAndB32(dst=vgpr(rReg), src0=(divisor-1), src1=vgpr(dReg), comment=rComment))
    else:
        assert tmpVgprRes and tmpVgprRes.size >= 2
        tmpVgpr = tmpVgprRes.idx
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        shift = 32+1
        magic = ((2**shift) // divisor) + 1
        if magic <= 64 and magic >= -16:
            module.add(VMulHIU32(dst=vgpr(tmpVgpr+1), src0=vgpr(dReg), src1=hex(magic), comment=dComment))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+0), src0=vgpr(dReg), src1=hex(magic), comment=dComment))
        else:
            module.add(VMovB32(dst=vgpr(tmpVgpr+0), src=hex(magic)))
            module.add(VMulHIU32(dst=vgpr(tmpVgpr+1), src0=vgpr(dReg), src1=vgpr(tmpVgpr+0), comment=dComment))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+0), src0=vgpr(dReg), src1=vgpr(tmpVgpr+0), comment=dComment))
        module.add(VLShiftRightB64(dst=vgpr(tmpVgpr,2), shiftHex=hex(shift), src=vgpr(tmpVgpr,2), comment=dComment))
        module.add(VMovB32(dst=vgpr(qReg), src=vgpr(tmpVgpr), comment=dComment))
        if doRemainder:
            module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=vgpr(qReg), src1=hex(divisor), comment=rComment))
            module.add(VSubU32(dst=vgpr(rReg), src0=vgpr(dReg), src1=vgpr(tmpVgpr), comment=rComment))
    return module

def vectorStaticDivide(qReg, dReg, divisor, tmpVgprRes: Optional[RegisterPoolResource], comment=""):
    rReg = -1 # unused
    module = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgprRes, False, comment)
    module.name = "vectorStaticDivide (reg=-1)"
    return module

def vectorStaticRemainder(qReg, rReg, dReg, divisor, tmpVgprRes: Optional[RegisterPoolResource], \
                        tmpSgprRes: Optional[RegisterPoolResource], comment=""):
    if comment == "":
        comment = "%s = %s %% %s" % (vgpr(rReg), vgpr(dReg), divisor)

    module = Module("vectorStaticRemainder")
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        module.add(VAndB32(dst=vgpr(rReg), src0=(divisor-1), src1=vgpr(dReg), comment=comment))
    else:
        assert tmpVgprRes and tmpVgprRes.size >= 2
        tmpVgpr = tmpVgprRes.idx
        assert tmpSgprRes and tmpSgprRes.size >= 1
        tmpSgpr = tmpSgprRes.idx
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 7:
            shift = 32+3
        elif divisor >= 6:
            shift = 32+2 # this was 32+3 but divisor hex didn't fit into 32 bits
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        shift = 32+1
        magic = ((2**shift) // divisor) + 1
        if magic <= 64 and magic >= -16:
            module.add(VMulHIU32(dst=vgpr(tmpVgpr+1), src0=vgpr(dReg), src1=hex(magic), comment=comment))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+0), src0=vgpr(dReg), src1=hex(magic), comment=comment))
        else:
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(magic), comment=comment))
            module.add(VMulHIU32(dst=vgpr(tmpVgpr+1), src0=vgpr(dReg), src1=sgpr(tmpSgpr), comment=comment))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr+0), src0=vgpr(dReg), src1=sgpr(tmpSgpr), comment=comment))
        module.add(VLShiftRightB64(dst=vgpr(tmpVgpr,2), shiftHex=hex(shift), src=vgpr(tmpVgpr,2), comment=comment))
        module.add(VMovB32(dst=vgpr(qReg), src=vgpr(tmpVgpr), comment=comment))
        if divisor <= 64 and divisor >= -16:
            module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=vgpr(qReg), src1=hex(divisor), comment=comment))
        else:
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(divisor), comment=comment))
            module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=vgpr(qReg), src1=sgpr(tmpSgpr), comment=comment))
        module.add(VSubU32(dst=vgpr(rReg), src0=vgpr(dReg), src1=vgpr(tmpVgpr), comment=comment))
    return module

# only used for loop unroll and GlobalSplitU
# doRemainder==0 : compute quotient only
# doRemainder==1 : compute quotient and remainder
# doRemainder==2 : only compute remainder (not quotient unless required for remainder)
# dreg == dividend
# tmpSgpr must be 2 SPGRs
# qReg and dReg can be "sgpr[..]" or names of sgpr (will call sgpr)
def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgprRes: Optional[RegisterPoolResource], \
        doRemainder=1):

    qRegSgpr = qReg if isinstance(qReg, RegisterContainer) and qReg.regType == 's' else sgpr(qReg)

    dRegSgpr = dReg if isinstance(dReg, RegisterContainer) and dReg.regType == 's' else sgpr(dReg)

    module = Module("scalarStaticDivideAndRemainder")
    if ((divisor & (divisor - 1)) == 0): # pow of 2
        divisor_log2 = log2(divisor)
        if doRemainder != 2:
            module.add(SLShiftRightB32(dst=qRegSgpr, shiftHex=divisor_log2, src=dRegSgpr, \
                    comment="%s = %s / %u"%(qRegSgpr, dRegSgpr, divisor)))
        if doRemainder:
            module.add(SAndB32(dst=sgpr(rReg), src0=(divisor-1), src1=dRegSgpr, \
                    comment="%s = %s %% %u"%(sgpr(rReg), dRegSgpr, divisor)))
    else:
        assert tmpSgprRes and tmpSgprRes.size >= 2
        tmpSgpr = tmpSgprRes.idx
        assert qReg != tmpSgpr
        """
        if divisor == 30:
            shift = 32+2
        elif divisor >= 14:
            shift = 32+4
        elif divisor >= 6:
            shift = 32+3
        elif divisor >= 5:
            shift = 32+2
        elif divisor >= 3:
            shift = 32+1
        """
        shift = 32+1
        magic = ((2**shift) // divisor) + 1
        magicHi = magic // (2**16)
        magicLo = magic & (2**16-1)

        module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(0), comment="STATIC_DIV: divisior=%s"%divisor))
        module.add(SMulI32(dst=sgpr(tmpSgpr+0), src0=hex(magicHi), src1=dRegSgpr, comment="tmp1 = dividend * magic hi"))
        module.add(SLShiftLeftB64(dst=sgpr(tmpSgpr,2), shiftHex=hex(16), src=sgpr(tmpSgpr,2), comment="left shift 16 bits"))
        module.add(SMulI32(dst=qRegSgpr, src0=dRegSgpr, src1=hex(magicLo), comment="tmp0 = dividend * magic lo"))
        module.add(SAddU32(dst=sgpr(tmpSgpr+0), src0=qRegSgpr, src1=sgpr(tmpSgpr+0), comment="add lo"))
        module.add(SAddCU32(dst=sgpr(tmpSgpr+1), src0=sgpr(tmpSgpr+1), src1=hex(0), comment="add hi"))
        module.add(SLShiftRightB64(dst=sgpr(tmpSgpr,2), shiftHex=hex(shift), src=sgpr(tmpSgpr,2), comment="tmp1 = (dividend * magic) << shift"))
        module.add(SMovB32(dst=qRegSgpr, src=sgpr(tmpSgpr), comment="quotient"))
        if doRemainder:
            module.add(SMulI32(dst=sgpr(tmpSgpr), src0=qRegSgpr, src1=hex(divisor), comment="quotient*divisor"))
            module.add(SSubU32(dst=sgpr(rReg), src0=dRegSgpr, src1=sgpr(tmpSgpr), comment="rReg = dividend - quotient*divisor"))
    return module

########################################
# Scalar Magic Div
# product register, operand register, multiplier
########################################

# dividend is a symbol (constant or sgpr).  Used directly not inside automatic sgpr(..)
# dst is 2 consecutive SGPR
#   result returned in dst0. dst1 is used as a temp,
# dst[1] cannot be same as divident, dst[0] can be same as dividend and this can be useful
def scalarMagicDivExplicit(dst, dividend, magicNumber, magicAbit, magicShift):
    module = Module("scalarMagicDivExplicit")
    module.addComment1("dst1:0 = dividend(%s) / magicTag(%s)" % (dividend, magicNumber))
    module.add(SMulHIU32(dst=sgpr(dst+1), src0=dividend, src1=sgpr(magicNumber), comment="scalar magic div (magicnum)"))
    module.add(SMulI32(dst=sgpr(dst+0), src0=dividend, src1=sgpr(magicAbit), comment="scalar magic div (abit)"))
    module.add(SAddU32(dst=sgpr(dst+0), src0=sgpr(dst+0), src1=sgpr(dst+1), comment="scalar magic div (combine)"))
    module.add(SLShiftRightB32(dst=sgpr(dst+0), shiftHex=sgpr(magicShift), src=sgpr(dst+0), \
                   comment="scalar magic div (shift), quotient in s%s"%dst))
    return module

def scalarMagicDiv(dst, dividend, magicTag):
    return scalarMagicDivExplicit(dst, dividend,
                                  magicNumber="MagicNumberSize"+magicTag,
                                  magicAbit="MagicAbitSize"+magicTag,
                                  magicShift="MagicShiftSize"+magicTag)

##############################################################################
# Perform a magic division (mul by magic number and shift)
# dest is two consec SGPR, used for intermediate temp as well as final result
# result quotient returned in sgpr(dest,1)
# tmpVgpr: Size 2
##############################################################################
def sMagicDiv(dest, hasSMulHi, dividend, magicNumber, magicShift, tmpVgpr):
    module = Module("sMagicDiv")
    module.addModuleAsFlatItems(SMulInt64to32(hasSMulHi, \
                                sgpr(dest), sgpr(dest+1), dividend, magicNumber, \
                                False, tmpVgpr, "s_magic mul"))
    module.add(SLShiftRightB64(dst=sgpr(dest,2), shiftHex=magicShift, src=sgpr(dest,2), comment="sMagicDiv"))
    return module

##############################################################################
# Perform a sgpr version of magic division algo 2 (mul by magic number, Abit and shift)
# dest is three consec SGPR, used for intermediate temp as well as final result
# result quotient returned in sgpr(dest,1)
##############################################################################
def sMagicDivAlg2(dest, dividend, magicNumber, magicShiftAbit):
    # dest+0: q,
    # dest+1: intermediate for magic div
    # dest+2: A tmpS to store the 'Abit' and the final Shift (use tmpS to save sgpr)
    tmpS = dest+2

    module = Module("sMagicDivAlg2")
    module.add(SMulHIU32(dst=sgpr(dest+1), src0=dividend, src1=magicNumber, comment=" s_magic mul, div alg 2"))
    module.add(SLShiftRightB32(dst=sgpr(tmpS), shiftHex=31, src=magicShiftAbit, comment=" tmpS = extract abit"))                             # tmpS = MagicAbit
    module.add(SMulI32(dst=sgpr(dest), src0=dividend, src1=sgpr(tmpS), comment=" s_magic mul, div alg 2"))
    module.add(SAddU32(dst=sgpr(dest), src0=sgpr(dest), src1=sgpr(dest+1), comment=""))

    module.add(SAndB32(dst=sgpr(tmpS), src0=magicShiftAbit, src1=hex(0x7fffffff), comment=" tmpS = remove abit to final shift"))   # tmpS = MagicShift
    module.add(SLShiftRightB32(dst=sgpr(dest), shiftHex=sgpr(tmpS), src=sgpr(dest), comment=" sMagicDiv Alg 2"))
    return module

########################################
# Multiply
# product register, operand register, multiplier
########################################

def staticMultiply(product, operand, multiplier, tmpSgprRes: Optional[RegisterPoolResource], comment=""):
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    module = Module("staticMultiply")
    if multiplier == 0:
        module.add(VMovB32(dst=product, src=hex(multiplier), comment=comment))
    elif ((multiplier & (multiplier - 1)) == 0): # pow of 2
        multiplier_log2 = log2(multiplier)
        if multiplier_log2==0 and product == operand:
            module.addCommentAlign(comment + " (multiplier is 1, do nothing)")
        else:
            module.add(VLShiftLeftB32(dst=product, shiftHex=hex(multiplier_log2), src=operand, comment=comment))
    else:
        if multiplier <= 64 and multiplier >= -16:
            module.add(VMulLOU32(dst=product, src0=hex(multiplier), src1=operand, comment=comment))
        else:
            assert tmpSgprRes and tmpSgprRes.size >= 1
            tmpSgpr = tmpSgprRes.idx
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(multiplier), comment=comment))
            module.add(VMulLOU32(dst=product, src0=sgpr(tmpSgpr), src1=operand, comment=comment))
    return module


########################################
# Multiply scalar for 64bit
# product register, operand register, multiplier
########################################

def scalarStaticMultiply(product, operand, multiplier, tmpSgpr=None, comment=""):
    module = Module("scalarStaticMultiply")
    if comment == "":
        comment = "%s = %s * %s" % (product, operand, multiplier)

    if multiplier == 0:
        module.add(SMovB64(dst=product, src=hex(multiplier), comment=comment))
        return module

    # TODO- to support non-pow2, need to use mul_32 and mul_hi_32 ?
    assert ((multiplier & (multiplier - 1)) == 0) # assert pow of 2

    multiplier_log2 = log2(multiplier)
    if multiplier_log2==0 and product == operand:
        module.addCommentAlign(comment + " (multiplier is 1, do nothing)")
    else:
        # notice that the src-order of s_lshl_b64 is different from v_lshlrev_b32.
        module.add(SLShiftLeftB64(dst=product, shiftHex=hex(multiplier_log2), src=operand, comment=comment))
    return module
