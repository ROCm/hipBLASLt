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

from enum import Enum

class SignatureValueKind(Enum):
    SIG_VALUE        = 1
    SIG_GLOBALBUFFER = 2

class InstType(Enum):
    INST_F8         = 1
    INST_F16        = 2
    INST_F32        = 3
    INST_F64        = 4
    INST_I8         = 5
    INST_I16        = 6
    INST_I32        = 7
    INST_U8         = 8
    INST_U16        = 9
    INST_U32        = 10
    INST_U64        = 11
    INST_LO_I32     = 12
    INST_HI_I32     = 13
    INST_LO_U32     = 14
    INST_HI_U32     = 15
    INST_BF16       = 16
    INST_B8         = 17
    INST_B16        = 18
    INST_B32        = 19
    INST_B64        = 20
    INST_B128       = 21
    INST_B256       = 22
    INST_B512       = 23
    INST_B8_HI_D16  = 24
    INST_D16_U8     = 25
    INST_D16_HI_U8  = 26
    INST_D16_U16    = 27
    INST_D16_HI_U16 = 28
    INST_D16_B8     = 29
    INST_D16_HI_B8  = 30
    INST_D16_B16    = 31
    INST_D16_HI_B16 = 32
    INST_CVT        = 33
    INST_MACRO      = 34
    INST_NOTYPE     = 35

class SelectBit(Enum):
    DWORD  = 1
    BYTE_0 = 2
    BYTE_1 = 3
    BYTE_2 = 4
    BYTE_3 = 5
    WORD_0 = 6
    WORD_1 = 7

class UnusedBit(Enum):
    UNUSED_PAD      = 1
    UNUSED_SEXT     = 2
    UNUSED_PRESERVE = 3

class CvtType(Enum):
    CVT_F16_to_F32 = 1
    CVT_F32_to_F16 = 2
    CVT_U32_to_F32 = 3
    CVT_F32_to_U32 = 4
    CVT_I32_to_F32 = 5
    CVT_F32_to_I32 = 6

class RoundType(Enum):
    ROUND_UP = 0
    ROUND_TO_NEAREST_EVEN = 1