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

from Tensile.TensileInstructions import FLATModifiers, MUBUFModifiers, VLShiftLeftOrB32, \
                                        VLShiftLeftB32, _VLShiftLeftOrB32, VOrB32, VAddI32, \
                                        SSleep, BufferAtomicAddF32, FlatAtomicCmpswapB32, \
                                        vgpr, sgpr

def test_v_lshl_or_b32():
    dst = vgpr(0)
    src0, src1 = vgpr(0), vgpr(1)
    shift = "0x8"
    inst = VLShiftLeftOrB32(dst, shift, src0, src1)
    asmCapsCandidates = [{"HasLshlOr": True}, {"HasLshlOr": False}]
    numInstructions = [1, 2]
    instructionsTypes = [[_VLShiftLeftOrB32], [VLShiftLeftB32, VOrB32]]
    ios = [((dst, (src0, shift, src1)),), ((dst, (shift, src0)), (dst, (dst, src1)),)]

    for asmCaps, n, instTypes, io in zip(asmCapsCandidates, numInstructions, instructionsTypes, ios):
        inst.setProperties({}, asmCaps, {}, {})
        inst.setupInstructions()
        assert len(inst.instructions) == n, "Num instructions mismatched"

        for i, t, internalIO in zip(inst.instructions, instTypes, io):
            assert type(i) is t, "Internal inst type mismatched"
            assert i.dst == internalIO[0]
            assert len(i.src) == len(internalIO[1])
            for instSrc, expectedSrc in zip(i.src, internalIO[1]):
                assert instSrc == expectedSrc

def test_v_add_i32():
    dst = vgpr(0)
    src0, src1 = vgpr(1), vgpr(2)
    inst = VAddI32(dst, src0, src1)
    asmBugs = [{'ExplicitNC': True, 'ExplicitCO': False}, \
               {'ExplicitNC': False, 'ExplicitCO': True},
               {'ExplicitNC': False, 'ExplicitCO': False},]
    instructions = ('v_add_nc_i32', 'v_add_i32', 'v_add_i32')

    for asmBug, instStr in zip(asmBugs, instructions):
        inst.setProperties(None, None, asmBug, None)
        inst.preStr()
        assert inst.instStr == instStr

def test_s_sleep():
    for i in range(1, 10):
        inst = SSleep(i)
        assert inst.instStr == "s_sleep"
        assert inst.simm16 == i

def test_buffer_atomic_add_f32():
    v = vgpr(0)
    u = vgpr(1)
    srd = sgpr(0, 4)
    i = 0
    offset12 = 10
    inst = BufferAtomicAddF32(v, u, srd, i, MUBUFModifiers(offen=True, offset12=offset12))
    inst.setProperties({}, {}, {}, {})
    assert inst.instStr == "buffer_atomic_add_f32"
    assert inst.mubuf.offen == True
    assert inst.mubuf.offset12 == offset12
    assert inst.saddr == srd
    assert inst.vaddr == u
    assert inst.srcData == v
    assert str(inst) == f"buffer_atomic_add_f32 v0, v1, s[0:3], {i} offen offset:{offset12}\n"

def test_flat_atomic_cmpswap():
    dstAddr = vgpr(0)
    tmp = vgpr(1)
    srcVal = vgpr(2)
    inst = FlatAtomicCmpswapB32(dstAddr, tmp, srcVal, FLATModifiers(glc=True))
    assert inst.instStr == "flat_atomic_cmpswap"
    assert inst.vaddr == dstAddr
    assert inst.tmp == tmp
    assert inst.srcData == srcVal
    assert inst.getArgStr() == "v0, v1, v2"
    assert inst.flat.glc is True
    assert str(inst).rstrip() == "flat_atomic_cmpswap v0, v1, v2 glc"
