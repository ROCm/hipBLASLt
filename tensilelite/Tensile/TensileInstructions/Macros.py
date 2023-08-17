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

def SYNCHRONIZERLSHRREV() -> Module:
    module = Module("SYNCHRONIZERLSHRREV")
    #module.addComment1("Magic div and mod functions")
    macro = Macro("SYNCHRONIZERLSHRREV", "vgprscale")
    # macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))
    macro.addGSUSYNC("V_LSHRREV_B32 \\vgprscale, 0x1, \\vgprscale\n")
    module.add(macro)
    return module

def GSUSYNC2(StoreVectorWidth, issingle) -> Module:
    module = Module("GSUSYNC2")
    #module.addComment1("Magic div and 2mod functions")
    if StoreVectorWidth==2:
        macro = Macro("GSUSYNC2", "vgprstart", "vgprstart2", "vgproffset")
    else:
        macro = Macro("GSUSYNC2", "vgprstart", "vgproffset")
    # macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))

    if StoreVectorWidth==2:
        contents = \
    "\n\
    v_mov_b32 v[\\vgprstart+2], v[\\vgprstart+0]\n\
    v_mov_b32 v[\\vgprstart+3], v[\\vgprstart+1]\n\
    v_mov_b32 v[\\vgprstart+0], v[\\vgprstart2+0]\n\
    v_mov_b32 v[\\vgprstart+1], v[\\vgprstart2+1]\n"
        macro.addGSUSYNC(contents)

    contents = ""
    if not issingle:
        contents = \
    "\n\
    V_LSHRREV_B32 \\vgproffset, 0x1, \\vgproffset\n\
    \n"
    macro.addGSUSYNC(contents)

    contents = \
    "\n\
    s_mov_b32 s[sgprSrdD+2], 0x80000000\n\
    s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD\n\
    \n\
    s_mul_i32 s[sgprtmp2E], MT1, s[sgprWorkGroup1]              // <- wg1*MT1\n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprtmp2E], s[sgprStrideC1J]            // ScaleC s62 by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprtmp2E], s[sgprStrideC1J]               // ScaleC s62 by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 1                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprAddressTC+0], s[sgprtmp0E]    // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprAddressTC+1], s[sgprtmp1E]   // add hi to SRD\n\
    \n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 1                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp0E]        // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp1E]       // add hi to SRD\n"

    if issingle:
        contents = \
    "\n\
    s_mov_b32 s[sgprSrdD+2], 0x80000000\n\
    s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD\n\
    \n\
    s_mul_i32 s[sgprtmp2E], MT1, s[sgprWorkGroup1]              // <- wg1*MT1\n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprtmp2E], s[sgprStrideC1J]            // ScaleC s62 by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprtmp2E], s[sgprStrideC1J]               // ScaleC s62 by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 2                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprAddressTC+0], s[sgprtmp0E]    // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprAddressTC+1], s[sgprtmp1E]   // add hi to SRD\n\
    \n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 2                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp0E]        // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp1E]       // add hi to SRD\n"
    else:
        contents = \
    "\n\
    s_mov_b32 s[sgprSrdD+2], 0x80000000\n\
    s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD\n\
    \n\
    s_mul_i32 s[sgprtmp2E], MT1, s[sgprWorkGroup1]              // <- wg1*MT1\n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprtmp2E], s[sgprStrideC1J]            // ScaleC s62 by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprtmp2E], s[sgprStrideC1J]               // ScaleC s62 by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 1                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprAddressTC+0], s[sgprtmp0E]    // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprAddressTC+1], s[sgprtmp1E]   // add hi to SRD\n\
    \n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp0E+1], s[sgprtmp0E:sgprtmp0E+1], 1                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp0E]        // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp1E]       // add hi to SRD\n"

    macro.addGSUSYNC(contents)

    if issingle:
        contents = \
        "\n\
        \n\
        buffer_store_dwordx4 v[\\vgprstart:\\vgprstart+3], \\vgproffset, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D\n\
        \n\
        //GSUFusion\n"
    else:
        contents = \
        "\n\
        v_cvt_f16_f32 v[\\vgprstart+0], v[\\vgprstart+0]\n\
        v_cvt_f16_f32 v[\\vgprstart+1], v[\\vgprstart+1]\n\
        v_cvt_f16_f32 v[\\vgprstart+2], v[\\vgprstart+2]\n\
        v_cvt_f16_f32 v[\\vgprstart+3], v[\\vgprstart+3]\n\
        \n\
        v_pack_b32_f16 v[\\vgprstart+0], v[\\vgprstart+0], v[\\vgprstart+1]\n\
        v_pack_b32_f16 v[\\vgprstart+1], v[\\vgprstart+2], v[\\vgprstart+3]\n\
        \n\
        //v_cvt_pkrtz_f16_f32 v[\\vgprstart+0], v[\\vgprstart+0], v[\\vgprstart+1]\n\
        //v_cvt_pkrtz_f16_f32 v[\\vgprstart+1], v[\\vgprstart+2], v[\\vgprstart+3]\n\
        \n\
        buffer_store_dwordx2 v[\\vgprstart:\\vgprstart+1], \\vgproffset, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D\n\
        \n\
        //GSUFusion\n"
    macro.addGSUSYNC(contents)
    module.add(macro)
    return module

def GSUSYNC0(GSU, MT0, MT1) -> Module:
    module = Module("GSUSYNC0")
    #module.addComment1("Magic div and 2mod functions")

    macro = Macro("GSUSYNC0", "labelname", "labelendname")
    if MT1>MT0:
        WaveNum = "MT1/MT0"
    else:
        WaveNum = "MT0/MT1"
    contents = \
    "\n\
    //Victor\n\
    \labelname:\n\
    s_mov_b32 s[sgprGSUSync], 0\n\
    s_atomic_add s[sgprGSUSync], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 glc\n\
    //s_waitcnt 0\n\
    //s_cmp_ge_u32 s[sgprGSUSync], "+WaveNum+"                    // Beta == 0\n\
    //s_cbranch_scc0 \labelname           // jump if XX required\n\
\n\
    //s_mov_b32 s[sgprGSUSync] 1\n\
\n\
    //s_mov_b32 s[sgprtmp0E], s[sgprGSUSumIdx]\n\
    //s_lshl_b32 s[sgprtmp0E], s[sgprtmp0E], 2\n\
\n\
    //s_mul_hi_u32 s[sgprtmp3E], s[sgprStrideCK], GSU            // ScaleC s62 by Stride\n\
    //s_mul_i32 s[sgprtmp2E], s[sgprStrideCK], GSU               // ScaleC s62 by Stride\n\
    //s_lshl_b64 s[sgprtmp2E:sgprtmp2E+1], s[sgprtmp2E:sgprtmp2E+1], 2                   // scale by bpe\n\
\n\
    //s_mov_b32 s[sgprSrdDd+2], 0x80000000\n\
    //s_mov_b32 s[sgprSrdDd+3], Srd127_96                 // Set bits 127_96 in post-loop SRD\n\
\n\
    //s_add_u32 s[sgprSrdDd+0], s[sgprAddressD+0], s[sgprtmp2E]    // add lo to SRD\n\
    //s_addc_u32 s[sgprSrdDd+1], s[sgprAddressD+1], s[sgprtmp3E]   // add hi to SRD\n\
\n\
    //s_add_u32 s[sgprSrdDd+0], s[sgprSrdDd+0], s[sgprtmp0E]    // add lo to SRD\n\
    //s_addc_u32 s[sgprSrdDd+1], s[sgprSrdDd+1], 0   // add hi to SRD\n\
\n\
    //s_buffer_atomic_add s[sgprGSUSync], s[sgprSrdDd:sgprSrdDd+3], offset:0 glc\n\
    //s_waitcnt 0\n\
    //s_waitcnt lgkmcnt(0)\n\
\n\
    //s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
    //s_cbranch_scc0 \labelendname //label_GW_End_1 //label_AFTERsummary_Edge\n\
    //Victor\n\
\n\
    //GSUFusion\n"
    macro.addGSUSYNC(contents)
    module.add(macro)
    return module

def GSUSYNC1(GSU, MT0, MT1) -> Module:
    module = Module("GSUSYNC1")
    #module.addComment1("Magic div and 2mod functions")

    macro = Macro("GSUSYNC1", "labelname", "labelendname")
    if MT1>MT0:
        WaveNum = "MT1/MT0"
    else:
        WaveNum = "MT0/MT1"
    contents = \
    "\n\
    //Victor\n\
    //\labelname:\n\
    //s_mov_b32 s[sgprGSUSync], 0\n\
    //s_atomic_add s[sgprGSUSync], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 glc\n\
    //s_waitcnt 0\n\
    //s_cmp_ge_u32 s[sgprGSUSync], "+WaveNum+"                    // Beta == 0\n\
    //s_cbranch_scc0 \labelname           // jump if XX required\n\
\n\
    //s_mov_b32 s[sgprGSUSync] 1\n\
\n\
    //s_mov_b32 s[sgprtmp0E], s[sgprGSUSumIdx]\n\
    //s_lshl_b32 s[sgprtmp0E], s[sgprtmp0E], 2\n\
\n\
    //s_mul_hi_u32 s[sgprtmp3E], s[sgprStrideCK], GSU            // ScaleC s62 by Stride\n\
    //s_mul_i32 s[sgprtmp2E], s[sgprStrideCK], GSU               // ScaleC s62 by Stride\n\
    //s_lshl_b64 s[sgprtmp2E:sgprtmp2E+1], s[sgprtmp2E:sgprtmp2E+1], 2                   // scale by bpe\n\
\n\
    //s_mov_b32 s[sgprSrdDd+2], 0x80000000\n\
    //s_mov_b32 s[sgprSrdDd+3], Srd127_96                 // Set bits 127_96 in post-loop SRD\n\
\n\
    //s_add_u32 s[sgprSrdDd+0], s[sgprAddressD+0], s[sgprtmp2E]    // add lo to SRD\n\
    //s_addc_u32 s[sgprSrdDd+1], s[sgprAddressD+1], s[sgprtmp3E]   // add hi to SRD\n\
\n\
    //s_add_u32 s[sgprSrdDd+0], s[sgprSrdDd+0], s[sgprtmp0E]    // add lo to SRD\n\
    //s_addc_u32 s[sgprSrdDd+1], s[sgprSrdDd+1], 0   // add hi to SRD\n\
\n\
    //s_buffer_atomic_add s[sgprGSUSync], s[sgprSrdDd:sgprSrdDd+3], offset:0 glc\n\
    //s_waitcnt 0\n\
    //s_waitcnt lgkmcnt(0)\n\
\n\
    //s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
    //s_cbranch_scc0 \labelendname //label_GW_End_1 //label_AFTERsummary_Edge\n\
    //Victor\n\
\n\
    //GSUFusion\n"
    macro.addGSUSYNC(contents)
    module.add(macro)
    return module

def GSUSYNC(GSU, MT0, MT1, StoreVectorWidth) -> Module:
    module = Module("GSUSYNC")
    #module.addComment1("Magic div and 2mod functions")
    if StoreVectorWidth==2:
        macro = Macro("GSUSYNC", "labelname", "labelendname", "vgprstart", "vgprstart2", "vgproffset")
    else:
        macro = Macro("GSUSYNC", "labelname", "labelendname", "vgprstart", "vgproffset")
    # macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))

    # if MT1>MT0:
    #     WaveNum = "MT1/MT0"
    # else:
    #     WaveNum = "MT0/MT1"

    WaveNum = str(MT1*MT0)

    contents = \
    "\n\
    //Victor\n\
    //\labelname:\n\
    //s_mov_b32 s[sgprGSUSync], 0\n\
    //s_atomic_add s[sgprGSUSync], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 glc\n\
    s_waitcnt lgkmcnt(0)\n\
    s_cmp_ge_u32 s[sgprGSUSync], "+WaveNum+"                    // Beta == 0\n\
    s_cbranch_scc0 \labelname           // jump if XX required\n\
\n\
    s_mov_b32 s[sgprGSUSync] 1\n\
\n\
    s_mov_b32 s[sgprtmp0E], s[sgprGSUSumIdx]\n\
    s_lshl_b32 s[sgprtmp0E], s[sgprtmp0E], 2\n\
\n\
    s_mul_hi_u32 s[sgprtmp3E], s[sgprStrideCK], GSU            // ScaleC s62 by Stride\n\
    s_mul_i32 s[sgprtmp2E], s[sgprStrideCK], GSU               // ScaleC s62 by Stride\n\
    s_lshl_b64 s[sgprtmp2E:sgprtmp2E+1], s[sgprtmp2E:sgprtmp2E+1], 2                   // scale by bpe\n\
\n\
    s_mov_b32 s[sgprSrdDd+2], 0x80000000\n\
    s_mov_b32 s[sgprSrdDd+3], Srd127_96                 // Set bits 127_96 in post-loop SRD\n\
\n\
    s_add_u32 s[sgprSrdDd+0], s[sgprAddressD+0], s[sgprtmp2E]    // add lo to SRD\n\
    s_addc_u32 s[sgprSrdDd+1], s[sgprAddressD+1], s[sgprtmp3E]   // add hi to SRD\n\
\n\
    s_add_u32 s[sgprSrdDd+0], s[sgprSrdDd+0], s[sgprtmp0E]    // add lo to SRD\n\
    s_addc_u32 s[sgprSrdDd+1], s[sgprSrdDd+1], 0   // add hi to SRD\n\
\n\
\n\
    //s_mov_b32 s[sgprGSUSumIdx] 1\n\
    s_mul_i32 s[sgprtmp2E], MT1, s[sgprWorkGroup1]              // <- wg1*MT1\n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprtmp2E], s[sgprStrideD1J]            // ScaleD s64 by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprtmp2E], s[sgprStrideD1J]               // ScaleD s64 by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp1E], s[sgprtmp0E:sgprtmp1E], 2                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s[sgprtmp0E]    // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s[sgprtmp1E]   // add hi to SRD\n\
\n\
    s_mul_hi_u32 s[sgprtmp1E], s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride\n\
    s_mul_i32 s[sgprtmp0E], s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride\n\
    s_lshl_b64 s[sgprtmp0E:sgprtmp1E], s[sgprtmp0E:sgprtmp1E], 2                   // scale by bpe\n\
    s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp0E]        // add lo to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp1E]       // add hi to SRD\n\
\n\
    s_waitcnt vmcnt(0)\n\
    s_buffer_atomic_add s[sgprGSUSync], s[sgprSrdDd:sgprSrdDd+3], offset:0 glc\n\
    s_waitcnt lgkmcnt(0)\n\
    s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
    s_cbranch_scc0 \labelendname //label_GW_End_1 //label_AFTERsummary_Edge\n\
    //Victor\n\
\n\
    //GSUFusion\n\
\n\
    buffer_load_dwordx4 v[\\vgprstart+4*0:\\vgprstart+3+4*0], \\vgproffset, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D\n\
\n\
    // GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s\n\
    //s_mul_hi_u32 s[sgprtmp1E], s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0\n\
    //s_mul_i32 s[sgprtmp0E], s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0\n\
    s_sub_u32 s[sgprtmp5E], s[sgprSizesFree+1], 1               // Free1\n\
    //s_mul_i32 s[sgprtmp5E], s[sgprtmp5E], s[sgprGSUSumIdx]               // Free1\n\
    s_mul_hi_u32 s[sgprtmp3E], s[sgprtmp5E], s[sgprStrideC1J]            // Free1\n\
    s_mul_i32 s[sgprtmp2E], s[sgprtmp5E], s[sgprStrideC1J]               // Free1\n\
    s_add_u32 s[sgprtmp0E], s[sgprSizesFree+0], s[sgprtmp2E]                            // Free1\n\
    s_addc_u32 s[sgprtmp1E], 0, s[sgprtmp3E]                           // Free1\n\
    s_sub_u32 s[sgprtmp5E], s[sgprSizesFree+2], 1               // Free2\n\
    //s_mul_i32 s[sgprtmp5E], s[sgprtmp5E], s[sgprGSUSumIdx]               // Free2\n\
    s_mul_hi_u32 s[sgprtmp3E], s[sgprtmp5E], s[sgprStrideCK]             // Free2\n\
    s_mul_i32 s[sgprtmp2E], s[sgprtmp5E], s[sgprStrideCK]                // Free2\n\
    s_add_u32 s[sgprtmp0E], s[sgprtmp0E], s[sgprtmp2E]                            // Free2\n\
    s_addc_u32 s[sgprtmp1E], s[sgprtmp1E], s[sgprtmp3E]                           // Free2\n\
    s_lshl_b64 s[sgprtmp2E:sgprtmp3E], s[sgprtmp0E:sgprtmp1E], 2                   // scale by bpe\n\
\n\
    //s_mov_b32 s[sgprGSUSumIdx] 1\n\
    //s_mul_i32 s[sgprtmp0E], s[sgprtmp2E], s[sgprGSUSumIdx] // Free0\n\
    //s_mul_hi_u32 s[sgprtmp5E], s[sgprtmp2E], s[sgprGSUSumIdx] // Free0\n\
    //s_mul_i32 s[sgprtmp1E], s[sgprtmp3E], s[sgprGSUSumIdx] // Free0\n\
    //s_add_u32 s[sgprtmp1E], s[sgprtmp1E], s[sgprtmp5E]\n"
    macro.addGSUSYNC(contents)

    contents = \
"\n\
    s_waitcnt lgkmcnt(0)\n\
    s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
    s_cbranch_scc0 \labelendname //label_GW_End_1 //label_AFTERsummary_Edge\n"
    macro.addGSUSYNC(contents)

    for i in range(1,GSU):
        print(i)
        contents = \
        "\n\
    s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s[sgprtmp2E]        // add lo GSU offset to SRD\n\
    s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s[sgprtmp3E]       // add hi GSU offset to SRD\n\
    buffer_load_dwordx4 v[\\vgprstart+4*"+str(i)+":\\vgprstart+3+4*"+str(i)+"], \\vgproffset, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D\n"
        macro.addGSUSYNC(contents)

    contents = \
"\n\
    s_waitcnt lgkmcnt(0)\n\
\n\
    s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
    s_cbranch_scc0 \labelendname //label_GW_End_1 //label_AFTERsummary_Edge\n"
    # macro.addGSUSYNC(contents)

    for i in range(1,GSU):
        print(i)
        contents = \
        "\n\
    s_waitcnt vmcnt("+str(GSU-1-i)+")\n\
    V_PK_ADD_F32 v[\\vgprstart+0:\\vgprstart+1], v[\\vgprstart+0:\\vgprstart+1], v[\\vgprstart+4*"+str(i)+"+0:\\vgprstart+4*"+str(i)+"+1]\n\
    V_PK_ADD_F32 v[\\vgprstart+2:\\vgprstart+3], v[\\vgprstart+2:\\vgprstart+3], v[\\vgprstart+4*"+str(i)+"+2:\\vgprstart+4*"+str(i)+"+3]\n"
        macro.addGSUSYNC(contents)

    if StoreVectorWidth==2:
        contents = \
    "\n\
    v_mov_b32 v[\\vgprstart2+0], v[\\vgprstart+0]\n\
    v_mov_b32 v[\\vgprstart2+1], v[\\vgprstart+1]\n\
    v_mov_b32 v[\\vgprstart+0], v[\\vgprstart+2]\n\
    v_mov_b32 v[\\vgprstart+1], v[\\vgprstart+3]\n"
        macro.addGSUSYNC(contents)

    contents = \
"\n\
    s_waitcnt lgkmcnt(0)\n\
\n\
    s_cmp_ge_u32 s[sgprGSUSync], GSU*"+WaveNum+"-1                // s[Alpha] == 0.0f ?\n\
    s_cbranch_scc0 \labelendname //label_GW_End_1 //label_AFTERsummary_Edge\n"
    # macro.addGSUSYNC(contents)

    module.add(macro)
    return module

# def GSUSYNCspgr() -> Module:
#     module = Module("GSUSYNCspgr")
#     # module.addComment1("Magic div and mod functions")
#     macro = Macro("GSUSYNCspgr", "vgprscale")
#     # macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))
#     contents = \
# "\n\
# //GSUFusion\n\
# .set sgprtmp0E, 88\n\
# .set sgprtmp1E, sgprtmp0E+1\n\
# .set sgprtmp2E, sgprtmp0E+2\n\
# .set sgprtmp3E, sgprtmp0E+3\n\
# .set sgprtmp4E, sgprtmp0E+4\n\
# .set sgprtmp5E, sgprtmp0E+5\n\
# .set sgprtmp6E, sgprtmp0E+6\n\
# .set sgprtmp7E, sgprtmp0E+7\n\
# .set sgprSrdDd, sgprtmp0E+8\n\
# .set sgprSrdtmp, sgprSrdDd+4 //sgprtmp0E+12(+4)\n\
# //GSUFusion\n"
#     macro.addGSUSYNC(contents)
#     module.add(macro)
#     return module

def GSUSYNCzero(GSU) -> Module:
    module = Module("GSUSYNCzero")
    # module.addComment1("Magic div and mod functions")
    macro = Macro("GSUSYNCzero")
    # macro.add(VCvtU32toF32(dst="v[\\vQuotient]", src="v[\\vDivisor]"))
    contents = \
    "\n\
    //Victor\n\
    \n\
    v_mov_b32 v0, 0.0 //src\n\
    v_mov_b32 v1, 0.0 //cmp\n\
    v_mov_b32 v2, 0.0 \n\
    v_mov_b32 v3, 0.0 //cmp\n\
    \n\
    S_OR_B32 s[sgprSrdDd], s[sgprWorkGroup0], s[sgprWorkGroup1]\n\
    s_cmp_eq_u32 s[sgprSrdDd], 0                // s[Alpha] == 0.0f ?\n\
    s_cbranch_scc0 label_AFTERINITZERO           // jump if XX required\n\
    \n\
    s_cmp_eq_u32 s[sgprGSUSumIdx], 0                // s[Alpha] == 0.0f ?\n\
    s_cbranch_scc0 label_AFTERINITZERO           // jump if XX required\n\
    \n\
    //s_mul_hi_u32 s67, s[sgprStrideCK], s[sgprStrideC1J]            // ScaleC s62 by Stride\n\
    //s_mul_i32 s66, s[sgprStrideCK], s[sgprStrideC1J]               // ScaleC s62 by Stride\n\
    \n\
    //s_lshl_b64 s[66:67], s[66:67], 2                   // scale by bpe\n\
    \n\
    s_mul_hi_u32 s[sgprtmp3E], s[sgprStrideCK], GSU            // ScaleC s62 by Stride\n\
    s_mul_i32 s[sgprtmp2E], s[sgprStrideCK], GSU               // ScaleC s62 by Stride\n\
    s_lshl_b64 s[sgprtmp2E:sgprtmp2E+1], s[sgprtmp2E:sgprtmp2E+1], 2                   // scale by bpe\n\
    //s_mul_i32 s[66], s[sgprStrideCK], 6\n\
    //s_lshl_b32 s[66], s[66], 2                   // scale by bpe\n\
    \n\
    s_mov_b32 s[sgprSrdDd+2], 0x80000000\n\
    s_mov_b32 s[sgprSrdDd+3], Srd127_96                 // Set bits 127_96 in post-loop SRD\n\
    \n\
    s_add_u32 s[sgprSrdDd+0], s[sgprAddressD+0], s[sgprtmp2E]    // add lo to SRD\n\
    s_addc_u32 s[sgprSrdDd+1], s[sgprAddressD+1], s[sgprtmp3E]   // add hi to SRD\n"
    macro.addGSUSYNC(contents)

    r, mod = divmod(GSU, 4)
    print("sdgdfpghk[rk[pkdhkdf[gpldp]]]")
    print(GSU)
    print(r)
    print(mod)

    for i in range(GSU//4):
        contents = \
        "\n\
        buffer_store_dwordx4 v[0:3], v0, s[sgprSrdDd:sgprSrdDd+3], 0 offen offset:4*"+str(i)+" // attempt write avi=0\n"
        macro.addGSUSYNC(contents)
    i = GSU//4
    if mod == 1:
        contents = \
        "\n\
        buffer_store_dword v[0], v0, s[sgprSrdDd:sgprSrdDd+3], 0 offen offset:4*"+str(i)+" // attempt write avi=0\n"
    if mod == 2:
        contents = \
        "\n\
        buffer_store_dwordx2 v[0:1], v0, s[sgprSrdDd:sgprSrdDd+3], 0 offen offset:4*"+str(i)+" // attempt write avi=0\n"
    if mod == 3:
        contents = \
        "\n\
        buffer_store_dwordx2 v[0:1], v0, s[sgprSrdDd:sgprSrdDd+3], 0 offen offset:4*"+str(i)+" // attempt write avi=0\n\
        buffer_store_dword v[0], v0, s[sgprSrdDd:sgprSrdDd+3], 0 offen offset:4*"+str(i)+"+2 // attempt write avi=0\n"
    macro.addGSUSYNC(contents)

    contents = \
    "\n\
    s_mov_b32 s[sgprGSUSync] 1\n\
    s_atomic_add s[sgprGSUSync], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 glc\n\
    s_waitcnt vmcnt(0)                                 // 8wait for global read\n\
    \n\
    label_AFTERINITZERO:                              // jump to end\n\
    //Victor\n"
    macro.addGSUSYNC(contents)
    module.add(macro)
    return module