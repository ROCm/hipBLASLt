
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected Custom_Cijk_Ailk_Bljk_F8H_HHS_BH_Bias_AS_SAB_SAV_MT128x16x128_MI16x16x1_WaveSpecialization
.globl Custom_Cijk_Ailk_Bljk_F8H_HHS_BH_Bias_AS_SAB_SAV_MT128x16x128_MI16x16x1_WaveSpecialization
.p2align 8
.type Custom_Cijk_Ailk_Bljk_F8H_HHS_BH_Bias_AS_SAB_SAV_MT128x16x128_MI16x16x1_WaveSpecialization,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Custom_Cijk_Ailk_Bljk_F8H_HHS_BH_Bias_AS_SAB_SAV_MT128x16x128_MI16x16x1_WaveSpecialization
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 192 // accvgpr offset
  .amdhsa_next_free_vgpr 200 // vgprs
  .amdhsa_next_free_sgpr 80 // sgprs
  .amdhsa_group_segment_fixed_size 21504 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
/* Num VGPR   =160 */
/* Num AccVGPR=8 */
/* Num SGPR   =80 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 8 x 1 */
/* SubGroup= 16 x 16 */
/* VectorWidthA=2 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=16, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
custom.config:
   ProblemType:
      OperationType: GEMM
      DataTypeA: f8
      DataTypeB: h
      UseScaleAB: True
      DataType: h
      DestDataType: h
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: False
      UseBeta: True
      Batched: True
      UseBias: True
      Activation: True
      UseScaleAlphaVec: True
   MatrixInstruction: [16, 16, 16, 1, 1, 2,1, 4,2]
   1LDSBuffer: 1
   DepthU: 128
   StaggerU: 0
   WorkGroupMapping: 1
   GlobalReadVectorWidthA: 16
   GlobalReadVectorWidthB: 8
   AssertFree0ElementMultiple: 16
   AssertSummationElementMultiple: 1
   GlobalSplitU: 4
   GlobalSplitUAlgorithm: MultipleBuffer
   InternalSupportParams: {SupportCustomWGM: True, SupportUserGSU: True, SupportCustomStaggerU: True}
   NoReject: 1
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Custom_Cijk_Ailk_Bljk_F8H_HHS_BH_Bias_AS_SAB_SAV_MT128x16x128_MI16x16x1_WaveSpecialization
    .symbol: 'Custom_Cijk_Ailk_Bljk_F8H_HHS_BH_Bias_AS_SAB_SAV_MT128x16x128_MI16x16x1_WaveSpecialization.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            SizesFree0
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            D
        .size:            8
        .offset:          16
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          24
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          48
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          52
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          56
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          60
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      f32
      - .name:            internalArgs
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            AddressScaleA
        .size:            8
        .offset:          92
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleB
        .size:            8
        .offset:          100
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleAlphaVec
        .size:            8
        .offset:          108
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            bias
        .size:            8
        .offset:          116
        .value_kind:      global_buffer
        .value_type:      void
        .address_space:   generic
      - .name:            biasType
        .size:            4
        .offset:          124
        .value_kind:      by_value
        .value_type:      u32
      - .name:            StrideBias
        .size:            4
        .offset:          128
        .value_kind:      by_value
        .value_type:      u32
      - .name:            activationAlpha
        .size:            4
        .offset:          132
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationBeta
        .size:            4
        .offset:          136
        .value_kind:      by_value
        .value_type:      f32
      - .name:            activationType
        .size:            4
        .offset:          140
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   21504
    .kernarg_segment_align:      8
    .kernarg_segment_size:       144
    .max_flat_workgroup_size:    512
    .private_segment_fixed_size: 0
    .sgpr_count:                 80
    .sgpr_spill_count:           0
    .vgpr_count:                 192
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Custom_Cijk_Ailk_Bljk_F8H_HHS_BH_Bias_AS_SAB_SAV_MT128x16x128_MI16x16x1_WaveSpecialization:

/* Magic div and mod functions */
.macro V_MAGIC_DIV dstIdx:req dividend:req magicNumber:req magicShift:req magicA:req
    v_mul_hi_u32 v[\dstIdx+1] \dividend \magicNumber
    v_mul_lo_u32 v[\dstIdx+0] \dividend \magicA
    v_add_u32 v[\dstIdx+0] v[\dstIdx+0] v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0] \magicShift v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-0), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 0
.set vgprValuA_X1_I0, 0
.set vgprValuA_X2_I0, 0
.set vgprValuA_X3_I0, 0
.set vgprValuA_X4_I0, 0
.set vgprValuA_X5_I0, 0
.set vgprValuA_X6_I0, 0
.set vgprValuA_X7_I0, 0
.set vgprValuA_X0_I0_D0, 4
.set vgprValuA_X1_I0_D0, 5
.set vgprValuA_X2_I0_D0, 6
.set vgprValuA_X3_I0_D0, 7
.set vgprValuA_X4_I0_D0, 8
.set vgprValuA_X5_I0_D0, 9
.set vgprValuA_X6_I0_D0, 10
.set vgprValuA_X7_I0_D0, 11
.set vgprValuA_X0_I0_D1, 12
.set vgprValuA_X1_I0_D1, 13
.set vgprValuA_X2_I0_D1, 14
.set vgprValuA_X3_I0_D1, 15
.set vgprValuA_X4_I0_D1, 16
.set vgprValuA_X5_I0_D1, 17
.set vgprValuA_X6_I0_D1, 18
.set vgprValuA_X7_I0_D1, 19
.set vgprValuA_X0_I0_D2, 20
.set vgprValuA_X1_I0_D2, 21
.set vgprValuA_X2_I0_D2, 22
.set vgprValuA_X3_I0_D2, 23
.set vgprValuA_X4_I0_D2, 24
.set vgprValuA_X5_I0_D2, 25
.set vgprValuA_X6_I0_D2, 26
.set vgprValuA_X7_I0_D2, 27
.set vgprValuA_X0_I0_D3, 28
.set vgprValuA_X1_I0_D3, 29
.set vgprValuA_X2_I0_D3, 30
.set vgprValuA_X3_I0_D3, 31
.set vgprValuA_X4_I0_D3, 32
.set vgprValuA_X5_I0_D3, 33
.set vgprValuA_X6_I0_D3, 34
.set vgprValuA_X7_I0_D3, 35
.set vgprValuB_X0_I0, 36
.set vgprValuB_X1_I0, 38
.set vgprValuB_X2_I0, 40
.set vgprValuB_X3_I0, 42
.set vgprValuB_X4_I0, 44
.set vgprValuB_X5_I0, 46
.set vgprValuB_X6_I0, 48
.set vgprValuB_X7_I0, 50
.set vgprCvtTemp, 52
.set vgprLocalWriteAddrA, 54
.set vgprLocalWriteAddrB, 55
.set vgprGlobalReadOffsetA, 56
.set vgprGlobalReadOffsetB, 57
.set vgprG2LA, 58
.set vgprG2LB, 90
.set vgprLocalReadAddrA, 94
.set vgprLocalReadAddrB, 95
.set vgprSerial, 96
.set vgprG2LA1, 128
.set vgprG2LB1, 160

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprGSUSumIdx, 6
.set sgprGSULog2BpeC, 5
.set sgprGSULog2BpeD, 8
.set sgprWGM, 9
.set sgprLoopCounterL, 10
.set sgprOrigLoopCounter, 11
.set sgprSrdD, 12
.set sgprSrdC, 16
.set sgprNumWorkGroups0, 20
.set sgprNumWorkGroups1, 21
.set sgprSizesFree, 24
.set sgprSizesSum, 27
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprGSU, 46
.set sgprSrdA, 48
.set sgprSrdB, 52
.set sgprShadowLimitA, 22
.set sgprShadowLimitB, 56
.set sgprStaggerU, 47
.set sgprStaggerUIter, 58
.set sgprWrapUA, 59
.set sgprWrapUB, 61
.set sgprGlobalReadIncsA, 63
.set sgprGlobalReadIncsB, 64
.set sgprPackKForV0, 65
.set sgprPackKForV1, 66
.set sgprScalarGlobalReadOffsetA, 67

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideA0I, 1
.set sgprStrideAL, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 128
.set MT1, 16
.set DepthU, 128
.set BpeA, 2
.set BpeALog2, 1
.set BpeB, 2
.set BpeBLog2, 1
.set BpeAGR, 1
.set BpeAGRLog2, 0
.set BpeBGR, 2
.set BpeBGRLog2, 1
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 16
.set SrdShiftLeftB, 8
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x00020000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x00020000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffset0I:req vgprOffsetL:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideAL] v[\vgprOffsetL] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffset0I] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x10 v[\vgprAddr+0]     // add prepad for pointer shift
                                                       // offset *= bytes/element (multiplier is 1 do nothing)
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideB1J] v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffsetL] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x8 v[\vgprAddr+0]      // add prepad for pointer shift
    v_lshlrev_b32 v[\vgprAddr+0] 0x1 v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
    v_cvt_f32_u32 v[\vQuotient] v[\vDivisor]
    v_rcp_f32 v[\vQuotient] v[\vQuotient]
    v_mul_f32 v[\vQuotient] 0x4f800000 v[\vQuotient]
    v_cvt_u32_f32 v[\vQuotient] v[\vQuotient]
    v_mul_lo_u32 v[\vRemainder] v[\vDivisor] v[\vQuotient]
    v_mul_hi_u32 v[\vTmp0] v[\vDivisor] v[\vQuotient]
    v_sub_co_u32 v[\vTmp1] vcc 0x0 v[\vRemainder]
    v_cmp_ne_i32 s[\sTmp:\sTmp+1] 0x0 v[\vTmp0]
    v_cndmask_b32 v[\vRemainder] v[\vTmp1] v[\vRemainder] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vRemainder] v[\vRemainder] v[\vQuotient]
    v_sub_co_u32 v[\vTmp0] vcc v[\vQuotient] v[\vRemainder]
    v_add_co_u32 v[\vQuotient] vcc v[\vQuotient] v[\vRemainder]
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vTmp0] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vQuotient] v[\vQuotient] v[\vDividend]
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vTmp0] vcc v[\vDividend] v[\vRemainder]
    v_cmp_ge_u32 s[\sTmp:\sTmp+1] v[\vDividend] v[\vRemainder]
    v_add_co_u32 v[\vRemainder] vcc 0x1 v[\vQuotient]
    v_add_co_u32 v[\vTmp1] vcc -1 v[\vQuotient]
    v_cmp_le_u32 vcc v[\vDivisor] v[\vTmp0]
    s_and_b64 vcc s[\sTmp:\sTmp+1] vcc
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vRemainder] vcc
    v_cndmask_b32 v[\vQuotient] v[\vTmp1] v[\vQuotient] s[\sTmp:\sTmp+1]
    v_cmp_ne_i32 vcc 0x0 v[\vDivisor]
    v_cndmask_b32 v[\vQuotient] -1 v[\vQuotient] vcc // final result
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vRemainder] vcc v[\vDividend] v[\vRemainder] // final result
.endm


/**********************************************/
/* Check waves and switch to load or math path*/
/**********************************************/

s_mov_b32 s25, 0     /* s25/26 is overwritten, so should not be live here */
v_readlane_b32 s26, v0, s25
s_lshr_b32 s26, s26, 0x8
s_cmp_eq_u32 s26, 0x1 
s_cbranch_scc1 math_wave_start 

/******************************************/
/* LOAD WAVE START                        */
/******************************************/


/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_load_dword s46, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_mov_b32 s[sgprPackKForV0], 0x05040100
s_mov_b32 s[sgprPackKForV1], 0x07060302
s_mov_b32 m0, 0x5400                               // LDS clamp at 21504 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
/* init: add vgpr [0...54) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...8) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v0, 0x1, v0                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_and_b32 v1, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0xa, v1                          // 5. K offset: lrKOffset = kIdx * mStride(1024)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v1, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v1, 3, v1                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshlrev_b32 v1, 0x5, v1                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)
v_add_u32 v0, v1, v0                               // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x7, v1                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v2, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x3, v2                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */

/* local read addresses: final offsets b */

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 3, v[vgprSerial]                 // v1 = v[vgprSerial] / 8
v_and_b32 v0, 7, v[vgprSerial]                     // v0 = v[vgprSerial] % 8
/* tile *= glvw */
v_lshlrev_b32 v0, 0x4, v0                          // v0 = v0 * 16
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // v2 = v[vgprSerial] / 16
v_and_b32 v3, 15, v[vgprSerial]                    // v3 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v4     // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS(1)
v_lshrrev_b32 v6, 10, v[vgprLocalWriteAddrA]       // padding 32 per block 1024
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 1024
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // add padding 32 per block 1024

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrB], v6, v[vgprLocalWriteAddrB] // add padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x4200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=16896
s_waitcnt lgkmcnt(0)                               // wait for 92 bytes of kern args
s_and_b32 s[sgprWGM], s[sgprGSU], 0xff00
s_lshr_b32 s[sgprWGM], s[sgprWGM], 0x8
s_and_b32 s[sgprStaggerU], s[sgprGSU], 0xffff0000
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s[sgprGSU], 0xff
label_loadwave_stop:
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_loadwave_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_loadwave_AlphaNonZero:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
// TODO: remove it? 
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_loadwave_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
v_cvt_f32_u32 v6, s[sgprGSU]                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mul_u32_u24 v7, v6, s[sgprGSU]                   // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_sub_u32 v7, s[sgprWorkGroup1], v7                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cmpx_eq_u32 exec, v7, s[sgprGSU]                 // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s[sgprGSU]
s_mov_b64 exec, -1                                 // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_readfirstlane_b32 s[sgprWorkGroup1], v6
v_readfirstlane_b32 s[sgprGSUSumIdx], v7
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_loadwave_GSU_End
label_loadwave_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
label_loadwave_GSU_End:
s_cmp_le_u32 s[sgprWGM], 1                         // WGM <= 1 ?
s_cbranch_scc1 label_loadwave_WGM                           // branch if WGM <= 1
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprWorkGroup1], v7                // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s72, v6
s_mul_i32 s73, s72, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s73, s[sgprWorkGroup1], s73              // WorkGroup1=remainder
s_mul_i32 s73, s73, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s73, s73, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups1]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprNumWorkGroups1], v7            // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s70, v6
s_mul_i32 s71, s[sgprWGM], s70                     // quotient * non-magic divisor
s_sub_u32 s74, s[sgprNumWorkGroups1], s71          // WorkGroup1=remainder
s_cmp_eq_u32 s74, 0                                // remainder == 0 ?
s_cmov_b32 s74, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s72, s70                              // blockId >= numFullBlocks ?
s_cselect_b32 s70, s74, s[sgprWGM]
v_cvt_f32_u32 v6, s70                              // s[sgprWorkGroup0] = s73 / s70
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup0] = s73 / s70
v_cvt_f32_u32 v7, s73                              // s[sgprWorkGroup0] = s73 / s70
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup0] = s73 / s70
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup0] = s73 / s70
v_mul_u32_u24 v7, v6, s70                          // s[sgprWorkGroup0] = s73 / s70
v_sub_u32 v7, s73, v7                              // s[sgprWorkGroup0] = s73 / s70
v_cmpx_eq_u32 exec, v7, s70                        // s[sgprWorkGroup0] = s73 / s70
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup0] = s73 / s70
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s73 % s70
s_mov_b64 exec, -1                                 // s[sgprWorkGroup0] = s73 / s70
v_readfirstlane_b32 s[sgprWorkGroup0], v6
v_readfirstlane_b32 s[sgprWorkGroup1], v7
s_mul_i32 s72, s72, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s72 // wg1 += blockId * WGM
label_loadwave_WGM:

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v0 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v2 */

/* global read addresses: unroll assignment a */
/* v1 */

/* global read addresses: unroll assignment b */
/* v3 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */

/* global read addresses: tile offsets b */

/* global read addresses: unroll offsets a */

/* global read addresses: unroll offsets b */

/* global read addresses: final offsets a */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  0,  1, 6 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideAL], 32 // compute offset diff (scaled unrollDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)
s_mul_i32 s[sgprScalarGlobalReadOffsetA+1], s[sgprStrideAL], 64 // compute offset diff (scaled unrollDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)
s_mul_i32 s[sgprScalarGlobalReadOffsetA+2], s[sgprStrideAL], 96 // compute offset diff (scaled unrollDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)

/* global read addresses: final offsets b */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 6 // gROB_0_0_0_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s73, s[sgprWorkGroup0], 128           // WorkGroup[01] * MT
s_mul_i32 s72, s[sgprWorkGroup0], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s71, 128, s[sgprGSUSumIdx]            // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_mul_i32 s70, 128, s[sgprGSUSumIdx]               // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_mul_hi_u32 s71, s70, s[sgprStrideAL]             // tlu=1, scaled unroll-offset by stride
s_mul_i32 s70, s70, s[sgprStrideAL]                // tlu=1, scaled unroll-offset by stride
s_add_u32 s72, s72, s70                            // accum GsuOffset term to tilestart
s_addc_u32 s73, s73, s71                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitA+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitA+1], 0                 // init tensor size
s_sub_u32 s70, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s71, constStrideA0I, s70              // stride x (size-1)
s_mul_i32 s70, constStrideA0I, s70                 // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // sum tensor size
s_sub_u32 s70, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s71, s[sgprStrideAL], s70             // stride x (size-1)
s_mul_i32 s70, s[sgprStrideAL], s70                // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // sub tileStart
                                                   // Set limit to use bytes (byte is 1, do nothing)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s71, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s70, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s72, s72, s70                            // accum wg term to tilestart
s_addc_u32 s73, s73, s71                           // accum wg term to tilestart
                                                   // tileStart *= BPE (multiplier is 1, do nothing)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s72    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s73   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s73, s[sgprWorkGroup1], 16            // WorkGroup[01] * MT
s_mul_i32 s72, s[sgprWorkGroup1], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s73, s72, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s72, s72, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_hi_u32 s71, 128, s[sgprGSUSumIdx]            // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_mul_i32 s70, 128, s[sgprGSUSumIdx]               // gsuOffset = DepthU*bpeGR*GSUSumIdx
s_add_u32 s72, s72, s70                            // accum GsuOffset term to tilestart
s_addc_u32 s73, s73, s71                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitB+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitB+1], 0                 // init tensor size
s_sub_u32 s70, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s71, constStrideBL, s70               // stride x (size-1)
s_mul_i32 s70, constStrideBL, s70                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // sum tensor size
s_sub_u32 s70, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s71, s[sgprStrideB1J], s70            // stride x (size-1)
s_mul_i32 s70, s[sgprStrideB1J], s70               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x1 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s71, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s70, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s72, s72, s70                            // accum wg term to tilestart
s_addc_u32 s73, s73, s71                           // accum wg term to tilestart
s_lshl_b64 s[72:73], s[72:73], 0x1                 // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s72    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s73   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD
s_mul_i32 s70, s[sgprGSU], DepthU*BpeAGR
s_mul_i32 s[sgprGlobalReadIncsA+0], s70, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_mul_i32 s70, s[sgprGSU], DepthU*BpeBGR
s_mov_b32 s[sgprGlobalReadIncsB+0], s70            // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_loadwave_GSU_1                         // branch if GSU == 1
v_cvt_f32_u32 v0, s[sgprGSU]                       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mul_u32_u24 v1, v0, s[sgprGSU]                   // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_sub_u32 v1, s[sgprLoopCounterL], v1              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cmpx_eq_u32 exec, v1, s[sgprGSU]                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_add_u32 v0, 1, v0                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSU]
s_mov_b64 exec, -1                                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_readfirstlane_b32 s[sgprLoopCounterL], v0
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1
s_add_u32 s70, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s70                // numIterMyWg++ if needed
label_loadwave_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s72, s[sgprStaggerU], 0x1f00
s_lshr_b32 s72, s72, 0x8
s_and_b32 s73, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s70, s[sgprStaggerU]                     // init staggerU
label_loadwave_beginStaggerUIter:
s_lshl_b32 s71, s70, s72                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s71           // loopCount >= current shift Count
s_cbranch_scc1 label_loadwave_endStaggerUIter               // jump to end
s_lshr_b32 s70, s70, 1                             // step down to smaller stagger
s_branch label_loadwave_beginStaggerUIter                   // jump to begin
label_loadwave_endStaggerUIter:
s_sub_u32 s71, s70, 1                              // staggerU mask
s_cmp_ge_u32 s70, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s71, 0          // set Mask
s_cmp_eq_u32 s73, 0x0
s_cbranch_scc1 label_loadwave_StaggerUMapping_1
s_mov_b32 s70, s[sgprWorkGroup0]
s_branch label_loadwave_staggerInputEnd
label_loadwave_StaggerUMapping_1:
s_cmp_eq_u32 s73, 0x2000
s_cbranch_scc1 label_loadwave_StaggerUMapping_2
s_mov_b32 s70, s[sgprWorkGroup1]
s_branch label_loadwave_staggerInputEnd
label_loadwave_StaggerUMapping_2:
s_cmp_eq_u32 s73, 0x4000
s_cbranch_scc1 label_loadwave_StaggerUMapping_3
s_mov_b32 s70, -0x1
s_branch label_loadwave_staggerInputEnd
label_loadwave_StaggerUMapping_3:
s_cmp_eq_u32 s73, 0x6000
s_cbranch_scc1 label_loadwave_StaggerUMapping_4
s_mul_i32 s71, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s70, s70, s71
s_add_u32 s70, s70, s[sgprWorkGroup0]
s_branch label_loadwave_staggerInputEnd
label_loadwave_StaggerUMapping_4:
s_cmp_eq_u32 s73, 0x8000
s_cbranch_scc1 label_loadwave_staggerInputEnd
s_mov_b32 s70, -0x1
s_branch label_loadwave_staggerInputEnd
label_loadwave_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s70 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s72 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s71, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s70, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s71, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s70, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_loadwave_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0

// 1st set GR 
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s72, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s72              // Is this wrapIter? (pf)
s_cselect_b32 s70, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s72, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s72              // Is this wrapIter? (pf)
s_cselect_b32 s70, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32


// TODO: fixed K=DU case 
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_loadwave_skipPGR2_0                    // PGR=2 but only 1 loop
// 2nd set of GR 
buffer_load_dwordx4 v[vgprG2LA1+0:vgprG2LA1+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA1+4:vgprG2LA1+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA1+8:vgprG2LA1+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA1+12:vgprG2LA1+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB1+0:vgprG2LB1+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
label_loadwave_skipPGR2_0:





/******************************************/
/* End setupNewTile                       */
/******************************************/
label_loadwave_ShadowInitStart:


s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_loadwave_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
// Todo: remove it?
s_mul_hi_u32 s71, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s70, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s74, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s74, s74, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s73, s74, s[sgprStrideC1J]            // Free1
s_mul_i32 s72, s74, s[sgprStrideC1J]               // Free1
s_add_u32 s70, s70, s72                            // Free1
s_addc_u32 s71, s71, s73                           // Free1
s_sub_u32 s74, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s74, s74, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s73, s74, s[sgprStrideCK]             // Free2
s_mul_i32 s72, s74, s[sgprStrideCK]                // Free2
s_add_u32 s70, s70, s72                            // Free2
s_addc_u32 s71, s71, s73                           // Free2
s_lshl_b64 s[70:71], s[70:71], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi GSU offset to SRD
label_loadwave_GSU_2:
//.set sgprGSULog2BpeC, UNDEF

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...8) from pool */

/* initC: remove ValuA/B vgpr buffer [0...54) from pool */

// TODO: need to check it later 
//s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
//s_cbranch_scc0 label_loadwave_NoBranch_PDGSA4QTO45DNN5O_0   // Only branch on scc1
//s_getpc_b64 s[70:71]                               // addr of next instr
//s_add_i32 s72, label_loadwave_PrefetchGlobalLastIterEnd, 0x4 // target branch offset
//s_add_u32 s70, s70, s72                            // add target branch offset
//s_addc_u32 s71, s71, 0                             // add high and carry
//s_setpc_b64 s[70:71]                               // branch to label_loadwave_PrefetchGlobalLastIterEnd
//label_loadwave_NoBranch_PDGSA4QTO45DNN5O_0:


// wait until 1st set pf GR is done  
s_waitcnt vmcnt(5)                                 // 8wait for global read

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4224 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4224
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8448 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8448
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12672 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12672

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write

// Signal math-wave LW is done 
s_barrier


/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_loadwave_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_loadwave_toPGR1_0                      // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_loadwave_LoopEndL_evenexit             // do not enter LoopL
label_loadwave_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

/* iter 0 */
/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s70, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?

/* iter 1 */
s_cselect_b32 s70, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32


buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA+8:vgprG2LA+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA+12:vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0



s_waitcnt vmcnt(9)                                

// Wait for LR to done in math-wave 
s_barrier

ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+0:vgprG2LA1+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt vmcnt(8)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+4:vgprG2LA1+4+3] offset:4224 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4224
s_waitcnt vmcnt(7)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+8:vgprG2LA1+8+3] offset:8448 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8448
s_waitcnt vmcnt(6)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+12:vgprG2LA1+12+3] offset:12672 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12672
s_waitcnt vmcnt(5)                                 
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB1+0:vgprG2LB1+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

s_waitcnt lgkmcnt(0)                               

// signal math wave that LW is done 
s_barrier

/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc1 label_loadwave_LoopEndL_oddexit              // exit LoopL

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

/* iter 0 */
/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s70, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s70, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s71       // gra SRD += inc(upper)

s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32


buffer_load_dwordx4 v[vgprG2LA1+0:vgprG2LA1+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA1+4:vgprG2LA1+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LA1+8:vgprG2LA1+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LA1+12:vgprG2LA1+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB1+0:vgprG2LB1+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0


s_waitcnt vmcnt(9)    

// wait until LR is done in math 
s_barrier

ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt vmcnt(8)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4224 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4224
s_waitcnt vmcnt(7)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8448 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8448
s_waitcnt vmcnt(6)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12672 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12672
s_waitcnt vmcnt(5)                                 
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

s_waitcnt lgkmcnt(0)                               

// signal math wave that LW is done 
s_barrier


/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_loadwave_LoopBeginL                    // restart LoopL
label_loadwave_LoopEndL_evenexit:  /// unroll loop eveniter exit
label_loadwave_LoopEndL_oddexit:  /// unroll loop odditer exit

/* Select high bank of LDS */
label_loadwave_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop - Begin          */
/******************************************/

/* iter 0 */

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s70, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s70, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s71, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32


// wait until LR is done 
s_barrier

s_waitcnt vmcnt(4)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+0:vgprG2LA1+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
s_waitcnt vmcnt(3)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+4:vgprG2LA1+4+3] offset:4224 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4224
s_waitcnt vmcnt(2)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+8:vgprG2LA1+8+3] offset:8448 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8448
s_waitcnt vmcnt(1)                                 
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA1+12:vgprG2LA1+12+3] offset:12672 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12672
s_waitcnt vmcnt(0)                                 
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB1+0:vgprG2LB1+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

s_waitcnt lgkmcnt(0)    

// signal math wave that LW is done 
s_barrier

label_loadwave_toPGR1_0:
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc0 label_loadwave_GSU_3                         // branch if GSU != 1
label_loadwave_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */


s_barrier


/******************************************/
/* Tail Loop                              */
/******************************************/

// FIXME: Tail Loop won't work 

label_loadwave_SkipTailLoopL:

label_loadwave_KernelEnd:
s_endpgm                                           // Kernel End


/*****************************************************/
/* MATH WAVE START                                   */
/*****************************************************/

math_wave_start:

/* change the wave id to match 4-waves wave-id  */

s_mov_b32 s24, -256
v_add_i32 v0, s24, v0

label_debug_math_wave: 


/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_load_dword s46, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_mov_b32 s[sgprPackKForV0], 0x05040100
s_mov_b32 s[sgprPackKForV1], 0x07060302
s_mov_b32 m0, 0x5400                               // LDS clamp at 21504 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
/* init: add vgpr [0...54) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...8) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v0, 0x1, v0                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_and_b32 v1, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0xa, v1                          // 5. K offset: lrKOffset = kIdx * mStride(1024)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v1, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v1, 3, v1                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshlrev_b32 v1, 0x5, v1                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)
v_add_u32 v0, v1, v0                               // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x7, v1                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v2, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x3, v2                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s70, 16384                               // LSU offset: stride = lsuStride(128)*(MT0(128) + PAD0(0))
v_mul_lo_u32 v2, s70, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v2, v0            // Final Offset: offset = (lro0+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v3, 10, v[vgprLocalReadAddrA]        // Final Offset: padding 32 per block 1024
v_lshlrev_b32 v3, 0x5, v3                          // Final Offset: padding 32 per block 1024
v_add_u32 v[vgprLocalReadAddrA], v3, v[vgprLocalReadAddrA] // Final Offset: add padding 32 per block 1024

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s70, 128                                 // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v0, s70, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v2, 0x5, v2                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrB], v2, v[vgprLocalReadAddrB] // Final Offset: add padding 32 per block 256

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 8 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 3, v[vgprSerial]                 // v1 = v[vgprSerial] / 8
v_and_b32 v0, 7, v[vgprSerial]                     // v0 = v[vgprSerial] % 8
/* tile *= glvw */
v_lshlrev_b32 v0, 0x4, v0                          // v0 = v0 * 16
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // v2 = v[vgprSerial] / 16
v_and_b32 v3, 15, v[vgprSerial]                    // v3 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x80, v4     // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS(1)
v_lshrrev_b32 v6, 10, v[vgprLocalWriteAddrA]       // padding 32 per block 1024
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 1024
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // add padding 32 per block 1024

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v2     // lwBL**(DepthU_Compute + PAD)
v_add_lshl_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB], 0x1 // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrB], v6, v[vgprLocalWriteAddrB] // add padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x4200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=16896
s_waitcnt lgkmcnt(0)                               // wait for 92 bytes of kern args
s_and_b32 s[sgprWGM], s[sgprGSU], 0xff00
s_lshr_b32 s[sgprWGM], s[sgprWGM], 0x8
s_and_b32 s[sgprStaggerU], s[sgprGSU], 0xffff0000
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s[sgprGSU], 0xff
label_stop:
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
v_cmp_eq_f32 vcc, s[sgprAlpha], 0.0                // s[Alpha] == 0.0f ?
s_cbranch_vccz label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
v_cvt_f32_u32 v6, s[sgprGSU]                       // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mul_u32_u24 v7, v6, s[sgprGSU]                   // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_sub_u32 v7, s[sgprWorkGroup1], v7                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_cmpx_eq_u32 exec, v7, s[sgprGSU]                 // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s[sgprGSU]
s_mov_b64 exec, -1                                 // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s[sgprGSU]
v_readfirstlane_b32 s[sgprWorkGroup1], v6
v_readfirstlane_b32 s[sgprGSUSumIdx], v7
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 1
s_mov_b32 s[sgprGSULog2BpeD], 1
label_GSU_End:
s_cmp_le_u32 s[sgprWGM], 1                         // WGM <= 1 ?
s_cbranch_scc1 label_WGM                           // branch if WGM <= 1
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprWorkGroup1], v7                // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s72, v6
s_mul_i32 s73, s72, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s73, s[sgprWorkGroup1], s73              // WorkGroup1=remainder
s_mul_i32 s73, s73, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s73, s73, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups1]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprNumWorkGroups1], v7            // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s70, v6
s_mul_i32 s71, s[sgprWGM], s70                     // quotient * non-magic divisor
s_sub_u32 s74, s[sgprNumWorkGroups1], s71          // WorkGroup1=remainder
s_cmp_eq_u32 s74, 0                                // remainder == 0 ?
s_cmov_b32 s74, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s72, s70                              // blockId >= numFullBlocks ?
s_cselect_b32 s70, s74, s[sgprWGM]
v_cvt_f32_u32 v6, s70                              // s[sgprWorkGroup0] = s73 / s70
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup0] = s73 / s70
v_cvt_f32_u32 v7, s73                              // s[sgprWorkGroup0] = s73 / s70
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup0] = s73 / s70
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup0] = s73 / s70
v_mul_u32_u24 v7, v6, s70                          // s[sgprWorkGroup0] = s73 / s70
v_sub_u32 v7, s73, v7                              // s[sgprWorkGroup0] = s73 / s70
v_cmpx_eq_u32 exec, v7, s70                        // s[sgprWorkGroup0] = s73 / s70
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup0] = s73 / s70
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s73 % s70
s_mov_b64 exec, -1                                 // s[sgprWorkGroup0] = s73 / s70
v_readfirstlane_b32 s[sgprWorkGroup0], v6
v_readfirstlane_b32 s[sgprWorkGroup1], v7
s_mul_i32 s72, s72, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s72 // wg1 += blockId * WGM
label_WGM:

/* global read addresses: addresses a */
/* global read addresses: addresses b */

/* global read addresses: increments b */
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 7 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 128
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
v_cvt_f32_u32 v0, s[sgprGSU]                       // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mul_u32_u24 v1, v0, s[sgprGSU]                   // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_sub_u32 v1, s[sgprLoopCounterL], v1              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_cmpx_eq_u32 exec, v1, s[sgprGSU]                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_add_u32 v0, 1, v0                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSU]
s_mov_b64 exec, -1                                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSU]
v_readfirstlane_b32 s[sgprLoopCounterL], v0
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1
s_add_u32 s70, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s70                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s72, s[sgprStaggerU], 0x1f00
s_lshr_b32 s72, s72, 0x8
s_and_b32 s73, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s70, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s71, s70, s72                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s71           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s70, s70, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s71, s70, 1                              // staggerU mask
s_cmp_ge_u32 s70, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s71, 0          // set Mask
s_cmp_eq_u32 s73, 0x0
s_cbranch_scc1 label_StaggerUMapping_1
s_mov_b32 s70, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s73, 0x2000
s_cbranch_scc1 label_StaggerUMapping_2
s_mov_b32 s70, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s73, 0x4000
s_cbranch_scc1 label_StaggerUMapping_3
s_mov_b32 s70, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s73, 0x6000
s_cbranch_scc1 label_StaggerUMapping_4
s_mul_i32 s71, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s70, s70, s71
s_add_u32 s70, s70, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s73, 0x8000
s_cbranch_scc1 label_staggerInputEnd
s_mov_b32 s70, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s70 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s72 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0

/* global read inc A loopL */

/* global read inc B loopL */

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0x80000000
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0x80000000
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s72, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s71, s72, s[sgprStrideC1J]            // ScaleC s72 by Stride
s_mul_i32 s70, s72, s[sgprStrideC1J]               // ScaleC s72 by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s70    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s71   // add hi to SRD
s_mul_hi_u32 s71, s72, s[sgprStrideD1J]            // ScaleD s72 by Stride
s_mul_i32 s70, s72, s[sgprStrideD1J]               // ScaleD s72 by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s70    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s71   // add hi to SRD

s_mul_hi_u32 s71, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s70, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s71       // add hi to SRD
s_mul_hi_u32 s71, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s70, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[70:71], s[70:71], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi to SRD

s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s71, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s70, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s74, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s74, s74, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s73, s74, s[sgprStrideC1J]            // Free1
s_mul_i32 s72, s74, s[sgprStrideC1J]               // Free1
s_add_u32 s70, s70, s72                            // Free1
s_addc_u32 s71, s71, s73                           // Free1
s_sub_u32 s74, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s74, s74, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s73, s74, s[sgprStrideCK]             // Free2
s_mul_i32 s72, s74, s[sgprStrideCK]                // Free2
s_add_u32 s70, s70, s72                            // Free2
s_addc_u32 s71, s71, s73                           // Free2
s_lshl_b64 s[70:71], s[70:71], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s70        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s71       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...8) from pool */

/* initC: remove ValuA/B vgpr buffer [0...54) from pool */
v_accvgpr_write acc0, 0x0                          // initC
v_accvgpr_write acc1, 0x0                          // initC
v_accvgpr_write acc2, 0x0                          // initC
v_accvgpr_write acc3, 0x0                          // initC
v_accvgpr_write acc4, 0x0                          // initC
v_accvgpr_write acc5, 0x0                          // initC
v_accvgpr_write acc6, 0x0                          // initC
v_accvgpr_write acc7, 0x0                          // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_PDGSA4QTO45DNN5O_0   // Only branch on scc1
s_getpc_b64 s[70:71]                               // addr of next instr
s_add_i32 s72, label_PrefetchGlobalLastIterEnd, 0x4 // target branch offset
s_add_u32 s70, s70, s72                            // add target branch offset
s_addc_u32 s71, s71, 0                             // add high and carry
s_setpc_b64 s[70:71]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_PDGSA4QTO45DNN5O_0:



// wait for LW to be done in loadwave 
s_barrier

/* local read prefetch a */
ds_read_u16 v[vgprValuA_X0_I0_D0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read prefetch a */
ds_read_u16 v[vgprValuA_X1_I0_D0+0], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D2+0], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D3+0], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0


/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_toPGR1_0                      // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL_evenexit             // do not enter LoopL
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/2 - Begin              */
/******************************************/

/* iter 0 */
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=5
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_u16 v[vgprValuA_X2_I0_D0+0], v[vgprLocalReadAddrA] offset:4224 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D1+0], v[vgprLocalReadAddrA] offset:4352 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D2+0], v[vgprLocalReadAddrA] offset:4480 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D3+0], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 1 */
ds_read_u16 v[vgprValuA_X3_I0_D0+0], v[vgprLocalReadAddrA] offset:4736 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D1+0], v[vgprLocalReadAddrA] offset:4864 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D2+0], v[vgprLocalReadAddrA] offset:4992 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0
//s_cselect_b32 s70, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_waitcnt lgkmcnt(7)                               // wait for prior local read local write old=0, new=7 newLW=0 newLR=7
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_u16 v[vgprValuA_X3_I0_D3+0], v[vgprLocalReadAddrA] offset:5120 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 2 */
ds_read_u16 v[vgprValuA_X4_I0_D0+0], v[vgprLocalReadAddrA] offset:8448 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D1+0], v[vgprLocalReadAddrA] offset:8576 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0

s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X2_I0+0], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+1], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+2], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+3], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
ds_read_u16 v[vgprValuA_X4_I0_D2+0], v[vgprLocalReadAddrA] offset:8704 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D3+0], v[vgprLocalReadAddrA] offset:8832 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=4 iui=0

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 3 */
ds_read_u16 v[vgprValuA_X5_I0_D0+0], v[vgprLocalReadAddrA] offset:8960 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D1+0], v[vgprLocalReadAddrA] offset:9088 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X3_I0+0], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+1], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+2], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+3], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_u16 v[vgprValuA_X5_I0_D2+0], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D3+0], v[vgprLocalReadAddrA] offset:9344 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=5 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 4 (swap and reset local write pointers iteration)  */
ds_read_u16 v[vgprValuA_X6_I0_D0+0], v[vgprLocalReadAddrA] offset:12672 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D1+0], v[vgprLocalReadAddrA] offset:12800 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D2+0], v[vgprLocalReadAddrA] offset:12928 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D3+0], v[vgprLocalReadAddrA] offset:13056 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D0+0], v[vgprLocalReadAddrA] offset:13184 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D1+0], v[vgprLocalReadAddrA] offset:13312 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D2+0], v[vgprLocalReadAddrA] offset:13440 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D3+0], v[vgprLocalReadAddrA] offset:13568 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=7 iui=0

s_waitcnt lgkmcnt(0)


// Signal load wave that LR is done 

s_barrier

v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X4_I0+0], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+1], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+2], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X4_I0+3], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+2+0+0:vgprValuA_X4_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X5_I0+0], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+1], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+2], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X5_I0+3], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+0+0+0:vgprValuA_X5_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+2+0+0:vgprValuA_X5_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 6 */

v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X6_I0+0], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+1], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+2], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X6_I0+3], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */


// wait until LW is done
s_barrier

ds_read_u16 v[vgprValuA_X0_I0_D0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+2+0+0:vgprValuA_X6_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 7 */
ds_read_u16 v[vgprValuA_X1_I0_D0+0], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D2+0], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0

// check??
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8

/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X7_I0+0], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+1], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+2], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X7_I0+3], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+0+0+0:vgprValuA_X7_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_u16 v[vgprValuA_X1_I0_D3+0], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+2+0+0:vgprValuA_X7_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/******************************************/
/* Unrolled Loop - End 1/2                */
/******************************************/

/* closeLoop loopL finalLoop=0 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc1 label_LoopEndL_oddexit              // exit LoopL

/******************************************/
/* Unrolled Loop 2/2 - Begin              */
/******************************************/

/* iter 0 */
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=5
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_u16 v[vgprValuA_X2_I0_D0+0], v[vgprLocalReadAddrA] offset:4224 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D1+0], v[vgprLocalReadAddrA] offset:4352 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D2+0], v[vgprLocalReadAddrA] offset:4480 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D3+0], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 1 */
ds_read_u16 v[vgprValuA_X3_I0_D0+0], v[vgprLocalReadAddrA] offset:4736 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D1+0], v[vgprLocalReadAddrA] offset:4864 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D2+0], v[vgprLocalReadAddrA] offset:4992 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0

s_waitcnt lgkmcnt(7)                               // wait for prior local read local write old=0, new=7 newLW=0 newLR=7
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_u16 v[vgprValuA_X3_I0_D3+0], v[vgprLocalReadAddrA] offset:5120 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 2 */
ds_read_u16 v[vgprValuA_X4_I0_D0+0], v[vgprLocalReadAddrA] offset:8448 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D1+0], v[vgprLocalReadAddrA] offset:8576 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X2_I0+0], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+1], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+2], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+3], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
ds_read_u16 v[vgprValuA_X4_I0_D2+0], v[vgprLocalReadAddrA] offset:8704 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D3+0], v[vgprLocalReadAddrA] offset:8832 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=4 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 3 */
ds_read_u16 v[vgprValuA_X5_I0_D0+0], v[vgprLocalReadAddrA] offset:8960 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D1+0], v[vgprLocalReadAddrA] offset:9088 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X3_I0+0], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+1], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+2], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+3], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_u16 v[vgprValuA_X5_I0_D2+0], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D3+0], v[vgprLocalReadAddrA] offset:9344 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=5 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 4 (swap and reset local write pointers iteration)  */
ds_read_u16 v[vgprValuA_X6_I0_D0+0], v[vgprLocalReadAddrA] offset:12672 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D1+0], v[vgprLocalReadAddrA] offset:12800 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D2+0], v[vgprLocalReadAddrA] offset:12928 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D3+0], v[vgprLocalReadAddrA] offset:13056 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D0+0], v[vgprLocalReadAddrA] offset:13184 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D1+0], v[vgprLocalReadAddrA] offset:13312 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D2+0], v[vgprLocalReadAddrA] offset:13440 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D3+0], v[vgprLocalReadAddrA] offset:13568 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=7 iui=0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)


// Signal load wave that LR is done 

s_barrier

/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X4_I0+0], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+1], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+2], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X4_I0+3], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+2+0+0:vgprValuA_X4_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X5_I0+0], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+1], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+2], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X5_I0+3], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+0+0+0:vgprValuA_X5_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+2+0+0:vgprValuA_X5_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 6 */


/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X6_I0+0], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+1], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+2], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X6_I0+3], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */


// wait for LW to be done 
s_barrier


ds_read_u16 v[vgprValuA_X0_I0_D0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+2+0+0:vgprValuA_X6_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 7 */
ds_read_u16 v[vgprValuA_X1_I0_D0+0], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D2+0], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0

// need to check 
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X7_I0+0], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+1], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+2], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X7_I0+3], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+0+0+0:vgprValuA_X7_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_u16 v[vgprValuA_X1_I0_D3+0], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+2+0+0:vgprValuA_X7_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/******************************************/
/* Unrolled Loop - End 2/2 (final)        */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL_evenexit:  /// unroll loop eveniter exit
label_LoopEndL_oddexit:  /// unroll loop odditer exit

/* Select high bank of LDS */
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop - Begin          */
/******************************************/

/* iter 0 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=5
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_u16 v[vgprValuA_X2_I0_D0+0], v[vgprLocalReadAddrA] offset:4224 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D1+0], v[vgprLocalReadAddrA] offset:4352 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D2+0], v[vgprLocalReadAddrA] offset:4480 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D3+0], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 1 */
ds_read_u16 v[vgprValuA_X3_I0_D0+0], v[vgprLocalReadAddrA] offset:4736 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D1+0], v[vgprLocalReadAddrA] offset:4864 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D2+0], v[vgprLocalReadAddrA] offset:4992 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0
s_waitcnt lgkmcnt(7)                               // wait for prior local read local write old=0, new=7 newLW=0 newLR=7
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_u16 v[vgprValuA_X3_I0_D3+0], v[vgprLocalReadAddrA] offset:5120 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 2 */
/*  mfmaIndex:4  */
ds_read_u16 v[vgprValuA_X4_I0_D0+0], v[vgprLocalReadAddrA] offset:8448 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D1+0], v[vgprLocalReadAddrA] offset:8576 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X2_I0+0], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+1], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+2], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+3], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
ds_read_u16 v[vgprValuA_X4_I0_D2+0], v[vgprLocalReadAddrA] offset:8704 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D3+0], v[vgprLocalReadAddrA] offset:8832 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=4 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 3 */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:6  */
ds_read_u16 v[vgprValuA_X5_I0_D0+0], v[vgprLocalReadAddrA] offset:8960 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D1+0], v[vgprLocalReadAddrA] offset:9088 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X3_I0+0], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+1], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+2], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+3], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_u16 v[vgprValuA_X5_I0_D2+0], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D3+0], v[vgprLocalReadAddrA] offset:9344 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=5 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 4 (swap and reset local write pointers iteration)  */
ds_read_u16 v[vgprValuA_X6_I0_D0+0], v[vgprLocalReadAddrA] offset:12672 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D1+0], v[vgprLocalReadAddrA] offset:12800 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D2+0], v[vgprLocalReadAddrA] offset:12928 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D3+0], v[vgprLocalReadAddrA] offset:13056 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D0+0], v[vgprLocalReadAddrA] offset:13184 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D1+0], v[vgprLocalReadAddrA] offset:13312 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D2+0], v[vgprLocalReadAddrA] offset:13440 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D3+0], v[vgprLocalReadAddrA] offset:13568 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=7 iui=0

// need to move some cvt here 
// calc cycles

// signal load wave that LR is done 
s_waitcnt lgkmcnt(0)
s_barrier


/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X4_I0+0], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+1], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+2], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X4_I0+3], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */

/* local write swap offsets b */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+2+0+0:vgprValuA_X4_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 5 (reset local read pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:9, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:10  */
s_waitcnt lgkmcnt(13)                              // wait for prior local read local write old=4, new=13 newLW=5 newLR=4
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X5_I0+0], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+1], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+2], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X5_I0+3], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+0+0+0:vgprValuA_X5_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */

v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+2+0+0:vgprValuA_X5_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 6 */

// wait for lW to be done in load wave 
s_barrier

v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X6_I0+0], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+1], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+2], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X6_I0+3], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
ds_read_u16 v[vgprValuA_X0_I0_D0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+2+0+0:vgprValuA_X6_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 7 */
ds_read_u16 v[vgprValuA_X1_I0_D0+0], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D1+0], v[vgprLocalReadAddrA] offset:640 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=1 iui=0
ds_read_u16 v[vgprValuA_X1_I0_D2+0], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=1 iui=0
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X7_I0+0], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+1], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+2], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X7_I0+3], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+0+0+0:vgprValuA_X7_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
ds_read_u16 v[vgprValuA_X1_I0_D3+0], v[vgprLocalReadAddrA] offset:896 // L -> Reg lro=512 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=1 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=32 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+2+0+0:vgprValuA_X7_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
label_toPGR1_0:
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:9, lwEndMfmaIndex:9  */
/*  numMfmaForLR:1, syncPlrMfmaIndex:12  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(5)                               // wait for prior local read local write old=0, new=5 newLW=0 newLR=5
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_u16 v[vgprValuA_X2_I0_D0+0], v[vgprLocalReadAddrA] offset:4224 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D1+0], v[vgprLocalReadAddrA] offset:4352 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D2+0], v[vgprLocalReadAddrA] offset:4480 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=2 iui=0
ds_read_u16 v[vgprValuA_X2_I0_D3+0], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=4096 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=64 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 1 (last unrolled loop) */
ds_read_u16 v[vgprValuA_X3_I0_D0+0], v[vgprLocalReadAddrA] offset:4736 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D1+0], v[vgprLocalReadAddrA] offset:4864 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=3 iui=0
ds_read_u16 v[vgprValuA_X3_I0_D2+0], v[vgprLocalReadAddrA] offset:4992 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=3 iui=0
s_waitcnt lgkmcnt(7)                               // wait for prior local read local write old=0, new=7 newLW=0 newLR=7
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X1_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X1_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X1_I0+0], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+1], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X1_I0+2], v[vgprValuA_X1_I0_D1+0], v[vgprValuA_X1_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X1_I0+3], v[vgprValuA_X1_I0_D3+0], v[vgprValuA_X1_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+0+0+0:vgprValuA_X1_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:3  */
ds_read_u16 v[vgprValuA_X3_I0_D3+0], v[vgprLocalReadAddrA] offset:5120 // L -> Reg lro=4608 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=3 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=96 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X1_I0+2+0+0:vgprValuA_X1_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 2 (last unrolled loop) */
ds_read_u16 v[vgprValuA_X4_I0_D0+0], v[vgprLocalReadAddrA] offset:8448 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D1+0], v[vgprLocalReadAddrA] offset:8576 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=4 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X2_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X2_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X2_I0+0], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+1], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X2_I0+2], v[vgprValuA_X2_I0_D1+0], v[vgprValuA_X2_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X2_I0+3], v[vgprValuA_X2_I0_D3+0], v[vgprValuA_X2_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:5  */
ds_read_u16 v[vgprValuA_X4_I0_D2+0], v[vgprLocalReadAddrA] offset:8704 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=4 iui=0
ds_read_u16 v[vgprValuA_X4_I0_D3+0], v[vgprLocalReadAddrA] offset:8832 // L -> Reg lro=8192 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=4 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+2+0+0:vgprValuA_X2_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 3 (last unrolled loop) */
/*  mfmaIndex:6  */
ds_read_u16 v[vgprValuA_X5_I0_D0+0], v[vgprLocalReadAddrA] offset:8960 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D1+0], v[vgprLocalReadAddrA] offset:9088 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=5 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X3_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X3_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X3_I0+0], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+1], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X3_I0+2], v[vgprValuA_X3_I0_D1+0], v[vgprValuA_X3_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X3_I0+3], v[vgprValuA_X3_I0_D3+0], v[vgprValuA_X3_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+0+0+0:vgprValuA_X3_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_u16 v[vgprValuA_X5_I0_D2+0], v[vgprLocalReadAddrA] offset:9216 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=5 iui=0
ds_read_u16 v[vgprValuA_X5_I0_D3+0], v[vgprLocalReadAddrA] offset:9344 // L -> Reg lro=8704 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=5 iui=0
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X3_I0+2+0+0:vgprValuA_X3_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 4 (last unrolled loop) */
ds_read_u16 v[vgprValuA_X6_I0_D0+0], v[vgprLocalReadAddrA] offset:12672 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D1+0], v[vgprLocalReadAddrA] offset:12800 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D2+0], v[vgprLocalReadAddrA] offset:12928 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X6_I0_D3+0], v[vgprLocalReadAddrA] offset:13056 // L -> Reg lro=12288 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=6 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D0+0], v[vgprLocalReadAddrA] offset:13184 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D1+0], v[vgprLocalReadAddrA] offset:13312 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D2+0], v[vgprLocalReadAddrA] offset:13440 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=7 iui=0
ds_read_u16 v[vgprValuA_X7_I0_D3+0], v[vgprLocalReadAddrA] offset:13568 // L -> Reg lro=12800 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=7 iui=0
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
s_waitcnt lgkmcnt(12)                              // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X4_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X4_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X4_I0+0], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+1], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X4_I0+2], v[vgprValuA_X4_I0_D1+0], v[vgprValuA_X4_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X4_I0+3], v[vgprValuA_X4_I0_D3+0], v[vgprValuA_X4_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:9  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+2+0+0:vgprValuA_X4_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 5 (last unrolled loop) */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=4, new=8 newLW=0 newLR=4
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X5_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X5_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X5_I0+0], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+1], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X5_I0+2], v[vgprValuA_X5_I0_D1+0], v[vgprValuA_X5_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X5_I0+3], v[vgprValuA_X5_I0_D3+0], v[vgprValuA_X5_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+0+0+0:vgprValuA_X5_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:11  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X5_I0+2+0+0:vgprValuA_X5_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 6 (last unrolled loop) */
/*  mfmaIndex:12  */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X6_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X6_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X6_I0+0], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+1], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X6_I0+2], v[vgprValuA_X6_I0_D1+0], v[vgprValuA_X6_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X6_I0+3], v[vgprValuA_X6_I0_D3+0], v[vgprValuA_X6_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:13  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+2+0+0:vgprValuA_X6_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* iter 7 (last unrolled loop) */
/*  mfmaIndex:14  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:16, packBIdx:0 */
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X7_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X7_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X7_I0+0], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+1], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X7_I0+2], v[vgprValuA_X7_I0_D1+0], v[vgprValuA_X7_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X7_I0+3], v[vgprValuA_X7_I0_D3+0], v[vgprValuA_X7_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+0+0+0:vgprValuA_X7_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/*  mfmaIndex:15  */
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X7_I0+2+0+0:vgprValuA_X7_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]
label_PrefetchGlobalLastIterEnd:

/******************************************/
/* Tail Loop                              */
/******************************************/

// TODO: assuming no tail loop for now 

/* Tail: add ValuA/B vgpr buffer [0...54) to pool */

/* local write reset offsets a */

/* local write reset offsets b */

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 127, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 128
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0x0                // numIter=0 if gsuSimIdx!=remainder
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s70, 3, s[sgprStaggerUIter]
s_mul_hi_i32 s71, s70, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_sub_u32 s70, s70, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s71, s71, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s70, 3, s[sgprStaggerUIter]
s_mul_hi_i32 s71, s70, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s70, s70, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_sub_u32 s70, s70, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s71, s71, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s70        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s71       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s70 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s71 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* global read A */
/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v0, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:1 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v1, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v2, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:3 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v4, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:5 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v5, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v6, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:7 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v8, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:9 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v9, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v10, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:11 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:13 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v0, 0x8, v0                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v0      // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v1      // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v2, 0x8, v2                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v2      // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v4, 0x8, v4                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v4      // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v5      // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v6, 0x8, v6                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v6      // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v8, 0x8, v8                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v8      // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v9      // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v10, 0x8, v10                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v10     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v14     // pack a sub 8-bit with dest
/* g2l=4, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_ubyte_d16 v0, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:1 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_ubyte_d16_hi v1, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:2 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_ubyte_d16_hi v2, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:3 // load one buffer value
/* g2l=4, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:4 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_ubyte_d16 v4, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:5 // load one buffer value
/* g2l=4, load component 6 */
buffer_load_ubyte_d16_hi v5, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:6 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_ubyte_d16_hi v6, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:7 // load one buffer value
/* g2l=4, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+4+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:8 // load one buffer value
/* g2l=4, load component 9 */
buffer_load_ubyte_d16 v8, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:9 // load one buffer value
/* g2l=4, load component 10 */
buffer_load_ubyte_d16_hi v9, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:10 // load one buffer value
/* g2l=4, load component 11 */
buffer_load_ubyte_d16_hi v10, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:11 // load one buffer value
/* g2l=4, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:12 // load one buffer value
/* g2l=4, load component 13 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:13 // load one buffer value
/* g2l=4, load component 14 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:14 // load one buffer value
/* g2l=4, load component 15 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v0, 0x8, v0                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v0      // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v1      // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v2, 0x8, v2                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v2      // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v4, 0x8, v4                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v4      // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v5      // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v6, 0x8, v6                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v6      // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v8, 0x8, v8                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v8      // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v9      // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v10, 0x8, v10                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v10     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v14     // pack a sub 8-bit with dest
/* g2l=8, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+8+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:0 // load one buffer value
/* g2l=8, load component 1 */
buffer_load_ubyte_d16 v0, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:1 // load one buffer value
/* g2l=8, load component 2 */
buffer_load_ubyte_d16_hi v1, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:2 // load one buffer value
/* g2l=8, load component 3 */
buffer_load_ubyte_d16_hi v2, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:3 // load one buffer value
/* g2l=8, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+8+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:4 // load one buffer value
/* g2l=8, load component 5 */
buffer_load_ubyte_d16 v4, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:5 // load one buffer value
/* g2l=8, load component 6 */
buffer_load_ubyte_d16_hi v5, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:6 // load one buffer value
/* g2l=8, load component 7 */
buffer_load_ubyte_d16_hi v6, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:7 // load one buffer value
/* g2l=8, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+8+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:8 // load one buffer value
/* g2l=8, load component 9 */
buffer_load_ubyte_d16 v8, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:9 // load one buffer value
/* g2l=8, load component 10 */
buffer_load_ubyte_d16_hi v9, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:10 // load one buffer value
/* g2l=8, load component 11 */
buffer_load_ubyte_d16_hi v10, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:11 // load one buffer value
/* g2l=8, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+8+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:12 // load one buffer value
/* g2l=8, load component 13 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:13 // load one buffer value
/* g2l=8, load component 14 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:14 // load one buffer value
/* g2l=8, load component 15 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v0, 0x8, v0                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+0], v[vgprG2LA+8+0], v0      // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+8+0], v[vgprG2LA+8+0], v1      // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v2, 0x8, v2                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+0], v[vgprG2LA+8+0], v2      // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v4, 0x8, v4                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+1], v[vgprG2LA+8+1], v4      // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+8+1], v[vgprG2LA+8+1], v5      // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v6, 0x8, v6                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+1], v[vgprG2LA+8+1], v6      // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v8, 0x8, v8                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+2], v[vgprG2LA+8+2], v8      // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+8+2], v[vgprG2LA+8+2], v9      // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v10, 0x8, v10                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+2], v[vgprG2LA+8+2], v10     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+3], v[vgprG2LA+8+3], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+8+3], v[vgprG2LA+8+3], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+8+3], v[vgprG2LA+8+3], v14     // pack a sub 8-bit with dest
/* g2l=12, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+12+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:0 // load one buffer value
/* g2l=12, load component 1 */
buffer_load_ubyte_d16 v0, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:1 // load one buffer value
/* g2l=12, load component 2 */
buffer_load_ubyte_d16_hi v1, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:2 // load one buffer value
/* g2l=12, load component 3 */
buffer_load_ubyte_d16_hi v2, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:3 // load one buffer value
/* g2l=12, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+12+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:4 // load one buffer value
/* g2l=12, load component 5 */
buffer_load_ubyte_d16 v4, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:5 // load one buffer value
/* g2l=12, load component 6 */
buffer_load_ubyte_d16_hi v5, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:6 // load one buffer value
/* g2l=12, load component 7 */
buffer_load_ubyte_d16_hi v6, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:7 // load one buffer value
/* g2l=12, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+12+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:8 // load one buffer value
/* g2l=12, load component 9 */
buffer_load_ubyte_d16 v8, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:9 // load one buffer value
/* g2l=12, load component 10 */
buffer_load_ubyte_d16_hi v9, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:10 // load one buffer value
/* g2l=12, load component 11 */
buffer_load_ubyte_d16_hi v10, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:11 // load one buffer value
/* g2l=12, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+12+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:12 // load one buffer value
/* g2l=12, load component 13 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:13 // load one buffer value
/* g2l=12, load component 14 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:14 // load one buffer value
/* g2l=12, load component 15 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v0, 0x8, v0                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+0], v[vgprG2LA+12+0], v0    // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+12+0], v[vgprG2LA+12+0], v1    // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v2, 0x8, v2                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+0], v[vgprG2LA+12+0], v2    // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v4, 0x8, v4                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+1], v[vgprG2LA+12+1], v4    // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+12+1], v[vgprG2LA+12+1], v5    // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v6, 0x8, v6                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+1], v[vgprG2LA+12+1], v6    // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v8, 0x8, v8                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+2], v[vgprG2LA+12+2], v8    // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+12+2], v[vgprG2LA+12+2], v9    // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v10, 0x8, v10                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+2], v[vgprG2LA+12+2], v10   // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+3], v[vgprG2LA+12+3], v12   // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+12+3], v[vgprG2LA+12+3], v13   // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+12+3], v[vgprG2LA+12+3], v14   // pack a sub 8-bit with dest

/* Update M0 for DTLDS */

/* global read B */
/* g2l=0, load component 0 */
buffer_load_short_d16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_short_d16_hi v0, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v0      // HasEccHalf: pack
/* g2l=0, load component 2 */
buffer_load_short_d16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_short_d16_hi v0, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v0      // HasEccHalf: pack
/* g2l=0, load component 4 */
buffer_load_short_d16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_short_d16_hi v0, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:10 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v0      // HasEccHalf: pack
/* g2l=0, load component 6 */
buffer_load_short_d16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_short_d16_hi v0, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:14 // load one buffer value
s_waitcnt vmcnt(0)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v0      // HasEccHalf: pack
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip force waitcnt0
s_barrier

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4224 // lwoA_0_0_1_0 = (0*LSCA) + (1*LSPA)(*MT0I+PAD) = 4224
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+8:vgprG2LA+8+3] offset:8448 // lwoA_0_0_2_0 = (0*LSCA) + (2*LSPA)(*MT0I+PAD) = 8448
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+12:vgprG2LA+12+3] offset:12672 // lwoA_0_0_3_0 = (0*LSCA) + (3*LSPA)(*MT0I+PAD) = 12672

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* Recalc local read offsets */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
v_lshlrev_b32 v0, 0x1, v0                          // 4. apply VectorWidth: bnOffset = bnOffset * vw(2)
v_and_b32 v1, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x9, v1                          // 5. K offset: lrKOffset = kIdx * mStride(512)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v1, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v1, 3, v1                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(4)
v_lshlrev_b32 v1, 0x5, v1                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)
v_add_u32 v0, v1, v0                               // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x7, v1                          // 1. N offset: nOffset = nIdx * nStride(128)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v2, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x2, v2                          // 5. K offset: lrKOffset = kIdx * mStride(4)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s5, 16384                                // LSU offset: stride = lsuStride(128)*(MT0(128) + PAD0(0))
v_mul_lo_u32 v2, s5, v2                            // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v2, v0            // Final Offset: offset = (lro0+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v3, 10, v[vgprLocalReadAddrA]        // Final Offset: padding 32 per block 1024
v_lshlrev_b32 v3, 0x5, v3                          // Final Offset: padding 32 per block 1024
v_add_u32 v[vgprLocalReadAddrA], v3, v[vgprLocalReadAddrA] // Final Offset: add padding 32 per block 1024
/* N/A */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s5, 128                                  // LSU offset: stride = lsuStride(128) when umlds==True
v_mul_lo_u32 v0, s5, v0                            // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v1, 0x1  // Final Offset: offset = (lro1+lsuoffset)*bpeDS
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v2, 0x5, v2                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrB], v2, v[vgprLocalReadAddrB] // Final Offset: add padding 32 per block 256
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x4200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip force waitcnt0
s_barrier

/* local read reset offsets a */

/* local read reset offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
label_TailLoopBeginL:

/* Tail: remove ValuA/B vgpr buffer [0...54) from pool */

/* Tail: add address/G2L vgpr [54...96) to pool */

/* local read a */
ds_read_u16 v[vgprValuA_X0_I0_D0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u16 v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:384 // L -> Reg lro=0 swapByteOffset=0 ti=128 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
s_mov_b32 s5, 0x840                                // inc
v_add_co_u32 v[vgprLocalReadAddrA], vcc, s5, v[vgprLocalReadAddrA] // lrA += 2112 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s5, 0x20                                 // inc
v_add_co_u32 v[vgprLocalReadAddrB], vcc, s5, v[vgprLocalReadAddrB] // lrB += 32 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D0+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D0+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D1+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D1+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D2+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D2+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_cvt_pk_f32_fp8 v[vgprCvtTemp:vgprCvtTemp+1], v[vgprValuA_X0_I0_D3+0] src0_sel:WORD_0 // convert to F32
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+0] dst_sel:WORD_0 // Convert to FP16
v_cvt_f16_f32 v[vgprValuA_X0_I0_D3+0], v[vgprCvtTemp+1] dst_sel:WORD_1 // Convert to FP16
v_perm_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV0] // select K=01 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV0] // select K=23 for vector=0
v_perm_b32 v[vgprValuA_X0_I0+2], v[vgprValuA_X0_I0_D1+0], v[vgprValuA_X0_I0_D0+0], s[sgprPackKForV1] // select K=01 for vector=1
v_perm_b32 v[vgprValuA_X0_I0+3], v[vgprValuA_X0_I0_D3+0], v[vgprValuA_X0_I0_D2+0], s[sgprPackKForV1] // select K=23 for vector=1
v_and_b32 v54, 63, v[vgprSerial]                   // v54 = v[vgprSerial] % 64
v_lshrrev_b32 v54, 4, v54                          // v54 = v54 / 16
v_lshlrev_b32 v54, 0x2, v54                        // v54 = v54 * 4
v_cmp_ge_i32 s[70:71], v54, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0], v[vgprValuA_X0_I0+0+0], 0x0, s[70:71] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+0], v[vgprValuA_X0_I0+2+0], 0x0, s[70:71] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+1], v[vgprValuA_X0_I0+0+1], 0x0, s[70:71] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+2+1], v[vgprValuA_X0_I0+2+1], 0x0, s[70:71] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0], v[vgprValuB_X0_I0+0+0], 0x0, s[70:71] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+1], v[vgprValuB_X0_I0+0+1], 0x0, s[70:71] // set 0 if K_idx >= sizeL
v_sub_u32 v54, s[sgprLoopCounterL], v54            // get distance between size and k index
v_cmp_lt_i32 s[70:71], v54, 4                      // set partial 0 if distance less than input per thread
s_and_b32 s72, s[sgprLoopCounterL], 3              // get inputs for edge thread
s_sub_u32 s72, 4, s72                              // use shift to fill 0 for outside element
s_lshl_b32 s72, s72, 4                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[56:57], s72, v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v56, s[70:71]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v57, s[70:71]
v_lshlrev_b64 v[56:57], s72, v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+0], v[vgprValuA_X0_I0+2+0+0+0], v56, s[70:71]
v_cndmask_b32 v[vgprValuA_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+2+0+0+1], v57, s[70:71]
v_lshlrev_b64 v[56:57], s72, v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v56, s[70:71]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v57, s[70:71]
s_nop 1
v_mfma_f32_16x16x16_f16 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
v_mfma_f32_16x16x16_f16 acc[4:7], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+2+0+0:vgprValuA_X0_I0+2+0+0+1], acc[4:7] // left value = acc[4+0:7+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x10 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x10 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:

/* Tail: remove address/G2L [54...96) from pool */
label_Summation_End_X1FMFNDOHNM0P0XF_0:
/* endSummation: add vgpr [0...96) to pool */
.set sgprStaggerU, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprPackKForV0, UNDEF
.set sgprPackKForV1, UNDEF
.set sgprScalarGlobalReadOffsetA, UNDEF
/* load store sgprs */
.set sgprAddressScaleA, 48
.set sgprAddressScaleB, 50
.set sgprAddressScaleAlphaVec, 52
.set sgprAddressBias, 54
.set sgprBiasType, 56
.set sgprBiasStride, 57
.set sgpractivationAlpha, 58
.set sgpractivationBeta, 59
.set sgprActivationType, 60
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc0 label_GSU_4                         // branch if GSU != 1
s_load_dwordx8 s[48:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x5c
s_load_dwordx4 s[56:59], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x7c
s_load_dword s60, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8c
label_GSU_4:
.set sgprSrdScaleAlphaVec, 64
.set sgprSrdBias, 68

/* Mapping of Acc register -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_lshrrev_b32 v5, 2, v4                            // v5 = v4 / 4
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 1
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // v1 = v1 / 16
v_lshlrev_b32 v1, 0x2, v1                          // thread0 * continuous_output
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 3, v4                                // v0 = v4 % 4
v_mul_lo_u32 v0, 0x10, v0                          // wave coordination offset 0
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v0, v5, v0, 1                       // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s5, 128, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v0, s5, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s5, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v1, s5, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 52 bytes of kern args.
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_5                         // branch if GSU == 1
s_and_b32 s72, 127, s[sgprSizeI]                   // s72 = s[sgprSizeI] % 128
s_add_u32 s73, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s73                // wg0 >= nwg0-1 ?
s_cselect_b32 s72, s72, 0                          // set rMT0
s_cmpk_gt_u32 s72, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
s_and_b32 s72, 15, s[sgprSizeJ]                    // s72 = s[sgprSizeJ] % 16
s_add_u32 s73, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s73                // wg1 >= nwg1-1
s_cselect_b32 s72, s72, 0                          // set rMT1
s_cmpk_gt_u32 s72, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
label_GW_B0_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=74 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_lshl_u32 v6, v3, v0, 0x2                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+12], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+13], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+14], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+15], acc7           // copy acc to vreg[7]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[8:9], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s62, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx2 v[10:11], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s62, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx2 v[12:13], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s62, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s62        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dwordx2 v[14:15], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=50 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v18, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[72:73], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[76:77], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[76:77], s[72:73], s[76:77]             // in0 && in1
v_add_lshl_u32 v6, v3, v0, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v6, v18, v6, s[76:77]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[72:73], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[76:77], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[76:77], s[72:73], s[76:77]             // in0 && in1
v_add_lshl_u32 v7, v3, v0, 0x2                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, v18, v7, s[76:77]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[72:73], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[76:77], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[76:77], s[72:73], s[76:77]             // in0 && in1
v_add_lshl_u32 v12, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v18, v12, s[76:77]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[72:73], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[76:77], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[76:77], s[72:73], s[76:77]             // in0 && in1
v_add_lshl_u32 v13, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v13, v18, v13, s[76:77]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+8], acc0            // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+9], acc4            // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+10], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+11], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+15], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+16], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+17], acc7           // copy acc to vreg[7]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
buffer_store_dwordx2 v[8:9], v6, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx2 v[10:11], v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx2 v[14:15], v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dwordx2 v[16:17], v13, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[72:73]                               // addr of next instr
s_add_i32 s74, label_KernelEnd, 0x4                // target branch offset
s_add_u32 s72, s72, s74                            // add target branch offset
s_addc_u32 s73, s73, 0                             // add high and carry
s_setpc_b64 s[72:73]                               // branch to label_KernelEnd
label_GSU_5:
s_mov_b32 s5, 1.0                                  // init as 1
s_cmp_eq_u64 s[sgprAddressScaleA:sgprAddressScaleA+1], 0 // s[AddressScaleA] == 0 ?
s_cbranch_scc1 label_ScaleAValid                   // branch if s[AddressScaleA] == 0
s_load_dword s5, s[sgprAddressScaleA:sgprAddressScaleA+1], 0 // load scaleA
label_ScaleAValid:
s_mov_b32 s47, 1.0                                 // init as 1
s_cmp_eq_u64 s[sgprAddressScaleB:sgprAddressScaleB+1], 0 // s[AddressScaleB] == 0 ?
s_cbranch_scc1 label_ScaleBValid                   // branch if s[AddressScaleB] == 0
s_load_dword s47, s[sgprAddressScaleB:sgprAddressScaleB+1], 0 // load scaleB
label_ScaleBValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+0], s[sgprAddressScaleAlphaVec+0] // init SRD base address (lower)
s_mov_b32 s[sgprSrdScaleAlphaVec+1], s[sgprAddressScaleAlphaVec+1] // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdScaleAlphaVec+3], Srd127_96     // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], 0 // s[AddressScaleAlphaVec] == 0 ?
s_cbranch_scc0 label_ScaleAlphaVecAddrValid        // branch if s[AddressScaleAlphaVec] != 0
s_mov_b32 s[sgprSrdScaleAlphaVec+2], 0
s_branch label_ScaleAlphaVecAddrValid_End
label_ScaleAlphaVecAddrValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+2], s[sgprSizeI]
label_ScaleAlphaVecAddrValid_End:

s_mul_i32 s[sgprSrdScaleAlphaVec+2], 0x4, s[sgprSrdScaleAlphaVec+2] // ScaleAlphaVec scaled by BPE
s_add_u32 s61, s[sgprWorkGroup2], 0x1
s_mul_i32 s61, s[sgprBiasStride], s61              // stride * (wg+1)
s_cmp_eq_u32 s61, 0x0                              // bias stride = 0?
s_cselect_b32 s61, s[sgprSizeI], s61
s_mov_b32 s[sgprSrdBias+0], s[sgprAddressBias+0]   // init SRD base address (lower)
s_mov_b32 s[sgprSrdBias+1], s[sgprAddressBias+1]   // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdBias+3], Srd127_96              // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressBias:sgprAddressBias+1], 0 // s[AddressBias] == 0 ?
s_cbranch_scc0 label_BiasAddrValid                 // branch if s[AddressBias] != 0
s_mov_b32 s[sgprSrdBias+2], 0
s_branch label_BiasAddrValid_End
label_BiasAddrValid:
s_mov_b32 s[sgprSrdBias+2], s61
label_BiasAddrValid_End:

label_Load_Biasf32:
s_cmpk_lg_u32 s[sgprBiasType], 0                   // BiasType != 0
s_cbranch_scc1 label_Load_Biasf16                  // Branch if true

/******************************************/
/* Read Bias to LDS                       */
/******************************************/
s_mul_i32 s[sgprSrdBias+2], 0x4, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s61, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v8, s61, v[vgprSerial]                   // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s61, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v8, s61, v8                              // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v8, 0x2, v8                          // Global bias address scaled by BPE
buffer_load_dword v4, v8, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // load bias
v_lshlrev_b32 v8, 0x2, v[vgprSerial]               // Local bias address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for bias load
s_barrier                                          // Wait for all wavefronts
ds_write_b32 v8, v4 offset:0                       // store bias
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Biasf16:
s_cmpk_lg_u32 s[sgprBiasType], 4                   // BiasType != 4
s_cbranch_scc1 label_Load_Bias_End                 // Branch if true

/******************************************/
/* Read Bias to LDS                       */
/******************************************/
s_mul_i32 s[sgprSrdBias+2], 0x2, s[sgprSrdBias+2]  // scaled by BPE
s_mul_i32 s61, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_add_u32 v8, s61, v[vgprSerial]                   // coord 0 = wgp0 * MT0 + thread offset
s_mul_i32 s61, s[sgprBiasStride], s[sgprWorkGroup2] // Stride * WG
v_add_u32 v8, s61, v8                              // coord 0 = wgp0 * MT0 + thread offset + Stride * WG
v_lshlrev_b32 v8, 0x1, v8                          // Global bias address scaled by BPE
buffer_load_short_d16 v4, v8, s[sgprSrdBias:sgprSrdBias+3], 0 offen offset:0 // load bias
v_lshlrev_b32 v8, 0x2, v[vgprSerial]               // Local bias address scaled by BPE
s_waitcnt vmcnt(0)                                 // wait for bias load
s_barrier                                          // Wait for all wavefronts
v_cvt_f32_f16 v4, v4                               // convert to FP32
ds_write_b32 v8, v4 offset:0                       // store bias
s_branch label_Load_Bias_End                       // Branch to load bias end
label_Load_Bias_End:
v_mov_b32 v4, s[sgprAlpha]
s_waitcnt lgkmcnt(0)                               // wait for scaleAB load
v_mul_f32 v4, v4, s5
v_mul_f32 v4, v4, s47
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprAlpha], v4               // Update Alpha
s_cmpk_eq_u32 s[sgprActivationType], 1             // activationType == 1
s_cbranch_scc1 label_To_Activation_Abs_VW2         // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 2             // activationType == 2
s_cbranch_scc1 label_To_Activation_Clippedrelu_VW2 // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 3             // activationType == 3
s_cbranch_scc1 label_To_Activation_Gelu_VW2        // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 4             // activationType == 4
s_cbranch_scc1 label_To_Activation_Leakyrelu_VW2   // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 5             // activationType == 5
s_cbranch_scc1 label_To_Activation_Relu_VW2        // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 6             // activationType == 6
s_cbranch_scc1 label_To_Activation_Sigmoid_VW2     // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 7             // activationType == 7
s_cbranch_scc1 label_To_Activation_Tanh_VW2        // Branch if true
s_cmpk_eq_u32 s[sgprActivationType], 9             // activationType == 9
s_cbranch_scc1 label_To_Activation_Geluscaling_VW2 // Branch if true
label_To_Activation_None_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_None_VW2, 0x4       // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Abs_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Abs_VW2, 0x4        // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Clippedrelu_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Clippedrelu_VW2, 0x4 // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Gelu_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Gelu_VW2, 0x4       // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Leakyrelu_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Leakyrelu_VW2, 0x4  // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Relu_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Relu_VW2, 0x4       // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Sigmoid_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Sigmoid_VW2, 0x4    // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Tanh_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Tanh_VW2, 0x4       // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Geluscaling_VW2:
s_getpc_b64 s[62:63]                               // addr of next instr
s_add_i32 s5, label_Activation_Geluscaling_VW2, 0x4 // target branch offset
s_add_u32 s62, s62, s5                             // add target branch offset
s_addc_u32 s63, s63, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_ActivationSetPCAddrEnd:
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_GW_Beta                       // Branch if Beta is not zero

s_and_b32 s74, 127, s[sgprSizeI]                   // s74 = s[sgprSizeI] % 128
s_add_u32 s75, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s75                // wg0 >= nwg0-1 ?
s_cselect_b32 s74, s74, 0                          // set rMT0
s_cmpk_gt_u32 s74, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
s_and_b32 s74, 15, s[sgprSizeJ]                    // s74 = s[sgprSizeJ] % 16
s_add_u32 s75, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s75                // wg1 >= nwg1-1
s_cselect_b32 s74, s74, 0                          // set rMT1
s_cmpk_gt_u32 s74, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
label_GW_B0_E0_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=24 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v9, v0, s74
v_lshlrev_b32 v9, 0x2, v9                          // Bias address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
s_barrier                                          // Bias LDS write barrier
ds_read_b64 v[12:13], v9 offset:0                  // load bias
v_lshlrev_b32 v10, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
buffer_load_dwordx2 v[14:15], v10, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // load scaleAlphaVecI
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_lshl_u32 v7, v3, v0, 0x1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+20], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+21], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+22], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+23], acc7           // copy acc to vreg[7]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha

/* apply mask, calc new C and issue writes */

s_waitcnt 0                                        // vmcnt(0) = 1 - 1 (scaleAlphaVec) lgkmcnt(0) = 1 - 1 (bias) (interleaved)
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], v[14:15], v[vgprValuC+16:vgprValuC+16+1] // *= scaleAlphaVecVMulPK(14)(0)
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+16:vgprValuC+16+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v16, v4
v_mov_b32 v17, v5
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
buffer_store_dword v16, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+18:vgprValuC+18+1], v[14:15], v[vgprValuC+18:vgprValuC+18+1] // *= scaleAlphaVecVMulPK(14)(0)
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+18:vgprValuC+18+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v18, v4
v_mov_b32 v19, v5
v_cvt_f16_f32 v[vgprValuC+18], v[vgprValuC+18]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+19], v[vgprValuC+19]     // convert C to fp16
v_pack_b32_f16 v18, v[vgprValuC+18], v[vgprValuC+19] // Pack with neighbor
s_lshl_b32 s74, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v18, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], v[14:15], v[vgprValuC+20:vgprValuC+20+1] // *= scaleAlphaVecVMulPK(14)(0)
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+20:vgprValuC+20+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v20, v4
v_mov_b32 v21, v5
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
s_lshl_b32 s74, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v20, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], v[14:15], v[vgprValuC+22:vgprValuC+22+1] // *= scaleAlphaVecVMulPK(14)(0)
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+22:vgprValuC+22+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v22, v4
v_mov_b32 v23, v5
v_cvt_f16_f32 v[vgprValuC+22], v[vgprValuC+22]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+23], v[vgprValuC+23]     // convert C to fp16
v_pack_b32_f16 v22, v[vgprValuC+22], v[vgprValuC+23] // Pack with neighbor
s_lshl_b32 s74, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v22, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_E1_1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=14 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v29, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v8, v0, s74
v_lshlrev_b32 v8, 0x2, v8                          // Bias address scaled by BPE
v_cndmask_b32 v8, v29, v8, s[78:79]                // LDBias clip if OOB. offset
s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
s_barrier                                          // Bias LDS write barrier
ds_read_b64 v[10:11], v8 offset:0                  // load bias
v_lshlrev_b32 v9, 0x2, v0                          // ScaleAlphaVec address scaled by BPE
buffer_load_dwordx2 v[12:13], v9, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // load scaleAlphaVecI
v_add_lshl_u32 v7, v3, v0, 0x1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, v29, v7, s[78:79]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v17, v0, s74
v_lshlrev_b32 v17, 0x2, v17                        // Bias address scaled by BPE
v_cndmask_b32 v17, v29, v17, s[78:79]              // LDBias clip if OOB. offset
v_lshlrev_b32 v18, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
v_add_lshl_u32 v16, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, v29, v16, s[78:79]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v22, v0, s74
v_lshlrev_b32 v22, 0x2, v22                        // Bias address scaled by BPE
v_cndmask_b32 v22, v29, v22, s[78:79]              // LDBias clip if OOB. offset
v_lshlrev_b32 v23, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
v_add_lshl_u32 v19, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v19, v29, v19, s[78:79]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v27, v0, s74
v_lshlrev_b32 v27, 0x2, v27                        // Bias address scaled by BPE
v_cndmask_b32 v27, v29, v27, s[78:79]              // LDBias clip if OOB. offset
v_lshlrev_b32 v28, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
v_add_lshl_u32 v26, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v29, v26, s[78:79]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+14], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+15], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+20], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+21], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+24], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+25], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+30], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+31], acc7           // copy acc to vreg[7]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
s_waitcnt 0                                        // wait for ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v12, 1.0, v12, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v13, 1.0, v13, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+14:vgprValuC+14+1], v[12:13], v[vgprValuC+14:vgprValuC+14+1] // *= scaleAlphaVecVMulPK(12)(0)
v_pk_add_f32 v[4:5], v[10:11], v[vgprValuC+14:vgprValuC+14+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v14, v4
v_mov_b32 v15, v5
v_cvt_f16_f32 v[vgprValuC+14], v[vgprValuC+14]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+15], v[vgprValuC+15]     // convert C to fp16
v_pack_b32_f16 v14, v[vgprValuC+14], v[vgprValuC+15] // Pack with neighbor
buffer_store_dword v14, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v12, 1.0, v12, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v13, 1.0, v13, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], v[12:13], v[vgprValuC+20:vgprValuC+20+1] // *= scaleAlphaVecVMulPK(12)(0)
v_pk_add_f32 v[4:5], v[10:11], v[vgprValuC+20:vgprValuC+20+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v20, v4
v_mov_b32 v21, v5
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
buffer_store_dword v20, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v12, 1.0, v12, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v13, 1.0, v13, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+24:vgprValuC+24+1], v[12:13], v[vgprValuC+24:vgprValuC+24+1] // *= scaleAlphaVecVMulPK(12)(0)
v_pk_add_f32 v[4:5], v[10:11], v[vgprValuC+24:vgprValuC+24+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v24, v4
v_mov_b32 v25, v5
v_cvt_f16_f32 v[vgprValuC+24], v[vgprValuC+24]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+25], v[vgprValuC+25]     // convert C to fp16
v_pack_b32_f16 v24, v[vgprValuC+24], v[vgprValuC+25] // Pack with neighbor
buffer_store_dword v24, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v12, 1.0, v12, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v13, 1.0, v13, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+30:vgprValuC+30+1], v[12:13], v[vgprValuC+30:vgprValuC+30+1] // *= scaleAlphaVecVMulPK(12)(0)
v_pk_add_f32 v[4:5], v[10:11], v[vgprValuC+30:vgprValuC+30+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v30, v4
v_mov_b32 v31, v5
v_cvt_f16_f32 v[vgprValuC+30], v[vgprValuC+30]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+31], v[vgprValuC+31]     // convert C to fp16
v_pack_b32_f16 v30, v[vgprValuC+30], v[vgprValuC+31] // Pack with neighbor
buffer_store_dword v30, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_Beta:
s_and_b32 s74, 127, s[sgprSizeI]                   // s74 = s[sgprSizeI] % 128
s_add_u32 s75, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s75                // wg0 >= nwg0-1 ?
s_cselect_b32 s74, s74, 0                          // set rMT0
s_cmpk_gt_u32 s74, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
s_and_b32 s74, 15, s[sgprSizeJ]                    // s74 = s[sgprSizeJ] % 16
s_add_u32 s75, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s75                // wg1 >= nwg1-1
s_cselect_b32 s74, s74, 0                          // set rMT1
s_cmpk_gt_u32 s74, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
label_GW_B1_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=20 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v8, v2, v0, 0x1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
buffer_load_dword v11, v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v9, v0, s74
v_lshlrev_b32 v9, 0x2, v9                          // Bias address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
s_barrier                                          // Bias LDS write barrier
ds_read_b64 v[12:13], v9 offset:0                  // load bias
v_lshlrev_b32 v10, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
buffer_load_dwordx2 v[14:15], v10, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // load scaleAlphaVecI
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s74, s[sgprStrideC1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dword v18, v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s74, s[sgprStrideC1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dword v19, v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s74, s[sgprStrideC1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_dword v24, v8, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v7, v3, v0, 0x1                     // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+20], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+21], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+22], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+23], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+26], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+27], acc7           // copy acc to vreg[7]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha

/* apply mask, calc new C and issue writes */

s_waitcnt lgkmcnt(0), vmcnt(3)                     // vmcnt(3) = 5 - 1 (beta) - 1 (scaleAlphaVec) lgkmcnt(0) = 1 - 1 (bias) (interleaved)
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], v[14:15], v[vgprValuC+16:vgprValuC+16+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v11, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v11, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+16:vgprValuC+16+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v16, v4
v_mov_b32 v17, v5
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
buffer_store_dword v16, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(2) = 5 - 2 (beta) - 1 (scaleAlphaVec) (interleaved)
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+20:vgprValuC+20+1], v[14:15], v[vgprValuC+20:vgprValuC+20+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+20], s[sgprBeta], v18, v[vgprValuC+20] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+21], s[sgprBeta], v18, v[vgprValuC+21] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+20:vgprValuC+20+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v20, v4
v_mov_b32 v21, v5
v_cvt_f16_f32 v[vgprValuC+20], v[vgprValuC+20]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+21], v[vgprValuC+21]     // convert C to fp16
v_pack_b32_f16 v20, v[vgprValuC+20], v[vgprValuC+21] // Pack with neighbor
s_lshl_b32 s74, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v20, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(1) = 5 - 3 (beta) - 1 (scaleAlphaVec) (interleaved)
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], v[14:15], v[vgprValuC+22:vgprValuC+22+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+22], s[sgprBeta], v19, v[vgprValuC+22] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+23], s[sgprBeta], v19, v[vgprValuC+23] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+22:vgprValuC+22+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v22, v4
v_mov_b32 v23, v5
v_cvt_f16_f32 v[vgprValuC+22], v[vgprValuC+22]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+23], v[vgprValuC+23]     // convert C to fp16
v_pack_b32_f16 v22, v[vgprValuC+22], v[vgprValuC+23] // Pack with neighbor
s_lshl_b32 s74, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v22, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(0) = 5 - 4 (beta) - 1 (scaleAlphaVec) (interleaved)
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+26:vgprValuC+26+1], v[14:15], v[vgprValuC+26:vgprValuC+26+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+26], s[sgprBeta], v24, v[vgprValuC+26] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+27], s[sgprBeta], v24, v[vgprValuC+27] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+26:vgprValuC+26+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v26, v4
v_mov_b32 v27, v5
v_cvt_f16_f32 v[vgprValuC+26], v[vgprValuC+26]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+27], v[vgprValuC+27]     // convert C to fp16
v_pack_b32_f16 v26, v[vgprValuC+26], v[vgprValuC+27] // Pack with neighbor
s_lshl_b32 s74, s[sgprStrideD1J], 1                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s74        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v26, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=14 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw2); (0,0,1,0:vw2); (0,0,2,0:vw2); (0,0,3,0:vw2) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v33, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
v_add_lshl_u32 v7, v2, v0, 0x1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, v33, v7, s[78:79]                // LDC clip if OOB. offset
buffer_load_dword v10, v7, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v8, v0, s74
v_lshlrev_b32 v8, 0x2, v8                          // Bias address scaled by BPE
v_cndmask_b32 v8, v33, v8, s[78:79]                // LDBias clip if OOB. offset
s_waitcnt lgkmcnt(0)                               // Wait for Bias LDS write
s_barrier                                          // Bias LDS write barrier
ds_read_b64 v[12:13], v8 offset:0                  // load bias
v_lshlrev_b32 v9, 0x2, v0                          // ScaleAlphaVec address scaled by BPE
buffer_load_dwordx2 v[14:15], v9, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // load scaleAlphaVecI
v_add_lshl_u32 v7, v3, v0, 0x1                     // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v7, v33, v7, s[78:79]                // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
v_add_lshl_u32 v11, v2, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v33, v11, s[78:79]              // LDC clip if OOB. offset
buffer_load_dword v20, v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v18, v0, s74
v_lshlrev_b32 v18, 0x2, v18                        // Bias address scaled by BPE
v_cndmask_b32 v18, v33, v18, s[78:79]              // LDBias clip if OOB. offset
v_lshlrev_b32 v19, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
v_add_lshl_u32 v11, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v11, v33, v11, s[78:79]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
v_add_lshl_u32 v21, v2, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v21, v33, v21, s[78:79]              // LDC clip if OOB. offset
buffer_load_dword v26, v21, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v24, v0, s74
v_lshlrev_b32 v24, 0x2, v24                        // Bias address scaled by BPE
v_cndmask_b32 v24, v33, v24, s[78:79]              // LDBias clip if OOB. offset
v_lshlrev_b32 v25, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
v_add_lshl_u32 v21, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v21, v33, v21, s[78:79]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[74:75], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[78:79], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[78:79], s[74:75], s[78:79]             // in0 && in1
v_add_lshl_u32 v27, v2, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v33, v27, s[78:79]              // LDC clip if OOB. offset
buffer_load_dword v32, v27, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s74, 128, s[sgprWorkGroup0]              // wgp0 * MT0
v_sub_u32 v30, v0, s74
v_lshlrev_b32 v30, 0x2, v30                        // Bias address scaled by BPE
v_cndmask_b32 v30, v33, v30, s[78:79]              // LDBias clip if OOB. offset
v_lshlrev_b32 v31, 0x2, v0                         // ScaleAlphaVec address scaled by BPE
v_add_lshl_u32 v27, v3, v0, 0x1                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v33, v27, s[78:79]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+16], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+17], acc4           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+22], acc1           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+23], acc5           // copy acc to vreg[3]
v_accvgpr_read_b32 v[vgprValuC+28], acc2           // copy acc to vreg[4]
v_accvgpr_read_b32 v[vgprValuC+29], acc6           // copy acc to vreg[5]
v_accvgpr_read_b32 v[vgprValuC+34], acc3           // copy acc to vreg[6]
v_accvgpr_read_b32 v[vgprValuC+35], acc7           // copy acc to vreg[7]
s_nop 1                                            // 2 wait states required before reading vgpr

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec, Bias LDS

/* apply mask, calc new C and issue writes */
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+16:vgprValuC+16+1], v[14:15], v[vgprValuC+16:vgprValuC+16+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+16], s[sgprBeta], v10, v[vgprValuC+16] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+17], s[sgprBeta], v10, v[vgprValuC+17] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+16:vgprValuC+16+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v16, v4
v_mov_b32 v17, v5
v_cvt_f16_f32 v[vgprValuC+16], v[vgprValuC+16]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+17], v[vgprValuC+17]     // convert C to fp16
v_pack_b32_f16 v16, v[vgprValuC+16], v[vgprValuC+17] // Pack with neighbor
buffer_store_dword v16, v7, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+22:vgprValuC+22+1], v[14:15], v[vgprValuC+22:vgprValuC+22+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+22], s[sgprBeta], v20, v[vgprValuC+22] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+23], s[sgprBeta], v20, v[vgprValuC+23] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+22:vgprValuC+22+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v22, v4
v_mov_b32 v23, v5
v_cvt_f16_f32 v[vgprValuC+22], v[vgprValuC+22]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+23], v[vgprValuC+23]     // convert C to fp16
v_pack_b32_f16 v22, v[vgprValuC+22], v[vgprValuC+23] // Pack with neighbor
buffer_store_dword v22, v11, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+28:vgprValuC+28+1], v[14:15], v[vgprValuC+28:vgprValuC+28+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+28], s[sgprBeta], v26, v[vgprValuC+28] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+29], s[sgprBeta], v26, v[vgprValuC+29] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+28:vgprValuC+28+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v28, v4
v_mov_b32 v29, v5
v_cvt_f16_f32 v[vgprValuC+28], v[vgprValuC+28]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+29], v[vgprValuC+29]     // convert C to fp16
v_pack_b32_f16 v28, v[vgprValuC+28], v[vgprValuC+29] // Pack with neighbor
buffer_store_dword v28, v21, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
v_cndmask_b32 v14, 1.0, v14, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_cndmask_b32 v15, 1.0, v15, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
v_pk_mul_f32 v[vgprValuC+34:vgprValuC+34+1], v[14:15], v[vgprValuC+34:vgprValuC+34+1] // *= scaleAlphaVecVMulPK(14)(0)
v_fma_mix_f32 v[vgprValuC+34], s[sgprBeta], v32, v[vgprValuC+34] op_sel:[0,0,0] op_sel_hi:[0,1,0] // //C*=beta
v_fma_mix_f32 v[vgprValuC+35], s[sgprBeta], v32, v[vgprValuC+35] op_sel:[0,1,0] op_sel_hi:[0,1,0] // //C*=beta
v_pk_add_f32 v[4:5], v[12:13], v[vgprValuC+34:vgprValuC+34+1] // C += bias
s_swappc_b64 s[72:73], s[62:63]
v_mov_b32 v34, v4
v_mov_b32 v35, v5
v_cvt_f16_f32 v[vgprValuC+34], v[vgprValuC+34]     // convert C to fp16
v_cvt_f16_f32 v[vgprValuC+35], v[vgprValuC+35]     // convert C to fp16
v_pack_b32_f16 v34, v[vgprValuC+34], v[vgprValuC+35] // Pack with neighbor
buffer_store_dword v34, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_Activation_None_VW2:
s_setpc_b64 s[72:73]
label_Activation_Abs_VW2:
v_and_b32 v[vgprValuC+4], 0x7fffffff, v[vgprValuC+4] // Remove sign bit
v_and_b32 v[vgprValuC+5], 0x7fffffff, v[vgprValuC+5] // Remove sign bit
s_setpc_b64 s[72:73]
label_Activation_Clippedrelu_VW2:
v_cmp_gt_f32 vcc, v[vgprValuC+4], s[sgpractivationAlpha] // x > alpha ?
v_min_f32 v[vgprValuC+4], s[sgpractivationBeta], v[vgprValuC+4] // min(x, beta)
v_cndmask_b32 v[vgprValuC+4], 0.0, v[vgprValuC+4], vcc // set x to 0 if <= alpha
v_cmp_gt_f32 vcc, v[vgprValuC+5], s[sgpractivationAlpha] // x > alpha ?
v_min_f32 v[vgprValuC+5], s[sgpractivationBeta], v[vgprValuC+5] // min(x, beta)
v_cndmask_b32 v[vgprValuC+5], 0.0, v[vgprValuC+5], vcc // set x to 0 if <= alpha
s_setpc_b64 s[72:73]
label_Activation_Gelu_VW2:
v_mul_f32 v6, 0x3d372713, v[vgprValuC+4]           // k1 * x
v_fma_f32 v6, v[vgprValuC+4], v6, 1.0              // 1 + (k1 * x * x)
v_mul_f32 v6, v[vgprValuC+4], v6                   // x * (1 + k1 * x * x)
v_mul_f32 v6, 0x40135761, v6                       //  (fused 2.302208)
v_exp_f32 v6, v6                                   // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v6, 1.0, v6                              // e^2x + 1
v_rcp_f32 v6, v6                                   // 1 / (e^2x + 1)
s_nop 0                                            // 1 wait states
v_fma_f32 v6, -2.0, v6, 2.0                        //  ( + 1 (fused))
v_mul_f32 v6, v[vgprValuC+4], v6                   // x * (1 + tanh(...))
v_mul_f32 v[vgprValuC+4], 0.5, v6                  // 0.5 * x * (1 + tanh(...))
v_mul_f32 v6, 0x3d372713, v[vgprValuC+5]           // k1 * x
v_fma_f32 v6, v[vgprValuC+5], v6, 1.0              // 1 + (k1 * x * x)
v_mul_f32 v6, v[vgprValuC+5], v6                   // x * (1 + k1 * x * x)
v_mul_f32 v6, 0x40135761, v6                       //  (fused 2.302208)
v_exp_f32 v6, v6                                   // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v6, 1.0, v6                              // e^2x + 1
v_rcp_f32 v6, v6                                   // 1 / (e^2x + 1)
s_nop 0                                            // 1 wait states
v_fma_f32 v6, -2.0, v6, 2.0                        //  ( + 1 (fused))
v_mul_f32 v6, v[vgprValuC+5], v6                   // x * (1 + tanh(...))
v_mul_f32 v[vgprValuC+5], 0.5, v6                  // 0.5 * x * (1 + tanh(...))
s_setpc_b64 s[72:73]
label_Activation_Leakyrelu_VW2:
v_mul_f32 v6, s[sgpractivationAlpha], v[vgprValuC+4] // tmp = x * alpha
v_cmp_ge_f32 vcc, v[vgprValuC+4], 0.0              // x >= 0 ?
v_cndmask_b32 v[vgprValuC+4], v6, v[vgprValuC+4], vcc // set x to tmp if < 0
v_mul_f32 v6, s[sgpractivationAlpha], v[vgprValuC+5] // tmp = x * alpha
v_cmp_ge_f32 vcc, v[vgprValuC+5], 0.0              // x >= 0 ?
v_cndmask_b32 v[vgprValuC+5], v6, v[vgprValuC+5], vcc // set x to tmp if < 0
s_setpc_b64 s[72:73]
label_Activation_Relu_VW2:
v_max_f32 v[vgprValuC+4], v[vgprValuC+4], 0        // x = max(0, x)
v_max_f32 v[vgprValuC+5], v[vgprValuC+5], 0        // x = max(0, x)
s_setpc_b64 s[72:73]
label_Activation_Sigmoid_VW2:
v_mul_f32 v[vgprValuC+4], 0xbfb8aa3b, v[vgprValuC+4] //  (fused -1.442695)
v_exp_f32 v[vgprValuC+4], v[vgprValuC+4]           // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v[vgprValuC+4], 1.0, v[vgprValuC+4]      // 1 + exp(-x)
v_rcp_f32 v[vgprValuC+4], v[vgprValuC+4]           // 1 / (1 + exp(-x))
s_nop 0                                            // 1 wait states
v_mul_f32 v[vgprValuC+5], 0xbfb8aa3b, v[vgprValuC+5] //  (fused -1.442695)
v_exp_f32 v[vgprValuC+5], v[vgprValuC+5]           // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v[vgprValuC+5], 1.0, v[vgprValuC+5]      // 1 + exp(-x)
v_rcp_f32 v[vgprValuC+5], v[vgprValuC+5]           // 1 / (1 + exp(-x))
s_nop 0                                            // 1 wait states
s_setpc_b64 s[72:73]
label_Activation_Tanh_VW2:
v_mul_f32 v[vgprValuC+4], s[sgpractivationAlpha], v[vgprValuC+4] // x * alpha
v_mul_f32 v[vgprValuC+4], 0x4038aa3b, v[vgprValuC+4] //  (fused 2)
v_exp_f32 v[vgprValuC+4], v[vgprValuC+4]           // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v[vgprValuC+4], 1.0, v[vgprValuC+4]      // e^2x + 1
v_rcp_f32 v[vgprValuC+4], v[vgprValuC+4]           // 1 / (e^2x + 1)
s_nop 0                                            // 1 wait states
v_fma_f32 v[vgprValuC+4], -2.0, v[vgprValuC+4], 1.0 // (-2) * (1 / (e^2x + 1)) + 1
v_mul_f32 v[vgprValuC+4], s[sgpractivationBeta], v[vgprValuC+4] // beta * tanh(x)
v_mul_f32 v[vgprValuC+5], s[sgpractivationAlpha], v[vgprValuC+5] // x * alpha
v_mul_f32 v[vgprValuC+5], 0x4038aa3b, v[vgprValuC+5] //  (fused 2)
v_exp_f32 v[vgprValuC+5], v[vgprValuC+5]           // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v[vgprValuC+5], 1.0, v[vgprValuC+5]      // e^2x + 1
v_rcp_f32 v[vgprValuC+5], v[vgprValuC+5]           // 1 / (e^2x + 1)
s_nop 0                                            // 1 wait states
v_fma_f32 v[vgprValuC+5], -2.0, v[vgprValuC+5], 1.0 // (-2) * (1 / (e^2x + 1)) + 1
v_mul_f32 v[vgprValuC+5], s[sgpractivationBeta], v[vgprValuC+5] // beta * tanh(x)
s_setpc_b64 s[72:73]
label_Activation_Geluscaling_VW2:
v_mul_f32 v6, 0x3d372713, v[vgprValuC+4]           // k1 * x
v_fma_f32 v6, v[vgprValuC+4], v6, 1.0              // 1 + (k1 * x * x)
v_mul_f32 v6, v[vgprValuC+4], v6                   // x * (1 + k1 * x * x)
v_mul_f32 v6, 0x40135761, v6                       //  (fused 2.302208)
v_exp_f32 v6, v6                                   // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v6, 1.0, v6                              // e^2x + 1
v_rcp_f32 v6, v6                                   // 1 / (e^2x + 1)
s_nop 0                                            // 1 wait states
v_fma_f32 v6, -2.0, v6, 2.0                        //  ( + 1 (fused))
v_mul_f32 v6, v[vgprValuC+4], v6                   // x * (1 + tanh(...))
v_mul_f32 v6, 0.5, v6                              // 0.5 * x * (1 + tanh(...))
v_mul_f32 v[vgprValuC+4], s[sgpractivationAlpha], v6 // 0.5 * x * (1 + tanh(...)) * scale
v_mul_f32 v6, 0x3d372713, v[vgprValuC+5]           // k1 * x
v_fma_f32 v6, v[vgprValuC+5], v6, 1.0              // 1 + (k1 * x * x)
v_mul_f32 v6, v[vgprValuC+5], v6                   // x * (1 + k1 * x * x)
v_mul_f32 v6, 0x40135761, v6                       //  (fused 2.302208)
v_exp_f32 v6, v6                                   // exp step 2
s_nop 0                                            // 1 wait states
v_add_f32 v6, 1.0, v6                              // e^2x + 1
v_rcp_f32 v6, v6                                   // 1 / (e^2x + 1)
s_nop 0                                            // 1 wait states
v_fma_f32 v6, -2.0, v6, 2.0                        //  ( + 1 (fused))
v_mul_f32 v6, v[vgprValuC+5], v6                   // x * (1 + tanh(...))
v_mul_f32 v6, 0.5, v6                              // 0.5 * x * (1 + tanh(...))
v_mul_f32 v[vgprValuC+5], s[sgpractivationAlpha], v6 // 0.5 * x * (1 + tanh(...)) * scale
s_setpc_b64 s[72:73]
label_GW_End_1:
label_KernelEnd:
s_endpgm                                           // Kernel End


