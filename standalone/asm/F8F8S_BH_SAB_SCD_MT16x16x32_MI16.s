
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected Cijk_Ailk_Bljk_F8F8S_BH_SAB_SCD_MT16x16x32_MI16x16x1_SN_K1_MIWT1_1
.globl Cijk_Ailk_Bljk_F8F8S_BH_SAB_SCD_MT16x16x32_MI16x16x1_SN_K1_MIWT1_1
.p2align 8
.type Cijk_Ailk_Bljk_F8F8S_BH_SAB_SCD_MT16x16x32_MI16x16x1_SN_K1_MIWT1_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Ailk_Bljk_F8F8S_BH_SAB_SCD_MT16x16x32_MI16x16x1_SN_K1_MIWT1_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 48 // accvgpr offset
  .amdhsa_next_free_vgpr 52 // vgprs
  .amdhsa_next_free_sgpr 70 // sgprs
  // .amdhsa_group_segment_fixed_size 2048 // lds bytes
  .amdhsa_group_segment_fixed_size 2080 // lds bytes + amax
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_user_sgpr_count 13
  .amdhsa_user_sgpr_kernarg_preload_length 11
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.end_amdhsa_kernel
.text
/* Num VGPR   =44 */
/* Num AccVGPR=4 */
/* Num SGPR   =70 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 1 */
/* SubGroup= 4 x 16 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=False */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 1
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Cijk_Ailk_Bljk_F8F8S_BH_SAB_SCD_MT16x16x32_MI16x16x1_SN_K1_MIWT1_1
    .symbol: 'Cijk_Ailk_Bljk_F8F8S_BH_SAB_SCD_MT16x16x32_MI16x16x1_SN_K1_MIWT1_1.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            Gemm info
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info0
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info1
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            numWG
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          24
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          28
        .value_kind:      by_value
        .value_type:      u32
      - .name:            D
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      fp8_fp8
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      fp8_fp8
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      fp8
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      fp8
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      f32
      - .name:            beta
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      f32
      - .name:            AddressScaleA
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleB
        .size:            8
        .offset:          112
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleC
        .size:            8
        .offset:          120
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AddressScaleD
        .size:            8
        .offset:          128
        .value_kind:      global_buffer
        .value_type:      f32
        .address_space:   generic
      - .name:            AmaxDOutput
        .size:            8
        .offset:          136
        .value_kind:      global_buffer
        .address_space:   generic
      - .name:            Workspace
        .size:            8
        .offset:          144
        .value_kind:      global_buffer
        .address_space:   generic
      - .name:            Sync
        .size:            8
        .offset:          152
        .value_kind:      global_buffer
        .address_space:   generic
      // - .name:            numWGs
      //   .size:            4
      //   .offset:          160
      //   .value_kind:      by_value
      //   .value_type:      f32
    .group_segment_fixed_size:   2048
    .kernarg_segment_align:      8
    // .kernarg_segment_size:       136
    .kernarg_segment_size:       160
    .max_flat_workgroup_size:    64
    .private_segment_fixed_size: 0
    .sgpr_count:                 70
    .sgpr_spill_count:           0
    .vgpr_count:                 44
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Ailk_Bljk_F8F8S_BH_SAB_SCD_MT16x16x32_MI16x16x1_SN_K1_MIWT1_1:
label_ASM_Start:  /// Main body of the asm kernel

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
.set vgprValuA_X0_I0_D1, 2
.set vgprValuA_X0_I0_D2, 4
.set vgprValuA_X0_I0_D3, 6
.set vgprValuB_X0_I0, 8
.set vgprLocalWriteAddrA, 11
.set vgprLocalWriteAddrB, 12
.set vgprGlobalReadOffsetA, 13
.set vgprGlobalReadOffsetB, 14
.set vgprG2LA, 16
.set vgprG2LB, 18
.set vgprPackTemp, 10
.set vgprLocalReadAddrA, 20
.set vgprLocalReadAddrB, 21
.set vgprSerial, 22

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
.set sgprStaggerU, 9
.set sgprWGM, 10
.set sgprLoopCounterL, 11
.set sgprOrigLoopCounter, 12
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprNumWorkGroups0, 13
.set sgprNumWorkGroups1, 14
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

.set MT0, 16
.set MT1, 16
.set DepthU, 32
.set BpeA, 1
.set BpeALog2, 0
.set BpeB, 1
.set BpeBLog2, 0
.set BpeAGR, 1
.set BpeAGRLog2, 0
.set BpeBGR, 1
.set BpeBGRLog2, 0
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 8
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
    v_add_u32 v[\vgprAddr+0] 0x8 v[\vgprAddr+0]      // add prepad for pointer shift
                                                       // offset *= bytes/element (multiplier is 1 do nothing)
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideB1J] v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffsetL] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x8 v[\vgprAddr+0]      // add prepad for pointer shift
                                                       // offset *= bytes/element (multiplier is 1 do nothing)
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

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s47, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0

/* Load GSU data */
s_load_dword s[sgprGSU], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8
s_waitcnt lgkmcnt(0)
s_lshr_b32 s48, s47, 0x1e                          // Get arg type
s_and_b32 s47, 0x3fffffff, s47                     // Get nums of gemm
s_cmp_eq_u32 s48, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_waitcnt lgkmcnt(0)
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_branch label_common_kernel_entry

/* pad 39 snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue */
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
label_Preload_Offset_Start:
s_and_b32 s47, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s48, s2, 0x1e                           // Get arg type
s_mov_b32 s[sgprGSU], s3                           // Preload internal args
s_cmp_eq_u32 s48, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dword s31, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x1c
s_load_dwordx8 s[32:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_mov_b32 s24, s6                                  // move preload data to correct sgpr
s_mov_b32 s25, s7                                  // move preload data to correct sgpr
s_mov_b32 s26, s8                                  // move preload data to correct sgpr
s_mov_b32 s27, s9                                  // move preload data to correct sgpr
s_mov_b32 s28, s10                                 // move preload data to correct sgpr
s_mov_b32 s29, s11                                 // move preload data to correct sgpr
s_mov_b32 s30, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprStaggerU], s[sgprGSU], 0xffff0000  // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s[sgprGSU], 0xffff           // Restore GSU
s_setprio 3                                        // optimization store
s_mov_b32 m0, 0x800                                // LDS clamp at 2048 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id
s_cmp_eq_u32 s48, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [0...11) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...4) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v1, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x7, v1                          // 5. K offset: lrKOffset = kIdx * mStride(128)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x5, v1                          // 1. N offset: nOffset = nIdx * nStride(32)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v2, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x3, v2                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 512                                 // LSU offset: stride = lsuStride(32)*(MT0(16) + PAD0(0))
v_mul_lo_u32 v2, s49, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v2, v0            // Final Offset: offset = (lro0+lsuoffset)*bpeDS(1)

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 32                                  // LSU offset: stride = lsuStride(32) when umlds==True
v_mul_lo_u32 v0, s49, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v0, v1            // Final Offset: offset = (lro1+lsuoffset)*bpeDS(1)

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 2 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 1, v[vgprSerial]                 // v1 = v[vgprSerial] / 2
v_and_b32 v0, 1, v[vgprSerial]                     // v0 = v[vgprSerial] % 2
/* tile *= glvw */
v_lshlrev_b32 v0, 0x3, v0                          // v0 = v0 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 4 */
/* v3 = B-unroll = serial%LVCB */
v_and_b32 v5, 63, v[vgprSerial]                    // v5 = v[vgprSerial] % 64
v_lshrrev_b32 v2, 2, v5                            // v2 = v5 / 4
v_and_b32 v3, 3, v5                                // v3 = v5 % 4
v_readfirstlane_b32 s49, v[vgprSerial]             // WaveIdxWavefrontWidth
s_lshr_b32 s49, s49, 0x6                           // WaveId
s_mul_i32 s49, s49, 16                             // Each wave loads continuous lsp(16)*nrp(1) columns
v_add_u32 v2, s49, v2                              // Add back to column index
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v4     // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS(1)

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x20, v2     // lwBL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS(1)
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=512
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
s_waitcnt lgkmcnt(0)                               // wait for 44 bytes of kern args

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s48, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_mul_i32 s48, s48, s[sgprGSU]
v_cvt_f32_u32 v6, s48                              // s48 = s[sgprWorkGroup0] / s48
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s48
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s48
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s48
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s48
v_mul_u32_u24 v7, v6, s48                          // s48 = s[sgprWorkGroup0] / s48
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s48
v_cmpx_eq_u32 exec, v7, s48                        // s48 = s[sgprWorkGroup0] / s48
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s48
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s48
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s48
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s48, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s48, s48, s[sgprWorkGroup2]
s_mul_i32 s48, s48, s[sgprGSU]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v7, s[sgprNumWorkGroups0]      // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s48
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s48, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
s_branch label_MultiGemmEnd
label_MultiGemm:
s_mov_b32 s14, 120
s_mul_i32 s54, s47, 4
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s13, 1
s_mov_b32 s55, 0
s_load_dwordx4 s[24:27], s[48:49], s54
s_cmpk_eq_u32 s47, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 4                             // s52 = s24 / 16
s_and_b32 s50, 15, s24                             // s50 = s24 % 16
s_addc_u32 s52, s52, 0x0
s_lshr_b32 s53, s25, 4                             // s53 = s25 / 16
s_and_b32 s50, 15, s25                             // s50 = s25 % 16
s_addc_u32 s53, s53, 0x0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_mul_i32 s52, s52, s[sgprGSU]
s_add_u32 s55, s55, s52
s_cmp_lt_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_FOUND
s_add_u32 s54, s54, s14
s_load_dwordx4 s[24:27], s[48:49], s54
s_add_u32 s13, s13, 1
s_cmp_lt_u32 s13, s47
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 4                             // s52 = s24 / 16
s_and_b32 s50, 15, s24                             // s50 = s24 % 16
s_addc_u32 s52, s52, 0x0
s_lshr_b32 s53, s25, 4                             // s53 = s25 / 16
s_and_b32 s50, 15, s25                             // s50 = s25 % 16
s_addc_u32 s53, s53, 0x0
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_mul_i32 s52, s52, s[sgprGSU]
s_add_u32 s55, s55, s52

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s49, s13, 1
s_sub_u32 s48, s55, s52
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s47, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 120
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
/* init: add vgpr [0...11) to pool */
/* init: add vgpr [0...0) to pool */
/* init: add agpr [0...4) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
                                                   // 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v1, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x7, v1                          // 5. K offset: lrKOffset = kIdx * mStride(128)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x5, v1                          // 1. N offset: nOffset = nIdx * nStride(32)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_and_b32 v2, 63, v[vgprSerial]                    // 5. thread id in wave: wtid = tid % wavelength(64)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x3, v2                          // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 0, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 512                                 // LSU offset: stride = lsuStride(32)*(MT0(16) + PAD0(0))
v_mul_lo_u32 v2, s49, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v2, v0            // Final Offset: offset = (lro0+lsuoffset)*bpeDS(1)

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 0, v0                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 32                                  // LSU offset: stride = lsuStride(32) when umlds==True
v_mul_lo_u32 v0, s49, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v0, v1            // Final Offset: offset = (lro1+lsuoffset)*bpeDS(1)

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x200, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 2 */
/* v1 = A-unroll = serial/LVCA */
v_lshrrev_b32 v1, 1, v[vgprSerial]                 // v1 = v[vgprSerial] / 2
v_and_b32 v0, 1, v[vgprSerial]                     // v0 = v[vgprSerial] % 2
/* tile *= glvw */
v_lshlrev_b32 v0, 0x3, v0                          // v0 = v0 * 8
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 4 */
/* v3 = B-unroll = serial%LVCB */
v_and_b32 v5, 63, v[vgprSerial]                    // v5 = v[vgprSerial] % 64
v_lshrrev_b32 v2, 2, v5                            // v2 = v5 / 4
v_and_b32 v3, 3, v5                                // v3 = v5 % 4
v_readfirstlane_b32 s49, v[vgprSerial]             // WaveIdxWavefrontWidth
s_lshr_b32 s49, s49, 0x6                           // WaveId
s_mul_i32 s49, s49, 16                             // Each wave loads continuous lsp(16)*nrp(1) columns
v_add_u32 v2, s49, v2                              // Add back to column index
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x3, v3                          // v3 = v3 * 8
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x10, v4     // lwAL**(MTA + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpeDS(1)

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x20, v2     // lwBL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS(1)
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x200, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=512
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
s_waitcnt lgkmcnt(0)                               // wait for 44 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0x0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s48, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_mul_i32 s48, s48, s[sgprGSU]
v_cvt_f32_u32 v6, s48                              // s48 = s[sgprWorkGroup0] / s48
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s48
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s48
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s48
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s48
v_mul_u32_u24 v7, v6, s48                          // s48 = s[sgprWorkGroup0] / s48
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s48
v_cmpx_eq_u32 exec, v7, s48                        // s48 = s[sgprWorkGroup0] / s48
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s48
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s48
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s48
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s48, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s48, s48, s[sgprWorkGroup2]
s_mul_i32 s48, s48, s[sgprGSU]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v7, s[sgprNumWorkGroups0]      // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s48
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s48, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 48
.set sgprSrdB, 52
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprStaggerUIter, 15
.set sgprWrapUA, 60
.set sgprWrapUB, 62
.set sgprGlobalReadIncsA, 47
.set sgprGlobalReadIncsB, 64
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 8  // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 8  // pre-pad to make room for possible pointer shift
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
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v7           // remainder
s_mov_b32 s[sgprGSULog2BpeC], 0
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 0
s_mov_b32 s[sgprGSULog2BpeD], 0
label_GSU_End:
s_cmp_gt_i32 s[sgprWGM], 1                         // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_abs_i32 s[sgprWGM], s[sgprWGM]                   // abs(WGM)
s_cmp_le_i32 s[sgprWGM], 1                         // WGM <= 1 ?
s_cbranch_scc1 label_WGM                           // branch if WGM <= 1
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprWorkGroup0], v7                // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s65, v6                        // quotient
s_mul_i32 s68, s65, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s68, s[sgprWorkGroup0], s68              // WorkGroup0=remainder
s_mul_i32 s68, s68, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s68, s68, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups0]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprNumWorkGroups0], v7            // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s66, v6                        // quotient
s_mul_i32 s67, s[sgprWGM], s66                     // quotient * non-magic divisor
s_sub_u32 s67, s[sgprNumWorkGroups0], s67          // NumWorkGroups0=remainder
s_cmp_eq_u32 s67, 0                                // remainder == 0 ?
s_cmov_b32 s67, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s65, s66                              // blockId >= numFullBlocks ?
s_cselect_b32 s66, s67, s[sgprWGM]
v_cvt_f32_u32 v6, s66                              // s[sgprWorkGroup1] = s68 / s66
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s68 / s66
v_cvt_f32_u32 v7, s68                              // s[sgprWorkGroup1] = s68 / s66
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s68 / s66
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s68 / s66
v_mul_u32_u24 v7, v6, s66                          // s[sgprWorkGroup1] = s68 / s66
v_sub_u32 v7, s68, v7                              // s[sgprWorkGroup1] = s68 / s66
v_cmpx_eq_u32 exec, v7, s66                        // s[sgprWorkGroup1] = s68 / s66
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup1] = s68 / s66
v_mov_b32 v7, 0                                    // s[sgprWorkGroup0] = s68 % s66
s_mov_b64 exec, -1                                 // s[sgprWorkGroup1] = s68 / s66
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v7          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s66 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s68, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s65, s65, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s65 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
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
v_readfirstlane_b32 s65, v6                        // quotient
s_mul_i32 s68, s65, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s68, s[sgprWorkGroup1], s68              // WorkGroup1=remainder
s_mul_i32 s68, s68, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s68, s68, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
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
v_readfirstlane_b32 s66, v6                        // quotient
s_mul_i32 s67, s[sgprWGM], s66                     // quotient * non-magic divisor
s_sub_u32 s67, s[sgprNumWorkGroups1], s67          // NumWorkGroups1=remainder
s_cmp_eq_u32 s67, 0                                // remainder == 0 ?
s_cmov_b32 s67, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s65, s66                              // blockId >= numFullBlocks ?
s_cselect_b32 s66, s67, s[sgprWGM]
v_cvt_f32_u32 v6, s66                              // s[sgprWorkGroup0] = s68 / s66
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup0] = s68 / s66
v_cvt_f32_u32 v7, s68                              // s[sgprWorkGroup0] = s68 / s66
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup0] = s68 / s66
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup0] = s68 / s66
v_mul_u32_u24 v7, v6, s66                          // s[sgprWorkGroup0] = s68 / s66
v_sub_u32 v7, s68, v7                              // s[sgprWorkGroup0] = s68 / s66
v_cmpx_eq_u32 exec, v7, s66                        // s[sgprWorkGroup0] = s68 / s66
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup0] = s68 / s66
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s68 % s66
s_mov_b64 exec, -1                                 // s[sgprWorkGroup0] = s68 / s66
v_readfirstlane_b32 s[sgprWorkGroup0], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v7          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s66 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s68, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s65, s65, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s65 // wg1 += blockId * WGM
label_WGM:

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
v_mov_b32 v6, v0                                   // groA0I_0

/* global read addresses: tile offsets b */
v_mov_b32 v7, v2                                   // groB1J_0

/* global read addresses: unroll offsets a */
v_mov_b32 v8, v1                                   // groAL_0

/* global read addresses: unroll offsets b */
v_mov_b32 v9, v3                                   // groBL_0

/* global read addresses: shift a */
s_mul_i32 s65, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_sub_u32 s65, s[sgprSizeI], s65                   // edge = Size0I - WG*MT
s_sub_u32 s65, s65, 8                              // edge -= margin(8)
v_mov_b32 v10, s65                                 // edge vgpr = Size0I- WG*MT - margin(8)
v_min_i32 v6, v10, v6                              // offset = (offset < edge) ? offset(v6) : edge(v10)

/* global read addresses: final offsets a */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  6,  8, 23 // gROA_0_0_0_0

/* global read addresses: final offsets b */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  9,  7, 23 // gROB_0_0_0_0

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s69, s[sgprWorkGroup0], 16            // WorkGroup[01] * MT
s_mul_i32 s68, s[sgprWorkGroup0], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s67, 32, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s66, 32, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_mul_hi_u32 s67, s66, s[sgprStrideAL]             // tlu=1, scaled unroll-offset by stride
s_mul_i32 s66, s66, s[sgprStrideAL]                // tlu=1, scaled unroll-offset by stride
s_add_u32 s68, s68, s66                            // accum GsuOffset term to tilestart
s_addc_u32 s69, s69, s67                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitA+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitA+1], 0                 // init tensor size
s_sub_u32 s66, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s67, constStrideA0I, s66              // stride x (size-1)
s_mul_i32 s66, constStrideA0I, s66                 // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // sum tensor size
s_sub_u32 s66, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s67, s[sgprStrideAL], s66             // stride x (size-1)
s_mul_i32 s66, s[sgprStrideAL], s66                // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s68 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s69 // sub tileStart
                                                   // Set limit to use bytes (byte is 1, do nothing)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s67, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s66, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s68, s68, s66                            // accum wg term to tilestart
s_addc_u32 s69, s69, s67                           // accum wg term to tilestart
                                                   // tileStart *= BPE (multiplier is 1, do nothing)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s68    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s69   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s69, s[sgprWorkGroup1], 16            // WorkGroup[01] * MT
s_mul_i32 s68, s[sgprWorkGroup1], 16               // WorkGroup[01] * MT
s_mul_hi_u32 s69, s68, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s68, s68, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_hi_u32 s67, 32, s[sgprGSUSumIdx]             // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s66, 32, s[sgprGSUSumIdx]                // gsuOffset = DepthU*GSUSumIdx
s_add_u32 s68, s68, s66                            // accum GsuOffset term to tilestart
s_addc_u32 s69, s69, s67                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitB+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitB+1], 0                 // init tensor size
s_sub_u32 s66, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s67, constStrideBL, s66               // stride x (size-1)
s_mul_i32 s66, constStrideBL, s66                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // sum tensor size
s_sub_u32 s66, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s67, s[sgprStrideB1J], s66            // stride x (size-1)
s_mul_i32 s66, s[sgprStrideB1J], s66               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s68 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s69 // sub tileStart
                                                   // Set limit to use bytes (byte is 1, do nothing)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s67, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s66, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s68, s68, s66                            // accum wg term to tilestart
s_addc_u32 s69, s69, s67                           // accum wg term to tilestart
                                                   // tileStart *= BPE (multiplier is 1, do nothing)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s68    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s69   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: increments a */
s_mul_i32 s65, s[sgprGSU], DepthU*BpeAGR
s_mul_i32 s[sgprGlobalReadIncsA+0], s65, s[sgprStrideAL] // incrA unrollIdx)

/* global read addresses: increments b */
s_mul_i32 s65, s[sgprGSU], DepthU*BpeBGR
s_mov_b32 s[sgprGlobalReadIncsB+0], s65            // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 5 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 32
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
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_add_u32 s66, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s66                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s68, s[sgprStaggerU], 0x1f00
s_lshr_b32 s68, s68, 0x8
s_and_b32 s69, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s66, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s67, s66, s68                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s67           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s66, s66, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s67, s66, 1                              // staggerU mask
s_cmp_ge_u32 s66, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s67, 0          // set Mask
s_cmp_eq_u32 s69, 0x0
s_cbranch_scc1 label_StaggerUMapping_1
s_mov_b32 s66, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s69, 0x2000
s_cbranch_scc1 label_StaggerUMapping_2
s_mov_b32 s66, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s69, 0x4000
s_cbranch_scc1 label_StaggerUMapping_3
s_mov_b32 s66, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s69, 0x6000
s_cbranch_scc1 label_StaggerUMapping_4
s_mul_i32 s67, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s66, s66, s67
s_add_u32 s66, s66, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s69, 0x8000
s_cbranch_scc1 label_staggerInputEnd
s_mov_b32 s66, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s66 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s68 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s67, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s66, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s67, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s66, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_setprio 0                                        // optimization store
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
buffer_load_dwordx2 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx2 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* global read inc A loopL */
s_add_u32 s68, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s68              // Is this wrapIter? (pf)
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s68, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s68              // Is this wrapIter? (pf)
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

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


s_mul_i32 s68, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s67, s68, s[sgprStrideC1J]            // ScaleC s68 by Stride
s_mul_i32 s66, s68, s[sgprStrideC1J]               // ScaleC s68 by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s66    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s67   // add hi to SRD
s_mul_hi_u32 s67, s68, s[sgprStrideD1J]            // ScaleD s68 by Stride
s_mul_i32 s66, s68, s[sgprStrideD1J]               // ScaleD s68 by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s66    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s67   // add hi to SRD

s_mul_hi_u32 s67, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s66, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s67       // add hi to SRD
s_mul_hi_u32 s67, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s66, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[66:67], s[66:67], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi to SRD

s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s67, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s66, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s65, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s65, s65, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s68, s65, s[sgprStrideC1J]            // Free1
s_mul_i32 s65, s65, s[sgprStrideC1J]               // Free1
s_add_u32 s66, s66, s65                            // Free1
s_addc_u32 s67, s67, s68                           // Free1
s_sub_u32 s65, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s65, s65, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s68, s65, s[sgprStrideCK]             // Free2
s_mul_i32 s65, s65, s[sgprStrideCK]                // Free2
s_add_u32 s66, s66, s65                            // Free2
s_addc_u32 s67, s67, s68                           // Free2
s_lshl_b64 s[66:67], s[66:67], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s66        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s67       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF

/* initC: remove ValuC vgpr buffer [0...0) from pool */

/* initC: remove acc vgpr buffer [0...4) from pool */

/* initC: remove ValuA/B vgpr buffer [0...11) from pool */
v_accvgpr_write acc0, 0x0                          // initC
v_accvgpr_write acc1, 0x0                          // initC
v_accvgpr_write acc2, 0x0                          // initC
v_accvgpr_write acc3, 0x0                          // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_3EGGS9AO0TDK0GJM_0   // Only branch on scc1
s_getpc_b64 s[66:67]                               // addr of next instr
s_add_i32 s68, label_PrefetchGlobalLastIterEnd, 0x4 // target branch offset
s_add_u32 s66, s66, s68                            // add target branch offset
s_addc_u32 s67, s67, 0                             // add high and carry
s_setpc_b64 s[66:67]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_3EGGS9AO0TDK0GJM_0:
s_waitcnt vmcnt(0)                                 // 8wait for global read

/* local write a */
ds_write_b64 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+1] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0

/* local write b */
ds_write_b64 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* local write swap a */
v_xor_b32 v[vgprLocalWriteAddrA+0], 0x400, v[vgprLocalWriteAddrA+0] // swap Red Blk

/* local write swap b */
v_xor_b32 v[vgprLocalWriteAddrB+0], 0x400, v[vgprLocalWriteAddrB+0] // swap Red Blk
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_0                    // PGR=2 but only 1 loop
buffer_load_dwordx2 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx2 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
label_skipPGR2_0:

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_toPGR1_0                      // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 1wait for local write
// Skip barrier: NumThreads=644sync for global read

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:0  */
/*  mfmaIndex:0  */
ds_read_u8 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+1], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+1], v[vgprLocalReadAddrA] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_setprio 3                                        // store optimization
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b64 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+1] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
buffer_load_dwordx2 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b64 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
buffer_load_dwordx2 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0

/* local write swap offsets a */
v_xor_b32 v[vgprLocalWriteAddrA+0], 0x400, v[vgprLocalWriteAddrA+0] // swap Red Blk

/* local write swap offsets b */
v_xor_b32 v[vgprLocalWriteAddrB+0], 0x400, v[vgprLocalWriteAddrB+0] // swap Red Blk

/* local read swap offsets a */
v_xor_b32 v[vgprLocalReadAddrA], 0x400, v[vgprLocalReadAddrA] // swap Red Blk

/* local read swap offsets b */
v_xor_b32 v[vgprLocalReadAddrB], 0x400, v[vgprLocalReadAddrB] // swap Red Blk

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=2 newLR=0
/* pack scheduling: packAIdx:6, packBIdx:0 */
v_lshl_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], 8, v[vgprValuA_X0_I0+0] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+0], v[vgprValuA_X0_I0_D3+0], 8, v[vgprValuA_X0_I0_D2+0] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D2+0] // pack two half Vgpr to one Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1], 8, v[vgprValuA_X0_I0+1] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+1], v[vgprValuA_X0_I0_D3+1], 8, v[vgprValuA_X0_I0_D2+1] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D2+1] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_fp8_fp8 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=1 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop - Begin          */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 4wait for local write
// Skip barrier: NumThreads=64

/* iter 0 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:0  */
/*  mfmaIndex:0  */
ds_read_u8 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+1], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+1], v[vgprLocalReadAddrA] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_cselect_b32 s66, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s67, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_setprio 3                                        // store optimization
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b64 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+1] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0
/* sched write - iter 0 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b64 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* local write swap offsets a */
v_xor_b32 v[vgprLocalWriteAddrA+0], 0x400, v[vgprLocalWriteAddrA+0] // swap Red Blk

/* local write swap offsets b */
v_xor_b32 v[vgprLocalWriteAddrB+0], 0x400, v[vgprLocalWriteAddrB+0] // swap Red Blk

/* local read swap offsets a */
v_xor_b32 v[vgprLocalReadAddrA], 0x400, v[vgprLocalReadAddrA] // swap Red Blk

/* local read swap offsets b */
v_xor_b32 v[vgprLocalReadAddrB], 0x400, v[vgprLocalReadAddrB] // swap Red Blk

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=2 newLR=0
/* pack scheduling: packAIdx:6, packBIdx:0 */
v_lshl_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], 8, v[vgprValuA_X0_I0+0] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+0], v[vgprValuA_X0_I0_D3+0], 8, v[vgprValuA_X0_I0_D2+0] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D2+0] // pack two half Vgpr to one Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1], 8, v[vgprValuA_X0_I0+1] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+1], v[vgprValuA_X0_I0_D3+1], 8, v[vgprValuA_X0_I0_D2+1] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D2+1] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_fp8_fp8 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
s_setprio 0                                        // store optimization
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=1 */
label_toPGR1_0:
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/
s_waitcnt lgkmcnt(0)                               // 4wait for local write
// Skip barrier: NumThreads=64

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:0, lwEndMfmaIndex:0  */
/*  numMfmaForLR:0, syncPlrMfmaIndex:0  */
/*  mfmaIndex:0  */
ds_read_u8 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+1], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+1], v[vgprLocalReadAddrA] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
s_setprio 3                                        // store optimization
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
/* pack scheduling: packAIdx:6, packBIdx:0 */
v_lshl_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], 8, v[vgprValuA_X0_I0+0] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+0], v[vgprValuA_X0_I0_D3+0], 8, v[vgprValuA_X0_I0_D2+0] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D2+0] // pack two half Vgpr to one Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1], 8, v[vgprValuA_X0_I0+1] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+1], v[vgprValuA_X0_I0_D3+1], 8, v[vgprValuA_X0_I0_D2+1] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D2+1] // pack two half Vgpr to one Vgpr
s_nop 1                                            // VALU packing writes to be consumed by matrix instruction
v_mfma_f32_16x16x32_fp8_fp8 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=8 */
/* dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=1 */
label_PrefetchGlobalLastIterEnd:

/******************************************/
/* Tail Loop                              */
/******************************************/

/* Tail: add ValuA/B vgpr buffer [0...11) to pool */

/* local write reset offsets a */
v_and_b32 v[vgprLocalWriteAddrA], 0xf003ff, v[vgprLocalWriteAddrA] // reset to Red

/* local write reset offsets b */
v_and_b32 v[vgprLocalWriteAddrB], 0xf003ff, v[vgprLocalWriteAddrB] // reset to Red

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 31, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 32
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0x0                // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s66, 3, s[sgprStaggerUIter]
s_mul_hi_i32 s67, s66, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_sub_u32 s66, s66, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s67 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s66, 3, s[sgprStaggerUIter]
s_mul_hi_i32 s67, s66, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s66, s66, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_sub_u32 s66, s66, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s67, s67, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s66        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s67       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s66 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s67 // limit -= inc)
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
s_waitcnt vmcnt(6)
v_lshlrev_b32 v0, 0x8, v0                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v0      // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v1      // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v2, 0x8, v2                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v2      // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v4, 0x8, v4                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v4      // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v5      // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v6, 0x8, v6                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v6      // pack a sub 8-bit with dest

/* Update M0 for DTLDS */

/* global read B */
/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v0, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:1 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v1, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v2, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:3 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v4, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:5 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v5, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v6, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:7 // load one buffer value
s_waitcnt vmcnt(6)
v_lshlrev_b32 v0, 0x8, v0                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v0      // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v1      // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v2, 0x8, v2                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v2      // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v4, 0x8, v4                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v4      // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v5      // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v6, 0x8, v6                          // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v6      // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip barrier: NumThreads=64

/* local write a */
ds_write_b64 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+1] offset:0 // lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0

/* local write b */
ds_write_b64 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+1] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0

/* Recalc local read offsets */
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip barrier: NumThreads=64

/* local read reset offsets a */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrA], 0x3ff, v[vgprLocalReadAddrA] // reset Red,Blk -> Red

/* local read reset offsets b */

/* localReadResetOffsets */
/* handled internally */
v_and_b32 v[vgprLocalReadAddrB], 0x3ff, v[vgprLocalReadAddrB] // reset Red,Blk -> Red

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
label_TailLoopBeginL:

/* Tail: remove ValuA/B vgpr buffer [0...11) from pool */

/* Tail: add address/G2L vgpr [11...22) to pool */

/* local read a */
ds_read_u8 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+0], v[vgprLocalReadAddrA] offset:16 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=1 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+0], v[vgprLocalReadAddrA] offset:32 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=2 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+0], v[vgprLocalReadAddrA] offset:48 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=3 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0+1], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=4 oIdx=0 buffer=0 iui=0
ds_read_u8 v[vgprValuA_X0_I0_D1+1], v[vgprLocalReadAddrA] offset:80 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=5 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D2+1], v[vgprLocalReadAddrA] offset:96 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=6 oIdx=0 buffer=0 iui=0
ds_read_u8_d16_hi v[vgprValuA_X0_I0_D3+1], v[vgprLocalReadAddrA] offset:112 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=7 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=16 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
s_mov_b32 s5, 0x200                                // inc
v_add_co_u32 v[vgprLocalReadAddrA], vcc, s5, v[vgprLocalReadAddrA] // lrA += 512 ((MT+PAD)*bpeDS)

/* local read inc b */
s_mov_b32 s5, 0x20                                 // inc
v_add_co_u32 v[vgprLocalReadAddrB], vcc, s5, v[vgprLocalReadAddrB] // lrB += 32 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_lshl_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D1+0], 8, v[vgprValuA_X0_I0+0] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+0], v[vgprValuA_X0_I0_D3+0], 8, v[vgprValuA_X0_I0_D2+0] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0+0], v[vgprValuA_X0_I0_D2+0] // pack two half Vgpr to one Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D1+1], 8, v[vgprValuA_X0_I0+1] // pack two int8 Vgpr to one half Vgpr
v_lshl_or_b32 v[vgprValuA_X0_I0_D2+1], v[vgprValuA_X0_I0_D3+1], 8, v[vgprValuA_X0_I0_D2+1] // pack two int8 Vgpr to one half Vgpr
v_or_b32 v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0+1], v[vgprValuA_X0_I0_D2+1] // pack two half Vgpr to one Vgpr
v_and_b32 v11, 63, v[vgprSerial]                   // v11 = v[vgprSerial] % 64
v_lshrrev_b32 v11, 4, v11                          // v11 = v11 / 16
v_lshlrev_b32 v11, 0x3, v11                        // v11 = v11 * 8
v_cmp_ge_i32 s[66:67], v11, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0], v[vgprValuA_X0_I0+0+0], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+1], v[vgprValuA_X0_I0+0+1], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0], v[vgprValuB_X0_I0+0+0], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+1], v[vgprValuB_X0_I0+0+1], 0x0, s[66:67] // set 0 if K_idx >= sizeL
v_sub_u32 v11, s[sgprLoopCounterL], v11            // get distance between size and k index
v_cmp_lt_i32 s[66:67], v11, 8                      // set partial 0 if distance less than input per thread
s_and_b32 s65, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s65, 8, s65                              // use shift to fill 0 for outside element
s_lshl_b32 s65, s65, 3                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[12:13], s65, v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v12, s[66:67]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v13, s[66:67]
v_lshlrev_b64 v[12:13], s65, v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v12, s[66:67]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v13, s[66:67]
s_nop 1
v_mfma_f32_16x16x32_fp8_fp8 acc[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], acc[0:3] // left value = acc[0+0:3+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:

/* Tail: remove address/G2L [11...22) from pool */
label_Summation_End_83JUN68H8G0WG15W_0:
s_setprio 0                                        // optimization store
/* endSummation: add vgpr [0...22) to pool */
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsB, UNDEF
/* load store sgprs */
.set sgprAddressScaleA, 48
.set sgprAddressScaleB, 50
.set sgprAddressScaleC, 52
.set sgprAddressScaleD, 54
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc0 label_GSU_4                         // branch if GSU != 1
s_load_dwordx8 s[48:55], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
label_GSU_4:
.set sgprScaleD, 10

/* Mapping of Acc register -> C Vgpr register */

/* shift vector components d0 */
v_mov_b32 v3, s[sgprWorkGroup0]
v_mul_i32_i24 v3, -0x10, v3                        // wg*MT
v_add_co_u32 v3, vcc, s[sgprSizesFree+0], v3       // wgMT = Size - wg*MT
v_mov_b32 v4, 0x10                                 // MT
v_cmp_lt_u32 s[32:33], v3, v4                      // wgMT < MT
v_cndmask_b32 v3, v4, v3, s[32:33]                 // wgMT = (wgMT < MT) ? wgMT : MT
v_lshrrev_b32 v5, 6, v[vgprSerial]                 // v5 = v[vgprSerial] / 64
v_and_b32 v5, 0, v5                                // v5 = v5 % 1
v_lshrrev_b32 v6, 4, v3                            // v6 = v3 / 16
v_and_b32 v6, 0, v6                                // v6 = v6 % 1
v_cmp_eq_u32 s[32:33], v6, v5                      // wave_id == block_belong_to_wave?
v_cndmask_b32 v3, v4, v3, s[32:33]                 // wgMT = (wgMT < MT) ? wgMT : MT

/* mbReg: which mb block need to shift, mb(matrixInstCoal(16) * VectorWidth(1)) */
v_lshrrev_b32 v4, 4, v3                            // v4 = v3 / 16
v_lshlrev_b32 v6, 0x0, v5                          // v6 = v5 * 1
v_sub_u32 v4, v4, v6

/* gbReg: glvw block id */
v_lshrrev_b32 v6, 3, v3                            // v6 = v3 / 8

/* tgbReg: glvw block id */
v_lshrrev_b32 v7, 0, v[vgprSerial]                 // v7 = v[vgprSerial] / 1
v_and_b32 v7, 15, v7                               // v7 = v7 % 16
                                                   // v7 = v7 * 1 (multiplier is 1, do nothing)
v_lshrrev_b32 v7, 3, v7                            // v7 = v7 / 8
v_lshlrev_b32 v5, 0x1, v5                          // v5 = v5 * 2
v_add_co_u32 v7, vcc, v5, v7                       // tgbReg = (tid_coal * continOut) / GLVW
v_sub_u32 v6, v6, v7

/* vwReg: glvw in which vw block? */
v_and_b32 v5, 0, v3                                // permute register between threads
v_lshrrev_b32 v5, 3, v5                            // permute register between threads

/* rReg : reminder of M_size % GlobalReadVectorWidth */
v_and_b32 v7, 7, v3                                // v7 = v3 % 8
v_cmp_eq_u32 vcc, v7, 0x1                          // wgMT%VW == 1
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_0 // branch to shift d0 r=1
v_cmp_eq_u32 vcc, v7, 0x2                          // wgMT%VW == 2
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_0 // branch to shift d0 r=2
v_cmp_eq_u32 vcc, v7, 0x3                          // wgMT%VW == 3
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_0 // branch to shift d0 r=3
v_cmp_eq_u32 vcc, v7, 0x4                          // wgMT%VW == 4
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_0 // branch to shift d0 r=4
v_cmp_eq_u32 vcc, v7, 0x5                          // wgMT%VW == 5
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_0 // branch to shift d0 r=5
v_cmp_eq_u32 vcc, v7, 0x6                          // wgMT%VW == 6
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_0 // branch to shift d0 r=6
v_cmp_eq_u32 vcc, v7, 0x7                          // wgMT%VW == 7
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_0 // branch to shift d0 r=7
s_branch label_ShiftVectorComponents0_GLVW0_0      // no shifting

/******************************************/
/* shift d0 r=1                           */
/******************************************/
label_ShiftVectorComponents0_GLVW1_0:
v_cmp_eq_u32 vcc, v4, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_0 // branch to shift d0 r1 mb0

/******************************************/
/* shift d0 r=2                           */
/******************************************/
label_ShiftVectorComponents0_GLVW2_0:
v_cmp_eq_u32 vcc, v4, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0_0 // branch to shift d0 r2 mb0

/******************************************/
/* shift d0 r=3                           */
/******************************************/
label_ShiftVectorComponents0_GLVW3_0:
v_cmp_eq_u32 vcc, v4, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0_0 // branch to shift d0 r3 mb0

/******************************************/
/* shift d0 r=4                           */
/******************************************/
label_ShiftVectorComponents0_GLVW4_0:
v_cmp_eq_u32 vcc, v4, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0_0 // branch to shift d0 r4 mb0

/******************************************/
/* shift d0 r=5                           */
/******************************************/
label_ShiftVectorComponents0_GLVW5_0:
v_cmp_eq_u32 vcc, v4, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0_0 // branch to shift d0 r5 mb0

/******************************************/
/* shift d0 r=6                           */
/******************************************/
label_ShiftVectorComponents0_GLVW6_0:
v_cmp_eq_u32 vcc, v4, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0_0 // branch to shift d0 r6 mb0

/******************************************/
/* shift d0 r=7                           */
/******************************************/
label_ShiftVectorComponents0_GLVW7_0:
v_cmp_eq_u32 vcc, v4, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0_0 // branch to shift d0 r7 mb0

/******************************************/
/* shift d0 r=1 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_0:  /// r1 mb0
v_cmp_eq_u32 vcc, v5, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_VW0_0 // branch to shift d0 r1 mb0 vw0

/******************************************/
/* shift d0 r=2 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0_0:  /// r2 mb0
v_cmp_eq_u32 vcc, v5, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0_VW0_0 // branch to shift d0 r2 mb0 vw0

/******************************************/
/* shift d0 r=3 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0_0:  /// r3 mb0
v_cmp_eq_u32 vcc, v5, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0_VW0_0 // branch to shift d0 r3 mb0 vw0

/******************************************/
/* shift d0 r=4 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM0_0:  /// r4 mb0
v_cmp_eq_u32 vcc, v5, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0_VW0_0 // branch to shift d0 r4 mb0 vw0

/******************************************/
/* shift d0 r=5 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM0_0:  /// r5 mb0
v_cmp_eq_u32 vcc, v5, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0_VW0_0 // branch to shift d0 r5 mb0 vw0

/******************************************/
/* shift d0 r=6 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM0_0:  /// r6 mb0
v_cmp_eq_u32 vcc, v5, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0_VW0_0 // branch to shift d0 r6 mb0 vw0

/******************************************/
/* shift d0 r=7 mb=0                      */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM0_0:  /// r7 mb0
v_cmp_eq_u32 vcc, v5, 0x0
s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0_VW0_0 // branch to shift d0 r7 mb0 vw0

/******************************************/
/* shift d0 r=1 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW1_BM0_VW0_0:  /// r1 mb0 vw0
s_mov_b32 s32, 0
v_cmpx_eq_u32 s[32:33], v6, s32                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 1 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:28               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc1                        // glvw 1 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:28               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc2                        // glvw 1 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:28               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc3                        // glvw 1 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:28               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
s_mov_b64 s[32:33], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[32:33]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=2 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW2_BM0_VW0_0:  /// r2 mb0 vw0
s_mov_b32 s32, 0
v_cmpx_eq_u32 s[32:33], v6, s32                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 2 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:24               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc1                        // glvw 2 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:24               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc2                        // glvw 2 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:24               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc3                        // glvw 2 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:24               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
s_mov_b64 s[32:33], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[32:33]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=3 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW3_BM0_VW0_0:  /// r3 mb0 vw0
s_mov_b32 s32, 0
v_cmpx_eq_u32 s[32:33], v6, s32                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 3 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:20               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc1                        // glvw 3 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:20               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc2                        // glvw 3 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:20               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc3                        // glvw 3 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:20               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
s_mov_b64 s[32:33], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[32:33]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=4 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW4_BM0_VW0_0:  /// r4 mb0 vw0
s_mov_b32 s32, 0
v_cmpx_eq_u32 s[32:33], v6, s32                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 4 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:16               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc1                        // glvw 4 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:16               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc2                        // glvw 4 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:16               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc3                        // glvw 4 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:16               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
s_mov_b64 s[32:33], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[32:33]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=5 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW5_BM0_VW0_0:  /// r5 mb0 vw0
s_mov_b32 s32, 0
v_cmpx_eq_u32 s[32:33], v6, s32                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 5 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:12               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc1                        // glvw 5 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:12               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc2                        // glvw 5 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:12               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc3                        // glvw 5 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:12               // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
s_mov_b64 s[32:33], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[32:33]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=6 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW6_BM0_VW0_0:  /// r6 mb0 vw0
s_mov_b32 s32, 0
v_cmpx_eq_u32 s[32:33], v6, s32                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 6 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:8                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc1                        // glvw 6 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:8                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc2                        // glvw 6 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:8                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc3                        // glvw 6 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:8                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
s_mov_b64 s[32:33], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[32:33]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting


/******************************************/
/* shift d0 r=7 mb=0 vw0                  */
/******************************************/
label_ShiftVectorComponents0_GLVW7_BM0_VW0_0:  /// r7 mb0 vw0
s_mov_b32 s32, 0
v_cmpx_eq_u32 s[32:33], v6, s32                    // is thread in edge glvw region
v_and_b32 v0, 63, v[vgprSerial]                    // permute register between threads
v_lshlrev_b32 v0, 2, v0                            // permute register between threads
v_accvgpr_read_b32 v7, acc0                        // glvw 7 mb 0 tt1 0 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc0, v7
v_accvgpr_read_b32 v7, acc1                        // glvw 7 mb 0 tt1 1 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc1, v7
v_accvgpr_read_b32 v7, acc2                        // glvw 7 mb 0 tt1 2 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc2, v7
v_accvgpr_read_b32 v7, acc3                        // glvw 7 mb 0 tt1 3 r 0
s_nop 1                                            // v_accvgpr read vgpr after write vgpr: 2 wait states
ds_bpermute_b32 v7, v0, v7 offset:4                // permute edge values
s_waitcnt 0                                        // (Wait all)
v_accvgpr_write_b32 acc3, v7
s_mov_b64 s[32:33], 0xFFFFFFFFFFFFFFFF             // to restore all threads active
s_or_saveexec_b64 vcc, s[32:33]                    // all threads active
s_branch label_ShiftVectorComponents0_GLVW0_0      // done shifting

label_ShiftVectorComponents0_GLVW0_0:  /// end shift0

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // v4 = v[vgprSerial] / 64
v_lshrrev_b32 v5, 0, v4                            // v5 = v4 / 1
v_mul_lo_u32 v5, 0x10, v5                          // wave coordination offset 1
v_and_b32 v1, 63, v[vgprSerial]                    // v1 = v[vgprSerial] % 64
v_lshrrev_b32 v1, 4, v1                            // v1 = v1 / 16
v_lshlrev_b32 v1, 0x2, v1                          // thread0 * continuous_output
v_add_lshl_u32 v1, v5, v1, 0                       // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v2, v1, s[sgprStrideC1J]              //  offset 1
v_mul_lo_u32 v3, v1, s[sgprStrideD1J]              //  offset 1
v_and_b32 v0, 0, v4                                // v0 = v4 % 1
v_mul_lo_u32 v0, 0x10, v0                          // wave coordination offset 0
v_and_b32 v5, 15, v[vgprSerial]                    // v5 = v[vgprSerial] % 16
v_add_lshl_u32 v0, v5, v0, 0                       // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s5, 16, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_u32 v0, s5, v0                               // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s5, 16, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v1, s5, v1                               // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

.set vgprAmaxOut, 42
.set vgprAmaxOutB, 43

v_mov_b32 v[vgprAmaxOut], 0


/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 32 bytes of kern args.
s_cmp_eq_u32 s[sgprGSU], 1                         // GSU == 1 ?
s_cbranch_scc1 label_GSU_5                         // branch if GSU == 1
s_and_b32 s32, 15, s[sgprSizeI]                    // s32 = s[sgprSizeI] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
s_and_b32 s32, 15, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
label_GW_B0_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=31 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 biasDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_lshl_u32 v10, v3, v0, 0x2                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
v_mov_b32 v7, 0x207                                // Nan and +/- inf
v_mov_b32 v9, 0x43700000                           // Fp8 Max value 240 as float32
v_mov_b32 v8, 0xc3700000                           // Fp8 Min value -240 as float32
buffer_store_dword v12, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s32, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v13, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s32, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v14, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s32, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v15, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_B0_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 biasDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v18, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v10, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v18, v10, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v12, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v18, v12, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v14, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v18, v14, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v16, v3, v0, 0x2                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, v18, v16, s[60:61]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+15], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+17], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */

/* apply mask, calc new C and issue writes */
v_mov_b32 v7, 0x207                                // Nan and +/- inf
v_mov_b32 v9, 0x43700000                           // Fp8 Max value 240 as float32
v_mov_b32 v8, 0xc3700000                           // Fp8 Min value -240 as float32
buffer_store_dword v11, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v13, v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v15, v14, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v17, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:
s_getpc_b64 s[32:33]                               // addr of next instr
s_add_i32 s34, label_KernelEnd, 0x4                // target branch offset
s_add_u32 s32, s32, s34                            // add target branch offset
s_addc_u32 s33, s33, 0                             // add high and carry
s_setpc_b64 s[32:33]                               // branch to label_KernelEnd
label_GSU_5:
s_mov_b32 s5, 1.0                                  // init as 1
s_cmp_eq_u64 s[sgprAddressScaleA:sgprAddressScaleA+1], 0 // s[AddressScaleA] == 0 ?
s_cbranch_scc1 label_ScaleAValid                   // branch if s[AddressScaleA] == 0
s_load_dword s5, s[sgprAddressScaleA:sgprAddressScaleA+1], 0 // load scaleA
label_ScaleAValid:
s_mov_b32 s12, 1.0                                 // init as 1
s_cmp_eq_u64 s[sgprAddressScaleB:sgprAddressScaleB+1], 0 // s[AddressScaleB] == 0 ?
s_cbranch_scc1 label_ScaleBValid                   // branch if s[AddressScaleB] == 0
s_load_dword s12, s[sgprAddressScaleB:sgprAddressScaleB+1], 0 // load scaleB
label_ScaleBValid:
s_mov_b32 s[sgprScaleD], 1.0                       // init as 1
s_mov_b32 s[sgprScaleD+1], 1.0                     // init as 1
s_cmp_eq_u64 s[sgprAddressScaleD:sgprAddressScaleD+1], 0 // s[AddressScaleD] == 0 ?
s_cbranch_scc1 label_ScaleDValid                   // branch if s[AddressScaleD] == 0
s_load_dword s[sgprScaleD], s[sgprAddressScaleD:sgprAddressScaleD+1], 0 // load scaleD
label_ScaleDValid:
s_mov_b32 s15, 1.0                                 // init as 1
s_cmp_eq_u64 s[sgprAddressScaleC:sgprAddressScaleC+1], 0 // s[AddressScaleC] == 0 ?
s_cbranch_scc1 label_ScaleCValid                   // branch if s[AddressScaleC] == 0
s_load_dword s15, s[sgprAddressScaleC:sgprAddressScaleC+1], 0 // load scaleC
label_ScaleCValid:
v_mov_b32 v4, s[sgprAlpha]
s_waitcnt lgkmcnt(0)                               // wait for scaleAB load
v_mul_f32 v4, v4, s5
v_mul_f32 v4, v4, s12
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprAlpha], v4               // Update Alpha
v_mov_b32 v4, s[sgprBeta]
v_mul_f32 v4, v4, s15
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprBeta], v4                // Update Beta
s_mov_b32 s[sgprScaleD+1], s[sgprScaleD]
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_GW_Beta_1                     // Branch if Beta is not zero

s_and_b32 s32, 15, s[sgprSizeI]                    // s32 = s[sgprSizeI] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
s_and_b32 s32, 15, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
label_GW_B0_E0_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=31 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 biasDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_lshl_u32 v10, v3, v0, 0x0                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+14], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+15], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v7, 0x207                                // Nan and +/- inf
v_mov_b32 v9, 0x43700000                           // Fp8 Max value 240 as float32
v_mov_b32 v8, 0xc3700000                           // Fp8 Min value -240 as float32
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+12])
v_mul_f32 v[vgprValuC+12], v[vgprValuC+12], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+12], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+12], v8, v9
v_cndmask_b32 v[vgprValuC+12], v6, v[vgprValuC+12], s[32:33]
v_cvt_pk_fp8_f32 v12, v[vgprValuC+12], v[vgprValuC+12] op_sel:[0,0,0]
buffer_store_byte v12, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+13])
v_mul_f32 v[vgprValuC+13], v[vgprValuC+13], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+13], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+13], v8, v9
v_cndmask_b32 v[vgprValuC+13], v6, v[vgprValuC+13], s[32:33]
v_cvt_pk_fp8_f32 v13, v[vgprValuC+13], v[vgprValuC+13] op_sel:[0,0,0]
s_lshl_b32 s32, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v13, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+14])
v_mul_f32 v[vgprValuC+14], v[vgprValuC+14], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+14], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+14], v8, v9
v_cndmask_b32 v[vgprValuC+14], v6, v[vgprValuC+14], s[32:33]
v_cvt_pk_fp8_f32 v14, v[vgprValuC+14], v[vgprValuC+14] op_sel:[0,0,0]
s_lshl_b32 s32, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v14, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+15])
v_mul_f32 v[vgprValuC+15], v[vgprValuC+15], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+15], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+15], v8, v9
v_cndmask_b32 v[vgprValuC+15], v6, v[vgprValuC+15], s[32:33]
v_cvt_pk_fp8_f32 v15, v[vgprValuC+15], v[vgprValuC+15] op_sel:[0,0,0]
s_lshl_b32 s32, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v15, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_E1_1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 biasDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v18, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v10, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v18, v10, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v12, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v12, v18, v12, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v14, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v14, v18, v14, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v16, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, v18, v16, s[60:61]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+11], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+13], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+15], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+17], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v7, 0x207                                // Nan and +/- inf
v_mov_b32 v9, 0x43700000                           // Fp8 Max value 240 as float32
v_mov_b32 v8, 0xc3700000                           // Fp8 Min value -240 as float32
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+11])
v_mul_f32 v[vgprValuC+11], v[vgprValuC+11], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+11], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+11], v8, v9
v_cndmask_b32 v[vgprValuC+11], v6, v[vgprValuC+11], s[56:57]
v_cvt_pk_fp8_f32 v11, v[vgprValuC+11], v[vgprValuC+11] op_sel:[0,0,0]
buffer_store_byte v11, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+13])
v_mul_f32 v[vgprValuC+13], v[vgprValuC+13], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+13], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+13], v8, v9
v_cndmask_b32 v[vgprValuC+13], v6, v[vgprValuC+13], s[56:57]
v_cvt_pk_fp8_f32 v13, v[vgprValuC+13], v[vgprValuC+13] op_sel:[0,0,0]
buffer_store_byte v13, v12, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+15])
v_mul_f32 v[vgprValuC+15], v[vgprValuC+15], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+15], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+15], v8, v9
v_cndmask_b32 v[vgprValuC+15], v6, v[vgprValuC+15], s[56:57]
v_cvt_pk_fp8_f32 v15, v[vgprValuC+15], v[vgprValuC+15] op_sel:[0,0,0]
buffer_store_byte v15, v14, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+17])
v_mul_f32 v[vgprValuC+17], v[vgprValuC+17], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+17], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+17], v8, v9
v_cndmask_b32 v[vgprValuC+17], v6, v[vgprValuC+17], s[56:57]
v_cvt_pk_fp8_f32 v17, v[vgprValuC+17], v[vgprValuC+17] op_sel:[0,0,0]
buffer_store_byte v17, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_Beta_1:
s_and_b32 s32, 15, s[sgprSizeI]                    // s32 = s[sgprSizeI] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s33                // wg0 >= nwg0-1 ?
s_cselect_b32 s32, s32, 0                          // set rMT0
s_cmpk_gt_u32 s32, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
s_and_b32 s32, 15, s[sgprSizeJ]                    // s32 = s[sgprSizeJ] % 16
s_add_u32 s33, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s33                // wg1 >= nwg1-1
s_cselect_b32 s32, s32, 0                          // set rMT1
s_cmpk_gt_u32 s32, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
label_GW_B1_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=15 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 biasDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v11, v2, v0, 0x0                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
buffer_load_ubyte_d16 v12, v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s32, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v14, v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s32, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v16, v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s32, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v18, v11, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v10, v3, v0, 0x0                    // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0
v_accvgpr_read_b32 v[vgprValuC+13], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+15], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+17], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+19], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha

/* apply mask, calc new C and issue writes */
v_mov_b32 v7, 0x207                                // Nan and +/- inf
v_mov_b32 v9, 0x43700000                           // Fp8 Max value 240 as float32
v_mov_b32 v8, 0xc3700000                           // Fp8 Min value -240 as float32

s_waitcnt vmcnt(3)                                 // vmcnt(3) = 4 - 1 (beta) (interleaved)
v_cvt_f32_fp8 v4, v12 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+13], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+13])
v_mul_f32 v[vgprValuC+13], v[vgprValuC+13], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+13], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+13], v8, v9
v_cndmask_b32 v[vgprValuC+13], v6, v[vgprValuC+13], s[32:33]
v_cvt_pk_fp8_f32 v13, v[vgprValuC+13], v[vgprValuC+13] op_sel:[0,0,0]
buffer_store_byte v13, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(2) = 4 - 2 (beta) (interleaved)
v_cvt_f32_fp8 v4, v14 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+15], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+15])
v_mul_f32 v[vgprValuC+15], v[vgprValuC+15], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+15], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+15], v8, v9
v_cndmask_b32 v[vgprValuC+15], v6, v[vgprValuC+15], s[32:33]
v_cvt_pk_fp8_f32 v15, v[vgprValuC+15], v[vgprValuC+15] op_sel:[0,0,0]
s_lshl_b32 s32, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v15, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(1) = 4 - 3 (beta) (interleaved)
v_cvt_f32_fp8 v4, v16 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+17], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+17])
v_mul_f32 v[vgprValuC+17], v[vgprValuC+17], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+17], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+17], v8, v9
v_cndmask_b32 v[vgprValuC+17], v6, v[vgprValuC+17], s[32:33]
v_cvt_pk_fp8_f32 v17, v[vgprValuC+17], v[vgprValuC+17] op_sel:[0,0,0]
s_lshl_b32 s32, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v17, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(3)                                 // vmcnt(0) = 4 - 4 (beta) (interleaved)
v_cvt_f32_fp8 v4, v18 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+19], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+19])
v_mul_f32 v[vgprValuC+19], v[vgprValuC+19], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[32:33], v[vgprValuC+19], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+19], v8, v9
v_cndmask_b32 v[vgprValuC+19], v6, v[vgprValuC+19], s[32:33]
v_cvt_pk_fp8_f32 v19, v[vgprValuC+19], v[vgprValuC+19] op_sel:[0,0,0]
s_lshl_b32 s32, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s32        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v19, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B1_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=11 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 biasDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v23, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v10, v2, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v23, v10, s[60:61]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v11, v10, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v10, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v10, v23, v10, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v13, v2, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v13, v23, v13, s[60:61]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v14, v13, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v13, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v13, v23, v13, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v16, v2, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, v23, v16, s[60:61]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v17, v16, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v16, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v16, v23, v16, s[60:61]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v1, vcc, v1, 1                        // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v2, v2, s[sgprStrideC1J]                 // ROWINC- Move cinRowPtr to next row
v_add_u32 v3, v3, s[sgprStrideD1J]                 // Move coutRowPtrD to next row
v_cmp_lt_u32 s[56:57], v0, s[sgprSizeI]            // coord0 < size0
v_cmp_lt_u32 s[60:61], v1, s[sgprSizeJ]            // coord1 < size1
s_and_b64 s[60:61], s[56:57], s[60:61]             // in0 && in1
v_add_lshl_u32 v19, v2, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v19, v23, v19, s[60:61]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v20, v19, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v19, v3, v0, 0x0                    // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v19, v23, v19, s[60:61]              // LDD clip if OOB. offset
v_accvgpr_read_b32 v[vgprValuC+12], acc0           // copy acc to vreg[0]
v_accvgpr_read_b32 v[vgprValuC+15], acc1           // copy acc to vreg[1]
v_accvgpr_read_b32 v[vgprValuC+18], acc2           // copy acc to vreg[2]
v_accvgpr_read_b32 v[vgprValuC+21], acc3           // copy acc to vreg[3]

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0)] */
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
s_waitcnt vmcnt(0)                                 // wait for Beta

/* apply mask, calc new C and issue writes */
v_mov_b32 v7, 0x207                                // Nan and +/- inf
v_mov_b32 v9, 0x43700000                           // Fp8 Max value 240 as float32
v_mov_b32 v8, 0xc3700000                           // Fp8 Min value -240 as float32
v_cvt_f32_fp8 v4, v11 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+12], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+12])
v_mul_f32 v[vgprValuC+12], v[vgprValuC+12], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+12], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+12], v8, v9
v_cndmask_b32 v[vgprValuC+12], v6, v[vgprValuC+12], s[56:57]
v_cvt_pk_fp8_f32 v12, v[vgprValuC+12], v[vgprValuC+12] op_sel:[0,0,0]
buffer_store_byte v12, v10, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f32_fp8 v4, v14 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+15], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+15])
v_mul_f32 v[vgprValuC+15], v[vgprValuC+15], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+15], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+15], v8, v9
v_cndmask_b32 v[vgprValuC+15], v6, v[vgprValuC+15], s[56:57]
v_cvt_pk_fp8_f32 v15, v[vgprValuC+15], v[vgprValuC+15] op_sel:[0,0,0]
buffer_store_byte v15, v13, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f32_fp8 v4, v17 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+18], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+18])
v_mul_f32 v[vgprValuC+18], v[vgprValuC+18], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+18], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+18], v8, v9
v_cndmask_b32 v[vgprValuC+18], v6, v[vgprValuC+18], s[56:57]
v_cvt_pk_fp8_f32 v18, v[vgprValuC+18], v[vgprValuC+18] op_sel:[0,0,0]
buffer_store_byte v18, v16, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_cvt_f32_fp8 v4, v20 src0_sel:BYTE_0
s_nop 0
v_fmac_f32 v[vgprValuC+21], v4, s[sgprBeta]        // finalSum = sum*alpha + C*beta
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValuC+21])
v_mul_f32 v[vgprValuC+21], v[vgprValuC+21], s[sgprScaleD] // result *= ScaleD
v_cmp_class_f32 s[56:57], v[vgprValuC+21], v7      // Nan and +/- inf
v_med3_f32 v6, v[vgprValuC+21], v8, v9
v_cndmask_b32 v[vgprValuC+21], v6, v[vgprValuC+21], s[56:57]
v_cvt_pk_fp8_f32 v21, v[vgprValuC+21], v[vgprValuC+21] op_sel:[0,0,0]
buffer_store_byte v21, v19, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:


/** AMAXD **/
.set vgprWidx, 10
.set vgprOffset, 11
.set vgprTmp, 12
.set vgprValue, 13

.set sgprSrc, 24
.set sgprDst, 28
.set sgprOffset, 32
.set sgprTmp, 34

.set sgprAMAXOut, 60
.set sgprAddressWk, 62
.set sgprAddressSy, 64
.set sgprNumGroup, 66
.set sgprWGIdx, 68

s_load_dwordx4 s[60:63], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78
s_load_dwordx2 s[64:65], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x88
// s_load_dword s[66], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
s_mul_i32 s[sgprNumGroup], s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_mul_i32 s[sgprWGIdx], s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_add_i32 s[sgprWGIdx], s[sgprWGIdx], s[sgprWorkGroup0]

/// reduction
/* intra_wave_reduction */
s_mov_b32 s[sgprTmp], 1
label_permute_middle:  /// permute_middle

v_add_u32 v[vgprTmp], s[sgprTmp], v[vgprSerial]
v_and_b32 v[vgprTmp], 63, v[vgprTmp]
v_lshlrev_b32 v[vgprTmp], 2, v[vgprTmp]

ds_bpermute_b32 v[vgprAmaxOutB], v[vgprTmp], v[vgprAmaxOut]
s_waitcnt lgkmcnt(0)

v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], v[vgprAmaxOutB]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 1
s_cmp_lt_u32 s[sgprTmp], 64
s_cbranch_scc1 label_permute_middle

/* inter_wave_reduction */
v_lshrrev_b32 v[vgprWidx], 6, v[vgprSerial]
s_mov_b32 s[sgprOffset], 1 // self.num_workitems // self.wave_size
label_wave_inter:  /// wave_inter
s_lshr_b32 s[sgprOffset], s[sgprOffset], 1
s_cmp_eq_u32 s[sgprOffset], 0
s_cbranch_scc1 label_wave_end
s_lshl_b32 s[sgprTmp], s[sgprOffset], 1
v_cmp_lt_u32 s[sgprTmp+2:sgprTmp+2+1], v[vgprWidx], s[sgprTmp]
v_cmp_ge_u32 s[sgprTmp+4:sgprTmp+4+1], v[vgprWidx], s[sgprOffset]
s_and_b64 vcc, s[sgprTmp+2:sgprTmp+2+1], s[sgprTmp+4:sgprTmp+4+1]
s_cbranch_vccnz label_wave_upper
v_cmp_lt_u32 vcc, v[vgprWidx], s[sgprOffset]
s_cbranch_vccnz label_wave_lower
s_branch label_wave_empty
label_wave_upper:  /// wave_upper
v_sub_u32 v[vgprTmp], v[vgprWidx], s[sgprOffset]
v_lshlrev_b32 v[vgprTmp], 2, v[vgprTmp]
ds_write_b32 v[vgprTmp], v[vgprAmaxOut] offset:2048
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_wave_inter
label_wave_lower:  /// wave_lower
s_barrier
v_lshlrev_b32 v[vgprTmp], 2, v[vgprWidx]
ds_read_b32 v[vgprAmaxOutB], v[vgprTmp] offset:2048
s_waitcnt lgkmcnt(0)
v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], v[vgprAmaxOutB]
s_branch label_wave_inter
label_wave_empty:  /// wave_empty
s_barrier
s_branch label_wave_inter
label_wave_end:  /// wave_end

/* broadcast */
v_cmp_eq_u32 vcc, v[vgprWidx], 0
s_cbranch_vccz label_broadcast_lower
ds_write_b32 v[vgprWidx], v[vgprAmaxOut] offset:2048
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_broadcast_end
label_broadcast_lower:  /// broadcast_lower
s_barrier
v_mov_b32 v[vgprTmp], 0
ds_read_b32 v[vgprAmaxOut], v[vgprTmp] offset:2048
s_waitcnt lgkmcnt(0)
label_broadcast_end:  /// broadcast_end


/* output_result */

v_readfirstlane_b32 s[sgprTmp], v[vgprSerial]
s_cmp_eq_u32 s[sgprTmp], 0
s_cbranch_scc0 label_end

s_cmp_eq_u32 s[sgprNumGroup], 1
s_cbranch_scc1 label_final_output
s_lshl_b32 s[sgprTmp+0], s[sgprNumGroup], 2
s_mov_b32 s[sgprDst+0], s[sgprAddressWk+0]
s_mov_b32 s[sgprDst+1], s[sgprAddressWk+1]
s_mov_b32 s[sgprDst+2], s[sgprTmp+0]
s_mov_b32 s[sgprDst+3], Srd127_96
s_lshl_b32 s[sgprOffset], s[sgprWGIdx], 2
v_mov_b32 v[vgprOffset], 0
buffer_store_dword v[vgprAmaxOut], v[vgprOffset], s[sgprDst:sgprDst+3], s[sgprOffset] offen offset:0, sc0 sc1
s_waitcnt vmcnt(0)

s_sub_i32 s[sgprTmp], s[sgprNumGroup], 1
s_atomic_dec s[sgprTmp], s[sgprAddressSy:sgprAddressSy+1],  glc
s_waitcnt 0
s_cmp_eq_u32 s[sgprTmp], 1
s_cbranch_scc0 label_end

s_lshl_b32 s[sgprTmp], s[sgprNumGroup], 2
s_mov_b32 s[sgprSrc+0], s[sgprAddressWk+0]
s_mov_b32 s[sgprSrc+1], s[sgprAddressWk+1]
s_mov_b32 s[sgprSrc+2], s[sgprTmp]
s_mov_b32 s[sgprSrc+3], Srd127_96

v_lshlrev_b32 v[vgprOffset], 2, v[vgprSerial]

v_mov_b32 v[vgprAmaxOut], 0

label_final_loop:  /// final_loop
buffer_load_dword v[vgprValue], v[vgprOffset], s[sgprSrc:sgprSrc+3], 0 offen offset:0, sc0 sc1
s_waitcnt vmcnt(0)

v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], abs(v[vgprValue+0])

s_mov_b32 s[sgprTmp], 256 // self.wave_size * self.i_type.numBytes()
v_add_u32 v[vgprOffset], v[vgprOffset], s[sgprTmp]

s_sub_i32 s[sgprNumGroup], s[sgprNumGroup], 64
s_cmp_gt_i32 s[sgprNumGroup], 0
s_cbranch_scc1 label_final_loop

/* intra_wave_reduction */
s_mov_b32 s[sgprTmp], 1
label_permute_final:  /// permute_final

v_add_u32 v[vgprTmp], s[sgprTmp], v[vgprSerial]
v_and_b32 v[vgprTmp], 63, v[vgprTmp]
v_lshlrev_b32 v[vgprTmp], 2, v[vgprTmp]

ds_bpermute_b32 v[vgprAmaxOutB], v[vgprTmp], v[vgprAmaxOut]
s_waitcnt lgkmcnt(0)

v_max_f32 v[vgprAmaxOut], v[vgprAmaxOut], v[vgprAmaxOutB]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 1
s_cmp_lt_u32 s[sgprTmp], 64
s_cbranch_scc1 label_permute_final


label_final_output:  /// final_output
s_mov_b32 s[sgprDst+0], s[sgprAMAXOut+0]
s_mov_b32 s[sgprDst+1], s[sgprAMAXOut+1]
s_mov_b32 s[sgprDst+2], 4
s_mov_b32 s[sgprDst+3], Srd127_96

v_mov_b32 v[vgprOffset], 0
buffer_store_dword v[vgprAmaxOut], v[vgprOffset], s[sgprDst:sgprDst+3], 0 offen offset:0


label_end:  /// end


label_KernelEnd:
s_endpgm                                           // Kernel End
label_ASM_End:  /// The end of the kernel
