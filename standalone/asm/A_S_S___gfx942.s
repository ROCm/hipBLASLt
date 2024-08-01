.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected AMax_Ti_S_To_S
.globl AMax_Ti_S_To_S
.p2align 8
.type AMax_Ti_S_To_S,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel AMax_Ti_S_To_S
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 64 // accvgpr offset
  .amdhsa_next_free_vgpr 64 // vgprs
  .amdhsa_next_free_sgpr 40 // sgprs
  .amdhsa_group_segment_fixed_size 32 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text

.amdgpu_metadata
---
amdhsa.kernels:
- .agpr_count: 0
  .args:
  - .address_space: global
    .offset: 0
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 8
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 16
    .size: 8
    .value_kind: global_buffer
  - .address_space: global
    .offset: 24
    .size: 8
    .value_kind: global_buffer
  - .offset: 32
    .size: 4
    .value_kind: by_value
  - .offset: 36
    .size: 4
    .value_kind: by_value
  - .offset: 40
    .size: 4
    .value_kind: by_value
  - .offset: 44
    .size: 4
    .value_kind: by_value
  - .offset: 48
    .size: 4
    .value_kind: by_value
  .group_segment_fixed_size: 32
  .kernarg_segment_align: 8
  .kernarg_segment_size: 56
  .max_flat_workgroup_size: 256
  .name: AMax_Ti_S_To_S
  .private_segment_fixed_size: 0
  .sgpr_count: 40
  .symbol: AMax_Ti_S_To_S.kd
  .vgpr_count: 64
  .wavefront_size: 64
amdhsa.version:
- 1
- 1

.end_amdgpu_metadata

.set vgprSerial, 0
.set vgprOutput, 1
.set vgprOutputB, 2
.set vgprWidx, 3
.set vgprOffset, 4
.set vgprValue, 12
.set vgprTmp, 44

.set sgprKernelArg, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprAddressOut, 6
.set sgprAddressIn, 8
.set sgprAddressWk, 10
.set sgprAddressSy, 12
.set sgprSizeLength, 5
.set sgprIsDivided, 14
.set sgprDivided, 15
.set sgprWorkSize, 16
.set sgprNumGroup, 17
.set sgprMainLoop, 18
.set sgprOffset, 19
.set sgprLogWorkSize, 20
.set sgprWGIdx, 21
.set sgprSrc, 24
.set sgprDst, 28
.set sgprTmp, 32

.set Srd127_96, 0x00020000


AMax_Ti_S_To_S:
/* Load kernel args */
s_load_dwordx2 s[sgprAddressOut:sgprAddressOut+1], s[sgprKernelArg:sgprKernelArg+1], 0
s_load_dwordx2 s[sgprAddressIn:sgprAddressIn+1], s[sgprKernelArg:sgprKernelArg+1], 8
s_load_dwordx2 s[sgprAddressWk:sgprAddressWk+1], s[sgprKernelArg:sgprKernelArg+1], 16
s_load_dwordx2 s[sgprAddressSy:sgprAddressSy+1], s[sgprKernelArg:sgprKernelArg+1], 24
s_load_dword s[sgprSizeLength], s[sgprKernelArg:sgprKernelArg+1], 32
s_load_dword s[sgprIsDivided], s[sgprKernelArg:sgprKernelArg+1], 36
s_load_dword s[sgprDivided], s[sgprKernelArg:sgprKernelArg+1], 40
s_load_dword s[sgprWorkSize], s[sgprKernelArg:sgprKernelArg+1], 44
s_load_dword s[sgprNumGroup], s[sgprKernelArg:sgprKernelArg+1], 48
s_waitcnt lgkmcnt(0)

v_mov_b32 v[vgprTmp], s[sgprWorkSize]
v_cvt_f32_u32 v[vgprTmp], v[vgprTmp]
s_nop 0
v_log_f32 v[vgprTmp], v[vgprTmp]
s_nop 0
v_cvt_u32_f32 v[vgprTmp], v[vgprTmp]
s_nop 0
v_readfirstlane_b32 s[sgprLogWorkSize], v[vgprTmp]


/* init_param */
s_lshl_b32 s[sgprTmp], s[sgprWorkSize], 2
s_mul_i32 s[sgprTmp], s[sgprWorkGroup0], s[sgprTmp]

s_add_u32 s[sgprSrc+0], s[sgprAddressIn+0], s[sgprTmp]
s_addc_u32 s[sgprSrc+1], s[sgprAddressIn+1], 0
s_lshl_b32 s[sgprSrc+2], s[sgprSizeLength], 2
s_sub_u32 s[sgprSrc+2], s[sgprSrc+2], s[sgprTmp]
s_mov_b32 s[sgprSrc+3], Srd127_96

v_mov_b32 v[vgprOutput], 0
s_mov_b32 s[sgprWGIdx], s[sgprWorkGroup0]


/* calculate_global_address */
v_lshrrev_b32 v[vgprWidx], 6, v[vgprSerial]
v_and_b32 v[vgprOffset+0], 63, v[vgprSerial]
v_lshlrev_b32 v[vgprOffset+0], 4, v[vgprOffset+0]
s_mov_b32 s[sgprTmp], 1024
v_add_u32 v[vgprOffset+1], v[vgprOffset+0], s[sgprTmp]
v_add_u32 v[vgprOffset+2], v[vgprOffset+1], s[sgprTmp]
v_add_u32 v[vgprOffset+3], v[vgprOffset+2], s[sgprTmp]
v_add_u32 v[vgprOffset+4], v[vgprOffset+3], s[sgprTmp]
v_add_u32 v[vgprOffset+5], v[vgprOffset+4], s[sgprTmp]
v_add_u32 v[vgprOffset+6], v[vgprOffset+5], s[sgprTmp]
v_add_u32 v[vgprOffset+7], v[vgprOffset+6], s[sgprTmp]

v_lshlrev_b32 v[vgprTmp], 13, v[vgprWidx]
v_add_u32 v[vgprOffset+0], v[vgprOffset+0], v[vgprTmp]
v_add_u32 v[vgprOffset+1], v[vgprOffset+1], v[vgprTmp]
v_add_u32 v[vgprOffset+2], v[vgprOffset+2], v[vgprTmp]
v_add_u32 v[vgprOffset+3], v[vgprOffset+3], v[vgprTmp]
v_add_u32 v[vgprOffset+4], v[vgprOffset+4], v[vgprTmp]
v_add_u32 v[vgprOffset+5], v[vgprOffset+5], v[vgprTmp]
v_add_u32 v[vgprOffset+6], v[vgprOffset+6], v[vgprTmp]
v_add_u32 v[vgprOffset+7], v[vgprOffset+7], v[vgprTmp]


/* sum_per_blocksize */
s_lshr_b32 s[sgprMainLoop], s[sgprSizeLength], s[sgprLogWorkSize]
label_sum_per_blocksize:  /// sum_per_blocksize
s_cmp_ge_i32 s[sgprWGIdx], s[sgprMainLoop]
s_cbranch_scc1 label_sum_per_blocksize_end

s_mov_b32 s[sgprOffset], 0
s_lshl_b32 s[sgprTmp], s[sgprWorkSize], 1
s_cmp_ge_i32 s[sgprOffset], s[sgprTmp]
s_cbranch_scc1 label_loop_end

buffer_load_dwordx4 v[vgprValue+0:vgprValue+0+3], v[vgprOffset+0], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
buffer_load_dwordx4 v[vgprValue+4:vgprValue+4+3], v[vgprOffset+1], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
buffer_load_dwordx4 v[vgprValue+8:vgprValue+8+3], v[vgprOffset+2], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
buffer_load_dwordx4 v[vgprValue+12:vgprValue+12+3], v[vgprOffset+3], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
buffer_load_dwordx4 v[vgprValue+16:vgprValue+16+3], v[vgprOffset+4], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
buffer_load_dwordx4 v[vgprValue+20:vgprValue+20+3], v[vgprOffset+5], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
buffer_load_dwordx4 v[vgprValue+24:vgprValue+24+3], v[vgprOffset+6], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
buffer_load_dwordx4 v[vgprValue+28:vgprValue+28+3], v[vgprOffset+7], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0

s_add_i32 s[sgprOffset], s[sgprOffset], 32768

label_loop:  /// loop

s_lshl_b32 s[sgprTmp], s[sgprWorkSize], 2
s_cmp_ge_i32 s[sgprOffset], s[sgprTmp]
s_cbranch_scc1 label_last_loop

/* block_max */
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+1])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+2])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+3])
buffer_load_dwordx4 v[vgprValue+0:vgprValue+0+3], v[vgprOffset+0], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+4])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+5])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+6])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+7])
buffer_load_dwordx4 v[vgprValue+4:vgprValue+4+3], v[vgprOffset+1], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+8])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+9])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+10])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+11])
buffer_load_dwordx4 v[vgprValue+8:vgprValue+8+3], v[vgprOffset+2], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+12])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+13])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+14])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+15])
buffer_load_dwordx4 v[vgprValue+12:vgprValue+12+3], v[vgprOffset+3], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+16])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+17])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+18])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+19])
buffer_load_dwordx4 v[vgprValue+16:vgprValue+16+3], v[vgprOffset+4], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+20])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+21])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+22])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+23])
buffer_load_dwordx4 v[vgprValue+20:vgprValue+20+3], v[vgprOffset+5], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+24])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+25])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+26])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+27])
buffer_load_dwordx4 v[vgprValue+24:vgprValue+24+3], v[vgprOffset+6], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+28])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+29])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+30])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+31])
buffer_load_dwordx4 v[vgprValue+28:vgprValue+28+3], v[vgprOffset+7], s[sgprSrc:sgprSrc+3], s[sgprOffset] offen offset:0

s_add_i32 s[sgprOffset], s[sgprOffset], 32768
s_branch label_loop

label_last_loop:  /// last_loop

/* block_max */
s_waitcnt vmcnt(7)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+1])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+2])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+3])
s_waitcnt vmcnt(6)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+4])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+5])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+6])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+7])
s_waitcnt vmcnt(5)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+8])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+9])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+10])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+11])
s_waitcnt vmcnt(4)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+12])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+13])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+14])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+15])
s_waitcnt vmcnt(3)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+16])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+17])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+18])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+19])
s_waitcnt vmcnt(2)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+20])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+21])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+22])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+23])
s_waitcnt vmcnt(1)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+24])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+25])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+26])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+27])
s_waitcnt vmcnt(0)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+28])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+29])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+30])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+31])


label_loop_end:  /// loop_end

s_lshl_b32 s[sgprTmp], s[sgprWorkSize], 2
s_mul_i32 s[sgprTmp], s[sgprTmp], s[sgprNumGroup]
v_add_u32 v[vgprOffset+0], v[vgprOffset+0], s[sgprTmp]
v_add_u32 v[vgprOffset+1], v[vgprOffset+1], s[sgprTmp]
v_add_u32 v[vgprOffset+2], v[vgprOffset+2], s[sgprTmp]
v_add_u32 v[vgprOffset+3], v[vgprOffset+3], s[sgprTmp]
v_add_u32 v[vgprOffset+4], v[vgprOffset+4], s[sgprTmp]
v_add_u32 v[vgprOffset+5], v[vgprOffset+5], s[sgprTmp]
v_add_u32 v[vgprOffset+6], v[vgprOffset+6], s[sgprTmp]
v_add_u32 v[vgprOffset+7], v[vgprOffset+7], s[sgprTmp]

s_add_i32 s[sgprWGIdx], s[sgprWGIdx], s[sgprNumGroup]
s_branch label_sum_per_blocksize

label_sum_per_blocksize_end:  /// sum_per_blocksize_end


/* adjust_global_address */
s_sub_u32 s[sgprTmp], s[sgprWorkSize], 1
s_and_b32 s[sgprTmp], s[sgprSizeLength], s[sgprTmp]
s_cmp_eq_u32 s[sgprTmp], 0
s_cbranch_scc1 label_reduction

s_lshr_b32 s[sgprTmp], s[sgprSizeLength], s[sgprLogWorkSize]
s_cmp_eq_u32 s[sgprWGIdx], s[sgprTmp]
s_cbranch_scc0 label_reduction

s_lshr_b32 s[sgprTmp], s[sgprSizeLength], s[sgprLogWorkSize]
s_add_u32 s[sgprTmp+1], s[sgprLogWorkSize], 2
s_lshl_b32 s[sgprTmp], s[sgprTmp], s[sgprTmp+1]

s_add_u32 s[sgprSrc+0], s[sgprAddressIn+0], s[sgprTmp]
s_addc_u32 s[sgprSrc+1], s[sgprAddressIn+1], 0
s_lshl_b32 s[sgprSrc+2], s[sgprSizeLength], 2
s_sub_u32 s[sgprSrc+2], s[sgprSrc+2], s[sgprTmp]
s_mov_b32 s[sgprSrc+3], Srd127_96

v_lshlrev_b32 v[vgprOffset], 4, v[vgprSerial]
s_mov_b32 s[sgprTmp], 4096
v_add_u32 v[vgprOffset+1], v[vgprOffset+0], s[sgprTmp]
v_add_u32 v[vgprOffset+2], v[vgprOffset+1], s[sgprTmp]
v_add_u32 v[vgprOffset+3], v[vgprOffset+2], s[sgprTmp]

s_sub_u32 s[sgprTmp], s[sgprWorkSize], 1
s_and_b32 s[sgprSizeLength], s[sgprSizeLength], s[sgprTmp]

/* sum_per_threadxdwordx4x4 */
s_lshr_b32 s[sgprMainLoop], s[sgprSizeLength], 12
label_sum_per_threadx4x4:  /// loop sum_per_threadx4x4 starts
s_cmp_eq_u32 s[sgprMainLoop], 0
s_cbranch_scc1 label_sum_per_threadx4x4_end

buffer_load_dwordx4 v[vgprValue+0:vgprValue+0+3], v[vgprOffset+0], s[sgprSrc:sgprSrc+3], 0 offen offset:0
buffer_load_dwordx4 v[vgprValue+4:vgprValue+4+3], v[vgprOffset+1], s[sgprSrc:sgprSrc+3], 0 offen offset:0
buffer_load_dwordx4 v[vgprValue+8:vgprValue+8+3], v[vgprOffset+2], s[sgprSrc:sgprSrc+3], 0 offen offset:0
buffer_load_dwordx4 v[vgprValue+12:vgprValue+12+3], v[vgprOffset+3], s[sgprSrc:sgprSrc+3], 0 offen offset:0

s_waitcnt vmcnt(3)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+1])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+2])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+3])
s_waitcnt vmcnt(2)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+4])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+5])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+6])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+7])
s_waitcnt vmcnt(1)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+8])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+9])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+10])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+11])
s_waitcnt vmcnt(0)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+12])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+13])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+14])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+15])

s_mov_b32 s[sgprTmp], 16384
v_add_u32 v[vgprOffset+0], v[vgprOffset+0], s[sgprTmp]
v_add_u32 v[vgprOffset+1], v[vgprOffset+1], s[sgprTmp]
v_add_u32 v[vgprOffset+2], v[vgprOffset+2], s[sgprTmp]
v_add_u32 v[vgprOffset+3], v[vgprOffset+3], s[sgprTmp]

s_sub_u32 s[sgprMainLoop], s[sgprMainLoop], 1
s_branch label_sum_per_threadx4x4
label_sum_per_threadx4x4_end:  /// loop sum_per_threadx4x4 ends


/* sum_per_threadx4 */
s_lshr_b32 s[sgprMainLoop], s[sgprSizeLength], 10
s_and_b32 s[sgprMainLoop], 3, s[sgprMainLoop]
label_sum_per_threadx4:  /// loop sum_per_threadx4 starts
s_cmp_eq_u32 s[sgprMainLoop], 0
s_cbranch_scc1 label_sum_per_threadx4_end

buffer_load_dwordx4 v[vgprValue:vgprValue+3], v[vgprOffset], s[sgprSrc:sgprSrc+3], 0 offen offset:0

s_waitcnt vmcnt(0)
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+1])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+2])
v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+3])

s_mov_b32 s[sgprTmp], 4096
v_add_u32 v[vgprOffset], v[vgprOffset], s[sgprTmp]
s_sub_u32 s[sgprMainLoop], s[sgprMainLoop], 1
s_branch label_sum_per_threadx4
label_sum_per_threadx4_end:  /// loop sum_per_threadx4 ends


/* adjust_global_address_2 */
v_mul_lo_u32 v[vgprTmp], 3, v[vgprSerial]
v_lshlrev_b32 v[vgprTmp], 2, v[vgprTmp]
v_sub_u32 v[vgprOffset], v[vgprOffset], v[vgprTmp]



/* sum_per_thread */
s_lshr_b32 s[sgprMainLoop], s[sgprSizeLength], 8
s_and_b32 s[sgprMainLoop], s[sgprMainLoop], 3
label_sum_per_thread:  /// loop sum_per_thread starts
s_cmp_eq_u32 s[sgprMainLoop], 0
s_cbranch_scc1 label_sum_per_thread_end

buffer_load_dword v[vgprValue], v[vgprOffset], s[sgprSrc:sgprSrc+3], 0 offen offset:0
s_waitcnt vmcnt(0)

v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])
s_mov_b32 s[sgprTmp], 1024
v_add_u32 v[vgprOffset], v[vgprOffset], s[sgprTmp]

s_sub_u32 s[sgprMainLoop], s[sgprMainLoop], 1
s_branch label_sum_per_thread
label_sum_per_thread_end:  /// loop sum_per_thread ends

/* sum_in_some_thread */
s_and_b32 s[sgprMainLoop], s[sgprSizeLength], 255
v_cmp_lt_u32 vcc, v[vgprSerial], s[sgprMainLoop]
s_cbranch_vccz label_sum
s_mov_b64 exec, vcc
s_nop 1
buffer_load_dword v[vgprValue], v[vgprOffset], s[sgprSrc:sgprSrc+3], 0 offen offset:0
s_waitcnt vmcnt(0)

v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])
s_mov_b64 exec, -1
s_nop 1
label_sum:  /// loop sum end


label_reduction:  /// reduction
/* intra_wave_reduction */
s_mov_b32 s[sgprTmp], 1
label_permute_middle:  /// permute_middle

v_add_u32 v[vgprTmp], s[sgprTmp], v[vgprSerial]
v_and_b32 v[vgprTmp], 63, v[vgprTmp]
v_lshlrev_b32 v[vgprTmp], 2, v[vgprTmp]

ds_bpermute_b32 v[vgprOutputB], v[vgprTmp], v[vgprOutput]
s_waitcnt lgkmcnt(0)

v_max_f32 v[vgprOutput], v[vgprOutput], v[vgprOutputB]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 1
s_cmp_lt_u32 s[sgprTmp], 64
s_cbranch_scc1 label_permute_middle

/* inter_wave_reduction */
v_lshrrev_b32 v[vgprWidx], 6, v[vgprSerial]
s_mov_b32 s[sgprOffset], 4
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
ds_write_b32 v[vgprTmp], v[vgprOutput] offset:0
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_wave_inter
label_wave_lower:  /// wave_lower
s_barrier
v_lshlrev_b32 v[vgprTmp], 2, v[vgprWidx]
ds_read_b32 v[vgprOutputB], v[vgprTmp] offset:0
s_waitcnt lgkmcnt(0)
v_max_f32 v[vgprOutput], v[vgprOutput], v[vgprOutputB]
s_branch label_wave_inter
label_wave_empty:  /// wave_empty
s_barrier
s_branch label_wave_inter
label_wave_end:  /// wave_end

/* broadcast */
v_cmp_eq_u32 vcc, v[vgprWidx], 0
s_cbranch_vccz label_broadcast_lower
ds_write_b32 v[vgprWidx], v[vgprOutput] offset:0
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_broadcast_end
label_broadcast_lower:  /// broadcast_lower
s_barrier
v_mov_b32 v[vgprTmp], 0
ds_read_b32 v[vgprOutput], v[vgprTmp] offset:0
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
s_lshl_b32 s[sgprOffset], s[sgprWorkGroup0], 2
v_mov_b32 v[vgprOffset], 0
buffer_store_dword v[vgprOutput], v[vgprOffset], s[sgprDst:sgprDst+3], s[sgprOffset] offen offset:0, sc0 sc1
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

v_mov_b32 v[vgprOutput], 0

label_final_loop:  /// final_loop
buffer_load_dword v[vgprValue], v[vgprOffset], s[sgprSrc:sgprSrc+3], 0 offen offset:0, sc0 sc1
s_waitcnt vmcnt(0)

v_max_f32 v[vgprOutput], v[vgprOutput], abs(v[vgprValue+0])

s_mov_b32 s[sgprTmp], 256
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

ds_bpermute_b32 v[vgprOutputB], v[vgprTmp], v[vgprOutput]
s_waitcnt lgkmcnt(0)

v_max_f32 v[vgprOutput], v[vgprOutput], v[vgprOutputB]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 1
s_cmp_lt_u32 s[sgprTmp], 64
s_cbranch_scc1 label_permute_final


label_final_output:  /// final_output
s_mov_b32 s[sgprDst+0], s[sgprAddressOut+0]
s_mov_b32 s[sgprDst+1], s[sgprAddressOut+1]
s_mov_b32 s[sgprDst+2], 4
s_mov_b32 s[sgprDst+3], Srd127_96

v_mov_b32 v[vgprOffset], 0
s_nop 1
s_cmp_eq_i32 s[sgprIsDivided], 0
s_cbranch_scc1 label_divided
v_rcp_f32 v[vgprOutput], v[vgprOutput]
s_nop 1
v_mul_f32 v[vgprOutput], v[vgprOutput], s[sgprDivided]
label_divided:  /// divided
buffer_store_dword v[vgprOutput], v[vgprOffset], s[sgprDst:sgprDst+3], 0 offen offset:0


label_end:  /// end


s_endpgm
.LAMax_Ti_S_To_S_end:
.size AMax_Ti_S_To_S, .LAMax_Ti_S_To_S_end - AMax_Ti_S_To_S
