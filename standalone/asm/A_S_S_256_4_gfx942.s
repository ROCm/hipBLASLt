.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected AMax_Ti_S_To_S_W_256_C_4
.globl AMax_Ti_S_To_S_W_256_C_4
.p2align 8
.type AMax_Ti_S_To_S_W_256_C_4,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel AMax_Ti_S_To_S_W_256_C_4
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 40 // accvgpr offset
  .amdhsa_next_free_vgpr 40 // vgprs
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
  - .offset: 16
    .size: 4
    .value_kind: by_value
  .group_segment_fixed_size: 32
  .kernarg_segment_align: 8
  .kernarg_segment_size: 24
  .max_flat_workgroup_size: 256
  .name: AMax_Ti_S_To_S_W_256_C_4
  .private_segment_fixed_size: 0
  .sgpr_count: 34
  .symbol: AMax_Ti_S_To_S_W_256_C_4.kd
  .vgpr_count: 40
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
.set vgprValue, 8
.set vgprTmp, 24

.set sgprKernelArg, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprAddressOut, 6
.set sgprAddressOutD, 8
.set sgprAddressIn, 10
.set sgprAddressScale, 12
.set sgprSizeLength, 5
.set sgprMainLoop, 14
.set sgprOffset, 15
.set sgprSrc, 16
.set sgprDst, 24
.set sgprTmp, 28

.set Srd127_96, 0x00020000


AMax_Ti_S_To_S_W_256_C_4:
/* Load kernel args */
s_load_dwordx2 s[sgprAddressOut:sgprAddressOut+1], s[sgprKernelArg:sgprKernelArg+1], 0
s_load_dwordx2 s[sgprAddressIn:sgprAddressIn+1], s[sgprKernelArg:sgprKernelArg+1], 8
s_load_dword s[sgprSizeLength], s[sgprKernelArg:sgprKernelArg+1], 16
s_waitcnt lgkmcnt(0)


/* init_param */
s_lshl_b32 s[sgprTmp], s[sgprSizeLength], 2

s_mov_b32 s[sgprDst+0], s[sgprAddressOut+0]
s_mov_b32 s[sgprDst+1], s[sgprAddressOut+1]
s_mov_b32 s[sgprDst+2], 4
s_mov_b32 s[sgprDst+3], Srd127_96

s_mov_b32 s[sgprSrc+0], s[sgprAddressIn+0]
s_mov_b32 s[sgprSrc+1], s[sgprAddressIn+1]
s_mov_b32 s[sgprSrc+2], s[sgprTmp]
s_mov_b32 s[sgprSrc+3], Srd127_96

v_mov_b32 v[vgprOutput], 0


/* calculate_global_address */
v_lshlrev_b32 v[vgprOffset+0], 0x4, v[vgprSerial]
s_mov_b32 s[sgprTmp], 4096
v_add_u32 v[vgprOffset+1], v[vgprOffset+0], s[sgprTmp]
v_add_u32 v[vgprOffset+2], v[vgprOffset+1], s[sgprTmp]
v_add_u32 v[vgprOffset+3], v[vgprOffset+2], s[sgprTmp]



/* sum_per_threadxN */
s_lshr_b32 s[sgprMainLoop], s[sgprSizeLength], 12
label_sum_per_threadxN:  /// loop sum_per_threadxN starts
s_cmp_eq_u32 s[sgprMainLoop], 0
s_cbranch_scc1 label_sum_per_threadxN_end

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
s_branch label_sum_per_threadxN
label_sum_per_threadxN_end:  /// loop sum_per_threadxN ends


/* sum_per_threadx4 */
s_lshr_b32 s[sgprMainLoop], s[sgprSizeLength], 10
s_and_b32 s[sgprMainLoop], 0x3, s[sgprMainLoop]
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


/* adjust_global_address */
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


/* intra_wave_reduction */
s_mov_b32 s[sgprTmp], 1
label_permute:  /// permute

v_add_u32 v[vgprTmp], s[sgprTmp], v[vgprSerial]
v_and_b32 v[vgprTmp], 63, v[vgprTmp]
v_lshlrev_b32 v[vgprTmp], 2, v[vgprTmp]

ds_bpermute_b32 v[vgprOutputB], v[vgprTmp], v[vgprOutput]
s_waitcnt lgkmcnt(0)

v_max_f32 v[vgprOutput], v[vgprOutput], v[vgprOutputB]
s_lshl_b32 s[sgprTmp], s[sgprTmp], 1
s_cmp_lt_u32 s[sgprTmp], 64
s_cbranch_scc1 label_permute

/* inter_wave_reduction */
v_lshrrev_b32 v[vgprWidx], 6, v[vgprSerial]
s_mov_b32 s[sgprOffset], 4
label_inter:  /// inter
s_lshr_b32 s[sgprOffset], s[sgprOffset], 1
s_cmp_eq_u32 s[sgprOffset], 0
s_cbranch_scc1 label_end
s_lshl_b32 s[sgprTmp], s[sgprOffset], 1
v_cmp_lt_u32 s[sgprTmp+2:sgprTmp+2+1], v[vgprWidx], s[sgprTmp]
v_cmp_ge_u32 s[sgprTmp+4:sgprTmp+4+1], v[vgprWidx], s[sgprOffset]
s_and_b64 vcc, s[sgprTmp+2:sgprTmp+2+1], s[sgprTmp+4:sgprTmp+4+1]
s_cbranch_vccnz label_upper
v_cmp_lt_u32 vcc, v[vgprWidx], s[sgprOffset]
s_cbranch_vccnz label_lower
s_branch label_empty
label_upper:  /// upper
v_sub_u32 v[vgprTmp], v[vgprWidx], s[sgprOffset]
v_lshlrev_b32 v[vgprTmp], 2, v[vgprTmp]
ds_write_b32 v[vgprTmp], v[vgprOutput] offset:0
s_waitcnt lgkmcnt(0)
s_barrier
s_branch label_inter
label_lower:  /// lower
s_barrier
v_lshlrev_b32 v[vgprTmp], 2, v[vgprWidx]
ds_read_b32 v[vgprOutputB], v[vgprTmp] offset:0
s_waitcnt lgkmcnt(0)
v_max_f32 v[vgprOutput], v[vgprOutput], v[vgprOutputB]
s_branch label_inter
label_empty:  /// empty
s_barrier
s_branch label_inter
label_end:  /// end

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
v_mov_b32 v[vgprOffset], 0
buffer_store_dword v[vgprOutput], v[vgprOffset], s[sgprDst:sgprDst+3], 0 offen offset:0

s_endpgm
.LAMax_Ti_S_To_S_W_256_C_4_end:
.size AMax_Ti_S_To_S_W_256_C_4, .LAMax_Ti_S_To_S_W_256_C_4_end - AMax_Ti_S_To_S_W_256_C_4
